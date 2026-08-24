"""Distill a state-based teacher into a depth-vision student (DAgger).

Phase 6 of the port. Modelled on ``train.py``, but the pipeline diverges after
``gym.make``: there is no ``RlGamesVecEnvWrapper`` and no ``Runner``.

Pipeline::

    argparse (AppLauncher flags; --enable_cameras is FORCED on)
        v
    AppLauncher (boots Kit; must precede any isaaclab.* import)
        v
    @hydra_task_config_with_yaml   (configclass <- task YAML <- Hydra CLI)
        v
    student_obs.enabled = True     (spawns the TiledCamera; ~4.1x step cost)
        v
    gym.make(task_id, cfg=env_cfg) -> env.unwrapped is the DirectRLEnv
        v
    Teacher (rl_games PpoPlayerContinuous, frozen)   +   Dagger (our loop)

Two deliberate absences, both established earlier in the port:

* **No rl_games env wrapper.** ``RlGamesVecEnvWrapper`` raises unless the env
  exposes a ``"policy"`` obs key (``isaaclab_rl/rl_games.py:128-130``), and the
  student contract emits ``proprio``/``img`` instead. The teacher's ``env_info``
  is built from dims directly. DEXTRAH's ``Dagger`` reads the raw gym env too.
* **No ``Runner``.** rl_games is used only to *build* the student model; the
  training loop is ``isaacsimenvs/distillation/dagger.py``.

Note ``--num-envs`` IS the batch size, not a throughput knob. Measured on an
RTX 5090 32 GB with the camera on: 256 envs ~17.4 Hz, 512 ~10.8 Hz, and 1024
OOMs the machine. Start at 256.

Examples::

    # Smoke test: does it run at all?
    python isaacsimenvs/distill.py --headless --num-envs 8 --iters 200

    # The real thing
    python isaacsimenvs/distill.py --headless --num-envs 256 \\
        --teacher-checkpoint pretrained_assembly/tight_insertion/model.pth \\
        --out-dir runs/distill_tight_insertion

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Same precedent as evaluation/eval_isaacsim.py and eval_offline.py. Without it
# Kit stops on an interactive EULA prompt and dies with "EOF when reading a
# line", which reads like a launcher bug rather than a missing env var.
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

def _git_provenance() -> dict:
    """Pin a run to the code that produced it.

    A result nobody can trace back to a commit is not a result. Records the
    commit, branch and dirty flag, and when the tree is dirty stashes the full
    diff, since "commit abc123 + these 40 lines" is the only honest description
    of a run made from an uncommitted working tree.
    """
    import subprocess

    def _run(*cmd):
        try:
            return subprocess.run(
                cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    porcelain = _run("git", "status", "--porcelain")
    meta = {
        "commit": _run("git", "rev-parse", "HEAD"),
        "commit_short": _run("git", "rev-parse", "--short", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(porcelain),
        "dirty_files": [ln[3:] for ln in porcelain.splitlines()],
        "remote": _run("git", "config", "--get", "remote.origin.url"),
    }
    if meta["dirty"]:
        meta["diff"] = _run("git", "diff", "HEAD")
    return meta


DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
DEFAULT_TEACHER_AGENT = "rl_games_sapg_cfg_entry_point"
STUDENT_ENTRY_POINT = "rl_games_student_cfg_entry_point"
DEFAULT_CHECKPOINT = "pretrained_assembly/tight_insertion/model.pth"


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="DAgger distillation: state teacher -> depth student.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--teacher-agent", default=DEFAULT_TEACHER_AGENT)
    parser.add_argument(
        "--teacher-checkpoint", default=DEFAULT_CHECKPOINT,
        help="Frozen teacher weights. Run `python download_checkpoints.py` first.",
    )
    parser.add_argument(
        "--teacher-block-id", default=None,
        help="SAPG block-id column appended to the teacher's obs. 50.0 (default) "
             "measured 96.9%%, 0.0 measured 95.4%%, 'ramp' reproduces rl_games' "
             "per-env linspace. See isaacsimenvs/distillation/teacher.py.",
    )
    parser.add_argument("--problem", default="tight_insertion")
    parser.add_argument(
        "--num-envs", type=int, default=256,
        help="ALSO the batch size. 256 ~17.4 Hz, 512 ~10.8 Hz, 1024 OOMs 32 GB.",
    )
    parser.add_argument(
        "--iters", type=int, default=None,
        help="Env steps (NOT gradient steps: with seq_length=16 you get "
             "iters/16 gradient steps). Default: the student YAML's max_iters.",
    )
    # --- student overrides: the knobs most likely to need changing ---
    parser.add_argument("--seq-length", type=int, default=None, help="BPTT length.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--aux-coeff", type=float, default=None)
    parser.add_argument("--sigma-loss-coef", type=float, default=None)
    parser.add_argument("--mu-weight-mode", choices=("uniform", "inv_sigma2"), default=None)
    parser.add_argument(
        "--loss-form", choices=("l2_norm", "mse"), default=None,
        help="l2_norm (default) is DEXTRAH's: norm over the action dim, mean "
             "over the batch. mse was this port's first attempt and stalls as "
             "the error shrinks, since its gradient decays with the error.",
    )
    parser.add_argument(
        "--beta-warmup-grad-steps", type=int, default=None,
        help="Teacher-driven warmup length in GRADIENT steps (not env steps). "
             "At seq_length=16, 15000 grad steps = 240k env steps.",
    )
    parser.add_argument("--beta-anneal-grad-steps", type=int, default=None)
    parser.add_argument(
        "--no-depth-aug", action="store_true",
        help="Disable depth augmentation. On by default to match DEXTRAH, which "
             "augments every step; play2perfect's port of the same five stages "
             "ships disabled.",
    )
    parser.add_argument(
        "--drop-bad-inits", action="store_true",
        help="Terminate episodes where the peg has been dropped on the table. "
             "OFF by default, matching DEXTRAH (and this env's own default). "
             "~14%% of tight_insertion first episodes start unstable (measured "
             "143/1024); eval_offline discards those trials but DAgger visits "
             "them, and the teacher's labels on an already-dropped peg are "
             "worthless.",
    )
    parser.add_argument(
        "--port-deviations", action="store_true",
        help="Switch from DEXTRAH's behaviour (the default) to this port's "
             "alternatives, as a group: seq_length=16, beta warmup 15000 grad "
             "steps, mu_weight_mode=uniform, sigma_loss_coef=0.0, teacher "
             "block_id=50.0, and terminate dropped-peg episodes. Each also has "
             "its own flag; explicit flags win over this preset. See "
             "isaacsimenvs/distillation/dagger.py for the measurement behind "
             "each one.",
    )
    # --- logging ---
    # Flag names mirror train.py's wandb block. Tensorboard is written whenever
    # --out-dir is set, matching DEXTRAH (which writes it unconditionally).
    parser.add_argument("--wandb-activate", action="store_true")
    parser.add_argument("--wandb-project", default="foundation-touch")
    parser.add_argument(
        "--wandb-entity", default=None,
        help="Defaults to wandb's configured default entity, which on a team "
             "account is the TEAM (ai2-robotics here), not you. Pass your "
             "username for a personal project, or export WANDB_ENTITY to make "
             "it stick without hardcoding an account in the repo.",
    )
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None,
                        help="Defaults to <problem>-<num_envs>envs.")
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    parser.add_argument("--wandb-notes", default="")
    # --- run management ---
    parser.add_argument("--out-dir", default=None, help="Checkpoint + log dir.")
    parser.add_argument("--save-every", type=int, default=5000, help="Env steps.")
    parser.add_argument("--log-every", type=int, default=100, help="Env steps.")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # student_obs spawns a TiledCamera, which is unusable without this. Forced
    # rather than validated: forgetting it is the single easiest way to get a
    # confusing failure deep inside scene setup.
    args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    # Safe to import isaaclab-backed modules now.
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401  triggers gym.register side effects
    from isaacsimenvs.distillation import Dagger, Teacher
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml

    @hydra_task_config_with_yaml(args_cli.task, args_cli.teacher_agent)
    def run(env_cfg, teacher_agent_cfg: dict) -> None:
        env_cfg.sim.device = args_cli.sim_device
        env_cfg.scene.num_envs = int(args_cli.num_envs)
        env_cfg.precise_assembly.problem = args_cli.problem

        # The full student contract: proprio / img / teacher_obs / critic /
        # aux_info. (emit_in_observations False is the viser-eval mode, which
        # keeps {policy, critic} so the rl_games player still works.)
        env_cfg.student_obs.enabled = True
        env_cfg.student_obs.emit_in_observations = True

        # DEXTRAH augments the depth image every step -- correlated noise,
        # normal noise, pixel dropout + random-uniform replacement, and stick
        # artifacts (distillation.py:387-390, augment_depth at :634).
        # play2perfect already carries a 1:1 port of those same five stages
        # (scene_utils.py:615-625), applied inside get_student_obs(), but ships
        # it off. On to match DEXTRAH.
        env_cfg.student_obs.use_depth_aug = not args_cli.no_depth_aug

        # Plan risk #2. tight_insertion discards ~14% of first episodes as
        # unstable inits (measured 143/1024 in the Phase 2 gate): the peg falls
        # or is ungraspable at reset. eval_offline drops those trials, but DAgger
        # *visits* them, and the teacher's labels on an already-dropped peg are
        # worthless -- during the beta=1 warmup that is 14% of teacher-driven
        # data teaching nonsense. Terminating early ends them in a few steps.
        # DEXTRAH does not terminate dropped-peg episodes; neither does this
        # env by default. See --drop-bad-inits for why you might want to.
        env_cfg.precise_assembly.enable_dropped_on_table_term = bool(
            args_cli.drop_bad_inits or args_cli.port_deviations
        )

        # hole_pos is the primary aux target and random-goal envs replace it
        # with a (0, 0, -1) sentinel; the env raises on the combination, so fail
        # here with a message that names the flag instead.
        if float(env_cfg.precise_assembly.random_goal_fraction) > 0.0:
            raise SystemExit(
                "distillation requires precise_assembly.random_goal_fraction == 0.0 "
                f"(got {env_cfg.precise_assembly.random_goal_fraction}); random-goal "
                "envs have no valid hole_pos label."
            )

        student_cfg = load_cfg_from_registry(
            args_cli.task.split(":")[-1], STUDENT_ENTRY_POINT
        )
        scfg = student_cfg["params"]["config"]
        scfg["num_actors"] = int(args_cli.num_envs)
        scfg["device"] = scfg["device_name"] = args_cli.rl_device

        # Preset first, so explicit flags below still override it. The YAML
        # itself carries DEXTRAH's values, so with no flags this runs DEXTRAH's
        # configuration.
        if args_cli.port_deviations:
            scfg["mu_weight_mode"] = "uniform"
            scfg["sigma_loss_coef"] = 0.0
            scfg["seq_length"] = 16
            scfg["beta_warmup_grad_steps"] = 15_000
            scfg["beta_anneal_grad_steps"] = 2_000
            print(
                "[distill] --port-deviations: seq_length=16, warmup=15000 grad "
                "steps, uniform mu weights, sigma_loss_coef=0.0, block_id=50.0, "
                "dropped-peg episodes terminated",
                flush=True,
            )
        for flag, key in [
            ("seq_length", "seq_length"),
            ("lr", "learning_rate"),
            ("aux_coeff", "aux_coeff"),
            ("sigma_loss_coef", "sigma_loss_coef"),
            ("mu_weight_mode", "mu_weight_mode"),
            ("loss_form", "loss_form"),
            ("beta_warmup_grad_steps", "beta_warmup_grad_steps"),
            ("beta_anneal_grad_steps", "beta_anneal_grad_steps"),
        ]:
            val = getattr(args_cli, flag)
            if val is not None:
                scfg[key] = val
        if args_cli.seed is not None:
            student_cfg["params"]["seed"] = int(args_cli.seed)
            torch.manual_seed(int(args_cli.seed))

        ckpt = Path(args_cli.teacher_checkpoint)
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        if not ckpt.is_file():
            raise SystemExit(
                f"teacher checkpoint not found: {ckpt}\n"
                "Run `python download_checkpoints.py` first."
            )

        git = _git_provenance()

        out_dir = Path(args_cli.out_dir) if args_cli.out_dir else None
        if out_dir is not None:
            if not out_dir.is_absolute():
                out_dir = REPO_ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_meta.json").write_text(json.dumps({
                "git": {k: v for k, v in git.items() if k != "diff"},
                "argv": sys.argv,
                "num_envs": int(args_cli.num_envs),
                "problem": args_cli.problem,
                "student_cfg": student_cfg["params"]["config"],
                "network": student_cfg["params"]["network"],
            }, indent=2, default=str))
            if git.get("diff"):
                (out_dir / "uncommitted.diff").write_text(git["diff"])

        env = gym.make(args_cli.task, cfg=env_cfg)
        uenv = env.unwrapped  # Dagger reads _get_observations() directly

        # Default "ramp" is what rl_games' BasePlayer actually appends
        # (player.py:93) and what eval_offline measured 96.9% with. The constant
        # 50.0 measured 96.9% too and is label-consistent across envs, which is
        # why --port-deviations selects it.
        block_id = args_cli.teacher_block_id
        if block_id is None:
            block_id = "50.0" if args_cli.port_deviations else "ramp"
        block_id = block_id if block_id == "ramp" else float(block_id)
        teacher = Teacher(
            teacher_agent_cfg, str(ckpt),
            num_envs=uenv.num_envs,
            teacher_obs_dim=int(uenv.cfg.observation_space),
            critic_dim=int(uenv.cfg.state_space),
            action_dim=int(uenv.cfg.action_space),
            device=args_cli.rl_device,
            block_id=block_id,
        )
        if args_cli.wandb_activate:
            import wandb

            wandb.init(
                project=args_cli.wandb_project,
                entity=args_cli.wandb_entity,
                group=args_cli.wandb_group,
                name=args_cli.wandb_name
                or f"{args_cli.problem}-{args_cli.num_envs}envs",
                tags=list(args_cli.wandb_tags),
                notes=args_cli.wandb_notes,
                config={
                    "git_commit": git["commit_short"],
                    "git_branch": git["branch"],
                    "git_dirty": git["dirty"],
                    "problem": args_cli.problem,
                    "num_envs": int(args_cli.num_envs),
                    "teacher_block_id": str(block_id),
                    "port_deviations": bool(args_cli.port_deviations),
                    "drop_bad_inits": bool(
                        env_cfg.precise_assembly.enable_dropped_on_table_term
                    ),
                    **{k: student_cfg["params"]["config"][k] for k in (
                        "seq_length", "learning_rate", "aux_coeff",
                        "mu_weight_mode", "sigma_loss_coef",
                        "beta_warmup_grad_steps", "beta_anneal_grad_steps",
                    )},
                },
            )

        dagger = Dagger(
            uenv, student_cfg, teacher,
            run_meta={"git": {k: v for k, v in git.items() if k != "diff"},
                      "argv": sys.argv},
            device=args_cli.rl_device, log_every=int(args_cli.log_every),
            log_dir=str(out_dir / "summaries") if out_dir is not None else None,
            use_wandb=bool(args_cli.wandb_activate),
        )

        max_iters = int(args_cli.iters) if args_cli.iters is not None else dagger.max_iters
        print(
            f"\n[distill] commit={git['commit_short']} branch={git['branch']}"
            f"{' +DIRTY(' + str(len(git['dirty_files'])) + ' files)' if git['dirty'] else ''}\n"
            f"[distill] problem={args_cli.problem} envs={uenv.num_envs} "
            f"seq_length={dagger.seq_length} -> {max_iters // max(1, dagger.seq_length)} "
            f"gradient steps over {max_iters} env steps\n"
            f"[distill] teacher block_id={block_id} ckpt={ckpt.name}\n"
            f"[distill] warmup={dagger.beta_warmup_grad_steps} grad steps "
            f"(= {dagger.beta_warmup_grad_steps * dagger.seq_length} env steps), "
            f"anneal={dagger.beta_anneal_grad_steps}\n"
            f"[distill] drop_on_table_term="
            f"{env_cfg.precise_assembly.enable_dropped_on_table_term} "
            f"depth_aug={env_cfg.student_obs.use_depth_aug}\n"
            f"[distill] aux_coeff={dagger.aux_coeff} "
            f"sigma_loss_coef={dagger.sigma_loss_coef} "
            f"mu_weight={dagger.mu_weight_mode} loss_form={dagger.loss_form}\n"
            f"[distill] out_dir={out_dir}  "
            f"tensorboard={'on' if out_dir else 'off (no --out-dir)'}  "
            f"wandb={'on' if args_cli.wandb_activate else 'off'}\n",
            flush=True,
        )

        try:
            if out_dir is None or args_cli.save_every <= 0:
                history = dagger.distill(max_iters=max_iters)
            else:
                # Chunked so checkpoints land without threading save logic
                # through the loop. dagger.iter persists across calls.
                history = []
                while dagger.iter < max_iters:
                    stop = min(dagger.iter + int(args_cli.save_every), max_iters)
                    history = dagger.distill(max_iters=stop)
                    dagger.save(str(out_dir / f"student_{dagger.iter:08d}.pth"))
                    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
                    print(f"[distill] saved at iter {dagger.iter}", flush=True)
        finally:
            if out_dir is not None:
                dagger.save(str(out_dir / "student_final.pth"))
                (out_dir / "history.json").write_text(json.dumps(dagger.history, indent=2))
            dagger.close()
            env.close()

        if dagger.history:
            last = dagger.history[-1]
            print(
                f"\n[distill] done. iter={last['iter']} grad_steps={last['grad_steps']} "
                f"total={last['total']:.4f} mu={last['mu']:.4f} "
                f"hole_rmse={last.get('hole_rmse_mm', float('nan')):.1f}mm"
            )
            print(
                "[distill] hole_rmse_mm is the number that matters: the teacher "
                "tolerates 2 mm of goal error, so that is the student's spec.\n"
                f"[distill] ignore-the-image baseline is "
                f"{dagger.hole_rmse_baseline_mm:.1f} mm (always predict the "
                f"centre of the hole range). ~1.00x base means the encoder "
                f"learned the mean, not the hole."
            )

    run()
    app.close()


if __name__ == "__main__":
    main()
