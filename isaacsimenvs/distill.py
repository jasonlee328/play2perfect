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

    # Suspect the encoder cannot localize? The one knob most likely to matter.
    python isaacsimenvs/distill.py --headless --num-envs 256 --spatial-pool flatten
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
        "--teacher-block-id", default="50.0",
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
        "--beta-warmup-grad-steps", type=int, default=None,
        help="Teacher-driven warmup length in GRADIENT steps (not env steps). "
             "At seq_length=16, 15000 grad steps = 240k env steps.",
    )
    parser.add_argument("--beta-anneal-grad-steps", type=int, default=None)
    parser.add_argument(
        "--keep-bad-inits", action="store_true",
        help="Do NOT terminate episodes where the peg has been dropped on the "
             "table. Off by default: ~14%% of tight_insertion first episodes "
             "start unstable (measured 143/1024), eval_offline discards them, "
             "but DAgger visits them and the teacher's labels there are "
             "worthless. Terminating early stops the student training on a "
             "teacher flailing at an object that already fell.",
    )
    parser.add_argument(
        "--spatial-pool", choices=("avgpool", "flatten"), default=None,
        help="Encoder pooling. avgpool (DEXTRAH default) discards the 3x8 "
             "spatial map; flatten keeps it. Try flatten if hole_pos RMSE "
             "stalls well above 2 mm while the mu loss converges.",
    )
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

        # Plan risk #2. tight_insertion discards ~14% of first episodes as
        # unstable inits (measured 143/1024 in the Phase 2 gate): the peg falls
        # or is ungraspable at reset. eval_offline drops those trials, but DAgger
        # *visits* them, and the teacher's labels on an already-dropped peg are
        # worthless -- during the beta=1 warmup that is 14% of teacher-driven
        # data teaching nonsense. Terminating early ends them in a few steps.
        env_cfg.precise_assembly.enable_dropped_on_table_term = not args_cli.keep_bad_inits

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
        for flag, key in [
            ("seq_length", "seq_length"),
            ("lr", "learning_rate"),
            ("aux_coeff", "aux_coeff"),
            ("sigma_loss_coef", "sigma_loss_coef"),
            ("mu_weight_mode", "mu_weight_mode"),
            ("beta_warmup_grad_steps", "beta_warmup_grad_steps"),
            ("beta_anneal_grad_steps", "beta_anneal_grad_steps"),
        ]:
            val = getattr(args_cli, flag)
            if val is not None:
                scfg[key] = val
        if args_cli.spatial_pool is not None:
            student_cfg["params"]["network"]["student_image"]["spatial_pool"] = (
                args_cli.spatial_pool
            )
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

        out_dir = Path(args_cli.out_dir) if args_cli.out_dir else None
        if out_dir is not None:
            if not out_dir.is_absolute():
                out_dir = REPO_ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)

        env = gym.make(args_cli.task, cfg=env_cfg)
        uenv = env.unwrapped  # Dagger reads _get_observations() directly

        block_id = args_cli.teacher_block_id
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
        dagger = Dagger(
            uenv, student_cfg, teacher,
            device=args_cli.rl_device, log_every=int(args_cli.log_every),
        )

        max_iters = int(args_cli.iters) if args_cli.iters is not None else dagger.max_iters
        print(
            f"\n[distill] problem={args_cli.problem} envs={uenv.num_envs} "
            f"seq_length={dagger.seq_length} -> {max_iters // max(1, dagger.seq_length)} "
            f"gradient steps over {max_iters} env steps\n"
            f"[distill] teacher block_id={block_id} ckpt={ckpt.name}\n"
            f"[distill] warmup={dagger.beta_warmup_grad_steps} grad steps "
            f"(= {dagger.beta_warmup_grad_steps * dagger.seq_length} env steps), "
            f"anneal={dagger.beta_anneal_grad_steps}\n"
            f"[distill] drop_on_table_term="
            f"{env_cfg.precise_assembly.enable_dropped_on_table_term}\n"
            f"[distill] spatial_pool="
            f"{student_cfg['params']['network']['student_image']['spatial_pool']} "
            f"aux_coeff={dagger.aux_coeff} sigma_loss_coef={dagger.sigma_loss_coef} "
            f"mu_weight={dagger.mu_weight_mode}\n"
            f"[distill] out_dir={out_dir}\n",
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
                "tolerates 2 mm of goal error, so that is the student's spec."
            )

    run()
    app.close()


if __name__ == "__main__":
    main()
