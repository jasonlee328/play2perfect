"""Success-rate evaluation of a distilled depth-vision student.

`eval_offline.py` scores rl_games teacher checkpoints through a `Runner`/player.
The student is not an rl_games agent -- it is built by `ModelBuilder` but driven
by our own loop -- so it needs its own harness. The *scoring* is deliberately
identical, so the number is directly comparable to the teacher's 96.9%:

  * all envs run in parallel and only each env's FIRST episode is scored, so N
    envs are N i.i.d. trials;
  * episodes ending within --min-episode-steps are discarded as unstable inits
    (peg fell or was ungraspable at reset -- the task was impossible, not the
    policy's fault);
  * success = the first episode reached every subgoal.

Also reports `hole_pos` RMSE against the ignore-the-image baseline, since the
aux heads are still attached and that is the diagnostic for *why* a success rate
is what it is: a student that cannot localise the hole cannot insert into it.

    python evaluation/eval_student.py --checkpoint runs/distill_v2/student_final.pth
    python evaluation/eval_student.py --checkpoint ... --depth-aug   # realistic
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
STUDENT_ENTRY_POINT = "rl_games_student_cfg_entry_point"
# eval_offline, tight_insertion, 1024 envs.
TEACHER_REFERENCE = {"success": 0.969, "goal_ratio": 0.971, "retract": 0.969}


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--problem", default="tight_insertion")
    parser.add_argument("--num-envs", type=int, default=256,
                        help="= number of i.i.d. trials. 1024 OOMs with the camera on.")
    parser.add_argument("--steps", type=int, default=2500,
                        help="Max control steps; ends early once every env finishes ep 1.")
    parser.add_argument("--min-episode-steps", type=int, default=100,
                        help="Discard first episodes shorter than this as unstable inits.")
    parser.add_argument("--depth-aug", action="store_true",
                        help="Evaluate WITH depth augmentation. Off by default so the "
                             "number is a clean-sensor upper bound, matching how "
                             "eval_offline scores the teacher (keep_dr=False).")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--sim-device", default="cuda:0")
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.enable_cameras = True          # the student needs its camera
    sys.argv = [sys.argv[0]]

    app = AppLauncher(args).app

    import gymnasium as gym  # noqa: F401
    import torch
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from rl_games.algos_torch.model_builder import ModelBuilder

    import isaacsimenvs  # noqa: F401
    from evaluation.eval_isaacsim import (
        _apply_env_overrides,
        _instantiate_env,
        _load_env_cfg,
    )
    from rl_games.algos_torch import model_builder

    from isaacsimenvs.distillation.a2c_aux_cnn import A2CBuilder

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = REPO_ROOT / ckpt_path
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Prefer the config the checkpoint was trained with; fall back to the
    # registry for older checkpoints that predate embedding it.
    student_cfg = ck.get("agent_cfg")
    if student_cfg is None:
        print("[eval] checkpoint has no embedded agent_cfg; using the registry "
              "default.")
        student_cfg = load_cfg_from_registry(TASK, STUDENT_ENTRY_POINT)

    env_cfg = _load_env_cfg(TASK)
    _apply_env_overrides(
        env_cfg, problem=args.problem, goal_mode="preInsertAndFinal",
        random_goal_fraction=0.0, insertion_success_tolerance=0.01,
        retract_success_tolerance=0.005, num_envs=args.num_envs,
        sim_device=args.sim_device, sdf=False, keep_dr=False, extra_overrides={},
    )
    env_cfg.student_obs.enabled = True
    env_cfg.student_obs.emit_in_observations = True
    env_cfg.student_obs.use_depth_aug = bool(args.depth_aug)

    env = _instantiate_env(TASK, env_cfg)
    N, dev = env.num_envs, env.device

    # Reset FIRST. Reading observations from a never-reset env triggers a camera
    # render before the sim has been reset, and the subsequent env.reset() then
    # deadlocks with the GPU idle -- which is what made the first version of this
    # script hang indefinitely before taking a single step.
    print("[eval] env.reset() ...", flush=True)
    obs, _ = env.reset()
    print(f"[eval] reset OK; obs keys = {sorted(obs)}", flush=True)

    model_builder.register_network("a2c_aux_cnn_net", A2CBuilder)
    model = (
        ModelBuilder().load(student_cfg["params"]).build({
            "actions_num": int(env.cfg.action_space),
            "input_shape": (int(obs["proprio"].shape[-1]),),
            "num_seqs": N,
            "value_size": 1,
            "normalize_value": bool(student_cfg["params"]["config"].get("normalize_value", True)),
            "normalize_input": bool(student_cfg["params"]["config"].get("normalize_input", True)),
        }).to(dev)
    )
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[eval] state_dict missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    # eval() freezes both RunningMeanStd normalizers. Leaving train() on would
    # let the eval's own observations shift the statistics mid-measurement.
    model.eval()
    net = model.a2c_network
    print(f"\n[eval] {ckpt_path.name}  iter={ck.get('iter')} grad_steps={ck.get('grad_steps')}", flush=True)
    print(f"[eval] envs={N} depth_aug={args.depth_aug} problem={args.problem}", flush=True)

    # ignore-the-image baseline for hole_pos, same formula the trainer logs
    pih = env.cfg.precise_assembly
    xr = float(pih.hole_x_range[1]) - float(pih.hole_x_range[0])
    yr = float(pih.hole_y_range[1]) - float(pih.hole_y_range[0])
    baseline_mm = ((xr ** 2 + yr ** 2) / 12.0) ** 0.5 * 1000.0

    states = [s.to(dev) for s in net.get_default_rnn_state()]
    prev_actions = torch.zeros(N, int(env.cfg.action_space), device=dev)
    print("[eval] states + prev_actions ready", flush=True)

    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    first_len = torch.zeros(N, dtype=torch.long, device=dev)
    first_full = torch.zeros(N, dtype=torch.bool, device=dev)
    first_ratio = torch.zeros(N, dtype=torch.float, device=dev)
    first_retract = torch.zeros(N, dtype=torch.bool, device=dev)
    has_retract = hasattr(env, "retract_succeeded")
    hole_sq_err, hole_n = 0.0, 0

    print(f"[eval] rolling out (max {args.steps} steps)...", flush=True)
    import time as _time
    t0 = _time.time()
    step = 0
    while step < args.steps and not bool(done_once.all()):
        with torch.no_grad():
            res = model({
                "is_train": True,          # is_train=False samples from the
                                           # distribution; we want deterministic mus
                "prev_actions": prev_actions,
                "obs": obs["proprio"], "img": obs["img"],
                "rnn_states": states, "seq_length": 1, "rnn_masks": None,
            })
        rs = res["rnn_states"]
        if isinstance(rs, tuple) and len(rs) == 2 and isinstance(rs[1], dict):
            states, last_aux = [t for t in rs[0]], rs[1]   # a2c_aux_cnn.py:671
        else:
            states, last_aux = list(rs), {}
        action = torch.clamp(res["mus"], -1.0, 1.0).float()

        # Track hole_pos accuracy only on envs still in their first episode, so
        # it describes the same trials the success rate does.
        aux = last_aux
        if "hole_pos" in aux:
            live = ~done_once
            if bool(live.any()):
                e = (aux["hole_pos"][live] - obs["aux_info"]["hole_pos"][live])
                hole_sq_err += float(e.pow(2).sum())
                hole_n += int(live.sum())

        obs, _rew, terminated, truncated, _extras = env.step(action)
        dones = (terminated | truncated).reshape(-1).bool()

        newly = dones & ~done_once
        ids = torch.nonzero(newly, as_tuple=False).reshape(-1)
        if ids.numel():
            succ = env._prev_episode_successes[ids]
            maxg = env.prev_episode_env_max_goals[ids].clamp_min(1)
            first_len[ids] = step + 1
            first_full[ids] = succ >= env.prev_episode_env_max_goals[ids]
            first_ratio[ids] = (succ.float() / maxg.float()).clamp(0.0, 1.0)
            rs = env.extras.get("episode_final", {}).get("retract_success") if has_retract else None
            if rs is not None:
                first_retract[ids] = rs[ids].to(dev) > 0.5
            done_once[ids] = True

        if bool(dones.any()):
            keep = (~dones).to(states[0].dtype).view(1, -1, 1)
            states = [s * keep for s in states]
        step += 1
        # Without this the rollout is silent for minutes and a hang is
        # indistinguishable from slowness.
        if step % 50 == 0:
            el = _time.time() - t0
            print(f"[eval] step {step:>5d}/{args.steps}  "
                  f"finished_ep1 {int(done_once.sum()):>4d}/{N}  "
                  f"{step/max(el,1e-9):.1f} steps/s", flush=True)

    valid = done_once & (first_len >= args.min_episode_steps)
    nv, n_done = int(valid.sum()), int(done_once.sum())
    hole_rmse = math.sqrt(hole_sq_err / max(1, hole_n)) * 1000.0

    result = {
        "checkpoint": str(ckpt_path),
        "grad_steps": ck.get("grad_steps"),
        "depth_aug": bool(args.depth_aug),
        "num_envs": N,
        "trials": nv,
        "bad_init": n_done - nv,
        "success_rate": float(first_full[valid].float().mean()) if nv else 0.0,
        "mean_goal_ratio": float(first_ratio[valid].mean()) if nv else 0.0,
        "retract_rate": float(first_retract[valid].float().mean()) if (has_retract and nv) else None,
        "hole_rmse_mm": hole_rmse,
        "hole_rmse_vs_baseline": hole_rmse / baseline_mm,
        "steps_run": step,
    }

    print("\n" + "=" * 78)
    print(f"{'':22s} {'student':>12s} {'teacher':>12s}")
    print("-" * 78)
    print(f"{'trials':22s} {result['trials']:>12d} {'911':>12s}")
    print(f"{'bad inits discarded':22s} {result['bad_init']:>12d} {'113':>12s}")
    print(f"{'success rate':22s} {result['success_rate']*100:>11.1f}% "
          f"{TEACHER_REFERENCE['success']*100:>11.1f}%")
    print(f"{'mean goal ratio':22s} {result['mean_goal_ratio']*100:>11.1f}% "
          f"{TEACHER_REFERENCE['goal_ratio']*100:>11.1f}%")
    if result["retract_rate"] is not None:
        print(f"{'retract rate':22s} {result['retract_rate']*100:>11.1f}% "
              f"{TEACHER_REFERENCE['retract']*100:>11.1f}%")
    print("-" * 78)
    print(f"{'hole_pos RMSE':22s} {result['hole_rmse_mm']:>11.1f}mm "
          f"({result['hole_rmse_vs_baseline']:.2f}x the {baseline_mm:.0f}mm "
          f"ignore-the-image baseline; spec is 2mm)")
    print("=" * 78)
    print("Scoring matches eval_offline: first episode per env, unstable inits")
    print("discarded. Teacher column is its measured tight_insertion result.")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out_json}")

    sys.stdout.flush()
    env.close()
    app.close()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
