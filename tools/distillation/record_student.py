"""Record the distilled student executing, and score it.

Writes two videos per run:

  scene.mp4   third-person viewport (DirectRLEnv.render with render_mode
              "rgb_array", the same path train.py --capture_video uses)
  depth.mp4   the student's own 90x160 depth input, upscaled -- what the policy
              actually sees, after preprocess / window-normalize / delay

and prints the same first-episode success scoring `eval_offline` applies to the
teacher, so the recording doubles as a measurement.

`--blank-image` replaces the camera with a constant. The hole is randomized over
+-187 x +-100 mm and is not recoverable from proprioception, so if success
survives that, the policy is not using vision -- it is locating the hole some
other way (contact, search) within the env's 10 mm insertion tolerance. Run it
both ways and compare the printed success rates.

    python tools/distillation/record_student.py --checkpoint runs/distill_v6/student_final.pth
    python tools/distillation/record_student.py --checkpoint ... --blank-image
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"

cli = argparse.ArgumentParser()
cli.add_argument("--checkpoint", required=True)
cli.add_argument("--problem", default="tight_insertion")
cli.add_argument("--num-envs", type=int, default=64,
                 help="Scored trials. The video always follows env 0.")
cli.add_argument("--steps", type=int, default=1200)
cli.add_argument("--min-episode-steps", type=int, default=100)
cli.add_argument("--blank-image", action="store_true",
                 help="Feed a constant image. If success survives, vision is unused.")
cli.add_argument("--depth-aug", action="store_true")
cli.add_argument("--fps", type=int, default=30)
cli.add_argument("--out-dir", default=None)
cli.add_argument("--device", default="cuda:0")
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = True
app = AppLauncher(_a).app

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: F401,E402
from evaluation.eval_isaacsim import (  # noqa: E402
    _apply_env_overrides,
    _load_env_cfg,
)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from isaacsimenvs.distillation.student_policy import StudentPolicy  # noqa: E402

ckpt = Path(args.checkpoint)
if not ckpt.is_absolute():
    ckpt = REPO / ckpt
if not ckpt.is_file():
    raise SystemExit(f"checkpoint not found: {ckpt}")

tag = "blank" if args.blank_image else "camera"
out = Path(args.out_dir) if args.out_dir else REPO / "runs" / f"record_{ckpt.parent.name}_{tag}"
out.mkdir(parents=True, exist_ok=True)

cfg = _load_env_cfg(TASK)
_apply_env_overrides(
    cfg, problem=args.problem, goal_mode="preInsertAndFinal",
    random_goal_fraction=0.0, insertion_success_tolerance=0.01,
    retract_success_tolerance=0.005, num_envs=args.num_envs,
    sim_device=args.device, sdf=False, keep_dr=False, extra_overrides={},
)
cfg.student_obs.enabled = True
# False keeps the env's {policy, critic} contract; StudentPolicy reads the
# camera through get_student_obs() directly, so it does not need the remap.
cfg.student_obs.emit_in_observations = False
cfg.student_obs.use_depth_aug = bool(args.depth_aug)

# render_mode="rgb_array" makes DirectRLEnv.render() lazily create ONE
# replicator render product at cfg.viewer.cam_prim_path -- num_envs-independent,
# so it does not scale the memory the way a per-env Camera sensor would.
import gymnasium as gym  # noqa: E402

env = gym.make(TASK, cfg=cfg, render_mode="rgb_array")
uenv = env.unwrapped
N, dev = uenv.num_envs, uenv.device

print("[rec] env.reset() ...", flush=True)
obs, _ = env.reset()
print("[rec] reset OK", flush=True)

student = StudentPolicy(
    str(ckpt), num_envs=N, action_dim=int(uenv.cfg.action_space),
    proprio_dim=int(uenv.get_student_obs()["proprio"].shape[-1]),
    device=args.device,
    agent_cfg=load_cfg_from_registry(TASK, "rl_games_student_cfg_entry_point"),
    blank_image=args.blank_image,
)
print(f"[rec] student grad_steps={student.grad_steps} blank_image={args.blank_image}", flush=True)

done_once = torch.zeros(N, dtype=torch.bool, device=dev)
first_len = torch.zeros(N, dtype=torch.long, device=dev)
first_full = torch.zeros(N, dtype=torch.bool, device=dev)
first_ratio = torch.zeros(N, dtype=torch.float, device=dev)

scene_frames: list[np.ndarray] = []
depth_frames: list[np.ndarray] = []

for step in range(args.steps):
    action = student.act(uenv)

    # env 0's depth input, as the policy receives it (blanked if ablating)
    g = uenv.get_student_obs()["image"][0, 0].detach().float()
    if args.blank_image:
        g = torch.zeros_like(g)
    vis = (g.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    depth_frames.append(np.repeat(np.repeat(vis, 5, 0), 5, 1))   # 5x nearest-neighbour

    rgb = env.render()
    if rgb is not None:
        scene_frames.append(np.asarray(rgb, dtype=np.uint8))

    obs, _rew, term, trunc, _extras = env.step(action)
    dones = (term | trunc).reshape(-1).bool()

    newly = dones & ~done_once
    ids = torch.nonzero(newly, as_tuple=False).reshape(-1)
    if ids.numel():
        succ = uenv._prev_episode_successes[ids]
        maxg = uenv.prev_episode_env_max_goals[ids].clamp_min(1)
        first_len[ids] = step + 1
        first_full[ids] = succ >= uenv.prev_episode_env_max_goals[ids]
        first_ratio[ids] = (succ.float() / maxg.float()).clamp(0, 1)
        done_once[ids] = True
    if bool(dones.any()):
        student.reset(dones)
    if (step + 1) % 200 == 0:
        print(f"[rec] step {step+1}/{args.steps}  finished_ep1 {int(done_once.sum())}/{N}",
              flush=True)
    if bool(done_once.all()):
        break

valid = done_once & (first_len >= args.min_episode_steps)
nv = int(valid.sum())
result = {
    "checkpoint": str(ckpt), "blank_image": bool(args.blank_image),
    "grad_steps": student.grad_steps, "num_envs": N,
    "trials": nv, "bad_init": int(done_once.sum()) - nv,
    "success_rate": float(first_full[valid].float().mean()) if nv else 0.0,
    "mean_goal_ratio": float(first_ratio[valid].mean()) if nv else 0.0,
    "steps_run": step + 1,
}

for name, frames in (("scene", scene_frames), ("depth", depth_frames)):
    if not frames:
        print(f"[rec] no {name} frames (render returned None?)")
        continue
    path = out / f"{name}.mp4"
    imageio.mimwrite(str(path), frames, fps=args.fps, quality=8, macro_block_size=1)
    print(f"[rec] wrote {path}  ({len(frames)} frames)")

(out / "result.json").write_text(json.dumps(result, indent=2))
print("\n" + "=" * 66)
print(f"  image source      {'BLANK (constant)' if args.blank_image else 'camera'}")
print(f"  trials            {result['trials']}  (bad inits discarded {result['bad_init']})")
print(f"  success rate      {result['success_rate']*100:.1f}%")
print(f"  mean goal ratio   {result['mean_goal_ratio']*100:.1f}%")
print("=" * 66)
print("teacher on this scoring (eval_offline, tight_insertion): 96.9%")
print("Run with and without --blank-image; if the two match, vision is unused.")
sys.stdout.flush()
env.close()
app.close()
os._exit(0)
