"""Throughput benchmark for tight_insertion with the student depth camera on.

One process per config (Kit can't re-init a different env in-process).
Prints a single BENCH_JSON line.
"""
import argparse, json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, required=True)
cli.add_argument("--camera", type=int, default=1)      # 1 = student depth cam on
cli.add_argument("--warmup", type=int, default=30)
cli.add_argument("--steps", type=int, default=120)
args = cli.parse_args()

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args([])
a.headless = True
a.enable_cameras = bool(args.camera)
app = AppLauncher(a).app

import torch
import isaacsimenvs  # noqa: F401
from evaluation.eval_isaacsim import _load_env_cfg, _apply_env_overrides, _instantiate_env

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
res = {"num_envs": args.num_envs, "camera": bool(args.camera)}
env = None
try:
    cfg = _load_env_cfg(TASK)
    _apply_env_overrides(cfg, problem="tight_insertion", goal_mode="preInsertAndFinal",
        random_goal_fraction=0.0, insertion_success_tolerance=0.01,
        retract_success_tolerance=0.005, num_envs=args.num_envs, sim_device="cuda:0",
        sdf=False, keep_dr=False, extra_overrides={})
    if args.camera:
        cfg.student_obs.enabled = True
        cfg.student_obs.image_enabled = True
        cfg.student_obs.image_modality = "depth"
        cfg.student_obs.camera_backend = "tiled"
        cfg.student_obs.use_depth_aug = True     # realistic: aug is on in training

    t0 = time.perf_counter()
    env = _instantiate_env(TASK, cfg)
    res["build_s"] = round(time.perf_counter() - t0, 1)

    act = torch.zeros(env.num_envs, 29, device=env.device)
    env.reset()
    for _ in range(args.warmup):
        env.step(act)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()

    # (a) step only
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(act)
    torch.cuda.synchronize()
    dt_step = time.perf_counter() - t0

    # (b) step + student obs read (image preprocess + delay queues)
    dt_full = dt_step
    if args.camera:
        t0 = time.perf_counter()
        for _ in range(args.steps):
            env.step(act)
            env.unwrapped.get_student_obs()
        torch.cuda.synchronize()
        dt_full = time.perf_counter() - t0

    res["step_hz"]        = round(args.steps / dt_step, 1)
    res["full_hz"]        = round(args.steps / dt_full, 1)
    res["envsteps_per_s"] = int(args.steps * args.num_envs / dt_full)
    res["peak_vram_gb"]   = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["torch_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    res["ok"] = True
except Exception as e:
    res["ok"] = False
    res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
finally:
    print("BENCH_JSON " + json.dumps(res), flush=True)
    try:
        if env is not None: env.close()
    except Exception: pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
