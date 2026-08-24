"""Phase 3 check: the viser student-camera panel's data path.

The panel itself needs a browser, so this verifies everything up to the socket:
the env config, the rl_games wrapper that the interactive eval depends on, the
frame encoder, and the pickle round-trip the multiprocessing connection does.

The load-bearing assertion is `register_rlgames_env` succeeding with the camera
live. `RlGamesVecEnvWrapper` raises unless the env exposes a "policy" obs key
(isaaclab_rl/rl_games.py:128-130), so the Phase 1 student contract is
fundamentally incompatible with eval_isaacsim's rl_games-driven player. The new
`student_obs.emit_in_observations=False` is what lets the camera exist without
swapping the contract -- if that regresses, --student-cam takes the whole
interactive eval down with it.

    python tools/distillation/check_phase3_student_panel.py
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
AGENT = "rl_games_sapg_cfg_entry_point"

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=2)
cli.add_argument("--steps", type=int, default=8)
cli.add_argument("--device", default="cuda:0")
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = True  # what --student-cam sets
app = AppLauncher(_a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: F401,E402
from evaluation.eval_isaacsim import (  # noqa: E402
    _apply_env_overrides,
    _configure_agent,
    _instantiate_env,
    _load_env_cfg,
    _student_frame,
)
from isaacsimenvs.utils.rlgames_utils import register_rlgames_env  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


# Exactly what the worker does under --student-cam.
cfg = _load_env_cfg(TASK)
_apply_env_overrides(
    cfg,
    problem="tight_insertion",
    goal_mode="preInsertAndFinal",
    random_goal_fraction=0.0,
    insertion_success_tolerance=0.01,
    retract_success_tolerance=0.005,
    num_envs=args.num_envs,
    sim_device=args.device,
    sdf=False,
    keep_dr=False,
    extra_overrides={},
)
cfg.student_obs.enabled = True
cfg.student_obs.emit_in_observations = False

print("\n1. config decoupling")
check("yaml default keeps emit_in_observations True",
      bool(_load_env_cfg(TASK).student_obs.emit_in_observations))
env = _instantiate_env(TASK, cfg)
N = env.num_envs

print("\n2. obs contract is untouched")
env.reset()
for _ in range(2):
    env.step(torch.zeros(N, 29, device=env.device))
obs = env._get_observations()
check("keys are still {policy, critic}", set(obs) == {"policy", "critic"}, f"got {sorted(obs)}")
check("policy is (N, 140)", tuple(obs["policy"].shape) == (N, 140), f"got {tuple(obs['policy'].shape)}")
check("critic is (N, 162)", tuple(obs["critic"].shape) == (N, 162), f"got {tuple(obs['critic'].shape)}")
check("single_observation_space has no img/proprio keys",
      not ({"img", "proprio"} & set(env.single_observation_space.spaces)),
      f"got {sorted(env.single_observation_space.spaces)}")

print("\n3. rl_games wrapper still constructible (the blocker)")
agent_cfg = _configure_agent(
    TASK, AGENT, rl_device=args.device, num_envs=N,
    deterministic=True, games=10**9, extra_overrides={},
)
clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
try:
    wrapped = register_rlgames_env(
        env, rl_device=args.device, clip_obs=clip_obs, clip_actions=clip_actions
    )
    check("register_rlgames_env succeeds with the camera live", True)
    check("wrapper observation_space is the teacher's 140",
          tuple(wrapped.observation_space.shape) == (140,),
          f"got {tuple(wrapped.observation_space.shape)}")
except Exception as exc:  # noqa: BLE001
    check("register_rlgames_env succeeds with the camera live", False,
          f"{type(exc).__name__}: {exc}")

print("\n4. the camera is actually live and the frame encodes")
u8, stats = _student_frame(env)
check("frame is (90, 160) uint8", u8.shape == (90, 160) and u8.dtype == np.uint8,
      f"got {u8.shape} {u8.dtype}")
check("frame is not constant (sky-camera regression)", stats["n_unique"] > 1, f"stats={stats}")
check("stats are finite", all(np.isfinite(v) for v in stats.values()), f"stats={stats}")
check("normalized range within [0, 1]", 0.0 <= stats["min"] and stats["max"] <= 1.0,
      f"min={stats['min']:.4f} max={stats['max']:.4f}")

print("\n5. wire format (multiprocessing.Connection pickles)")
payload = ("student_img", u8, stats)
rt = pickle.loads(pickle.dumps(payload))
check("payload survives a pickle round-trip", rt[0] == "student_img" and np.array_equal(rt[1], u8))
check("payload is small enough to stream", len(pickle.dumps(payload)) < 64_000,
      f"{len(pickle.dumps(payload))} bytes/frame")
rgb = np.repeat(rt[1][:, :, None], 3, axis=2)  # what the parent hands viser
check("parent expands to viser's HxWx3 uint8",
      rgb.shape == (90, 160, 3) and rgb.dtype == np.uint8, f"got {rgb.shape} {rgb.dtype}")

print("\n6. frames change as the scene moves")
firsts = []
for _ in range(args.steps):
    env.step(torch.zeros(N, 29, device=env.device))
    firsts.append(_student_frame(env)[0])
check("consecutive frames are not all identical",
      any(not np.array_equal(firsts[0], f) for f in firsts[1:]),
      f"{len(firsts)} frames sampled")
check("no frame went constant mid-rollout",
      all(int(np.unique(f).size) > 1 for f in firsts),
      f"n_unique = {[int(np.unique(f).size) for f in firsts]}")

print("\n" + "=" * 62)
passed = not FAILURES
print("PHASE 3 CHECK PASSED" if passed else
      f"PHASE 3 CHECK FAILED — {len(FAILURES)} failure(s): {FAILURES}")
print("=" * 62)
print("NOTE: the viser widget itself needs a browser; this covers the data")
print("path up to the socket, not the rendered panel.")
print(f"RESULT_GATE {'PASS' if passed else 'FAIL'}")
sys.stdout.flush()
sys.stderr.flush()
env.close()
app.close()
os._exit(0 if passed else 1)
