"""Phase 1 acceptance check: the depth-distillation obs contract.

Asserts that `PreciseAssemblyEnv._get_observations()` emits the five keys the
DAgger loop expects, in the right shapes, with sane aux labels — and that the
student image is not constant.

That last check is the important one. `depth_preprocess_mode=window_normalize`
saturates everything past `depth_max_m` to 1.0, so a camera aimed at nothing
returns a perfectly valid-looking constant image rather than an error. That is
exactly how the identity-quaternion sky-camera bug survived unnoticed. Keep
this assertion permanent.

    python tools/distillation/check_phase1_obs.py --num-envs 4
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=4)
cli.add_argument("--steps", type=int, default=6, help="warmup steps before reading obs")
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = True  # student_obs requires a camera-enabled app
app = AppLauncher(_a).app

import torch  # noqa: E402
import isaacsimenvs  # noqa: F401,E402
from evaluation.eval_isaacsim import (  # noqa: E402
    _load_env_cfg,
    _apply_env_overrides,
    _instantiate_env,
)

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def make_cfg(num_envs: int, random_goal_fraction: float = 0.0):
    cfg = _load_env_cfg(TASK)
    _apply_env_overrides(
        cfg,
        problem="tight_insertion",
        goal_mode="preInsertAndFinal",
        random_goal_fraction=random_goal_fraction,
        insertion_success_tolerance=0.01,
        retract_success_tolerance=0.005,
        num_envs=num_envs,
        sim_device="cuda:0",
        sdf=False,
        keep_dr=False,
        extra_overrides={},
    )
    cfg.student_obs.enabled = True
    return cfg


# --- 0. the sentinel guard fires before any scene is built -------------------
print("\n0. random_goal_fraction guard")
try:
    _instantiate_env(TASK, make_cfg(args.num_envs, random_goal_fraction=0.5))
    check("rejects random_goal_fraction > 0", False, "no ValueError raised")
except ValueError as e:
    check("rejects random_goal_fraction > 0", "sentinel" in str(e), f"({e})")
except Exception as e:  # noqa: BLE001
    check("rejects random_goal_fraction > 0", False, f"wrong exception: {type(e).__name__}: {e}")

# --- build the real env -----------------------------------------------------
cfg = make_cfg(args.num_envs)
env = _instantiate_env(TASK, cfg)
uenv = env.unwrapped
N = uenv.num_envs

env.reset()
for _ in range(args.steps):
    env.step(torch.zeros(N, 29, device=uenv.device))

obs = uenv._get_observations()

# --- 1. key set -------------------------------------------------------------
print("\n1. obs keys")
expected = {"proprio", "img", "teacher_obs", "critic", "aux_info"}
check("exactly 5 keys", set(obs) == expected, f"got {sorted(obs)}")

# --- 2. tensor shapes -------------------------------------------------------
print("\n2. tensor shapes")
for key, shape in [
    ("proprio", (N, 87)),
    ("img", (N, 1, 90, 160)),
    ("teacher_obs", (N, 140)),
    ("critic", (N, 162)),
]:
    got = tuple(obs[key].shape) if key in obs else None
    check(f"{key} == {shape}", got == shape, f"got {got}")

# --- 3. the image is not constant ------------------------------------------
print("\n3. image content (sky-camera regression test)")
img = obs["img"].detach().float()
n_unique_per_env = [int(img[i].unique().numel()) for i in range(N)]
check(
    "every env has a non-constant image",
    all(u > 1 for u in n_unique_per_env),
    f"n_unique per env = {n_unique_per_env}",
)
check(
    "no NaN/Inf in image",
    bool(torch.isfinite(img).all()),
    f"finite fraction = {torch.isfinite(img).float().mean().item():.4f}",
)
lo, hi = float(img.min()), float(img.max())
check("image within window-normalized [0, 1]", 0.0 <= lo and hi <= 1.0, f"range=({lo:.4f}, {hi:.4f})")

# --- 4. aux labels ----------------------------------------------------------
print("\n4. aux_info labels")
aux = obs["aux_info"]
check("aux keys", set(aux) == {"hole_pos", "keypoints_rel_goal", "object_pos"}, f"got {sorted(aux)}")
for key, shape in [("hole_pos", (N, 3)), ("keypoints_rel_goal", (N, 12)), ("object_pos", (N, 3))]:
    got = tuple(aux[key].shape) if key in aux else None
    check(f"{key} == {shape}", got == shape, f"got {got}")

hole = aux["hole_pos"]
check(
    "no (0, 0, -1) sentinel in hole_pos",
    bool((hole[:, 2] > 0.0).all()),
    f"z range=({float(hole[:, 2].min()):.4f}, {float(hole[:, 2].max()):.4f})",
)
xr = tuple(float(v) for v in uenv.cfg.precise_assembly.hole_x_range)
yr = tuple(float(v) for v in uenv.cfg.precise_assembly.hole_y_range)
check(
    "hole_pos xy within configured ranges",
    bool((hole[:, 0] >= xr[0]).all() and (hole[:, 0] <= xr[1]).all()
         and (hole[:, 1] >= yr[0]).all() and (hole[:, 1] <= yr[1]).all()),
    f"x in {xr}, y in {yr}",
)
# hole_pos and object_pos must share a frame (both env-relative). If one were
# world-frame the gap would blow up with env_spacing across the grid.
gap = (aux["object_pos"] - hole).norm(dim=-1)
check(
    "object_pos and hole_pos share the env-relative frame",
    bool((gap < 1.0).all()),
    f"max |object - hole| = {float(gap.max()):.4f} m",
)
check(
    "aux labels are finite",
    all(bool(torch.isfinite(v).all()) for v in aux.values()),
)

# --- 5. teacher_obs is still the noisy tensor ------------------------------
print("\n5. teacher_obs / critic wiring")
# `keypoints_rel_goal` sits in obs_list, so the teacher's copy is the noisy,
# goal-noise-adjusted one. It must NOT equal the clean aux label, else the
# teacher is being fed privileged information it never trained on.
if uenv._goal_kp_obs_slice is not None:
    teacher_kp = obs["teacher_obs"][:, uenv._goal_kp_obs_slice]
    check(
        "teacher_obs keypoints_rel_goal differs from the clean aux label",
        not torch.allclose(teacher_kp, aux["keypoints_rel_goal"]),
        f"max abs diff = {float((teacher_kp - aux['keypoints_rel_goal']).abs().max()):.6f}",
    )
else:
    print("  [SKIP] goal-noise slice not configured")

# --- 6. spaces and DR buffers were not corrupted ---------------------------
print("\n6. spaces / DR buffers")
space_keys = set(uenv.single_observation_space.spaces)
check(
    "single_observation_space describes the 4 tensor keys",
    space_keys == {"proprio", "img", "teacher_obs", "critic"},
    f"got {sorted(space_keys)}",
)
check(
    "aux_info absent from the gym space (it is a label, not an observation)",
    "aux_info" not in space_keys,
)
check(
    "teacher_obs kept in the space (setup_rlgames_env keys off it)",
    "teacher_obs" in space_keys,
)
check(
    "cfg.observation_space still the teacher obs_list dim",
    int(uenv.cfg.observation_space) == 140,
    f"got {int(uenv.cfg.observation_space)}",
)
if hasattr(uenv, "_obs_queue"):
    check(
        "DR _obs_queue sized to the teacher obs dim",
        int(uenv._obs_queue.shape[-1]) == 140,
        f"got {tuple(uenv._obs_queue.shape)}",
    )

print("\n" + "=" * 62)
if FAILURES:
    print(f"PHASE 1 CHECK FAILED — {len(FAILURES)} failure(s):")
    for f in FAILURES:
        print(f"  - {f}")
else:
    print("PHASE 1 CHECK PASSED")
print("=" * 62)

env.close()
app.close()
sys.exit(1 if FAILURES else 0)
