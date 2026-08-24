"""Is the hole actually inside the camera frame, over its whole range?

Everything checked so far only proves the depth image is not constant. That
rules out the camera pointing at the sky; it says nothing about whether the
thing the student must localize is visible.

`hole_pos` is randomized per episode over `hole_x_range` x `hole_y_range`
(+-187.5 x +-100 mm by default). If part of that rectangle falls outside the
frustum, then for those episodes the hole is simply not in the image, the aux
target is unpredictable, and `hole_rmse` cannot fall below the spread of the
unobservable part -- which would look exactly like an encoder that plateaus.

This forces the hole to the corners and centre of its range, renders each, and
both projects `hole_pos` into pixel coordinates and writes a PNG so the framing
can be eyeballed rather than argued about.

    python tools/distillation/check_camera_framing.py --out /tmp/framing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"

cli = argparse.ArgumentParser()
cli.add_argument("--out", default="/tmp/play2perfect_framing")
cli.add_argument("--device", default="cuda:0")
cli.add_argument("--settle-steps", type=int, default=4)
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = True
app = AppLauncher(_a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: F401,E402
from evaluation.eval_isaacsim import (  # noqa: E402
    _apply_env_overrides,
    _instantiate_env,
    _load_env_cfg,
)

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

cfg = _load_env_cfg(TASK)
_apply_env_overrides(
    cfg, problem="tight_insertion", goal_mode="preInsertAndFinal",
    random_goal_fraction=0.0, insertion_success_tolerance=0.01,
    retract_success_tolerance=0.005, num_envs=9,
    sim_device=args.device, sdf=False, keep_dr=False, extra_overrides={},
)
cfg.student_obs.enabled = True
cfg.student_obs.use_depth_aug = False   # want the clean geometry, not noise
env = _instantiate_env(TASK, cfg)
sc = env.cfg.student_obs
pih = env.cfg.precise_assembly

# --- intrinsics from the USD pinhole parameters ---------------------------
W, H = int(sc.image_input_width), int(sc.image_input_height)
f = float(sc.focal_length)
ha = float(sc.horizontal_aperture)
va = ha * H / W                     # square pixels
fx, fy = f * W / ha, f * H / va
cx = W / 2.0 + float(sc.horizontal_aperture_offset) * fx / f
cy = H / 2.0 + float(sc.vertical_aperture_offset) * fy / f
print(f"\n[intrinsics] {W}x{H}  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

cam_pos = np.array(sc.camera_pos, dtype=np.float64)
qw, qx, qy, qz = (float(v) for v in sc.camera_quat_wxyz)
# quat -> rotation matrix (camera->env). Convention is "ros": +Z optical axis,
# +X right, +Y down.
R = np.array([
    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
])
print(f"[extrinsics] eye={tuple(cam_pos)} quat_wxyz=({qw},{qx},{qy},{qz}) ros")


def project(p_env: np.ndarray):
    """Env-relative point -> (u, v, depth_along_optical_axis)."""
    p_cam = R.T @ (p_env - cam_pos)
    X, Y, Z = p_cam
    if Z <= 1e-6:
        return None, None, Z
    return fx * X / Z + cx, fy * Y / Z + cy, Z


# --- the corners and centre of hole_pos' sampling range -------------------
x0, x1 = (float(v) for v in pih.hole_x_range)
y0, y1 = (float(v) for v in pih.hole_y_range)
cases = [
    ("x_min_y_min", x0, y0), ("x_min_y_max", x0, y1),
    ("x_max_y_min", x1, y0), ("x_max_y_max", x1, y1),
    ("centre", 0.0, 0.0),
    ("x_min_y_0", x0, 0.0), ("x_max_y_0", x1, 0.0),
    ("x_0_y_min", 0.0, y0), ("x_0_y_max", 0.0, y1),
]
print(f"\nhole_x_range = ({x0}, {x1})   hole_y_range = ({y0}, {y1})")

env.reset()
# One env per case, hole forced to that position.
z = float(env.hole_pos[0, 2])
for i, (_name, hx, hy) in enumerate(cases):
    env.hole_pos[i, 0] = hx
    env.hole_pos[i, 1] = hy
origins = env.scene.env_origins
hole_q = env.hole_quat_wxyz.clone()
pose = torch.cat([env.hole_pos + origins, hole_q], dim=-1)
env.hole.write_root_pose_to_sim(pose)
for _ in range(args.settle_steps):
    env.step(torch.zeros(env.num_envs, 29, device=env.device))

img = env.get_student_obs()["image"].detach().float().cpu().numpy()

print(f"\n{'case':14s} {'hole (x,y) mm':>18s} {'pixel (u,v)':>16s} {'depth m':>8s}  in-frame")
print("-" * 76)
in_frame = []
for i, (name, hx, hy) in enumerate(cases):
    u, v, d = project(np.array([hx, hy, z]))
    ok = u is not None and 0 <= u < W and 0 <= v < H
    in_frame.append(ok)
    uv = f"({u:6.1f},{v:6.1f})" if u is not None else "  behind cam  "
    print(f"{name:14s} {f'({hx*1000:+7.1f},{hy*1000:+6.1f})':>18s} {uv:>16s} "
          f"{d:8.3f}  {'YES' if ok else 'NO  <-- OUT OF FRAME'}")

# --- margin: how much of the rectangle is inside? -------------------------
NS = 41
inside = 0
for xi in np.linspace(x0, x1, NS):
    for yi in np.linspace(y0, y1, NS):
        u, v, d = project(np.array([xi, yi, z]))
        if u is not None and 0 <= u < W and 0 <= v < H:
            inside += 1
frac = inside / (NS * NS)
print(f"\nfraction of the hole_pos rectangle inside the frame: {frac*100:.1f}% "
      f"({inside}/{NS*NS} grid points)")

# --- write images with the projected hole marked --------------------------
try:
    from PIL import Image, ImageDraw

    for i, (name, hx, hy) in enumerate(cases):
        g = img[i, 0]
        finite = g[np.isfinite(g)]
        lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
        vis = np.clip((np.nan_to_num(g, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1)
        im = Image.fromarray((vis * 255).astype(np.uint8)).convert("RGB")
        im = im.resize((W * 5, H * 5), Image.NEAREST)
        u, v, _d = project(np.array([hx, hy, z]))
        if u is not None:
            dr = ImageDraw.Draw(im)
            U, V = u * 5, v * 5
            dr.line([(U - 14, V), (U + 14, V)], fill=(255, 0, 0), width=2)
            dr.line([(U, V - 14), (U, V + 14)], fill=(255, 0, 0), width=2)
        im.save(OUT / f"{i}_{name}.png")
    print(f"\nwrote {len(cases)} images to {OUT} (red cross = projected hole_pos)")
except ImportError:
    print("\n(PIL missing; skipped image dump)")

print("\n" + "=" * 76)
if all(in_frame) and frac > 0.99:
    print("FRAMING OK: the whole hole_pos range projects inside the image.")
else:
    print("FRAMING PROBLEM: part of the hole_pos range is outside the frame.")
    print("For those episodes the hole is not observable, the aux target is")
    print("unpredictable, and hole_rmse cannot converge. Move the camera.")
print("=" * 76)
sys.stdout.flush()
env.close()
app.close()
os._exit(0 if (all(in_frame) and frac > 0.99) else 1)
