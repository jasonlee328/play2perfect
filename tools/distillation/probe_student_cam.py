"""Probe v2: compute a correct look-at camera pose, then verify the depth image."""
import argparse, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
OUT = Path(os.environ.get("PROBE_OUT_DIR", "/tmp/play2perfect_probe"))
OUT.mkdir(parents=True, exist_ok=True)

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a).app

import numpy as np, torch
from isaaclab.utils.math import (create_rotation_matrix_from_view, quat_from_matrix,
                                 convert_camera_frame_orientation_convention)
import isaacsimenvs  # noqa: F401
from evaluation.eval_isaacsim import _load_env_cfg, _apply_env_overrides, _instantiate_env

EYE    = (0.0, -0.75, 0.95)     # pulled in from (0,-1,1) so the table fits the depth window
TARGET = (0.0,  0.0,  0.53)     # table top = table_reset_z 0.38 + TABLE_HALF_HEIGHT 0.15

# helper returns OPENGL convention (-Z forward); convert to ROS (+Z forward)
R = create_rotation_matrix_from_view(torch.tensor([EYE]), torch.tensor([TARGET]),
                                     up_axis="Z", device="cpu")
q_gl  = quat_from_matrix(R)
q_ros = convert_camera_frame_orientation_convention(q_gl, origin="opengl", target="ros")
QUAT = tuple(float(v) for v in q_ros[0])
print(f"[probe] quat opengl = {tuple(round(float(v),6) for v in q_gl[0])}")
print(f"[probe] eye={EYE} target={TARGET}")
print(f"[probe] computed quat_wxyz = {tuple(round(v,6) for v in QUAT)}")
print(f"[probe] eye->target distance = {np.linalg.norm(np.array(TARGET)-np.array(EYE)):.3f} m")

cfg = _load_env_cfg("Isaacsimenvs-PreciseAssembly-Direct-v0")
_apply_env_overrides(cfg, problem="tight_insertion", goal_mode="preInsertAndFinal",
    random_goal_fraction=0.0, insertion_success_tolerance=0.01,
    retract_success_tolerance=0.005, num_envs=4, sim_device="cuda:0",
    sdf=False, keep_dr=False, extra_overrides={})
cfg.student_obs.enabled = True
cfg.student_obs.image_enabled = True
cfg.student_obs.image_modality = "depth"
cfg.student_obs.camera_backend = "tiled"
cfg.student_obs.use_depth_aug = False
cfg.student_obs.hide_goal_viz = True
# camera_pos/quat now come from the repo default
cfg.student_obs.depth_preprocess_mode = "window_normalize"   # raw metres, so we can judge the window

env = _instantiate_env("Isaacsimenvs-PreciseAssembly-Direct-v0", cfg)
env.reset()
for _ in range(6):
    env.step(torch.zeros(env.num_envs, 29, device=env.device))

s = env.unwrapped.get_student_obs()
img = s["image"].detach().float().cpu().numpy()
print(f"[probe] image tensor {img.shape}  proprio {tuple(s['proprio'].shape)}")
np.save(OUT/"depth_default.npy", img)

for i in range(img.shape[0]):
    im = img[i,0]; fin = im[np.isfinite(im)]
    print(f"[probe] env{i}: n_unique={np.unique(im).size}  "
          f"depth range=({fin.min():.3f},{fin.max():.3f})m  median={np.median(fin):.3f}m  "
          f"frac in (0.45,1.25)={(np.logical_and(fin>0.45,fin<1.25)).mean():.2%}")

from PIL import Image
im0 = np.nan_to_num(img[0,0], nan=0.0, posinf=0.0)
lo, hi = np.percentile(im0[im0>0], [2,98]) if (im0>0).any() else (0,1)
vis = np.clip((im0-lo)/max(hi-lo,1e-6), 0, 1)
Image.fromarray((vis*255).astype(np.uint8)).resize((640,360), Image.NEAREST).save(OUT/"depth_default.png")
print(f"[probe] wrote {OUT/'depth_fixed.png'}")
env.close(); app.close()
