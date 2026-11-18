"""Visual domain randomization for the student cameras (A6, 2026-11-18).

Off unless ``student_obs.vdr_enabled=true``. Everything here changes only what the
cameras see; physics, rewards and the teacher observations are untouched.

Per episode (on reset), per env:
- camera extrinsics: both student cameras moved by U(+-vdr_cam_pos_m) per axis and
  rotated by U(+-vdr_cam_rot_deg) about each world axis, around their configured pose;
- camera intrinsics: focal length x U(vdr_focal_scale) (= zoom; principal point stays
  centred), both cameras;
- colours: table, tray (Hole), sticks (Object + Distractors share one colour), floor
  patch and backdrop walls get random diffuse colour + roughness (vdr_colors);
- lighting: a per-env sphere light with random position / intensity / colour
  (vdr_lights) and the global dome light intensity x U(vdr_dome_scale) (shared by all
  envs, resampled whenever any env resets).

The floor patch and backdrop walls (vdr_backdrop) are visual-only cubes (no collision).
"""

from __future__ import annotations

import math

import torch

_PARTS_ENV = ("Table", "Hole", "Object")   # prims under /World/envs/env_i/


def enabled(env) -> bool:
    cfg = getattr(env.cfg, "student_obs", None)
    return bool(cfg is not None and getattr(cfg, "vdr_enabled", False) and getattr(env, "student_camera", None) is not None)


def _stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


def _make_material(stage, path: str):
    from pxr import Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    col = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    rough = sh.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat, col.GetAttr(), rough.GetAttr()


def _bind(stage, prim_path: str, mat) -> bool:
    from pxr import UsdShade
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return True


def setup(env) -> None:
    """Create per-env materials / lights / backdrop. Call after the scene is spawned."""
    if not enabled(env):
        return
    from pxr import Gf, UsdGeom, UsdLux

    cfg = env.cfg.student_obs
    stage = _stage()
    g = torch.Generator().manual_seed(int(getattr(cfg, "vdr_seed", 0)))
    env._vdr_gen = g
    env._vdr = {"color": [], "light": [], "focal": {}}
    n_dist = len(getattr(env, "distractors", ()) or ())
    for i in range(env.num_envs):
        root = f"/World/envs/env_{i}"
        attrs = {}
        if getattr(cfg, "vdr_backdrop", False):
            # floor patch (3 x 3 m) + back wall behind the robot (y = 1.3) + two side walls
            for name, pos, size in (("VdrFloor", (0.0, 0.3, 0.002), (3.0, 3.0, 0.002)),
                                    ("VdrWallBack", (0.0, 1.3, 1.0), (3.0, 0.02, 2.0)),
                                    ("VdrWallL", (-1.3, 0.3, 1.0), (0.02, 3.0, 2.0)),
                                    ("VdrWallR", (1.3, 0.3, 1.0), (0.02, 3.0, 2.0))):
                cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.XformCommonAPI(cube)
                xf.SetTranslate(Gf.Vec3d(*pos))
                xf.SetScale(Gf.Vec3f(*size))
        if getattr(cfg, "vdr_colors", False):
            groups = {"table": [f"{root}/Table"], "tray": [f"{root}/Hole"],
                      "sticks": [f"{root}/Object"] + [f"{root}/Distractor{k}" for k in range(n_dist)]}
            if getattr(cfg, "vdr_backdrop", False):
                groups["floor"] = [f"{root}/VdrFloor"]
                groups["wall"] = [f"{root}/VdrWallBack", f"{root}/VdrWallL", f"{root}/VdrWallR"]
            for gname, paths in groups.items():
                mat, col, rough = _make_material(stage, f"{root}/VdrLooks/{gname}")
                if any([_bind(stage, p, mat) for p in paths]):
                    attrs[gname] = (col, rough)
        env._vdr["color"].append(attrs)
        if getattr(cfg, "vdr_lights", False):
            light = UsdLux.SphereLight.Define(stage, f"{root}/VdrLight")
            light.CreateRadiusAttr(0.15)
            light.CreateIntensityAttr(0.0)
            light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
            env._vdr["light"].append(light)
    dome = stage.GetPrimAtPath("/World/Light")
    env._vdr["dome"] = dome if dome.IsValid() else None
    env._vdr["dome_base"] = float(dome.GetAttribute("inputs:intensity").Get()) if dome.IsValid() else None
    print(f"[visual_dr] setup: {env.num_envs} envs, colour groups "
          f"{sorted(env._vdr['color'][0].keys()) if env._vdr['color'] else []}, "
          f"lights {bool(env._vdr['light'])}", flush=True)


def _u(g, lo, hi, *shape):
    return lo + (hi - lo) * torch.rand(*shape, generator=g)


def _rand_color(g, kind: str):
    if kind == "sticks":      # keep sticks wood-like-to-arbitrary, but never near-black
        return _u(g, 0.15, 1.0, 3)
    return _u(g, 0.0, 1.0, 3)


def _cam_pose_jitter(env, cam, pos, quat_wxyz, env_ids, cfg):
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
    n = env_ids.numel()
    dev = env.device
    p_amp = float(getattr(cfg, "vdr_cam_pos_m", 0.0))
    r_amp = math.radians(float(getattr(cfg, "vdr_cam_rot_deg", 0.0)))
    if p_amp <= 0 and r_amp <= 0:
        return
    base_p = torch.tensor(pos, device=dev, dtype=torch.float32).expand(n, 3)
    base_q = torch.tensor(quat_wxyz, device=dev, dtype=torch.float32).expand(n, 4)
    dp = (torch.rand(n, 3, device=dev) * 2 - 1) * p_amp
    e = (torch.rand(n, 3, device=dev) * 2 - 1) * r_amp
    dq = quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2])
    q = quat_mul(dq, base_q)                       # small world-frame rotation of the base pose
    p = env.scene.env_origins[env_ids] + base_p + dp
    cam.set_world_poses(p, q, env_ids=env_ids.tolist(), convention=str(cfg.camera_convention))


def _focal_jitter(env, cam, key, env_ids, cfg):
    lo, hi = (float(x) for x in getattr(cfg, "vdr_focal_scale", (1.0, 1.0)))
    if lo == 1.0 and hi == 1.0:
        return
    base = float(cfg.focal_length)
    s = _u(env._vdr_gen, lo, hi, env_ids.numel())
    for j, i in enumerate(env_ids.tolist()):
        cam._sensor_prims[i].GetFocalLengthAttr().Set(base * float(s[j]))
    cam._update_intrinsic_matrices(env_ids.tolist())
    env._vdr["focal"].setdefault(key, torch.ones(env.num_envs))[env_ids.cpu()] = s


def reset(env, env_ids) -> None:
    """Resample the visual randomization for the given envs (call from _reset_idx)."""
    if not enabled(env) or not hasattr(env, "_vdr"):
        return
    from pxr import Gf, Sdf

    cfg = env.cfg.student_obs
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return
    g = env._vdr_gen
    _cam_pose_jitter(env, env.student_camera, cfg.camera_pos, cfg.camera_quat_wxyz, env_ids, cfg)
    _focal_jitter(env, env.student_camera, "main", env_ids, cfg)
    if getattr(env, "student_camera_view2", None) is not None:
        _cam_pose_jitter(env, env.student_camera_view2, cfg.view2_pos, cfg.view2_quat_wxyz, env_ids, cfg)
        _focal_jitter(env, env.student_camera_view2, "view2", env_ids, cfg)

    lo_i, hi_i = (float(x) for x in getattr(cfg, "vdr_light_intensity", (2e4, 2e5)))
    with Sdf.ChangeBlock():
        for i in env_ids.tolist():
            for gname, (col, rough) in env._vdr["color"][i].items():
                col.Set(Gf.Vec3f(*[float(v) for v in _rand_color(g, gname)]))
                rough.Set(float(_u(g, 0.2, 1.0, 1)))
            if env._vdr["light"]:
                lt = env._vdr["light"][i]
                xyz = (float(_u(g, -0.8, 0.8, 1)), float(_u(g, -0.6, 1.0, 1)), float(_u(g, 1.2, 2.2, 1)))
                from pxr import UsdGeom
                UsdGeom.XformCommonAPI(lt).SetTranslate(Gf.Vec3d(*xyz))
                # log-uniform intensity, warm-to-cool tint
                lt.GetIntensityAttr().Set(float(math.exp(_u(g, math.log(lo_i), math.log(hi_i), 1))))
                t = float(_u(g, 0.0, 1.0, 1))
                lt.GetColorAttr().Set(Gf.Vec3f(1.0, 0.85 + 0.15 * t, 0.7 + 0.3 * t))
        dome = env._vdr.get("dome")
        if dome is not None:
            lo, hi = (float(x) for x in getattr(cfg, "vdr_dome_scale", (1.0, 1.0)))
            if (lo, hi) != (1.0, 1.0):
                dome.GetAttribute("inputs:intensity").Set(env._vdr["dome_base"] * float(_u(g, lo, hi, 1)))
