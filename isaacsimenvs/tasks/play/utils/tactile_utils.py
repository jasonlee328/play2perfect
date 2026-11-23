"""Sharpa TacMap fingerpad tactile observations, ray-cast against the object for every env.

Follows sharpa-tacmap's ``SharpaTacmap`` sensor model: rays start on the elastomer skin
and point along the inward normal (max 15 mm); the hit distance is the object's
penetration depth at that point, kept only if a back-cast from just inside the hit also
hits the object; depth is quantized with ``deform_quantize`` (0.5 mm -> 100/255).

Differences from the sensor, which renders a 240x240 image per pad:

- Coarse taxels: each pad is a ``resolution x resolution`` grid. A taxel covers a
  (240 / resolution)^2 block of the map, is sampled by ``samples_per_taxel^2`` rays and
  reads the mean depth over its in-pad rays. No Gaussian blur.
- play2perfect merges fixed joints, so the ``left_*_elastomer`` bodies don't exist; the
  elastomer frame is rebuilt from ``left_*_DP`` plus the URDF's fixed-joint rotation.
- The right-hand maps are used for the left hand (the pad is x-symmetric).
- Only the manipulated object is ray-cast, from its collision meshes, one mesh per object asset
  (multi-asset PreciseAssembly groups envs by ``_object_asset_index_per_env``); no scale noise.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")
MAP_SIZE: int = 240   # pixels per side of the TacMap maps
MAX_DIST: float = 0.015      # m, sensor max_distance
CPD_MAX_DIST: float = 0.5    # m, sensor cpd_max_dist
CPD_EPS: float = -1e-4       # m, sensor cpd_eps

REPO_ROOT = Path(__file__).resolve().parents[4]


def tactile_obs_dim(tactile_cfg) -> int:
    return len(FINGERS) * int(tactile_cfg.resolution) ** 2


def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx  # URDF convention


# left_*_elastomer_fix_joint: origin xyz 0, rpy (pi/2, -pi/2, 0) on all five fingers.
R_DP_ELASTOMER = _rpy_matrix(math.pi / 2, -math.pi / 2, 0.0)


def _object_mesh(env, env_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """env ``env_id``'s object collision meshes (all prims merged), in the object's rigid-body frame."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(env.object.cfg.prim_path.replace("env_.*", f"env_{int(env_id)}"))
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    body = next((p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)), root)
    # Mesh prims, plus Cube prims (URDF <box> collisions, e.g. the tight_insertion L-peg, become UsdGeom.Cube)
    meshes = [p for p in prims if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube)]
    coll = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI) or "collision" in str(p.GetPath()).lower()]
    meshes = coll or meshes
    if not meshes:
        raise RuntimeError(f"tactile: no meshes under {root.GetPath()}")
    cache = UsdGeom.XformCache()
    pts_all, tris_all, n = [], [], 0
    for m in meshes:
        if m.IsA(UsdGeom.Cube):
            h = 0.5 * float(UsdGeom.Cube(m).GetSizeAttr().Get())
            pts = h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
            counts = np.full(6, 4)  # quads, outward winding (vertex k = 4x + 2y + z bits)
            idx = np.array([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
        else:
            g = UsdGeom.Mesh(m)
            pts = np.asarray(g.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(g.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(g.GetFaceVertexIndicesAttr().Get())
        rel, _ = cache.ComputeRelativeTransform(m, body)
        pts = np.c_[pts, np.ones(len(pts))] @ np.array(rel)  # row-vector convention: [p 1] @ M
        tris, o = [], 0
        for c in counts:  # fan-triangulate
            tris += [(idx[o], idx[o + k], idx[o + k + 1]) for k in range(1, c - 1)]
            o += c
        pts_all.append(pts[:, :3])
        tris_all.append(np.asarray(tris) + n)
        n += len(pts)
    return np.concatenate(pts_all), np.concatenate(tris_all)


def _deform_quantize(deform: torch.Tensor) -> torch.Tensor:
    """sharpa-tacmap torch_jit_utils.deform_quantize: 0-0.5 mm -> 0-100, then 0.03 mm per level."""
    deform = deform * 1e3
    deform = torch.where(deform < 0.5, deform / 5e-3, (deform - 0.5) / 3e-2 + 100)
    return deform.clamp(0, 255).floor()


class TactileSensor:
    """(num_envs, 5 * res * res) taxel readings in [0, 1], fingers in ``FINGERS`` order."""

    def __init__(self, env, tactile_cfg) -> None:
        from isaaclab.utils.warp import convert_to_warp_mesh

        res, sub = int(tactile_cfg.resolution), int(tactile_cfg.samples_per_taxel)
        if MAP_SIZE % res:
            raise ValueError(f"tactile.resolution must divide {MAP_SIZE}, got {res}")
        cell = MAP_SIZE // res
        if not 1 <= sub <= cell:
            raise ValueError(f"tactile.samples_per_taxel must be in [1, {cell}] at resolution {res}, got {sub}")
        # One ray-cast mesh per object asset (multi-asset PreciseAssembly: env i holds asset
        # env._object_asset_index_per_env[i]); a single asset is the legacy path.
        n_assets = len(env._object_urdf_paths)
        asset_idx = getattr(env, "_object_asset_index_per_env", None)
        if n_assets > 1 and asset_idx is None:
            raise NotImplementedError("tactile obs with several object assets needs _object_asset_index_per_env")
        lo, hi = env.cfg.domain_randomization.object_scale_noise_multiplier_range
        if (lo, hi) != (1.0, 1.0):
            raise NotImplementedError("tactile obs does not support object_scale_noise_multiplier_range != (1, 1)")

        self.env, self.device, self.res = env, env.device, res
        self.meshes, self.asset_env_ids = [], []
        for k in range(n_assets):
            ids = (torch.arange(env.num_envs, device=self.device) if n_assets == 1
                   else (asset_idx == k).nonzero().reshape(-1))
            if ids.numel() == 0:
                continue
            pts, tris = _object_mesh(env, int(ids[0]))
            self.meshes.append(convert_to_warp_mesh(pts, tris, device=self.device))
            self.asset_env_ids.append(ids)
        self.mesh = self.meshes[0]

        self.body_ids = [env.robot.find_bodies(f"left_{f}_DP")[0][0] for f in FINGERS]
        map_dir = Path(tactile_cfg.map_dir)
        if not map_dir.is_absolute():
            map_dir = REPO_ROOT / map_dir
        # Sample points: sub x sub per taxel, at sub-cell centres of the 240x240 map.
        off = ((np.arange(sub) + 0.5) * cell / sub).astype(int)
        rows = (np.arange(res)[:, None] * cell + off[None]).reshape(-1)
        grid = (rows[:, None] * MAP_SIZE + rows[None, :]).reshape(-1)
        R = torch.tensor(R_DP_ELASTOMER, dtype=torch.float32, device=self.device)
        starts, dirs, finger, taxel = [], [], [], []
        for i, f in enumerate(FINGERS):
            tag = "TH" if f == "thumb" else "4F"
            p = np.load(map_dir / f"tactileSensor_map_{tag}_point.npy").reshape(-1, 3)
            nrm = np.load(map_dir / f"tactileSensor_map_{tag}_normal.npy").reshape(-1, 3)
            sel = grid[np.abs(p[grid]).sum(1) > 0]  # all-zero map entries are off the pad
            nrm = nrm[sel] / (np.linalg.norm(nrm[sel], axis=1, keepdims=True) + 1e-12)
            starts.append(torch.tensor(p[sel] * 1e-3, dtype=torch.float32, device=self.device) @ R.T)
            dirs.append(-torch.tensor(nrm, dtype=torch.float32, device=self.device) @ R.T)  # inward
            finger.append(torch.full((len(sel),), i, device=self.device))
            taxel.append(torch.tensor(i * res * res + (sel // MAP_SIZE // cell) * res + sel % MAP_SIZE // cell,
                                      device=self.device))
        self.starts = torch.cat(starts)   # (R, 3) in DP body frame
        self.dirs = torch.cat(dirs)       # (R, 3)
        self.finger = torch.cat(finger)   # (R,) finger index of each ray
        self.taxel = torch.cat(taxel)     # (R,) flat taxel index of each ray
        self.counts = torch.bincount(self.taxel, minlength=len(FINGERS) * res * res).clamp_min(1).float()
        # Rays per raycast_mesh call, to bound temporaries at large num_envs.
        self.envs_per_chunk = max(1, int(tactile_cfg.max_rays_per_chunk) // len(self.starts))
        print(f"[tactile] {len(FINGERS)} pads x {res}x{res} taxels, {len(self.starts)} rays/env, "
              f"object mesh {len(pts)} verts / {len(tris)} tris", flush=True)

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat
        from isaaclab.utils.warp import raycast_mesh

        env = self.env
        body = env.robot.data.body_state_w[:, self.body_ids, 0:7]  # (N, 5, 7) pos + wxyz
        obj_pos = env.object.data.root_pos_w                        # (N, 3)
        R_wo_T = matrix_from_quat(env.object.data.root_quat_w).transpose(-1, -2)  # world -> object
        # DP body -> object frame, per env and finger.
        R_od = R_wo_T.unsqueeze(1) @ matrix_from_quat(body[..., 3:7])                  # (N, 5, 3, 3)
        t_od = (R_wo_T.unsqueeze(1) @ (body[..., 0:3] - obj_pos.unsqueeze(1)).unsqueeze(-1)).squeeze(-1)

        result = torch.zeros(env.num_envs, len(self.counts), device=self.device)
        for mesh, ids in zip(self.meshes, self.asset_env_ids):
            for a in range(0, len(ids), self.envs_per_chunk):
                c = ids[a:a + self.envs_per_chunk]
                Rc, tc = R_od[c][:, self.finger], t_od[c][:, self.finger]
                s = (Rc @ self.starts.unsqueeze(-1)).squeeze(-1) + tc  # (n, R, 3) object frame
                d = (Rc @ self.dirs.unsqueeze(-1)).squeeze(-1)
                _, dist, _, _ = raycast_mesh(s, d, mesh, max_dist=MAX_DIST, return_distance=True)
                back = s + d * (dist.clamp_min(0.0) + CPD_EPS).nan_to_num(0.0, 0.0, 0.0).unsqueeze(-1)
                _, dist2, _, _ = raycast_mesh(back, -d, mesh, max_dist=CPD_MAX_DIST, return_distance=True)
                depth = torch.where(torch.isfinite(dist) & torch.isfinite(dist2), dist, torch.zeros_like(dist))
                taxels = torch.zeros(len(depth), len(self.counts), device=self.device).index_add_(1, self.taxel, depth)
                result[c] = taxels / self.counts
        return _deform_quantize(result) / 255.0


class TactileFlow:
    """A23 11-23: TactileLab-style tactile flow (arXiv 2611.07881) from the same rays as ``TactileSensor``.
    Per taxel: mean over its contact rays of the tangential velocity of the object surface relative to the pad
    (v_obj(p) - v_pad(p) with the component along the ray removed), in two fixed tangent axes per pad (DP frame),
    divided by ``scale`` (m/s) and clipped to +-clip. 0 where no ray of the taxel is in contact. Output
    (N, 5 * res * res * 2), taxel-major then (t1, t2). Rigid-body kinematics only (no gel shear)."""

    def __init__(self, sensor: "TactileSensor", scale: float = 0.05, clip: float = 3.0) -> None:
        self.s, self.scale, self.clip = sensor, float(scale), float(clip)
        dev = sensor.device
        F = len(sensor.body_ids)
        self.t1 = torch.zeros(F, 3, device=dev); self.t2 = torch.zeros(F, 3, device=dev)
        for f in range(F):
            n = sensor.dirs[sensor.finger == f].mean(0); n = n / n.norm()
            a = torch.tensor([1.0, 0.0, 0.0], device=dev)
            if abs(float(n @ a)) > 0.9:
                a = torch.tensor([0.0, 1.0, 0.0], device=dev)
            t1 = a - (a @ n) * n; t1 = t1 / t1.norm()
            self.t1[f], self.t2[f] = t1, torch.linalg.cross(n, t1)
        self.T = len(sensor.counts)

    def compute(self) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat
        from isaaclab.utils.warp import raycast_mesh
        s, env = self.s, self.s.env
        bs = env.robot.data.body_state_w[:, s.body_ids]
        R_wd = matrix_from_quat(bs[..., 3:7])
        os_ = env.object.data.root_state_w
        R_ow = matrix_from_quat(os_[:, 3:7]).transpose(-1, -2)
        R_od = R_ow.unsqueeze(1) @ R_wd
        t_od = (R_ow.unsqueeze(1) @ (bs[..., 0:3] - os_[:, None, 0:3]).unsqueeze(-1)).squeeze(-1)
        out = torch.zeros(env.num_envs, self.T, 2, device=s.device)
        for mesh, ids in zip(s.meshes, s.asset_env_ids):
            for a in range(0, len(ids), s.envs_per_chunk):
                c = ids[a:a + s.envs_per_chunk]
                Rc, tc = R_od[c][:, s.finger], t_od[c][:, s.finger]
                st = (Rc @ s.starts.unsqueeze(-1)).squeeze(-1) + tc
                d = (Rc @ s.dirs.unsqueeze(-1)).squeeze(-1)
                _, dist, _, _ = raycast_mesh(st, d, mesh, max_dist=MAX_DIST, return_distance=True)
                back = st + d * (dist.clamp_min(0.0) + CPD_EPS).nan_to_num(0.0, 0.0, 0.0).unsqueeze(-1)
                _, dist2, _, _ = raycast_mesh(back, -d, mesh, max_dist=CPD_MAX_DIST, return_distance=True)
                m = (torch.isfinite(dist) & torch.isfinite(dist2) & (dist > 0)).float()
                Rwd = R_wd[c][:, s.finger]
                p = bs[c][:, s.finger, 0:3] + (Rwd @ s.starts.unsqueeze(-1)).squeeze(-1)
                v_pad = bs[c][:, s.finger, 7:10] + torch.linalg.cross(bs[c][:, s.finger, 10:13], p - bs[c][:, s.finger, 0:3], dim=-1)
                v_obj = os_[c, None, 7:10] + torch.linalg.cross(os_[c, None, 10:13].expand_as(p), p - os_[c, None, 0:3], dim=-1)
                v_d = (Rwd.transpose(-1, -2) @ (v_obj - v_pad).unsqueeze(-1)).squeeze(-1)
                v_t = v_d - (v_d * s.dirs).sum(-1, keepdim=True) * s.dirs
                f2 = torch.stack([(v_t * self.t1[s.finger]).sum(-1), (v_t * self.t2[s.finger]).sum(-1)], -1) * m.unsqueeze(-1)
                cnt = torch.zeros(len(c), self.T, device=s.device).index_add_(1, s.taxel, m).clamp_min(1.0)
                out[c] = torch.zeros(len(c), self.T, 2, device=s.device).index_add_(1, s.taxel, f2) / cnt.unsqueeze(-1)
        return (out / self.scale).clamp(-self.clip, self.clip).reshape(env.num_envs, -1)


__all__ = ["FINGERS", "TactileFlow", "TactileSensor", "tactile_obs_dim"]
