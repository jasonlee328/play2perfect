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
- Only the manipulated object is ray-cast, from its collision meshes, at its asset scale (no per-env
  scale noise). With several object assets (env._object_asset_index_per_env), their meshes are placed
  ASSET_SPACING apart in one warp mesh and each env's rays are shifted to its own asset.
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
ASSET_SPACING: float = 2.0    # m, spacing of the per-asset meshes in the merged ray-cast mesh (> CPD_MAX_DIST + size)

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
    root = stage.GetPrimAtPath(env.object.cfg.prim_path.replace("env_.*", f"env_{env_id}"))
    prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    body = next((p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)), root)
    meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
    coll = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI) or "collision" in str(p.GetPath()).lower()]
    meshes = coll or meshes
    if not meshes:
        raise RuntimeError(f"tactile: no meshes under {root.GetPath()}")
    cache = UsdGeom.XformCache()
    pts_all, tris_all, n = [], [], 0
    for m in meshes:
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
        n_assets = len(env._object_urdf_paths)
        if n_assets > 1 and len(getattr(env, "_pih_peg_shapes", ())) != n_assets:
            # only PreciseAssembly's peg_shapes is supported: same scale for all assets, index in
            # env._object_asset_index_per_env (the play task's procedural assets are scaled per env)
            raise NotImplementedError("tactile obs with several object assets needs precise_assembly.peg_shapes")
        lo, hi = env.cfg.domain_randomization.object_scale_noise_multiplier_range
        if (lo, hi) != (1.0, 1.0):
            raise NotImplementedError("tactile obs does not support object_scale_noise_multiplier_range != (1, 1)")

        self.env, self.device, self.res = env, env.device, res
        # Several assets (e.g. precise_assembly.peg_shapes): one warp mesh holding asset k's mesh shifted by
        # k * ASSET_SPACING along x; each env's rays are shifted by its asset's offset (in the object frame).
        self.ray_offset = None
        if n_assets == 1:
            pts, tris = _object_mesh(env)
        else:
            idx = env._object_asset_index_per_env
            P, T, n = [], [], 0
            for k in range(n_assets):
                e = (idx == k).nonzero()
                if len(e) == 0:
                    continue
                p, t = _object_mesh(env, int(e[0]))
                for j in e[1:4].flatten().tolist():  # check the spawn order matches the asset index
                    q, _ = _object_mesh(env, int(j))
                    if q.shape != p.shape or np.abs(q - p).max() > 1e-6:
                        raise RuntimeError(f"tactile: env {j} (asset {k}) has a different mesh than env {int(e[0])}")
                p = p + np.array([k * ASSET_SPACING, 0.0, 0.0])
                P.append(p)
                T.append(t + n)
                n += len(p)
            pts, tris = np.concatenate(P), np.concatenate(T)
            self.ray_offset = torch.zeros(env.num_envs, 1, 3, device=self.device)
            self.ray_offset[:, 0, 0] = idx.float() * ASSET_SPACING
        self.mesh = convert_to_warp_mesh(pts, tris, device=self.device)

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

        out = []
        for a in range(0, env.num_envs, self.envs_per_chunk):
            Rc, tc = R_od[a:a + self.envs_per_chunk][:, self.finger], t_od[a:a + self.envs_per_chunk][:, self.finger]
            s = (Rc @ self.starts.unsqueeze(-1)).squeeze(-1) + tc  # (n, R, 3) object frame
            if self.ray_offset is not None:
                s = s + self.ray_offset[a:a + self.envs_per_chunk]
            d = (Rc @ self.dirs.unsqueeze(-1)).squeeze(-1)
            _, dist, _, _ = raycast_mesh(s, d, self.mesh, max_dist=MAX_DIST, return_distance=True)
            back = s + d * (dist.clamp_min(0.0) + CPD_EPS).nan_to_num(0.0, 0.0, 0.0).unsqueeze(-1)
            _, dist2, _, _ = raycast_mesh(back, -d, self.mesh, max_dist=CPD_MAX_DIST, return_distance=True)
            depth = torch.where(torch.isfinite(dist) & torch.isfinite(dist2), dist, torch.zeros_like(dist))
            taxels = torch.zeros(len(depth), len(self.counts), device=self.device).index_add_(1, self.taxel, depth)
            out.append(taxels / self.counts)
        return _deform_quantize(torch.cat(out)) / 255.0


__all__ = ["FINGERS", "TactileSensor", "tactile_obs_dim"]
