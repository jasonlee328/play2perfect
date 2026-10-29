"""Whole-hand tactile skin: taxels on every hand link (palm, thumb MC/PP/DP, finger MCP/PP/MP/DP), ray-cast against
the object and, optionally, the distractors.

Experimental and not sensor-faithful. Each link's collision surface is sampled every ``whole_hand_spacing_mm``; a
sample is a ray starting ``whole_hand_skin_mm`` outside the surface and pointing inward, read like the TacMap pads
(``tactile_utils.TactileSensor``): the depth the body reaches into the skin, kept if a back-cast from just inside the
hit also hits the body, then quantized with ``deform_quantize``. Each link is unrolled around its longest principal
axis (position along it x angle around it) into a ``whole_hand_grid`` x ``whole_hand_grid`` grid; a taxel reads the
max depth over its rays (rays are sparse, so a mean would dilute a contact). All bodies are read into one channel
(target and neighbours look the same, as for a real skin).

Output: (num_envs, len(HAND_LINKS) * grid^2) in [0, 1], links in ``HAND_LINKS`` order, row-major (along, around).
"""

from __future__ import annotations

import numpy as np
import torch

from .tactile_utils import CPD_EPS, CPD_MAX_DIST, _deform_quantize

# The palm (left_hand_C_MC) is merged into iiwa14_link_7 by the fixed joints link_7 -> ee -> mount -> palm: keep only
# its hand meshes. Every other link is its own body.
HAND_LINKS: tuple[str, ...] = (
    "left_hand_C_MC",
    "left_thumb_MC", "left_thumb_PP", "left_thumb_DP",
    "left_index_MCP_VL", "left_index_PP", "left_index_MP", "left_index_DP",
    "left_middle_MCP_VL", "left_middle_PP", "left_middle_MP", "left_middle_DP",
    "left_ring_MCP_VL", "left_ring_PP", "left_ring_MP", "left_ring_DP",
    "left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_PP", "left_pinky_MP", "left_pinky_DP",
)
_ALIASES = {"left_hand_C_MC": ("iiwa14_link_7", ("C_MC", "hand"))}


def hand_skin_obs_dim(tactile_cfg) -> int:
    return len(HAND_LINKS) * int(tactile_cfg.whole_hand_grid) ** 2


def prim_mesh(root_path: str, env_id: int = 0, keep: tuple[str, ...] | None = None):
    """Collision triangles of the rigid body at (or first under) ``root_path`` in its own frame, leaving out meshes
    under a nested rigid body; ``keep`` keeps only mesh paths containing one of the substrings."""
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path.replace("env_.*", f"env_{int(env_id)}"))
    if not root.IsValid():
        raise RuntimeError(f"hand skin: no prim {root_path}")
    it = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    body, meshes = None, []
    for p in it:
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            if body is None:
                body = p
            elif p != body:
                it.PruneChildren()
                continue
        if (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube)) and (
                p.HasAPI(UsdPhysics.CollisionAPI) or "collision" in str(p.GetPath()).lower()):
            meshes.append(p)
    body = body or root
    if keep is not None:
        meshes = [m for m in meshes if any(k in str(m.GetPath()) for k in keep)]
    if not meshes:
        raise RuntimeError(f"hand skin: no collision meshes under {root.GetPath()} (keep={keep})")
    cache = UsdGeom.XformCache()
    pts_all, tris_all, n = [], [], 0
    for m in meshes:
        if m.IsA(UsdGeom.Cube):
            h = 0.5 * float(UsdGeom.Cube(m).GetSizeAttr().Get())
            pts = h * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
            counts = np.full(6, 4)
            idx = np.array([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
        else:
            g = UsdGeom.Mesh(m)
            pts = np.asarray(g.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(g.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(g.GetFaceVertexIndicesAttr().Get())
        rel, _ = cache.ComputeRelativeTransform(m, body)
        pts = (np.c_[pts, np.ones(len(pts))] @ np.array(rel))[:, :3]
        tris, o = [], 0
        for c in counts:  # fan-triangulate
            tris += [(idx[o], idx[o + k], idx[o + k + 1]) for k in range(1, c - 1)]
            o += c
        pts_all.append(pts)
        tris_all.append(np.asarray(tris) + n)
        n += len(pts)
    return np.concatenate(pts_all), np.concatenate(tris_all)


class HandSkinSensor:
    """(num_envs, len(HAND_LINKS) * grid^2) whole-hand taxel readings in [0, 1]."""

    def __init__(self, env, tactile_cfg) -> None:
        import trimesh
        from isaaclab.utils.warp import convert_to_warp_mesh

        if len(env._object_urdf_paths) > 1:
            raise NotImplementedError("whole-hand tactile supports a single object asset")
        self.env, self.device = env, env.device
        g, sp = int(tactile_cfg.whole_hand_grid), float(tactile_cfg.whole_hand_spacing_mm)
        skin = float(tactile_cfg.whole_hand_skin_mm) * 1e-3
        rng = np.random.default_rng(0)  # fixed ray layout
        starts, dirs, body, taxel = [], [], [], []
        for k, name in enumerate(HAND_LINKS):
            bname, keep = _ALIASES.get(name, (name, None))
            pts, tris = prim_mesh(f"{env.robot.cfg.prim_path}/{bname}", keep=keep)
            tm = trimesh.Trimesh(pts * 1e3, tris, process=True)  # mm: trimesh tolerances suit mm
            n = max(4 * g * g, int(tm.area / sp ** 2))
            p, fi = trimesh.sample.sample_surface_even(tm, n, radius=0.9 * sp, seed=rng)
            nrm = tm.face_normals[fi]
            if ((p - tm.centroid) * nrm).sum(1).mean() < 0:  # inward-wound mesh
                nrm = -nrm
            p = p * 1e-3
            c = p.mean(0)
            _, _, vt = np.linalg.svd(p - c, full_matrices=False)
            q = (p - c) @ vt.T
            s, th = q[:, 0], np.arctan2(q[:, 2], q[:, 1])
            si = np.clip(((s - s.min()) / max(np.ptp(s), 1e-9) * g).astype(int), 0, g - 1)
            ti = np.clip(((th + np.pi) / (2 * np.pi) * g).astype(int), 0, g - 1)
            starts.append(p + nrm * skin); dirs.append(-nrm)
            body.append(np.full(len(p), k)); taxel.append(k * g * g + si * g + ti)
        f32 = dict(dtype=torch.float32, device=self.device)
        self.starts = torch.tensor(np.concatenate(starts), **f32)  # (R, 3) in link frames
        self.dirs = torch.tensor(np.concatenate(dirs), **f32)
        self.body = torch.tensor(np.concatenate(body), dtype=torch.long, device=self.device)
        self.taxel = torch.tensor(np.concatenate(taxel), dtype=torch.long, device=self.device)
        self.n_tax = len(HAND_LINKS) * g * g
        self.body_ids = [env.robot.find_bodies(_ALIASES.get(n, (n,))[0])[0][0] for n in HAND_LINKS]
        self.max_dist = skin + 3e-3  # a body can also sink a little into the collision surface
        self.targets = [env.object] + (list(getattr(env, "distractors", ())) if tactile_cfg.whole_hand_distractors
                                       else [])
        self.meshes = [convert_to_warp_mesh(*prim_mesh(o.cfg.prim_path), device=self.device) for o in self.targets]
        self.envs_per_chunk = max(1, int(tactile_cfg.max_rays_per_chunk) // len(self.starts))
        empty = int((torch.bincount(self.taxel, minlength=self.n_tax) == 0).sum())
        print(f"[hand skin] {len(HAND_LINKS)} links x {g}x{g} taxels ({empty} without rays), {len(self.starts)} "
              f"rays/env, skin {skin * 1e3:g} mm, cast against {len(self.targets)} bodies", flush=True)

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        from isaaclab.utils.math import matrix_from_quat
        from isaaclab.utils.warp import raycast_mesh

        env = self.env
        link = env.robot.data.body_state_w[:, self.body_ids, 0:7]  # (N, B, 7)
        R_wl = matrix_from_quat(link[..., 3:7])
        result = torch.zeros(env.num_envs, self.n_tax, device=self.device)
        for a in range(0, env.num_envs, self.envs_per_chunk):
            c = slice(a, a + self.envs_per_chunk)
            depth = None
            for obj, mesh in zip(self.targets, self.meshes):
                R_ow = matrix_from_quat(obj.data.root_quat_w[c]).transpose(-1, -2).unsqueeze(1)  # world -> body
                R = (R_ow @ R_wl[c])[:, self.body]                                                 # (n, R, 3, 3)
                t = (R_ow @ (link[c, :, 0:3] - obj.data.root_pos_w[c].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
                s = (R @ self.starts.unsqueeze(-1)).squeeze(-1) + t[:, self.body]
                d = (R @ self.dirs.unsqueeze(-1)).squeeze(-1)
                _, dist, _, _ = raycast_mesh(s, d, mesh, max_dist=self.max_dist, return_distance=True)
                back = s + d * (dist.clamp_min(0.0) + CPD_EPS).nan_to_num(0.0, 0.0, 0.0).unsqueeze(-1)
                _, dist2, _, _ = raycast_mesh(back, -d, mesh, max_dist=CPD_MAX_DIST, return_distance=True)
                dj = torch.where(torch.isfinite(dist) & torch.isfinite(dist2), dist, torch.zeros_like(dist))
                depth = dj if depth is None else torch.maximum(depth, dj)
            result[c] = torch.zeros(depth.shape[0], self.n_tax, device=self.device).scatter_reduce_(
                1, self.taxel.expand(depth.shape[0], -1), depth, "amax")
        return _deform_quantize(result) / 255.0


__all__ = ["HAND_LINKS", "HandSkinSensor", "hand_skin_obs_dim"]
