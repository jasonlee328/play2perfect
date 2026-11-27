"""A13: in-hand repose of a key (ContactBench task 11, key only).

The key starts in a fingertip pinch (start-state bank). The goal is the key's start pose IN THE PALM FRAME turned by
an angle about the key's own long axis (object frame +x, through the COM), so moving the arm / wrist does not
help: the key has to be turned in the fingers. Goal k of the episode = flip angle ``key_flip_deg[k]`` (e.g.
[45, 90, 135, 180]: play2perfect-style waypoints along the flip); all goals reached = task success (no retract).

Every policy step the goal marker is rewritten as palm_pose * base_rel * Rx(angle_k), where base_rel = the key's
pose in the palm frame at the episode start, so the env's keypoint reward / near-goal / success machinery works
unchanged in the palm frame. Success tolerance: the keypoint max distance (reward.fixed_size cube, see
teacher/keyrepose/env_keyrepose.txt) maps 1.5 cm to ~15 deg of rotation or 1.5 cm of position.

Dropped (fail): the key's COM is more than ``key_drop_dist`` from every fingertip for ``key_drop_steps`` steps,
or the key's COM is within ``key_drop_table_m`` of the table top (``key_drop_below`` > 0: also more than that
below the palm).
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_from_angle_axis, quat_mul


class KeyRepose:
    def __init__(self, env):
        self.env = env
        c = env.cfg.precise_assembly
        N, dev = env.num_envs, env.device
        self.flip = torch.tensor([float(v) for v in c.key_flip_deg], device=dev) * (math.pi / 180.0)
        self.base_p = torch.zeros(N, 3, device=dev)
        self.base_q = torch.zeros(N, 4, device=dev); self.base_q[:, 0] = 1.0
        self.key0_p = torch.zeros(N, 3, device=dev)
        self.key0_q = torch.zeros(N, 4, device=dev); self.key0_q[:, 0] = 1.0
        self.ready = torch.zeros(N, dtype=torch.bool, device=dev)
        self.far_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self.rot_err = torch.zeros(N, device=dev)      # deg, to the current goal
        self.pos_err = torch.zeros(N, device=dev)      # m
        self.final_rot_err = torch.zeros(N, device=dev)  # deg, to the final (full-flip) goal
        self.min_final_rot_err = torch.full((N,), 180.0, device=dev)
        ax = torch.tensor([float(v) for v in getattr(c, "key_flip_axis", (1.0, 0.0, 0.0))], device=dev)
        self.x_axis = ax / ax.norm()
        pv = getattr(c, "key_pivot_point", None)
        self.pivot = torch.tensor([float(v) for v in pv], device=dev) if pv else None   # object frame (m)          # flip axis in the object frame (key: long axis x; Allen key: pinch axis z)
        self._step = 0
        # A13 cube (key_goal_mode=cube24): per-episode goal = start (palm frame) x one face-aligned rotation R_e
        self.mode = str(getattr(c, "key_goal_mode", "flip"))
        if self.mode == "cube24":
            self.cube_set = _cube_rotations(str(getattr(c, "key_cube_set", "all")), dev)   # (M, 4) wxyz, no identity
            self.goal_r = torch.zeros(N, 4, device=dev); self.goal_r[:, 0] = 1.0
            self.goal_id = torch.zeros(N, dtype=torch.long, device=dev)

    # ------------------------------------------------------------------ helpers
    def scale(self) -> float:
        c = self.env.cfg.precise_assembly
        n = int(c.key_flip_ramp_steps)
        if n <= 0:
            return 1.0
        t = (self._step + int(c.key_flip_ramp_offset_steps)) / n
        s0 = float(c.key_flip_ramp_start)
        return float(s0 + (1.0 - s0) * min(max(t, 0.0), 1.0))

    def _rot(self, env_ids, k):
        if self.mode == "cube24":
            return self.goal_r[env_ids]
        ang = self.flip[k.clamp(max=self.flip.numel() - 1)] * self.scale()
        return quat_from_angle_axis(ang, self.x_axis.expand(env_ids.numel(), 3))

    def _palm(self, env_ids=None):
        e = self.env
        s = e.robot.data.body_state_w[:, e._palm_body_id, :]
        if env_ids is not None:
            s = s[env_ids]
        return s[:, 0:3], s[:, 3:7]

    def goal_world(self, env_ids, k):
        """(n, 7) world goal pose for goal index k (n,)."""
        r = self._rot(env_ids, k)
        rd = self.ready[env_ids]
        pp, pq = self._palm(env_ids)
        p_ready = pp + quat_apply(pq, self.base_p[env_ids])
        q_ready = quat_mul(quat_mul(pq, self.base_q[env_ids]), r)
        p_new = self.key0_p[env_ids]
        k0 = self.key0_q[env_ids]
        pre = getattr(self.env, "_key_pre_flip", None)
        if pre is not None:
            k0 = quat_mul(k0, quat_from_angle_axis(-pre[env_ids], self.x_axis.expand(env_ids.numel(), 3)))
        q_new = quat_mul(k0, r)
        g = getattr(self, "pivot", None)
        if g is not None:                       # rotate about the grasp point instead of the object origin (COM)
            n = env_ids.numel(); gg = g.expand(n, 3)
            q_b = quat_mul(pq, self.base_q[env_ids])
            p_ready = p_ready + quat_apply(q_b, gg) - quat_apply(q_ready, gg)
            p_new = p_new + quat_apply(k0, gg) - quat_apply(q_new, gg)
        p = torch.where(rd.unsqueeze(-1), p_ready, p_new)
        q = torch.where(rd.unsqueeze(-1), q_ready, q_new)
        return torch.cat([p, q], -1)

    # ------------------------------------------------------------------ hooks
    def reset(self, env_ids):
        """Called from _write_goal_pose(is_first_goal=True), after the start state is written."""
        e = self.env
        self.key0_p[env_ids] = e.object.data.root_pos_w[env_ids]
        self.key0_q[env_ids] = e.object.data.root_quat_w[env_ids]
        self.ready[env_ids] = False
        self.far_steps[env_ids] = 0
        self.dropped[env_ids] = False
        self.min_final_rot_err[env_ids] = 180.0
        if self.mode == "cube24":
            j = torch.randint(0, self.cube_set.shape[0], (env_ids.numel(),), device=e.device)
            self.goal_id[env_ids] = j
            self.goal_r[env_ids] = self.cube_set[j]

    def update(self):
        """Start of _get_dones (fresh physics state): fix base_rel for new episodes, rewrite every goal, drop check."""
        e = self.env
        c = e.cfg.precise_assembly
        self._step += 1
        new = (~self.ready).nonzero(as_tuple=False).squeeze(-1)
        if new.numel() > 0:
            pp, pq = self._palm(new)
            self.base_p[new] = quat_apply_inverse(pq, self.key0_p[new] - pp)
            self.base_q[new] = quat_mul(quat_conjugate(pq), self.key0_q[new])
            pre = getattr(e, "_key_pre_flip", None)   # bank start already turned by pre: base = start * Rx(-pre)
            if pre is not None:
                self.base_q[new] = quat_mul(self.base_q[new], quat_from_angle_axis(-pre[new], self.x_axis.expand(new.numel(), 3)))
            self.ready[new] = True
            if not hasattr(self, "fresh"):
                self.fresh = torch.zeros_like(self.ready)
            self.fresh[new] = True
        ids = torch.arange(e.num_envs, device=e.device)
        k = (e._successes % e.env_max_goals.clamp_min(1)).long()
        g = self.goal_world(ids, k)
        e.goal_viz.write_root_pose_to_sim(g, env_ids=ids)
        e.goal_viz.write_root_velocity_to_sim(torch.zeros(e.num_envs, 6, device=e.device), env_ids=ids)

        # palm-frame errors (for logs and evals)
        kp, kq = e.object.data.root_pos_w, e.object.data.root_quat_w
        self.pos_err = (kp - g[:, 0:3]).norm(dim=-1)
        dq = quat_mul(quat_conjugate(g[:, 3:7]), kq)
        self.rot_err = torch.rad2deg(2.0 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))
        # swing-twist about the key's long axis (object x): twist = flip still missing, swing = the rest
        _va = (dq[:, 1:4] * self.x_axis).sum(-1)
        tw = torch.cat([dq[:, 0:1], _va.unsqueeze(-1) * self.x_axis], -1)
        tw = tw / tw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        self.twist_err = torch.rad2deg(2.0 * torch.acos(tw[:, 0].abs().clamp(max=1.0)))
        ts = 2.0 * torch.atan2(_va, dq[:, 0])
        self.twist_signed = torch.atan2(torch.sin(ts), torch.cos(ts))   # rad, key-frame x, current goal -> key
        sw = quat_mul(dq, quat_conjugate(tw))
        self.swing_err = torch.rad2deg(2.0 * torch.acos(sw[:, 0].abs().clamp(max=1.0)))
        last = torch.full_like(k, self.flip.numel() - 1)
        gf = self.goal_world(ids, last)
        dqf = quat_mul(quat_conjugate(gf[:, 3:7]), kq)
        self.final_rot_err = torch.rad2deg(2.0 * torch.acos(dqf[:, 0].abs().clamp(max=1.0)))
        self.min_final_rot_err = torch.minimum(self.min_final_rot_err, self.final_rot_err)
        # potential shaping toward the FINAL goal (waypoint switches don't jump): twist about the key x and swing (rest)
        twf = torch.cat([dqf[:, 0:1], ((dqf[:, 1:4] * self.x_axis).sum(-1)).unsqueeze(-1) * self.x_axis], -1)
        twf = twf / twf.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        ftw = 2.0 * torch.acos(twf[:, 0].abs().clamp(max=1.0))
        fsw = 2.0 * torch.acos(quat_mul(dqf, quat_conjugate(twf))[:, 0].abs().clamp(max=1.0))
        if self.mode == "cube24":
            ftw = torch.deg2rad(self.final_rot_err); fsw = torch.zeros_like(ftw)
        phi = -(ftw + float(c.key_shape_swing_w) * fsw + float(getattr(c, "key_shape_pos_w", 0.0)) * (kp - gf[:, 0:3]).norm(dim=-1))
        if not hasattr(self, "phi_prev"):
            self.phi_prev = phi.clone()
        fresh = self.fresh if hasattr(self, "fresh") else torch.ones_like(self.ready)
        self.phi_prev = torch.where(fresh, phi, self.phi_prev)
        self.shape = phi - self.phi_prev
        self.phi_prev = phi
        self.fresh = torch.zeros_like(self.ready)

        # dropped
        ft = e.robot.data.body_state_w[:, e._fingertip_body_ids, 0:3]
        dmin = (ft - kp.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
        pp, pq = self._palm()
        below = torch.zeros_like(dmin, dtype=torch.bool)
        if float(c.key_drop_below) > 0.0:
            below = (pp[:, 2] - kp[:, 2]) > float(c.key_drop_below)
        from .precise_assembly_env import TABLE_HALF_HEIGHT
        top = e._table_z_per_env + TABLE_HALF_HEIGHT + e.scene.env_origins[:, 2]
        below = below | (kp[:, 2] < top + float(c.key_drop_table_m))   # key on / near the table = dropped
        far = dmin > float(c.key_drop_dist)
        if self.mode == "cube24":                  # cube resting on the palm: dropped = > key_drop_dist from its start spot in the palm
            far = self.pos_err > float(c.key_drop_dist)
        self.far_steps = torch.where(far, self.far_steps + 1, torch.zeros_like(self.far_steps))
        self.dropped = (self.far_steps >= int(c.key_drop_steps)) | below
        self.fingertip_dmin = dmin

    def log(self, extras):
        extras["key_rot_err_deg"] = self.rot_err.mean()
        extras["key_twist_err_deg"] = self.twist_err.mean()
        extras["key_swing_err_deg"] = self.swing_err.mean()
        extras["key_pos_err_mm"] = self.pos_err.mean() * 1e3
        extras["key_final_rot_err_deg"] = self.final_rot_err.mean()
        extras["key_flip_scale"] = self.scale()
        if self.mode == "cube24":
            extras["cube_goal_angle_deg"] = (torch.rad2deg(2.0 * torch.acos(self.goal_r[:, 0].abs().clamp(max=1.0)))).mean()


def _cube_rotations(which: str, dev):
    """Face-aligned cube rotations (signed permutation matrices, det +1) as wxyz quats, identity excluded.
    which: 'all' = the 23 non-identity ones; 'quarter' = the 6 quarter turns about a face axis; 'face' = + the 3 half
    turns about face axes (9)."""
    import itertools
    import numpy as np
    from scipy.spatial.transform import Rotation
    Rs = []
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = sg[i]
            if np.linalg.det(M) > 0.5 and not np.allclose(M, np.eye(3)):
                Rs.append(M)
    out = []
    for M in Rs:
        r = Rotation.from_matrix(M)
        ang = np.degrees(np.linalg.norm(r.as_rotvec()))
        ax = r.as_rotvec() / max(np.linalg.norm(r.as_rotvec()), 1e-9)
        face = np.isclose(np.abs(ax).max(), 1.0)
        if which == "quarter" and not (face and np.isclose(ang, 90.0)):
            continue
        if which == "face" and not face:
            continue
        x, y, z, w = r.as_quat()
        out.append([w, x, y, z])
    assert len(out) == {"all": 23, "quarter": 6, "face": 9}[which], (which, len(out))
    return torch.tensor(out, dtype=torch.float32, device=dev)
