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
or it is more than ``key_drop_below`` below the palm.
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
        self.x_axis = torch.tensor([1.0, 0.0, 0.0], device=dev)
        self._step = 0

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
        q_new = quat_mul(self.key0_q[env_ids], r)
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
            self.ready[new] = True
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
        last = torch.full_like(k, self.flip.numel() - 1)
        gf = self.goal_world(ids, last)
        dqf = quat_mul(quat_conjugate(gf[:, 3:7]), kq)
        self.final_rot_err = torch.rad2deg(2.0 * torch.acos(dqf[:, 0].abs().clamp(max=1.0)))
        self.min_final_rot_err = torch.minimum(self.min_final_rot_err, self.final_rot_err)

        # dropped
        ft = e.robot.data.body_state_w[:, e._fingertip_body_ids, 0:3]
        dmin = (ft - kp.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
        pp, pq = self._palm()
        below = (pp[:, 2] - kp[:, 2]) > float(c.key_drop_below)
        far = dmin > float(c.key_drop_dist)
        self.far_steps = torch.where(far, self.far_steps + 1, torch.zeros_like(self.far_steps))
        self.dropped = (self.far_steps >= int(c.key_drop_steps)) | below
        self.fingertip_dmin = dmin

    def log(self, extras):
        extras["key_rot_err_deg"] = self.rot_err.mean()
        extras["key_pos_err_mm"] = self.pos_err.mean() * 1e3
        extras["key_final_rot_err_deg"] = self.final_rot_err.mean()
        extras["key_flip_scale"] = self.scale()
