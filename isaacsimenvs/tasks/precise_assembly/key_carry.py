"""A31 key carry: an already-pinched Allen key is carried to a socket; at a hidden time a hidden external push /
torque acts on the key (a physical wrench on the body, not a cue) and can make it slip or rotate in the pinch; then
the long-leg tip has to go into the socket (the problem's goals, tip position only).

Per episode (hidden, never observed): push start step U(push_start_range), duration U(push_steps_range) policy steps,
force = random unit direction x U(push_force_range) N, torque = random unit axis x U(push_torque_range) N m (both
fixed for the push, applied every physics substep while it lasts), with probability push_prob (else no push).

Bookkeeping (for logging / probes, not in any obs): the key's pose in the palm frame at the first step of the episode
(base) and now -> in-hand shift (mm) and rotation (deg); the shift right after the push ends.

Dropped (fail): key COM more than drop_dist from every fingertip body for drop_steps steps, or the key COM within
drop_table_m of the table top while the tip is not in the socket.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul


def _u(lo, hi, n, dev):
    return float(lo) + (float(hi) - float(lo)) * torch.rand(n, device=dev)


def _unit(n, dev):
    v = torch.randn(n, 3, device=dev)
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)


class KeyCarry:
    def __init__(self, env):
        self.env = env
        c = env.cfg.precise_assembly
        N, dev = env.num_envs, env.device
        self.c = c
        self.t0 = torch.zeros(N, dtype=torch.long, device=dev)
        self.dur = torch.zeros(N, dtype=torch.long, device=dev)
        self.F = torch.zeros(N, 3, device=dev)       # world frame
        self.T = torch.zeros(N, 3, device=dev)
        self.base_p = torch.zeros(N, 3, device=dev)
        self.base_q = torch.zeros(N, 4, device=dev); self.base_q[:, 0] = 1.0
        self.ready = torch.zeros(N, dtype=torch.bool, device=dev)
        self.shift_mm = torch.zeros(N, device=dev)
        self.rot_deg = torch.zeros(N, device=dev)
        self.post_shift_mm = torch.full((N,), float("nan"), device=dev)
        self.post_rot_deg = torch.full((N,), float("nan"), device=dev)
        self.far_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self.pushing = torch.zeros(N, dtype=torch.bool, device=dev)
        self.pushed = torch.zeros(N, dtype=torch.bool, device=dev)
        lnames = list(env.robot.root_physx_view.shared_metatype.link_names)
        self._palm = env._palm_body_id
        self._tips = [env.robot.find_bodies(n)[0][0] for n in
                      ("left_thumb_DP", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_pinky_DP") if n in lnames]

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids):
        c, dev, n = self.c, self.env.device, env_ids.numel()
        self.t0[env_ids] = torch.randint(int(c.kc_push_start_range[0]), int(c.kc_push_start_range[1]) + 1, (n,), device=dev)
        self.dur[env_ids] = torch.randint(int(c.kc_push_steps_range[0]), int(c.kc_push_steps_range[1]) + 1, (n,), device=dev)
        on = (torch.rand(n, device=dev) < float(c.kc_push_prob)).float().unsqueeze(-1)
        self.F[env_ids] = _unit(n, dev) * _u(*c.kc_push_force_range, n, dev).unsqueeze(-1) * on
        self.T[env_ids] = _unit(n, dev) * _u(*c.kc_push_torque_range, n, dev).unsqueeze(-1) * on
        self.ready[env_ids] = False
        self.far_steps[env_ids] = 0
        self.dropped[env_ids] = False
        self.pushing[env_ids] = False
        self.pushed[env_ids] = False
        self.shift_mm[env_ids] = 0.0
        self.rot_deg[env_ids] = 0.0
        self.post_shift_mm[env_ids] = float("nan")
        self.post_rot_deg[env_ids] = float("nan")

    # ------------------------------------------------------------------ helpers
    def rel_palm(self):
        rob, obj = self.env.robot, self.env.object
        pp, pq = rob.data.body_pos_w[:, self._palm], rob.data.body_quat_w[:, self._palm]
        p = quat_apply_inverse(pq, obj.data.root_pos_w - pp)
        q = quat_mul(quat_conjugate(pq), obj.data.root_quat_w)
        return p, q

    # ------------------------------------------------------------------ physics (every policy step, before stepping)
    def apply_push(self):
        env = self.env
        k = env.episode_length_buf
        self.pushing = (k >= self.t0) & (k < self.t0 + self.dur) & (self.F.norm(dim=-1) + self.T.norm(dim=-1) > 0)
        m = self.pushing.float().unsqueeze(-1)
        q = env.object.data.root_quat_w
        # body frame (A7: IsaacLab caches the orientation of global wrenches)
        f_b = quat_apply_inverse(q, self.F * m).unsqueeze(1)
        t_b = quat_apply_inverse(q, self.T * m).unsqueeze(1)
        env.object.set_external_force_and_torque(f_b, t_b, is_global=False)

    # ------------------------------------------------------------------ bookkeeping (in _get_dones)
    def update(self):
        env, c = self.env, self.c
        p, q = self.rel_palm()
        fresh = ~self.ready
        self.base_p = torch.where(fresh.unsqueeze(-1), p, self.base_p)
        self.base_q = torch.where(fresh.unsqueeze(-1), q, self.base_q)
        self.ready |= True
        self.shift_mm = (p - self.base_p).norm(dim=-1) * 1e3
        dq = quat_mul(quat_conjugate(self.base_q), q)
        self.rot_deg = torch.rad2deg(2 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))
        just_end = self.pushed & ~self.pushing & torch.isnan(self.post_shift_mm)
        self.post_shift_mm = torch.where(just_end, self.shift_mm, self.post_shift_mm)
        self.post_rot_deg = torch.where(just_end, self.rot_deg, self.post_rot_deg)
        self.pushed |= self.pushing
        # dropped
        com = env.object.data.root_pos_w + quat_apply(env.object.data.root_quat_w, self._com_off())
        tips = env.robot.data.body_pos_w[:, self._tips]
        dmin = (tips - com.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
        self.far_steps = torch.where(dmin > float(c.kc_drop_dist), self.far_steps + 1, torch.zeros_like(self.far_steps))
        top = env.scene.env_origins[:, 2] + env._table_z_per_env + 0.0
        try:
            from .precise_assembly_env import TABLE_HALF_HEIGHT
            top = top + TABLE_HALF_HEIGHT
        except Exception:
            pass
        on_table = com[:, 2] < top + float(c.kc_drop_table_m)
        self.dropped = (self.far_steps >= int(c.kc_drop_steps)) | on_table

    def _com_off(self):
        if not hasattr(self, "_com_b"):
            try:
                coms = self.env.object.root_physx_view.get_coms().to(self.env.device)
                self._com_b = coms[:, :3].clone()
            except Exception:
                self._com_b = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        return self._com_b

    def log(self, extras):
        extras["keycarry/pushing"] = self.pushing.float().mean()
        extras["keycarry/shift_mm"] = self.shift_mm.mean()
        extras["keycarry/rot_deg"] = self.rot_deg.mean()
        v = self.post_shift_mm[~torch.isnan(self.post_shift_mm)]
        if v.numel():
            extras["keycarry/post_push_shift_mm"] = v.median()
            extras["keycarry/post_push_rot_deg"] = self.post_rot_deg[~torch.isnan(self.post_rot_deg)].median()
