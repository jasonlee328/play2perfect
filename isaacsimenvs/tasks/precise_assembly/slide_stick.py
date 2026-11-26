"""A25: let a pinched stick slide down through the fingers to a requested length, then re-grip.

The stick (ContactBench chopstick, long axis = object x, thin end at +x) starts pinched upright between the thumb and
index pads (start-state bank, arm frozen), with 2-6 cm sticking out below the pinch. Per episode a requested change
dL ~ U(slide_change_range) is drawn (clipped so the target protrusion stays in slide_target_range). The actor sees
[target, dL] (field slide_target) but never the stick pose: it has to loosen the grip, feel how far the stick slid
and re-grip.

protrusion = slide_x_end - (pinch point in the stick frame).x, pinch point = midpoint of the thumb / index pad centres.
Success: |protrusion - target| <= slide_tol and protrusion speed < slide_still_mps for slide_hold_steps steps.
Fail: overshoot (protrusion > target + slide_overshoot), dropped (stick axis more than slide_drop_lateral from the
pinch point, or the pinch above the stick's top end, or the stick bottom at the table), crushed (env crush limit),
timeout.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse


class SlideStick:
    def __init__(self, env):
        self.env = env
        c = env.cfg.precise_assembly
        N, dev = env.num_envs, env.device
        self.N, self.dev = N, dev
        self.target = torch.zeros(N, device=dev)
        self.change = torch.zeros(N, device=dev)
        self.start = torch.zeros(N, device=dev)
        self.prot = torch.zeros(N, device=dev)
        self.prot_prev = torch.zeros(N, device=dev)
        self.speed = torch.zeros(N, device=dev)        # m/s, EMA of |d protrusion / dt|
        self.lateral = torch.zeros(N, device=dev)
        self.hold = torch.zeros(N, dtype=torch.long, device=dev)
        self.fresh = torch.ones(N, dtype=torch.bool, device=dev)
        self.draw_on_update = torch.ones(N, dtype=torch.bool, device=dev)
        self.success = torch.zeros(N, dtype=torch.bool, device=dev)
        self.overshoot = torch.zeros(N, dtype=torch.bool, device=dev)
        self.dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self.in_tol = torch.zeros(N, dtype=torch.bool, device=dev)
        self.shape = torch.zeros(N, device=dev)
        self.phi_prev = torch.zeros(N, device=dev)
        self.max_prot = torch.zeros(N, device=dev)
        tac = env._tactile
        self.tip_ids = tac.body_ids[:2]                # thumb, index DP bodies
        self.pc = torch.stack([tac.starts[tac.finger == 0].mean(0), tac.starts[tac.finger == 1].mean(0)]).to(dev)
        self.dt = float(env.physics_dt) * int(env.cfg.decimation)
        self.x_end = float(c.slide_x_end)
        self.x_top = float(c.slide_x_top)
        self._step = 0

    # ------------------------------------------------------------------ geometry
    def pinch_point(self):
        rb = self.env.robot
        p = rb.data.body_pos_w[:, self.tip_ids]        # (N, 2, 3)
        q = rb.data.body_quat_w[:, self.tip_ids]
        pads = p + quat_apply(q.reshape(-1, 4), self.pc.unsqueeze(0).expand(self.N, 2, 3).reshape(-1, 3)).view(self.N, 2, 3)
        return pads.mean(1)

    def measure(self):
        o = self.env.object.data
        rel = quat_apply_inverse(o.root_quat_w, self.pinch_point() - o.root_pos_w)   # pinch in the stick frame
        return self.x_end - rel[:, 0], rel[:, 1:].norm(dim=-1), rel[:, 0]

    # ------------------------------------------------------------------ hooks
    def reset(self, env_ids):
        """After the start state is written (from _write_goal_pose(is_first_goal=True)): draw the request."""
        c = self.env.cfg.precise_assembly
        n = env_ids.numel()
        self.fresh[env_ids] = True
        self.hold[env_ids] = 0
        self.success[env_ids] = False
        self.overshoot[env_ids] = False
        self.dropped[env_ids] = False
        self.speed[env_ids] = 0.0
        self.max_prot[env_ids] = 0.0
        # draw the request from the bank's settled start protrusion (robot bodies are stale until the next physics
        # step); without it, draw on the first update
        p0 = getattr(self.env, "_slide_prot0", None)
        self.draw_on_update[env_ids] = p0 is None
        if p0 is not None:
            self._draw(env_ids, p0[env_ids])
        else:
            self.target[env_ids] = float(c.slide_target_range[0])
            self.change[env_ids] = 0.0

    def _draw(self, ids, p0):
        c = self.env.cfg.precise_assembly
        lo_t, hi_t = (float(v) for v in c.slide_target_range)
        lo_c, hi_c = (float(v) for v in c.slide_change_range)
        s0 = tuple(float(v) for v in c.slide_change_start_range)
        if s0[1] > 0.0 and int(c.slide_change_ramp_steps) > 0:   # curriculum on the requested change
            f = min(max((self._step + int(c.slide_change_ramp_offset_steps)) / int(c.slide_change_ramp_steps), 0.0), 1.0)
            lo_c, hi_c = s0[0] + f * (lo_c - s0[0]), s0[1] + f * (hi_c - s0[1])
        lo = torch.maximum(torch.full_like(p0, lo_c), lo_t - p0)
        hi = torch.minimum(torch.full_like(p0, hi_c), hi_t - p0)
        hi = torch.maximum(hi, lo)
        dl = lo + torch.rand_like(p0) * (hi - lo)
        self.start[ids] = p0
        self.change[ids] = dl
        self.target[ids] = p0 + dl

    def update(self):
        """Start of _get_dones (fresh physics state)."""
        e = self.env
        c = e.cfg.precise_assembly
        self._step += 1
        prot, lat, xp = self.measure()
        new = (self.fresh & self.draw_on_update).nonzero(as_tuple=False).squeeze(-1)
        if new.numel() > 0:
            self._draw(new, prot[new])
        self.prot_prev = torch.where(self.fresh, prot, self.prot_prev)
        v = (prot - self.prot_prev).abs() / self.dt
        a = float(c.slide_speed_ema)
        self.speed = torch.where(self.fresh, v, (1.0 - a) * self.speed + a * v)
        self.prot_prev = prot.clone()
        self.prot, self.lateral = prot, lat
        self.max_prot = torch.maximum(self.max_prot, prot)
        err = prot - self.target
        self.in_tol = err.abs() <= float(c.slide_tol)
        still = self.speed < float(c.slide_still_mps)
        self.hold = torch.where(self.in_tol & still, self.hold + 1, torch.zeros_like(self.hold))
        self.success = self.hold >= int(c.slide_hold_steps)
        self.overshoot = err > float(c.slide_overshoot)
        from .precise_assembly_env import TABLE_HALF_HEIGHT
        top = e._table_z_per_env + TABLE_HALF_HEIGHT + e.scene.env_origins[:, 2]
        o = e.object.data
        bottom = o.root_pos_w + quat_apply(o.root_quat_w, torch.tensor([self.x_end, 0.0, 0.0], device=self.dev).expand(self.N, 3))
        self.dropped = (lat > float(c.slide_drop_lateral)) | (xp < self.x_top) | (bottom[:, 2] < top + 0.005)
        # shaping potential on the target error (m -> cm units)
        phi = -err.abs() / 0.01
        P = float(getattr(c, "slide_peak_scale", 0.0))
        if P > 0.0:   # sharp peak at the target: stopping 5 mm short pays little under the linear term alone
            phi = phi + P * torch.exp(-err.abs() / float(c.slide_peak_width))
        self.phi_prev = torch.where(self.fresh, phi, self.phi_prev)
        self.shape = phi - self.phi_prev
        self.phi_prev = phi
        self.fresh[:] = False

    def obs(self):
        return torch.stack([self.target / 0.1, self.change / 0.1], -1)

    def hint(self):
        err = (self.prot - self.target) / 0.01
        return torch.stack([err.clamp(-5.0, 5.0), (self.speed / 0.01).clamp(max=5.0)], -1)

    def state(self):
        err = self.prot - self.target
        return torch.stack([self.prot / 0.1, err / 0.01, self.speed / 0.01, self.lateral / 0.01,
                            self.hold.float() / 30.0, self.max_prot / 0.1], -1)

    def reward(self, terms):
        c = self.env.cfg.precise_assembly
        r = float(c.slide_shape_scale) * self.shape
        r = r + float(c.slide_tol_rew) * self.in_tol.float()
        r = r + float(getattr(c, "slide_hold_rew", 0.0)) * (self.hold.float() / float(c.slide_hold_steps)).clamp(max=1.0)
        k = float(getattr(c, "slide_near_speed_pen", 0.0))
        if k > 0.0:   # speed^2 near the target: slowing down before the window pays (linear speed would cost the same at any speed)
            near = torch.exp(-(self.prot - self.target).abs() / float(c.slide_near_width))
            v = (self.speed / 0.01).clamp(max=50.0)   # cm/s
            r = r - k * near * v * v
        r = r + float(c.slide_success_bonus) * terms["max_successes"].float()
        r = r - float(c.slide_fail_penalty) * (terms["slide_overshoot"] | terms["slide_dropped"]).float()
        r = r - float(c.slide_timeout_penalty) * (terms["timeout"] & ~terms["max_successes"]).float()
        return r

    def log(self, extras):
        extras["slide/protrusion_mm"] = self.prot.mean() * 1e3
        extras["slide/abs_err_mm"] = (self.prot - self.target).abs().mean() * 1e3
        extras["slide/in_tol"] = self.in_tol.float().mean()
        extras["slide/speed_mm_s"] = self.speed.mean() * 1e3
        extras["slide/lateral_mm"] = self.lateral.mean() * 1e3
        extras["slide/change_mean_mm"] = self.change.mean() * 1e3
        extras["slide/slid_mm"] = (self.prot - self.start).mean() * 1e3


def apply_act_dr(env, env_ids):
    """Hidden per-env hand motor randomization (copied from A23 tools/a23/slip_probe.py --act_dr), resampled at every
    reset: per hand joint, stiffness and damping x logU(hand_gain_log_range), joint friction U(0, hand_joint_friction_max)
    (dynamic = 0.8 x static)."""
    dr = env.cfg.domain_randomization
    lo, hi = (float(v) for v in getattr(dr, "hand_gain_log_range", (1.0, 1.0)))
    fr = float(getattr(dr, "hand_joint_friction_max", 0.0))
    if (lo, hi) == (1.0, 1.0) and fr <= 0.0:
        return
    rb = env.robot
    dev = env.device
    if getattr(env, "_act_dr_base", None) is None:
        hand = [i for i, s in enumerate(rb.joint_names) if not s.startswith("iiwa")]
        env._act_dr_hand = hand
        env._act_dr_base = (rb.root_physx_view.get_dof_stiffnesses().to(dev)[:, hand].clone(),
                            rb.root_physx_view.get_dof_dampings().to(dev)[:, hand].clone())
        env._act_dr_gain = torch.ones(env.num_envs, len(hand), device=dev)
        env._act_dr_fric = torch.zeros(env.num_envs, len(hand), device=dev)
    hand = env._act_dr_hand
    n = env_ids.numel()
    g = torch.exp(torch.empty(n, len(hand), device=dev).uniform_(math.log(lo), math.log(hi)))
    k0, d0 = env._act_dr_base
    rb.write_joint_stiffness_to_sim(k0[env_ids] * g, joint_ids=hand, env_ids=env_ids)
    rb.write_joint_damping_to_sim(d0[env_ids] * g, joint_ids=hand, env_ids=env_ids)
    mu = torch.empty(n, len(hand), device=dev).uniform_(0.0, max(fr, 0.0))
    rb.write_joint_friction_coefficient_to_sim(mu, joint_dynamic_friction_coeff=0.8 * mu, joint_ids=hand, env_ids=env_ids)
    env._act_dr_gain[env_ids] = g
    env._act_dr_fric[env_ids] = mu
