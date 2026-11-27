"""A28: ContactBench task 30 "Turn dial to target position", official verdict (force_bench appendix_tasks/
turn_dial_to_target.py:40-51, 146-153), computed every policy step for cfg.precise_assembly.dial_task != "off".

Knob = the object, jointed to the housing (joint 0: revolute, or slide_revolute for the push dial); standard / snap
pads = distractors 0-5 (joints 1-6, prismatic, displacement from the preloaded reset pose).

Checks (official thresholds are the cfg defaults):
  dial_at_target            |angle - target| <= dial_target_tol_deg (7.5 = half the 15 deg tick); snap dial:
                            <= dial_snap_tol_deg (3) with every pad back on the hub flats (mean pad displacement from
                            the seated preload within dial_snap_seat_mm, 0.15 mm)
  released_and_clear        hand contact on the knob < dial_release_force (0.05 N) for the rest of the episode and
                            the palm (iiwa14_link_7) >= dial_clear_dist (40 mm) from where it was at the release
  dial_stays_after_release  angle drift < dial_drift_deg (2) over dial_still_steps (60 policy steps = 1 s)
  held_down_off_zero (push) logged only: knob pressed >= 2.5 mm at the end (physics: a knob let go at an off-zero
                            angle stays down under the tooth / peg, at 0 it pops up)
Success (gate mode) = at target + released and clear held for dial_still_steps with the drift check. No extra fail
rules (realistic physics only): episodes end at success or at the timeout.
Shaping (rewards only): potential on the angle error (dial_turn_rew_scale x reduction of |error| in rad), push press
potential (dial_press_rew_scale x depth gain up to the gate, while the angle error > the tolerance), release-and-clear
(dial_clear_rew_scale x palm distance from the release point, capped at dial_clear_dist, while at target), success
bonus.
"""
from __future__ import annotations

import math

import torch


class DialTask:
    def __init__(self, env):
        self.env = env
        self.c = c = env.cfg.precise_assembly
        d, N = env.device, env.num_envs
        z = lambda dt=torch.float32: torch.zeros(N, device=d, dtype=dt)
        self.N, self.dev = N, d
        self.kind = str(c.dial_kind)
        self.push = self.kind == "push"
        self.snap = self.kind == "snap"
        self.n_pads = 6 if self.kind in ("standard", "snap") else 0
        tgt_deg = float(c.dial_target_deg)
        self.target = torch.full((N,), math.radians(tgt_deg), device=d)
        self.tol = math.radians(float(c.dial_snap_tol_deg if self.snap else c.dial_target_tol_deg))
        bi, _ = env.robot.find_bodies("iiwa14_link_7")
        self.i7 = int(bi[0])
        self.angle = z(); self.prev_raw = z(); self.err_prev = z(); self.depth_prev = z()
        self.step = z(torch.long)
        self.released = z(torch.bool); self.rel_palm = torch.zeros(N, 3, device=d)
        self.still = z(torch.long); self.still_ref = z()
        self.success = z(torch.bool); self.just_success = z(torch.bool)
        self.checks, self.meas = {}, {}
        self.fail = {}
        fr = tuple(float(v) for v in getattr(c, "dial_force_th_range", (0.0, 0.0)))
        self.force_on = fr[1] > 0.0
        self.f_lo, self.f_hi = fr
        self.f_th = torch.full((N,), fr[1], device=d) if self.force_on else None
        self.force_pen = z()
        self.tf_steps = int(getattr(c, "dial_turn_fail_steps", 0))
        self.tf_count = z(torch.long)
        self.tf_fail = z(torch.bool)

    def gate_lift(self):
        c = self.c
        if not self.push or float(c.dial_gate_lift_start_mm) <= 0.0:
            return 0.0
        n = float(getattr(self.env, "common_step_counter", 0)) + float(c.dial_gate_offset_steps)
        f = min(max(n / max(float(c.dial_gate_ramp_steps), 1.0), 0.0), 1.0)
        return float(c.dial_gate_lift_start_mm) * 1e-3 * (1.0 - f)

    def reset(self, env_ids):
        lift = self.gate_lift()
        if lift > 0.0:
            e = self.env
            k = int(self.c.dial_gate_idx)
            pose = list(e._pih_distractor_pose_rel[k]); pose[2] += lift
            w = e._pose_rel_hole_to_world(env_ids, tuple(pose))
            e.distractors[k].write_root_pose_to_sim(w, env_ids=env_ids)
        self.lift_now = lift
        if self.force_on:
            u = torch.rand(len(env_ids), device=self.dev) if not torch.is_tensor(env_ids) else \
                torch.rand(env_ids.numel(), device=self.dev)
            self.f_th[env_ids] = self.f_lo + u * (self.f_hi - self.f_lo)
        self.step[env_ids] = 0
        self.angle[env_ids] = 0.0
        self.prev_raw[env_ids] = 0.0
        self.err_prev[env_ids] = (self.target[env_ids]).abs()
        self.depth_prev[env_ids] = 0.0
        self.released[env_ids] = False
        self.still[env_ids] = 0
        self.success[env_ids] = False
        self.tf_count[env_ids] = 0
        self.tf_fail[env_ids] = False
        self.just_success[env_ids] = False

    def update(self):
        e, c = self.env, self.c
        e._update_joint_angles()
        raw = e.cb_joint_angle[0]
        fresh = self.step == 0
        dlt = torch.atan2(torch.sin(raw - self.prev_raw), torch.cos(raw - self.prev_raw))
        self.angle = torch.where(fresh, raw, self.angle + dlt)
        self.prev_raw = raw.clone()
        self.step += 1
        err = (self.angle - self.target).abs()
        at_t = err <= self.tol
        depth = -e.cb_joint_disp[0] if self.push else torch.zeros_like(err)
        if self.snap:
            pad = e.cb_joint_disp[1:1 + self.n_pads].mean(0)
            seated = pad.abs() <= float(c.dial_snap_seat_mm) * 1e-3
            at_t = at_t & seated
            self.meas["pad_mm"] = pad * 1e3
        f = getattr(e, "hand_force", None)
        if f is None:
            f = torch.zeros_like(err)
        contact = f >= float(c.dial_release_force)
        palm = e.robot.data.body_pos_w[:, self.i7]
        newly = ~self.released & ~contact & (self.step > 1)
        self.rel_palm = torch.where(newly.unsqueeze(-1), palm, self.rel_palm)
        self.released = (self.released | newly) & ~contact
        away = (palm - self.rel_palm).norm(dim=-1)
        clear = self.released & (away >= float(c.dial_clear_dist))
        ok = at_t & clear
        start = ok & (self.still == 0)
        self.still_ref = torch.where(start, self.angle, self.still_ref)
        drift = (self.angle - self.still_ref).abs()
        self.still = torch.where(ok & (drift < math.radians(float(c.dial_drift_deg))), self.still + 1,
                                 torch.zeros_like(self.still))
        succ = self.still >= int(c.dial_still_steps)
        self.just_success = succ & ~self.success
        self.success = self.success | succ
        self.err, self.depth, self.at_t, self.clear, self.away, self.contact = err, depth, at_t, clear, away, contact
        self.checks = dict(dial_at_target=at_t, released_and_clear=clear,
                           dial_stays_after_release=self.still > 0)
        if self.push:
            self.checks["held_down_off_zero"] = depth >= 0.0025
        self.meas.update(angle_deg=torch.rad2deg(self.angle), err_deg=torch.rad2deg(err), depth_mm=depth * 1e3,
                         hand_force=f, palm_from_release=away)
        if self.force_on and self.tf_steps > 0 and self.push:
            n = float(getattr(e, "common_step_counter", 0)) + float(c.dial_turn_fail_offset_steps)
            fr = min(max(n / max(float(c.dial_turn_fail_ramp_steps), 1.0), 0.0), 1.0)
            lim = self.f_th + (float(c.dial_turn_fail_start_N) - self.f_th) * (1.0 - fr)
            tp = (depth >= 0.0025) & (torch.rad2deg(self.angle).abs() > 10.0) & ~at_t & ~self.success
            self.tf_count = torch.where(tp & (f > lim), self.tf_count + 1, torch.zeros_like(self.tf_count))
            self.tf_fail = (self.tf_count >= self.tf_steps) & ~self.success
            self.fail["dial_force"] = self.tf_fail
            self.meas["turn_lim"] = lim
            self.tf_ramp = fr
        if self.force_on:
            self.meas["force_over"] = f > self.f_th
            self.meas["f_th"] = self.f_th
            self.force = f

    def reward(self):
        c = self.c
        fresh = self.step <= 1
        d_err = torch.where(fresh, torch.zeros_like(self.err), self.err_prev - self.err)
        self.err_prev = self.err.clone()
        r = float(c.dial_turn_rew_scale) * d_err
        if self.push and float(c.dial_press_rew_scale) > 0.0:
            dc = self.depth.clamp(0.0, 0.00325)
            d_dep = torch.where(fresh, torch.zeros_like(dc), dc - self.depth_prev)
            self.depth_prev = dc.clone()
            r = r + float(c.dial_press_rew_scale) * d_dep * (~self.at_t).float()
        if float(c.dial_clear_rew_scale) > 0.0:
            r = r + float(c.dial_clear_rew_scale) * (self.at_t & self.released).float() \
                * self.away.clamp(max=float(c.dial_clear_dist))
        r = r + float(c.dial_success_bonus) * self.just_success.float()
        if self.tf_steps > 0:
            r = r - float(c.dial_turn_fail_penalty) * self.tf_fail.float()
        if self.force_on and float(c.dial_force_pen_scale) > 0.0:
            self.force_pen = float(c.dial_force_pen_scale) * (self.force - self.f_th).clamp(min=0.0)
            r = r - self.force_pen
        return r
