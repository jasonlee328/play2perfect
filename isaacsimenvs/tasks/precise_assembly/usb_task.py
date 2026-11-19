"""A12: ContactBench "Plug in USB" official verdict (force_bench appendix_tasks/plug_in_usb.py:58-61, 561-590),
computed every policy step for cfg.precise_assembly.usb_task != "off".

Frames: the receptive (hole) frame = the official slot frame (FoundationTouch tools/contactbench/usb_tasks.py): z up
the slot axis, plug tip at z = 10 mm when seated. Object = the stick, distractors 0 / 1 = the two jaws (prismatic).

Checks (official thresholds are the cfg defaults):
  held_in_hand_at_start   over usb_hold_steps after usb_settle_steps: pinch loaded every step (>= 2 robot contacts
                          on the stick: force on >= 2 links or >= 2 TacMap pads), stick moves < 0.5 mm in the link7
                          frame, nothing but the hand touches it
  finger_pinch_real       official: from the start to the release, the mean force on the thumb pad AND on the index pad
                          is >= usb_pinch_mean_n (3 N) (usb_pinch_mode=frac = the old share-of-steps rule)
  fully_inserted          plug tip and stick origin within usb_seat_tol (0.5 mm) of their CAD seat positions
  released_and_clear      no robot contact on the stick / jaws / slot (< 0.05 N, no TacMap contact) for
                          usb_clear_steps, link7 >= usb_clear_dist (40 mm) from where it was at the release
  no_motion_when_pushed   then the env pushes the stick along -z (slot frame) with usb_push_force (8 N) for
                          usb_push_steps (0.6 s): tip depth gained < 0.3 mm (official: push >= 0.75 x 8 N and gain
                          < 0.3 mm; ours applies exactly 8 N, so the force half always holds)
  stays_seated_after_release  final stick origin < 0.3 mm from its pose at the release
Success = all six at the end of the push. Fails (termination, gate mode): released_unseated (released and not
touched for usb_unseated_steps while not fully inserted), push_failed (the push ended and a check failed).
"""
from __future__ import annotations

import re

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

TIP_SEAT_Z = 0.010


def _glob(p: str) -> str:
    return re.sub(r"env_\d+", "env_*", p)


class UsbTask:
    def __init__(self, env):
        self.env = env
        self.c = env.cfg.precise_assembly
        d, N = env.device, env.num_envs
        z = lambda dt=torch.float32: torch.zeros(N, device=d, dtype=dt)
        self.step = z(torch.long)
        self.hold_ok = z(torch.bool); self.slip_hold = z()
        self.pre_steps = z(); self.pre_loaded = z(); self.pre_f_th = z(); self.pre_f_ix = z(); self.ins_steps = z(); self.ever_ins = z(torch.bool)
        self.released = z(torch.bool); self.nocontact = z(torch.long)
        self.rel_p = torch.zeros(N, 3, device=d); self.rel_palm = torch.zeros(N, 3, device=d)
        self.rel_tipz = z()
        self.pushing = z(torch.bool); self.push_count = z(torch.long); self.push_tip0 = z()
        self.success = z(torch.bool); self.just_rel_seated = z(torch.bool)
        self.ref_p = torch.zeros(N, 3, device=d)
        self.fail = {k: z(torch.bool) for k in ("released_unseated", "push_failed")}
        self.checks, self.meas = {}, {}
        self._views = False
        bi, _ = env.robot.find_bodies("iiwa14_link_7")
        self.i7 = int(bi[0])
        self.i_th = int(env.robot.find_bodies("left_thumb_DP")[0][0]); self.i_ix = int(env.robot.find_bodies("left_index_DP")[0][0])
        seat = env._pih_insert_pose_sequence[-1]
        self.seat_p = torch.tensor(seat[:3], device=d, dtype=torch.float32)
        self.seat_q = torch.tensor([seat[6], seat[3], seat[4], seat[5]], device=d, dtype=torch.float32)
        # plug tip in the stick frame: the seated tip sits at slot (0, 0, 10 mm)
        self.tip_local = quat_apply_inverse(self.seat_q[None], (torch.tensor([0.0, 0.0, TIP_SEAT_Z], device=d) - self.seat_p)[None])[0]

    def _make_views(self):
        env = self.env
        links = [_glob(p) for p in env.robot.root_physx_view.link_paths[0]]
        sim = env.object._physics_sim_view
        obj = _glob(env.object.root_physx_view.prim_paths[0])
        self.v_rob_stick = sim.create_rigid_contact_view(obj, filter_patterns=links)
        self.v_rob_scene = [sim.create_rigid_contact_view(_glob(b.root_physx_view.prim_paths[0]), filter_patterns=links)
                            for b in list(env.distractors) + [env.hole]]
        self._views = True

    def _rel_hole(self, pos_w, quat_w):
        env = self.env
        hp = env.hole_pos + env.scene.env_origins
        hq = env.hole_quat_wxyz
        return quat_apply_inverse(hq, pos_w - hp), quat_mul(quat_conjugate(hq), quat_w)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids):
        for t in (self.step, self.slip_hold, self.pre_steps, self.pre_loaded, self.pre_f_th, self.pre_f_ix, self.ins_steps, self.nocontact, self.push_count):
            t[env_ids] = 0
        for t in (self.released, self.pushing, self.success, self.just_rel_seated, self.ever_ins):
            t[env_ids] = False
        self.hold_ok[env_ids] = True
        for t in self.fail.values():
            t[env_ids] = False
        self._set_push()

    def _set_push(self):
        env = self.env
        f = torch.zeros(env.num_envs, 1, 3, device=env.device)
        if self.pushing.any():
            down = quat_apply(env.hole_quat_wxyz, torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3))
            f[:, 0] = down * float(self.c.usb_push_force) * self.pushing.float().unsqueeze(-1)
        env.object.set_external_force_and_torque(f, torch.zeros_like(f), is_global=True)

    # ------------------------------------------------------------------ step
    def update(self):
        env, c = self.env, self.c
        if not self._views:
            self._make_views()
        dt = float(env.physics_dt)
        N = env.num_envs
        F_rs = self.v_rob_stick.get_contact_force_matrix(dt).to(env.device).norm(dim=-1)        # (N, links)
        F_scene = sum(v.get_contact_force_matrix(dt).to(env.device).norm(dim=-1).sum(-1) for v in self.v_rob_scene)
        n_links = (F_rs > c.usb_pad_force).sum(-1)
        tac_on = None
        if getattr(env, "_tactile", None) is not None:
            T = env._tactile.compute().view(N, 5, -1)
            tac_on = T.amax(-1) > 0.0
            raw = (n_links >= 2) | (tac_on.sum(-1) >= 2) | ((tac_on.sum(-1) >= 1) & (n_links >= 1))
            touching = (F_rs.sum(-1) >= c.usb_clear_force) | tac_on.any(-1)
        else:
            raw = n_links >= 2
            touching = F_rs.sum(-1) >= c.usb_clear_force
        if not hasattr(self, "pinch_hist"):
            self.pinch_hist = torch.zeros(3, N, dtype=torch.bool, device=env.device)
        self.pinch_hist = torch.roll(self.pinch_hist, 1, 0); self.pinch_hist[0] = raw
        pinch = self.pinch_hist.any(0)

        b = env.robot.data.body_pos_w[:, self.i7]
        bq = env.robot.data.body_quat_w[:, self.i7]
        op, oq = env.object.data.root_pos_w, env.object.data.root_quat_w
        p7 = quat_apply_inverse(bq, op - b)
        pc, qc = self._rel_hole(op, oq)
        tip = pc + quat_apply(qc, self.tip_local.expand(N, 3))
        seat_err = torch.maximum((pc - self.seat_p).norm(dim=-1),
                                 (tip - torch.tensor([0.0, 0.0, TIP_SEAT_Z], device=env.device)).norm(dim=-1))
        inserted = seat_err < c.usb_seat_tol
        lat = tip[:, :2].norm(dim=-1)                                   # plug tip off the slot axis (m)
        ax_s = quat_apply(qc, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(N, 3))
        ax_0 = quat_apply(self.seat_q[None].expand(N, 4), torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(N, 3))
        tilt_deg = torch.rad2deg(torch.acos((ax_s * ax_0).sum(-1).clamp(-1.0, 1.0)))

        st = self.step - int(c.usb_settle_steps)
        first = st == 0
        self.ref_p[first] = p7[first]
        in_hold = (st >= 0) & (st < c.usb_hold_steps)
        slip = (p7 - self.ref_p).norm(dim=-1)
        self.slip_hold = torch.where(in_hold, torch.maximum(self.slip_hold, slip), self.slip_hold)
        others = F_scene * 0.0   # stick <-> slot / jaw contact is expected (the stick starts above the mouth)
        self.hold_ok &= ~in_hold | pinch
        pre = (st >= 0) & ~self.released
        self.pre_steps += pre.float(); self.pre_loaded += (pre & pinch).float()
        if str(getattr(c, "usb_pinch_window", "insert")) == "insert":
            # official (plug_in_usb.py:403-435, 553): pad force averaged over the INSERT phase only = after the
            # initial hold, until the stick first seats (the seated hold before the release is not counted)
            win = (st >= c.usb_hold_steps) & ~self.ever_ins & ~self.released
        else:   # "pre_release": every step from the settle to the release (old, stricter)
            win = pre
        self.ins_steps += win.float()
        self.pre_f_th += win.float() * F_rs[:, self.i_th]; self.pre_f_ix += win.float() * F_rs[:, self.i_ix]

        self.ever_ins |= inserted
        free = ~touching
        self.nocontact = torch.where(free, self.nocontact + 1, torch.zeros_like(self.nocontact))
        just_rel = (~self.released) & (st >= c.usb_hold_steps) & (self.nocontact >= c.usb_release_steps)
        self.rel_p[just_rel] = pc[just_rel]
        self.rel_tipz[just_rel] = tip[just_rel, 2]
        self.rel_palm[just_rel] = b[just_rel]
        self.released |= just_rel
        self.just_rel_seated = just_rel & inserted
        palm_from_rel = (b - self.rel_palm).norm(dim=-1) * self.released.float()
        scene_free = free & (F_scene < c.usb_clear_force)
        clear = self.released & (palm_from_rel >= c.usb_clear_dist) & (self.nocontact >= c.usb_clear_steps) & scene_free

        # env-applied push test, started once clear while inserted
        start = clear & inserted & ~self.pushing & ~self.success & ~self.fail["push_failed"]
        self.push_tip0[start] = tip[start, 2]
        self.pushing |= start
        self.push_count = torch.where(self.pushing, self.push_count + 1, self.push_count)
        gain = (self.push_tip0 - tip[:, 2]).clamp(min=0.0)
        if str(getattr(c, "usb_stay_mode", "depth")) == "pos":   # old A12 rule: 3D stick drift since the release
            stays = (pc - self.rel_p).norm(dim=-1) < c.usb_stay_tol
        else:   # official (plug_in_usb.py:578-580): |final depth - depth before release| < 0.3 mm, along the slot axis
            stays = (tip[:, 2] - self.rel_tipz).abs() < c.usb_stay_tol
        no_motion = gain < c.usb_push_gain_tol
        mean_th = self.pre_f_th / self.ins_steps.clamp(min=1.0); mean_ix = self.pre_f_ix / self.ins_steps.clamp(min=1.0)
        if str(getattr(c, "usb_pinch_mode", "mean_force")) == "frac":   # old A12 rule: pinch loaded on >= usb_pinch_frac of the steps
            pinch_real = self.pre_loaded >= c.usb_pinch_frac * self.pre_steps.clamp(min=1.0)
        else:   # official (plug_in_usb.py:553, 590-592): mean force on EACH pad during the insertion >= 3 N
            pinch_real = torch.minimum(mean_th, mean_ix) >= float(getattr(c, "usb_pinch_mean_n", 3.0))
        done_push = self.pushing & (self.push_count >= c.usb_push_steps)
        ok = done_push & inserted & no_motion & stays & pinch_real & self.hold_ok & scene_free
        self.success |= ok
        self.fail["push_failed"] |= done_push & ~ok
        self.pushing &= ~done_push
        self.fail["released_unseated"] |= self.released & ~inserted & ~self.pushing & ~self.success \
            & (self.nocontact >= c.usb_unseated_steps)
        self._set_push()
        self.step += 1

        self.checks = dict(held_in_hand_at_start=self.hold_ok & (st >= c.usb_hold_steps), finger_pinch_real=pinch_real,
                           fully_inserted=inserted, released_and_clear=clear, no_motion_when_pushed=no_motion & (self.push_count > 0),
                           stays_seated_after_release=stays & self.released)
        self.meas = dict(seat_err_mm=seat_err * 1e3, tip_z_mm=tip[:, 2] * 1e3, lat_mm=lat * 1e3, tilt_deg=tilt_deg, push_gain_mm=gain * 1e3,
                         palm_from_release=palm_from_rel, pinch=pinch.float(), slip_hold_mm=self.slip_hold * 1e3,
                         f_rob_stick=F_rs.sum(-1), f_rob_scene=F_scene,
                         pinch_share=self.pre_loaded / self.pre_steps.clamp(min=1.0), pad_mean_thumb_n=mean_th, pad_mean_index_n=mean_ix,
                         f_thumb=F_rs[:, self.i_th], f_index=F_rs[:, self.i_ix])
        self.inserted, self.clear = inserted, clear
