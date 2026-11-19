"""A10 (11-18): ContactBench task 5 "Retrieve single card from deck" official checks
(force_bench appendix_tasks/retrieve_card_from_deck.py criteria(), lines 107-142), computed every policy step for
cfg.precise_assembly.card_task == "retrieve".

Scene (problem cb_card_retrieve, FoundationTouch tools/contactbench/card_tasks.py): 4 cards leaning 12 deg on the -x
posts of the kit holder; the object is the FRONT card (most +x, its +x face free), distractors 0..2 are the other three.
Frames: receptive (hole) frame = official kit frame, z = 0 at the table top. Our card body: local x = length (85 mm),
y = height (55), z = face normal (1.2).

Official verdict (thresholds = cfg defaults):
  single_card_on_table   picked card lying on a face (|face normal . z| > 0.95), centre z < 4 mm, centre x > 0
                         (outside the holder footprint side)                          retrieve_card_from_deck.py:114-118
  deck_remains_neat      each other card moved < 5 mm and tilted < 3 deg (face normals) from its start, centre x still
                         within the slot +- 10 mm                                     :119-131, OTHER_DISP_MAX / TILT_MAX
  released_and_clear     robot <-> cards + holder contact < card_clear_force N and link7 >= 50 mm from where it was at
                         the release (derive.py released_and_clear, min_move 0.05 m)
Success = all three hold for card_final_steps consecutive steps.
Fails (terminations): deck_disturbed (deck not neat for card_deck_fail_hold_steps steps in a row), broken (fragile:
robot press force on the picked card > card_press_max N for card_break_steps in a row while the card is still in the
slot).
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

from .card_task import _glob, HALF, SLOT_X

SLOT_Y = 0.04325


class RetrieveTask:
    def __init__(self, env):
        self.env = env
        self.c = env.cfg.precise_assembly
        d, N = env.device, env.num_envs
        z = lambda dt=torch.float32: torch.zeros(N, device=d, dtype=dt)
        self.step = z(torch.long)
        self.released = z(torch.bool); self.nocontact = z(torch.long); self.lifted = z(torch.bool)
        self.rel_palm = torch.zeros(N, 3, device=d)
        self.final_count = z(torch.long); self.deck_over = z(torch.long); self.break_count = z(torch.long)
        self.success = z(torch.bool)
        self.just_rel = z(torch.bool)
        self.rise_max = z()
        self.fail = {k: z(torch.bool) for k in ("deck_disturbed", "broken", "dropped")}
        self.checks, self.meas = {}, {}
        bi, _ = env.robot.find_bodies("iiwa14_link_7")
        self.i7 = int(bi[0])
        self.arm_ids = env.robot.find_bodies("iiwa14_link_.*")[0]
        self.deck_ref_p = None
        self.ref_p = torch.zeros(N, 3, device=d)
        self._views = False
        self.f_press = z(); self.f_press_ep_max = z()
        self.phi = torch.full((N,), float("nan"), device=d)

    def _make_views(self):
        env = self.env
        links = [_glob(p) for p in env.robot.root_physx_view.link_paths[0]]
        names = list(env.robot.root_physx_view.shared_metatype.link_names)
        self.tip_ids = [names.index(n) for n in ("left_thumb_DP", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_pinky_DP")]
        sim = env.object._physics_sim_view
        obj = _glob(env.object.root_physx_view.prim_paths[0])
        self.v_rob_card = sim.create_rigid_contact_view(obj, filter_patterns=links)
        self.v_rob_deck = [sim.create_rigid_contact_view(_glob(b.root_physx_view.prim_paths[0]), filter_patterns=links)
                           for b in env.distractors]
        try:
            self.v_rob_holder = [sim.create_rigid_contact_view(_glob(env.hole.root_physx_view.prim_paths[0]), filter_patterns=links)]
            self.v_rob_holder[0].filter_count
        except Exception as e:  # noqa: BLE001
            print(f"[card_retrieve] WARNING: no robot-holder contact view ({e}); released_and_clear ignores the holder")
            self.v_rob_holder = []
        others = [_glob(b.root_physx_view.prim_paths[0]) for b in env.distractors] + [_glob(env.hole.root_physx_view.prim_paths[0])]
        self.v_card_env = sim.create_rigid_contact_view(obj, filter_patterns=others)
        self._views = True

    def _rel_hole(self, pos_w, quat_w):
        env = self.env
        hp = env.hole_pos + env.scene.env_origins
        hq = env.hole_quat_wxyz
        return quat_apply_inverse(hq, pos_w - hp), quat_mul(quat_conjugate(hq), quat_w)

    @staticmethod
    def _normal(q):
        return quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3))

    def reset(self, env_ids):
        for t in (self.step, self.nocontact, self.final_count, self.deck_over, self.break_count, self.rise_max,
                  self.f_press_ep_max):
            t[env_ids] = 0
        for t in (self.released, self.success, self.lifted):
            t[env_ids] = False
        for t in self.fail.values():
            t[env_ids] = False
        self.phi[env_ids] = float("nan")
        if hasattr(self, "phi_reach"):
            self.phi_reach[env_ids] = float("nan")
        env = self.env
        if self.deck_ref_p is None:
            vals = list(getattr(self.c, "retr_fric_values", ()) or ())
            if vals:   # fragile variant: hidden card friction per env (fixed for the env's life; bucketed = few materials)
                idx = torch.arange(env.num_envs) % len(vals)
                mu = torch.tensor(vals, dtype=torch.float32)[idx]
                self.card_mu = mu.to(env.device)
                all_ids = torch.arange(env.num_envs)
                for b in [env.object] + list(env.distractors):
                    m = b.root_physx_view.get_material_properties()
                    m[:, :, 0] = mu[:, None]; m[:, :, 1] = mu[:, None]
                    b.root_physx_view.set_material_properties(m, all_ids)
            K = len(env.distractors)
            self.deck_ref_p = torch.zeros(K, env.num_envs, 3, device=env.device)
            self.deck_ref_n = torch.zeros(K, env.num_envs, 3, device=env.device)
        for k, pose in enumerate(env._pih_distractor_pose_rel):
            self.deck_ref_p[k, env_ids] = torch.tensor(pose[:3], device=env.device)
            q = torch.tensor([[pose[6], pose[3], pose[4], pose[5]]], device=env.device)
            self.deck_ref_n[k, env_ids] = self._normal(q)[0]
        self._fresh = env_ids if getattr(self, "_fresh", None) is None else torch.cat([self._fresh, env_ids])

    def _forces(self):
        N = self.env.num_envs
        F_rc = self.v_rob_card.get_contact_force_matrix(self.env.physics_dt).norm(dim=-1).reshape(N, -1)
        f_rob_card = F_rc.sum(-1)
        f_tips = F_rc[:, self.tip_ids]
        f_rob_deck = torch.zeros(N, device=self.env.device)
        for v in self.v_rob_deck:
            f_rob_deck = f_rob_deck + v.get_contact_force_matrix(self.env.physics_dt).norm(dim=-1).reshape(N, -1).sum(-1)
        f_rob_hold = torch.zeros(N, device=self.env.device)
        for v in self.v_rob_holder:
            f_rob_hold = f_rob_hold + v.get_contact_force_matrix(self.env.physics_dt).norm(dim=-1).reshape(N, -1).sum(-1)
        f_card_env = self.v_card_env.get_contact_force_matrix(self.env.physics_dt).norm(dim=-1).reshape(N, -1).sum(-1)
        return f_rob_card, f_tips, f_rob_deck, f_rob_hold, f_card_env

    def update(self):
        if not self._views:
            self._make_views()
        env, c = self.env, self.c
        d = env.device
        self.step += 1
        if getattr(self, "_fresh", None) is not None:
            # deck references = the poses after the reset write (official: start of the rollout)
            for k, b in enumerate(env.distractors):
                pk, qk = self._rel_hole(b.data.root_pos_w, b.data.root_quat_w)
                ids = self._fresh
                self.deck_ref_p[k, ids] = pk[ids]
                self.deck_ref_n[k, ids] = self._normal(qk)[ids]
            pc0, _ = self._rel_hole(env.object.data.root_pos_w, env.object.data.root_quat_w)
            self.ref_p[self._fresh] = pc0[self._fresh]
            self._fresh = None
        pc, qc = self._rel_hole(env.object.data.root_pos_w, env.object.data.root_quat_w)
        nc = self._normal(qc)
        f_rob_card, f_tips, f_rob_deck, f_rob_hold, f_card_env = self._forces()
        self.f_rob_deck, self.f_rob_card = f_rob_deck, f_rob_card
        # press force on the picked card (robot -> card) while it is still in the slot (centre below the post tops)
        in_slot_card = (pc[:, 2] < 0.045) & (pc[:, 0].abs() < SLOT_X + 0.01)
        self.f_press = f_rob_card * in_slot_card.float()
        self.f_press_ep_max = torch.maximum(self.f_press_ep_max, self.f_press)
        rise = pc[:, 2] - self.ref_p[:, 2]
        self.rise_max = torch.maximum(self.rise_max, rise)
        self.lifted |= rise > float(getattr(c, "retr_lift_m", 0.04))
        # release: after the lift, no robot <-> card contact for card_release_steps steps
        touch = f_rob_card > c.card_clear_force
        self.nocontact = torch.where(touch, torch.zeros_like(self.nocontact), self.nocontact + 1)
        palm = env.robot.data.body_pos_w[:, self.i7] - env.scene.env_origins
        just = (~self.released) & self.lifted & (self.nocontact >= c.card_release_steps)
        # hold rule (A10 11-19): the card must stay in the hand until it is set down at the place point; a release
        # higher than retr_release_max_z above the table or farther than retr_release_xy_tol from the place point
        # (a flick / drop) fails the episode as 'dropped'
        place_xy = torch.tensor(list(c.retr_place_xy), device=d)
        zr, xr = float(getattr(c, "retr_release_max_z", 1.0)), float(getattr(c, "retr_release_xy_tol", 10.0))
        rr = float(getattr(c, "retr_release_ramp_steps", 0.0))
        if rr > 0:   # curriculum: limits shrink from the *_start values to the final ones
            a_r = min(1.0, float(env.common_step_counter + getattr(c, "retr_release_offset_steps", 0.0)) / rr)
            zr = float(c.retr_release_max_z_start) + (zr - float(c.retr_release_max_z_start)) * a_r
            xr = float(c.retr_release_xy_tol_start) + (xr - float(c.retr_release_xy_tol_start)) * a_r
        self.rel_lim = (zr, xr)
        low = pc[:, 2] < zr
        near = (pc[:, :2] - place_xy).norm(dim=-1) < xr
        self.just_drop = just & ~(low & near)
        # graded set-down score at the release: 1 at the table on the place point, 0 at >= 10 cm up or >= 15 cm away
        hz = float(getattr(c, "retr_setdown_z", 0.10)); hx = float(getattr(c, "retr_setdown_xy", 0.15))
        self.setdown = just.float() * (1 - pc[:, 2] / hz).clamp(0, 1) * (1 - (pc[:, :2] - place_xy).norm(dim=-1) / hx).clamp(0, 1)
        if bool(getattr(c, "retr_hold_check", False)):
            self.fail["dropped"] = self.just_drop
            just = just & ~self.just_drop
        self.rel_palm[just] = palm[just]
        self.released |= just
        self.just_rel = just
        # deck
        neat = torch.ones(env.num_envs, dtype=torch.bool, device=d)
        k0, ramp = float(getattr(c, "retr_tol_start", 1.0)), float(getattr(c, "retr_tol_ramp_steps", 0.0))
        a_t = 1.0 if ramp <= 0 else min(1.0, float(env.common_step_counter + getattr(c, "retr_tol_offset_steps", 0)) / ramp)
        k1 = float(getattr(c, "retr_tol_end", 1.0))
        tol_k = k0 + (k1 - k0) * a_t
        self.tol_k = tol_k
        disp_max = torch.zeros(env.num_envs, device=d); tilt_max = torch.zeros(env.num_envs, device=d)
        for k, b in enumerate(env.distractors):
            pk, qk = self._rel_hole(b.data.root_pos_w, b.data.root_quat_w)
            disp = (pk - self.deck_ref_p[k]).norm(dim=-1)
            tilt = torch.rad2deg(torch.acos((self._normal(qk) * self.deck_ref_n[k]).sum(-1).abs().clamp(max=1.0)))
            in_slot = (pk[:, 0] > -SLOT_X - 0.01) & (pk[:, 0] < SLOT_X + 0.01) & (pk[:, 1].abs() < SLOT_Y)
            neat &= (disp < c.retr_other_disp_max * tol_k) & (tilt < c.retr_other_tilt_max * tol_k) & in_slot
            disp_max = torch.maximum(disp_max, disp); tilt_max = torch.maximum(tilt_max, tilt)
        self.deck_disp, self.deck_tilt = disp_max, tilt_max
        flat = nc[:, 2].abs() > 0.95
        on_table = pc[:, 2] < 0.004
        outside = pc[:, 0] > 0.0
        on_tab = flat & on_table & outside
        moved = (palm - self.rel_palm).norm(dim=-1)
        clear = self.released & ((f_rob_card + f_rob_deck + f_rob_hold) < c.card_clear_force) & (moved >= c.card_clear_dist)
        end_ok = on_tab & neat & clear
        self.final_count = torch.where(end_ok, self.final_count + 1, torch.zeros_like(self.final_count))
        self.success = self.final_count >= c.card_final_steps
        # fails
        self.deck_over = torch.where(~neat, self.deck_over + 1, torch.zeros_like(self.deck_over))
        self.fail["deck_disturbed"] = self.deck_over >= max(1, int(c.card_deck_fail_hold_steps))
        pmax = float(getattr(c, "retr_press_max", 0.0))
        if pmax > 0.0:
            self.break_count = torch.where(self.f_press > pmax, self.break_count + 1, torch.zeros_like(self.break_count))
            self.fail["broken"] = self.break_count >= int(c.card_break_steps)
        # shaping potential: card rise (capped at the lift height), then distance to the place point once lifted
        place = torch.tensor(list(c.retr_place_xy) + [0.0006], device=d)
        lift_h = float(getattr(c, "retr_lift_m", 0.04))
        phi_rise = rise.clamp(min=0.0, max=lift_h)
        dist = (pc - place).norm(dim=-1)
        carry = self.lifted.float()
        if bool(getattr(c, "retr_carry_held_only", False)):
            # transport progress counts only while the hand holds the card (or after a valid set-down): a flick or
            # drop loses it, so throwing the card toward the place point earns nothing
            carry = carry * (touch | self.released).float()
        phi = phi_rise + carry * (0.3 - dist).clamp(min=0.0)
        self.dphi = (phi - self.phi).nan_to_num(0.0)
        self.phi = phi
        n_tips = (f_tips > 0.05).sum(-1).float()
        if not hasattr(self, "_reach_ids"):
            self._reach_ids = env.robot.find_bodies(["left_thumb_DP", "left_index_DP"], preserve_order=True)[0]
            self.phi_reach = torch.full((env.num_envs,), float("nan"), device=d)
        up = quat_apply(qc, torch.tensor([0.0, 1.0, 0.0], device=d).expand(env.num_envs, 3))
        up = up * torch.sign(up[:, 2:3])
        top_w = env.hole_pos + env.scene.env_origins + quat_apply(env.hole_quat_wxyz, pc + up * 0.0275)
        dtip = (env.robot.data.body_pos_w[:, self._reach_ids] - top_w[:, None]).norm(dim=-1).max(-1).values
        phr = -dtip.clamp(max=0.3) * (~self.lifted).float()
        self.dphi_reach = (phr - self.phi_reach).nan_to_num(0.0) * (~self.lifted).float()
        self.phi_reach = phr
        self.reach_dist = dtip
        self.n_tips = n_tips
        self.checks = dict(single_card_on_table=on_tab, deck_remains_neat=neat, released_and_clear=clear)
        mu_m = self.card_mu if hasattr(self, "card_mu") else torch.zeros(env.num_envs, device=d)
        self.meas = dict(card_mu=mu_m, rise_mm=rise * 1e3, rise_max_mm=self.rise_max * 1e3, deck_disp_mm=disp_max * 1e3, deck_tilt_deg=tilt_max,
                         card_z_mm=pc[:, 2] * 1e3, card_x_mm=pc[:, 0] * 1e3, flat_cos=nc[:, 2].abs(), f_press=self.f_press,
                         f_press_ep_max=self.f_press_ep_max, f_rob_deck=f_rob_deck, f_card_env=f_card_env,
                         palm_from_release=moved * self.released.float(), reach_dist_mm=self.reach_dist * 1e3, lifted=self.lifted.float(), released=self.released.float())

    def reward(self, tr):
        """Per-step task reward (cfg retr_*)."""
        c = self.c
        r = c.card_success_bonus * tr["max_successes"].float()
        gate = self.checks["deck_remains_neat"].float() if bool(getattr(c, "retr_path_neat_gate", False)) else 1.0
        r = r + c.retr_path_rew_scale * self.dphi * gate
        r = r + float(getattr(c, "retr_reach_rew_scale", 0.0)) * self.dphi_reach
        r = r + c.retr_release_bonus * self.just_rel.float()
        r = r + float(getattr(c, "retr_setdown_bonus", 0.0)) * self.setdown * (~self.just_drop).float()
        r = r + c.retr_tip_rew * (self.n_tips.clamp(max=2.0) / 2.0) * (~self.released).float()
        r = r - c.card_rob_deck_pen * self.f_rob_deck.clamp(max=c.card_rob_deck_pen_cap)
        dm = self.deck_disp / c.retr_other_disp_max + self.deck_tilt / c.retr_other_tilt_max
        r = r - c.card_deck_motion_pen * dm.clamp(max=c.card_deck_motion_pen_cap)
        if c.card_clear_rew_scale > 0.0:
            r = r + c.card_clear_rew_scale * self.checks["single_card_on_table"].float() \
                * self.meas["palm_from_release"].clamp(max=c.card_clear_dist)
        fails = [v for k, v in tr.items() if k.startswith("card_")]
        if fails:
            r = r - c.card_fail_penalty * torch.stack(fails).any(0).float()
        return r
