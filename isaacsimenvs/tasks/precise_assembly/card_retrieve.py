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

import math
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

from .card_task import _glob, HALF, SLOT_X, WALL_TOP_Z

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
        if bool(getattr(self.c, "retr_detent", False)):
            self.fail["scuffed"] = z(torch.bool)
        self._tip_body_ids = env.robot.find_bodies(["left_thumb_DP", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_pinky_DP"], preserve_order=True)[0]
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
        from ..play.utils.hand_skin import HAND_LINKS, _ALIASES
        self.skin_ids = [names.index(_ALIASES.get(n, (n,))[0]) for n in HAND_LINKS]   # whole-hand skin gel force
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

    def press_tactile(self, t):
        """Touch-only load cue (gel indentation model): each pad image x (1 + card_press_gain_per_N x that fingertip's
        contact force on the card), clipped to 1. The ray-cast depth of a rigid card does not grow with press force."""
        g = float(getattr(self.c, "card_press_gain_per_N", 0.0))
        ft = getattr(self, "f_tips", None)
        if g <= 0.0 or ft is None:
            return t
        N = t.shape[0]
        return (t.view(N, 5, -1) * (1.0 + g * ft)[..., None]).clamp(max=1.0).reshape(N, -1)

    def _detent(self, pc, qc, r, dw):
        """Fragile press detent (A10 11-19, FD4). The front card's bottom edge sits behind a spring pawl that holds it
        retr_detent_snap_m proud of the deck. Pressing it with >= F (hidden, logU retr_lock_force_range) for dw steps
        while the pawl is armed snaps the card past the pawl flat onto the deck ("seated"). Seated, the card can be slid
        up; it clears the pawl after retr_detent_groove_m of rise ("out", free for the rest of the episode). While seated
        and not yet out, pressing > r F (over-travel) or < retr_detent_hold_frac F (let go too early) makes the pawl
        throw the card back out to the engaged spot; the pawl re-arms only once the press drops below
        retr_detent_rearm_frac F. Engaged = the card is held at its engaged pose (pose-level, emulates the pawl; contacts
        with the fingers are real). Skidding loaded fingertips across the held card's face scuffs it: cumulative
        tangential tip slip (> 0.05 mm per step, tip force > 0.5 N) beyond retr_detent_scuff_mm = fail 'scuffed'.
        Touch cue = the real indentation change at the snap / pop (the card face moves snap_m along its normal)."""
        env, c, d = self.env, self.c, self.env.device
        F = self.lock_F
        snap = float(getattr(c, "retr_detent_snap_m", 0.0008)); groove = float(getattr(c, "retr_detent_groove_m", 0.004))
        lo = float(getattr(c, "retr_detent_hold_frac", 0.5)); rearm = float(getattr(c, "retr_detent_rearm_frac", 0.5))
        obj = env.object
        ids = getattr(self, "_scuff_fresh", None)
        if ids is not None:   # engaged pose = the rest pose (after the reset write) + snap along the free face normal
            self.seat_pose[ids, :3] = obj.data.root_pos_w[ids]; self.seat_pose[ids, 3:] = obj.data.root_quat_w[ids]
            n_w = quat_apply(obj.data.root_quat_w[ids], torch.tensor([0.0, 0.0, 1.0], device=d).expand(len(ids), 3))
            n_h = quat_apply_inverse(env.hole_quat_wxyz[ids], n_w)
            n_w = n_w * torch.sign(n_h[:, 0:1])
            self.out_dir[ids] = n_w
            self.lock_pose[ids] = self.seat_pose[ids]; self.lock_pose[ids, :3] += snap * n_w
            self._scuff_fresh = None
        p = self.f_press
        st = self.det_state
        rise = pc[:, 2] - self.ref_p[:, 2]
        eng = st == 0
        self.det_armed |= eng & (p < rearm * F)
        self.det_count = torch.where(eng & self.det_armed & (p >= F), self.det_count + 1, torch.zeros_like(self.det_count))
        snap_now = eng & (self.det_count >= max(1, int(round(dw))))
        seated = st == 1
        pop = seated & ((p > r * F) | (p < lo * F)) & (rise < groove)
        out = seated & ~pop & (rise >= groove)
        first_snap = snap_now & (self.n_snap == 0)
        self.n_snap += snap_now.float(); self.n_pop += pop.float()
        self.unlock_step[first_snap] = self.step[first_snap].float()
        first_out = out & self.locked
        st = torch.where(snap_now, torch.ones_like(st), st)
        st = torch.where(pop, torch.zeros_like(st), st)
        st = torch.where(out, torch.full_like(st, 2), st)
        self.det_armed &= ~pop
        self.det_state = st
        self.locked = st != 2
        self.just_unl = first_out          # unlock bonus once, when the card clears the pawl
        self.just_pop = pop
        self.lock_inwin = st == 1          # per-step reward while seated
        # scuff: loaded fingertips skidding over the card while the pawl holds it
        tips = env.robot.data.body_pos_w[:, self._tip_body_ids]                       # N x 5 x 3 (world)
        cq = obj.data.root_quat_w; cp = obj.data.root_pos_w
        loc = quat_apply_inverse(cq[:, None].expand(-1, 5, -1).reshape(-1, 4), (tips - cp[:, None]).reshape(-1, 3)).reshape(-1, 5, 3)
        if self.tip_prev is None:
            self.tip_prev = loc.clone()
        dl = (loc[..., :2] - self.tip_prev[..., :2]).norm(dim=-1)                     # tangential (card plane)
        loaded = self.f_tips > 0.5
        held = (eng | snap_now).float()
        fr = torch.zeros_like(self.scuff, dtype=torch.bool) if ids is None else torch.zeros_like(self.scuff, dtype=torch.bool).index_fill_(0, ids, True)
        self.scuff += ((dl > 5e-5) & loaded).float().mul(dl).sum(-1) * held * (~fr).float()
        self.tip_prev = loc
        lim = float(getattr(c, "retr_detent_scuff_mm", 3.0))
        lr = float(getattr(c, "retr_lock_ramp_steps", 0.0)); s0 = float(getattr(c, "retr_detent_scuff_mm_start", 0.0))
        if lr > 0 and s0 > 0:   # curriculum: scuff limit shrinks with the lock ramp
            a_l = min(1.0, float(env.common_step_counter + getattr(c, "retr_lock_offset_steps", 0.0)) / lr)
            lim = s0 + (lim - s0) * a_l
        self.scuff_lim = lim
        self.fail["scuffed"] = self.scuff > lim * 1e-3
        # card pose: engaged -> held at the engaged pose; just snapped -> placed flat on the deck (seat pose)
        hold_ids = (st == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(hold_ids):
            obj.write_root_pose_to_sim(self.lock_pose[hold_ids], hold_ids)
            obj.write_root_velocity_to_sim(torch.zeros(len(hold_ids), 6, device=d), hold_ids)
        sn = snap_now.nonzero(as_tuple=False).squeeze(-1)
        if len(sn):
            obj.write_root_pose_to_sim(self.seat_pose[sn], sn)
            obj.write_root_velocity_to_sim(torch.zeros(len(sn), 6, device=d), sn)
        # privileged shaping (reward only): press / F capped at 1 while engaged, 1 once seated / out
        php = torch.where(st == 0, (p / F).clamp(0.0, 1.0), torch.ones_like(p))
        if not hasattr(self, "phi_press"):
            self.phi_press = torch.full_like(php, float("nan"))
        self.dphi_press = (php - self.phi_press).nan_to_num(0.0)
        self.phi_press = php
        # card overload (fixed, physical): press > retr_press_max N for card_break_steps while in the slot = broken
        pmax = float(getattr(c, "retr_press_max", 10.0)) or 10.0
        self.break_count = torch.where(p > pmax, self.break_count + 1, torch.zeros_like(self.break_count))
        self.fail["broken"] = self.break_count >= int(c.card_break_steps)

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
        lo, hi = tuple(getattr(self.c, "retr_lock_force_range", (0.0, 0.0)))
        if hi > 0.0:   # press-force lock: hidden per-episode unlock force F, log-uniform in [lo, hi]
            if not hasattr(self, "lock_F"):
                N = self.env.num_envs; dv = self.env.device
                self.lock_F = torch.zeros(N, device=dv); self.locked = torch.zeros(N, dtype=torch.bool, device=dv)
                self.lock_count = torch.zeros(N, dtype=torch.long, device=dv)
                self.lock_pose = torch.zeros(N, 7, device=dv); self.unlock_step = torch.zeros(N, device=dv)
            u = torch.rand(len(env_ids), device=self.env.device)
            self.lock_F[env_ids] = float(lo) * (float(hi) / float(lo)) ** u
            self.locked[env_ids] = True; self.lock_count[env_ids] = 0; self.unlock_step[env_ids] = 0
            if bool(getattr(self.c, "retr_detent", False)):
                if not hasattr(self, "det_state"):
                    N = self.env.num_envs; dv = self.env.device
                    self.det_state = torch.zeros(N, dtype=torch.long, device=dv)   # 0 engaged, 1 seated (in groove), 2 out
                    self.det_armed = torch.ones(N, dtype=torch.bool, device=dv)
                    self.det_count = torch.zeros(N, dtype=torch.long, device=dv)
                    self.seat_pose = torch.zeros(N, 7, device=dv); self.out_dir = torch.zeros(N, 3, device=dv)
                    self.n_snap = torch.zeros(N, device=dv); self.n_pop = torch.zeros(N, device=dv)
                    self.scuff = torch.zeros(N, device=dv); self.tip_prev = None
                    self.just_pop = torch.zeros(N, dtype=torch.bool, device=dv)
                self.det_state[env_ids] = 0; self.det_armed[env_ids] = True; self.det_count[env_ids] = 0
                self.n_snap[env_ids] = 0; self.n_pop[env_ids] = 0; self.scuff[env_ids] = 0
                self._scuff_fresh = env_ids
                self.fail["scuffed"][env_ids] = False; self.break_count[env_ids] = 0
            if hasattr(self, "phi_press"):
                self.phi_press[env_ids] = float("nan")
            self._lock_fresh = env_ids
        if hasattr(self, "phi_reach"):
            self.phi_reach[env_ids] = float("nan")
        env = self.env
        if self.deck_ref_p is None:
            vals = list(getattr(self.c, "retr_fric_values", ()) or ())
            fr = tuple(getattr(self.c, "retr_fric_range", (0.0, 0.0)))
            if not vals and fr[1] > 0.0:   # DR: card friction U[lo, hi], bucketed into 8 materials (PhysX material limit)
                vals = [float(fr[0]) + (float(fr[1]) - float(fr[0])) * k / 7.0 for k in range(8)]
            ms = tuple(getattr(self.c, "retr_mass_scale_range", (1.0, 1.0)))
            if ms != (1.0, 1.0):   # DR: per-env card mass scale (all 4 cards), fixed for the env's life
                all_ids = torch.arange(env.num_envs)
                u = torch.rand(env.num_envs)
                if bool(getattr(self.c, "retr_mass_log", False)):   # log-uniform (press window scales with card weight)
                    lo, hi = math.log(float(ms[0])), math.log(float(ms[1]))
                    sc = torch.exp(lo + (hi - lo) * u)
                else:
                    sc = float(ms[0]) + (float(ms[1]) - float(ms[0])) * u
                # one scale per env for all cards (11-20); retr_mass_per_card=true = old per-card draws (D1 trained so)
                self.mass_scale = sc.to(env.device)
                for b in [env.object] + list(env.distractors):
                    m = b.root_physx_view.get_masses()
                    scb = (float(ms[0]) + (float(ms[1]) - float(ms[0])) * torch.rand(env.num_envs)) \
                        if bool(getattr(self.c, "retr_mass_per_card", False)) else sc
                    b.root_physx_view.set_masses(m * scb[:, None], all_ids)
            if vals:   # fragile variant: hidden card friction per env (fixed for the env's life; bucketed = few materials)
                idx = torch.randperm(env.num_envs) % len(vals) if fr[1] > 0.0 else torch.arange(env.num_envs) % len(vals)
                mu = torch.tensor(vals, dtype=torch.float32)[idx]
                self.card_mu = mu.to(env.device)
                all_ids = torch.arange(env.num_envs)
                jit = float(getattr(self.c, "retr_fric_card_jitter", 0.0))
                self.card_mu_each = []
                for b in [env.object] + list(env.distractors):
                    # per-card jitter (DR): each card gets its own friction around the env's value, so neighbouring
                    # cards can stick to each other more than to the next one (double-pull when pressed too hard)
                    mb = (mu + (2 * torch.rand(env.num_envs) - 1) * jit).clamp(min=0.02) if jit > 0 else mu
                    self.card_mu_each.append(mb.to(env.device))
                    m = b.root_physx_view.get_material_properties()
                    m[:, :, 0] = mb[:, None]; m[:, :, 1] = mb[:, None]
                    b.root_physx_view.set_material_properties(m, all_ids)
            fixed = list(getattr(self.c, "retr_fric_card_mu", ()) or ())
            if fixed:   # fixed friction per card, front -> back (front card, card 2, card 3, ...); < 0 = leave as is
                all_ids = torch.arange(env.num_envs)
                jf = float(getattr(self.c, "retr_fric_fixed_jitter", 0.0))
                self.card_mu_fixed = []
                for b, v in zip([env.object] + list(env.distractors)[::-1], fixed):
                    if float(v) < 0:
                        continue
                    # retr_fric_fixed_jitter j > 0: hidden per-episode-env jitter, each card's friction = v + U(-j, j)
                    # (pair friction = average, so mu12 - mu23 = (mu1 - mu3) / 2 moves the too-hard edge per env)
                    mv = (float(v) + (2 * torch.rand(env.num_envs) - 1) * jf).clamp(min=0.02) if jf > 0 \
                        else torch.full((env.num_envs,), float(v))
                    self.card_mu_fixed.append(mv.to(env.device))
                    m = b.root_physx_view.get_material_properties()
                    m[:, :, 0] = mv[:, None]; m[:, :, 1] = mv[:, None]
                    b.root_physx_view.set_material_properties(m, all_ids)
            K = len(env.distractors)
            self.deck_ref_p = torch.zeros(K, env.num_envs, 3, device=env.device)
            self.deck_ref_n = torch.zeros(K, env.num_envs, 3, device=env.device)
        for k, pose in enumerate(env._pih_distractor_pose_rel):
            self.deck_ref_p[k, env_ids] = torch.tensor(pose[:3], device=env.device)
            q = torch.tensor([[pose[6], pose[3], pose[4], pose[5]]], device=env.device)
            self.deck_ref_n[k, env_ids] = self._normal(q)[0]
        self._fresh = env_ids if getattr(self, "_fresh", None) is None else torch.cat([self._fresh, env_ids])
        self._rise_fresh = env_ids if getattr(self, "_rise_fresh", None) is None else torch.cat([self._rise_fresh, env_ids])

    def _forces(self):
        N = self.env.num_envs
        F_rc = self.v_rob_card.get_contact_force_matrix(self.env.physics_dt).norm(dim=-1).reshape(N, -1)
        f_rob_card = F_rc.sum(-1)
        f_tips = F_rc[:, self.tip_ids]
        f_rob_deck = torch.zeros(N, device=self.env.device)
        F_link = F_rc.clone()
        for v in self.v_rob_deck:
            F_rd = v.get_contact_force_matrix(self.env.physics_dt).norm(dim=-1).reshape(N, -1)
            f_rob_deck = f_rob_deck + F_rd.sum(-1)
            F_link = F_link + F_rd
        self.env._tactile_link_force = F_link[:, self.skin_ids]   # per skin link, card + deck contact force (N)
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
            if float(getattr(c, "retr_actor_freeze_mm", 0.0)) > 0:   # actor sees card 1 at its start pose until it clears
                if not hasattr(self, "frz_pos"):
                    self.frz_pos = torch.zeros(env.num_envs, 3, device=d); self.frz_rot = torch.zeros(env.num_envs, 4, device=d)
                    self.frz_on = torch.zeros(env.num_envs, dtype=torch.bool, device=d)
                ids = self._fresh
                self.frz_pos[ids] = env.object.data.root_pos_w[ids] - env.scene.env_origins[ids]
                self.frz_rot[ids] = env.object.data.root_quat_w[ids]
                self.frz_on[ids] = True
            self._fresh = None
        pc, qc = self._rel_hole(env.object.data.root_pos_w, env.object.data.root_quat_w)
        nc = self._normal(qc)
        if hasattr(self, "frz_on"):   # unfreeze for good once the card centre is retr_actor_freeze_mm above its start
            self.frz_on &= (pc[:, 2] - self.ref_p[:, 2]) < float(c.retr_actor_freeze_mm) * 1e-3
        f_rob_card, f_tips, f_rob_deck, f_rob_hold, f_card_env = self._forces()
        self.f_tips = f_tips
        env._tactile_tip_force = f_tips   # gel-like tactile force reading (tactile.gel_force_k)
        self.f_rob_deck, self.f_rob_card = f_rob_deck, f_rob_card
        # press force on the picked card (robot -> card) while it is still in the slot (centre below the post tops)
        in_slot_card = (pc[:, 2] < 0.045) & (pc[:, 0].abs() < SLOT_X + 0.01)
        self.f_press = f_rob_card * in_slot_card.float()
        self.f_press_ep_max = torch.maximum(self.f_press_ep_max, self.f_press)
        npmax = float(getattr(c, "retr_nonpad_max", 0.0))
        if npmax > 0.0:   # user 11-20: press with the fingertips only. While card 1 is in the holder, contact force on
            # it from any other hand / arm link (palm, finger segments) > npmax for retr_nonpad_steps steps = fail
            if "nonpad" not in self.fail:
                self.fail["nonpad"] = torch.zeros(env.num_envs, dtype=torch.bool, device=d)
                self.nonpad_count = torch.zeros(env.num_envs, dtype=torch.long, device=d)
            fresh = self.step <= 1
            self.nonpad_count[fresh] = 0
            self.f_nonpad = (f_rob_card - f_tips.sum(-1)).clamp(min=0.0) * in_slot_card.float()
            np0 = float(getattr(c, "retr_nonpad_max_start", 0.0)); npr = float(getattr(c, "retr_nonpad_ramp_steps", 0.0))
            if np0 > 0.0 and npr > 0.0:   # curriculum: limit tightens from retr_nonpad_max_start to retr_nonpad_max
                a_np = min(1.0, float(env.common_step_counter + getattr(c, "retr_nonpad_offset_steps", 0.0)) / npr)
                npmax = np0 + (npmax - np0) * a_np
            self.nonpad_lim = npmax
            self.nonpad_count = torch.where(self.f_nonpad > npmax, self.nonpad_count + 1, torch.zeros_like(self.nonpad_count))
            self.fail["nonpad"] = self.nonpad_count >= int(getattr(c, "retr_nonpad_steps", 3))
        if hasattr(self, "lock_F"):
            r = float(getattr(c, "retr_lock_ratio", 1.4)); dw = float(getattr(c, "retr_lock_dwell", 8))
            lr = float(getattr(c, "retr_lock_ramp_steps", 0.0))
            if lr > 0:   # curriculum: window ratio and dwell shrink from the *_start values to the final ones
                a_l = min(1.0, float(env.common_step_counter + getattr(c, "retr_lock_offset_steps", 0.0)) / lr)
                r = float(c.retr_lock_ratio_start) + (r - float(c.retr_lock_ratio_start)) * a_l
                dw = float(c.retr_lock_dwell_start) + (dw - float(c.retr_lock_dwell_start)) * a_l
            self.lock_lim = (r, dw)
            if getattr(self, "_lock_fresh", None) is not None:   # lock pose = the card pose after the reset write
                ids = self._lock_fresh
                self.lock_pose[ids, :3] = env.object.data.root_pos_w[ids]; self.lock_pose[ids, 3:] = env.object.data.root_quat_w[ids]
                self._lock_fresh = None
            if hasattr(self, "det_state"):
                self._detent(pc, qc, r, dw)
        if hasattr(self, "lock_F") and not hasattr(self, "det_state"):
            inwin = (self.f_press >= self.lock_F) & (self.f_press <= r * self.lock_F)
            self.lock_count = torch.where(self.locked & inwin, self.lock_count + 1, torch.zeros_like(self.lock_count))
            unl = self.locked & (self.lock_count >= max(1, int(round(dw))))
            self.unlock_step[unl] = self.step[unl].float()
            self.locked &= ~unl
            self.just_unl = unl
            self.lock_inwin = self.locked & inwin
            # press shaping potential (privileged F, reward only): press / F capped at 1 while locked, 1 once unlocked
            php = torch.where(self.locked, (self.f_press / self.lock_F).clamp(0.0, 1.0), torch.ones_like(self.f_press))
            if not hasattr(self, "phi_press"):
                self.phi_press = torch.full_like(php, float("nan"))
            self.dphi_press = (php - self.phi_press).nan_to_num(0.0)
            self.phi_press = php
            ids = self.locked.nonzero(as_tuple=False).squeeze(-1)
            if len(ids):   # still locked: the card is held at its start pose (pose-level stick, velocity zeroed)
                env.object.write_root_pose_to_sim(self.lock_pose[ids], ids)
                env.object.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=d), ids)
            # pressing harder than r x F while the card is in the slot breaks it (hidden per-episode limit)
            self.break_count = torch.where(self.f_press > r * self.lock_F, self.break_count + 1, torch.zeros_like(self.break_count))
            self.fail["broken"] = self.break_count >= int(c.card_break_steps)
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
        rises = [(self._rel_hole(b.data.root_pos_w, b.data.root_quat_w)[0][:, 2] - self.deck_ref_p[k][:, 2]) for k, b in enumerate(env.distractors)]
        self.deck_rise = torch.stack(rises, -1)   # N x K (m), per deck card
        if not hasattr(self, "deck_rise_ep") or self.deck_rise_ep.shape != self.deck_rise.shape:
            self.deck_rise_ep = torch.zeros_like(self.deck_rise)
        fr = getattr(self, "_rise_fresh", None)
        if fr is not None:
            self.deck_rise_ep[fr] = 0.0; self._rise_fresh = None
        self.deck_rise_ep = torch.maximum(self.deck_rise_ep, self.deck_rise)
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
        # retr_deck_out_mm > 0 (user 11-20): deck cards may rise partly during the episode as long as they fall back;
        # fail only if one comes fully out (centre risen > out mm). Success still needs the deck neat at the end.
        out_mm = float(getattr(c, "retr_deck_out_mm", 0.0))
        if bool(getattr(c, "retr_deck_inslot_only", False)):
            # user 11-20: every deck card may rise any amount while it stays in the holder. Out of the deck = its
            # lowest corner above the holder wall top, or off the slot sideways. Success needs all deck cards in.
            in_deck = torch.ones(env.num_envs, dtype=torch.bool, device=d)
            ex, ey_, ez_ = (torch.tensor(v, device=d, dtype=torch.float32).expand(env.num_envs, 3)
                            for v in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)))
            for k, b in enumerate(env.distractors):
                pk, qk = self._rel_hole(b.data.root_pos_w, b.data.root_quat_w)
                low = pk[:, 2] - sum(quat_apply(qk, e)[:, 2].abs() * h for e, h in zip((ex, ey_, ez_), HALF))
                in_deck &= (low < WALL_TOP_Z) & (pk[:, 0].abs() < SLOT_X + 0.01) & (pk[:, 1].abs() < SLOT_Y)
            self.deck_in = in_deck
            bad = ~in_deck
            neat = in_deck
            if bool(getattr(c, "retr_deck_table_only", False)):
                # user 11-20 (later): a deck card may leave the holder; fail only once one lies on the table
                # (lowest corner within 3 mm of the table top, outside the holder footprint). Success needs none there.
                on_tbl = torch.zeros(env.num_envs, dtype=torch.bool, device=d)
                for k, b in enumerate(env.distractors):
                    pk, qk = self._rel_hole(b.data.root_pos_w, b.data.root_quat_w)
                    low = pk[:, 2] - sum(quat_apply(qk, e)[:, 2].abs() * h for e, h in zip((ex, ey_, ez_), HALF))
                    outside_h = (pk[:, 0].abs() > SLOT_X + 0.01) | (pk[:, 1].abs() > SLOT_Y)
                    on_tbl |= (low < 0.003) & outside_h
                self.deck_on_table = on_tbl
                bad = on_tbl
                neat = ~on_tbl
            end_ok = on_tab & neat & clear
            self.final_count = torch.where(end_ok, self.final_count + 1, torch.zeros_like(self.final_count))
            self.success = self.final_count >= c.card_final_steps
        else:
            bad = (self.deck_rise.max(-1).values > out_mm * 1e-3) if out_mm > 0.0 else ~neat
        self.deck_over = torch.where(bad, self.deck_over + 1, torch.zeros_like(self.deck_over))
        self.fail["deck_disturbed"] = self.deck_over >= max(1, int(c.card_deck_fail_hold_steps))
        pmax = float(getattr(c, "retr_press_max", 0.0))
        if pmax > 0.0 and not hasattr(self, "lock_F"):
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
        if hasattr(self, "lock_F"):
            lock_m = dict(lock_F=self.lock_F, locked=self.locked.float(), unlock_step=self.unlock_step)
            if hasattr(self, "det_state"):
                lock_m.update(det_state=self.det_state.float(), n_snap=self.n_snap, n_pop=self.n_pop, scuff_mm=self.scuff * 1e3)
        else:
            lock_m = {}
        self.meas = dict(**lock_m, card_mu=mu_m, rise_mm=rise * 1e3, rise_max_mm=self.rise_max * 1e3, deck_disp_mm=disp_max * 1e3, deck_tilt_deg=tilt_max,
                         card_z_mm=pc[:, 2] * 1e3, card_x_mm=pc[:, 0] * 1e3, flat_cos=nc[:, 2].abs(), f_press=self.f_press,
                         f_press_ep_max=self.f_press_ep_max, deck_rise_max_mm=self.deck_rise_ep.max(-1).values * 1e3,
                         **({f"mu_card{k}": m for k, m in enumerate(self.card_mu_each)} if hasattr(self, "card_mu_each") else {}),
                         **{f"deck_rise{k}_mm": self.deck_rise_ep[:, k] * 1e3 for k in range(self.deck_rise.shape[1])}, f_rob_deck=f_rob_deck, f_card_env=f_card_env,
                         palm_from_release=moved * self.released.float(), reach_dist_mm=self.reach_dist * 1e3, lifted=self.lifted.float(), released=self.released.float())

    def reward(self, tr):
        """Per-step task reward (cfg retr_*)."""
        c = self.c
        r = c.card_success_bonus * tr["max_successes"].float()
        gate = self.checks["deck_remains_neat"].float() if bool(getattr(c, "retr_path_neat_gate", False)) else 1.0
        r = r + c.retr_path_rew_scale * self.dphi * gate
        r = r + float(getattr(c, "retr_reach_rew_scale", 0.0)) * self.dphi_reach
        r = r + c.retr_release_bonus * self.just_rel.float()
        if hasattr(self, "dphi_press"):   # fragile lock shaping
            r = r + float(getattr(c, "retr_lock_press_rew", 0.0)) * self.dphi_press
            r = r + float(getattr(c, "retr_lock_win_rew", 0.0)) * self.lock_inwin.float()
            r = r + float(getattr(c, "retr_lock_unlock_bonus", 0.0)) * self.just_unl.float()
        r = r + float(getattr(c, "retr_setdown_bonus", 0.0)) * self.setdown * (~self.just_drop).float()
        r = r + c.retr_tip_rew * (self.n_tips.clamp(max=2.0) / 2.0) * (~self.released).float()
        r = r - c.card_rob_deck_pen * self.f_rob_deck.clamp(max=c.card_rob_deck_pen_cap)
        if hasattr(self, "f_nonpad"):   # fingertips-only shaping: dense cost of non-fingertip force on card 1 in the holder
            r = r - float(getattr(c, "retr_nonpad_pen", 0.0)) * self.f_nonpad.clamp(max=10.0)
            pad = self.f_tips.sum(-1) * (self.f_nonpad < 0.5).float() * (~self.released).float()
            r = r + float(getattr(c, "retr_padpress_rew", 0.0)) * pad.clamp(max=5.0) / 5.0
        dm = self.deck_disp / c.retr_other_disp_max + self.deck_tilt / c.retr_other_tilt_max
        r = r - c.card_deck_motion_pen * dm.clamp(max=c.card_deck_motion_pen_cap)
        if c.card_clear_rew_scale > 0.0:
            r = r + c.card_clear_rew_scale * self.checks["single_card_on_table"].float() \
                * self.meas["palm_from_release"].clamp(max=c.card_clear_dist)
        fails = [v for k, v in tr.items() if k.startswith("card_")]
        if fails:
            r = r - c.card_fail_penalty * torch.stack(fails).any(0).float()
        return r
