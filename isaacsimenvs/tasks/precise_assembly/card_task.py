"""A10: ContactBench "Place card into holder" official checks (force_bench appendix_tasks/place_card_into_holder.py
verdict, docstring lines 46-63), computed every policy step for cfg.precise_assembly.card_task != "off".

Frames: the receptive (hole) frame is the official kit frame (FoundationTouch tools/contactbench/card_tasks.py).
Our card body: local x = length (85 mm), y = height (55), z = face normal (1.2). "Bottom" = local -y edge.
The object is the held card; distractors 0..N-1 are the pre-loaded deck (N-1 = front card).

Checks (official thresholds are the cfg defaults):
  held_in_hand_at_start   over the first card_hold_steps (0.6 s): card moves < 0.5 mm in the link7 frame (max over
                          its 8 corners), pinch loaded (thumb pad + >= 1 other fingertip link > card_pad_force N) on
                          every step, nothing else touches it (card <-> deck/holder < card_clear_force N)
  held_through_transport  from the end of the hold to the release: < 1 mm slip in the link7 frame, pinch loaded on
                          >= 98 % of the steps
  released_inside_slot    at the release (first of card_release_steps steps with no robot contact on the card) the
                          card's 4 bottom corners are inside the slot walls and below their tops (z 17 mm)
  card_in_holder          4 bottom corners inside the slot (x +-7, y +-43.25 mm), lowest point within 0.5 mm of the
                          floor (z 2), lean within 3 deg of the front card's, face gap < 0.6 mm, speed < 2 mm/s
  existing_cards_remain   every deck card moved < 3 mm and < 3 deg from its reset pose, bottom corners inside the
                          slot, lowest point within 0.5 mm of the floor
  released_and_clear      released, robot <-> card / deck / holder contact < 0.05 N, link7 >= 50 mm from its position
                          at the release
  robot_stopped           arm links (iiwa14_link_*) < 5 mm/s
Success = the held / release flags hold and the four end checks hold for card_final_steps consecutive steps (official
final window T_STILL 1.2 s = 72 steps at 60 Hz).
Fragile variant: card_break_force > 0: card <-> deck + holder contact force sum > card_break_force for
card_break_steps steps in a row = termination "broken".
"""
from __future__ import annotations

import re

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate

HALF = (0.0425, 0.0275, 0.0006)
SLOT_X, SLOT_Y, FLOOR_Z, WALL_TOP_Z = 0.007, 0.04325, 0.002, 0.017


def _glob(p: str) -> str:
    return re.sub(r"env_\d+", "env_*", p)


class CardTask:
    def __init__(self, env):
        self.env = env
        self.c = env.cfg.precise_assembly
        d = env.device
        N = env.num_envs
        h = torch.tensor(HALF, device=d)
        s = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], device=d, dtype=torch.float32)
        self.corners = s * h                                   # (8, 3)
        self.bottom = self.corners[self.corners[:, 1] < 0]     # (4, 3)
        self._views = False
        z = lambda dt=torch.float32: torch.zeros(N, device=d, dtype=dt)
        self.step = z(torch.long)
        self.hold_ok = z(torch.bool)
        self.slip_hold = z(); self.slip_tr = z()
        self.tr_steps = z(); self.tr_loaded = z()
        self.released = z(torch.bool); self.rel_inside = z(torch.bool); self.nocontact = z(torch.long)
        self.rel_palm = torch.zeros(N, 3, device=d)
        self.final_count = z(torch.long)
        self.break_count = z(torch.long)
        self.f_env = z(); self.f_env_ep_max = z(); self.f_rob = z()
        self.ref_p = torch.zeros(N, 3, device=d); self.ref_q = torch.zeros(N, 4, device=d); self.ref_q[:, 0] = 1
        self.tr_p = self.ref_p.clone(); self.tr_q = self.ref_q.clone()
        self.success = z(torch.bool)
        self.just_rel_inside = z(torch.bool)
        self.pos_hist = torch.zeros(12, N, 3, device=d)   # card centre (receptive frame), last 12 policy steps
        self.fail = {k: z(torch.bool) for k in ("dropped_at_start", "slipped", "released_outside", "deck_disturbed", "broken")}
        self.checks = {}
        bi, _ = env.robot.find_bodies("iiwa14_link_7")
        self.i7 = int(bi[0])
        self.arm_ids = env.robot.find_bodies("iiwa14_link_.*")[0]
        self.deck_ref_p = None

    # ------------------------------------------------------------------ views
    def _make_views(self):
        env = self.env
        links = [_glob(p) for p in env.robot.root_physx_view.link_paths[0]]
        names = list(env.robot.root_physx_view.shared_metatype.link_names)
        self.tip_ids = [names.index(n) for n in ("left_thumb_DP", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_pinky_DP")]
        sim = env.object._physics_sim_view
        obj = _glob(env.object.root_physx_view.prim_paths[0])
        self.v_rob_card = sim.create_rigid_contact_view(obj, filter_patterns=links)
        others = [_glob(b.root_physx_view.prim_paths[0]) for b in env.distractors] + [_glob(env.hole.root_physx_view.prim_paths[0])]
        self.v_card_env = sim.create_rigid_contact_view(obj, filter_patterns=others)
        self.v_rob_deck = [sim.create_rigid_contact_view(_glob(b.root_physx_view.prim_paths[0]), filter_patterns=links)
                           for b in env.distractors]
        self._views = True

    # ------------------------------------------------------------------ helpers
    def _rel_hole(self, pos_w, quat_w):
        env = self.env
        hp = env.hole_pos + env.scene.env_origins
        hq = env.hole_quat_wxyz
        return quat_apply_inverse(hq, pos_w - hp), quat_mul(quat_conjugate(hq), quat_w)

    def _pts(self, p, q, local):
        """(N, K, 3) points `local` (K, 3) of a body at (p, q)."""
        K = local.shape[0]
        return p.unsqueeze(1) + quat_apply(q.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4),
                                           local.unsqueeze(0).expand(p.shape[0], -1, -1).reshape(-1, 3)).reshape(-1, K, 3)

    def _in_slot(self, pts):
        """(N,) all points inside the slot walls / end walls."""
        return ((pts[..., 0].abs() <= SLOT_X) & (pts[..., 1].abs() <= SLOT_Y)).all(-1)

    @staticmethod
    def _lean_deg(q):
        up = quat_apply(q, torch.tensor([0.0, 1.0, 0.0], device=q.device).expand(q.shape[0], 3))
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1)))

    @staticmethod
    def _ang_deg(qa, qb):
        dq = quat_mul(quat_conjugate(qa), qb)
        return torch.rad2deg(2 * torch.acos(dq[:, 0].abs().clamp(max=1.0)))

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids):
        for t in (self.step, self.slip_hold, self.slip_tr, self.tr_steps, self.tr_loaded, self.nocontact,
                  self.final_count, self.break_count, self.f_env_ep_max):
            t[env_ids] = 0
        self.pos_hist[:, env_ids] = float('nan')
        for t in (self.released, self.rel_inside, self.success):
            t[env_ids] = False
        self.hold_ok[env_ids] = True
        for t in self.fail.values():
            t[env_ids] = False
        env = self.env
        if self.deck_ref_p is None:
            K = len(env.distractors)
            self.deck_ref_p = torch.zeros(K, env.num_envs, 3, device=env.device)
            self.deck_ref_q = torch.zeros(K, env.num_envs, 4, device=env.device)
        # deck reset poses (receptive frame) straight from the problem
        for k, pose in enumerate(env._pih_distractor_pose_rel):
            self.deck_ref_p[k, env_ids] = torch.tensor(pose[:3], device=env.device)
            self.deck_ref_q[k, env_ids] = torch.tensor([pose[6], pose[3], pose[4], pose[5]], device=env.device)
        rng = tuple(self.c.card_deck_lean_range_deg)
        if rng:
            self._relayout_deck(env_ids, rng)

    def _relayout_deck(self, env_ids, rng):
        """Fragile variant: hidden per-episode deck lean (deg, uniform in rng), the official closed-form layout
        (_card_place.py deck_layout) re-solved per env; goals stay at the problem's nominal lean (not observed)."""
        from isaaclab.utils.math import quat_from_matrix
        env, d = self.env, self.env.device
        n, K = env_ids.numel(), len(env.distractors)
        th = torch.deg2rad(torch.empty(n, device=d).uniform_(float(rng[0]), float(rng[1])))
        self.deck_lean = getattr(self, "deck_lean", torch.zeros(env.num_envs, device=d))
        self.deck_lean[env_ids] = torch.rad2deg(th)
        T, H, POST, FLOOR = 0.0012, 0.055, 0.037, 0.002
        z = FLOOR + 2e-5 + 0.5 * T * torch.sin(th) + 0.5 * H * torch.cos(th)
        x0 = -0.007 + (POST - z) * torch.tan(th) + (0.5 * T + 2e-5) / torch.cos(th)
        pitch = (T + 1e-5) / torch.cos(th)
        P = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=d)
        c, s_ = torch.cos(-th), torch.sin(-th)
        R = torch.zeros(n, 3, 3, device=d)
        R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s_, 1.0, -s_, c
        q = quat_from_matrix(R @ P)
        hp = env.hole_pos[env_ids] + env.scene.env_origins[env_ids]
        hq = env.hole_quat_wxyz[env_ids]
        ref = torch.tensor(self.c.strict_ref_local, device=d).expand(n, 3)
        for k, body in enumerate(env.distractors):
            p_rel = torch.stack([x0 + k * pitch, torch.zeros_like(z), z], -1)
            pw = hp + quat_apply(hq, p_rel)
            qw = quat_mul(hq, q)
            body.write_root_pose_to_sim(torch.cat([pw, qw], -1), env_ids=env_ids)
            body.write_root_velocity_to_sim(torch.zeros(n, 6, device=d), env_ids=env_ids)
            self.deck_ref_p[k, env_ids] = p_rel
            self.deck_ref_q[k, env_ids] = q
            if hasattr(env, "_distractor_reset_pos"):
                env._distractor_reset_pos[k, env_ids] = pw
                env._distractor_reset_ref[k, env_ids] = pw + quat_apply(qw, ref)

    # ------------------------------------------------------------------ step
    def update(self):
        env, c = self.env, self.c
        if not self._views:
            self._make_views()
        dt = float(env.physics_dt)
        F_rc = self.v_rob_card.get_contact_force_matrix(dt).to(env.device).norm(dim=-1)      # (N, links)
        F_ce = self.v_card_env.get_contact_force_matrix(dt).to(env.device).norm(dim=-1)      # (N, K+1)
        F_rd = torch.stack([v.get_contact_force_matrix(dt).to(env.device).norm(dim=-1).sum(-1) for v in self.v_rob_deck], -1) \
            if self.v_rob_deck else torch.zeros(env.num_envs, 1, device=env.device)
        tips = F_rc[:, self.tip_ids] > c.card_pad_force
        pinch_f = tips[:, 0] & (tips[:, 1:].sum(-1) >= 1)
        f_robot_card = F_rc.sum(-1)
        # the force view reads one substep and flickers on a held card (A10 11-17: 0 N for 15+ steps while the card
        # stayed in the pinch), so pad contact also comes from the TacMap ray-cast (FINGERS: thumb, index, middle, ...)
        # "both pads loaded" for a 5-finger hand: >= 2 robot contacts on the card (force on >= 2 links, or >= 2 TacMap
        # pads, or 1 pad + measured force), debounced over 3 steps (single-substep force reads flicker)
        n_links = (F_rc > c.card_pad_force).sum(-1)
        tac_on = None
        if getattr(env, "_tactile", None) is not None:
            T = env._tactile.compute().view(env.num_envs, 5, -1)
            tac_on = T.amax(-1) > 0.0
            raw = pinch_f | (n_links >= 2) | (tac_on.sum(-1) >= 2) | ((tac_on.sum(-1) >= 1) & (f_robot_card > c.card_pad_force))
        else:
            raw = pinch_f | (n_links >= 2)
        if not hasattr(self, "pinch_hist"):
            self.pinch_hist = torch.zeros(3, env.num_envs, dtype=torch.bool, device=env.device)
        self.pinch_hist = torch.roll(self.pinch_hist, 1, 0); self.pinch_hist[0] = raw
        pinch = self.pinch_hist.any(0)
        self.f_env = F_ce.sum(-1)
        self.f_env_ep_max = torch.maximum(self.f_env_ep_max, self.f_env)
        f_rob_scene = f_robot_card + F_rd.sum(-1)
        self.f_rob = f_rob_scene

        # card in the link7 frame
        b = env.robot.data.body_pos_w[:, self.i7]
        bq = env.robot.data.body_quat_w[:, self.i7]
        op, oq = env.object.data.root_pos_w, env.object.data.root_quat_w
        p7 = quat_apply_inverse(bq, op - b)
        q7 = quat_mul(quat_conjugate(bq), oq)
        cur = self._pts(p7, q7, self.corners)
        st = self.step
        first = st == 0
        self.ref_p[first], self.ref_q[first] = p7[first], q7[first]
        in_hold = st < c.card_hold_steps
        slip0 = (cur - self._pts(self.ref_p, self.ref_q, self.corners)).norm(dim=-1).max(-1).values
        self.slip_hold = torch.where(in_hold, torch.maximum(self.slip_hold, slip0), self.slip_hold)
        self.hold_ok &= ~in_hold | (pinch & (self.f_env < c.card_clear_force))
        end_hold = st == c.card_hold_steps
        self.tr_p[end_hold], self.tr_q[end_hold] = p7[end_hold], q7[end_hold]
        transport = (st >= c.card_hold_steps) & ~self.released
        slip1 = (cur - self._pts(self.tr_p, self.tr_q, self.corners)).norm(dim=-1).max(-1).values
        self.slip_tr = torch.where(transport, torch.maximum(self.slip_tr, slip1), self.slip_tr)
        self.tr_steps += transport.float()
        self.tr_loaded += (transport & pinch).float()

        # card + deck in the receptive frame
        pc, qc = self._rel_hole(op, oq)
        bot = self._pts(pc, qc, self.bottom)
        free = (f_robot_card < c.card_clear_force) if tac_on is None else (f_robot_card < c.card_clear_force) & ~tac_on.any(-1)
        self.nocontact = torch.where(free, self.nocontact + 1, torch.zeros_like(self.nocontact))
        # release = pads off the card for card_release_steps AND the card moved > 2 mm in the hand frame (a real
        # release: it slides / drops), or no contact at all for 15 steps; a few contact-free substep reads while the
        # card stays put in the pinch (PhysX patch flicker) are not a release
        moved_in_hand = slip1 > 0.002
        just_rel = (~self.released) & (st >= c.card_hold_steps) & (
            ((self.nocontact >= c.card_release_steps) & moved_in_hand) | (self.nocontact >= 15))
        if just_rel.any():
            # the release step = the first contact-free step; its corners ~ the current ones (a few ms later)
            self.rel_inside[just_rel] = (self._in_slot(bot) & (bot[..., 2] <= WALL_TOP_Z).all(-1))[just_rel]
            self.rel_palm[just_rel] = b[just_rel]
            self.released |= just_rel
        self.just_rel_inside = just_rel & self.rel_inside

        dists = env.distractors
        deck_ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        deck_moved = torch.zeros(env.num_envs, device=env.device)
        deck_rot = torch.zeros(env.num_envs, device=env.device)
        for k, body in enumerate(dists):
            pk, qk = self._rel_hole(body.data.root_pos_w, body.data.root_quat_w)
            mv = (pk - self.deck_ref_p[k]).norm(dim=-1)
            rt = self._ang_deg(qk, self.deck_ref_q[k])
            bk = self._pts(pk, qk, self.bottom)
            deck_ok &= (mv < c.card_deck_move_tol) & (rt < c.card_deck_rot_tol) & self._in_slot(bk) \
                & ((bk[..., 2].min(-1).values - FLOOR_Z).abs() <= c.card_floor_tol)
            deck_moved = torch.maximum(deck_moved, mv)
            deck_rot = torch.maximum(deck_rot, rt)
            if k == len(dists) - 1:
                front_p, front_q = pk, qk
        zlow = bot[..., 2].min(-1).values
        n_front = quat_apply(front_q, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
        gap = ((pc - front_p) * n_front).sum(-1) - 2 * HALF[2]
        lean_d = (self._lean_deg(qc) - self._lean_deg(front_q)).abs()
        # "at rest over the final window" from the card's DISPLACEMENT over the last 12 policy steps (0.2 s): PhysX
        # reports a few mm/s of contact-correction velocity on a resting card whose position does not change
        self.pos_hist = torch.roll(self.pos_hist, 1, 0)
        self.pos_hist[0] = pc
        pdt = float(env.physics_dt) * int(env.cfg.decimation)
        speed = ((self.pos_hist[0] - self.pos_hist[-1]).norm(dim=-1) / (pdt * (self.pos_hist.shape[0] - 1))).nan_to_num(1.0)
        in_holder = self._in_slot(bot) & ((zlow - FLOOR_Z).abs() <= c.card_floor_tol) & (lean_d < c.card_lean_tol) \
            & (gap < c.card_gap_tol) & (gap > -c.card_gap_tol) & (speed < c.card_rest_speed)
        clear = self.released & (f_rob_scene < c.card_clear_force) & ((b - self.rel_palm).norm(dim=-1) >= c.card_clear_dist)
        stopped = env.robot.data.body_lin_vel_w[:, self.arm_ids].norm(dim=-1).max(-1).values < c.card_stop_speed
        held = self.hold_ok & (self.slip_hold < c.card_hold_slip) & (self.slip_tr < c.card_transport_slip) \
            & (self.tr_loaded >= c.card_loaded_frac * self.tr_steps)
        end_ok = in_holder & deck_ok & clear & stopped & self.rel_inside & held
        self.final_count = torch.where(end_ok, self.final_count + 1, torch.zeros_like(self.final_count))
        self.success = self.final_count >= c.card_final_steps

        # early-fail conditions (used for termination when card_terminate_on_fail)
        hold_done = st >= c.card_hold_steps
        self.fail["dropped_at_start"] = hold_done & ~(self.hold_ok & (self.slip_hold < c.card_hold_slip)) if c.card_fail_hold else torch.zeros_like(deck_ok)
        self.fail["slipped"] = transport & (self.slip_tr >= c.card_transport_slip) if c.card_fail_slip else torch.zeros_like(deck_ok)
        self.fail["released_outside"] = self.released & ~self.rel_inside
        # early-termination tolerance: curriculum from (card_deck_fail_move_start, _rot_start) to the official values
        # over card_deck_fail_curriculum_steps env steps; the success check always uses the official 3 mm / 3 deg
        a = 1.0
        if c.card_deck_fail_curriculum_steps > 0:
            a = min(1.0, float(env.common_step_counter + c.card_deck_fail_curriculum_offset) / float(c.card_deck_fail_curriculum_steps))
        mv_tol = c.card_deck_fail_move_start + a * (c.card_deck_move_tol - c.card_deck_fail_move_start)
        rt_tol = c.card_deck_fail_rot_start + a * (c.card_deck_rot_tol - c.card_deck_fail_rot_start)
        self.fail["deck_disturbed"] = (deck_moved >= mv_tol) | (deck_rot >= rt_tol)
        env.extras["card/deck_fail_move_tol_mm"] = mv_tol * 1e3
        if c.card_break_force > 0.0:
            over = self.f_env > c.card_break_force
            self.break_count = torch.where(over, self.break_count + 1, torch.zeros_like(self.break_count))
            self.fail["broken"] = self.break_count >= c.card_break_steps
        self.checks = dict(held_in_hand_at_start=self.hold_ok & (self.slip_hold < c.card_hold_slip),
                           held_through_transport=(self.slip_tr < c.card_transport_slip) & (self.tr_loaded >= c.card_loaded_frac * self.tr_steps),
                           released_inside_slot=self.released & self.rel_inside, card_in_holder=in_holder,
                           existing_cards_remain=deck_ok, released_and_clear=clear, robot_stopped=stopped)
        self.meas = dict(gap_mm=gap * 1e3, lean_diff_deg=lean_d, zlow_mm=(zlow - FLOOR_Z) * 1e3, deck_moved_mm=deck_moved * 1e3,
                         deck_rot_deg=deck_rot, slip_hold_mm=self.slip_hold * 1e3, slip_tr_mm=self.slip_tr * 1e3,
                         f_env=self.f_env, f_env_ep_max=self.f_env_ep_max, f_robot_scene=f_rob_scene,
                         pinch=pinch.float(), card_speed=speed, palm_from_release=(b - self.rel_palm).norm(dim=-1))
        self.step += 1
        ex = env.extras
        for k, v in self.checks.items():
            ex[f"card/{k}"] = v.float().mean()
        ex["card/f_env_mean"] = self.f_env.mean()
        ex["card/f_env_ep_max_mean"] = self.f_env_ep_max.mean()
        ex["card/released"] = self.released.float().mean()
        ex["card/pinch"] = pinch.float().mean()
        ex["card/success"] = self.success.float().mean()
