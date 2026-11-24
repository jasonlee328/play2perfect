"""A26: deal one card off a held deck with the thumb.

Scene: object = the TOP card, distractors = the cards below it (distractor0 = card 2, ..., last = bottom card). The
episode starts from a held-deck start bank (deck resting on a palm-up hand, thumb on the top card, arm frozen).

Per env and episode:
  target  = slide distance of the top card relative to the deck, U(deal_target_m) (actor obs field ``deal_target``).
Every step (all in the palm frame, so the frozen arm / wrist do not matter):
  slide   = planar (deck-plane) displacement of the top card relative to the bottom card, from the episode start
  others  = max planar displacement of the middle cards (card 2 .. card C-1) relative to the bottom card (deck frame)
  card2   = displacement of card 2 relative to the bottom card (the card the thumb would drag along)
  held    = bottom card within deal_drop_m of its start in the palm frame
Success: |slide - target| <= deal_tol_m, others < deal_others_m, held, for deal_hold_steps steps in a row.
Fail (termination): card2 > deal_card2_fail_m ("deal_dragged"), not held ("deal_dropped"), top card > target +
deal_over_m ("deal_over"); crush and the 8 s timeout come from the env (crush_force, episode_length_s).

Hidden physics (actor never sees it; critic gets ``deal_priv``):
  * per-card friction (scene_utils: distractor_friction_scale_range, object_friction_scale_range), init-only buckets
  * hand motor DR: per joint kp/kd x log-U(deal_motor_gain) and joint friction U(0, deal_motor_fric), per episode
    (same as tools/a23/slip_probe.py --act_dr)
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply_inverse


class DealCard:
    def __init__(self, env):
        self.env = env
        c = env.cfg.precise_assembly
        N, dev = env.num_envs, env.device
        self.cards = [env.object] + list(getattr(env, "distractors", ()))   # top .. bottom
        if len(self.cards) < 2:
            raise ValueError("deal_task needs a deck (object + >= 1 distractor card)")
        self.C = len(self.cards)
        z = lambda *s: torch.zeros(*s, device=dev)
        self.target = z(N)
        self.top0 = z(N, 3)            # top rel bottom, bottom-card frame, at episode start
        self.c20 = z(N, 3)
        self.low0 = z(self.C - 1, N, 3)  # lower cards in the palm frame at start
        self.slide = z(N); self.slide_vec = z(N, 2); self.others = z(N); self.card2 = z(N); self.bottom_move = z(N)
        self.held = torch.ones(N, dtype=torch.bool, device=dev)
        self.in_steps = torch.zeros(N, dtype=torch.long, device=dev)
        self.success = torch.zeros(N, dtype=torch.bool, device=dev)
        self.fail = {k: torch.zeros(N, dtype=torch.bool, device=dev) for k in ("dragged", "dropped", "over")}
        self.step_i = torch.zeros(N, dtype=torch.long, device=dev)
        self.err_prev = z(N)
        self.slide_max = z(N); self.others_max = z(N)
        self.ready = torch.zeros(N, dtype=torch.bool, device=dev)
        self._hand = env._hand_joint_ids
        self._k0 = None
        rb = env.robot
        self.tip_ids = env._tactile.body_ids if getattr(env, "_tactile", None) is not None else None
        self.palm = env._palm_body_id
        self.gain = torch.ones(N, len(self._hand) if not torch.is_tensor(self._hand) else self._hand.numel(), device=dev)
        self.jfric = torch.zeros_like(self.gain)

    # ---------------------------------------------------------------- frames
    def _palm_pose(self):
        s = self.env.robot.data.body_state_w[:, self.palm]
        return s[:, :3], s[:, 3:7]

    def _measure(self):
        pp, pq = self._palm_pose()
        bot = self.cards[-1].data.root_state_w
        bq = bot[:, 3:7]
        top_rel = quat_apply_inverse(bq, self.cards[0].data.root_pos_w - bot[:, :3])      # bottom-card frame
        c2_rel = quat_apply_inverse(bq, self.cards[1].data.root_pos_w - bot[:, :3]) if self.C > 2 else torch.zeros_like(top_rel)
        # middle cards (card 2 .. card C-1) rel. the bottom card (deck frame); the bottom card itself in the palm frame
        mid = [quat_apply_inverse(bq, b.data.root_pos_w - bot[:, :3]) for b in self.cards[1:-1]]
        bot_p = quat_apply_inverse(pq, bot[:, :3] - pp)
        low = torch.stack(mid + [bot_p], 0)
        return top_rel, c2_rel, low

    # ---------------------------------------------------------------- reset
    def _motor_dr(self, env_ids):
        c = self.env.cfg.precise_assembly
        lo, hi = (float(v) for v in c.deal_motor_gain)
        fr = float(c.deal_motor_fric)
        if (lo, hi) == (1.0, 1.0) and fr <= 0.0:
            return
        rb = self.env.robot
        hand = self._hand
        if self._k0 is None:
            self._k0 = rb.root_physx_view.get_dof_stiffnesses().to(self.env.device)[:, hand].clone()
            self._d0 = rb.root_physx_view.get_dof_dampings().to(self.env.device)[:, hand].clone()
        n = env_ids.numel()
        g = torch.exp(torch.empty(n, self.gain.shape[1], device=self.env.device).uniform_(math.log(lo), math.log(hi)))
        mu = torch.empty(n, self.gain.shape[1], device=self.env.device).uniform_(0.0, max(fr, 0.0))
        self.gain[env_ids] = g; self.jfric[env_ids] = mu
        rb.write_joint_stiffness_to_sim(self._k0[env_ids] * g, joint_ids=hand, env_ids=env_ids)
        rb.write_joint_damping_to_sim(self._d0[env_ids] * g, joint_ids=hand, env_ids=env_ids)
        rb.write_joint_friction_coefficient_to_sim(mu, joint_dynamic_friction_coeff=0.8 * mu, joint_ids=hand, env_ids=env_ids)

    def reset(self, env_ids):
        c = self.env.cfg.precise_assembly
        lo, hi = (float(v) for v in c.deal_target_m)
        n_r = int(getattr(c, "deal_target_ramp_steps", 0))
        if n_r > 0:   # curriculum: target range ramps from deal_target_start_m to deal_target_m (env steps; a resume replays it)
            f = min(1.0, (int(self.env.common_step_counter) + int(getattr(c, "deal_target_ramp_offset_steps", 0))) / n_r)
            lo0, hi0 = (float(v) for v in c.deal_target_start_m)
            lo, hi = lo0 + f * (lo - lo0), hi0 + f * (hi - hi0)
        self.target[env_ids] = lo + (hi - lo) * torch.rand(env_ids.numel(), device=self.env.device)
        self.ready[env_ids] = False          # reference captured at the first update (poses written by now)
        self.step_i[env_ids] = 0
        self.in_steps[env_ids] = 0
        self.success[env_ids] = False
        for v in self.fail.values():
            v[env_ids] = False
        self.slide[env_ids] = 0.0; self.others[env_ids] = 0.0; self.card2[env_ids] = 0.0
        self.slide_max[env_ids] = 0.0; self.others_max[env_ids] = 0.0
        self.held[env_ids] = True
        self._motor_dr(env_ids)

    # ---------------------------------------------------------------- per step
    def update(self):
        c = self.env.cfg.precise_assembly
        top_rel, c2_rel, low = self._measure()
        new = ~self.ready
        if new.any():
            self.top0[new] = top_rel[new]; self.c20[new] = c2_rel[new]; self.low0[:, new] = low[:, new]
            self.ready |= new
        self.step_i += 1
        d = top_rel - self.top0
        self.slide_vec = d[:, :2]
        self.slide = d[:, :2].norm(dim=-1)
        dl = (low - self.low0)[..., :2].norm(dim=-1)              # (C-1, N): middle cards in the deck plane, bottom in palm xy
        self.others = dl[:-1].amax(0) if self.C > 2 else torch.zeros_like(dl[0])
        self.bottom_move = (low[-1] - self.low0[-1]).norm(dim=-1)
        self.card2 = (c2_rel - self.c20)[:, :2].norm(dim=-1) if self.C > 2 else dl[0]
        self.held = self.bottom_move < float(c.deal_drop_m)
        self.slide_max = torch.maximum(self.slide_max, self.slide)
        self.others_max = torch.maximum(self.others_max, self.others)
        err = (self.slide - self.target).abs()
        ok = (err <= float(c.deal_tol_m)) & (self.others < float(c.deal_others_m)) & self.held
        self.in_steps = torch.where(ok, self.in_steps + 1, torch.zeros_like(self.in_steps))
        self.success = self.in_steps >= int(c.deal_hold_steps)
        self.err = err
        self.fail["dragged"] = (self.card2 > float(c.deal_card2_fail_m)) | (self.others > float(c.deal_others_fail_m))
        self.fail["dropped"] = ~self.held
        self.fail["over"] = self.slide > self.target + float(c.deal_over_m)
        for k in self.fail:
            self.fail[k] &= ~self.success

    def reward(self):
        """Potential on the slide error (palm-free), small penalty on lower-card motion, success / fail terms."""
        c = self.env.cfg.precise_assembly
        phi = -self.err
        fresh = self.step_i <= 1
        dphi = torch.where(fresh, torch.zeros_like(phi), phi - self.err_prev_phi) if hasattr(self, "err_prev_phi") else torch.zeros_like(phi)
        self.err_prev_phi = phi.clone()
        r = float(c.deal_progress_scale) * dphi
        r = r - float(c.deal_others_penalty) * (self.others / max(float(c.deal_others_m), 1e-6)).clamp(max=3.0)
        r = r + float(c.deal_in_tol_rew) * (self.in_steps > 0).float()
        return r

    # ---------------------------------------------------------------- obs
    def actor_obs(self):
        return (self.target * 100.0).unsqueeze(-1)          # cm

    def critic_obs(self):
        mu = getattr(self.env, "_deal_card_mu", None)
        if mu is None:
            mu = torch.zeros(self.env.num_envs, self.C, device=self.env.device)
        return torch.cat([self.target.unsqueeze(-1) * 100.0, self.slide_vec * 100.0, self.slide.unsqueeze(-1) * 100.0,
                          self.others.unsqueeze(-1) * 100.0, self.card2.unsqueeze(-1) * 100.0,
                          (self.in_steps.float() / 30.0).unsqueeze(-1), mu,
                          self.gain.mean(-1, keepdim=True), self.jfric.mean(-1, keepdim=True) * 20.0], -1)

    def log(self, extras):
        extras["deal/slide_mm"] = self.slide.mean() * 1e3
        extras["deal/err_mm"] = self.err.mean() * 1e3
        extras["deal/others_mm"] = self.others.mean() * 1e3
        extras["deal/in_tol"] = (self.in_steps > 0).float().mean()
