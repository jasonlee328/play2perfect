"""A10: the scripted cb_card expert (FoundationTouch tools/contactbench/card_expert.py, vertical route) as a
vectorized per-env controller inside PreciseAssemblyEnv, so RL can learn a residual on top of it
(`precise_assembly.card_expert_residual=true`: joint target = expert target + scale x policy action).

Phases per env (policy steps since its reset; durations from the cfg, defaults = sweep22 V0r35o1, 20.5 % official
success with zero residual):
  0 hold     card_expert_t_hold     targets = the start state's pinch targets
  1 carry    card_expert_t_carry    card -> upright at (x, 0, z_hi) in the hole frame, smoothstep path from the pose
                                    at the end of the hold; arm by one damped-least-squares step on the card pose error
  2 align    card_expert_t_align    track (x, 0, z_hi) upright
  3 descend  card centre z_hi -> release height (card bottom = release_mm above the 2 mm floor) at v, v_slow once the
             bottom is below slow_z; + 20 steps at the release height
  4 open     card_expert_t_open     thumb / index / middle flexion targets -> open x their value
  5 retreat  card_expert_t_retreat  link7 +40 mm x, +60 mm z (hole frame), orientation kept
  6 still    rest of the episode    targets held
Not ported (V0r35o1 has them off): tuck, squeeze_add, lean_deg, ki, recording / debug / video.
The expert's IK iterations all use the same (unstepped) state, so ik_iters > 1 gives the same target as 1.
"""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_from_angle_axis, quat_mul

OPEN_JOINTS = ["left_1_thumb_CMC_FE", "left_thumb_MCP_FE", "left_thumb_IP", "left_2_index_MCP_FE", "left_index_PIP",
               "left_index_DIP", "left_3_middle_MCP_FE", "left_middle_PIP", "left_middle_DIP"]
ARM = ["iiwa14_joint_%d" % k for k in range(1, 8)]
N_PHASE = 7


class CardExpert:
    def __init__(self, env, card):
        self.env, self.card = env, card
        c = env.cfg.precise_assembly
        g = lambda k, d: float(getattr(c, "card_expert_" + k, d))   # noqa: E731
        rob = env.robot
        dev, N = env.device, env.num_envs
        names = list(rob.joint_names); jid = {n: i for i, n in enumerate(names)}
        self.arm = torch.tensor([jid[n] for n in ARM], device=dev)
        self.opn = torch.tensor([jid[n] for n in OPEN_JOINTS if n in jid], device=dev)
        self.lo = rob.data.soft_joint_pos_limits[0, :, 0].clone(); self.hi = rob.data.soft_joint_pos_limits[0, :, 1].clone()
        self.i7 = card.i7
        pdt = float(env.physics_dt) * int(env.cfg.decimation)
        self.gain = g("gain", 0.6); self.open = g("open", 0.45)
        sq = env._pih_object_start_rel[3:7]
        self.q_up = torch.tensor([sq[3], sq[0], sq[1], sq[2]], device=dev).expand(N, 4).contiguous()
        x_g = g("x_mm", 3.5) * 1e-3; z_hi = g("z_hi_mm", 100.0) * 1e-3
        z_lo = 0.002 + 0.0275 + g("release_mm", 35.0) * 1e-3
        self.p_hi = torch.tensor([x_g, 0.0, z_hi], device=dev).expand(N, 3).contiguous()
        v, v_slow, slow_z = g("v_mm_s", 20.0), g("v_slow_mm_s", 5.0), g("slow_z_mm", 50.0)
        zp = [z_hi]
        while zp[-1] > z_lo:
            bot = zp[-1] - 0.0275
            vv = v_slow if (v_slow > 0 and bot < slow_z * 1e-3) else v
            zp.append(max(z_lo, zp[-1] - vv * 1e-3 * pdt))
        n_desc = len(zp) - 1
        # z target at descent step j = zp[min(j + 1, n_desc)]
        self.z_desc = torch.tensor([zp[min(j + 1, n_desc)] for j in range(n_desc + 20)], device=dev)
        lens = [int(g("t_hold", 1.3) / pdt), int(g("t_carry", 4.0) / pdt), int(g("t_align", 1.0) / pdt),
                n_desc + 20, int(g("t_open", 0.1) / pdt), int(g("t_retreat", 1.5) / pdt)]
        self.lens = torch.tensor(lens + [10 ** 9], device=dev)
        self.starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.cumsum(self.lens, 0)[:-1]])
        J = len(names)
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self.tgt = torch.zeros(N, J, device=dev)
        self.q_open = torch.zeros(N, J, device=dev)
        self.t_open0 = torch.zeros(N, J, device=dev)
        self.p0 = torch.zeros(N, 3, device=dev); self.q0 = torch.zeros(N, 4, device=dev); self.q0[:, 0] = 1
        self.p70 = torch.zeros(N, 3, device=dev); self.q70 = torch.zeros(N, 4, device=dev); self.q70[:, 0] = 1
        self.ret = torch.tensor([0.04, 0.0, 0.06], device=dev)

    def reset(self, env_ids):
        """Call after the start state is written: env._cur_targets[env_ids] holds the bank's pinch targets."""
        self.k[env_ids] = 0
        self.tgt[env_ids] = self.env._cur_targets[env_ids]
        qo = self.tgt[env_ids].clone()
        qo[:, self.opn] = qo[:, self.opn] * self.open
        self.q_open[env_ids] = qo

    def phase(self):
        ph = (self.k.unsqueeze(-1) >= self.starts.unsqueeze(0)).sum(-1) - 1
        return ph, self.k - self.starts[ph]

    def phase_obs(self):
        ph, j = self.phase()
        oh = torch.nn.functional.one_hot(ph, N_PHASE).float()
        frac = (j.float() / self.lens[ph].clamp(max=10 ** 4).float()).clamp(0, 1)
        return torch.cat([oh, frac.unsqueeze(-1)], -1)

    def _dls(self, e):
        rob = self.env.robot
        J = rob.root_physx_view.get_jacobians()
        J = J[:, self.i7 - 1, :, :][:, :, self.arm] if rob.is_fixed_base else J[:, self.i7, :, :][:, :, 6 + self.arm]
        JJt = J @ J.transpose(1, 2) + 0.03 ** 2 * torch.eye(6, device=e.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e.unsqueeze(-1))).squeeze(-1)
        q = rob.data.joint_pos[:, self.arm] + dq.clamp(-0.05, 0.05)
        return torch.minimum(torch.maximum(q, self.lo[self.arm]), self.hi[self.arm])

    def step(self):
        """Advance one policy step; returns the expert's joint targets (robot joint order)."""
        env, card, rob = self.env, self.card, self.env.robot
        hq = env.hole_quat_wxyz
        ph, j = self.phase()
        pc, qc = card._rel_hole(env.object.data.root_pos_w, env.object.data.root_quat_w)
        # phase-start captures
        m = (ph == 1) & (j == 0)
        if m.any():
            self.p0[m] = pc[m]; self.q0[m] = qc[m]
        m = (ph == 4) & (j == 0)
        if m.any():
            self.t_open0[m] = self.tgt[m]
        m = (ph == 5) & (j == 0)
        if m.any():
            self.p70[m] = rob.data.body_pos_w[m, self.i7]; self.q70[m] = rob.data.body_quat_w[m, self.i7]
        # card-pose tracking (carry / align / descend)
        trk = (ph >= 1) & (ph <= 3)
        if trk.any():
            a = ((j + 1).float() / self.lens[1].float()).clamp(0, 1)
            s = (a * a * (3 - 2 * a)).unsqueeze(-1)
            aa = axis_angle_from_quat(quat_mul(self.q_up, quat_conjugate(self.q0)))
            q_c = quat_mul(quat_from_angle_axis(s.squeeze(-1) * aa.norm(dim=-1),
                                                torch.nn.functional.normalize(aa + 1e-9, dim=-1)), self.q0)
            p_c = self.p0 + s * (self.p_hi - self.p0)
            p_d = self.p_hi.clone()
            p_d[:, 2] = self.z_desc[j.clamp(0, self.z_desc.numel() - 1)]
            car = (ph == 1).unsqueeze(-1)
            p_t = torch.where(car, p_c, torch.where((ph == 3).unsqueeze(-1), p_d, self.p_hi))
            q_t = torch.where(car, q_c, self.q_up)
            e_p = quat_apply(hq, p_t - pc) * self.gain
            e_r = quat_apply(hq, axis_angle_from_quat(quat_mul(q_t, quat_conjugate(qc)))) * self.gain
            r = env.object.data.root_pos_w - rob.data.body_pos_w[:, self.i7]
            qa = self._dls(torch.cat([e_p - torch.cross(e_r, r, dim=-1), e_r], -1))
            self.tgt[:, self.arm] = torch.where(trk.unsqueeze(-1), qa, self.tgt[:, self.arm])
        # open
        m = ph == 4
        if m.any():
            a = ((j + 1).float() / self.lens[4].float()).clamp(0, 1).unsqueeze(-1)
            new = self.t_open0[:, self.opn] + a * (self.q_open[:, self.opn] - self.t_open0[:, self.opn])
            self.tgt[:, self.opn] = torch.where(m.unsqueeze(-1), new, self.tgt[:, self.opn])
        # retreat
        m = ph == 5
        if m.any():
            a = ((j + 1).float() / self.lens[5].float()).clamp(0, 1).unsqueeze(-1)
            p7t = self.p70 + a * quat_apply(hq, self.ret.expand_as(self.p70))
            e = torch.cat([(p7t - rob.data.body_pos_w[:, self.i7]) * self.gain,
                           axis_angle_from_quat(quat_mul(self.q70, quat_conjugate(rob.data.body_quat_w[:, self.i7]))) * self.gain], -1)
            qa = self._dls(e)
            self.tgt[:, self.arm] = torch.where(m.unsqueeze(-1), qa, self.tgt[:, self.arm])
        self.k += 1
        return self.tgt
