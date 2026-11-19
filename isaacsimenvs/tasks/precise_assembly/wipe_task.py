"""A16: "wipe a whiteboard with a held eraser block" (FoundationTouch tools/contactbench/wipe_{assets,tasks}.py).

Active for cfg.precise_assembly.wipe_task != "off" ("report" = logged only, "gate" = the verdict is task success).

Scene: object = the eraser (felt = local -z), distractor0 = the board plate on a prismatic spring joint along the
fixture z. Every env has a HIDDEN board rest height h (m, fixed for the env's life; env._wipe_h_np, drawn at init
from layout["wipe"]["height_range_mm"] or cfg wipe_height_range_mm): the joint's fixture anchor is shifted by
h + m g / k, the plate starts at nominal + h.

Per policy step:
  F        = |eraser <-> board contact force| (PhysX contact view, x wipe_force_scale)
  felt_ok  = the eraser's -z within felt_tilt_deg of the board's -z
  under    = ink cells (5 mm, in the board frame) inside the felt footprint (eraser x / y extents, eraser yaw)
  a cell gains a step when under & felt_ok & F in force_window_N; it is erased after erase_steps steps
  pressed_too_hard (fail) = F > fail_force_N for fail_steps steps in a row
  dropped (fail)          = eraser > wipe_drop_dist from the palm centre, or felt tilted > wipe_drop_tilt_deg
  incomplete (fail)       = the last waypoint is credited with < success_frac of the cells erased
  success                 = the last waypoint is credited with >= success_frac erased
Goals: the waypoints are at the nominal board height; with wipe_task on the env credits them z-free (keypoint
distance with the mean height difference removed, obs_utils.compute_intermediate_values), so neither goal timing nor
the keypoint reward depends on h.
Actor obs (wipe_virtual_obs): object_rot / object_vel / keypoints_rel_palm / keypoints_rel_goal come from a VIRTUAL
eraser = palm pose x the in-hand offset measured on the episode's first step (rigid-grasp assumption), so the eraser
being stopped / tilted / slipping on the board is not visible; reward / lifted_object / closest_keypoint_max_dist are
zeroed for the actor. The critic sees the true values.
"""
from __future__ import annotations

import json
import math
import re

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul


def _glob(p: str) -> str:
    return re.sub(r"env_\d+", "env_*", p)


def load_layout(problem_name: str) -> dict:
    from evaluation.problems.contactbench.objects import ASSETS_DIR
    meta = json.loads((ASSETS_DIR / "meta.json").read_text())
    short = problem_name.split(".", 1)[-1]
    return meta["problems"][short]["layout"]["wipe"]


class WipeTask:
    def __init__(self, env):
        self.env = env
        self.c = c = env.cfg.precise_assembly
        L = env._wipe_layout
        d, N = env.device, env.num_envs
        self.L = L
        z = lambda dt=torch.float32: torch.zeros(N, device=d, dtype=dt)
        xs = torch.arange(L["ink_x_mm"][0] + L["cell_mm"] / 2, L["ink_x_mm"][1], L["cell_mm"], device=d) * 1e-3
        ys = torch.arange(L["ink_y_mm"][0] + L["cell_mm"] / 2, L["ink_y_mm"][1], L["cell_mm"], device=d) * 1e-3
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        self.cells = torch.stack([gx.flatten(), gy.flatten()], -1)            # (C, 2) board frame
        self.C = self.cells.shape[0]
        self.cell_steps = torch.zeros(N, self.C, device=d, dtype=torch.long)
        self.erased = torch.zeros(N, self.C, device=d, dtype=torch.bool)
        self.frac = z(); self.new_erased = z()
        self.over_count = z(torch.long)
        self.F = z(); self.F_ep_max = z(); self.inwin_steps = z(); self.contact_steps = z()
        self.success = z(torch.bool); self.over_ever = z(torch.bool)
        self.fail = {k: z(torch.bool) for k in ("pressed_too_hard", "dropped", "incomplete")}
        self.meas = {}
        self.h = torch.as_tensor(env._wipe_h_np, device=d, dtype=torch.float32)
        self.half = (0.5e-3 * L["felt_xy_mm"][0], 0.5e-3 * L["felt_xy_mm"][1])
        self.eh = float(L["eraser_half_h_m"])
        self.board_half_t = 0.004
        self.cos_felt = math.cos(math.radians(float(L["felt_tilt_deg"])))
        self.cos_drop = math.cos(math.radians(float(c.wipe_drop_tilt_deg)))
        self.win = [float(v) for v in L["force_window_N"]]
        self.fail_F = float(L["fail_force_N"]); self.fail_steps = int(L["fail_steps"])
        self.erase_steps = int(L["erase_steps"]); self.succ_frac = float(L["success_frac"])
        # virtual eraser: in-hand offset (palm body frame) measured on the first step
        self.off_p = torch.zeros(N, 3, device=d); self.off_q = torch.zeros(N, 4, device=d); self.off_q[:, 0] = 1.0
        self.need_off = torch.ones(N, device=d, dtype=torch.bool)
        self._views = False

    def _make_views(self):
        env = self.env
        sim = env.object._physics_sim_view
        board = _glob(env.distractors[0].root_physx_view.prim_paths[0])
        self.v_eb = sim.create_rigid_contact_view(_glob(env.object.root_physx_view.prim_paths[0]), filter_patterns=[board])
        self._views = True

    def reset(self, env_ids):
        self.cell_steps[env_ids] = 0
        self.erased[env_ids] = False
        for t in (self.frac, self.new_erased, self.over_count, self.F, self.F_ep_max, self.inwin_steps,
                  self.contact_steps):
            t[env_ids] = 0
        self.success[env_ids] = False; self.over_ever[env_ids] = False
        for v in self.fail.values():
            v[env_ids] = False
        self.need_off[env_ids] = True

    # ------------------------------------------------------------------ geometry
    def _board_frame(self):
        b = self.env.distractors[0]
        return b.data.root_pos_w, b.data.root_quat_w

    def update(self):
        env = self.env
        if not self._views:
            self._make_views()
        F = self.v_eb.get_contact_force_matrix(float(env.physics_dt)).to(env.device).norm(dim=-1).sum(-1)
        F = F * float(self.c.wipe_force_scale)
        self.F = F
        self.F_ep_max = torch.maximum(self.F_ep_max, F)
        bp, bq = self._board_frame()
        ep, eq = env.object.data.root_pos_w, env.object.data.root_quat_w
        e_b = quat_apply_inverse(bq, ep - bp)                                       # eraser centre, board frame
        q_rel = quat_mul(quat_conjugate(bq), eq)
        N = env.num_envs
        ez = quat_apply(q_rel, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(N, 3))
        felt_ok = ez[:, 2] > self.cos_felt
        ex = quat_apply(q_rel, torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(N, 3))[:, :2]
        ex = ex / ex.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ey = torch.stack([-ex[:, 1], ex[:, 0]], -1)
        rel = self.cells[None] - e_b[:, None, :2]                                   # (N, C, 2)
        u = (rel * ex[:, None]).sum(-1); v = (rel * ey[:, None]).sum(-1)
        under = (u.abs() <= self.half[0]) & (v.abs() <= self.half[1])
        gap = e_b[:, 2] - self.eh - self.board_half_t                               # felt above the board top (m)
        inwin = (F >= self.win[0]) & (F <= self.win[1]) & felt_ok & (gap < 0.003)
        before = self.erased.sum(-1)
        self.cell_steps += (under & inwin[:, None]).long()
        self.erased = self.cell_steps >= self.erase_steps
        after = self.erased.sum(-1)
        self.new_erased = (after - before).float()
        self.frac = after.float() / self.C
        self.inwin_steps += inwin.float()
        self.contact_steps += (F > 0.5).float()
        over = F > self.fail_F
        self.over_count = torch.where(over, self.over_count + 1, torch.zeros_like(self.over_count))
        self.fail["pressed_too_hard"] = self.over_count >= self.fail_steps
        self.over_ever = self.over_ever | self.fail["pressed_too_hard"]
        palm = env.robot.data.body_state_w[:, env._palm_body_id, 0:3]
        tilt_bad = ez[:, 2] < self.cos_drop
        self.fail["dropped"] = ((ep - palm).norm(dim=-1) > float(self.c.wipe_drop_dist)) | tilt_bad
        self.meas = dict(shear_mm=getattr(self, 'shear_mm', torch.zeros(N, 5, device=env.device)).amax(-1), force=F, frac=self.frac, gap_mm=gap * 1e3, felt_cos=ez[:, 2], h_mm=self.h * 1e3,
                         F_ep_max=self.F_ep_max, inwin_steps=self.inwin_steps, contact_steps=self.contact_steps)
        self.inwin = inwin

    def gate(self, stroke_done: torch.Tensor) -> torch.Tensor:
        """stroke_done = the last waypoint credited this step -> success / incomplete."""
        ok = stroke_done & (self.frac >= self.succ_frac)
        self.success = ok
        self.fail["incomplete"] = stroke_done & ~ok
        return ok

    # ------------------------------------------------------------------ actor obs
    def actor_obs_hook(self, obs: dict, palm_pos_w, palm_rot, palm_center, goal_kp, kp_offsets, origins, keypoints_world):
        env = self.env
        ep, eq = env.object.data.root_pos_w, env.object.data.root_quat_w
        m = self.need_off & (env.episode_length_buf >= 1)
        if m.any():
            self.off_p[m] = quat_apply_inverse(palm_rot[m], ep[m] - palm_pos_w[m])
            self.off_q[m] = quat_mul(quat_conjugate(palm_rot[m]), eq[m])
            self.need_off[m] = False
        use = ~self.need_off
        vp = palm_pos_w + quat_apply(palm_rot, self.off_p)
        vq = quat_mul(palm_rot, self.off_q)
        vp = torch.where(use[:, None], vp, ep); vq = torch.where(use[:, None], vq, eq)
        vkp = keypoints_world(vp - origins, vq, kp_offsets)
        obs["object_rot"] = torch.cat([vq[:, 1:], vq[:, :1]], -1)                  # xyzw
        palm_vel = env.robot.data.body_state_w[:, env._palm_body_id, 7:13]
        obs["object_vel"] = torch.where(use[:, None], palm_vel, obs["object_vel"])
        obs["keypoints_rel_palm"] = vkp - palm_center.unsqueeze(1)
        obs["keypoints_rel_goal"] = vkp - goal_kp
        for k in ("reward", "lifted_object", "closest_keypoint_max_dist"):
            if k in obs:
                obs[k] = torch.zeros_like(obs[k])

    # ------------------------------------------------------------------ touch: pad shear
    def _shear_setup(self):
        tac = self.env._tactile
        res = tac.res
        nf = int(tac.finger.max()) + 1
        d = self.env.device
        self.sh_res, self.sh_nf = res, nf
        self.sh_pinv, self.sh_nrm = [], []
        for f in range(nf):
            m = tac.finger == f
            loc = tac.taxel[m] - f * res * res
            cnt = torch.bincount(loc, minlength=res * res).float()
            cen = torch.zeros(res * res, 3, device=d).index_add_(0, loc, tac.starts[m]) / cnt.clamp_min(1)[:, None]
            ok = cnt > 0
            k = torch.arange(res * res, device=d)[ok]
            X = torch.stack([torch.ones_like(k, dtype=torch.float32), (k // res).float(), (k % res).float()], -1)
            coef = torch.linalg.lstsq(X, cen[ok]).solution        # (3, 3): c0, d/drow, d/dcol (DP frame, m)
            A = coef[1:].T                                         # (3, 2)
            self.sh_pinv.append(torch.linalg.pinv(A))              # (2, 3): DP-frame displacement -> (drow, dcol)
            n = tac.dirs[m].mean(0); self.sh_nrm.append(n / n.norm())
        self.sh_pinv = torch.stack(self.sh_pinv); self.sh_nrm = torch.stack(self.sh_nrm)
        self.sh_body = tac.body_ids
        self.v_eb_vec = self.v_eb

    def shear_tactile(self, tac_img: torch.Tensor) -> torch.Tensor:
        """Shift each in-contact pad image along the pad by its share of the board load (gel shear model)."""
        import torch.nn.functional as Fn
        env = self.env
        if not self._views:
            self._make_views()
        if not hasattr(self, "sh_pinv"):
            self._shear_setup()
        N, res, nf = env.num_envs, self.sh_res, self.sh_nf
        img = tac_img.view(N, nf, res, res)
        fv = self.v_eb.get_contact_force_matrix(float(env.physics_dt)).to(env.device).sum(1)   # (N, 3) on the eraser
        fv = fv * float(self.c.wipe_force_scale)
        on = img.flatten(2).amax(-1) > 0                                                        # (N, nf) pads in contact
        share = fv[:, None] / on.sum(-1, keepdim=True).clamp_min(1)[..., None]                  # (N, nf, 3) world
        bq = env.robot.data.body_quat_w[:, self.sh_body]                                        # (N, nf, 4)
        t = quat_apply_inverse(bq.reshape(-1, 4), share.reshape(-1, 3)).view(N, nf, 3)          # DP frame
        t = t - (t * self.sh_nrm[None]).sum(-1, keepdim=True) * self.sh_nrm[None]               # in the pad plane
        s = t * float(self.c.wipe_shear_mm_per_N) * 1e-3                                         # m
        sn = s.norm(dim=-1, keepdim=True)
        cap = float(self.c.wipe_shear_cap_mm) * 1e-3
        s = s * (cap / sn.clamp_min(1e-9)).clamp(max=1.0)
        drc = torch.einsum("fij,nfj->nfi", self.sh_pinv, s) * on[..., None].float()             # (N, nf, 2) taxels
        self.shear_mm = sn.squeeze(-1).clamp(max=cap) * 1e3 * on.float()
        # I'(r, c) = I(r - dr, c - dc): sample the source at (r - dr, c - dc), align_corners grid in [-1, 1]
        r = torch.arange(res, device=env.device, dtype=torch.float32)
        rr, cc = torch.meshgrid(r, r, indexing="ij")
        src_r = rr[None, None] - drc[..., 0, None, None]; src_c = cc[None, None] - drc[..., 1, None, None]
        grid = torch.stack([src_c, src_r], -1) / (res - 1) * 2 - 1                              # x = col, y = row
        out = Fn.grid_sample(img.reshape(N * nf, 1, res, res), grid.reshape(N * nf, res, res, 2),
                             mode="bilinear", padding_mode="zeros", align_corners=True)
        return out.view(N, -1)

    # ------------------------------------------------------------------ reward
    def reward(self, tr: dict) -> torch.Tensor:
        c = self.c
        r = c.wipe_cell_bonus * self.new_erased + c.wipe_success_bonus * tr["max_successes"].float()
        for k in self.fail:
            if f"wipe_{k}" in tr:
                r = r - c.wipe_fail_penalty * tr[f"wipe_{k}"].float()
        if c.wipe_force_rew > 0.0:
            r = r + c.wipe_force_rew * self.inwin.float()
        if c.wipe_over_rew_scale > 0.0:
            r = r - c.wipe_over_rew_scale * (self.F - self.win[1]).clamp_min(0.0)
        return r
