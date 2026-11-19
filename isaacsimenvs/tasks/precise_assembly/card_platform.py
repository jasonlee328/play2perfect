"""A14 (11-19): ContactBench task 6 "Pick up object from platform" official verdict
(force_bench appendix_tasks/pick_up_object_on_platform.py:98-133), computed every policy step for
cfg.precise_assembly.card_task == "platform".

Scene (problems cb_platform / cb_platform_key, FoundationTouch tools/contactbench/platform_tasks.py): board top
220 mm above the table, push edge at hole-frame y = plat_edge_y (-117.5 mm); the part lies flat 40 mm inboard.

Official verdict, applied over a hold window of plat_hold_steps (1.5 s at 60 Hz) that starts once the part is lifted:
  lifted          part centre z - start z >= plat_lift_m (30 mm) at the end of the window      :104-107
  pinched         thumb DP pad AND one opposing finger DP pad each > plat_pad_N (0.5 N) on
                  >= plat_pad_frac (95%) of the window frames                                   :108-120
  no_slip         part position relative to the thumb / finger midpoint moves < plat_slip_m
                  (2 mm) over the window (max deviation from the first frame)                   :121-130
Success = all three for one full window. A window that fails restarts at the next lifted step (a policy may re-grip).
Fail 'dropped': part centre below the board top - plat_drop_m (fell off the board).
Reported only: starts_mid_platform (start centre > 50 mm inboard), overhang (part beyond the edge).
"""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse

from .card_task import _glob

TIPS = ("left_thumb_DP", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_pinky_DP")


class PlatformTask:
    def __init__(self, env):
        self.env = env
        self.c = env.cfg.precise_assembly
        d, N = env.device, env.num_envs
        z = lambda dt=torch.float32: torch.zeros(N, device=d, dtype=dt)
        self.step = z(torch.long)
        self.win = z(torch.long); self.win_loaded = z(torch.long); self.win_finger = z(torch.long)
        self.win_rel0 = torch.zeros(N, 3, device=d); self.win_slip = z()
        self.success = z(torch.bool); self.ever_lifted = z(torch.bool)
        self.ref_p = torch.zeros(N, 3, device=d)
        self.rise_max = z(); self.best_loaded = z(); self.windows = z()
        self.fail = {"dropped": z(torch.bool)}
        self.checks, self.meas = {}, {}
        self.phi = torch.full((N,), float("nan"), device=d)
        self._views = False
        self._fresh = None

    def _make_views(self):
        env = self.env
        links = [_glob(p) for p in env.robot.root_physx_view.link_paths[0]]
        names = list(env.robot.root_physx_view.shared_metatype.link_names)
        self.tip_ids = [names.index(n) for n in TIPS]
        sim = env.object._physics_sim_view
        self.v_rob_obj = sim.create_rigid_contact_view(_glob(env.object.root_physx_view.prim_paths[0]), filter_patterns=links)
        self.tip_body = env.robot.find_bodies(list(TIPS), preserve_order=True)[0]
        self._views = True

    def _rel_hole(self, pos_w):
        env = self.env
        return quat_apply_inverse(env.hole_quat_wxyz, pos_w - (env.hole_pos + env.scene.env_origins))

    def reset(self, env_ids):
        for t in (self.step, self.win, self.win_loaded, self.win_finger, self.win_slip, self.rise_max,
                  self.best_loaded, self.windows):
            t[env_ids] = 0
        for t in (self.success, self.ever_lifted):
            t[env_ids] = False
        for t in self.fail.values():
            t[env_ids] = False
        self.phi[env_ids] = float("nan")
        if hasattr(self, "phi_reach"):
            self.phi_reach[env_ids] = float("nan")
        self._fresh = env_ids if self._fresh is None else torch.cat([self._fresh, env_ids])

    def update(self):
        if not self._views:
            self._make_views()
        env, c = self.env, self.c
        d, N = env.device, env.num_envs
        ar = torch.arange(N, device=d)
        self.step += 1
        po = self._rel_hole(env.object.data.root_pos_w)
        if self._fresh is not None:
            self.ref_p[self._fresh] = po[self._fresh]
            self._fresh = None
        F = self.v_rob_obj.get_contact_force_matrix(env.physics_dt).norm(dim=-1).reshape(N, -1)
        f_tip = F[:, self.tip_ids]                       # (N, 5): thumb, index, middle, ring, pinky
        f_thumb, f_oth = f_tip[:, 0], f_tip[:, 1:]
        f_other, k_other = f_oth.max(-1)
        self.f_total = F.sum(-1)
        rise = po[:, 2] - self.ref_p[:, 2]
        self.rise_max = torch.maximum(self.rise_max, rise)
        lift_m = float(c.plat_lift_m)
        lifted = rise >= lift_m
        self.ever_lifted |= lifted
        tips_w = env.robot.data.body_pos_w[:, self.tip_body]           # (N, 5, 3)
        # --- hold window ---
        start = (self.win == 0) & lifted
        self.win_finger = torch.where(start, k_other + 1, self.win_finger)
        mid = 0.5 * (tips_w[:, 0] + tips_w[ar, self.win_finger.clamp(min=1)])
        rel = env.object.data.root_pos_w - mid
        self.win_rel0 = torch.where(start[:, None], rel, self.win_rel0)
        active = (self.win > 0) | start
        f_sel = f_tip[ar, self.win_finger.clamp(min=1)]
        loaded = (f_thumb > c.plat_pad_N) & (f_sel > c.plat_pad_N)
        self.win = torch.where(active, self.win + 1, self.win)
        self.win_loaded = torch.where(active, self.win_loaded + loaded.long(), self.win_loaded)
        slip = (rel - self.win_rel0).norm(dim=-1)
        self.win_slip = torch.where(active, torch.maximum(self.win_slip * (~start).float(), slip), self.win_slip)
        H = int(c.plat_hold_steps)
        max_miss = H - int(torch.ceil(torch.tensor(c.plat_pad_frac * H)).item())
        miss = self.win - self.win_loaded
        done_win = active & (self.win >= H)
        ok = done_win & lifted & (self.win_slip < c.plat_slip_m) & (miss <= max_miss)
        self.success = self.success | ok
        # a window that can no longer pass restarts (re-grip allowed): too many unloaded frames, slip, or dropped below
        bad = active & ((miss > max_miss) | (self.win_slip >= c.plat_slip_m) | (rise < lift_m - c.plat_lift_slack_m))
        restart = (bad | done_win) & ~ok
        self.windows = self.windows + restart.float()
        self.best_loaded = torch.maximum(self.best_loaded, torch.where(active, self.win.float(), self.best_loaded) *
                                         (miss <= max_miss).float() * (self.win_slip < c.plat_slip_m).float())
        for t in (self.win, self.win_loaded, self.win_slip):
            t[restart] = 0
        self.fail["dropped"] = po[:, 2] < float(c.plat_top_z) - float(c.plat_drop_m)
        # --- shaping ---
        edge = float(c.plat_edge_y)
        front = po[:, 1] - float(c.plat_front_m)              # outboard edge of the part (hole-frame y)
        overhang = (edge - front).clamp(min=0.0)
        oh_cap = float(c.plat_overhang_cap_m)
        on_board = po[:, 2] > float(c.plat_top_z) - 0.01
        phi = float(c.plat_slide_w) * overhang.clamp(max=oh_cap) + rise.clamp(min=0.0, max=float(c.plat_rise_cap_m))
        self.dphi = (phi - self.phi).nan_to_num(0.0)
        self.phi = phi
        if not hasattr(self, "phi_reach"):
            self.phi_reach = torch.full((N,), float("nan"), device=d)
        dtip = (tips_w[:, :2] - env.object.data.root_pos_w[:, None]).norm(dim=-1).max(-1).values
        phr = -dtip.clamp(max=0.3)
        self.dphi_reach = (phr - self.phi_reach).nan_to_num(0.0)
        self.phi_reach = phr
        self.pinch = loaded
        self.loaded_any = (f_thumb > c.plat_pad_N) & (f_other > c.plat_pad_N)
        start_mid = (self.ref_p[:, 1] - edge) > 0.050
        self.checks = dict(lifted=lifted, pinched=self.loaded_any, starts_mid_platform=start_mid)
        self.meas = dict(rise_mm=rise * 1e3, rise_max_mm=self.rise_max * 1e3, overhang_mm=overhang * 1e3,
                         f_thumb=f_thumb, f_other=f_other, f_total=self.f_total, win=self.win.float(),
                         win_slip_mm=self.win_slip * 1e3, best_win=self.best_loaded, windows=self.windows,
                         reach_mm=dtip * 1e3, ever_lifted=self.ever_lifted.float(), on_board=on_board.float())

    def reward(self, tr):
        c = self.c
        r = c.card_success_bonus * tr["max_successes"].float()
        r = r + c.plat_path_rew_scale * self.dphi
        r = r + c.plat_reach_rew_scale * self.dphi_reach
        r = r + c.plat_pinch_rew * self.loaded_any.float()
        r = r + c.plat_hold_rew * (self.win > 0).float() * self.pinch.float()
        fails = [v for k, v in tr.items() if k.startswith("card_")]
        if fails:
            r = r - c.card_fail_penalty * torch.stack(fails).any(0).float()
        return r
