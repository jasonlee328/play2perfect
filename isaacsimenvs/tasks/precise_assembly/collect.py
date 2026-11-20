"""Collect mode (A19 ContactBench task 8 "collect fasteners"; task-agnostic, also usable for task 7 utensils).

Slot bodies = [env.object] + env.distractors (the Problem's object + distractors). Every reset draws N active slots
(slot 0 always active, the rest a random subset), scatters them flat on the table at random non-overlapping poses
(spread from a tight patch to the whole table), and parks the inactive ones out of reach (re-teleported every step,
masked out of obs / reward / success). The receptive fixture is the bin.

Success (all must hold for ``collect_success_steps`` consecutive steps):
  * every active slot in the bin: COM inside the bin's inner floor footprint (minus a margin) and every corner of its
    collision AABB below the rim (bin frame);
  * every active slot at rest: linear speed < ``collect_rest_speed`` (5 mm/s, the official AT_REST of
    contactbench-sim appendix_tasks/retrieve_utensil_from_bin.py);
  * no hand contact: fingertip pads read 0 (they ray-cast every slot with tactile.fingertip_distractors) and every
    fingertip > ``collect_release_dist`` from every slot COM;
  * robot stopped: palm speed < ``collect_hand_still_speed``.
Fail: an active slot below the table top (fell off), or the hand > 1.5 m away. Reward: potential-based on per-slot
progress (lifted, carried toward the bin, in the bin), approach to the nearest slot not yet in the bin, release
distance once all are in; plus a success bonus, a fall penalty and the usual joint-velocity penalties.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_mul

TABLE_HALF_HEIGHT = 0.15


class CollectManager:
    def __init__(self, env, n_range, bin_geom: dict) -> None:
        self.env = env
        c = env.cfg.precise_assembly
        self.c = c
        self.dev = env.device
        self.bodies = [env.object] + list(getattr(env, "distractors", ()))
        self.K = len(self.bodies)
        lo, hi = (int(v) for v in (c.collect_n_range or n_range))
        self.n_lo, self.n_hi = max(1, lo), min(self.K, hi)
        g = bin_geom
        self.floor_z, self.rim_z = float(g["floor_z"]), float(g["rim_z"])
        self.half_floor = torch.tensor(g["half_floor"], device=self.dev)
        self.half_rim = torch.tensor(g["half_rim"], device=self.dev)
        # per-slot collision AABB corners (body frame) from the env-0 prims
        from isaacsimenvs.tasks.play.utils.hand_skin import prim_mesh

        corners, radii = [], []
        for b in self.bodies:
            pts, _ = prim_mesh(b.cfg.prim_path)
            pts = torch.as_tensor(pts, dtype=torch.float32, device=self.dev)
            mn, mx = pts.min(0).values, pts.max(0).values
            cs = torch.stack([torch.stack([x, y, z]) for x in (mn[0], mx[0]) for y in (mn[1], mx[1])
                              for z in (mn[2], mx[2])])
            corners.append(cs)
            radii.append(float(pts.norm(dim=-1).max()))
        self.corners = torch.stack(corners)              # (K, 8, 3)
        self.radius = torch.tensor(radii, device=self.dev)  # (K,) bounding-sphere radius
        # flat rest: nut (thin axis z) flat on either face; long parts (long axis x) on their side, any roll
        ext = self.corners.max(1).values - self.corners.min(1).values
        self.long_axis_x = ext[:, 0] > 1.25 * ext[:, 2]
        N = env.num_envs
        self.active = torch.zeros(N, self.K, dtype=torch.bool, device=self.dev)
        self.active[:, 0] = True
        self.in_bin = torch.zeros_like(self.active)
        self.z0 = torch.zeros(N, self.K, device=self.dev)
        self.prev_phi = torch.zeros(N, device=self.dev)
        self.fresh = torch.ones(N, dtype=torch.bool, device=self.dev)
        self.ok_steps = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.n_active = torch.ones(N, device=self.dev)
        self.ever_in_bin = torch.zeros_like(self.active)
        self.W = int(c.collect_rest_window)
        self.pos_hist = torch.zeros(self.W, N, self.K, 3, device=self.dev)   # ring buffer for the windowed speed
        self.hist_n = torch.zeros(N, dtype=torch.long, device=self.dev)
        self.hist_i = 0
        self.success = torch.zeros(N, dtype=torch.bool, device=self.dev)
        self.fell = torch.zeros_like(self.success)
        self.park = torch.zeros(self.K, 3, device=self.dev)
        for k in range(self.K):
            self.park[k] = torch.tensor([0.0, 0.0, float(c.collect_park_z) + 0.12 * k], device=self.dev)
        print(f"[collect] {self.K} slots, N in [{self.n_lo},{self.n_hi}], radii mm "
              f"{[round(r * 1e3, 1) for r in radii]}", flush=True)

    # ------------------------------------------------------------------ reset
    def _n_hi_now(self) -> int:
        c = self.c
        if c.collect_n_curriculum_steps <= 0:
            return self.n_hi
        a = min(1.0, float(self.env.common_step_counter + c.collect_n_curriculum_offset_steps)
                / float(c.collect_n_curriculum_steps))
        start = max(self.n_lo, min(self.n_hi, int(c.collect_n_start_max)))
        return int(round(start + a * (self.n_hi - start)))

    def reset(self, env_ids: torch.Tensor) -> None:
        env, c, n, dev = self.env, self.c, env_ids.numel(), self.dev
        if n == 0:
            return
        # last-episode summary (read by evals: the auto-reset runs before the step returns)
        if not hasattr(self, "last"):
            z = torch.zeros(self.env.num_envs, device=dev)
            self.last = {"n_active": z.clone(), "n_in_bin": z.clone(), "ever_in_bin": z.clone(), "success": z.clone()}
        self.last["n_active"][env_ids] = self.active[env_ids].sum(1).float()
        self.last["n_in_bin"][env_ids] = self.in_bin[env_ids].sum(1).float()
        self.last["ever_in_bin"][env_ids] = self.ever_in_bin[env_ids].sum(1).float()
        self.last["success"][env_ids] = self.success[env_ids].float()
        hi = self._n_hi_now()
        n_act = torch.randint(self.n_lo, hi + 1, (n,), device=dev)
        perm = torch.rand(n, self.K - 1, device=dev).argsort(dim=1)  # random subset of slots 1..K-1
        rank = torch.empty_like(perm)
        rank.scatter_(1, perm, torch.arange(self.K - 1, device=dev).expand(n, -1))
        act = torch.ones(n, self.K, dtype=torch.bool, device=dev)
        act[:, 1:] = rank < (n_act - 1).unsqueeze(1)
        origins = env.scene.env_origins[env_ids]
        table_top = env._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT
        bin_xy = env.hole_pos[env_ids, :2]
        bin_q = env.hole_quat_wxyz[env_ids]
        # scatter patch per env
        rx, ry = c.collect_region_x, c.collect_region_y
        s = torch.empty(n, device=dev).uniform_(*c.collect_spread_range)
        cx = torch.empty(n, device=dev).uniform_(rx[0], rx[1])
        cy = torch.empty(n, device=dev).uniform_(ry[0], ry[1])
        M = 48
        placed = torch.zeros(n, self.K, 2, device=dev)
        has = torch.zeros(n, self.K, dtype=torch.bool, device=dev)
        bin_half = self.half_rim + 0.004 + float(c.collect_bin_clear)
        for k in range(self.K):
            # half the candidates in the episode's patch, half anywhere in the region (fallback when it is full)
            u = torch.rand(n, M, 2, device=dev) * 2 - 1
            px = (cx.unsqueeze(1) + u[..., 0] * s.unsqueeze(1)).clamp(rx[0], rx[1])
            py = (cy.unsqueeze(1) + u[..., 1] * s.unsqueeze(1)).clamp(ry[0], ry[1])
            w = torch.rand(n, M, 2, device=dev)
            px[:, M // 2:] = rx[0] + w[:, M // 2:, 0] * (rx[1] - rx[0])
            py[:, M // 2:] = ry[0] + w[:, M // 2:, 1] * (ry[1] - ry[0])
            cand = torch.stack([px, py], -1)                                       # (n, M, 2)
            rel = cand - bin_xy.unsqueeze(1)
            relb = quat_apply_inverse(bin_q.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4),
                                      torch.cat([rel, torch.zeros_like(rel[..., :1])], -1).reshape(-1, 3)
                                      ).reshape(n, M, 3)[..., :2]
            ok = ((relb.abs() - bin_half - self.radius[k]) > 0).any(-1)            # outside bin + margin
            if k:
                d = (cand.unsqueeze(2) - placed[:, :k].unsqueeze(1)).norm(dim=-1)  # (n, M, k)
                need = self.radius[k] + self.radius[:k] + float(c.collect_gap)
                clash = (d < need) & has[:, :k].unsqueeze(1)
                ok &= ~clash.any(-1)
            first = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros(n, dtype=torch.long, device=dev))
            pick = cand[torch.arange(n, device=dev), first]
            good = ok.any(1)
            placed[:, k] = pick
            has[:, k] = act[:, k] & (good | (k == 0))
            if k:
                act[:, k] &= good
        # orientations: random yaw; flip (nut) or roll about the long axis (bolts / screws)
        zax = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        xax = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
        for k, body in enumerate(self.bodies):
            yaw = torch.rand(n, device=dev) * 2 * math.pi
            if bool(self.long_axis_x[k]):
                roll = torch.rand(n, device=dev) * 2 * math.pi
            else:
                roll = (torch.rand(n, device=dev) < 0.5).float() * math.pi
            q = quat_mul(quat_from_angle_axis(yaw, zax), quat_from_angle_axis(roll, xax))
            cw = quat_apply(q.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4),
                            self.corners[k].unsqueeze(0).expand(n, -1, -1).reshape(-1, 3)).reshape(n, 8, 3)
            z = table_top - cw[..., 2].min(1).values + 0.001
            pos = torch.cat([placed[:, k], z.unsqueeze(1)], -1)
            a = act[:, k]
            park = self.park[k].expand(n, 3)
            pos = torch.where(a.unsqueeze(1), pos, park)
            pose = torch.cat([pos + origins, q], -1)
            body.write_root_pose_to_sim(pose, env_ids=env_ids)
            body.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
            self.z0[env_ids, k] = z
        self.active[env_ids] = act
        self.n_active[env_ids] = act.sum(1).float()
        self.in_bin[env_ids] = False
        self.ever_in_bin[env_ids] = False
        self.ok_steps[env_ids] = 0
        self.hist_n[env_ids] = 0
        self.fresh[env_ids] = True
        self.success[env_ids] = False
        self.fell[env_ids] = False
        env._object_init_z[env_ids] = self.z0[env_ids, 0]

    def park_inactive(self) -> None:
        env = self.env
        for k, body in enumerate(self.bodies):
            ids = (~self.active[:, k]).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                continue
            m = ids.numel()
            pose = torch.zeros(m, 7, device=self.dev)
            pose[:, :3] = self.park[k] + env.scene.env_origins[ids]
            pose[:, 3] = 1.0
            body.write_root_pose_to_sim(pose, env_ids=ids)
            body.write_root_velocity_to_sim(torch.zeros(m, 6, device=self.dev), env_ids=ids)

    # ------------------------------------------------------------------ state
    def _slots(self):
        env = self.env
        pos = torch.stack([b.data.root_pos_w for b in self.bodies], 1) - env.scene.env_origins.unsqueeze(1)
        quat = torch.stack([b.data.root_quat_w for b in self.bodies], 1)
        vel = torch.stack([b.data.root_lin_vel_w for b in self.bodies], 1)
        return pos, quat, vel   # (N, K, 3/4/3)

    def update(self) -> None:
        env, c = self.env, self.c
        N, K = env.num_envs, self.K
        pos, quat, vel = self._slots()
        self.pos, self.vel = pos, vel
        bq = env.hole_quat_wxyz.unsqueeze(1).expand(-1, K, -1).reshape(-1, 4)
        rel = quat_apply_inverse(bq, (pos - env.hole_pos.unsqueeze(1)).reshape(-1, 3)).reshape(N, K, 3)
        self.rel_bin = rel
        cw = quat_apply(quat.unsqueeze(2).expand(-1, -1, 8, -1).reshape(-1, 4),
                        self.corners.unsqueeze(0).expand(N, -1, -1, -1).reshape(-1, 3)).reshape(N, K, 8, 3)
        corners_b = quat_apply_inverse(env.hole_quat_wxyz.unsqueeze(1).unsqueeze(1).expand(-1, K, 8, -1).reshape(-1, 4),
                                       cw.reshape(-1, 3)).reshape(N, K, 8, 3) + rel.unsqueeze(2)
        m = float(c.collect_bin_margin)
        inside_xy = (rel[..., 0].abs() < self.half_floor[0] - m) & (rel[..., 1].abs() < self.half_floor[1] - m)
        top = corners_b[..., 2].max(-1).values
        bottom = corners_b[..., 2].min(-1).values
        self.in_bin = inside_xy & (top < self.rim_z) & (bottom > -0.005) & self.active
        self.ever_in_bin |= self.in_bin
        table_top = (env._table_z_per_env + TABLE_HALF_HEIGHT).unsqueeze(1)
        self.fell = ((pos[..., 2] < table_top - 0.03) & self.active).any(1)
        ft = env.robot.data.body_state_w[:, env._fingertip_body_ids, 0:3] - env.scene.env_origins.unsqueeze(1)
        d = (ft.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)                      # (N, 5, K)
        d = torch.where(self.active.unsqueeze(1), d, torch.full_like(d, 9.0))
        self.ft_min = d.min(1).values                                              # (N, K)
        self.ft_min_all = self.ft_min.min(1).values
        out = self.active & ~self.in_bin
        self.ft_min_out = torch.where(out, self.ft_min, torch.full_like(self.ft_min, 9.0)).min(1).values
        all_in = (self.in_bin == self.active).all(1)
        # at rest = mean speed over the last W policy steps (displacement / time) below collect_rest_speed: PhysX
        # contact velocities of light stacked parts jitter by a few mm/s per step without any net motion
        old = self.pos_hist[self.hist_i].clone()
        self.pos_hist[self.hist_i] = pos
        self.hist_i = (self.hist_i + 1) % self.W
        self.hist_n = (self.hist_n + 1).clamp(max=self.W + 1)
        self.speed_w = (pos - old).norm(dim=-1) / (self.W * float(env.step_dt))
        rest = (((self.speed_w < float(c.collect_rest_speed)) | ~self.active).all(1)) & (self.hist_n > self.W)
        tac = getattr(env, "_last_tactile", None)
        no_touch = torch.ones(N, dtype=torch.bool, device=self.dev) if tac is None else \
            tac.reshape(N, -1).sum(-1) <= 0.0
        hs = getattr(env, "_last_tactile_hand", None)
        if hs is not None:
            no_touch &= hs.reshape(N, -1).sum(-1) <= 0.0
        palm_v = env.robot.data.body_state_w[:, env._palm_body_id, 7:10].norm(dim=-1)
        ok = all_in & rest & no_touch & (self.ft_min_all > float(c.collect_release_dist)) & \
            (palm_v < float(c.collect_hand_still_speed))
        self.ok_steps = torch.where(ok, self.ok_steps + 1, torch.zeros_like(self.ok_steps))
        self.success = self.ok_steps >= int(c.collect_success_steps)
        self.all_in = all_in
        # potential
        h = ((pos[..., 2] - self.z0) / float(c.collect_lift_h)).clamp(0.0, 1.0)
        dxy = (pos[..., :2] - env.hole_pos[:, :2].unsqueeze(1)).norm(dim=-1)
        carry = (1.0 - dxy / float(c.collect_carry_d0)).clamp(0.0, 1.0)
        phi_k = torch.where(self.in_bin, torch.full_like(h, float(c.collect_w_bin)),
                            float(c.collect_w_lift) * h + float(c.collect_w_carry) * h * carry)
        phi = (phi_k * self.active.float()).sum(1)
        phi = phi - float(c.collect_w_approach) * self.ft_min_out.clamp(max=0.3) * (~all_in).float()
        phi = phi + float(c.collect_w_release) * (self.ft_min_all / 0.05).clamp(0.0, 1.0) * all_in.float()
        self.phi = phi

    # ------------------------------------------------------------------ outputs
    def obs(self, palm_pos: torch.Tensor) -> torch.Tensor:
        N, K = self.env.num_envs, self.K
        if not hasattr(self, "pos"):
            self.update()
        a = self.active.float().unsqueeze(-1)
        o = torch.cat([(self.pos - palm_pos.unsqueeze(1)) * a, self.rel_bin * a, a, self.in_bin.float().unsqueeze(-1)], -1)
        return o.reshape(N, K * 8)

    def reward(self) -> torch.Tensor:
        c = self.c
        r = torch.where(self.fresh, torch.zeros_like(self.phi), self.phi - self.prev_phi)
        self.prev_phi = self.phi.clone()
        self.fresh[:] = False
        bonus = float(c.collect_success_bonus) * self.success.float()
        fall = -float(c.collect_fall_penalty) * self.fell.float()
        return r, bonus, fall
