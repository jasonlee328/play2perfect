"""A18: ContactBench task 7 "Collect utensils" for PreciseAssemblyEnv (precise_assembly.collect=true).

Slot 0 = the PreciseAssembly object (always active), slots 1.. = the problem's distractors. Rules ported from the
official task-2 scene (contactbench-sim appendix_tasks/retrieve_utensil_from_bin.py): in the bin = lowest point inside
the inner footprint (half-width linear floor -> rim) and <= 10 mm above the rim; at rest = speed < 5 mm/s.
"""

from __future__ import annotations

import json
import math

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_angle_axis, quat_mul

TABLE_HALF_HEIGHT = 0.15
N_PTS = 96


def _wxyz(q_xyzw):
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def setup(env) -> None:
    spec = json.loads(env._pih_problem.collect_json)
    c = env.cfg.precise_assembly
    dev = env.device
    bodies = [env.object] + list(env.distractors)
    K = len(bodies)
    types = spec["types"]
    if len(types) != K:
        raise ValueError(f"collect: {len(types)} slot types but {K} bodies")
    if K > int(c.collect_obs_slots):
        raise ValueError(f"collect: {K} slots > collect_obs_slots {c.collect_obs_slots}")
    info = spec["type_info"]
    rest_q, rest_z, half, cxy, pts = [], [], [], [], []
    for t in types:
        i = info[t]
        rest_q.append(_wxyz(i["rest_quat_xyzw"]))
        rest_z.append(i["rest_z"])
        half.append(i["half"])
        cxy.append(i["cxy"])
        p = torch.tensor(i["pts"], dtype=torch.float32)
        idx = torch.linspace(0, len(p) - 1, N_PTS).round().long()
        pts.append(p[idx])
    b = spec["bin"]
    # long axis of each type in its body frame (first principal axis of its surface points), for collect_rew_place
    long_ax = []
    for p in pts:
        pc = p - p.mean(0)
        long_ax.append(torch.linalg.svd(pc, full_matrices=False).Vh[0])
    n_range = tuple(int(v) for v in (c.collect_n_range or spec["n_range"]))
    st = dict(
        bodies=bodies, K=K, types=types,
        rest_q=torch.tensor(rest_q, device=dev), rest_z=torch.tensor(rest_z, device=dev),
        half=torch.tensor(half, device=dev), cxy=torch.tensor(cxy, device=dev),
        pts=torch.stack(pts).to(dev), long_ax=torch.stack(long_ax).to(dev),
        rim=float(b["rim"]), floor=float(b["floor"]),
        inner_floor=torch.tensor(b["inner_floor"], device=dev), inner_rim=torch.tensor(b["inner_rim"], device=dev),
        outer=torch.tensor(b["outer"], device=dev),
        above_rim=float(b.get("in_bin_above_rim", 0.010)), rest_speed=float(b.get("at_rest_speed", 0.005)),
        n_lo=n_range[0], n_hi=n_range[-1],
    )
    N = env.num_envs
    st["active"] = torch.zeros(N, K, dtype=torch.bool, device=dev)
    st["active"][:, 0] = True
    st["inbin"] = torch.zeros(N, K, dtype=torch.bool, device=dev)
    st["n_active"] = torch.ones(N, dtype=torch.long, device=dev)
    st["n_active0"] = torch.ones(N, dtype=torch.long, device=dev)
    st["prev_phi"] = torch.zeros(N, device=dev)
    st["phi_max"] = torch.zeros(N, device=dev)
    st["fresh"] = torch.ones(N, dtype=torch.bool, device=dev)
    st["clear"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["scoop_clear"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["success"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["fell"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["n_in_ep_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["n_held_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["scoop_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["since_contact"] = torch.full((N,), 10 ** 6, dtype=torch.long, device=dev)
    st["n_scoop"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["prev_pos"] = torch.zeros(N, K, 3, device=dev)
    st["at_rest"] = torch.zeros(N, K, dtype=torch.bool, device=dev)
    st["lift_done"] = torch.zeros(N, K, dtype=torch.bool, device=dev)
    st["pinch_done"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["pinch"] = torch.zeros(N, dtype=torch.bool, device=dev)
    # scooping: trips = deposit events (in-bin count rises; one per hand visit to the bin, every rise while the hand
    # is away also counts); over budget = fail "collect_trips"
    st["trips"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["near"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["visit_dep"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["prev_n_in"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["trips_over"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["multi_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["visit_n"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["prev_n_in_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["deposit_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["place_ep_max"] = torch.zeros(N, device=dev)
    # hand link positions at the start pose (origin-relative), for collect_hand_clear_m (A18 11-21: knives spawned under
    # the hanging fingers were launched at reset)
    nm = env.robot.body_names
    hid = [i for i, b in enumerate(nm) if not (b.startswith("iiwa14_link_") and b != "iiwa14_link_7")]
    st["hand_pts0"] = (env.robot.data.body_pos_w[:, hid] - env.scene.env_origins.unsqueeze(1)).clone()   # (N, H, 3)
    env._collect = st
    print(f"[collect] slots {types}, N range {n_range}, bin rim {st['rim']*1000:.1f} mm, "
          f"inner floor {[round(float(x)*1000, 1) for x in st['inner_floor']]} mm", flush=True)


def _n_hi_now(env) -> int:
    c, st = env.cfg.precise_assembly, env._collect
    steps = int(c.collect_n_curriculum_steps)
    if steps <= 0:
        return st["n_hi"]
    a = min(1.0, float(env.common_step_counter + int(c.collect_n_curriculum_offset_steps)) / steps)
    lo = max(st["n_lo"], min(int(c.collect_n_start_max), st["n_hi"]))
    return int(round(lo + a * (st["n_hi"] - lo)))


def _rect_overlap(c1, y1, h1, c2, y2, h2) -> torch.Tensor:
    """2D oriented rectangles overlap (separating axis). c (..., 2), yaw (...), half extents h (..., 2)."""
    def axes(y):
        return torch.stack([torch.stack([torch.cos(y), torch.sin(y)], -1),
                            torch.stack([-torch.sin(y), torch.cos(y)], -1)], -2)    # (..., 2, 2) rows = axes
    a1, a2 = axes(y1), axes(y2)
    d = c2 - c1
    sep = torch.zeros(d.shape[:-1], dtype=torch.bool, device=d.device)
    for A in (a1, a2):
        for k in range(2):
            ax = A[..., k, :]
            r1 = (h1[..., 0] * (a1[..., 0, :] * ax).sum(-1).abs() + h1[..., 1] * (a1[..., 1, :] * ax).sum(-1).abs())
            r2 = (h2[..., 0] * (a2[..., 0, :] * ax).sum(-1).abs() + h2[..., 1] * (a2[..., 1, :] * ax).sum(-1).abs())
            sep |= (d * ax).sum(-1).abs() > r1 + r2
    return ~sep


def reset(env, env_ids: torch.Tensor) -> None:
    if not hasattr(env, "_collect"):
        setup(env)
    st, c = env._collect, env.cfg.precise_assembly
    dev, K, n = env.device, st["K"], env_ids.numel()
    if n == 0:
        return
    # episode-end snapshot (read by evals after the auto-reset)
    last = st.setdefault("last", {k: torch.zeros(env.num_envs, device=dev) for k in
                                  ("n_active", "n_in", "n_in_ep_max", "n_held_max", "scoop_max", "success")})
    if "n_in" in st:
        last["n_active"][env_ids] = st["n_active"][env_ids].float()
        last["n_in"][env_ids] = st["n_in"][env_ids].float()
        last["n_in_ep_max"][env_ids] = st["n_in_ep_max"][env_ids].float()
        last["n_held_max"][env_ids] = st["n_held_max"][env_ids].float()
        last["scoop_max"][env_ids] = st["scoop_max"][env_ids].float()
        last["success"][env_ids] = st["success"][env_ids].float()
    # number of active utensils
    hi = max(st["n_lo"], min(_n_hi_now(env), K))
    n_act = torch.randint(st["n_lo"], hi + 1, (n,), device=dev)
    active = torch.zeros(n, K, dtype=torch.bool, device=dev)
    active[:, 0] = True
    if K > 1:
        r = torch.rand(n, K - 1, device=dev)
        rank = r.argsort(dim=1).argsort(dim=1)                      # random rank of each other slot
        active[:, 1:] = rank < (n_act - 1).unsqueeze(1)
    # bin footprint (grown) in the table frame
    hq = env.hole_quat_wxyz[env_ids]
    bin_yaw = 2.0 * torch.atan2(hq[:, 3], hq[:, 0])
    bin_c = env.hole_pos[env_ids, :2]
    bin_h = (st["outer"] + float(c.collect_bin_margin_m)).expand(n, 2)
    x0, x1, y0, y1 = (float(v) for v in c.collect_region)
    T = 128
    placed_c = torch.zeros(n, K, 2, device=dev)
    placed_y = torch.zeros(n, K, device=dev)
    placed = torch.zeros(n, K, dtype=torch.bool, device=dev)
    gap = float(c.collect_gap_m) / 2
    grid = c.collect_grid
    if grid:
        gx, gy = int(grid[0]), int(grid[1])
        if gx * gy < K:
            raise ValueError(f"collect_grid {gx}x{gy} < {K} slots")
        ii = torch.arange(gx * gy, device=dev)
        cx = x0 + (x1 - x0) * ((ii % gx).float() + 0.5) / gx
        cy = y0 + (y1 - y0) * ((ii // gx).float() + 0.5) / gy
        key = torch.rand(n, gx * gy, device=dev)
        if bool(getattr(c, "collect_grid_contiguous", False)):
            # A18 11-23: the active knives fill one run of adjacent cells with a random start (a pile, not scattered)
            C = gx * gy
            s0 = (torch.rand(n, device=dev) * (C - n_act + 1).float()).floor().long().clamp(min=0)
            run = (ii[None, :] >= s0[:, None]) & (ii[None, :] < (s0 + n_act)[:, None])
            key = key - 2.0 * run.float()
        hc0 = float(c.collect_hand_clear_m)
        if hc0 > 0.0:                                     # cells whose centre lies near the start-pose hand go last
            hp0 = st["hand_pts0"][env_ids]
            low0 = hp0[..., 2] < (env._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT + 0.10).unsqueeze(1)
            cc0 = torch.stack([cx, cy], -1)                                              # (C, 2)
            d0 = hp0[:, None, :, :2] - cc0[None, :, None, :]                             # (n, C, H, 2), knife along y
            h0 = st["half"][0]
            near = ((d0[..., 0].abs() <= h0[1] + hc0 + 0.01) & (d0[..., 1].abs() <= h0[0] + hc0 + 0.01) & low0[:, None, :]).any(-1)
            key = key + near.float()
        order = key.argsort(1)                                                           # cells, best (far from hand) first
        # A18 11-22: active slots take the best cells first (by slot index), so a near-hand cell only goes to an active
        # knife when no free cell is left; before, inactive slots could take free cells and active ones got parked.
        arank = ((~active).long() * K + torch.arange(K, device=dev)).argsort(1).argsort(1)   # rank of each slot, active first
        cell = torch.gather(order, 1, arank)                                             # (n, K) distinct cell per slot
        jx, jy, jyaw = (float(v) for v in c.collect_grid_jitter)
    ok_any_k = []
    for k in range(K):
        if grid:
            cc = torch.stack([cx[cell[:, k]][:, None] + torch.empty(n, T, device=dev).uniform_(-jx, jx),
                              cy[cell[:, k]][:, None] + torch.empty(n, T, device=dev).uniform_(-jy, jy)], -1)
        else:
            cc = torch.stack([torch.empty(n, T, device=dev).uniform_(x0, x1),
                              torch.empty(n, T, device=dev).uniform_(y0, y1)], -1)   # footprint centres
        yc = bin_yaw[:, None] if bool(c.collect_yaw_rel_bin) else math.radians(c.collect_yaw_center_deg)
        yr = jyaw if grid else _yaw_range_now(env)
        yy = yc + torch.empty(n, T, device=dev).uniform_(-1, 1) * math.radians(yr)
        yy = yy + math.pi * (torch.rand(n, T, device=dev) < 0.5).float()            # either end first
        hk = (st["half"][k] + gap).expand(n, T, 2)
        bad = _rect_overlap(cc, yy, hk, bin_c[:, None].expand(n, T, 2), bin_yaw[:, None].expand(n, T),
                            bin_h[:, None].expand(n, T, 2))
        hc = float(c.collect_hand_clear_m)
        if hc > 0.0:
            hp = st["hand_pts0"][env_ids]                                               # (n, H, 3)
            low = hp[..., 2] < (env._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT + 0.10).unsqueeze(1)
            d = hp[:, None, :, :2] - cc[:, :, None, :]                                     # (n, T, H, 2)
            cy_, sy_ = torch.cos(yy)[..., None], torch.sin(yy)[..., None]
            lx = (d[..., 0] * cy_ + d[..., 1] * sy_).abs(); ly = (-d[..., 0] * sy_ + d[..., 1] * cy_).abs()
            h0 = st["half"][k]
            inside = (lx <= h0[0] + hc) & (ly <= h0[1] + hc) & low[:, None, :]
            bad |= inside.any(-1)
        for j in range(k):
            hj = (st["half"][j] + gap).expand(n, T, 2)
            ov = _rect_overlap(cc, yy, hk, placed_c[:, j:j + 1].expand(n, T, 2), placed_y[:, j:j + 1].expand(n, T), hj)
            bad |= ov & placed[:, j:j + 1]
        ok = ~bad
        first = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros(n, dtype=torch.long, device=dev))
        sel = torch.arange(n, device=dev)
        placed_c[:, k] = cc[sel, first]
        placed_y[:, k] = yy[sel, first]
        placed[:, k] = active[:, k] & (ok.any(1) | (k == 0))
        ok_any_k.append(ok.any(1))
    if grid and bool(getattr(c, "collect_grid_fallback", False)):
        # A18 11-22: an active knife whose cell was blocked (start-pose hand / overlap) tries every unused cell centre
        # (no jitter) before it is parked, so compact rows still get the full 6-10 knives
        used = torch.zeros(n, gx * gy, dtype=torch.bool, device=dev)
        for k in range(K):
            used.scatter_(1, cell[:, k:k + 1], (placed[:, k] & (ok_any_k[k] | (k == 0))).unsqueeze(1))
        hc_passes = [float(c.collect_hand_clear_m)]
        if bool(getattr(c, "collect_grid_fallback_relax", False)):
            hc_passes.append(0.0)                       # A18 11-23: 2nd pass, no clearance margin (footprint still off the hand)
        for k, hc_now in [(k, h) for h in hc_passes for k in range(1, K)]:
            miss = active[:, k] & ~placed[:, k]
            if not bool(miss.any()):
                continue
            for jc in range(gx * gy):
                ci = order[:, jc]                                               # cells best-first
                cand = miss & ~torch.gather(used, 1, ci[:, None]).squeeze(1)
                if not bool(cand.any()):
                    continue
                cc1 = torch.stack([cx[ci], cy[ci]], -1)[:, None]                 # (n, 1, 2)
                yc1 = bin_yaw[:, None] if bool(c.collect_yaw_rel_bin) else torch.full((n, 1), math.radians(c.collect_yaw_center_deg), device=dev)
                hk = (st["half"][k] + gap).expand(n, 1, 2)
                bad1 = _rect_overlap(cc1, yc1, hk, bin_c[:, None], bin_yaw[:, None], bin_h[:, None])
                hc = hc_now
                if True:
                    hp = st["hand_pts0"][env_ids]
                    low = hp[..., 2] < (env._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT + 0.10).unsqueeze(1)
                    d = hp[:, None, :, :2] - cc1[:, :, None, :]
                    cy_, sy_ = torch.cos(yc1)[..., None], torch.sin(yc1)[..., None]
                    lx = (d[..., 0] * cy_ + d[..., 1] * sy_).abs(); ly = (-d[..., 0] * sy_ + d[..., 1] * cy_).abs()
                    h0 = st["half"][k]
                    bad1 |= ((lx <= h0[0] + hc) & (ly <= h0[1] + hc) & low[:, None, :]).any(-1)
                for j in range(K):
                    hj = (st["half"][j] + gap).expand(n, 1, 2)
                    ov = _rect_overlap(cc1, yc1, hk, placed_c[:, j:j + 1], placed_y[:, j:j + 1], hj)
                    bad1 |= ov & placed[:, j:j + 1] & (j != k)
                take = cand & ~bad1[:, 0]
                placed_c[:, k] = torch.where(take[:, None], cc1[:, 0], placed_c[:, k])
                placed_y[:, k] = torch.where(take, yc1[:, 0], placed_y[:, k])
                placed[:, k] |= take
                used.scatter_(1, ci[:, None], (torch.gather(used, 1, ci[:, None]).squeeze(1) | take).unsqueeze(1))
                miss = miss & ~take
                if not bool(miss.any()):
                    break
    active = placed                        # a slot that found no free spot is parked (slot 0 always placed)
    st["active"][env_ids] = active
    st["n_active"][env_ids] = active.sum(1)
    st["n_active0"][env_ids] = active.sum(1)
    origins = env.scene.env_origins[env_ids]
    table_top = env._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    for k, body in enumerate(st["bodies"]):
        qz = quat_from_angle_axis(placed_y[:, k], z_axis)
        q = quat_mul(qz, st["rest_q"][k].expand(n, 4))
        off = quat_apply(qz, torch.cat([st["cxy"][k], torch.zeros(1, device=dev)]).expand(n, 3))
        pos = torch.cat([placed_c[:, k], (table_top + st["rest_z"][k] + float(c.collect_drop_m)).unsqueeze(1)], 1) - off
        # parked: on the floor under the table, long axis along x, one lane per slot
        ppos = torch.stack([torch.zeros(n, device=dev), torch.full((n,), -0.16 + 0.045 * k, device=dev),
                            torch.full((n,), float(st["rest_z"][k]) + 0.003, device=dev)], 1)
        pq = st["rest_q"][k].expand(n, 4)
        act = active[:, k:k + 1]
        pose = torch.cat([torch.where(act, pos, ppos) + origins, torch.where(act, q, pq)], 1)
        body.write_root_pose_to_sim(pose, env_ids=env_ids)
        body.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        st["prev_pos"][env_ids, k] = pose[:, :3] - origins
        if k == 0:
            env._object_init_z[env_ids] = pose[:, 2] - origins[:, 2]
    if float(c.collect_prefill_frac) > 0.0 and int(c.collect_prefill_max) > 0:
        _prefill(env, env_ids, active, bin_yaw)
    if c.collect_bank_file and float(c.collect_bank_frac) > 0.0:
        _bank_reset(env, env_ids)
    st["inbin"][env_ids] = False
    st["lift_done"][env_ids] = False
    st["pinch_done"][env_ids] = False
    st["fresh"][env_ids] = True
    st["clear"][env_ids] = 0
    st["scoop_clear"][env_ids] = 0
    st["success"][env_ids] = False
    st["fell"][env_ids] = False
    st["n_in_ep_max"][env_ids] = 0
    st["n_held_max"][env_ids] = 0
    st["scoop_max"][env_ids] = 0
    st["since_contact"][env_ids] = 10 ** 6
    st["place_ep_max"][env_ids] = 0.0
    for k in ("trips", "prev_n_in", "multi_max", "visit_n", "prev_n_in_max", "deposit_max"):
        st[k][env_ids] = 0
    for k in ("near", "visit_dep", "trips_over"):
        st[k][env_ids] = False


def _scoop_frac(env) -> float:
    c = env.cfg.precise_assembly
    steps = int(c.collect_scoop_ramp_steps)
    if steps <= 0:
        return 1.0
    return min(1.0, float(env.common_step_counter + int(c.collect_scoop_ramp_offset_steps)) / steps)


def _scoop_clear_now(env) -> int:
    """Scoop curriculum: consecutive held steps needed ramp from collect_scoop_clear_start to collect_clear_steps."""
    c = env.cfg.precise_assembly
    hi = int(c.collect_clear_steps)
    lo = int(c.collect_scoop_clear_start)
    if lo <= 0:
        return hi
    return max(1, int(round(lo + (hi - lo) * _scoop_frac(env))))


def _scoop_h_now(env) -> float:
    """Scoop curriculum (A18 11-21): held height ramps from collect_scoop_h_start to collect_scoop_h."""
    c = env.cfg.precise_assembly
    lo = float(c.collect_scoop_h_start)
    hi = float(c.collect_scoop_h)
    return hi if lo < 0 else lo + (hi - lo) * _scoop_frac(env)


def _scoop_min_now(env) -> int:
    """Scoop curriculum: knives needed for success ramp from collect_scoop_min_start to collect_success_min_in."""
    c = env.cfg.precise_assembly
    hi = max(1, int(c.collect_success_min_in))
    lo = int(c.collect_scoop_min_start)
    if lo <= 0:
        return hi
    return int(round(lo + (hi - lo) * _scoop_frac(env)))


def _yaw_range_now(env) -> float:
    """Utensil start yaw half-range (deg). collect_yaw_start_range_deg >= 0 ramps it to collect_yaw_range_deg over
    collect_yaw_ramp_steps env steps (A18 11-21 curriculum: start with knives lying almost along the slot)."""
    c = env.cfg.precise_assembly
    hi = float(c.collect_yaw_range_deg)
    lo = float(c.collect_yaw_start_range_deg)
    steps = int(c.collect_yaw_ramp_steps)
    if lo < 0.0 or steps <= 0:
        return hi
    a = min(1.0, float(env.common_step_counter + int(c.collect_yaw_offset_steps)) / steps)
    return lo + (hi - lo) * a


def _prefill(env, env_ids, active, bin_yaw) -> None:
    """Reverse curriculum on the count (A18 11-21): a share collect_prefill_frac of resets start with p ~ U{1..pmax}
    active utensils already lying in the bin (dropped flat from just above the floor, side by side, then stacked), so
    fewer deposits reach collect_success_min_in. pmax ramps from collect_prefill_max to 0 over
    collect_prefill_ramp_steps env steps (0 = no ramp; offset collect_prefill_offset_steps)."""
    st, c, dev = env._collect, env.cfg.precise_assembly, env.device
    pmax = int(c.collect_prefill_max)
    steps = int(c.collect_prefill_ramp_steps)
    if steps > 0:
        a = min(1.0, float(env.common_step_counter + int(c.collect_prefill_offset_steps)) / steps)
        pmax = int(math.floor(pmax * (1.0 - a) + 0.5))
    if pmax <= 0:
        return
    n = env_ids.numel()
    use = torch.rand(n, device=dev) < float(c.collect_prefill_frac)
    p = torch.randint(1, pmax + 1, (n,), device=dev) * use.long()
    p = torch.minimum(p, (active.sum(1) - 1).clamp(min=0))       # never prefill every utensil (blind success)
    if int(p.max()) == 0:
        return
    K = st["K"]
    r = torch.rand(n, K, device=dev) + (~active).float() * 2.0            # inactive slots sort last
    rank = r.argsort(dim=1).argsort(dim=1)
    take = (rank < p.unsqueeze(1)) & active
    origins = env.scene.env_origins[env_ids]
    hp = env.hole_pos[env_ids]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    hw = float(st["inner_floor"][1])
    for k, body in enumerate(st["bodies"]):
        m = take[:, k]
        if not bool(m.any()):
            continue
        idx = rank[:, k].float()                                           # 0, 1, ... within the env
        side = torch.where(idx % 2 == 0, -1.0, 1.0)
        lat = side * max(hw - float(st["half"][k][1]) - 0.002, 0.0)
        layer = torch.floor(idx / 2)
        yaw = bin_yaw + math.pi * (torch.rand(n, device=dev) < 0.5).float()
        qz = quat_from_angle_axis(yaw, z_axis)
        q = quat_mul(qz, st["rest_q"][k].expand(n, 4))
        qb = quat_from_angle_axis(bin_yaw, z_axis)
        cpos = hp + quat_apply(qb, torch.stack([torch.zeros(n, device=dev), lat, torch.zeros(n, device=dev)], 1))
        off = quat_apply(qz, torch.cat([st["cxy"][k], torch.zeros(1, device=dev)]).expand(n, 3))
        z = hp[:, 2] + st["floor"] + st["rest_z"][k] + 0.004 + 0.014 * layer
        pos = torch.cat([cpos[:, :2], z.unsqueeze(1)], 1) - off
        pose = torch.cat([pos + origins, q], 1)
        ids = env_ids[m]
        body.write_root_pose_to_sim(pose[m], env_ids=ids)
        body.write_root_velocity_to_sim(torch.zeros(int(m.sum()), 6, device=dev), env_ids=ids)
        st["prev_pos"][ids, k] = pos[m]
        if k == 0:
            env._object_init_z[ids] = pos[m, 2]


BANK_KEYS = ("joint_pos", "joint_vel", "cur_targets", "prev_targets", "hole_pos", "hole_quat", "ut_pos", "ut_quat",
             "active")


def _bank_reset(env, env_ids: torch.Tensor) -> None:
    """collect_bank_file: a share collect_bank_frac of resets start from a recorded state (robot joints + targets, bin
    pose, every utensil pose, active mask; record_collect_bank.py), e.g. a utensil already held above the table."""
    st, c, dev = env._collect, env.cfg.precise_assembly, env.device
    if "bank" not in st:
        import numpy as np
        d = np.load(c.collect_bank_file)
        st["bank"] = {k: torch.as_tensor(d[k], device=dev) for k in BANK_KEYS}
        if st["bank"]["ut_pos"].shape[1] != st["K"]:
            raise ValueError(f"collect bank has {st['bank']['ut_pos'].shape[1]} slots, env {st['K']}")
        print(f"[collect] bank {c.collect_bank_file}: {len(d['active'])} states, share {c.collect_bank_frac}", flush=True)
    b = st["bank"]
    sel = env_ids[torch.rand(env_ids.numel(), device=dev) < float(c.collect_bank_frac)]
    n = sel.numel()
    if n == 0:
        return
    k = torch.randint(0, b["active"].shape[0], (n,), device=dev)
    origins = env.scene.env_origins[sel]
    env.hole_pos[sel] = b["hole_pos"][k].float()
    env.hole_quat_wxyz[sel] = b["hole_quat"][k].float()
    env.hole.write_root_pose_to_sim(torch.cat([env.hole_pos[sel] + origins, env.hole_quat_wxyz[sel]], 1), env_ids=sel)
    env.hole.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=sel)
    env.robot.write_joint_state_to_sim(b["joint_pos"][k].float(), b["joint_vel"][k].float(), env_ids=sel)
    env._cur_targets[sel] = b["cur_targets"][k].float()
    env._prev_targets[sel] = b["prev_targets"][k].float()
    act = b["active"][k].bool()
    act[:, 0] = True
    for j, body in enumerate(st["bodies"]):
        pos = b["ut_pos"][k, j].float()
        body.write_root_pose_to_sim(torch.cat([pos + origins, b["ut_quat"][k, j].float()], 1), env_ids=sel)
        body.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=sel)
        st["prev_pos"][sel, j] = pos
        if j == 0:
            env._object_init_z[sel] = pos[:, 2]
    st["active"][sel] = act
    st["n_active"][sel] = act.sum(1)
    st["n_active0"][sel] = act.sum(1)


def bank_snapshot(env, ids: torch.Tensor) -> dict:
    """The state of envs ``ids`` in the bank format (all local to the env origin)."""
    st = env._collect
    pos, quat, _ = _state(env)
    return dict(joint_pos=env.robot.data.joint_pos[ids], joint_vel=env.robot.data.joint_vel[ids],
                cur_targets=env._cur_targets[ids], prev_targets=env._prev_targets[ids],
                hole_pos=env.hole_pos[ids], hole_quat=env.hole_quat_wxyz[ids], ut_pos=pos[ids], ut_quat=quat[ids],
                active=st["active"][ids])


def _state(env):
    st = env._collect
    o = env.scene.env_origins.unsqueeze(1)
    pos = torch.stack([b.data.root_pos_w for b in st["bodies"]], 1) - o            # (N, K, 3)
    quat = torch.stack([b.data.root_quat_w for b in st["bodies"]], 1)              # (N, K, 4)
    vel = torch.stack([b.data.root_lin_vel_w for b in st["bodies"]], 1)
    return pos, quat, vel


def target_state(env, palm_pos: torch.Tensor):
    """collect_scoop_target: the virtual object the policy's object fields show = the pile of utensils still out of the
    bin. Position = mean of the out-of-bin active utensils (all active if none left), orientation = that of the
    out-of-bin utensil nearest the palm, velocity = mean. Env-local frame (like the obs builder's obj_pos)."""
    st = env._collect
    pos, quat, vel = _state(env)
    out = st["active"] & ~st["inbin"]
    m = torch.where(out.any(1, keepdim=True), out, st["active"]).float()                 # (N, K)
    w = m / m.sum(1, keepdim=True).clamp_min(1.0)
    c = (pos * w.unsqueeze(-1)).sum(1)
    v = (vel * w.unsqueeze(-1)).sum(1)
    d = (pos - palm_pos.unsqueeze(1)).norm(dim=-1) + (1.0 - m) * 10.0
    q = torch.gather(quat, 1, d.argmin(1)[:, None, None].expand(-1, 1, 4)).squeeze(1)
    return c, q, v


def update(env) -> None:
    """Per-step: in-bin / rest / contact / success / fell + the shaped reward (stored for _get_rewards)."""
    st, c = env._collect, env.cfg.precise_assembly
    N, K, P = env.num_envs, st["K"], N_PTS
    pos, quat, vel = _state(env)
    active = st["active"]
    n_gone = torch.zeros(N, dtype=torch.long, device=env.device)
    if not bool(c.collect_fell_terminal):
        # user 11-20 (>= 3 of ~10): a utensil knocked off the table just leaves play (one-time penalty), no episode end
        tt = (env._table_z_per_env + TABLE_HALF_HEIGHT).unsqueeze(1)
        gone = (pos[..., 2] < tt - 0.05) & active
        n_gone = gone.sum(1)
        st["active"] = active & ~gone
        active = st["active"]
        st["n_active"] = active.sum(1)
    # lowest point of each utensil, in the bin frame
    q = quat.unsqueeze(2).expand(N, K, P, 4).reshape(-1, 4)
    pts = quat_apply(q, st["pts"].unsqueeze(0).expand(N, K, P, 3).reshape(-1, 3)).view(N, K, P, 3) + pos.unsqueeze(2)
    low_i = pts[..., 2].argmin(-1)
    low = torch.gather(pts, 2, low_i[..., None, None].expand(N, K, 1, 3)).squeeze(2)    # (N, K, 3)
    hq = env.hole_quat_wxyz.unsqueeze(1).expand(N, K, 4).reshape(-1, 4)
    rel = quat_apply_inverse(hq, (low - env.hole_pos.unsqueeze(1)).reshape(-1, 3)).view(N, K, 3)
    zb = rel[..., 2]
    u = ((torch.minimum(zb, torch.full_like(zb, st["rim"])) - 0.003) / (st["rim"] - 0.004)).clamp(0, 1)
    hw = st["inner_floor"] + u.unsqueeze(-1) * (st["inner_rim"] - st["inner_floor"])
    inside = (rel[..., 0].abs() < hw[..., 0] + 0.001) & (rel[..., 1].abs() < hw[..., 1] + 0.001) \
        & (zb < st["rim"] + st["above_rim"])
    inbin = inside & active
    st["inbin"] = inbin
    n_in = inbin.sum(1)
    st["n_in_ep_max"] = torch.maximum(st["n_in_ep_max"], n_in)
    # at rest = COM speed < 5 mm/s, measured as displacement over the last policy step: PhysX's reported velocity of
    # bodies resting in a stack reads 1-4 cm/s while they move < 0.1 mm in 2 s (probe 11-20), so it can't be used.
    step_dt = float(env.physics_dt) * int(env.cfg.decimation)
    rest = (pos - st["prev_pos"]).norm(dim=-1) / step_dt < st["rest_speed"]
    st["prev_pos"] = pos.clone()
    st["at_rest"] = rest
    all_in = (inbin | ~active).all(1)
    all_rest = (rest | ~active).all(1)
    # contact: fingertip pads + whole-hand skin (both ray-cast against every utensil)
    thr = float(c.collect_contact_thresh)
    contact = torch.zeros(N, dtype=torch.bool, device=env.device)
    for k in ("_last_tactile", "_last_tactile_hand"):
        t = getattr(env, k, None)
        if t is not None:
            contact |= (t.reshape(N, -1) > thr).any(-1)
    palm_v = env.robot.data.body_lin_vel_w[:, env._palm_body_id].norm(dim=-1)
    m_in = int(c.collect_success_min_in)
    if m_in > 0:   # user 11-20: success = >= m_in utensils in the bin (those at rest), not all of them
        rest_in = (rest | ~inbin).all(1)
        enough = n_in >= torch.clamp(st["n_active0"], max=m_in)   # count at reset: knocking utensils off can't help
        ok = enough & rest_in & ~contact & (palm_v < float(c.collect_palm_speed))
    else:
        ok = all_in & all_rest & ~contact & (palm_v < float(c.collect_palm_speed))
    st["clear"] = torch.where(ok, st["clear"] + 1, torch.zeros_like(st["clear"]))
    st["success"] = st["clear"] >= int(c.collect_clear_steps)
    table_top = (env._table_z_per_env + TABLE_HALF_HEIGHT).unsqueeze(1)
    st["fell"] = ((pos[..., 2] < table_top - 0.05) & active).any(1)
    # utensils off the table but held up (> 2 cm) = carried; count simultaneous carries
    h = (pos[..., 2] - table_top - st["rest_z"]).clamp(min=0.0)
    out = active & ~inbin
    lifted = (h > 0.02) & out
    st["n_held_max"] = torch.maximum(st["n_held_max"], lifted.sum(1))
    st["lifted"] = lifted
    # scoop mode (A18 user 11-21: pick up as many knives as possible at once, no bin): a utensil is held when its COM
    # is >= collect_scoop_h above its rest height while the hand touches utensils (pads / skin; a knife 10 cm up is
    # either in the hand or resting on knives that are). Success = >= collect_success_min_in held for
    # collect_clear_steps consecutive steps.
    # tactile readings of knives resting on finger sides sit at the threshold and flicker on/off every step (scoop
    # check 11-21: median contact run 1 step while the knife stayed up), so "touching" = contact within the last
    # collect_scoop_contact_mem steps
    st["since_contact"] = torch.where(contact, torch.zeros_like(st["since_contact"]), st["since_contact"] + 1)
    touch_recent = st["since_contact"] <= int(c.collect_scoop_contact_mem)
    if float(c.collect_scoop_hand_d) > 0.0:
        # geometric "in the hand" (scoop check v3: tactile on knives held against finger sides reads ~never): some
        # surface point of the utensil within collect_scoop_hand_d of a fingertip or the palm, and COM slower than
        # collect_scoop_max_v (a knife 10 cm up for 30 steps at low speed is not falling or thrown)
        o = env.scene.env_origins.unsqueeze(1)
        if "hand_body_ids" not in st:   # every hand link (iiwa links 0-6 excluded; the palm is merged into link 7)
            nm = env.robot.body_names
            st["hand_body_ids"] = [i for i, n in enumerate(nm)
                                   if not (n.startswith("iiwa14_link_") and n != "iiwa14_link_7")]
        hp = env.robot.data.body_state_w[:, st["hand_body_ids"], 0:3] - o                               # (N, B, 3)
        dh = (pts.unsqueeze(2) - hp[:, None, :, None, :]).norm(dim=-1).amin(dim=(2, 3))                 # (N, K)
        near_hand = (dh < float(c.collect_scoop_hand_d)) & (vel.norm(dim=-1) < float(c.collect_scoop_max_v))
        held_s = active & (h >= _scoop_h_now(env)) & near_hand
    else:
        held_s = active & (h >= _scoop_h_now(env)) & touch_recent.unsqueeze(1)
    n_scoop = held_s.sum(1)
    st["n_scoop"] = n_scoop
    st["scoop_max"] = torch.maximum(st["scoop_max"], n_scoop)
    if bool(c.collect_scoop_mode):
        # own counter: the bin-rule block above rewrites st["clear"] to 0 every step in scoop mode (no utensil is in
        # a bin), so sharing it capped the held run at 1 step and no scoop episode could ever succeed (bug, 11-21)
        ok_s = n_scoop >= _scoop_min_now(env)
        st["scoop_clear"] = torch.where(ok_s, st["scoop_clear"] + 1, torch.zeros_like(st["scoop_clear"]))
        st["clear"] = st["scoop_clear"]
        st["success"] = st["scoop_clear"] >= _scoop_clear_now(env)
    # potential
    hn = (h / float(c.collect_lift_h)).clamp(max=1.0)
    d_bin = (pos[..., :2] - env.hole_pos[:, None, :2]).norm(dim=-1)
    carry = (1.0 - d_bin / float(c.collect_carry_d)).clamp(0, 1) * lifted.float()
    if float(c.collect_carry_align_pow) > 0.0:
        # A18 11-21: the carry potential only pays in full for a utensil lined up with the slot (it saturated ~0.9 at
        # 35 mm from the bin at any yaw, so hovering beside the bin with a crossed knife earned almost everything)
        hq0 = env.hole_quat_wxyz.unsqueeze(1).expand(N, K, 4).reshape(-1, 4)
        lx = quat_apply(quat.reshape(-1, 4), st["long_ax"].unsqueeze(0).expand(N, K, 3).reshape(-1, 3))
        lx = quat_apply_inverse(hq0, lx).view(N, K, 3)
        ca = (lx[..., 0].abs() / lx[..., :2].norm(dim=-1).clamp_min(1e-6)) ** float(c.collect_carry_align_pow)
        fl = float(c.collect_carry_align_floor)
        carry = carry * (fl + (1.0 - fl) * ca)
    phi = (float(c.collect_rew_in) * inbin.float()
           + out.float() * (float(c.collect_rew_lift) * hn + float(c.collect_rew_carry) * carry)).sum(1)
    ft = env.robot.data.body_state_w[:, env._fingertip_body_ids, 0:3] - env.scene.env_origins.unsqueeze(1)   # (N,5,3)
    dft = (ft.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1).min(dim=1).values                                # (N, K)
    big = torch.full_like(dft, 10.0)
    d_out = torch.where(out, dft, big).min(1).values
    any_out = out.any(1)
    reach = torch.where(any_out, (1.0 - d_out / float(c.collect_reach_d)).clamp(0, 1), torch.ones_like(d_out))
    phi = phi + float(c.collect_rew_reach) * reach
    # place (A18 11-21): a held utensil lined up with the slot = long axis along the bin's long axis, centre over the
    # slot, low above the rim. Knives (197 mm) fit the 213 x 50 mm trough only within ~10 deg; the carry term alone
    # brought them near the bin at any yaw (bank_s7_overbin: median yaw error 25 deg, 35 mm off sideways).
    if float(c.collect_rew_place) > 0.0:
        lax = quat_apply(quat.reshape(-1, 4), st["long_ax"].unsqueeze(0).expand(N, K, 3).reshape(-1, 3)).view(N, K, 3)
        lax_b = quat_apply_inverse(hq, lax.reshape(-1, 3)).view(N, K, 3)
        align = (lax_b[..., 0].abs() / lax_b[..., :2].norm(dim=-1).clamp_min(1e-6)) ** float(c.collect_place_align_pow)
        com_b = quat_apply_inverse(hq, (pos - env.hole_pos.unsqueeze(1)).reshape(-1, 3)).view(N, K, 3)
        along = (1.0 - com_b[..., 0].abs() / float(c.collect_place_along_d)).clamp(0, 1)
        across = (1.0 - com_b[..., 1].abs() / float(c.collect_place_across_d)).clamp(0, 1)
        low_t = (1.0 - (zb - st["rim"]).clamp(min=0.0) / float(c.collect_place_h)).clamp(0, 1)
        if str(c.collect_place_mode) == "sum":
            # each factor pulls on its own (the product had ~0 gradient while any factor was small); gated on being
            # within collect_carry_d of the bin so it can't pay away from it
            near_g = (1.0 - d_bin / float(c.collect_carry_d)).clamp(0, 1)
            place = 0.25 * (align + along + across + low_t) * near_g * lifted.float()
            full = align * along * across * low_t * lifted.float()
        else:
            place = align * along * across * low_t * lifted.float()
            full = place
        st["place_ep_max"] = torch.maximum(st["place_ep_max"], full.max(1).values)
        phi = phi + float(c.collect_rew_place) * place.sum(1)
    # pinch: thumb pad and another fingertip pad in contact, both tips within collect_pinch_d of the same out-of-bin
    # utensil's surface points (tactile is ray-cast against every utensil, so the distance check picks the utensil)
    pinch = torch.zeros(N, dtype=torch.bool, device=env.device)
    if float(c.collect_rew_pinch) > 0.0 or float(c.collect_pinch_bonus) > 0.0:
        tt = getattr(env, "_last_tactile", None)
        if tt is not None:
            pad = tt.reshape(N, 5, -1).amax(-1) > thr                                                  # (N, 5)
            dall = (ft[:, :, None, None, :] - pts.unsqueeze(1)).norm(dim=-1).amin(-1)                  # (N, 5, K)
            near = dall < float(c.collect_pinch_d)
            th = pad[:, 0:1] & near[:, 0]                                                               # (N, K)
            oth = (pad[:, 1:].unsqueeze(-1) & near[:, 1:]).any(1)                                        # (N, K)
            pinch = (th & oth & out).any(1)
    st["pinch"] = pinch
    phi = phi + float(c.collect_rew_pinch) * pinch.float()
    d_act = torch.where(active, dft, big).min(1).values
    release = all_in.float() * (d_act / float(c.collect_release_d)).clamp(0, 1)
    phi = phi + float(c.collect_rew_release) * release
    phi = phi + float(c.collect_rew_multi) * (lifted.sum(1) - 1).clamp(min=0).float()   # scooping: carried together
    phi = phi + float(c.collect_rew_scoop) * n_scoop.float()   # scoop mode: per utensil held high
    if bool(c.collect_scoop_mode) and float(c.collect_rew_hold) > 0.0:
        # hold-time potential (scp2: knives were picked up and dropped within 30 steps, nothing paid for holding on)
        phi = phi + float(c.collect_rew_hold) * (st["scoop_clear"].float() / float(_scoop_clear_now(env))).clamp(max=1.0)
    if bool(getattr(c, "collect_phi_ratchet", False)):
        # A18 11-22: pay only new episode highs of the potential, so putting a 2-knife bundle down to retry costs
        # nothing (with plain phi - prev_phi a drop costs ~1400 and the policy held 2 knives to the timeout)
        pm = torch.where(st["fresh"], phi, st["phi_max"])
        r = torch.where(st["fresh"], torch.zeros_like(phi), (phi - pm).clamp(min=0.0))
        st["phi_max"] = torch.maximum(pm, phi)
    else:
        r = torch.where(st["fresh"], torch.zeros_like(phi), phi - st["prev_phi"])
    tp = float(getattr(c, "collect_time_penalty", 0.0))
    if tp > 0.0:
        # A18 11-23: per-step cost after collect_time_penalty_after steps (det fails hold 2 knives up to the timeout)
        late = env.episode_length_buf >= int(getattr(c, "collect_time_penalty_after", 300))
        r = r - tp * (late & ~st["fresh"]).float()
    # one-time bonus the first time each utensil is lifted (> 2 cm, out of the bin) in an episode
    new_lift = lifted & ~st["lift_done"] & ~st["fresh"].unsqueeze(1)
    st["lift_done"] |= lifted
    r = r + float(c.collect_lift_bonus) * new_lift.float().sum(1)
    new_pinch = pinch & ~st["pinch_done"] & ~st["fresh"]
    st["pinch_done"] |= pinch
    r = r + float(c.collect_pinch_bonus) * new_pinch.float()
    # scooping: one-time bonus per new simultaneous-carry maximum (2nd, 3rd ... utensil held at once)
    n_lift = lifted.sum(1)
    new_max = torch.maximum(st["multi_max"], n_lift)
    gain = (new_max.clamp(min=1) - st["multi_max"].clamp(min=1)).clamp(min=0)
    r = r + float(c.collect_multi_bonus) * gain.float() * (~st["fresh"]).float()
    st["multi_max"] = new_max
    # trips
    palm = env.robot.data.body_pos_w[:, env._palm_body_id] - env.scene.env_origins
    near = (palm[:, :2] - env.hole_pos[:, :2]).norm(dim=-1) < float(c.collect_trip_near_m)
    entered = near & ~st["near"]
    st["visit_dep"] = st["visit_dep"] & ~entered
    rose = (n_in > st["prev_n_in"]) & ~st["fresh"]
    dep = rose & (~near | ~st["visit_dep"])
    st["trips"] = st["trips"] + dep.long()
    # deposit-size bonus: the k-th utensil landing in the same bin visit earns (k-1) x collect_deposit_bonus
    gain_in = (st["n_in_ep_max"] - st["prev_n_in_max"]).clamp(min=0)
    st["visit_n"] = torch.where(entered | ~near, torch.zeros_like(st["visit_n"]), st["visit_n"])
    if float(c.collect_deposit_bonus) > 0.0:
        k0 = st["visit_n"].float()
        g = gain_in.float()
        r = r + float(c.collect_deposit_bonus) * (g * k0 + g * (g - 1) / 2) * (~st["fresh"]).float()
    st["visit_n"] = st["visit_n"] + gain_in
    st["deposit_max"] = torch.maximum(st["deposit_max"], st["visit_n"] + (gain_in * (~near).long()))
    st["prev_n_in_max"] = st["n_in_ep_max"].clone()
    st["visit_dep"] = st["visit_dep"] | (rose & near)
    st["near"] = near
    st["prev_n_in"] = n_in
    if int(c.collect_per_trip) > 0:
        budget = (st["n_active"] + int(c.collect_per_trip) - 1) // int(c.collect_per_trip)
        st["trips_over"] = st["trips"] > budget
    st["prev_phi"] = phi
    st["fresh"][:] = False
    r = r - float(c.collect_fall_penalty) * n_gone.float() * (~st["fresh"]).float()
    st["rew"] = r
    st["n_in"] = n_in
    st["contact"] = contact


def obs(env, palm_pos: torch.Tensor) -> torch.Tensor:
    """collect_obs: per slot [pos rel palm 3, pos in bin frame 3, long axis (world) 3, active, in bin] padded to
    collect_obs_slots, then [bin centre rel palm 3, cos / sin bin yaw]."""
    st = env._collect
    N, K = env.num_envs, st["K"]
    S = int(env.cfg.precise_assembly.collect_obs_slots)
    pos, quat, _ = _state(env)
    hq = env.hole_quat_wxyz
    rel_b = quat_apply_inverse(hq.unsqueeze(1).expand(N, K, 4).reshape(-1, 4),
                               (pos - env.hole_pos.unsqueeze(1)).reshape(-1, 3)).view(N, K, 3)
    ax = quat_apply(quat.reshape(-1, 4), torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(N * K, 3)).view(N, K, 3)
    a = st["active"].float().unsqueeze(-1)
    slot = torch.cat([pos - palm_pos.unsqueeze(1), rel_b, ax, a, st["inbin"].float().unsqueeze(-1)], -1) * a
    if S > K:
        slot = torch.cat([slot, torch.zeros(N, S - K, slot.shape[-1], device=env.device)], 1)
    yaw = 2.0 * torch.atan2(hq[:, 3], hq[:, 0])
    return torch.cat([slot.reshape(N, -1), env.hole_pos - palm_pos, torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]], -1)


def obs_size(slots: int) -> int:
    return 11 * slots + 5
