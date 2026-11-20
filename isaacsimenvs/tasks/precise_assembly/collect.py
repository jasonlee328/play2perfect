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
    n_range = tuple(int(v) for v in (c.collect_n_range or spec["n_range"]))
    st = dict(
        bodies=bodies, K=K, types=types,
        rest_q=torch.tensor(rest_q, device=dev), rest_z=torch.tensor(rest_z, device=dev),
        half=torch.tensor(half, device=dev), cxy=torch.tensor(cxy, device=dev),
        pts=torch.stack(pts).to(dev),
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
    st["prev_phi"] = torch.zeros(N, device=dev)
    st["fresh"] = torch.ones(N, dtype=torch.bool, device=dev)
    st["clear"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["success"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["fell"] = torch.zeros(N, dtype=torch.bool, device=dev)
    st["n_in_ep_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["n_held_max"] = torch.zeros(N, dtype=torch.long, device=dev)
    st["prev_pos"] = torch.zeros(N, K, 3, device=dev)
    st["at_rest"] = torch.zeros(N, K, dtype=torch.bool, device=dev)
    st["lift_done"] = torch.zeros(N, K, dtype=torch.bool, device=dev)
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
                                  ("n_active", "n_in", "n_in_ep_max", "n_held_max", "success")})
    if "n_in" in st:
        last["n_active"][env_ids] = st["n_active"][env_ids].float()
        last["n_in"][env_ids] = st["n_in"][env_ids].float()
        last["n_in_ep_max"][env_ids] = st["n_in_ep_max"][env_ids].float()
        last["n_held_max"][env_ids] = st["n_held_max"][env_ids].float()
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
    for k in range(K):
        cc = torch.stack([torch.empty(n, T, device=dev).uniform_(x0, x1),
                          torch.empty(n, T, device=dev).uniform_(y0, y1)], -1)       # footprint centres
        yy = (math.radians(c.collect_yaw_center_deg)
              + torch.empty(n, T, device=dev).uniform_(-1, 1) * math.radians(c.collect_yaw_range_deg))
        yy = yy + math.pi * (torch.rand(n, T, device=dev) < 0.5).float()            # either end first
        hk = (st["half"][k] + gap).expand(n, T, 2)
        bad = _rect_overlap(cc, yy, hk, bin_c[:, None].expand(n, T, 2), bin_yaw[:, None].expand(n, T),
                            bin_h[:, None].expand(n, T, 2))
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
    active = placed                        # a slot that found no free spot is parked (slot 0 always placed)
    st["active"][env_ids] = active
    st["n_active"][env_ids] = active.sum(1)
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
    if c.collect_bank_file and float(c.collect_bank_frac) > 0.0:
        _bank_reset(env, env_ids)
    st["inbin"][env_ids] = False
    st["lift_done"][env_ids] = False
    st["fresh"][env_ids] = True
    st["clear"][env_ids] = 0
    st["success"][env_ids] = False
    st["fell"][env_ids] = False
    st["n_in_ep_max"][env_ids] = 0
    st["n_held_max"][env_ids] = 0


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


def update(env) -> None:
    """Per-step: in-bin / rest / contact / success / fell + the shaped reward (stored for _get_rewards)."""
    st, c = env._collect, env.cfg.precise_assembly
    N, K, P = env.num_envs, st["K"], N_PTS
    pos, quat, vel = _state(env)
    active = st["active"]
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
    # potential
    hn = (h / float(c.collect_lift_h)).clamp(max=1.0)
    d_bin = (pos[..., :2] - env.hole_pos[:, None, :2]).norm(dim=-1)
    carry = (1.0 - d_bin / float(c.collect_carry_d)).clamp(0, 1) * lifted.float()
    phi = (float(c.collect_rew_in) * inbin.float()
           + out.float() * (float(c.collect_rew_lift) * hn + float(c.collect_rew_carry) * carry)).sum(1)
    ft = env.robot.data.body_state_w[:, env._fingertip_body_ids, 0:3] - env.scene.env_origins.unsqueeze(1)   # (N,5,3)
    dft = (ft.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1).min(dim=1).values                                # (N, K)
    big = torch.full_like(dft, 10.0)
    d_out = torch.where(out, dft, big).min(1).values
    any_out = out.any(1)
    reach = torch.where(any_out, (1.0 - d_out / float(c.collect_reach_d)).clamp(0, 1), torch.ones_like(d_out))
    phi = phi + float(c.collect_rew_reach) * reach
    d_act = torch.where(active, dft, big).min(1).values
    release = all_in.float() * (d_act / float(c.collect_release_d)).clamp(0, 1)
    phi = phi + float(c.collect_rew_release) * release
    r = torch.where(st["fresh"], torch.zeros_like(phi), phi - st["prev_phi"])
    # one-time bonus the first time each utensil is lifted (> 2 cm, out of the bin) in an episode
    new_lift = lifted & ~st["lift_done"] & ~st["fresh"].unsqueeze(1)
    st["lift_done"] |= lifted
    r = r + float(c.collect_lift_bonus) * new_lift.float().sum(1)
    st["prev_phi"] = phi
    st["fresh"][:] = False
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
