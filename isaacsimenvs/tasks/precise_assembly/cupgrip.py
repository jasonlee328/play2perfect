"""A21 cup grip window (cfg precise_assembly.cupgrip*): lift only the top cup off an upside-down stack.

Problem cb_cupgrip: object = the top inverted cup, distractor 0 = the loose second cup nested under it, fixture = the
fused cups below. Per episode (hidden from the actor, sampled at reset):
  S  nest retention (N), log-uniform in cupgrip_retain_range_N: a downward force on the top cup while it is still
     seated (breakaway: holds against the hand's upward pull beyond m g up to S, then frees; re-sticks when re-seated).
     Physical consequence: the pads must squeeze hard enough that friction carries S + m g, else the cup slips.
  r  drag ratio, uniform in cupgrip_drag_ratio_range: while the top cup is still nested on the second cup
     (rise < cupgrip_nest_overlap_mm), a hand squeeze (horizontal contact force, N) above D = r (S + m g) / cupgrip_mu_cal
     while the hand pulls the cup up
     latches a coupling that carries the second cup with the top cup (MODEL, not rigid-body physics: a rigid cup can't
     be squeezed onto the one inside). The coupling force F on the second cup (gravity compensation + PD on the
     nested offset) is applied with -F on the top cup, so the hand carries the extra load. Released when the squeeze
     drops below 0.9 D.
  k  hand PD gain scale (cupgrip_hand_kp_range; stiffness x k, damping x sqrt k), so joint tracking error does not
     read grip force.
Failures: "dragged" = the second cup rose > cupgrip_drag_fail_mm; "slipped" = while gripping (hand force >=
cupgrip_contact_N) the palm slid > cupgrip_slip_mm in total up the top cup while the cup was still nested.
Window mode (cupgrip_mode=window, 11-20 redesign): the physical breakaway (above) leaks to proprio - the hand / arm
carry up to S + m g while the cup is stuck and the load drops at breakaway, so "squeeze until the pull drops" works
without touch (probe: joint state AUC 0.92 vs touch 0.72 for "cup lags the rising palm"). In window mode there is no
retention; the hidden minimum squeeze G_lo (log-U cupgrip_glo_range_N) and D = r G_lo define the window; lifting the
nested cup with squeeze < G_lo for cupgrip_slip_steps steps = "slipped" (MODEL); touch-only incipient-slip cue
(tactile_cue) below cupgrip_cue_ratio x G_lo.
Actor obs (cupgrip_hide_object): object_rot / keypoints_rel_palm / keypoints_rel_goal are computed from the top cup's
reset pose, so the actor knows where the cup started but not whether it is moving with the hand (only touch tells).
"""

from __future__ import annotations

import math

import torch

G = 9.81


def enabled(env) -> bool:
    return bool(getattr(env.cfg.precise_assembly, "cupgrip", False))


def _init(env) -> None:
    n, d = env.num_envs, env.device
    env.cg_S = torch.zeros(n, device=d)
    env.cg_r = torch.ones(n, device=d)
    env.cg_D = torch.zeros(n, device=d)
    env.cg_kp = torch.ones(n, device=d)
    env.cg_coupled = torch.zeros(n, dtype=torch.bool, device=d)
    env.cg_stuck = torch.ones(n, dtype=torch.bool, device=d)
    env.cg_G = torch.zeros(n, device=d)
    env.cg_low = torch.zeros(n, dtype=torch.long, device=d)
    env.cg_break_count = torch.zeros(n, dtype=torch.long, device=d)
    env.cg_ever_coupled = torch.zeros(n, dtype=torch.bool, device=d)
    env.cg_grasped = torch.zeros(n, dtype=torch.bool, device=d)
    env.cg_rel0 = torch.zeros(n, device=d)
    env.cg_slide = torch.zeros(n, device=d)
    env.cg_rise = torch.zeros(n, device=d)
    env.cg_rise2 = torch.zeros(n, device=d)
    env.cg_obj0 = torch.zeros(n, 7, device=d)
    env.cg_z2_0 = torch.zeros(n, device=d)
    env.cg_off = torch.zeros(n, device=d)
    env.cg_hand = torch.zeros(n, device=d)
    m = env.object.root_physx_view.get_masses().to(d).sum(-1)
    env.cg_m1 = m
    env.cg_m2 = env.distractors[0].root_physx_view.get_masses().to(d).sum(-1)
    ids, _ = env.robot.find_joints("left_.*")
    env._cg_hand_ids = torch.as_tensor(ids, device=d, dtype=torch.long)
    env._cg_kp0 = env.robot.data.default_joint_stiffness[:, env._cg_hand_ids].clone()
    env._cg_kd0 = env.robot.data.default_joint_damping[:, env._cg_hand_ids].clone()


def reset(env, env_ids: torch.Tensor) -> None:
    """After the object / distractors are placed (end of _reset_peg_episode)."""
    if not enabled(env):
        return
    if not hasattr(env, "cg_S"):
        _init(env)
    c = env.cfg.precise_assembly
    n = env_ids.numel()
    lo, hi = (float(v) for v in c.cupgrip_retain_range_N)
    rs = int(c.cupgrip_retain_ramp_steps)
    if rs > 0:   # curriculum: the upper bound of S grows from cupgrip_retain_start_hi_N to hi
        f = min(1.0, (int(env.common_step_counter) + int(c.cupgrip_retain_ramp_offset_steps)) / rs)
        hi = math.exp(math.log(float(c.cupgrip_retain_start_hi_N)) * (1.0 - f) + math.log(hi) * f)
    env.cg_S_hi = hi
    u = torch.rand(n, device=env.device)
    env.cg_S[env_ids] = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if hi > lo else lo
    rlo, rhi = (float(v) for v in c.cupgrip_drag_ratio_range)
    r = rlo + (rhi - rlo) * torch.rand(n, device=env.device)
    steps = int(c.cupgrip_drag_ramp_steps)
    if steps > 0:   # curriculum: r starts at cupgrip_drag_ratio_start and shrinks linearly to the range
        f = min(1.0, (int(env.common_step_counter) + int(c.cupgrip_drag_ramp_offset_steps)) / steps)
        r = float(c.cupgrip_drag_ratio_start) * (1.0 - f) + r * f
    env.cg_r[env_ids] = r
    w = env.cg_m1[env_ids] * G
    if str(c.cupgrip_mode) == "window":
        # modelled grip window: no physical retention (the arm feels only the 15 g cup); hidden minimum squeeze G_lo
        glo, ghi = (float(v) for v in c.cupgrip_glo_range_N)
        u2 = torch.rand(n, device=env.device)
        env.cg_G[env_ids] = torch.exp(math.log(glo) + u2 * (math.log(ghi) - math.log(glo))) if ghi > glo else glo
        env.cg_S[env_ids] = 0.0
        env.cg_D[env_ids] = env.cg_r[env_ids] * env.cg_G[env_ids]
    else:
        env.cg_G[env_ids] = (env.cg_S[env_ids] + w) / float(c.cupgrip_mu_cal)
        env.cg_D[env_ids] = env.cg_r[env_ids] * (env.cg_S[env_ids] + w) / float(c.cupgrip_mu_cal)
    env.cg_low[env_ids] = 0
    klo, khi = (float(v) for v in c.cupgrip_hand_kp_range)
    k = klo + (khi - klo) * torch.rand(n, device=env.device)
    env.cg_kp[env_ids] = k
    if not (klo == 1.0 and khi == 1.0):
        env.robot.write_joint_stiffness_to_sim(env._cg_kp0[env_ids] * k[:, None], joint_ids=env._cg_hand_ids,
                                               env_ids=env_ids)
        env.robot.write_joint_damping_to_sim(env._cg_kd0[env_ids] * k.sqrt()[:, None], joint_ids=env._cg_hand_ids,
                                             env_ids=env_ids)
    env.cg_coupled[env_ids] = False
    env.cg_stuck[env_ids] = True
    env.cg_break_count[env_ids] = 0
    env.cg_ever_coupled[env_ids] = False
    env.cg_grasped[env_ids] = False
    env.cg_slide[env_ids] = 0.0
    env.cg_rise[env_ids] = 0.0
    env.cg_rise2[env_ids] = 0.0
    o = env.scene.env_origins[env_ids]
    env.cg_obj0[env_ids, :3] = env.object.data.root_pos_w[env_ids] - o
    env.cg_obj0[env_ids, 3:] = env.object.data.root_quat_w[env_ids]
    # data buffers may not reflect the write yet: take the commanded reset pose from the problem
    w0 = env._pose_rel_hole_to_world(env_ids, env._pih_object_start_rel)
    env.cg_obj0[env_ids, :3] = w0[:, :3] - o
    env.cg_obj0[env_ids, 3:] = w0[:, 3:]
    w2 = env._pose_rel_hole_to_world(env_ids, env._pih_distractor_pose_rel[0])
    env.cg_z2_0[env_ids] = w2[:, 2] - o[:, 2]
    env.cg_off[env_ids] = env.cg_obj0[env_ids, 2] - env.cg_z2_0[env_ids]


def _palm_z(env) -> torch.Tensor:
    return env.robot.data.body_state_w[:, env._palm_body_id, 2] - env.scene.env_origins[:, 2]


def pre_physics(env) -> None:
    """Every policy step after the action pipeline (which set the object's DR wrench in env._object_forces)."""
    if not enabled(env) or not hasattr(env, "cg_S"):
        return
    c = env.cfg.precise_assembly
    o = env.scene.env_origins
    z1 = env.object.data.root_pos_w[:, 2] - o[:, 2]
    cup2 = env.distractors[0]
    z2 = cup2.data.root_pos_w[:, 2] - o[:, 2]
    rise = z1 - env.cg_obj0[:, 2]
    env.cg_rise = rise
    env.cg_rise2 = z2 - env.cg_z2_0
    # Breakaway nest retention: while "stuck" the top cup is held on the stack against any upward pull from the
    # hand up to S (the downward force cancels the hand's upward contact force of the previous step, plus a PD hold
    # on any rise); it breaks free once the hand pulls up harder than S. No force at rest (no pressing into the
    # stack). Re-sticks when it is set back down seated and still.
    v1 = env.object.data.root_lin_vel_w[:, 2]
    fz = getattr(env, "hand_force_z", None)
    fz = torch.zeros_like(z1) if fz is None else fz.clamp(min=0.0)
    tau_h = float(c.cupgrip_hold_tau_s)
    pull = (fz - env.cg_m1 * G).clamp(min=0.0)   # upward pull beyond the cup's weight
    brk = env.cg_stuck & (pull > env.cg_S)
    env.cg_stuck = env.cg_stuck & ~brk
    # re-stick only a cup that was set back down: seated, still, and not being pulled (never in the break step)
    restick = (~env.cg_stuck & ~brk & (rise < float(c.cupgrip_restick_mm) / 1000.0) & (v1.abs() < 0.02)
               & (pull < 0.25 * env.cg_S))
    env.cg_stuck = env.cg_stuck | restick
    hold = pull + env.cg_m1 * (rise.clamp(min=0.0) / (tau_h * tau_h) + v1.clamp(min=0.0) / tau_h)
    f_ret = torch.where(env.cg_stuck, torch.minimum(hold, env.cg_S).clamp(min=0.0),
                        torch.zeros_like(z1))
    env.cg_break_count = env.cg_break_count + brk.long()

    # squeeze = horizontal contact force of the hand on the top cup; the drag engages only for a squeeze above D while
    # the hand pulls the cup up (pressing down or knocking the cup does not lift the second cup)
    hand = getattr(env, "hand_force_xy", None)
    hand = torch.zeros_like(z1) if hand is None else hand
    env.cg_hand = hand
    nested = rise < float(c.cupgrip_nest_overlap_mm) / 1000.0
    carrying = fz > 0.5 * env.cg_m1 * G          # the hand holds the cup up (not pressing it down / knocking it)
    on = (hand > env.cg_D) & nested & carrying
    off = (hand < 0.9 * env.cg_D) | ~nested
    env.cg_coupled = (env.cg_coupled | on) & ~off
    env.cg_ever_coupled |= env.cg_coupled

    # second cup follows the top cup at its nested offset while coupled
    tau = float(c.cupgrip_couple_tau_s)
    v1 = env.object.data.root_lin_vel_w[:, 2]
    v2 = cup2.data.root_lin_vel_w[:, 2]
    tgt = z1 - env.cg_off
    a = (tgt - z2) / (tau * tau) + (v1 - v2) / tau
    f2 = env.cg_m2 * (G + a)
    f2 = torch.where(env.cg_coupled, f2.clamp(0.0, float(c.cupgrip_couple_max_N)), torch.zeros_like(f2))

    F1 = torch.zeros(env.num_envs, 1, 3, device=env.device)
    base = getattr(env, "_object_forces", None)
    if base is not None:
        F1 = F1 + base
    F1[:, 0, 2] -= f_ret + f2
    T1 = getattr(env, "_object_torques", None)
    T1 = torch.zeros_like(F1) if T1 is None else T1
    env.object.set_external_force_and_torque(forces=F1, torques=T1, is_global=True)
    F2 = torch.zeros(env.num_envs, 1, 3, device=env.device)
    F2[:, 0, 2] = f2
    cup2.set_external_force_and_torque(forces=F2, torques=torch.zeros_like(F2), is_global=True)


def pin_substep(env) -> None:
    """Every physics substep (_apply_action): the second cup can't rise on its seat unless the drag coupling is on
    (its own nest stickiness is far stronger than the wall friction a squeezed top cup exerts; the squeeze-drag path
    is the model in pre_physics). Upward motion only."""
    if not enabled(env) or not hasattr(env, "cg_S") or not bool(env.cfg.precise_assembly.cupgrip_pin_second):
        return
    o = env.scene.env_origins
    cup2 = env.distractors[0]
    rise2 = cup2.data.root_pos_w[:, 2] - o[:, 2] - env.cg_z2_0
    v2 = cup2.data.root_lin_vel_w[:, 2]
    pin = ~env.cg_coupled & ((rise2 > 0.0002) | (v2 > 0.0))
    ids = pin.nonzero().reshape(-1)
    if ids.numel():
        pose = torch.cat([cup2.data.root_pos_w[ids], cup2.data.root_quat_w[ids]], -1).clone()
        pose[:, 2] = torch.minimum(pose[:, 2], env.cg_z2_0[ids] + o[ids, 2])
        vel = torch.cat([cup2.data.root_lin_vel_w[ids], cup2.data.root_ang_vel_w[ids]], -1).clone()
        vel[:, 2] = vel[:, 2].clamp(max=0.0)
        cup2.write_root_pose_to_sim(pose, env_ids=ids)
        cup2.write_root_velocity_to_sim(vel, env_ids=ids)


def dones(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(dragged, slipped) for _get_dones (after _hand_force_update)."""
    z = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not enabled(env) or not hasattr(env, "cg_S"):
        return z, z
    c = env.cfg.precise_assembly
    o = env.scene.env_origins
    z1 = env.object.data.root_pos_w[:, 2] - o[:, 2]
    rise = z1 - env.cg_obj0[:, 2]
    rise2 = env.distractors[0].data.root_pos_w[:, 2] - o[:, 2] - env.cg_z2_0
    rel = _palm_z(env) - z1
    # slide = palm-vs-cup upward motion accumulated only on steps the hand grips the cup (hand force >= contact_N in
    # this and the previous step); letting go and moving the hand away does not count. Clamped at 0 from below.
    gripping = env.hand_force >= float(c.cupgrip_contact_N)
    both = gripping & env.cg_grasped                 # cg_grasped = gripping on the previous step
    d_rel = rel - env.cg_rel0                        # cg_rel0 = rel on the previous step
    env.cg_slide = torch.where(both, (env.cg_slide + d_rel).clamp(min=0.0), env.cg_slide)
    env.cg_rel0 = rel
    env.cg_grasped = gripping
    nested = rise < float(c.cupgrip_nest_overlap_mm) / 1000.0
    # only the modelled squeeze-drag fails (an uncoupled one-substep friction blip is snapped back by the pin)
    dragged = (rise2 > float(c.cupgrip_drag_fail_mm) / 1000.0) & (env.cg_coupled | ~bool(c.cupgrip_pin_second))
    from isaaclab.utils.math import quat_apply
    dn = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    env.cg_last_rise2 = rise2.clone()
    env.cg_last_tilt2 = torch.rad2deg(torch.acos(quat_apply(env.distractors[0].data.root_quat_w, dn)[:, 2].clamp(-1, 1)))
    env.cg_last_xy2 = (env.distractors[0].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
                       - env.cg_obj0[:, :2]).norm(dim=-1)
    slipped = (env.cg_slide > float(c.cupgrip_slip_mm) / 1000.0) & nested & ~dragged
    if str(c.cupgrip_mode) == "window":
        # modelled "too soft": lifting the nested cup (it rose > 2 mm and moves up) with a squeeze below G_lo
        sq = getattr(env, "hand_force_xy", torch.zeros_like(rise))
        v1 = env.object.data.root_lin_vel_w[:, 2]
        low = gripping & nested & (rise > 0.002) & (v1 > 0.005) & (sq < env.cg_G)
        env.cg_low = env.cg_low + low.long()
        slipped = (env.cg_low >= int(c.cupgrip_slip_steps)) & ~dragged
    if float(c.cupgrip_slip_mm) <= 0.0:
        slipped = z
    for k, v in (("cupgrip_S_mean", env.cg_S.mean()), ("cupgrip_r_mean", env.cg_r.mean()), ("cupgrip_S_hi", torch.tensor(float(getattr(env, "cg_S_hi", 0.0)))), ("cupgrip_coupled_frac", env.cg_coupled.float().mean()),
                 ("cupgrip_rise_max", rise.max()), ("cupgrip_slide_mean", env.cg_slide.mean())):
        env.extras[k] = v
    return dragged, slipped


def tactile_cue(env, tac: torch.Tensor) -> torch.Tensor:
    """cupgrip_cue_ratio > 0 (TOUCH-ONLY cue, modelled incipient slip): while the hand grips the nested top cup with a
    squeeze below cue_ratio x G_lo, every pad image is shifted by one taxel along its rows, alternating direction
    each step (micro-slip jitter). (N, 5 * res * res) -> same shape."""
    c = env.cfg.precise_assembly
    if not enabled(env) or not hasattr(env, "cg_G") or float(c.cupgrip_cue_ratio) <= 0.0:
        return tac
    sq = getattr(env, "hand_force_xy", None)
    if sq is None:
        return tac
    nested = env.cg_rise < float(c.cupgrip_nest_overlap_mm) / 1000.0
    zone = (sq >= float(c.cupgrip_contact_N)) & nested & (sq < float(c.cupgrip_cue_ratio) * env.cg_G)
    env.cg_cue = zone
    if not bool(zone.any()):
        return tac
    res = int(env.cfg.tactile.resolution)
    t = tac.view(env.num_envs, -1, res, res)
    sh = 1 if int(env.common_step_counter) % 2 == 0 else -1
    rolled = torch.roll(t, shifts=sh, dims=-2)
    out = torch.where(zone.view(-1, 1, 1, 1), rolled, t)
    return out.reshape(tac.shape)


def actor_object_override(env, obs: dict, kp_world_fn, palm_pos, goal_kp, kp_offsets) -> None:
    """cupgrip_hide_object: actor object fields from the top cup's reset pose (in place on the noisy obs dict)."""
    if not enabled(env) or not bool(env.cfg.precise_assembly.cupgrip_hide_object) or not hasattr(env, "cg_obj0"):
        return
    from isaaclab.utils.math import convert_quat
    p0, q0 = env.cg_obj0[:, :3], env.cg_obj0[:, 3:]
    kp0 = kp_world_fn(p0, q0, kp_offsets)
    obs["object_rot"] = convert_quat(q0, to="xyzw")
    obs["keypoints_rel_palm"] = kp0 - palm_pos.unsqueeze(1)
    obs["keypoints_rel_goal"] = kp0 - goal_kp
