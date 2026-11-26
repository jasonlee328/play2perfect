"""Action processing and random object wrench helpers."""

from __future__ import annotations

import math

import torch


def sample_log_uniform(lo_hi: tuple[float, float], n: int) -> torch.Tensor:
    """Log-uniform sample on CPU (caller moves to device)."""
    lo, hi = lo_hi
    return torch.exp(
        torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12))
    )


def _tip_ik_setup(env):
    rb = env.robot
    fixed = bool(rb.is_fixed_base)
    tips = [rb.find_bodies("left_thumb_DP")[0][0], rb.find_bodies("left_index_DP")[0][0]]
    J = rb.root_physx_view.get_jacobians()
    off = 0 if fixed else 6
    hand_ids = [int(j) for j in env._hand_joint_ids]
    chains = []
    for b in tips:
        row = b - (1 if fixed else 0)
        cols = (J[0, row].abs().sum(0) > 1e-9).nonzero().reshape(-1).tolist()
        joints = [c - off for c in cols if (c - off) in hand_ids]
        joints.sort()
        chains.append(dict(body=b, row=row, joints=joints, cols=[j + off for j in joints],
                           hk=[hand_ids.index(j) for j in joints]))
    env._tip_ik = dict(chains=chains)
    print("[tip_ik] thumb", [rb.joint_names[j] for j in chains[0]["joints"]],
          "index", [rb.joint_names[j] for j in chains[1]["joints"]], flush=True)


def _tip_ik_targets(env, actions, hand_raw, scale, act_cfg):
    """A27: thumb / index tip Cartesian deltas (palm frame) -> joint target deltas by damped least squares."""
    from isaaclab.utils.math import quat_apply
    if getattr(env, "_tip_ik", None) is None:
        _tip_ik_setup(env)
    rb = env.robot
    J = rb.root_physx_view.get_jacobians()
    palm_q = rb.data.body_state_w[:, env._palm_body_id, 3:7]
    wrot = float(getattr(act_cfg, "hand_tip_ik_wrot", 0.02))
    lam = float(getattr(act_cfg, "hand_tip_ik_lambda", 0.001))
    lim_margin = float(getattr(act_cfg, "hand_tip_ik_limit_margin", 0.0))
    eye6 = torch.eye(6, device=env.device)
    hand_raw = hand_raw.clone()
    hand_act = actions[:, 7:]
    for ch in env._tip_ik["chains"]:
        hk = ch["hk"]
        d_palm = scale * hand_act[:, hk[:3]]                       # (n, 3) tip delta in the palm frame
        e = torch.cat([quat_apply(palm_q, d_palm), torch.zeros_like(d_palm)], -1)   # hold tip orientation
        Jk = J[:, ch["row"]][:, :, ch["cols"]]
        Jk = torch.cat([Jk[:, :3], wrot * Jk[:, 3:]], 1)
        dq = Jk.transpose(1, 2) @ torch.linalg.solve(Jk @ Jk.transpose(1, 2) + lam ** 2 * eye6, e.unsqueeze(-1))
        if lim_margin > 0.0:
            # A27: limit-aware IK. A joint whose target sits within lim_margin (share of its range) of a limit and whose
            # step would push further into it is dropped from the Jacobian, then the step is re-solved so the other
            # joints of the finger carry the tip motion (plain DLS keeps commanding the saturated joint -> tip stalls).
            qt = env._prev_targets[:, ch["joints"]]
            lo = env._hand_lower[:, hk] if env._hand_lower.dim() == 2 else env._hand_lower[hk]
            hi = env._hand_upper[:, hk] if env._hand_upper.dim() == 2 else env._hand_upper[hk]
            mg = lim_margin * (hi - lo)
            d0 = dq.squeeze(-1)
            block = ((qt <= lo + mg) & (d0 < 0)) | ((qt >= hi - mg) & (d0 > 0))
            if bool(block.any()):
                Jm = Jk * (~block).float().unsqueeze(1)
                dq = Jm.transpose(1, 2) @ torch.linalg.solve(Jm @ Jm.transpose(1, 2) + lam ** 2 * eye6, e.unsqueeze(-1))
        hand_raw[:, hk] = env._prev_targets[:, ch["joints"]] + dq.squeeze(-1)
    return hand_raw


def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    """Apply delay, canonical-to-Lab mapping, smoothing, and target clamps."""
    # Debug replay path accepts an already Lab-order target.
    replay_target = getattr(env, "_replay_target_lab_order", None)
    if replay_target is not None:
        env._cur_targets[:] = replay_target
        env._prev_targets = env._cur_targets.clone()
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt  # policy step (= decimation * sim.dt = 1/60 s)

    actions = actions.clone().to(env.device)

    # Canonical policy order -> Lab parser order.
    actions = actions[:, env._perm_canon_to_lab]

    # Action delay.
    if dr.use_action_delay and dr.action_delay_max > 0:
        episode_start = (env.episode_length_buf == 0) & (env._successes == env._start_successes)
        if episode_start.any():
            env._action_queue[episode_start] = actions[episode_start].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        delay_idx = torch.randint(
            0, env._action_queue.shape[1], (env.num_envs,), device=env.device
        )
        actions = env._action_queue[
            torch.arange(env.num_envs, device=env.device), delay_idx
        ]

    # Arm: velocity-delta accumulator.
    arm_action = actions[:, :7] * float(getattr(act_cfg, "arm_action_scale", 1.0))   # A13: 0 = arm frozen (fingers only)
    arm_raw = env._prev_targets[:, :7] + act_cfg.dof_speed_scale * dt * arm_action
    arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
    arm_smoothed = (
        act_cfg.arm_moving_average * arm_raw
        + (1.0 - act_cfg.arm_moving_average) * env._prev_targets[:, :7]
    )
    arm_smoothed = torch.clamp(arm_smoothed, env._arm_lower, env._arm_upper)

    # Hand: absolute [-1, 1] scale.
    hand_action = actions[:, 7:]
    hand_delta = float(getattr(act_cfg, "hand_delta_scale", 0.0))
    if hand_delta > 0.0:
        # A12: relative hand (like the arm): zero action holds the current grip
        hand_raw = env._prev_targets[:, 7:] + hand_delta * dt * hand_action
    else:
        hand_raw = env._hand_lower + 0.5 * (hand_action + 1.0) * (
            env._hand_upper - env._hand_lower
        )
    tip_scale = float(getattr(act_cfg, "hand_tip_ik_scale", 0.0))
    if tip_scale > 0.0:
        hand_raw = _tip_ik_targets(env, actions, hand_raw, tip_scale, act_cfg)
    hand_smoothed = (
        act_cfg.hand_moving_average * hand_raw
        + (1.0 - act_cfg.hand_moving_average) * env._prev_targets[:, 7:]
    )
    hand_smoothed = torch.clamp(hand_smoothed, env._hand_lower, env._hand_upper)

    # Write Lab-order targets and cache them for the next step.
    env._cur_targets[:, env._arm_joint_ids] = arm_smoothed
    env._cur_targets[:, env._hand_joint_ids] = hand_smoothed
    env._prev_targets = env._cur_targets.clone()


def apply_wrench_dr(env) -> None:
    """Apply decayed random force/torque impulses to the object."""
    dr = env.cfg.domain_randomization
    dt_pol = env.step_dt

    # Decay previous wrench.
    if dr.force_decay > 0.0:
        env._object_forces *= dr.force_decay ** (dt_pol / dr.force_decay_interval)
    else:
        env._object_forces.zero_()
    if dr.torque_decay > 0.0:
        env._object_torques *= dr.torque_decay ** (dt_pol / dr.torque_decay_interval)
    else:
        env._object_torques.zero_()

    # Sample new wrench with per-env probability.
    force_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_force_prob
    ).view(-1, 1, 1)
    torque_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_torque_prob
    ).view(-1, 1, 1)
    mass = env._object_mass.unsqueeze(-1)  # (N, 1, 1)
    new_force = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.force_scale
    )
    new_torque = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.torque_scale
    )
    tr = tuple(getattr(dr, "torque_mult_log_range", ()) or ())
    if len(tr) == 2:   # A23: hidden per-episode twist strength, log-uniform multiplier on torque_scale
        if getattr(env, "_torque_mult", None) is None:
            env._torque_mult = torch.ones(env.num_envs, device=env.device)
            env._torque_mult_fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        fresh = env.episode_length_buf <= 1
        lo, hi = math.log(float(tr[0])), math.log(float(tr[1]))
        draw = torch.exp(lo + (hi - lo) * torch.rand(env.num_envs, device=env.device))
        newep = fresh & env._torque_mult_fresh
        env._torque_mult = torch.where(newep, draw, env._torque_mult)
        env._torque_mult_fresh = ~fresh
        new_torque = new_torque * env._torque_mult.view(-1, 1, 1)
    env._object_forces = torch.where(force_fire, new_force, env._object_forces)
    env._object_torques = torch.where(torque_fire, new_torque, env._object_torques)

    # _lifted_object is from the previous step because rewards update later.
    if dr.force_only_when_lifted:
        env._object_forces *= env._lifted_object.float().view(-1, 1, 1)
    if dr.torque_only_when_lifted:
        env._object_torques *= env._lifted_object.float().view(-1, 1, 1)

    # Optional task-level gate: subclasses can disable wrench DR per env (e.g.
    # peg_in_hole zeros impulses once the final insert is achieved so the part
    # doesn't fly out of the hole during retract). Same opt-in convention as
    # _curriculum_eligible_mask in termination_utils.
    if hasattr(env, "_wrench_dr_active_mask"):
        active = env._wrench_dr_active_mask()
        if active is not None:
            active_f = active.float().view(-1, 1, 1)
            env._object_forces *= active_f
            env._object_torques *= active_f

    env.object.set_external_force_and_torque(
        forces=env._object_forces,
        torques=env._object_torques,
        is_global=True,
    )


__all__ = ["apply_action_pipeline", "apply_wrench_dr", "sample_log_uniform"]
