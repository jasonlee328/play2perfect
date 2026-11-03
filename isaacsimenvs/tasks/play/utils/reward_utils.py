"""Pure-torch reward helpers for Play."""

from __future__ import annotations

import torch


def lifting_reward(
    object_z: torch.Tensor,            # (N,)
    object_init_z: torch.Tensor,       # (N,)
    prev_lifted: torch.Tensor,         # (N,) bool
    lifting_bonus_threshold: float,
    lifting_bonus: float,
    lifting_rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift progress, one-shot lift bonus, and latched lifted state."""
    z_lift = 0.05 + object_z - object_init_z
    lift_rew = torch.clamp(z_lift, 0.0, 0.5)
    lifted = (z_lift > lifting_bonus_threshold) | prev_lifted
    just_crossed = lifted & ~prev_lifted
    lift_bonus = lifting_bonus * just_crossed.float()
    lift_rew = lift_rew * (~lifted).float()
    return lift_rew * lifting_rew_scale, lift_bonus, lifted


def distance_delta_reward(
    curr_fingertip_dist: torch.Tensor,    # (N, num_fingertips)
    closest_fingertip_dist: torch.Tensor,  # (N, num_fingertips)
    lifted: torch.Tensor,                  # (N,) bool
    rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward fingertip progress before the object is lifted."""
    deltas = closest_fingertip_dist - curr_fingertip_dist
    new_closest = torch.minimum(closest_fingertip_dist, curr_fingertip_dist)
    deltas = torch.clamp(deltas, 0.0, 10.0)
    rew = deltas.sum(dim=-1) * (~lifted).float()
    return rew * rew_scale, new_closest


def keypoint_reward(
    keypoints_max_dist: torch.Tensor,         # (N,)
    closest_keypoint_max_dist: torch.Tensor,   # (N,)
    lifted: torch.Tensor,                      # (N,) bool
    rew_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward keypoint progress after the object is lifted."""
    delta = closest_keypoint_max_dist - keypoints_max_dist
    new_closest = torch.minimum(closest_keypoint_max_dist, keypoints_max_dist)
    delta = torch.clamp(delta, 0.0, 100.0)
    rew = delta * lifted.float()
    return rew * rew_scale, new_closest


def action_penalty(
    joint_vel: torch.Tensor,        # (N, num_dofs)
    arm_ids,
    hand_ids,
    kuka_scale: float,
    hand_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """L1 joint-velocity penalty, despite the legacy action-penalty name."""
    kuka = -kuka_scale * joint_vel[:, arm_ids].abs().sum(dim=-1)
    hand = -hand_scale * joint_vel[:, hand_ids].abs().sum(dim=-1)
    return kuka, hand


def update_near_goal_steps(
    near_goal: torch.Tensor,        # (N,) bool
    near_goal_steps: torch.Tensor,   # (N,) long
    force_consecutive: bool,
) -> torch.Tensor:
    """Update near-goal counter, optionally requiring consecutive steps."""
    ng = near_goal.long()
    if force_consecutive:
        return (near_goal_steps + ng) * ng
    return near_goal_steps + ng


def reach_goal_bonus(
    near_goal: torch.Tensor,        # (N,) bool
    is_success: torch.Tensor,        # (N,) bool
    reach_goal_bonus_value: float,
    success_steps: int,
    force_consecutive: bool,
) -> torch.Tensor:
    """Return either lump-sum or amortized goal bonus."""
    if force_consecutive:
        return is_success.float() * reach_goal_bonus_value
    return near_goal.float() * (reach_goal_bonus_value / success_steps)


def compute_rewards(env) -> torch.Tensor:
    """Sum reward terms and update reward trackers."""
    rew_cfg = env.cfg.reward
    term_cfg = env.cfg.termination
    env_origins = env.scene.env_origins
    obj_pos = env.object.data.root_pos_w - env_origins

    lift_rew, lift_bonus_rew, new_lifted = lifting_reward(
        object_z=obj_pos[:, 2],
        object_init_z=env._object_init_z,
        prev_lifted=env._lifted_object,
        lifting_bonus_threshold=rew_cfg.lifting_bonus_threshold,
        lifting_bonus=rew_cfg.lifting_bonus,
        lifting_rew_scale=rew_cfg.lifting_rew_scale,
    )
    env._lifted_object = new_lifted

    ft_rew, new_closest_ft = distance_delta_reward(
        curr_fingertip_dist=env._curr_fingertip_distances,
        closest_fingertip_dist=env._closest_fingertip_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.distance_delta_rew_scale,
    )
    env._closest_fingertip_dist = new_closest_ft

    kp_rew, new_closest_kp = keypoint_reward(
        keypoints_max_dist=env._keypoints_max_dist,
        closest_keypoint_max_dist=env._closest_keypoint_max_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.keypoint_rew_scale,
    )
    env._closest_keypoint_max_dist = new_closest_kp

    kuka_pen, hand_pen = action_penalty(
        joint_vel=env.robot.data.joint_vel,
        arm_ids=env._arm_joint_ids,
        hand_ids=env._hand_joint_ids,
        kuka_scale=rew_cfg.kuka_actions_penalty_scale,
        hand_scale=rew_cfg.hand_actions_penalty_scale,
    )

    bonus = reach_goal_bonus(
        near_goal=env._near_goal,
        is_success=env._is_success,
        reach_goal_bonus_value=rew_cfg.reach_goal_bonus,
        success_steps=term_cfg.success_steps,
        force_consecutive=term_cfg.force_consecutive_near_goal_steps,
    )

    reward = lift_rew + lift_bonus_rew + ft_rew + kp_rew + kuka_pen + hand_pen + bonus
    env._reward_terms = {
        "fingertip_delta_rew": ft_rew,
        "lifting_rew": lift_rew,
        "lift_bonus_rew": lift_bonus_rew,
        "keypoint_rew": kp_rew,
        "kuka_actions_penalty": kuka_pen,
        "hand_actions_penalty": hand_pen,
        "bonus_rew": bonus,
        "total_reward": reward,
    }

    return reward


def contact_persistence_reward(env) -> torch.Tensor:
    """Hold / break / jerk terms on fingertip contact while the object is lifted (RewardCfg.contact_*).
    Also logs the contact-stability proxy (pads on, breaks, 2-pad share while lifted) in env.extras."""
    rc = env.cfg.reward
    hand = getattr(rc, "contact_source", "tips") == "hand"  # "hand": 21 skin links (tactile_hand) instead of 5 pads
    tac = getattr(env, "_last_tactile_hand" if hand else "_last_tactile", None)
    zero = torch.zeros(env.num_envs, device=env.device)
    if tac is None:
        return zero
    tac = tac.view(env.num_envs, 21 if hand else 5, -1)
    on = (tac > rc.contact_thr).any(-1)
    prev_on = getattr(env, "_contact_prev_on", None)
    prev_tac = getattr(env, "_contact_prev_tac", None)
    fresh = env.episode_length_buf <= 1  # episode just reset: no previous contact state
    if prev_on is None:
        prev_on, prev_tac = torch.zeros_like(on), torch.zeros_like(tac)
    prev_on = prev_on & ~fresh.unsqueeze(1)
    rise = env.object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - env._object_init_z
    lifted = (rise > rc.contact_gate_rise).float()
    n_on = on.sum(1)
    hold = (n_on >= rc.contact_min_pads).float()
    paid = getattr(env, "_contact_hold_paid", None)
    if paid is None:
        paid = torch.zeros(env.num_envs, device=env.device)
    paid = torch.where(fresh, torch.zeros_like(paid), paid)
    hold_pay = hold * lifted * (paid < rc.contact_hold_max_steps).float()
    env._contact_hold_paid = paid + hold_pay
    breaks = (prev_on & ~on).sum(1).float()
    jerk = torch.where(on.any(1) & ~fresh, (tac - prev_tac).abs().mean((1, 2)), zero)
    rew = rc.contact_hold_rew_scale * hold_pay - lifted * (rc.contact_break_penalty_scale * breaks
                                                         + rc.contact_jerk_penalty_scale * jerk)
    env._contact_prev_on, env._contact_prev_tac = on.clone(), tac.clone()
    nl = lifted.sum().clamp(min=1.0)
    env.extras["contact/lifted_2pad_frac"] = float((hold * lifted).sum() / nl)
    env.extras["contact/lifted_pads_on"] = float((n_on.float() * lifted).sum() / nl)
    env.extras["contact/lifted_breaks_per_step"] = float((breaks * lifted).sum() / nl)
    env.extras["contact/lifted_share"] = float(lifted.mean())
    return rew


def coverage_reward(env) -> torch.Tensor:
    """Touch coverage of the target's surface (RewardCfg.coverage_*): + scale x newly covered share of the surface
    cells, a cell counting once its ray has touched the target on this step and the previous one. Logs coverage/*."""
    rc = env.cfg.reward
    sk = getattr(env, "_hand_skin", None)
    cells = getattr(env, "_skin_cells", None)
    zero = torch.zeros(env.num_envs, device=env.device)
    if sk is None or cells is None:
        return zero
    N, C = env.num_envs, sk.n_cells
    fresh = env.episode_length_buf <= 1
    vis = getattr(env, "_cov_visited", None)
    prev = getattr(env, "_cov_prev_cells", None)
    if vis is None or vis.shape[0] != N:
        vis = torch.zeros(N, C + 1, dtype=torch.bool, device=env.device)
        prev = torch.full_like(cells, -1)
    # coverage at the end of finished episodes (for logging), before the reset clears it
    done_cov = vis[fresh, :C].float().mean(1)
    vis[fresh] = False
    prev = torch.where(fresh.unsqueeze(1), torch.full_like(prev, -1), prev)
    held = (cells >= 0) & (prev >= 0)                      # the ray touched the target last step too
    idx = torch.where(held, cells, torch.full_like(cells, C))
    before = vis[:, :C].sum(1)
    vis.scatter_(1, idx, True)
    vis[:, C] = False
    new = (vis[:, :C].sum(1) - before).float()
    env._cov_visited, env._cov_prev_cells = vis, cells.clone()
    rew = rc.coverage_rew_scale * new / C
    cov = vis[:, :C].float().mean(1)
    env.extras["coverage/frac"] = float(cov.mean())
    env.extras["coverage/new_cells_per_step"] = float(new.mean())
    env.extras["coverage/rays_on_target"] = float((cells >= 0).sum(1).float().mean())
    if done_cov.numel():
        env.extras["coverage/episode_end_frac"] = float(done_cov.mean())
    return rew


__all__ = [
    "contact_persistence_reward",
    "lifting_reward",
    "distance_delta_reward",
    "keypoint_reward",
    "action_penalty",
    "update_near_goal_steps",
    "reach_goal_bonus",
    "compute_rewards",
]
