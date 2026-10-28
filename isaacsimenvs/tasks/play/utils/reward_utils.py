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
    lifting_rew_offset: float = 0.05,
    grasp_ok: torch.Tensor | None = None,     # (N,) bool: dense lift reward only while grasped
    grasp_held: torch.Tensor | None = None,   # (N,) bool: latch only after a held grasped lift
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift progress, one-shot lift bonus, and latched lifted state."""
    z_lift = 0.05 + object_z - object_init_z
    # The offset pays lifting_rew_scale * offset per step before the lift even with the object untouched; with long
    # episodes and a lift reward forced on (precise_assembly.force_lift_reward_active) that beats the lift bonus.
    lift_rew = torch.clamp(z_lift - 0.05 + lifting_rew_offset, 0.0, 0.5)
    crossed = z_lift > lifting_bonus_threshold
    if grasp_held is not None:
        crossed = crossed & grasp_held
    lifted = crossed | prev_lifted
    just_crossed = lifted & ~prev_lifted
    lift_bonus = lifting_bonus * just_crossed.float()
    lift_rew = lift_rew * (~lifted).float()
    if grasp_ok is not None:
        lift_rew = lift_rew * grasp_ok.float()
    return lift_rew * lifting_rew_scale, lift_bonus, lifted


def distance_delta_reward(
    curr_fingertip_dist: torch.Tensor,    # (N, num_fingertips)
    closest_fingertip_dist: torch.Tensor,  # (N, num_fingertips)
    lifted: torch.Tensor,                  # (N,) bool
    rew_scale: float,
    weights: torch.Tensor | None = None,   # (num_fingertips,)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward fingertip progress before the object is lifted."""
    deltas = closest_fingertip_dist - curr_fingertip_dist
    new_closest = torch.minimum(closest_fingertip_dist, curr_fingertip_dist)
    deltas = torch.clamp(deltas, 0.0, 10.0)
    if weights is not None:
        deltas = deltas * weights
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

    # Grasp-gated lift (reward.lift_grasp_fingers > 0): the lift reward and the latched "lifted" state (lift bonus,
    # start of the keypoint reward) only count while >= lift_grasp_fingers fingertips are within lift_grasp_dist of
    # the object, held lift_grasp_steps steps for the latch. A small part in a bin otherwise earns the bonus by being
    # knocked a few cm up.
    grasp_ok = None
    if rew_cfg.lift_grasp_fingers > 0:
        grasp_ok = (env._curr_fingertip_distances < rew_cfg.lift_grasp_dist).sum(dim=-1) >= rew_cfg.lift_grasp_fingers
        z_ok = (0.05 + obj_pos[:, 2] - env._object_init_z) > rew_cfg.lifting_bonus_threshold
        hold = getattr(env, "_lift_grasp_hold", None)
        if hold is None or hold.shape[0] != grasp_ok.shape[0]:
            hold = torch.zeros_like(grasp_ok, dtype=torch.long)
        hold = torch.where(grasp_ok & z_ok, hold + 1, torch.zeros_like(hold))
        hold = torch.where(env.episode_length_buf <= 1, torch.zeros_like(hold), hold)
        env._lift_grasp_hold = hold
    lift_rew, lift_bonus_rew, new_lifted = lifting_reward(
        object_z=obj_pos[:, 2],
        object_init_z=env._object_init_z,
        prev_lifted=env._lifted_object,
        lifting_bonus_threshold=rew_cfg.lifting_bonus_threshold,
        lifting_bonus=rew_cfg.lifting_bonus,
        lifting_rew_scale=rew_cfg.lifting_rew_scale,
        lifting_rew_offset=rew_cfg.lifting_rew_offset,
        grasp_ok=grasp_ok,
        grasp_held=None if grasp_ok is None else env._lift_grasp_hold >= rew_cfg.lift_grasp_steps,
    )
    env._lifted_object = new_lifted

    ft_rew, new_closest_ft = distance_delta_reward(
        curr_fingertip_dist=env._curr_fingertip_distances,
        closest_fingertip_dist=env._closest_fingertip_dist,
        lifted=env._lifted_object,
        rew_scale=rew_cfg.distance_delta_rew_scale,
        weights=getattr(env, "_fingertip_rew_weights", None),
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
    # reward.grasp_rew_scale: per step with the grasp gate (lift_grasp_fingers near the object) before the lift
    grasp_rew = torch.zeros_like(reward)
    if grasp_ok is not None and rew_cfg.grasp_rew_scale > 0:
        grasp_rew = rew_cfg.grasp_rew_scale * (grasp_ok & ~env._lifted_object).float()
        reward = reward + grasp_rew
    env._reward_terms = {
        "fingertip_delta_rew": ft_rew,
        "lifting_rew": lift_rew,
        "grasp_rew": grasp_rew,
        "lift_bonus_rew": lift_bonus_rew,
        "keypoint_rew": kp_rew,
        "kuka_actions_penalty": kuka_pen,
        "hand_actions_penalty": hand_pen,
        "bonus_rew": bonus,
        "total_reward": reward,
    }

    return reward


__all__ = [
    "lifting_reward",
    "distance_delta_reward",
    "keypoint_reward",
    "action_penalty",
    "update_near_goal_steps",
    "reach_goal_bonus",
    "compute_rewards",
]
