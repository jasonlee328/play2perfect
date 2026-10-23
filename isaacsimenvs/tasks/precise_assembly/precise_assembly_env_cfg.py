"""Config for the peg-in-hole Play variant."""

from __future__ import annotations

from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.tasks.play.play_env_cfg import (
    AssetsCfg,
    PlayEnvCfg,
    _default_sim_cfg,
)


VALID_GOAL_MODES = (
    "preInsertAndFinal",
    "finalGoalOnly",
    "transportPreInsertFinal",
    # Fixtured-env-only mode: trajectory waypoints come from scenes.npz
    # (per-scene precomputed dense paths). Validation happens via the
    # fixtured env's _configure_problem override.
    "dense",
)


@configclass
class PreciseAssemblyAssetsCfg(AssetsCfg):
    object_name: str = "peg"
    object_urdf: str = "assets/urdf/peg_in_hole/peg/peg.urdf"
    receptive_urdf: str = "assets/urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf"
    table_urdf: str = "assets/urdf/table_narrow.urdf"
    object_scale: tuple[float, float, float] = (6.25, 0.75, 0.5)
    num_assets_per_type: int = 1


@configclass
class PreciseAssemblyCfg:
    problem: str = "tight_insertion"
    goal_mode: str = "preInsertAndFinal"

    hole_x_range: tuple[float, float] = (-0.1875, 0.1875)
    hole_y_range: tuple[float, float] = (-0.1, 0.1)
    # Per-episode yaw randomization about +Z applied to the hole and to the
    # derived insertion goals. Range is symmetric: yaw ~ U(-range, +range).
    # Default 0.0 keeps legacy behavior (identity hole quat).
    hole_yaw_range_deg: float = 0.0

    goal_xy_obs_noise: float = 0.002
    # Per-episode yaw obs noise about world +Z applied to the observed goal
    # (insertion targets only). Range is symmetric: yaw_noise ~ U(-deg, +deg).
    # Default 0.0 keeps legacy behavior.
    goal_yaw_obs_noise_deg: float = 0.0
    random_goal_fraction: float = 0.0
    random_goal_max_successes: int = 5
    random_goal_curriculum_success_threshold: float = 2.0
    lift_bonus_fade_threshold: float = 2.0

    insertion_success_tolerance: float = 0.01
    # Eval scoring: reaching any later insertion waypoint (held success_steps steps)
    # counts it and every waypoint before it, instead of requiring them in order.
    # Leave off for training: it changes which goal the teacher sees.
    allow_goal_skip: bool = False
    # A skipped-to waypoint must also match its depth along the hole axis within
    # this (m). The keypoint tolerance alone barely sees depth (2.3 mm per quarter
    # turn), so without it pushing straight down would skip the turns.
    goal_skip_axial_tol: float = 0.003

    enable_retract: bool = True
    retract_reward_scale: float = 1.0
    retract_distance_threshold: float = 0.1
    retract_success_bonus: float = 1000.0
    retract_success_tolerance: float = 0.005
    # Eval: end the episode the first time the hand has retracted after the last goal, success or not. Success = the peg is
    # still within retract_success_tolerance of the final pose at that moment. Without it the episode runs on
    # and the hand can come back and knock the peg (or a later re-retract can count a tilted peg as success).
    end_on_retract: bool = False
    # With end_on_retract: the hand counts as retracted only when every fingertip is more than this far from the
    # peg's keypoint centroid AND no tactile pad has touched anything for retract_clear_steps steps.
    retract_min_fingertip_dist: float = 0.0
    retract_clear_steps: int = 10
    # How "still seated" is judged when the hand retracts. "keypoint": every peg keypoint within
    # retract_success_tolerance * keypoint_scale of the final waypoint (also checks rotation about the axis, ~±28° at
    # 1.5 cm). "depth": the peg root is at most retract_seated_axial_tol above the final waypoint along the hole axis,
    # within retract_seated_lateral_tol of it sideways, and tilted at most retract_seated_tilt_deg; rotation about
    # the hole axis is ignored (a seated peg that turns back rises with the thread, which the depth check catches).
    retract_seated_mode: str = "keypoint"
    retract_seated_axial_tol: float = 0.003
    retract_seated_lateral_tol: float = 0.003
    retract_seated_tilt_deg: float = 10.0

    # When True, the lift_rew / lift_bonus terms apply to insertion-only envs
    # (not only random-goal envs). Used by dense-trajectory experiments where
    # the policy needs explicit Z-progress shaping during the lift-in-place
    # prelude stage.
    force_lift_reward_active: bool = False

    # Terminate when the peg has slipped out of the gripper and is sitting on
    # the table. The check fires when (1) object z is within
    # ``dropped_on_table_z_margin`` of the table top AND (2) the mean
    # fingertip-to-object distance exceeds ``dropped_on_table_ft_distance``,
    # AND (3) the env is not in retract_phase (the deliberate finger-pull-away
    # would otherwise false-trigger this). Off by default — opt in per .sub.
    enable_dropped_on_table_term: bool = False
    dropped_on_table_z_margin: float = 0.05
    dropped_on_table_ft_distance: float = 0.15

    # Start episodes mid-task from a bank of recorded teacher states (npz from
    # dagger/record_start_states.py): robot joint state + targets, peg pose in
    # the hole frame, and the goal index reached. Each reset samples one entry;
    # the episode then continues from that goal. "" keeps the normal reset.
    start_state_file: str = ""
    # Eval: pick bank entry (start_state_offset + env_id) % bank size instead of a random one, so trial i always
    # starts from the same state.
    start_state_sequential: bool = False
    start_state_offset: int = 0

    # Screwing only: body-width variant of the peg (square_table_leg4), in mm. 0 (or 30) = the original
    # 30 x 30 mm body. Other values load assets/.../square_table_leg4/width_variants/w<10*W>/ (made by
    # tools/peg_variants/make_leg4_width_variants.py): body cross-section W x W mm, same length, same thread
    # and mass. Keypoints / object_scale stay the original's, so goals and the teacher obs don't change.
    # With reset.fixed_start_pose set, its z is shifted by (W - 30) / 2 mm so the peg still rests on the table.
    peg_width_mm: float = 0.0


def _default_precise_assembly_sim_cfg() -> SimulationCfg:
    """Return a render-overlay-compatible sim cfg for teacher and student tasks."""
    sim_cfg = _default_sim_cfg()
    # UWLab partial-assembly defaults for tight insertion/contact-rich problems.
    # Keep this task's 60 Hz policy control cadence; these are only PhysX
    # solver/contact/buffer settings.
    sim_cfg.physx.solver_type = 1
    sim_cfg.physx.min_position_iteration_count = 1
    sim_cfg.physx.max_position_iteration_count = 192
    sim_cfg.physx.min_velocity_iteration_count = 0
    sim_cfg.physx.max_velocity_iteration_count = 1
    sim_cfg.physx.bounce_threshold_velocity = 0.02
    sim_cfg.physx.friction_offset_threshold = 0.01
    sim_cfg.physx.friction_correlation_distance = 0.0005
    sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2**23
    sim_cfg.physx.gpu_max_rigid_contact_count = 2**23
    sim_cfg.physx.gpu_max_rigid_patch_count = 2**23
    sim_cfg.physx.gpu_collision_stack_size = 2**31
    sim_cfg.render.rendering_mode = "performance"
    sim_cfg.render.antialiasing_mode = "Off"
    return sim_cfg


@configclass
class PreciseAssemblyEnvCfg(PlayEnvCfg):
    sim: SimulationCfg = _default_precise_assembly_sim_cfg()
    assets: PreciseAssemblyAssetsCfg = PreciseAssemblyAssetsCfg()
    precise_assembly: PreciseAssemblyCfg = PreciseAssemblyCfg()


__all__ = [
    "PreciseAssemblyEnvCfg",
    "PreciseAssemblyAssetsCfg",
    "PreciseAssemblyCfg",
    "VALID_GOAL_MODES",
]
