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
    # Multi-problem training/eval: env i gets multi_problems[i % K] (object, hole and goal-viz assets spawned
    # round-robin with MultiUsdFileCfg). All problems must share the insertion waypoints and hole_z_offset (true
    # for the fmb_* problems). Overrides `problem` when non-empty. Empty = single problem (legacy).
    multi_problems: tuple[str, ...] = ()
    goal_mode: str = "preInsertAndFinal"

    hole_x_range: tuple[float, float] = (-0.1875, 0.1875)
    hole_y_range: tuple[float, float] = (-0.1, 0.1)
    # Per-episode yaw randomization about +Z applied to the hole and to the
    # derived insertion goals. Range is symmetric: yaw ~ U(-range, +range).
    # Default 0.0 keeps legacy behavior (identity hole quat).
    hole_yaw_range_deg: float = 0.0
    # Fixture yaw = hole_yaw_center_deg + U(+-hole_yaw_range_deg) (A12 USB: official slot yaw 90 +- 20)
    hole_yaw_center_deg: float = 0.0
    joint_fixture_seed: int = 0

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
    # A12 USB (usb_task.py): "off" | "report" (contact views + verdict logged) | "gate" (verdict = task success)
    usb_task: str = "off"
    object_mass_kg: float = 0.0        # > 0: override the object's mass (inertia scaled alike) after the scene is built
    hand_stiffness_scale: float = 1.0  # A12: hand joint stiffness and damping x this (1 = unchanged)
    usb_settle_steps: int = 0          # steps after a start-bank reset before the checks start (pinch settles)
    usb_hold_steps: int = 36           # held_in_hand_at_start window (0.6 s)
    usb_pad_force: float = 0.2         # N, a link "carries" the stick
    usb_clear_force: float = 0.05      # N, official released_and_clear contact threshold
    usb_pinch_frac: float = 0.9        # finger_pinch_real: share of pre-release steps with the pinch loaded
    usb_pinch_mode: str = "mean_force"  # finger_pinch_real: "mean_force" (official, mean pad force >= usb_pinch_mean_n) | "frac"
    usb_pinch_mean_n: float = 3.0      # N per pad (plug_in_usb.py:591)
    usb_pinch_window: str = "insert"   # pad-force window: "insert" (official: hold end -> first seated) | "pre_release"
    usb_hand_lock_until_seated: bool = False   # hold the hand targets until the stick first seats (relative hand only)
    usb_seat_tol: float = 0.0005       # fully_inserted (plug_in_usb.py:153)
    usb_release_steps: int = 3
    usb_clear_steps: int = 10
    usb_clear_dist: float = 0.04       # hand moved >= 40 mm after the release
    usb_push_force: float = 8.0        # N, env-applied push along -slot z (plug_in_usb.py:136-139)
    usb_push_steps: int = 36           # 0.6 s
    usb_push_gain_tol: float = 0.0003  # no_motion_when_pushed (l.154)
    usb_stay_tol: float = 0.0003       # stays_seated_after_release (l.155)
    usb_stay_mode: str = "depth"       # stays_seated: "depth" (official, axial depth change) | "pos" (old 3D drift)
    usb_unseated_steps: int = 30       # released + untouched + not inserted for this long = fail
    usb_terminate_on_fail: bool = True
    usb_success_bonus: float = 1000.0
    usb_release_bonus: float = 200.0   # released while fully inserted (once)
    usb_fail_penalty: float = 200.0
    usb_clear_rew_scale: float = 0.0   # per step, x min(palm distance from release, usb_clear_dist), released + inserted
    usb_hold_rew: float = 0.0          # per step while pinched and not released
    usb_seat_rew_scale: float = 0.0    # potential: x change of -seat_err (m), before the release
    usb_lat_rew_scale: float = 0.0     # potential: x change of -(plug tip distance off the slot axis, m), before the release
    usb_pad_rew_scale: float = 0.0     # per step before the release x min(thumb, index pad force, cap) / cap
    usb_pad_rew_cap: float = 5.0       # N
    # Clutter problems (Problem.distractors): per-step reward -= distractor_penalty_scale * the largest distractor
    # displacement from its reset pose (m); with terminate_on_disturb the episode ends once it exceeds
    # distractor_disturb_tol. Both off by default (and no effect on problems without distractors).
    distractor_penalty_scale: float = 0.0
    distractor_disturb_tol: float = 0.01
    terminate_on_disturb: bool = False
    # Clutter "stay in the box" mode: (xmin, xmax, ymin, ymax, zmax) in m, receptive frame. When set, the penalty and
    # termination above use how far the furthest distractor's root is outside this box (0 while inside) instead of
    # its displacement, so neighbours may be pushed around inside the bin but not lifted or dragged out of it.
    distractor_box: tuple[float, ...] = ()
    # Dense "don't lift the neighbours" shaping: per-step reward -= distractor_lift_penalty_scale * the largest
    # distractor rise above its reset height beyond distractor_lift_slack (m). Sliding / rolling inside the bin is free.
    distractor_lift_penalty_scale: float = 0.0
    distractor_lift_slack: float = 0.005
    # Privileged teacher obs: each distractor's position relative to the palm and to the object (6 per distractor,
    # field "distractor_obs", added to obs_list and state_list). A checkpoint trained without it warm-starts with
    # the new inputs zero-initialized when P2P_NEW_OBS_FIELD=distractor_obs (train.py).
    obs_distractors: bool = False
    # > 0: feed only the obs_distractors_k distractors whose reset pose is nearest the object's start (kept in
    # problem order), so a policy trained with k neighbours can run on a layout with more. 0 = all of them.
    obs_distractors_k: int = 0
    # Curriculum over env steps (common_step_counter): the disturb tolerance shrinks linearly from
    # distractor_disturb_tol_start to distractor_disturb_tol, and the distractor penalties ramp from 0 to their
    # scales, over distractor_curriculum_steps. 0 = off (the final values from the start).
    distractor_curriculum_steps: int = 0
    distractor_disturb_tol_start: float = 0.10
    distractor_curriculum_offset_steps: int = 0  # added to common_step_counter (set on resume)
    # One-time reward subtracted when an episode ends because a distractor left the box / moved too far.
    distractor_disturb_terminal_penalty: float = 0.0
    # A1 hand force (physical, PhysX contact report on the object filtered to the robot's links): hand_force = sum over
    # robot links of |contact force on the object| (N; a two-finger pinch at N each reads 2N), every policy step, last
    # substep. Logged as extras hand_force_{mean,max}; per-episode max in self.hand_force_ep_max (evals). Upper-bound
    # rule: hand_force > crush_force (N, 0 = off) for crush_steps steps in a row = termination "crushed" (- crush_penalty).
    hand_force_sense: bool = False
    hand_force_scale: float = 0.5   # calibration of the PhysX force matrix (reads 2x on a flat contact)
    crush_force: float = 0.0
    crush_steps: int = 3
    crush_penalty: float = 200.0
    # Eval: pick bank entry (start_state_offset + env_id) % bank size instead of a random one, so trial i always
    # starts from the same state.
    start_state_sequential: bool = False
    start_state_offset: int = 0

    # Object / distractor rigid bodies: PhysX position iterations (4 = unchanged) and, when not None, the collision
    # rest offset (m, may be negative) of the object, distractor and receptive shapes (None: play scene_utils
    # _REST_OFFSET = 0). A pair rests at the sum of its two shapes' rest offsets.
    object_solver_position_iterations: int = 4
    object_rest_offset: float = 0.0   # 0.0 = the play default _REST_OFFSET (a None default rejects float overrides)

    # Official-style ("strict") success for the last goal, from the ContactBench sim's retrieve_top_cup_from_stack
    # verdict (force_bench appendix_tasks): the object upright (tilt < strict_max_tilt_deg) and every distractor
    # still where it was reset ("stack_remains_neat": reference point strict_ref_local (object frame, m; the cup's
    # base centre) within strict_neat_axis_tol sideways and strict_neat_step_tol vertically of its reset position,
    # tilt < strict_max_tilt_deg). "off" = old rule; "report" = old rule, but the checks at the moment the last goal
    # is credited are recorded (env._strict_at_success, for evals); "gate" = the last goal only counts while they hold.
    strict_success: str = "off"
    strict_max_tilt_deg: float = 3.0
    strict_neat_axis_tol: float = 0.002
    strict_neat_step_tol: float = 0.0015
    strict_ref_local: tuple[float, float, float] = (0.0, 0.0, 0.0)


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
