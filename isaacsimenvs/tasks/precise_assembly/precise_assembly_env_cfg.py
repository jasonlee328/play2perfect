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
    # A7: > 0 = retract success also needs the held object's root more than this far (m, horizontally) from the turn
    # joint's axis (official Allen-key key_off_screw: the key may not be left in the screw).
    retract_object_off_joint_m: float = 0.0
    # Release curriculum (end_on_retract): ramp the release rule from (retract_ramp_start_dist, retract_ramp_start_clear)
    # to (retract_min_fingertip_dist, retract_clear_steps) over retract_ramp_steps env steps (0 = off); a resume restarts
    # common_step_counter, so set retract_ramp_offset_steps to continue.
    retract_ramp_steps: int = 0
    # True: retract reward = retract_reward_scale x the per-step GAIN in mean fingertip distance (capped at the current
    # release distance), so hovering away without releasing earns nothing (the default per-step distance reward pays it).
    retract_reward_delta: bool = False
    retract_ramp_offset_steps: int = 0
    retract_ramp_start_dist: float = 0.05
    retract_ramp_start_clear: int = 10
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
    # A7: actor + critic field screw_flats = [sin 4a, cos 4a] of the turn joint's angle (recess orientation mod 90 deg)
    obs_screw_flats: bool = False
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
    # one-time reward cut when an episode ends because the object fell (object z < 0.1 m, off the table)
    fall_terminal_penalty: float = 0.0
    # Eval: pick bank entry (start_state_offset + env_id) % bank size instead of a random one, so trial i always
    # starts from the same state.
    start_state_sequential: bool = False
    start_state_offset: int = 0
    # Hidden unlock angle (ContactBench doorknob, cb_doorknob_unlock): non-empty = every env gets its own unlock angle
    # theta (deg; uniform in [lo, hi], or unlock_angle_values_deg[i % K] when that is set), fixed for the env's
    # lifetime (PhysX joint limits are set when the scene is built), never observed. Joint unlock_joint's end stop
    # becomes theta + unlock_stop_margin_deg, and the problem's LAST waypoint is credited (instead of its keypoint
    # check) while the joint angle is >= theta - unlock_tol_deg (same near-goal step counting as other goals).
    unlock_angle_range_deg: tuple[float, ...] = ()
    unlock_angle_values_deg: tuple[float, ...] = ()
    unlock_stop_margin_deg: float = 2.0
    unlock_tol_deg: float = 3.0
    unlock_joint: int = 0
    unlock_seed: int = 0
    # Multi-turn hidden stop (ContactBench bottle cap, cb_bottle<k>; needs unlock_angle_*): the stop is the env's unlock
    # angle in deg (turns x 360, may exceed 360) on the UNWRAPPED joint angle, enforced in software every physics
    # substep (PhysX revolute limits can't span 2 pi and the GPU pipeline has no gear joint): past the stop the cap is
    # rotated back onto it and its tightening spin removed, so a finger pushing on a tight cap slides over it. No PhysX
    # limit is set. Goals come from the unwrapped angle: waypoint 0 at turn_first_goal_deg, the last at the stop (within
    # unlock_tol_deg); the "keypoint distance" = remaining angle x turn_progress_m_per_rad, which drives the keypoint
    # reward (dense turning reward), the critic's closest_keypoint_max_dist and the retract at-goal check.
    turn_mode: bool = False
    turn_lower_deg: float = -180.0
    turn_first_goal_deg: float = 30.0
    turn_progress_m_per_rad: float = 0.02
    # Actor obs hide the turn state: object_rot / keypoints use the cap's unturned orientation, object angular velocity
    # along the axis is removed, and the reward / successes / closest_keypoint_max_dist fields are zeroed (they would
    # reveal when the stop is reached). The critic keeps everything.
    turn_hide_from_actor: bool = True
    # Over-tightening ("stripped thread"): > 0 = fail once the cap has been within turn_stop_band_deg of its stop with
    # any fingertip pad in contact for more than this many steps in the episode (penalty turn_strip_penalty).
    turn_strip_steps: int = 0
    turn_strip_penalty: float = 0.0
    # Early release (end_on_retract): the hand fully released (the current release rule: every fingertip beyond the
    # release distance, no touch, for the clear steps) BEFORE the stop goal, after the cap has turned past
    # turn_first_goal_deg = fail `released_early` (-turn_early_release_penalty). Off by default.
    turn_early_release_fail: bool = False
    turn_early_release_penalty: float = 300.0
    turn_stop_band_deg: float = 3.0
    # What counts as "working the cap" at the stop for turn_strip_steps: > 0 = any fingertip within this distance (m)
    # of the cap centre (hovering on / pushing a tight cap); 0 = a fingertip tactile pad in contact.
    turn_strip_near_dist: float = 0.0
    # Hidden turning resistance: non-empty [lo, hi] = every env's turn joint gets its own viscous damping (N m s / rad,
    # log-uniform in [lo, hi], fixed for the env's life, never observed), replacing JointSpec.damping. A stiff cap then
    # feels like a locked one to the joint state, so the stop has to be told apart by touch (ribs sweeping past).
    turn_damping_range: tuple[float, ...] = ()
    # Diagnostic: false = the stop is not enforced (the cap turns freely past it; goals / strip still use the angle),
    # so the policy gets no physical signal at the stop. Tells a physical cue (stall) from an observation leak.
    turn_stop_physical: bool = True
    # Slip-clutch stop (torque-limiter cap): true = past the hidden stop the cap keeps turning exactly as before (no
    # physical stop, so the joint state carries nothing about it), while the fingertip tactile sensor ray-casts the cap
    # frozen at the stop angle: the felt surface (hull + tactile-only knurl) slips under the pads by the over-turn.
    # Touch is then the only cue. Pair with turn_strip_over_deg so a blind fixed number of turns strips the cap.
    # Hidden pre-turn: non-empty [lo, hi] (deg) = each episode starts as if the cap were already turned by
    # U(lo, hi) (the bookkeeping angle starts there; the cap pose is unchanged), so the stop comes after
    # stop - offset of actual turning: it lands at any phase of a turn and can't be predicted by counting.
    # The first goal and the lower bound are relative to the start; the actor's hidden cap pose ignores the offset.
    turn_start_offset_range_deg: tuple[float, ...] = ()
    turn_clutch: bool = False
    # Compliant stop for the clutch (curriculum): past the stop, each physics substep removes this fraction of the cap's
    # tightening spin (1 = stops dead each substep, like the hard stop; 0 = free clutch). Ramped from
    # turn_clutch_brake to turn_clutch_brake_end over turn_clutch_brake_ramp_steps env steps (+ _offset_steps).
    turn_clutch_brake: float = 0.0
    # Clutch coasting drag: past the stop, each substep removes this fraction of the cap's tightening spin (velocity only,
    # no pose correction). A cap driven by the fingers is unaffected (they set its motion through contact), but a cap
    # flicked and left spinning stops coasting past tight instead of free-wheeling into a strip.
    turn_clutch_coast_damp: float = 0.0
    turn_clutch_brake_end: float = 0.0
    turn_clutch_brake_ramp_steps: int = 0
    turn_clutch_brake_offset_steps: int = 0
    # > 0: fail `stripped` (-turn_strip_penalty) once the cap is turned more than this past its stop (any mode).
    turn_strip_over_deg: float = 0.0
    # Over-turn curriculum: > 0 = ramp the strip window from turn_strip_over_ramp_start_deg down to turn_strip_over_deg
    # (and, if turn_strip_steps_ramp_start > 0, the at-stop step limit from it down to turn_strip_steps) over this many
    # env steps (common_step_counter + turn_strip_over_ramp_offset_steps; a resume restarts the counter).
    turn_strip_over_ramp_steps: int = 0
    turn_strip_over_ramp_start_deg: float = 720.0
    turn_strip_steps_ramp_start: int = 0
    turn_strip_over_ramp_offset_steps: int = 0
    # Dense over-turn cost: reward -= this x (new max over-turn past the stop, rad) each step (total = scale x over).
    turn_over_penalty: float = 0.0
    # Faithful printed thread (official ContactBench sim, appendix_tasks/_kit_thread.py), turn_mode only. The turn joint
    # becomes cylindrical (turn + slide along the axis; PhysX limit = the lip seat) and every substep the cap's height
    # above the seat u is kept in the thread's band for its angle: with r = angle left to the seat, u in
    # [max(0, r L / 2 pi), r L / 2 pi + play] (self-locking: axial loads never turn it). L = the lead of the env's
    # revolution count (unlock_angle_values_deg / 360 -> turn_thread_leads_mm, else turn_thread_lead_k_mm / revs),
    # play = L - turn_thread_ridge_mm (2.9 mm at 2 rev, 0.6 at 4). Each episode the cap starts on the thread
    # turn_thread_engaged_mm above the seat less a drop U(0, play) (where it bites at a random phase); it descends L per
    # turn, lands on the lip, turns freely through the take-up (play / L turns) and LOCKS (hard stop = unlock_angle,
    # recomputed per episode; _unlock_deg_np keeps the nominal revolutions as the label). Tight = the torque the hand
    # pushes into the locked cap (estimated from the spin the stop removes: I_axis x omega / dt) >= turn_tight_torque
    # for turn_tight_steps consecutive steps; the last goal needs tight (a cap turned back > 1 deg off the lock loses
    # it). turn_strip_torque > 0: our own over-torque failure (stripped); the official task has none (0 = off).
    turn_thread: bool = False
    turn_thread_leads_mm: tuple[tuple[float, float], ...] = ((2.0, 4.665), (2.5, 3.75), (3.0, 3.13), (3.5, 2.68),
                                                            (4.0, 2.34))
    turn_thread_lead_k_mm: float = 9.33
    turn_thread_ridge_mm: float = 1.75
    turn_thread_engaged_mm: float = 9.4
    # A7 (OmniReset-style near-goal resets): non-empty [lo, hi] mm = the start height above the seat is U(lo, hi) per
    # episode (replaces turn_thread_engaged_mm - drop); 0 = seated. object_start_follow_thread: the held tool's start
    # pose (built for turn_thread_engaged_mm) moves along the axis with the actual start height.
    turn_thread_engaged_range_mm: tuple[float, ...] = ()
    object_start_follow_thread: bool = False
    # A7 training aid (insertion curriculum): non-empty [lo, hi] mm = the held tool's start pose is raised along the
    # hole axis by U(lo, hi') mm per episode, hi' = lo + (hi - lo) * min(1, (step + offset) / ramp) (ramp 0 = full range).
    # With object_start_hold_* the tool is pinned there, aligned above the recess, until the hand has it; the hand must
    # then insert it (the in-hand start, at a growing height). Off at eval.
    object_start_lift_mm: tuple[float, ...] = ()
    object_start_lift_ramp_steps: int = 0
    object_start_lift_offset_steps: int = 0
    # A7 (Allen key screw): thread drag on the jointed body, applied every substep at the velocity level (official
    # _allen_key ThreadModel): Coulomb running friction (N m; official TAU_RUN0 0.0015) + viscous (N m s / rad; official
    # SCREW_KD 0.004). Never reverses the spin, so an undriven light screw stops (stiction). 0 = off (bottle default).
    turn_thread_friction: float = 0.0
    turn_thread_viscous: float = 0.0   # unused (11-17: viscous drag = JointSpec.damping on the PhysX drive)
    turn_thread_stiction_w: float = 0.03   # rad/s: tanh width of the Coulomb friction torque
    # A9 (f652587): > 0 = elasto-plastic bristle stiction instead of the tanh Coulomb (official ThreadModel,
    # STICTION_K 2 N m/rad): bristle z integrates the spin, |k z| capped at the friction cap (running drag + seat
    # preload); torque k z + d w while sticking. The tanh model creeps below breakaway; the bristle holds.
    turn_thread_bristle_k: float = 0.0
    turn_thread_bristle_d: float = 0.02
    # A7 axial helix spring (N/m, N s/m; 0 = off = pose-level band only, which a press undoes within one physics step
    # for a light body): force toward the helix height + the body's weight; the pose band then only clamps errors
    # beyond turn_thread_axial_backstop_m. Needs a heavy enough jointed body (k dt^2 / m < 1).
    turn_thread_axial_k: float = 0.0
    turn_thread_axial_d: float = 0.0
    turn_thread_axial_backstop_m: float = 0.0015
    turn_thread_band_tol_m: float = 0.0   # A7: pose band writes only beyond this error (m); 0 = every substep
    turn_thread_gravity_comp: bool = False   # cancel the jointed body's weight (heavy body on the exact pose band)
    # Turn mode: the first N goals are ordinary object-pose goals (tool above the screw, tool engaged), the last two
    # the turn goals (turn_first_goal_deg, then the stop / tight). 0 = all goals are turn goals (bottle).
    turn_prelude_goals: int = 0
    # Seat preload (official _allen_key): the friction cap rises by this fraction of turn_lock_torque_max (the torque
    # pushed into the seat, per policy step) while within turn_preload_free_deg of the seat; 0 = off.
    turn_seat_preload_frac: float = 0.0
    # A7 tool rule (official turned_by_the_tool_in_hand): tool tips in the held object's frame (m); the jointed body may
    # turn at most turn_toolless_max_deg (0 = off) in total over policy steps where no tip is in the recess (radial
    # < turn_tool_radial_m from the axis, height < turn_tool_out_m above the head top = seat + turn_tool_head_top_m + u).
    turn_tool_tips_m: tuple[tuple[float, float, float], ...] = ()
    turn_tool_head_top_m: float = 0.007
    turn_tool_radial_m: float = 0.0015
    turn_tool_out_m: float = 0.0005
    turn_toolless_max_deg: float = 0.0
    turn_toolless_penalty: float = 0.0
    # A7 shaping (training only): per-step penalty scale x (tilt of the tool's long axis (object frame x) from the hole
    # axis - turn_tool_tilt_free_deg, rad, >= 0), so the key stays upright in the recess through regrasps (0 = off)
    turn_tool_tilt_penalty_scale: float = 0.0
    turn_tool_tilt_free_deg: float = 10.0
    # A7 training aid (curriculum, off in evals): a spring wrench that keeps an engaged tool in the recess through
    # regrasps (tip pulled to the axis, long axis (object x) to the hole axis), only while a tip is within
    # turn_capture_radius_m of the axis and turn_capture_band_m of the head top; strength fades 1 -> 0 over
    # turn_capture_ramp_steps env steps (+ _offset_steps). 0 stiffness = off.
    turn_capture_k: float = 0.0
    turn_capture_d: float = 0.4
    turn_capture_k_rot: float = 0.05
    turn_capture_d_rot: float = 0.002
    turn_capture_radius_m: float = 0.004
    turn_capture_band_m: float = 0.006
    turn_capture_ramp_steps: int = 0
    turn_capture_offset_steps: int = 0
    # A7: the turn-progress reward / turn goals count only while a tool tip is in the recess (no reward for turning the
    # head with the fingers).
    turn_progress_requires_tool: bool = False
    # A7: + turning progress only with >= N fingertip pads on the key (official both-pads-loaded turning); 0 = off
    turn_progress_requires_pads: int = 0
    # A7: per-step reward while >= turn_pad_grip_min fingertip pads touch the key (encourage a fingertip pinch)
    turn_pad_grip_rew: float = 0.0
    turn_pad_grip_min: int = 2
    turn_preload_free_deg: float = 10.0
    turn_tight_torque: float = 0.15
    turn_tight_steps: int = 9
    turn_strip_torque: float = 0.0
    # A7 'break' variant (generic; also A9's fragile screwdriver): > 0 = fail `stripped` (-turn_strip_penalty) once the
    # SUSTAINED torque into the lock (min over turn_tight_steps policy steps) exceeds this (N m). 0 = off (official).
    turn_break_torque: float = 0.0
    # Lock stiffness: the software stop zeroes the cap's spin every substep, which acts as a viscous brake of I_axis / dt
    # (~1.6e-4 N m s/rad for a 3.6 g cap: the hand would hardly feel the lock). While the cap is within
    # turn_lock_zone_deg of the lock its moment of inertia about the axis is raised to this value (kg m^2; 0 = off), so
    # the stop resists like a stiff brake (0.05 -> 6 N m s/rad: 0.3 N m creeps 3 deg/s, undone every substep) and the
    # spin it removes measures the torque pushed into the lock. Updated once per policy step.
    turn_lock_inertia: float = 0.05
    turn_lock_zone_deg: float = 0.5
    # Elastic lock (A1 11-16; 0 = off = the pose-level hard stop). While the raised lock inertia is active, the lock is
    # a spring + damper on the cap about its axis, tau = -(K over + D omega) for over = angle past the lock in
    # (0, turn_lock_backstop_deg] (explicit torque; stable only with the raised inertia, so a cap that reaches the lock
    # before the inertia switch still gets the hard stop that substep); past the backstop the pose stop holds it there.
    # The torque pushed into the lock is then the spring torque K over + D omega: a physical reading, bounded by what
    # the hand can push, unlike the hard stop's I omega / dt (which counts contact depenetration as torque). K default
    # 30 N m/rad = 0.52 N m per deg: official kit locks within ~1 deg at the 0.3-0.9 N m pad-slip peaks.
    turn_lock_stiffness: float = 0.0
    turn_lock_damping: float = 1.0
    turn_lock_backstop_deg: float = 5.0
    # PhysX position iterations of the object body (0 = 4, the default). A jointed knob's end stop is soft under the
    # hand's grip at 4 (the screw teacher pushed it 3-7 deg past the limit).
    object_position_iterations: int = 0
    distractor_position_iterations: int = 0   # A7: PhysX position iterations of the distractor bodies (0 = 4)
    # A7: PhysX velocity iterations of the object / distractor bodies (default 0). A light tool chattering in a
    # recess against a heavy jointed body needs velocity-level contact resolution.
    object_velocity_iterations: int = 0
    # A7 no-hand probe only: D6 joint GoalViz -> Object with an implicit yaw drive (N m/rad, N m s/rad); 0 = off.
    # A7: pin the object at its start pose for the first N policy steps of each episode (a tool in a holder; 0 = off).
    object_start_hold_steps: int = 0
    # A7: > 0 = after object_start_hold_steps keep holding until >= this many fingertip pads touch the object (a tool in
    # a holder until the hand has it), at most object_start_hold_max_steps policy steps.
    object_start_hold_until_grasp: int = 0
    object_start_hold_max_steps: int = 360
    # A7 grasp stage: goals are credited only after the holder released the object because the hand grasped it.
    object_start_hold_gate_goals: bool = False
    probe_key_driver_k: float = 0.0
    probe_key_driver_d: float = 0.0
    distractor_velocity_iterations: int = 0


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
