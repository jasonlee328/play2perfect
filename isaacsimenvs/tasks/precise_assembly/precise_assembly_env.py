"""Dynamic peg-in-hole task built as a PlayEnv variant."""

from __future__ import annotations

import sys
from pathlib import Path

import math

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_from_angle_axis, quat_mul

from isaacsimenvs.tasks.play.play_env import PlayEnv
from isaacsimenvs.tasks.play.utils.goal_sampling import (
    sample_absolute_goal_pose,
    sample_delta_goal_pose,
)
from isaacsimenvs.tasks.play.utils.logging_utils import log_step_metrics
from isaacsimenvs.tasks.play.utils.obs_utils import (
    OBS_FIELD_SIZES,
    _keypoints_world,
    build_observations,
    compute_intermediate_values,
)
from isaacsimenvs.tasks.play.utils.reward_utils import compute_rewards
from isaacsimenvs.tasks.play.utils.termination_utils import (
    update_tolerance_curriculum,
)

from .precise_assembly_env_cfg import PreciseAssemblyEnvCfg, VALID_GOAL_MODES
from .scene_utils import REPO_ROOT, setup_scene


TABLE_HALF_HEIGHT = 0.15


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=-1)


def _obs_field_slice(fields: tuple[str, ...], field: str) -> slice | None:
    offset = 0
    for name in fields:
        size = OBS_FIELD_SIZES[name]
        if name == field:
            return slice(offset, offset + size)
        offset += size
    return None


def _resolve_asset_path(path: str | Path) -> Path:
    asset_path = Path(path)
    if asset_path.is_absolute():
        return asset_path

    candidates = (REPO_ROOT / asset_path, REPO_ROOT / "assets" / asset_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve asset path {path!r}; checked {checked}")


class PreciseAssemblyEnv(PlayEnv):
    cfg: PreciseAssemblyEnvCfg

    def __init__(
        self, cfg: PreciseAssemblyEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        self._configure_problem(cfg)

        super().__init__(cfg, render_mode, **kwargs)

        # When student_obs is enabled, the env exposes three obs groups via
        # `_get_observations`: "policy" = student obs (image+proprio, what
        # the depth-CNN reads), "critic" = state_list (privileged), and
        # "teacher_obs" = obs_list (proprio + object_state, noisy — what the
        # frozen state-MLP teacher reads). The default RlGamesVecEnvWrapper
        # drops keys it doesn't know; use DAggerRlGamesVecEnvWrapper from
        # `isaacsimenvs.utils.rlgames_utils` to pass "teacher_obs" through to
        # the agent as `self.obs["teacher"]`.
        student_cfg = getattr(cfg, "student_obs", None)
        if student_cfg is not None and student_cfg.enabled and student_cfg.image_enabled:
            from gymnasium import spaces
            import numpy as _np
            from isaacsimenvs.tasks.play.utils.obs_utils import _student_proprio_dict

            image_numel = (
                int(student_cfg.image_input_height)
                * int(student_cfg.image_input_width)
                * (1 if student_cfg.image_modality.lower() == "depth" else 3)
                * (2 if getattr(student_cfg, "view2_enabled", False) else 1)
            )
            proprio_sample = _student_proprio_dict(self)
            proprio_dim = sum(
                int(proprio_sample[field].reshape(self.num_envs, -1).shape[-1])
                for field in student_cfg.proprio_list
            )
            student_dim = image_numel + proprio_dim
            actor_dim = int(cfg.observation_space)        # original obs_list dim (teacher actor input)
            critic_dim = int(cfg.state_space)             # state_list dim (asymmetric critic input)

            cfg.observation_space = student_dim
            box = lambda d: spaces.Box(low=-_np.inf, high=_np.inf, shape=(d,))
            self.observation_space = spaces.Dict({
                "policy": box(student_dim),
                "critic": box(critic_dim),
                "teacher_obs": box(actor_dim),
            })
            self.single_observation_space = spaces.Dict({
                "policy": box(student_dim),
                "critic": box(critic_dim),
                "teacher_obs": box(actor_dim),
            })
            self._teacher_actor_obs_dim = actor_dim

        insert_poses = torch.as_tensor(
            self._pih_insert_pose_sequence, dtype=torch.float32, device=self.device
        )
        self._insert_pos_rel = insert_poses[:, 0:3].contiguous()
        self._insert_quat_wxyz = _xyzw_to_wxyz(insert_poses[:, 3:7]).contiguous()

        self.hole_pos = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.hole_quat_wxyz = torch.zeros(
            self.num_envs, 4, dtype=torch.float32, device=self.device
        )
        self.hole_quat_wxyz[:, 0] = 1.0  # identity default
        self.is_random_goal_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.prev_episode_is_random_goal = torch.zeros_like(self.is_random_goal_env)

        self.env_max_goals = torch.full(
            (self.num_envs,),
            self._num_total_insertion_goals,
            dtype=torch.long,
            device=self.device,
        )
        pih_cfg = self.cfg.precise_assembly
        random_goal_fraction = float(pih_cfg.random_goal_fraction)
        if random_goal_fraction > 0.0:
            self.is_random_goal_env[:] = (
                torch.rand(self.num_envs, device=self.device) < random_goal_fraction
            )
            random_goal_max = torch.full_like(
                self.env_max_goals, int(pih_cfg.random_goal_max_successes)
            )
            insertion_max = torch.full_like(
                self.env_max_goals, self._num_total_insertion_goals
            )
            self.env_max_goals[:] = torch.where(
                self.is_random_goal_env, random_goal_max, insertion_max
            )

        # Per-env world-frame prelude waypoints (lift_in_place, over_hole)
        # rebuilt at every reset for transportPreInsertFinal mode. Stored as
        # (pos_xyz, quat_wxyz) to match goal_viz convention.
        self._prelude_pose_world = torch.zeros(
            self.num_envs,
            max(self._num_prelude_goals, 1),
            7,
            dtype=torch.float32,
            device=self.device,
        )

        self.prev_episode_env_max_goals = self.env_max_goals.clone()
        self.prev_episode_is_random_goal[:] = self.is_random_goal_env

        self.insertion_success_tolerance = float(pih_cfg.insertion_success_tolerance)
        self.retract_success_bonus = float(
            int(pih_cfg.random_goal_max_successes) * self.cfg.reward.reach_goal_bonus
        )
        self.lift_bonus_active = True

        self.retract_phase = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.retract_succeeded = torch.zeros_like(self.retract_phase)
        self.retract_failed = torch.zeros_like(self.retract_phase)
        self._just_entered_retract = torch.zeros_like(self.retract_phase)
        self._just_retracted = torch.zeros_like(self.retract_phase)
        # allow_goal_skip: furthest later waypoint currently near, and steps held there.
        self._skip_cand = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._skip_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.goal_pos_obs_noise = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        # Per-env yaw noise (radians) applied to the observed goal orientation.
        self.goal_yaw_obs_noise = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._goal_kp_obs_slice = _obs_field_slice(
            tuple(cfg.obs.obs_list), "keypoints_rel_goal"
        )
        # hidden unlock angle (rad, per env) + the furthest the knob got this episode (for evals)
        self.unlock_angle = None
        if self._unlock_deg_np is not None:
            self.unlock_angle = torch.as_tensor(self._unlock_deg_np, device=self.device, dtype=torch.float32).deg2rad()
            self.unlock_max_angle = torch.zeros(self.num_envs, device=self.device)
            self.unlock_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.unlock_prev_ep_max = torch.zeros(self.num_envs, device=self.device)
            print(f"[PreciseAssemblyEnv] unlock angle per env: {self._unlock_deg_np.min():.1f}-"
                  f"{self._unlock_deg_np.max():.1f} deg (stop + {pih_cfg.unlock_stop_margin_deg}, "
                  f"tol {pih_cfg.unlock_tol_deg})", flush=True)
        # multi-turn hidden stop (bottle cap): unwrapped joint angle, tracked every physics substep
        self.turn_mode = bool(pih_cfg.turn_mode)
        if self.turn_mode:
            if self.unlock_angle is None:
                raise ValueError("precise_assembly.turn_mode needs unlock_angle_values_deg / unlock_angle_range_deg")
            n = self.num_envs
            self.turn_angle = torch.zeros(n, device=self.device)          # unwrapped, rad, + = tightening
            self.turn_offset = torch.zeros(n, device=self.device)         # hidden pre-turn, rad
            self._turn_prev_wrapped = torch.zeros(n, device=self.device)
            self.turn_max_angle = torch.zeros(n, device=self.device)
            self.turn_prev_ep_max = torch.zeros(n, device=self.device)
            self.turn_strip_count = torch.zeros(n, dtype=torch.long, device=self.device)
            self.turn_stripped = torch.zeros(n, dtype=torch.bool, device=self.device)
            # A7: screw rotation while no tool tip is in the recess (deg) and its failure flag
            self.turn_toolless_deg = torch.zeros(n, device=self.device)
            self.turn_toolless = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._turn_just_toolless = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._turn_prev_step_angle = torch.zeros(n, device=self.device)
            self._turn_just_stripped = torch.zeros(n, dtype=torch.bool, device=self.device)
            self.turn_released_early = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._turn_just_released_early = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._turn_early_clear = torch.zeros(n, dtype=torch.long, device=self.device)
            self.turn_at_stop_steps = torch.zeros(n, dtype=torch.long, device=self.device)
            self.turn_prev_ep_at_stop_steps = torch.zeros(n, dtype=torch.long, device=self.device)
            self.turn_thread = bool(pih_cfg.turn_thread)
            if self.turn_thread:
                import numpy as np
                if pih_cfg.turn_clutch:
                    raise ValueError("precise_assembly.turn_thread and turn_clutch are exclusive")
                revs = self._unlock_deg_np / 360.0
                table = {round(float(r), 3): float(l) for r, l in pih_cfg.turn_thread_leads_mm}
                lead_mm = np.array([table.get(round(float(r), 3), float(pih_cfg.turn_thread_lead_k_mm) / float(r))
                                    for r in revs])
                self.turn_lead = torch.as_tensor(lead_mm * 1e-3, device=self.device, dtype=torch.float32)   # m / turn
                self.turn_play = torch.clamp(self.turn_lead - float(pih_cfg.turn_thread_ridge_mm) * 1e-3, min=0.0)
                self.turn_u_start = torch.zeros(n, device=self.device)        # start height above the seat (m)
                self.turn_seat_angle = torch.zeros(n, device=self.device)     # unwrapped angle at which it lands
                self.turn_tau_acc = torch.zeros(n, device=self.device)        # torque pushed into the lock (sum)
                self._turn_tau_n = 0
                self.turn_lock_torque = torch.zeros(n, device=self.device)    # per-step mean (N m)
                self.turn_lock_torque_max = torch.zeros(n, device=self.device)
                self.turn_prev_ep_lock_torque_max = torch.zeros(n, device=self.device)
                self.turn_tight_count = torch.zeros(n, dtype=torch.long, device=self.device)
                self.turn_tight = torch.zeros(n, dtype=torch.bool, device=self.device)
                self.turn_prev_ep_tight = torch.zeros(n, dtype=torch.bool, device=self.device)
                self.turn_u = torch.zeros(n, device=self.device)
                self._turn_I_axis = None
                self.turn_lock_flag = torch.zeros(n, dtype=torch.bool, device=self.device)
                self._turn_I_orig = None
                print(f"[PreciseAssemblyEnv] turn thread: leads {sorted(set(np.round(lead_mm, 3)))} mm, ridge "
                      f"{pih_cfg.turn_thread_ridge_mm} mm, engaged {pih_cfg.turn_thread_engaged_mm} mm, tight >= "
                      f"{pih_cfg.turn_tight_torque} N m x {pih_cfg.turn_tight_steps} steps, strip torque "
                      f"{pih_cfg.turn_strip_torque}", flush=True)
            print(f"[PreciseAssemblyEnv] turn mode: software stop at {self._unlock_deg_np.min():.0f}-"
                  f"{self._unlock_deg_np.max():.0f} deg, strip after {pih_cfg.turn_strip_steps} steps at the stop "
                  f"with contact, hide from actor {pih_cfg.turn_hide_from_actor}", flush=True)

        print(
            f"[PreciseAssemblyEnv] problem={self._pih_problem_name} "
            f"object={cfg.assets.object_name} "
            f"goals={self._num_total_insertion_goals} "
            f"(prelude={self._num_prelude_goals}, tail={self._num_insertion_goals}) "
            f"goal_mode={pih_cfg.goal_mode} "
            f"random_goal_fraction={random_goal_fraction}",
            flush=True,
        )

    def _configure_problem(self, cfg: PreciseAssemblyEnvCfg) -> None:
        pih_cfg = cfg.precise_assembly
        if pih_cfg.goal_mode not in VALID_GOAL_MODES:
            raise ValueError(
                f"goal_mode must be one of {VALID_GOAL_MODES}, got {pih_cfg.goal_mode!r}"
            )

        random_goal_fraction = float(pih_cfg.random_goal_fraction)
        if not 0.0 <= random_goal_fraction <= 1.0:
            raise ValueError(
                "precise_assembly.random_goal_fraction must be in [0, 1], "
                f"got {random_goal_fraction}"
            )

        repo_root_str = str(REPO_ROOT)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from evaluation.objects import NAME_TO_OBJECT
        from evaluation.problems import PROBLEM_REGISTRY

        multi = pih_cfg.multi_problems or ()
        if isinstance(multi, str):  # "a,b" / "[a,b]" from a plain-string override
            multi = [p.strip() for p in multi.strip("[]").split(",") if p.strip()]
        multi = [str(p) for p in multi]
        problem_names = multi if multi else [str(pih_cfg.problem)]
        for name in problem_names:
            if name not in PROBLEM_REGISTRY:
                raise KeyError(
                    f"Unknown peg-in-hole problem {name!r}; "
                    f"known problems: {sorted(PROBLEM_REGISTRY)}"
                )
            obj_name = PROBLEM_REGISTRY[name].insertion_object_name
            if obj_name not in NAME_TO_OBJECT:
                raise KeyError(
                    f"Problem {name!r} references object {obj_name!r}, but it is not in NAME_TO_OBJECT."
                )
        problem_name = problem_names[0] if len(problem_names) == 1 else "+".join(problem_names)
        problem = PROBLEM_REGISTRY[problem_names[0]]
        for name in problem_names[1:]:
            other = PROBLEM_REGISTRY[name]
            same = len(other.insert_pose_rel_receptive) == len(problem.insert_pose_rel_receptive) and all(
                abs(a - b) < 1e-9
                for pa, pb in zip(other.insert_pose_rel_receptive, problem.insert_pose_rel_receptive)
                for a, b in zip(pa, pb)
            ) and abs(other.hole_z_offset - problem.hole_z_offset) < 1e-9 and (
                other.prelude_lift_offset == problem.prelude_lift_offset)
            if not same:
                raise ValueError(
                    f"multi_problems must share insertion waypoints / hole_z_offset: {name!r} differs from "
                    f"{problem_names[0]!r}"
                )
        object_specs = [NAME_TO_OBJECT[PROBLEM_REGISTRY[n].insertion_object_name] for n in problem_names]
        self._pih_problem_names = tuple(problem_names)
        self._pih_object_urdf_abs_list = tuple(str(_resolve_asset_path(o.urdf_path)) for o in object_specs)
        self._pih_receptive_urdf_abs_list = tuple(
            str(_resolve_asset_path(PROBLEM_REGISTRY[n].receptive_urdf)) for n in problem_names
        )
        self._pih_object_scale_list = tuple(tuple(float(v) for v in o.scale) for o in object_specs)
        object_spec = object_specs[0]
        object_urdf = _resolve_asset_path(object_spec.urdf_path)
        receptive_urdf = _resolve_asset_path(problem.receptive_urdf)

        insert_pose_sequence = problem.insert_pose_rel_receptive
        if pih_cfg.goal_mode == "finalGoalOnly":
            insert_pose_sequence = (problem.final_insert_pose_rel_receptive,)
        if len(insert_pose_sequence) == 0:
            raise ValueError(f"Problem {problem_name!r} has no insertion subgoals.")

        self._pih_problem_name = problem_name
        self._pih_problem = problem
        self._pih_object_urdf_abs = str(object_urdf)
        self._pih_receptive_urdf_abs = str(receptive_urdf)
        self._pih_object_scale = tuple(float(v) for v in object_spec.scale)
        self._pih_hole_z_offset = float(problem.hole_z_offset)
        self._pih_insert_pose_sequence = tuple(insert_pose_sequence)
        self._num_insertion_goals = len(self._pih_insert_pose_sequence)

        # transportPreInsertFinal prepends two per-episode prelude waypoints
        # (lift-in-place, over-hole) onto the hole-frame insertion sequence.
        # The lift height is carried by the Problem (prelude_lift_offset).
        if pih_cfg.goal_mode == "transportPreInsertFinal":
            if float(problem.prelude_lift_offset) <= 0.0:
                raise ValueError(
                    f"goal_mode='transportPreInsertFinal' requires problem "
                    f"{problem_name!r} to set prelude_lift_offset > 0; got "
                    f"{problem.prelude_lift_offset}."
                )
            self._num_prelude_goals = 2
            self._prelude_lift_offset = float(problem.prelude_lift_offset)
        else:
            self._num_prelude_goals = 0
            self._prelude_lift_offset = 0.0
        self._num_total_insertion_goals = (
            self._num_insertion_goals + self._num_prelude_goals
        )

        # Clutter: distractor bodies + an optional object start pose, both in the receptive frame (single problem).
        if len(problem_names) > 1 and any(
                PROBLEM_REGISTRY[n].distractors or PROBLEM_REGISTRY[n].object_start_pose_rel_receptive
                for n in problem_names):
            raise ValueError("distractors / object_start_pose_rel_receptive need a single problem")
        self._pih_distractor_urdf_abs = tuple(str(_resolve_asset_path(u)) for u, _ in problem.distractors)
        self._pih_distractor_pose_rel = tuple(tuple(float(v) for v in pose) for _, pose in problem.distractors)
        self._pih_object_start_rel = tuple(float(v) for v in problem.object_start_pose_rel_receptive)
        self._obs_distractor_idx = None
        if cfg.precise_assembly.obs_distractors and self._pih_distractor_urdf_abs:
            n_d = len(self._pih_distractor_urdf_abs)
            k = int(cfg.precise_assembly.obs_distractors_k)
            if 0 < k < n_d:
                o = self._pih_object_start_rel[:3]
                d2 = [sum((a - b) ** 2 for a, b in zip(p[:3], o)) for p in self._pih_distractor_pose_rel]
                self._obs_distractor_idx = sorted(sorted(range(n_d), key=lambda i: d2[i])[:k])
                n_d = k
            OBS_FIELD_SIZES["distractor_obs"] = 6 * n_d
            for lst in ("obs_list", "state_list"):
                if "distractor_obs" not in getattr(cfg.obs, lst):
                    setattr(cfg.obs, lst, (*getattr(cfg.obs, lst), "distractor_obs"))
        if cfg.precise_assembly.obs_screw_flats:
            # A7: [sin 4a, cos 4a] of the jointed body's turn angle = the square recess's orientation mod 90 deg
            # (what the official expert reads to re-insert the key); carries no progress / seat information
            OBS_FIELD_SIZES["screw_flats"] = 2
            for lst in ("obs_list", "state_list"):
                if "screw_flats" not in getattr(cfg.obs, lst):
                    setattr(cfg.obs, lst, (*getattr(cfg.obs, lst), "screw_flats"))
        if len(problem_names) > 1 and any(PROBLEM_REGISTRY[n].joints for n in problem_names):
            raise ValueError("Problem.joints need a single problem")
        self._pih_joints = tuple(problem.joints)
        for j in self._pih_joints:
            if j.body != "object" and not (j.body.startswith("distractor")
                                           and int(j.body[10:]) < len(self._pih_distractor_urdf_abs)):
                raise ValueError(f"JointSpec.body must be 'object' or 'distractor<k>', got {j.body!r}")
            if j.body == "object" and not self._pih_object_start_rel:
                raise ValueError("a joint on the object needs Problem.object_start_pose_rel_receptive")
        # Jointed problems: one fixture pose per env for its lifetime (see scene_utils._create_fixture_joints)
        self._joint_hole_np = None
        if self._pih_joints:
            import numpy as np
            rng = np.random.default_rng(int(pih_cfg.unlock_seed) + 1)
            n_env = int(cfg.scene.num_envs)
            hz = float(cfg.reset.table_reset_z) + TABLE_HALF_HEIGHT + float(problem.hole_z_offset)
            self._joint_hole_np = np.stack([rng.uniform(*[float(v) for v in pih_cfg.hole_x_range], n_env),
                                            rng.uniform(*[float(v) for v in pih_cfg.hole_y_range], n_env),
                                            np.full(n_env, hz)], -1)
            if float(cfg.reset.table_reset_z_range) != 0.0 or float(pih_cfg.hole_yaw_range_deg) != 0.0:
                print("[PreciseAssemblyEnv] WARNING: jointed problem with table_reset_z_range / hole_yaw_range_deg "
                      "!= 0: the fixture keeps its spawn pose (use env.reset.table_reset_z_range=0.0)", flush=True)
        self._unlock_deg_np = None
        if pih_cfg.unlock_angle_range_deg or pih_cfg.unlock_angle_values_deg:
            import numpy as np
            if not (0 <= int(pih_cfg.unlock_joint) < len(self._pih_joints)):
                raise ValueError("unlock_angle_*: the problem has no joint unlock_joint")
            n_env = int(cfg.scene.num_envs)
            vals = [float(v) for v in pih_cfg.unlock_angle_values_deg]
            if vals:
                self._unlock_deg_np = np.array([vals[i % len(vals)] for i in range(n_env)], dtype=np.float64)
            else:
                lo, hi = (float(v) for v in pih_cfg.unlock_angle_range_deg)
                self._unlock_deg_np = np.random.default_rng(int(pih_cfg.unlock_seed)).uniform(lo, hi, n_env)

        self._turn_damping_np = None
        if pih_cfg.turn_damping_range:
            import numpy as np
            lo, hi = (float(v) for v in pih_cfg.turn_damping_range)
            u = np.random.default_rng(int(pih_cfg.unlock_seed) + 2).uniform(0.0, 1.0, int(cfg.scene.num_envs))
            self._turn_damping_np = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))

        cfg.assets.object_name = problem.insertion_object_name if len(problem_names) == 1 else "multi"
        cfg.assets.object_urdf = self._pih_object_urdf_abs
        cfg.assets.receptive_urdf = self._pih_receptive_urdf_abs
        cfg.assets.object_scale = self._pih_object_scale
        cfg.assets.table_urdf = "assets/urdf/table_narrow.urdf"

        random_max = int(pih_cfg.random_goal_max_successes)
        if random_goal_fraction > 0.0:
            cfg.termination.max_consecutive_successes = max(
                self._num_total_insertion_goals, random_max
            )
        else:
            cfg.termination.max_consecutive_successes = self._num_total_insertion_goals

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "prev_episode_env_max_goals"):
            self.prev_episode_env_max_goals[env_ids] = self.env_max_goals[env_ids]
            self.prev_episode_is_random_goal[env_ids] = self.is_random_goal_env[env_ids]
        super()._reset_idx(env_ids)
        self._reset_peg_episode(env_ids)

    def _reset_peg_episode(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        pih_cfg = self.cfg.precise_assembly
        env_origins = self.scene.env_origins[env_ids]
        is_random_goal = self.is_random_goal_env[env_ids]

        insertion_max = torch.full(
            (n,), self._num_total_insertion_goals, dtype=torch.long, device=self.device
        )
        random_max = torch.full(
            (n,),
            int(pih_cfg.random_goal_max_successes),
            dtype=torch.long,
            device=self.device,
        )
        self.env_max_goals[env_ids] = torch.where(
            is_random_goal, random_max, insertion_max
        )

        table_top_z = self._table_z_per_env[env_ids] + TABLE_HALF_HEIGHT

        # Object pose was already written by PlayEnv._reset_object_pose
        # (called from super()._reset_idx via reset_env_state). It honors
        # cfg.reset.fixed_start_pose when set, otherwise samples
        # cfg.reset.reset_position_noise_x/y/z + random_orientation. We do not
        # override it here.

        hole_x_min, hole_x_max = (float(v) for v in pih_cfg.hole_x_range)
        hole_y_min, hole_y_max = (float(v) for v in pih_cfg.hole_y_range)
        self.hole_pos[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            hole_x_min, hole_x_max
        )
        self.hole_pos[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            hole_y_min, hole_y_max
        )
        self.hole_pos[env_ids, 2] = table_top_z + self._pih_hole_z_offset
        if self._joint_hole_np is not None:  # jointed problem: the env's fixed fixture pose
            if not hasattr(self, "_joint_hole_t"):
                self._joint_hole_t = torch.as_tensor(self._joint_hole_np, device=self.device, dtype=torch.float32)
            self.hole_pos[env_ids] = self._joint_hole_t[env_ids]
        if is_random_goal.any():
            rg_ids = env_ids[is_random_goal]
            self.hole_pos[rg_ids, 0:2] = 0.0
            self.hole_pos[rg_ids, 2] = -1.0

        yaw_range_deg = float(pih_cfg.hole_yaw_range_deg) if self._joint_hole_np is None else 0.0
        if yaw_range_deg > 0.0:
            yaw = (
                torch.rand(n, device=self.device) * 2.0 - 1.0
            ) * yaw_range_deg * (math.pi / 180.0)
            z_axis = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            ).expand(n, -1)
            hole_quat = quat_from_angle_axis(yaw, z_axis)
        else:
            hole_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
            ).unsqueeze(0).expand(n, -1).contiguous()
        self.hole_quat_wxyz[env_ids] = hole_quat
        hole_pose = torch.cat([self.hole_pos[env_ids] + env_origins, hole_quat], dim=-1)
        self.hole.write_root_pose_to_sim(hole_pose, env_ids=env_ids)
        self.hole.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

        self.goal_pos_obs_noise[env_ids] = 0.0
        self.goal_yaw_obs_noise[env_ids] = 0.0
        noise = float(pih_cfg.goal_xy_obs_noise)
        insertion_mask = ~is_random_goal
        if noise > 0.0 and insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            self.goal_pos_obs_noise[ins_ids, 0:2] = torch.empty(
                ins_ids.numel(), 2, device=self.device
            ).uniform_(-noise, noise)
        yaw_noise_deg = float(pih_cfg.goal_yaw_obs_noise_deg)
        if yaw_noise_deg > 0.0 and insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            yaw_max = yaw_noise_deg * (math.pi / 180.0)
            self.goal_yaw_obs_noise[ins_ids] = torch.empty(
                ins_ids.numel(), device=self.device
            ).uniform_(-yaw_max, yaw_max)

        self.retract_phase[env_ids] = False
        self.retract_succeeded[env_ids] = False
        self.retract_failed[env_ids] = False
        if not hasattr(self, "_retract_measured"):
            self._retract_measured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._retract_meas = {}
        self._retract_measured[env_ids] = False
        if hasattr(self, "_retract_clear_count"):
            self._retract_clear_count[env_ids] = 0
        self._just_entered_retract[env_ids] = False
        self._just_retracted[env_ids] = False

        # Build the per-env world-frame prelude (lift_in_place, over_hole) for
        # insertion-only envs when goal_mode=transportPreInsertFinal. Both
        # waypoints share the pre-insert orientation (so the policy reorients
        # during the lift) and sit at the same height = start_z + lift_offset.
        if self._num_prelude_goals > 0 and insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            # post-reset peg world position (local to env_origin)
            start_pos = (
                self.object.data.root_pos_w[ins_ids]
                - self.scene.env_origins[ins_ids]
            )
            lift_z = start_pos[:, 2] + self._prelude_lift_offset
            # Pre-insert orientation is the orientation of the first hole-frame
            # tail waypoint (transport_above / pre_insert / final all share it).
            tail0_q = self._insert_quat_wxyz[0].unsqueeze(0).expand(ins_ids.numel(), -1)
            pre_insert_quat_world = quat_mul(self.hole_quat_wxyz[ins_ids], tail0_q)
            # waypoint 0: lift_in_place — directly above start XY
            self._prelude_pose_world[ins_ids, 0, 0] = start_pos[:, 0]
            self._prelude_pose_world[ins_ids, 0, 1] = start_pos[:, 1]
            self._prelude_pose_world[ins_ids, 0, 2] = lift_z
            self._prelude_pose_world[ins_ids, 0, 3:7] = pre_insert_quat_world
            # waypoint 1: over_hole — directly above hole XY, same height
            self._prelude_pose_world[ins_ids, 1, 0] = self.hole_pos[ins_ids, 0]
            self._prelude_pose_world[ins_ids, 1, 1] = self.hole_pos[ins_ids, 1]
            self._prelude_pose_world[ins_ids, 1, 2] = lift_z
            self._prelude_pose_world[ins_ids, 1, 3:7] = pre_insert_quat_world

        self._reset_distractors(env_ids)
        if self._pih_object_start_rel and insertion_mask.any():
            self._apply_object_start_pose(env_ids[insertion_mask])

        if pih_cfg.start_state_file and insertion_mask.any():
            self._apply_start_states(env_ids[insertion_mask])

        if getattr(self, "unlock_angle", None) is not None:
            self.unlock_prev_ep_max[env_ids] = self.unlock_max_angle[env_ids]  # kept across the auto-reset
            self.unlock_max_angle[env_ids] = 0.0
            self.unlock_reached[env_ids] = False
        if getattr(self, "turn_mode", False):
            self.turn_prev_ep_max[env_ids] = self.turn_max_angle[env_ids]
            self.turn_prev_ep_at_stop_steps[env_ids] = self.turn_at_stop_steps[env_ids]
            rng_off = self.cfg.precise_assembly.turn_start_offset_range_deg
            if rng_off:
                lo, hi = (math.radians(float(v)) for v in rng_off)
                self.turn_offset[env_ids] = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
            self.turn_angle[env_ids] = self.turn_offset[env_ids]
            if getattr(self, "turn_thread", False):
                # lands on the lip after u_start / L turns, locks play / L turns later (the per-episode stop)
                two_pi = 2.0 * math.pi
                if not hasattr(self, 'turn_prev_ep_lock_angle'):
                    self.turn_prev_ep_lock_angle = torch.zeros_like(self.unlock_angle)
                self.turn_prev_ep_lock_angle[env_ids] = self.unlock_angle[env_ids]   # absolute, like turn_prev_ep_max
                self.turn_seat_angle[env_ids] = (self.turn_offset[env_ids]
                                                 + two_pi * self.turn_u_start[env_ids] / self.turn_lead[env_ids])
                self.unlock_angle[env_ids] = (self.turn_seat_angle[env_ids]
                                              + two_pi * self.turn_play[env_ids] / self.turn_lead[env_ids])
                self.turn_prev_ep_tight[env_ids] = self.turn_tight[env_ids]
                self.turn_prev_ep_lock_torque_max[env_ids] = self.turn_lock_torque_max[env_ids]
                self.turn_tight[env_ids] = False
                self.turn_tight_count[env_ids] = 0
                self.turn_lock_torque[env_ids] = 0.0
                self.turn_lock_torque_max[env_ids] = 0.0
                if getattr(self, "_turn_tau_hist", None) is not None:
                    if not hasattr(self, "turn_prev_ep_hold_max"):
                        self.turn_prev_ep_hold_max = torch.zeros_like(self.turn_lock_torque_hold_max)
                    self.turn_prev_ep_hold_max[env_ids] = self.turn_lock_torque_hold_max[env_ids]
                    self._turn_tau_hist[:, env_ids] = 0.0
                    self.turn_lock_torque_hold_max[env_ids] = 0.0
                self.turn_tau_acc[env_ids] = 0.0
                self.turn_u[env_ids] = self.turn_u_start[env_ids]
                if getattr(self, "_turn_bristle_z", None) is not None:
                    self._turn_bristle_z[env_ids] = 0.0
                self._turn_pending_unlock = getattr(self, '_turn_pending_unlock', torch.zeros_like(self.turn_tight))
                self._turn_pending_unlock[env_ids] |= self.turn_lock_flag[env_ids]
            self._turn_prev_wrapped[env_ids] = 0.0
            self.turn_max_angle[env_ids] = self.turn_offset[env_ids]
            self.turn_strip_count[env_ids] = 0
            self.turn_stripped[env_ids] = False
            if not hasattr(self, "turn_prev_ep_toolless_deg"):
                self.turn_prev_ep_toolless_deg = torch.zeros_like(self.turn_toolless_deg)
            self.turn_prev_ep_toolless_deg[env_ids] = self.turn_toolless_deg[env_ids]
            self.turn_toolless_deg[env_ids] = 0.0
            self.turn_toolless[env_ids] = False
            if getattr(self, "_tl_seg_start", None) is not None:
                self._tl_seg_start[env_ids] = self.turn_angle[env_ids]
                self._tl_out[env_ids] = False
                self._tl_committed[env_ids] = 0.0
            self._turn_prev_step_angle[env_ids] = self.turn_angle[env_ids]
            self.turn_released_early[env_ids] = False
            self._turn_early_clear[env_ids] = 0
            self.turn_at_stop_steps[env_ids] = 0
        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids, is_first_goal=True)

    def _pose_rel_hole_to_world(self, env_ids: torch.Tensor, pose7) -> torch.Tensor:
        """(n, 7) world pose (pos + quat wxyz) of a receptive-frame pose7 (xyz + quat xyzw) for env_ids."""
        n = env_ids.numel()
        p = torch.tensor(pose7[:3], device=self.device, dtype=torch.float32).expand(n, 3)
        q = _xyzw_to_wxyz(torch.tensor([pose7[3:7]], device=self.device, dtype=torch.float32)).expand(n, 4)
        hq = self.hole_quat_wxyz[env_ids]
        pos = self.hole_pos[env_ids] + quat_apply(hq, p) + self.scene.env_origins[env_ids]
        return torch.cat([pos, quat_mul(hq, q)], dim=-1)

    def _reset_distractors(self, env_ids: torch.Tensor) -> None:
        """Place every distractor at its receptive-frame pose; remember it to measure disturbance."""
        dists = getattr(self, "distractors", ())
        if not dists:
            return
        if not hasattr(self, "_distractor_reset_pos"):
            self._distractor_reset_pos = torch.zeros(len(dists), self.num_envs, 3, device=self.device)
            self.distractor_disp = torch.zeros(self.num_envs, device=self.device)
            self.distractor_disp_ep_max = torch.zeros(self.num_envs, device=self.device)
            self.distractor_moved_ep_max = torch.zeros(self.num_envs, device=self.device)
            self.distractor_rise = torch.zeros(self.num_envs, device=self.device)
        thread_k = None
        if getattr(self, "turn_thread", False) and not self._turn_thread_body_is_object():
            thread_k = int(self._pih_joints[int(self.cfg.precise_assembly.unlock_joint)].body[10:])
        for k, (body, pose) in enumerate(zip(dists, self._pih_distractor_pose_rel)):
            w = self._pose_rel_hole_to_world(env_ids, pose)
            if k == thread_k:
                w = self._turn_thread_lift(env_ids, w)   # jointed screw: starts up its thread
            body.write_root_pose_to_sim(w, env_ids=env_ids)
            body.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids)
            self._distractor_reset_pos[k, env_ids] = w[:, :3]
        self.distractor_disp[env_ids] = 0.0
        self.distractor_disp_ep_max[env_ids] = 0.0
        self.distractor_moved_ep_max[env_ids] = 0.0
        self.distractor_rise[env_ids] = 0.0

    def _distractor_curriculum(self) -> tuple[float, float]:
        """(disturb tolerance, penalty multiplier in [0, 1]) at the current env step."""
        c = self.cfg.precise_assembly
        if c.distractor_curriculum_steps <= 0:
            return c.distractor_disturb_tol, 1.0
        a = min(1.0, float(self.common_step_counter + c.distractor_curriculum_offset_steps)
                / float(c.distractor_curriculum_steps))
        return c.distractor_disturb_tol_start + a * (c.distractor_disturb_tol - c.distractor_disturb_tol_start), a

    def _update_distractor_disp(self) -> None:
        dists = getattr(self, "distractors", ())
        if not dists or not hasattr(self, "_distractor_reset_pos"):
            return
        pos = torch.stack([b.data.root_pos_w for b in dists])  # (K, N, 3)
        moved = (pos - self._distractor_reset_pos).norm(dim=-1).max(dim=0).values
        self.distractor_rise = (pos[..., 2] - self._distractor_reset_pos[..., 2]).max(dim=0).values
        self.distractor_moved_ep_max = torch.maximum(self.distractor_moved_ep_max, moved)
        box = self.cfg.precise_assembly.distractor_box
        if box:
            K = pos.shape[0]
            rel = pos - (self.hole_pos + self.scene.env_origins).unsqueeze(0)
            rel = quat_apply_inverse(self.hole_quat_wxyz.unsqueeze(0).expand(K, -1, -1).reshape(-1, 4),
                                     rel.reshape(-1, 3)).reshape(K, -1, 3)
            lo = torch.tensor([box[0], box[2], -1e9], device=self.device)
            hi = torch.tensor([box[1], box[3], box[4]], device=self.device)
            out = (torch.clamp(lo - rel, min=0.0) + torch.clamp(rel - hi, min=0.0)).norm(dim=-1)
            self.distractor_disp = out.max(dim=0).values
        else:
            self.distractor_disp = moved
        self.distractor_disp_ep_max = torch.maximum(self.distractor_disp_ep_max, self.distractor_disp)

    def _joint_bodies(self):
        return [self.object if j.body == "object" else self.distractors[int(j.body[10:])] for j in self._pih_joints]

    def _joint_reset_pose_rel(self, j) -> tuple:
        return self._pih_object_start_rel if j.body == "object" else self._pih_distractor_pose_rel[int(j.body[10:])]

    def _update_joint_angles(self) -> None:
        """Signed angle of every jointed body about its axis since reset (rad), self.cb_joint_angle (J, N)."""
        if not self._pih_joints:
            return
        hq = self.hole_quat_wxyz
        angles = []
        for j, body in zip(self._pih_joints, self._joint_bodies()):
            q0 = _xyzw_to_wxyz(torch.tensor([self._joint_reset_pose_rel(j)[3:7]], device=self.device,
                                            dtype=torch.float32)).expand(self.num_envs, 4)
            q_rel = quat_mul(quat_conjugate(q0), quat_mul(quat_conjugate(hq), body.data.root_quat_w))  # body frame
            axis_b = quat_apply_inverse(q0, torch.tensor(j.axis, device=self.device,
                                                         dtype=torch.float32).expand(self.num_envs, 3))
            s = (q_rel[:, 1:] * axis_b).sum(-1)
            angles.append(2.0 * torch.atan2(s, q_rel[:, 0]))
        a = torch.stack(angles)
        self.cb_joint_angle = torch.atan2(torch.sin(a), torch.cos(a))

    def _apply_joint_detents(self) -> None:
        if not any(j.detent_count > 0 and j.detent_torque != 0.0 for j in self._pih_joints):
            return
        self._update_joint_angles()
        for k, (j, body) in enumerate(zip(self._pih_joints, self._joint_bodies())):
            if j.detent_count <= 0 or j.detent_torque == 0.0:
                continue
            tau = -j.detent_torque * torch.sin(j.detent_count * self.cb_joint_angle[k])
            axis_w = quat_apply(self.hole_quat_wxyz, torch.tensor(j.axis, device=self.device,
                                                                  dtype=torch.float32).expand(self.num_envs, 3))
            torque = (tau.unsqueeze(-1) * axis_w).unsqueeze(1)
            body.set_external_force_and_torque(torch.zeros_like(torque), torque, is_global=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        self._apply_joint_detents()

    def _apply_action(self) -> None:
        super()._apply_action()
        self._object_start_hold()
        if getattr(self, "turn_mode", False):
            self._turn_substep()

    def _object_start_hold(self) -> None:
        """A7: object_start_hold_steps N > 0 = for the first N policy steps of every episode the object is pinned at its
        start pose (pose write + zero velocity every physics substep), like a tool standing in a holder, so the hand can
        grasp it before physics takes over (e.g. an Allen key standing in the screw's recess, which tips over alone)."""
        N = int(getattr(self.cfg.precise_assembly, "object_start_hold_steps", 0))
        if N <= 0 or getattr(self, "_object_start_pose_w", None) is None:
            return
        if not hasattr(self, "_hold_substeps"):
            self._hold_substeps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        held = self._hold_substeps < N * int(self.cfg.decimation)
        self._hold_substeps += 1
        ids = held.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        self.object.write_root_pose_to_sim(self._object_start_pose_w[ids], env_ids=ids)
        self.object.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=self.device), env_ids=ids)

    def _turn_axis_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World axis direction and a world point on the turn joint's axis, (N, 3) each."""
        j = self._pih_joints[int(self.cfg.precise_assembly.unlock_joint)]
        hq = self.hole_quat_wxyz
        axis_w = quat_apply(hq, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(self.num_envs, 3))
        anchor_w = self.hole_pos + self.scene.env_origins + quat_apply(
            hq, torch.tensor(j.anchor, device=self.device, dtype=torch.float32).expand(self.num_envs, 3))
        return axis_w, anchor_w

    def _turn_substep(self) -> None:
        """Every physics substep: accumulate the unwrapped joint angle and enforce the hidden stop (+ lower bound)."""
        self._update_joint_angles()
        k = int(self.cfg.precise_assembly.unlock_joint)
        a = self.cb_joint_angle[k]
        d = torch.atan2(torch.sin(a - self._turn_prev_wrapped), torch.cos(a - self._turn_prev_wrapped))
        self.turn_angle += d
        self._turn_prev_wrapped = a
        lower = math.radians(float(self.cfg.precise_assembly.turn_lower_deg))
        over = self.turn_angle - self.unlock_angle
        pc = self.cfg.precise_assembly
        bfrac = None
        if pc.turn_clutch and float(pc.turn_clutch_coast_damp) > 0.0:
            ids_c = (over > 0).nonzero(as_tuple=False).squeeze(-1)
            if ids_c.numel() > 0:
                body_c = self._joint_bodies()[k]
                ax_c = self._turn_axis_w()[0][ids_c]
                vel_c = body_c.data.root_vel_w[ids_c].clone()
                w_c = (vel_c[:, 3:6] * ax_c).sum(-1).clamp_min(0.0)
                vel_c[:, 3:6] -= (float(pc.turn_clutch_coast_damp) * w_c).unsqueeze(-1) * ax_c
                body_c.write_root_velocity_to_sim(vel_c, env_ids=ids_c)
        if pc.turn_clutch:
            # compliant stop: undo a fraction b of this substep's new over-turn (b = 1: the hard stop's correction)
            b = self._turn_clutch_brake()
            inc = torch.clamp(over, min=0.0) - torch.clamp(over - d, min=0.0)
            over = b * inc if b > 0.0 else torch.zeros_like(over)
            bfrac = b
        elif not pc.turn_stop_physical:
            over = torch.zeros_like(over)
        under = lower + self.turn_offset - self.turn_angle
        corr = torch.where(over > 0, -over, torch.where(under > 0, under, torch.zeros_like(over)))
        ids = (corr != 0).nonzero(as_tuple=False).squeeze(-1)
        thread = getattr(self, "turn_thread", False)
        if ids.numel() == 0 and not thread:
            return
        body = self._joint_bodies()[k]
        axis_w, anchor_w = self._turn_axis_w()
        if thread:
            self._turn_thread_substep(body, axis_w, anchor_w, corr, a)
            return
        ax, an, c = axis_w[ids], anchor_w[ids], corr[ids]
        rot = quat_from_angle_axis(c, ax)
        pos = body.data.root_pos_w[ids]
        quat = body.data.root_quat_w[ids]
        new_pos = an + quat_apply(rot, pos - an)
        new_quat = quat_mul(rot, quat)
        vel = body.data.root_vel_w[ids].clone()
        w_ax = (vel[:, 3:6] * ax).sum(-1)
        bad = ((c < 0) & (w_ax > 0)) | ((c > 0) & (w_ax < 0))   # spinning further past the bound
        keep = torch.ones_like(c) if bfrac is None else torch.where(c < 0, torch.full_like(c, bfrac), torch.ones_like(c))
        vel[:, 3:6] -= (w_ax * bad.float() * keep).unsqueeze(-1) * ax
        vel[:, 0:3] = torch.where(bad.unsqueeze(-1), torch.zeros_like(vel[:, 0:3]), vel[:, 0:3])
        body.write_root_pose_to_sim(torch.cat([new_pos, new_quat], -1), env_ids=ids)
        body.write_root_velocity_to_sim(vel, env_ids=ids)
        self.turn_angle[ids] += c
        self._turn_prev_wrapped[ids] = torch.atan2(torch.sin(a[ids] + c), torch.cos(a[ids] + c))

    def _turn_axis_inertia(self, body, axis_w) -> torch.Tensor:
        """Cap moment of inertia about the turn axis (kg m^2, (N,)), from the PhysX body inertia (body frame)."""
        if self._turn_I_axis is None:
            I = body.root_physx_view.get_inertias().to(self.device).reshape(self.num_envs, 3, 3)
            a_b = quat_apply_inverse(body.data.root_quat_w, axis_w)
            self._turn_I_axis = torch.einsum("ni,nij,nj->n", a_b, I, a_b).clamp_min(1e-9)
        return self._turn_I_axis

    def _turn_update_lock_inertia(self) -> None:
        """turn_lock_inertia: cap inside the lock zone -> raised moment of inertia about its axis (see the cfg)."""
        I_lock = float(self.cfg.precise_assembly.turn_lock_inertia)
        if I_lock <= 0.0:
            return
        zone = self.turn_angle >= self.unlock_angle - math.radians(float(self.cfg.precise_assembly.turn_lock_zone_deg))
        pend = getattr(self, "_turn_pending_unlock", None)
        changed = zone != self.turn_lock_flag
        if pend is not None:
            changed |= pend
            pend.zero_()
        if not bool(changed.any()):
            return
        view = self._joint_bodies()[int(self.cfg.precise_assembly.unlock_joint)].root_physx_view
        inert = view.get_inertias().clone()
        if self._turn_I_orig is None:
            self._turn_I_orig = inert.clone()
        ids = changed.nonzero(as_tuple=False).squeeze(-1).cpu()
        z = zone.cpu()
        inert[ids] = self._turn_I_orig[ids]
        lock_ids = ids[z[ids]]
        if lock_ids.numel() > 0:
            # body z = the bottle axis (the cap is COM-centred, fixture-aligned at its reset pose): Izz is entry 8
            inert[lock_ids, 8] = I_lock
        view.set_inertias(inert, ids.to(torch.int32) if ids.dtype != torch.int32 else ids)
        if lock_ids.numel() > 0 and float(getattr(self.cfg.precise_assembly, "turn_lock_stiffness", 0.0)) > 0.0:
            # conserve the axial angular momentum across the switch (I_orig w -> I_lock w'): keeping w would multiply
            # the cap's spin energy by I_lock / I_orig (~2e4) and slam an elastic lock into its backstop
            body = self._joint_bodies()[int(self.cfg.precise_assembly.unlock_joint)]
            li = lock_ids.to(self.device)
            axis_w, _ = self._turn_axis_w()
            ax = axis_w[li]
            vel = body.data.root_vel_w[li].clone()
            w = (vel[:, 3:6] * ax).sum(-1)
            ratio = self._turn_I_orig[lock_ids, 8].to(self.device) / I_lock
            vel[:, 3:6] -= ((1.0 - ratio) * w).unsqueeze(-1) * ax
            body.write_root_velocity_to_sim(vel, env_ids=li)
        self.turn_lock_flag = zone

    def _turn_thread_substep(self, body, axis_w, anchor_w, corr, a) -> None:
        """turn_thread, every physics substep: (1) the lock (hard stop at unlock_angle, as turn_mode) plus the torque
        the hand pushes into it, I_axis x (spin the stop removes) / dt; (2) the helix band: height above the seat
        u in [max(0, r L / 2 pi), r L / 2 pi + play], r = seat angle - angle; outside it the cap is moved back along
        the axis and its axial velocity into the band's wall removed (a self-locking thread: no turning from axial
        loads). The lip seat (u >= 0) is also the cylindrical joint's PhysX limit."""
        n = self.num_envs
        dt = float(self.physics_dt)
        pos = body.data.root_pos_w.clone()
        quat = body.data.root_quat_w.clone()
        vel = body.data.root_vel_w.clone()
        pc = self.cfg.precise_assembly
        K_lock = float(getattr(pc, "turn_lock_stiffness", 0.0))
        tau_c0 = float(getattr(pc, "turn_thread_friction", 0.0))
        tau_fric = torch.zeros(n, device=self.device)
        if tau_c0 > 0.0:
            # A7 thread friction (official _allen_key: running friction TAU_RUN0 + seat preload = frac x the torque
            # measured into the seat, released after backing off turn_preload_free_deg), as a smooth Coulomb torque
            # -tau_c tanh(w / w0) applied during the step (a velocity-level correction after the step cannot stop a
            # light screw moving within it). Explicitly stable while tau_c dt / (w0 I) < 2: give the screw enough axial
            # inertia. Viscous drag goes on the joint's (implicit) PhysX drive: JointSpec.damping.
            tau_c = torch.full((n,), tau_c0, device=self.device)
            frac = float(getattr(pc, "turn_seat_preload_frac", 0.0))
            if frac > 0.0:
                free = self.turn_angle < self.unlock_angle - math.radians(float(pc.turn_preload_free_deg))
                hm = getattr(self, "turn_lock_torque_hold_max", None)
                if hm is not None:
                    hm.copy_(torch.where(free, torch.zeros_like(hm), hm))
                    # only inside the lock zone, where the raised lock inertia keeps the stiff tanh stable (the
                    # official releases it after backing off turn_preload_free_deg; here leaving the 0.5 deg zone does)
                    tau_c = tau_c + frac * hm * self.turn_lock_flag.float()
            w_f = (vel[:, 3:6] * axis_w).sum(-1)
            kb = float(getattr(pc, "turn_thread_bristle_k", 0.0))
            if kb > 0.0:   # A9 (f652587): elasto-plastic stiction (official STICTION_K), holds below breakaway, no creep
                if getattr(self, "_turn_bristle_z", None) is None:
                    self._turn_bristle_z = torch.zeros(n, device=self.device)
                zmax = tau_c / kb
                z = torch.maximum(torch.minimum(self._turn_bristle_z + w_f * dt, zmax), -zmax)
                stick = z.abs() < zmax
                self._turn_bristle_z = z
                tau_fric = kb * z + torch.where(stick, float(pc.turn_thread_bristle_d) * w_f, torch.zeros_like(w_f))
            else:
                tau_fric = tau_c * torch.tanh(w_f / float(pc.turn_thread_stiction_w))
        k_ax = float(getattr(pc, "turn_thread_axial_k", 0.0))
        f_ax = torch.zeros(n, device=self.device)
        g_comp = bool(getattr(pc, "turn_thread_gravity_comp", False))
        if g_comp and k_ax <= 0.0:
            # heavy jointed body on the exact pose band: only cancel its weight (the band holds the height)
            if not hasattr(self, "_turn_body_mass"):
                self._turn_body_mass = body.root_physx_view.get_masses().to(self.device).reshape(n, -1)[:, 0]
            f_ax = self._turn_body_mass * 9.81
        if k_ax > 0.0:
            # A7 implicit-enough helix: an axial spring + damper pulls the body to its helix height (band clamp of the
            # current height), + gravity compensation (the body's mass is raised so the explicit spring is stiff and
            # stable; official: helix potential 1/2 k (z + L q)^2). The pose band below only acts as a backstop.
            if not hasattr(self, "_turn_seat_pos_w"):
                ids_all = torch.arange(n, device=self.device)
                j_t = self._pih_joints[int(pc.unlock_joint)]
                self._turn_seat_pos_w = (self._pose_rel_hole_to_world(ids_all, self._joint_reset_pose_rel(j_t))[:, 0:3]
                                         + self.scene.env_origins)
            up_s = -axis_w
            u_s = ((pos - self._turn_seat_pos_w) * up_s).sum(-1)
            r_s = self.turn_seat_angle - self.turn_angle
            hel = r_s * self.turn_lead / (2.0 * math.pi)
            tgt = torch.minimum(torch.maximum(u_s, torch.clamp(hel, min=0.0)), torch.clamp(hel + self.turn_play, min=0.0))
            v_s = (vel[:, 0:3] * up_s).sum(-1)
            if not hasattr(self, "_turn_body_mass"):
                self._turn_body_mass = body.root_physx_view.get_masses().to(self.device).reshape(n, -1)[:, 0]
            f_ax = k_ax * (tgt - u_s) - float(pc.turn_thread_axial_d) * v_s + self._turn_body_mass * 9.81
        elastic = None
        if K_lock > 0.0:
            # elastic lock where the raised inertia is on: no pose correction up to the backstop, spring + damper instead
            over_l = torch.clamp(-corr, min=0.0)                       # angle past the lock (corr < 0 there)
            back = math.radians(float(pc.turn_lock_backstop_deg))
            elastic = self.turn_lock_flag & (over_l > 0)
            corr = torch.where(elastic, -torch.clamp(over_l - back, min=0.0), corr)
            w_now = (vel[:, 3:6] * axis_w).sum(-1)
            over_e = torch.clamp(over_l, max=back)
            tau_l = torch.where(elastic, K_lock * over_e + float(pc.turn_lock_damping) * w_now.clamp_min(0.0),
                                torch.zeros_like(over_l))
            self._turn_dbg_el = (elastic.clone(), tau_l.clone(), over_l.clone(), w_now.clone())
        if K_lock > 0.0 or tau_c0 > 0.0 or k_ax > 0.0 or g_comp:
            tau_tot = (tau_l if K_lock > 0.0 else torch.zeros(n, device=self.device)) + tau_fric   # both oppose +spin
            zero = torch.zeros(n, 1, 3, device=self.device)
            # world -> body frame with the CURRENT orientation (IsaacLab's permanent wrench composer caches the link
            # orientation from its first call for is_global=True, so global wrenches on a turning body go wrong)
            q_b = body.data.root_quat_w.unsqueeze(1)
            if body is self.object:
                f = getattr(self, "_object_forces", None); t = getattr(self, "_object_torques", None)
                f_w = zero if f is None else f
                t_w = (zero if t is None else t) - (tau_tot.unsqueeze(-1) * axis_w).unsqueeze(1)
                self.object.set_external_force_and_torque(quat_apply_inverse(q_b.expand(-1, f_w.shape[1], 4), f_w),
                                                          quat_apply_inverse(q_b.expand(-1, t_w.shape[1], 4), t_w),
                                                          is_global=False)
            else:
                # _turn_ext_force / _turn_ext_torque: an extra wrench on the jointed body (no-hand probes), kept
                f_x = getattr(self, "_turn_ext_force", None); t_x = getattr(self, "_turn_ext_torque", None)
                f_w = (zero if f_x is None else f_x) + (f_ax.unsqueeze(-1) * -axis_w).unsqueeze(1)
                t_w = (zero if t_x is None else t_x) - (tau_tot.unsqueeze(-1) * axis_w).unsqueeze(1)
                body.set_external_force_and_torque(quat_apply_inverse(q_b.expand(-1, f_w.shape[1], 4), f_w),
                                                   quat_apply_inverse(q_b.expand(-1, t_w.shape[1], 4), t_w),
                                                   is_global=False)
        touched = corr != 0
        # (1) angle: stop / lower bound
        c = corr
        rot = quat_from_angle_axis(c, axis_w)
        pos = torch.where(touched.unsqueeze(-1), anchor_w + quat_apply(rot, pos - anchor_w), pos)
        quat = torch.where(touched.unsqueeze(-1), quat_mul(rot, quat), quat)
        w_ax = (vel[:, 3:6] * axis_w).sum(-1)
        bad = touched & (((c < 0) & (w_ax > 0)) | ((c > 0) & (w_ax < 0)))
        at_lock = bad & (c < 0)
        self._turn_dbg = (w_ax.clone(), c.clone(), bad.clone())
        I_cur = self._turn_axis_inertia(body, axis_w)
        if float(self.cfg.precise_assembly.turn_lock_inertia) > 0.0:
            I_cur = torch.where(self.turn_lock_flag, torch.full_like(I_cur, float(self.cfg.precise_assembly.turn_lock_inertia)), I_cur)
        # only substeps where the cap already sat on the lock the substep before: there the spin removed is what the
        # hand's torque built up in one substep (I w / dt); on the first contact substep it is the cap's arrival
        # momentum (an impact, x I_lock / I_orig after the zone's inertia switch), not a held torque
        prev = getattr(self, "_turn_at_lock_prev", None)
        if prev is None:
            prev = torch.zeros_like(at_lock)
        tau = torch.where(at_lock & prev, I_cur * w_ax / dt, torch.zeros_like(w_ax))
        self._turn_at_lock_prev = at_lock.clone()
        if elastic is not None:
            # elastic lock: the torque pushed into it is the spring + damper torque (physical); pose-stop substeps
            # (backstop / before the inertia switch) are not counted
            tau = torch.where(elastic, tau_l, torch.zeros_like(tau))
        self.turn_tau_acc += tau
        self._turn_tau_n += 1
        vel[:, 3:6] -= (w_ax * bad.float()).unsqueeze(-1) * axis_w
        self.turn_angle += c
        self._turn_prev_wrapped = torch.where(touched, torch.atan2(torch.sin(a + c), torch.cos(a + c)),
                                              self._turn_prev_wrapped)
        # (2) height: helix band
        if not hasattr(self, "_turn_seat_pos_w"):
            ids_all = torch.arange(n, device=self.device)
            j_t = self._pih_joints[int(pc.unlock_joint)]
            self._turn_seat_pos_w = (self._pose_rel_hole_to_world(ids_all, self._joint_reset_pose_rel(j_t))[:, 0:3]
                                     + self.scene.env_origins)
        up = -axis_w
        u = ((pos - self._turn_seat_pos_w) * up).sum(-1)
        r = self.turn_seat_angle - self.turn_angle
        helix = r * self.turn_lead / (2.0 * math.pi)
        lo = torch.clamp(helix, min=0.0)
        hi = torch.clamp(helix + self.turn_play, min=0.0)
        du = torch.where(u < lo, lo - u, torch.where(u > hi, hi - u, torch.zeros_like(u)))
        tol = float(getattr(pc, "turn_thread_band_tol_m", 0.0))
        if tol > 0.0:
            # A7: write the pose only when the body is more than tol off its helix band (per-substep pose writes of a
            # body in contact with a light tool may disturb that contact); the full error is corrected then
            du = torch.where(du.abs() > tol, du, torch.zeros_like(du))
        if float(getattr(pc, "turn_thread_axial_k", 0.0)) > 0.0:
            bs = float(pc.turn_thread_axial_backstop_m)
            du = torch.where(du.abs() > bs, du - torch.sign(du) * bs, torch.zeros_like(du))
        moved = du != 0
        pos = pos + up * du.unsqueeze(-1)
        v_ax = (vel[:, 0:3] * up).sum(-1)
        into = moved & (((du > 0) & (v_ax < 0)) | ((du < 0) & (v_ax > 0)))
        vel[:, 0:3] -= (v_ax * into.float()).unsqueeze(-1) * up
        self.turn_u = u + du
        ids = (touched | moved).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        body.write_root_pose_to_sim(torch.cat([pos[ids], quat[ids]], -1), env_ids=ids)
        body.write_root_velocity_to_sim(vel[ids], env_ids=ids)

    def _turn_clutch_brake(self) -> float:
        pc = self.cfg.precise_assembly
        b0 = float(pc.turn_clutch_brake)
        if int(pc.turn_clutch_brake_ramp_steps) <= 0:
            return b0
        f = min(1.0, float(self.common_step_counter + int(pc.turn_clutch_brake_offset_steps))
                / float(pc.turn_clutch_brake_ramp_steps))
        return b0 + f * (float(pc.turn_clutch_brake_end) - b0)

    def _tactile_object_pose(self):
        """Object (pos, quat) the fingertip tactile ray-casts against, or None (= the real pose). Turn clutch: the cap
        rotated back by its over-turn past the stop, i.e. the felt surface stays at the stop angle."""
        if not (getattr(self, "turn_mode", False) and self.cfg.precise_assembly.turn_clutch):
            return None
        over = torch.clamp(self.turn_angle - self.unlock_angle, min=0.0)
        axis_w, anchor_w = self._turn_axis_w()
        rot = quat_from_angle_axis(-over, axis_w)
        pos = self.object.data.root_pos_w
        return anchor_w + quat_apply(rot, pos - anchor_w), quat_mul(rot, self.object.data.root_quat_w)

    def _actor_turn_hide(self):
        """For obs_utils.build_observations: (world quat that undoes the cap's turn, axis) or None."""
        if not (getattr(self, "turn_mode", False) and self.cfg.precise_assembly.turn_hide_from_actor):
            return None
        axis_w, _ = self._turn_axis_w()
        return quat_from_angle_axis(-(self.turn_angle - self.turn_offset), axis_w), axis_w

    def _turn_goal_values(self) -> None:
        """Turn mode, after compute_intermediate_values: goals / keypoint distance from the unwrapped angle."""
        from isaacsimenvs.tasks.play.utils.reward_utils import update_near_goal_steps
        pih_cfg = self.cfg.precise_assembly
        n_pre = int(getattr(pih_cfg, "turn_prelude_goals", 0))
        if n_pre > 0:
            # the first n_pre goals are ordinary object-pose goals (e.g. tool above the screw, tool engaged); keep the
            # values compute_intermediate_values set for envs still in them
            pre = self._successes < n_pre
            saved = (self._keypoints_max_dist.clone(), self._closest_keypoint_max_dist.clone(),
                     self._near_goal.clone(), self._near_goal_steps.clone(), self._is_success.clone())
        over_prev = torch.clamp(self.turn_max_angle - self.unlock_angle, min=0.0)
        self.turn_max_angle = torch.maximum(self.turn_max_angle, self.turn_angle)
        self._turn_over_inc = torch.clamp(self.turn_max_angle - self.unlock_angle, min=0.0) - over_prev
        at_last = self._successes >= self.env_max_goals - 1
        target = torch.where(at_last, self.unlock_angle,
                             self.turn_offset + math.radians(float(pih_cfg.turn_first_goal_deg)))
        rem = torch.clamp(target - self.turn_angle, min=0.0)
        self._keypoints_max_dist = rem * float(pih_cfg.turn_progress_m_per_rad)
        # goal trackers cleared this step (-1 sentinel, replaced by the real keypoint distance in
        # compute_intermediate_values): start them from the turn distance instead
        fresh = self._turn_fresh_closest
        self._closest_keypoint_max_dist = torch.where(fresh, self._keypoints_max_dist, self._closest_keypoint_max_dist)
        near = rem <= math.radians(float(pih_cfg.unlock_tol_deg))
        if getattr(self, "turn_thread", False):
            # tight = the hand has pushed >= turn_tight_torque into the lock for turn_tight_steps consecutive steps;
            # turning the cap back > 1 deg off the lock loosens it (the preload is gone)
            self._turn_torque_step()
            near = torch.where(at_last, near & self.turn_tight, near)
            self._turn_update_lock_inertia()
            self.extras["turn_tight_frac"] = self.turn_tight.float().mean()
            self.extras["turn_lock_torque_mean"] = self.turn_lock_torque.mean()
            self.extras["turn_u_mm_mean"] = self.turn_u.mean() * 1e3
        self._near_goal = near
        self._near_goal_steps = update_near_goal_steps(
            near_goal=near, near_goal_steps=self._turn_prev_near_steps,
            force_consecutive=self.cfg.termination.force_consecutive_near_goal_steps)
        self._is_success = self._near_goal_steps >= self.cfg.termination.success_steps
        if n_pre > 0:
            self._keypoints_max_dist = torch.where(pre, saved[0], self._keypoints_max_dist)
            self._closest_keypoint_max_dist = torch.where(pre, saved[1], self._closest_keypoint_max_dist)
            self._near_goal = torch.where(pre, saved[2], self._near_goal)
            self._near_goal_steps = torch.where(pre, saved[3], self._near_goal_steps)
            self._is_success = torch.where(pre, saved[4], self._is_success)
        # over-tightening: at the stop with a fingertip pad in contact (real taxels, also under policy_override)
        at_stop = self.turn_angle >= self.unlock_angle - math.radians(float(pih_cfg.turn_stop_band_deg))
        self.turn_at_stop_steps += at_stop.long()
        near_d = float(pih_cfg.turn_strip_near_dist)
        if near_d > 0.0:
            contact = self._curr_fingertip_distances.min(dim=-1).values < near_d
        else:
            tac = getattr(self, "_last_tactile", None)
            contact = torch.zeros_like(at_stop) if tac is None else tac.reshape(tac.shape[0], -1).sum(-1) > 0
        self.turn_strip_count += (at_stop & contact).long()
        self._turn_just_stripped = torch.zeros_like(at_stop)
        self._turn_toolless_check()
        over_deg = float(pih_cfg.turn_strip_over_deg)
        strip_steps = int(pih_cfg.turn_strip_steps)
        if int(pih_cfg.turn_strip_over_ramp_steps) > 0:
            f = min(1.0, float(self.common_step_counter + int(pih_cfg.turn_strip_over_ramp_offset_steps))
                    / float(pih_cfg.turn_strip_over_ramp_steps))
            over_deg = float(pih_cfg.turn_strip_over_ramp_start_deg) + f * (over_deg - float(pih_cfg.turn_strip_over_ramp_start_deg))
            if int(pih_cfg.turn_strip_steps_ramp_start) > 0:
                strip_steps = int(round(int(pih_cfg.turn_strip_steps_ramp_start)
                                        + f * (strip_steps - int(pih_cfg.turn_strip_steps_ramp_start))))
            self.extras["turn_strip_over_deg_cur"] = over_deg
            self.extras["turn_strip_steps_cur"] = float(strip_steps)
        strip_tau = float(pih_cfg.turn_strip_torque) if getattr(self, "turn_thread", False) else 0.0
        if getattr(self, "turn_thread", False):
            strip_tau = max(strip_tau, float(getattr(pih_cfg, "turn_break_torque", 0.0)))
        if strip_steps > 0 or over_deg > 0.0 or strip_tau > 0.0:
            new = torch.zeros_like(at_stop)
            if strip_steps > 0:
                new |= self.turn_strip_count > strip_steps
            if over_deg > 0.0:
                new |= self.turn_angle > self.unlock_angle + math.radians(over_deg)
            if getattr(self, "turn_thread", False) and float(pih_cfg.turn_strip_torque) > 0.0:
                new |= self.turn_lock_torque > float(pih_cfg.turn_strip_torque)   # our over-torque rule (not official)
            brk = float(getattr(pih_cfg, "turn_break_torque", 0.0))
            if getattr(self, "turn_thread", False) and brk > 0.0 and getattr(self, "_turn_hold_now", None) is not None:
                # A7 'break' variant: the SUSTAINED seat torque (min over turn_tight_steps) above turn_break_torque
                # breaks the part (stripped); a slam spike does not count
                new |= self._turn_hold_now > brk
            new &= ~self.turn_stripped
            self._turn_just_stripped = new
            self.turn_stripped |= new
        self.extras["turn_angle_deg_mean"] = torch.rad2deg(self.turn_angle).mean()
        self.extras["turn_at_stop_frac"] = at_stop.float().mean()
        if pih_cfg.turn_clutch:
            self.extras["turn_clutch_brake_cur"] = self._turn_clutch_brake()
        self.extras["turn_over_deg_mean"] = torch.rad2deg(torch.clamp(self.turn_angle - self.unlock_angle, min=0.0)).mean()
        self.extras["turn_contact_frac"] = contact.float().mean()

    def _turn_thread_lift(self, env_ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """turn_thread start: the jointed body on the thread, engaged length above the seat less the drop U(0, play)
        before the thread bites. w = its seated world pose (n, 7); returns the lifted pose."""
        pc = self.cfg.precise_assembly
        drop = torch.rand(env_ids.numel(), device=self.device) * self.turn_play[env_ids]
        self.turn_u_start[env_ids] = float(pc.turn_thread_engaged_mm) * 1e-3 - drop
        axis_w, _ = self._turn_axis_w()
        w = w.clone()
        w[:, 0:3] -= axis_w[env_ids] * self.turn_u_start[env_ids].unsqueeze(-1)   # axis points into the bottle
        return w

    def _turn_thread_body_is_object(self) -> bool:
        return self._pih_joints[int(self.cfg.precise_assembly.unlock_joint)].body == "object"

    def _turn_torque_step(self) -> None:
        """turn_thread, once per policy step: the step's mean torque into the lock (turn_lock_torque), its running max,
        the sustained torque = min over the last turn_tight_steps steps (turn_lock_torque_hold; A7: the preload uses its
        max, so a slam into the seat does not count as torque held), and tight = sustained >= turn_tight_torque."""
        pc = self.cfg.precise_assembly
        self.turn_lock_torque = self.turn_tau_acc / float(max(1, self._turn_tau_n))
        self.turn_tau_acc.zero_()
        self._turn_tau_n = 0
        self.turn_lock_torque_max = torch.maximum(self.turn_lock_torque_max, self.turn_lock_torque)
        N = max(1, int(pc.turn_tight_steps))
        if getattr(self, "_turn_tau_hist", None) is None or self._turn_tau_hist.shape[0] != N:
            self._turn_tau_hist = torch.zeros(N, self.num_envs, device=self.device)
            self._turn_tau_hist_i = 0
            self.turn_lock_torque_hold_max = torch.zeros(self.num_envs, device=self.device)
        self._turn_tau_hist[self._turn_tau_hist_i % N] = self.turn_lock_torque
        self._turn_tau_hist_i += 1
        hold = self._turn_tau_hist.min(dim=0).values
        self._turn_hold_now = hold
        self.turn_lock_torque_hold_max = torch.maximum(self.turn_lock_torque_hold_max, hold)
        pushing = self.turn_lock_torque >= float(pc.turn_tight_torque)
        self.turn_tight_count = torch.where(pushing, self.turn_tight_count + 1, torch.zeros_like(self.turn_tight_count))
        self.turn_tight |= self.turn_tight_count >= N
        self.turn_tight &= self.turn_angle >= self.unlock_angle - math.radians(1.0)

    def _turn_toolless_check(self) -> None:
        """A7 (official turned_by_the_tool_in_hand): accumulate the jointed body's rotation over policy steps in which no
        tool tip (turn_tool_tips_m, object frame) is in its recess (within turn_tool_radial_m of the axis and below the
        head top + turn_tool_out_m); more than turn_toolless_max_deg = fail `toolless` (-turn_toolless_penalty)."""
        pc = self.cfg.precise_assembly
        tips = getattr(pc, "turn_tool_tips_m", ())
        d = (self.turn_angle - self._turn_prev_step_angle).abs()
        self._turn_prev_step_angle = self.turn_angle.clone()
        self._turn_just_toolless = torch.zeros_like(self.turn_toolless)
        if not tips:
            return
        axis_w, anchor_w = self._turn_axis_w()
        up = -axis_w
        u = getattr(self, "turn_u", torch.zeros_like(self.turn_angle))
        head_top = anchor_w + (u + float(pc.turn_tool_head_top_m)).unsqueeze(-1) * up
        pos, quat = self.object.data.root_pos_w, self.object.data.root_quat_w
        inside = torch.zeros_like(self.turn_toolless)
        for t in tips:
            tl = torch.tensor([float(v) for v in t], device=self.device).expand(self.num_envs, 3)
            tw = pos + quat_apply(quat, tl)
            rel = tw - head_top
            h = (rel * up).sum(-1)
            radial = (rel - h.unsqueeze(-1) * up).norm(dim=-1)
            inside |= (radial < float(pc.turn_tool_radial_m)) & (h < float(pc.turn_tool_out_m))
        # NET rotation over each stretch with no tool tip in the recess (sim jitter averages out; a slow real turn by
        # fingers still accumulates): committed (closed stretches) + |angle - angle at the stretch's start|
        if getattr(self, "_tl_seg_start", None) is None:
            self._tl_seg_start = self.turn_angle.clone()
            self._tl_out = torch.zeros_like(inside)
            self._tl_committed = torch.zeros_like(self.turn_toolless_deg)
        starts = ~inside & ~self._tl_out
        self._tl_seg_start = torch.where(starts, self.turn_angle, self._tl_seg_start)
        cur = torch.rad2deg((self.turn_angle - self._tl_seg_start).abs())
        ends = inside & self._tl_out
        self._tl_committed = self._tl_committed + torch.where(ends, cur, torch.zeros_like(cur))
        self._tl_out = ~inside
        self.turn_toolless_deg = self._tl_committed + torch.where(self._tl_out, cur, torch.zeros_like(cur))
        self.extras["turn_tool_in_recess_frac"] = inside.float().mean()
        lim = float(pc.turn_toolless_max_deg)
        if lim > 0.0:
            new = (self.turn_toolless_deg > lim) & ~self.turn_toolless
            self._turn_just_toolless = new
            self.turn_toolless |= new

    def _apply_object_start_pose(self, env_ids: torch.Tensor) -> None:
        w = self._pose_rel_hole_to_world(env_ids, self._pih_object_start_rel)
        if getattr(self, "turn_thread", False) and self._turn_thread_body_is_object():
            w = self._turn_thread_lift(env_ids, w)
        self.object.write_root_pose_to_sim(w, env_ids=env_ids)
        if int(getattr(self.cfg.precise_assembly, "object_start_hold_steps", 0)) > 0:
            if getattr(self, "_object_start_pose_w", None) is None:
                self._object_start_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
                self._hold_substeps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._object_start_pose_w[env_ids] = w
            self._hold_substeps[env_ids] = 0
        self.object.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids)
        # The table-drop reset set _object_init_z to the drop height (table + 0.25 m above its centre), so "lifted"
        # (z - init_z > 0.10, which gates the keypoint reward) would need ~20 cm above the bin. Measure from here.
        self._object_init_z[env_ids] = w[:, 2] - self.scene.env_origins[env_ids, 2]

    def _apply_start_states(self, env_ids: torch.Tensor) -> None:
        """Overwrite robot + peg state with random entries of the start-state bank."""
        if not hasattr(self, "_start_bank"):
            import numpy as np

            d = np.load(self.cfg.precise_assembly.start_state_file)
            self._start_bank = {k: torch.as_tensor(d[k], device=self.device)
                                for k in ("joint_pos", "joint_vel", "cur_targets", "prev_targets",
                                          "peg_pos_h", "peg_quat_h", "peg_linvel_h", "peg_angvel_h",
                                          "successes", "lifted", "object_init_z")}
            print(f"[PreciseAssemblyEnv] start states: {len(d['successes'])} from "
                  f"{self.cfg.precise_assembly.start_state_file}", flush=True)
            # Robot joints are absolute, so the bank only fits the hole layout it was recorded on.
            self._start_bank_hole_pos = (torch.as_tensor(d["hole_pos"], device=self.device).float()
                                         if "hole_pos" in d.files else None)
            if self._start_bank_hole_pos is None:
                print("[PreciseAssemblyEnv] start-state bank has no hole_pos; can't check the hole "
                      "layout matches (use the layout it was recorded on)", flush=True)
        b = self._start_bank
        n = env_ids.numel()
        if self.cfg.precise_assembly.start_state_sequential:
            k = (int(self.cfg.precise_assembly.start_state_offset) + env_ids.long()) % b["successes"].numel()
        else:
            k = torch.randint(0, b["successes"].numel(), (n,), device=self.device)
        if self._start_bank_hole_pos is not None:
            err = (self._start_bank_hole_pos[k] - self.hole_pos[env_ids]).norm(dim=-1).max().item()
            if err > 0.005:
                raise RuntimeError(f"start-state bank was recorded with the hole {err * 1000:.1f} mm away from "
                                   "this env's hole; use the same layout (e.g. the static-layout overrides)")

        self.robot.write_joint_state_to_sim(b["joint_pos"][k].float(), b["joint_vel"][k].float(), env_ids=env_ids)
        self._cur_targets[env_ids] = b["cur_targets"][k].float()
        self._prev_targets[env_ids] = b["prev_targets"][k].float()

        hole_q = self.hole_quat_wxyz[env_ids]
        pos = self.hole_pos[env_ids] + quat_apply(hole_q, b["peg_pos_h"][k].float())
        quat = quat_mul(hole_q, b["peg_quat_h"][k].float())
        vel = torch.cat([quat_apply(hole_q, b["peg_linvel_h"][k].float()),
                         quat_apply(hole_q, b["peg_angvel_h"][k].float())], dim=-1)
        self.object.write_root_pose_to_sim(torch.cat([pos + self.scene.env_origins[env_ids], quat], -1), env_ids=env_ids)
        self.object.write_root_velocity_to_sim(vel, env_ids=env_ids)

        self._successes[env_ids] = b["successes"][k].long()
        self._start_successes[env_ids] = b["successes"][k].long()
        self._lifted_object[env_ids] = b["lifted"][k].bool()
        self._object_init_z[env_ids] = b["object_init_z"][k].float()

    def _write_goal_pose(self, env_ids: torch.Tensor, is_first_goal: bool = False) -> None:
        n = env_ids.numel()
        env_origins = self.scene.env_origins[env_ids]
        is_random_goal = self.is_random_goal_env[env_ids]

        pos_local = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1).clone()

        insertion_mask = ~is_random_goal
        if insertion_mask.any():
            ins_ids = env_ids[insertion_mask]
            subgoal_idx = (
                self._successes[ins_ids] % self.env_max_goals[ins_ids]
            ).long()
            ins_pos_local = torch.zeros(
                ins_ids.numel(), 3, dtype=torch.float32, device=self.device
            )
            ins_quat = torch.zeros(
                ins_ids.numel(), 4, dtype=torch.float32, device=self.device
            )
            ins_quat[:, 0] = 1.0

            # Prelude path: world-frame poses prebuilt at reset.
            in_prelude = subgoal_idx < self._num_prelude_goals
            if in_prelude.any():
                pre_ids = ins_ids[in_prelude]
                pre_idx = subgoal_idx[in_prelude]
                pre_pose = self._prelude_pose_world[pre_ids, pre_idx]
                ins_pos_local[in_prelude] = pre_pose[:, 0:3]
                ins_quat[in_prelude] = pre_pose[:, 3:7]

            # Tail path: hole-frame poses, transformed by current hole pose.
            in_tail = ~in_prelude
            if in_tail.any():
                tail_ids = ins_ids[in_tail]
                tail_idx = subgoal_idx[in_tail] - self._num_prelude_goals
                hole_q = self.hole_quat_wxyz[tail_ids]
                insert_pos = self._insert_pos_rel[tail_idx]
                insert_q = self._insert_quat_wxyz[tail_idx]
                ins_pos_local[in_tail] = (
                    self.hole_pos[tail_ids] + quat_apply(hole_q, insert_pos)
                )
                ins_quat[in_tail] = quat_mul(hole_q, insert_q)

            pos_local[insertion_mask] = ins_pos_local
            quat[insertion_mask] = ins_quat

        if is_random_goal.any():
            rg_ids = env_ids[is_random_goal]
            reset_cfg = self.cfg.reset
            if reset_cfg.fixed_goal_pose is not None:
                fixed = torch.as_tensor(
                    reset_cfg.fixed_goal_pose,
                    device=self.device,
                    dtype=torch.float32,
                )
                pos_local[is_random_goal] = fixed[:3].unsqueeze(0).expand(
                    rg_ids.numel(), -1
                )
                quat[is_random_goal] = fixed[3:].unsqueeze(0).expand(
                    rg_ids.numel(), -1
                )
            elif is_first_goal or reset_cfg.goal_sampling_type == "absolute":
                rg_pos, rg_quat = sample_absolute_goal_pose(
                    mins=reset_cfg.target_volume_mins,
                    maxs=reset_cfg.target_volume_maxs,
                    scale=reset_cfg.target_volume_region_scale,
                    n_envs=rg_ids.numel(),
                    device=self.device,
                )
                pos_local[is_random_goal] = rg_pos
                quat[is_random_goal] = rg_quat
            elif reset_cfg.goal_sampling_type == "delta":
                rg_origins = self.scene.env_origins[rg_ids]
                prev_pos = self.goal_viz.data.root_pos_w[rg_ids] - rg_origins
                prev_quat = self.goal_viz.data.root_quat_w[rg_ids]
                rg_pos, rg_quat = sample_delta_goal_pose(
                    prev_pos=prev_pos,
                    prev_quat_wxyz=prev_quat,
                    delta_distance=reset_cfg.delta_goal_distance,
                    delta_rotation_degrees=reset_cfg.delta_rotation_degrees,
                    mins=reset_cfg.target_volume_mins,
                    maxs=reset_cfg.target_volume_maxs,
                    scale=reset_cfg.target_volume_region_scale,
                )
                pos_local[is_random_goal] = rg_pos
                quat[is_random_goal] = rg_quat
            else:
                raise ValueError(
                    "cfg.reset.goal_sampling_type must be 'delta' or 'absolute' "
                    f"for PreciseAssembly random-goal envs, got {reset_cfg.goal_sampling_type!r}."
                )

        pose = torch.cat([pos_local + env_origins, quat], dim=-1)
        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

    def _clear_goal_trackers(self, env_ids: torch.Tensor) -> None:
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._near_goal_steps[env_ids] = 0
        if hasattr(self, "_skip_cand"):
            self._skip_cand[env_ids] = -1
            self._skip_steps[env_ids] = 0

    def _keypoint_success_tolerance_m(self) -> torch.Tensor:
        pih_cfg = self.cfg.precise_assembly
        keypoint_scale = self.cfg.reward.keypoint_scale
        fixed_insertion_tol = pih_cfg.insertion_success_tolerance * keypoint_scale
        curriculum_tol = self._current_success_tolerance * keypoint_scale

        tol = torch.full(
            (self.num_envs,),
            float(fixed_insertion_tol),
            dtype=torch.float32,
            device=self.device,
        )
        if float(pih_cfg.random_goal_fraction) > 0.0:
            tol = torch.where(
                self.is_random_goal_env,
                torch.full_like(tol, float(curriculum_tol)),
                tol,
            )

        # In transportPreInsertFinal mode the coarse waypoints (prelude +
        # transport_above) use the curriculum tolerance (starts at
        # success_tolerance, shrinks toward target_success_tolerance), while
        # pre_insert and final keep the tight fixed insertion_success_tolerance.
        # The number of "coarse" stages is the prelude (2) plus the leading
        # transport_above pose (1) in the hole-frame tail = 3.
        if (
            self._num_prelude_goals > 0
            and pih_cfg.goal_mode == "transportPreInsertFinal"
        ):
            num_coarse = self._num_prelude_goals + 1  # +1 = transport_above
            subgoal_idx = (self._successes % self.env_max_goals).long()
            is_coarse = (subgoal_idx < num_coarse) & ~self.is_random_goal_env
            if is_coarse.any():
                tol = torch.where(
                    is_coarse,
                    torch.full_like(tol, float(curriculum_tol)),
                    tol,
                )
        return tol

    def _curriculum_eligible_mask(self) -> torch.Tensor | None:
        pih_cfg = self.cfg.precise_assembly
        if pih_cfg.goal_mode == "transportPreInsertFinal":
            # Drive the curriculum off insertion-only envs in dense-traj mode.
            return ~self.is_random_goal_env
        if float(pih_cfg.random_goal_fraction) <= 0.0:
            return None
        return self.is_random_goal_env

    def _curriculum_success_threshold(self) -> float | None:
        pih_cfg = self.cfg.precise_assembly
        if pih_cfg.goal_mode == "transportPreInsertFinal":
            # 5 subgoals; require averaging the 3 coarse waypoints before
            # we tighten — keeps the curriculum from shrinking on partial lifts.
            return float(self._num_prelude_goals + 1)
        if float(pih_cfg.random_goal_fraction) <= 0.0:
            return None
        return float(pih_cfg.random_goal_curriculum_success_threshold)

    def _wrench_dr_active_mask(self) -> torch.Tensor:
        """Disable wrench DR once the final insert is achieved (retract phase).

        The peg is unconstrained inside the hole during retract — random
        impulses there would knock it out before the fingers clear the
        keepout, defeating the retract reward.
        """
        return ~self.retract_phase


    def _retract_seat_measures(self):
        """Peg root vs the final waypoint: height above it along the hole axis, sideways offset, tilt (deg)."""
        N = self.num_envs
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        hq = self.hole_quat_wxyz
        seat = self.hole_pos + quat_apply(hq, self._insert_pos_rel[-1].unsqueeze(0).expand(N, 3))
        seat_q = quat_mul(hq, self._insert_quat_wxyz[-1].unsqueeze(0).expand(N, 4))
        axis = quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(N, 3))
        d = obj_pos - seat
        height = (d * axis).sum(-1)
        lateral = (d - height.unsqueeze(-1) * axis).norm(dim=-1)
        body_axis = quat_apply_inverse(seat_q, axis)          # peg-frame direction that is along the hole axis when seated
        cur = quat_apply(self.object.data.root_quat_w, body_axis)
        tilt = torch.rad2deg(torch.acos((cur * axis).sum(-1).clamp(-1.0, 1.0)))
        return {"height": height, "lateral": lateral, "tilt_deg": tilt}

    def _retract_seated_by_depth(self, pih_cfg):
        m = self._retract_seat_measures()
        return ((m["height"] <= float(pih_cfg.retract_seated_axial_tol))
                & (m["lateral"] <= float(pih_cfg.retract_seated_lateral_tol))
                & (m["tilt_deg"] <= float(pih_cfg.retract_seated_tilt_deg)))
    def _apply_goal_skip(self) -> None:
        """allow_goal_skip: jump to a later insertion waypoint reached out of order.

        Same test as the in-order goal (keypoint max dist within the insertion
        tolerance, held success_steps steps), applied to every waypoint after the
        current one. On a jump, _successes is set so the regular +1 in _get_dones
        lands just past the reached waypoint (crediting it and all before it).
        """
        if self._num_prelude_goals > 0:
            raise NotImplementedError("allow_goal_skip supports goal modes without prelude waypoints")
        active = ~self.is_random_goal_env & ~self.retract_phase & (self._successes < self.env_max_goals)
        if not active.any():
            return
        N, G = self.num_envs, self._num_insertion_goals
        kp = (self._keypoint_offsets_fixed if self.cfg.reward.fixed_size_keypoint_reward
              else self._keypoint_offsets)
        K = kp.shape[1]
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        obj_kp = _keypoints_world(obj_pos, self.object.data.root_quat_w, kp)              # (N, K, 3)

        hole_q = self.hole_quat_wxyz.unsqueeze(1).expand(N, G, 4).reshape(-1, 4)
        wp_pos = self.hole_pos.unsqueeze(1) + quat_apply(
            hole_q, self._insert_pos_rel.unsqueeze(0).expand(N, G, 3).reshape(-1, 3)
        ).reshape(N, G, 3)
        wp_q = quat_mul(hole_q, self._insert_quat_wxyz.unsqueeze(0).expand(N, G, 4).reshape(-1, 4))
        wp_kp = _keypoints_world(
            wp_pos.reshape(-1, 3), wp_q, kp.unsqueeze(1).expand(N, G, K, 3).reshape(-1, K, 3)
        ).reshape(N, G, K, 3)
        dist = (obj_kp.unsqueeze(1) - wp_kp).norm(dim=-1).max(dim=-1).values               # (N, G)
        hole_axis = quat_apply(
            self.hole_quat_wxyz, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(N, 3)
        )
        axial_err = ((obj_pos.unsqueeze(1) - wp_pos) * hole_axis.unsqueeze(1)).sum(-1).abs()  # (N, G)

        idx = torch.arange(G, device=self.device)
        near_later = (
            (dist <= self._keypoint_success_tolerance_m().unsqueeze(1))
            & (axial_err <= float(self.cfg.precise_assembly.goal_skip_axial_tol))
            & (idx.unsqueeze(0) > self._successes.unsqueeze(1))
            & active.unsqueeze(1)
        )
        far = torch.where(near_later, idx.unsqueeze(0), torch.full_like(near_later, -1, dtype=torch.long)).max(dim=1).values
        held = (far >= 0) & (far == self._skip_cand)
        self._skip_steps = torch.where(held, self._skip_steps + 1, (far >= 0).long())
        self._skip_cand = far
        jump = self._skip_steps >= int(self.cfg.termination.success_steps)
        if jump.any():
            self._successes[jump] = far[jump]
            self._is_success[jump] = True
            self._skip_cand[jump] = -1
            self._skip_steps[jump] = 0

    def _apply_unlock_goal(self, prev_near_steps: torch.Tensor) -> None:
        """Hidden unlock angle: at the last waypoint the near-goal test is joint angle >= theta - unlock_tol (the
        waypoint's pose is a virtual over-turn, the same for every theta, so the goal obs does not reveal theta)."""
        from isaacsimenvs.tasks.play.utils.reward_utils import update_near_goal_steps
        pih_cfg = self.cfg.precise_assembly
        self._update_joint_angles()
        ang = self.cb_joint_angle[int(pih_cfg.unlock_joint)]
        self.unlock_max_angle = torch.maximum(self.unlock_max_angle, ang)
        unlocked = ang >= self.unlock_angle - math.radians(float(pih_cfg.unlock_tol_deg))
        self.unlock_reached |= unlocked
        at_last = (self._successes == self.env_max_goals - 1) & ~self.is_random_goal_env
        steps = update_near_goal_steps(near_goal=unlocked, near_goal_steps=prev_near_steps,
                                       force_consecutive=self.cfg.termination.force_consecutive_near_goal_steps)
        self._near_goal = torch.where(at_last, unlocked, self._near_goal)
        self._near_goal_steps = torch.where(at_last, steps, self._near_goal_steps)
        self._is_success = self._near_goal_steps >= self.cfg.termination.success_steps
        self.extras["unlock_angle_deg_mean"] = torch.rad2deg(ang).mean()
        self.extras["unlock_reached_frac"] = self.unlock_reached.float().mean()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        prev_near_steps = self._near_goal_steps.clone()
        turn = getattr(self, "turn_mode", False)
        if turn:
            self._turn_fresh_closest = self._closest_keypoint_max_dist < 0.0
            self._turn_prev_near_steps = prev_near_steps
        compute_intermediate_values(self)

        pih_cfg = self.cfg.precise_assembly
        if turn:
            self._turn_goal_values()
        elif getattr(self, "unlock_angle", None) is not None:
            self._apply_unlock_goal(prev_near_steps)
        if pih_cfg.allow_goal_skip:
            self._apply_goal_skip()
        is_success = self._is_success.clone()
        if pih_cfg.enable_retract:
            is_success &= ~self.retract_phase
        self._is_success = is_success

        self._successes += is_success.long()
        self._successes.copy_(torch.minimum(self._successes, self.env_max_goals))
        success_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self.episode_length_buf[success_ids] = 0

        self._just_entered_retract[:] = False
        self._just_retracted[:] = False
        if pih_cfg.enable_retract:
            just_entered = (
                (self._successes >= self.env_max_goals)
                & ~self.retract_phase
                & ~self.is_random_goal_env
            )
            self._just_entered_retract.copy_(just_entered)
            self.retract_phase |= just_entered

            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            )
            if getattr(pih_cfg, "retract_seated_mode", "keypoint") == "depth":
                object_at_goal = self._retract_seated_by_depth(pih_cfg)
            off_d = float(getattr(pih_cfg, "retract_object_off_joint_m", 0.0))
            if off_d > 0.0 and getattr(self, "turn_mode", False):
                # A7 (official key_off_screw): the held tool must end away from the turn joint's axis (horizontal
                # distance of its root from the axis > off_d), e.g. not left standing in the screw's recess
                axis_w, anchor_w = self._turn_axis_w()
                rel = self.object.data.root_pos_w - anchor_w
                radial = rel - (rel * axis_w).sum(-1, keepdim=True) * axis_w
                object_at_goal = object_at_goal & (radial.norm(dim=-1) > off_d)
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            if pih_cfg.end_on_retract:
                # Strict: off the peg (distance to its middle, not its root) and no tactile contact for a while.
                r_dist, r_clear = float(pih_cfg.retract_min_fingertip_dist), int(pih_cfg.retract_clear_steps)
                if int(pih_cfg.retract_ramp_steps) > 0:
                    f = min(1.0, float(self.common_step_counter + int(pih_cfg.retract_ramp_offset_steps))
                            / float(pih_cfg.retract_ramp_steps))
                    r_dist = float(pih_cfg.retract_ramp_start_dist) + f * (r_dist - float(pih_cfg.retract_ramp_start_dist))
                    r_clear = int(round(int(pih_cfg.retract_ramp_start_clear)
                                        + f * (r_clear - int(pih_cfg.retract_ramp_start_clear))))
                    self.extras["retract_ramp_dist"] = r_dist
                    self.extras["retract_ramp_clear"] = float(r_clear)
                self._retract_cur_dist = r_dist
                far = self._curr_fingertip_dist_center.min(dim=-1).values > r_dist
                tac = getattr(self, "_last_tactile", None)
                clear = far if tac is None else far & (tac.reshape(tac.shape[0], -1).sum(-1) == 0)
                if not hasattr(self, "_retract_clear_count"):
                    self._retract_clear_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self._retract_clear_count = torch.where(clear & self.retract_phase, self._retract_clear_count + 1,
                                                        torch.zeros_like(self._retract_clear_count))
                hand_away = self._retract_clear_count >= r_clear
                if getattr(self, "turn_mode", False) and getattr(pih_cfg, "turn_early_release_fail", False):
                    started = self.turn_angle - self.turn_offset >= math.radians(float(pih_cfg.turn_first_goal_deg))
                    pre = clear & ~self.retract_phase & started
                    self._turn_early_clear = torch.where(pre, self._turn_early_clear + 1,
                                                         torch.zeros_like(self._turn_early_clear))
                    new_e = (self._turn_early_clear >= r_clear) & ~self.retract_phase & ~self.turn_released_early
                    self._turn_just_released_early = new_e
                    self.turn_released_early |= new_e
            else:
                hand_away = mean_fingertip_dist > pih_cfg.retract_distance_threshold
            just_retracted = (
                hand_away
                & self.retract_phase
                & ~self.retract_succeeded
                & object_at_goal
            )
            self._just_retracted.copy_(just_retracted)
            self.retract_succeeded |= just_retracted
            if pih_cfg.end_on_retract:
                self.retract_failed |= hand_away & self.retract_phase & ~self.retract_succeeded
                # Seat measurements at the moment the hand clears (kept across the auto-reset so evals can read them).
                if not hasattr(self, "_retract_measured"):
                    self._retract_measured = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._retract_meas = {}
                first = hand_away & self.retract_phase & ~self._retract_measured
                if first.any():
                    self._retract_measured |= first
                    m = self._retract_seat_measures()
                    for k, v in m.items():
                        buf = self._retract_meas.setdefault(k, torch.full((self.num_envs,), float("nan"), device=self.device))
                        buf[first] = v[first]
                    self._retract_meas.setdefault("kp_max", torch.full((self.num_envs,), float("nan"), device=self.device))[first] = \
                        self._keypoints_max_dist[first]

        if success_ids.numel() > 0:
            next_goal = (
                is_success
                & (self._successes < self.env_max_goals)
                & ~self.retract_phase
            )
            next_goal_ids = next_goal.nonzero(as_tuple=False).squeeze(-1)
            if next_goal_ids.numel() > 0:
                self._clear_goal_trackers(next_goal_ids)
                self._write_goal_pose(next_goal_ids, is_first_goal=False)

        object_z_local = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fall = object_z_local < 0.1
        if pih_cfg.enable_retract:
            random_goals_done = (
                (self._successes >= self.env_max_goals) & self.is_random_goal_env
            )
            max_successes = self.retract_succeeded | random_goals_done
            hand_far = (
                self._curr_fingertip_distances.max(dim=-1).values > 1.5
            ) & ~self.retract_phase
        else:
            max_successes = self._successes >= self.env_max_goals
            hand_far = self._curr_fingertip_distances.max(dim=-1).values > 1.5

        if pih_cfg.enable_dropped_on_table_term:
            table_top_local = self._table_z_per_env + TABLE_HALF_HEIGHT
            mean_ft_dist = self._curr_fingertip_distances.mean(dim=-1)
            dropped = (
                (object_z_local < table_top_local + pih_cfg.dropped_on_table_z_margin)
                & (mean_ft_dist > pih_cfg.dropped_on_table_ft_distance)
                & ~self.retract_phase
            )
        else:
            dropped = torch.zeros_like(fall)

        retract_fail = self.retract_failed if pih_cfg.enable_retract else torch.zeros_like(fall)
        self._update_distractor_disp()
        self._update_joint_angles()
        if pih_cfg.terminate_on_disturb and hasattr(self, "distractor_disp"):
            disturbed = self.distractor_disp > self._distractor_curriculum()[0]
        else:
            disturbed = torch.zeros_like(fall)
        # clones: reset_idx clears these flags in place, after which the reasons dict must still hold this step's ends
        stripped = self.turn_stripped.clone() if getattr(self, "turn_mode", False) else torch.zeros_like(fall)
        released_early = (self.turn_released_early.clone() if getattr(self, "turn_mode", False)
                          else torch.zeros_like(fall))
        toolless = (self.turn_toolless.clone() if getattr(self, "turn_mode", False) else torch.zeros_like(fall))
        terminated = (fall | max_successes | hand_far | dropped | retract_fail | disturbed | stripped | released_early
                      | toolless)
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
            "stripped": stripped,
            "released_early": released_early,
            "toolless": toolless,
            "fall": fall,
            "max_successes": max_successes,
            "retract_fail": retract_fail,
            "hand_far": hand_far,
            "dropped": dropped,
            "disturbed": disturbed,
            "timeout": truncated,
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        pih_cfg = self.cfg.precise_assembly

        mean_rg_eps = 0.0
        if self.is_random_goal_env.any():
            mean_rg_eps = (
                self._prev_episode_successes[self.is_random_goal_env]
                .float()
                .mean()
                .item()
            )
        if (
            self.lift_bonus_active
            and mean_rg_eps >= float(pih_cfg.lift_bonus_fade_threshold)
        ):
            self.lift_bonus_active = False

        if self.lift_bonus_active:
            if pih_cfg.force_lift_reward_active:
                lift_mask = torch.ones(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
            else:
                lift_mask = self.is_random_goal_env.float()
        else:
            lift_mask = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        lift_rew = self._reward_terms["lifting_rew"]
        lift_bonus_rew = self._reward_terms["lift_bonus_rew"]
        masked_lift_rew = lift_rew * lift_mask
        masked_lift_bonus_rew = lift_bonus_rew * lift_mask
        reward = reward - lift_rew - lift_bonus_rew + masked_lift_rew + masked_lift_bonus_rew
        self._reward_terms["lifting_rew"] = masked_lift_rew
        self._reward_terms["lift_bonus_rew"] = masked_lift_bonus_rew
        if getattr(self, "turn_mode", False) and float(pih_cfg.turn_strip_penalty) != 0.0:
            strip_pen = -float(pih_cfg.turn_strip_penalty) * self._turn_just_stripped.float()
            reward = reward + strip_pen
            self._reward_terms["turn_strip_penalty"] = strip_pen
        if getattr(self, "turn_mode", False) and float(getattr(pih_cfg, "turn_toolless_penalty", 0.0)) != 0.0:
            tl_pen = -float(pih_cfg.turn_toolless_penalty) * self._turn_just_toolless.float()
            reward = reward + tl_pen
            self._reward_terms["turn_toolless_penalty"] = tl_pen
        if getattr(self, "turn_mode", False) and float(pih_cfg.turn_over_penalty) != 0.0:
            over_pen = -float(pih_cfg.turn_over_penalty) * self._turn_over_inc
            reward = reward + over_pen
            self._reward_terms["turn_over_penalty"] = over_pen
        if getattr(self, "turn_mode", False) and getattr(pih_cfg, "turn_early_release_fail", False):
            early_pen = -float(pih_cfg.turn_early_release_penalty) * self._turn_just_released_early.float()
            reward = reward + early_pen
            self._reward_terms["turn_early_release_penalty"] = early_pen
        self._reward_terms["total_reward"] = reward
        self.extras["lift_bonus_active"] = float(self.lift_bonus_active)
        self.extras["mean_rg_prev_ep_successes"] = mean_rg_eps

        if pih_cfg.enable_retract:
            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            ).float()
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            if getattr(pih_cfg, "retract_reward_delta", False):
                cap = float(getattr(self, "_retract_cur_dist", 0.0) or pih_cfg.retract_min_fingertip_dist or 0.1) + 0.05
                d = mean_fingertip_dist.clamp(max=cap)
                if not hasattr(self, "_retract_prev_d"):
                    self._retract_prev_d = d.clone()
                prev = torch.where(self._just_entered_retract, d, self._retract_prev_d)
                self._retract_prev_d = d.clone()
                mean_fingertip_dist = d - prev
            retract_rew = (
                mean_fingertip_dist * pih_cfg.retract_reward_scale * object_at_goal
                + self._just_retracted.float() * self.retract_success_bonus
            ) * self.retract_phase.float()

            already_in_retract = self.retract_phase & ~self._just_entered_retract
            action_penalty = (
                self._reward_terms["kuka_actions_penalty"]
                + self._reward_terms["hand_actions_penalty"]
            )
            reward = torch.where(already_in_retract, action_penalty + retract_rew, reward)
            self._reward_terms["retract_rew"] = retract_rew
            self._reward_terms["total_reward"] = reward

            self.extras["retract_phase_ratio"] = self.retract_phase.float().mean()
            self.extras["retract_success_ratio"] = self.retract_succeeded.float().mean()
            self.extras["retract_success_tolerance"] = float(
                pih_cfg.retract_success_tolerance
            )
            self.extras["retract_success_bonus"] = self.retract_success_bonus

        if hasattr(self, "distractor_disp"):
            if pih_cfg.distractor_penalty_scale > 0.0:
                pen = pih_cfg.distractor_penalty_scale * self._distractor_curriculum()[1] * self.distractor_disp
                reward = reward - pen
                self._reward_terms["distractor_penalty"] = -pen
                self._reward_terms["total_reward"] = reward
            if pih_cfg.distractor_lift_penalty_scale > 0.0:
                lift_pen = pih_cfg.distractor_lift_penalty_scale * self._distractor_curriculum()[1] * torch.clamp(
                    self.distractor_rise - pih_cfg.distractor_lift_slack, min=0.0)
                reward = reward - lift_pen
                self._reward_terms["distractor_lift_penalty"] = -lift_pen
                self._reward_terms["total_reward"] = reward
            if pih_cfg.distractor_disturb_terminal_penalty > 0.0 and "disturbed" in self._termination_reasons:
                term_pen = pih_cfg.distractor_disturb_terminal_penalty * self._termination_reasons["disturbed"].float()
                reward = reward - term_pen
                self._reward_terms["distractor_terminal_penalty"] = -term_pen
                self._reward_terms["total_reward"] = reward
            self.extras["distractor_disp_mean"] = self.distractor_disp.mean()
            self.extras["distractor_rise_mean"] = self.distractor_rise.mean()
            self.extras["distractor_disturb_tol"] = torch.tensor(self._distractor_curriculum()[0], device=self.device)

        # One-time penalty when the episode ends with the object below the fall height: after the lift bonus the
        # remaining steps only cost action penalties, so throwing a small part off the table paid (bolt runs).
        if pih_cfg.fall_terminal_penalty > 0.0 and "fall" in self._termination_reasons:
            fall_pen = pih_cfg.fall_terminal_penalty * self._termination_reasons["fall"].float()
            reward = reward - fall_pen
            self._reward_terms["fall_terminal_penalty"] = -fall_pen
            self._reward_terms["total_reward"] = reward

        log_step_metrics(self)
        self._log_peg_metrics()
        return reward

    def _log_peg_metrics(self) -> None:
        success_ratio = self._successes.float() / self.env_max_goals.clamp_min(1).float()
        all_goals_hit = self._successes >= self.env_max_goals

        episode_final = self.extras.setdefault("episode_final", {})
        episode_final["success_ratio"] = success_ratio
        episode_final["all_goals_hit"] = all_goals_hit.float()
        if self.cfg.precise_assembly.enable_retract:
            episode_final["retract_success"] = self.retract_succeeded.float()

        # Plain per-episode outcome, logged as success/<key> (EnvStatsAlgoObserver): read at each episode's end and
        # averaged over the last games_to_track finished episodes. task = the episode did the whole task (all goals,
        # and the release check when retract is on); any_goal = reached at least one goal ("succeeded once").
        all_goals = all_goals_hit.float()
        task = (self.retract_succeeded.float() if self.cfg.precise_assembly.enable_retract else all_goals)
        sf = {"task": task, "all_goals": all_goals, "any_goal": (self._successes >= 1).float(),
              "goals_reached": self._successes.float(), "goals_total": self.env_max_goals.float(),
              "goals_frac": success_ratio}
        if hasattr(self, "distractor_disp_ep_max"):
            sf["neighbours_moved_cm"] = self.distractor_moved_ep_max * 100.0
            sf["neighbours_out_of_box_cm"] = self.distractor_disp_ep_max * 100.0
            sf["ended_disturbed"] = self._termination_reasons["disturbed"].float() \
                if "disturbed" in self._termination_reasons else torch.zeros_like(task)
        self.extras["success_final"] = {k: v.detach().clone() for k, v in sf.items()}

        prev_ratio = (
            self._prev_episode_successes.float()
            / self.prev_episode_env_max_goals.clamp_min(1).float()
        )
        self.extras["success_ratio"] = prev_ratio.mean()
        self.extras["all_goals_hit_ratio"] = (
            self._prev_episode_successes >= self.prev_episode_env_max_goals
        ).float().mean()
        self.extras["insertion_success_tolerance"] = float(
            self.cfg.precise_assembly.insertion_success_tolerance
        )

        if float(self.cfg.precise_assembly.random_goal_fraction) > 0.0:
            prev_s = self._prev_episode_successes
            prev_mg = self.prev_episode_env_max_goals.clamp_min(1).float()
            ins_mask = ~self.prev_episode_is_random_goal
            rg_mask = self.prev_episode_is_random_goal
            self.extras["random_goal_frac"] = self.is_random_goal_env.float().mean()
            if ins_mask.any():
                self.extras["insertion_success_ratio"] = (
                    prev_s[ins_mask].float() / prev_mg[ins_mask]
                ).mean()
                self.extras["insertion_all_goals_hit_ratio"] = (
                    prev_s[ins_mask] >= self.prev_episode_env_max_goals[ins_mask]
                ).float().mean()
            else:
                self.extras["insertion_success_ratio"] = 0.0
                self.extras["insertion_all_goals_hit_ratio"] = 0.0
            if rg_mask.any():
                self.extras["random_goal_success_ratio"] = (
                    prev_s[rg_mask].float() / prev_mg[rg_mask]
                ).mean()
                self.extras["random_goal_all_goals_hit_ratio"] = (
                    prev_s[rg_mask] >= self.prev_episode_env_max_goals[rg_mask]
                ).float().mean()
            else:
                self.extras["random_goal_success_ratio"] = 0.0
                self.extras["random_goal_all_goals_hit_ratio"] = 0.0

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs = build_observations(self)
        if self._goal_kp_obs_slice is not None:
            policy = obs["policy"]
            policy[:, self._goal_kp_obs_slice].view(self.num_envs, -1, 3).sub_(
                self.goal_pos_obs_noise.unsqueeze(1)
            )
            clip = self.cfg.obs.clamp_abs_observations
            obs["policy"] = policy.clamp(-clip, clip)

        # Depth-distillation contract: when student_obs is enabled, the env
        # exposes THREE obs groups (consumed by different parts of the dagger
        # pipeline):
        #   "policy"      = flattened student obs [image_flat, proprio] →
        #                   read by the depth-CNN student network
        #   "critic"      = state_list (privileged) → read by the asymmetric
        #                   central-value critic
        #   "teacher_obs" = obs_list (proprio + object state, noisy, what the
        #                   frozen state-MLP teacher trained on) → read by
        #                   DAggerA2CAgent for teacher labeling. The
        #                   DAggerRlGamesVecEnvWrapper passes this through as
        #                   `self.obs["teacher"]` in the agent.
        # SAPG appends 1 block-id column to "obs" and "states" only
        # (a2c_common.py:602-604) — "teacher" is left untouched, and our
        # `Teacher.get_action()` re-appends the block-id internally.
        student_cfg = getattr(self.cfg, "student_obs", None)
        if student_cfg is not None and student_cfg.enabled:
            student = self.get_student_obs()
            image_flat = student["image"].reshape(self.num_envs, -1)
            proprio = student["proprio"].reshape(self.num_envs, -1)
            student_flat = torch.cat([image_flat, proprio], dim=-1)
            obs = {
                "policy": student_flat,
                "critic": obs["critic"],
                "teacher_obs": obs["policy"],
            }
        return obs


__all__ = ["PreciseAssemblyEnv", "PreciseAssemblyEnvCfg"]
