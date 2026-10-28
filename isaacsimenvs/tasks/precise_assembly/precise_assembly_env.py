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
        if len(problem_names) > 1 and any(PROBLEM_REGISTRY[n].joints for n in problem_names):
            raise ValueError("Problem.joints need a single problem")
        self._pih_joints = tuple(problem.joints)
        for j in self._pih_joints:
            if j.body != "object" and not (j.body.startswith("distractor")
                                           and int(j.body[10:]) < len(self._pih_distractor_urdf_abs)):
                raise ValueError(f"JointSpec.body must be 'object' or 'distractor<k>', got {j.body!r}")
            if j.body == "object" and not self._pih_object_start_rel:
                raise ValueError("a joint on the object needs Problem.object_start_pose_rel_receptive")

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
        if is_random_goal.any():
            rg_ids = env_ids[is_random_goal]
            self.hole_pos[rg_ids, 0:2] = 0.0
            self.hole_pos[rg_ids, 2] = -1.0

        yaw_range_deg = float(pih_cfg.hole_yaw_range_deg)
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
        for k, (body, pose) in enumerate(zip(dists, self._pih_distractor_pose_rel)):
            w = self._pose_rel_hole_to_world(env_ids, pose)
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

    def _apply_object_start_pose(self, env_ids: torch.Tensor) -> None:
        w = self._pose_rel_hole_to_world(env_ids, self._pih_object_start_rel)
        self.object.write_root_pose_to_sim(w, env_ids=env_ids)
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
        hang = float(pih_cfg.distractors_hang_below)
        dists = getattr(self, "distractors", ())
        if hang > 0.0 and dists:
            # the goal only counts while every distractor hangs at least `hang` below the object (it is carried by
            # the object, not held by the hand too): a key lifted off a keychain, not the whole bunch scooped up
            dz = torch.stack([b.data.root_pos_w[:, 2] for b in dists]) - self.object.data.root_pos_w[:, 2]
            self._distractors_hanging = (dz <= -hang).all(dim=0)
            tol = torch.where(self._distractors_hanging, tol, torch.full_like(tol, -1.0))
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

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)

        pih_cfg = self.cfg.precise_assembly
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
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
            if pih_cfg.end_on_retract:
                # Strict: off the peg (distance to its middle, not its root) and no tactile contact for a while.
                far = self._curr_fingertip_dist_center.min(dim=-1).values > pih_cfg.retract_min_fingertip_dist
                tac = getattr(self, "_last_tactile", None)
                clear = far if tac is None else far & (tac.reshape(tac.shape[0], -1).sum(-1) == 0)
                if not hasattr(self, "_retract_clear_count"):
                    self._retract_clear_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                self._retract_clear_count = torch.where(clear & self.retract_phase, self._retract_clear_count + 1,
                                                        torch.zeros_like(self._retract_clear_count))
                hand_away = self._retract_clear_count >= pih_cfg.retract_clear_steps
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
        terminated = fall | max_successes | hand_far | dropped | retract_fail | disturbed
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
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
        self._reward_terms["total_reward"] = reward
        self.extras["lift_bonus_active"] = float(self.lift_bonus_active)
        self.extras["mean_rg_prev_ep_successes"] = mean_rg_eps

        if pih_cfg.enable_retract:
            object_at_goal = (
                self._keypoints_max_dist
                <= pih_cfg.retract_success_tolerance * self.cfg.reward.keypoint_scale
            ).float()
            mean_fingertip_dist = self._curr_fingertip_distances.mean(dim=-1)
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

        tac = getattr(self, "_last_tactile", None)
        if pih_cfg.pinch_contact_reward > 0.0 and tac is not None:
            # pads in FINGERS order (thumb first); tactile rays only hit the object, so contact = touching the target
            pad = tac.reshape(self.num_envs, 5, -1).amax(dim=-1) > 0.0
            pinch = pad[:, 0] & pad[:, 1:].any(dim=-1) & ~self._lifted_object
            pinch_rew = pih_cfg.pinch_contact_reward * pinch.float()
            reward = reward + pinch_rew
            self._reward_terms["pinch_contact_rew"] = pinch_rew
            self._reward_terms["total_reward"] = reward

        if pih_cfg.fall_terminal_penalty > 0.0 and "fall" in self._termination_reasons:
            fall_pen = pih_cfg.fall_terminal_penalty * self._termination_reasons["fall"].float()
            reward = reward - fall_pen
            self._reward_terms["fall_terminal_penalty"] = -fall_pen
            self._reward_terms["total_reward"] = reward

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
