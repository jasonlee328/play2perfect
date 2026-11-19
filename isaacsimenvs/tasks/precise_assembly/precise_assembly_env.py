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

# Faithful ContactBench door (precise_assembly.latch_cbf), contactbench-sim f51f6aa kit/mechanisms/doorknob_module.py:
# nominal unlock angle (deg) -> pinion pitch radius (mm); the printed cavity stops the rack after 15.44 mm (the knob's
# hard stop) and the bolt clears the strike jamb after 15.24 mm.
CBF_PINION_RP_MM = {40.0: 22.50, 48.0: 18.75, 60.0: 15.00, 72.0: 12.50, 80.0: 11.25}
CBF_LATCH_TRAVEL_MM, CBF_JAMB_CLEAR_MM = 15.44, 15.24
CBF_SPRINGS = {5.0: (0.48, 0.15), 7.5: (0.67, 0.25), 10.0: (0.78, 0.31)}   # spring mm -> (K N m/rad, tau0 N m)


def cbf_door_params(unlock_deg, spring_mm) -> dict:
    """Per-env official knob parameters: stop / clear angle (deg), spring K (N m/rad) and preload tau0 (N m)."""
    import numpy as np
    unlock_deg = np.asarray(unlock_deg, dtype=np.float64)
    bad = sorted({float(v) for v in unlock_deg} - set(CBF_PINION_RP_MM))
    if bad:
        raise ValueError(f"latch_cbf: unlock angles {bad} are not official pinions {sorted(CBF_PINION_RP_MM)}")
    springs = [float(v) for v in spring_mm]
    if not springs or any(v not in CBF_SPRINGS for v in springs):
        raise ValueError(f"latch_cbf_spring_mm {spring_mm}: choose from {sorted(CBF_SPRINGS)}")
    rp = np.array([CBF_PINION_RP_MM[float(v)] for v in unlock_deg])
    k_vals = len({float(v) for v in unlock_deg})
    sp = np.array([springs[(i // max(1, k_vals)) % len(springs)] for i in range(len(unlock_deg))])
    return dict(stop_deg=np.degrees(CBF_LATCH_TRAVEL_MM / rp), clear_deg=np.degrees(CBF_JAMB_CLEAR_MM / rp),
                spring_mm=sp, K=np.array([CBF_SPRINGS[v][0] for v in sp]), tau0=np.array([CBF_SPRINGS[v][1] for v in sp]))


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
            # latch_cbf: theta = the clear angle of the env's pinion (the label / eval angle stays _unlock_deg_np)
            src = self._cbf["clear_deg"] if getattr(self, "_cbf", None) is not None else self._unlock_deg_np
            self.unlock_angle = torch.as_tensor(src, device=self.device, dtype=torch.float32).deg2rad()
            self.unlock_max_angle = torch.zeros(self.num_envs, device=self.device)
            self.unlock_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.unlock_prev_ep_max = torch.zeros(self.num_envs, device=self.device)
            print(f"[PreciseAssemblyEnv] unlock angle per env: {self._unlock_deg_np.min():.1f}-"
                  f"{self._unlock_deg_np.max():.1f} deg (stop + {pih_cfg.unlock_stop_margin_deg}, "
                  f"tol {pih_cfg.unlock_tol_deg})", flush=True)

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
        if cfg.precise_assembly.latch_hint_obs and int(cfg.precise_assembly.latch_door_joint) >= 0:
            OBS_FIELD_SIZES["latch_hint"] = 2
            if "latch_hint" not in cfg.obs.obs_list:
                cfg.obs.obs_list = (*cfg.obs.obs_list, "latch_hint")
            if cfg.precise_assembly.latch_cue_obs:
                OBS_FIELD_SIZES["latch_cue"] = 3
                if "latch_cue" not in cfg.obs.obs_list:
                    cfg.obs.obs_list = (*cfg.obs.obs_list, "latch_cue")
        if cfg.precise_assembly.latch_critic_obs and int(cfg.precise_assembly.latch_door_joint) >= 0:
            OBS_FIELD_SIZES["latch_state"] = 5
            if "latch_state" not in cfg.obs.state_list:
                cfg.obs.state_list = (*cfg.obs.state_list, "latch_state")
        if len(problem_names) > 1 and any(PROBLEM_REGISTRY[n].joints for n in problem_names):
            raise ValueError("Problem.joints need a single problem")
        self._pih_joints = tuple(problem.joints)
        for j in self._pih_joints:
            if j.body != "object" and not (j.body.startswith("distractor")
                                           and int(j.body[10:]) < len(self._pih_distractor_urdf_abs)):
                raise ValueError(f"JointSpec.body must be 'object' or 'distractor<k>', got {j.body!r}")
            par = getattr(j, "parent", "fixture")
            if par != "fixture" and not (par.startswith("distractor")
                                         and int(par[10:]) < len(self._pih_distractor_urdf_abs)):
                raise ValueError(f"JointSpec.parent must be 'fixture' or 'distractor<k>', got {par!r}")
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
        self._button_slot_np = None
        if pih_cfg.button_mode:
            import numpy as np
            if not (self._pih_object_start_rel and self._pih_distractor_pose_rel):
                raise ValueError("button_mode needs object_start_pose + distractors (the key slots)")
            n_env, B = int(cfg.scene.num_envs), 1 + len(self._pih_distractor_pose_rel)
            rng = np.random.default_rng(int(pih_cfg.button_seed) + 7)
            tgt = (np.full(n_env, int(pih_cfg.button_fixed_slot)) if int(pih_cfg.button_fixed_slot) >= 0
                   else rng.integers(0, B, n_env))
            slots = np.zeros((n_env, B), dtype=np.int64)
            for i in range(n_env):
                rest = [b for b in range(B) if b != tgt[i]]
                slots[i] = [tgt[i]] + rest     # body 0 (target) in slot tgt, dead keys fill the rest in order
            self._button_slot_np = slots
            home = np.array([self._pih_object_start_rel[:3]] + [p[:3] for p in self._pih_distractor_pose_rel])
            self._button_home_np = home          # slot b = the problem's pose of body b
            # offset of body b in env i (receptive frame) = slot position - its default position
            self._button_off_np = home[slots] - home[None, :, :]        # (N, B, 3)
            print(f"[PreciseAssemblyEnv] button_mode: {B} keys, target slot counts "
                  f"{np.bincount(tgt, minlength=B).tolist()}", flush=True)
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
        self._cbf = None
        if bool(getattr(pih_cfg, "latch_cbf", False)):
            if self._unlock_deg_np is None or int(pih_cfg.latch_door_joint) < 0:
                raise ValueError("latch_cbf needs latch_door_joint >= 0 and unlock_angle_values_deg")
            if int(pih_cfg.latch_keeper) >= 0:
                raise ValueError("latch_cbf has no keeper: set latch_keeper=-1 (and kinematic_distractors=[])")
            self._cbf = cbf_door_params(self._unlock_deg_np, pih_cfg.latch_cbf_spring_mm)

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
        self._update_latch(env_ids)
        if self._pih_object_start_rel and insertion_mask.any():
            self._apply_object_start_pose(env_ids[insertion_mask])

        if pih_cfg.start_state_file and insertion_mask.any():
            self._apply_start_states(env_ids[insertion_mask])

        if pih_cfg.button_mode and hasattr(self, "button_travel_ep_max"):
            self.button_prev_ep_max = getattr(self, "button_prev_ep_max", torch.zeros_like(self.button_travel_ep_max))
            self.button_prev_ep_max[:, env_ids] = self.button_travel_ep_max[:, env_ids]   # kept for evals
            self.button_travel_ep_max[:, env_ids] = 0.0
            self.button_travel[:, env_ids] = 0.0
            if hasattr(self, "_button_phi_prev"):
                self._button_phi_prev[env_ids] = 0.0
            if hasattr(self, "_button_touched"):
                self._button_touched[:, env_ids] = False
        if getattr(self, "unlock_angle", None) is not None:
            self.unlock_prev_ep_max[env_ids] = self.unlock_max_angle[env_ids]  # kept across the auto-reset
            self.unlock_max_angle[env_ids] = 0.0
            self.unlock_reached[env_ids] = False
            if hasattr(self, "door_max_angle"):
                self.door_prev_ep_max[env_ids] = self.door_max_angle[env_ids]
                self.door_max_angle[env_ids] = 0.0
        self._clear_goal_trackers(env_ids)
        self._write_goal_pose(env_ids, is_first_goal=True)

    def _body_slot_offset_np(self, body: str, i: int):
        """button_mode: receptive-frame offset of ``body`` ("object" / "distractor<k>") in env i (zeros otherwise)."""
        import numpy as np
        if getattr(self, "_button_slot_np", None) is None:
            return np.zeros(3)
        b = 0 if body == "object" else 1 + int(body[10:])
        return self._button_off_np[i, b]

    def _button_off(self, b: int, env_ids: torch.Tensor) -> torch.Tensor | None:
        if getattr(self, "_button_slot_np", None) is None:
            return None
        if not hasattr(self, "_button_off_t"):
            self._button_off_t = torch.as_tensor(self._button_off_np, device=self.device, dtype=torch.float32)
        return self._button_off_t[env_ids, b]

    def _button_update(self) -> None:
        """Travel (m, + = pressed) of every key (B, N), per-episode max, wrong-key flag."""
        bodies = [self.object, *self.distractors]
        pos = torch.stack([b.data.root_pos_w for b in bodies])                     # (B, N, 3)
        rel = pos - (self.hole_pos + self.scene.env_origins).unsqueeze(0)
        B = rel.shape[0]
        rel = quat_apply_inverse(self.hole_quat_wxyz.unsqueeze(0).expand(B, -1, -1).reshape(-1, 4),
                                 rel.reshape(-1, 3)).reshape(B, -1, 3)
        if not hasattr(self, "_button_rest_z"):
            self._button_rest_z = torch.as_tensor(self._button_home_np[:, 2], device=self.device,
                                                  dtype=torch.float32).unsqueeze(-1)
            self.button_travel_ep_max = torch.zeros(B, self.num_envs, device=self.device)
        self.button_travel = (self._button_rest_z - rel[..., 2]).clamp(min=0.0)
        self.button_travel_ep_max = torch.maximum(self.button_travel_ep_max, self.button_travel)
        c = self.cfg.precise_assembly
        self.button_wrong = (self.button_travel[1:] > float(c.button_wrong_m)).any(dim=0)

    def _apply_button_goal(self, prev_near_steps: torch.Tensor) -> None:
        """The only goal is credited by the target's travel, not keypoints (3 mm is inside the keypoint tolerance)."""
        from isaacsimenvs.tasks.play.utils.reward_utils import update_near_goal_steps
        c = self.cfg.precise_assembly
        self._button_update()
        pressed = self.button_travel[0] >= float(c.button_actuate_m)
        if c.button_require_pad:
            pressed = pressed & self._button_pad_on_target()
        at_last = (self._successes == self.env_max_goals - 1) & ~self.is_random_goal_env
        steps = update_near_goal_steps(near_goal=pressed, near_goal_steps=prev_near_steps,
                                       force_consecutive=True)
        self._near_goal = torch.where(at_last, pressed, self._near_goal)
        self._near_goal_steps = torch.where(at_last, steps, self._near_goal_steps)
        self._is_success = self._near_goal_steps >= self.cfg.termination.success_steps
        self.extras["button_target_travel_mm"] = self.button_travel[0].mean() * 1e3
        self.extras["button_dead_travel_mm"] = self.button_travel[1:].max(dim=0).values.mean() * 1e3
        self.extras["button_pressed_frac"] = pressed.float().mean()

    def _button_pad_on_target(self) -> torch.Tensor:
        ts = getattr(self, "_tactile", None)
        if ts is None or not hasattr(ts, "body_pad_depth"):
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return (ts.body_pad_depth[0] > 0).any(dim=-1)

    def _button_virtual_pose(self):
        """Actor's view: (pos rel. env origin, quat wxyz) of a virtual key at the pad centre moved down by the max
        travel of all keys, and its virtual goal (same goal offset as the real one)."""
        if not hasattr(self, "_button_centre"):
            centre = self._button_home_np.mean(axis=0)
            self._button_centre = torch.as_tensor(centre, device=self.device, dtype=torch.float32)
            goal_off = torch.as_tensor(self._insert_pos_rel[-1], device=self.device, dtype=torch.float32) \
                - torch.as_tensor(self._button_home_np[0], device=self.device, dtype=torch.float32)
            self._button_goal_off = goal_off
            self._button_q0 = _xyzw_to_wxyz(torch.tensor([self._pih_object_start_rel[3:7]], device=self.device,
                                                         dtype=torch.float32))
        n = self.num_envs
        tr = getattr(self, "button_travel", None)
        down = tr.max(dim=0).values if tr is not None else torch.zeros(n, device=self.device)
        hq = self.hole_quat_wxyz
        p = self._button_centre.expand(n, 3).clone()
        p[:, 2] -= down
        pos = self.hole_pos + quat_apply(hq, p)
        q = quat_mul(hq, self._button_q0.expand(n, 4))
        gpos = self.hole_pos + quat_apply(hq, self._button_centre.expand(n, 3) + self._button_goal_off)
        return pos, q, gpos, q

    def _pose_rel_hole_to_world(self, env_ids: torch.Tensor, pose7, off: torch.Tensor | None = None) -> torch.Tensor:
        """(n, 7) world pose (pos + quat wxyz) of a receptive-frame pose7 (xyz + quat xyzw) for env_ids
        (+ an optional per-env receptive-frame position offset ``off`` (n, 3))."""
        n = env_ids.numel()
        p = torch.tensor(pose7[:3], device=self.device, dtype=torch.float32).expand(n, 3)
        if off is not None:
            p = p + off
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
            w = self._pose_rel_hole_to_world(env_ids, pose, self._button_off(1 + k, env_ids))
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
            par = getattr(j, "parent", "fixture")
            if par == "fixture":
                q_par = hq
            else:
                # parent's current world rotation x inverse(its reset rotation rel. the fixture) = where the
                # fixture frame would be if it moved with the parent, so the angle is relative to the parent
                kp = int(par[10:])
                qp0 = _xyzw_to_wxyz(torch.tensor([self._pih_distractor_pose_rel[kp][3:7]], device=self.device,
                                                 dtype=torch.float32)).expand(self.num_envs, 4)
                q_par = quat_mul(self.distractors[kp].data.root_quat_w, quat_conjugate(qp0))
            q_rel = quat_mul(quat_conjugate(q0), quat_mul(quat_conjugate(q_par), body.data.root_quat_w))  # body frame
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
        self._update_latch()

    def _apply_action(self) -> None:
        super()._apply_action()
        if self._latch_cbf_on():
            self._latch_cbf_substep()
            return
        if float(self.cfg.precise_assembly.latch_soft_stop_deg) > 0.0 and self._latch_on() and hasattr(self, "latch_open"):
            self._latch_soft_stop()
        if float(getattr(self.cfg.precise_assembly, "latch_knob_max_deg_s", 0.0)) > 0.0 and self._latch_on() \
                and hasattr(self, "latch_open"):
            self._latch_knob_speed_cap()

    def _latch_knob_speed_cap(self) -> None:
        """Every physics substep while the latch is closed: the knob may advance at most latch_knob_max_deg_s x sim dt
        about its axis; any extra turn is rotated back (pose level, like the soft stop: a velocity clamp is undone by the
        fingers' contacts) and the excess spin removed. Theta-independent, so joint state carries no theta."""
        c = self.cfg.precise_assembly
        kj = int(c.unlock_joint)
        self._update_joint_angles()
        a = self.cb_joint_angle[kj]
        if not hasattr(self, "_knob_cap_prev"):
            self._knob_cap_prev = a.clone()
            self._knob_cap_hits = torch.zeros(self.num_envs, device=self.device)
        prev = torch.where(torch.isnan(self._knob_cap_prev), a, self._knob_cap_prev)
        lim = math.radians(float(c.latch_knob_max_deg_s)) * float(self.cfg.sim.dt)
        delta = a - prev
        excess = delta - delta.clamp(-lim, lim)
        bad = (~self.latch_open) & (excess.abs() > 1e-6)
        excess = torch.where(bad, excess, torch.zeros_like(excess))
        self._knob_cap_prev = a - excess
        self._knob_cap_hits = self._knob_cap_hits + bad.float()
        ids = bad.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        j = self._pih_joints[kj]
        hq = self.hole_quat_wxyz[ids]
        n = ids.numel()
        ax = quat_apply(hq, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(n, 3))
        an = self.hole_pos[ids] + self.scene.env_origins[ids] + quat_apply(
            hq, torch.tensor(j.anchor, device=self.device, dtype=torch.float32).expand(n, 3))
        rot = quat_from_angle_axis(-excess[ids], ax)
        body = self.object
        pos, quat = body.data.root_pos_w[ids], body.data.root_quat_w[ids]
        body.write_root_pose_to_sim(torch.cat([an + quat_apply(rot, pos - an), quat_mul(rot, quat)], -1), env_ids=ids)
        vel = body.data.root_vel_w[ids].clone()
        door_w = self._joint_bodies()[int(c.latch_door_joint)].data.root_vel_w[ids, 3:6]
        rel = ((vel[:, 3:6] - door_w) * ax).sum(-1)
        cap = math.radians(float(c.latch_knob_max_deg_s))
        vel[:, 3:6] -= (rel - rel.clamp(-cap, cap)).unsqueeze(-1) * ax
        body.write_root_velocity_to_sim(vel, env_ids=ids)
        self.cb_joint_angle[kj, ids] -= excess[ids]

    def _tactile_object_pose(self):
        """Object (pos, quat) the fingertip tactile ray-casts against, or None (= the real pose). Door latch
        touch-only cue: knob raised latch_tactile_pop_mm (world z) while in the unlock window and still latched."""
        mm = float(getattr(self.cfg.precise_assembly, "latch_tactile_pop_mm", 0.0))
        om = float(getattr(self.cfg.precise_assembly, "latch_tactile_open_mm", 0.0))
        if (mm <= 0.0 and om == 0.0) or not self._latch_on() or not hasattr(self, "latch_in_window") \
                or self._latch_cbf_on():
            return None
        cue = getattr(self, "latch_cue_frac", self.latch_in_window.float()) * (~self.latch_open).float()
        dz = cue * (mm * 1e-3)
        if om != 0.0 and hasattr(self, "cb_joint_angle"):
            door = self.cb_joint_angle[int(self.cfg.precise_assembly.latch_door_joint)]
            shut = door < math.radians(float(self.cfg.precise_assembly.latch_soft_stop_deg) + 0.5)
            dz = dz + (self.latch_open & shut).float() * (om * 1e-3)
        up = torch.zeros_like(self.object.data.root_pos_w)
        up[:, 2] = dz
        return self.object.data.root_pos_w + up, self.object.data.root_quat_w

    def _latch_soft_stop(self) -> None:
        """Every physics substep while the latch is closed: door hinge angle <= latch_soft_stop_deg (door + knob rotated
        back about the hinge axis, the opening spin about it removed)."""
        c = self.cfg.precise_assembly
        k = int(c.latch_door_joint)
        self._update_joint_angles()
        over = self.cb_joint_angle[k] - math.radians(float(c.latch_soft_stop_deg))
        bad = (~self.latch_open) & (over > 0)
        if not hasattr(self, "_soft_stop_hits"):
            self._soft_stop_hits = torch.zeros(self.num_envs, device=self.device)
        self._soft_stop_hits = torch.where(bad, self._soft_stop_hits + 1, self._soft_stop_hits)
        self._hinge_rotate_back(bad, over)

    def _hinge_rotate_back(self, bad: torch.Tensor, over: torch.Tensor) -> None:
        """Rotate door (latch_door_joint's body) + knob (object) back by over (rad) about the hinge axis where bad, and
        remove their opening spin about it (pose level: a velocity / torque stop is out-pushed by the hand)."""
        c = self.cfg.precise_assembly
        k = int(c.latch_door_joint)
        ids = bad.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        j = self._pih_joints[k]
        hq = self.hole_quat_wxyz[ids]
        n = ids.numel()
        ax = quat_apply(hq, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(n, 3))
        an = self.hole_pos[ids] + self.scene.env_origins[ids] + quat_apply(
            hq, torch.tensor(j.anchor, device=self.device, dtype=torch.float32).expand(n, 3))
        rot = quat_from_angle_axis(-over[ids], ax)
        for body in (self.distractors[int(j.body[10:])], self.object):
            pos, quat = body.data.root_pos_w[ids], body.data.root_quat_w[ids]
            body.write_root_pose_to_sim(torch.cat([an + quat_apply(rot, pos - an), quat_mul(rot, quat)], -1), env_ids=ids)
            vel = body.data.root_vel_w[ids].clone()
            w_ax = (vel[:, 3:6] * ax).sum(-1).clamp(min=0.0)        # opening spin about the hinge
            vel[:, 3:6] -= w_ax.unsqueeze(-1) * ax
            vel[:, 0:3] = 0.0
            body.write_root_velocity_to_sim(vel, env_ids=ids)
        self.cb_joint_angle[k, ids] -= over[ids]

    def latch_hint_p(self) -> float:
        """Share of episodes that get the privileged theta hint now (1 -> latch_hint_final over the ramp)."""
        c = self.cfg.precise_assembly
        if int(c.latch_hint_ramp_steps) <= 0:
            return float(c.latch_hint_final)
        a = min(1.0, float(self.common_step_counter + int(c.latch_hint_offset_steps)) / float(c.latch_hint_ramp_steps))
        return 1.0 + a * (float(c.latch_hint_final) - 1.0)

    def _latch_hint_resample(self, env_ids) -> None:
        if not hasattr(self, "latch_hint_on"):
            self.latch_hint_on = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        p = self.latch_hint_p()
        self.latch_hint_on[env_ids] = torch.rand(len(env_ids), device=self.device) < p
        self.extras["latch_hint_p"] = torch.tensor(p, device=self.device)

    def _latch_on(self) -> bool:
        c = self.cfg.precise_assembly
        return int(c.latch_door_joint) >= 0 and (int(c.latch_keeper) >= 0 or getattr(self, "_cbf", None) is not None) \
            and getattr(self, "unlock_angle", None) is not None

    def _latch_cbf_on(self) -> bool:
        return getattr(self, "_cbf", None) is not None and getattr(self, "unlock_angle", None) is not None

    def _latch_cbf_substep(self) -> None:
        """latch_cbf, every physics substep (from _apply_action): the bolt is out of the strike once the knob (angle
        relative to the door) reaches this env's clear angle; it is back in while the knob is below it AND the door
        has been shut (<= latch_cbf_gap_deg + 0.05) since, i.e. it re-engages when the door comes down with the knob
        released (no 30 deg entry ramp in the official model either: there a released bolt rests on the pocket roof).
        While engaged the hinge is held at <= latch_cbf_gap_deg by rotating door + knob back (exact at any env count,
        unlike the kinematic keeper, findings 36). latch_open_ep = cleared at least once this episode."""
        c = self.cfg.precise_assembly
        self._update_joint_angles()
        if not hasattr(self, "_cbf_engaged"):
            self._init_latch_cbf_state()
        knob = self.cb_joint_angle[int(c.unlock_joint)]
        door = self.cb_joint_angle[int(c.latch_door_joint)]
        gap = math.radians(float(c.latch_cbf_gap_deg))
        cleared = knob >= self.unlock_angle
        self._cbf_engaged = ~cleared & (self._cbf_engaged | (door <= gap + math.radians(0.05)))
        self.latch_open = ~self._cbf_engaged
        self.latch_open_ep |= self.latch_open
        over = door - gap
        bad = self._cbf_engaged & (over > 0)
        self._soft_stop_hits = torch.where(bad, self._soft_stop_hits + 1, self._soft_stop_hits)
        self._cbf_over_max = torch.maximum(self._cbf_over_max, torch.where(self._cbf_engaged, over, torch.zeros_like(over)))
        if bool(getattr(c, "latch_cbf_elastic", False)):
            self._cbf_elastic_substep(knob, door, over)
        else:
            self._hinge_rotate_back(bad, over)
        self._cbf_spring_correction(self.cb_joint_angle[int(c.unlock_joint)])

    def _axis_inertia(self, body, ax: torch.Tensor, an: torch.Tensor) -> torch.Tensor:
        """Moment of inertia (kg m^2, (N,)) of a rigid body about the world line (an, ax): COM inertia + m d^2."""
        key = id(body)
        cache = self.__dict__.setdefault("_ax_inertia_cache", {})
        if key not in cache:
            v = body.root_physx_view
            cache[key] = (v.get_inertias().to(self.device).reshape(-1, 3, 3), v.get_masses().to(self.device).reshape(-1))
        I_b, m = cache[key]
        a_b = quat_apply_inverse(body.data.root_quat_w, ax)
        r = body.data.root_com_pos_w - an if hasattr(body.data, "root_com_pos_w") else body.data.root_pos_w - an
        d2 = (r * r).sum(-1) - (r * ax).sum(-1) ** 2
        return torch.einsum("ni,nij,nj->n", a_b, I_b, a_b) + m * d2

    def _elastic_rot(self, name: str, bodies, ax, an, over, active, K, D, back_deg, cap_deg=None):
        """Implicit elastic stop about the world line (an, ax) for a rigid group (bodies rotate together): over = angle
        past the stop (rad), active = envs where the stop acts. PhysX already integrated every other load into the pose
        (v_phys = (over - x_prev) / dt); the spring + damper act implicitly:
            v = (v_phys - dt K x_prev / I) / (1 + dt D / I + dt^2 K / I),  x = clamp(x_prev + dt v, 0, back).
        The group is rotated to x, its spin about the line set to v. Returns the stop torque K x + D v (N m)."""
        dt = float(self.physics_dt)
        n = self.num_envs
        st = self.__dict__.setdefault("_elastic_state", {})
        x_prev = st.get(name, torch.zeros(n, device=self.device))
        over_p = torch.clamp(over, min=0.0)
        el = active & (over_p > 0)
        I = sum(self._axis_inertia(b, ax, an) for b in bodies).clamp_min(1e-7)
        v_phys = (over_p - x_prev) / dt
        v = (v_phys - dt * K * x_prev / I) / (1.0 + dt * D / I + dt * dt * K / I)
        back = math.radians(back_deg)
        x = torch.clamp(x_prev + dt * v, min=0.0, max=back)
        v = torch.where(x >= back, torch.clamp(v, max=0.0), v)
        x = torch.where(el, x, torch.zeros_like(x)); v = torch.where(el, v, torch.zeros_like(v))
        st[name] = x
        corr = torch.where(el, x - over_p, torch.zeros_like(x))
        ids = el.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() > 0:
            rot = quat_from_angle_axis(corr[ids], ax[ids])
            for body in bodies:
                pos, quat = body.data.root_pos_w[ids], body.data.root_quat_w[ids]
                new_pos = an[ids] + quat_apply(rot, pos - an[ids])
                body.write_root_pose_to_sim(torch.cat([new_pos, quat_mul(rot, quat)], -1), env_ids=ids)
                vel = body.data.root_vel_w[ids].clone()
                w_ax = (vel[:, 3:6] * ax[ids]).sum(-1)
                dw = (v[ids] - w_ax).unsqueeze(-1) * ax[ids]
                vel[:, 3:6] += dw
                vel[:, 0:3] += torch.cross(dw, new_pos - an[ids], dim=-1)
                body.write_root_velocity_to_sim(vel, env_ids=ids)
        return torch.where(el, K * x + D * v, torch.zeros_like(x)), corr

    def _cbf_elastic_substep(self, knob: torch.Tensor, door: torch.Tensor, over_h: torch.Tensor) -> None:
        """latch_cbf_elastic (see the cfg): (1) engaged latch: elastic hinge stop at the gap for door + knob;
        (2) pinion stop: elastic knob stop at this env's stop angle (knob relative to the door, about the spindle)."""
        c = self.cfg.precise_assembly
        n = self.num_envs
        jh, jk = self._pih_joints[int(c.latch_door_joint)], self._pih_joints[int(c.unlock_joint)]
        door_body = self.distractors[int(jh.body[10:])]
        hq = self.hole_quat_wxyz
        base = self.hole_pos + self.scene.env_origins
        ax_h = quat_apply(hq, torch.tensor(jh.axis, device=self.device, dtype=torch.float32).expand(n, 3))
        an_h = base + quat_apply(hq, torch.tensor(jh.anchor, device=self.device, dtype=torch.float32).expand(n, 3))
        tau_h, corr_h = self._elastic_rot("hinge", (door_body, self.object), ax_h, an_h, over_h, self._cbf_engaged,
                                          float(c.latch_cbf_hinge_K), float(c.latch_cbf_hinge_D),
                                          float(c.latch_cbf_hinge_back_deg))
        self.cb_joint_angle[int(c.latch_door_joint)] += corr_h
        door_now = self.cb_joint_angle[int(c.latch_door_joint)]
        R = quat_from_angle_axis(door_now, ax_h)
        ax_k = quat_apply(R, quat_apply(hq, torch.tensor(jk.axis, device=self.device, dtype=torch.float32).expand(n, 3)))
        an_k0 = base + quat_apply(hq, torch.tensor(jk.anchor, device=self.device, dtype=torch.float32).expand(n, 3))
        an_k = an_h + quat_apply(R, an_k0 - an_h)
        if not hasattr(self, "_cbf_stop_rad"):
            self._cbf_stop_rad = torch.as_tensor([math.radians(float(v)) for v in self._cbf["stop_deg"]],
                                                 device=self.device, dtype=torch.float32)
        over_k = self.cb_joint_angle[int(c.unlock_joint)] - self._cbf_stop_rad
        tau_k, corr_k = self._elastic_rot("knob", (self.object,), ax_k, an_k, over_k, torch.ones_like(self._cbf_engaged),
                                          float(c.latch_cbf_stop_K), float(c.latch_cbf_stop_D),
                                          float(c.latch_cbf_stop_back_deg))
        self.cb_joint_angle[int(c.unlock_joint)] += corr_k
        # readings: max over the policy step's substeps (reset in _update_latch_cbf)
        if not hasattr(self, "door_stop_torque"):
            z = lambda: torch.zeros(n, device=self.device)
            self.door_stop_torque, self.door_pull_torque, self.door_pull_force = z(), z(), z()
            self._door_break_count = torch.zeros(n, dtype=torch.long, device=self.device)
        r_k = torch.linalg.norm(torch.cross(self.object.data.root_pos_w - an_h, ax_h, dim=-1), dim=-1).clamp_min(1e-3)
        # calibration (a1_door_probe --elastic, 11-17, stop K 30-300 / D 0-2, springs 5 / 10 mm, stops 40 / 80): the
        # raw stop reading = 0.96 x (torque into the stop) + 0.40 x (spring torque at the stop, tau0 + K stop) for every
        # K / D, so the spring's pull is only partly seen by the implicit step; subtract it (known per env) and rescale.
        if not hasattr(self, "_cbf_spring_at_stop"):
            self._cbf_spring_at_stop = (torch.as_tensor(self._cbf["tau0"], device=self.device, dtype=torch.float32)
                                        + torch.as_tensor(self._cbf["K"], device=self.device, dtype=torch.float32)
                                        * self._cbf_stop_rad)
        a_cal, b_cal = float(getattr(c, "latch_cbf_stop_cal_gain", 0.96)), float(getattr(c, "latch_cbf_stop_cal_spring", 0.40))
        tau_k = torch.where(tau_k > 0, torch.clamp((tau_k - b_cal * self._cbf_spring_at_stop) / a_cal, min=0.0), tau_k)
        # latch: the two-level probe (1.5 / 2.5 N m on the hinge) reads a slope of 0.75 -> pure gain (gravity matches)
        tau_h = tau_h / float(getattr(c, "latch_cbf_hinge_cal_gain", 0.75))
        self.door_stop_torque_now, self.door_pull_torque_now = tau_k, tau_h     # last substep (probes)
        self.door_stop_torque = torch.maximum(self.door_stop_torque, tau_k)
        self.door_pull_torque = torch.maximum(self.door_pull_torque, tau_h)
        self.door_pull_force = torch.maximum(self.door_pull_force, tau_h / r_k)

    def _cbf_spring_correction(self, knob: torch.Tensor) -> None:
        """The spring is a PhysX joint drive (implicit, stable) with stiffness K and target tgt = -tau0 / K (beyond the
        rest stop), but a D6 twist drive pulls with K * 2 sin(e / 2), e = angle - tgt, not K * e: up to 12 % soft at
        the 80 deg stop (probe). The difference -K (e - 2 sin(e / 2)) is added as an explicit torque pair on knob /
        door about the spindle (its stiffness <= K (1 - cos(e / 2)), ~0.3 K, is explicit-stable with the knob's
        spindle armature), so tau = tau0 + K theta exactly. Kept with any object wrench DR (added to it)."""
        c = self.cfg.precise_assembly
        if not hasattr(self, "_cbf_K"):
            self._cbf_K = torch.as_tensor(self._cbf["K"], device=self.device, dtype=torch.float32)
            self._cbf_tgt = -torch.as_tensor(self._cbf["tau0"], device=self.device, dtype=torch.float32) / self._cbf_K
        e = knob - self._cbf_tgt
        tau = -self._cbf_K * (e - 2.0 * torch.sin(0.5 * e))
        j = self._pih_joints[int(c.unlock_joint)]
        kp = int(j.parent[10:])
        door_body = self.distractors[kp]
        qp0 = _xyzw_to_wxyz(torch.tensor([self._pih_distractor_pose_rel[kp][3:7]], device=self.device,
                                         dtype=torch.float32)).expand(self.num_envs, 4)
        q_par = quat_mul(door_body.data.root_quat_w, quat_conjugate(qp0))
        ax = quat_apply(q_par, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(self.num_envs, 3))
        self._cbf_corr_w = tau.unsqueeze(-1) * ax                      # (N, 3) world torque on the knob
        f = getattr(self, "_object_forces", None)
        t = getattr(self, "_object_torques", None)
        zero = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.object.set_external_force_and_torque(zero if f is None else f,
                                                  (zero if t is None else t) + self._cbf_corr_w.unsqueeze(1),
                                                  is_global=True)
        door_body.set_external_force_and_torque(zero, -self._cbf_corr_w.unsqueeze(1), is_global=True)

    def _init_latch_cbf_state(self) -> None:
        z = lambda dt=torch.bool: torch.zeros(self.num_envs, dtype=dt, device=self.device)
        self._cbf_engaged = ~z()
        self.latch_open, self.latch_open_ep = z(), z()
        self.latch_in_window, self.latch_dwell = z(), z(torch.long)     # obs fields of the old design: always 0
        self._soft_stop_hits = z(torch.float32)
        self._cbf_over_max = z(torch.float32)          # max door angle above the gap while engaged, before correction
        self._cbf_prev_door = z(torch.float32)
        self._cbf_grasp_ok = z()
        self._cbf_door_still = ~z()

    def _update_latch_cbf(self, env_ids: torch.Tensor | None) -> None:
        """latch_cbf, per policy step / reset: reset the latch state; the latch itself is _latch_cbf_substep."""
        c = self.cfg.precise_assembly
        if not hasattr(self, "_cbf_engaged"):
            self._init_latch_cbf_state()
        if env_ids is not None:
            self._cbf_engaged[env_ids] = True
            self.latch_open[env_ids] = False
            self.latch_open_ep[env_ids] = False
            self._soft_stop_hits[env_ids] = 0.0
            self._cbf_over_max[env_ids] = 0.0
            self._cbf_prev_door[env_ids] = 0.0
            if hasattr(self, "door_stop_torque_ep_max"):
                self.door_stop_torque_last_ep_max[env_ids] = self.door_stop_torque_ep_max[env_ids]
                self.door_pull_force_last_ep_max[env_ids] = self.door_pull_force_ep_max[env_ids]
                self.door_stop_torque_ep_max[env_ids] = 0.0
                self.door_pull_force_ep_max[env_ids] = 0.0
                self._door_break_count[env_ids] = 0
            for nm in ("hinge", "knob"):
                st = self.__dict__.get("_elastic_state", {})
                if nm in st:
                    st[nm][env_ids] = 0.0
            if c.latch_hint_obs:
                self._latch_hint_resample(env_ids)
            for name in ("_latch_bonus_paid",):
                if hasattr(self, name):
                    getattr(self, name)[env_ids] = False
            if hasattr(self, "_latch_phi_prev"):
                self._latch_phi_prev[env_ids] = 0.0
            return
        self.extras["latch_open_frac"] = self.latch_open.float().mean()
        self.extras["latch_cbf_over_max_deg"] = torch.rad2deg(self._cbf_over_max).max()

    def _apply_latch_pop(self, pop: torch.Tensor, door: torch.Tensor) -> None:
        """Felt cue: lift the door to latch_pop_deg (bang-bang torque about the hinge) where pop is set."""
        c = self.cfg.precise_assembly
        close_tau = float(getattr(c, "latch_door_close_torque", 0.0))
        if float(c.latch_pop_deg) <= 0.0:
            if close_tau <= 0.0:
                return
            # door closer: constant torque about the hinge pushing the door shut (theta-independent physics)
            jd = self._pih_joints[int(c.latch_door_joint)]
            body = self._joint_bodies()[int(c.latch_door_joint)]
            axis_w = quat_apply(self.hole_quat_wxyz, torch.tensor(jd.axis, device=self.device,
                                                                  dtype=torch.float32).expand(self.num_envs, 3))
            torque = (-close_tau * axis_w).unsqueeze(1)
            body.set_external_force_and_torque(torch.zeros_like(torque), torque, is_global=True)
            return
        jd = self._pih_joints[int(c.latch_door_joint)]
        body = self._joint_bodies()[int(c.latch_door_joint)]
        axis_w = quat_apply(self.hole_quat_wxyz, torch.tensor(jd.axis, device=self.device,
                                                              dtype=torch.float32).expand(self.num_envs, 3))
        if float(c.latch_pop_kp) > 0.0:
            if not hasattr(self, "_latch_prev_door"):
                self._latch_prev_door = door.clone()
            wd = (door - self._latch_prev_door) / float(self.step_dt)
            self._latch_prev_door = door.clone()
            tau = (float(c.latch_pop_torque) + float(c.latch_pop_kp) * (math.radians(float(c.latch_pop_deg)) - door)
                   - float(c.latch_pop_kd) * wd).clamp(min=0.0) * pop.float()
        else:
            on = pop & (door < math.radians(float(c.latch_pop_deg)))
            tau = on.float() * float(c.latch_pop_torque)
        torque = (tau.unsqueeze(-1) * axis_w).unsqueeze(1)
        body.set_external_force_and_torque(torch.zeros_like(torque), torque, is_global=True)

    def _update_latch(self, env_ids: torch.Tensor | None = None) -> None:
        """Door latch: keeper at its problem pose (holding the door) until the knob reaches theta - unlock_tol, then
        moved by latch_retract_m; it re-engages only once the door is closed (< latch_reengage_deg) and the knob is
        below theta again. Resets (env_ids) force it engaged. Only envs whose state changes are written."""
        if not self._latch_on():
            return
        if self._latch_cbf_on():
            self._update_latch_cbf(env_ids)
            return
        c = self.cfg.precise_assembly
        if not hasattr(self, "latch_open"):
            self.latch_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.latch_open_ep = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            kp = self._pih_distractor_pose_rel[int(c.latch_keeper)]
            self._latch_pose_rel = (tuple(kp), tuple([kp[0] + float(c.latch_retract_m[0]), kp[1] + float(c.latch_retract_m[1]),
                                                       kp[2] + float(c.latch_retract_m[2]), *kp[3:7]]))
        keeper = self.distractors[int(c.latch_keeper)]
        win = float(c.latch_window_deg) > 0.0
        if not hasattr(self, "latch_dwell"):
            self.latch_dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.latch_in_window = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_ids is not None:
            self.latch_open[env_ids] = False
            self.latch_open_ep[env_ids] = False
            if hasattr(self, "latch_force_count"):
                self.latch_force_count[env_ids] = 0
            if hasattr(self, "latch_check_count"):
                self.latch_check_count[env_ids] = 0
                self._latch_quiet[env_ids] = 0
                self._latch_prev_loaded[env_ids] = False
            if hasattr(self, "_latch_armed"):
                self._latch_armed[env_ids] = False
            if hasattr(self, "latch_overturned"):
                self.latch_overturned[env_ids] = False
            if hasattr(self, "latch_false_stop"):
                self.latch_false_stop[env_ids] = False
                self.latch_false_count[env_ids] = 0
                if hasattr(self, "latch_false_used"):
                    self.latch_false_used[env_ids] = 0
            if hasattr(self, "_latch_phi_prev"):
                self._latch_phi_prev[env_ids] = 0.0
            if hasattr(self, "_soft_stop_hits"):
                self._soft_stop_hits[env_ids] = 0.0
            self.latch_dwell[env_ids] = 0
            self.latch_in_window[env_ids] = False
            if c.latch_hint_obs:
                self._latch_hint_resample(env_ids)
            if hasattr(self, "_knob_cap_prev"):
                self._knob_cap_prev[env_ids] = float("nan")
                self._knob_cap_hits[env_ids] = 0.0
            if hasattr(self, "_latch_bonus_paid"):
                self._latch_bonus_paid[env_ids] = False
            if hasattr(self, "_latch_prev_knob"):
                self._latch_prev_knob[env_ids] = 0.0
            keeper.write_root_pose_to_sim(self._pose_rel_hole_to_world(env_ids, self._latch_pose_rel[0]), env_ids=env_ids)
            return
        self._update_joint_angles()
        knob = self.cb_joint_angle[int(c.unlock_joint)]
        door = self.cb_joint_angle[int(c.latch_door_joint)]
        closed = door < math.radians(float(c.latch_reengage_deg))
        if win:
            w_deg, d_steps = float(c.latch_window_deg), int(c.latch_dwell_steps)
            if int(c.latch_curriculum_steps) > 0:
                a = min(1.0, float(self.common_step_counter + int(c.latch_curriculum_offset_steps))
                        / float(c.latch_curriculum_steps))
                w_deg = float(c.latch_window_start_deg) + a * (w_deg - float(c.latch_window_start_deg))
                d_steps = int(round(int(c.latch_dwell_start_steps) + a * (d_steps - int(c.latch_dwell_start_steps))))
            self.extras["latch_window_deg"] = torch.tensor(w_deg, device=self.device)
            self._latch_dwell_now = d_steps
            # lower edge always the final window (so a wide early window never contains the knob's rest angle);
            # only the over-turn side follows the curriculum
            d = knob - self.unlock_angle
            in_win = (d >= -math.radians(float(c.latch_window_deg))) & (d <= math.radians(w_deg))
            rest = torch.where(in_win, torch.full_like(door, math.radians(float(c.latch_pop_deg))), torch.zeros_like(door))
            loaded = door > rest + math.radians(float(c.latch_load_margin_deg))
            if float(c.latch_still_deg_s) > 0.0:
                if not hasattr(self, "_latch_prev_knob"):
                    self._latch_prev_knob = knob.clone()
                speed = (knob - self._latch_prev_knob).abs() / float(self.step_dt)
                self._latch_prev_knob = knob.clone()
                door_loaded = loaded.clone()
                moving = speed > math.radians(float(c.latch_still_deg_s))
                if int(c.latch_check_pull_fail) > 0:
                    if not hasattr(self, "latch_check_count"):
                        self.latch_check_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                        self._latch_quiet = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                        self._latch_prev_loaded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    # a pull = door up to the keeper (rest + latch_force_deg), like the tug count; light loads are free
                    door_loaded = door >= rest + math.radians(float(c.latch_force_deg))
                    onset = door_loaded & ~self._latch_prev_loaded & ~self.latch_open
                    if getattr(c, "latch_check_quiet_light", False):
                        # armed once a dwell-eligible run completes; stays armed (through light loads) until the knob moves
                        if not hasattr(self, "_latch_armed"):
                            self._latch_armed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        chk = onset & (self._latch_armed | (self._latch_quiet >= int(c.latch_dwell_steps)))
                    else:
                        chk = onset & (self._latch_quiet >= int(c.latch_dwell_steps))
                    self.latch_check_count += chk.long()
                    # quiet = a step that would advance the latch dwell (still and below the light-load margin);
                    # with latch_check_quiet_light a lightly loaded hold is not quiet: no pull after it can be informative
                    q_ok = ~moving & (~loaded if getattr(c, "latch_check_quiet_light", False) else ~door_loaded)
                    self._latch_quiet = torch.where(q_ok, self._latch_quiet + 1,
                                                    torch.zeros_like(self._latch_quiet))
                    if getattr(c, "latch_check_quiet_light", False):
                        self._latch_armed = (self._latch_armed | (self._latch_quiet >= int(c.latch_dwell_steps))) & ~moving
                    self._latch_prev_loaded = door_loaded
                    self.extras["latch_check_count_mean"] = self.latch_check_count.float().mean()
                if c.latch_relock_on_motion:
                    self._latch_motion_relock = moving & (door < math.radians(float(c.latch_reengage_deg)))
                loaded = loaded | moving
                if c.latch_false_stop:
                    # false stop: knob held still outside the window (past latch_false_stop_min_deg) for the dwell time
                    if not hasattr(self, "latch_false_count"):
                        self.latch_false_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                        self.latch_false_stop = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    still = speed <= math.radians(float(c.latch_still_deg_s))
                    off = still & ~in_win & ~self.latch_open & (knob > math.radians(float(c.latch_false_stop_min_deg)))
                    z_deg = float(getattr(c, "latch_false_stop_zone_deg", 0.0))
                    if z_deg > 0.0:
                        # tolerated approach zone: a stop within z_deg below the window's lower edge is not a false stop
                        lo_edge = -math.radians(float(c.latch_window_deg))
                        off = off & ~((d >= lo_edge - math.radians(z_deg)) & (d < lo_edge))
                    self.latch_false_count = torch.where(off, self.latch_false_count + 1, torch.zeros_like(self.latch_false_count))
                    ev = self.latch_false_count >= d_steps
                    if not hasattr(self, "latch_false_used"):
                        self.latch_false_used = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                    # where false stops happen (signed knob - theta, deg): below vs past the window
                    if bool(ev.any()):
                        dd = torch.rad2deg(d[ev])
                        self.extras["latch_false_stop_d_mean"] = dd.mean()
                        self.extras["latch_false_stop_below_frac"] = (dd < 0).float().mean()
                    # tolerated false stops: counted, penalised once (latch_false_stop_free_penalty), counter reset
                    free = ev & (self.latch_false_used < int(c.latch_false_stop_free))
                    self.latch_false_event = free
                    self.latch_false_used = self.latch_false_used + free.long()
                    self.latch_false_count = torch.where(free, torch.zeros_like(self.latch_false_count), self.latch_false_count)
                    self.latch_false_stop |= ev & ~free
            self.latch_dwell_adv = in_win & ~loaded & ~self.latch_open
            if float(c.latch_overturn_fail_deg) >= 0.0:
                if not hasattr(self, "latch_overturned"):
                    self.latch_overturned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self.latch_overturned |= (~self.latch_open) & (d > math.radians(w_deg + float(c.latch_overturn_fail_deg)))
            self.latch_dwell = torch.where(self.latch_dwell_adv, self.latch_dwell + 1,
                                           torch.zeros_like(self.latch_dwell))
            unlocked = self.latch_dwell >= d_steps
            self.latch_in_window = in_win
            r_deg = float(getattr(c, "latch_tactile_ramp_deg", 0.0))
            if r_deg > 0.0:
                # graded touch-only approach cue: 0 at r_deg below the window, rising linearly to 1 at its lower edge,
                # 1 inside the window, 0 past it
                lo = d + math.radians(float(c.latch_window_deg))
                frac = ((lo + math.radians(r_deg)) / math.radians(r_deg)).clamp(0.0, 1.0)
                self.latch_cue_frac = torch.where(d > math.radians(w_deg), torch.zeros_like(frac), frac)
            else:
                self.latch_cue_frac = in_win.float()
            keep_win = in_win
            if float(c.latch_relock_over_deg) > 0.0:
                keep_win = (d >= -math.radians(float(c.latch_window_deg))) & (
                    d <= math.radians(w_deg + float(c.latch_relock_over_deg)))
            new = torch.where(self.latch_open, ~(closed & ~keep_win), unlocked)
            if c.latch_relock_on_motion and hasattr(self, "_latch_motion_relock"):
                new = new & ~(self.latch_open & self._latch_motion_relock)
            self._apply_latch_pop(in_win & ~new, door)
        else:
            unlocked = knob >= self.unlock_angle - math.radians(float(c.unlock_tol_deg))
            new = torch.where(self.latch_open, ~(closed & ~unlocked), unlocked)
        changed = new != self.latch_open
        self.latch_open = new
        self.latch_open_ep |= new
        if changed.any():
            for want, pose in ((True, self._latch_pose_rel[1]), (False, self._latch_pose_rel[0])):
                ids = (changed & (new == want)).nonzero(as_tuple=False).squeeze(-1)
                if ids.numel():
                    keeper.write_root_pose_to_sim(self._pose_rel_hole_to_world(ids, pose), env_ids=ids)

    def _apply_object_start_pose(self, env_ids: torch.Tensor) -> None:
        w = self._pose_rel_hole_to_world(env_ids, self._pih_object_start_rel, self._button_off(0, env_ids))
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
                boff = self._button_off(0, tail_ids)
                if boff is not None:
                    insert_pos = insert_pos + boff
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
        if self._latch_on():
            door = self.cb_joint_angle[int(pih_cfg.latch_door_joint)]
            if not hasattr(self, "door_max_angle"):
                self.door_max_angle = torch.zeros(self.num_envs, device=self.device)
                self.door_prev_ep_max = torch.zeros(self.num_envs, device=self.device)
            self.door_max_angle = torch.maximum(self.door_max_angle, door)
            self.extras["door_angle_deg_mean"] = torch.rad2deg(door).mean()
            self.extras["latch_open_frac"] = self.latch_open.float().mean()
            unlocked = door >= math.radians(float(pih_cfg.door_open_deg))   # the credited event = door open
            if pih_cfg.latch_success_requires_open:
                unlocked = unlocked & self.latch_open_ep
            if hasattr(self, "_soft_stop_hits"):
                self.extras["latch_soft_stop_ep_frac"] = (self._soft_stop_hits > 0).float().mean()
            self.extras["door_open_no_latch_frac"] = ((door >= math.radians(float(pih_cfg.door_open_deg)))
                                                      & ~self.latch_open_ep).float().mean()
        consecutive = self.cfg.termination.force_consecutive_near_goal_steps
        if self._latch_cbf_on():
            unlocked = self._latch_cbf_success_now(door)
            consecutive = True
        at_last = (self._successes == self.env_max_goals - 1) & ~self.is_random_goal_env
        steps = update_near_goal_steps(near_goal=unlocked, near_goal_steps=prev_near_steps,
                                       force_consecutive=consecutive)
        self._near_goal = torch.where(at_last, unlocked, self._near_goal)
        self._near_goal_steps = torch.where(at_last, steps, self._near_goal_steps)
        self._is_success = self._near_goal_steps >= self.cfg.termination.success_steps
        self.extras["unlock_angle_deg_mean"] = torch.rad2deg(ang).mean()
        self.extras["unlock_reached_frac"] = self.unlock_reached.float().mean()

    def _latch_cbf_success_now(self, door: torch.Tensor) -> torch.Tensor:
        """latch_cbf success condition this policy step (counted over success_steps CONSECUTIVE steps): door open
        >= door_open_deg after the latch cleared via the knob this episode (latch_open_ep; the software latch keeps the
        door <= latch_cbf_gap_deg before), knob still held (>= latch_cbf_grasp_fingers fingertips within
        latch_cbf_grasp_dist of its centre) and, with latch_cbf_door_still_deg_s > 0, the door nearly still. The episode
        ends on success, so it ends with the door open and held (official: unlocked_by_knob_then_opened,
        finger_grip_sustained, motion_stopped_door_held_open)."""
        c = self.cfg.precise_assembly
        open_ = (door >= math.radians(float(c.door_open_deg))) & self.latch_open_ep
        ok = torch.ones_like(open_)
        if int(c.latch_cbf_grasp_fingers) > 0:
            self._cbf_grasp_ok = self._cbf_fingers_on_knob() >= int(c.latch_cbf_grasp_fingers)
            ok = ok & self._cbf_grasp_ok
        rate = torch.rad2deg((door - self._cbf_prev_door).abs()) / float(self.step_dt)
        self._cbf_prev_door = door.clone()
        if float(c.latch_cbf_door_still_deg_s) > 0.0:
            self._cbf_door_still = rate <= float(c.latch_cbf_door_still_deg_s)
            ok = ok & self._cbf_door_still
        self.extras["latch_cbf_door_open_frac"] = open_.float().mean()
        self.extras["latch_cbf_open_grasped_frac"] = (open_ & self._cbf_grasp_ok).float().mean()
        return open_ & ok

    def _cbf_fingers_on_knob(self) -> torch.Tensor:
        """Fingertips touching the knob: with tactile on, pads with any taxel deformed by the knob mesh in the last
        (clean, pre-override) tactile reading; otherwise fingertips within latch_cbf_grasp_dist of the knob's centre."""
        tac = getattr(self, "_last_tactile", None)
        if tac is not None and tac.shape[0] == self.num_envs:
            return (tac.view(self.num_envs, 5, -1).amax(dim=-1) > 0).sum(dim=-1)
        return (self._curr_fingertip_distances < float(self.cfg.precise_assembly.latch_cbf_grasp_dist)).sum(dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        prev_near_steps = self._near_goal_steps.clone()
        compute_intermediate_values(self)

        pih_cfg = self.cfg.precise_assembly
        if getattr(self, "unlock_angle", None) is not None:
            self._apply_unlock_goal(prev_near_steps)
        if pih_cfg.button_mode:
            self._apply_button_goal(prev_near_steps)
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
        forced = torch.zeros_like(fall)
        if int(pih_cfg.latch_force_fail_steps) > 0 and self._latch_on() and hasattr(self, "latch_open"):
            if not hasattr(self, "latch_force_count"):
                self.latch_force_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            door = self.cb_joint_angle[int(pih_cfg.latch_door_joint)]
            # measured from the latch rest angle: in the window the pop itself holds the door at latch_pop_deg,
            # which must not count as a tug (else a correct dwell is a forced fail once the budget is < dwell steps)
            rest = math.radians(float(pih_cfg.latch_pop_deg)) * getattr(self, "latch_in_window", torch.zeros_like(door, dtype=torch.bool)).float()
            tug = (~self.latch_open) & (door >= rest + math.radians(float(pih_cfg.latch_force_deg)))
            if getattr(pih_cfg, "latch_force_consecutive", False):
                # only an unbroken tug counts: a short check pull that lets go resets the count
                self.latch_force_count = torch.where(tug, self.latch_force_count + 1, torch.zeros_like(self.latch_force_count))
            else:
                self.latch_force_count += tug.long()
            budget = float(pih_cfg.latch_force_fail_steps)
            if int(pih_cfg.latch_force_fail_start_steps) > 0 and int(pih_cfg.latch_curriculum_steps) > 0:
                a = min(1.0, float(self.common_step_counter + int(pih_cfg.latch_curriculum_offset_steps))
                        / float(pih_cfg.latch_curriculum_steps))
                budget = float(pih_cfg.latch_force_fail_start_steps) + a * (budget - float(pih_cfg.latch_force_fail_start_steps))
            self.extras["latch_force_budget"] = torch.tensor(budget, device=self.device)
            forced = (self.latch_force_count >= int(round(budget))) & ~max_successes
            self.extras["latch_force_count_mean"] = self.latch_force_count.float().mean()
        if int(pih_cfg.latch_check_pull_fail) > 0 and self._latch_on() and hasattr(self, "latch_check_count"):
            forced = forced | ((self.latch_check_count >= int(pih_cfg.latch_check_pull_fail)) & ~max_successes)
        overturned = torch.zeros_like(fall)
        if hasattr(self, "latch_overturned"):
            overturned = self.latch_overturned & ~max_successes
            self.extras["latch_overturned_frac"] = self.latch_overturned.float().mean()
        false_stop = torch.zeros_like(fall)
        if hasattr(self, "latch_false_stop"):
            false_stop = self.latch_false_stop & ~max_successes & ~overturned
            self.extras["latch_false_stop_frac"] = self.latch_false_stop.float().mean()
        broken = torch.zeros_like(fall)
        if hasattr(self, "door_stop_torque"):
            # A1 upper bound (door_break_*): readings = max over this policy step's substeps
            lim_t, lim_f = float(pih_cfg.door_break_torque), float(pih_cfg.door_break_pull)
            over = torch.zeros_like(fall)
            if lim_t > 0.0:
                over |= self.door_stop_torque > lim_t
            if lim_f > 0.0:
                over |= self.door_pull_force > lim_f
            self._door_break_count = torch.where(over, self._door_break_count + 1, torch.zeros_like(self._door_break_count))
            broken = (self._door_break_count >= int(pih_cfg.door_break_steps)) & ~max_successes
            self.extras["door_stop_torque_mean"] = self.door_stop_torque.mean()
            self.extras["door_pull_force_mean"] = self.door_pull_force.mean()
            self.door_stop_torque_step, self.door_pull_force_step = self.door_stop_torque.clone(), self.door_pull_force.clone()
            if not hasattr(self, "door_stop_torque_ep_max"):
                self.door_stop_torque_ep_max = torch.zeros_like(self.door_stop_torque)
                self.door_pull_force_ep_max = torch.zeros_like(self.door_pull_force)
                self.door_stop_torque_last_ep_max = torch.zeros_like(self.door_stop_torque)
                self.door_pull_force_last_ep_max = torch.zeros_like(self.door_pull_force)
            self.door_stop_torque_ep_max = torch.maximum(self.door_stop_torque_ep_max, self.door_stop_torque)
            self.door_pull_force_ep_max = torch.maximum(self.door_pull_force_ep_max, self.door_pull_force)
            self.door_stop_torque.zero_(); self.door_pull_torque.zero_(); self.door_pull_force.zero_()
        wrong_key = torch.zeros_like(fall)
        if pih_cfg.button_mode and hasattr(self, "button_wrong"):
            wrong_key = self.button_wrong & ~max_successes
            self.extras["button_wrong_frac"] = self.button_wrong.float().mean()
        terminated = (fall | max_successes | hand_far | dropped | retract_fail | disturbed | forced | overturned
                      | false_stop | broken | wrong_key)
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
            "fall": fall,
            "max_successes": max_successes,
            "retract_fail": retract_fail,
            "hand_far": hand_far,
            "dropped": dropped,
            "disturbed": disturbed,
            "forced": forced,
            "overturned": overturned,
            "false_stop": false_stop,
            "broken": broken,
            "wrong_key": wrong_key,
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

        if pih_cfg.button_mode and hasattr(self, "button_travel"):
            if float(pih_cfg.button_wrong_penalty) > 0.0:
                wpen = float(pih_cfg.button_wrong_penalty) * self._termination_reasons["wrong_key"].float()
                reward = reward - wpen
                self._reward_terms["button_wrong_penalty"] = -wpen
            if float(pih_cfg.button_press_rew_scale) > 0.0:
                # potential on the target's travel up to the actuation point (privileged reward only)
                phi = (self.button_travel[0] / float(pih_cfg.button_actuate_m)).clamp(0.0, 1.0)
                if pih_cfg.button_require_pad:
                    # progress only counts with a pad on the target: phi holds its last pad value otherwise
                    if hasattr(self, "_button_phi_prev"):
                        phi = torch.where(self._button_pad_on_target(), phi, self._button_phi_prev)
                    else:
                        phi = phi * self._button_pad_on_target().float()
                if not hasattr(self, "_button_phi_prev"):
                    self._button_phi_prev = phi.clone()
                pr = float(pih_cfg.button_press_rew_scale) * (phi - self._button_phi_prev) * (~self.retract_phase).float() \
                    if hasattr(self, "retract_phase") else float(pih_cfg.button_press_rew_scale) * (phi - self._button_phi_prev)
                self._button_phi_prev = phi.clone()
                reward = reward + pr
                self._reward_terms["button_press_rew"] = pr
            if float(pih_cfg.button_touch_bonus) > 0.0 and getattr(self, "_tactile", None) is not None \
                    and hasattr(self._tactile, "body_pad_depth"):
                touched = (self._tactile.body_pad_depth > 0).any(dim=-1)                 # (B, N)
                if not hasattr(self, "_button_touched"):
                    self._button_touched = torch.zeros_like(touched)
                new = touched & ~self._button_touched
                self._button_touched |= touched
                tb = float(pih_cfg.button_touch_bonus) * new.float().sum(0)
                reward = reward + tb
                self._reward_terms["button_touch_bonus"] = tb
                self.extras["button_keys_touched"] = self._button_touched.float().sum(0).mean()
            self._reward_terms["total_reward"] = reward

        if float(getattr(pih_cfg, "door_break_penalty", 0.0)) > 0.0 and "broken" in self._termination_reasons:
            bpen = float(pih_cfg.door_break_penalty) * self._termination_reasons["broken"].float()
            reward = reward - bpen
            self._reward_terms["door_break_penalty"] = -bpen
            self._reward_terms["total_reward"] = reward

        if float(pih_cfg.latch_force_fail_penalty) > 0.0 and "forced" in self._termination_reasons:
            fpen = float(pih_cfg.latch_force_fail_penalty) * self._termination_reasons["forced"].float()
            reward = reward - fpen
            self._reward_terms["latch_force_fail_penalty"] = -fpen
            self._reward_terms["total_reward"] = reward

        if float(pih_cfg.latch_overturn_penalty) > 0.0 and "overturned" in self._termination_reasons:
            open_pen = float(pih_cfg.latch_overturn_penalty) * (self._termination_reasons["overturned"]
                                                                | self._termination_reasons["false_stop"]).float()
            reward = reward - open_pen
            self._reward_terms["latch_overturn_penalty"] = -open_pen
            self._reward_terms["total_reward"] = reward

        if float(pih_cfg.latch_false_stop_free_penalty) > 0.0 and hasattr(self, "latch_false_event"):
            fs_pen = float(pih_cfg.latch_false_stop_free_penalty) * self.latch_false_event.float()
            reward = reward - fs_pen
            self._reward_terms["latch_false_free_penalty"] = -fs_pen
            self._reward_terms["total_reward"] = reward

        if self._latch_on() and float(pih_cfg.latch_approach_rew_scale) > 0.0 and hasattr(self, "latch_open"):
            knob = self.cb_joint_angle[int(pih_cfg.unlock_joint)]
            phi = (knob / self.unlock_angle).clamp(0.0, 1.0)
            over_deg = float(getattr(pih_cfg, "latch_approach_over_deg", 0.0))
            if over_deg > 0.0:
                # two-sided: the potential falls past theta (1 per over_deg of over-turn), so overshoot costs reward
                over = (knob - self.unlock_angle).clamp(min=0.0) / math.radians(over_deg)
                phi = phi - over.clamp(max=float(getattr(pih_cfg, "latch_approach_over_cap", 2.0)))
            if not hasattr(self, "_latch_phi_prev"):
                self._latch_phi_prev = phi.clone()
            gate = (~self.latch_open_ep).float()
            if bool(getattr(pih_cfg, "latch_approach_after_open", False)):
                # keep the over-turn part of the potential after the latch opened (until the door is open)
                dj = int(getattr(pih_cfg, "latch_door_joint", -1))
                still = (self.cb_joint_angle[dj] < math.radians(5.0)).float() if dj >= 0 else torch.ones_like(gate)
                gate = gate + (1.0 - gate) * still * ((phi - self._latch_phi_prev) < 0).float()
            ar = float(pih_cfg.latch_approach_rew_scale) * (phi - self._latch_phi_prev) * gate
            self._latch_phi_prev = phi.clone()
            reward = reward + ar
            self._reward_terms["latch_approach_rew"] = ar
            self._reward_terms["total_reward"] = reward

        # Door latch: one-time bonus when the latch clears (privileged reward; the actor never sees theta).
        if self._latch_on() and float(pih_cfg.latch_unlock_bonus) > 0.0 and hasattr(self, "latch_open"):
            if not hasattr(self, "_latch_bonus_paid"):
                self._latch_bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            first = self.latch_open & ~self._latch_bonus_paid
            self._latch_bonus_paid |= self.latch_open
            bonus = float(pih_cfg.latch_unlock_bonus) * first.float()
            reward = reward + bonus
            self._reward_terms["latch_unlock_bonus"] = bonus
            self._reward_terms["total_reward"] = reward
        if self._latch_on() and pih_cfg.latch_gate_task_reward and hasattr(self, "latch_open"):
            closed = (~self.latch_open).float()
            gated = sum(self._reward_terms.get(k, torch.zeros_like(reward)) for k in ("lifting_rew", "lift_bonus_rew", "keypoint_rew"))
            reward = reward - gated * closed
            for k in ("lifting_rew", "lift_bonus_rew", "keypoint_rew"):
                if k in self._reward_terms:
                    self._reward_terms[k] = self._reward_terms[k] * (1.0 - closed)
            self._reward_terms["total_reward"] = reward
        if self._latch_on() and float(pih_cfg.latch_window_rew_scale) > 0.0 and hasattr(self, "latch_in_window"):
            on = getattr(self, "latch_dwell_adv", self.latch_in_window) if pih_cfg.latch_window_rew_dwell_only else self.latch_in_window
            wr = float(pih_cfg.latch_window_rew_scale) * (on & ~self.latch_open).float()
            reward = reward + wr
            self._reward_terms["latch_window_rew"] = wr
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
