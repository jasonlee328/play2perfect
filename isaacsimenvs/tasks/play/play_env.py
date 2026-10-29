"""Thin DirectRLEnv wrapper for Play.

The env owns Isaac Lab hook wiring and state buffers. Task math lives in the
utility modules called from each hook.
"""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from .play_env_cfg import PlayEnvCfg
from .utils.action_utils import apply_action_pipeline, apply_wrench_dr
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import (
    build_observations,
    build_student_observations,
    compute_intermediate_values,
    OBS_FIELD_SIZES,
    compute_obs_dim,
)
from .utils.hand_skin import HandSkinSensor, hand_skin_obs_dim
from .utils.tactile_utils import TactileSensor, tactile_obs_dim
from .utils.reset_utils import allocate_state_buffers, reset_env_state
from .utils.reward_utils import compute_rewards
from .utils.scene_utils import apply_physx_material_properties, setup_scene
from .utils.termination_utils import compute_terminations, update_tolerance_curriculum


__all__ = ["PlayEnv", "PlayEnvCfg"]


class PlayEnv(DirectRLEnv):
    cfg: PlayEnvCfg

    def __init__(
        self, cfg: PlayEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        if cfg.tactile.enabled:
            # Its size depends on the resolution, so register it before any dim lookup.
            OBS_FIELD_SIZES["tactile"] = tactile_obs_dim(cfg.tactile)
            if cfg.tactile.in_policy and "tactile" not in cfg.obs.obs_list:
                cfg.obs.obs_list = (*cfg.obs.obs_list, "tactile")
            if cfg.tactile.in_critic and "tactile" not in cfg.obs.state_list:
                cfg.obs.state_list = (*cfg.obs.state_list, "tactile")
            if cfg.tactile.whole_hand:
                OBS_FIELD_SIZES["tactile_hand"] = hand_skin_obs_dim(cfg.tactile)
                if cfg.tactile.in_policy and "tactile_hand" not in cfg.obs.obs_list:
                    cfg.obs.obs_list = (*cfg.obs.obs_list, "tactile_hand")
                if cfg.tactile.whole_hand_in_critic and "tactile_hand" not in cfg.obs.state_list:
                    cfg.obs.state_list = (*cfg.obs.state_list, "tactile_hand")
        # Override obs/state space from configured field lists before
        # DirectRLEnv / rl_games observes the configclass.
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list)

        super().__init__(cfg, render_mode, **kwargs)  # runs _setup_scene
        apply_physx_material_properties(self)
        allocate_state_buffers(self)
        self._tactile = TactileSensor(self, cfg.tactile) if cfg.tactile.enabled else None
        self._hand_skin = HandSkinSensor(self, cfg.tactile) if cfg.tactile.enabled and cfg.tactile.whole_hand else None
        self._tactile_hr = self._tactile_encoder = None
        so = getattr(cfg, "student_obs", None)
        if so is not None and so.enabled and str(getattr(so, "tactile_encoder", "none")).lower() != "none":
            import copy

            from .utils.tactile_encoders import TactileFoundationEncoder

            hr_cfg = copy.deepcopy(cfg.tactile)
            hr_cfg.resolution, hr_cfg.samples_per_taxel = int(so.tactile_encoder_resolution), 1
            self._tactile_hr = TactileSensor(self, hr_cfg)
            self._tactile_hr_res = int(so.tactile_encoder_resolution)
            self._tactile_encoder = TactileFoundationEncoder(
                str(so.tactile_encoder).lower(), self.device, size=int(so.tactile_encoder_image_size))
            OBS_FIELD_SIZES["tactile_feat"] = 5 * self._tactile_encoder.dim

    def _setup_scene(self) -> None:
        setup_scene(self)

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(
            self,
            torch.as_tensor(env_ids, device=self.device, dtype=torch.long),
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self) -> None:
        # Called decimation times per policy step; idempotent.
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return build_observations(self)

    def get_student_obs(self) -> dict[str, torch.Tensor]:
        """Return opt-in student observations for distillation code."""
        return build_student_observations(self)
