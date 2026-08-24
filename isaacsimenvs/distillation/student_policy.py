"""Inference-only wrapper around a distilled depth student.

Loads a ``Dagger`` checkpoint and turns ``get_student_obs()`` into actions. No
teacher, no optimizer, no aux targets -- just the forward pass, so this is also
the shape a real deployment would take.

Deliberately reads the student obs via ``env.get_student_obs()`` rather than the
Phase 1 obs contract, so it works with ``student_obs.emit_in_observations=False``
-- which is what lets the env stay wrappable by ``RlGamesVecEnvWrapper`` and
therefore drivable from ``evaluation/eval_isaacsim.py``'s viser worker.
"""

from __future__ import annotations

import torch


class StudentPolicy:
    """Frozen depth student. ``act(env)`` -> actions in the env's action space."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        num_envs: int,
        action_dim: int,
        proprio_dim: int,
        device: str = "cuda:0",
        agent_cfg: dict | None = None,
        spatial_pool: str | None = None,
    ) -> None:
        from rl_games.algos_torch.model_builder import ModelBuilder

        from isaacsimenvs.distillation.a2c_aux_cnn import register_student_networks

        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)

        ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        # Newer checkpoints embed the config that produced them; older ones need
        # it supplied, and getting spatial_pool wrong is a shape mismatch.
        cfg = ck.get("agent_cfg") or agent_cfg
        if cfg is None:
            raise ValueError(
                "checkpoint has no embedded agent_cfg; pass agent_cfg= explicitly"
            )
        if spatial_pool is not None:
            cfg["params"]["network"]["student_image"]["spatial_pool"] = spatial_pool
        self.agent_cfg = cfg
        self.spatial_pool = cfg["params"]["network"]["student_image"]["spatial_pool"]

        register_student_networks()
        run_cfg = cfg["params"]["config"]
        self.model = (
            ModelBuilder()
            .load(cfg["params"])
            .build({
                "actions_num": self.action_dim,
                "input_shape": (int(proprio_dim),),
                "num_seqs": self.num_envs,
                "value_size": 1,
                "normalize_value": bool(run_cfg.get("normalize_value", True)),
                "normalize_input": bool(run_cfg.get("normalize_input", True)),
            })
            .to(self.device)
        )
        missing, unexpected = self.model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"[student] state_dict missing={list(missing)[:4]} "
                  f"unexpected={list(unexpected)[:4]}")
        # eval() freezes both RunningMeanStd normalizers -- the proprio one and
        # the per-pixel image one. In train() mode they would keep adapting to
        # whatever we happen to show the policy.
        self.model.eval()
        self.net = self.model.a2c_network
        self.iter = int(ck.get("iter", -1))
        self.grad_steps = int(ck.get("grad_steps", -1))

        self.reset()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the LSTM state, for all envs or just ``env_ids``."""
        if env_ids is None:
            self._states = [
                s.to(self.device) for s in self.net.get_default_rnn_state()
            ]
            self._prev_actions = torch.zeros(
                self.num_envs, self.action_dim, device=self.device
            )
            return
        if env_ids.dtype == torch.bool:
            env_ids = torch.nonzero(env_ids, as_tuple=False).reshape(-1)
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(self._states[0].device)
        for s in self._states:
            s[:, ids] = 0.0

    @torch.no_grad()
    def act(self, env) -> torch.Tensor:
        """One deterministic step from the env's current student observation."""
        student = env.get_student_obs()
        res = self.model({
            # is_train=True returns mus without sampling; is_train=False would
            # draw from the distribution, and for a viewer we want the
            # deterministic policy.
            "is_train": True,
            "prev_actions": self._prev_actions,
            "obs": student["proprio"],
            "img": student["image"],
            "rnn_states": self._states,
            "seq_length": 1,
            "rnn_masks": None,
        })
        self._states = list(res["rnn_states"])
        action = torch.clamp(res["mus"], -1.0, 1.0).float()
        self._prev_actions = action
        return action

    @torch.no_grad()
    def predicted_hole_pos(self) -> torch.Tensor | None:
        """The aux head's hole estimate from the last ``act()``.

        Useful for a viewer: drawing this next to the true hole shows *why* the
        policy is doing what it is doing.
        """
        return self.net.get_aux_outputs().get("hole_pos")


__all__ = ["StudentPolicy"]
