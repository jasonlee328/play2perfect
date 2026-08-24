"""Frozen SAPG teacher, exposing raw ``(mus, sigmas)`` for DAgger labeling.

Phase 2 of the depth-distillation port. DEXTRAH builds its teacher with a bare
``network.build()``; that cannot work here. ``PreciseAssemblySAPG.yaml`` sets
``fixed_sigma: coef_cond``, which makes sigma a lookup table needing
``coef_ids`` + ``coef_id_idx`` at build time (``network_builder.py:289-292``)
and reads a block-id column appended to the observation at forward time
(``network_builder.py:410``). ``PpoPlayerContinuous`` wires all of that
automatically (``players.py:33-55``), plus ``running_mean_std`` restoration and
LSTM state, and ``evaluation/eval_offline.py`` already proves that path loads
the released checkpoints.

Two things this module does NOT do, both deliberate:

1. **No rl_games env wrapper.** ``RlGamesVecEnvWrapper`` raises unless the env
   exposes a ``"policy"`` obs key (``isaaclab_rl/rl_games.py:128-130``), and the
   Phase 1 student contract emits ``proprio``/``img`` instead. So the teacher's
   ``env_info`` is built directly from dims + the agent yaml's clip values.
   DEXTRAH's ``Dagger`` reads the raw gym env anyway, so nothing wants the
   wrapper.

2. **No ``player.get_action()``.** That returns ``rescale_actions(clamp(mu))``,
   throwing away the raw ``mus``/``sigmas`` the distillation loss needs. We call
   ``player.model(input_dict)`` directly, the same pattern already written in
   ``deployment/rl_player.py:141-174``.

The block-id column
-------------------
SAPG partitions envs into exploration blocks and appends one block-id column to
the observation. The network *conditions on that value*: it is part of the 141-d
input vector, and it also indexes the per-block sigma table.

``BasePlayer`` builds this column as ``linspace(50, 0, num_envs)``
(``player.py:93``) — a per-env ramp, so env *i* gets a different teacher than
env *j*. That is what ``eval_offline.py`` measured 96.9% with. It is the wrong
choice for distillation: the student cannot observe the block id, so an env-
varying teacher is irreducible label noise. ``deployment/rl_player.py:99`` uses
a constant 50.0 instead, which is also an exact member of the sigma table's
``coef_ids`` (``linspace(50, 0, 6)``) so the lookup hits row 0 by intent rather
than by ``argmax``-on-all-false accident.

Hence ``block_id`` is explicit here and defaults to the constant. Pass
``"ramp"`` to reproduce the player exactly.

Measured by ``tools/distillation/check_phase2_teacher.py`` on tight_insertion,
1024 envs, paired initial states — the constant is not a compromise, it is
strictly the best option:

    block_id   trials   success
    50.0          881     96.9%   <- matches eval_offline's 96.9% to 0.04 pp
    0.0           889     95.4%
    ramp          878     96.6%   <- what eval_offline actually used

So the block-id conditioning does matter (1.5 pp between the two constants);
it is not a free parameter to guess at. Re-run the gate if the checkpoint or
the task changes.

Do not use ``sigmas`` as loss weights
-------------------------------------
``sigma = sigma_act(sigma[idxs])`` (``network_builder.py:411``) indexes only on
the block id — never on the observation. So for a fixed ``block_id`` the sigma
returned here is a *constant 29-vector*, not state-dependent teacher
uncertainty. Measured per block (see ``check_phase2_student_env.py``):

    row  coef_id  sigma_min  sigma_max     1/sigma^2 weight range
      0     50.0     0.1618   170.4446     3.4e-05 .. 3.8e+01
      4     10.0     0.1138     1.1938     7.0e-01 .. 7.7e+01
      5      0.0     0.1147     1.2221     6.7e-01 .. 7.6e+01

coef_id is monotone in exploration magnitude, and row 0 — the block with the
best success rate — spans 1054x in sigma, i.e. 1.1e6x in ``1/sigma^2``.
DEXTRAH's ``1/sigma_T^2``-weighted mu regression would therefore silently zero
the loss on whichever joints that block happens to be noisy on. Phase 5 should
take ``mus`` from block 50.0 and weight uniformly (or borrow row 4/5's sigmas,
which are well-conditioned).
"""

from __future__ import annotations

import copy
import math

import torch

# linspace(50, 0, 6) — the sigma table's block ids (network_builder.py:410
# compares the appended obs column against these for an exact match).
COEF_IDS = (50.0, 40.0, 30.0, 20.0, 10.0, 0.0)
# Validated at 96.9% — see the module docstring's table before changing this.
DEFAULT_BLOCK_ID = 50.0


class _NumEnvsShim:
    """Stands in for ``config['vec_env']``.

    When ``env_info`` is supplied, ``BasePlayer`` skips env creation and sets
    ``self.env = config.get('vec_env')`` (``player.py:22-42``). It then reads
    exactly one thing off it — ``num_envs``, for the block-id column and for
    ``num_seqs`` — so a shim avoids handing rl_games a real env it would
    otherwise try to step.
    """

    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)


def teacher_env_info_from_dims(
    *,
    teacher_obs_dim: int,
    critic_dim: int,
    action_dim: int,
    clip_obs: float,
    clip_actions: float,
) -> dict:
    """Build the teacher's rl_games ``env_info`` without an env wrapper.

    The action space must be finite: ``PpoPlayerContinuous`` reads
    ``action_space.low/high`` into ``rescale_actions``, so an unbounded Box
    (which is what ``unwrapped.single_action_space`` is) silently NaNs every
    action.
    """
    import gymnasium as gym

    return {
        "observation_space": gym.spaces.Box(-clip_obs, clip_obs, (int(teacher_obs_dim),)),
        "state_space": gym.spaces.Box(-clip_obs, clip_obs, (int(critic_dim),)),
        "action_space": gym.spaces.Box(-clip_actions, clip_actions, (int(action_dim),)),
        "agents": 1,
        "value_size": 1,
    }


class Teacher:
    """Frozen state-MLP+LSTM teacher that returns raw ``mus`` / ``sigmas``.

    Args:
        agent_cfg: the rl_games agent config dict (``PreciseAssemblySAPG.yaml``
            as loaded by ``load_cfg_from_registry``). Deep-copied, not mutated.
        checkpoint_path: released ``model.pth``.
        num_envs: must match the env, since the block-id column and the LSTM
            hidden state are both sized from it.
        block_id: constant appended to every env's observation, or ``"ramp"``
            to reproduce ``BasePlayer``'s ``linspace(50, 0, num_envs)``.
    """

    def __init__(
        self,
        agent_cfg: dict,
        checkpoint_path: str,
        *,
        num_envs: int,
        teacher_obs_dim: int,
        critic_dim: int,
        action_dim: int,
        device: str = "cuda:0",
        block_id: float | str = DEFAULT_BLOCK_ID,
    ) -> None:
        from rl_games.torch_runner import Runner, _load_checkpoint_weights

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.teacher_obs_dim = int(teacher_obs_dim)

        cfg = copy.deepcopy(agent_cfg)
        env_params = cfg["params"].get("env", {})
        self.clip_obs = float(env_params.get("clip_observations", math.inf))
        self.clip_actions = float(env_params.get("clip_actions", math.inf))

        run_cfg = cfg["params"]["config"]
        run_cfg["device"] = device
        run_cfg["device_name"] = device
        run_cfg["num_actors"] = self.num_envs
        # Supplying env_info makes BasePlayer skip env creation entirely.
        run_cfg["env_info"] = teacher_env_info_from_dims(
            teacher_obs_dim=teacher_obs_dim,
            critic_dim=critic_dim,
            action_dim=action_dim,
            clip_obs=self.clip_obs,
            clip_actions=self.clip_actions,
        )
        run_cfg["vec_env"] = _NumEnvsShim(self.num_envs)
        player_cfg = run_cfg.setdefault("player", {})
        player_cfg["deterministic"] = True
        player_cfg["print_stats"] = False

        runner = Runner()
        runner.load(cfg)
        runner.reset()
        self.player = runner.create_player()
        self.player.set_weights(
            _load_checkpoint_weights(self.player, str(checkpoint_path))
        )
        self.player.has_batch_dimension = True
        self.player.reset()  # init_rnn(): zeroes the LSTM state

        self.is_rnn = bool(self.player.is_rnn)
        self.block_id = block_id
        self._block_col = self._build_block_column(block_id)

        # actions_low/high come from env_info's action_space, so they are the
        # finite clip bounds rather than +-inf.
        self._actions_low = self.player.actions_low
        self._actions_high = self.player.actions_high

    # -- block-id column ----------------------------------------------------
    def _build_block_column(self, block_id: float | str) -> torch.Tensor:
        if isinstance(block_id, str):
            if block_id != "ramp":
                raise ValueError(
                    f"block_id must be a float or 'ramp', got {block_id!r}"
                )
            # Exactly BasePlayer's construction (player.py:93).
            col = torch.linspace(50.0, 0.0, self.num_envs, device=self.device)
            return col.reshape(-1, 1)
        value = float(block_id)
        if not any(abs(value - c) < 1e-6 for c in COEF_IDS):
            # Not fatal: the sigma lookup argmaxes an all-false comparison and
            # silently falls back to row 0. Worth saying out loud.
            print(
                f"[Teacher] WARNING: block_id={value} is not one of the sigma "
                f"table's coef_ids {COEF_IDS}. The per-block sigma lookup will "
                f"fall through to row 0 via argmax-on-no-match."
            )
        return torch.full((self.num_envs, 1), value, device=self.device)

    # -- inference ----------------------------------------------------------
    def act(self, teacher_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Label one step.

        Args:
            teacher_obs: ``(num_envs, teacher_obs_dim)`` — the env's
                ``"teacher_obs"`` key, i.e. the NOISY ``obs_list`` tensor the
                teacher trained on. Do not pass the clean critic obs.

        Returns:
            ``mus`` / ``sigmas`` — raw network outputs, what the distillation
            loss regresses — and ``action``, the env-ready
            ``rescale(clamp(mu))`` that ``get_action`` would have returned.
        """
        if teacher_obs.shape != (self.num_envs, self.teacher_obs_dim):
            raise ValueError(
                f"expected teacher_obs {(self.num_envs, self.teacher_obs_dim)}, "
                f"got {tuple(teacher_obs.shape)}"
            )

        obs = teacher_obs.to(self.device)
        # Mirror the wrapper: clip the observation, then append the block-id
        # column UNCLIPPED. The wrapper clips in `_process_obs` and BasePlayer
        # appends afterwards (player.py:208), so a block id of 50.0 must
        # survive a clip_observations of 10.0.
        obs = obs.clamp(-self.clip_obs, self.clip_obs)
        obs = torch.cat([obs, self._block_col], dim=1)

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,  # normalize_input is applied inside the model
            "rnn_states": self.player.states,
        }
        with torch.no_grad():
            res = self.player.model(input_dict)
        self.player.states = res["rnn_states"]

        mus, sigmas = res["mus"], res["sigmas"]
        action = torch.clamp(mus, -1.0, 1.0)
        if self.player.clip_actions:
            from rl_games.algos_torch import players

            action = players.rescale_actions(
                self._actions_low, self._actions_high, action
            )
        return {"mus": mus, "sigmas": sigmas, "action": action}

    # -- LSTM bookkeeping ---------------------------------------------------
    def reset_states(self, env_ids: torch.Tensor | None = None) -> None:
        """Zero the LSTM hidden state, for all envs or just ``env_ids``.

        DAgger must call this on episode boundaries. The teacher's hidden state
        is conditioned on the trajectory it has seen; carrying a finished
        episode's state into a fresh one puts the teacher off its own training
        distribution, and the resulting labels are wrong in a way that presents
        as a student convergence problem.

        Note ``eval_offline.py`` does not do this, which is fine there — it
        scores only each env's first episode, so stale state after the first
        ``done`` never affects the number.
        """
        if not self.is_rnn or self.player.states is None:
            return
        if env_ids is None:
            self.player.reset()
            return
        if env_ids.dtype == torch.bool:
            env_ids = torch.nonzero(env_ids, as_tuple=False).reshape(-1)
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(self.player.states[0].device)
        for state in self.player.states:
            # (num_layers, num_seqs, hidden) — dim 1 is the env axis.
            state[:, ids] = 0.0
