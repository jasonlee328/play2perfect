"""rl_games SAPG teacher wrapper for DAgger.

Mirrors the loader pattern in :mod:`peg_in_hole_dynamic.eval_isaacsim` and
:class:`deployment.rl_player.RlPlayer`, but is scoped to "given an already-
``register_rlgames_env``-wrapped env, instantiate a frozen SAPG player and
expose ``get_action(base_obs)``". The SAPG block-id append (one extra obs
channel — see ``rl_games/common/player.py:95-101, 199-258``) is done here so
the caller passes in the *base* env obs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class _NumEnvsStub:
    """Minimal vec_env stand-in for ``BasePlayer``'s SAPG init (reads only
    ``self.env.num_envs`` at ``rl_games/common/player.py:93``). Avoids handing
    a real env into ``runner.set_vec_env`` — which would deepcopy its USD
    Stage references during ``runner.load(...)``.
    """

    def __init__(self, n: int) -> None:
        self.num_envs = int(n)


class Teacher:
    """Frozen SAPG rl_games player that labels env steps with target actions."""

    def __init__(
        self,
        *,
        task_id: str,
        agent_key: str,
        checkpoint_path: str | Path,
        num_envs: int,
        rl_device: str,
        env_info: dict | None = None,
        vec_env=None,
    ) -> None:
        """Load a frozen rl_games SAPG/PPO player.

        Callers build ``env_info`` from the central ``RlGamesVecEnvWrapper``'s
        own properties (``action_space``, ``state_space``, and a teacher view
        of ``observation_space``) — see ``isaacsimenvs.utils.rlgames_utils
        .teacher_env_info(wrapped)``. That wrapper already clips obs/actions
        using the agent yaml's ``clip_observations`` / ``clip_actions``, so
        the bounds are finite and yaml-driven without any manual Box
        construction here.

        Do NOT pass an env_info built from ``unwrapped.single_action_space``:
        Isaac Lab leaves that as ``Box(-inf, inf, ...)`` and rl_games'
        ``rescale_actions`` then evaluates ``(inf + -inf) / 2 = NaN`` on every
        action call, even though the model's ``mu`` is finite.
        """
        # Imports are local so the module stays cheap when only the policy
        # class is needed elsewhere.
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from rl_games.torch_runner import Runner, _load_checkpoint_weights

        agent_cfg = load_cfg_from_registry(task_id, agent_key)
        agent_cfg["params"]["config"]["device"] = rl_device
        agent_cfg["params"]["config"]["device_name"] = rl_device
        agent_cfg["params"]["config"]["num_actors"] = int(num_envs)
        # Return raw μ (unclamped) as the teacher action. Default rl_games
        # behavior (`PpoPlayerContinuous.get_action`, players.py:91-92) is to
        # `clamp(mu, -1, 1)` before returning, which truncates every coord
        # where the teacher's raw μ exceeded ±1. The student then learns to
        # predict the clamped value and never reaches saturation, while the
        # env still clips at execution — net result is the student moves
        # noticeably slower than the teacher. With clip_actions=False the
        # label is the raw μ; env's own clip handles execution.
        agent_cfg["params"]["config"]["clip_actions"] = False

        if env_info is not None:
            agent_cfg["params"]["config"]["env_info"] = env_info
        # Player-side knobs — deterministic actions are what we regress on.
        player_cfg = agent_cfg["params"]["config"].setdefault("player", {})
        player_cfg["deterministic"] = True
        player_cfg["print_stats"] = False
        # games_num is irrelevant for our use (we never call player.run()),
        # but the BasePlayer reads it.
        player_cfg.setdefault("games_num", 1)

        runner = Runner()
        runner.load(agent_cfg)        # deepcopies cfg — DON'T put unpicklable env in cfg before this
        runner.reset()
        # Inject vec_env after load() so its USD Stage refs aren't deepcopied.
        # BasePlayer's SAPG init (player.py:93) reads `self.env.num_envs`, so
        # we need *something* with `num_envs` even though we never call
        # player.run() / env_step.
        if vec_env is None:
            vec_env = _NumEnvsStub(num_envs)
        runner.set_vec_env(vec_env)
        player = runner.create_player()
        weights = _load_checkpoint_weights(player, str(checkpoint_path))
        player.set_weights(weights)
        player.has_batch_dimension = True
        player.reset()

        self.player = player
        self.checkpoint_path = str(checkpoint_path)
        self.action_dim = int(player.actions_num)
        self._device = rl_device

    @property
    def num_envs(self) -> int:
        return int(self.player.env.num_envs) if hasattr(self.player, "env") and self.player.env is not None else int(self.player.config["num_actors"])

    @property
    def intr_reward_coef_embd(self) -> Optional[torch.Tensor]:
        return getattr(self.player, "intr_reward_coef_embd", None)

    @intr_reward_coef_embd.setter
    def intr_reward_coef_embd(self, value: Optional[torch.Tensor]) -> None:
        self.player.intr_reward_coef_embd = value

    def pin_block_id(self, value: float, *, dtype: torch.dtype = torch.float32) -> None:
        """Replace ``intr_reward_coef_embd`` with a constant across all envs.

        The default rl_games ``BasePlayer`` (``player.py:93``) builds
        ``linspace(50, 0, num_envs)``, but the SAPG network's block lookup is
        an exact-equality argmax against the training block-ids
        ``{50, 40, 30, 20, 10, 0}`` — non-matching values fall through to
        block 0 anyway. Pinning makes the active block explicit and lets us
        retune from a single knob.
        """
        cur = self.intr_reward_coef_embd
        if cur is None:
            raise RuntimeError(
                "player has no intr_reward_coef_embd (not an SAPG checkpoint?)"
            )
        self.intr_reward_coef_embd = torch.full_like(cur, float(value), dtype=dtype)

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the teacher's deterministic action for ``obs``.

        ``obs`` is the *base* env obs of shape ``(B, obs_dim)``. We append the
        SAPG block-id channel (one column per env) here so the model sees the
        same input shape it was trained against.
        """
        if self.intr_reward_coef_embd is not None:
            obs = torch.cat([obs, self.intr_reward_coef_embd], dim=1)
        action = self.player.get_action(obs, is_deterministic=True)
        # Some rl_games player paths return a numpy or unsqueezed tensor.
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=obs.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return action

    @torch.no_grad()
    def get_action_and_sigma(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, sigma) of the teacher's policy for ``obs``.

        Like :meth:`get_action` but also exposes the teacher's per-dim sigma,
        needed for NLL/KL distillation. Bypasses ``player.get_action`` so we
        can read ``res_dict['sigmas']`` directly. Updates the player's RNN
        states the same way ``player.get_action`` does, so successive calls
        advance the teacher's RNN in lock-step.
        """
        if self.intr_reward_coef_embd is not None:
            obs = torch.cat([obs, self.intr_reward_coef_embd], dim=1)
        # Teacher.__init__ sets has_batch_dimension=True, so we don't need
        # the unsqueeze path here.
        obs = self.player._preproc_obs(obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.player.states,
        }
        res_dict = self.player.model(input_dict)
        self.player.states = res_dict["rnn_states"]
        mu = res_dict["mus"]
        sigma = res_dict["sigmas"]
        if mu.ndim == 1:
            mu = mu.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
        return mu, sigma

    def reset(self) -> None:
        """Reset the player's RNN state (call when all envs reset)."""
        self.player.reset()

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Zero out the player's RNN state for the given env indices.

        The teacher is recurrent (LSTM-1024 in our SAPG yaml), so envs that
        terminate need their hidden state cleared just like the student.
        """
        if not hasattr(self.player, "states") or self.player.states is None:
            return
        if env_ids.numel() == 0:
            return
        for s in self.player.states:
            s[:, env_ids, :] = 0.0


def latest_checkpoint(run_dir: str | Path) -> Path:
    """Return the most-recent .pth under a teacher run dir.

    rl_games (our isaacsimenvs setup) writes to:
        <run_dir>/0_<exp_name>/nn/<name>.pth          (running snapshot, frequent)
        <run_dir>/0_<exp_name>/nn/last_<name>_ep_*.pth (best so far, named)
        <run_dir>/0_<exp_name>/last/model.pth          (rolling latest, frequent)
        <run_dir>/0_<exp_name>/best/model.pth          (best so far, anonymous)
    Hardware-rollouts copies sometimes have a flat ``model.pth`` directly
    under the run dir. Walk all of these and pick whichever is newest by
    mtime — that is "the latest snapshot" of the live training job.
    """
    run_dir = Path(run_dir)
    candidates: list[Path] = []
    candidates.extend(run_dir.glob("**/last/model.pth"))
    candidates.extend(run_dir.glob("**/best/model.pth"))
    candidates.extend(run_dir.glob("**/nn/*.pth"))
    candidates.extend(run_dir.glob("model.pth"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no .pth checkpoints under {run_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]
