import os
from typing import Optional

import numpy as np
import torch
from gym import spaces
from rl_games.torch_runner import Runner, players

from deployment.rl_player_utils import (
    read_cfg,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class RlPlayer:
    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        config_path: str,
        checkpoint_path: Optional[str],
        device: str,
        num_envs: int = 1,
        inference_batch_size: Optional[int] = None,
    ) -> None:
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = device

        # Must create observation and action space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(num_actions,), dtype=np.float32
        )
        self.num_envs = num_envs
        self.inference_batch_size = inference_batch_size or num_envs
        self.set_env_state = lambda *args, **kwargs: None

        self.cfg = read_cfg(config_path=config_path, device=self.device)
        # self._run_sanity_checks()
        self.player = self.create_rl_player(checkpoint_path=checkpoint_path)

    def create_rl_player(
        self, checkpoint_path: Optional[str]
    ) -> players.PpoPlayerContinuous:
        from rl_games.common import env_configurations

        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kwargs: self, "vecenv_type": "RLGPU"}
        )

        config = self.cfg["train"]

        # Do we need this?
        if self.device == "cpu":
            try:
                config["params"]["config"]["player"]["device_name"] = "cpu"
            except KeyError:
                config["params"]["config"]["player"] = {"device_name": "cpu"}
            config["params"]["config"]["device"] = "cpu"

        if checkpoint_path is not None:
            config["load_path"] = checkpoint_path
        runner = Runner()
        runner.load(config)

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        player = runner.create_player()
        player.init_rnn()
        player.has_batch_dimension = True
        if checkpoint_path is not None:
            player.restore(checkpoint_path)
        return player

    def _run_sanity_checks(self) -> None:
        cfg_num_observations = self.cfg["task"]["env"]["numObservations"]
        cfg_num_actions = self.cfg["task"]["env"]["numActions"]

        if cfg_num_observations != self.num_observations and cfg_num_observations > 0:
            print(
                f"WARNING: num_observations in config ({cfg_num_observations}) does not match num_observations passed to RlPlayer ({self.num_observations})"
            )
        if cfg_num_actions != self.num_actions and cfg_num_actions > 0:
            print(
                f"WARNING: num_actions in config ({cfg_num_actions}) does not match num_actions passed to RlPlayer ({self.num_actions})"
            )

    def get_normalized_action(
        self, obs: torch.Tensor, deterministic_actions: bool = True
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))

        # SAPG HACK: Need to idx to end of observation
        obs = torch.cat(
            [obs, 50.0 + torch.zeros((batch_size, 1), device=self.device)], dim=1
        )

        if batch_size <= self.inference_batch_size:
            normalized_action = self.player.get_action(
                obs=obs, is_deterministic=deterministic_actions
            )
        else:
            normalized_action = self._get_action_chunked(
                obs=obs,
                deterministic_actions=deterministic_actions,
                chunk_size=self.inference_batch_size,
            )

        # DEBUG:
        DEBUGGING = False
        if DEBUGGING:
            zero_obs = torch.zeros_like(obs)
            zero_action = self.player.get_action(
                obs=zero_obs, is_deterministic=True, use_default_rnn_states=True
            )
            print(f"zero_obs ({zero_obs.shape}): {zero_obs}")
            print(f"zero_action ({zero_action.shape}): {zero_action}")
            breakpoint()

        normalized_action = normalized_action.reshape(-1, self.num_actions)
        assert_equals(normalized_action.shape, (batch_size, self.num_actions))
        return normalized_action

    def _slice_states(self, states, sl: slice):
        if states is None:
            return None
        return tuple(state[:, sl, :].contiguous() for state in states)

    def _concat_states(self, state_chunks):
        if not state_chunks:
            return None
        num_tensors = len(state_chunks[0])
        return tuple(torch.cat([chunk[i] for chunk in state_chunks], dim=1) for i in range(num_tensors))

    def _get_action_chunked(
        self,
        obs: torch.Tensor,
        deterministic_actions: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        player = self.player
        obs = player._preproc_obs(obs)
        prev_states = player.states
        actions = []
        next_state_chunks = []

        for start in range(0, obs.shape[0], chunk_size):
            end = min(start + chunk_size, obs.shape[0])
            sl = slice(start, end)
            chunk_states = self._slice_states(prev_states, sl)
            input_dict = {
                "is_train": False,
                "prev_actions": None,
                "obs": obs[sl],
                "rnn_states": chunk_states,
            }
            with torch.no_grad():
                res_dict = player.model(input_dict)
            chunk_action = res_dict["mus"] if deterministic_actions else res_dict["actions"]
            if player.clip_actions:
                chunk_action = players.rescale_actions(
                    player.actions_low, player.actions_high, torch.clamp(chunk_action, -1.0, 1.0)
                )
            actions.append(chunk_action)
            next_state_chunks.append(tuple(res_dict["rnn_states"]))

        player.states = self._concat_states(next_state_chunks)
        return torch.cat(actions, dim=0)

    def reset(self) -> None:
        self.player.reset()


def main() -> None:
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    CONFIG_PATH = Path("pretrained_policy/config.yaml")
    CHECKPOINT_PATH = Path("pretrained_policy/model.pth")
    NUM_OBSERVATIONS = 140
    NUM_ACTIONS = 29

    player = RlPlayer(
        num_observations=NUM_OBSERVATIONS,
        num_actions=NUM_ACTIONS,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    batch_size = 1
    obs = torch.zeros(batch_size, NUM_OBSERVATIONS).to(device)
    normalized_action = player.get_normalized_action(
        obs=obs, deterministic_actions=True
    )  # Careful about deterministic_actions=True here!
    print(f"Using player with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH}")
    print(f"And num_observations: {NUM_OBSERVATIONS} and num_actions: {NUM_ACTIONS}")
    print(f"Sampled obs: {obs} with shape: {obs.shape}")
    print(
        f"Got normalized_action: {normalized_action} with shape: {normalized_action.shape}"
    )
    print(f"player: {player.player.model}")


if __name__ == "__main__":
    main()
