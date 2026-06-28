"""Play task registration.

Registers ``Isaacsimenvs-Play-Direct-v0`` with the gymnasium registry
for the DirectRLEnv training path.

Entry points:
- ``env_cfg_entry_point``           → PlayEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/Play.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PlayPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PlaySAPG.yaml (default)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .play_env import PlayEnv
from .play_env_cfg import PlayEnvCfg

__all__ = ["PlayEnv", "PlayEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-Play-Direct-v0",
    entry_point="isaacsimenvs.tasks.play.play_env:PlayEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.play.play_env_cfg:PlayEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "Play.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PlayPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PlaySAPG.yaml"),
    },
)
