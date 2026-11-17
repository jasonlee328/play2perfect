"""A11 key task (ContactBench task 27): gym id Isaacsimenvs-KeyLock-Direct-v0 = PreciseAssembly + key_lock group."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"
TASK_ID = "Isaacsimenvs-KeyLock-Direct-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaacsimenvs.tasks.key_lock.key_env:KeyLockEnv",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaacsimenvs.tasks.key_lock.key_env:KeyLockEnvCfg",
            "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PreciseAssembly.yaml"),
            "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblyPPO.yaml"),
            "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblySAPG.yaml"),
        },
    )
