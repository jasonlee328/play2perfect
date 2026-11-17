"""A3 force-window tasks (FoundationTouch): fragile grasp and press-to-force on PreciseAssemblyEnv.

Importing this module registers the problems (``fw_fragile_m<g>``, ``fw_press``) and the gym id
``Isaacsimenvs-ForceWindow-Direct-v0`` (same cfg / agent entry points as PreciseAssembly, cfg class with an extra
``force_window`` group). Nothing in PreciseAssembly changes for other tasks.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from . import problems  # noqa: F401  (registers fw_* objects + problems)

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"
TASK_ID = "Isaacsimenvs-ForceWindow-Direct-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaacsimenvs.tasks.force_window.fw_env:ForceWindowEnv",
        order_enforce=False,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaacsimenvs.tasks.force_window.fw_env:ForceWindowEnvCfg",
            "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PreciseAssembly.yaml"),
            "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblyPPO.yaml"),
            "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblySAPG.yaml"),
        },
    )
