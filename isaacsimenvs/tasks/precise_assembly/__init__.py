"""Precise-assembly task registration.

Registers ``Isaacsimenvs-PreciseAssembly-Direct-v0`` with the gymnasium
registry. The specific assembly task is selected at runtime via
``env.precise_assembly.problem=<KEY>`` (one of the four keys in
``evaluation.problems.PROBLEM_REGISTRY``).

Entry points:
- ``env_cfg_entry_point``           → PreciseAssemblyEnvCfg (typed defaults in code)
- ``env_cfg_yaml_entry_point``      → cfg/task/PreciseAssembly.yaml overlay
- ``rl_games_cfg_entry_point``      → cfg/train/PreciseAssemblyPPO.yaml (baseline)
- ``rl_games_sapg_cfg_entry_point`` → cfg/train/PreciseAssemblySAPG.yaml (default)
- ``rl_games_student_cfg_entry_point`` → cfg/train/PreciseAssemblyStudent.yaml
  (depth student for DAgger distillation; see isaacsimenvs/distill.py)
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from .precise_assembly_env import PreciseAssemblyEnv
from .precise_assembly_env_cfg import PreciseAssemblyEnvCfg

__all__ = ["PreciseAssemblyEnv", "PreciseAssemblyEnvCfg"]

_CFG_DIR = Path(__file__).resolve().parents[2] / "cfg"

gym.register(
    id="Isaacsimenvs-PreciseAssembly-Direct-v0",
    entry_point="isaacsimenvs.tasks.precise_assembly.precise_assembly_env:PreciseAssemblyEnv",
    order_enforce=False,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg:PreciseAssemblyEnvCfg",
        "env_cfg_yaml_entry_point": str(_CFG_DIR / "task" / "PreciseAssembly.yaml"),
        "rl_games_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblyPPO.yaml"),
        "rl_games_sapg_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblySAPG.yaml"),
        # Depth-vision student for DAgger distillation (isaacsimenvs/distill.py).
        # Not an rl_games training config: only its `network:` block and a few
        # `config:` entries are consumed, by our own loop.
        "rl_games_student_cfg_entry_point": str(_CFG_DIR / "train" / "PreciseAssemblyStudent.yaml"),
    },
)
