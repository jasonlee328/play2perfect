"""Phase 2, part 3: bit-level equivalence with the proven eval path.

``check_phase2_teacher.py`` shows our teacher *scores* like eval_offline's
(96.9% vs 96.9%). That is a statistical argument at n~880: a bug worth ~0.5 pp
would hide inside the noise. This is the stronger claim -- run both code paths
side by side in one process on the identical observation sequence and require
the outputs to match exactly.

  reference path:  player.get_action(obs141)          <- eval_offline.py's path
  our path:        Teacher.act(obs141[:, :140])       <- isaacsimenvs/distillation

``block_id="ramp"`` is used here specifically because that is what ``BasePlayer``
appends, so the two paths are supposed to be identical. (Distillation runs the
constant instead -- see the module docstring in teacher.py for why, and the
success-rate gate for what it costs.) Passing this means the difference between
constant and ramp is a deliberate choice, not an artifact of our plumbing.

Also checks the LSTM states stay in lockstep, which catches the failure mode
where mus agree for a few steps and then drift.

    python tools/distillation/check_phase2_equivalence.py --num-envs 64 --steps 300
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
AGENT = "rl_games_sapg_cfg_entry_point"

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=64)
cli.add_argument("--steps", type=int, default=300)
cli.add_argument("--checkpoint", default="pretrained_assembly/tight_insertion/model.pth")
cli.add_argument("--device", default="cuda:0")
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = False
app = AppLauncher(_a).app

import torch  # noqa: E402

import isaacsimenvs  # noqa: F401,E402
from evaluation.eval_isaacsim import (  # noqa: E402
    _apply_env_overrides,
    _configure_agent,
    _instantiate_env,
    _load_env_cfg,
)
from isaacsimenvs.distillation import Teacher  # noqa: E402
from isaacsimenvs.utils.rlgames_utils import register_rlgames_env  # noqa: E402
from rl_games.torch_runner import Runner, _load_checkpoint_weights  # noqa: E402

ckpt = Path(args.checkpoint)
if not ckpt.is_absolute():
    ckpt = REPO / ckpt
if not ckpt.is_file():
    raise SystemExit(f"checkpoint not found: {ckpt}")

cfg = _load_env_cfg(TASK)
_apply_env_overrides(
    cfg,
    problem="tight_insertion",
    goal_mode="preInsertAndFinal",
    random_goal_fraction=0.0,
    insertion_success_tolerance=0.01,
    retract_success_tolerance=0.005,
    num_envs=args.num_envs,
    sim_device=args.device,
    sdf=False,
    keep_dr=False,
    extra_overrides={},
)
env = _instantiate_env(TASK, cfg)
N = env.num_envs

agent_cfg = _configure_agent(
    TASK, AGENT, rl_device=args.device, num_envs=N,
    deterministic=True, games=10**9, extra_overrides={},
)
clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))

# --- reference: exactly eval_offline.py's construction ---------------------
wrapped = register_rlgames_env(
    env, rl_device=args.device, clip_obs=clip_obs, clip_actions=clip_actions
)
runner = Runner()
runner.load(agent_cfg)
runner.reset()
player = runner.create_player()
player.set_weights(_load_checkpoint_weights(player, str(ckpt)))
player.has_batch_dimension = True
player.reset()

# --- ours ------------------------------------------------------------------
teacher = Teacher(
    agent_cfg, str(ckpt),
    num_envs=N,
    teacher_obs_dim=int(env.cfg.observation_space),
    critic_dim=int(env.cfg.state_space),
    action_dim=int(env.cfg.action_space),
    device=args.device,
    block_id="ramp",  # match BasePlayer's column so the paths must agree
)

print(f"\n[setup] envs={N} steps={args.steps} clip_obs={clip_obs} clip_actions={clip_actions}")
print(f"[setup] reference block column: linspace(50, 0, {N}) via BasePlayer")

obs141 = player.env_reset(wrapped)
print(f"[setup] player obs width = {tuple(obs141.shape)} (140 + 1 block-id column)")

# The block column BasePlayer appended must equal ours, or nothing below means
# anything.
ref_col = obs141[:, -1].detach().cpu()
our_col = teacher._block_col.reshape(-1).detach().cpu()
col_match = torch.equal(ref_col, our_col)
print(f"[setup] block column identical: {col_match} "
      f"(max diff {float((ref_col - our_col).abs().max()):.3e})")

max_action_diff = 0.0
max_state_diff = 0.0
mismatch_steps = 0
exact_steps = 0

for step in range(args.steps):
    obs140 = obs141[:, :140]

    ours = teacher.act(obs140)
    ref_action = player.get_action(obs141, is_deterministic=True)

    d = float((ours["action"] - ref_action).abs().max())
    max_action_diff = max(max_action_diff, d)
    if d == 0.0:
        exact_steps += 1
    else:
        mismatch_steps += 1

    # LSTM states must stay in lockstep, else mus agree now and drift later.
    for a, b in zip(player.states, teacher.player.states):
        max_state_diff = max(max_state_diff, float((a - b).abs().max()))

    obs141, _rew, dones, _infos = player.env_step(wrapped, ref_action)
    # Neither path resets hidden state here: eval_offline does not, and this
    # check is about equivalence, not about correct DAgger bookkeeping.

print("\n" + "=" * 68)
print(f"steps compared            : {args.steps}")
print(f"bit-exact action steps    : {exact_steps}")
print(f"steps with any difference : {mismatch_steps}")
print(f"max |action_ours - action_ref| : {max_action_diff:.3e}")
print(f"max |lstm_ours  - lstm_ref|    : {max_state_diff:.3e}")
print("=" * 68)

# float32 determinism: identical ops on identical inputs should be bit-exact.
# A tiny tolerance is allowed only for the state, in case a fused kernel is
# chosen differently between the two module instances.
passed = col_match and max_action_diff == 0.0 and max_state_diff <= 1e-6
if passed:
    print("EQUIVALENCE PASSED: our teacher path is indistinguishable from")
    print("eval_offline's player.get_action() on the same observations.")
    print("The 96.9% gate is therefore a property of the teacher, not of our plumbing.")
else:
    print("EQUIVALENCE FAILED:")
    if not col_match:
        print("  - the appended block-id column does not match BasePlayer's")
    if max_action_diff != 0.0:
        print(f"  - actions diverge (max {max_action_diff:.3e})")
    if max_state_diff > 1e-6:
        print(f"  - LSTM states diverge (max {max_state_diff:.3e})")

print(f"\nRESULT_GATE {'PASS' if passed else 'FAIL'}")
sys.stdout.flush()
sys.stderr.flush()
env.close()
app.close()
os._exit(0 if passed else 1)
