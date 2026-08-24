"""Phase 2 gate: does the DAgger teacher wrapper reproduce the teacher?

The distillation loss regresses the teacher's raw ``mus``. If those are subtly
wrong, every downstream loss is meaningless but will present as a student
convergence problem -- so this must pass before any student code is written.

Reproduces ``eval_offline.py``'s scoring (each env's FIRST episode only, so N
envs are N i.i.d. trials; first episodes shorter than --min-episode-steps are
discarded as unstable inits) but drives the env from ``Teacher.act()["action"]``
instead of ``player.get_action()``. Same env, same checkpoint, same metric --
the only difference is the code path under test.

It sweeps the block-id convention, because that choice is not obvious:

  50.0   constant, an exact member of the sigma table's coef_ids
         (``deployment/rl_player.py:99`` uses this)
  0.0    constant, the other end of the ramp
  ramp   ``linspace(50, 0, num_envs)``, exactly what ``BasePlayer`` builds
         (``player.py:93``) and therefore what eval_offline's 96.9% used

A constant is what distillation wants -- the student cannot observe the block
id, so an env-varying teacher is irreducible label noise -- but only if it
holds the success rate. That is what this measures.

All variants run against the same initial states (the torch RNG is reseeded
before each reset), so the comparison is paired rather than n~900 noise.

    python tools/distillation/check_phase2_teacher.py --num-envs 1024
    python tools/distillation/check_phase2_teacher.py --num-envs 64 --steps 600
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
AGENT = "rl_games_sapg_cfg_entry_point"
# eval_offline measured 96.9% on tight_insertion at 1024 envs.
REFERENCE_SR = 0.969

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=1024)
cli.add_argument("--steps", type=int, default=2500)
cli.add_argument("--min-episode-steps", type=int, default=100)
cli.add_argument("--problem", default="tight_insertion")
cli.add_argument("--checkpoint", default="pretrained_assembly/tight_insertion/model.pth")
cli.add_argument("--device", default="cuda:0")
cli.add_argument(
    "--tolerance", type=float, default=0.03,
    help="Allowed absolute shortfall vs the reference success rate.",
)
cli.add_argument(
    "--block-ids", default="50.0,0.0,ramp",
    help="Comma-separated block-id conventions to sweep.",
)
cli.add_argument("--seed", type=int, default=1234)
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = False  # teacher needs no camera; keeps 1024 envs in memory
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

ckpt = Path(args.checkpoint)
if not ckpt.is_absolute():
    ckpt = REPO / ckpt
if not ckpt.is_file():
    raise SystemExit(f"checkpoint not found: {ckpt}\nRun `python download_checkpoints.py` first.")

# --- env: student_obs OFF ---------------------------------------------------
# `teacher_obs` is literally the same tensor as `obs["policy"]` before the Phase
# 1 remap, so the teacher is indifferent to the student contract -- and leaving
# the camera off is what makes 1024 envs fit at all (1024 + camera OOMs a 32 GB
# card). `check_phase2_student_env.py` covers the camera-on path separately.
cfg = _load_env_cfg(TASK)
_apply_env_overrides(
    cfg,
    problem=args.problem,
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
dev = env.device

agent_cfg = _configure_agent(
    TASK, AGENT, rl_device=args.device, num_envs=N,
    deterministic=True, games=10**9, extra_overrides={},
)

TEACHER_OBS_DIM = int(env.cfg.observation_space)
CRITIC_DIM = int(env.cfg.state_space)
ACTION_DIM = int(env.cfg.action_space)
print(f"\n[dims] teacher_obs={TEACHER_OBS_DIM} critic={CRITIC_DIM} action={ACTION_DIM} envs={N}")


def score(block_id) -> dict:
    """Run one block-id convention and score each env's first episode."""
    teacher = Teacher(
        agent_cfg, str(ckpt),
        num_envs=N, teacher_obs_dim=TEACHER_OBS_DIM, critic_dim=CRITIC_DIM,
        action_dim=ACTION_DIM, device=args.device, block_id=block_id,
    )

    # Same initial states for every variant -> paired comparison.
    torch.manual_seed(args.seed)
    obs_dict, _ = env.reset()
    teacher.reset_states()

    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    first_len = torch.zeros(N, dtype=torch.long, device=dev)
    first_full = torch.zeros(N, dtype=torch.bool, device=dev)
    first_ratio = torch.zeros(N, dtype=torch.float, device=dev)
    first_retract = torch.zeros(N, dtype=torch.bool, device=dev)
    has_retract = hasattr(env, "retract_succeeded")
    nonfinite = 0

    for step in range(args.steps):
        out = teacher.act(obs_dict["policy"])
        if not torch.isfinite(out["mus"]).all():
            nonfinite += 1
        obs_dict, _rew, terminated, truncated, _extras = env.step(out["action"])
        dones = (terminated | truncated).reshape(-1).bool()

        newly = dones & ~done_once
        ids = torch.nonzero(newly, as_tuple=False).reshape(-1)
        if ids.numel():
            succ = env._prev_episode_successes[ids]
            maxg = env.prev_episode_env_max_goals[ids].clamp_min(1)
            first_len[ids] = step + 1
            first_full[ids] = succ >= env.prev_episode_env_max_goals[ids]
            first_ratio[ids] = (succ.float() / maxg.float()).clamp(0.0, 1.0)
            rs = env.extras.get("episode_final", {}).get("retract_success") if has_retract else None
            if rs is not None:
                first_retract[ids] = rs[ids].to(dev) > 0.5
            done_once[ids] = True

        # The env auto-resets on done, so the teacher's hidden state must be
        # cleared for exactly those envs or it carries a finished episode's
        # context into a fresh one. Keyed on `dones`, NOT on `newly` -- envs on
        # their second episode onward still need clearing even though they no
        # longer affect the first-episode score.
        if bool(dones.any()):
            teacher.reset_states(dones)

        if bool(done_once.all()):
            break

    valid = done_once & (first_len >= args.min_episode_steps)
    nv = int(valid.sum().item())
    n_done = int(done_once.sum().item())
    return {
        "block_id": block_id,
        "trials": nv,
        "bad_init": n_done - nv,
        "success_rate": float(first_full[valid].float().mean().item()) if nv else 0.0,
        "goal_ratio": float(first_ratio[valid].mean().item()) if nv else 0.0,
        "retract": float(first_retract[valid].float().mean().item()) if (has_retract and nv) else None,
        "steps_run": step + 1,
        "nonfinite_steps": nonfinite,
    }


variants = []
for tok in args.block_ids.split(","):
    tok = tok.strip()
    variants.append(tok if tok == "ramp" else float(tok))

rows = []
for v in variants:
    print(f"\n===== block_id = {v} =====", flush=True)
    rows.append(score(v))
    print(f"  -> {rows[-1]}", flush=True)

print("\n" + "=" * 92)
print(f"{'block_id':>10s} {'trials':>8s} {'bad_init':>9s} {'success':>9s} "
      f"{'goal_ratio':>11s} {'retract':>9s} {'steps':>7s} {'nonfin':>7s}")
print("-" * 92)
for r in rows:
    retr = "-" if r["retract"] is None else f"{r['retract'] * 100:6.1f}%"
    print(f"{str(r['block_id']):>10s} {r['trials']:>8d} {r['bad_init']:>9d} "
          f"{r['success_rate'] * 100:>8.1f}% {r['goal_ratio'] * 100:>10.1f}% "
          f"{retr:>9s} {r['steps_run']:>7d} {r['nonfinite_steps']:>7d}")
print("=" * 92)
print(f"reference (eval_offline, {args.problem}): {REFERENCE_SR * 100:.1f}%   "
      f"tolerance: {args.tolerance * 100:.1f} pp")

best = max(rows, key=lambda r: r["success_rate"])
gate_ok = best["success_rate"] >= REFERENCE_SR - args.tolerance
any_nonfinite = any(r["nonfinite_steps"] for r in rows)

print()
if any_nonfinite:
    print("GATE FAILED: non-finite mus observed -- the teacher is producing NaN/Inf.")
elif gate_ok:
    print(f"GATE PASSED: block_id={best['block_id']} reaches "
          f"{best['success_rate'] * 100:.1f}% (reference {REFERENCE_SR * 100:.1f}%).")
    print("Distill from this block_id. It is the value Teacher() should default to.")
else:
    print(f"GATE FAILED: best is block_id={best['block_id']} at "
          f"{best['success_rate'] * 100:.1f}%, below "
          f"{(REFERENCE_SR - args.tolerance) * 100:.1f}%.")
    print("Do NOT write the student until this is resolved: the mus are wrong,")
    print("and every downstream loss would be meaningless but look like slow convergence.")

passed = gate_ok and not any_nonfinite
# Machine-readable, because Kit's shutdown eats sys.exit codes -- hence the
# os._exit below (the same reason eval_offline.py ends with os._exit).
print(f"\nRESULT_GATE {'PASS' if passed else 'FAIL'} "
      f"best_block_id={best['block_id']} best_sr={best['success_rate']:.4f}")
sys.stdout.flush()
sys.stderr.flush()
env.close()
app.close()
os._exit(0 if passed else 1)
