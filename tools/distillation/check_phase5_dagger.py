"""Phase 5 smoke test: does the DAgger loop run and does the loss go down?

Small and short on purpose -- this is a plumbing check, not a training run. It
cannot tell you the student will reach 2 mm; it tells you the loop is wired,
the gradient step fires at the right cadence, the beta schedule behaves, and the
loss actually descends on a handful of gradient steps.

    python tools/distillation/check_phase5_dagger.py --num-envs 8 --iters 160
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
TEACHER_AGENT = "rl_games_sapg_cfg_entry_point"
STUDENT_CFG = REPO / "isaacsimenvs/cfg/train/PreciseAssemblyStudent.yaml"

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=8)
cli.add_argument("--iters", type=int, default=160, help="env steps")
cli.add_argument("--seq-length", type=int, default=4)
cli.add_argument("--warmup", type=int, default=40, help="beta=1 env steps")
cli.add_argument("--checkpoint", default="pretrained_assembly/tight_insertion/model.pth")
cli.add_argument("--device", default="cuda:0")
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = True
app = AppLauncher(_a).app

import torch  # noqa: E402
import yaml  # noqa: E402

import isaacsimenvs  # noqa: F401,E402
from evaluation.eval_isaacsim import (  # noqa: E402
    _apply_env_overrides,
    _configure_agent,
    _instantiate_env,
    _load_env_cfg,
)
from isaacsimenvs.distillation import Teacher  # noqa: E402
from isaacsimenvs.distillation.dagger import Dagger  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


ckpt = Path(args.checkpoint)
if not ckpt.is_absolute():
    ckpt = REPO / ckpt

cfg = _load_env_cfg(TASK)
_apply_env_overrides(
    cfg, problem="tight_insertion", goal_mode="preInsertAndFinal",
    random_goal_fraction=0.0, insertion_success_tolerance=0.01,
    retract_success_tolerance=0.005, num_envs=args.num_envs,
    sim_device=args.device, sdf=False, keep_dr=False, extra_overrides={},
)
cfg.student_obs.enabled = True          # emit_in_observations stays True: DAgger
env = _instantiate_env(TASK, cfg)       # needs the full student contract
N = env.num_envs

teacher_agent_cfg = _configure_agent(
    TASK, TEACHER_AGENT, rl_device=args.device, num_envs=N,
    deterministic=True, games=10**9, extra_overrides={},
)
teacher = Teacher(
    teacher_agent_cfg, str(ckpt), num_envs=N,
    teacher_obs_dim=int(env.cfg.observation_space),
    critic_dim=int(env.cfg.state_space),
    action_dim=int(env.cfg.action_space),
    device=args.device,
)

student_cfg = yaml.safe_load(STUDENT_CFG.read_text())
student_cfg["params"]["config"].update(
    seq_length=args.seq_length,
    beta_warmup_iters=args.warmup,
    beta_anneal_iters=args.warmup,   # exercise the anneal branch too
    max_iters=args.iters,
)

dagger = Dagger(env, student_cfg, teacher, device=args.device, log_every=20)

print("\n1. construction")
check("student proprio dim inferred as 87", dagger._infer_proprio_dim() == 87,
      f"got {dagger._infer_proprio_dim()}")
check("mu weighting defaults to uniform", dagger.mu_weight_mode == "uniform")
check("aux targets match the env", set(dagger.aux_targets)
      == {"hole_pos", "keypoints_rel_goal", "object_pos"}, f"{dagger.aux_targets}")

print("\n2. beta schedule")
check("beta=1 during warmup", dagger.beta(0) == 1.0 and dagger.beta(args.warmup - 1) == 1.0)
mid = dagger.beta(args.warmup + args.warmup // 2)
check("beta anneals between", 0.0 < mid < 1.0, f"beta(mid)={mid:.3f}")
check("beta=0 after anneal", dagger.beta(args.warmup * 3) == 0.0)

print(f"\n3. running {args.iters} env steps (seq_length={args.seq_length})")
hist = dagger.distill(max_iters=args.iters)

print("\n4. loop bookkeeping")
check("ran the requested env steps", dagger.iter == args.iters, f"got {dagger.iter}")
expected_grads = args.iters // args.seq_length
check("one gradient step per seq_length env steps",
      dagger.grad_steps == expected_grads,
      f"got {dagger.grad_steps}, expected {expected_grads}")
check("history was recorded", len(hist) > 0, f"{len(hist)} records")

print("\n5. learning signal")
first, last = hist[0], hist[-1]
for k in ("total", "mu", "aux_hole_pos"):
    print(f"   {k:16s} {first[k]:12.5f} -> {last[k]:12.5f}")
check("total loss decreased", last["total"] < first["total"],
      f"{first['total']:.4f} -> {last['total']:.4f}")
check("hole_pos aux loss decreased", last["aux_hole_pos"] < first["aux_hole_pos"],
      f"{first['aux_hole_pos']:.5f} -> {last['aux_hole_pos']:.5f}")
check("all logged losses finite",
      all(all(v == v for v in r.values()) for r in hist))
print(f"   hole_pos RMSE: {first['hole_rmse_mm']:.1f}mm -> {last['hole_rmse_mm']:.1f}mm "
      f"(spec is 2 mm; meaningless this early, printed to show the instrument works)")

print("\n6. gradients reached the encoder")
gsum = float(dagger.student_net.feature_extractor.cnn[0].weight.abs().sum())
check("first conv weights are non-zero after training", gsum > 0, f"|W| = {gsum:.4e}")
moved = any(
    float(dagger.optimizer.state[p]["exp_avg"].abs().sum()) > 0
    for p in dagger.student_net.feature_extractor.parameters()
    if p in dagger.optimizer.state
)
check("optimizer holds non-zero momentum for encoder params", moved)

print("\n7. hidden state hygiene")
check("states detached after a gradient step",
      all(not s.requires_grad for s in dagger._states),
      f"{[s.requires_grad for s in dagger._states]}")
check("states are (layers, N, units)",
      all(tuple(s.shape) == (dagger.student_net.rnn_layers, N,
                             dagger.student_net.rnn_units) for s in dagger._states))

print("\n" + "=" * 62)
passed = not FAILURES
print("PHASE 5 SMOKE TEST PASSED" if passed else
      f"PHASE 5 SMOKE TEST FAILED — {len(FAILURES)}: {FAILURES}")
print("=" * 62)
print("NOTE: plumbing only. This says nothing about whether the student can")
print("reach 2 mm -- that needs a real run at 256 envs.")
print(f"RESULT_GATE {'PASS' if passed else 'FAIL'}")
sys.stdout.flush()
sys.stderr.flush()
env.close()
app.close()
os._exit(0 if passed else 1)
