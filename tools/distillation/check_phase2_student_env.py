"""Phase 2, part 2: the teacher consumes the Phase 1 student contract.

``check_phase2_teacher.py`` measures success rate with ``student_obs`` OFF (the
camera costs ~4x step time and 1024 envs + camera OOMs a 32 GB card). That
proves the ``mus`` path, but not that the teacher is wired to the *student*
env's ``"teacher_obs"`` key. This does, at 4 envs with the camera on.

It also asserts the ``fixed_sigma: coef_cond`` wiring directly rather than
trusting it: with ``block_id=50.0`` the returned sigmas must equal row 0 of the
network's per-block sigma table. That is the one piece of the teacher that fails
*silently* -- ``network_builder.py:410`` argmaxes an exact-equality comparison,
so a block id matching nothing still yields row 0 and looks fine.

    python tools/distillation/check_phase2_student_env.py
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

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=4)
cli.add_argument("--steps", type=int, default=60)
cli.add_argument("--checkpoint", default="pretrained_assembly/tight_insertion/model.pth")
cli.add_argument("--device", default="cuda:0")
args, _ = cli.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args([])
_a.headless = True
_a.enable_cameras = True  # student_obs requires this
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
from isaacsimenvs.distillation.teacher import DEFAULT_BLOCK_ID  # noqa: E402

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


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
cfg.student_obs.enabled = True
env = _instantiate_env(TASK, cfg)
N = env.num_envs

agent_cfg = _configure_agent(
    TASK, AGENT, rl_device=args.device, num_envs=N,
    deterministic=True, games=10**9, extra_overrides={},
)
teacher = Teacher(
    agent_cfg, str(ckpt),
    num_envs=N,
    teacher_obs_dim=int(env.cfg.observation_space),
    critic_dim=int(env.cfg.state_space),
    action_dim=int(env.cfg.action_space),
    device=args.device,
    block_id=DEFAULT_BLOCK_ID,
)

obs, _ = env.reset()
teacher.reset_states()

# --- 1. the student contract is what the teacher is being fed --------------
print("\n1. student contract -> teacher input")
check("obs has the 5 student keys",
      set(obs) == {"proprio", "img", "teacher_obs", "critic", "aux_info"},
      f"got {sorted(obs)}")
check("teacher_obs is (N, 140)", tuple(obs["teacher_obs"].shape) == (N, 140),
      f"got {tuple(obs['teacher_obs'].shape)}")
check("teacher_obs within clip_observations",
      float(obs["teacher_obs"].abs().max()) <= teacher.clip_obs + 1e-4,
      f"max |obs| = {float(obs['teacher_obs'].abs().max()):.4f}, clip = {teacher.clip_obs}")

out = teacher.act(obs["teacher_obs"])

# --- 2. outputs are sane ---------------------------------------------------
print("\n2. teacher outputs")
check("mus is (N, 29)", tuple(out["mus"].shape) == (N, 29), f"got {tuple(out['mus'].shape)}")
check("mus finite", bool(torch.isfinite(out["mus"]).all()))
check("sigmas finite and positive",
      bool(torch.isfinite(out["sigmas"]).all() and (out["sigmas"] > 0).all()),
      f"range=({float(out['sigmas'].min()):.4f}, {float(out['sigmas'].max()):.4f})")
check("action within action-space bounds",
      float(out["action"].abs().max()) <= teacher.clip_actions + 1e-6,
      f"max |a| = {float(out['action'].abs().max()):.4f}")

# --- 3. coef_cond sigma wiring, asserted rather than trusted ---------------
print("\n3. fixed_sigma=coef_cond wiring")
net = teacher.player.model.a2c_network
check("network built in coef_cond mode", getattr(net, "fixed_sigma", None) == "coef_cond",
      f"fixed_sigma = {getattr(net, 'fixed_sigma', None)!r}")
sigma_ids = net.sigma_ids.detach().float().cpu()
check("block_id is an exact member of the sigma table",
      bool((sigma_ids - DEFAULT_BLOCK_ID).abs().min() < 1e-6),
      f"coef_ids = {[round(float(v), 1) for v in sigma_ids]}, block_id = {DEFAULT_BLOCK_ID}")
check("coef_id_idx points just past the teacher obs",
      int(net.sigma_id_idx) == 140, f"got {int(net.sigma_id_idx)}")
expected_row = int((sigma_ids - DEFAULT_BLOCK_ID).abs().argmin())
# The network emits LOGstd; ModelA2CContinuousLogStd exponentiates it
# (models.py:268). res_dict["sigmas"] is therefore exp(sigma_act(sigma[row])).
expected = torch.exp(net.sigma_act(net.sigma[expected_row])).detach()
check(f"sigmas match exp(sigma-table row {expected_row})",
      bool(torch.allclose(out["sigmas"][0].detach().cpu(), expected.cpu(), rtol=1e-5, atol=1e-6)),
      f"max diff = {float((out['sigmas'][0].detach().cpu() - expected.cpu()).abs().max()):.2e}")
check("every env got the same sigma row (constant block_id)",
      bool((out["sigmas"] == out["sigmas"][0:1]).all()))

# The per-block sigma spread drives Phase 5's loss weighting (DEXTRAH weights
# the mu regression by 1/sigma_T^2), so record it rather than discover it later.
print("\n   per-block sigma = exp(sigma_act(sigma[row])), and the 1/sigma^2")
print("   weight it implies for the DAgger mu-regression loss:")
print(f"   {'row':>4s} {'coef_id':>8s} {'sigma_min':>10s} {'sigma_max':>10s} "
      f"{'sigma_med':>10s} {'w_min':>10s} {'w_max':>10s}")
with torch.no_grad():
    for row in range(net.sigma.shape[0]):
        sig = torch.exp(net.sigma_act(net.sigma[row])).float().cpu()
        w = 1.0 / sig.pow(2)
        mark = "  <-- selected" if row == expected_row else ""
        print(f"   {row:>4d} {float(sigma_ids[row]):>8.1f} {float(sig.min()):>10.4f} "
              f"{float(sig.max()):>10.4f} {float(sig.median()):>10.4f} "
              f"{float(w.min()):>10.2e} {float(w.max()):>10.2e}{mark}")
sel = torch.exp(net.sigma_act(net.sigma[expected_row])).detach().float().cpu()
spread = float(sel.max() / sel.min())
print(f"   selected row spans {spread:.0f}x in sigma -> {spread ** 2:.1e}x in 1/sigma^2 weight")

# --- 4. LSTM state actually threads ---------------------------------------
print("\n4. LSTM state threading")
check("teacher is recurrent", teacher.is_rnn)
before = [s.clone() for s in teacher.player.states]
obs, _r, _t, _tr, _e = env.step(out["action"])
out2 = teacher.act(obs["teacher_obs"])
check("hidden state advanced after a step",
      any(not torch.allclose(a, b) for a, b in zip(before, teacher.player.states)))
check("mus changed with the state (teacher is reacting)",
      not torch.allclose(out["mus"], out2["mus"]),
      f"max diff = {float((out['mus'] - out2['mus']).abs().max()):.4e}")

teacher.reset_states(torch.tensor([0], device=env.device))
check("reset_states zeroes only the requested env",
      bool((teacher.player.states[0][:, 0] == 0).all()
           and (teacher.player.states[0][:, 1:].abs().sum() > 0)))

# --- 5. determinism -------------------------------------------------------
print("\n5. determinism")
teacher.reset_states()
saved = obs["teacher_obs"].clone()
a1 = teacher.act(saved)["mus"].clone()
teacher.reset_states()
a2 = teacher.act(saved)["mus"].clone()
check("same obs + same hidden state -> identical mus", bool(torch.equal(a1, a2)))

# --- 6. it still runs with the camera on ----------------------------------
print(f"\n6. rollout with camera on ({args.steps} steps)")
nonfinite = 0
for _ in range(args.steps):
    out = teacher.act(obs["teacher_obs"])
    if not torch.isfinite(out["mus"]).all():
        nonfinite += 1
    obs, _r, term, trunc, _e = env.step(out["action"])
    dones = (term | trunc).reshape(-1).bool()
    if bool(dones.any()):
        teacher.reset_states(dones)
check("no non-finite mus over the rollout", nonfinite == 0, f"{nonfinite} bad steps")
img = obs["img"]
check("student image still non-constant at the end of the rollout",
      all(int(img[i].unique().numel()) > 1 for i in range(N)),
      f"n_unique = {[int(img[i].unique().numel()) for i in range(N)]}")

print("\n" + "=" * 62)
passed = not FAILURES
if passed:
    print("PHASE 2 STUDENT-ENV CHECK PASSED")
else:
    print(f"PHASE 2 STUDENT-ENV CHECK FAILED — {len(FAILURES)} failure(s):")
    for f in FAILURES:
        print(f"  - {f}")
print("=" * 62)
print(f"RESULT_GATE {'PASS' if passed else 'FAIL'}")
sys.stdout.flush()
sys.stderr.flush()
env.close()
app.close()
os._exit(0 if passed else 1)
