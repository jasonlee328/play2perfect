"""Sensing ablations for distillation students (A4, 2026-11-16): what the student is allowed to sense, with the teacher
obs, the task rules and every reader of ``env._last_tactile`` left unchanged.

All options live in ``TactileCfg`` and are off by default (then every student-facing buffer IS the raw one):

- ``student_res`` (0 = off; divides ``resolution``): fingertip pads at a coarser spatial resolution. Each pad's taxels are
  averaged over ``(resolution / student_res)^2`` blocks, weighted by the number of rays per taxel, then upsampled back
  to ``resolution`` (nearest; off-pad taxels stay 0), so the student's encoder and tensor shape are unchanged and only
  the information drops. ``student_res=1`` = one taxel per pad (its mean contact depth).
- ``student_hand_grid`` (0 = off; divides ``whole_hand_grid``): the whole-hand skin at a coarser grid (block max, the
  skin's own taxel rule, upsampled back). Nested bins make both coarsenings exact functions of the recorded taxels
  (``pool_pads`` / ``pool_skin`` below also run offline on stored data).
- ``student_hold_steps`` (k >= 1): sensor rate. Every student touch / force buffer refreshes every k policy steps
  (60 / k Hz at the default 60 Hz control) and at every reset, and holds its value in between.
- ``wrench``: pooled touch = one net force + torque per pad (field ``tactile_wrench``, 5 x 6) and, with the whole-hand
  skin, per link (``tactile_hand_wrench``, 21 x 6), from the SAME rays as the maps: f_i = depth_i * inward ray
  direction (a linear-spring skin), F = mean over the pad's rays [mm of depth], torque = mean of (ray start - pad
  centre) x f_i [mm x cm], both in the pad / link body frame. Same contact information as the map, no spatial map.
  Depth only: like the maps it has no shear or friction.
- ``joint_ft``: PhysX joint reaction wrenches (``robot.data.body_incoming_joint_wrench_b``; parent -> child, the
  frame is reported by the A4 smoke test): ``wrist_ft`` (6) = iiwa joint a7 (the flange; the hand is merged into
  link 7, so it includes the hand's weight and inertia), ``tip_ft`` (5 x 6) = the five DIP joints (contact on the
  distal phalanx incl. friction, plus its tiny weight). N and N m.

Student-facing buffers: ``env._student_tactile``, ``env._student_tactile_hand``, ``env._student_tactile_wrench``,
``env._student_tactile_hand_wrench``, ``env._student_wrist_ft``, ``env._student_tip_ft``; the student obs fields
``tactile`` / ``tactile_hand`` / ``tactile_wrench`` / ``tactile_hand_wrench`` / ``wrist_ft`` / ``tip_ft`` read them.
"""

from __future__ import annotations

import torch

FINGERS = 5
WRIST_BODY = "iiwa14_link_7"
TIP_BODIES = tuple(f"left_{f}_DP" for f in ("thumb", "index", "middle", "ring", "pinky"))


def pool_pads(tac: torch.Tensor, res: int, sub: int, counts: torch.Tensor) -> torch.Tensor:
    """(..., 5*res*res) pad taxels -> same shape at sub x sub resolution (count-weighted block mean, upsampled back,
    taxels with no rays kept at 0). ``counts`` (5*res*res,) = rays per taxel (unclamped)."""
    if sub in (0, res):
        return tac
    if res % sub:
        raise ValueError(f"student_res {sub} must divide resolution {res}")
    b = res // sub
    lead = tac.shape[:-1]
    x = tac.reshape(-1, FINGERS, sub, b, sub, b).float()
    w = counts.to(x.device, x.dtype).reshape(1, FINGERS, sub, b, sub, b)
    m = (x * w).sum((3, 5)) / w.sum((3, 5)).clamp_min(1e-9)                        # (n, 5, sub, sub)
    up = m.repeat_interleave(b, -2).repeat_interleave(b, -1)                       # (n, 5, res, res)
    up = up * (w.reshape(1, FINGERS, res, res) > 0)
    return up.reshape(*lead, -1).to(tac.dtype)


def pool_skin(skin: torch.Tensor, grid: int, sub: int, links: int = 21) -> torch.Tensor:
    """(..., links*grid*grid) skin taxels -> same shape at sub x sub per link (block max, upsampled back)."""
    if sub in (0, grid):
        return skin
    if grid % sub:
        raise ValueError(f"student_hand_grid {sub} must divide whole_hand_grid {grid}")
    b = grid // sub
    lead = skin.shape[:-1]
    m = skin.reshape(-1, links, sub, b, sub, b).amax((3, 5))
    return m.repeat_interleave(b, -2).repeat_interleave(b, -1).reshape(*lead, -1)


def _hold(env, name: str, new: torch.Tensor, k: int) -> torch.Tensor:
    """Sample-and-hold at every k policy steps (and at reset, episode_length_buf == 0)."""
    prev = getattr(env, name, None)
    if k > 1 and prev is not None and prev.shape == new.shape:
        refresh = (env.episode_length_buf % k == 0).view(-1, *([1] * (new.dim() - 1)))
        new = torch.where(refresh, new, prev)
    setattr(env, name, new)
    return new


def joint_ft(env) -> tuple[torch.Tensor, torch.Tensor]:
    """(wrist (N, 6), tips (N, 30)) PhysX incoming joint wrenches."""
    w = env.robot.data.body_incoming_joint_wrench_b
    return w[:, env._a4_wrist_id], w[:, env._a4_tip_ids].reshape(env.num_envs, -1)


def add_sensing_obs(env, obs_clean: dict) -> None:
    """Called by build_observations right after the raw tactile is computed: adds the optional force fields to the
    (teacher-side) obs dict and sets every student-facing buffer (variants applied, then the sensor-rate hold)."""
    tc = env.cfg.tactile
    k = max(1, int(getattr(tc, "student_hold_steps", 1)))
    if getattr(tc, "wrench", False):
        obs_clean["tactile_wrench"] = env._tactile.last_wrench.reshape(env.num_envs, -1)
        if getattr(env, "_hand_skin", None) is not None:
            obs_clean["tactile_hand_wrench"] = env._hand_skin.last_wrench.reshape(env.num_envs, -1)
    if getattr(tc, "joint_ft", False):
        obs_clean["wrist_ft"], obs_clean["tip_ft"] = joint_ft(env)

    tac = obs_clean["tactile"]
    sub = int(getattr(tc, "student_res", 0))
    if sub:
        tac = pool_pads(tac, int(tc.resolution), sub, env._tactile.ray_counts)
    _hold(env, "_student_tactile", tac, k)
    if "tactile_hand" in obs_clean:
        sk = obs_clean["tactile_hand"]
        hs = int(getattr(tc, "student_hand_grid", 0))
        if hs:
            sk = pool_skin(sk, int(tc.whole_hand_grid), hs)
        _hold(env, "_student_tactile_hand", sk, k)
    for f in ("tactile_wrench", "tactile_hand_wrench", "wrist_ft", "tip_ft"):
        if f in obs_clean:
            _hold(env, f"_student_{f}", obs_clean[f], k)


__all__ = ["pool_pads", "pool_skin", "joint_ft", "add_sensing_obs", "WRIST_BODY", "TIP_BODIES"]
