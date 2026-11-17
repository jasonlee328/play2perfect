"""A11: ContactBench task 27 "Turn key to unlock door" on PreciseAssembly (A1's faithful door, latch_cbf).

Official reference: contactbench-sim force_bench/appendix_tasks/turn_key_unlock_door_flat.py + kit/mechanisms/
lock_module.py. Problems cb_key (90 deg lock) / cb_key_lock{180,270,360} (FoundationTouch tools/contactbench/
key_tasks.py). Needs precise_assembly.latch_cbf=true latch_cbf_mechanism=key unlock_angle_values_deg=[<lock>]
latch_door_joint=0 unlock_joint=1 latch_keeper=-1 kinematic_distractors=[] enable_retract=true end_on_retract=false
retract_seated_mode=depth retract_distance_threshold=0.0.

What this subclass adds to PreciseAssemblyEnv:
  * friction spring: the key joint gets a velocity drive (target 0, damping key_lock.fric_damping) capped at
    maxForce = the plug drag tau_f = 1.25 N x r (official FRICTION_N on the deadbolt), a Coulomb-like drag; per env
    tau_f x drag scale (key_lock.drag_scale_range, log-uniform per env, fixed for the env's life; phase 2 hidden drag).
  * unwrapped key angle (the 180-360 deg locks turn past +-180 deg): the latch, goals and rewards read it.
  * contact views: hand links vs the key and vs the door (PhysX contact force matrix), per-pad pinch forces.
  * the official checks (turn_key_unlock_door_flat.py l.242-293):
      start_locked_key_inserted: 1.5 N m opening torque on the hinge for t in [0.05, 0.45] s, leaf < 1.0 deg;
      key_fully_turned: deadbolt retraction (r x key angle) > 24.49 mm when the hand has left (= key angle >= clear);
      finger_pinch_during_turn: on turning steps >= 2 fingertip pads > 1 N for > 85 %, pads > 50 N on < 1 %;
      released_moved_away_still: every fingertip > release_dist (8 cm) from the key, hand-key and hand-door contact
          < 0.05 N, palm within 2 mm of its position, for still_steps (0.7 s);
      door_can_be_opened: then the same 1.5 N m hinge torque for <= 1.2 s must lift the leaf >= 5 deg.
    Success (max_successes) = all five, via the base retract machinery (the clear-angle goal enters retract phase,
    _retract_seated_by_depth = this subclass's checks).
  * phase 2 (key_lock.break_ratio > 0): the key "breaks" when the hand's total force on the key exceeds
    break_ratio x F_min(env) for break_steps steps, F_min(env) = fmin_per_Nm x tau_f(env) (calibrated by scripted
    squeeze sweeps); key_lock.hide_key_angle hides the key's rotation from the actor's object obs.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg import PreciseAssemblyEnvCfg

HAND_LINKS = ("left_thumb_MC", "left_thumb_PP", "left_thumb_DP",
              "left_index_MCP_VL", "left_index_PP", "left_index_MP", "left_index_DP",
              "left_middle_MCP_VL", "left_middle_PP", "left_middle_MP", "left_middle_DP",
              "left_ring_MCP_VL", "left_ring_PP", "left_ring_MP", "left_ring_DP",
              "left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_PP", "left_pinky_MP", "left_pinky_DP",
              "iiwa14_link_7")
PADS = ("left_thumb_DP", "left_index_DP", "left_middle_DP", "left_ring_DP", "left_pinky_DP")


@configclass
class KeyLockCfg:
    enabled: bool = True
    # friction spring (lock_module.FRICTION_N = 1.25 N on the deadbolt -> plug drag 1.25 N x r)
    fric_damping: float = 2.0            # N m s/rad of the saturating drag drive (drag = min(D w, tau_f))
    drag_scale_range: tuple[float, float] = (1.0, 1.0)   # per env, log-uniform (phase 2: (0.5, 8.0))
    drag_scale_values: tuple[float, ...] = ()            # if set: env i gets values[i % K] (sweeps)
    drag_seed: int = 7
    # official checks
    contact_n: float = 0.05              # common.CONTACT_FREE_N
    pinch_pad_n: float = 1.0
    pinch_frac: float = 0.85
    pinch_max_n: float = 50.0
    pinch_max_frac: float = 0.01
    pinch_min_turn_steps: int = 3
    turn_step_deg: float = 0.2           # a policy step is a turning step when the key advanced more than this
    require_pinch: bool = True
    release_dist: float = 0.08           # fingertips from the key (official: hand moved >= 8 cm)
    still_steps: int = 42                # 0.7 s at 60 Hz
    drift_mm: float = 2.0
    probe_torque: float = 1.5            # _door_kit.LOCK_PROBE_NM
    start_probe_steps: tuple[int, int] = (3, 27)   # 0.05 .. 0.45 s
    start_locked_max_deg: float = 1.0
    end_probe_steps: int = 72            # 1.2 s (T_PROBE)
    open_deg: float = 5.0
    # rewards
    release_rew_scale: float = 300.0     # potential on min(tip dist, release_dist) / release_dist after the clear goal
    still_rew_scale: float = 200.0       # potential on still_count / still_steps
    turnback_penalty: float = 2.0        # per step with the key back below its clear angle after the clear goal
    pinch_rew: float = 0.5               # per turning step with a valid pinch
    approach_unpinched_scale: float = 0.25   # latch_approach_rew gain on turning progress made without a 2-pad pinch
    fail_penalty: float = 100.0          # probe / pinch fail, broken
    # phase 2
    break_ratio: float = 0.0             # 0 = off; F_break = break_ratio x F_min(env)
    fmin_per_Nm: float = 0.0             # F_min = fmin_per_Nm x tau_f (N per N m), calibrated
    break_steps: int = 3
    hide_key_angle: bool = False


@configclass
class KeyLockEnvCfg(PreciseAssemblyEnvCfg):
    key_lock: KeyLockCfg = KeyLockCfg()


def _glob(p: str) -> str:
    return re.sub(r"env_\d+", "env_*", p)


class KeyLockEnv(PreciseAssemblyEnv):
    cfg: KeyLockEnvCfg

    # ---- scene -------------------------------------------------------------------------------------------------
    def _setup_scene(self) -> None:
        import isaacsimenvs.tasks.precise_assembly.scene_utils as su
        orig = su._pih_rigid_object_cfg

        def cfg_contacts(prim_path, usd_path):
            c = orig(prim_path, usd_path)
            if prim_path.endswith("/Object") or prim_path.endswith("/Distractor0"):
                c.spawn.activate_contact_sensors = True
            return c
        su._pih_rigid_object_cfg = cfg_contacts
        try:
            super()._setup_scene()
        finally:
            su._pih_rigid_object_cfg = orig
        kl = self.cfg.key_lock
        cbf = getattr(self, "_cbf", None)
        if cbf is None or cbf.get("mechanism") != "key":
            raise ValueError("KeyLockEnv needs precise_assembly.latch_cbf=true latch_cbf_mechanism=key")
        from pxr import UsdPhysics
        stage = self.scene.stage if hasattr(self.scene, "stage") else __import__("omni.usd").usd.get_context().get_stage()
        n = int(self.cfg.scene.num_envs)
        lo, hi = (float(v) for v in kl.drag_scale_range)
        rng = np.random.default_rng(int(kl.drag_seed))
        scale = np.exp(rng.uniform(math.log(lo), math.log(hi), n)) if hi > lo else np.full(n, lo)
        if kl.drag_scale_values:
            vals = [float(v) for v in kl.drag_scale_values]
            scale = np.array([vals[i % len(vals)] for i in range(n)])
        self._key_drag_scale_np = scale
        self._key_tau_f_np = np.asarray(cbf["drag_Nm"], dtype=np.float64) * scale
        k = int(self.cfg.precise_assembly.unlock_joint)
        for i in range(n):
            jp = stage.GetPrimAtPath(f"/World/envs/env_{i}/FixtureJoint{k}")
            drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0); drive.CreateTargetVelocityAttr(0.0)
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(float(kl.fric_damping) * math.pi / 180.0)
            drive.CreateMaxForceAttr(float(self._key_tau_f_np[i]))
        print(f"[key_lock] friction-spring drive on joint {k}: tau_f {self._key_tau_f_np.min():.4f}-"
              f"{self._key_tau_f_np.max():.4f} N m (official {float(np.asarray(cbf['drag_Nm'])[0]):.4f}), "
              f"damping {kl.fric_damping} N m s/rad", flush=True)

    # ---- init --------------------------------------------------------------------------------------------------
    def __init__(self, cfg: KeyLockEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        N, dev = self.num_envs, self.device
        z = lambda dt=torch.float32: torch.zeros(N, dtype=dt, device=dev)
        self.key_tau_f = torch.as_tensor(self._key_tau_f_np, device=dev, dtype=torch.float32)
        self.key_clear = torch.as_tensor(self._cbf["clear_deg"], device=dev, dtype=torch.float32).deg2rad()
        self.key_rp_m = torch.as_tensor(self._cbf["rp_mm"], device=dev, dtype=torch.float32) * 1e-3
        kl = self.cfg.key_lock
        self.key_fmin = float(kl.fmin_per_Nm) * self.key_tau_f
        # angle unwrapping
        self._ja_unwrap = None
        self._ja_prev = None
        self._ja_reinit = torch.ones(N, dtype=torch.bool, device=dev)
        # contact views
        paths = self.robot.root_physx_view.link_paths
        p0 = paths[0] if isinstance(paths[0], (list, tuple)) else paths
        links = [p for p in p0 if p.split("/")[-1] in HAND_LINKS]
        names = [p.split("/")[-1] for p in links]
        self._key_pad_idx = torch.tensor([names.index(p) for p in PADS], device=dev)
        self.key_link_names = names
        sim_view = self.object._physics_sim_view
        self._key_view = sim_view.create_rigid_contact_view(
            _glob(self.object.root_physx_view.prim_paths[0]), filter_patterns=[_glob(p) for p in links])
        door = self.distractors[int(self._pih_joints[int(self.cfg.precise_assembly.latch_door_joint)].body[10:])]
        self._door_view = sim_view.create_rigid_contact_view(
            _glob(door.root_physx_view.prim_paths[0]), filter_patterns=[_glob(p) for p in links])
        self._key_nl = len(links)
        self._palm_id = int(self.robot.find_bodies("iiwa14_link_7")[0][0])
        # per-episode state
        self.key_pad_f = torch.zeros(N, len(PADS), device=dev)
        self.key_hand_f = z(); self.door_hand_f = z()
        self._key_turn_steps, self._key_pinch_ok, self._key_pinch_over = z(), z(), z()
        self._key_prev_ang = z()
        self._key_still = z(torch.long)
        self._key_palm_anchor = torch.zeros(N, 3, device=dev)
        self._key_probe_t = z(torch.long)
        self._key_probe_on = z(torch.bool)
        self._key_start_max = z()
        self._key_ok = z(torch.bool)
        self._key_fail_probe, self._key_fail_pinch, self._key_broken, self._key_start_unlocked = (
            z(torch.bool), z(torch.bool), z(torch.bool), z(torch.bool))
        self._key_break_count = z(torch.long)
        self._key_rel_phi = z(); self._key_still_phi = z()
        self._key_min_tip = z()
        self._key_amax, self._key_dmax = z(), z()
        self._key_t = z(torch.long)
        self._key_turn_touch = z()                       # turning steps with any hand-key force > pinch_pad_n
        self._key_turn_linkf = torch.zeros(N, len(HAND_LINKS), device=dev)   # sum of per-link force on turning steps      # policy steps since reset (episode_length_buf restarts on every goal credit)
        self._key_cleared_ep, self._key_released_ep = z(torch.bool), z(torch.bool)
        print(f"[key_lock] contact views: key / door vs {self._key_nl} hand links; clear "
              f"{torch.rad2deg(self.key_clear[:4]).tolist()} deg; break_ratio {kl.break_ratio}", flush=True)

    # ---- unwrapped key angle -----------------------------------------------------------------------------------
    def _update_joint_angles(self) -> None:
        super()._update_joint_angles()
        if not self._pih_joints or not hasattr(self, "_ja_reinit"):
            return
        k = int(self.cfg.precise_assembly.unlock_joint)
        a = self.cb_joint_angle[k]
        if self._ja_unwrap is None:
            self._ja_unwrap = a.clone(); self._ja_prev = a.clone()
        d = torch.atan2(torch.sin(a - self._ja_prev), torch.cos(a - self._ja_prev))
        u = torch.where(self._ja_reinit, a, self._ja_unwrap + d)
        self._ja_reinit[:] = False
        self._ja_prev = a.clone()
        self._ja_unwrap = u
        self.cb_joint_angle[k] = u

    # ---- phase 2: hide the key's turn from the actor (obs_utils hook) ------------------------------------------
    def _actor_turn_hide(self):
        if not bool(self.cfg.key_lock.hide_key_angle) or not hasattr(self, "_ja_reinit"):
            return None
        c = self.cfg.precise_assembly
        j = self._pih_joints[int(c.unlock_joint)]
        door = self.distractors[int(j.parent[10:])]
        from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul
        kp = int(j.parent[10:])
        qp0 = torch.tensor([self._pih_distractor_pose_rel[kp][6], *self._pih_distractor_pose_rel[kp][3:6]],
                           device=self.device, dtype=torch.float32).expand(self.num_envs, 4)
        q_par = quat_mul(door.data.root_quat_w, quat_conjugate(qp0))
        axis_w = quat_apply(q_par, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(self.num_envs, 3))
        ang = self.cb_joint_angle[int(c.unlock_joint)]
        return quat_from_angle_axis(-ang, axis_w), axis_w

    # ---- physics substep: hinge probes (replaces the knob spring correction of latch_cbf) ----------------------
    def _cbf_spring_correction(self, knob: torch.Tensor) -> None:
        if not hasattr(self, "_key_probe_on"):
            return
        kl = self.cfg.key_lock
        c = self.cfg.precise_assembly
        t = self._key_t
        s0, s1 = (int(v) for v in kl.start_probe_steps)
        on = ((t >= s0) & (t <= s1)) | self._key_probe_on
        jd = self._pih_joints[int(c.latch_door_joint)]
        body = self.distractors[int(jd.body[10:])]
        axis_w = quat_apply(self.hole_quat_wxyz, torch.tensor(jd.axis, device=self.device,
                                                              dtype=torch.float32).expand(self.num_envs, 3))
        tq = (on.float() * float(kl.probe_torque)).unsqueeze(-1) * axis_w
        body.set_external_force_and_torque(torch.zeros(self.num_envs, 1, 3, device=self.device), tq.unsqueeze(1),
                                           is_global=True)

    # ---- success condition of the clear goal: the key is past its clear angle ----------------------------------
    def _latch_cbf_success_now(self, door: torch.Tensor) -> torch.Tensor:
        k = int(self.cfg.precise_assembly.unlock_joint)
        return self.cb_joint_angle[k] >= self.key_clear

    def _retract_seated_by_depth(self, pih_cfg):
        return self._key_ok

    # ---- per policy step ---------------------------------------------------------------------------------------
    def _key_measure(self) -> None:
        kl = self.cfg.key_lock
        c = self.cfg.precise_assembly
        N = self.num_envs
        F = self._key_view.get_contact_force_matrix(self.physics_dt).reshape(N, self._key_nl, 3).norm(dim=-1)
        Fd = self._door_view.get_contact_force_matrix(self.physics_dt).reshape(N, self._key_nl, 3).norm(dim=-1)
        self.key_pad_f = F[:, self._key_pad_idx]
        self.key_link_f = F
        self.key_hand_f = F.sum(-1)
        self.door_hand_f = Fd.sum(-1)
        self._update_joint_angles()
        k = int(c.unlock_joint)
        ang = self.cb_joint_angle[k]
        door = self.cb_joint_angle[int(c.latch_door_joint)]
        self._key_t += 1
        t = self._key_t
        # start lock probe
        s0, s1 = (int(v) for v in kl.start_probe_steps)
        inwin = (t >= s0) & (t <= s1 + 6)
        self._key_start_max = torch.where(inwin, torch.maximum(self._key_start_max, door), self._key_start_max)
        self._key_start_unlocked = (t == s1 + 6) & (self._key_start_max > math.radians(float(kl.start_locked_max_deg)))
        # pinch during the turn (before the clear goal)
        turning = ((ang - self._key_prev_ang) > math.radians(float(kl.turn_step_deg))) & ~self.retract_phase
        npad = (self.key_pad_f > float(kl.pinch_pad_n)).sum(-1)
        self._key_turn_steps += turning.float()
        self._key_pinch_ok += (turning & (npad >= 2)).float()
        self._key_pinch_over += (turning & (self.key_pad_f.max(-1).values > float(kl.pinch_max_n))).float()
        self._key_turning_ok = turning & (npad >= 2)
        self._key_turn_touch += (turning & (self.key_hand_f > float(kl.pinch_pad_n))).float()
        self._key_turn_linkf += turning.float().unsqueeze(-1) * self.key_link_f[:, :self._key_turn_linkf.shape[1]]
        self._key_prev_ang = ang.clone()
        # release / still / probe (after the clear goal = retract phase)
        tips = self.robot.data.body_pos_w[:, self._fingertip_body_ids, 0:3]
        key_pos = self.object.data.root_pos_w
        self._key_min_tip = (tips - key_pos.unsqueeze(1)).norm(dim=-1).min(-1).values
        palm = self.robot.data.body_pos_w[:, self._palm_id, 0:3]
        cleared = ang >= self.key_clear
        free = (self.key_hand_f < float(kl.contact_n)) & (self.door_hand_f < float(kl.contact_n))
        released = self.retract_phase & cleared & free & (self._key_min_tip > float(kl.release_dist))
        near_anchor = (palm - self._key_palm_anchor).norm(dim=-1) <= float(kl.drift_mm) * 1e-3
        still_ok = released & near_anchor & (self._key_still > 0)
        self._key_still = torch.where(still_ok, self._key_still + 1,
                                      torch.where(released, torch.ones_like(self._key_still),
                                                  torch.zeros_like(self._key_still)))
        restart = released & ~still_ok
        self._key_palm_anchor = torch.where(restart.unsqueeze(-1), palm, self._key_palm_anchor)
        start_probe = (self._key_still >= int(kl.still_steps)) & ~self._key_probe_on
        self._key_probe_on |= start_probe
        self._key_probe_t = torch.where(self._key_probe_on, self._key_probe_t + 1, self._key_probe_t)
        opened = self._key_probe_on & (door >= math.radians(float(kl.open_deg)))
        ts = self._key_turn_steps.clamp(min=1.0)
        pinch_ok = ((self._key_turn_steps >= float(kl.pinch_min_turn_steps))
                    & (self._key_pinch_ok / ts >= float(kl.pinch_frac))
                    & (self._key_pinch_over / ts <= float(kl.pinch_max_frac)))
        self._key_pinch_ok_ep = pinch_ok
        if not bool(kl.require_pinch):
            pinch_ok = torch.ones_like(pinch_ok)
        self._key_amax = torch.maximum(self._key_amax, ang)
        self._key_dmax = torch.maximum(self._key_dmax, door)
        self._key_cleared_ep |= cleared
        self._key_released_ep |= released
        self._key_ok = opened & pinch_ok
        self._key_fail_pinch = opened & ~pinch_ok
        self._key_fail_probe = self._key_probe_on & (self._key_probe_t > int(kl.end_probe_steps)) & ~opened
        # phase 2: breaking force
        if float(kl.break_ratio) > 0.0:
            over = self.key_hand_f > float(kl.break_ratio) * self.key_fmin
            self._key_break_count = torch.where(over, self._key_break_count + 1, torch.zeros_like(self._key_break_count))
            self._key_broken = self._key_break_count >= int(kl.break_steps)
        # logs
        e = self.extras
        e["key/angle_deg_mean"] = torch.rad2deg(ang).mean()
        e["key/retract_mm_mean"] = (ang.clamp(min=0.0) * self.key_rp_m).mean() * 1e3
        e["key/cleared_frac"] = cleared.float().mean()
        e["key/pinch2_turning_frac"] = (self._key_pinch_ok.sum() / self._key_turn_steps.sum().clamp(min=1.0))
        e["key/hand_force_mean"] = self.key_hand_f.mean()
        e["key/released_frac"] = released.float().mean()
        e["key/probe_on_frac"] = self._key_probe_on.float().mean()
        e["key/start_door_max_deg"] = torch.rad2deg(self._key_start_max).max()

    def _get_dones(self):
        self._key_measure()
        terminated, truncated = super()._get_dones()
        tr = self._termination_reasons
        succ = tr["max_successes"]
        probe_fail = self._key_fail_probe & ~succ
        pinch_fail = self._key_fail_pinch & ~succ
        broken = self._key_broken & ~succ
        start_unl = self._key_start_unlocked & ~succ
        self._termination_reasons = {"key_broken": broken, "probe_fail": probe_fail, "pinch_fail": pinch_fail,
                                     "start_unlocked": start_unl, **tr}
        self._key_term = broken | probe_fail | pinch_fail | start_unl
        # end-of-episode snapshots for evals (the auto reset after _get_dones clears the live counters)
        self._key_last_turn = self._key_turn_steps.clone()
        self._key_last_pinch = self._key_pinch_ok.clone()
        self._key_last_over = self._key_pinch_over.clone()
        self._key_last_amax = self._key_amax.clone()
        self._key_last_dmax = self._key_dmax.clone()
        self._key_last_cleared = self._key_cleared_ep.clone()
        self._key_last_released = self._key_released_ep.clone()
        self._key_last_probe = self._key_probe_on.clone()
        self._key_last_handf = self.key_hand_f.clone()
        self._key_last_touch = self._key_turn_touch.clone()
        self._key_last_linkf = self._key_turn_linkf.clone()
        return terminated | self._key_term, truncated

    def _get_rewards(self):
        r = super()._get_rewards()
        kl = self.cfg.key_lock
        k = int(self.cfg.precise_assembly.unlock_joint)
        ang = self.cb_joint_angle[k]
        rp = self.retract_phase.float()
        phi = (self._key_min_tip / float(kl.release_dist)).clamp(0.0, 1.0) * rp
        sphi = (self._key_still.float() / float(kl.still_steps)).clamp(0.0, 1.0) * rp
        rel = float(kl.release_rew_scale) * (phi - self._key_rel_phi) + float(kl.still_rew_scale) * (sphi - self._key_still_phi)
        self._key_rel_phi, self._key_still_phi = phi, sphi
        back = float(kl.turnback_penalty) * (self.retract_phase & (ang < self.key_clear)).float()
        pinch = float(kl.pinch_rew) * self._key_turning_ok.float()
        fail = float(kl.fail_penalty) * self._key_term.float()
        # the official check needs a 2-pad pinch: progress made by poking the bow (thumb only) pays less
        ar = self._reward_terms.get("latch_approach_rew")
        unp = torch.zeros_like(r)
        if ar is not None:
            unp = (1.0 - float(kl.approach_unpinched_scale)) * ar.clamp(min=0.0) * (~self._key_turning_ok).float()
        extra = rel - back + pinch - fail - unp
        r = r + extra
        for name, v in (("key_release_rew", rel), ("key_turnback_pen", -back), ("key_pinch_rew", pinch), ("key_unpinched_cut", -unp),
                        ("key_fail_pen", -fail)):
            self._reward_terms[name] = v
        self._reward_terms["total_reward"] = r
        return r

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "_ja_reinit"):
            self._ja_reinit[env_ids] = True
        super()._reset_idx(env_ids)
        if not hasattr(self, "_key_still"):
            return
        self._ja_reinit[env_ids] = True
        for t in (self._key_turn_steps, self._key_pinch_ok, self._key_pinch_over, self._key_prev_ang,
                  self._key_start_max, self._key_rel_phi, self._key_still_phi, self._key_amax, self._key_dmax,
                  self._key_turn_touch, self._key_turn_linkf):
            t[env_ids] = 0.0
        for t in (self._key_still, self._key_probe_t, self._key_break_count, self._key_t):
            t[env_ids] = 0
        for t in (self._key_probe_on, self._key_ok, self._key_cleared_ep, self._key_released_ep):
            t[env_ids] = False
