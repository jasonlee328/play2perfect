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
    pinch_frac_ramp_steps: int = 0       # > 0: the required pinch share ramps 0 -> pinch_frac over this many env steps
    pinch_frac_offset_steps: int = 0     # (a resume restarts common_step_counter: set this to continue the ramp)
    pinch_max_n: float = 50.0
    pinch_max_frac: float = 0.01
    pinch_min_turn_steps: int = 3
    turn_step_deg: float = 0.2           # a policy step is a turning step when the key advanced more than this
    require_pinch: bool = True
    release_dist: float = 0.08           # fingertips from the key (official: hand moved >= 8 cm)
    still_steps: int = 42                # 0.7 s at 60 Hz
    drift_mm: float = 2.0
    drift_net: bool = True               # official (turn_key_unlock_door_flat.py:264-267): NET palm displacement between the
                                         # first and last step of the still window < drift_mm; False = every step within
                                         # drift_mm of the window start (stricter, pre-11-17 behaviour)
    drift_start_mm: float = 0.0          # > 0: the stillness tolerance ramps drift_start_mm -> drift_mm over drift_ramp_steps
    drift_ramp_steps: int = 0
    drift_offset_steps: int = 0
    probe_torque: float = 1.5            # _door_kit.LOCK_PROBE_NM
    start_probe_steps: tuple[int, int] = (3, 27)   # 0.05 .. 0.45 s
    start_locked_max_deg: float = 1.0
    end_probe_steps: int = 72            # 1.2 s (T_PROBE)
    open_deg: float = 5.0
    # goal 0 = the key turned first_goal_deg (by its joint angle: the key's keypoints sit near the turning axis, so a
    # keypoint test passed at rest = a free reach_goal_bonus every episode, cb_keyA). Keypoint distance proxy for the
    # dense keypoint reward: remaining angle x progress_m_per_rad.
    first_goal_deg: float = 20.0
    progress_m_per_rad: float = 0.02
    # rewards
    release_rew_scale: float = 300.0     # potential on min(tip dist, release_dist) / release_dist after the clear goal
    still_rew_scale: float = 200.0       # potential on still_count / still_steps
    still_action_penalty: float = 0.0    # per step once released: x mean |arm action| (arm actions are joint velocities, so
                                         # holding still = zero arm action)
    still_speed_penalty: float = 0.0     # per step once released: x palm speed (m/s) / 0.01 + joint-target change (rad), released only
    turnback_penalty: float = 2.0        # per step with the key back below its clear angle after the clear goal
    pinch_rew: float = 1.0               # per turning step with a valid pinch
    approach_unpinched_scale: float = 0.5
    pad_contact_rew: float = 0.0         # per step before the clear goal: x (# fingertip pads on the key > pad_contact_n, max 2) / 2
    pad_contact_n: float = 0.5
    # grasp shaping (human demo: thumb + index pinch the bow from above): per step before the clear goal, for the thumb
    # and index pads on OPPOSITE sides of the bow plate: straddle_rew x exp(-(in-plane dist) / s_plane) x
    # exp(-(normal dist) / s_normal), summed over the two pads / 2. Bow centre / plate normal in the key object frame.
    straddle_rew: float = 0.0
    straddle_sigma_plane: float = 0.02
    straddle_sigma_normal: float = 0.02
    bow_center_obj: tuple[float, float, float] = (-0.00747, 0.00024, 0.0)
    # check the pinch when the key first clears: a turn without the required pinch share ends the episode there
    # (pinch_fail), so an unpinched turn earns none of the release / probe rewards
    pinch_fail_at_clear: bool = False   # latch_approach_rew gain on turning progress made without a 2-pad pinch
    fail_penalty: float = 100.0          # probe / pinch fail, broken
    # phase 2
    break_ratio: float = 0.0             # 0 = off; F_break = break_ratio x F_min(env)
    fmin_per_Nm: float = 0.0             # F_min = fmin_per_Nm x tau_f (N per N m), calibrated
    break_steps: int = 5
    hide_key_angle: bool = False
    # phase 2 v2 (touch-only slip cue, like the bottle clutch): per-episode hidden F_min ~ log-uniform p2_fmin_range (N);
    # hand force on the key below F_min while the key turns = SLIP: physics unchanged (joint state carries nothing),
    # the tactile ray-cast sees the key lag behind (felt key rotated back by p2_cue_gain x slipped angle, capped,
    # decaying p2_cue_decay / step once the grip holds); slipped angle > p2_slip_fail_deg = fail "slipped".
    # Hand force > p2_rho x F_min for break_steps = "key_broken". p2_rho <= 0 = no break limit.
    p2_enabled: bool = False
    p2_fmin_range: tuple[float, float] = (4.0, 24.0)
    p2_rho: float = 0.0
    p2_rho_start: float = 0.0            # > 0: rho ramps p2_rho_start -> p2_rho over p2_rho_ramp_steps (curriculum)
    p2_rho_ramp_steps: int = 0
    p2_rho_offset_steps: int = 0
    p2_slip_fail_deg: float = 30.0
    p2_cue_gain: float = 3.0
    p2_cue_cap_deg: float = 25.0
    p2_cue_decay: float = 0.2
    p2_slip_penalty: float = 2.0         # per slipping step
    p2_force: str = "pads"               # force the window acts on: "pads" (sum of the 5 fingertip pads = the pinch) | "hand"
    p2_force_ema: float = 0.2            # weight of the newest step in the smoothed force


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
        stops = np.asarray(cbf["stop_deg"], dtype=np.float64)
        self._key_soft_stop_np = stops > 179.0
        for i in range(n):
            jp = stage.GetPrimAtPath(f"/World/envs/env_{i}/FixtureJoint{k}")
            if self._key_soft_stop_np[i]:
                # PhysX clamps a revolute D6 limit to 180 deg ("Upper angle limit ... clamped to 180 degrees"): the
                # 270 / 360 deg locks get no USD limits; both stops are enforced in software (_key_soft_stop)
                for nm in ("physics:lowerLimit", "physics:upperLimit"):
                    if jp.HasAttribute(nm):
                        jp.RemoveProperty(nm)
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
        ft_names = [self.robot.body_names[i] for i in self._fingertip_body_ids]
        self._key_thumb_i = next(i for i, nm in enumerate(ft_names) if "thumb" in nm)
        self._key_index_i = next(i for i, nm in enumerate(ft_names) if "index" in nm)
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
        lo, hi = (float(v) for v in kl.p2_fmin_range)
        self._p2_lo, self._p2_hi = math.log(lo), math.log(hi)
        self.key_fmin2 = torch.exp(torch.empty(N, device=dev).uniform_(self._p2_lo, self._p2_hi))
        self._p2_slip_acc, self._p2_lag = z(), z()
        self._p2_slipping = z(torch.bool)
        self._key_slipped = z(torch.bool)
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
        if not hasattr(self, "cb_joint_angle"):
            self._update_joint_angles()          # first obs build at reset, before any measurement
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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._key_arm_act = actions[:, :7].abs().mean(-1).to(self.device)
        super()._pre_physics_step(actions)

    def _tactile_object_pose(self):
        """Phase 2 v2: the tactile ray-cast sees the key rotated back about the lock axis by the slip lag (physics and
        joint state unchanged). None = the real pose."""
        if not bool(self.cfg.key_lock.p2_enabled) or not hasattr(self, "_p2_lag") or not bool((self._p2_lag > 0).any()):
            return None
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul
        c = self.cfg.precise_assembly
        j = self._pih_joints[int(c.unlock_joint)]
        door = self.distractors[int(j.parent[10:])]
        dq = door.data.root_quat_w
        ax = quat_apply(dq, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(dq.shape[0], 3))
        an = door.data.root_pos_w + quat_apply(dq, torch.tensor(j.anchor, device=self.device, dtype=torch.float32).expand(dq.shape[0], 3))
        rot = quat_from_angle_axis(-self._p2_lag, ax)
        pos, quat = self.object.data.root_pos_w, self.object.data.root_quat_w
        return an + quat_apply(rot, pos - an), quat_mul(rot, quat)

    # ---- physics substep: hinge probes (replaces the knob spring correction of latch_cbf) ----------------------
    def _key_soft_stop(self) -> None:
        """Locks whose stop is past 179 deg: pose-level stops on the unwrapped key angle, every physics substep. Past
        the plug stop (or below the lower limit) the key is rotated back about the lock axis by the excess and its
        spin in that direction removed (a velocity / torque stop is out-pushed by the hand)."""
        if not hasattr(self, "_key_soft_stop_t"):
            self._key_soft_stop_t = torch.as_tensor(self._key_soft_stop_np, device=self.device)
            self._key_stop_hi = torch.as_tensor(np.radians(self._cbf["stop_deg"]), device=self.device, dtype=torch.float32)
            j = self._pih_joints[int(self.cfg.precise_assembly.unlock_joint)]
            self._key_stop_lo = math.radians(float(j.lower_deg if j.lower_deg is not None else -1.0))
        if not bool(self._key_soft_stop_t.any()):
            return
        c = self.cfg.precise_assembly
        k = int(c.unlock_joint)
        ang = self.cb_joint_angle[k]
        over = torch.where(ang > self._key_stop_hi, ang - self._key_stop_hi,
                           torch.where(ang < self._key_stop_lo, ang - self._key_stop_lo, torch.zeros_like(ang)))
        over = torch.where(self._key_soft_stop_t, over, torch.zeros_like(over))
        ids = (over != 0).nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        from isaaclab.utils.math import quat_from_angle_axis, quat_mul
        j = self._pih_joints[k]
        door = self.distractors[int(j.parent[10:])]
        dq = door.data.root_quat_w[ids]
        ax = quat_apply(dq, torch.tensor(j.axis, device=self.device, dtype=torch.float32).expand(ids.numel(), 3))
        an = door.data.root_pos_w[ids] + quat_apply(dq, torch.tensor(j.anchor, device=self.device, dtype=torch.float32).expand(ids.numel(), 3))
        rot = quat_from_angle_axis(-over[ids], ax)
        pos, quat = self.object.data.root_pos_w[ids], self.object.data.root_quat_w[ids]
        self.object.write_root_pose_to_sim(torch.cat([an + quat_apply(rot, pos - an), quat_mul(rot, quat)], -1), env_ids=ids)
        vel = self.object.data.root_vel_w[ids].clone()
        wd = vel[:, 3:6] - door.data.root_vel_w[ids, 3:6]
        w_ax = (wd * ax).sum(-1)
        sgn = torch.sign(over[ids])
        bad = (w_ax * sgn).clamp(min=0.0) * sgn          # spin further past the stop
        vel[:, 3:6] -= bad.unsqueeze(-1) * ax
        vel[:, 0:3] = door.data.root_vel_w[ids, 0:3]
        self.object.write_root_velocity_to_sim(vel, env_ids=ids)
        self.cb_joint_angle[k, ids] -= over[ids]
        if self._ja_unwrap is not None:
            # keep the unwrap continuous: the next measurement already includes the correction
            self._ja_unwrap[ids] -= over[ids]
            pv = self._ja_prev[ids] - over[ids]
            self._ja_prev[ids] = torch.atan2(torch.sin(pv), torch.cos(pv))
        self._key_stop_hits = getattr(self, "_key_stop_hits", 0) + int(ids.numel())

    def _cbf_spring_correction(self, knob: torch.Tensor) -> None:
        if not hasattr(self, "_key_probe_on"):
            return
        self._key_soft_stop()
        kl = self.cfg.key_lock
        c = self.cfg.precise_assembly
        t = self._key_t
        s0, s1 = (int(v) for v in kl.start_probe_steps)
        # the start (lock) probe acts only while the bolt is engaged: a fast policy can clear the key inside the
        # official 0.05-0.45 s window, and pushing an unlocked door then is not the locked-door check
        eng = getattr(self, "_cbf_engaged", torch.ones_like(self._key_probe_on))
        on = ((t >= s0) & (t <= s1) & eng) | self._key_probe_on
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

    def _apply_unlock_goal(self, prev_near_steps: torch.Tensor) -> None:
        """Goal 0 (the first waypoint) by the key angle, not by keypoints; the last goal stays the base unlock logic."""
        from isaacsimenvs.tasks.play.utils.reward_utils import update_near_goal_steps
        kl = self.cfg.key_lock
        self._update_joint_angles()
        ang = self.cb_joint_angle[int(self.cfg.precise_assembly.unlock_joint)]
        first = (self._successes < self.env_max_goals - 1) & ~self.is_random_goal_env
        rem = (math.radians(float(kl.first_goal_deg)) - ang).clamp(min=0.0)
        kp = rem * float(kl.progress_m_per_rad)
        self._keypoints_max_dist = torch.where(first, kp, self._keypoints_max_dist)
        if not hasattr(self, "_key_kp_fresh"):
            self._key_kp_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # first step after a reset: the base just seeded the tracker with the (tiny) real keypoint distance
        fresh = self._key_kp_fresh | (self._closest_keypoint_max_dist < 0)
        self._closest_keypoint_max_dist = torch.where(first & fresh, kp, self._closest_keypoint_max_dist)
        self._key_kp_fresh[:] = False
        near = rem <= 0.0
        steps = update_near_goal_steps(near_goal=near, near_goal_steps=prev_near_steps,
                                       force_consecutive=self.cfg.termination.force_consecutive_near_goal_steps)
        self._near_goal = torch.where(first, near, self._near_goal)
        self._near_goal_steps = torch.where(first, steps, self._near_goal_steps)
        super()._apply_unlock_goal(prev_near_steps)

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
        eng = getattr(self, "_cbf_engaged", torch.ones_like(inwin))
        self._key_start_max = torch.where(inwin & eng, torch.maximum(self._key_start_max, door), self._key_start_max)
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
        dmm = float(kl.drift_mm)
        if float(kl.drift_start_mm) > 0.0 and int(kl.drift_ramp_steps) > 0:
            f = min(1.0, float(self.common_step_counter + int(kl.drift_offset_steps)) / float(kl.drift_ramp_steps))
            dmm = float(kl.drift_start_mm) + f * (dmm - float(kl.drift_start_mm))
        self.extras["key/drift_mm_cur"] = torch.tensor(dmm, device=self.device)
        if bool(kl.drift_net):
            W = int(kl.still_steps)
            if not hasattr(self, "_key_palm_buf"):
                self._key_palm_buf = palm.unsqueeze(1).repeat(1, W, 1).clone()
                self._key_rel_run = torch.zeros_like(self._key_still)
            self._key_rel_run = torch.where(released, self._key_rel_run + 1, torch.zeros_like(self._key_rel_run))
            slot = self._key_t % W
            ar = torch.arange(palm.shape[0], device=palm.device)
            old_p = self._key_palm_buf[ar, slot].clone()            # palm W-1 steps ago (written W steps back)
            self._key_palm_buf[ar, slot] = palm
            self._key_net_mm = torch.where(released & (self._key_rel_run >= W), (palm - old_p).norm(dim=-1) * 1e3,
                                           torch.full_like(palm[:, 0], 1e3))
            net_ok = (palm - old_p).norm(dim=-1) <= dmm * 1e-3
            # still counter: released steps so far, capped; passes when a full window of released steps has a small net drift
            self._key_still = torch.where(released, torch.minimum(self._key_rel_run, torch.full_like(self._key_rel_run, W - 1)),
                                          torch.zeros_like(self._key_still))
            self._key_still = torch.where(released & (self._key_rel_run >= W) & net_ok,
                                          torch.full_like(self._key_still, W), self._key_still)
            still_ok = released
        else:
            near_anchor = (palm - self._key_palm_anchor).norm(dim=-1) <= dmm * 1e-3
            still_ok = released & near_anchor & (self._key_still > 0)
            self._key_still = torch.where(still_ok, self._key_still + 1,
                                          torch.where(released, torch.ones_like(self._key_still),
                                                      torch.zeros_like(self._key_still)))
        restart = released & ~still_ok
        self._key_released_now = released
        self._key_palm_anchor = torch.where(restart.unsqueeze(-1), palm, self._key_palm_anchor)
        start_probe = (self._key_still >= int(kl.still_steps)) & ~self._key_probe_on
        self._key_probe_on |= start_probe
        self._key_probe_t = torch.where(self._key_probe_on, self._key_probe_t + 1, self._key_probe_t)
        opened = self._key_probe_on & (door >= math.radians(float(kl.open_deg)))
        ts = self._key_turn_steps.clamp(min=1.0)
        pf = float(kl.pinch_frac)
        if int(kl.pinch_frac_ramp_steps) > 0:
            pf *= min(1.0, float(self.common_step_counter + int(kl.pinch_frac_offset_steps)) / float(kl.pinch_frac_ramp_steps))
        self.extras["key/pinch_frac_required"] = torch.tensor(pf, device=self.device)
        pinch_ok = ((self._key_turn_steps >= float(kl.pinch_min_turn_steps))
                    & (self._key_pinch_ok / ts >= pf)
                    & (self._key_pinch_over / ts <= float(kl.pinch_max_frac)))
        self._key_pinch_ok_ep = pinch_ok
        if not bool(kl.require_pinch):
            pinch_ok = torch.ones_like(pinch_ok)
        first_clear = cleared & ~self._key_cleared_ep
        self._key_amax = torch.maximum(self._key_amax, ang)
        self._key_dmax = torch.maximum(self._key_dmax, door)
        self._key_cleared_ep |= cleared
        self._key_released_ep |= released
        self._key_ok = opened & pinch_ok
        self._key_fail_pinch = opened & ~pinch_ok
        if bool(kl.pinch_fail_at_clear) and bool(kl.require_pinch):
            self._key_fail_pinch = self._key_fail_pinch | (first_clear & ~pinch_ok)
        self.extras["key/pinch_ok_at_clear_frac"] = (first_clear & pinch_ok).float().sum() / first_clear.float().sum().clamp(min=1.0)
        # grasp shaping: thumb + index pads straddling the bow plate
        self._key_straddle = torch.zeros_like(ang)
        if float(kl.straddle_rew) > 0.0:
            from isaaclab.utils.math import quat_apply as qa
            ft = self.robot.data.body_state_w[:, self._fingertip_body_ids, :]
            off = torch.tensor([0.02, 0.002, 0.0], device=self.device).expand(ft.shape[0], ft.shape[1], 3)
            pads = ft[..., 0:3] + qa(ft[..., 3:7].reshape(-1, 4), off.reshape(-1, 3)).reshape(ft.shape[0], ft.shape[1], 3)
            oq = self.object.data.root_quat_w
            cw = self.object.data.root_pos_w + qa(oq, torch.tensor(kl.bow_center_obj, device=self.device).expand(oq.shape[0], 3))
            nw = qa(oq, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(oq.shape[0], 3))
            terms, sides = [], []
            for i in (self._key_thumb_i, self._key_index_i):
                d = pads[:, i] - cw
                sn = (d * nw).sum(-1)
                dp = (d - sn.unsqueeze(-1) * nw).norm(dim=-1)
                terms.append(torch.exp(-dp / float(kl.straddle_sigma_plane)) * torch.exp(-sn.abs() / float(kl.straddle_sigma_normal)))
                sides.append(sn)
            opp = (sides[0] * sides[1] < 0).float()
            self._key_straddle = opp * 0.5 * (terms[0] + terms[1]) * (~self.retract_phase).float()
            self.extras["key/straddle_mean"] = self._key_straddle.mean()
        self._key_fail_probe = self._key_probe_on & (self._key_probe_t > int(kl.end_probe_steps)) & ~opened
        # phase 2 v2: hidden F_min, touch-only slip cue, break above rho x F_min
        if bool(kl.p2_enabled):
            dth = (ang - self._key_prev_ang_p2) if hasattr(self, "_key_prev_ang_p2") else torch.zeros_like(ang)
            self._key_prev_ang_p2 = ang.clone()
            fnow = self.key_pad_f.sum(-1) if str(kl.p2_force) == "pads" else self.key_hand_f
            if not hasattr(self, "_p2_f"):
                self._p2_f = fnow.clone()
            a_ = float(kl.p2_force_ema)
            self._p2_f = a_ * fnow + (1.0 - a_) * self._p2_f
            turn2 = (dth > 0) & ~self.retract_phase & (fnow > float(kl.contact_n))
            slipping = turn2 & (self._p2_f < self.key_fmin2)
            self._p2_slipping = slipping
            self._p2_slip_acc = self._p2_slip_acc + torch.where(slipping, dth, torch.zeros_like(dth))
            cap = math.radians(float(kl.p2_cue_cap_deg))
            lag = torch.where(slipping, (self._p2_lag + float(kl.p2_cue_gain) * dth).clamp(max=cap),
                              self._p2_lag * (1.0 - float(kl.p2_cue_decay)))
            self._p2_lag = lag
            self._key_slipped = self._p2_slip_acc > math.radians(float(kl.p2_slip_fail_deg))
            rho = float(kl.p2_rho)
            if float(kl.p2_rho_start) > 0.0 and int(kl.p2_rho_ramp_steps) > 0:
                f = min(1.0, float(self.common_step_counter + int(kl.p2_rho_offset_steps)) / float(kl.p2_rho_ramp_steps))
                rho = float(kl.p2_rho_start) + f * (rho - float(kl.p2_rho_start))
            self.extras["key/p2_rho_cur"] = torch.tensor(rho, device=self.device)
            if rho > 0.0:
                over = self._p2_f > rho * self.key_fmin2
                self._key_break_count = torch.where(over, self._key_break_count + 1, torch.zeros_like(self._key_break_count))
                self._key_broken = self._key_break_count >= int(kl.break_steps)
            self.extras["key/p2_slipping_frac"] = slipping.float().mean()
            self.extras["key/p2_force_mean"] = self._p2_f.mean()
            self.extras["key/p2_lag_deg_mean"] = torch.rad2deg(self._p2_lag).mean()
        # phase 2 (v1): breaking force
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
        slipped = self._key_slipped & ~succ & ~broken
        start_unl = self._key_start_unlocked & ~succ
        self._termination_reasons = {"key_broken": broken, "slipped": slipped, "probe_fail": probe_fail, "pinch_fail": pinch_fail,
                                     "start_unlocked": start_unl, **tr}
        self._key_term = broken | slipped | probe_fail | pinch_fail | start_unl
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
        self._key_last_fmin = self.key_fmin2.clone()
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
        pinch = float(kl.pinch_rew) * self._key_turning_ok.float() + float(kl.straddle_rew) * self._key_straddle
        if float(kl.pad_contact_rew) > 0.0:
            npads = (self.key_pad_f > float(kl.pad_contact_n)).sum(-1).clamp(max=2).float()
            pinch = pinch + float(kl.pad_contact_rew) * (npads / 2.0) * (~self.retract_phase).float()
        fail = float(kl.fail_penalty) * self._key_term.float()
        # the official check needs a 2-pad pinch: progress made by poking the bow (thumb only) pays less
        ar = self._reward_terms.get("latch_approach_rew")
        unp = torch.zeros_like(r)
        if ar is not None:
            unp = (1.0 - float(kl.approach_unpinched_scale)) * ar.clamp(min=0.0) * (~self._key_turning_ok).float()
        slip_pen = float(kl.p2_slip_penalty) * self._p2_slipping.float() if bool(kl.p2_enabled) else torch.zeros_like(r)
        spen = torch.zeros_like(r)
        if float(kl.still_speed_penalty) > 0.0 and hasattr(self, "_key_released_now"):
            pv = self.robot.data.body_lin_vel_w[:, self._palm_id, :].norm(dim=-1)
            spen = float(kl.still_speed_penalty) * (pv / 0.01).clamp(max=10.0) * self._key_released_now.float()
            self.extras["key/released_palm_speed"] = (pv * self._key_released_now.float()).sum() / self._key_released_now.float().sum().clamp(min=1.0)
        if float(kl.still_action_penalty) > 0.0 and hasattr(self, "_key_released_now") and hasattr(self, "_key_arm_act"):
            rl = self._key_released_now.float()
            spen = spen + float(kl.still_action_penalty) * self._key_arm_act * rl
            self.extras["key/released_arm_action"] = (self._key_arm_act * rl).sum() / rl.sum().clamp(min=1.0)
        extra = rel - back + pinch - fail - unp - slip_pen - spen
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
        if hasattr(self, "key_fmin2"):
            self.key_fmin2[env_ids] = torch.exp(torch.empty(len(env_ids), device=self.device).uniform_(self._p2_lo, self._p2_hi))
            self._p2_slip_acc[env_ids] = 0.0; self._p2_lag[env_ids] = 0.0
            if hasattr(self, "_p2_f"):
                self._p2_f[env_ids] = 0.0
            self._key_slipped[env_ids] = False; self._p2_slipping[env_ids] = False
            self._key_broken[env_ids] = False
            if hasattr(self, "_key_prev_ang_p2"):
                self._key_prev_ang_p2[env_ids] = 0.0
        for t in (self._key_still, self._key_probe_t, self._key_break_count, self._key_t):
            t[env_ids] = 0
        if hasattr(self, "_key_kp_fresh"):
            self._key_kp_fresh[env_ids] = True
        for t in (self._key_probe_on, self._key_ok, self._key_cleared_ep, self._key_released_ep):
            t[env_ids] = False
