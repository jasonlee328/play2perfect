"""A3 force-window env: PreciseAssemblyEnv + a ``force_window`` cfg group. Off (mode "off") it is PreciseAssembly.

mode "fragile" (problem fw_fragile / fw_fragile_m<g>): lift the can 10 cm (base lift goal). Grip force
G = sum over the 20 hand links + iiwa14_link_7 (palm) of |F_link on the can| (PhysX contact force matrix, last
substep of the step); ``crush_metric=peak`` uses the largest single link instead. Crushed (termination
"crushed", -crush_penalty) when G > crush_ratio * F_min(env) for crush_steps steps in a row, F_min = m g / mu_eff
(the static two-sided grip needed to hold the can). crush_ratio 0 = no limit. Hidden mass per env:
mass_values_kg (env i gets values[i % K], set through the PhysX view after start, inertia scaled with it).

mode "press" (problem fw_press): the rocker key. Press force F = contact force of the hand links on the key (same
contact view as the grip, smoothed with force_ema). The spring estimate k(env) * theta / pad_r is kept as
env.fw_force_spring only: in the no-hand calibration it under-reads by ~0.7 deg of angle per N, whatever k (pivot
compliance), so it is not used for the rule. In window = |F - f_star| <= window * f_star. Holding in
the window for hold_steps -> "held"; success (termination max_successes) once held and then every fingertip is
release_dist from the pad. Broken (termination "broken", -break_penalty) when F > f_star * (1 + break_ratio *
window). Single contact: more than one hand link touching the key for multi_steps steps -> "multi_contact" (fail).
Hidden spring per env: key_k_values (N m / rad, env i gets values[i % K]); written into each env's fixture-joint
drive before the sim starts. Dense reward: press_rew_scale * (contact ? 1 - min(|F - f*| / f*, 1) : 0) per step,
+ hold_bonus on held, + release_bonus on success.

Both modes: ``compliant_stiffness`` / ``compliant_damping`` > 0 author a compliant-contact PhysX material on the
object's collision prims before the sim starts (so penetration ~ F / k and the ray-cast taxels see force).
Termination reasons gain "crushed" / "broken" / "multi_contact" (first in the dict, as the evals read the first
true reason). env.fw_grip (N,), env.fw_links (N,) and env.fw_force (N, press) are exposed for evals / probes.
"""

from __future__ import annotations

import math
import re

import torch
from isaaclab.utils import configclass

from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv
from isaacsimenvs.tasks.precise_assembly.precise_assembly_env_cfg import PreciseAssemblyEnvCfg

GRIP_LINKS = ("left_thumb_MC", "left_thumb_PP", "left_thumb_DP",
              "left_index_MCP_VL", "left_index_PP", "left_index_MP", "left_index_DP",
              "left_middle_MCP_VL", "left_middle_PP", "left_middle_MP", "left_middle_DP",
              "left_ring_MCP_VL", "left_ring_PP", "left_ring_MP", "left_ring_DP",
              "left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_PP", "left_pinky_MP", "left_pinky_DP",
              "iiwa14_link_7")
G = 9.81


@configclass
class ForceWindowCfg:
    mode: str = "off"                 # off | fragile | press
    contact_n: float = 0.05           # N: a link counts as touching above this
    compliant_stiffness: float = 0.0  # N/m, object material (0 = rigid PhysX contact)
    compliant_damping: float = 0.0
    # fragile
    crush_ratio: float = 0.0          # F_break = crush_ratio * F_min(env); 0 = no limit
    mu_eff: float = 1.0               # friction used for F_min = m g / mu_eff
    crush_metric: str = "total"       # total | peak
    crush_steps: int = 3
    crush_penalty: float = 300.0
    crush_ratio_start: float = 0.0    # >0: crush ratio ramps (log-linear) from this down to crush_ratio
    crush_ramp_steps: int = 0         # env steps of the ramp (a resume replays it: set crush_ramp_offset_steps)
    crush_ramp_offset_steps: int = 0
    mass_values_kg: tuple[float, ...] = ()
    # press
    f_star: float = 4.0
    window: float = 0.25
    break_ratio: float = 2.0
    window_start: float = 0.0         # >0: press window ramps (log-linear) from this down to window
    window_ramp_steps: int = 0
    window_ramp_offset_steps: int = 0
    hold_steps: int = 30
    release_dist: float = 0.03
    pad_r: float = 0.064
    pad_pos_rel_pivot: tuple[float, float, float] = (0.064, 0.0, 0.009)
    key_k_values: tuple[float, ...] = ()
    multi_steps: int = 5
    break_penalty: float = 300.0
    force_ema: float = 0.5            # press force smoothing (weight of the newest step)
    press_rew_scale: float = 1.0
    hold_bonus: float = 200.0
    release_bonus: float = 500.0


@configclass
class ForceWindowEnvCfg(PreciseAssemblyEnvCfg):
    force_window: ForceWindowCfg = ForceWindowCfg()


def _glob(p: str) -> str:
    return re.sub(r"env_\d+", "env_*", p)


class ForceWindowEnv(PreciseAssemblyEnv):
    cfg: ForceWindowEnvCfg

    # ---- scene -------------------------------------------------------------------------------------------------
    def _setup_scene(self) -> None:
        fw = self.cfg.force_window
        import isaacsimenvs.tasks.precise_assembly.scene_utils as su
        orig = su._pih_rigid_object_cfg

        def cfg_contacts(prim_path, usd_path):
            c = orig(prim_path, usd_path)
            if prim_path.endswith("/Object"):
                c.spawn.activate_contact_sensors = True
            return c
        su._pih_rigid_object_cfg = cfg_contacts
        try:
            super()._setup_scene()
        finally:
            su._pih_rigid_object_cfg = orig
        if fw.mode == "off":
            return
        from pxr import Usd, UsdPhysics, UsdShade
        stage = self.scene.stage if hasattr(self.scene, "stage") else __import__("omni.usd").usd.get_context().get_stage()
        if fw.compliant_stiffness > 0.0:
            from pxr import PhysxSchema
            mat_path = "/World/FwCompliantMaterial"
            mat = UsdShade.Material.Define(stage, mat_path)
            UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            pm = UsdPhysics.MaterialAPI(mat.GetPrim())
            mu = float(self.cfg.assets.object_friction)
            pm.CreateStaticFrictionAttr(mu); pm.CreateDynamicFrictionAttr(mu); pm.CreateRestitutionAttr(0.0)
            px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
            px.CreateCompliantContactStiffnessAttr(float(fw.compliant_stiffness))
            px.CreateCompliantContactDampingAttr(float(fw.compliant_damping))
            n_bound = 0
            for i in range(self.num_envs):
                root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Object")
                for prim in Usd.PrimRange(root):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                        n_bound += 1
            print(f"[force_window] compliant material k={fw.compliant_stiffness} c={fw.compliant_damping} "
                  f"bound to {n_bound} collision prims", flush=True)
        if fw.mode == "press" and fw.key_k_values:
            ks = [float(fw.key_k_values[i % len(fw.key_k_values)]) for i in range(self.num_envs)]
            for i, k in enumerate(ks):
                jp = stage.GetPrimAtPath(f"/World/envs/env_{i}/FixtureJoint0")
                drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateTargetPositionAttr(0.0); drive.CreateTargetVelocityAttr(0.0)
                drive.CreateStiffnessAttr(k * math.pi / 180.0)
            self._fw_k_np = ks
            print(f"[force_window] key spring per env from {list(fw.key_k_values)}", flush=True)

    # ---- init --------------------------------------------------------------------------------------------------
    def __init__(self, cfg: ForceWindowEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        import isaacsimenvs.tasks.play.play_env as pe
        orig_mat = pe.apply_physx_material_properties
        if cfg.force_window.compliant_stiffness > 0.0:
            # The tensor-API material write replaces the object's USD material (and with it the compliant contact):
            # run it with the object hidden, so the object keeps the compliant material authored in _setup_scene.
            def mat_without_object(env):
                obj = env.__dict__.pop("object", None)
                try:
                    orig_mat(env)
                finally:
                    if obj is not None:
                        env.object = obj
                print("[force_window] tensor material write skipped the object (compliant material kept)", flush=True)
            pe.apply_physx_material_properties = mat_without_object
        try:
            super().__init__(cfg, render_mode, **kwargs)
        finally:
            pe.apply_physx_material_properties = orig_mat
        fw = self.cfg.force_window
        N, dev = self.num_envs, self.device
        self.fw_grip = torch.zeros(N, device=dev)
        self.fw_peak = torch.zeros(N, device=dev)
        self.fw_links = torch.zeros(N, device=dev)
        self.fw_force = torch.zeros(N, device=dev)
        self.fw_force_spring = torch.zeros(N, device=dev)
        self._fw_crush_count = torch.zeros(N, dtype=torch.long, device=dev)
        self._fw_hold = torch.zeros(N, dtype=torch.long, device=dev)
        self._fw_held = torch.zeros(N, dtype=torch.bool, device=dev)
        self._fw_multi = torch.zeros(N, dtype=torch.long, device=dev)
        self._fw_just_held = torch.zeros(N, dtype=torch.bool, device=dev)
        self._fw_success = torch.zeros(N, dtype=torch.bool, device=dev)
        self._fw_crushed = torch.zeros(N, dtype=torch.bool, device=dev)
        self._fw_broken = torch.zeros(N, dtype=torch.bool, device=dev)
        self._fw_multi_fail = torch.zeros(N, dtype=torch.bool, device=dev)
        if fw.mode == "off":
            return
        # grip contact view: object vs hand links
        paths = self.robot.root_physx_view.link_paths
        p0 = paths[0] if isinstance(paths[0], (list, tuple)) else paths
        p0 = [p for p in p0 if p.split("/")[-1] in GRIP_LINKS]
        self._fw_link_names = [p.split("/")[-1] for p in p0]
        self._fw_view = self.object._physics_sim_view.create_rigid_contact_view(
            _glob(self.object.root_physx_view.prim_paths[0]), filter_patterns=[_glob(p) for p in p0])
        self._fw_nl = len(p0)
        # hidden mass per env
        view = self.object.root_physx_view
        m0 = view.get_masses().clone()           # (N, 1) cpu
        if fw.mass_values_kg:
            vals = torch.tensor([float(fw.mass_values_kg[i % len(fw.mass_values_kg)]) for i in range(N)])
            ratio = (vals / m0[:, 0]).unsqueeze(-1)
            inert = view.get_inertias().clone()
            idx = torch.arange(N, dtype=torch.int32)
            view.set_masses(vals.unsqueeze(-1).to(m0.dtype), idx)
            view.set_inertias(inert * ratio, idx)
            m_now = view.get_masses()[:, 0].clone()
            print(f"[force_window] hidden mass per env from {list(fw.mass_values_kg)}; "
                  f"env0..3 mass now {m_now[:4].tolist()}", flush=True)
        self.fw_mass = view.get_masses()[:, 0].clone().to(dev)
        self.fw_fmin = self.fw_mass * G / float(fw.mu_eff)
        if fw.mode == "press":
            k = getattr(self, "_fw_k_np", None)
            k0 = float(self._pih_joints[0].stiffness) if self._pih_joints else 0.0
            self.fw_k = torch.tensor(k if k is not None else [k0] * N, device=dev, dtype=torch.float32)
            print(f"[force_window] press: f*={fw.f_star} window={fw.window} break at "
                  f"{fw.f_star * (1 + fw.break_ratio * fw.window):.2f} N, k env0..4 {self.fw_k[:5].tolist()}",
                  flush=True)
        print(f"[force_window] mode={fw.mode} grip view vs {self._fw_nl} links", flush=True)

    # ---- per step ----------------------------------------------------------------------------------------------
    def _fw_measure(self) -> None:
        fw = self.cfg.force_window
        F = self._fw_view.get_contact_force_matrix(self.physics_dt).reshape(self.num_envs, self._fw_nl, 3)
        mag = F.norm(dim=-1)
        self.fw_grip = mag.sum(-1)
        self.fw_peak = mag.max(-1).values
        self.fw_links = (mag > fw.contact_n).float().sum(-1)

    def _pad_pos_w(self) -> torch.Tensor:
        j = self._pih_joints[0]
        piv = torch.tensor(j.anchor, device=self.device, dtype=torch.float32).expand(self.num_envs, 3)
        from isaaclab.utils.math import quat_apply
        th = self.cb_joint_angle[0]
        r = torch.tensor(self.cfg.force_window.pad_pos_rel_pivot, device=self.device, dtype=torch.float32)
        # rotate the pad offset about the joint axis (+y in the fixture frame) by theta
        c, s = torch.cos(th), torch.sin(th)
        off = torch.stack([c * r[0] + s * r[2], torch.full_like(th, r[1]), -s * r[0] + c * r[2]], -1)
        return self.hole.data.root_pos_w + quat_apply(self.hole.data.root_quat_w, piv + off)

    def _fw_ramp(self, start, target, steps, offset):
        """Log-linear ramp start -> target over env steps (common_step_counter + offset); start <= 0 = target."""
        if start <= 0.0 or steps <= 0 or target <= 0.0:
            return target
        f = min(1.0, max(0.0, (self.common_step_counter + offset) / float(steps)))
        return math.exp(math.log(start) + f * (math.log(target) - math.log(start)))

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        fw = self.cfg.force_window
        if fw.mode == "off":
            return terminated, truncated
        self._fw_measure()
        zero = torch.zeros_like(terminated)
        crushed, broken, multi_fail, success = zero.clone(), zero.clone(), zero.clone(), zero.clone()
        self._fw_just_held = zero.clone()
        if fw.mode == "fragile" and fw.crush_ratio > 0.0:
            g = self.fw_peak if fw.crush_metric == "peak" else self.fw_grip
            self.fw_crush_ratio_now = self._fw_ramp(fw.crush_ratio_start, fw.crush_ratio, fw.crush_ramp_steps,
                                                    fw.crush_ramp_offset_steps)
            over = g > self.fw_crush_ratio_now * self.fw_fmin
            self._fw_crush_count = torch.where(over, self._fw_crush_count + 1, torch.zeros_like(self._fw_crush_count))
            crushed = self._fw_crush_count >= int(fw.crush_steps)
        if fw.mode == "press":
            self._update_joint_angles()
            self.fw_force_spring = self.fw_k * self.cb_joint_angle[0].clamp(min=0.0) / float(fw.pad_r)
            a = float(fw.force_ema)
            self.fw_force = a * self.fw_grip + (1.0 - a) * self.fw_force   # hand-on-key contact force, smoothed
            w = self._fw_ramp(fw.window_start, fw.window, fw.window_ramp_steps, fw.window_ramp_offset_steps)
            self.fw_window_now = w
            lo, hi = fw.f_star * (1.0 - w), fw.f_star * (1.0 + w)
            inwin = (self.fw_force >= lo) & (self.fw_force <= hi) & ~self._fw_held
            self._fw_hold = torch.where(inwin, self._fw_hold + 1, torch.zeros_like(self._fw_hold))
            self._fw_just_held = (self._fw_hold >= int(fw.hold_steps)) & ~self._fw_held
            self._fw_held |= self._fw_just_held
            broken = self.fw_force > fw.f_star * (1.0 + fw.break_ratio * w)
            self._fw_multi = torch.where(self.fw_links > 1, self._fw_multi + 1, torch.zeros_like(self._fw_multi))
            multi_fail = self._fw_multi >= int(fw.multi_steps)
            tips = self.robot.data.body_state_w[:, self._fingertip_body_ids, 0:3]
            pad = self._pad_pos_w()
            far = (tips - pad.unsqueeze(1)).norm(dim=-1).min(-1).values > fw.release_dist
            success = self._fw_held & far & (self.fw_links == 0) & ~broken
        self._fw_crushed, self._fw_broken, self._fw_multi_fail, self._fw_success = crushed, broken, multi_fail, success
        tr = self._termination_reasons
        if fw.mode == "press":
            tr["max_successes"] = success
        self._termination_reasons = {"crushed": crushed, "broken": broken, "multi_contact": multi_fail, **tr}
        terminated = terminated | crushed | broken | multi_fail | success
        return terminated, truncated

    def _get_rewards(self):
        r = super()._get_rewards()
        fw = self.cfg.force_window
        if fw.mode == "off":
            return r
        r = r - fw.crush_penalty * self._fw_crushed.float() - fw.break_penalty * (self._fw_broken | self._fw_multi_fail).float()
        if fw.mode == "press":
            touching = self.fw_links > 0
            closeness = 1.0 - ((self.fw_force - fw.f_star).abs() / fw.f_star).clamp(max=1.0)
            r = r + fw.press_rew_scale * torch.where(touching, closeness, torch.zeros_like(closeness))
            r = r + fw.hold_bonus * self._fw_just_held.float() + fw.release_bonus * self._fw_success.float()
        return r

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if not hasattr(self, "_fw_hold"):
            return
        for t in (self._fw_crush_count, self._fw_hold, self._fw_multi):
            t[env_ids] = 0
        self._fw_held[env_ids] = False
        self.fw_force[env_ids] = 0.0
