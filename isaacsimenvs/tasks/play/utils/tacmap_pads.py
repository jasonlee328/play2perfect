"""Original sharpa-tacmap contact physics for the Sharpa fingertip pads (A23, 2026-11-22).

The TacMap sensor (``tactile_utils.TactileSensor``) is the same ray-cast as the original ``SharpaTacmap``
(checked pixel by pixel). What our port had dropped is the physics the original sensor was calibrated with
(github.com/sharpa-robotics/sharpa-tacmap, env_cfg.py ``SharpaWaveEnvCfg``):

  robot (``UsdFileWithCompliantContactCfg``):
    - every robot collision shape: contact_offset 0.2 mm, rest_offset -0.1 mm   (ours: 2 mm / 0)
    - the five elastomer (gel pad) shapes only: compliant contact, stiffness 280, damping 20
      (a separate material on the pad; the hard DP shell stays rigid)
  object (their presser): contact_offset 0.1 mm, rest_offset -0.5 mm            (ours: 2 mm / 0 or per task)
  sim: dt 1/240, TGS, 8 position / 0 velocity iterations, bounce threshold 0.2 m/s (ours: 1/120, 1-192 / 0-1, 0.5)

``apply_original_tacmap_pads`` does the USD part (shapes, material, offsets) at scene setup. The sim part is the
override set in ``configs/tacmap_original.txt`` (FoundationTouch repo). Enabled by ``tactile.original_pads=true``.

Our hand (left, URDF import, fixed joints merged) has the same gel mesh as its own collision shape:
``Robot/left_<f>_DP/collisions/elastomer`` (thumb: ``thumb_elastomer``), next to the shell ``.../collisions/left_DP``.
"""
from __future__ import annotations

PAD_COLLISION_NAMES = ("elastomer", "thumb_elastomer")   # collision prim names of the gel pads under left_<f>_DP


def _is_pad(path: str) -> bool:
    """True for collision prims under .../collisions/elastomer (the importer puts the CollisionAPI on its child
    node_STL_BINARY_)."""
    parts = path.split("/")
    if "collisions" not in parts:
        return False
    j = parts.index("collisions")
    return j + 1 < len(parts) and parts[j + 1] in PAD_COLLISION_NAMES


ORIGINAL = dict(pad_compliant_stiffness=280.0, pad_compliant_damping=20.0, pad_friction=-1.0, pad_robot_offsets=True,
                robot_contact_offset=0.0002, robot_rest_offset=-0.0001, pad_object_offsets=True,
                object_contact_offset=0.0001, object_rest_offset=-0.0005)


def apply_original_tacmap_pads(env, params: dict | None = None) -> dict:
    """Bind the original pad physics in USD (call right after setup_scene, before the sim starts).
    params: dict with the ORIGINAL keys (default: read env.cfg.tactile). Returns counts; raises if the hand does not
    have exactly 5 pad shapes per env."""
    from types import SimpleNamespace
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade
    import time as _time
    import omni.usd

    t = SimpleNamespace(**params) if params is not None else env.cfg.tactile
    a = env.cfg.assets
    stage = omni.usd.get_context().get_stage()
    k, c = float(t.pad_compliant_stiffness), float(t.pad_compliant_damping)
    pad_mu = float(a.finger_tip_friction) if float(t.pad_friction) < 0 else float(t.pad_friction)

    def _mat(path, mu, kk=0.0, cc=0.0):
        m = UsdShade.Material.Define(stage, path)
        UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        pm = UsdPhysics.MaterialAPI(m.GetPrim())
        pm.CreateStaticFrictionAttr(float(mu)); pm.CreateDynamicFrictionAttr(float(mu)); pm.CreateRestitutionAttr(0.0)
        if kk > 0.0:
            px = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
            px.CreateCompliantContactStiffnessAttr(kk); px.CreateCompliantContactDampingAttr(cc)
        return m

    FINGERTIP_LINK_NAMES = tuple(f"left_{f}_DP" for f in ("thumb", "index", "middle", "ring", "pinky"))
    pad_mat = _mat("/World/TacmapPadMat", pad_mu, k, c)
    tip_mat = _mat("/World/TacmapTipShellMat", a.finger_tip_friction)
    rob_mat = _mat("/World/TacmapRobotMat", a.robot_friction)

    def _offsets(prim, contact, rest):
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr().Set(float(contact)); pc.CreateRestOffsetAttr().Set(float(rest))

    n_pad = n_tip = n_rob = n_obj = 0
    _t0 = _time.time()
    _cb = Sdf.ChangeBlock(); _cb.__enter__()   # A21 11-22: batch USD change notices (12288 envs hung > 40 min without)
    for i in range(env.cfg.scene.num_envs):
        if i % 1024 == 0:
            print(f"[tacmap-original] env {i}/{env.cfg.scene.num_envs} ({_time.time() - _t0:.1f}s)", flush=True)
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        pads_here = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            p = str(prim.GetPath())
            if t.pad_robot_offsets:
                _offsets(prim, t.robot_contact_offset, t.robot_rest_offset)
            if _is_pad(p):
                mat = pad_mat; n_pad += 1; pads_here += 1
            elif any(f"/{n}/" in p + "/" for n in FINGERTIP_LINK_NAMES):
                mat = tip_mat; n_tip += 1
            else:
                mat = rob_mat; n_rob += 1
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        if pads_here != 5:
            raise RuntimeError(f"tactile.original_pads: expected 5 gel pad collision shapes in env {i}, found {pads_here}")
        if t.pad_object_offsets:
            oroot = stage.GetPrimAtPath(env.object.cfg.prim_path.replace("env_.*", f"env_{i}"))
            for prim in Usd.PrimRange(oroot):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    _offsets(prim, t.object_contact_offset, t.object_rest_offset); n_obj += 1
    _cb.__exit__(None, None, None)
    print(f"[tacmap-original] USD patch done in {_time.time() - _t0:.1f}s", flush=True)
    env._finger_compliant = True   # the runtime tensor friction pass would replace the bound materials (A10 logic)
    out = dict(pad_shapes=n_pad, tip_shell_shapes=n_tip, robot_shapes=n_rob, object_shapes=n_obj, k=k, c=c, pad_mu=pad_mu)
    print(f"[tacmap-original] pads k={k} d={c} mu={pad_mu} on {n_pad} shapes (5/env), shells {n_tip}, robot {n_rob}; "
          f"robot offsets {t.robot_contact_offset}/{t.robot_rest_offset} ({'on' if t.pad_robot_offsets else 'off'}); "
          f"object offsets {t.object_contact_offset}/{t.object_rest_offset} on {n_obj} shapes "
          f"({'on' if t.pad_object_offsets else 'off'})", flush=True)
    return out
