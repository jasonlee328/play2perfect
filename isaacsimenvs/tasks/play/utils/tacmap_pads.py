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

``apply_original_tacmap_pads`` does the USD part (shapes, material, offsets) at scene setup: on env 0 just before
the envs are cloned (``install_before_clone``), so every clone inherits it, then ``check`` verifies envs. The sim part is the
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


def _params(env, params):
    from types import SimpleNamespace
    return SimpleNamespace(**params) if params is not None else env.cfg.tactile


def apply_original_tacmap_pads(env, params: dict | None = None, env_ids=None) -> dict:
    """Bind the original pad physics in USD on the given envs (default: all) through the Usd API. Fast enough for a
    few envs; for training use ``install_before_clone`` (env 0 only, then cloned). Raises unless 5 pads per env."""
    import time
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
    import omni.usd

    t0 = time.time()
    t = _params(env, params)
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

    tips = tuple(f"left_{f}_DP" for f in ("thumb", "index", "middle", "ring", "pinky"))
    mats = dict(pad=_mat("/World/TacmapPadMat", pad_mu, k, c), tip=_mat("/World/TacmapTipShellMat", a.finger_tip_friction),
                rob=_mat("/World/TacmapRobotMat", a.robot_friction))

    def _offsets(prim, contact, rest):
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr().Set(float(contact)); pc.CreateRestOffsetAttr().Set(float(rest))

    ids = range(env.cfg.scene.num_envs) if env_ids is None else env_ids
    cnt = dict(pad=0, tip=0, rob=0, obj=0)
    for i in ids:
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        pads_here = 0
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            p = str(prim.GetPath())
            if t.pad_robot_offsets:
                _offsets(prim, t.robot_contact_offset, t.robot_rest_offset)
            kd = "pad" if _is_pad(p) else ("tip" if any(f"/{n}/" in p + "/" for n in tips) else "rob")
            pads_here += kd == "pad"; cnt[kd] += 1
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[kd], UsdShade.Tokens.strongerThanDescendants, "physics")
        if pads_here != 5:
            raise RuntimeError(f"tactile.original_pads: expected 5 gel pad collision shapes in env {i}, found {pads_here}")
        if t.pad_object_offsets:
            oroot = stage.GetPrimAtPath(env.object.cfg.prim_path.replace("env_.*", f"env_{i}"))
            for prim in Usd.PrimRange(oroot):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    _offsets(prim, t.object_contact_offset, t.object_rest_offset); cnt["obj"] += 1
    env._finger_compliant = True   # the runtime tensor friction pass would replace the bound materials (A10 logic)
    out = dict(pad_shapes=cnt["pad"], tip_shell_shapes=cnt["tip"], robot_shapes=cnt["rob"], object_shapes=cnt["obj"],
               k=k, c=c, pad_mu=pad_mu, envs=len(ids), seconds=round(time.time() - t0, 2))
    print(f"[tacmap-original] pads k={k} d={c} mu={pad_mu} on {cnt['pad']} shapes in {len(ids)} env(s), shells "
          f"{cnt['tip']}, robot {cnt['rob']}; robot offsets {t.robot_contact_offset}/{t.robot_rest_offset} "
          f"({'on' if t.pad_robot_offsets else 'off'}); object offsets {t.object_contact_offset}/{t.object_rest_offset} "
          f"on {cnt['obj']} shapes ({'on' if t.pad_object_offsets else 'off'}); {out['seconds']} s", flush=True)
    return out


def _signature(stage, i):
    """Per collision shape of env i's robot: (relative path, bound physics material, offsets)."""
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade
    rob = f"/World/envs/env_{i}/Robot"
    out = []
    for pr in Usd.PrimRange(stage.GetPrimAtPath(rob)):
        if not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        rel = UsdShade.MaterialBindingAPI(pr).GetDirectBindingRel("physics")
        tg = rel.GetTargets() if rel else []
        pc = PhysxSchema.PhysxCollisionAPI(pr)
        co = pc.GetContactOffsetAttr().Get() if pr.HasAPI(PhysxSchema.PhysxCollisionAPI) else None
        ro = pc.GetRestOffsetAttr().Get() if pr.HasAPI(PhysxSchema.PhysxCollisionAPI) else None
        out.append((str(pr.GetPath())[len(rob):], str(tg[0]).replace(f"/World/envs/env_{i}", "") if tg else None,
                    None if co is None else round(float(co), 7), None if ro is None else round(float(ro), 7)))
    return out


def check(env, params: dict | None = None) -> dict:
    """After cloning: env 0 must have exactly 5 pads bound to the pad material, and the first, a middle and the last
    env must have the same per-shape materials and offsets as env 0. Object offsets (if on) are authored per env
    here (objects can differ between envs)."""
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    n = env.cfg.scene.num_envs
    s0 = _signature(stage, 0)
    npad = sum(1 for r, m, *_ in s0 if _is_pad(r) and m is not None and m.endswith("/TacmapPadMat"))
    if npad != 5:
        raise RuntimeError(f"tactile.original_pads: env 0 has {npad} pads bound to the pad material (want 5)")
    for i in sorted({1, n // 2, n - 1} - {0}):
        if i < n and _signature(stage, i) != s0:
            raise RuntimeError(f"tactile.original_pads: env {i} differs from env 0 (clone did not carry the pad physics)")
    t = _params(env, params)
    if t.pad_object_offsets:
        from pxr import PhysxSchema, Usd, UsdPhysics
        for i in range(n):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(env.object.cfg.prim_path.replace("env_.*", f"env_{i}"))):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    pc.CreateContactOffsetAttr().Set(float(t.object_contact_offset))
                    pc.CreateRestOffsetAttr().Set(float(t.object_rest_offset))
    print(f"[tacmap-original] check OK: {n} envs match env 0 (5 pads soft); checked envs {sorted({0, 1, n // 2, n - 1} & set(range(n)))}", flush=True)
    return dict(envs=n, checked=True)


def install_before_clone(env, params: dict | None = None):
    """Wrap env.scene.clone_environments so the pad physics is authored on env 0 right before the clone. Returns a
    finish() to call after setup_scene: it authors all envs directly if no clone happened, then runs check()."""
    orig = env.scene.clone_environments
    state = {}

    def wrapped(*a, **kw):
        p = dict(params) if params is not None else None
        if p is not None:
            p["pad_object_offsets"] = False   # objects are handled per env in check()
        state["info"] = apply_original_tacmap_pads(env, p, env_ids=[0]) if p is not None else _env0(env)
        return orig(*a, **kw)

    def _env0(e):
        from types import SimpleNamespace
        t = e.cfg.tactile
        d = {k_: getattr(t, k_) for k_ in ORIGINAL}
        d["pad_object_offsets"] = False
        return apply_original_tacmap_pads(e, d, env_ids=[0])

    env.scene.clone_environments = wrapped

    def finish():
        env.scene.clone_environments = orig
        if "info" not in state:   # no clone (replicate_physics False): every env already exists
            state["info"] = apply_original_tacmap_pads(env, params, env_ids=None)
        state["info"]["check"] = check(env, params)
        return state["info"]
    return finish


def bake_into_robot_usd(usd_path: str, params) -> str:
    """Write the pad physics into a copy of the (baked) robot USD: materials inside the asset, bindings + offsets on
    its collision shapes. Every env references the copy, so there is no per-env authoring (12k envs in seconds).
    Returns the copy's path (cached by content: same input file + params -> same copy)."""
    import hashlib, json, os, tempfile
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics, UsdShade
    t, a = params["t"], params["a"]
    key = dict(src=os.path.abspath(usd_path), mtime=os.path.getmtime(usd_path), k=float(t.pad_compliant_stiffness),
               c=float(t.pad_compliant_damping), pf=float(t.pad_friction), tf=float(a.finger_tip_friction), rf=float(a.robot_friction),
               ro=bool(t.pad_robot_offsets), co=float(t.robot_contact_offset), rr=float(t.robot_rest_offset), v=2)
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    out = os.path.join(os.path.dirname(usd_path), f"{os.path.splitext(os.path.basename(usd_path))[0]}_tacmap_{h}.usd")
    if os.path.exists(out):
        return out
    st = Usd.Stage.Open(usd_path)
    root = st.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"tactile.original_pads: {usd_path} has no default prim")
    k, c = float(t.pad_compliant_stiffness), float(t.pad_compliant_damping)
    pad_mu = float(a.finger_tip_friction) if float(t.pad_friction) < 0 else float(t.pad_friction)

    def _mat(name, mu, kk=0.0, cc=0.0):
        m = UsdShade.Material.Define(st, root.GetPath().AppendChild(name))
        UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        pm = UsdPhysics.MaterialAPI(m.GetPrim())
        pm.CreateStaticFrictionAttr(float(mu)); pm.CreateDynamicFrictionAttr(float(mu)); pm.CreateRestitutionAttr(0.0)
        if kk > 0.0:
            px = PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim())
            px.CreateCompliantContactStiffnessAttr(kk); px.CreateCompliantContactDampingAttr(cc)
        return m
    tips = tuple(f"left_{f}_DP" for f in ("thumb", "index", "middle", "ring", "pinky"))
    mats = dict(pad=_mat("TacmapPadMat", pad_mu, k, c), tip=_mat("TacmapTipShellMat", a.finger_tip_friction),
                rob=_mat("TacmapRobotMat", a.robot_friction))
    npad = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if False else Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        p = str(prim.GetPath())
        if t.pad_robot_offsets:
            pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            pc.CreateContactOffsetAttr().Set(float(t.robot_contact_offset)); pc.CreateRestOffsetAttr().Set(float(t.robot_rest_offset))
        kd = "pad" if _is_pad(p) else ("tip" if any(f"/{n}/" in p + "/" for n in tips) else "rob")
        npad += kd == "pad"
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mats[kd], UsdShade.Tokens.strongerThanDescendants, "physics")
    if npad != 5:
        raise RuntimeError(f"tactile.original_pads: robot USD {usd_path} has {npad} gel pad collision shapes (want 5)")
    fd, tmp = tempfile.mkstemp(suffix=".usd", dir=os.path.dirname(usd_path)); os.close(fd)
    st.GetRootLayer().Export(tmp)
    os.replace(tmp, out)
    print(f"[tacmap-original] baked pad physics into {out} (5 pads soft k={k} d={c} mu={pad_mu})", flush=True)
    return out


def install_usd_bake(env, scene_module, params: dict | None = None):
    """Make ``scene_module.build_robot_articulation_usd_cfg`` load a copy of the robot USD with the pad physics baked
    in. Returns finish() for after setup_scene: authors object offsets per env if on, runs check()."""
    t = _params(env, params)
    orig = scene_module.build_robot_articulation_usd_cfg
    state = {}

    def wrapped(usd_path, *a, **kw):
        state["usd"] = bake_into_robot_usd(usd_path, dict(t=t, a=env.cfg.assets))
        return orig(state["usd"], *a, **kw)
    scene_module.build_robot_articulation_usd_cfg = wrapped

    def finish():
        scene_module.build_robot_articulation_usd_cfg = orig
        if "usd" not in state:
            raise RuntimeError("tactile.original_pads: the robot USD was not built through build_robot_articulation_usd_cfg")
        env._finger_compliant = True   # the runtime tensor friction pass would replace the bound materials (A10 logic)
        info = dict(robot_usd=state["usd"], k=float(t.pad_compliant_stiffness), c=float(t.pad_compliant_damping))
        info["check"] = check(env, params)
        return info
    return finish
