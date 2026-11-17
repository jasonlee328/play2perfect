"""Register the A3 force-window objects / problems from assets/urdf/force_window/meta.json
(written by FoundationTouch tools/force_window/make_fw_assets.py). Problems:

- ``fw_fragile_m<g>``: lift the 66 x 100 mm can (mass g grams) 10 cm off the coaster.
- ``fw_fragile``: alias of ``fw_fragile_m110`` (the hidden per-env mass comes from ``force_window.mass_values_kg``).
- ``fw_press``: the rocker key (revolute joint to the base, spring from meta; the per-env hidden spring comes from
  ``force_window.key_k_values``). Its waypoint is unreachable (1 m up) so the base goal never fires: success is the
  force-window rule in ``fw_env.ForceWindowEnv``.
"""

import json

from evaluation.objects import NAME_TO_OBJECT, Object, get_repo_root_dir
from evaluation.problems import JointSpec, PROBLEM_REGISTRY, Problem

ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "force_window"
META = json.loads((ASSETS_DIR / "meta.json").read_text()) if (ASSETS_DIR / "meta.json").is_file() else None


def _register() -> None:
    if META is None:
        return
    for name, m in META["objects"].items():
        urdf = ASSETS_DIR / "objects" / name / f"{name}.urdf"
        if urdf.is_file():
            NAME_TO_OBJECT[name] = Object(urdf_path=urdf, scale=tuple(float(v) for v in m["object_scale"]),
                                          need_vhacd=False)
    for pname, p in META["problems"].items():
        if p["object"] not in NAME_TO_OBJECT:
            continue
        PROBLEM_REGISTRY[pname] = Problem(
            name=pname,
            insertion_object_name=p["object"],
            receptive_urdf=f"urdf/force_window/fixtures/{p['fixture']}/{p['fixture']}.urdf",
            hole_z_offset=float(p.get("hole_z_offset", 0.0)),
            object_start_pose_rel_receptive=tuple(p["object_start_pose"]),
            joints=tuple(JointSpec(**{**jd, "anchor": tuple(jd["anchor"]), "axis": tuple(jd["axis"])})
                         for jd in p.get("joints", ())),
            insert_pose_rel_receptive=tuple(tuple(w) for w in p["waypoints"]),
        )
    if "fw_fragile_m110" in PROBLEM_REGISTRY:
        import dataclasses
        PROBLEM_REGISTRY["fw_fragile"] = dataclasses.replace(PROBLEM_REGISTRY["fw_fragile_m110"], name="fw_fragile")
    press = [k for k in PROBLEM_REGISTRY if k.startswith("fw_press_k")]
    if press:
        import dataclasses
        base = PROBLEM_REGISTRY[sorted(press)[0]]
        far = tuple((w[0], w[1], w[2] + 1.0) + tuple(w[3:]) for w in base.insert_pose_rel_receptive)
        PROBLEM_REGISTRY["fw_press"] = dataclasses.replace(base, name="fw_press", insert_pose_rel_receptive=far)


_register()
