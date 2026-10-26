"""Register ContactBench problems (``contactbench.<name>``).

``cb_plug<k>``: task 23 "plug in charger", charging block k into the outlet (rigid shell, no flexure spring;
fixture root at the centre of its bottom face, on the table). Two goals like tight_insertion: pre-insert 3 cm
above the seated pose (blade tips about 14 mm above the outlet face), then seated (plug face on the outlet face,
blades 16 mm deep for block 1). ``cb_plug<k>_rigidspring``: the same with the 0.7 mm-indent spring fused into
the fixture as a rigid body (blades cannot enter; a check of what a rigid sim does with the flexure).
``cb_card_deck<n>``: task 24 "place card into holder", the card standing upright in the holder's channel against
n cards already there (fused into the fixture as a rigid deck; n = 0, 4, 8); pre-insert 5 cm above.
Placing under constraints (FoundationTouch tools/contactbench/place_tasks.py): ``cb_hang_<obj>`` (three explicit
waypoints onto the board's hook), ``cb_cupstack_n<k>``, ``cb_binv_<utensil>_n<k>``.
"""

import json
from pathlib import Path

from evaluation.objects import NAME_TO_OBJECT
from evaluation.problems import PROBLEM_REGISTRY, Problem, make_pre_insert_sequence

from .objects import ASSETS_DIR

PRE_INSERT_OFFSET = 0.03


def _register_all() -> None:
    meta_path = ASSETS_DIR / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text())
    for pname, p in meta["problems"].items():
        obj = f"cb_{p['object']}"
        variants = [("", p["fixture"])]
        if p["fixture"] == "outlet":
            variants.append(("_rigidspring", "outlet_rigidspring"))
        for suffix, fixture in variants:
            urdf_rel = f"urdf/contactbench/fixtures/{fixture}/{fixture}.urdf"
            if obj not in NAME_TO_OBJECT or not (ASSETS_DIR.parent.parent / urdf_rel).exists():
                continue
            name = f"contactbench.{pname}{suffix}"
            PROBLEM_REGISTRY[name] = Problem(
                name=name,
                insertion_object_name=obj,
                receptive_urdf=urdf_rel,
                insert_pose_rel_receptive=tuple(tuple(w) for w in p["waypoints"]) if "waypoints" in p else
                make_pre_insert_sequence(tuple(p["seated_pose"]),
                                         pre_insert_offset=p.get("pre_insert_offset", PRE_INSERT_OFFSET)),
            )


_register_all()
