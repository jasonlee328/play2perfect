"""Register FMB-shaped straight peg-insertion problems (``fmb.<shape>.tol0p5mm``).

Same convention as ``tight_insertion`` (Lpeg_matchedmass.tol0p5mm): hole block root at its bottom face
(10 mm base slab + 50 mm walls), peg long axis +x pointing up at the insert pose (-90 deg about y), two
goals: pre-insert 5 cm above the final pose (peg tip 1 mm above the hole top), then the final pose with the
peg tip 1 mm above the hole floor (49 mm inserted). The goal fixes the yaw; for non-circular shapes the hole
only admits the peg within a few degrees of it (or of a symmetric copy, which the keypoint goal check does
not credit).
"""

from pathlib import Path

from evaluation.objects import NAME_TO_OBJECT
from evaluation.problems import PROBLEM_REGISTRY, Problem, make_pre_insert_sequence

from .objects import FMB_SHAPES

PEG_LENGTH = 0.150
HOLE_FLOOR_Z = 0.010
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Peg center 1 mm + half a length above the hole floor, body +x up (same quaternion as the L-peg).
FMB_INSERT_POSE = (0.0, 0.0, HOLE_FLOOR_Z + 0.001 + PEG_LENGTH / 2, 0.0, -0.70710678, 0.0, 0.70710678)


def _register_all() -> None:
    for shape in FMB_SHAPES:
        obj = f"fmb_{shape}_peg"
        urdf_rel = f"urdf/fmb_insertion/{shape}/hole_tol0p5mm/fmb_{shape}_hole_tol0p5mm.urdf"
        if obj not in NAME_TO_OBJECT or not (_REPO_ROOT / "assets" / urdf_rel).exists():
            continue
        name = f"fmb.{shape}.tol0p5mm"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name=obj,
            receptive_urdf=urdf_rel,
            insert_pose_rel_receptive=make_pre_insert_sequence(FMB_INSERT_POSE),
        )


_register_all()
