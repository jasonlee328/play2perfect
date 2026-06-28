"""Register peg-in-hole Problem entries.

One Problem per hole tolerance variant under
``assets/urdf/peg_in_hole/holes/hole_tol*/``. All variants share the same
insertion object (``peg``) and the same canonical insert pose; only the
receptive (hole) URDF differs.
"""

from pathlib import Path

from evaluation.problems import (
    PROBLEM_REGISTRY,
    Problem,
    make_pre_insert_sequence,
    make_transport_pre_insert_sequence,
)


# Canonical pose: peg handle center 13.6 cm above hole base, body +X
# pointing down (-90 deg about Y, xyzw quaternion).
_PEG_INSERT_POSE = (
    0.0, 0.0, 0.136,
    0.0, -0.70710678, 0.0, 0.70710678,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOLES_DIR = _REPO_ROOT / "assets" / "urdf" / "peg_in_hole" / "holes"


def _dense_insert_sequence(final_pose, pre_insert_offsets_m=(0.15, 0.05)):
    """3+ pose insertion sequence with multiple pre-insert waypoints above
    the final pose. Each entry in ``pre_insert_offsets_m`` is a metric
    distance above the final pose (assumes vertical insertion); poses are
    emitted from farthest to closest, then the final pose. So the default
    ``(0.15, 0.05)`` yields (15 cm above, 5 cm above, final).
    """
    x, y, z, qx, qy, qz, qw = (float(v) for v in final_pose)
    poses = []
    for offset in sorted(pre_insert_offsets_m, reverse=True):
        poses.append((x, y, z + float(offset), qx, qy, qz, qw))
    poses.append((x, y, z, qx, qy, qz, qw))
    return tuple(poses)


# Receptive-frame coordinates for the peg's canonical start pose (computed
# from peg world start (0.135, 0.08, 0.54), default hole world position
# (-0.05, 0.08, 0.53) -> peg-start-rel-hole ~ (0.185, 0.0, 0.01)). The lift
# waypoint sits 10 cm above start at flat (identity) orientation: the same
# motion the play-only policy was trained to execute on the reorientation
# task. Subsequent waypoints carry the policy through the rotate-then-
# descend phase.
_PLAYNATURAL_LIFT_POSE = (0.185, 0.0, 0.11, 0.0, 0.0, 0.0, 1.0)


def _playnatural_insert_sequence(final_pose, pre_insert_offsets_m=(0.15, 0.05)):
    """4+ pose sequence matching the play policy's lift-then-rotate-then-
    descend natural trajectory. Order: lift-at-start (flat orientation),
    then standard pre-insert waypoints above the hole (vertical), then
    final.
    """
    poses = [_PLAYNATURAL_LIFT_POSE]
    poses.extend(_dense_insert_sequence(final_pose, pre_insert_offsets_m))
    return tuple(poses)


def _register_all() -> None:
    if not _HOLES_DIR.is_dir():
        return
    for hole_dir in sorted(_HOLES_DIR.iterdir()):
        if not hole_dir.is_dir() or not hole_dir.name.startswith("hole_tol"):
            continue
        urdf_rel = f"urdf/peg_in_hole/holes/{hole_dir.name}/{hole_dir.name}.urdf"
        if not (_REPO_ROOT / "assets" / urdf_rel).exists():
            continue
        # tag is the part after "hole_" (e.g. "tol0p5mm")
        tag = hole_dir.name[len("hole_"):]
        # (problem-prefix, object-name) pairs. `Lpeg_matchedmass` mirrors
        # `Lpeg` 1:1 (same geometry, same insert pose) but maps to the
        # corrected-mass URDF — use it to ablation-test whether policies
        # trained on the 4x-too-heavy lpeg still work at the realistic
        # 57.8 g / L-shape-inertia parameters.
        for prefix, object_name in (
            ("peg", "peg"),
            ("peg_matchedmass", "peg_matchedmass"),
            ("Lpeg", "lpeg"),
            ("Lpeg_matchedmass", "lpeg_matchedmass"),
        ):
            name = f"{prefix}.{tag}"
            PROBLEM_REGISTRY[name] = Problem(
                name=name,
                insertion_object_name=object_name,
                receptive_urdf=urdf_rel,
                insert_pose_rel_receptive=make_pre_insert_sequence(_PEG_INSERT_POSE),
            )
            # Dense variant: two pre-insert waypoints (15 cm + 5 cm above
            # final) before the final pose, giving the policy more
            # intermediate guidance during insertion. Used by Play-only
            # evals to test whether dense subgoals help an
            # never-trained-on-insertion baseline.
            dense_name = f"{prefix}_dense.{tag}"
            PROBLEM_REGISTRY[dense_name] = Problem(
                name=dense_name,
                insertion_object_name=object_name,
                receptive_urdf=urdf_rel,
                insert_pose_rel_receptive=_dense_insert_sequence(
                    _PEG_INSERT_POSE, pre_insert_offsets_m=(0.15, 0.05),
                ),
            )
            # Playnatural variant: starts with a "lift at start pose"
            # waypoint (flat orientation) matching the play-only policy's
            # natural lift-then-rotate motion, then the standard dense
            # descent. Gives the play-only baseline credit for the lift
            # phase it actually knows how to execute.
            playnatural_name = f"{prefix}_playnatural.{tag}"
            PROBLEM_REGISTRY[playnatural_name] = Problem(
                name=playnatural_name,
                insertion_object_name=object_name,
                receptive_urdf=urdf_rel,
                insert_pose_rel_receptive=_playnatural_insert_sequence(
                    _PEG_INSERT_POSE, pre_insert_offsets_m=(0.15, 0.05),
                ),
            )
            # dense_traj variant: 3 hole-frame waypoints (transport_above,
            # pre_insert, final) paired with a per-episode prelude
            # (lift-in-place + over-hole) constructed by the env at reset.
            dense_traj_name = f"{prefix}_dense_traj.{tag}"
            PROBLEM_REGISTRY[dense_traj_name] = Problem(
                name=dense_traj_name,
                insertion_object_name=object_name,
                receptive_urdf=urdf_rel,
                insert_pose_rel_receptive=make_transport_pre_insert_sequence(
                    _PEG_INSERT_POSE,
                    pre_insert_offset=0.05,
                    transport_offset=0.10,
                ),
                pre_insert_offset=0.05,
                prelude_lift_offset=0.20,
            )


_register_all()
