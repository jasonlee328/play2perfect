"""Register FurnitureBench Problem entries.

Currently registers sparse and dense ``furniture_bench.one_leg`` variants
(insert square_table_leg4 into one corner of the square_table_top). The
receptive URDF + per-task setup metadata are produced by
``one_leg_problem_setup/step2_generate_assets.py``.

Pose composition (see step2 for the canonical conventions):
  * Receivers: ``R_storage_to_canonical = R_x(+90°)``. Loaded at world
    origin with identity URDF rotation; top surface at world +Z.
  * Inserters: ``R_storage_to_canonical = R_y(+90°) ∘ R_x(+90°)``.
    Long axis on canonical X, tenon at canonical -X.

For the inserter URDF pose relative to the receiver:
  * pos_canonical = R_x(+90°) ∘ pos_yup
  * q_urdf = R_x(+90°) ∘ R_yup_relative ∘ R_storage_to_canonical_inserter^-1
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from evaluation.objects import NAME_TO_OBJECT
from evaluation.problems import (
    PROBLEM_REGISTRY,
    Problem,
    make_transport_pre_insert_sequence,
)

_LOG = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ASSETS_FB = _REPO_ROOT / "assets" / "urdf" / "furniture_bench"

R_X90 = R.from_euler("x", 90, degrees=True)
ONE_LEG_PRE_INSERT_OFFSET_M = 0.025
ONE_LEG_THREAD_PITCH_M = 0.00937368684342171
BULB_PRE_INSERT_OFFSET_M = 0.030
BULB_THREAD_PITCH_M = 0.00937368684342171  # same as one_leg until tuned


def _one_leg_screw_insert_waypoints(final_pose, turns):
    """Return lead-in + screw waypoints at the requested turns + final pose."""
    final_pose = tuple(float(v) for v in final_pose)
    final_pos = np.asarray(final_pose[:3], dtype=float)
    final_quat = np.asarray(final_pose[3:7], dtype=float)
    final_rot = R.from_quat(final_quat)
    insertion_dir = np.asarray((0.0, 0.0, -1.0), dtype=float)

    waypoints = []

    # Lead-in: 25 mm above final with the same orientation. This preserves the
    # existing one-leg pre-insert heuristic while keeping it explicit in the
    # problem's subgoal list.
    lead_pos = final_pos - insertion_dir * ONE_LEG_PRE_INSERT_OFFSET_M
    waypoints.append((*lead_pos.tolist(), *final_quat.tolist()))

    # Screw guidance. Full-turn waypoints are intentionally the same SO(3)
    # orientation as final; their ordering/z offset carries phase.
    for turn in turns:
        backoff = float(turn) * ONE_LEG_THREAD_PITCH_M
        pos = final_pos - insertion_dir * backoff
        quat = (R.from_euler("z", 360.0 * float(turn), degrees=True) * final_rot).as_quat()
        if float(np.dot(quat, final_quat)) < 0.0:
            quat = -quat
        waypoints.append((*pos.tolist(), *quat.tolist()))

    waypoints.append(final_pose)
    return tuple(waypoints)


def _one_leg_dense_insert_waypoints(final_pose):
    """Lead-in + screw waypoints every 180° (0.5 turn step) + final."""
    return _one_leg_screw_insert_waypoints(final_pose, turns=(2.0, 1.5, 1.0, 0.5))


def _one_leg_super_dense_insert_waypoints(final_pose):
    """Lead-in + screw waypoints every 90° (0.25 turn step) + final."""
    return _one_leg_screw_insert_waypoints(
        final_pose, turns=(2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
    )


def _one_leg_semi_dense_insert_waypoints(final_pose):
    """Return lead-in + one-turn screw waypoints + final assembled pose."""
    return _one_leg_screw_insert_waypoints(final_pose, turns=(1.0, 0.5))


def _one_leg_flat_dense_insert_waypoints(
    final_pose, pre_insert_above_15t_m: float = 0.005
):
    """Waypoint sequence tuned for the flat receiver variant.

    Differs from `super_dense`:
      - Drops the 2.0-turn waypoint (with a continuous flat surface
        around the hole, the leg doesn't need a starting waypoint that
        far above the assembled pose).
      - The pre-insert / lead-in pose is positioned at the 1.5-turn
        height plus a small ``pre_insert_above_15t_m`` margin (default
        5 mm), with the SAME orientation as the 1.5T screw waypoint
        (i.e. pre-rotated +540° from final) so the policy can transition
        from pre_insert straight into screwing without a sudden rotation.

    Order: [pre_insert, 1.5T, 1.25T, 1.0T, 0.75T, 0.5T, 0.25T, final].
    """
    final_pose = tuple(float(v) for v in final_pose)
    final_pos = np.asarray(final_pose[:3], dtype=float)
    final_quat = np.asarray(final_pose[3:7], dtype=float)
    final_rot = R.from_quat(final_quat)
    insertion_dir = np.asarray((0.0, 0.0, -1.0), dtype=float)

    # Orientation pre-rotated by +1.5 turns from final, shared by both the
    # pre_insert waypoint and the 1.5T screw waypoint.
    first_turn_quat = (R.from_euler("z", 360.0 * 1.5, degrees=True) * final_rot).as_quat()
    if float(np.dot(first_turn_quat, final_quat)) < 0.0:
        first_turn_quat = -first_turn_quat

    waypoints = []
    # Pre-insert: 1.5T height + margin, oriented same as the 1.5T waypoint.
    pre_insert_backoff = 1.5 * ONE_LEG_THREAD_PITCH_M + float(pre_insert_above_15t_m)
    pre_insert_pos = final_pos - insertion_dir * pre_insert_backoff
    waypoints.append((*pre_insert_pos.tolist(), *first_turn_quat.tolist()))

    # Super-dense screw waypoints from 1.5 down to 0.25 (0.25-turn step).
    for turn in (1.5, 1.25, 1.0, 0.75, 0.5, 0.25):
        backoff = float(turn) * ONE_LEG_THREAD_PITCH_M
        pos = final_pos - insertion_dir * backoff
        quat = (R.from_euler("z", 360.0 * float(turn), degrees=True) * final_rot).as_quat()
        if float(np.dot(quat, final_quat)) < 0.0:
            quat = -quat
        waypoints.append((*pos.tolist(), *quat.tolist()))

    waypoints.append(final_pose)
    return tuple(waypoints)


def _register_problem_variant(
    *,
    name: str,
    insertion_object_name: str,
    receptive_urdf: str,
    insert_poses,
    hole_z_offset: float,
    prelude_lift_offset: float = 0.0,
) -> None:
    PROBLEM_REGISTRY[name] = Problem(
        name=name,
        insertion_object_name=insertion_object_name,
        receptive_urdf=receptive_urdf,
        insert_pose_rel_receptive=insert_poses,
        hole_z_offset=hole_z_offset,
        # One 25 mm thread/tenon length: with insertion_direction=(0,0,-1),
        # the pre-insert pose starts with the threaded tip at the hole entrance.
        pre_insert_offset=ONE_LEG_PRE_INSERT_OFFSET_M,
        prelude_lift_offset=prelude_lift_offset,
    )


def _register_one_leg(
    piece: str = "square_table",
    problem_suffix: str = "",
    setup_filename: str = "one_leg_setup.json",
) -> None:
    """Register the one_leg insertion problem variants for ``piece``.

    ``piece`` selects the source dir under ``assets/urdf/furniture_bench/``
    (``square_table`` for the base assets, ``square_table_2x`` for the
    scaled clone generated by ``scene_generation/scale_one_leg.py``).
    ``problem_suffix`` is appended to every registered problem name
    (e.g. ``_2x`` → ``furniture_bench.one_leg_2x``).
    ``setup_filename`` lets multiple inserter variants share a piece dir —
    e.g. ``one_leg_setup_leg4_125mm.json`` for the longer leg with the same
    receiver, only differing in ``child_part`` and ``pos_canonical_centroid``.
    """
    setup_path = _ASSETS_FB / piece / setup_filename
    assembly_path = _ASSETS_FB / piece / "assembly.json"
    if not setup_path.is_file() or not assembly_path.is_file():
        _LOG.info("furniture_bench.one_leg%s: missing setup/assembly JSON — skipping",
                  problem_suffix)
        return

    setup = json.loads(setup_path.read_text())
    assembly = json.loads(assembly_path.read_text())

    # Inserter Object key (registered by objects.py).
    child_part = setup["child_part"]
    inserter_key = f"furniture_bench_{child_part}_coacd"
    has_coacd = inserter_key in NAME_TO_OBJECT
    if not has_coacd:
        _LOG.info("furniture_bench.one_leg%s: %s not in NAME_TO_OBJECT — "
                  "skipping CoACD variants (sdf_hybrid still registers)",
                  problem_suffix, inserter_key)

    # Read the inserter's R_storage_to_canonical from canonical_meta.json so
    # we never drift out of sync with what step2 actually applied.
    meta_path = _ASSETS_FB / piece / child_part / "canonical_meta.json"
    if not meta_path.is_file():
        _LOG.warning("furniture_bench.one_leg%s: missing %s — skipping",
                     problem_suffix, meta_path)
        return
    meta = json.loads(meta_path.read_text())
    P_inserter = np.asarray(meta["R_storage_to_canonical"], dtype=float)

    # Receptive URDF path (relative to assets/).
    recv_rel = setup["receptive_urdf"]

    # Step2 already applied the OBJ-origin → bbox-centroid offset
    # correction and rotated to canonical frame, so just use it.
    if "pos_canonical_centroid" not in setup:
        _LOG.warning(
            "furniture_bench.one_leg: setup JSON predates the centroid-offset "
            "fix — re-run step2_generate_assets.py. Skipping.",
        )
        return
    pos_canonical = np.asarray(setup["pos_canonical_centroid"], dtype=float)
    rpy_yup = np.asarray(setup["active_corner_rpy_yup"], dtype=float)

    # q_urdf = R_x(+90°) ∘ R_yup_relative ∘ Pᵀ_inserter.
    R_yup_rel = R.from_euler("xyz", rpy_yup)
    q = R_X90 * R_yup_rel * R.from_matrix(P_inserter.T)
    qx, qy, qz, qw = (float(v) for v in q.as_quat())

    # hole_z_offset: the receiver's bottom face should sit on the table
    # surface in the env. The receiver's canonical mesh is centered, so
    # half its Z extent puts the bottom face at z=0 in world.
    # square_table_top z extent ≈ 31.3 mm; offset = 0.0156 m. Scaled clones
    # carry larger extents, computed below from the actual mesh.
    import trimesh
    top_canonical = trimesh.load(
        str(_ASSETS_FB / piece / setup["parent_part"]
            / f"{setup['parent_part']}_canonical.obj"),
        force="mesh", process=False,
    )
    hole_z_offset = float((top_canonical.bounds[1, 2] - top_canonical.bounds[0, 2]) / 2)

    final_pose = (
        float(pos_canonical[0]), float(pos_canonical[1]), float(pos_canonical[2]),
        qx, qy, qz, qw,
    )
    dense_waypoints = _one_leg_dense_insert_waypoints(final_pose)
    super_dense_waypoints = _one_leg_super_dense_insert_waypoints(final_pose)
    semi_dense_waypoints = _one_leg_semi_dense_insert_waypoints(final_pose)
    sparse_waypoints = (dense_waypoints[0], dense_waypoints[-1])
    # dense_traj: 3 hole-frame waypoints (transport_above, pre_insert, final)
    # paired at problem-config time with a 0.20 m per-episode prelude
    # (lift-in-place + over-hole) that the env constructs at reset.
    dense_traj_waypoints = make_transport_pre_insert_sequence(
        final_pose,
        pre_insert_offset=ONE_LEG_PRE_INSERT_OFFSET_M,
        transport_offset=0.10,
    )

    base_name = f"furniture_bench.one_leg{problem_suffix}"
    if has_coacd:
        for variant_name, insert_poses in (
            (base_name, dense_waypoints),
            (f"{base_name}_super_dense", super_dense_waypoints),
            (f"{base_name}_dense", dense_waypoints),
            (f"{base_name}_semi_dense", semi_dense_waypoints),
            (f"{base_name}_sparse", sparse_waypoints),
        ):
            _register_problem_variant(
                name=variant_name,
                insertion_object_name=inserter_key,
                receptive_urdf=recv_rel,
                insert_poses=insert_poses,
                hole_z_offset=hole_z_offset,
            )

    hybrid_recv = (
        _ASSETS_FB / piece / "insertion_fixtures"
        / "one_leg_sdf_hybrid.urdf"
    )
    hybrid_key = f"furniture_bench_{child_part}_sdf_hybrid"
    if hybrid_recv.is_file() and hybrid_key in NAME_TO_OBJECT:
        hybrid_base_name = f"{base_name}_sdf_hybrid"
        hybrid_recv_rel = hybrid_recv.relative_to(_REPO_ROOT / "assets").as_posix()
        for variant_name, insert_poses in (
            (hybrid_base_name, dense_waypoints),
            (f"{hybrid_base_name}_super_dense", super_dense_waypoints),
            (f"{hybrid_base_name}_dense", dense_waypoints),
            (f"{hybrid_base_name}_semi_dense", semi_dense_waypoints),
            (f"{hybrid_base_name}_sparse", sparse_waypoints),
        ):
            _register_problem_variant(
                name=variant_name,
                insertion_object_name=hybrid_key,
                receptive_urdf=hybrid_recv_rel,
                insert_poses=insert_poses,
                hole_z_offset=hole_z_offset,
            )

    # Matched-mass siblings (CoACD + SDF-hybrid). Same receptive URDF, same
    # poses; only the inserter object is swapped to its `_matchedmass_*`
    # variant so the grasped leg carries measured mass + mesh-derived inertia.
    matched_coacd_key = f"furniture_bench_{child_part}_matchedmass_coacd"
    if matched_coacd_key in NAME_TO_OBJECT:
        matched_base = f"{base_name}_matchedmass"
        for variant_name, insert_poses in (
            (matched_base, dense_waypoints),
            (f"{matched_base}_super_dense", super_dense_waypoints),
            (f"{matched_base}_dense", dense_waypoints),
            (f"{matched_base}_semi_dense", semi_dense_waypoints),
            (f"{matched_base}_sparse", sparse_waypoints),
        ):
            _register_problem_variant(
                name=variant_name,
                insertion_object_name=matched_coacd_key,
                receptive_urdf=recv_rel,
                insert_poses=insert_poses,
                hole_z_offset=hole_z_offset,
            )
    matched_sdfh_key = f"furniture_bench_{child_part}_matchedmass_sdf_hybrid"
    if hybrid_recv.is_file() and matched_sdfh_key in NAME_TO_OBJECT:
        matched_sdfh_base = f"{base_name}_matchedmass_sdf_hybrid"
        hybrid_recv_rel = hybrid_recv.relative_to(_REPO_ROOT / "assets").as_posix()
        for variant_name, insert_poses in (
            (matched_sdfh_base, dense_waypoints),
            (f"{matched_sdfh_base}_super_dense", super_dense_waypoints),
            (f"{matched_sdfh_base}_dense", dense_waypoints),
            (f"{matched_sdfh_base}_semi_dense", semi_dense_waypoints),
            (f"{matched_sdfh_base}_sparse", sparse_waypoints),
        ):
            _register_problem_variant(
                name=variant_name,
                insertion_object_name=matched_sdfh_key,
                receptive_urdf=hybrid_recv_rel,
                insert_poses=insert_poses,
                hole_z_offset=hole_z_offset,
            )
        # dense_traj: 3 hole-frame waypoints + 0.20 m per-episode prelude
        # constructed by the env at reset time.
        _register_problem_variant(
            name=f"{matched_sdfh_base}_dense_traj",
            insertion_object_name=matched_sdfh_key,
            receptive_urdf=hybrid_recv_rel,
            insert_poses=dense_traj_waypoints,
            hole_z_offset=hole_z_offset,
            prelude_lift_offset=0.20,
        )
        # Flat receiver variant: same matched-mass inserter, but the
        # receptive is a 45 mm-diameter cylindrical boss around the active
        # hole (no rim walls / floor slab / inactive cylinders). Useful for
        # studying slide-into-hole behavior with a continuous flat surface.
        flat_recv = (
            _ASSETS_FB / piece / "insertion_fixtures"
            / "one_leg_sdf_hybrid_flat.urdf"
        )
        if flat_recv.is_file():
            flat_recv_rel = flat_recv.relative_to(_REPO_ROOT / "assets").as_posix()
            _register_problem_variant(
                name=f"{matched_sdfh_base}_flat",
                insertion_object_name=matched_sdfh_key,
                receptive_urdf=flat_recv_rel,
                insert_poses=sparse_waypoints,
                hole_z_offset=hole_z_offset,
            )
            # Flat + custom dense screw waypoints: pre-insert at 1.5T + 5 mm
            # with final orientation, then super-dense (every 0.25T) from
            # 1.5T down to 0.25T, then final. No 2.0T waypoint.
            flat_dense_waypoints = _one_leg_flat_dense_insert_waypoints(final_pose)
            _register_problem_variant(
                name=f"{matched_sdfh_base}_flat_dense",
                insertion_object_name=matched_sdfh_key,
                receptive_urdf=flat_recv_rel,
                insert_poses=flat_dense_waypoints,
                hole_z_offset=hole_z_offset,
            )


_register_one_leg()
_register_one_leg(piece="square_table_1p5x", problem_suffix="_1p5x")
_register_one_leg(piece="square_table_2x", problem_suffix="_2x")
# Longer-body leg4 variants (same receiver, same thread, same hole).
_register_one_leg(
    setup_filename="one_leg_setup_leg4_125mm.json",
    problem_suffix="_leg4_125mm",
)
_register_one_leg(
    setup_filename="one_leg_setup_leg4_200mm.json",
    problem_suffix="_leg4_200mm",
)


# ─── Bulb-into-base (lamp assembly) ────────────────────────────────────────

def _bulb_screw_waypoints(final_pose, turns):
    """Same shape as one_leg waypoints, but parameterized on bulb constants."""
    final_pose = tuple(float(v) for v in final_pose)
    final_pos = np.asarray(final_pose[:3], dtype=float)
    final_quat = np.asarray(final_pose[3:7], dtype=float)
    final_rot = R.from_quat(final_quat)
    insertion_dir = np.asarray((0.0, 0.0, -1.0), dtype=float)

    waypoints = []
    lead_pos = final_pos - insertion_dir * BULB_PRE_INSERT_OFFSET_M
    waypoints.append((*lead_pos.tolist(), *final_quat.tolist()))
    for turn in turns:
        backoff = float(turn) * BULB_THREAD_PITCH_M
        pos = final_pos - insertion_dir * backoff
        quat = (R.from_euler("z", 360.0 * float(turn), degrees=True) * final_rot).as_quat()
        if float(np.dot(quat, final_quat)) < 0.0:
            quat = -quat
        waypoints.append((*pos.tolist(), *quat.tolist()))
    waypoints.append(final_pose)
    return tuple(waypoints)


def _register_bulb_screw() -> None:
    setup_path = _ASSETS_FB / "lamp" / "bulb_screw_setup.json"
    assembly_path = _ASSETS_FB / "lamp" / "assembly.json"
    if not setup_path.is_file() or not assembly_path.is_file():
        _LOG.info("furniture_bench.bulb_screw: missing setup/assembly JSON — skipping")
        return

    setup = json.loads(setup_path.read_text())
    child_part = setup["child_part"]

    # Only the sdf_hybrid variant is registered (step2 does not produce a
    # non-hybrid receptive URDF for the bulb — the lamp_base mesh isn't a
    # flat plate, so the synthetic-slab visual doesn't apply).
    hybrid_recv = _ASSETS_FB / "lamp" / "insertion_fixtures" / "bulb_screw_sdf_hybrid.urdf"
    hybrid_key = f"furniture_bench_{child_part}_sdf_hybrid"
    if not hybrid_recv.is_file() or hybrid_key not in NAME_TO_OBJECT:
        _LOG.warning(
            "furniture_bench.bulb_screw: missing sdf_hybrid asset (%s, %s) -- skipping",
            hybrid_recv, hybrid_key,
        )
        return

    meta_path = _ASSETS_FB / "lamp" / child_part / "canonical_meta.json"
    if not meta_path.is_file():
        _LOG.warning("furniture_bench.bulb_screw: missing %s -- skipping", meta_path)
        return
    meta = json.loads(meta_path.read_text())
    P_inserter = np.asarray(meta["R_storage_to_canonical"], dtype=float)

    # Read the receiver's R from canonical_meta.json so the q_urdf composition
    # picks up the lamp_base's R_x(-90°) override (instead of the default
    # R_x(+90°) used by one_leg).
    recv_meta_path = _ASSETS_FB / "lamp" / setup["parent_part"] / "canonical_meta.json"
    if not recv_meta_path.is_file():
        _LOG.warning("furniture_bench.bulb_screw: missing %s -- skipping", recv_meta_path)
        return
    P_receiver = np.asarray(
        json.loads(recv_meta_path.read_text())["R_storage_to_canonical"], dtype=float
    )

    pos_canonical = np.asarray(setup["pos_canonical_centroid"], dtype=float)
    rpy_yup = np.asarray(setup["active_corner_rpy_yup"], dtype=float)
    R_yup_rel = R.from_euler("xyz", rpy_yup)
    q = R.from_matrix(P_receiver) * R_yup_rel * R.from_matrix(P_inserter.T)
    qx, qy, qz, qw = (float(v) for v in q.as_quat())

    import trimesh
    base_canonical = trimesh.load(
        str(_ASSETS_FB / "lamp" / setup["parent_part"]
            / f"{setup['parent_part']}_canonical.obj"),
        force="mesh", process=False,
    )
    hole_z_offset = float((base_canonical.bounds[1, 2] - base_canonical.bounds[0, 2]) / 2)

    final_pose = (
        float(pos_canonical[0]), float(pos_canonical[1]), float(pos_canonical[2]),
        qx, qy, qz, qw,
    )
    dense_waypoints      = _bulb_screw_waypoints(final_pose, turns=(2.0, 1.5, 1.0, 0.5))
    super_dense_waypoints = _bulb_screw_waypoints(
        final_pose, turns=(2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25)
    )
    semi_dense_waypoints = _bulb_screw_waypoints(final_pose, turns=(1.0, 0.5))
    sparse_waypoints     = (dense_waypoints[0], dense_waypoints[-1])

    base_name = "furniture_bench.bulb_screw_sdf_hybrid"
    recv_rel = hybrid_recv.relative_to(_REPO_ROOT / "assets").as_posix()
    for variant_name, insert_poses in (
        (base_name, dense_waypoints),
        (f"{base_name}_super_dense", super_dense_waypoints),
        (f"{base_name}_dense", dense_waypoints),
        (f"{base_name}_semi_dense", semi_dense_waypoints),
        (f"{base_name}_sparse", sparse_waypoints),
    ):
        PROBLEM_REGISTRY[variant_name] = Problem(
            name=variant_name,
            insertion_object_name=hybrid_key,
            receptive_urdf=recv_rel,
            insert_pose_rel_receptive=insert_poses,
            hole_z_offset=hole_z_offset,
            pre_insert_offset=BULB_PRE_INSERT_OFFSET_M,
        )


_register_bulb_screw()
