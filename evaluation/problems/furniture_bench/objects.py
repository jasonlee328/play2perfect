"""Register FurnitureBench parts in the global NAME_TO_OBJECT registry.

For each part that has been re-baked by
``furniture_bench/<task>_problem_setup/step2_generate_assets.py`` (i.e.
that has ``<part>/canonical_meta.json`` and a CoACD subdir), we register
the *_coacd entry. Visual-only canonical URDFs are also registered for
viewer use.
"""

import json
from pathlib import Path

import trimesh

from evaluation.objects import Object, rescale_by_factor, NAME_TO_OBJECT
from evaluation.objects import get_repo_root_dir

ASSETS_FB = get_repo_root_dir() / "assets" / "urdf" / "furniture_bench"

# Pieces that may contain re-baked parts. Mirrors the directory layout
# under assets/urdf/furniture_bench/<piece>/<part>/.
PIECES = ["cabinet", "chair", "desk", "drawer", "lamp",
          "round_table", "square_table",
          "square_table_1p5x", "square_table_2x", "stool"]


def _register_one_part(piece: str, part_name: str) -> None:
    part_dir = ASSETS_FB / piece / part_name
    canonical_obj = part_dir / f"{part_name}_canonical.obj"
    canonical_urdf = part_dir / f"{part_name}_canonical.urdf"
    coacd_urdf = part_dir / "coacd" / f"{part_name}_coacd.urdf"
    if not canonical_obj.is_file() or not canonical_urdf.is_file():
        return
    m = trimesh.load(str(canonical_obj), force="mesh", process=False)
    scale = rescale_by_factor(tuple(m.extents.tolist()), factor=25)
    key = f"furniture_bench_{part_name}"
    NAME_TO_OBJECT[f"{key}_canonical"] = Object(
        urdf_path=canonical_urdf, scale=scale, need_vhacd=False,
    )
    if coacd_urdf.is_file():
        NAME_TO_OBJECT[f"{key}_coacd"] = Object(
            urdf_path=coacd_urdf, scale=scale, need_vhacd=False,
        )
    sdf_hybrid_urdf = part_dir / "sdf_hybrid" / f"{part_name}_sdf_hybrid.urdf"
    if sdf_hybrid_urdf.is_file():
        NAME_TO_OBJECT[f"{key}_sdf_hybrid"] = Object(
            urdf_path=sdf_hybrid_urdf, scale=scale, need_vhacd=False,
        )
    # Physically-matched-mass siblings: explicit <mass>+<inertia> from
    # measured part mass (Isaac silently ignores URDF <density>).
    matched_coacd = part_dir / "coacd" / f"{part_name}_matchedmass_coacd.urdf"
    if matched_coacd.is_file():
        NAME_TO_OBJECT[f"{key}_matchedmass_coacd"] = Object(
            urdf_path=matched_coacd, scale=scale, need_vhacd=False,
        )
    matched_sdf_hybrid = part_dir / "sdf_hybrid" / f"{part_name}_matchedmass_sdf_hybrid.urdf"
    if matched_sdf_hybrid.is_file():
        NAME_TO_OBJECT[f"{key}_matchedmass_sdf_hybrid"] = Object(
            urdf_path=matched_sdf_hybrid, scale=scale, need_vhacd=False,
        )


def _scan_and_register() -> None:
    if not ASSETS_FB.is_dir():
        return
    for piece in PIECES:
        piece_dir = ASSETS_FB / piece
        if not piece_dir.is_dir():
            continue
        for part_dir in sorted(piece_dir.iterdir()):
            if not part_dir.is_dir():
                continue
            _register_one_part(piece, part_dir.name)


_scan_and_register()
