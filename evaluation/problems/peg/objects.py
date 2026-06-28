"""Register the L-peg (matched-mass) inserter in the global NAME_TO_OBJECT registry.

This release ships the single peg used by the `tight_insertion` task. The L-peg
has physically-matched mass + L-shape inertia (measured 57.8 g, two-box
parallel-axis tensor) rather than the URDF-authored solid-PLA assumption.
"""

from evaluation.objects import NAME_TO_OBJECT, Object, get_repo_root_dir, rescale_by_factor

ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "peg_in_hole"

PEG_NAME_TO_OBJECT = {
    "lpeg_matchedmass": Object(
        urdf_path=ASSETS_DIR / "lpeg_matchedmass" / "lpeg_matchedmass.urdf",
        scale=rescale_by_factor((0.25, 0.03, 0.02), factor=25),
        need_vhacd=False,
    ),
}

NAME_TO_OBJECT.update(PEG_NAME_TO_OBJECT)
