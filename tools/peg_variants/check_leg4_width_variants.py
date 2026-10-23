"""Geometric checks of the square_table_leg4 body-width variants (make_leg4_width_variants.py).

For each width: the hulls' shoulder plane and extent, the thread vs the body (x separation, thread radius vs the
body's half-width), the shoulder's support on the receiver when seated (square footprint, any rotation about the
axis, vs the hole boss: an annulus between the bore and the boss edge, measured from one_leg_hole_detail.obj), and
that nothing else of the receiver (rim walls, whose top is flush with the boss top) can be reached by the body.
Usage: python tools/peg_variants/check_leg4_width_variants.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
SQ = ROOT / "assets/urdf/furniture_bench/square_table"
LEG = SQ / "square_table_leg4"


def hulls_of(d):
    return [trimesh.load(p, force="mesh", process=False) for p in glob.glob(str(d / "body_coacd/decomp_*.obj"))]


def main():
    thread = trimesh.load(LEG / "sdf_hybrid/square_table_leg4_thread.obj", force="mesh", process=False)
    t_r = np.linalg.norm(thread.vertices[:, 1:], axis=1).max()
    t_xmax = thread.vertices[:, 0].max()
    hole = trimesh.load(SQ / "hole_patches/sdf_hybrid/one_leg_hole_detail.obj", force="mesh", process=False)
    top_z = hole.vertices[:, 2].max()
    loops = hole.section(plane_origin=[0, 0, top_z - 1e-4], plane_normal=[0, 0, 1]).discrete
    # bore centre = centre of the inner loop's bounding box; the waypoints are 0.62 mm off it (CLAUDE.md)
    inner = min(loops, key=lambda e: np.ptp(e[:, 0]))
    outer = max(loops, key=lambda e: np.ptp(e[:, 0]))
    c = (inner[:, :2].min(0) + inner[:, :2].max(0)) / 2
    r_bore = np.linalg.norm(inner[:, :2] - c, axis=1).max()
    r_boss = np.linalg.norm(outer[:, :2] - c, axis=1).min()
    rim_inner = 0.078751 - 0.0025  # rim walls (one_leg_sdf_hybrid.urdf), top z 0.015625
    d_rim = min(abs(-rim_inner - c[0]), abs(rim_inner - c[1]))
    print(f"thread: x <= {t_xmax * 1e3:.2f} mm, max radius {t_r * 1e3:.2f} mm | receiver top z {top_z * 1e3:.3f} mm, "
          f"bore centre {np.round(c, 5)}, bore r <= {r_bore * 1e3:.2f} mm, boss r >= {r_boss * 1e3:.2f} mm, "
          f"rim inner faces >= {d_rim * 1e3:.2f} mm from the axis, rim top z 15.625 mm")
    g = np.linspace(-0.03, 0.03, 601)
    Y, Z = np.meshgrid(g, g)
    rr = np.hypot(Y, Z)
    annulus = (rr >= r_bore) & (rr <= r_boss)
    cell = (g[1] - g[0]) ** 2
    out = {}
    for w, d in [(30.0, LEG / "sdf_hybrid")] + [
            (int(p.name[1:]) / 10, p) for p in sorted((LEG / "width_variants").glob("w*"))]:
        hs = hulls_of(d)
        V = np.concatenate([h.vertices for h in hs])
        x0, half = V[:, 0].min(), np.abs(V[:, 1:]).max()
        # shoulder face: hull points within 0.2 mm of the lowest x, their lateral extent
        sh = V[V[:, 0] < x0 + 2e-4]
        sh_half = np.abs(sh[:, 1:]).max()
        areas = []
        for ang in np.linspace(0, np.pi / 2, 19):
            ca, sa = np.cos(ang), np.sin(ang)
            u, v = ca * Y + sa * Z, -sa * Y + ca * Z
            sq = (np.abs(u) <= sh_half) & (np.abs(v) <= sh_half)
            areas.append((sq & annulus).sum() * cell)
        thread_in_body = t_r < half - 0.344e-3  # thread section inside the body's nominal footprint
        out[w] = {"n_hulls": len(hs), "shoulder_x_mm": round(x0 * 1e3, 3), "hull_half_width_mm": round(half * 1e3, 3),
                  "shoulder_half_width_mm": round(sh_half * 1e3, 3),
                  "corner_radius_mm": round(sh_half * np.sqrt(2) * 1e3, 2),
                  "support_area_mm2_min_max": [round(min(areas) * 1e6, 1), round(max(areas) * 1e6, 1)],
                  "thread_x_max_mm": round(t_xmax * 1e3, 3), "thread_inside_body_footprint": bool(thread_in_body),
                  "rests_flat_on_table_by_body": bool(half > t_r),
                  "overhangs_boss": bool(sh_half > r_boss), "corner_over_rim_top": bool(sh_half * np.sqrt(2) > d_rim)}
        print(w, out[w])
    print("Seated, the shoulder plane is at the boss/rim top (z 15.625 mm), and the whole body is above it, so a "
          "corner reaching over the rim walls only meets their top face, which is flush with the boss top.")
    (LEG / "width_variants" / "geometry_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
