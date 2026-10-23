"""Geometric checks of the FMB-shape body variants of the screwing peg (make_leg4_fmb_variants.py).

Like check_leg4_width_variants.py, per shape (and the original): the hulls' shoulder plane, the thread vs the body
(is the thread base disk, r 12.19 mm at the shoulder, inside the body's section?), the shoulder's support on the
receiver when seated (hull section just above the shoulder, intersected with the hole boss annulus between the bore
and the boss edge, measured from one_leg_hole_detail.obj, the same at any seated angle), and whether the body
reaches over the rim walls (their top is flush with the boss top, so a seated body only meets their top face: no
penetration), plus how the peg rests on the table in the static layout (identity quaternion, z up): the support
the COM height as it rolls on the section's convex hull, and the
barrier to roll away, i.e. whether it rolls.
Usage: python tools/peg_variants/check_leg4_fmb_variants.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
SQ = ROOT / "assets/urdf/furniture_bench/square_table"
LEG = SQ / "square_table_leg4"
OUT = LEG / "fmb_variants"


def hulls_of(d):
    return [trimesh.load(p, force="mesh", process=False) for p in sorted(glob.glob(str(d / "body_coacd/decomp_*.obj")))]


def section(hulls, x):
    polys = []
    for h in hulls:
        s = h.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
        if s is None:
            continue
        for e in s.discrete:
            polys.append(Polygon(e[:, 1:]).buffer(0))  # (y, z)
    return unary_union(polys)


def main():
    thread = trimesh.load(LEG / "sdf_hybrid/square_table_leg4_thread.obj", force="mesh", process=False)
    t_r = np.linalg.norm(thread.vertices[:, 1:], axis=1).max()
    t_xmax = thread.vertices[:, 0].max()
    hole = trimesh.load(SQ / "hole_patches/sdf_hybrid/one_leg_hole_detail.obj", force="mesh", process=False)
    top_z = hole.vertices[:, 2].max()
    loops = hole.section(plane_origin=[0, 0, top_z - 1e-4], plane_normal=[0, 0, 1]).discrete
    inner = min(loops, key=lambda e: np.ptp(e[:, 0]))
    outer = max(loops, key=lambda e: np.ptp(e[:, 0]))
    c = (inner[:, :2].min(0) + inner[:, :2].max(0)) / 2
    r_bore = np.linalg.norm(inner[:, :2] - c, axis=1).max()
    r_boss = np.linalg.norm(outer[:, :2] - c, axis=1).min()
    rim_inner = 0.078751 - 0.0025  # rim walls (one_leg_sdf_hybrid.urdf), top z 0.015625
    d_rim = min(abs(-rim_inner - c[0]), abs(rim_inner - c[1]))
    print(f"thread: x <= {t_xmax * 1e3:.2f} mm, max radius {t_r * 1e3:.2f} mm | bore r <= {r_bore * 1e3:.2f} mm, "
          f"boss r >= {r_boss * 1e3:.2f} mm, rim inner faces >= {d_rim * 1e3:.2f} mm from the axis")
    annulus = Point(0, 0).buffer(r_boss, 256).difference(Point(0, 0).buffer(r_bore, 256))
    disk = Point(0, 0).buffer(t_r, 256)
    variants = json.loads((OUT / "variants.json").read_text())
    out = {}
    for name, d in [("original", LEG / "sdf_hybrid")] + [(n, OUT / n) for n in variants]:
        hs = hulls_of(d)
        V = np.concatenate([h.vertices for h in hs])
        x0 = V[:, 0].min()
        sh = section(hs, x0 + 3e-4)  # just above the shoulder face
        mid = section(hs, 0.0125)
        if name == "original":  # the original body is a hollow tube: its shoulder end is the solid face
            sh = Polygon(sh.exterior) if sh.geom_type == "Polygon" else sh
        support = sh.intersection(annulus).area  # the annulus is round, so this doesn't depend on the seated angle
        com = np.array(variants[name]["com"]) if name in variants else np.array([0.005745, 0.0, -1e-6])
        # Resting on the table (identity quat, z up), rolling about the long axis sees the convex hull of the
        # section. h(t) = COM height above the table after rolling by t; the peg is stable at t = 0 if h rises both
        # ways: roll barrier = the lower of the two maxima of h within 90 deg either way minus h(0), at roll angle "tip". A circle has a
        # ~0 barrier (rolls freely).
        ch = np.asarray(mid.convex_hull.exterior.coords)[:-1] - com[1:]
        ts = np.radians(np.arange(-180.0, 180.01, 0.25))
        hs_ = np.array([-(ch[:, 0] * np.sin(t) * -1 + ch[:, 1] * np.cos(t)).min() for t in ts])  # COM above lowest pt
        i0 = int(np.argmin(np.where(np.abs(ts) < np.radians(10), hs_, np.inf)))  # rest angle (local min near 0)
        peaks = []
        for side in (slice(i0 + 1, i0 + 361), slice(i0 - 360, i0)):  # roll up to 90 deg either way
            seg, tseg = hs_[side], ts[side]
            top = seg.max()
            j = int(np.argmax(seg >= top - 1e-5)) if side.start > i0 else int(len(seg) - 1 - np.argmax(seg[::-1] >= top - 1e-5))
            peaks.append((top - hs_[i0], abs(np.degrees(tseg[j] - ts[i0]))))
        barrier, tip = min(peaks)
        rest = "rolls (no flat side)" if barrier < 2e-4 else "stable" if tip > 10 else "marginal"
        out[name] = {
            "n_hulls": len(hs), "shoulder_x_mm": round(x0 * 1e3, 3),
            "section_area_mm2": round(mid.area * 1e6, 1),
            "thread_x_max_mm": round(t_xmax * 1e3, 3),
            "thread_disk_in_hull_section_frac": round(sh.intersection(disk).area / disk.area, 4),
            "thread_inside_body": bool(sh.buffer(1e-7).contains(disk)),
            "support_area_mm2": round(support * 1e6, 1),
            "boss_annulus_area_mm2": round(annulus.area * 1e6, 1),
            "max_radius_mm": round(float(np.linalg.norm(V[:, 1:], axis=1).max() * 1e3), 2),
            "overhangs_boss": bool(np.linalg.norm(V[:, 1:], axis=1).max() > r_boss),
            "reaches_over_rim_top": bool(np.linalg.norm(V[:, 1:], axis=1).max() > d_rim),
            "rest_below_axis_mm": round(-V[:, 2].min() * 1e3, 3),
            "roll_barrier_mm": round(float(barrier) * 1e3, 3), "roll_tip_angle_deg": round(float(tip), 1), "rest_on_table": rest,
            "thread_clears_table_at_rest": bool(-V[:, 2].min() > t_r),
        }
        print(name, out[name])
    print("Seated, the shoulder plane is at the boss/rim top (z 15.625 mm) and the whole body is above it, so a body "
          "reaching over the rim walls only meets their top face, which is flush with the boss top: no penetration.")
    (OUT / "geometry_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
