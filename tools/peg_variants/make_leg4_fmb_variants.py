"""FMB-shaped body variants of the screwing peg (furniture_bench square_table_leg4, sdf_hybrid).

The original peg is a 25 mm thread (x in [-43.75, -18.75] mm, SDF collision) under a 30 x 30 mm square body
(x in [-18.75, 43.75] mm, 12 CoACD hulls). A variant replaces the body by a prism whose cross-section comes from
the FMB benchmark (rail-berkeley/fmb, Functional Manipulation Benchmark, peg.step: 27 single-extrusion pegs), scaled
uniformly so its area is the original's 900 mm^2 (or another --area), centred (bounding-box centre) on the thread
axis and extruded over the original body's x range. The thread mesh, mass (22.75 g), keypoints and object_scale are
unchanged, so goals and the teacher obs are identical; only the body collision/visual differs.

Cross-sections: sections.json (in OUT, committed) holds the raw FMB outlines (mm, FMB coordinates), extracted once
from the STEP solids exported to STL (``--fmb-stl-dir <dir with s00..s26.stl>``, the 100 mm prisms, extrusion along
z). Without --fmb-stl-dir the committed sections.json is used.

Per shape it writes assets/urdf/furniture_bench/square_table/square_table_leg4/fmb_variants/<name>/:
    square_table_leg4_body.obj   the exact extruded body (visual reference, not used by the URDF)
    body_coacd/decomp_<i>.obj    CoACD hulls of the body (threshold 0.05, max 12 hulls, like the original and the
                                 width variants). preprocess_mode="on": CoACD's manifold remesh (resolution 50) inflates
                                 the hulls by ~0.34 mm, the original's convention (the original body mesh isn't
                                 manifold, so "auto" remeshed it; the clean extrusion would skip it and get 0 mm)
    square_table_leg4_matchedmass_sdf_hybrid.urdf
        thread = ../../sdf_hybrid/square_table_leg4_thread.obj (shared, SDF 256), body = the hulls (visual+collision),
        mass 22.75 g; COM and inertia = the original's plus the change of a uniform-density solid model
        (cylinder thread + prism body, see model_props) from the 30 mm square to this cross-section.
    variant.json                 area, scale, extents, rest offset for the static layout (read by the env)

Orientation: the outline's (u, v) maps to the peg's (y, z); with the static layout's identity start quaternion z is up,
so the shapes are rotated (``rot_deg``) to lie on a flat side / on two star points / on the arch's feet.

Needs coacd (not in the Isaac venv): pip download coacd (abi3 wheel), unzip into <dir>, PYTHONPATH=<dir>.
Usage: python tools/peg_variants/make_leg4_fmb_variants.py [--fmb-stl-dir DIR] [--shapes circle hexagon ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon
from shapely import affinity

ROOT = Path(__file__).resolve().parents[2]
LEG = ROOT / "assets/urdf/furniture_bench/square_table/square_table_leg4"
SRC = LEG / "sdf_hybrid"
OUT = LEG / "fmb_variants"
MASS = 0.022750
X0, X1 = -0.01875, 0.04375  # body x range (m), shoulder at X0
THREAD_R = 0.0121875
HULL_REST_BELOW_AXIS0 = 0.0153438  # original hulls' extent below the axis (m); fixed_start_pose z was set for it
COM0 = np.array([0.005745, 0.0, -0.000001])
I0 = np.array([[2.570880e-06, -2.009384e-10, -8.317314e-09],
               [-2.009384e-10, 1.202610e-05, -2.897770e-12],
               [-8.317314e-09, -2.897770e-12, 1.202563e-05]])

# name: (FMB solid index in peg.step, in-plane rotation deg, family). Picked per family for canonical proportions.
SHAPES = {
    "circle": (10, 0.0, "circle"),            # 40.3 mm circle
    "hexagon": (11, 90.0, "hexagon"),         # regular hexagon, rotated to lie on a flat side
    "ellipse": (20, 0.0, "ellipse"),          # 53 x 36 ellipse (1.5:1), major axis horizontal
    "star": (17, 0.0, "star"),                # 5-point star, point up (rests on two points)
    "arch": (19, 0.0, "arch"),                # 48 x 29 "U"/arch, half-disc cutout opening down (rests on its feet)
    "rrect": (9, 0.0, "rounded rectangle"),   # 40 x 26 rounded rectangle (1.55:1), long side down
    "star_big": (17, 0.0, "star"),            # same star, scaled so its inner corners cover the thread base disk
}
AREA0_MM2 = 900.0


def extract_sections(stl_dir: Path, ids):
    out = {}
    for i in sorted(set(ids)):
        m = trimesh.load(stl_dir / f"s{i:02d}.stl", force="mesh")
        z = m.bounds[:, 2].mean()
        loops = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1]).discrete
        if len(loops) != 1:
            raise ValueError(f"FMB solid {i} has {len(loops)} outlines (multi-prong); not supported")
        out[str(i)] = np.asarray(loops[0][:, :2], float).round(5).tolist()
    return out


def shape_polygon(outline_mm, rot_deg, area_mm2=None, scale=None):
    p = Polygon(outline_mm).buffer(0)
    p = affinity.rotate(p, rot_deg, origin="centroid")
    b = p.bounds
    p = affinity.translate(p, -(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2)  # bbox centre on the axis
    if scale is None:
        scale = float(np.sqrt(area_mm2 / p.area))
    p = affinity.scale(p, scale, scale, origin=(0, 0))
    return p.simplify(0.02), scale


def area_moments(p: Polygon):
    """Area, centroid and second moments (about the centroid) of a simple polygon, in its own units."""
    xy = np.asarray(p.exterior.coords)
    x, y = xy[:-1, 0], xy[:-1, 1]
    xn, yn = xy[1:, 0], xy[1:, 1]
    cr = x * yn - xn * y
    A = cr.sum() / 2
    cx = ((x + xn) * cr).sum() / (6 * A)
    cy = ((y + yn) * cr).sum() / (6 * A)
    Ixx = ((y * y + y * yn + yn * yn) * cr).sum() / 12  # int y^2 dA about the origin
    Iyy = ((x * x + x * xn + xn * xn) * cr).sum() / 12
    Ixy = ((x * yn + 2 * x * y + 2 * xn * yn + xn * y) * cr).sum() / 24
    s = np.sign(A)
    A, Ixx, Iyy, Ixy = abs(A), s * Ixx, s * Iyy, s * Ixy
    return A, np.array([cx, cy]), Iyy - A * cx * cx, Ixx - A * cy * cy, Ixy - A * cx * cy  # int u^2, int v^2, int uv


def model_props(poly_m: Polygon):
    """Unit-mass, uniform-density solid model: thread = solid cylinder r 11 mm over x in [-43.75, -18.75] mm (as in
    make_leg4_width_variants.py), body = prism of the cross-section (u -> y, v -> z) over x in [X0, X1].
    Returns COM and the inertia tensor about the COM."""
    r, xt0, xt1 = 0.011, -0.04375, -0.01875
    A, c, Juu, Jvv, Juv = area_moments(poly_m)
    Lt, Lb = xt1 - xt0, X1 - X0
    vt, vb = np.pi * r * r * Lt, A * Lb
    mt, mb = vt / (vt + vb), vb / (vt + vb)
    ct = np.array([(xt0 + xt1) / 2, 0, 0])
    cb = np.array([(X0 + X1) / 2, c[0], c[1]])
    com = mt * ct + mb * cb
    It = mt * np.diag([r * r / 2, (3 * r * r + Lt * Lt) / 12, (3 * r * r + Lt * Lt) / 12])
    rho = mb / (A * Lb)  # mass per volume of the body
    Ib = rho * Lb * np.array([[Juu + Jvv, 0, 0],
                              [0, Jvv + A * Lb * Lb / 12, -Juv],
                              [0, -Juv, Juu + A * Lb * Lb / 12]])
    par = lambda m, cc: m * (np.dot(cc - com, cc - com) * np.eye(3) - np.outer(cc - com, cc - com))
    return com, It + par(mt, ct) + Ib + par(mb, cb)


def extrude(poly_m: Polygon) -> trimesh.Trimesh:
    m = trimesh.creation.extrude_polygon(poly_m, X1 - X0)  # along +z from 0
    v = m.vertices.copy()
    m.vertices = np.stack([v[:, 2] + X0, v[:, 0], v[:, 1]], 1)  # (u, v, h) -> (x = X0 + h, y = u, z = v)
    m.fix_normals()
    return m


def urdf(name: str, fam: str, n_hulls: int, com, I) -> str:
    rows = []
    for i in range(n_hulls):
        g = f'<geometry><mesh filename="body_coacd/decomp_{i}.obj" scale="1 1 1"/></geometry>'
        rows.append(f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/>{g}<material name="body"/></visual>')
        rows.append(f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/>{g}</collision>')
    return f"""<?xml version="1.0" encoding="utf-8"?>
<robot name="square_table_leg4_fmb_{name}_sdf_hybrid_matchedmass">
  <!-- FMB-shape body variant ({fam}): body cross-section from the FMB benchmark, same length, same thread.
       Generated by tools/peg_variants/make_leg4_fmb_variants.py. -->
  <link name="base_link">
    <inertial>
      <mass value="{MASS:.6f}"/>
      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>
      <inertia ixx="{I[0, 0]:.6e}" ixy="{I[0, 1]:.6e}" ixz="{I[0, 2]:.6e}" iyy="{I[1, 1]:.6e}" iyz="{I[1, 2]:.6e}" izz="{I[2, 2]:.6e}"/>
    </inertial>
  </link>
  <material name="thread">
    <color rgba="0.95 0.55 0.10 1.0"/>
  </material>
  <material name="body">
    <color rgba="0.50 0.50 0.50 1.0"/>
  </material>
  <link name="thread_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../../sdf_hybrid/square_table_leg4_thread.obj" scale="1 1 1"/></geometry>
      <material name="thread"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="../../sdf_hybrid/square_table_leg4_thread.obj" scale="1 1 1"/></geometry>
      <sdf resolution="256"/>
    </collision>
  </link>
  <joint name="base_to_thread" type="fixed">
    <parent link="base_link"/>
    <child link="thread_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <link name="body_link">
{chr(10).join(rows)}
  </link>
  <joint name="base_to_body" type="fixed">
    <parent link="base_link"/>
    <child link="body_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def hull_section_fidelity(hulls, poly_m, x=0.0125, res=2e-4):
    """Grid-sample the hulls' union at one x: fraction of the true section missed, and hull area outside the section
    by more than 0.5 mm (i.e. beyond the inflation), in mm^2."""
    b = np.array(poly_m.bounds) + [-0.003, -0.003, 0.003, 0.003]
    U, V = np.meshgrid(np.arange(b[0], b[2], res), np.arange(b[1], b[3], res))
    P = np.stack([np.full(U.size, x), U.ravel(), V.ravel()], 1)
    inside_h = np.zeros(len(P), bool)
    for h in hulls:
        inside_h |= h.contains(P)
    from shapely import contains_xy
    inside_p = contains_xy(poly_m, P[:, 1], P[:, 2])
    near = contains_xy(poly_m.buffer(0.0005), P[:, 1], P[:, 2])
    cell = res * res
    return (float((inside_p & ~inside_h).sum() / max(inside_p.sum(), 1)),
            float((inside_h & ~near).sum() * cell * 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmb-stl-dir", type=Path, default=None)
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES))
    ap.add_argument("--area", type=float, default=AREA0_MM2, help="target cross-section area, mm^2")
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--max-hulls", type=int, default=12)
    a = ap.parse_args()
    import coacd

    coacd.set_log_level("error")
    OUT.mkdir(parents=True, exist_ok=True)
    sec_path = OUT / "sections.json"
    if a.fmb_stl_dir is not None:
        sections = extract_sections(a.fmb_stl_dir, [SHAPES[n][0] for n in SHAPES])
        sec_path.write_text(json.dumps(sections))
    sections = json.loads(sec_path.read_text())

    sq = Polygon([(-0.015, -0.015), (0.015, -0.015), (0.015, 0.015), (-0.015, 0.015)])
    com_m0, I_m0 = model_props(sq)
    info = json.loads((OUT / "variants.json").read_text()) if (OUT / "variants.json").is_file() else {}
    for name in a.shapes:
        idx, rot, fam = SHAPES[name]
        outline = sections[str(idx)]
        if name == "star_big":
            # smallest scale at which the star's section covers the thread base disk (r 12.19 mm) + 0.25 mm
            base, _ = shape_polygon(outline, rot, scale=1.0)
            lo, hi = 0.5, 4.0
            for _ in range(40):
                mid = (lo + hi) / 2
                p, _ = shape_polygon(outline, rot, scale=mid)
                lo, hi = (lo, mid) if p.contains(Point(0, 0).buffer(THREAD_R * 1e3 + 0.25, 64)) else (mid, hi)
            poly_mm, s = shape_polygon(outline, rot, scale=hi)
        else:
            poly_mm, s = shape_polygon(outline, rot, area_mm2=a.area)
        poly_m = affinity.scale(poly_mm, 1e-3, 1e-3, origin=(0, 0))
        d = OUT / name
        (d / "body_coacd").mkdir(parents=True, exist_ok=True)
        for f in (d / "body_coacd").glob("decomp_*.obj"):
            f.unlink()
        body = extrude(poly_m)
        body.export(d / "square_table_leg4_body.obj")
        parts = coacd.run_coacd(coacd.Mesh(body.vertices, body.faces), threshold=a.threshold,
                                max_convex_hull=a.max_hulls, preprocess_mode="on")
        hulls = []
        for i, (v, f) in enumerate(parts):
            h = trimesh.Trimesh(v, f, process=True)
            h.export(d / "body_coacd" / f"decomp_{i}.obj")
            hulls.append(h)
        com_m, I_m = model_props(poly_m)
        com = COM0 + (com_m - com_m0)
        inertia = I0 + MASS * (I_m - I_m0)
        inertia = (inertia + inertia.T) / 2
        assert np.all(np.linalg.eigvalsh(inertia) > 0), inertia
        (d / "square_table_leg4_matchedmass_sdf_hybrid.urdf").write_text(urdf(name, fam, len(hulls), com, inertia))
        V = np.concatenate([h.vertices for h in hulls])
        lo, hi = V.min(0), V.max(0)
        disk = Point(0, 0).buffer(THREAD_R * 1e3, 128)
        missed, outside = hull_section_fidelity(hulls, poly_m)
        b = poly_mm.bounds
        info[name] = {
            "family": fam, "fmb_index": idx, "rot_deg": rot, "scale": round(s, 4),
            "fmb_area_mm2": round(Polygon(outline).area, 1), "area_mm2": round(poly_mm.area, 1),
            "section_bbox_mm": [round(b[2] - b[0], 2), round(b[3] - b[1], 2)],
            "section_centroid_mm": [round(poly_mm.centroid.x, 3), round(poly_mm.centroid.y, 3)],
            "section_max_radius_mm": round(float(np.linalg.norm(np.asarray(poly_mm.exterior.coords), axis=1).max()), 2),
            "section_min_radius_mm": round(float(poly_mm.exterior.distance(Point(0, 0))), 2),
            "thread_disk_covered_frac": round(poly_mm.intersection(disk).area / disk.area, 4),
            "thread_disk_covered": bool(poly_mm.buffer(1e-6).contains(disk)),
            "hull_inflation_mm": round(float((X0 - lo[0]) * 1e3), 3), "hulls": len(hulls), "hull_bounds_mm": [(lo * 1e3).round(3).tolist(), (hi * 1e3).round(3).tolist()],
            "hull_section_missed_frac": round(missed, 4), "hull_area_beyond_0p5mm_mm2": round(outside, 2),
            "rest_below_axis_mm": round(-lo[2] * 1e3, 3),
            "rest_z_shift_m": round(float(-lo[2] - HULL_REST_BELOW_AXIS0), 6),
            "com": com.round(7).tolist(), "inertia": inertia.tolist(),
        }
        (d / "variant.json").write_text(json.dumps(info[name], indent=1))
        print(name, {k: v for k, v in info[name].items() if k not in ("inertia",)}, flush=True)
    (OUT / "variants.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
