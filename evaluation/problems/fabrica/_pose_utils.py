"""Generic URDF-construction helper used by problem registration."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence, Tuple
from xml.dom import minidom

from scipy.spatial.transform import Rotation as R

# Distinct fallback colors when caller doesn't supply per-part rgba.
_PART_COLOR_PALETTE: Tuple[Tuple[float, float, float, float], ...] = (
    (0.85, 0.30, 0.30, 1.0),
    (0.30, 0.65, 0.85, 1.0),
    (0.40, 0.75, 0.35, 1.0),
    (0.90, 0.70, 0.20, 1.0),
    (0.70, 0.40, 0.80, 1.0),
    (0.30, 0.80, 0.75, 1.0),
)


def write_fixture_urdf(
    output_path: Path,
    parts: Iterable[Tuple[str, Tuple[float, float, float, float, float, float, float], Sequence[str]]],
    robot_name: str | None = None,
) -> Path:
    """Generate a URDF that places multiple parts (each potentially a stack
    of CoACD hulls) at given poses, following the multi-link + fixed-joint
    convention used by ``assets/urdf/fabrica/.../insertion_scenes/scene_*.urdf``.

    Each entry in ``parts`` is ``(part_id, pose_xyz_xyzw, [mesh_rel, ...])``:
      * ``part_id``: short label, used in link names and material name.
      * ``pose_xyz_xyzw``: ``(x, y, z, qx, qy, qz, qw)``, applied as the joint
        origin from ``root`` to the part's link.
      * ``[mesh_rel, ...]``: one or more mesh paths (relative to the output
        URDF's directory). All meshes for a given part share that part's
        link, joint, and material colour, so per-part visual identity is
        preserved across CoACD hulls.

    Layout::

        <robot>
          <link name="root"/>
          <material name="part_<id>"><color rgba=".. .. .. .."/></material>
          <link name="part_<id>">
            <visual><origin xyz="0 0 0"/><geometry><mesh ../></geometry>
                    <material name="part_<id>"/></visual>
            <collision><origin xyz="0 0 0"/><geometry><mesh ../></geometry></collision>
            ... (repeat per hull) ...
          </link>
          <joint name="root_to_part_<id>" type="fixed">
            <parent link="root"/><child link="part_<id>"/>
            <origin xyz=".." rpy=".."/>
          </joint>
        </robot>
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot_name = robot_name or output_path.stem
    root = ET.Element("robot", attrib={"name": robot_name})
    ET.SubElement(root, "link", attrib={"name": "root"})

    for idx, (part_id, pose, mesh_paths) in enumerate(parts):
        material_name = f"part_{part_id}"
        r, g, b, a = _PART_COLOR_PALETTE[idx % len(_PART_COLOR_PALETTE)]
        material = ET.SubElement(root, "material", attrib={"name": material_name})
        ET.SubElement(material, "color", attrib={
            "rgba": f"{r:.4f} {g:.4f} {b:.4f} {a:.4f}",
        })

        link_name = f"part_{part_id}"
        link = ET.SubElement(root, "link", attrib={"name": link_name})
        for mesh_rel in mesh_paths:
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom_v = ET.SubElement(visual, "geometry")
            ET.SubElement(geom_v, "mesh", attrib={"filename": mesh_rel, "scale": "1 1 1"})
            ET.SubElement(visual, "material", attrib={"name": material_name})

            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
            geom_c = ET.SubElement(collision, "geometry")
            ET.SubElement(geom_c, "mesh", attrib={"filename": mesh_rel, "scale": "1 1 1"})

        x, y, z, qx, qy, qz, qw = pose
        rpy = R.from_quat([qx, qy, qz, qw]).as_euler("xyz")
        joint = ET.SubElement(root, "joint", attrib={
            "name": f"root_to_{link_name}", "type": "fixed",
        })
        ET.SubElement(joint, "parent", attrib={"link": "root"})
        ET.SubElement(joint, "child", attrib={"link": link_name})
        ET.SubElement(joint, "origin", attrib={
            "xyz": f"{x:.6f} {y:.6f} {z:.6f}",
            "rpy": f"{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}",
        })

    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ", encoding="utf-8")
    output_path.write_bytes(pretty)
    return output_path
