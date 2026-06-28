"""Object registry shared by the assembly problem definitions.

An ``Object`` bundles a URDF path with the policy-facing grasp-box scale and a
flag for whether the collision mesh needs a V-HACD convex decomposition. The
per-task ``objects.py`` modules (peg / fabrica / furniture_bench) populate the
global ``NAME_TO_OBJECT`` registry on import; problems then look objects up by
name.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import trimesh


def get_repo_root_dir() -> Path:
    """Repo root (the directory that contains ``assets/`` and ``evaluation/``)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class Object:
    urdf_path: Path
    """Path to the object URDF file."""

    scale: Tuple[float, float, float]
    """Scale of the object's grasp bounding box in x, y, z. Not a metric scale —
    this is the scale handed to the policy."""

    need_vhacd: bool
    """Whether the object needs a V-HACD convex decomposition (its convex hull is
    very different from the original mesh)."""

    def __post_init__(self):
        assert self.urdf_path.exists(), f"Filepath {self.urdf_path} does not exist"

    def get_object_mesh_path_and_scale(self) -> Tuple[Path, np.ndarray]:
        from yourdfpy import URDF

        object_urdf_path = self.urdf_path
        assert object_urdf_path.exists(), object_urdf_path
        urdf = URDF.load(str(object_urdf_path))

        mesh_path_and_scale_list = []
        for link in urdf.robot.links:
            if len(link.collisions) == 0:
                continue
            for collision_link in link.collisions:
                mesh_path = (
                    object_urdf_path.parent / collision_link.geometry.mesh.filename
                )
                assert mesh_path.exists(), mesh_path
                mesh_scale = (
                    np.array([1, 1, 1])
                    if collision_link.geometry.mesh.scale is None
                    else np.array(collision_link.geometry.mesh.scale)
                )
                mesh_path_and_scale_list.append((mesh_path, mesh_scale))

        # Assume urdf has only 1 link with only 1 collision mesh.
        assert len(mesh_path_and_scale_list) == 1, (
            f"{mesh_path_and_scale_list} has len {len(mesh_path_and_scale_list)}"
        )
        mesh_path, mesh_scale = mesh_path_and_scale_list[0]
        return mesh_path, mesh_scale

    def get_object_mesh(self) -> trimesh.Trimesh:
        mesh_path, mesh_scale = self.get_object_mesh_path_and_scale()
        mesh = trimesh.load_mesh(str(mesh_path))
        mesh.apply_scale(mesh_scale)
        return mesh


def rescale_by_factor(
    scale: Tuple[float, float, float], factor: float
) -> Tuple[float, float, float]:
    return (scale[0] * factor, scale[1] * factor, scale[2] * factor)


# Populated by the per-task objects.py modules on import.
NAME_TO_OBJECT: Dict[str, Object] = {}
