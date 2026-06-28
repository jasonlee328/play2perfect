"""Shared utilities for Fabrica viser viewers."""

import json
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets" / "urdf" / "fabrica"

ALL_ASSEMBLIES = [
    "beam",
    "car",
    "cooling_manifold",
    "duct",
    "gamepad",
    "plumbers_block",
    "stool_circular",
]

COLORS = [
    (0.9, 0.2, 0.2),
    (0.2, 0.7, 0.2),
    (0.2, 0.3, 0.9),
    (0.9, 0.7, 0.1),
    (0.8, 0.3, 0.8),
    (0.1, 0.8, 0.8),
    (0.9, 0.5, 0.1),
    (0.5, 0.5, 0.5),
    (0.6, 0.2, 0.6),
]


def load_assembly_order(assets_dir, assembly, parts):
    """Load assembly order and start rotations from JSON.

    Returns (ordered_ids, start_rotations) where start_rotations is a dict
    of part_id -> [axis_x, axis_y, axis_z, angle_deg].
    """
    order_path = Path(assets_dir) / assembly / "assembly_order.json"
    part_ids = [pid for pid, _ in parts]
    start_rotations = {}
    if order_path.exists():
        data = json.loads(order_path.read_text())
        ordered = data.get("steps", part_ids)
        start_rotations = data.get("start_rotations", {})
        if set(ordered) == set(part_ids):
            return ordered, start_rotations
        print(f"WARNING: assembly_order.json mismatch for {assembly}, using default")
    return part_ids, start_rotations


def load_assembly_parts(assets_dir, assembly):
    """Load all part meshes for an assembly. Returns list of (part_id, mesh)."""
    assembly_dir = Path(assets_dir) / assembly
    if not assembly_dir.exists():
        return []

    part_dirs = sorted(
        [d for d in assembly_dir.iterdir() if d.is_dir() and d.name != "fixture"],
        key=lambda d: int(d.name),
    )

    parts = []
    for part_dir in part_dirs:
        part_id = part_dir.name
        obj_path = part_dir / f"{part_id}.obj"
        if not obj_path.exists():
            continue
        mesh = trimesh.load_mesh(str(obj_path), process=False)
        parts.append((part_id, mesh))
    return parts


def load_all_assemblies(assets_dir=ASSETS_DIR):
    """Load all assemblies, returning {name: [(part_id, mesh), ...]}."""
    assembly_parts = {}
    for assembly in ALL_ASSEMBLIES:
        parts = load_assembly_parts(assets_dir, assembly)
        if parts:
            assembly_parts[assembly] = parts
            print(f"  {assembly}: {len(parts)} parts")
    return assembly_parts


def compute_explode_offsets(parts, spread=0.10):
    """Compute per-part offset vectors for explosion from centroid."""
    if not parts:
        return []

    all_centers = [mesh.centroid for _, mesh in parts]
    overall_center = np.mean(all_centers, axis=0)

    offsets = []
    for _, mesh in parts:
        direction = mesh.centroid - overall_center
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([0, 0, 1.0])
        offsets.append(direction * spread)
    return offsets


class SceneManager:
    """Tracks viser scene handles for easy cleanup."""

    def __init__(self):
        self._handles = []

    def add(self, handle):
        self._handles.append(handle)
        return handle

    def clear(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
