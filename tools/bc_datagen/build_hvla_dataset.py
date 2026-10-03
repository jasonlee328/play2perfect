"""Turn record_teacher.py episodes into a HierarchicalVLA (FrankaDataset) dataset.

Needs only numpy + pyarrow (no Isaac Sim). Layout written, which is what
``hvla/dataloader/franka_dataset.py`` reads:

    <out>/meta/info.json        fps, features (+ video shapes / depth encoding), camera_intrinsics
    <out>/meta/stats.json       min/max/mean/std/q01..q99 per numeric column
    <out>/meta/episodes.jsonl   per-episode length / success / source (informational)
    <out>/meta/tasks.jsonl
    <out>/meta/bc_datagen.json  recorder summaries of every source shard
    <out>/data/chunk-000/file-XXX.parquet           rows in episode order
    <out>/videos/observation.images.front_camera/chunk-CCC/file-FFF.mp4   one per episode
    <out>/videos/observation.depth.front_camera/chunk-CCC/file-FFF.mp4    RG8, one per episode

Column mapping onto the loader's Franka names (a 29-DoF arm+hand has no gripper):
    observation.state.joint_position  joint_pos (29)
    observation.state.joint_velocity  joint_vel (29)
    observation.state.ee_position     palm pose xyz + quat wxyz (7; the loader uses [:3])
    observation.state.gripper_position, action.gripper_position, action.pd_mode   zeros (1)
    action.joint_target               absolute joint target label (29) -> --action_keys action.joint_target
    observation.tactile               5x8x8 fingerpad tactile (320)
    goal.position                     final screw seat projected into the 224 front image (u, v px),
                                      the target pixel for HVLA's --token heatmap
Extra columns (prev targets, object / hole pose, teacher mu, goal progress) ride along for analysis.

    python build_hvla_dataset.py --src <recorder out_dir> [<more>] --out <dataset_root>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RGB_KEY = "observation.images.front_camera"
DEPTH_KEY = "observation.depth.front_camera"
INTR_KEY = "observation.camera_intrinsics.front_camera"

# parquet column -> (lowdim.npz field or None for constant zeros, dim)
VECTOR_COLS = {
    "observation.state.joint_position": ("joint_pos", 29),
    "observation.state.joint_velocity": ("joint_vel", 29),
    "observation.state.ee_position": ("palm_pose", 7),
    "observation.state.gripper_position": (None, 1),
    "observation.state.prev_targets": ("prev_targets", 29),
    "observation.state.object_pose": ("object_pose", 7),
    "observation.state.hole_pose": ("hole_pose", 7),
    "observation.state.insert_target": ("insert_target", 3),
    "observation.state.goals_reached": ("goals_reached", 1),
    "observation.state.retract_phase": ("retract_phase", 1),
    "observation.tactile": ("tactile", 320),
    "action.joint_target": ("action_joint_target", 29),
    "action.teacher_mu": ("teacher_mu", 29),
    "action.gripper_position": (None, 1),
    "action.pd_mode": (None, 1),
}
JOINT_NAMES = (
    [f"iiwa14_joint_{i}" for i in range(1, 8)]
    + [f"thumb_{i}" for i in range(5)] + [f"index_{i}" for i in range(4)]
    + [f"middle_{i}" for i in range(4)] + [f"ring_{i}" for i in range(4)]
    + [f"pinky_{i}" for i in range(5)]
)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def _quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def project_points(p_env: np.ndarray, cam: dict) -> np.ndarray:
    """Env-frame points (n, 3) -> (u, v) pixels of the saved image; NaN if behind the camera.

    The camera pose is env-frame with the ROS optical convention (+Z forward, +Y down),
    exactly as the recorder's TiledCamera OffsetCfg was configured.
    """
    if cam["convention"] != "ros":
        raise ValueError(f"unsupported camera convention {cam['convention']!r}")
    R = _quat_wxyz_to_mat(np.asarray(cam["quat_wxyz"], np.float64))
    pc = (p_env - np.asarray(cam["pos"], np.float64)) @ R
    K = np.asarray(cam["K_out"], np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2],
                       K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], axis=1)
    uv[pc[:, 2] <= 0] = np.nan
    return uv.astype(np.float32)


def _vec_array(x: np.ndarray) -> pa.Array:
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    offsets = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, pa.array(x.reshape(-1)))


class _Stats:
    """Exact min/max/mean/std; quantiles from a bounded random row sample."""

    def __init__(self, max_rows: int, seed: int = 0) -> None:
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)
        self.acc: dict[str, dict] = {}

    def add(self, key: str, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        a = self.acc.setdefault(key, {"n": 0, "sum": 0.0, "sq": 0.0, "min": None, "max": None, "sample": []})
        a["n"] += len(x)
        a["sum"] = a["sum"] + x.sum(0)
        a["sq"] = a["sq"] + (x ** 2).sum(0)
        a["min"] = x.min(0) if a["min"] is None else np.minimum(a["min"], x.min(0))
        a["max"] = x.max(0) if a["max"] is None else np.maximum(a["max"], x.max(0))
        a["sample"].append(x)

    def finalize(self, total_rows: int) -> dict:
        out = {}
        for key, a in self.acc.items():
            mean = a["sum"] / a["n"]
            std = np.sqrt(np.maximum(a["sq"] / a["n"] - mean ** 2, 0.0))
            sample = np.concatenate(a["sample"], 0)
            if len(sample) > self.max_rows:
                sample = sample[self.rng.choice(len(sample), self.max_rows, replace=False)]
            q = np.quantile(sample, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
            out[key] = {
                "min": a["min"].tolist(), "max": a["max"].tolist(),
                "mean": mean.tolist(), "std": std.tolist(), "count": [int(a["n"])],
                "q01": q[0].tolist(), "q10": q[1].tolist(), "q50": q[2].tolist(),
                "q90": q[3].tolist(), "q99": q[4].tolist(),
            }
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", nargs="+", required=True, help="record_teacher.py --out-dir(s)")
    p.add_argument("--out", required=True)
    p.add_argument("--task", default="screw the leg into the socket",
                   help="Constant language string written to goal.action.")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--episodes-per-parquet", type=int, default=200)
    p.add_argument("--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    p.add_argument("--stats-rows", type=int, default=400_000,
                   help="Rows sampled per column for the quantiles (per-frame sampling below).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)

    episodes, summaries = [], []
    for src in args.src:
        src = Path(src)
        for s in sorted(src.glob("summary_*.json")):
            summaries.append({"src": str(src), **json.loads(s.read_text())})
        for d in sorted((src / "episodes").iterdir()):
            if all((d / f).is_file() for f in ("meta.json", "lowdim.npz", "front_rgb.mp4", "front_depth.mp4")):
                episodes.append(d)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit("no complete episodes found")
    if not summaries:
        raise SystemExit("no summary_*.json in the sources (camera metadata lives there)")
    cams = {json.dumps(s["camera"]["K_out"]) for s in summaries}
    if len(cams) != 1:
        raise SystemExit(f"sources disagree on camera intrinsics: {cams}")
    fps_set = {s["fps"] for s in summaries}
    if len(fps_set) != 1:
        raise SystemExit(f"sources disagree on fps: {fps_set}")
    fps = fps_set.pop()
    cam = summaries[0]["camera"]
    S = int(cam["out_size"])
    K9 = np.asarray(cam["K_out"], dtype=np.float32).reshape(-1)

    stats = _Stats(args.stats_rows)
    # Per-frame quantile sampling keeps memory bounded: ~stats_rows / n_frames per frame.
    est_frames = sum(json.loads((d / "meta.json").read_text())["length"] for d in episodes)
    keep_p = min(1.0, 2.0 * args.stats_rows / max(1, est_frames))
    rng = np.random.default_rng(0)

    ep_lines, writer, file_idx, global_idx, total = [], None, 0, 0, 0
    data_dir = out / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    for ep_i, d in enumerate(episodes):
        meta = json.loads((d / "meta.json").read_text())
        z = np.load(d / "lowdim.npz")
        n = int(z["joint_pos"].shape[0])
        cols = {
            "episode_index": pa.array(np.full(n, ep_i, dtype=np.int64)),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "index": pa.array(np.arange(global_idx, global_idx + n, dtype=np.int64)),
            "timestamp": pa.array((np.arange(n) / fps).astype(np.float32)),
            "task_index": pa.array(np.zeros(n, dtype=np.int64)),
        }
        sel = rng.random(n) < keep_p
        for col, (field, dim) in VECTOR_COLS.items():
            if field is not None and field not in z:
                continue
            x = np.zeros((n, dim), np.float32) if field is None else z[field].reshape(n, dim).astype(np.float32)
            cols[col] = _vec_array(x)
            stats.add(col, x[sel] if sel.any() else x[:1])
        intr = np.repeat(K9[None], n, 0)
        cols[INTR_KEY] = _vec_array(intr)
        stats.add(INTR_KEY, intr[:1])
        cols["goal.action"] = pa.array([args.task] * n, type=pa.string())
        if "insert_target" in z:
            cols["goal.position"] = _vec_array(project_points(z["insert_target"].astype(np.float64), cam))
        table = pa.table(cols)

        if ep_i % args.episodes_per_parquet == 0:
            if writer is not None:
                writer.close()
            writer = pq.ParquetWriter(data_dir / f"file-{file_idx:03d}.parquet", table.schema)
            file_idx += 1
        writer.write_table(table)

        chunk, fi = ep_i // 1000, ep_i % 1000
        for key, fname in ((RGB_KEY, "front_rgb.mp4"), (DEPTH_KEY, "front_depth.mp4")):
            _link_or_copy(d / fname, out / "videos" / key / f"chunk-{chunk:03d}" / f"file-{fi:03d}.mp4",
                          args.link_mode)
        ep_lines.append({"episode_index": ep_i, "length": n, "tasks": [args.task],
                         "source": str(d), **{k: meta[k] for k in ("success", "goals_ratio", "termination")}})
        global_idx += n
        total += n
        if (ep_i + 1) % 100 == 0:
            print(f"[build] {ep_i + 1}/{len(episodes)} episodes, {total} frames", flush=True)
    writer.close()

    def _video_feat(depth: bool) -> dict:
        info = {"video.height": S, "video.width": S, "video.channels": 3, "video.fps": fps,
                "video.codec": "h264", "video.is_depth_map": depth, "has_audio": False}
        if depth:
            info.update({"video.pix_fmt": "rgb24", "video.depth_encoding": "rg8_mm",
                         "video.lossless": True})
        else:
            info["video.pix_fmt"] = "yuv420p"
        return {"dtype": "video", "shape": [S, S, 3], "names": ["height", "width", "channel"], "info": info}

    features = {
        RGB_KEY: _video_feat(False),
        DEPTH_KEY: _video_feat(True),
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "goal.action": {"dtype": "string", "shape": [1], "names": None},
        INTR_KEY: {"dtype": "float32", "shape": [9], "names": None},
        "goal.position": {"dtype": "float32", "shape": [2], "names": ["u", "v"]},
    }
    for col, (_f, dim) in VECTOR_COLS.items():
        names = JOINT_NAMES if dim == 29 else None
        features[col] = {"dtype": "float32", "shape": [dim], "names": names}

    info = {
        "codebase_version": "v3.0",
        "robot_type": "kuka_iiwa14_sharpa_hand",
        "total_episodes": len(episodes),
        "total_frames": total,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "camera_intrinsics": {"front_camera": K9.tolist()},
        "camera_extrinsics": {"front_camera": {"pos": cam["pos"], "quat_wxyz": cam["quat_wxyz"],
                                               "convention": cam["convention"], "frame": "env"}},
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out / "meta" / "stats.json").write_text(json.dumps(stats.finalize(total), indent=2))
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for line in ep_lines:
            f.write(json.dumps(line) + "\n")
    (out / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": args.task}) + "\n")
    (out / "meta" / "bc_datagen.json").write_text(json.dumps(summaries, indent=2))
    print(f"[build] wrote {out}: {len(episodes)} episodes, {total} frames, {file_idx} parquet files", flush=True)


if __name__ == "__main__":
    main()
