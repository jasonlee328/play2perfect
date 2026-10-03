"""Roll out a frozen SAPG teacher on screwing and record offline-BC episodes.

Each successful episode is written as a self-contained directory, which
``build_hvla_dataset.py`` later turns into a HierarchicalVLA (ManiFlow /
FrankaDataset) dataset:

    <out_dir>/episodes/<shard>_e<env>_n<k>/
        front_rgb.mp4      224x224 rgb24 -> H.264 yuv420p, 60 fps, gop 2
        front_depth.mp4    224x224 RG8 (R = low byte, G = high byte of uint16 mm),
                           lossless libx264rgb -qp 0, gop 1
        lowdim.npz         per-frame arrays, see LOWDIM_FIELDS
        meta.json          success / length / goals / termination reason

Frame t pairs the observation the teacher acted on (camera image, joints,
tactile) with the absolute joint target its action produced (``_cur_targets``
right after the action pipeline, canonical order). Run with every delay off
(obs / action / object-state / camera) so that pairing is exact; this script
forces those settings.

The camera is the depth student's world-mounted ZED-matched camera, rendered
at --render-width x --render-height as RGB-D, cropped to --crop (x0 y0 x1 y1,
render pixels, exclusive end) and resized to --out-size square.

--no-save turns it into a teacher-only success-rate eval of the same config.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LOWDIM_FIELDS = {
    "joint_pos": "rad, canonical order (arm 7 + hand 22), before the step",
    "joint_vel": "rad/s, canonical order",
    "prev_targets": "rad, joint targets before this step's action, canonical order",
    "palm_pose": "palm centre xyz (env frame, m) + quat wxyz",
    "object_pose": "screw xyz (env frame, m) + quat wxyz (privileged, for diagnostics)",
    "hole_pose": "fixture xyz (env frame, m) + quat wxyz",
    "insert_target": "final screw seat (last goal's position, env frame, m); projected to goal.position for --token",
    "tactile": "5 fingerpads (thumb..pinky) x 8 x 8 TacMap, [0, 1], flattened finger*64+row*8+col",
    "goals_reached": "goals hit so far in this episode (0..10)",
    "retract_phase": "1 once every goal is hit and the hand is retracting",
    "teacher_mu": "raw teacher action (unclipped), canonical order",
    "action_joint_target": "LABEL: absolute joint target produced by clamp(mu, -1, 1), rad, canonical order",
}


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _open_encoder(path: Path, size: int, fps: int, lossless_rgb: bool) -> subprocess.Popen:
    cmd = [_FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{size}x{size}", "-r", str(fps), "-i", "-", "-threads", "1"]
    if lossless_rgb:
        cmd += ["-c:v", "libx264rgb", "-preset", "veryfast", "-qp", "0", "-g", "1",
                "-pix_fmt", "rgb24"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", "2",
                "-pix_fmt", "yuv420p"]
    cmd.append(str(path))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)


class _Episode:
    """One env's in-progress episode: two streaming encoders + low-dim rows."""

    def __init__(self, ep_dir: Path, size: int, fps: int, save: bool) -> None:
        self.dir = ep_dir
        self.rows: dict[str, list[np.ndarray]] = {k: [] for k in LOWDIM_FIELDS}
        self.n = 0
        self.save = save
        self.rgb = self.depth = None
        if save:
            ep_dir.mkdir(parents=True, exist_ok=True)
            self.rgb = _open_encoder(ep_dir / "front_rgb.mp4", size, fps, lossless_rgb=False)
            self.depth = _open_encoder(ep_dir / "front_depth.mp4", size, fps, lossless_rgb=True)

    def write_frames(self, rgb: np.ndarray, rg8: np.ndarray) -> None:
        self.rgb.stdin.write(rgb.tobytes())
        self.depth.stdin.write(rg8.tobytes())

    def close(self, keep: bool, meta: dict) -> bool:
        """Finish both encoders; keep the episode on disk only if `keep`."""
        if not self.save:
            return keep
        ok = True
        for p in (self.rgb, self.depth):
            try:
                p.stdin.close()
            except BrokenPipeError:
                ok = False
            err = p.stderr.read().decode(errors="replace")
            if p.wait() != 0:
                ok = False
                print(f"[record] ffmpeg failed for {self.dir}: {err[-500:]}", flush=True)
        if keep and ok:
            arrays = {k: np.stack(v) for k, v in self.rows.items()}
            np.savez(self.dir / "lowdim.npz", **arrays)
            (self.dir / "meta.json").write_text(json.dumps(meta))
            return True
        shutil.rmtree(self.dir, ignore_errors=True)
        return False


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", default="Isaacsimenvs-PreciseAssemblyDepthStudent-Direct-v0")
    parser.add_argument("--agent", default="rl_games_dagger_sapg_cfg_entry_point")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--teacher-tactile", action="store_true",
                        help="Teacher was trained with 8x8 tactile in its obs (screw_tactile8). "
                             "Tactile is recorded either way.")
    parser.add_argument("--block-id", type=float, default=50.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shard", default="s0", help="Prefix for episode dir names.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Env seed of the first attempt; a rerun into the same --out-dir "
                             "(e.g. after preemption) offsets it, so episodes never repeat.")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="Stop after this many saved (successful) episodes; with --no-save, "
                             "after this many finished episodes.")
    parser.add_argument("--min-episode-steps", type=int, default=100,
                        help="Shorter episodes are unstable inits and are dropped.")
    parser.add_argument("--max-episode-steps", type=int, default=3000,
                        help="Longer episodes are dropped (still run to completion).")
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=288)
    parser.add_argument("--crop", type=int, nargs=4, default=[224, 0, 512, 288],
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--out-size", type=int, default=224)
    parser.add_argument("--max-depth-m", type=float, default=2.5)
    parser.add_argument("--success-tolerance", type=float, default=0.075,
                        help="Pins the waypoint tolerance (termination.eval_success_tolerance) so the "
                             "training-time curriculum can't tighten it mid-run. 0.075 is the start "
                             "value, the setting the 86%% teacher-only DAgger check ran with.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--preview", type=int, default=0,
                        help="Save full + cropped PNGs of the first N envs at a few steps, then continue.")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    import torch.nn.functional as F
    from isaaclab.utils.math import quat_apply

    import isaacsimenvs  # noqa: F401  registers the tasks
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.tasks.play.utils import obs_utils
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info

    global _FFMPEG
    _FFMPEG = _ffmpeg_exe()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    out_dir = Path(args.out_dir)
    ep_root = out_dir / "episodes"
    # Resume: keep the complete episodes of earlier attempts, drop half-written ones, and
    # give this attempt its own name prefix and seed.
    n_prev = 0
    attempt = 0
    if not args.no_save and ep_root.is_dir():
        for d in ep_root.iterdir():
            if (d / "meta.json").is_file():
                n_prev += 1
            else:
                shutil.rmtree(d, ignore_errors=True)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        attempt = len(list(out_dir.glob("attempt_*.json")))
        (out_dir / f"attempt_{attempt:02d}.json").write_text(json.dumps(
            {"time": time.time(), "episodes_already_saved": n_prev, "argv": sys.argv}))
    tag = f"{args.shard}a{attempt:02d}"
    print(f"[record] attempt {attempt}: {n_prev} episodes already saved, target {args.num_episodes}", flush=True)
    S = int(args.out_size)
    x0, y0, x1, y1 = args.crop

    @hydra_task_config_with_yaml(args.task, args.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        env_cfg.sim.device = args.sim_device
        env_cfg.seed = int(args.seed) + 100003 * attempt
        # Camera: the student camera as RGB-D at the recording resolution. The policy
        # image the env builds is unused here; uncropped so its shape check passes.
        so = env_cfg.student_obs
        so.enabled = True
        so.image_modality = "rgbd"
        so.image_width, so.image_height = args.render_width, args.render_height
        so.crop_enabled = False
        so.image_input_width, so.image_input_height = args.render_width, args.render_height
        so.use_camera_delay, so.camera_delay_max = False, 0
        so.use_student_obs_delay, so.student_obs_delay_max = False, 0
        so.use_depth_aug = False
        # Every delay off, so frame t's observation and label line up exactly.
        dr = env_cfg.domain_randomization
        dr.use_obs_delay = False
        dr.use_action_delay = False
        dr.use_object_state_delay_noise = False
        # Tactile is always computed (recorded); it enters the teacher obs only for the
        # tactile teacher.
        env_cfg.tactile.enabled = True
        env_cfg.tactile.resolution = 8
        env_cfg.tactile.in_policy = bool(args.teacher_tactile)
        env_cfg.tactile.in_critic = False
        env_cfg.termination.eval_success_tolerance = float(args.success_tolerance)

        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        N = u.num_envs
        fps = int(round(1.0 / u.step_dt))
        wrapped = register_rlgames_env(env, rl_device=args.rl_device, clip_obs=10.0, clip_actions=1.0)
        teacher = Teacher(
            task_id="Isaacsimenvs-PreciseAssembly-Direct-v0",
            agent_key="rl_games_sapg_cfg_entry_point",
            checkpoint_path=args.teacher_checkpoint,
            num_envs=N,
            rl_device=args.rl_device,
            env_info=teacher_env_info(wrapped),
        )
        teacher.pin_block_id(args.block_id)
        # rl_games' Runner.load() reseeds every RNG from the agent yaml's fixed seed, which
        # would make every shard and attempt replay the same episodes. Reseed from ours.
        import random

        random.seed(env_cfg.seed)
        np.random.seed(env_cfg.seed % 2**32)
        torch.manual_seed(env_cfg.seed)
        torch.cuda.manual_seed_all(env_cfg.seed)

        # Label = the absolute target this step's action produces. Read it inside the
        # step, before resets overwrite _cur_targets for envs that finish.
        perm = u._perm_lab_to_canon
        orig_pre = u._pre_physics_step

        def _pre_physics_step(actions):
            orig_pre(actions)
            u._bc_label = u._cur_targets[:, perm].clone()

        u._pre_physics_step = _pre_physics_step

        cam = u.student_camera
        obs, _ = env.reset()

        # Intrinsics of the saved SxS image (static camera; same for every env). Built
        # from the pinhole cfg: Isaac Lab's intrinsic_matrices put the principal point
        # at the image centre and ignore the aperture offsets the renderer applies
        # (the yaml's ZED calibration gives cx=80.02, cy=47.13 at 160x90).
        W, H = args.render_width, args.render_height
        f_px = W * so.focal_length / so.horizontal_aperture
        v_ap = so.horizontal_aperture * H / W
        K = np.array([[f_px, 0.0, W / 2 + so.horizontal_aperture_offset / so.horizontal_aperture * W],
                      [0.0, f_px, H / 2 + so.vertical_aperture_offset / v_ap * H],
                      [0.0, 0.0, 1.0]])
        K_isaac = cam.data.intrinsic_matrices[0].double().cpu().numpy()
        print(f"[record] K render (cfg) {K.round(3).tolist()} | isaac {K_isaac.round(3).tolist()}", flush=True)
        sx, sy = S / (x1 - x0), S / (y1 - y0)
        K_out = np.array([[K[0, 0] * sx, 0.0, (K[0, 2] - x0) * sx],
                          [0.0, K[1, 1] * sy, (K[1, 2] - y0) * sy],
                          [0.0, 0.0, 1.0]])

        def capture():
            """Frame (rgb, rg8) for every env, as uint8 (N, S, S, 3) numpy."""
            out = cam.data.output
            rgb = out["rgb"][..., :3][:, y0:y1, x0:x1].permute(0, 3, 1, 2).float()
            rgb = F.interpolate(rgb, size=(S, S), mode="bilinear", antialias=True, align_corners=False)
            rgb = rgb.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
            d = out["distance_to_image_plane"][..., 0][:, y0:y1, x0:x1].unsqueeze(1)
            d = F.interpolate(d, size=(S, S), mode="nearest-exact")[:, 0]
            valid = torch.isfinite(d) & (d > 0) & (d <= args.max_depth_m)
            mm = torch.where(valid, (d * 1000.0).round(), torch.zeros_like(d)).to(torch.int32)
            rg8 = torch.stack([mm & 0xFF, (mm >> 8) & 0xFF, torch.zeros_like(mm)], dim=-1).to(torch.uint8)
            return rgb.cpu().numpy(), rg8.cpu().numpy()

        def lowdim():
            rd = u.robot.data
            palm = rd.body_state_w[:, u._palm_body_id, :]
            palm_c = obs_utils._apply_local_offset(palm[:, :3], palm[:, 3:7],
                                                   obs_utils.PALM_CENTER_OFFSET, (N,))
            org = u.scene.env_origins
            tac = getattr(u, "_last_tactile", None)
            if tac is None:
                tac = torch.zeros(N, 320, device=u.device)
            return {
                "joint_pos": rd.joint_pos[:, perm],
                "joint_vel": rd.joint_vel[:, perm],
                "prev_targets": u._prev_targets[:, perm],
                "palm_pose": torch.cat([palm_c - org, palm[:, 3:7]], dim=-1),
                "object_pose": torch.cat([u.object.data.root_pos_w - org, u.object.data.root_quat_w], dim=-1),
                "hole_pose": torch.cat([u.hole_pos, u.hole_quat_wxyz], dim=-1),
                "insert_target": u.hole_pos + quat_apply(u.hole_quat_wxyz, u._insert_pos_rel[-1].expand(N, 3)),
                "tactile": tac,
                "goals_reached": u._successes.float().unsqueeze(-1),
                "retract_phase": u.retract_phase.float().unsqueeze(-1),
            }

        if args.preview:
            import imageio.v3 as iio

            pdir = out_dir / "preview"
            pdir.mkdir(parents=True, exist_ok=True)

        save = not args.no_save
        if save:
            ep_root.mkdir(parents=True, exist_ok=True)
        counters = [0] * N
        episodes = [_Episode(ep_root / f"{tag}_e{i:04d}_n0000", S, fps, save) for i in range(N)]
        pool = ThreadPoolExecutor(max_workers=32)
        closers = []
        n_saved = n_prev
        n_done = n_success = 0
        lengths_success, lengths_all = [], []
        reasons_count: dict[str, int] = {}
        stop = False
        t0 = time.time()
        step = 0
        while not stop:
            frames = capture() if save or args.preview else None
            ld = {k: v.float().cpu().numpy() for k, v in lowdim().items()}
            with torch.no_grad():
                mu = teacher.get_action(obs["teacher_obs"])
            obs, _, term, trunc, extras = env.step(mu.clamp(-1.0, 1.0))
            label = u._bc_label.cpu().numpy()
            mu_np = mu.float().cpu().numpy()

            if args.preview and step in (0, 1, 2, 30, 120, 300):
                full = cam.data.output["rgb"][..., :3][: args.preview].cpu().numpy()
                for i in range(min(args.preview, N)):
                    iio.imwrite(pdir / f"full_env{i}_t{step:04d}.png", full[i])
                    iio.imwrite(pdir / f"crop_env{i}_t{step:04d}.png", frames[0][i])
                    dm = frames[1][i, ..., 0].astype(np.uint16) | (frames[1][i, ..., 1].astype(np.uint16) << 8)
                    iio.imwrite(pdir / f"depth_env{i}_t{step:04d}.png",
                                (np.clip((dm.astype(np.float32) - 500) / 1000.0, 0, 1) * 255).astype(np.uint8))

            if save:
                list(pool.map(lambda i: episodes[i].write_frames(frames[0][i], frames[1][i]), range(N)))
            for i in range(N):
                ep = episodes[i]
                for k in ld:
                    ep.rows[k].append(ld[k][i])
                ep.rows["teacher_mu"].append(mu_np[i])
                ep.rows["action_joint_target"].append(label[i])
                ep.n += 1

            done = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
            if done.numel() > 0:
                fin = extras["episode_final"]
                succ = fin["retract_success"][done].bool().cpu().tolist()
                ratio = fin["success_ratio"][done].float().cpu().tolist()
                reasons = {k: v[done].cpu().tolist() for k, v in u._termination_reasons.items()}
                for j, i in enumerate(done.cpu().tolist()):
                    ep = episodes[i]
                    reason = next((k for k in ("max_successes", "fall", "dropped", "hand_far", "timeout")
                                   if reasons[k][j]), "unknown")
                    valid_len = args.min_episode_steps <= ep.n <= args.max_episode_steps
                    if ep.n >= args.min_episode_steps:
                        n_done += 1
                        lengths_all.append(ep.n)
                        reasons_count[reason] = reasons_count.get(reason, 0) + 1
                        if succ[j]:
                            n_success += 1
                            lengths_success.append(ep.n)
                    keep = save and succ[j] and valid_len and n_saved < args.num_episodes
                    if keep:
                        n_saved += 1
                    meta = {"success": bool(succ[j]), "length": ep.n, "goals_ratio": ratio[j],
                            "termination": reason, "env": i, "shard": tag, "seed": env_cfg.seed}
                    closers.append(pool.submit(ep.close, keep, meta))
                    counters[i] += 1
                    episodes[i] = _Episode(ep_root / f"{tag}_e{i:04d}_n{counters[i]:04d}", S, fps, save)
                teacher.reset_idx(done)

            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                sr = n_success / max(1, n_done)
                print(f"[record] step {step} {el:.0f}s {step * N / el:.0f} env-steps/s | done {n_done} "
                      f"success {n_success} ({sr:.1%}) saved {n_saved}/{args.num_episodes} | "
                      f"mean len succ {np.mean(lengths_success) if lengths_success else 0:.0f} | {reasons_count}",
                      flush=True)
            stop = (n_saved >= args.num_episodes) if save else (n_done >= args.num_episodes)

        # Discard the episodes still in flight, then wait for every encoder.
        for ep in episodes:
            closers.append(pool.submit(ep.close, False, {}))
        kept = sum(bool(c.result()) for c in closers)
        pool.shutdown()

        summary = {
            "teacher_checkpoint": args.teacher_checkpoint,
            "teacher_tactile": bool(args.teacher_tactile),
            "block_id": args.block_id,
            "num_envs": N,
            "fps": fps,
            "episodes_finished": n_done,
            "episodes_success": n_success,
            "success_rate": n_success / max(1, n_done),
            "episodes_saved": kept,
            "episodes_saved_total": n_prev + kept,
            "attempt": attempt,
            "seed": env_cfg.seed,
            "mean_len_success": float(np.mean(lengths_success)) if lengths_success else None,
            "len_success_pctl": (np.percentile(lengths_success, [5, 50, 95]).tolist()
                                 if lengths_success else None),
            "terminations": reasons_count,
            "camera": {"render": [args.render_width, args.render_height], "crop_xyxy": args.crop,
                       "out_size": S, "K_out": K_out.tolist(), "K_render": K.tolist(),
                       "K_render_isaac": K_isaac.tolist(),
                       "pos": list(so.camera_pos), "quat_wxyz": list(so.camera_quat_wxyz),
                       "convention": so.camera_convention},
            "lowdim_fields": LOWDIM_FIELDS,
            "hydra_overrides": hydra_args,
            "wall_s": time.time() - t0,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
        print("[record] SUMMARY " + json.dumps({k: summary[k] for k in (
            "episodes_finished", "episodes_success", "success_rate", "episodes_saved",
            "mean_len_success", "len_success_pctl", "terminations", "wall_s")}), flush=True)

    run()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


_FFMPEG = "ffmpeg"

if __name__ == "__main__":
    main()
