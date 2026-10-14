"""Record a start-state bank for ``precise_assembly.start_state_file``.

Rolls out the teacher and, in each env's first episode, saves the state right after
the episode reaches ``--goal`` successes (default 1 = lead-in reached, peg grasped
25 mm above the hole, screwing about to start): robot joint state and targets, the
peg pose/velocity in the hole frame, and the episode trackers the reset restores.

Goals hardly check depth, so ``--max-height-mm H`` captures by geometry instead: the
first step the peg is within H mm (along the hole axis) of the final seated waypoint and
within ``--max-lateral-mm`` of it sideways, i.e. down in the threads.

    python record_start_states.py --headless --enable_cameras --teacher-checkpoint <pth> \
        --teacher-tactile --out bank.npz <env overrides, e.g. the static layout>
"""
import argparse
import sys

import numpy as np


def main():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("--task", default="Isaacsimenvs-PreciseAssemblyDepthStudent-Direct-v0")
    p.add_argument("--agent", default="rl_games_dagger_sapg_cfg_entry_point")
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--teacher-tactile", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--goal", type=int, default=1)
    p.add_argument("--max-height-mm", type=float, default=None)
    p.add_argument("--max-lateral-mm", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1500)
    AppLauncher.add_app_launcher_args(p)
    args, hydra_args = p.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.dagger.teacher import Teacher
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env, teacher_env_info

    @hydra_task_config_with_yaml(args.task, args.agent)
    def run(env_cfg, agent_cfg):
        env_cfg.seed = args.seed
        env_cfg.tactile.in_policy = bool(args.teacher_tactile)
        env_cfg.precise_assembly.start_state_file = ""
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        N = u.num_envs
        wrapped = register_rlgames_env(env, rl_device="cuda:0", clip_obs=10.0, clip_actions=1.0)
        teacher = Teacher(task_id="Isaacsimenvs-PreciseAssembly-Direct-v0",
                          agent_key="rl_games_sapg_cfg_entry_point",
                          checkpoint_path=args.teacher_checkpoint, num_envs=N,
                          rl_device="cuda:0", env_info=teacher_env_info(wrapped))
        teacher.pin_block_id(50.0)
        torch.manual_seed(args.seed)

        obs, _ = env.reset()
        keys = ["joint_pos", "joint_vel", "cur_targets", "prev_targets", "peg_pos_h", "peg_quat_h",
                "peg_linvel_h", "peg_angvel_h", "successes", "lifted", "object_init_z", "hole_pos", "step"]
        bank = {k: [] for k in keys}
        taken = torch.zeros(N, dtype=torch.bool, device=u.device)   # captured or first episode over
        seat = u._insert_pos_rel[-1]                                  # final waypoint, hole frame
        if args.max_height_mm is not None:
            print(f"[start-states] capture when peg <= {args.max_height_mm} mm above seat "
                  f"{(seat * 1000).tolist()} mm (hole frame), lateral <= {args.max_lateral_mm} mm", flush=True)
        for step in range(args.max_steps):
            with torch.no_grad():
                obs, _, term, trunc, _ = env.step(teacher.get_action(obs["teacher_obs"]).clamp(-1.0, 1.0))
            done = term | trunc
            if args.max_height_mm is None:
                hit = (u._successes == args.goal) & ~taken & ~done
            else:
                rel_all = quat_apply_inverse(u.hole_quat_wxyz, u.object.data.root_pos_w - u.scene.env_origins
                                             - u.hole_pos) - seat
                hit = ((rel_all[:, 2] <= args.max_height_mm / 1000)
                       & (rel_all[:, :2].norm(dim=-1) <= args.max_lateral_mm / 1000) & ~taken & ~done)
            if hit.any():
                ids = hit.nonzero(as_tuple=False).squeeze(-1)
                hq = u.hole_quat_wxyz[ids]
                od = u.object.data
                rel = od.root_pos_w[ids] - u.scene.env_origins[ids] - u.hole_pos[ids]
                vals = {
                    "joint_pos": u.robot.data.joint_pos[ids], "joint_vel": u.robot.data.joint_vel[ids],
                    "cur_targets": u._cur_targets[ids], "prev_targets": u._prev_targets[ids],
                    "peg_pos_h": quat_apply_inverse(hq, rel),
                    "peg_quat_h": quat_mul(quat_conjugate(hq), od.root_quat_w[ids]),
                    "peg_linvel_h": quat_apply_inverse(hq, od.root_lin_vel_w[ids]),
                    "peg_angvel_h": quat_apply_inverse(hq, od.root_ang_vel_w[ids]),
                    "successes": u._successes[ids], "lifted": u._lifted_object[ids],
                    "object_init_z": u._object_init_z[ids], "hole_pos": u.hole_pos[ids],
                    "step": torch.full((ids.numel(),), step, device=u.device),
                }
                for k in keys:
                    bank[k].append(vals[k].detach().cpu().numpy())
            taken |= hit | done
            if step % 100 == 0:
                n = sum(len(v) for v in bank["successes"])
                print(f"[start-states] step {step} captured {n}/{N}", flush=True)
            if bool(taken.all()):
                break
        out = {k: np.concatenate(v) for k, v in bank.items()}
        np.savez(args.out, **out)
        h = (out["peg_pos_h"][:, 2] - seat[2].item()) * 1000
        print(f"[start-states] saved {len(out['successes'])} states (goal {args.goal}, max height "
              f"{args.max_height_mm}) to {args.out}; capture step median {np.median(out['step']):.0f}; "
              f"height above seat p5/50/95 {np.percentile(h, [5, 50, 95]).round(1).tolist()} mm; "
              f"goals credited {np.unique(out['successes'], return_counts=True)}", flush=True)
        env.close()

    run()
    app.close()


if __name__ == "__main__":
    main()
