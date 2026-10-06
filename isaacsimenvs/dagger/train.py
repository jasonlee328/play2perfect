"""SAPG/PPO + DAgger depth-distillation training entrypoint.

Mirrors `isaacsimenvs/train.py`'s bootstrap (AppLauncher → hydra cfg → gym.make
→ register_rlgames_env → Runner) but:
  - registers the `depth_cnn_lstm` network (`isaacsimenvs.dagger.networks`)
  - registers the `dagger_sapg` / `dagger_ppo` algos (DAggerA2CAgent)
  - injects CLI overrides (--teacher-run-dir / --teacher-checkpoint / --lambda-d-*)
    into the yaml's `dagger:` block
  - flips `central_value_config` + `network.symmetric_critic` based on --critic
  - supports `--probe-only` to exercise env + camera only (no agent, no teacher)
    for memory probing, then exit.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="DAgger SAPG/PPO depth-distillation trainer.")
    # --- Task / agent ---
    parser.add_argument("--task", default="Isaacsimenvs-PreciseAssemblyDepthStudent-Direct-v0")
    parser.add_argument(
        "--agent",
        default="rl_games_dagger_sapg_cfg_entry_point",
        help="Hydra entry-point key for the dagger train yaml. Use one of the dagger keys "
             "for training (rl_games_dagger_{sapg,ppo}_cfg_entry_point) or the teacher "
             "entry point for probe-only memory sweeps where the agent_cfg is discarded.",
    )
    # --- Teacher ---
    parser.add_argument("--teacher-run-dir", default=None,
                        help="Directory containing the frozen SAPG teacher's rl_games run. "
                             "Required unless --probe-only.")
    parser.add_argument("--teacher-checkpoint", default=None,
                        help="Optional explicit .pth path; else newest under --teacher-run-dir.")
    # --- DAgger λ_D curriculum ---
    parser.add_argument("--lambda-d-start", type=float, default=None)
    parser.add_argument("--lambda-d-floor", type=float, default=None)
    parser.add_argument("--lambda-d-decay-frac", type=float, default=None)
    parser.add_argument("--value-warmup-epochs", type=int, default=None,
                        help="If >0, the first N updates train critic + BC MSE only "
                             "(actor PPO loss, entropy, bounds suppressed). λ_D stays "
                             "pinned to lambda_d_start during warmup, then the decay "
                             "schedule applies over the remaining max_epochs.")
    parser.add_argument("--distill-loss", choices=("mse", "nll"), default=None,
                        help="BC distillation objective. 'mse' (default) trains only "
                             "the student's μ head against teacher μ. 'nll' minimizes "
                             "expected NLL of teacher samples and trains μ_s + σ_s to "
                             "match the teacher's full Gaussian.")
    parser.add_argument("--init-sigma-from-teacher-block", type=float, default=None,
                        help="If set, the student's log_sigma is overwritten with the "
                             "teacher's log_sigma for this block id AFTER the student "
                             "checkpoint is loaded. For coef_cond student networks, all "
                             "blocks are initialized to the same teacher row.")
    parser.add_argument("--replay-capacity", type=int, default=None,
                        help="Keep this many past rollouts in a GPU FIFO for aggregated-data "
                             "replay (0 = off).")
    parser.add_argument("--replay-updates", type=int, default=None,
                        help="Extra BC-only minibatches per epoch drawn from the replay FIFO.")
    parser.add_argument("--deterministic-rollouts", action="store_true",
                        help="Use student μ as rollout action (no N(μ, σ) sampling). "
                             "Still DAgger — teacher labels every student-visited state.")
    # --- Critic toggle ---
    parser.add_argument("--critic", choices=("asymmetric", "symmetric"), default="asymmetric",
                        help="asymmetric uses the yaml's central_value_config; symmetric drops it "
                             "and adds a value head on the student trunk.")
    # --- Standard rl_games knobs ---
    parser.add_argument("--checkpoint", default=None, help="Path to .pth to restore")
    parser.add_argument("--checkpoint_load_mode", choices=("resume", "weights"), default="resume")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    # --- Probe mode (memory sweep, no agent / no teacher) ---
    parser.add_argument("--probe-only", action="store_true",
                        help="Exercise env init + N env.step() with zero actions, then exit.")
    parser.add_argument("--probe-steps", type=int, default=50)
    # --- Pose-only interactive viewer ---
    parser.add_argument("--capture_viewer", action="store_true")
    parser.add_argument("--capture_viewer_len", type=int, default=600)
    parser.add_argument("--capture_viewer_interval", type=int, default=6000)
    parser.add_argument("--capture_viewer_env_id", type=int, default=0)
    parser.add_argument("--capture_viewer_wandb_key", default="interactive_viewer")
    parser.add_argument("--capture_viewer_github_raw_base", default="")
    parser.add_argument("--capture_viewer_url_check", choices=("skip", "warn", "error"), default="skip")
    # --- wandb ---
    parser.add_argument("--wandb_activate", action="store_true")
    parser.add_argument("--wandb_project", default="isaacsimenvs")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_name", default="")
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument("--wandb_notes", default="")
    parser.add_argument("--wandb_logcode_dir", default="")
    # --- AppLauncher (--headless, --enable_cameras, ...) ---
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    # Multi-GPU (torchrun): each rank simulates its own envs on its own GPU and rl_games
    # averages gradients (config multi_gpu). Only rank 0 logs to wandb / captures viewer
    # videos / writes checkpoints (rl_games gates its writer + saves on global_rank 0).
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    multi_gpu = world_size > 1
    if multi_gpu:
        args_cli.distributed = True  # AppLauncher: Kit + renderer on cuda:LOCAL_RANK
        args_cli.sim_device = args_cli.rl_device = f"cuda:{local_rank}"
        if rank != 0:
            args_cli.wandb_activate = False
            args_cli.capture_viewer = False
        print(f"[multi-gpu] rank {rank}/{world_size} local_rank {local_rank} -> {args_cli.sim_device}", flush=True)

    if not args_cli.probe_only and args_cli.teacher_run_dir is None:
        parser.error("--teacher-run-dir is required unless --probe-only.")

    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args_cli).app

    # Imports that need Kit running.
    import gymnasium as gym
    import torch
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    from rl_games.torch_runner import Runner

    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.dagger import networks as _net  # noqa: F401  registers depth_cnn_lstm
    from isaacsimenvs.dagger.dagger_agent import DAggerA2CAgent
    from isaacsimenvs.dagger.metrics import DistillMetricsObserver
    from isaacsimenvs.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.utils.rlgames_utils import (
        EnvStatsAlgoObserver,
        MultiObserver,
        register_rlgames_env,
    )

    @hydra_task_config_with_yaml(args_cli.task, args_cli.agent)
    def run(env_cfg, agent_cfg: dict) -> None:
        hydra_run_dir = HydraConfig.get().runtime.output_dir
        env_cfg.sim.device = args_cli.sim_device
        if multi_gpu:
            env_cfg.seed = int(env_cfg.seed if env_cfg.seed is not None else 42) + 1000 * rank

        env = gym.make(args_cli.task, cfg=env_cfg)

        if args_cli.capture_viewer:
            from pathlib import Path
            from isaacsimenvs.tasks.play.pose_viewer import PlayPoseViewerWrapper

            env = PlayPoseViewerWrapper(
                env,
                output_dir=Path(hydra_run_dir) / "interactive_viewer",
                capture_len=args_cli.capture_viewer_len,
                capture_interval=args_cli.capture_viewer_interval,
                env_id=args_cli.capture_viewer_env_id,
                wandb_key=args_cli.capture_viewer_wandb_key,
                github_raw_base=args_cli.capture_viewer_github_raw_base,
                url_check=args_cli.capture_viewer_url_check,
            )

        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
        register_rlgames_env(
            env,
            rl_device=args_cli.rl_device,
            clip_obs=clip_obs,
            clip_actions=clip_actions,
        )

        # ---------------- Probe mode: env init + N zero-action steps + exit. ----------------
        if args_cli.probe_only:
            print(f"[probe] env constructed, num_envs={env_cfg.scene.num_envs}", flush=True)
            t0 = time.time()
            obs, _ = env.reset()
            zero_action = torch.zeros(
                env_cfg.scene.num_envs,
                env.action_space.shape[-1],
                device=args_cli.rl_device,
                dtype=torch.float32,
            )
            for n in range(args_cli.probe_steps):
                env.step(zero_action)
                if (n + 1) % 10 == 0:
                    print(f"[probe] step {n + 1}/{args_cli.probe_steps} elapsed={time.time() - t0:.1f}s", flush=True)
            elapsed = time.time() - t0
            print(
                f"[probe] num_envs={env_cfg.scene.num_envs} steps={args_cli.probe_steps} "
                f"elapsed={elapsed:.1f}s step_avg_ms={elapsed * 1000 / max(1, args_cli.probe_steps):.1f}",
                flush=True,
            )
            return  # exit cleanly; outer del app + os._exit handle teardown

        # ---------------- Inject CLI overrides into agent_cfg. ----------------
        dagger_cfg = agent_cfg["params"]["config"].setdefault("dagger", {})
        dagger_cfg["teacher_run_dir"] = args_cli.teacher_run_dir
        if args_cli.teacher_checkpoint is not None:
            dagger_cfg["teacher_checkpoint"] = args_cli.teacher_checkpoint
        if args_cli.lambda_d_start is not None:
            dagger_cfg["lambda_d_start"] = args_cli.lambda_d_start
        if args_cli.lambda_d_floor is not None:
            dagger_cfg["lambda_d_floor"] = args_cli.lambda_d_floor
        if args_cli.lambda_d_decay_frac is not None:
            dagger_cfg["lambda_d_decay_frac"] = args_cli.lambda_d_decay_frac
        if args_cli.value_warmup_epochs is not None:
            dagger_cfg["value_warmup_epochs"] = args_cli.value_warmup_epochs
        if args_cli.distill_loss is not None:
            dagger_cfg["distill_loss"] = args_cli.distill_loss
        if args_cli.init_sigma_from_teacher_block is not None:
            dagger_cfg["init_sigma_from_teacher_block_id"] = args_cli.init_sigma_from_teacher_block
        if args_cli.deterministic_rollouts:
            dagger_cfg["deterministic_rollouts"] = True
        if args_cli.replay_capacity is not None:
            dagger_cfg["replay_capacity"] = args_cli.replay_capacity
        if args_cli.replay_updates is not None:
            dagger_cfg["replay_updates"] = args_cli.replay_updates

        # ---------------- Critic toggle. ----------------
        if args_cli.critic == "symmetric":
            agent_cfg["params"]["config"].pop("central_value_config", None)
            agent_cfg["params"]["network"]["symmetric_critic"] = True
        else:
            agent_cfg["params"]["network"]["symmetric_critic"] = False

        # ---------------- Observers + wandb. ----------------
        # EnvStatsAlgoObserver = every raw env extra (debugging firehose);
        # DistillMetricsObserver = the curated `metrics/` namespace (headline numbers).
        observers = [EnvStatsAlgoObserver(), DistillMetricsObserver()]
        if args_cli.wandb_activate:
            from isaacsimenvs.utils.wandb_utils import WandbAlgoObserver

            wandb_cfg = OmegaConf.create(
                {
                    "wandb_activate": True,
                    "wandb_project": args_cli.wandb_project,
                    "wandb_group": args_cli.wandb_group,
                    "wandb_entity": args_cli.wandb_entity,
                    "wandb_name": args_cli.wandb_name or agent_cfg["params"]["config"]["name"],
                    "wandb_tags": list(args_cli.wandb_tags),
                    "wandb_notes": args_cli.wandb_notes,
                    "wandb_logcode_dir": args_cli.wandb_logcode_dir,
                }
            )
            observers.append(WandbAlgoObserver(wandb_cfg))

        # ---------------- Build Runner with custom algo registrations. ----------------
        runner = Runner(MultiObserver(observers))
        runner.algo_factory.register_builder(
            "dagger_sapg", lambda **kw: DAggerA2CAgent(**kw)
        )
        runner.algo_factory.register_builder(
            "dagger_ppo", lambda **kw: DAggerA2CAgent(**kw)
        )

        agent_cfg["params"]["config"]["train_dir"] = hydra_run_dir
        agent_cfg["params"]["config"]["device"] = args_cli.rl_device
        agent_cfg["params"]["config"]["device_name"] = args_cli.rl_device
        if multi_gpu:
            agent_cfg["params"]["config"]["multi_gpu"] = True

        runner.load(agent_cfg)
        runner.reset()
        runner.run(
            {
                "train": True,
                "play": False,
                "checkpoint": args_cli.checkpoint,
                "checkpoint_load_mode": args_cli.checkpoint_load_mode,
            }
        )

    run()

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
