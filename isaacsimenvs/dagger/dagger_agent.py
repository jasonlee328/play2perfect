"""DAgger SAPG/PPO agent — subclass of rl_games' A2CAgent that adds an MSE
imitation term to the standard PPO/SAPG update.

Two algo names registered (`dagger_sapg`, `dagger_ppo`) point at the same
class; the SAPG-vs-PPO distinction is purely yaml-configured (via
``coef_cond`` / ``intr_reward_coef`` / ``expl_blocks`` / ``off_policy_ratio``
/ ``central_value_config`` knobs already supported by A2CAgent).

Override surface is intentionally minimal:
    - ``__init__``      build the teacher + cache λ_D scheduler params
    - ``init_tensors``  allocate experience_buffer['teacher_actions'] and add to tensor_list
    - ``play_steps``    monkey-patch ``self.env_step`` to record teacher labels per step
    - ``prepare_dataset`` patch teacher_actions into the dataset values dict
    - ``calc_gradients`` copy of parent body with one line added to mix in λ_D · MSE

Loss is `(1 - λ_D) · L_PPO + λ_D · MSE(student_μ, teacher_μ)` with a linear
λ_D curriculum from `lambda_d_start → lambda_d_floor` over
`lambda_d_decay_frac × max_epochs`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common import common_losses

from .teacher import Teacher, latest_checkpoint


class DAggerA2CAgent(A2CAgent):
    """rl_games A2CAgent + DAgger MSE imitation term.

    Args (via ``params['config']['dagger']``):
        teacher_run_dir        Directory containing the teacher's rl_games checkpoint(s).
        teacher_checkpoint     Optional explicit .pth path; else newest under run_dir.
        teacher_task_id        Gym task id used to load the teacher (must match the
                               teacher's training env so num_actors / obs_shape align).
        teacher_agent_key      Hydra entry-point key for the teacher's train yaml.
        lambda_d_start         λ_D at epoch 0.
        lambda_d_floor         λ_D after the decay window.
        lambda_d_decay_frac    Fraction of max_epochs over which λ_D linearly decays.
    """

    def __init__(self, base_name: str, params: dict):
        super().__init__(base_name, params)

        cfg = params["config"]["dagger"]
        ckpt = Path(cfg["teacher_checkpoint"]) if cfg.get("teacher_checkpoint") else latest_checkpoint(cfg["teacher_run_dir"])

        # Teacher was trained on the env's *actor obs* (obs_list, ~140-d). The
        # live env exposes that as ``teacher_obs`` in addition to the student's
        # depth+proprio policy obs. The central ``RlGamesVecEnvWrapper`` knows
        # both — read env_info from it via ``teacher_env_info`` (single source
        # of truth for shapes AND yaml-derived clip bounds).
        from isaacsimenvs.utils.rlgames_utils import teacher_env_info
        wrapped = self.vec_env.env if hasattr(self.vec_env, "env") else self.vec_env
        teacher_info = teacher_env_info(wrapped) if hasattr(wrapped, "teacher_obs_space") else None

        self.teacher = Teacher(
            task_id=cfg.get("teacher_task_id", params["config"]["env_name"]),
            agent_key=cfg.get("teacher_agent_key", "rl_games_sapg_cfg_entry_point"),
            checkpoint_path=ckpt,
            num_envs=self.num_actors,
            rl_device=str(self.ppo_device),
            env_info=teacher_info,
        )
        # SAPG default is `linspace(50, 0, num_envs)`, but the network's block
        # lookup is exact-equality argmax against {50, 40, 30, 20, 10, 0} so
        # off-grid values fall through to block 0 (=50) anyway. Pin explicitly;
        # tune via `teacher_block_id` in the dagger yaml.
        self.teacher.pin_block_id(float(cfg.get("teacher_block_id", 50.0)))

        # Teacher reads `self.obs["teacher"]` (the un-augmented actor obs that
        # `DAggerRlGamesVecEnvWrapper` passes through). SAPG's block-id is
        # appended internally by `Teacher.get_action`.
        self._teacher_obs_key = "teacher"

        self.lambda_d_start = float(cfg.get("lambda_d_start", 1.0))
        self.lambda_d_floor = float(cfg.get("lambda_d_floor", 0.1))
        self.lambda_d_decay_frac = float(cfg.get("lambda_d_decay_frac", 0.5))

        # Value-only warmup: first `value_warmup_epochs` updates train the
        # critic + BC MSE only, with the actor PPO / entropy / bounds losses
        # zeroed. λ_D stays pinned to `lambda_d_start` during warmup, then the
        # regular decay schedule kicks in over the remaining max_epochs.
        self.value_warmup_epochs = int(cfg.get("value_warmup_epochs", 0))

        # Distillation objective. "mse" matches teacher μ via squared error
        # on the student μ head (σ never gets gradient). "nll" minimizes the
        # closed-form expected NLL of teacher samples under the student's
        # full Gaussian, which trains both μ_s and σ_s to match the teacher's
        # distribution. The σ pull comes from the `1/σ_s²` weighting on the
        # μ error plus the `log σ_s` regularizer.
        self.distill_loss_type = str(cfg.get("distill_loss", "mse")).lower()
        if self.distill_loss_type not in ("mse", "nll"):
            raise ValueError(f"distill_loss must be 'mse' or 'nll', got {self.distill_loss_type!r}")

        # If set, after the student checkpoint is loaded, the student's
        # log_sigma is overwritten with the teacher's log_sigma for the
        # specified block id. For `coef_cond` student networks, all blocks
        # get the same value (one teacher row broadcast across all student
        # rows). Set to None (default) to leave the student's σ at its init.
        self.init_sigma_from_teacher_block_id = cfg.get("init_sigma_from_teacher_block_id", None)
        if self.init_sigma_from_teacher_block_id is not None:
            self.init_sigma_from_teacher_block_id = float(self.init_sigma_from_teacher_block_id)

        # If True, training rollouts use the student's deterministic μ instead
        # of sampling from N(μ, σ). Still DAgger (student visits states, teacher
        # labels), but with no exploration noise. Useful when σ never gets
        # gradient (pure-BC), avoids ±0.37 jitter on every action. PPO ratio
        # stays = 1 (replay log_prob computed at μ matches stored log_prob at μ).
        self.deterministic_rollouts = bool(cfg.get("deterministic_rollouts", False))

        # Counter incremented inside the wrapped env_step (one per env-step).
        self._dagger_step_counter = 0
        self._dagger_buffer_allocated = False

        # Optional override of the rl_games scheduler bounds. rl_games hardcodes
        # AdaptiveScheduler.min_lr=1e-6 and max_lr=1e-2, but those are reachable
        # via Hydra by setting agent.params.config.scheduler_min_lr / _max_lr.
        if hasattr(self.scheduler, "min_lr"):
            cfg_min_lr = self.config.get("scheduler_min_lr", None)
            if cfg_min_lr is not None:
                self.scheduler.min_lr = float(cfg_min_lr)
        if hasattr(self.scheduler, "max_lr"):
            cfg_max_lr = self.config.get("scheduler_max_lr", None)
            if cfg_max_lr is not None:
                self.scheduler.max_lr = float(cfg_max_lr)

        # During value warmup the loss is pure MSE (no PPO actor gradient),
        # which makes the policy KL between snapshots large (~0.2-0.6 vs the
        # default kl_threshold of 0.008). An AdaptiveScheduler would halve LR
        # every epoch and floor it at 1e-6 well before warmup ends. So we wrap
        # the rl_games scheduler.update to be a no-op while epoch_num <
        # value_warmup_epochs; the adaptive logic engages from epoch_num >=
        # value_warmup_epochs onward, when PPO clipping keeps KL bounded.
        if self.value_warmup_epochs > 0:
            _orig_scheduler_update = self.scheduler.update

            def _gated_scheduler_update(current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
                if self.epoch_num < self.value_warmup_epochs:
                    return current_lr, entropy_coef
                return _orig_scheduler_update(current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs)

            self.scheduler.update = _gated_scheduler_update

    # ----- lenient checkpoint load (cross-critic-arch transfer) -----

    def set_weights(self, weights):
        """Lenient state_dict load so a symmetric-critic checkpoint can seed
        an asymmetric-critic run (or vice versa). The shared trunk + actor
        heads transfer; the `a2c_network.value_head.*` keys present only in
        symmetric checkpoints are dropped, and the asymmetric central value
        net keeps its fresh init.

        Also runs the optional ``init_sigma_from_teacher_block_id`` override
        AFTER the checkpoint load — otherwise the loaded log_sigma would
        clobber any earlier init."""
        missing, unexpected = self.model.load_state_dict(weights["model"], strict=False)
        if missing:
            print(f"[DAgger weights load] missing keys (kept at init): {missing}")
        if unexpected:
            print(f"[DAgger weights load] unexpected keys (skipped): {unexpected}")
        self.set_stats_weights(weights)
        if self.init_sigma_from_teacher_block_id is not None:
            self._init_log_sigma_from_teacher(self.init_sigma_from_teacher_block_id)

    def _init_log_sigma_from_teacher(self, block_id: float) -> None:
        """Copy teacher's effective sigma at the given block id into every
        row of the student's log_sigma parameter.

        Handles the naming mismatch between rl_games' standard SAPG network
        (parameter ``a2c_network.sigma`` + optional ``sigma_act``) and the
        depth_cnn_lstm student (parameter ``a2c_network.log_sigma``, no act).
        The teacher's effective σ is read by applying ``sigma_act`` to its
        raw param; the student stores ``log σ`` directly.
        """
        t_net = self.teacher.player.model.a2c_network
        s_net = self.model.a2c_network
        # --- Teacher: locate raw param + apply sigma_act to get effective σ ---
        if hasattr(t_net, "log_sigma"):
            t_param, teacher_is_logspace = t_net.log_sigma, True
        elif hasattr(t_net, "sigma") and isinstance(t_net.sigma, torch.nn.Parameter):
            t_param, teacher_is_logspace = t_net.sigma, False
        else:
            print("[DAgger sigma init] teacher has no recognizable sigma parameter; skipping")
            return
        if hasattr(t_net, "sigma_ids") and t_param.ndim == 2:
            ids = t_net.sigma_ids.detach().cpu().reshape(-1)
            match = (ids == float(block_id)).nonzero(as_tuple=True)[0]
            if match.numel() == 0:
                print(f"[DAgger sigma init] teacher has no block {block_id}; using row 0 ({float(ids[0])})")
                row = t_param[0]
            else:
                row = t_param[int(match[0])]
        else:
            row = t_param.reshape(-1)
        row = row.detach()
        if teacher_is_logspace:
            actual_sigma = torch.exp(row)
        elif hasattr(t_net, "sigma_act"):
            actual_sigma = t_net.sigma_act(row)
        else:
            actual_sigma = row
        # Floor at 0.05: rl_games SAPG with `sigma_activation: None` lets the
        # teacher's σ parameter drift negative during training, which surfaces
        # as ≤ 0 entries on some action dims (3 of 29 in this checkpoint).
        # Copying those into the student would give σ_s ≈ 0, making the NLL
        # term (μ_s-μ_t)²/σ_s² explode. 0.05 is a sane minimum exploration
        # scale for action units of order O(1).
        actual_sigma = actual_sigma.clamp(min=0.05)
        target_log_sigma = torch.log(actual_sigma)
        # --- Student: write into log_sigma (depth_cnn_lstm naming) ---
        if not hasattr(s_net, "log_sigma"):
            print("[DAgger sigma init] student has no log_sigma; skipping")
            return
        with torch.no_grad():
            if s_net.log_sigma.ndim == 2:
                target = target_log_sigma.to(s_net.log_sigma.device).unsqueeze(0).expand_as(s_net.log_sigma)
                s_net.log_sigma.copy_(target)
                print(f"[DAgger sigma init] copied teacher block {block_id} sigma into all "
                      f"{s_net.log_sigma.shape[0]} student blocks; sigma~={actual_sigma.tolist()}")
            else:
                s_net.log_sigma.copy_(target_log_sigma.to(s_net.log_sigma.device))
                print(f"[DAgger sigma init] copied teacher block {block_id} sigma "
                      f"({s_net.log_sigma.shape[-1]} dims); sigma~={actual_sigma.tolist()}")

    # ----- buffers -----

    def init_tensors(self):
        super().init_tensors()
        # Allocate teacher_actions tensor inside the experience buffer (same
        # (T, B, A) shape as 'actions'), so prepare_dataset / dataset slicing
        # treats it like any other rollout tensor.
        if not self._dagger_buffer_allocated:
            self.experience_buffer.tensor_dict["teacher_actions"] = torch.zeros(
                (self.horizon_length, self.num_actors, self.actions_num),
                dtype=torch.float32,
                device=self.ppo_device,
            )
            # Teacher's per-dim sigma at each rollout step, used by NLL
            # distillation. Always allocated (cheap) so the buffer schema
            # is stable whether the loss is MSE or NLL.
            self.experience_buffer.tensor_dict["teacher_sigmas"] = torch.zeros(
                (self.horizon_length, self.num_actors, self.actions_num),
                dtype=torch.float32,
                device=self.ppo_device,
            )
            # Make sure swap_and_flatten01 sees these keys when building batch_dict.
            for k in ("teacher_actions", "teacher_sigmas"):
                if k not in self.tensor_list:
                    self.tensor_list = list(self.tensor_list) + [k]
            self._dagger_buffer_allocated = True

    # ----- deterministic rollout override -----

    def get_action_values(self, obs, *args, **kwargs):
        """Override to use student's μ as the rollout action when configured."""
        res_dict = super().get_action_values(obs, *args, **kwargs)
        if self.deterministic_rollouts:
            res_dict["actions"] = res_dict["mus"]
        return res_dict

    # ----- λ_D schedule -----

    def _lambda_d(self) -> float:
        # During value warmup, hold λ_D at start (the warmup loss branch
        # ignores it anyway, but logging still uses this value).
        if self.epoch_num < self.value_warmup_epochs:
            return self.lambda_d_start
        eff_epoch = self.epoch_num - self.value_warmup_epochs
        eff_max = max(self.max_epochs - self.value_warmup_epochs, 1)
        end_epoch = self.lambda_d_decay_frac * float(eff_max)
        if end_epoch <= 0 or eff_epoch >= end_epoch:
            return self.lambda_d_floor
        frac = float(eff_epoch) / end_epoch
        return self.lambda_d_start + (self.lambda_d_floor - self.lambda_d_start) * frac

    # ----- rollout: monkey-patch env_step to record teacher labels -----

    def play_steps(self):
        # Make sure teacher_actions buffer exists (defensive — should be set by init_tensors).
        if "teacher_actions" not in self.experience_buffer.tensor_dict:
            self.init_tensors()

        self._dagger_step_counter = 0
        original_env_step = self.env_step

        def wrapped_env_step(actions):
            # Query the teacher on the SAME obs we just fed to get_action_values,
            # so its RNN advances in lock-step with the student's. Always fetch
            # both μ and σ so the buffer schema is independent of the loss type.
            base_obs = self.obs[self._teacher_obs_key]
            with torch.no_grad():
                teacher_mu, teacher_sigma = self.teacher.get_action_and_sigma(base_obs)
            n = self._dagger_step_counter
            self.experience_buffer.tensor_dict["teacher_actions"][n].copy_(teacher_mu)
            self.experience_buffer.tensor_dict["teacher_sigmas"][n].copy_(teacher_sigma)
            self._dagger_step_counter += 1

            result = original_env_step(actions)
            # Reset teacher RNN on env dones — same indices the parent uses for its own RNN reset.
            dones = result[3]
            done_idx = dones.nonzero(as_tuple=False).flatten()
            if done_idx.numel() > 0:
                self.teacher.reset_idx(done_idx)
            return result

        self.env_step = wrapped_env_step
        try:
            return super().play_steps()
        finally:
            self.env_step = original_env_step

    # ----- dataset: ensure teacher_actions reaches calc_gradients via input_dict -----

    def prepare_dataset(self, batch_dict, train_value_mean_std=True):
        super().prepare_dataset(batch_dict, train_value_mean_std=train_value_mean_std)
        # Patch teacher_actions + teacher_sigmas into the same flat ordering as the rest
        # of the dataset. batch_dict was built with swap_and_flatten01 → shape (T*B, A).
        self.dataset.values_dict["teacher_actions"] = batch_dict["teacher_actions"]
        self.dataset.values_dict["teacher_sigmas"] = batch_dict["teacher_sigmas"]

    # ----- per-minibatch loss: copy of A2CAgent.calc_gradients with the DAgger term added -----

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)
        teacher_actions_batch = input_dict["teacher_actions"]   # (mb, action_dim) — teacher μ
        teacher_sigmas_batch = input_dict["teacher_sigmas"]     # (mb, action_dim) — teacher σ

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros((len(values), 1), device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(len(mu), device=self.ppo_device)

            if self.expl_type.startswith("mixed_expl") and self.config.get("expl_reward_type") == "entropy":
                ec_candidates = self.intr_reward_coef[:: self.intr_coef_block_size]
                ec_identifiers = self.intr_reward_coef_embd[:: self.intr_coef_block_size, 0].reshape(-1, 1)
                ec_indices = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == ec_identifiers).float(), dim=0)
                entropy_coef = ec_candidates[ec_indices]
            elif self.expl_type.startswith("simple") and self.config.get("expl_reward_type") == "entropy":
                entropy_coef = self.intr_reward_coef
            else:
                entropy_coef = self.entropy_coef

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, (entropy_coef * entropy).unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy_loss, b_loss = losses[0], losses[1], losses[2], losses[3]

            ppo_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_loss + b_loss * self.bounds_loss_coef

            # ---- DAgger distillation term ----
            # Per-element distill loss (no reduction yet) so we can mask + average like the other losses.
            if self.distill_loss_type == "nll":
                # Closed-form expected NLL of teacher samples under the student:
                #   E_{a~N(μ_t, σ_t)} [-log π_s(a)]
                #     = 0.5 * ((μ_s-μ_t)² + σ_t²) / σ_s²  +  log σ_s  + const
                # σ_s appears as a 1/σ_s² weight on μ-error and as a log σ_s
                # regularizer, so the student's σ is pulled toward teacher's σ.
                eps = 1e-6
                inv_var = 1.0 / (sigma * sigma + eps)
                nll_per_dim = (
                    0.5 * ((mu - teacher_actions_batch).pow(2) + teacher_sigmas_batch.pow(2)) * inv_var
                    + torch.log(sigma + eps)
                )
                distill_per_elem = nll_per_dim.mean(dim=-1, keepdim=True)  # (mb, 1)
            else:
                distill_per_elem = (mu - teacher_actions_batch).pow(2).mean(dim=-1, keepdim=True)
            (distill_loss,), _ = torch_ext.apply_masks([distill_per_elem], rnn_masks)
            # Keep `mse_loss` as an alias for the wandb scalar name `dagger/L_D`
            # so existing dashboards don't break.
            mse_loss = distill_loss

            lam_d = self._lambda_d()
            if self.epoch_num < self.value_warmup_epochs:
                # Value warmup: critic + BC distillation only. Suppress actor
                # PPO loss, entropy bonus, and bounds penalty so the policy
                # is updated by BC labels alone while the value head catches up.
                loss = distill_loss + 0.5 * c_loss * self.critic_coef
            else:
                loss = (1.0 - lam_d) * ppo_loss + lam_d * distill_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        all_grads = self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        contrib = torch.logical_and(ratio < 1.0 + curr_e_clip, ratio > 1.0 - curr_e_clip).float()

        extras = {
            "on_policy_contrib": contrib.mean().item(),
            "off_policy_contrib": 0,
            "on_policy_grads": all_grads.detach().cpu(),
            "off_policy_grads": torch.zeros_like(all_grads).cpu(),
        }
        # rl_games' train_epoch only forwards specific keys from `extras` to
        # the writer (a2c_common.py:1421-1426). Custom keys are dropped, so
        # write DAgger scalars directly to the SummaryWriter (synced to wandb
        # via tb-mirroring). `self.frame` is the global env-frame counter.
        if self.writer is not None:
            self.writer.add_scalar("dagger/L_D", mse_loss.detach().item(), self.frame)
            self.writer.add_scalar("dagger/lambda_d", lam_d, self.frame)
            self.writer.add_scalar("dagger/teacher_action_l2",
                                   teacher_actions_batch.detach().pow(2).mean().sqrt().item(),
                                   self.frame)
            self.writer.add_scalar("dagger/student_action_l2",
                                   mu.detach().pow(2).mean().sqrt().item(),
                                   self.frame)
        if self.expl_type.startswith("mixed_expl") and self.intr_reward_coef_embd is not None:
            bl_ids = self.intr_reward_coef_embd[:: self.intr_coef_block_size, 0].reshape(-1, 1)
            bl_idxs = torch.argmax((obs_batch[:, -self.intr_reward_coef_embd.shape[1]] == bl_ids).float(), dim=0)
            extras["entropies"] = [
                torch.nan_to_num(entropy[bl_idxs == i].detach().mean()).item()
                for i in range(self.num_actors // self.intr_coef_block_size)
            ]

        self.train_result = (
            a_loss,
            c_loss,
            torch_ext.apply_masks([entropy.unsqueeze(1)], rnn_masks)[0][0],
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
            extras,
        )
