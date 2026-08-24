"""DAgger loop: distill the state-based teacher into the depth student.

Phase 5, from ``~/DEXTRAH/dextrah_lab/distillation/distillation.py``. The loop
shape is preserved -- roll out, label with the teacher, accumulate loss over
``seq_length`` steps, one gradient step -- with the deviations below.

BPTT without a second forward pass
----------------------------------
Upstream accumulates ``total_loss`` across steps and steps the optimizer every
``seq_length`` iterations, detaching the hidden state only at that point
(``distillation.py:528-540``). That gives real ``seq_length``-step BPTT while
reusing the rollout's forward pass as the training forward -- no replay buffer,
no second pass over stored images, and no ``(N*T, F)`` batch to lay out.

That last point matters: the student network reshapes such a batch env-major,
while a rollout naturally accumulates time-major, and a mismatch is silent (see
``a2c_aux_cnn.py``'s ``forward`` docstring). Accumulating per-step losses
sidesteps the trap entirely.

The cost is memory: ``seq_length`` forward graphs are alive at once, including
the conv activations. At 256 envs x 16 steps that is on the order of a couple of
GB on top of the env. Lower ``seq_length`` before lowering ``num_envs`` if it
does not fit -- ``num_envs`` is the batch size here.

Deviations from the DEXTRAH source
----------------------------------
1. **No DDP.** Upstream hard-requires ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``
   (``distillation.py:101-103``) and wraps in DDP. Single-GPU only here.

2. **The teacher-driven warmup is honored.** Upstream writes
   ``if log_counter < 15_000: beta = 1.`` and then unconditionally clobbers it
   with ``beta = 0.`` two lines later (``:376``). DEXTRAH gets away with
   ``beta=0`` because its geometric fabric bounds the reachable state set: 11
   bounded actions through a damped second-order controller mean even a garbage
   policy produces smooth, collision-free motion. play2perfect has no fabric --
   29 raw joint targets mean a cold student reaches genuinely unrecoverable
   states, and the teacher's LSTM goes off-distribution along the student's
   trajectory, so its labels there are worthless. **This is the single most
   important deviation.**

3. **``seq_length`` is honored.** Upstream reads the config value and then
   overwrites it with 1 (``:177``), so the recurrent weights never learn to
   write a predictively-useful hidden state. Note the consequence for the
   throughput budget: with ``seq_length=16``, N env steps buy N/16 gradient
   steps, not N.

4. **Dropped:** the per-step ``torch.cuda.empty_cache()`` (``:540``) and the
   done-time flush at ``:572`` (unreachable at ``seq_length == 1``, and
   redundant with the periodic step otherwise).

5. **``mus`` are not weighted by the teacher's ``sigmas``.** Upstream uses
   ``weights = (1 / actions_teacher['sigmas'][0]) ** 2``. Those sigmas index
   only on the SAPG block id, never on the observation, so they are a constant
   29-vector -- and for the block with the best success rate they span 1054x,
   i.e. 1.1e6x in ``1/sigma^2``, which would silently zero the loss on whichever
   joints that block happens to be noisy on. See
   ``isaacsimenvs/distillation/teacher.py``. Default here is uniform;
   ``mu_weight_mode="inv_sigma2"`` reproduces upstream for comparison.

6. **The ``sigmas`` matching term is off by default** (``sigma_loss_coef: 0.0``).
   The plan's loss is ``weighted L2 on mus + L2 on sigmas + aux``, but measured
   at init the sigma term carries **2806 of a 2806 total** -- roughly 450x the
   ``mus`` term -- because the teacher's block-50 sigmas reach 170 while the
   student starts at ``exp(0) = 1``, and ``(170 - 1)^2 / 29`` alone is ~990.

   It does not corrupt learning: ``fixed_sigma: True`` makes the student's sigma
   a standalone parameter, and the network's ``sigma = mu * 0.0 + sigma_param``
   zeroes the gradient path to the trunk (verified: trunk grad 0.0, sigma-param
   grad 338). But it makes ``total`` useless as a progress signal, and what it
   *does* train is the student's sigma toward a per-block SAPG exploration
   hyperparameter that describes the teacher's training schedule, not the task.
   Raise the coefficient only if you have a reason to want that.

7. **``dones`` are threaded and hidden state is masked, not indexed.** Upstream
   never puts ``dones`` in the student batch dict, which is harmless only at
   ``seq_length == 1``. Zeroing is done with a multiplicative mask rather than
   in-place index assignment, so it stays inside the autograd graph.

Also avoided: upstream's aux-loss block reads ``obs["mask"]``
(``distillation.py:465``) unconditionally, before the per-target loop, even
though the mask is only used by image-reconstruction heads. Porting that
verbatim while skipping segmentation masks would ``KeyError`` before any loss
was computed.
"""

from __future__ import annotations

import time
from collections import deque

import torch


class Dagger:
    """Single-GPU DAgger from a frozen teacher into a recurrent depth student.

    Args:
        env: the raw (unwrapped) gym env, with ``student_obs`` enabled and
            ``emit_in_observations`` True, so ``_get_observations`` yields
            ``proprio`` / ``img`` / ``teacher_obs`` / ``critic`` / ``aux_info``.
        agent_cfg: ``PreciseAssemblyStudent.yaml`` as a dict.
        teacher: an ``isaacsimenvs.distillation.Teacher``.
    """

    def __init__(
        self,
        env,
        agent_cfg: dict,
        teacher,
        *,
        device: str = "cuda:0",
        log_every: int = 100,
    ) -> None:
        from rl_games.algos_torch.model_builder import ModelBuilder

        from isaacsimenvs.distillation.a2c_aux_cnn import register_student_networks

        self.env = env
        self.teacher = teacher
        self.device = torch.device(device)
        self.log_every = int(log_every)

        params = agent_cfg["params"]
        cfg = params["config"]
        self.num_envs = int(env.num_envs)
        self.seq_length = int(cfg.get("seq_length", 1))
        self.aux_coeff = float(cfg.get("aux_coeff", 10.0))
        self.grad_norm = float(cfg.get("grad_norm", 1.0))
        self.truncate_grads = bool(cfg.get("truncate_grads", True))
        self.max_iters = int(cfg.get("max_iters", 100_000))
        self.beta_warmup_iters = int(cfg.get("beta_warmup_iters", 15_000))
        self.beta_anneal_iters = int(cfg.get("beta_anneal_iters", 0))
        self.sigma_loss_coef = float(cfg.get("sigma_loss_coef", 0.0))
        self.mu_weight_mode = str(cfg.get("mu_weight_mode", "uniform"))
        if self.mu_weight_mode not in ("uniform", "inv_sigma2"):
            raise ValueError(
                f"mu_weight_mode must be 'uniform' or 'inv_sigma2', "
                f"got {self.mu_weight_mode!r}"
            )

        self.aux_targets = list(params["network"]["aux_outputs"].keys())
        self.action_dim = int(env.cfg.action_space)

        register_student_networks()
        self.student_model = (
            ModelBuilder()
            .load(params)
            .build({
                "actions_num": self.action_dim,
                "input_shape": (self._infer_proprio_dim(),),
                "num_seqs": self.num_envs,
                "value_size": 1,
                "normalize_value": bool(cfg.get("normalize_value", True)),
                "normalize_input": bool(cfg.get("normalize_input", True)),
            })
            .to(self.device)
        )
        self.student_net = self.student_model.a2c_network
        self.optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=float(cfg.get("learning_rate", 1e-4)),
            eps=1e-8,
        )

        self._states = [s.to(self.device) for s in self.student_net.get_default_rnn_state()]
        self._prev_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self.iter = 0
        self.grad_steps = 0
        self.history: list[dict] = []
        self._recent = deque(maxlen=self.log_every)

    # -- setup helpers -------------------------------------------------------
    def _infer_proprio_dim(self) -> int:
        """Proprio width, read off the env rather than assumed."""
        obs = self.env._get_observations()
        if "proprio" not in obs:
            raise KeyError(
                "env does not emit 'proprio'. DAgger needs student_obs.enabled "
                "AND student_obs.emit_in_observations=True; with the latter "
                "False the env keeps the {policy, critic} contract."
            )
        return int(obs["proprio"].shape[-1])

    # -- schedule ------------------------------------------------------------
    def beta(self, it: int) -> float:
        """Probability an env is stepped by the TEACHER rather than the student.

        ``beta=1`` is a pure teacher rollout: the student still trains on every
        label, it just does not yet steer. Deviation 2 above is why this exists
        rather than being pinned to 0.
        """
        if it < self.beta_warmup_iters:
            return 1.0
        if self.beta_anneal_iters > 0:
            past = it - self.beta_warmup_iters
            if past < self.beta_anneal_iters:
                return 1.0 - past / self.beta_anneal_iters
        return 0.0

    # -- one student forward -------------------------------------------------
    def student_step(self, obs: dict) -> dict:
        """One ``seq_length=1`` forward. States stay attached for BPTT."""
        batch = {
            "is_train": True,
            "prev_actions": self._prev_actions,
            "obs": obs["proprio"],
            "img": obs["img"],
            "rnn_states": self._states,
            "seq_length": 1,
            "rnn_masks": None,
        }
        res = self.student_model(batch)
        self._states = list(res["rnn_states"])
        return res

    # -- loss ----------------------------------------------------------------
    def compute_loss(self, student_res, teacher_out, aux_gt) -> tuple[torch.Tensor, dict]:
        mu_err = student_res["mus"] - teacher_out["mus"]
        if self.mu_weight_mode == "inv_sigma2":
            # Upstream's weighting, kept only for comparison. See deviation 5.
            w = (1.0 / teacher_out["sigmas"][0]).pow(2).detach()
            mu_loss = (mu_err.pow(2) * w).mean()
        else:
            mu_loss = mu_err.pow(2).mean()

        # Off by default -- see deviation 7. Still computed so it stays visible
        # in the log even at coefficient zero.
        sigma_loss = (student_res["sigmas"] - teacher_out["sigmas"]).pow(2).mean()

        aux_out = self.student_net.get_aux_outputs()
        aux_losses = {}
        for name in self.aux_targets:
            target = aux_gt[name].reshape(self.num_envs, -1).detach()
            aux_losses[name] = (aux_out[name] - target).pow(2).mean()

        total = (
            mu_loss
            + self.sigma_loss_coef * sigma_loss
            + self.aux_coeff * sum(aux_losses.values())
        )
        parts = {
            "mu": float(mu_loss.detach()),
            "sigma": float(sigma_loss.detach()),
            **{f"aux_{k}": float(v.detach()) for k, v in aux_losses.items()},
            "total": float(total.detach()),
        }
        # RMS error on hole_pos in millimetres: the number that decides whether
        # this student can do the task at all (2 mm spec).
        if "hole_pos" in aux_out:
            err = (aux_out["hole_pos"] - aux_gt["hole_pos"]).detach()
            parts["hole_rmse_mm"] = float(err.pow(2).sum(-1).mean().sqrt() * 1000.0)
        return total, parts

    # -- main loop -----------------------------------------------------------
    def distill(self, max_iters: int | None = None) -> list[dict]:
        """Run the loop. ``max_iters`` counts ENV STEPS, not gradient steps."""
        max_iters = int(max_iters if max_iters is not None else self.max_iters)
        self.student_model.train()

        obs, _ = self.env.reset()
        self.teacher.reset_states()
        self.optimizer.zero_grad(set_to_none=True)

        window_loss = 0.0
        accum: torch.Tensor | None = None
        t_start = time.time()

        while self.iter < max_iters:
            beta = self.beta(self.iter)

            teacher_out = self.teacher.act(obs["teacher_obs"])
            student_res = self.student_step(obs)

            loss, parts = self.compute_loss(student_res, teacher_out, obs["aux_info"])
            accum = loss if accum is None else accum + loss
            parts["beta"] = beta
            self._recent.append(parts)
            window_loss += parts["total"]

            # --- who steps the env ---------------------------------------
            # Written plainly; upstream double-thresholds a boolean against beta
            # (`p = rand > beta` then `p > beta`), which happens to work only
            # because bool casts to {0, 1}.
            student_action = torch.clamp(student_res["mus"].detach(), -1.0, 1.0)
            if beta >= 1.0:
                action = teacher_out["action"]
            elif beta <= 0.0:
                action = student_action
            else:
                use_teacher = (
                    torch.rand(self.num_envs, device=self.device) < beta
                ).unsqueeze(-1)
                action = torch.where(use_teacher, teacher_out["action"], student_action)

            self._prev_actions = action.detach()
            obs, _rew, terminated, truncated, _extras = self.env.step(action.detach())
            dones = (terminated | truncated).reshape(-1).bool()

            if bool(dones.any()):
                # Multiplicative mask, not in-place indexing: these tensors are
                # inside the live BPTT graph.
                keep = (~dones).to(self._states[0].dtype).view(1, -1, 1)
                self._states = [s * keep for s in self._states]
                self.teacher.reset_states(dones)
                self._prev_actions = self._prev_actions * (~dones).unsqueeze(-1).to(
                    self._prev_actions.dtype
                )

            self.iter += 1

            # --- gradient step every seq_length env steps ----------------
            if self.iter % self.seq_length == 0:
                self.optimizer.zero_grad(set_to_none=True)
                accum.backward()
                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(), self.grad_norm
                    )
                self.optimizer.step()
                self.grad_steps += 1
                accum = None
                self._states = [s.detach() for s in self._states]

            if self.iter % self.log_every == 0:
                rec = self._summarize(beta, t_start)
                self.history.append(rec)
                print(
                    f"[dagger] iter {self.iter:>7d} grad {self.grad_steps:>6d} "
                    f"beta {beta:.2f} total {rec['total']:.4f} mu {rec['mu']:.4f} "
                    f"hole_rmse {rec.get('hole_rmse_mm', float('nan')):.1f}mm "
                    f"{rec['steps_per_s']:.1f} steps/s",
                    flush=True,
                )
                window_loss = 0.0

        return self.history

    def _summarize(self, beta: float, t_start: float) -> dict:
        keys = [k for k in self._recent[0] if k != "beta"]
        rec = {k: sum(r[k] for r in self._recent) / len(self._recent) for k in keys}
        rec.update(
            iter=self.iter,
            grad_steps=self.grad_steps,
            beta=beta,
            steps_per_s=self.iter / max(1e-9, time.time() - t_start),
        )
        return rec

    # -- checkpointing -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "model": self.student_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter": self.iter,
            "grad_steps": self.grad_steps,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)


__all__ = ["Dagger"]
