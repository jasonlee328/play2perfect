"""Curated training metrics for depth-student distillation.

rl_games + ``EnvStatsAlgoObserver`` already dump every env ``extras`` key to
tensorboard/wandb (``successes``, ``success_ratio``, ``episode_final/*``,
``successes_per_block/*`` …). That firehose is useful for debugging but makes
it hard to answer "is the student working?" at a glance. This module adds one
small, stable namespace on top — ``metrics/`` — with a fixed set of
episode-level numbers, each computed the unbiased way (at episode end, over
the last ``games_to_track`` finished episodes):

    metrics/success_rate            fraction of finished episodes where every
                                    insertion goal was hit (and, when retract is
                                    enabled, the retract succeeded). The headline
                                    number.
    metrics/insertion_rate          fraction of finished episodes where all
                                    insertion goals were hit (ignores retract).
    metrics/insertions_per_episode  raw insertion count per finished episode.
    metrics/retract_success_rate    fraction of finished episodes whose retract
                                    succeeded (only when retract is enabled).
    metrics/episode_length          mean policy steps per finished episode.
    metrics/episodes_finished       finished episodes in the tracking window.

The distillation-loss scalars live next door under ``distill/`` and are
written by ``DAggerA2CAgent`` (see ``dagger_agent.py::_log_distill_metrics``):

    distill/imitation_loss          λ_D-weighted target: per-dim MSE (or NLL)
                                    between student μ and teacher label
    distill/action_error_rms        sqrt(MSE) — same units as the actions
    distill/lambda_d                current imitation weight
    distill/teacher_action_rms      RMS of the teacher labels
    distill/student_action_rms      RMS of the student μ

See ``docs/distillation_metrics.md`` for the full table incl. the raw keys.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from rl_games.common.algo_observer import AlgoObserver

from isaacsimenvs.utils.rlgames_utils import _value_at

# (metric name, episode_final key). Missing keys are skipped silently so the
# observer works for envs that don't define retract / successes.
_EPISODE_FINAL_SOURCES: tuple[tuple[str, str], ...] = (
    ("insertion_rate", "all_goals_hit"),
    ("insertions_per_episode", "successes"),
    ("retract_success_rate", "retract_success"),
)


class DistillMetricsObserver(AlgoObserver):
    """Emit the curated ``metrics/`` namespace once per epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.algo = None
        self.writer = None
        self._buffers: dict[str, deque] = {}
        self._episode_lengths: deque | None = None
        self._dirty = False

    # -- rl_games hooks -----------------------------------------------------
    def after_init(self, algo) -> None:
        self.algo = algo
        self.writer = algo.writer
        n = int(getattr(algo, "games_to_track", 3000))
        for name, _ in _EPISODE_FINAL_SOURCES:
            self._buffers[name] = deque([], maxlen=n)
        self._buffers["success_rate"] = deque([], maxlen=n)
        self._episode_lengths = deque([], maxlen=n)

    def process_infos(self, infos, done_indices, **kwargs) -> None:
        if not isinstance(infos, dict) or self.algo is None:
            return
        done = done_indices.reshape(-1).detach().cpu().tolist()
        if not done:
            return
        final = infos.get("episode_final") or {}
        all_hit = final.get("all_goals_hit")
        retract = final.get("retract_success")
        for idx in done:
            for name, key in _EPISODE_FINAL_SOURCES:
                if key in final:
                    self._buffers[name].append(_value_at(final[key], idx))
            if all_hit is not None:
                ok = _value_at(all_hit, idx) >= 0.5
                if retract is not None:
                    ok = ok and _value_at(retract, idx) >= 0.5
                self._buffers["success_rate"].append(1.0 if ok else 0.0)
        # Episode lengths: rl_games keeps current lengths in algo.current_lengths
        # and zeroes them on done *after* observers run, so read them here.
        cur = getattr(self.algo, "current_lengths", None)
        if isinstance(cur, torch.Tensor) and self._episode_lengths is not None:
            cur_cpu = cur.detach().cpu()
            for idx in done:
                if idx < cur_cpu.numel():
                    self._episode_lengths.append(float(cur_cpu[idx].item()))
        self._dirty = True

    def after_clear_stats(self) -> None:
        for buf in self._buffers.values():
            buf.clear()
        if self._episode_lengths is not None:
            self._episode_lengths.clear()
        self._dirty = False

    def after_print_stats(self, frame, epoch_num, total_time) -> None:
        if self.writer is None or not self._dirty:
            return
        for name, buf in self._buffers.items():
            if buf:
                self.writer.add_scalar(f"metrics/{name}", float(np.mean(buf)), frame)
        if self._episode_lengths:
            self.writer.add_scalar("metrics/episode_length", float(np.mean(self._episode_lengths)), frame)
        self.writer.add_scalar("metrics/episodes_finished", float(len(self._buffers["success_rate"])), frame)
        self._dirty = False


__all__ = ["DistillMetricsObserver"]
