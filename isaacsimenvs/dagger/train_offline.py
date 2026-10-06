"""Offline training of the depth student on aggregated DAgger data (no simulator).

Reads rollouts written by ``DAggerA2CAgent`` with ``--dump-dir`` (one
``rank<r>/rollout_<epoch>.npz`` per rollout, time-ordered ``(T, B, ...)``),
stitches consecutive rollouts of the same env into long windows, and regresses
the student's mean on the teacher's labels with the LSTM run over each window
(reset at episode starts, loss masked during a burn-in prefix).

The network is the run's own ``depth_cnn_lstm`` built from its agent config, and
checkpoints are written rl_games-style under ``<out>/0_<name>/nn/`` with the run's
``.hydra`` copied next to them, so TAMP/inference_maniflow_tactile_oct4.py
``--dagger_checkpoint`` evaluates them directly.

    python isaacsimenvs/dagger/train_offline.py --run-dir <dagger run dir> \\
        --data <dump_dir> [<dump_dir> ...] --out-dir <dir> [--steps 200000]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def _streams(data_dirs: list[str]) -> list[list[Path]]:
    """Contiguous runs of rollout files per (dump dir, rank): one list per stream."""
    out = []
    for d in data_dirs:
        for rank_dir in sorted(Path(d).glob("rank*")):
            files = sorted(rank_dir.glob("rollout_*.npz"))
            epochs = [int(f.stem.split("_")[1]) for f in files]
            run: list[Path] = []
            for f, e in zip(files, epochs):
                if run and e != int(run[-1].stem.split("_")[1]) + 1:
                    out.append(run)
                    run = []
                run.append(f)
            if run:
                out.append(run)
    return out


class _Cache:
    def __init__(self, cap: int = 64):
        self.cap, self.d = cap, OrderedDict()

    def get(self, f: Path) -> dict:
        if f in self.d:
            self.d.move_to_end(f)
            return self.d[f]
        z = np.load(f)
        v = {k: z[k] for k in ("image", "low", "teacher_mu", "dones")}
        self.d[f] = v
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", required=True, help="DAgger run dir (its .hydra config defines the network).")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default=None)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--burn-in", type=int, default=16)
    p.add_argument("--batch", type=int, default=128, help="Windows per gradient step.")
    p.add_argument("--groups-per-batch", type=int, default=4, help="Rollout groups sampled per batch.")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--block-id", type=float, default=50.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true", help="Continue from <out>/.../latest.pth if present.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_entity", default="ai2-robotics")
    p.add_argument("--wandb_project", default="foundation-touch")
    args = p.parse_args()

    from gymnasium.spaces import Box
    from omegaconf import OmegaConf
    from rl_games.torch_runner import Runner

    # Load the student network module straight from its file: importing the isaacsimenvs
    # package registers every Isaac Lab task, which needs a running simulator.
    import importlib.util

    from rl_games.algos_torch import model_builder

    spec = importlib.util.spec_from_file_location(
        "depth_cnn_lstm", Path(__file__).resolve().parent / "networks" / "depth_cnn_lstm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model_builder.register_network("depth_cnn_lstm", mod.DepthCNNLSTMBuilder)

    class _NumEnvsStub:
        def __init__(self, n: int) -> None:
            self.num_envs = int(n)

    run_dir = Path(args.run_dir)
    streams = _streams(args.data)
    if not streams:
        raise SystemExit(f"no rollouts under {args.data}")
    meta = json.loads((Path(streams[0][0]).parent / "meta.json").read_text())
    T = meta["horizon"]
    k = math.ceil(args.window / T) + 1  # rollouts per group so any offset fits a window
    groups = [(s, i) for s in streams for i in range(len(s) - k + 1)]
    n_roll = sum(len(s) for s in streams)
    print(f"[offline] {len(streams)} streams, {n_roll} rollouts (~{n_roll * T * meta['num_envs'] / 1e6:.1f}M samples), "
          f"{len(groups)} window groups", flush=True)
    if not groups:
        raise SystemExit("not enough consecutive rollouts for one window")

    # Network: the run's student, built as an rl_games player (no env / sim needed).
    acfg = OmegaConf.to_container(OmegaConf.load(run_dir / ".hydra" / "config.yaml"), resolve=True)["agent"]
    acfg["params"]["algo"]["name"] = "a2c_continuous"
    c = acfg["params"]["config"]
    c.pop("central_value_config", None)
    acfg["params"]["network"]["symmetric_critic"] = True
    c["device"] = c["device_name"] = args.device
    obs_dim = meta["image_channels"] * meta["image_hw"][0] * meta["image_hw"][1] + meta["proprio_dim"] \
        + meta["tactile_dim"] + 1
    A = meta["action_dim"]
    c["num_actors"] = args.batch
    # rl_games' SAPG path appends the block-id column to the declared obs size itself.
    c["env_info"] = {"observation_space": Box(-np.inf, np.inf, (obs_dim - 1,)), "state_space": None,
                     "action_space": Box(-1.0, 1.0, (A,)), "agents": 1, "value_size": 1}
    c.setdefault("player", {}).update(deterministic=True, print_stats=False, games_num=1)
    runner = Runner()
    runner.load(acfg)
    runner.reset()
    runner.set_vec_env(_NumEnvsStub(args.batch))
    player = runner.create_player()
    model = player.model
    net = model.a2c_network
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    name = args.name or f"offline_{run_dir.name}"
    out = Path(args.out_dir)
    nn_dir = out / f"0_{name}" / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)
    (out / ".hydra").mkdir(exist_ok=True)
    for f in ("config.yaml", "overrides.yaml", "hydra.yaml"):
        if (run_dir / ".hydra" / f).exists():
            shutil.copy2(run_dir / ".hydra" / f, out / ".hydra" / f)
    (out / "offline_args.json").write_text(json.dumps(vars(args), indent=2))

    step = 0
    latest = nn_dir / "latest.pth"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["epoch"])
        print(f"[offline] resumed at step {step}", flush=True)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group="dagger_offline",
                         name=name, id=name, resume="allow", config=vars(args) | {"meta": meta})

    def save(path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": step,
                    "frame": step * args.batch * args.window}, tmp)
        os.replace(tmp, path)

    cache = _Cache()
    dev = args.device
    H, Wd = meta["image_hw"]
    block = torch.full((1,), args.block_id, device=dev)
    loss_mask = torch.zeros(args.window, device=dev)
    loss_mask[args.burn_in:] = 1.0
    t0, losses = time.time(), []
    while step < args.steps:
        per = args.batch // args.groups_per_batch
        imgs, lows, labs, dns = [], [], [], []
        for _ in range(args.groups_per_batch):
            s, i = random.choice(groups)
            parts = [cache.get(f) for f in s[i:i + k]]
            cat = {key: np.concatenate([q[key] for q in parts], 0) for key in parts[0]}  # (k*T, B, ...)
            B = cat["image"].shape[1]
            envs = np.random.randint(B, size=per)
            offs = np.random.randint(0, k * T - args.window + 1, size=per)
            ti = offs[:, None] + np.arange(args.window)[None]  # (per, W)
            imgs.append(cat["image"][ti, envs[:, None]])
            lows.append(cat["low"][ti, envs[:, None]])
            labs.append(cat["teacher_mu"][ti, envs[:, None]])
            d = cat["dones"][ti, envs[:, None]].copy()
            d[:, 0] = True  # every window starts from a zero LSTM state
            dns.append(d)
        img = torch.from_numpy(np.concatenate(imgs)).to(dev).float() / 255.0      # (Bw, W, C*H*W)
        low = torch.from_numpy(np.concatenate(lows)).to(dev).float()
        lab = torch.from_numpy(np.concatenate(labs)).to(dev).float()
        dn = torch.from_numpy(np.concatenate(dns)).to(dev).float()
        Bw = img.shape[0]
        obs = torch.cat([img, low, block.expand(Bw, args.window, 1)], -1).reshape(Bw * args.window, -1)
        L = net.lstm_layers if hasattr(net, "lstm_layers") else 1
        hid = net.lstm_hidden
        states = (torch.zeros(L, Bw, hid, device=dev), torch.zeros(L, Bw, hid, device=dev))
        mu, _, _, _ = net({"obs": obs, "rnn_states": states, "seq_length": args.window,
                           "dones": dn.reshape(Bw * args.window, 1)})
        err = (mu.view(Bw, args.window, A) - lab).pow(2).mean(-1)                 # (Bw, W)
        loss = (err * loss_mask).sum() / (loss_mask.sum() * Bw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        opt.step()
        step += 1
        losses.append(loss.item())
        if step % args.log_every == 0:
            ml = float(np.mean(losses))
            losses = []
            print(f"[offline] step {step} loss {ml:.4f} {args.log_every / (time.time() - t0):.1f} it/s", flush=True)
            t0 = time.time()
            if run is not None:
                run.log({"offline/imitation_loss": ml}, step=step)
        if step % args.save_every == 0 or step == args.steps:
            save(latest)
            save(nn_dir / f"last_0_{name}_ep_{step}_rew_0.pth")
    if run is not None:
        run.finish()
    print(f"[offline] done: {nn_dir}", flush=True)


if __name__ == "__main__":
    main()
