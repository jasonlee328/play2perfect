"""Phase 4 check: the depth-student network.

Needs no simulator -- the network is pure torch, so this runs in seconds and is
the cheapest check in the project. It deliberately loads the network module by
path rather than importing `isaacsimenvs`, which would drag in every task
subpackage and require a launched Omniverse app.

Geometry is cross-checked against BOTH StudentObsCfg's declared defaults and the
task YAML overlay, since a silent mismatch trains the encoder on the wrong image
shape -- which is precisely what DEXTRAH's hardcoded 320x240 RGB does in the file
its own port plan calls the mono-depth variant.

    python tools/distillation/check_phase4_student_net.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

cli = argparse.ArgumentParser()
cli.add_argument("--num-envs", type=int, default=8)
cli.add_argument("--device", default="cuda:0")
args, _ = cli.parse_known_args()

import ast  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402

def _load_student_net_module():
    """Load a2c_aux_cnn.py by path, bypassing the isaacsimenvs package.

    `import isaacsimenvs` pulls in every task subpackage for its gym.register
    side effects, which requires a launched Omniverse SimulationApp. The student
    network needs only torch + rl_games, and keeping this check sim-free is the
    point: it turns network iteration into a two-second loop instead of a
    sixty-second one.
    """
    import importlib.util

    path = REPO / "isaacsimenvs/distillation/a2c_aux_cnn.py"
    spec = importlib.util.spec_from_file_location("_a2c_aux_cnn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_net_mod = _load_student_net_module()
CNN_OUT_FEATURES = _net_mod.CNN_OUT_FEATURES
CustomCNN = _net_mod.CustomCNN
A2CAuxCNNBuilder = _net_mod.A2CAuxCNNBuilder
register_student_networks = _net_mod.register_student_networks


def _student_obs_cfg_defaults() -> dict:
    """StudentObsCfg's declared defaults, read without importing isaaclab.

    A plain import of play_env_cfg pulls in isaaclab and bootstraps Omniverse
    Kit, which is absurd for comparing two integers -- and would make the one
    sim-free check in this project depend on a working Kit install.
    """
    src = (REPO / "isaacsimenvs/tasks/play/play_env_cfg.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "StudentObsCfg":
            out = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except ValueError:
                        pass
            return out
    raise RuntimeError("StudentObsCfg not found")

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


CFG_PATH = REPO / "isaacsimenvs/cfg/train/PreciseAssemblyStudent.yaml"
agent_cfg = yaml.safe_load(CFG_PATH.read_text())
net_params = agent_cfg["params"]["network"]

PROPRIO_DIM = 87
ACTIONS = 29
N = args.num_envs
dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

# --- 1. the yaml agrees with the env's student config ----------------------
print("\n1. student_image vs the env's student config")
si = net_params["student_image"]
# Two independent sources of truth: the configclass defaults and the task YAML
# overlay that actually configures the env at runtime. Both must agree, since a
# silent mismatch here trains the encoder on the wrong geometry -- the exact
# thing DEXTRAH's hardcoded 320x240 RGB would have done.
defaults = _student_obs_cfg_defaults()
task_yaml = yaml.safe_load(
    (REPO / "isaacsimenvs/cfg/task/PreciseAssembly.yaml").read_text()
)["student_obs"]

for src_name, src in (("StudentObsCfg defaults", defaults), ("task YAML", task_yaml)):
    check(f"height matches {src_name}",
          int(si["height"]) == int(src["image_input_height"]),
          f"yaml={si['height']} {src_name}={src['image_input_height']}")
    check(f"width matches {src_name}",
          int(si["width"]) == int(src["image_input_width"]),
          f"yaml={si['width']} {src_name}={src['image_input_width']}")
    exp_ch = 1 if str(src["image_modality"]).lower() == "depth" else 3
    check(f"channels match {src_name} modality", int(si["channels"]) == exp_ch,
          f"yaml={si['channels']} modality={src['image_modality']!r} -> {exp_ch}")

check("proprio_list is the 3 x 29-DOF fields -> 87",
      list(task_yaml["proprio_list"]) == ["joint_pos", "joint_vel", "prev_action_targets"]
      and 3 * 29 == PROPRIO_DIM,
      f"{task_yaml['proprio_list']}")
check("clip_observations matches the env clamp",
      float(agent_cfg["params"]["env"]["clip_observations"]) == 10.0)

# --- 2. conv geometry ------------------------------------------------------
print("\n2. conv stack geometry")
cnn = CustomCNN(int(si["height"]), int(si["width"]), in_channels=int(si["channels"]))
h, w = cnn.final_spatial
print(f"   final feature map: {cnn.final_channels} x {h} x {w}  "
      f"(discarded by avgpool: {cnn.final_channels * h * w - cnn.final_channels} values)")
check("conv stack does not collapse", h >= 1 and w >= 1, f"{h}x{w}")
y = cnn(torch.zeros(N, int(si["channels"]), int(si["height"]), int(si["width"])))
check(f"encoder outputs (N, {CNN_OUT_FEATURES})", tuple(y.shape) == (N, CNN_OUT_FEATURES),
      f"got {tuple(y.shape)}")

cnn_flat = CustomCNN(int(si["height"]), int(si["width"]),
                     in_channels=int(si["channels"]), spatial_pool="flatten")
yf = cnn_flat(torch.zeros(N, int(si["channels"]), int(si["height"]), int(si["width"])))
check("spatial_pool='flatten' also builds and runs",
      tuple(yf.shape) == (N, CNN_OUT_FEATURES),
      f"flat_size={cnn_flat.flat_size} vs avgpool {cnn.flat_size}")

# --- 3. build through rl_games' ModelBuilder ------------------------------
print("\n3. rl_games ModelBuilder integration")
register_student_networks()
from rl_games.algos_torch.model_builder import ModelBuilder  # noqa: E402

builder = ModelBuilder()
model = builder.load(agent_cfg["params"]).build({
    "actions_num": ACTIONS,
    "input_shape": (PROPRIO_DIM,),
    "num_seqs": N,
    "value_size": 1,
    "normalize_value": True,
    "normalize_input": True,
}).to(dev)
model.train()
check("model builds via ModelBuilder", model is not None)
check("model reports recurrent", bool(model.is_rnn()))
net = model.a2c_network
check("trunk input is proprio + CNN features", net.trunk_in == PROPRIO_DIM + CNN_OUT_FEATURES,
      f"got {net.trunk_in}")
check("model-level normalizer covers proprio only",
      tuple(model.running_mean_std.running_mean.shape) == (PROPRIO_DIM,),
      f"got {tuple(model.running_mean_std.running_mean.shape)}")
check("image normalizer is per-pixel over (C,H,W)",
      tuple(net.img_running_mean_std.running_mean.shape)
      == (int(si["channels"]), int(si["height"]), int(si["width"])),
      f"got {tuple(net.img_running_mean_std.running_mean.shape)}")

# --- 4. forward ------------------------------------------------------------
print("\n4. forward pass")
states = [s.to(dev) for s in net.get_default_rnn_state()]
check("default rnn state is (layers, N, units)",
      all(tuple(s.shape) == (net.rnn_layers, N, net.rnn_units) for s in states),
      f"got {[tuple(s.shape) for s in states]}")

batch = {
    "is_train": True,
    "prev_actions": torch.zeros(N, ACTIONS, device=dev),
    "obs": torch.randn(N, PROPRIO_DIM, device=dev),
    "img": torch.rand(N, int(si["channels"]), int(si["height"]), int(si["width"]), device=dev),
    "rnn_states": states,
    "seq_length": 1,
    "rnn_masks": None,
}
res = model(batch)
check("mus is (N, 29)", tuple(res["mus"].shape) == (N, ACTIONS), f"got {tuple(res['mus'].shape)}")
check("sigmas is (N, 29)", tuple(res["sigmas"].shape) == (N, ACTIONS))
check("mus and sigmas finite",
      bool(torch.isfinite(res["mus"]).all() and torch.isfinite(res["sigmas"]).all()))
check("sigmas positive", bool((res["sigmas"] > 0).all()),
      f"range=({float(res['sigmas'].min()):.4f}, {float(res['sigmas'].max()):.4f})")
check("rnn_states returned", "rnn_states" in res)

# --- 5. aux heads ----------------------------------------------------------
print("\n5. auxiliary heads")
aux = net.get_aux_outputs()
want = {k: v["size"] for k, v in net_params["aux_outputs"].items()}
check("aux keys match the yaml", set(aux) == set(want), f"got {sorted(aux)}")
for k, size in want.items():
    check(f"{k} is (N, {size})", tuple(aux[k].shape) == (N, size),
          f"got {tuple(aux[k].shape) if k in aux else None}")
check("aux outputs finite", all(bool(torch.isfinite(v).all()) for v in aux.values()))
check("aux targets match PreciseAssemblyEnv._aux_info() keys",
      set(want) == {"hole_pos", "keypoints_rel_goal", "object_pos"}, f"got {sorted(want)}")

# --- 6. gradients actually reach the encoder ------------------------------
# The upstream forward wraps normalization in no_grad(); if that ever widened to
# cover the encoder, the CNN would silently never train and the aux losses would
# look like a plateau rather than an error.
print("\n6. gradient flow")
model.zero_grad()
loss = res["mus"].square().mean() + sum(v.square().mean() for v in aux.values())
loss.backward()
first_conv = net.feature_extractor.cnn[0]
gn = first_conv.weight.grad
check("first conv layer received gradient",
      gn is not None and float(gn.abs().sum()) > 0,
      f"|grad| = {float(gn.abs().sum()):.4e}" if gn is not None else "grad is None")
head_grads = {
    k: float(net.aux_networks[k][0].weight.grad.abs().sum()) for k in want
}
check("every aux head received gradient", all(v > 0 for v in head_grads.values()),
      f"{ {k: f'{v:.3e}' for k, v in head_grads.items()} }")
check("image normalizer has no trainable params",
      not any(p.requires_grad for p in net.img_running_mean_std.parameters()),
      f"{sum(p.numel() for p in net.img_running_mean_std.parameters())} params")

# --- 7. BPTT shape handling ------------------------------------------------
print("\n7. BPTT (seq_length > 1)")
T = int(agent_cfg["params"]["config"]["seq_length"])
batch_t = dict(batch)
batch_t["obs"] = torch.randn(N * T, PROPRIO_DIM, device=dev)
batch_t["img"] = torch.rand(
    N * T, int(si["channels"]), int(si["height"]), int(si["width"]), device=dev
)
batch_t["prev_actions"] = torch.zeros(N * T, ACTIONS, device=dev)
batch_t["seq_length"] = T
batch_t["dones"] = torch.zeros(N * T, device=dev)
batch_t["rnn_states"] = [s.to(dev) for s in net.get_default_rnn_state()]
res_t = model(batch_t)
check(f"seq_length={T} gives mus (N*T, 29)", tuple(res_t["mus"].shape) == (N * T, ACTIONS),
      f"got {tuple(res_t['mus'].shape)}")
check(f"seq_length={T} aux is (N*T, size)",
      all(tuple(v.shape) == (N * T, want[k]) for k, v in net.get_aux_outputs().items()))

# --- 7b. BPTT batch layout is env-major, verified not assumed --------------
# The reshape (N*T, F) -> (num_seqs, seq_length, F) requires all T timesteps of
# env 0 first, then env 1, and so on. A time-major batch -- all envs at t=0,
# then all envs at t=1, which is how a rollout naturally accumulates -- is
# silently reinterpreted, mixing envs into fake sequences with no error.
#
# Test: give each env an input that is CONSTANT across its T timesteps under the
# env-major layout. Then a single seq_length=T call must equal T sequential
# seq_length=1 calls that thread the hidden state. eval() so the image
# normalizer's running stats stay frozen between the two runs.
print("\n7b. BPTT batch layout (env-major)")
model.eval()
with torch.no_grad():
    per_env_obs = torch.randn(N, PROPRIO_DIM, device=dev)
    per_env_img = torch.rand(
        N, int(si["channels"]), int(si["height"]), int(si["width"])
    ).to(dev)

    # env-major: row (e*T + t) is env e -> repeat_interleave along the env axis
    obs_major = per_env_obs.repeat_interleave(T, dim=0)
    img_major = per_env_img.repeat_interleave(T, dim=0)
    res_bptt = model({
        "is_train": True,
        "prev_actions": torch.zeros(N * T, ACTIONS, device=dev),
        "obs": obs_major, "img": img_major,
        "rnn_states": [s.to(dev) for s in net.get_default_rnn_state()],
        "seq_length": T, "dones": torch.zeros(N * T, device=dev), "rnn_masks": None,
    })
    mus_bptt = res_bptt["mus"].reshape(N, T, ACTIONS)

    st = [s.to(dev) for s in net.get_default_rnn_state()]
    mus_seq = []
    for _t in range(T):
        r = model({
            "is_train": True,
            "prev_actions": torch.zeros(N, ACTIONS, device=dev),
            "obs": per_env_obs, "img": per_env_img,
            "rnn_states": st, "seq_length": 1, "rnn_masks": None,
        })
        st = list(r["rnn_states"])
        mus_seq.append(r["mus"])
    mus_seq = torch.stack(mus_seq, dim=1)  # (N, T, A)

diff = float((mus_bptt - mus_seq).abs().max())
check("seq_length=T equals T threaded single steps (env-major layout)",
      diff < 2e-4, f"max diff = {diff:.3e}")
# Sanity: the test would actually catch a wrong layout. Under a time-major
# batch the same inputs must NOT reproduce the sequential result.
with torch.no_grad():
    obs_time = per_env_obs.repeat(T, 1)
    img_time = per_env_img.repeat(T, 1, 1, 1)
    res_tm = model({
        "is_train": True,
        "prev_actions": torch.zeros(N * T, ACTIONS, device=dev),
        "obs": obs_time, "img": img_time,
        "rnn_states": [s.to(dev) for s in net.get_default_rnn_state()],
        "seq_length": T, "dones": torch.zeros(N * T, device=dev), "rnn_masks": None,
    })
tm_diff = float((res_tm["mus"].reshape(N, T, ACTIONS) - mus_seq).abs().max())
check("a time-major batch is genuinely different (test has teeth)",
      tm_diff > 1e-3, f"max diff = {tm_diff:.3e}")
model.train()

# --- 8. rejections --------------------------------------------------------
print("\n8. misconfiguration is rejected loudly")
import copy  # noqa: E402


def build_raises(mutate) -> tuple[bool, str]:
    p = copy.deepcopy(agent_cfg["params"]["network"])
    mutate(p)
    b = A2CAuxCNNBuilder()
    b.load(p)
    try:
        b.build("x", actions_num=ACTIONS, input_shape=(PROPRIO_DIM,), num_seqs=N, value_size=1)
        return False, "no exception"
    except Exception as exc:  # noqa: BLE001
        return True, f"{type(exc).__name__}"


for label, mut in [
    ("separate: True", lambda p: p.__setitem__("separate", True)),
    ("missing student_image", lambda p: p.pop("student_image")),
    ("rnn.before_mlp: False", lambda p: p["rnn"].__setitem__("before_mlp", False)),
    ("missing rnn block", lambda p: p.pop("rnn")),
    ("rnn.concat_input: True", lambda p: p["rnn"].__setitem__("concat_input", True)),
    ("bad spatial_pool",
     lambda p: p["student_image"].__setitem__("spatial_pool", "maxpool")),
]:
    ok, why = build_raises(mut)
    check(f"rejects {label}", ok, why)

nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n   trainable parameters: {nparams:,}")

print("\n" + "=" * 62)
passed = not FAILURES
print("PHASE 4 CHECK PASSED" if passed else
      f"PHASE 4 CHECK FAILED — {len(FAILURES)}: {FAILURES}")
print("=" * 62)
print(f"RESULT_GATE {'PASS' if passed else 'FAIL'}")
sys.exit(0 if passed else 1)
