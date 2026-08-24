"""Depth-vision student network: CNN encoder + LSTM + MLP + auxiliary heads.

Phase 4 of the distillation port, from
``~/DEXTRAH/dextrah_lab/distillation/a2c_with_aux_cnn.py``. Registered as
``a2c_aux_cnn_net`` and built through rl_games' ``ModelBuilder``, so the DAgger
loop gets a normal ``ModelA2CContinuousLogStd`` returning ``mus`` / ``sigmas`` /
``value`` / ``rnn_states``, plus ``get_aux_outputs()`` for the auxiliary
regression heads.

Data flow (single trunk; ``separate: True`` is rejected)::

    img (N,1,90,160) ─> img_running_mean_std ─> CustomCNN ─> 32 features ─┐
    proprio (N,87) ──> model-level running_mean_std ────────────────────> cat (119)
                                                                          │
                            ┌─────────────────────────────────────────────┘
                            v
                     LSTM(512) ─> LayerNorm ─> MLP[512,512,256] ─> mu / sigma / value
                            │                            │
                            └── cat(mlp_out, trunk_in) ──┴─> aux_mlp[512,256] ─> heads

Note there are two independent normalizers: rl_games' model-level
``running_mean_std`` over the 87 proprio inputs (``normalize_input: True``), and
this network's ``img_running_mean_std`` over the (C,H,W) image. DEXTRAH names
the latter ``running_mean_std`` too, which shadows the concept confusingly
without actually colliding; renamed here.

Deviations from the DEXTRAH source, all deliberate
--------------------------------------------------
1. **Image geometry and modality come from config.** The upstream file hardcodes
   ``img_height = 120*2``, ``img_width = 160*2`` and ``use_depth = False`` — so
   despite being the file the port plan calls "the mono depth variant", as
   committed it is configured for 320x240 RGB. Ours reads a ``student_image``
   block, and the check script asserts it against both ``StudentObsCfg``'s
   defaults and the task YAML overlay.

2. **Depth only.** The RGB branch (``obs_dict["rgb"]`` minus its spatial mean)
   and the unused ResNet-normalization transform are dropped, which also drops
   the ``torchvision`` dependency. play2perfect's student is
   ``image_modality="depth"``.

3. **Single trunk only.** The upstream ``separate`` branch duplicates ~90 lines
   of RNN plumbing for a config we never use (``separate: False``), and DAgger
   does not train a value function at all. Rejected loudly instead of carried.

4. **``spatial_pool`` is configurable, and this matters.** Upstream ends the
   conv stack with ``AdaptiveAvgPool2d((1,1))``, collapsing the final feature
   map to one vector per channel. Global average pooling over a
   translation-equivariant conv stack is approximately translation *invariant* —
   it encodes *what* is in frame while discarding *where*. Our task is hole
   localization to 2 mm from a ~10x10 pixel target, i.e. almost purely a "where"
   problem. At 90x160 the final map is 128x3x8, so avgpool discards 2944 of
   3072 values.

   The default stays ``"avgpool"`` to match DEXTRAH's proven configuration, per
   the plan's "do not tune prematurely". But ``spatial_pool: flatten`` keeps the
   3x8 map (128*3*8 = 3072 -> 32), and the ``hole_pos`` aux head is the
   instrument that tells you which you need: if its error stalls well above
   2 mm while the loss otherwise converges, this is the first thing to change.

5. Uses rl_games' ``NetworkBuilder`` rather than DEXTRAH's vendored copy of it,
   which was 160 lines of duplicated upstream code free to drift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd

CNN_OUT_FEATURES = 32

# (out_channels, kernel, stride) per conv layer — DEXTRAH's stack, unchanged.
CONV_SPEC = ((16, 6, 2), (32, 4, 2), (64, 4, 2), (128, 4, 2))


def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Spatial size after one conv, for building the per-layer LayerNorm shapes."""
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (pad, pad) if isinstance(pad, int) else pad
    h = (h_w[0] + 2 * ph - dilation * (kh - 1) - 1) // sh + 1
    w = (h_w[1] + 2 * pw - dilation * (kw - 1) - 1) // sw + 1
    return h, w


class CustomCNN(nn.Module):
    """4-layer conv encoder -> ``CNN_OUT_FEATURES``.

    LayerNorm is per-position (shape ``[C, H, W]``), so it carries learnable
    parameters tied to spatial location. That is the only thing breaking
    translation equivariance when ``spatial_pool="avgpool"`` — see deviation 4
    in the module docstring.
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        in_channels: int = 1,
        out_features: int = CNN_OUT_FEATURES,
        spatial_pool: str = "avgpool",
    ) -> None:
        super().__init__()
        if spatial_pool not in ("avgpool", "flatten"):
            raise ValueError(
                f"spatial_pool must be 'avgpool' or 'flatten', got {spatial_pool!r}"
            )
        self.spatial_pool = spatial_pool
        self.in_channels = int(in_channels)
        self.input_height = int(input_height)
        self.input_width = int(input_width)

        layers: list[nn.Module] = []
        h, w = self.input_height, self.input_width
        c_in = self.in_channels
        for c_out, k, s in CONV_SPEC:
            h, w = conv_output_size((h, w), kernel_size=k, stride=s)
            if h < 1 or w < 1:
                raise ValueError(
                    f"conv stack collapses to {h}x{w} for input "
                    f"{self.input_height}x{self.input_width}; the image is too small."
                )
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=0),
                nn.ReLU(),
                nn.LayerNorm([c_out, h, w]),
            ]
            c_in = c_out

        self.final_spatial = (h, w)
        self.final_channels = c_in
        if spatial_pool == "avgpool":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            flat_size = c_in
        else:
            flat_size = c_in * h * w
        self.flat_size = flat_size

        self.cnn = nn.Sequential(*layers)
        self.linear = nn.Sequential(nn.Linear(flat_size, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cnn(x)
        return self.linear(out.reshape(out.shape[0], -1))


class A2CAuxCNNBuilder(NetworkBuilder):
    """rl_games network builder for the depth student."""

    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return A2CAuxCNNBuilder.Network(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop("actions_num")
            input_shape = kwargs.pop("input_shape")
            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = kwargs.pop("num_seqs", 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            if self.separate:
                raise ValueError(
                    "a2c_aux_cnn_net supports separate=False only. A separate "
                    "critic trunk doubles the RNN plumbing for a value function "
                    "DAgger never trains."
                )
            if not self.has_rnn:
                raise ValueError(
                    "a2c_aux_cnn_net expects an `rnn:` block; the student is "
                    "recurrent (the teacher is, and the depth view is partial)."
                )
            if not self.is_rnn_before_mlp:
                raise ValueError("a2c_aux_cnn_net expects rnn.before_mlp: True.")
            if self.rnn_concat_input:
                # Upstream only honors this in the `before_mlp: False` path,
                # which we reject -- so setting it there does nothing. Silently
                # ignoring a config knob is worse than refusing it.
                raise ValueError(
                    "a2c_aux_cnn_net does not implement rnn.concat_input "
                    "(it is a no-op in the before_mlp: True path)."
                )

            # --- image encoder ------------------------------------------------
            self.img_height = int(self.student_image["height"])
            self.img_width = int(self.student_image["width"])
            self.img_channels = int(self.student_image.get("channels", 1))
            self.normalize_img = bool(self.student_image.get("normalize", True))
            self.feature_extractor = CustomCNN(
                input_height=self.img_height,
                input_width=self.img_width,
                in_channels=self.img_channels,
                out_features=CNN_OUT_FEATURES,
                spatial_pool=self.student_image.get("spatial_pool", "avgpool"),
            )
            # Distinct from rl_games' model-level running_mean_std, which
            # normalizes the proprio vector. This one is per-pixel over the image.
            self.img_running_mean_std = RunningMeanStd(
                (self.img_channels, self.img_height, self.img_width)
            )

            # The trunk sees proprio ++ CNN features.
            trunk_in = int(input_shape[0]) + CNN_OUT_FEATURES
            self.proprio_dim = int(input_shape[0])
            self.trunk_in = trunk_in

            out_size = self.units[-1] if len(self.units) else self.rnn_units

            # --- rnn -> mlp ---------------------------------------------------
            self.rnn = self._build_rnn(self.rnn_name, trunk_in, self.rnn_units, self.rnn_layers)
            if self.rnn_ln:
                self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            self.actor_mlp = self._build_mlp(
                input_size=self.rnn_units,
                units=self.units,
                activation=self.activation,
                norm_func_name=self.normalization,
                dense_func=torch.nn.Linear,
                d2rl=self.is_d2rl,
                norm_only_first_layer=self.norm_only_first_layer,
            )

            # --- auxiliary heads ---------------------------------------------
            # Fed cat(mlp_out, trunk_in): the raw trunk input is re-supplied so
            # the heads can read the CNN features directly instead of only
            # through whatever the LSTM chose to retain.
            if self.is_aux:
                self.aux_mlp = self._build_mlp(
                    input_size=out_size + trunk_in,
                    units=self.aux_units,
                    activation=self.aux_activation,
                    norm_func_name=self.aux_network.get("normalization", None),
                    dense_func=torch.nn.Linear,
                    d2rl=self.aux_is_d2rl,
                    norm_only_first_layer=self.aux_norm_only_first_layer,
                )
                self.aux_networks = nn.ModuleDict()
                for name in self.aux_outputs:
                    self.aux_networks[name] = nn.Sequential(
                        nn.Linear(self.aux_units[-1], int(self.aux_heads[name]["size"])),
                        self.activations_factory.create(self.aux_out_activation),
                    )
            self.last_aux_out: dict[str, torch.Tensor] = {}
            self._warned_no_dones = False

            # --- heads --------------------------------------------------------
            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if not self.is_continuous:
                raise ValueError("a2c_aux_cnn_net supports continuous actions only.")
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            self.sigma_act = self.activations_factory.create(
                self.space_config["sigma_activation"]
            )
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])
            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

        # -- rl_games interface -------------------------------------------------
        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return True

        def get_default_rnn_state(self):
            num_layers = self.rnn_layers
            if self.rnn_name == "lstm":
                return (
                    torch.zeros((num_layers, self.num_seqs, self.rnn_units)),
                    torch.zeros((num_layers, self.num_seqs, self.rnn_units)),
                )
            return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)),)

        def get_aux_outputs(self) -> dict[str, torch.Tensor]:
            """Auxiliary predictions from the most recent forward pass."""
            return self.last_aux_out

        # -- forward ------------------------------------------------------------
        def forward(self, obs_dict):
            """Forward pass.

            **Batch layout contract for ``seq_length > 1``.** Rows are reshaped
            ``(N*T, F) -> (num_seqs, seq_length, F)``, so the caller must lay the
            batch out **env-major**: all ``T`` timesteps of env 0, then all ``T``
            of env 1, and so on. A time-major batch (all envs at t=0, then all
            envs at t=1 -- which is the *natural* way a rollout accumulates)
            silently reinterprets envs as timesteps. Nothing raises; you simply
            train on garbage sequences.

            DEXTRAH never hit this because it overwrote ``seq_length`` with 1
            (``distillation.py:177``). Honoring the config value, as the port
            plan requires, makes this live. ``check_phase4_student_net.py``
            asserts the convention via a seq_length=T vs T-single-steps
            equivalence test.

            **``dones`` should be supplied** whenever ``seq_length > 1``:
            ``LSTMWithDones`` uses it to zero the hidden state at episode
            boundaries *inside* the BPTT window. DEXTRAH's student batch dict
            omits it entirely, which is harmless only at ``seq_length == 1``.
            """
            obs = obs_dict["obs"]
            if "img" not in obs_dict:
                raise KeyError(
                    "a2c_aux_cnn_net requires an 'img' entry in obs_dict; got keys "
                    f"{sorted(obs_dict)}. The DAgger loop must copy env obs['img'] "
                    "into the network batch dict."
                )

            img = obs_dict["img"]
            if img.dim() != 4:
                raise ValueError(f"expected img (N, C, H, W), got {tuple(img.shape)}")
            # Normalization statistics are not learned, hence no_grad — but the
            # encoder itself is outside it, so gradients do reach the conv stack.
            if self.normalize_img:
                with torch.no_grad():
                    img = self.img_running_mean_std(img)
            img_features = self.feature_extractor(img)
            trunk_in = torch.cat([obs, img_features], dim=-1)

            states = obs_dict.get("rnn_states", None)
            if states is None:
                raise KeyError(
                    "a2c_aux_cnn_net requires 'rnn_states' in obs_dict; the "
                    "student is recurrent and its hidden state is caller-owned."
                )
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)
            seq_length = obs_dict.get("seq_length", 1)
            if seq_length > 1 and dones is None and not self._warned_no_dones:
                self._warned_no_dones = True
                print(
                    "[a2c_aux_cnn_net] WARNING: seq_length="
                    f"{seq_length} with no 'dones' in obs_dict. The LSTM hidden "
                    "state will carry across episode boundaries inside the BPTT "
                    "window. Pass dones (N*T,) unless that is intended."
                )

            out = trunk_in
            batch_size = out.size()[0]
            num_seqs = batch_size // seq_length
            out = out.reshape(num_seqs, seq_length, -1)

            if len(states) == 1:
                states = states[0]

            out = out.transpose(0, 1)
            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
            out, states = self.rnn(out, states, dones, bptt_len)
            out = out.transpose(0, 1)
            out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)
            if self.rnn_ln:
                out = self.layer_norm(out)

            if not isinstance(states, tuple):
                states = (states,)

            out = self.actor_mlp(out)

            if self.is_aux:
                self.last_aux_out = {}
                aux_hidden = self.aux_mlp(torch.cat([out, trunk_in], dim=-1))
                for name in self.aux_outputs:
                    self.last_aux_out[name] = self.aux_networks[name](aux_hidden)

            value = self.value_act(self.value(out))
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
            return mu, sigma, value, states

        # -- config -------------------------------------------------------------
        def load(self, params):
            self.separate = params.get("separate", False)
            self.units = params["mlp"]["units"]
            self.activation = params["mlp"]["activation"]
            self.initializer = params["mlp"]["initializer"]
            self.is_d2rl = params["mlp"].get("d2rl", False)
            self.norm_only_first_layer = params["mlp"].get("norm_only_first_layer", False)
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_rnn = "rnn" in params
            self.has_space = "space" in params
            self.central_value = params.get("central_value", False)

            self.student_image = params.get("student_image", {})
            if not self.student_image:
                raise ValueError(
                    "a2c_aux_cnn_net requires a `student_image:` block "
                    "(height/width/channels). DEXTRAH hardcoded 320x240 RGB here; "
                    "leaving it implicit is how you silently train on the wrong "
                    "geometry."
                )

            self.is_aux = "aux_outputs" in params
            if self.is_aux:
                self.aux_network = params["aux_network"]
                self.aux_heads = params["aux_outputs"]
                self.aux_outputs = list(params["aux_outputs"].keys())
                self.aux_units = self.aux_network["mlp"]["units"]
                self.aux_activation = self.aux_network["mlp"]["activation"]
                self.aux_out_activation = self.aux_network["mlp"]["out_activation"]
                self.aux_is_d2rl = self.aux_network["mlp"].get("d2rl", False)
                self.aux_norm_only_first_layer = self.aux_network["mlp"].get(
                    "norm_only_first_layer", False
                )

            if self.has_space:
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                    self.fixed_sigma = self.space_config["fixed_sigma"]
            else:
                self.is_continuous = False

            if self.has_rnn:
                self.rnn_units = params["rnn"]["units"]
                self.rnn_layers = params["rnn"]["layers"]
                self.rnn_name = params["rnn"]["name"]
                self.rnn_ln = params["rnn"].get("layer_norm", False)
                self.is_rnn_before_mlp = params["rnn"].get("before_mlp", False)
                self.rnn_concat_input = params["rnn"].get("concat_input", False)

            self.has_cnn = False


def register_student_networks() -> None:
    """Register ``a2c_aux_cnn_net`` with rl_games' model builder.

    Must run before ``ModelBuilder().load(params)``.
    """
    from rl_games.algos_torch import model_builder

    model_builder.register_network("a2c_aux_cnn_net", A2CAuxCNNBuilder)


__all__ = [
    "A2CAuxCNNBuilder",
    "CustomCNN",
    "CNN_OUT_FEATURES",
    "register_student_networks",
    "conv_output_size",
]
