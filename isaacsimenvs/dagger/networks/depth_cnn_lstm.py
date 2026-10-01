"""rl_games-compatible CNN+LSTM+Gaussian network for the depth-conditioned student.

Plugs into rl_games' standard `network:` block via
`model_builder.register_network('depth_cnn_lstm', ...)` (done in
`isaacsimenvs/dagger/networks/__init__.py`). The `forward(obs_dict)` contract
matches `A2CBuilder.Network` (continuous, separate=False, RNN=True):
returns ``(mu, sigma, value, states)`` where ``states`` is a tuple of
``(h, c)`` LSTM tensors.

Architecture (matches the v2 plan):
    obs_dict['obs']: (B, image_flat + proprio + 1)  # last channel = block_id (SAPG-side)
                              ↓ split
    image (B, C, H, W) ──── 4-conv CNN (stride 2, GroupNorm, ReLU) + GAP + Linear(c4, 256) ──┐
                                                                                              ├─→ concat (256+64) → LSTM-1024 ─┐
    proprio (B, P+block_id?) ─── 2-layer ELU MLP → 64 ─────────────────────────────────────┘                                  │
                                                                                                                              ↓
                              μ_head (LayerNorm + Linear + ELU + Linear) → action mean
                              σ from yaml-driven branch:
                                  fixed_sigma == 'coef_cond' → Param[num_blocks, action_dim], indexed by obs[:,sigma_id_idx]
                                  fixed_sigma == 'fixed'     → Param[action_dim] broadcast over batch
                              value_head (optional, if symmetric_critic): Linear → 1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class DepthCNNLSTMBuilder(NetworkBuilder):
    """rl_games NetworkBuilder shim — `load(params)` stores yaml params, `build(...)` returns the actual nn.Module."""

    def __init__(self, **kwargs):
        super().__init__()
        self.params: dict | None = None

    def load(self, params: dict) -> None:
        self.params = params

    def build(self, name: str, **kwargs) -> "DepthCNNLSTMBuilder.Network":
        return DepthCNNLSTMBuilder.Network(self.params, **kwargs)

    class Network(NetworkBuilder.BaseNetwork):
        """The actual network. Constructed by ModelA2C via `network_builder.build(...)`.

        rl_games passes ``actions_num`` and ``input_shape`` as kwargs at build time.
        SAPG additionally passes ``coef_ids`` and ``coef_id_idx`` when
        ``fixed_sigma == 'coef_cond'`` (see ``a2c_continuous.py:34``).
        """

        def __init__(self, params: dict, **kwargs) -> None:
            super().__init__()
            self.actions_num = int(kwargs["actions_num"])
            # rl_games passes `num_seqs` from the agent (= num_actors * num_agents).
            # We need it on `self` so `get_default_rnn_state()` can build (h, c)
            # of shape (num_layers, num_seqs, hidden) — see A2CBuilder.Network:195.
            self.num_seqs = int(kwargs.pop("num_seqs", 1))
            input_shape = tuple(kwargs["input_shape"])
            assert len(input_shape) == 1, (
                f"depth_cnn_lstm expects a flat obs vector (image_flat + proprio + maybe block_id); got {input_shape}"
            )
            obs_dim = int(input_shape[0])

            # ---- yaml hyperparams ----
            self.image_hw = tuple(params["image_hw"])
            self.image_channels = int(params.get("image_channels", 1))
            self.proprio_dim = int(params["proprio_dim"])
            self.has_block_id = bool(params.get("has_block_id", True))
            self.symmetric_critic = bool(params.get("symmetric_critic", False))

            self.fixed_sigma = params.get("fixed_sigma", "fixed")  # 'fixed' or 'coef_cond'
            assert self.fixed_sigma in ("fixed", "coef_cond"), (
                f"depth_cnn_lstm only supports fixed_sigma in ('fixed', 'coef_cond'); got {self.fixed_sigma!r}"
            )

            cnn_channels = tuple(params.get("cnn_channels", (32, 64, 128, 128)))
            cnn_feat_dim = int(params.get("cnn_feat_dim", 256))
            proprio_hidden = tuple(params.get("proprio_hidden", (128, 64)))
            self.lstm_hidden = int(params.get("lstm_hidden", 1024))
            self.lstm_layers = int(params.get("lstm_layers", 1))
            head_hidden = int(params.get("head_hidden", 256))
            log_sigma_init = float(params.get("log_sigma_init", -1.0))

            self._image_numel = self.image_channels * self.image_hw[0] * self.image_hw[1]
            self._block_id_numel = 1 if self.has_block_id else 0
            expected_obs_dim = self._image_numel + self.proprio_dim + self._block_id_numel
            assert obs_dim == expected_obs_dim, (
                f"obs_dim={obs_dim} but expected {expected_obs_dim} = image_flat({self._image_numel}) "
                f"+ proprio({self.proprio_dim}) + block_id({self._block_id_numel}). "
                "Check image_hw / proprio_dim / has_block_id in the yaml."
            )

            # ---- CNN encoder (4 conv, stride 2, GroupNorm, ReLU) + GAP ----
            c1, c2, c3, c4 = cnn_channels
            self.cnn = nn.Sequential(
                nn.Conv2d(self.image_channels, c1, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, c1), num_channels=c1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, c2), num_channels=c2),
                nn.ReLU(inplace=True),
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, c3), num_channels=c3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, c4), num_channels=c4),
                nn.ReLU(inplace=True),
            )
            self.cnn_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c4, cnn_feat_dim),
                nn.ReLU(inplace=True),
            )

            # ---- proprio MLP. Block-id is concatenated to proprio (per plan v2) ----
            p_in = self.proprio_dim + self._block_id_numel
            p1, p2 = proprio_hidden
            self.proprio_mlp = nn.Sequential(
                nn.Linear(p_in, p1),
                nn.ELU(inplace=True),
                nn.Linear(p1, p2),
                nn.ELU(inplace=True),
            )
            self._proprio_out_dim = p2

            # ---- LSTM ----
            # `LSTMWithDones` masks (h, c) to zero at every `dones=1` boundary so
            # the BPTT unroll inside `calc_gradients` doesn't stitch hidden state
            # across two different episodes mid-sequence. The teacher trained
            # with the same wrapper (rl_games/common/layers/recurrent.py:74);
            # using a plain nn.LSTM here was a silent asymmetry.
            from rl_games.common.layers.recurrent import LSTMWithDones

            self.lstm = LSTMWithDones(
                input_size=cnn_feat_dim + self._proprio_out_dim,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
            )

            # ---- μ head (Tanh-clipped) ----
            self.mu_head = nn.Sequential(
                nn.LayerNorm(self.lstm_hidden),
                nn.Linear(self.lstm_hidden, head_hidden),
                nn.ELU(inplace=True),
                nn.Linear(head_hidden, self.actions_num),
            )

            # ---- σ branch ----
            if self.fixed_sigma == "coef_cond":
                # SAPG block-keyed σ table. coef_ids / coef_id_idx come from a2c_continuous.py:34.
                if "coef_ids" not in kwargs or "coef_id_idx" not in kwargs:
                    raise ValueError(
                        "fixed_sigma='coef_cond' requires `coef_ids` and `coef_id_idx` kwargs; "
                        "rl_games passes these from A2CAgent when intr_reward_coef_embd is set."
                    )
                self.register_buffer("sigma_ids", kwargs["coef_ids"].float().reshape(-1))
                self.sigma_id_idx = int(kwargs["coef_id_idx"])
                num_blocks = int(self.sigma_ids.numel())
                self.log_sigma = nn.Parameter(
                    torch.full((num_blocks, self.actions_num), log_sigma_init, dtype=torch.float32)
                )
            else:  # 'fixed'
                self.log_sigma = nn.Parameter(
                    torch.full((self.actions_num,), log_sigma_init, dtype=torch.float32)
                )

            # ---- Optional value head on the shared trunk (symmetric critic) ----
            if self.symmetric_critic:
                self.value_head = nn.Sequential(
                    nn.LayerNorm(self.lstm_hidden),
                    nn.Linear(self.lstm_hidden, head_hidden),
                    nn.ELU(inplace=True),
                    nn.Linear(head_hidden, 1),
                )
            else:
                self.value_head = None

        # ----- rl_games hooks -----

        def is_separate_critic(self) -> bool:
            return False

        def is_rnn(self) -> bool:
            return True

        def get_default_rnn_state(self) -> tuple[torch.Tensor, torch.Tensor]:
            num_seqs = self.num_seqs        # set by rl_games before calling this
            shape = (self.lstm_layers, num_seqs, self.lstm_hidden)
            return (torch.zeros(shape), torch.zeros(shape))

        # ----- forward -----

        def _split_obs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Slice flat obs into (image, proprio_aug). proprio_aug includes block_id as last col when has_block_id."""
            B = obs.shape[0]
            image = obs[:, : self._image_numel].view(B, self.image_channels, *self.image_hw)
            tail = obs[:, self._image_numel :]
            return image, tail

        def forward(self, obs_dict):
            obs = obs_dict["obs"]                                   # (seq*B, obs_dim) at train; (B, obs_dim) at inference
            image, proprio_aug = self._split_obs(obs)

            # CNN encoder + GAP + projection.
            cnn_feat = self.cnn_proj(self.cnn(image))               # (seq*B, cnn_feat_dim)
            prop_feat = self.proprio_mlp(proprio_aug)               # (seq*B, p2)
            fused = torch.cat([cnn_feat, prop_feat], dim=-1)        # (seq*B, F)

            # LSTM with the rl_games (seq*B → (seq, num_seqs, F) → LSTM → (seq*B, hidden)) reshape.
            seq_length = int(obs_dict.get("seq_length", 1))
            states = obs_dict.get("rnn_states", None)
            dones = obs_dict.get("dones", None)
            B_total = fused.shape[0]
            num_seqs = B_total // seq_length
            x = fused.view(num_seqs, seq_length, -1).transpose(0, 1).contiguous()  # (seq, num_seqs, F)
            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
            if states is not None and len(states) == 1:
                # rl_games passes state as a 1-tuple in some non-LSTM paths; for LSTM it's (h, c).
                states = states[0]
            out, new_states = self.lstm(x, states, dones)           # out: (seq, num_seqs, H)
            out = out.transpose(0, 1).contiguous().view(B_total, self.lstm_hidden)  # (seq*B, H)

            # μ head. Match the teacher's network exactly: SimToolRealSAPG.yaml
            # sets `mu_activation: None`, so the teacher emits raw μ (no Tanh).
            # If the student tanhs while the teacher doesn't, MSE has a hard
            # floor wherever the teacher's μ falls outside (-1, 1) — even at
            # convergence the student cannot match the label.
            mu = self.mu_head(out)                                  # (seq*B, action_dim)

            # σ branch.
            if self.fixed_sigma == "coef_cond":
                # Match block_id (in obs[:, sigma_id_idx]) against the registered block-id table → row index.
                idxs = (
                    (obs[:, self.sigma_id_idx].reshape(-1, 1) == self.sigma_ids).float().argmax(dim=1)
                )
                logstd = self.log_sigma[idxs]                       # (seq*B, action_dim)
            else:
                logstd = self.log_sigma.expand_as(mu)               # broadcast (action_dim,) → (seq*B, action_dim)

            # Value.
            if self.value_head is not None:
                value = self.value_head(out)                        # (seq*B, 1)
            else:
                # Asymmetric mode — actor's value head is unused, but rl_games still expects a value tensor.
                value = mu.new_zeros(mu.shape[0], 1)

            # `ModelA2CContinuousLogStd` exp()s the second return to recover σ.
            # logstd is already (seq*B, action_dim) in both branches above.
            return mu, logstd, value, (new_states[0], new_states[1])
