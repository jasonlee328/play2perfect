"""rl_games-compatible custom networks for DAgger depth distillation.

Importing this module registers the `depth_cnn_lstm` network with rl_games'
global `model_builder.NETWORK_REGISTRY`. Import once per process before
`Runner.load(...)`.
"""

from rl_games.algos_torch import model_builder

from .depth_cnn_lstm import DepthCNNLSTMBuilder

model_builder.register_network("depth_cnn_lstm", DepthCNNLSTMBuilder)
