import os
import time
import numpy as np
import random
from copy import deepcopy
import torch

from rl_games.common import object_factory
from rl_games.common import tr_helpers

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.algos_torch import players
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
from rl_games.algos_torch import torch_ext


def _restore(agent, args):
    if 'checkpoint' in args and args['checkpoint'] is not None and args['checkpoint'] !='':
        load_mode = args.get('checkpoint_load_mode', 'resume')
        if load_mode == 'resume':
            agent.restore(args['checkpoint'])
        elif load_mode == 'weights':
            weights = _load_checkpoint_weights(agent, args['checkpoint'])
            if args.get('checkpoint_obs_insert'):
                weights = _expand_new_obs_inputs(weights, agent.model, args['checkpoint_obs_insert'])
            agent.set_weights(weights)
            if getattr(agent, 'has_central_value', False) and 'assymetric_vf_nets' in weights:
                critic = weights['assymetric_vf_nets']
                if args.get('checkpoint_states_insert'):
                    critic = _expand_new_obs_inputs(
                        {'model': critic}, agent.central_value_net, args['checkpoint_states_insert'])['model']
                try:
                    agent.central_value_net.load_state_dict(critic)
                except RuntimeError as exc:
                    print(f"Skipping central value checkpoint weights: {exc}")
            print(f"=> initialized model weights from '{args['checkpoint']}'")
        else:
            raise ValueError(f"checkpoint_load_mode must be resume/weights, got {load_mode!r}")


def _expand_new_obs_inputs(weights, model, obs_insert):
    """Pad a checkpoint whose obs predates fields added at columns [offset, offset + n).

    Every model tensor that is exactly n shorter than the live model's along one dim
    (the first input layer's weight, the running obs mean/var) gets n entries inserted
    at ``offset``: zeros, so the new inputs start out ignored and the warm-started policy
    acts as before, and ones for the running variance. Other mismatches are left for
    ``load_state_dict`` to report.
    """
    offset, n = int(obs_insert[0]), int(obs_insert[1])
    target = model.state_dict()
    state = dict(weights['model'])
    for key, value in state.items():
        want = target.get(key)
        if want is None or value.shape == want.shape:
            continue
        dims = [d for d in range(value.dim()) if value.shape[d] != want.shape[d]]
        if len(dims) != 1 or want.shape[dims[0]] - value.shape[dims[0]] != n:
            continue
        d = dims[0]
        pad_shape = list(value.shape)
        pad_shape[d] = n
        fill = torch.ones if key.endswith('running_var') else torch.zeros
        pad = fill(pad_shape, dtype=value.dtype, device=value.device)
        state[key] = torch.cat(
            [value.narrow(d, 0, offset), pad, value.narrow(d, offset, value.shape[d] - offset)], dim=d)
        print(f"=> {key}: {tuple(value.shape)} -> {tuple(state[key].shape)} (new obs inputs zero-initialized)")
    return {**weights, 'model': state}


def _load_checkpoint_weights(agent, checkpoint_path):
    checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict):
        if getattr(agent, 'global_rank', None) in checkpoint:
            return checkpoint[agent.global_rank]
        if 0 in checkpoint:
            return checkpoint[0]
    return checkpoint

def _override_sigma(agent, args):
    if 'sigma' in args and args['sigma'] is not None:
        net = agent.model.a2c_network
        if hasattr(net, 'sigma') and hasattr(net, 'fixed_sigma'):
            if net.fixed_sigma == 'fixed':
                with torch.no_grad():
                    net.sigma.fill_(float(args['sigma']))
            else:
                print('Print cannot set new sigma because fixed_sigma is False')


class Runner:

    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        #self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))
        #self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if params["config"].get('multi_gpu', False):
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            # set different random seed for each GPU
            self.seed += self.global_rank

            print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size}")

        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

            # deal with environment specific seed if applicable
            if 'env_config' in params['config']:
                if not 'seed' in params['config']['env_config']:
                    params['config']['env_config']['seed'] = self.seed
                else:
                    if params["config"].get('multi_gpu', False):
                        params['config']['env_config']['seed'] += self

        config = params['config']
        config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def set_vec_env(self, vec_env):
        self.params['config']['vec_env'] = vec_env

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)
        return agent.train()

    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)
        player.run()

    def create_player(self):
        return self.player_factory.create(self.algo_name, params=self.params)

    def reset(self):
        pass

    def run(self, args):
        if args['train']:
            return self.run_train(args)
        elif args['play']:
            return self.run_play(args)
        else:
            return self.run_train(args)
