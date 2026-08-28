"""PopSAN: spiking actor + ANN critic, wired up for rl_games."""

from typing import Tuple

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder

from ..ann.critic import ANNMLPCritic
from .pop_spiking_actor import PopulationSpikingActorNetwork


class POPSANNetworkBuilder(NetworkBuilder):
    def load(self, params):
        """Called when the config is loaded - extract SNN params from the YAML.

        The YAML organizes SNN params under `network.actor`, with the population dim
        nested in `actor.encoder.pop_dim`. PopulationSpikingActorNetwork expects a flat
        config that also has `pop_dim` at the top level, so surface it here.
        """
        self.actor_config = dict(params["actor"])
        self.actor_config["pop_dim"] = self.actor_config["encoder"]["pop_dim"]
        self.critic_config = params["critic"]

    def build(self, name, **kwargs):
        return POPSANNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            critic_config=self.critic_config,
            **self.actor_config,
        )


class POPSANNetwork(nn.Module):
    """Spiking actor with a conventional (non-spiking) MLP critic.

    Only the actor is deployed to neuromorphic hardware, so the critic — used purely as
    a training-time baseline — stays an ANN.
    """

    def __init__(self, input_dim, action_dim, critic_config, **actor_config):
        super().__init__()

        self.spiking_actor = PopulationSpikingActorNetwork(
            input_dim, action_dim, **actor_config
        )
        self.critic = ANNMLPCritic(obs_dim=input_dim, critic_config=critic_config)

    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network."""
        return False

    def get_aux_loss(self):
        """Required by rl_games >= 1.6.5, which calls this on every a2c_network."""
        return None

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Returns (mu, log_std, value, states); states is None for a feed-forward net."""
        action_mu, action_log_std = self.spiking_actor(obs_dict)
        value = self.critic(obs_dict["obs"])
        return action_mu, action_log_std, value, None
