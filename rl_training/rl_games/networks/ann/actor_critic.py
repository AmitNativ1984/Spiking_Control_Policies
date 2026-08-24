"""Feed-forward MLP actor-critic for rl_games."""

from typing import Tuple

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder

from ._utils import xavier_init_linear
from .actor import ANNMLPActor
from .critic import ANNMLPCritic


class MLPActorCriticNetworkBuilder(NetworkBuilder):
    def load(self, params):
        """rl_games calls this with params = the YAML's `network:` block (already unwrapped)."""
        self.config = params

    def build(self, name, **kwargs):
        return ANNMLPActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            **self.config,
        )


class ANNMLPActorCriticNetwork(nn.Module):
    """Separate actor and critic trunks, no shared body.

    Config keys (all optional, both under `network:`):
        actor.hidden_dims / actor.activation
        critic.hidden_dims / critic.activation
    """

    def __init__(self, input_dim, action_dim, **config):
        super().__init__()

        self.actor = ANNMLPActor(
            obs_dim=input_dim,
            action_dim=action_dim,
            actor_config=config.get("actor", {}),
        )
        self.critic = ANNMLPCritic(
            obs_dim=input_dim,
            critic_config=config.get("critic", {}),
        )

        xavier_init_linear(self)

    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network."""
        return False

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Returns (mu, log_std, value, states); states is None for a feed-forward net."""
        state = obs_dict["obs"]
        mu, log_std = self.actor(state)
        value = self.critic(state)
        return mu, log_std, value, None
