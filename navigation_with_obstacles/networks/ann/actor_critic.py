"""
Standard MLP Actor-Critic Network.
"""

from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import torch
from typing import Tuple

from .actor import ANNMLPActor
from .critic import ANNMLPCritic


class MLPActorCriticNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        """rl_games calls this with params = the YAML's `network:` block (already unwrapped)."""
        self.config = params

    def build(self, name, **kwargs):
        """Build and return the actual network"""
        return ANNMLPActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            **self.config
        )


class ANNMLPActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, **config):
        """
        Standard MLP Actor-Critic Network

        Parameters:
        - input_dim (int): Dimension of the input observation space.
        - action_dim (int): Dimension of the action space.
        - config (dict): Configuration parameters for the MLP architecture, with
          optional `actor` and `critic` sub-dicts (each accepting `hidden_dims`
          and `activation`).
        """

        super(ANNMLPActorCriticNetwork, self).__init__()

        self.actor = ANNMLPActor(
            obs_dim=input_dim,
            action_dim=action_dim,
            actor_config=config.get("actor", {}),
        )
        self.critic = ANNMLPCritic(
            obs_dim=input_dim,
            critic_config=config.get("critic", {}),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network"""
        return False

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Forward pass through the network

        Parameters:
            - obs_dict (dict) containing the observations

        Returns:
            - mu (tensor): Action means (unbounded, no activation)
            - log_std (tensor): Log standard deviations
            - value (tensor): value estimate
            - states (None): return for recurrent network
        """

        state = obs_dict["obs"]
        mu, log_std = self.actor(state)
        value = self.critic(state)
        return mu, log_std, value, None
