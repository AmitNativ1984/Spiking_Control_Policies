import torch
import torch.nn as nn
from .pop_spiking_cubalif_actor_network import PopulationEncodedCubaLifSpikingActorNetwork
from navigation_with_obstacles.networks.ann.critic import ANNMLPCritic
from rl_games.algos_torch.network_builder import NetworkBuilder
from typing import Tuple
from navigation_with_obstacles.config.task_config import task_config


class PopSANCubaLifNetworkBuilder(NetworkBuilder):
    """rl_games NetworkBuilder for PopSANCubaLifActorCriticNetwork."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        """Called when config is loaded - extract PopSAN params from YAML """
        self.actor_config = params["actor"]
        self.critic_config = params["critic"]
        self.obs_bounds = task_config.observation_bounds    # list of (min, max) tuples for each obs dimension, used for encoder initialization
        

    def build(self, name, **kwargs):
        """Build and return the actual network """

        return PopSANCubaLifActorCriticNetwork(
            obs_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            obs_bounds=self.obs_bounds,
            actor_config=self.actor_config,
            critic_config=self.critic_config
        )



class PopSANCubaLifActorCriticNetwork(nn.Module):
    """Combined Actor-Critic Network for PopSAN.
    
    This class encapsulates both the spiking actor network and the non-spiking critic network.
    It provides a unified interface for forward passes through both networks, allowing for efficient training and 
    inference for the PopSAN algorithm.
    """

    def __init__(self, obs_dim: int, action_dim: int, obs_bounds: list, actor_config: dict, critic_config: dict) -> None:
        """Initialize the PopSANActorCriticNetwork.
        
        Args:
            obs_dim (int): Dimension of the input observation space.
            action_dim (int): Dimension of the action space (number of action parameters).
            obs_bounds (list): List of (min, max) tuples for each observation dimension.
            actor_config (dict): Configuration dictionary for the actor network.
            critic_config (dict): Configuration dictionary for the critic network.
        """

        super(PopSANCubaLifActorCriticNetwork, self).__init__()

        self.snn_actor = PopulationEncodedCubaLifSpikingActorNetwork(obs_dim=obs_dim, 
                                                              action_dim=action_dim, 
                                                              obs_bounds=obs_bounds, 
                                                              actor_config=actor_config)
        
        self.critic = ANNMLPCritic(obs_dim=obs_dim, critic_config=critic_config)

    def is_rnn(self) -> bool:
        """Required by rl_games: indicates whether the network contains recurrent layers (LSTM/GRU)."""
        return False
    
    def forward(self, input_dict: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Forward pass through both the actor and critic networks.

        Args:
            input_dict (dict): rl_games input dict containing 'obs' key with tensor of shape [batch_size, obs_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - action_mu (torch.Tensor): Mean of the action distribution from the actor, shape [batch_size, action_dim].
                - action_log_std (torch.Tensor): Log standard deviation of the action distribution from the actor, shape [batch_size, action_dim].
                - value (torch.Tensor): Value estimate from the critic, shape [batch_size, 1].
        """
        obs = input_dict['obs']
        action_mu, action_log_std = self.snn_actor(obs)
        value = self.critic(obs)
        states = None
        return action_mu, action_log_std, value, states