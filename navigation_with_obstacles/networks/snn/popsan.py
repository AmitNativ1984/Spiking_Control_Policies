from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple
from navigation_with_obstacles.networks.snn.pop_spiking_actor import PopulationSpikingActorNetwork
from navigation_with_obstacles.networks.ann.critic import ANNMLPCritic

class POPSANNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        """Called when config is loaded - extract SNN params from YAML.

        The YAML organizes SNN params under `network.actor` (with the
        population dim nested in `actor.encoder.pop_dim`). POPSANNetwork
        expects a flat config that also has `pop_dim` at the top level, so
        surface it here.
        """
        self.actor_config = dict(params["actor"])
        self.actor_config["pop_dim"] = self.actor_config["encoder"]["pop_dim"]
        self.critic_config = params["critic"]

    def build(self, name, **kwargs):
        """Build and return the actual network """

        return POPSANNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            critic_config=self.critic_config,
            **self.actor_config
        )


class POPSANNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, critic_config, **actor_config):
        """
        Spiking Actor-Critic Network Using LIF Neurons

        Parameters:
        - input_dim (int): Dimension of the input observation space.
        - action_dim (int): Dimension of the action space.
        - snn_config (dict): Configuration parameters for the SNN architecture, with the following keys:
            - hidden_dims (list): Hidden layer dimensions (e.g., [256, 128, 64]).
            - spike_grad (str): Type of surrogate gradient function to use.
            - learn_beta (bool): Whether to learn the membrane potential decay factor.
            - beta (float): Initial value for the membrane potential decay factor.
            - reset_mechanism (str): Type of reset mechanism after spike.
            - reset_delay (int): Delay steps for reset mechanism.
            - threshold (float): Neuron firing threshold.
            - learn_threshold (bool): Whether to learn the firing threshold.

        """

        super(POPSANNetwork, self).__init__()

        self.spiking_actor = PopulationSpikingActorNetwork(input_dim, action_dim, **actor_config)
        self.critic = ANNMLPCritic(obs_dim=input_dim, critic_config=critic_config)
    
    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network."""
        return False

    def forward(self, obs_dict) -> Tuple[torch.tensor, torch.tensor, torch.tensor, None]:
        """
        Forward pass over multiple time steps.

        Parameters:
            - obs_dict (dict) containing the observations

        Returns:
           - mu (tensor)
           - sigma (tensor)
           - value (tensor)
           - states (tensor) = None
        """

        # Actor forward pass
        action_mu, action_log_std = self.spiking_actor(obs_dict)
        
        # Critic forward pass
        value = self.critic(obs_dict["obs"])
        
        states = None
        
        return action_mu, action_log_std, value, states
