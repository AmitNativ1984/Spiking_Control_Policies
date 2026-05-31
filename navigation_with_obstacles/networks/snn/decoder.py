from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple

class SpikeDecoder(nn.Module):
    """ Spike decoder module for PopSAN.
    This module takes the latent spike activity from the last hidden layer of the actor SNN and decodes it into the parameters of the action distribution (e.g. mean and std for Gaussian policy).
    """

    def __init__(self, input_dim: int, action_dim: int, pop_dim: int) -> None:
        """Initialize the SpikeDecoder.
        
        Args:
            input_dim (int): Dimension of the input latent spike activity (last hidden layer size).
            action_dim (int): Dimension of the action space (number of actions).
            pop_dim (int): Dimension of the population code.
        """

        super(SpikeDecoder, self).__init__()

        self.action_dim = action_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(in_channels=input_dim, 
                                 out_channels=action_dim, 
                                 kernel_size=pop_dim,
                                 groups=action_dim)  # Depthwise convolution to decode each action dimension from its corresponding population code
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, mean_spikes: torch.Tensor) -> torch.Tensor:
        """Decode the latent spike activity into action distribution parameters.
        
        Args:
            latent_mean_spikes (torch.Tensor): Input latent mean spike activity tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Action distribution parameters tensor of shape [batch_size, action_dim].
        """

        x = mean_spikes.view(-1, self.action_dim, self.pop_dim)  # Reshape to [batch_size, action_dim, pop_dim]
        action_mu = self.decoder(x).view(-1, self.action_dim)  # Decode and reshape to [batch_size, action_dim]
        action_log_std = self.log_std.expand_as(action_mu)  # Expand log std to match action_mu shape
        return action_mu, action_log_std
    