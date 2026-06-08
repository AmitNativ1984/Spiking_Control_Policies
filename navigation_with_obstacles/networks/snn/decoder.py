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

    def __init__(self, action_dim: int, pop_dim: int) -> None:
        """Initialize the SpikeDecoder.
        
        Args:
            input_dim (int): Dimension of the input latent spike activity (last hidden layer size).
            action_dim (int): Dimension of the action space (number of actions).
            pop_dim (int): Dimension of the population code.
        """

        super(SpikeDecoder, self).__init__()

        self.action_dim = action_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(in_channels=action_dim, 
                                 out_channels=action_dim, 
                                 kernel_size=pop_dim,
                                 groups=action_dim)  # Shape: [batch_size, action_dim, 1] after decoding - one value per action dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, mean_spikes: torch.Tensor) -> torch.Tensor:
        """Decode the latent spike activity into action distribution parameters.
        
        Args:
            mean_spikes(torch.Tensor): Tensor of shape [batch_size, action_dim, pop_dim]
        Returns:
            torch.Tensor: Action distribution parameters tensor of shape [batch_size, action_dim].
        """

        
        action_mu = self.decoder(mean_spikes).squeeze(-1)  # Shape: [batch_size, action_dim]
        action_log_std = self.log_std.expand_as(action_mu)
        return action_mu, action_log_std
    