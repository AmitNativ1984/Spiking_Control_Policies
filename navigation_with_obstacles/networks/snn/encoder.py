from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple


class PopulationSpikeEncoder(nn.Module):
    """ Population encoding module for PopSAN.
    The input observation is already normalized and bounded [-5, 5] (RL-GAMES normalization).
    POPSAN clamps the normalized observations further to [-3, 3]
    Each dimension of the input observation is encoded into the activity of a population of neurons.
    Each neuron in the population is modeled as a Gaussian N~(μ, σ). The mean μ is initialized to be evenly spaced across the input range [-3, 3] and is learnable. The standard deviation σ is initialized to cover the input space with overlapping Gaussians and is also learnable.
    """

    def __init__(self, obs_dim: int, obs_bounds: list, num_steps: int, encoder_config: dict) -> None:

        """Initialize the PopulationSpikeEncoder.

        Args:
            obs_dim (int): Dimension of the input observation space.
            obs_bounds (list): List of (min, max) tuples for each observation dimension, used for initializing the means and stds of the Gaussian encoding neurons.
            num_steps (int): Number of time steps for the spike simulation. Shared with the outer SNN.
            encoder_config (dict): Encoder configuration (pop_dim, threshold).
        """

        super(PopulationSpikeEncoder, self).__init__()
        self.obs_dim =  obs_dim
        self.pop_dim = encoder_config["pop_dim"]
        self.encoder_neuron_num = self.obs_dim * self.pop_dim
        self.num_steps = num_steps
        self.threshold = encoder_config["threshold"]
        
        # Initialize evenly spaced means across the specified range for each input dimension
        spacing = torch.linspace(0, 1, self.pop_dim).unsqueeze(0)  # shape [1, pop_dim]
        self.register_buffer("obs_bounds", torch.tensor(obs_bounds, dtype=torch.float))  # Register as a buffer to ensure it's moved to the correct device with the model
        obs_min = self.obs_bounds[:, 0].unsqueeze(1)  # shape [obs_dim, 1]
        obs_max = self.obs_bounds[:, 1].unsqueeze(1)  # shape [obs_dim, 1]
        obs_range = obs_max - obs_min   # shape [obs_dim, 1]
        self.means = nn.Parameter((obs_min + spacing * obs_range).unsqueeze(0)) # shape [1, obs_dim, pop_dim]
        
        # Initialize stds to cover the input range with overlapping Gaussians.
        # We want to make sure that all the range of the input is covered by Gaussian receptive fields,
        # And inits at least a single spike down the road.      
        means_spacing = torch.abs(self.means[:, :, 1] - self.means[:, :, 0])  # shape [1, obs_dim]
        init_stds = means_spacing * 0.75  # shape [1, obs_dim]
        init_log_stds = torch.log(init_stds.unsqueeze(2).expand(-1, -1, self.pop_dim).contiguous())  # shape [1, obs_dim, pop_dim]
        
        self.log_stds = nn.Parameter(init_log_stds)  # Learnable log standard deviations, shape [1, obs_dim, pop_dim]
        
        self.if1 = snn.Leaky(beta=1.0,  # no leak => IF neuron
                            threshold=self.threshold,
                            spike_grad=surrogate.straight_through_estimator(),   # Passthrough gradient for the non-differentiable spiking function
                            reset_mechanism="subtract"
                            )

        # Debug-only: when record=True, forward() appends per-step traces to _trace.
        # Set externally (e.g. by tools/runner during --play --plot-encoding). Inert otherwise.
        self.record = False
        self._trace = []
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the input observation into population spike activity.
        
        The sitimulation strenght for each neuron is computed as a Gaussian function of the distance between the input observation and the neuron's mean. The stimulation strength is then used to drive the spiking activity of the ENCODER neurons. 
        This is done by repeating the process to introduce temporal dynamics. The temporal stimulus is then passed through a IF (Integrate-and-Fire) neuron model to generate the final spike activity.
        
        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].

        Returns:
            torch.Tensor: Encoded spike activity tensor of shape [batch_size, encoder_neuron_num, num_steps].
        """

        batch_size = obs.shape[0]

        # Clamp the input observations
        lo = self.obs_bounds[:, 0]
        hi = self.obs_bounds[:, 1]
        obs = torch.clamp(obs, min=lo, max=hi)  # shape [batch_size, obs_dim]

        # Expand obs to shape [batch_size, obs_dim, pop_dim]
        obs_expanded = obs.unsqueeze(2).expand(-1, -1, self.pop_dim)
        stds = torch.exp(self.log_stds)  # shape [1, obs_dim, pop_dim]
        pop_activity = torch.exp(-0.5 * (obs_expanded - self.means).pow(2) / stds.pow(2)).view(batch_size, -1)  # shape [batch_size, obs_dim * pop_dim]
        pop_spikes = torch.zeros(batch_size, self.obs_dim * self.pop_dim, self.num_steps, device=obs.device)  # shape [batch_size, obs_dim * pop_dim, num_steps]
        pop_mem = self.if1.reset_mem()
        for t in range(self.num_steps):
            spikes, pop_mem = self.if1(pop_activity, pop_mem)  # shape [batch_size, obs_dim * pop_dim]
            pop_spikes[:, :, t] = spikes

        if self.record:
            self._trace.append({
                "obs": obs.detach().cpu(),                                                    # [B, obs_dim]
                "pop_activity": pop_activity.detach().cpu().view(batch_size, self.obs_dim, self.pop_dim),
                "pop_spikes": pop_spikes.detach().cpu(),                                      # [B, obs_dim*pop_dim, num_steps]
            })

        return pop_spikes
