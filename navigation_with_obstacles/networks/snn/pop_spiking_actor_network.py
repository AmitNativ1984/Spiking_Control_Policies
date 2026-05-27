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
    Each neuron in the population is modeled as a Gaussian N~(μ, σ) with fixed (non-learned)
    parameters at this training stage — means and stds are registered as buffers.
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
        self.register_buffer("means", (obs_min + spacing * obs_range).unsqueeze(0))  # shape [1, obs_dim, pop_dim]
        
        # Initialize stds to cover the input range with overlapping Gaussians.
        # We want to make sure that all the range of the input is covered by Gaussian receptive fields,
        # And inits at least a single spike down the road.      
        means_spacing = torch.abs(self.means[:, :, 1] - self.means[:, :, 0])  # shape [1, obs_dim]
        init_stds = means_spacing * 0.75  # shape [1, obs_dim]
        self.register_buffer("stds", init_stds.unsqueeze(2).expand(-1, -1, self.pop_dim).contiguous())  # shape [1, obs_dim, pop_dim]
        
        self.if1 = snn.Leaky(beta=1.0,  # no leak => IF neuron
                            threshold=self.threshold,
                            spike_grad=surrogate.straight_through_estimator(),   # Passthrough gradient for the non-differentiable spiking function
                            reset_mechanism="subtract"
                            )
        
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

        pop_activity = torch.exp(-0.5 * (obs_expanded - self.means).pow(2) / self.stds.pow(2)).view(batch_size, -1)  # shape [batch_size, obs_dim * pop_dim]
        pop_spikes = torch.zeros(batch_size, self.obs_dim * self.pop_dim, self.num_steps, device=obs.device)  # shape [batch_size, obs_dim * pop_dim, num_steps]
        pop_mem = self.if1.reset_mem()
        for t in range(self.num_steps):
            spikes, pop_mem = self.if1(pop_activity, pop_mem)  # shape [batch_size, obs_dim * pop_dim]
            pop_spikes[:, :, t] = spikes
            
        return pop_spikes

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

class PopulationEncodedSpikingActorNetwork(nn.Module):
    """ Spiking Actor Network for PopSAN.
    This network takes the encoded population spike activity as input and processes it through
    multiple layers of spiking neurons to produce the action distribution parameters.
    """

    def __init__(self, obs_dim: int, action_dim: int, obs_bounds: list, actor_config: dict) -> None:
        """Initialize the SpikingActorNetwork.
        
        Args:
            input_dim (int): Dimension of the input spike activity (encoder_neuron_num).
            action_dim (int): Dimension of the action space (number of actions).
            obs_bounds (list): List of (min, max) tuples for each observation dimension.
            actor_config (dict): Configuration dictionary for the actor network.
        """

        super(PopulationEncodedSpikingActorNetwork, self).__init__()

        assert "hidden_dims" in actor_config, "actor configuration must include 'hidden_dims' key"
        hidden_dims = actor_config["hidden_dims"]
        self.num_steps = actor_config["num_steps"]  # Number of time steps to run the SNN for each input observation
        self.pop_encoder = PopulationSpikeEncoder(obs_dim, obs_bounds, self.num_steps, actor_config["encoder"])
     
        input_dim = self.pop_encoder.encoder_neuron_num
        pop_dim_out = actor_config["encoder"]["pop_dim"] * action_dim

        # Select surrogate gradient function
        if actor_config["spike_grad"] == "sigmoid":
            spike_grad = surrogate.sigmoid(slope=25)
        elif actor_config["spike_grad"] == "atan":
            spike_grad = surrogate.atan(alpha=2.0)
        elif actor_config["spike_grad"] == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=25)
        else:
            raise ValueError(f"Unsupported spike_grad: {actor_config['spike_grad']}")

        # Hidden-layer LIF threshold. Lower threshold (e.g. 0.5) keeps the SNN
        # active at init so gradients can flow to weight matrices; matches the
        # reference PopSAN implementation (vth=0.5).
        # Note: the encoder owns its own IF layer (pop_encoder.if1); no outer encoding IF layer here.
        lif_threshold = actor_config.get("threshold", 0.5)

        self.actor_fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.actor_lif1 = snn.Leaky(beta=actor_config["beta"],
                                    threshold=lif_threshold,
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)


        self.actor_fc2 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.actor_lif2 = snn.Leaky(beta=actor_config["beta"],
                                    threshold=lif_threshold,
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)

        self.actor_fc3 = nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2])
        self.actor_lif3 = snn.Leaky(beta=actor_config["beta"],
                                    threshold=lif_threshold,
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)

        self.actor_fc4 = nn.Linear(in_features=hidden_dims[2], out_features=pop_dim_out)
        self.actor_lif4 = snn.Leaky(beta=actor_config["beta"],
                                    threshold=lif_threshold,
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)
        self.action_decoder = SpikeDecoder(input_dim=action_dim, action_dim=action_dim, pop_dim=actor_config["encoder"]["pop_dim"])
       

    def reset_membranes(self) -> None:
        """Reset the membrane potentials of all hidden spiking layers."""

        self.mem1 = self.actor_lif1.reset_mem()
        self.mem2 = self.actor_lif2.reset_mem()
        self.mem3 = self.actor_lif3.reset_mem()
        self.mem4 = self.actor_lif4.reset_mem()

    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SpikingActorNetwork.
        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].
        Returns:
            torch.Tensor: Action distribution parameters tensor of shape [batch_size, action_dim].
        """

        self.reset_membranes()
        
        spikes_acc = []
        
        in_pop_spikes = self.pop_encoder(obs)
        for t in range(self.num_steps):
            
            # Actor network - Layer 1
            actor_fc1_out = self.actor_fc1(in_pop_spikes[:, :, t])
            spk1, self.mem1 = self.actor_lif1(actor_fc1_out, self.mem1)

            # Actor network - Layer 2
            actor_fc2_out = self.actor_fc2(spk1)
            spk2, self.mem2 = self.actor_lif2(actor_fc2_out, self.mem2)

            # Actor network - Layer 3
            actor_fc3_out = self.actor_fc3(spk2)
            spk3, self.mem3 = self.actor_lif3(actor_fc3_out, self.mem3)

            # Actor network - Layer 4 - Output Population code
            actor_fc4_out = self.actor_fc4(spk3)
            spk4, self.mem4 = self.actor_lif4(actor_fc4_out, self.mem4)  #

            spikes_acc.append(spk4) # Shape [batch_size, hidden_dims[2]]

        actor_mean_spikes = torch.stack(spikes_acc, dim=0).mean(dim=0)  # Average over time steps, shape [batch_size, hidden_dims[2]]

        # Decode the mean spikes into action distribution parameters
        action_mu, action_log_std = self.action_decoder(actor_mean_spikes)  # shape [batch_size, action_dim]

        return action_mu, action_log_std
    
