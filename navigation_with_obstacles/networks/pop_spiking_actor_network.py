from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple


class PopulationSpikeEncoder(nn.Module):
    """ Population encoding module for PopSAN.
    Each dimension of the input observation is encoded into the activity of a population of neurons.
    Each neuron in the population is modeled as a Gaussian N~(μ, σ) with learnable parameters. After training, 
    Each neuron stimulates a Gaussian-shaped response in the input space, allowing the network to learn a distributed 
    representation of the input features.

    The stimulus each neuron is computed over time steps.
    """

    def __init__(self, obs_dim:int, obs_bounds: list, encoder_config: dict) -> None:
        
        """Initialize the PopulationSpikeEncoder.
        
        Args:
            obs_dim (int): Dimension of the input observation space.
            pop_dim (int): Number of neurons in each population (per input dimension).
            spike_ts (int): Number of time steps to encode the stimulus over.
            obs_range (Tuple[float, float]) shape [obs_dim]: Range for initializing the mean parameters of the Gaussian neurons.
                    column 0 = min, column 1 = max for each input dimension.
        """

        super(PopulationSpikeEncoder, self).__init__()
        self.obs_dim =  obs_dim
        self.pop_dim = encoder_config["pop_dim"]
        self.register_buffer("obs_bounds", torch.tensor(obs_bounds, dtype=torch.float32))  # shape [obs_dim, 2]
        self.encoder_neuron_num = self.obs_dim * self.pop_dim
        # Initialize evenly spaced means across the specified range for each input dimension
        spacing = torch.linspace(0, 1, self.pop_dim).unsqueeze(0)  # shape [1, pop_dim]
        obs_min = self.obs_bounds[:, 0]  # shape [obs_dim]
        obs_max = self.obs_bounds[:, 1]  # shape [obs_dim]
        assert torch.all(obs_min < obs_max), "Invalid obs_range: left column (min) must be less than right column (max)"
        obs_range = obs_max - obs_min
        self.means = nn.Parameter((obs_min.unsqueeze(1) + spacing * obs_range.unsqueeze(1)).unsqueeze(0), requires_grad=True)  # shape [1, obs_dim, pop_dim]

        # Initialize standard deviations to the specified value
        std = obs_range / (2 * self.pop_dim - 1) # heuristic to cover the input range with overlapping Gaussians
        self.stds = nn.Parameter((torch.ones(self.obs_dim, self.pop_dim) * std.unsqueeze(1)).unsqueeze(0), requires_grad=True)  # shape [1, obs_dim, pop_dim]
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the input observation into population spike activity.
        
        The sitimulation strenght for each neuron is computed as a Gaussian function of the distance between the input observation and the neuron's mean. The stimulation strength is then used to drive the spiking activity of the ENCODER neurons. 
        This is done by repeating the process to introduce temporal dynamics. The temporal stimulus is then passed through a IF (Integrate-and-Fire) neuron model to generate the final spike activity.
        
        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].

        Returns:
            torch.Tensor: Encoded spike activity tensor of shape [batch_size, encoder_neuron_num, spike_ts].
        """

        batch_size = obs.shape[0]

        # Expand obs to shape [batch_size, obs_dim, pop_dim]
        obs_expanded = obs.unsqueeze(2).expand(-1, -1, self.pop_dim)

        # Transform the observation values into the stimulation strength for each neuron in the population
        pop_activity = torch.exp(-0.5 * ((obs_expanded - self.means) / self.stds) ** 2)  # shape [batch_size, obs_dim, pop_dim]
        pop_activity = pop_activity.view(batch_size, self.encoder_neuron_num)  # shape [batch_size, obs_dim * pop_dim

        return pop_activity   

class SpikeDecoder(nn.Module):
    """ Spike decoder module for PopSAN.
    This module takes the latent spike activity from the last hidden layer of the actor SNN and decodes it into the parameters of the action distribution (e.g. mean and std for Gaussian policy).
    """

    def __init__(self, input_dim: int, action_dim: int) -> None:
        """Initialize the SpikeDecoder.
        
        Args:
            input_dim (int): Dimension of the input latent spike activity (last hidden layer size).
            action_dim (int): Dimension of the action space (number of actions).
        """

        super(SpikeDecoder, self).__init__()

        # Action head: converts latent spikes to action distribution parameters (mu and sigma)
        self.action_head = nn.Linear(in_features=input_dim, out_features=action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, latent_mean_spikes: torch.Tensor) -> torch.Tensor:
        """Decode the latent spike activity into action distribution parameters.
        
        Args:
            latent_mean_spikes (torch.Tensor): Input latent mean spike activity tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Action distribution parameters tensor of shape [batch_size, action_dim].
        """

        action_mu = self.action_head(latent_mean_spikes)   # shape [batch_size, action_dim]
        action_log_std = self.log_std.expand_as(action_mu) # shape [batch_size, action_dim]
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
        self.pop_encoder = PopulationSpikeEncoder(obs_dim, obs_bounds, actor_config["encoder"])
     
        input_dim = self.pop_encoder.encoder_neuron_num

        # Select surrogate gradient function
        if actor_config["spike_grad"] == "sigmoid":
            spike_grad = surrogate.sigmoid(slope=25)
        elif actor_config["spike_grad"] == "atan":
            spike_grad = surrogate.atan(alpha=2.0)
        elif actor_config["spike_grad"] == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=25)
        else:
            raise ValueError(f"Unsupported spike_grad: {actor_config['spike_grad']}")

        
        
        self.encoding_neurons = snn.Leaky(beta=1.0,  # no leak => IF neuron
                                        threshold=1.0,
                                       spike_grad=surrogate.straight_through_estimator(),   # Passthrough gradient for the non-differentiable spiking function
                                       reset_mechanism="subtract"
                                       )

        self.actor_fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])
        self.actor_lif1 = snn.Leaky(beta=actor_config["beta"],
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)

        
        self.actor_fc2 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.actor_lif2 = snn.Leaky(beta=actor_config["beta"],
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)
        
        self.actor_fc3 = nn.Linear(in_features=hidden_dims[1], out_features=hidden_dims[2])
        self.actor_lif3 = snn.Leaky(beta=actor_config["beta"],
                                    reset_mechanism=actor_config["reset_mechanism"],
                                    reset_delay=actor_config["reset_delay"],
                                    spike_grad=spike_grad,
                                    learn_beta=True)

        self.action_decoder = SpikeDecoder(input_dim=hidden_dims[2], action_dim=action_dim)
       

    def reset_membranes(self) -> None:
        """Reset the membrane potentials of all spiking layers."""
        
        self.input_mem = self.encoding_neurons.reset_mem()
        self.mem1 = self.actor_lif1.reset_mem()
        self.mem2 = self.actor_lif2.reset_mem()
        self.mem3 = self.actor_lif3.reset_mem()

    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SpikingActorNetwork.
        Args:
            obs (torch.Tensor): Input observation tensor of shape [batch_size, obs_dim].
        Returns:
            torch.Tensor: Action distribution parameters tensor of shape [batch_size, action_dim].
        """

        self.reset_membranes()
        
        spikes_acc = []
        
        for t in range(self.num_steps):
            # Actor network - Encoding layer
            pop_activity = self.pop_encoder(obs)
            in_spks, self.input_mem = self.encoding_neurons(pop_activity, self.input_mem)
            
            # Actor network - Layer 1
            actor_fc1_out = self.actor_fc1(in_spks)
            spk1, self.mem1 = self.actor_lif1(actor_fc1_out, self.mem1)

            # Actor network - Layer 2
            actor_fc2_out = self.actor_fc2(spk1)
            spk2, self.mem2 = self.actor_lif2(actor_fc2_out, self.mem2)

            # Actor network - Layer 3
            actor_fc3_out = self.actor_fc3(spk2)
            spk3, self.mem3 = self.actor_lif3(actor_fc3_out, self.mem3)

            spikes_acc.append(spk3) # Shape [batch_size, hidden_dims[2]]

        actor_mean_spikes = torch.stack(spikes_acc, dim=0).mean(dim=0)  # Average over time steps, shape [batch_size, hidden_dims[2]]

        # Decode the mean spikes into action distribution parameters
        action_mu, action_log_std = self.action_decoder(actor_mean_spikes)  # shape [batch_size, action_dim]

        return action_mu, action_log_std
    
