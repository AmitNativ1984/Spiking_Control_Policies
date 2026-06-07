from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple
from .encoder import PopulationSpikeEncoder
from .decoder import SpikeDecoder

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
        self.spike_decoder = SpikeDecoder(pop_dim_out, action_dim, self.num_steps)

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
    
