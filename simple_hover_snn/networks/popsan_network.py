from rl_games.algos_torch.network_builder import NetworkBuilder
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple
from navigation_with_obstacles.networks.snn.encoder import PopulationSpikeEncoder
from navigation_with_obstacles.networks.snn.decoder import SpikeDecoder
from navigation_with_obstacles.networks.snn.popsan_cubalif import PopSANCubaLifNetworkBuilder
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
        self.snn_config = dict(params["actor"])
        self.snn_config["pop_dim"] = self.snn_config["encoder"]["pop_dim"]
        self.critic_config = params["critic"]

    def build(self, name, **kwargs):
        """Build and return the actual network """

        return POPSANNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            critic_config=self.critic_config,
            **self.snn_config
        )


class POPSANNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, critic_config, **snn_config):
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

        # Extract hidden dimensions (support both old "hidden_dim" and new "hidden_dims")
        assert "hidden_dims" in snn_config or "hidden_dim" in snn_config, "SNN config must specify 'hidden_dims'"
        hidden_dims = snn_config["hidden_dims"]
        
        # Select surrogate gradient function
        assert "spike_grad" in snn_config, "SNN config must specify 'spike_grad'"
        if snn_config["spike_grad"] == "sigmoid":
            spike_grad = surrogate.sigmoid(slope=25)
        elif snn_config["spike_grad"] == "atan":
            spike_grad = surrogate.atan(alpha=2.0)
        elif snn_config["spike_grad"] == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=25)
        else:
            raise ValueError(f"Unsupported spike_grad: {snn_config['spike_grad']}")

        assert "num_steps" in snn_config, "SNN config must specify 'num_steps'"
        self.num_steps = snn_config["num_steps"]

        self.threshold = snn_config["threshold"]
        
        self.pop_dim = snn_config["pop_dim"]
        
        self.pop_encoder = PopulationSpikeEncoder(obs_dim=input_dim, 
                                                obs_bounds=[
                                                    (-3.0, 3.0) * input_dim
                                                ],
                                                num_steps=self.num_steps,
                                                encoder_config=snn_config["encoder"])
        
        
        self.action_dim = action_dim
        
        # Defining the Actor architecure
        self.actor_fc1 = nn.Linear(in_features=input_dim * self.pop_dim, out_features=hidden_dims[0])
        self.actor_lif1 = snn.Synaptic(alpha=snn_config["alpha"],
                                       beta=snn_config["beta"],
                                       threshold=snn_config["threshold"],
                                       reset_mechanism=snn_config["reset_mechanism"],
                                       reset_delay=snn_config["reset_delay"],
                                       spike_grad=spike_grad)
        
        self.actor_fc2 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.actor_lif2 = snn.Synaptic(alpha=snn_config["alpha"],
                                       beta=snn_config["beta"],
                                       threshold=snn_config["threshold"],
                                       reset_mechanism=snn_config["reset_mechanism"],
                                       reset_delay=snn_config["reset_delay"],
                                       spike_grad=spike_grad)
        
        self.actor_fc3 = nn.Linear(in_features=hidden_dims[1], out_features=action_dim * self.pop_dim)
        self.actor_lif3 = snn.Synaptic(alpha=snn_config["alpha"],
                                       beta=snn_config["beta"],
                                       threshold=snn_config["threshold"],
                                       reset_mechanism=snn_config["reset_mechanism"],
                                       reset_delay=snn_config["reset_delay"],
                                       spike_grad=spike_grad)

        self.action_decoder = SpikeDecoder(action_dim=self.action_dim, 
                                           pop_dim=snn_config["pop_dim"])

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

        x = obs_dict["obs"]
        value = self.critic(x)

        # Initialize the cubalif layers' synaptic and membrane states:
        actor_syn1, actor_mem1 = self.actor_lif1.reset_mem()
        actor_syn2, actor_mem2 = self.actor_lif2.reset_mem()
        actor_syn3, actor_mem3 = self.actor_lif3.reset_mem()
        
        output_spike_act = torch.zeros(x.size(0), self.actor_fc3.out_features, device=x.device)  # Accumulate spikes for actor output
        spike_train_in = self.pop_encoder(x)

        # === Iterate over timesteps ===
        for t in range(self.num_steps):
            # Actor network - Layer 1
            actor_cur1 = self.actor_fc1(spike_train_in[:, :, t])           
            actor_spk1, actor_syn1, actor_mem1 = self.actor_lif1(actor_cur1, actor_syn1, actor_mem1)
            
            # Actor network - Layer 2
            actor_cur2 = self.actor_fc2(actor_spk1)
            actor_spk2, actor_syn2, actor_mem2 = self.actor_lif2(actor_cur2, actor_syn2, actor_mem2)

            # Spiking output layer
            actor_cur3 = self.actor_fc3(actor_spk2)     # Shape: [batch_size, action_dim * pop_dim]
            actor_spk3, actor_syn3, actor_mem3 = self.actor_lif3(actor_cur3, actor_syn3, actor_mem3)
            
            output_spike_act += actor_spk3  # Accumulate spikes over timesteps
                                            # Shape: [batch_size, action_dim * pop_dim] - this is the mean spike activity we'll decode into actions after the loop
            
            
        output_spike_act /= self.num_steps  # Average over timesteps to get mean spike activity for decoding
        output_spike_act = output_spike_act.view(-1, self.action_dim, self.pop_dim)
        action_mu, action_log_std = self.action_decoder(output_spike_act)
        states = None
        return action_mu, action_log_std, value, states
