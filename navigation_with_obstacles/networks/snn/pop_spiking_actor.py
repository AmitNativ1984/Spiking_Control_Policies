from rl_games.algos_torch.network_builder import NetworkBuilder
import math
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import torch
from typing import Tuple
from .encoder import PopulationSpikeEncoder
from .decoder import SpikeDecoder
from navigation_with_obstacles.config.task_config import task_config

class PopulationSpikingActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, **actor_config):
        """
        Spiking Actor Network Using LIF Neurons

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

        super(PopulationSpikingActorNetwork, self).__init__()

        # Extract hidden dimensions (support both old "hidden_dim" and new "hidden_dims")
        assert "hidden_dims" in actor_config or "hidden_dim" in actor_config, "SNN config must specify 'hidden_dims'"
        hidden_dims = actor_config["hidden_dims"]
        
        # Select surrogate gradient function
        assert "spike_grad" in actor_config, "SNN config must specify 'spike_grad'"
        if actor_config["spike_grad"] == "sigmoid":
            spike_grad = surrogate.sigmoid(slope=25)
        elif actor_config["spike_grad"] == "atan":
            spike_grad = surrogate.atan(alpha=2.0)
        elif actor_config["spike_grad"] == "fast_sigmoid":
            spike_grad = surrogate.fast_sigmoid(slope=25)
        else:
            raise ValueError(f"Unsupported spike_grad: {actor_config['spike_grad']}")

        assert "num_steps" in actor_config, "SNN config must specify 'num_steps'"
        self.num_steps = actor_config["num_steps"]

        self.threshold = actor_config["threshold"]
        
        self.pop_dim = actor_config["pop_dim"]
        
        # Per-dimension (min, max) bounds for the population encoder, built from
        # task_config.observation_layout (one (lo, hi) tuple per obs dim). Must be
        # length input_dim so the encoder builds means/stds of shape
        # [1, obs_dim, pop_dim] — NOT a single shared Gaussian set.
        obs_bounds = task_config.observation_bounds
        assert len(obs_bounds) == input_dim, (
            f"observation_bounds has {len(obs_bounds)} entries but input_dim={input_dim}; "
            "they must match (one (min, max) per observation dimension)."
        )
        self.pop_encoder = PopulationSpikeEncoder(obs_dim=input_dim,
                                                obs_bounds=obs_bounds,
                                                num_steps=self.num_steps,
                                                encoder_config=actor_config["encoder"])
        
        
        self.action_dim = action_dim
        
        # Defining the Actor architecure
        self.actor_fc1 = nn.Linear(in_features=input_dim * self.pop_dim, out_features=hidden_dims[0])
        self.actor_lif1 = snn.Synaptic(alpha=actor_config["alpha"],
                                       beta=actor_config["beta"],
                                       threshold=actor_config["threshold"],
                                       reset_mechanism=actor_config["reset_mechanism"],
                                       reset_delay=actor_config["reset_delay"],
                                       spike_grad=spike_grad)
        
        self.actor_fc2 = nn.Linear(in_features=hidden_dims[0], out_features=hidden_dims[1])
        self.actor_lif2 = snn.Synaptic(alpha=actor_config["alpha"],
                                       beta=actor_config["beta"],
                                       threshold=actor_config["threshold"],
                                       reset_mechanism=actor_config["reset_mechanism"],
                                       reset_delay=actor_config["reset_delay"],
                                       spike_grad=spike_grad)
        
        self.actor_fc3 = nn.Linear(in_features=hidden_dims[1], out_features=action_dim * self.pop_dim)
        self.actor_lif3 = snn.Synaptic(alpha=actor_config["alpha"],
                                       beta=actor_config["beta"],
                                       threshold=actor_config["threshold"],
                                       reset_mechanism=actor_config["reset_mechanism"],
                                       reset_delay=actor_config["reset_delay"],
                                       spike_grad=spike_grad)

        self.action_decoder = SpikeDecoder(action_dim=self.action_dim,
                                           pop_dim=actor_config["pop_dim"])

        # State-independent log std of the action distribution. A learnable network
        # parameter, unrelated to the spiking decoder. Initialized so exp(log_std) ==
        # sigma_init (default 1.0).
        sigma_init = actor_config.get("sigma_init", 1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(sigma_init)))

        # First encoder column belonging to the VAE-latent block. The population encoder
        # lays out obs dim d in columns [d*pop_dim : (d+1)*pop_dim], so the latents (obs
        # dims [state_dims : input_dim]) start at state_dims*pop_dim. Derived from config
        # (not a hardcoded 49) so it adapts to the latent size; for a state-only network
        # (input_dim == state_dims) this equals the encoder width and the gate is inert.
        state_dims = input_dim - (
            task_config.vae_config.latent_dims if task_config.vae_config.use_vae else 0
        )
        self._lat_col_start = state_dims * self.pop_dim

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Back-compat: log_std used to live on the decoder (action_decoder.log_std)
        # and warmup left it at its stale init (std=1.0). Legacy checkpoints have no
        # top-level log_std, so seed it from the freshly built sigma_init value to
        # satisfy strict loading; checkpoints that already trained log_std keep theirs.
        state_dict.pop(prefix + "action_decoder.log_std", None)
        log_std_key = prefix + "log_std"
        if log_std_key not in state_dict:
            state_dict[log_std_key] = self.log_std.data.clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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

        # Initialize the cubalif layers' synaptic and membrane states:
        actor_syn1, actor_mem1 = self.actor_lif1.reset_mem()
        actor_syn2, actor_mem2 = self.actor_lif2.reset_mem()
        actor_syn3, actor_mem3 = self.actor_lif3.reset_mem()
        
        output_spike_act = torch.zeros(x.size(0), self.actor_fc3.out_features, device=x.device)  # Accumulate spikes for actor output
        spike_train_in = self.pop_encoder(x)

        # VAE curriculum gate: scale the encoded VAE-latent spike block by task_config.vae_gate
        # (set by the task's curriculum state machine; 1.0 by default and at inference). At
        # gate=0.0 the latent spiking inputs contribute zero current to actor_fc1, removing
        # the population-encoder dilution while the policy learns pure navigation. Applied
        # AFTER rl_games norm_obs, so the latent RunningMeanStd stats stay warm for un-gating.
        gate = getattr(task_config, "vae_gate", 1.0)
        if gate != 1.0 and self._lat_col_start < spike_train_in.shape[1]:
            spike_train_in[:, self._lat_col_start:, :] = (
                spike_train_in[:, self._lat_col_start:, :] * gate
            )

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
        action_mu = self.action_decoder(output_spike_act)
        action_log_std = self.log_std.expand_as(action_mu)
        return action_mu, action_log_std