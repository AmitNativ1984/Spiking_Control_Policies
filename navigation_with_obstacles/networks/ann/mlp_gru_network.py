"""
Standard MLP Actor-Critic Network.
"""

from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.common.layers.recurrent import GRUWithDones
import torch.nn as nn
import torch
from typing import Tuple


class GRUActorCriticNetworkBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self, params):
        """rl_games calls this with params = the YAML's `network:` block (already unwrapped)."""
        self.config = params

    def build(self, name, **kwargs):
        """Build and return the actual network.

        rl_games passes num_seqs (= num_actors * num_agents) here — forward it
        so the network can size its initial hidden state correctly.
        """
        return GRUActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            num_seqs=kwargs.get("num_seqs", 1),
            **self.config
        )
    
class GRUActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, num_seqs=1, **config):
        """
        Standard MLP Actor-Critic Network
        
        Parameters:
        - input_dim (int): Dimension of the input observation space.
        - action_dim (int): Dimension of the action space.
        - config (dict): Configuration parameters for the MLP architecture:
            - units (list): Hidden layer sizes (default: [256, 128, 64])
            - activation (str): Activation function (default: "elu")        
        """

        super(GRUActorCriticNetwork, self).__init__()

        # Extract config parameters
        actor_config = config.get("actor", {})        
        actor_hidden_dims = actor_config.get("hidden_dims", [256, 128, 64])
        actor_activation_name = actor_config.get("activation", "elu")

        critic_config = config.get("critic", {})
        critic_hidden_dims = critic_config.get("hidden_dims", [256, 128, 64])
        critic_activation_name = critic_config.get("activation", "elu")

        self.gru_hidden_size = actor_config["gru"]["hidden_size"]
        self.gru_num_layers = actor_config["gru"]["num_layers"]
        # rl_games passes num_seqs = num_actors * num_agents into build();
        # it's the batch dim of the GRU hidden state during rollout.
        self.num_seqs = num_seqs

        # Select activation function
        if actor_activation_name == "elu":
            actor_activation = nn.ELU()
        elif actor_activation_name == "relu":
            actor_activation = nn.ReLU()
        elif actor_activation_name == "tanh":
            actor_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {actor_activation_name}")  

        # Build policy network (actor)
        actor_layers = []
        in_features = input_dim
        for hidden_layer_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(in_features, hidden_layer_dim))
            actor_layers.append(actor_activation)
            in_features = hidden_layer_dim
        self.actor_net = nn.Sequential(*actor_layers)
        self.actor_gru = GRUWithDones(
            input_size = in_features, 
            hidden_size = self.gru_hidden_size,
            num_layers = self.gru_num_layers
        )
        
        # Action head: outputs are unbounded (no activation) and represent mu for Gaussian policy
        # Output order: [thrust, roll, pitch, yaw_rate]
        self.action_head = nn.Linear(self.gru_hidden_size, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std for Gaussian policy

        
        # Select activation function for critic
        # Select activation function
        if critic_activation_name == "elu":
            critic_activation = nn.ELU()
        elif critic_activation_name == "relu":
            critic_activation = nn.ReLU()
        elif critic_activation_name == "tanh":
            critic_activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {critic_activation_name}")  
        
        # Build the critic network  
        critic_layers = []
        in_features = input_dim
        for hidden_layer_dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_features, hidden_layer_dim))
            critic_layers.append(critic_activation)
            in_features = hidden_layer_dim
        self.critic_net = nn.Sequential(*critic_layers)

        # Value head: outputs a single scalar value estimate
        self.value_head = nn.Linear(in_features, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def is_rnn(self):
        """Required by rl_games - indicates this is a RNN network"""
        return True

    def get_default_rnn_state(self):
        # Tuple, even with one element - rl_games unpacks it as states[0]
        # Shape: (num_layers, num_seqs, hidden_size). num_seqs is set on this
        # network by rl_games at runtime (= num_actors * num_agents / seq_length).
        return (torch.zeros(self.gru_num_layers, self.num_seqs, self.gru_hidden_size),)
    
    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Forward pass through the network

        Parameters:
            - obs_dict (dict) containing the observations

        Returns:
            - mu (tensor): Action means (unbounded, no activation)
            - log_std (tensor): Log standard deviations
            - value (tensor): value estimate
            - states (tensor): hidden state for recurrent network
        """

        obs             = obs_dict["obs"]
        hidden_states    = obs_dict.get("rnn_states", None)
        dones           = obs_dict.get("dones", None)
        bptt            = obs_dict.get("bptt_len", 0)
        seq_length      = obs_dict.get("seq_length", 1)

        # Actor network
        actor_features = self.actor_net(obs)
        
        B = actor_features.size(0)
        num_seqs = B // seq_length
        actor_features = actor_features.reshape(num_seqs, seq_length, -1).transpose(0, 1) # Reshape to (seq_length, num_seqs, feature_dim) for GRU

        if dones is not None:
            dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)  # Reshape dones to match GRU input

        
        # rl_games passes rnn_states as a tuple matching get_default_rnn_state(); unwrap to a single tensor for GRU.
        recurrent_state = hidden_states[0] if isinstance(hidden_states, (tuple, list)) else hidden_states
        actor_features, recurrent_state = self.actor_gru(actor_features, recurrent_state, dones, bptt)
        actor_features = actor_features.transpose(0, 1).contiguous().reshape(B, -1)  # Reshape back to (B, feature_dim)

        mu = self.action_head(actor_features)
        log_std = self.action_log_std.unsqueeze(0).expand(mu.shape[0], -1)

        # Value network
        critic_features = self.critic_net(obs)
        value = self.value_head(critic_features)

        return mu, log_std, value, (recurrent_state,)