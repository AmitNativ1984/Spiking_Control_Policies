"""Recurrent (GRU) MLP actor-critic for rl_games.

The GRU sits on the actor branch only; the critic stays feed-forward.
"""

from typing import Tuple

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.common.layers.recurrent import GRUWithDones

from ._utils import build_mlp_trunk, get_activation, xavier_init_linear


class GRUActorCriticNetworkBuilder(NetworkBuilder):
    def load(self, params):
        """rl_games calls this with params = the YAML's `network:` block (already unwrapped)."""
        self.config = params

    def build(self, name, **kwargs):
        """rl_games passes num_seqs (= num_actors * num_agents); the network needs it to
        size its initial hidden state."""
        return GRUActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            num_seqs=kwargs.get("num_seqs", 1),
            **self.config,
        )


class GRUActorCriticNetwork(nn.Module):
    """Config keys (under `network:`):

        actor.hidden_dims / actor.activation
        actor.gru.hidden_size / actor.gru.num_layers   (required)
        critic.hidden_dims / critic.activation
    """

    def __init__(self, input_dim, action_dim, num_seqs=1, **config):
        super().__init__()

        actor_config = config.get("actor", {})
        critic_config = config.get("critic", {})

        actor_hidden_dims = actor_config.get("hidden_dims", [256, 128, 64])
        critic_hidden_dims = critic_config.get("hidden_dims", [256, 128, 64])

        self.gru_hidden_size = actor_config["gru"]["hidden_size"]
        self.gru_num_layers = actor_config["gru"]["num_layers"]
        # rl_games passes num_seqs = num_actors * num_agents into build();
        # it's the batch dim of the GRU hidden state during rollout.
        self.num_seqs = num_seqs

        # Actor: MLP trunk -> GRU -> Gaussian head.
        self.actor_net = build_mlp_trunk(
            input_dim, actor_hidden_dims, get_activation(actor_config.get("activation", "elu"))
        )
        self.actor_gru = GRUWithDones(
            input_size=actor_hidden_dims[-1],
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
        )
        # Action head: unbounded mu for Gaussian policy.
        # Output order: [thrust, roll, pitch, yaw_rate]
        self.action_head = nn.Linear(self.gru_hidden_size, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: feed-forward MLP trunk -> scalar value.
        self.critic_net = build_mlp_trunk(
            input_dim, critic_hidden_dims, get_activation(critic_config.get("activation", "elu"))
        )
        self.value_head = nn.Linear(critic_hidden_dims[-1], 1)

        xavier_init_linear(self)

    def is_rnn(self):
        """Required by rl_games - indicates this IS an RNN network."""
        return True

    def get_aux_loss(self):
        """Required by rl_games >= 1.6.5, which calls this on every a2c_network."""
        return None

    def get_default_rnn_state(self):
        # Tuple, even with one element - rl_games unpacks it as states[0].
        # Shape: (num_layers, num_seqs, hidden_size).
        return (torch.zeros(self.gru_num_layers, self.num_seqs, self.gru_hidden_size),)

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Returns (mu, log_std, value, (hidden_state,))."""
        obs = obs_dict["obs"]
        hidden_states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt = obs_dict.get("bptt_len", 0)
        seq_length = obs_dict.get("seq_length", 1)

        actor_features = self.actor_net(obs)

        # GRUWithDones wants (seq_length, num_seqs, features); rl_games hands us a flat batch.
        batch_size = actor_features.size(0)
        num_seqs = batch_size // seq_length
        actor_features = actor_features.reshape(num_seqs, seq_length, -1).transpose(0, 1)
        if dones is not None:
            dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

        # rl_games passes rnn_states as a tuple matching get_default_rnn_state();
        # unwrap to a single tensor for the GRU.
        recurrent_state = (
            hidden_states[0] if isinstance(hidden_states, (tuple, list)) else hidden_states
        )
        actor_features, recurrent_state = self.actor_gru(
            actor_features, recurrent_state, dones, bptt
        )
        actor_features = actor_features.transpose(0, 1).contiguous().reshape(batch_size, -1)

        mu = self.action_head(actor_features)
        log_std = self.action_log_std.unsqueeze(0).expand(mu.shape[0], -1)

        value = self.value_head(self.critic_net(obs))

        return mu, log_std, value, (recurrent_state,)
