"""Recurrent (GRU) MLP actor-critic for rl_games.

The actor always has a GRU. The critic's GRU is optional: set `critic.gru` in the
config to recurrent-critic, omit it to keep the feed-forward critic (original
behavior, used by the existing ppo_gru_*.yaml configs).
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
        critic.gru.hidden_size / critic.gru.num_layers (optional - critic is
            feed-forward if omitted)

    `get_default_rnn_state()` returns (actor_state,) or, with a critic GRU,
    (actor_state, critic_state). rl_games treats this tuple purely positionally
    (building/slicing/zeroing each element independently), so that order is
    load-bearing everywhere the state round-trips through rl_games internals.
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

        critic_gru_config = critic_config.get("gru")
        self.has_critic_gru = critic_gru_config is not None
        if self.has_critic_gru:
            self.critic_gru_hidden_size = critic_gru_config["hidden_size"]
            self.critic_gru_num_layers = critic_gru_config["num_layers"]

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

        # Critic: MLP trunk -> (optional GRU) -> scalar value.
        self.critic_net = build_mlp_trunk(
            input_dim, critic_hidden_dims, get_activation(critic_config.get("activation", "elu"))
        )
        if self.has_critic_gru:
            self.critic_gru = GRUWithDones(
                input_size=critic_hidden_dims[-1],
                hidden_size=self.critic_gru_hidden_size,
                num_layers=self.critic_gru_num_layers,
            )
            value_input_dim = self.critic_gru_hidden_size
        else:
            value_input_dim = critic_hidden_dims[-1]
        self.value_head = nn.Linear(value_input_dim, 1)

        xavier_init_linear(self)

    def is_rnn(self):
        """Required by rl_games - indicates this IS an RNN network."""
        return True

    def get_aux_loss(self):
        """Required by rl_games >= 1.6.5, which calls this on every a2c_network."""
        return None

    def get_default_rnn_state(self):
        # Shape per tensor: (num_layers, num_seqs, hidden_size). rl_games indexes
        # this tuple positionally, so order must match forward()'s (actor, critic).
        states = (torch.zeros(self.gru_num_layers, self.num_seqs, self.gru_hidden_size),)
        if self.has_critic_gru:
            states += (
                torch.zeros(
                    self.critic_gru_num_layers, self.num_seqs, self.critic_gru_hidden_size
                ),
            )
        return states

    @staticmethod
    def _to_seq_major(x: torch.Tensor, seq_length: int, num_seqs: int) -> torch.Tensor:
        """Flat rl_games batch -> GRU's (seq_length, num_seqs, features)."""
        return x.reshape(num_seqs, seq_length, -1).transpose(0, 1)

    @staticmethod
    def _to_batch_major(x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Inverse of _to_seq_major, flattened back to rl_games' batch layout."""
        return x.transpose(0, 1).contiguous().reshape(batch_size, -1)

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """Returns (mu, log_std, value, new_states) where new_states matches the
        shape/order of get_default_rnn_state()."""
        obs = obs_dict["obs"]
        hidden_states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt = obs_dict.get("bptt_len", 0)
        seq_length = obs_dict.get("seq_length", 1)

        if isinstance(hidden_states, (tuple, list)):
            actor_state = hidden_states[0]
            critic_state = hidden_states[1] if len(hidden_states) > 1 else None
        else:
            actor_state, critic_state = hidden_states, None

        batch_size = obs.size(0)
        num_seqs = batch_size // seq_length
        seq_dones = self._to_seq_major(dones, seq_length, num_seqs) if dones is not None else None

        # GRUWithDones wants (seq_length, num_seqs, features); rl_games hands us a flat batch.
        actor_features = self._to_seq_major(self.actor_net(obs), seq_length, num_seqs)
        actor_features, actor_state = self.actor_gru(actor_features, actor_state, seq_dones, bptt)
        actor_features = self._to_batch_major(actor_features, batch_size)

        mu = self.action_head(actor_features)
        log_std = self.action_log_std.unsqueeze(0).expand(mu.shape[0], -1)

        critic_features = self.critic_net(obs)
        if self.has_critic_gru:
            critic_features = self._to_seq_major(critic_features, seq_length, num_seqs)
            critic_features, critic_state = self.critic_gru(
                critic_features, critic_state, seq_dones, bptt
            )
            critic_features = self._to_batch_major(critic_features, batch_size)
            new_states = (actor_state, critic_state)
        else:
            new_states = (actor_state,)

        value = self.value_head(critic_features)

        return mu, log_std, value, new_states
