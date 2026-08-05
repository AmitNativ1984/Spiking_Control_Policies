"""MLP Critic network producing a scalar value estimate."""

import torch
import torch.nn as nn

from ._utils import build_mlp_trunk, get_activation


class ANNMLPCritic(nn.Module):
    """MLP trunk + scalar value head."""

    def __init__(self, obs_dim: int, critic_config: dict) -> None:
        super().__init__()

        hidden_dims = critic_config.get("hidden_dims", [256, 128, 64])
        activation = get_activation(critic_config.get("activation", "elu"))

        self.trunk = build_mlp_trunk(obs_dim, hidden_dims, activation)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.trunk(state)
        return self.value_head(features)
