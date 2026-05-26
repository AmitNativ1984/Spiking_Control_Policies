"""MLP Critic network producing a scalar value estimate."""

import torch
import torch.nn as nn

from ._utils import get_activation


class ANNMLPCritic(nn.Module):
    """MLP trunk + scalar value head."""

    def __init__(self, obs_dim: int, critic_config: dict) -> None:
        super().__init__()

        hidden_dims = critic_config.get("hidden_dims", [256, 128, 64])
        activation = get_activation(critic_config.get("activation", "elu"))

        layers = []
        in_features = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation)
            in_features = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Linear(in_features, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.trunk(state)
        return self.value_head(features)
