"""MLP Actor network for Gaussian policies."""

from typing import Tuple

import torch
import torch.nn as nn

from ._utils import build_mlp_trunk, get_activation


class ANNMLPActor(nn.Module):
    """MLP trunk + Gaussian action head (mu, learnable log_std)."""

    def __init__(self, obs_dim: int, action_dim: int, actor_config: dict) -> None:
        super().__init__()

        hidden_dims = actor_config.get("hidden_dims", [256, 128, 64])
        activation = get_activation(actor_config.get("activation", "elu"))

        self.trunk = build_mlp_trunk(obs_dim, hidden_dims, activation)

        # Action head: unbounded mu for Gaussian policy.
        # Output order: [thrust, roll, pitch, yaw_rate]
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        mu = self.action_head(features)
        log_std = self.action_log_std.unsqueeze(0).expand(mu.shape[0], -1)
        return mu, log_std
