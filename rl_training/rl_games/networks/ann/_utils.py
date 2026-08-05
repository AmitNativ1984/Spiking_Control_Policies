"""Shared helpers for ANN actor/critic modules."""

import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp_trunk(in_features: int, hidden_dims: list, activation: nn.Module) -> nn.Sequential:
    """Stack Linear+activation layers, returning the trunk and leaving the caller to add a head."""
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(activation)
        in_features = hidden_dim
    return nn.Sequential(*layers)


def xavier_init_linear(module: nn.Module) -> None:
    """Xavier-uniform every nn.Linear in `module`, zeroing biases."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("linear"))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
