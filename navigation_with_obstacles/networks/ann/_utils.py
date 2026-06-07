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
