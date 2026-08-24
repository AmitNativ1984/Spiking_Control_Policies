import torch
import torch.nn as nn


class SpikeDecoder(nn.Module):
    """Decodes the actor's output population spike rates into action means.

    One grouped 1-D convolution per action dimension collapses that action's population
    of `pop_dim` rates down to a single scalar.
    """

    def __init__(self, action_dim: int, pop_dim: int) -> None:
        """
        Args:
            action_dim: Dimension of the action space.
            pop_dim: Dimension of the population code.
        """
        super().__init__()

        self.action_dim = action_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(
            in_channels=action_dim,
            out_channels=action_dim,
            kernel_size=pop_dim,
            groups=action_dim,  # one independent kernel per action dimension
        )

    def forward(self, mean_spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mean_spikes: Tensor of shape [batch_size, action_dim, pop_dim].

        Returns:
            Action mean tensor of shape [batch_size, action_dim].
        """
        return self.decoder(mean_spikes).squeeze(-1)
