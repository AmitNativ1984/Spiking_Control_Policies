"""The trained hover policies: rl_games checkpoints behind the BasePolicy contract.

Two architectures, one observation vector. HoverPolicy is the dense actor; SnnHoverPolicy
is the PopSAN spiking actor. They differ only in what computes mu -- the 16-D observation,
the frozen input normalization and the [-1, 1] action clamp are identical, which is why the
flight side can swap one for the other by changing a path.

Needs torch. For a torch-free deployment export the checkpoint with deploy/export_onnx.py and
use control_policy_api.onnx.OnnxHoverPolicy instead -- it builds the same observation from
the same module and produces the same actions (tests/test_golden_actions.py checks both).
"""

from pathlib import Path
from typing import Union

import numpy as np

from control_policy_api.base import DroneState
from control_policy_api.observations import HOVER_OBS_DIM, build_hover_observation
from .checkpoint import MlpActorPolicy
from .snn_checkpoint import PopSANPolicy

__all__ = [
    "HOVER_OBS_DIM",
    "HoverPolicy",
    "SnnHoverPolicy",
    "build_hover_observation",
]


class _HoverObservation:
    """Go to a point and hold it, on IMU-grade state. No camera.

    A mixin rather than a method on each policy: the claim that the dense and spiking
    policies consume the SAME vector is the thing that lets one replace the other, and it
    is worth more as one definition than as two that happen to agree today.
    """

    def build_observation(self, state: DroneState, target: np.ndarray) -> np.ndarray:
        return build_hover_observation(state, target, self._prev_action)


class HoverPolicy(_HoverObservation, MlpActorPolicy):
    """The dense hover actor."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", activation: str = "elu"):
        super().__init__(
            checkpoint_path=checkpoint_path,
            obs_dim=HOVER_OBS_DIM,
            action_dim=4,
            activation=activation,
            device=device,
        )


class SnnHoverPolicy(_HoverObservation, PopSANPolicy):
    """The PopSAN spiking hover actor.

    `config_path` is the run's frozen config.yaml, not a file from cfg/ -- see
    snn_checkpoint's docstring for why that distinction is load-bearing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Union[str, Path],
        device: str = "cpu",
    ):
        super().__init__(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            obs_dim=HOVER_OBS_DIM,
            action_dim=4,
            device=device,
        )
