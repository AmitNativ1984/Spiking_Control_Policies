"""The trained hover policy: an rl_games checkpoint behind the BasePolicy contract.

Needs torch. For a torch-free deployment export the checkpoint with tools/export_onnx.py and
use control_policy_api.onnx.OnnxHoverPolicy instead -- it builds the same observation from
the same module and produces the same actions (tests/test_golden_actions.py checks both).
"""

import numpy as np

from control_policy_api.base import DroneState
from control_policy_api.observations import HOVER_OBS_DIM, build_hover_observation
from .checkpoint import RlGamesPolicy

__all__ = ["HOVER_OBS_DIM", "HoverPolicy", "build_hover_observation"]


class HoverPolicy(RlGamesPolicy):
    """Go to a point and hold it, on IMU-grade state. No camera."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", activation: str = "elu"):
        super().__init__(
            checkpoint_path=checkpoint_path,
            obs_dim=HOVER_OBS_DIM,
            action_dim=4,
            activation=activation,
            device=device,
        )

    def build_observation(self, state: DroneState, target: np.ndarray) -> np.ndarray:
        return build_hover_observation(state, target, self._prev_action)
