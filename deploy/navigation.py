"""The trained navigation policy: an rl_games checkpoint plus the DepthVAE, behind BasePolicy.

This is the torch reference for the navigation deployment. It exists so three things can be
checked against ONE definition rather than against each other's transcriptions:

  * export_onnx / export_vae_onnx export the modules this class holds;
  * record_golden records what this class produces;
  * tests/test_deploy_nav_obs_parity.py compares its observation against the live task.

WHY IT WRAPS DepthVAEImageEncoder INSTEAD OF REIMPLEMENTING THE PREPROCESSING
----------------------------------------------------------------------------
The flight side reimplements it (control_policy_api/depth.py, in numpy) because it must --
there is no torch there. This side must NOT, because a second transcription in the
environment that owns the original is a way to make both sides agree with each other and
neither agree with training.

The one adaptation is the units. DepthVAEImageEncoder is fed the SIMULATOR's normalized
buffer (depth_m / sensor_max_range) and multiplies by `sensor_max_range` to recover metres.
This class is fed METRES, because that is what a camera driver hands over and therefore what
the flight side's DroneState.depth is defined to carry. Setting sensor_max_range = 1.0 makes
that multiply the identity, so the remaining pipeline -- resize, clamp, invert, encode -- is
byte-for-byte the training one.

NO aerial_gym IMPORT HERE, DELIBERATELY. Reaching the task config would pull in isaacgym,
which must be imported before torch and would put a GPU-shaped dependency in front of what
is otherwise pure checkpoint arithmetic. The encoder's geometry and clamp window instead
come from control_policy_api.depth -- the numbers the DEPLOYMENT will use -- and
tests/test_deploy_nav_obs_parity.py asserts those equal the task's vae_config. Pinning them
by test rather than by import is what keeps this module importable anywhere.
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch

from control_policy_api.base import DroneState
from control_policy_api.depth import MAX_DEPTH_M, MIN_DEPTH_M, TARGET_HEIGHT, TARGET_WIDTH
from control_policy_api.observations_nav import (
    DISTANCE_NORM_M,
    NAV_LATENT_DIM,
    NAV_OBS_DIM,
    build_nav_observation,
)

from .checkpoint import MlpActorPolicy

__all__ = [
    "NAV_OBS_DIM",
    "NavigationPolicy",
    "build_nav_observation",
    "latent_dim_of",
    "nav_task_config",
]

#: The task config, relative to the repo root (deploy/ sits directly under it).
_TASK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config/task_config/f450_attitude_navigation_task_config.py"
)


def nav_task_config():
    """The navigation task config, loaded WITHOUT importing the aerial_gym package.

    `import config.task_config...` would execute config/__init__.py, which registers the
    env, robot and task with aerial_gym and therefore drags in isaacgym -- which must be
    imported before torch and wants a GPU. None of that is needed to read four numbers and
    a checkpoint path out of a file whose only imports are `math` and `torch`.

    So it is loaded by path instead. The narrowness is the safety: if this file ever grows
    an aerial_gym import, this raises here rather than silently working through a side
    effect somewhere else.

    Used only to DEFAULT command-line arguments. Nothing in the policy's arithmetic reads
    it -- see the module docstring on why the deployment constants live in
    control_policy_api.depth and are pinned to this config by test instead.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_f450_nav_task_config", _TASK_CONFIG_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.task_config


def latent_dim_of(vae_checkpoint: Union[str, Path]) -> int:
    """Read the latent width out of the VAE checkpoint rather than being told it.

    The encoder's last layer emits `2 * latent_dim` (mu and logvar concatenated,
    vae_depth/model.py:54), so the checkpoint knows this and a caller cannot get it wrong.
    Worth the six lines: a latent width that disagrees with the policy's input width fails
    loudly here instead of producing a correctly-shaped, wrongly-sliced observation.
    """
    state = torch.load(str(vae_checkpoint), map_location="cpu")["model_state_dict"]
    for key in ("encoder.fc.2.weight", "encoder.fc.3.weight"):
        if key in state:
            return state[key].shape[0] // 2
    candidates = [
        value.shape[0] // 2
        for key, value in state.items()
        if key.startswith("encoder.fc.") and key.endswith(".weight") and value.ndim == 2
    ]
    if not candidates:
        raise ValueError(f"{vae_checkpoint}: no encoder FC weights; not a DepthVAE checkpoint")
    return candidates[-1]


class _MetricDepthVaeConfig:
    """vae_config with the simulator's normalization factored out. See the module docstring."""

    def __init__(self, model_file, latent_dims):
        self.model_file = str(model_file)
        self.latent_dims = int(latent_dims)
        self.target_height = TARGET_HEIGHT
        self.target_width = TARGET_WIDTH
        self.max_depth_m = MAX_DEPTH_M
        self.min_depth_m = MIN_DEPTH_M
        # The identity, so `encode` consumes metres directly.
        self.sensor_max_range = 1.0


class NavigationPolicy(MlpActorPolicy):
    """The dense navigation actor, with the DepthVAE encoder in front of it.

    `state.depth` must be a (H, W) array of METRIC depth. Unlike the hover policy this one
    cannot produce an observation without an image: the vision channels have no defined
    value before the first frame, and zeros are not a neutral substitute (the VAE gate that
    once trained the policy against zeroed latents was removed).
    """

    def __init__(
        self,
        checkpoint_path: str,
        vae_checkpoint: Union[str, Path] = None,
        device: str = "cpu",
        activation: str = "elu",
        distance_norm_m: float = DISTANCE_NORM_M,
    ):
        self.distance_norm_m = float(distance_norm_m)
        self._prev_action_transformed = np.zeros(4, dtype=np.float32)
        self._latent = np.zeros(NAV_LATENT_DIM, dtype=np.float32)
        self._have_latent = False

        super().__init__(
            checkpoint_path=checkpoint_path,
            obs_dim=NAV_OBS_DIM,
            action_dim=4,
            activation=activation,
            device=device,
        )

        # Optional, so export_onnx can export the ACTOR without loading a VAE it never
        # touches. Anything that needs an observation needs the encoder and says so below.
        if vae_checkpoint is None:
            self.vae_checkpoint = None
            self.encoder = None
            return

        from vae_depth.vae_image_encoder import DepthVAEImageEncoder

        latent_dims = latent_dim_of(vae_checkpoint)
        if latent_dims != NAV_LATENT_DIM:
            raise ValueError(
                f"{vae_checkpoint} has latent_dim={latent_dims}, but the observation "
                f"reserves {NAV_LATENT_DIM} channels: wrong VAE for this policy"
            )
        self.vae_checkpoint = Path(vae_checkpoint)
        self.vae_config = _MetricDepthVaeConfig(vae_checkpoint, latent_dims)
        self.encoder = DepthVAEImageEncoder(config=self.vae_config, device=device)

    # -- the DepthVAE -----------------------------------------------------------------

    def encode_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """(H, W) metric depth -> (32,) latent, through the training encoder."""
        if self.encoder is None:
            raise RuntimeError(
                "this NavigationPolicy was built without a VAE checkpoint; it can export "
                "or inspect the actor but cannot build an observation"
            )
        tensor = torch.as_tensor(
            np.asarray(depth_m, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        return self.encoder.encode(tensor)[0].cpu().numpy().astype(np.float32)

    @property
    def latent(self) -> np.ndarray:
        return self._latent.copy()

    @property
    def prev_action_transformed(self) -> np.ndarray:
        return self._prev_action_transformed.copy()

    # -- BasePolicy -------------------------------------------------------------------

    def build_observation(self, state: DroneState, target: np.ndarray) -> np.ndarray:
        if state.depth is not None:
            self._latent = self.encode_depth(state.depth)
            self._have_latent = True
        elif not self._have_latent:
            raise RuntimeError(
                "no depth frame has been supplied yet; the navigation observation has no "
                "defined value before the first image"
            )
        return build_nav_observation(
            state,
            target,
            self._prev_action_transformed,
            self._latent,
            distance_norm_m=self.distance_norm_m,
        )

    def act(self, state: DroneState, target: np.ndarray) -> np.ndarray:
        """As BasePolicy.act, plus the transformed-action latch the observation needs.

        The extra latch is not bookkeeping: observation channels [13:17] carry the
        TRANSFORMED command (radians, rad/s), not the raw [-1, 1] output that BasePolicy
        latches. Derived from the clamped action in that order, matching the task's own
        clamp-then-scale in action_transformation_function.
        """
        action = super().act(state, target)
        command = self.to_attitude_command(action)
        self._prev_action_transformed = np.array(
            [command.thrust, command.roll, command.pitch, command.yaw_rate],
            dtype=np.float32,
        )
        return action

    def reset(self) -> None:
        super().reset()
        self._prev_action_transformed = np.zeros(4, dtype=np.float32)
