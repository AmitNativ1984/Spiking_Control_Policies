"""Loading an rl_games PPO checkpoint outside rl_games.

Two things have to be reproduced exactly, and only one of them is the network.

1. INPUT NORMALIZATION. `normalize_input: True` in the training config
   (rl_training/rl_games/cfg/ppo_hover_local.yaml:81) means rl_games wrapped the model in a
   RunningMeanStd whose statistics are FROZEN at inference and shipped inside the
   checkpoint. Skipping it does not degrade the policy gracefully -- it feeds a network
   trained on unit-variance inputs a vector whose channels range over metres and rad/s, and
   the output is meaningless. This is the single most common way a working policy fails on
   deployment, and it fails silently, because every tensor still has the right shape.

2. THE ACTION IS THE GAUSSIAN MEAN, NOT A SAMPLE. `player.deterministic: True`
   (ppo_hover_local.yaml:91), so `action_log_std` exists only for exploration during
   training and is not read here.

The network itself is deliberately rebuilt from the state_dict rather than by importing
rl_games: the deployment target is a ROS 2 container that has no reason to carry the
training stack. The layer shapes come from the checkpoint, so the only thing this file has
to be told is the activation -- which lives in the YAML, not in the weights.

RlGamesPolicy below is architecture-agnostic: it owns the .pth, the frozen normalizer and
the forward call, and delegates network construction to a subclass. MlpActorPolicy is the
dense case. The spiking case lives in snn_checkpoint.py, where rebuilding by hand would be
a transcription risk rather than a saving.
"""

import re
from typing import List, Tuple

import numpy as np
import torch

from control_policy_api.base import BasePolicy

_ACTIVATIONS = {
    "elu": torch.nn.ELU,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
}

# rl_games' RunningMeanStd uses this epsilon inside the sqrt, and clamps the normalized
# result to +/-5 sigma. Both are part of the trained transform, not tunables.
_NORM_EPS = 1e-5
_NORM_CLAMP = 5.0


class RlGamesPolicy(BasePolicy):
    """An rl_games `a2c_continuous` actor, restored from a .pth checkpoint.

    Owns everything that does not depend on the architecture: reading the checkpoint,
    restoring the frozen input normalizer, and running the forward pass. What the actor IS
    comes from `_build_actor`, which a subclass supplies.

    Still abstract twice over: `_build_actor` is the architecture's business (see
    MlpActorPolicy, snn_checkpoint.PopSANPolicy) and `build_observation` is the training
    task's (see HoverPolicy).
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int,
        action_dim: int = 4,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        super().__init__()

        self.device = torch.device(device)
        # weights_only=False is REQUIRED, and must be explicit.
        #
        # torch 2.6 flipped this default to True. An rl_games checkpoint stores
        # `last_mean_rewards` as a numpy.float32, and weights_only=True rejects numpy
        # scalars unless they are allowlisted -- so on any modern torch the default would
        # raise UnpicklingError and the policy would simply not load. Leaving it implicit
        # means the behaviour depends on which torch the container happens to have.
        #
        # The safety this gives up is protection against a maliciously crafted checkpoint.
        # These are our own training artifacts, and the alternative -- add_safe_globals --
        # is a torch 2.x API that would break the 1.x floor this package supports.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = checkpoint["model"]

        self._actor = self._build_actor(weights).to(self.device).eval()
        self._obs_mean, self._obs_std = self._load_normalizer(weights)

        # A checkpoint whose observation width disagrees with the layout this policy builds
        # is a mismatched pair of files. Caught here rather than at the first inference,
        # where it would surface as an opaque shape error mid-flight.
        checkpoint_obs_dim = self._actor_input_dim()
        if checkpoint_obs_dim != obs_dim:
            raise ValueError(
                f"checkpoint expects obs_dim={checkpoint_obs_dim}, "
                f"policy builds obs_dim={obs_dim}: wrong checkpoint for this policy"
            )

    # -- construction -------------------------------------------------------------------

    def _build_actor(self, weights: dict) -> torch.nn.Module:
        """Rebuild the actor from the state_dict. Must return a tensor -> mu Module."""
        raise NotImplementedError

    def _actor_input_dim(self) -> int:
        """Observation width the rebuilt actor expects, for the mismatch check above."""
        raise NotImplementedError

    @staticmethod
    def _load_normalizer(weights: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Pull the frozen input statistics out of the checkpoint.

        Absence is an error, not a reason to fall back to identity: a policy trained with
        `normalize_input: True` and run without it produces confident nonsense. Better to
        refuse to load than to fly it.
        """
        if "running_mean_std.running_mean" not in weights:
            raise ValueError(
                "checkpoint has no running_mean_std buffers. Either it was trained with "
                "normalize_input: False -- in which case use a policy class that does not "
                "subclass RlGamesPolicy -- or the wrong file was supplied."
            )
        mean = weights["running_mean_std.running_mean"].double().numpy()
        var = weights["running_mean_std.running_var"].double().numpy()
        return mean, np.sqrt(var + _NORM_EPS)

    # -- BasePolicy ---------------------------------------------------------------------

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """(x - mean) / sqrt(var + eps), clipped to +/-5. Exactly rl_games' RunningMeanStd."""
        return np.clip((obs - self._obs_mean) / self._obs_std, -_NORM_CLAMP, _NORM_CLAMP)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            mu = self._actor(tensor.unsqueeze(0)).squeeze(0)
        return mu.cpu().numpy()


class MlpActorPolicy(RlGamesPolicy):
    """The dense case: a plain Linear/activation stack, rebuilt from the weights."""

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int,
        action_dim: int = 4,
        activation: str = "elu",
        device: str = "cpu",
    ) -> None:
        # Set before super().__init__, which calls _build_actor.
        self._activation = activation
        super().__init__(
            checkpoint_path=checkpoint_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

    def _actor_input_dim(self) -> int:
        return self._actor[0].in_features

    def _build_actor(self, weights: dict) -> torch.nn.Sequential:
        """Rebuild trunk + action head from the state_dict, inferring widths from shapes.

        Mirrors ANNMLPActor (rl_training/rl_games/networks/ann/actor.py): Linear+activation
        per hidden layer, then a bare Linear head emitting mu. The trunk keys are
        `a2c_network.actor.trunk.{i}.weight`, where `i` skips the activation layers, so they
        are sorted numerically rather than lexically -- `trunk.10` must not precede
        `trunk.2`.
        """
        if self._activation not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation {self._activation!r}")
        make_activation = _ACTIVATIONS[self._activation]

        trunk_indices: List[int] = sorted(
            int(m.group(1))
            for key in weights
            if (m := re.fullmatch(r"a2c_network\.actor\.trunk\.(\d+)\.weight", key))
        )
        if not trunk_indices:
            raise ValueError(
                "no a2c_network.actor.trunk.* weights in checkpoint: this is not an "
                "ANNMLPActor checkpoint (a recurrent or spiking policy needs its own loader)"
            )

        layers: List[torch.nn.Module] = []
        for index in trunk_indices:
            weight = weights[f"a2c_network.actor.trunk.{index}.weight"]
            bias = weights[f"a2c_network.actor.trunk.{index}.bias"]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            linear.weight.data.copy_(weight)
            linear.bias.data.copy_(bias)
            layers.append(linear)
            layers.append(make_activation())

        head_weight = weights["a2c_network.actor.action_head.weight"]
        head_bias = weights["a2c_network.actor.action_head.bias"]
        head = torch.nn.Linear(head_weight.shape[1], head_weight.shape[0])
        head.weight.data.copy_(head_weight)
        head.bias.data.copy_(head_bias)
        layers.append(head)

        return torch.nn.Sequential(*layers)
