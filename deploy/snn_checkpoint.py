"""Loading a PopSAN spiking actor outside rl_games.

Everything RlGamesPolicy says about normalization and the deterministic action applies
here unchanged; this file only supplies the network. Two things differ from the dense case.

THE NETWORK IS IMPORTED, NOT REBUILT. checkpoint.py rebuilds the MLP by hand because it is
three Linears and a shape is a complete description of one. A PopSAN actor is not: it is a
population of Gaussian receptive fields feeding an IF layer feeding three snntorch
`Synaptic` layers with a same-step reset, and re-deriving those semantics in this file
would be exactly the transcription risk the whole deploy/ boundary exists to remove. So it
imports the trained class. That costs a dependency on rl_training (and through it snntorch
and rl_games) in the TRAINING container only -- the flight container still sees nothing but
an ONNX graph.

`num_steps` IS NOT IN THE CHECKPOINT, AND IT IS NOT SHAPE-CHECKED. Every weight in a PopSAN
actor has a shape independent of the number of spiking timesteps, so a checkpoint loaded at
the wrong `num_steps` loads with strict=True, warns about nothing, and produces a policy
that correlates ~0.99 with the real one while being wrong by ~16% of full scale on every
actuator. Measured on the f450_hover_snn ep_1000 checkpoint:

    exported num_steps    max abs deviation    correlation
        3                      0.356               0.961
        4                      0.162               0.994
        6                      0.132               0.997
        8                      0.297               0.989

It flies. It flies badly, forever, and nothing reports it.

So `num_steps` has to come from outside the weights, and this file will only take it from
the config.yaml the runner froze into the run directory -- not from cfg/*.yaml, which is a
living file that drifts independently of any checkpoint trained from it. To make that file
trustworthy rather than merely conventional, `verify_config_against_weights` below checks
every hyperparameter that IS recoverable from the state_dict against the same config. Ten
independent quantities have to agree before the config is accepted. A config.yaml that
matches all ten is the config this checkpoint was trained with, which is the strongest
available evidence that its `num_steps` is right too.

Only `num_steps` and `reset_delay` cannot be reached this way. `spike_grad` also cannot,
and does not matter: it is the surrogate used for the backward pass, and inference never
touches it.
"""

from pathlib import Path
from typing import Dict, List, Union

import torch
import yaml

from rl_training.rl_games.networks.snn.pop_spiking_actor import (
    PopulationSpikingActorNetwork,
)

from .checkpoint import RlGamesPolicy

# Where the spiking actor's weights sit inside an rl_games a2c_continuous checkpoint.
ACTOR_PREFIX = "a2c_network.spiking_actor."

# snntorch stores the reset rule as a number (snntorch._neurons.neurons.SpikingNeuron
# .reset_dict), which is what lands in the state_dict.
RESET_MECHANISM_VALUES = {"subtract": 0.0, "zero": 1.0, "none": 2.0}

# Hyperparameters no amount of checking can recover from the weights. Named here because
# the exporter reports them, and because the list being short is the point.
UNVERIFIABLE_KEYS = ("num_steps", "reset_delay")

_FLOAT_TOLERANCE = 1e-6


def load_actor_config(config_path: Union[str, Path]) -> Dict:
    """Read the actor block from a run's frozen config.yaml.

    Expects the file the runner writes to runs/<experiment>/<timestamp>/config.yaml, which
    records the config AFTER command-line overrides and encoder-bound resolution -- i.e.
    the network that was actually built, not the one the source YAML asked for.
    """
    config_path = Path(config_path)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    try:
        network = config["params"]["network"]
    except (KeyError, TypeError):
        raise ValueError(
            f"{config_path} has no params.network block: this is not an rl_games config"
        ) from None

    if str(network.get("name", "")).lower() != "popsan":
        raise ValueError(
            f"{config_path} describes network {network.get('name')!r}, not 'popsan'. "
            "Wrong config for a spiking checkpoint."
        )

    actor = dict(network["actor"])
    for key in UNVERIFIABLE_KEYS:
        if key not in actor:
            raise ValueError(
                f"{config_path} does not set network.actor.{key}. It cannot be recovered "
                "from the checkpoint, so there is no safe default -- see this module's "
                "docstring."
            )
    # PopulationSpikingActorNetwork reads pop_dim at the top level and again inside the
    # encoder block; the YAML only states it once.
    actor["pop_dim"] = actor["encoder"]["pop_dim"]
    return actor


def verify_config_against_weights(
    actor_config: Dict, actor_weights: Dict, source: Union[str, Path]
) -> List[str]:
    """Check the config against every hyperparameter the state_dict actually records.

    Returns the list of quantities verified. Raises ValueError listing every mismatch, not
    just the first -- a config from the wrong run usually disagrees in several places at
    once, and seeing all of them is what identifies which run it came from.
    """
    mismatches: List[str] = []
    verified: List[str] = []

    def compare(name, expected, actual):
        verified.append(name)
        if isinstance(expected, float) or isinstance(actual, float):
            if abs(float(expected) - float(actual)) > _FLOAT_TOLERANCE:
                mismatches.append(f"  {name}: config says {expected}, weights say {actual}")
        elif expected != actual:
            mismatches.append(f"  {name}: config says {expected}, weights say {actual}")

    means = actor_weights["pop_encoder.means"]           # [1, obs_dim, pop_dim]
    fc1 = actor_weights["actor_fc1.weight"]              # [hidden0, obs_dim * pop_dim]
    fc2 = actor_weights["actor_fc2.weight"]              # [hidden1, hidden0]
    fc3 = actor_weights["actor_fc3.weight"]              # [action_dim * pop_dim, hidden1]
    decoder = actor_weights["action_decoder.decoder.weight"]  # [action_dim, 1, pop_dim]

    pop_dim = int(means.shape[2])
    compare("pop_dim", int(actor_config["pop_dim"]), pop_dim)
    compare("encoder.pop_dim", int(actor_config["encoder"]["pop_dim"]), int(decoder.shape[2]))
    compare(
        "hidden_dims",
        [int(d) for d in actor_config["hidden_dims"]],
        [int(fc1.shape[0]), int(fc2.shape[0])],
    )
    # fc3 fans out to action_dim * pop_dim; with pop_dim pinned above this pins action_dim.
    compare("action_dim (via actor_fc3)", int(fc3.shape[0]), int(decoder.shape[0]) * pop_dim)

    # The three Synaptic layers were built from one set of values, so any of them will do --
    # but check all three, because a hand-edited checkpoint is exactly when that stops
    # being true.
    for layer in ("actor_lif1", "actor_lif2", "actor_lif3"):
        compare(f"{layer}.alpha", float(actor_config["alpha"]), float(actor_weights[f"{layer}.alpha"]))
        compare(f"{layer}.beta", float(actor_config["beta"]), float(actor_weights[f"{layer}.beta"]))
        compare(
            f"{layer}.threshold",
            float(actor_config["threshold"]),
            float(actor_weights[f"{layer}.threshold"]),
        )
        compare(
            f"{layer}.reset_mechanism",
            RESET_MECHANISM_VALUES[actor_config["reset_mechanism"]],
            float(actor_weights[f"{layer}.reset_mechanism_val"]),
        )

    compare(
        "encoder.threshold",
        float(actor_config["encoder"]["threshold"]),
        float(actor_weights["pop_encoder.if1.threshold"]),
    )

    # The encoder's clamp window. A buffer, never trained, so it should be bit-identical to
    # what the config asked for -- and it is the one place a stale bounds measurement would
    # show up.
    config_bounds = torch.tensor(
        [[float(lo), float(hi)] for lo, hi in actor_config["observation_bounds"]],
        dtype=torch.float32,
    )
    checkpoint_bounds = actor_weights["pop_encoder.obs_bounds"].float()
    verified.append("observation_bounds")
    if config_bounds.shape != checkpoint_bounds.shape:
        mismatches.append(
            f"  observation_bounds: config has {tuple(config_bounds.shape)}, "
            f"weights have {tuple(checkpoint_bounds.shape)}"
        )
    else:
        worst = float((config_bounds - checkpoint_bounds).abs().max())
        if worst > _FLOAT_TOLERANCE:
            bad = (config_bounds - checkpoint_bounds).abs().max(dim=1).values
            rows = [i for i, d in enumerate(bad) if float(d) > _FLOAT_TOLERANCE]
            mismatches.append(
                f"  observation_bounds: differs by up to {worst:.3e} on dims {rows}"
            )

    if mismatches:
        raise ValueError(
            f"{source} does not describe this checkpoint:\n"
            + "\n".join(mismatches)
            + "\n\nThis config is from a different run. Use the config.yaml the runner "
            "froze into the checkpoint's own run directory."
        )
    return verified


class MuOnlyActor(torch.nn.Module):
    """Adapts the training network's (obs_dict) -> (mu, log_std) signature to tensor -> mu.

    Also the thing that gets exported, so the graph the vehicle runs is literally the
    callable this policy's `infer` runs -- not a parallel re-derivation of it.
    """

    def __init__(self, network: PopulationSpikingActorNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network({"obs": obs})[0]


class PopSANPolicy(RlGamesPolicy):
    """A PopSAN spiking actor, restored from a .pth plus its run's frozen config.yaml.

    Still abstract: `build_observation` is the training task's business (see
    SnnHoverPolicy).
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Union[str, Path],
        obs_dim: int,
        action_dim: int = 4,
        device: str = "cpu",
    ) -> None:
        # Both set before super().__init__, which calls _build_actor.
        self._config_path = Path(config_path)
        self._actor_config = load_actor_config(self._config_path)
        self.verified_hyperparameters: List[str] = []
        super().__init__(
            checkpoint_path=checkpoint_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

    @property
    def num_steps(self) -> int:
        """Spiking timesteps per forward pass. Frozen into the graph by unrolling."""
        return int(self._actor_config["num_steps"])

    @property
    def pop_dim(self) -> int:
        return int(self._actor_config["pop_dim"])

    @property
    def reset_delay(self) -> bool:
        return bool(self._actor_config["reset_delay"])

    def _actor_input_dim(self) -> int:
        return int(self._actor.network.pop_encoder.obs_dim)

    def _build_actor(self, weights: dict) -> torch.nn.Module:
        actor_weights = {
            key[len(ACTOR_PREFIX):]: value
            for key, value in weights.items()
            if key.startswith(ACTOR_PREFIX)
        }
        if not actor_weights:
            raise ValueError(
                f"no {ACTOR_PREFIX}* weights in checkpoint: this is not a PopSAN "
                "checkpoint (a dense policy loads with deploy.checkpoint.MlpActorPolicy)"
            )

        self.verified_hyperparameters = verify_config_against_weights(
            self._actor_config, actor_weights, self._config_path
        )

        network = PopulationSpikingActorNetwork(
            self.obs_dim, self.action_dim, **self._actor_config
        )
        # strict=True: after the check above, an unexpected or missing key means the
        # network class has moved on from the checkpoint, which is not something to
        # tolerate silently on a deployment path.
        network.load_state_dict(actor_weights, strict=True)
        return MuOnlyActor(network)
