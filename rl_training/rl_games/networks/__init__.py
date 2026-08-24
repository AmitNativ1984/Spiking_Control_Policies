"""Custom rl_games network builders.

Importing this package registers every builder below, so a runner only needs::

    import rl_training.rl_games.networks  # noqa: F401

then names them from a YAML's `network.name`.
"""

from rl_games.algos_torch import model_builder

from .ann import GRUActorCriticNetworkBuilder, MLPActorCriticNetworkBuilder
from .snn import POPSANNetworkBuilder, bounds_from_layout
from .teacher_student import build_teacher

__all__ = [
    "MLPActorCriticNetworkBuilder",
    "GRUActorCriticNetworkBuilder",
    "POPSANNetworkBuilder",
    "build_teacher",
    "bind_encoder_bounds",
    "bounds_from_layout",
]


def _register(name, builder):
    """Register `builder` under `name`, refusing to overwrite an existing entry.

    rl_games' NETWORK_REGISTRY is a plain process-global dict and register_network()
    overwrites silently. Importing this package alongside the legacy
    navigation_with_obstacles runner (which registers its own copies of these
    networks) must fail loudly rather than quietly swap one tree's networks for the
    other's.
    """
    if name in model_builder.NETWORK_REGISTRY:
        raise RuntimeError(
            f"rl_games network name {name!r} is already registered. Something else — "
            "most likely navigation_with_obstacles.training.runner — registered it "
            "first. Import only one of the two trees per process."
        )
    model_builder.register_network(name, builder)


_register("mlp_actor_critic", MLPActorCriticNetworkBuilder)
_register("mlp_gru_actor_critic", GRUActorCriticNetworkBuilder)
_register("popsan", POPSANNetworkBuilder)


def bind_encoder_bounds(config: dict, task_config) -> dict:
    """Give the PopSAN encoder its per-dimension clamp bounds, derived from the task.

    The only task -> network bridge in this tree, and it exists because rl_games has no
    channel for one: a network builder receives exactly `actions_num`, `input_shape`,
    `num_seqs`, `value_size`, `normalize_value`, `normalize_input`. That is enough for
    every stock rl_games network, which is why upstream aerial_gym needs nothing like
    this. The population encoder additionally needs a window per observation dimension,
    which it gets by expanding the task's observation_layout through its own per-type
    table (snn/encoder.py: DEFAULT_TYPE_BOUNDS).

    Deriving rather than restating it in the YAML keeps a 49-entry list out of the config
    and makes it impossible for the bounds to disagree with the observation vector.

    A value already in the YAML wins, so a config can pin bounds explicitly — which is
    also how the runner installs bounds measured from a teacher rollout. Networks with no
    `actor` block, and tasks that publish no layout, are left alone.

    Args:
        config: The full parsed rl_games config (the dict with a "params" key).
        task_config: The task config class whose observation layout to expand.

    Returns:
        The same `config`, mutated in place.
    """
    actor = config["params"]["network"].get("actor")
    layout = getattr(task_config, "observation_layout", None)
    if actor is None or layout is None:
        return config

    actor.setdefault(
        "observation_bounds",
        bounds_from_layout(layout, task_config.observation_space_dim),
    )
    return config
