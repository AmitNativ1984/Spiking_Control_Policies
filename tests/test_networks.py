"""Network builders: correct shapes, and no hidden dependency on a task config.

The point of the last group of tests is that these networks are constructible from a
plain dict. The legacy PopSAN read task_config at module scope, which made it
un-testable without a simulator and coupled it to a global mutated mid-run.
"""
import pytest
import torch

from rl_training.rl_games.networks import (
    GRUActorCriticNetworkBuilder,
    MLPActorCriticNetworkBuilder,
    POPSANNetworkBuilder,
    bind_encoder_bounds,
    bounds_from_layout,
)
from rl_training.rl_games.networks.snn import DEFAULT_TYPE_BOUNDS

OBS_DIM = 49
ACTION_DIM = 4
BATCH = 8

MLP_PARAMS = {
    "actor": {"hidden_dims": [64, 32], "activation": "elu"},
    "critic": {"hidden_dims": [64, 32], "activation": "elu"},
}
GRU_PARAMS = {
    "actor": {"hidden_dims": [64, 32], "activation": "elu",
              "gru": {"hidden_size": 32, "num_layers": 1}},
    "critic": {"hidden_dims": [64, 32], "activation": "elu"},
}
POPSAN_PARAMS = {
    "actor": {
        "hidden_dims": [64, 32], "num_steps": 5, "spike_grad": "fast_sigmoid",
        "alpha": 0.5, "beta": 0.75, "threshold": 0.5,
        "reset_mechanism": "subtract", "reset_delay": False,
        "encoder": {"pop_dim": 4, "threshold": 0.95},
        "observation_bounds": [(-3.0, 3.0)] * OBS_DIM,
    },
    "critic": {"hidden_dims": [64, 32], "activation": "elu"},
}


def build(builder_cls, params, **kwargs):
    builder = builder_cls()
    builder.load(params)
    return builder.build("net", input_shape=(OBS_DIM,), actions_num=ACTION_DIM, **kwargs)


@pytest.fixture
def obs():
    return {"obs": torch.randn(BATCH, OBS_DIM)}


@pytest.mark.parametrize("builder_cls,params", [
    (MLPActorCriticNetworkBuilder, MLP_PARAMS),
    (POPSANNetworkBuilder, POPSAN_PARAMS),
])
def test_feedforward_forward_shapes(builder_cls, params, obs):
    net = build(builder_cls, params)
    assert net.is_rnn() is False

    mu, log_std, value, states = net(obs)
    assert mu.shape == (BATCH, ACTION_DIM)
    assert log_std.shape == (BATCH, ACTION_DIM)
    assert value.shape == (BATCH, 1)
    assert states is None
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


def test_gru_forward_shapes_and_state(obs):
    net = build(GRUActorCriticNetworkBuilder, GRU_PARAMS, num_seqs=BATCH)
    assert net.is_rnn() is True

    hidden = net.get_default_rnn_state()
    assert hidden[0].shape == (1, BATCH, 32)

    mu, log_std, value, states = net({**obs, "rnn_states": hidden, "seq_length": 1})
    assert mu.shape == (BATCH, ACTION_DIM)
    assert value.shape == (BATCH, 1)
    assert states[0].shape == hidden[0].shape
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()


# --- the decoupling ---------------------------------------------------------


def test_popsan_builds_without_a_task_config(obs):
    """Constructible from a plain dict — no task config, no simulator."""
    net = build(POPSANNetworkBuilder, POPSAN_PARAMS)
    assert torch.isfinite(net(obs)[0]).all()


def test_popsan_rejects_mismatched_bounds():
    """Wrong-length bounds must fail loudly at construction, not silently mis-encode."""
    params = {"actor": {**POPSAN_PARAMS["actor"], "observation_bounds": [(-3.0, 3.0)] * 10},
              "critic": POPSAN_PARAMS["critic"]}
    with pytest.raises(AssertionError, match="observation_bounds"):
        build(POPSANNetworkBuilder, params)


# --- bounds_from_layout -----------------------------------------------------


def test_bounds_expand_per_type(task_config):
    """Each index gets the window its layout entry's TYPE declares."""
    bounds = bounds_from_layout(task_config.observation_layout,
                                task_config.observation_space_dim)

    assert len(bounds) == task_config.observation_space_dim
    for obs_slice, obs_type in task_config.observation_layout:
        expected = DEFAULT_TYPE_BOUNDS[obs_type]
        assert all(b == expected for b in bounds[obs_slice])


def test_bounds_honour_per_type_overrides():
    """The point of a per-type table: widen one kind of input without touching the rest."""
    layout = [(slice(0, 2), "linvel"), (slice(2, 6), "vae_latent")]
    bounds = bounds_from_layout(layout, 6, {**DEFAULT_TYPE_BOUNDS,
                                            "vae_latent": (-5.0, 5.0)})

    assert bounds[:2] == [DEFAULT_TYPE_BOUNDS["linvel"]] * 2
    assert bounds[2:] == [(-5.0, 5.0)] * 4


def test_bounds_reject_an_unknown_type():
    """An observation type with no declared window must fail loudly — a silent default
    would mis-scale that dimension's receptive fields."""
    with pytest.raises(KeyError, match="brand_new_sensor"):
        bounds_from_layout([(slice(0, 3), "brand_new_sensor")], 3)


def test_bounds_reject_an_incomplete_layout():
    with pytest.raises(ValueError, match="uncovered"):
        bounds_from_layout([(slice(0, 2), "linvel")], 4)


# --- bind_encoder_bounds ----------------------------------------------------


def test_bind_derives_bounds_from_the_task_layout(task_config):
    cfg = {"params": {"network": {"name": "popsan", "actor": {}, "critic": {}}}}
    bind_encoder_bounds(cfg, task_config)

    assert cfg["params"]["network"]["actor"]["observation_bounds"] == bounds_from_layout(
        task_config.observation_layout, task_config.observation_space_dim
    )


def test_bind_does_not_override_explicit_yaml_values(task_config):
    """How the runner installs teacher-measured bounds: it writes them first, and this
    must leave them alone."""
    cfg = {"params": {"network": {"actor": {"observation_bounds": [(-1.0, 1.0)]}}}}
    bind_encoder_bounds(cfg, task_config)

    assert cfg["params"]["network"]["actor"]["observation_bounds"] == [(-1.0, 1.0)]


def test_bind_ignores_networks_without_an_actor_block(task_config):
    """rl_games' built-in actor_critic has no `actor:` block; binding must be a no-op."""
    cfg = {"params": {"network": {"name": "actor_critic", "mlp": {"units": [64]}}}}
    bind_encoder_bounds(cfg, task_config)
    assert cfg["params"]["network"] == {"name": "actor_critic", "mlp": {"units": [64]}}


def test_bind_ignores_a_task_with_no_layout():
    """A task that publishes no observation_layout (a hover task, say) must not break
    a run that uses a network needing no encoder bounds."""
    class HoverCfg:
        observation_space_dim = 13
        action_space_dim = 4

    cfg = {"params": {"network": {"name": "mlp_actor_critic", "actor": {}, "critic": {}}}}
    bind_encoder_bounds(cfg, HoverCfg)
    assert cfg["params"]["network"]["actor"] == {}
