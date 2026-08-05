"""Every shipped config must resolve into a model rl_games can actually build.

A YAML naming an unregistered network or a stale task fails only once a run starts —
after the sim is built, minutes in. These checks are instant.
"""
import glob
import os

import pytest
import yaml
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import ModelBuilder

import rl_training.rl_games.networks  # noqa: F401  — registers the custom builders
from aerial_gym.registry.task_registry import task_registry
from rl_training.rl_games.networks import bind_encoder_bounds

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "rl_training", "rl_games", "cfg")
CFG_PATHS = sorted(glob.glob(os.path.join(CFG_DIR, "*.yaml")))


def cfg_id(path):
    return os.path.basename(path)


@pytest.fixture(scope="module")
def cfg_paths():
    assert CFG_PATHS, f"no configs found in {CFG_DIR}"
    return CFG_PATHS


def test_configs_exist(cfg_paths):
    assert len(cfg_paths) >= 1


@pytest.mark.parametrize("path", CFG_PATHS, ids=cfg_id)
def test_config_names_a_registered_task(path):
    params = yaml.safe_load(open(path))["params"]
    assert params["config"]["env_name"] in task_registry.get_task_names()


def test_custom_network_builders_are_registered():
    """model_builder.NETWORK_REGISTRY holds only CUSTOM builders — rl_games' built-ins
    (e.g. 'actor_critic') resolve through NetworkBuilder's own factory and never appear
    here. So this checks our three, and test_config_builds_a_model covers the rest."""
    assert {"mlp_actor_critic", "mlp_gru_actor_critic", "popsan"} <= set(
        model_builder.NETWORK_REGISTRY
    )


@pytest.mark.parametrize("path", CFG_PATHS, ids=cfg_id)
def test_config_builds_a_model(path, task_config):
    """Build the model the way the runner does, and check it is sized for this task."""
    config_dict = yaml.safe_load(open(path))
    bind_encoder_bounds(config_dict, task_config)
    params = config_dict["params"]

    model = ModelBuilder().load(
        {"model": params["model"], "network": params["network"]}
    ).build({
        "actions_num": task_config.action_space_dim,
        "input_shape": (task_config.observation_space_dim,),
        "num_seqs": 8,
        "value_size": 1,
        "normalize_input": params["config"]["normalize_input"],
        "normalize_value": params["config"]["normalize_value"],
    })

    assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.parametrize("path", CFG_PATHS, ids=cfg_id)
def test_minibatch_divides_the_batch(path):
    """rl_games asserts minibatch_size <= num_actors * horizon_length at startup."""
    cfg = yaml.safe_load(open(path))["params"]["config"]
    assert cfg["minibatch_size"] <= cfg["num_actors"] * cfg["horizon_length"]


@pytest.mark.parametrize("path", CFG_PATHS, ids=cfg_id)
def test_env_count_has_one_source_of_truth(path):
    """num_actors is the environment count; the runner derives env_config.num_envs from
    it. A YAML declaring both lets them drift, and the mismatch surfaces only as a tensor-
    shape RuntimeError after the sim has finished building."""
    cfg = yaml.safe_load(open(path))["params"]["config"]
    assert "num_envs" not in cfg["env_config"], \
        "remove env_config.num_envs — num_actors is the single source of truth"
    assert cfg["num_actors"] > 0


@pytest.mark.parametrize("path", CFG_PATHS, ids=cfg_id)
def test_teacher_critic_matches_student_critic(path):
    """For distillation runs the ANN critic is copied into the SNN student, so the two
    critic trunks must have identical hidden_dims or the warm-start is silently skipped
    (a2c_teacher_agent catches the shape error and continues)."""
    params = yaml.safe_load(open(path))["params"]
    distill = params["config"].get("distillation")
    if distill is None:
        pytest.skip("not a distillation config")

    assert distill["network"]["critic"]["hidden_dims"] == params["network"]["critic"]["hidden_dims"]
