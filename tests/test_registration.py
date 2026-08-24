"""`import config` must register everything, in an order that actually resolves.

config/task_config registers a task naming an env that config/env_config registers. If
config/__init__.py ever stops importing them in dependency order, the task resolves to a
missing env only when someone tries to build it — a failure a long way from its cause.
"""
import config  # noqa: F401  — the import under test
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry

TASK_NAME = "f450_navigation_task"
ENV_NAME = "forest_with_obstacles_env"
ROBOT_NAME = "f450"


def test_task_is_registered():
    assert TASK_NAME in task_registry.get_task_names()


def test_env_is_registered():
    assert ENV_NAME in env_config_registry.get_env_names()


def test_robot_is_registered():
    assert ROBOT_NAME in robot_registry.get_robot_names()


def test_task_config_names_a_registered_env_and_robot(task_config):
    """The task config's env_name/robot_name have to resolve, or make_task fails at
    build time with a KeyError far from the config that caused it."""
    assert task_config.env_name in env_config_registry.get_env_names()
    assert task_config.robot_name in robot_registry.get_robot_names()


def test_observation_layout_tiles_the_observation_space(task_config):
    """The layout says what every dimension means; a gap or an overlap would mislabel
    observations for every consumer (encoder bounds, obs-stats names, trace plots)."""
    indices = [i for obs_slice, _ in task_config.observation_layout
               for i in range(obs_slice.start, obs_slice.stop)]

    assert sorted(indices) == list(range(task_config.observation_space_dim))
