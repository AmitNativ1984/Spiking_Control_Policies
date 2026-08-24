"""Shared pytest fixtures for the refactored tree (config/, task/, rl_training/).

isaacgym must be imported before torch — done here so it happens during pytest
collection, before any test module imports torch.

Isaac Gym does NOT support creating a second sim in the same process, so the one test
that needs a real environment shares a single session-scoped task.
"""
import isaacgym  # noqa: F401  (must precede torch)

import pytest
import torch

import config  # noqa: F401  — registers the env, robot and task
from aerial_gym.registry.task_registry import task_registry
from config.task_config import F450NavTaskConfig

TASK_NAME = "f450_navigation_task"
NUM_ENVS = 16


@pytest.fixture(scope="session")
def task_config():
    return F450NavTaskConfig


@pytest.fixture(scope="session")
def num_envs():
    return NUM_ENVS


@pytest.fixture
def zero_actions(task, task_config):
    """A no-op action batch sized for the task."""
    return torch.zeros((NUM_ENVS, task_config.action_space_dim), device=task.device)


@pytest.fixture(scope="session")
def task():
    """The real task — built once, shared by every test that needs it, closed at the end.

    Skipped without a GPU; Isaac Gym's tensor pipeline needs CUDA here.
    """
    if not torch.cuda.is_available():
        pytest.skip("building the task needs CUDA")

    t = task_registry.make_task(TASK_NAME, num_envs=NUM_ENVS, headless=True, use_warp=True)
    t.reset()

    # Evidence for the FIRST episode, recorded here because it cannot be recovered later:
    # the fixture steps below, and only one sim may exist per process so a test cannot build
    # a second task to look at episode 1. Isaac Gym drops root states staged before the
    # first simulate(), which used to silently dump every env at the world origin for
    # episode 1 only. See test_first_episode_spawns_inside_the_spawn_box.
    zero = torch.zeros((NUM_ENVS, F450NavTaskConfig.action_space_dim), device=t.device)
    # Bounds are cloned at the same instant: they are re-randomized on every reset, so the
    # spawn ratio is only meaningful against the bounds that were live when it was drawn.
    t.first_episode_spawn = t.obs_dict["robot_position"].clone()
    t.first_episode_bounds = (
        t.obs_dict["env_bounds_min"].clone(),
        t.obs_dict["env_bounds_max"].clone(),
    )
    t.step(zero)
    t.first_episode_after_one_step = t.obs_dict["robot_position"].clone()

    for _ in range(4):  # a few more steps so the state isn't the trivial post-reset one
        t.step(zero)
    yield t
    t.close()
