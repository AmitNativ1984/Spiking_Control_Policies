"""
Shared pytest fixtures for the navigation_with_obstacles tests.

Why a session-scoped shared task: each test builds a full Isaac Gym sim, and
Isaac Gym does NOT support creating a second sim instance in the same process.
Running both test files under one `pytest` invocation would otherwise build the
sim twice and hang/abort the process. So we build it exactly ONCE here, share it
across every test, step it a few times for non-trivial state, and close it at
teardown.

isaacgym must be imported before torch — done here so it happens during pytest
collection, before any test module imports torch.
"""
import isaacgym  # noqa: F401  (must precede torch)
import torch
import pytest

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor

from navigation_with_obstacles.task.navigation_task import NavigationWithObstaclesTask
from navigation_with_obstacles.config.task_config import task_config
from navigation_with_obstacles.config.env_config import NavigationObstacleEnvCfg
from navigation_with_obstacles.config.robot_config import NavQuadWithCameraCfg

NUM_ENVS = 16
TASK_NAME = "navigation_with_obstacles_task"

# Register env/robot/task once at collection time (register_task overwrites, so
# repeated registration is harmless, but doing it once here keeps it tidy).
env_config_registry.register("navigation_obstacle_env", NavigationObstacleEnvCfg)
robot_registry.register("nav_quadrotor_with_camera", BaseMultirotor, NavQuadWithCameraCfg)
task_registry.register_task(TASK_NAME, NavigationWithObstaclesTask, task_config)


def build_task(num_envs=NUM_ENVS, warmup_steps=5):
    """Build the real task, reset, and step a few times for non-trivial state.

    Shared by the pytest fixture and the standalone __main__ paths in each test
    file (so a single file can still be run directly with `python <file>`).
    """
    task = task_registry.make_task(TASK_NAME, num_envs=num_envs, headless=True)
    task.reset()
    for _ in range(warmup_steps):
        task.step(torch.zeros((num_envs, 4), device=task.device))
    return task


@pytest.fixture(scope="session")
def task():
    """Session-scoped real task — built once, shared by all tests, closed at end."""
    t = build_task()
    yield t
    t.close()
