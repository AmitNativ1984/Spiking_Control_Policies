from aerial_gym.registry.task_registry import task_registry

from navigation_with_obstacles.task.navigation_task import NavigationWithObstaclesTask

from .f450_attitude_navigation_task_config import task_config as F450NavTaskConfig

# env_name = "navigation_obstacle_env" (set in F450NavTaskConfig) is registered separately,
# via config/env_config/__init__.py — must be imported before this task actually runs.

# TODO: Update with the correct navigation with obstacles task class and config once they are implemented for the F450 quadrotor.
task_registry.register_task(
    "f450_navigation_task", NavigationWithObstaclesTask, F450NavTaskConfig
)
