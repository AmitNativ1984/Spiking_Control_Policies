from aerial_gym.registry.task_registry import task_registry

from task.attitude_navigation_task import NavigationWithObstaclesTask
from task.hover_task import HoverTask

from .f450_attitude_navigation_task_config import task_config as F450NavTaskConfig
from .f450_hover_task_config import task_config as HoverTaskConfig
# env_name = "forest_with_obstacles_env" (set in F450NavTaskConfig) is registered in
# config/env_config/__init__.py — must be imported before this task actually runs.

# NOTE: this is the F450 attitude-control task in task/attitude_navigation_task.py, NOT
# the older navigation_with_obstacles/task/navigation_task.py. Both classes are named
# NavigationWithObstaclesTask, so check the import path before assuming which one runs.
task_registry.register_task(
    "f450_navigation_task", NavigationWithObstaclesTask, F450NavTaskConfig
)

task_registry.register_task(
    "f450_hover_task", HoverTask, HoverTaskConfig
)