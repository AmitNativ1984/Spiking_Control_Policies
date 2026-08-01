from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor

from .f450_config import F450Config

robot_registry.register("f450", BaseMultirotor, F450Config)
