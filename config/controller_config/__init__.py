from aerial_gym.registry.controller_registry import controller_registry
from aerial_gym.control.controllers.attitude_control import LeeAttitudeController

from .f450_lee_attitude_config import F450LeeAttitudeConfig

# Registered under its own name rather than replacing "lee_attitude_control", so the stock
# controller keeps its stock (unrandomized) config for every other task in the tree.
controller_registry.register_controller(
    "f450_lee_attitude_control", LeeAttitudeController, F450LeeAttitudeConfig
)
