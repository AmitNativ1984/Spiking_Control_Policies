"""Registers the depth-collection env and camera carrier with aerial_gym.

Both are thin subclasses of what the navigation task itself uses, so importing this
package also imports `config`, which registers "forest_with_obstacles_env" and "f450".
That is intentional -- the whole design is that the dataset is rendered in the env the
policy flies in, so the two must be loaded from the same place.

The urdfpy Cylinder/Box/Sphere monkey-patch that used to live here is gone. urdfpy 0.0.22
is abandoned and its Cylinder.meshes is broken, but that is now fixed at the import layer
by the urchin shim (Dockerfile.base, plus user site-packages) rather than per-caller --
which is why task/attitude_navigation_task.py loads the same cylinder-based tree URDFs
with no patch of its own.
"""

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor

from data_generation.config.env_config import DataGenEnvCfg
from data_generation.config.robot_config import DataGenF450Cfg

env_config_registry.register("data_gen_env", DataGenEnvCfg)
robot_registry.register("data_gen_quad", BaseMultirotor, DataGenF450Cfg)
