"""Configuration package for this repo, mirroring aerial_gym.config.

Importing this package registers everything this repo adds to aerial_gym's registries.
A runner only needs::

    import isaacgym   # must precede torch
    import config     # env, robot and task are now registered

The sub-imports below are ordered by dependency and MUST stay in this order:

    env_config   registers "forest_with_obstacles_env"  (uses asset_config classes)
    robot_config registers "f450"                       (uses sensor_config classes)
    task_config  registers "f450_navigation_task"       (names the env and robot above)

asset_config and sensor_config are plain data modules with no registration side effects;
they are imported directly by the modules that use them.
"""

from . import env_config  # noqa: F401
from . import robot_config  # noqa: F401
from . import task_config  # noqa: F401
