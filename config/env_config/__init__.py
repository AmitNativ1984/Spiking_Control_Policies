from aerial_gym.registry.env_registry import env_config_registry

from .env_forest_with_obstacles import ForestEnvCfg
from .env_empty import EmptyEnvNewCfg

env_config_registry.register("forest_with_obstacles_env", ForestEnvCfg)
env_config_registry.register("empty_new_env", EmptyEnvNewCfg)
