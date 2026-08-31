"""Adapter between an aerial_gym task and rl_games' vectorized-env interface.

An aerial_gym task returns a dict of observations and separate terminated/truncated
flags; rl_games wants a flat observation tensor and a single `dones`. The two classes
below bridge that, and `register_task()` wires a task name into both of rl_games'
registries.
"""

import gym
import numpy as np
import torch
from gym import spaces
from loguru import logger
from rl_games.common import env_configurations, vecenv

from aerial_gym.registry.task_registry import task_registry

VECENV_TYPE = "AERIAL-RLGPU"


class ExtractObsWrapper(gym.Wrapper):
    """Unwraps the task's observation dict to the plain tensor rl_games expects, and
    merges terminated/truncated into a single `dones` flag."""

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )
        return observations["observations"], rewards, dones, infos


class AerialRLGPUEnv(vecenv.IVecEnv):
    """Creates the task registered under `config_name` and wraps it for rl_games."""

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def render(self, mode="human"):
        # rl_games' player calls this when player.render is True. The viewer is already
        # drawn inside step(); the task's own render() would trigger a full-batch sensor
        # capture instead of a viewer draw, so forwarding it would be wrong, not just slow.
        return None

    def get_env_info(self):
        """Observation and action spaces, taken from the task config.

        Actions are the policy's raw [-1, 1] output; the task maps them onto the
        controller's real command range. Observations are unbounded here because
        rl_games normalizes them itself (normalize_input).
        """
        task_config = self.env.task_config
        info = {
            "action_space": spaces.Box(
                -np.ones(task_config.action_space_dim),
                np.ones(task_config.action_space_dim),
            ),
            "observation_space": spaces.Box(
                np.ones(task_config.observation_space_dim) * -np.Inf,
                np.ones(task_config.observation_space_dim) * np.Inf,
            ),
        }
        logger.info(f"Action space: {info['action_space']}")
        logger.info(f"Observation space: {info['observation_space']}")
        return info


def register_task(task_name: str) -> None:
    """Make an aerial_gym task name usable as an rl_games `env_name`.

    The task itself must already be in aerial_gym's task_registry (`import config`).
    """
    env_configurations.register(
        task_name,
        {
            "env_creator": lambda **kwargs: task_registry.make_task(task_name, **kwargs),
            "vecenv_type": VECENV_TYPE,
        },
    )
    vecenv.register(
        VECENV_TYPE,
        lambda config_name, num_actors, **kwargs: AerialRLGPUEnv(
            config_name, num_actors, **kwargs
        ),
    )
