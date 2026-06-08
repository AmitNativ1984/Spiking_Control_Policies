"""
Custom runner for Navigation with Obstacles Task.

Registers the custom task, environment, and robot with aerial_gym and rl_games,
then runs PPO training.

Usage:
    cd /workspaces/aerial_gym_docker
    python -m navigation_with_obstacles.training.runner \
        --file=navigation_with_obstacles/training/ppo_navigation.yaml --train
"""
import isaacgym
import argparse
import logging
import os
import sys
import yaml
import numpy as np

logging.getLogger("asset_manager").setLevel(logging.ERROR)

sys.path.insert(0, "/workspaces/aerial_gym_docker")
import wandb
import torch
import gym
from gym import spaces
from loguru import logger
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner


from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor
from aerial_gym.utils.helpers import parse_arguments
from distutils.util import strtobool
from navigation_with_obstacles.task.navigation_task import (
    NavigationWithObstaclesTask,
)
from navigation_with_obstacles.config.task_config import task_config
from navigation_with_obstacles.config.env_config import NavigationObstacleEnvCfg
from navigation_with_obstacles.config.robot_config import NavQuadWithCameraCfg
from navigation_with_obstacles.networks.snn.popsan import POPSANNetworkBuilder
from navigation_with_obstacles.networks.ann.actor_critic import MLPActorCriticNetworkBuilder
from navigation_with_obstacles.networks.ann.gru_actor_critic import GRUActorCriticNetworkBuilder
from rl_games.algos_torch import model_builder

# =============================================================================
# Register Custom Environment, Task, and Networks
# =============================================================================

# Register environment configuration
env_config_registry.register("navigation_obstacle_env", NavigationObstacleEnvCfg)

# Register robot configuration
robot_registry.register(
    "nav_quadrotor_with_camera", BaseMultirotor, NavQuadWithCameraCfg
)

# Register task
task_registry.register_task(
    "navigation_with_obstacles_task",
    NavigationWithObstaclesTask,
    task_config,
)

# Register custom SNN network builder with rl_games
model_builder.register_network("PopSAN", POPSANNetworkBuilder)
model_builder.register_network('mlp_actor_critic', MLPActorCriticNetworkBuilder)
model_builder.register_network('mlp_gru_actor_critic', GRUActorCriticNetworkBuilder)

# =============================================================================
# RL Games Integration
# =============================================================================


class ExtractObsWrapper(gym.Wrapper):
    """
    Wrapper that extracts the 'observations' tensor from the observation dict.
    rl_games expects flat observation tensors, not dicts.
    """

    def __init__(self, env):
        super().__init__(env)

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


class AERIALRLGPUEnv(vecenv.IVecEnv):
    """
    Vectorized environment wrapper for rl_games.
    Creates the task environment and wraps it for compatibility.
    """

    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](
            **kwargs
        )
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        """Return observation and action space info for rl_games."""
        
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.Inf,
            np.ones(self.env.task_config.observation_space_dim) * np.Inf,
        )
        logger.info(f"Action space: {info['action_space']}")
        logger.info(f"Observation space: {info['observation_space']}")
        return info


# Register task with rl_games env_configurations
env_configurations.register(
    "navigation_with_obstacles_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "navigation_with_obstacles_task", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU-NAV",
    },
)

# Register the vectorized environment type
vecenv.register(
    "AERIAL-RLGPU-NAV",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(
        config_name, num_actors, **kwargs
    ),
)


# =============================================================================
# Argument Parsing
# =============================================================================


def get_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--train", "action": "store_true", "help": "Train network"},
        {"name": "--play", "action": "store_true", "help": "Play/test network"},
        {"name": "--checkpoint", "type": str, "help": "Path to checkpoint"},
        {"name": "--file", "type": str, "default": "...", "help": "Path to config"},
        {"name": "--num_envs", "type": int, "default": None, "help": "Num envs (overrides YAML when set)"},
        {
            "name": "--headless",
            "type": lambda x: bool(strtobool(x)),
            "default": "False",
            "help": "Headless mode",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(strtobool(x)),
            "default": "True",
            "help": "Use warp",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "default": "navigation_with_obstacles",
            "help": "Experiment name",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_with_obstacles_task",
            "help": "Task name",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "Track with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "aerial_gym",
            "help": "Wandb project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "Wandb entity (team)",
        },
        {
            "name": "--curriculum_level",
            "type": int,
            "default": None,
            "help": "Fix curriculum (obstacle density) at this level (0-25). Overrides min/max.",
        },
        {
            "name": "--exceed_margin",
            "type": float,
            "default": None,
            "help": "Out-of-bounds margin multiplier (e.g. 1.5 = 50%% beyond bounds before termination)",
        },
        {
            "name": "--plot-encoding",
            "action": "store_true",
            "help": "Debug: when combined with --play, record PopSAN encoder activations and plot at end. Forces num_envs=1.",
        },
    ]
    args = parse_arguments(
        description="Navigation with Obstacles",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):
    """Update training config with command line arguments."""
    config["params"]["config"]["env_name"] = args["task"]
    config["params"]["config"]["name"] = args["experiment_name"]

    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]

    # Only override num_envs / num_actors if --num_envs was explicitly passed.
    # Otherwise the YAML's values stand.
    if args.get("num_envs") is not None and args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
        # Clamp minibatch_size to batch_size so rl_games assertion passes
        batch_size = args["num_envs"] * config["params"]["config"]["horizon_length"]
        if config["params"]["config"]["minibatch_size"] > batch_size:
            config["params"]["config"]["minibatch_size"] = batch_size

    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    # Fix curriculum level (obstacle density)
    if args.get("curriculum_level") is not None:
        level = args["curriculum_level"]
        task_config.curriculum.min_level = level
        task_config.curriculum.max_level = level

    # Expand out-of-bounds termination margin
    if args.get("exceed_margin") is not None:
        task_config.exceed_bounds_margin = args["exceed_margin"]

    # Resume from checkpoint
    if args.get("checkpoint"):
        config["params"]["load_checkpoint"] = True
        config["params"]["load_path"] = args["checkpoint"]

    # Merge use_vecenv into existing player config (don't overwrite YAML settings)
    player_cfg = config["params"]["config"].get("player", {})
    player_cfg["use_vecenv"] = True
    config["params"]["config"]["player"] = player_cfg

    return config


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Save runs/ under the navigation_with_obstacles folder
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(project_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    args = vars(get_args())
    config_name = args["file"]

    logger.info(f"Loading config: {config_name}")
    logger.info(f"Number of environments: {args['num_envs']}")
    logger.info(f"Headless: {args['headless']}")
    logger.info(f"Use warp: {args['use_warp']}")

    # Debug-only: --play --plot-encoding records the encoder during a single-env rollout
    # and plots Gaussian receptive fields + spike rasters at the end.
    plot_encoding = bool(args.get("play") and args.get("plot_encoding"))
    if plot_encoding:
        logger.warning("--plot-encoding set: forcing num_envs=1 for clean single-trajectory plots")
        args["num_envs"] = 1

    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        config = update_config(config, args)
        # Ensure rl_games saves tensorboard logs and checkpoints under
        # navigation_with_obstacles/runs/
        config["params"]["config"]["train_dir"] = runs_dir

        # DEBUG: surface the runtime values rl_games will actually use
        logger.debug(f"[DEBUG] args['num_envs'] = {args.get('num_envs')!r} (type={type(args.get('num_envs')).__name__})")
        logger.debug(f"[DEBUG] config.num_actors            = {config['params']['config']['num_actors']}")
        logger.debug(f"[DEBUG] config.env_config.num_envs   = {config['params']['config']['env_config']['num_envs']}")
        logger.debug(f"[DEBUG] config.horizon_length        = {config['params']['config']['horizon_length']}")
        logger.debug(f"[DEBUG] config.minibatch_size        = {config['params']['config']['minibatch_size']}")
        logger.debug(f"[DEBUG] config.seq_length            = {config['params']['config'].get('seq_length')}")

        runner = Runner(algo_observer=IsaacAlgoObserver())
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading config: {exc}")
            sys.exit(1)

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if args["track"] and rank == 0:
        wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            sync_tensorboard=True,
            config=config,
            monitor_gym=True,
            save_code=True,
        )

    logger.info(
        "Starting training..." if args.get("train") else "Starting playback..."
    )

    if plot_encoding:
        # Wrap run_play: enable recording on the encoder right after player construction,
        # plot once the play loop returns. Confined to the --plot-encoding branch.
        orig_run_play = runner.run_play

        def run_play_with_recording(args_):
            print("Started to play (with encoder recording)")
            player = runner.create_player()
            from rl_games.torch_runner import _restore, _override_sigma
            _restore(player, args_)
            _override_sigma(player, args_)

            # rl_games wraps the raw network in a ModelA2C*.Network at player.model,
            # which exposes the underlying network as `.a2c_network`. For PopSAN that's
            # POPSANNetwork → .spiking_actor → .pop_encoder.
            encoder = player.model.a2c_network.spiking_actor.pop_encoder
            encoder.record = True
            encoder._trace = []
            logger.info(f"[plot-encoding] encoder recording enabled on {type(encoder).__name__}")
            try:
                player.run()
            except KeyboardInterrupt:
                logger.info("[plot-encoding] play loop interrupted by user — proceeding to plot")
            finally:
                encoder.record = False
                logger.info(f"[plot-encoding] play loop finished, recorded {len(encoder._trace)} forward passes")
                try:
                    from navigation_with_obstacles.tools.plot_encoder_trace import plot_encoder_trace
                    plot_encoder_trace(encoder, encoder._trace, task_config.observation_layout)
                except Exception:
                    import traceback
                    logger.error("[plot-encoding] plot helper raised — full traceback below")
                    traceback.print_exc()

        runner.run_play = run_play_with_recording

    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()

    logger.info("Done!")
