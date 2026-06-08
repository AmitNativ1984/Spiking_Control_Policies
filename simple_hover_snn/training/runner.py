"""
Custom runner for Hover SNN task using RL Games framework.

The script registers the hover SNN task and environment with aerial_gym and rl_games,
then runs PPO training.

This is identical to simple_hover runner, but with SNN-specific registrations.
The actual SNN network architecture will be added in a future implementation.
"""

import isaacgym
import argparse
import os
import sys
import yaml
import shutil
import numpy as np
from datetime import datetime

from simple_hover_snn.networks.popsan_network import POPSANNetworkBuilder
sys.path.insert(0, "/workspaces/aerial_gym_docker")
import wandb
import torch
import gym
from gym import spaces
from loguru import logger
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from rl_games.common import a2c_common

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor
from aerial_gym.utils.helpers import parse_arguments
from distutils.util import strtobool

from simple_hover_snn.task.simple_hover_snn_task import HoverSNNTask
from simple_hover_snn.config.task_config import task_config
from simple_hover_snn.config.env_config import SimpleHoverSNNEnvCfg
from simple_hover_snn.config.robot_config import CustomQuadWithImuSNNCfg

from rl_games.algos_torch import model_builder
from simple_hover_snn.networks import SNNNetworkBuilder, MLPNetworkBuilder
from simple_hover_snn.networks.weight_transfer import transfer_mlp_to_snn_weights, MLP_TO_SNN_MAPPING

# =============================================================================
# Register Custom Environment and Task
# =============================================================================


# Register environment config with aerial_gym
env_config_registry.register("simple_hover_snn_env",
                             SimpleHoverSNNEnvCfg)

# Register robot config
robot_registry.register("custom_quad_with_imu_snn",
                        BaseMultirotor,
                        CustomQuadWithImuSNNCfg)
# Register task
task_registry.register_task("simple_hover_snn_task",
                       HoverSNNTask,
                       task_config)

# Register both network types (choose via YAML config)
model_builder.register_network('snn_actor_critic', SNNNetworkBuilder)
model_builder.register_network('mlp_actor_critic', MLPNetworkBuilder)
model_builder.register_network('PopSAN', POPSANNetworkBuilder)

# =============================================================================
# Monkey Patch: Simplify Checkpoint Naming (Remove Reward Suffix)
# =============================================================================

# Store the original train_epoch method
_original_train_epoch = a2c_common.DiscreteA2CBase.train_epoch if hasattr(a2c_common, 'DiscreteA2CBase') else None
_original_continuous_train_epoch = a2c_common.ContinuousA2CBase.train_epoch if hasattr(a2c_common, 'ContinuousA2CBase') else None

def _patch_checkpoint_naming():
    """
    Monkey patch to simplify checkpoint naming by removing reward suffix.
    Original: {name}_ep_{epoch}_rew_{reward}.pth
    New: {name}_ep_{epoch}.pth
    """
    import types

    # Patch the checkpoint name generation in the train method
    original_train = a2c_common.A2CBase.train

    def patched_train(self):
        # Store original save method
        original_save_method = self.save

        def patched_save(filename):
            # Simplify checkpoint name by removing reward suffix
            # Replace pattern: {name}_ep_{num}_rew_{reward}.pth with {name}_ep_{num}.pth
            # Match _rew_ followed by integer or float (with optional negative sign)
            import re
            simplified_filename = re.sub(r'_rew_-?\d+(?:\.\d+)?', '', filename)
            return original_save_method(simplified_filename)

        # Temporarily replace save method
        self.save = patched_save

        try:
            # Call original train
            result = original_train(self)
        finally:
            # Restore original save method
            self.save = original_save_method

        return result

    a2c_common.A2CBase.train = patched_train

# Apply the patch
_patch_checkpoint_naming()


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
            torch.zeros_like(terminated)
        )

        return observations["observations"], rewards, dones, infos

class AERIALRLGPUEnv(vecenv.IVecEnv):
    """Vectorized environment wrapper for rl_games."""

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
             np.ones(self.env.task_config.action_space_dim)
        )
        info["observation_space"] = spaces.Box(
            -np.ones(self.env.task_config.observation_space_dim) * np.inf,
             np.ones(self.env.task_config.observation_space_dim) * np.inf
        )
        return info


# Custom env creator that handles reward_params separately
def create_hover_task(**kwargs):
    """Create task, extracting reward_params to pass directly to HoverSNNTask."""
    reward_params = kwargs.pop("reward_params", None)
    success_rate_window_episodes = kwargs.pop("success_rate_window_episodes", None)
    task = task_registry.make_task("simple_hover_snn_task", **kwargs)
    # If reward_params provided, override the task's reward parameters
    if reward_params is not None:
        for key, value in reward_params.items():
            if key in task.task_config.reward_parameters:
                task.task_config.reward_parameters[key] = torch.tensor([value], device=task.device)
                logger.info(f"Reward param override: {key} = {value}")
    # Override the windowed success-rate buffer size if provided via YAML
    if success_rate_window_episodes is not None:
        task.set_success_rate_window(int(success_rate_window_episodes))
        logger.info(f"Success rate window episodes = {success_rate_window_episodes}")
    return task

# Register task with rl_games env_configurations
env_configurations.register(
    "simple_hover_snn_task",
    {
        "env_creator": create_hover_task,
        "vecenv_type": "AERIAL-RLGPU-SNN",
    },
)

# Register vectorized environment type
vecenv.register(
    "AERIAL-RLGPU-SNN",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs)
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
        {"name": "--num_envs", "type": int, "default": 1024, "help": "Num envs"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": "False", "help": "Headless mode"},
        {"name": "--use_warp", "type": lambda x: bool(strtobool(x)), "default": "False", "help": "Use warp"},
        {"name": "--experiment_name", "type": str, "default": None, "help": "Experiment name (defaults to config name)"},
        {"name": "--task", "type": str, "default": "simple_hover_snn_task", "help": "Task name"},
        {"name": "--track", "action": "store_true", "help": "Track with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "aerial_gym_snn", "help": "Wandb project name"},
        {"name": "--wandb-entity", "type": str, "default": None, "help": "Wandb entity (team)"},
        {"name": "--mlp_checkpoint", "type": str, "default": None, "help": "Path to MLP checkpoint to transfer weights from (for SNN training)"},
        {"name": "--plot-encoding", "action": "store_true", "help": "Debug: when combined with --play, record PopSAN encoder activations and plot at end (encoder_trace_*.png). Forces num_envs=1."},
    ]
    args = parse_arguments(description="Simple Hover SNN Task", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args

def update_config(config, args):
    """Update training config with CLI arguments."""
    config["params"]["config"]["env_name"] = args["task"]
    # Use experiment name from config if not provided via CLI
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]

    # Build the run folder name with a FULL timestamp (year-month-day_hour-min-sec).
    # rl_games' default only uses "_%d-%H-%M-%S" (no year/month), so runs can
    # collide across months. Setting full_experiment_name overrides rl_games'
    # naming entirely (see a2c_common.py: full_experiment_name short-circuits the
    # default timestamped name). The run dir becomes runs/<full_experiment_name>.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["params"]["config"]["full_experiment_name"] = (
        f"{config['params']['config']['name']}_{timestamp}"
    )

    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]

    # Pass reward parameters from YAML to env_config (will be picked up by task)
    if "reward_params" in config["params"]["config"]:
        config["params"]["config"]["env_config"]["reward_params"] = config["params"]["config"]["reward_params"]

    # Pass windowed success-rate window size from YAML to env_config (picked up by task)
    if "success_rate_window_episodes" in config["params"]["config"]:
        config["params"]["config"]["env_config"]["success_rate_window_episodes"] = \
            config["params"]["config"]["success_rate_window_episodes"]

    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]

    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Create output directories
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Parse arguments
    args = vars(get_args())
    config_name = args["file"]

    # Debug-only: --play --plot-encoding records the PopSAN encoder during a
    # single-env rollout and plots Gaussian receptive fields + spike rasters at
    # the end (encoder_trace_*.png). Force num_envs=1 BEFORE update_config writes
    # it into the config, so we get one clean trajectory.
    plot_encoding = bool(args.get("play") and args.get("plot_encoding"))
    if plot_encoding:
        logger.warning("--plot-encoding set: forcing num_envs=1 for clean single-trajectory plots")
        args["num_envs"] = 1

    logger.info(f"Loading config: {config_name}")
    logger.info(f"Number of environments: {args['num_envs']}")
    logger.info(f"Headless: {args['headless']}")
    logger.info(f"Use warp: {args['use_warp']}")

    # Load and update config
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        config = update_config(config, args)

        # Initialize rl_games runner with IsaacAlgoObserver for custom metric logging
        runner = Runner(algo_observer=IsaacAlgoObserver())
        try:
            runner.load(config)
        except yaml.YAMLError as exc:
            logger.error(f"Error loading config: {exc}")
            sys.exit(1)

    # Initialize wandb tracking if enabled
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

    # Run training or playing
    logger.info("Starting training..." if args.get("train") else "Starting playback...")

    # Handle MLP checkpoint transfer to SNN
    mlp_checkpoint_path = args.get("mlp_checkpoint")
    if mlp_checkpoint_path and args.get("train"):
        # Verify we're training SNN (not MLP)
        network_name = config["params"]["network"]["name"]
        if network_name != "snn_actor_critic":
            logger.warning(f"--mlp_checkpoint provided but network is '{network_name}', not 'snn_actor_critic'. Ignoring.")
            mlp_checkpoint_path = None
        elif not os.path.exists(mlp_checkpoint_path):
            logger.error(f"MLP checkpoint not found: {mlp_checkpoint_path}")
            sys.exit(1)
        else:
            logger.info(f"Will transfer weights from MLP checkpoint: {mlp_checkpoint_path}")

    # Patch to save config when experiment directory is created
    if args.get("train"):
        import glob
        from rl_games.common.a2c_common import A2CBase
        original_init_tensors = A2CBase.init_tensors

        def patched_init_tensors(self):
            result = original_init_tensors(self)
            # Save config to experiment directory
            config_save_path = os.path.join(self.experiment_dir, "config.yaml")
            with open(config_save_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to: {config_save_path}")
            return result

        A2CBase.init_tensors = patched_init_tensors

    # Patch to transfer MLP weights to SNN after model creation
    if mlp_checkpoint_path:
        from rl_games.algos_torch.a2c_continuous import A2CAgent
        original_init_model = A2CAgent.init_tensors

        def patched_init_model_with_transfer(self):
            result = original_init_model(self)

            # Load MLP checkpoint
            logger.info(f"Loading MLP checkpoint: {mlp_checkpoint_path}")
            mlp_checkpoint = torch.load(mlp_checkpoint_path, map_location=self.device)

            # Extract model state dict from checkpoint
            if 'model' in mlp_checkpoint:
                mlp_state_dict = mlp_checkpoint['model']
            else:
                mlp_state_dict = mlp_checkpoint

            # Get current SNN model state dict
            snn_state_dict = self.model.state_dict()

            # Transfer weights
            logger.info("Transferring MLP weights to SNN...")
            updated_state_dict = transfer_mlp_to_snn_weights(
                mlp_state_dict, snn_state_dict, verbose=True
            )

            # Load updated weights into model
            self.model.load_state_dict(updated_state_dict)
            logger.info("MLP to SNN weight transfer complete!")

            return result

        A2CAgent.init_tensors = patched_init_model_with_transfer

    if plot_encoding:
        # Wrap run_play: enable recording on the encoder right after player
        # construction, plot once the play loop returns. Mirrors the
        # navigation_with_obstacles runner. The PopSAN network here is
        # POPSANNetwork with the encoder directly at `.pop_encoder` (no
        # `.snn_actor` wrapper), so the attribute path differs from navigation.
        orig_run_play = runner.run_play

        def run_play_with_recording(args_):
            logger.info("Started to play (with encoder recording)")
            player = runner.create_player()
            from rl_games.torch_runner import _restore, _override_sigma
            _restore(player, args_)
            _override_sigma(player, args_)

            # rl_games wraps the raw network in a ModelA2C*.Network at
            # player.model, exposing it as `.a2c_network` -> POPSANNetwork.
            encoder = player.model.a2c_network.pop_encoder
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
                    # Save under the same runs/ directory the simple_hover_snn
                    # experiments use (the runner's train_dir), not the plot
                    # helper's default (navigation_with_obstacles/runs/).
                    runs_dir = os.path.abspath("runs")
                    plot_encoder_trace(
                        encoder,
                        encoder._trace,
                        task_config.observation_layout,
                        save_dir=runs_dir,
                    )
                except Exception:
                    import traceback
                    logger.error("[plot-encoding] plot helper raised — full traceback below")
                    traceback.print_exc()

        runner.run_play = run_play_with_recording

    runner.run(args)

    # Finish wandb tracking
    if args["track"] and rank == 0:
        wandb.finish()

    logger.info("Done!")
