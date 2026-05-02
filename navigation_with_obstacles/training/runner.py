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
import matplotlib
matplotlib.use("Agg")  # headless backend, mandatory for SLURM
import matplotlib.pyplot as plt

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
from navigation_with_obstacles.networks.popsan import PopSANNetworkBuilder
from rl_games.algos_torch.model_builder import register_network

# =============================================================================
# PopSAN Encoder Observer
# =============================================================================

OBS_NAMES = [
    "log(d_hor+1)", "log(d_vert+1)", "cos(azimuth)", "sin(azimuth)",
    "elevation", "cos(yaw)", "sin(yaw)", "v_hor", "v_vert",
    "cos(track_az)", "sin(track_az)", "track_elev",
] + [f"vae_{i:02d}" for i in range(32)]


class PopSANAlgoObserver(IsaacAlgoObserver):
    """Extends IsaacAlgoObserver to log PopSAN encoder μ and σ statistics."""

    def __init__(self):
        super().__init__()
        self._fig_log_every = 10  # render heatmap figure every N iters

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)

        try:
            encoder = self.algo.model.a2c_network.snn_actor.pop_encoder
        except AttributeError:
            return

        # means: [1, obs_dim, pop_dim] → [obs_dim, pop_dim]
        means = encoder.means.data.squeeze(0)
        stds  = encoder.stds.data.squeeze(0).abs()

        # --- Scalars ---
        self.writer.add_scalar("popsan_encoder/means_mean", means.mean().item(), epoch_num)
        self.writer.add_scalar("popsan_encoder/means_std",  means.std().item(),  epoch_num)
        self.writer.add_scalar("popsan_encoder/stds_mean",  stds.mean().item(),  epoch_num)
        self.writer.add_scalar("popsan_encoder/stds_min",   stds.min().item(),   epoch_num)
        self.writer.add_scalar("popsan_encoder/stds_max",   stds.max().item(),   epoch_num)

        self.writer.add_scalar("popsan_encoder/state_stds_mean", stds[:12].mean().item(), epoch_num)
        self.writer.add_scalar("popsan_encoder/vae_stds_mean",   stds[12:].mean().item(), epoch_num)

        # Fraction of neurons that would fire at least once (A_E >= 1/num_steps)
        # evaluated at the centre of each dimension's obs_bounds
        num_steps = self.algo.model.a2c_network.snn_actor.num_steps
        min_A_E   = 1.0 / num_steps
        obs_mid   = encoder.obs_bounds.mean(dim=1).unsqueeze(1).expand_as(means)
        A_E_mid   = torch.exp(-0.5 * ((obs_mid - means) / stds.clamp(min=1e-3)) ** 2)
        self.writer.add_scalar("popsan_encoder/frac_firing_at_bound_center",
                               (A_E_mid >= min_A_E).float().mean().item(), epoch_num)

        # --- Histograms ---
        # TensorBoard: visible in DISTRIBUTIONS tab as overlaid density plots over time
        # W&B: sync_tensorboard captures these but renders as percentile bands, not full histograms.
        #      The wandb.log() calls below produce proper W&B histogram panels.
        self.writer.add_histogram("popsan_encoder/all_means",        means.flatten(),    epoch_num)
        self.writer.add_histogram("popsan_encoder/all_stds",         stds.flatten(),     epoch_num)
        self.writer.add_histogram("popsan_encoder/state_means",      means[:12].flatten(), epoch_num)
        self.writer.add_histogram("popsan_encoder/state_stds",       stds[:12].flatten(),  epoch_num)
        self.writer.add_histogram("popsan_encoder/vae_means",        means[12:].flatten(), epoch_num)
        self.writer.add_histogram("popsan_encoder/vae_stds",         stds[12:].flatten(),  epoch_num)
        self.writer.add_histogram("popsan_encoder/per_dim_stds_mean", stds.mean(dim=1),   epoch_num)

        # W&B native histograms — proper distribution panels, not percentile bands
        if wandb.run is not None:
            wandb.log({
                "popsan_encoder/all_means":         wandb.Histogram(means.flatten().cpu().numpy()),
                "popsan_encoder/all_stds":          wandb.Histogram(stds.flatten().cpu().numpy()),
                "popsan_encoder/state_means":       wandb.Histogram(means[:12].flatten().cpu().numpy()),
                "popsan_encoder/state_stds":        wandb.Histogram(stds[:12].flatten().cpu().numpy()),
                "popsan_encoder/vae_means":         wandb.Histogram(means[12:].flatten().cpu().numpy()),
                "popsan_encoder/vae_stds":          wandb.Histogram(stds[12:].flatten().cpu().numpy()),
                "popsan_encoder/per_dim_stds_mean": wandb.Histogram(stds.mean(dim=1).cpu().numpy()),
            }, step=epoch_num, commit=False)

        # --- Encoder diagnostics figure (every N iters) ---
        if epoch_num % self._fig_log_every == 0:
            fig = self._plot_encoder_diagnostics(encoder, means, stds, num_steps)
            if fig is not None:
                self.writer.add_figure("popsan_encoder/diagnostics", fig, epoch_num)
                if wandb.run is not None:
                    wandb.log({"popsan_encoder/diagnostics": wandb.Image(fig)},
                              step=epoch_num, commit=False)
                plt.close(fig)

    def _plot_encoder_diagnostics(self, encoder, means, stds, num_steps):
        """Render a 3-panel heatmap: receptive field coverage, observed obs density,
        and per-neuron firing counts. Returns matplotlib figure or None."""
        obs_dim, pop_dim = means.shape
        x_lo, x_hi, n_bins = -3.5, 3.5, 200
        x_grid = torch.linspace(x_lo, x_hi, n_bins, device=means.device)  # [n_bins]

        # Panel A — receptive field coverage [obs_dim, n_bins]
        # coverage[d, x] = sum_k exp(-0.5 * ((x - mu[d,k]) / sigma[d,k])^2)
        x_exp = x_grid.view(1, 1, n_bins)                 # [1, 1, n_bins]
        mu_exp = means.unsqueeze(2)                        # [obs_dim, pop_dim, 1]
        sg_exp = stds.clamp(min=1e-3).unsqueeze(2)         # [obs_dim, pop_dim, 1]
        coverage = torch.exp(-0.5 * ((x_exp - mu_exp) / sg_exp) ** 2).sum(dim=1)  # [obs_dim, n_bins]
        coverage_np = coverage.cpu().numpy()

        # Panel B & C: only if a training batch has been captured
        last_obs = encoder._last_obs
        last_act = encoder._last_pop_activity
        has_activity = last_obs is not None and last_act is not None

        density_np = None
        firing_np = None
        if has_activity:
            # Panel B — observed obs density [obs_dim, n_bins]
            edges = torch.linspace(x_lo, x_hi, n_bins + 1, device=last_obs.device)
            density = torch.zeros(obs_dim, n_bins, device=last_obs.device)
            for d in range(obs_dim):
                hist = torch.histc(last_obs[:, d], bins=n_bins, min=x_lo, max=x_hi)
                density[d] = hist
            row_max = density.max(dim=1, keepdim=True).values.clamp(min=1.0)
            density_np = (density / row_max).cpu().numpy()

            # Panel C — per-neuron firing counts [obs_dim, pop_dim]
            min_A_E = 1.0 / num_steps
            firing = (last_act >= min_A_E).sum(dim=0).float()  # [obs_dim, pop_dim]
            row_max = firing.max(dim=1, keepdim=True).values.clamp(min=1.0)
            firing_np = (firing / row_max).cpu().numpy()

        # --- Render ---
        n_panels = 3 if has_activity else 1
        fig, axes = plt.subplots(
            1, n_panels, figsize=(5 * n_panels, 10),
            gridspec_kw={"width_ratios": [4, 4, 1][:n_panels]},
        )
        if n_panels == 1:
            axes = [axes]

        # Panel A: coverage
        im = axes[0].imshow(coverage_np, aspect="auto", origin="lower",
                            extent=[x_lo, x_hi, -0.5, obs_dim - 0.5],
                            cmap="viridis", interpolation="nearest")
        axes[0].set_title("Receptive field coverage\n(sum of Gaussian responses)")
        axes[0].set_xlabel("input value (post-normalization)")
        axes[0].set_yticks(range(obs_dim))
        axes[0].set_yticklabels(OBS_NAMES, fontsize=6)
        axes[0].axvline(-3, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        axes[0].axvline(+3, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
        plt.colorbar(im, ax=axes[0], fraction=0.04)

        if has_activity:
            # Panel B: observed density
            im = axes[1].imshow(density_np, aspect="auto", origin="lower",
                                extent=[x_lo, x_hi, -0.5, obs_dim - 0.5],
                                cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
            axes[1].set_title("Observed obs density\n(last batch, row-normalized)")
            axes[1].set_xlabel("input value")
            axes[1].set_yticks([])
            axes[1].axvline(-3, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
            axes[1].axvline(+3, color="white", linestyle="--", linewidth=0.5, alpha=0.5)
            plt.colorbar(im, ax=axes[1], fraction=0.04)

            # Panel C: per-neuron firing
            im = axes[2].imshow(firing_np, aspect="auto", origin="lower",
                                cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
            axes[2].set_title("Per-neuron\nfiring (row-norm)")
            axes[2].set_xlabel("neuron idx")
            axes[2].set_yticks([])
            axes[2].set_xticks(range(pop_dim))
            plt.colorbar(im, ax=axes[2], fraction=0.1)

        fig.tight_layout()
        return fig


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
register_network("PopSAN", PopSANNetworkBuilder)


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

    def render(self, mode="human"):
        """No-op render — Isaac Gym handles its own viewer."""
        pass

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
        {"name": "--num_envs", "type": int, "default": -1, "help": "Num envs (overrides YAML; -1 = use YAML value)"},
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

    if args["num_envs"] > 0:
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

    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        config = update_config(config, args)
        # Ensure rl_games saves tensorboard logs and checkpoints under
        # navigation_with_obstacles/runs/
        config["params"]["config"]["train_dir"] = runs_dir

        observer = PopSANAlgoObserver() if not args.get("play") else IsaacAlgoObserver()
        runner = Runner(algo_observer=observer)
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
    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()

    logger.info("Done!")
