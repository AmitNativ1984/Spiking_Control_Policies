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
from datetime import datetime
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
        {
            "name": "--recompute_bounds",
            "action": "store_true",
            "help": "Force re-collection of PopSAN encoder observation_bounds even if a cache exists (student/PopSAN --train runs only).",
        },
        {
            "name": "--bounds_steps",
            "type": int,
            "default": 10000,
            "help": "Steps to collect when auto-computing PopSAN encoder observation_bounds.",
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


def _auto_set_observation_bounds(teacher_ckpt, config_path, num_envs, num_steps, recompute,
                                 min_episodes=0, out_dir=None, curriculum_level=25,
                                 bound_method="gaussian"):
    """Set task_config.observation_bounds for the PopSAN encoder from collected
    p01/p99 stats. Runs the collector in a SEPARATE subprocess (Isaac Gym allows
    only one sim per process), which writes a JSON cache; this loads the cache.

    `config_path` is the student YAML, forwarded to the collector so it reads the
    teacher network architecture from config.distillation (single source of truth).

    Reuses an existing cache only if it matches BOTH obs_dim AND the current teacher
    checkpoint (a cache built with a different teacher is stale). `recompute` forces a
    fresh collection. When `min_episodes`/`out_dir` are given they are forwarded to the
    collector (episode-based stop + where the bounds-encoder PNGs are written).
    `curriculum_level` pins the collector's env to the teacher's level (default 25) and
    VAE-on state, so bounds reflect the world the student is actually deployed in.
    """
    import json
    import subprocess
    from navigation_with_obstacles.tools.collect_obs_stats import DEFAULT_BOUNDS_CACHE

    obs_dim = task_config.observation_space_dim
    cache = DEFAULT_BOUNDS_CACHE

    def _load_valid_cache():
        if not os.path.exists(cache):
            return None
        try:
            with open(cache) as f:
                payload = json.load(f)
        except Exception as e:
            logger.warning(f"[obs-bounds] cache unreadable ({e}); will recompute.")
            return None
        if payload.get("obs_dim") != obs_dim or \
           len(payload.get("observation_bounds", [])) != obs_dim:
            logger.warning(f"[obs-bounds] cache obs_dim mismatch "
                           f"({payload.get('obs_dim')} != {obs_dim}); will recompute.")
            return None
        if payload.get("teacher_checkpoint") != teacher_ckpt:
            logger.warning(f"[obs-bounds] cache was built for a different teacher "
                           f"({payload.get('teacher_checkpoint')!r} != {teacher_ckpt!r}); "
                           "will recompute.")
            return None
        return [tuple(b) for b in payload["observation_bounds"]]

    bounds = None if recompute else _load_valid_cache()

    if bounds is None:
        logger.info(f"[obs-bounds] collecting bounds in a subprocess "
                    f"(teacher={teacher_ckpt}, steps={num_steps}, episodes={min_episodes}, "
                    f"envs={num_envs})")
        cmd = [
            sys.executable, "-m", "navigation_with_obstacles.tools.collect_obs_stats",
            f"--teacher_checkpoint={teacher_ckpt}",
            f"--config={config_path}",
            f"--num_steps={num_steps}",
            f"--num_envs={num_envs}",
            f"--bounds_cache={cache}",
            f"--curriculum_level={curriculum_level}",
            f"--bound_method={bound_method}",
            "--no_wandb",
        ]
        if min_episodes and min_episodes > 0:
            cmd.append(f"--min_episodes={min_episodes}")
        if out_dir:
            cmd.append(f"--out_dir={out_dir}")
        result = subprocess.run(cmd, cwd="/workspaces/aerial_gym_docker")
        if result.returncode != 0:
            raise RuntimeError(
                f"[obs-bounds] collection subprocess failed (exit {result.returncode}); "
                "fix the teacher checkpoint / collector before training the student.")
        bounds = _load_valid_cache()
        if bounds is None:
            raise RuntimeError("[obs-bounds] subprocess finished but produced no valid cache.")

    assert len(bounds) == obs_dim, \
        f"[obs-bounds] got {len(bounds)} bounds, expected {obs_dim}"
    task_config.observation_bounds = bounds
    logger.info(f"[obs-bounds] task_config.observation_bounds set from cache "
                f"({len(bounds)} dims).")


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

        # rl_games' default run name uses only "_%d-%H-%M-%S" (no year/month),
        # which makes run folders ambiguous across months. Override it with a
        # full date+time stamp: <name>_YYYY-MM-DD_HH-MM-SS
        experiment_name = config["params"]["config"]["name"]
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config["params"]["config"]["full_experiment_name"] = f"{experiment_name}_{timestamp}"

        # DEBUG: surface the runtime values rl_games will actually use
        logger.debug(f"[DEBUG] args['num_envs'] = {args.get('num_envs')!r} (type={type(args.get('num_envs')).__name__})")
        logger.debug(f"[DEBUG] config.num_actors            = {config['params']['config']['num_actors']}")
        logger.debug(f"[DEBUG] config.env_config.num_envs   = {config['params']['config']['env_config']['num_envs']}")
        logger.debug(f"[DEBUG] config.horizon_length        = {config['params']['config']['horizon_length']}")
        logger.debug(f"[DEBUG] config.minibatch_size        = {config['params']['config']['minibatch_size']}")
        logger.debug(f"[DEBUG] config.seq_length            = {config['params']['config'].get('seq_length')}")

        # --- Phase 2: auto-set PopSAN encoder observation_bounds ------------
        # For a student (PopSAN) training run with a configured teacher, set
        # task_config.observation_bounds from per-dim p01/p99 collected in the
        # teacher's NORMALIZED obs space, BEFORE the network is built (the encoder
        # reads observation_bounds at construction).
        #
        # Collection itself runs in a SEPARATE subprocess: Isaac Gym does not
        # support creating a second sim in a process that will create another, so
        # we never build the collector's env in the training process. The
        # subprocess writes a JSON cache; we load it here. Cache is reused unless
        # --recompute_bounds is passed.
        net_name = config["params"]["network"]["name"]
        distill_cfg = config["params"]["config"].get("distillation")
        if args.get("train") and net_name == "PopSAN" and distill_cfg is not None:
            teacher_ckpt = distill_cfg.get("checkpoint")
            if teacher_ckpt and os.path.exists(teacher_ckpt):
                _auto_set_observation_bounds(
                    teacher_ckpt=teacher_ckpt,
                    config_path=config_name,
                    num_envs=min(config["params"]["config"]["env_config"]["num_envs"], 64),
                    num_steps=args.get("bounds_steps", 10000),
                    recompute=args.get("recompute_bounds", False),
                )
            else:
                logger.warning(
                    f"[obs-bounds] distillation.checkpoint missing/not found "
                    f"({teacher_ckpt!r}); using task_config default bounds.")

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

    # Inference/play always runs with the VAE fully enabled, regardless of any curriculum
    # warm-up phase a checkpoint was saved in. The training-time gate (Phase A=0.0) is a
    # warm-up device only; the task's curriculum state machine drives it during --train.
    if not args.get("train"):
        task_config.vae_gate = 1.0
        logger.info("[VAE warm-up] play mode: forcing task_config.vae_gate = 1.0 (full VAE)")

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
                    # Save the plots in the run directory — the parent of the checkpoint's
                    # containing folder (weights live in <run_dir>/nn/, plots go in <run_dir>).
                    # Falls back to the package runs/ dir when no checkpoint was given.
                    ckpt = args_.get("checkpoint")
                    save_dir = (
                        os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
                        if ckpt else None
                    )
                    plot_encoder_trace(encoder, encoder._trace,
                                       task_config.observation_layout, save_dir=save_dir)
                except Exception:
                    import traceback
                    logger.error("[plot-encoding] plot helper raised — full traceback below")
                    traceback.print_exc()

        runner.run_play = run_play_with_recording

    runner.run(args)

    if args["track"] and rank == 0:
        wandb.finish()

    logger.info("Done!")
