"""rl_games training/playback entry point for the F450 navigation task.

Usage:
    cd /workspaces/aerial_gym_docker
    python -m rl_training.rl_games.runner \
        --file=rl_training/rl_games/cfg/ppo_f450_navigation.yaml --train

Everything this repo adds is registered by import side effect:
  `import config`   -> the env, robot and task in aerial_gym's registries
  `import ...networks` -> the custom rl_games network builders
so this file only parses arguments and wires the config together.
"""
import isaacgym  # noqa: F401  — MUST precede torch; see aerial_gym/__init__.py

import logging
import os
import sys
from datetime import datetime
from distutils.util import strtobool

import yaml
from loguru import logger
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from aerial_gym.utils.helpers import parse_arguments

import config  # noqa: F401  — registers the env, robot and task
from config.task_config import F450NavTaskConfig as task_config

from . import RUNS_DIR
from .agents import register_algos
from .networks import bind_encoder_bounds
from .vecenv import register_task


# aerial_gym's asset_manager logs one line per asset per env; at 256 envs that buries
# everything else. It uses stdlib logging (aerial_gym.utils.logging.CustomLogger), not loguru.
logging.getLogger("asset_manager").setLevel(logging.ERROR)


# =============================================================================
# Argument parsing
# =============================================================================


def get_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--train", "action": "store_true", "help": "Train network"},
        {"name": "--play", "action": "store_true", "help": "Play/test network"},
        {"name": "--checkpoint", "type": str, "help": "Path to checkpoint"},
        {"name": "--file", "type": str, "default": None, "help": "Path to config YAML"},
        {"name": "--num_envs", "type": int, "default": None,
         "help": "Num envs (overrides the YAML when set)"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": "False",
         "help": "Headless mode"},
        {"name": "--use_warp", "type": lambda x: bool(strtobool(x)), "default": "True",
         "help": "Use warp"},
        {"name": "--experiment_name", "type": str, "default": "f450_navigation",
         "help": "Experiment name"},
        {"name": "--task", "type": str, "required": True, "help": "Task name"},
        {"name": "--track", "action": "store_true",
         "help": "Track with Weights and Biases"},
        {"name": "--wandb-project-name", "type": str, "default": "aerial_gym",
         "help": "Wandb project name"},
        {"name": "--wandb-entity", "type": str, "default": None,
         "help": "Wandb entity (team)"},
        {"name": "--curriculum_level", "type": int, "default": None,
         "help": "Fix curriculum (obstacle density) at this level (0-25). Overrides min/max."},
        {"name": "--exceed_margin", "type": float, "default": None,
         "help": "Out-of-bounds margin multiplier (e.g. 1.5 = 50%% beyond bounds "
                 "before termination)"},
        {"name": "--plot-encoding", "action": "store_true",
         "help": "Debug: with --play, record PopSAN encoder activations and plot at the "
                 "end. Forces num_envs=1."},
        {"name": "--recompute_bounds", "action": "store_true",
         "help": "Force re-collection of PopSAN encoder observation_bounds even if a "
                 "cache exists (student/PopSAN --train runs only)."},
        {"name": "--bounds_steps", "type": int, "default": 10000,
         "help": "Steps to collect when auto-computing PopSAN encoder observation_bounds."},
    ]
    args = parse_arguments(
        description="F450 navigation with obstacles", custom_parameters=custom_parameters
    )
    # aerial_gym's parse_arguments only forwards name/type/default/action/help to argparse,
    # so `required` would be silently ignored — check it here instead.
    if not args.file:
        sys.exit("--file is required: path to a config YAML under "
                 "rl_training/rl_games/cfg/")

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


# =============================================================================
# Config assembly
# =============================================================================


def apply_args(config_dict, args):
    """Fold command-line overrides into the parsed YAML."""
    cfg = config_dict["params"]["config"]
    cfg["env_name"] = args["task"]
    cfg["name"] = args["experiment_name"]
    cfg["env_config"]["headless"] = args["headless"]
    cfg["env_config"]["use_warp"] = args["use_warp"]

    # Environment count has ONE source of truth: --num_envs if given, else the YAML's
    # num_actors. rl_games sizes its rollout buffers from num_actors while our vecenv
    # builds the simulator from env_config.num_envs — the two are the same number, and
    # letting a YAML state both invites them to drift (the failure is a tensor-shape
    # RuntimeError raised only after the sim has finished building).
    num_envs = args.get("num_envs") or cfg["num_actors"]
    cfg["num_actors"] = num_envs
    cfg["env_config"]["num_envs"] = num_envs

    # Clamp minibatch_size to batch_size so rl_games' assertion passes.
    cfg["minibatch_size"] = min(cfg["minibatch_size"], num_envs * cfg["horizon_length"])

    if args["seed"] > 0:
        config_dict["params"]["seed"] = args["seed"]
        cfg["env_config"]["seed"] = args["seed"]

    # Task-config overrides. These reach the task through the config class itself, which
    # the task reads at construction — so they must be set before the env is built.
    if args.get("curriculum_level") is not None:
        task_config.curriculum.min_level = args["curriculum_level"]
        task_config.curriculum.max_level = args["curriculum_level"]
    if args.get("exceed_margin") is not None:
        task_config.exceed_bounds_margin = args["exceed_margin"]

    if args.get("checkpoint"):
        config_dict["params"]["load_checkpoint"] = True
        config_dict["params"]["load_path"] = args["checkpoint"]

    # Merge, don't overwrite: the YAML's own player settings must survive.
    cfg.setdefault("player", {})["use_vecenv"] = True

    return config_dict


def resolve_encoder_bounds(config_dict, args):
    """For a PopSAN student with a teacher, replace the task's default encoder bounds with
    ones measured in the teacher's NORMALIZED observation space.

    The population encoder builds its Gaussian receptive fields from these bounds at
    construction, so they have to be settled before rl_games builds the model. Collection
    runs in a subprocess (Isaac Gym allows one sim per process) and is cached.
    """
    if not (args.get("train") and config_dict["params"]["network"]["name"] == "popsan"):
        return

    distill_cfg = config_dict["params"]["config"].get("distillation")
    if distill_cfg is None:
        return

    teacher_ckpt = distill_cfg.get("checkpoint")
    if not (teacher_ckpt and os.path.exists(teacher_ckpt)):
        logger.warning(f"[obs-bounds] distillation.checkpoint missing/not found "
                       f"({teacher_ckpt!r}); using the task config's default bounds.")
        return

    from .tools.collect_obs_stats import load_or_collect_bounds

    bounds = load_or_collect_bounds(
        teacher_checkpoint=teacher_ckpt,
        config_path=args["file"],
        num_envs=min(config_dict["params"]["config"]["env_config"]["num_envs"], 64),
        num_steps=args["bounds_steps"],
        recompute=args["recompute_bounds"],
    )
    config_dict["params"]["network"]["actor"]["observation_bounds"] = bounds
    logger.info(f"[obs-bounds] encoder bounds set from collected stats ({len(bounds)} dims).")


def load_config(args):
    """Parse the YAML and produce the config rl_games will run."""
    with open(args["file"]) as f:
        config_dict = yaml.safe_load(f)

    config_dict = apply_args(config_dict, args)

    # Run artifacts land in runs/<experiment>/<timestamp>/{nn,summaries}.
    #
    # rl_games builds its output path as train_dir/full_experiment_name (a2c_common.py:267),
    # with no notion of a per-experiment grouping level. Folding the experiment name into
    # train_dir and leaving only the timestamp as full_experiment_name gets the nesting we
    # want without patching rl_games: every run of one experiment collects under a single
    # folder instead of scattering across a flat runs/ directory that mixes experiments.
    #
    # rl_games' own default name uses "_%d-%H-%M-%S" (no year or month), which makes run
    # folders ambiguous across months; the full date+time stamp below fixes that too.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = config_dict["params"]["config"]["name"]
    config_dict["params"]["config"]["train_dir"] = os.path.join(RUNS_DIR, experiment_name)
    config_dict["params"]["config"]["full_experiment_name"] = timestamp

    # Measured bounds win; bind_encoder_bounds fills in the layout-derived defaults for
    # any run that didn't collect them (setdefault, so it never overwrites the above).
    resolve_encoder_bounds(config_dict, args)
    bind_encoder_bounds(config_dict, task_config)

    return config_dict


# =============================================================================
# --play --plot-encoding
# =============================================================================


def install_encoder_recording(runner):
    """Wrap runner.run_play so the PopSAN encoder records its activations during playback
    and plots them once the play loop returns. Debug-only.

    Takes no args: rl_games passes them to run_play itself, as `play_args` below.
    """
    def run_play_with_recording(play_args):
        from rl_games.torch_runner import _override_sigma, _restore

        player = runner.create_player()
        _restore(player, play_args)
        _override_sigma(player, play_args)

        # rl_games wraps the raw network in a ModelA2C*.Network at player.model, which
        # exposes the underlying network as `.a2c_network`. For PopSAN that's
        # POPSANNetwork -> .spiking_actor -> .pop_encoder.
        encoder = player.model.a2c_network.spiking_actor.pop_encoder
        encoder.record = True
        encoder._trace = []
        logger.info(f"[plot-encoding] recording enabled on {type(encoder).__name__}")

        try:
            player.run()
        except KeyboardInterrupt:
            logger.info("[plot-encoding] interrupted by user — proceeding to plot")
        finally:
            encoder.record = False
            logger.info(f"[plot-encoding] recorded {len(encoder._trace)} forward passes")
            _plot_recorded_trace(encoder, play_args.get("checkpoint"))

    runner.run_play = run_play_with_recording


def _plot_recorded_trace(encoder, checkpoint):
    """Save the encoder plots next to the checkpoint's run directory (weights live in
    <run_dir>/nn/, so the run dir is two levels up). Best-effort."""
    try:
        from .tools.plot_encoder_trace import plot_encoder_trace

        save_dir = (
            os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
            if checkpoint else None
        )
        plot_encoder_trace(encoder, encoder._trace,
                           task_config.observation_layout, save_dir=save_dir)
    except Exception:
        import traceback
        logger.error("[plot-encoding] plot helper raised — full traceback below")
        traceback.print_exc()


# =============================================================================
# Main
# =============================================================================


def main():
    args = vars(get_args())

    # Debug-only: --play --plot-encoding records the encoder during a single-env rollout.
    plot_encoding = bool(args.get("play") and args.get("plot_encoding"))
    if plot_encoding:
        logger.warning("--plot-encoding set: forcing num_envs=1 for clean single-"
                       "trajectory plots")
        args["num_envs"] = 1

    logger.info(f"Loading config: {args['file']}")
    logger.info(f"num_envs={args['num_envs']} headless={args['headless']} "
                f"use_warp={args['use_warp']}")

    # Must match the env_name apply_args writes into the config, or rl_games looks up a
    # vecenv that was never registered.
    register_task(args["task"])

    try:
        config_dict = load_config(args)
    except yaml.YAMLError as exc:
        logger.error(f"Error loading config: {exc}")
        sys.exit(1)

    runner = Runner(algo_observer=IsaacAlgoObserver())
    register_algos(runner)
    runner.load(config_dict)

    # W&B is rank-0 only; other ranks train without tracking.
    tracking = args["track"] and int(os.getenv("LOCAL_RANK", "0")) == 0
    info = _start_wandb(config_dict, args) if tracking else None

    logger.info("Starting training..." if args.get("train") else "Starting playback...")
    if plot_encoding:
        install_encoder_recording(runner)
    runner.run(args)

    if tracking:
        _finish_wandb(config_dict, args, info)

    logger.info("Done!")


def _start_wandb(config_dict, args):
    import wandb

    from .wandb_utils import enable_checkpoint_upload, git_info

    info = git_info()
    logger.info(f"[wandb] git commit {info['git_commit_short']} "
                f"(#{info['git_commit_number']}, branch {info['git_branch']}, "
                f"dirty={info['git_dirty']})")

    # Record git provenance in the run config so the run can always be restored to the
    # exact code state that produced it.
    #
    # full_experiment_name is only the timestamp (the experiment name lives in train_dir --
    # see load_config), so recombine them here: W&B has no folder nesting and a run called
    # "2026-08-11_10-52-30" would be unidentifiable in a project listing.
    cfg = config_dict["params"]["config"]
    wandb.init(
        project=args["wandb_project_name"],
        entity=args["wandb_entity"],
        name=f"{cfg['name']}_{cfg['full_experiment_name']}",
        sync_tensorboard=True,
        config={**config_dict, "git": info},
        monitor_gym=True,
        save_code=True,
    )
    enable_checkpoint_upload()
    return info


def _finish_wandb(config_dict, args, info):
    import wandb

    from .wandb_utils import log_final_weights

    if args.get("train"):
        cfg = config_dict["params"]["config"]
        # train_dir already carries the experiment name (runs/<experiment>), so joining it
        # with full_experiment_name (the timestamp) reproduces rl_games' own experiment_dir.
        log_final_weights(cfg["train_dir"], cfg["full_experiment_name"], cfg["name"], info)
    wandb.finish()


if __name__ == "__main__":
    main()
