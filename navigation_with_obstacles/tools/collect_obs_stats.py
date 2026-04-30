"""
Observation statistics collector for NavigationWithObstaclesTask.

Runs the environment with random actions for a fixed number of steps,
then saves per-dimension statistics to CSV and logs histograms to TensorBoard/W&B.

Usage:
    cd /workspaces/aerial_gym_docker
    python -m navigation_with_obstacles.tools.collect_obs_stats \
        --num_steps=10000 \
        --num_envs=64 \
        --out_dir=navigation_with_obstacles/obs_stats
"""
import isaacgym  # must be first

import argparse
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, "/workspaces/aerial_gym_docker")

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor

from navigation_with_obstacles.task.navigation_task import NavigationWithObstaclesTask
from navigation_with_obstacles.config.task_config import task_config
from navigation_with_obstacles.config.env_config import NavigationObstacleEnvCfg
from navigation_with_obstacles.config.robot_config import NavQuadWithCameraCfg

# Register components
env_config_registry.register("navigation_obstacle_env", NavigationObstacleEnvCfg)
robot_registry.register("nav_quadrotor_with_camera", BaseMultirotor, NavQuadWithCameraCfg)
task_registry.register_task("navigation_with_obstacles_task", NavigationWithObstaclesTask, task_config)

# Observation dimension names
OBS_NAMES = (
    ["log(d_hor+1)", "log(d_vert+1)", "cos(azimuth)", "sin(azimuth)",
     "elevation", "cos(yaw)", "sin(yaw)", "v_hor", "v_vert",
     "cos(track_az)", "sin(track_az)", "track_elev"]
    + [f"vae_{i:02d}" for i in range(32)]
)


def collect(num_steps: int, num_envs: int, out_dir: str, use_wandb: bool):
    os.makedirs(out_dir, exist_ok=True)

    # Init W&B BEFORE creating SummaryWriter so sync_tensorboard mirrors everything
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="aerial_gym",
                name=f"obs_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["obs_stats"],
                sync_tensorboard=True,
            )
        except Exception as e:
            print(f"W&B init failed (continuing without W&B): {e}")
            wandb_run = None

    # Build task
    task_config.num_envs = num_envs
    task_config.vae_config.encode_batch_size = num_envs
    task = task_registry.make_task(
        "navigation_with_obstacles_task",
        num_envs=num_envs,
        headless=True,
        use_warp=True,
    )

    obs_dim = task_config.observation_space_dim
    action_dim = task_config.action_space_dim
    device = task_config.device

    # Accumulate all observations: list of [num_envs, obs_dim] tensors
    all_obs = []

    task.reset()
    steps_collected = 0

    print(f"Collecting {num_steps} steps with {num_envs} envs "
          f"({num_steps // num_envs + 1} rollouts)...")

    while steps_collected < num_steps:
        # Random actions uniform in [-1, 1]
        actions = torch.rand(num_envs, action_dim, device=device) * 2.0 - 1.0
        obs_dict, rewards, terminations, truncations, infos = task.step(actions)

        obs = obs_dict["observations"].clone().cpu()  # [num_envs, obs_dim]
        all_obs.append(obs)
        steps_collected += num_envs

        if steps_collected % max(num_envs * 10, 1000) == 0:
            print(f"  {steps_collected}/{num_steps} steps collected")

    task.close()

    # Stack: [total_steps, obs_dim]
    all_obs = torch.cat(all_obs, dim=0).numpy()
    print(f"Total observations collected: {all_obs.shape[0]} x {all_obs.shape[1]}")

    # --- Compute per-dim statistics ---
    stats = {
        "dim":  list(range(obs_dim)),
        "name": OBS_NAMES,
        "min":  all_obs.min(axis=0).tolist(),
        "max":  all_obs.max(axis=0).tolist(),
        "mean": all_obs.mean(axis=0).tolist(),
        "std":  all_obs.std(axis=0).tolist(),
        "p05":  np.percentile(all_obs, 5,  axis=0).tolist(),
        "p25":  np.percentile(all_obs, 25, axis=0).tolist(),
        "p50":  np.percentile(all_obs, 50, axis=0).tolist(),
        "p75":  np.percentile(all_obs, 75, axis=0).tolist(),
        "p95":  np.percentile(all_obs, 95, axis=0).tolist(),
    }

    # --- Save CSV ---
    csv_path = os.path.join(out_dir, "obs_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "name", "min", "max", "mean", "std",
                                                "p05", "p25", "p50", "p75", "p95"])
        writer.writeheader()
        for i in range(obs_dim):
            writer.writerow({k: stats[k][i] for k in stats})
    print(f"CSV saved to: {csv_path}")

    # --- TensorBoard histograms ---
    tb_dir = os.path.join(out_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)

    for i in range(obs_dim):
        name = OBS_NAMES[i]
        data = all_obs[:, i]
        writer.add_histogram(f"obs_dist/{name}", data, global_step=0)
        writer.add_scalar(f"obs_stats/mean/{name}", float(stats["mean"][i]), global_step=0)
        writer.add_scalar(f"obs_stats/std/{name}",  float(stats["std"][i]),  global_step=0)
        writer.add_scalar(f"obs_stats/min/{name}",  float(stats["min"][i]),  global_step=0)
        writer.add_scalar(f"obs_stats/max/{name}",  float(stats["max"][i]),  global_step=0)
        writer.add_scalar(f"obs_stats/p05/{name}",  float(stats["p05"][i]),  global_step=0)
        writer.add_scalar(f"obs_stats/p95/{name}",  float(stats["p95"][i]),  global_step=0)

    writer.close()
    print(f"TensorBoard logs saved to: {tb_dir}")
    print(f"  Launch with: tensorboard --logdir {tb_dir}")

    # --- W&B native histograms (in addition to synced TensorBoard ones) ---
    if wandb_run is not None:
        import wandb
        log_dict = {}
        for i in range(obs_dim):
            name = OBS_NAMES[i]
            log_dict[f"obs_hist/{name}"] = wandb.Histogram(all_obs[:, i])
        wandb.log(log_dict, step=0)
        wandb.finish()
        print("W&B histograms logged.")

    # --- Print summary table ---
    print()
    print(f"{'dim':<5} {'name':<18} {'min':>8} {'p05':>8} {'mean':>8} {'p95':>8} {'max':>8} {'std':>8}")
    print("-" * 75)
    for i in range(obs_dim):
        print(f"{i:<5} {OBS_NAMES[i]:<18} "
              f"{stats['min'][i]:>8.3f} {stats['p05'][i]:>8.3f} "
              f"{stats['mean'][i]:>8.3f} {stats['p95'][i]:>8.3f} "
              f"{stats['max'][i]:>8.3f} {stats['std'][i]:>8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect observation statistics with random actions")
    parser.add_argument("--num_steps",  type=int,  default=10000,
                        help="Minimum total steps to collect (rounded up to next full rollout)")
    parser.add_argument("--num_envs",   type=int,  default=64,
                        help="Number of parallel environments")
    parser.add_argument("--out_dir",    type=str,
                        default="navigation_with_obstacles/obs_stats",
                        help="Output directory for CSV and TensorBoard logs")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb",
                        help="Disable W&B logging (default: enabled)")
    parser.set_defaults(wandb=True)
    args = parser.parse_args()

    collect(
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        out_dir=args.out_dir,
        use_wandb=args.wandb,
    )
