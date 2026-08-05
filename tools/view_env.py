"""Open the F450 navigation env in the Isaac Gym viewer and just hover, so the obstacle
field, the spawn box and the target can be looked at.

Not a training run: the drone holds a hover command and episodes time out and reset, so
each reset draws a fresh Poisson obstacle field at the chosen curriculum level.

In the viewer window:
    red wireframe sphere    the navigation target (on one of the four vertical walls)
    green wireframe sphere  where the drone spawned this episode
    separate OpenCV window  the drone's depth camera

Needs a display. Inside this container DISPLAY is already set; from the host run
`xhost +local:docker` first.

Usage:
    cd /workspaces/aerial_gym_docker
    python -m tools.view_env                          # 4 envs, max clutter
    python -m tools.view_env --curriculum_level 0     # empty world, for comparison
    python -m tools.view_env --num_envs 16 --steps 2000
"""
import isaacgym  # noqa: F401  — must precede torch

import argparse

import torch

import config  # noqa: F401  — registers the env, robot and task
from aerial_gym.registry.task_registry import task_registry
from config.task_config import F450NavTaskConfig as task_config

TASK_NAME = "f450_navigation_task"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Envs to build. Small is easier to look at; they are laid "
                             "out in a grid.")
    parser.add_argument("--curriculum_level", type=int, default=25,
                        help="Obstacle density level (0 = empty, 25 = max).")
    parser.add_argument("--steps", type=int, default=0,
                        help="Steps to run; 0 means run until Ctrl-C.")
    args = parser.parse_args()

    # Pin the curriculum before make_task: the task reads the level in __init__ and the
    # obstacle field is spawned from it.
    task_config.curriculum.min_level = args.curriculum_level
    task_config.curriculum.max_level = args.curriculum_level

    task = task_registry.make_task(
        TASK_NAME, num_envs=args.num_envs, headless=False, use_warp=True
    )

    # Slots [0:keep_in_env] are structural (the ground plane, and any perimeter walls):
    # always present, placed from their ratio config, not part of the Poisson process.
    # Counting them here would report a floor as an obstacle and never show a truly
    # empty world at level 0.
    num_keep = task.sim_env.keep_in_env
    live = (task.obs_dict["env_asset_state_tensor"][:, num_keep:, 0] > -900).sum(dim=1)
    print(f"\n  curriculum level : {task.curriculum_level}")
    print(f"  obstacle density : {task.obs_dict['obstacle_intensity']:.4f} /m^3")
    print(f"  structural assets: {num_keep} (ground plane / walls, always present)")
    print(f"  obstacles per env: {live.float().mean():.1f} "
          f"(min {int(live.min())}, max {int(live.max())})")
    print("\n  Ctrl-C to quit.\n")

    # Zero is the neutral attitude command: LeeAttitudeController maps thrust via
    # (cmd + 1) * m * g, so 0 == hover, level, no yaw rate.
    hover = torch.zeros((args.num_envs, task_config.action_space_dim), device=task.device)

    task.reset()
    try:
        step = 0
        while args.steps == 0 or step < args.steps:
            task.step(hover)
            step += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        task.close()


if __name__ == "__main__":
    main()
