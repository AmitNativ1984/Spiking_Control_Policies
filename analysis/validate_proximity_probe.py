"""Validate ProximityProbe against two independent references before trusting it.

1. ALTITUDE: with the floor treated as an obstacle, a drone over clear ground should
   report d_obstacle close to its height above the floor.
2. DEPTH IMAGE: the camera renders the same geometry through a completely different code
   path. The probe is a FULL SPHERE so it sees at least as much as the 87 x 56 deg image,
   meaning probe <= image_min always. A large POSITIVE gap means the probe is missing
   geometry and the query is wrong; a negative gap is expected and healthy (the probe
   found something outside the camera cone).

usage: python analysis/validate_proximity_probe.py <level> [--num_envs 64] [--steps 40]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import math

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry
from env_manager.proximity_probe import ProximityProbe, find_mesh_ids

MAX_RANGE = 10.0
HALF_H = math.radians(87.0) / 2.0
HALF_V = math.atan(math.tan(HALF_H) * (180 / 320))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("level", type=int)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()

    from config.task_config import F450NavTaskConfig
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level

    task = task_registry.make_task("f450_navigation_task", num_envs=args.num_envs,
                                   headless=True, use_warp=True)
    task.reset()
    dev = task.device

    mesh_ids = find_mesh_ids(task.sim_env)
    print(f"mesh_ids_array: {type(mesh_ids)} "
          f"{'len ' + str(len(mesh_ids)) if mesh_ids is not None else 'NOT FOUND'}")
    if mesh_ids is None:
        raise SystemExit("could not reach the warp mesh ids -- probe cannot run")

    probe = ProximityProbe(args.num_envs, dev, max_dist=MAX_RANGE)

    zeros = torch.zeros(args.num_envs, 4, device=dev)
    alt_err, cone_err = [], []
    for s in range(args.steps):
        task.step(zeros)
        d = probe.measure(task.obs_dict["robot_position"], mesh_ids)

        # --- check 1: altitude, on envs where the floor should be the nearest thing ---
        z = task.obs_dict["robot_position"][:, 2]
        floor_z = task.obs_dict["env_bounds_min"][:, 2]
        alt = z - floor_z
        clear = d >= alt - 0.05           # floor is the nearest surface
        if clear.any():
            alt_err.append(float((d[clear] - alt[clear]).abs().mean()))

        # --- check 2: probe-in-cone vs depth image minimum ---
        img = task.obs_dict["depth_range_pixels"].squeeze(1).clone()
        img[img < 0] = 1.0
        img_min = img.flatten(1).min(dim=1).values * MAX_RANGE
        cone_err.append(float((d - img_min).mean()))

        if s == args.steps - 1:
            print(f"\nlast step, first 8 envs:")
            print(f"  altitude      {alt[:8].tolist()}")
            print(f"  probe d_obst  {d[:8].tolist()}")
            print(f"  depth-img min {img_min[:8].tolist()}")

    print(f"\ncheck 1  |probe - altitude| where floor is nearest: "
          f"{sum(alt_err)/max(len(alt_err),1):.3f} m  (want << 0.3)")
    print(f"check 2  mean(probe_full - image_min): {sum(cone_err)/len(cone_err):+.3f} m")
    print("         probe sees a WIDER set of directions than the image, so this should"
          " be <= 0. A large POSITIVE value means the probe is missing geometry.")
    task.close()


if __name__ == "__main__":
    main()
