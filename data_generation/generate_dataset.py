"""Generate the depth-image dataset for VAE training.

Renders depth from the SAME environment the navigation policy flies in
(config/env_config/env_forest_with_obstacles.py) through the SAME camera it reads from
(config/sensor_config/realsense_d435_cam_config.py), by teleporting a camera carrier to
random poses inside the obstacle field.

Usage:
    python -m data_generation.generate_dataset                                # 85k, defaults
    python -m data_generation.generate_dataset --num_images 128 --num_envs 4  # smoke test
"""

import isaacgym  # noqa: F401  -- MUST precede torch; see aerial_gym/__init__.py

import os
import argparse
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image

from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("data_generation")

# Upper edges of the near-obstacle stratification bands, in metres. See --band_cap.
BAND_EDGES_M = [1.0, 2.0, 4.0, 7.0, float("inf")]
BAND_LABELS = ["<1m", "1-2m", "2-4m", "4-7m", ">7m"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth image dataset for VAE training")
    parser.add_argument("--num_images", type=int, default=85000,
                        help="Total number of images to generate")
    parser.add_argument("--num_envs", type=int, default=32,
                        help="Number of parallel environments. This is a MEMORY knob, not a "
                             "throughput one: reset and render both measured linear in env "
                             "count (8/32/64 envs -> 229/209/205 ms per image), so raising "
                             "it buys nothing but VRAM pressure.")
    parser.add_argument("--poses_per_layout", type=int, default=4,
                        help="Camera poses rendered per obstacle layout. Re-placing the "
                             "obstacles costs ~3.4 s at 32 envs (Poisson draw over ~200 "
                             "assets, plus the warp BVH refit) while re-randomizing the "
                             "camera pose costs 1.5 ms, so reusing a layout for K poses "
                             "cuts the per-image cost by ~1.6x at K=4. The price is "
                             "correlation: K frames share an obstacle field. At 85k images "
                             "and K=4 that is still ~21k independent layouts.")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.expanduser("~/DATA/depth-images-forest"),
                        help="Output directory for depth images")
    parser.add_argument("--format", type=str, default="png", choices=["png", "npy"],
                        help="Image save format: png (16-bit) or npy (float32)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_valid_ratio", type=float, default=0.05,
                        help="Minimum ratio of valid pixels (0-1) to accept an image")
    parser.add_argument("--empty_level_prob", type=float, default=0.05,
                        help="Probability of drawing a near-empty curriculum level (0-2). "
                             "Those frames carry almost no collision signal, so they are "
                             "deliberately rare -- but not absent, since the policy does "
                             "fly them early in the curriculum.")
    parser.add_argument("--band_cap", type=float, default=0.0,
                        help="Max share of the dataset any one near-obstacle band may take; "
                             "0 (the default) disables stratification. Bands are the "
                             "5th-percentile depth of a frame, split at 1/2/4/7 m. OFF by "
                             "default because the measured natural mix over 1024 frames is "
                             "already near-heavy -- 21.9%% under 1 m, 31.2%% 1-2 m, 39.6%% "
                             "2-4 m, 6.8%% 4-7 m, 0.6%% beyond -- since the floor slab is in "
                             "frame in nearly every view. A cap only has ~7.4%% of far "
                             "frames to redistribute into, so anything at or below 0.31 is "
                             "arithmetically unsatisfiable and just burns renders. Raise it "
                             "only to deliberately reshape the mix.")
    parser.add_argument("--stratify_patience", type=int, default=200,
                        help="Give up on the band quota after this many consecutive layouts "
                             "saved nothing, rather than spinning forever on a band the "
                             "environment cannot actually produce.")
    return parser.parse_args()


def setup_environment(args):
    """Build the sim and swap in the navigation task's obstacle sampler."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Registers data_gen_env / data_gen_quad, and by import chain the whole `config`
    # package (forest env, f450, the attitude controller, the nav task).
    import data_generation  # noqa: F401

    from aerial_gym.sim.sim_builder import SimBuilder
    from config.robot_config.f450_config import F450Config
    from config.task_config.f450_attitude_navigation_task_config import (
        task_config as NavTaskConfig,
    )
    from env_manager.poisson_asset_manager import PoissonAssetManager

    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="data_gen_env",
        robot_name="data_gen_quad",
        controller_name=NavTaskConfig.controller_name,
        args=None,
        device=args.device,
        num_envs=args.num_envs,
        headless=True,
        use_warp=True,
    )

    # Same swap the navigation task performs (task/attitude_navigation_task.py). Without
    # it the new asset pool is placed by the UPSTREAM ratio-box manager, which confines
    # every obstacle class to x-ratio [0.30, 0.85] -- the anisotropy the Poisson process
    # exists to remove.
    asset_manager = PoissonAssetManager(env_manager.global_tensor_dict, env_manager.keep_in_env)

    # The POLICY's spawn box, not this robot's. The pair sizes the obstacle keep-out
    # ellipsoid, so passing DataGenF450Cfg's whole-env roam range would drive free_volume
    # negative and yield an empty world on every reset. What we want is an obstacle field
    # statistically identical to the one the policy meets -- which means the keep-out has
    # to be the one the policy's spawn box carves out. The camera then roams over the
    # result, including into obstacles; those frames are rejected below.
    asset_manager.spawn_ratio_lo = torch.tensor(
        F450Config.init_config.min_init_state[0:3], device=args.device)
    asset_manager.spawn_ratio_hi = torch.tensor(
        F450Config.init_config.max_init_state[0:3], device=args.device)
    asset_manager.clearance = NavTaskConfig.obstacle_spawn_clearance
    env_manager.asset_manager = asset_manager

    return env_manager, NavTaskConfig


def sample_intensity(nav_cfg, empty_level_prob):
    """Draw a curriculum level and convert it to a Poisson intensity (obstacles/m^3).

    Mirrors NavigationWithObstaclesTask._publish_obstacle_intensity exactly: density ramps
    linearly with the ABSOLUTE level, reaching obstacle_density_max at density_at_level.
    Sampling the level rather than the density means the dataset is spread over the
    clutter the policy actually meets, weighted the way the curriculum weights it.
    """
    top = nav_cfg.curriculum.density_at_level
    if np.random.rand() < empty_level_prob:
        level = np.random.randint(0, 3)          # 0-2: near-empty worlds
    else:
        level = np.random.randint(3, top + 1)    # 3-25: the clutter that carries signal
    return nav_cfg.obstacle_density_max * min(level / max(top, 1), 1.0), level


def save_depth_image(depth_np, filepath, fmt="png"):
    """Write one [H, W] normalized-depth frame to disk."""
    if fmt == "npy":
        np.save(filepath + ".npy", depth_np.astype(np.float32))
    else:
        # 16-bit PNG. The clamp also folds the sensor's near-out-of-range sentinel (-1)
        # onto 0; that is not a loss, because both 0 and -1 land on min_depth_m once
        # vae_depth.preprocessing.normalize_depth clamps, on the training and the
        # inference path alike.
        depth_uint16 = (np.clip(depth_np, 0.0, 1.0) * 65535.0).astype(np.uint16)
        Image.fromarray(depth_uint16, mode="I;16").save(filepath + ".png")


def band_of(depth_m_valid):
    """Index of the near-obstacle band a frame belongs to.

    Keyed on the 5th-percentile depth: how close the NEAREST real structure is, robust to
    a handful of stray pixels. This is the axis the collision target is most sensitive to
    -- the dilation radius goes as 1/d, 149 px at 0.75 m against 16 px at 7 m -- so an
    unstratified dataset, which in a ~2000 m^3 box is mostly far/empty views, trains a VAE
    that resolves exactly the range that matters least.
    """
    p5 = np.percentile(depth_m_valid, 5.0)
    for i, edge in enumerate(BAND_EDGES_M):
        if p5 < edge:
            return i
    return len(BAND_EDGES_M) - 1


def generate_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logger.warning(f"Generating {args.num_images} depth images with {args.num_envs} parallel envs")
    logger.warning(f"Output directory: {args.output_dir}")

    env_manager, nav_cfg = setup_environment(args)
    sensor_max_range = nav_cfg.vae_config.sensor_max_range

    actions = torch.zeros((args.num_envs, 4), device=args.device)
    all_envs = torch.arange(args.num_envs, device=args.device)

    band_cap = int(args.band_cap * args.num_images) if args.band_cap > 0 else 0
    band_counts = Counter()
    level_counts = Counter()

    env_manager.reset()

    total_saved = total_skipped = total_off_quota = 0
    step_count = steps_since_save = 0
    stratifying = band_cap > 0
    start_time = time.time()

    while total_saved < args.num_images:
        # One intensity per layout. PoissonAssetManager reads it out of the tensor dict at
        # reset time; it is a scalar, so an env's obstacle COUNT still varies (its own
        # Poisson draw over its own randomized volume) while the density does not.
        intensity, level = sample_intensity(nav_cfg, args.empty_level_prob)
        env_manager.global_tensor_dict["obstacle_intensity"] = intensity

        saved_this_step = 0
        for pose_idx in range(max(args.poses_per_layout, 1)):
            if pose_idx == 0:
                env_manager.reset()                       # obstacles AND camera pose
            else:
                # Camera pose only; the obstacle layout stands. This is env_manager.reset_idx
                # with the two expensive stages dropped -- asset_manager.reset_idx (the
                # Poisson draw) and warp_env.reset_idx (the BVH refit) -- keeping the robot
                # reset and, crucially, the write_to_sim() that ends it.
                #
                # write_to_sim() is NOT optional and is not covered by the step() below.
                # robot_manager.reset_idx only stages the pose in robot_state; without the
                # explicit push, post_physics_step refreshes that tensor back from the sim
                # and the "new" pose silently reverts to the old one. Verified: dropping it
                # left the camera at 0.00 m of movement across poses, rendering the same
                # view K times with only the mount jitter to tell them apart.
                #
                # robot_manager (not robot) so the camera sensor's mount randomization is
                # re-drawn too, exactly as on the full-reset path.
                env_manager.robot_manager.reset_idx(all_envs)
                env_manager.IGE_env.write_to_sim()

            env_manager.step(actions=actions)
            env_manager.render(render_components="sensors")
            env_manager.reset_terminated_and_truncated_envs()

            # [num_envs, num_sensors, H, W] -> [num_envs, H, W], once, on the host. One
            # transfer per render beats one per image.
            depth = env_manager.global_tensor_dict["depth_range_pixels"][:, 0].cpu().numpy()

            for env_idx in range(args.num_envs):
                if total_saved >= args.num_images:
                    break

                frame = depth[env_idx]
                # Valid = actually measured. 0 and 1 are the near/far out-of-range
                # sentinels (the near one arrives as -1 and is clipped on write).
                valid = (frame > 0.0) & (frame < 1.0)
                if valid.mean() < args.min_valid_ratio:
                    total_skipped += 1
                    continue

                band = band_of(frame[valid] * sensor_max_range)
                if stratifying and band_counts[band] >= band_cap:
                    total_off_quota += 1
                    continue

                save_depth_image(frame,
                                 os.path.join(args.output_dir, f"depth_{total_saved:06d}"),
                                 fmt=args.format)
                band_counts[band] += 1
                level_counts[level] += 1
                total_saved += 1
                saved_this_step += 1

            if total_saved >= args.num_images:
                break

        step_count += 1
        steps_since_save = 0 if saved_this_step else steps_since_save + 1

        if stratifying and steps_since_save >= args.stratify_patience:
            # Every band the environment can still produce is full. Spinning here would
            # never terminate, so drop the quota and say so -- a silently truncated run is
            # worse than a knowingly unbalanced one.
            stratifying = False
            logger.warning(
                f"Band quota unsatisfiable after {steps_since_save} empty steps at "
                f"{total_saved}/{args.num_images} images; continuing unstratified. "
                f"Bands so far: {dict(sorted(band_counts.items()))}")

        if step_count % 50 == 0:
            elapsed = time.time() - start_time
            logger.warning(
                f"Layout {step_count}: {total_saved}/{args.num_images} saved, "
                f"{total_skipped} invalid, {total_off_quota} off-quota "
                f"({total_saved / max(elapsed, 1e-9):.1f} img/s)")

    elapsed = time.time() - start_time
    logger.warning(
        f"Done: {total_saved} images in {elapsed:.1f}s ({total_saved / elapsed:.1f} img/s); "
        f"{total_skipped} invalid, {total_off_quota} rejected by the band quota")
    logger.warning("Nearest-structure bands: " + ", ".join(
        f"{BAND_LABELS[i]} {band_counts[i]} "
        f"({100.0 * band_counts[i] / max(total_saved, 1):.1f}%)"
        for i in range(len(BAND_LABELS))))
    logger.warning("Curriculum levels drawn: " + ", ".join(
        f"{lv}:{n}" for lv, n in sorted(level_counts.items())))


if __name__ == "__main__":
    generate_dataset(parse_args())
