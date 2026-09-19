"""Measure the ONE obstacle-density statistic that is comparable across papers.

Obstacle COUNTS are not comparable: "30 obstacles" means nothing without the size and
shape of each, and every paper defines density differently (per m^2, per m^3, or as a
Poisson-disc spacing). What is comparable, and is what actually governs navigability, is:

    S(d) = P(a straight ray of length d from a random pose hits nothing)

For any Poisson obstacle field this is exponential, S(d) = exp(-d / lambda), and lambda
is the MEAN FREE PATH in metres. It is a property of the geometry alone -- it absorbs
obstacle size, shape and count into one number, with no modelling assumptions, and it is
measured here from the SAME depth camera the policy flies with.

Closed form for the literature (no simulation needed):
    Agile Autonomy forests are vertical cylinders of radius r = 0.3 m at areal density
    n = 1/s^2, so for a horizontal ray  S(d) = exp(-2*r*n*d)  and  lambda = 1/(2*r*n):

        spacing s = 4 m  (n = 0.0625/m^2, their DENSEST)   lambda = 26.7 m
        spacing s = 5 m  (n = 0.0400/m^2)                  lambda = 41.7 m
        spacing s = 6 m  (n = 0.0278/m^2)                  lambda = 60.0 m
        spacing s = 7 m  (n = 0.0204/m^2, their sparsest)  lambda = 81.7 m

    DCE (Kulkarni & Alexis) CANNOT be placed on this scale: the paper gives obstacle
    counts and room sizes but not obstacle dimensions, so its lambda is not recoverable
    from the publication. Do not claim a comparison to it.

Reading the result:
    lambda > 27 m   -> your field is SPARSER than Agile Autonomy's densest forest
    lambda 15-27 m  -> comparable to their forest range
    lambda < 15 m   -> denser than anything in either paper

Usage (inside the container):
    python tools/measure_free_path.py --level 30 --frames 400
    python tools/measure_free_path.py --level 30 --frames 400 --no_trees

    The --no_trees run is the point of the whole exercise: the difference in lambda
    between the two is the trees' TRUE contribution, integrating the real canopy meshes
    with no assumption about how porous they are.
"""
import argparse
import math

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, default=30, help="curriculum level to pin")
    p.add_argument("--frames", type=int, default=400, help="depth frames to accumulate")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--poses_per_layout", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_trees", action="store_true",
                   help="zero the tree pool before building, to isolate their contribution")
    p.add_argument("--row_band", type=float, default=0.25,
                   help="fraction of image height about the centre row to keep, so only "
                        "near-horizontal rays enter the fit (the floor and ceiling are "
                        "structure, not clutter)")
    p.add_argument("--wall_margin", type=float, default=4.0,
                   help="reject poses closer than this to the env bounds, so the "
                        "perimeter walls do not truncate rays and bias lambda down")
    p.add_argument("--fit_max", type=float, default=6.0,
                   help="upper end of the fit window in metres. Must stay well inside "
                        "the box or the boundary, not the clutter, sets the slope")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_trees:
        # Must happen before the env is built: asset pools are read at load time.
        from config.asset_config.enlarged_object_config import tree_asset_params
        tree_asset_params.num_assets = 0
        print("[cfg] tree pool zeroed for this run")

    # Reuses the dataset collector's builder verbatim, so the obstacle field is exactly
    # the one the policy trains in (Poisson placement, same pools, same keep-out).
    from vae_depth.data_generation.generate_dataset import setup_sim

    env_manager, nav_cfg = setup_sim(args)
    top = nav_cfg.curriculum.density_at_level
    intensity = nav_cfg.obstacle_density_max * min(args.level / max(top, 1), 1.0)
    print(f"[cfg] level {args.level} -> intensity {intensity:.4f} obstacles/m^3")

    sensor_max = nav_cfg.vae_config.sensor_max_range
    actions = torch.zeros((args.num_envs, 4), device=args.device)
    all_envs = torch.arange(args.num_envs, device=args.device)

    hits = []          # measured ray lengths, metres
    censored = []      # rays that reached max range without hitting anything
    kept = 0
    while kept < args.frames:
        env_manager.global_tensor_dict["obstacle_intensity"] = intensity
        for pose_idx in range(max(args.poses_per_layout, 1)):
            if pose_idx == 0:
                env_manager.reset()
            else:
                env_manager.robot_manager.reset_idx(all_envs)
                env_manager.IGE_env.write_to_sim()
            env_manager.step(actions=actions)
            env_manager.render(render_components="sensors")
            env_manager.reset_terminated_and_truncated_envs()

            depth = env_manager.global_tensor_dict["depth_range_pixels"][:, 0].cpu().numpy()
            pos = env_manager.global_tensor_dict["robot_position"].cpu().numpy()
            lo = env_manager.global_tensor_dict["env_bounds_min"].cpu().numpy()
            hi = env_manager.global_tensor_dict["env_bounds_max"].cpu().numpy()
            if lo.ndim == 3:            # (envs, assets, 3) -> (envs, 3)
                lo, hi = lo[:, 0, :], hi[:, 0, :]

            H = depth.shape[1]
            half = max(int(H * args.row_band / 2), 1)
            rows = slice(H // 2 - half, H // 2 + half)

            for e in range(args.num_envs):
                if kept >= args.frames:
                    break
                # Only poses well inside the box: a ray that stops on a perimeter wall is
                # measuring the room, not the clutter.
                if (pos[e] - lo[e] < args.wall_margin).any() or \
                   (hi[e] - pos[e] < args.wall_margin).any():
                    continue
                frame = depth[e][rows]
                near_sentinel = frame <= 0.0          # closer than min_range
                far_sentinel = frame >= 1.0           # nothing within max_range
                d = frame * sensor_max
                good = ~near_sentinel & ~far_sentinel
                hits.append(d[good].ravel())
                censored.append(int(far_sentinel.sum()))
                kept += 1

    hits_all = np.concatenate(hits)
    n_cens = int(np.sum(censored))
    n_total = hits_all.size + n_cens
    print(f"[data] {kept} frames, {n_total/1e6:.2f} M rays "
          f"({100*n_cens/n_total:.1f}% reached {sensor_max:.0f} m without a hit)")

    # Survival curve. Censored rays count as survivors at every d in the window, which is
    # exactly right -- dropping them would bias lambda down.
    grid = np.arange(0.5, min(args.fit_max, sensor_max) + 1e-9, 0.25)
    S = np.array([(hits_all > d).sum() + n_cens for d in grid], float) / n_total

    ok = S > 1e-4
    lam = -1.0 / np.polyfit(grid[ok], np.log(S[ok]), 1)[0]

    print("\n  d (m)   S(d) measured   S(d) if lambda were 26.7 m (AA densest)")
    for d, s in zip(grid, S):
        if abs(d - round(d)) < 1e-9:
            print(f"  {d:5.1f}   {s:12.3f}   {math.exp(-d/26.7):12.3f}")
    print(f"\n  MEAN FREE PATH lambda = {lam:.1f} m")
    print("  Agile Autonomy: 26.7 m (densest, s=4) .. 81.7 m (sparsest, s=7)")
    if lam > 27:
        print("  -> SPARSER than their densest forest")
    elif lam >= 15:
        print("  -> COMPARABLE to their forest range")
    else:
        print("  -> DENSER than anything in either paper")


if __name__ == "__main__":
    main()
