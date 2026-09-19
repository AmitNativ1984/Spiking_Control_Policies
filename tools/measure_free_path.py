"""Measure the ONE obstacle-density statistic that is comparable across papers.

Obstacle COUNTS are not comparable: "30 obstacles" means nothing without the size and
shape of each, and every paper defines density differently (per m^2, per m^3, or as a
Poisson-disc spacing). What is comparable, and is what actually governs navigability, is:

    S(d) = P(a straight ray of length d from a random pose hits nothing)

For any Poisson obstacle field this is exponential, S(d) = exp(-d / lambda), and lambda
is the MEAN FREE PATH in metres. It is a property of the geometry alone -- it absorbs
obstacle size, shape and count into one number, with no modelling assumptions, and it is
measured here from the SAME depth camera the policy flies with.

lambda is estimated by the right-censored maximum-likelihood estimator

    lambda_hat = (total distance travelled by every ray) / (number of rays that hit)

where a ray that reaches --range_cap without hitting contributes the cap and no event.
A log-linear fit to S(d) was tried first and rejected: over a short window it is badly
biased when lambda is large (+36% at lambda = 42 m, +219% in a sparse field). The MLE
above is accurate to +/-3.4% from lambda = 7 m to lambda = 400 m, verified against
closed-form sphere and cylinder fields.

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
    p.add_argument("--range_cap", type=float, default=5.0,
                   help="censoring distance in metres. Rays are truncated here and count "
                        "as survivors, so no ray can reach a perimeter wall and be "
                        "miscounted as clutter. Must stay below the env half-width")
    return p.parse_args()


def horizon_margins(nav_cfg, args):
    """Pose-rejection margins implied by --range_cap and --row_band.

    Horizontal: a ray may leave along the full +/-43.5 deg horizontal FOV, so the pose
    must be range_cap from the +/-x and +/-y bounds.
    Vertical: --row_band keeps only rows near the image centre, so the steepest retained
    ray climbs range_cap * sin(elevation) -- a few tens of centimetres, not range_cap.
    That distinction matters: the box is only 4-6 m tall, so a range_cap z-margin would
    reject every pose.
    """
    from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig as cam
    vfov_half = math.atan(math.tan(math.radians(cam.horizontal_fov_deg/2))
                          * cam.height / cam.width)
    elev = math.atan(2*(args.row_band/2)*math.tan(vfov_half))
    return args.range_cap, args.range_cap*math.sin(elev) + 0.3


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

    xy_margin, z_margin = horizon_margins(nav_cfg, args)
    print(f"[cfg] censoring at {args.range_cap:.1f} m; pose margins "
          f"xy {xy_margin:.1f} m, z {z_margin:.2f} m")

    # Accumulate per frame, but TAG each frame with the obstacle layout it came from.
    # Rays inside a frame are correlated, and so are frames that share a layout -- the
    # dominant variance in a Poisson field is which obstacles got drawn, not which
    # pixel you look at. The interval below therefore resamples LAYOUTS.
    frame_exposure, frame_events, frame_layout = [], [], []
    layout_id = 0
    kept = 0
    while kept < args.frames:
        env_manager.global_tensor_dict["obstacle_intensity"] = intensity
        layout_id += 1          # env_manager.reset() below re-draws every env's obstacles
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
                # Only poses well inside the box: a ray that stops on a perimeter wall
                # is measuring the room, not the clutter.
                m = np.array([xy_margin, xy_margin, z_margin])
                if (pos[e] - lo[e] < m).any() or (hi[e] - pos[e] < m).any():
                    continue
                frame = depth[e][rows]
                near_sentinel = frame <= 0.0          # closer than min_range: inside something
                if near_sentinel.mean() > 0.5:
                    continue                          # camera buried in an obstacle
                d = np.where(frame >= 1.0, np.inf, frame * sensor_max)
                hit = np.isfinite(d) & (d < args.range_cap) & ~near_sentinel
                frame_exposure.append(float(np.where(hit, d, args.range_cap)
                                            [~near_sentinel].sum()))
                frame_events.append(int(hit.sum()))
                # Each env holds its own independent Poisson draw, so the layout key is
                # (reset index, env index), not the reset alone.
                frame_layout.append((layout_id, e))
                kept += 1

    exposure = np.array(frame_exposure)
    events = np.array(frame_events)
    if events.sum() < 500:
        print(f"[warn] only {events.sum()} hits; lambda will be noisy. Raise --frames.")
    lam = exposure.sum() / max(events.sum(), 1)

    # Cluster bootstrap over LAYOUTS. Resampling frames (or rays) would understate the
    # interval, because the dominant variance is which obstacles were drawn.
    keys = sorted(set(frame_layout))
    by_layout = {k: [] for k in keys}
    for i, k in enumerate(frame_layout):
        by_layout[k].append(i)
    idx_of = [np.array(by_layout[k]) for k in keys]

    rs = np.random.default_rng(args.seed)
    boot = []
    for _ in range(1000):
        pick = np.concatenate([idx_of[j] for j in
                               rs.integers(0, len(idx_of), len(idx_of))])
        boot.append(exposure[pick].sum() / max(events[pick].sum(), 1))
    lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])

    print(f"[data] {kept} usable frames over {len(keys)} independent layouts, "
          f"{events.sum()} hits within {args.range_cap:.1f} m")
    if len(keys) < 60:
        print(f"[warn] only {len(keys)} layouts. The interval is driven by layout "
              f"variance -- raise --frames (or lower --poses_per_layout) until it is "
              f"tight enough to decide against the 27 m threshold.")
    # The statistical interval is not the whole story. Pooling rays across obstacle
    # layouts makes the free-path distribution a MIXTURE of exponentials, which is not
    # itself exponential, so the MLE carries a small systematic offset -- measured at
    # 1-3% against closed-form fields, and it is what makes a pure bootstrap interval
    # under-cover (67% at nominal 95% once enough layouts narrow it). Report both, and
    # refuse to call a difference this tool cannot resolve.
    SYS = 0.034          # validated systematic, fraction
    lo_tot, hi_tot = lo_ci*(1-SYS), hi_ci*(1+SYS)
    print(f"\n  MEAN FREE PATH lambda = {lam:.1f} m")
    print(f"    statistical 95% CI      {lo_ci:5.1f} - {hi_ci:5.1f} m")
    print(f"    incl. 3.4% systematic   {lo_tot:5.1f} - {hi_tot:5.1f} m   <- use this one")
    print("\n  Agile Autonomy: 26.7 m (densest, s=4) .. 81.7 m (sparsest, s=7)")
    if lo_tot > 26.7:
        print("  -> SPARSER than their densest forest")
    elif hi_tot < 15.0:
        print("  -> DENSER than anything in either paper")
    elif lo_tot >= 15.0 and hi_tot <= 26.7:
        print("  -> COMPARABLE to their forest range")
    else:
        print("  -> TOO CLOSE TO CALL at this sample size. The interval straddles a "
              "threshold; raise --frames and re-run before concluding anything.")


if __name__ == "__main__":
    main()
