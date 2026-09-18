"""Validate RaySphereProbe before any reward depends on it.

The probe replaces an exact closest-point query that faults under Warp 1.0.0 (see
env_manager/proximity_probe.py). Ray casting can only OVER-estimate clearance -- geometry
thinner than the ray spacing is missed -- and an over-estimate makes the CBF permit a
higher closing speed than it should. So the error has to be measured, not assumed.

Three checks, each against something the probe does not share code with:

  1. AGAINST THE EXACT ANSWER, at curriculum level 0. Level 0 is the one configuration
     where mesh_query_point_no_sign survives (few triangles), so there the exact distance
     is available and the ray probe can be scored directly against it. Reports the error
     distribution and asserts the sign: ray >= exact, always, to within float noise.

  2. AGAINST THE DEPTH IMAGE, at level 30. The camera renders the same geometry through a
     completely separate path. The probe is a full sphere and the image is an 87x56 deg
     cone, so probe <= image_min must hold; a large POSITIVE gap means the probe is missing
     geometry the camera can see.

  3. NO FAULTS, at level 30, over many steps -- the failure that killed the exact query.
     Also reports the clearance distribution, which is what d_safe has to fit inside, and
     the per-step cost of the probe.

Isaac Gym allows ONE sim per process, so the two curriculum levels cannot share a run:
invoke this twice.

usage: python analysis/validate_ray_probe.py --phase exact [--exact_level 0]
       python analysis/validate_ray_probe.py --phase field [--level 30] [--steps 100]
       common: [--num_envs 128] [--rays 16384] [--max_dist 6.0]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import math
import time

import torch
import warp as wp

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry
from env_manager.proximity_probe import (
    RaySphereProbe,
    _nearest_surface,
    find_mesh_ids,
)

IMG_RANGE = 10.0  # the depth image encodes range as value * 10 m
HALF_H = math.radians(87.0) / 2.0
HALF_V = math.atan(math.tan(HALF_H) * (180 / 320))


def quantiles(t, qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)):
    s = torch.sort(t).values
    n = s.numel()
    return {q: float(s[min(n - 1, max(0, int(round(q * (n - 1)))))]) for q in qs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, default=30)
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--rays", type=int, default=16384)
    p.add_argument("--max_dist", type=float, default=6.0)
    p.add_argument("--exact_level", type=int, default=0)
    p.add_argument("--phase", choices=["exact", "field"], required=True,
                   help="one sim per process: 'exact' runs check 1, 'field' runs 2 and 3")
    args = p.parse_args()

    from config.task_config import F450NavTaskConfig

    # ---------------- check 1: against the exact query, at level 0 --------------------
    if args.phase == "exact":
      print("=" * 72, flush=True)
      print(f"CHECK 1  ray probe vs the EXACT closest-point query, level {args.exact_level}",
            flush=True)
      F450NavTaskConfig.curriculum.min_level = args.exact_level
      F450NavTaskConfig.curriculum.max_level = args.exact_level

      task = task_registry.make_task(
          "f450_navigation_task", num_envs=args.num_envs, headless=True, use_warp=True
      )
      task.reset()
      dev = str(task.device)
      mesh_ids = find_mesh_ids(task.sim_env)
      if mesh_ids is None:
          raise SystemExit("could not reach the warp mesh ids")

      probe = RaySphereProbe(args.num_envs, dev, max_dist=args.max_dist, num_rays=args.rays)
      spacing = math.degrees(math.sqrt(4.0 * math.pi / args.rays))
      print(f"  {args.rays} rays, {spacing:.2f} deg spacing, "
            f"catches r >= {2.0 * math.radians(spacing) / 2.0:.3f} m at 2 m", flush=True)

      exact = torch.empty(args.num_envs, dtype=torch.float32, device=dev)
      exact_wp = wp.from_torch(exact, dtype=wp.float32)
      pts = torch.empty((args.num_envs, 3), dtype=torch.float32, device=dev)
      pts_wp = wp.from_torch(pts, dtype=wp.vec3)

      errs, zeros = [], torch.zeros(args.num_envs, 4, device=dev)
      for s in range(30):
          pts.copy_(task.obs_dict["robot_position"])
          exact.fill_(args.max_dist)
          wp.launch(kernel=_nearest_surface, dim=args.num_envs,
                    inputs=[mesh_ids, pts_wp, float(args.max_dist)],
                    outputs=[exact_wp], device=dev)
          d_ray = probe.measure(task.obs_dict["robot_position"], mesh_ids).clone()
          torch.cuda.synchronize()
          # only where BOTH found something, so the shared clip does not flatter the score
          m = (exact < args.max_dist - 1e-3) & (d_ray < args.max_dist - 1e-3)
          if m.any():
              errs.append((d_ray[m] - exact[m]).clone())
          task.step(zeros)

      if errs:
          e = torch.cat(errs)
          q = quantiles(e)
          print(f"  n={e.numel()}  mean {float(e.mean()):+.4f} m  "
                f"median {q[0.5]:+.4f}  p90 {q[0.9]:+.4f}  p99 {q[0.99]:+.4f}  "
                f"max {float(e.max()):+.4f}", flush=True)
          print(f"  most negative: {float(e.min()):+.5f} m "
                f"(ray casting cannot under-estimate; anything below ~-1e-3 is a bug)",
                flush=True)
          assert float(e.min()) > -5e-3, (
              f"ray probe read {float(e.min()):.5f} m CLOSER than the exact query -- rays "
              f"sample the same surfaces, so they can only ever over-estimate."
          )
          print("  PASS: sign is correct and the error is bounded", flush=True)
      else:
          print("  INCONCLUSIVE: no env had both a ray hit and an exact hit in range",
                flush=True)
      task.close()
      print("\nPHASE 'exact' DONE -- now run --phase field", flush=True)
      return

    # ---------------- checks 2 and 3: at the training level ---------------------------
    print("=" * 72, flush=True)
    print(f"CHECKS 2+3  vs the depth image, and fault-free, level {args.level}", flush=True)
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level
    task = task_registry.make_task(
        "f450_navigation_task", num_envs=args.num_envs, headless=True, use_warp=True
    )
    task.reset()
    dev = str(task.device)
    mesh_ids = find_mesh_ids(task.sim_env)
    probe = RaySphereProbe(args.num_envs, dev, max_dist=args.max_dist, num_rays=args.rays)

    gaps, ds, alts = [], [], []
    zeros = torch.zeros(args.num_envs, 4, device=dev)
    t_probe = 0.0

    for s in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d = probe.measure(task.obs_dict["robot_position"], mesh_ids)
        torch.cuda.synchronize()
        t_probe += time.perf_counter() - t0
        d = d.clone()

        img = task.obs_dict["depth_range_pixels"].squeeze(1).clone()
        img[img < 0] = 1.0
        img_min = img.flatten(1).min(dim=1).values * IMG_RANGE
        both = (d < args.max_dist - 1e-3) & (img_min < IMG_RANGE - 1e-3)
        if both.any():
            gaps.append((d[both] - img_min[both]).clone())
        ds.append(d)
        alts.append((task.obs_dict["robot_position"][:, 2]
                     - task.obs_dict["env_bounds_min"][:, 2]).clone())
        task.step(zeros)
        if (s + 1) % 25 == 0:
            print(f"  step {s + 1}/{args.steps} clean", flush=True)

    print(f"\n  CHECK 3 PASS: {args.steps} steps x {args.num_envs} envs, no fault", flush=True)
    print(f"  probe cost {1000 * t_probe / args.steps:.2f} ms/step "
          f"at {args.num_envs} envs x {args.rays} rays", flush=True)

    if gaps:
        g = torch.cat(gaps)
        q = quantiles(g)
        print(f"\n  CHECK 2  probe - image_min over {g.numel()} samples:", flush=True)
        print(f"    median {q[0.5]:+.3f} m  p90 {q[0.9]:+.3f}  p99 {q[0.99]:+.3f}  "
              f"max {float(g.max()):+.3f}", flush=True)
        print("    The probe is a SPHERE and the image a cone, so this should sit at or "
              "below 0.\n    A large positive value means the probe misses geometry the "
              "camera can see.", flush=True)
        frac_bad = float((g > 0.10).float().mean())
        print(f"    fraction more than 0.10 m ABOVE the image: {100 * frac_bad:.2f}%",
              flush=True)

    dall = torch.cat(ds)
    q = quantiles(dall)
    print(f"\n  CLEARANCE DISTRIBUTION (full sphere, floor included), "
          f"{dall.numel()} samples:", flush=True)
    print("    " + "  ".join(f"p{int(k * 100)} {v:.2f}" for k, v in q.items()), flush=True)
    for thr in (0.7, 1.0, 1.25, 1.5, 2.0):
        print(f"      below {thr:.2f} m: {100 * float((dall < thr).float().mean()):.1f}%",
              flush=True)
    aall = torch.cat(alts)
    print(f"    height above floor: median {quantiles(aall)[0.5]:.2f} m", flush=True)
    print(f"    the FLOOR is the nearest surface on "
          f"{100 * float(((dall - aall).abs() < 0.05).float().mean()):.1f}% of samples",
          flush=True)
    task.close()
    print("\nALL CHECKS DONE", flush=True)


if __name__ == "__main__":
    main()
