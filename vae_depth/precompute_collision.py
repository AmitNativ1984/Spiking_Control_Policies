"""Precompute the collision-encoding targets for a depth dataset.

The target is a deterministic function of the resized depth image, and it commutes with the
crop/flip augmentation (tests/test_collision.py), so it can be built once per dataset
instead of per training step. That is what makes the accurate per-range dilation free at
training time -- on-the-fly it costs about twice a full VAE step.

    python -m vae_depth.precompute_collision --limit 200 --validate 16

Writes <cache_dir>/<same basename>.png as 16-bit depth in the source's units, plus a
manifest.json recording every geometric parameter. The cache directory name encodes those
parameters too, so a stale cache cannot be picked up silently.
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from glob import glob

import cv2
import numpy as np
import torch

from vae_depth.collision import (
    cache_dir_for,
    cache_path_for,
    cache_signature,
    collision_radius,
    collision_target,
    intrinsics,
    reach_px,
    reference_exact,
)
from vae_depth.config import VAEConfig


def load_depth(path, config):
    """16-bit PNG -> z-depth in metres at the target resolution."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"unreadable image: {path}")
    depth = img.astype(np.float32) / 65535.0 * config.depth_scale
    return cv2.resize(depth, (config.target_width, config.target_height),
                      interpolation=cv2.INTER_NEAREST)


def save_target(path, target_m, config):
    """Store in the source's units so one loader reads both."""
    q = np.clip(target_m / config.depth_scale, 0.0, 1.0) * 65535.0
    cv2.imwrite(path, q.round().astype(np.uint16))


def half_res(config):
    """A copy of `config` at half resolution -- same FOV, so fx halves with the width.

    The brute-force reference is O(r^2) in the dilation radius, and at full resolution a
    near-field bucket asks for a radius of a few hundred pixels. Half resolution keeps the
    check affordable while testing the same geometry.
    """
    import copy
    c = copy.copy(config)
    c.target_height = config.target_height // 2
    c.target_width = config.target_width // 2
    return c


def validate(paths, config, n, device):
    """Compare the production builder against the brute-force sphere-sweep.

    `unsafe` is the direction that matters: the fast builder reporting MORE free space than
    the exact geometry allows. Over-conservatism is reported but not gated.
    """
    cfg_v = half_res(config)
    rng = np.random.default_rng(0)
    sample = [paths[i] for i in rng.choice(len(paths), min(n, len(paths)), replace=False)]

    unsafe_max, unsafe_frac, cons, errs = 0.0, [], [], []
    for p in sample:
        d = load_depth(p, cfg_v)
        t = torch.from_numpy(d).to(device)
        fast = collision_target(t, cfg_v)
        ref = reference_exact(t, cfg_v)
        e = (fast - ref).cpu().numpy()
        unsafe_max = max(unsafe_max, float(e.max()))
        unsafe_frac.append(float((e > 0.01).mean()))
        cons.append(float((e < 0).mean()))
        errs.append(float(np.abs(e).mean()))

    return {
        "images": len(sample),
        "resolution": f"{cfg_v.target_height}x{cfg_v.target_width}",
        "max_unsafe_m": round(unsafe_max, 4),
        "mean_unsafe_pixel_fraction": round(float(np.mean(unsafe_frac)), 4),
        "mean_conservative_pixel_fraction": round(float(np.mean(cons)), 4),
        "mean_abs_error_m": round(float(np.mean(errs)), 4),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None, help="cache root; default <data_dir>/../depth-collision")
    ap.add_argument("--limit", type=int, default=None, help="only the first N images")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--validate", type=int, default=16,
                    help="images to check against the brute-force reference (0 to skip)")
    ap.add_argument("--max_unsafe", type=float, default=0.15,
                    help="gate: fail if the fast builder is optimistic by more than this")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config = VAEConfig()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.out:
        config.collision_cache_dir = args.out

    paths = sorted(glob(os.path.join(config.data_dir, f"*.{config.image_format}")))
    paths = [p for p in paths if os.path.getsize(p) > 0]
    if not paths:
        raise FileNotFoundError(f"no images in {config.data_dir}")
    if args.limit:
        paths = paths[:args.limit]

    cache_dir = cache_dir_for(config)
    os.makedirs(cache_dir, exist_ok=True)
    fx, _, _, _ = intrinsics(config)

    print(f"collision target precompute")
    print(f"  source        {config.data_dir}  ({len(paths)} images)")
    print(f"  cache         {cache_dir}")
    print(f"  sphere R      {collision_radius(config):.3f} m "
          f"(= {config.collision_radius_m} + {config.safety_margin_m})")
    print(f"  clamp         [{config.near_floor_m}, {config.max_depth_m}] m")
    print(f"  fx            {fx:.1f} px at {config.target_height}x{config.target_width}")
    print(f"  buckets       {config.collision_buckets} x {config.collision_shells} shells")
    print(f"  radius        " + ", ".join(
        f"{d} m: {reach_px(d, config)} px" for d in (0.5, 1, 3, 7)))
    print()

    report = None
    if args.validate:
        print(f"validating against the brute-force sphere-sweep "
              f"({args.validate} images, half resolution)...")
        t0 = time.time()
        report = validate(paths, config, args.validate, args.device)
        print(f"  max unsafe          {report['max_unsafe_m']:+.4f} m  "
              f"(gate {args.max_unsafe:+.3f})")
        print(f"  unsafe pixels       {100*report['mean_unsafe_pixel_fraction']:.2f}%")
        print(f"  conservative pixels {100*report['mean_conservative_pixel_fraction']:.1f}%")
        print(f"  mean |error|        {report['mean_abs_error_m']:.3f} m")
        print(f"  took {time.time()-t0:.1f} s\n")
        if report["max_unsafe_m"] > args.max_unsafe:
            raise SystemExit(
                f"VALIDATION FAILED: the target claims up to {report['max_unsafe_m']:.3f} m "
                f"more free space than the exact geometry allows (gate {args.max_unsafe}). "
                "Nothing was written.")

    todo = paths if args.overwrite else [
        p for p in paths if not os.path.exists(cache_path_for(p, config, cache_dir))]
    print(f"building {len(todo)} targets ({len(paths)-len(todo)} already cached)...")

    t0 = time.time()
    written = 0
    for i in range(0, len(todo), args.batch_size):
        chunk = todo[i:i + args.batch_size]
        batch = torch.from_numpy(np.stack([load_depth(p, config) for p in chunk])).to(args.device)
        targets = collision_target(batch, config).cpu().numpy()
        for p, t in zip(chunk, targets):
            save_target(cache_path_for(p, config, cache_dir), t, config)
            written += 1
        if i % (args.batch_size * 8) == 0 and i:
            rate = written / (time.time() - t0)
            print(f"  {written}/{len(todo)}  {rate:.0f} img/s  "
                  f"eta {(len(todo)-written)/max(rate,1e-6)/60:.1f} min")

    elapsed = time.time() - t0
    size_mb = sum(os.path.getsize(os.path.join(cache_dir, f))
                  for f in os.listdir(cache_dir) if f.endswith(".png")) / 1e6

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "signature": cache_signature(config),
        "source_dir": config.data_dir,
        "n_source_images": len(paths),
        "n_cached": len([f for f in os.listdir(cache_dir) if f.endswith(".png")]),
        "cache_size_mb": round(size_mb, 1),
        "geometry": {
            "collision_radius_m": config.collision_radius_m,
            "safety_margin_m": config.safety_margin_m,
            "sphere_radius_m": collision_radius(config),
            "near_floor_m": config.near_floor_m,
            "min_depth_m": config.min_depth_m,
            "max_depth_m": config.max_depth_m,
            "hfov_deg": config.hfov_deg,
            "fx_px": round(fx, 2),
            "target_hw": [config.target_height, config.target_width],
            "depth_scale": config.depth_scale,
            "buckets": config.collision_buckets,
            "shells": config.collision_shells,
            "algo_version": config.collision_algo_version,
        },
        "validation": report,
        "build_seconds": round(elapsed, 1),
    }
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nwrote {written} targets in {elapsed:.1f} s "
          f"({written/max(elapsed,1e-6):.0f} img/s), cache {size_mb:.0f} MB")
    print(f"manifest: {os.path.join(cache_dir, 'manifest.json')}")
    if len(paths) < 43087:
        full = 43087 * elapsed / max(written, 1) / 60
        print(f"extrapolated to the full 43,087-image dataset: "
              f"{full:.0f} min, {43087 * size_mb / max(written,1) / 1000:.1f} GB")


if __name__ == "__main__":
    main()
