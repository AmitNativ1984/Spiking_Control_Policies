"""Collision-encoding target for the depth VAE (Deep Collision Encoding).

The decoder target is not a depth image. It is the Minkowski sum of the obstacle set with
the drone's collision sphere, re-rendered as *the depth at which the drone's centre first
touches something along this ray*:

    t_contact(u) = u.p - sqrt(R^2 - |p|^2 + (u.p)^2)     for rays passing within R of p
    target(u,v)  = min over obstacle points p of  t_contact * u_z      (back to z-depth)

The previous target used a single min-pool with a kernel calibrated at one range, which
misses three terms, all in the unsafe direction:

  1. lateral reach goes as 1/d  -- a fixed kernel encodes 0.05 m of margin at 1 m and
     0.33 m at 7 m, when the drone needs a constant 0.375 m;
  2. off-axis reach is slightly larger than fx*R/d, because the perpendicular distance to
     a tilted ray is shorter than the lateral distance at constant z (measured: 29 px vs
     27.7 px predicted at the corner of an 87-degree camera, hence REACH_SAFETY below);
  3. the surface is also pulled TOWARD the camera by up to R -- a flat min-pool can never
     do this, so it under-reports collision distance by a constant 0.375 m at every range.

`collision_target` is the production builder: bucket by source depth, dilate each bucket
with its own radius, min across buckets. The radius must come from the SOURCE (obstacle)
pixel -- dilation is a scatter, not a gather -- which is what the bucketing buys.
`reference_exact` is the brute-force ground truth, for tests and the precompute's
validation gate only; it is orders of magnitude slower.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# Extra margin on the CALIBRATED reach (see calibrate_reach). Small, because the reach is
# measured rather than modelled; it only absorbs the coarseness of the calibration grid.
REACH_SAFETY = 1.05

# Source positions sampled when calibrating reach, per axis. The reach of an off-axis
# source is not a scaled copy of an on-axis one, so a single probe position is not enough.
_CALIBRATION_GRID = 7

_reach_cache = {}


def intrinsics(config):
    """(fx, fy, cx, cy) in pixels, for the config's TARGET resolution.

    Square pixels, so fy == fx and the vertical FOV follows from the aspect ratio rather
    than being a free parameter. Principal point uses the integer-pixel-centre convention
    ((w-1)/2), matching OpenCV and every calibration toolbox.
    """
    fx = (config.target_width / 2.0) / math.tan(math.radians(config.hfov_deg) / 2.0)
    return fx, fx, (config.target_width - 1) / 2.0, (config.target_height - 1) / 2.0


def collision_radius(config):
    """Radius of the drone's collision sphere, metres.

    The Minkowski sum needs the FULL drone radius plus the safety margin. The legacy
    kernel used `safety_margin_fraction * drone_radius_m` = 0.125 m, i.e. half the drone
    radius -- 3x too small at every range, independent of the range-dependence bug.
    """
    return config.collision_radius_m + config.safety_margin_m


def _ray_grid(config, device="cpu"):
    fx, fy, cx, cy = intrinsics(config)
    v, u = torch.meshgrid(
        torch.arange(config.target_height, device=device, dtype=torch.float32),
        torch.arange(config.target_width, device=device, dtype=torch.float32),
        indexing="ij")
    ray = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1)
    return ray, ray / ray.norm(dim=-1, keepdim=True)


def _max_ray_norm(config):
    """||ray|| at the image corner: 1 on the optical axis, ~1.47 at the corner of an
    87-degree frame. The bound between lateral offset at constant z and perpendicular
    distance to a tilted ray."""
    ray, _ = _ray_grid(config)
    return float(ray.norm(dim=-1).max())


def reach_px(depth_m, config, fx=None):
    """Pixel reach of an obstacle at `depth_m` -- MEASURED on the ray grid, not modelled.

    The closed form fx*R/sqrt(d^2-R^2) is exact only on the optical axis. Off-axis the
    reach is fx*(tan(theta+alpha) - tan(theta)), which for a near obstacle at a wide angle
    is several times larger -- at 0.5 m the half-angle alpha is 48 degrees, so a source
    near the edge of an 87-degree frame can occlude most of it. A single closed-form factor
    cannot cover that, and guessing one low is exactly how the target ends up claiming free
    space that does not exist.

    So: sweep a coarse grid of source positions, ask the exact contact condition how far it
    actually reaches from each, and take the worst case. Cached per (config, depth).
    """
    key = (cache_signature(config), round(float(depth_m), 4))
    if key in _reach_cache:
        return _reach_cache[key]

    R = collision_radius(config)
    d = max(float(depth_m), config.near_floor_m)
    big = max(config.target_height, config.target_width)
    ray, U = _ray_grid(config)
    H, W = config.target_height, config.target_width

    worst = 0
    ys = np.linspace(0, H - 1, _CALIBRATION_GRID).round().astype(int)
    xs = np.linspace(0, W - 1, _CALIBRATION_GRID).round().astype(int)
    for sy in ys:
        for sx in xs:
            p = ray[sy, sx] * d                       # the obstacle point, optical frame
            dot = (U * p).sum(-1)
            perp2 = (p * p).sum() - dot * dot         # squared distance from p to each ray
            hit = perp2 <= R * R
            if not bool(hit.any()):
                continue
            idx = hit.nonzero()
            reach = max(int((idx[:, 0] - sy).abs().max()), int((idx[:, 1] - sx).abs().max()))
            worst = max(worst, reach)

    r = min(int(math.ceil(REACH_SAFETY * worst)), big)
    _reach_cache[key] = r
    return r


def _minpool_separable(t, k):
    """Min-pool with a square (Chebyshev) element.

    Separable -- 1D along rows then columns -- which is the difference between 75 ms and
    3245 ms per batch at the radii this needs. min = -max(-x).
    """
    if k % 2 == 0:
        k += 1
    t = -F.max_pool2d(-t, (1, k), stride=1, padding=(0, k // 2))
    return -F.max_pool2d(-t, (k, 1), stride=1, padding=(k // 2, 0))


def collision_target(depth_m, config):
    """Collision-encoding target, metres, same shape as `depth_m`.

    Args:
        depth_m: [..., H, W] z-depth in metres. Accepts (H,W), (B,H,W) or (B,1,H,W).
        config: VAEConfig.

    Returns:
        Tensor of the same shape: depth at which the drone's centre first collides.
    """
    squeeze_dims = 4 - depth_m.dim()
    x = depth_m
    for _ in range(max(squeeze_dims, 0)):
        x = x.unsqueeze(0)

    fx, _, _, _ = intrinsics(config)
    R = collision_radius(config)
    H, W = x.shape[-2:]
    big = max(H, W)

    # The clamp mirrors normalize_depth's, so the target and the encoder input agree on
    # what "near" and "far" mean. The near floor also bounds the radius: at d <= R the
    # drone's sphere already contains the point and no finite dilation exists.
    d = x.clamp(config.near_floor_m, config.max_depth_m)

    edges = np.geomspace(config.near_floor_m, config.max_depth_m, config.collision_buckets + 1)
    out = torch.full_like(d, float("inf"))

    for b in range(config.collision_buckets):
        lo, hi = float(edges[b]), float(edges[b + 1])
        last = b == config.collision_buckets - 1
        mask = (d >= lo) & (d <= hi if last else d < hi)
        if not bool(mask.any()):
            continue

        # Pixels outside the bucket are pushed out of reach rather than masked, so the
        # pooling never sees them. The radius comes from the bucket's NEAR edge, i.e. the
        # largest radius any member needs.
        masked = torch.where(mask, d, torch.full_like(d, 1e4))
        r_full = reach_px(lo, config, fx)

        # The hemispherical element -sqrt(R^2 - rho^2) as a staircase of flat shells: each
        # shell carries the drop of its INNER edge (the largest drop it contains), so the
        # approximation errs toward reporting collision earlier.
        # A pixel offset maps to a SMALLER perpendicular distance off-axis (the ray is
        # tilted), by up to 1/||ray||_max. Dividing by that bound keeps rho an
        # under-estimate, hence the drop an over-estimate, hence the target conservative.
        # Using the paraxial rho directly under-applies the drop and the target then claims
        # free space that is not there.
        cos_min = 1.0 / _max_ray_norm(config)

        prev_r = 0
        for level in range(1, config.collision_shells + 1):
            r = min(int(math.ceil(r_full * level / config.collision_shells)), big)
            rho = prev_r * lo / fx * cos_min             # shell's inner edge, metres
            drop = math.sqrt(max(R * R - rho * rho, 0.0))
            if r >= big:
                dilated = masked.amin(dim=(-2, -1), keepdim=True).expand_as(d)
            else:
                dilated = _minpool_separable(masked, 2 * r + 1)
            out = torch.minimum(out, dilated - drop)
            prev_r = r

    # Every pixel sees its own surface at perpendicular distance zero, so the fallback only
    # fires for pixels whose bucket was empty -- keep it consistent with that surface.
    out = torch.where(torch.isfinite(out), out, d - R)
    out = out.clamp(config.min_depth_m, config.max_depth_m)

    for _ in range(max(squeeze_dims, 0)):
        out = out.squeeze(0)
    return out


def reference_exact(depth_m, config, reach_safety=1.6):
    """Brute-force sphere-sweep. Ground truth for tests and validation, far too slow for
    production use (seconds per image, vs milliseconds for `collision_target`).

    Args:
        depth_m: [H, W] or [B, H, W] z-depth in metres.
        reach_safety: search-window inflation. Larger is slower but certain to contain
            every contact; the default is well above the measured requirement (~1.05).
    """
    x = depth_m if depth_m.dim() == 3 else depth_m.unsqueeze(0)
    fx, fy, cx, cy = intrinsics(config)
    R = collision_radius(config)
    B, H, W = x.shape
    dev = x.device

    d = x.clamp(config.near_floor_m, config.max_depth_m)
    u = torch.arange(W, device=dev, dtype=torch.float32)
    v = torch.arange(H, device=dev, dtype=torch.float32)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    ray = torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], dim=-1)
    U = ray / ray.norm(dim=-1, keepdim=True)              # unit ray per pixel
    P = ray.unsqueeze(0) * d.unsqueeze(-1)                # 3D point per pixel
    P2 = (P * P).sum(-1)

    out = torch.full_like(d, float("inf"))
    edges = np.geomspace(config.near_floor_m, config.max_depth_m, config.collision_buckets + 1)

    for b in range(config.collision_buckets):
        lo, hi = float(edges[b]), float(edges[b + 1])
        last = b == config.collision_buckets - 1
        mask = (d >= lo) & (d <= hi if last else d < hi)
        if not bool(mask.any()):
            continue
        r = min(int(math.ceil(reach_safety * reach_px(lo, config, fx))), max(H, W))
        Pm = torch.where(mask.unsqueeze(-1), P, torch.full_like(P, 1e6))
        P2m = torch.where(mask, P2, torch.full_like(P2, 1e12))

        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if i * i + j * j > r * r:
                    continue
                sv0, sv1 = max(0, -i), min(H, H - i)
                su0, su1 = max(0, -j), min(W, W - j)
                if sv0 >= sv1 or su0 >= su1:
                    continue
                Ps = Pm[:, sv0:sv1, su0:su1]
                P2s = P2m[:, sv0:sv1, su0:su1]
                Ud = U[sv0 + i:sv1 + i, su0 + j:su1 + j].unsqueeze(0)
                dot = (Ud * Ps).sum(-1)
                disc = R * R - P2s + dot * dot            # R^2 - (perpendicular distance)^2
                t = dot - torch.sqrt(disc.clamp(min=0))
                z = torch.where(disc >= 0, t * Ud[..., 2], torch.full_like(t, float("inf")))
                sub = out[:, sv0 + i:sv1 + i, su0 + j:su1 + j]
                torch.minimum(sub, z, out=sub)

    out = torch.where(torch.isfinite(out), out, d - R)
    out = out.clamp(config.min_depth_m, config.max_depth_m)
    return out if depth_m.dim() == 3 else out.squeeze(0)


# ---------------------------------------------------------------------------------------
# Cache addressing
# ---------------------------------------------------------------------------------------


def cache_signature(config):
    """Directory name encoding every parameter the cached targets depend on.

    A change to any of these invalidates the cache. Encoding them in the path (rather than
    only in a manifest) means a stale cache cannot be picked up by accident.
    """
    fx, _, _, _ = intrinsics(config)
    return (f"collision_v{config.collision_algo_version}"
            f"_R{collision_radius(config):.3f}"
            f"_f{fx:.1f}"
            f"_n{config.near_floor_m:.2f}"
            f"_x{config.max_depth_m:.1f}"
            f"_{config.target_height}x{config.target_width}")


def cache_dir_for(config):
    root = config.collision_cache_dir or os.path.join(
        os.path.dirname(os.path.normpath(config.data_dir)), "depth-collision")
    return os.path.join(root, cache_signature(config))


def cache_path_for(src_path, config, cache_dir=None):
    cache_dir = cache_dir or cache_dir_for(config)
    return os.path.join(cache_dir, os.path.basename(src_path))
