"""Collision-encoding target: geometry, safety direction, and the cache's premise.

Every failure mode here is silent -- a wrong radius or a missing longitudinal term still
produces a plausible-looking depth map -- so these check numbers against the brute-force
sphere-sweep rather than eyeballing output.
"""
import math

import numpy as np
import pytest
import torch

from vae_depth.collision import (
    cache_signature,
    collision_radius,
    collision_target,
    intrinsics,
    reach_px,
    reference_exact,
)
from vae_depth.config import VAEConfig


@pytest.fixture(scope="module")
def cfg():
    """Small frame so the brute-force reference stays tractable; the geometry is
    resolution-independent because fx follows from the width."""
    c = VAEConfig()
    c.target_height, c.target_width = 90, 160
    return c


@pytest.fixture(scope="module")
def scene(cfg):
    """Three posts at different ranges on a far background."""
    d = np.full((cfg.target_height, cfg.target_width), 7.0, dtype=np.float32)
    d[20:70, 30:45] = 1.2
    d[10:80, 70:85] = 2.5
    d[30:60, 110:150] = 5.0
    return torch.from_numpy(d)


def test_intrinsics_match_the_configured_fov(cfg):
    fx, fy, cx, cy = intrinsics(cfg)
    assert fx == pytest.approx(fy)
    hfov = 2 * math.degrees(math.atan((cfg.target_width / 2) / fx))
    assert hfov == pytest.approx(cfg.hfov_deg)
    assert (cx, cy) == pytest.approx(((cfg.target_width - 1) / 2, (cfg.target_height - 1) / 2))


def test_radius_matches_the_f450_airframe(cfg):
    """Two independent errors used to live here: the sphere was half the drone radius, and
    the drone radius itself was half the 450 mm wheelbase -- which is motor-to-motor and
    excludes the props. The airframe's own geometry is the authority.
    """
    hub = 0.162635 * math.sqrt(2)          # prop hub distance from base_link, per the URDF
    prop = 0.127                           # prop disc radius, per the URDF
    assert cfg.collision_radius_m == pytest.approx(hub + prop, abs=1e-3)
    assert collision_radius(cfg) == pytest.approx(cfg.collision_radius_m + cfg.safety_margin_m)


def test_near_floor_clears_the_sphere_with_headroom(cfg):
    """reach_px divides by sqrt(d^2 - R^2); a floor grazing R makes every near bucket the
    whole frame."""
    assert cfg.near_floor_m > collision_radius(cfg) * 1.3


def test_reach_scales_inversely_with_range(cfg):
    """The whole point of the change: a fixed kernel is the bug."""
    r1, r3, r7 = reach_px(1.0, cfg), reach_px(3.0, cfg), reach_px(7.0, cfg)
    assert r1 > r3 > r7
    # Away from the sphere radius the relation is very nearly 1/d.
    assert r3 / r7 == pytest.approx(7.0 / 3.0, rel=0.1)


def test_reach_is_bounded_below_the_near_floor(cfg):
    """Depths below the floor must not produce an unbounded radius."""
    big = max(cfg.target_height, cfg.target_width)
    assert reach_px(0.01, cfg) <= big
    assert reach_px(cfg.near_floor_m, cfg) == reach_px(cfg.near_floor_m / 10, cfg)


def test_near_floor_must_exceed_the_sphere(cfg):
    bad = VAEConfig()
    bad.near_floor_m = 0.3          # below collision_radius_m + safety_margin_m = 0.375
    with pytest.raises(ValueError, match="near_floor_m"):
        bad.__post_init__()


def test_target_never_exceeds_the_input(cfg, scene):
    """Dilation only ever moves surfaces closer. A target farther than the raw depth would
    mean the encoding invented free space."""
    out = collision_target(scene, cfg)
    assert bool((out <= scene.clamp(cfg.min_depth_m, cfg.max_depth_m) + 1e-5).all())


def test_surface_is_pulled_toward_the_camera(cfg, scene):
    """The longitudinal term a flat min-pool cannot produce: on-axis the contact happens R
    in front of the surface."""
    out = collision_target(scene, cfg)
    interior = out[40:55, 33:42]                       # well inside the 1.2 m post
    assert float(interior.max()) <= 1.2 - collision_radius(cfg) + 0.05


def test_matches_reference_within_tolerance(cfg, scene):
    """Against the brute-force sphere-sweep. `unsafe` means the fast builder claims more
    free space than exists -- that is the direction that matters."""
    fast = collision_target(scene, cfg)
    ref = reference_exact(scene, cfg)
    err = (fast - ref).numpy()
    assert err.max() <= 0.15, f"max unsafe {err.max():.3f} m"
    assert np.abs(err).mean() <= 0.75


def test_far_field_reduces_to_a_constant_pull_in(cfg):
    """With nothing within reach, the target is the surface minus R -- no lateral effect."""
    flat = torch.full((cfg.target_height, cfg.target_width), 6.0)
    out = collision_target(flat, cfg)
    assert float(out.mean()) == pytest.approx(6.0 - collision_radius(cfg), abs=0.02)


def test_cached_path_is_never_less_conservative_than_cropping_first(cfg, scene):
    """The cache's premise: the target may be computed ONCE, before augmentation.

    A crop-and-zoom by s scales both the obstacle and its required radius by 1/s, so the
    two orders agree on the geometry. They are not identical, and the difference runs one
    way only: dilating first keeps the influence of obstacles the crop later removes, while
    cropping first discards them. The cached path is therefore the MORE correct of the two,
    and what must hold is that it never reports more free space than cropping first.
    """
    import cv2

    H, W = cfg.target_height, cfg.target_width
    s, top, left = 0.8, 6, 14
    ch, cw = int(H * s), int(W * s)

    def crop(img):
        c = img[top:top + ch, left:left + cw]
        return cv2.resize(c, (W, H), interpolation=cv2.INTER_NEAREST)

    zoomed = VAEConfig()
    zoomed.target_height, zoomed.target_width = H, W
    # Cropping and zooming by s multiplies the effective focal length by 1/s, which for a
    # fixed sensor width is the same as narrowing the FOV.
    zoomed.hfov_deg = 2 * math.degrees(math.atan(math.tan(math.radians(cfg.hfov_deg) / 2) * s))

    dilate_then_crop = crop(collision_target(scene, cfg).numpy())
    crop_then_dilate = collision_target(torch.from_numpy(crop(scene.numpy())), zoomed).numpy()

    unsafe = dilate_then_crop - crop_then_dilate      # >0 = cached claims MORE free space
    assert unsafe.max() <= 0.02, f"cached target is less conservative by {unsafe.max():.3f} m"


def test_cache_signature_tracks_every_geometric_parameter(cfg):
    base = cache_signature(cfg)
    for field, value in [("collision_radius_m", 0.3), ("near_floor_m", 1.0),
                         ("max_depth_m", 8.0), ("hfov_deg", 70.0),
                         ("collision_algo_version", 2)]:
        other = VAEConfig()
        other.target_height, other.target_width = cfg.target_height, cfg.target_width
        setattr(other, field, value)
        assert cache_signature(other) != base, f"{field} does not invalidate the cache"
