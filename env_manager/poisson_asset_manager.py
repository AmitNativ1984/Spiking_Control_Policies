"""Poisson-process obstacle placement for the F450 navigation task.

Replaces the upstream :class:`AssetManager`, which places a fixed number of obstacles
(= curriculum level) inside per-asset ratio boxes. Those boxes confine every obstacle
class to x-ratio [0.30, 0.85] -- a leftover from aerial_gym's low-x spawn / high-x goal
layout -- which leaves a wide free corridor in front of the -x wall and none in +/-y. With
targets now on all four vertical walls that asymmetry is directly learnable ("-x is
cheap"), so the policy could succeed without reading the depth image.

A homogeneous Poisson point process over the whole env box is isotropic by construction
and removes the bias outright.

Isaac Gym constraint
--------------------
Actors cannot be created after ``prepare_sim()``: ``create_actor`` runs during env
construction and ``acquire_actor_root_state_tensor`` then allocates one fixed
``(num_envs, num_assets_per_env, 13)`` tensor. That does NOT force a fixed obstacle count.
Positions are freely resampled every reset, and the number of obstacles actually inside
the env is controlled by parking the unused ones at -1000 (upstream already does this).
The preallocated pool only bounds N, so it just has to be large enough that the Poisson
draw rarely saturates it -- see config/asset_config/enlarged_object_config.py for sizing.

Injected in place of the upstream manager by the task (SimBuilder.build_env returns the
EnvManager, and asset_manager is a plain attribute on it), so no aerial_gym source is
modified.
"""

import math

import torch

from aerial_gym.env_manager.asset_manager import AssetManager
from aerial_gym.utils.logging import CustomLogger
from aerial_gym.utils.math import (
    quat_from_euler_xyz_tensor,
    torch_interpolate_ratio,
    torch_rand_float_tensor,
)

logger = CustomLogger("poisson_asset_manager")


class PoissonAssetManager(AssetManager):
    """Homogeneous Poisson point process over the env box, thinned by a keep-out
    ellipsoid around the env centre (where the drone spawns).

    Structural assets (the ground plane and any perimeter walls) are NOT part of the
    process. asset_loader.select_and_order_assets() appendleft()s every keep_in_env asset,
    so they occupy slots [0:num_keep_in_env]; those slots keep the upstream ratio-box
    placement, are always present, and are excluded from the Poisson count. Only slots
    [num_keep_in_env:] are obstacles. Anything that should be randomly placed and cullable
    must therefore be keep_in_env=False -- see config/asset_config/enlarged_object_config.py,
    which flips panels and trees over for exactly this reason.

    Within the obstacle slots, an asset that PINS a position axis in its ratio config
    (min_state_ratio == max_state_ratio on that axis) keeps that axis; only the free axes
    are sampled. Trees pin z = 0.0 and so stay rooted on the floor while the process
    scatters them in x/y.

    The caller must set, after construction:
        spawn_ratio_lo / spawn_ratio_hi : (3,) tensors, the drone's spawn box in ratio
                                          units -- read straight from the robot config so
                                          the spawn box has a single source of truth.
        clearance                       : float, metres added to the spawn-box
                                          half-extents to form the keep-out ellipsoid
                                          (max obstacle radius + drone radius).
    """

    def init_tensors(self, global_tensor_dict, num_keep_in_env):
        super().init_tensors(global_tensor_dict, num_keep_in_env)
        # Retained so reset_idx can read the intensity the task publishes each curriculum
        # update. This keeps the upstream env_manager.reset_idx call signature untouched.
        self._global_tensor_dict = global_tensor_dict
        # Upstream AssetManager has no self.device; derive it from the state tensor.
        self.device = self.env_asset_state_tensor.device
        self.spawn_ratio_lo = None
        self.spawn_ratio_hi = None
        self.clearance = 0.0
        # (num_envs, num_assets, 3) — True where the asset config pins that position axis
        # to a single value. Computed once: the ratios are fixed at load time.
        self._pinned_axes = (
            self.asset_min_state_ratio[..., 0:3] == self.asset_max_state_ratio[..., 0:3]
        )

    def reset_idx(self, env_ids, num_obstacles_per_env=0):
        """Resample obstacle poses for ``env_ids``.

        ``num_obstacles_per_env`` is accepted for signature compatibility with the
        upstream caller (env_manager.reset_idx) and deliberately ignored: the count is now
        a Poisson draw, not a curriculum level.
        """
        if len(env_ids) == 0:
            return

        intensity = self._global_tensor_dict.get("obstacle_intensity", 0.0)
        num_resets = len(env_ids)
        num_slots = self.env_asset_state_tensor.shape[1]
        num_keep = self.num_keep_in_env
        num_free = num_slots - num_keep

        # env_bounds_{min,max} were expanded over the asset axis in init_tensors; take
        # column 0 back to per-env (n, 3).
        lower = self.env_bounds_min[env_ids, 0, :]
        upper = self.env_bounds_max[env_ids, 0, :]
        centre = 0.5 * (lower + upper)
        extent = upper - lower

        sampled_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )

        # 0. Structural assets (ground plane, perimeter walls) keep the upstream ratio-box
        #    placement: bottom_wall's ratio is a fixed [0.5, 0.5, 0.0], so it lands on the
        #    floor, centred, every reset. They are never sampled by the process below and
        #    never culled -- the floor must not move and must not vanish at density 0.
        if num_keep > 0:
            self.env_asset_state_tensor[env_ids, :num_keep, 0:3] = torch_interpolate_ratio(
                min=self.env_bounds_min,
                max=self.env_bounds_max,
                ratio=sampled_ratio[..., 0:3],
            )[env_ids, :num_keep, 0:3]

        if num_free == 0:
            self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
                sampled_ratio[env_ids, :, 3:6]
            )
            return

        # 1. Candidate positions, uniform over the whole box -> isotropic by construction.
        positions = lower.unsqueeze(1) + extent.unsqueeze(1) * torch.rand(
            num_resets, num_free, 3, device=self.device
        )

        # 1b. Honour the axes the asset config PINS (min_state_ratio == max_state_ratio on
        #     that axis) and only sample the free ones. Trees pin z = 0.0: they stand on
        #     the ground, and sampling their z would leave them floating in mid-air.
        #     Panels, objects, spheres and cylinders pin nothing and stay fully 3D, which
        #     is the free-floating field this task was calibrated on. No new config knob:
        #     a degenerate [min, max] range on an axis already means "this one is fixed".
        pinned = self._pinned_axes[env_ids, num_keep:, :]
        anchored = torch_interpolate_ratio(
            min=self.env_bounds_min, max=self.env_bounds_max, ratio=sampled_ratio[..., 0:3]
        )[env_ids, num_keep:, 0:3]
        positions = torch.where(pinned, anchored, positions)

        # 2. Keep-out ellipsoid, sized per env from that env's OWN bounds so it tracks the
        #    randomized box exactly. A sphere fits the spawn box badly once the z range is
        #    widened for elevation randomization (it would need R ~= 2.0 m and remove
        #    ~34 m^3); the ellipsoid removes ~15 m^3 for the same guarantee.
        half_extent = 0.5 * (self.spawn_ratio_hi - self.spawn_ratio_lo) * extent
        semi_axes = half_extent + self.clearance

        # 3. Thin the process: drop candidates inside the ellipsoid. Reject-and-drop, NOT
        #    push-to-surface -- pushing would pile up a density spike on the shell.
        #
        #    Pinned axes are dropped from the distance. A ground-anchored tree sits at
        #    z = floor, metres below the spawn altitude, so the 3D test would clear every
        #    tree -- including one standing directly under the drone, whose trunk and
        #    canopy run straight through the spawn point. Zeroing the pinned axes turns
        #    the test into an elliptical COLUMN in x/y for those assets, which is the
        #    separation that actually holds along the axis the asset extends over.
        #
        #    An asset with NO free axis (the side walls, pinned to a boundary face) has
        #    nothing sampled, so there is nothing to reject: its position is exactly where
        #    the config author put it. Without this it would score distance 0, read as
        #    "inside the keep-out", and be culled on every single reset.
        delta = (positions - centre.unsqueeze(1)) / semi_axes.unsqueeze(1)
        free = ~pinned
        valid = (delta * free.float()).norm(dim=2) >= 1.0
        valid |= ~free.any(dim=2)

        # 4. Per-env obstacle count from that env's own free volume. The floor is not
        #    clutter, so the count covers the obstacle slots only and its lower bound is 0:
        #    at curriculum level 0 the intensity is 0, the draw is 0, and the env is empty
        #    apart from the structural assets placed in step 0.
        #
        #    The upper bound is the number of VALID candidates, not num_free. Invalid ones
        #    are ranked last by step 5 and would otherwise survive whenever the draw
        #    exceeds the valid count -- putting an obstacle inside the spawn keep-out.
        free_volume = extent.prod(dim=1) - (4.0 / 3.0) * math.pi * semi_axes.prod(dim=1)
        counts = torch.poisson(intensity * free_volume.clamp(min=0.0)).clamp_(
            min=0.0, max=float(num_free)
        )
        counts = torch.minimum(counts, valid.sum(dim=1).float())

        # 5. Random-subset cull. argsort of uniform noise is a random permutation per env,
        #    so the surviving set is a random mixture of asset TYPES rather than the slot
        #    order frozen at load time (asset_loader.py shuffles once, per env, at
        #    startup -- a suffix cull would then give each env the same mixture forever).
        #    Adding 1.0 to invalid candidates ranks them last so they are always culled.
        noise = torch.rand(num_resets, num_free, device=self.device) + (~valid).float()
        rank = noise.argsort(dim=1).argsort(dim=1)
        cull = rank >= counts.unsqueeze(1)
        positions[cull] = -1000.0

        self.env_asset_state_tensor[env_ids, num_keep:, 0:3] = positions
        # Orientations still come from the per-asset ratio config for every slot; only the
        # position entries [0:3] are bypassed, and only for the obstacle slots.
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_ratio[env_ids, :, 3:6]
        )
