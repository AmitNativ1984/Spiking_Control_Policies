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
from aerial_gym.utils.math import quat_from_euler_xyz_tensor, torch_rand_float_tensor

logger = CustomLogger("poisson_asset_manager")


class PoissonAssetManager(AssetManager):
    """Homogeneous Poisson point process over the env box, thinned by a keep-out
    ellipsoid around the env centre (where the drone spawns).

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

        # env_bounds_{min,max} were expanded over the asset axis in init_tensors; take
        # column 0 back to per-env (n, 3).
        lower = self.env_bounds_min[env_ids, 0, :]
        upper = self.env_bounds_max[env_ids, 0, :]
        centre = 0.5 * (lower + upper)
        extent = upper - lower

        # 1. Candidate positions, uniform over the whole box -> isotropic by construction.
        positions = lower.unsqueeze(1) + extent.unsqueeze(1) * torch.rand(
            num_resets, num_slots, 3, device=self.device
        )

        # 2. Keep-out ellipsoid, sized per env from that env's OWN bounds so it tracks the
        #    randomized box exactly. A sphere fits the spawn box badly once the z range is
        #    widened for elevation randomization (it would need R ~= 2.0 m and remove
        #    ~34 m^3); the ellipsoid removes ~15 m^3 for the same guarantee.
        half_extent = 0.5 * (self.spawn_ratio_hi - self.spawn_ratio_lo) * extent
        semi_axes = half_extent + self.clearance

        # 3. Thin the process: drop candidates inside the ellipsoid. Reject-and-drop, NOT
        #    push-to-surface -- pushing would pile up a density spike on the shell.
        valid = (
            (positions - centre.unsqueeze(1)) / semi_axes.unsqueeze(1)
        ).norm(dim=2) >= 1.0

        # 3b. Move the valid candidates to the front. The keep_in_env slots (step 5) are
        #     present unconditionally, so without this a panel or the tree could be placed
        #     inside the keep-out region and land on the drone at spawn. Candidates are
        #     i.i.d. uniform, so permuting them introduces no bias, and a stable sort keeps
        #     the ordering among valid candidates random. ~97% of slots are valid, so there
        #     are effectively always enough to cover the handful of kept assets.
        order = (~valid).to(torch.uint8).argsort(dim=1, stable=True)
        positions = positions.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
        valid = valid.gather(1, order)

        # 4. Per-env TOTAL obstacle count from that env's own free volume. Mirrors the
        #    upstream semantics num_obstacles_per_env = max(curriculum_level, keep), with
        #    the Poisson draw replacing curriculum_level: the always-kept assets count
        #    towards the total rather than adding on top of it, which is how the target
        #    density was calibrated against the old level-25 clutter.
        free_volume = extent.prod(dim=1) - (4.0 / 3.0) * math.pi * semi_axes.prod(dim=1)
        counts = torch.poisson(intensity * free_volume.clamp(min=0.0)).clamp_(
            min=float(self.num_keep_in_env), max=float(num_slots)
        )

        # 5. Random-subset cull. argsort of uniform noise is a random permutation per env,
        #    so the surviving set is a random mixture of asset TYPES rather than the slot
        #    order frozen at load time (asset_loader.py shuffles once, per env, at
        #    startup -- a suffix cull would then give each env the same mixture forever).
        #    Adding 1.0 to invalid candidates ranks them last so they are always culled.
        noise = torch.rand(num_resets, num_slots, device=self.device) + (~valid).float()
        #    Assets flagged keep_in_env sit at the front of the ordered list and are always
        #    present (3 panels + 1 tree + the floor, for this env config). Ranking them
        #    first makes them survive any count >= num_keep_in_env, which step 4 guarantees.
        if self.num_keep_in_env > 0:
            noise[:, : self.num_keep_in_env] = -1.0
        rank = noise.argsort(dim=1).argsort(dim=1)
        cull = rank >= counts.unsqueeze(1)
        positions[cull] = -1000.0

        self.env_asset_state_tensor[env_ids, :, 0:3] = positions
        # Orientations still come from the per-asset ratio config; only the position
        # entries [0:3] of min/max_state_ratio are bypassed by the Poisson sampler.
        sampled_ratio = torch_rand_float_tensor(
            self.asset_min_state_ratio, self.asset_max_state_ratio
        )
        self.env_asset_state_tensor[env_ids, :, 3:7] = quat_from_euler_xyz_tensor(
            sampled_ratio[env_ids, :, 3:6]
        )
