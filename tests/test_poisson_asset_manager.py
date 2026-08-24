"""Unit tests for PoissonAssetManager — CPU only, no Isaac Gym sim.

The manager only ever touches the tensors handed to it through global_tensor_dict, so it
can be exercised against synthetic ones. That keeps the two properties these tests guard
cheap enough to run everywhere:

    * structural assets (slots [0:num_keep_in_env] — the ground plane and any perimeter
      walls) are placed from their ratio config and are never moved or culled by the
      process, at any density including zero;
    * the obstacle count floors at 0, so curriculum level 0 is an empty world.
"""
import math
import types

import pytest
import torch

from env_manager.poisson_asset_manager import PoissonAssetManager
from task.attitude_navigation_task import NavigationWithObstaclesTask

NUM_ENVS = 8
NUM_SLOTS = 20
NUM_KEEP = 1  # one structural asset: the ground plane

# The floor's upstream ratio: centred in x/y, flat on the lower z bound.
FLOOR_RATIO = [0.5, 0.5, 0.0]
LOWER = [-3.0, -4.0, -2.0]
UPPER = [10.0, 4.0, 3.0]


def make_manager(intensity):
    """A manager wired to synthetic tensors, one structural slot, the rest obstacles."""
    state = torch.zeros(NUM_ENVS, NUM_SLOTS, 13)

    ratio_lo = torch.zeros(NUM_ENVS, NUM_SLOTS, 13)
    ratio_hi = torch.zeros(NUM_ENVS, NUM_SLOTS, 13)
    for ratio in (ratio_lo, ratio_hi):
        ratio[:, :NUM_KEEP, 0:3] = torch.tensor(FLOOR_RATIO)
    # Obstacle ratio positions are deliberately absurd — the process must ignore them.
    # They must still be a genuine RANGE: a degenerate min == max means "pinned axis"
    # (how trees stay on the ground), which is not what these slots are testing.
    ratio_lo[:, NUM_KEEP:, 0:3] = 0.42
    ratio_hi[:, NUM_KEEP:, 0:3] = 0.43

    tensor_dict = {
        "env_asset_state_tensor": state,
        "asset_min_state_ratio": ratio_lo,
        "asset_max_state_ratio": ratio_hi,
        "env_bounds_min": torch.tensor(LOWER).expand(NUM_ENVS, 3).contiguous(),
        "env_bounds_max": torch.tensor(UPPER).expand(NUM_ENVS, 3).contiguous(),
        "obstacle_intensity": intensity,
    }

    manager = PoissonAssetManager(tensor_dict, NUM_KEEP)
    manager.spawn_ratio_lo = torch.tensor([0.4, 0.4, 0.4])
    manager.spawn_ratio_hi = torch.tensor([0.6, 0.6, 0.6])
    manager.clearance = 0.95
    return manager, state


def expected_floor_position():
    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    return lower + (upper - lower) * torch.tensor(FLOOR_RATIO)


ALL_ENVS = torch.arange(NUM_ENVS)


def test_zero_intensity_gives_an_empty_world():
    """Curriculum level 0 publishes intensity 0. Every obstacle slot must be parked."""
    manager, state = make_manager(0.0)
    manager.reset_idx(ALL_ENVS)

    live = state[:, NUM_KEEP:, 0] > -900
    assert live.sum().item() == 0, "level 0 must place no obstacles"


def test_structural_slot_is_placed_from_its_ratio_config():
    """The floor sits where its ratio says, not where the Poisson process would put it."""
    for intensity in (0.0, 0.067):
        manager, state = make_manager(intensity)
        manager.reset_idx(ALL_ENVS)

        floor = state[:, 0, 0:3]
        assert torch.allclose(floor, expected_floor_position().expand_as(floor), atol=1e-5)


def test_structural_slot_survives_every_reset():
    """It is outside the process, so no draw can cull it to -1000 and none can move it."""
    manager, state = make_manager(0.067)
    for _ in range(50):
        manager.reset_idx(ALL_ENVS)
        floor = state[:, 0, 0:3]
        assert (floor[:, 0] > -900).all(), "the ground plane was culled"
        assert torch.allclose(floor, expected_floor_position().expand_as(floor), atol=1e-5)


def test_obstacles_are_placed_at_high_intensity():
    manager, state = make_manager(0.067)
    manager.reset_idx(ALL_ENVS)

    live = (state[:, NUM_KEEP:, 0] > -900).sum(dim=1)
    assert (live > 0).all(), "max density produced an empty env"
    assert (live <= NUM_SLOTS - NUM_KEEP).all()


def test_no_live_obstacle_lands_inside_the_spawn_keep_out():
    """The count is capped at the number of valid candidates, so an invalid one can never
    be promoted into the surviving set when the draw runs hot."""
    manager, state = make_manager(5.0)  # absurd density: the draw saturates the pool
    manager.reset_idx(ALL_ENVS)

    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    centre = 0.5 * (lower + upper)
    semi_axes = 0.5 * (manager.spawn_ratio_hi - manager.spawn_ratio_lo) * (
        upper - lower
    ) + manager.clearance

    positions = state[:, NUM_KEEP:, 0:3]
    live = positions[..., 0] > -900
    normalized = ((positions - centre) / semi_axes).norm(dim=2)
    assert (normalized[live] >= 1.0).all(), "an obstacle spawned inside the keep-out"


def test_obstacles_span_the_whole_box():
    """The point of the process: no ratio-box corridor, so obstacles reach every wall."""
    manager, state = make_manager(0.067)
    seen_min = torch.full((3,), float("inf"))
    seen_max = torch.full((3,), float("-inf"))
    for _ in range(30):
        manager.reset_idx(ALL_ENVS)
        positions = state[:, NUM_KEEP:, 0:3]
        live = positions[..., 0] > -900
        seen_min = torch.minimum(seen_min, positions[live].min(dim=0).values)
        seen_max = torch.maximum(seen_max, positions[live].max(dim=0).values)

    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    span = upper - lower
    assert ((seen_min - lower) / span < 0.05).all(), "obstacles never reach the low walls"
    assert ((upper - seen_max) / span < 0.05).all(), "obstacles never reach the high walls"


def make_manager_with_anchored_slots(intensity, num_anchored):
    """As make_manager, but the last `num_anchored` obstacle slots pin z = 0.0 the way
    tree_asset_params does — ground-anchored obstacles."""
    manager, state = make_manager(intensity)
    lo = manager.asset_min_state_ratio
    hi = manager.asset_max_state_ratio
    for ratio in (lo, hi):
        ratio[:, -num_anchored:, 0] = 0.1  # x free-ish; overwritten by the process anyway
        ratio[:, -num_anchored:, 1] = 0.1
        ratio[:, -num_anchored:, 2] = 0.0
    hi[:, -num_anchored:, 0] = 0.9  # x and y are genuine ranges, so they stay unpinned
    hi[:, -num_anchored:, 1] = 0.9
    manager._pinned_axes = lo[..., 0:3] == hi[..., 0:3]
    return manager, state


def test_ground_anchored_obstacles_keep_their_pinned_axis():
    """Trees pin z = 0.0. Sampling it would leave them floating in mid-air."""
    manager, state = make_manager_with_anchored_slots(0.067, num_anchored=6)
    manager.reset_idx(ALL_ENVS)

    trees = state[:, -6:, 0:3]
    live = trees[..., 0] > -900
    assert live.any(), "no anchored obstacle survived, test proves nothing"
    assert torch.allclose(trees[..., 2][live], torch.tensor(LOWER[2]), atol=1e-5), (
        "a ground-anchored obstacle left the floor"
    )


def test_ground_anchored_obstacles_still_scatter_in_the_free_axes():
    """Pinning z must not pin x/y — the trees still come from the Poisson process."""
    manager, state = make_manager_with_anchored_slots(0.067, num_anchored=6)
    seen = []
    for _ in range(20):
        manager.reset_idx(ALL_ENVS)
        trees = state[:, -6:, 0:3]
        seen.append(trees[trees[..., 0] > -900][:, 0:2])
    xy = torch.cat(seen)
    assert xy[:, 0].std() > 1.0, "anchored obstacles are not scattering in x"
    assert xy[:, 1].std() > 1.0, "anchored obstacles are not scattering in y"


def test_ground_anchored_obstacles_respect_a_column_keep_out():
    """The 3D ellipsoid test clears every floor-level asset, so a tree could stand
    directly under the spawn point with its canopy through it. Pinned axes drop out of
    the distance, leaving an elliptical column in x/y."""
    manager, state = make_manager_with_anchored_slots(5.0, num_anchored=6)
    manager.reset_idx(ALL_ENVS)

    lower = torch.tensor(LOWER)
    upper = torch.tensor(UPPER)
    centre = 0.5 * (lower + upper)
    semi_axes = 0.5 * (manager.spawn_ratio_hi - manager.spawn_ratio_lo) * (
        upper - lower
    ) + manager.clearance

    trees = state[:, -6:, 0:3]
    live = trees[..., 0] > -900
    horizontal = ((trees[..., 0:2] - centre[0:2]) / semi_axes[0:2]).norm(dim=2)
    assert (horizontal[live] >= 1.0).all(), "a tree stands inside the spawn column"


def test_fully_pinned_obstacles_can_be_placed():
    """The side walls pin all three axes. Nothing is sampled for them, so there is nothing
    for the keep-out to reject — but a zero-length distance vector reads as "inside the
    keep-out", which would cull them on every reset and they would never appear."""
    manager, state = make_manager(0.067)
    lo, hi = manager.asset_min_state_ratio, manager.asset_max_state_ratio
    wall = torch.tensor([0.5, 1.0, 0.5])  # left_wall's ratio: flush on the +y face
    lo[:, -1, 0:3] = wall
    hi[:, -1, 0:3] = wall
    manager._pinned_axes = lo[..., 0:3] == hi[..., 0:3]

    placed = 0
    for _ in range(30):
        manager.reset_idx(ALL_ENVS)
        placed += (state[:, -1, 0] > -900).sum().item()
    assert placed > 0, "a fully pinned obstacle was culled on every single reset"

    manager.reset_idx(ALL_ENVS)
    live = state[:, -1, 0] > -900
    if live.any():
        lower = torch.tensor(LOWER)
        upper = torch.tensor(UPPER)
        expect = lower + (upper - lower) * wall
        assert torch.allclose(state[:, -1, 0:3][live], expect, atol=1e-5), (
            "a pinned wall did not land on its boundary face"
        )


def test_only_the_ground_plane_is_structural():
    """The manager reads keep_in_env as "structural, never culled". Anything else flagged
    keep_in_env is pinned into every env at every level, so level 0 would not be empty —
    upstream flags panels and trees, which config/asset_config/enlarged_object_config.py
    exists to override."""
    from config.env_config.env_forest_with_obstacles import ForestEnvCfg

    cfg = ForestEnvCfg.env_config
    structural = [
        name
        for name, params in cfg.asset_type_to_dict_map.items()
        if cfg.include_asset_type.get(name, True)
        and params.num_assets > 0
        and params.keep_in_env
    ]
    assert structural == ["bottom_wall"], (
        f"only the ground plane may be structural, found {structural}"
    )


@pytest.mark.parametrize("level", [0, 12, 25])
def test_arrive_bonus_is_defined_at_every_pinned_curriculum_level(task_config, level):
    """Pinning the curriculum sets min_level == max_level == N. Scaling the arrival bonus
    on level / max_level then divides by zero at level 0; it must key off the progress
    fraction instead."""
    stub = types.SimpleNamespace(
        task_config=task_config,
        curriculum_progress_fraction=min(
            level / max(task_config.curriculum.density_at_level, 1), 1.0
        ),
    )
    bonus = NavigationWithObstaclesTask._reward_arrive(stub)

    params = task_config.reward_parameters
    assert math.isfinite(bonus)
    assert params["arrive_bonus_min"] <= bonus <= params["arrive_bonus_max"]
