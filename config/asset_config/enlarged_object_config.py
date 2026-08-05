"""Pool-size and keep_in_env overrides for the upstream aerial_gym environment assets.

The obstacle field is a Poisson point process (see env_manager/poisson_asset_manager.py),
so the number of obstacles placed in an env each reset is a random draw N ~ Poisson(lambda * V).
Isaac Gym cannot create actors after prepare_sim(), so the actor pool has to be large
enough that N rarely hits the ceiling: pool >= mu_max + 3*sigma_max.

With lambda_max = 0.067 /m^3 and the fixed 20 x 20 m env (max volume 2400 m^3 at the top
of the randomized z range, minus a ~29 m^3 keep-out), mu_max ~= 159, sigma ~= 12.6 and
mu + 3*sigma ~= 197, so the pool needs ~197 CULLABLE slots. It is set to 200.

keep_in_env now means "structural": PoissonAssetManager places slots [0:num_keep_in_env]
from their ratio config and never culls them, and every other slot is Poisson clutter.
Upstream flags panels and trees keep_in_env=True, which would pin 3 panels and a tree in
every env at every curriculum level -- level 0 is supposed to be an empty world, so they
are flipped to False here and join the cullable pool. The ground plane keeps the flag: it
must stay on the floor and must survive a zero-density draw.

    structural (bottom_wall)                                =   1
    cullable   : objects 95 + spheres 44 + cylinders 44
                 + panels 7 + trees 6 + side walls 4        = 200
    total slots                                             = 201

Raising a pool size does NOT raise the obstacle count -- that is a single Poisson draw
over all cullable slots, and the cull keeps a uniformly random subset of them. It changes
the MIX: a type's share of the pool is its expected share of the obstacles that appear.

TREES ARE DELIBERATELY RARE (6 slots, 3% of the mix, ~4.8 per env at max density). They
are not point obstacles like the rest. Measured over the 100 shipped meshes: 26 branches
each, mean horizontal reach 4.71 m (max 9.43) and branching up to 7.84 m. In this 20 x 20 m
env that is a ~70 m^2 canopy footprint per tree, and since the flyable z span is only
4-6 m a tree spans the ENTIRE height -- there is no flying over one. At the old 37 slots an
env drew ~30 trees and high curriculum levels were impassable. The density model counts
objects, not occupied space, so it cannot see this: one tree sweeps ~70 m^2 where a 0.6 m
sphere sweeps ~1.1 m^2. The real fix is a separate per-type intensity for trees; this is
the cheap version of it, and 6 is a tuning knob -- raise or lower it freely, but see the
pool note below before doing so.

Cutting a type's slots shrinks the TOTAL pool, which must stay >= mu + 3*sigma or the
Poisson draw clips and the realized density silently falls below the configured one. The
31 slots freed from trees were therefore redistributed proportionally across objects,
spheres, cylinders and panels, holding the total at 200.

The four SIDE walls are in the cullable pool too, so each one is present on a given reset
with probability count/200 -- ~0.8 at max density, 0 at level 0, ramping with the
curriculum. They are pinned to their boundary face on all three axes, so when they survive
the cull they seal that side of the env exactly; the drone terminates at the bounds
regardless, but a wall gives the depth camera something to SEE there instead of empty
space. num_assets is not a frequency knob for them: a wall is fully pinned, so a second
copy would sit inside the first.

Subclassed rather than mutated in place: other tasks in this repo import the same upstream
objects and must keep the original counts and flags.
"""

from aerial_gym.config.asset_config.env_object_config import (
    object_asset_params as _upstream_object_asset_params,
    panel_asset_params as _upstream_panel_asset_params,
    tree_asset_params as _upstream_tree_asset_params,
    left_wall as _upstream_left_wall,
    right_wall as _upstream_right_wall,
    front_wall as _upstream_front_wall,
    back_wall as _upstream_back_wall,
)


class object_asset_params(_upstream_object_asset_params):
    """Upstream objects with a larger pool (35 -> 79) to feed the Poisson sampler.

    NOTE: min/max_state_ratio positions [0:3] are IGNORED by PoissonAssetManager, which
    samples positions over the whole env volume. Only the rotation entries [3:6] are read.
    The folder holds 5 URDFs and asset_loader picks WITH replacement, so a pool far larger
    than the mesh count is fine and always was (35 from 5 upstream).
    """

    num_assets = 95


class panel_asset_params(_upstream_panel_asset_params):
    """Upstream panels, demoted to cullable clutter (keep_in_env True -> False), 3 -> 7.

    Positions come from the Poisson process like any other obstacle; see the note on
    object_asset_params about min/max_state_ratio.
    """

    keep_in_env = False
    num_assets = 7


class _cullable_wall:
    """Mixin: a boundary wall that is drawn from the Poisson pool instead of being a
    permanent fixture, so each side is sealed only some of the time.

    keep_in_env=False is what moves it out of the structural slots. Its position stays
    exactly where upstream put it -- all three axes are pinned (min_state_ratio ==
    max_state_ratio), and PoissonAssetManager honours pinned axes, so the wall lands
    flush on its boundary face rather than somewhere inside the env.
    """

    keep_in_env = False
    num_assets = 1


class left_wall(_cullable_wall, _upstream_left_wall):
    pass


class right_wall(_cullable_wall, _upstream_right_wall):
    pass


class front_wall(_cullable_wall, _upstream_front_wall):
    pass


class back_wall(_cullable_wall, _upstream_back_wall):
    pass


class tree_asset_params(_upstream_tree_asset_params):
    """Upstream trees, demoted to cullable clutter (keep_in_env True -> False) and held to
    a SMALL share of the pool: 6 slots, ~4.8 trees per env at max density.

    A tree is not a point obstacle -- ~70 m^2 of canopy spanning the full flyable height in
    a 400 m^2 env. See the tree paragraph in the module docstring for the measurements and
    for why the pool total has to be held at 200 when this number changes.

    Unlike the other obstacles these are GROUND-ANCHORED: min/max_state_ratio pin z to
    0.0, and PoissonAssetManager honours pinned axes, so a tree keeps its trunk on the
    floor and only its x/y come from the process. 100 tree meshes ship with aerial_gym, so
    even a small pool draws a different set per env.
    """

    keep_in_env = False
    num_assets = 6
