"""Pool-size overrides for the upstream aerial_gym environment assets.

The obstacle field is a Poisson point process (see task/poisson_asset_manager.py), so the
number of obstacles placed in an env each reset is a random draw N ~ Poisson(lambda * V).
Isaac Gym cannot create actors after prepare_sim(), so the actor pool has to be large
enough that N rarely hits the ceiling: pool >= mu_max + 3*sigma_max.

With lambda_max = 0.067 /m^3 and the enlarged env (max volume ~829 m^3, minus a ~17 m^3
keep-out), mu ~= 54 and mu + 3*sigma ~= 77, so the pool needs ~77 CULLABLE slots.

Note that panels, trees and bottom_wall are keep_in_env=True upstream: they sit at the
front of the ordered asset list and are present in every env at every curriculum level,
so they do NOT count towards the cullable pool (this matches the old behaviour, where
asset_manager floored num_obstacles_per_env at num_keep_in_env).

    always kept : panels 3 + trees 1 + bottom_wall 1        =  5
    cullable    : objects 40 + spheres 18 + cylinders 18    = 76
    total slots                                             = 81

Subclassed rather than mutated in place: other tasks in this repo import the same upstream
objects and must keep the original counts.
"""

from aerial_gym.config.asset_config.env_object_config import (
    object_asset_params as _upstream_object_asset_params,
)


class object_asset_params(_upstream_object_asset_params):
    """Upstream objects with a larger pool (35 -> 40) to feed the Poisson sampler.

    NOTE: min/max_state_ratio positions [0:3] are IGNORED by PoissonAssetManager, which
    samples positions over the whole env volume. Only the rotation entries [3:6] are read.
    """

    num_assets = 40
