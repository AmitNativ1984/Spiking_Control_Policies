from aerial_gym.config.asset_config.env_object_config import (
    thin_asset_params,
    tile_asset_params,
    bottom_wall,
    top_wall,
)
from config.asset_config.sphere_cylinder_config import (
    sphere_asset_params,
    cylinder_asset_params,
)
# Pool-size override (objects 35 -> 40) so the Poisson sampler rarely hits the actor
# ceiling, plus panels/trees demoted to keep_in_env=False so they are cullable clutter
# rather than fixtures — the ground plane is the only structural asset.
# The four side walls come from here too: subclassed to keep_in_env=False so they are
# drawn from the Poisson pool and seal their side of the env only some of the time.
from config.asset_config.enlarged_object_config import (
    object_asset_params,
    panel_asset_params,
    tree_asset_params,
    left_wall,
    right_wall,
    front_wall,
    back_wall,
)

import numpy as np


class ForestEnvCfg:
    class env:
        # num_envs intentionally not set here — must be passed explicitly via CLI
        # (task_registry.make_task(num_envs=...)); this env config has no sensible default
        # and env_manager.py will raise AttributeError immediately if it's ever missing.
        num_env_actions = 0  # no dynamically-actuated entities: panels/objects/thin/trees/
        # spheres/cylinders are all fix_base_link=True, positioned once per reset and static.

        num_physics_steps_per_env_step_mean = 3  # number of steps between camera renders mean
        num_physics_steps_per_env_step_std = 1  # number of steps between camera renders std

        render_viewer_every_n_steps = 1  # render the viewer every n steps
        reset_on_collision = (
            True  # reset environment when contact force on quadrotor is above a threshold
        )
        collision_force_threshold = 0.005  # collision force threshold [N]
        create_ground_plane = False  # create a ground plane
        sample_timestep_for_latency = True  # sample the timestep for the latency noise
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False  # write to sim at every timestep

        use_warp = True
        # FIXED 20 x 20 m footprint, centred on x = 4.0, y = 0 -- the env centre the drone
        # spawns at and everything else is calibrated around. The x/y area is deliberately
        # NOT randomized any more: min == max on both axes, so every env has the same
        # footprint. That also makes the footprint exactly match bottom_wall.urdf, which is
        # a fixed 20 x 20 x 0.2 m slab and used to overhang the bounds by metres per side.
        #
        # z IS still randomized (4-6 m of headroom), so per-env volume still varies and the
        # per-env obstacle draw still means something. Targets sit on the four VERTICAL
        # walls, so z buys no path length -- only volume, and therefore obstacle count and
        # actor cost.
        #
        # The footprint grows ~13 x 7.7 -> 20 x 20 m, so volume goes ~550 -> ~2000 m^3 and
        # the longest centre-to-wall run goes ~6.5 -> 10 m. Episode length is unaffected:
        # 800 steps x (dt 0.01 * 3 physics steps) = 24 s, ample for 10 m. Obstacle DENSITY
        # is held at the calibrated 0.067 /m^3, so the actor pool had to grow with the
        # volume -- see config/asset_config/enlarged_object_config.py.
        lower_bound_min = [-6.0, -10.0, -3.0]  # lower bound for the environment space
        lower_bound_max = [-6.0, -10.0, -2.0]  # lower bound for the environment space
        upper_bound_min = [14.0, 10.0, 2.0]  # upper bound for the environment space
        upper_bound_max = [14.0, 10.0, 3.0]  # upper bound for the environment space

    class env_config:
        include_asset_type = {
            "panels": True,
            "objects": True,
            "thin": True,
            "trees": True,
            "spheres": True,
            "cylinders": True,
            # The four SIDE walls are cullable pool members (keep_in_env=False), so each
            # is sealed on ~count/200 of resets rather than always. top_wall stays off:
            # nothing flies out through the ceiling that the bounds check does not catch,
            # and it would only darken every depth image.
            "left_wall": True,
            "right_wall": True,
            "back_wall": True,
            "front_wall": True,
            "top_wall": False,
            "bottom_wall": True,
            # Must be listed explicitly: asset_loader.select_and_order_assets() only skips
            # a type when include_asset_type[type] is False, so anything present in
            # asset_type_to_dict_map but absent here is loaded by default. tile_asset_params
            # points at a "tile_meshes" folder that does not ship with aerial_gym, so
            # omitting this key raises FileNotFoundError before the sim is even built.
            "tiles": False,
        }

        # maps the above names to the classes defining the assets. They can be enabled and disabled above in include_asset_type
        asset_type_to_dict_map = {
            "panels": panel_asset_params,
            "thin": thin_asset_params,
            "trees": tree_asset_params,
            "objects": object_asset_params,
            "spheres": sphere_asset_params,
            "cylinders": cylinder_asset_params,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "back_wall": back_wall,
            "front_wall": front_wall,
            "bottom_wall": bottom_wall,
            "top_wall": top_wall,
            "tiles": tile_asset_params,
        }