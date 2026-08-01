from aerial_gym.config.asset_config.env_object_config import (
    panel_asset_params,
    thin_asset_params,
    tree_asset_params,
    object_asset_params,
    tile_asset_params,
    left_wall,
    right_wall,
    back_wall,
    front_wall,
    bottom_wall,
    top_wall,
)
from config.asset_config.sphere_cylinder_config import (
    sphere_asset_params,
    cylinder_asset_params,
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
        lower_bound_min = [-2.0, -4.0, -3.0]  # lower bound for the environment space
        lower_bound_max = [-1.0, -2.5, -2.0]  # lower bound for the environment space
        upper_bound_min = [9.0, 2.5, 2.0]  # upper bound for the environment space
        upper_bound_max = [10.0, 4.0, 3.0]  # upper bound for the environment space

    class env_config:
        include_asset_type = {
            "panels": True,
            "objects": True,
            "thin": True,
            "trees": True,
            "spheres": True,
            "cylinders": True,
            "left_wall": False,
            "right_wall": False,
            "back_wall": False,
            "front_wall": False,
            "top_wall": False,
            "bottom_wall": True,
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