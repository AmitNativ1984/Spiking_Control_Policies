"""
Environment configuration for navigation with obstacles.
Uses dense obstacle configs matching the VAE training distribution.
Randomized environment bounds: L×W×H in [8,12]×[5,8]×[4,6].
"""
from aerial_gym.config.asset_config.env_object_config import (
    bottom_wall,
    left_wall,
    right_wall,
    back_wall,
    front_wall,
    top_wall,
)
from data_generation.config.env_config import (
    dense_panel_params,
    dense_object_params,
    dense_thin_params,
    dense_tree_params,
)


class NavigationObstacleEnvCfg:
    """
    Navigation environment with dense obstacles and randomized bounds.
    - Dense obstacle configs matching VAE training distribution
    - Randomized box-shaped environments: L×W×H in [8,12]×[5,8]×[4,6]
    - Reset on collision enabled
    """

    class env:
        num_envs = 64  # Overridden by task config
        num_env_actions = 4
        env_spacing = 5.0

        # Physics simulation
        # simulation dt is fixed at 0.01s in the sim config; (can be changed)
        num_physics_steps_per_env_step_mean = 10 # 100msec = 10Hz control policy rate
        num_physics_steps_per_env_step_std = 1

        # Rendering
        render_viewer_every_n_steps = 1

        # Collision handling
        reset_on_collision = True
        collision_force_threshold = 0.05  # Newtons

        # Environment setup
        create_ground_plane = False
        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 10
        write_to_sim_at_every_timestep = False

        use_warp = True

        # Environment bounds — randomized to produce L×W×H in [8,12]×[5,8]×[4,6]
        #
        # X-axis (length 8-12m):
        #   lower_x fixed at -1.0, upper_x ranges from 7.0 to 11.0
        #   => total X range: 8.0 to 12.0
        #
        # Y-axis (width 5-8m, symmetric):
        #   lower_y ranges from -4.0 to -2.5, upper_y ranges from 2.5 to 4.0
        #   => total Y range: 5.0 to 8.0
        #
        # Z-axis (height 4-6m, symmetric):
        #   lower_z ranges from -3.0 to -2.0, upper_z ranges from 2.0 to 3.0
        #   => total Z range: 4.0 to 6.0
        lower_bound_min = [-1.0, -4.0, -3.0]
        lower_bound_max = [-1.0, -2.5, -2.0]
        upper_bound_min = [7.0, 2.5, 2.0]
        upper_bound_max = [11.0, 4.0, 3.0]

    class env_config:
        # Enable all dense obstacle types + bottom wall
        include_asset_type = {
            "panels": True,
            "objects": True,
            "thin": True,
            "trees": True,
            "left_wall": False,
            "right_wall": False,
            "back_wall": False,
            "front_wall": False,
            "top_wall": False,
            "bottom_wall": True,
        }

        # Map asset type names to dense configuration classes
        asset_type_to_dict_map = {
            "panels": dense_panel_params,
            "objects": dense_object_params,
            "thin": dense_thin_params,
            "trees": dense_tree_params,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "back_wall": back_wall,
            "front_wall": front_wall,
            "bottom_wall": bottom_wall,
            "top_wall": top_wall,
        }
