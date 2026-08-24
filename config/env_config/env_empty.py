"""
Environment configuration for simple hover SNN task.
Defines a ground plane, no obstacles.

Identical to simple_hover environment - the SNN is only
a different neural network architecture, not a different environment.
"""

from aerial_gym import AERIAL_GYM_DIRECTORY
import numpy as np

class EmptyEnvNewCfg:
    """
    Simple hover SNN environment configuration.
    - Ground plane only, no obstacles
    - Environment boundaries
    - Reset on crash or out-of-bounds enabled
    """

    class env:
        # num_envs intentionally not set here — must be passed explicitly via CLI
                # (task_registry.make_task(num_envs=...)); this env config has no sensible default
                # and env_manager.py will raise AttributeError immediately if it's ever missing.
        num_env_actions = 0     # this is the number of actions handled by the environment
                                # these are the actions that are sent to environment entities
                                # and some of them may be used to control various entities in the environment
                                # e.g. motion of obstacles, etc.
        env_spacing = 5.0  # Matching position_setpoint_task (was 5.0)

        # Environment setup
        create_ground_plane = False  # Matching position_setpoint_task (was True)
        reset_out_of_bounds = True

        # Physics simulation
        num_physics_steps_per_env_step_mean = 3  
        num_physics_steps_per_env_step_std = 1

        # Rendering
        render_viewer_every_n_steps = 1
        use_warp = False     # Rendering for camera. Use warp if going to use camera images for RL

        manual_camera_trigger = False # if true, camera rendering is triggered manually from task

        # Collision handling
        reset_on_collision = True
        collision_force_threshold = 0.010  # Matching position_setpoint_task (was 0.05)

        sample_timestep_for_latency = True
        perturb_observations = True
        keep_same_env_for_num_episodes = 1
        write_to_sim_at_every_timestep = False

        # Environment bounds (matching position_setpoint_task)
        e_s = env_spacing
        lower_bound_min = [-e_s, -e_s, -e_s]  # Matching position_setpoint_task (was [-e_s, -e_s, 0.0])
        lower_bound_max = [-e_s, -e_s, -e_s]  # Matching position_setpoint_task (was [-e_s, -e_s, 0.0])
        upper_bound_min = [e_s, e_s, e_s]
        upper_bound_max = [e_s, e_s, e_s]

    class env_config:
        # No assets (matching position_setpoint_task - was bottom_wall)
        include_asset_type = {}
        asset_type_to_dict_map = {}
