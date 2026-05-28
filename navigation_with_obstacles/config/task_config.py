"""
Task configuration for navigation with obstacles.
Defines observation/action spaces, reward parameters, curriculum, and VAE settings.
"""
import math

class task_config:
    """
    Configuration for NavigationWithObstaclesTask.

    Key features:
    - Acceleration control (accel_x, accel_y, accel_z, yaw_rate)
    - Custom 32D DepthVAE encoding
    - 30-level curriculum (panels then cumulative panels + objects)
    - Randomized environment bounds
    """

    seed = -1

    # Simulation components
    sim_name = "base_sim"
    env_name = "navigation_obstacle_env"
    robot_name = "nav_quadrotor_with_camera"
    controller_name = "lee_acceleration_control"
    args = {}

    # Environment settings
    # NOTE: num_envs is set at runtime from the YAML (env_config.num_envs).
    # The task __init__ assigns task_config.num_envs = num_envs from YAML.
    use_warp = True
    headless = True
    device = "cuda:0"

    # Observation space: 12 (state) + 32 (VAE latents) = 44
    observation_space_dim = 12 + 32
    privileged_observation_space_dim = 0

    # Per-dimension observation bounds for the PopSAN population encoder.
    #
    # Bounds are in the rl_games-normalized space (z-scores, hard-clamped to
    # [-5, 5] by RunningMeanStd when normalize_input=True), NOT raw units.
    # Tune from tools/collect_obs_stats.py if empirically tighter values help.
    #
    # observation_layout is the single source of truth for the 44D vector;
    # observation_bounds is derived from it below. Editing the layout or the
    # per-type bounds is enough — no per-index numbers to maintain.
    observation_layout = [
        (slice(0, 2),   "log_distance"),        # log(d_hor+1), log(|d_vert|+1) — world
        (slice(2, 4),   "bearing_azimuth"),     # cos/sin bearing azimuth — world
        (slice(4, 5),   "elevation_angle"),     # elevation angle to target — world
        (slice(5, 7),   "yaw"),                 # cos/sin drone yaw — world
        (slice(7, 8),   "v_xy"),                # horizontal speed — body
        (slice(8, 9),   "v_z"),                 # vertical speed — body
        (slice(9, 11),  "track_bearing"),       # cos/sin track azimuth — body (masked)
        (slice(11, 12), "track_elevation"),     # track elevation — body (masked)
        (slice(12, 44), "vae_latent"),          # DepthVAE latents
    ]

    observation_type_bounds = {
        "log_distance":    (-3.0, 3.0),
        "bearing_azimuth": (-3.0, 3.0),
        "elevation_angle": (-3.0, 3.0),
        "yaw":             (-3.0, 3.0),
        "v_xy":            (-3.0, 3.0),
        "v_z":             (-3.0, 3.0),
        "track_bearing":   (-3.0, 3.0),
        "track_elevation": (-3.0, 3.0),
        "vae_latent":      (-3.0, 3.0),
    }

    # Expand layout + per-type bounds into a flat per-index list of (min, max).
    # Runs once at class-definition time; consumed by popsan.py via
    # task_config.observation_bounds.
    observation_bounds = [None] * observation_space_dim
    for obj_slice, obj_type in observation_layout:
        lo, hi = observation_type_bounds[obj_type]
        for idx in range(obj_slice.start, obj_slice.stop):
            observation_bounds[idx] = (lo, hi)
    assert all(b is not None for b in observation_bounds), \
        "observation_layout has gaps — every index in [0, observation_space_dim) must be covered"

    # Action space: [accel_x, accel_y, accel_z, yaw_rate]
    action_space_dim = 4

    # Action scaling: network outputs [-1, 1], scaled to physical units
    max_accel = 2.0              # m/s² per axis (symmetric: [-max, +max])
    max_yaw_rate = math.pi / 3  # rad/s (~60 deg/s, symmetric: [-max, +max])
    
    # Speed threshold for excess speed penalty (m/s)
    v_max = 5.0

    # Episode length
    episode_len_steps = 50

    return_state_before_reset = False

    # Out-of-bounds margin multiplier for the exceed check.
    # 1.0 = terminate exactly at env bounds (training default).
    # 1.5 = allow drone to fly 50% beyond bounds before terminating (useful for inference).
    exceed_bounds_margin = 1.0

    # Target waypoint placement (as ratio of environment bounds)
    # Target is placed in the far end of the environment
    target_min_ratio = [0.95, 0.10, 0.10]
    target_max_ratio = [1.00, 0.90, 0.90]

    # Reward parameters
    reward_parameters = {
        # Terminal rewards
        "arrive_bonus_min": 10.0,        # arrival reward at curriculum level 0 (easy)
        "arrive_bonus_max": 15.0,        # arrival reward at max curriculum level (hard)
        "collision_penalty": -15.0,     # obstacle collision termination
        "exceed_penalty": -20.0,        # out-of-bounds termination
        "timeout_penalty": -10.0,          # episode timeout termination
        "d_min": 0.4,                   # arrival distance threshold (meters)
        # Progress reward (dense shaping, all lambda < 0)
        "lambda_d": -0.01,           # distance to target (horizontal + vertical)
        "lambda_dz": -0.01,          # vertical distance to target (encourage altitude adjustments)
        "lambda_v": -0.01,         # velocity-goal direction misalignment
        "lambda_bearing": -0.01,           # projection of velocity onto target direction (encourage movement towards target)
        "lambda_path_deviation": -0.005,    # velocity misalignment with target direction (encourage movement towards target)
        "lambda_jerk": 0.0,      # jerk penalty to encourage smooth control
    }
    
    class vae_config:
        """Custom 32D DepthVAE configuration."""
        use_vae = True
        latent_dims = 32

        # Path to trained DepthVAE checkpoint
        model_file = "/workspaces/aerial_gym_docker/vae_depth/runs/20260218_204641/checkpoints/epoch_150.pth"

        # DepthVAE input resolution
        target_height = 180
        target_width = 320

        # Depth range parameters
        max_depth_m = 7.0
        min_depth_m = 0.1
        sensor_max_range = 10.0

    class curriculum:
        """
        Curriculum configuration — same thresholds as original NavigationTask.
        Levels 0-5: large panels
        Levels 6-30: cumulative panels + small objects
        """
        min_level = 0
        max_level = 25
        check_after_num_rollouts = 16  # curriculum check every N rollouts (instances = num_rollouts * num_envs)
        increase_step = 1                  # slower progression, no double-jumps (was 2)
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

    @staticmethod
    def action_transformation_function(action):
        """
        Transform network output [-1, 1] to acceleration commands.

        Scaling is driven by task_config.max_accel and task_config.max_yaw_rate.
        Input: action tensor (num_envs, 4) in range [-1, 1]
        Output: [accel_x, accel_y, accel_z, yaw_rate] for lee_acceleration_control
        """
        import torch
        clamped_action = torch.clamp(action, -1.0, 1.0)

        processed = torch.zeros_like(clamped_action)
        processed[:, 0:3] = clamped_action[:, 0:3] * task_config.max_accel
        processed[:, 3] = clamped_action[:, 3] * task_config.max_yaw_rate

        return processed


