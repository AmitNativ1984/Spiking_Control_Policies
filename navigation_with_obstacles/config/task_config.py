"""
Task configuration for navigation with obstacles.
Defines observation/action spaces, reward parameters, curriculum, and VAE settings.
"""
import math
import torch

class task_config:
    """
    Configuration for NavigationWithObstaclesTask.

    Key features:
    - Attitude control (thrust, roll, pitch, yaw_rate)
    - Custom 32D DepthVAE encoding
    - 30-level curriculum (panels then cumulative panels + objects)
    - Randomized environment bounds
    """

    seed = -1

    # Simulation components
    sim_name = "base_sim"
    env_name = "navigation_obstacle_env"
    robot_name = "nav_quadrotor_with_camera"
    controller_name = "lee_attitude_control"
    args = {}

    # Environment settings
    # NOTE: num_envs is set at runtime from the YAML (env_config.num_envs).
    # The task __init__ assigns task_config.num_envs = num_envs from YAML.
    use_warp = True
    headless = True
    device = "cuda:0"

    class vae_config:
        """Custom 32D DepthVAE configuration.

        use_vae is the single source of truth for whether depth-VAE latents are part
        of the observation. Set use_vae = False to train a state-only (17D) policy
        with NO vision input: the observation layout/dim, the PopSAN encoder bounds,
        and the VAE encode step in the task all key off this flag and stay in sync.
        (The depth camera stays attached to the robot; to also stop rendering it,
        disable enable_camera in robot_config.)
        """
        use_vae = False
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

    # Observation space: 17 (state) [+ latent_dims (VAE latents) when use_vae].
    #   [0:3]   unit vector to target (vehicle frame)
    #   [3]     normalized distance to target, clamped [0, 1]
    #   [4:7]   body linear velocity
    #   [7:10]  body angular velocity
    #   [10:13] gravity vector in body frame (normalized)
    #   [13:17] previous (transformed) action: thrust, roll, pitch, yaw_rate
    #   [17:17+latent_dims] DepthVAE latents (only when vae_config.use_vae)
    observation_space_dim = 17 + (vae_config.latent_dims if vae_config.use_vae else 0)
    privileged_observation_space_dim = 0

    # Per-dimension observation bounds for the PopSAN population encoder.
    #
    # Bounds are in the rl_games-normalized space (z-scores, hard-clamped to
    # [-5, 5] by RunningMeanStd when normalize_input=True), NOT raw units.
    # Tune from tools/collect_obs_stats.py if empirically tighter values help.
    #
    # observation_layout is the single source of truth for the observation
    # vector and MUST match process_obs_for_task() in navigation_task.py;
    # observation_bounds is derived from it below. Editing the layout or the
    # per-type bounds is enough — no per-index numbers to maintain.
    observation_layout = [
        (slice(0, 3),   "direction_to_target"), # unit vector to target — vehicle frame
        (slice(3, 4),   "distance"),            # normalized distance to target, clamped [0,1]
        (slice(4, 7),   "linvel"),              # body linear velocity
        (slice(7, 10),  "angvel"),              # body angular velocity
        (slice(10, 13), "gravity"),             # gravity in body frame (normalized)
        (slice(13, 17), "prev_action"),         # transformed action: thrust, roll, pitch, yaw_rate
    ]
    # VAE latents only when enabled; appended so the state dims keep indices [0:17].
    if vae_config.use_vae:
        observation_layout.append(
            (slice(17, 17 + vae_config.latent_dims), "vae_latent")  # DepthVAE latents
        )

    observation_type_bounds = {
        "direction_to_target": (-3.0, 3.0),
        "distance":            (-3.0, 3.0),
        "linvel":              (-3.0, 3.0),
        "angvel":              (-3.0, 3.0),
        "gravity":             (-3.0, 3.0),
        "prev_action":         (-3.0, 3.0),
        "vae_latent":          (-3.0, 3.0),
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

    # Action space: [thrust, roll, pitch, yaw_rate] for lee_attitude_control
    action_space_dim = 4

    # Action scaling: network outputs [-1, 1].
    # thrust is kept in [-1, 1] (controller maps it to [0, 2*m*g], hover at 0);
    # roll/pitch and yaw_rate are scaled to physical units below.
    max_inclination_angle_rad = math.pi / 4  # max roll/pitch (45 deg, symmetric: [-max, +max])
    max_yaw_rate = math.pi / 3               # rad/s (~60 deg/s, symmetric: [-max, +max])
    
    # Speed threshold for excess speed penalty (m/s)
    v_max = 5.0

    # Episode length
    episode_len_steps = 800

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
        "arrive_bonus_min": 50.0,        # arrival reward at curriculum level 0 (easy)
        "arrive_bonus_max": 75.0,        # arrival reward at max curriculum level (hard)
        "collision_penalty": -20.0,     # obstacle collision termination
        "exceed_penalty": -200.0,        # out-of-bounds termination
        "timeout_penalty": -2.0,          # episode timeout termination
        "d_min": 0.4,                   # arrival distance threshold (meters)
        
        # Progress reward (dense shaping)
        "lambda_b": 0.1,          # Rewards velocity in target direction (encourage movement towards target)
        "lambda_p": 0.1,           # Rewards closing distance to target (encourage progress)

        "lambda_v": -0.01,         # Penlizes velocity above v_max (encourage speed control for safety)
        "lambda_jerk": -0.001,      # Penalty on jerk (change in acceleration) to encourage smooth control
    }

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
        Transform network output [-1, 1] to attitude commands for
        lee_attitude_control: [thrust, roll, pitch, yaw_rate] (vehicle frame).

        The network outputs are in [-1, 1] for all 4 dimensions.
        - thrust  : kept in [-1, 1]; controller maps it via (thrust+1)*m*g,
                    so 0 = hover, -1 = zero thrust, +1 = 2*hover.
        - roll/pitch: scaled to [-max_inclination_angle_rad, +max_inclination_angle_rad] (radians).
        - yaw_rate: scaled to [-max_yaw_rate, +max_yaw_rate] (rad/s).
        """
        clamped_action = torch.clamp(action, -1.0, 1.0)

        processed = torch.zeros_like(clamped_action)
        processed[:, 0] = clamped_action[:, 0]                                          # thrust: no scaling
        processed[:, 1:3] = clamped_action[:, 1:3] * task_config.max_inclination_angle_rad
        processed[:, 3] = clamped_action[:, 3] * task_config.max_yaw_rate

        return processed



