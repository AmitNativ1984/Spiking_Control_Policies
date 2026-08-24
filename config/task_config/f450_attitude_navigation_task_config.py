import math
import torch

class task_config:
    """
    Configuration for NavigationWithObstaclesTask.

    Key features:
    - Attitude control (thrust, roll, pitch, yaw_rate)
    - Custom 32D DepthVAE encoding
    - 30-level curriculum
    - Randomized environment bounds
    """

    seed = 42
    sim_name = "base_sim"
    env_name = "forest_with_obstacles_env"
    robot_name = "f450"
    # Project-local registration of the stock LeeAttitudeController, differing only in
    # randomize_params = True (see config/controller_config/f450_lee_attitude_config.py).
    controller_name = "f450_lee_attitude_control"
    args = {}

    use_warp = True
    headless = True
    device = "cuda:0"

    # --- TARGET SAMPLING ---
    # The target lands on one of the FOUR VERTICAL env walls (+x, -x, +y, -y), each with
    # probability 1/4, so the world-frame traversal direction is balanced across the batch.
    # Combined with the centre spawn (robot_config.init_config), this removes the old
    # "always fly +x" structure.
    #
    # Walls are pulled in by target_wall_inset so the FULL arrival ball of radius d_min
    # sits inside the bounds. exceed_mask has priority over arrive_mask in
    # compute_rewards(), so a target flush with the wall would be reachable only from a
    # half-ball and any overshoot would score as `exceed` instead of `arrive`.
    target_wall_inset = [0.8, 0.8, 0.0]  # m in from each wall; >= 2*d_min (d_min = 0.4)
    # Window for the two un-pinned axes (the pinned one is overwritten with the wall).
    # The z window is wide on purpose so the vertical bearing component actually varies.
    target_free_ratio_min = [0.05, 0.05, 0.12]
    target_free_ratio_max = [0.95, 0.95, 0.90]
    # Best-of-K rejection so the goal does not end up buried inside an obstacle (an
    # episode that could then only ever time out). Evaluated against the final obstacle
    # positions, which are already written by the time task.reset_idx runs.
    target_clearance_candidates = 8

    # --- OBSTACLE FIELD (Poisson point process) ---
    # Obstacles are placed by a homogeneous Poisson point process over the env volume,
    # thinned by a keep-out ellipsoid around the spawn box. See task/poisson_asset_manager.py.
    #
    # density_max was calibrated to reproduce the level-25 clutter of the ORIGINAL 357 m^3
    # env (24 obstacles). The env was then enlarged in x/y to restore path length (see the
    # env config) WITHOUT lowering the density, so the count scaled with the volume: the
    # box is now 20 x 20 x [4, 6] m = 1600-2400 m^3, and level 25 draws ~110-160 obstacles
    # per env, not 24. Local clutter matches the old distribution; total count does not.
    #
    # The count is a Poisson draw per env, mean = density * free_volume (free_volume is the
    # box minus the spawn keep-out ellipsoid), so it varies env to env by ~sqrt(mean).
    # To target a count N at level 25: density = N / ~1900 (e.g. 24 obstacles -> 0.0126).
    obstacle_density_max = 0.067  # obstacles / m^3 at curriculum max_level
    obstacle_spawn_clearance = 0.95  # m added to the spawn-box half-extents to form the
                                     # keep-out ellipsoid: max sphere radius 0.6 + F450 ~0.35

    # ---- EPISODE LENGTH ---
    episode_len_steps = 800
    exceed_bounds_margin = 1.0  # Out-of-bounds margin multiplier: 1.0 = terminate exactly at env bounds, 
                                # 1.5 = terminate at 1.5x env bounds 

    # --- ACTIONS ---
    action_space_dim = 4
    # Action scaling: network outputs [-1, 1].
    # thrust is kept in [-1, 1] (controller maps it to [0, 2*m*g], hover at 0);
    # roll/pitch and yaw_rate are scaled to physical units below.
    max_inclination_angle_rad = math.pi / 4  # max roll/pitch (45 deg, symmetric: [-max, +max])
    max_yaw_rate = math.pi / 3               # rad/s (~60 deg/s, symmetric: [-max, +max])
    v_max = 5.0  # Speed threshold for excess speed penalty (m/s)

    # --- OBSERVATIONS ---
    state_dim = 17
    privileged_observation_space_dim = 0
    # False (default, and what PPO wants): step() returns the first observation of the
    # NEW episode for envs that terminated this step. True returns the terminal
    # observation instead — on that path the VAE latents are one step stale, because the
    # sensors are not re-rendered until after the reward calculation.
    return_state_before_reset = False

    class state_estimation_noise:
        """Corrupts the odometry-derived observation channels to match EKF2/VIO reality.

        Aerial Gym hands the task exact simulator state; the real F450 gets position and
        velocity from EKF2 fusing IMU + navsat + mag (or VIO indoors), whose error is
        DRIFT and BIAS, not white noise. Applied to the observation only -- rewards and
        terminations keep using true state, otherwise the policy would be paid for
        reaching a hallucinated target.

        TODO(latency): this models estimator error but NOT transport delay. The real
        D435 -> Orin -> PX4 path is ~50-80 ms, i.e. 1.5-3 policy steps of dead time, and
        num_physics_steps_per_env_step only sets the loop RATE, not a delay. Needs an
        explicit N-step obs/action FIFO. Deferred deliberately.
        """
        enable = True
        # NOTE: the random-walk timestep is NOT set here. It is derived at runtime from
        # sim dt x num_physics_steps_per_env_step_mean (see _setup_domain_randomization),
        # so changing the sim rate or the substep count cannot silently desync the drift.
        pos_bias_init_std = 0.05    # m, turn-on offset, resampled per episode
        pos_random_walk_std = 0.02  # m/sqrt(s), slow drift within an episode
        vel_noise_std = 0.05        # m/s, white
        yaw_bias_std = 0.05         # rad (~3 deg), constant per episode

    class vae_config:
        """Custom 32D DepthVAE configuration.

        use_vae is the single source of truth for whether depth-VAE latents are part
        of the observation. Set use_vae = False to train a state-only (17D) policy
        with NO vision input: the observation layout/dim, the PopSAN encoder bounds,
        and the VAE encode step in the task all key off this flag and stay in sync.
        (The depth camera stays attached to the robot; to also stop rendering it,
        disable enable_camera in robot_config.)
        """
        use_vae = True
        latent_dims = 32

        # Path to trained DepthVAE checkpoint. No F450-specific VAE has been trained yet,
        # so this points at the same checkpoint navigation_with_obstacles uses.
        model_file = "/workspaces/aerial_gym_docker/vae_depth/runs/20260218_204641/checkpoints/epoch_150.pth"

        # DepthVAE input resolution
        target_height = 180
        target_width = 320

        # Depth range parameters
        max_depth_m = 7.0
        min_depth_m = 0.1
        sensor_max_range = 10.0

    # Observation space: state_dim [+ vae_config.latent_dims when use_vae].
    observation_space_dim = state_dim + (vae_config.latent_dims if vae_config.use_vae else 0)

    # --- OBSERVATION LAYOUT ---
    #
    # The single source of truth for what each dimension of the observation vector MEANS.
    # MUST match process_obs_for_task() below.
    #
    # This says nothing about how any consumer scales or clamps those dimensions — the
    # PopSAN encoder's per-type clamp windows live with the encoder
    # (rl_training/rl_games/networks/snn/encoder.py: DEFAULT_TYPE_BOUNDS), since the task
    # runs perfectly well under an MLP or GRU policy that has no encoder at all.
    #
    # Other consumers: the obs-stats collector's column names, and the encoder trace plots.
    observation_layout = [
            (slice(0, 3),   "direction_to_target"), # unit vector to target — vehicle frame
            (slice(3, 4),   "distance"),            # normalized distance to target, clamped [0,1]
            (slice(4, 7),   "linvel"),              # vehicle linear velocity
            (slice(7, 10),  "angvel"),              # body angular velocity
            (slice(10, 13), "gravity"),             # gravity in body frame (normalized)
            (slice(13, 17), "prev_action"),         # transformed action: thrust, roll, pitch, yaw_rate
    ]
    # VAE latents only when enabled; appended so the state dims keep indices [0:17].
    if vae_config.use_vae:
        observation_layout.append(
            (slice(17, 17 + vae_config.latent_dims), "vae_latent")  # DepthVAE latents
        )

    # The layout must tile [0, observation_space_dim) exactly. Checked here rather than at
    # a consumer, so an edit to the layout fails at import, not mid-rollout.
    assert sorted(i for sl, _ in observation_layout for i in range(sl.start, sl.stop)) \
        == list(range(observation_space_dim)), \
        "observation_layout must cover every index in [0, observation_space_dim) exactly once"


    # --- REWARD PARAMETERS ---
    reward_parameters = {
        # Terminal rewards
        "arrive_bonus_min": 10.0,        # arrival reward at curriculum level 0 (easy)
        "arrive_bonus_max": 15.0,        # arrival reward at max curriculum level (hard)
        "collision_penalty": -10.0,     # obstacle collision termination
        "exceed_penalty": -10.0,        # out-of-bounds termination
        "timeout_penalty": -2.0,          # episode timeout termination
        "d_min": 0.4,                   # arrival distance threshold (meters)
        
        # Progress reward (dense shaping)
        "lambda_b": 0.1,          # Rewards velocity in target direction (encourage movement towards target)
        "lambda_p": 0.5,           # Rewards closing distance to target (encourage progress)

        "lambda_v": -0.1,         # Penlizes velocity above v_max (encourage speed control for safety)
        "lambda_jerk": -0.01,      # Penalty on jerk (change in acceleration) to encourage smooth control
    }


    class curriculum:
        """
        Curriculum configuration — same thresholds as original NavigationTask.
        Levels 0-5: large panels
        Levels 6-30: cumulative panels + small objects
        """
        min_level = 0
        max_level = 25

        # The level at which obstacle density reaches obstacle_density_max. Deliberately
        # NOT derived from min_level/max_level: pinning the curriculum (--curriculum_level
        # N, or the obs-stats collector) sets min == max == N, which would collapse a
        # (level - min) / (max - min) ramp to 0/1 and silently empty the world at EVERY
        # pinned level. Density is a function of the absolute level, so it keys off this
        # fixed reference instead. Equals max_level, so the un-pinned ramp is unchanged.
        density_at_level = 25
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