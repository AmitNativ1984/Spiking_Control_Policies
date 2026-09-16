import math
import os
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
    # 0.0804, rescaled from 0.067 when max_level/density_at_level went 25 -> 30.
    # density = obstacle_density_max * min(level / density_at_level, 1.0), so scaling
    # BOTH by 30/25 leaves the density at every level 0-25 bit-identical (level 24 is
    # still 0.06432, level 25 still 0.067) while levels 26-30 extend into new ground up
    # to 0.0804. Raising max_level alone would NOT have worked: the fraction clamps at
    # 1.0, so 26-30 would have had exactly the same density as 25.
    obstacle_density_max = 0.0804  # obstacles / m^3 at curriculum density_at_level
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
    # ~60 deg/s by default. B5 experiment: override via env var F450_MAX_YAW_RATE_DEG (degrees,
    # converted below) to sweep without touching this file for other runs.
    #
    # *** CONTRACT WARNING ***: max_yaw_rate (with max_inclination_angle_rad) is a
    # training<->flight contract, asserted equal to BasePolicy.max_yaw_rate_rad_s in the
    # external sail-uav-core repo by tests/test_deploy_nav_obs_parity.py:379-380. A checkpoint
    # trained with this overridden is NOT deployable without a coordinated change to
    # sail-uav-core's BasePolicy. It also silently renormalizes p_jerk's _action_scale
    # (task/attitude_navigation_task.py:207-213) -- changing this is never a single-variable
    # change in the strict sense, it moves two things at once.
    #
    # Raising this ALONE (from scratch, no other reward/loss change) was already tried and
    # falsified: the policy just scaled its demand proportionally (84->236 deg/s) and stayed
    # ~66% clamped -- bounds_loss was too weak (0.001) to stop it. B5 pairs this with B4's
    # bounds_loss_coef=0.01 specifically to test whether that mechanism, absent last time,
    # prevents the same runaway.
    max_yaw_rate = math.radians(float(os.environ.get("F450_MAX_YAW_RATE_DEG", 60.0)))  # rad/s
    # Override via env var F450_V_MAX to sweep without touching this file (this task
    # config is shared -- other training jobs may already be running against it).
    v_max = float(os.environ.get("F450_V_MAX", 5.0))  # Speed threshold for excess speed penalty (m/s)

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
        model_file = "/workspaces/aerial_gym_docker/vae_depth/runs/20260828_060313/checkpoints/epoch_200.pth"

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
        #
        # r_bearing (lambda_b * dot(n, v_hat)) WAS here and has been REMOVED. It measured
        # +0.0842 EMA at B4, i.e. 12.97 of the 25.00 per-episode return -- more than the
        # whole arrive bonus contribution (12.22) and 2.9x r_progress. It was not shaping,
        # it was the objective, and what it paid for was pointing the velocity vector at
        # the goal. Against it, colliding cost 1.81/episode: the reward was offering 7:1
        # odds in favour of beelining and eating the crashes.
        #
        # It was also incapable of doing the one job its name implies. n and v are both
        # rotated by the SAME yaw-only robot_vehicle_orientation (base_multirotor.py:290),
        # so the yaw cancels in the dot product and the term carried exactly zero yaw
        # information -- see p_blind in the task's reward function, which replaces it.
        #
        # Deleted rather than zeroed: a zero-weight term is a live footgun, and the
        # per-step directional tax it applied (-lambda_b on every step of a retreat) is
        # precisely what made backing out of a dead end irrational.
        #
        # lambda_p is UNCHANGED at 0.5, deliberately. Raising it to absorb the freed dense
        # budget was considered and rejected: r_progress's toward-vs-away differential is
        # 2*lambda_p*speed*dt, which at lambda_p=1.75 and 3 m/s is 0.315/step -- LARGER
        # than the 0.200/step differential of the r_heading being removed. That would have
        # reintroduced a bigger directional tax than the one deleted. Return magnitude does
        # not need preserving; goal-seeking is carried by the arrive bonus and gamma
        # discounting (0.99 at 33 Hz is a 3.0 s horizon against a 4.6 s episode, so a
        # direct run beats wander-then-go by 2.73x on discounted return).
        "lambda_p": 0.5,           # Rewards closing distance to target (encourage progress)

        "lambda_v": -0.1,         # Penlizes velocity above v_max (encourage speed control for safety)
        # -0.01 by default. B2 experiment (smoothness/saturation plan): override via
        # env var F450_LAMBDA_JERK to sweep without touching this file (this task
        # config is shared -- other training jobs may already be running against it).
        "lambda_jerk": float(os.environ.get("F450_LAMBDA_JERK", -0.01)),

        # B3/B4 experiment: bounded action-MAGNITUDE penalty, NTNU-style saturating shape
        # lambda_action_mag * (exp(-||a/action_scale||^2 / action_mag_nu) - 1) -- see
        # task/attitude_navigation_task.py's reward function for the implementation.
        # Unlike p_jerk (unbounded linear, penalizes CHANGE), this penalizes MAGNITUDE
        # and saturates, so it can never dominate the loss the way an unbounded term
        # could. 0.0 by default (inert; identical to not having the term at all).
        # action_mag_nu is a SHAPE constant, not meant to be swept: pinned so the term
        # evaluates to ~-0.025 (~30% of r_heading's measured EMA of +0.0838) at the
        # ep_1800/level-0 measured operating point (combined clamped-action L2 norm
        # ~1.4, from analysis/measure_erratic.py's per-channel mean|mu| at that
        # checkpoint) when lambda_action_mag=0.04:
        #   exp(-1.4^2 / 2.0) - 1 = exp(-0.98) - 1 = -0.625;  0.04 * -0.625 = -0.025
        # A well-behaved low-magnitude policy (combined norm ~0.7, ep_50-era) gets a
        # much smaller penalty (0.04 * (exp(-0.245)-1) = -0.0087) -- saturating, not
        # linear, so it discourages high sustained magnitude without cratering reward
        # for a policy that occasionally needs a large but brief command.
        #
        # PROMOTED TO DEFAULT from the smoothness/saturation B-series: 0.04, B3's isolated
        # value, held up combined with bounds_loss_coef=0.01 in B4 (6.6x lower bounds_loss
        # than baseline standalone, task performance statistically unchanged). Override via
        # env var F450_LAMBDA_ACTION_MAG if a specific job needs 0.0 (inert) or another
        # value without touching this file.
        "lambda_action_mag": float(os.environ.get("F450_LAMBDA_ACTION_MAG", 0.04)),
        "action_mag_nu": 2.0,

        # --- p_blind: motion the camera cannot see ---
        # -lambda_blind * horizontal_speed * ((1 - forward/horizontal_speed) / 2)^2
        #
        # The ONLY term in which yaw appears at all, positively or negatively. With
        # r_bearing gone, yaw would otherwise enter the reward solely through p_jerk and
        # p_action_mag -- as cost, never as benefit -- so a policy that refuses to yaw
        # would be behaving optimally. That is why the drone never yaws today.
        #
        # SIZING. Coupled to lambda_p through the requirement that faster is always
        # better: forward motion stays profitable iff lambda_blind * misalignment^2 <
        # lambda_p * dt (dt = sim dt 0.01 * 3 substeps = 0.03 s), i.e. < 0.015. At the
        # 3/8 expected misalignment^2 of a yaw-uncorrelated policy -- which is what we
        # have, nothing having ever paid it to yaw -- 0.03 * 0.375 = 0.01125 clears that,
        # so there is no slow-down incentive at the starting operating point. It only
        # starts favouring "slow down" past 90 deg off-nose, where slowing and yawing is
        # the wanted behaviour anyway.
        #
        # Projected worth: -5.2/episode untrained, -0.7 once yaw is learned, so ~4.5
        # points of headroom on an 11.4-point achievable return. Large enough to find,
        # not large enough to displace the task.
        "lambda_blind": float(os.environ.get("F450_LAMBDA_BLIND", 0.03)),

        # --- p_fov: velocity pointed outside what the camera can actually see ---
        # -lambda_fov * min(speed, fov_v_ref) * r_fov^fov_power,  r_fov = 1.0 on the boundary
        #
        # p_blind measures misalignment from the NOSE AXIS and is horizontal-only. Both
        # are wrong relative to the measured failure. Crash-cause eval of b4 and p_blind
        # at level 30 (analysis/crash_cause_eval.py) found:
        #   - 50.6% of crash velocities beyond the 43.5 deg horizontal half-FOV
        #   - 34.3% beyond the 28.1 deg VERTICAL half-FOV, an axis p_blind cannot see
        #   - ~55% of struck obstacles beyond the vertical half-angle
        # so the quantity that predicts a crash is not "how far off the nose" but "how
        # far outside the sensed cone", and the vertical cone is the tighter one.
        #
        # SHAPE. Each axis is normalized by ITS OWN half-angle, so the asymmetry of the
        # 87 x 56 deg frustum is encoded in the penalty rather than hand-weighted:
        #   r_fov = sqrt((psi/half_h)^2 + (theta/half_v)^2),  r_fov = 1 on the boundary
        # The same absolute overshoot therefore costs ~1.5x more vertically (43.5/28.1),
        # which is correct -- leaving the tighter cone strands you in a smaller sensed
        # volume. This is the ellipse inscribed in the rectangular frustum, so it is
        # slightly conservative at the image corners; that is the cheap side to err on.
        #
        # GROWTH is a power law in the distance from BORESIGHT, r_fov^power, MONOTONE
        # over the whole domain. There is deliberately no clamp: a cutoff
        # flattens the gradient beyond it, and p_blind's own note rejects a saturating
        # shape for exactly that reason -- "at init yaw is uncorrelated with velocity
        # (median misalignment 90 deg), so the policy would start in the region where
        # such a shape teaches nothing". Anything that dies at 90 deg dies precisely
        # where a from-scratch policy lives.
        #
        # SCALE is anchored at the CONE EDGE, not at the worst case: r_fov is already
        # 1.0 on the boundary, so lambda_fov IS the per-step cost of flying straight at
        # the edge of what the camera can see. That is the right unit because it is where
        # the failures live -- crash velocities have a MEDIAN azimuth of 43.9 deg against
        # a 43.5 deg half-angle. Dividing through by r_fov_max (= 5.23, set by the
        # full-reversal corner) would peg the scale to a case that essentially never
        # happens and leave the edge at 1/27th of it.
        #
        # The term stays bounded by geometry -- psi and theta are bounded, so the worst
        # reachable value is r_fov_max^power = 27.4 at quadratic -- so it cannot run away
        # the way a genuinely unbounded term could.
        #
        # EXPONENT trades in-cone freedom against tail severity, in units of the edge
        # cost: at quadratic, half a cone width costs 0.25 lambda, the edge 1.0, 90 deg
        # off 4.3, full reversal 17.1. Quartic would drop half a cone width to 0.06 but
        # push full reversal to 292, which over-weights a corner the policy is rarely in.
        # Raise fov_power via F450_FOV_POWER if the quadratic proves too permissive near
        # the axis.
        #
        # SPEED is SATURATED, not raw. Speed belongs in the term (flying fast into unseen
        # space is worse than drifting into it) but a raw multiplier lets the policy
        # satisfy a larger lambda by SLOWING DOWN instead of by aiming -- which just
        # rediscovers the v_max cap. Saturating keeps hover free and the risk gradient
        # intact while making aim the only remaining lever above fov_v_ref.
        #
        # The half-angles are NOT duplicated here: the task derives them from the robot's
        # live camera config at init, so widening the lens automatically widens the free
        # region rather than silently leaving the penalty keyed to the old frustum.
        # SIZING (analysis/crash_cause_eval.py, 10k episodes per policy at level 30,
        # measured in the BODY frame over EVERY step, not crashes only):
        #   b4       mean r_fov 1.119 -- the AVERAGE step is already outside the cone --
        #            term 2.375/step at lambda=1, of which 0.622 is a floor no yaw can
        #            remove -> 1.753 removable (73.8%). 33% of steps have r_fov > 1.
        #   p_blind  mean r_fov 0.926, term 1.523/step, floor 0.778 -> 0.745 removable.
        #            Only 48.9% removable: p_blind has already harvested the azimuth
        #            slack, so what is left is mostly elevation.
        #
        # Pitch/roll contribute less than expected: body frame raises mean r_fov by only
        # 2.6-3.2% over the yaw-only frame. But it raises the FLOOR by ~19% while barely
        # moving the total, because tilt lands entirely in elevation, which yaw cannot
        # undo -- so the irreducible share goes 22.6 -> 26.2% (b4) and 45.9 -> 51.1%
        # (p_blind). The body frame is still the right one (it is the actual frustum),
        # it just reallocates rather than inflates.
        #
        # 0.015 is where three independent constraints meet, which is why it is the
        # documented value rather than a swept one:
        #   1. SLOW-DOWN SETPOINT. Below fov_v_ref, slowing beats aiming iff
        #      lambda * r_fov^power > lambda_p * dt = 0.5 * 0.03 = 0.015. At exactly
        #      lambda_fov = 0.015 that crossover lands on r_fov = 1 -- the cone boundary.
        #      Inside the cone, fly as fast as you like; outside it, slowing starts to
        #      pay. That is Falanga's "drive within your sensing range" (RAL 2019)
        #      falling out of the arithmetic rather than being tuned in. It costs
        #      slow-down pressure on the 33% of b4 steps that sit beyond the cone.
        #   2. MAGNITUDE. 0.015 * 1.753 = 0.026/step removable against b4, inside the
        #      0.025-0.085 band the other components occupy (r_bearing measured 0.0842,
        #      r_progress 0.029, p_action_mag 0.025).
        #   3. PRECEDENT. 5.4 points/episode total and 3.9 removable on b4's 25.9
        #      return, against p_blind's own sizing of "-5.2/episode untrained ... ~4.5
        #      points of headroom". Same regime, arrived at independently.
        #
        # Fine-tuning from p_blind rather than b4 needs ~0.034 for the same removable
        # magnitude, since only half its remaining misalignment is yaw-addressable.
        "lambda_fov": float(os.environ.get("F450_LAMBDA_FOV", 0.0)),  # 0.0 = inert; 0.015 to enable
        "fov_power": float(os.environ.get("F450_FOV_POWER", 2.0)),    # 4 = quartic
        "fov_v_ref": float(os.environ.get("F450_FOV_V_REF", 2.0)),    # m/s saturation
    }


    class curriculum:
        """
        Curriculum configuration — same thresholds as original NavigationTask.
        Levels 0-5: large panels
        Levels 6-30: cumulative panels + small objects
        """
        # 0. Back to a from-scratch start, per the warning this comment used to carry:
        # min_level is the level a run STARTS at, and 23 existed only to resume run
        # 261745's epoch-2550 checkpoint. An untrained policy started at 23 lands in
        # 0.062 obstacles/m^3 with no chance of clearing the gate.
        #
        # It has to be 0 for the retune in rl_training/rl_games/cfg/ppo_mlp_*.yaml
        # (bounds_loss_coef, kl_threshold, horizon_length/minibatch_size) to mean anything:
        # resuming loads the previous run's action_log_std, and every post-refactor
        # checkpoint carries an already-inflated sigma -- run of4veb49 resumed at 0.894 and
        # went to 2.30. Starting from a saved policy would import the failure the retune is
        # meant to prevent.
        #
        # Set it back to a level only to resume a specific checkpoint, and say which.
        min_level = 0
        max_level = 30

        # The level at which obstacle density reaches obstacle_density_max. Deliberately
        # NOT derived from min_level/max_level: pinning the curriculum (--curriculum_level
        # N, or the obs-stats collector) sets min == max == N, which would collapse a
        # (level - min) / (max - min) ramp to 0/1 and silently empty the world at EVERY
        # pinned level. Density is a function of the absolute level, so it keys off this
        # fixed reference instead. Equals max_level, so the un-pinned ramp is unchanged.
        density_at_level = 30
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