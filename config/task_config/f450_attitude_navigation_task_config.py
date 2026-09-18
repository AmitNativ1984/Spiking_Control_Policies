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
        # -lambda_fov * ||v||^2 * r_fov^fov_power,  r_fov = 1.0 on the cone boundary
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
        # SPEED ENTERS AS v^2. Earlier revisions used min(speed, v_ref), which needed a
        # saturation threshold nobody could derive -- it was picked by eye and quietly
        # meant different things for different policies (b4 flies mostly below 2 m/s so
        # the multiplier was a live ramp; p_blind flies above it so it was nearly flat).
        # v^2 removes the knob and does three jobs at once:
        #   - hover is free, because v^2 -> 0. Station-keeping drift has a meaningless
        #     direction, and at 0.1 m/s it costs ~0.7% of the cone edge at 2 m/s, so no
        #     gate is needed to suppress it.
        #   - the tolerated misalignment NARROWS AS 1/v: iso-penalty contours satisfy
        #     |v| * r_fov = const, so the cone the policy is allowed to stray outside
        #     halves every time speed doubles.
        #   - the crossover where slowing beats aiming moves inward as speed rises
        #     (r_fov = 1.00 at 1 m/s, 0.71 at 2, 0.58 at 3), which is Falanga's
        #     speed-vs-sensing relation (RAL 2019) appearing without being encoded.
        #
        # Deliberately NOT derived from a reachability model. Available acceleration is
        # not a constant: holding altitude at 45 deg tilt already spends 1.41 mg of the
        # 2 mg the controller can command, leaving 0.41 g of vertical escape authority
        # against 1.0 g when level -- and that is the narrow FOV axis. Modelling the
        # reachable acceleration set per state is a planner, not a reward term. v^2
        # claims only that faster means more committed, which holds whatever budget is
        # left.
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
        # SIZING RULE: set lambda_fov so the mean per-step penalty is comparable to
        # r_progress, which measures about 0.029/step. That is the whole criterion --
        # loud enough for the policy to attend to, not loud enough to displace the task.
        # lambda_fov = 0.029 / mean(||v||^2 * r_fov^2), measured over every step rather
        # than crashes only. NOTE the units changed with the v^2 multiplier: lambda is
        # now per (m/s)^2, so it is not comparable to the pre-v^2 value of 0.015.
        #
        # MEASURED on b4, 10,097 episodes at level 30 (analysis/data/scale_b4_v2.json):
        #   mean(||v||^2 * r_fov^2) = 4.207 /step at lambda = 1
        #   yaw-slaved floor         = 1.870  -> 2.336 removable (55.5%)
        #   mean r_fov 1.125, mean speed 2.61 m/s, 152 steps/episode
        # => lambda_fov = 0.029 / 4.207 = 0.0069 per (m/s)^2, giving 0.029/step mean and
        #    4.4 points/episode against b4's 25.9 return. Sizing against the REMOVABLE
        #    part instead gives 0.0124; the honest answer is somewhere between, since the
        #    floor lowers return without creating gradient.
        #
        # The speed distribution this run finally measured separately is BIMODAL: a slow
        # mode around 1.0-1.2 m/s and a larger cruise mode at 3.6-4.4 m/s, 61% of steps
        # above 2 m/s. The old fov_v_ref = 2.0 sat almost exactly in the TROUGH between
        # them, so the saturating multiplier behaved as a linear ramp in one regime and a
        # flat constant in the other -- the worst place to put a threshold, and a good
        # retrospective argument for having removed it.
        #
        # WHY v^2 IS NOT REFUTED BY THE CRASH DISTRIBUTION. Measured on b4: crashes
        # concentrate in the SLOW regime -- 42% of them below 1 m/s, which is only 17.6%
        # of flight time, while the 4+ m/s band holds 21.8% of flight and just 1.4% of
        # crashes. It is tempting to read that as v^2 loading its pressure where nothing
        # is failing. That reading is wrong: the distribution is the OUTPUT of a policy
        # that already reserves high speed for space it has established is clear, so the
        # low crash count up there is evidence the speed/risk tradeoff is being managed,
        # not that fast flight is harmless. Inferring hazard from the frequency of a
        # behaviour the policy already optimises against is the same error as concluding
        # high altitude is safe because few crashes happen there. What a reward term must
        # price is the COUNTERFACTUAL -- what the policy would do without it -- and
        # blind fast flight is genuinely worse, because achievable steering angle falls
        # as ~1/v^2 at fixed sensing range (Falanga, RAL 2019): at speed you are
        # committed. v^2 encodes that.
        #
        # The distribution does imply something narrower and worth keeping in view: the
        # dominant current failure is clipping obstacles abeam while threading clutter at
        # ~1 m/s, where v^2 = 1 and this term is small. So p_fov is mostly PREVENTIVE --
        # it stops blind fast flight from emerging -- rather than corrective against the
        # clipping that dominates today, and should not be expected to fix that mode.
        "lambda_fov": float(os.environ.get("F450_LAMBDA_FOV", 0.0)),  # 0.0 = inert; 0.0069 to enable
        "fov_power": float(os.environ.get("F450_FOV_POWER", 2.0)),    # 4 = quartic

        # --- p_cbf: discrete-time control barrier function on obstacle clearance -----
        #
        #   h(x)    = DF(p) - d_safe              margin to the nearest surface
        #   barrier   h(x_{t+1}) >= (1 - alpha*dt) * h(x_t)
        #   p_cbf   = -lambda_cbf * max(0, v_close - alpha*h_t),  v_close = -dh/dt
        #
        # DF is the EXACT distance to the nearest point on any triangle in the env's warp
        # mesh, queried per env per step (env_manager/proximity_probe.py) -- full 3D
        # geometry, not a depth-image reduction, so it includes the surfaces abeam and
        # behind the drone that the camera cannot see. The floor and the side walls are
        # in that mesh too, so DF = min(altitude, nearest obstacle).
        #
        # WHAT IT ADDS over the terms already here: alpha*h is an allowed closing speed
        # PROPORTIONAL TO REMAINING CLEARANCE. p_speed charges against a fixed v_max
        # wherever the drone is; p_fov charges for where the velocity points. Neither
        # knows how much room is left, and "3 m/s with 4 m of room" versus "3 m/s with
        # 0.6 m of room" is precisely the distinction the crash data says is missing.
        #
        # --- d_safe = 1.0 m, SIZED FROM MEASUREMENT, NOT FROM THE AIRFRAME ------------
        # The obvious choice is "collision radius plus margin", which argues for 1.5 m.
        # The env refutes it. Measured on b4 at level 30 over 256k steps, nearest surface
        # IN VIEW (analysis/data/freespace_b4_l30.json):
        #   p10 0.94 m | p25 1.09 m | median 1.56 m | p75 2.03 m | p90 2.50 m
        #   45.2% of ALL steps below 1.5 m | 17.4% below 1.0 m | 4.3% below 0.7 m
        # and that is an UPPER bound on what the probe reports, because the probe is a
        # full sphere including the floor and can only ever find something closer than
        # the forward cone does.
        #
        # So d_safe = 1.5 m would put h < 0 on at least half of all steps. In that regime
        # the barrier does not ask for a slower approach, it demands clearance be REGAINED
        # at alpha*|h| -- retreat -- which is not achievable while traversing clutter at
        # this density. The term would stop being a closing-speed signal and become a
        # near-constant tax competing with r_progress.
        #
        # SINCE MEASURED DIRECTLY, and the in-FOV figures above were optimistic by ~0.46 m
        # exactly as expected. The probe's own full-sphere distribution on the p_fov
        # baseline at level 30 (analysis/data/cbf_margin_p_fov_l30.json) is
        #   p1 0.40 | p5 0.60 | p10 0.65 | p25 0.85 | p50 1.10 | p75 1.40 | p90 1.65 m
        # so d_safe = 1.0 sits just below the MEDIAN clearance: h < 0 on ~40% of steps, and
        # the barrier is violated on 51.0% of them. That is tight but not degenerate --
        # half the steps are still compliant, so the gradient has structure. d_safe = 0.70
        # is what this distribution would argue for on its own (~13% of steps inside,
        # 41.6% violation); it is one env var away (F450_D_SAFE) if 1.0 proves too hot.
        #
        # The altitude-tax worry did NOT materialise: the floor is the nearest surface on
        # only 0.1% of steps (median height above floor 2.25 m), so including it in the
        # probe costs nothing in practice.
        #
        # Second reason to stay at or below 1.0 m: targets are sampled 0.8 m off a side
        # wall (target_wall_inset) with a 0.4 m arrival ball, and those walls are cullable
        # pool members that are meshed on some resets. Any d_safe > 0.8 m means the final
        # approach of a SUCCESSFUL flight is unavoidably inside the bubble, so the term
        # charges for arriving.
        "d_safe": float(os.environ.get("F450_D_SAFE", 1.0)),  # m; safe set is DF >= d_safe
        #
        # --- alpha = 0.5 /s: DELIBERATELY TIGHT, AND EXPECTED TO BIND ------------------
        # alpha is a RATE (1/s) and sets the allowed closing speed v_allow = alpha*h.
        # NOTE the [0, 1] bound often quoted for alpha belongs to the dimensionless decay
        # factor alpha*dt, NOT to alpha: at dt = 0.03 s the discrete barrier is well-posed
        # for alpha up to ~33 /s. The task asserts alpha*dt <= 1 at init.
        #
        # At 0.5 /s the drone may close on a surface 2.0 m away (h = 1.0) at 0.5 m/s, and
        # on one 1.5 m away at 0.25 m/s, against a measured cruise of 2.6 m/s. That is
        # far tighter than the policy currently flies and the violation rate should start
        # near 1 -- this is a chosen starting point, to see what the term does before
        # loosening it, not a sized value. Sizing alpha ~ v_cruise / h_typical would give
        # 2-6 /s instead. metrics/cbf_violation_rate is the number to watch: if it sits
        # at ~1.0 the barrier is being violated on every step and p_cbf has degenerated
        # into a flat speed tax with no gradient structure, which is the signal to raise
        # alpha. Sweep with F450_ALPHA_CBF, no file edit needed.
        "alpha_cbf": float(os.environ.get("F450_ALPHA_CBF", 0.5)),  # 1/s
        #
        # --- lambda_cbf: per (m/s) of EXCESS closing speed -----------------------------
        # 0.0 = inert (identical to not having the term), same convention as lambda_fov
        # and lambda_action_mag. The rate form means this weight is dt-INDEPENDENT: it is
        # the cost of one m/s of excess closing speed, so changing the sim rate or the
        # substep count cannot silently rescale the term.
        #
        # SIZING RULE, the same one used for lambda_fov: pick the mean per-step penalty
        # to sit near r_progress's, which the p_fov baseline measures at 0.0255/step --
        # loud enough to attend to, not loud enough to displace the task. So
        # lambda_cbf = 0.0255 / mean(max(0, v_close - alpha*h)).
        #
        # MEASURED on the p_fov baseline at level 30, 1500 steps x 256 envs, 381,798 paired
        # steps (analysis/data/cbf_margin_p_fov_l30.json):
        #     d_safe  alpha   violation   mean excess   lambda_cbf
        #       0.70   0.50       41.6%      0.303 m/s      0.0843
        #       0.70   4.00       19.8%      0.156          0.1637
        #       1.00   0.50       51.0%      0.372          0.0686   <-- shipped pair
        #       1.00   4.00       45.5%      0.544          0.0468
        #       1.25   0.50       58.6%      0.440          0.0579
        #       1.50   0.50       65.7%      0.518          0.0492
        # => 0.0686 at the shipped (d_safe 1.0, alpha 0.5).
        #
        # NOTE THE NON-MONOTONICITY at d_safe >= 1.0: raising alpha there does NOT reduce
        # the violation rate, it inflates the mean excess (0.372 -> 0.891 m/s from alpha
        # 0.5 to 8.0). That is the h < 0 regime showing itself -- below d_safe the
        # allowance alpha*h is NEGATIVE, so a larger alpha demands a faster retreat and
        # violations get bigger, not rarer. At d_safe = 0.70, where h < 0 is uncommon,
        # alpha behaves as intended and does cut the rate (41.6% -> 15.8%).
        # Non-zero builds RaySphereProbe, which costs a measured 19.3 ms/step at 128
        # envs x 16384 rays -- rays here are maximally incoherent, so this is far more than
        # the same count of camera rays would suggest. Budget it: F450_CBF_RAYS trades
        # accuracy for that time, linearly. (The exact closest-point query this replaces
        # faults outright under Warp 1.0.0; see env_manager/proximity_probe.py.)
        "lambda_cbf": float(os.environ.get("F450_LAMBDA_CBF", 0.0)),  # 0.0 = inert
        #
        # Probe range. The query cost grows with it, and it need only exceed the largest
        # clearance that can occur: the floor is always beneath the drone and the env is
        # 4-6 m tall, so DF <= ~6 m always and nothing is ever truncated in practice. An
        # env whose previous h sat at this limit is exempt from the penalty anyway (see
        # the task), since a truncated DF understates the allowed closing speed.
        "cbf_max_range": float(os.environ.get("F450_CBF_MAX_RANGE", 6.0)),  # m
        #
        # --- cbf_rays: how many directions the clearance sphere samples ----------------
        # DF is measured by ray casting, not by an exact closest-point query -- not by
        # choice: Warp 1.0.0's mesh_query_point faults on this env's 73k-triangle meshes at
        # every curriculum level above 0. Ray queries over the same meshes are fine (the
        # depth camera casts 320x180 = 57,600 of them per env per step).
        #
        # THE RAY COUNT IS SET BY THIN GEOMETRY, NOT BY SURFACES. A large surface is read
        # almost exactly: a ray missing the true nearest point by angle a gives d/cos(a),
        # an overestimate of ~d*a^2/2, which is 1.5 mm at 2 m even at 2048 rays. What the
        # count buys is detection of THIN things -- a cylinder of radius r at distance d is
        # only caught if a ray passes within its angular radius r/d. With n directions the
        # spacing is ~sqrt(4*pi/n):
        #      2048 -> 4.5 deg -> catches r >= 7.8 cm at 2 m   (3.6% of camera ray cost)
        #      8192 -> 2.2 deg -> catches r >= 3.9 cm at 2 m   (14%)
        #     16384 -> 1.6 deg -> catches r >= 2.8 cm at 2 m   (28%)
        #     32768 -> 1.1 deg -> catches r >= 2.0 cm at 2 m   (57%)
        # 16384 is the default because tree branches -- 26 per tree, and the geometry that
        # argued for an exact query in the first place -- sit in the few-centimetre range.
        #
        # THE ERROR IS SIGNED THE UNSAFE WAY: a missed feature makes clearance read LARGER
        # than it is, and the barrier then permits a higher closing speed than it should.
        # That is the reason to spend rays here rather than economise. Raise via
        # F450_CBF_RAYS; the init log prints the spacing and the feature size it resolves.
        "cbf_rays": int(os.environ.get("F450_CBF_RAYS", 16384)),
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