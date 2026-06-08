"""
Task configuration for simple hover SNN task with onboard IMU.
Defines:
    - Observation space dimensions
    - Action space dimensions
    - Reward parameters

This is identical to simple_hover task config - the SNN is only
a different neural network architecture, not a different task.
"""

import torch

class task_config:

    seed = 42  # Fixed random seed for reproducibility
    sim_name = "base_sim"
    env_name = "simple_hover_snn_env"  # SNN variant environment

    robot_name = "base_quadrotor"  # Base quadrotor (matching position_setpoint_task)
    controller_name = "lee_attitude_control"  # Attitude control
    args = {}

    # Environment settings
    num_envs = 4096  # Matching position_setpoint_task
    use_warp = False
    headless = False
    device = "cuda:0"

    privileged_observation_space_dim = 0

    # Observation space dim (matching position_setpoint_task):
    # Position error to target (3): [tx - px, ty - py, tz - pz]
    # Robot orientation (4): quaternion [qx, qy, qz, qw]
    # Body Linear Velocity (3): [vx, vy, vz]
    # Body Angular Velocity (3): [wx, wy, wz]
    observation_space_dim = 13

    # Per-dimension labels for the 13D observation vector. Used only by the
    # PopSAN encoder-trace debug plot (tools/plot_encoder_trace.py) to title
    # each row; mirrors the (slice, type_name) format used by the navigation
    # task. There is no "vae_latent" here, so every dim is plotted.
    observation_layout = [
        (slice(0, 3),   "pos_error"),    # target - robot_position (x, y, z)
        (slice(3, 7),   "orientation"),  # quaternion (qx, qy, qz, qw)
        (slice(7, 10),  "body_linvel"),  # body linear velocity (vx, vy, vz)
        (slice(10, 13), "body_angvel"),  # body angular velocity (wx, wy, wz)
    ]

    # Per-dimension (min, max) bounds for the PopSAN population encoder.
    #
    # Bounds are in the rl_games-normalized space (z-scores, hard-clamped to
    # [-5, 5] by RunningMeanStd when normalize_input=True), NOT raw units.
    # Consumed by popsan_network.py via task_config.observation_bounds; it must
    # be length observation_space_dim so the encoder builds means/stds of shape
    # [1, obs_dim, pop_dim] (one learnable Gaussian set PER dimension).
    observation_type_bounds = {
        "pos_error":   (-3.0, 3.0),
        "orientation": (-3.0, 3.0),
        "body_linvel": (-3.0, 3.0),
        "body_angvel": (-3.0, 3.0),
    }

    observation_bounds = [None] * observation_space_dim
    for obj_slice, obj_type in observation_layout:
        lo, hi = observation_type_bounds[obj_type]
        for idx in range(obj_slice.start, obj_slice.stop):
            observation_bounds[idx] = (lo, hi)
    assert all(b is not None for b in observation_bounds), \
        "observation_layout has gaps — every index in [0, observation_space_dim) must be covered"

    # Action space dim (network output): [thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd]
    # Matches LeeAttitudeController expected format directly
    action_space_dim = 4

    # Episode length
    # Shortened 800 -> 400 (20s) to bound the accumulated per-step time penalty
    # so giving-up-via-crash is not a profitable escape (still >> the 3s hold
    # needed for success). See k_time / timeout_penalty below.
    episode_len_steps = 400  # 20 seconds per episode (400 * 0.05s)
    return_state_before_reset = False

    # Success condition: hover at target for 3 seconds
    success_threshold = 0.10      # Distance threshold (meters) - 10cm
    success_hold_duration = 3.0   # Time to hold position (seconds)
    success_hold_steps = 60       # = success_hold_duration / env_step_dt (0.05s)

    # One-time terminal reward when the hover-hold success condition fires
    # (held within success_threshold for success_hold_steps). Applied in the
    # task's step() against the authoritative `success` flag, so it directly
    # rewards the exact success event and keeps reward correlated with the
    # success metric. Success also terminates the episode, so this is the
    # dominant terminal positive signal.
    success_bonus = 100.0

    # One-time terminal penalty when an episode ends in timeout WITHOUT ever
    # achieving success. Mirrors the crash penalty so that "give up and crash to
    # stop the per-step time penalty" is not a profitable escape: crashing and
    # timing-out end with a comparable terminal cost. Applied in step().
    timeout_penalty = 50.0

    # Windowed success rate: success rate computed over the last N completed
    # episodes (success or failure). Logged to tensorboard as success_rate_window.
    # Can be overridden from the YAML via config.success_rate_window_episodes.
    success_rate_window_episodes = 4096

    # Reward parameters - Potential-based reward
    # R_total = R_progress + R_velocity + R_jitter + R_success + R_crash
    #
    # Components:
    # 1. Progress: k_progress * (prev_dist - curr_dist) - rewards approaching goal
    # 2. Gated velocity: -k_vel * g(dist) * ||v|| - penalizes velocity near goal
    #    g(dist) = sigmoid((gate_center - dist) / gate_width) - smooth 0→1 ramp
    # 3. Jitter: -k_jitter * ||action_diff|| - penalizes rapid action changes
    # 4. Success: +k_success per step inside threshold - continuous positive reward
    # 5. Crash: -k_crash - large penalty for exceeding max_distance
    reward_parameters = {
        # Crash penalty
        "k_crash": [50.0],            # Crash penalty magnitude
        "max_distance": [6.0],        # Distance beyond which robot crashes

        # Distance penalty: -k_dist * curr_dist (encourages getting closer)
        "k_dist": [1.0],              # Distance penalty coefficient

        # Progress reward: k_progress * (prev_dist - curr_dist)
        "k_progress": [10.0],         # Reward for moving toward goal (increased for faster progress)

        # Per-step time penalty: -k_time every step (non-crash branch).
        # Conservative value chosen ABOVE the hover-bonus ceiling (k_hover=0.3
        # below) so each loitering step is net-negative (~-0.2/step), pushing the
        # agent to achieve hover quickly and let the episode end on success,
        # while staying far below k_progress(=10) so approaching remains rewarding.
        "k_time": [0.5],

        # Gated velocity penalty: -k_vel * g(dist) * ||v||
        # g(dist) = sigmoid((gate_center - dist) / gate_width)
        "gate_center": [0.5],         # Distance at which gate is 0.5 (meters)
        "gate_width": [0.1],          # Controls steepness of sigmoid ramp

        # Action jitter penalty: -k_jitter * ||action_diff||
        "k_jitter": [0.5],            # Jitter penalty coefficient

        # Attitude penalty: -k_tilt * g(dist) * (pitch^2 + roll^2)
        # Penalizes tilt when near goal (gated by sigmoid)
        "k_tilt": [1.0],              # Tilt penalty coefficient

        # Angular rate penalty: -k_angvel * g(dist) * ||angvel||
        # Penalizes rotation when near goal (gated by sigmoid)
        "k_angvel": [0.5],            # Angular velocity penalty coefficient

        # Hover bonus: +k_hover * exp(-||vel|| / vel_scale) - reward for stable hovering
        # Active when dist < threshold_hover, bonus decays exponentially with velocity.
        # Shrunk 5.0 -> 0.3 so the per-step ceiling (0.3) is BELOW k_time (0.5):
        # this makes loitering net-negative and removes the "farm hover bonus for
        # the whole episode" trap that previously dominated the return.
        "k_hover": [0.3],             # Hover bonus coefficient (kept below k_time)
        # Tightened from 0.20m -> 0.12m to align the per-step hover bonus with the
        # success zone (success_threshold = 0.10m). A small 2cm margin keeps an
        # approach gradient at the boundary while removing the "loiter at ~15cm
        # and farm hover bonus without ever succeeding" reward-optimal trap.
        "threshold_hover": [0.12],    # Distance threshold for hover (meters) - 12cm
        "vel_scale_hover": [0.1],     # Velocity decay scale (m/s) - smaller = stricter
    }
