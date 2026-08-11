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
import math

class task_config:

    seed = 42  # Fixed random seed for reproducibility
    sim_name = "base_sim"
    env_name = "empty_new_env"  # SNN variant environment
    robot_name = "f450_nocam"  # Base quadrotor (matching position_setpoint_task)
    # Project-local registration of the stock LeeAttitudeController, differing only in
    # randomize_params = True (see config/controller_config/f450_lee_attitude_config.py).
    controller_name = "f450_lee_attitude_control"  # Attitude control
    args = {}

    # Environment settings
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

    # Action space dim (network output): [thrust_cmd, roll_cmd, pitch_cmd, yaw_rate_cmd]
    # Matches LeeAttitudeController expected format directly
    action_space_dim = 4
    max_inclination_angle_rad = math.pi / 4  # max roll/pitch (45 deg, symmetric: [-max, +max])
    max_yaw_rate = math.pi / 3               # rad/s (~60 deg/s, symmetric: [-max, +max])
        

    # Episode length
    # Shortened 800 -> 400 (20s) to bound the accumulated per-step time penalty
    # so giving-up-via-crash is not a profitable escape (still >> the 3s hold
    # needed for success). See k_time / timeout_penalty below.
    episode_len_sec = 20.0  # seconds per episode
    episode_len_steps = int(episode_len_sec / (0.01 * 3)) # episode_len_sec / (sim_dt * num_physics_steps_per_env_step_mean) = 20 / (0.01 * 3) = 666.67 ~ 667 steps
    return_state_before_reset = False

    # Success condition: hover at target for ~3.6 seconds.
    #
    # A METRIC ONLY. It pays no bonus and does not end the episode -- the agent is paid
    # continuously for proximity instead (see reward_parameters). Both the old
    # success_bonus and timeout_penalty were removed with it: a large terminal bonus on a
    # condition this narrow re-creates the reward cliff that made "approach but fail to
    # hold" score six times worse than "leave the box immediately".
    success_threshold = 0.10      # Distance threshold (meters) - 10cm
    success_hold_duration = 3.6   # Time to hold position (seconds)
    success_hold_steps = 120      # = success_hold_duration / env_step_dt (0.01 * 3 = 0.03s)

    # Radius for the continuous "fraction of steps near the target" diagnostic. Chosen
    # much looser than success_threshold on purpose: success is a binary that reads 0 for
    # a long time and tells you nothing while it does, whereas this moves from epoch 1.
    diagnostic_radius = 0.5  # meters

    # Windowed success rate: success rate computed over the last N completed
    # episodes (success or failure). Logged to tensorboard as success_rate_window.
    # Can be overridden from the YAML via config.success_rate_window_episodes.
    success_rate_window_episodes = 4096

    # Out-of-bounds termination, applied to the env box from obs_dict["env_bounds_*"].
    # 1.0 = terminate exactly at the env bounds, 1.5 = terminate at 1.5x the bounds.
    # A margin > 1.0 buys overshoot room between the target-sampling region (inset by
    # target_inset) and the termination surface; see the note on target_inset in the task.
    # Kept out of reward_parameters on purpose: everything in there is tensorized at
    # task construction, and this is consumed as a plain Python float.
    exceed_bounds_margin = 1.2

    # Reward parameters - dense POSITIVE proximity field.
    #
    #   R = R_pos + R_progress - R_vel - R_tilt - R_angvel - R_jitter      (non-crash)
    #   R = -k_crash                                                       (crash)
    #
    # The design rule behind these numbers: PROXIMITY PAYS, DISTANCE IS FREE. Nothing here
    # charges a per-step cost that accumulates with episode length, because two earlier
    # versions did and both were unlearnable:
    #
    #   1. "k_dist * curr_dist" per step. Flying to a goal 4.45 m away cost more in
    #      accumulated distance penalty than the one-off crash penalty, so leaving the box
    #      immediately maximised return. Measured over 591k episodes: 100% ended out of
    #      bounds, 0 successes, and mean episode length FELL as training progressed.
    #   2. k_dist removed, k_time kept. The positional field went completely flat -- a
    #      stationary drone scored -k_time per step whether it sat on the goal or 4.45 m
    #      away -- and the only positional signal left was a 12 cm step-function hover
    #      bonus that exploration never reached.
    #
    # Because success no longer ends the episode, every episode runs the full
    # episode_len_steps and the return is the integral of R_pos: "how close, for how long".
    reward_parameters = {
        # Crash penalty. Fires on leaving the env box (see exceed_bounds_margin above).
        # The env has no ground plane and no assets, so out-of-bounds is the ONLY crash
        # source -- obs_dict["crashes"] can never be set by the simulator itself.
        # Modest on purpose: the real cost of crashing is forfeiting the remaining
        # ~600 steps of proximity reward, which dwarfs this.
        "k_crash": [50.0],            # Crash penalty magnitude

        # PROXIMITY (the main term): k_near*exp(-(d/s_near)^2) + k_far*exp(-(d/s_far)^2)
        #
        # Two scales, and both are needed. The NARROW term is the precision peak: it is
        # what makes the difference between 30 cm and 5 cm worth chasing, which a single
        # wide Gaussian is far too flat to express. The WIDE term is the guidance basin,
        # keeping a non-zero gradient all the way out to the env walls so a policy that has
        # never been near the goal still learns which way to fly.
        #
        # Peak is k_near + k_far = 4.0/step at the target, ~0.78 at 1 m, ~0.11 at 4.45 m.
        # Bounded above, so it cannot be farmed into swamping the crash penalty, and it
        # tends to 0 rather than going negative -- being far away is worthless, not costly.
        # sigma_near widened 0.3 -> 0.6: at 0.3 the peak was essentially dead beyond 0.9 m,
        # leaving only the near-flat wide basin to guide the final approach.
        "k_near": [3.0],              # Narrow peak height
        "sigma_near": [0.6],          # Narrow width (m). ALSO the width of the hold-penalty
                                      # gate -- the two must stay equal, see compute_reward.
        "k_far": [1.0],               # Wide basin height
        "sigma_far": [3.0],           # Wide width (m) -- comparable to the env half-extent

        # Progress reward: k_progress * (prev_dist - curr_dist).
        # Cut 10.0 -> 1.0. It telescopes to k_progress*(d_0 - d_final), i.e. it is near
        # potential-based, so it shapes exploration without moving the optimum -- useful
        # early, redundant once R_pos exists. At 10.0 against an R_pos peak of 4.0 it
        # dominated, paying more for dashing at the goal than for settling on it.
        "k_progress": [1.0],

        # "Hold still" penalties: (k_vel*||v|| + k_tilt*tilt + k_angvel*||w||), clamped to
        # penalty_clamp_frac * k_near, then multiplied by exp(-(d/sigma_near)^2).
        #
        # They convert "visit the goal" into "stop at the goal" -- without them the
        # top-scoring behaviour is to keep flying THROUGH the target, re-collecting the
        # proximity peak on every pass.
        #
        # THE CLAMP IS STRUCTURAL, NOT COSMETIC. Sharing sigma_near with R_pos's near term
        # makes the near-field reward [k_near - penalty] * gate(d), so clamping the penalty
        # below k_near forces that bracket positive and the field monotone in distance for
        # ANY flight state. The previous version used an independent sigmoid gate
        # (center 0.5, width 0.1) with no clamp; it ramped in faster than R_pos rose and
        # created a local maximum at 0.9 m with a valley at 0.5 m. Training stalled at
        # 1.29 m mean distance and time-within-0.5 m FELL (0.008 -> 0.002) as the policy
        # learned to avoid the valley. Never reintroduce an independent gate width.
        "penalty_clamp_frac": [0.9],  # Ceiling as a fraction of k_near. Must stay < 1.0.
        "k_vel": [0.5],               # Linear velocity penalty coefficient
        "k_tilt": [1.0],              # Tilt penalty coefficient
        "k_angvel": [0.5],            # Angular velocity penalty coefficient

        # Action jitter penalty: -k_jitter * ||a_curr - a_prev||.
        # Computed in the NORMALIZED [-1, 1] action space, not on transformed commands, so
        # all four channels share a range and one coefficient means the same thing for each.
        "k_jitter": [0.5],            # Jitter penalty coefficient
    }

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