"""
Simple Hover SNN Task module for Aerial Gym Simulator.

A hover task where:
 - The quadrotor has access to an onboard IMU.
 - The environment consists of a ground plane only, with no obstacles.
 - The quadrotor must maintain a stable hover position using attitude commands.
 - No position information is provided in observations.

This task is identical to simple_hover - the SNN is only a different
neural network architecture used in training, not a different task.
"""

from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import quat_apply_inverse, quat_axis, get_euler_xyz
from aerial_gym.utils.logging import CustomLogger

import torch
import numpy as np
from gym.spaces import Dict, Box
from typing import Tuple, Dict as DictType
logger = CustomLogger("simple_hover_snn_task")


class HoverSNNTask(BaseTask):
    """
    Simple hover SNN task for quadrotor.
    - Observation space: IMU readings (linear acceleration, angular velocity)
    - Action space: Attitude commands (roll, pitch, yaw_rate, thrust)

    Succeeds if the quadrotor maintains a stable hover.
    """

    def __init__(self,
                 task_config,
                 seed=None,
                 num_envs=None,
                 headless=None,
                 device=None,
                 use_warp=None
    ):
        """
        Initialize the Simple Hover SNN Task.

        Args:
            task_config: Task configuration class
            seed: Random seed for reproducibility (overrides task_config.seed if provided)
            num_envs: Number of parallel environments (overrides task_config.num_envs if provided)
            headless: Runs without rendering if True (overrides task_config.headless if provided)
            device: Device to run the simulation on (overrides task_config.device if provided)
            use_warp: Whether to use Warp for rendering (overrides task_config.use_warp if provided)
        """
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = task_config.device

        # Convert reward parameters to tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key],
                device=self.device
            )

        logger.info("Building environment for Simple Hover SNN Task...")
        logger.info(
            "\nSim Name: {}\nEnv Name: {}\nRobot Name: {}\nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name
            )
        )

        logger.info(
            "\nNum Envs: {}\nUse Warp: {}\nHeadless: {}\nDevice: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
                self.task_config.device
            )
        )

        # Build the simulation with SimBuilder
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.task_config.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless
        )

        # Initialize action tensors
        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False
        )

        self.prev_actions = torch.zeros_like(self.actions, device=self.device)

        # Get observations dictionary reference from environment
        # This dict is updated in-place by the sim step
        self.obs_dict = self.sim_env.get_obs()

        # Initialize tensors (rewards, terminations, truncations)
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        # Define observations Metadata for RL libraries
        # (Doesn't hold actual data - data is in obs_dict)
        self.observation_space = Dict(
            {"observations": Box(
                low=-1.0,
                high=1.0,
                shape=(self.task_config.observation_space_dim,),
                dtype=np.float32
                )
            }
        )

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32
        )


        # Initialize task_obs dict
        self.task_obs = {
            "observations": torch.zeros(
                (self.task_config.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False
            ),
            "collisions": torch.zeros(
                (self.task_config.num_envs, 1),
                device=self.device,
                requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.task_config.num_envs, 1),
                device=self.device,
                requires_grad=False
            )
        }

        self.num_envs = self.sim_env.num_envs
        self.counter = 0
        # Info dictionary for additional info logging
        self.infos = {}

        # Fixed target position at origin for all envs (matching position_setpoint_task)
        self.target_position = torch.zeros(
            (self.num_envs, 3),
            device=self.device
        )

        # Success tracking: count consecutive steps within threshold
        self.success_counter = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32  # Use float for JIT compatibility
        )

        # Track if success bonus was already awarded (to avoid double-counting)
        self.success_achieved = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32  # Use float for JIT compatibility
        )

        # Cumulative success statistics (for monitoring)
        self.total_successes = 0  # Total number of successful episode completions
        self.total_episodes = 0   # Total number of episodes completed (success or failure)

        # Windowed success rate: rolling buffer of the last N completed-episode
        # outcomes (1.0 = success, 0.0 = failure). success_rate over the window
        # = mean of the filled portion of the buffer.
        self.success_rate_window_episodes = int(
            getattr(self.task_config, "success_rate_window_episodes", 4096)
        )
        self.success_window = torch.zeros(
            self.success_rate_window_episodes,
            device=self.device,
            dtype=torch.float32,
        )
        self.success_window_ptr = 0      # next write index (circular)
        self.success_window_filled = 0   # number of valid entries (<= window size)

        # Episode step counter for each environment
        self.episode_steps = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.int32
        )

        # Previous distance for progress reward (initialized to large value)
        self.prev_dist = torch.ones(
            self.num_envs,
            device=self.device,
            dtype=torch.float32
        ) * 5.0  # Start with reasonable initial distance

        logger.info(
            f"Simple Hover SNN Task initialized with {self.num_envs} environments."
            f" Observation space dim: {self.task_config.observation_space_dim},"
            f" Action space dim: {self.task_config.action_space_dim}"
        )


    def close(self):
        """
        Clean up the environment and free resources.
        """
        self.sim_env.delete_env()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_success_rate_window(self, num_episodes):
        """
        (Re)allocate the windowed success-rate buffer.

        Used to override the window size from the YAML config after the task
        is constructed. Resets any accumulated window statistics.

        Args:
            num_episodes: Number of most-recent completed episodes to average over.
        """
        self.success_rate_window_episodes = int(num_episodes)
        self.success_window = torch.zeros(
            self.success_rate_window_episodes,
            device=self.device,
            dtype=torch.float32,
        )
        self.success_window_ptr = 0
        self.success_window_filled = 0

    def _record_episode_outcomes(self, outcomes):
        """
        Append per-episode outcomes (1.0 success / 0.0 failure) into the rolling
        success-rate window (circular buffer).

        Args:
            outcomes: 1D float tensor, one entry per completed episode this step.
        """
        n = outcomes.numel()
        if n == 0:
            return
        window_size = self.success_rate_window_episodes
        # If more episodes complete this step than the window holds, keep only
        # the most recent `window_size` of them.
        if n >= window_size:
            self.success_window[:] = outcomes[-window_size:]
            self.success_window_ptr = 0
            self.success_window_filled = window_size
            return
        ptr = self.success_window_ptr
        end = ptr + n
        if end <= window_size:
            self.success_window[ptr:end] = outcomes
        else:
            first = window_size - ptr
            self.success_window[ptr:] = outcomes[:first]
            self.success_window[: end - window_size] = outcomes[first:]
        self.success_window_ptr = end % window_size
        self.success_window_filled = min(self.success_window_filled + n, window_size)

    def reset(self):
        """
        Reset the task and environment.

        Returns:
            observations: Initial observations after reset
        """
        self.infos = {}
        self.sim_env.reset()
        # Target remains at origin [0, 0, 0] (matching position_setpoint_task)
        self.target_position[:, 0:3] = 0.0
        # Reset success tracking for all environments
        self.success_counter[:] = 0
        self.success_achieved[:] = 0
        # Reset episode step counter
        self.episode_steps[:] = 0
        # Initialize prev_dist to actual distance (avoid spurious progress reward on first step)
        self.obs_dict = self.sim_env.get_obs()
        robot_position = self.obs_dict["robot_position"]
        self.prev_dist = torch.norm(robot_position - self.target_position, dim=1)
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        """
        Reset specific environments by their IDs.
        In this task - reset robot state, target stays at origin.
        """
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        # Target remains at origin [0, 0, 0] (matching position_setpoint_task)
        self.target_position[env_ids, 0:3] = 0.0
        # Reset success tracking for reset environments
        self.success_counter[env_ids] = 0
        self.success_achieved[env_ids] = 0
        # Initialize prev_dist to actual distance for reset envs (avoid spurious progress reward)
        robot_position = self.obs_dict["robot_position"]
        self.prev_dist[env_ids] = torch.norm(
            robot_position[env_ids] - self.target_position[env_ids], dim=1
        )
        return

    def render(self):
        """
        Render the current state of the environment.
        """
        return None

    def step(self, actions):
        """
        Take a simulation step with the given actions.
        Does the following:
            - Save current actions to prev_actions before updating
            - Transform actions
            - Step the simulation
            - Compute rewards
            - Check for terminations/truncations
            - Handle resets if needed

        Returns:
            Tuple of (observations, rewards, terminations, truncations, info)


        """

        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        # Increment episode step counter for all active environments
        self.episode_steps += 1

        # Run the actions in the simulation environment (direct pass, no transformation)
        self.sim_env.step(self.actions)

        # Calculate the rewards and check for terminations/truncations
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        # Check for success: hovering within threshold
        position = self.obs_dict["robot_position"]
        pos_error = torch.norm(position - self.target_position, dim=1)
        within_threshold = pos_error < self.task_config.success_threshold

        # Increment counter if within threshold, reset otherwise
        self.success_counter = torch.where(
            within_threshold,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter)
        )

        # Success = held position within threshold for the required # of steps.
        # Success TERMINATES the episode (truncation). This is safe now that the
        # per-step time penalty (-k_time) exceeds the per-step hover bonus
        # ceiling (k_hover < k_time), so loitering is net-negative: the agent can
        # no longer profit by hovering well while avoiding the success condition.
        # Succeeding quickly and ending the episode (with success_bonus) is the
        # reward-optimal behavior, re-coupling reward with the success metric.
        success = self.success_counter >= self.task_config.success_hold_steps
        num_successes = success.sum().item()

        # One-time terminal success bonus, awarded the step success is first met.
        # The success_achieved latch ensures it is awarded once per episode; it
        # is reset to 0 in reset()/reset_idx(), so it re-arms each episode.
        newly_successful = success.bool() & (self.success_achieved < 0.5)
        self.rewards = self.rewards + newly_successful.float() * self.task_config.success_bonus
        # Latch: mark envs that have achieved success at any point this episode.
        self.success_achieved = torch.where(success.bool(), 1.0, self.success_achieved)

        # Terminate on success (use truncations: post_reward_calculation_step()
        # resets envs whose truncation flag is set).
        self.truncations[:] = torch.where(
            success,
            torch.ones_like(self.truncations),
            self.truncations
        )

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        # Also truncate if episode length exceeded (timeout = failure)
        timeout = self.sim_env.sim_steps > self.task_config.episode_len_steps
        num_timeouts = timeout.sum().item()
        self.truncations[:] = torch.where(
            timeout, 1, self.truncations
        )

        # One-time timeout penalty for episodes that hit the time limit WITHOUT
        # ever achieving success. Mirrors the crash penalty so that crashing to
        # escape the per-step time penalty is not a profitable shortcut (giving
        # up ends with a comparable terminal cost whether by crash or timeout).
        # Computed from the success_achieved latch before reset_idx() clears it.
        timeout_failure = timeout.bool() & (self.success_achieved < 0.5)
        self.rewards = self.rewards - timeout_failure.float() * self.task_config.timeout_penalty

        # Track statistics before reset.
        # Episodes end on success, crash, or timeout. An episode counts as a
        # success if the hover-hold condition was achieved at any point during it
        # -> use the success_achieved latch, captured here before reset_idx()
        # clears it (the latch is set this same step when success fires).
        will_reset = (self.terminations | self.truncations).bool()
        num_resets = will_reset.sum().item()

        # Per-episode success outcome (1.0 if the episode ever achieved success).
        episode_success = self.success_achieved.bool()

        # Record per-episode outcomes into the rolling success-rate window.
        if num_resets > 0:
            episode_outcomes = episode_success[will_reset].float()
            self._record_episode_outcomes(episode_outcomes)

        # Track average episode length for episodes that succeeded and are
        # resetting this step (steps elapsed up to crash/timeout).
        avg_success_steps = 0.0
        successful_resets_mask = episode_success & will_reset
        num_successful_episodes = successful_resets_mask.sum().item()
        if num_successful_episodes > 0:
            successful_episode_lengths = self.episode_steps[successful_resets_mask]
            avg_success_steps = successful_episode_lengths.float().mean().item()

        if num_resets > 0:
            self.total_episodes += num_resets
            self.total_successes += num_successful_episodes

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            # Reset episode step counter for reset environments
            self.episode_steps[reset_envs] = 0
            self.reset_idx(reset_envs)

        # Calculate success rate (percentage of completed episodes that were successful)
        success_rate = (self.total_successes / self.total_episodes * 100.0) if self.total_episodes > 0 else 0.0

        # Windowed success rate: fraction of the last N completed episodes that
        # were successful (percentage). N = success_rate_window_episodes.
        if self.success_window_filled > 0:
            success_rate_window = (
                self.success_window[: self.success_window_filled].mean().item() * 100.0
            )
        else:
            success_rate_window = 0.0

        # Log metrics for tensorboard (IsaacAlgoObserver logs scalar values from infos)
        self.infos = {
            # Instantaneous metrics (per step)
            "successes": num_successes,           # Envs currently holding the success condition this step
            "timeouts": num_timeouts,             # Number of timeouts this step
            "avg_success_episode_length": avg_success_steps,  # Avg episode length of successful episodes resetting this step

            # Cumulative metrics (over entire training run)
            "total_successes": self.total_successes,  # Total episodes that achieved success
            "total_episodes": self.total_episodes,    # Total episodes completed
            "success_rate": success_rate,             # Success rate percentage (successful episodes / total episodes)

            # Windowed metric: success rate over the last N completed episodes
            "success_rate_window": success_rate_window,  # Logged as success_rate_window/iter, /frame, /time
        }

        if not self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos
        )

    def process_obs_for_task(self):
        """
        Build 13D observation vector (matching position_setpoint_task)

        [0:3]   Position error: target - robot_position (NO normalization)
        [3:7]   Robot orientation (quaternion)
        [7:10]  Body Linear Velocity (vx, vy, vz) (NO normalization)
        [10:13] Body Angular Velocity (wx, wy, wz)
        """

        # Position error (target - robot_position)
        self.task_obs["observations"][:, 0:3] = (
            self.target_position - self.obs_dict["robot_position"]
        )

        # Robot orientation (quaternion)
        self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_orientation"]

        # Body linear velocity (NO normalization)
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]

        # Body angular velocity
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        """
        Compute rewards and check for crashes.

        Potential-based reward structure:
        1. Progress reward: positive for moving toward goal
        2. Gated tilt penalty: penalize pitch/roll deviation when near goal
        3. Gated angular velocity penalty: penalize rotation when near goal
        4. Action jitter penalty: penalize rapid action changes
        5. Hover bonus: reward for being near goal with low velocity
        6. Crash penalty: large negative for crashes

        Args:
            obs_dict: Dictionary of observations from the environment

        Returns:
            rewards: Tensor of shape (num_envs,) with computed rewards
            crashes: Tensor of shape (num_envs,) with crash flags (1 if crashed, 0 otherwise)
        """
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_linvel = obs_dict["robot_body_linvel"]
        robot_angvel = obs_dict["robot_body_angvel"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )

        rewards, crashes, self.success_counter, self.prev_dist = compute_reward(
            pos_error_vehicle_frame,
            robot_linvel,
            robot_angvel,
            robot_vehicle_orientation,
            obs_dict["crashes"],
            self.actions,
            self.prev_actions,
            self.success_counter,
            self.prev_dist,
            self.task_config.reward_parameters,
        )
        return rewards, crashes


@torch.jit.script
def compute_reward(
    pos_error,
    robot_linvel,
    robot_angvel,
    robot_orientation,
    crashes,
    current_action,
    prev_actions,
    success_counter,
    prev_dist,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    """
    Potential-based reward for hover task.

    Components:
    0. Distance penalty: -k_dist * curr_dist
       - Constant pressure to get closer to goal
    1. Progress reward: k_progress * (prev_dist - curr_dist)
       - Positive when moving toward goal, negative when moving away
    2. Gated tilt penalty: -k_tilt * g(dist) * sqrt(pitch^2 + roll^2)
       - g(dist) = sigmoid((gate_center - dist) / gate_width)
       - Penalizes attitude deviation when near goal
    3. Gated angular velocity penalty: -k_angvel * g(dist) * ||angvel||
       - Penalizes rotation when near goal
    4. Action jitter penalty: -k_jitter * ||action_diff||
    5. Hover bonus: +k_hover * exp(-||vel|| / vel_scale) when dist < threshold
       - Rewards being near goal with low velocity (exponential decay, no hard cutoff)
    6. Crash penalty: -k_crash (one-time)
    7. Time penalty: -k_time every (non-crash) step
       - Constant per-step cost. With k_hover < k_time, loitering is net-negative,
         so the agent is pushed to achieve hover quickly and end the episode on
         success rather than farm the hover bonus.

    Optimal behavior: approach goal quickly, then stabilize with level attitude,
    minimal rotation, minimal velocity, and smooth control.
    """
    # Extract parameters
    k_dist = parameter_dict["k_dist"][0]
    k_progress = parameter_dict["k_progress"][0]
    gate_midpoint = parameter_dict["gate_center"][0]
    gate_steepness = parameter_dict["gate_width"][0]
    k_jitter = parameter_dict["k_jitter"][0]
    k_tilt = parameter_dict["k_tilt"][0]
    k_angvel = parameter_dict["k_angvel"][0]
    k_hover = parameter_dict["k_hover"][0]
    threshold_hover = parameter_dict["threshold_hover"][0]
    vel_scale_hover = parameter_dict["vel_scale_hover"][0]
    k_crash = parameter_dict["k_crash"][0]
    max_distance = parameter_dict["max_distance"][0]
    k_time = parameter_dict["k_time"][0]

    # Current distance to target
    curr_dist = torch.norm(pos_error, dim=1)

    # ============================================================
    # CRASH CHECK (distance beyond max)
    # ============================================================
    crashes = torch.where(curr_dist > max_distance, torch.ones_like(crashes), crashes)

    # ============================================================
    # GATING FUNCTION (SIGMOID)
    # ============================================================
    sigma = torch.sigmoid((gate_midpoint - curr_dist) / gate_steepness)
    
    # ============================================================
    # 1. PROGRESS REWARD (positive for approaching goal)
    # ============================================================
    progress = prev_dist - curr_dist  # Positive when getting closer
    R_progress = k_progress * progress  # Positive reward

    # ============================================================
    # 2. TILT PENALTY (negative for pitch/roll deviation)
    # ============================================================
    # Extract roll, pitch, yaw from quaternion using built-in function
    roll, pitch, yaw = get_euler_xyz(robot_orientation)

    # Tilt magnitude: sqrt(pitch^2 + roll^2)
    tilt_magnitude = torch.sqrt(pitch ** 2 + roll ** 2)
    R_tilt = k_tilt * sigma * tilt_magnitude  # Negative penalty (more negative for stronger tilt)

    # ============================================================
    # 3. ANGULAR VELOCITY PENALTY (negative for rotation)
    # ============================================================
    angvel_magnitude = torch.norm(robot_angvel, dim=1)
    R_angvel = k_angvel * sigma * angvel_magnitude  # Negative penalty (more negative for higher angular velocity)

    # ============================================================
    # 4. ACTION JITTER PENALTY (negative for rapid action changes)
    # ============================================================
    action_diff = torch.norm(current_action - prev_actions, dim=1)
    R_jitter = k_jitter * action_diff  # Negative penalty (more negative for stronger jitter)

    # ============================================================
    # 5. HOVER BONUS (positive for stable hovering near goal)
    # ============================================================
  
    # Update success counter for episode termination tracking
    success_counter = torch.where(
        curr_dist < threshold_hover,
        success_counter + 1,
        torch.zeros_like(success_counter)
    )

    # Compute velocity magnitude
    vel_magnitude = torch.norm(robot_linvel, dim=1)

    # Velocity bonus: exponential decay with velocity
    # vel=0 → bonus=1.0, vel=vel_scale → bonus≈0.37, approaches 0 asymptotically
    vel_bonus_scale = torch.exp(-vel_magnitude / vel_scale_hover)

    # Hover bonus: only active when within distance threshold, scaled by velocity
    R_hover = torch.where(
        curr_dist < threshold_hover,
        k_hover * vel_bonus_scale,  # Positive reward (more positive when stationary)
        torch.zeros_like(curr_dist)
    )

    # ============================================================
    # TOTAL REWARD: R_total = -k_dist*dist + R_progress - R_tilt - R_angvel - R_jitter + R_hover
    # ============================================================
    # If crashed: only crash penalty
    # Otherwise: sum all reward components
    R_total = torch.where(
        crashes > 0.0,
        -k_crash * torch.ones_like(curr_dist),
        -k_dist * curr_dist + R_progress - R_tilt - R_angvel - R_jitter + R_hover - k_time
    )

    # Update prev_dist for next step (return as output)
    new_prev_dist = curr_dist.clone()

    return R_total, crashes, success_counter, new_prev_dist
