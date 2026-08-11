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
from aerial_gym.utils.math import quat_apply_inverse, quat_axis, get_euler_xyz, ssa
from aerial_gym.utils.logging import CustomLogger

import torch
import numpy as np
from gym.spaces import Dict, Box
from typing import Tuple, Dict as DictType
from isaacgym import gymapi, gymutil
logger = CustomLogger("hover_task")
logger.setLevel("INFO")

class HoverTask(BaseTask):
    """
    Simple hover task for quadrotor.
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
        Initialize the Simple Hover Task.

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

        logger.info("Building environment for Hover Task...")
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

        # Build the simulation with SimBuilder. Keep the builder: build_env() returns an
        # EnvManager, but delete_env() lives on the builder, so close() needs this handle.
        self.sim_builder = SimBuilder()
        self.sim_env = self.sim_builder.build_env(
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

        # Action transformation function
        self.action_transformation_function = (
            self.task_config.action_transformation_function
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

        self.target_inset = 5.0 * torch.tensor(self.task_config.success_threshold, device=self.device)

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

        # Debug visualization: goal (red) and spawn (green) wireframe spheres.
        #
        # Not cosmetic for this task. The target is sampled anywhere in the env box and is
        # otherwise INVISIBLE, so watching a playback shows a drone flying towards a point
        # the viewer cannot see -- "is it hovering on target?" is unanswerable by eye.
        #
        # The goal sphere is drawn at success_threshold so what you see IS the capture
        # region, not an arbitrary marker: the episode succeeds when the drone stays inside
        # that sphere for success_hold_steps.
        self._headless = self.task_config.headless
        if not self._headless:
            self._gym = self.sim_env.IGE_env.gym
            self._viewer = self.sim_env.IGE_env.viewer.viewer
            self._env_handles = self.sim_env.IGE_env.env_handles
            self._goal_sphere = gymutil.WireframeSphereGeometry(
                float(self.task_config.success_threshold), 16, 16, None, color=(1, 0, 0)
            )
            self._start_sphere = gymutil.WireframeSphereGeometry(
                0.15, 12, 12, None, color=(0, 1, 0)
            )
            self._start_positions = self.obs_dict["robot_position"].clone()

        logger.info(
            f"Hover Task initialized with {self.num_envs} environments."
            f" Observation space dim: {self.task_config.observation_space_dim},"
            f" Action space dim: {self.task_config.action_space_dim}"
        )


    def close(self):
        """
        Clean up the environment and free resources.

        delete_env() is a SimBuilder method, not an EnvManager one — calling it on
        self.sim_env (as every shipped aerial_gym task does) raises AttributeError, so
        cleanup never runs and VRAM is not reclaimed between task constructions.
        """
        self.sim_builder.delete_env()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _sample_target(self, env_ids):
        """Uniform target in the env box"""

        lower = self.obs_dict["env_bounds_min"][env_ids] + self.target_inset
        upper = self.obs_dict["env_bounds_max"][env_ids] - self.target_inset
        ratio = torch.rand(len(env_ids), 3, device=self.device)
        return lower + (upper - lower) * ratio

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
        all_envs = torch.arange(self.num_envs, device=self.device)
        self.target_position[:, 0:3] = self._sample_target(all_envs)
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
        self.target_position[env_ids, 0:3] = self._sample_target(env_ids)
        # Reset success tracking for reset environments
        self.success_counter[env_ids] = 0
        self.success_achieved[env_ids] = 0
        # Initialize prev_dist to actual distance for reset envs (avoid spurious progress reward)
        robot_position = self.obs_dict["robot_position"]
        self.prev_dist[env_ids] = torch.norm(
            robot_position[env_ids] - self.target_position[env_ids], dim=1
        )
        # Spawn markers follow the new episode's spawn, so the green sphere always shows
        # where THIS episode started rather than where the run began.
        if not self._headless:
            self._start_positions[env_ids] = robot_position[env_ids].clone()
        return

    def render(self):
        """
        Render the current state of the environment.
        """
        return None

    def _draw_debug_visuals(self):
        """Draw the goal (red, radius = success_threshold) and spawn (green) spheres.

        clear_lines() wipes the whole viewer's debug geometry, so everything has to be
        redrawn every frame -- this cannot be done once at reset.
        """
        self._gym.clear_lines(self._viewer)
        for i in range(self.num_envs):
            goal_pose = gymapi.Transform(
                p=gymapi.Vec3(*self.target_position[i].cpu().numpy())
            )
            gymutil.draw_lines(
                self._goal_sphere, self._gym, self._viewer,
                self._env_handles[i], goal_pose,
            )
            start_pose = gymapi.Transform(
                p=gymapi.Vec3(*self._start_positions[i].cpu().numpy())
            )
            gymutil.draw_lines(
                self._start_sphere, self._gym, self._viewer,
                self._env_handles[i], start_pose,
            )

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
        
        # Transform network outputs to controller commands
        transformed_action = self.action_transformation_function(actions)

        # The jitter penalty tracks the RAW (normalized) action, not the transformed one.
        #
        # After transformation the four channels carry different units -- thrust is
        # dimensionless in [-1, 1], roll/pitch are radians in [-pi/4, pi/4], yaw_rate is
        # rad/s in [-pi/3, pi/3] -- so ||a_curr - a_prev|| would sum quantities that are not
        # comparable, and whichever channel has the widest numeric range would dominate the
        # norm. k_jitter would then mean something different for every channel.
        #
        # In the [-1, 1] network space all four channels already share a range, so the norm
        # is dimensionless by construction and each channel contributes its change as a
        # fraction of what it can command. (The nav task solves the same problem the other
        # way, dividing the transformed action by a per-channel _action_scale.)
        #
        # Clamped, so that two successive out-of-range outputs that both saturate to the
        # same command are correctly scored as zero jitter rather than as movement.
        current_action = torch.clamp(actions, -1.0, 1.0).clone()

        # Step the simulation and update the observation dictionary
        self.sim_env.step(actions=transformed_action)
        self.actions = current_action


        # Increment episode step counter for all active environments
        self.episode_steps += 1

        # Calculate the rewards and check for terminations/truncations
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        self.prev_actions[:] = current_action

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
        #
        # PURELY A METRIC. It pays no bonus and does NOT end the episode, and both of those
        # are deliberate:
        #   - terminating on success caps how long the agent is ever asked to hold, so it
        #     never learns to hold longer than success_hold_steps -- the opposite of a
        #     hover task;
        #   - a large terminal bonus re-creates the reward cliff the dense R_pos field was
        #     introduced to remove, and made "try and fail" score far worse than "give up
        #     immediately" (see compute_reward's docstring).
        # The agent is paid continuously for proximity instead, so reward and this metric
        # already point the same way without coupling them.
        success = self.success_counter >= self.task_config.success_hold_steps
        num_successes = success.sum().item()

        # Latch: marks envs that have achieved success at any point this episode. Read for
        # the per-episode outcome below, reset in reset()/reset_idx().
        self.success_achieved = torch.where(success.bool(), 1.0, self.success_achieved)

        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        # Truncate on the step limit. This is now the ONLY way a non-crashed episode ends,
        # which is what makes the return the integral of R_pos over a fixed horizon.
        timeout = self.sim_env.sim_steps > self.task_config.episode_len_steps
        num_timeouts = timeout.sum().item()
        self.truncations[:] = torch.where(
            timeout, 1, self.truncations
        )

        # rl_games value-bootstrap signal, consumed by a2c_common when value_bootstrap=True:
        # shaped_rewards += gamma * V(s') * time_outs. Bootstrap ONLY on the artificial
        # step-limit cutoff -- success, crash and out-of-bounds are TRUE terminals whose
        # value really is zero, so bootstrapping them would inflate their returns.
        #
        # Load-bearing here in a way it is not for most tasks: EVERY non-crashed episode
        # ends at the step limit, so without this key rl_games treats the horizon as a real
        # terminal and the critic learns a cliff at episode_len_steps -- which would undo
        # the whole point of paying continuously for proximity.
        #
        # Crashes are excluded because they ARE true terminals. Success is no longer
        # excluded: it neither ends the episode nor prevents the timeout, so an env that
        # succeeded still deserves the bootstrap when it later reaches the limit.
        #
        # Held in a local and written into the self.infos literal below, NOT assigned here:
        # that literal REBUILDS the dict, so anything set on self.infos before it is lost.
        time_outs = (timeout.bool() & (~self.terminations.bool())).float()

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

        # After the resets, so the spheres show the NEW episode's goal and spawn rather
        # than one frame of the episode that just ended.
        if not self._headless:
            self._draw_debug_visuals()

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

        # CONTINUOUS progress diagnostics. `pos_error` is this step's distance for every
        # env, captured BEFORE the resets above moved anyone, so it describes the flying
        # population rather than fresh spawns.
        #
        # These exist because the success metric is a binary that stays at exactly 0 until
        # the policy can hold a 10 cm ball for 3.6 s, which can be many hundreds of epochs
        # of real progress -- during which it reports nothing at all. Mean distance and
        # near-fraction move from the first epoch, so they answer "is R_pos working?" long
        # before success does. Watch these first on any retrain.
        mean_dist = float(pos_error.mean())
        frac_near = float((pos_error < self.task_config.diagnostic_radius).float().mean())

        # Log metrics for tensorboard (IsaacAlgoObserver logs scalar values from infos)
        self.infos = {
            # FUNCTIONAL, not logging: rl_games' value-bootstrap mask (see above).
            # Per-env tensor, unlike every other entry here, which are scalars.
            "time_outs": time_outs,

            # Continuous progress signals -- move long before "successes" leaves zero.
            "metrics/mean_dist_to_target": mean_dist,
            "metrics/frac_within_radius": frac_near,

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

        # Position error in vehicle frame (target - robot_position)
        self.task_obs["observations"][:, 0:3] = quat_apply_inverse(
            self.obs_dict["robot_vehicle_orientation"],
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

        # Effective termination box. The margin is applied here rather than inside
        # compute_reward so it can stay a plain float in the config instead of being
        # tensorized into reward_parameters.
        bounds_min = obs_dict["env_bounds_min"]
        bounds_max = obs_dict["env_bounds_max"]
        margin = self.task_config.exceed_bounds_margin
        if margin != 1.0:
            center = 0.5 * (bounds_min + bounds_max)
            half_extent = 0.5 * (bounds_max - bounds_min) * margin
            bounds_min = center - half_extent
            bounds_max = center + half_extent

        rewards, crashes, self.prev_dist = compute_reward(
            pos_error_vehicle_frame,
            robot_position,
            bounds_min,
            bounds_max,
            robot_linvel,
            robot_angvel,
            robot_vehicle_orientation,
            obs_dict["crashes"],
            self.actions,
            self.prev_actions,
            self.prev_dist,
            self.task_config.reward_parameters,
        )
        return rewards, crashes


@torch.jit.script
def compute_reward(
    pos_error,
    robot_position,
    bounds_min,
    bounds_max,
    robot_linvel,
    robot_angvel,
    robot_orientation,
    crashes,
    current_action,
    prev_actions,
    prev_dist,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """
    Dense proximity reward for the go-to-and-hover task.

    THE CENTRAL TERM IS R_pos, AND IT IS POSITIVE AND BOUNDED. Everything else shapes it.
    That sign matters more than any coefficient here: an earlier version paid
    `-k_dist * curr_dist` per step, which made the accumulated cost of FLYING to a distant
    goal exceed the one-off crash penalty, so leaving the box immediately was the
    return-maximising policy and PPO duly learned it (measured: 100% of episodes ended out
    of bounds, 0 successes in 591k episodes, and mean episode length FELL over training as
    the policy got better at dying quickly).

    Removing that term without replacing it was no better -- it left the positional field
    completely FLAT: a stationary drone earned -k_time per step whether it sat on the goal
    or 4.45 m away, with the only positional signal a 12 cm step-function hover bonus that
    random exploration never finds. Nothing rewarded BEING close, only BECOMING close.

    So: distance costs nothing, proximity pays.

    Components:
    1. Proximity:  +k_near * exp(-(d/s_near)^2) + k_far * exp(-(d/s_far)^2)
       - The reward the task is actually about. Two scales on purpose: the narrow term
         makes the last few cm worth fighting for, the wide one keeps a usable gradient
         alive right across the env box. Bounded above, so it cannot be farmed into
         dominating the terminal penalties, and asymptotically 0 rather than negative, so
         being far away is merely worthless instead of expensive.
    2. Progress:   k_progress * (prev_dist - curr_dist)
       - Near potential-based (Phi(s') - Phi(s)), so it shapes exploration without moving
         the optimum. Kept small: with a real positional field it is redundant, and at a
         large weight it pays for dashing at the goal rather than settling on it.
    3. Hold-still group:  -min(k_vel*||v|| + k_tilt*tilt + k_angvel*||w||, 0.9*k_near) * g(d)
       - g(d) = exp(-(d/s_near)^2), THE SAME GAUSSIAN as the proximity peak, and the sum is
         clamped below k_near. Together those two facts make the near-field reward
         [k_near - penalty] * g(d), i.e. a positive multiple of a Gaussian, hence monotone
         decreasing in d for ANY flight state. Without the group these penalties let the
         drone maximise R_pos by repeatedly flying THROUGH the target; without the shared
         width and the clamp they instead build a barrier around it. See the block comment
         at the implementation.
    4. Action jitter:           -k_jitter * ||a_curr - a_prev||   (normalized action space)
    5. Crash:                   -k_crash, one-time, on leaving the env box

    There is deliberately NO per-step time penalty and NO success bonus. Success no longer
    ends the episode, so every episode runs to the step limit and the return becomes the
    integral of R_pos over the episode -- literally "how close was it, for how long", which
    is the go-to-and-hover objective. A time penalty on top of that only re-creates the
    die-early pathology, and a terminal success bonus re-creates the cliff.

    Optimal behavior: reach the goal quickly, then hold it level, still, and smoothly, for
    the rest of the episode.
    """
    # Extract parameters
    k_near = parameter_dict["k_near"][0]
    sigma_near = parameter_dict["sigma_near"][0]
    k_far = parameter_dict["k_far"][0]
    sigma_far = parameter_dict["sigma_far"][0]
    k_progress = parameter_dict["k_progress"][0]
    penalty_clamp_frac = parameter_dict["penalty_clamp_frac"][0]
    k_jitter = parameter_dict["k_jitter"][0]
    k_tilt = parameter_dict["k_tilt"][0]
    k_angvel = parameter_dict["k_angvel"][0]
    k_vel = parameter_dict["k_vel"][0]
    k_crash = parameter_dict["k_crash"][0]

    # Current distance to target
    curr_dist = torch.norm(pos_error, dim=1)

    # ============================================================
    # CRASH CHECK (robot left the environment box)
    # ============================================================
    # Tested against the WORLD box, not a radius around the target: the env is a box, so
    # a sphere centred on the goal both cuts the corners off and makes "out of bounds"
    # depend on where the goal happens to have been sampled. With an empty env (no ground
    # plane, no assets) nothing can be collided with, so this is the only crash source.
    exceed = ((robot_position < bounds_min) | (robot_position > bounds_max)).any(dim=1)
    crashes = torch.where(exceed, torch.ones_like(crashes), crashes)

    # ============================================================
    # GATING FUNCTION (SIGMOID)
    # ============================================================
    # The gate is the SAME GAUSSIAN as the proximity peak below, and that is the whole
    # trick -- see the "hold still" penalty block for why. It is deliberately not an
    # independent sigmoid with its own centre and width: that version put a local maximum
    # in the reward field at 0.9 m and a valley at 0.5 m, and training parked at a mean
    # distance of 1.29 m with the fraction of time inside 0.5 m FALLING (0.008 -> 0.002)
    # as the policy learned to avoid the valley.
    gate = torch.exp(-((curr_dist / sigma_near) ** 2))

    # ============================================================
    # 1. PROXIMITY REWARD (positive, bounded, smooth -- the main term)
    # ============================================================
    # Sum of a narrow and a wide Gaussian in the distance to target. Monotone decreasing
    # in d everywhere, so every metre closer pays, with no cliff for exploration to fall
    # off. See the docstring for why this is positive rather than a distance penalty.
    R_pos = k_near * gate + k_far * torch.exp(-((curr_dist / sigma_far) ** 2))

    # ============================================================
    # 2. PROGRESS REWARD (positive for approaching goal)
    # ============================================================
    progress = prev_dist - curr_dist  # Positive when getting closer
    R_progress = k_progress * progress  # Positive reward

    # ============================================================
    # 3. "HOLD STILL" PENALTIES -- gated, and CLAMPED BELOW k_near
    # ============================================================
    # Velocity, tilt and angular rate, summed and applied through the same Gaussian as the
    # proximity peak. They are what turn "visit the goal" into "stop at the goal": without
    # them the top-scoring behaviour is to keep flying THROUGH the target, re-collecting
    # the peak on every pass.
    #
    # THE CLAMP IS WHAT MAKES THE REWARD FIELD MONOTONE, and it is not a safety net -- it
    # is the mechanism. Because both R_pos's near term and this penalty are the SAME
    # gate(d), the near-field reward collapses to
    #
    #     [k_near - penalty_raw] * gate(d)  +  k_far * exp(-(d/sigma_far)^2)
    #
    # and clamping penalty_raw below k_near forces the bracket positive. A positive
    # multiple of a Gaussian is monotone decreasing in d, so moving closer ALWAYS pays,
    # for any velocity, tilt or spin -- however badly the vehicle happens to be flying.
    #
    # Without the clamp (and with an independently-parameterised sigmoid gate) the penalty
    # ramped in faster than R_pos rose, producing a local maximum at 0.9 m and a valley at
    # 0.5 m. Training stalled at 1.29 m mean distance and the time spent within 0.5 m
    # DECREASED as the policy learned to avoid the valley. Do not reintroduce a gate with
    # its own centre/width, and do not remove the clamp.
    roll, pitch, yaw = get_euler_xyz(robot_orientation)
    roll, pitch = ssa(roll), ssa(pitch)
    tilt_magnitude = torch.sqrt(pitch ** 2 + roll ** 2)
    vel_magnitude = torch.norm(robot_linvel, dim=1)
    angvel_magnitude = torch.norm(robot_angvel, dim=1)

    hold_penalty_raw = (
        k_vel * vel_magnitude
        + k_tilt * tilt_magnitude
        + k_angvel * angvel_magnitude
    )
    R_hold = torch.clamp(hold_penalty_raw, max=float(penalty_clamp_frac * k_near)) * gate

    # ============================================================
    # 4. ACTION JITTER PENALTY (negative for rapid action changes)
    # ============================================================
    # Deliberately UNGATED: it does not vary with distance, so it cannot create a barrier,
    # and control smoothness is wanted during the approach as much as at the goal.
    action_diff = torch.norm(current_action - prev_actions, dim=1)
    R_jitter = k_jitter * action_diff  # Negative penalty (more negative for stronger jitter)

    # ============================================================
    # TOTAL: R_pos + R_progress - R_hold - R_jitter
    # ============================================================
    # No per-step time term: the episode always runs to the step limit, so the return is
    # the integral of R_pos and "spend longer near the goal" is the objective, not a thing
    # to be taxed. If crashed, ONLY the crash penalty applies -- forfeiting the rest of the
    # episode's proximity reward is itself the larger part of the punishment.
    R_total = torch.where(
        crashes > 0.0,
        -k_crash * torch.ones_like(curr_dist),
        R_pos + R_progress - R_hold - R_jitter
    )

    # Update prev_dist for next step (return as output)
    new_prev_dist = curr_dist.clone()

    return R_total, crashes, new_prev_dist
