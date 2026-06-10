"""
Navigation with Obstacles Task for Aerial Gym Simulator.

A navigation task where a quadrotor must:
1. Navigate to a target waypoint in a box-shaped environment
2. Avoid obstacles using depth camera observations encoded by a custom DepthVAE
3. Use attitude control (thrust, roll, pitch, yaw_rate)

Features:
- 25-level curriculum: panels (levels 0-5), cumulative panels + objects (levels 6-25)
- Custom 32D DepthVAE encoding (matching VAE training distribution)
- Randomized environment bounds: L×W×H in [8,12]×[5,8]×[4,6]
- Observation (49D): state(17) + VAE latent(32). See process_obs_for_task() for layout.
"""
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger

from vae_depth.vae_image_encoder import DepthVAEImageEncoder

import os
import torch
import numpy as np
import cv2
import gymnasium as gym
from gym.spaces import Dict, Box
from isaacgym import gymapi, gymutil

# Set Qt plugin path for OpenCV GUI (Isaac Gym container lacks system xcb plugin)
os.environ.setdefault(
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "/usr/local/lib/python3.8/dist-packages/cv2/qt/plugins/platforms/",
)

logger = CustomLogger("navigation_with_obstacles_task")


class NavigationWithObstaclesTask(BaseTask):
    """
    Navigation task with obstacle curriculum and attitude control.

    Observation: see process_obs_for_task() for full layout.
        
    Action (4D) for lee_attitude_control (vehicle frame):
        [0]     thrust command (in [-1, 1]; controller maps to [0, 2*m*g])
        [1]     roll command (rad)
        [2]     pitch command (rad)
        [3]     yaw rate command (rad/s)
    """

    def __init__(
        self,
        task_config,
        seed=None,
        num_envs=None,
        headless=None,
        device=None,
        use_warp=None,
    ):
        # Override config params if provided
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
        self.device = self.task_config.device

        # Convert reward params to tensors
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building Navigation with Obstacles environment")
        logger.info(
            f"Sim: {task_config.sim_name}, Env: {task_config.env_name}, "
            f"Robot: {task_config.robot_name}, Controller: {task_config.controller_name}"
        )

        # Build simulation environment
        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        # Target position for each environment
        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # Target sampling ratios
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device
        ).expand(self.sim_env.num_envs, -1)

        # Previous distance to target (for progress tracking)
        self.prev_dist = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Previous transformed action for jerk penalty
        self.prev_action = torch.zeros(
            (self.sim_env.num_envs, 4), device=self.device, requires_grad=False
        )

        # VAE encoder for depth images (custom DepthVAE). Encodes all envs in one batch.
        if self.task_config.vae_config.use_vae:
            self.vae_model = DepthVAEImageEncoder(
                config=self.task_config.vae_config, device=self.device
            )
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = None
            self.image_latents = None

        # Get observation dictionary reference from environment
        self.obs_dict = self.sim_env.get_obs()

        # Environment step dt = physics_dt * num_physics_steps_per_env_step
        physics_dt = self.obs_dict.get("dt", 0.01)
        num_physics_steps = getattr(
            self.sim_env.cfg.env, "num_physics_steps_per_env_step_mean", 10
        )
        self.env_step_dt = physics_dt * num_physics_steps

        # Curriculum setup
        self.curriculum_level = self.task_config.curriculum.min_level
        self.obs_dict["num_obstacles_in_env"] = max(
            self.curriculum_level,
            self.sim_env.keep_in_env
        )
        self.curriculum_progress_fraction = 0.0

        # Curriculum tracking aggregates
        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0
        self.exceeds_aggregate = 0

        # Logged metrics for tensorboard (updated each curriculum check)
        self.logged_success_rate = 0.0
        self.logged_crash_rate = 0.0
        self.logged_exceed_rate = 0.0
        self.logged_timeout_rate = 0.0

        # EMA reward components for tensorboard (horizon-independent).
        # IsaacAlgoObserver overwrites direct_info each step and only logs the
        # last step's values, so we use an EMA to smooth across steps.
        self._reward_comp_ema = {
            "r_heading": 0.0, "r_progress": 0.0, "p_speed": 0.0, "p_jerk": 0.0,
        }
        self._ema_alpha = 0.02  # smooth over ~50 steps

        self._logged_ep_dist_to_target = 0.0

        # Termination/truncation tensors
        # IMPORTANT: self.terminations is a SEPARATE tensor, NOT an alias of
        # obs_dict["crashes"]. obs_dict["crashes"] is the simulator's collision
        # buffer (dtype=bool) used by post_reward_calculation_step. We must not
        # overwrite it with exceed/arrive flags.
        self.terminations = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.bool
        )
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Define observation and action spaces for rl_games
        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Action transformation function
        self.action_transformation_function = (
            self.task_config.action_transformation_function
        )

        self.num_envs = self.sim_env.num_envs

        # Task observation tensor
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }

        self.infos = {}

        # Episode counter for keep_same_env_for_num_episodes
        self.episode_counter = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.int32
        )
        self.keep_same_env_episodes = getattr(
            self.sim_env.cfg.env, "keep_same_env_for_num_episodes", 1
        )

        # Debug visualization: goal/start spheres + depth camera window
        self._headless = self.task_config.headless
        if not self._headless:
            self._gym = self.sim_env.IGE_env.gym
            self._viewer = self.sim_env.IGE_env.viewer.viewer
            self._env_handles = self.sim_env.IGE_env.env_handles
            self._goal_sphere = gymutil.WireframeSphereGeometry(
                0.5, 16, 16, None, color=(1, 0, 0)  # red
            )
            self._start_sphere = gymutil.WireframeSphereGeometry(
                0.3, 12, 12, None, color=(0, 1, 0)  # green
            )
            # Store initial drone positions for start marker
            self._start_positions = self.obs_dict["robot_position"].clone()
            # Create OpenCV window for depth camera
            cv2.namedWindow("Depth Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Depth Camera", 640, 360)

        logger.info(
            f"Task initialized with {self.num_envs} environments, "
            f"obs_dim={self.task_config.observation_space_dim}, "
            f"action_dim={self.task_config.action_space_dim}, "
            f"curriculum_level={self.curriculum_level}"
        )

    def close(self):
        """Clean up simulation resources."""
        if not self._headless:
            cv2.destroyAllWindows()
        del self.sim_env
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def reset(self):
        """Reset all environments."""
        self.reset_idx(
            torch.arange(self.sim_env.num_envs, device=self.device),
            force_obstacle_reset=True,
        )
        self.process_image_observation()
        self.process_obs_for_task()
        return self.get_return_tuple()

    def reset_idx(self, env_ids, force_obstacle_reset=False):
        """
        Reset specific environments.

        Args:
            env_ids: Tensor of environment indices to reset
            force_obstacle_reset: If True, reset obstacles regardless of episode counter
        """
        if len(env_ids) == 0:
            return

        # Increment episode counter
        self.episode_counter[env_ids] += 1

        # Determine which environments need full obstacle reset
        if force_obstacle_reset:
            envs_needing_obstacle_reset = env_ids
        else:
            needs_reset_mask = (
                self.episode_counter[env_ids] >= self.keep_same_env_episodes
            )
            envs_needing_obstacle_reset = env_ids[needs_reset_mask]

        # Reset episode counter for envs getting new obstacles
        if len(envs_needing_obstacle_reset) > 0:
            self.episode_counter[envs_needing_obstacle_reset] = 0

        # Full reset including obstacles
        if len(envs_needing_obstacle_reset) > 0:
            self.sim_env.reset_idx(envs_needing_obstacle_reset)

        # Robot-only reset for remaining environments
        envs_robot_only = (
            env_ids[~torch.isin(env_ids, envs_needing_obstacle_reset)]
            if len(envs_needing_obstacle_reset) > 0
            else env_ids
        )
        if len(envs_robot_only) > 0:
            self.sim_env.robot_manager.reset_idx(envs_robot_only)
            self.sim_env.IGE_env.write_to_sim()
            self.sim_env.sim_steps[envs_robot_only] = 0

        # Sample new target positions within bounds
        target_ratio = torch_rand_float_tensor(
            self.target_min_ratio, self.target_max_ratio
        )
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )

        # Reset previous distance for progress tracking
        self.prev_dist[env_ids] = torch.norm(
            self.target_position[env_ids] - self.obs_dict["robot_position"][env_ids],
            dim=1,
        )

        # Reset previous action to the neutral/hover command. The drone spawns
        # level and motionless (see robot_config init_state: zero attitude and
        # velocity), so the physically-consistent "previous action" is the
        # neutral transformed command [thrust=0, roll=0, pitch=0, yaw_rate=0]:
        self.prev_action[env_ids] = 0.0

        # Reset previous angular velocity to current value (avoids spike on first step)
        # Reset VAE latents so first observation doesn't contain stale encodings
        if self.image_latents is not None:
            self.image_latents[env_ids] = 0.0

        # Store start positions for debug visualization
        if not self._headless:
            self._start_positions[env_ids] = self.obs_dict["robot_position"][env_ids].clone()

    def render(self):
        """Render the environment."""
        return self.sim_env.render()

    def _draw_debug_visuals(self):
        """Draw goal (red) and start (green) spheres in the viewer."""
        self._gym.clear_lines(self._viewer)
        for i in range(self.num_envs):
            # Goal sphere (red)
            goal_pos = self.target_position[i].cpu().numpy()
            goal_pose = gymapi.Transform(p=gymapi.Vec3(*goal_pos))
            gymutil.draw_lines(
                self._goal_sphere, self._gym, self._viewer,
                self._env_handles[i], goal_pose,
            )
            # Start sphere (green)
            start_pos = self._start_positions[i].cpu().numpy()
            start_pose = gymapi.Transform(p=gymapi.Vec3(*start_pos))
            gymutil.draw_lines(
                self._start_sphere, self._gym, self._viewer,
                self._env_handles[i], start_pose,
            )

    def _show_depth_camera(self):
        """Display depth camera feed in an OpenCV window."""
        depth = self.obs_dict["depth_range_pixels"][0, 0].cpu().numpy()
        depth_vis = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
        cv2.imshow("Depth Camera", depth_color)
        cv2.waitKey(1)

    def step(self, actions):
        """
        Execute one step of the simulation.

        Args:
            actions: Tensor of actions (num_envs, 4) in range [-1, 1]

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos)
        """
        # Transform network outputs to controller commands
        transformed_action = self.action_transformation_function(actions)
        current_action = transformed_action.clone()  # snapshot before sim overwrites robot_actions/robot_prev_actions

        # Step the simulation
        self.sim_env.step(actions=transformed_action)

        # Compute rewards, terminations, and event masks
        self.rewards[:], self.terminations[:], arrive_mask, exceed_mask = (
            self.compute_rewards(self.obs_dict, current_action)
        )
        self.prev_action[:] = current_action

        self.prev_dist[:] = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )

        # Check for episode timeout (truncation), only for non-terminated envs
        timeout_mask = (self.sim_env.sim_steps > self.task_config.episode_len_steps) & (
            self.terminations == 0
        )

        # Apply timeout penalty (MAVRL-style: discourage passive/slow policies)
        self.rewards[timeout_mask] = self.task_config.reward_parameters["timeout_penalty"]

        # Write exceed/arrive into truncation buffer so post_reward_calculation_step
        # picks them up for reset. Collisions are already in obs_dict["crashes"].
        self.truncations[:] = (timeout_mask | arrive_mask | exceed_mask)

        # Success = arrived at target (from compute_rewards)
        successes = arrive_mask.float()

        # Exceeds = out-of-bounds terminations
        exceeds = exceed_mask.float()

        # Crashes = collision terminations (not arrivals, not exceeds)
        crashes = ((self.terminations > 0) & (~arrive_mask) & (~exceed_mask)).float()

        # Timeouts = episode ran out of steps (not arrive/exceed/collision)
        timeouts = timeout_mask.float()

        # rl_games value-bootstrap signal: bootstrap value ONLY on timeout (artificial
        # step-limit cutoff), NOT on arrive/exceed/collision (true terminals). Consumed by
        # a2c_common when value_bootstrap=True: shaped_rewards += gamma * V(s') * time_outs.
        # timeout_mask already excludes terminated envs, so those get time_outs=0.
        self.infos["time_outs"] = timeout_mask.float()
        # Scalar metrics for tensorboard logging (IsaacAlgoObserver logs scalars from infos)
        self.infos["curriculum_level"] = float(self.curriculum_level)
        self.infos["success_rate"] = self.logged_success_rate
        self.infos["crash_rate"] = self.logged_crash_rate
        self.infos["exceed_rate"] = self.logged_exceed_rate
        self.infos["timeout_rate"] = self.logged_timeout_rate

        # Reward components (EMA across steps, horizon-independent)
        self.infos["reward/r_heading"] = self._reward_comp_ema["r_heading"]
        self.infos["reward/r_progress"] = self._reward_comp_ema["r_progress"]
        self.infos["reward/p_speed"] = self._reward_comp_ema["p_speed"]
        self.infos["reward/p_jerk"] = self._reward_comp_ema["p_jerk"]

        # Displacement to target (used by multiple metrics below)
        robot_pos = self.obs_dict["robot_position"]
        disp = self.target_position - robot_pos

        # Episode-end distance to target
        ended_mask = (self.terminations > 0) | timeout_mask
        if ended_mask.any():
            self._logged_ep_dist_to_target = float(
                torch.norm(disp[ended_mask], dim=1).mean()
            )
        self.infos["metrics/dist_to_target_episode_end"] = self._logged_ep_dist_to_target

        # Flight metrics (mean across all envs)
        self.infos["metrics/dist_to_target"] = float(torch.norm(disp, dim=1).mean())
        self.infos["metrics/v_horizontal"] = float(
            torch.norm(self.obs_dict["robot_linvel"][:, :2], dim=1).mean()
        )
        self.infos["metrics/episode_length"] = float(self.sim_env.sim_steps.float().mean())

        # Update curriculum
        self.check_and_update_curriculum_level(
            successes, crashes, timeouts, exceeds
        )

        # Handle resets for terminated/truncated environments
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        # Process observations after resets so the policy sees fresh state
        # for any env that was just reset
        self.process_image_observation()
        self.process_obs_for_task()

        # Debug visualization (only in non-headless mode)
        if not self._headless:
            self._draw_debug_visuals()
            self._show_depth_camera()

        return self.get_return_tuple()

    def process_image_observation(self):
        """Encode depth image through custom DepthVAE to get latent representation."""
        if self.task_config.vae_config.use_vae and self.vae_model is not None:
            image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
            self.image_latents[:] = self.vae_model.encode(image_obs)

    def get_return_tuple(self):
        """Build and return the step/reset output tuple."""
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _direction_and_distance_to_target(self):
        """Body-frame unit vector to target and raw distance (meters), current step.

        Single source of truth for both the observation vector
        (process_obs_for_task) and the heading reward (_reward_progress), so the
        two never drift out of sync on frame or staleness.

        Returns:
            direction: (num_envs, 3) unit vector to target in body frame
            dist:      (num_envs, 1) distance to target in raw meters
        """
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_orientation"],
            self.target_position - self.obs_dict["robot_position"],
        )
        dist = torch.linalg.norm(vec_to_tgt + 1e-6, dim=1, keepdim=True)
        direction = vec_to_tgt / (dist + 1e-6)
        return direction, dist

    def process_obs_for_task(self):
        """
        Build observation vector.

        - [0:3]     unit vector from drone to target                    body
        - [3]       distance to target (normalized & clamped)           body 
        - [4:7]     linear velocity (vx, vy, vz)                        body
        - [7:10]    angular velocity (wx, wy, wz)                       body
        - [10:13]   gravity vector in body frame (gx, gy, gz)           body
        - [13:17]   previous action (thrust, roll, pitch, yaw_rate)     cmd (frameless)
        - [17:49]   VAE latent encoding (32D)                           N/A

        Total: 49D observation vector (can be reduced by removing components if needed)."""
        
        # Body-frame direction (unit) and raw distance to target. Shared with the
        # heading reward via _direction_and_distance_to_target so the obs vector
        # and the reward stay consistent in frame and step.
        direction_to_target, dist_to_tgt = self._direction_and_distance_to_target()

        # [0:3] Unit vector to target in body frame
        self.task_obs["observations"][:, 0:3] = direction_to_target

        # [3] Distance to target, normalized by env diagonal and clamped to [0, 1].
        # dist_to_tgt is (num_envs, 1); squeeze to (num_envs,) to match max_dist (num_envs,).
        max_dist = torch.norm(
            self.obs_dict["env_bounds_max"] - self.obs_dict["env_bounds_min"], dim=-1)
        self.task_obs["observations"][:, 3] = torch.clamp(
            dist_to_tgt.squeeze(1) / (max_dist + 1e-6), 0.0, 1.0)
        
        # [4:7] Linear velocity in body frame
        self.task_obs["observations"][:, 4:7] = self.obs_dict["robot_body_linvel"]  # TODO: NORMALIZE?

        # [7:10] Angular velocity in body frame
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_angvel"] # TODO: NORMALIZE?
        
        # [10:13] Gravity vector in body frame (normalized by g)
        gravity_world = self.obs_dict["gravity"]
        gravity_body = quat_rotate_inverse(
            self.obs_dict["robot_orientation"], 
            gravity_world
        )
        self.task_obs["observations"][:, 10:13] = gravity_body / torch.linalg.norm(gravity_body + 1e-6, dim=-1, keepdim=True)
        # [13:17] Previous action (thrust, roll, pitch, yaw_rate). This is a controller
        # command vector, not a spatial vector, so it has no coordinate frame; roll/pitch/
        # yaw_rate are body-axis attitude setpoints for lee_attitude_control.
        self.task_obs["observations"][:, 13:17] = self.prev_action
        
        # ADD VAE LATENTS (32D) TO OBSERVATION VECTOR
        # [17:49] VAE latent encoding (32D)
        if self.task_config.vae_config.use_vae and self.image_latents is not None:
            self.task_obs["observations"][:, 17:49] = self.image_latents

    def check_and_update_curriculum_level(self, successes, crashes, timeouts, exceeds):
        """
        Update curriculum level based on success rate.
        Same logic as NavigationTask.check_and_update_curriculum_level.
        """
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)
        self.exceeds_aggregate += torch.sum(exceeds)

        instances = (
            self.success_aggregate
            + self.crashes_aggregate
            + self.timeouts_aggregate
            + self.exceeds_aggregate
        )

        check_threshold = self.task_config.curriculum.check_after_num_rollouts * self.sim_env.num_envs
        if instances >= check_threshold:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances
            exceed_rate = self.exceeds_aggregate / instances

            # Update logged metrics for tensorboard
            self.logged_success_rate = float(success_rate)
            self.logged_crash_rate = float(crash_rate)
            self.logged_exceed_rate = float(exceed_rate)
            self.logged_timeout_rate = float(timeout_rate)

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # Clamp curriculum level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / max(
                self.task_config.curriculum.max_level
                - self.task_config.curriculum.min_level,
                1,
            )

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, "
                f"Progress: {self.curriculum_progress_fraction:.2f}"
            )
            logger.warning(
                f"Success Rate: {success_rate:.3f}, "
                f"Crash Rate: {crash_rate:.3f}, "
                f"Exceed Rate: {exceed_rate:.3f}, "
                f"Timeout Rate: {timeout_rate:.3f}"
            )
            logger.warning(
                f"Successes: {self.success_aggregate}, "
                f"Crashes: {self.crashes_aggregate}, "
                f"Exceeds: {self.exceeds_aggregate}, "
                f"Timeouts: {self.timeouts_aggregate}"
            )

            # Reset aggregates
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0
            self.exceeds_aggregate = 0

    def compute_rewards(self, obs_dict, current_action):
        """
        Compute reward from four mutually exclusive components (priority order):
        1. r_exceed:    out-of-bounds penalty (terminates)
        2. r_arrive:    reached target bonus (terminates as success)
        3. r_collision: obstacle collision penalty (terminates)
        4. r_prog:      progress reward for normal steps (STUB)

        Returns:
            Tuple of (rewards, terminations, arrive_mask, exceed_mask) tensors
        """
        robot_pos = obs_dict["robot_position"]
        crashes = obs_dict["crashes"]

        # Distance to target
        disp = self.target_position - robot_pos
        dist = torch.norm(disp, dim=1)

        # Condition masks (mutually exclusive, priority order)
        # Expand bounds by exceed_bounds_margin (1.0 = exact bounds, 1.5 = 50% beyond)
        margin = self.task_config.exceed_bounds_margin
        bounds_min = obs_dict["env_bounds_min"]
        bounds_max = obs_dict["env_bounds_max"]
        if margin != 1.0:
            center = (bounds_min + bounds_max) / 2
            half_extent = (bounds_max - bounds_min) / 2
            bounds_min = center - half_extent * margin
            bounds_max = center + half_extent * margin
        exceed_mask = (
            (robot_pos < bounds_min).any(dim=1)
            | (robot_pos > bounds_max).any(dim=1)
        )

        # Ground handling: classify near-floor positions as exceed (out-of-bounds-bottom)
        # rather than as an obstacle collision. The drone makes contact slightly ABOVE
        # the bottom bound (bottom_wall sits at env_bounds_min[z]), so a pure
        # robot_pos < bounds_min check would miss it and it would fall through to
        # collision_mask. Fold a margin band above the bottom bound into exceed so the
        # collision/"crash" path means a real obstacle hit. See task_config.ground_*.
        if getattr(self.task_config, "ground_as_exceed", False):
            ground_z = bounds_min[:, 2] + self.task_config.ground_collision_margin
            ground_mask = robot_pos[:, 2] < ground_z
            exceed_mask = exceed_mask | ground_mask

        arrive_mask = (~exceed_mask) & (
            dist < self.task_config.reward_parameters["d_min"]
        )
        collision_mask = (~exceed_mask) & (~arrive_mask) & (crashes > 0)
        progress_mask = (~exceed_mask) & (~arrive_mask) & (~collision_mask)

        # Compute each reward component
        reward = torch.zeros(self.num_envs, device=self.device)
        reward[exceed_mask] = self._reward_exceed()
        reward[arrive_mask] = self._reward_arrive()
        reward[collision_mask] = self._reward_collision()
        reward[progress_mask] = self._reward_progress(progress_mask, current_action)

        # All three event types terminate the episode
        terminations = exceed_mask | arrive_mask | collision_mask

        return reward, terminations, arrive_mask, exceed_mask

    def _reward_exceed(self):
        """Penalty for flying out of environment bounds."""
        return self.task_config.reward_parameters["exceed_penalty"]

    def _reward_arrive(self):
        """Bonus for reaching the target, scaled by curriculum level (MAVRL-style).
        Higher curriculum (more obstacles) = bigger reward."""
        params = self.task_config.reward_parameters
        bonus_min = params["arrive_bonus_min"]
        bonus_max = params["arrive_bonus_max"]
        t = self.curriculum_level / self.task_config.curriculum.max_level
        return bonus_min + t * (bonus_max - bonus_min)

    def _reward_collision(self):
        """Penalty for colliding with an obstacle."""
        return self.task_config.reward_parameters["collision_penalty"]

    def _reward_progress(self, mask, current_action):
        """
        Dense shaping reward for non-terminal steps. Balances goal-reaching,
        flight stability and safety.

        Components (the total reward is their sum):
        1. r_bearing:  lambda_b * dot(unit_vec_to_target, unit_vec_velocity)
                       - REWARDS flying toward the target (cosine of the angle
                         between the velocity and the direction to the target)
        2. r_progress: lambda_p * (prev_dist - current_dist)
                       - REWARDS closing distance to the target this step
                         (both distances in raw meters)
        3. p_speed:    lambda_v * v_hor * max(0, v_hor - v_max)
                       - PENALTIZE horizontal speed above v_max
        4. p_jerk:     lambda_jerk * ||a_curr - a_prev||
                       - PENALIZE large changes in the (transformed) action

        Args:
            mask: Boolean tensor indicating which envs get this reward
        Returns:
            Reward tensor for masked envs
        """
        params = self.task_config.reward_parameters
        target_pos = self.target_position

        # 1. Reward heading towards target: λ_b * dot(unit_vec_to_target, unit_vec_velocity)
        # Compute n/v from the current-step state (NOT task_obs["observations"],
        # which is only refreshed after compute_rewards in step() and would be one
        # step stale). The helper is the same source process_obs_for_task uses for
        # observations[:, 0:3]; robot_body_linvel is the source for [:, 4:7].
        n, _ = self._direction_and_distance_to_target()  # current-step, body frame
        v = self.obs_dict["robot_body_linvel"]           # current-step, body frame

        r_bearing = params["lambda_b"] *  torch.linalg.vecdot(n, v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-6), dim=1)

        # 2. Reward progress towards target: λ_p * (prev_dist - current_dist).
        # current_dist MUST be in the same units as self.prev_dist (raw meters,
        # see step()/reset_idx). Compute it from positions here rather than using
        # observations[:, 3], which is normalized by the env diagonal to [0, 1].
        current_dist = torch.norm(
            target_pos - self.obs_dict["robot_position"], dim=1
        )
        r_progress = params["lambda_p"] * (self.prev_dist - current_dist)
        
        # 3. Excess speed penalty: λ_v * v_hor * max(0, v_hor - v_max)
        v_hor = torch.linalg.norm(v[:, :2], dim=1)
        p_speed = params["lambda_v"] * v_hor * torch.clamp(
            v_hor - self.task_config.v_max, min=0.0
        )

        # 4. Penalize jerk (difference between current and previous transformed actions)
        a_curr = current_action
        a_prev = self.prev_action
        p_jerk = params["lambda_jerk"] * torch.linalg.norm(a_curr - a_prev, dim=1)

        # Apply mask to zero out rewards for envs that had terminal events
        r_bearing = r_bearing[mask]
        r_progress = r_progress[mask]
        p_speed = p_speed[mask]
        p_jerk = p_jerk[mask]
        

        # Update EMA for tensorboard reward component logging
        a = self._ema_alpha
        self._reward_comp_ema["r_heading"] += a * (float(r_bearing.mean()) - self._reward_comp_ema["r_heading"])
        self._reward_comp_ema["r_progress"] += a * (float(r_progress.mean()) - self._reward_comp_ema["r_progress"])
        self._reward_comp_ema["p_speed"] += a * (float(p_speed.mean()) - self._reward_comp_ema["p_speed"])
        self._reward_comp_ema["p_jerk"] += a * (float(p_jerk.mean()) - self._reward_comp_ema["p_jerk"])

        return r_bearing + r_progress + p_speed + p_jerk
