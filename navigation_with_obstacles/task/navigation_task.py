"""
Navigation with Obstacles Task for Aerial Gym Simulator.

A navigation task where a quadrotor must:
1. Navigate to a target waypoint in a box-shaped environment
2. Avoid obstacles using depth camera observations encoded by a custom DepthVAE
3. Use acceleration control (accel_x, accel_y, accel_z, yaw_rate)

Features:
- 25-level curriculum: panels (levels 0-5), cumulative panels + objects (levels 6-25)
- Custom 32D DepthVAE encoding (matching VAE training distribution)
- Randomized environment bounds: L×W×H in [8,12]×[5,8]×[4,6]
- Observation (44D): state(12) + VAE latent(32). See process_obs_for_task() for layout.
"""
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import (
    torch_rand_float_tensor,
    torch_interpolate_ratio,
)
from aerial_gym.utils.logging import CustomLogger

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
    Navigation task with obstacle curriculum and acceleration control.

    Observation (44D): see process_obs_for_task() for full layout.
        [0:12]  state: distances, bearing, yaw, velocities, track angles
        [12:44] VAE latent encoding (32D)

    Action (4D):
        [0:3]   acceleration command (body frame, m/s²)
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

        # VAE encoder for depth images (custom DepthVAE)
        if self.task_config.vae_config.use_vae:
            from vae_depth.vae_image_encoder import DepthVAEImageEncoder
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
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
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

        # EMA reward components for tensorboard (horizon-independent).
        # IsaacAlgoObserver overwrites direct_info each step and only logs the
        # last step's values, so we use an EMA to smooth across steps.
        self._reward_comp_ema = {
            "r_dist_hor": 0.0, "r_dist_vert": 0.0, "r_speed": 0.0,
            "r_bearing": 0.0, "r_path_deviation": 0.0, "r_jerk": 0.0,
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

        # Step the simulation
        self.sim_env.step(actions=transformed_action)

        # Compute rewards, terminations, and event masks
        self.rewards[:], self.terminations[:], arrive_mask, exceed_mask = (
            self.compute_rewards(self.obs_dict)
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

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = crashes
        self.infos["exceeds"] = exceeds
        # Scalar metrics for tensorboard logging (IsaacAlgoObserver logs scalars from infos)
        self.infos["curriculum_level"] = float(self.curriculum_level)
        self.infos["success_rate"] = self.logged_success_rate
        self.infos["crash_rate"] = self.logged_crash_rate
        self.infos["exceed_rate"] = self.logged_exceed_rate

        # Reward components (EMA across steps, horizon-independent)
        self.infos["reward/r_dist_hor"] = self._reward_comp_ema["r_dist_hor"]
        self.infos["reward/r_dist_vert"] = self._reward_comp_ema["r_dist_vert"]
        self.infos["reward/r_speed"] = self._reward_comp_ema["r_speed"]
        self.infos["reward/r_bearing"] = self._reward_comp_ema["r_bearing"]
        self.infos["reward/r_path_deviation"] = self._reward_comp_ema["r_path_deviation"]
        self.infos["reward/r_jerk"] = self._reward_comp_ema["r_jerk"]

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
            # Batch VAE encoding to avoid CUDA OOM on large num_envs
            batch_size = 512
            n = image_obs.shape[0]
            if n <= batch_size:
                self.image_latents[:] = self.vae_model.encode(image_obs)
            else:
                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    self.image_latents[start:end] = self.vae_model.encode(
                        image_obs[start:end]
                    )

    def get_return_tuple(self):
        """Build and return the step/reset output tuple."""
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        """
        Build observation vector (44D total).

        Observation structure:
        - [0]     log(horizontal_distance_to_target + 1)       world
        - [1]     log(vertical_distance + 1): vertical dist mag world
        - [2:4]   cos/sin azimuth (bearing) to target          world
        - [4]     elevation angle to target                    world
        - [5:7]   cos/sin yaw angle of drone                   world
        - [7]     v_xy: horizontal speed                       body
        - [8]     v_z: vertical speed                          body
        - [9:11]  cos/sin track bearing azimuth                body
        - [11]    track bearing elevation                      body
        - [12:44] vae_latent: DepthVAE encoding (32D)          N/A

        Total: 44D (12 state + 32 VAE latents)
        """
        robot_pos = self.obs_dict["robot_position"]
        target_pos = self.target_position

        # Displacement from drone to target (world frame)
        disp = target_pos - robot_pos

        # Horizontal distance (XY plane)
        d_hor = torch.norm(disp[:, :2], dim=1)

        # Vertical distance (magnitude, sign encoded in elevation angle)
        d_vert = torch.abs(disp[:, 2])

        # [0] log(horizontal distance)
        self.task_obs["observations"][:, 0] = torch.log(d_hor + 1)

        # [1] log(vertical distance)
        self.task_obs["observations"][:, 1] = torch.log(d_vert + 1)

        # [2:4] cos/sinazimuth (bearing) to target in world frame
        bearing_azimuth = torch.atan2(disp[:, 1], disp[:, 0])
        self.task_obs["observations"][:, 2] = torch.cos(bearing_azimuth)
        self.task_obs["observations"][:, 3] = torch.sin(bearing_azimuth)

        # [4] elevation angle to target in world frame
        elevation = torch.atan2(disp[:, 2], d_hor)
        self.task_obs["observations"][:, 4] = elevation
        
        # [5:7] yaw of drone in world frame (cos/sin)
        drone_yaw = self.obs_dict["robot_euler_angles"][:, 2]
        self.task_obs["observations"][:, 5] = torch.cos(drone_yaw)
        self.task_obs["observations"][:, 6] = torch.sin(drone_yaw)

        # [7] horizontal speed (body frame)
        linvel_body = self.obs_dict["robot_body_linvel"]
        v_xy = torch.norm(linvel_body[:, :2], dim=1)
        self.task_obs["observations"][:, 7] = v_xy

        # [8] vertical speed (body frame)
        v_z = linvel_body[:, 2]
        self.task_obs["observations"][:, 8] = v_z

        # [9:11] track azimuth (bearing in body frame) as cos/sin of atan2(vy/vx)
        # Masked to zero when near-stationary to avoid noisy direction signal
        track_azimuth = torch.atan2(linvel_body[:, 1], linvel_body[:, 0])
        speed_mask = (v_xy > 0.1).float()
        self.task_obs["observations"][:, 9] = torch.cos(track_azimuth) * speed_mask
        self.task_obs["observations"][:, 10] = torch.sin(track_azimuth) * speed_mask

        # [11] track elevation as atan2(vz / v_xy), masked when near-stationary
        track_elevation = torch.atan2(linvel_body[:, 2], v_xy + 1e-6)
        self.task_obs["observations"][:, 11] = track_elevation * speed_mask
        
        # [12:44] VAE latent encoding (32D)
        if self.task_config.vae_config.use_vae and self.image_latents is not None:
            self.task_obs["observations"][:, 12:44] = self.image_latents

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

    def compute_rewards(self, obs_dict):
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
        reward[progress_mask] = self._reward_progress(progress_mask)

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

    def _reward_progress(self, mask):
        """
        Dense shaping reward for non-terminal steps. Balances goal-reaching,
        flight stability and safety. All lambda coefficients are negative.

        Components:
        1. r_dist_hor:       lambda_d * log(d_hor + 1)
        2. r_dist_vert:      lambda_dz * log(|d_z| + 1)
        3. r_speed:          lambda_v * v_hor * max(0, v_hor - v_max)
        4. r_bearing:        lambda_bearing * |wrap(vel_angle + yaw - target_angle)|
        5. r_path_deviation: lambda_path_deviation * |vel_angle|
        6. r_jerk:           lambda_jerk * ||a_curr - a_prev||

        Args:
            mask: Boolean tensor indicating which envs get this reward
        Returns:
            Reward tensor for masked envs
        """
        params = self.task_config.reward_parameters
        robot_pos = self.obs_dict["robot_position"]
        target_pos = self.target_position
        
        linvel_body = self.obs_dict["robot_body_linvel"]
        
        disp = target_pos - robot_pos

        # 1. Horizontal distance penalty: λ_d * (log(d_hor+1))
        d_hor = torch.norm(disp[:, :2], dim=1)
        r_dist_hor = params["lambda_d"] * torch.log(d_hor + 1)
        
        # Vertical distance penalty: λ_dz * log(d_z+1)
        d_z = torch.abs(disp[:, 2])
        r_dist_vert = params["lambda_dz"] * torch.log(d_z + 1)

        # 2. Excess speed penalty: λ_v * v_hor * max(0, v_hor - v_max)
        vel_hor = torch.norm(linvel_body[:, :2], dim=1)
        r_speed = torch.clamp(vel_hor - self.task_config.v_max, min=0.0) * params["lambda_v"] * vel_hor

        # 3. Bearing misalignment penalty: λ_bearing * (1 - cos(angle between velocity and target direction))
        vel_angle = torch.atan2(linvel_body[:, 1], linvel_body[:, 0])
        target_angle = torch.atan2(disp[:, 1], disp[:, 0])
        drone_yaw = self.obs_dict["robot_euler_angles"][:, 2]
        angle_diff = vel_angle + drone_yaw - target_angle
        r_bearing = params["lambda_bearing"] * torch.abs(
            torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        )

        # 4. Penalize lateral velocity deviation
        r_path_deviation = params["lambda_path_deviation"] * torch.abs(vel_angle)

        # 5. Penalize jerk (difference between current actions and prev actions)
        r_jerk = params["lambda_jerk"] * torch.norm(
            self.obs_dict["robot_actions"] - self.obs_dict["robot_prev_actions"], dim=1
        )


        # Apply mask to zero out rewards for envs that had terminal events
        r_dist_hor = r_dist_hor[mask]
        r_dist_vert = r_dist_vert[mask]
        r_speed = r_speed[mask]
        r_bearing = r_bearing[mask]
        r_path_deviation = r_path_deviation[mask]
        r_jerk = r_jerk[mask]

        # Update EMA for tensorboard reward component logging
        a = self._ema_alpha
        self._reward_comp_ema["r_dist_hor"] += a * (float(r_dist_hor.mean()) - self._reward_comp_ema["r_dist_hor"])
        self._reward_comp_ema["r_dist_vert"] += a * (float(r_dist_vert.mean()) - self._reward_comp_ema["r_dist_vert"])
        self._reward_comp_ema["r_speed"] += a * (float(r_speed.mean()) - self._reward_comp_ema["r_speed"])
        self._reward_comp_ema["r_bearing"] += a * (float(r_bearing.mean()) - self._reward_comp_ema["r_bearing"])
        self._reward_comp_ema["r_path_deviation"] += a * (float(r_path_deviation.mean()) - self._reward_comp_ema["r_path_deviation"])
        self._reward_comp_ema["r_jerk"] += a * (float(r_jerk.mean()) - self._reward_comp_ema["r_jerk"])

        return r_dist_hor + r_dist_vert + r_speed + r_bearing + r_path_deviation + r_jerk
