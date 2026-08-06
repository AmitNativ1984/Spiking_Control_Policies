"""
Navigation with Obstacles Task for Aerial Gym Simulator.

A navigation task where a quadrotor must:
1. Navigate to a target waypoint in a box-shaped environment
2. Avoid obstacles using depth camera observations encoded by a custom DepthVAE
3. Use attitude control (thrust, roll, pitch, yaw_rate)

Features:
- 26-level curriculum (0-25). Obstacles come from a Poisson point process whose
  intensity ramps linearly with the level, from 0 at level 0 to
  task_config.obstacle_density_max at level 25. Density -- not count -- is the
  invariant, so clutter stays constant as the randomized env bounds vary in size.
- Custom 32D DepthVAE encoding (matching VAE training distribution)
- Randomized environment bounds: L×W×H in [12.0,14.4]×[6.0,9.6]×[4.0,6.0]
  (derived from config/env_config/env_forest_with_obstacles.py)
- Observation (49D): state(17) + VAE latent(32). See process_obs_for_task() for layout.
"""
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.math import *
from aerial_gym.utils.logging import CustomLogger

from aerial_gym.registry.robot_registry import robot_registry

from env_manager.poisson_asset_manager import PoissonAssetManager
from vae_depth.vae_image_encoder import DepthVAEImageEncoder

import os
import math
import torch
import numpy as np
import cv2
# NOTE: old `gym`, not `gymnasium`, on purpose -- rl_games and training/runner.py are
# both built against old gym, and the task is wrapped by an old-gym Wrapper. Mixing
# space types across that boundary is the bug this import avoids.
from gym.spaces import Dict, Box
from isaacgym import gymapi, gymutil

# Set Qt plugin path for OpenCV GUI (Isaac Gym container lacks system xcb plugin)
os.environ.setdefault(
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "/usr/local/lib/python3.8/dist-packages/cv2/qt/plugins/platforms/",
)

logger = CustomLogger("attitude_navigation_task")
logger.setLevel("INFO")  # DEBUG, INFO, WARNING, ERROR, CRITICAL


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

        # Fail before building the sim: with use_warp=False this robot's depth camera
        # returns the same stale frame forever, silently.
        #
        # The Isaac-Gym-native camera path only re-renders when
        # EnvManager.post_reward_calculation_step() calls IGE_env.step_graphics(), which
        # it gates on robot_manager.has_IGE_sensors. That flag is set in exactly one
        # place (robots/robot_manager.py:270), an `elif use_warp == False and
        # camera_sensor is not None` chained off `if sensor_config.enable_imu:`. The F450
        # config sets enable_imu = True, so the `if` always wins and the `elif` is
        # unreachable -- the flag stays False and no render ever happens. Upstream bug;
        # the warp path is effectively the only working camera path for this robot.
        if not task_config.use_warp and task_config.vae_config.use_vae:
            raise ValueError(
                "use_warp=False is not supported for this task while vae_config.use_vae "
                "is True: the Isaac Gym camera would never re-render and the VAE would "
                "encode one frozen depth frame for the whole run (see the comment above "
                "this check). Run with use_warp=True, or set use_vae=False to train a "
                "state-only policy that needs no camera."
            )

        super().__init__(task_config)
        self.device = self.task_config.device

        # Convert reward params to tensors
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )

        logger.info("Building Attitude Navigation environment")
        logger.info(f"Sim: {task_config.sim_name}")
        logger.info(f"Env: {task_config.env_name}")
        logger.info(f"Robot: {task_config.robot_name}")
        logger.info(f"Controller: {task_config.controller_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Headless: {self.task_config.headless}")
        logger.info(f"Warp: {self.task_config.use_warp}")
        logger.info(f"Num Envs: {self.task_config.num_envs}")

        # Build simulation environment. Keep the SimBuilder: build_env() returns an
        # EnvManager, but delete_env() lives on the builder, so close() needs this handle.
        self.sim_builder = SimBuilder()
        self.sim_env = self.sim_builder.build_env(
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

        # Target sampling: the goal lands on one of the four vertical env walls.
        # These are (3,) tensors, broadcast against the per-env bounds at sample time.
        self.target_wall_inset = torch.tensor(
            self.task_config.target_wall_inset, device=self.device
        )
        self.target_free_ratio_min = torch.tensor(
            self.task_config.target_free_ratio_min, device=self.device
        )
        self.target_free_ratio_max = torch.tensor(
            self.task_config.target_free_ratio_max, device=self.device
        )
        # Which wall each env's current target is on (0:+x, 1:-x, 2:+y, 3:-y). Kept so
        # success can be broken down per direction -- the diagnostic that tells us whether
        # the policy generalized across bearings or only solved the short ones.
        self.target_face = torch.zeros(
            self.sim_env.num_envs, device=self.device, dtype=torch.long
        )

        # Previous distance to target (for progress tracking)
        self.prev_dist = torch.zeros(self.sim_env.num_envs, device=self.device)

        # Previous transformed action, for the jerk penalty and observation dims [13:17].
        self.prev_action = torch.zeros(
            (self.sim_env.num_envs, 4), device=self.device, requires_grad=False
        )

        # Per-channel command range of the TRANSFORMED action, used to make the jerk
        # penalty dimensionless. Without it, ||a_curr - a_prev|| sums a dimensionless
        # thrust in [-1, 1] with radians and rad/s, so the channel with the widest
        # numeric range (yaw_rate, +/-pi/3) dominates the norm and lambda_jerk means
        # something different for each channel. Dividing by these makes every channel
        # contribute its change as a fraction of its own range.
        # Mirrors action_transformation_function -- keep the two in step.
        self._action_scale = torch.tensor(
            [
                1.0,  # thrust: passed through untransformed, already in [-1, 1]
                self.task_config.max_inclination_angle_rad,  # roll
                self.task_config.max_inclination_angle_rad,  # pitch
                self.task_config.max_yaw_rate,               # yaw_rate
            ],
            device=self.device,
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

        self._install_imu_substep_accumulator()
        self._setup_domain_randomization()

        # Curriculum setup
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]

        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self._publish_obstacle_intensity()

        # Swap in the Poisson obstacle sampler. SimBuilder.build_env returns the
        # EnvManager and asset_manager is a plain attribute on it, so this needs no
        # changes to aerial_gym source. The spawn box comes from the robot config, which
        # is the single source of truth -- the keep-out ellipsoid is derived from it so
        # obstacles can never be placed where the drone is about to appear.
        robot_cfg = robot_registry.get_robot_config(self.task_config.robot_name)
        asset_manager = PoissonAssetManager(
            self.sim_env.global_tensor_dict, self.sim_env.keep_in_env
        )
        asset_manager.spawn_ratio_lo = torch.tensor(
            robot_cfg.init_config.min_init_state[0:3], device=self.device
        )
        asset_manager.spawn_ratio_hi = torch.tensor(
            robot_cfg.init_config.max_init_state[0:3], device=self.device
        )
        asset_manager.clearance = self.task_config.obstacle_spawn_clearance
        self.sim_env.asset_manager = asset_manager

        # One throwaway physics step BEFORE the reset below, and it is load-bearing.
        #
        # This exists to absorb OUR OWN _setup_domain_randomization() call above, not to work
        # around anything in Isaac Gym or aerial_gym. That call writes rigid-body properties
        # via gym.set_actor_rigid_body_properties, which under the GPU pipeline discards the
        # root states staged for the next simulate() -- the same mechanism documented on
        # _randomize_mass_properties. At build time it lands between build_env()'s initial
        # placement and the reset below, and restaging does NOT recover it: a set_actor_root_
        # _state_tensor call after the property write is dropped too, until a simulate() has
        # run. So the FIRST episode's spawn was lost and every env started at the world
        # origin. Later episodes were unaffected, which is what made it so easy to miss.
        #
        # Measured on the pre-fix code, one env, spawn box centred at ~[4.8, 0.3, 0.8]:
        #   randomize_mass_properties=True,  no warm-up  -> [0.00, 0.00, 0.00]  (4.90 m off)
        #   randomize_mass_properties=True,  restage only-> [0.00, 0.00, 0.00]  (no help)
        #   randomize_mass_properties=False, no warm-up  -> at the spawn, 0.005 m
        #   randomize_mass_properties=True,  warm-up     -> at the spawn, 0.001 m
        #
        # So this step becomes unnecessary the moment the rigid-body randomization is
        # dropped; keep the two together.
        # Zero is the neutral attitude command for lee_attitude_control, so the step is inert.
        self.sim_env.step(
            actions=torch.zeros(
                (self.sim_env.num_envs, self.task_config.action_space_dim),
                device=self.device,
            )
        )

        # build_env already placed obstacles with the upstream manager; re-reset so the
        # very first episode sees a Poisson layout rather than the old ratio-box one.
        self.sim_env.reset()

        # Curriculum tracking aggregates
        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0
        self.exceeds_aggregate = 0

        # Per-wall success accounting (0:+x, 1:-x, 2:+y, 3:-y). The pooled success rate
        # mixes directions, and the +/-y walls are ~30% closer than +/-x, so the easy
        # bearings can drag the aggregate over the advancement threshold while the far
        # ones still fail. Curriculum increases gate on the WORST wall instead.
        self._face_attempts = torch.zeros(4, device=self.device)
        self._face_successes = torch.zeros(4, device=self.device)

        # Logged metrics for tensorboard (updated each curriculum check)
        self.logged_success_rate = 0.0
        self.logged_crash_rate = 0.0
        self.logged_exceed_rate = 0.0
        self.logged_timeout_rate = 0.0
        self.logged_face_success = [0.0] * 4

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

        # Task observation tensor
        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
        }

        self.infos = {}

        # TODO: This should be implemented on inference side, not in the task. The task should only provide the raw depth image.
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
            f"Task initialized with {self.sim_env.num_envs} environments, "
            f"obs_dim={self.task_config.observation_space_dim}, "
            f"action_dim={self.task_config.action_space_dim}, "
            f"curriculum_level={self.curriculum_level}"
        )

    def close(self):
        """Clean up simulation resources.

        Upstream's task template calls self.sim_env.delete_env(), which raises
        AttributeError: build_env() returns an EnvManager and delete_env() is a
        SimBuilder method. Every shipped aerial_gym task has this bug, so cleanup never
        runs and VRAM is not reclaimed between task constructions — invisible in a single
        training run, but it bites in sweeps and test suites.
        """
        if not self._headless:
            cv2.destroyAllWindows()

        self.sim_builder.delete_env()

    def reset(self):
        """Reset all environments."""
        self.reset_idx(torch.arange(self.sim_env.num_envs, device=self.device))
        return self.get_return_tuple()

    def _publish_obstacle_intensity(self):
        """Recompute the curriculum progress fraction and write the Poisson intensity
        (obstacles/m^3) that PoissonAssetManager reads.

        Density ramps linearly with the ABSOLUTE curriculum level: 0 at level 0, and
        obstacle_density_max at curriculum.density_at_level. It deliberately does not
        normalize by (max_level - min_level) -- pinning the curriculum sets min == max,
        which would make every pinned level produce an empty world. See the comment on
        density_at_level in the task config.

        Density -- not count -- is the invariant, so clutter stays constant as the
        randomized env bounds vary in size.
        """
        self.curriculum_progress_fraction = min(
            self.curriculum_level / max(self.task_config.curriculum.density_at_level, 1),
            1.0,
        )
        self.obs_dict["obstacle_intensity"] = (
            self.task_config.obstacle_density_max * self.curriculum_progress_fraction
        )

    def _sample_target_on_vertical_walls(self, env_ids):
        """Sample targets on the four vertical faces of each env box.

        Face k: 0=+x, 1=-x, 2=+y, 3=-y, each with probability 1/4 so the world-frame
        traversal direction is balanced across the batch. Together with the centre spawn
        this removes the old "always fly +x" structure.

        Faces are inset by target_wall_inset so the full arrival ball of radius d_min lies
        inside the bounds: exceed_mask has priority over arrive_mask in compute_rewards(),
        so a target flush with the wall would be reachable only from a half-ball and any
        overshoot would score as `exceed` instead of `arrive`.

        Obstacle positions are already final when this runs (asset_manager.reset_idx
        precedes task.reset_idx), so candidates are scored by clearance and the freest is
        kept -- a target buried inside a tree is an episode that could only time out.

        Returns:
            position: (len(env_ids), 3) target positions in world coordinates
            face:     (len(env_ids),) index of the wall each target landed on
        """
        num_resets = len(env_ids)
        num_candidates = self.task_config.target_clearance_candidates

        lower = self.obs_dict["env_bounds_min"][env_ids] + self.target_wall_inset
        upper = self.obs_dict["env_bounds_max"][env_ids] - self.target_wall_inset

        # Free (un-pinned) coordinates, uniform over the configured ratio window.
        ratio = self.target_free_ratio_min + (
            self.target_free_ratio_max - self.target_free_ratio_min
        ) * torch.rand(num_resets, num_candidates, 3, device=self.device)
        candidates = lower.unsqueeze(1) + (upper - lower).unsqueeze(1) * ratio

        # Pin one coordinate to a wall. The face is drawn once per env, not per candidate,
        # so the clearance search cannot bias the 1/4-per-face distribution.
        face = torch.randint(0, 4, (num_resets,), device=self.device)
        axis = (face // 2).view(-1, 1, 1).expand(-1, num_candidates, 1)  # 0 -> x, 1 -> y
        on_max = (face % 2 == 0).view(-1, 1, 1).expand(-1, num_candidates, 1)
        wall = torch.where(
            on_max,
            upper.unsqueeze(1).expand(-1, num_candidates, -1).gather(2, axis),
            lower.unsqueeze(1).expand(-1, num_candidates, -1).gather(2, axis),
        )
        candidates.scatter_(2, axis, wall)

        # Keep the candidate furthest from any obstacle. Culled obstacles sit at -1000, so
        # they never win the min.
        obstacles = self.obs_dict["env_asset_state_tensor"][env_ids][:, :, 0:3]
        clearance = torch.cdist(candidates, obstacles).min(dim=2).values
        best = clearance.argmax(dim=1)
        position = candidates[torch.arange(num_resets, device=self.device), best]
        return position, face

    # ---- domain randomization ------------------------------------------------
    # Two independent pieces, deliberately configured in different places:
    #   inertia         -> robot_config.domain_randomization  (what the vehicle IS)
    #   estimator error -> task_config.state_estimation_noise (what the agent is TOLD)
    #
    # They resample on DIFFERENT schedules, and that asymmetry is load-bearing:
    #   inertia         -> ONCE, at build. See _randomize_mass_properties.
    #   estimator error -> per episode, from reset_idx().
    #
    # The guiding rule for anything added here: randomize the MISMATCH, not the parameter.
    # A quantity that is randomized and then handed to the controller is invisible to the
    # policy -- that is exactly why the mass draw was removed.

    def _setup_domain_randomization(self):
        """Cache the nominal rigid-body properties and allocate the estimator bias buffers."""
        dr = self.sim_env.robot_manager.cfg.domain_randomization
        self._mass_dr = dr if dr.randomize_mass_properties else None
        if self._mass_dr is not None:
            p = self.sim_env.robot_manager.gym.get_actor_rigid_body_properties(
                self.sim_env.IGE_env.env_handles[0],
                self.sim_env.robot_manager.robot_handles[0],
            )[0]
            # Every resample scales THIS, never the live value: scaling in place would
            # compound across episodes and walk the airframe out of its configured band.
            self._nominal_inertia = (p.inertia.x.x, p.inertia.y.y, p.inertia.z.z, p.inertia.x.z)

        # One env step in seconds, from the live sim rather than the config: the position
        # random walk scales as sqrt(dt), so a hardcoded value would silently misscale the
        # drift the moment sim.dt or the substep count changed.
        self._env_step_dt = (
            self.obs_dict["dt"] * self.sim_env.cfg.env.num_physics_steps_per_env_step_mean
        )

        n = self.sim_env.num_envs
        self._est_pos_bias = torch.zeros((n, 3), device=self.device, requires_grad=False)
        self._est_yaw_bias = torch.zeros(n, device=self.device, requires_grad=False)

        # Draw the airframe ONCE, here, before any stepping -- never again. See
        # _randomize_mass_properties for why calling it later destroys the episode reset.
        all_envs = torch.arange(n, device=self.device)
        self._randomize_mass_properties(all_envs)
        self._randomize_domain(all_envs)

    def _randomize_domain(self, env_ids):
        """Per-episode randomization, called from reset_idx().

        Rigid-body properties are deliberately NOT resampled here -- see
        _randomize_mass_properties.
        """
        self._resample_estimator_bias(env_ids)

    def _randomize_mass_properties(self, env_ids):
        """Resample base_link inertia. Prop links are 13 g and left alone.

        CALL THIS ONCE, AT BUILD, BEFORE ANY STEPPING. Never from reset_idx().

        gym.set_actor_rigid_body_properties is a CPU-side actor API. Under the GPU pipeline,
        calling it mid-episode DISCARDS the root states that were staged for the next
        simulate() -- and the episode reset stages exactly those. env_manager.reset_idx()
        writes the sampled spawn and calls IGE_env.write_to_sim(), then task.reset_idx()
        runs; a rigid-body-properties call in there throws the spawn away and every actor
        reverts to its creation pose at the world origin.

        Measured, one env, spawn pinned to a fixed ratio:
            randomize_mass_properties = True  -> spawn written [4.05, -0.69, 0.63],
                                                 one step later [0.00, 0.00, 0.00]
            randomize_mass_properties = False -> spawn holds to 0.003 m
        The drone silently started every episode at the origin, and the viewer's green
        start sphere -- which correctly marks the SAMPLED spawn -- never sat on it.
        Re-pushing set_actor_root_state_tensor afterwards does NOT recover it.

        Drawing once at build still gives each env its own airframe for the whole run, which
        is the mismatch that matters (robot_manager copies ONE inertia to every env, so the
        controller never learns the per-env draw). What is lost is per-episode resampling.

        INERTIA ONLY. Mass and CoM were dropped deliberately -- see
        robot_config.domain_randomization for the measurements behind that. In short: the
        controller reads obs_dict["robot_mass"] for its (a+1)*m*g map, so telling it about a
        mass draw cancels the draw; and CoM writes never reach PhysX after prepare_sim.

        Because mass is no longer touched, obs_dict["robot_mass"] keeps its build-time value
        and stays correct -- do not reintroduce a write here without re-reading that note.
        """
        if self._mass_dr is None or len(env_ids) == 0:
            return
        cfg = self._mass_dr
        gym = self.sim_env.robot_manager.gym
        envs = self.sim_env.IGE_env.env_handles
        actors = self.sim_env.robot_manager.robot_handles

        for i in env_ids.tolist():
            props = gym.get_actor_rigid_body_properties(envs[i], actors[i])
            base = props[0]
            inertia_scale = np.random.uniform(*cfg.inertia_scale_range)
            ixx, iyy, izz, ixz = (v * inertia_scale for v in self._nominal_inertia)
            base.inertia.x.x, base.inertia.y.y, base.inertia.z.z = ixx, iyy, izz
            base.inertia.x.z = base.inertia.z.x = ixz
            # recomputeInertia=False, else PhysX overwrites the tensor we just set.
            gym.set_actor_rigid_body_properties(envs[i], actors[i], props, False)

    def _resample_estimator_bias(self, env_ids):
        """Fresh turn-on bias per episode, so the position walk restarts each flight."""
        cfg = self.task_config.state_estimation_noise
        if not cfg.enable or len(env_ids) == 0:
            return
        n = len(env_ids)
        self._est_pos_bias[env_ids] = torch.randn((n, 3), device=self.device) * cfg.pos_bias_init_std
        self._est_yaw_bias[env_ids] = torch.randn(n, device=self.device) * cfg.yaw_bias_std

    def _estimated_state(self):
        """Odometry as the estimator sees it: (direction, distance, vehicle_linvel).

        Observation only. The reward path calls _direction_and_distance_to_target() on true
        state, so the agent is paid for reaching the real target while seeing a drifting
        estimate of where it is.

        Advances the position random walk, so call exactly once per env step.
        """
        cfg = self.task_config.state_estimation_noise
        vel = self.obs_dict["robot_vehicle_linvel"]
        if not cfg.enable:
            return (*self._direction_and_distance_to_target(), vel)

        # Position error DRIFTS rather than being white: an estimator's error is correlated
        # in time, which is exactly what a policy cannot average away over an episode.
        self._est_pos_bias += torch.randn_like(self._est_pos_bias) * (
            cfg.pos_random_walk_std * math.sqrt(self._env_step_dt)
        )
        vec = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            self.target_position - (self.obs_dict["robot_position"] + self._est_pos_bias),
        )
        vel = vel + torch.randn_like(vel) * cfg.vel_noise_std

        # A yaw-estimate error rotates the whole horizontal picture. The vehicle frame is
        # yaw-only, so it is one planar rotation of both bearing and velocity by -yaw_bias.
        cy, sy = torch.cos(self._est_yaw_bias), torch.sin(self._est_yaw_bias)
        vec = torch.stack((cy * vec[:, 0] + sy * vec[:, 1], cy * vec[:, 1] - sy * vec[:, 0], vec[:, 2]), 1)
        vel = torch.stack((cy * vel[:, 0] + sy * vel[:, 1], cy * vel[:, 1] - sy * vel[:, 0], vel[:, 2]), 1)

        dist = torch.linalg.norm(vec + 1e-6, dim=1, keepdim=True)
        return vec / (dist + 1e-6), dist, vel

    def _install_imu_substep_accumulator(self):
        """Accumulate the IMU across physics substeps so the policy sees a mean, not a sample.

        imu_sensor.update() runs once per PHYSICS substep but the policy reads once per ENV
        step, so 2 of every 3 samples would be discarded and the policy would see a single
        noisy instant. The real pipeline low-passes the gyro before any 30 Hz consumer sees
        it, so sampling one instant would give the policy MORE noise than the hardware has.
        Averaging N substeps cuts the noise by sqrt(N) and approximates that filtering.

        Upstream exposes no per-substep hook, so we wrap the bound method. Remove this once
        it does. imu_sensor.imu_meas is (num_envs, 6): [0:3] accel, [3:6] gyro, body frame.
        """
        imu = getattr(self.sim_env.robot_manager, "imu_sensor", None)
        if imu is None:
            # enable_imu = False in the robot config: observation falls back to
            # ground-truth angular velocity everywhere (count stays 0).
            self._imu_gyro_accum = None
            return

        self._imu_gyro_accum = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        # Per-env, not a scalar: reset_idx() has to zero individual envs (see below).
        self._imu_substep_count = torch.zeros(
            (self.sim_env.num_envs,), device=self.device, requires_grad=False
        )

        _wrapped_update = imu.update

        def _accumulating_update():
            _wrapped_update()
            self._imu_gyro_accum += imu.imu_meas[:, 3:6]
            self._imu_substep_count += 1.0

        imu.update = _accumulating_update

    def _consume_imu_gyro(self):
        """Mean gyro over this env step's substeps, falling back to ground truth.

        Two cases hit the fallback, both real rather than defensive:
          1. num_physics_step_per_env_step is max(floor(gauss(mean, std)), 0) and can
             legitimately be ZERO, leaving no samples.
          2. Envs that reset this step. post_reward_calculation_step() re-renders only
             camera/warp sensors, never the IMU, so the accumulator still holds the
             PREVIOUS episode; reset_idx() zeroes those envs so they land here.

        Safe either way: init_state draws angular velocity in +/-0.1 rad/s.
        """
        ground_truth = self.obs_dict["robot_body_angvel"]
        if self._imu_gyro_accum is None:
            return ground_truth
        n = self._imu_substep_count.unsqueeze(1)
        gyro = torch.where(n > 0, self._imu_gyro_accum / n.clamp_min(1.0), ground_truth)
        self._imu_gyro_accum.zero_()
        self._imu_substep_count.zero_()
        return gyro

    def reset_idx(self, env_ids):
        """
        Reset specific environments.

        Args:
            env_ids: Tensor of environment indices to reset
        """
        # Drop IMU samples belonging to the episode that just ended -- see
        # _consume_imu_gyro() for why they would otherwise leak into the new episode's
        # first observation.
        if self._imu_gyro_accum is not None:
            self._imu_gyro_accum[env_ids] = 0.0
            self._imu_substep_count[env_ids] = 0.0

        # Fresh airframe and fresh estimator bias, so an agent cannot memorise one vehicle
        # and the position walk restarts each flight.
        self._randomize_domain(env_ids)

        # Sample new target positions on the vertical env walls
        self.target_position[env_ids], self.target_face[env_ids] = (
            self._sample_target_on_vertical_walls(env_ids)
        )

        # Reset previous distance for progress tracking. Redundant on the step() path --
        # step() re-snapshots prev_dist for all envs at its top, after this reset has
        # landed. Kept so prev_dist is never stale for a caller that resets without
        # immediately stepping.
        self.prev_dist[env_ids] = self._get_dist_to_target(env_ids)

        # Reset to the neutral transformed command [thrust=0, roll=0, pitch=0, yaw_rate=0].
        # Zeros are neutral only because LeeAttitudeController maps thrust via
        # (cmd + 1) * m * g, so 0 == hover. If that convention changes (e.g. to
        # [0, max_thrust]), put the explicit hover command here instead.
        self.prev_action[env_ids] = 0.0

        # Reset VAE latents so the first observation doesn't contain stale encodings.
        # NOTE: dead on the step() path -- process_image_observation() overwrites
        # image_latents[:] for ALL envs immediately after, from a render taken at the
        # post-reset pose, which is what we want there. Load-bearing only on the reset()
        # path, which does no render, so the first obs of a run carries zero latents.
        # Do not "fix" it by masking that overwrite to non-reset envs.
        if self.image_latents is not None:
            self.image_latents[env_ids] = 0.0

        # Store start positions for debug visualization
        if not self._headless:
            self._start_positions[env_ids] = self.obs_dict["robot_position"][env_ids].clone()

    def render(self):
        """Render the environment.

        Required by BaseTask (abstract) but unused in the training path. Note that
        EnvManager.render defaults to render_components="sensors", so calling this
        triggers a full-batch sensor capture, not a viewer draw.
        """
        return self.sim_env.render()

    def _draw_debug_visuals(self):
        """Draw goal (red) and start (green) spheres in the viewer."""
        self._gym.clear_lines(self._viewer)
        for i in range(self.sim_env.num_envs):
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
        self.prev_dist[:] = self._get_dist_to_target()

        # Step the simulation and update the observation dictionary
        self.sim_env.step(actions=transformed_action)

        # Compute rewards, terminations, and event masks
        self.rewards[:], self.terminations[:], arrive_mask, exceed_mask = (
            self.compute_rewards(self.obs_dict, current_action)
        )
        # After compute_rewards: the jerk penalty needs a_prev to still be step t-1's action.
        self.prev_action[:] = current_action

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
        exceeds = exceed_mask.float()
        timeouts = timeout_mask.float()
        crashes = ((self.terminations > 0) & (~arrive_mask) & (~exceed_mask)).float()

        # Per-wall success accounting. Must happen BEFORE post_reward_calculation_step(),
        # which triggers reset_idx and overwrites target_face with the next episode's wall.
        ended = (self.terminations > 0) | timeout_mask
        face_onehot = torch.nn.functional.one_hot(self.target_face, 4).float()
        self._face_attempts += (ended.float().unsqueeze(1) * face_onehot).sum(0)
        self._face_successes += (successes.unsqueeze(1) * face_onehot).sum(0)

        self._update_infos(successes, timeout_mask, ended)

        # Update curriculum
        self.check_and_update_curriculum_level(
            successes, crashes, timeouts, exceeds
        )

        # Capture the TERMINAL observation, before the reset below overwrites the state.
        # Off by default: the agent should receive the first observation of the NEW
        # episode, so returning it after the reset is the correct PPO bootstrapping order.
        #
        # Placed here rather than at upstream's line (right after compute_rewards), because
        # get_return_tuple() reads self.truncations and self.infos -- upstream's earlier
        # placement captures the PREVIOUS step's values for both. The image half of the
        # observation is one step stale on this path either way, since the sensors are not
        # re-rendered until post_reward_calculation_step().
        return_tuple = None
        if self.task_config.return_state_before_reset:
            return_tuple = self.get_return_tuple()

        # Handle resets for terminated/truncated environments
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        # Encode depth AFTER post_reward_calculation_step(), which renders the sensors
        # from the post-reset robot pose (see EnvManager.post_reward_calculation_step).
        # This is the image half of the observation only -- the state half is built in
        # get_return_tuple() so that reset() gets it too. The two are split because the
        # image encode is expensive and has a precondition (a fresh sensor render) that
        # get_return_tuple cannot guarantee.
        self.process_image_observation()

        # Debug visualization (only in non-headless mode)
        if not self._headless:
            self._draw_debug_visuals()
            self._show_depth_camera()

        if return_tuple is None:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def _update_infos(self, successes, timeout_mask, ended):
        """Write the per-step entries of self.infos, returned to the RL algorithm.

        Two entries are FUNCTIONAL, not logging -- do not drop them when pruning metrics:
          - "arrivals":  per-env arrival flag. arrive and exceed both land in
                         `truncations`, so this is the only way to separate them
                         downstream. Consumers need it to count successes from the step
                         return tuple WITHOUT reading the task's *_aggregate fields, which
                         the curriculum state machine zeroes mid-rollout (corrupting any
                         delta taken across the reset).
          - "time_outs": rl_games value-bootstrap signal. Consumed by a2c_common when
                         value_bootstrap=True: shaped_rewards += gamma * V(s') * time_outs.
                         Bootstrap ONLY on timeout (an artificial step-limit cutoff), never
                         on arrive/exceed/collision, which are true terminals. timeout_mask
                         already excludes terminated envs, so those get 0.

        Everything else is scalar tensorboard logging (IsaacAlgoObserver logs scalars
        found in infos).

        Args:
            successes:    (num_envs,) float, 1.0 where the env arrived this step
            timeout_mask: (num_envs,) bool, envs that hit the step limit
            ended:        (num_envs,) bool, envs whose episode ended for any reason.
                          Passed in rather than recomputed so it cannot drift from the
                          copy step() uses for per-wall accounting.
        """
        self.infos["arrivals"] = successes
        self.infos["time_outs"] = timeout_mask.float()

        self.infos["curriculum_level"] = float(self.curriculum_level)
        self.infos["success_rate"] = self.logged_success_rate
        self.infos["crash_rate"] = self.logged_crash_rate
        self.infos["exceed_rate"] = self.logged_exceed_rate
        self.infos["timeout_rate"] = self.logged_timeout_rate
        self.infos["episode_length"] = float(self.sim_env.sim_steps.float().mean())


        # Per-wall success: a persistent spread here means the bearing is still leaking
        # (via the obstacle field or the distance asymmetry). Flat == generalized.
        for face_idx, face_name in enumerate(("px", "nx", "py", "ny")):
            self.infos[f"metrics/success_face_{face_name}"] = self.logged_face_success[face_idx]
        self.infos["metrics/obstacle_intensity"] = float(self.obs_dict["obstacle_intensity"])

        # Reward components (EMA across steps, horizon-independent)
        self.infos["reward/r_heading"] = self._reward_comp_ema["r_heading"]
        self.infos["reward/r_progress"] = self._reward_comp_ema["r_progress"]
        self.infos["reward/p_speed"] = self._reward_comp_ema["p_speed"]
        self.infos["reward/p_jerk"] = self._reward_comp_ema["p_jerk"]

        # Episode-end distance to target. Only refreshed on steps where something ended,
        # so the cached value carries between those steps.
        if ended.any():
            self._logged_ep_dist_to_target = float(self._get_dist_to_target(ended).mean())
        self.infos["metrics/dist_to_target_episode_end"] = self._logged_ep_dist_to_target

        # Flight metrics (mean across all envs)
        self.infos["metrics/v_horizontal"] = float(
            torch.norm(self.obs_dict["robot_linvel"][:, :2], dim=1).mean()
        )
        # Vertical velocity = world-frame z component (signed mean: +up / -down)
        self.infos["metrics/v_vertical"] = float(self.obs_dict["robot_linvel"][:, 2].mean())

    def process_image_observation(self):
        """Encode depth image through custom DepthVAE to get latent representation."""
        if self.task_config.vae_config.use_vae and self.vae_model is not None:
            image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
            self.image_latents[:] = self.vae_model.encode(image_obs)

    def get_return_tuple(self):
        """Build and return the step/reset output tuple.

        NOT a pure accessor: it rebuilds task_obs["observations"] from the current
        obs_dict. Building the state observation here (rather than in step()) is what
        makes every exit path correct by construction -- reset() and step() cannot
        return without it. Cheap and idempotent, so a double call is harmless.
        """
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _get_dist_to_target(self, env_ids=slice(None)):
        """World-frame distance from robot to target, in raw meters.

        Single source of truth for the scalar distance. Note this is a WORLD-frame
        magnitude -- for the vehicle-frame direction vector use
        _direction_and_distance_to_target() instead (the magnitudes agree, since
        rotation preserves length).

        Args:
            env_ids: index tensor or bool mask selecting a subset of envs.
                     Defaults to all envs.

        Returns:
            (num_envs,) or (len(env_ids),) tensor of distances in meters.
        """
        return torch.norm(
            self.target_position[env_ids] - self.obs_dict["robot_position"][env_ids],
            dim=1,
        )

    def _direction_and_distance_to_target(self):
        """Vehicle-frame unit vector to target and raw distance (meters), current step.

        Uses robot_vehicle_orientation (yaw-only, gravity-aligned heading frame), so
        the direction is invariant to the drone's roll/pitch and shares a frame with
        the vehicle-frame velocity used in the bearing reward.

        Single source of truth for both the observation vector
        (process_obs_for_task) and the heading reward (_reward_progress), so the
        two never drift out of sync on frame or staleness.

        Returns:
            direction: (num_envs, 3) unit vector to target in vehicle frame
            dist:      (num_envs, 1) distance to target in raw meters
        """
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            self.target_position - self.obs_dict["robot_position"],
        )
        dist = torch.linalg.norm(vec_to_tgt + 1e-6, dim=1, keepdim=True)
        direction = vec_to_tgt / (dist + 1e-6)
        return direction, dist

    def process_obs_for_task(self):
        """
        Build observation vector.

        - [0:3]     unit vector from drone to target                    vehicle
        - [3]       distance to target (normalized & clamped)           vehicle 
        - [4:7]     linear velocity (vx, vy, vz)                        vehicle
        - [7:10]    angular velocity (wx, wy, wz)                       body
        - [10:13]   gravity vector in body frame (gx, gy, gz)           body
        - [13:17]   previous action (thrust, roll, pitch, yaw_rate)     cmd (vehicle)
        - [17:49]   VAE latent encoding (32D)                           N/A

        Total: 49D observation vector (can be reduced by removing components if needed)."""
        
        # Estimator's view of the target geometry and velocity, NOT ground truth. The
        # reward path still calls _direction_and_distance_to_target() directly, so the
        # policy is paid for reaching the real target while only ever seeing a drifting
        # estimate of where it is -- which is the situation on the real vehicle.
        direction_to_target, dist_to_tgt, est_vehicle_linvel = self._estimated_state()

        # [0:3] Unit vector to target in vehicle frame
        self.task_obs["observations"][:, 0:3] = direction_to_target

        # [3] Distance to target, normalized by env diagonal and clamped to [0, 1].
        # dist_to_tgt is (num_envs, 1); squeeze to (num_envs,) to match max_dist (num_envs,).
        max_dist = torch.norm(
            self.obs_dict["env_bounds_max"] - self.obs_dict["env_bounds_min"], dim=-1)
        self.task_obs["observations"][:, 3] = torch.clamp(
            dist_to_tgt.squeeze(1) / (max_dist + 1e-6), 0.0, 1.0)
        
        # [4:7] Linear velocity in vehicle frame, as estimated (see _estimated_state)
        self.task_obs["observations"][:, 4:7] = est_vehicle_linvel  # TODO: NORMALIZE?

        # [7:10] Angular velocity in body frame -- from the SIMULATED IMU GYRO, not ground
        # truth, averaged over this env step's physics substeps (_consume_imu_gyro).
        #
        # This is a replacement, not an addition, and that distinction is the whole point.
        # Appending the IMU as extra channels would achieve nothing: the policy would still
        # have this clean ground-truth copy of the same quantity and would simply learn to
        # read that one. The IMU's job here is to DEGRADE the observation down to what the
        # real F450 can actually supply, so it has to replace the clean source.
        #
        # Scope: the gyro is the only channel the IMU honestly covers. direction_to_target,
        # distance and linvel all come from EKF2/VIO on the real vehicle, whose error model
        # is drift and latency rather than white noise -- integrating IMU accel for velocity
        # drifts unboundedly. [10:13] stays on true orientation until the accelerometer's
        # sign/offset convention is verified at hover (gravity_compensation=False and
        # world_frame=False mean imu_meas[:, 0:3] is a specific-force reading).
        self.task_obs["observations"][:, 7:10] = self._consume_imu_gyro() # TODO: NORMALIZE?
        
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

            # Per-wall success rates. Walls with no completed episodes this window are
            # treated as 0 so a wall that is never finished cannot silently pass the gate.
            face_rates = self._face_successes / self._face_attempts.clamp(min=1.0)
            face_rates = torch.where(
                self._face_attempts > 0, face_rates, torch.zeros_like(face_rates)
            )
            worst_face_rate = float(face_rates.min())

            # Update logged metrics for tensorboard
            self.logged_success_rate = float(success_rate)
            self.logged_crash_rate = float(crash_rate)
            self.logged_exceed_rate = float(exceed_rate)
            self.logged_timeout_rate = float(timeout_rate)
            self.logged_face_success = [float(r) for r in face_rates]

            # INCREASES gate on the WORST wall, not the pooled rate. The +/-y walls sit
            # ~30% closer to the centre spawn than +/-x, so short-bearing episodes
            # succeed first; gating on the pool would let them carry the aggregate over
            # the threshold and advance a policy that has only solved the easy
            # directions. DECREASES stay pooled -- a single bad wall should not
            # ratchet the whole curriculum down.
            if worst_face_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # Clamp curriculum level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self._publish_obstacle_intensity()

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, "
                f"Progress: {self.curriculum_progress_fraction:.2f}, "
                f"Obstacle Density: {self.obs_dict['obstacle_intensity']:.4f}/m^3"
            )
            logger.warning(
                f"Success Rate: {success_rate:.3f}, "
                f"Crash Rate: {crash_rate:.3f}, "
                f"Exceed Rate: {exceed_rate:.3f}, "
                f"Timeout Rate: {timeout_rate:.3f}"
            )
            logger.warning(
                "Success by wall (+x, -x, +y, -y): "
                + ", ".join(f"{r:.3f}" for r in self.logged_face_success)
                + f" | worst={worst_face_rate:.3f} (gates curriculum increases)"
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
            self._face_attempts.zero_()
            self._face_successes.zero_()

    def compute_rewards(self, obs_dict, current_action):
        """
        Compute reward from four mutually exclusive components (priority order):
        1. r_exceed:    out-of-bounds penalty (terminates)
        2. r_arrive:    reached target bonus (terminates as success)
        3. r_collision: obstacle collision penalty (terminates)
        4. r_prog:      dense shaping reward for normal steps (see _reward_progress)

        Returns:
            Tuple of (rewards, terminations, arrive_mask, exceed_mask) tensors
        """
        robot_pos = obs_dict["robot_position"]
        crashes = obs_dict["crashes"]

        # Distance to target
        dist = self._get_dist_to_target()

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
        reward = torch.zeros(self.sim_env.num_envs, device=self.device)
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
        Higher curriculum (more obstacles) = bigger reward.

        Scales on curriculum_progress_fraction, not level / curriculum.max_level: pinning
        the curriculum (--curriculum_level N) sets min_level == max_level == N, so the
        latter is 0/0 -> ZeroDivisionError at level 0 and a constant 1.0 at every other
        pinned level. The progress fraction is the same ramp for an un-pinned run, is
        already clamped and guarded against a zero denominator, and keys off the same
        reference level as the obstacle density -- so the bonus tracks the clutter the
        drone actually flew through, which is what this reward is for."""
        params = self.task_config.reward_parameters
        bonus_min = params["arrive_bonus_min"]
        bonus_max = params["arrive_bonus_max"]
        t = self.curriculum_progress_fraction
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
        3. p_speed:    lambda_v * v * max(0, v - v_max)
                       - PENALIZE 3D linear speed above v_max. v is the full
                         (vx, vy, vz) vehicle-frame velocity, so climbs and dives
                         count the same as forward flight. Yaw rate is NOT
                         penalized here -- only linear velocity.
        4. p_jerk:     lambda_jerk * ||(a_curr - a_prev) / action_scale||
                       - PENALIZE large changes in the (transformed) action, with each
                         channel normalized by its own command range so thrust, roll,
                         pitch and yaw_rate all weigh equally

        Args:
            mask: Boolean tensor indicating which envs get this reward
        Returns:
            Reward tensor for masked envs
        """
        params = self.task_config.reward_parameters

        # 1. Reward heading towards target: λ_b * dot(unit_vec_to_target, unit_vec_velocity)
        # Compute n/v from the current-step state (NOT task_obs["observations"],
        # which is only refreshed after compute_rewards in step() and would be one
        # step stale). The helper is the same source process_obs_for_task uses for
        # observations[:, 0:3]; robot_vehicle_linvel is the source for [:, 4:7].
        n, _ = self._direction_and_distance_to_target()  # current-step, vehicle frame
        v = self.obs_dict["robot_vehicle_linvel"]           # current-step, vehicle frame

        r_bearing = params["lambda_b"] *  torch.linalg.vecdot(n, v / (torch.linalg.norm(v, dim=1, keepdim=True) + 1e-6), dim=1)

        # 2. Reward progress towards target: λ_p * (prev_dist - current_dist).
        # current_dist MUST be in the same units as self.prev_dist (raw meters,
        # see step()/reset_idx). Compute it from positions here rather than using
        # observations[:, 3], which is normalized by the env diagonal to [0, 1].
        current_dist = self._get_dist_to_target()
        r_progress = params["lambda_p"] * (self.prev_dist - current_dist)
        
        # 3. Excess speed penalty: λ_v * v * max(0, v - v_max).
        # Full 3D linear speed (vx, vy, vz) -- intentional, not just horizontal.
        speed = torch.linalg.norm(v, dim=1)
        p_speed = params["lambda_v"] * speed * torch.clamp(
            speed - self.task_config.v_max, min=0.0
        )

        # 4. Penalize jerk (change in the transformed action), per-channel normalized by
        # each channel's own command range so the norm is dimensionless -- see
        # self._action_scale in __init__ for why the raw difference would not be.
        a_curr = current_action
        a_prev = self.prev_action
        p_jerk = params["lambda_jerk"] * torch.linalg.norm(
            (a_curr - a_prev) / self._action_scale, dim=1
        )

        # Apply mask to zero out rewards for envs that had terminal events
        r_bearing = r_bearing[mask]
        r_progress = r_progress[mask]
        p_speed = p_speed[mask]
        p_jerk = p_jerk[mask]

        # Update EMA for tensorboard reward component logging.
        # Guarded: when every env terminates on the same step the mask is empty, and
        # torch's mean() of an empty tensor is NaN -- which would poison the EMA
        # PERMANENTLY, since NaN propagates through every later update. Skipping the
        # step just holds the last value, which is what an EMA should do with no data.
        if mask.any():
            a = self._ema_alpha
            self._reward_comp_ema["r_heading"] += a * (float(r_bearing.mean()) - self._reward_comp_ema["r_heading"])
            self._reward_comp_ema["r_progress"] += a * (float(r_progress.mean()) - self._reward_comp_ema["r_progress"])
            self._reward_comp_ema["p_speed"] += a * (float(p_speed.mean()) - self._reward_comp_ema["p_speed"])
            self._reward_comp_ema["p_jerk"] += a * (float(p_jerk.mean()) - self._reward_comp_ema["p_jerk"])

        return r_bearing + r_progress + p_speed + p_jerk
