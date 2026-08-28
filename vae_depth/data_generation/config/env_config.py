"""Environment for depth-dataset collection: the navigation env, not flying.

Everything that determines what the camera SEES -- obstacle mix and pool sizes, Poisson
placement, env bounds, the floor slab, the cullable perimeter walls -- is INHERITED from
ForestEnvCfg rather than restated here. That is the entire point of this file.

The previous version of it declared its own obstacle set (panels/thin/trees/objects in
per-asset ratio boxes, no floor) and then drifted: by the time the navigation env had
grown spheres, cylinders, perimeter walls and a ground plane, the VAE was still being
trained on a world containing none of them. Anything restated here is something that can
drift again, so only collection-specific env FLAGS are overridden below.
"""

from config.env_config.env_forest_with_obstacles import ForestEnvCfg


class DataGenEnvCfg(ForestEnvCfg):
    class env(ForestEnvCfg.env):
        # num_envs comes from the CLI, as it does for ForestEnvCfg.

        # The camera is teleported into the obstacle field and will sometimes land inside
        # something. That is a frame to reject (generate_dataset's valid-pixel filter),
        # not an episode to terminate -- and a reset here would throw away the other
        # envs' poses too.
        reset_on_collision = False

        # Nothing is being simulated. One substep per capture is enough to push the reset
        # state into the sim before the render; the nav env's 3 +/- 1 only exists to set
        # the control rate.
        num_physics_steps_per_env_step_mean = 1
        num_physics_steps_per_env_step_std = 0

        # Both are policy-training devices, and both would put noise into the geometry the
        # VAE is supposed to learn. The camera's OWN noise/mount randomization stays on --
        # that lives in RealSenseD435CamConfig and is part of the sensor model.
        perturb_observations = False
        sample_timestep_for_latency = False

        render_viewer_every_n_steps = 1000000  # headless collection; effectively never
