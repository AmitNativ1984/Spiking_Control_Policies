"""Camera carrier for depth-dataset collection: the F450, not flying.

Subclasses F450Config so the camera the dataset is rendered through is literally the one
the policy will read from -- RealSenseD435CamConfig, 320x180 native, with its mount
translation/rotation randomization and its noise model. The old data-gen config declared
a separate 1280x720 D435 that had to be downsampled offline, which meant the encoder was
trained on a nearest-downsample of a 1280x720 ray-cast and then run on a native 320x180
one. Different ray directions per pixel, for no benefit.

What IS overridden is only the difference between carrying a camera and flying: the pose
is teleported rather than flown, so gravity, disturbances and mass randomization are off.
"""

import numpy as np

from config.robot_config.f450_config import F450Config
from config.task_config.f450_attitude_navigation_task_config import (
    task_config as NavTaskConfig,
)

# The real viewpoint envelope is the COMMANDED tilt limit, not the spawn tilt. Sourced
# from the task rather than written out, so a change to the attitude envelope shows up in
# the dataset instead of silently going out of sync.
#
# f450_config.init_config uses +-pi/6, but that is only where an episode STARTS; the
# policy spends the rest of it anywhere up to max_inclination_angle_rad. The old value
# here was +-pi/3, wider than the drone can ever command.
_MAX_TILT = NavTaskConfig.max_inclination_angle_rad


class DataGenF450Cfg(F450Config):
    class init_config:
        # [ratio_x, ratio_y, ratio_z, roll, pitch, yaw, 1.0, vx, vy, vz, wx, wy, wz]
        #
        # Position roams the WHOLE env, unlike the policy's small centre spawn box: the
        # dataset wants viewpoints from everywhere the drone can reach during an episode,
        # not just where episodes begin.
        #
        # NOTE this range must NOT be fed to PoissonAssetManager.spawn_ratio_lo/hi. That
        # pair sizes the obstacle keep-out ellipsoid, and a keep-out spanning the whole
        # env would drive free_volume negative and produce an empty world on every reset.
        # generate_dataset.py passes the F450's real spawn box instead -- see the comment
        # there.
        #
        # z stays clear of both the floor slab (bottom_wall is 0.2 m thick, centred on the
        # env's z minimum) and the ceiling, so the camera is never inside the floor.
        min_init_state = [
            0.05, 0.05, 0.15,
            -_MAX_TILT, -_MAX_TILT, -np.pi,
            1.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]
        max_init_state = [
            0.95, 0.95, 0.85,
            _MAX_TILT, _MAX_TILT, np.pi,
            1.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]

    class sensor_config(F450Config.sensor_config):
        # Camera inherited (RealSenseD435CamConfig). The IMU is dead weight here.
        enable_imu = False

    class disturbance:
        enable_disturbance = False
        prob_apply_disturbance = 0.0
        max_force_and_torque_disturbance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    class domain_randomization:
        # Per-env inertia spread only matters to a policy that has to fly through it.
        randomize_mass_properties = False
        inertia_scale_range = [1.0, 1.0]

    class robot_asset(F450Config.robot_asset):
        # The pose is written by reset_idx and must survive until the render. With gravity
        # on, the single physics substep would drop it ~0.5 mm -- harmless, but there is
        # no reason to integrate a fall at all.
        disable_gravity = True
