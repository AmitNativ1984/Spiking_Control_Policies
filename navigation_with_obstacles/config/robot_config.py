"""
Custom robot configuration for navigation with obstacles.
Inherits from BaseQuadWithCameraCfg and overrides spawn position.
"""
import numpy as np
from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraCfg


class NavQuadWithCameraCfg(BaseQuadWithCameraCfg):
    """
    Quadrotor with depth camera for navigation task.
    Spawns near the start of the environment (low X) with random yaw.
    """

    class init_config(BaseQuadWithCameraCfg.init_config):
        # Tensor format: [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        # Position indices 0-2: ratio of env bounds (0-1), interpolated to world coords at reset
        # Attitude indices 3-5: direct radians
        # Velocity indices 7-9: direct m/s (world frame)
        # Angular rate indices 10-12: direct rad/s (body frame)
        min_init_state = [
            0.00,                    # X: 0-5% of env bounds
            0.10,                    # Y: 10-90%
            0.10,                    # Z: 10-90%
            -np.pi / 6,              # Roll:  -30 deg
            -np.pi / 6,              # Pitch: -30 deg
            -np.pi,                  # Yaw:   -180 deg (full 360 coverage of drone-to-target bearing)
            1.0,
            -0.5,                    # vx: +-0.5 m/s
            -0.5,                    # vy: +-0.5 m/s
            -0.3,                    # vz: +-0.3 m/s
            -0.3,                    # wx: +-0.3 rad/s (~17 deg/s roll rate)
            -0.3,                    # wy: +-0.3 rad/s (~17 deg/s pitch rate)
            -0.2,                    # wz: +-0.2 rad/s (~11 deg/s yaw rate)
        ]
        max_init_state = [
            0.05,                    # X
            0.90,                    # Y
            0.90,                    # Z
            np.pi / 6,              # Roll:  +30 deg
            np.pi / 6,              # Pitch: +30 deg
            np.pi,                   # Yaw:   +180 deg (full 360 coverage of drone-to-target bearing)
            1.0,
            0.5,                     # vx
            0.5,                     # vy
            0.3,                     # vz
            0.3,                     # wx
            0.3,                     # wy
            0.2,                     # wz
        ]
