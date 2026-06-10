"""
Custom robot configuration for navigation with obstacles.
Inherits from BaseQuadWithCameraCfg and overrides spawn position.
"""
import numpy as np
from aerial_gym.config.robot_config.base_quad_config import BaseQuadWithCameraCfg


class NavQuadWithCameraCfg(BaseQuadWithCameraCfg):
    """
    Quadrotor with depth camera for navigation task.
    Spawns at a randomized position near the start of the environment (low X,
    random Y/Z) but level and motionless: attitude, linear velocity and angular
    velocity are all fixed at zero (min == max below).
    """

    class init_config(BaseQuadWithCameraCfg.init_config):
        # Tensor format: [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        # Position indices 0-2: ratio of env bounds (0-1), interpolated to world coords at reset
        # Attitude indices 3-5: direct radians
        # Velocity indices 7-9: direct m/s (world frame)
        # Angular rate indices 10-12: direct rad/s (body frame)
        min_init_state = [
            0.00,                    # X: 0% of env bounds
            0.10,                    # Y: 10% of env bounds
            0.10,                    # Z: 10% of env bounds
            0.0,                     # Roll:  0 rad (no randomization)
            0.0,                     # Pitch: 0 rad (no randomization)
            0.0,                     # Yaw:   0 rad (no randomization)
            1.0,
            0.0,                     # vx: 0 m/s (no randomization)
            0.0,                     # vy: 0 m/s (no randomization)
            0.0,                     # vz: 0 m/s (no randomization)
            0.0,                     # wx: 0 rad/s (no randomization)
            0.0,                     # wy: 0 rad/s (no randomization)
            0.0,                     # wz: 0 rad/s (no randomization)
        ]
        max_init_state = [
            0.05,                    # X: 5% of env bounds  -> position is randomized in [0%, 5%]
            0.90,                    # Y: 90% of env bounds -> position is randomized in [10%, 90%]
            0.90,                    # Z: 90% of env bounds -> position is randomized in [10%, 90%]
            0.0,                     # Roll:  0 rad (== min -> level, not randomized)
            0.0,                     # Pitch: 0 rad (== min -> level, not randomized)
            0.0,                     # Yaw:   0 rad (== min -> fixed heading, not randomized)
            1.0,
            0.0,                     # vx: 0 m/s (== min -> motionless, not randomized)
            0.0,                     # vy: 0 m/s (== min -> motionless, not randomized)
            0.0,                     # vz: 0 m/s (== min -> motionless, not randomized)
            0.0,                     # wx: 0 rad/s (== min -> no rotation, not randomized)
            0.0,                     # wy: 0 rad/s (== min -> no rotation, not randomized)
            0.0,                     # wz: 0 rad/s (== min -> no rotation, not randomized)
        ]
