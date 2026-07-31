from operator import index

import numpy as np
from pathlib import Path
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig
from aerial_gym.config.sensor_config.imu_config.vn100_config import VN100Config
from aerial_gym.config.sensor_config.imu_config.bosch_bmi088_config import BoschBMI088Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig
from sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig
from sensor_config.gazebo_imu_config import GazeboImuConfig
class F450Config:
    """
    F450 quadrotor configuration.
    This class defines the configuration parameters for the F450 quadrotor, including its physical properties and sensor configurations.
    """

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        min_init_state = [
            0.1, 0.15, 0.15,
            -np.pi / 6.0, -np.pi / 6.0, -np.pi,
            1.0,
            -0.1, -0.1, -0.1,
            -0.1, -0.1, -0.1
        ]
        max_init_state = [
            0.9, 0.85, 0.85,
            np.pi / 6.0, np.pi / 6.0, np.pi,
            1.0,
            0.1, 0.1, 0.1,
            0.1, 0.1, 0.1
        ]

    class sensor_config:
        enable_camera = True
        camera_config = RealSenseD435CamConfig

        enable_imu = True
        imu_config = GazeboImuConfig

        enable_lidar = False

    class disturbance:
        enable_disturbance = True
        prob_apply_disturbance = 0.05
        max_force_and_torque_disturbance = [1.5, 1.5, 1.5, 0.1, 0.1, 0.1]

    class damping:
        linvel_linear_damping_coefficient = [0.0029, 0.0029, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class robot_asset:
        asset_folder = str(
            Path(__file__).resolve().parent.parent.parent / "resources" / "robots" / "f450"
        )
        file = "model.urdf"
        name =  "base_quadrotor"
        base_link_name = "base_link"
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        collision_mask = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        density = 0.000001
        angular_damping = 0.01
        linear_damping = 0.01
        max_angular_velocity = 100.0  # rad/s
        max_linear_velocity = 100.0 # m/s
        armature = 0.001

        semantic_id = 0
        per_link_semantic = False

        # NOTE: unused for the robot itself (the robot's spawn-state randomization is
        # driven by init_config.min_init_state/max_init_state instead). Kept here only
        # because asset_loader.py reads this attribute unconditionally for every asset.
        min_state_ratio = [
            0.1,
            0.1,
            0.1,
            0,
            0,
            -np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        max_state_ratio = [
            0.3,
            0.9,
            0.9,
            0,
            0,
            np.pi,
            1.0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]

        color = None
        semantic_masked_links = {}
        keep_in_env = True

        min_position_ratio = None
        max_position_ratio = None

        min_euler_angle = [-np.pi, -np.pi, -np.pi]
        max_euler_angle = [np.pi, np.pi, np.pi]

        place_force_sensor = True
        force_sensor_parent_link = "base_link"
        force_sensor_transform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # [x, y, z, qx, qy, qz, qw]

        use_collision_mesh_instead_of_visual = False

    class control_allocator_config:
        num_motors = 4
        force_application_level = "motor_link"

        # index 0: 'base_link'
        # index 1: 'back_left_prop'
        # index 2: 'back_right_prop'
        # index 3: 'front_left_prop'
        # index 4: 'front_right_prop'

        application_mask = [4, 1, 3, 2] # [front_right_prop, back_left_prop, front_left_prop, back_right_prop] (matches the motor_link names in the URDF)
        motor_directions = [1, 1, -1, -1]


        # The allocation matrix maps the input motor speeds to the resulting wrenches (forces and torques) applied to the robot's base link. The matrix is structured as follows:
        # | Fx |    
        # | Fy |              | u4 |
        # | Fz | = [A(6x4)] * | u1 |  
        # | Tx |              | u3 |  
        # | Ty |              | u2 |
        # | Tz | 
        
        # (1) The thrust of a motor around its axis (base_link z-axis) is defined as: F = K_T * w^2, where K_T is the thrust coefficient and w is the motor speed in [rad/s]. 
        # (2) The torque of a motor around its axis (base_link z-axis) is defined as  T = K_M * w^2, where K_M is the torque coefficient and w is the motor speed in [rad/s].
        # (3) The torque around the x and y axes is generated by the thrust of the motors, which is applied at a distance from the center of mass:
        #       Tx = K_T * w^2 * Lx
        #       Ty = K_T * w^2 * Ly
        #
        # NOTE: In the allocation matrix, each column i,
        # is what motor i contributes to (Fx, Fy, Fz, Tx, Ty, Tz)
        # per UNIT of thrust! 
        #
        theta_deg = 45.0  # angle of the motors relative to the x-axis in the x-y plane
        theta = np.radians(theta_deg)

        K_T = 1.004544e-05
        K_M = 0.016

        arm_length = 0.23 # [m] distance from the center of mass to each motor (in the x-y plane)
        # Columns are ordered [front_right, back_left, front_left, back_right], matching
        # application_mask/motor_directions above. Positions (body frame, x-forward, y-left):
        #   front_right = (+arm*cos(theta), -arm*sin(theta))
        #   back_left   = (-arm*cos(theta), +arm*sin(theta))
        #   front_left  = (+arm*cos(theta), +arm*sin(theta))
        #   back_right  = (-arm*cos(theta), -arm*sin(theta))
        # Tx = y_i, Ty = -x_i, Tz = -motor_directions[i] * K_M
        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [-arm_length * np.sin(theta), arm_length * np.sin(theta), arm_length * np.sin(theta), -arm_length * np.sin(theta)],
            [-arm_length * np.cos(theta), arm_length * np.cos(theta), -arm_length * np.cos(theta), arm_length * np.cos(theta)],
            [-K_M, -K_M, K_M, K_M],
        ]

        class motor_model_config:
            use_rps = True
            # nominal K_T (Gazebo motorConstant) = 1.004544e-05, randomized +/-10% per episode reset
            motor_thrust_constant_min = 9.040896e-06
            motor_thrust_constant_max = 1.104998e-05
            # nominal spin-up tau (Gazebo timeConstantUp) = 0.0125 s, randomized +/-20% per episode reset
            motor_time_constant_increasing_min = 0.01
            motor_time_constant_increasing_max = 0.015
            # nominal spin-down tau (Gazebo timeConstantDown) = 0.025 s, randomized +/-20% per episode reset
            motor_time_constant_decreasing_min = 0.02
            motor_time_constant_decreasing_max = 0.03
            max_thrust = 10.045  # K_T * maxRotVelocity^2 = 1.004544e-05 * 1000^2 (Gazebo's rotor speed ceiling)
            min_thrust = 0.0
            max_thrust_rate = 100000.0
            thrust_to_torque_ratio = 0.016  # K_M, same value used in allocation_matrix's yaw row
            use_discrete_approximation = True