import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from aerial_gym.config.sensor_config.camera_config.base_depth_camera_config import BaseDepthCameraConfig
from aerial_gym.config.sensor_config.imu_config.vn100_config import VN100Config
from aerial_gym.config.sensor_config.imu_config.bosch_bmi088_config import BoschBMI088Config
from aerial_gym.config.sensor_config.imu_config.base_imu_config import BaseImuConfig
from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig
from config.sensor_config.gazebo_imu_config import GazeboImuConfig

_ASSET_FOLDER = Path(__file__).resolve().parent.parent.parent / "resources" / "robots" / "f450"
_URDF_FILE = "model.urdf"


def _urdf_center_of_mass(urdf_path):
    """Mass-weighted center of mass of every link, in the base_link frame.

    DERIVED, not hardcoded, and that is the point. control_allocator_config below takes
    moments about this point while PhysX takes them about the CoM it computes from the same
    URDF. If the two ever disagree, "commanded zero torque" stops meaning zero torque: four
    equal thrusts about a CoM 3.6 mm off-axis are m*g*0.0036 = 0.071 N.m of pitch torque,
    and the Lee attitude controller is pure PD (no integral term), so it can only balance
    that with a permanent attitude error -- measured at 4.06 deg of pitch, which flew the
    drone sideways at 2.6 m/s under a zero action. A constant here would silently
    reintroduce that bias the first time the URDF's inertial block changed.

    Assumes the F450's flat tree of fixed joints with no rotation between link frames,
    which is what model.urdf has; raises if the joints are not ordered root-first.
    """
    root = ET.parse(urdf_path).getroot()

    # Link frame origins relative to base_link.
    frames = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        if parent not in frames:
            if frames:
                raise ValueError(f"{urdf_path}: joints are not ordered root-first ('{parent}' unseen)")
            frames[parent] = np.zeros(3)  # first parent encountered is the root
        frames[child] = frames[parent] + xyz

    total_mass, weighted = 0.0, np.zeros(3)
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        com = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        total_mass += mass
        weighted += mass * (frames.get(link.get("name"), np.zeros(3)) + com)

    return weighted / total_mass


# (0.003602, 0.0, -0.007492) for the current URDF, matching the CoM Isaac Gym logs at build.
_COM = _urdf_center_of_mass(_ASSET_FOLDER / _URDF_FILE)


class F450Config:
    """
    F450 quadrotor configuration.
    This class defines the configuration parameters for the F450 quadrotor, including its physical properties and sensor configurations.
    """

    class init_config:
        # init_state tensor is of the format [ratio_x, ratio_y, ratio_z, roll_radians, pitch_radians, yaw_radians, 1.0 (for maintaining shape), vx, vy, vz, wx, wy, wz]
        # [ratio_x, ratio_y, ratio_z, roll_rad, pitch_rad, yaw_rad, 1.0, vx, vy, vz, wx, wy, wz]
        #
        # Spawn in a SMALL BOX AT THE ENV CENTRE. Targets are sampled on the four vertical
        # env walls (see task_config.target_*), so starting at the centre makes every
        # bearing equally likely instead of the old "always fly +x" layout.
        #
        # The z range is deliberately wider than x/y: it is the only source of elevation
        # randomization at spawn, and the vertical bearing component (observations[2]) is
        # otherwise never exercised.
        #
        # Yaw stays uniform on +/-pi. This is what makes the OBSERVED bearing uniform:
        # observations[0:3] is expressed in the vehicle (yaw-only) frame, so a random
        # heading spreads the target direction over all azimuths. Do not narrow it.
        #
        # SINGLE SOURCE OF TRUTH for the spawn box: PoissonAssetManager reads
        # min/max_init_state[0:3] to size its keep-out ellipsoid, so obstacles can never
        # be placed where the drone is about to appear.
        min_init_state = [
            0.45, 0.45, 0.35,
            -np.pi / 6.0, -np.pi / 6.0, -np.pi,
            1.0,
            -0.1, -0.1, -0.1,
            -0.1, -0.1, -0.1
        ]
        max_init_state = [
            0.55, 0.55, 0.65,
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

    class domain_randomization:
        # Per-env INERTIA spread, drawn ONCE AT BUILD via gym.set_actor_rigid_body_properties.
        #
        # Once, not per episode, and that is not a preference: under the GPU pipeline that
        # call discards the root states the episode reset just staged, so resampling it from
        # reset_idx() made every episode start at the world origin instead of the sampled
        # spawn. See task/attitude_navigation_task.py:_randomize_mass_properties.
        #
        # Inertia is the only one of mass/CoM/inertia worth randomizing here, because it is
        # the only one the controller is not told about. robot_manager copies ONE build-time
        # inertia to every env (robot_manager.py:471), so a per-env draw is a genuine
        # plant/model mismatch the policy has to absorb.
        #
        # Dropped, both verified empirically rather than assumed:
        #   mass  - the task wrote every draw into obs_dict["robot_mass"], which is exactly
        #           the tensor the Lee controller uses for (a+1)*m*g. Handing the controller
        #           the answer makes hover stay at command 0 for ANY mass: forcing a 2x draw
        #           left the drone still hovering instead of climbing. It bought no
        #           robustness. Thrust-to-weight uncertainty is already covered from the
        #           other side by motor_thrust_constant_min/max, which the controller does
        #           NOT see. To randomize T/W directly, decouple the controller's ASSUMED
        #           mass from the physics mass -- randomizing both together cancels out.
        #   CoM   - gym.set_actor_rigid_body_properties silently drops CoM writes after
        #           prepare_sim. Readback returns the URDF value under both `p.com.x = v`
        #           and `p.com = gymapi.Vec3(...)`, and a forced +-5 cm offset left every env
        #           at an identical attitude. The knob did nothing. It would also fight
        #           control_allocator_config, whose moment arms are referenced to the URDF
        #           CoM; moving the real CoM without updating those arms is precisely the
        #           bias that was just removed.
        randomize_mass_properties = True
        inertia_scale_range = [0.85, 1.15]

    class damping:
        linvel_linear_damping_coefficient = [0.28, 0.28, 0.0]  # along the body [x, y, z] axes
        linvel_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes
        angular_linear_damping_coefficient = [0.077, 0.077, 0.01]  # along the body [x, y, z] axes
        angular_quadratic_damping_coefficient = [0.0, 0.0, 0.0]  # along the body [x, y, z] axes

    class robot_asset:
        # Same constants the CoM above is parsed from, so the allocation matrix can never
        # be derived from a different URDF than the one Isaac Gym loads.
        asset_folder = str(_ASSET_FOLDER)
        file = _URDF_FILE
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

        arm_length = 0.23 # [m] distance from the base_link origin to each motor (in the x-y plane)
        # Columns are ordered [front_right, back_left, front_left, back_right], matching
        # application_mask/motor_directions above. Positions (body frame, x-forward, y-left),
        # relative to the base_link ORIGIN:
        #   front_right = (+arm*cos(theta), -arm*sin(theta))
        #   back_left   = (-arm*cos(theta), +arm*sin(theta))
        #   front_left  = (+arm*cos(theta), +arm*sin(theta))
        #   back_right  = (-arm*cos(theta), -arm*sin(theta))
        motor_x = arm_length * np.cos(theta) * np.array([+1.0, -1.0, +1.0, -1.0])
        motor_y = arm_length * np.sin(theta) * np.array([-1.0, +1.0, +1.0, -1.0])

        # Moment arms are referenced to the CENTER OF MASS, not the base_link origin, because
        # that is the point PhysX takes moments about. The two differ by 3.6 mm on this
        # airframe (the battery sits forward), and referencing the origin instead made a
        # commanded zero torque produce 0.071 N.m of real pitch torque -- see
        # _urdf_center_of_mass above for what that did to the vehicle.
        #
        # Tx = (y_i - com_y), Ty = -(x_i - com_x), Tz = -motor_directions[i] * K_M
        allocation_matrix = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            list(motor_y - _COM[1]),
            list(-(motor_x - _COM[0])),
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