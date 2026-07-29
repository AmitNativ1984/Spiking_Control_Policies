import numpy as np
from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig

class GazeboImuConfig(BaseSensorConfig):
    """
    Gazebo IMU sensor configuration.
    Inherits from BaseSensorConfig and overrides specific parameters for the Gazebo IMU.

    The gazebo IMU sensor is defined at: sail-uav-core/gazebo-sitl/models/f450/model.sdf

    NOTE: The Gazebo IMU noise values are raw per-sample std at 250Hz, not the normalized density
    used in AerialGym. Therefore, the noise values need to be scaled by sqrt(Freq[Hz]) to get the normalized density values.
    """
    
    num_sensors = 1  # number of sensors of this type. More than 1 sensor on the same link don't make sense. Can be implemented if needed for multiple different links.

    sensor_type = "imu"  # sensor type

    world_frame = False

    # enable or disable noise and bias. Setting to False will simulate a perfect, noise- and bias-free IMU
    enable_noise = True
    enable_bias = True

    
    gazebo_update_rate = 250.0  # update rate of the Gazebo IMU sensor in Hz

    # bias and noise values as obtained from a sample Gazebo IMU model SDF
    # Aerial Gym noise density = Gazebo noise std * sqrt(Gazebo update rate)
    bias_std = [
        9.782812831313576e-07,
        9.782812831313576e-07,
        9.782812831313576e-07,
        2.6541629581345176e-05,
        2.6541629581345176e-05,
        2.6541629581345176e-05,
    ]  # first 3 values for acc bias std, next 3 for gyro bias std
    imu_noise_std = (np.array([
        0.021707945151263165,
        0.021707945151263165,
        0.021707945151263165,
        0.0009658627480635097,
        0.0009658627480635097,
        0.0009658627480635097,
        ])/ np.sqrt(gazebo_update_rate)  # first 3 vaues for acc noise std, next 3 for gyro noise std
    ).tolist()  

    max_measurement_value = [
        100.0,
        100.0,
        100.0,
        10.0,
        10.0,
        10.0,
    ]  # max measurement value for acc and gyro outputs will be clamped by + & - of these

    max_bias_init_value = [
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
        1.0e-03,
    ]  # max bias init value for acc and gyro biases will be sampled within +/- of this range

    # Setting this to true will provide acceelration of the object in a static frame w.r.t ground.

    gravity_compensation = False  # usually the force sensor computes total force including gravity, so set this to False

    # The position of this is hardcoded at the center of the asset. This can be changed by the user in code if needed.

    # Randomize the orientation of the sensor w.r.t the parent link. The position is still [0,0,0] in the parent link frame
    randomize_placement = False
    min_translation = [0.07, -0.06, 0.01]
    max_translation = [0.12, 0.03, 0.04]
    min_euler_rotation_deg = [-2.0, -2.0, -2.0]
    max_euler_rotation_deg = [2.0, 2.0, 2.0]