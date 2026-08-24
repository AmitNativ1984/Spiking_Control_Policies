import numpy as np
from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig

class GazeboImuConfig(BaseSensorConfig):
    """
    Gazebo IMU sensor configuration.
    Inherits from BaseSensorConfig and overrides specific parameters for the Gazebo IMU.

    The gazebo IMU sensor is defined at: sail-uav-core/gazebo-sitl/models/f450/model.sdf

    NOTE: The Gazebo IMU noise values are raw per-sample std at 250 Hz, not the
    continuous-time noise DENSITY that Aerial Gym expects. The two are related by

        per_sample_std = density * sqrt(f)   <=>   density = per_sample_std / sqrt(f)

    so converting Gazebo -> Aerial Gym means DIVIDING by sqrt(250). That is what the
    code below does.

    Do not "correct" it to a multiplication. Aerial Gym's imu_sensor.py:76 computes
    `noise = randn * imu_noise_std / sqrt_dt`, which is what makes imu_noise_std a
    density in the first place; multiplying instead of dividing inflates the noise by
    f = 250x (accel 1.37e-03 -> 3.43e-01). This docstring and the comment on the
    imu_noise_std assignment both stated the conversion backwards until 2026-08-05,
    while the code was right the whole time.
    """
    
    num_sensors = 1  # number of sensors of this type. More than 1 sensor on the same link don't make sense. Can be implemented if needed for multiple different links.

    sensor_type = "imu"  # sensor type

    world_frame = False

    # enable or disable noise and bias. Setting to False will simulate a perfect, noise- and bias-free IMU
    enable_noise = True
    enable_bias = True

    
    gazebo_update_rate = 250.0  # update rate of the Gazebo IMU sensor in Hz

    # Noise values read verbatim from model.sdf's <imu> block (verified 2026-08-05:
    # they match the SDF exactly, on all 3 axes for both accel and gyro).
    #   Aerial Gym noise density = Gazebo per-sample std / sqrt(Gazebo update rate)
    # -> accel 1.3729e-03, gyro 6.1087e-05. See the docstring on the divide vs multiply.
    #
    # BIAS: ported from the SDF (was the verbatim VN100 default from base_imu_config.py).
    #
    # The two simulators model bias differently, so this is not a value swap:
    #   gz-sensors   draws bias ONCE at load and holds it -- a fixed turn-on offset. The
    #                SDF sets no dynamic_bias_stddev, so there is no drift.
    #   Aerial Gym   random-walks it every substep, `bias += randn * bias_std * sqrt_dt`
    #                (imu_sensor.update_bias), seeded by a uniform draw in
    #                +/- max_bias_init_value (imu_sensor.reset_idx).
    #
    # Parity therefore means: kill the walk (bias_std = 0) and put the SDF's bias_stddev
    # into max_bias_init_value, so each episode gets a fixed turn-on bias of the right
    # magnitude and holds it -- which is what the real gyro does within a flight.
    #
    # Distribution: Gazebo draws bias ~ N(0, sigma), Aerial Gym draws uniform on
    # +/- max_bias_init_value. A Gaussian is unbounded, so matching the BOUND is
    # meaningless -- match the variance instead. Uniform on +/- a has std a/sqrt(3), so
    # a = sigma * sqrt(3). That factor is folded into the values below.
    #
    # Previously max_bias_init_value was 1.0e-03 on every axis: 65x the SDF's accel bias
    # sigma and 2.4x the gyro's, with an unmodelled random walk on top.
    bias_std = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # no dynamic bias in the SDF -> no drift
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

    # Turn-on bias, drawn uniformly in +/- these values once per episode and then held
    # (bias_std = 0 above). Values are the SDF's <bias_stddev> * sqrt(3), so the uniform
    # draw has the same VARIANCE as Gazebo's Gaussian:
    #   accel 1.5466700035883543e-05 * sqrt(3) = 2.678911e-05
    #   gyro  4.196600114477603e-04  * sqrt(3) = 7.268725e-04
    max_bias_init_value = [
        2.678911e-05,
        2.678911e-05,
        2.678911e-05,
        7.268725e-04,
        7.268725e-04,
        7.268725e-04,
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