from aerial_gym.config.sensor_config.base_sensor_config import BaseSensorConfig
import numpy as np

class RealSenseD435CamConfig(BaseSensorConfig):
    """
    RealSense D435 camera sensor configuration.
    Inherits from BaseSensorConfig and overrides specific parameters for the RealSense D435 camera.

    The RealSense D435 camera is defined at: sail-uav-core/sensors/realsense_d435/model.sdf
    """

    num_sensors = 1  # number of sensors of this type. More than 1 sensor on the same link don't make sense. Can be implemented if needed for multiple different links.

    sensor_type = "camera"  # sensor type

    height = 180
    width = 320
    horizontal_fov_deg = 87.0  # horizontal field of view in degrees
    max_range = 10.0  # maximum range of the camera in meters
    min_range = 0.1  # minimum range of the camera in meters

    calculate_depth = (
        True    # Get a depth image and not a range image. False will result in a range image.
    )

    return_pointcloud = False  # Return point cloud instead of an image. Above depth option will be ignored if True.
    # NOTE: spelled "pointcloud", not "point_cloud". aerial_gym reads this exact name in
    # warp_sensor.normalize_observation(); the underscored variant silently never matches
    # and raises AttributeError during prepare_for_sim.
    pointcloud_in_world_frame = False
    segmentation_camera = False  # If True, the camera will return a segmentation image instead of a depth image. The segmentation image will have the same size as the depth image and will contain the object IDs of the objects in the scene.

    # Tranform sensor element coordinates to camera_link frame.
    euler_frame_rot_deg = [-90.0, 0.0, -90.0]  
    
    if (return_pointcloud and pointcloud_in_world_frame):
        normalize_range = False
    else:
        normalize_range = True
    
     # what to do with out of range values
    far_out_of_range_value = (
        max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
    near_out_of_range_value = (
        -max_range if normalize_range == True else -1.0
    )  # Will be [-1]U[0,1] if normalize_range is True, otherwise will be value set by user in place of -1.0
    
    
    # randomize placement of the sensor
    randomize_placement = True  # NOTE: KEEP ALWAYS TRUE.
    min_translation = [0.07, -0.03, -0.04]
    max_translation = [0.12, 0.03, -0.01]
    min_euler_rotation_deg = [-5.0, -5.0, -5.0]
    max_euler_rotation_deg = [5.0, 5.0, 5.0]

    # nominal position and orientation (only for Isaac Gym Camera Sensors)
    # If you choose to use Isaac Gym sensors, their position and orientation will NOT be randomized
    nominal_position = [0.10, 0.0, -0.03]
    nominal_orientation_euler_deg = [0.0, 0.0, 0.0]

    use_collision_geometry = False

    class sensor_noise:
        enable_sensor_noise = False
        pixel_dropout_prob = 0.01
        pixel_std_dev_multiplier = 0.01
    