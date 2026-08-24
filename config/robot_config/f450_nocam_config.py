from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig
from config.sensor_config.gazebo_imu_config import GazeboImuConfig
from .f450_config import F450Config

class F450NoCamConfig(F450Config):
    """
    F450 quadrotor configuration, without camera
    """

    class sensor_config:
        enable_camera = False
        camera_config = RealSenseD435CamConfig

        enable_imu = True
        imu_config = GazeboImuConfig

        enable_lidar = False