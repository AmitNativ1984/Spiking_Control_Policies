from config.sensor_config.gazebo_imu_config import GazeboImuConfig
from config.sensor_config.genx320_event_cam_config import GenX320EventCamConfig

from .f450_config import F450Config


class F450EventCamConfig(F450Config):
    """F450 with a Prophesee GenX320 event camera in place of the RealSense D435."""

    class sensor_config:
        enable_camera = True
        camera_config = GenX320EventCamConfig

        enable_imu = True
        imu_config = GazeboImuConfig

        enable_lidar = False
