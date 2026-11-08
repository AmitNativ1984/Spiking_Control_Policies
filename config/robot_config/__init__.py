from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor

from env_manager import warp_event_sensor_patch

from .f450_config import F450Config
from .f450_event_cam_config import F450EventCamConfig
from .f450_nocam_config import F450NoCamConfig

robot_registry.register("f450", BaseMultirotor, F450Config)
robot_registry.register("f450_nocam", BaseMultirotor, F450NoCamConfig)

# upstream WarpSensor rejects sensor_type="event_camera"; the patch passes other types through
warp_event_sensor_patch.apply()
robot_registry.register("f450_event_cam", BaseMultirotor, F450EventCamConfig)
