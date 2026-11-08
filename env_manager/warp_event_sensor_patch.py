"""Teach aerial_gym's warp sensor path about sensor_type = "event_camera".

Upstream ``WarpSensor.__init__`` raises NotImplementedError for any type it does not know,
and the event camera needs three things upstream does not provide:

* a clock: events depend on elapsed sim time, but ``EnvManager.step`` draws its substep
  count into a local variable. ``EnvManager.simulate`` is wrapped to count physics steps
  into ``global_tensor_dict["physics_step_count"]``; the sensor reads count * dt.
* per-asset albedo: ``velocities[:, 1]`` of every env mesh is set to the asset's index
  within its env (``velocities[:, 0]`` stays segmentation). The BVH-rebuild patch builds
  new meshes from ``old.velocities``, so the index survives resets.
* reset notification: ``WarpSensor.reset_idx`` only re-samples the mount; the event camera
  also has to re-seed L_ref, or the teleport to the new spawn fires a burst of events.

Output: ``global_tensor_dict["event_frames"]``, (num_envs, num_frames, H, W) int8 DIFF3D,
oldest frame first. ``depth_range_pixels`` receives metric depth when write_depth is on.

Other sensor types go through upstream unchanged.
"""
import warp as wp

import aerial_gym.robots.robot_manager as robot_manager_module
from aerial_gym.env_manager.env_manager import EnvManager
from aerial_gym.sensors.base_sensor import BaseSensor
from aerial_gym.sensors.warp.warp_sensor import WarpSensor
from aerial_gym.utils.logging import CustomLogger

from sensors.event_camera import WarpEventCam

logger = CustomLogger("warp_event_sensor")

EVENT_TYPE = "event_camera"
_APPLIED = False


class EventAwareWarpSensor(WarpSensor):
    def __init__(self, sensor_config, num_envs, mesh_id_list, device):
        if sensor_config.sensor_type != EVENT_TYPE:
            super().__init__(sensor_config, num_envs, mesh_id_list, device)
            return
        BaseSensor.__init__(self, sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.mesh_id_list = mesh_id_list
        self.num_sensors = self.cfg.num_sensors
        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)
        self.sensor = WarpEventCam(num_envs, self.cfg, self.mesh_ids_array, device)
        logger.info("Event camera sensor initialized")

    def init_tensors(self, global_tensor_dict):
        if self.cfg.sensor_type != EVENT_TYPE:
            return super().init_tensors(global_tensor_dict)
        # BaseSensor only binds pixel buffers for the types it knows
        self.pixels = global_tensor_dict.get("depth_range_pixels")
        self.segmentation_pixels = None
        _write_asset_indices(global_tensor_dict, self.cfg.max_assets_per_env)
        dt_ns = int(round(global_tensor_dict["dt"] * 1e9))
        global_tensor_dict.setdefault("physics_step_count", 0)
        self.sensor.clock = lambda: global_tensor_dict["physics_step_count"] * dt_ns
        super().init_tensors(global_tensor_dict)
        global_tensor_dict["event_frames"] = self.sensor.frames

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.cfg.sensor_type == EVENT_TYPE:
            self.sensor.reset_idx(env_ids)


def _write_asset_indices(global_tensor_dict, max_assets_per_env):
    meshes = global_tensor_dict["CONST_WARP_MESH_PER_ENV"]
    vertex_to_asset = global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"]
    offset = 0
    for mesh in meshes:
        vel = wp.to_torch(mesh.velocities)
        g = vertex_to_asset[offset : offset + vel.shape[0]]
        local = g - g.min()
        if int(local.max()) >= max_assets_per_env:
            raise ValueError(
                f"env has asset index {int(local.max())} >= max_assets_per_env={max_assets_per_env}"
            )
        vel[:, 1] = local.to(vel.dtype)
        offset += vel.shape[0]
    if offset != vertex_to_asset.shape[0]:
        raise RuntimeError(f"mesh vertices ({offset}) != vertex->asset map ({vertex_to_asset.shape[0]})")


_upstream_simulate = EnvManager.simulate


def _counting_simulate(self, actions, env_actions):
    self.global_tensor_dict["physics_step_count"] = (
        self.global_tensor_dict.get("physics_step_count", 0) + 1
    )
    return _upstream_simulate(self, actions, env_actions)


def apply():
    """Install the patch. Idempotent; must run before the sim is built."""
    global _APPLIED
    if _APPLIED:
        return
    robot_manager_module.WarpSensor = EventAwareWarpSensor
    EnvManager.simulate = _counting_simulate
    _APPLIED = True
    logger.warning(
        "Patched warp sensors: sensor_type='event_camera' available, EnvManager.simulate "
        "counts physics steps (see env_manager/warp_event_sensor_patch.py)."
    )
