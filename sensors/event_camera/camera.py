"""WarpEventCam: event camera over aerial_gym's per-env warp meshes, DIFF3D output.

Sample instants sit on a global clock at multiples of frame_period / oversample, so the
output does not depend on how many physics substeps the env step happened to take. Every
`oversample`-th sample closes a DIFF3D frame into a ring of `num_frames` slots.

The camera pose between two captures is interpolated (lerp + slerp); obstacles are
static within an episode, so all events come from ego-motion.
"""
import math

import torch
import warp as wp

from .kernels import event_kernel
from .lens import intrinsics_from_fov, ray_lut


class WarpEventCam:
    def __init__(self, num_envs, config, mesh_ids_array, device="cuda:0"):
        cfg = config
        if cfg.num_sensors != 1:
            raise ValueError("WarpEventCam supports exactly one sensor per env")
        if cfg.diff3d_overflow not in ("wrap", "saturate"):
            raise ValueError(f"diff3d_overflow must be 'wrap' or 'saturate', got {cfg.diff3d_overflow!r}")
        if not 2 <= cfg.diff3d_bits <= 8:
            raise ValueError(f"diff3d_bits must be in [2, 8], got {cfg.diff3d_bits}")
        if mesh_ids_array.shape[0] != num_envs:
            raise ValueError(f"{mesh_ids_array.shape[0]} mesh ids for {num_envs} envs")

        self.cfg = cfg
        self.num_envs = num_envs
        self.mesh_ids_array = mesh_ids_array
        self.device = device
        self.width = cfg.width
        self.height = cfg.height
        self.num_frames = cfg.num_frames
        self.oversample = cfg.oversample

        self.frame_ns = int(round(cfg.frame_period_s * 1e9))
        if self.frame_ns % self.oversample:
            raise ValueError(
                f"frame period {self.frame_ns} ns is not divisible by oversample={self.oversample}"
            )
        self.sample_ns = self.frame_ns // self.oversample

        if cfg.intrinsics is not None:
            self.fx, self.fy, self.cx, self.cy = cfg.intrinsics
        else:
            self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(
                cfg.width, cfg.height, cfg.fov_deg, cfg.fov_type, cfg.lens_model
            )
        lut = ray_lut(cfg.width, cfg.height, self.fx, self.fy, self.cx, self.cy,
                      cfg.lens_model, cfg.kb4_coeffs)
        self.ray_lut_np = lut
        self.ray_lut = wp.array(lut, dtype=wp.vec3, device=device)

        N, H, W = num_envs, cfg.height, cfg.width
        z = dict(device=device, requires_grad=False)
        self.l_ref = torch.zeros((N, H, W), dtype=torch.float32, **z)
        self.acc = torch.zeros((N, H, W), dtype=torch.int32, **z)
        self.ring = torch.zeros((N, self.num_frames, H, W), dtype=torch.int8, **z)
        self.frames = torch.zeros_like(self.ring)  # oldest -> newest, refreshed every capture
        B = self.num_frames
        self._orders = torch.tensor(
            [[(r - B + 1 + b) % B for b in range(B)] for r in range(B)], device=device
        )
        self.depth = torch.zeros((N, H, W), dtype=torch.float32, **z)

        self.albedo = torch.zeros((N, cfg.max_assets_per_env), dtype=torch.float32, **z)
        self.light = torch.zeros((N, 3), dtype=torch.float32, **z)
        self.ambient = torch.zeros(N, dtype=torch.float32, **z)
        self.diffuse = torch.zeros(N, dtype=torch.float32, **z)
        self.sky = torch.zeros(N, dtype=torch.float32, **z)
        self.c_on_mean = torch.zeros(N, dtype=torch.float32, **z)
        self.c_off_mean = torch.zeros(N, dtype=torch.float32, **z)
        self.ba_rate = torch.zeros(N, dtype=torch.float32, **z)
        self.env_seed = torch.zeros(N, dtype=torch.int32, **z)
        self.reinit = torch.ones(N, dtype=torch.int32, **z)

        self.pos_prev = torch.zeros((N, 3), dtype=torch.float32, **z)
        self.quat_prev = torch.zeros((N, 4), dtype=torch.float32, **z)
        self.quat_prev[:, 3] = 1.0
        self.pos_curr = None
        self.quat_curr = None

        self.t_ns = None  # time of the last capture on the global clock
        self.clock = None  # callable -> int ns; used when capture() gets no time
        self._step = 0
        self._cmd = None  # recorded launch; rebuilt whenever an array binding changes

        self._wp = {
            name: wp.from_torch(getattr(self, name), dtype=dt)
            for name, dt in (
                ("l_ref", wp.float32), ("acc", wp.int32), ("ring", wp.int8), ("depth", wp.float32),
                ("albedo", wp.float32), ("light", wp.vec3), ("ambient", wp.float32),
                ("diffuse", wp.float32), ("sky", wp.float32), ("c_on_mean", wp.float32),
                ("c_off_mean", wp.float32), ("ba_rate", wp.float32), ("env_seed", wp.int32),
                ("reinit", wp.int32), ("pos_prev", wp.vec3), ("quat_prev", wp.quat),
            )
        }
        self.reset_idx(torch.arange(N, device=device))

    # --- aerial_gym WarpSensor interface -------------------------------------------------

    def set_pose_tensor(self, positions, orientations):
        """positions (N, 1, 3), orientations (N, 1, 4) xyzw, world frame, updated in place."""
        self.pos_curr = positions.view(self.num_envs, 3)
        self.quat_curr = orientations.view(self.num_envs, 4)
        self._wp["pos_curr"] = wp.from_torch(self.pos_curr, dtype=wp.vec3)
        self._wp["quat_curr"] = wp.from_torch(self.quat_curr, dtype=wp.quat)
        self._cmd = None

    def set_image_tensors(self, pixels, segmentation_pixels=None):
        """Optional (N, 1, H, W) float buffer that receives metric depth when write_depth is on."""
        if pixels is not None:
            self.depth = pixels.view(self.num_envs, self.height, self.width)
            self._wp["depth"] = wp.from_torch(self.depth, dtype=wp.float32)
            self._cmd = None

    def reset_idx(self, env_ids):
        """New episode: resample scene/sensor parameters and re-seed L_ref at the next capture."""
        cfg = self.cfg
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)
        if n == 0:
            return

        def uniform(lo_hi, shape=(n,)):
            lo, hi = lo_hi
            return lo + (hi - lo) * torch.rand(shape, device=self.device)

        az = uniform((0.0, 2.0 * math.pi))
        el = torch.deg2rad(uniform(cfg.light_elevation_deg_range))
        self.light[env_ids] = torch.stack(
            [torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], dim=-1
        )
        self.ambient[env_ids] = uniform(cfg.ambient_range)
        self.diffuse[env_ids] = uniform(cfg.diffuse_range)
        self.sky[env_ids] = uniform(cfg.sky_intensity_range)
        self.albedo[env_ids] = uniform(cfg.albedo_range, (n, cfg.max_assets_per_env))
        self.c_on_mean[env_ids] = uniform(cfg.c_on_mean_range)
        self.c_off_mean[env_ids] = uniform(cfg.c_off_mean_range)
        self.ba_rate[env_ids] = uniform(cfg.ba_rate_hz_range)
        self.env_seed[env_ids] = torch.randint(
            0, 2**30, (n,), device=self.device, dtype=torch.int32
        )
        self.reinit[env_ids] = 1

    def capture(self, t_now_ns=None):
        """Advance the sensor to t_now_ns and return the last num_frames DIFF3D frames.

        Returns (N, num_frames, H, W) int8, oldest first; frame j covers ((j-1)T, jT].
        """
        if self.pos_curr is None:
            raise RuntimeError("set_pose_tensor() must be called before capture()")
        if t_now_ns is None:
            if self.clock is None:
                raise RuntimeError("capture() needs t_now_ns or a clock")
            t_now_ns = self.clock()
        t_now_ns = int(t_now_ns)
        if self.t_ns is None:
            self.t_ns = t_now_ns
        dt = t_now_ns - self.t_ns
        if dt < 0:
            raise ValueError(f"clock went backwards: {self.t_ns} -> {t_now_ns} ns")

        s_first = self.t_ns // self.sample_ns + 1
        s_last = t_now_ns // self.sample_ns
        n_samples = max(0, s_last - s_first + 1) if dt > 0 else 0
        alpha0 = (s_first * self.sample_ns - self.t_ns) / dt if n_samples else 0.0
        dalpha = self.sample_ns / dt if n_samples else 0.0

        if self._cmd is None:
            self._cmd = self._record_launch()
        self._step += 1
        cmd = self._cmd
        cmd.set_param_by_name("n_samples", int(n_samples))
        cmd.set_param_by_name("alpha0", float(alpha0))
        cmd.set_param_by_name("dalpha", float(dalpha))
        cmd.set_param_by_name("s0", int(s_first % (self.oversample * self.num_frames)))
        cmd.set_param_by_name("step_seed", int((self._step * 2654435761) & 0x3FFFFFFF))
        cmd.launch(stream=wp.stream_from_torch(self.device))

        self.pos_prev.copy_(self.pos_curr)
        self.quat_prev.copy_(self.quat_curr)
        self.reinit.zero_()
        self.t_ns = t_now_ns

        latest = t_now_ns // self.frame_ns
        torch.index_select(self.ring, 1, self._orders[latest % self.num_frames], out=self.frames)
        return self.frames

    def _record_launch(self):
        """Pack the ~40 kernel arguments once; capture() only updates the per-step scalars."""
        cfg, w = self.cfg, self._wp
        return wp.launch(
            event_kernel,
            dim=(self.num_envs, self.height, self.width),
            inputs=[
                self.mesh_ids_array, w["pos_prev"], w["quat_prev"], w["pos_curr"], w["quat_curr"],
                self.ray_lut, float(cfg.max_range), w["albedo"], w["light"], w["ambient"],
                w["diffuse"], w["sky"], w["c_on_mean"], w["c_off_mean"], w["ba_rate"],
                w["env_seed"], w["reinit"], float(cfg.c_sigma_pixel), float(cfg.c_sigma_temporal),
                float(cfg.c_min), float(cfg.c_max), float(cfg.log_eps), 0, 0.0, 0.0, 0,
                int(self.oversample), int(self.num_frames), self.sample_ns * 1e-9, 0,
                int(1 << (cfg.diff3d_bits - 1)), int(cfg.diff3d_overflow == "wrap"),
                w["l_ref"], w["acc"], w["ring"], w["depth"], int(cfg.write_depth),
            ],
            device=self.device,
            record_cmd=True,
        )
