"""Isaac Gym rollout for the event-SNN experiment: the depth teacher flies (plus perturbations),
the event camera watches, and every 10 ms step yields inputs and targets (see event_snn.data).

Timing: 1 physics substep of 10 ms per env step; the teacher acts every `teacher_every` steps
and its action is held in between. The teacher sees D435 depth resampled from the event
camera's own z-depth (same pose, pinhole -> pinhole, nearest), encoded by its frozen VAE.

CLI:
    python -m event_snn.sim --check_teacher [--seconds 60]     teacher-only flight report
    python -m event_snn.sim --heldout                          write the held-out set
"""
import isaacgym  # noqa: F401  -- must precede torch

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

import config  # noqa: F401  -- registers env/robot/task and the event-camera patch
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.robots import BaseMultirotor
from aerial_gym.utils.math import quat_rotate_inverse

from analysis.crash_cause_eval import _NORM_CLAMP, _NORM_EPS, build_actor
from config.robot_config.f450_event_cam_config import F450EventCamConfig
from config.sensor_config.genx320_event_cam_config import GenX320EventCamConfig
from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig
from config.task_config import F450NavTaskConfig
from task.attitude_navigation_task import NavigationWithObstaclesTask

from .config import heldout_path, load_config, steps
from .data import AttitudeNoise, EventMaskTracker, events_to_input, inverse_depth_target, state_input

ROBOT = "f450_event_cam_snn"
TASK = "f450_event_snn_task"
MODES = ("teacher", "yaw_burst", "speed", "hover", "approach", "excursion")


class EventSNNNavTask(NavigationWithObstaclesTask):
    def process_image_observation(self):
        pass  # the rollout encodes remapped depth itself, only on steps where the teacher acts


def register(cfg):
    c = cfg.sim.camera
    half_diag = math.radians(GenX320EventCamConfig.fov_deg) / 2.0
    hfov = 2.0 * math.degrees(math.atan(math.tan(half_diag) * c.width / math.hypot(c.width, c.height)))

    cam = type("EventSNNCamConfig", (GenX320EventCamConfig,), dict(
        width=c.width, height=c.height, write_depth=True, num_frames=1, oversample=1,
        frame_period_s=0.010, texture_enabled=bool(c.texture_enabled), supersample=int(c.supersample),
        horizontal_fov_deg=hfov,  # read by the nav task's p_fov reward only
    ))
    sensors = type("sensor_config", (F450EventCamConfig.sensor_config,), dict(camera_config=cam))
    robot = type("F450EventSNNConfig", (F450EventCamConfig,), dict(sensor_config=sensors))
    robot_registry.register(ROBOT, BaseMultirotor, robot)

    env_cfg = env_config_registry.get_env_config(F450NavTaskConfig.env_name)
    env_cfg.env.num_physics_steps_per_env_step_mean = 1
    env_cfg.env.num_physics_steps_per_env_step_std = 0

    level = int(cfg.sim.curriculum_level)
    curriculum = type("curriculum", (F450NavTaskConfig.curriculum,), dict(min_level=level, max_level=level))
    task = type("EventSNNTaskConfig", (F450NavTaskConfig,), dict(
        robot_name=ROBOT, episode_len_steps=steps(cfg.sim.episode_seconds), curriculum=curriculum,
    ))
    task_registry.register_task(TASK, EventSNNNavTask, task)
    return cam


class D435Remap:
    """Event-camera z-depth (pinhole, OpenCV centres) -> D435 normalised depth as aerial_gym renders it."""

    def __init__(self, cam, device):
        d = RealSenseD435CamConfig
        W, H = d.width, d.height
        f = W / 2.0 / math.tan(math.radians(d.horizontal_fov_deg) / 2.0)  # upstream warp_cam: u0 = W/2
        v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
        x, y = (u - W / 2.0) / f, (v - H / 2.0) / f
        ue, ve = cam.fx * x + cam.cx, cam.fy * y + cam.cy
        gx = ue / (cam.width - 1) * 2.0 - 1.0
        gy = ve / (cam.height - 1) * 2.0 - 1.0
        self.grid = torch.stack([gx, gy], -1)[None].to(device)
        self.coverage = float(((gx.abs() <= 1) & (gy.abs() <= 1)).float().mean())
        self.max_range, self.min_range = d.max_range, d.min_range

    def __call__(self, depth):
        g = self.grid.expand(depth.shape[0], -1, -1, -1)
        z = F.grid_sample(depth[:, None], g, mode="nearest", padding_mode="border", align_corners=True)[:, 0]
        z = torch.where(z > self.max_range, torch.full_like(z, self.max_range), z)
        z = torch.where(z < self.min_range, torch.full_like(z, -self.max_range), z)
        return z / self.max_range


class Teacher:
    def __init__(self, path, device):
        w = torch.load(path, map_location="cpu")["model"]
        self.actor, _ = build_actor(w)
        self.actor = self.actor.to(device).eval()
        self.mean = w["running_mean_std.running_mean"].float().to(device)
        self.std = torch.sqrt(w["running_mean_std.running_var"].float().to(device) + _NORM_EPS)
        self.obs_dim = self.actor[0].in_features

    @torch.no_grad()
    def __call__(self, obs):
        return self.actor(torch.clamp((obs - self.mean) / self.std, -_NORM_CLAMP, _NORM_CLAMP)).clamp(-1, 1)


class Perturbations:
    """Per-env override of the teacher: yaw bursts, speed scaling, stop-and-hover, straight
    approaches and random velocity excursions. Velocity-hold modes use a P loop to attitude."""

    def __init__(self, cfg, n, device, max_tilt, max_yaw_rate, dt=0.01):
        self.cfg, self.n, self.dev, self.dt = cfg, n, device, dt
        self.max_tilt, self.max_yaw = max_tilt, max_yaw_rate
        self.mode = torch.zeros(n, dtype=torch.long, device=device)
        self.left = torch.zeros(n, dtype=torch.long, device=device)
        self.yaw = torch.zeros(n, device=device)
        self.scale = torch.ones(n, device=device)
        self.v_des = torch.zeros((n, 3), device=device)
        p = cfg.probs
        self.probs = torch.tensor([getattr(p, m) for m in MODES[1:]], device=device)

    def _u(self, lo_hi, k):
        lo, hi = lo_hi
        return lo + (hi - lo) * torch.rand(k, device=self.dev)

    def _start(self, ids):
        k = len(ids)
        mode = torch.multinomial(self.probs, k, replacement=True) + 1
        self.mode[ids] = mode
        c = self.cfg
        dur = {1: c.yaw_burst_s, 2: c.speed_s, 3: c.hover_s, 4: c.approach_s, 5: c.excursion_s}
        secs = torch.zeros(k, device=self.dev)
        for m, rng in dur.items():
            sel = mode == m
            secs[sel] = self._u(rng, int(sel.sum()))
        self.left[ids] = (secs / self.dt).long().clamp(min=1)
        sign = torch.where(torch.rand(k, device=self.dev) < 0.5, -1.0, 1.0)
        self.yaw[ids] = sign * self._u((0.5, 1.0), k)
        self.scale[ids] = self._u(c.speed_scale, k)
        v = torch.zeros((k, 3), device=self.dev)
        v[:, 0] = self._u(c.approach_speed, k)
        exc = mode == 5
        ne = int(exc.sum())
        if ne:
            d = torch.randn((ne, 3), device=self.dev)
            d[:, 2] *= 0.3
            v[exc] = d / d.norm(dim=-1, keepdim=True) * self._u(c.excursion_speed, ne)[:, None]
        v[mode == 3] = 0.0
        self.v_des[ids] = v

    def reset(self, done):
        self.mode[done] = 0
        self.left[done] = 0

    def __call__(self, teacher_action, vehicle_vel, centre_rho):
        c = self.cfg
        free = (self.mode == 0) & (torch.rand(self.n, device=self.dev) < c.rate_hz * self.dt)
        if free.any():
            self._start(free.nonzero().squeeze(1))
        abort = (self.mode == 4) & (centre_rho > 1.0 / c.approach_abort_depth_m)
        self.left[abort] = 0

        a = teacher_action.clone()
        m = self.mode
        a[m == 1, 3] = self.yaw[m == 1]
        sp = m == 2
        a[sp, 1:3] = (a[sp, 1:3] * self.scale[sp, None]).clamp(-1, 1)
        hold = (m == 3) | (m == 4) | (m == 5)
        if hold.any():
            a[hold] = self._velocity_hold(self.v_des[hold], vehicle_vel[hold])

        self.left = (self.left - 1).clamp(min=0)
        self.mode[self.left == 0] = 0
        return a

    def _velocity_hold(self, v_des, v):
        g = 9.81
        acc = (self.cfg.vel_gain * (v_des - v))
        acc[:, :2] = acc[:, :2].clamp(-4.0, 4.0)
        acc[:, 2] = acc[:, 2].clamp(-3.0, 3.0)
        pitch = torch.atan2(acc[:, 0], torch.full_like(acc[:, 0], g)).clamp(-self.max_tilt, self.max_tilt)
        roll = torch.atan2(-acc[:, 1], torch.full_like(acc[:, 1], g)).clamp(-self.max_tilt, self.max_tilt)
        thrust = (g + acc[:, 2]) / (g * torch.cos(roll) * torch.cos(pitch)) - 1.0
        return torch.stack([thrust.clamp(-1, 1), roll / self.max_tilt, pitch / self.max_tilt,
                            torch.zeros_like(thrust)], dim=1)


class Rollout:
    """Flies num_envs drones and returns (T, N, ...) chunks of inputs and targets on the GPU."""

    KEYS = ("events", "imu", "grav", "grav_true", "rho", "vel", "m_now", "m_recent", "done",
            "yaw_rate", "mode", "age")

    def __init__(self, cfg, num_envs=None, seed=None):
        self.cfg = cfg
        self.dev = cfg.device
        n = int(num_envs or cfg.sim.num_envs)
        cam_cfg = register(cfg)
        self.task = task_registry.make_task(TASK, seed=cfg.seed if seed is None else seed, num_envs=n,
                                            headless=True, use_warp=True)
        self.n = n
        self.env = self.task.sim_env
        self.obs_dict = self.task.obs_dict
        self.cam = self.env.robot_manager.warp_sensor.sensor
        self.far_m = float(cam_cfg.max_range)

        self.teacher = Teacher(cfg.sim.teacher_checkpoint, self.dev)
        if self.teacher.obs_dim != self.task.task_config.observation_space_dim:
            raise RuntimeError(f"teacher wants {self.teacher.obs_dim}-D obs, task gives "
                               f"{self.task.task_config.observation_space_dim}-D")
        self.vae = self.task.vae_model
        self.remap = D435Remap(self.cam, self.dev)

        imu = self.env.robot_manager.imu_sensor
        rw = torch.tensor([cfg.imu.accel_bias_rw] * 3 + [cfg.imu.gyro_bias_rw] * 3, device=imu.bias_std.device)
        imu.bias_std = rw.expand_as(imu.bias_std)  # upstream's is an expanded view: replace, don't write
        self.imu = imu

        a = cfg.attitude
        self.attitude = AttitudeNoise(n, a.noise_deg, a.bias_init_deg, a.bias_rw_deg, device=self.dev)
        d = cfg.data
        self.masks = EventMaskTracker(n, d.depth_grid, steps(d.recent_s), self.dev)
        tc = self.task.task_config
        self.perturb = Perturbations(cfg.perturb, n, self.dev, tc.max_inclination_angle_rad, tc.max_yaw_rate)

        self.obs = self.task.reset()[0]["observations"]
        self.action = torch.zeros((n, 4), device=self.dev)
        self.k = 0
        self.age = torch.zeros(n, dtype=torch.long, device=self.dev)
        self.pending_done = torch.ones(n, dtype=torch.bool, device=self.dev)
        self.centre_rho = torch.zeros(n, device=self.dev)
        self.outcomes = {"arrive": 0, "crash": 0, "timeout": 0}

    @torch.no_grad()
    def _teacher_step(self):
        obs = self.obs.clone()
        obs[:, 17:49] = self.vae.encode(self.remap(self.cam.depth))
        self.action = self.teacher(obs)

    @torch.no_grad()
    def step(self):
        if self.k % self.cfg.sim.teacher_every == 0:
            self._teacher_step()
        act = self.perturb(self.action, self.obs_dict["robot_vehicle_linvel"], self.centre_rho)
        o, _, term, trunc, infos = self.task.step(act)
        self.obs = o["observations"]
        self.k += 1

        term, trunc = term.bool(), trunc.bool()
        done = term | trunc
        arrive = infos["arrivals"].bool()
        self.outcomes["arrive"] += int(arrive.sum())
        self.outcomes["crash"] += int((term & ~trunc).sum())
        self.outcomes["timeout"] += int((trunc & ~term).sum())

        # this step's sensor data already belongs to the new episode for envs that just reset
        done = done | self.pending_done
        self.pending_done = torch.zeros_like(done)
        self.perturb.reset(done)
        self.age = torch.where(done, torch.zeros_like(self.age), self.age + 1)

        d = self.cfg.data
        events = events_to_input(self.env.global_tensor_dict["event_frames"][:, -1], d.event_clip)
        rho = inverse_depth_target(self.cam.depth, d.depth_grid, self.far_m, d.min_depth_m, d.depth_pool)
        m_now, m_recent = self.masks.update(events, done)
        g = quat_rotate_inverse(self.obs_dict["robot_orientation"], self.obs_dict["gravity"])
        g_true = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel = self.obs_dict["robot_body_linvel"].clone()
        c = rho.shape[-1]
        self.centre_rho = rho[:, c * 3 // 8: c * 5 // 8, c * 3 // 8: c * 5 // 8].flatten(1).max(1).values
        return dict(events=events, imu=self.imu.imu_meas.clone(), grav=self.attitude(g_true, done),
                    grav_true=g_true, rho=rho, vel=vel, m_now=m_now, m_recent=m_recent, done=done,
                    yaw_rate=self.obs_dict["robot_body_angvel"][:, 2].clone(),
                    mode=self.perturb.mode.to(torch.int8), age=self.age.to(torch.int32))

    def collect(self, T, keys=KEYS):
        out = None
        for t in range(T):
            s = self.step()
            if out is None:
                out = {k: torch.empty((T,) + s[k].shape, dtype=s[k].dtype, device=s[k].device) for k in keys}
            for k in keys:
                out[k][t] = s[k]
        return out


def measure_bounds(rollout, n_steps, percentiles):
    ch = rollout.collect(n_steps, keys=("imu", "grav"))
    x = state_input(ch).flatten(0, 1)
    q = torch.tensor([p / 100.0 for p in percentiles], device=x.device)
    lo, hi = torch.quantile(x, q[0], dim=0), torch.quantile(x, q[1], dim=0)
    return lo.cpu(), hi.cpu()


def write_heldout(cfg, rollout=None):
    """Fixed evaluation flights: the first heldout_envs envs for heldout_seconds, saved on CPU."""
    d = cfg.data
    path = heldout_path(cfg)
    torch.manual_seed(d.heldout_seed)
    ro = rollout or Rollout(cfg, num_envs=max(d.heldout_envs, 1), seed=d.heldout_seed)
    T, E = steps(d.heldout_seconds), d.heldout_envs
    ch = ro.collect(T)
    out = {k: v[:, :E].cpu() for k, v in ch.items()}
    out["rho"] = out["rho"].half()
    out["meta"] = dict(level=cfg.sim.curriculum_level, seed=d.heldout_seed, dt=0.01,
                       attitude=vars(cfg.attitude), far_m=ro.far_m)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(out, path)
    print(f"wrote held-out set {path}: {E} envs x {T} steps")
    return path


def check_teacher(cfg, seconds):
    cfg.perturb.rate_hz = 0.0
    ro = Rollout(cfg)
    print(f"D435 view covered by the event camera: {100 * ro.remap.coverage:.1f}% of pixels")
    t0 = time.time()
    T = steps(seconds)
    speed, ev = [], []
    for k in range(T):
        s = ro.step()
        speed.append(float(s["vel"].norm(dim=-1).mean()))
        ev.append(float((s["events"].float().sum((1, 2, 3))).mean()))
    n = ro.outcomes
    tot = max(sum(n.values()), 1)
    print(f"{T} steps x {ro.n} envs in {time.time() - t0:.1f} s ({T * ro.n / (time.time() - t0):.0f} env-steps/s)")
    print(f"episodes {tot}: arrive {100 * n['arrive'] / tot:.1f}%  crash {100 * n['crash'] / tot:.1f}%  "
          f"timeout {100 * n['timeout'] / tot:.1f}%")
    print(f"mean speed {sum(speed) / T:.2f} m/s, events/step/env {sum(ev) / T:.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. sim.num_envs=32")
    p.add_argument("--check_teacher", action="store_true")
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--seconds", type=float, default=30.0)
    a = p.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    if a.check_teacher:
        check_teacher(cfg, a.seconds)
    elif a.heldout:
        write_heldout(cfg)
    else:
        p.error("pick --check_teacher or --heldout")


if __name__ == "__main__":
    main()
