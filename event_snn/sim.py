"""Isaac Gym rollout for the event-SNN experiment: the depth teacher flies (plus perturbations),
the event camera watches, and every 10 ms step yields inputs and targets (see event_snn.data).

Timing: 1 physics substep of 10 ms per env step; the teacher acts every `teacher_every` steps
and its action is held in between. The teacher sees D435 depth resampled from the event
camera's own z-depth (same pose, pinhole -> pinhole, nearest), encoded by its frozen VAE.

CLI:
    python -m event_snn.sim --check_teacher [--seconds 60]     teacher-only flight report
    python -m event_snn.sim --heldout                          write the held-out set
    python -m event_snn.sim --record runs/event_snn/train_s1000.pt --envs 16 --seconds 120 --seed 1000
                                                               record a training set (offline tests)
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
from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse

from analysis.crash_cause_eval import _NORM_CLAMP, _NORM_EPS, build_actor
from config.robot_config.f450_event_cam_config import F450EventCamConfig
from config.sensor_config.genx320_event_cam_config import GenX320EventCamConfig
from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig
from config.task_config import F450NavTaskConfig
from task.attitude_navigation_task import NavigationWithObstaclesTask

from .config import heldout_path, load_config, steps
from .data import (AttitudeNoise, EventMaskTracker, camera_velocity, events_to_input, inv_ttc_target,
                   inverse_depth_target, state_input)

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
    if getattr(cfg.sim, "curriculum_free", False):
        # end-to-end RL (event_snn.e2e_ppo): the task's own curriculum, from its first level upward
        curriculum = F450NavTaskConfig.curriculum
    else:
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

    KEYS = ("events", "imu", "grav", "grav_true", "rho", "ttc", "near", "vel", "m_now", "m_recent", "done",
            "yaw_rate", "mode", "age", "action", "tobs")

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
        self.rays = torch.as_tensor(self.cam.ray_lut_np, device=self.dev)  # (H, W, 3) optical frame
        self.mount = [float(x) for x in cam_cfg.min_translation]

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
        self.teacher_obs = torch.zeros((n, self.teacher.obs_dim), device=self.dev)
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
        self.teacher_obs = obs

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
        v_cam = camera_velocity(self.obs_dict["robot_body_linvel"], self.obs_dict["robot_body_angvel"],
                                self.mount)
        ttc = inv_ttc_target(self.cam.depth, self.rays, v_cam, d.depth_grid, self.far_m,
                             d.ttc_drone_radius_m, d.ttc_max, pool=d.ttc_pool)
        near = inv_ttc_target(self.cam.depth, self.rays, v_cam, d.depth_grid, self.far_m,
                              d.ttc_drone_radius_m, d.ttc_max, pool=d.ttc_pool, mode="speed")
        m_now, m_recent = self.masks.update(events, done)
        g = quat_rotate_inverse(self.obs_dict["robot_orientation"], self.obs_dict["gravity"])
        g_true = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        vel = self.obs_dict["robot_body_linvel"].clone()
        c = rho.shape[-1]
        self.centre_rho = rho[:, c * 3 // 8: c * 5 // 8, c * 3 // 8: c * 5 // 8].flatten(1).max(1).values
        return dict(events=events, imu=self.imu.imu_meas.clone(), grav=self.attitude(g_true, done),
                    grav_true=g_true, rho=rho, ttc=ttc, near=near, vel=vel, m_now=m_now, m_recent=m_recent, done=done,
                    # what the teacher last saw (17 state + 32 depth latent) and commanded: probe targets
                    action=self.action.clone(), tobs=self.teacher_obs.clone(),
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
    out["ttc"] = out["ttc"].half()
    out["near"] = out["near"].half()
    k = ro.rays.shape[0] // d.depth_grid
    rays_grid = F.normalize(F.avg_pool2d(ro.rays.permute(2, 0, 1)[None], k)[0].permute(1, 2, 0), dim=-1)
    out["meta"] = dict(level=cfg.sim.curriculum_level, seed=d.heldout_seed, dt=0.01,
                       attitude=vars(cfg.attitude), far_m=ro.far_m,
                       geom=dict(rays_grid=rays_grid.cpu(), drone_radius_m=d.ttc_drone_radius_m,
                                 ttc_max=d.ttc_max, far_m=ro.far_m, mount=ro.mount))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(out, path)
    print(f"wrote held-out set {path}: {E} envs x {T} steps")
    return path


RECORD_KEYS = ("events", "imu", "grav", "ttc", "near", "vel", "done")


def record_dataset(cfg, path, num_envs, seconds, seed, chunk_s=5.0):
    """Training flights for the offline tests (event_snn.overfit --train_data): num_envs drones
    for `seconds`, collected in chunks and moved to CPU as they come, so a long recording fits in
    GPU memory. Maps are stored as float16; inverse depth is not stored (no depth head is trained).
    About 12 KB per drone-step: 16 drones x 120 s = 2.4 GB."""
    torch.manual_seed(seed)
    ro = Rollout(cfg, num_envs=num_envs, seed=seed)
    T, C = steps(seconds), steps(chunk_s)
    parts, t0, done_steps = [], time.time(), 0
    while done_steps < T:
        n = min(C, T - done_steps)
        ch = ro.collect(n, keys=RECORD_KEYS)
        part = {k: v.cpu() for k, v in ch.items()}
        for k in ("ttc", "near"):
            part[k] = part[k].half()
        parts.append(part)
        done_steps += n
        rate = done_steps * num_envs / (time.time() - t0)
        print(f"  {done_steps / 100:6.1f} / {T / 100:.0f} s  ({rate:.0f} drone-steps/s, "
              f"{(T - done_steps) * num_envs / rate / 60:.1f} min left)", flush=True)
    out = {k: torch.cat([q[k] for q in parts]) for k in RECORD_KEYS}
    out["meta"] = dict(level=cfg.sim.curriculum_level, seed=seed, dt=0.01, kind="train", far_m=ro.far_m)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(out, path)
    gb = sum(v.numel() * v.element_size() for k, v in out.items() if k != "meta") / 1e9
    print(f"wrote training set {path}: {num_envs} drones x {T} steps, {int(out['done'].sum())} episode starts, "
          f"{gb:.2f} GB")
    return path


def check_teacher(cfg, seconds):
    cfg.perturb.rate_hz = 0.0
    ro = Rollout(cfg)
    print(f"D435 view covered by the event camera: {100 * ro.remap.coverage:.1f}% of pixels")
    t0 = time.time()
    T = steps(seconds)
    speed, ev = [], []
    # invTTC self-check: the world point the centre ray hits at step k is P = c + R d r. Its
    # distance from the camera one step later shrinks at the closing speed, which must equal the
    # analytic v.d built from body velocity, body rates and the mount offset. Tests the frames.
    cy, cx = ro.cam.height // 2, ro.cam.width // 2
    ray_c = ro.rays[cy, cx]
    prev, pairs, ttc_hist = None, [], torch.zeros(4)
    for k in range(T):
        s = ro.step()
        pos, quat = ro.cam.pos_curr.clone(), ro.cam.quat_curr.clone()
        r = ro.cam.depth[:, cy, cx] / ray_c[2]
        v_cam = camera_velocity(ro.obs_dict["robot_body_linvel"], ro.obs_dict["robot_body_angvel"], ro.mount)
        if prev is not None:
            P, r0, vd0 = prev
            ok = (~s["done"]) & (r0 < 0.9 * ro.far_m)
            pairs.append(torch.stack([vd0[ok], ((r0 - (P - pos).norm(dim=-1)) / 0.01)[ok]], 1).cpu())
        prev = (pos + quat_rotate(quat, ray_c.expand_as(pos)) * r[:, None], r.clone(), v_cam @ ray_c)
        t = s["ttc"].flatten()
        ttc_hist += torch.tensor([float((t > x).float().mean()) for x in (0.0, 0.5, 1.0, 0.999 * cfg.data.ttc_max)])
        speed.append(float(s["vel"].norm(dim=-1).mean()))
        ev.append(float((s["events"].float().sum((1, 2, 3))).mean()))
    n = ro.outcomes
    tot = max(sum(n.values()), 1)
    print(f"{T} steps x {ro.n} envs in {time.time() - t0:.1f} s ({T * ro.n / (time.time() - t0):.0f} env-steps/s)")
    print(f"episodes {tot}: arrive {100 * n['arrive'] / tot:.1f}%  crash {100 * n['crash'] / tot:.1f}%  "
          f"timeout {100 * n['timeout'] / tot:.1f}%")
    print(f"mean speed {sum(speed) / T:.2f} m/s, events/step/env {sum(ev) / T:.0f}")
    h = (ttc_hist / T * 100).tolist()
    print(f"invTTC cells: {h[0]:.1f}% > 0, {h[1]:.1f}% > 0.5/s, {h[2]:.1f}% > 1/s (danger), {h[3]:.2f}% clipped")
    p = torch.cat(pairs) if pairs else torch.zeros((0, 2))
    if len(p) > 100:
        slope = float((p[:, 0] * p[:, 1]).sum() / (p[:, 0] ** 2).sum().clamp(min=1e-9))
        corr = float(torch.corrcoef(p.T)[0, 1])
        print(f"invTTC check, centre ray, {len(p)} samples: measured closing speed = {slope:.3f} x (v.d), "
              f"corr {corr:.3f} (both ~1.0 when frames and mount offset are right)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. sim.num_envs=32")
    p.add_argument("--check_teacher", action="store_true")
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--seconds", type=float, default=30.0, help="check_teacher / record length")
    p.add_argument("--record", default=None, metavar="PATH", help="record a training set to PATH")
    p.add_argument("--envs", type=int, default=16, help="record: drones")
    p.add_argument("--seed", type=int, default=1000, help="record: seed (use a new one per file; held-out is 12345)")
    a = p.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    if a.check_teacher:
        check_teacher(cfg, a.seconds)
    elif a.heldout:
        write_heldout(cfg)
    elif a.record:
        if a.seed == cfg.data.heldout_seed:
            p.error("--seed equals the held-out seed: the training flights would be the validation flights")
        record_dataset(cfg, a.record, a.envs, a.seconds, a.seed)
    else:
        p.error("pick --check_teacher, --heldout or --record PATH")


if __name__ == "__main__":
    main()
