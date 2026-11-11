"""Sensor preprocessing, targets, masks and the attitude-estimator noise model (plain torch).

Conventions shared by rollout, training and evaluation, one 10 ms step at a time:
  events    (N, 2, 64, 64) uint8   ON/OFF counts, sum-pooled 2x from 128^2, clipped
  imu       (N, 6)         float   accel (specific force) + gyro, body frame
  grav      (N, 3)         float   gravity direction in body frame from the noisy estimator
  rho       (N, 32, 32)    float   inverse depth 1/Z, 0 for sky / beyond the far plane
  vel       (N, 3)         float   body-frame linear velocity
  ttc       (N, 32, 32)    float   inverse time-to-contact [1/s] along each cell's rays, 0 = not closing
  near      (N, 32, 32)    float   speed / range [1/s], same clip; 0 only at rest or for sky
  m_now     (N, 32, 32)    bool    >= 1 event in the cell this step
  m_recent  (N, 32, 32)    bool    events in the cell within recent_s, not this step
  done      (N,)           bool    this step is the first of a new episode: reset state first
"""
import math

import torch
import torch.nn.functional as F

REGIMES = ("hover", "cruise", "turn", "approach")


def events_to_input(frames, clip=7, pool=2):
    """(N, H, W) signed DIFF3D counts -> (N, 2, H/pool, W/pool) uint8 [ON, OFF], sum-pooled, clipped."""
    f = frames.to(torch.float32)
    x = torch.stack([f.clamp(min=0.0), (-f).clamp(min=0.0)], dim=1)
    if pool > 1:
        x = F.avg_pool2d(x, pool) * (pool * pool)
    return x.clamp(max=clip).round().to(torch.uint8)


def inverse_depth_target(depth, grid=32, far_m=20.0, min_m=0.1, pool="mean"):
    """(N, H, W) metric z-depth -> (N, grid, grid) inverse depth; misses (>= far_m) -> 0."""
    d = depth.to(torch.float32)
    rho = torch.where(d >= far_m * 0.999, torch.zeros_like(d), 1.0 / d.clamp(min=min_m))
    k = d.shape[-1] // grid
    if d.shape[-1] != k * grid or d.shape[-2] != k * grid:
        raise ValueError(f"depth {tuple(d.shape[-2:])} is not a multiple of grid {grid}")
    rho = rho.unsqueeze(1)
    rho = F.avg_pool2d(rho, k) if pool == "mean" else F.max_pool2d(rho, k)
    return rho.squeeze(1)


def body_to_optical(v):
    """Body (x forward, y left, z up) -> camera optical (x right, y down, z forward), forward mount."""
    return torch.stack([-v[..., 1], -v[..., 2], v[..., 0]], dim=-1)


def camera_velocity(v_body, w_body, mount_xyz):
    """Velocity of the camera centre in the optical frame: v + w x r_mount (static scene, so this
    is also the velocity relative to every surface)."""
    r = torch.as_tensor(mount_xyz, dtype=v_body.dtype, device=v_body.device).expand_as(v_body)
    return body_to_optical(v_body + torch.cross(w_body, r, dim=-1))


def inv_ttc_target(depth, rays, v_cam, grid=32, far_m=20.0, drone_radius_m=0.3, ttc_max=10.0,
                   min_gap_m=0.05, pool="max", mode="radial"):
    """Inverse time-to-contact per ray, clip(max(0, v.d) / max(r - r_drone, min_gap), 0, ttc_max).

    mode="speed" replaces the closing speed v.d by the speed |v|: speed / range, the scale-free
    nearness that translational image motion measures in every direction (sideways included).

    depth (N, H, W) metric z-depth; rays (H, W, 3) unit rays in the optical frame; v_cam (N, 3)
    camera velocity in the optical frame. Range r = z / d_z. Misses (>= far_m) -> 0. Pooled to
    (N, grid, grid): max keeps the most imminent surface of each cell, mean averages it away.
    """
    d = depth.to(torch.float32)
    rays = rays.to(d)
    r = d / rays[..., 2].clamp(min=1e-6)
    if mode == "speed":
        closing = v_cam.to(d).norm(dim=-1)[:, None, None].expand_as(r)
    else:
        closing = torch.einsum("nc,hwc->nhw", v_cam.to(d), rays).clamp(min=0.0)
    ttc = (closing / (r - drone_radius_m).clamp(min=min_gap_m)).clamp(max=ttc_max)
    ttc = torch.where(d >= far_m * 0.999, torch.zeros_like(ttc), ttc)
    k = d.shape[-1] // grid
    if d.shape[-1] != k * grid or d.shape[-2] != k * grid:
        raise ValueError(f"depth {tuple(d.shape[-2:])} is not a multiple of grid {grid}")
    ttc = ttc.unsqueeze(1)
    ttc = F.max_pool2d(ttc, k) if pool == "max" else F.avg_pool2d(ttc, k)
    return ttc.squeeze(1)


def select_target(chunk, name):
    """The map the ttc head is trained on: 'ttc' (radial) or 'near' (speed / range). Returns the
    chunk with that map under "ttc", so loss and evaluation need not know which one it is."""
    if name == "ttc":
        return chunk
    if name not in chunk:
        raise SystemExit(f"data has no {name!r} map: rerun `python -m event_snn.sim --heldout`")
    out = dict(chunk)
    out["ttc"] = chunk[name]
    return out


def log_rho(rho, eps):
    return torch.log(rho + eps)


def cell_event_counts(events, grid=32):
    """(N, 2, h, w) counts -> (N, grid, grid) total events per target cell."""
    tot = events.to(torch.float32).sum(1, keepdim=True)
    k = tot.shape[-1] // grid
    return (F.avg_pool2d(tot, k) * k * k).squeeze(1)


class EventMaskTracker:
    """Steps since the last event per target cell -> 'now' and 'recent' masks."""

    NEVER = 30000

    def __init__(self, num_envs, grid, recent_steps, device):
        self.recent_steps = recent_steps
        self.since = torch.full((num_envs, grid, grid), self.NEVER, dtype=torch.int32, device=device)

    def update(self, events, done):
        self.since[done] = self.NEVER
        active = cell_event_counts(events, self.since.shape[-1]) >= 1
        self.since = torch.where(active, torch.zeros_like(self.since),
                                 (self.since + 1).clamp(max=self.NEVER))
        now = self.since == 0
        recent = (self.since > 0) & (self.since < self.recent_steps)
        return now, recent


def rotate(v, rotvec):
    """Rodrigues: rotate rows of v (N, 3) by rotation vectors (N, 3) [rad]."""
    theta = rotvec.norm(dim=-1, keepdim=True)
    k = rotvec / theta.clamp(min=1e-12)
    c, s = torch.cos(theta), torch.sin(theta)
    return v * c + torch.cross(k, v, dim=-1) * s + k * (k * v).sum(-1, keepdim=True) * (1.0 - c)


class AttitudeNoise:
    """Onboard attitude estimate: true gravity direction rotated by bias (per-episode init +
    random walk) + white noise. Degrees are total tilt RMS, split over two axes."""

    def __init__(self, num_envs, noise_deg, bias_init_deg, bias_rw_deg, dt=0.01, device="cpu",
                 generator=None):
        self.n, self.dt, self.device, self.gen = num_envs, dt, device, generator
        r = math.radians
        self.noise, self.bias_init, self.bias_rw = r(noise_deg), r(bias_init_deg), r(bias_rw_deg)
        self.bias = torch.zeros((num_envs, 3), device=device)
        self.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

    def _randn(self, n):
        return torch.randn((n, 3), device=self.device, generator=self.gen) / math.sqrt(2.0)

    def reset(self, mask):
        n = int(mask.sum())
        if n:
            self.bias[mask] = self._randn(n) * self.bias_init

    def __call__(self, g_true, done):
        self.reset(done)
        self.bias += self._randn(self.n) * self.bias_rw * math.sqrt(self.dt)
        rv = self.bias + self._randn(self.n) * self.noise
        g = rotate(g_true, rv)
        return g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def state_input(chunk):
    """(T, N, 9) [acc, gyro, grav-estimate] as the SNN's state encoder sees it."""
    return torch.cat([chunk["imu"].float(), chunk["grav"].float()], dim=-1)



def regime_labels(vel, yaw_rate, rho, hover_speed=0.3, turn_rate=0.5, approach_depth_m=3.0,
                  approach_speed=0.5):
    """(N,) int8 index into REGIMES. Approach: flying forward with something close ahead."""
    speed = vel.norm(dim=-1)
    g = rho.shape[-1]
    c = rho[:, g * 3 // 8: g * 5 // 8, g * 3 // 8: g * 5 // 8]
    near = c.flatten(1).max(1).values > 1.0 / approach_depth_m
    lab = torch.full(speed.shape, REGIMES.index("cruise"), dtype=torch.int8, device=vel.device)
    lab[near & (vel[:, 0] > approach_speed)] = REGIMES.index("approach")
    lab[yaw_rate.abs() > turn_rate] = REGIMES.index("turn")
    lab[speed < hover_speed] = REGIMES.index("hover")
    return lab
