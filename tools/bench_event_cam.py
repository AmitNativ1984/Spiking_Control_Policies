"""Time the warp event camera per env step against upstream's depth kernel on the same scenes.

Synthetic forests (ground + cylinders/spheres/boxes); env i uses forest i % 8. Each env step
advances the clock 30 ms (the F450 nav task's mean), so a capture renders 30/T*oversample
samples per pixel; the depth baseline renders one.

Usage:
    python -m tools.bench_event_cam
    python -m tools.bench_event_cam --num_envs 64 256 --res 64 160 320 --num_obstacles 60
"""
import isaacgym  # noqa: F401  — must precede torch

import argparse
import math
import time

import numpy as np
import torch
import warp as wp

from aerial_gym.sensors.warp.warp_kernels.warp_camera_kernels import DepthCameraWarpKernels

import config  # noqa: F401
from config.sensor_config.genx320_event_cam_config import GenX320EventCamConfig
from sensors.event_camera import WarpEventCam
from sensors.event_camera.synthetic import SyntheticScene, camera_quat, random_forest

DEV = "cuda:0"
STEP_MS = 30


def poses(n, k):
    t = k * STEP_MS * 1e-3
    pos = torch.zeros((n, 1, 3), device=DEV)
    pos[:, 0, 0] = 4.0 * t
    pos[:, 0, 2] = 1.5
    quat = torch.tensor(camera_quat(0.5 * math.sin(t)), dtype=torch.float32, device=DEV)
    return pos, quat.expand(n, 1, 4).contiguous()


def time_it(fn, reps):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(reps):
        fn(k)
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / reps


def bench_event(scene, n, res, period_ms, oversample, reps, texture=False, supersample=1,
                smooth=True):
    cfg = type("B", (GenX320EventCamConfig,), dict(
        width=res, height=res, frame_period_s=period_ms * 1e-3, oversample=oversample,
        texture_enabled=texture, supersample=supersample, smooth_normals=smooth))
    cam = WarpEventCam(n, cfg, scene.mesh_ids_array, DEV, meshes=scene.meshes)
    pos, quat = poses(n, 0)
    cam.set_pose_tensor(pos, quat)
    cam.capture(0)

    def step(k):
        p, q = poses(n, k + 1)
        pos.copy_(p)
        quat.copy_(q)
        cam.capture((k + 1) * STEP_MS * 1_000_000)

    step(0)  # compile / warm up
    return time_it(lambda k: step(k + 1), reps)


def bench_depth(mesh_ids, n, res, reps):
    f = (res / 2) / math.tan(math.radians(84.3) / 2)
    K = wp.mat44(f, 0.0, res / 2, 0.0, 0.0, f, res / 2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    K_inv = wp.inverse(K)
    pixels = torch.zeros((n, 1, res, res), device=DEV)
    pos, quat = poses(n, 1)
    args = [mesh_ids, wp.from_torch(pos, dtype=wp.vec3), wp.from_torch(quat, dtype=wp.quat), K_inv,
            20.0, wp.from_torch(pixels), res // 2, res // 2, True]

    def step(_):
        wp.launch(DepthCameraWarpKernels.draw_optimized_kernel_depth_range,
                  dim=(n, 1, res, res), inputs=args, device=DEV)

    step(0)
    return time_it(step, reps)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--num_envs", type=int, nargs="+", default=[64, 256])
    p.add_argument("--res", type=int, nargs="+", default=[64, 160, 320])
    p.add_argument("--modes", nargs="+", default=["30x1", "10x1", "10x4"],
                   help="frame_period_ms x oversample")
    p.add_argument("--num_obstacles", type=int, default=40)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--texture", action="store_true", help="enable the procedural texture")
    p.add_argument("--supersample", type=int, default=1, help="k x k sub-rays per pixel")
    p.add_argument("--flat_normals", action="store_true", help="face normals instead of smooth")
    args = p.parse_args()
    wp.init()

    rng = np.random.default_rng(0)
    forests = [random_forest(rng, args.num_obstacles) for _ in range(8)]
    print(f"{torch.cuda.get_device_name()} | texture {'on' if args.texture else 'off'} | "
          f"supersample {args.supersample} | {'flat' if args.flat_normals else 'smooth'} normals | "
          f"{args.num_obstacles} obstacles/env | "
          f"{STEP_MS} ms env step | ms per env step\n")
    header = ["envs", "res", "depth 1 ray"] + [f"T={m.split('x')[0]}ms os={m.split('x')[1]}"
                                               for m in args.modes]
    print(" | ".join(f"{h:>15}" for h in header))
    for n in args.num_envs:
        scene = SyntheticScene([forests[i % 8] for i in range(n)], DEV)
        for res in args.res:
            row = [str(n), f"{res}x{res}", f"{bench_depth(scene.mesh_ids_array, n, res, args.reps):.2f}"]
            for m in args.modes:
                period, os_ = (int(v) for v in m.split("x"))
                ms = bench_event(scene, n, res, period, os_, args.reps, args.texture,
                                 args.supersample, not args.flat_normals)
                row.append(f"{ms:.2f}")
            print(" | ".join(f"{c:>15}" for c in row))
        del scene
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
