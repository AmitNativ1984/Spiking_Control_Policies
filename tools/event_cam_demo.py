"""Render the warp event camera to a video and print DIFF3D frame statistics.

Two scenes:
    synthetic   ground + random cylinders/spheres/boxes, scripted fly-through (no physics)
    aerial_gym  forest_with_obstacles_env with the f450_event_cam robot, flown open-loop
                with attitude commands through the real sim (also an integration check)

By default each video frame is one env step: the signed sum of every DIFF3D frame that closed
during that step (red = net ON, blue = net OFF, brightness ~ |sum|), beside metric depth. No frame is
dropped or counted twice as long as num_frames covers the longest step. Statistics follow
Ou et al. (arXiv 2610.37543, Table II) on those per-step sums: share of active pixels, ON
share among them, and p50/p90 of |sum|. With --per_frame every DIFF3D frame becomes its own
video frame (rate 1/T), and the depth panel holds its env step's image.

Usage:
    python -m tools.event_cam_demo                              # synthetic, 3 s
    python -m tools.event_cam_demo --scene aerial_gym --num_envs 4 --num_obstacles 30
    python -m tools.event_cam_demo --frame_period_ms 5 --oversample 4 --out runs/ev.mp4
    python -m tools.event_cam_demo --texture --out runs/ev_tex.mp4
    python -m tools.event_cam_demo --per_frame --frame_period_ms 5 --slowdown 4
"""
import isaacgym  # noqa: F401  — must precede torch

import argparse
import math
import os

import cv2
import numpy as np
import torch

import config  # noqa: F401  — registers robots and applies the event sensor patch
from config.sensor_config.genx320_event_cam_config import GenX320EventCamConfig

DEV = "cuda:0"


def diff3d_to_bgr(frame, depth, max_range, scale=2):
    f = frame.astype(np.float32)
    mag = np.clip(np.abs(f) / 4.0, 0.0, 1.0)
    img = np.full(f.shape + (3,), 25, dtype=np.float32)
    img[f > 0] = (np.array([40, 40, 255]) * mag[f > 0, None] + 25 * (1 - mag[f > 0, None]))
    img[f < 0] = (np.array([255, 120, 40]) * mag[f < 0, None] + 25 * (1 - mag[f < 0, None]))
    d = np.clip(depth / max_range, 0.0, 1.0)
    d_img = cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    out = np.concatenate([img.astype(np.uint8), d_img], axis=1)
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def frame_stats(frames):
    """frames: (F, H, W) signed counts -> dict of Ou et al. Table II style statistics."""
    f = frames.to(torch.int32)
    active = f != 0
    n = int(active.sum())
    mags = f.abs()[active].float()
    return {
        "active_px_%": 100.0 * n / f.numel(),
        "on_share_%": 100.0 * int((f > 0).sum()) / max(n, 1),
        "p50": float(mags.median()) if n else 0.0,
        "p90": float(torch.quantile(mags[: 1_000_000], 0.9)) if n else 0.0,
    }


def make_cfg(args):
    return type("DemoCfg", (GenX320EventCamConfig,), dict(
        frame_period_s=args.frame_period_ms * 1e-3, oversample=args.oversample,
        num_frames=args.num_frames, write_depth=True, texture_enabled=args.texture,
    ))


def run_synthetic(args):
    from sensors.event_camera import WarpEventCam
    from sensors.event_camera.synthetic import SyntheticScene, camera_quat, random_forest

    scene = SyntheticScene([random_forest(np.random.default_rng(args.seed), args.num_obstacles)], DEV)
    cam = WarpEventCam(1, make_cfg(args), scene.mesh_ids_array, DEV, meshes=scene.meshes)
    pos = torch.zeros((1, 1, 3), device=DEV)
    quat = torch.zeros((1, 1, 4), device=DEV)
    cam.set_pose_tensor(pos, quat)

    def pose_at(t):
        pos[0, 0] = torch.tensor([args.speed * t, 1.0 * math.sin(1.3 * t), 1.5 + 0.3 * math.sin(2.1 * t)])
        quat[0, 0] = torch.tensor(camera_quat(0.4 * math.sin(0.9 * t), 0.05 * math.sin(1.7 * t)),
                                  dtype=torch.float32)

    step_ms = 30
    pose_at(0.0)
    cam.capture(0)
    for k in range(1, int(args.seconds * 1000 / step_ms) + 1):
        pose_at(k * step_ms * 1e-3)
        frames = cam.capture(k * step_ms * 1_000_000)
        yield frames[0], cam.depth[0], cam.cfg.max_range, cam


def run_aerial_gym(args):
    from aerial_gym.sim.sim_builder import SimBuilder

    GenX320EventCamConfig.frame_period_s = args.frame_period_ms * 1e-3
    GenX320EventCamConfig.oversample = args.oversample
    GenX320EventCamConfig.num_frames = args.num_frames
    GenX320EventCamConfig.write_depth = True
    GenX320EventCamConfig.texture_enabled = args.texture

    env = SimBuilder().build_env(
        sim_name="base_sim", env_name="forest_with_obstacles_env", robot_name="f450_event_cam",
        controller_name="f450_lee_attitude_control", args=None, device=DEV,
        num_envs=args.num_envs, headless=True, use_warp=True,
    )
    from env_manager import warp_bvh_rebuild_patch
    warp_bvh_rebuild_patch.bind(env)
    env.global_tensor_dict["num_obstacles_in_env"] = args.num_obstacles
    env.reset()
    gtd = env.global_tensor_dict
    assert gtd["event_frames"].shape == (args.num_envs, args.num_frames, 320, 320)
    assert gtd["event_frames"].dtype == torch.int8

    actions = torch.zeros((args.num_envs, 4), device=DEV)
    step = 0
    while step * 0.03 < args.seconds:
        t = step * 0.03
        actions[:, 0] = 0.05  # [thrust, roll, pitch, yaw_rate]; 0 is level hover
        actions[:, 1] = 0.15 * math.sin(2.0 * t)
        actions[:, 2] = 0.15 * math.sin(1.3 * t + 1.0)
        actions[:, 3] = 0.8 * math.sin(0.9 * t)
        env.step(actions=actions)
        env.post_reward_calculation_step()  # resets crashed envs, then renders sensors
        step += 1
        depth = gtd["depth_range_pixels"][0, 0]
        yield gtd["event_frames"][0], depth, GenX320EventCamConfig.max_range, env.robot_manager.warp_sensor.sensor


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scene", choices=["synthetic", "aerial_gym"], default="synthetic",
                   help="synthetic: scripted fly-through, no physics; aerial_gym: real sim with "
                        "the f450_event_cam robot")
    p.add_argument("--seconds", type=float, default=3.0,
                   help="simulated duration [s]")
    p.add_argument("--frame_period_ms", type=float, default=10.0,
                   help="DIFF3D frame period T [ms]")
    p.add_argument("--num_frames", type=int, default=None,
                   help="DIFF3D frames kept in the output window (B). Default: just enough for "
                        "the longest env step, ceil(max_step / T): 30 ms synthetic, 60 ms aerial_gym")
    p.add_argument("--oversample", type=int, default=1,
                   help="event samples per frame (O); sample rate = O / T")
    p.add_argument("--num_obstacles", type=int, default=40,
                   help="obstacles per env")
    p.add_argument("--num_envs", type=int, default=2,
                   help="parallel envs (aerial_gym scene only; env 0 is recorded)")
    p.add_argument("--speed", type=float, default=4.0,
                   help="forward speed of the scripted camera [m/s] (synthetic scene only)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the scene and per-episode randomization")
    p.add_argument("--texture", action="store_true",
                   help="enable the world-anchored fBm albedo texture")
    p.add_argument("--per_frame", action="store_true",
                   help="one video frame per DIFF3D frame (rate 1/T) instead of one per env step "
                        "(~30 ms, signed sum of the step's frames); depth is held per env step")
    p.add_argument("--slowdown", type=float, default=2.0,
                   help="playback slow-motion factor: video fps = event rate / slowdown")
    p.add_argument("--out", default="runs/event_cam_demo.mp4",
                   help="output video path; one frame per env step (left: signed sum of the "
                        "step's DIFF3D frames, red ON / blue OFF; right: depth)")
    args = p.parse_args()
    if args.num_frames is None:
        max_step_ms = 30.0 if args.scene == "synthetic" else 60.0  # aerial_gym: up to ~6 x 10 ms substeps
        args.num_frames = math.ceil(max_step_ms / args.frame_period_ms)

    torch.manual_seed(args.seed)
    gen = run_synthetic(args) if args.scene == "synthetic" else run_aerial_gym(args)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # step mode: one video frame per env step (~30 ms); per_frame: one per DIFF3D frame (T)
    fps = (1000.0 / args.frame_period_ms if args.per_frame else 1000.0 / 30.0) / args.slowdown
    writer, sums, prev_latest, dropped, written = None, [], 0, 0, 0  # clocks start at t = 0
    for frames, depth, max_range, cam in gen:
        latest = cam.t_ns // cam.frame_ns
        new = latest - prev_latest
        prev_latest = latest
        dropped += max(0, new - frames.shape[0])
        new = min(new, frames.shape[0])
        fresh = frames[frames.shape[0] - new:].to(torch.int32)  # frames closed this step, oldest first
        step_sum = fresh.sum(0) if new else \
            torch.zeros(frames.shape[1:], dtype=torch.int32, device=frames.device)
        sums.append(step_sum)
        depth_np = depth.cpu().numpy()
        # per_frame holds this step's depth over each of its frames: depth exists once per step
        shown = list(fresh) if args.per_frame else [step_sum]
        for ev in shown:
            img = diff3d_to_bgr(ev.cpu().numpy(), depth_np, max_range)
            if writer is None:
                writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
            written += 1
    writer.release()

    s = frame_stats(torch.stack(sums[1:]))
    unit = f"DIFF3D frames ({args.frame_period_ms:g} ms each)" if args.per_frame else "env steps"
    print(f"\nwrote {args.out}  ({written} {unit} at {fps:.1f} fps = {args.slowdown:g}x slow motion)")
    if dropped:
        print(f"WARNING: {dropped} DIFF3D frames closed while older than the {args.num_frames}-frame "
              "window and are missing from the sums; raise --num_frames")
    print("per-step sum stats: " + "  ".join(f"{k}={v:.2f}" for k, v in s.items()))
    if s["active_px_%"] == 0.0:
        raise SystemExit("no events were generated")


if __name__ == "__main__":
    main()
