"""Deterministic crash-cause eval: run a checkpoint at a fixed curriculum level, no
training, and log per-collision diagnostics needed to tell apart the candidate crash
causes (spawn traps, a specific obstacle type, or missing memory of an obstacle that
left the camera's view).

For every step where the task's own collision_mask (obs_dict["crashes"] > 0, excluding
exceed/arrive per compute_rewards()) fires for an env, records:
  - step number within the episode (task.sim_env.sim_steps) -> spawn-trap flag if < 5
  - the type of the nearest live obstacle to the drone at that instant (nearest-neighbour
    over env_asset_state_tensor, since this task exposes obstacle position but not a
    contact-body id -- see the CAVEATS note below)
  - whether that obstacle was inside the camera frustum (nominal mount pose, no
    translation offset or +/-5 deg jitter modeled) at each of the last `--hist` frames
  - the angle between body-frame velocity and the camera boresight (+x body axis), and
    impact speed

usage: python analysis/crash_cause_eval.py <checkpoint.pth> <level> <out.json> \
           [--num_envs 256] [--num_steps 6000] [--hist 10] [--spawn_trap_steps 5]

CAVEATS (see the resulting report for how these are called out):
  - No contact-body id exists in this task; the "obstacle hit" is a nearest-live-obstacle
    lookup at the collision step, which can misattribute a crash when two obstacles are
    within a few tens of cm of each other.
  - Camera FOV check uses the nominal (un-jittered) mount pose: ignores the <=12cm
    translation offset and the +/-5 deg rotation randomization applied per-env at reset.
  - Vertical FOV is not configured upstream; it is derived from horizontal_fov_deg and the
    pixel aspect ratio assuming square pixels, not read from the simulator directly.
  - "objects" (upstream's generic clutter mesh) is 47.5% of the obstacle pool -- the
    single largest category -- and is not one of tree/panel/sphere/cylinder/wall.
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import json
import math
import os
import re
import sys

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.math import quat_rotate_inverse

_NORM_EPS, _NORM_CLAMP = 1e-5, 5.0

# Camera geometry (config/sensor_config/realsense_d435_cam_config.py). Nominal
# (un-jittered) mount: position offset ignored (<=12cm, small vs. obstacle standoff),
# rotation offset ignored (nominal_orientation_euler_deg = [0,0,0] -> boresight is the
# body +x axis). Vertical FOV is not configured upstream; assumed from the pixel aspect
# ratio (square pixels), which is the standard pinhole assumption but unverified against
# aerial_gym's actual camera projection -- flagged in the report.
_H_FOV_DEG = 87.0
_CAM_H, _CAM_W = 180, 320
_HALF_H_FOV = math.radians(_H_FOV_DEG) / 2.0
_HALF_V_FOV = math.atan(math.tan(_HALF_H_FOV) * (_CAM_H / _CAM_W))
_MIN_RANGE, _MAX_RANGE = 0.1, 10.0

# env_forest_with_obstacles.py asset_type_to_dict_map -> the 5 categories the user asked
# about, plus "object" (upstream's generic clutter mesh, 47.5% of the cullable pool --
# the single largest category, and NOT one of tree/panel/sphere/cylinder/wall) and
# "thin" (also enabled, share not stated locally). See analysis/CRASH_CAUSE_NOTES.md.
_TYPE_MAP = {
    "trees": "tree",
    "panels": "panel",
    "spheres": "sphere",
    "cylinders": "cylinder",
    "left_wall": "wall", "right_wall": "wall",
    "front_wall": "wall", "back_wall": "wall",
    "bottom_wall": "wall", "top_wall": "wall",
    "objects": "object",
    "thin": "thin",
}


def build_actor(weights):
    idx = sorted(int(m.group(1)) for k in weights
                 if (m := re.fullmatch(r"a2c_network\.actor\.trunk\.(\d+)\.weight", k)))
    layers = []
    for i in idx:
        w = weights[f"a2c_network.actor.trunk.{i}.weight"]
        b = weights[f"a2c_network.actor.trunk.{i}.bias"]
        lin = torch.nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w); lin.bias.data.copy_(b)
        layers += [lin, torch.nn.ELU()]
    hw = weights["a2c_network.actor.action_head.weight"]
    hb = weights["a2c_network.actor.action_head.bias"]
    head = torch.nn.Linear(hw.shape[1], hw.shape[0])
    head.weight.data.copy_(hw); head.bias.data.copy_(hb)
    layers.append(head)
    return torch.nn.Sequential(*layers), hw.shape[1]  # (model, action_dim) -- unused


def classify(raw_type):
    return _TYPE_MAP.get(raw_type, raw_type)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("level", type=int)
    p.add_argument("out")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=6000)
    p.add_argument("--hist", type=int, default=10)
    p.add_argument("--spawn_trap_steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    dev = "cuda:0"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    w = ck["model"]
    if any(k.startswith("a2c_network.actor.rnn") for k in w):
        raise NotImplementedError(
            "checkpoint has a recurrent actor (a2c_network.actor.rnn.*); "
            "build_actor() only reconstructs a plain MLP trunk."
        )
    actor, _ = build_actor(w)
    actor = actor.to(dev).eval()
    mean = w["running_mean_std.running_mean"].float().to(dev)
    std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)

    from config.task_config import F450NavTaskConfig
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level
    if args.seed is not None:
        F450NavTaskConfig.seed = args.seed
    print(f"seed: {F450NavTaskConfig.seed}, level: {args.level}, "
          f"num_envs: {args.num_envs}, num_steps: {args.num_steps}")

    task = task_registry.make_task(
        "f450_navigation_task", num_envs=args.num_envs, headless=True, use_warp=True
    )

    obs_dim = actor[0].in_features
    assert obs_dim == task.task_config.observation_space_dim, (
        f"checkpoint actor expects {obs_dim}-D observations but the currently checked-out "
        f"task config produces {task.task_config.observation_space_dim}-D -- this worktree's "
        f"code does not match what the checkpoint was trained under, results would be garbage."
    )

    num_envs = args.num_envs
    HIST = args.hist

    hist_pos = torch.zeros(HIST, num_envs, 3, device=dev)
    hist_quat = torch.zeros(HIST, num_envs, 4, device=dev)
    hist_valid = torch.zeros(HIST, num_envs, dtype=torch.bool, device=dev)
    hist_ptr = [0]
    prev_steps = torch.full((num_envs,), -1, dtype=torch.long, device=dev)

    records = []
    counters = {"crash_events": 0}

    orig_compute_rewards = task.compute_rewards

    def patched_compute_rewards(obs_dict, current_action):
        reward, terminations, arrive_mask, exceed_mask = orig_compute_rewards(
            obs_dict, current_action
        )
        collision_mask = terminations & (~arrive_mask) & (~exceed_mask)

        cur_steps = task.sim_env.sim_steps.clone()
        restarted = cur_steps <= prev_steps
        prev_steps.copy_(cur_steps)
        if restarted.any():
            hist_valid[:, restarted] = False
        ptr = hist_ptr[0]
        hist_pos[ptr] = obs_dict["robot_position"]
        hist_quat[ptr] = obs_dict["robot_orientation"]
        hist_valid[ptr] = True
        hist_ptr[0] = (ptr + 1) % HIST

        if collision_mask.any():
            crashed_idx = collision_mask.nonzero(as_tuple=True)[0]
            robot_pos = obs_dict["robot_position"]
            robot_ori = obs_dict["robot_orientation"]
            robot_linvel = obs_dict["robot_linvel"]
            asset_state = obs_dict["env_asset_state_tensor"]
            for i in crashed_idx.tolist():
                counters["crash_events"] += 1
                step_num = int(cur_steps[i])

                v_world = robot_linvel[i:i + 1]
                v_body = quat_rotate_inverse(robot_ori[i:i + 1], v_world)[0]
                speed = float(torch.linalg.norm(v_body))
                if speed > 1e-6:
                    angle_to_cam_deg = math.degrees(
                        math.acos(max(-1.0, min(1.0, float(v_body[0]) / speed)))
                    )
                else:
                    angle_to_cam_deg = float("nan")

                positions = asset_state[i, :, 0:3]
                valid = positions[:, 0] > -999.0
                obstacle_type = "unknown"
                obstacle_dist = float("nan")
                in_view_frac = float("nan")
                hist_frames_available = 0
                in_view_count = 0
                if valid.any():
                    valid_positions = positions[valid]
                    dists = torch.linalg.norm(valid_positions - robot_pos[i], dim=1)
                    nearest_local = int(torch.argmin(dists))
                    obstacle_dist = float(dists[nearest_local])
                    valid_slot_indices = valid.nonzero(as_tuple=True)[0]
                    slot_idx = int(valid_slot_indices[nearest_local])
                    obstacle_pos = valid_positions[nearest_local]
                    raw_type = task.sim_env.global_asset_dicts[i][slot_idx]["asset_type"]
                    obstacle_type = classify(raw_type)

                    valid_hist = hist_valid[:, i]
                    hist_frames_available = int(valid_hist.sum())
                    if hist_frames_available > 0:
                        h_pos = hist_pos[valid_hist, i]
                        h_quat = hist_quat[valid_hist, i]
                        rel_world = obstacle_pos.unsqueeze(0) - h_pos
                        rel_body = quat_rotate_inverse(h_quat, rel_world)
                        dist_h = torch.linalg.norm(rel_body, dim=1)
                        h_angle = torch.atan2(rel_body[:, 1], rel_body[:, 0])
                        v_angle = torch.atan2(rel_body[:, 2], rel_body[:, 0])
                        in_frustum = (
                            (rel_body[:, 0] > 0)
                            & (dist_h >= _MIN_RANGE) & (dist_h <= _MAX_RANGE)
                            & (h_angle.abs() <= _HALF_H_FOV)
                            & (v_angle.abs() <= _HALF_V_FOV)
                        )
                        in_view_count = int(in_frustum.sum())
                        in_view_frac = in_view_count / hist_frames_available

                records.append({
                    "env": i,
                    "step": step_num,
                    "spawn_trap": step_num < args.spawn_trap_steps,
                    "obstacle_type": obstacle_type,
                    "obstacle_dist_m": obstacle_dist,
                    "hist_frames_available": hist_frames_available,
                    "in_view_count": in_view_count,
                    "in_view_frac": in_view_frac,
                    "speed_mps": speed,
                    "angle_to_camera_deg": angle_to_cam_deg,
                })
        return reward, terminations, arrive_mask, exceed_mask

    task.compute_rewards = patched_compute_rewards

    obs = task.reset()[0]["observations"]
    total_episodes = 0
    with torch.no_grad():
        for step_i in range(args.num_steps):
            mu = actor(torch.clamp((obs - mean) / std, -_NORM_CLAMP, _NORM_CLAMP))
            cmd = mu.clamp(-1, 1)
            o, _, term, trunc, _ = task.step(cmd)
            obs = o["observations"]
            total_episodes += int((term | trunc).sum())
            if (step_i + 1) % 500 == 0:
                print(f"step {step_i + 1}/{args.num_steps}: "
                      f"{total_episodes} episodes, {counters['crash_events']} crashes")

    crash_rate = counters["crash_events"] / total_episodes if total_episodes else float("nan")
    print(f"done: {total_episodes} episodes, {counters['crash_events']} crashes, "
          f"crash_rate={crash_rate:.4f}")

    json.dump({
        "checkpoint": args.checkpoint.split("/")[-1],
        "epoch": int(ck.get("epoch", -1)),
        "level": args.level,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "hist_len": HIST,
        "spawn_trap_steps": args.spawn_trap_steps,
        "camera_half_h_fov_deg": math.degrees(_HALF_H_FOV),
        "camera_half_v_fov_deg": math.degrees(_HALF_V_FOV),
        "camera_max_range_m": _MAX_RANGE,
        "total_episodes": total_episodes,
        "total_crashes": counters["crash_events"],
        "crash_rate": crash_rate,
        "records": records,
    }, open(args.out, "w"))
    print(f"wrote {args.out}")
    task.close()


if __name__ == "__main__":
    main()
