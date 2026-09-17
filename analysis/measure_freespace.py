"""Measure the free space the drone actually flies through, from the rendered depth image.

Answers "is there anywhere to fly?" against the REAL collision geometry -- tree branches
included -- rather than against a Poisson-sphere estimate, which cannot represent a tree
(26 branches, 4.71 m mean horizontal reach) as anything but a 0.4 m ball.

Runs a checkpoint at a pinned curriculum level and, every step, reduces the depth image to:
  - nearest surface anywhere in the FOV        (how boxed-in the drone is right now)
  - widest contiguous free column band         (is there a corridor to fly down?)
  - fraction of the image closer than a drone-width
and records the same quantities in the steps immediately before each collision, so the
"boxed in with nowhere to go" hypothesis can be separated from "had room, hit it anyway".

usage: python analysis/measure_freespace.py <checkpoint.pth> <level> <out.json>
           [--num_envs 128] [--num_steps 2000]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import json
import math
import re

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry

_NORM_EPS, _NORM_CLAMP = 1e-5, 5.0
MAX_RANGE = 10.0          # RealSenseD435CamConfig.max_range
DRONE_R = 0.35            # F450 450 mm diagonal + prop, effective radius
CORRIDOR_M = 2 * DRONE_R  # clear width the drone needs


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
    return torch.nn.Sequential(*layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("level", type=int)
    p.add_argument("out")
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--num_steps", type=int, default=2000)
    args = p.parse_args()

    dev = "cuda:0"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    w = ck["model"]
    actor = build_actor(w).to(dev).eval()
    mean = w["running_mean_std.running_mean"].float().to(dev)
    std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)

    from config.task_config import F450NavTaskConfig
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level

    task = task_registry.make_task("f450_navigation_task", num_envs=args.num_envs,
                                   headless=True, use_warp=True)

    # --- depth encoding: report it rather than assume it ---
    obs = task.reset()[0]["observations"]
    d0 = task.obs_dict["depth_range_pixels"]
    print(f"depth tensor {tuple(d0.shape)}  min {float(d0.min()):.4f}  "
          f"max {float(d0.max()):.4f}  mean {float(d0.mean()):.4f}")

    HIST_N = 64
    nearest_hist = torch.zeros(HIST_N, device=dev)   # 0..10 m
    corridor_hist = torch.zeros(HIST_N, device=dev)  # 0..10 m
    boxed_steps = 0
    total_steps = 0
    pre_crash = []

    orig_compute_rewards = task.compute_rewards
    LOOKBACK = 40   # ~1.2 s: far enough that lateral authority is metres
    ring = {"near": [], "corr": []}

    def depth_stats():
        """-> (nearest surface [m], widest free corridor width [m]) per env."""
        d = task.obs_dict["depth_range_pixels"].squeeze(1)      # (E, H, W)
        m = d.clone()
        m[m < 0] = 1.0                       # negative = no return = beyond range
        metres = m * MAX_RANGE
        nearest = metres.flatten(1).min(dim=1).values
        # widest contiguous run of image COLUMNS whose closest pixel is far enough to
        # fly down: a crude but honest "is there a gap" measure that sees branches.
        col_min = metres.min(dim=1).values                      # (E, W) nearest per column
        free = (col_min > 1.5).float()
        # longest run of 1s per row, vectorised by cumulative reset
        runs = torch.zeros_like(free)
        acc = torch.zeros(free.shape[0], device=free.device)
        for j in range(free.shape[1]):
            acc = (acc + free[:, j]) * free[:, j]
            runs[:, j] = acc
        widest_cols = runs.max(dim=1).values
        # convert a column count to an angular width, then to metres at 3 m standoff
        deg_per_col = 87.0 / free.shape[1]
        widest_m = 2 * 3.0 * torch.tan(torch.deg2rad(widest_cols * deg_per_col) / 2)
        return nearest, widest_m

    def patched(obs_dict, current_action):
        nonlocal boxed_steps, total_steps
        reward, term, arrive, exceed = orig_compute_rewards(obs_dict, current_action)
        near, corr = depth_stats()
        nearest_hist.add_(torch.histc(near, bins=HIST_N, min=0.0, max=10.0))
        corridor_hist.add_(torch.histc(corr, bins=HIST_N, min=0.0, max=10.0))
        boxed_steps += int((corr < CORRIDOR_M).sum())
        total_steps += near.shape[0]
        ring["near"].append(near.clone())
        ring["corr"].append(corr.clone())
        if len(ring["near"]) > LOOKBACK:
            ring["near"].pop(0); ring["corr"].pop(0)
        coll = term & (~arrive) & (~exceed)
        if coll.any():
            for i in coll.nonzero(as_tuple=True)[0].tolist():
                pre_crash.append({
                    "nearest_at_impact": float(near[i]),
                    "corridor_at_impact": float(corr[i]),
                    "corridor_lookback": float(ring["corr"][0][i]),
                    "lookback_steps": len(ring["corr"]),
                    "nearest_lookback": float(ring["near"][0][i]),
                })
        return reward, term, arrive, exceed

    task.compute_rewards = patched

    with torch.no_grad():
        for s in range(args.num_steps):
            mu = actor(torch.clamp((obs - mean) / std, -_NORM_CLAMP, _NORM_CLAMP))
            o, _, _, _, _ = task.step(mu.clamp(-1, 1))
            obs = o["observations"]
            if (s + 1) % 500 == 0:
                print(f"step {s+1}/{args.num_steps}  crashes logged {len(pre_crash)}")

    json.dump({
        "checkpoint": args.checkpoint.split("/")[-1],
        "level": args.level,
        "steps": total_steps,
        "drone_width_m": CORRIDOR_M,
        "frac_steps_no_corridor": boxed_steps / max(total_steps, 1),
        "nearest_hist": nearest_hist.tolist(),
        "corridor_hist": corridor_hist.tolist(),
        "hist_edges": [i * 10.0 / HIST_N for i in range(HIST_N + 1)],
        "pre_crash": pre_crash,
    }, open(args.out, "w"))
    print(f"wrote {args.out}")
    task.close()


if __name__ == "__main__":
    main()
