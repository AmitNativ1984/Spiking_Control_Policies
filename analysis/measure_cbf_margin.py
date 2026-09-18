"""Size d_safe, alpha and lambda_cbf for p_cbf from measurement rather than intuition.

Rolls a checkpoint deterministically at a fixed curriculum level and records, per step:

  - d_obstacle: the FULL-SPHERE exact distance to the nearest surface, read through the
    task's own _clearance() so this measures precisely what the reward measures -- not a
    depth-image proxy. The percentiles of this are what d_safe has to fit inside.
  - the height above the env floor, and how often the FLOOR is the nearest surface. The
    probe includes the floor, so a d_safe larger than the cruise altitude would turn
    p_cbf partly into an altitude tax. This says whether that is happening.
  - v_close = -dh/dt, the rate clearance is actually being spent, and from it the
    violation rate and mean excess over a GRID of (d_safe, alpha) -- so the cost of each
    candidate pair is visible before a single GPU-hour is spent training on it.

SIZING RULE (the same one used for lambda_fov): pick the mean per-step penalty to land
near r_progress's, which the p_fov baseline measures at 0.0255/step -- loud enough for
the policy to attend to, not loud enough to displace the task. The table prints
    lambda_cbf = target / mean(max(0, v_close - alpha*h))
for each cell, which is the number to paste into the config.

PAIRING. v_close for step t is (h_t - h_{t+1})/dt, and h_{t+1} is only the same env's
clearance if it did not terminate: post_reward_calculation_step() teleports the robot and
rebuilds that env's warp mesh. Terminated envs are therefore dropped from the pairing --
which is also exactly the set the reward excludes, since p_cbf only applies under
progress_mask. The two agree step for step.

RESET LEAKAGE CHECK. DF is 1-Lipschitz in position, so |v_close| can never exceed the
drone's speed. A teleport that leaked into the pairing would show up as a v_close of tens
of m/s and nothing else can produce one, so the script asserts it and reports the margin.

REQUIRES a validated probe. Run analysis/validate_proximity_probe.py first: if the mesh
query is wrong, every number here is wrong in the same direction and nothing says so.

usage: python analysis/measure_cbf_margin.py <checkpoint.pth> <level> <out.json> \
           [--num_envs 256] [--num_steps 4000] [--target 0.0255]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import json

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry
try:  # `python -m analysis.measure_cbf_margin`, repo root on sys.path
    from analysis.crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP
except ImportError:  # `python analysis/measure_cbf_margin.py`, sibling on sys.path
    from crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP

HIST_N = 120
HIST_MAX = 6.0  # m, matches the probe's default range

# The grid the table is printed over. d_safe beyond ~1.2 m is included so the tradeoff is
# visible rather than asserted, not because it is a candidate.
D_SAFE_GRID = [0.7, 1.0, 1.25, 1.5]
ALPHA_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]


def _pct(hist, edges, qs):
    tot = float(hist.sum())
    out, c, i = {}, 0.0, 0
    for b in range(len(hist)):
        c += float(hist[b])
        while i < len(qs) and c / tot >= qs[i]:
            out[qs[i]] = edges[b + 1]
            i += 1
    for q in qs[i:]:
        out[q] = edges[-1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("level", type=int)
    p.add_argument("out")
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=4000)
    p.add_argument("--target", type=float, default=0.0255,
                   help="target mean per-step penalty; default is r_progress as measured "
                        "on the p_fov baseline at level 30")
    args = p.parse_args()

    dev = "cuda:0"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    w = ck["model"]
    actor, _ = build_actor(w)
    actor = actor.to(dev).eval()
    mean = w["running_mean_std.running_mean"].float().to(dev)
    std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)

    from config.task_config import F450NavTaskConfig
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level
    # Force the probe on without changing behaviour: the actor is deterministic and the
    # reward is not fed back during an eval rollout, so a non-zero lambda_cbf only builds
    # the probe. 1.0 is chosen so the task's own reward/p_cbf EMA reads -mean(excess)
    # directly, giving an independent cross-check on the table this script prints.
    F450NavTaskConfig.reward_parameters["lambda_cbf"] = 1.0
    d_safe_task = F450NavTaskConfig.reward_parameters["d_safe"]

    task = task_registry.make_task(
        "f450_navigation_task", num_envs=args.num_envs, headless=True, use_warp=True
    )
    assert task._cbf_active, "probe did not come up; p_cbf would be inert"
    dt = task._env_step_dt

    obs_dim = actor[0].in_features
    assert obs_dim == task.task_config.observation_space_dim, (
        f"checkpoint expects {obs_dim}-D observations, this tree produces "
        f"{task.task_config.observation_space_dim}-D -- results would be garbage."
    )

    edges = [i * HIST_MAX / HIST_N for i in range(HIST_N + 1)]
    d_hist = torch.zeros(HIST_N, device=dev)
    alt_hist = torch.zeros(HIST_N, device=dev)
    floor_nearest = torch.zeros((), device=dev)
    n_samples = torch.zeros((), device=dev)

    # Per-cell accumulators for the (d_safe, alpha) table.
    cells = {(ds, al): [torch.zeros((), device=dev), torch.zeros((), device=dev)]
             for ds in D_SAFE_GRID for al in ALPHA_GRID}
    n_pairs = torch.zeros((), device=dev)
    max_vclose = torch.zeros((), device=dev)
    max_speed = torch.zeros((), device=dev)

    obs = task.reset()[0]["observations"]
    prev_d = None
    ended_last = torch.ones(args.num_envs, dtype=torch.bool, device=dev)

    with torch.no_grad():
        for s in range(args.num_steps):
            # d at the START of the step: the same position and the same mesh the task's
            # own prev_h snapshot reads a moment later.
            d = task._clearance() + d_safe_task
            z_agl = (task.obs_dict["robot_position"][:, 2]
                     - task.obs_dict["env_bounds_min"][:, 2])
            speed = torch.linalg.norm(task.obs_dict["robot_linvel"], dim=1)

            d_hist.add_(torch.histc(d, bins=HIST_N, min=0.0, max=HIST_MAX))
            alt_hist.add_(torch.histc(z_agl, bins=HIST_N, min=0.0, max=HIST_MAX))
            # "The floor is what the probe found": d within 5 cm of the height above it.
            floor_nearest.add_(((d - z_agl).abs() < 0.05).sum())
            n_samples.add_(d.numel())
            max_speed.copy_(torch.maximum(max_speed, speed.max()))

            if prev_d is not None:
                alive = ~ended_last
                if alive.any():
                    v_close = (prev_d[alive] - d[alive]) / dt
                    n_pairs.add_(v_close.numel())
                    max_vclose.copy_(torch.maximum(max_vclose, v_close.abs().max()))
                    for (ds, al), acc in cells.items():
                        h = prev_d[alive] - ds
                        excess = torch.clamp(v_close - al * h, min=0.0)
                        acc[0].add_((excess > 0).sum())
                        acc[1].add_(excess.sum())

            mu = actor(torch.clamp((obs - mean) / std, -_NORM_CLAMP, _NORM_CLAMP))
            o, _, term, trunc, _ = task.step(mu.clamp(-1, 1))
            obs = o["observations"]
            ended_last = (term > 0) | (trunc > 0)
            prev_d = d.clone()

            if (s + 1) % 500 == 0:
                print(f"step {s + 1}/{args.num_steps}")

    # --- reset leakage: DF is 1-Lipschitz, so |v_close| <= speed, always ---------------
    mv, ms = float(max_vclose), float(max_speed)
    print(f"\nmax |v_close| {mv:.2f} m/s vs max speed {ms:.2f} m/s")
    assert mv <= ms * 1.05 + 0.5, (
        f"max |v_close| {mv:.2f} m/s exceeds the max speed {ms:.2f} m/s: DF is "
        f"1-Lipschitz in position, so this can only be a teleport leaking across a reset "
        f"into the pairing. The barrier would read it as an enormous closing speed."
    )

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
    d_pct = _pct(d_hist, edges, qs)
    alt_pct = _pct(alt_hist, edges, qs)
    frac_floor = float(floor_nearest) / float(n_samples)

    print("\nd_obstacle (full sphere, floor included) percentiles [m]:")
    print("  " + "  ".join(f"p{int(q * 100)} {d_pct[q]:.2f}" for q in qs))
    print("height above floor percentiles [m]:")
    print("  " + "  ".join(f"p{int(q * 100)} {alt_pct[q]:.2f}" for q in qs))
    print(f"the FLOOR is the nearest surface on {100 * frac_floor:.1f}% of steps")

    print(f"\nlambda_cbf for a {args.target}/step target, "
          f"over {int(n_pairs)} paired steps:")
    print(f"{'d_safe':>7}  {'alpha':>6}  {'viol%':>7}  {'mean excess':>12}  {'lambda':>9}")
    table = {}
    for (ds, al), acc in sorted(cells.items()):
        rate = float(acc[0]) / float(n_pairs)
        mean_ex = float(acc[1]) / float(n_pairs)
        lam = args.target / mean_ex if mean_ex > 0 else float("inf")
        table[f"{ds}_{al}"] = {
            "d_safe": ds, "alpha": al, "violation_rate": rate,
            "mean_excess_mps": mean_ex, "lambda_cbf": lam,
        }
        print(f"{ds:>7.2f}  {al:>6.2f}  {100 * rate:>6.1f}%  {mean_ex:>12.4f}  {lam:>9.4f}")

    print("\nRead the violation rate first. Near 1.0 means the barrier is violated on "
          "essentially every step, so p_cbf has degenerated into a flat speed tax with "
          "no gradient structure left -- raise alpha. Near 0.0 means it never fires.")

    json.dump({
        "checkpoint": args.checkpoint,
        "level": args.level,
        "steps": args.num_steps,
        "num_envs": args.num_envs,
        "dt": dt,
        "target_per_step": args.target,
        "d_pct": {str(k): v for k, v in d_pct.items()},
        "alt_pct": {str(k): v for k, v in alt_pct.items()},
        "frac_floor_nearest": frac_floor,
        "max_v_close": mv,
        "max_speed": ms,
        "hist_edges": edges,
        "d_hist": d_hist.tolist(),
        "alt_hist": alt_hist.tolist(),
        "table": table,
    }, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
