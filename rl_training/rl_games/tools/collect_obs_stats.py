"""Observation statistics collector for the F450 navigation task.

Drives the environment for a fixed number of steps and computes per-dimension
statistics, saved to CSV and logged to TensorBoard/W&B.

Two rollout drivers:
  * random actions (default) — uniform in [-1, 1].
  * teacher-driven (--teacher_checkpoint) — actions = clamp(teacher_mu, -1, 1),
    matching how the warm-started student will actually be driven. Preferred for
    setting PopSAN encoder bounds so the bounds reflect the on-policy state
    distribution, not the random one.

PopSAN encoder bounds live in rl_games-NORMALIZED space (z-scores, hard-clamped to
[-5, 5] by RunningMeanStd). When a teacher checkpoint is given we normalize the collected
raw obs through the teacher's frozen running_mean_std before computing the band, so the
emitted `observation_bounds` are directly usable by the encoder. The VAE-latent dims are
measured the same way from the rollout, so they reflect the CURRENT VAE, not the teacher's.

The result is written to a JSON cache that the student runner loads at startup (see
`load_or_collect_bounds`).

Usage:
    cd /workspaces/aerial_gym_docker
    python -m rl_training.rl_games.tools.collect_obs_stats \
        --num_steps=10000 --num_envs=64 \
        --teacher_checkpoint=runs/<teacher_run>/nn/teacher.pth \
        --config=rl_training/rl_games/cfg/popsan_teacher_student_local.yaml
"""
import isaacgym  # must be first

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import config  # registers the env, robot and task
from aerial_gym.registry.task_registry import task_registry

from .. import DEFAULT_BOUNDS_CACHE, OBS_STATS_DIR, REPO_ROOT

DEFAULT_TASK_NAME = "f450_navigation_task"


def resolve_task(config_path, task_name=None):
    """(task_name, task_config) for the student YAML at `config_path`.

    The config names its own task in `config.env_name`, the same key the runner uses, so
    the measured bounds cannot be collected against a different task than the one that
    will consume them.
    """
    if task_name is None and config_path is not None:
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                task_name = yaml.safe_load(f)["params"]["config"]["env_name"]
        except Exception as e:
            print(f"[obs-stats] couldn't read env_name from {config_path} ({e}); "
                  f"falling back to {DEFAULT_TASK_NAME}.")
    task_name = task_name or DEFAULT_TASK_NAME
    return task_name, task_registry.get_task_config(task_name)

# This module's own dotted path, used to re-launch it as a subprocess (see
# load_or_collect_bounds). Hard-coded rather than derived, because __spec__.name is
# "__main__" when the module runs as the script.
MODULE_PATH = "rl_training.rl_games.tools.collect_obs_stats"

# Percentile band used for encoder bounds.
LOWER_PCT = 1.0
UPPER_PCT = 99.0

# How encoder clamp bounds are derived from the (normalized) teacher obs distribution:
#   "percentile" : empirical p{lower_pct}/p{upper_pct} per dim. Distribution-free; follows
#                  asymmetric tails exactly, but the clamp window can sit off the center of mass.
#   "gaussian"   : fit (mu, sigma) per dim, clamp = mu +/- z*sigma where z is the normal quantile
#                  for the SAME coverage band (z(99%) ~= 2.326). CENTERED on the mean (center of
#                  mass) and symmetric. Cleaner centering, but imposes symmetry — for strongly
#                  skewed dims it clips the long-tail side and over-extends the short side.
DEFAULT_BOUND_METHOD = "gaussian"  # "gaussian" is the other option


def _gaussian_z(upper_pct):
    """Normal-distribution quantile z such that P(|X-mu| <= z*sigma) matches the requested
    central band. For an upper percentile p (e.g. 99), z = Phi^{-1}(p/100) (e.g. ~2.326).
    Uses scipy if available, else a rational approximation (Acklam) — both accurate to ~1e-4.
    """
    p = upper_pct / 100.0
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except Exception:
        # Acklam's inverse-normal approximation (no scipy dependency).
        import math
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _compute_bounds(bounds_arr, method, lower_pct, upper_pct):
    """Per-dim (lo, hi) encoder clamp bounds from the (normalized) obs array [N, obs_dim].

    method="percentile": lo/hi = empirical p{lower_pct}/p{upper_pct} per dim.
    method="gaussian":   lo/hi = mu -/+ z*sigma per dim, z the normal quantile for upper_pct,
                         centered on each dim's mean (center of mass), symmetric.
    """
    if method == "percentile":
        lo = np.percentile(bounds_arr, lower_pct, axis=0)
        hi = np.percentile(bounds_arr, upper_pct, axis=0)
    elif method == "gaussian":
        mu = bounds_arr.mean(axis=0)
        sigma = bounds_arr.std(axis=0)
        z = _gaussian_z(upper_pct)
        lo = mu - z * sigma
        hi = mu + z * sigma
    else:
        raise ValueError(f"unknown bound method {method!r} (expected 'percentile' or 'gaussian')")
    return lo, hi


def _obs_names(obs_dim: int, task_config):
    """Names per obs dim, derived from the task's observation_layout so they stay in sync
    with the real vector (state dims + however many VAE latents)."""
    names = [None] * obs_dim
    for obj_slice, obj_type in task_config.observation_layout:
        span = range(obj_slice.start, obj_slice.stop)
        for k, idx in enumerate(span):
            names[idx] = f"{obj_type}_{k}" if (obj_slice.stop - obj_slice.start) > 1 else obj_type
    return [n if n is not None else f"dim_{i}" for i, n in enumerate(names)]


def _load_teacher_cfg(config_path, teacher_config_path=None):
    """The teacher's architecture + normalization, from exactly one source of truth.

    Two shapes, because there are two ways to have a teacher:
      * `--teacher_config` pointing at the ANN's OWN training YAML (e.g. ppo_hover_local),
        whose `network:` block IS the teacher. This is the path for measuring bounds off a
        trained policy without distilling from it.
      * otherwise the student YAML's `config.distillation` block, which restates the
        teacher's architecture for the distillation run.
    """
    import yaml

    if teacher_config_path is not None:
        with open(teacher_config_path, encoding="utf-8") as f:
            params = yaml.safe_load(f)["params"]
        cfg = params["config"]
        return {
            "network": params["network"],
            "model_name": params["model"]["name"],
            "normalize_input": cfg.get("normalize_input", True),
            "normalize_value": cfg.get("normalize_value", True),
        }

    with open(config_path, encoding="utf-8") as f:
        params = yaml.safe_load(f)["params"]
    distill = params.get("config", {}).get("distillation")
    assert distill is not None, (
        f"{config_path} has no config.distillation block. Pass --teacher_config pointing "
        "at the ANN's own training YAML to measure bounds off a trained policy instead.")
    # model.name (e.g. continuous_a2c_logstd) lives at params.model, not in distillation.
    distill = dict(distill)
    distill.setdefault("model_name", params["model"]["name"])
    return distill


def _build_teacher_from_checkpoint(checkpoint_path, distill_cfg, obs_dim, action_dim, device):
    """Build the frozen ANN teacher (full rl_games wrapper, with its running_mean_std) from
    the student YAML's distillation block, so the dims live in exactly one place (the YAML).

    Importing the networks package registers the custom builders (e.g. 'mlp_actor_critic')
    so rl_games' ModelBuilder can resolve the teacher's `network.name`.
    """
    from ..networks import build_teacher  # noqa: F401 — import also registers the builders

    return build_teacher(
        teacher_network_cfg=distill_cfg["network"],
        model_name=distill_cfg["model_name"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        checkpoint_path=checkpoint_path,
        device=device,
        normalize_input=distill_cfg["normalize_input"],
        normalize_value=distill_cfg["normalize_value"],
    )


@torch.no_grad()
def _teacher_action(teacher, raw_obs):
    """Deterministic teacher action = clamp(mu, -1, 1), matching how rl_games' player
    drives the env at inference (it clamps the action to [-1, 1])."""
    res = teacher({
        "is_train": False,
        "prev_actions": None,
        "obs": raw_obs,
        "rnn_states": None,
    })
    return torch.clamp(res["mus"], -1.0, 1.0)


def _build_encoder(observation_bounds, device, pop_dim, threshold, num_steps):
    """The population encoder the student will use, built with these bounds."""
    from ..networks.snn.encoder import PopulationSpikeEncoder

    encoder = PopulationSpikeEncoder(
        obs_dim=len(observation_bounds),
        obs_bounds=observation_bounds,
        num_steps=num_steps,
        encoder_config={"pop_dim": pop_dim, "threshold": threshold},
    ).to(device)
    encoder.eval()
    return encoder


# Share of the rollout the diagnostics (encoder plots, silent-neuron check) run on.
# A fraction rather than a fixed count, so the picture scales with the collection instead
# of shrinking to a rounding error on a long run.
DIAGNOSTIC_FRACTION = 0.25


def _subsample(sample_arr, fraction=DIAGNOSTIC_FRACTION, seed=0, cap=None):
    """A RANDOM `fraction` of the rollout, never a contiguous slice.

    Rows are appended per step (num_envs at a time), so sample_arr[:n] is the first
    n/num_envs STEPS — the post-reset transient, not the episode. On hover that slice has
    ~2.8x the std of the full rollout, which made the diagnostics describe a different
    distribution than the bounds were computed from.

    Seeded, so the plot and the silent-neuron check see the SAME rows and their verdicts
    can be read against each other.
    """
    n = max(1, int(len(sample_arr) * fraction))
    if cap is not None:
        n = min(n, cap)
    if n >= len(sample_arr):
        return sample_arr
    idx = np.random.default_rng(seed).choice(len(sample_arr), n, replace=False)
    return sample_arr[np.sort(idx)]


def _check_silent_neurons(observation_bounds, sample_arr, device,
                          pop_dim=10, threshold=0.95, num_steps=5,
                          fraction=DIAGNOSTIC_FRACTION):
    """Feed a batch of (normalized) obs through the encoder and report any column that
    never spikes across the batch — those are dead inputs to the actor.

    Encoder hyperparameters only affect this diagnostic, not the cached bounds.
    """
    encoder = _build_encoder(observation_bounds, device, pop_dim, threshold, num_steps)
    obs_dim = len(observation_bounds)

    batch = torch.as_tensor(_subsample(sample_arr, fraction), dtype=torch.float, device=device)
    print(f"[silent-neuron check] {len(batch)} samples "
          f"({fraction:.0%} of {len(sample_arr)}, randomly drawn)")
    with torch.no_grad():
        spikes = encoder(batch)  # [B, obs_dim*pop_dim, num_steps]

    # A column is silent if it never spikes across the whole batch and all steps.
    col_active = spikes.sum(dim=(0, 2)) > 0  # [obs_dim*pop_dim]
    silent = (~col_active).view(obs_dim, pop_dim)
    silent_dims = torch.nonzero(silent.any(dim=1), as_tuple=False).flatten().tolist()

    total_silent = int((~col_active).sum().item())
    if total_silent == 0:
        print(f"[silent-neuron check] OK: all {obs_dim * pop_dim} encoder columns "
              f"spike across the batch.")
    else:
        print(f"[silent-neuron check] WARNING: {total_silent}/{obs_dim * pop_dim} "
              f"encoder columns never spiked. Affected obs dims: {silent_dims}. "
              "Consider widening those bounds or lowering the encoder threshold.")


def _plot_bounds_encoder(observation_bounds, sample_arr, device, save_dir, task_config,
                         pop_dim=10, threshold=0.95, num_steps=5,
                         fraction=DIAGNOSTIC_FRACTION):
    """Render the encoder receptive fields + activations these bounds define, and save the
    PNGs into save_dir — so the clamping can be inspected BEFORE the bounds reach the
    student. Best-effort: any failure is logged, not raised.
    """
    try:
        from .plot_encoder_trace import plot_encoder_trace
    except Exception as e:
        print(f"[bounds-plot] import failed ({e}); skipping encoder plots.")
        return

    encoder = _build_encoder(observation_bounds, device, pop_dim, threshold, num_steps)
    encoder.record = True
    encoder._trace = []

    batch = torch.as_tensor(_subsample(sample_arr, fraction), dtype=torch.float, device=device)
    print(f"[bounds-plot] {len(batch)} samples "
          f"({fraction:.0%} of {len(sample_arr)}, randomly drawn)")
    with torch.no_grad():
        encoder(batch)  # records one entry covering the full collected distribution

    try:
        os.makedirs(save_dir, exist_ok=True)
        plot_encoder_trace(encoder, encoder._trace,
                           task_config.observation_layout, save_dir=save_dir)
        print(f"[bounds-plot] encoder receptive-field/activation PNGs saved to: {save_dir}")
    except Exception:
        import traceback
        print("[bounds-plot] plotting failed (non-fatal):")
        traceback.print_exc()


def _read_encoder_cfg(config_path):
    """Encoder hyperparameters from a student YAML's actor block, for the diagnostics
    above. Falls back to the PopSAN defaults."""
    pop_dim, threshold, num_steps = 10, 0.95, 5
    if config_path is None:
        return pop_dim, threshold, num_steps
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            actor = yaml.safe_load(f)["params"]["network"]["actor"]
        return actor["encoder"]["pop_dim"], actor["encoder"]["threshold"], actor["num_steps"]
    except Exception as e:
        print(f"[bounds-plot] couldn't read encoder cfg from {config_path} ({e}); "
              "using PopSAN defaults.")
        return pop_dim, threshold, num_steps


def collect(num_steps, num_envs, out_dir, use_wandb, teacher_checkpoint=None,
            config_path=None, bounds_cache=DEFAULT_BOUNDS_CACHE, min_episodes=0,
            lower_pct=LOWER_PCT, upper_pct=UPPER_PCT, curriculum_level=25,
            bound_method=DEFAULT_BOUND_METHOD, task_name=None, teacher_config_path=None,
            plot_bounds=None, max_episode_steps=None):
    """Run a rollout, compute per-dim stats, write the CSV + bounds cache, return the bounds."""
    os.makedirs(out_dir, exist_ok=True)

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="aerial_gym",
                name=f"obs_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=["obs_stats"],
                sync_tensorboard=True,
            )
        except Exception as e:
            print(f"W&B init failed (continuing without W&B): {e}")

    task_name, task_config = resolve_task(config_path, task_name)

    # Shorten the episode so the sample is not dominated by steady-state hover.
    #
    # A converged policy spends ~90% of a 667-step episode parked on the target, so the
    # pooled distribution is that one narrow mode plus a thin approach tail — and every
    # mass-weighted estimator (mean/std, percentiles) is dragged onto the mode. Bounds fit
    # to it clip the approach hard, which is exactly when the policy needs to see where it
    # is. Truncating restarts the env on a fresh target instead, so the same budget buys
    # many approaches over diverse start/target pairs.
    #
    # Done through the task's OWN episode length rather than by resetting envs from here:
    # an external reset_idx leaves cur_obs stale for one step and bypasses the task's
    # episode bookkeeping.
    if max_episode_steps is not None:
        original = getattr(task_config, "episode_len_steps", None)
        task_config.episode_len_steps = max_episode_steps
        print(f"[obs-stats] episode truncated to {max_episode_steps} steps "
              f"(task default {original}) — sampling approaches, not hover dwell")
    print(f"[obs-stats] task: {task_name} "
          f"({task_config.observation_space_dim}-D obs)")
    task_config.num_envs = num_envs

    # Pin the obstacle-density curriculum to the teacher's level BEFORE make_task. The task
    # constructor inits curriculum_level = curriculum.min_level and spawns the obstacle field
    # from it, so setting min == max == level here makes the env born at the right difficulty,
    # and the task's own clamp holds it fixed (no drift with success).
    has_curriculum = hasattr(task_config, "curriculum")
    if curriculum_level is not None and curriculum_level >= 0 and has_curriculum:
        task_config.curriculum.min_level = curriculum_level
        task_config.curriculum.max_level = curriculum_level
        print(f"[obs-stats] curriculum pinned at level {curriculum_level} before make_task")
    elif not has_curriculum:
        curriculum_level = None  # recorded in the cache; this task has no curriculum

    # Warp is only needed by a task with cameras; forcing it on a camera-less task
    # (hover) costs a renderer it never reads.
    task = task_registry.make_task(
        task_name, num_envs=num_envs, headless=True,
        use_warp=getattr(task_config, "use_warp", True),
    )
    if has_curriculum:
        print(f"[obs-stats] env born at curriculum level {task.curriculum_level}")

    obs_dim = task_config.observation_space_dim
    action_dim = task_config.action_space_dim
    device = task_config.device
    obs_names = _obs_names(obs_dim, task_config)

    teacher = None
    if teacher_checkpoint is not None:
        assert os.path.exists(teacher_checkpoint), \
            f"teacher_checkpoint not found: {teacher_checkpoint}"
        assert config_path is not None or teacher_config_path is not None, \
            "--config (student YAML) or --teacher_config (the ANN's own YAML) is required " \
            "with --teacher_checkpoint, to read the teacher network architecture"
        distill_cfg = _load_teacher_cfg(config_path, teacher_config_path)
        teacher = _build_teacher_from_checkpoint(
            teacher_checkpoint, distill_cfg, obs_dim, action_dim, device)
        print(f"Teacher-driven rollout using: {teacher_checkpoint}")
    else:
        print("Random-action rollout (no teacher checkpoint).")

    all_obs = []  # list of [num_envs, obs_dim] raw obs tensors (cpu)

    obs_dict = task.reset()
    # task.reset() may return a dict or a tuple depending on the wrapper path.
    cur_obs = obs_dict["observations"] if isinstance(obs_dict, dict) else obs_dict[0]["observations"]
    steps_collected = 0
    episodes_done = 0  # cumulative completed episodes (termination OR truncation)

    driver = "teacher" if teacher is not None else "random"
    # Stopping rule: run until BOTH the step floor (num_steps) AND the episode floor
    # (min_episodes) are met. min_episodes=0 disables the episode floor. Collecting whole
    # episodes gives the bounds the TRUE state distribution (start-to-end of each
    # trajectory), not just the post-reset slice a short step budget would sample.
    print(f"Collecting >= {num_steps} steps AND >= {min_episodes} episodes with {num_envs} "
          f"envs ({driver}-driven)...")

    while steps_collected < num_steps or episodes_done < min_episodes:
        if teacher is not None:
            actions = _teacher_action(teacher, cur_obs)
        else:
            actions = torch.rand(num_envs, action_dim, device=device) * 2.0 - 1.0

        obs_dict, rewards, terminations, truncations, infos = task.step(actions)
        cur_obs = obs_dict["observations"]

        all_obs.append(cur_obs.clone().cpu())
        steps_collected += num_envs
        # An env's episode ended this step if it terminated (arrive/crash/exceed) or truncated.
        episodes_done += int((terminations.bool() | truncations.bool()).sum().item())

        if steps_collected % max(num_envs * 10, 1000) == 0:
            print(f"  {steps_collected} steps / {episodes_done} episodes collected "
                  f"(targets: {num_steps} steps, {min_episodes} episodes)")

    task.close()
    print(f"Stopped at {steps_collected} steps / {episodes_done} episodes.")

    all_obs = torch.cat(all_obs, dim=0)  # [total_steps, obs_dim] (cpu, raw)
    print(f"Total observations collected: {tuple(all_obs.shape)}")

    # --- Build the array used for ENCODER BOUNDS ----------------------------
    # Bounds live in normalized space. If a teacher is available, normalize raw obs through
    # its frozen running_mean_std (the same stats the warm-started student starts from);
    # otherwise fall back to raw space and warn.
    if teacher is not None:
        rms = teacher.running_mean_std
        mean = rms.running_mean.detach().cpu().float()
        var = rms.running_var.detach().cpu().float()
        norm_obs = torch.clamp((all_obs - mean) / torch.sqrt(var + 1e-5), -5.0, 5.0)
        bounds_space = "normalized (teacher running_mean_std)"
        bounds_arr = norm_obs.numpy()
    else:
        print("WARNING: no teacher checkpoint — emitting RAW-space bounds. The PopSAN "
              "encoder expects NORMALIZED bounds; pass --teacher_checkpoint.")
        bounds_space = "raw (NO normalization — likely wrong for the encoder)"
        bounds_arr = all_obs.numpy()

    raw_np = all_obs.numpy()

    # Stats in the space the ENCODER BOUNDS live in. The raw table below is for physical
    # sanity-checking; this is the one you read to choose a window.
    zstats = {
        "z_min": bounds_arr.min(axis=0), "z_max": bounds_arr.max(axis=0),
        "z_mean": bounds_arr.mean(axis=0), "z_std": bounds_arr.std(axis=0),
        "z_p01": np.percentile(bounds_arr, 1, axis=0),
        "z_p99": np.percentile(bounds_arr, 99, axis=0),
    }

    # --- Per-dim statistics (raw space, for inspection) ---------------------
    stats = {
        "dim":  list(range(obs_dim)),
        "name": obs_names,
        "min":  raw_np.min(axis=0).tolist(),
        "max":  raw_np.max(axis=0).tolist(),
        "mean": raw_np.mean(axis=0).tolist(),
        "std":  raw_np.std(axis=0).tolist(),
        "p01":  np.percentile(raw_np, 1,  axis=0).tolist(),
        "p05":  np.percentile(raw_np, 5,  axis=0).tolist(),
        "p50":  np.percentile(raw_np, 50, axis=0).tolist(),
        "p95":  np.percentile(raw_np, 95, axis=0).tolist(),
        "p99":  np.percentile(raw_np, 99, axis=0).tolist(),
    }

    # --- Encoder bounds (bounds space), by the selected method --------------
    lo, hi = _compute_bounds(bounds_arr, bound_method, lower_pct, upper_pct)
    print(f"[obs-stats] encoder bounds method = {bound_method}"
          + (f" (z={_gaussian_z(upper_pct):.4f} for {upper_pct}%)" if bound_method == "gaussian"
             else f" (p{lower_pct}/p{upper_pct})"))

    # Guard against degenerate (lo == hi) dims — a flat dim would give a zero-width encoder
    # range and silent neurons. Pad to a small symmetric band.
    flat = (hi - lo) < 1e-4
    if flat.any():
        pad = 0.1
        lo = np.where(flat, lo - pad, lo)
        hi = np.where(flat, hi + pad, hi)
    observation_bounds = [(round(float(l), 4), round(float(h), 4)) for l, h in zip(lo, hi)]

    # --- Diagnostics on the bounds we're about to cache ---------------------
    enc_pop_dim, enc_threshold, enc_num_steps = _read_encoder_cfg(config_path)
    _check_silent_neurons(observation_bounds, bounds_arr, device,
                          pop_dim=enc_pop_dim, threshold=enc_threshold,
                          num_steps=enc_num_steps)
    # The plot window is decoupled from the computed bounds on purpose: the histogram is
    # drawn with range=(lo, hi), so plotting against the very window under evaluation hides
    # the tails outside it. --plot_bounds pins a fixed, wider frame (e.g. RunningMeanStd's
    # own [-5, 5]) so you can SEE what a candidate window would clip. Only the picture
    # changes; the cached bounds are always the computed ones.
    plot_window = observation_bounds
    if plot_bounds is not None:
        plot_window = [tuple(plot_bounds)] * obs_dim
        print(f"[bounds-plot] plotting against a fixed window {tuple(plot_bounds)} "
              f"(cached bounds are unchanged)")
    _plot_bounds_encoder(plot_window, bounds_arr, device, out_dir, task_config,
                         pop_dim=enc_pop_dim, threshold=enc_threshold,
                         num_steps=enc_num_steps)

    # --- Save CSV ------------------------------------------------------------
    csv_path = os.path.join(out_dir, "obs_stats.csv")
    with open(csv_path, "w", newline="") as f:
        zkeys = ["z_min", "z_p01", "z_mean", "z_p99", "z_max", "z_std"]
        writer = csv.DictWriter(f, fieldnames=["dim", "name", "min", "max", "mean",
                                               "std", "p01", "p05", "p50", "p95", "p99"]
                                              + zkeys + ["bound_lo", "bound_hi", "pct_clipped"])
        writer.writeheader()
        for i in range(obs_dim):
            row = {k: stats[k][i] for k in stats}
            row.update({k: float(zstats[k][i]) for k in zkeys})
            row["bound_lo"], row["bound_hi"] = observation_bounds[i]
            row["pct_clipped"] = round(float(
                ((bounds_arr[:, i] < observation_bounds[i][0]) |
                 (bounds_arr[:, i] > observation_bounds[i][1])).mean() * 100), 3)
            writer.writerow(row)
    print(f"CSV saved to: {csv_path}")

    # --- Write the bounds cache the student runner reads --------------------
    cache_payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "obs_dim": obs_dim,
        "bound_method": bound_method,
        "percentiles": [lower_pct, upper_pct],
        "curriculum_level": curriculum_level,
        "space": bounds_space,
        "teacher_checkpoint": teacher_checkpoint,
        "task": task_name,
        "use_vae": getattr(getattr(task_config, "vae_config", None), "use_vae", False),
        "latent_dims": getattr(getattr(task_config, "vae_config", None), "latent_dims", 0),
        "observation_bounds": observation_bounds,
    }
    os.makedirs(os.path.dirname(bounds_cache), exist_ok=True)
    with open(bounds_cache, "w") as f:
        json.dump(cache_payload, f, indent=2)
    print(f"Bounds cache written to: {bounds_cache}")

    # --- TensorBoard histograms ---------------------------------------------
    tb_dir = os.path.join(out_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    for i in range(obs_dim):
        name = obs_names[i]
        writer.add_histogram(f"obs_dist/{name}", raw_np[:, i], global_step=0)
        for key in ("mean", "std", "min", "max", "p01", "p99"):
            writer.add_scalar(f"obs_stats/{key}/{name}", float(stats[key][i]), global_step=0)
    writer.close()
    print(f"TensorBoard logs saved to: {tb_dir}")

    if wandb_run is not None:
        import wandb
        wandb.log({f"obs_hist/{obs_names[i]}": wandb.Histogram(raw_np[:, i])
                   for i in range(obs_dim)}, step=0)
        wandb.finish()
        print("W&B histograms logged.")

    # --- Print summary + ready-to-paste bounds ------------------------------
    print()
    print(f"RAW space (physical units — for sanity-checking, NOT for setting bounds)")
    print(f"{'dim':<5} {'name':<22} {'min':>8} {'p01':>8} {'mean':>8} {'p99':>8} {'max':>8} {'std':>8}")
    print("-" * 80)
    for i in range(obs_dim):
        print(f"{i:<5} {obs_names[i]:<22} "
              f"{stats['min'][i]:>8.3f} {stats['p01'][i]:>8.3f} "
              f"{stats['mean'][i]:>8.3f} {stats['p99'][i]:>8.3f} "
              f"{stats['max'][i]:>8.3f} {stats['std'][i]:>8.3f}")

    # The table to actually read: observation_bounds are set in THIS space.
    print()
    print(f"NORMALIZED space — {bounds_space}")
    print("This is the space observation_bounds live in. z_min/z_max are the extremes the")
    print("encoder will ever see; a window covering them clips nothing. %clip is what the")
    print(f"proposed {bound_method} window below would discard.")
    print(f"{'dim':<5} {'name':<22} {'z_min':>8} {'z_p01':>8} {'z_p99':>8} {'z_max':>8} "
          f"{'z_std':>7} | {'bound_lo':>9} {'bound_hi':>9} {'%clip':>7}")
    print("-" * 104)
    for i in range(obs_dim):
        lo, hi = observation_bounds[i]
        clipped = ((bounds_arr[:, i] < lo) | (bounds_arr[:, i] > hi)).mean() * 100
        flag = "  <-- clips >1%" if clipped > 1.0 else ""
        print(f"{i:<5} {obs_names[i]:<22} "
              f"{zstats['z_min'][i]:>8.3f} {zstats['z_p01'][i]:>8.3f} "
              f"{zstats['z_p99'][i]:>8.3f} {zstats['z_max'][i]:>8.3f} "
              f"{zstats['z_std'][i]:>7.3f} | {lo:>9.3f} {hi:>9.3f} {clipped:>6.2f}%{flag}")

    # YAML, indented for a config's `network.actor` block — the only place these are
    # consumed. Paste over the existing observation_bounds rows.
    print(f"\n# {bound_method} bounds in {bounds_space}")
    print(f"# paste into {config_path or '<student yaml>'} under network.actor:")
    print("        observation_bounds:")
    width = max(len(n) for n in obs_names)
    for i, (l, h) in enumerate(observation_bounds):
        entry = f"[{l}, {h}]"
        print(f"          - {entry:<22}# {i:>2} {obs_names[i]:<{width}}")

    return observation_bounds


# ---------------------------------------------------------------------------
# Cache access — what the runner calls
# ---------------------------------------------------------------------------


def _load_valid_cache(cache_path, obs_dim, teacher_checkpoint, task_name=None):
    """Return the cached bounds, or None if the cache is missing, unreadable, or stale.

    A cache is stale if it was built for a different observation size or a different
    teacher — either makes its normalized bounds meaningless for this run.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path) as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[obs-bounds] cache unreadable ({e}); will recompute.")
        return None

    if payload.get("obs_dim") != obs_dim or \
            len(payload.get("observation_bounds", [])) != obs_dim:
        print(f"[obs-bounds] cache obs_dim mismatch "
              f"({payload.get('obs_dim')} != {obs_dim}); will recompute.")
        return None

    if task_name is not None and payload.get("task") not in (None, task_name):
        print(f"[obs-bounds] cache was built for task {payload.get('task')!r} "
              f"(need {task_name!r}); will recompute.")
        return None

    if payload.get("teacher_checkpoint") != teacher_checkpoint:
        print(f"[obs-bounds] cache was built for a different teacher "
              f"({payload.get('teacher_checkpoint')!r} != {teacher_checkpoint!r}); "
              "will recompute.")
        return None

    print(f"[obs-bounds] loaded cached bounds from {cache_path} "
          f"(created {payload.get('created')}, space={payload.get('space')})")
    return [tuple(b) for b in payload["observation_bounds"]]


def load_or_collect_bounds(teacher_checkpoint, config_path, num_envs=64, num_steps=10000,
                           recompute=False, cache_path=DEFAULT_BOUNDS_CACHE,
                           min_episodes=0, out_dir=None, curriculum_level=25,
                           bound_method=DEFAULT_BOUND_METHOD, task_name=None):
    """Return per-dim encoder bounds for `teacher_checkpoint`, collecting them if needed.

    Collection runs in a SEPARATE subprocess: Isaac Gym does not support creating a second
    sim in a process, and the caller (the runner) is about to create its own. The subprocess
    writes the JSON cache; this loads it back.

    Raises RuntimeError if collection fails — training a student on wrong encoder bounds
    silently produces a policy whose inputs are clipped in the wrong places.
    """
    task_name, task_config = resolve_task(config_path, task_name)
    obs_dim = task_config.observation_space_dim

    bounds = None if recompute else _load_valid_cache(
        cache_path, obs_dim, teacher_checkpoint, task_name)
    if bounds is None:
        print(f"[obs-bounds] collecting bounds in a subprocess "
              f"(teacher={teacher_checkpoint}, steps={num_steps}, "
              f"episodes={min_episodes}, envs={num_envs})")
        cmd = [
            sys.executable, "-m", MODULE_PATH,
            f"--teacher_checkpoint={teacher_checkpoint}",
            f"--config={config_path}",
            f"--num_steps={num_steps}",
            f"--num_envs={num_envs}",
            f"--bounds_cache={cache_path}",
            f"--curriculum_level={curriculum_level}",
            f"--task={task_name}",
            f"--bound_method={bound_method}",
            "--no_wandb",
        ]
        if min_episodes:
            cmd.append(f"--min_episodes={min_episodes}")
        if out_dir:
            cmd.append(f"--out_dir={out_dir}")

        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(
                f"[obs-bounds] collection subprocess failed (exit {result.returncode}); "
                "fix the teacher checkpoint / collector before training the student.")

        bounds = _load_valid_cache(cache_path, obs_dim, teacher_checkpoint, task_name)
        if bounds is None:
            raise RuntimeError("[obs-bounds] subprocess finished but produced no valid cache.")

    assert len(bounds) == obs_dim, \
        f"[obs-bounds] got {len(bounds)} bounds, expected {obs_dim}"
    return bounds


def _parse_args():
    parser = argparse.ArgumentParser(description="Collect observation statistics")
    parser.add_argument("--num_steps", type=int, default=10000,
                        help="Minimum total env-steps to collect (summed over envs).")
    parser.add_argument("--min_episodes", type=int, default=0,
                        help="Also keep collecting until at least this many episodes have "
                             "COMPLETED (termination or truncation), for true start-to-end "
                             "state statistics. Stops when both --num_steps and this are met. "
                             "0 = step-count only.")
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--out_dir", type=str, default=OBS_STATS_DIR,
                        help="Output directory for CSV/TensorBoard logs")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Drive the rollout with the ANN teacher (clamped mu) and "
                             "normalize bounds through its running_mean_std. If omitted, "
                             "uses random actions and raw-space bounds.")
    parser.add_argument("--config", type=str, default=None,
                        help="Student YAML. Required with --teacher_checkpoint: the teacher "
                             "network architecture and normalization are read from its "
                             "config.distillation block (single source of truth).")
    parser.add_argument("--task", type=str, default=None,
                        help="Task to collect against. Defaults to the --config YAML's "
                             "config.env_name, so it normally needs no setting.")
    parser.add_argument("--teacher_config", type=str, default=None,
                        help="The ANN teacher's OWN training YAML (e.g. ppo_hover_local). "
                             "Use this to measure bounds off a trained policy when the "
                             "student YAML has no config.distillation block.")
    parser.add_argument("--bounds_cache", type=str, default=DEFAULT_BOUNDS_CACHE,
                        help="Where to write the JSON bounds cache the runner reads")
    parser.add_argument("--lower_pct", type=float, default=LOWER_PCT,
                        help=f"Lower percentile for encoder bounds (default {LOWER_PCT}). "
                             "Lower => wider clamp, captures more of the left tail.")
    parser.add_argument("--upper_pct", type=float, default=UPPER_PCT,
                        help=f"Upper percentile for encoder bounds (default {UPPER_PCT}). "
                             "Higher => wider clamp, captures more of the right tail.")
    parser.add_argument("--curriculum_level", type=int, default=25,
                        help="Pin the obstacle-density curriculum at this level (the "
                             "teacher's final level = 25) so bounds reflect the world the "
                             "student is deployed in. <0 leaves the task default.")
    parser.add_argument("--bound_method", type=str, default=DEFAULT_BOUND_METHOD,
                        choices=["gaussian", "percentile"],
                        help="How encoder clamp bounds are derived (default "
                             f"{DEFAULT_BOUND_METHOD}). 'gaussian': mu +/- z*sigma per dim, "
                             "centered on the center of mass, symmetric. 'percentile': "
                             "empirical p{lower}/p{upper}, follows asymmetric tails. The "
                             "coverage band (--upper_pct) sets z for the gaussian method too.")
    parser.add_argument("--max_episode_steps", type=int, default=None,
                        help="Truncate episodes to this many steps during collection. A "
                             "converged policy dwells in hover for most of a full episode, "
                             "which drags mean/std and percentiles onto that one narrow "
                             "mode; truncating resamples the approach across many fresh "
                             "targets instead. Try ~150 for hover (default 667).")
    parser.add_argument("--plot_bounds", type=str, default=None, metavar="LO,HI",
                        help="Draw the encoder plots against this FIXED window instead of "
                             "the computed bounds, e.g. --plot_bounds=-5,5 (RunningMeanStd's "
                             "own clamp range). The histogram is clipped to the plotted "
                             "window, so a wide frame is what shows you the tails a "
                             "candidate window would cut. Does not change the cached bounds.")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb",
                        help="Disable W&B logging (default: enabled)")
    parser.set_defaults(wandb=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    collect(
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        out_dir=args.out_dir,
        use_wandb=args.wandb,
        teacher_checkpoint=args.teacher_checkpoint,
        config_path=args.config,
        bounds_cache=args.bounds_cache,
        min_episodes=args.min_episodes,
        lower_pct=args.lower_pct,
        upper_pct=args.upper_pct,
        curriculum_level=args.curriculum_level,
        bound_method=args.bound_method,
        task_name=args.task,
        teacher_config_path=args.teacher_config,
        plot_bounds=([float(v) for v in args.plot_bounds.split(",")]
                     if args.plot_bounds else None),
        max_episode_steps=args.max_episode_steps,
    )
