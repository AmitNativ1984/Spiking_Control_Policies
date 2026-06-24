"""
Observation statistics collector for NavigationWithObstaclesTask.

Drives the environment for a fixed number of steps and computes per-dimension
statistics, saved to CSV and logged to TensorBoard/W&B.

Two rollout drivers:
  * random actions (default) — uniform in [-1, 1].
  * teacher-driven (--teacher_checkpoint) — actions = clamp(teacher_mu, -1, 1),
    matching how the warm-started student will actually be driven. Preferred for
    setting PopSAN encoder bounds (Phase 2) so the bounds reflect the on-policy
    state distribution, not the random one.

PopSAN encoder bounds live in rl_games-NORMALIZED space (z-scores, hard-clamped
to [-5, 5] by RunningMeanStd). When a teacher checkpoint is given we normalize the
collected raw obs through the teacher's frozen running_mean_std before computing
percentiles, so the emitted `observation_bounds` are directly usable by the
encoder. The VAE-latent dims are measured the same way from the rollout, so they
reflect the CURRENT VAE (not the teacher's), per the Phase 2 checklist.

The chosen percentile band (default p01/p99) is emitted as a ready-to-paste
`observation_bounds` list AND written to a JSON cache that the student runner
loads automatically at startup (see `ensure_observation_bounds`).

Usage:
    cd /workspaces/aerial_gym_docker
    python -m navigation_with_obstacles.tools.collect_obs_stats \
        --num_steps=10000 --num_envs=64 \
        --teacher_checkpoint=navigation_with_obstacles/runs/.../nn/teacher.pth \
        --out_dir=navigation_with_obstacles/obs_stats
"""
import isaacgym  # must be first

import argparse
import json
import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, "/workspaces/aerial_gym_docker")

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots import BaseMultirotor

from navigation_with_obstacles.task.navigation_task import NavigationWithObstaclesTask
from navigation_with_obstacles.config.task_config import task_config
from navigation_with_obstacles.config.env_config import NavigationObstacleEnvCfg
from navigation_with_obstacles.config.robot_config import NavQuadWithCameraCfg

# Register components (idempotent — safe if the runner already registered them)
env_config_registry.register("navigation_obstacle_env", NavigationObstacleEnvCfg)
robot_registry.register("nav_quadrotor_with_camera", BaseMultirotor, NavQuadWithCameraCfg)
task_registry.register_task("navigation_with_obstacles_task", NavigationWithObstaclesTask, task_config)

# Default location of the cached bounds the student runner reads at startup.
DEFAULT_BOUNDS_CACHE = "navigation_with_obstacles/obs_stats/observation_bounds.json"

# Percentile band used for encoder bounds (Phase 2 chose p01/p99).
LOWER_PCT = 1.0
UPPER_PCT = 99.0


def _obs_names(obs_dim: int):
    """Names per obs dim, derived from the task's observation_layout so they stay
    in sync with the real vector (state dims + however many VAE latents)."""
    names = [None] * obs_dim
    for obj_slice, obj_type in task_config.observation_layout:
        span = range(obj_slice.start, obj_slice.stop)
        for k, idx in enumerate(span):
            names[idx] = f"{obj_type}_{k}" if (obj_slice.stop - obj_slice.start) > 1 else obj_type
    return [n if n is not None else f"dim_{i}" for i, n in enumerate(names)]


def _load_distillation_cfg(config_path):
    """Read the `params.config.distillation` block from a student YAML. This is the
    SINGLE source of truth for the teacher's architecture / normalization, so the
    collector never duplicates the teacher network dims."""
    import yaml
    with open(config_path) as f:
        params = yaml.safe_load(f)["params"]
    distill = params.get("config", {}).get("distillation")
    assert distill is not None, f"{config_path} has no config.distillation block"
    # model.name (e.g. continuous_a2c_logstd) lives at params.model, not in distillation.
    distill = dict(distill)
    distill.setdefault("model_name", params["model"]["name"])
    return distill


def _build_teacher_from_checkpoint(checkpoint_path, distill_cfg, obs_dim, action_dim, device):
    """Build the frozen ANN teacher (full rl_games wrapper, with its running_mean_std)
    from the student YAML's distillation block. Returns the model.

    `distill_cfg` is the parsed `config.distillation` dict (see _load_distillation_cfg);
    its `network` sub-block defines the teacher architecture, so the dims live in
    exactly one place (the YAML). We mirror runner.py's network registration so
    ModelBuilder can resolve the custom 'mlp_actor_critic' builder.
    """
    from rl_games.algos_torch import model_builder
    from navigation_with_obstacles.networks.ann.actor_critic import (
        MLPActorCriticNetworkBuilder,
    )
    from navigation_with_obstacles.networks.teacher_student.teacher_builder import (
        build_teacher,
    )

    model_builder.register_network("mlp_actor_critic", MLPActorCriticNetworkBuilder)

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
    """Deterministic teacher action = clamp(mu, -1, 1), matching how rl_games'
    player drives the env at inference (it clamps the action to [-1, 1])."""
    res = teacher({
        "is_train": False,
        "prev_actions": None,
        "obs": raw_obs,
        "rnn_states": None,
    })
    return torch.clamp(res["mus"], -1.0, 1.0)


def _check_silent_neurons(observation_bounds, sample_arr, device,
                          pop_dim=10, threshold=0.95, num_steps=5, max_batch=4096):
    """Phase 2 bullet 6: build the population encoder with `observation_bounds`,
    feed a batch of (normalized) obs through it, and report any encoder column
    that never spikes across the batch — those are dead inputs to the actor.

    Encoder hyperparameters default to the PopSAN YAML's encoder block; they only
    affect this diagnostic, not the cached bounds.
    """
    from navigation_with_obstacles.networks.snn.encoder import PopulationSpikeEncoder

    obs_dim = len(observation_bounds)
    encoder = PopulationSpikeEncoder(
        obs_dim=obs_dim,
        obs_bounds=observation_bounds,
        num_steps=num_steps,
        encoder_config={"pop_dim": pop_dim, "threshold": threshold},
    ).to(device)
    encoder.eval()

    batch = torch.as_tensor(sample_arr[:max_batch], dtype=torch.float, device=device)
    with torch.no_grad():
        spikes = encoder(batch)  # [B, obs_dim*pop_dim, num_steps]
    # A column is silent if it never spikes across the whole batch and all steps.
    col_active = (spikes.sum(dim=(0, 2)) > 0)  # [obs_dim*pop_dim]
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


def _plot_teacher_bounds_encoder(observation_bounds, sample_arr, device, save_dir,
                                 pop_dim=10, threshold=0.95, num_steps=5, max_batch=4096):
    """Render the population-encoder receptive fields + activations using the TEACHER-collected
    per-dim bounds, and save the PNGs into save_dir.

    This is the artifact the user wants: a visual of the clamping/encoding the bounds define,
    produced right after teacher collection and BEFORE these bounds are applied to the student
    SNN encoder. Builds the SAME PopulationSpikeEncoder the silent-neuron check uses (same
    bounds), records one forward over a batch of the collected (normalized) obs, and plots.
    Best-effort: any failure is logged, not raised.
    """
    try:
        from navigation_with_obstacles.networks.snn.encoder import PopulationSpikeEncoder
        from navigation_with_obstacles.tools.plot_encoder_trace import plot_encoder_trace
    except Exception as e:
        print(f"[bounds-plot] import failed ({e}); skipping encoder plots.")
        return

    obs_dim = len(observation_bounds)
    encoder = PopulationSpikeEncoder(
        obs_dim=obs_dim,
        obs_bounds=observation_bounds,
        num_steps=num_steps,
        encoder_config={"pop_dim": pop_dim, "threshold": threshold},
    ).to(device)
    encoder.eval()
    encoder.record = True
    encoder._trace = []

    batch = torch.as_tensor(sample_arr[:max_batch], dtype=torch.float, device=device)
    with torch.no_grad():
        encoder(batch)  # records one entry with B = batch rows (the full collected distribution)

    try:
        os.makedirs(save_dir, exist_ok=True)
        plot_encoder_trace(encoder, encoder._trace,
                           task_config.observation_layout, save_dir=save_dir)
        print(f"[bounds-plot] encoder receptive-field/activation PNGs saved to: {save_dir}")
    except Exception:
        import traceback
        print("[bounds-plot] plotting failed (non-fatal):")
        traceback.print_exc()


def collect(num_steps, num_envs, out_dir, use_wandb, teacher_checkpoint=None,
            config_path=None, bounds_cache=DEFAULT_BOUNDS_CACHE, min_episodes=0,
            lower_pct=LOWER_PCT, upper_pct=UPPER_PCT, curriculum_level=25):
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
            wandb_run = None

    task_config.num_envs = num_envs

    # Pin the obstacle-density curriculum to the teacher's level BEFORE make_task. The task
    # constructor inits curriculum_level = curriculum.min_level (navigation_task.py:147) and
    # spawns the obstacle field from it, so setting min==max==level here makes the env born at
    # the right difficulty — the navigation_task.py:255 init log then prints this level (no
    # misleading "level 0" line), and the task's own clamp holds it fixed (no drift with success).
    if curriculum_level is not None and curriculum_level >= 0:
        task_config.curriculum.min_level = curriculum_level
        task_config.curriculum.max_level = curriculum_level
        print(f"[obs-stats] curriculum pinned at level {curriculum_level} before make_task")

    task = task_registry.make_task(
        "navigation_with_obstacles_task",
        num_envs=num_envs,
        headless=True,
        use_warp=True,
    )

    # VAE fully ON (phase "C", gate 1.0) so bounds reflect the world the student is deployed in.
    # This MUST come after make_task: NavigationTask.__init__ unconditionally forces phase "A" /
    # gate 0.0 when use_vae=True (navigation_task.py:158-160), which we override here. The teacher
    # saw real VAE latents, so the collected obs distribution must include them ungated.
    if task_config.vae_config.use_vae:
        task.vae_phase = "C"
    task_config.vae_gate = 1.0
    print(f"[obs-stats] VAE ON (phase C, gate 1.0); env born at curriculum level "
          f"{task.curriculum_level}")

    obs_dim = task_config.observation_space_dim
    action_dim = task_config.action_space_dim
    device = task_config.device
    obs_names = _obs_names(obs_dim)

    teacher = None
    if teacher_checkpoint is not None:
        assert os.path.exists(teacher_checkpoint), \
            f"teacher_checkpoint not found: {teacher_checkpoint}"
        assert config_path is not None, \
            "--config (student YAML) is required with --teacher_checkpoint so the " \
            "teacher network architecture can be read from config.distillation.network"
        distill_cfg = _load_distillation_cfg(config_path)
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
    # (min_episodes) are met. min_episodes=0 disables the episode floor (step-count behavior).
    # Collecting whole episodes gives the bounds the TRUE state distribution (start-to-end of
    # each trajectory), not just the post-reset slice you'd get from a short step budget.
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
        done = (terminations.bool() | truncations.bool())
        episodes_done += int(done.sum().item())

        if steps_collected % max(num_envs * 10, 1000) == 0:
            print(f"  {steps_collected} steps / {episodes_done} episodes collected "
                  f"(targets: {num_steps} steps, {min_episodes} episodes)")

    task.close()
    print(f"Stopped at {steps_collected} steps / {episodes_done} episodes.")

    all_obs = torch.cat(all_obs, dim=0)  # [total_steps, obs_dim] (cpu, raw)
    print(f"Total observations collected: {tuple(all_obs.shape)}")

    # --- Build the array used for ENCODER BOUNDS ----------------------------
    # Bounds live in normalized space. If a teacher is available, normalize raw
    # obs through its frozen running_mean_std (the same stats the warm-started
    # student starts from); otherwise fall back to raw space and warn.
    if teacher is not None:
        rms = teacher.running_mean_std
        mean = rms.running_mean.detach().cpu().float()
        var = rms.running_var.detach().cpu().float()
        norm_obs = torch.clamp((all_obs - mean) / torch.sqrt(var + 1e-5), -5.0, 5.0)
        bounds_space = "normalized (teacher running_mean_std)"
        bounds_arr = norm_obs.numpy()
    else:
        print("WARNING: no teacher checkpoint — emitting RAW-space bounds. The "
              "PopSAN encoder expects NORMALIZED bounds; pass --teacher_checkpoint.")
        bounds_space = "raw (NO normalization — likely wrong for the encoder)"
        bounds_arr = all_obs.numpy()

    raw_np = all_obs.numpy()

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

    # --- Encoder bounds from the chosen percentile band (bounds space) ------
    lo = np.percentile(bounds_arr, lower_pct, axis=0)
    hi = np.percentile(bounds_arr, upper_pct, axis=0)
    # Guard against degenerate (lo == hi) dims — a flat dim would give a zero-width
    # encoder range and silent neurons. Pad to a small symmetric band.
    flat = (hi - lo) < 1e-4
    if flat.any():
        pad = 0.1
        lo = np.where(flat, lo - pad, lo)
        hi = np.where(flat, hi + pad, hi)
    observation_bounds = [(round(float(l), 4), round(float(h), 4)) for l, h in zip(lo, hi)]

    # --- Silent-neuron check (Phase 2 bullet 6) -----------------------------
    # Build the population encoder with the NEW bounds and feed a batch of the
    # (normalized) obs through it: every encoder column must produce at least one
    # spike somewhere in the batch, or those neurons are dead inputs to the actor.
    _check_silent_neurons(observation_bounds, bounds_arr, device)

    # --- Encoder plots with the TEACHER bounds (before they reach the student) ----
    # Save receptive-field / activation PNGs so the per-dim clamping the bounds define can
    # be inspected before being applied to the student SNN encoder. Encoder hyperparameters
    # come from the student YAML's actor.encoder block when available (else PopSAN defaults).
    enc_pop_dim, enc_threshold, enc_num_steps = 10, 0.95, 5
    if config_path is not None:
        try:
            import yaml as _yaml
            with open(config_path) as _f:
                _actor = _yaml.safe_load(_f)["params"]["network"]["actor"]
            enc_pop_dim = _actor["encoder"]["pop_dim"]
            enc_threshold = _actor["encoder"]["threshold"]
            enc_num_steps = _actor["num_steps"]
        except Exception as _e:
            print(f"[bounds-plot] couldn't read encoder cfg from {config_path} ({_e}); "
                  "using PopSAN defaults.")
    _plot_teacher_bounds_encoder(observation_bounds, bounds_arr, device, out_dir,
                                 pop_dim=enc_pop_dim, threshold=enc_threshold,
                                 num_steps=enc_num_steps)

    # --- Save CSV ------------------------------------------------------------
    csv_path = os.path.join(out_dir, "obs_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "name", "min", "max", "mean",
                                               "std", "p01", "p05", "p50", "p95", "p99"])
        writer.writeheader()
        for i in range(obs_dim):
            writer.writerow({k: stats[k][i] for k in stats})
    print(f"CSV saved to: {csv_path}")

    # --- Write the bounds cache the student runner reads --------------------
    cache_payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "obs_dim": obs_dim,
        "percentiles": [lower_pct, upper_pct],
        "space": bounds_space,
        "teacher_checkpoint": teacher_checkpoint,
        "use_vae": task_config.vae_config.use_vae,
        "latent_dims": task_config.vae_config.latent_dims if task_config.vae_config.use_vae else 0,
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
        log_dict = {f"obs_hist/{obs_names[i]}": wandb.Histogram(raw_np[:, i])
                    for i in range(obs_dim)}
        wandb.log(log_dict, step=0)
        wandb.finish()
        print("W&B histograms logged.")

    # --- Print summary + ready-to-paste bounds ------------------------------
    print()
    print(f"{'dim':<5} {'name':<22} {'min':>8} {'p01':>8} {'mean':>8} {'p99':>8} {'max':>8} {'std':>8}")
    print("-" * 80)
    for i in range(obs_dim):
        print(f"{i:<5} {obs_names[i]:<22} "
              f"{stats['min'][i]:>8.3f} {stats['p01'][i]:>8.3f} "
              f"{stats['mean'][i]:>8.3f} {stats['p99'][i]:>8.3f} "
              f"{stats['max'][i]:>8.3f} {stats['std'][i]:>8.3f}")

    print(f"\n# observation_bounds from p{lower_pct:g}/p{upper_pct:g} in {bounds_space}:")
    print("observation_bounds = [")
    for i, (l, h) in enumerate(observation_bounds):
        print(f"    ({l}, {h}),   # {i:>2} {obs_names[i]}")
    print("]")

    return observation_bounds


def ensure_observation_bounds(teacher_checkpoint, config_path, num_steps=10000,
                              num_envs=64, bounds_cache=DEFAULT_BOUNDS_CACHE,
                              recompute=False):
    """Make sure task_config.observation_bounds is set from collected stats, then
    return them. Called by the student runner BEFORE the network is built.

    If a valid cache exists (matching obs_dim) and recompute is False, it is loaded
    without running a rollout. Otherwise a teacher-driven collection runs once and
    writes the cache. The latents are measured from the live env (current VAE), so
    no separate VAE pass is needed.
    """
    obs_dim = task_config.observation_space_dim
    bounds = None

    if not recompute and os.path.exists(bounds_cache):
        try:
            with open(bounds_cache) as f:
                payload = json.load(f)
            if payload.get("obs_dim") == obs_dim and \
               len(payload.get("observation_bounds", [])) == obs_dim:
                bounds = [tuple(b) for b in payload["observation_bounds"]]
                print(f"[obs-bounds] loaded cached bounds from {bounds_cache} "
                      f"(created {payload.get('created')}, space={payload.get('space')})")
            else:
                print(f"[obs-bounds] cache obs_dim mismatch "
                      f"({payload.get('obs_dim')} != {obs_dim}); recomputing.")
        except Exception as e:
            print(f"[obs-bounds] failed to read cache ({e}); recomputing.")

    if bounds is None:
        out_dir = os.path.dirname(bounds_cache) or "."
        bounds = collect(
            num_steps=num_steps,
            num_envs=num_envs,
            out_dir=out_dir,
            use_wandb=False,
            teacher_checkpoint=teacher_checkpoint,
            config_path=config_path,
            bounds_cache=bounds_cache,
        )

    assert len(bounds) == obs_dim, (
        f"collected {len(bounds)} bounds but observation_space_dim={obs_dim}"
    )
    task_config.observation_bounds = bounds
    print(f"[obs-bounds] task_config.observation_bounds set ({len(bounds)} dims).")
    return bounds


if __name__ == "__main__":
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
    parser.add_argument("--out_dir", type=str,
                        default="navigation_with_obstacles/obs_stats",
                        help="Output directory for CSV/TensorBoard logs")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Drive the rollout with the ANN teacher (clamped mu) "
                             "and normalize bounds through its running_mean_std. "
                             "If omitted, uses random actions and raw-space bounds.")
    parser.add_argument("--config", type=str, default=None,
                        help="Student YAML. Required with --teacher_checkpoint: the "
                             "teacher network architecture and normalization are read "
                             "from its config.distillation block (single source of truth).")
    parser.add_argument("--bounds_cache", type=str, default=DEFAULT_BOUNDS_CACHE,
                        help="Where to write the JSON bounds cache the runner reads")
    parser.add_argument("--lower_pct", type=float, default=LOWER_PCT,
                        help=f"Lower percentile for encoder bounds (default {LOWER_PCT}). "
                             "Lower => wider clamp, captures more of the left tail.")
    parser.add_argument("--upper_pct", type=float, default=UPPER_PCT,
                        help=f"Upper percentile for encoder bounds (default {UPPER_PCT}). "
                             "Higher => wider clamp, captures more of the right tail.")
    parser.add_argument("--curriculum_level", type=int, default=25,
                        help="Pin the obstacle-density curriculum at this level (teacher's "
                             "final level = 25) so bounds reflect the world the student is "
                             "deployed in. <0 leaves the curriculum at the task default.")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb",
                        help="Disable W&B logging (default: enabled)")
    parser.set_defaults(wandb=True)
    args = parser.parse_args()

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
    )
