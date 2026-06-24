"""
Phase 3 — BC warm-up: behavior-clone an ANN teacher into the PopSAN SNN student.

Cold-starting PPO on the spiking student is unstable, so we first behavior-clone the
teacher's deterministic action (`mu`) into the SNN, then save a checkpoint PPO can resume
via `--checkpoint`. The teacher is a frozen rl_games `ModelA2CContinuousLogStd.Network`
(with its `running_mean_std`); the student is built as the SAME full rl_games wrapper around
`POPSANNetwork`, so the saved checkpoint round-trips into the existing runner with no key
mismatch.

This script is PURELY ADDITIVE — it imports `training.runner` for its registration
side-effects and reuses the teacher loader / bounds collector / obs helpers. It does NOT
modify the env, task, or any network class.

Normalization ownership: we copy the teacher's frozen `running_mean_std` into the student
wrapper and keep it frozen for the whole warm-up, so teacher and student see identical
normalized obs — the exact space the PopSAN encoder bounds were measured in (Phase 2).

Usage:
    cd /workspaces/aerial_gym_docker
    python -m navigation_with_obstacles.training.warmup_snn_from_ann \
        --file=navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml \
        --num_envs=256 --headless=True --max_steps=200000 --out=/tmp/warmup_snn.pth

"""
import isaacgym  # must be imported before torch

import os
import sys
import yaml
from datetime import datetime

sys.path.insert(0, "/workspaces/aerial_gym_docker")

import torch
import numpy as np

# Importing the runner registers the task, env, vecenv, and all network builders
# (PopSAN / mlp_actor_critic / mlp_gru_actor_critic) with aerial_gym + rl_games. This is
# the single source of truth for that wiring — we never duplicate it here.
import navigation_with_obstacles.training.runner as nav_runner  # noqa: F401  (side effects)
from navigation_with_obstacles.training.runner import _auto_set_observation_bounds

from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments

from navigation_with_obstacles.config.task_config import task_config
from navigation_with_obstacles.tools.collect_obs_stats import (
    _load_distillation_cfg,
    _build_teacher_from_checkpoint,
    _teacher_action,
)

from rl_games.algos_torch.model_builder import ModelBuilder
from loguru import logger


# =============================================================================
# Distillation loss + logging
# =============================================================================

def distillation_loss(student_mu: torch.Tensor, teacher_mu: torch.Tensor) -> torch.Tensor:
    """BC loss: MSE between the SNN student's action mean and the teacher's.

        student_mu : [B, action_dim] — SNN actor mu (requires_grad).
        teacher_mu : [B, action_dim] — detached teacher target (clamped to [-1, 1]).
        returns    : scalar tensor.

    Both policies are diagonal Gaussians. Only mu enters this loss, so the student's sigma
    (decoder.log_std) gets no gradient and stays fixed; the teacher is frozen. The policy KL then
    reduces to (mu_S - mu_T)^2 / (2*sigma^2) plus sigma-only constants with no gradient — i.e. a
    sigma-weighted MSE of the means, so plain MSE is gradient-equivalent up to a constant LR
    rescale. If a later phase also trains sigma, switch to the full Gaussian KL.
    """
    return torch.nn.functional.mse_loss(student_mu, teacher_mu)


# Module-level TensorBoard writer for the warm-up; set by init_warmup_logging().
_tb_writer = None


def init_warmup_logging(log_dir):
    """Open a TensorBoard SummaryWriter for the warm-up under `log_dir`.

    Mirrors the project's logging convention (torch.utils.tensorboard, as in
    tools/collect_obs_stats.py). Safe to call once from main(); if it fails we keep the
    console-only fallback in log_distillation_metrics."""
    global _tb_writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        _tb_writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"[warmup] tensorboard logging to {log_dir}")
    except Exception as e:
        logger.warning(f"[warmup] tensorboard init failed ({e}); console logging only.")
        _tb_writer = None


def log_distillation_metrics(step, loss, beta, solo_metric=None, console=True):
    """Log warm-up metrics to console + TensorBoard: BC loss, DAgger beta, and (on eval
    steps) the SNN-solo success/return/completions.

    `step` is the cumulative env-step count (x-axis). `solo_metric` is the dict returned by
    snn_solo_eval (keys: mean_return, success_rate, completions) on eval steps, else None.
    TensorBoard scalars are always written; `console` gates only the per-step printed line so
    the terminal stays readable while the TB curves stay dense.
    """
    if console:
        msg = f"[warmup] step={step} loss={float(loss):.5f} beta={beta:.3f}"
        if solo_metric is not None:
            msg += (f" solo_success={solo_metric['success_rate']:.3f}"
                    f" solo_return={solo_metric['mean_return']:.3f}"
                    f" (n={solo_metric['completions']}"
                    f" arr={solo_metric['successes']} crash={solo_metric['crashes']}"
                    f" exc={solo_metric['exceeds']} to={solo_metric['timeouts']})")
        logger.info(msg)

    if _tb_writer is not None:
        _tb_writer.add_scalar("warmup/bc_loss", float(loss), step)
        _tb_writer.add_scalar("warmup/beta", float(beta), step)
        if solo_metric is not None:
            _tb_writer.add_scalar("warmup/solo_success_rate", solo_metric["success_rate"], step)
            _tb_writer.add_scalar("warmup/solo_mean_return", solo_metric["mean_return"], step)
            _tb_writer.add_scalar("warmup/solo_completions", solo_metric["completions"], step)
            _tb_writer.add_scalar("warmup/solo_crash_rate",
                                  solo_metric["crashes"] / max(solo_metric["completions"], 1), step)
            _tb_writer.add_scalar("warmup/solo_exceed_rate",
                                  solo_metric["exceeds"] / max(solo_metric["completions"], 1), step)
            _tb_writer.add_scalar("warmup/solo_timeout_rate",
                                  solo_metric["timeouts"] / max(solo_metric["completions"], 1), step)


# =============================================================================
# Helpers
# =============================================================================

def get_args():
    custom_parameters = [
        {"name": "--file", "type": str,
         "default": "navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml",
         "help": "Student YAML (PopSAN + distillation)"},
        {"name": "--num_envs", "type": int, "default": None,
         "help": "Number of parallel envs. If unset, uses the YAML's config.env_config.num_envs."},
        {"name": "--headless", "type": lambda x: x.lower() in ("1", "true", "yes"),
         "default": "True", "help": "Headless mode"},
        {"name": "--use_warp", "type": lambda x: x.lower() in ("1", "true", "yes"),
         "default": "True", "help": "Use warp"},
        {"name": "--out", "type": str, "default": None,
         "help": "Output checkpoint path (rl_games format). Default: <run_dir>/nn/warmup_snn.pth "
                 "where run_dir is the per-warm-up runs/warmup_snn_<timestamp>/ folder."},
        {"name": "--max_steps", "type": int, "default": 200000, "help": "Total BC env steps"},
        {"name": "--lr", "type": float, "default": 1e-3, "help": "Adam LR for the SNN actor"},
        {"name": "--buffer_size", "type": int, "default": 200000,
         "help": "Replay-buffer capacity in (norm_obs, teacher_mu) pairs. The buffer decouples "
                 "gradient steps from the slow sim: each env step appends num_envs fresh pairs "
                 "and we draw i.i.d. minibatches from the whole buffer. CPU-resident (~212 B/pair, "
                 "so 200k ~= 42 MB); fixed-size ring (overwrites oldest), never grows unbounded."},
        {"name": "--grad_steps_per_env_step", "type": int, "default": 8,
         "help": "Gradient updates per env step (K). Multiplies gradient throughput per expensive "
                 "sim step by ~K at near-zero extra sim cost; steps run sequentially so peak VRAM "
                 "is unchanged. 1 reproduces the old one-step-per-env-step behavior."},
        {"name": "--batch_size", "type": int, "default": None,
         "help": "Minibatch size for each gradient step (drawn from the buffer). Default None => "
                 "num_envs, so peak VRAM matches the old per-rollout batch. Raise to trade VRAM "
                 "for a larger/more-diverse batch."},
        {"name": "--learning_starts", "type": int, "default": 4096,
         "help": "Minimum buffer fill (pairs) before any gradient step, so early minibatches are "
                 "not drawn from a near-empty buffer."},
        {"name": "--reuse_bounds", "action": "store_true",
         "help": "Reuse a cached observation_bounds.json if it matches this teacher. By default "
                 "the warm-up RECOMPUTES the bounds every run (fresh teacher-driven collection)."},
        {"name": "--bounds_steps", "type": int, "default": 10000,
         "help": "Minimum env-steps to collect when auto-computing observation_bounds"},
        {"name": "--bounds_episodes", "type": int, "default": 0,
         "help": "Also collect until >= this many COMPLETED episodes for the bounds (true "
                 "start-to-end statistics). 0 = step-count only."},
        {"name": "--bounds_envs", "type": int, "default": 64,
         "help": "Max parallel envs for the bounds-collection subprocess (kept small/independent "
                 "of the training num_envs to bound its memory)."},
        {"name": "--log_every", "type": int, "default": 1280,
         "help": "Env steps between console log lines (TensorBoard logs every step)"},
        {"name": "--eval", "action": "store_true",
         "help": "Enable periodic SNN-solo (beta=0) evals that gate DAgger beta annealing. "
                 "OFF by default: each eval does a full obstacle reset + rollout, a memory "
                 "spike that can freeze memory-limited/uncapped-container machines. With evals "
                 "off, beta stays 1.0 (pure teacher-driven BC) — still a valid warm-up. Enable "
                 "on the cluster to get the solo-success signal + beta anneal."},
        {"name": "--eval_every", "type": int, "default": 5000,
         "help": "Env steps between SNN-solo (beta=0) eval rollouts (only if --eval)"},
        {"name": "--eval_max_steps", "type": int, "default": 800,
         "help": "Fixed env-step budget per SNN-solo eval rollout (one episode_len_steps; "
                 "many episodes end sooner so completions still accumulate). Keep small on "
                 "memory-limited machines — each eval also does one full obstacle reset."},
        {"name": "--solo_success_threshold", "type": float, "default": 0.2,
         "help": "Start annealing beta once the SNN-solo eval success_rate >= this"},
        {"name": "--anneal_steps", "type": int, "default": 100000,
         "help": "Env steps to linearly anneal beta 1->0 once the trigger fires"},
        {"name": "--curriculum_level", "type": int, "default": 25,
         "help": "Fix the obstacle-density curriculum at this level for the whole warm-up "
                 "(default 25 = the ep_450 teacher's final level). Set <0 to leave the "
                 "curriculum free to advance from min_level as the teacher succeeds."},
        {"name": "--seed", "type": int, "default": 42, "help": "Random seed"},
        {"name": "--track", "action": "store_true",
         "help": "Track with Weights & Biases (syncs the warm-up TensorBoard scalars)"},
        {"name": "--wandb-project-name", "type": str, "default": "aerial_gym",
         "help": "W&B project name"},
        {"name": "--wandb-entity", "type": str, "default": None,
         "help": "W&B entity (team)"},
        {"name": "--experiment_name", "type": str, "default": "warmup_snn_from_ann",
         "help": "Experiment/run name (used for the W&B run name)"},
    ]
    return parse_arguments(description="BC warm-up ANN->SNN", custom_parameters=custom_parameters)


def _extract_obs(reset_or_step_out):
    """task.reset()/step() return the obs dict under key 'observations'."""
    if isinstance(reset_or_step_out, dict):
        return reset_or_step_out["observations"]
    return reset_or_step_out[0]["observations"]


def build_student_wrapper(student_network_cfg, model_name, obs_dim, action_dim,
                          device, normalize_input, normalize_value):
    """Build the student as the full rl_games wrapper around POPSANNetwork.

    Mirrors networks/teacher_student/teacher_builder.build_teacher's construction path, but
    keeps the model TRAINABLE (no freeze) and uses the student's `network` block, which has
    `name: PopSAN` -> resolves to the already-registered POPSANNetworkBuilder.
    """
    params = {"model": {"name": model_name}, "network": student_network_cfg}
    builder = ModelBuilder()
    model_factory = builder.load(params)
    model = model_factory.build({
        "actions_num": action_dim,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_input": normalize_input,
        "normalize_value": normalize_value,
    })
    model.to(device)
    return model


def _freeze_running_mean_std(module):
    """Put a RunningMeanStd module in eval mode and stop it updating / requiring grad.

    eval() makes RunningMeanStd a pure normalizer (it only updates stats in train mode);
    requires_grad_(False) guarantees BC gradients never touch the obs/value stats.
    """
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    for b in module.buffers():
        b.requires_grad_(False)


class BCReplayBuffer:
    """Fixed-size CPU ring buffer of (normalized_obs, teacher_mu) BC pairs.

    Decouples gradient steps from the slow simulator: each env step appends num_envs fresh
    pairs; gradient steps draw i.i.d. minibatches from the whole buffer (breaking the strong
    within-rollout correlation of a single sim batch). CPU-resident and pre-allocated, so it
    adds bounded host RAM (~ (obs_dim+act_dim)*4 bytes/pair) and zero GPU VRAM; minibatches are
    moved to `device` on draw. We store the NORMALIZED obs (post running_mean_std) so the
    student forward is a plain spiking_actor call with no re-normalization, and because the
    obs-norm stats are frozen during warm-up the stored values never go stale.
    """

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = int(capacity)
        assert self.capacity > 0, "buffer_size must be > 0"
        self.device = device
        self.obs = torch.zeros(self.capacity, obs_dim, dtype=torch.float32)   # CPU
        self.mu = torch.zeros(self.capacity, act_dim, dtype=torch.float32)    # CPU
        self.pos = 0      # next write index (wraps)
        self.full = False  # True once we've wrapped at least once

    def __len__(self):
        return self.capacity if self.full else self.pos

    @torch.no_grad()
    def add(self, norm_obs, teacher_mu):
        """Append a batch [B, obs_dim] / [B, act_dim], wrapping the ring as needed."""
        b = norm_obs.shape[0]
        # A single add splits into at most two contiguous spans (pre-wrap tail + post-wrap head),
        # which only covers the ring once. A batch larger than the whole buffer would need >1 wrap
        # and silently drop/overwrite its own samples — disallow it rather than corrupt the ring.
        assert b <= self.capacity, (
            f"add batch size {b} exceeds buffer capacity {self.capacity}; "
            f"increase --buffer_size (>= num_envs)."
        )
        obs_c = norm_obs.detach().to("cpu", torch.float32)
        mu_c = teacher_mu.detach().to("cpu", torch.float32)
        # Split into up to two contiguous spans so a batch that crosses the end wraps correctly.
        first = min(b, self.capacity - self.pos)
        self.obs[self.pos:self.pos + first] = obs_c[:first]
        self.mu[self.pos:self.pos + first] = mu_c[:first]
        rem = b - first
        if rem > 0:
            self.obs[:rem] = obs_c[first:]
            self.mu[:rem] = mu_c[first:]
            self.full = True
        self.pos = (self.pos + b) % self.capacity
        if self.pos == 0:
            self.full = True

    @torch.no_grad()
    def sample(self, batch_size):
        """Draw a random minibatch (with replacement) moved to the compute device."""
        n = len(self)
        idx = torch.randint(0, n, (batch_size,))
        return self.obs[idx].to(self.device), self.mu[idx].to(self.device)


@torch.no_grad()
def _student_solo_action(model, raw_obs):
    """Deterministic SNN action: the spiking actor's mu.

    Assumes the caller has set task_config.vae_gate = 1.0 (done in main() after make_task), so
    the actor's VAE-latent spikes are active. Normalize raw obs through the (frozen) student
    running_mean_std exactly as the wrapper does at train/play time, then run the spiking actor
    and take its mu. No pre-clamp: task.step's action_transformation_function clamps to [-1, 1]."""
    norm_obs = model.norm_obs(raw_obs)
    mu, _ = model.a2c_network.spiking_actor({"obs": norm_obs})
    return mu


# Task fields that the curriculum/VAE state machine mutates as a side effect of stepping.
# We snapshot these before a solo-eval and restore them after, so the eval never advances
# the curriculum, never flips the VAE gate, and never discards the trigger it feeds.
_CURRICULUM_STATE_FIELDS = (
    "success_aggregate", "crashes_aggregate", "timeouts_aggregate", "exceeds_aggregate",
    "logged_success_rate", "logged_crash_rate", "logged_exceed_rate", "logged_timeout_rate",
    "curriculum_level", "curriculum_progress_fraction", "vae_phase",
)


def _snapshot_task_state(task):
    snap = {f: getattr(task, f) for f in _CURRICULUM_STATE_FIELDS if hasattr(task, f)}
    snap["_vae_gate"] = task_config.vae_gate
    return snap


def _restore_task_state(task, snap):
    for f, v in snap.items():
        if f == "_vae_gate":
            task_config.vae_gate = v
        else:
            setattr(task, f, v)


@torch.no_grad()
def snn_solo_eval(model, task, max_steps, device, progress_every=200):
    """Run a pure-student (beta=0) rollout and return a self-contained metric + the final obs.

    This is the REAL convergence signal (not the BC loss): once the SNN drives the env on its
    own well enough, we hand exploration over by annealing beta. To make the signal trustworthy:

      * Outcomes are counted HERE directly from each step's return tuple — NOT from the task's
        *_aggregate fields. The task adds outcomes to those aggregates every step
        (navigation_task.py:595-598) and ZEROES them whenever enough episodes accumulate
        (navigation_task.py:681-685), so ANY delta/increment scheme on them loses every wrap
        step's counts. Instead we read the per-env terminal signals the task exports each step:
          - arrivals = infos["arrivals"]          (success; arrive_mask, exported for this purpose)
          - timeouts = infos["time_outs"]          (episode hit the step limit)
          - exceeds  = truncations & ~arrivals & ~timeouts   (out-of-bounds = the rest of truncations)
          - crashes  = terminations & ~arrivals & ~exceeds   (collisions only)
        NOTE: `terminations` is `exceed | arrive | collision` (navigation_task.py:740), so it is
        TRUE for arrivals and exceeds too — raw `terminations` is NOT the crash mask. Crashes are
        the terminations left after removing arrive and exceed (matches navigation_task.py:430).
        `truncations` is `timeout | arrive | exceed` (line 421); exceeds are what remain after
        removing arrive and timeout. These four are mutually exclusive and partition every episode
        end, so summing them over the rollout gives exact totals with no task-owned mutable state.
      * The rollout runs `max_steps` env-steps (a fixed budget; many 800-step episodes end sooner
        on arrival/crash, so completions accumulate within it).
      * All curriculum/VAE side effects are snapshotted before and restored after, so the eval
        does not advance the curriculum, flip the VAE gate, or bias the trigger it feeds.

    Returns (metric_dict, final_obs). final_obs is handed back to the training loop so it can
    resume WITHOUT a second full task.reset() (each reset force-re-randomizes all obstacles — the
    expensive, RAM-heavy op we want to do once per eval, not twice).
    """
    snap = _snapshot_task_state(task)
    try:
        num_envs = task.sim_env.num_envs
        totals = {"success": 0.0, "crash": 0.0, "timeout": 0.0, "exceed": 0.0}

        cur_obs = _extract_obs(task.reset())
        total_reward = torch.zeros(num_envs, device=device)
        for i in range(int(max_steps)):
            actions = _student_solo_action(model, cur_obs)
            obs_out, rewards, terminations, truncations, infos = task.step(actions)
            cur_obs = obs_out["observations"]
            total_reward += rewards

            # Count this step's terminal events from the return tuple (see docstring). Bool masks
            # so the four categories stay mutually exclusive and never double-count an env.
            # terminations = exceed|arrive|collision and truncations = timeout|arrive|exceed, so
            # crashes/exceeds must be peeled out of them (NOT raw terminations/truncations).
            arrivals = infos["arrivals"].bool()
            timeouts = infos["time_outs"].bool()
            exceeds = truncations.bool() & ~arrivals & ~timeouts
            crashes = terminations.bool() & ~arrivals & ~exceeds
            totals["success"] += float(arrivals.sum())
            totals["crash"] += float(crashes.sum())
            totals["timeout"] += float(timeouts.sum())
            totals["exceed"] += float(exceeds.sum())

            if progress_every and (i + 1) % progress_every == 0:
                logger.info(f"[warmup]   eval {i + 1}/{int(max_steps)} steps...")

        completions = totals["success"] + totals["crash"] + totals["timeout"] + totals["exceed"]
        success_rate = (totals["success"] / completions) if completions > 0 else 0.0

        metric = {
            "mean_return": float(total_reward.mean()),
            "success_rate": success_rate,
            "completions": int(completions),
            "successes": int(totals["success"]),
            "crashes": int(totals["crash"]),
            "timeouts": int(totals["timeout"]),
            "exceeds": int(totals["exceed"]),
        }
        return metric, cur_obs
    finally:
        _restore_task_state(task, snap)


# =============================================================================
# Main
# =============================================================================

def main():
    args = vars(get_args())
    config_path = args["file"]
    device = task_config.device

    # Seed every RNG the warm-up touches: torch (SNN init + DAgger torch.rand),
    # numpy, and the task's own seed (obstacle layouts / spawn) via task_config.
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    np.random.seed(args["seed"])
    task_config.seed = args["seed"]

    # W&B (optional). Init BEFORE the TensorBoard writer so sync_tensorboard mirrors every
    # warmup/* scalar to W&B automatically — same pattern as training/runner.py.
    wandb_run = None
    if args.get("track"):
        import wandb
        wandb_run = wandb.init(
            project=args["wandb_project_name"],
            entity=args["wandb_entity"],
            name=f"{args['experiment_name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            sync_tensorboard=True,
            config=args,
        )

    # Single run dir per warm-up, so ALL artifacts live together (mirrors rl_games' layout):
    #   runs/warmup_snn_<ts>/summaries/   TensorBoard
    #   runs/warmup_snn_<ts>/nn/          checkpoint (--out default)
    #   runs/warmup_snn_<ts>/             encoder_trace_*.png
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(project_dir, "runs",
                           f"warmup_snn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(os.path.join(run_dir, "nn"), exist_ok=True)
    init_warmup_logging(os.path.join(run_dir, "summaries"))

    # Default the checkpoint into this run dir unless the user gave an explicit --out.
    if not args.get("out"):
        args["out"] = os.path.join(run_dir, "nn", "warmup_snn.pth")

    # --- Read student YAML + distillation block --------------------------------
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)
    student_network_cfg = full_cfg["params"]["network"]
    model_name = full_cfg["params"]["model"]["name"]

    # num_envs: CLI overrides if given, else the YAML's env_config.num_envs (single source of
    # truth, matching runner.py). Resolve it back into args so all downstream uses agree.
    if args.get("num_envs") is None or args["num_envs"] <= 0:
        args["num_envs"] = full_cfg["params"]["config"]["env_config"]["num_envs"]
    logger.info(f"[warmup] num_envs = {args['num_envs']}")

    distill = _load_distillation_cfg(config_path)  # single source of truth for the teacher
    teacher_ckpt = distill["checkpoint"]
    normalize_input = distill["normalize_input"]
    normalize_value = distill["normalize_value"]
    assert os.path.exists(teacher_ckpt), f"teacher checkpoint not found: {teacher_ckpt}"

    # --- Phase 2: set PopSAN encoder observation_bounds BEFORE building the net ---
    # Runs the collector in a subprocess (Isaac Gym: one sim per process), caches to JSON,
    # sets task_config.observation_bounds, AND saves the teacher-bounds encoder PNGs into the
    # warm-up run dir (before those bounds are applied to the student encoder). Collection
    # stops on episodes (--bounds_episodes) when given. The cache is auto-recomputed if it was
    # built for a different teacher (see runner._auto_set_observation_bounds).
    _auto_set_observation_bounds(
        teacher_ckpt=teacher_ckpt,
        config_path=config_path,
        num_envs=min(args["num_envs"], args["bounds_envs"]),
        num_steps=args["bounds_steps"],
        recompute=not args["reuse_bounds"],  # recompute by default; --reuse_bounds opts out
        min_episodes=args["bounds_episodes"],
        out_dir=run_dir,
        curriculum_level=args["curriculum_level"],  # bounds at the SAME level the warm-up pins
    )

    # --- Build the env ---------------------------------------------------------
    task_config.num_envs = args["num_envs"]
    task = task_registry.make_task(
        "navigation_with_obstacles_task",
        num_envs=args["num_envs"],
        headless=args["headless"],
        use_warp=args["use_warp"],
    )
    obs_dim = task_config.observation_space_dim
    action_dim = task_config.action_space_dim

    # Warm-up runs with the VAE fully ON, matching the PPO hand-off (the teacher saw the
    # full obs incl. VAE latents, so the student must too). This MUST come AFTER make_task:
    # NavigationTask.__init__ unconditionally sets vae_phase="A" / vae_gate=0.0 when use_vae
    # is True (navigation_task.py:158-160), which would otherwise zero the student's VAE-latent
    # spike block every forward (pop_spiking_actor.py:144-146). Forcing phase "C" also lands the
    # task's curriculum state machine in its normal-curriculum branch, so it never flips the
    # gate back during warm-up.
    if task_config.vae_config.use_vae:
        task.vae_phase = "C"
    task_config.vae_gate = 1.0

    # Pin the obstacle-density curriculum to a fixed level for the whole warm-up. We clone the
    # teacher at the difficulty it actually operates at (its ep_450 final level = 25), not a
    # moving target: phase "C" otherwise lets check_and_update_curriculum_level drift the level
    # up/down with success. Pinning min==max==level makes the task's own clamp
    # (navigation_task.py:650-654) hold it fixed regardless of measured success. Set
    # --curriculum_level < 0 to leave it free.
    fixed_level = args["curriculum_level"]
    if fixed_level is not None and fixed_level >= 0:
        task_config.curriculum.min_level = fixed_level
        task_config.curriculum.max_level = fixed_level
        task.curriculum_level = fixed_level
        task.obs_dict["num_obstacles_in_env"] = max(fixed_level, task.sim_env.keep_in_env)
        logger.info(f"[warmup] curriculum pinned at level {fixed_level} (teacher's final level)")

    # --- Build the frozen teacher ----------------------------------------------
    teacher = _build_teacher_from_checkpoint(
        teacher_ckpt, distill, obs_dim, action_dim, device)
    logger.info(f"[warmup] teacher loaded (frozen) from {teacher_ckpt}")

    # --- Build the student as the full rl_games wrapper ------------------------
    model = build_student_wrapper(
        student_network_cfg, model_name, obs_dim, action_dim,
        device, normalize_input, normalize_value)
    student_net = model.a2c_network  # POPSANNetwork

    # --- Copy teacher normalization stats into the student, then FREEZE them ----
    # Same RunningMeanStd class on both wrappers -> 1:1 load. Keeping them frozen during BC
    # ensures the encoder sees the exact normalized space its bounds were measured in.
    if normalize_input:
        model.running_mean_std.load_state_dict(teacher.running_mean_std.state_dict())
        _freeze_running_mean_std(model.running_mean_std)
        logger.info("[warmup] copied + froze teacher running_mean_std into student")
    if normalize_value and hasattr(model, "value_mean_std") and hasattr(teacher, "value_mean_std"):
        # Unused during BC (no value targets), but carried for the round-trip checkpoint.
        model.value_mean_std.load_state_dict(teacher.value_mean_std.state_dict())
        _freeze_running_mean_std(model.value_mean_std)

    # --- Init the student critic from the ANN critic (NOT trained in warm-up) ---
    # Both are ANNMLPCritic -> weights copy 1:1. PPO trains it live later (Phase 4.5).
    student_net.critic.load_state_dict(teacher.a2c_network.critic.state_dict())
    for p in student_net.critic.parameters():
        p.requires_grad_(False)
    logger.info("[warmup] initialized student critic from ANN critic (frozen for warm-up)")

    # --- Optimizer over the SNN actor only -------------------------------------
    actor_params = list(student_net.spiking_actor.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=args["lr"])

    # --- Replay buffer (decouples gradient steps from the slow sim) -------------
    batch_size = args["batch_size"] or args["num_envs"]
    replay = BCReplayBuffer(args["buffer_size"], obs_dim, action_dim, device)
    grad_steps_per_env_step = max(1, args["grad_steps_per_env_step"])
    logger.info(f"[warmup] replay buffer cap={args['buffer_size']} pairs, "
                f"{grad_steps_per_env_step} grad steps/env step, batch={batch_size}, "
                f"learning_starts={args['learning_starts']}")

    # =========================================================================
    # Warm-up loop (BC + DAgger beta annealing)
    # =========================================================================
    cur_obs = _extract_obs(task.reset())
    steps = 0
    beta = 1.0
    loss = torch.zeros((), device=device)  # last minibatch loss (for logging before first update)
    anneal_active = False
    anneal_start_step = None
    # Monotonic cadence targets: `steps` advances by num_envs per iteration, so a modulo
    # window (steps % every < num_envs) is fragile when num_envs and the period aren't
    # commensurate. Track the next firing threshold instead — exactly one fire per period.
    next_eval_at = args["eval_every"]
    next_log_at = args["log_every"]

    while steps < args["max_steps"]:
        # 1) teacher target (detached, clamped) — same contract as collect_obs_stats.
        teacher_mu = _teacher_action(teacher, cur_obs)  # [B, action_dim], no grad

        # 2) normalized obs (frozen running_mean_std). Stored in the buffer AND used for the
        # DAgger action below; we normalize once per env step.
        norm_obs = model.norm_obs(cur_obs)

        # 3) Append this step's (norm_obs, teacher_mu) pairs to the replay buffer.
        replay.add(norm_obs, teacher_mu)

        # 4) K gradient steps on i.i.d. minibatches drawn from the buffer (once it has filled
        # past learning_starts). Sequential -> peak VRAM == one minibatch, not K. This is where
        # the speedup comes from: many BC updates per expensive sim step, on decorrelated data.
        if len(replay) >= args["learning_starts"]:
            for _ in range(grad_steps_per_env_step):
                b_obs, b_mu = replay.sample(batch_size)
                b_student_mu, _ = student_net.spiking_actor({"obs": b_obs})
                loss = distillation_loss(b_student_mu, b_mu)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 5) DAgger env stepping: per-env, action = teacher w/ prob beta else student.
        # task.step -> action_transformation_function already clamps to [-1,1], so we don't
        # pre-clamp the student action here. teacher_mu is the clamped BC target (it doubles
        # as the teacher's stepping action, matching collect_obs_stats). The student action for
        # stepping is computed under no_grad on the CURRENT obs (the gradient steps above already
        # consumed the buffer minibatches), so it never builds an autograd graph here.
        with torch.no_grad():
            if beta >= 1.0:
                mixed = teacher_mu
            else:
                student_act, _ = student_net.spiking_actor({"obs": norm_obs})
                if beta <= 0.0:
                    mixed = student_act
                else:
                    use_teacher = (torch.rand(cur_obs.shape[0], 1, device=device) < beta)
                    mixed = torch.where(use_teacher, teacher_mu, student_act)
        obs_out, *_ = task.step(mixed)
        cur_obs = obs_out["observations"]
        steps += cur_obs.shape[0]

        # 6) periodic SNN-solo eval -> trigger / advance beta annealing (only if --eval).
        # Each eval does a full obstacle reset + rollout: a memory spike. Off by default so
        # local/uncapped-container runs don't freeze; with it off beta stays 1.0 (teacher-driven).
        # Console line cadence advances independently of the eval cadence.
        show = steps >= next_log_at
        if show:
            next_log_at += args["log_every"]

        do_eval = args["eval"] and steps >= next_eval_at
        if do_eval:
            next_eval_at += args["eval_every"]
            logger.info(f"[warmup] step={steps} running SNN-solo eval "
                        f"({args['eval_max_steps']} sim steps)...")
            # The eval ends having reset+rolled the env; reuse its final obs so the training loop
            # resumes WITHOUT a second full (obstacle-re-randomizing) reset.
            solo, cur_obs = snn_solo_eval(model, task, args["eval_max_steps"], device)
            log_distillation_metrics(steps, loss, beta, solo_metric=solo)
            if not anneal_active and solo["success_rate"] >= args["solo_success_threshold"]:
                anneal_active = True
                anneal_start_step = steps
                logger.info(f"[warmup] SNN-solo success {solo['success_rate']:.3f} >= "
                            f"{args['solo_success_threshold']}; starting beta anneal")
        else:
            # TB scalars every step; console line only on the log_every cadence.
            log_distillation_metrics(steps, loss, beta, console=show)

        # 7) advance beta if annealing is active.
        if anneal_active:
            frac = (steps - anneal_start_step) / max(args["anneal_steps"], 1)
            beta = max(0.0, 1.0 - frac)

    # =========================================================================
    # Phase 4 — save a PPO-loadable checkpoint
    # =========================================================================
    # model.state_dict() contains a2c_network.* (actor+critic), running_mean_std.*, and
    # (if enabled) value_mean_std.* — the exact structure rl_games' set_weights() loads.
    #
    # CRITICAL: a `--train --checkpoint` resume does NOT go through set_weights() alone — it
    # calls agent.restore() -> set_full_state_weights() (rl_games a2c_common.py), which reads
    # weights['optimizer'] *unguarded* (and last_mean_rewards/env_state via .get). A dict with
    # only model/epoch/frame therefore raises KeyError: 'optimizer' at resume. So we also save
    # a full-model Adam state_dict: rl_games builds its optimizer as Adam(self.model.parameters()),
    # so a fresh Adam over the SAME model.parameters() yields matching param-groups that
    # load_state_dict accepts (PPO re-fills the empty state on its first step anyway). The
    # warm-up's own optimizer covers only the spiking actor (a subset), so it can't be reused here.
    out_path = args["out"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    full_model_optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    torch.save({
        "model": model.state_dict(),
        "optimizer": full_model_optimizer.state_dict(),
        "epoch": 0,
        "frame": 0,
        "last_mean_rewards": -1000000000,
    }, out_path)
    logger.info(f"[warmup] saved warm-up checkpoint to {out_path}")

    if _tb_writer is not None:
        _tb_writer.flush()
        _tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    task.close()
    logger.info("[warmup] done.")


if __name__ == "__main__":
    main()
