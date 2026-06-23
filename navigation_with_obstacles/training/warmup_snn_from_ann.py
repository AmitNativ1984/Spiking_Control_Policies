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
# Distillation loss + logging — STUBS (user implements these)
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
                    f" (n={solo_metric['completions']})")
        logger.info(msg)

    if _tb_writer is not None:
        _tb_writer.add_scalar("warmup/bc_loss", float(loss), step)
        _tb_writer.add_scalar("warmup/beta", float(beta), step)
        if solo_metric is not None:
            _tb_writer.add_scalar("warmup/solo_success_rate", solo_metric["success_rate"], step)
            _tb_writer.add_scalar("warmup/solo_mean_return", solo_metric["mean_return"], step)
            _tb_writer.add_scalar("warmup/solo_completions", solo_metric["completions"], step)


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
        {"name": "--out", "type": str, "default": "navigation_with_obstacles/runs/warmup_snn.pth",
         "help": "Output checkpoint path (rl_games format)"},
        {"name": "--max_steps", "type": int, "default": 200000, "help": "Total BC env steps"},
        {"name": "--lr", "type": float, "default": 1e-3, "help": "Adam LR for the SNN actor"},
        {"name": "--recompute_bounds", "action": "store_true",
         "help": "Force re-collection of PopSAN observation_bounds even if cached"},
        {"name": "--bounds_steps", "type": int, "default": 10000,
         "help": "Steps to collect when auto-computing observation_bounds"},
        {"name": "--log_every", "type": int, "default": 1280,
         "help": "Env steps between console log lines (TensorBoard logs every step)"},
        {"name": "--eval_every", "type": int, "default": 5000,
         "help": "Env steps between SNN-solo (beta=0) eval rollouts"},
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

      * Success is measured HERE, from the task's per-episode arrival count (success_aggregate
        delta) over completed episodes during THIS rollout — NOT the curriculum's stale, windowed
        logged_success_rate, which is dominated by the teacher-driven training steps.
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

        # Baseline aggregates so we count only THIS rollout's outcomes.
        base_succ = float(task.success_aggregate)
        base_crash = float(task.crashes_aggregate)
        base_timeout = float(task.timeouts_aggregate)
        base_exceed = float(task.exceeds_aggregate)

        cur_obs = _extract_obs(task.reset())
        total_reward = torch.zeros(num_envs, device=device)
        for i in range(int(max_steps)):
            actions = _student_solo_action(model, cur_obs)
            obs_out, rewards, *_ = task.step(actions)
            cur_obs = obs_out["observations"]
            total_reward += rewards
            if progress_every and (i + 1) % progress_every == 0:
                logger.info(f"[warmup]   eval {i + 1}/{int(max_steps)} steps...")

        successes = float(task.success_aggregate) - base_succ
        crashes = float(task.crashes_aggregate) - base_crash
        timeouts = float(task.timeouts_aggregate) - base_timeout
        exceeds = float(task.exceeds_aggregate) - base_exceed
        completions = successes + crashes + timeouts + exceeds
        success_rate = (successes / completions) if completions > 0 else 0.0

        metric = {
            "mean_return": float(total_reward.mean()),
            "success_rate": success_rate,
            "completions": int(completions),
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

    torch.manual_seed(args["seed"])

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

    # TensorBoard under runs/warmup_snn_<timestamp>/ (alongside rl_games' own run dirs).
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_dir, "runs",
                           f"warmup_snn_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    init_warmup_logging(log_dir)

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
    # and sets task_config.observation_bounds. The encoder reads bounds at construction.
    _auto_set_observation_bounds(
        teacher_ckpt=teacher_ckpt,
        config_path=config_path,
        num_envs=min(args["num_envs"], 64),
        num_steps=args["bounds_steps"],
        recompute=args["recompute_bounds"],
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

    # =========================================================================
    # Warm-up loop (BC + DAgger beta annealing)
    # =========================================================================
    cur_obs = _extract_obs(task.reset())
    steps = 0
    beta = 1.0
    anneal_active = False
    anneal_start_step = None

    while steps < args["max_steps"]:
        # 1) teacher target (detached, clamped) — same contract as collect_obs_stats.
        teacher_mu = _teacher_action(teacher, cur_obs)  # [B, action_dim], no grad

        # 2) student mu (with grad) through the SAME normalization the wrapper uses.
        norm_obs = model.norm_obs(cur_obs)
        student_mu, _ = student_net.spiking_actor({"obs": norm_obs})

        # 3) BC loss + backprop (loss is a STUB until implemented).
        loss = distillation_loss(student_mu, teacher_mu.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4) DAgger env stepping: per-env, action = teacher w/ prob beta else student.
        # task.step -> action_transformation_function already clamps to [-1,1], so we don't
        # pre-clamp the student action here. teacher_mu is the clamped BC target (it doubles
        # as the teacher's stepping action, matching collect_obs_stats).
        with torch.no_grad():
            student_act = student_mu.detach()
            if beta >= 1.0:
                mixed = teacher_mu
            elif beta <= 0.0:
                mixed = student_act
            else:
                use_teacher = (torch.rand(cur_obs.shape[0], 1, device=device) < beta)
                mixed = torch.where(use_teacher, teacher_mu, student_act)
        obs_out, *_ = task.step(mixed)
        cur_obs = obs_out["observations"]
        steps += cur_obs.shape[0]

        # 5) periodic SNN-solo eval -> trigger / advance beta annealing.
        if steps % args["eval_every"] < cur_obs.shape[0]:
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
            show = (steps % args["log_every"]) < cur_obs.shape[0]
            log_distillation_metrics(steps, loss, beta, console=show)

        # 6) advance beta if annealing is active.
        if anneal_active:
            frac = (steps - anneal_start_step) / max(args["anneal_steps"], 1)
            beta = max(0.0, 1.0 - frac)

    # =========================================================================
    # Phase 4 — save a PPO-loadable checkpoint
    # =========================================================================
    # model.state_dict() contains a2c_network.* (actor+critic), running_mean_std.*, and
    # (if enabled) value_mean_std.* — the exact structure teacher_builder loads via
    # model.load_state_dict(ckpt["model"], strict=True), so PPO's --checkpoint round-trips.
    out_path = args["out"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "epoch": 0,
        "frame": 0,
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
