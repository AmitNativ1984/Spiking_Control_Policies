"""Weights & Biases helpers: git provenance, checkpoint upload, and the metric axis."""

import inspect
import os
import subprocess
import traceback

import wandb
from loguru import logger
from rl_games.algos_torch.a2c_continuous import A2CAgent
# The writer class rl_games actually instantiates: tensorboardX's, NOT
# torch.utils.tensorboard's. Import it from a2c_common so the patch below can never
# target a different SummaryWriter than the one the agent writes through.
from rl_games.common.a2c_common import A2CBase, SummaryWriter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_info() -> dict:
    """Commit hash, number, branch and dirty flag, so a run can be restored to the exact
    code state that produced it. Every field is None if git isn't available."""

    def run(*cmd):
        try:
            return (
                subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:
            return None

    commit = run("git", "rev-parse", "HEAD")
    count = run("git", "rev-list", "--count", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_commit_short": commit[:8] if commit else None,
        "git_commit_number": int(count) if count and count.isdigit() else None,
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def enable_checkpoint_upload() -> None:
    """Wrap A2CAgent.save so every rl_games checkpoint is uploaded to W&B.

    rl_games calls ``agent.save(fn)`` for best/last/periodic checkpoints, where ``fn`` is
    the path WITHOUT the ``.pth`` extension (``save_checkpoint`` appends it). Everything
    goes to a single artifact name and overwrites it, so W&B only ever stores the latest
    weights (keeps storage small).
    """
    original_save = A2CAgent.save

    def save_with_upload(self, fn):
        result = original_save(self, fn)
        if wandb.run is None:  # only upload if a W&B run is active
            return result

        ckpt_path = fn if fn.endswith(".pth") else fn + ".pth"
        if not os.path.exists(ckpt_path):
            logger.warning(f"[wandb] checkpoint not found, skipping upload: {ckpt_path}")
            return result

        try:
            artifact = wandb.Artifact(name=f"{wandb.run.id}-checkpoint", type="model")
            # Always the same filename inside the artifact, so the latest weights
            # overwrite the previous version.
            artifact.add_file(ckpt_path, name="latest.pth")
            wandb.log_artifact(artifact, aliases=["latest"])
            logger.info(f"[wandb] uploaded checkpoint as artifact (latest): {ckpt_path}")
        except Exception:
            logger.error("[wandb] checkpoint upload failed — full traceback below")
            traceback.print_exc()
        return result

    A2CAgent.save = save_with_upload


def log_final_weights(runs_dir: str, full_experiment_name: str, experiment_name: str,
                      info: dict) -> None:
    """Attach the trained weights to the active run as a versioned artifact, so the graphs
    and the model that produced them stay together."""
    ckpt = os.path.join(runs_dir, full_experiment_name, "nn", f"{experiment_name}.pth")
    if not os.path.exists(ckpt):
        logger.warning(f"[wandb] no checkpoint found at {ckpt}; skipping artifact upload")
        return

    artifact = wandb.Artifact(
        name=f"{experiment_name}-weights",
        type="model",
        metadata={k: info[k] for k in
                  ("git_commit_short", "git_commit_number", "git_branch", "git_dirty")},
    )
    artifact.add_file(ckpt)
    wandb.log_artifact(artifact)
    logger.info(f"[wandb] logged model artifact from {ckpt}")


# --- W&B metric axis -------------------------------------------------------------------
#
# rl_games logs every metric to TensorBoard three times, once against each of its three
# step notions: `frame` (environment transitions), `epoch_num` and wall-clock seconds --
# `rewards/step`, `rewards/iter`, `rewards/time` all carry the same value. That only works
# because TensorBoard honours the per-tag step. W&B's `sync_tensorboard` does not: it
# discards the step and plots everything against its own row counter, so the three
# families collapse into identical curves on a meaningless x-axis.
#
# Instead we intercept the TensorBoard writes, drop the `/iter` and `/time` duplicates, and
# emit one W&B row per epoch with `step=frame`. The x-axis then reads environment steps,
# and `epoch`/`elapsed_s` ride along as columns so either can be picked as the x-axis in
# the UI. TensorBoard output is untouched.

_DUPLICATE_SUFFIXES = ("/iter", "/time")
_FRAME_SUFFIXES = ("/frame", "/step")


def _wandb_key(tag: str):
    """TensorBoard tag -> W&B key, or None for a duplicate that should not be logged."""
    if tag.endswith(_DUPLICATE_SUFFIXES):
        return None
    for suffix in _FRAME_SUFFIXES:
        if tag.endswith(suffix):
            return tag[: -len(suffix)]
    return tag


class _EpochMetrics:
    """Buffers a training epoch's scalars and commits them as one W&B row."""

    def __init__(self):
        self.buffer = {}
        self.pending = None  # (frame, epoch, elapsed) the buffered scalars belong to
        self.last_step = None
        self.warned_empty = False

    def record(self, tag, value):
        key = _wandb_key(tag)
        if key is None:
            return
        try:
            self.buffer[key] = float(value)
        except (TypeError, ValueError):
            pass  # non-scalar (histogram, image); TensorBoard still gets it

    def flush(self):
        if not self.buffer:
            # Past the first epoch an empty buffer means add_scalar is not being seen --
            # a silent failure that costs a whole run's metrics, so say so loudly.
            if self.pending is not None and not self.warned_empty:
                logger.warning("[wandb] no scalars captured for epoch "
                               f"{self.pending[1]}; the frame-axis hook is not seeing "
                               "rl_games' SummaryWriter -- metrics will be empty")
                self.warned_empty = True
            return
        if self.pending is None:
            self.buffer.clear()
            return
        frame, epoch, elapsed = self.pending
        step = int(frame)
        # wandb rejects a step that moves backwards; frame is monotonic, but a resumed run
        # restores agent.frame from the checkpoint and could repeat one.
        if self.last_step is not None and step <= self.last_step:
            step = self.last_step + 1
        payload = dict(self.buffer)
        payload.update(frame=frame, epoch=epoch, elapsed_s=elapsed)
        wandb.log(payload, step=step)
        self.last_step = step
        self.buffer.clear()


_EPOCH_METRICS = _EpochMetrics()


def enable_frame_axis_logging() -> None:
    """Mirror rl_games' TensorBoard scalars into W&B keyed on `frame` instead of letting
    `sync_tensorboard` plot them against W&B's row counter."""
    if getattr(SummaryWriter.add_scalar, "_aerial_gym_wandb", False):
        return

    original_add_scalar = SummaryWriter.add_scalar

    def add_scalar_with_wandb(self, tag, scalar_value, *args, **kwargs):
        result = original_add_scalar(self, tag, scalar_value, *args, **kwargs)
        if wandb.run is not None:
            _EPOCH_METRICS.record(tag, scalar_value)
        return result

    add_scalar_with_wandb._aerial_gym_wandb = True
    SummaryWriter.add_scalar = add_scalar_with_wandb

    original_write_stats = A2CBase.write_stats
    signature = inspect.signature(original_write_stats)

    def write_stats_with_wandb(self, *args, **kwargs):
        # write_stats is the first TensorBoard write of an epoch, so on entry the buffer
        # holds the *previous* epoch in full (its losses, its observer metrics and its
        # rewards, which rl_games writes after write_stats returns).
        if wandb.run is not None:
            _EPOCH_METRICS.flush()
            try:
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                _EPOCH_METRICS.pending = (bound.arguments["frame"],
                                          bound.arguments["epoch_num"],
                                          bound.arguments["total_time"])
            except (TypeError, KeyError):
                logger.warning("[wandb] unexpected write_stats signature; "
                               "metrics will be logged without a frame axis")
                _EPOCH_METRICS.pending = None
        return original_write_stats(self, *args, **kwargs)

    A2CBase.write_stats = write_stats_with_wandb


def flush_final_metrics() -> None:
    """Commit the last epoch, which has no following write_stats to trigger its flush."""
    if wandb.run is not None:
        _EPOCH_METRICS.flush()
