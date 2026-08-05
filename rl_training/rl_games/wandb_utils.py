"""Weights & Biases helpers: git provenance and checkpoint upload."""

import os
import subprocess
import traceback

import wandb
from loguru import logger
from rl_games.algos_torch.a2c_continuous import A2CAgent

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
