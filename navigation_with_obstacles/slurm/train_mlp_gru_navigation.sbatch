#!/bin/bash
#SBATCH --job-name=nav_ann_gru
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/nav_ann_gru_%j.out
#SBATCH --error=slurm_logs/nav_ann_gru_%j.err

# =============================================================================
# Navigation VAE Training on SLURM with Pyxis
# =============================================================================
#
# Submit from the repo root, e.g.:
#   cd /workspaces/aerial_gym_docker
#   sbatch navigation_with_obstacles/slurm/train_mlp_gru_navigation.sbatch
#
# Log files land in <repo_root>/slurm_logs/nav_ann_gru_<jobid>.{out,err}
# (the path on lines 6-7 is relative to SLURM_SUBMIT_DIR).
#
# Override defaults via env vars:
#   CONTAINER_IMAGE=... EXPERIMENT=... WANDB_PROJECT=... CONFIG_FILE=... \
#     sbatch navigation_with_obstacles/slurm/train_mlp_gru_navigation.sbatch
#
# Prerequisites:
#   1. Docker image built and available (see slurm/README_cluster.md)
#   2. W&B API key in slurm/.env (see README)
#   3. <repo_root>/slurm_logs/ must exist (SLURM does not create it)
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
EXPERIMENT="${EXPERIMENT:-nav_ann_gru_cluster}"
WANDB_PROJECT="${WANDB_PROJECT:-nav_vae_gru_ann}"      
CONFIG_FILE="${CONFIG_FILE:-navigation_with_obstacles/training/ppo_navigation_ann_gru_cluster.yaml}"
CHECKPOINT="${CHECKPOINT:-}"  # Path to checkpoint file to resume training (empty = start fresh)

# --- Paths ---
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked
REPO_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_DIR}/navigation_with_obstacles/slurm"

# --- Load W&B API key ---
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
    echo "Loaded environment from ${ENV_FILE}"
else
    echo "WARNING: ${ENV_FILE} not found. W&B tracking will fail unless WANDB_API_KEY is set."
fi

# --- Print job info ---
echo "============================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            ${SLURMD_NODENAME}"
echo "GPU:             ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Container:       ${CONTAINER_IMAGE}"
echo "Config:          ${CONFIG_FILE}"
echo "Experiment:      ${EXPERIMENT}"
echo "Checkpoint:      ${CHECKPOINT:-none (fresh start)}"
echo "Repo dir:        ${REPO_DIR}"
echo "============================================="

# --- Build checkpoint argument ---
CHECKPOINT_ARG=""
if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARG="--checkpoint=${CHECKPOINT}"
fi

# --- Run training inside container via Pyxis ---
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:/workspaces/aerial_gym_docker,${REPO_DIR}/outputs:/workspace/aerial_gym_simulator/outputs,${REPO_DIR}/logs:/workspace/aerial_gym_simulator/logs" \
    --container-workdir=/workspaces/aerial_gym_docker \
    bash -c "
        echo 'Container started, Python: \$(python --version)'
        echo 'Isaac Gym check...'
        python -c 'from isaacgym import gymapi; print(\"Isaac Gym OK\")'
        echo 'Starting training...'
        python -m navigation_with_obstacles.training.runner \
            --file=${CONFIG_FILE} \
            --headless=True \
            --train \
            --track \
            --experiment_name=${EXPERIMENT} \
            --wandb-project-name=${WANDB_PROJECT} \
            ${CHECKPOINT_ARG}
    "

echo "Training complete. Job ${SLURM_JOB_ID} finished at $(date)"
