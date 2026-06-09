#!/bin/bash
#SBATCH --job-name=hover-popsan
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/hover-popsan_%j.out
#SBATCH --error=slurm_logs/hover-popsan_%j.err

# =============================================================================
# Simple Hover SNN (PopSAN) Training on SLURM with Pyxis
# =============================================================================
#
# Usage:
#   sbatch simple_hover_snn/slurm/train_hover_snn.sbatch
#
# Override defaults:
#   CONTAINER_IMAGE=aerial-gym:latest NUM_ENVS=2048 sbatch simple_hover_snn/slurm/train_hover_snn.sbatch
#
# Prerequisites:
#   1. Docker image built and available (see slurm/README_cluster.md)
#   2. W&B API key in slurm/.env (shared repo-root env file)
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
NUM_ENVS="${NUM_ENVS:-4096}"  # Mirrors num_envs/num_actors in the YAML config
EXPERIMENT="${EXPERIMENT:-hover_popsan_cluster}"
CONFIG_FILE="${CONFIG_FILE:-simple_hover_snn/training/ppo_hover_popsan.yaml}"
CHECKPOINT="${CHECKPOINT:-}"  # Path to checkpoint file to resume training (empty = start fresh)

# --- Paths ---
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked (the repo root)
REPO_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_DIR}/slurm"

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
echo "Num envs:        ${NUM_ENVS}"
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
        python -m simple_hover_snn.training.runner \
            --file=${CONFIG_FILE} \
            --num_envs=${NUM_ENVS} \
            --headless=True \
            --train \
            --track \
            --experiment_name=${EXPERIMENT} \
            --wandb-project-name=aerial_gym_snn \
            ${CHECKPOINT_ARG}
    "

echo "Training complete. Job ${SLURM_JOB_ID} finished at $(date)"
