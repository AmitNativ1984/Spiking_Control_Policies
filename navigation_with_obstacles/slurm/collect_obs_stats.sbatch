#!/bin/bash
#SBATCH --job-name=nav-obs-stats
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/nav-obs-stats_%j.out
#SBATCH --error=slurm_logs/nav-obs-stats_%j.err

# =============================================================================
# Observation Statistics Collection on SLURM with Pyxis
# =============================================================================
#
# Runs the navigation env with random actions for N steps, then saves
# per-dim observation statistics (CSV) and histograms (TensorBoard + W&B).
#
# Usage:
#   sbatch slurm/collect_obs_stats.sbatch
#
# Override defaults:
#   NUM_STEPS=50000 NUM_ENVS=128 sbatch slurm/collect_obs_stats.sbatch
#
# Prerequisites:
#   1. Docker image built and available (see slurm/README_cluster.md)
#   2. W&B API key in slurm/.env (see README)
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
NUM_STEPS="${NUM_STEPS:-10000}"
NUM_ENVS="${NUM_ENVS:-64}"
OUT_DIR="${OUT_DIR:-navigation_with_obstacles/obs_stats/${SLURM_JOB_ID}}"

# --- Paths ---
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
    echo "WARNING: ${ENV_FILE} not found. W&B logging will fail unless WANDB_API_KEY is set."
fi

# --- Print job info ---
echo "============================================="
echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            ${SLURMD_NODENAME}"
echo "GPU:             ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Container:       ${CONTAINER_IMAGE}"
echo "Num steps:       ${NUM_STEPS}"
echo "Num envs:        ${NUM_ENVS}"
echo "Output dir:      ${OUT_DIR}"
echo "Repo dir:        ${REPO_DIR}"
echo "============================================="

# --- Run inside container via Pyxis ---
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:/workspaces/aerial_gym_docker,${REPO_DIR}/outputs:/workspace/aerial_gym_simulator/outputs,${REPO_DIR}/logs:/workspace/aerial_gym_simulator/logs" \
    --container-workdir=/workspaces/aerial_gym_docker \
    bash -c "
        echo 'Container started, Python: \$(python --version)'
        echo 'Isaac Gym check...'
        python -c 'from isaacgym import gymapi; print(\"Isaac Gym OK\")'
        echo 'Starting observation stats collection...'
        python -m navigation_with_obstacles.tools.collect_obs_stats \
            --num_steps=${NUM_STEPS} \
            --num_envs=${NUM_ENVS} \
            --out_dir=${OUT_DIR}
    "
echo "Collection complete. Job ${SLURM_JOB_ID} finished at $(date)"
echo "Results saved to ${REPO_DIR}/${OUT_DIR}"
