#!/bin/bash
#SBATCH --job-name=f450-crash-eval
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/crash-eval_%j.out
#SBATCH --error=slurm_logs/crash-eval_%j.err

# =============================================================================
# Deterministic crash-cause eval of a checkpoint at a fixed curriculum level.
# No training -- runs analysis/crash_cause_eval.py inside the same container used
# for training (see slurm/train_navigation.sbatch, which this mirrors).
#
# Usage:
#   sbatch slurm/crash_cause_eval.sbatch
#
# Override via environment:
#   CHECKPOINT=runs/.../nn/ckpt.pth LEVEL=30 NUM_ENVS=256 NUM_STEPS=6000 \
#     OUT=analysis/data/crash_causes_b4_level30.json \
#     sbatch slurm/crash_cause_eval.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
CONTAINER_REPO=/workspaces/aerial_gym_docker
REPO_DIR="${SLURM_SUBMIT_DIR}"

CHECKPOINT="${CHECKPOINT:-runs/f450_nav_b4_combined/2026-09-04_14-23-02/nn/last_f450_nav_b4_combined_ep_2000_rew_25.875937.pth}"
LEVEL="${LEVEL:-30}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-6000}"
OUT="${OUT:-analysis/data/crash_causes_$(basename "${CHECKPOINT}" .pth)_level${LEVEL}.json}"

mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/$(dirname "${OUT}")"

echo "============================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Level:       ${LEVEL}"
echo "Num envs:    ${NUM_ENVS}"
echo "Num steps:   ${NUM_STEPS}"
echo "Out:         ${OUT}"
echo "============================================="

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:${CONTAINER_REPO}" \
    --container-workdir="${CONTAINER_REPO}" \
    bash -c "
        echo 'Container started, Python: \$(python --version)'
        python -c 'from isaacgym import gymapi; print(\"Isaac Gym OK\")'
        python analysis/crash_cause_eval.py \
            ${CHECKPOINT} ${LEVEL} ${OUT} \
            --num_envs=${NUM_ENVS} --num_steps=${NUM_STEPS}
    "

echo "Eval complete. Job ${SLURM_JOB_ID} finished at $(date)"
echo "Output: ${REPO_DIR}/${OUT}"
