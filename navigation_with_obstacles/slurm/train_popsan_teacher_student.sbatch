#!/bin/bash
#SBATCH --job-name=nav_popsan_distill
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/nav_popsan_distill_%j.out
#SBATCH --error=slurm_logs/nav_popsan_distill_%j.err

# =============================================================================
# PopSAN Teacher->Student PPO Distillation on SLURM with Pyxis
# =============================================================================
#
# Continues from a BC warm-up checkpoint and runs PPO with an annealed
# ANN-teacher -> SNN-student distillation tail (algo: a2c_teacher; see
# navigation_with_obstacles/agents/a2c_teacher_agent.py). The CHECKPOINT is the
# warm-up SNN student (NOT the teacher); the teacher is loaded internally from
# the YAML's config.distillation.checkpoint. Restoring the warm-up checkpoint
# carries over the trained input-normalization stats, population encoder, and
# actor mu, so distillation starts where warm-up left off.
#
# Usage:
#   CHECKPOINT=navigation_with_obstacles/runs/warmup_snn_<ts>/nn/warmup_snn.pth \
#     sbatch slurm/train_popsan_teacher_student.sbatch
#
# Override defaults (max_epochs lives in the YAML's config.max_epochs):
#   CURRICULUM_LEVEL=25 CHECKPOINT=.../warmup_snn.pth \
#     sbatch slurm/train_popsan_teacher_student.sbatch
#
#   # Leave the curriculum free to advance (instead of pinning at level 25):
#   CURRICULUM_LEVEL=-1 CHECKPOINT=.../warmup_snn.pth \
#     sbatch slurm/train_popsan_teacher_student.sbatch
#
# Prerequisites:
#   1. Docker image built and available (see slurm/README_cluster.md)
#   2. W&B API key in slurm/.env (see README)
#   3. A BC warm-up checkpoint (run slurm/warmup_snn_from_ann.sbatch first)
#   4. Teacher checkpoint referenced in CONFIG_FILE's config.distillation.checkpoint
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
CONFIG_FILE="${CONFIG_FILE:-navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml}"
CURRICULUM_LEVEL="${CURRICULUM_LEVEL:-25}"   # pin obstacle density at the teacher's final level; <0 leaves it free
EXPERIMENT="${EXPERIMENT:-nav_popsan_distill_cluster}"
WANDB_PROJECT="${WANDB_PROJECT:-popsan_ann_teacher_distill}"
# Warm-up SNN student checkpoint to resume from. Required for a proper warm-start;
# without it the student trains from a randomly-initialized actor.
CHECKPOINT="${CHECKPOINT:-navigation_with_obstacles/runs/warmup_snn_2026-06-25_21-54-13/nn/warmup_snn.pth}"

# --- Paths ---
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked
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

# --- Build checkpoint argument (warn loudly if missing: warm-start is the point) ---
CHECKPOINT_ARG=""
if [[ -n "${CHECKPOINT}" ]]; then
    CHECKPOINT_ARG="--checkpoint=${CHECKPOINT}"
else
    echo "WARNING: no CHECKPOINT set. Distillation will start from a randomly-initialized"
    echo "         SNN actor (no warm-start). Set CHECKPOINT=.../warmup_snn.pth to resume."
fi

# --- Build optional --curriculum_level flag (empty/<0 => leave curriculum free) ---
CURRICULUM_ARG=""
if [[ -n "${CURRICULUM_LEVEL}" && "${CURRICULUM_LEVEL}" -ge 0 ]]; then
    CURRICULUM_ARG="--curriculum_level=${CURRICULUM_LEVEL}"
fi

# --- Print job info ---
echo "============================================="
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Node:                ${SLURMD_NODENAME}"
echo "GPU:                 ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Container:           ${CONTAINER_IMAGE}"
echo "Config:              ${CONFIG_FILE}"
echo "Max epochs:          (from YAML config.max_epochs)"
echo "Curriculum level:    ${CURRICULUM_LEVEL} (<0 => free)"
echo "Experiment:          ${EXPERIMENT}"
echo "W&B project:         ${WANDB_PROJECT}"
echo "Warm-up checkpoint:  ${CHECKPOINT:-none (fresh actor!)}"
echo "Repo dir:            ${REPO_DIR}"
echo "============================================="

# --- Run distillation inside container via Pyxis ---
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:/workspaces/aerial_gym_docker,${REPO_DIR}/outputs:/workspace/aerial_gym_simulator/outputs,${REPO_DIR}/logs:/workspace/aerial_gym_simulator/logs" \
    --container-workdir=/workspaces/aerial_gym_docker \
    bash -c "
        echo 'Container started, Python: \$(python --version)'
        echo 'Isaac Gym check...'
        python -c 'from isaacgym import gymapi; print(\"Isaac Gym OK\")'
        echo 'Starting teacher-student PPO distillation...'
        python -m navigation_with_obstacles.training.runner \
            --file=${CONFIG_FILE} \
            --headless=True \
            --train \
            ${CURRICULUM_ARG} \
            ${CHECKPOINT_ARG} \
            --track \
            --experiment_name=${EXPERIMENT} \
            --wandb-project-name=${WANDB_PROJECT}
    "

echo "Distillation complete. Job ${SLURM_JOB_ID} finished at $(date)"
