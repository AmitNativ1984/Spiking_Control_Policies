#!/bin/bash
#SBATCH --job-name=nav_warmup_snn
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/nav_warmup_snn_%j.out
#SBATCH --error=slurm_logs/nav_warmup_snn_%j.err

# =============================================================================
# BC warm-up (ANN teacher -> PopSAN SNN student) on SLURM with Pyxis
# =============================================================================
#
# Behavior-clones a trained ANN teacher's action mean into the spiking PopSAN
# student, then saves an rl-games-format checkpoint that PPO can resume via
# --checkpoint. See navigation_with_obstacles/agents/warmup_snn_from_ann.py.
#
# The script internally spawns the observation-bounds collector in a subprocess
# (Isaac Gym allows one sim per process); both run inside this container.
# Metrics go to TensorBoard and, with --track, to Weights & Biases
# (sync_tensorboard mirrors every warmup/* scalar).
#
# Usage:
#   sbatch slurm/warmup_snn_from_ann.sbatch
#
# Override defaults (num_envs comes from CONFIG_FILE's config.env_config.num_envs):
#   MAX_STEPS=2000000 CURRICULUM_LEVEL=25 \
#     sbatch slurm/warmup_snn_from_ann.sbatch
#
# Prerequisites:
#   1. Docker image built and available (see slurm/README_cluster.md)
#   2. W&B API key in slurm/.env (see README)
#   3. Teacher checkpoint referenced in CONFIG_FILE's config.distillation.checkpoint
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
CONFIG_FILE="${CONFIG_FILE:-navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml}"
MAX_STEPS="${MAX_STEPS:-2000000}"
CURRICULUM_LEVEL="${CURRICULUM_LEVEL:-25}"   # teacher's final level; <0 leaves curriculum free
EVAL_EVERY="${EVAL_EVERY:-50000}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-800}"
SOLO_SUCCESS_THRESHOLD="${SOLO_SUCCESS_THRESHOLD:-0.2}"
ANNEAL_STEPS="${ANNEAL_STEPS:-1000000}"
LR="${LR:-1e-3}"
# BC replay buffer: K grad steps/env step on i.i.d. minibatches decouples gradient throughput
# from the slow sim (the main BC-convergence speedup). BATCH_SIZE empty => num_envs (VRAM-neutral).
BUFFER_SIZE="${BUFFER_SIZE:-200000}"
GRAD_STEPS_PER_ENV_STEP="${GRAD_STEPS_PER_ENV_STEP:-8}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARNING_STARTS="${LEARNING_STARTS:-4096}"
BOUNDS_STEPS="${BOUNDS_STEPS:-10000}"
BOUNDS_EPISODES="${BOUNDS_EPISODES:-100}"     # also collect >= this many completed episodes
BOUNDS_ENVS="${BOUNDS_ENVS:-64}"              # envs for the bounds subprocess (memory bound)
REUSE_BOUNDS="${REUSE_BOUNDS:-0}"             # bounds RECOMPUTE every run by default; set 1 to reuse a matching cache
EXPERIMENT="${EXPERIMENT:-warmup_snn_cluster}"
WANDB_PROJECT="${WANDB_PROJECT:-ann_teacher_distill_warmup}"
# OUT empty => the script groups all artifacts (checkpoint, TensorBoard, encoder plots)
# under a single runs/warmup_snn_<timestamp>/ dir. Set OUT=path.pth to override.
OUT="${OUT:-}"

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

# --- Build optional --reuse_bounds flag (default: recompute every run) ---
REUSE_ARG=""
if [[ "${REUSE_BOUNDS}" == "1" ]]; then
    REUSE_ARG="--reuse_bounds"
fi

# --- Build optional --out flag (empty => script picks <run_dir>/nn/warmup_snn.pth) ---
OUT_ARG=""
if [[ -n "${OUT}" ]]; then
    OUT_ARG="--out=${OUT}"
fi

# --- Build optional --batch_size flag (empty => script uses num_envs) ---
BATCH_SIZE_ARG=""
if [[ -n "${BATCH_SIZE}" ]]; then
    BATCH_SIZE_ARG="--batch_size=${BATCH_SIZE}"
fi

# --- Print job info ---
echo "============================================="
echo "Job ID:              ${SLURM_JOB_ID}"
echo "Node:                ${SLURMD_NODENAME}"
echo "GPU:                 ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Container:           ${CONTAINER_IMAGE}"
echo "Config:              ${CONFIG_FILE}"
echo "Num envs:            (from YAML config.env_config.num_envs)"
echo "Max steps:           ${MAX_STEPS}"
echo "Curriculum level:    ${CURRICULUM_LEVEL}"
echo "Eval every / steps:  ${EVAL_EVERY} / ${EVAL_MAX_STEPS}"
echo "Solo threshold:      ${SOLO_SUCCESS_THRESHOLD}"
echo "Anneal steps:        ${ANNEAL_STEPS}"
echo "LR:                  ${LR}"
echo "BC buffer/K/batch:   ${BUFFER_SIZE} / ${GRAD_STEPS_PER_ENV_STEP} / ${BATCH_SIZE:-num_envs} (learning_starts=${LEARNING_STARTS})"
echo "Bounds steps/eps:    ${BOUNDS_STEPS} / ${BOUNDS_EPISODES} (envs=${BOUNDS_ENVS}, reuse=${REUSE_BOUNDS})"
echo "W&B project:         ${WANDB_PROJECT}"
echo "Output checkpoint:   ${OUT:-<run_dir>/nn/warmup_snn.pth (auto)}"
echo "Repo dir:            ${REPO_DIR}"
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
        echo 'Starting BC warm-up...'
        python -m navigation_with_obstacles.agents.warmup_snn_from_ann \
            --file=${CONFIG_FILE} \
            --headless=True \
            --max_steps=${MAX_STEPS} \
            --curriculum_level=${CURRICULUM_LEVEL} \
            --eval \
            --eval_every=${EVAL_EVERY} \
            --eval_max_steps=${EVAL_MAX_STEPS} \
            --solo_success_threshold=${SOLO_SUCCESS_THRESHOLD} \
            --anneal_steps=${ANNEAL_STEPS} \
            --lr=${LR} \
            --buffer_size=${BUFFER_SIZE} \
            --grad_steps_per_env_step=${GRAD_STEPS_PER_ENV_STEP} \
            --learning_starts=${LEARNING_STARTS} \
            ${BATCH_SIZE_ARG} \
            --bounds_steps=${BOUNDS_STEPS} \
            --bounds_episodes=${BOUNDS_EPISODES} \
            --bounds_envs=${BOUNDS_ENVS} \
            ${REUSE_ARG} \
            --track \
            --experiment_name=${EXPERIMENT} \
            --wandb-project-name=${WANDB_PROJECT} \
            ${OUT_ARG}
    "

echo "Warm-up complete. Job ${SLURM_JOB_ID} finished at $(date)"
if [[ -n "${OUT}" ]]; then
    echo "Checkpoint saved to ${REPO_DIR}/${OUT}"
else
    echo "Artifacts (checkpoint/nn, TensorBoard/summaries, encoder plots) under ${REPO_DIR}/navigation_with_obstacles/runs/warmup_snn_<timestamp>/"
fi
