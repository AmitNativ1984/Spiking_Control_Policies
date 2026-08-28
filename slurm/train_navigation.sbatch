#!/bin/bash
#SBATCH --job-name=f450-nav
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/f450-nav_%j.out
#SBATCH --error=slurm_logs/f450-nav_%j.err

# =============================================================================
# F450 navigation-with-obstacles training on SLURM with Pyxis
# =============================================================================
#
# Entry point: rl_training.rl_games.runner  (the aerial_gym-structured tree:
# config/ registries + task/attitude_navigation_task.py). This is NOT the legacy
# navigation_with_obstacles.training.runner -- see
# navigation_with_obstacles/slurm/ for that one. Importing both in one process
# raises on duplicate rl_games network names, so never mix the two.
#
# Usage:
#   sbatch slurm/train_navigation.sbatch                       # ANN MLP PPO
#
# Override defaults (all via environment):
#   CONFIG_FILE=rl_training/rl_games/cfg/popsan_cluster.yaml \
#   EXPERIMENT=f450_nav_snn sbatch slurm/train_navigation.sbatch
#
#   CHECKPOINT=runs/f450_nav_ann/<ts>/nn/f450_nav_ann.pth \
#     sbatch slurm/train_navigation.sbatch                     # resume
#
#   CURRICULUM_LEVEL=25 NUM_ENVS=512 SEED=3 \
#     sbatch slurm/train_navigation.sbatch
#
#   EXTRA_ARGS="--exceed_margin=1.5" sbatch slurm/train_navigation.sbatch
#
# num_actors (the env count) and max_epochs live in the YAML, so a config file
# reproduces an experiment on its own; NUM_ENVS is a VRAM escape hatch only.
#
# Prerequisites:
#   1. Docker/enroot image available (see slurm/README_cluster.md)
#   2. W&B API key in slurm/.env  (WANDB_API_KEY=...)
#   3. The DepthVAE checkpoint named by the task config exists, or its
#      vae_config.use_vae is False -- checked below before the GPU is touched.
# =============================================================================

set -euo pipefail

# --- Configuration (override via environment variables) ---
CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
CONFIG_FILE="${CONFIG_FILE:-rl_training/rl_games/cfg/ppo_mlp_cluster.yaml}"
TASK="${TASK:-f450_navigation_task}"
EXPERIMENT="${EXPERIMENT:-f450_nav_ann}"
WANDB_PROJECT="${WANDB_PROJECT:-${TASK}}"
CHECKPOINT="${CHECKPOINT:-}"              # resume training from this checkpoint
NUM_ENVS="${NUM_ENVS:-}"                  # empty => YAML's num_actors
SEED="${SEED:-0}"                         # >0 overrides the YAML
CURRICULUM_LEVEL="${CURRICULUM_LEVEL:-}"  # empty/<0 => curriculum advances freely
EXTRA_ARGS="${EXTRA_ARGS:-}"              # any further runner flags, verbatim

# --- Paths ---
# SLURM_SUBMIT_DIR is the directory where sbatch was invoked (the repo root).
REPO_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${REPO_DIR}/slurm"
CONTAINER_REPO=/workspaces/aerial_gym_docker
TASK_CONFIG="${REPO_DIR}/config/task_config/f450_attitude_navigation_task_config.py"

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

# --- Preflight: the VAE checkpoint is hard-coded in the task config, and the run
#     dies only after the simulator has finished building. Catch it here instead. ---
if [[ -f "${TASK_CONFIG}" ]]; then
    USE_VAE=$(sed -n 's/^[[:space:]]*use_vae[[:space:]]*=[[:space:]]*\([A-Za-z]*\).*/\1/p' "${TASK_CONFIG}" | head -1)
    VAE_CKPT=$(sed -n 's/^[[:space:]]*model_file[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "${TASK_CONFIG}" | head -1)
    if [[ "${USE_VAE}" == "True" && -n "${VAE_CKPT}" ]]; then
        VAE_CKPT_HOST="${REPO_DIR}${VAE_CKPT#${CONTAINER_REPO}}"
        if [[ ! -f "${VAE_CKPT_HOST}" ]]; then
            echo "ERROR: task config sets use_vae = True but its checkpoint is missing:" >&2
            echo "       ${VAE_CKPT}" >&2
            echo "       (looked at ${VAE_CKPT_HOST})" >&2
            echo "       Copy it to the cluster, repoint vae_config.model_file, or set" >&2
            echo "       vae_config.use_vae = False to train the state-only 17-D policy." >&2
            exit 1
        fi
        echo "VAE checkpoint OK: ${VAE_CKPT}"
    fi
fi

# --- Optional flags ---
OPT_ARGS=()
[[ -n "${CHECKPOINT}" ]] && OPT_ARGS+=("--checkpoint=${CHECKPOINT}")
[[ -n "${NUM_ENVS}" ]] && OPT_ARGS+=("--num_envs=${NUM_ENVS}")
[[ -n "${CURRICULUM_LEVEL}" && "${CURRICULUM_LEVEL}" -ge 0 ]] && \
    OPT_ARGS+=("--curriculum_level=${CURRICULUM_LEVEL}")
# shellcheck disable=SC2206  # word splitting is the point: EXTRA_ARGS is a flag list
[[ -n "${EXTRA_ARGS}" ]] && OPT_ARGS+=(${EXTRA_ARGS})

# --- Print job info ---
echo "============================================="
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Node:              ${SLURMD_NODENAME}"
echo "GPU:               ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Container:         ${CONTAINER_IMAGE}"
echo "Entry point:       rl_training.rl_games.runner"
echo "Config:            ${CONFIG_FILE}"
echo "Task:              ${TASK}"
echo "Experiment:        ${EXPERIMENT}"
echo "W&B project:       ${WANDB_PROJECT}"
echo "Num envs:          ${NUM_ENVS:-from YAML num_actors}"
echo "Seed:              ${SEED}"
echo "Curriculum level:  ${CURRICULUM_LEVEL:-free}"
echo "Checkpoint:        ${CHECKPOINT:-none (fresh start)}"
echo "Extra args:        ${EXTRA_ARGS:-none}"
echo "Repo dir:          ${REPO_DIR}"
echo "============================================="

mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/runs"

# --- Run training inside the container via Pyxis ---
srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:${CONTAINER_REPO}" \
    --container-workdir="${CONTAINER_REPO}" \
    bash -c "
        echo 'Container started, Python: \$(python --version)'
        python -c 'from isaacgym import gymapi; print(\"Isaac Gym OK\")'
        echo 'Starting training...'
        python -m rl_training.rl_games.runner \
            --file=${CONFIG_FILE} \
            --task=${TASK} \
            --experiment_name=${EXPERIMENT} \
            --headless=True \
            --seed=${SEED} \
            --train \
            --track \
            --wandb-project-name=${WANDB_PROJECT} \
            ${OPT_ARGS[*]}
    "

echo "Training complete. Job ${SLURM_JOB_ID} finished at $(date)"
echo "Run artifacts: ${REPO_DIR}/runs/${EXPERIMENT}/<timestamp>/{nn,summaries,config.yaml}"
