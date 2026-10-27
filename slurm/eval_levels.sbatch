#!/bin/bash
#SBATCH --job-name=f450-eval
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err

# =============================================================================
# Outcome breakdown (arrive / crash / timeout / exceed) of one checkpoint at one fixed
# curriculum level -- runs analysis/eval_levels.py inside the training container. MLP,
# GRU and PopSAN checkpoints all work. One level per job (Isaac Gym: one sim per process).
#
# Usage:
#   CHECKPOINT=runs/.../nn/ckpt.pth LEVEL=30 [SAMPLE=1] [NUM_ENVS=1024] [NUM_STEPS=2000] \
#     [OUT=analysis/data/eval.json] sbatch --exclude=dgx01 slurm/eval_levels.sbatch
#
# SAMPLE=1 draws mu + sigma*N(0,1) like training does; unset = deterministic mean action.
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
CONTAINER_REPO=/workspaces/aerial_gym_docker
REPO_DIR="${SLURM_SUBMIT_DIR}"

CHECKPOINT="${CHECKPOINT:?set CHECKPOINT=runs/.../nn/ckpt.pth}"
LEVEL="${LEVEL:-30}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_STEPS="${NUM_STEPS:-2000}"
SAMPLE="${SAMPLE:-}"
SEED="${SEED:-0}"
MODE=$([[ -n "${SAMPLE}" ]] && echo sampled || echo det)
OUT="${OUT:-analysis/data/eval_$(basename "${CHECKPOINT}" .pth)_${MODE}_level${LEVEL}.json}"
SAMPLE_FLAG=$([[ -n "${SAMPLE}" ]] && echo "--sample" || true)

mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/$(dirname "${OUT}")"

echo "============================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Level:       ${LEVEL}"
echo "Actions:     ${MODE} (seed ${SEED})"
echo "Num envs:    ${NUM_ENVS}"
echo "Num steps:   ${NUM_STEPS}"
echo "Out:         ${OUT}"
echo "============================================="

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:${CONTAINER_REPO}" \
    --container-workdir="${CONTAINER_REPO}" \
    bash -c "
        python -c 'from isaacgym import gymapi; print(\"Isaac Gym OK\")'
        python -u analysis/eval_levels.py \
            ${CHECKPOINT} ${LEVEL} \
            --num_envs=${NUM_ENVS} --num_steps=${NUM_STEPS} --seed=${SEED} \
            --out=${OUT} ${SAMPLE_FLAG}
    "

echo "Eval complete. Job ${SLURM_JOB_ID} finished at $(date)"
echo "Output: ${REPO_DIR}/${OUT}"
