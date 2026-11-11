#!/bin/bash
#SBATCH --job-name=event-snn
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=dgx01
#SBATCH --time=1-12:00:00   # short limits let the scheduler backfill; pass --time=00:15:00 for test/check/heldout
#SBATCH --output=slurm_logs/event-snn_%j.out
#SBATCH --error=slurm_logs/event-snn_%j.err

# =============================================================================
# event_snn on SLURM with Pyxis: unit tests, teacher/target check, held-out set,
# encoder pretraining, evaluation. One GPU per job.
#
#   MODE=test    sbatch slurm/event_snn.sbatch       # pytest tests/test_event_snn.py
#   MODE=check   sbatch slurm/event_snn.sbatch       # teacher flight + invTTC frame check + throughput
#   MODE=heldout sbatch slurm/event_snn.sbatch       # fixed evaluation flights (once per level/seed)
#   MODE=train VARIANT=bn_both SEED=0 sbatch slurm/event_snn.sbatch
#   MODE=eval  CKPT=runs/event_snn/<run>/best.pt sbatch slurm/event_snn.sbatch
#
# NUM_ENVS (default 64) and SET="key=value ..." (dotted overrides, see event_snn/cfg/default.yaml)
# apply to check / heldout / train. A train run evaluates its best checkpoint when it ends.
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
MODE="${MODE:-train}"
VARIANT="${VARIANT:-bn_both}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
SECONDS_CHECK="${SECONDS_CHECK:-30}"
SET="${SET:-}"
CKPT="${CKPT:-}"
RUN_NAME="${RUN_NAME:-${VARIANT}_s${SEED}_${SLURM_JOB_ID}}"

REPO_DIR="${SLURM_SUBMIT_DIR}"
CONTAINER_REPO=/workspaces/aerial_gym_docker
ENV_FILE="${REPO_DIR}/slurm/.env"
if [[ -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi

case "${MODE}" in
  test)    CMD="PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_event_snn.py -q" ;;
  check)   CMD="python -m event_snn.sim --check_teacher --seconds ${SECONDS_CHECK} --set sim.num_envs=${NUM_ENVS} ${SET}" ;;
  heldout) CMD="python -m event_snn.sim --heldout --set sim.num_envs=${NUM_ENVS} ${SET}" ;;
  train)   export WANDB_NOTES="${WANDB_NOTES:-event encoder pretraining, ${VARIANT} seed ${SEED}, ${NUM_ENVS} envs, level 20; teacher f450_nav_lat_pfov; job ${SLURM_JOB_ID}}"
           CMD="python -m event_snn.train --variant ${VARIANT} --set seed=${SEED} run_name=${RUN_NAME} sim.num_envs=${NUM_ENVS} train.wandb=true ${SET} \
                && python -m event_snn.evaluate runs/event_snn/${RUN_NAME}/best.pt" ;;
  eval)    [[ -n "${CKPT}" ]] || { echo "MODE=eval needs CKPT" >&2; exit 1; }
           CMD="python -m event_snn.evaluate ${CKPT}" ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 1 ;;
esac

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: MODE=${MODE} VARIANT=${VARIANT} SEED=${SEED} NUM_ENVS=${NUM_ENVS} SET='${SET}'"
echo "${CMD}"
mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/runs/event_snn"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:${CONTAINER_REPO}" \
    --container-workdir="${CONTAINER_REPO}" \
    bash -c "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; ${CMD}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
