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
#   MODE=overfit VARIANT=mb_ttc_ann sbatch --time=0:45:00 slurm/event_snn.sbatch   # all inputs / no events / no inertial
#   MODE=generalize VARIANT=mb_ttc_ann sbatch --time=1:30:00 slurm/event_snn.sbatch
#   MODE=e2e_smoke sbatch --time=0:30:00 slurm/event_snn.sbatch   # end-to-end PPO from events: 3 tiny iterations + both evaluations
#   MODE=e2e HOURS=30 sbatch slurm/event_snn.sbatch                # train (E2E_ENVS, default 256), then evaluate it and the teacher at level 20
#
# From a separate worktree (no runs/, no slurm/.env there) pass DATA_DIR=<main checkout>: its runs/ and
# vae_depth/runs/ are mounted over the worktree's and its slurm/.env is read.
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

HOURS="${HOURS:-30}"
E2E_ENVS="${E2E_ENVS:-256}"
E2E_NAME="${E2E_NAME:-e2e_ppo_s${SEED}_${SLURM_JOB_ID}}"

REPO_DIR="${SLURM_SUBMIT_DIR}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}}"
CONTAINER_REPO=/workspaces/aerial_gym_docker
ENV_FILE="${DATA_DIR}/slurm/.env"
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
  overfit|generalize)
           G=""; [[ "${MODE}" == generalize ]] && G="--generalize"
           CMD="python -m event_snn.overfit ${G} --variant ${VARIANT} --wandb --set ${SET} \
                ; python -m event_snn.overfit ${G} --variant ${VARIANT} --no_events --wandb --set ${SET} \
                ; python -m event_snn.overfit ${G} --variant ${VARIANT} --no_inertial --wandb --set ${SET}" ;;
  e2e_smoke)
           E2E="runs/event_snn/e2e_ppo/smoke_${SLURM_JOB_ID}"
           CMD="PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_event_snn.py -q -k e2e \
                && python -m event_snn.e2e_ppo --envs ${E2E_ENVS} --rollout 16 --iterations 3 --name smoke_${SLURM_JOB_ID} --set ${SET} \
                && python -m event_snn.e2e_ppo --eval ${E2E}/last.pt --envs 64 --episodes 30 --set ${SET} \
                && python -m event_snn.e2e_ppo --eval_teacher --envs 64 --episodes 60 --set ${SET}" ;;
  e2e)     E2E="runs/event_snn/e2e_ppo/${E2E_NAME}"
           N1="end-to-end navigation from event frames with PPO, non-spiking, after arXiv 2610.37543 (3 signed 64x64 event frames + 17-D state, 16/32/64 conv trunk, PPO rollout 128, 2 epochs, 8 minibatches, lr 3e-4); task, reward, actions and curriculum of the depth teacher f450_nav_lat_pfov; rendered events, ${E2E_ENVS} envs, ${HOURS} h budget; job ${SLURM_JOB_ID}"
           N2="closed-loop evaluation at level 20 of the end-to-end event PPO policy ${E2E_NAME}; job ${SLURM_JOB_ID}"
           N3="closed-loop evaluation at level 20 of the depth teacher f450_nav_lat_pfov in the event_snn.e2e_ppo loop, reference for ${E2E_NAME}; job ${SLURM_JOB_ID}"
           CMD="WANDB_NOTES='${N1}' python -m event_snn.e2e_ppo --envs ${E2E_ENVS} --hours ${HOURS} --seed ${SEED} --name ${E2E_NAME} --wandb --set ${SET} \
                ; WANDB_NOTES='${N2}' python -m event_snn.e2e_ppo --eval ${E2E}/last.pt --envs ${E2E_ENVS} --episodes 1000 --wandb --set ${SET} \
                ; WANDB_NOTES='${N3}' python -m event_snn.e2e_ppo --eval_teacher --envs ${E2E_ENVS} --episodes 1000 --wandb --set ${SET}" ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 1 ;;
esac

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: MODE=${MODE} VARIANT=${VARIANT} SEED=${SEED} NUM_ENVS=${NUM_ENVS} SET='${SET}'"
echo "${CMD}"
mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/runs" "${REPO_DIR}/vae_depth/runs" "${DATA_DIR}/runs/event_snn"

MOUNTS="${REPO_DIR}:${CONTAINER_REPO}"
if [[ "${DATA_DIR}" != "${REPO_DIR}" ]]; then   # worktree: untracked data comes from the main checkout
    MOUNTS="${MOUNTS},${DATA_DIR}/runs:${CONTAINER_REPO}/runs,${DATA_DIR}/vae_depth/runs:${CONTAINER_REPO}/vae_depth/runs"
fi
echo "mounts: ${MOUNTS}"

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${MOUNTS}" \
    --container-workdir="${CONTAINER_REPO}" \
    bash -c "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; ${CMD}"

echo "Job ${SLURM_JOB_ID} finished at $(date)"
