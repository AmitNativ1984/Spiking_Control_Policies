#!/bin/bash
#SBATCH --job-name=f450-sigdiag
#SBATCH --partition=dlc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/sigdiag_%j.out
#SBATCH --error=slurm_logs/sigdiag_%j.err

# =============================================================================
# Why does the PopSAN student's sigma climb under PPO? Runs analysis/sigma_diag.py on three
# checkpoints, one process each (Isaac Gym: one sim per process), no training:
#   snn_ep450         the distilled student at KD release, where every fine-tune started
#   snn_lr1e4_ep1000  the same student after 550 epochs at constant lr 1e-4 (sigma ~0.29)
#   teacher           the ANN teacher f450_nav_lat_pfov (its sigma SHRANK under the same PPO)
# All at curriculum level 30 with the teacher's reward (F450_LAMBDA_FOV=0.015, BLIND=0).
#
# Usage:  sbatch --exclude=dgx01 slurm/sigma_diag.sbatch
#         [LRS="1e-6 1e-5 1e-4"] [BATCHES=3] [MODELS="snn_ep450 snn_lr1e4_ep1000 teacher"]
#         [WANDB_PROJECT=popsan_ann_teacher_distillation]  (empty = no W&B)  [NUM_ENVS=256 smoke test]
# Output: analysis/data/sigma_diag/<model>.json (+ _steps.json), one W&B run per model.
# =============================================================================

set -uo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-${HOME}/DOCKER_IMAGES/aerial-gym.sqsh}"
CONTAINER_REPO=/workspaces/aerial_gym_docker
REPO_DIR="${SLURM_SUBMIT_DIR}"

ENV_FILE="${REPO_DIR}/slurm/.env"
if [[ -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi

export F450_LAMBDA_FOV=0.015 F450_LAMBDA_BLIND=0.0
export LRS="${LRS:-1e-6 1e-5 1e-4}"
export BATCHES="${BATCHES:-3}"
export NUM_ENVS="${NUM_ENVS:-}"   # empty = the YAML's num_actors (1024); set only for smoke tests
export MODELS="${MODELS:-snn_ep450 snn_lr1e4_ep1000 teacher}"
export WANDB_PROJECT="${WANDB_PROJECT-popsan_ann_teacher_distillation}"
export WANDB_GROUP="sigma_diag_${SLURM_JOB_ID}"

export CFG_SNN=rl_training/rl_games/cfg/popsan_postrelease_ft_constlr_cluster.yaml
export CK_snn_ep450=runs/popsan_ann_teacher_distillation/2026-10-11_13-19-26/nn/last_popsan_ann_teacher_distillation_ep_450_rew_12.908483.pth
export CK_snn_lr1e4_ep1000=runs/popsan_ann_teacher_distillation_ft_constlr1e4_teacherreward/2026-10-13_09-08-11/nn/last_popsan_ann_teacher_distillation_ft_constlr1e4_teacherreward_ep_1000_rew_9.276099.pth
export CFG_TEACHER=runs/f450_nav_ann/f450_nav_lat_pfov/2026-10-07_13-37-08/config.yaml
export CK_teacher=runs/f450_nav_ann/f450_nav_lat_pfov/2026-10-07_13-37-08/nn/f450_nav_lat_pfov.pth

export NOTES_COMMON="SIGMA DIAGNOSTIC (no training; job ${SLURM_JOB_ID}, group ${WANDB_GROUP}). Why does the PopSAN student's action sigma climb under PPO while the ANN teacher's shrinks under the same env, reward and PPO settings? From the checkpoint, collect ${BATCHES} rollout batches (1024 envs x 128 steps, curriculum 30, teacher reward F450_LAMBDA_FOV=0.015 BLIND=0); on each, run one full PPO epoch (4 mini-epochs x 16 minibatches) per setting, lr in {${LRS}} x clip {0.2, off} x entropy_coef {0.001, 0} (SNN and teacher both trained with 0.001), every trial restarted from the same weights/optimizer/normalizers. Per minibatch step it records the log_std gradient of the clipped PPO loss, of the unclipped surrogate, and the part the clip removed (split by advantage sign); per trial the actual change of log_std and the mean-action change after ONE step and after the epoch (fraction unchanged, rms/p99/kurtosis in sigma units). push = -dLoss/dlog_std (>0 widens sigma); the entropy bonus alone gives +0.001. HYPOTHESES: (1) quantization -- the spiking mean moves in spike-count jumps (many samples unchanged, a few by ~0.4-0.5 sigma), unlike the smooth ANN; (2) clip censorship -- those advantage-unrelated jumps push samples out of the clip window, the clip removes mostly sigma-shrinking votes, so push_full > push_unclipped and log_std widens with the clip on but not with it off. Code: analysis/sigma_diag.py + slurm/sigma_diag.sbatch (uncommitted) on master 3cea015."
export NOTES_snn_ep450="MODEL: PopSAN student ep_450 of wavp4w4j (distilled, KD just released, sigma ~0.145), the start point of every fine-tune. Config popsan_postrelease_ft_constlr_cluster.yaml (KD off, entropy 0.001). ${NOTES_COMMON}"
export NOTES_snn_lr1e4_ep1000="MODEL: PopSAN student ep_1000 of j5skxllu (constant lr 1e-4 fine-tune, sigma ~0.29, success 0.81). Tests whether the push weakens at high sigma (a finer grid relative to sigma). Config popsan_postrelease_ft_constlr_cluster.yaml. ${NOTES_COMMON}"
export NOTES_teacher="MODEL: ANN teacher f450_nav_lat_pfov (epoch 2663, sigma ~0.07; its sigma shrank 0.087 -> 0.071 over its last 1000 training epochs). The control: same PPO settings, smooth mean. Config: the teacher run's own config.yaml (a2c_continuous). ${NOTES_COMMON}"

mkdir -p "${REPO_DIR}/slurm_logs" "${REPO_DIR}/analysis/data/sigma_diag"

echo "============================================="
echo "Job ID:   ${SLURM_JOB_ID}   Node: ${SLURMD_NODENAME}"
echo "Models:   ${MODELS}"
echo "LRs:      ${LRS}   Batches: ${BATCHES}"
echo "W&B:      ${WANDB_PROJECT:-off} (group ${WANDB_GROUP})"
echo "============================================="

srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts="${REPO_DIR}:${CONTAINER_REPO}" \
    --container-workdir="${CONTAINER_REPO}" \
    bash -c '
        python -c "from isaacgym import gymapi; print(\"Isaac Gym OK\")"
        for M in ${MODELS}; do
            CK_VAR="CK_${M}"; NOTES_VAR="NOTES_${M}"
            if [[ "${M}" == teacher ]]; then CFG="${CFG_TEACHER}"; else CFG="${CFG_SNN}"; fi
            WB=(); [[ -n "${WANDB_PROJECT}" ]] && WB=(--wandb_project "${WANDB_PROJECT}" --wandb_group "${WANDB_GROUP}" --wandb_notes "${!NOTES_VAR}")
            echo "===== ${M}: ${!CK_VAR}"
            python -u analysis/sigma_diag.py --label "${M}" \
                --out "analysis/data/sigma_diag/${M}.json" --lrs ${LRS} --batches "${BATCHES}" "${WB[@]}" -- \
                --file="${CFG}" --task=f450_navigation_task --checkpoint="${!CK_VAR}" \
                --curriculum_level=30 --headless=True --train --experiment_name=sigma_diag \
                ${NUM_ENVS:+--num_envs=${NUM_ENVS}} \
                || echo "===== ${M} FAILED (exit $?)"
        done
    '

echo "Sigma diagnostic complete. Job ${SLURM_JOB_ID} finished at $(date)"
