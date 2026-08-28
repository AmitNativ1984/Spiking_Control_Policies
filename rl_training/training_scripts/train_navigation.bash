# ANN PPO on the navigation task. Switch architecture without editing this file:
#   FILE=rl_training/rl_games/cfg/ppo_gru_local.yaml bash train_navigation.bash
#   FILE=rl_training/rl_games/cfg/popsan_local.yaml  EXPERIMENT=f450_nav_snn ...
# Extra runner flags pass straight through, e.g. --curriculum_level=25.
#
# use_warp stays True (the runner's default): the task refuses use_warp=False while the
# VAE is on, because the Isaac Gym camera never re-renders and every latent would freeze.
TASK=f450_navigation_task
FILE=${FILE:-rl_training/rl_games/cfg/ppo_mlp_local.yaml}
EXPERIMENT=${EXPERIMENT:-f450_nav_ann}
TRACK=${TRACK:-1}

cd /workspaces/aerial_gym_docker

TRACK_ARGS=()
if [[ "${TRACK}" != "0" ]]; then
    # Same key file the SLURM job reads (gitignored). Env wins if already exported.
    if [[ -z "${WANDB_API_KEY}" && -f slurm/.env ]]; then
        set -a; source slurm/.env; set +a
    fi
    if [[ -z "${WANDB_API_KEY}" ]]; then
        echo "train_navigation: WANDB_API_KEY not set and slurm/.env has no key." >&2
        echo "  export it, or rerun with TRACK=0 to train without W&B." >&2
        exit 1
    fi
    TRACK_ARGS=(--track --wandb-project-name="${TASK}")
fi

python -m rl_training.rl_games.runner \
    --file="${FILE}" \
    --task="${TASK}" \
    --experiment_name="${EXPERIMENT}" \
    --headless=True \
    --train "${TRACK_ARGS[@]}" "$@"


# Playback replays the run's FROZEN config, not cfg/ — the YAML drifts, the run does not.
#
# EXPERIMENT_PATH=/workspaces/aerial_gym_docker/runs/f450_nav_ann/<timestamp>
# cd /workspaces/aerial_gym_docker
# python -m rl_training.rl_games.runner \
#     --file="${EXPERIMENT_PATH}/config.yaml" \
#     --task=f450_navigation_task \
#     --checkpoint="${EXPERIMENT_PATH}/nn/f450_nav_ann.pth" \
#     --num_envs=1 \
#     --headless=False \
#     --play
