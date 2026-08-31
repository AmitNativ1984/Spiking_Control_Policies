# Playback of a navigation policy. Override with:
#   CKPT=/path/to.pth bash infer_navigation.bash
#   EXPERIMENT_PATH=/workspaces/aerial_gym_docker/runs/f450_nav_snn/<timestamp> bash ...
# Extra runner flags pass straight through, e.g. --curriculum_level=25.
#
# FILE is the run's FROZEN config, not cfg/ — the YAML drifts, the run does not.
# use_warp stays True: the task refuses use_warp=False while the VAE is on, because the
# Isaac Gym camera never re-renders and every latent would freeze.
EXPERIMENT_PATH=${EXPERIMENT_PATH:-/workspaces/aerial_gym_docker/runs/f450_nav_ann/2026-08-30_11-18-55}
CKPT=${CKPT:-${EXPERIMENT_PATH}/nn/last_f450_nav_ann_ep_3350_rew_21.926056.pth}
FILE=${FILE:-${EXPERIMENT_PATH}/config.yaml}

cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file="${FILE}" \
    --task=f450_navigation_task \
    --checkpoint="${CKPT}" \
    --num_envs=${NUM_ENVS:-4} \
    --headless=False \
    --play "$@"
