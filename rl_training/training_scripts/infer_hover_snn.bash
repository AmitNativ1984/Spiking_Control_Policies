# Override the checkpoint with:  CKPT=/path/to.pth bash infer_hover_snn.bash
# Extra runner flags pass straight through, e.g. --plot-encoding (forces num_envs=1).
EXPERIMENT_PATH=${EXPERIMENT_PATH:-/workspaces/aerial_gym_docker/runs/f450_hover_snn/2026-08-24_22-21-00}
CKPT=${CKPT:-${EXPERIMENT_PATH}/nn/f450_hover_snn.pth}
FILE=${FILE:-${EXPERIMENT_PATH}/config.yaml}

cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file="${FILE}" \
    --task=f450_hover_task \
    --checkpoint="${CKPT}" \
    --num_envs=64 \
    --headless=False \
    --use_warp=False \
    --play "$@" \
    --plot-encoding \
    --max_episodes=200 \
    --max_episode_steps=150 \
