# Override the checkpoint with:  CKPT=/path/to.pth bash infer_hover_snn.bash
# Extra runner flags pass straight through, e.g. --plot-encoding (forces num_envs=1).
CKPT=${CKPT:-/workspaces/aerial_gym_docker/runs/f450_hover_snn/2026-08-24_22-21-00/nn/f450_hover_snn.pth}

cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_hover_local.yaml \
    --task=f450_hover_task \
    --checkpoint="${CKPT}" \
    --num_envs=8 \
    --headless=False \
    --use_warp=False \
    --play "$@"
