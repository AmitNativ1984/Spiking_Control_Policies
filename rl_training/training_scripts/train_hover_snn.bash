
cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_hover_local.yaml \
    --task=f450_hover_task \
    --experiment_name=f450_hover_snn \
    --num_envs=4096 \
    --headless=True \
    --use_warp=False \
    --train


# cd /workspaces/aerial_gym_docker
# python -m rl_training.rl_games.runner \
#     --file=rl_training/rl_games/cfg/popsan_hover_local.yaml \
#     --task=f450_hover_task \
#     --checkpoint=runs/f450_hover_snn/<timestamp>/nn/f450_hover_snn.pth \
#     --num_envs=1 \
#     --headless=False \
#     --use_warp=False \
#     --play
