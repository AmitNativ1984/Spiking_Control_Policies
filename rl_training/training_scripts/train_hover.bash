
cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/ppo_hover_local.yaml \
    --task=f450_hover_task \
    --experiment_name=f450_hover \
    --num_envs=4096 \
    --headless=True \
    --use_warp=False \
    --train


# cd /workspaces/aerial_gym_docker
# python -m rl_training.rl_games.runner \
#     --file=rl_training/rl_games/cfg/ppo_hover_local.yaml \
#     --task=f450_hover_task \
#     --checkpoint=runs/f450_hover_2026-08-11_06-49-35/nn/f450_hover.pth \
#     --num_envs=1 \
#     --headless=False \
#     --use_warp=False \
#     --play