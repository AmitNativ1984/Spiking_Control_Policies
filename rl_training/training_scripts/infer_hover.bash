
cd /workspaces/aerial_gym_docker
cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/ppo_hover_local.yaml \
    --task=f450_hover_task \
    --checkpoint=/workspaces/aerial_gym_docker/runs/f450_hover/2026-08-11_14-50-32/nn/last_f450_hover_ep_1000_rew_2253.7856.pth \
    --num_envs=8 \
    --headless=False \
    --use_warp=False \
    --play