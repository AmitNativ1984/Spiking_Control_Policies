
cd /workspaces/aerial_gym_docker
cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/ppo_hover_local.yaml \
    --task=f450_hover_task \
    --checkpoint=/workspaces/aerial_gym_docker/runs/f450_hover_2026-08-11_06-49-35/nn/last_f450_hover_ep_400_rew_-214.48932.pth \
    --num_envs=2 \
    --headless=False \
    --use_warp=False \
    --play