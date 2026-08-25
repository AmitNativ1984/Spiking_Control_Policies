
# Measure PopSAN encoder bounds for the hover task from a trained ANN policy.
#
# Collection stops when BOTH floors are met (--num_steps AND --min_episodes); with
# --max_episode_steps=150, 256 episodes costs ~39k samples.
#
# --max_episode_steps is the important one. A converged policy parks on the target for
# ~90% of a full 667-step episode, so the pooled distribution collapses onto that one
# narrow mode and every mass-weighted estimator follows it: measured z_std halves and the
# resulting window clips the approach phase. Truncating to 150 steps resamples the
# approach across many fresh targets instead — z_std roughly doubles on every dim, and
# prev_action_3's window finally contains its known support (0.00% clipped, vs 5.84%).
#
# --plot_bounds=-5,5 draws the histograms on a FIXED frame instead of the computed window.
# The histogram is clipped to whatever it is drawn against, so plotting against the window
# under evaluation hides exactly the tails you are trying to judge.
#
# Writes runs/obs_stats/{obs_stats.csv, observation_bounds.json, encoder PNGs}.
# Paste observation_bounds into popsan_hover_local.yaml's network.actor block to use them.

cd /workspaces/aerial_gym_docker
python -m rl_training.rl_games.tools.collect_obs_stats \
    --config=rl_training/rl_games/cfg/popsan_hover_local.yaml \
    --teacher_config=rl_training/rl_games/cfg/ppo_hover_local.yaml \
    --teacher_checkpoint=/workspaces/aerial_gym_docker/runs/f450_hover/2026-08-11_14-50-32/nn/f450_hover.pth \
    --num_steps=20000 \
    --min_episodes=200 \
    --max_episode_steps=100 \
    --num_envs=64 \
    --bound_method=gaussian \
    --plot_bounds=-5,5 \
    --no_wandb


# gravity_2 is one-sided (z_min -0.393, z_max 5.0), so a window centred on the mean puts
# half its span where data never goes — that is the 2/10 silent columns still reported.
# percentile follows the real tails instead of imposing symmetry:
#
# cd /workspaces/aerial_gym_docker
# python -m rl_training.rl_games.tools.collect_obs_stats \
#     --config=rl_training/rl_games/cfg/popsan_hover_local.yaml \
#     --teacher_config=rl_training/rl_games/cfg/ppo_hover_local.yaml \
#     --teacher_checkpoint=/workspaces/aerial_gym_docker/runs/f450_hover/2026-08-11_14-50-32/nn/f450_hover.pth \
#     --num_steps=20000 \
#     --min_episodes=200 \
#     --max_episode_steps=150 \
#     --num_envs=64 \
#     --bound_method=percentile \
#     --lower_pct=1.0 \
#     --upper_pct=99.0 \
#     --no_wandb


# Reading the output: the NORMALIZED table is the one to use — observation_bounds are set
# in that space. z_min/z_max are the extremes the encoder will ever see (a window covering
# them clips nothing) and %clip is what the proposed window would discard.
#
# The four prev_action dims are CLAMPED to [-1, 1] raw, so their z_min/z_max are exact
# rather than estimated — set those from z_min/z_max directly and they clip 0%.
