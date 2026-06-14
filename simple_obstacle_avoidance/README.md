# Simple Obstacle Avoidance Task

A first step toward vision-based navigation: the quadrotor must **fly to a waypoint at the
far end of the room while avoiding a small, fixed number of obstacles**, using a depth
camera compressed through a Variational Autoencoder (VAE).

It is a deliberately simplified version of [`navigation_with_obstacles`](../navigation_with_obstacles/):
no curriculum (a fixed obstacle count), a simpler reward, and a longer episode budget to make
learning easier.

---

## Theory

This task adds two hard problems on top of plain hovering:

1. **Goal-directed flight.** Instead of regulating to zero motion, the agent must make
   *progress* toward a target waypoint. The observation gives a unit vector to the target
   (in the vehicle frame) and a normalized distance, and the reward shapes behaviour with a
   dense progress signal so the policy gets continuous feedback as it closes the gap.

2. **Perception from depth.** Obstacles are only observable through a forward-facing depth
   camera. A raw depth image (270×480) is far too high-dimensional to feed to an RL policy,
   so it is encoded by a **pre-trained VAE** into a compact 64-D latent vector. The policy
   never sees raw pixels — it sees a learned, low-dimensional summary of "what is in front of
   me and how far away." This decouples representation learning (done once, offline) from
   policy learning (done with RL), which is far more sample-efficient than learning to see and
   act simultaneously.

The reward combines an **exponential proximity reward** (large near the goal), a **progress
reward** (positive whenever the distance to the target shrinks this step), and a large
**collision penalty**. The optimal behaviour is to head straight for the goal while steering
around anything the depth latent flags as close.

As in the hover task, the policy outputs **attitude commands** (roll, pitch, yaw-rate,
thrust) tracked by the Lee attitude controller.

---

## Environment

| Property | Value |
|----------|-------|
| Env | `simple_obstacle_env` (sparse, fixed obstacles) |
| Obstacles | Fixed at 10 (curriculum disabled) |
| Target | Far end of room, ratio `[0.80–0.95, 0.20–0.80, 0.20–0.80]` of bounds |
| Success | Reach within 1.0 m of target before timeout, no crash |
| Episode length | 300 steps |
| Collision | Episode terminates (−100 penalty) |
| Controller | `lee_attitude_control` |
| Robot | `custom_quadrotor_with_camera` (depth camera) |
| Rendering | Warp ray-casting (`use_warp = True`) |

---

## Observations (77-D)

13 state dims + 64 VAE depth-latent dims.

| Index | Observation | Notes |
|-------|-------------|-------|
| 0–2 | Unit vector to target (vehicle frame) | |
| 3 | Distance to target | ÷ 20.0 |
| 4 | Roll | ÷ π |
| 5 | Pitch | ÷ π |
| 6 | Reserved | 0 |
| 7–9 | Body linear velocity | ÷ 5.0 |
| 10–12 | Body angular velocity | ÷ 5.0 |
| 13–76 | VAE latent encoding of the depth image | 64-D |

## Actions (4-D)

Network outputs in `[-1, 1]`, transformed to attitude commands:

| Index | Action | Range |
|-------|--------|-------|
| 0 | Roll command | ±π/6 (±30°) |
| 1 | Pitch command | ±π/6 (±30°) |
| 2 | Yaw-rate command | ±π/3 (±60°/s) |
| 3 | Thrust command | 0 – 15 m/s² |

---

## Reward

| Component | Value | Formula |
|-----------|-------|---------|
| Position (proximity) | mag 5.0, exp 0.5 | `5.0 · exp(−0.5 · dist)` |
| Progress | 5.0 | `5.0 · (prev_dist − dist)` |
| Collision | −100 (overrides) | applied on crash |

`reward = 5·exp(−0.5·dist) + 5·(prev_dist − dist)`, replaced by `−100` on a crash.

---

## Depth VAE

Uses the pre-trained Aerial Gym VAE weights
(`.../vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth`):

| Setting | Value |
|---------|-------|
| Latent dims | 64 |
| Input resolution | 270 × 480 |
| Interpolation | nearest |
| Output | sampled latent |

The depth image is squeezed to `(num_envs, H, W)` and encoded each step; the resulting
latents fill observation dims 13–76.

---

## Network & Training

| Parameter | Value |
|-----------|-------|
| Architecture | MLP `[256, 128, 64]` (shared actor-critic) |
| Algorithm | PPO (`a2c_continuous`, `continuous_a2c_logstd`) |
| Num environments | 1024 |
| Horizon length | 32 |
| Minibatch size | 2048 |
| Learning rate | 1e-4 |
| Max epochs | 500 |

Config: [`training/ppo_simple_obstacle.yaml`](training/ppo_simple_obstacle.yaml).

---

## Running

### Train

```bash
cd /workspaces/aerial_gym_docker
python simple_obstacle_avoidance/training/runner.py \
    --file=simple_obstacle_avoidance/training/ppo_simple_obstacle.yaml \
    --train \
    --num_envs=1024 \
    --headless=True
```

Add `--track --wandb-project-name=aerial_gym` for Weights & Biases logging.

### Inference / Play

```bash
python simple_obstacle_avoidance/training/runner.py \
    --file=simple_obstacle_avoidance/training/ppo_simple_obstacle.yaml \
    --play \
    --checkpoint=runs/simple_obstacle_avoidance/nn/simple_obstacle_avoidance.pth \
    --num_envs=64 \
    --headless=False
```

### Visualize the environment

```bash
python simple_obstacle_avoidance/tests/view_env.py
```

### Monitor

```bash
tensorboard --logdir runs/
```

---

## File Structure

```
simple_obstacle_avoidance/
├── README.md                              # This file
├── config/
│   ├── task_config.py                     # Obs/action dims, rewards, VAE config, action transform
│   ├── env_config.py                      # Sparse obstacle environment
│   └── robot_config.py                    # Quadrotor + depth camera, spawn ranges
├── task/
│   └── simple_obstacle_avoidance_task.py  # Task logic, reward, VAE encoding, success tracking
├── tests/
│   └── view_env.py                        # Environment visualization helper
└── training/
    ├── runner.py                          # rl_games registration + PPO entry point
    └── ppo_simple_obstacle.yaml           # PPO + network hyperparameters
```

> For the full curriculum version with dense clutter and spiking/recurrent policies, see
> [`../navigation_with_obstacles/`](../navigation_with_obstacles/).
