# Simple Hover Task

The entry-point task: teach a quadrotor to **stabilize itself and hover in place** using
only inertial / proprioceptive sensing — no position feedback, no obstacles, no vision.

This is the simplest control problem in the repo and a good sanity check that the
simulator, the `rl_games` PPO integration, and the attitude controller all work end-to-end.

---

## Theory

A quadrotor is an inherently **unstable, underactuated** system: it has 6 degrees of
freedom but only 4 control inputs (collective thrust + 3 torques), and small attitude
errors integrate quickly into translational drift and crashes. Classical control solves
this with cascaded loops; here we instead learn an attitude-command policy with RL.

The twist in this task is that the agent is **not told where it is**. The observation
contains only body velocity, orientation, and raw IMU readings (linear acceleration +
angular velocity). The policy therefore has to learn to *null out motion* — drive linear
and angular velocities to zero and keep the body upright — purely from inertial cues. This
mirrors the real-world situation where reliable attitude/IMU estimates are always available
but global position is not.

The reward is a pure **regulation penalty**: it punishes any linear velocity, any angular
velocity, and jittery control, plus a large terminal penalty for crashing. The optimal
behaviour is to settle into a still, level hover.

**Action interface.** The network emits 4 values in `[-1, 1]`, mapped by the Lee attitude
controller to roll/pitch angle commands, a yaw-rate command, and a thrust command — so the
policy commands *desired attitude*, and a low-level controller tracks it.

---

## Environment

| Property | Value |
|----------|-------|
| Env spacing / bounds | 5 m × 5 m × 5 m (`env_spacing = 5.0`) |
| Ground plane | Yes |
| Obstacles | None |
| Collision / ground hit | Episode terminates (crash) |
| Episode length | 1000 steps |
| Controller | `lee_attitude_control` |
| Robot | `custom_quad_with_imu` (Bosch BMI088 IMU) |

---

## Observations (12-D)

All values are normalized to roughly `[-1, 1]`. **No position information is included.**

| Index | Observation | Normalization |
|-------|-------------|---------------|
| 0–2 | Body linear velocity (vx, vy, vz) | ÷ 5.0 m/s |
| 3–5 | Euler angles (roll, pitch, yaw) | ÷ π |
| 6–8 | IMU linear acceleration (ax, ay, az) | ÷ 20.0 m/s² |
| 9–11 | IMU angular velocity (ωx, ωy, ωz) | ÷ 10.0 rad/s |

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

All terms are penalties (negative is better):

| Component | Weight | Formula |
|-----------|--------|---------|
| Linear velocity | 0.1 | `‖v‖` |
| Angular velocity | 0.1 | `‖ω‖` |
| Action jitter | 0.05 | `‖aₜ − aₜ₋₁‖` |
| Collision | −100 (fixed) | applied on crash, overrides the rest |

`reward = −(0.1·‖v‖ + 0.1·‖ω‖ + 0.05·‖Δa‖)`, replaced by `−100` on a crash.

---

## Network & Training

Standard shared-trunk actor-critic MLP via `rl_games` PPO.

| Parameter | Value |
|-----------|-------|
| Architecture | MLP `[128, 64, 32]`, ELU |
| Algorithm | PPO (`a2c_continuous`, `continuous_a2c_logstd`) |
| Num environments | 8192 |
| Horizon length | 256 |
| Minibatch size | 2048 |
| Learning rate | 3e-4 |
| Gamma / GAE tau | 0.99 / 0.95 |
| Max epochs | 500 |

Config: [`training/ppo_hover.yaml`](training/ppo_hover.yaml).

---

## Running

### Train

```bash
cd /workspaces/aerial_gym_docker
python simple_hover/training/runner.py \
    --file=simple_hover/training/ppo_hover.yaml \
    --train \
    --num_envs=8192 \
    --headless=True
```

Add `--track --wandb-project-name=aerial_gym` to log to Weights & Biases.

### Inference / Play

```bash
python simple_hover/training/runner.py \
    --file=simple_hover/training/ppo_hover.yaml \
    --play \
    --checkpoint=runs/simple_hover/nn/simple_hover.pth \
    --num_envs=64 \
    --headless=False
```

### Monitor

```bash
tensorboard --logdir runs/
```

---

## File Structure

```
simple_hover/
├── README.md                      # This file
├── config/
│   ├── task_config.py             # Obs/action dims, reward weights, action transform
│   ├── env_config.py              # Bounds, ground plane, physics
│   └── robot_config.py            # Quadrotor + IMU, spawn ranges
├── task/
│   └── simple_hover_task.py       # Task logic, reward, observation assembly
└── training/
    ├── runner.py                  # rl_games registration + PPO entry point
    └── ppo_hover.yaml             # PPO + network hyperparameters
```

> A spiking-network variant of this same hover problem lives in
> [`../simple_hover_snn/`](../simple_hover_snn/).
