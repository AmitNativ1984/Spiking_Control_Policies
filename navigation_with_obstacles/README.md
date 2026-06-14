# Navigation with Obstacles

The flagship task: learn **vision-based point-to-point navigation through dense, cluttered
environments** via a difficulty **curriculum**, with a choice of policy architectures —
including a population-coded **spiking neural network (PopSAN)** for neuromorphic control,
plus MLP and GRU baselines.

The quadrotor must fly from one side of a randomized room to a target waypoint on the far
side, dodging obstacles it perceives only through a forward depth camera encoded by a custom
32-D **Depth VAE**.

---

## Theory

This task combines every challenge in the repo and adds three new ideas.

### 1. Curriculum learning

Learning to navigate dense clutter from scratch is nearly impossible — the agent crashes
before it ever experiences reaching the goal. Instead, difficulty ramps up automatically:

- **Levels 0–5:** large panels only.
- **Levels 6–25:** cumulative panels **+** small objects (increasingly dense).

The task tracks rolling success / crash / timeout / out-of-bounds rates over a window of
rollouts and adjusts the obstacle count: success rate **> 0.7 → level up**, **< 0.6 → level
down** (one level at a time). The *arrival bonus* also scales with the level (harder rooms
give a bigger payoff), à la MAVRL, so the agent stays motivated as the task gets harder.

### 2. Depth VAE with two-phase warm-up gating

Perception uses the custom [Depth VAE](../vae_depth/) (Deep Collision Encoding): the encoder
takes raw depth and produces a 32-D latent whose decoder target is *collision-dilated* depth,
so the latent implicitly bakes in the drone's safety margin.

Feeding 32 vision latents to a freshly-initialized policy in an empty room is just noise that
slows learning. So the task runs a **VAE warm-up state machine** at level 0:

- **Phase A (gated, `vae_gate = 0`):** train pure navigation with the VAE latents zeroed out,
  until level-0 success ≥ `vae_gate_until_success` (0.90).
- **Phase B (consolidation):** enable the VAE, but stay pinned at level 0 until an ungated
  window also reaches `vae_consolidate_success` (0.90).
- **Phase C:** normal curriculum advancement with vision fully on.

At inference the gate is always forced to `1.0` (full vision), regardless of which phase a
checkpoint was saved in.

### 3. Population-coded spiking policy (PopSAN)

The headline policy is a **PopSAN** (Population-coded Spiking Actor Network): each scalar
observation is encoded by a population of neurons with Gaussian receptive fields into spike
trains, processed by Leaky Integrate-and-Fire (LIF) layers over several timesteps, then
decoded back to continuous actions. This is biologically inspired and maps efficiently onto
neuromorphic hardware. MLP and GRU actor-critics are provided as conventional baselines (the
GRU adds memory for partial observability).

### Reward (dense shaping + sparse terminals)

The reward has four **mutually exclusive terminal/step** outcomes (priority order):

1. **Out of bounds** → `exceed_penalty` (−10), terminates.
2. **Arrived** (within `d_min` = 0.4 m) → `arrive_bonus` (10→15, scales with curriculum), terminates as success.
3. **Collision** → `collision_penalty` (−10), terminates.
4. **Normal step** → dense progress shaping:

| Term | Coef | Meaning |
|------|------|---------|
| `r_bearing` | `lambda_b` = 0.1 | reward velocity aligned with direction to target (cosine) |
| `r_progress` | `lambda_p` = 0.5 | reward closing distance to target (meters/step) |
| `p_speed` | `lambda_v` = −0.1 | penalize horizontal speed above `v_max` (5 m/s) |
| `p_jerk` | `lambda_jerk` = −0.01 | penalize change in action (smoothness) |

As with the other tasks, the policy commands **attitude** (thrust, roll, pitch, yaw-rate)
tracked by `lee_attitude_control`. Note thrust here stays in `[-1, 1]` (0 = hover, controller
maps it to `[0, 2mg]`); roll/pitch scale to ±45°, yaw-rate to ±60°/s.

---

## Environment

| Property | Value |
|----------|-------|
| Env | `navigation_obstacle_env` (randomized bounds, panels + objects) |
| Curriculum | 0–25 (panels → panels + objects) |
| Target | Far wall, ratio `[0.95–1.00, 0.10–0.90, 0.10–0.90]` of bounds |
| Arrival threshold `d_min` | 0.4 m |
| Episode length | 800 steps |
| Out-of-bounds margin | 1.0 (exact bounds; set higher for lenient inference) |
| Controller | `lee_attitude_control` |
| Robot | `nav_quadrotor_with_camera` |
| Rendering | Warp ray-casting (`use_warp = True`) |

---

## Observations (17-D state + 32-D VAE = 49-D)

State dims `[0:17]` are stable whether or not vision is enabled; VAE latents are appended.

| Index | Observation |
|-------|-------------|
| 0–2 | Unit vector to target (vehicle frame) |
| 3 | Normalized distance to target, clamped `[0, 1]` |
| 4–6 | Vehicle linear velocity |
| 7–9 | Body angular velocity |
| 10–12 | Gravity vector in body frame (normalized) |
| 13–16 | Previous (transformed) action: thrust, roll, pitch, yaw-rate |
| 17–48 | DepthVAE latents (only when `vae_config.use_vae = True`) |

Set `vae_config.use_vae = False` to train a **state-only (17-D)** policy with no vision —
the observation layout, PopSAN encoder bounds, and the VAE encode step all key off this single
flag and stay in sync.

## Actions (4-D)

| Index | Action | Range |
|-------|--------|-------|
| 0 | Thrust | `[-1, 1]` (0 = hover, controller → `[0, 2mg]`) |
| 1 | Roll | ±π/4 (±45°) |
| 2 | Pitch | ±π/4 (±45°) |
| 3 | Yaw-rate | ±π/3 (±60°/s) |

---

## Policy architectures & configs

The runner registers three custom `rl_games` networks; pick one via the `--file` config.

| Policy | Config (local / cluster) | Network |
|--------|--------------------------|---------|
| **PopSAN (SNN)** | `popsan_navigation_local.yaml` / `popsan_navigation_cluster.yaml` | Population-coded spiking actor, 5 SNN timesteps/forward |
| **MLP** | `ppo_navigation_ann_local.yaml` / `ppo_navigation_ann_cluster.yaml` | `mlp_actor_critic` |
| **GRU (recurrent)** | `ppo_navigation_ann_gru_local.yaml` / `ppo_navigation_ann_gru_cluster.yaml` | `mlp_gru_actor_critic`, 1 GRU layer |
| **MLP + GRU (built-in rl_games)** | `ppo_navigation.yaml` / `ppo_navigation_cluster.yaml` | `actor_critic` `[512, 256, 64]` + GRU(64) |

All use PPO (`a2c_continuous`, `continuous_a2c_logstd`), `horizon_length = 64`,
`learning_rate = 1e-4` (local) / `3e-4` (cluster), `max_epochs = 1000`+.

> **Local vs cluster** configs differ mainly in scale (num_envs, minibatch size, epochs).
> See [`slurm/README_cluster.md`](slurm/README_cluster.md) for the full comparison.

---

## Running

All commands run from the repo root and use the package module form.

### Train (local — PopSAN/SNN)

```bash
cd /workspaces/aerial_gym_docker
python -m navigation_with_obstacles.training.runner \
    --file=navigation_with_obstacles/training/popsan_navigation_local.yaml \
    --train --headless=True
```

Swap `--file` for any config above to train the MLP or GRU policy. Add
`--track --wandb-project-name=aerial_gym` for Weights & Biases.

Useful overrides:

| Flag | Effect |
|------|--------|
| `--num_envs N` | override the YAML's env count (minibatch auto-clamped) |
| `--curriculum_level L` | pin obstacle density at level `L` (0–25) |
| `--exceed_margin M` | allow flying `M×` beyond bounds before termination |
| `--checkpoint PATH` | resume training from a checkpoint |

### Train (cluster / SLURM)

```bash
cd /workspaces/aerial_gym_docker
sbatch navigation_with_obstacles/slurm/train_navigation.sbatch        # PopSAN
sbatch navigation_with_obstacles/slurm/train_mlp_navigation.sbatch    # MLP
sbatch navigation_with_obstacles/slurm/train_mlp_gru_navigation.sbatch # GRU
sbatch navigation_with_obstacles/slurm/train_popsan_navigation.sbatch # PopSAN (explicit)

# Override defaults:
NUM_ENVS=4096 MAX_EPOCHS=1000 sbatch navigation_with_obstacles/slurm/train_navigation.sbatch
```

See [`slurm/README_cluster.md`](slurm/README_cluster.md) for image build, W&B setup, and monitoring.

### Inference / Play

Play always runs with the VAE fully enabled (`vae_gate = 1.0`):

```bash
python -m navigation_with_obstacles.training.runner \
    --file=navigation_with_obstacles/training/popsan_navigation_local.yaml \
    --play \
    --checkpoint=navigation_with_obstacles/runs/<run>/nn/<ckpt>.pth \
    --num_envs=16 --headless=False
```

To watch a trained policy in a harder-than-trained room, raise `--curriculum_level` and/or
`--exceed_margin 1.5`.

### Debug: visualize the PopSAN encoder

```bash
python -m navigation_with_obstacles.training.runner \
    --file=navigation_with_obstacles/training/popsan_navigation_local.yaml \
    --play --plot-encoding \
    --checkpoint=navigation_with_obstacles/runs/<run>/nn/<ckpt>.pth
```

Forces `num_envs=1`, records the population encoder during a single rollout, and saves
Gaussian receptive-field + spike-raster plots next to the checkpoint
(via [`tools/plot_encoder_trace.py`](tools/plot_encoder_trace.py)).

### Tune observation bounds (PopSAN)

The PopSAN population encoder needs per-dimension observation bounds (in rl_games-normalized
z-score space). Collect empirical statistics with:

```bash
python -m navigation_with_obstacles.tools.collect_obs_stats   # see slurm/collect_obs_stats.sbatch for cluster
```

### Monitor

```bash
tensorboard --logdir navigation_with_obstacles/runs/
```

Key metrics: `curriculum_level`, success/crash/timeout/exceed rates, and EMA reward
components (`reward/r_progress`, `r_heading`, `p_speed`, `p_jerk`).

### Tests

```bash
bash navigation_with_obstacles/tests/run_tests.sh
```

---

## File Structure

```
navigation_with_obstacles/
├── README.md                          # This file
├── config/
│   ├── task_config.py                 # Obs layout, rewards, curriculum, VAE + warm-up gating
│   ├── env_config.py                  # Randomized clutter environment
│   └── robot_config.py                # Quadrotor + depth camera
├── task/
│   └── navigation_task.py             # Task logic, reward components, curriculum state machine
├── networks/
│   ├── snn/                           # PopSAN: encoder, LIF actor, decoder, popsan builder
│   └── ann/                           # MLP and GRU actor-critic builders
├── training/
│   ├── runner.py                      # rl_games registration + PPO entry point
│   ├── popsan_navigation_*.yaml       # PopSAN (SNN) configs (local / cluster)
│   ├── ppo_navigation_ann_*.yaml      # MLP configs (local / cluster)
│   ├── ppo_navigation_ann_gru_*.yaml  # GRU configs (local / cluster)
│   └── ppo_navigation*.yaml           # built-in rl_games MLP+GRU configs
├── tools/
│   ├── collect_obs_stats.py           # observation statistics for PopSAN bounds
│   └── plot_encoder_trace.py          # encoder receptive-field / spike-raster plots
├── slurm/                             # SLURM sbatch scripts + cluster README
└── tests/                             # unit tests (direction/distance math, task helpers)
```

---

## Related

- [`../vae_depth/`](../vae_depth/) — the Depth VAE (DCE) used for the 32-D depth latents.
- [`../data_generation/`](../data_generation/) — generates the depth dataset for the VAE.
- [`../simple_obstacle_avoidance/`](../simple_obstacle_avoidance/) — the simplified, no-curriculum precursor to this task.
