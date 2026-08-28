# `rl_training` — PPO training for the F450 navigation task

rl_games integration for this repo: custom ANN and SNN network builders, a teacher-student
(ANN → SNN) distillation agent, and a single entry point that wires them to an aerial_gym task.

Everything is registered by import side effect, so the runner only parses arguments:

```python
import config                       # registers env, robot and task in aerial_gym's registries
import rl_training.rl_games.networks  # registers the custom rl_games network builders
```

---

## Contents

```
rl_training/rl_games/
├── runner.py                  entry point (train / play)
├── vecenv.py                  aerial_gym task -> rl_games IVecEnv adapter
├── wandb_utils.py             git provenance + checkpoint upload
├── agents/
│   └── a2c_teacher_agent.py   PPO + annealed ANN->SNN distillation tail  (algo name: a2c_teacher)
├── networks/
│   ├── ann/                   mlp_actor_critic, mlp_gru_actor_critic
│   ├── snn/                   popsan (population-coded spiking actor, snnTorch)
│   └── teacher_student/       frozen-teacher loader
├── tools/
│   ├── collect_obs_stats.py   measures PopSAN encoder bounds from a teacher rollout
│   └── plot_encoder_trace.py  debug plots of encoder activations
└── cfg/                       10 YAML configs (see the matrix below)
```

---

## Task

Two registered tasks:

| Task name              | Class                                                    | Config                                          |
| ---------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| `f450_navigation_task` | `task/attitude_navigation_task.py::NavigationWithObstaclesTask` | `config/task_config/f450_attitude_navigation_task_config.py` |
| `f450_hover_task`      | `task/hover_task.py::HoverTask`                          | `config/task_config/f450_hover_task_config.py`  |

**`f450_navigation_task`** — F450, attitude-rate control, flying through a procedurally
generated forest with an obstacle-density curriculum.

- **Action space:** 4 (raw `[-1, 1]`; the task maps it onto the controller's command range)
- **Observation space:** 49 = 17 state dims + 32 DepthVAE latent dims

Set `vae_config.use_vae = False` in the task config to train a state-only 17-D policy — the
observation layout, the encoder bounds and the VAE encode step all key off that one flag.

**`f450_hover_task`** — F450, attitude control, hold a sampled target point in an empty box.
No camera, no VAE, no curriculum.

- **Action space:** 4
- **Observation space:** 16 = position error (3) + gravity in body (3) + linvel (3) +
  angvel (3) + previous action (4)

`--task` must match a name in aerial_gym's `task_registry` (registered from
`config/task_config/__init__.py`), and must match the config's `env_name` — the runner
resolves the task config from that name to derive PopSAN's encoder bounds.

> **Note on naming:** the legacy tree `navigation_with_obstacles/` contains a *different* class
> that is also called `NavigationWithObstaclesTask`. Check the import path before assuming which
> one is running.

---

## Prerequisites

Run everything **inside the container**, from the repo root:

```bash
docker compose run aerial-gym        # headless (training)
cd /workspaces/aerial_gym_docker
```

The task loads a trained DepthVAE checkpoint. Its path is hard-coded in the task config
(`vae_config.model_file`) and must exist, or set `use_vae = False`:

```
vae_depth/runs/20260218_204641/checkpoints/epoch_150.pth
```

For W&B tracking (`--track`), export `WANDB_API_KEY` first.

---

## Quick start

**ANN (MLP PPO baseline):**

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/ppo_mlp_local.yaml \
    --train --headless=True
```

**SNN (PopSAN, trained from scratch with PPO):**

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_local.yaml \
    --train --headless=True
```

**SNN distilled from an ANN teacher** (see the full workflow below):

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_teacher_student_local.yaml \
    --train --headless=True
```

`--file` is required. Add `--track` to log to W&B, `--num_envs=N` to override the env count.

---

## Config matrix

Each navigation architecture ships a `_local` (single workstation / 8 GB VRAM) and a
`_cluster` (A100-class) variant, differing only in batch sizing and learning-rate schedule.
The two `*_hover_*` configs target `f450_hover_task`; everything else targets
`f450_navigation_task`.

| Config                              | Kind | `network.name`         | `algo.name`     | `num_actors` |
| ----------------------------------- | ---- | ---------------------- | --------------- | -----------: |
| `ppo_mlp_local.yaml`                | ANN  | `mlp_actor_critic`     | `a2c_continuous` |          128 |
| `ppo_mlp_cluster.yaml`              | ANN  | `mlp_actor_critic`     | `a2c_continuous` |         1024 |
| `ppo_gru_local.yaml`                | ANN  | `mlp_gru_actor_critic` | `a2c_continuous` |           64 |
| `ppo_gru_cluster.yaml`              | ANN  | `mlp_gru_actor_critic` | `a2c_continuous` |         1024 |
| `ppo_baseline_local.yaml`           | ANN  | `actor_critic` (stock rl_games, GRU) | `a2c_continuous` | 1024 |
| `ppo_baseline_cluster.yaml`         | ANN  | `actor_critic` (stock rl_games, GRU) | `a2c_continuous` | 4096 |
| `ppo_hover_local.yaml`              | ANN  | `mlp_actor_critic`     | `a2c_continuous` |          128 |
| `popsan_hover_local.yaml`           | SNN  | `popsan`               | `a2c_continuous` |          128 |
| `popsan_local.yaml`                 | SNN  | `popsan`               | `a2c_continuous` |          128 |
| `popsan_cluster.yaml`               | SNN  | `popsan`               | `a2c_continuous` |         1024 |
| `popsan_teacher_student_local.yaml` | SNN  | `popsan`               | `a2c_teacher`   |          128 |
| `popsan_teacher_student_cluster.yaml` | SNN | `popsan`              | `a2c_teacher`   |         1024 |

`num_actors` **is** the environment count — the runner derives `env_config.num_envs` from it (or
from `--num_envs`), so there is exactly one source of truth. `minibatch_size` is clamped to
`num_envs * horizon_length` automatically.

---

## ANN networks

Registered in `networks/__init__.py`, selected by `network.name` in the YAML.

### `mlp_actor_critic`
Separate actor and critic MLPs. Configured with `hidden_dims` / `activation`:

```yaml
network:
  name: mlp_actor_critic
  separate: True
  actor:  { hidden_dims: [256, 256, 64], activation: elu }
  critic: { hidden_dims: [256, 256, 64], activation: elu }
```

### `mlp_gru_actor_critic`
Same, with a GRU on the actor trunk for partial observability. Needs `seq_length` in the config
block:

```yaml
network:
  name: mlp_gru_actor_critic
  actor:
    hidden_dims: [256, 256, 64]
    gru: { hidden_size: 64, num_layers: 1 }
```

### `actor_critic`
Stock rl_games network (shared trunk + `rnn: gru`), used by the `ppo_baseline_*` configs as a
reference point against the custom builders.

---

## SNN network (`popsan`)

Population-coded spiking actor-critic built on snnTorch: observations are encoded into
populations of LIF neurons, run for `num_steps` simulation timesteps, and decoded into
continuous actions. The critic stays a plain ANN MLP.

```yaml
network:
  name: popsan
  device: cuda
  actor:
    encoder:
      pop_dim: 10             # neurons per observation dimension
      threshold: 0.95         # encoder activation threshold
    num_steps: 5              # SNN timesteps per forward pass
    hidden_dims: [256, 256]
    beta: 0.75                # LIF voltage decay
    alpha: 0.5                # LIF current decay
    threshold: 0.5            # LIF firing threshold
    spike_grad: fast_sigmoid  # surrogate gradient
    reset_mechanism: subtract # or "zero"
    sigma_init: 0.5           # initial action std (learnable)
  critic:
    hidden_dims: [256, 256, 64]
```

### Encoder observation bounds

The population encoder needs a per-dimension clamp window to place its Gaussian receptive
fields. You never write those numbers by hand (49 for navigation, 16 for hover) — they are
resolved at startup, in this priority order:

1. `network.actor.observation_bounds` explicitly set in the YAML — always wins.
2. **Measured** from a teacher rollout, for a `popsan` + `distillation` `--train` run
   (`resolve_encoder_bounds` → `tools/collect_obs_stats.py`). Collection runs in a subprocess
   (Isaac Gym allows one sim per process) and is cached to `runs/obs_stats/observation_bounds.json`.
3. **Derived** from the task's `observation_layout` via a per-type table
   (`bind_encoder_bounds` → `snn/encoder.py::DEFAULT_TYPE_BOUNDS`). This is the hover path —
   `popsan_hover_local.yaml` has no teacher, so its 16 windows come from here.

`bind_encoder_bounds` takes no task argument: it reads `config.env_name` and pulls the task
config out of `task_registry`, so the config's task and the encoder's bounds cannot disagree.
Adding an observation type to a layout without adding it to `DEFAULT_TYPE_BOUNDS` raises a
`KeyError` at startup by design, and `tests/test_networks.py` checks every registered task's
layout against the table.

Bounds live in rl_games-**normalized** space (z-scores), which is why measuring them requires
pushing raw observations through the teacher's frozen `running_mean_std`.

Force a re-measure with `--recompute_bounds`, and control the sample count with
`--bounds_steps=N` (default 10000).

You can also run the collector standalone:

```bash
python -m rl_training.rl_games.tools.collect_obs_stats \
    --num_steps=10000 --num_envs=64 \
    --teacher_checkpoint=runs/<teacher_run>/nn/<teacher>.pth \
    --config=rl_training/rl_games/cfg/popsan_teacher_student_local.yaml
```

Useful flags: `--bound_method={gaussian,percentile}` (default `gaussian`: `mu ± z·sigma`,
centered; `percentile` follows asymmetric tails), `--lower_pct` / `--upper_pct` (default 1/99),
`--curriculum_level` (default 25 — pin the obstacle density to the world the student is
deployed in).

---

## Teacher-student: ANN → SNN distillation

`algo: a2c_teacher` (`agents/a2c_teacher_agent.py`) is standard PPO plus a distillation tail:

```
loss = ppo_loss + kd_scale(epoch) * (kd_actor_coeff * actor_kd + kd_critic_coeff * critic_kd)
```

The teacher is a **frozen** rl_games `ModelA2CContinuousLogStd.Network` carrying its own
`running_mean_std`. It is fed the same **raw** observations as the student and normalizes
internally, so teacher and student never share a normalizer.

`kd_scale` is a monotonic, success-gated anneal: it starts at 1.0 and each curriculum check with
success ≥ `kd_release_success` subtracts `kd_anneal_step`, clamped at 0. A check below the bar
does nothing — the scale holds and never climbs back.

### Workflow

**1 — Train the ANN teacher.** The teacher's architecture must match the `distillation.network`
block in the student YAML, and its critic `hidden_dims` must match the student's `critic` block
(the ANN→SNN critic warm-start copies those weights).

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/ppo_mlp_local.yaml \
    --train --headless=True --experiment_name=f450_teacher_ann
```

**2 — Point the student config at the teacher checkpoint.** Edit
`popsan_teacher_student_local.yaml`:

```yaml
config:
  distillation:
    checkpoint: runs/f450_teacher_ann_<timestamp>/nn/<best>.pth
    normalize_input: True
    normalize_value: True
    kd_actor_coeff: 0.5        # weight on the actor distillation term
    kd_critic_coeff: 0.0       # 0 = critic trains purely on PPO returns
    kd_actor_loss: kl          # 'kl' (full diagonal-Gaussian KL) or 'mse' (means only)
    kd_release_success: 0.50   # success bar that releases one anneal step
    kd_anneal_step: 0.05       # per-check decrement (~20 sustained checks -> 0)
    network:                   # MUST match the teacher's architecture
      name: mlp_actor_critic
      separate: True
      actor:  { hidden_dims: [256, 256, 64], activation: elu }
      critic: { hidden_dims: [256, 256, 64], activation: elu }
```

**3 — Train the student.** Encoder bounds are measured from the teacher automatically on this
path (step 2 of the bounds resolution order above).

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_teacher_student_local.yaml \
    --train --headless=True --track
```

> ⚠️ **The checkpoint currently in both teacher-student YAMLs is stale.** It was trained on the
> legacy `navigation_with_obstacles` task. The observation vector is the same width (17 + 32 = 49)
> but the state semantics and reward are the old task's. Retrain a teacher on
> `f450_navigation_task` before trusting distillation from it. If the path is missing the runner
> warns and falls back to the task's default encoder bounds.

---

## Playback and evaluation

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_local.yaml \
    --play --checkpoint=runs/<run>/nn/<file>.pth \
    --headless=False --num_envs=1
```

Player settings (`deterministic`, `games_num`, `render`) come from the YAML's `player` block.
A `a2c_teacher` run plays back with the stock continuous PPO player — distillation only affects
training.

**Debug the SNN encoder** with `--plot-encoding` (forces `num_envs=1`): records the PopSAN
encoder's activations during the rollout and writes plots next to the checkpoint's run
directory.

```bash
python -m rl_training.rl_games.runner \
    --file=rl_training/rl_games/cfg/popsan_local.yaml \
    --play --checkpoint=runs/<run>/nn/<file>.pth --plot-encoding
```

---

## CLI reference

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--file` | — | **Required.** Path to a config YAML under `cfg/` |
| `--train` / `--play` | — | Train, or roll out a checkpoint |
| `--checkpoint` | — | Checkpoint to load (resume training, or play) |
| `--num_envs` | YAML `num_actors` | Env count; overrides the YAML |
| `--headless` | `False` | `True` for training, `False` to watch |
| `--use_warp` | `True` | Warp-based camera rendering |
| `--seed` | `0` | Random seed (`>0` overrides the YAML) |
| `--task` | `f450_navigation_task` | Task name in aerial_gym's registry |
| `--experiment_name` | `f450_navigation` | Run-name prefix |
| `--track` | off | Log to W&B (rank 0 only) |
| `--wandb-project-name` | `aerial_gym` | W&B project |
| `--wandb-entity` | — | W&B team |
| `--curriculum_level` | — | Pin obstacle density at a level (0–25) |
| `--exceed_margin` | — | Out-of-bounds termination margin (e.g. `1.5` = 50 % beyond) |
| `--recompute_bounds` | off | Force re-collection of PopSAN encoder bounds |
| `--bounds_steps` | `10000` | Steps to collect when measuring bounds |
| `--plot-encoding` | off | With `--play`: record + plot encoder activations |

---

## Outputs

rl_games writes to `runs/` at the repo root (`RUNS_DIR` in `rl_training/rl_games/__init__.py`):

```
runs/
├── <experiment_name>_<YYYY-MM-DD_HH-MM-SS>/
│   ├── nn/            checkpoints
│   └── summaries/     tensorboard
└── obs_stats/         observation statistics + observation_bounds.json cache
```

The timestamp is a full date+time — rl_games' own default (`_%d-%H-%M-%S`) omits year and month
and makes run folders ambiguous across months.

With `--track`, each W&B run records the git commit, branch and dirty flag in its config, so a
run can always be traced back to the exact code state that produced it.

```bash
tensorboard --logdir runs/
```

---

## Cluster / SLURM

`slurm/train_navigation.sbatch` runs **this** tree (`rl_training.rl_games.runner`); the legacy
scripts under `navigation_with_obstacles/slurm/` run that one. See `slurm/README_cluster.md`
for building the enroot image and setting `slurm/.env` (`WANDB_API_KEY`). Submit from the repo
root — `SLURM_SUBMIT_DIR` is what gets bind-mounted into the container:

```bash
sbatch slurm/train_navigation.sbatch                       # ANN MLP, ppo_mlp_cluster.yaml

CONFIG_FILE=rl_training/rl_games/cfg/popsan_teacher_student_cluster.yaml \
EXPERIMENT=f450_popsan_ts \
sbatch slurm/train_navigation.sbatch
```

Every knob is an environment variable: `CONFIG_FILE`, `TASK` (default `f450_navigation_task`),
`EXPERIMENT`, `WANDB_PROJECT`, `CHECKPOINT` (resume), `NUM_ENVS` (VRAM escape hatch — otherwise
the YAML's `num_actors` decides), `SEED`, `CURRICULUM_LEVEL`, `CONTAINER_IMAGE`, and
`EXTRA_ARGS` for any other runner flag verbatim.

The job refuses to start if the task config has `use_vae = True` and the checkpoint it names is
not on the cluster — that failure otherwise surfaces only after the simulator has finished
building.

---

## Gotchas

- **`import isaacgym` must precede `import torch`.** Any new entry point must import it first;
  `rl_training/rl_games/__init__.py` deliberately holds only path constants so importing it can
  never pull torch in early.
- **One network tree per process.** `networks/__init__.py` refuses to overwrite an existing
  rl_games network name and raises instead. If you see
  *"rl_games network name 'mlp_actor_critic' is already registered"*, the legacy
  `navigation_with_obstacles.training.runner` was imported in the same process. Import only one.
- **`num_actors` is the env count.** Do not also set `env_config.num_envs` in a YAML — they would
  drift, and the failure is a tensor-shape `RuntimeError` raised only after the simulator has
  finished building.
- **VRAM.** With cameras enabled, 8 GB fits about 128–256 envs. Use the `_local` configs, or
  lower `--num_envs`.
- **Teacher/student critic shapes must match.** The critic warm-start copies teacher critic
  weights into the student, so `distillation.network.critic.hidden_dims` and the student's
  `critic.hidden_dims` have to be identical.
- **Encoder bounds are cached.** A stale `runs/obs_stats/observation_bounds.json` will be reused
  across runs — pass `--recompute_bounds` after changing the task, the observation layout, or the
  teacher.
