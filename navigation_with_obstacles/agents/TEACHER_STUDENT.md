# Teacher–Student Distillation: ANN → PopSAN (SNN)

How to warm-start and fine-tune a **population-coded spiking policy (PopSAN)** from a trained
**ANN actor–critic teacher**, and *why* each step is built the way it is.

The spiking student is hard to train with PPO from scratch: the population encoder + LIF
dynamics make the early policy landscape noisy, and a cold critic gives garbage advantages.
So we split learning into two stages:

1. **Behavior-cloning (BC) warm-up** — supervised imitation of the teacher's action mean,
   with DAgger-style on-policy data, until the SNN can fly on its own.
2. **PPO fine-tune with an annealed distillation tail** — standard PPO on the spiking
   student, plus a *decaying* teacher-distillation loss so PPO improves the policy without
   immediately washing out the warm-start before the critic catches up.

This is the [Kim et al. PopSAN](https://arxiv.org/abs/2010.09635) actor wrapped in
[rl_games](https://github.com/Denys88/rl_games) PPO, distilled from a conventional MLP
actor–critic. The companion build log is [`TEACHER_STUDENT_CHECKLIST.md`](../TEACHER_STUDENT_CHECKLIST.md).

> **Prerequisite:** read the task [`README.md`](../README.md) first — observation layout (17-D
> state + 32-D Depth-VAE = 49-D), action space (4-D attitude), curriculum, and the VAE
> warm-up gate are all defined there and are shared verbatim by teacher and student.

---

## 1. The cast of characters

| Role | Class / file | Trainable? | Sees |
|------|--------------|-----------|------|
| **Teacher** | `mlp_actor_critic` → `ANNMLPActorCriticNetwork` ([`networks/ann/actor_critic.py`](../networks/ann/actor_critic.py)) wrapped in rl_games `ModelA2CContinuousLogStd.Network` | **Frozen** (`eval`, `requires_grad_(False)`) | RAW obs → its **own** `running_mean_std` |
| **Student** | `PopSAN` → `POPSANNetwork` ([`networks/snn/popsan.py`](../networks/snn/popsan.py)) in the same rl_games wrapper | Trainable | RAW obs → its **own** `running_mean_std` |
| **Student actor** | `PopulationSpikingActorNetwork` ([`networks/snn/pop_spiking_actor.py`](../networks/snn/pop_spiking_actor.py)) | Trainable | normalized obs → population encoder |
| **Student critic** | `ANNMLPCritic` ([`networks/ann/critic.py`](../networks/ann/critic.py)) — **same class** as the teacher critic | Trainable (init from teacher) | normalized obs |

The teacher and student are the **same rl_games wrapper** (`continuous_a2c_logstd`), so
checkpoints round-trip and the critic copies 1:1. Crucially, **each owns its own
normalizer** — both are fed identical RAW observations and normalize internally, so there is
never a shared-normalizer coupling between a frozen teacher and a drifting student.

### Loaders / entry points

| File | Purpose |
|------|---------|
| [`networks/teacher_student/teacher_builder.py`](../networks/teacher_student/teacher_builder.py) | `build_teacher(...)` — builds the full rl_games teacher wrapper, loads the checkpoint (weights **+** `running_mean_std` together), `.eval()`, freezes all params. |
| [`tools/collect_obs_stats.py`](../tools/collect_obs_stats.py) | Collects per-dim observation bounds for the population encoder, driven by the teacher's actions. |
| [`agents/warmup_snn_from_ann.py`](warmup_snn_from_ann.py) | **Stage 1** — the BC + DAgger warm-up loop; saves a PPO-loadable checkpoint. |
| [`agents/a2c_teacher_agent.py`](a2c_teacher_agent.py) | **Stage 2** — `A2CTeacherAgent`: PPO + annealed KL distillation tail; critic init from teacher. |
| [`training/runner.py`](../training/runner.py) | Registers task/env/networks **and** the `a2c_teacher` algo, runs PPO. |

---

## 2. The policy parameterization (what the losses act on)

Both teacher and student output a **diagonal-Gaussian** policy over the 4-D action:

$$\pi(a\mid s) = \mathcal{N}\!\big(a;\ \mu(s),\ \operatorname{diag}(\sigma^2)\big),\qquad a\in\mathbb{R}^4 .$$

At environment step time the action is clamped to $[-1,1]$ by the task's
`action_transformation_function` (then mapped to thrust / roll / pitch / yaw-rate).

### Student forward (PopSAN), step by step

Let the normalized observation be $\tilde{s}\in\mathbb{R}^{49}$ (after the wrapper's
`running_mean_std`).

**(a) Population encoding** — [`networks/snn/encoder.py`](../networks/snn/encoder.py). Each obs
dim $d$ is encoded by $P$ = `pop_dim` (=10) neurons with Gaussian receptive fields. Clamp
$\tilde{s}_d$ to its empirical bounds $[\ell_d, h_d]$ (see §4), then for neuron $j$ with
learnable mean $\mu^{\text{enc}}_{d,j}$ and std $\sigma^{\text{enc}}_{d,j}$:

$$A_{d,j} = \exp\!\left(-\tfrac{1}{2}\,\frac{(\tilde{s}_d-\mu^{\text{enc}}_{d,j})^2}{(\sigma^{\text{enc}}_{d,j})^2}\right)\in[0,1].$$

This stimulus drives an Integrate-and-Fire (IF, no leak) neuron for $T$ = `num_steps` (=5)
timesteps, producing a binary spike train $x_{d,j}[t]\in\{0,1\}$. Means are initialized evenly
across $[\ell_d,h_d]$ and stds to $0.75\times$ the inter-mean spacing (overlapping coverage so
no input region is silent). Bounds matter: a dim whose range is wrong fires either always or
never → dead encoder channels.

**(b) Spiking actor** — three `Linear → snn.Synaptic` (CUBA-LIF) layers iterated over the
same $T$ steps; output spikes are accumulated and time-averaged:

$$\bar{z} = \frac{1}{T}\sum_{t=1}^{T} \text{spk}_3[t]\ \in\mathbb{R}^{4\times P}.$$

**(c) Spike decoding** — [`networks/snn/decoder.py`](../networks/snn/decoder.py). A grouped
`Conv1d` (one group per action dim, kernel $=P$) linearly reads out each action's population:

$$\mu(s) = \text{Conv1d}_{\text{grouped}}(\bar{z}),\qquad \log\sigma = \theta_{\log\sigma}\ \ (\text{a learned per-action vector}).$$

So the student's $\sigma$ is **state-independent** (a free parameter), while the teacher's may
be too — both are diagonal Gaussians, which is what makes the closed-form KL below cheap.

**VAE gate.** During the task's VAE warm-up Phase A the latent spike block is multiplied by
`task_config.vae_gate` (0 → off). The distillation pipeline always runs with the gate at
`1.0` (full vision) so teacher and student see the same information; the warm-up forces this
explicitly (`task.vae_phase = "C"`, `vae_gate = 1.0`).

---

## 3. The two-stage flow at a glance

```
                  ANN teacher checkpoint (ppo_navigation_ann_*.yaml)
                                   │
          ┌────────────────────────┴───────────────────────────┐
          │  collect_obs_stats.py  (teacher-driven rollout)     │  Phase 2
          │  → observation_bounds.json  (encoder receptive      │
          │     fields, in NORMALIZED obs space, no dead neurons)│
          └────────────────────────┬───────────────────────────┘
                                   │
   STAGE 1  agents/warmup_snn_from_ann.py                          Phase 3–4
   ─────────────────────────────────────────────
   • build frozen teacher + trainable PopSAN student
   • copy+freeze teacher running_mean_std into student
   • init student critic ← teacher critic (frozen during BC)
   • DAgger loop: minimize MSE(μ_S, μ_T); β anneals 1→0
   • save rl_games-format checkpoint  →  warmup_snn.pth
                                   │
   STAGE 2  runner.py  --train  --checkpoint warmup_snn.pth        Phase 4.5–5
   ─────────────────────────────────────────────  (algo: a2c_teacher)
   • resume student (actor+critic+norm stats) from checkpoint
   • PPO fine-tune; critic keeps training (tracks the SNN policy)
   • + annealed KL distillation tail: kd_scale(epoch)·KL(teacher‖student)
   • kd_scale: 1 → 0 linearly over kd_anneal_epochs, then 0
```

---

## 4. Phase 2 — observation bounds for the encoder

**Why:** the population encoder's Gaussian means must span the *actual* distribution of each
normalized obs dim, or neurons sit permanently silent/saturated. Random-action rollouts give
the wrong distribution; we instead drive the env with the **teacher's** deterministic action
(clamped to $[-1,1]$, exactly play-time behavior) and measure per-dim **p01/p99** in the
**teacher's normalized space** (raw → `running_mean_std` → clamp $[-5,5]$).

This runs **automatically** at the start of both stages (it must, because the encoder reads
the bounds at construction). Isaac Gym allows only one sim per process, so the collector runs
in a **subprocess** that writes `obs_stats/observation_bounds.json`; the parent loads it.
Cache is reused only if it matches both `obs_dim` and the current teacher checkpoint.

Manual run (rarely needed — the stages trigger it):

```bash
cd /workspaces/aerial_gym_docker
python -m navigation_with_obstacles.tools.collect_obs_stats \
    --config=navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml \
    --teacher_checkpoint=<path-to-ann-teacher>.pth \
    --num_envs=64 --num_steps=10000 --curriculum_level=25
```

Bounds are recomputed by default each warm-up (`--reuse_bounds` to opt out, or
`--recompute_bounds` to force during a PPO run).

---

## 5. Stage 1 — BC warm-up (`agents/warmup_snn_from_ann.py`)

### What it does

Behavior-clone the teacher's **deterministic action mean** into the spiking student, using
DAgger so the data distribution shifts toward the student's own state visitation as it
improves. A replay buffer decouples gradient steps from the slow simulator.

### Loss

The student trains only its actor mean against the teacher target (the teacher and the
student $\sigma$ are fixed during BC):

$$\mathcal{L}_{\text{BC}} = \big\| \mu_S(\tilde{s}) - \operatorname{clip}_{[-1,1]}\mu_T(s) \big\|_2^2 .$$

**Why MSE and not full KL here?** Both policies are diagonal Gaussians with *fixed* $\sigma$
during BC, so the policy KL reduces to

$$\mathrm{KL}\big(\pi_T\,\|\,\pi_S\big) = \sum_{d}\frac{(\mu_{T,d}-\mu_{S,d})^2}{2\sigma_{S,d}^2} + \text{const}(\sigma),$$

i.e. a $\sigma$-weighted MSE of the means plus a constant with no gradient. With $\sigma$
frozen, plain MSE is gradient-equivalent up to a per-dim learning-rate rescale — simpler and
numerically gentler for the warm-up. (The *full* KL with $\sigma$ gradients is used later, in
Stage 2; see §6.)

### DAgger data collection

Per env step, the action that *drives the simulator* mixes teacher and student per-env:

$$a_t = \begin{cases}\mu_T & \text{with prob. } \beta\\[2pt] \mu_S & \text{with prob. } 1-\beta\end{cases},\qquad \beta:\ 1 \to 0 .$$

$\beta$ starts at 1 (pure teacher rollouts = standard BC) and anneals toward 0 (student drives,
teacher only labels) once an **SNN-solo eval** (β=0 rollout) clears
`--solo_success_threshold`. The solo success rate — *not* the BC loss — is the real
convergence signal: low MSE doesn't guarantee the SNN can actually fly the compounding-error
trajectory. (Evals are opt-in via `--eval` because each does a full obstacle re-randomization,
a memory spike; without it β stays 1.0, still a valid warm-up.)

### Normalization & critic ownership during BC

- The teacher's `running_mean_std` is **copied into the student and frozen** for the whole
  warm-up, so teacher and student see the *identical* normalized space — the exact space the
  encoder bounds were measured in.
- The student critic is **initialized from the ANN critic and frozen** (BC computes no
  advantages, so the critic is unused; it's carried for the round-trip checkpoint).
- Only the **spiking actor** parameters get an optimizer (Adam, `--lr 1e-3`).

### Run it

```bash
cd /workspaces/aerial_gym_docker
python -m navigation_with_obstacles.agents.warmup_snn_from_ann \
    --file=navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml \
    --num_envs=1024 --headless=True \
    --max_steps=2000000 \
    --curriculum_level=25 \
    --eval --eval_every=50000 --eval_max_steps=800 \
    --solo_success_threshold=0.2 --anneal_steps=1000000 \
    --out=navigation_with_obstacles/runs/warmup_snn.pth
```

Local quick test: drop `--num_envs` to 256, `--max_steps` to ~200000, omit `--eval`.

**Output:** an rl_games-format checkpoint at `--out` (default
`runs/warmup_snn_<timestamp>/nn/warmup_snn.pth`) containing `model` (actor + critic +
`running_mean_std` + `value_mean_std`), plus an `optimizer` state and bookkeeping keys so a
`--train --checkpoint` resume (which calls `set_full_state_weights`) doesn't `KeyError`.

> **Cluster:** `sbatch navigation_with_obstacles/slurm/warmup_snn_from_ann.sbatch`
> (override `MAX_STEPS`, `CURRICULUM_LEVEL`, … via env vars). ⚠️ The sbatch currently invokes
> the old `training.warmup_snn_from_ann` module path — update it to
> `agents.warmup_snn_from_ann` after the file move.

### Sanity / round-trip checks

- Encoder builds: `len(observation_bounds) == input_dim` (asserted in the encoder and runner).
- No silent neurons: the collector feeds the normalized batch through the encoder and warns on
  any zero-spike column.
- Checkpoint round-trips: load `warmup_snn.pth` with `--play --checkpoint` and confirm no
  key-mismatch and a rollout resembling the warmed-up policy.

---

## 6. Stage 2 — PPO fine-tune with distillation tail

### Algorithm: `A2CTeacherAgent` (`agents/a2c_teacher_agent.py`)

A drop-in subclass of rl_games `A2CAgent`, selected by `algo.name: a2c_teacher` in the YAML and
registered in `runner.py`. It re-implements `calc_gradients` **identically** to the base PPO
step, then injects the distillation term into the *same* backward pass.

### The combined loss

Standard clipped PPO loss (unchanged from rl_games):

$$
\mathcal{L}_{\text{PPO}} =
\underbrace{\mathbb{E}\big[-\min(r_t\hat{A}_t,\ \operatorname{clip}(r_t,1\pm\epsilon)\hat{A}_t)\big]}_{\text{actor (clip, }\epsilon=0.2)}
\;+\; c_v\,\underbrace{\mathbb{E}\big[(V(s_t)-\hat{R}_t)^2\big]}_{\text{critic}}
\;-\; c_e\,\underbrace{\mathbb{E}[\mathcal{H}(\pi)]}_{\text{entropy}}
\;+\; c_b\,\mathcal{L}_{\text{bound}},
$$

where $r_t = \exp(\log\pi_\theta(a_t|s_t) - \log\pi_{\theta_{\text{old}}}(a_t|s_t))$,
$\hat{A}_t$ is GAE($\gamma=0.99,\lambda=0.95$), $c_v$=`critic_coef`(2), $c_e$=`entropy_coef`
(0.01), $c_b$=`bounds_loss_coef`(1e-4). The total loss adds the **distillation tail**:

$$
\boxed{\;\mathcal{L} = \mathcal{L}_{\text{PPO}} + s(k)\big[\,\alpha_a\,\mathcal{D}_{\text{actor}} + \alpha_c\,\mathcal{D}_{\text{critic}}\,\big]\;}
$$

with anneal scale $s(k)$ at epoch $k$, actor coeff $\alpha_a$=`kd_actor_coeff`, critic coeff
$\alpha_c$=`kd_critic_coeff`.

**Actor distillation (default `kd_actor_loss: kl`) — full diagonal-Gaussian KL.** With
$p=\pi_T$ (teacher, detached) and $q=\pi_S$ (student), summed over action dims:

$$
\mathcal{D}_{\text{actor}} = \mathrm{KL}\big(\pi_T \,\|\, \pi_S\big)
= \sum_{d}\left[\ \log\frac{\sigma_{S,d}}{\sigma_{T,d}}
+ \frac{\sigma_{T,d}^2 + (\mu_{T,d}-\mu_{S,d})^2}{2\,\sigma_{S,d}^2}
- \tfrac{1}{2}\ \right].
$$

Unlike the warm-up MSE, this gradient flows into **both** $\mu_S$ *and* $\sigma_S$ — the
student is pulled to match the teacher's *uncertainty*, not just its mean. We use the
**forward** KL $\mathrm{KL}(\text{teacher}\,\|\,\text{student})$ (mode-covering): the student
must put probability mass wherever the teacher does, which is the right inductive bias when the
teacher is the trusted reference. (`kd_actor_loss: mse` falls back to the §5 mean-only target.)

**Critic distillation (optional, default `kd_critic_coeff: 0.0`).** Compared in real
(denormalized) return space so it's invariant to the value normalizer:

$$\mathcal{D}_{\text{critic}} = \big\| \operatorname{denorm}V_S(s) - V_T(s)\big\|_2^2 .$$

Off by default — the critic learns better from true PPO returns than from the teacher's value
function (see §6.3).

**Anneal schedule** (`_current_distill_coef`):

$$
s(k) = \max\!\Big(0,\ 1 - \tfrac{k}{K}\Big),\qquad K = \texttt{kd\_anneal\_epochs}\ (\text{default }100),
$$

so the tail holds the warm-start for ~$K$ epochs while the critic calibrates, then linearly
vanishes and PPO optimizes the true objective unbiased. `K \le 0` ⇒ constant KD (no anneal).
When $s(k)\!\cdot\!(\alpha_a{+}\alpha_c)=0$ the teacher forward is skipped entirely (no cost).

### 6.3 Critic handling — *why init-then-train, never freeze*

A critic is **policy-specific**: $V^\pi(s)$ estimates returns *under a specific policy* $\pi$.

- **Initialize from the teacher** — the ANN and SNN critics are the *same* `ANNMLPCritic`
  class with matching `hidden_dims`, so weights copy 1:1. This is done in **two** places: the
  warm-up checkpoint carries it, and `A2CTeacherAgent._init_critic_from_teacher()` re-copies it
  in `__init__` (before any `--checkpoint` restore wins) so even a *cold* PPO start gets the
  teacher critic. The matching `running_mean_std` and `value_mean_std` are carried with it —
  under `normalize_value: True` a warm-started critic must see the same input/value scaling it
  was trained on, or the warm-start is wasted.
- **Keep training it** — the SNN policy drifts away from the ANN during PPO; a *frozen* critic
  would keep describing the ANN's policy and inject biased advantages. So the critic is never
  frozen in Stage 2; PPO's standard critic loss owns it. (It was frozen only in Stage 1, where
  no advantages are computed.)

### Run it

```bash
cd /workspaces/aerial_gym_docker
python -m navigation_with_obstacles.training.runner \
    --file=navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml \
    --train --headless=True \
    --checkpoint=navigation_with_obstacles/runs/warmup_snn.pth \
    --track --wandb-project-name=aerial_gym
```

Use `popsan_teacher_student_local.yaml` for a local run. The `distillation` block (teacher
checkpoint, coeffs, anneal) lives in the **same YAML** — single source of truth for both
stages and the bounds collector.

### Distillation config block (YAML)

```yaml
config:
  distillation:
    checkpoint: .../last_nav_vae_ann_cluster_ep_450_rew_15.5.pth  # the ANN teacher
    normalize_input:  True      # teacher owns its running_mean_std
    normalize_value:  True      # teacher owns its value_mean_std
    kd_actor_coeff:   0.1       # α_a  — weight on actor distillation
    kd_critic_coeff:  0.0       # α_c  — weight on critic distillation (0 = PPO-only critic)
    kd_actor_loss:    kl        # 'kl' (full Gaussian, μ+σ) | 'mse' (means only, == warm-up)
    kd_anneal_epochs: 100       # K — linear 1→0 anneal of the KD scale; 0 = constant
    network:                    # teacher architecture (must match its checkpoint)
      name: mlp_actor_critic
      separate: True
      actor:  { hidden_dims: [256, 256, 64], activation: elu }
      critic: { hidden_dims: [256, 256, 64], activation: elu }
```

> The student `network.critic.hidden_dims` **must equal** `distillation.network.critic.hidden_dims`
> for the 1:1 critic copy.

---

## 7. Consistency invariants (verify every run)

These must hold across both stages or the distillation silently degrades:

- **Same VAE / obs layout** as the teacher was trained with — else the 32 latent dims don't
  transfer (re-train or re-collect).
- **`vae_gate = 1.0`** in both stages (full vision); the warm-up forces phase `C`.
- **Same `num_steps`** (=5) for the SNN in warm-up and PPO (it's read from the same YAML).
- **Normalization ownership:** teacher and student each normalize RAW obs with their own
  `running_mean_std`; they are never cross-wired. Encoder bounds live in that normalized space.
- **Determinism of targets:** distillation uses the teacher's **`μ`** (mean), never a sampled
  action.
- **Coordinate frames** of every obs dim (attitude, velocity, angular velocity, accel/IMU)
  match between teacher and student — verify per source, never assume a shared frame.

---

## 8. Monitoring

```bash
tensorboard --logdir navigation_with_obstacles/runs/
```

| Scalar | Stage | Meaning |
|--------|-------|---------|
| `warmup/bc_loss` | 1 | MSE($\mu_S,\mu_T$) per gradient step |
| `warmup/beta` | 1 | DAgger β (teacher-action probability) |
| `warmup/solo_success_rate` | 1 | **the** convergence signal: β=0 SNN-solo arrivals |
| `warmup/solo_{crash,exceed,timeout}_rate` | 1 | failure breakdown of the solo eval |
| `distill/kd_scale` | 2 | anneal multiplier $s(k)$ |
| `distill/actor_kd` | 2 | $\mathcal{D}_{\text{actor}}$ (KL or MSE) |
| `distill/critic_kd` | 2 | $\mathcal{D}_{\text{critic}}$ (0 unless enabled) |
| `rewards/*`, `episode_lengths/*`, curriculum/success rates | 2 | standard PPO + task metrics |

A healthy Stage 2: episode return climbs above the warm-up baseline while `distill/actor_kd`
stays small early (warm-start held) and `kd_scale` ramps to 0 by epoch $K$, after which PPO
owns the policy outright.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Encoder build asserts `len(observation_bounds) != input_dim` | stale bounds cache (wrong obs dim) | `--recompute_bounds` (PPO) / drop `--reuse_bounds` (warm-up) |
| Collector warns "zero-spike columns" | bounds too tight / wrong teacher | recompute with the correct teacher checkpoint |
| `KeyError: 'optimizer'` on PPO resume | checkpoint lacks the optimizer/full-state keys | re-save via the warm-up script (it writes them) |
| PPO collapses immediately after resume | KD off or critic mis-scaled | confirm `algo.name: a2c_teacher`, `kd_actor_coeff>0`, `normalize_value` matches teacher |
| Critic state_dict mismatch | student vs teacher `critic.hidden_dims` differ | make them equal in the YAML |
| Warm-up MSE low but solo success ~0 | compounding error off the teacher's state distribution | enable `--eval` so β anneals on real solo performance; train longer |

---

## Related

- [`TEACHER_STUDENT_CHECKLIST.md`](../TEACHER_STUDENT_CHECKLIST.md) — the ordered build log / status.
- [`README.md`](../README.md) — the task itself (obs/action/reward/curriculum/VAE).
- [`../vae_depth/`](../../vae_depth/) — the Depth VAE producing the 32-D latents.
