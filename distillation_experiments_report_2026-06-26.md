# ANN→SNN Distillation: Initial Post-BC PPO Experiments

**Date:** 2026-06-26
**Method:** Teacher–Student — distill a frozen ANN PPO teacher (rl_games `ModelA2CContinuousLogStd`, MLP) into a population-coded LIF SNN student (PopSAN), navigation-with-obstacles, curriculum pinned at level 25.
**Baseline to beat:** BC warmup checkpoint = **~70% success** (deterministic eval) at curriculum 25.
**Goal of these runs:** get PPO fine-tuning to *hold and improve* on the BC policy instead of degrading it.

All experiment names align with the W&B run names (`navigation_with_obstacles_<timestamp>`, W&B id in parentheses).

---

## Summary table

| W&B run | id | Key change | Epochs | Success (peak / final) | mean KL | LR floored | exp_var (max) | Outcome |
|---|---|---|---|---|---|---|---|---|
| `..._14-59-23` | pf90mdeq | first attempt | 0 | — | — | — | — | **Crashed at startup** |
| `..._15-03-53` | e2eji3ms | fixed crash; baseline distill | 16 | 0.41 / 0.41 | 0.037 | — | 0.39 | LR-brake stall (short) |
| `..._15-41-39` | m0csb4nn | (same config, longer) | 221 | **0.53** / 0.52 | 0.037 | ~100% | 0.44 | **Stuck at ~0.5, LR floored** |
| `..._19-20-24` | 4ju7dqoj | `kl_threshold` 0.025→0.08 | 35 | 0.17 / 0.13 | 0.122 | 0% | 0.37 | **LR freed → policy ran the wrong way** |
| `..._19-56-59` | (latest) | + `kd_actor_coeff` 0.1→1.0 | 759 | **0.59** / ~0.00 | 0.034 | 2% | **0.57** | **Peaked 0.59 @ ep64, then collapsed to 0** |

---

## Experiment-by-experiment

### 1. `navigation_with_obstacles_2026-06-25_14-59-23` (pf90mdeq) — crashed at startup
- **What:** first distillation attempt after BC.
- **Result:** died before epoch 1.
- **Cause:** config mismatch — `num_actors: 128` vs env `num_envs: 64`. rl_games allocated a 128-env experience buffer but the sim returned 64-env observations → `RuntimeError: expanded size (128) must match existing size (64)`.
- **Fix carried forward:** set `num_actors = 64` to match `num_envs`. Also fixed two regime bugs found here: the student was starting at **curriculum 0 with the VAE gated off**; we pinned curriculum to the teacher's level (25) and **removed the VAE warm-up gate so the VAE is always on**. Sigma was also being initialized to 1.0 (zero-init `log_std`); we now **seed the student's `log_std` from the teacher's converged value**.

### 2 & 3. `..._15-03-53` (e2eji3ms) and `..._15-41-39` (m0csb4nn) — the LR-brake stall
- **What:** the fixed baseline distillation. 15-03-53 is the short version (16 epochs); 15-41-39 is the same config run to 221 epochs.
- **Result:** success **plateaus at ~0.5** and never advances; curriculum never moves off 25.
- **Diagnosed cause — the central finding:** the KD (teacher-copy) term and PPO's adaptive learning-rate brake fight each other.
  - PPO uses an **adaptive LR** keyed on policy-change (KL): if KL > `kl_threshold` (0.025), it cuts the LR.
  - The KD pull *adds* policy motion, so measured **KL sits at ~0.037 — above 0.025 in 100% of epochs**.
  - → the scheduler **floors the LR at `min_lr` (down to 2.3e-6)** ~100% of the time → the policy can barely move → stuck at 0.5.
- **Also noted:** `exp_var` only reaches ~0.44 — the critic explains <half the returns (weak-critic signal, returns later).
- **Conclusion:** the bottleneck is not the KD schedule but the **adaptive-LR brake misreading the harmless teacher-pull as recklessness**. *(Side correction along the way: the KD anneal was confirmed to fade over ~100 epochs / ~810k frames as configured — an earlier "drops at 2k steps" reading was a misread of the W&B axis.)*

### 4. `..._19-20-24` (4ju7dqoj) — raised the KL threshold; LR freed but policy diverged
- **Change:** `kl_threshold` 0.025 → **0.08** (above the observed ~0.037 KL band), to stop the brake tripping.
- **Result on the brake (worked):** LR came **off the floor** (0% floored, mean 6.8e-4). ✅
- **Result on the policy (worse):** once free to move, the policy **moved violently the wrong way** — KL shot to **0.12 (peak 0.17)**, success **collapsed to 0.12**, crash 0.66.
- **Conclusion — key insight:** the floored LR had been *accidentally protecting* us. Unfreezing it exposed the real instability: with `exp_var` only ~0.37, the **critic gives bad advantage estimates**, and a free PPO acts on them hard. The prior at `kd_actor_coeff = 0.1` was too weak to hold the student on the teacher's safe manifold.

### 5. `..._19-56-59` (latest) — strengthened the prior; best peak, then collapse when the prior anneals out
- **Changes:** `kd_actor_coeff` 0.1 → **1.0** (10× stronger teacher-pull), `kl_threshold` 0.08 retained.
- **Early result (best so far):** success climbed to a **peak of 0.59 around epoch 64** — the highest of any run. The stronger prior clearly helped while it was active.
- **Late result — collapse, and its cause is now pinned:** after the peak, success fell steadily to **~0 by ~ep 150** and stayed there through 759 epochs. The fine-grained trajectory shows the collapse tracks the **KD anneal** almost exactly:

  | epoch | success | kd_scale | actor_kd | entropy |
  |---|---|---|---|---|
  | 64 (peak) | 0.594 | 0.36 | 1.48 | 2.81 |
  | 90 | 0.565 | 0.10 | 1.60 | 3.06 |
  | **100** | 0.540 | **0.00** | **0.00** | 3.20 |
  | 130 | 0.270 | 0.00 | 0.00 | 4.02 |
  | 150 | 0.100 | 0.00 | 0.00 | 4.42 |
  | 200 | **0.000** | 0.00 | 0.00 | **5.40** |

  The teacher prior (`kd_scale`) fades to 0 exactly at **epoch 100** (`kd_anneal_epochs=100`). Decline starts as it shrinks and becomes freefall once it hits 0. Meanwhile **entropy climbs monotonically (2.8 → 5.4)** — the policy's action distribution keeps widening: with `entropy_coef=0.01` pushing entropy up and no prior left to constrain sigma, the policy unravels into noise → crashes → success 0.
- **Conclusion (corrected):** the bottleneck is **not the critic**. `exp_var` in the 0.4–0.6 range is *normal and sufficient* — the teacher ANN itself succeeds at ~76% with `exp_var ≈ 0.38` (see below). The real failure is that **the student cannot stand on its own once the teacher prior is removed**: the KD prior was holding the whole policy up, and when it anneals to 0 the policy collapses (entropy runaway, success → 0). The `exp_var → 0` seen at the end is a *symptom* of that collapse, not its cause.

#### Teacher exp_var reference (why ~0.6 is fine)
| | exp_var (final / peak) | success (final / peak) |
|---|---|---|
| Teacher ANN (`nav_vae_ann_cluster_2026-06-22`, the distilled one) | 0.38 / 0.88 | 0.76 / 0.99 |
| Older teacher run (`..._29-18`) | 0.54 / 0.75 | 0.79 / — |
| Best student (19-56-59) | 0.57 peak | 0.59 peak |

A critic explaining ~40–55% of return variance was more than enough to train the strong teacher. Navigation-with-obstacles is inherently high-variance (random layouts, sparse arrival reward, crashes), so the unexplained variance is genuine environment randomness — not critic error. **Chasing `exp_var → 0.9` is the wrong target.**

---

## Overall progression & conclusions

1. **Fix the plumbing** (env-count crash, curriculum-0/VAE-gate-off, σ=1.0) → got distillation to *run*.
2. **Found the real blocker:** KD vs adaptive-LR — the teacher-pull keeps KL above `kl_threshold`, so the LR brake floors the learning rate and the student freezes at **~0.5**.
3. **Raising `kl_threshold` (0.08)** freed the LR but **revealed an unstable policy** (success → 0.12): the floored LR had been masking a weak critic.
4. **Strengthening the prior (`kd_actor_coeff`=1.0)** gave the **best peak yet (0.59)** — but the policy **collapsed once the KD prior annealed to 0 at epoch 100** (entropy ran away 2.8→5.4, success → 0).

**Where we are vs BC:** best PPO peak ≈ **0.59** vs BC ≈ **0.70** — still short, **and not yet stable** (the best run collapses after the prior is removed).

**Leading remaining cause (corrected):** **the student cannot stand on its own without the teacher prior.** The collapse tracks the KD anneal exactly (peak ep64 while prior active → freefall after `kd_scale`=0 at ep100), driven by an **entropy runaway** (`entropy_coef=0.01` widening the policy with no prior left to hold sigma). The critic is **not** the problem — `exp_var ≈ 0.4–0.6` is normal and matches the teacher (which succeeds at 76% with exp_var 0.38).

**Recommended next experiments (in priority order):**
1. **Don't anneal the prior to zero** — floor `kd_actor_coeff` at a small permanent value (e.g. 0.05–0.1) or greatly lengthen `kd_anneal_epochs`, so the teacher keeps gently regularizing the policy indefinitely. Directly targets the observed collapse mechanism.
2. **Cut the entropy pressure** — lower or zero `entropy_coef` (currently 0.01) during the tail; a distilled, near-optimal policy does not need entropy pushing it toward randomness, and that push is what inflates sigma after the prior fades.
3. **Gate the anneal on success** — only fade the prior once the student is self-sufficient (e.g. success above a threshold), rather than on a fixed 100-epoch timer.

(Secondary, unrelated to the collapse: with `kd_actor_coeff=1.0` the KD↔adaptive-LR fight returns — KL ~0.11 > 0.08 — so consider nudging `kl_threshold` to ~0.13 to keep the LR off the floor.)

---

## Plots to attach (the four that carry the conclusions)

For each experiment, the W&B/TensorBoard panels that tell the story:

- **`success_rate`** — the headline outcome; shows the ~0.5 plateau, the 0.12 collapse, and the 0.59-peak-then-crash.
- **`info/kl`** — shows the KD-vs-threshold fight (KL above `kl_threshold` in 100% of the stalled runs).
- **`info/last_lr`** — shows the LR floored at `min_lr` (stall) vs freed (after threshold raise).
- **`diagnostics/exp_var`** — the critic signal; never sustains, collapses to 0 in the latest run (the unresolved root cause).

Compare across runs by overlaying the same metric with the legend set to the W&B run names above.
