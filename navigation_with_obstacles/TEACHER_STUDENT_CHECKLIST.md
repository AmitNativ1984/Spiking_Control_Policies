# Teacher–Student (ANN → SNN) Implementation Checklist

Goal: use a trained **ANN actor** as a teacher to warm-start the **PopSAN SNN** student,
then fine-tune with PPO. Ordered so each step unblocks the next, grouped into phases
so you can stop at a working milestone.

A good first milestone is **Phase 0 → 2** (teacher loads correctly + bounds set + no
silent neurons). That's independently verifiable and de-risks everything after.

---

## Phase 0 — Teacher: have a trained ANN actor
- [x] Train (or locate) an ANN actor checkpoint using `ppo_navigation_ann_*.yaml`. Note its path.
- [x] Confirm it was trained with the **same** obs layout, action dim, and **same VAE** you'll use for the SNN. If the VAE differs, the latent dims won't transfer — re-train or re-collect.
- [x] Open the checkpoint and confirm it contains both the network weights **and** `running_mean_std` (rl-games saves these together). Note the exact keys.

## Phase 1 — Teacher loader (frozen, correct normalization)
- [x] Write a helper that builds the **full rl-games teacher model wrapper** (the `ModelA2CContinuousLogStd.Network`, which includes `running_mean_std`), not just the bare `ANNMLPActorCriticNetwork`.
- [x] Load the checkpoint into it, call `.eval()`, and `requires_grad_(False)` on all params.
- [x] Feed it **raw** obs; let it normalize internally. Verify by spot-checking: same obs → teacher in your loader produces the same `mu` as the teacher does at play time.
- [x] Sanity test: feed a batch of real obs, confirm `mu` is finite and in a sane action range.

## Phase 2 — Observation bounds for the encoder (no silent neurons)
- [ ] Add an optional `--teacher_checkpoint` to `collect_obs_stats.py` so the rollout is **driven by the ANN's `mu`** instead of random actions (line 94). Keep random as the default fallback.
- [ ] Run it, collect per-dim `p01/p99` (or `p05/p95`).
- [ ] Have it emit a ready-to-paste `observation_bounds` list from the chosen percentile.
- [ ] Set `task_config.observation_bounds` from these. **Handle VAE latents separately** — measure them from the current VAE, not the teacher, if the VAE isn't identical.
- [ ] Verify the encoder builds: `len(observation_bounds) == input_dim` (asserted in `networks/snn/pop_spiking_actor.py`).
- [ ] Quick check for silent neurons: feed a batch through `pop_encoder`, confirm every column has nonzero spikes across the batch.

## Phase 3 — BC warm-up script (`warmup_snn_from_ann.py`)
- [ ] Build the SNN actor (`PopulationSpikingActorNetwork`) and the env (reuse the runner's registration block).
- [ ] Decide **normalization ownership** and keep it identical in warm-up *and* later PPO (encoder-only, or rl-games normalizer copied — pick one).
- [ ] Force `task_config.vae_gate` to match the PPO hand-off phase (likely `1.0`).
- [ ] Loop: each step compute `teacher_mu = teacher(raw_obs).detach()`, `student_mu = snn_actor(raw_obs)`, `loss = MSE(student_mu, teacher_mu)`, backprop, Adam (lr≈1e-3).
- [ ] Use **DAgger-style** env stepping: action = teacher with prob β, else student; anneal β 1→0.
- [ ] Match the **action squashing** (tanh/clamp) used by PPO when stepping the env.
- [ ] Keep `num_steps` identical to the PPO config (5).
- [ ] Log: MSE, and periodically an **SNN-solo rollout return** (β=0) as the real stopping metric.

## Phase 4 — Save a checkpoint PPO can load
- [ ] Save the warmed-up SNN in the **rl-games checkpoint format** (same dict structure PPO's `--checkpoint` expects: `model` state dict, optionally `running_mean_std`).
- [ ] If you copied the ANN normalizer, include it so PPO starts with correct obs stats.
- [ ] Round-trip test: load it with `--checkpoint` into the existing SNN runner and confirm no key-mismatch errors, network loads, and a `--play` rollout behaves like the warmed-up policy.

## Phase 4.5 — Critic handling (init from teacher, then keep training)
> A critic is **policy-specific**: `V(s)` estimates returns under a *specific* policy.
> The teacher's critic describes the *ANN's* policy, so it cannot stay frozen for the
> whole SNN PPO run — as the SNN drifts from the ANN, a frozen critic becomes wrong
> and biases PPO's advantages.
- [ ] **Initialize** the SNN's critic from the ANN critic's weights (both use the same `ANNMLPCritic` class — `networks/ann/critic.py` — so weights copy 1:1, no shape issues).
- [ ] **Keep training the critic** during SNN PPO so it tracks the SNN's evolving policy. (Do *not* freeze it for the RL phase.)
- [ ] During the **BC warm-up** phase the critic is unused (no advantages computed) — you can ignore it there; only wire it in at PPO start.
- [ ] Carry over the matching **obs-normalization stats** with the critic weights, and mind `normalize_value: True` — a warm-started critic must see the same input/value scaling it was trained on, or the warm-start is wasted.

## Phase 5 — PPO fine-tune (warm-started)
- [ ] Start PPO from the warm-up checkpoint via `--checkpoint` on `popsan_navigation_*.yaml`.
- [ ] (Optional but recommended) Add a **short annealed distillation tail**: a custom A2C agent that adds `distill_coef · MSE(student_mu, teacher_mu)` for the first ~50–100 epochs so PPO doesn't wash out the warm-start before the critic catches up. Register it in `training/runner.py`.
- [ ] Confirm consistency: same obs normalization, same `vae_gate` schedule, same `num_steps` as warm-up.

---

## Cross-cutting checks (verify at each phase)
- [ ] **Coordinate frames** of every obs dim match between teacher and student (attitude, velocity, angular velocity, accel, IMU).
- [ ] Teacher and student see the **same raw obs** at the same step (no off-by-one between `env.step` and the obs you feed each network).
- [ ] No double-normalization surprise: currently `normalize_input: True` *and* encoder clamping — decide who owns scaling before warm-up.
- [ ] Determinism: use teacher `mu` (not sampled actions) as the BC target.
