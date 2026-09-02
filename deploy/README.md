# deploy/ — from a training checkpoint to something the vehicle runs

This is the torch side of a fence. The flight repo (`sail-uav-core`) contains no torch at
all: it builds the observation in numpy and runs an exported ONNX graph.

The fence is not tidiness. The two environments *cannot* run the same torch — this one is
Python 3.8 with an NVIDIA NGC build that was never released to any index, the flight image
is Python 3.12 whose floor is torch 2.2. A checkpoint interpreted on the vehicle would be
interpreted by a different torch than trained it. An exported graph is a frozen description
of the arithmetic and raises no such question.

Because version pinning cannot span that gap, parity is **measured** instead: the flight
suite replays recorded input/output pairs and checks the numbers still agree.

## Quick start

```bash
cd /workspaces/aerial_gym_docker

# spiking (PopSAN) policy
python -m deploy.publish --arch snn \
    --checkpoint runs/f450_hover_snn/<run>/nn/last_....pth \
    --policy-dir <sail-uav-core>/libs/control-policy-api/policies/hover_snn

# dense (MLP) policy
python -m deploy.publish \
    --checkpoint runs/f450_hover/<run>/nn/f450_hover.pth \
    --policy-dir <sail-uav-core>/libs/control-policy-api/policies/hover

# navigation policy — 49-D observation, two graphs
python -m deploy.publish --task navigation \
    --checkpoint runs/f450_nav_ann/<run>/nn/last_....pth \
    --policy-dir <sail-uav-core>/libs/control-policy-api/policies/navigation
```

Then, **in the flight container**:

```bash
pytest libs/control-policy-api
```

Running that test here proves almost nothing — it is the deployment environment's
arithmetic that is in question.

## The three artifacts

`publish` writes all three into one policy directory. They are build products: gitignored,
regenerated, never committed.

| file | written by | what it is |
|---|---|---|
| `hover.onnx` | `export_onnx` | the graph — normalized observation in, raw action out |
| `hover.npz` | `export_onnx` | frozen input-normalization statistics, plus provenance |
| `hover_golden.npz` | `record_golden` | 256 observation/action pairs the flight side must reproduce |

Normalization and the `[-1, 1]` action clamp stay *outside* the graph, in numpy on both
sides. Freezing them in would buy nothing and make the statistics harder to inspect.

### The navigation policy is four

Its observation is 17 state channels plus a 32-D DepthVAE latent, so there is a second
network in front of the actor — and it stays a **separate graph**:

| file | written by | what it is |
|---|---|---|
| `navigation.onnx` | `export_onnx --task navigation` | the actor — 49-D normalized observation in, raw action out |
| `navigation.npz` | `export_onnx --task navigation` | frozen input-normalization statistics, plus provenance |
| `depth_vae.onnx` | `export_vae_onnx` | the DepthVAE encoder — normalized 180×320 depth in, 32-D `mu` out |
| `navigation_golden.npz` | `record_golden --task navigation` | 32 (depth, state, target) → action samples, **including the depth images** |

Two graphs rather than one fused graph, because they run at different rates and cost wildly
different amounts: the encoder is a few hundred MFLOPs consumed once per camera frame, the
actor is ~35k MACs consumed once per decision. Fusing would weld the actor to the camera's
rate and put a GPU round trip in front of arithmetic that takes microseconds on a CPU core.

Two things are settled inside the encoder graph rather than left to the caller:

- **the `mu` slice.** The encoder emits `[B, 64]` — `mu` concatenated with `logvar`. Taking
  the wrong half gives a well-formed 32-vector and a policy steering on noise variance.
- **the input resolution.** 180×320 is structural, not a preference: the FC head is
  `Linear(fc_channel_dim * 6 * 10, 512)`, so the convolution stack lands correctly on that
  shape and no other.

The **preprocessing** (resize → clamp to [0.1, 7] m → `1 - d/7`) stays in numpy on the flight
side, and the golden carries the depth images so that reimplementation is checked rather than
assumed. It is the one part of the chain with two independent implementations.

The golden records 32 samples rather than 256 because each carries a 180×320 float32 image —
230 kB against 100 bytes for a hover sample.

## Why publish is one command

Exporting the graph and recording the golden are two independent readings of the same
checkpoint, and that independence is the point — the parity test means something only
because neither artifact was derived from the other.

But as two commands they can be run with *different* inputs, and that failure is invisible:
each file is internally consistent, nothing looks broken, and the parity test quietly starts
comparing two different policies while reporting the difference as numerical drift. This has
already happened once here — an updated export sat beside a stale golden and the suite stayed
green. Both files now carry the checkpoint's SHA-256 (and, for a spiking policy, the
config's), so the flight suite detects it. `publish` removes the opportunity instead.

The underlying scripts still work standalone if you need one of them alone.

## The one thing that can go silently wrong: `num_steps`

**Every PopSAN weight shape is independent of the number of spiking timesteps.** A
checkpoint loaded at the wrong `num_steps` loads with `strict=True`, warns about nothing,
and produces a policy that correlates ~0.99 with the real one while being wrong on every
actuator. Measured on `f450_hover_snn` ep_1000:

| exported `num_steps` | max abs deviation | correlation |
|---|---|---|
| 3 | 0.356 | 0.961 |
| 4 | 0.162 | 0.994 |
| 5 (correct) | — | 1.000 |
| 6 | 0.132 | 0.997 |
| 8 | 0.297 | 0.989 |

In a `[-1, 1]` action space that is up to 16% of full scale. It flies. It flies badly,
forever, and nothing reports it.

`num_steps` lives only in the run's `config.yaml` — never in the weights — so three things
defend it:

1. **`publish` derives the config from the checkpoint path.** A checkpoint always sits at
   `<run>/nn/<name>.pth`, so the right config is always `<run>/config.yaml`. Never point this
   at `rl_training/rl_games/cfg/*.yaml`: that file is alive and drifts independently of every
   checkpoint trained from it.
2. **`verify_config_against_weights` fingerprints the config.** Eighteen hyperparameters
   *are* recoverable from the state_dict — `pop_dim`, `hidden_dims`, per-layer
   `alpha`/`beta`/`threshold`/`reset_mechanism`, the encoder threshold, `observation_bounds`.
   All eighteen must agree before the config is accepted. A config that matches all
   eighteen is the config this checkpoint trained under, which is the strongest available
   evidence that its `num_steps` is right too.
3. **Both artifacts record the config's SHA-256**, and the flight suite requires them to
   match, so a graph and a golden built from different configs are caught.

What does **not** defend it: counting GEMM nodes in the exported graph. The graph and the
config come from the same source, so they agree by construction. That check catches a tracer
that stopped unrolling, nothing more.

Only `num_steps` and `reset_delay` sit outside the fingerprint. `spike_grad` also does, and
does not matter — it is the surrogate for the backward pass, and inference never touches it.

### Runs from before the config dump

`rl_training/rl_games/runner.py` writes `<run>/config.yaml` on `--train`. Older runs have
none; reconstruct it from the `cfg/*.yaml` that run used and drop it in. The fingerprint
above is what makes a reconstructed config trustworthy rather than merely plausible — if it
is from the wrong run, it is refused.

## Why the spiking actor exports at all

No special handling, no ONNX `Loop`, no hidden state for the flight side to plumb.
`pop_spiking_actor.forward` calls `reset_mem()` on every call, so the actor is a pure
function of the observation and tracing unrolls the loop into a static graph. `num_steps`
becomes structural — there is no input or attribute downstream that could change it:

| `num_steps` | 1 | 2 | 3 | 5 | 8 | 16 |
|---|---|---|---|---|---|---|
| graph nodes | 155 | 276 | 394 | 630 | 984 | 1928 |

(Counted with a fixed batch. The shipped graph keeps the batch axis dynamic for offline
evaluation, which blocks some constant folding — 750 nodes at `num_steps=5`, same
arithmetic.)

Cost is linear at ~35 µs per timestep — 5 steps is ~158 µs single-threaded against a 30 ms
control interval, so about 0.5% of budget. Do not optimize it.

ONNX discards the spiking structure entirely: the export is five unrolled dense GEMMs. It is
the CPU-parity path and a dead end for neuromorphic hardware, which would need a separate
NIR export.

## Flying it

```bash
cd ros2_ws && colcon build --packages-select control_policy_runner
source install/setup.bash

ros2 launch control_policy_runner policy_runner.launch.py \
  policy_checkpoint:=$(ros2 pkg prefix --share control_policy_runner)/policies/hover_snn/hover.onnx
```

The colcon build is required — `setup.py` copies every directory under `policies/` into the
package share.

**No flight-side code differs between the two architectures.** `OnnxHoverPolicy` loads a
graph plus a sibling `.npz` and knows nothing about what produced it; both policies share
the same 16-D observation, the same normalization and the same action clamp. Switching is a
path.

## Module map

| module | role |
|---|---|
| `publish.py` | **the entry point** — runs the two below against one checkpoint |
| `export_onnx.py` | writes the graph + stats, verifies against torch, refuses to ship a disagreement |
| `record_golden.py` | replays plausible flight states through the policy and freezes the answers |
| `checkpoint.py` | `RlGamesPolicy` (architecture-agnostic: `.pth`, normalizer, forward) and `MlpActorPolicy` |
| `snn_checkpoint.py` | `PopSANPolicy` — the spiking loader, and the config fingerprint |
| `hover.py` | `HoverPolicy` / `SnnHoverPolicy` — same observation, different network |
| `export_vae_onnx.py` | writes `depth_vae.onnx` — the DepthVAE encoder with the `mu` slice welded on |
| `navigation.py` | `NavigationPolicy` — the 49-D actor with the DepthVAE in front of it |

`checkpoint.py` rebuilds the MLP by hand from weight shapes. `snn_checkpoint.py` deliberately
does not: it imports the trained `PopulationSpikingActorNetwork`. Re-deriving snntorch's
`Synaptic` semantics — including the `reset_delay=False` same-step reset — is exactly the
transcription risk this whole boundary exists to remove.

`navigation.py` follows the same rule for depth: it wraps `vae_depth`'s own
`DepthVAEImageEncoder` rather than restating the preprocessing, feeding it metres by setting
`sensor_max_range = 1.0` so the simulator's normalization multiply becomes the identity. The
flight side reimplements it because it must; this side must not, or the two would agree with
each other and neither with training.

It also imports **no aerial_gym**. Reaching the task config would pull in isaacgym, which
must precede torch and wants a GPU — in front of what is otherwise checkpoint arithmetic. The
encoder's geometry and clamp window come from `control_policy_api.depth` instead, and
`tests/test_deploy_nav_obs_parity.py` asserts those equal the task's `vae_config`. Pinned by
test rather than by import.

## Prerequisites

`deploy/` imports `control_policy_api` for the observation layout, so the deployment package
and the training container agree on the vector by construction rather than by transcription:

```bash
pip install -e <sail-uav-core>/libs/frame-transforms -e <sail-uav-core>/libs/control-policy-api
```
