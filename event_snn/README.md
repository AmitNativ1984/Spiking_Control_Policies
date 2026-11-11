# event_snn

Implements [docs/event_snn_experiment_plan.md](../docs/event_snn_experiment_plan.md): a stateful SNN
watches the event stream (+ IMU + attitude estimate) while the depth teacher flies, and learns
inverse depth (32×32) and body velocity by supervised BPTT.

| File | Role |
|---|---|
| `cfg/default.yaml` | All settings; variants overlay it (`config.VARIANTS`) |
| `sim.py` | Isaac Gym rollout at 100 Hz: teacher (~33 Hz, D435 depth resampled from the event camera) + perturbations, IMU bias walk, attitude noise, targets and masks; held-out writer |
| `model.py` | `EventSNN`: 3 spiking convs → adaptive-LIF 8×8 memory (+ population-coded state) → depth / velocity leaky readouts |
| `loss.py` | Whole-image log-inverse-depth L1 + scale-invariant term, velocity MSE, spike-rate regulariser, burn-in mask |
| `train.py` | Fly a window, 2 BPTT passes in env mini-batches, carry detached state; window curriculum on validation plateau |
| `visualize.py` | Held-out flight video: events, target vs predicted 1/Z, error, cell weights, velocity |
| `evaluate.py` | Held-out metrics, blackout, attitude sweep, regimes, linear probes, plots, `--compare` |

## Scale-free encoder (current design)

`mb_*` variants: conv 1-3 -> 128 recurrent adaptive-LIF neurons (conv-3 spikes and the 90 inertial
spikes both feed them directly) -> training-only decoder. The 128 spikes are the memory and the
policy interface. No depth and no velocity output; the loss is one 32x32 map plus the rate penalty.

| Variant | Map | Decoder |
|---|---|---|
| `mb_ttc_snn` / `mb_ttc_ann` | inverse time-to-contact: closing speed / range | spiking / ReLU convs on a leaky spike trace |
| `mb_near_snn` / `mb_near_ann` | speed / range (non-zero sideways too) | spiking / ReLU convs on a leaky spike trace |

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_event_snn.py
python -m event_snn.sim --heldout                              # once; holds both maps
python -m event_snn.train --variant mb_ttc_ann --set train.wandb=true
python -m event_snn.evaluate runs/event_snn/<run>/best.pt
python -m event_snn.grad_diag runs/event_snn/<run>/last.pt     # per-parameter gradient norms
```

Validation prints the map loss next to `constant map`, the loss of one value everywhere: a run
that stays near it has learned the average level and nothing spatial. `evaluate` still scores
blackout, attitude sweep and regimes on the depth output, which `mb_*` does not train.

## Representation study with the 8x8 grid memory (superseded)

`bn_*` variants put a 128-neuron spiking bottleneck between the memory and the heads. Those spikes
are what a spiking policy will read; the heads are training-only and see nothing else.

| Variant | Heads must decode |
|---|---|
| `bn_depth` | inverse depth + body velocity |
| `bn_ttc` | inverse time-to-contact (invTTC) + body velocity |
| `bn_both` | all three |

`evaluate` scores each on the same held-out flights: invTTC error in the danger region (direct head,
and rebuilt from depth × velocity), and `act recov`, the share of the teacher's depth-latent
contribution to its action that a linear readout recovers from the bottleneck spikes.

```bash
MODE=check   sbatch slurm/event_snn.sbatch                    # teacher flight, invTTC frame check, throughput
MODE=train VARIANT=bn_both SEED=0 sbatch slurm/event_snn.sbatch
```

```bash
python -m event_snn.sim --check_teacher --seconds 30          # does the teacher fly at 100 Hz?
python -m event_snn.sim --heldout                             # fixed evaluation flights (once)
python -m event_snn.train --variant a_rec                     # a_rec | a_ff | b | a_rec_noimu | a_rec_rawimu
python -m event_snn.train --variant b --set train.lr=5e-4 sim.num_envs=32
python -m event_snn.evaluate runs/event_snn/<run>/best.pt
python -m event_snn.evaluate --compare "runs/event_snn/*"
python -m event_snn.visualize runs/event_snn/<run>/best.pt --env 3 --seconds 10   # depth video
python -m event_snn.visualize <ckpt> --env 3 --blackout 5.0 0.5                     # events off 5.0-5.5 s
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_event_snn.py
```

Laptop (RTX 3070 Ti, 8 GB), 64 envs: one 10 ms step costs ~77 ms, almost all of it the event-camera
kernel (two rays per pixel: event sample + `write_depth`). A 2.0 s window with mini-batches of 16
envs peaks at ~6.6 GB including the sim.
