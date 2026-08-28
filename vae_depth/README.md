# Depth VAE with Deep Collision Encoding (DCE)

Variational Autoencoder that compresses 320x180 depth images into a 32-dimensional latent vector for drone navigation RL. The key idea: the encoder receives **raw depth** while the decoder learns to reconstruct **collision-dilated depth**, so the latent space implicitly encodes collision safety without requiring dilation at inference time.

## Method

### Deep Collision Encoding

Standard depth VAEs encode and reconstruct the same image. DCE changes this: the encoder input is the original depth image, but the reconstruction target is a min-pool dilated version that expands obstacles by the drone's collision radius. This forces the latent representation to internalize obstacle safety margins.

At RL inference time, the encoder receives raw depth from the simulator and outputs a latent vector that already accounts for the drone's physical size -- no dilation preprocessing needed.

### Preprocessing Pipeline

**Encoder input** (raw depth):
1. Load 16-bit PNG depth image (Intel RealSense D435, 1280x720)
2. Convert to meters: `pixel_value / 65535 * depth_scale`
3. Resize to 320x180 (nearest-neighbor)
4. Linear normalization: `normalized = 1 - depth / max_depth` (near=1, far=0)

**Decoder target** (collision depth):
1. Same as above, but after step 3:
2. Collision dilation (`collision.py`), precomputed and cached
3. Then normalize

The target is the Minkowski sum of the obstacle set with the drone's collision sphere,
re-rendered as *the depth at which the drone's centre first collides along each ray*:

```
t_contact(u) = u.p - sqrt(R^2 - |p|^2 + (u.p)^2)      rays passing within R of p
target(u,v)  = min over obstacle points p of  t_contact * u_z
```

with `R = collision_radius_m + safety_margin_m` = 0.357 + 0.125 = **0.482 m**. The drone
radius is the F450's own swept circle, taken from `resources/robots/f450/model.urdf`: prop
hubs 0.230 m out, prop discs 0.127 m — so 0.357 m, and 0.579 m tip to tip. (Do not use half
the 450 mm wheelbase: that is measured motor-to-motor and excludes the props.)

Depth is clamped to `[near_floor_m, max_depth_m]` = `[0.75, 7.0]`. The far end matches the
encoder input's clamp so both sides agree on what "far" means; the near end is *higher* than
the input's `min_depth_m`, because below `R` the drone's sphere already contains the point
and no finite dilation radius exists — those pixels are pinned to the floor rather than
given an unbounded kernel.

Consequently the dilation radius is **range-dependent**, roughly as `1/d`:

| range | radius | margin encoded |
|---|---|---|
| 0.75 m | 149 px | 0.482 m |
| 1 m | 112 px | 0.482 m |
| 3 m | 37 px | 0.482 m |
| 7 m | 16 px | 0.482 m |

### Building the target cache

Required before training in the default `collision` mode:

```bash
python -m vae_depth.precompute_collision --validate 16
```

~21 min and ~2.3 GB for the 85k dataset, measured. The script prints its own projection
from a `--limit` trial run.

The cache directory name encodes every geometric parameter, so within one dataset a stale
cache cannot be picked up silently. It does **not** encode the dataset, which is why
`collision_cache_dir` is set explicitly per dataset — see "Cache pairing" below.

The script refuses to write if the fast builder disagrees with a brute-force sphere-sweep
by more than `--max_unsafe` (default 0.15 m) in the optimistic direction. On the 85k forest
dataset it passes at +0.108 m, 98.2 % of pixels conservative.

Caching is valid because the dilation commutes with the crop/flip augmentation — verified
in `tests/test_collision.py`, which also shows the cached path is never *less* conservative
than dilating after the crop.

### Legacy target (A/B only)

`--collision_target_mode legacy` restores the original single-range min-pool: a fixed 17 px
kernel calibrated at 3 m, inflating by `0.5 × 0.25 m` = 0.125 m — half of a drone radius
that was itself wrong. Against the F450's real 0.482 m sphere that is **3.9× too small even
at the reference range**. It needs no cache.
Kept only to compare the two encodings; it under-inflates at close range (0.047 m of margin
at 1 m, 38 % of intended) and never applies the longitudinal pull-in.

### Loss Function

```
loss = weighted_MSE(predicted, dilated_target) + beta * KL_divergence
```

- **Weighted MSE**: Distance-proportional weighting gives higher weight to near obstacles. Weight = `1 + obstacle_weight * normalized_depth^power` for all pixels.
- **KL divergence**: Standard VAE regularization with beta warmup (0 -> 0.001 over 10 epochs) to prevent posterior collapse.

### Architecture

**Encoder**: 4 convolutional blocks (32->64->128->256 channels), each with two 3x3 convs + BatchNorm + ELU. Followed by 1x1 channel reduction, flatten, and FC head (512 -> 2*latent_dim). Outputs concatenated mu and logvar.

**Decoder**: FC head (latent_dim -> 512 -> reshape 6x10), 1x1 channel expansion, then 5 transposed conv layers (256->128->128->64->32->1) with BatchNorm + ELU. Final Sigmoid activation.

**Latent space**: 32 dimensions. During training, uses reparameterization trick for sampling. During RL inference, uses mu only (deterministic).

## Dataset Generation

Depth images are rendered in the environment the policy flies in
(`config/env_config/env_forest_with_obstacles.py`), through the camera the policy reads
from (`config/sensor_config/realsense_d435_cam_config.py`). See [`data_generation/`](data_generation/README.md).

```bash
python -m vae_depth.data_generation.generate_dataset          # 85k images, ~3.7 h, ~1.8 GB
```

Output: 16-bit PNG in `~/DATA/depth-images-forest/`, **native 320x180** — so the resize in
`dataset.py` is a no-op and the encoder sees the same ray-cast at train and inference time.
`depth_m = pixel / 65535 * 10.0`. RealSense D435 parameters: 87 deg HFOV, 0.1-10 m.

Obstacle layouts are a Poisson process over spheres, cylinders, objects, panels, trees and
cullable perimeter walls, above a ground plane, at a density drawn per layout from the
task's curriculum.

> The dataset in `~/DATA/depth-images/` predates this and was collected in an environment
> with no ground plane, no spheres, no cylinders and no walls. It is not a valid training
> set for the current task.

### Cache pairing

`data_dir` and `collision_cache_dir` must be repointed **together**. `cache_path_for()`
keys the cache on the image basename and `cache_signature()` encodes only geometry, so two
datasets whose files are both named `depth_000000.png` resolve to the same cache entry.
The stale target then loads without error and trains the decoder against a different
image. Repointing one without the other is silently wrong, not merely stale.

## Training

```bash
# Smoke test
python -m vae_depth.train --num_epochs 2 --batch_size 16

# Full training
python -m vae_depth.train --num_epochs 100 --batch_size 64

# Monitor
tensorboard --logdir vae_depth/runs/
```

TensorBoard shows 3-column reconstruction images (Raw Depth | Dilated GT | Predicted), error heatmaps, per-dimension KL divergence, and active latent dimension count.

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| latent_dim | 32 | Latent space dimensionality |
| batch_size | 64 | Training batch size |
| learning_rate | 1e-4 | Adam optimizer LR |
| beta_target | 0.001 | KL divergence weight |
| obstacle_weight | 5.0 | Near-obstacle loss emphasis |
| max_depth_m | 7.0 | Depth clipping distance |

## Evaluation

```bash
python -m vae_depth.evaluate --checkpoint vae_depth/runs/<timestamp>/checkpoints/best.pth
```

Generates reconstruction comparison images, error heatmaps, and latent space statistics.

## RL Integration

```python
from vae_depth.vae_image_encoder import DepthVAEImageEncoder

encoder = DepthVAEImageEncoder(config, device="cuda:0")
latents = encoder.encode(depth_images)  # [num_envs, 32]
```

The encoder receives raw depth from the simulator (no dilation needed) and outputs a 32-dim latent vector with implicit collision safety.
