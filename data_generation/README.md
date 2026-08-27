# Depth Image Dataset Generation for VAE Training

Renders the depth images that `vae_depth` trains on, following
[Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding](https://arxiv.org/abs/2402.03947).

## The one rule

**The dataset is rendered in the environment the policy flies in, through the camera the
policy reads from.** Nothing about the world or the sensor is declared here; it is
imported:

| what | where it comes from | what this package adds |
|---|---|---|
| obstacles, bounds, floor, walls | `config/env_config/env_forest_with_obstacles.py` | collection flags only (no collision reset, 1 substep) |
| obstacle placement | `env_manager/poisson_asset_manager.py` | injected exactly as the nav task injects it |
| obstacle density | `config/task_config/…_navigation_task_config.py` | a curriculum level drawn per layout |
| camera | `config/sensor_config/realsense_d435_cam_config.py` | nothing — used as-is |
| airframe | `config/robot_config/f450_config.py` | wide pose roam, gravity off |

The previous version of this package declared its own obstacle set and its own 1280x720
camera. Both drifted. By the time the navigation env had grown spheres, cylinders,
perimeter walls and a ground plane, the VAE was still being trained on a world containing
none of them — and on downsampled 1280x720 frames while inference ran on native 320x180
renders. Anything restated here is something that can drift again.

## What varies per frame

- **Obstacle layout** — a homogeneous Poisson point process over the env box, thinned by
  a keep-out ellipsoid around the policy's spawn box.
- **Obstacle density** — a curriculum level is drawn per layout (0-25) and converted to a
  Poisson intensity by the same formula the task uses, so the dataset is spread over the
  clutter the policy actually meets. Levels 0-2 are near-empty and drawn only 5% of the
  time (`--empty_level_prob`).
- **Camera pose** — position anywhere in the env (z clear of the floor slab and the
  ceiling), yaw uniform, roll/pitch up to the task's `max_inclination_angle_rad` (±45°).
  Not ±60°, which is outside anything the drone can command.
- **Camera mount** — ±5° and a few cm, from the sensor config, re-drawn every pose.

Frames with fewer than 5% valid pixels (camera inside an obstacle) are dropped.

## Poses per layout

Re-placing the obstacles costs ~3.4 s at 32 envs — the Poisson draw over ~200 assets plus
the warp BVH refit — while re-randomizing the camera pose costs 1.5 ms. `--poses_per_layout`
(default 4) renders K viewpoints per layout and cuts the per-image cost ~1.6x.

The cost is correlation: K frames share an obstacle field. Measured at K=4, frames sharing
a layout differ by as much mean depth as frames from different envs (3.48 m vs 3.55 m),
and 85k images still means ~21k independent layouts.

`--num_envs` is a **memory** knob, not a throughput one. Reset and render are both linear
in env count:

| envs | reset | render | per image (K=1) | per image (K=4) |
|---|---|---|---|---|
| 8 | 832 ms | 788 ms | 229 ms | 151 ms |
| 32 | 3399 ms | 3152 ms | 209 ms | 130 ms |
| 64 | 6640 ms | 6275 ms | 205 ms | 127 ms |

Raising it buys nothing but VRAM pressure.

## Usage

```bash
cd /workspaces/aerial_gym_docker
python -m data_generation.generate_dataset                                # 85k, defaults
python -m data_generation.generate_dataset --num_images 128 --num_envs 4  # smoke test
```

Measured: **6.3 img/s** at 32 envs with K=4, so 85k images take **~3.7 h** and **~1.8 GB**
(20.8 KB/frame, 16-bit PNG at 320x180).

### Output format

16-bit grayscale PNG, `depth_m = pixel / 65535 * 10.0`. Native 320x180 — the resize in
`vae_depth/dataset.py` is a no-op on this data, which is the point. `--format npy` writes
float32 in [0, 1] instead.

The sensor's near-out-of-range sentinel (-1) is clipped to 0 on write. Not a loss: 0 and
-1 both land on `min_depth_m` once `vae_depth.preprocessing.normalize_depth` clamps, on
the training and the inference path alike.

### After generating

The collision-target cache and the dataset **travel together** — `cache_path_for()` keys
on the image basename and `cache_signature()` encodes only geometry, so two datasets whose
files are both named `depth_000000.png` resolve to the same cache entry and the stale one
loads silently. `vae_depth/config.py` sets both:

```python
data_dir            = "~/DATA/depth-images-forest/"
collision_cache_dir = "~/DATA/depth-collision-forest/"
```

Then build the targets and train:

```bash
python -m vae_depth.precompute_collision --validate 16
python -m vae_depth.train --num_epochs 100 --batch_size 64
```

## Known gap: the panel pool is one mesh

`aerial_gym/resources/models/environment_assets/panels/` contains a single `panel.urdf`,
so all 7 panel slots in every env draw the same mesh. `generate_panels.py` writes 50
randomized panel URDFs there and fixes it.

It is **not** run by default, because that directory is shared with RL training: adding
panels changes the world the policy flies in as well as the world the dataset is rendered
in. Run it before collecting, or not at all — but do not run it *between* collecting the
dataset and training the policy, or the two will disagree again.

```bash
python data_generation/generate_panels.py   # optional; changes the RL env too
```

## File structure

```
data_generation/
├── __init__.py           # registry registration
├── config/
│   ├── env_config.py     # ForestEnvCfg + collection flags
│   └── robot_config.py   # F450Config + roaming pose, gravity off
├── generate_dataset.py   # collection loop
├── generate_panels.py    # optional panel URDF variety (see above)
└── generate_objects.py
```

`camera_config.py` is gone on purpose — the camera is `RealSenseD435CamConfig`.
