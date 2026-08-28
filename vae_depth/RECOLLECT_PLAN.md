# Re-collecting the depth dataset for the new obstacle environment

> **Status: phases 1-3 implemented and smoke-tested.** `vae_depth/data_generation/` now imports the
> nav env, the Poisson manager and the RL camera instead of restating them, and
> `vae_depth/config.py` points at the new dataset and its own cache root. Phases 4-6
> (build the cache, retrain, validate) are still to run.
>
> Two things below were changed by measurement rather than argument:
>
> - **Phase 2 stratification is OFF by default.** The premise — that a big box yields
>   mostly far/empty views — is wrong here, because the floor slab is in frame in nearly
>   every view. Measured over 1024 frames: 21.9% under 1 m, 31.2% 1-2 m, 39.6% 2-4 m,
>   6.8% 4-7 m, 0.6% beyond. The natural mix is already near-heavy, and with only ~7.4%
>   of far frames to redistribute into, any cap at or below 0.31 is arithmetically
>   unsatisfiable. The flag is kept for deliberate reshaping.
> - **`--num_envs` is not a throughput knob.** Reset and render are both linear in env
>   count, so per-image cost is flat (~205-229 ms across 8/32/64 envs). The lever that
>   does work is `--poses_per_layout`, which amortizes the 3.4 s obstacle reset over K
>   camera poses: 3.9 -> 6.3 img/s at K=4.


The 43k images in `~/DATA/depth-images/` were generated from `vae_depth/data_generation/config/env_config.py`,
which predates the navigation env rewrite. The VAE trained on them has never seen a ground
plane, a sphere, a cylinder or a perimeter wall — and a ground plane is in the lower third of
essentially every frame the policy now receives. Re-collect.

## 1. What actually diverged

`vae_depth/data_generation/config/env_config.py` (`DataGenEnvCfg`) vs `config/env_config/env_forest_with_obstacles.py`
(`ForestEnvCfg`, what the RL task builds):

| | data-gen (old) | nav env (now) |
|---|---|---|
| obstacle types | panels 15, thin 8, trees 4, objects 50 | objects 95, spheres 44, cylinders 44, panels 7, trees 6, 4 side walls |
| primitives | — | spheres r=0.2/0.4/0.6, cylinders post/stub/barrel |
| placement | per-asset ratio box, x-ratio [0.05,0.95] | homogeneous Poisson over env box, keep-out ellipsoid at spawn |
| count | `randint(keep_in_env, total)` uniform | `N ~ Poisson(λV)`, λ ramps 0 → 0.067 /m³ over curriculum levels 0–25 |
| trees | 4 of 77 slots (~5%), free z | 6 of 200 (3%), **z pinned to floor** |
| floor | none (`create_ground_plane=False`, no `bottom_wall`) | `bottom_wall` structural, present every reset |
| side walls | none | each present w.p. `num_assets/200`, ramps with curriculum |
| bounds | ~18–25 × 14–20 × 8–14 m, xy randomized | fixed 20 × 20, z randomized 4–6 m |
| render res | 1280×720, nearest-downsampled to 320×180 offline | native 320×180 |
| camera mount | fixed `[0.10, 0, 0.03]`, no noise | randomized ±5° / ±3 cm (`RealSenseD435CamConfig`) |
| view attitude | roll/pitch ±60° uniform | flight attitudes (spawn ±30°, commanded tilt smaller) |

Everything below the "placement" row is a distribution shift the encoder cannot compensate
for at inference.

## 2. Principle

The VAE's training set must be the *marginal distribution of depth images the policy will
actually see*. That means: same env config object, same asset manager, same camera config —
imported, not re-declared. Any parameter that exists in two places will drift again.

## 3. Plan

### Phase 0 — decide the two open knobs

**Render resolution: go native 320×180.** The RL path
(`vae_depth/vae_image_encoder.py:encode`) skips `F.interpolate` when the incoming shape
already matches the target, so at inference the encoder sees a *natively ray-cast* 320×180
image. Training on a nearest-downsample of a 1280×720 ray-cast is a free, avoidable shift
(different ray directions per pixel). Native also cuts generation time ~16× and the dataset
from 4.2 GB to well under 1 GB.

Follow-on edits: `VAEConfig.source_height/width` → 180/320, and the `cv2.resize` in
`DepthImageDataset._load` becomes a no-op (leave it; it is guarded by shape in the encoder,
not here — verify it costs nothing or short-circuit it).

**Pose distribution: teleport, but with the flight envelope.** Rolling out the current policy
to collect frames along real trajectories is the more faithful option, but it couples dataset
generation to a checkpoint that is itself trained on the old VAE. Do the cheap thing now:
sample poses uniformly in the env box with roll/pitch restricted to the actual flight envelope
(±30°, matching `f450_config.init_config`) rather than ±60°, and z restricted to the flyable
band. Note the rollout variant as the upgrade if reconstruction quality on live frames
(§5c) turns out to be pose-limited.

### Phase 1 — rewire `data_generation` onto the nav stack

1. **Env config**: delete `DataGenEnvCfg`'s asset declarations; subclass `ForestEnvCfg`.
   Override only what data collection needs:
   - `reset_on_collision = False` (camera may land inside an obstacle; the valid-pixel
     filter rejects those frames)
   - `num_physics_steps_per_env_step_mean = 1`, `std = 0`
   - `render_viewer_every_n_steps` large, `perturb_observations = False`
   Leave bounds, `include_asset_type`, `asset_type_to_dict_map` inherited. That is the whole
   point — the obstacle mix must not be restated.

2. **Asset manager**: inject `PoissonAssetManager` exactly as `task/attitude_navigation_task.py:223`
   does — same three post-construction assignments (`spawn_ratio_lo/hi` from
   `f450_config.init_config.min/max_init_state[0:3]`, `clearance` from
   `task_config.obstacle_spawn_clearance`). Without this the new asset pool is placed by the
   *upstream* ratio-box manager and the x-bias is back.

3. **Density**: replace the `num_obstacles_in_env` randint in `generate_dataset.py` with a
   per-reset `obstacle_intensity` draw. Sample a curriculum level `L ~ U{0..25}`, set
   `intensity = obstacle_density_max * L / density_at_level`, write it to
   `global_tensor_dict["obstacle_intensity"]`. Down-weight L ≤ 2 to ~5% of draws — near-empty
   frames carry no collision signal and the valid-pixel filter will discard many anyway now
   that the floor guarantees non-empty images.

4. **Camera**: use `config/sensor_config/realsense_d435_cam_config.py` (the RL one) in place of
   `vae_depth/data_generation/config/camera_config.py`. That brings 320×180, `min_range=0.1`, mount
   randomization and the noise block along with it. Delete the data-gen camera config so it
   cannot be picked up again.

5. **Robot pose**: keep the wide-position sampling in `DataGenQuadCfg`, narrow roll/pitch to
   ±π/6, and clamp the z ratio so the camera stays in the flyable band rather than pressed
   against the 4–6 m ceiling.

### Phase 2 — near-obstacle stratification

Uniform placement in a ~2000 m³ box mostly yields far/empty views, while the DCE target is
dominated by near geometry (the dilation radius goes as 1/d: 149 px at 0.75 m, 16 px at 7 m).
Add an acceptance test on top of `min_valid_ratio`: bucket each frame by its 5th-percentile
depth into `[<1, 1–2, 2–4, 4–7, >7] m` and cap each bucket's share (e.g. reject once a bucket
exceeds ~30% of the accepted total). Cheap — it is one `torch.quantile` per frame — and it is
the difference between a VAE that resolves close obstacles and one that averages them away.

### Phase 3 — generate

```bash
python vae_depth/data_generation/generate_panels.py           # only if the panel URDFs are missing
python -m vae_depth.data_generation.generate_dataset \
    --num_images 85000 --num_envs 64 \
    --output_dir ~/DATA/depth-images-forest
```
85k matches the DCE paper and native-res makes it cheap (~0.5–1 GB). Bump `--num_envs` well
past 16 — at 320×180 the ray-cast is no longer the bottleneck, PNG writing is.

Smoke first: `--num_images 64 --num_envs 4`, then eyeball ~20 frames for a visible floor,
spheres/cylinders, and occasional walls. If the floor is absent, `bottom_wall` was culled —
check `keep_in_env` survived the env subclassing.

### Phase 4 — rebuild the collision cache

**Footgun, read this first.** `cache_path_for` (`vae_depth/collision.py:298`) keys the cache on
the image **basename only**, and `cache_dir_for` on `dirname(data_dir)/../depth-collision/<geometry signature>`.
A new dataset at `~/DATA/depth-images-forest/depth_000000.png` therefore resolves to
`~/DATA/depth-collision/collision_v1_R0.482_.../depth_000000.png` — byte-identical path to the
*old* cache entry for a completely different image. It will load silently and train on
mismatched targets.

Set `collision_cache_dir` explicitly for the new run, or put the new data dir under a
different parent. Consider folding a dataset id into `cache_signature()` so this cannot
happen again.

```bash
python -m vae_depth.precompute_collision --validate 16   # with the new data_dir + cache dir
```
Keep `--validate`; it cross-checks the fast builder against a brute-force sphere sweep and
refuses to write if the fast path is optimistic by more than 0.15 m.

### Phase 5 — retrain and validate

Train the new VAE, keep `runs/20260218_204641/checkpoints/epoch_150.pth` as the A/B control.

Validation, in order of how much it tells you:

a. **Held-out reconstruction on new-env frames** — old VAE vs new VAE on the same new
   validation split. The old one should be clearly worse; if it is not, the env change did
   not move the image distribution as much as expected and the re-collection was cheap
   insurance rather than a fix.
b. **Active latent dimensions and per-dim KL** — already logged to TensorBoard. A drop in
   active dims on the richer dataset means the KL warmup needs retuning, not that the data
   is worse.
c. **Live-frame closed loop** — pull a batch of `depth_range_pixels` straight out of a running
   `attitude_navigation_task`, encode, decode, and compare against the collision target
   computed on those same frames. This is the only test that exercises the actual inference
   path including the mount randomization and sensor noise.
d. **Downstream** — short PPO run at a pinned curriculum level, old VAE vs new VAE frozen,
   same seed. Collision rate is the metric that matters.

### Phase 6 — bookkeeping

- Point `f450_attitude_navigation_task_config.vae_config.model_file` at the new checkpoint.
- `VAEConfig.data_dir`, `source_height/width`, `collision_cache_dir`.
- `max_depth_m = 7.0` against the sensor's `max_range = 10.0` is intentional and unchanged —
  both the encoder input clamp and the collision target's far edge use 7.0.
- Update `vae_depth/data_generation/README.md`'s obstacle table; it currently documents the old mix.
- Keep the old dataset until Phase 5a/5d are done, then delete.

## 4. Order of work

Phases 1 → 3 → 4 → 5 are sequential. Phase 2 can be skipped on the first pass and added if
5a shows near-range reconstruction is the weak spot — but it is ~20 lines, so prefer doing it
up front.
