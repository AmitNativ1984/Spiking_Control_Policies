import os
import math
from dataclasses import dataclass, field


@dataclass
class VAEConfig:
    # Data
    # data_dir and collision_cache_dir TRAVEL TOGETHER. cache_path_for() keys the cache on
    # the image BASENAME alone, and cache_signature() encodes only geometry -- neither
    # carries any dataset identity. Two datasets whose files are both named depth_000000.png
    # therefore resolve to the same cache path, and the stale entry loads without error,
    # training the decoder against targets belonging to a different image. Repointing
    # data_dir without repointing the cache root is silently wrong, not merely stale.
    data_dir: str = os.path.expanduser("~/DATA/depth-images-forest/")
    image_format: str = "png"
    # Frames are ray-cast natively at 320x180 by RealSenseD435CamConfig -- the same sensor
    # config the RL task reads through -- so source == target and dataset.py's resize is a
    # no-op. The old 1280x720 came from a separate data-gen camera that no longer exists;
    # training on a nearest-downsample of it while running on a native 320x180 render meant
    # the encoder saw different ray directions per pixel at train and inference time.
    source_height: int = 180
    source_width: int = 320

    target_height: int = 180
    target_width: int = 320

    # Depth normalization
    max_depth_m: float = 7.0
    min_depth_m: float = 0.1
    depth_scale: float = 10.0  # pixel_value / 65535 * depth_scale = meters

    hfov_deg: float = 87.0

    # --- Collision target (Deep Collision Encoding) -------------------------------------
    # "collision": per-range Minkowski dilation, precomputed and cached (vae_depth/collision.py)
    # "legacy":    the original single-range min-pool, kept for A/B comparison only
    collision_target_mode: str = "collision"
    # F450 swept radius, derived from resources/robots/f450/model.urdf: the prop hubs sit at
    # (+-0.162635, +-0.162635) from base_link, i.e. 0.230 m out, and each prop is a 0.127 m
    # radius disc -- so a yawing F450 sweeps a circle of 0.230 + 0.127 = 0.357 m and spans
    # 0.579 m tip to tip. (0.25 m, the previous value, is roughly half the 450 mm WHEELBASE,
    # which is measured motor-to-motor and ignores the propellers entirely.)
    collision_radius_m: float = 0.357
    safety_margin_m: float = 0.125      # added to it; the sphere is the SUM, see collision_radius()
    near_floor_m: float = 0.75          # depths below this are pinned: the drone's sphere
                                        # already contains the point, so no finite radius
                                        # exists. Must clear the 0.482 m sphere with room --
                                        # just above it, 1/sqrt(d^2-R^2) explodes.
    collision_buckets: int = 16         # depth buckets; radius comes from each bucket's near edge
    collision_shells: int = 8           # flat shells approximating the hemispherical element
    # Set EXPLICITLY, per dataset. The "" default resolves to
    # <data_dir>/../depth-collision/<signature>, which puts every dataset under ~/DATA/ into
    # ONE cache root; the signature is pure geometry, so depth-images/ and
    # depth-images-forest/ would collide entry for entry. See the note on data_dir.
    collision_cache_dir: str = os.path.expanduser("~/DATA/depth-collision-forest/")
    collision_algo_version: int = 1     # bump to invalidate every cached target

    # --- Legacy min-pool dilation (collision_target_mode == "legacy" only) ---------------
    drone_radius_m: float = 0.25
    safety_margin_fraction: float = 0.5
    reference_distance_m: float = 3.0
    dilation_kernel_size: int = 0  # 0 = auto-compute from drone params

    # Augmentation
    crop_prob: float = 0.5
    crop_scale_min: float = 0.8  # random crop keeps 80-100% of the resized image
    crop_scale_max: float = 1.0
    flip_prob: float = 0.5

    # VAE architecture
    latent_dim: int = 32
    encoder_channels: list = field(default_factory=lambda: [32, 64, 128, 256, 256])
    fc_hidden_dim: int = 256

    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    val_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True

    # Loss
    beta_target: float = 0.001
    beta_warmup_epochs: int = 10
    obstacle_weight: float = 5.0
    obstacle_threshold: float = 0.0  # 0 = fully continuous weighting across all depths
    range_weight_power: float = 1.0  # exponent for distance-proportional weighting (0 = binary)

    # LR schedule
    lr_patience: int = 10
    lr_factor: float = 0.5
    lr_min: float = 1e-5
    lr_threshold: float = 5e-4

    # Checkpointing
    checkpoint_dir: str = "vae_depth/checkpoints"
    checkpoint_interval: int = 10

    # TensorBoard
    log_dir: str = "vae_depth/runs"
    image_log_interval: int = 5  # log reconstruction images every N epochs

    # Device
    device: str = "cuda:0"
    seed: int = 42

    def __post_init__(self):
        if self.collision_target_mode not in ("collision", "legacy"):
            raise ValueError(
                f"collision_target_mode must be 'collision' or 'legacy', "
                f"got {self.collision_target_mode!r}")
        # reach_px divides by sqrt(d^2 - R^2): a floor at or below the sphere radius makes
        # the radius infinite rather than merely large.
        if self.near_floor_m <= self.collision_radius_m + self.safety_margin_m:
            raise ValueError(
                f"near_floor_m ({self.near_floor_m}) must exceed the collision sphere "
                f"radius ({self.collision_radius_m + self.safety_margin_m}); at or below it "
                "the drone already contains the point and no dilation radius is finite.")

        if self.dilation_kernel_size == 0:
            self.dilation_kernel_size = compute_dilation_kernel(
                drone_radius_m=self.drone_radius_m,
                safety_margin_fraction=self.safety_margin_fraction,
                reference_distance_m=self.reference_distance_m,
                hfov_deg=self.hfov_deg,
                image_width=self.target_width,
            )


def compute_dilation_kernel(
    drone_radius_m: float,
    safety_margin_fraction: float,
    reference_distance_m: float,
    hfov_deg: float,
    image_width: int,
) -> int:
    """Compute min-pool dilation kernel size from drone geometry.

    Args:
        drone_radius_m: Half the widest drone dimension including propellers.
        safety_margin_fraction: Fraction of drone radius to use as dilation margin.
        reference_distance_m: Distance at which the kernel size is calibrated.
        hfov_deg: Camera horizontal field of view in degrees.
        image_width: Image width in pixels.

    Returns:
        Odd integer kernel size (minimum 3).
    """
    hfov_rad = math.radians(hfov_deg)
    pixel_size_m = 2.0 * reference_distance_m * math.tan(hfov_rad / 2.0) / image_width
    margin_m = safety_margin_fraction * drone_radius_m
    margin_pixels = math.ceil(margin_m / pixel_size_m)
    kernel_size = 2 * margin_pixels + 1
    return max(kernel_size, 3)
