"""Main VAE training script.

Usage:
    python -m vae_depth.train [--args]

Example:
    # Smoke test
    python -m vae_depth.train --num_epochs 2 --batch_size 16

    # Full training
    python -m vae_depth.train --num_epochs 100 --batch_size 64
"""

import argparse
import os
import time
from datetime import datetime
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vae_depth.config import VAEConfig
from vae_depth.dataset import DepthImageDataset
from vae_depth.loss import beta_schedule, vae_loss
from vae_depth.model import DepthVAE
from vae_depth.preprocessing import min_pool_dilation, normalize_depth


def prepare_input(batch_depth_m, config):
    """Prepare encoder input: normalize only (no dilation).

    Args:
        batch_depth_m: [B, 1, H, W] raw depth in meters (on GPU).
        config: VAEConfig.

    Returns:
        [B, 1, H, W] normalized depth in [0, 1].
    """
    return normalize_depth(batch_depth_m, config.max_depth_m, config.min_depth_m)


def prepare_target(batch_target_m, config):
    """Prepare decoder target: normalize the collision target.

    In "collision" mode the dataset already supplies the precomputed collision-encoding
    target, so there is nothing to dilate here. In "legacy" mode it supplies raw depth and
    the old single-range min-pool runs on it, for A/B comparison only.

    Args:
        batch_target_m: [B, 1, H, W] target depth in meters (on GPU).
        config: VAEConfig.

    Returns:
        [B, 1, H, W] normalized target in [0, 1].
    """
    x = batch_target_m
    if config.collision_target_mode == "legacy":
        x = min_pool_dilation(x, config.dilation_kernel_size)
    return normalize_depth(x, config.max_depth_m, config.min_depth_m)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Depth VAE")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--beta_target", type=float, default=None)
    parser.add_argument("--beta_warmup_epochs", type=int, default=None)
    parser.add_argument("--obstacle_weight", type=float, default=None)
    parser.add_argument("--range_weight_power", type=float, default=None)
    parser.add_argument("--collision_target_mode", type=str, default=None,
                        choices=["collision", "legacy"],
                        help="'collision' uses the precomputed per-range target; "
                             "'legacy' is the old single-range min-pool, for A/B only")
    parser.add_argument("--collision_radius_m", type=float, default=None)
    parser.add_argument("--safety_margin_m", type=float, default=None)
    parser.add_argument("--near_floor_m", type=float, default=None)
    parser.add_argument("--collision_cache_dir", type=str, default=None)
    parser.add_argument("--dilation_kernel_size", type=int, default=None)
    parser.add_argument("--drone_radius_m", type=float, default=None)
    parser.add_argument("--safety_margin_fraction", type=float, default=None)
    parser.add_argument("--reference_distance_m", type=float, default=None)
    parser.add_argument("--max_depth_m", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def apply_args(config: VAEConfig, args) -> VAEConfig:
    """Override config defaults with CLI arguments (if provided)."""
    for key, value in vars(args).items():
        if value is not None and hasattr(config, key):
            setattr(config, key, value)
    # Re-run post_init to recompute dilation kernel if drone params changed
    config.__post_init__()
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "config": {k: v for k, v in config.__dict__.items()},
        },
        filepath,
    )


def apply_colormap(tensor_1ch, cmap_name="turbo", per_image_norm=True):
    """Apply a matplotlib colormap to a [N, 1, H, W] tensor -> [N, 3, H, W] RGB.

    Args:
        per_image_norm: If True, normalize each image to its own [min, max] so the
            full color range is used. Otherwise uses raw values (assumes [0, 1]).
    """
    import matplotlib.cm as cm

    cmap = cm.get_cmap(cmap_name)
    np_imgs = tensor_1ch.squeeze(1).cpu().numpy()  # [N, H, W]
    rgb_list = []
    for img in np_imgs:
        if per_image_norm:
            vmin, vmax = img.min(), img.max()
            if vmax - vmin > 1e-6:
                img = (img - vmin) / (vmax - vmin)
            else:
                img = np.zeros_like(img)
        rgba = cmap(img)  # [H, W, 4] float in [0,1]
        rgb = rgba[:, :, :3]  # drop alpha
        rgb_list.append(rgb)
    rgb_np = np.stack(rgb_list, axis=0)  # [N, H, W, 3]
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()  # [N, 3, H, W]
    return rgb_tensor


def select_diverse_samples(val_loader, device, config, num_images=8):
    """Pick validation images spanning a range of obstacle content."""
    candidates = []
    with torch.no_grad():
        for batch_depth_m, batch_target_m in val_loader:
            batch_depth_m = batch_depth_m.to(device)
            batch_target_m = batch_target_m.to(device)
            batch_target = prepare_target(batch_target_m, config)
            for i in range(batch_target.size(0)):
                obstacle_frac = (batch_target[i] > config.obstacle_threshold).float().mean().item()
                candidates.append((obstacle_frac, batch_depth_m[i:i+1], batch_target_m[i:i+1]))
            if len(candidates) >= 500:
                break
    # Sort by obstacle fraction, pick evenly spaced samples
    candidates.sort(key=lambda x: x[0])
    step = max(1, len(candidates) // num_images)
    selected = [candidates[i * step] for i in range(num_images)]
    return (torch.cat([s[1] for s in selected], dim=0),
            torch.cat([s[2] for s in selected], dim=0))


def log_reconstruction_images(writer, model, val_loader, device, epoch, config, num_images=8,
                              _cached_samples=[None]):
    """Log sample reconstruction images to TensorBoard: Original | Dilated GT | Predicted."""
    model.eval()
    with torch.no_grad():
        # Cache diverse samples on first call so we track the same images across epochs
        if _cached_samples[0] is None:
            _cached_samples[0] = select_diverse_samples(val_loader, device, config, num_images)

        batch_depth_m = _cached_samples[0][0].to(device)
        batch_input = prepare_input(batch_depth_m, config)
        batch_target = prepare_target(_cached_samples[0][1].to(device), config)
        x_recon, _, _, _ = model(batch_input)

        # Apply turbo colormap with shared [0,1] scale (no per-image normalization)
        input_rgb = apply_colormap(batch_input, per_image_norm=False)
        target_rgb = apply_colormap(batch_target, per_image_norm=False)
        recon_rgb = apply_colormap(x_recon, per_image_norm=False)

        # Side by side: [N, 3, H, 3*W] -- Original | Dilated GT | Predicted
        comparison = torch.cat([input_rgb, target_rgb, recon_rgb], dim=3)
        writer.add_images("reconstruction/input_dilated_recon", comparison, epoch)

        # Error heatmap: predicted vs dilated GT
        error = torch.abs(batch_target - x_recon)
        error_rgb = apply_colormap(error, cmap_name="hot")
        writer.add_images("reconstruction/error", error_rgb, epoch)


def train_one_epoch(model, train_loader, optimizer, device, beta, config):
    model.train()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="  Train", leave=False)
    for batch_depth_m, batch_target_m in pbar:
        batch_depth_m = batch_depth_m.to(device)
        batch_input = prepare_input(batch_depth_m, config)
        batch_target = prepare_target(batch_target_m.to(device), config)

        x_recon, mu, logvar, z = model(batch_input)
        loss, recon_loss, kl_loss = vae_loss(
            x_recon, batch_target, mu, logvar, beta,
            config.obstacle_weight, config.obstacle_threshold,
            config.range_weight_power,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()

        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.5f}", recon=f"{recon_loss.item():.5f}", kl=f"{kl_loss.item():.3f}")

    return (
        total_loss_sum / num_batches,
        recon_loss_sum / num_batches,
        kl_loss_sum / num_batches,
    )


@torch.no_grad()
def validate(model, val_loader, device, beta, config):
    model.eval()
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    num_batches = 0

    all_mu = []
    all_logvar = []

    pbar = tqdm(val_loader, desc="  Val  ", leave=False)
    for batch_depth_m, batch_target_m in pbar:
        batch_depth_m = batch_depth_m.to(device)
        batch_input = prepare_input(batch_depth_m, config)
        batch_target = prepare_target(batch_target_m.to(device), config)

        x_recon, mu, logvar, z = model(batch_input)
        loss, recon_loss, kl_loss = vae_loss(
            x_recon, batch_target, mu, logvar, beta,
            config.obstacle_weight, config.obstacle_threshold,
            config.range_weight_power,
        )

        total_loss_sum += loss.item()
        recon_loss_sum += recon_loss.item()
        kl_loss_sum += kl_loss.item()
        num_batches += 1

        all_mu.append(mu.cpu())
        all_logvar.append(logvar.cpu())

        pbar.set_postfix(loss=f"{loss.item():.5f}")

    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)

    return (
        total_loss_sum / num_batches,
        recon_loss_sum / num_batches,
        kl_loss_sum / num_batches,
        all_mu,
        all_logvar,
    )


def main():
    args = parse_args()
    config = VAEConfig()
    config = apply_args(config, args)

    # Seed
    torch.manual_seed(config.seed)

    print(f"VAE Depth Training")
    print(f"  latent_dim:           {config.latent_dim}")
    print(f"  dilation_kernel_size: {config.dilation_kernel_size}")
    print(f"  max_depth_m:          {config.max_depth_m}")
    print(f"  beta_target:          {config.beta_target}")
    print(f"  obstacle_weight:      {config.obstacle_weight}")
    print(f"  batch_size:           {config.batch_size}")
    print(f"  device:               {config.device}")
    print()

    # Dataset (filter out empty/corrupt files)
    all_paths = sorted(glob(os.path.join(config.data_dir, f"*.{config.image_format}")))
    all_paths = [p for p in all_paths if os.path.getsize(p) > 0]
    if len(all_paths) == 0:
        raise FileNotFoundError(f"No valid .{config.image_format} files found in {config.data_dir}")
    print(f"Found {len(all_paths)} valid images in {config.data_dir}")

    # Train/val split
    split_idx = int(len(all_paths) * (1 - config.val_split))
    rng = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(all_paths), generator=rng).tolist()
    train_paths = [all_paths[i] for i in perm[:split_idx]]
    val_paths = [all_paths[i] for i in perm[split_idx:]]
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    train_dataset = DepthImageDataset(train_paths, config, augment=True)
    val_dataset = DepthImageDataset(val_paths, config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )

    # Model
    device = torch.device(config.device)
    model = DepthVAE(latent_dim=config.latent_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    encoder_count = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model params: {param_count:,} total, {encoder_count:,} encoder")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.lr_patience,
        factor=config.lr_factor,
        min_lr=config.lr_min,
        threshold=config.lr_threshold,
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    # Experiment directory (TensorBoard + checkpoints together)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config.log_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    config.checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(experiment_dir)
    print(f"Experiment dir: {experiment_dir}")
    print()

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        t0 = time.time()
        beta = beta_schedule(epoch, config.beta_warmup_epochs, config.beta_target)
        lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_recon, train_kl = train_one_epoch(
            model, train_loader, optimizer, device, beta, config
        )

        # Validate
        val_loss, val_recon, val_kl, val_mu, val_logvar = validate(
            model, val_loader, device, beta, config
        )

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        # Console logging
        print(
            f"Epoch {epoch:3d}/{config.num_epochs} | "
            f"train: {train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.4f}) | "
            f"val: {val_loss:.6f} (recon={val_recon:.6f}, kl={val_kl:.4f}) | "
            f"beta={beta:.4f} lr={lr:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard scalars
        writer.add_scalar("loss/train_total", train_loss, epoch)
        writer.add_scalar("loss/train_recon", train_recon, epoch)
        writer.add_scalar("loss/train_kl", train_kl, epoch)
        writer.add_scalar("loss/val_total", val_loss, epoch)
        writer.add_scalar("loss/val_recon", val_recon, epoch)
        writer.add_scalar("loss/val_kl", val_kl, epoch)
        writer.add_scalar("params/beta", beta, epoch)
        writer.add_scalar("params/learning_rate", lr, epoch)

        # Per-dimension KL
        kl_per_dim = -0.5 * (1.0 + val_logvar - val_mu.pow(2) - val_logvar.exp())
        kl_per_dim_mean = kl_per_dim.mean(dim=0)
        for d in range(config.latent_dim):
            writer.add_scalar(f"kl_per_dim/dim_{d:02d}", kl_per_dim_mean[d].item(), epoch)
        active_dims = (kl_per_dim_mean > 0.1).sum().item()
        writer.add_scalar("latent/active_dims", active_dims, epoch)

        # Latent histograms
        writer.add_histogram("latent/mu", val_mu, epoch)
        writer.add_histogram("latent/logvar", val_logvar, epoch)

        # Reconstruction images
        if epoch % config.image_log_interval == 0:
            log_reconstruction_images(writer, model, val_loader, device, epoch, config)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config.checkpoint_dir, "best.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, best_path)
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.6f})")

        # Save periodic checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch + 1}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, config, ckpt_path)
            print(f"  -> Saved checkpoint epoch_{epoch + 1}.pth")

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Experiment dir: {experiment_dir}")
    print(f"TensorBoard: tensorboard --logdir {experiment_dir}")


if __name__ == "__main__":
    main()
