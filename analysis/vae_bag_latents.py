"""Encode a recorded depth rosbag with the deployed DepthVAE and log it to TensorBoard.

Decoding each latent shows what the policy can know about the scene: if a tree straight
ahead is in the collision target but not in the reconstruction, the latent dropped it.
A random sample of the VAE training set is processed the same way as the in-distribution
baseline.

Writes <bag>/vae_latents.npz and <bag>/vae_tb/.

usage: python analysis/vae_bag_latents.py <bag_dir> [--train-samples 3000]
       tensorboard --logdir <bag>/vae_tb --samples_per_plugin images=5000
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mcap.reader import make_reader
from rosbags.serde import deserialize_cdr
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/home/amit/DEV/SAIL/sail-uav-core/libs/control-policy-api/src")
from control_policy_api.depth import preprocess_depth  # noqa: E402  -- the deployed preprocessing
from vae_depth.collision import collision_target  # noqa: E402
from vae_depth.config import VAEConfig  # noqa: E402
from vae_depth.model import DepthVAE  # noqa: E402

CHECKPOINT = "/workspaces/aerial_gym_docker/vae_depth/runs/20260828_060313/checkpoints/epoch_200.pth"
TRAIN_DIR = "/home/amit/DATA/depth-images-forest"
MAX_D = 7.0
# "Straight ahead": the collision target is already dilated by the drone's swept radius,
# so a small central window is the flight corridor along the optical axis.
AHEAD = (slice(72, 108), slice(128, 192))
MISS_NEAR_M, MISS_GAP_M = 3.0, 1.5
# Gazebo F450: the propellers enter the top corners of the D435 view at 0.1-0.5 m (mostly
# -inf, nearer than the clip plane). Aerial Gym's warp camera never sees the robot.
PROP_BAND = (slice(0, 50), (slice(0, 90), slice(230, 320)))
PROP_NEAR_M = 0.5


def prop_mask(d):
    band = np.zeros(d.shape, bool)
    for cols in PROP_BAND[1]:
        band[PROP_BAND[0], cols] = True
    return band & ((d < PROP_NEAR_M) | np.isneginf(d))


def bag_frames(bag_dir):
    mcap_path = next(Path(bag_dir).glob("*.mcap"))
    with open(mcap_path, "rb") as f:
        for _, _, msg in make_reader(f).iter_messages():
            m = deserialize_cdr(msg.data, "sensor_msgs/msg/Image")
            if m.encoding == "32FC1":
                d = np.frombuffer(m.data.tobytes(), np.float32).reshape(m.height, m.width)
            else:  # 16UC1 millimetres (RealSense driver)
                d = np.frombuffer(m.data.tobytes(), np.uint16).reshape(m.height, m.width) / 1000.0
            yield msg.log_time, d


def masked(frames):
    for t, d in frames:
        d = d.copy()
        d[prop_mask(d)] = np.inf
        yield t, d


def train_frames(n, seed=0):
    paths = sorted(glob.glob(f"{TRAIN_DIR}/*.png"))
    for p in random.Random(seed).sample(paths, n):
        yield 0, cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0 * 10.0


@torch.no_grad()
def process(frames, vae, cfg, device, batch=128):
    out = {k: [] for k in ("t", "mu", "x", "target", "recon", "ahead_raw", "ahead_target", "ahead_recon")}

    def flush(ts, xs):
        x = torch.from_numpy(np.concatenate(xs)).to(device)
        mu = vae.encode(x)
        recon = vae.decode(mu).clamp(0, 1)
        depth_m = (1.0 - x) * MAX_D
        target = 1.0 - collision_target(depth_m, cfg).clamp(cfg.min_depth_m, MAX_D) / MAX_D
        for name, img in (("raw", x), ("target", target), ("recon", recon)):
            out[f"ahead_{name}"].append(((1.0 - img[:, 0, AHEAD[0], AHEAD[1]].amax((1, 2))) * MAX_D).cpu())
        out["t"] += ts
        out["mu"].append(mu.cpu())
        for k, v in (("x", x), ("target", target), ("recon", recon)):
            out[k].append((v[:, 0] * 255).byte().cpu())

    ts, xs = [], []
    for t, d in frames:
        ts.append(t)
        xs.append(preprocess_depth(d))
        if len(xs) == batch:
            flush(ts, xs)
            ts, xs = [], []
    if xs:
        flush(ts, xs)
    res = {k: torch.cat(v).numpy() for k, v in out.items() if k != "t"}
    res["t"] = (np.array(out["t"], dtype=np.float64) - out["t"][0]) / 1e9
    return res


def mahalanobis(ref, z):
    mu, cov_inv = ref.mean(0), np.linalg.inv(np.cov(ref.T) + 1e-6 * np.eye(ref.shape[1]))
    d = z - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv, d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir", type=Path)
    ap.add_argument("--train-samples", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = VAEConfig()
    vae = DepthVAE(cfg.latent_dim).to(args.device).eval()
    vae.load_state_dict(torch.load(CHECKPOINT, map_location=args.device)["model_state_dict"])

    bag = process(bag_frames(args.bag_dir), vae, cfg, args.device)
    bag_m = process(masked(bag_frames(args.bag_dir)), vae, cfg, args.device)
    bag["prop_px"] = np.array([prop_mask(d).sum() for _, d in bag_frames(args.bag_dir)])
    bag["mu_props_masked"] = bag_m["mu"]
    bag["ahead_recon_props_masked"] = bag_m["ahead_recon"]
    bag["ahead_target_props_masked"] = bag_m["ahead_target"]
    train = process(train_frames(args.train_samples), vae, cfg, args.device)
    for r in (bag, train, bag_m):
        r["miss"] = (r["ahead_target"] < MISS_NEAR_M) & (r["ahead_recon"] > r["ahead_target"] + MISS_GAP_M)
    bag["maha"] = mahalanobis(train["mu"], bag["mu"])
    train["maha"] = mahalanobis(train["mu"], train["mu"])
    bag_m["maha"] = mahalanobis(train["mu"], bag_m["mu"])
    bag["latent_shift"] = (np.abs(bag["mu"] - bag_m["mu"]) / train["mu"].std(0)).mean(1)

    np.savez_compressed(
        args.bag_dir / "vae_latents.npz", checkpoint=CHECKPOINT,
        **{f"bag_{k}": v for k, v in bag.items() if k not in ("x", "target", "recon")},
        **{f"train_{k}": v for k, v in train.items() if k not in ("x", "target", "recon")})

    tb = SummaryWriter(str(args.bag_dir / "vae_tb"))
    for i in range(len(bag["t"])):
        tb.add_scalar("ahead_m/raw", bag["ahead_raw"][i], i)
        tb.add_scalar("ahead_m/collision_target", bag["ahead_target"][i], i)
        tb.add_scalar("ahead_m/vae_recon", bag["ahead_recon"][i], i)
        tb.add_scalar("ahead_m/recon_minus_target", bag["ahead_recon"][i] - bag["ahead_target"][i], i)
        tb.add_scalar("ahead_m/vae_recon_props_masked", bag["ahead_recon_props_masked"][i], i)
        tb.add_scalar("ahead_m/collision_target_props_masked", bag["ahead_target_props_masked"][i], i)
        tb.add_scalar("props/pixels_in_view", bag["prop_px"][i], i)
        tb.add_scalar("props/latent_shift_train_std", bag["latent_shift"][i], i)
        tb.add_scalar("ood/mahalanobis", bag["maha"][i], i)
        tb.add_scalar("ood/mahalanobis_props_masked", bag_m["maha"][i], i)
        tb.add_scalar("ood/bag_time_s", bag["t"][i], i)
        for d in range(bag["mu"].shape[1]):
            tb.add_scalar(f"latent/d{d:02d}", bag["mu"][i, d], i)
        # Input | collision target | decoded latent. Bright = near.
        panel = np.concatenate([bag["x"][i], bag["target"][i], bag["recon"][i]], axis=1)
        tb.add_image("frames/input|target|recon", panel[None], i)
        panel = np.concatenate([bag_m["x"][i], bag_m["target"][i], bag_m["recon"][i]], axis=1)
        tb.add_image("frames_props_masked/input|target|recon", panel[None], i)

    misses = np.flatnonzero(bag["miss"])
    lines = [f"| step | t [s] | target ahead [m] | recon ahead [m] |", "|---|---|---|---|"]
    lines += [f"| {i} | {bag['t'][i]:.2f} | {bag['ahead_target'][i]:.2f} | {bag['ahead_recon'][i]:.2f} |"
              for i in misses]
    tb.add_text("misses", f"{len(misses)} / {len(bag['t'])} bag frames (train rate "
                f"{train['miss'].mean() * 100:.1f}%)\n\n" + "\n".join(lines), 0)

    thumbs = torch.from_numpy(np.concatenate([bag["x"], bag_m["x"], train["x"]])[:, None]).float() / 255.0
    thumbs = F.interpolate(thumbs, size=(36, 64), mode="area").repeat(1, 3, 1, 1)
    meta = [[ "bag", str(i), f"{bag['t'][i]:.2f}", f"{bag['ahead_target'][i]:.1f}",
              f"{bag['ahead_recon'][i]:.1f}", str(int(bag["miss"][i])), f"{bag['maha'][i]:.1f}"]
            for i in range(len(bag["t"]))]
    meta += [["bag_props_masked", str(i), f"{bag['t'][i]:.2f}", f"{bag_m['ahead_target'][i]:.1f}",
              f"{bag_m['ahead_recon'][i]:.1f}", str(int(bag_m["miss"][i])), f"{bag_m['maha'][i]:.1f}"]
             for i in range(len(bag["t"]))]
    meta += [["train", str(i), "", f"{train['ahead_target'][i]:.1f}", f"{train['ahead_recon'][i]:.1f}",
              str(int(train["miss"][i])), f"{train['maha'][i]:.1f}"] for i in range(len(train["mu"]))]
    tb.add_embedding(torch.from_numpy(np.concatenate([bag["mu"], bag_m["mu"], train["mu"]])), metadata=meta,
                     metadata_header=["source", "idx", "t_s", "ahead_target_m", "ahead_recon_m", "miss", "maha"],
                     label_img=thumbs, tag="vae_latent")
    tb.close()

    print(f"bag:   {len(bag['t'])} frames, misses {bag['miss'].sum()} ({bag['miss'].mean() * 100:.1f}%)")
    print(f"train: {len(train['mu'])} frames, misses {train['miss'].sum()} ({train['miss'].mean() * 100:.1f}%)")
    w = bag["prop_px"] > 0
    print(f"props in view: {w.mean() * 100:.0f}% of frames; recon ahead median "
          f"{np.median(bag['ahead_recon'][w]):.2f} m -> {np.median(bag_m['ahead_recon'][w]):.2f} m masked")
    for name, r in (("bag", bag), ("bag_props_masked", bag_m), ("train", train)):
        near = r["ahead_target"] < MISS_NEAR_M
        print(f"{name}: frames with obstacle ahead <{MISS_NEAR_M} m: {near.sum()}, "
              f"miss rate among them {r['miss'][near].mean() * 100 if near.any() else float('nan'):.1f}%")


if __name__ == "__main__":
    main()
