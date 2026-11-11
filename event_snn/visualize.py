"""Render a held-out flight as video: events | target 1/Z | predicted 1/Z | |log error| | cell weights.

    python -m event_snn.visualize runs/event_snn/<run>/best.pt --env 3 --seconds 10
    python -m event_snn.visualize <ckpt> --env 3 --blackout 5.0 0.5    # events zeroed 5.0-5.5 s

Inverse depth shares one colour scale (bright = near, 0 at --rho_max^-1 m and closer is
saturated); sky / beyond the far plane is 1/Z = 0. Mask panel: white = event this step,
grey = event in the last 100 ms, black = none.
"""
import argparse
import os

import cv2
import numpy as np
import torch

from .config import heldout_path, steps
from .data import REGIMES, log_rho, regime_labels
from .evaluate import forward
from .model import load_checkpoint

PANEL = 256


def _up(img):
    return cv2.resize(img, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)


def _cmap(x, vmax, cmap):
    u8 = (np.clip(x / vmax, 0.0, 1.0) * 255).astype(np.uint8)
    return _up(cv2.applyColorMap(u8, cmap))


def events_panel(ev):
    on, off = ev[0].astype(np.float32), ev[1].astype(np.float32)
    img = np.full(on.shape + (3,), 25, np.float32)
    img[..., 2] += np.clip(on / 4.0, 0, 1) * 230   # ON red (BGR)
    img[..., 0] += np.clip(off / 4.0, 0, 1) * 230  # OFF blue
    return _up(np.clip(img, 0, 255).astype(np.uint8))


def mask_panel(now, recent):
    m = np.zeros(now.shape, np.uint8)
    m[recent] = 110
    m[now] = 255
    return _up(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))


def label(img, text):
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--heldout", default=None)
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--start", type=float, default=0.0, help="first second shown (model still runs from 0)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--blackout", type=float, nargs=2, default=None, metavar=("START_S", "DUR_S"))
    p.add_argument("--rho_max", type=float, default=1.0, help="1/Z at full brightness [1/m]")
    p.add_argument("--slowdown", type=float, default=2.0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    model, cfg, _ = load_checkpoint(a.checkpoint, a.device)
    H = torch.load(a.heldout or heldout_path(cfg))
    T0, T1 = steps(a.start), min(steps(a.start + a.seconds), H["done"].shape[0])
    H = {k: v[:T1, a.env:a.env + 1] for k, v in H.items() if torch.is_tensor(v)}
    events = H["events"].clone()
    if a.blackout:
        b0 = steps(a.blackout[0])
        events[b0:b0 + steps(a.blackout[1])] = 0
    out = forward(model, H, a.device or cfg.device, events=events)

    eps = cfg.data.inv_depth_eps
    rho = H["rho"].float()[:, 0]
    rho_hat = (out["depth"][:, 0].exp() - eps).clamp(min=0.0)
    err = (out["depth"][:, 0] - log_rho(rho, eps)).abs()
    lab = regime_labels(H["vel"][:, 0], H["yaw_rate"][:, 0].float(), rho)

    path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "eval",
                                 f"depth_env{a.env}" + ("_blackout" if a.blackout else "") + ".mp4")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fps = 100.0 / a.slowdown
    writer = None
    for t in range(T0, T1):
        blk = a.blackout and steps(a.blackout[0]) <= t < steps(a.blackout[0]) + steps(a.blackout[1])
        panels = [
            label(events_panel(events[t, 0].numpy()), "events" + (" (BLACKOUT)" if blk else "")),
            label(_cmap(rho[t].numpy(), a.rho_max, cv2.COLORMAP_VIRIDIS), "target 1/Z"),
            label(_cmap(rho_hat[t].numpy(), a.rho_max, cv2.COLORMAP_VIRIDIS), "predicted 1/Z"),
            label(_cmap(err[t].numpy(), 2.0, cv2.COLORMAP_INFERNO), "|log err| (0-2)"),
            label(mask_panel(H["m_now"][t, 0].numpy(), H["m_recent"][t, 0].numpy()), "event cells"),
        ]
        frame = np.concatenate(panels, axis=1)
        v, vh = H["vel"][t, 0].tolist(), out["vel"][t, 0].tolist()
        bar = np.zeros((28, frame.shape[1], 3), np.uint8)
        cv2.putText(bar, f"t {t * 0.01:5.2f}s  age {int(H['age'][t, 0]) * 0.01:5.2f}s  "
                    f"{REGIMES[int(lab[t])]:8s}  v [{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]  "
                    f"v_hat [{vh[0]:+.2f} {vh[1]:+.2f} {vh[2]:+.2f}] m/s",
                    (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.concatenate([frame, bar], axis=0)
        if writer is None:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    writer.release()
    print(f"wrote {path}  ({T1 - T0} steps at {fps:g} fps = {a.slowdown:g}x slow motion)")


if __name__ == "__main__":
    main()
