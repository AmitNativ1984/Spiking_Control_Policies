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



def pick_frames(ttc, valid, n=6, pct=0.98, min_gap=50):
    """(t, e, score) for n frames spanning the range of the target map, from open space to
    imminent contact. score = the pct-quantile of the frame's map (how close the nearest
    obstacle is, ignoring single cells); frames are taken at evenly spaced score quantiles
    over all valid steps, at least min_gap steps apart within one env. Depends only on the
    targets, so a fixed held-out set gives the same frames at every validation."""
    T, E = ttc.shape[:2]
    flat = ttc.float().reshape(T, E, -1)
    k = max(1, int(round(pct * flat.shape[-1])))
    score = flat.kthvalue(k, dim=-1).values.cpu()                     # (T, E)
    ok = valid.bool().cpu()
    cand = ok.nonzero().tolist()
    if not cand:
        return []
    cand.sort(key=lambda te: float(score[te[0], te[1]]))
    picked = []
    for i in range(n):
        target = round(i * (len(cand) - 1) / max(n - 1, 1))
        for off in range(len(cand)):                                     # nearest free candidate
            for j in (target - off, target + off):
                if 0 <= j < len(cand):
                    t, e = cand[j]
                    if all(e != pe or abs(t - pt) >= min_gap for pt, pe, _ in picked):
                        picked.append((t, e, float(score[t, e])))
                        break
            else:
                continue
            break
    return sorted(picked, key=lambda x: x[2])


def depth_image(b, d, eps, n_envs=4, rho_max=1.0, ttc=None, ttc_eps=0.05, ttc_show=3.0, depth=True,
                ttc_name="invTTC", frames=None):
    """(H, W, 3) uint8 RGB, one row per frame:
    events | target 1/Z | predicted 1/Z | |log err| | target invTTC | predicted invTTC | |log err|.
    frames: [(t, e, score)] from pick_frames; default the last step of the first n_envs envs.
    The depth panels are dropped when depth is not trained, the invTTC panels when there is no head.
    invTTC panels share one colour scale, 0-ttc_show 1/s (brighter = sooner contact)."""
    if frames is None:
        frames = [(d.shape[0] - 1, e, None) for e in range(min(n_envs, d.shape[1]))]
    rows = []
    for i, (t, e, score) in enumerate(frames):
        panels, names = [events_panel(b["events"][t, e].cpu().numpy())], ["events"]
        if depth:
            rho = b["rho"][t, e].float().cpu()
            pred = d[t, e].float().cpu()
            panels += [_cmap(rho.numpy(), rho_max, cv2.COLORMAP_VIRIDIS),
                       _cmap((pred.exp() - eps).clamp(min=0.0).numpy(), rho_max, cv2.COLORMAP_VIRIDIS),
                       _cmap((pred - log_rho(rho, eps)).abs().numpy(), 2.0, cv2.COLORMAP_INFERNO)]
            names += ["target 1/Z", "predicted 1/Z", "|log err| (0-2)"]
        if ttc is not None:
            y = b["ttc"][t, e].float().cpu()
            lp = ttc[t, e].float().cpu()
            panels += [_cmap(y.numpy(), ttc_show, cv2.COLORMAP_MAGMA),
                       _cmap((lp.exp() - ttc_eps).clamp(min=0.0).numpy(), ttc_show, cv2.COLORMAP_MAGMA),
                       _cmap((lp - torch.log(y + ttc_eps)).abs().numpy(), 2.0, cv2.COLORMAP_INFERNO)]
            names += [f"target {ttc_name} (0-{ttc_show:g}/s)", f"predicted {ttc_name}", "|log err| (0-2)"]
        if i == 0:
            for img, text in zip(panels, names):
                label(img, text)
        tag = f"env {e}  t {t / 100:.2f}s" + (f"  p98 {score:.2f}/s" if score is not None else "")
        cv2.putText(panels[0], tag, (6, PANEL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate(panels, axis=1))
    return cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_BGR2RGB)


def event_sequence(b, t, e, ttc=None, ttc_eps=0.05, window_s=0.3, every=2, ttc_show=3.0, ttc_name="invTTC",
                   per_row=6, panel=192, env_name=None):
    """(H, W, 3) uint8 RGB: the event frames that lead up to one target map, oldest first, then
    the target map at step t and (if ttc, the head's log output) the prediction.
    Each event panel sums `every` consecutive 10 ms frames and is labelled by its age relative to t.
    Frames before the drone's last reset (another episode, which the network never saw with
    this state) are dimmed and tagged "prev ep". window_s ~ 0.3 s covers the conv (62 ms) and
    memory (100 ms) membrane time constants a few times over."""
    n = max(1, int(round(window_s / 0.01)) // every)
    done = b["done"][:t + 1, e].bool().cpu()
    resets = done.nonzero().flatten()
    t_reset = int(resets[-1]) if len(resets) else 0          # first step of the current episode
    small = lambda img: cv2.resize(img, (panel, panel), interpolation=cv2.INTER_NEAREST)
    tiles = []
    for i in range(n - 1, -1, -1):
        t1 = t - i * every                                      # newest step in this panel
        t0 = t1 - every + 1
        if t1 < 0:
            continue
        ev = b["events"][max(t0, 0):t1 + 1, e].float().sum(0).cpu().numpy()
        img = small(events_panel(ev))
        old = t1 < t_reset
        if old:
            img = (img * 0.35).astype(np.uint8)
        txt = f"{-(t - t1) * 10:d} ms" if every == 1 else f"{-(t - t0) * 10:d}..{-(t - t1) * 10:d} ms"
        label(img, txt + ("  prev ep" if old else ""))
        n_ev = int(ev.sum())
        cv2.putText(img, f"{n_ev} ev", (6, panel - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(img)
    y = b["ttc"][t, e].float().cpu()
    tiles.append(label(small(_cmap(y.numpy(), ttc_show, cv2.COLORMAP_MAGMA)), f"target {ttc_name} t=0"))
    if ttc is not None:
        lp = ttc[t, e].float().cpu()
        tiles.append(label(small(_cmap((lp.exp() - ttc_eps).clamp(min=0.0).numpy(), ttc_show,
                                       cv2.COLORMAP_MAGMA)), "predicted"))
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + per_row], axis=1) for i in range(0, len(tiles), per_row)]
    img = np.concatenate(rows, axis=0)
    flat = y.flatten()
    bar = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(bar, f"env {e if env_name is None else env_name}  t {t / 100:.2f}s  episode age {(t - t_reset) / 100:.2f}s  "
                f"target p98 {float(flat.kthvalue(max(1, int(round(0.98 * flat.numel())))).values):.2f}/s  "
                f"(0-{ttc_show:g}/s; events summed over {every * 10} ms per panel)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(np.concatenate([bar, img], axis=0), cv2.COLOR_BGR2RGB)


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
