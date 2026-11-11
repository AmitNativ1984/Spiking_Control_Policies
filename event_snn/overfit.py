"""Offline tests on the held-out flights (no Isaac Gym): overfit one clip, or generalise across drones.

Overfit (default): train on ONE fixed clip, over and over, from a cold state each iteration.
    python -m event_snn.overfit --variant mb_ttc_ann --wandb
    python -m event_snn.overfit --variant mb_ttc_ann --no_events --wandb     # control: inertial only
    python -m event_snn.overfit --variant mb_ttc_ann --no_inertial --wandb   # control: events only
If the map error cannot get far below the per-pixel reference on data it sees every
iteration, the pipeline (architecture, decoder, target alignment) is the problem.

Generalise (--generalize): train on random windows of held-out drones 0..train_envs-1 and
validate on the remaining drones' whole flights, which the network never sees.
    python -m event_snn.overfit --generalize --variant mb_ttc_ann --wandb
    python -m event_snn.overfit --generalize --variant mb_ttc_ann --no_inertial --wandb
    python -m event_snn.overfit --generalize --variant grid_ttc_ann --train_data runs/event_snn/train_s1000.pt --wandb
    python -m event_snn.overfit --generalize --variant grid_ttc_ann --wandb     # no bottleneck: 8x8 grid
Memorising (drone identity, time since start) does not help here, so the validation error
below the per-pixel reference is what the inputs genuinely predict.

References (lower is better, same loss as training):
  constant   one value everywhere (median log target)
  per-pixel  the median log target of each cell: a fixed image, e.g. "ground below"
In --generalize they come from the TRAINING drones and are scored on the validation drones.

--no_events   zeroes the event input.
--no_inertial replaces the 9 inertial numbers [acc, gyro, grav] by their mean (no information).
"""
import argparse
import os
import time

import cv2
import torch

from .config import heldout_path, load_config, steps, to_dict
from .data import select_target, state_input
from .loss import depth_velocity_loss, ttc_loss, valid_mask
from .baseline import StackedEventCNN
from .model import EventSNN, count_parameters
from .visualize import depth_image, pick_frames

KEYS = ("events", "imu", "grav", "rho", "ttc", "vel", "done")


def ablate(b, no_events, no_inertial, inertial_mean=None):
    if no_events:
        b["events"] = torch.zeros_like(b["events"])
    if no_inertial:
        m = inertial_mean if inertial_mean is not None else {k: b[k].float().mean((0, 1)) for k in ("imu", "grav")}
        for k in ("imu", "grav"):
            b[k] = m[k].to(device=b[k].device, dtype=b[k].dtype).expand_as(b[k]).clone()
    return b


def median_maps(b, valid, eps, max_frames=20000):
    """Median log target (scalar) and per cell (H, W) over valid frames (a random subset if many)."""
    idx = valid.bool().nonzero()
    if len(idx) > max_frames:
        idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))[:max_frames]]
    y = torch.log(b["ttc"][idx[:, 0], idx[:, 1]].float() + eps)         # (N, H, W)
    return y.median(), y.median(dim=0).values


def ref_losses(b, valid, const, pix, eps, ttc_max):
    out = {}
    for name, val in (("constant", torch.full_like(b["ttc"], float(const))), ("per_pixel", pix.expand_as(b["ttc"]))):
        out[name] = float((ttc_loss(val, b["ttc"], eps, ttc_max) * valid).sum() / valid.sum().clamp(min=1.0))
    return out


def run(model, b, cfg, burn):
    ex = {}
    d, v, _, rates = model.unroll(b["events"], state_input(b), b["done"], model.init_state(b["done"].shape[1]),
                                  extras=ex)
    loss, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, burn,
                                     ex.get("ttc"), cfg.data.ttc_eps, cfg.data.ttc_max)
    return loss, logs, rates, d, ex["ttc"]


def maps_image(b, d, pred, cfg, valid=None, n=6):
    """Validation: n frames spanning open space to imminent contact (pick_frames).
    Without a mask (overfit clip): the last step of the first 4 envs."""
    frames = pick_frames(b["ttc"], valid, n) if valid is not None else None
    return depth_image(b, d.detach(), cfg.data.inv_depth_eps, ttc=pred.detach(), ttc_eps=cfg.data.ttc_eps,
                       depth=False, ttc_name="speed/range" if cfg.data.target_map == "near" else "invTTC",
                       frames=frames)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", default="mb_ttc_ann")
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. model.decoder=spiking")
    p.add_argument("--generalize", action="store_true", help="train on some drones, validate on the others")
    p.add_argument("--envs", type=int, default=4, help="overfit: drones in the clip")
    p.add_argument("--train_envs", type=int, default=8, help="generalize: drones 0..n-1 train, the rest validate")
    p.add_argument("--batch_envs", type=int, default=8, help="generalize: drones (random windows) per iteration")
    p.add_argument("--train_data", nargs="+", default=None, metavar="PATH",
                   help="generalize: train on these recorded sets (event_snn.sim --record) instead of held-out "
                        "drones 0..train_envs-1; validation stays on held-out drones train_envs..")
    p.add_argument("--seconds", type=float, default=1.0, help="clip (overfit) or window (generalize) length")
    p.add_argument("--start_s", type=float, default=2.0, help="overfit: clip start inside the flights")
    p.add_argument("--iterations", type=int, default=None, help="default 1500 overfit, 3000 generalize")
    p.add_argument("--val_every", type=int, default=100, help="generalize")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no_events", action="store_true")
    p.add_argument("--no_inertial", action="store_true")
    p.add_argument("--baseline", action="store_true",
                   help="non-spiking reference instead of the SNN: CNN on the last --k event frames + inertial history")
    p.add_argument("--k", type=int, default=30, help="baseline: event frames stacked (10 ms each)")
    p.add_argument("--width", type=int, default=64, help="baseline: channels of the first CNN stage")
    p.add_argument("--baseline_events_only", action="store_true",
                   help="baseline: events only (default: also the last --k steps of acc, gyro and gravity)")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", action="store_true", help="log to W&B (project event_snn)")
    a = p.parse_args()
    iters = a.iterations or (3000 if a.generalize else 1500)

    cfg = load_config(a.config, a.variant, a.set)
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    T, burn = steps(a.seconds), steps(cfg.loss.burn_in_s)
    if a.baseline:
        # the CNN needs k-1 frames of history: windows start k-1 steps earlier and, after any
        # reset or cold start, steps without a full history are not scored (validation and references too)
        burn = max(burn, a.k - 1)
        T += a.k - 1
    eps, tmax = cfg.data.ttc_eps, cfg.data.ttc_max
    H = select_target(torch.load(heldout_path(cfg)), cfg.data.target_map)
    H = {k: H[k] for k in KEYS if k in H}
    H["rho"], H["ttc"] = H["rho"].float(), H["ttc"].float()
    Ttot, Eall = H["done"].shape

    if a.generalize:
        if not 0 < a.train_envs < Eall:
            raise SystemExit(f"--train_envs must be in 1..{Eall - 1}")
        va = {k: x[:, a.train_envs:].to(dev) for k, x in H.items()}
        if a.train_data:
            sets = [select_target(torch.load(f), cfg.data.target_map) for f in a.train_data]
            Tmin = min(d["done"].shape[0] for d in sets)
            tr = {k: torch.cat([d[k][:Tmin] for d in sets], dim=1) for k in KEYS if k != "rho" and k in sets[0]}
            print(f"training set: {', '.join(a.train_data)} -> {tr['done'].shape[1]} drones x {Tmin / 100:.0f} s "
                  f"({tr['done'].shape[1] * Tmin / 100 / 60:.0f} min of flight)")
        else:
            tr = {k: x[:, :a.train_envs] for k, x in H.items()}
        Ttot = tr["done"].shape[0]
        src = tr
    else:
        t0 = steps(a.start_s)
        src = tr = {k: x[t0:t0 + T, :a.envs].to(dev) for k, x in H.items()}
    mean = {k: src[k].float().mean((0, 1)) for k in ("imu", "grav")}
    x = state_input({k: src[k].to(dev) for k in ("imu", "grav")}).flatten(0, 1)
    lo, hi = torch.quantile(x, 0.01, dim=0).cpu(), torch.quantile(x, 0.99, dim=0).cpu()
    # references from the training data (all its steps, cold-start burn-in excluded)
    tr_valid = valid_mask(tr["done"], burn)
    const, pix = median_maps({"ttc": tr["ttc"]}, tr_valid, eps)
    if a.generalize:
        va = ablate(va, a.no_events, a.no_inertial, mean)
        va_valid = valid_mask(va["done"], burn).float()
        ref = ref_losses(va, va_valid, const.to(dev), pix.to(dev), eps, tmax)
        # fixed training probe, scored and drawn like validation: is the net learning the events on data it
        # trains on? (whole flights of the first training drones, as long as the validation flights)
        Tp, Ep = min(tr["done"].shape[0], va["done"].shape[0]), min(Eall - a.train_envs, tr["done"].shape[1])
        pr = {k: x[:Tp, :Ep].to(dev) for k, x in tr.items()}
        pr["ttc"] = pr["ttc"].float()
        if "rho" not in pr:
            pr["rho"] = torch.zeros_like(pr["ttc"])
        pr = ablate(pr, a.no_events, a.no_inertial, mean)
        pr_valid = valid_mask(pr["done"], burn).float()
    else:
        tr = ablate(tr, a.no_events, a.no_inertial, mean)
        ref = ref_losses(tr, tr_valid.float(), const, pix, eps, tmax)

    model = (StackedEventCNN(a.k, a.width, imu=not a.baseline_events_only) if a.baseline else EventSNN(cfg.model)).to(dev)
    model.set_input_bounds(lo, hi)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    mode = "generalize" if a.generalize else "overfit"
    tag = ("_noev" if a.no_events else "") + ("_noimu" if a.no_inertial else "")
    net = f"cnn_k{a.k}{'_imu' if not a.baseline_events_only else ''}_{cfg.data.target_map}" if a.baseline else cfg.variant
    name = f"{mode}_{net}{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = a.out or os.path.join(cfg.out_dir, mode, name)
    os.makedirs(out, exist_ok=True)
    inputs = "events only" if a.no_inertial and not a.no_events else "inertial only" if a.no_events and not a.no_inertial \
        else "NO inputs" if a.no_events else "events + inertial"
    if a.baseline:
        inputs = (f"last {a.k} event frames" + (" + acc, gyro, gravity" if not a.baseline_events_only else " only, no IMU")
                  + " (non-spiking CNN baseline)")
        if a.no_events or (not a.baseline_events_only and a.no_inertial):
            inputs += f"; ablated: {'events ' if a.no_events else ''}{'inertial' if a.no_inertial else ''}"
    if a.generalize:
        src_txt = (f"recorded {', '.join(os.path.basename(f) for f in a.train_data)} ({tr['done'].shape[1]} drones"
                   if a.train_data else f"drones 0-{a.train_envs - 1} ({a.train_envs} drones")
        what = (f"train {src_txt}, {Ttot / 100:.0f} s each, {min(a.batch_envs, tr['done'].shape[1])} random "
                f"{a.seconds} s windows per iteration), "
                f"validate drones {a.train_envs}-{Eall - 1} whole flights")
    else:
        what = f"one fixed clip, {a.envs} drones x {a.seconds} s from {a.start_s} s"
    print(f"{mode} {net}: {count_parameters(model)} parameters; {what}; inputs: {inputs}; lr {a.lr}")
    print(f"references{' on the validation drones' if a.generalize else ''}: "
          f"constant {ref['constant']:.4f}  per-pixel {ref['per_pixel']:.4f}")
    print(f"writing images to {out}")
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project="event_snn", group=mode, name=name, dir=out,
                        config={**to_dict(cfg), mode: vars(a)},
                        notes=os.environ.get("WANDB_NOTES") or f"{mode} test: {net}, {what}; inputs: {inputs}; "
                        f"lr {a.lr}; references constant {ref['constant']:.3f}, per-pixel {ref['per_pixel']:.3f}")
        wb.summary.update({"ref/constant": ref["constant"], "ref/per_pixel": ref["per_pixel"]})

    def log(d, it, img=None, train_img=None):
        imgs = {k: v for k, v in (("maps", img), ("train_maps", train_img)) if v is not None}
        if wb:
            d = {**d, "ref/constant": ref["constant"], "ref/per_pixel": ref["per_pixel"]}
            d.update({f"{mode}/{k}": wandb.Image(v) for k, v in imgs.items()})
            wb.log(d, step=it)
        for k, v in imgs.items():
            cv2.imwrite(os.path.join(out, f"{k}_{it:05d}.png"), cv2.cvtColor(v, cv2.COLOR_RGB2BGR))

    best, t_start = float("inf"), time.time()
    ar = torch.arange(T)
    for it in range(1, iters + 1):
        if a.generalize:
            Etr = tr["done"].shape[1]
            E = min(a.batch_envs, Etr)
            starts = torch.randint(0, Ttot - T + 1, (E,))
            ti, ei = starts[None, :] + ar[:, None], torch.randperm(Etr)[:E][None, :]
            b = {k: x[ti, ei].to(dev) for k, x in tr.items()}
            b["ttc"] = b["ttc"].float()
            if "rho" not in b:                                       # recorded sets carry no depth
                b["rho"] = torch.zeros_like(b["ttc"])
            b = ablate(b, a.no_events, a.no_inertial, mean)
        else:
            b = tr
        loss, logs, rates, d, pred = run(model, b, cfg, burn)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        pg = {n: float(q.grad.norm()) for n, q in model.named_parameters() if q.grad is not None} \
            if (it % 5 == 0 or it == 1) else None
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip))
        opt.step()
        train_ttc = float(logs["ttc"])
        if not a.generalize:
            best = min(best, train_ttc)
        scal = {f"{mode}/train_ttc": train_ttc, f"{mode}/rate_reg": float(logs["rate_reg"]),
                f"{mode}/grad_norm": gn, **{f"{mode}/rate/{k}": float(r) for k, r in rates.items()}}
        if pg:  # pre-clip gradient norm of every parameter: which one dominates the total
            scal.update({f"grad/{n}": g for n, g in pg.items()})
        img = train_img = None
        if a.generalize and (it % a.val_every == 0 or it == iters):
            model.eval()
            with torch.no_grad():
                _, vlogs, vrates, vd, vpred = run(model, va, cfg, burn)
                _, plogs, _, pd, ppred = run(model, pr, cfg, burn)
            model.train()
            pv = float(plogs["ttc"])
            scal[f"{mode}/train_probe_ttc"] = pv
            train_img = maps_image(pr, pd, ppred, cfg, pr_valid)
            v = float(vlogs["ttc"])
            best = min(best, v)
            scal.update({f"{mode}/val_ttc": v, f"{mode}/best": best,
                         **{f"{mode}/val_rate/{k}": float(r) for k, r in vrates.items()}})
            img = maps_image(va, vd, vpred, cfg, va_valid)
            print(f"  val ttc {v:.4f} (best {best:.4f})  train probe {pv:.4f}  vs per-pixel {ref['per_pixel']:.4f}, "
                  f"constant {ref['constant']:.4f}", flush=True)
        elif not a.generalize:
            scal[f"{mode}/best"] = best
            if it % 250 == 0 or it == iters:
                img = maps_image(b, d, pred, cfg)
        if it % 5 == 0 or it == 1 or img is not None:
            log(scal, it, img, train_img)
        if it % 25 == 0 or it == 1:
            r = "  ".join(f"{k} {float(x):.3f}" for k, x in rates.items())
            print(f"it {it:5d}  train ttc {train_ttc:.4f}  grad {gn:8.2f}  rates {r}  {time.time() - t_start:.0f}s",
                  flush=True)
            top = sorted(pg.items(), key=lambda kv: -kv[1])[:3]
            print("           top grads: " + "  ".join(f"{n} {g:.3g} ({100 * g * g / max(gn * gn, 1e-30):.0f}%)"
                                               for n, g in top), flush=True)

    pp = ref["per_pixel"]
    if a.generalize:
        verdict = ("PASS: clearly better than a fixed image on unseen drones" if best < 0.75 * pp else
                   "WEAK: slightly better than a fixed image on unseen drones" if best < 0.95 * pp else
                   "FAIL: no better than a fixed image on unseen drones")
    else:
        verdict = ("PASS: fits far below the per-pixel reference" if best < 0.5 * pp else
                   "PARTIAL: beats the per-pixel reference but does not fit" if best < 0.9 * pp else
                   "FAIL: no better than a fixed image on data it sees every iteration")
    if wb:
        wb.summary.update({"verdict": verdict.split(":")[0], "best_ttc": best})
        wb.finish()
    print(f"\nbest {'val' if a.generalize else 'train'} ttc {best:.4f}  vs constant {ref['constant']:.4f}, "
          f"per-pixel {pp:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
