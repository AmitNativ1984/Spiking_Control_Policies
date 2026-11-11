"""Overfit test: train on ONE fixed clip of the held-out flights, over and over (no Isaac Gym).

    python -m event_snn.overfit --variant mb_ttc_ann
    python -m event_snn.overfit --variant mb_ttc_ann --no_events          # control: inertial input only
    python -m event_snn.overfit --variant mb_ttc_ann --wandb              # also log to W&B project event_snn
    python -m event_snn.overfit --variant mb_ttc_ann --set loss.lambda_s=0 model.decoder=spiking

Each iteration unrolls the same E envs x T steps from a cold state and takes one Adam step.
If the map error cannot get far below the per-pixel reference on data it sees every
iteration, the pipeline (architecture, decoder, target alignment) is the problem, not the
amount of data. Two references are printed, both computed on the clip itself:
  constant   one value everywhere (the median log target)
  per-pixel  the median log target of each cell over the clip: a fixed image, e.g. "ground below"
--no_events zeroes the event input; the gap between the two runs is what the events contribute.
"""
import argparse
import os
import time

import cv2
import torch

from .config import heldout_path, load_config, steps, to_dict
from .data import select_target, state_input
from .loss import depth_velocity_loss, ttc_loss, valid_mask
from .model import EventSNN, count_parameters
from .train import depth_image

KEYS = ("events", "imu", "grav", "rho", "ttc", "vel", "done")


def references(b, valid, eps, ttc_max):
    """Loss of the constant-map and per-pixel-map references on the clip."""
    y = torch.log(b["ttc"] + eps)                                      # (T, E, H, W)
    m = valid.bool()
    const = y[m].median()
    pix = y[m].median(dim=0).values                                    # (H, W)
    out = {}
    for name, pred in (("constant", torch.full_like(y, float(const))), ("per-pixel", pix.expand_as(y))):
        out[name] = float((ttc_loss(pred, b["ttc"], eps, ttc_max) * valid).sum() / valid.sum().clamp(min=1.0))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", default="mb_ttc_ann")
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=3e-4")
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--start_s", type=float, default=2.0, help="clip start inside the held-out flights")
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no_events", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", action="store_true", help="log to W&B (project event_snn, group overfit)")
    a = p.parse_args()

    cfg = load_config(a.config, a.variant, a.set)
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    t0, T = steps(a.start_s), steps(a.seconds)
    H = select_target(torch.load(heldout_path(cfg)), cfg.data.target_map)
    b = {k: H[k][t0:t0 + T, :a.envs].to(dev) for k in KEYS if k in H}
    b["rho"], b["ttc"] = b["rho"].float(), b["ttc"].float()
    if a.no_events:
        b["events"] = torch.zeros_like(b["events"])
    x = state_input(b).flatten(0, 1)
    lo, hi = torch.quantile(x, 0.01, dim=0).cpu(), torch.quantile(x, 0.99, dim=0).cpu()

    model = EventSNN(cfg.model).to(dev)
    model.set_input_bounds(lo, hi)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    burn = steps(cfg.loss.burn_in_s)
    valid = valid_mask(b["done"], burn).float()
    ref = references(b, valid, cfg.data.ttc_eps, cfg.data.ttc_max)
    name = f"overfit_{cfg.variant}{'_noev' if a.no_events else ''}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = a.out or os.path.join(cfg.out_dir, "overfit", name)
    os.makedirs(out, exist_ok=True)
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project="event_snn", group="overfit", name=name, dir=out,
                        config={**to_dict(cfg), "overfit": vars(a)},
                        notes=os.environ.get("WANDB_NOTES") or
                        f"overfit test: {cfg.variant} on one fixed held-out clip ({a.envs} envs x {a.seconds} s from "
                        f"{a.start_s} s), {'events ZEROED (control)' if a.no_events else 'real events'}; "
                        f"references constant {ref['constant']:.3f}, per-pixel {ref['per-pixel']:.3f}")
        wb.summary.update({"ref/constant": ref["constant"], "ref/per_pixel": ref["per-pixel"]})
    print(f"{cfg.variant}: {count_parameters(model)} parameters; clip {a.envs} envs x {T} steps from "
          f"{a.start_s:.1f} s, {int(valid.sum())} supervised maps, resets in clip {int(b['done'].sum())}"
          f"{', EVENTS ZEROED' if a.no_events else ''}")
    print(f"references on this clip: constant {ref['constant']:.4f}  per-pixel {ref['per-pixel']:.4f}")
    print(f"writing images to {out}")

    best, t_start = float("inf"), time.time()
    for it in range(1, a.iterations + 1):
        ex = {}
        d, v, _, rates = model.unroll(b["events"], state_input(b), b["done"], model.init_state(a.envs), extras=ex)
        loss, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, burn,
                                         ex.get("ttc"), cfg.data.ttc_eps, cfg.data.ttc_max)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip))
        opt.step()
        best = min(best, float(logs["ttc"]))
        if wb and (it % 5 == 0 or it == 1):
            wb.log({"overfit/ttc": float(logs["ttc"]), "overfit/best": best, "overfit/rate_reg": float(logs["rate_reg"]),
                    "overfit/grad_norm": gn, "ref/constant": ref["constant"], "ref/per_pixel": ref["per-pixel"],
                    **{f"overfit/rate/{k}": float(x) for k, x in rates.items()}}, step=it)
        if it % 25 == 0 or it == 1:
            r = "  ".join(f"{k} {float(x):.2f}" for k, x in rates.items())
            print(f"it {it:5d}  ttc {float(logs['ttc']):.4f} (best {best:.4f})  rate_reg {float(logs['rate_reg']):.4f}"
                  f"  grad {gn:8.2f}  rates {r}  {time.time() - t_start:.0f}s", flush=True)
        if it % 250 == 0 or it == a.iterations:
            img = depth_image(b, d.detach(), cfg.data.inv_depth_eps, ttc=ex["ttc"].detach(), ttc_eps=cfg.data.ttc_eps,
                              depth=False, ttc_name="speed/range" if cfg.data.target_map == "near" else "invTTC")
            cv2.imwrite(os.path.join(out, f"maps_{it:05d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if wb:
                wb.log({"overfit/maps": wandb.Image(img)}, step=it)

    f = best
    verdict = ("PASS: fits far below the per-pixel reference" if f < 0.5 * ref["per-pixel"] else
               "PARTIAL: beats the per-pixel reference but does not fit" if f < 0.9 * ref["per-pixel"] else
               "FAIL: no better than a fixed image on data it sees every iteration")
    if wb:
        wb.summary.update({"verdict": verdict.split(":")[0], "best_ttc": f})
        wb.finish()
    print(f"\nbest ttc {f:.4f}  vs constant {ref['constant']:.4f}, per-pixel {ref['per-pixel']:.4f}  ->  {verdict}")


if __name__ == "__main__":
    main()
