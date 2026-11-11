"""Train the event SNN with BPTT on chunks flown while training (docs/event_snn_experiment_plan.md §4).

Each iteration flies one BPTT window in all envs, runs `passes` passes of BPTT over that
chunk in env mini-batches, then discards it. Neuron state carries (detached) into the next
window of the same flights. The window grows through bptt_curriculum_s when the held-out
validation loss plateaus.

    python -m event_snn.train --variant a_rec
    python -m event_snn.train --variant b --set sim.num_envs=32 train.minibatch_envs=16
"""
import isaacgym  # noqa: F401  -- must precede torch

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .config import heldout_path, load_config, steps, to_dict
from .loss import depth_velocity_loss
from .model import SPIKING_LAYERS, EventSNN, Monitor, count_parameters, surrogate_half_width
from .data import log_rho, state_input
from .sim import Rollout, measure_bounds, write_heldout
from .visualize import _cmap, events_panel, label

TRAIN_KEYS = ("events", "imu", "grav", "rho", "vel", "done")


def to_device(chunk, dev, t=slice(None), e=slice(None)):
    return {k: v[t, e].to(dev, non_blocking=True) for k, v in chunk.items() if k in TRAIN_KEYS}


@torch.no_grad()
def validate(model, heldout, cfg):
    """Loss on the first val_envs x val_seconds of the held-out flights, from a cold state."""
    dev = cfg.device
    E, T = min(cfg.train.val_envs, heldout["done"].shape[1]), steps(cfg.train.val_seconds)
    b = to_device(heldout, dev, slice(0, T), slice(0, E))
    b["rho"] = b["rho"].float()
    d, v, _, rates = model.unroll(b["events"], state_input(b), b["done"], model.init_state(E))
    _, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, steps(cfg.loss.burn_in_s))
    logs.update({f"rate/{k}": r for k, r in rates.items()})
    return {k: float(x) for k, x in logs.items()}, (b, d)


def depth_image(b, d, eps, n_envs=4, rho_max=1.0):
    """(H, W, 3) uint8 RGB: one row per held-out env at the last validation step:
    events | target 1/Z | predicted 1/Z | |log error| (0-2)."""
    t = d.shape[0] - 1
    rho = b["rho"][t].cpu()
    pred = d[t].float().cpu()
    rho_hat = (pred.exp() - eps).clamp(min=0.0)
    err = (pred - log_rho(rho, eps)).abs()
    rows = []
    for e in range(min(n_envs, d.shape[1])):
        panels = [events_panel(b["events"][t, e].cpu().numpy()),
                  _cmap(rho[e].numpy(), rho_max, cv2.COLORMAP_VIRIDIS),
                  _cmap(rho_hat[e].numpy(), rho_max, cv2.COLORMAP_VIRIDIS),
                  _cmap(err[e].numpy(), 2.0, cv2.COLORMAP_INFERNO)]
        if e == 0:
            for img, text in zip(panels, ("events", "target 1/Z", "predicted 1/Z", "|log err| (0-2)")):
                label(img, text)
        rows.append(np.concatenate(panels, axis=1))
    return cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_BGR2RGB)


class WindowCurriculum:
    def __init__(self, cfg):
        self.windows = [steps(s) for s in cfg.bptt_curriculum_s]
        self.stage, self.best, self.bad = 0, math.inf, 0
        self.patience, self.rel = cfg.plateau_patience, cfg.plateau_rel

    @property
    def T(self):
        return self.windows[self.stage]

    def update(self, val_loss):
        if val_loss < self.best * (1.0 - self.rel):
            self.best, self.bad = val_loss, 0
            return False
        self.bad += 1
        if self.bad >= self.patience and self.stage + 1 < len(self.windows):
            self.stage, self.bad, self.best = self.stage + 1, 0, math.inf
            return True
        return False


def train(cfg):
    torch.manual_seed(cfg.seed)
    dev = cfg.device
    run_name = cfg.run_name or f"{cfg.variant}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)

    rollout = Rollout(cfg)
    hp = heldout_path(cfg)
    if not os.path.exists(hp):
        if cfg.data.heldout_envs > rollout.n:
            raise SystemExit(f"held-out set needs {cfg.data.heldout_envs} envs; run "
                             "`python -m event_snn.sim --heldout` first")
        write_heldout(cfg, rollout)
    heldout = torch.load(hp)
    print(f"held-out set {hp}: {tuple(heldout['done'].shape)} (steps, envs)")

    lo, hi = measure_bounds(rollout, cfg.data.bounds_steps, cfg.data.bounds_percentiles)
    model = EventSNN(cfg.model).to(dev)
    model.set_input_bounds(lo, hi)
    print(f"{cfg.variant}: {count_parameters(model)} parameters; state bounds lo {lo.tolist()} hi {hi.tolist()}")

    tc = cfg.train
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (
        tc.lr_min + 0.5 * (tc.lr - tc.lr_min) * (1 + math.cos(math.pi * min(i / tc.iterations, 1.0)))) / tc.lr)
    curriculum = WindowCurriculum(tc)
    writer = SummaryWriter(os.path.join(run_dir, "tb"), flush_secs=10)
    wb = None
    if tc.wandb:
        import wandb
        wb = wandb.init(project="event_snn", name=run_name, config=to_dict(cfg), dir=run_dir)
    monitor_log = open(os.path.join(run_dir, "monitor.jsonl"), "a")

    N, mb = rollout.n, min(tc.minibatch_envs, rollout.n)
    burn = steps(cfg.loss.burn_in_s)
    half_width = surrogate_half_width(cfg.model.surrogate_alpha)
    state = model.init_state(N)
    best_val = math.inf

    for it in range(1, tc.iterations + 1):
        T = curriculum.T
        t0 = time.time()
        chunk = rollout.collect(T, keys=TRAIN_KEYS)
        t_collect = time.time() - t0

        s0 = model.detach_state(state)
        new_state = {k: v.clone() for k, v in s0.items()}
        agg, n_mb = {}, 0
        mon = None
        for p in range(tc.passes):
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, mb):
                ids = perm[i:i + mb]
                b = {k: v[:, ids] for k, v in chunk.items()}
                st = {k: v[ids] for k, v in s0.items()}
                record = None
                if p == 0 and i == 0 and it % tc.monitor_every == 0:
                    mon = record = Monitor(half_width)
                d, v, st_end, rates = model.unroll(b["events"], state_input(b), b["done"], st,
                                                   grad_decay=tc.bptt_grad_decay, record=record)
                loss, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, burn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                opt.step()
                if p == tc.passes - 1:
                    for k, x in st_end.items():
                        new_state[k][ids] = x.detach()
                logs["grad_norm"] = gnorm.detach()
                for k, r in rates.items():
                    logs[f"rate/{k}"] = r.detach()
                for k, x in logs.items():
                    agg[k] = agg.get(k, 0.0) + x
                n_mb += 1
        state = new_state
        sched.step()
        t_train = time.time() - t0 - t_collect

        if it % tc.log_every == 0 or it == 1:
            scal = {f"train/{k}": float(x) / n_mb for k, x in agg.items()}
            scal.update({"train/lr": sched.get_last_lr()[0], "train/window_steps": T,
                         "time/collect_s": t_collect, "time/train_s": t_train})
            for k, x in scal.items():
                writer.add_scalar(k, x, it)
            if wb:
                wb.log(scal, step=it)
            print(f"it {it:5d}  T {T:3d}  loss {scal['train/loss']:.4f}  depth {scal['train/depth']:.4f}  "
                  f"vel {scal['train/vel']:.4f}  mem rate {scal['train/rate/mem']:.3f}  "
                  f"collect {t_collect:.1f}s train {t_train:.1f}s", flush=True)

        if mon is not None:
            rec = {"it": it, "T": T, "grad": mon.grad, "near": mon.near}
            monitor_log.write(json.dumps(rec) + "\n")
            monitor_log.flush()
            for k in SPIKING_LAYERS:
                g = mon.grad[k]
                writer.add_scalar(f"monitor/near_threshold/{k}", sum(mon.near[k]) / len(mon.near[k]), it)
                if g and g[-1] > 0:
                    writer.add_scalar(f"monitor/grad_first_over_last/{k}", g[0] / g[-1], it)
                    writer.add_scalar(f"monitor/grad_last/{k}", g[-1], it)

        if it % tc.val_every == 0:
            model.eval()
            val, (vb, vd) = validate(model, heldout, cfg)
            writer.add_image("val/depth", depth_image(vb, vd, cfg.data.inv_depth_eps), it, dataformats="HWC")
            model.train()
            for k, x in val.items():
                writer.add_scalar(f"val/{k}", x, it)
            if wb:
                wb.log({f"val/{k}": x for k, x in val.items()}, step=it)
            grew = curriculum.update(val["loss"])
            print(f"  val loss {val['loss']:.4f} depth {val['depth']:.4f} vel {val['vel']:.4f}"
                  + (f"  -> BPTT window {curriculum.T} steps" if grew else ""), flush=True)
            if val["loss"] < best_val:
                best_val = val["loss"]
                save(os.path.join(run_dir, "best.pt"), model, cfg, it, curriculum, lo, hi, val)

        if it % tc.ckpt_every == 0 or it == tc.iterations:
            save(os.path.join(run_dir, f"ckpt_{it:06d}.pt"), model, cfg, it, curriculum, lo, hi)
            save(os.path.join(run_dir, "last.pt"), model, cfg, it, curriculum, lo, hi)

    monitor_log.close()
    writer.close()
    print(f"done: {run_dir}")
    return run_dir


def save(path, model, cfg, it, curriculum, lo, hi, val=None):
    torch.save({"model": model.state_dict(), "config": to_dict(cfg), "iteration": it,
                "window_stage": curriculum.stage, "state_bounds": (lo, hi), "val": val}, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=3e-4")
    a = p.parse_args()
    train(load_config(a.config, a.variant, a.set))


if __name__ == "__main__":
    main()
