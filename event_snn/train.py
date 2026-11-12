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
from .loss import depth_velocity_loss, episode_age, ttc_loss, valid_mask
from .evaluate import ttc_errors
from .model import EventSNN, Monitor, calibrate_rates, count_parameters, surrogate_half_width
from .data import log_rho, select_target, state_input
from .sim import Rollout, measure_bounds, write_heldout
from .visualize import depth_image, pick_frames

TRAIN_KEYS = ("events", "imu", "grav", "rho", "ttc", "near", "vel", "done")


def to_device(chunk, dev, t=slice(None), e=slice(None)):
    return {k: v[t, e].to(dev, non_blocking=True) for k, v in chunk.items() if k in TRAIN_KEYS}


@torch.no_grad()
def validate(model, heldout, cfg):
    """Loss on the first val_envs x val_seconds of the held-out flights, from a cold state."""
    dev = cfg.device
    E, T = min(cfg.train.val_envs, heldout["done"].shape[1]), steps(cfg.train.val_seconds)
    b = to_device(heldout, dev, slice(0, T), slice(0, E))
    b["rho"] = b["rho"].float()
    if "ttc" in b:
        b["ttc"] = b["ttc"].float()
    ex = {}
    d, v, _, rates = model.unroll(b["events"], state_input(b), b["done"], model.init_state(E), extras=ex)
    _, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, steps(cfg.loss.burn_in_s),
                                  ex.get("ttc"), cfg.data.ttc_eps, cfg.data.ttc_max)
    logs.update({f"rate/{k}": r for k, r in rates.items()})
    if "ttc" in ex:
        # reference: the same loss for one constant value everywhere (the median log target).
        # A model near this number has learned the average level and nothing spatial.
        valid = valid_mask(b["done"], steps(cfg.loss.burn_in_s)).to(ex["ttc"].dtype)
        const = torch.log(b["ttc"] + cfg.data.ttc_eps)[valid.bool()].median()
        ref = ttc_loss(torch.full_like(ex["ttc"], float(const)), b["ttc"], cfg.data.ttc_eps, cfg.data.ttc_max)
        logs["ttc_const"] = (ref * valid).sum() / valid.sum().clamp(min=1.0)
        # stronger reference: the best single value per frame (its median log target) -- knows how near
        # things are overall in every frame but not where. Only the gap below this is spatial skill.
        lt = torch.log(b["ttc"] + cfg.data.ttc_eps)
        fc = lt.flatten(2).median(-1).values[..., None, None].expand_as(lt)
        ref_f = ttc_loss(fc, b["ttc"], cfg.data.ttc_eps, cfg.data.ttc_max)
        logs["ttc_frame_const"] = (ref_f * valid).sum() / valid.sum().clamp(min=1.0)
    if "ttc" in ex:
        # does the map come from this flight's events? the same flights with another drone's events
        # (rolled by one env) and with no events; a model that reads obstacles gets much worse
        for name, ev in (("ttc_swapped", b["events"].roll(1, dims=1)), ("ttc_noevents", torch.zeros_like(b["events"]))):
            ex2 = {}
            d2, v2, _, r2 = model.unroll(ev, state_input(b), b["done"], model.init_state(E), extras=ex2)
            _, l2 = depth_velocity_loss(d2, v2, r2, b, cfg.loss, cfg.data.inv_depth_eps, steps(cfg.loss.burn_in_s),
                                        ex2["ttc"], cfg.data.ttc_eps, cfg.data.ttc_max)
            logs[name] = l2["ttc"]
        # cells the drone is closing on fast (invTTC > 1/s): share flagged (recall), and share of
        # clearly safe cells (<= 0.5/s) flagged anyway (false alarm)
        te = ttc_errors(ex["ttc"], {"ttc": b["ttc"]}, cfg.data.ttc_eps, valid.bool())
        for k in ("danger_recall", "false_alarm", "danger_rel_median"):
            if k in te:
                logs[k] = te[k]
    return {k: float(x) for k, x in logs.items()}, (b, d, ex.get("ttc"))


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
    run_name = cfg.run_name or f"{cfg.variant}{'' if getattr(cfg.model, 'spiking', True) else '_rate'}_{time.strftime('%Y%m%d_%H%M%S')}"
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
    heldout = select_target(torch.load(hp), cfg.data.target_map)
    print(f"held-out set {hp}: {tuple(heldout['done'].shape)} (steps, envs)")

    lo, hi = measure_bounds(rollout, cfg.data.bounds_steps, cfg.data.bounds_percentiles)
    if cfg.loss.lambda_ttc > 0 and not cfg.model.ttc_head:
        raise SystemExit("loss.lambda_ttc > 0 needs model.ttc_head=true")
    if cfg.loss.lambda_ttc > 0 and "ttc" not in heldout:
        raise SystemExit(f"{hp} predates the invTTC target; delete it or rerun `python -m event_snn.sim --heldout`")
    model = EventSNN(cfg.model).to(dev)
    model.set_input_bounds(lo, hi)
    print(f"{cfg.variant}: {count_parameters(model)} parameters; state bounds lo {lo.tolist()} hi {hi.tolist()}")
    init_rate = getattr(cfg.model, "init_rate", None)
    if init_rate:
        n = min(16, rollout.n)
        cal = rollout.collect(steps(0.5), keys=("events", "imu", "grav", "done"))
        cal = {k: v[:, :n] for k, v in cal.items()}
        rep = calibrate_rates(model, cal["events"], state_input(cal), cal["done"], target=float(init_rate))
        print("calibrated initial firing: " + "  ".join(f"{k} x{m:.3g} -> {r:.3f}" for k, (m, r) in rep.items()),
              flush=True)

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
    age = torch.zeros(N, dtype=torch.long, device=dev)   # steps since each drone's state was reset (cold: 0)
    best_val = math.inf

    for it in range(1, tc.iterations + 1):
        T = curriculum.T
        t0 = time.time()
        chunk = select_target(rollout.collect(T, keys=TRAIN_KEYS), cfg.data.target_map)
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
                age0 = age[ids]
                record = None
                if p == 0 and i == 0 and it % tc.monitor_every == 0:
                    mon = record = Monitor(half_width)
                ex = {}
                d, v, st_end, rates = model.unroll(b["events"], state_input(b), b["done"], st,
                                                   grad_decay=tc.bptt_grad_decay, record=record, extras=ex)
                loss, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, burn,
                                                 ex.get("ttc"), cfg.data.ttc_eps, cfg.data.ttc_max, age0)
                last = (b, d.detach(), ex["ttc"].detach() if "ttc" in ex else None, age0)
                if not torch.isfinite(loss):
                    print(f"it {it}: non-finite loss, batch skipped", flush=True)
                    continue
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
        age = episode_age(chunk["done"], age)[-1]
        sched.step()
        t_train = time.time() - t0 - t_collect

        if (it % tc.log_every == 0 or it == 1) and n_mb:
            scal = {f"train/{k}": float(x) / n_mb for k, x in agg.items()}
            scal.update({"train/lr": sched.get_last_lr()[0], "train/window_steps": T,
                         "time/collect_s": t_collect, "time/train_s": t_train})
            for k, x in scal.items():
                writer.add_scalar(k, x, it)
            if wb:
                wb.log(scal, step=it)
            parts = [f"{n} {scal['train/' + n]:.4f}" for n, w in (("depth", cfg.loss.lambda_d),
                     ("ttc", cfg.loss.lambda_ttc), ("vel", cfg.loss.lambda_v)) if w > 0]
            print(f"it {it:5d}  T {T:3d}  loss {scal['train/loss']:.4f}  " + "  ".join(parts) + "  "
                  f"{model.interface} rate {scal['train/rate/' + model.interface]:.3f}  "
                  f"collect {t_collect:.1f}s train {t_train:.1f}s", flush=True)

        if mon is not None:
            rec = {"it": it, "T": T, "grad": mon.grad, "near": mon.near}
            monitor_log.write(json.dumps(rec) + "\n")
            monitor_log.flush()
            for k in model.spiking_layers:
                g = mon.grad[k]
                writer.add_scalar(f"monitor/near_threshold/{k}", sum(mon.near[k]) / len(mon.near[k]), it)
                if g and g[-1] > 0:
                    writer.add_scalar(f"monitor/grad_first_over_last/{k}", g[0] / g[-1], it)
                    writer.add_scalar(f"monitor/grad_last/{k}", g[-1], it)

        if it % tc.val_every == 0:
            model.eval()
            val, (vb, vd, vt) = validate(model, heldout, cfg)
            frames = pick_frames(vb["ttc"], valid_mask(vb["done"], steps(cfg.loss.burn_in_s))) if vt is not None else None
            maps = depth_image(vb, vd, cfg.data.inv_depth_eps, ttc=vt, ttc_eps=cfg.data.ttc_eps,
                               depth=cfg.loss.lambda_d > 0, frames=frames,
                               ttc_name="speed/range" if cfg.data.target_map == "near" else "invTTC")
            writer.add_image("val/depth", maps, it, dataformats="HWC")
            model.train()
            for k, x in val.items():
                writer.add_scalar(f"val/{k}", x, it)
            if wb:
                wb.log({f"val/{k}": x for k, x in val.items()}, step=it)
                wb.log({"val/maps": wandb.Image(maps)}, step=it)
            # the same panels on the newest training window (fresh sim data, drawn after this iteration's update)
            tb, td, tt, tage = last
            tframes = pick_frames(tb["ttc"], valid_mask(tb["done"], burn, tage)) if tt is not None else None
            tmaps = depth_image(tb, td, cfg.data.inv_depth_eps, ttc=tt, ttc_eps=cfg.data.ttc_eps,
                                depth=cfg.loss.lambda_d > 0, frames=tframes,
                                ttc_name="speed/range" if cfg.data.target_map == "near" else "invTTC")
            writer.add_image("train/depth", tmaps, it, dataformats="HWC")
            if wb:
                wb.log({"train/maps": wandb.Image(tmaps)}, step=it)
            os.makedirs(os.path.join(run_dir, "maps"), exist_ok=True)
            for nm, im in (("val", maps), ("train", tmaps)):
                cv2.imwrite(os.path.join(run_dir, "maps", f"{nm}_{it:05d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            grew = curriculum.update(val["loss"])
            vparts = [f"{n} {val[n]:.4f}" for n, w in (("depth", cfg.loss.lambda_d), ("ttc", cfg.loss.lambda_ttc),
                      ("vel", cfg.loss.lambda_v)) if w > 0]
            if "ttc_const" in val:
                vparts.append(f"(constant map: {val['ttc_const']:.4f}, per-frame constant: {val['ttc_frame_const']:.4f})")
            if "ttc_swapped" in val:
                vparts.append(f"swapped events {val['ttc_swapped']:.4f} no events {val['ttc_noevents']:.4f}")
            if "danger_recall" in val:
                vparts.append(f"danger recall {val['danger_recall']:.3f} false alarm {val['false_alarm']:.3f}")
            print(f"  val loss {val['loss']:.4f} " + " ".join(vparts)
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
