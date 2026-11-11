"""Evaluate a checkpoint on the held-out flights (docs/event_snn_experiment_plan.md §6). No Isaac Gym.

    python -m event_snn.evaluate runs/event_snn/<run>/best.pt
    python -m event_snn.evaluate --compare runs/event_snn/a_rec_* runs/event_snn/b_*

Writes <run>/eval/metrics.json and plots. All variants are scored on the same steps: every
step at least EVAL_BURN_IN_S into its episode, whatever burn-in the variant trained with.
"""
import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .config import heldout_path, steps  # noqa: E402
from .data import REGIMES, AttitudeNoise, log_rho, regime_labels, select_target  # noqa: E402
from .loss import scale_invariant  # noqa: E402
from .model import load_checkpoint  # noqa: E402

EVAL_BURN_IN_S = 0.2
AGE_BINS_S = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, math.inf]
BLACKOUT_MS = [50, 100, 200, 500]
ATTITUDE_DEG = [0.0, 1.0, 2.0, 3.0, 5.0]
PROBE_EVERY = 5


TRACE_DECAY = 0.9  # per 10 ms step: ~100 ms low-pass of the interface spikes, as a policy would integrate them


class MemRecorder:
    """Every `every` steps keeps the memory membrane and a low-pass trace of the interface spikes
    (model.step calls record.step each step)."""

    def __init__(self, every, interface):
        self.every, self.interface, self.k, self.u, self.z, self.trace = every, interface, 0, [], [], None

    def step(self, mems, spikes):
        s = spikes[self.interface].flatten(1)
        self.trace = s if self.trace is None else TRACE_DECAY * self.trace + s
        if self.k % self.every == 0:
            if "mem" in mems:
                self.u.append(mems["mem"][0].half().cpu())
            self.z.append(self.trace.half().cpu())
        self.k += 1


@torch.no_grad()
def forward(model, H, dev, events=None, grav=None, record_every=0, chunk=100):
    E, T = H["done"].shape[1], H["done"].shape[0]
    events = H["events"] if events is None else events
    grav = H["grav"] if grav is None else grav
    st = model.init_state(E, dev)
    rec = MemRecorder(record_every, model.interface) if record_every else None
    pd, pv, pt, rate_sum = [], [], [], {k: 0.0 for k in model.spiking_layers}
    for t0 in range(0, T, chunk):
        sl = slice(t0, min(t0 + chunk, T))
        si = torch.cat([H["imu"][sl].float(), grav[sl].float()], -1).to(dev)
        ex = {}
        d, v, st, rates = model.unroll(events[sl].to(dev), si, H["done"][sl].to(dev), st, record=rec, extras=ex)
        pd.append(d.half().cpu())
        if "ttc" in ex:
            pt.append(ex["ttc"].half().cpu())
        pv.append(v.cpu())
        for k, r in rates.items():
            rate_sum[k] += float(r) * (sl.stop - sl.start)
    out = {"depth": torch.cat(pd).float(), "vel": torch.cat(pv),
           "rates": {k: x / T for k, x in rate_sum.items()}}
    if pt:
        out["ttc"] = torch.cat(pt).float()
    if rec is not None:
        if rec.u:
            out["mem_u"] = torch.stack(rec.u)  # (T / every, E, C, 8, 8)
        out["iface"] = torch.stack(rec.z)  # (T / every, E, n_interface)
    return out


def _cells(H):
    now, rec = H["m_now"], H["m_recent"]
    return {"now": now, "recent": rec, "never": ~now & ~rec}


def depth_errors(err, mask, valid):
    """err (T, E, G, G) log-inverse-depth error. mean |err| and mean per-image sqrt(SI) over mask."""
    m = mask & valid[..., None, None]
    n = int(m.sum())
    if n == 0:
        return {"abs": float("nan"), "si": float("nan"), "cells": 0}
    abs_err = float(err.abs()[m].mean())
    per_img = mask.flatten(2).sum(-1)
    ok = valid & (per_img >= 5)
    si = scale_invariant(err, mask).clamp(min=0).sqrt()
    return {"abs": abs_err, "si": float(si[ok].mean()) if ok.any() else float("nan"), "cells": n}


def velocity_errors(pred, true, valid):
    p, t = pred[valid], true[valid]
    if len(t) == 0:
        return {"rmse": float("nan"), "mag": float("nan"), "dir_deg": float("nan")}
    moving = t.norm(dim=-1) > 0.5
    cos = F.cosine_similarity(p[moving], t[moving], dim=-1).clamp(-1, 1)
    return {"rmse": float(((p - t) ** 2).sum(-1).mean().sqrt()),
            "mag": float((p.norm(dim=-1) - t.norm(dim=-1)).abs().mean()),
            "dir_deg": float(torch.rad2deg(torch.acos(cos)).mean()) if moving.any() else float("nan")}


def score(out, H, eps, valid):
    err = out["depth"] - log_rho(H["rho"].float(), eps)
    cells = _cells(H)
    return {"depth": {k: depth_errors(err, m, valid) for k, m in cells.items()},
            "vel": velocity_errors(out["vel"], H["vel"], valid)}


def by_age(out, H, eps):
    err = (out["depth"] - log_rho(H["rho"].float(), eps)).abs()
    age_s = H["age"].float() * 0.01
    rows = []
    for lo, hi in zip(AGE_BINS_S[:-1], AGE_BINS_S[1:]):
        sel = (age_s >= lo) & (age_s < hi)
        m = H["m_now"] & sel[..., None, None]
        ve = ((out["vel"] - H["vel"]) ** 2).sum(-1)[sel]
        rows.append({"age_lo_s": lo, "age_hi_s": hi, "steps": int(sel.sum()),
                     "depth_abs_now": float(err[m].mean()) if m.any() else float("nan"),
                     "vel_rmse": float(ve.mean().sqrt()) if len(ve) else float("nan")})
    return rows


def blackout(model, H, dev, eps, base):
    T = H["done"].shape[0]
    t0 = steps(5.0) if T > steps(5.0) + steps(1.5) else T // 3
    err_b = (base["depth"] - log_rho(H["rho"].float(), eps)).abs().mean((-2, -1))
    res = {"t0_s": t0 * 0.01, "durations": {}}
    for ms in BLACKOUT_MS:
        dur = steps(ms / 1000.0)
        lo, hi = max(t0 - 20, 0), min(t0 + dur + 100, T)
        ok = ~H["done"][lo:hi].any(0)
        if not ok.any():
            continue
        ev = H["events"].clone()
        ev[t0:t0 + dur] = 0
        out = forward(model, H, dev, events=ev)
        err = (out["depth"] - log_rho(H["rho"].float(), eps)).abs().mean((-2, -1))
        vb = ((base["vel"] - H["vel"]) ** 2).sum(-1).sqrt()
        vo = ((out["vel"] - H["vel"]) ** 2).sum(-1).sqrt()
        d_ex = (err - err_b)[lo:hi, ok].mean(1)
        v_ex = (vo - vb)[lo:hi, ok].mean(1)
        end = t0 + dur - 1 - lo
        res["durations"][ms] = {
            "envs": int(ok.sum()), "t_rel_s": [(t - t0) * 0.01 for t in range(lo, hi)],
            "depth_excess": d_ex.tolist(), "vel_excess": v_ex.tolist(),
            "depth_excess_at_end": float(d_ex[end]), "vel_excess_at_end": float(v_ex[end]),
            "depth_base": float(err_b[lo:hi, ok].mean()), "vel_base": float(vb[lo:hi, ok].mean())}
    return res


def attitude_sweep(model, cfg, H, dev, eps, valid):
    if "grav" not in model.state_inputs:
        return None
    a = cfg.attitude
    rows = []
    for deg in ATTITUDE_DEG:
        f = deg / a.noise_deg if a.noise_deg > 0 else 0.0
        E = H["done"].shape[1]
        noise = AttitudeNoise(E, deg, a.bias_init_deg * f, a.bias_rw_deg * f,
                              generator=torch.Generator().manual_seed(0))
        g = torch.stack([noise(H["grav_true"][t].float(), H["done"][t]) for t in range(H["done"].shape[0])])
        s = score(forward(model, H, dev, grav=g), H, eps, valid)
        rows.append({"noise_deg": deg, "depth_abs_now": s["depth"]["now"]["abs"],
                     "depth_si_now": s["depth"]["now"]["si"], "vel_rmse": s["vel"]["rmse"]})
    return rows


def by_regime(out, H, eps, valid):
    lab = torch.stack([regime_labels(H["vel"][t], H["yaw_rate"][t].float(), H["rho"][t].float())
                       for t in range(H["done"].shape[0])])
    err = out["depth"] - log_rho(H["rho"].float(), eps)
    rows = {}
    for i, name in enumerate(REGIMES):
        sel = valid & (lab == i)
        rows[name] = {"share": float(sel.float().mean()),
                      "depth_now": depth_errors(err, H["m_now"], sel),
                      "vel": velocity_errors(out["vel"], H["vel"], sel)}
    return rows


def _ridge(Xtr, Ytr, Xte, Yte, lam=1e-2):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1)], 1)
    Xte = torch.cat([Xte, torch.ones(len(Xte), 1)], 1)
    A = Xtr.T @ Xtr + lam * len(Xtr) * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ Ytr)
    res = ((Xte @ W - Yte) ** 2).sum(0)
    tot = ((Yte - Yte.mean(0)) ** 2).sum(0).clamp(min=1e-12)
    return 1.0 - res / tot


def linear_probes(out, H, eps):
    """Ridge probes on the frozen memory membrane: per 8x8 location -> log(1/Z + eps) of that
    region; grid mean -> body velocity. Train on the first half of the envs, R^2 on the rest."""
    u = out["mem_u"].float()                                    # (S, E, C, 8, 8)
    idx = torch.arange(0, H["done"].shape[0], PROBE_EVERY)[: u.shape[0]]
    ok = H["age"][idx] >= steps(EVAL_BURN_IN_S)                 # (S, E)
    rho8 = F.avg_pool2d(H["rho"][idx].float().flatten(0, 1)[:, None], 4)[:, 0]
    y_d = log_rho(rho8, eps).view(len(idx), -1, 8, 8)
    E = u.shape[1]
    tr, te = slice(0, E // 2), slice(E // 2, E)
    r2_loc = torch.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            r2_loc[i, j] = _ridge(u[:, tr, :, i, j][ok[:, tr]], y_d[:, tr, i, j][ok[:, tr]][:, None],
                                  u[:, te, :, i, j][ok[:, te]], y_d[:, te, i, j][ok[:, te]][:, None])[0]
    pooled = u.mean((-2, -1))
    v = H["vel"][idx]
    r2_v = _ridge(pooled[:, tr][ok[:, tr]], v[:, tr][ok[:, tr]], pooled[:, te][ok[:, te]], v[:, te][ok[:, te]])
    return {"depth_r2_mean": float(r2_loc.mean()), "depth_r2_map": r2_loc.tolist(),
            "vel_r2": r2_v.tolist(), "vel_r2_mean": float(r2_v.mean())}


def ttc_errors(pred_log, H, eps, valid, danger=1.0):
    """Inverse time-to-contact error of a (T, E, G, G) log(invTTC + eps) map. Danger region =
    true invTTC > `danger` 1/s: median relative error there, and how often it is flagged at all."""
    true = H["ttc"].float()
    pred = (pred_log.exp() - eps).clamp(min=0.0)
    v = valid[..., None, None]
    dz, safe = (true > danger) & v, (true <= 0.5 * danger) & v
    if not dz.any():
        return {"danger_cells": 0}
    rel = (pred[dz] - true[dz]).abs() / true[dz]
    return {"danger_cells": int(dz.sum()), "danger_share": float(dz.float().mean()),
            "danger_rel_median": float(rel.median()), "danger_rel_mean": float(rel.mean()),
            "danger_recall": float((pred[dz] > danger).float().mean()),
            "false_alarm": float((pred[safe] > danger).float().mean()) if safe.any() else float("nan"),
            "log_abs": float((pred_log - torch.log(true + eps)).abs()[v.expand_as(true)].mean())}


def derived_ttc(out, H, eps_rho, eps_ttc):
    """log(invTTC + eps) rebuilt from the depth and velocity heads: max(0, v_hat.d) / (r_hat - r_drone)
    with r_hat = 1 / (rho_hat d_z). Compared with the direct head, it shows whether a dedicated
    invTTC output adds anything over geometry x velocity."""
    g = H["meta"]["geom"]
    rays = g["rays_grid"].float()                                # (G, G, 3) optical, unit
    rho = (out["depth"].exp() - eps_rho).clamp(min=1e-4)
    r = 1.0 / (rho * rays[..., 2])
    v_cam = torch.stack([-out["vel"][..., 1], -out["vel"][..., 2], out["vel"][..., 0]], -1)
    closing = torch.einsum("tec,hwc->tehw", v_cam, rays).clamp(min=0.0)
    ttc = (closing / (r - g["drone_radius_m"]).clamp(min=0.05)).clamp(max=g["ttc_max"])
    ttc = torch.where(rho <= 1.0 / g["far_m"], torch.zeros_like(ttc), ttc)
    return torch.log(ttc + eps_ttc)


def action_probe(out, H):
    """How much of what the teacher's depth latent contributes to its action is linearly available
    in the interface spikes. Ridge to the teacher's action from: its 17 state inputs alone (floor),
    state + interface spike trace, state + its own 32-D depth latent (ceiling). `recovered` =
    (R2_iface - R2_floor) / (R2_ceiling - R2_floor). Train on the first half of the envs."""
    z = out["iface"].float()                                    # (S, E, n)
    idx = torch.arange(0, H["done"].shape[0], PROBE_EVERY)[: z.shape[0]]
    ok = H["age"][idx] >= steps(EVAL_BURN_IN_S)
    if "mode" in H:
        ok = ok & (H["mode"][idx] == 0)                         # the teacher is flying, not a perturbation
    tobs, act = H["tobs"][idx].float(), H["action"][idx].float()
    state, latent = tobs[..., :17], tobs[..., 17:]
    E = z.shape[1]
    tr, te = slice(0, E // 2), slice(E // 2, E)

    def r2(x, y):
        return _ridge(x[:, tr][ok[:, tr]], y[:, tr][ok[:, tr]], x[:, te][ok[:, te]], y[:, te][ok[:, te]])

    floor, ceil = r2(state, act), r2(torch.cat([state, latent], -1), act)
    mine = r2(torch.cat([state, z], -1), act)
    gap = (ceil - floor).clamp(min=1e-3)
    return {"action_r2_floor": floor.tolist(), "action_r2_iface": mine.tolist(), "action_r2_ceiling": ceil.tolist(),
            "recovered": ((mine - floor) / gap).tolist(), "recovered_mean": float(((mine - floor) / gap).mean()),
            "latent_r2_mean": float(r2(z, latent).mean()), "features": int(z.shape[-1]),
            "train_samples": int(ok[:, tr].sum())}


def plots(m, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    rows = [r for r in m["by_age"] if r["steps"]]
    x = [0.5 * (r["age_lo_s"] + min(r["age_hi_s"], 20.0)) for r in rows]
    ax[0].plot(x, [r["depth_abs_now"] for r in rows], "o-")
    ax[0].set(xscale="log", xlabel="time since flight start [s]", ylabel="|log 1/Z err| (event cells)")
    ax[1].plot(x, [r["vel_rmse"] for r in rows], "o-")
    ax[1].set(xscale="log", xlabel="time since flight start [s]", ylabel="velocity RMSE [m/s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "err_vs_time.png"), dpi=120)
    plt.close(fig)

    if m["blackout"]["durations"]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for ms, r in m["blackout"]["durations"].items():
            ax[0].plot(r["t_rel_s"], r["depth_excess"], label=f"{ms} ms")
            ax[1].plot(r["t_rel_s"], r["vel_excess"], label=f"{ms} ms")
        ax[0].set(xlabel="time since blackout start [s]", ylabel="excess depth error (all cells)")
        ax[1].set(xlabel="time since blackout start [s]", ylabel="excess velocity error [m/s]")
        ax[0].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "blackout.png"), dpi=120)
        plt.close(fig)

    if m.get("attitude_sweep"):
        r = m["attitude_sweep"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot([x["noise_deg"] for x in r], [x["depth_abs_now"] for x in r], "o-")
        ax[0].set(xlabel="attitude noise [deg]", ylabel="|log 1/Z err| (event cells)")
        ax[1].plot([x["noise_deg"] for x in r], [x["vel_rmse"] for x in r], "o-")
        ax[1].set(xlabel="attitude noise [deg]", ylabel="velocity RMSE [m/s]")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "attitude_sweep.png"), dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    reg = m["by_regime"]
    ax[0].bar(list(reg), [reg[k]["depth_now"]["abs"] for k in reg])
    ax[0].set(ylabel="|log 1/Z err| (event cells)")
    ax[1].bar(list(reg), [reg[k]["vel"]["rmse"] for k in reg])
    ax[1].set(ylabel="velocity RMSE [m/s]")
    ax[2].bar(list(m["spike_rates"]), list(m["spike_rates"].values()))
    ax[2].set(ylabel="spike rate per step")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regimes_rates.png"), dpi=120)
    plt.close(fig)

    if "probes" in m:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(m["probes"]["depth_r2_map"], vmin=0, vmax=1)
        ax.set_title("probe R^2: memory -> 1/Z")
        fig.colorbar(im)
        fig.savefig(os.path.join(out_dir, "probe_depth_r2.png"), dpi=120)
        plt.close(fig)


def plot_monitor(run_dir, out_dir):
    path = os.path.join(run_dir, "monitor.jsonl")
    if not os.path.exists(path):
        return
    recs = [json.loads(l) for l in open(path)]
    if not recs:
        return
    last = recs[-1]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for k in last["grad"]:
        ax[0].semilogy(last["grad"][k], label=k)
        ax[1].plot(last["near"][k], label=k)
    ax[0].set(xlabel="step in window", ylabel="|dL/dU|", title=f"it {last['it']}")
    ax[1].set(xlabel="step in window", ylabel="share |U - theta| < half-width")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grad_monitor.png"), dpi=120)
    plt.close(fig)


def evaluate(ckpt, heldout=None, out_dir=None, device=None):
    model, cfg, _ = load_checkpoint(ckpt, device)
    dev = device or cfg.device
    hp = heldout or heldout_path(cfg)
    H = select_target(torch.load(hp), getattr(cfg.data, "target_map", "ttc"))
    eps = cfg.data.inv_depth_eps
    valid = H["age"] >= steps(EVAL_BURN_IN_S)
    run_dir = os.path.dirname(os.path.abspath(ckpt))
    out_dir = out_dir or os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    base = forward(model, H, dev, record_every=PROBE_EVERY)
    m = {"checkpoint": ckpt, "heldout": hp, "variant": cfg.variant, **score(base, H, eps, valid),
         "spike_rates": base["rates"], "by_age": by_age(base, H, eps),
         "by_regime": by_regime(base, H, eps, valid)}
    m["blackout"] = blackout(model, H, dev, eps, base)
    m["attitude_sweep"] = attitude_sweep(model, cfg, H, dev, eps, valid)
    if "mem_u" in base:
        m["probes"] = linear_probes(base, H, eps)
    m["interface"] = {"layer": model.interface, "neurons": int(base["iface"].shape[-1]),
                      "spikes_per_step": base["rates"][model.interface] * int(base["iface"].shape[-1])}
    if "action" in H:
        m["action_probe"] = action_probe(base, H)
    if "ttc" in H:
        te = cfg.data.ttc_eps
        if "geom" in H["meta"] and model.depth_head and model.vel_head:
            m["ttc_derived"] = ttc_errors(derived_ttc(base, H, eps, te), H, te, valid)
        if "ttc" in base:
            m["ttc"] = ttc_errors(base["ttc"], H, te, valid)
            m["ttc"]["target"] = getattr(cfg.data, "target_map", "ttc")
            # same error where the cell has events now / had them in the last 100 ms / neither
            m["ttc"]["log_abs_by_events"] = {
                k: float((base["ttc"] - torch.log(H["ttc"].float() + te)).abs()[c & valid[..., None, None]].mean())
                for k, c in _cells(H).items()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    plots(m, out_dir)
    plot_monitor(run_dir, out_dir)
    print(summary_row(m, header=True))
    print(f"wrote {out_dir}")
    return m


SUMMARY = [
    ("abs@now", lambda m: m["depth"]["now"]["abs"]),
    ("SI@now", lambda m: m["depth"]["now"]["si"]),
    ("abs@recent", lambda m: m["depth"]["recent"]["abs"]),
    ("abs@never", lambda m: m["depth"]["never"]["abs"]),
    ("v rmse", lambda m: m["vel"]["rmse"]),
    ("v mag", lambda m: m["vel"]["mag"]),
    ("v dir°", lambda m: m["vel"]["dir_deg"]),
    ("blk200 d+", lambda m: m["blackout"]["durations"].get("200", m["blackout"]["durations"].get(200, {})).get("depth_excess_at_end", float("nan"))),
    ("blk200 v+", lambda m: m["blackout"]["durations"].get("200", m["blackout"]["durations"].get(200, {})).get("vel_excess_at_end", float("nan"))),
    ("probe d R²", lambda m: m.get("probes", {}).get("depth_r2_mean", float("nan"))),
    ("probe v R²", lambda m: m.get("probes", {}).get("vel_r2_mean", float("nan"))),
    ("act recov", lambda m: m.get("action_probe", {}).get("recovered_mean", float("nan"))),
    ("ttc rel", lambda m: m.get("ttc", {}).get("danger_rel_median", float("nan"))),
    ("ttc rel (d×v)", lambda m: m.get("ttc_derived", {}).get("danger_rel_median", float("nan"))),
    ("iface spk/step", lambda m: m.get("interface", {}).get("spikes_per_step", float("nan"))),
]


def summary_row(m, header=False):
    row = f"| {m['variant']:12s} | " + " | ".join(f"{f(m):.3f}" for _, f in SUMMARY) + " |"
    if not header:
        return row
    head = "| variant      | " + " | ".join(n for n, _ in SUMMARY) + " |"
    return head + "\n|" + "---|" * (len(SUMMARY) + 1) + "\n" + row


def compare(run_dirs):
    rows = []
    for d in run_dirs:
        p = os.path.join(d, "eval", "metrics.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    if not rows:
        raise SystemExit("no eval/metrics.json found")
    print(summary_row(rows[0], header=True))
    for m in rows[1:]:
        print(summary_row(m))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("--heldout", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--compare", nargs="+", default=None, help="run dirs (globs ok) with eval/metrics.json")
    a = p.parse_args()
    if a.compare:
        compare(sorted({d for g in a.compare for d in glob.glob(g)}))
    elif a.checkpoint:
        evaluate(a.checkpoint, a.heldout, a.out, a.device)
    else:
        p.error("give a checkpoint or --compare")


if __name__ == "__main__":
    main()
