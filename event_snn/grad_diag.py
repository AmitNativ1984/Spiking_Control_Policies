"""Per-parameter gradient norms of one BPTT window on the held-out flights (no Isaac Gym).

    python -m event_snn.grad_diag runs/event_snn/<run>/last.pt [--steps 128] [--envs 16]

Answers where a large pre-clip gradient norm comes from: which parameters, and how it scales
with the window length.
"""
import argparse

import torch

from .config import heldout_path, steps
from .data import state_input
from .loss import depth_velocity_loss
from .model import load_checkpoint


def run(ckpt, T, E, device):
    model, cfg, ck = load_checkpoint(ckpt, device)
    model.train()
    from .data import select_target
    H = select_target(torch.load(heldout_path(cfg)), getattr(cfg.data, "target_map", "ttc"))
    keys = ("events", "imu", "grav", "rho", "ttc", "vel", "done")  # "ttc" is the selected target map
    b = {k: H[k][:T, :E].to(device) for k in keys if k in H}
    b["rho"], b["ttc"] = b["rho"].float(), b["ttc"].float()
    ex = {}
    d, v, _, rates = model.unroll(b["events"], state_input(b), b["done"], model.init_state(E, device), extras=ex)
    loss, logs = depth_velocity_loss(d, v, rates, b, cfg.loss, cfg.data.inv_depth_eps, steps(cfg.loss.burn_in_s),
                                     ex.get("ttc"), cfg.data.ttc_eps, cfg.data.ttc_max)
    model.zero_grad()
    loss.backward()
    rows = [(float(p.grad.norm()), float(p.grad.abs().max()), n, tuple(p.shape))
            for n, p in model.named_parameters() if p.grad is not None]
    total = sum(r[0] ** 2 for r in rows) ** 0.5
    print(f"\n{ckpt}  iteration {ck['iteration']}  T={T} E={E}  loss {float(loss):.3f}  total grad norm {total:.3e}")
    print("  " + "  ".join(f"{k} {float(x):.3f}" for k, x in logs.items()))
    print("  rates " + "  ".join(f"{k} {float(x):.3f}" for k, x in rates.items()))
    for g, m, n, s in sorted(rows, reverse=True)[:12]:
        print(f"  {g:10.3e}  share {100 * g * g / total ** 2:5.1f}%  max|g| {m:9.2e}  {n} {s}")
    for n, p in model.named_parameters():
        if "beta" in n or "tau" in n:
            x = p.detach().exp() if "log_tau" in n else p.detach()
            print(f"  value {n}: min {float(x.min()):.3f} mean {float(x.mean()):.3f} max {float(x.max()):.3f}")
    if model.memory.rec is not None:
        print(f"  recurrent gain (uncapped) {float(model.memory.rec_gain(50)):.3f}, cap {model.memory.rec_max_gain}")
    for n in ("bn_fc", "bn_expand"):
        if hasattr(model, n):
            w = getattr(model, n).weight
            print(f"  {n}.weight std {float(w.std()):.4f} max {float(w.abs().max()):.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--steps", type=int, nargs="+", default=[32, 128])
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    for c in a.checkpoints:
        for T in a.steps:
            run(c, T, a.envs, a.device)


if __name__ == "__main__":
    main()
