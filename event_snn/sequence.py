"""Show the event frames that lead up to one target map (and the prediction, with a checkpoint).

    python -m event_snn.sequence --env 3 --t 7.5                       # held-out set, no model
    python -m event_snn.sequence --data runs/event_snn/train_s1000.pt --env 0 --t 42
    python -m event_snn.sequence --checkpoint runs/event_snn/<run>/best.pt --env 3 --t 7.5
    python -m event_snn.sequence --env 3 --nearest                     # the env's closest-contact step

Panels run oldest -> newest; each sums --every 10 ms frames over the last --window seconds.
"""
import argparse
import os

import cv2
import torch

from .config import heldout_path, load_config, steps
from .data import select_target
from .visualize import event_sequence


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=None, help="held-out set (default) or a recorded set")
    p.add_argument("--checkpoint", default=None, help="also draw this model's prediction")
    p.add_argument("--variant", default=None, help="without a checkpoint: which target map (via data.target_map)")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--t", type=float, default=None, help="time of the target map [s]")
    p.add_argument("--nearest", action="store_true", help="pick the env's step with the highest p98 target")
    p.add_argument("--window", type=float, default=0.3, help="seconds of events before t")
    p.add_argument("--every", type=int, default=2, help="10 ms frames summed per panel")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    if a.checkpoint:
        from .model import load_checkpoint
        model, cfg, _ = load_checkpoint(a.checkpoint, a.device)
    else:
        model, cfg = None, load_config(None, a.variant, a.set)
    path = a.data or heldout_path(cfg)
    H = select_target(torch.load(path), cfg.data.target_map)
    T = H["done"].shape[0]
    y = H["ttc"][:, a.env].float()
    if a.nearest:
        flat = y.flatten(1)
        score = flat.kthvalue(max(1, int(round(0.98 * flat.shape[1]))), dim=1).values
        t = int(score.argmax())
    elif a.t is not None:
        t = min(steps(a.t), T - 1)
    else:
        raise SystemExit("give --t SECONDS or --nearest")
    H1 = {k: v[:t + 1, a.env:a.env + 1] for k, v in H.items() if torch.is_tensor(v) and v.dim() >= 2}
    pred = None
    if model is not None:
        from .evaluate import forward
        out = forward(model, H1, a.device or cfg.device)   # runs the whole flight up to t, so the state is warm
        pred = out.get("ttc")
    img = event_sequence(H1, t, 0, pred, cfg.data.ttc_eps, a.window, a.every, env_name=a.env,
                         ttc_name="speed/range" if cfg.data.target_map == "near" else "invTTC")
    out_path = a.out or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint or path)),
                                     f"sequence_env{a.env}_t{t / 100:.2f}.png")
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"wrote {out_path}  (env {a.env}, t {t / 100:.2f} s)")


if __name__ == "__main__":
    main()
