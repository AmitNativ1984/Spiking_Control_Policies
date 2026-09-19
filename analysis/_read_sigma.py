"""TEMPORARY: per-channel action sigma from every periodic checkpoint of a run.

fixed_sigma = True, so exploration is one learned log_std vector shared across states
(a2c_network.action_log_std). Reading it off the checkpoints recovers the exploration
TRAJECTORY, which the wandb summary only holds the latest value of -- the question being
whether sigma bottomed out at the epoch where the policy stopped improving.

usage: python analysis/_read_sigma.py <run_dir>
"""
import glob
import os
import re
import sys

import torch

run_dir = sys.argv[1]
paths = glob.glob(os.path.join(run_dir, "nn", "*_ep_*.pth"))
print(f"{len(paths)} checkpoints under {run_dir}/nn", flush=True)

rows = []
for p in paths:
    m = re.search(r"_ep_(\d+)_", os.path.basename(p))
    if not m:
        continue
    ep = int(m.group(1))
    try:
        w = torch.load(p, map_location="cpu", weights_only=False)["model"]
    except Exception as e:  # noqa: BLE001
        print(f"  {ep}: unreadable ({type(e).__name__})", flush=True)
        continue
    key = next((k for k in w if k.endswith(("action_log_std", "sigma"))), None)
    if key is None:
        print(f"  {ep}: no sigma parameter; sample keys {list(w)[:4]}", flush=True)
        continue
    rows.append((ep, key, w[key].float().flatten().exp().tolist()))

rows.sort()
if not rows:
    print("no sigma rows recovered", flush=True)
    raise SystemExit(1)

print(f"{'epoch':>6}  {'thrust':>8} {'roll':>8} {'pitch':>8} {'yaw_rate':>9}   {'geo-mean':>9}")
seen = set()
for ep, key, s in rows:
    if ep in seen:      # both the plain and the double-underscore duplicate get saved
        continue
    seen.add(ep)
    gm = float(torch.tensor(s).log().mean().exp())
    print(f"{ep:>6}  " + " ".join(f"{v:8.4f}" for v in s) + f"   {gm:9.4f}")
print(f"\n(sigma parameter: {rows[0][1]})")
