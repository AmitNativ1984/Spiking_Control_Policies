"""Print the parameter groups in a checkpoint, to tell whether a GRU run actually
built a recurrent critic or silently fell back to the feed-forward one.

usage: python analysis/inspect_gru_ckpt.py <ckpt.pth> [<ckpt.pth> ...]
"""
import sys

import torch

for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = ck.get("model", ck)
    groups = {}
    for k, v in w.items():
        head = ".".join(k.split(".")[:3])
        groups.setdefault(head, 0)
        groups[head] += v.numel() if hasattr(v, "numel") else 0
    print(f"\n=== {path.split('/')[-1]}  (epoch {ck.get('epoch', '?')}) ===")
    for g, n in sorted(groups.items()):
        print(f"   {g:50s} {n:>10,} params")
    has_actor_gru = any("actor_gru" in k for k in w)
    has_critic_gru = any("critic_gru" in k for k in w)
    print(f"   -> actor_gru present:  {has_actor_gru}")
    print(f"   -> critic_gru present: {has_critic_gru}")
