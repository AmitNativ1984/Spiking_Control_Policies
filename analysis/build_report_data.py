import json
import math
import sys
from collections import Counter

IN = sys.argv[1]
OUT = sys.argv[2]

d = json.load(open(IN))
recs = d["records"]
n = len(recs)

POOL_SHARE_PCT = {
    "object": 47.5, "sphere": 22.0, "cylinder": 22.0,
    "panel": 3.5, "tree": 3.0, "wall": 2.0,
}
ORDER = ["tree", "panel", "sphere", "cylinder", "wall", "object"]

type_counts = Counter(r["obstacle_type"] for r in recs)


def per_type_stats(t):
    rs = [r for r in recs if r["obstacle_type"] == t]
    if not rs:
        return None
    iv = [r["in_view_frac"] for r in rs]
    ang = [r["angle_to_camera_deg"] for r in rs if not math.isnan(r["angle_to_camera_deg"])]
    return {
        "count": len(rs),
        "crash_share_pct": 100.0 * len(rs) / n,
        "pool_share_pct": POOL_SHARE_PCT.get(t),
        "spawn_trap_pct": 100.0 * sum(1 for r in rs if r["spawn_trap"]) / len(rs),
        "mean_in_view_frac": sum(iv) / len(iv),
        "never_in_view_pct": 100.0 * sum(1 for x in iv if x == 0.0) / len(iv),
        "mean_speed": sum(r["speed_mps"] for r in rs) / len(rs),
        "mean_angle_deg": sum(ang) / len(ang) if ang else None,
        "median_step": sorted(r["step"] for r in rs)[len(rs) // 2],
    }


def hist(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if (lo <= v < hi) or (i == len(edges) - 2 and v == hi):
                counts[i] += 1
                break
    return counts


step_edges = list(range(0, 310, 10))
step_hist = hist([r["step"] for r in recs], step_edges)

iv_edges = [i / 10 for i in range(11)]
iv_hist = hist([r["in_view_frac"] for r in recs], iv_edges)

speed_edges = [i * 0.5 for i in range(13)]
speed_hist = hist([r["speed_mps"] for r in recs], speed_edges)

angle_edges = list(range(0, 190, 15))
angle_vals = [r["angle_to_camera_deg"] for r in recs if not math.isnan(r["angle_to_camera_deg"])]
angle_hist = hist(angle_vals, angle_edges)

scatter = [
    {"x": round(r["angle_to_camera_deg"], 1), "y": round(r["speed_mps"], 2), "t": r["obstacle_type"]}
    for r in recs if not math.isnan(r["angle_to_camera_deg"])
]

summary = {
    "meta": {k: v for k, v in d.items() if k != "records"},
    "n_crashes": n,
    "spawn_trap_count": sum(1 for r in recs if r["spawn_trap"]),
    "spawn_trap_pct": 100.0 * sum(1 for r in recs if r["spawn_trap"]) / n,
    "never_in_view_pct": 100.0 * sum(1 for r in recs if r["in_view_frac"] == 0.0) / n,
    "mostly_not_in_view_pct": 100.0 * sum(1 for r in recs if r["in_view_frac"] < 0.5) / n,
    "mean_in_view_frac": sum(r["in_view_frac"] for r in recs) / n,
    "mean_speed": sum(r["speed_mps"] for r in recs) / n,
    "median_speed": sorted(r["speed_mps"] for r in recs)[n // 2],
    "max_speed": max(r["speed_mps"] for r in recs),
    "mean_angle_deg": sum(angle_vals) / len(angle_vals),
    "median_angle_deg": sorted(angle_vals)[len(angle_vals) // 2],
    "by_type": {t: per_type_stats(t) for t in ORDER if type_counts.get(t)},
    "type_order": [t for t in ORDER if type_counts.get(t)],
    "step_hist": {"edges": step_edges, "counts": step_hist},
    "in_view_hist": {"edges": iv_edges, "counts": iv_hist},
    "speed_hist": {"edges": speed_edges, "counts": speed_hist},
    "angle_hist": {"edges": angle_edges, "counts": angle_hist},
    "scatter": scatter,
}

json.dump(summary, open(OUT, "w"))
print(f"wrote {OUT}")
print(json.dumps(summary["by_type"], indent=2))
