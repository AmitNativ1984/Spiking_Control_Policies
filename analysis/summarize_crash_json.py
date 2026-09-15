import json
import math
import statistics as st
import sys
from collections import Counter

d = json.load(open(sys.argv[1]))
print("meta:", {k: v for k, v in d.items() if k != "records"})
recs = d["records"]
print("num records:", len(recs))

types = Counter(r["obstacle_type"] for r in recs)
print("obstacle_type counts:", types.most_common())

spawn = sum(1 for r in recs if r["spawn_trap"])
print("spawn_trap:", spawn, spawn / len(recs))

steps = [r["step"] for r in recs]
steps_sorted = sorted(steps)
print("step stats: min", min(steps), "median", st.median(steps),
      "p90", steps_sorted[int(0.9 * len(steps))], "max", max(steps))

inview = [r["in_view_frac"] for r in recs if r["hist_frames_available"] > 0]
print("in_view_frac stats: n=", len(inview), "mean", sum(inview) / len(inview))
mostly_not = sum(1 for x in inview if x < 0.5)
print("mostly NOT in view (<50%):", mostly_not, mostly_not / len(inview))
never_in_view = sum(1 for x in inview if x == 0.0)
print("never in view (0%):", never_in_view, never_in_view / len(inview))

speeds = [r["speed_mps"] for r in recs]
print("speed stats: mean", sum(speeds) / len(speeds), "median", st.median(speeds), "max", max(speeds))

angles = [r["angle_to_camera_deg"] for r in recs if not math.isnan(r["angle_to_camera_deg"])]
print("angle stats: n=", len(angles), "mean", sum(angles) / len(angles), "median", st.median(angles))

no_hist = sum(1 for r in recs if r["hist_frames_available"] == 0)
print("records with zero history frames:", no_hist)

# obstacle type x spawn_trap crosstab
print("\nper-type: count, spawn_trap%, mean in_view_frac, mean speed, mean angle, median step")
for t, c in types.most_common():
    rs = [r for r in recs if r["obstacle_type"] == t]
    sp = sum(1 for r in rs if r["spawn_trap"]) / len(rs)
    iv = [r["in_view_frac"] for r in rs if r["hist_frames_available"] > 0]
    iv_mean = sum(iv) / len(iv) if iv else float("nan")
    sm = sum(r["speed_mps"] for r in rs) / len(rs)
    ang = [r["angle_to_camera_deg"] for r in rs if not math.isnan(r["angle_to_camera_deg"])]
    ang_mean = sum(ang) / len(ang) if ang else float("nan")
    med_step = st.median(r["step"] for r in rs)
    print(f"  {t:10s} n={c:5d} spawn_trap={sp:.3f} in_view={iv_mean:.3f} "
          f"speed={sm:.2f} angle={ang_mean:.1f} med_step={med_step:.0f}")
