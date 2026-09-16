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

if "vel_azimuth_deg" in recs[0]:
    HALF_H = d["meta"]["camera_half_h_fov_deg"] if "meta" in d else d["camera_half_h_fov_deg"]
    HALF_V = d.get("camera_half_v_fov_deg", 28.09)
    az = [abs(r["vel_azimuth_deg"]) for r in recs if not math.isnan(r["vel_azimuth_deg"])]
    el = [abs(r["vel_elev_deg"]) for r in recs if not math.isnan(r["vel_elev_deg"])]
    print("\n--- velocity misalignment decomposition (|deg|) ---")
    print(f"  azimuth : median {st.median(az):.1f}  mean {sum(az)/len(az):.1f}  "
          f"> half-H-FOV ({HALF_H:.1f}): {100*sum(1 for x in az if x > HALF_H)/len(az):.1f}%")
    print(f"  elevation: median {st.median(el):.1f}  mean {sum(el)/len(el):.1f}  "
          f"> half-V-FOV ({HALF_V:.1f}): {100*sum(1 for x in el if x > HALF_V)/len(el):.1f}%")

    oaz = [abs(r["obst_azimuth_deg"]) for r in recs if not math.isnan(r["obst_azimuth_deg"])]
    oel = [abs(r["obst_elev_deg"]) for r in recs if not math.isnan(r["obst_elev_deg"])]
    print("\n--- struck-obstacle bearing at impact (|deg|) ---")
    print(f"  azimuth : median {st.median(oaz):.1f}  "
          f"> half-H-FOV: {100*sum(1 for x in oaz if x > HALF_H)/len(oaz):.1f}%")
    print(f"  elevation: median {st.median(oel):.1f}  "
          f"> half-V-FOV: {100*sum(1 for x in oel if x > HALF_V)/len(oel):.1f}%")

    print("\n--- counterfactual visibility over the last 10 frames ---")
    print("    (fraction of crashes where the obstacle was seen in >=1 frame)")
    n = len(recs)
    for key, label in [
        ("in_view_count", "actual (body frame, real camera)"),
        ("in_view_vehicle_baseline", "same FOV, vehicle frame (tilt removed)"),
        ("in_view_yaw_slaved", "yaw slaved to velocity, 87x56"),
        ("in_view_wide_fov", "current yaw, 120x90 FOV"),
        ("in_view_yaw_slaved_wide", "yaw slaved + 120x90 FOV"),
    ]:
        seen = sum(1 for r in recs if r.get(key, 0) > 0)
        frames = sum(r.get(key, 0) for r in recs)
        denom = sum(r["hist_frames_available"] for r in recs)
        print(f"  {label:40s} seen-at-all {100*seen/n:5.1f}%   "
              f"mean frac {frames/denom:.3f}")

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
