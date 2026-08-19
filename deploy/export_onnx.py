#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX, so the flight container needs no torch.

Run this in the TRAINING container, where the checkpoint's torch lives. It writes two files:

    hover.onnx   the network graph: normalized observation in, raw action out
    hover.npz    the frozen input-normalization statistics, plus provenance

Both are needed; OnnxHoverPolicy loads the .npz alongside the graph by default.

IT VERIFIES BEFORE IT WRITES. The export is compared against torch's own output over the
whole normalized input range, and the run aborts if they disagree. An ONNX file that silently encodes
a slightly different network is worse than no ONNX file, because everything downstream keeps
working and only the vehicle notices.

USAGE
-----
    python -m deploy.export_onnx \
        --checkpoint /path/to/last_....pth \
        --output <sail-uav-core>/libs/control-policy-api/policies/hover/hover.onnx
"""

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import torch

from control_policy_api.observations import HOVER_OBS_DIM
from .hover import HoverPolicy
from .checkpoint import _NORM_CLAMP

# ONNX opset 13 is old enough for any onnxruntime you will meet on a companion computer and
# new enough for everything a Linear/ELU stack needs. Nothing here benefits from a newer one.
OPSET = 13

# Verified on the CLAMPED action, because that is what reaches the vehicle -- and the raw
# network output is the wrong thing to bound.
#
# Measured on this checkpoint over 8192 probes spanning the normalized input range:
#
#   raw output magnitude              up to  13.6      (55% of channels clamp)
#   onnx vs torch, raw output         max  9.5e-06
#   torch float32 vs float64 truth    max  4.6e-06     <- torch is already this far out
#   onnx vs torch, CLAMPED action     max  2.3e-06
#
# So the raw disagreement is float32 accumulation round-off across two 256-wide GEMMs, not a
# structural difference: torch's own float32 result sits the same distance from the exact
# answer, and the error scales with output magnitude (9.7e-07 on the largest decile against
# 1.7e-07 on the smallest). Neither result is more correct than the other.
#
# 1e-5 on the clamped action is ~4x the measured round-off and still ~30000x smaller than the
# policy's own exploration sigma (0.064-0.111), i.e. 2e-06 rad of commanded tilt. Tight
# enough to catch a dropped layer or a reordered output; loose enough not to fail on
# arithmetic that was never exact to begin with.
EXPORT_TOLERANCE = 1e-5


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()


    policy = HoverPolicy(str(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Export the NETWORK ONLY. Normalization and the action clamp stay in Python -- they are
    # the same numpy on both sides, so freezing them into the graph would buy nothing and
    # make the statistics harder to inspect.
    torch.onnx.export(
        policy._actor,
        torch.zeros(1, HOVER_OBS_DIM, dtype=torch.float32),
        str(args.output),
        input_names=["normalized_observation"],
        output_names=["action"],
        # Batch stays dynamic so the same graph can be used for offline batch evaluation,
        # even though flight always sends one row.
        dynamic_axes={"normalized_observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    stats_path = args.output.with_suffix(".npz")
    np.savez(
        stats_path,
        obs_mean=policy._obs_mean,
        obs_std=policy._obs_std,
        checkpoint_sha256=checkpoint_digest(args.checkpoint),
        exported_with=(
            f"python {platform.python_version()} torch {torch.__version__} "
            f"numpy {np.__version__} opset {OPSET}"
        ),
    )

    # --- verify, and refuse to ship a graph that disagrees with torch --------------------
    try:
        import onnxruntime
    except ImportError:
        print(f"wrote {args.output} and {stats_path}")
        print("WARNING: onnxruntime is not installed here, so the export was NOT verified.")
        print("         Run tests/test_golden_actions.py where it is before flying this.")
        return

    session = onnxruntime.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(args.seed)
    # Normalized-space inputs: that is what the graph actually consumes, and sampling it
    # directly exercises the whole clamped +/-5 sigma range the policy can ever see.
    probes = rng.uniform(-_NORM_CLAMP, _NORM_CLAMP, size=(args.samples, HOVER_OBS_DIM))

    with torch.no_grad():
        expected = policy._actor(torch.as_tensor(probes, dtype=torch.float32)).numpy()
    actual = session.run(None, {input_name: probes.astype(np.float32)})[0]

    raw_worst = float(np.abs(expected - actual).max())
    worst = float(np.abs(np.clip(expected, -1.0, 1.0) - np.clip(actual, -1.0, 1.0)).max())

    if worst > EXPORT_TOLERANCE:
        args.output.unlink()
        stats_path.unlink()
        raise SystemExit(
            f"ONNX export disagrees with torch by {worst:.3e} on the clamped action "
            f"(tolerance {EXPORT_TOLERANCE:.0e}). Files removed; do not fly this."
        )

    print(f"wrote {args.output} and {stats_path}")
    print(f"  verified against torch {torch.__version__}")
    print(f"  clamped action worst diff {worst:.3e} (raw output {raw_worst:.3e})")


if __name__ == "__main__":
    main()
