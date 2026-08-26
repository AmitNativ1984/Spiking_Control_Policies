#!/usr/bin/env python3
"""Publish a checkpoint to the flight repo: export the graph and record the golden, together.

WHY THIS IS ONE COMMAND. Exporting the graph and recording the golden actions are two
independent readings of the same checkpoint, and that independence is the point -- the
parity test is only meaningful because neither was derived from the other. But running them
as two commands means they can be run with DIFFERENT inputs, and that failure is invisible:
each artifact is internally consistent, nothing looks broken, and the parity test quietly
starts comparing two different policies and reporting the difference as numerical drift.

That has already happened once in this repo (see test_artifacts_share_one_checkpoint). The
artifacts now carry the SHA-256 of the checkpoint AND, for a spiking policy, of the config,
so the flight suite detects it. This script removes the opportunity instead: one checkpoint,
one config, both artifacts, or nothing.

THE CONFIG IS DERIVED, NOT ASKED FOR. For --arch snn the run's config.yaml is the only
source of `num_steps`, which appears in no weight shape (see snn_checkpoint.py). Since a
checkpoint always lives at <run>/nn/<name>.pth, the right config is always <run>/config.yaml
and this script finds it. Passing --config is possible but should be rare; the derived path
is the one that cannot be stale.

USAGE
-----
    python -m deploy.publish --arch snn \
        --checkpoint runs/f450_hover_snn/<run>/nn/last_....pth \
        --policy-dir <sail-uav-core>/libs/control-policy-api/policies/hover_snn

    python -m deploy.publish \
        --checkpoint runs/f450_hover/<run>/nn/f450_hover.pth \
        --policy-dir <sail-uav-core>/libs/control-policy-api/policies/hover

Then, in the FLIGHT container: `pytest libs/control-policy-api`. Verifying here proves
almost nothing -- it is the deployment environment's arithmetic that is in question.
"""

import argparse
import sys
from pathlib import Path

from . import export_onnx, record_golden

# The names OnnxHoverPolicy and the flight test expect inside a policy directory.
GRAPH_NAME = "hover.onnx"
GOLDEN_NAME = "hover_golden.npz"


def derive_config(checkpoint: Path) -> Path:
    """<run>/nn/<name>.pth -> <run>/config.yaml, the config frozen by the runner.

    Raises rather than falling back to a cfg/*.yaml: that file drifts independently of the
    checkpoints trained from it, and for a spiking policy a stale one silently changes the
    arithmetic.
    """
    run_dir = checkpoint.resolve().parent.parent
    config = run_dir / "config.yaml"
    if not config.is_file():
        raise SystemExit(
            f"no config.yaml in {run_dir}.\n"
            "The runner writes it on --train, but runs from before that change have none. "
            "Reconstruct it from the cfg/*.yaml the run used and put it there -- the "
            "exporter checks it against 18 hyperparameters in the weights, so a wrong one "
            "is refused rather than shipped. Or pass --config explicitly."
        )
    return config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the ONNX graph and record the golden actions from one checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--policy-dir", required=True, type=Path,
                        help="destination directory, e.g. "
                             "<sail-uav-core>/libs/control-policy-api/policies/hover_snn")
    parser.add_argument("--arch", choices=("ann", "snn"), default="ann",
                        help="dense actor (default) or PopSAN spiking actor")
    parser.add_argument("--config", type=Path, default=None,
                        help="--arch snn: override the derived <run>/config.yaml. "
                             "Rarely correct; the derived path cannot be stale.")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")
    if args.arch == "ann" and args.config is not None:
        raise SystemExit("--config applies to --arch snn only")

    shared = ["--arch", args.arch,
              "--checkpoint", str(args.checkpoint),
              "--samples", str(args.samples),
              "--seed", str(args.seed)]
    if args.arch == "snn":
        config = args.config if args.config is not None else derive_config(args.checkpoint)
        if not config.is_file():
            raise SystemExit(f"no config at {config}")
        shared += ["--config", str(config)]
        print(f"[publish] config: {config}")

    graph = args.policy_dir / GRAPH_NAME
    golden = args.policy_dir / GOLDEN_NAME

    # Export first: it verifies the graph against torch and deletes its own output on
    # disagreement, so a failure here leaves no half-published policy behind.
    print(f"\n[publish] 1/2 exporting graph -> {graph}")
    export_onnx.main(shared + ["--output", str(graph)])

    print(f"\n[publish] 2/2 recording golden -> {golden}")
    record_golden.main(shared + ["--output", str(golden)])

    print(f"\n[publish] {args.policy_dir} is complete.")
    print("[publish] Now run `pytest libs/control-policy-api` IN THE FLIGHT CONTAINER.")


if __name__ == "__main__":
    sys.exit(main())
