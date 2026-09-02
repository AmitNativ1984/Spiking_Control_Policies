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

BOTH ARCHITECTURES PRODUCE THE SAME KIND OF GRAPH. A spiking actor exports through the stock
tracer with no special handling, because `forward` resets its membrane state on every call:
no state crosses the graph boundary, so tracing UNROLLS the spiking loop into a static graph
with no ONNX Loop and no hidden state for the flight side to plumb. One consequence is worth
naming -- `num_steps` becomes structural. There is no input, attribute or knob downstream
that could change it, and the node counts below are the proof:

    num_steps    1      2      3      5      8     16
    nodes      155    276    394    630    984   1928

The flip side is that a WRONG num_steps at export time is invisible: it produces a
well-formed graph of the wrong depth, and nothing downstream can tell. Note carefully what
does and does not defend against that:

  * `verify_timesteps` below counts the unrolled GEMMs, but it counts them against the same
    config that built the network, so the two agree by construction. It catches a tracer
    that stopped unrolling (a future torch emitting an ONNX Loop, constant folding
    collapsing a layer) -- NOT a config that lies about the checkpoint.
  * The golden recording does not catch it either, for the same reason: record_golden runs
    the same policy object, built from the same config.

The only thing that establishes num_steps is snn_checkpoint.verify_config_against_weights,
which pins the config against 18 hyperparameters that ARE recoverable from the state_dict.
A config agreeing on all 18 is the config this checkpoint trained under. That is why
--config must point at the run directory's own config.yaml and not at cfg/.

What IS checked across the two commands is that they used the same config file: both
artifacts record its SHA-256, and the flight suite compares them.

USAGE
-----
    python -m deploy.export_onnx \
        --checkpoint /path/to/last_....pth \
        --output <sail-uav-core>/libs/control-policy-api/policies/hover/hover.onnx

    python -m deploy.export_onnx --arch snn \
        --checkpoint runs/f450_hover_snn/<run>/nn/last_....pth \
        --config    runs/f450_hover_snn/<run>/config.yaml \
        --output <sail-uav-core>/libs/control-policy-api/policies/hover_snn/hover.onnx

    python -m deploy.export_onnx --task navigation \
        --checkpoint runs/f450_nav_ann/<run>/nn/last_....pth \
        --output <sail-uav-core>/libs/control-policy-api/policies/navigation/navigation.onnx

The navigation actor is only HALF of that policy: it consumes a 49-D observation whose last
32 channels come from the DepthVAE, which is a second graph with its own exporter
(deploy/export_vae_onnx.py). Both must land in the same directory, and record_golden ties
them together by recording depth frames through one into the other.
"""

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import torch

from .hover import HoverPolicy, SnnHoverPolicy
from .navigation import NavigationPolicy
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


# One linear layer per spiking layer, per timestep. The encoder contributes no GEMM (it is
# elementwise plus a threshold) and the decoder is a grouped Conv, so this counts exactly the
# unrolled spiking core.
GEMMS_PER_TIMESTEP = 3


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Kept under the old name: record_golden and the flight suite both speak "checkpoint_sha256".
checkpoint_digest = file_digest


def build_policy(args):
    """The policy whose `_actor` gets exported.

    The navigation policy is built WITHOUT its VAE. Only `_actor` is exported here, and
    loading a half-gigabyte encoder to not use it would make this command need a GPU it has
    no work for. The encoder is a separate graph with a separate exporter -- see
    deploy/export_vae_onnx.py and the two-graph note in control_policy_api.onnx.
    """
    if args.task == "navigation":
        if args.arch != "ann":
            raise SystemExit("the navigation policy has no spiking variant yet")
        if args.config is not None:
            raise SystemExit("--config applies to --arch snn only")
        return NavigationPolicy(str(args.checkpoint))

    if args.arch == "ann":
        if args.config is not None:
            raise SystemExit("--config applies to --arch snn only")
        return HoverPolicy(str(args.checkpoint))

    if args.config is None:
        raise SystemExit(
            "--arch snn requires --config: num_steps is not recoverable from the "
            "checkpoint. Pass the config.yaml in the checkpoint's own run directory."
        )
    return SnnHoverPolicy(str(args.checkpoint), args.config)


def verify_timesteps(model_path: Path, num_steps: int) -> int:
    """Check the graph was unrolled to the depth the config asked for.

    Structural only -- see this module's docstring for what this does NOT prove. It fires
    when tracing stops producing a flat unrolled graph, which would mean the export no
    longer has the property the whole deployment path is built on. Counts MatMul beside
    Gemm because constant folding is free to emit either.
    """
    import onnx

    graph = onnx.load(str(model_path)).graph
    found = sum(1 for node in graph.node if node.op_type in ("Gemm", "MatMul"))
    expected = GEMMS_PER_TIMESTEP * num_steps
    if found != expected:
        raise SystemExit(
            f"exported graph has {found} GEMM/MatMul nodes, expected {expected} "
            f"({GEMMS_PER_TIMESTEP} per timestep x num_steps={num_steps}). The graph does "
            "not have the depth the config asked for; do not fly this."
        )
    return found


def stamp_metadata(model_path: Path, entries: dict) -> None:
    """Write provenance into the graph's own metadata_props.

    Puts num_steps in the artifact rather than only in a sidecar, so a graph that gets
    separated from its .npz can still say what it is.
    """
    import onnx

    model = onnx.load(str(model_path))
    for key, value in entries.items():
        entry = model.metadata_props.add()
        entry.key = str(key)
        entry.value = str(value)
    onnx.save(model, str(model_path))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=("hover", "navigation"), default="hover",
                        help="which observation layout the actor consumes (16-D or 49-D)")
    parser.add_argument("--arch", choices=("ann", "snn"), default="ann",
                        help="dense actor (default) or PopSAN spiking actor")
    parser.add_argument("--config", type=Path, default=None,
                        help="--arch snn: the run's frozen config.yaml, beside nn/")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)

    policy = build_policy(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    provenance = {"arch": args.arch, "task": args.task, "obs_dim": policy.obs_dim}
    if args.arch == "snn":
        provenance.update(
            num_steps=policy.num_steps,
            pop_dim=policy.pop_dim,
            reset_delay=policy.reset_delay,
            run_config=str(args.config),
            config_sha256=file_digest(args.config),
        )

    # Export the NETWORK ONLY. Normalization and the action clamp stay in Python -- they are
    # the same numpy on both sides, so freezing them into the graph would buy nothing and
    # make the statistics harder to inspect.
    torch.onnx.export(
        policy._actor,
        torch.zeros(1, policy.obs_dim, dtype=torch.float32),
        str(args.output),
        input_names=["normalized_observation"],
        output_names=["action"],
        # Batch stays dynamic so the same graph can be used for offline batch evaluation,
        # even though flight always sends one row.
        dynamic_axes={"normalized_observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    if args.arch == "snn":
        provenance["gemm_nodes"] = verify_timesteps(args.output, policy.num_steps)
        provenance["verified_hyperparameters"] = ",".join(policy.verified_hyperparameters)
    stamp_metadata(args.output, provenance)

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
        **{key: str(value) for key, value in provenance.items()},
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
    probes = rng.uniform(-_NORM_CLAMP, _NORM_CLAMP, size=(args.samples, policy.obs_dim))

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
    print(f"  task {args.task} (obs_dim {policy.obs_dim}), arch {args.arch}")
    print(f"  verified against torch {torch.__version__}")
    print(f"  clamped action worst diff {worst:.3e} (raw output {raw_worst:.3e})")
    if args.arch == "snn":
        print(f"  num_steps {policy.num_steps} confirmed by {provenance['gemm_nodes']} "
              f"GEMM nodes; {len(policy.verified_hyperparameters)} hyperparameters "
              f"matched against the weights")


if __name__ == "__main__":
    main()
