#!/usr/bin/env python3
"""Record golden observation/action pairs from the TRAINING environment.

WHY THIS EXISTS
---------------
The training container and the flight container cannot run the same torch. Training is
Python 3.8 with torch 1.14.0a0+410ce96 -- an NVIDIA NGC build that was never released and
exists on no package index -- while ROS 2 Jazzy is Python 3.12, whose earliest torch is 2.2
and earliest numpy 1.26. There is no set of pins that makes those identical.

So instead of asserting the versions match, we assert the NUMBERS match. Run this once in
the aerial_gym container to freeze what the policy does; the test alongside it then re-runs
those same inputs wherever the policy is deployed and checks the outputs still agree.

That is the stronger claim anyway. Identical version strings would only be a proxy for
identical arithmetic, and this measures the arithmetic.

USAGE
-----
    python -m deploy.record_golden \
        --checkpoint /workspaces/aerial_gym_docker/runs/f450_hover/.../last_....pth \
        --output <sail-uav-core>/libs/control-policy-api/policies/hover/hover_golden.npz

    python -m deploy.record_golden --arch snn \
        --checkpoint runs/f450_hover_snn/<run>/nn/last_....pth \
        --config    runs/f450_hover_snn/<run>/config.yaml \
        --output <sail-uav-core>/libs/control-policy-api/policies/hover_snn/hover_golden.npz

For a spiking policy the recording also carries the SHA-256 of the --config it used, and the
flight suite requires it to match the graph's. That is what makes the two commands provably
one policy: `num_steps` lives only in that file, so exporting the graph from one config and
recording the golden from another would otherwise compare two different networks and report
it as numerical drift. (Neither file can tell you the config is TRUE for this checkpoint --
snn_checkpoint.verify_config_against_weights is what establishes that, at load time.)

Re-record whenever the checkpoint changes, and re-export the graph from the SAME .pth --
both files carry the checkpoint's SHA-256, and test_artifacts_share_one_checkpoint fails on
a mismatched pair rather than letting the parity test report it as numerical drift.
"""

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np

from control_policy_api.base import DroneState
from .hover import HoverPolicy, SnnHoverPolicy

NUM_SAMPLES = 256


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_states(rng, count):
    """Random but PLAUSIBLE flight states.

    Plausible matters: the checkpoint's input normalizer clamps at +/-5 sigma, so garbage
    inputs would all saturate to the same place and the comparison would pass no matter how
    wrong the network was. These ranges sit inside the training distribution.
    """
    states, targets = [], []
    for _ in range(count):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        states.append(
            DroneState(
                position=rng.uniform(-4.0, 4.0, size=3),
                velocity_world=rng.normal(scale=1.5, size=3),
                quat=quat,
                angvel_body=rng.normal(scale=0.8, size=3),
                stamp_us=0,
            )
        )
        targets.append(rng.uniform(-4.0, 4.0, size=3))
    return states, targets


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arch", choices=("ann", "snn"), default="ann",
                        help="dense actor (default) or PopSAN spiking actor")
    parser.add_argument("--config", type=Path, default=None,
                        help="--arch snn: the run's frozen config.yaml, beside nn/")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)

    import torch  # imported late so the module docstring is readable without it

    if args.arch == "ann":
        if args.config is not None:
            raise SystemExit("--config applies to --arch snn only")
        policy = HoverPolicy(str(args.checkpoint))
    else:
        if args.config is None:
            raise SystemExit(
                "--arch snn requires --config: num_steps is not recoverable from the "
                "checkpoint. Pass the config.yaml in the checkpoint's own run directory."
            )
        policy = SnnHoverPolicy(str(args.checkpoint), args.config)
    rng = np.random.default_rng(args.seed)
    states, targets = sample_states(rng, args.samples)

    observations, actions, prev_actions = [], [], []
    for state, target in zip(states, targets):
        # Drive prev_action from the policy's own history, exactly as flight does: each
        # action becomes the next observation's [12:16]. A fresh reset per sample would
        # leave that channel at zero throughout and never exercise it.
        prev_actions.append(policy.prev_action)
        observations.append(policy.build_observation(state, target))
        actions.append(policy.act(state, target))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=np.array([s.position for s in states]),
        velocities=np.array([s.velocity_world for s in states]),
        quats=np.array([s.quat for s in states]),
        angvels=np.array([s.angvel_body for s in states]),
        targets=np.array(targets),
        prev_actions=np.array(prev_actions),
        observations=np.array(observations),
        actions=np.array(actions),
        checkpoint_sha256=checkpoint_digest(args.checkpoint),
        arch=args.arch,
        **({"config_sha256": checkpoint_digest(args.config)} if args.arch == "snn" else {}),
        recorded_with=(
            f"python {platform.python_version()} torch {torch.__version__} "
            f"numpy {np.__version__}"
        ),
    )
    print(f"wrote {args.output} ({args.samples} samples)")
    print(f"  arch {args.arch}", f"num_steps {policy.num_steps}" if args.arch == "snn" else "")
    print(f"  torch {torch.__version__}, numpy {np.__version__}")
    print(f"  action range [{np.min(actions):.4f}, {np.max(actions):.4f}]")


if __name__ == "__main__":
    main()
