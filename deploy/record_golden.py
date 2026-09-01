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

    python -m deploy.record_golden --task navigation \
        --checkpoint runs/f450_nav_ann/<run>/nn/last_....pth \
        --output <sail-uav-core>/libs/control-policy-api/policies/navigation/navigation_golden.npz

A navigation recording also carries the DEPTH IMAGES it was made from, and the latents they
produced. The flight side has no way to regenerate either -- the encoder is a separate graph
and numpy makes no promise that a Generator stream survives four releases -- so the inputs
travel with the outputs.

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
from .navigation import NavigationPolicy, nav_task_config

NUM_SAMPLES = 256

#: Fewer for navigation, because each sample carries a 180 x 320 float32 depth image --
#: 230 kB apiece, against 100 bytes for a hover sample. 32 images span the four probe
#: families several times over; 256 would be a 59 MB artifact in a git repo to prove the
#: same thing.
NUM_SAMPLES_NAV = 32


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_states_nav(rng, count):
    """As sample_states, over the navigation task's wider world, and WITH depth images.

    The env is 20 x 20 m rather than a hover box, and the speed penalty only bites above
    5 m/s, so both position and velocity range further. The depth images come from
    export_vae_onnx's probe generator -- the same four families, so the golden exercises the
    encoder on the structures it was checked against rather than on a second invention.
    """
    from .export_vae_onnx import probe_images

    images = probe_images(rng, count)
    states, targets = [], []
    for image in images:
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        states.append(
            DroneState(
                position=rng.uniform(-8.0, 8.0, size=3),
                velocity_world=rng.normal(scale=2.0, size=3),
                quat=quat,
                angvel_body=rng.normal(scale=0.8, size=3),
                stamp_us=0,
                depth=image,
            )
        )
        targets.append(rng.uniform(-10.0, 10.0, size=3))
    return states, targets


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
    parser.add_argument("--task", choices=("hover", "navigation"), default="hover",
                        help="which observation layout to record (16-D or 49-D)")
    parser.add_argument("--arch", choices=("ann", "snn"), default="ann",
                        help="dense actor (default) or PopSAN spiking actor")
    parser.add_argument("--config", type=Path, default=None,
                        help="--arch snn: the run's frozen config.yaml, beside nn/")
    parser.add_argument("--vae-checkpoint", type=Path, default=None,
                        help="--task navigation: the DepthVAE .pth. Defaults to the one "
                             "the navigation task config names.")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)

    import torch  # imported late so the module docstring is readable without it

    if args.task == "navigation":
        if args.arch != "ann":
            raise SystemExit("the navigation policy has no spiking variant yet")
        if args.config is not None:
            raise SystemExit("--config applies to --arch snn only")
        if args.vae_checkpoint is None:
            args.vae_checkpoint = Path(nav_task_config().vae_config.model_file)
            print(f"--vae-checkpoint defaulted to the task config's: {args.vae_checkpoint}")
        policy = NavigationPolicy(str(args.checkpoint), args.vae_checkpoint)
    elif args.arch == "ann":
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
    if args.samples is None:
        args.samples = NUM_SAMPLES_NAV if args.task == "navigation" else NUM_SAMPLES

    rng = np.random.default_rng(args.seed)
    sampler = sample_states_nav if args.task == "navigation" else sample_states
    states, targets = sampler(rng, args.samples)

    observations, actions, prev_actions = [], [], []
    prev_actions_transformed = []
    for state, target in zip(states, targets):
        # Drive prev_action from the policy's own history, exactly as flight does: each
        # action becomes the next observation's [12:16]. A fresh reset per sample would
        # leave that channel at zero throughout and never exercise it.
        prev_actions.append(policy.prev_action)
        if args.task == "navigation":
            # The RAW previous action is what BasePolicy latches, but it is NOT what the
            # navigation observation carries -- [13:17] holds the transformed command, in
            # radians. Recording both makes the recording self-describing rather than
            # leaving a reader to infer which convention obs[13:17] follows.
            prev_actions_transformed.append(policy.prev_action_transformed)
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
        task=args.task,
        **({"config_sha256": checkpoint_digest(args.config)} if args.arch == "snn" else {}),
        # The depth images are recorded, not regenerated from a seed on the other side:
        # numpy makes no promise that a Generator stream is identical across versions, and
        # the two containers are four numpy releases apart. The latents are what is being
        # compared, so their INPUTS have to travel with them.
        **(
            {
                "depths": np.array([s.depth for s in states], dtype=np.float32),
                "latents": np.array([o[17:] for o in observations], dtype=np.float32),
                "prev_actions_transformed": np.array(prev_actions_transformed),
                "vae_checkpoint_sha256": checkpoint_digest(args.vae_checkpoint),
            }
            if args.task == "navigation"
            else {}
        ),
        recorded_with=(
            f"python {platform.python_version()} torch {torch.__version__} "
            f"numpy {np.__version__}"
        ),
    )
    print(f"wrote {args.output} ({args.samples} samples)")
    print(f"  task {args.task}, arch {args.arch}",
          f"num_steps {policy.num_steps}" if args.arch == "snn" else "")
    print(f"  torch {torch.__version__}, numpy {np.__version__}")
    print(f"  action range [{np.min(actions):.4f}, {np.max(actions):.4f}]")


if __name__ == "__main__":
    main()
