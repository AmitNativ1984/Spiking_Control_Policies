#!/usr/bin/env python3
"""Export the DepthVAE ENCODER to ONNX, so the flight container needs no torch for vision.

The companion of export_onnx.py, for the other half of the navigation policy. The navigation
observation is 17 state channels plus 32 DepthVAE latents, and those latents cost far more
arithmetic than the actor does -- a few hundred MFLOPs against 35k MACs -- so they are a
separate graph that can sit on a separate execution provider and run at the camera's rate
rather than the controller's.

    depth_vae.onnx    normalized depth image in, 32-D mu out

WHAT IS AND IS NOT IN THE GRAPH
-------------------------------
IN:  the encoder's convolution stack, its FC head, AND the mu slice.
OUT: the preprocessing -- resize, clamp, invert -- which stays in numpy
     (control_policy_api/depth.py) because it is a handful of array operations that want to
     be readable and testable, not frozen.

THE mu SLICE IS INSIDE THE GRAPH ON PURPOSE. The encoder emits [B, 2*latent_dim]: mu
concatenated with logvar. Taking the wrong half yields a well-formed 32-vector, a
well-formed observation and a policy steering on noise variance. That is a decision with
exactly one correct answer, so it is settled here, once, rather than restated at every
consumer.

DECODER NOT EXPORTED. Inference is deterministic and uses mu only; the decoder exists to
train the encoder and has no role after that.

IT VERIFIES BEFORE IT WRITES, like export_onnx.py: the graph is compared against torch over
depth images spanning the encoder's whole input range, and nothing is written if they
disagree. The probes are IMAGES, not uniform noise in latent space -- a convolution stack's
disagreement with its export is spatial, and uniform noise exercises none of the structure
(edges, planes, flat sky) the encoder actually responds to.

USAGE
-----
    python -m deploy.export_vae_onnx \\
        --output <sail-uav-core>/libs/control-policy-api/policies/navigation/depth_vae.onnx

--vae-checkpoint defaults to the one the navigation task config names, which is the only
way to be sure the exported encoder is the encoder the policy was trained against.
"""

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import torch

from control_policy_api.depth import (
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    preprocess_depth,
)
from control_policy_api.observations_nav import NAV_LATENT_DIM

from .export_onnx import OPSET, file_digest, stamp_metadata
from .navigation import latent_dim_of, nav_task_config

# Looser than export_onnx.py's 1e-5, and for a structural reason rather than a hopeful one.
# That threshold covers two 256-wide GEMMs; this graph is nine convolutions deep with 256
# channels at its widest, so float32 accumulation has an order of magnitude more room to
# reorder. The latent is consumed by a network whose own input normalizer divides it by a
# per-channel standard deviation of order 1, so 1e-4 on a latent is far below anything the
# policy can resolve -- and still tight enough to catch a dropped layer or a wrong slice.
EXPORT_TOLERANCE = 1e-4


class _EncoderMu(torch.nn.Module):
    """The encoder with the mu slice welded on. See the module docstring."""

    def __init__(self, encoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim

    def forward(self, image):
        return self.encoder(image)[:, : self.latent_dim]


def probe_images(rng, count):
    """Depth images that look like the ones the encoder was trained on.

    Structured, not uniform noise. The encoder is a convolution stack: its response is to
    edges, planes and open space, and an export defect (a dropped layer, a transposed
    kernel, a wrong padding) shows up in how it treats STRUCTURE. Uniform per-pixel noise is
    nearly flat to every one of those and would let a broken graph pass.

    Four families, matching what a forest flight actually presents: open sky, a ground
    plane, vertical trunks, and a wall at close range.
    """
    images = []
    for index in range(count):
        family = index % 4
        depth = np.full((TARGET_HEIGHT, TARGET_WIDTH), rng.uniform(5.0, 12.0), dtype=np.float32)

        if family == 1:  # a receding ground plane across the lower half
            rows = np.linspace(1.0, 9.0, TARGET_HEIGHT, dtype=np.float32)[:, None]
            depth = np.repeat(rows, TARGET_WIDTH, axis=1)
        elif family == 2:  # a few vertical trunks
            for _ in range(rng.integers(1, 5)):
                left = int(rng.integers(0, TARGET_WIDTH - 20))
                width = int(rng.integers(6, 40))
                depth[:, left : left + width] = rng.uniform(0.8, 6.0)
        elif family == 3:  # a wall, close
            depth[:] = rng.uniform(0.3, 2.5)

        # A little sensor grain on top, so no probe is perfectly piecewise-constant.
        depth = depth + rng.normal(scale=0.03, size=depth.shape).astype(np.float32)
        images.append(np.clip(depth, 0.05, 30.0))
    return images


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=None,
        help="the DepthVAE .pth. Defaults to the one the navigation task config names.",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.vae_checkpoint is None:
        args.vae_checkpoint = Path(nav_task_config().vae_config.model_file)
        print(f"--vae-checkpoint defaulted to the task config's: {args.vae_checkpoint}")

    if not args.vae_checkpoint.exists():
        raise SystemExit(f"no such VAE checkpoint: {args.vae_checkpoint}")

    latent_dim = latent_dim_of(args.vae_checkpoint)
    if latent_dim != NAV_LATENT_DIM:
        raise SystemExit(
            f"{args.vae_checkpoint} has latent_dim={latent_dim}; the navigation "
            f"observation reserves {NAV_LATENT_DIM} channels"
        )

    from vae_depth.model import DepthVAE

    vae = DepthVAE(latent_dim=latent_dim)
    vae.load_state_dict(
        torch.load(str(args.vae_checkpoint), map_location=args.device)["model_state_dict"]
    )
    module = _EncoderMu(vae.encoder, latent_dim).to(args.device).eval()
    for parameter in module.parameters():
        parameter.requires_grad = False

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # The batch axis stays dynamic for offline evaluation; H and W do NOT. The encoder's FC
    # head is Linear(fc_channel_dim * 6 * 10, 512), so 180 x 320 is the only input its
    # convolution stack lands correctly on -- a dynamic spatial axis would export a graph
    # that accepts shapes it cannot actually compute.
    torch.onnx.export(
        module,
        torch.zeros(1, 1, TARGET_HEIGHT, TARGET_WIDTH, dtype=torch.float32),
        str(args.output),
        input_names=["normalized_depth"],
        output_names=["latent"],
        dynamic_axes={"normalized_depth": {0: "batch"}, "latent": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
    )

    stamp_metadata(
        args.output,
        {
            "kind": "depth_vae_encoder",
            "latent_dim": latent_dim,
            "input_height": TARGET_HEIGHT,
            "input_width": TARGET_WIDTH,
            "max_depth_m": MAX_DEPTH_M,
            "min_depth_m": MIN_DEPTH_M,
            "vae_checkpoint_sha256": file_digest(args.vae_checkpoint),
            "exported_with": (
                f"python {platform.python_version()} torch {torch.__version__} "
                f"numpy {np.__version__} opset {OPSET}"
            ),
        },
    )

    # --- verify, and refuse to ship a graph that disagrees with torch --------------------
    try:
        import onnxruntime
    except ImportError:
        print(f"wrote {args.output}")
        print("WARNING: onnxruntime is not installed here, so the export was NOT verified.")
        return

    session = onnxruntime.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    rng = np.random.default_rng(args.seed)
    tensors = np.concatenate([preprocess_depth(image) for image in probe_images(rng, args.samples)])

    with torch.no_grad():
        expected = module(torch.as_tensor(tensors, device=args.device)).cpu().numpy()
    actual = session.run(None, {input_name: tensors})[0]

    worst = float(np.abs(expected - actual).max())
    if worst > EXPORT_TOLERANCE:
        args.output.unlink()
        raise SystemExit(
            f"ONNX export disagrees with torch by {worst:.3e} on the latent "
            f"(tolerance {EXPORT_TOLERANCE:.0e}). File removed; do not fly this."
        )

    print(f"wrote {args.output}")
    print(f"  encoder from {args.vae_checkpoint}")
    print(f"  latent_dim {latent_dim}, input 1 x 1 x {TARGET_HEIGHT} x {TARGET_WIDTH}")
    print(f"  verified against torch {torch.__version__} on {args.samples} depth images")
    print(f"  worst latent diff {worst:.3e}")
    print(f"  latent range [{expected.min():.3f}, {expected.max():.3f}]")


if __name__ == "__main__":
    main()
