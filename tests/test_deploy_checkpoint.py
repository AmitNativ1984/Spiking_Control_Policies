"""The training-side checkpoint loader, and the golden file it produces.

The other half of this pair lives in the flight repo
(sail-uav-core/libs/control-policy-api/tests/test_golden_actions.py), which replays the same
golden file with no torch present. Between them they close the loop: this side asserts torch
reproduces what was recorded, that side asserts onnxruntime does.

Skipped unless the deployment package is installed and a checkpoint is pointed at:

    CONTROL_POLICY_CHECKPOINT=runs/f450_hover/.../last_....pth pytest tests/test_deploy_checkpoint.py
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip(
    "control_policy_api", reason="pip install -e <sail-uav-core>/libs/control-policy-api"
)
from control_policy_api.base import DroneState  # noqa: E402

from deploy.hover import HoverPolicy  # noqa: E402

# Matches the flight-side test: three float32 GEMMs summed in a different order differ in the
# last couple of ULPs, and nothing larger than that is round-off.
ACTION_TOLERANCE = 1e-5


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def checkpoint():
    path = os.environ.get("CONTROL_POLICY_CHECKPOINT")
    if not path:
        pytest.skip("set CONTROL_POLICY_CHECKPOINT to the .pth under test")
    path = Path(path)
    if not path.exists():
        pytest.skip(f"checkpoint not found: {path}")
    return path


@pytest.fixture(scope="module")
def golden(checkpoint):
    """The recorded pairs, and proof they came from THIS checkpoint.

    A golden file recorded from a different checkpoint would compare two unrelated networks
    and fail confusingly -- or, if they happened to be close, pass.
    """
    path = os.environ.get("CONTROL_POLICY_GOLDEN")
    if not path:
        pytest.skip("set CONTROL_POLICY_GOLDEN to the recorded hover_golden.npz")
    path = Path(path)
    if not path.exists():
        pytest.skip(f"golden file not found: {path}")

    data = np.load(path, allow_pickle=False)
    expected = str(data["checkpoint_sha256"])
    actual = checkpoint_digest(checkpoint)
    if actual != expected:
        pytest.fail(
            f"golden file was recorded from a different checkpoint\n"
            f"  golden sha256    {expected}\n"
            f"  checkpoint under test {actual}\n"
            f"Re-record with `python -m deploy.record_golden`."
        )
    return data


def test_checkpoint_loads_and_has_frozen_statistics(checkpoint):
    """rl_games' input normalizer must survive the load.

    Its absence is the classic silent deployment failure: the network still runs, on inputs
    ranging over metres and rad/s that it was trained to see whitened.
    """
    policy = HoverPolicy(str(checkpoint))
    assert policy._obs_mean.shape == (policy.obs_dim,)
    assert policy._obs_std.shape == (policy.obs_dim,)
    assert np.all(policy._obs_std > 0.0), "a zero-variance channel would divide by ~0"
    # Gravity-in-body-z at level hover. Anything else means the checkpoint was trained
    # against a different observation layout than control_policy_api builds.
    assert policy._obs_mean[5] == pytest.approx(-0.985, abs=0.05)


def test_torch_reproduces_the_golden_actions(checkpoint, golden):
    """Guards the recorder itself: the golden file must still describe this checkpoint."""
    policy = HoverPolicy(str(checkpoint))

    worst = 0.0
    for i in range(len(golden["actions"])):
        state = DroneState(
            position=golden["positions"][i],
            velocity_world=golden["velocities"][i],
            quat=golden["quats"][i],
            angvel_body=golden["angvels"][i],
            stamp_us=0,
        )
        # Replay the recorded history rather than the policy's own, so one divergent sample
        # cannot cascade into every sample after it and hide where it started.
        policy._prev_action = golden["prev_actions"][i].astype(np.float32)
        action = policy.act(state, golden["targets"][i])
        worst = max(worst, float(np.abs(action - golden["actions"][i]).max()))

    assert worst < ACTION_TOLERANCE, f"torch drifted by {worst} from the recorded actions"
