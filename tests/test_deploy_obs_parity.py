"""The deployed observation must be the same function as the trained one.

control_policy_api reimplements hover_task.process_obs_for_task in numpy, because the
deployment target is a ROS 2 container with no Isaac Gym in it. A reimplementation is a
place for the two to drift apart silently -- the vectors keep their shape, the policy keeps
producing actions, and the actions are wrong. This test pins them together.

It compares against aerial_gym's own math functions on random states rather than against a
live simulation: the arithmetic is what can drift, and a real env build would make the check
too slow to run often.

IT LIVES HERE, NOT IN sail-uav-core. This is the only test of the pair that needs aerial_gym,
and the flight repo is deliberately free of isaacgym and aerial_gym -- it receives a raw .pth
and nothing else. So the comparison runs on this side, and its result travels to the flight
side as the golden vectors recorded by control-policy-api's tools/record_golden.py.

Run it whenever either side of the pair changes.
"""

import numpy as np
import pytest
import torch
from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_rotate_inverse,
    vehicle_frame_quat_from_quat,
)

# The deployment package. Installed editable from the sail-uav-core monorepo; skipped rather
# than failed when it is absent, so this repo still tests standalone.
control_policy_api = pytest.importorskip(
    "control_policy_api", reason="pip install -e <sail-uav-core>/libs/control-policy-api"
)
from control_policy_api.base import DroneState  # noqa: E402
from control_policy_api.observations import HOVER_OBS_DIM, build_hover_observation  # noqa: E402

NUM_SAMPLES = 200
TOLERANCE = 1e-9


def random_state(rng):
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return {
        "position": rng.normal(scale=3.0, size=3),
        "velocity": rng.normal(scale=2.0, size=3),
        "quat": quat,
        "angvel": rng.normal(scale=1.0, size=3),
        "target": rng.normal(scale=3.0, size=3),
        "prev_action": rng.uniform(-1.0, 1.0, size=4),
    }


def reference_observation(sample):
    """The training task's computation, using aerial_gym's own tensor helpers.

    Mirrors task/hover_task.py:749-816 line for line, including the 1e-6 that the task adds
    inside the gravity normalization.
    """
    position = torch.tensor(sample["position"], dtype=torch.float64).unsqueeze(0)
    velocity = torch.tensor(sample["velocity"], dtype=torch.float64).unsqueeze(0)
    quat = torch.tensor(sample["quat"], dtype=torch.float64).unsqueeze(0)
    target = torch.tensor(sample["target"], dtype=torch.float64).unsqueeze(0)
    gravity = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64)

    obs = torch.zeros((1, HOVER_OBS_DIM), dtype=torch.float64)
    obs[:, 0:3] = quat_apply_inverse(vehicle_frame_quat_from_quat(quat), target - position)
    gravity_body = quat_rotate_inverse(quat, gravity)
    obs[:, 3:6] = gravity_body / torch.linalg.norm(gravity_body + 1e-6, dim=-1, keepdim=True)
    obs[:, 6:9] = quat_rotate_inverse(quat, velocity)
    obs[:, 9:12] = torch.tensor(sample["angvel"], dtype=torch.float64)
    obs[:, 12:16] = torch.tensor(sample["prev_action"], dtype=torch.float64)
    return obs.squeeze(0).numpy()


def deployed_observation(sample):
    state = DroneState(
        position=sample["position"],
        velocity_world=sample["velocity"],
        quat=sample["quat"],
        angvel_body=sample["angvel"],
        stamp_us=0,
    )
    return build_hover_observation(state, sample["target"], sample["prev_action"])


def test_observation_matches_training_task():
    rng = np.random.default_rng(20260816)
    worst = 0.0
    for _ in range(NUM_SAMPLES):
        sample = random_state(rng)
        difference = np.abs(reference_observation(sample) - deployed_observation(sample))
        worst = max(worst, difference.max())
    assert worst < TOLERANCE, f"deployed observation drifted from training by {worst}"


def test_gravity_channel_matches_the_trained_checkpoint_statistics():
    """A level vehicle must produce obs[5] ~ -1.

    Not a tautology: the trained checkpoint's own input normalizer has running_mean[5] =
    -0.9854, measured over 2.6e8 training samples. Any sign or axis error in the gravity
    channel shows up here as +1 or as a value on the wrong index, and would push the
    normalized input to roughly -40 sigma at every step of a real flight.
    """
    level = DroneState(
        position=np.zeros(3),
        velocity_world=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        angvel_body=np.zeros(3),
        stamp_us=0,
    )
    obs = build_hover_observation(level, np.zeros(3), np.zeros(4))
    assert obs[3] == pytest.approx(0.0, abs=1e-6)
    assert obs[4] == pytest.approx(0.0, abs=1e-6)
    assert obs[5] == pytest.approx(-1.0, abs=1e-5)


def test_position_error_is_yaw_only_and_ignores_tilt():
    """[0:3] must not respond to roll or pitch -- only to heading.

    This is the single distinction between the [0:3] and [6:9] frames, and the one a
    deployment is most likely to get wrong, because the two agree whenever the vehicle is
    near level, which is most of a hover.
    """
    target = np.array([1.0, 2.0, 3.0])
    reference = None
    for pitch in (0.0, 0.3, -0.3):
        quat = quat_from_euler_xyz(
            torch.tensor([0.2]), torch.tensor([pitch]), torch.tensor([0.7])
        ).squeeze(0).numpy()
        state = DroneState(
            position=np.zeros(3),
            velocity_world=np.zeros(3),
            quat=quat,
            angvel_body=np.zeros(3),
            stamp_us=0,
        )
        obs = build_hover_observation(state, target, np.zeros(4))
        if reference is None:
            reference = obs[0:3]
        else:
            assert np.allclose(obs[0:3], reference, atol=1e-12)
