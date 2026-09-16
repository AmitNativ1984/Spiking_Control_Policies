"""Unit tests for p_blind, and for the absence of the r_bearing term it replaces.

Drives the real _reward_progress against a stub `self`, so the arithmetic under test is
the shipped arithmetic rather than a reimplementation that can drift. Distances are held
equal across steps so r_progress contributes exactly zero and p_blind is the only thing
moving.

The property that motivates the whole change is test_reward_is_yaw_sensitive: before
p_blind, no term could tell a drone facing its direction of travel from one flying the
same path sideways, because n and v were rotated by the same yaw-only vehicle quaternion
and the yaw cancelled in r_bearing's dot product.
"""
import math
import types

import pytest
import torch

from task.attitude_navigation_task import NavigationWithObstaclesTask
from config.task_config.f450_attitude_navigation_task_config import task_config
from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig as CAM

EMA_KEYS = ["r_progress", "p_speed", "p_jerk", "p_action_mag", "p_blind", "p_fov"]

LAMBDA_BLIND = task_config.reward_parameters["lambda_blind"]


def _stub(**overrides):
    """A minimal `self` carrying only what _reward_progress reads."""
    params = dict(task_config.reward_parameters)
    params.update(overrides)

    stub = types.SimpleNamespace()
    stub.task_config = types.SimpleNamespace(
        reward_parameters=params, v_max=task_config.v_max
    )
    stub.device = "cpu"
    stub._ema_alpha = 0.02
    stub._reward_comp_ema = {k: 0.0 for k in EMA_KEYS}
    stub._action_scale = torch.tensor(
        [1.0, math.pi / 4, math.pi / 4, task_config.max_yaw_rate]
    )
    # p_fov geometry, derived from the same camera config the task reads at init.
    stub._half_h_fov = math.radians(CAM.horizontal_fov_deg) / 2.0
    stub._half_v_fov = math.atan(
        math.tan(stub._half_h_fov) * (CAM.height / CAM.width)
    )
    return stub


def _reward(stub, v, dist=5.0):
    """Evaluate _reward_progress with zero actions and zero net progress."""
    num = v.shape[0]
    d = torch.full((num,), dist)
    zeros = torch.zeros(num, 4)

    stub.obs_dict = {"robot_vehicle_linvel": v}
    stub.prev_dist = d
    stub.prev_action = zeros
    stub._get_dist_to_target = lambda: d

    mask = torch.ones(num, dtype=torch.bool)
    return NavigationWithObstaclesTask._reward_progress(stub, mask, zeros)


def _p_blind_only(v):
    """p_blind isolated: reward with lambda_blind set, minus reward with it zeroed."""
    return _reward(_stub(), v) - _reward(_stub(lambda_blind=0.0), v)


# Vehicle frame is yaw-aligned and FLU: x forward (nose), y left, z up.
FORWARD = torch.tensor([[3.0, 0.0, 0.0]])
SIDEWAYS = torch.tensor([[0.0, 3.0, 0.0]])
BACKWARDS = torch.tensor([[-3.0, 0.0, 0.0]])
HOVER = torch.tensor([[0.0, 0.0, 0.0]])
CLIMB = torch.tensor([[0.0, 0.0, 3.0]])


def test_flying_along_the_nose_is_free():
    assert _p_blind_only(FORWARD).item() == pytest.approx(0.0, abs=1e-6)


def test_sideways_costs_a_quarter_of_the_maximum():
    """misalignment = 0.5 at 90 deg, squared -> 0.25."""
    expected = -LAMBDA_BLIND * 3.0 * 0.25
    assert _p_blind_only(SIDEWAYS).item() == pytest.approx(expected, abs=1e-5)


def test_backwards_costs_the_maximum():
    """misalignment = 1.0 at 180 deg. Four times the sideways penalty, by the square."""
    expected = -LAMBDA_BLIND * 3.0 * 1.0
    assert _p_blind_only(BACKWARDS).item() == pytest.approx(expected, abs=1e-5)
    assert _p_blind_only(BACKWARDS).item() == pytest.approx(
        4.0 * _p_blind_only(SIDEWAYS).item(), abs=1e-5
    )


def test_hover_is_free_at_any_yaw():
    """Scaling by horizontal_speed is what lets the drone yaw freely at rest -- the
    dead-zone escape depends on turning being cheap when stationary."""
    assert _p_blind_only(HOVER).item() == pytest.approx(0.0, abs=1e-9)


def test_pure_vertical_climb_is_finite_and_free():
    """v_x = v_y = 0 puts a zero in the denominator; the 1e-6 guard must hold, and
    v_z is excluded by design because yaw cannot correct vertical blindness."""
    r = _reward(_stub(), CLIMB)
    assert torch.isfinite(r).all()
    assert _p_blind_only(CLIMB).item() == pytest.approx(0.0, abs=1e-9)


def test_penalty_scales_linearly_with_horizontal_speed():
    """The risk gradient: fast blind flight must cost more than slow blind flight."""
    slow = _p_blind_only(torch.tensor([[0.0, 1.0, 0.0]])).item()
    fast = _p_blind_only(torch.tensor([[0.0, 4.0, 0.0]])).item()
    assert fast == pytest.approx(4.0 * slow, abs=1e-5)


def test_soft_deadzone_is_quartic_near_boresight():
    """10 deg off-nose must cost ~0.02% of the 90 deg penalty, not the 1.5% a bare
    (1 - cos) would charge. This is the deadzone, and it is why the term is squared."""
    speed = 3.0
    v10 = torch.tensor([[speed * math.cos(math.radians(10)),
                         speed * math.sin(math.radians(10)), 0.0]])
    ratio = _p_blind_only(v10).item() / _p_blind_only(SIDEWAYS).item()
    assert ratio == pytest.approx(0.00023, abs=5e-5)


def test_penalty_is_monotonic_in_misalignment():
    """No saturation: the gradient must keep growing, unlike a saturating exponential
    whose slope dies near 90 deg -- where an untrained policy spends most of its time."""
    speed = 3.0
    vals = []
    for deg in [0, 15, 30, 45, 60, 90, 135, 180]:
        r = math.radians(deg)
        v = torch.tensor([[speed * math.cos(r), speed * math.sin(r), 0.0]])
        vals.append(_p_blind_only(v).item())
    assert all(b < a for a, b in zip(vals, vals[1:])), vals

    # And strictly convex in the angle: each step down is larger than the last.
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    assert diffs[-1] < diffs[0], diffs


def test_reward_is_yaw_sensitive():
    """The whole point. Same speed, same distance-to-target, different heading.

    Before p_blind no term could separate these two, so the policy had no reason to
    yaw -- yaw appeared only in p_jerk and p_action_mag, as cost.
    """
    facing = _reward(_stub(), FORWARD).item()
    yawed = _reward(_stub(), SIDEWAYS).item()
    assert facing > yawed

    # ...and with the term disabled they are indistinguishable again.
    off = _stub(lambda_blind=0.0)
    assert _reward(off, FORWARD).item() == pytest.approx(
        _reward(_stub(lambda_blind=0.0), SIDEWAYS).item(), abs=1e-6
    )


def test_r_bearing_is_gone():
    """lambda_b was 52% of the episode return and is deleted, not zeroed: a zero-weight
    term is a live footgun. Guard against it being reintroduced by a merge."""
    assert "lambda_b" not in task_config.reward_parameters
    assert "r_heading" not in EMA_KEYS


def test_faster_beats_hovering_at_the_expected_untrained_misalignment():
    """Sizing invariant: lambda_blind * E[misalignment^2] < lambda_p * dt, so a policy
    with yaw uncorrelated from velocity still gains by moving rather than slowing down.

    E[misalignment^2] = 3/8 for a uniformly distributed heading error; dt is sim dt
    (0.01) x num_physics_steps_per_env_step_mean (3).
    """
    dt = 0.01 * 3
    lambda_p = task_config.reward_parameters["lambda_p"]
    assert LAMBDA_BLIND * (3.0 / 8.0) < lambda_p * dt
