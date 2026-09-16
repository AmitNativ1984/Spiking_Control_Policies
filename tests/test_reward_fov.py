"""Unit tests for p_fov: velocity pointed outside the cone the camera can actually see.

Same approach as test_reward_blind_motion.py -- drives the real _reward_progress against
a stub `self`, so the arithmetic under test is the shipped arithmetic. Distances are held
equal across steps so r_progress contributes exactly zero.

The property that motivates the term is test_leaving_the_vertical_cone_costs_more: the
frustum is 87 x 56 deg, so the vertical half-angle (28.1) is tighter than the horizontal
(43.5), and crash-cause eval found ~55% of struck obstacles beyond the vertical one. Each
axis is normalized by its own half-angle, so that asymmetry falls out of the shape rather
than being a tuned per-axis weight.
"""
import math
import types

import pytest
import torch

from task.attitude_navigation_task import NavigationWithObstaclesTask
from config.task_config.f450_attitude_navigation_task_config import task_config
from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig as CAM

EMA_KEYS = ["r_progress", "p_speed", "p_jerk", "p_action_mag", "p_blind", "p_fov"]

HALF_H = math.radians(CAM.horizontal_fov_deg) / 2.0
HALF_V = math.atan(math.tan(HALF_H) * (CAM.height / CAM.width))
LAMBDA_FOV = 0.05  # exercised value; the shipped default is 0.0 (inert)
POWER = task_config.reward_parameters["fov_power"]
V_REF = task_config.reward_parameters["fov_v_ref"]


def _stub(**overrides):
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
    stub._half_h_fov = HALF_H
    stub._half_v_fov = HALF_V
    return stub


def _reward(stub, v, dist=5.0):
    num = v.shape[0]
    d = torch.full((num,), dist)
    zeros = torch.zeros(num, 4)

    stub.obs_dict = {"robot_vehicle_linvel": v}
    stub.prev_dist = d
    stub.prev_action = zeros
    stub._get_dist_to_target = lambda: d

    mask = torch.ones(num, dtype=torch.bool)
    return NavigationWithObstaclesTask._reward_progress(stub, mask, zeros)


def _p_fov_only(v, lam=LAMBDA_FOV):
    """p_fov isolated: reward with lambda_fov set, minus reward with it zeroed."""
    return (
        _reward(_stub(lambda_fov=lam), v) - _reward(_stub(lambda_fov=0.0), v)
    ).item()


def _vel(speed, azimuth_deg=0.0, elevation_deg=0.0):
    """Vehicle frame is yaw-aligned and FLU: x forward (nose), y left, z up."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return torch.tensor([[
        speed * math.cos(el) * math.cos(az),
        speed * math.cos(el) * math.sin(az),
        speed * math.sin(el),
    ]])


def test_inert_by_default():
    """The shipped default must not perturb any existing run."""
    assert task_config.reward_parameters["lambda_fov"] == 0.0
    v = _vel(3.0, azimuth_deg=120.0)
    assert _p_fov_only(v, lam=task_config.reward_parameters["lambda_fov"]) == 0.0


def test_flying_along_the_nose_is_free():
    assert _p_fov_only(_vel(3.0)) == pytest.approx(0.0, abs=1e-9)


def test_hover_is_free():
    """atan2(0, 0) = 0, so a hover sits at the centre of the cone and pays nothing."""
    assert _p_fov_only(torch.zeros(1, 3)) == pytest.approx(0.0, abs=1e-9)


def test_growth_is_superlinear_from_boresight():
    """Doubling the angular distance from boresight must cost MORE than double."""
    near = abs(_p_fov_only(_vel(3.0, azimuth_deg=20.0)))
    far = abs(_p_fov_only(_vel(3.0, azimuth_deg=40.0)))
    assert far > 2.0 * near
    assert far == pytest.approx(near * 2.0 ** POWER, rel=1e-6)


def test_outside_the_cone_costs():
    assert _p_fov_only(_vel(3.0, azimuth_deg=90.0)) < 0.0


def test_cost_grows_with_distance_outside():
    near = _p_fov_only(_vel(3.0, azimuth_deg=50.0))
    far = _p_fov_only(_vel(3.0, azimuth_deg=80.0))
    assert far < near < 0.0


def test_never_saturates():
    """No clamp anywhere: a from-scratch policy sits near 90 deg misaligned, and a
    shape that flattens there teaches it nothing (see the p_blind note this mirrors)."""
    prev = 0.0
    for deg in (30.0, 60.0, 90.0, 120.0, 150.0, 179.0):
        cur = abs(_p_fov_only(_vel(3.0, azimuth_deg=deg)))
        assert cur > prev, f"gradient died by {deg} deg"
        prev = cur


def test_leaving_the_vertical_cone_costs_more():
    """The whole point: 43.5 deg horizontal vs 28.1 deg vertical, so the same absolute
    excursion strands you further outside the tighter cone and must cost more."""
    horizontal = _p_fov_only(_vel(3.0, azimuth_deg=45.0))
    vertical = _p_fov_only(_vel(3.0, elevation_deg=45.0))
    assert vertical < horizontal < 0.0


def test_speed_saturates():
    """Above fov_v_ref the only remaining lever is aim, not slowing down."""
    at_ref = _p_fov_only(_vel(V_REF, azimuth_deg=90.0))
    way_over = _p_fov_only(_vel(4.0 * V_REF, azimuth_deg=90.0))
    assert at_ref == pytest.approx(way_over, rel=1e-6)


def test_slowing_below_v_ref_still_helps():
    """Below the saturation point the risk gradient is intact, so hover stays free."""
    slow = _p_fov_only(_vel(0.5 * V_REF, azimuth_deg=90.0))
    fast = _p_fov_only(_vel(V_REF, azimuth_deg=90.0))
    assert fast < slow < 0.0


def test_lambda_is_the_cost_at_the_cone_edge():
    """The whole point of dropping the normalizer: lambda_fov reads directly as the
    per-step cost of flying AT the edge of what the camera can see -- the operating
    point the crashes sit at (median azimuth 43.9 deg vs a 43.5 deg half-angle)."""
    at_h_edge = _p_fov_only(_vel(V_REF, azimuth_deg=math.degrees(HALF_H)))
    at_v_edge = _p_fov_only(_vel(V_REF, elevation_deg=math.degrees(HALF_V)))
    assert at_h_edge == pytest.approx(-LAMBDA_FOV * V_REF, rel=1e-5)
    assert at_v_edge == pytest.approx(-LAMBDA_FOV * V_REF, rel=1e-5)


def test_inside_the_cone_is_a_fraction_of_the_edge():
    """Half a cone width costs a quarter of the edge at quadratic, so ordinary in-cone
    manoeuvring stays cheap relative to skirting the boundary."""
    half = abs(_p_fov_only(_vel(V_REF, azimuth_deg=0.5 * math.degrees(HALF_H))))
    edge = abs(_p_fov_only(_vel(V_REF, azimuth_deg=math.degrees(HALF_H))))
    assert half == pytest.approx(0.25 * edge, rel=1e-5)


def test_bounded_by_geometry():
    """No clamp, but psi and theta are bounded, so the term cannot run away."""
    r_max = math.sqrt((math.pi / HALF_H) ** 2 + ((math.pi / 2) / HALF_V) ** 2)
    worst = min(
        _p_fov_only(_vel(9.0, azimuth_deg=az, elevation_deg=el))
        for az in (0.0, 90.0, 180.0)
        for el in (-90.0, -45.0, 0.0, 45.0, 90.0)
    )
    assert worst >= -LAMBDA_FOV * V_REF * r_max ** POWER * (1 + 1e-5)
