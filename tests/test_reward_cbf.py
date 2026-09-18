"""Unit tests for p_cbf: a discrete-time control barrier function on obstacle clearance.

Same approach as test_reward_fov.py -- drives the real _reward_progress against a stub
`self`, so the arithmetic under test is the shipped arithmetic. Distances are held equal
across steps so r_progress contributes exactly zero, and every quantity that is not p_cbf
is identical between the two stubs that get differenced, so it cancels.

The probe itself is stubbed out: _clearance() is the seam. These tests are about the
barrier arithmetic and can run anywhere; whether the probe returns the true distance to
the nearest surface is a different question, answered by
analysis/validate_proximity_probe.py against the depth image and the altitude.

The property that motivates the term is test_the_allowance_scales_with_clearance: the
same closing speed is free with room to spare and expensive without it. No other term in
the reward knows how much room is left.
"""
import math
import types

import pytest
import torch

from task.attitude_navigation_task import NavigationWithObstaclesTask
from config.task_config.f450_attitude_navigation_task_config import task_config
from config.sensor_config.realsense_d435_cam_config import RealSenseD435CamConfig as CAM

EMA_KEYS = ["r_progress", "p_speed", "p_jerk", "p_action_mag", "p_blind", "p_fov", "p_cbf"]

HALF_H = math.radians(CAM.horizontal_fov_deg) / 2.0
HALF_V = math.atan(math.tan(HALF_H) * (CAM.height / CAM.width))

DT = 0.03          # sim dt 0.01 x 3 substeps, the shipped env step
LAMBDA_CBF = 0.12  # exercised value; the shipped default is 0.0 (inert)
ALPHA = 0.5        # 1/s, the shipped default
D_SAFE = 1.0       # m, the shipped default
MAX_RANGE = 6.0    # m, the probe's range limit
SPEED = 2.0        # reference speed; below v_max, so p_speed stays out of the way


def _stub(h_prev, h_next, lam=LAMBDA_CBF, alpha=ALPHA, dt=DT, max_range=MAX_RANGE):
    params = dict(task_config.reward_parameters)
    params.update(
        lambda_cbf=lam, alpha_cbf=alpha, d_safe=D_SAFE, cbf_max_range=max_range
    )

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

    # p_cbf state. _clearance() is stubbed, so no warp mesh and no GPU are needed.
    stub._cbf_lambda = lam
    stub._cbf_active = lam != 0.0
    stub._cbf_h_max = max_range - D_SAFE
    stub._env_step_dt = dt
    stub._cbf_d_ema = 0.0
    stub._cbf_viol_ema = 0.0
    stub.prev_h = torch.as_tensor(h_prev, dtype=torch.float32).reshape(-1)
    h_n = torch.as_tensor(h_next, dtype=torch.float32).reshape(-1)
    stub._clearance = lambda: h_n
    return stub


def _reward(stub, v, dist=5.0):
    num = v.shape[0]
    d = torch.full((num,), dist)
    zeros = torch.zeros(num, 4)

    stub.obs_dict = {
        "robot_vehicle_linvel": v,
        "robot_orientation": torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(num, 4),
        "robot_linvel": v,
    }
    stub.prev_dist = d
    stub.prev_action = zeros
    stub._get_dist_to_target = lambda: d

    mask = torch.ones(num, dtype=torch.bool)
    return NavigationWithObstaclesTask._reward_progress(stub, mask, zeros)


def _p_cbf_only(h_prev, h_next, lam=LAMBDA_CBF, **kw):
    """p_cbf isolated: reward with lambda_cbf set, minus reward with it zeroed. Every
    other term sees identical inputs in the two calls, so it cancels exactly."""
    v = _vel(SPEED)
    live = _reward(_stub(h_prev, h_next, lam=lam, **kw), v)
    inert = _reward(_stub(h_prev, h_next, lam=0.0, **kw), v)
    return (live - inert).squeeze()


def _vel(speed):
    """FLU: x forward. Held along the nose so p_fov and p_blind are zero anyway."""
    return torch.tensor([[speed, 0.0, 0.0]])


def _closing(h_prev, v_close, dt=DT):
    """The h_next that corresponds to closing on the nearest surface at v_close m/s."""
    return h_prev - v_close * dt


# --- the shipped defaults must not perturb anything ----------------------------------


def test_inert_by_default():
    """The shipped default must not perturb any existing run."""
    assert task_config.reward_parameters["lambda_cbf"] == 0.0
    lam = task_config.reward_parameters["lambda_cbf"]
    assert _p_cbf_only(1.0, _closing(1.0, 5.0), lam=lam).item() == 0.0


def test_the_discrete_barrier_is_well_posed_at_the_shipped_alpha():
    """alpha is a RATE; the [0, 1] bound belongs to the decay factor alpha*dt. Above 1
    the factor (1 - alpha*dt) goes negative and the barrier inverts, demanding clearance
    GROW at positive h. The task asserts this at init; this pins the shipped value."""
    assert 0.0 <= task_config.reward_parameters["alpha_cbf"] * DT <= 1.0


def test_d_safe_fits_inside_the_measured_clearance_distribution():
    """d_safe above the clearance the env affords makes h < 0 the normal case, where the
    barrier demands retreat rather than a slower approach. Measured on b4 at level 30
    (analysis/data/freespace_b4_l30.json): p25 of nearest-in-view is 1.09 m, and the full
    sphere can only read lower. 1.0 m keeps the safe set reachable; 1.5 m would not."""
    assert task_config.reward_parameters["d_safe"] <= 1.0
    # Also below the 0.8 m wall inset + 0.4 m arrival ball, so arriving is not penalised
    # outright whenever the target's wall happens to be meshed.
    assert task_config.reward_parameters["d_safe"] <= 1.2


# --- the barrier condition ------------------------------------------------------------


def test_closing_within_the_allowance_is_free():
    """h = 2.0 with alpha = 0.5 permits 1.0 m/s. Closing at 0.8 must cost nothing."""
    assert _p_cbf_only(2.0, _closing(2.0, 0.8)).item() == pytest.approx(0.0, abs=1e-9)


def test_closing_at_exactly_the_allowance_is_free():
    """The boundary belongs to the safe side: max(0, .) is zero at equality."""
    assert _p_cbf_only(2.0, _closing(2.0, ALPHA * 2.0)).item() == pytest.approx(0.0, abs=1e-6)


def test_excess_closing_costs_lambda_times_the_excess():
    """The whole term, checked against the closed form in m/s."""
    h, v_close = 2.0, 3.0
    excess = v_close - ALPHA * h          # 3.0 - 1.0 = 2.0 m/s
    got = _p_cbf_only(h, _closing(h, v_close)).item()
    assert got == pytest.approx(-LAMBDA_CBF * excess, rel=1e-5)


def test_retreating_is_free():
    """Clearance increasing can never violate a barrier that caps its decrease."""
    assert _p_cbf_only(1.0, 1.5).item() == pytest.approx(0.0, abs=1e-9)


def test_holding_station_outside_the_bubble_is_free():
    assert _p_cbf_only(1.5, 1.5).item() == pytest.approx(0.0, abs=1e-9)


def test_the_allowance_scales_with_clearance():
    """THE point of the term. The same closing speed is free with room and expensive
    without it -- which is exactly what p_speed (fixed v_max) and p_fov (direction only)
    cannot express."""
    v_close = 1.5
    roomy = _p_cbf_only(4.0, _closing(4.0, v_close)).item()   # allowance 2.0 m/s
    tight = _p_cbf_only(0.4, _closing(0.4, v_close)).item()   # allowance 0.2 m/s
    assert roomy == pytest.approx(0.0, abs=1e-9)
    assert tight < 0.0
    assert tight == pytest.approx(-LAMBDA_CBF * (v_close - ALPHA * 0.4), rel=1e-5)


def test_the_penalty_grows_as_clearance_shrinks_at_fixed_speed():
    """Monotone in the direction that matters: same speed, less room, more cost."""
    v_close = 2.0
    costs = [_p_cbf_only(h, _closing(h, v_close)).item() for h in (2.0, 1.0, 0.5, 0.0)]
    assert all(b < a for a, b in zip(costs, costs[1:]))


# --- inside the safe set's boundary ---------------------------------------------------


def test_holding_station_inside_the_bubble_costs_the_recovery_rate():
    """h < 0 is not special-cased: the allowance goes NEGATIVE, so the barrier demands
    clearance be regained at alpha*|h| and standing still is already a violation. This
    documents the behaviour that makes d_safe's sizing load-bearing."""
    h = -0.5
    got = _p_cbf_only(h, h).item()
    assert got == pytest.approx(-LAMBDA_CBF * ALPHA * 0.5, rel=1e-5)


def test_recovering_fast_enough_inside_the_bubble_is_free():
    """Regaining clearance at exactly alpha*|h| satisfies the barrier."""
    h = -0.5
    h_next = h + ALPHA * abs(h) * DT
    assert _p_cbf_only(h, h_next).item() == pytest.approx(0.0, abs=1e-6)


# --- the rate form ---------------------------------------------------------------------


def test_the_rate_form_is_dt_independent():
    """The reason lambda_cbf is divided by dt rather than folded into it: at a FIXED
    physical closing speed, halving the env step must not halve the weight of the term.
    With dt folded in, changing the substep count would silently rescale p_cbf."""
    h, v_close = 1.0, 2.0
    slow = _p_cbf_only(h, _closing(h, v_close, dt=DT), dt=DT).item()
    fast = _p_cbf_only(h, _closing(h, v_close, dt=DT / 2), dt=DT / 2).item()
    assert slow == pytest.approx(fast, rel=1e-5)
    assert slow == pytest.approx(-LAMBDA_CBF * (v_close - ALPHA * h), rel=1e-5)


def test_matches_the_raw_discrete_barrier_form():
    """Equivalence to -lambda * max(0, (1 - alpha*dt)*h_t - h_{t+1}), the form the term
    is specified in, with lambda_raw = lambda_cbf / dt."""
    h, h_next = 1.2, 1.1
    raw = -(LAMBDA_CBF / DT) * max(0.0, (1.0 - ALPHA * DT) * h - h_next)
    assert _p_cbf_only(h, h_next).item() == pytest.approx(raw, rel=1e-5)


# --- the probe's range limit ------------------------------------------------------------


def test_envs_at_the_probe_range_limit_are_exempt():
    """The probe reports max_range when it found nothing, which UNDERSTATES h and so
    understates the allowance. Charging on that would charge against a distance that was
    never measured."""
    h_clipped = MAX_RANGE - D_SAFE
    assert _p_cbf_only(h_clipped, h_clipped - 1.0).item() == pytest.approx(0.0, abs=1e-9)


def test_just_inside_the_range_limit_is_not_exempt():
    """The exemption must be a knife edge at the clip, not a dead band below it."""
    h = MAX_RANGE - D_SAFE - 0.05
    assert _p_cbf_only(h, h - 1.0).item() < 0.0


# --- plumbing ---------------------------------------------------------------------------


def test_the_term_reaches_the_total_and_the_ema():
    """p_cbf must be summed into the returned reward and logged, not computed and
    dropped -- the failure mode a pure closed-form test would miss."""
    stub = _stub(0.5, _closing(0.5, 3.0))
    total = _reward(stub, _vel(SPEED))
    assert stub._reward_comp_ema["p_cbf"] < 0.0
    assert total.item() < _reward(_stub(0.5, _closing(0.5, 3.0), lam=0.0), _vel(SPEED)).item()


def test_clearance_diagnostics_are_populated_when_active():
    """metrics/d_obstacle and metrics/cbf_violation_rate are how a run says whether the
    term is firing at all, so they must move."""
    stub = _stub(0.5, _closing(0.5, 3.0))
    _reward(stub, _vel(SPEED))
    assert stub._cbf_d_ema > 0.0            # h_next + d_safe, a real distance
    assert stub._cbf_viol_ema > 0.0         # this step violates


def test_per_env_independence():
    """Batched: each env's penalty depends only on its own clearance."""
    h_prev = torch.tensor([4.0, 0.4, -0.5])
    h_next = h_prev - torch.tensor([1.5, 1.5, 0.0]) * DT
    v = _vel(SPEED).expand(3, 3).contiguous()
    live = _reward(_stub(h_prev, h_next), v)
    inert = _reward(_stub(h_prev, h_next, lam=0.0), v)
    got = (live - inert)
    assert got[0].item() == pytest.approx(0.0, abs=1e-9)
    assert got[1].item() == pytest.approx(-LAMBDA_CBF * (1.5 - ALPHA * 0.4), rel=1e-5)
    assert got[2].item() == pytest.approx(-LAMBDA_CBF * ALPHA * 0.5, rel=1e-5)
