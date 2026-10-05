"""ControlLatency: the action timeline that models capture->output dead time.

CPU-only. Drives the class the way the task does -- push() once per env step, then
on_substep() once per physics substep -- and checks what the controller would fly.
"""
import random

import pytest
import torch

from task.control_latency import ControlLatency

DT = 0.01


def make(n=4, mean_s=0.0, std_s=0.0, min_s=0.0, max_s=0.1, buffer_len=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return ControlLatency(
        num_envs=n, action_dim=4, sim_dt=DT, mean_s=mean_s, std_s=std_s,
        min_s=min_s, max_s=max_s, buffer_len=buffer_len, device="cpu", generator=gen,
    )


def action(k, n=4):
    """A distinguishable action for env step k: every entry equals k + 1."""
    return torch.full((n, 4), float(k + 1))


def run(lat, substeps):
    """Push action k before env step k, then run substeps[k] substeps.

    Returns, per substep, the value of the action being flown by env 0 (0 = neutral).
    """
    flown = []
    for k, m in enumerate(substeps):
        lat.push(action(k, lat.num_envs))
        for _ in range(m):
            flown.append(lat.on_substep()[0, 0].item())
    return flown


def test_zero_latency_is_the_old_behaviour():
    lat = make()
    assert run(lat, [3, 2, 4]) == [1, 1, 1, 2, 2, 3, 3, 3, 3]


def test_constant_latency_shifts_by_whole_substeps():
    # 50 ms = 5 substeps; steps of 3 substeps are captured at ticks 0, 3, 6, ...
    lat = make(mean_s=0.05)
    flown = run(lat, [3, 3, 3, 3])
    # Neutral for ticks 0-4; a_0 arrives at tick 5, a_1 at 8, a_2 at 11.
    assert flown == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]


def test_latency_longer_than_a_period_keeps_an_action_in_flight():
    # 40 ms = 4 substeps against a 3-substep period -> a_k flies [3k+4, 3k+7).
    lat = make(mean_s=0.04)
    flown = run(lat, [3] * 5)
    assert flown[:4] == [0, 0, 0, 0]
    assert flown[4:7] == [1, 1, 1]
    assert flown[7:10] == [2, 2, 2]


def test_jitter_never_reorders_actions():
    lat = make(n=256, mean_s=0.045, std_s=0.02, seed=1)
    rng = random.Random(0)
    last = torch.zeros(256)
    for k in range(400):
        lat.push(action(k, 256))
        for _ in range(max(int(rng.gauss(3.5, 1)), 0)):
            flown = lat.on_substep()[:, 0]
            assert torch.all(flown >= last), "an older action replaced a newer one"
            last = flown


def test_realised_latency_matches_the_measurement():
    lat = make(n=1024, mean_s=0.045, std_s=0.010, seed=2)
    realised = []
    rng = random.Random(1)
    for k in range(200):
        lat.push(action(k, 1024))
        realised.append(lat.last_latency_s.clone())
        for _ in range(max(int(rng.gauss(3.5, 1)), 0)):
            lat.on_substep()
    realised = torch.stack(realised[10:])
    # Round-to-nearest keeps the mean unbiased; the no-overtaking rule can only add,
    # and at a 30 ms period with 10 ms jitter it adds little.
    assert abs(realised.mean().item() - 0.045) < 0.003
    assert 0.008 < realised.std().item() < 0.014
    assert realised.min().item() >= 0.0 and realised.max().item() <= 0.1 + 1e-9


def test_zero_substep_steps_supersede_without_dropping_order():
    # Upstream can draw 0 substeps: two pushes land on the same capture tick.
    lat = make(mean_s=0.02)
    flown = run(lat, [0, 3, 3])
    # a_0 and a_1 both arrive at tick 2; the newer (a_1) must win. a_2 lands at 5.
    assert flown == [0, 0, 2, 2, 2, 3]


def test_reset_drops_in_flight_actions_for_those_envs_only():
    lat = make(n=2, mean_s=0.05)
    lat.push(action(0, 2))
    for _ in range(6):
        lat.on_substep()
    lat.push(action(1, 2))
    lat.reset(torch.tensor([1]))
    for _ in range(6):
        flown = lat.on_substep()
    # env 0 gets a_1 at tick 11; env 1 was reset, so it holds neutral.
    assert flown[0, 0].item() == 2
    assert flown[1, 0].item() == 0
    # env 1 picks the timeline back up from its next push.
    lat.push(action(2, 2))
    for _ in range(6):
        flown = lat.on_substep()
    assert flown[1, 0].item() == 3


def test_buffer_wraps_without_losing_pending_actions():
    # Five actions are in flight at once here (ticks k+1..k+5), so 6 slots wrap three
    # times over 20 pushes without ever overwriting one that has not arrived.
    lat = make(mean_s=0.05, buffer_len=6)
    flown = run(lat, [1] * 20)
    # a_k is pushed at tick k and arrives at k + 5: always one fresh action per tick.
    assert flown[5:] == [float(k + 1) for k in range(15)]


def test_rejects_inverted_clamp():
    with pytest.raises(ValueError):
        make(min_s=0.2, max_s=0.1)
