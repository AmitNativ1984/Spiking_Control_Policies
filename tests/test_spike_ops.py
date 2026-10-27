"""SpikeCounter's SynOps accounting, checked against an independent re-run of the network."""
import pytest
import torch

from analysis.spike_ops import SpikeCounter, linear_macs
from rl_training.rl_games.networks.snn.pop_spiking_actor import PopulationSpikingActorNetwork

OBS_DIM, ACTION_DIM, POP, T = 6, 4, 4, 5
ACTOR = {
    "hidden_dims": [64, 32], "num_steps": T, "spike_grad": "fast_sigmoid",
    "alpha": 0.5, "beta": 0.75, "threshold": 0.5,
    "reset_mechanism": "subtract", "reset_delay": False, "pop_dim": POP,
    "encoder": {"pop_dim": POP, "threshold": 0.95},
    "observation_bounds": [(-3.0, 3.0)] * OBS_DIM,
}


@pytest.fixture
def net():
    torch.manual_seed(0)
    return PopulationSpikingActorNetwork(OBS_DIM, ACTION_DIM, **ACTOR).eval()


def _manual_spikes(net, obs):
    """Spike totals per layer from a hand-unrolled forward, sharing no code with the hooks."""
    syn1, mem1 = net.actor_lif1.reset_mem()
    syn2, mem2 = net.actor_lif2.reset_mem()
    syn3, mem3 = net.actor_lif3.reset_mem()
    enc = net.pop_encoder(obs)
    tot = {"encoder": float(enc.sum()), "lif1": 0.0, "lif2": 0.0, "lif3": 0.0}
    for t in range(T):
        s1, syn1, mem1 = net.actor_lif1(net.actor_fc1(enc[:, :, t]), syn1, mem1)
        s2, syn2, mem2 = net.actor_lif2(net.actor_fc2(s1), syn2, mem2)
        s3, syn3, mem3 = net.actor_lif3(net.actor_fc3(s2), syn3, mem3)
        tot["lif1"] += float(s1.sum()); tot["lif2"] += float(s2.sum())
        tot["lif3"] += float(s3.sum())
    return tot


def test_synops_match_a_hand_count(net):
    obs = torch.randn(32, OBS_DIM)
    with torch.no_grad():
        expected = _manual_spikes(net, obs)
        counter = SpikeCounter(net)
        net({"obs": obs})
    r = counter.report()

    for name, n in expected.items():
        assert r["layers"][name]["spikes_per_step"] == pytest.approx(n / 32)
    want = (expected["encoder"] * 64 + expected["lif1"] * 32
            + expected["lif2"] * ACTION_DIM * POP) / 32
    assert r["synops_per_step"] == pytest.approx(want)
    assert 0.0 <= r["overall_rate"] <= 1.0


def test_counts_accumulate_over_calls_and_hooks_detach(net):
    obs = torch.randn(8, OBS_DIM)
    with torch.no_grad():
        counter = SpikeCounter(net)
        net({"obs": obs}); net({"obs": obs})
        assert counter.samples == 16
        counter.remove()
        net({"obs": obs})
    assert counter.samples == 16


def test_ann_comparison(net):
    with torch.no_grad():
        counter = SpikeCounter(net)
        net({"obs": torch.randn(4, OBS_DIM)})
    ann = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.ELU(), torch.nn.Linear(64, 4))
    macs = linear_macs(ann)
    assert macs == 6 * 64 + 64 * 4
    r = counter.report(ann_macs=macs)
    assert r["energy_ratio_ann_over_snn"] == pytest.approx(r["ann_energy_pj"] / r["snn_energy_pj"])
