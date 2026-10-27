"""Spike-rate and SynOps accounting for a PopulationSpikingActorNetwork.

A SynOp is one spike crossing one synapse: an accumulate, not a multiply-accumulate,
because the spike is binary. Per control step

    SynOps = sum_t [ enc_spikes[t] * fc1.out + lif1_spikes[t] * fc2.out
                     + lif2_spikes[t] * fc3.out ]

lif3's spikes are never propagated through a synapse: they are averaged over the window
and decoded ONCE by the grouped Conv1d, which is dense arithmetic and counted as MACs
alongside the encoder's Gaussian receptive fields (also once per step, since the encoder
current is constant across the window).

Isaac Gym free on purpose, so the accounting can be unit-tested without a simulator.
"""
from dataclasses import dataclass, field
from typing import Dict

import torch

# 45 nm, 32-bit float (Horowitz, ISSCC 2014) -- the convention in the SNN-energy literature.
E_MAC_PJ = 4.6
E_AC_PJ = 0.9

LAYERS = ("encoder", "lif1", "lif2", "lif3")


@dataclass
class _Tally:
    n_neurons: int
    spikes: float = 0.0
    per_neuron: torch.Tensor = None  # spike count per neuron, summed over samples and steps


@dataclass
class SpikeCounter:
    """Forward hooks on the encoder and the three LIF layers.

    Hooks fire once per spiking timestep; `samples` counts control steps x envs, which is
    what every rate below is normalised by.
    """
    net: torch.nn.Module
    tallies: Dict[str, _Tally] = field(default_factory=dict)
    samples: int = 0

    def __post_init__(self):
        self._handles = [
            self.net.pop_encoder.register_forward_hook(self._encoder_hook),
            self.net.actor_lif1.register_forward_hook(self._lif_hook("lif1")),
            self.net.actor_lif2.register_forward_hook(self._lif_hook("lif2")),
            self.net.actor_lif3.register_forward_hook(self._lif_hook("lif3")),
        ]

    def _add(self, name, spk):
        flat = spk.detach().reshape(spk.shape[0], -1).float().sum(0)
        t = self.tallies.get(name)
        if t is None:
            t = self.tallies[name] = _Tally(n_neurons=flat.numel(),
                                            per_neuron=torch.zeros_like(flat))
        t.per_neuron += flat
        t.spikes += float(flat.sum())

    def _encoder_hook(self, module, inputs, out):
        # [B, obs_dim * pop_dim, T]: the whole window in one call.
        self.samples += out.shape[0]
        self._add("encoder", out.sum(-1))

    def _lif_hook(self, name):
        def hook(module, inputs, out):
            self._add(name, out[0])  # snntorch Synaptic -> (spk, syn, mem)
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()

    def report(self, ann_macs: int = None) -> dict:
        net, T = self.net, self.net.num_steps
        fanout = {"encoder": net.actor_fc1.out_features,
                  "lif1": net.actor_fc2.out_features,
                  "lif2": net.actor_fc3.out_features,
                  "lif3": 0}
        layers, synops = {}, 0.0
        for name in LAYERS:
            t = self.tallies[name]
            per_step = t.spikes / self.samples
            rate = per_step / (t.n_neurons * T)
            ops = per_step * fanout[name]
            synops += ops
            layers[name] = {
                "neurons": t.n_neurons,
                "spikes_per_step": per_step,
                "rate": rate,
                "silent_frac": float((t.per_neuron == 0).float().mean()),
                "synops_per_step": ops,
            }

        synapses = sum(fanout[n] * self.tallies[n].n_neurons for n in LAYERS)
        dense_macs = (net.pop_encoder.obs_dim * net.pop_dim       # Gaussian receptive fields
                      + net.action_dim * net.pop_dim)             # population decoder
        neuron_updates = T * sum(self.tallies[n].n_neurons for n in LAYERS)
        snn_pj = synops * E_AC_PJ + dense_macs * E_MAC_PJ

        out = {
            "num_steps": T,
            "samples": self.samples,
            "layers": layers,
            "overall_rate": (sum(self.tallies[n].spikes for n in LAYERS)
                             / (self.samples * T
                                * sum(self.tallies[n].n_neurons for n in LAYERS))),
            "synops_per_step": synops,
            "synops_if_all_fired": synapses * T,
            "dense_macs_per_step": dense_macs,
            "neuron_updates_per_step": neuron_updates,
            "snn_energy_pj": snn_pj,
        }
        if ann_macs:
            ann_pj = ann_macs * E_MAC_PJ
            out.update({"ann_macs_per_step": ann_macs, "ann_energy_pj": ann_pj,
                        "energy_ratio_ann_over_snn": ann_pj / snn_pj,
                        "breakeven_synops": (ann_pj - dense_macs * E_MAC_PJ) / E_AC_PJ})
        return out


def linear_macs(module: torch.nn.Module) -> int:
    """MACs of one forward pass through every nn.Linear in a dense actor."""
    return sum(m.in_features * m.out_features
               for m in module.modules() if isinstance(m, torch.nn.Linear))


def format_report(r: dict) -> str:
    lines = [f"{r['samples']} control steps, T = {r['num_steps']} spiking steps each", "",
             f"{'layer':8s} {'neurons':>7s} {'rate':>7s} {'silent':>7s} "
             f"{'spikes/step':>12s} {'SynOps/step':>12s}"]
    for name, l in r["layers"].items():
        lines.append(f"{name:8s} {l['neurons']:7d} {100 * l['rate']:6.2f}% "
                     f"{100 * l['silent_frac']:6.1f}% {l['spikes_per_step']:12.1f} "
                     f"{l['synops_per_step']:12.0f}")
    lines += ["",
              f"overall firing rate      {100 * r['overall_rate']:.2f}%",
              f"SynOps / step            {r['synops_per_step']:.0f}  "
              f"({100 * r['synops_per_step'] / r['synops_if_all_fired']:.1f}% of "
              f"{r['synops_if_all_fired']:.0f} if every neuron fired every step)",
              f"dense MACs / step        {r['dense_macs_per_step']}  (encoder + decoder)",
              f"neuron updates / step    {r['neuron_updates_per_step']}  (not in energy)",
              f"SNN energy / step        {r['snn_energy_pj'] / 1e3:.1f} nJ  "
              f"(AC {E_AC_PJ} pJ, MAC {E_MAC_PJ} pJ)"]
    if "ann_macs_per_step" in r:
        lines += [f"ANN MACs / step          {r['ann_macs_per_step']}",
                  f"ANN energy / step        {r['ann_energy_pj'] / 1e3:.1f} nJ",
                  f"ANN / SNN energy         {r['energy_ratio_ann_over_snn']:.2f}x",
                  f"break-even SynOps        {r['breakeven_synops']:.0f}"]
    return "\n".join(lines)
