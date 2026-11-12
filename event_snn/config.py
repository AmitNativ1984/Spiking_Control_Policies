"""YAML config with variant overlays and dotted command-line overrides."""
import copy
import os
from types import SimpleNamespace

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "cfg", "default.yaml")

_A_REC = {"recurrent": True, "stateful": True, "state_inputs": ["acc", "gyro", "grav"]}

VARIANTS = {
    "a_rec": {"model": {"recurrent": True, "stateful": True, "state_inputs": ["acc", "gyro", "grav"]}},
    "a_ff": {"model": {"recurrent": False, "stateful": True, "state_inputs": ["acc", "gyro", "grav"]}},
    "b": {"model": {"recurrent": False, "stateful": False, "state_inputs": ["acc", "gyro", "grav"]},
          "loss": {"burn_in_s": 0.0}},
    "a_rec_noimu": {"model": {"recurrent": True, "stateful": True, "state_inputs": ["grav"]}},
    "a_rec_rawimu": {"model": {"recurrent": True, "stateful": True, "state_inputs": ["acc", "gyro"]}},
    # Representation study: a_rec with a 128-neuron spiking bottleneck as the policy interface;
    # the three differ only in what the training-only heads must decode from those spikes.
    "bn_depth": {"model": {**_A_REC, "bottleneck": 128}},
    "bn_ttc": {"model": {**_A_REC, "bottleneck": 128, "ttc_head": True},
               "loss": {"lambda_d": 0.0, "lambda_ttc": 1.0}},
    "bn_both": {"model": {**_A_REC, "bottleneck": 128, "ttc_head": True},
                "loss": {"lambda_d": 1.0, "lambda_ttc": 1.0}},
    # Scale-free study: conv3 -> flatten -> concat inertial spikes -> 128 recurrent adaptive-LIF
    # neurons (memory and policy interface in one) -> decoder. Loss on one map only.
    **{f"mb_{t}_{d}": {"model": {**_A_REC, "bottleneck": 128, "ttc_head": True, "memory": "bottleneck",
                                 "vel_head": False, "depth_head": False, "decoder": {"snn": "spiking", "ann": "ann"}[d]},
                       "loss": {"lambda_d": 0.0, "lambda_ttc": 1.0, "lambda_v": 0.0, "lambda_s": 0.1},
                       "data": {"target_map": t}}
       for t in ("ttc", "near") for d in ("snn", "ann")},
    # Spatial control for the scale-free study: conv3 + inertial spikes -> 8x8 conv-recurrent
    # adaptive-LIF memory grid -> decoder, no 128-neuron bottleneck. The ann decoder reads a leaky
    # trace of the grid's spikes, so only the bottleneck differs from mb_*.
    **{f"grid_{t}_{d}": {"model": {**_A_REC, "bottleneck": 0, "ttc_head": True, "memory": "grid",
                                   "vel_head": False, "depth_head": False, "decoder": {"snn": "spiking", "ann": "ann"}[d]},
                         "loss": {"lambda_d": 0.0, "lambda_ttc": 1.0, "lambda_v": 0.0, "lambda_s": 0.1},
                         "data": {"target_map": t}}
       for t in ("ttc", "near") for d in ("snn", "ann")},
    # Vision-only latent (like the teacher's depth latent, inertial goes to the policy, not the latent):
    # grid_ttc_ann with no inertial input to the memory; *_gyro adds the gyro only as rotational-flow
    # maps at the conv-1 input (derotation prior, spatial), never as a broadcast current.
    "grid_ttc_vis": {"model": {"recurrent": False, "stateful": True, "state_inputs": [], "bottleneck": 0,
                               "ttc_head": True, "memory": "grid", "vel_head": False, "depth_head": False,
                               "decoder": "ann"},
                     "loss": {"lambda_d": 0.0, "lambda_ttc": 1.0, "lambda_v": 0.0, "lambda_s": 0.1},
                     "data": {"target_map": "ttc"}},
}
# Long memory: an obstacle edge r px from the focus of expansion moves r/TTC px/s, so the 2-3 px
# needed to see motion take ~0.3-1 s near the centre at TTC 1.5-3 s. Memory time constants spread
# 50 ms .. 1.5 s over the channels, and BPTT windows long enough to train the slow ones.
VARIANTS["grid_ttc_vis_long"] = copy.deepcopy(VARIANTS["grid_ttc_vis"])
VARIANTS["grid_ttc_vis_long"]["model"]["tau_mem_init_s"] = [0.05, 1.5]
VARIANTS["grid_ttc_vis_long"]["train"] = {"bptt_curriculum_s": [1.0, 2.0]}
VARIANTS["grid_ttc_vis_gyro"] = copy.deepcopy(VARIANTS["grid_ttc_vis"])
VARIANTS["grid_ttc_vis_gyro"]["model"]["gyro_maps"] = True


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _set_dotted(d, key, value):
    *path, last = key.split(".")
    for p in path:
        if p not in d or not isinstance(d[p], dict):
            raise KeyError(f"unknown config key {key!r}")
        d = d[p]
    if last not in d:
        raise KeyError(f"unknown config key {key!r}")
    if isinstance(value, str):
        value = yaml.safe_load(value)
        if isinstance(value, str):
            try:
                value = float(value)  # YAML 1.1 reads "3e-4" as a string
            except ValueError:
                pass
    d[last] = value


def to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_ns(v) for v in d]
    return d


def to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [to_dict(v) for v in ns]
    return ns


def load_config(path=None, variant=None, overrides=()):
    """Default yaml (or `path`) <- variant overlay <- `key.sub=value` overrides. Returns a namespace."""
    with open(path or DEFAULT_PATH) as f:
        d = yaml.safe_load(f)
    if path and path != DEFAULT_PATH:  # a partial yaml overlays the defaults
        with open(DEFAULT_PATH) as f:
            d = _merge(yaml.safe_load(f), d)
    variant = variant or d["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    d["variant"] = variant
    _merge(d, VARIANTS[variant])
    for o in overrides:
        k, _, v = o.partition("=")
        _set_dotted(d, k.strip(), v)
    return to_ns(d)


def steps(seconds, dt=0.01):
    return int(round(seconds / dt))


def heldout_path(cfg):
    d = cfg.data
    return d.heldout_path or os.path.join(cfg.out_dir, f"heldout_level{cfg.sim.curriculum_level}_seed{d.heldout_seed}.pt")
