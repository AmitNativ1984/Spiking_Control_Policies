"""YAML config with variant overlays and dotted command-line overrides."""
import copy
import os
from types import SimpleNamespace

import yaml

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "cfg", "default.yaml")

VARIANTS = {
    "a_rec": {"model": {"recurrent": True, "stateful": True, "state_inputs": ["acc", "gyro", "grav"]}},
    "a_ff": {"model": {"recurrent": False, "stateful": True, "state_inputs": ["acc", "gyro", "grav"]}},
    "b": {"model": {"recurrent": False, "stateful": False, "state_inputs": ["acc", "gyro", "grav"]},
          "loss": {"burn_in_s": 0.0}},
    "a_rec_noimu": {"model": {"recurrent": True, "stateful": True, "state_inputs": ["grav"]}},
    "a_rec_rawimu": {"model": {"recurrent": True, "stateful": True, "state_inputs": ["acc", "gyro"]}},
}


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
