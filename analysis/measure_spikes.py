"""Spike rate and SynOps of a PopSAN navigation checkpoint, under its own closed-loop flight.

Rates are measured on the observations the policy actually produces at a fixed curriculum
level, not on random inputs: the encoder's activity depends entirely on where the
observations fall relative to its receptive fields, so off-distribution inputs give a
meaningless number. Deterministic mean action, clamped, as in eval_levels.py.

The run's frozen config.yaml (beside nn/) supplies num_steps, which the weights cannot --
see deploy/snn_checkpoint.py, whose loader is not reused because it pulls in the flight
stack (control_policy_api). Pass --ann_checkpoint to compare against a dense actor's MACs.

Isaac Gym allows ONE sim per process, so one level per invocation.

usage: python analysis/measure_spikes.py <popsan.pth> <level> [--ann_checkpoint teacher.pth]
                                         [--num_envs 256] [--num_steps 1000] [--out out.json]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import json
import os

import torch
import yaml

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry
from rl_training.rl_games.networks.snn.pop_spiking_actor import PopulationSpikingActorNetwork

try:  # `python -m analysis.measure_spikes`, repo root on sys.path
    from analysis.crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP
    from analysis.spike_ops import SpikeCounter, format_report, linear_macs
except ImportError:  # `python analysis/measure_spikes.py`, sibling on sys.path
    from crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP
    from spike_ops import SpikeCounter, format_report, linear_macs


ACTOR_PREFIX = "a2c_network.spiking_actor."


def load_popsan(ckpt_path, dev):
    w = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    actor_cfg = dict(cfg.get("params", cfg)["network"]["actor"])
    actor_cfg["pop_dim"] = actor_cfg["encoder"]["pop_dim"]
    actor_w = {k[len(ACTOR_PREFIX):]: v for k, v in w.items() if k.startswith(ACTOR_PREFIX)}
    obs_dim = actor_w["pop_encoder.means"].shape[1]
    action_dim = actor_w["log_std"].shape[0]
    net = PopulationSpikingActorNetwork(obs_dim, action_dim, **actor_cfg)
    net.load_state_dict(actor_w, strict=True)
    mean = w["running_mean_std.running_mean"].float().to(dev)
    std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)
    return net.to(dev).eval(), mean, std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("level", type=int)
    p.add_argument("--ann_checkpoint", default=None)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = "cuda:0"
    net, mean, std = load_popsan(args.checkpoint, dev)
    ann_macs = None
    if args.ann_checkpoint:
        aw = torch.load(args.ann_checkpoint, map_location="cpu", weights_only=False)["model"]
        ann_macs = linear_macs(build_actor(aw)[0])

    from config.task_config import F450NavTaskConfig
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level
    task = task_registry.make_task(
        "f450_navigation_task", num_envs=args.num_envs, headless=True, use_warp=True
    )
    assert net.pop_encoder.obs_dim == task.task_config.observation_space_dim, (
        f"checkpoint expects {net.pop_encoder.obs_dim}-D observations but the task produces "
        f"{task.task_config.observation_space_dim}-D")

    counter = SpikeCounter(net)
    arrivals = episodes = 0
    obs = task.reset()[0]["observations"]
    with torch.no_grad():
        for s in range(args.num_steps):
            mu = net({"obs": torch.clamp((obs - mean) / std, -_NORM_CLAMP, _NORM_CLAMP)})[0]
            o, _, term, trunc, infos = task.step(mu.clamp(-1, 1))
            obs = o["observations"]
            arrivals += int(infos["arrivals"].bool().sum())
            episodes += int((term.bool() | trunc.bool()).sum())
            if (s + 1) % 250 == 0:
                print(f"  step {s + 1}/{args.num_steps}  episodes {episodes}  "
                      f"success {100 * arrivals / max(episodes, 1):.1f}%", flush=True)

    report = counter.report(ann_macs)
    report.update({"checkpoint": args.checkpoint, "ann_checkpoint": args.ann_checkpoint,
                   "level": args.level, "episodes": episodes,
                   "success_rate": arrivals / max(episodes, 1)})
    print(f"\nLEVEL {args.level}  |  {episodes} episodes  |  "
          f"success {100 * report['success_rate']:.1f}%")
    print(format_report(report))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    task.close()


if __name__ == "__main__":
    main()
