"""Deterministic outcome breakdown of a checkpoint at a fixed curriculum level.

crash_cause_eval.py answers WHY collisions happen and reports a crash rate; this answers
the complementary question -- what fraction of episodes arrive, crash, time out or leave
the bounds -- which is what you need to ask whether a policy pinned at one curriculum level
generalises to another, or to compare two policies trained under different rewards.

HOW THE FOUR OUTCOMES ARE SEPARATED. The task puts arrive and exceed in BOTH termination
buffers, so the flags alone are ambiguous; infos["arrivals"] exists precisely to break the
tie (see _update_infos, which marks it functional rather than logging):

    arrive     infos["arrivals"] > 0
    exceed     terminations AND truncations AND NOT arrive
    collision  terminations AND NOT truncations
    timeout    truncations AND NOT terminations

MLP AND RECURRENT CHECKPOINTS BOTH WORK. A feed-forward checkpoint is rebuilt by
crash_cause_eval.build_actor. A GRU checkpoint is rebuilt through the project's own
GRUActorCriticNetworkBuilder, configured from the run's config.yaml (found next to the nn/
directory), because hand-rolling a recurrent trunk would be a second implementation to keep
in sync. The hidden state is carried across steps and ZEROED for every env whose episode
ended on the previous step -- without that the policy would enter a new episode still
conditioned on the last one, which reads as a much worse policy and is silent.

No gradient updates and no exploration noise: the mean action is used, clamped, so this
measures the policy rather than the sampler. p_cbf is not evaluated (lambda_cbf defaults to
0.0), so the clearance probe is never built and the rollout costs nothing extra.

Isaac Gym allows ONE sim per process, so one level per invocation.

usage: python analysis/eval_levels.py <checkpoint.pth> <level> [--num_envs 256]
                                      [--num_steps 3000] [--out out.json]
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import json
import os

import torch

import config  # noqa: F401
from aerial_gym.registry.task_registry import task_registry

try:  # `python -m analysis.eval_levels`, repo root on sys.path
    from analysis.crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP
except ImportError:  # `python analysis/eval_levels.py`, sibling on sys.path
    from crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP


def _run_config_for(ckpt_path):
    """The config.yaml rl_games wrote beside the run's nn/ directory."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"need the run's config.yaml to rebuild a recurrent net: {path}")
    import yaml
    c = yaml.safe_load(open(path))
    return c.get("params", c)


def build_gru_policy(weights, num_envs, dev, ckpt_path):
    """Rebuild the recurrent actor through the project's own builder."""
    from rl_training.rl_games.networks import GRUActorCriticNetworkBuilder

    params = _run_config_for(ckpt_path)
    builder = GRUActorCriticNetworkBuilder()
    builder.load(params["network"])

    # input_shape comes from the task, not the checkpoint, and is asserted against the
    # trunk's own first layer below -- so a mismatch cannot pass silently.
    obs_dim = None
    for k, v in weights.items():
        if k.endswith("actor_net.0.weight"):
            obs_dim = v.shape[1]
    if obs_dim is None:
        raise SystemExit("could not find actor_net.0.weight to infer the observation dim")

    net = builder.build(
        "network",
        input_shape=(obs_dim,),
        actions_num=weights["a2c_network.action_head.weight"].shape[0],
        num_seqs=num_envs,
    )

    inner = {k[len("a2c_network."):]: v for k, v in weights.items()
             if k.startswith("a2c_network.")}
    missing, unexpected = net.load_state_dict(inner, strict=False)
    actor_missing = [k for k in missing
                     if k.startswith(("actor_net", "actor_gru", "action_head"))]
    print(f"  GRU rebuilt: obs_dim {obs_dim}, "
          f"gru {net.gru_num_layers}x{net.gru_hidden_size}, "
          f"critic_gru {net.has_critic_gru}", flush=True)
    if missing or unexpected:
        print(f"  state_dict missing {len(missing)} / unexpected {len(unexpected)} keys",
              flush=True)
    assert not actor_missing, (
        f"actor weights missing from the checkpoint: {actor_missing} -- the rebuilt network "
        f"does not match what was trained, so results would be meaningless."
    )
    net = net.to(dev).eval()
    return net, obs_dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("level", type=int)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=3000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    dev = "cuda:0"
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    w = ck["model"]
    recurrent = any("actor_gru" in k for k in w)
    print(f"checkpoint {os.path.basename(args.checkpoint)}  epoch {ck.get('epoch', '?')}",
          flush=True)
    print(f"policy: {'GRU (recurrent)' if recurrent else 'MLP (feed-forward)'}", flush=True)

    mean = w["running_mean_std.running_mean"].float().to(dev)
    std = torch.sqrt(w["running_mean_std.running_var"].float().to(dev) + _NORM_EPS)

    if recurrent:
        net, obs_dim = build_gru_policy(w, args.num_envs, dev, args.checkpoint)
    else:
        actor, _ = build_actor(w)
        actor = actor.to(dev).eval()
        obs_dim = actor[0].in_features

    from config.task_config import F450NavTaskConfig
    F450NavTaskConfig.curriculum.min_level = args.level
    F450NavTaskConfig.curriculum.max_level = args.level

    task = task_registry.make_task(
        "f450_navigation_task", num_envs=args.num_envs, headless=True, use_warp=True
    )
    assert obs_dim == task.task_config.observation_space_dim, (
        f"checkpoint expects {obs_dim}-D observations but this tree produces "
        f"{task.task_config.observation_space_dim}-D -- results would be garbage."
    )
    print(f"level {args.level}, {args.num_envs} envs, {args.num_steps} steps", flush=True)

    n = {"arrive": 0, "crash": 0, "timeout": 0, "exceed": 0}
    obs = task.reset()[0]["observations"]
    states = None
    if recurrent:
        states = tuple(s.to(dev) for s in net.get_default_rnn_state())

    with torch.no_grad():
        for s in range(args.num_steps):
            normed = torch.clamp((obs - mean) / std, -_NORM_CLAMP, _NORM_CLAMP)
            if recurrent:
                mu, _, _, states = net({"obs": normed, "rnn_states": states,
                                        "seq_length": 1})
            else:
                mu = actor(normed)

            o, _, term, trunc, infos = task.step(mu.clamp(-1, 1))
            obs = o["observations"]

            term = term.bool()
            trunc = trunc.bool()
            arrive = infos["arrivals"].bool()
            n["arrive"] += int(arrive.sum())
            n["crash"] += int((term & ~trunc).sum())
            n["timeout"] += int((trunc & ~term).sum())
            n["exceed"] += int((term & trunc & ~arrive).sum())

            if recurrent:
                # Zero the hidden state of every env whose episode just ended. Carrying it
                # into the next episode is silent and just looks like a worse policy.
                ended = term | trunc
                if ended.any():
                    states = tuple(s.clone() for s in states)
                    for st in states:
                        st[:, ended, :] = 0.0

            if (s + 1) % 500 == 0:
                tot = sum(n.values())
                print(f"  step {s + 1}/{args.num_steps}  episodes {tot}  "
                      f"success {100 * n['arrive'] / max(tot, 1):.2f}%", flush=True)

    tot = sum(n.values())
    rates = {k: v / max(tot, 1) for k, v in n.items()}
    print(f"\nLEVEL {args.level}  |  {tot} episodes")
    for k in ("arrive", "crash", "timeout", "exceed"):
        print(f"  {k:8s} {n[k]:6d}   {100 * rates[k]:6.2f}%")
    print(f"  success rate {100 * rates['arrive']:.2f}%   "
          f"crash rate {100 * rates['crash']:.2f}%")

    if args.out:
        json.dump({
            "checkpoint": args.checkpoint, "epoch": ck.get("epoch"),
            "recurrent": recurrent, "level": args.level,
            "num_envs": args.num_envs, "num_steps": args.num_steps,
            "episodes": tot, "counts": n, "rates": rates,
        }, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
    task.close()


if __name__ == "__main__":
    main()
