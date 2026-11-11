"""End-to-end navigation from the event camera with PPO (non-spiking policy). A test.

Recipe from Ou et al., "Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of
Autonomous Robot Navigation" (arXiv 2610.37543): PPO on the 3 latest signed 64x64 event frames
+ a small state vector, compact conv trunk, no recurrence (see event_snn.e2e_policy).

What is this repo's and not the paper's:
  * events are rendered by the simulator (the paper's point is an analytic, rendering-free event
    model that is ~400x faster; its own rendered-events PPO baseline got 40-53 % after 1 GPU-day);
  * task, reward, 4-D attitude action, 17-D state and curriculum are the navigation task the
    depth teacher (f450_nav_lat_pfov) was trained on; the reward is summed over the 10 ms ticks
    of one decision (the teacher's training stepped 30 ms at a time);
  * one decision every sim.teacher_every ticks (30 ms, the teacher's rate; the paper: 75 ms),
    one event frame per decision (30 ms window; the paper: 20 ms windows);
  * 256 envs by default (the paper: 512; the teacher: 4096), rewards scaled by the running
    return std (the paper: fixed 0.05).

    python -m event_snn.e2e_ppo --hours 30 --wandb                      # train (task curriculum from level 0)
    python -m event_snn.e2e_ppo --eval runs/event_snn/e2e_ppo/<run>/last.pt --episodes 1000 --wandb
    python -m event_snn.e2e_ppo --eval_teacher --episodes 1000 --wandb  # the depth teacher, same loop
"""
import isaacgym  # noqa: F401  -- must precede torch

import argparse
import json
import os
import time

import torch

from .config import load_config, to_dict
from .data import events_to_input
from .e2e_policy import STATE_DIM, EventPolicy, EventStack, ReturnScale, gae, ppo_update
from .sim import Rollout


class EventNavEnv:
    """The navigation task behind a (frames, state) observation; one step() = one decision."""

    def __init__(self, cfg, num_envs, seed):
        cfg.perturb.rate_hz = 0.0                      # nobody but the policy flies
        self.ro = Rollout(cfg, num_envs=num_envs, seed=seed)
        self.task, self.n, self.dev = self.ro.task, self.ro.n, self.ro.dev
        self.every, self.clip = int(cfg.sim.teacher_every), cfg.data.event_clip
        self.stack = EventStack(self.n, self.dev)
        self.state = self.ro.obs[:, :STATE_DIM].clone()
        self.frames = self.stack.roll()
        self.ticks = 0

    @property
    def level(self):
        return int(self.task.curriculum_level)

    @torch.no_grad()
    def step(self, action=None):
        """action (N, 4) in [-1, 1], held for `every` ticks; None = the depth teacher flies.
        Returns frames (N, 3, 64, 64) int8, state (N, 17), reward (N,) summed until the episode
        ends, done (N,) bool, info with arrive / crash / timeout (N,) bool for the ended episodes."""
        n, dev = self.n, self.dev
        if action is None:
            self.ro._teacher_step()
            action = self.ro.action
        a = action.clamp(-1.0, 1.0)
        rew = torch.zeros(n, device=dev)
        done = torch.zeros(n, dtype=torch.bool, device=dev)
        arrive, crash, timeout = done.clone(), done.clone(), done.clone()
        for _ in range(self.every):
            o, r, term, trunc, infos = self.task.step(a)
            self.ro.obs = o["observations"]
            term, trunc = term.bool(), trunc.bool()
            d = term | trunc
            live = ~done
            rew += r.float() * live
            arrive |= infos["arrivals"].bool() & live
            crash |= term & ~trunc & live
            timeout |= infos["time_outs"].bool() & live
            done |= d
            ev = events_to_input(self.ro.env.global_tensor_dict["event_frames"][:, -1], self.clip)
            self.stack.add(ev, d)
            self.ticks += n
        self.frames = self.stack.roll()
        self.state = self.ro.obs[:, :STATE_DIM].clone()
        return self.frames, self.state, rew, done, dict(arrive=arrive, crash=crash, timeout=timeout)


def load_policy(path, device):
    ck = torch.load(path, map_location="cpu")
    policy = EventPolicy().to(device)
    policy.load_state_dict(ck["model"])
    return policy.eval(), ck


@torch.no_grad()
def evaluate(env, policy, episodes, max_decisions=200000):
    """Closed loop, deterministic. Counts only episodes that start after the evaluation begins.
    policy None = the depth teacher."""
    started = torch.zeros(env.n, dtype=torch.bool, device=env.dev)
    c = {"arrive": 0, "crash": 0, "timeout": 0, "other": 0}
    speed, k = 0.0, 0
    frames, state = env.frames, env.state
    while sum(c.values()) < episodes and k < max_decisions:
        a = None if policy is None else policy.act(frames, state, deterministic=True)[0]
        frames, state, _, done, info = env.step(a)
        cnt = done & started
        c["arrive"] += int((info["arrive"] & cnt).sum())
        c["crash"] += int((info["crash"] & cnt).sum())
        c["timeout"] += int((info["timeout"] & cnt).sum())
        c["other"] += int((cnt & ~info["arrive"] & ~info["crash"] & ~info["timeout"]).sum())
        started |= done
        speed += float(state[:, 4:7].norm(dim=-1).mean())
        k += 1
    tot = max(sum(c.values()), 1)
    out = {f"{key}_pct": 100.0 * v / tot for key, v in c.items()}
    out.update(episodes=sum(c.values()), decisions=k, mean_speed=speed / max(k, 1), level=env.level)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. sim.curriculum_level=10")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--rollout", type=int, default=128, help="decisions per env per iteration")
    p.add_argument("--iterations", type=int, default=None, help="stop after this many (default: --hours)")
    p.add_argument("--hours", type=float, default=30.0, help="training wall-clock budget")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--ent_coef", type=float, default=5e-3)
    p.add_argument("--fixed_level", action="store_true",
                   help="train at sim.curriculum_level only (default: the task's curriculum from level 0)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--eval", default=None, metavar="CKPT", help="evaluate a checkpoint at sim.curriculum_level")
    p.add_argument("--eval_teacher", action="store_true", help="evaluate the depth teacher in the same loop")
    p.add_argument("--episodes", type=int, default=1000, help="evaluation episodes")
    p.add_argument("--wandb", action="store_true")
    a = p.parse_args()

    evaluating = bool(a.eval or a.eval_teacher)
    over = list(a.set)
    if not evaluating and not a.fixed_level:
        over.append("sim.curriculum_free=true")
    cfg = load_config(a.config, overrides=over)
    cfg.seed = a.seed
    torch.manual_seed(a.seed)
    dev = cfg.device
    env = EventNavEnv(cfg, a.envs, seed=a.seed + (777 if evaluating else 0))

    if evaluating:
        policy, who = (None, "depth teacher") if a.eval_teacher else (load_policy(a.eval, dev)[0], a.eval)
        res = evaluate(env, policy, a.episodes)
        print(f"closed-loop evaluation of {who} at level {res['level']}: {res['episodes']} episodes, "
              f"arrive {res['arrive_pct']:.1f}%  crash {res['crash_pct']:.1f}%  timeout {res['timeout_pct']:.1f}%  "
              f"other {res['other_pct']:.1f}%  mean speed {res['mean_speed']:.2f} m/s", flush=True)
        out_dir = os.path.dirname(os.path.abspath(a.eval)) if a.eval else os.path.join(cfg.out_dir, "e2e_ppo")
        os.makedirs(out_dir, exist_ok=True)
        tag = "teacher" if a.eval_teacher else os.path.splitext(os.path.basename(a.eval))[0]
        with open(os.path.join(out_dir, f"eval_{tag}_level{res['level']}.json"), "w") as f:
            json.dump(res, f, indent=1)
        if a.wandb:
            import wandb
            run = os.path.basename(out_dir) if a.eval else "teacher"
            wb = wandb.init(project="event_snn", group="e2e_eval", name=f"eval_{run}_{tag}_level{res['level']}",
                            dir=out_dir, config={"eval": vars(a)},
                            notes=os.environ.get("WANDB_NOTES") or
                            f"closed-loop evaluation at level {res['level']}, {res['episodes']} episodes: {who}")
            wb.summary.update({f"eval/{k}": v for k, v in res.items()})
            wb.finish()
        return

    policy = EventPolicy().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=a.lr, eps=1e-5)
    n_par = sum(q.numel() for q in policy.parameters())
    name = a.name or f"e2e_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    out = os.path.join(cfg.out_dir, "e2e_ppo", name)
    os.makedirs(out, exist_ok=True)
    N, T = env.n, a.rollout
    print(f"e2e PPO from events: {n_par} parameters, {N} envs x {T} decisions ({env.every} ticks each) per "
          f"iteration, start level {env.level}, budget {a.hours} h -> {out}", flush=True)
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project="event_snn", group="e2e_ppo", name=name, dir=out,
                        config={**to_dict(cfg), "e2e": vars(a), "parameters": n_par},
                        notes=os.environ.get("WANDB_NOTES") or
                        "end-to-end navigation from event frames with PPO (non-spiking), after arXiv 2610.37543; "
                        "task/reward/actions of the depth teacher f450_nav_lat_pfov")

    buf = dict(frames=torch.zeros((T, N, 3, 64, 64), dtype=torch.int8, device=dev),
               state=torch.zeros((T, N, STATE_DIM), device=dev), act=torch.zeros((T, N, 4), device=dev),
               logp=torch.zeros((T, N), device=dev), val=torch.zeros((T, N), device=dev),
               rew=torch.zeros((T, N), device=dev), done=torch.zeros((T, N), dtype=torch.bool, device=dev))
    scale = ReturnScale(N, dev, a.gamma)
    ep_ret = torch.zeros(N, device=dev)
    frames, state = env.frames, env.state
    t_start, it = time.time(), 0

    def save(tag):
        torch.save({"model": policy.state_dict(), "cfg": to_dict(cfg), "args": vars(a), "it": it,
                    "ticks": env.ticks, "level": env.level}, os.path.join(out, f"{tag}.pt"))

    while True:
        it += 1
        t0, ticks0 = time.time(), env.ticks
        c = {"arrive": 0, "crash": 0, "timeout": 0, "ended": 0}
        ret_sum = 0.0
        for t in range(T):
            policy.state_norm.update(state)
            act, logp, val = policy.act(frames, state)
            buf["frames"][t], buf["state"][t], buf["act"][t], buf["logp"][t], buf["val"][t] = \
                frames, state, act, logp, val
            frames, state, rew, done, info = env.step(act)
            ep_ret += rew
            # an episode cut by the time limit is not a failure: bootstrap with the value of its last state
            buf["rew"][t] = scale(rew, done) + a.gamma * val * info["timeout"].float()
            buf["done"][t] = done
            c["arrive"] += int(info["arrive"].sum())
            c["crash"] += int(info["crash"].sum())
            c["timeout"] += int(info["timeout"].sum())
            c["ended"] += int(done.sum())
            ret_sum += float(ep_ret[done].sum())
            ep_ret[done] = 0.0
        t_collect = time.time() - t0
        last_val = policy.act(frames, state)[2]
        adv, ret = gae(buf["rew"], buf["val"], buf["done"], last_val, a.gamma, a.lam)
        flat = {k: buf[k].flatten(0, 1) for k in ("frames", "state", "act", "logp")}
        flat["adv"], flat["ret"] = adv.flatten(), ret.flatten()
        policy.train()
        st = ppo_update(policy, opt, flat, a.epochs, a.minibatches, ent_coef=a.ent_coef)
        policy.eval()
        ev = float(1.0 - (flat["ret"] - buf["val"].flatten()).var() / flat["ret"].var().clamp(min=1e-8))

        n_end = max(c["ended"], 1)
        hours = (time.time() - t_start) / 3600.0
        log = {"train/arrive_pct": 100.0 * c["arrive"] / n_end, "train/crash_pct": 100.0 * c["crash"] / n_end,
               "train/timeout_pct": 100.0 * c["timeout"] / n_end, "train/episodes": c["ended"],
               "train/episode_return": ret_sum / n_end, "train/level": env.level,
               "train/reward_per_decision": float(buf["rew"].mean()), "train/action_std": float(policy.log_std.exp().mean()),
               "train/explained_variance": ev, "time/ticks_per_s": (env.ticks - ticks0) / t_collect,
               "time/collect_s": t_collect, "time/update_s": time.time() - t0 - t_collect, "time/hours": hours,
               "time/ticks_total": env.ticks, **{f"ppo/{k}": v for k, v in st.items()}}
        if wb:
            wb.log(log, step=it)
        if it % 5 == 0 or it == 1:
            print(f"it {it:5d}  level {env.level:2d}  arrive {log['train/arrive_pct']:5.1f}%  crash "
                  f"{log['train/crash_pct']:5.1f}%  ret {log['train/episode_return']:7.2f}  std "
                  f"{log['train/action_std']:.2f}  kl {st['kl']:.4f}  ev {ev:.2f}  "
                  f"{log['time/ticks_per_s']:.0f} ticks/s  {hours:.2f} h", flush=True)
        if it % 25 == 0:
            save("last")
        if it % 500 == 0:
            save(f"it_{it:06d}")
        if (a.iterations and it >= a.iterations) or (not a.iterations and hours >= a.hours):
            break
    save("last")
    if wb:
        wb.summary.update({"final_level": env.level, "ticks_total": env.ticks})
        wb.finish()
    print(f"done: {it} iterations, {env.ticks / 1e6:.1f} M ticks, level {env.level}; checkpoint {out}/last.pt")


if __name__ == "__main__":
    main()
