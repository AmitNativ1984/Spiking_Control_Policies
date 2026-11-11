"""End-to-end event-camera navigation policy (non-spiking) and PPO, without Isaac Gym imports.

Follows "Rendering-Free Event Frame Simulation for Fast Reinforcement Learning of Autonomous
Robot Navigation" (Ou et al., arXiv 2610.37543): the policy sees the 3 most recent signed
64x64 event frames (ON - OFF counts, saturated) plus a small state vector, through a compact
16/32/64 conv trunk (~1.2 M parameters, no recurrence), and is trained with clipped PPO
(rollout 128, 2 epochs, 8 minibatches, gamma 0.99, lambda 0.95, clip 0.2, Adam 3e-4,
entropy 5e-3, value coefficient 0.5).

Different here (see event_snn.e2e_ppo): events are rendered by the simulator, the state is this
repo's 17-D navigation state, the action is its 4-D attitude command and the reward is its
navigation reward, i.e. the task the depth teacher was trained on.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 17   # direction to goal 3, distance 1, velocity 3, gyro 3, gravity 3, previous action 4
N_FRAMES = 3
EVENT_SAT = 3.0  # signed count at which tanh(x / EVENT_SAT) is ~0.76


class EventStack:
    """Per-env history of signed event frames, one per policy decision.

    add(events, done) every simulator tick: events (N, 2, 64, 64) ON/OFF counts. A tick on which
    an env's episode ended is dropped for that env and its history cleared, so the policy never
    sees the previous episode. roll() at each decision closes the current frame.
    """

    def __init__(self, n, device, n_frames=N_FRAMES, size=64):
        self.acc = torch.zeros((n, size, size), device=device)
        self.frames = torch.zeros((n, n_frames, size, size), dtype=torch.int8, device=device)

    def add(self, events, done):
        self.acc += events[:, 0].float() - events[:, 1].float()
        if done.any():
            self.acc[done] = 0.0
            self.frames[done] = 0

    def roll(self):
        """Close the frame of the decision period just ended; returns (N, n_frames, 64, 64) int8, oldest first."""
        self.frames = torch.roll(self.frames, -1, dims=1)
        self.frames[:, -1] = self.acc.clamp(-127, 127).round().to(torch.int8)
        self.acc.zero_()
        return self.frames.clone()


class RunningNorm(nn.Module):
    """Running mean / variance; update() on raw batches, forward() normalises and clips."""

    def __init__(self, shape, clip=5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, *self.mean.shape)
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        tot = self.count + n
        delta = b_mean - self.mean
        self.var.copy_((self.var * self.count + b_var * n + delta ** 2 * self.count * n / tot) / tot)
        self.mean.copy_(self.mean + delta * n / tot)
        self.count.copy_(tot)

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


class EventPolicy(nn.Module):
    """frames (B, 3, 64, 64) int8 signed counts, state (B, 17) -> action mean (B, 4), value (B,)."""

    def __init__(self, state_dim=STATE_DIM, act_dim=4, n_frames=N_FRAMES, channels=(16, 32, 64), hidden=256,
                 log_std_init=-0.7):
        super().__init__()
        c1, c2, c3 = channels
        self.trunk = nn.Sequential(
            nn.Conv2d(n_frames, c1, 5, 2, 2), nn.ELU(),      # 32x32
            nn.Conv2d(c1, c2, 3, 2, 1), nn.ELU(),            # 16x16
            nn.Conv2d(c2, c3, 3, 2, 1), nn.ELU(),            # 8x8
            nn.Flatten(), nn.Linear(c3 * 64, hidden), nn.ELU())
        self.state_norm = RunningNorm((state_dim,))
        self.state_fc = nn.Sequential(nn.Linear(state_dim, 64), nn.ELU())
        self.body = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.ELU())
        self.mu = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.value.weight, 1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, frames, state):
        x = torch.tanh(frames.float() / EVENT_SAT)
        h = self.body(torch.cat([self.trunk(x), self.state_fc(self.state_norm(state))], dim=1))
        return self.mu(h), self.value(h)[:, 0]

    def dist(self, mu):
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @torch.no_grad()
    def act(self, frames, state, deterministic=False):
        mu, v = self(frames, state)
        a = mu if deterministic else self.dist(mu).sample()
        return a, self.dist(mu).log_prob(a).sum(-1), v


def gae(rew, val, done, last_val, gamma=0.99, lam=0.95):
    """rew, val, done (T, N); done[t] = the episode ended on step t (no bootstrap past it).
    Returns advantages and returns (T, N)."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    nxt, run = last_val, torch.zeros_like(last_val)
    for t in range(T - 1, -1, -1):
        live = 1.0 - done[t].float()
        delta = rew[t] + gamma * nxt * live - val[t]
        run = delta + gamma * lam * live * run
        adv[t] = run
        nxt = val[t]
    return adv, adv + val


def ppo_update(policy, opt, batch, epochs=2, minibatches=8, clip=0.2, ent_coef=5e-3, vf_coef=0.5,
               bounds_coef=1e-4, max_grad_norm=1.0):
    """batch: flat tensors frames, state, act, logp, adv, ret. Returns mean diagnostics."""
    n = batch["act"].shape[0]
    mb = max(1, n // minibatches)
    stats = {"pi": 0.0, "v": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
    k = 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["act"].device)
        for i in range(0, n - mb + 1, mb):
            ids = perm[i:i + mb]
            mu, v = policy(batch["frames"][ids], batch["state"][ids])
            d = policy.dist(mu)
            logp = d.log_prob(batch["act"][ids]).sum(-1)
            ratio = (logp - batch["logp"][ids]).exp()
            adv = batch["adv"][ids]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pi_loss = -torch.min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv).mean()
            v_loss = F.mse_loss(v, batch["ret"][ids])
            ent = d.entropy().sum(-1).mean()
            bounds = ((mu.abs() - 1.1).clamp(min=0.0) ** 2).sum(-1).mean()   # as rl_games' bounds loss
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent + bounds_coef * bounds
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                stats["pi"] += float(pi_loss)
                stats["v"] += float(v_loss)
                stats["entropy"] += float(ent)
                stats["kl"] += float((batch["logp"][ids] - logp).mean())
                stats["clip_frac"] += float(((ratio - 1.0).abs() > clip).float().mean())
                stats["grad_norm"] += float(gn)
            k += 1
    return {a: b / max(k, 1) for a, b in stats.items()}


class ReturnScale:
    """Scales rewards by the running std of the discounted return (as VecNormalize does)."""

    def __init__(self, n, device, gamma=0.99):
        self.ret = torch.zeros(n, device=device)
        self.norm = RunningNorm(()).to(device)
        self.gamma = gamma

    def __call__(self, rew, done):
        self.ret = self.ret * self.gamma + rew
        self.norm.update(self.ret)
        self.ret = self.ret * (1.0 - done.float())
        return rew / math.sqrt(float(self.norm.var) + 1e-8)
