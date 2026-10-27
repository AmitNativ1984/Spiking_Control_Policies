"""Why does the PopSAN student's action sigma climb under PPO? An instrumented-epoch diagnostic.

Every post-release fine-tune of the spiking student widened its state-independent log_std at
a steady 15-35% of Adam's maximum step, with or without the entropy bonus, while the ANN
teacher -- same env, same reward, same PPO settings -- narrowed its own. Two hypotheses are
tested here, on real rollouts from a checkpoint, with no training run:

  QUANTIZATION. The spiking actor's mean is a weighted sum of spike COUNTS (5 timesteps,
  so every output rate is a multiple of 0.2). It is piecewise constant in its weights: a
  small update leaves most samples' means untouched and makes a few jump by a whole grid
  step (~0.4-0.5 sigma at ep_450). A smooth ANN mean moves a little everywhere instead.
  Measured: the distribution of mean changes after ONE optimizer step and after a full
  epoch -- fraction unchanged, tail quantiles and kurtosis, in units of sigma -- split into
  the part caused by the weight update and the part caused by the observation normalizer's
  own update in mini-epoch 0.

  CLIP CENSORSHIP. When the mean moves for reasons unrelated to the advantage, samples leave
  PPO's clip window at random. The clip zeroes the gradient of positive-advantage samples
  the mean moved TOWARD (they sit near the new mean and would have voted to shrink sigma)
  and of negative-advantage samples it moved AWAY from (far from the mean; pushing them
  down would also shrink sigma). What survives votes to widen. Measured, at every minibatch
  step: the log_std gradient of the clipped PPO loss, of the unclipped surrogate, and the
  part the clip removed, split by advantage sign; and the actual change of log_std over the
  epoch with the clip on and with it off (e_clip -> 1e6), and with the entropy bonus on
  and off (entropy_coef -> 0), so the clip's and the bonus's shares can be separated.

HOW. The agent is built exactly as the runner builds it (same YAML, same --curriculum_level,
same encoder bounds), restored from the checkpoint, and driven through rl_games' own
play_steps / prepare_dataset / train_actor_critic. Only actor_loss_func is wrapped: it
returns the unchanged loss, so the optimizer steps are the ones training would take. For
each rollout batch, every (lr, clip) trial restarts from the same weights, optimizer state
and normalizers, on a fresh copy of the same batch, so trials differ only in that setting.

Sign convention: "push" = -dLoss/dlog_std, so push > 0 means gradient descent WIDENS sigma.
The entropy bonus alone pushes +entropy_coef per action dim (d entropy / d log_std = 1).

usage (inside the container, repo root):
  python -u analysis/sigma_diag.py --label snn_ep450 --out analysis/data/sigma_diag/snn_ep450.json \\
      [--lrs 1e-6 1e-5 1e-4] [--batches 2] [--no_clip_trials] [--wandb_project P] -- \\
      --file=rl_training/rl_games/cfg/<yaml> --task=f450_navigation_task --checkpoint=<ckpt> \\
      --curriculum_level=30 --headless=True --train --experiment_name=sigma_diag
Everything after "--" goes to the runner's own argument parser unchanged.
"""
import isaacgym  # noqa: F401  -- MUST precede torch

import argparse
import copy
import json
import math
import os
import sys
import time

import torch

# Run as a script (python analysis/sigma_diag.py), the repo root is not on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _split_argv():
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: sigma_diag.py [diag options] -- [runner options]; see the docstring")
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def _diag_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="name of this model in the outputs")
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument("--lrs", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4])
    p.add_argument("--batches", type=int, default=3, help="independent rollout batches")
    p.add_argument("--no_clip_trials", action="store_true",
                   help="skip the e_clip-off counterfactual trials")
    p.add_argument("--no_entropy_trials", action="store_true",
                   help="skip the entropy_coef = 0 counterfactual trials")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_notes", default="")
    return p.parse_args(argv)


def _find_log_std(model, actions_num):
    """The state-independent action log-std parameter (PopSAN: spiking_actor.log_std; this
    repo's ANN actor: actor.action_log_std). The encoder's per-neuron 'log_stds' is plural
    and differently shaped, so it never matches."""
    hits = [(n, p) for n, p in model.named_parameters()
            if n.endswith("log_std") and p.numel() == actions_num]
    if len(hits) != 1:
        raise SystemExit(f"expected one action log_std parameter, found {[n for n, _ in hits]}")
    return hits[0]


def _grid_steps(model):
    """PopSAN only: the mean change caused by ONE output spike (rate step 1/num_steps) in
    each decoder neuron, per action dim -> (min, mean, max) over that dim's population."""
    net = model.a2c_network
    actor = getattr(net, "spiking_actor", None)
    if actor is None:
        return None
    w = actor.action_decoder.decoder.weight.detach().abs().squeeze(1)  # [A, pop]
    step = w / float(actor.num_steps)
    return {"min": step.min(1).values.tolist(), "mean": step.mean(1).tolist(),
            "max": step.max(1).values.tolist(), "num_steps": int(actor.num_steps)}


@torch.no_grad()
def _eval_mu(agent, obs):
    """Mean action for raw obs, in eval mode (no normalizer update), as during rollouts."""
    was_training = agent.model.training
    agent.model.eval()
    res = agent.model({"is_train": False, "obs": agent._preproc_obs(obs), "prev_actions": None})
    if was_training:
        agent.model.train()
    return res["mus"].detach().clone()


def _jump_stats(d, sigma):
    """Distribution of mean changes d [N, A], in units of the (state-independent) sigma [A]."""
    z = (d.abs() / sigma).flatten().float()
    nz = z[z > 1e-6]
    q = lambda t, x: float(torch.quantile(t, x)) if t.numel() else 0.0
    m2 = float((z ** 2).mean())
    return {
        "frac_unchanged": float((d.abs() < 1e-6).float().mean()),
        "rms_over_sigma": math.sqrt(m2),
        "mean_abs_over_sigma": float(z.mean()),
        "p50": q(z, 0.5), "p90": q(z, 0.9), "p99": q(z, 0.99), "max": float(z.max()),
        "p50_of_changed": q(nz, 0.5), "p90_of_changed": q(nz, 0.9),
        "frac_gt_0.1sigma": float((z > 0.1).float().mean()),
        "frac_gt_0.25sigma": float((z > 0.25).float().mean()),
        # E[z^4]/E[z^2]^2: 3 for Gaussian changes, >>3 when a few samples carry all the change.
        "kurtosis": float((z ** 4).mean()) / (m2 ** 2) if m2 > 0 else 0.0,
        "per_dim_rms_over_sigma": ((d / sigma) ** 2).mean(0).sqrt().tolist(),
    }


class LossProbe:
    """Wraps agent.actor_loss_func. Returns the identical per-sample loss; while active,
    also records the log_std gradient of each piece of the PPO actor objective."""

    def __init__(self, agent, log_std):
        self.base = agent.actor_loss_func
        self.log_std = log_std
        self.ref_clip = float(agent.e_clip)
        self.active = False
        self.rec = None

    def __call__(self, old_nlp, new_nlp, adv, is_ppo, e_clip):
        a_loss = self.base(old_nlp, new_nlp, adv, is_ppo, e_clip)
        if not self.active:
            return a_loss
        n = adv.numel()
        ratio = torch.exp(old_nlp - new_nlp)
        surr1 = -adv * ratio
        # The clamp branch wins (and carries no gradient) exactly when it binds against
        # the advantage's sign: A > 0 and ratio > 1+eps, or A < 0 and ratio < 1-eps.
        pos = (adv > 0) & (ratio > 1.0 + e_clip)
        neg = (adv < 0) & (ratio < 1.0 - e_clip)
        g = lambda t: torch.autograd.grad(t, self.log_std, retain_graph=True)[0].detach()
        g_full = g(a_loss.mean())
        g_unclip = g(surr1.mean())
        g_pos = g((surr1 * pos.detach()).sum() / n)
        g_neg = g((surr1 * neg.detach()).sum() / n)
        logr = (old_nlp - new_nlp).detach()
        out = ((logr < math.log(1 - self.ref_clip)) | (logr > math.log(1 + self.ref_clip)))
        self.rec = {
            "push_full": (-g_full).tolist(),
            "push_unclipped": (-g_unclip).tolist(),
            "push_removed_by_clip_Apos": (-g_pos).tolist(),
            "push_removed_by_clip_Aneg": (-g_neg).tolist(),
            "frac_censored_Apos": float(pos.float().mean()),
            "frac_censored_Aneg": float(neg.float().mean()),
            # rl_games' own diagnostics/clip_frac definition (sign-agnostic), always against
            # the config's e_clip -- so the clip-off trials report what WOULD have clipped.
            "clip_frac": float(out.float().mean()),
            "mean_abs_logratio": float(logr.abs().mean()),
        }
        return a_loss


def _run_trial(agent, probe, log_std, batch, lr, e_clip, ent_coef, probe_obs, snap):
    """One instrumented PPO epoch from the snapshot state on a private copy of `batch`."""
    agent.model.load_state_dict(snap["model"])
    agent.optimizer.load_state_dict(snap["optim"])
    for g in agent.optimizer.param_groups:
        g["lr"] = lr
    agent.last_lr = lr
    agent.e_clip = e_clip
    agent.entropy_coef = ent_coef

    b = {k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k, v in batch.items()}
    agent.set_train()
    if agent.normalize_input:
        agent.model.running_mean_std.train()
    b.pop("played_frames", None)
    agent.prepare_dataset(b)

    ls0 = log_std.detach().clone()
    sigma0 = ls0.exp()
    mu0 = _eval_mu(agent, probe_obs)

    steps, first_step = [], None
    probe.active = True
    for mini_ep in range(agent.mini_epochs_num):
        for i in range(len(agent.dataset)):
            rms_before = (copy.deepcopy(agent.model.running_mean_std.state_dict())
                          if agent.normalize_input else None)
            res = agent.train_actor_critic(agent.dataset[i])
            cmu, csigma, kl = res[6], res[7], res[3]
            agent.dataset.update_mu_sigma(cmu, csigma)
            rec = dict(probe.rec, mini_epoch=mini_ep, minibatch=i, kl=float(kl))
            steps.append(rec)

            if first_step is None:
                # Mean change after ONE step, split into weight update vs normalizer update.
                mu_after = _eval_mu(agent, probe_obs)
                split = {}
                if rms_before is not None:
                    rms_after = copy.deepcopy(agent.model.running_mean_std.state_dict())
                    agent.model.running_mean_std.load_state_dict(rms_before)
                    mu_weights_only = _eval_mu(agent, probe_obs)
                    agent.model.running_mean_std.load_state_dict(rms_after)
                    split = {"weights_only": _jump_stats(mu_weights_only - mu0, sigma0),
                             "normalizer_only": _jump_stats(mu_after - mu_weights_only, sigma0)}
                first_step = {"total": _jump_stats(mu_after - mu0, sigma0), **split}
        agent.diagnostics.mini_epoch(agent, mini_ep)  # resets rl_games' per-mini-epoch lists
        if agent.normalize_input:
            agent.model.running_mean_std.eval()  # as rl_games: stats update in mini-epoch 0 only
    probe.active = False

    mu_end = _eval_mu(agent, probe_obs)
    ls1 = log_std.detach().clone()
    n = len(steps)
    mean = lambda k: [sum(s[k][d] for s in steps) / n for d in range(len(ls0))]
    dims_mean = lambda v: sum(v) / len(v)
    push_full = mean("push_full")
    summary = {
        "lr": lr, "e_clip": e_clip, "entropy_coef": ent_coef, "n_steps": n,
        "sigma_before": sigma0.tolist(),
        "dlog_sigma": (ls1 - ls0).tolist(),
        "dlog_sigma_mean": float((ls1 - ls0).mean()),
        # Fraction of Adam's maximum possible move (n_steps * lr) actually taken.
        "dlog_sigma_over_max_step": float((ls1 - ls0).mean()) / (n * lr),
        "push_full": push_full, "push_full_mean": dims_mean(push_full),
        "push_unclipped": mean("push_unclipped"),
        "push_unclipped_mean": dims_mean(mean("push_unclipped")),
        "push_removed_by_clip_Apos_mean": dims_mean(mean("push_removed_by_clip_Apos")),
        "push_removed_by_clip_Aneg_mean": dims_mean(mean("push_removed_by_clip_Aneg")),
        "push_entropy_bonus": float(agent.entropy_coef),
        "frac_steps_push_full_positive": sum(dims_mean(s["push_full"]) > 0 for s in steps) / n,
        "frac_steps_push_unclipped_positive":
            sum(dims_mean(s["push_unclipped"]) > 0 for s in steps) / n,
        "clip_frac_by_mini_epoch": [
            sum(s["clip_frac"] for s in steps if s["mini_epoch"] == m)
            / max(1, sum(1 for s in steps if s["mini_epoch"] == m))
            for m in range(agent.mini_epochs_num)],
        "frac_censored_mean": sum(s["frac_censored_Apos"] + s["frac_censored_Aneg"]
                                  for s in steps) / n,
        "step0_clip_frac": steps[0]["clip_frac"],
        "step0_mean_abs_logratio": steps[0]["mean_abs_logratio"],
        "mean_kl": sum(s["kl"] for s in steps) / n,
        "mu_jump_first_step": first_step,
        "mu_jump_epoch": _jump_stats(mu_end - mu0, sigma0),
    }
    return summary, steps


def main():
    diag_argv, runner_argv = _split_argv()
    dargs = _diag_args(diag_argv)
    sys.argv = [sys.argv[0]] + runner_argv

    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner

    from rl_training.rl_games import runner as R
    from rl_training.rl_games.agents import register_algos

    args = vars(R.get_args())
    if not args.get("checkpoint"):
        sys.exit("--checkpoint is required")
    R.register_task(args["task"])
    config_dict = R.load_config(args)
    runner = Runner(algo_observer=IsaacAlgoObserver())
    register_algos(runner)
    runner.load(config_dict)
    agent = runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)
    agent.restore(args["checkpoint"])
    ck_epoch = int(agent.epoch_num)

    log_std_name, log_std = _find_log_std(agent.model, agent.actions_num)
    # rl_games' PpoDiagnostics takes log(1 - e_clip), which the clip-off trials (e_clip 1e6)
    # cannot survive; LossProbe records the same clip fraction, so swap in the no-op one.
    from rl_games.common.diagnostics import DefaultDiagnostics
    agent.diagnostics = DefaultDiagnostics()
    probe = LossProbe(agent, log_std)
    agent.actor_loss_func = probe
    grid = _grid_steps(agent.model)
    sigma_ck = log_std.detach().exp().tolist()
    print(f"[sigma_diag] {dargs.label}: algo {runner.algo_name}, checkpoint epoch {ck_epoch}, "
          f"{log_std_name} sigma {[round(s, 4) for s in sigma_ck]}", flush=True)
    if grid:
        print(f"[sigma_diag] one-output-spike mean step per dim (mean over population): "
              f"{[round(s, 4) for s in grid['mean']]}  = "
              f"{[round(s / g, 3) for s, g in zip(grid['mean'], sigma_ck)]} sigma", flush=True)

    snap = {"model": copy.deepcopy(agent.model.state_dict()),
            "optim": copy.deepcopy(agent.optimizer.state_dict())}
    e_clip0 = float(agent.e_clip)
    ent0 = float(agent.entropy_coef)
    # (e_clip, entropy_coef) settings: as trained, clip off, entropy off, both off.
    settings = [(e_clip0, ent0)]
    if not dargs.no_clip_trials:
        settings.append((1e6, ent0))
    if not dargs.no_entropy_trials and ent0 > 0:
        settings.append((e_clip0, 0.0))
        if not dargs.no_clip_trials:
            settings.append((1e6, 0.0))

    agent.init_tensors()
    agent.obs = agent.env_reset()
    agent.curr_frames = agent.batch_size_envs

    trials, all_steps = [], []
    for bi in range(dargs.batches):
        agent.model.load_state_dict(snap["model"])
        agent.optimizer.load_state_dict(snap["optim"])
        if hasattr(agent.vec_env, "set_train_info"):
            agent.vec_env.set_train_info(agent.frame, agent)
        agent.set_eval()
        t0 = time.time()
        with torch.no_grad():
            batch = agent.play_steps()
        mb = agent.minibatch_size
        probe_obs = batch["obses"][:mb].clone()
        # Rollout-vs-update consistency before any step: the stored rollout means against a
        # fresh eval-mode forward of the same obs (should be identical for a deterministic net).
        mu_now = _eval_mu(agent, probe_obs)
        consistency = _jump_stats(mu_now - batch["mus"][:mb], log_std.detach().exp())
        print(f"[sigma_diag] batch {bi}: rollout {time.time() - t0:.0f}s, "
              f"rollout-vs-eval mean mismatch rms {consistency['rms_over_sigma']:.2e} sigma",
              flush=True)
        for lr in dargs.lrs:
            for ec, ent in settings:
                t1 = time.time()
                s, steps = _run_trial(agent, probe, log_std, batch, lr, ec, ent, probe_obs, snap)
                s.update(batch=bi, rollout_consistency=consistency)
                trials.append(s)
                for st in steps:
                    all_steps.append(dict(st, batch=bi, lr=lr, e_clip=ec, entropy_coef=ent))
                fj = s["mu_jump_first_step"]["total"]
                print(f"[sigma_diag] batch {bi} lr {lr:.0e} clip {'on ' if ec < 1 else 'off'} "
                      f"ent {ent:g}: "
                      f"dlog_sigma {s['dlog_sigma_mean']:+.2e} ({s['dlog_sigma_over_max_step']:+.2f} of max) | "
                      f"push full {s['push_full_mean']:+.2e} unclipped {s['push_unclipped_mean']:+.2e} "
                      f"removed A+ {s['push_removed_by_clip_Apos_mean']:+.2e} A- {s['push_removed_by_clip_Aneg_mean']:+.2e} "
                      f"(entropy {s['push_entropy_bonus']:+.1e}) | clip_frac/mini-ep "
                      f"{[round(c, 3) for c in s['clip_frac_by_mini_epoch']]} | step-1 mean jump: "
                      f"unchanged {fj['frac_unchanged']:.2f} rms {fj['rms_over_sigma']:.3f}s "
                      f"p99 {fj['p99']:.3f}s kurt {fj['kurtosis']:.1f} | {time.time() - t1:.0f}s",
                      flush=True)

    out = {"label": dargs.label, "checkpoint": args["checkpoint"], "checkpoint_epoch": ck_epoch,
           "config": args["file"], "algo": runner.algo_name, "log_std_param": log_std_name,
           "sigma_checkpoint": sigma_ck, "grid_step": grid, "e_clip": e_clip0,
           "entropy_coef": ent0, "settings": settings, "lrs": dargs.lrs,
           "batches": dargs.batches, "minibatch_size": agent.minibatch_size,
           "mini_epochs": agent.mini_epochs_num, "trials": trials}
    os.makedirs(os.path.dirname(os.path.abspath(dargs.out)), exist_ok=True)
    json.dump(out, open(dargs.out, "w"), indent=1)
    steps_path = dargs.out.replace(".json", "_steps.json")
    json.dump(all_steps, open(steps_path, "w"))
    print(f"[sigma_diag] wrote {dargs.out} and {steps_path}", flush=True)

    if dargs.wandb_project:
        try:
            import wandb
            run = wandb.init(project=dargs.wandb_project, group=dargs.wandb_group,
                             name=f"sigma_diag_{dargs.label}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                             notes=dargs.wandb_notes, job_type="diagnostic",
                             config={k: v for k, v in out.items() if k != "trials"},
                             dir=os.makedirs("runs/sigma_diag", exist_ok=True) or "runs/sigma_diag")
            cols = ["batch", "lr", "e_clip", "entropy_coef", "dlog_sigma_mean", "dlog_sigma_over_max_step",
                    "push_full_mean", "push_unclipped_mean", "push_removed_by_clip_Apos_mean",
                    "push_removed_by_clip_Aneg_mean", "push_entropy_bonus",
                    "frac_steps_push_full_positive", "frac_censored_mean", "mean_kl",
                    "step0_clip_frac"]
            jcols = ["frac_unchanged", "rms_over_sigma", "p99", "kurtosis"]
            table = wandb.Table(columns=cols + [f"step1_{c}" for c in jcols]
                                + [f"epoch_{c}" for c in jcols])
            for t in trials:
                table.add_data(*[t[c] for c in cols],
                               *[t["mu_jump_first_step"]["total"][c] for c in jcols],
                               *[t["mu_jump_epoch"][c] for c in jcols])
            st_cols = ["batch", "lr", "e_clip", "entropy_coef", "mini_epoch", "minibatch", "clip_frac",
                       "frac_censored_Apos", "frac_censored_Aneg", "kl"]
            st_table = wandb.Table(columns=st_cols + ["push_full", "push_unclipped",
                                                      "push_removed_Apos", "push_removed_Aneg"])
            for s in all_steps:
                st_table.add_data(*[s[c] for c in st_cols],
                                  sum(s["push_full"]) / len(s["push_full"]),
                                  sum(s["push_unclipped"]) / len(s["push_unclipped"]),
                                  sum(s["push_removed_by_clip_Apos"]) / len(s["push_full"]),
                                  sum(s["push_removed_by_clip_Aneg"]) / len(s["push_full"]))
            run.log({"trials": table, "steps": st_table})
            run.finish()
        except Exception as exc:  # the JSON is the record; W&B is a convenience
            print(f"[sigma_diag] W&B logging failed: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
