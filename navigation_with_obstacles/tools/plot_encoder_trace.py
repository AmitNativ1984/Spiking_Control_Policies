"""
Debug visualization for the PopSAN PopulationSpikeEncoder.

Plots, per non-VAE observation dimension:
  1. Fixed Gaussian receptive fields (one curve per encoder neuron in that dim).
  2. Histogram of observation values actually seen during a play rollout.
  3. Spike raster across the rollout (snntorch.spikeplot.raster).

Consumed by training/runner.py when invoked with --play --plot-encoding.
"""
import os
import datetime
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt


def plot_encoder_trace(encoder, trace, observation_layout, save_dir=None):
    """Render encoder receptive fields + per-dim activations from a recorded rollout.

    Args:
        encoder: PopulationSpikeEncoder instance (provides means, stds, obs_bounds, pop_dim).
        trace: list of dicts produced by encoder forward when record=True. Each dict has
            "obs" [B, obs_dim], "pop_activity" [B, obs_dim, pop_dim],
            "pop_spikes" [B, obs_dim*pop_dim, num_steps]. We assume B=1 (single env).
        observation_layout: list of (slice, type_name) from task_config.observation_layout.
            VAE entries are skipped.
    """
    print(f"[plot_encoder_trace] called — trace length: {len(trace)}, matplotlib backend: {matplotlib.get_backend()}")
    if not trace:
        print("[plot_encoder_trace] empty trace, nothing to plot. "
              "Either the rollout terminated before any forward pass, or recording was never enabled.")
        return

    # Stack rollout: each entry is one step. B is 1 (single env mode).
    obs_all      = torch.cat([t["obs"]          for t in trace], dim=0).numpy()         # [T, obs_dim]
    spikes_all   = torch.cat([t["pop_spikes"]   for t in trace], dim=0).numpy()         # [T, obs_dim*pop_dim, num_steps]
    activity_all = torch.cat([t["pop_activity"] for t in trace], dim=0).numpy()         # [T, obs_dim, pop_dim]

    T = obs_all.shape[0]
    pop_dim = encoder.pop_dim
    means = encoder.means.detach().cpu().numpy().squeeze(0)        # [obs_dim, pop_dim]
    stds  = encoder.stds.detach().cpu().numpy().squeeze(0)         # [obs_dim, pop_dim]
    bounds = encoder.obs_bounds.detach().cpu().numpy()             # [obs_dim, 2]

    # Build the list of state-only dims (skip VAE).
    state_dims = []   # list of (dim_index, type_name)
    for sl, type_name in observation_layout:
        if type_name == "vae_latent":
            continue
        for d in range(sl.start, sl.stop):
            state_dims.append((d, type_name))

    n = len(state_dims)
    if n == 0:
        print("[plot_encoder_trace] no state dims to plot (layout has only VAE?).")
        return

    # Layout: one row per dim, two columns (receptive fields + spike raster).
    fig, axes = plt.subplots(n, 2, figsize=(18, 3.0 * n), squeeze=False)
    fig.suptitle(f"Encoder trace — {T} rollout steps, pop_dim={pop_dim}, num_steps={encoder.num_steps}", y=1.0)

    for row, (d, type_name) in enumerate(state_dims):
        ax_rf = axes[row, 0]
        ax_sp = axes[row, 1]

        lo, hi = float(bounds[d, 0]), float(bounds[d, 1])
        x = np.linspace(lo, hi, 400)

        # 1. Gaussian receptive fields for this dim.
        for k in range(pop_dim):
            y = np.exp(-0.5 * (x - means[d, k]) ** 2 / max(stds[d, k] ** 2, 1e-8))
            ax_rf.plot(x, y, color="C0", alpha=0.35, linewidth=1.0)
            ax_rf.axvline(means[d, k], color="C0", alpha=0.15, linewidth=0.5)

        # 2. Observation histogram for this dim (assumes B=1 → obs_all[:, d] is the trajectory).
        ax_rf_twin = ax_rf.twinx()
        ax_rf_twin.hist(obs_all[:, d], bins=40, range=(lo, hi),
                        color="C3", alpha=0.35, edgecolor="none")
        ax_rf_twin.set_ylabel("obs count", color="C3", fontsize=8)
        ax_rf_twin.tick_params(axis="y", labelcolor="C3", labelsize=7)

        # Title with mean Gaussian response per neuron (silent neurons get flagged).
        mean_resp = activity_all[:, d, :].mean(axis=0)   # [pop_dim]
        silent = int((mean_resp < 1e-3).sum())
        ax_rf.set_title(f"dim {d}  ({type_name})  | silent neurons: {silent}/{pop_dim}", fontsize=9)
        ax_rf.set_xlabel("obs value (normalized)", fontsize=8)
        ax_rf.set_ylabel("Gaussian response", fontsize=8)
        ax_rf.set_xlim(lo, hi)
        ax_rf.set_ylim(0, 1.05)
        ax_rf.tick_params(labelsize=7)

        # 3. Spike raster for this dim's pop_dim neurons across the full rollout.
        # spikes_all[:, d*pop_dim:(d+1)*pop_dim, :] has shape [T, pop_dim, num_steps].
        # Flatten time × inner-step axis to get [T*num_steps, pop_dim] for raster.
        dim_spikes = spikes_all[:, d * pop_dim:(d + 1) * pop_dim, :]           # [T, pop_dim, num_steps]
        dim_spikes = dim_spikes.transpose(0, 2, 1).reshape(-1, pop_dim)        # [T*num_steps, pop_dim]
        splt.raster(torch.from_numpy(dim_spikes), ax_sp, s=4, c="black")
        ax_sp.set_xlabel("time (rollout_step × num_steps)", fontsize=8)
        ax_sp.set_ylabel("neuron idx", fontsize=8)
        ax_sp.set_title("spike raster", fontsize=9)
        ax_sp.tick_params(labelsize=7)
        ax_sp.set_ylim(-0.5, pop_dim - 0.5)

    fig.tight_layout()

    # Always save to disk so a non-interactive (agg) matplotlib backend doesn't lose the plot.
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"encoder_trace_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_encoder_trace] saved figure → {out_path}")

    # If an interactive backend is available, also pop up the window. With agg this is a no-op.
    if matplotlib.get_backend().lower() != "agg":
        plt.show()
    else:
        print(f"[plot_encoder_trace] matplotlib backend is 'agg' (non-interactive); "
              f"no window will open. View the PNG above.")
