"""Loss of docs/event_snn_experiment_plan.md section 3 and the shared depth/velocity error metrics."""
import torch

from .data import log_rho


def episode_age(done, age0=None):
    """(T, B) long: steps since the network state was last reset, at each step of the window.
    age0 (B,) is the age at the end of the previous window (None: the state is cold, age 0).
    A done step resets the age to 0."""
    T, B = done.shape
    a = torch.zeros(B, dtype=torch.long, device=done.device) if age0 is None else age0.to(done.device)
    age = torch.empty((T, B), dtype=torch.long, device=done.device)
    for t in range(T):
        a = torch.where(done[t], torch.zeros_like(a), a + 1)
        age[t] = a
    return age


def valid_mask(done, burn_in_steps, age0=None):
    """(T, B) bool: steps that count. Burn-in applies only after a state reset (an episode start,
    or a cold state when age0 is None); state carried from the previous window is already warm,
    so a window boundary alone excludes nothing."""
    if burn_in_steps <= 0:
        return torch.ones_like(done)
    return episode_age(done, age0) > burn_in_steps


def scale_invariant(d, m=None):
    """Eigen et al. 2014 per image, over the cells in m (all if None): mean(d^2) - mean(d)^2."""
    m = torch.ones_like(d) if m is None else m.to(d.dtype)
    n = m.sum((-2, -1)).clamp(min=1.0)
    mean = (d * m).sum((-2, -1)) / n
    return (d * d * m).sum((-2, -1)) / n - mean * mean


def ttc_loss(pred, ttc, eps, ttc_max):
    """(T, B) weighted L1 on log(invTTC + eps); cell weight 1 + ttc/ttc_max, so imminent contact
    counts up to twice. Every cell counts: 'not closing' (0) is information too."""
    w = 1.0 + ttc / ttc_max
    return ((pred - torch.log(ttc + eps)).abs() * w).sum((-2, -1)) / w.sum((-2, -1))


def depth_velocity_loss(pred_depth, pred_vel, rates, batch, cfg_loss, eps, burn_in_steps,
                        pred_ttc=None, ttc_eps=0.05, ttc_max=10.0, age0=None):
    """pred_depth (T, B, 32, 32) = log(rho_hat + eps); pred_vel (T, B, 3); batch holds
    rho, vel, done (and ttc when lambda_ttc > 0) with the same leading (T, B). Every cell of the
    current image counts equally in the depth term."""
    valid = valid_mask(batch["done"], burn_in_steps, age0).to(pred_depth.dtype)
    nv = valid.sum().clamp(min=1.0)

    d = pred_depth - log_rho(batch["rho"].to(pred_depth.dtype), eps)
    per_image = d.abs().mean((-2, -1)) + cfg_loss.lambda_si * scale_invariant(d)
    depth = (per_image * valid).sum() / nv

    lam_d, lam_ttc = getattr(cfg_loss, "lambda_d", 1.0), getattr(cfg_loss, "lambda_ttc", 0.0)
    ttc = depth.new_zeros(())
    if lam_ttc > 0:
        ttc = (ttc_loss(pred_ttc, batch["ttc"].to(pred_depth.dtype), ttc_eps, ttc_max) * valid).sum() / nv

    vel = (((pred_vel - batch["vel"]) ** 2).sum(-1) * valid).sum() / nv
    # an output that is not trained has no meaningful error: report 0, not the error of an untrained head
    if lam_d == 0:
        depth = depth.new_zeros(())
    if cfg_loss.lambda_v == 0:
        vel = vel.new_zeros(())
    rate = sum(((r - cfg_loss.rate_target) ** 2 for r in rates.values()), depth.new_zeros(()))
    total = lam_d * depth + lam_ttc * ttc + cfg_loss.lambda_v * vel + cfg_loss.lambda_s * rate
    return total, {"loss": total.detach(), "depth": depth.detach(), "ttc": ttc.detach(), "vel": vel.detach(),
                   "rate_reg": rate.detach(), "valid_frac": valid.mean().detach()}
