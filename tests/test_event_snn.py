"""Unit tests for event_snn (no Isaac Gym): shapes, state carry vs reset, loss masks, overfit."""
import math

import pytest
import torch

from event_snn.config import VARIANTS, load_config
from event_snn.data import (AttitudeNoise, EventMaskTracker, events_to_input, inverse_depth_target,
                            regime_labels, rotate)
from event_snn.loss import depth_velocity_loss, scale_invariant, valid_mask
from event_snn.model import EventSNN, Monitor, count_parameters, surrogate_half_width

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_model(variant="a_rec", **model_over):
    cfg = load_config(variant=variant)
    for k, v in model_over.items():
        setattr(cfg.model, k, v)
    torch.manual_seed(0)
    m = EventSNN(cfg.model).to(DEV)
    m.set_input_bounds([-20.0] * 3 + [-3.0] * 3 + [-1.0] * 3, [20.0] * 3 + [3.0] * 3 + [1.0] * 3)
    return cfg, m


def random_batch(T, B, seed=0, p_event=0.05):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ev = (torch.rand((T, B, 2, 64, 64), generator=g) < p_event).to(torch.uint8) * 3
    st = torch.randn((T, B, 9), generator=g)
    st[..., 6:9] = torch.tensor([0.0, 0.0, -1.0])
    return ev.to(DEV), st.to(DEV), torch.zeros((T, B), dtype=torch.bool, device=DEV)


# --- config ---------------------------------------------------------------------------

def test_variants_load():
    for v in VARIANTS:
        cfg = load_config(variant=v)
        assert cfg.variant == v
    assert load_config(variant="b").loss.burn_in_s == 0.0
    assert load_config(variant="a_ff").model.recurrent is False
    assert load_config(overrides=["train.lr=3e-4"]).train.lr == pytest.approx(3e-4)
    with pytest.raises(KeyError):
        load_config(overrides=["train.not_a_key=1"])


# --- preprocessing --------------------------------------------------------------------

def test_events_to_input_split_pool_clip():
    f = torch.zeros((1, 4, 4), dtype=torch.int8)
    f[0, 0, 0], f[0, 0, 1], f[0, 1, 0] = 3, 2, -1  # one 2x2 block: ON 5, OFF 1
    f[0, 2:, 2:] = 4                                # ON 16 -> clipped to 7
    x = events_to_input(f, clip=7)
    assert x.shape == (1, 2, 2, 2) and x.dtype == torch.uint8
    assert x[0, :, 0, 0].tolist() == [5, 1]
    assert x[0, :, 1, 1].tolist() == [7, 0]
    assert int(x[0, :, 0, 1].sum()) == 0


def test_inverse_depth_target():
    d = torch.full((1, 128, 128), 2.0)
    d[0, :4, :4] = 20.0   # sky / beyond the far plane
    d[0, :4, 4:8] = 0.01  # closer than min_depth
    rho = inverse_depth_target(d, 32, far_m=20.0, min_m=0.1)
    assert rho.shape == (1, 32, 32)
    assert rho[0, 0, 0] == 0.0
    assert rho[0, 0, 1] == pytest.approx(10.0)
    assert rho[0, 5, 5] == pytest.approx(0.5)
    mx = inverse_depth_target(d, 32, far_m=20.0, pool="max")
    assert mx[0, 5, 5] == pytest.approx(0.5)


def test_mask_tracker_now_recent_and_reset():
    tr = EventMaskTracker(1, 32, recent_steps=10, device="cpu")
    ev = torch.zeros((1, 2, 64, 64), dtype=torch.uint8)
    ev[0, 0, 0, 0] = 1
    no = torch.zeros(1, dtype=torch.bool)
    now, rec = tr.update(ev, no)
    assert now[0, 0, 0] and not rec[0, 0, 0] and int(now.sum()) == 1
    for k in range(1, 10):
        now, rec = tr.update(torch.zeros_like(ev), no)
        assert not now[0, 0, 0] and bool(rec[0, 0, 0]) == (k < 10)
    now, rec = tr.update(torch.zeros_like(ev), no)
    assert not rec[0, 0, 0]  # 10 steps = 100 ms later
    tr.update(ev, no)
    now, rec = tr.update(torch.zeros_like(ev), torch.ones(1, dtype=torch.bool))
    assert not rec.any() and not now.any()  # new episode forgets the old events


def test_attitude_noise_magnitude():
    g = torch.tensor([[0.0, 0.0, -1.0]]).repeat(4000, 1)
    an = AttitudeNoise(4000, noise_deg=2.0, bias_init_deg=0.0, bias_rw_deg=0.0,
                       generator=torch.Generator().manual_seed(0))
    est = an(g, torch.zeros(4000, dtype=torch.bool))
    ang = torch.rad2deg(torch.acos((est * g).sum(-1).clamp(-1, 1)))
    assert torch.allclose(est.norm(dim=-1), torch.ones(4000), atol=1e-5)
    assert math.sqrt(float((ang ** 2).mean())) == pytest.approx(2.0, rel=0.1)
    v = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(rotate(v, torch.tensor([[0.0, 0.0, math.pi / 2]])),
                          torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)


def test_regime_labels():
    vel = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]])
    yaw = torch.tensor([0.0, 0.0, 1.0, 0.0])
    rho = torch.zeros((4, 32, 32))
    rho[3, 16, 16] = 1.0  # 1 m straight ahead
    assert regime_labels(vel, yaw, rho).tolist() == [0, 1, 2, 3]


# --- model ----------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_shapes_and_param_count(variant):
    _, m = make_model(variant)
    ev, st, done = random_batch(5, 3)
    d, v, s, rates = m.unroll(ev, st, done, m.init_state(3))
    assert d.shape == (5, 3, 32, 32) and v.shape == (5, 3, 3)
    assert set(rates) == {"c1", "c2", "c3", "mem", "d1"}
    if variant == "a_rec":
        assert 100_000 < count_parameters(m) < 140_000


def _drive(m, T=30, B=2):
    ev, st, done = random_batch(T, B, p_event=0.2)
    with torch.no_grad():
        return m.unroll(ev, st, done, m.init_state(B)), (ev, st, done)


def test_state_carries_in_stateful_variant():
    _, m = make_model("a_rec")
    (_, _, s, _), (ev, st, _) = _drive(m)
    one = torch.zeros((1, 2), dtype=torch.bool, device=DEV)
    with torch.no_grad():
        d_warm, v_warm, _, _ = m.unroll(ev[:1], st[:1], one, s)
        d_cold, v_cold, _, _ = m.unroll(ev[:1], st[:1], one, m.init_state(2))
    assert not torch.allclose(d_warm, d_cold) or not torch.allclose(v_warm, v_cold)


def test_variant_b_leaks_no_state():
    """B resets every state, readouts included: output depends only on the current input."""
    _, m = make_model("b")
    (_, _, s, _), (ev, st, _) = _drive(m)
    s = {k: v + 5.0 for k, v in s.items()}  # garbage history
    one = torch.zeros((1, 2), dtype=torch.bool, device=DEV)
    with torch.no_grad():
        d1, v1, _, _ = m.unroll(ev[:1], st[:1], one, s)
        d2, v2, _, _ = m.unroll(ev[:1], st[:1], one, m.init_state(2))
    assert torch.equal(d1, d2) and torch.equal(v1, v2)


def test_done_resets_only_that_env():
    _, m = make_model("a_rec")
    (_, _, s, _), (ev, st, _) = _drive(m)
    done = torch.tensor([[True, False]], device=DEV)
    with torch.no_grad():
        d_reset, v_reset, _, _ = m.unroll(ev[:1], st[:1], done, s)
        d_fresh, v_fresh, _, _ = m.unroll(ev[:1], st[:1], torch.zeros_like(done), m.init_state(2))
        d_keep, _, _, _ = m.unroll(ev[:1], st[:1], torch.zeros_like(done), s)
    assert torch.allclose(d_reset[:, 0], d_fresh[:, 0]) and torch.allclose(v_reset[:, 0], v_fresh[:, 0])
    assert torch.allclose(d_reset[:, 1], d_keep[:, 1])


def test_grad_decay_keeps_forward_scales_gradient():
    _, m = make_model("a_rec")
    ev, st, done = random_batch(12, 2, p_event=0.2)
    outs, grads = [], []
    for gamma in (1.0, 0.5):
        m.zero_grad()
        s0 = {k: v.clone().requires_grad_(True) for k, v in m.init_state(2).items()}
        d, v, _, _ = m.unroll(ev, st, done, s0, grad_decay=gamma)
        (d[-1].sum() + v[-1].sum()).backward()
        outs.append(d.detach())
        grads.append(float(s0["y_vel"].grad.norm()))
    assert torch.allclose(outs[0], outs[1])
    assert grads[1] < grads[0] * 1e-2  # 0.5^11 of the gradient through 11 carried steps


def test_monitor_records_grad_and_near_threshold():
    cfg, m = make_model("a_rec")
    ev, st, done = random_batch(4, 2, p_event=0.2)
    mon = Monitor(surrogate_half_width(cfg.model.surrogate_alpha))
    d, v, _, _ = m.unroll(ev, st, done, m.init_state(2), record=mon)
    (d.sum() + v.sum()).backward()
    for k in mon.near:
        assert len(mon.near[k]) == 4 and len(mon.grad[k]) == 4
    assert any(g > 0 for g in mon.grad["mem"])
    assert surrogate_half_width(2.0) == pytest.approx(0.318, abs=1e-3)


# --- loss -----------------------------------------------------------------------------

def test_valid_mask_burn_in_and_episode_reset():
    done = torch.zeros((30, 2), dtype=torch.bool)
    done[22, 1] = True
    v = valid_mask(done, 5)
    assert not v[:5].any() and v[5:, 0].all()
    assert v[5:22, 1].all() and not v[22:28, 1].any() and v[28:, 1].all()
    assert valid_mask(done, 0).all()


def test_scale_invariant_ignores_global_offset():
    d = torch.randn(2, 8, 8)
    m = torch.rand(2, 8, 8) > 0.5
    assert torch.allclose(scale_invariant(d, m), scale_invariant(d + 3.0, m), atol=1e-5)


def test_loss_is_uniform_over_the_image():
    """Every cell counts the same, whatever the event history: the masks must not matter."""
    cfg = load_config()
    T, B, eps = 3, 1, cfg.data.inv_depth_eps
    rho = torch.full((T, B, 32, 32), 0.5)
    pred = torch.log(rho + eps)
    pred[..., 0, 0] += 1.0  # error of 1 in one cell
    rates = {"x": torch.tensor(cfg.loss.rate_target)}
    batch = dict(rho=rho, vel=torch.zeros((T, B, 3)), done=torch.zeros((T, B), dtype=torch.bool))
    _, logs = depth_velocity_loss(pred, torch.zeros((T, B, 3)), rates, batch, cfg.loss, eps, 0)
    n = 32 * 32
    si = 1.0 / n - (1.0 / n) ** 2
    assert float(logs["depth"]) == pytest.approx(1.0 / n + cfg.loss.lambda_si * si, rel=1e-4)
    batch["m_now"] = torch.ones((T, B, 32, 32), dtype=torch.bool)
    _, with_masks = depth_velocity_loss(pred, torch.zeros((T, B, 3)), rates, batch, cfg.loss, eps, 0)
    assert float(with_masks["depth"]) == float(logs["depth"])
    assert float(logs["rate_reg"]) == 0.0


def test_tiny_overfit_one_sequence():
    cfg, m = make_model("a_rec")
    T, B = 40, 2
    ev, st, done = random_batch(T, B, p_event=0.1)
    g = torch.Generator(device="cpu").manual_seed(1)
    rho = (torch.rand((T, B, 32, 32), generator=g) * 0.5 + 0.1).to(DEV)
    vel = torch.tensor([[1.5, -0.5, 0.2], [0.3, 0.8, -0.4]], device=DEV).expand(T, B, 3)
    batch = dict(rho=rho, vel=vel, done=done)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    first = None
    for it in range(150):
        d, v, _, rates = m.unroll(ev, st, done, m.init_state(B))
        loss, logs = depth_velocity_loss(d, v, rates, batch, cfg.loss, cfg.data.inv_depth_eps, 5)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.3 * first
    assert float(logs["vel"]) < 0.1


# --- evaluation -----------------------------------------------------------------------

def test_evaluate_runs_on_synthetic_heldout(tmp_path):
    from event_snn.config import to_dict
    from event_snn.evaluate import evaluate

    cfg, m = make_model("a_rec")
    T, E = 300, 4
    ev, st, done = random_batch(T, E, p_event=0.05)
    done = done.cpu()
    done[0] = True
    done[150, 1] = True
    age = torch.zeros((T, E), dtype=torch.int32)
    a = torch.zeros(E, dtype=torch.int32)
    for t in range(T):
        a = torch.where(done[t], torch.zeros_like(a), a + 1)
        age[t] = a
    m_now = (ev.cpu().sum(2) > 0).view(T, E, 32, 2, 32, 2).any(-1).any(-2)
    H = dict(events=ev.cpu(), imu=st[..., :6].cpu(), grav=st[..., 6:].cpu(), grav_true=st[..., 6:].cpu(),
             rho=(torch.rand((T, E, 32, 32)) * 0.5).half(), vel=torch.randn((T, E, 3)), m_now=m_now,
             m_recent=torch.zeros_like(m_now), done=done, yaw_rate=torch.randn((T, E)),
             mode=torch.zeros((T, E), dtype=torch.int8), age=age)
    hp, ck = tmp_path / "heldout.pt", tmp_path / "run" / "last.pt"
    ck.parent.mkdir()
    torch.save(H, hp)
    cfg.device = DEV
    torch.save({"model": m.state_dict(), "config": to_dict(cfg)}, ck)
    out = evaluate(str(ck), str(hp), device=DEV)
    assert out["depth"]["now"]["cells"] > 0 and math.isfinite(out["vel"]["rmse"])
    assert set(out["blackout"]["durations"]) == {50, 100, 200, 500}
    assert len(out["attitude_sweep"]) == 5
    assert (ck.parent / "eval" / "metrics.json").exists()


def test_recurrent_gain_is_capped():
    """Blown-up recurrent weights still act with operator norm <= rec_max_gain, and the
    gradient through the memory shrinks going back in time instead of exploding."""
    cfg, m = make_model("a_rec")
    assert cfg.model.rec_max_gain is not None
    with torch.no_grad():
        m.memory.rec.weight.mul_(100.0)
    m.memory.prepare()
    w = m.memory._w_rec.detach()
    v = torch.randn(1, w.shape[1], 8, 8, device=DEV)
    for _ in range(200):
        v = torch.nn.functional.conv_transpose2d(torch.nn.functional.conv2d(v, w, padding=1), w, padding=1)
        v = v / v.norm()
    gain = float(torch.nn.functional.conv2d(v, w, padding=1).norm())
    assert gain == pytest.approx(cfg.model.rec_max_gain, rel=0.02)

    ev, st, done = random_batch(60, 2, p_event=0.2)
    mon = Monitor(surrogate_half_width(cfg.model.surrogate_alpha))
    d, v_, _, _ = m.unroll(ev, st, done, m.init_state(2), record=mon)
    (d[-1].sum() + v_[-1].sum()).backward()
    g = mon.grad["mem"]
    assert g[0] < g[40]
