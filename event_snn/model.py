"""Event SNN: spiking conv encoder -> conv-recurrent adaptive-LIF memory (8x8) -> [spiking bottleneck]
-> depth (+ inverse time-to-contact) and velocity heads.

With model.bottleneck > 0 the heads read only the bottleneck's spikes, so whatever they decode
is available to a policy that is fed those same spikes; the heads are training-only.

One call to step() is one 10 ms step. All state is explicit (a dict of tensors) so BPTT
windows, per-env resets, gradient time decay and variant B (reset every step) are handled
by unroll() rather than hidden inside the neurons.
"""
import math

import snntorch as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from snntorch import surrogate

DT = 0.01
STATE_INPUTS = {"acc": slice(0, 3), "gyro": slice(3, 6), "grav": slice(6, 9)}
SPIKING_LAYERS = ("c1", "c2", "c3", "mem", "d1")  # + "bn" when model.bottleneck > 0: see EventSNN.spiking_layers


def _leaky(channels, beta_init, spike_grad):
    return snn.Leaky(beta=torch.full((channels, 1, 1), beta_init), learn_beta=True, threshold=1.0,
                     spike_grad=spike_grad, reset_mechanism="subtract")


def _readout(shape, beta_init):
    """Non-spiking leaky integrator U = beta*U + I: a Leaky whose membrane is read, never reset."""
    return snn.Leaky(beta=torch.full(shape, beta_init), learn_beta=True, reset_mechanism="none")


class PopEncoder(nn.Module):
    """Gaussian receptive fields (PopSAN-style) driving IF neurons: numbers -> spikes, one step.

    Bounds are measured on the student's own data (set_bounds) and stored as buffers.
    """

    def __init__(self, n_in, pop_dim, threshold):
        super().__init__()
        self.n_in, self.pop_dim, self.threshold = n_in, pop_dim, threshold
        self.register_buffer("means", torch.zeros(n_in, pop_dim))
        self.register_buffer("stds", torch.ones(n_in, pop_dim))
        self.register_buffer("lo", torch.full((n_in,), -1.0))
        self.register_buffer("hi", torch.full((n_in,), 1.0))
        self.set_bounds(self.lo, self.hi)

    @torch.no_grad()
    def set_bounds(self, lo, hi):
        lo, hi = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
        self.lo.copy_(lo)
        self.hi.copy_(hi)
        frac = torch.linspace(0.0, 1.0, self.pop_dim)
        self.means.copy_(lo[:, None] + frac[None] * (hi - lo)[:, None])
        self.stds.copy_(((hi - lo) / (self.pop_dim - 1) * 0.75).clamp(min=1e-6)[:, None].expand(-1, self.pop_dim))

    @property
    def n_out(self):
        return self.n_in * self.pop_dim

    def forward(self, x, mem):
        x = torch.max(torch.min(x, self.hi), self.lo)
        a = torch.exp(-0.5 * ((x[..., None] - self.means) / self.stds) ** 2).flatten(1)
        mem = mem + a
        spk = (mem >= self.threshold).to(a.dtype)
        mem = (mem - spk * self.threshold).clamp(max=self.threshold)
        return spk, mem


class ALIFMemory(nn.Module):
    """Adaptive LIF on the 8x8 grid: per-region C3 input + broadcast state spikes + 3x3 recurrence.

    Learnable membrane time constant per neuron, adaptation time constant per channel.

    The recurrent conv's operator norm on the grid is capped at rec_max_gain: above ~1 the
    gradient grows backward in time (x1.3 per step measured at gain 4.7) and BPTT explodes.
    """

    def __init__(self, c_in, channels, size, n_state, recurrent, tau_mem_init, tau_adapt_init,
                 adapt_strength, spike_grad, rec_init_scale=1.0, rec_max_gain=None):
        super().__init__()
        self.ff = nn.Conv2d(c_in, channels, 1)
        self.state_fc = nn.Linear(n_state, channels, bias=False) if n_state > 0 else None
        self.rec = nn.Conv2d(channels, channels, 3, padding=1, bias=False) if recurrent else None
        self.rec_max_gain = rec_max_gain
        self._w_rec = None
        if self.rec is not None:
            with torch.no_grad():
                self.rec.weight.mul_(rec_init_scale)
            # power-iteration vector; not saved, so it re-converges after loading
            self.register_buffer("_rec_v", F.normalize(torch.randn(1, channels, size, size).flatten(), dim=0)
                                 .view(1, channels, size, size), persistent=False)
            self._rec_v_warm = False
        self.log_tau_mem = nn.Parameter(torch.full((channels, size, size), math.log(tau_mem_init)))
        self.log_tau_adapt = nn.Parameter(torch.full((channels, 1, 1), math.log(tau_adapt_init)))
        self.adapt_strength = adapt_strength
        self.spike_grad = spike_grad

    def alpha(self):
        return torch.exp(-DT / self.log_tau_mem.exp())

    def rec_gain(self, iters=None):
        """Operator norm of the recurrent conv on the grid (differentiable in the weight)."""
        w = self.rec.weight
        with torch.no_grad():
            v = self._rec_v
            for _ in range(iters or (3 if self._rec_v_warm else 50)):
                v = F.conv_transpose2d(F.conv2d(v, w, padding=1), w, padding=1)
                v = v / v.norm().clamp(min=1e-12)
            self._rec_v.copy_(v)
            self._rec_v_warm = True
        return F.conv2d(v, w, padding=1).norm()

    def prepare(self):
        """Effective recurrent weight for this unroll: the weight scaled down to rec_max_gain."""
        if self.rec is None:
            return
        w = self.rec.weight
        if self.rec_max_gain is not None:
            w = w * (self.rec_max_gain / self.rec_gain()).clamp(max=1.0)
        self._w_rec = w

    def forward(self, x, s_state, u, b, s_prev):
        cur = self.ff(x)
        if self.state_fc is not None:
            cur = cur + self.state_fc(s_state)[:, :, None, None]
        if self.rec is not None:
            if self._w_rec is None:
                self.prepare()
            cur = cur + F.conv2d(s_prev, self._w_rec, padding=1)
        rho = torch.exp(-DT / self.log_tau_adapt.exp())
        theta_prev = 1.0 + self.adapt_strength * b
        u = self.alpha() * u + cur - s_prev.detach() * theta_prev  # soft reset, detached as in snnTorch
        b = rho * b + (1.0 - rho) * s_prev
        theta = 1.0 + self.adapt_strength * b
        s = self.spike_grad(u - theta)
        return s, u, b, theta


class MemoryBottleneck(nn.Module):
    """The bottleneck is the memory: conv-3 spikes (flattened) concatenated with the inertial
    spikes drive `n` recurrent adaptive-LIF neurons. Their spikes are the policy interface.

    Same neuron as ALIFMemory, fully connected: membrane and adaptation time constants are
    learnable per neuron, and the recurrent matrix's spectral norm is capped at rec_max_gain.
    """

    def __init__(self, n_in, n, recurrent, tau_mem_init, tau_adapt_init, adapt_strength, spike_grad,
                 rec_init_scale=1.0, rec_max_gain=None):
        super().__init__()
        self.ff = nn.Linear(n_in, n)
        self.rec = nn.Linear(n, n, bias=False) if recurrent else None
        if self.rec is not None:
            with torch.no_grad():
                self.rec.weight.mul_(rec_init_scale)
        self.rec_max_gain = rec_max_gain
        self._w_rec = None
        self.log_tau_mem = nn.Parameter(torch.full((n,), math.log(tau_mem_init)))
        self.log_tau_adapt = nn.Parameter(torch.full((n,), math.log(tau_adapt_init)))
        self.adapt_strength = adapt_strength
        self.spike_grad = spike_grad

    def rec_gain(self, iters=None):
        return torch.linalg.matrix_norm(self.rec.weight, ord=2)

    def prepare(self):
        if self.rec is None:
            return
        w = self.rec.weight
        if self.rec_max_gain is not None:
            w = w * (self.rec_max_gain / self.rec_gain()).clamp(max=1.0)
        self._w_rec = w

    def forward(self, x, u, b, s_prev):
        cur = self.ff(x)
        if self.rec is not None:
            if self._w_rec is None:
                self.prepare()
            cur = cur + F.linear(s_prev, self._w_rec)
        alpha = torch.exp(-DT / self.log_tau_mem.exp())
        rho = torch.exp(-DT / self.log_tau_adapt.exp())
        theta_prev = 1.0 + self.adapt_strength * b
        u = alpha * u + cur - s_prev.detach() * theta_prev
        b = rho * b + (1.0 - rho) * s_prev
        theta = 1.0 + self.adapt_strength * b
        s = self.spike_grad(u - theta)
        return s, u, b, theta


class EventSNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.stateful = bool(cfg.stateful)
        self.state_inputs = list(cfg.state_inputs)
        for k in self.state_inputs:
            if k not in STATE_INPUTS:
                raise ValueError(f"unknown state input {k!r}; known: {sorted(STATE_INPUTS)}")
        self.register_buffer("state_index", torch.cat(
            [torch.arange(9)[STATE_INPUTS[k]] for k in self.state_inputs]) if self.state_inputs
            else torch.zeros(0, dtype=torch.long))

        sg = surrogate.atan(alpha=float(cfg.surrogate_alpha))
        c1, c2, c3 = cfg.conv_channels
        m = cfg.mem_channels
        self.conv1 = nn.Conv2d(2, c1, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.lif1, self.lif2, self.lif3 = (_leaky(c, cfg.beta_init, sg) for c in (c1, c2, c3))
        self.encoder = PopEncoder(len(self.state_index), cfg.pop_dim, cfg.encoder_threshold) \
            if self.state_inputs else None
        n_pop = self.encoder.n_out if self.encoder is not None else 0
        self.n_bottleneck = int(getattr(cfg, "bottleneck", 0) or 0)
        self.ttc_head = bool(getattr(cfg, "ttc_head", False))
        self.memory_kind = getattr(cfg, "memory", "grid")       # grid | bottleneck
        self.decoder_kind = getattr(cfg, "decoder", "spiking")  # spiking | ann
        if self.memory_kind not in ("grid", "bottleneck") or self.decoder_kind not in ("spiking", "ann"):
            raise ValueError(f"memory={self.memory_kind!r} decoder={self.decoder_kind!r}")
        nb = self.n_bottleneck
        rec_args = (bool(cfg.recurrent), cfg.tau_mem_init_s, cfg.tau_adapt_init_s, cfg.adapt_strength, sg,
                    getattr(cfg, "rec_init_scale", 1.0), getattr(cfg, "rec_max_gain", None))
        if self.memory_kind == "bottleneck":
            # conv3 -> flatten -> concat inertial spikes -> recurrent adaptive-LIF bottleneck -> decoder
            if not nb:
                raise ValueError("memory='bottleneck' needs model.bottleneck > 0")
            self.memory = MemoryBottleneck(c3 * 64 + n_pop, nb, *rec_args)
        else:
            self.memory = ALIFMemory(c3, m, 8, n_pop, *rec_args)
            if nb:
                self.bn_fc = nn.Linear(m * 64, nb)
                self.lif_bn = snn.Leaky(beta=torch.full((nb,), float(cfg.beta_init)), learn_beta=True,
                                        threshold=1.0, spike_grad=sg, reset_mechanism="subtract")
        if nb:
            self.bn_expand = nn.Linear(nb, m * 64)
        elif self.decoder_kind == "ann":
            raise ValueError("decoder='ann' needs model.bottleneck > 0")
        self.spiking_layers = (("c1", "c2", "c3") + (("mem",) if self.memory_kind == "grid" else ())
                               + (("bn",) if nb else ()) + (("d1",) if self.decoder_kind == "spiking" else ()))
        # decoder output channels: [inverse depth] [invTTC or speed/range map], each only if trained
        self.depth_head = bool(getattr(cfg, "depth_head", True))
        n_maps = int(self.depth_head) + int(self.ttc_head)
        if n_maps == 0:
            raise ValueError("the decoder needs model.depth_head or model.ttc_head")
        self.n_maps, self.ttc_ch = n_maps, n_maps - 1
        self.deconv1 = nn.ConvTranspose2d(m, 32, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, n_maps, 4, stride=2, padding=1)
        if self.decoder_kind == "spiking":
            self.lif_d1 = _leaky(32, cfg.beta_init, sg)
            self.depth_out = _readout((n_maps, 1, 1), cfg.readout_beta_init)
        else:
            # non-spiking decoder on a leaky trace of the interface spikes; no state of its own
            self.dec_trace = _readout((nb,), cfg.readout_beta_init)
        self.vel_head = bool(getattr(cfg, "vel_head", True))
        if self.vel_head:
            self.vel_fc = nn.Linear(nb or m, 3)
            self.vel_out = _readout((3,), cfg.readout_beta_init)

    @torch.no_grad()
    def set_input_bounds(self, lo9, hi9):
        """Bounds for all 9 state numbers [acc, gyro, grav]; the encoder keeps its own subset."""
        if self.encoder is not None:
            idx = self.state_index.cpu()
            self.encoder.set_bounds(torch.as_tensor(lo9)[idx], torch.as_tensor(hi9)[idx])

    # --- state ------------------------------------------------------------------------

    def init_state(self, batch, device=None):
        device = device or next(self.parameters()).device
        c1, c2, c3 = self.cfg.conv_channels
        m = self.cfg.mem_channels
        z = lambda *s: torch.zeros((batch,) + s, device=device)
        ms = (self.n_bottleneck,) if self.memory_kind == "bottleneck" else (m, 8, 8)
        st = dict(c1=z(c1, 32, 32), c2=z(c2, 16, 16), c3=z(c3, 8, 8),
                  mem_u=z(*ms), mem_b=z(*ms), mem_s=z(*ms),
                  y_depth=z(self.n_maps, 32, 32), y_vel=z(3))
        if self.decoder_kind == "spiking":
            st["d1"] = z(32, 16, 16)
        else:
            st["dec_tr"] = z(self.n_bottleneck)
        if self.n_bottleneck and self.memory_kind == "grid":
            st["bn"] = z(self.n_bottleneck)
        if self.encoder is not None:
            st["enc"] = z(self.encoder.n_out)
        return st

    @staticmethod
    def reset_where(st, done):
        keep = (~done).to(torch.float32)
        return {k: v * keep.view((-1,) + (1,) * (v.dim() - 1)) for k, v in st.items()}

    @staticmethod
    def detach_state(st):
        return {k: v.detach() for k, v in st.items()}

    @staticmethod
    def decay_grad(st, gamma):
        """Forward unchanged; gradient through the carried state scaled by gamma per step."""
        return {k: gamma * v + (1.0 - gamma) * v.detach() for k, v in st.items()}

    # --- one 10 ms step ----------------------------------------------------------------

    def step(self, events, state_in, st, record=None):
        """events (B, 2, 64, 64) counts; state_in (B, 9) raw [acc, gyro, grav]. Returns
        (log(rho+eps) prediction (B, 32, 32), velocity (B, 3), new state, spikes per layer).
        With ttc_head, the log map is new_state["y_depth"][:, self.ttc_ch]; without depth_head the
        depth prediction returned is zeros. The policy interface is spikes[self.interface]."""
        x = events.to(torch.float32)
        s1, m1 = self.lif1(self.conv1(x), st["c1"])
        s2, m2 = self.lif2(self.conv2(s1), st["c2"])
        s3, m3 = self.lif3(self.conv3(s2), st["c3"])

        new = {"c1": m1, "c2": m2, "c3": m3}
        if self.encoder is not None:
            s_pop, new["enc"] = self.encoder(state_in[:, self.state_index], st["enc"])
        else:
            s_pop = None
        spikes = {"c1": s1, "c2": s2, "c3": s3}
        mems = {"c1": (m1, 1.0), "c2": (m2, 1.0), "c3": (m3, 1.0)}
        grid = (-1, self.cfg.mem_channels, 8, 8)
        if self.memory_kind == "bottleneck":
            x_in = s3.flatten(1) if s_pop is None else torch.cat([s3.flatten(1), s_pop], dim=1)
            sb, u, b, theta = self.memory(x_in, st["mem_u"], st["mem_b"], st["mem_s"])
            new.update(mem_u=u, mem_b=b, mem_s=sb)
            spikes["bn"], mems["bn"] = sb, (u, theta)
            vel_in = sb
        else:
            sm, u, b, theta = self.memory(s3, s_pop, st["mem_u"], st["mem_b"], st["mem_s"])
            new.update(mem_u=u, mem_b=b, mem_s=sm)
            spikes["mem"], mems["mem"] = sm, (u, theta)
            if self.n_bottleneck:
                sb, mb = self.lif_bn(self.bn_fc(sm.flatten(1)), st["bn"])
                new["bn"] = mb
                spikes["bn"], mems["bn"] = sb, (mb, 1.0)
                vel_in = sb
            else:
                sb, vel_in = None, sm.mean(dim=(2, 3))

        if self.decoder_kind == "spiking":
            dec_in = self.bn_expand(sb).view(grid) if sb is not None else sm
            sd, md = self.lif_d1(self.deconv1(dec_in), st["d1"])
            new["d1"] = md
            _, new["y_depth"] = self.depth_out(self.deconv2(sd), st["y_depth"])
            spikes["d1"], mems["d1"] = sd, (md, 1.0)
        else:
            _, new["dec_tr"] = self.dec_trace(sb, st["dec_tr"])
            h = F.relu(self.bn_expand(new["dec_tr"])).view(grid)
            new["y_depth"] = self.deconv2(F.relu(self.deconv1(h)))
        if self.vel_head:
            _, new["y_vel"] = self.vel_out(self.vel_fc(vel_in), st["y_vel"])
        else:
            new["y_vel"] = st["y_vel"]  # stays zero: no velocity output

        if record is not None:
            record.step(mems, spikes)
        depth = new["y_depth"][:, 0]
        return (depth if self.depth_head else torch.zeros_like(depth)), new["y_vel"], new, spikes

    @property
    def interface(self):
        """Layer whose spikes a downstream policy reads."""
        return "bn" if self.n_bottleneck else "mem"

    def unroll(self, events, state_in, done, st, grad_decay=1.0, record=None, extras=None):
        """Run T steps. events (T, B, 2, 64, 64), state_in (T, B, 9), done (T, B) bool.

        Returns depth (T, B, 32, 32), vel (T, B, 3), final state, mean spike rate per layer
        (scalar tensors, differentiable). With ttc_head and a dict passed as `extras`,
        extras["ttc"] receives log(invTTC+eps) (T, B, 32, 32)."""
        T, B = done.shape
        ttc = []
        self.memory.prepare()
        depth, vel = [], []
        rate_sum = {k: 0.0 for k in self.spiking_layers}
        for t in range(T):
            if not self.stateful:
                st = self.init_state(B, events.device)
            else:
                if done[t].any():
                    st = self.reset_where(st, done[t])
                if grad_decay < 1.0 and t > 0:
                    st = self.decay_grad(st, grad_decay)
            d, v, st, spikes = self.step(events[t], state_in[t], st, record)
            depth.append(d)
            vel.append(v)
            if self.ttc_head and extras is not None:
                ttc.append(st["y_depth"][:, self.ttc_ch])
            for k, s in spikes.items():
                rate_sum[k] = rate_sum[k] + s.mean()
        rates = {k: v / T for k, v in rate_sum.items()}
        if ttc:
            extras["ttc"] = torch.stack(ttc)
        return torch.stack(depth), torch.stack(vel), st, rates


class Monitor:
    """Per layer and per time step: gradient norm of the membrane, and share of neurons inside
    the surrogate's half-width (|U - theta| < half_width)."""

    def __init__(self, half_width):
        self.half_width = half_width
        self.near, self.grad = {}, {}

    def step(self, mems, spikes=None):
        for k, (u, theta) in mems.items():
            self.near.setdefault(k, [])
            self.grad.setdefault(k, [])
            self.near[k].append(float(((u - theta).abs() < self.half_width).float().mean()))
            if u.requires_grad:
                slot = len(self.grad[k])
                self.grad[k].append(0.0)
                u.register_hook(self._hook(k, slot))

    def _hook(self, k, slot):
        def h(g):
            self.grad[k][slot] = float(g.norm())
        return h


def surrogate_half_width(alpha):
    """|x| where snnTorch's atan surrogate falls to half its peak: 2 / (pi * alpha)."""
    return 2.0 / (math.pi * alpha)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path, device=None):
    from .config import to_ns
    ck = torch.load(path, map_location="cpu")
    cfg = to_ns(ck["config"])
    model = EventSNN(cfg.model)
    model.load_state_dict(ck["model"])
    return model.to(device or cfg.device).eval(), cfg, ck
