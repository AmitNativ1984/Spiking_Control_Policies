"""Non-spiking reference: a plain CNN on a stack of the last K event frames -> target map.

It answers one question: can the target map be read from these events at all? By default it sees
only the events; with imu=True also the last K steps of [acc, gyro, grav] (the gyro tells image
motion from rotation apart from motion toward obstacles). It has no spiking neurons and no state other than the frame history, and
an encoder -> 8x8 wide-context middle -> decoder with no skip connections, so everything reaches the
map through the same 8x8 grid as in the SNN; it is not limited by receptive field or by rate coding.
If it cannot draw obstacles either, the problem is the data, target or loss, not the SNN.

Drop-in for EventSNN in event_snn.overfit (`--baseline`): same unroll() signature and outputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(c_in, c_out, stride=1, dilation=1):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, padding=dilation, dilation=dilation),
                         nn.GroupNorm(8, c_out), nn.ReLU(inplace=True))


class StackedEventCNN(nn.Module):
    """events (2, 64, 64) per 10 ms step; input = the last K steps as 2K channels (log1p counts),
    frames from before the drone's last reset zeroed. Output log map (32, 32) per step."""

    spiking_layers = ()
    interface = None

    def __init__(self, k=30, width=64, chunk=64, imu=False):
        super().__init__()
        self.k, self.chunk, self.imu = k, chunk, imu
        w = width
        self.e1 = nn.Sequential(nn.Conv2d(2 * k, w, 5, 2, 2), nn.GroupNorm(8, w), nn.ReLU(inplace=True),
                                _block(w, w))                                    # 32x32
        self.e2 = nn.Sequential(_block(w, 2 * w, 2), _block(2 * w, 2 * w))         # 16x16
        self.e3 = nn.Sequential(_block(2 * w, 4 * w, 2), _block(4 * w, 4 * w))     # 8x8
        # wide context at 8x8: dilated convs (receptive field = whole image) + a global feature
        self.mid = nn.Sequential(_block(4 * w, 4 * w, dilation=2), _block(4 * w, 4 * w, dilation=4))
        self.glob = nn.Linear(4 * w, 4 * w)
        if imu:
            # the last K steps of [acc, gyro, grav] (normalised to about -1..1 by the data bounds),
            # added to every cell of the 8x8 middle: rotation over the window and attitude
            self.imu_fc = nn.Sequential(nn.Linear(9 * k, 4 * w), nn.ReLU(inplace=True), nn.Linear(4 * w, 4 * w))
            self.register_buffer("imu_mid", torch.zeros(9))
            self.register_buffer("imu_half", torch.ones(9))
        # decoder reads only the 8x8 middle (no skip connections), as the SNN's decoder reads only its 8x8 memory
        self.d2 = nn.Sequential(_block(4 * w, 2 * w), _block(2 * w, 2 * w))          # 16x16
        self.d1 = nn.Sequential(_block(2 * w, w), _block(w, w))                      # 32x32
        self.out = nn.Conv2d(w, 1, 1)

    @torch.no_grad()
    def set_input_bounds(self, lo, hi):
        if self.imu:
            lo, hi = torch.as_tensor(lo, dtype=torch.float32), torch.as_tensor(hi, dtype=torch.float32)
            self.imu_mid.copy_((lo + hi) / 2)
            self.imu_half.copy_(((hi - lo) / 2).clamp(min=1e-6))

    def init_state(self, batch, device=None):
        device = device or next(self.parameters()).device
        return {"hist": torch.zeros((self.k - 1, batch, 2, 64, 64), device=device),
                "hist_imu": torch.zeros((self.k - 1, batch, 9), device=device),
                "hist_ok": torch.zeros((self.k - 1, batch), dtype=torch.bool, device=device)}

    @staticmethod
    def detach_state(st):
        return {k: v.detach() for k, v in st.items()}

    def net(self, x, imu=None):
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.mid(self.e3(h2))
        h3 = h3 + self.glob(h3.mean((2, 3)))[:, :, None, None]
        if imu is not None:
            h3 = h3 + self.imu_fc(imu)[:, :, None, None]
        u = F.interpolate(h3, scale_factor=2, mode="nearest")
        u = self.d2(u)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.d1(u)
        return self.out(u)[:, 0]

    def unroll(self, events, state_in, done, st, grad_decay=1.0, record=None, extras=None):
        """events (T, B, 2, 64, 64), done (T, B). Returns depth zeros (T, B, 32, 32), vel zeros (T, B, 3),
        final state, {} rates; extras["ttc"] = log map (T, B, 32, 32)."""
        T, B = done.shape
        K = self.k
        full = torch.cat([st["hist"], torch.log1p(events.float())], 0)            # (K-1+T, B, 2, 64, 64)
        if self.imu:
            si = ((state_in.float() - self.imu_mid) / self.imu_half).clamp(-3.0, 3.0)
            full_imu = torch.cat([st["hist_imu"], si], 0)                         # (K-1+T, B, 9)
        # episode id per frame: history = -1 if invalid else 0, current steps 0 + resets so far
        ep_cur = torch.cumsum(done.long(), 0)                                    # (T, B)
        ep_hist = torch.where(st["hist_ok"], torch.zeros_like(st["hist_ok"], dtype=torch.long),
                              torch.full_like(st["hist_ok"], -1, dtype=torch.long))
        ep = torch.cat([ep_hist, ep_cur], 0)                                     # (K-1+T, B)
        outs = []
        for t0 in range(0, T, self.chunk):
            t1 = min(t0 + self.chunk, T)
            idx = torch.arange(t0, t1, device=done.device)[:, None] + torch.arange(K, device=done.device)[None]
            x = full[idx]                                                        # (n, K, B, 2, 64, 64)
            keep = (ep[idx] == ep_cur[t0:t1, None]).to(x.dtype)                  # (n, K, B)
            x = x * keep[..., None, None, None]
            n = t1 - t0
            x = x.permute(0, 2, 1, 3, 4, 5).reshape(n * B, 2 * K, 64, 64)       # oldest .. newest
            imu = None
            if self.imu:
                imu = (full_imu[idx] * keep[..., None]).permute(0, 2, 1, 3).reshape(n * B, 9 * K)
            outs.append(self.net(x, imu).view(n, B, 32, 32))
        ttc = torch.cat(outs, 0)
        last_ep = ep_cur[-1]
        new = {"hist": full[-(K - 1):].detach(), "hist_ok": ep[-(K - 1):] == last_ep[None],
               "hist_imu": full_imu[-(K - 1):].detach() if self.imu else st["hist_imu"]}
        if extras is not None:
            extras["ttc"] = ttc
        return torch.zeros_like(ttc), torch.zeros((T, B, 3), device=ttc.device), new, {}
