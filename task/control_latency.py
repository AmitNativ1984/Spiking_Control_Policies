"""Image-capture -> control-output dead time, modelled at physics-substep resolution.

The real loop measured in Gazebo SITL (2026-10-05): 45 ms mean / 10 ms std from depth
capture to the attitude command leaving the policy node, with the policy itself running at
~33 Hz (+/- ~10 Hz). Latency is LONGER than one policy period, so the real pipeline is
pipelined: while action k is in flight, the controller is still flying action k-1 (or
k-2). The simulator, by contrast, renders at the end of env step k and applies a_k from the
very first substep of step k+1 -- zero dead time.

This class restores the dead time without touching the observation path. Each pushed action
gets an arrival tick = capture tick + round(L / sim_dt), L ~ N(mean, std) clamped to
[min, max], and at every physics substep the controller is handed the newest action that
has arrived. Delaying the ACTION by L is the same closed loop as delaying the whole
observation by L for a memoryless policy, and it keeps 10 ms resolution where an
observation FIFO would be quantised to whole env steps (20-40 ms).

ASSUMPTION worth knowing: this makes the STATE channels exactly as stale as the image. If
the deployed node reads odometry fresh at inference time rather than synchronised to the
frame, the state is really ~L - inference-time fresher than modelled here, i.e. this is the
pessimistic side.

Two properties are deliberate:
  - No overtaking. A single-threaded capture->VAE->policy pipeline cannot finish frame k+1
    before frame k, so arrival_k = max(capture_k + L_k, arrival_{k-1}). Without this, the
    jitter would occasionally apply a_{k+1} and then revert to a_k.
  - Round-to-nearest, not ceil. The controller acts at substep starts, so an arrival between
    two substeps has to snap to one of them; ceil would add +dt/2 (5 ms) of bias to the mean.

Pure torch, no Isaac Gym, so it is unit-testable on CPU (tests/test_control_latency.py).
"""
import torch

# Arrival tick for an empty slot. Large enough to never be reached, small enough that
# clock + ticks cannot overflow int64.
_NEVER = 2**62


class ControlLatency:
    def __init__(
        self,
        num_envs,
        action_dim,
        sim_dt,
        mean_s,
        std_s,
        min_s,
        max_s,
        buffer_len,
        device,
        generator=None,
    ):
        """
        buffer_len must exceed the most actions that can be in flight at once. At the
        measured rate that is ~2, and max_s / sim_dt (+ any zero-substep env steps, which
        push without advancing the clock) bounds it. An overwritten slot that had not yet
        arrived would silently drop that action, so size this generously -- it costs
        num_envs x buffer_len x action_dim floats.
        """
        if max_s < min_s:
            raise ValueError(f"control latency max_s={max_s} < min_s={min_s}")
        self.num_envs = num_envs
        self.sim_dt = float(sim_dt)
        self.mean_s = float(mean_s)
        self.std_s = float(std_s)
        self.min_s = float(min_s)
        self.max_s = float(max_s)
        self.buffer_len = int(buffer_len)
        self.device = device
        self._gen = generator

        n, b = num_envs, self.buffer_len
        self._actions = torch.zeros((n, b, action_dim), device=device)
        self._arrival = torch.full((n, b), _NEVER, dtype=torch.long, device=device)
        # Push order, so "newest arrived" is well-defined even when two actions land on the
        # same tick (a zero-substep env step pushes twice without the clock moving).
        self._seq = torch.full((n, b), -1, dtype=torch.long, device=device)
        self._last_arrival = torch.zeros(n, dtype=torch.long, device=device)
        # What the controller is flying right now. Zero is the neutral command (hover
        # thrust, level attitude, no yaw rate) -- see reset_idx in the task.
        self._applied = torch.zeros((n, action_dim), device=device)
        self._env_index = torch.arange(n, device=device)

        # Physics substeps since construction. Shared by all envs: upstream draws ONE
        # substep count per env step for the whole batch, so every env's clock agrees.
        self._clock = 0
        self._head = 0
        self._pushes = 0

        # Latency actually realised by the last push (after clamp, rounding and the
        # no-overtaking rule), in seconds. For logging.
        self.last_latency_s = torch.zeros(n, device=device)

    def push(self, actions):
        """Queue the actions computed from the frame captured at the current tick."""
        noise = torch.randn(self.num_envs, device=self.device, generator=self._gen)
        lat = (self.mean_s + self.std_s * noise).clamp(self.min_s, self.max_s)
        ticks = torch.round(lat / self.sim_dt).long()
        arrival = torch.maximum(self._clock + ticks, self._last_arrival)

        slot = self._head
        self._actions[:, slot] = actions
        self._arrival[:, slot] = arrival
        self._seq[:, slot] = self._pushes
        self._last_arrival = arrival
        self.last_latency_s = (arrival - self._clock).float() * self.sim_dt

        self._head = (self._head + 1) % self.buffer_len
        self._pushes += 1

    def on_substep(self):
        """Advance one physics substep; return the command the controller holds during it."""
        arrived = self._arrival <= self._clock
        newest_seq, newest_slot = torch.where(arrived, self._seq, -1).max(dim=1)
        has_new = (newest_seq >= 0).unsqueeze(1)
        newest = self._actions[self._env_index, newest_slot]
        self._applied = torch.where(has_new, newest, self._applied)
        self._clock += 1
        return self._applied

    def reset(self, env_ids):
        """Forget in-flight actions of envs starting a new episode, and hold neutral."""
        self._arrival[env_ids] = _NEVER
        self._seq[env_ids] = -1
        self._applied[env_ids] = 0.0
        self._last_arrival[env_ids] = self._clock
        self.last_latency_s[env_ids] = 0.0
