import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

# Default clamp window per observation TYPE, keyed by the type names a task publishes in
# its observation_layout. The task owns what each dimension MEANS; the encoder owns how
# wide a window it spans, which is why this table lives here and not in the task config.
#
# Values are in the rl_games-NORMALIZED space (z-scores, hard-clamped to [-5, 5] by
# RunningMeanStd when normalize_input=True), NOT raw units — so +/-3 sigma covers ~99.7%
# of a normal dimension and the encoder clamps the rest.
#
# Uniform today. Per-type entries exist so a kind of input whose distribution is genuinely
# wider can be widened on its own: VAE latents are the usual first candidate, since their
# spread depends on the trained VAE rather than on physical units. Measured per-dimension
# bounds from tools/collect_obs_stats.py override this table wholesale.
DEFAULT_TYPE_BOUNDS = {
    "direction_to_target": (-3.0, 3.0),
    "distance":            (-3.0, 3.0),
    "position_error":      (-3.0, 3.0),
    "linvel":              (-3.0, 3.0),
    "angvel":              (-3.0, 3.0),
    "gravity":             (-3.0, 3.0),
    "prev_action":         (-3.0, 3.0),
    "vae_latent":          (-3.0, 3.0),
}


def bounds_from_layout(observation_layout, obs_dim, type_bounds=None):
    """Expand a task's observation_layout into the flat per-index list __init__ wants.

    Args:
        observation_layout: list of (slice, type_name), as published by a task config.
        obs_dim: the task's observation_space_dim; every index must be covered.
        type_bounds: per-type overrides; defaults to DEFAULT_TYPE_BOUNDS.

    Returns:
        list of (lo, hi), length obs_dim.

    Raises:
        KeyError: the layout names a type with no entry — someone added an observation
            without deciding how wide its encoder window should be. Silently falling back
            to a default would mis-scale that dimension's receptive fields.
        ValueError: the layout leaves indices uncovered.
    """
    type_bounds = DEFAULT_TYPE_BOUNDS if type_bounds is None else type_bounds

    bounds = [None] * obs_dim
    for obs_slice, obs_type in observation_layout:
        if obs_type not in type_bounds:
            raise KeyError(
                f"observation_layout names type {obs_type!r}, which has no entry in "
                f"type_bounds (known: {sorted(type_bounds)}). Add one to "
                f"DEFAULT_TYPE_BOUNDS in {__name__}."
            )
        for idx in range(obs_slice.start, obs_slice.stop):
            bounds[idx] = type_bounds[obs_type]

    missing = [i for i, b in enumerate(bounds) if b is None]
    if missing:
        raise ValueError(
            f"observation_layout leaves indices {missing} uncovered; every index in "
            f"[0, {obs_dim}) must belong to exactly one layout entry."
        )
    return bounds


class PopulationSpikeEncoder(nn.Module):
    """Population encoding module for PopSAN.

    Each dimension of the input observation is encoded into the activity of a population
    of neurons. Each neuron is a Gaussian receptive field N~(mu, sigma): the means are
    initialized evenly spaced across that dimension's (min, max) bound and are learnable,
    and the stds are initialized to overlap neighbouring fields, also learnable.

    Observations arriving here are already normalized by rl_games; `obs_bounds` are the
    bounds in that NORMALIZED space (see rl_training.rl_games.tools.collect_obs_stats).
    """

    def __init__(self, obs_dim: int, obs_bounds: list, num_steps: int, encoder_config: dict) -> None:
        """
        Args:
            obs_dim: Dimension of the input observation space.
            obs_bounds: One (min, max) tuple per observation dimension, used to initialize
                the Gaussian means/stds and to clamp the incoming observation.
            num_steps: Number of time steps for the spike simulation. Shared with the outer SNN.
            encoder_config: Encoder configuration (pop_dim, threshold).
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.pop_dim = encoder_config["pop_dim"]
        self.encoder_neuron_num = self.obs_dim * self.pop_dim
        self.num_steps = num_steps
        self.threshold = encoder_config["threshold"]

        # Registered as a buffer so it follows the model across devices.
        self.register_buffer("obs_bounds", torch.tensor(obs_bounds, dtype=torch.float))
        obs_min = self.obs_bounds[:, 0].unsqueeze(1)  # [obs_dim, 1]
        obs_max = self.obs_bounds[:, 1].unsqueeze(1)  # [obs_dim, 1]
        obs_range = obs_max - obs_min

        spacing = torch.linspace(0, 1, self.pop_dim).unsqueeze(0)  # [1, pop_dim]
        self.means = nn.Parameter((obs_min + spacing * obs_range).unsqueeze(0))  # [1, obs_dim, pop_dim]

        # Stds initialized to overlap neighbouring receptive fields, so the whole input
        # range is covered and at least one neuron spikes for any observation.
        means_spacing = torch.abs(self.means[:, :, 1] - self.means[:, :, 0])  # [1, obs_dim]
        init_stds = means_spacing * 0.75
        init_log_stds = torch.log(
            init_stds.unsqueeze(2).expand(-1, -1, self.pop_dim).contiguous()
        )  # [1, obs_dim, pop_dim]
        self.log_stds = nn.Parameter(init_log_stds)

        self.if1 = snn.Leaky(
            beta=1.0,  # no leak => IF neuron
            threshold=self.threshold,
            # Passthrough gradient for the non-differentiable spiking function.
            spike_grad=surrogate.straight_through_estimator(),
            reset_mechanism="subtract",
        )

        # Debug-only: when record=True, forward() appends per-step traces to _trace.
        # Set externally (e.g. by the runner during --play --plot-encoding). Inert otherwise.
        self.record = False
        self._trace = []

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the input observation into population spike activity.

        The stimulation strength of each neuron is a Gaussian function of the distance
        between the observation and that neuron's mean. That constant current drives an
        IF neuron over `num_steps` steps to produce the spike train.

        Args:
            obs: Input observation tensor of shape [batch_size, obs_dim].

        Returns:
            Spike tensor of shape [batch_size, obs_dim * pop_dim, num_steps].
        """
        batch_size = obs.shape[0]

        lo = self.obs_bounds[:, 0]
        hi = self.obs_bounds[:, 1]
        obs = torch.clamp(obs, min=lo, max=hi)  # [batch_size, obs_dim]

        obs_expanded = obs.unsqueeze(2).expand(-1, -1, self.pop_dim)  # [B, obs_dim, pop_dim]
        stds = torch.exp(self.log_stds)  # [1, obs_dim, pop_dim]
        pop_activity = torch.exp(
            -0.5 * (obs_expanded - self.means).pow(2) / stds.pow(2)
        ).view(batch_size, -1)  # [B, obs_dim * pop_dim]

        pop_spikes = torch.zeros(
            batch_size, self.encoder_neuron_num, self.num_steps, device=obs.device
        )
        pop_mem = self.if1.reset_mem()
        for t in range(self.num_steps):
            spikes, pop_mem = self.if1(pop_activity, pop_mem)
            pop_spikes[:, :, t] = spikes

        if self.record:
            self._trace.append(
                {
                    "obs": obs.detach().cpu(),  # [B, obs_dim]
                    "pop_activity": pop_activity.detach()
                    .cpu()
                    .view(batch_size, self.obs_dim, self.pop_dim),
                    "pop_spikes": pop_spikes.detach().cpu(),  # [B, obs_dim*pop_dim, num_steps]
                }
            )

        return pop_spikes
