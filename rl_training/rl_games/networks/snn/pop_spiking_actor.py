import math
from typing import Tuple

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

from .decoder import SpikeDecoder
from .encoder import PopulationSpikeEncoder

_SPIKE_GRADS = {
    "sigmoid": lambda: surrogate.sigmoid(slope=25),
    "atan": lambda: surrogate.atan(alpha=2.0),
    "fast_sigmoid": lambda: surrogate.fast_sigmoid(slope=25),
}


class PopulationSpikingActorNetwork(nn.Module):
    """Population-coded spiking actor (PopSAN): Gaussian encoder -> 3 Synaptic LIF layers
    -> population decoder, producing (mu, log_std) for a Gaussian policy.

    Everything it needs comes from `actor_config` — it never reads a task config. The
    runner bridges task -> network by injecting `observation_bounds` into the YAML's
    `network.actor` block before rl_games builds the model.

    Config keys:
        hidden_dims        two hidden layer sizes
        num_steps          spiking timesteps per forward pass
        spike_grad         one of _SPIKE_GRADS
        alpha, beta, threshold, reset_mechanism, reset_delay   snntorch.Synaptic params
        encoder            dict passed to PopulationSpikeEncoder (pop_dim, threshold)
        observation_bounds one (lo, hi) per observation dim, in rl_games-NORMALIZED space
        sigma_init         initial action std (default 1.0)
    """

    def __init__(self, input_dim, action_dim, **actor_config):
        super().__init__()

        hidden_dims = actor_config["hidden_dims"]
        self.num_steps = actor_config["num_steps"]
        self.threshold = actor_config["threshold"]
        self.pop_dim = actor_config["pop_dim"]
        self.action_dim = action_dim

        spike_grad_name = actor_config["spike_grad"]
        if spike_grad_name not in _SPIKE_GRADS:
            raise ValueError(
                f"Unsupported spike_grad: {spike_grad_name}. "
                f"Expected one of {sorted(_SPIKE_GRADS)}."
            )
        spike_grad = _SPIKE_GRADS[spike_grad_name]()

        # Per-dimension (min, max) bounds for the population encoder. Must be length
        # input_dim so the encoder builds means/stds of shape [1, obs_dim, pop_dim] —
        # NOT a single shared Gaussian set.
        obs_bounds = actor_config["observation_bounds"]
        assert len(obs_bounds) == input_dim, (
            f"observation_bounds has {len(obs_bounds)} entries but input_dim={input_dim}; "
            "they must match (one (min, max) per observation dimension)."
        )
        self.pop_encoder = PopulationSpikeEncoder(
            obs_dim=input_dim,
            obs_bounds=obs_bounds,
            num_steps=self.num_steps,
            encoder_config=actor_config["encoder"],
        )

        def lif():
            return snn.Synaptic(
                alpha=actor_config["alpha"],
                beta=actor_config["beta"],
                threshold=actor_config["threshold"],
                reset_mechanism=actor_config["reset_mechanism"],
                reset_delay=actor_config["reset_delay"],
                spike_grad=spike_grad,
            )

        self.actor_fc1 = nn.Linear(input_dim * self.pop_dim, hidden_dims[0])
        self.actor_lif1 = lif()
        self.actor_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.actor_lif2 = lif()
        self.actor_fc3 = nn.Linear(hidden_dims[1], action_dim * self.pop_dim)
        self.actor_lif3 = lif()

        self.action_decoder = SpikeDecoder(
            action_dim=self.action_dim, pop_dim=self.pop_dim
        )

        # State-independent log std of the action distribution. A learnable network
        # parameter, unrelated to the spiking decoder. Initialized so exp(log_std) ==
        # sigma_init (default 1.0).
        sigma_init = actor_config.get("sigma_init", 1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(sigma_init)))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Back-compat: log_std used to live on the decoder (action_decoder.log_std)
        # and warmup left it at its stale init (std=1.0). Legacy checkpoints have no
        # top-level log_std, so seed it from the freshly built sigma_init value to
        # satisfy strict loading; checkpoints that already trained log_std keep theirs.
        state_dict.pop(prefix + "action_decoder.log_std", None)
        log_std_key = prefix + "log_std"
        if log_std_key not in state_dict:
            state_dict[log_std_key] = self.log_std.data.clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def is_rnn(self):
        """Required by rl_games - indicates this is not an RNN network."""
        return False

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the spiking actor for num_steps and decode the mean spike rate.

        Returns:
            (mu, log_std), each of shape [batch_size, action_dim].
        """
        x = obs_dict["obs"]

        syn1, mem1 = self.actor_lif1.reset_mem()
        syn2, mem2 = self.actor_lif2.reset_mem()
        syn3, mem3 = self.actor_lif3.reset_mem()

        spike_train_in = self.pop_encoder(x)

        # Accumulated output spikes, averaged over timesteps before decoding.
        output_spikes = torch.zeros(
            x.size(0), self.actor_fc3.out_features, device=x.device
        )
        for t in range(self.num_steps):
            cur1 = self.actor_fc1(spike_train_in[:, :, t])
            spk1, syn1, mem1 = self.actor_lif1(cur1, syn1, mem1)

            cur2 = self.actor_fc2(spk1)
            spk2, syn2, mem2 = self.actor_lif2(cur2, syn2, mem2)

            cur3 = self.actor_fc3(spk2)  # [batch_size, action_dim * pop_dim]
            spk3, syn3, mem3 = self.actor_lif3(cur3, syn3, mem3)

            output_spikes += spk3

        output_spikes /= self.num_steps
        output_spikes = output_spikes.view(-1, self.action_dim, self.pop_dim)

        action_mu = self.action_decoder(output_spikes)
        action_log_std = self.log_std.expand_as(action_mu)
        return action_mu, action_log_std
