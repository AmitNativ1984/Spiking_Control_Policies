"""Actor-critic with the depth encoder INSIDE the network, so PPO gradients reach it.

WHY THIS EXISTS. Everywhere else in this tree the DepthVAE is frozen and lives in the ENV:
task.process_image_observation() encodes the depth image and writes 32 latents into the
observation tensor, which crosses into rl_games as DATA. No gradient can reach the encoder
from the PPO loss, so the latent is whatever minimises RECONSTRUCTION error -- and there is
no reason the 32 dimensions that best reconstruct a depth image are the 32 that best fly
through a forest. This network moves the encoder inside forward() so the policy and value
losses shape the latent for CONTROL.

THE PRICE IS THE ROLLOUT BUFFER. The observation must now carry the raw depth image instead
of 32 numbers: 17 + 180*320 = 57,617 floats per env against 49. At the 1024 actors the other
arms use that is ~30 GB of rollout buffer, which does not fit beside the sim on a 40 GB
card, so this configuration runs at 128 actors (~3.8 GB). Consequences to keep in mind when
reading a result: 16,384 samples per update instead of 131,072, and a rollout/episode ratio
of ~0.55 against episodes of 230+ steps, so updates are noisier than the other arms'.

THE SHAPES ARE CHOSEN SO A p_cbf CHECKPOINT TRANSFERS EXACTLY. `actor` and `critic` are the
same ANNMLPActor/ANNMLPCritic as the feed-forward network, at the same obs_dim
(state_dim + latent_dim = 49), under the same attribute names -- so their weights load
unchanged and, with the pre-trained encoder in front, this network is the SAME FUNCTION as
the policy it starts from. tests/test_vae_actor_critic.py asserts that numerically.

THIS NETWORK LOADS ITS OWN PRE-TRAINED WEIGHTS, from `encoder_checkpoint` and
`policy_checkpoint` in the yaml, rather than going through rl_games' --checkpoint resume.
That is deliberate: resuming would require an optimizer state matching the parameter set,
and the parameter set has CHANGED (the encoder's weights are new), so rl_games would either
refuse it or silently restore a mismatched one. Starting fresh gives a correctly-sized
optimizer and a clean epoch counter, and the weights arrive here instead.

INPUT NORMALISATION MOVES IN HERE TOO, and it has to. rl_games' normalize_input would fit a
running mean/std over all 57,617 inputs, including the depth pixels -- which would hand the
pre-trained encoder something quite unlike the [0,1] depth it was trained on, making those
weights worthless. So the yaml sets normalize_input: False and this network normalises the
49 FEATURES instead, after encoding: state AND latent, exactly as the donor policy's
normalize_input did. Normalising only the state was tried and is wrong -- the donor's trunk
was trained on normalised latents, so feeding it raw VAE mu shifts every action (the
equivalence test caught it at max |diff| 1.28). Depth reaches the encoder through the
encoder's own documented pipeline: * sensor_max_range -> metres -> normalize_depth.

The statistics are FIXED buffers copied from the donor, not a running estimate. They go
stale as the encoder drifts, which is harmless: a fixed affine on the encoder's output is
something the encoder itself can absorb, whereas a running normaliser over a latent that is
simultaneously being trained is a moving target under PPO.

--- THE FIRST ATTEMPT AT THIS DESTROYED THE POLICY, AND WHY -----------------------------

Job 271932 started from a verified exact copy of a policy at 88.4% success and reached
92.9% CRASH within 309 epochs. The diagnostic was the KL: 0.0666 against a 0.008 target,
with the adaptive scheduler having already cut the learning rate to 3.8e-5. Updates far
outside the trust region despite a small LR means the gradient direction was inconsistent
between minibatches, and there were two compounding reasons:

  1. THE REPRESENTATION MOVES UNDER THE POLICY. PPO's trust region constrains the policy
     PARAMETERS, but training the encoder also changes the observation -> feature map, so a
     small parameter step can be a large behavioural step and the KL guard cannot do its
     job. This is the structural hazard of going end-to-end from a policy that was trained
     against a frozen encoder.
  2. THE VALUE LOSS DOMINATED THE ENCODER. 2.4M encoder parameters share one grad_norm clip
     with ~150k head parameters, and critic_coef is 2, so the value loss carried twice the
     policy loss into them. Early on the critic is mis-calibrated for the 8x smaller batch
     this configuration forces, so large value gradients landed on the pre-trained features
     and wrecked them -- and once the latent is noise, the policy that reads it fails.

Two mechanisms answer those, both off by default in the sense that setting them neutral
reproduces the failed run exactly:

  encoder_grad_scale       multiplies every gradient entering the encoder, leaving the heads
                           untouched, so the encoder learns ~100x slower than the policy on
                           top of it. This is the standard way to fine-tune a pre-trained
                           trunk, expressed as a gradient scale rather than a second
                           optimizer param group because rl_games builds the optimizer over
                           model.parameters() and does not expose groups.
  critic_detaches_encoder  the value head reads the latent but sends no gradient back into
                           it, so only the policy loss shapes the representation. The critic
                           still learns freely on top of whatever the encoder produces.
"""

from typing import Tuple

import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder

from vae_depth.model import DepthVAE
from vae_depth.preprocessing import normalize_depth

from ._utils import xavier_init_linear
from .actor import ANNMLPActor
from .critic import ANNMLPCritic

_NORM_CLAMP = 5.0  # matches rl_games' normalize_input clipping


class _ScaleGrad(torch.autograd.Function):
    """Identity forward, scaled gradient backward.

    Lets the encoder train at a different effective learning rate from the heads without a
    second optimizer param group -- rl_games constructs the optimizer over
    model.parameters() and exposes no way to group them. Because the forward pass is exactly
    the identity, a network with any scale still reproduces its seed policy's actions, which
    is what tests/test_vae_actor_critic.py checks.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class VAEActorCriticNetworkBuilder(NetworkBuilder):
    def load(self, params):
        """rl_games calls this with params = the YAML's `network:` block (already unwrapped)."""
        self.config = params

    def build(self, name, **kwargs):
        return VAEActorCriticNetwork(
            input_dim=kwargs["input_shape"][0],
            action_dim=kwargs["actions_num"],
            **self.config,
        )


class VAEActorCriticNetwork(nn.Module):
    """Config keys (under `network:`), with the defaults matching the task config:

        state_dim / img_height / img_width / latent_dim
        sensor_max_range / max_depth_m / min_depth_m
        train_encoder            False freezes it again, for an A/B against this very run
        encoder_grad_scale       gradient multiplier into the encoder (default 0.01)
        critic_detaches_encoder  True keeps the value loss out of the encoder (default True)
        encoder_checkpoint       DepthVAE .pth to seed the encoder from
        policy_checkpoint        MLP policy .pth to seed actor/critic and the state stats
        actor.hidden_dims / actor.activation
        critic.hidden_dims / critic.activation
    """

    def __init__(self, input_dim, action_dim, **config):
        super().__init__()

        self.state_dim = int(config.get("state_dim", 17))
        self.img_h = int(config.get("img_height", 180))
        self.img_w = int(config.get("img_width", 320))
        self.latent_dim = int(config.get("latent_dim", 32))
        self.sensor_max_range = float(config.get("sensor_max_range", 10.0))
        self.max_depth_m = float(config.get("max_depth_m", 7.0))
        self.min_depth_m = float(config.get("min_depth_m", 0.1))
        self.train_encoder = bool(config.get("train_encoder", True))
        # See the module docstring: 1.0 and False together reproduce the run that collapsed.
        self.encoder_grad_scale = float(config.get("encoder_grad_scale", 0.01))
        self.critic_detaches_encoder = bool(config.get("critic_detaches_encoder", True))

        expected = self.state_dim + self.img_h * self.img_w
        assert input_dim == expected, (
            f"this network expects state + a flattened {self.img_h}x{self.img_w} depth "
            f"image = {expected} inputs, but the task produces {input_dim}. Set "
            f"vae_config.train_encoder = True so the task emits raw depth instead of "
            f"latents."
        )

        self.encoder = DepthVAE(latent_dim=self.latent_dim).encoder

        feat_dim = self.state_dim + self.latent_dim
        self.actor = ANNMLPActor(
            obs_dim=feat_dim, action_dim=action_dim, actor_config=config.get("actor", {})
        )
        self.critic = ANNMLPCritic(obs_dim=feat_dim, critic_config=config.get("critic", {}))

        # Xavier the heads only. Walking the whole module would re-initialise the encoder,
        # throwing away the pre-training this network exists to build on.
        xavier_init_linear(self.actor)
        xavier_init_linear(self.critic)

        # Feature normalisation over [state, latent], replacing rl_games' normalize_input
        # (see the module docstring). Identity until seeded.
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))

        self._load_pretrained(
            config.get("encoder_checkpoint"), config.get("policy_checkpoint")
        )

        if not self.train_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _load_pretrained(self, encoder_ckpt, policy_ckpt):
        """Seed the encoder from a DepthVAE checkpoint and the heads from an MLP policy.

        Both are optional and both are LOUD: a path that is set but unusable raises rather
        than leaving randomly-initialised weights in place, because a silently un-seeded
        encoder is indistinguishable from a bad experiment until the run has burned a day.
        """
        if encoder_ckpt:
            ck = torch.load(encoder_ckpt, map_location="cpu")
            sd = ck.get("model_state_dict", ck)
            enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            if not enc:
                raise RuntimeError(
                    f"no encoder.* weights in {encoder_ckpt}; keys look like "
                    f"{list(sd)[:4]}"
                )
            missing, unexpected = self.encoder.load_state_dict(enc, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"encoder weights do not match this DepthVAE: missing {missing}, "
                    f"unexpected {unexpected}"
                )
            print(f"[vae_actor_critic] encoder seeded from {encoder_ckpt}", flush=True)

        if policy_ckpt:
            ck = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
            w = ck["model"]
            inner = {k[len("a2c_network."):]: v for k, v in w.items()
                     if k.startswith("a2c_network.")}
            heads = {k: v for k, v in inner.items()
                     if k.startswith(("actor.", "critic."))}
            if not heads:
                raise RuntimeError(
                    f"no a2c_network.actor/critic weights in {policy_ckpt}; keys look like "
                    f"{list(inner)[:4]}"
                )
            for mod, prefix in ((self.actor, "actor."), (self.critic, "critic.")):
                part = {k[len(prefix):]: v for k, v in heads.items() if k.startswith(prefix)}
                missing, unexpected = mod.load_state_dict(part, strict=False)
                if missing or unexpected:
                    raise RuntimeError(
                        f"{prefix[:-1]} weights do not match: missing {missing}, "
                        f"unexpected {unexpected} -- the trunk shape must be identical for "
                        f"the transfer to mean anything."
                    )
            # And the input normalisation the donor was trained with, over ALL 49
            # features -- state and latent. Taking only the state half leaves the trunk
            # reading raw VAE mu, which it has never seen.
            rm = w.get("running_mean_std.running_mean")
            rv = w.get("running_mean_std.running_var")
            if rm is None or rv is None:
                raise RuntimeError(
                    f"{policy_ckpt} has no running_mean_std -- this network replaces "
                    f"rl_games' normalize_input and needs the donor's statistics."
                )
            n = self.feat_mean.numel()
            if rm.numel() != n:
                raise RuntimeError(
                    f"{policy_ckpt} normalises {rm.numel()} inputs but this network has "
                    f"{n} features (state {self.state_dim} + latent {self.latent_dim}); the "
                    f"donor must share the feature layout for the transfer to be exact."
                )
            self.feat_mean.copy_(rm.float())
            self.feat_std.copy_(torch.sqrt(rv.float() + 1e-5))
            print(f"[vae_actor_critic] actor/critic + state stats seeded from "
                  f"{policy_ckpt}", flush=True)

    def is_rnn(self):
        return False

    def get_aux_loss(self):
        """Required by rl_games >= 1.6.5, which calls this on every a2c_network."""
        return None

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, state_dim + H*W) -> (B, state_dim + latent_dim) features.

        The depth half goes through exactly the pipeline DepthVAEImageEncoder.encode uses,
        so a frozen encoder here reproduces the frozen encoder there bit for bit:
        simulator-normalised depth -> metres -> normalize_depth -> encoder -> take mu.
        The resize that wrapper performs is a no-op at this resolution (the sensor and the
        VAE's target are both 180x320) and is therefore left out rather than reproduced.

        NOTE the sign convention the depth image carries: a NEGATIVE pixel means nearer than
        min_range, not a missing return, and normalize_depth's clamp maps it to 1.0 (as
        near as the encoding can express). That is the safe reading and it matches what the
        frozen encoder has always seen.
        """
        state = obs[:, : self.state_dim]
        img = obs[:, self.state_dim:].reshape(-1, 1, self.img_h, self.img_w)

        depth_m = img * self.sensor_max_range
        x = normalize_depth(depth_m, self.max_depth_m, self.min_depth_m)
        z = self.encoder(x)[:, : self.latent_dim]  # mu; the encoder emits [mu, logvar]
        if self.train_encoder and self.encoder_grad_scale != 1.0:
            # Identity forward, so this cannot change what the policy does -- only how fast
            # the encoder moves relative to the heads reading it.
            z = _ScaleGrad.apply(z, self.encoder_grad_scale)

        # Normalise [state, latent] as ONE vector, which is what the donor's normalize_input
        # did -- its trunk never saw a raw latent.
        feats = torch.cat([state, z], dim=1)
        return torch.clamp(
            (feats - self.feat_mean) / self.feat_std, -_NORM_CLAMP, _NORM_CLAMP
        )

    def forward(self, obs_dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Returns (mu, log_std, value, states); states is None for a feed-forward net.

        Actor and critic share the encoder's OUTPUT, but by default only the policy loss
        shapes the encoder itself: critic_coef is 2, so letting the value loss through gave
        it twice the policy's weight over the representation, and that is what wrecked the
        first attempt (see the module docstring). Detaching costs the encoder some gradient
        signal and buys a representation that only the objective we care about moves.

        Detaching the whole feature vector is equivalent to detaching just the latent here:
        the state half descends from the observation, which is an input, not a parameter.
        """
        features = self.encode(obs_dict["obs"])
        mu, log_std = self.actor(features)
        value = self.critic(features.detach() if self.critic_detaches_encoder else features)
        return mu, log_std, value, None
