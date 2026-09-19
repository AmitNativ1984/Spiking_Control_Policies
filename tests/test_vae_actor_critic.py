"""The end-to-end VAE network must start as the SAME FUNCTION as the policy it seeds from.

That is the one property worth testing here, and it is a strong one: it can only hold if the
depth pipeline, the image reshape, the latent slice (mu, not logvar), the state
normalisation and the actor/critic weight mapping are ALL correct. Any one of them wrong and
the actions diverge. It is also the property that makes the experiment interpretable -- the
run starts from p_cbf's behaviour and changes only because the encoder is now learning.

The reference is deliberately built from the OTHER implementation: the frozen
DepthVAEImageEncoder that the env uses, plus crash_cause_eval.build_actor's reconstruction
of the MLP trunk, plus rl_games' own normalise-and-clip. So this compares two independent
paths rather than a function against itself.

Needs the real checkpoints; skips cleanly without them.
"""
import math
import os

import pytest
import torch

from config.task_config.f450_attitude_navigation_task_config import task_config

VAE_CKPT = task_config.vae_config.model_file
POLICY_CKPT = (
    "runs/f450_nav_p_cbf_ft_hot/2026-09-18_13-50-26/nn/"
    "last_f450_nav_p_cbf_ft_hot_ep_2350_rew_1.7070831.pth"
)

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VAE_CKPT) and os.path.exists(POLICY_CKPT)),
    reason="needs the DepthVAE and p_cbf checkpoints",
)

H = task_config.vae_config.target_height
W = task_config.vae_config.target_width
STATE_DIM = task_config.state_dim
LATENT = task_config.vae_config.latent_dims
NET_CFG = {
    "state_dim": STATE_DIM,
    "img_height": H,
    "img_width": W,
    "latent_dim": LATENT,
    "sensor_max_range": 10.0,
    "max_depth_m": task_config.vae_config.max_depth_m,
    "min_depth_m": task_config.vae_config.min_depth_m,
    "encoder_checkpoint": VAE_CKPT,
    "policy_checkpoint": POLICY_CKPT,
    "actor": {"hidden_dims": [256, 256, 64], "activation": "elu"},
    "critic": {"hidden_dims": [256, 256, 64], "activation": "elu"},
}


def _build_net():
    from rl_training.rl_games.networks.ann import VAEActorCriticNetwork
    net = VAEActorCriticNetwork(
        input_dim=STATE_DIM + H * W, action_dim=4, **NET_CFG
    )
    return net.eval()


def _fake_batch(n=8, seed=0):
    """A plausible depth frame and state. Depth carries the simulator's own encoding:
    [0, 1] scaled by sensor_max_range, with NEGATIVE meaning nearer than min_range -- so a
    few negative pixels are included on purpose, since they are the case most likely to be
    mishandled."""
    g = torch.Generator().manual_seed(seed)
    img = torch.rand((n, 1, H, W), generator=g) * 0.7 + 0.05
    img[:, :, :4, :4] = -1.0                      # nearer than min_range
    img[:, :, -4:, -4:] = 1.0                     # no return / beyond max
    state = torch.randn((n, STATE_DIM), generator=g)
    return img, state


def _reference_mu(img, state):
    """mu through the frozen env-side encoder plus the independently rebuilt MLP trunk."""
    from vae_depth.vae_image_encoder import DepthVAEImageEncoder
    from analysis.crash_cause_eval import build_actor, _NORM_EPS, _NORM_CLAMP

    enc = DepthVAEImageEncoder(config=task_config.vae_config, device="cpu")
    latent = enc.encode(img.squeeze(1))

    ck = torch.load(POLICY_CKPT, map_location="cpu", weights_only=False)
    w = ck["model"]
    actor, _ = build_actor(w)
    actor.eval()
    mean = w["running_mean_std.running_mean"].float()
    std = torch.sqrt(w["running_mean_std.running_var"].float() + _NORM_EPS)

    obs49 = torch.cat([state, latent], dim=1)
    normed = torch.clamp((obs49 - mean) / std, -_NORM_CLAMP, _NORM_CLAMP)
    with torch.no_grad():
        return actor(normed)


def test_it_reproduces_the_seed_policy():
    """The whole point: identical actions at initialisation."""
    net = _build_net()
    img, state = _fake_batch()
    obs = torch.cat([state, img.reshape(img.shape[0], -1)], dim=1)

    with torch.no_grad():
        mu, _, _, states = net({"obs": obs})
    ref = _reference_mu(img, state)

    assert states is None, "feed-forward net must report no recurrent state"
    assert mu.shape == ref.shape == (img.shape[0], 4)
    # Two independent implementations in float32 through a conv stack: a few 1e-5 of
    # accumulated difference is expected, a systematic error is not.
    assert torch.allclose(mu, ref, atol=2e-4, rtol=1e-3), (
        f"max |diff| {float((mu - ref).abs().max()):.2e}\nnet {mu[0]}\nref {ref[0]}"
    )


def test_the_latent_half_is_mu_not_logvar():
    """The encoder emits [mu, logvar] concatenated; taking the wrong half would still run
    and still train, just from a meaningless code."""
    net = _build_net()
    img, state = _fake_batch()
    obs = torch.cat([state, img.reshape(img.shape[0], -1)], dim=1)
    with torch.no_grad():
        feats = net.encode(obs)
        raw = net.encoder(
            __import__("vae_depth.preprocessing", fromlist=["normalize_depth"])
            .normalize_depth(img * 10.0, task_config.vae_config.max_depth_m,
                             task_config.vae_config.min_depth_m)
        )
    assert feats.shape == (img.shape[0], STATE_DIM + LATENT)

    # encode() normalises AND clamps to +/-5, so replicate both rather than inverting them:
    # synthetic noise images push the latent well outside the donor's statistics, the clamp
    # bites, and a de-normalisation could not recover the raw value.
    def expect(half):
        return torch.clamp(
            (half - net.feat_mean[STATE_DIM:]) / net.feat_std[STATE_DIM:], -5.0, 5.0
        )

    assert torch.allclose(feats[:, STATE_DIM:], expect(raw[:, :LATENT]), atol=1e-5)
    # And it must actually discriminate: the logvar half would also have the right shape.
    assert not torch.allclose(feats[:, STATE_DIM:], expect(raw[:, LATENT:]), atol=1e-2)


def test_gradients_reach_the_encoder():
    """The reason the network exists. A frozen encoder here would train nothing new."""
    net = _build_net()
    img, state = _fake_batch()
    obs = torch.cat([state, img.reshape(img.shape[0], -1)], dim=1)
    mu, _, value, _ = net({"obs": obs})
    (mu.square().mean() + value.square().mean()).backward()
    grads = [p.grad for p in net.encoder.parameters() if p.grad is not None]
    assert grads, "no encoder parameter received a gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads), "encoder gradients are all zero"


def test_freezing_the_encoder_is_respected():
    cfg = dict(NET_CFG, train_encoder=False)
    from rl_training.rl_games.networks.ann import VAEActorCriticNetwork
    net = VAEActorCriticNetwork(input_dim=STATE_DIM + H * W, action_dim=4, **cfg).eval()
    assert all(not p.requires_grad for p in net.encoder.parameters())
    assert all(p.requires_grad for p in net.actor.parameters())


def test_feature_stats_cover_state_AND_latent():
    """The donor's normalize_input normalised all 49 inputs, so its trunk never saw a raw
    latent. Copying only the state half was the original bug here -- it shifted every action
    by up to 1.28 -- so this pins the full-width copy."""
    net = _build_net()
    w = torch.load(POLICY_CKPT, map_location="cpu", weights_only=False)["model"]
    assert net.feat_mean.numel() == STATE_DIM + LATENT
    assert torch.allclose(net.feat_mean, w["running_mean_std.running_mean"].float())
    assert torch.allclose(
        net.feat_std, torch.sqrt(w["running_mean_std.running_var"].float() + 1e-5)
    )


def test_a_wrong_observation_size_is_rejected_loudly():
    from rl_training.rl_games.networks.ann import VAEActorCriticNetwork
    with pytest.raises(AssertionError, match="flattened"):
        VAEActorCriticNetwork(input_dim=49, action_dim=4, **NET_CFG)


def test_the_config_default_keeps_the_encoder_frozen():
    """End-to-end is opt-in via F450_TRAIN_VAE; the default must leave every other arm and
    the 49-D observation untouched."""
    assert task_config.vae_config.train_encoder is False
    assert task_config.observation_space_dim == STATE_DIM + LATENT


def test_the_buffer_cost_is_what_the_yaml_assumes():
    """A guard on the number that forced num_actors down: if the sensor resolution changes,
    this fails and whoever changed it has to revisit the actor count."""
    floats_per_env = STATE_DIM + H * W
    gb = 128 * 128 * floats_per_env * 4 / 1024 ** 3
    assert floats_per_env == 57617
    assert math.isclose(gb, 3.52, abs_tol=0.2), f"{gb:.2f} GB at 128 actors x 128 horizon"
