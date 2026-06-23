import isaacgym
import torch
import pytest
import os 
import yaml
from rl_games.algos_torch import model_builder
from navigation_with_obstacles.networks.teacher_student.teacher_builder import build_teacher
from navigation_with_obstacles.config.task_config import task_config
from navigation_with_obstacles.networks.ann.actor_critic import MLPActorCriticNetworkBuilder

CONFIG_PATH = "navigation_with_obstacles/training/popsan_teacher_student_cluster.yaml"

@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        params = yaml.safe_load(f)["params"]
    return params

@pytest.fixture(scope="module")
def distill_cfg(cfg):
    d = cfg["config"].get("distillation")
    assert d is not None, "config.distillation missing from YAML"
    return d

@pytest.fixture(scope="module", autouse=True)
def _guard(distill_cfg):
    if not torch.cuda.is_available():
        pytest.skip("teacher build needs CUDA")
    if not os.path.exists(distill_cfg["checkpoint"]):
        pytest.skip(f"missing checkpoint: {distill_cfg['checkpoint']}")

@pytest.fixture(scope="module")
def dims():
    obs_dims = task_config.observation_space_dim
    action_dims = task_config.action_space_dim
    return obs_dims, action_dims

@pytest.fixture(scope="module")
def teacher(cfg, distill_cfg, dims):
    obs_dim, action_dim = dims

    # Mirror runner.py registration so ModelBuilder can resolve 'mlp_actor_critic'
    model_builder.register_network("mlp_actor_critic", MLPActorCriticNetworkBuilder)

    return build_teacher(
        teacher_network_cfg=distill_cfg["network"],
        model_name=cfg["model"]["name"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        checkpoint_path=distill_cfg["checkpoint"],
        device="cuda",
        normalize_input=distill_cfg["normalize_input"],
        normalize_value=distill_cfg["normalize_value"],
    )

def test_teacher_frozen_and_eval(teacher):
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())

def test_forward_no_grad(teacher, dims):
    obs_dim, _ = dims
    # is_train=False is the play-time / distillation path. (is_train=True would
    # require a real prev_actions tensor for neglogp, not None.)
    out = teacher({"is_train": False, "obs": torch.randn(4, obs_dim, device="cuda"),
                   "prev_actions": None, "rnn_states": None})
    assert not out["mus"].requires_grad


@pytest.fixture(scope="module")
def raw_obs(teacher, dims):
    """A realistic RAW (un-normalized) obs batch drawn from the teacher's own
    learned obs statistics, so magnitudes are in-distribution."""
    obs_dim, _ = dims
    torch.manual_seed(0)
    rms = teacher.running_mean_std
    mean = rms.running_mean.detach().to("cuda").float()
    std = rms.running_var.detach().to("cuda").float().sqrt()
    return mean + std * torch.randn(64, obs_dim, device="cuda")


def test_internal_normalization_matches_manual(teacher, raw_obs):
    """Phase 1 bullet 3: feed RAW obs, let the wrapper normalize internally, and
    verify the resulting mu matches a hand-rolled normalize-then-actor pass.
    This proves the internal running_mean_std is actually applied and correct."""
    with torch.no_grad():
        mu_play = teacher({
            "is_train": False,
            "prev_actions": None,
            "obs": raw_obs.clone(),
            "rnn_states": None,
        })["mus"]

        # rl_games RunningMeanStd: (x - mean)/sqrt(var + eps), clamped to +-5.
        rms = teacher.running_mean_std
        mean = rms.running_mean.to("cuda").float()
        var = rms.running_var.to("cuda").float()
        norm = torch.clamp((raw_obs - mean) / torch.sqrt(var + 1e-5), -5.0, 5.0)
        mu_manual, _ = teacher.a2c_network.actor(norm)

    max_diff = (mu_play - mu_manual).abs().max().item()
    assert max_diff < 1e-4, f"loader mu != manual normalize+actor (max {max_diff:.3e})"


def test_normalization_is_not_a_noop(teacher, raw_obs):
    """Guard against a silently-unloaded normalizer: passing RAW obs straight
    into the bare actor must differ from the normalized (play-time) mu."""
    with torch.no_grad():
        mu_play = teacher({
            "is_train": False, "prev_actions": None,
            "obs": raw_obs.clone(), "rnn_states": None,
        })["mus"]
        mu_on_raw, _ = teacher.a2c_network.actor(raw_obs)
    assert (mu_on_raw - mu_play).abs().max().item() > 1e-3, \
        "normalized vs raw mu identical — running_mean_std likely not loaded"


def test_mu_finite_and_sane_range(teacher, raw_obs):
    """Phase 1 bullet 4: mu is finite and within a plausible (pre-squash) range."""
    with torch.no_grad():
        mu = teacher({
            "is_train": False, "prev_actions": None,
            "obs": raw_obs.clone(), "rnn_states": None,
        })["mus"]
    assert torch.isfinite(mu).all(), "teacher mu contains NaN/Inf"
    assert mu.abs().max().item() < 50.0, \
        f"mu magnitude {mu.abs().max().item():.1f} implausible — obs scale mismatch?"