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
    out = teacher({"is_train": True, "obs": torch.randn(4, obs_dim, device="cuda"), "prev_actions": None})
    assert not out["mus"].requires_grad