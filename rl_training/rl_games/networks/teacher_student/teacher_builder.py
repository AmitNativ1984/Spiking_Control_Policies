"""Loader for a frozen ANN teacher used in teacher-student (ANN -> SNN) distillation.

Builds the FULL rl_games model wrapper (ModelA2CContinuousLogStd.Network), which owns the
obs/value normalization (running_mean_std / value_mean_std). The teacher checkpoint stores
those stats *inside* its `model` state_dict, so a single load_state_dict restores weights
and normalization together. The returned model normalizes obs internally in its forward(),
so callers feed it RAW (un-normalized) observations.
"""

import torch
from rl_games.algos_torch.model_builder import ModelBuilder


def build_teacher(
    teacher_network_cfg: dict,
    model_name: str = "continuous_a2c_logstd",
    obs_dim: int = 4,
    action_dim: int = 4,
    checkpoint_path: str = None,
    device: str = "cuda",
    normalize_input: bool = True,
    normalize_value: bool = True,
) -> torch.nn.Module:
    """Build, load, and freeze the ANN teacher as a full rl_games model wrapper.

    Args:
        teacher_network_cfg: The teacher's `network:` block (hidden layers, activation, ...).
        model_name: rl_games model to build.
        obs_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        checkpoint_path: Path to the teacher checkpoint file.
        device: Device to load the model onto.
        normalize_input: Whether the model normalizes input observations.
        normalize_value: Whether the model normalizes value outputs.

    Returns:
        A frozen, eval-mode teacher model.
    """
    builder = ModelBuilder()
    model_factory = builder.load(
        {"model": {"name": model_name}, "network": teacher_network_cfg}
    )
    model = model_factory.build(
        {
            "actions_num": action_dim,
            "input_shape": (obs_dim,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_input": normalize_input,
            "normalize_value": normalize_value,
        }
    )
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    assert "model" in ckpt, (
        f"Checkpoint at {checkpoint_path} has no 'model' key; got keys: {list(ckpt.keys())}"
    )
    model.load_state_dict(ckpt["model"], strict=True)

    # Freeze + eval. No gradients ever flow into the teacher.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model
