"""
Loader for a frozed ANN teacher used in teacher-student (ANN -> SNN)
distillation.

Builds the FULL rl_games model wrapper (ModelA2CContinuousLogStd.Network), which owns
the obs/value normalization (running_mean_std / value_mean_std). The teacher checkpoint
stores those stats *inside* its `model` state_dict, si a single load_state_dict restores
weights + normalization together. The returned model normalizes obs internally in its forwward(),
so callers feed it RAW (un-normalized) observations.
"""

import torch
from rl_games.algos_torch.model_builder import ModelBuilder

def build_teacher(teacher_network_cfg: dict,
                  model_name: str = 'continuous_a2c_logstd',
                  obs_dim: int = 4,
                  action_dim: int = 4,
                  checkpoint_path: str = None,
                  device: str = 'cuda',
                  normalize_input: bool = True,
                  normalize_value: bool = True) -> torch.nn.Module:
    """Build, load, and freeze the ANN teacher as a full rl_games model wrapper.
    
    Args:
        teacher_network_cfg (dict): Configuration for the teacher network (e.g., hidden layers, activation).
        model_name (str): Name of the rl_games model to build (default: 'continuous_a2c_logstd').
        obs_dim (int): Dimension of the observation space.
        action_dim (int): Dimension of the action space.
        checkpoint_path (str): Path to the teacher checkpoint file.
        device (str): Device to load the model onto ('cuda' or 'cpu').
        normalize_input (bool): Whether to normalize input observations.
        normalize_value (bool): Whether to normalize value outputs.

    Returns:
        torch.nn.Module: frozen, eval-mode teacher model.     
    """

    params = {
        "model": {"name": model_name},
        "network": teacher_network_cfg
    }

    # Build the model factory
    builder = ModelBuilder()
    model_factory = builder.load(params)
    model = model_factory.build({
        "actions_num": action_dim,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_input": normalize_input,
        "normalize_value": normalize_value,
    })
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    assert "model" in ckpt, (
        f"Checkpoint at {checkpoint_path} has no 'model' key; got keys: {list(ckpt.keys())}"
    )
    model.load_state_dict(ckpt["model"], strict=True)

    # Freeze + eval. No gradients ever flow into the teacher:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model