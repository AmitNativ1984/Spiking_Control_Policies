from pathlib import Path

import numpy as np

from aerial_gym.config.asset_config.env_object_config import asset_state_params

_ENVIRONMENT_ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "resources" / "models" / "environment_assets"


class sphere_asset_params(asset_state_params):
    """3 fixed-radius spheres (0.2/0.4/0.6 m), randomly selected with replacement.
    Same randomization style as object_asset_params: full rotation, culled by curriculum
    (keep_in_env=False)."""

    num_assets = 15

    asset_folder = str(_ENVIRONMENT_ASSETS_DIR / "spheres")

    min_state_ratio = [
        0.30,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.9,
        0.9,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = False
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance


class cylinder_asset_params(asset_state_params):
    """3 fixed (radius, length) cylinders (post/stub/barrel), randomly selected with
    replacement. Same randomization style as object_asset_params."""

    num_assets = 15

    asset_folder = str(_ENVIRONMENT_ASSETS_DIR / "cylinders")

    min_state_ratio = [
        0.30,
        0.05,
        0.05,
        -np.pi,
        -np.pi,
        -np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    max_state_ratio = [
        0.85,
        0.9,
        0.9,
        np.pi,
        np.pi,
        np.pi,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    keep_in_env = False
    per_link_semantic = False
    semantic_id = -1  # will be assigned incrementally per instance
