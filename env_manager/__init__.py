"""Environment-manager extensions for this repo.

Mirrors aerial_gym.env_manager: classes that own the per-reset state of the environment
(as opposed to config/, which only holds data, and task/, which owns reward and
observation logic).
"""

from .poisson_asset_manager import PoissonAssetManager

__all__ = ["PoissonAssetManager"]
