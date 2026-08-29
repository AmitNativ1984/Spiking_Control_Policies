"""Place (and park) env assets even when the curriculum asks for zero obstacles.

THE BUG (upstream, aerial_gym/env_manager/env_manager.py :: EnvManager.reset_idx)
---------------------------------------------------------------------------------
Asset placement is guarded on the requested obstacle count::

    num_obstacles = self.global_tensor_dict["num_obstacles_in_env"]
    if num_obstacles > 0:
        self.asset_manager.reset_idx(env_ids, num_obstacles)
        ...

and this task publishes ``num_obstacles_in_env = curriculum_level``. At curriculum
level 0 that is 0, so ``asset_manager.reset_idx`` is NEVER CALLED.

Placement is also what PARKS the unused assets at -1000. Skipping it does not give an
empty room -- it leaves every preallocated asset (201 per env here: the 20x20 m floor
slabs, panels, trees, spheres, cylinders, perimeter walls) stacked at its spawn position,
the world origin, for the entire run.

MEASURED (32 envs, level 0, after a full env_manager.reset_idx + step): all 201 assets
per env at (0,0,0), none parked. Warp mesh vertices inside the env box, by x:

    x in [ -6, -4):      1211
    x in [ -4, -2):      3169
    x in [ -2,  0):    506577      <-- 99% of the geometry
    x in [  0,  2):    555663      <--
    x in [  2,  4):      5663
    x in [  4,  6):      1027
    x in [  6, 14):       128

The robot spawns at x in [3, 5]. Targets sit on the four vertical walls, so:

  * +x  (x = 13.2)  -> corridor is empty                  -> best success rate
  * +/-y (y = +/-9.2) -> mostly lateral, clips the pile   -> intermediate
  * -x  (x = -5.2)  -> must cross x in [-2, 2]            -> EXACTLY 0.000, always

and that deadlocks the whole curriculum, because increases gate on the worst face:

    level 0 -> assets never placed -> pile at origin -> -x impossible
            -> worst_face_rate = 0 -> curriculum never increases -> level 0

Observed over 389 epochs of run 260764 and reproduced in 260784: -x pinned at exactly
0.000 while +x/+y/-y climbed to 0.23/0.18/0.19, obstacle_intensity stuck at 0.0, and a
~81% crash rate in what was believed to be an empty room.

THE FIX
-------
Always call ``asset_manager.reset_idx``. With a zero draw it places the structural slots
and parks everything else at -1000, which is what "no obstacles" is supposed to mean.
The extra half-density resample stays guarded on ``num_obstacles > 0`` exactly as
upstream has it, so behaviour above level 0 is unchanged.

Applied as a monkeypatch because aerial_gym lives in the container image. Remove once we
build against an aerial_gym carrying the fix.
"""

import torch

from aerial_gym.env_manager.env_manager import EnvManager
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("asset_placement_patch")

_APPLIED = False


def fixed_reset_idx(self, env_ids=None):
    """Upstream EnvManager.reset_idx, with placement no longer skipped at zero obstacles."""
    # first reset the Isaac Gym environment since that determines the environment bounds
    self.IGE_env.reset_idx(env_ids)

    num_obstacles = self.global_tensor_dict["num_obstacles_in_env"]

    # >>> THE FIX <<< unconditional. At num_obstacles == 0 this still runs, placing the
    # structural slots and parking every other asset at -1000. Upstream's `if
    # num_obstacles > 0` guard leaves them all stacked at the origin instead.
    self.asset_manager.reset_idx(env_ids, num_obstacles)

    if num_obstacles > 0:
        # Unchanged from upstream: a 15% subset of envs gets a second, half-density
        # draw. Meaningless at zero obstacles, so it stays guarded.
        nk = self.asset_manager.num_keep_in_env
        self.asset_manager.num_keep_in_env = self.asset_manager.num_keep_in_env // 2
        samples = torch.bernoulli(0.15 * torch.ones(len(env_ids), device=self.device))
        selected_indices = torch.nonzero(samples).squeeze(-1)
        if len(selected_indices) > 0:
            self.asset_manager.reset_idx(env_ids[selected_indices], num_obstacles // 2)
        self.asset_manager.num_keep_in_env = nk

    if self.cfg.env.use_warp:
        self.warp_env.reset_idx(env_ids)
    self.robot_manager.reset_idx(env_ids)
    self.IGE_env.write_to_sim()


def apply():
    """Install the patch. Idempotent; must run before the sim is built."""
    global _APPLIED
    if _APPLIED:
        return
    EnvManager.reset_idx = fixed_reset_idx
    _APPLIED = True
    logger.warning(
        "Patched EnvManager.reset_idx: assets are now placed/parked even when the "
        "curriculum requests zero obstacles (see env_manager/asset_placement_patch.py)."
    )
