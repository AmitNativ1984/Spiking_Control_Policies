"""Rebuild each env's warp BVH after placement instead of only refitting it.

THE PROBLEM
-----------
``WarpEnv.reset_idx`` moves every vertex into world space and then calls
``wp_mesh.refit()``. ``refit()`` updates node BOUNDS but never the TOPOLOGY, which is
fixed when the ``wp.Mesh`` is constructed. Obstacles are re-placed at random positions on
every reset, so the tree quickly stops partitioning space in any useful way.

The symptom is that render cost becomes LINEAR in obstacle count -- i.e. the acceleration
structure stops accelerating. Measured at 256 envs, forced resets, refit vs rebuild:

    level  obstacles/env   refit render   rebuild render
      0          0             4.2 ms          --
      1        ~5.4          914.4 ms         5.3 ms     (173x)
      5       ~26.8         5126.9 ms        15.0 ms     (342x)

    scaling 5.4 -> 26.8 obstacles:  refit 5.6x (linear)   rebuild 2.8x (sub-linear)

Extrapolated to curriculum level 25 (~134 obstacles/env), refit projects to ~26 s per
step (~28 min/epoch), which puts the upper curriculum out of reach entirely. Rebuild
projects to roughly 60 ms.

Two things this is NOT (both measured and ruled out):
  * drone position -- scattering robots across the box changed nothing (767 vs 866 ms)
  * where culled assets are parked -- moving them from -1000 to -40 made it WORSE at both
    level 1 (1374 vs 866 ms) and level 5 (7127 vs 5255 ms)

THE FIX
-------
Construct a fresh ``wp.Mesh`` for each env being reset, over the same (already
world-space) point/index/velocity arrays. Warp builds a new BVH over the actual current
geometry.

The new meshes have new ids, and the camera's CUDA render graph captured the sensor's
``mesh_ids_array``, so the ids must be written into THAT SAME array in place
(``wp.copy``) -- rebinding the attribute would leave the graph reading the old array.

``bind(sim_env)`` must be called after the sim is built, so the sensors exist and can be
found. Without it the patch falls back to refit and says so, rather than silently
rendering stale geometry.
"""

import torch
import warp as wp

from aerial_gym.env_manager.warp_env_manager import WarpEnv
from aerial_gym.utils.math import tf_apply
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("warp_bvh_rebuild")

_APPLIED = False
_HOLDERS = []          # objects owning the mesh_ids_array the render graph captured
_STATS = {"calls": 0, "meshes": 0}


def _find_mesh_id_holders(root, max_depth=4):
    """Locate every object holding a `mesh_ids_array` (WarpSensor and the WarpCam it feeds)."""
    found, seen = [], set()

    def scan(obj, depth):
        if depth > max_depth or id(obj) in seen:
            return
        seen.add(id(obj))
        d = getattr(obj, "__dict__", None)
        if not d:
            return
        for key, val in list(d.items()):
            if key == "mesh_ids_array" and val is not None:
                found.append(obj)
            elif hasattr(val, "__dict__"):
                scan(val, depth + 1)
            elif isinstance(val, (list, tuple)):
                for item in val[:16]:
                    if hasattr(item, "__dict__"):
                        scan(item, depth + 1)

    scan(root, 0)
    return found


def rebuilding_reset_idx(self, env_ids):
    """WarpEnv.reset_idx with a BVH rebuild in place of refit()."""
    if self.global_vertex_counter == 0:
        return

    # identical to upstream: push the assets' world transforms onto the vertices
    self.vertex_maps_per_env_updated[:] = tf_apply(
        self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 3:7],
        self.unfolded_env_vec_root_tensor[self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR, 0:3],
        self.VERTEX_MAPS_PER_ENV_ORIGINAL[:],
    )

    if not _HOLDERS:
        # No sensor bound: refit rather than rebuild. Rebuilding without publishing the
        # new ids would leave the camera rendering meshes nothing points at.
        for i in env_ids:
            self.warp_mesh_per_env[i].refit()
        return

    _STATS["calls"] += 1
    ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
    for i in ids:
        idx = int(i)
        old = self.warp_mesh_per_env[idx]
        # same arrays, fresh tree: wp.Mesh builds a BVH over the current point values
        new = wp.Mesh(points=old.points, indices=old.indices, velocities=old.velocities)
        self.warp_mesh_per_env[idx] = new
        self.warp_mesh_id_list[idx] = new.id
        _STATS["meshes"] += 1

    if ids:
        # in place, so the captured render graph sees the new ids
        new_ids = wp.array(self.warp_mesh_id_list, dtype=wp.uint64)
        for holder in _HOLDERS:
            wp.copy(holder.mesh_ids_array, new_ids)


def bind(sim_env):
    """Point the patch at the built sim so it can publish new mesh ids. Call after build."""
    global _HOLDERS
    if not _APPLIED:
        return
    _HOLDERS = _find_mesh_id_holders(sim_env)
    if _HOLDERS:
        logger.warning(
            f"warp BVH rebuild bound to {len(_HOLDERS)} mesh_ids_array holder(s); "
            "BVHs will be rebuilt on reset instead of refit."
        )
    else:
        logger.error(
            "warp BVH rebuild could NOT find a mesh_ids_array holder -- falling back to "
            "refit(). Render cost will stay linear in obstacle count."
        )


def stats():
    return dict(_STATS)


def apply():
    """Install the patch. Idempotent; must run before the sim is built."""
    global _APPLIED
    if _APPLIED:
        return
    WarpEnv.reset_idx = rebuilding_reset_idx
    _APPLIED = True
    logger.warning(
        "Patched WarpEnv.reset_idx: BVHs rebuilt rather than refit "
        "(see env_manager/warp_bvh_rebuild_patch.py). Awaiting bind()."
    )
