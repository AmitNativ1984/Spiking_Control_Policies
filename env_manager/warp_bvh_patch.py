"""Fix the degenerate warp BVH built by aerial_gym's WarpEnv.

THE BUG (upstream, aerial_gym/env_manager/warp_env_manager.py)
--------------------------------------------------------------
``WarpEnv.prepare_for_simulation`` allocates two vertex buffers::

    self.vertex_maps_per_env_original = torch.zeros((global_vertex_counter, 3), ...)
    self.vertex_maps_per_env_updated  = self.vertex_maps_per_env_original.clone()

then, per environment, fills the ORIGINAL buffer with the real mesh vertices but hands
the still-zero UPDATED buffer to warp::

    self.vertex_maps_per_env_original[sl] = torch.tensor(env_mesh.vertices, ...)
    vertex_vec3_array = wp.from_torch(self.vertex_maps_per_env_updated[sl], dtype=wp.vec3)
    wp_mesh = wp.Mesh(points=vertex_vec3_array, ...)

So every ``wp.Mesh`` is constructed while all of its vertices are (0, 0, 0). Warp builds
each BVH over a mesh collapsed to a single point: all centroids identical, so the split
heuristic produces an arbitrary, badly unbalanced tree.

``reset_idx`` later writes real world-space positions into that same buffer (``wp.from_torch``
shares storage) and calls ``mesh.refit()`` -- but refit only recomputes node BOUNDS, never
the TOPOLOGY. The tree therefore stays degenerate for the entire run.

Nothing errors and the depth images are geometrically CORRECT. The only symptom is speed:
ray traversal visits a huge fraction of nodes, and refit on that tree is expensive too.

MEASURED IMPACT (f450_navigation_task, 64 envs, A100, 20 steps)
---------------------------------------------------------------
                     baseline      with this fix
    render          2398.17 ms          12.16 ms   (197x)
    reset            648.32 ms           4.87 ms   (133x)
    total           3062.75 ms          32.33 ms    (95x)
    fps                   20.9          1979.5      (95x)

THE FIX
-------
Seed the buffer warp actually sees with the real vertices before ``wp.Mesh`` is built.
``fixed_prepare_for_simulation`` below is a faithful copy of upstream with exactly one
line added (marked ``THE FIX``).

Applied as a monkeypatch because aerial_gym is installed inside the container image; this
keeps the correction with our source so it survives image rebuilds. Remove once the fix
is carried by the aerial_gym we build against.
"""

import torch
import trimesh as tm
import warp as wp

from aerial_gym.env_manager.warp_env_manager import WarpEnv
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger("warp_bvh_patch")

_APPLIED = False


def fixed_prepare_for_simulation(self, global_tensor_dict):
    """Upstream WarpEnv.prepare_for_simulation, with the vertex buffer seeded correctly."""
    self.global_tensor_dict = global_tensor_dict
    if self.global_vertex_counter == 0:
        logger.warning(
            "No assets have been added to the environment. Skipping preparation for simulation"
        )
        for key in (
            "CONST_WARP_MESH_ID_LIST",
            "CONST_WARP_MESH_PER_ENV",
            "CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR",
            "VERTEX_MAPS_PER_ENV_ORIGINAL",
        ):
            self.global_tensor_dict[key] = None
        return 1

    self.global_vertex_to_asset_index_tensor = torch.tensor(
        self.global_vertex_to_asset_index_map, device=self.device, requires_grad=False
    )
    self.vertex_maps_per_env_original = torch.zeros(
        (self.global_vertex_counter, 3), device=self.device, requires_grad=False
    )
    self.vertex_maps_per_env_updated = self.vertex_maps_per_env_original.clone()

    for i in range(len(self.env_meshes)):
        self.global_env_mesh_list.append(tm.util.concatenate(self.env_meshes[i]))

    vertex_iterator = 0
    for env_mesh in self.global_env_mesh_list:
        n_verts = len(env_mesh.vertices)
        sl = slice(vertex_iterator, vertex_iterator + n_verts)

        self.vertex_maps_per_env_original[sl] = torch.tensor(
            env_mesh.vertices, device=self.device, requires_grad=False
        )

        # >>> THE FIX <<<
        # Give warp real geometry at BVH-build time. Upstream leaves this buffer zeroed
        # here, so wp.Mesh below builds its BVH over a mesh collapsed to the origin, and
        # refit() can never repair the topology afterwards. See module docstring.
        self.vertex_maps_per_env_updated[sl] = self.vertex_maps_per_env_original[sl]

        faces_tensor = torch.tensor(
            env_mesh.faces, device=self.device, requires_grad=False, dtype=torch.int32
        )
        vertex_velocities = torch.zeros(
            n_verts, 3, device=self.device, requires_grad=False
        )
        segmentation_tensor = torch.tensor(
            self.global_vertex_segmentation_list[sl], device=self.device, requires_grad=False
        )
        # upstream hijacks the velocity field to carry segmentation ids
        vertex_velocities[:, 0] = segmentation_tensor

        vertex_vec3_array = wp.from_torch(
            self.vertex_maps_per_env_updated[sl], dtype=wp.vec3
        )
        faces_wp_int32_array = wp.from_torch(faces_tensor.flatten(), dtype=wp.int32)
        velocities_vec3_array = wp.from_torch(vertex_velocities, dtype=wp.vec3)

        wp_mesh = wp.Mesh(
            points=vertex_vec3_array,
            indices=faces_wp_int32_array,
            velocities=velocities_vec3_array,
        )

        self.warp_mesh_per_env.append(wp_mesh)
        self.warp_mesh_id_list.append(wp_mesh.id)
        vertex_iterator += n_verts

    self.CONST_WARP_MESH_ID_LIST = self.warp_mesh_id_list
    self.CONST_WARP_MESH_PER_ENV = self.warp_mesh_per_env
    self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR = self.global_vertex_to_asset_index_tensor
    self.VERTEX_MAPS_PER_ENV_ORIGINAL = self.vertex_maps_per_env_original

    self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] = self.CONST_WARP_MESH_ID_LIST
    self.global_tensor_dict["CONST_WARP_MESH_PER_ENV"] = self.CONST_WARP_MESH_PER_ENV
    self.global_tensor_dict["CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR"] = (
        self.CONST_GLOBAL_VERTEX_TO_ASSET_INDEX_TENSOR
    )
    self.global_tensor_dict["VERTEX_MAPS_PER_ENV_ORIGINAL"] = self.VERTEX_MAPS_PER_ENV_ORIGINAL

    self.unfolded_env_vec_root_tensor = self.global_tensor_dict[
        "unfolded_env_asset_state_tensor"
    ]
    return 1


def apply():
    """Install the patch. Idempotent; must run before the sim is built."""
    global _APPLIED
    if _APPLIED:
        return
    WarpEnv.prepare_for_simulation = fixed_prepare_for_simulation
    _APPLIED = True
    logger.warning(
        "Patched WarpEnv.prepare_for_simulation: warp BVHs are now built on real "
        "vertices instead of a zero-filled buffer (see env_manager/warp_bvh_patch.py)."
    )
