"""Obstacle clearance: distance from the drone to the nearest surface, any direction.

Gives d_obstacle for the p_prox reward term, as the exact distance to the closest point
on any triangle in the env's warp mesh -- a full sphere about the drone, not a cone.

WHY A FULL SPHERE. Minimum clearance during a pass happens ABEAM or slightly behind: the
measured crash geometry is a median 91 deg off the direction of travel with 51% already
past it. A frontal cone stops measuring exactly at the closest approach, which is the
moment worth charging for. The obvious objection -- that a penalty for something behind
the drone is unlearnable because the camera faces forward -- does not hold: the policy
does not have to perceive the obstacle at the instant of penalty, TD backs that value up
to the approach, where it was visible.

WHY mesh_query_point AND NOT RAYCASTING. Rays quantise: 256 directions is ~5 deg spacing,
which resolves a 0.18 m feature at 2 m and under-detects thinner geometry -- and thin
geometry (tree branches, 26 per tree) is the case that motivated exact clearance in the
first place. mesh_query_point returns the true closest point on any triangle, in one
query per env instead of 256 rays. _no_sign skips the winding-number work, since only the
distance is wanted, not inside/outside.

THE FLOOR IS INCLUDED and treated like any other obstacle -- it is a real collision
surface. Consequence when tuning: d_obstacle is then min(altitude, nearest obstacle), so
at low altitude it simply reports the height above ground, and a large d_ref turns p_prox
partly into an altitude tax. Measure the distribution before fixing d_ref.
"""

import torch
import warp as wp


@wp.kernel
def _nearest_surface(
    mesh_ids: wp.array(dtype=wp.uint64),
    points: wp.array(dtype=wp.vec3),
    max_dist: float,
    out: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    mid = mesh_ids[e]
    if mid == wp.uint64(0):
        return
    p = points[e]
    q = wp.mesh_query_point_no_sign(mid, p, max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mid, q.face, q.u, q.v)
        out[e] = wp.length(cp - p)


class ProximityProbe:
    """One exact nearest-surface query per env per step."""

    def __init__(self, num_envs, device, max_dist=6.0):
        self.num_envs = num_envs
        self.device = device
        self.max_dist = max_dist
        self._out = torch.empty(num_envs, dtype=torch.float32, device=device)

    def measure(self, positions, mesh_ids_array):
        """positions (E,3) world. Returns (E,) metres to the nearest surface, clipped at
        max_dist where nothing was found within range."""
        self._out.fill_(self.max_dist)
        wp.launch(
            kernel=_nearest_surface,
            dim=self.num_envs,
            inputs=[
                mesh_ids_array,
                wp.from_torch(positions.contiguous(), dtype=wp.vec3),
                float(self.max_dist),
            ],
            outputs=[wp.from_torch(self._out, dtype=wp.float32)],
            device=str(self.device),
        )
        return self._out


def find_mesh_ids(sim_env):
    """The per-env wp.array of mesh ids the warp sensors render against."""
    from env_manager import warp_bvh_rebuild_patch as rp
    holders = rp._HOLDERS or rp._find_mesh_id_holders(sim_env)
    for h in holders:
        arr = getattr(h, "mesh_ids_array", None)
        if arr is not None:
            return arr
    return None
