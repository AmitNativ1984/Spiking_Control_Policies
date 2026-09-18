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

*** BLOCKED ON WARP 1.0.0 -- DO NOT WIRE THIS INTO A REWARD YET ***

mesh_query_point_no_sign FAULTS with CUDA error 700 (illegal memory access) on this env's
populated warp meshes. Measured on an A100, level 30, 16-128 envs, Warp 1.0.0:

    level 30, radius 1.5 / 2.0 / 2.5 m   -> fault, immediately, at the first launch
    level 15, radius 2.0 m               -> fault, immediately
    level  0, radius 1.5 / 2.0 m         -> CLEAN over 200 steps x 128 envs

What was ruled out along the way: the kernel and the Warp 1.0.0 signature are correct (the
same kernel point-queries a mesh built in-script and returns the exact answer); the CUDA
context is clean before the probe runs; the vertex data is finite and the triangle indices
are in range; the mesh ids match the live wp.Mesh objects; buffer lifetime is not involved
(a persistent contiguous points buffer faults identically); and it is not env-specific --
15 of 16 envs answer correctly in one batched launch, and which env fails moves with the
search radius.

What remains is mesh COMPLEXITY. Each per-env mesh here is 37,156 points / 73,220
triangles (the whole preallocated asset pool, with culled assets parked ~1000 m away), and
Warp's closest-point traversal keeps pending BVH nodes on a fixed-size stack. Level 0 --
floor and walls only -- is the only configuration that survives, at any radius. Ray
queries over the SAME meshes are unaffected, which is why the depth camera has always
worked: a ray keeps far fewer nodes pending.

So exact closest-point distance is not available on this stack. The alternatives, in the
order they were considered: a ray-sphere probe (works today, quantises -- N rays over a
sphere give ~sqrt(4*pi/N) rad spacing, so 2048 rays is ~4.5 deg and resolves 0.16 m at
2 m, and it can only OVER-estimate clearance, which is the unsafe direction); a depth-image
reduction (already implemented in analysis/measure_freespace.py, but in-FOV only); or
upgrading Warp in the container image, which fixes the root cause and touches every run.

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
