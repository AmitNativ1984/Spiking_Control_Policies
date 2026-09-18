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


@wp.kernel
def _ray_min_dist(
    mesh_ids: wp.array(dtype=wp.uint64),
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    max_dist: float,
    out: wp.array(dtype=wp.float32),
):
    """One thread per (env, direction). Reduces with an atomic min, so the ray count can
    grow without the kernel holding a per-ray result array."""
    e, i = wp.tid()
    mid = mesh_ids[e]
    if mid == wp.uint64(0):
        return
    q = wp.mesh_query_ray(mid, origins[e], dirs[i], max_dist)
    if q.result:
        wp.atomic_min(out, e, q.t)


def _fibonacci_sphere(n):
    """n near-uniformly spaced unit directions. Deterministic, so every env and every step
    samples the SAME directions -- a clearance signal that jittered its own sample set
    would inject noise the policy cannot distinguish from real geometry."""
    import math

    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = golden * i
        out.append((r * math.cos(th), r * math.sin(th), z))
    return out


class RaySphereProbe:
    """Nearest surface in any direction, by ray casting instead of a closest-point query.

    WHY THIS EXISTS. ProximityProbe above is the exact answer and it faults under Warp
    1.0.0 (see the module docstring). Ray queries over the SAME meshes are unaffected --
    the depth camera casts 320 x 180 = 57,600 of them per env per step -- so this trades
    exactness for a query that runs.

    HOW ACCURATE. Two different error modes, and the second is the one that sets the ray
    count:
      * A LARGE surface is measured almost exactly. If the closest ray misses the true
        nearest point by angle a, a locally flat surface reads d/cos(a), an overestimate of
        about d*a^2/2 -- at 2048 rays (a ~ 2.2 deg) that is 1.5 mm at 2 m.
      * THIN geometry can be missed entirely. A cylinder of radius r at distance d is only
        caught if some ray passes within its angular radius r/d, so the spacing has to beat
        2r/d. n directions give a spacing of about sqrt(4*pi/n) rad, hence:
            n =  2048 -> 4.5 deg -> catches r >= 7.8 cm at 2 m
            n =  8192 -> 2.2 deg -> catches r >= 3.9 cm at 2 m
            n = 16384 -> 1.6 deg -> catches r >= 2.8 cm at 2 m
            n = 32768 -> 1.1 deg -> catches r >= 2.0 cm at 2 m
        Tree branches -- 26 per tree, and the case that argued for an exact query in the
        first place -- are the reason the default is 16384 rather than the 2048 that would
        already be enough for walls and trunks. 16384 rays is 28% of the ray count the
        depth camera already casts per env per step, so this is cheap headroom, not a
        stretch.

    THE ERROR IS SIGNED, AND IT IS SIGNED THE WRONG WAY. Missing geometry between rays can
    only make the measured clearance LARGER than the truth, never smaller, and a barrier
    fed an over-estimate permits a higher closing speed than it should.

    MEASURED (analysis/validate_ray_probe.py, A100, 16384 rays, 128 envs):
      * vs the EXACT query at level 0, the one level where the exact query survives:
        error +0.0001 m at every quantile, never negative. The machinery and the sign are
        right, though level 0 only exercises large flat surfaces.
      * no faults over 100 steps x 128 envs at level 30 with a 6 m range -- the thing the
        exact query could not do.
      * cost 19.3 ms/step at 128 envs. Rays here are maximally INCOHERENT (every ray a
        different direction), unlike a camera's neighbouring pixels, so they do not cost
        what camera rays cost -- budget by measurement, not by ray count.
      * vs the rendered DEPTH IMAGE at level 30: median -0.36 m, which is the right sign
        (a sphere sees closer things than an 87x56 deg cone), but 27% of samples read more
        than 0.10 m ABOVE the image minimum, p99 +0.56 m. That is this probe missing
        geometry the camera resolves, and the reason is angular density: the image packs
        57,600 samples into ~1.32 sr (~43,600/sr) while 16,384 rays spread over 4*pi
        (~1,300/sr), so the camera is ~33x denser inside its cone. Matching it everywhere
        would take ~550k rays.

    WHY NOT JUST TAKE min(rays, depth image) AND GET THAT DENSITY FREE. Because of when
    the image is rendered. The sensors are re-rendered in post_reward_calculation_step(),
    AFTER the reward, so the image on hand is aligned with h(x_t) but one step stale for
    h(x_{t+1}) -- it is in fact the SAME image. It would then enter both terms identically
    and cancel out of h_t - h_{t+1}, so the barrier would read zero closing speed on every
    step where the image was the binding term. The probe has to carry both time points by
    itself.

    SO THE HONEST POSITION: clearance is biased optimistic by order 0.1-0.3 m near thin
    geometry. For a shaping term that is tolerable noise on top of an alpha already chosen
    conservatively; for a hard safety filter it would not be. Do not promote this to a
    filter without fixing the estimator.
    """

    def __init__(self, num_envs, device, max_dist=6.0, num_rays=16384):
        self.num_envs = num_envs
        self.device = device
        self.max_dist = max_dist
        self.num_rays = num_rays
        self._out = torch.empty(num_envs, dtype=torch.float32, device=device)
        self._out_wp = wp.from_torch(self._out, dtype=wp.float32)
        # Directions are built ONCE. They live in the world frame and the sphere is
        # isotropic, so no per-step rotation into the body frame is needed or wanted.
        d = torch.tensor(_fibonacci_sphere(num_rays), dtype=torch.float32, device=device)
        self._dirs = d.contiguous()
        self._dirs_wp = wp.from_torch(self._dirs, dtype=wp.vec3)
        # Persistent, contiguous origins: obs_dict["robot_position"] is a non-contiguous
        # slice of the root-state tensor, and a temporary from .contiguous() would hand
        # warp memory with no owner.
        self._pts = torch.empty((num_envs, 3), dtype=torch.float32, device=device)
        self._pts_wp = wp.from_torch(self._pts, dtype=wp.vec3)

    def measure(self, positions, mesh_ids_array):
        """positions (E,3) world. Returns (E,) metres to the nearest surface in ANY
        direction, clipped at max_dist where no ray hit anything."""
        self._pts.copy_(positions)
        self._out.fill_(self.max_dist)
        wp.launch(
            kernel=_ray_min_dist,
            dim=(self.num_envs, self.num_rays),
            inputs=[
                mesh_ids_array,
                self._pts_wp,
                self._dirs_wp,
                float(self.max_dist),
            ],
            outputs=[self._out_wp],
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
