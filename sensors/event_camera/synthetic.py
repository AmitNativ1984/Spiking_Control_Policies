"""Synthetic warp scenes and camera poses for tests, the demo and the benchmark.

Builds the same per-env wp.Mesh layout aerial_gym's WarpEnv produces (one mesh per env,
velocities[:, 1] = asset index within the env), without needing Isaac Gym.
"""
import math

import numpy as np
import torch
import trimesh as tm
import warp as wp

# optical frame (x right, y down, z forward) -> world (z up) for a camera looking along +x
_R_OPTICAL_TO_WORLD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class SyntheticScene:
    """One wp.Mesh per env from lists of trimesh assets. Keeps the buffers warp references."""

    def __init__(self, env_assets, device="cuda:0"):
        self.meshes, self._buffers, ids = [], [], []
        for assets in env_assets:
            merged = tm.util.concatenate(assets)
            asset_idx = np.concatenate([np.full(len(a.vertices), i) for i, a in enumerate(assets)])
            points = torch.tensor(np.asarray(merged.vertices), dtype=torch.float32, device=device)
            faces = torch.tensor(np.asarray(merged.faces), dtype=torch.int32, device=device).flatten()
            vel = torch.zeros((len(points), 3), dtype=torch.float32, device=device)
            vel[:, 1] = torch.tensor(asset_idx, dtype=torch.float32, device=device)
            mesh = wp.Mesh(
                points=wp.from_torch(points, dtype=wp.vec3),
                indices=wp.from_torch(faces, dtype=wp.int32),
                velocities=wp.from_torch(vel, dtype=wp.vec3),
            )
            self._buffers.append((points, faces, vel))
            self.meshes.append(mesh)
            ids.append(mesh.id)
        self.mesh_ids_array = wp.array(ids, dtype=wp.uint64, device=device)


def quat_from_matrix(R):
    """xyzw quaternion of a 3x3 rotation matrix."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(x, R[2, 1] - R[1, 2])
    y = math.copysign(y, R[0, 2] - R[2, 0])
    z = math.copysign(z, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def camera_quat(yaw=0.0, pitch=0.0):
    """Optical-frame quaternion for a camera at heading `yaw` (about world z), level by default."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return quat_from_matrix(Rz @ Ry @ _R_OPTICAL_TO_WORLD)


def wall_x(x, y_lo, y_hi, z_lo, z_hi, thickness=0.05):
    """Axis-aligned box whose front face is the plane world-x = x, facing -x."""
    box = tm.creation.box(extents=[thickness, y_hi - y_lo, z_hi - z_lo])
    box.apply_translation([x + thickness / 2.0, (y_lo + y_hi) / 2.0, (z_lo + z_hi) / 2.0])
    return box


def random_forest(rng, num_obstacles, extent=20.0, height=6.0):
    """Ground + cylinders/spheres/boxes scattered ahead of the origin; asset 0 is the ground."""
    ground = tm.creation.box(extents=[4 * extent, 4 * extent, 0.1])
    ground.apply_translation([extent, 0.0, -0.05])
    assets = [ground]
    for _ in range(num_obstacles):
        x, y = rng.uniform(2.0, extent), rng.uniform(-extent / 2, extent / 2)
        kind = rng.integers(3)
        if kind == 0:
            r = rng.uniform(0.1, 0.5)
            m = tm.creation.cylinder(radius=r, height=height, sections=24)
            m.apply_translation([x, y, height / 2])
        elif kind == 1:
            r = rng.uniform(0.3, 1.0)
            m = tm.creation.icosphere(subdivisions=2, radius=r)
            m.apply_translation([x, y, rng.uniform(r, height - r)])
        else:
            e = rng.uniform(0.3, 1.5, size=3)
            m = tm.creation.box(extents=e)
            m.apply_translation([x, y, e[2] / 2])
        assets.append(m)
    return assets
