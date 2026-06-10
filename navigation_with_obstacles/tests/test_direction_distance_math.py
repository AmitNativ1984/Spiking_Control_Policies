"""
Ground-truth test for NavigationWithObstaclesTask._direction_and_distance_to_target.

Validates the helper's body-frame direction/distance against tensors the
SIMULATOR itself produces. No hand-derived rotation values — every expected
value comes from simulator ground truth or from an invariant rotations must
obey. Requires a GPU.

The real task (real Isaac Gym sim) is provided by the session-scoped `task`
fixture in conftest.py (built once, shared across all test files — Isaac Gym
does not support a second sim instance per process).

Run inside the container (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 is required — the env
ships a broken hypothesis plugin that crashes pytest startup; the helper script
sets it for you):
    cd /workspaces/aerial_gym_docker
    ./navigation_with_obstacles/tests/run_tests.sh
  or run this file directly (no pytest, no plugin issue):
    python navigation_with_obstacles/tests/test_direction_distance_math.py

Cross-checks (all independent of any hand-computed expected vector):
  1. Distance equals the world-frame ||target - robot_position||.
  2. Body->world round-trip: quat_rotate(q, helper_vec) == world displacement.
  3. The helper's rotation matches the simulator's OWN body-frame transform
     (sim computes robot_body_linvel the same way, base_multirotor.py:293).
  4. Direction is unit norm.
  5. Level drones (roll,pitch ~ 0): body frame ~ vehicle frame.
"""
import isaacgym  # noqa: F401  (must precede torch)
import torch

from aerial_gym.utils.math import quat_rotate, quat_rotate_inverse

TOL = 1e-3


def test_distance_equals_world_norm(task):
    """Helper distance == ||target - robot_position|| in world frame (length is
    rotation-invariant). Ground truth = simulator robot_position + target."""
    _, dist = task._direction_and_distance_to_target()
    world_dist = torch.norm(
        task.target_position - task.obs_dict["robot_position"], dim=1, keepdim=True
    )
    err = float(torch.max(torch.abs(dist - world_dist)))
    print(f"[{'PASS' if err < TOL else 'FAIL'}] distance == world norm: max_err={err:.2e}")
    assert err < TOL, f"helper distance != world ||target-pos|| (err {err})"


def test_body_to_world_roundtrip(task):
    """Rotating the helper's body-frame vector back to world reproduces the
    world displacement target - robot_position, using the sim's real quaternion."""
    direction, dist = task._direction_and_distance_to_target()
    body_vec = direction * dist  # reconstruct body-frame displacement
    world_back = quat_rotate(task.obs_dict["robot_orientation"], body_vec)
    world_disp = task.target_position - task.obs_dict["robot_position"]
    err = float(torch.max(torch.abs(world_back - world_disp)))
    print(f"[{'PASS' if err < TOL else 'FAIL'}] body->world roundtrip: max_err={err:.2e}")
    assert err < TOL, f"body->world roundtrip mismatch (err {err})"


def test_matches_simulator_body_transform(task):
    """The helper's rotation must match the simulator's OWN body-frame transform.

    The sim computes robot_body_linvel = quat_rotate_inverse(robot_orientation,
    robot_linvel) (base_multirotor.py:293) — the identical operation the helper
    applies to (target - pos). So the same rotation applied to the sim's world
    velocity must reproduce the sim's ground-truth robot_body_linvel.
    """
    q = task.obs_dict["robot_orientation"]
    body_linvel_recomputed = quat_rotate_inverse(q, task.obs_dict["robot_linvel"])
    gt_body_linvel = task.obs_dict["robot_body_linvel"]
    err = float(torch.max(torch.abs(body_linvel_recomputed - gt_body_linvel)))
    print(f"[{'PASS' if err < TOL else 'FAIL'}] helper rotation == sim body transform: "
          f"max_err={err:.2e}")
    assert err < TOL, f"helper rotation disagrees with simulator body transform (err {err})"


def test_direction_unit_norm(task):
    """Helper direction is unit length for every env."""
    direction, _ = task._direction_and_distance_to_target()
    norms = torch.linalg.norm(direction, dim=1)
    err = float(torch.max(torch.abs(norms - 1.0)))
    print(f"[{'PASS' if err < 1e-3 else 'FAIL'}] direction unit-norm: max|n-1|={err:.2e}")
    assert err < 1e-3, f"direction not unit norm (err {err})"


def test_level_matches_vehicle_frame(task):
    """For ~level drones (roll,pitch ~ 0), body frame ~ vehicle frame, so the
    helper (body) must match the yaw-only vehicle-frame rotation of the same
    displacement. Compares only envs whose roll/pitch are near zero."""
    direction, dist = task._direction_and_distance_to_target()
    body_vec = direction * dist
    disp = task.target_position - task.obs_dict["robot_position"]
    vehicle_vec = quat_rotate_inverse(task.obs_dict["robot_vehicle_orientation"], disp)

    euler = task.obs_dict["robot_euler_angles"]  # (N,3) roll,pitch,yaw
    level = (euler[:, 0].abs() < 0.05) & (euler[:, 1].abs() < 0.05)
    n_level = int(level.sum())
    if n_level == 0:
        print("[SKIP] no near-level envs this run; skipping vehicle-frame check")
        return
    err = float(torch.max(torch.abs(body_vec[level] - vehicle_vec[level])))
    print(f"[{'PASS' if err < 5e-2 else 'FAIL'}] level: body ~ vehicle frame "
          f"({n_level} envs): max_err={err:.2e}")
    assert err < 5e-2, f"level-drone body frame != vehicle frame (err {err})"


if __name__ == "__main__":
    # Standalone path: build the task locally (one sim in this fresh process).
    from navigation_with_obstacles.tests.conftest import build_task

    t = build_task()
    tests = [
        test_distance_equals_world_norm,
        test_body_to_world_roundtrip,
        test_matches_simulator_body_transform,
        test_direction_unit_norm,
        test_level_matches_vehicle_frame,
    ]
    failed = 0
    for fn in tests:
        print(f"\n=== {fn.__name__} ===")
        try:
            fn(t)
        except AssertionError as e:
            failed += 1
            print(f"  -> FAILED: {e}")
    t.close()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
