"""
Integration test: the helper is the SINGLE SOURCE OF TRUTH for both the
observation vector and the heading reward, and the reward reads CURRENT-step
(non-stale) state. Built on a REAL, fully-stepped Isaac Gym sim provided by the
session-scoped `task` fixture (conftest.py) — robot orientation/position/
velocity come from the simulator (ground truth); we only control
target_position, which the task legitimately owns.

REQUIRES A GPU. Run inside the container (PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 is
required — the env ships a broken hypothesis plugin that crashes pytest startup;
the helper script sets it for you):
    cd /workspaces/aerial_gym_docker
    ./navigation_with_obstacles/tests/run_tests.sh
  or run this file directly (no pytest, no plugin issue):
    python navigation_with_obstacles/tests/test_navigation_task_helper.py

Checks:
  1. observations[:,0:3] == helper direction; observations[:,3] == helper
     distance normalized by env diagonal and clamped (process_obs_for_task uses
     the helper).
  2. The velocity the reward uses (obs_dict["robot_body_linvel"]) equals what
     process_obs_for_task writes to observations[:,4:7] — same current-step
     source, no duplication.
  3. Stale-bug regression: after the robot moves (a sim step), the helper (used
     by _reward_progress) reflects the new state while a pre-step buffer snapshot
     does not.
"""
import isaacgym  # noqa: F401  (must precede torch)
import torch

TOL = 1e-3


def test_observations_use_helper(task):
    """observations[:,0:3] and [:,3] are exactly what the helper produces."""
    direction, dist = task._direction_and_distance_to_target()
    task.process_obs_for_task()
    obs = task.task_obs["observations"]

    dir_err = float(torch.max(torch.abs(obs[:, 0:3] - direction)))

    max_dist = torch.norm(
        task.obs_dict["env_bounds_max"] - task.obs_dict["env_bounds_min"], dim=-1
    )
    exp_dist_obs = torch.clamp(dist.squeeze(1) / (max_dist + 1e-6), 0.0, 1.0)
    dist_err = float(torch.max(torch.abs(obs[:, 3] - exp_dist_obs)))

    print(f"[{'PASS' if dir_err < TOL else 'FAIL'}] obs[:,0:3] == helper dir: max_err={dir_err:.2e}")
    print(f"[{'PASS' if dist_err < TOL else 'FAIL'}] obs[:,3] == norm/clamp dist: max_err={dist_err:.2e}")
    assert dir_err < TOL, f"observations[:,0:3] != helper direction (err {dir_err})"
    assert dist_err < TOL, f"observations[:,3] != normalized helper distance (err {dist_err})"


def test_reward_velocity_source_matches_obs(task):
    """The velocity the reward uses (obs_dict['robot_body_linvel']) is the same
    current-step tensor process_obs_for_task writes to observations[:,4:7]."""
    task.process_obs_for_task()
    obs_v = task.task_obs["observations"][:, 4:7]
    reward_v = task.obs_dict["robot_body_linvel"]
    err = float(torch.max(torch.abs(obs_v - reward_v)))
    print(f"[{'PASS' if err < TOL else 'FAIL'}] reward v == obs[:,4:7]: max_err={err:.2e}")
    assert err < TOL, f"reward velocity source != observation velocity (err {err})"


def test_stale_bug_regression(task):
    """After the robot moves, the helper (reward path) tracks the new state; a
    pre-step buffer snapshot does not.

    This is the bug we fixed: _reward_progress used to read n/v from the buffer,
    which is only refreshed AFTER compute_rewards in step() -> one step stale.
    """
    # Snapshot a freshly-built buffer and the helper output at the same instant.
    task.process_obs_for_task()
    buffer_dir_before = task.task_obs["observations"][:, 0:3].clone()
    helper_dir_before, _ = task._direction_and_distance_to_target()
    sync_err = float(torch.max(torch.abs(buffer_dir_before - helper_dir_before)))
    print(f"[{'PASS' if sync_err < TOL else 'FAIL'}] buffer == helper when synced: "
          f"max_err={sync_err:.2e}")
    assert sync_err < TOL, "buffer and helper disagree even when synced"

    # Move the robot (real sim step), then compare the current-step helper output
    # against the PRE-step buffer snapshot: it must have changed.
    n_envs = task.obs_dict["robot_position"].shape[0]
    task.step(torch.zeros((n_envs, 4), device=task.device))
    helper_dir_after, _ = task._direction_and_distance_to_target()

    drift = float(torch.max(torch.abs(helper_dir_after - buffer_dir_before)))
    print(f"[{'PASS' if drift > 1e-4 else 'FAIL'}] helper tracks current step "
          f"(drift vs pre-step buffer={drift:.2e}, expect > 0)")
    assert drift > 1e-4, "helper did not change after the robot moved (stale!)"


if __name__ == "__main__":
    # Standalone path: build the task locally (one sim in this fresh process).
    from navigation_with_obstacles.tests.conftest import build_task

    t = build_task()
    tests = [
        test_observations_use_helper,
        test_reward_velocity_source_matches_obs,
        test_stale_bug_regression,
    ]
    failed = 0
    for fn in tests:
        print(f"\n=== {fn.__name__} ===")
        try:
            fn(t)
        except AssertionError as e:
            failed += 1
            print(f"  -> FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  -> ERROR: {type(e).__name__}: {e}")
    t.close()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
