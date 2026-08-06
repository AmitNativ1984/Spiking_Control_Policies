"""End-to-end smoke test: the real task, on the GPU, for a handful of steps.

Guards the things an import check cannot: that the task emits observations of the width
its config advertises, that nothing goes non-finite, and that the aliasing rule holds
(obs_dict is a live view onto simulator tensors, not a copy).
"""
import torch

from aerial_gym.registry.robot_registry import robot_registry


def test_observation_matches_configured_dim(task, task_config, zero_actions, num_envs):
    obs, *_ = task.step(zero_actions)
    assert obs["observations"].shape == (num_envs, task_config.observation_space_dim)


def test_step_returns_finite_values(task, zero_actions, num_envs):
    obs, rewards, terminated, truncated, _ = task.step(zero_actions)

    assert torch.isfinite(obs["observations"]).all(), "non-finite observation"
    assert torch.isfinite(rewards).all(), "non-finite reward"
    assert rewards.shape == (num_envs,)
    assert terminated.shape == (num_envs,)
    assert truncated.shape == (num_envs,)


def test_rewards_vary_across_envs(task, zero_actions):
    """A reward identical in every env means the signal isn't reaching the agent —
    the plumbing looks fine and the policy learns nothing."""
    _, rewards, *_ = task.step(zero_actions)
    assert rewards.std() > 0, "reward is constant across envs"


def test_return_state_before_reset_branch_runs(task, task_config, zero_actions, num_envs):
    """The knob is declared in the config, so both branches must execute. Default False
    (the agent gets the first observation of the new episode, which is what PPO wants);
    True returns the terminal observation instead.

    Smoke-level: this proves the True branch is live and well-formed. It does not assert
    the terminal-observation semantics, which would need a step where a reset is
    guaranteed to fire.
    """
    original = task_config.return_state_before_reset
    try:
        task_config.return_state_before_reset = True
        obs, rewards, terminated, truncated, infos = task.step(zero_actions)

        assert obs["observations"].shape == (num_envs, task_config.observation_space_dim)
        assert torch.isfinite(obs["observations"]).all()
        assert torch.isfinite(rewards).all()
    finally:
        task_config.return_state_before_reset = original


def test_obs_dict_is_a_live_view(task, zero_actions):
    """env.get_obs() returns the live global_tensor_dict. Anything a reward needs to
    survive a step must be cloned; this test documents the aliasing so a future change
    to copy-on-get doesn't go unnoticed."""
    before = task.obs_dict["robot_position"]
    snapshot = before.clone()

    task.step(zero_actions)

    assert task.obs_dict["robot_position"] is before, \
        "obs_dict entry was replaced, not updated in place"
    assert not torch.allclose(before, snapshot), \
        "robot_position did not advance over a step"


def test_gyro_channel_comes_from_the_imu_not_ground_truth(task, zero_actions):
    """obs[7:10] must carry the simulated IMU gyro, not robot_body_angvel.

    The point of routing the IMU in is to DEGRADE the observation to what the real F450
    can supply. If someone reverts the swap in process_obs_for_task, the two tensors go
    bit-identical and the degradation silently disappears — nothing else would fail.
    """
    obs, *_ = task.step(zero_actions)

    if task._imu_gyro_accum is None:
        import pytest
        pytest.skip("enable_imu is False in the robot config")

    gyro_obs = obs["observations"][:, 7:10]
    ground_truth = task.obs_dict["robot_body_angvel"]

    assert not torch.allclose(gyro_obs, ground_truth), \
        "obs[7:10] is bit-identical to ground truth — the IMU swap is not active"


def test_imu_bias_is_a_held_turn_on_offset_matching_the_sdf(task, zero_actions):
    """The IMU bias must be a FIXED turn-on offset of the SDF's magnitude, not a drift.

    gz-sensors draws the bias once at load and holds it (the F450 SDF sets no
    dynamic_bias_stddev), so Aerial Gym's random walk is switched off via bias_std = 0 and
    the magnitude lives in max_bias_init_value. Both halves are easy to undo by accident --
    restoring base_imu_config's VN100 defaults would reintroduce a 65x-too-large accel bias
    plus an unmodelled walk, and nothing else would notice.
    """
    imu = getattr(task.sim_env.robot_manager, "imu_sensor", None)
    if imu is None:
        import pytest
        pytest.skip("enable_imu is False in the robot config")

    cfg = task.sim_env.robot_manager.cfg.sensor_config.imu_config
    bound = torch.tensor(cfg.max_bias_init_value, device=task.device)

    assert not any(cfg.bias_std), \
        "bias_std is non-zero — the bias now drifts, which the F450 SDF does not model"

    start = imu.bias.clone()
    assert bool((start.abs() <= bound + 1e-12).all()), \
        f"initial bias exceeds max_bias_init_value: {start.abs().max(dim=0).values.tolist()}"

    for _ in range(5):
        task.step(zero_actions)

    assert torch.equal(imu.bias, start), \
        "IMU bias changed during an episode — it should be a held turn-on offset"


def test_gyro_channel_is_the_gyro_and_not_the_accelerometer(task, zero_actions):
    """imu_meas is (num_envs, 6) = [0:3] accel, [3:6] gyro. Slicing [0:3] by mistake is a
    plausible edit, and it would not crash: the shapes match. It would however feed the
    policy a specific-force reading (~9.8 m/s^2 at hover) where an angular rate belongs.

    So bound the deviation from ground truth. Sensor noise is small; a wrong slice is not.
    """
    obs, *_ = task.step(zero_actions)

    if task._imu_gyro_accum is None:
        import pytest
        pytest.skip("enable_imu is False in the robot config")

    deviation = (obs["observations"][:, 7:10] - task.obs_dict["robot_body_angvel"]).abs().max()

    assert torch.isfinite(deviation), "gyro channel went non-finite"
    assert deviation < 1.0, (
        f"obs[7:10] deviates from true angular velocity by {deviation:.3f} rad/s — far "
        "beyond IMU noise. Most likely the accel slice [0:3] is being read instead of "
        "the gyro slice [3:6]."
    )


def test_inertia_is_randomized_per_env_and_hidden_from_the_controller(task, num_envs):
    """Inertia DR must reach PhysX while staying INVISIBLE to the controller.

    That asymmetry is the whole reason inertia is the one rigid-body property still
    randomized here. robot_manager copies a single build-time inertia to every env, so a
    per-env PhysX draw is a real plant/model mismatch the policy has to absorb. If someone
    "fixes" the discrepancy by publishing the per-env draw into global_tensor_dict, the
    controller starts compensating for it exactly and the randomization stops buying
    anything — the same trap the mass draw fell into.
    """
    if task._mass_dr is None:
        import pytest
        pytest.skip("randomize_mass_properties is off")

    gym = task.sim_env.robot_manager.gym
    physx_iyy = torch.tensor(
        [gym.get_actor_rigid_body_properties(e, a)[0].inertia.y.y
         for e, a in zip(task.sim_env.IGE_env.env_handles,
                         task.sim_env.robot_manager.robot_handles)],
        device=task.device,
    )
    assert physx_iyy.std() > 0, "PhysX inertias identical across envs — DR did not apply"

    published = task.obs_dict["robot_inertia"]
    assert published.std(dim=0).max() == 0, (
        "robot_inertia varies per env — the controller is being told about the inertia "
        "draw, which cancels the mismatch it was supposed to create"
    )


def test_mass_is_not_randomized_and_the_controller_knows_the_true_mass(task, num_envs):
    """Mass DR was removed on purpose; this guards both halves of that decision.

    The controller maps thrust as (a+1)*mass*g through a VIEW onto
    global_tensor_dict["robot_mass"]. Randomizing the physics mass and publishing it there
    hands the controller the answer, so hover stays at command 0 for any mass and the draw
    buys no robustness (a forced 2x draw left the drone still hovering). Randomizing it
    WITHOUT publishing would instead leave every env hovering at the wrong throttle.
    Neither is wanted: mass stays nominal, and robot_mass must keep matching PhysX.
    """
    gym = task.sim_env.robot_manager.gym
    physx = torch.tensor(
        [sum(b.mass for b in gym.get_actor_rigid_body_properties(e, a))
         for e, a in zip(task.sim_env.IGE_env.env_handles,
                         task.sim_env.robot_manager.robot_handles)],
        device=task.device,
    )
    published = task.obs_dict["robot_mass"]

    assert physx.std() == 0, "PhysX masses vary across envs — mass randomization is back"
    assert torch.allclose(physx, published, atol=1e-4), \
        "global_tensor_dict['robot_mass'] disagrees with PhysX — controller would use the wrong mass"

    ctrl = task.sim_env.robot_manager.robot.controller
    assert torch.allclose(ctrl.mass.squeeze(1), published), \
        "controller.mass is no longer a view onto robot_mass"


def test_observation_uses_estimated_state_but_reward_uses_truth(task, zero_actions):
    """The whole point of the estimator noise: the policy SEES drift, the reward does not.

    If someone routes the noisy state into _direction_and_distance_to_target(), the agent
    starts being paid for reaching a hallucinated target and this test fails.
    """
    if not task.task_config.state_estimation_noise.enable:
        import pytest
        pytest.skip("state_estimation_noise disabled")

    task.step(zero_actions)

    true_dir, true_dist = task._direction_and_distance_to_target()
    obs_dir = task.task_obs["observations"][:, 0:3]

    assert not torch.allclose(obs_dir, true_dir), \
        "obs[0:3] equals the true bearing — estimator noise is not reaching the observation"

    # The reward helper must stay deterministic: two calls, no bias advance, same answer.
    again_dir, again_dist = task._direction_and_distance_to_target()
    assert torch.allclose(true_dir, again_dir) and torch.allclose(true_dist, again_dist), \
        "_direction_and_distance_to_target() is not pure — reward path has been contaminated"


def test_estimator_error_is_bounded(task, zero_actions):
    """Drift must stay in a plausible EKF range. A runaway random walk (missing per-episode
    resample, or sqrt(dt) dropped) would show up here long before it showed up in a reward
    curve."""
    if not task.task_config.state_estimation_noise.enable:
        import pytest
        pytest.skip("state_estimation_noise disabled")

    for _ in range(20):
        task.step(zero_actions)

    bias = task._est_pos_bias.abs().max()
    assert torch.isfinite(bias), "position bias went non-finite"
    assert bias < 2.0, f"position bias reached {bias:.2f} m — random walk is not bounded"


def test_inertia_draw_is_within_band_and_resamples_from_nominal(task, num_envs):
    """The build-time inertia draw must sit in the configured band and come from NOMINALS.

    Scaling the live values instead of the stored nominals compounds: repeated draws walk
    the airframe away permanently and nothing flags it. Calling the draw repeatedly here is
    safe only because it stages no reset -- see the spawn test below for why
    _randomize_mass_properties must never run mid-episode.
    """
    if task._mass_dr is None:
        import pytest
        pytest.skip("randomize_mass_properties is off")

    gym = task.sim_env.robot_manager.gym
    envs = task.sim_env.IGE_env.env_handles
    actors = task.sim_env.robot_manager.robot_handles
    all_ids = torch.arange(num_envs, device=task.device)

    nom_iyy = task._nominal_inertia[1]
    lo, hi = task._mass_dr.inertia_scale_range
    band = (nom_iyy * lo - 1e-9, nom_iyy * hi + 1e-9)

    seen = []
    for _ in range(15):
        task._randomize_mass_properties(all_ids)
        iyy = torch.tensor(
            [gym.get_actor_rigid_body_properties(e, a)[0].inertia.y.y
             for e, a in zip(envs, actors)],
            device=task.device,
        )
        assert iyy.min() >= band[0] and iyy.max() <= band[1], (
            f"inertia {iyy.min():.6f}-{iyy.max():.6f} escaped band "
            f"{band[0]:.6f}-{band[1]:.6f} — scaling is compounding instead of "
            "resampling from nominal"
        )
        seen.append(iyy)

    stacked = torch.stack(seen)
    assert stacked.std(dim=0).min() > 0, "inertia never changed across draws — not resampling"


def test_reset_idx_does_not_touch_rigid_body_properties(task, num_envs):
    """reset_idx() must NOT resample rigid-body properties.

    gym.set_actor_rigid_body_properties is a CPU actor API; under the GPU pipeline calling
    it mid-episode discards the root states the reset just staged, so the robot silently
    reverts to its creation pose at the world origin instead of the sampled spawn. Guards
    the schedule rather than the symptom, so it fails fast and for the right reason.
    """
    if task._mass_dr is None:
        import pytest
        pytest.skip("randomize_mass_properties is off")

    gym = task.sim_env.robot_manager.gym
    envs = task.sim_env.IGE_env.env_handles
    actors = task.sim_env.robot_manager.robot_handles

    before = [gym.get_actor_rigid_body_properties(e, a)[0].inertia.y.y
              for e, a in zip(envs, actors)]
    task.reset_idx(torch.arange(num_envs, device=task.device))
    after = [gym.get_actor_rigid_body_properties(e, a)[0].inertia.y.y
             for e, a in zip(envs, actors)]

    assert before == after, (
        "reset_idx changed rigid-body properties — that discards the staged spawn and the "
        "robot will start every episode at the world origin"
    )


def test_spawn_survives_the_step_after_reset(task, num_envs):
    """The sampled spawn must still be there one step later.

    The end-to-end regression for the bug above: the reset writes the spawn into the root
    state tensor, but if anything invalidates the staged states before the next simulate(),
    the robot snaps back to the origin and every episode starts from the same place.
    """
    task.sim_env.sim_steps[:] = task.task_config.episode_len_steps + 1  # force a reset
    task.step(torch.zeros((num_envs, 4), device=task.device))
    spawn = task.obs_dict["robot_position"].clone()

    task.step(torch.zeros((num_envs, 4), device=task.device))
    moved = (task.obs_dict["robot_position"] - spawn).norm(dim=1)

    # One env step of near-hover flight moves centimetres, not metres.
    assert float(moved.max()) < 0.5, (
        f"robot moved {float(moved.max()):.2f} m in one step after reset — the spawn was "
        "discarded and the robot reverted to its creation pose"
    )


def test_first_episode_spawns_inside_the_spawn_box(task):
    """Episode 1 must spawn like every other episode.

    The task's own _setup_domain_randomization() writes rigid-body properties during
    __init__, which discards the root states staged for the next simulate(). Without the
    warm-up step that follows it, the first episode starts at the world origin in EVERY env
    while every later episode is fine. That asymmetry is what made it survive the earlier
    spawn fix. Not an upstream issue: with randomize_mass_properties off, the pre-fix code
    spawned correctly with no warm-up at all.

    Measured on the fixture's own first episode (recorded in conftest, because only one sim
    may exist per process and the fixture has stepped on by the time tests run).
    """
    spawn = task.first_episode_spawn
    after = task.first_episode_after_one_step

    moved = (after - spawn).norm(dim=1)
    assert float(moved.max()) < 0.5, (
        f"robot moved {float(moved.max()):.2f} m on the first step of the run — the first "
        "episode's spawn was discarded (missing warm-up simulate before the initial reset)"
    )

    # And the spawn must actually be in the configured box, not at the origin by luck.
    lo, hi = task.first_episode_bounds
    ratio = (spawn - lo) / (hi - lo)
    cfg = robot_registry.get_robot_config(task.task_config.robot_name).init_config
    lo_r = torch.tensor(cfg.min_init_state[0:3], device=task.device) - 1e-3
    hi_r = torch.tensor(cfg.max_init_state[0:3], device=task.device) + 1e-3
    assert bool(((ratio >= lo_r) & (ratio <= hi_r)).all()), (
        f"first-episode spawn ratios {ratio.tolist()} fall outside init_config's box "
        f"{lo_r.tolist()}..{hi_r.tolist()}"
    )
