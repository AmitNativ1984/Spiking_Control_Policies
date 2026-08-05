"""End-to-end smoke test: the real task, on the GPU, for a handful of steps.

Guards the things an import check cannot: that the task emits observations of the width
its config advertises, that nothing goes non-finite, and that the aliasing rule holds
(obs_dict is a live view onto simulator tensors, not a copy).
"""
import torch


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
