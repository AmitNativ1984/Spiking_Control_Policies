"""The deployed NAVIGATION observation must be the same function as the trained one.

The companion of test_deploy_obs_parity.py, for the 49-D navigation vector. Same reasoning:
control_policy_api reimplements the observation in numpy because the flight container has
no Isaac Gym, and a reimplementation is exactly where a deployment drifts from its training
silently -- shapes stay right, actions keep coming, and the actions are wrong.

The navigation vector gives that failure THREE more places to happen than hover's did, and
each one has its own test below:

  * velocity is in the VEHICLE frame here, the BODY frame there (they agree while level);
  * the target is a unit direction plus a separately-normalized scalar range;
  * prev_action carries the TRANSFORMED command in radians, not the raw [-1, 1] output.

The depth half is checked too, against the training encoder itself rather than against a
description of it -- see test_depth_preprocessing_matches_training.

Reference: task/attitude_navigation_task.py:1023-1088, with
`state_estimation_noise.enable = False`. The noise path is training-only: on a real vehicle
the estimator supplies its own error, and injecting more would be modelling the same thing
twice.
"""

import numpy as np
import pytest
import torch
from aerial_gym.utils.math import (
    quat_from_euler_xyz,
    quat_rotate_inverse,
    vehicle_frame_quat_from_quat,
)

control_policy_api = pytest.importorskip(
    "control_policy_api", reason="pip install -e <sail-uav-core>/libs/control-policy-api"
)
from control_policy_api.base import DroneState  # noqa: E402
from control_policy_api.depth import (  # noqa: E402
    MAX_DEPTH_M,
    MIN_DEPTH_M,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    preprocess_depth,
)
from control_policy_api.observations_nav import (  # noqa: E402
    DISTANCE_NORM_M,
    NAV_LATENT_DIM,
    NAV_OBS_DIM,
    build_nav_observation,
)

NUM_SAMPLES = 200
TOLERANCE = 1e-9

# The env box the policy trained in (config/env_config/env_forest_with_obstacles.py:68-71):
# 20 x 20 m in x/y, height randomized in [4, 6] m. Its diagonal is what the task divides
# the distance channel by.
TRAINING_EXTENT_MIN = np.array([20.0, 20.0, 4.0])
TRAINING_EXTENT_MAX = np.array([20.0, 20.0, 6.0])

MAX_TILT_RAD = np.pi / 4
MAX_YAW_RATE_RAD_S = np.pi / 3


def random_state(rng):
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    return {
        "position": rng.normal(scale=3.0, size=3),
        "velocity": rng.normal(scale=2.0, size=3),
        "quat": quat,
        "angvel": rng.normal(scale=1.0, size=3),
        "target": rng.normal(scale=8.0, size=3),
        # Already transformed: thrust in [-1, 1], roll/pitch in radians, yaw rate in rad/s.
        "prev_action": np.array(
            [
                rng.uniform(-1.0, 1.0),
                rng.uniform(-MAX_TILT_RAD, MAX_TILT_RAD),
                rng.uniform(-MAX_TILT_RAD, MAX_TILT_RAD),
                rng.uniform(-MAX_YAW_RATE_RAD_S, MAX_YAW_RATE_RAD_S),
            ]
        ),
        "latent": rng.normal(scale=1.5, size=NAV_LATENT_DIM),
        "extent": rng.uniform(TRAINING_EXTENT_MIN, TRAINING_EXTENT_MAX),
    }


def reference_observation(sample):
    """The training task's computation, using aerial_gym's own tensor helpers.

    Mirrors process_obs_for_task() line for line, including both 1e-6 terms: the one added
    componentwise INSIDE the distance norm and the one in each divisor. They are not
    interchangeable and neither is cosmetic at this tolerance.
    """
    position = torch.tensor(sample["position"], dtype=torch.float64).unsqueeze(0)
    velocity = torch.tensor(sample["velocity"], dtype=torch.float64).unsqueeze(0)
    quat = torch.tensor(sample["quat"], dtype=torch.float64).unsqueeze(0)
    target = torch.tensor(sample["target"], dtype=torch.float64).unsqueeze(0)
    extent = torch.tensor(sample["extent"], dtype=torch.float64).unsqueeze(0)
    gravity = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64)

    vehicle_quat = vehicle_frame_quat_from_quat(quat)

    obs = torch.zeros((1, NAV_OBS_DIM), dtype=torch.float64)

    vec_to_target = quat_rotate_inverse(vehicle_quat, target - position)
    dist = torch.linalg.norm(vec_to_target + 1e-6, dim=1, keepdim=True)
    obs[:, 0:3] = vec_to_target / (dist + 1e-6)

    max_dist = torch.norm(extent, dim=-1)
    obs[:, 3] = torch.clamp(dist.squeeze(1) / (max_dist + 1e-6), 0.0, 1.0)

    obs[:, 4:7] = quat_rotate_inverse(vehicle_quat, velocity)
    obs[:, 7:10] = torch.tensor(sample["angvel"], dtype=torch.float64)

    gravity_body = quat_rotate_inverse(quat, gravity)
    obs[:, 10:13] = gravity_body / torch.linalg.norm(gravity_body + 1e-6, dim=-1, keepdim=True)

    obs[:, 13:17] = torch.tensor(sample["prev_action"], dtype=torch.float64)
    obs[:, 17:NAV_OBS_DIM] = torch.tensor(sample["latent"], dtype=torch.float64)
    return obs.squeeze(0).numpy()


def deployed_observation(sample):
    state = DroneState(
        position=sample["position"],
        velocity_world=sample["velocity"],
        quat=sample["quat"],
        angvel_body=sample["angvel"],
        stamp_us=0,
    )
    return build_nav_observation(
        state,
        sample["target"],
        sample["prev_action"],
        sample["latent"],
        # The task divides by the LIVE env diagonal, so parity is checked against that
        # rather than against the deployment's frozen constant. What that constant should
        # be is a separate question, tested in test_default_distance_norm_*.
        distance_norm_m=float(np.linalg.norm(sample["extent"])),
    )


def test_observation_matches_training_task():
    rng = np.random.default_rng(20260831)
    worst = 0.0
    for _ in range(NUM_SAMPLES):
        sample = random_state(rng)
        difference = np.abs(reference_observation(sample) - deployed_observation(sample))
        worst = max(worst, difference.max())
    assert worst < TOLERANCE, f"deployed observation drifted from training by {worst}"


def test_velocity_is_vehicle_frame_not_body_frame():
    """[4:7] must not respond to roll or pitch -- only to heading.

    The hover observation puts velocity in the FULL body frame; this one puts it in the
    yaw-only vehicle frame. The two agree exactly whenever the vehicle is level, so copying
    the hover implementation across produces something that passes every hover-shaped check
    and is wrong in every turn.
    """
    velocity = np.array([2.0, -1.0, 0.5])
    reference = None
    for pitch in (0.0, 0.4, -0.4):
        quat = (
            quat_from_euler_xyz(
                torch.tensor([0.25]), torch.tensor([pitch]), torch.tensor([0.9])
            )
            .squeeze(0)
            .numpy()
        )
        state = DroneState(
            position=np.zeros(3),
            velocity_world=velocity,
            quat=quat,
            angvel_body=np.zeros(3),
            stamp_us=0,
        )
        obs = build_nav_observation(state, np.array([5.0, 0.0, 2.0]), np.zeros(4), np.zeros(32))
        if reference is None:
            reference = obs[4:7].copy()
        else:
            assert np.allclose(obs[4:7], reference, atol=1e-12)


def test_direction_channel_is_a_unit_vector():
    """[0:3] carries no range information at all.

    All of it is in [3]. A deployment that put the raw displacement here -- which is what
    the hover observation's [0:3] is -- would hand the policy metres in a channel whose
    training statistics span [-1, 1].
    """
    rng = np.random.default_rng(7)
    for _ in range(50):
        sample = random_state(rng)
        obs = deployed_observation(sample)
        assert np.linalg.norm(obs[0:3]) == pytest.approx(1.0, abs=1e-6)


def test_gravity_channel_is_at_10_not_3():
    """A level vehicle must produce obs[12] ~ -1, and obs[3:6] must be something else.

    The hover layout puts gravity at [3:6]; here [3] is the distance scalar and gravity has
    moved to [10:13]. Both vectors are 'mostly zeros with a -1', so a mis-indexed copy
    stays plausible-looking right up until it flies.
    """
    level = DroneState(
        position=np.zeros(3),
        velocity_world=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        angvel_body=np.zeros(3),
        stamp_us=0,
    )
    obs = build_nav_observation(level, np.array([10.0, 0.0, 0.0]), np.zeros(4), np.zeros(32))
    assert obs[10] == pytest.approx(0.0, abs=1e-6)
    assert obs[11] == pytest.approx(0.0, abs=1e-6)
    assert obs[12] == pytest.approx(-1.0, abs=1e-5)
    # The distance channel, not a gravity component.
    assert obs[3] == pytest.approx(10.0 / DISTANCE_NORM_M, abs=1e-4)


def test_distance_channel_clamps_at_one():
    far = DroneState(
        position=np.zeros(3),
        velocity_world=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        angvel_body=np.zeros(3),
        stamp_us=0,
    )
    obs = build_nav_observation(far, np.array([500.0, 0.0, 0.0]), np.zeros(4), np.zeros(32))
    assert obs[3] == 1.0


def test_default_distance_norm_sits_inside_the_training_band():
    """The frozen deployment constant must be a value the task actually produced.

    The task divides by the live env diagonal, which is randomized every reset. The
    deployment cannot be, so it takes the middle of the band -- and this pins how wide that
    band is. If the env config's bounds change, this fails rather than letting the constant
    quietly become wrong.
    """
    low = float(np.linalg.norm(TRAINING_EXTENT_MIN))
    high = float(np.linalg.norm(TRAINING_EXTENT_MAX))
    assert low <= DISTANCE_NORM_M <= high
    # And the band is narrow enough that picking one number costs almost nothing.
    assert (high - low) / DISTANCE_NORM_M < 0.02


def test_prev_action_is_the_transformed_command_not_the_raw_output():
    """Channels [14:16] are RADIANS and [16] is RAD/S.

    attitude_navigation_task.py:797,809 latches action_transformation_function's output;
    hover_task.py:560,572 latches the clamped raw action. Copying hover's convention here
    understates roll and pitch by 4/pi and the yaw rate by 3/pi, in a channel the jerk
    penalty trained the policy to read carefully.

    This test states the convention rather than deriving it, on purpose: it is the one
    thing about this layout that cannot be inferred from looking at the vector.
    """
    raw = np.array([0.5, 1.0, -1.0, 1.0])
    transformed = np.array([0.5, MAX_TILT_RAD, -MAX_TILT_RAD, MAX_YAW_RATE_RAD_S])
    state = DroneState(
        position=np.zeros(3),
        velocity_world=np.zeros(3),
        quat=np.array([0.0, 0.0, 0.0, 1.0]),
        angvel_body=np.zeros(3),
        stamp_us=0,
    )
    obs = build_nav_observation(state, np.array([1.0, 0.0, 0.0]), transformed, np.zeros(32))
    assert np.allclose(obs[13:17], transformed)
    assert not np.allclose(obs[13:17], raw)


# ==========================================================================================
# The depth half
# ==========================================================================================


def torch_reference_preprocess(depth_m):
    """The training preprocessing, from vae_depth itself.

    Imported rather than restated: the numpy side is the reimplementation under test, and a
    test that compared it against a second hand-written description would only prove the two
    descriptions agree.
    """
    import torch.nn.functional as F
    from vae_depth.preprocessing import normalize_depth

    x = torch.as_tensor(np.asarray(depth_m, dtype=np.float32)).reshape(1, 1, *depth_m.shape)
    if x.shape[-2:] != (TARGET_HEIGHT, TARGET_WIDTH):
        x = F.interpolate(x, (TARGET_HEIGHT, TARGET_WIDTH), mode="nearest")
    return normalize_depth(x, MAX_DEPTH_M, MIN_DEPTH_M).numpy()


@pytest.mark.parametrize("shape", [(180, 320), (480, 848), (720, 1280), (240, 424)])
def test_depth_preprocessing_matches_training(shape):
    """numpy preprocessing == the training pipeline, at every resolution a D435 emits.

    The resolutions matter individually because the resize is nearest-neighbour: torch uses
    floor(dst * src/dst), not round-to-nearest, and the two disagree on about half the
    output columns at a non-integer scale. Gazebo already publishes 180x320 so SITL would
    never catch it; real hardware would, in flight.
    """
    rng = np.random.default_rng(11)
    depth = rng.uniform(0.15, 9.5, size=shape).astype(np.float32)
    assert np.array_equal(preprocess_depth(depth), torch_reference_preprocess(depth))


def test_depth_normalization_is_inverted_near_is_high():
    """Close obstacles are 1.0, open space is 0.0.

    Getting this backwards produces a well-formed image, a well-formed latent, and a policy
    that steers into things.
    """
    near = preprocess_depth(np.full((180, 320), 0.2, dtype=np.float32))
    far = preprocess_depth(np.full((180, 320), 20.0, dtype=np.float32))
    assert near.min() > 0.9
    assert far.max() == 0.0


def test_invalid_pixels_read_as_free_space_not_as_an_obstacle():
    """NaN, +inf and 0 must not become 'something touching the lens'.

    0 is the dangerous one: a RealSense D435 writes it where stereo matching failed, and
    taken literally it clamps to min_depth and reads as the closest possible obstacle. The
    simulator has no equivalent -- aerial_gym writes +/-max_range past its clip planes -- so
    nothing in SITL exercises this.
    """
    depth = np.full((180, 320), 5.0, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    depth[0, 2] = 0.0
    out = preprocess_depth(depth, zero_is_far=True)
    assert np.all(np.isfinite(out))
    assert out[0, 0, 0, 0] == 0.0
    assert out[0, 0, 0, 1] == 0.0
    assert out[0, 0, 0, 2] == 0.0

    # -inf keeps the simulator's meaning: nearer than the near plane is a real obstacle.
    depth[0, 3] = -np.inf
    assert preprocess_depth(depth)[0, 0, 0, 3] == pytest.approx(1.0 - MIN_DEPTH_M / MAX_DEPTH_M)

    # And the literal reading is still reachable, for a driver that means it.
    assert preprocess_depth(depth, zero_is_far=False)[0, 0, 0, 2] == pytest.approx(
        1.0 - MIN_DEPTH_M / MAX_DEPTH_M
    )


def test_deployment_depth_constants_match_the_task_config():
    """The flight side's encoder geometry must be the task's, not a lookalike.

    control_policy_api/depth.py restates the resolution and the clamp window as plain
    constants, because the flight container cannot import the aerial_gym config tree to
    look them up -- and deploy/navigation.py reads them from there for the same reason,
    which is what keeps isaacgym out of the export path.

    That restatement is only safe if something checks it. This is that something. It fails
    if the task's vae_config is retuned without the deployment following, which would
    otherwise show up as a policy that flies slightly wrong in a way no shape check sees.
    """
    from config.task_config.f450_attitude_navigation_task_config import task_config

    vae = task_config.vae_config
    assert (vae.target_height, vae.target_width) == (TARGET_HEIGHT, TARGET_WIDTH)
    assert vae.max_depth_m == MAX_DEPTH_M
    assert vae.min_depth_m == MIN_DEPTH_M
    assert vae.latent_dims == NAV_LATENT_DIM
    assert task_config.state_dim + vae.latent_dims == NAV_OBS_DIM


def test_deployment_action_scaling_matches_the_task_config():
    """max_tilt / max_yaw_rate are part of the training contract, not a limiter.

    BasePolicy carries them for the flight side; the task carries them for training. They
    are the same two numbers and nothing but this connects them.
    """
    from config.task_config.f450_attitude_navigation_task_config import task_config
    from control_policy_api.base import BasePolicy

    assert task_config.max_inclination_angle_rad == BasePolicy.max_tilt_rad
    assert task_config.max_yaw_rate == BasePolicy.max_yaw_rate_rad_s
