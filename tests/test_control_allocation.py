"""The F450 allocation matrix must take moments about the CENTER OF MASS.

PhysX takes moments about the CoM. If the allocation matrix references the base_link
origin instead, a commanded zero torque is not zero torque: on this airframe the two
points differ by 3.0 mm, so four equal thrusts leave m*g*0.003034 = 0.071 N.m of pitch
torque on the vehicle. The Lee attitude controller is pure PD with no integral term, so
it can only balance a constant bias with a constant attitude error -- measured at 4.06 deg
of steady pitch, which flew the drone sideways at 2.6 m/s under a zero action.

These are pure-CPU algebra checks on the config; no sim is built.
"""
import numpy as np

from config.robot_config.f450_config import _COM, _urdf_center_of_mass, F450Config

CAC = F450Config.control_allocator_config
A = np.array(CAC.allocation_matrix, dtype=float)

#: Full-vehicle mass, matching resources/robots/f450/params.yaml and the Gazebo SITL
#: source of truth. Only scales both sides of these assertions, but keeping it honest
#: stops the file reading as if it still described the 1.999 kg pre-repower vehicle.
MASS_KG = 2.373


def test_parsed_com_matches_isaac_gyms_own_computation():
    """The parse must agree with what robot_manager computes from the same URDF.

    Isaac Gym logs `Robot COM: [0.003034, 0.0, -0.015708]` at build for this airframe
    (verified end-to-end by resources/robots/f450/verify_mass_properties.py).
    """
    assert np.allclose(_COM, [0.003034, 0.0, -0.015708], atol=1e-5), _COM


def test_com_is_derived_from_the_urdf_not_hardcoded():
    """Changing the URDF's inertial block must move the CoM the allocation matrix uses."""
    import tempfile
    import shutil
    from pathlib import Path
    from config.robot_config.f450_config import _ASSET_FOLDER, _URDF_FILE

    with tempfile.TemporaryDirectory() as d:
        patched = Path(d) / _URDF_FILE
        text = (_ASSET_FOLDER / _URDF_FILE).read_text()
        assert '<origin xyz="0.003107 0.0 -0.016366"/>' in text, \
            "URDF inertial origin changed shape; update this test and re-check the matrix"
        patched.write_text(text.replace('<origin xyz="0.003107 0.0 -0.016366"/>',
                                        '<origin xyz="0.0 0.0 -0.016366"/>'))
        assert abs(_urdf_center_of_mass(patched)[0]) < 1e-9, \
            "CoM is not actually being read from the URDF"


def test_moment_arms_are_referenced_to_the_com():
    """Ty row must be -(x_i - com_x), not -x_i. This is the regression guard."""
    motor_x = CAC.arm_length * np.cos(CAC.theta) * np.array([+1.0, -1.0, +1.0, -1.0])
    motor_y = CAC.arm_length * np.sin(CAC.theta) * np.array([-1.0, +1.0, +1.0, -1.0])

    assert np.allclose(A[3], motor_y - _COM[1]), "Tx row is not CoM-referenced"
    assert np.allclose(A[4], -(motor_x - _COM[0])), "Ty row is not CoM-referenced"


def test_equal_thrusts_are_correctly_reported_as_a_pitch_torque():
    """With CoM-referenced arms, four equal thrusts are NOT a zero-torque command.

    This is the physical fact the old matrix denied. The allocator inverts this matrix, so
    getting it right is what makes a commanded Ty=0 come out as unequal thrusts that
    produce genuinely zero torque about the CoM.
    """
    hover = np.full(4, MASS_KG * 9.81 / 4.0)
    assert np.isclose(A[3] @ hover, 0.0, atol=1e-9), "equal thrusts should give zero roll torque"
    assert np.isclose(A[4] @ hover, MASS_KG * 9.81 * _COM[0], rtol=1e-6), \
        "equal thrusts should show up as m*g*com_x of pitch torque"


def test_commanded_zero_torque_produces_zero_torque_about_the_com():
    """Round-trip through the allocator's pseudo-inverse, which is what actually runs."""
    wrench = np.array([0.0, 0.0, MASS_KG * 9.81, 0.0, 0.0, 0.0])  # hover, zero torque
    thrusts = np.linalg.pinv(A) @ wrench
    assert np.allclose(A[3:6] @ thrusts, 0.0, atol=1e-9), \
        "solving for zero commanded torque left a residual torque about the CoM"
    assert np.isclose(thrusts.sum(), MASS_KG * 9.81, rtol=1e-6), "thrust no longer sums to weight"
    assert thrusts.std() > 0, \
        "thrusts are equal — a CoM-offset airframe needs differential thrust to hold level"
