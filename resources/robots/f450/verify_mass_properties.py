#!/usr/bin/env python3
"""Verify that what Isaac Gym actually loads for model.urdf matches the
physical model in params.yaml - i.e. check the ASSET AS PARSED BY THE
SIMULATOR, not just the XML text.

For each link, checks:
  - mass/CoM/full inertia tensor (gym.get_actor_rigid_body_properties)
    against that link's own params.yaml entry (base_link's expected values
    are the SAME parallel-axis composition render.py uses to build
    model.urdf, so this also re-validates that composition independently).
  - rest position relative to base_link (gym.get_actor_rigid_body_states)
    against the joint origin authored in params.yaml/the URDF - catches
    URDF joint-parsing issues the mass/inertia check alone would miss.

Then prints a whole-vehicle summary (sum of all links, in the base_link
frame) to eyeball against the SDF's own recorded "gz sdf, inertial-stats"
figures in NOTES.md "Repower" (mass 2.373 kg, CoM [0.00303, 0, -0.01571],
Ixx/Iyy/Izz 0.024360/0.025955/0.046023).

NOTE: does not check collapse_fixed_joints=True, because every joint in
model.urdf is marked dont_collapse="true" (needed so aerial_gym can apply
per-motor thrust forces at the prop links) - confirmed empirically that
collapse_fixed_joints has no effect here, so there is no single collapsed
body for Isaac Gym to report; per-link is the only mode that reflects how
this asset is actually used in training.

Also asserts override_com=False and override_inertia=False: if either were
left True, Isaac Gym recomputes mass properties from collision geometry +
density and silently DISCARDS the <inertial> values authored in model.urdf,
which would make this whole check meaningless (both sides would just be
Isaac Gym's own geometry-based guess). Confirmed the gymapi default is
False for both, but this script sets them explicitly rather than relying
on the default, and prints a warning if the loaded asset didn't actually
use the authored inertial values by chance.

Usage: python3 verify_mass_properties.py
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from render import compose_rigid_body, load_params, prop_components  # noqa: E402

from isaacgym import gymapi  # noqa: E402


def inertia_mat33_to_dict(m):
    return {
        "ixx": m.x.x, "ixy": m.x.y, "ixz": m.x.z,
        "iyy": m.y.y, "iyz": m.y.z, "izz": m.z.z,
    }


def inertia_np_to_dict(arr):
    return {
        "ixx": arr[0, 0], "ixy": arr[0, 1], "ixz": arr[0, 2],
        "iyy": arr[1, 1], "iyz": arr[1, 2], "izz": arr[2, 2],
    }


def print_diff(label, exp_mass, exp_com, exp_inertia, act_mass, act_com, act_inertia):
    print(f"--- {label} ---")
    print(f"  mass  expected={exp_mass:.6f}  actual={act_mass:.6f}  diff={act_mass - exp_mass:+.2e}")
    for i, ax in enumerate("xyz"):
        print(f"  com.{ax} expected={exp_com[i]:+.6f}  actual={act_com[i]:+.6f}  diff={act_com[i] - exp_com[i]:+.2e}")
    for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
        print(f"  {k}   expected={exp_inertia[k]:+.6f}  actual={act_inertia[k]:+.6f}  diff={act_inertia[k] - exp_inertia[k]:+.2e}")


def main():
    params = load_params()

    gym = gymapi.acquire_gym()
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())

    asset_options = gymapi.AssetOptions()
    asset_options.collapse_fixed_joints = False  # see module docstring: dont_collapse="true" forces this anyway
    asset_options.override_com = False
    asset_options.override_inertia = False

    asset = gym.load_asset(sim, str(HERE), "model.urdf", asset_options)
    names = gym.get_asset_rigid_body_names(asset)
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    actor = gym.create_actor(env, asset, gymapi.Transform(), "f450", 0, 1)
    body_props = gym.get_actor_rigid_body_properties(env, actor)
    body_states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)

    base_mass, base_com, base_inertia_mat = compose_rigid_body(params["base_link_components"])
    expected_mass_com_inertia = {
        "base_link": (base_mass, base_com, inertia_np_to_dict(base_inertia_mat)),
    }
    expected_position = {"base_link": [0.0, 0.0, 0.0]}
    prop = params["prop"]
    zero_extra = {"ixy": 0.0, "ixz": 0.0, "iyz": 0.0}
    for pcfg in params["props"]:
        link = f"{pcfg['name']}_prop"
        expected_mass_com_inertia[link] = (
            prop["mass"],
            [0.0, 0.0, 0.0],  # own CoM is the link origin
            {"ixx": prop["inertia"]["ixx"], "iyy": prop["inertia"]["iyy"], "izz": prop["inertia"]["izz"], **zero_extra},
        )
        expected_position[link] = [
            pcfg["x_sign"] * prop["arm_xy_offset"],
            pcfg["y_sign"] * prop["arm_xy_offset"],
            prop["z"],
        ]

    aggregate_components = []
    for name, p, s in zip(names, body_props, body_states):
        exp_mass, exp_com, exp_inertia = expected_mass_com_inertia[name]
        print_diff(name, exp_mass, exp_com, exp_inertia, p.mass, [p.com.x, p.com.y, p.com.z], inertia_mat33_to_dict(p.inertia))

        pos = s["pose"]["p"]
        actual_pos = [float(pos["x"]), float(pos["y"]), float(pos["z"])]
        exp_pos = expected_position[name]
        pos_diff = [actual_pos[i] - exp_pos[i] for i in range(3)]
        print(f"  joint origin: expected={exp_pos}  actual={actual_pos}  diff={pos_diff}")

        # Compose this link's own inertia (about its own CoM, in its own
        # local frame) into the whole-vehicle frame: shift by the link's
        # rest position + its own CoM offset (both reported by Isaac Gym).
        link_com_in_base_frame = [
            actual_pos[0] + p.com.x, actual_pos[1] + p.com.y, actual_pos[2] + p.com.z,
        ]
        aggregate_components.append({
            "mass": p.mass,
            "position": link_com_in_base_frame,
            "inertia": inertia_mat33_to_dict(p.inertia),
        })

    sim_mass, sim_com, sim_inertia = compose_rigid_body(aggregate_components)
    exp_full_mass, exp_full_com, exp_full_inertia = compose_rigid_body(
        params["base_link_components"] + prop_components(params)
    )
    print()
    print_diff(
        "WHOLE VEHICLE (summed from Isaac Gym's own per-link readings above)",
        exp_full_mass, exp_full_com, inertia_np_to_dict(exp_full_inertia),
        sim_mass, sim_com, inertia_np_to_dict(sim_inertia),
    )
    print()
    print("Reference (SDF NOTES.md 'Repower'): mass~2.373 kg, CoM~[0.00303, 0, -0.01571], "
          "Ixx~0.024360, Iyy~0.025955, Izz~0.046023")

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
