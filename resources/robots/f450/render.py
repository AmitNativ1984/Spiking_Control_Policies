#!/usr/bin/env python3
"""Render model.urdf from templates/*.jinja + params.yaml.

Usage: python3 render.py
Run this after editing params.yaml or any templates/*.jinja file; Isaac Gym
loads the generated model.urdf directly and does not know about Jinja.

base_link's composite mass/CoM/inertia is DERIVED here from
params.yaml's base_link_components (frame, 4 motors, RealSense, Orin,
battery) via the parallel axis theorem, not hand-typed - editing any one
component's mass/position/inertia and rerunning this script keeps
base_link's aggregate numbers correct automatically.

compose_rigid_body() and load_params() are also imported by
verify_mass_properties.py to cross-check what Isaac Gym actually loads
against this same source of truth, with no numbers duplicated between them.
"""
import pathlib

import jinja2
import numpy as np
import yaml

HERE = pathlib.Path(__file__).parent


def load_params():
    return yaml.safe_load((HERE / "params.yaml").read_text())


def compose_rigid_body(components):
    """Combine components (each with mass/position/full inertia tensor, all
    expressed in the same axis-aligned frame) into one rigid body's
    mass, CoM, and full inertia tensor about that CoM (parallel axis theorem).

    Each component's own inertia may include off-diagonal product terms
    (ixy/ixz/iyz) - e.g. when composing an already-aggregated body (which can
    have nonzero products even if every real leaf part doesn't) back in as a
    "component". ixy/ixz/iyz default to 0 for plain diagonal inputs.
    """
    total_mass = sum(c["mass"] for c in components)
    com = np.zeros(3)
    for c in components:
        com += c["mass"] * np.array(c["position"], dtype=float)
    com /= total_mass

    inertia = np.zeros((3, 3))
    for c in components:
        r = np.array(c["position"], dtype=float) - com
        i = c["inertia"]
        ixy, ixz, iyz = i.get("ixy", 0.0), i.get("ixz", 0.0), i.get("iyz", 0.0)
        own = np.array([
            [i["ixx"], ixy, ixz],
            [ixy, i["iyy"], iyz],
            [ixz, iyz, i["izz"]],
        ])
        inertia += own + c["mass"] * (r.dot(r) * np.eye(3) - np.outer(r, r))

    return total_mass, com, inertia


def prop_components(params):
    prop = params["prop"]
    return [
        {
            "mass": prop["mass"],
            "position": [
                p["x_sign"] * prop["arm_xy_offset"],
                p["y_sign"] * prop["arm_xy_offset"],
                prop["z"],
            ],
            "inertia": prop["inertia"],
        }
        for p in params["props"]
    ]


def main():
    params = load_params()

    base_mass, base_com, base_inertia = compose_rigid_body(params["base_link_components"])

    base_link = dict(params["base_link_geometry"])
    base_link["mass"] = round(float(base_mass), 6)
    base_link["com"] = [round(float(x), 6) for x in base_com]
    base_link["inertia"] = {
        "ixx": round(float(base_inertia[0, 0]), 6),
        "ixy": round(float(base_inertia[0, 1]), 6),
        "ixz": round(float(base_inertia[0, 2]), 6),
        "iyy": round(float(base_inertia[1, 1]), 6),
        "iyz": round(float(base_inertia[1, 2]), 6),
        "izz": round(float(base_inertia[2, 2]), 6),
    }

    # Full-vehicle sanity check against the SDF's own recorded
    # "gz sdf, inertial-stats" output (see NOTES.md): compose
    # base_link_components together with the 4 props and compare
    # mass/CoM/Ixx/Iyy/Izz.
    full_mass, full_com, full_inertia = compose_rigid_body(
        params["base_link_components"] + prop_components(params)
    )
    print(
        f"Full-vehicle check: mass={full_mass:.4f} kg  "
        f"CoM={np.round(full_com, 6).tolist()}  "
        f"Ixx={full_inertia[0,0]:.6f}  Iyy={full_inertia[1,1]:.6f}  "
        f"Izz={full_inertia[2,2]:.6f}  Ixz={full_inertia[0,2]:.6f}"
    )
    print(
        "  (expect from NOTES.md 'Repower': mass~2.373, CoM~[0.00303, 0, -0.01571], "
        "Ixx~0.024360, Iyy~0.025955, Izz~0.046023, Ixz~0.000103)"
    )

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(HERE / "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = env.get_template("model.urdf.jinja").render(base_link=base_link, **params)

    out_path = HERE / "model.urdf"
    out_path.write_text(rendered)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
