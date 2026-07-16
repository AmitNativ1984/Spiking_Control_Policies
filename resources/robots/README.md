# Creating a robot URDF for this repo

This describes the workflow used to build `f450/model.urdf` - use it as the
template for adding another robot here. It's aimed at drones with a real
physical source of truth (a Gazebo SDF, a datasheet, a CAD export), where the
mass properties actually matter for RL training, not just a visual stand-in.

Isaac Gym only loads a flat `.urdf` file - there's no native way to split one
across files (that's an SDF feature, `<include>`, that URDF doesn't have). So
each robot folder here is generated from Jinja templates + a single params
file, and the flat `.urdf` is committed alongside them as the actual build
output Isaac Gym reads.

## Layout of a robot folder

```
resources/robots/<name>/
  params.yaml               # every mass/inertia/position number, with derivations
  templates/
    model.urdf.jinja         # top-level, {% include %}s the fragments below
    _base_link.urdf.jinja    # base_link (visual/collision/inertial)
    _prop.urdf.jinja         # one prop link+joint, instantiated per prop
  render.py                  # params.yaml -> model.urdf
  verify_mass_properties.py  # loads model.urdf in Isaac Gym, diffs against params.yaml
  model.urdf                 # GENERATED - do not hand-edit, see header comment
```

`f450/` is the reference implementation - copy its structure for a new robot
rather than starting from scratch.

## Step by step

1. **Find the source of truth.** A Gazebo/PX4 SITL SDF is ideal (real
   per-part mass, position, and inertia, already used for a working sim).
   Failing that: manufacturer datasheets, or first-principles estimates from
   CAD/measured dimensions (state clearly which numbers are which - see
   `f450/params.yaml`'s header comment for the pattern).

2. **List every physical component individually** in `params.yaml`: mass,
   position (relative to the robot's origin), and own inertia tensor
   (`ixx`/`iyy`/`izz`, plus `ixy`/`ixz`/`iyz` if not symmetric) for each one.
   Don't hand-compute a combined number for anything with more than one
   component - list the parts and let code combine them (step 4). A composite
   number typed directly into params.yaml can't be independently edited later
   and silently goes stale the next time someone swaps a part.

3. **Decide which parts need their own URDF link vs. get lumped into
   `base_link`.** A part needs its own link only if something in the sim
   needs to act on it individually - in aerial_gym's motor model, thrust and
   reaction torque are applied at the prop links, so props (and only props)
   stay separate. Everything else (frame, motors, battery, companion
   computer, cameras, ...) has no such requirement and gets lumped into one
   `base_link` rigid body, matching every other quad in this pipeline
   (`x500`, `lmf1`, etc.) - adding more separate-but-inert links only adds
   rigid bodies for the solver to track with no benefit.

4. **Let `render.py` do the parallel-axis composition** for whatever gets
   lumped into `base_link` (mass, CoM, full inertia tensor about that CoM) -
   see `compose_rigid_body()` in `f450/render.py`. Never hand-type a lumped
   body's mass/CoM/inertia in params.yaml; if you change one component's
   mass or position, the aggregate must update automatically when you rerun
   `render.py`, or it will quietly drift from what's actually being edited.

5. **Write the Jinja templates.** `model.urdf.jinja` includes a base_link
   fragment and loops over a list of repeated parts (props, in f450's case)
   via `{% for %}` + `{% include %}`, so N identical parts come from one
   template instead of N copy-pasted blocks. If you're building another
   quad, `f450/templates/` needs only parameter changes, not structural ones.

6. **Run `python3 render.py`.** It prints a full-vehicle sanity check (total
   mass, CoM, Ixx/Iyy/Izz) - compare this against your source of truth (e.g.
   a Gazebo `gz sdf --inertial-stats` output, or a manufacturer's stated
   takeoff weight) before trusting the result. Rerun it any time
   `params.yaml` or a template changes; `model.urdf` is derived output, never
   edit it directly.

7. **Collision geometry: keep it to one tight box**, sized to the real
   envelope of the whole robot (arm/prop reach for X/Y, leg-tip-to-tallest-
   point for Z) rather than a per-part collision mesh. See the derivation
   comment on `f450/params.yaml`'s `collision_box` for the pattern - compute
   each real part's own extent and take the axis-aligned min/max, don't
   guess a safety margin.

8. **Verify what Isaac Gym actually loaded**, not just the XML text -
   `f450/verify_mass_properties.py` is a template for this: it loads
   `model.urdf` the same way aerial_gym's `robot_asset` config would
   (`collapse_fixed_joints=False`, `override_com=False`,
   `override_inertia=False` - the last two matter: if left `True`, Isaac Gym
   silently recomputes mass properties from collision geometry and discards
   your authored `<inertial>` values), reads back each link's actual
   mass/CoM/inertia and rest position, and cross-checks the sum against the
   source of truth. Adapt it for a new robot rather than skipping this step -
   it's what catches URDF-authoring mistakes (wrong units, a dropped
   component, an axis mixed up) that XML validation can't.

## Sanity checks worth running on any new robot

- Total mass vs. the vehicle's known/measured weight.
- Thrust-to-weight ratio (sum of max per-motor thrust / total weight) is in
  a sane range (~2:1 or higher for stable hover).
- Inertia tensor is diagonal-dominant, with `Izz` the largest for a flat
  multirotor (`Izz > Ixx ~= Iyy`), and any product terms (`Ixy`/`Ixz`/`Iyz`)
  small relative to the diagonal - a large product term usually means an
  asymmetric CoM offset that's real (e.g. an off-center payload) rather than
  a units/axis bug, but check which one it is.
