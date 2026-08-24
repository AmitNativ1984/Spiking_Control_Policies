"""Attitude-controller config for the F450, with per-episode gain randomization on.

Subclasses the stock aerial_gym `control` config rather than editing it in place: the
upstream class is shared by every lee_* controller in the registry, so flipping
randomize_params there would silently change every other task in the tree.

Why randomize the gains at all: this is the cheapest sim-to-real hedge available here.
The real F450 flies a controller_configPX4 cascade whose effective attitude stiffness is not the Lee
controller's, and unlike mass, the gains are NOT communicated to the policy in any
form -- the policy only ever sees the resulting motion. That makes gain spread a genuine
plant/model mismatch, and it costs nothing to apply: randomize_params is pure tensor
indexing inside the controller (base_lee_controller.randomize_params), with none of the
per-env gym API calls that the rigid-body randomization needs.

Ranges are inherited unchanged from upstream (K_rot 0.8-1.2, K_angvel 0.1-0.2, i.e.
+/-20% around the nominal). Note these are absolute N.m/rad -- nothing in
control_allocation.py normalizes the controller's torque command by inertia -- so K_rot
= 1.0 is a soft attitude loop for a 2 kg airframe. That softness is why a small residual
torque bias used to show up as several degrees of steady-state tilt; the fix for that
belongs in the allocation matrix (see f450_config.control_allocator_config), not here.
"""

from aerial_gym.config.controller_config.lee_controller_config import control


class F450LeeAttitudeConfig(control):
    # Drawn per env from [min, max] on every episode reset, via
    # BaseMultirotor.reset_idx -> controller.randomize_params(env_ids).
    randomize_params = True
