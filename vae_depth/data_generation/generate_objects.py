"""Generate object URDFs with varying sizes for diverse obstacle scenes.

Creates three object types with randomized dimensions:
- Cubes/cuboids: 0.2-2.0m per side
- Rods: thin cross-section (0.05-0.2m), tall (0.5-4.0m)
- Walls: thin (0.05-0.15m), varied width (0.3-2.5m) and height (0.3-2.5m)

Output directory: the aerial gym objects asset folder.
"""

import os
import numpy as np

OBJECT_TEMPLATE = """<?xml version='1.0' encoding='UTF-8'?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual name="base_link_visual">
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision name="base_link_collision">
      <geometry>
        <box size="{sx} {sy} {sz}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
  </link>
</robot>"""

OUTPUT_DIR = "/app/aerial_gym/aerial_gym_simulator/resources/models/environment_assets/objects"


def generate_objects():
    np.random.seed(42)

    objects = []

    # --- Cubes / cuboids (20) ---
    for i in range(20):
        sx = np.random.uniform(0.2, 2.0)
        sy = np.random.uniform(0.2, 2.0)
        sz = np.random.uniform(0.2, 2.0)
        objects.append((f"cube_{i}", sx, sy, sz))

    # --- Rods (15) ---
    for i in range(15):
        cross = np.random.uniform(0.05, 0.2)
        length = np.random.uniform(0.5, 4.0)
        # Random orientation: rod along x, y, or z
        axis = np.random.randint(3)
        dims = [cross, cross, cross]
        dims[axis] = length
        objects.append((f"rod_{i}", dims[0], dims[1], dims[2]))

    # --- Walls / slabs (15) ---
    for i in range(15):
        thickness = np.random.uniform(0.05, 0.15)
        width = np.random.uniform(0.3, 2.5)
        height = np.random.uniform(0.3, 2.5)
        # Random thin axis
        axis = np.random.randint(3)
        dims = [width, width, height]
        dims[axis] = thickness
        objects.append((f"wall_{i}", dims[0], dims[1], dims[2]))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, sx, sy, sz in objects:
        urdf_content = OBJECT_TEMPLATE.format(
            name=name,
            sx=f"{sx:.3f}",
            sy=f"{sy:.3f}",
            sz=f"{sz:.3f}",
        )
        filepath = os.path.join(OUTPUT_DIR, f"{name}.urdf")
        with open(filepath, "w") as f:
            f.write(urdf_content)

    print(f"Generated {len(objects)} object URDFs in {OUTPUT_DIR}")
    print(f"  Cubes:  20  (sides 0.2-2.0m)")
    print(f"  Rods:   15  (cross 0.05-0.2m, length 0.5-4.0m)")
    print(f"  Walls:  15  (thickness 0.05-0.15m, face 0.3-2.5m)")


if __name__ == "__main__":
    generate_objects()
