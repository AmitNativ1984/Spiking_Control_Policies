"""Generate panel URDFs with varying widths and heights for diverse obstacle scenes.

Creates panels ranging from narrow slits to wide barriers.
Output directory: the aerial gym panels asset folder.
"""

import os
import numpy as np

PANEL_TEMPLATE = """<?xml version='1.0' encoding='UTF-8'?>
<robot name="panel_{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>
      <mass value="1.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual name="base_link_visual">
      <geometry>
        <box size="{thickness} {width} {height}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision name="base_link_collision">
      <geometry>
        <box size="{thickness} {width} {height}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
  </link>
</robot>"""

OUTPUT_DIR = "/app/aerial_gym/aerial_gym_simulator/resources/models/environment_assets/panels"


def generate_panels():
    np.random.seed(0)

    # Keep original panel
    # panel.urdf: 0.1 x 1.2 x 3.0 (already exists)

    # Generate varied panels:
    # - widths from 0.5m (narrow) to 5.0m (wide barrier)
    # - heights from 1.0m (low) to 6.0m (tall)
    # - thickness stays thin: 0.05 to 0.15m
    configs = []
    for i in range(50):
        thickness = np.random.uniform(0.05, 0.15)
        width = np.random.uniform(0.5, 5.0)
        height = np.random.uniform(1.0, 6.0)
        configs.append((thickness, width, height))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, (thickness, width, height) in enumerate(configs):
        name = f"panel_{i}"
        urdf_content = PANEL_TEMPLATE.format(
            name=name,
            thickness=f"{thickness:.3f}",
            width=f"{width:.3f}",
            height=f"{height:.3f}",
        )
        filepath = os.path.join(OUTPUT_DIR, f"{name}.urdf")
        with open(filepath, "w") as f:
            f.write(urdf_content)

    print(f"Generated {len(configs)} panel URDFs in {OUTPUT_DIR}")
    print(f"Width range: {min(c[1] for c in configs):.2f} - {max(c[1] for c in configs):.2f}m")
    print(f"Height range: {min(c[2] for c in configs):.2f} - {max(c[2] for c in configs):.2f}m")


if __name__ == "__main__":
    generate_panels()
