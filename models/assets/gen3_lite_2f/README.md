# Kinova Gen3 lite stock two-finger gripper

The five STL files, `upstream_gen3_lite_2f_macro.xacro`, and `LICENSE` are copied
from Kinova's official `ros2_kortex` repository, pinned to commit
`b2aa46b29ae8fadd5b293c57e3500925c61eb8e8`:

https://github.com/Kinovarobotics/ros2_kortex/tree/b2aa46b29ae8fadd5b293c57e3500925c61eb8e8/kortex_description/grippers/gen3_lite_2f

The meshes use meters; they have no millimeter scale factor. The source's
BSD-3-Clause license is retained in `LICENSE`.

## MuJoCo conversion

`models/gen3_lite_gripper.xml` contains the body hierarchy to include inside the
arm's `end_effector_link`. The sibling includes `assets.xml`, `equality.xml`,
and `actuator.xml` belong in the parent model's corresponding sections. Mesh
paths assume the existing parent compiler's `meshdir="assets/gen3_lite"`.

Joint origin RPY values are converted to equivalent quaternions. Joint axes,
limits, per-link masses, centers of mass and full inertia tensors are taken
from the source's **hardware** branch (`use_fake_hardware`, `sim_isaac`, and
`sim_gazebo` all false; `moveit_active` false). This is intentional: its joint
origins and mimic coefficients differ from both the old ROS 1 description and
the source's separate MoveIt branch. Total stock gripper mass is 0.197766 kg.

The driver is `gripper_right_finger_bottom_joint`; the independent actuator is
`gripper`. Its command is radians in [0, 0.85], increasing toward closed. The
four joints are ordered right bottom, right tip, left bottom, left tip:

```
[g, -0.1 - 0.276*g, g, -0.1 - 0.276*g]
```

Three joint equality constraints reproduce the source mimic relationships.
The source tool frame is retained as `gripper_tool_frame` at flange z=0.130 m,
with a 90-degree yaw. Finger closing is approximately along the flange y axis.

## Simulation choices

- Visual geometry uses the exact meshes. Collision geometry is a separate
  convex hull for each stock link, computed by MuJoCo from the same meshes.
  The full gripper is never combined into one hull, preserving the open space
  between its fingers. Small link recesses are simplified by the link hulls.
- Collision bit 16 denotes the gripper; affinity 15 permits environment,
  arm, tool and tip contacts. Contacts within the stock mechanism are disabled
  by these masks. Intentional held-tool contacts are handled explicitly by the
  parent model's grasp exclusions, not by disabling all external contacts.
- Per-joint damping 0.1, armature 0.001, position gain 20, velocity gain 0.2,
  and actuator force range +/-10 are simulation settings, not identified
  hardware controller parameters. They give stable intermediate targets with
  the repository's 2 ms implicitfast integrator. Bodies use the same gravity
  compensation convention as the existing arm.
- `gripper_left_pad` and `gripper_right_pad` are representative central inner
  face reference sites at distal-link x=-25 mm and y=-7.4/+7.4 mm. Their
  separation is an aperture estimate at that face location; the articulated
  fingers do not have a uniform gap along their entire length. A command of
  about 0.6174 rad gives 42 mm separation, around flange z=0.1453 m. The
  source contains a small left/right frame offset (about 70 micrometers).

The custom pipette, its grasp shape and its separate actuation mechanism
are not part of these upstream assets.
