# R1Pro Robot Prior

This guide contains task-independent, approximate geometry for the R1Pro hands.
It is a robot-size prior, not task memory. It applies equally to all BEHAVIOR
tasks.

## Hand frame terminology

- `wrist_cam`: wrist RGB-D camera.
- `palm`: the `gripper_link` origin used as a palm reference; it is not a
  calibrated physical palm center.
- `grip_point`: the `eef_link` planning and attachment reference; it is not
  guaranteed to be the physical contact or grasp center.
- `finger1_root` / `finger2_root`: the two finger-joint frame origins at
  `q=0`.
- `finger_tip_rough`: qualitative fingertip region. No calibrated point is
  provided for it here.

The robot assets give the left and right hands the same nominal local offsets
and joint limits. This symmetry does **not** prove identical live calibration,
authorize mirroring, or select a hand for an episode. The active hand must
still be resolved from fresh observations and current runtime state.

## Structured prior

```yaml
r1pro_hand_prior:
  units: meters
  left_right_same: true

  wrist_camera_origin_in_gripper_link:
    translation_xyz: [0.05051, 0.0028934, 0.0051317]
    translation_only: true
    complete_transform_required_for_3d_conversion: true
    distance_to_palm: 0.0509

  hand_points_in_palm:
    palm: [0.0, 0.0, 0.0]
    grip_point: [0.0, 0.0, -0.06]
    finger1_root_closed: [-0.000088709, 0.013453, -0.03689]
    finger2_root_closed: [0.000089046, -0.013453, -0.03689]

  fixed_distances_from_wrist_cam:
    palm: 0.0509
    grip_point: 0.0825
    finger1_root_closed: 0.0666
    finger2_root_closed: 0.0676

  gripper_kinematics:
    each_finger_joint_range: [0.0, 0.05]
    combined_added_finger_separation_range: [0.0, 0.10]
    physical_fingertip_aperture_calibrated: false
```

All values are approximate nominal dimensions. They do not replace current
camera calibration, transforms, joint feedback, collision checking, or fresh
visual evidence. The two finger-root coordinates apply only at `q=0`; they move
with live finger joint position. The combined `0.10 m` range is the sum of two
opposed prismatic joint travels, not a calibrated physical gap between finger
pads.

## Using wrist RGB-D measurements

When a wrist RGB-D camera sees a target, first convert the LLM-selected target
pixel and its valid depth into a 3D point in the wrist-camera frame. Never
estimate palm or grasp-center distance by subtracting a fixed offset from one
scalar depth value.

For the current public RGB-D contract, `optical_axis_depth_m` is the target
surface's Z distance to the image plane and is the depth used with the selected
pixel and camera intrinsics for back-projection. `camera_range_m` is the
Euclidean range from the camera center to that visible surface point. Do not
interchange these two quantities.

A successful wrist-camera `depth_probe` also returns the calibrated,
frame-bound `target_point_camera_xyz_m` in effective USD camera coordinates
(`+X` right, `+Y` up, `-Z` forward). For `left_wrist` or `right_wrist`, when
the same capture contains the complete live transforms for that literal
anatomical hand, the runtime reports:

- `target_to_palm_m`;
- `target_to_grip_point_m`;
- `target_to_finger_roots_m`, the minimum distance to the two live finger-root
  frame origins.

These distances are computed by the runtime from the selected visible-surface
point and same-capture live robot transforms. Prefer them over manually
combining the nominal offsets below. A head-camera probe has no unique hand
identity and therefore does not provide the three hand-relative distances.
Missing or invalid transform lineage makes the hand-distance result
unavailable rather than approximate by guessing.

The literal wrist-camera name binds geometry to its matching physical side; it
does not authorize selecting that hand for an analytic action. Analytic hand
selection still requires the fresh public head-frame evidence defined by the
runtime contract.

When the current physical wrist-camera-to-hand transform is available:

1. Back-project the selected pixel and depth to a 3D target point in
   `wrist_cam`.
2. Transform that point from `wrist_cam` into the corresponding `palm` frame
   with the complete current `T_palm_from_camera` rotation and translation.
   The translation vector in this guide is not a complete transform.
3. Compute Euclidean distances in a common frame:

   ```text
   target_to_palm =
       distance(target, palm)

   target_to_grip_point =
       distance(target, grip_point)

   target_to_finger_roots =
       min(
           distance(target, finger1_root_closed),
           distance(target, finger2_root_closed),
       )
   ```

4. Use the results only as approximate reach and grasp guidance. Re-observe
   after motion because the camera pose, target visibility, and occlusion can
   change. Even runtime-computed distances are explicitly non-semantic,
   non-collision-authorizing, and not a `close` or `open` gate.

If only scalar Euclidean `camera_range_m = r` is available:

- With camera-to-reference distance `d`, only the triangle-inequality bound
  `abs(r - d) <= target_to_reference <= r + d` is justified.
- For the palm reference, use `d = 0.0509 m`.
- For the grip-point reference, use `d = 0.0825 m`.
- The scalar value does not identify direction in the camera frame and cannot
  establish palm clearance, fingertip clearance, object centering, or contact.
- Do not apply these Euclidean bounds directly to `optical_axis_depth_m`.

## Safety and evidence limits

This prior may supplement geometric reasoning for picking, positioning, and
reach assessment. It must never be used by itself as:

- a contact or release gate;
- proof that an object is inside the gripper or a receptacle;
- grasp, primitive, or official task-success evidence;
- collision clearance or motion authorization;
- a substitute for current RGB-D, camera extrinsics, kinematics, attachment
  state, or controller feedback;
- justification for a fixed hand choice, fixed pixel, fixed pose, or fixed
  motion distance.

Invalid, missing, stale, occluded, or mixed-surface depth remains
non-authorizing. Acquire fresh evidence or fail closed.

## Compact prior for the LLM

> R1Pro hand prior: The robot assets give the left and right hands the same
> nominal local offsets, but this neither proves identical live calibration nor
> chooses the active hand. Resolve the hand from fresh episode evidence. The
> wrist-camera origin is about 5.1 cm from the `gripper_link` palm reference
> and about 8.3 cm from the `eef_link` grip reference. The grip reference is
> 6.0 cm below the palm reference in the hand frame. Closed (`q=0`) finger
> joint roots are about 6.7 cm from the wrist-camera origin. Use these values
> only as a rough robot-size prior. When wrist RGB-D sees an object, combine
> the selected pixel, optical-axis depth, and camera intrinsics to form a 3D
> camera-frame point, then use a complete current camera-to-hand transform
> before estimating distance to the palm, grip point, or finger roots. Never
> treat scalar depth as palm or grip-point distance by subtracting a fixed
> offset. This prior does not authorize contact, release, collision clearance,
> or success.
