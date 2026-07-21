# `turning_on_radio` button visual and geometry prior

Read `held_hand` and `press_hand` from the checkpoint-bound post-pick state;
never hard-code left or right.

## Face classification

The true button is on a red broad face and has all four positive fields:

Visually, this is a black round or oval disk, a white outer ring, and a raised
red center bump on the red front face.

- `red_front_face=true`
- `black_round_or_oval_disk=true`
- `white_outer_ring=true`
- `red_center_bump=true`

Only this complete signature permits a bbox, center pixel, or depth projection.
Incomplete signatures are `button_visible=false` / `NOT_VISIBLE` and must not
produce coordinates.

Use one negative class with this precedence:

1. `clear_slotted_back_face`: a red broad face contains a clearly elongated
   black horizontal slot/line. It is the non-button back face, even if the white
   side port is also visible. Its opposite broad face is the button face.
2. `side_port`: the recognizable white side face contains a black oval port and
   no slotted red broad face is visible. Do not infer the button-face direction
   from this view.
3. `ambiguous`: edge, red/white boundary, handle, grille, cable, shadow,
   gripper contact, seam, partial face, or any other insufficient view. Do not
   infer the button-face direction.

For side-port or ambiguous views, only a small held-hand observation-improving
rotation or translation is allowed before a fresh observation. A clear slotted
back face may directly authorize the opposite-face button alignment goal.

## Held alignment is a button goal

Never ask the LLM to provide an exact held EEF xyz/quaternion. The public held
goal states the desired radio/button geometry:

- move the button center toward the robot by the requested amount;
- raise the radio enough to maintain the requested table clearance, while
  allowing XY and orientation changes for reachability;
- make the button face point toward the dynamic press-hand side; and
- make the button outward normal approximately perpendicular to the head-camera
  optical axis, so the head sees a near side view.

The runtime—not the LLM—applies the live grasp transform, generates multiple
radio and held-EEF pose candidates, scores them by button geometry, and sends
eligible candidates to CuRobo. CuRobo provides reachability and collision
selection. A literal EEF target is not part of the public tool contract.

Head near-side-view is intentional. It can hide part of the positive button
signature, so head imagery is only a global pose/safety audit after held
alignment. The final button localization must use a fresh dynamic
`press_wrist` frame.

## Press staging is projection-driven

After held alignment completes, observe from `press_wrist`. Require the full
positive signature in that exact frame, then call `pixel_to_world` on its button
center. Use the resulting `projection_id`, 3D center, and outward normal to
request a non-contact press-staging button goal.

Use the raised red center bump as the visual button center. In the fresh
`press_wrist` image, prefer that red center within a radius of `0.08 *
min(image_width, image_height)` from the image center, and require it within
`0.10 * min(image_width, image_height)` for final pre-press acceptance. These
are normalized current-frame constraints, never reusable historical pixels.
If the red center is outside the required center region, do not press or save
`state_checkpoint_2`; obtain fresh wrist geometry and refine the non-contact
staging alignment.

First try the final 0.03--0.06 m press EEF standoff. If every roll-equivalent
candidate fails real CuRobo plan-only reachability, do not force a literal EEF
pose and do not split the translation into small steps. Use the same fresh
button projection for one farther 0.06--0.25 m non-contact wrist-camera staging
move (normally 0.12--0.20 m), then observe and project again. This fallback is
only an observation staging pose; it is not sufficient for saving
`state_checkpoint_2`. Set `alignment_phase="observation"`; the runtime uses the
live EEF-to-camera transform so the real wrist-camera optical axis, rather than
an assumed EEF axis, points at the raised red center. Final alignment uses
`alignment_phase="final"`, whose approach direction points into the button and
is antiparallel to the outward button normal. CuRobo selects a reachable
collision-free roll candidate in either phase.
The empty press hand may move to the selected goal in one certified motion and
must not be artificially divided into small steps.

Accept the final pre-press geometry only when:

- approach-line distance is at most 0.010 m;
- opposition error `angle(button_normal, -press_direction)` is at most 15
  degrees;
- axial standoff is positive and between 0.03 and 0.06 m;
- the true button's raised red center is within the central `press_wrist`
  acceptance region defined above;
- the radio remains held; and
- `press_hand` has not contacted the radio.

Every execution requires a matching plan-only certificate for the current
checkpoint, env step, visual/projection evidence, dynamic role, button goal,
selected candidate, and exact planned q trajectory. The held gripper is forced
closed at each waypoint and the attached radio participates in collision
checking. Consolidated endpoint stability and three-view review are sufficient;
do not require redundant per-waypoint contact/co-motion evidence.

If the radio drops or lies flat on the table, stop the env and fail the run;
never attempt a second grasp. Do not press the button in stage 2. Save only the robot-motion
`state_checkpoint_2` at `pre_press_alignment`, pause physics, and keep official
task success separate from stage-2 success.
