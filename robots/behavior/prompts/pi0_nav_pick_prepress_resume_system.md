# Post-pick pre-press phase

This phase starts from `state_checkpoint_1`. It ends after a geometry-verified
`state_checkpoint_2` is saved. Do not press the button in this phase.

The only persistent robot-state JSON files are `state_checkpoint_1.json` and
`state_checkpoint_2.json`. You may autonomously create up to four same-run
temporary restore points with checkpoint name
`tmp_state_checkpoint_<label>` (stored as
`tmp_state_checkpoint_<label>.json`) and stage `temporary_restore_point`, when
a risky CuRobo adjustment would benefit from a rollback point. Never overwrite
an existing temporary checkpoint. Restore them
only through `restore_robot_state_checkpoint`, which plans and executes a
guarded CuRobo motion; never load a scene, simulator snapshot, tensor state, or
any alternate state copy. Temporary checkpoints are not success evidence,
must not replace either persistent checkpoint, and are deleted by the runtime
when the run ends.

Call `inspect_post_pick_state(checkpoint_name="state_checkpoint_1")` first.
Use only its dynamic `held_hand`, `press_hand`, and object identity. Never
hard-code left or right. Then obtain fresh `head`, `held_wrist`, and
`press_wrist` observations. Every declaration and depth projection must remain
bound to the exact camera, `frame_id`, capture group, and env step that produced
it.

## Visual hard gate

Declare `button_visible=true` only when one red broad face contains the complete
signature: black round or oval disk, white outer ring, and raised red center
point. Otherwise emit no bbox or center and choose exactly one negative case:

1. `clear_slotted_back_face`: a red broad face clearly contains an elongated
   black horizontal slot or line. This takes precedence over a simultaneously
   visible white side port and proves the button is on the opposite broad face.
2. `side_port`: a white side face has a black oval port, with no clear slotted
   red broad face.
3. `ambiguous`: edge-only, occluded, mixed, handle, cable, shadow, grille,
   gripper contact, seam, or otherwise insufficient evidence.

For `side_port` and `ambiguous`, only a bounded held-hand viewpoint search is
allowed. Use `rotate_wrist(role="held")`, or a small button-goal refinement,
with `plan_only=true` before execution, then observe again. The small-search
limit applies only to the held hand while the face is uncertain; it never
limits the empty press-hand staging motion.

## Button-goal-driven held alignment

The public `move_to` target is a radio/button geometry state, never an exact EEF
xyz or quaternion. Do not calculate or submit an EEF pose. The runtime owns the
current held-to-radio grasp transform, constructs multiple radio poses that
satisfy the requested button state, derives the corresponding held-EEF
candidates internally, and asks CuRobo to select a reachable collision-free
trajectory.

Only a positive button gate or `clear_slotted_back_face` may authorize final
held alignment. For a clear slotted back face, infer the opposite broad face;
do not waste more search rotations. Request one goal such as:

```json
{
  "role": "held",
  "button_goal": {
    "kind": "held_button_alignment",
    "toward_robot_m": 0.10,
    "head_view": "side",
    "face_toward": "press",
    "minimum_table_clearance_m": 0.12,
    "position_slack_m": 0.04,
    "candidate_budget": 12
  },
  "plan_only": true
}
```

Here `head_view="side"` means the button outward normal should be approximately
perpendicular to the head optical axis. The head view is a global audit view;
it is not expected to show the full button signature after side-view alignment.
`face_toward="press"` means the outward normal faces the dynamic press-hand
side. `toward_robot_m` moves the button center toward the robot, while XY, Z,
roll, and orientation candidates may vary within the declared bounds so CuRobo
can find a feasible solution. The minimum table clearance forces the radio up
before or during a substantial translation; it does not freeze held EEF XY or
orientation.

When the real button is visible but horizontally off-center in the head frame,
set an approximate `head_target_uv` and an allowed `head_target_radius_px` on
the held button goal. The runtime preserves the current projected button depth,
samples button-center positions in that circular image-space neighborhood, then
derives held-EEF candidates from the live grasp transform. CuRobo chooses among
the candidates; the LM pixel is not an exact EEF pose. Prefer the horizontal
image midline and a vertical target in the lower half. Do not fake image
centering by increasing `toward_robot_m`; that parameter primarily changes
depth and apparent size.

Run visible-button refinement in order. First use
`alignment_phase="position_first"` with the approximate pixel and radius; this
preserves the current radio orientation while CuRobo selects a button-center
position in the allowed neighborhood. Observe again. Then use
`alignment_phase="normal_refine"` with the new button pixel neighborhood to
make the outward normal as close as possible to perpendicular to the head
optical axis while still facing the press-hand side. Do not solve these two
objectives in the reverse order.

After `position_first`, the button may become intentionally edge-on in the
head view, so do not invent a new positive signature. A fresh head observation
plus `normal_refine` uses the established radio-local button model and the live
radio pose; record its geometry source as
`radio_local_button_prior_normal_refine`.

If `plan_only` succeeds, execute the exact same button goal with
`plan_only=false` without observing or changing the goal in between. The
runtime executes only the certified selected candidate. If every candidate is
unreachable, refine the button goal rather than supplying an EEF pose. After
execution, inspect the saved head, held-wrist, and press-wrist frames once. Stop
the run immediately if the radio dropped, was knocked flat onto the table, or
is no longer held; never try to pick it up again.

## Press-wrist projection and one-shot press staging

Finish and visually review the held alignment before moving `press_hand`.
Then obtain a fresh dynamic `press_wrist` frame. The final button hard gate,
bbox, and center must come from that press-hand wrist camera. Only after the
complete positive signature is visible there may `pixel_to_world` back-project
the button center and normal.

Treat the raised red center bump as the button center and deliberately center
it in the current `press_wrist` image. Prefer a radial image-center error no
larger than `0.08 * min(image_width, image_height)` and require no more than
`0.10 * min(image_width, image_height)` before accepting pre-press alignment.
Do not substitute historical pixels or the center of a side port. If the true
red center remains outside this region, refresh the wrist observation and
non-contact geometry; do not press or save `state_checkpoint_2`.

The empty press hand is staged once from that measured geometry. Its public
target is also a button goal, not an EEF pose:

```json
{
  "role": "press",
  "button_goal": {
    "kind": "press_staging",
    "projection_id": "the-fresh-press-wrist-projection-id",
    "alignment_phase": "final",
    "standoff_m": 0.055,
    "candidate_budget": 8
  },
  "plan_only": true
}
```

The runtime generates roll-equivalent press EEF candidates whose approach axis
is antiparallel to the measured outward button normal and whose origin is at the
requested non-contact standoff. CuRobo chooses a reachable collision-free
candidate. On success, execute the exact same goal once; do not split press
staging into artificial small segments. Re-observe the three views once after
the complete move.

Try the final 0.03--0.06 m standoff first. Only if every roll-equivalent
candidate fails real CuRobo plan-only reachability, use the same fresh
press-wrist projection for one farther 0.06--0.25 m observation staging goal
(normally 0.12--0.20 m). This remains button-goal-driven and non-contact. It is
not final pre-press geometry and cannot authorize `state_checkpoint_2`. Set
`alignment_phase="observation"` so the runtime uses the live wrist-camera
extrinsic to center the red bump; after the move, re-observe, back-project the
new frame, and retry the close goal with `alignment_phase="final"`.

During every executed CuRobo trajectory the runtime must force the dynamic held
gripper closed at every waypoint and include the attached radio in collision
checking. Endpoint review is consolidated: confirm the radio is still held and
the press hand has not touched it. Do not demand redundant per-waypoint
co-motion, attachment, contact, air-gap, or press-clear proofs.

Call `save_robot_state_checkpoint` for `state_checkpoint_2` only after fresh
press-wrist button projection and pre-press geometry show: approach-line
distance at most 0.010 m, opposition
error `angle(button_normal, -press_direction)` at most 15 degrees, and positive
0.03--0.06 m axial standoff. The same fresh `press_wrist` review must also show
the true raised red center inside the required image-center region. Use stage
`pre_press_alignment` and visual review.
If this composed system prompt also includes a separately supplied, explicitly
authorized stage-3 press system fragment, continue into that phase immediately
in the same evaluation after saving and verifying `state_checkpoint_2`; do not
ask for another review. Otherwise leave physics paused and wait for user
review. Record official `info["done"]["success"]` separately; it is not the
stage-2 success criterion.
