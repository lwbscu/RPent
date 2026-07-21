# Post-prepress button-press phase

This phase is valid only after the user has explicitly approved the reviewed
`state_checkpoint_2` and authorized stage 3. It starts from that exact
checkpoint and ends as soon as the button center is visibly green and raw
`info["done"]["success"]` reports true. It never authorizes another navigation
or grasp call.

During stage 3 you may create a same-run temporary CuRobo restore point with
checkpoint name `tmp_state_checkpoint_<label>` (stored as
`tmp_state_checkpoint_<label>.json`) and stage `temporary_restore_point` before
a risky alignment. Keep at most four, never overwrite one, and never treat one
as task success. Restore only through `restore_robot_state_checkpoint`; scene,
simulator-snapshot, tensor-state, and alternate state-copy save/load are
forbidden. The runtime deletes every temporary checkpoint when the run ends,
while preserving only `state_checkpoint_1.json` and `state_checkpoint_2.json`.

Use this fragment only with a stage-3 runtime that actually exposes and routes
the named composite handlers below. If any handler is absent, if the checkpoint
lineage cannot be verified, or if the current tool surface is still the bounded
stage-2 surface, stop fail-closed instead of emulating a press with generic
tools or literal joint / EEF commands.

## Resume and provenance gate

Inspect the authoritative `state_checkpoint_2.json` before motion. Verify its
path and SHA, `stage="pre_press_alignment"`, current task / instance / seed,
object identity, and its exact fresh `press_wrist` gate and projection lineage.
Read `held_hand` and `press_hand` dynamically from the checkpoint/runtime;
never turn this run's right/left assignment into a general rule. Resume only by
planning and executing a guarded CuRobo motion from this run's checkpoint JSON;
never load or save a simulator snapshot or scene copy.

Do not call `pi0_nav_pick`, `pi0_pick`, or `pi0_navigate_to` again. Do not reset
or teleport the simulator. A robot-motion restore is not a simulator reset.
Before stage-3 motion, require the radio to remain stably held, both held-hand
finger attachments to remain present, and the radio not to be dropped or flat
on the table. The held gripper remains closed for the entire phase.

## Fresh visual and geometry gate

Start from the reviewed state-2 geometry, but treat any camera frame or
projection as stale after a restore, gripper command, or motion. Obtain a fresh
dynamic `press_wrist` frame. The actual raised red center bump, black disk, and
white outer ring must be visible; never substitute a side port, slot, seam,
shadow, or historical pixel. Back-project the red center from that same frame.

Aim the closed fingertip as close as possible to the raised red center. The
visual contact gate is satisfied when the fingertip lies inside the black disk
and is visibly near the red center; an abstract EEF-axis or link-center distance
must not veto an otherwise clear fingertip-on-button alignment. Conversely,
extra axial depth cannot repair a visible lateral miss.

Before contact, call `post_pick_close_press_gripper` and require the dynamic
press gripper to be fully closed. Command zero opening and require the measured
opening to reach the physical closed limit; latch at most `-0.99` and measured
opening at most `0.003 m` are only fail-closed verification bounds, not
permission to intentionally leave a partial opening. Closing invalidates the
old camera/projection binding, so observe, declare, and project again afterward.
Never begin contact with an open or merely partly closed press gripper.

## Contact strategy learned from the successful run

The simulator toggle is not triggered by requested travel distance alone. It
requires both real robot-finger contact with the radio and overlap between a
finger collision body and the button marker for the runtime-reported number of
consecutive physics updates; that value was five in the successful run. Do not
repeat blind axial advances when the red point stays laterally off the fingertip.

When live evidence shows that axial motion preserves a lateral miss or makes the
radio and press hand co-move, lock the base and trunk during the final relative
alignment. Keep the fully closed press hand stationary when appropriate and,
only when the live handler reports this correction applicable, use
`post_pick_recenter_held_button` to move the held radio/button relative to that
fingertip. This was the successful run's decisive correction: the dynamic held
arm brought the button marker onto the stationary closed press fingertip. The
pressing contact was still made by the dynamic press-hand fingertip even though
the final relative closing motion came from the held arm.

This prompt is not authority to disable collision checks. A final
contact-seeking CuRobo handler may apply only the collision policy already
machine-scoped by an explicitly authorized stage-3 runtime. Continue enforcing
dynamic-role binding, finite joint targets, joint and dynamics limits, a closed
held gripper, a fully closed press gripper, stable radio attachment, and bounded
travel. Do not globally disable these guards or emulate the handler.

After any alignment motion, check raw official success before issuing another
motion. If it is still false, obtain a new `press_wrist` frame and projection.
Use `inspect_toggle_geometry` only as diagnostic evidence; finger-link center
distance is not the collision-surface distance and is not a hard press gate. If
the fingertip is not visibly inside the black disk near the red bump, realign
laterally rather than pushing farther. If the visual and physical alignment are
current but the toggle still has not fired, a bounded
`post_pick_direct_finger_toggle` may move the nearest fully closed fingertip
through the live marker and hold contact; never reuse a stale projection.

## Success, green-frame synchronization, and stop

Official success is only `task_success=true` sourced from raw
`info["done"]["success"]`. Contact counts, CuRobo completion, red-point motion,
`primitive_success`, or local press completion are not substitutes.

When official success first becomes true, do not press again. Hold the robot
stationary for a bounded render-synchronization window, normally
`post_success_hold_frames(frames=4)`, and obtain a fresh dynamic `press_wrist`
image. The center marker must be visibly green. The state transition can precede
the rendered color by several frames; this delay is camera/material
synchronization, not a second toggle.

Once both conditions are present -- the center marker is visibly green and raw
official success is true -- immediately pause environment stepping. Do not
retreat the press hand and do not open either gripper in this terminal workflow.
Preserve the final head / held-wrist / press-wrist images, raw result fields,
final environment step, motion trace, and state-2 SHA for review. A later zero
consecutive-contact counter does not undo a latched
`ToggledOn=true` or official success.

If the radio drops, becomes flat, the dynamic roles or checkpoint lineage
change, the fresh red-center gate is lost, a required handler is unavailable,
or official success remains false after the bounded evidence-driven contact,
pause and report the exact evidence. Never claim success, repeat navigation or
grasping, or continue with unbounded blind presses.
