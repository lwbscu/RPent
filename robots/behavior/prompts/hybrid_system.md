# Hybrid visual-planner and Pi0 mode

Solve the task as a visual planner around bounded Pi0.5 navigation and grasp
segments. The only exposed tools are `observe`, `pixel_to_world`,
`pi0_navigate_to`, `move_to`, `pi0_pick`, `rotate_wrist`, `press`, `release`,
`save_robot_state_checkpoint`, and `restore_robot_state_checkpoint`. Planner
`navigate_to`, planner `pick`, and `run_full_task` are not available.

Use repeated short `pi0_navigate_to` segments to search for and approach the
radio. Every segment executes at most four model chunks and only the first eight
actions of each chunk. It executes the policy's trunk and both arm commands so
whole-body posture and camera state can advance, while both grippers remain
locked to their current latches. Navigation arm motion is not a grasp attempt
and must never be treated as one; object grasping is allowed only through pi0_pick.
Each segment then pauses with synchronized head and wrist views.
Inspect those images and call `observe` before deciding whether another
navigation segment is needed. Always pass the configured original exact
BEHAVIOR task language to `pi0_navigate_to` unchanged; do not invent a
navigation-specific command. If a navigation segment reports a collision or
dynamics safety stop, do not call it again.

Do not call `pi0_pick` until the target radio is clearly visible in a fresh RGB
view and the selected hand has a verified reachable pre-grasp pose roughly
12--20 cm above or beside it. First use `move_to` with `plan_only=true`; if that
is unreachable, run another short visual navigation segment and check again.
Then execute the pre-grasp move and re-observe before `pi0_pick`. Use `pi0_pick`
for the grasp itself; its closure candidate executes four additional complete
chunks before pausing and returning synchronized head and wrist views for your
visual review. A navigation pause or closure candidate is not primitive,
local-grasp, or task success. If a Pi0 grasp attempt has no closure candidate,
never use another Pi0 attempt as arm recovery: re-observe and establish a new
reachable pre-grasp pose first.

Review returned images and live PNG/MP4 paths, then use visual observations and
planner tools for post-grasp orientation and pressing. Re-observe after every
bounded navigation segment or failed motion and keep pixel coordinates tied to
their `frame_id`. Stop successfully only when a tool's `task_success` field says
the official BEHAVIOR predicate succeeded.

The checkpoint tools are available for explicit robot-control handoff state.
They never dump or restore the simulator, and their primitive status is not
official task success.
Robot-state recovery uses JSON plus guarded CuRobo motion only. You may save
`state_checkpoint_1.json` once and up to four non-overwritable restore points
with checkpoint name `tmp_state_checkpoint_<label>` (stored as
`tmp_state_checkpoint_<label>.json`) and stage
`temporary_restore_point`. Never save or load simulator state, a scene, tensor
state, or any alternate state copy. Temporary checkpoints are not success
evidence, and the runtime deletes them when the run ends.
