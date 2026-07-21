# Continuous Pi0 navigation-and-grasp mode

Call `BehaviorPrimitives.pi0_nav_pick` through the `pi0_nav_pick` tool exactly
once. This is one continuous Pi0.5/VLA navigation-to-grasp loop. It predicts and
executes only complete 32-step action chunks and does not call or emulate
`pi0_navigate_to`, `pi0_pick`, planner tools, or `run_full_task`. The policy
chooses its own navigation trajectory and hand; do not provide a hand, gripper
threshold, validation window, or local chunk limit.

The env-side fail-closed validator may stop inside a chunk only after it has
verified a real held object, saved the post-pick robot-motion checkpoint, and
handed off a paused runtime. This checkpoint is not a simulator snapshot. Treat
`primitive_success` / `local_grasp_success` only as this local
navigation-and-grasp outcome. It is not BEHAVIOR evaluation success. Treat only
`task_success`, sourced from raw `info.done.success`, as official task success.
The returned MP4, validator trace, and synchronized head/wrist images still
require independent visual review.

The successful handoff itself produces `state_checkpoint_1.json` with the
dynamically selected held and press hands. Do not overwrite it. Neither that
robot-motion checkpoint nor later pre-press evidence can establish official
task success.
