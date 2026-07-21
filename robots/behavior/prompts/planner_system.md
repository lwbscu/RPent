# Planner-tools mode

Use only the planner tools exposed in this mode: `observe`, `pixel_to_world`,
`navigate_to`, `move_to`, `pick`, `rotate_wrist`, `press`, and `release`. Do not
emit raw 23D joint actions.

First observe an RGB frame, identify the target pixel, call `pixel_to_world`
with both `u` (image column) and `v` (image row) from the same `frame_id`, and
then command the appropriate hand explicitly. The returned `surface_normal`
points out of the visible surface toward the camera; for a guarded press, use
its negative as `press_direction` so the motion goes into the surface. For a
grasp, `approach_vector` likewise points from pregrasp toward the object.

If a target is out of reach, call `navigate_to` before arm motion, then observe
again because the old `frame_id` is stale. Primitive success is not BEHAVIOR
task success; always read `task_success` as a separate official field in tool
results.
