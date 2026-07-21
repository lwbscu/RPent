# Full-task VLA mode

Call `BehaviorPrimitives.run_full_task` through the `run_full_task` tool exactly
once. It synchronously runs the configured Pi0.5 policy until official task
success, environment termination/truncation, the episode horizon, or an error.
Do not call lower-level robot tools: none are exposed in this integration mode.

Treat only `task_success` from the tool as evaluation success. Reward,
termination, truncation, and local progress are not success signals.
