# Local Pi0 grasp mode

Call `BehaviorPrimitives.pi0_pick` through the `pi0_pick` tool exactly once.
This is a local Pi0.5/VLA grasp loop, not a full-task run and not the planner
pick. It executes validated 23D whole-body action chunks while actual
selected-hand gripper proprio is monitored at every simulator step, only until
a local grasp validator accepts the candidate, an official environment stop,
the local chunk limit, or an error.

A closure candidate is recorded but is neither a stop condition nor proof that
the object was picked: inspect the saved MP4 before accepting the grasp. Never
infer official task success from `primitive_success` or `local_grasp_success`;
only `task_success` mirrors raw `info.done.success`.
