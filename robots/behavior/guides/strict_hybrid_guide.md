# Strict BEHAVIOR Hybrid Execution Guide

This guide defines invariants shared by BEHAVIOR Explore and Eval. Capability
schemas and runtime guards are authoritative.

## Perception isolation

Use only public RGB-D, public calibration and lineage, non-privileged robot
state, and public capability returns. Do not inspect task initialization files,
scene graphs, simulator tensors, ground-truth object poses, hidden contacts,
hidden attachments, segmentation identifiers, or private environment metadata.

Task priors and reviewed memory describe semantics and failure categories. They
cannot authorize a current action or carry an episode-specific hand, camera,
pixel, coordinate, pose, or object layout into a new run.

## Analytic manipulation

`move_to`, `rotate_wrist`, `open`, `close`, and `press` require a literal
`hand` equal to `left` or `right`. Every call requires a fresh current public
head frame whose `visual_hand_check.selected_hand` exactly equals the requested
hand.

Either or both hands may have attachments. Attachment multiplicity does not
make literal physical-side selection ambiguous. Subject to each primitive's
other guards, a fresh-head-authorized hand may be addressed independently.

For `move_to`, `rotate_wrist`, and `press`, `hand` identifies only the target
EEF. R1Pro whole-body 21-DOF CuRobo jointly plans the base, trunk, and both arms;
none of those body groups is semantically isolated merely because one target EEF
was selected. Objects held by either hand are included in collision checking
throughout planning and execution.

`open` and `close` retain their selected-gripper-only contract: the base, trunk,
both arms, and the non-target gripper remain unchanged. Missing feedback,
controller-slot mismatch, collision-state inconsistency, or an unexpected
attachment change stops the analytic action before another step and returns a
recoverable failure.

## Analytic base navigation

`navigate_to` accepts either a fresh projection grounded in a current public
head frame, or one explicit relative base motion. Relative translation moves
straight forward or backward along the body's heading at call start; relative
rotation turns left or right in place and needs no projection. The two modes
are mutually exclusive. Both arms, both grippers, and the trunk keep their
joint values relative to the base, so the body moves and rotates with it, while
attachment identities remain unchanged. Navigation does not establish semantic
target identity or final arrival. Because admitted base motion changes the
viewpoint and invalidates earlier geometry, obtain a fresh public head
observation before using target identity or geometry to justify another
scene-changing action.

This is a capability-local evidence and isolation contract, not a fixed tool
sequence. Conditional task-specific safety guidance may require base staging
before a local manipulation when current evidence makes that flow applicable.

## Pi0 execution

`pi0_nav_pick` does not accept `hand`, `role`, or `visual_hand_check`. It
requires `instruction` and `chunks=N`, where the planner chooses a positive
integer N from the current subgoal and remaining episode accounting. `chunks`
has no fixed maximum and is not a per-tool usage quota. Calls that cannot fit
`32 * N` into the remaining episode-step budget are rejected before motion.

The VLA retains control of both arms through the unchanged action schema. An
admitted call normally executes exactly N complete `[32,23]` chunks, each
containing all 32 actions, while raw official success remains absent. Raw
official success stops physical execution at the exact successful environment
step; a receipt-bound partial chunk is a normal successful terminal outcome
and is not counted as a complete chunk. Attachment detection and controller
handoff alone cannot shorten the requested count. Other early returns require
a real environment termination or truncation, or an explicit runtime safety or
infrastructure exception.

## Task isolation

Task-specific terminal and surface policies are read only from the selected
TaskSpec. A policy registered for one task does not become a generic visual
failure, action restriction, prior, or publication rule for another task.
State, memory, Recipe, receipt, manifest, and publication namespaces follow the
selected task's tag and compound identity.

The shared Pi0.5 checkpoint is one cross-task profile. A run must validate the
expected binding during job preflight and must not inherit a task-specific
checkpoint from ambient configuration.

## Terminal semantics

Official success is only raw `info["done"]["success"] == true`, surfaced as
`task_success == true`. Keep task success, primitive success, terminal evidence
quality, workflow completion, and publication completion independent.

The runner owns termination and artifact sealing. Failure or infrastructure
abnormality never produces a success Recipe or Task Memory. A visual checkpoint
is an observation artifact, not simulator rollback, task completion, or motion
authorization.
