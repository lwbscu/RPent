# BEHAVIOR Explore and Evaluation Guide

BEHAVIOR uses one perception-isolated hybrid driver for task-local Explore and
Eval cells. The shared runner, Codex SDK, Dashboard, public toolkit, artifact
framework, and Pi0.5 checkpoint do not erase task identity.

## Public cell identity

A cell is resolved through its `BehaviorTaskSpec`. Task identity uses the
compound key `(task_name, activity_definition_id, activity_instance_id)`.
Public seed mappings, state directories, target priors, reviewed experience,
terminal policies, Recipes, and publication namespaces remain task-local.
A native instance number alone never identifies a task or its public status.

The authoritative task language is injected by the runner. Current observations
must establish object identity, geometry, target-EEF selection, and action
authorization; task memory cannot provide current measurements.

## Explore

The outer harness owns fresh Explore attempts. Each attempt has a new reasoning
context, environment process, episode, evidence lineage, and bounded tool and
wall-clock budget. Prior-attempt summaries are bounded, de-instantiated
forensic hints rather than observations or executable instructions.

Explore may publish symbolic Recipe and Task Memory artifacts only after the
runtime has observed raw official success. Failure, operator stop, unknown
outcome, budget exhaustion, or infrastructure termination remains an audit or
failure-pool result and cannot produce success publication.

## Evaluation

Eval is one fresh episode with frozen, runtime-verified Explore memory. Frozen
guidance is advisory: it cannot prescribe primitive order, camera schedule,
model instruction, invocation count, hand, pixel, coordinate, pose, instance
layout, or hidden state. Eval never changes or extends the frozen publication.

## Hybrid capability surface

Exactly ten public primitives are exposed:

- `close`
- `move_to`
- `navigate_to`
- `observe`
- `open`
- `pi0_nav_pick`
- `pixel_to_world`
- `press`
- `rotate_wrist`
- `save_robot_state_checkpoint`

They are peer capabilities. List order and grouping carry no execution meaning.
The planner selects among them from current evidence and runtime guards. Except
for capability-local preconditions and conditional task-specific safety flows,
this list prescribes no tool sequence.

`navigate_to` accepts either a fresh frame-bound projection from a current
public head view, or one explicit relative base translation or rotation.
Relative translation moves forward or backward along the body's call-start
heading; relative rotation turns left or right in place and needs no
projection. The modes cannot be mixed. Arms, grippers, and trunk keep their
joint values relative to the base, so the body moves or rotates with it, and
attachment identities remain unchanged. Its return neither proves final
arrival nor preserves the earlier view: obtain a fresh head observation before
using target identity or geometry to authorize another scene-changing action.
This evidence contract is local to a navigation stage, not a mandatory global
schedule.

Every `pi0_nav_pick` call requires `instruction` and `chunks=N`. The planner
chooses the positive integer N from the current subgoal and remaining episode
accounting. `chunks` has no fixed maximum and is not a per-tool usage quota.
Runtime rejects the whole call without motion when `32 * N` cannot fit the
remaining episode-step budget.

Once admitted, `chunks=N` is an upper requested work bound. If raw official
success and the allowed terminal exceptions remain absent, Pi0 executes exactly
N complete chunks in the unchanged `[32,23]` action shape. Every completed chunk
executes all 32 actions. Attachment observations and controller handoff alone do
not shorten that normal count.

Raw official success stops physical task execution at the exact successful
environment step, including mid-chunk. The successful action's returned
observation and immutable receipt may be recorded, but no later action,
prediction, public-tool call, observation, capability read, or other task RPC
is allowed. Only no-action VLA disable/health, environment freeze/finalize,
transport shutdown, and artifact sealing remain allowlisted. A receipt-bound
success partial is a normal terminal outcome and is not counted as a complete
chunk. Other early returns require a real environment termination or
truncation, or an explicit runtime safety or infrastructure exception.
`move_to`, `rotate_wrist`, and `press` use R1Pro whole-body 21-DOF CuRobo joint
planning. Their literal `hand` selects only the target EEF; it does not select an
isolated arm-only embodiment. Planning may coordinate the base, trunk, and both
arms, and objects held by either hand participate in collision checking.

The fresh visual hand contract remains required. `open` and `close` remain
selected-gripper-only primitives; `navigate_to` remains pure-base; perception,
checkpoint, and Pi0 semantics are unchanged.

## Success and continuation

The only official success source is raw `info["done"]["success"] == true`,
reported publicly as `task_success == true`. A grasp, attachment, contact,
primitive result, checkpoint, visual change, model completion, or reward cannot
replace it. Success first observed inside an admitted Pi0 call is latched
immediately, and runner termination plus the freeze on later task work take
effect at that successful environment step. Runtime-only disable, health, and
paused-runtime finalization calls may complete without task motion.

While execution remains available and official success is false, the planner
continues toward another legal tool call unless a trusted stop condition from
the task-specific prompt and runtime is established. A rejected or
inconclusive action is a reason to re-ground, not an implicit terminal result.
