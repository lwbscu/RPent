"""Task-specific prompt nodes for ``picking_up_trash``."""

from __future__ import annotations

from typing import Final

from rpent.context.prompt_utils import PromptNode

PROMPT_PROFILE_ID: Final = "picking_up_trash"

TRASH_OBJECT_HANDLING: Final = """Prompt profile: `picking_up_trash`.
For the narrow act of releasing an attached can, the rules below are mandatory
task-specific safety preconditions; they do not establish general tool priority.
For each placement, use fresh current head RGB to select the literal anatomical
`left` or `right` hand controlling the intended soda can; never hard-code a side
from task knowledge. Pass that physical side as `hand` to each analytic
primitive. Either or both hands may have attachments, and this does not make
literal hand selection ambiguous. Keep the selected can hand closed and
preserve its same attachment throughout transport. The other arm, its gripper,
and anything it holds remain stationary."""

TRASH_RECEPTACLE_GROUNDING_AND_NAVIGATION: Final = """Ground the receptacle and
verify a safe local base stage; relocate only when needed. Operator review of s2
established a general failure mode: a returned VLA image may visibly contain the
trash receptacle while the policy has not semantically grounded it. Mere
presence in the image, or motion generally toward it, does not establish the
receptacle's task identity, its opening, or a safe base target.

When `pi0_nav_pick` is useful, give one invocation exactly one currently visible
search or navigation subgoal grounded in current evidence. Do not combine
search, navigation, grasp, transport, and release into one end-to-end
instruction.
When an invocation returns and raw official success has not been latched,
obtain a fresh `observe(camera="head")` before the next scene-changing action;
every applicable reassessment rule below remains mandatory.

For every Trash `pi0_nav_pick` invocation, choose `chunks=N` explicitly and
adaptively. Estimate the smallest positive number of complete 32-action chunks
needed for only the current single subgoal. Separately estimate and reserve
episode action steps for the shortest currently justified scripted finishing
and recovery plan, including base-navigation, local transfer, release, and
correction actions. Convert only the unreserved action remainder into a
complete-chunk budget. Invoke Pi0 only when the single-subgoal estimate fits
that budget, and pass that estimate as `N`; do not truncate `N` merely to
consume the remaining budget. If the estimate does not fit, narrow the subgoal
only when current evidence supports a smaller, independently useful and safe
observation boundary; otherwise switch to scripted tools or keep task motion
frozen while obtaining non-scene-changing observations. If a safe reserve
cannot be estimated from current evidence and runtime accounting, do not invoke
Pi0.

Never choose `N` merely from attachment count, a legacy chunk value, a fixed
public seed, a remembered instance layout, or `primitive_success` from a prior
call. Adapt `N` only between invocations after fresh public evidence; the
planner cannot observe or replan inside an admitted invocation. Do not
automatically repeat or increase the previous `N`: increase the next horizon
only after a fresh frame proves monotonic, low-risk progress toward the same
subgoal, and reduce it or switch tools when progress is absent, ambiguous, or
physically risky.

A narrow two-held-cans search rule applies only when fresh runtime attachment
facts report that `left` and `right` each currently holds an attachment that the
fresh head evidence identifies as a required task soda can, while that same
fresh head frame has not yet identified the task receptacle's body, rim, or
opening. In exactly that state, Pi0 may perform one bounded receptacle-search
subgoal using the global adaptive horizon rule above.
The instruction must preserve both attachments, keep both grippers closed, and
forbid releasing, placing, dropping, swapping, or re-grasping either can.

After the bounded invocation returns, immediately call
`observe(camera="head")` and reassess the task residual from that fresh frame.
Do not issue a second `pi0_nav_pick` without this intervening fresh observation;
consecutive VLA calls are forbidden in this state.

If fresh evidence shows that the trash receptacle is loaded or may contain a
previously placed can, or that a held can is at or near the receptacle rim, do
not use Pi0 for any search, navigation, or local correction. First obtain a
fresh `observe(camera="head")`. If the loaded or contents-uncertain receptacle
is hand-held, keep the base fixed as well: do not call `navigate_to` until fresh
evidence proves the receptacle is no longer hand-held and is stably supported.
Use only a locally justified analytic lift, transfer, correction, or release
action that preserves or secures the existing containment. If the remaining
task cannot be completed locally without moving a hand-held loaded receptacle,
keep task motion frozen rather than risk spilling completed placements. While
raw official success remains false, use only fresh non-scene-changing
observations to seek evidence that resolves the uncertainty until a safe local
analytic action or a trusted stop condition is available; do not declare
completion or impossibility on your own. Preserve the receptacle, every
contained or possibly contained can, and every current hand attachment while
recovering.

As soon as fresh head RGB shows enough of the task receptacle's body, rim, or
opening to identify it, stop blind Pi0 search or navigation toward it. The opening
need not be perfectly centered or completely visible to establish receptacle
identity and ground a base-navigation target. If base relocation is needed
before safe local transfer, select an interior, stable visible point on the
receptacle away from silhouette and depth discontinuities, then project that
current point with `pixel_to_world`. Pass the fresh receipt to `navigate_to`
with a standoff chosen from current visible clearance. Use the public tool's
default `timeout_s=300` by omitting `timeout_s`; if an explicit timeout is
required by current runtime accounting, choose one sufficient for planning and
execution and never force `timeout_s=20`. The runtime selects a traversable base
station around the projected target and executes one bounded pure-base stage.
Use current evidence to prefer an approach that keeps the robot and any held
object clear of the receptacle, furniture, and other obstacles. Visual evidence
alone does not certify the path. Runtime traversability and isolation guards
remain authoritative, and no visual or traversability check guarantees
clearance from all dynamic or held geometry.
After navigation returns, immediately obtain another fresh
`observe(camera="head")`; the base motion invalidates the earlier viewpoint and
geometry. If another base stage is justified by the new evidence, select and
project a new visually plausible clear candidate rather than reusing prior
geometry. If the base is already safely staged for local transfer, do not
navigate merely because the receptacle has been identified.

When fresh head evidence, either before relocation or after a navigation stage,
shows that the residual is local, use the ordinary task-specific lift, transfer,
and release safeguards below. Select whichever literal anatomical hand
currently controls the intended can from fresh head evidence, lift for current
clearance, move that hand and can above the visually grounded opening, visually
confirm the release condition from a new head frame, and only then open that
same hand.

This conditional safety flow encodes no fixed route, waypoint, room or landmark
order, physical hand, pixel, coordinate, base pose, standoff, Pi0 invocation
count, or fixed `chunks` value. Derive every target, stage, hand, and VLA horizon
from current public evidence, runtime guards, the single current subgoal, the
scripted finishing and recovery reserve, and remaining accounting."""

TRASH_PREGRASP_DEPTH_CONFIRMATION: Final = """Confirm depth before grasping.
Before calling `close` on a soda can, obtain current public evidence that clearly
shows the selected gripper and the intended can grasp surface. Prefer a current
`left_wrist` or `right_wrist` view matching the hand already selected from
fresh head evidence; if that target is occluded, use another useful current
public view rather than guessing geometry.

Inspect the fresh RGB, select an interior pixel on the visible can grasp
surface, and obtain a `depth_probe` from that exact camera and frame. For the
physical wrist camera matching the selected hand, use its frame-bound
`target_point_camera_xyz_m`, `target_to_palm_m`,
`target_to_grip_point_m`, and `target_to_finger_roots_m` as approximate
geometric guidance for whether another cautious refinement is needed. A head
probe intentionally has no hand-relative distances. Visual centering, scalar
depth, and the runtime-computed distances are all insufficient to close by
themselves: they do not verify object identity, clearance, contact, or a safe
gripper command.

If fresh RGB-D evidence indicates that the selected gripper remains too high,
request only a small, predominantly downward world-Z relative correction for
that same hand, chosen from current clearance. Do not derive its displacement
directly from camera depth or range. After any correction, obtain fresh head
evidence for the analytic hand authorization and fresh target RGB-D evidence
before `close`. Never prescribe a fixed hand, wrist-camera side, pixel, depth
threshold, coordinate, descent distance, trunk displacement, or instance
layout. If a collision-certified whole-body `move_to` is unreachable, do not
request an arm-only or trunk-assisted fallback. Re-observe and re-ground; if
the target is remote and a safe fresh head projection exists, use
`navigate_to` for the separate pure-base approach before retrying a local
whole-body target."""

TRASH_LIFT_BEFORE_TRANSFER: Final = """Lift before transfer. When the selected
hand holds a can at a low position, do not move it directly or diagonally toward
the trash can. First call `move_to` for that same hand with a small, predominantly
upward positive world-frame Z relative translation, using a target shaped as
`{"delta_xyz": [0, 0, <positive_z>], "frame": "world"}` and selecting the lift
from current visual clearance. After the lift action, obtain a fresh
`observe(camera="head")`.
Before any lateral transfer, visually confirm that the complete hand and held
can clear nearby furniture, the receptacle rim, the other hand, and anything
held by the other hand. If clearance is uncertain, do not move sideways; obtain
better current evidence or make another small upward move.

From head camera view, locate the center of the receptacle opening and use
`pixel_to_world` on that center. Move the same holding hand toward that fresh
projection with a safe standoff so the complete held can is above the opening
center. Approach from above; do not sweep the hand or can across the
receptacle's side wall or rim. This move-to-above-opening step does not itself
authorize `open`; a new post-motion head frame must provide the release evidence
below. The strict opening-containment check belongs only to that fresh
post-transfer release gate immediately before `open`; it is not a prerequisite
for identifying the receptacle, projecting a navigation target, calling
`navigate_to`, lifting, or transferring a held can."""

TRASH_RELEASE_CONFIRMATION: Final = """Confirm before release. A successful
transfer move changes the scene, so obtain another fresh
`observe(camera="head")`. Call `open(hand="<left or right>")` for the same
physical hand only when that fresh RGB visually confirms that the same hand
still holds the same can, the complete can is fully inside the receptacle
opening's circular rim (rather than outside or overlapping the rim), and it has
a clear downward path into the receptacle.

When `open` would release an attachment, pass both the ordinary
`visual_hand_check` and:

```json
"release_visual_check": {
  "camera": "head",
  "frame_id": "<the same fresh head frame_id>",
  "selected_hand": "<the same left or right value as hand>",
  "assessment": "attached_object_fully_inside_receptacle_opening"
}
```

The release check must cite exactly the same frame and physical hand as
`visual_hand_check`. If any part of that confirmation is uncertain, do not
open; re-observe or correct the above-opening position first. Exact depth and
hand-distance outputs remain guidance only and cannot substitute for this
semantic release confirmation. This lift-transfer-confirm rule applies only
when `open` would release an attachment; opening an empty hand does not require
`release_visual_check`."""

PICKING_UP_TRASH_PROMPT_NODES: Final[PromptNode] = (
    TRASH_OBJECT_HANDLING,
    TRASH_RECEPTACLE_GROUNDING_AND_NAVIGATION,
    TRASH_PREGRASP_DEPTH_CONFIRMATION,
    TRASH_LIFT_BEFORE_TRANSFER,
    TRASH_RELEASE_CONFIRMATION,
)

TRASH_PROMPT_NODES: Final[PromptNode] = PICKING_UP_TRASH_PROMPT_NODES

__all__ = [
    "PICKING_UP_TRASH_PROMPT_NODES",
    "PROMPT_PROFILE_ID",
    "TRASH_LIFT_BEFORE_TRANSFER",
    "TRASH_OBJECT_HANDLING",
    "TRASH_PREGRASP_DEPTH_CONFIRMATION",
    "TRASH_PROMPT_NODES",
    "TRASH_RECEPTACLE_GROUNDING_AND_NAVIGATION",
    "TRASH_RELEASE_CONFIRMATION",
]
