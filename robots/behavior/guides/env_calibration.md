# BEHAVIOR Public Perception and Calibration Guide

BEHAVIOR exposes synchronized public RGB-D from three physical cameras:
`head`, `left_wrist`, and `right_wrist`. A wrist-camera name identifies the
robot's anatomical side; it does not describe what that hand holds or which
interaction it should perform.

## Public frame lineage

Every observation carries a frame ID, capture group, environment step, attempt,
camera, and calibration metadata. A pixel or geometric claim is meaningful
only inside that lineage. Scene-changing actions invalidate affected evidence;
an observation from another step, attempt, or episode is not a substitute.

`pixel_to_world` consumes a public pixel in a current RGB-D frame and returns a
projection receipt. The runtime owns intrinsics, extrinsics, depth validation,
finite-value checks, workspace checks, and receipt lifetime. A `navigate_to`
target must be projected from a fresh current public head frame. The planner
must not reconstruct geometry from private simulator state or encode a pixel,
coordinate, pose, or layout as task knowledge.

## Cameras

- The head view supports scene interpretation and visual confirmation of the
  anatomical target EEF selected for an analytic primitive.
- Each physical wrist view can add local object or contact evidence when its
  current viewpoint makes that evidence useful.
- No camera has protocol priority or a required position in an observation
  schedule. The planner chooses views from current uncertainty.

An inconclusive view authorizes no geometric or hand claim. Acquire new public
evidence instead of completing the claim from a prior, a wrist-camera name
alone, a remembered hand selection, or hidden attachment information.

## Hand evidence

`move_to`, `rotate_wrist`, `open`, `close`, and `press` require a literal
`hand` equal to `left` or `right` and a `visual_hand_check` bound to the latest
synchronized public head capture at the current environment step. Its
`selected_hand` must equal the requested `hand`. Both hands may have current
attachments; attachments remain independent per-hand facts and do not create
or resolve semantic hand roles.

After the head frame authorizes one anatomical hand, the matching physical
`left_wrist` or `right_wrist` view may provide local RGB-D evidence. The wrist
name alone never authorizes hand selection.

For `move_to`, `rotate_wrist`, and `press`, the authorized `hand` names only the
target EEF. R1Pro whole-body 21-DOF CuRobo may jointly coordinate the base,
trunk, and both arms, and objects held by either hand participate in collision
checking. For `open` and `close`, `hand` continues to name only the target
gripper; their selected-gripper-only isolation contract is unchanged.

Evidence can be reused only while its environment step and runtime TTL remain
current. A real environment action requires a new head observation before
another analytic manipulation call. `pi0_nav_pick` is outside this evidence
contract: its public call requires `instruction` and `chunks`, while its
three-camera input, 23D action tensor, and VLA-controlled choice of both arms
remain unchanged.

`navigate_to` is also outside the literal-hand contract. It performs one
runtime-bounded pure-base stage either from a fresh head-frame projection or
from one explicit relative motion. Relative forward/backward translation uses
the body's call-start heading; relative left/right rotation turns the body with
the base in place and needs no projection. Arms, grippers, and trunk keep their
joint values relative to the base, and attachment identities remain unchanged.
Because admitted base motion changes the camera viewpoint, its return
invalidates the earlier scene geometry; obtain a fresh head observation before
relying on target identity or geometry to authorize another scene-changing
action. This local navigation-observation contract does not prescribe a global
tool sequence.

## Calibration failures

Missing depth, stale lineage, an expired or consumed projection, invalid
calibration, a hand mismatch, or inconsistent feedback is a fail-closed
precondition failure. It is recoverable unless the runtime explicitly declares
an unrecoverable infrastructure termination. None of these conditions is
official task success or, by itself, task failure.
