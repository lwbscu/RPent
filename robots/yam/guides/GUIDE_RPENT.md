# YAM RPent Guide

YAM uses three views: `top`, `left`, and `right`. The left and right cameras are
wrist cameras. The state and action layout is absolute `qpos14`:
`[left_q0..q5, left_gripper, right_q0..q5, right_gripper]`, with grippers
normalized as `0=closed, 1=open`.

Use `view_env_state` first. The top image is the global semantic view. Wrist
images refine grasp geometry for the same selected object; do not silently switch
to a look-alike visible in a wrist camera. `sample_world_xyz` and
`query_world_map` read persisted same-frame depth projections in the YAM
left-base frame.

Without a VLA endpoint, both evaluation and Explore use primitives only;
`pi05_act` is absent from the available tools. Once a trained endpoint is supplied,
`pi05_act` runs that policy; keep chunks short on the real robot.
`move_to` delegates reachability,
IK, table protection, and waypoint generation to the env server, then executes
the full returned waypoint list.

The dual-arm names follow RoboTwin: `move_to(arm="left"|"right", xyz=...,
quat=...)`, `rotate_wrist(arm=..., delta_yaw_deg=...)`,
`set_gripper(arm=..., val=..., steps=10)`, and `release(arm=..., steps=10)`.
Open with `val=1` (or `release`); close with `val=0`. There are no separate
`open`/`close` aliases. LIBERO's signed controller gripper actions are not YAM
gripper values. `gripper_val` reports measured position; primitive execution
success does not prove a secure grasp.

Each geometric action selects one arm and holds the other arm at the measured
qpos captured at the start of the chunk. The control machine executes all
waypoints at 30 Hz with per-step feedback and stop handling. This differs from
RoboTwin's per-waypoint Agent RPC, and deliberately does not expose its path
subsampling option. Two arms can work in observed alternating phases, but there
is no simultaneous dual-arm Cartesian primitive or coordinated collision planner.

All xyz targets use `left_base` as the shared world frame, in metres. Pose
quaternions are **wxyz**. Each arm's native FK has its own base; the server
converts right-arm poses through the calibrated base-to-base transform.
The current YAM grasp convention reported from the physical rig is Y_site
across the closing fingers, -Z_site the approach direction, and a grasp site
98 mm along the gripper body's -Z. A strict vertical top-down grasp was reported
unreachable in the validated setup; prefer the measured forward/down approach.
Do not invent a new TCP offset on the Agent side. The supplied quaternion is
an exact request: the planner may fail instead of silently changing orientation.

The current geometric guard checks sampled TCP clearance above the configured
table. It does not cover arm links, self-collision, two-arm collisions, held
objects, fixtures, or force/contact limits. Start with one arm in a cleared
workspace and keep the other arm in its operator-confirmed staging area.
When a view reports `world_xyz_limitation`, inspect it before attempting pixel
localization. Invalid depth or an unaligned wrist frame is not a usable target.

For free-space manipulation, verify a hold before transport. For contact-rich
grasping, re-grasp, insertion, tool use, or bimanual coordination, proceed only
within demonstrated primitive capabilities or a connected trained VLA's validated
capabilities. An untrained or absent VLA is not a fallback. Re-observe
after every motion and protect already achieved task relations.

Success is not inferred from an agent statement or primitive return. `finish`
checks the latest env `eval_success` flag. During early real-robot exploration,
that flag is expected to come from an operator receipt or a station-specific
success checker exposed by the env server.

Evaluation has one operator-prepared attempt and read-only memory. Exploration
can request reset only after an operator has restored the scene and written a
fresh ready receipt for the advertised episode ID. The operator's `start` command
is for the first episode before Agent recording; `ready` only writes the receipt
for the Agent's retry reset. Reset does not home, fold,
clear motor faults, or turn torque off. Never write the operator receipt file,
call a hidden API, or substitute an Agent judgement for an operator verdict.
Archive failed attempts; export only the current successful attempt's recipe.
