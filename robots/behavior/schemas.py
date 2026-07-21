"""Validated BEHAVIOR/R1Pro observation and action contracts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ACTION_DIM = 23
DEFAULT_ACTION_CHUNK = 32
CAMERA_KEYS = ("main", "left_wrist", "right_wrist")
FULL_TASK_VLA_MODE = "full_task_vla"
PLANNER_TOOLS_MODE = "planner_tools"
PI0_PICK_VLA_MODE = "pi0_pick_vla"
HYBRID_VLM_PI0_MODE = "hybrid_vlm_pi0"
PI0_NAV_PICK_VLA_MODE = "pi0_nav_pick_vla"
CONTROL_MODES = (
    FULL_TASK_VLA_MODE,
    PLANNER_TOOLS_MODE,
    PI0_PICK_VLA_MODE,
    HYBRID_VLM_PI0_MODE,
    PI0_NAV_PICK_VLA_MODE,
)
VLA_CONTROL_MODES = (
    FULL_TASK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    HYBRID_VLM_PI0_MODE,
    PI0_NAV_PICK_VLA_MODE,
)
PLANNER_TOOL_NAMES = (
    "observe",
    "pixel_to_world",
    "navigate_to",
    "move_to",
    "pick",
    "rotate_wrist",
    "press",
    "release",
)
ROBOT_STATE_CHECKPOINT_TOOL_NAMES = (
    "save_robot_state_checkpoint",
    "restore_robot_state_checkpoint",
)
POST_PICK_TOOL_NAMES = (
    "inspect_post_pick_state",
    "observe",
    "declare_button_visibility",
    "pixel_to_world",
    "evaluate_prepress_geometry",
    "move_to",
    "rotate_wrist",
    *ROBOT_STATE_CHECKPOINT_TOOL_NAMES,
)
HYBRID_TOOL_NAMES = (
    "observe",
    "pixel_to_world",
    "pi0_navigate_to",
    "move_to",
    "pi0_pick",
    "rotate_wrist",
    "press",
    "release",
    *ROBOT_STATE_CHECKPOINT_TOOL_NAMES,
)

# Pi0.5 compacts raw R1Pro proprio in this order. In particular, both arms
# precede both grippers.
POLICY_STATE_SEGMENTS: dict[str, slice] = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "left_arm": slice(7, 14),
    "right_arm": slice(14, 21),
    "left_gripper": slice(21, 22),
    "right_gripper": slice(22, 23),
}

# OmniGibson's R1Pro action contract places each gripper after its arm. This
# must never be replaced with POLICY_STATE_SEGMENTS.
ENV_ACTION_SEGMENTS: dict[str, slice] = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "left_arm": slice(7, 14),
    "left_gripper": slice(14, 15),
    "right_arm": slice(15, 22),
    "right_gripper": slice(22, 23),
}

# Raw R1Pro proprio indices used by RLinf's pi05_behavior input transform.
RAW_PROPRIO_SEGMENTS: dict[str, slice] = {
    "left_arm": slice(158, 165),
    "left_gripper": slice(193, 195),
    "right_arm": slice(197, 204),
    "right_gripper": slice(232, 234),
    "trunk": slice(236, 240),
    "base": slice(253, 256),
}


def _validate_segments(name: str, segments: Mapping[str, slice]) -> None:
    covered: list[int] = []
    for segment, indices in segments.items():
        if indices.start is None or indices.stop is None or indices.step not in (None, 1):
            raise ValueError(f"{name}.{segment} must be a contiguous slice")
        covered.extend(range(indices.start, indices.stop))
    if covered != list(range(ACTION_DIM)):
        raise ValueError(f"{name} must cover 0..{ACTION_DIM - 1} exactly, got {covered}")


_validate_segments("POLICY_STATE_SEGMENTS", POLICY_STATE_SEGMENTS)
_validate_segments("ENV_ACTION_SEGMENTS", ENV_ACTION_SEGMENTS)
if POLICY_STATE_SEGMENTS == ENV_ACTION_SEGMENTS:
    raise ValueError("policy state and env action layouts must remain distinct")


def segment_ranges(segments: Mapping[str, slice]) -> dict[str, list[int]]:
    """Return JSON-safe half-open ranges for metadata and artifacts."""
    return {name: [part.start, part.stop] for name, part in segments.items()}


def validate_policy_state(state: Any) -> np.ndarray:
    """Validate one compact policy state in policy-state order."""
    array = np.asarray(state, dtype=np.float32)
    if array.shape != (ACTION_DIM,):
        raise ValueError(f"compact policy state must be [{ACTION_DIM}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("compact policy state contains NaN or infinity")
    return array


def extract_policy_state(raw_proprio: Any) -> np.ndarray:
    """Extract the 23D compact policy state from raw R1Pro proprio."""
    raw = np.asarray(raw_proprio, dtype=np.float32)
    if raw.ndim != 1 or raw.shape[0] < RAW_PROPRIO_SEGMENTS["base"].stop:
        raise ValueError(
            "raw R1Pro proprio must be a vector with at least "
            f"{RAW_PROPRIO_SEGMENTS['base'].stop} values, got {raw.shape}"
        )
    compact = np.concatenate(
        [
            raw[RAW_PROPRIO_SEGMENTS["base"]],
            raw[RAW_PROPRIO_SEGMENTS["trunk"]],
            raw[RAW_PROPRIO_SEGMENTS["left_arm"]],
            raw[RAW_PROPRIO_SEGMENTS["right_arm"]],
            np.asarray([raw[RAW_PROPRIO_SEGMENTS["left_gripper"]].sum()]),
            np.asarray([raw[RAW_PROPRIO_SEGMENTS["right_gripper"]].sum()]),
        ]
    )
    return validate_policy_state(compact)


def validate_action_chunk(actions: Any, *, max_horizon: int | None = None) -> np.ndarray:
    """Validate a finite ``[T,23]`` R1Pro env-action chunk."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] < 1:
        raise ValueError(f"BEHAVIOR actions must be [T,{ACTION_DIM}], got {array.shape}")
    if max_horizon is not None and array.shape[0] > int(max_horizon):
        raise ValueError(
            f"BEHAVIOR action horizon {array.shape[0]} exceeds {int(max_horizon)}"
        )
    if not np.isfinite(array).all():
        raise ValueError("BEHAVIOR actions contain NaN or infinity")
    return array


ENV_WIRE_SCHEMA: dict[str, Any] = {
    "name": "behavior_env_rpc",
    "version": 1,
    "observation": {
        "main_images": "uint8[H,W,3]",
        "wrist_images": "uint8[2,H,W,3]",
        "states": "float[raw_proprio_dim]",
        "task_descriptions": "str",
    },
    "action": {
        "shape": f"float[T,{ACTION_DIM}]",
        "segments": segment_ranges(ENV_ACTION_SEGMENTS),
    },
    "official_success_path": ["info", "done", "success"],
}

VLA_WIRE_SCHEMA: dict[str, Any] = {
    "name": "behavior_vla_http",
    "version": 1,
    "request": {
        "instruction": "str",
        "images": dict.fromkeys(CAMERA_KEYS, "png-base64"),
        "state": "float[1,raw_proprio_dim]",
        "compact_state_segments": segment_ranges(POLICY_STATE_SEGMENTS),
        "mode": "eval",
    },
    "response": {
        "actions": f"float[1,T,{ACTION_DIM}]",
        "env_action_segments": segment_ranges(ENV_ACTION_SEGMENTS),
    },
}

RUN_FULL_TASK_SPEC: dict[str, Any] = {
    "name": "run_full_task",
    "description": (
        "Run the current BEHAVIOR task end to end with the configured Pi0.5 "
        "policy. Takes no input. success exactly mirrors official task_success "
        "from the raw environment info.done.success field."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

PI0_PICK_SPEC: dict[str, Any] = {
    "name": "pi0_pick",
    "description": (
        "Run a local Pi0.5/VLA grasp loop from the current BEHAVIOR observation. "
        "Each iteration predicts and executes one validated [T,23] whole-body "
        "action chunk. The PI0 env path monitors actual selected-gripper "
        "proprio at every simulator step. By default a closure candidate is recorded "
        "but does not stop the loop unless a configured local validator accepts the "
        "grasp. An explicit visual-review pause may instead run a bounded number of "
        "post-candidate chunks before returning without claiming success. "
        "Otherwise the loop remains bounded by its local chunk limit, an "
        "official environment stop, the episode horizon, or an error. "
        "Closure alone is not pick success: primitive_success requires a "
        "configured local grasp validator and the result always requires MP4 "
        "visual verification. It never implies official task_success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "hand": {
                "type": "string",
                "enum": ["left", "right"],
                "description": (
                    "Hand whose gripper closure triggers a local grasp candidate."
                ),
            },
            "instruction": {
                "type": "string",
                "minLength": 1,
                "description": "Local VLA grasp instruction for the current observation.",
            },
            "max_chunks": {
                "type": "integer",
                "default": 24,
                "minimum": 1,
            },
            "gripper_closed_threshold": {
                "type": "number",
                "default": 0.045,
                "minimum": 0.0,
                "description": (
                    "Maximum selected-hand compact gripper opening recorded as "
                    "a local grasp candidate. Closure alone neither stops the "
                    "loop nor establishes success."
                ),
            },
            "required_closed_chunks": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Consecutive completed chunks satisfying closure.",
            },
            "stop_on_closure_candidate": {
                "type": "boolean",
                "default": False,
                "description": (
                    "After the first closure candidate, execute the configured "
                    "number of post-candidate chunks and then pause for visual "
                    "review. This never establishes primitive or task success."
                ),
            },
            "post_candidate_chunks": {
                "type": "integer",
                "default": 4,
                "minimum": 0,
                "description": (
                    "Complete action chunks to execute after the first closure "
                    "candidate before pausing. max_chunks remains the total cap."
                ),
            },
        },
        "required": ["hand", "instruction"],
        "additionalProperties": False,
    },
}

PI0_NAVIGATE_TO_SPEC: dict[str, Any] = {
    "name": "pi0_navigate_to",
    "description": (
        "Run one short Pi0.5/VLA visual-navigation segment in the current "
        "BEHAVIOR episode. Each model prediction is truncated to its first "
        "eight actions. The normalized base output is adapted to bounded local "
        "position deltas while the predicted trunk and both arm segments are "
        "executed so whole-body visual/proprio state advances. Both grippers "
        "remain locked to their current latches, so this tool cannot grasp. "
        "The segment pauses after at most four chunks and returns synchronized "
        "head and wrist images for visual review. A normal visual pause is not "
        "primitive or task success. Only task_success mirrors official "
        "info.done.success; object grasping is exclusively pi0_pick."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "The original exact BEHAVIOR task language, passed to the "
                    "VLA unchanged for every short navigation segment."
                ),
            },
            "max_chunks": {
                "type": "integer",
                "default": 4,
                "minimum": 1,
                "maximum": 4,
                "description": (
                    "Model-prediction chunks in this visual segment. Each "
                    "chunk executes no more than its first eight actions."
                ),
            },
        },
        "required": ["instruction"],
        "additionalProperties": False,
    },
}


PI0_NAV_PICK_SPEC: dict[str, Any] = {
    "name": "pi0_nav_pick",
    "description": (
        "Run one continuous Pi0.5/VLA navigation-to-grasp loop in the current "
        "BEHAVIOR episode. Every model prediction and env RPC input is exactly "
        "one complete [32,23] action chunk; this primitive never truncates or "
        "delegates to pi0_navigate_to or pi0_pick. The env-side fail-closed "
        "validator dynamically selects the actually held hand and may stop "
        "inside a chunk only after atomically saving the post-pick checkpoint "
        "and handing off a paused runtime. primitive_success and "
        "local_grasp_success mirror only that local validator. task_success "
        "independently mirrors the official info.done.success bit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Exact VLA task language used unchanged for every complete "
                    "navigation-and-grasp action chunk."
                ),
            },
        },
        "required": ["instruction"],
        "additionalProperties": False,
    },
}


SAVE_ROBOT_STATE_CHECKPOINT_SPEC: dict[str, Any] = {
    "name": "save_robot_state_checkpoint",
    "description": (
        "Save one explicit robot-control checkpoint for a held-object handoff. "
        "The checkpoint records robot/EEF/gripper, held-object, intended press "
        "hand, validation, and visual evidence. It never dumps or serializes "
        "simulator state, scene state, or BEHAVIOR task predicates. Local "
        "primitive success and official task_success remain independent."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "checkpoint_name": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                "default": "state_checkpoint_1",
            },
            "stage": {
                "type": "string",
                "minLength": 1,
                "default": "post_pi0_nav_pick",
            },
            "held_hand": {
                "type": "string",
                "enum": ["left", "right"],
            },
            "press_hand": {
                "type": "string",
                "enum": ["left", "right"],
            },
            "object_name": {
                "type": "string",
                "minLength": 1,
                "default": "radio",
            },
            "require_current_grasp": {
                "type": "boolean",
                "default": True,
            },
            "visual_review": {
                "type": "boolean",
                "default": True,
            },
        },
        "required": ["held_hand", "press_hand"],
        "additionalProperties": False,
    },
}


RESTORE_ROBOT_STATE_CHECKPOINT_SPEC: dict[str, Any] = {
    "name": "restore_robot_state_checkpoint",
    "description": (
        "Plan and execute a cuRobo motion back to an explicit robot-control "
        "checkpoint without reset, scene restore, or simulator-state loading. "
        "The held gripper stays closed throughout execution. Object drift, "
        "drop, held-object mismatch, or unexpected press contact stops the "
        "motion fail-closed. primitive_success never implies official "
        "task_success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "checkpoint_name": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
                "default": "state_checkpoint_1",
            },
            "checkpoint_path": {
                "type": ["string", "null"],
                "default": None,
            },
            "mode": {
                "type": "string",
                "enum": ["plan_and_execute"],
                "default": "plan_and_execute",
            },
            "keep_held_gripper_closed": {
                "type": "boolean",
                "default": True,
            },
            "require_object_still_held": {
                "type": "boolean",
                "default": True,
            },
            "timeout_s": {
                "type": "number",
                "default": 90,
                "exclusiveMinimum": 0,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}


_HAND_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": ["left", "right"],
    "description": "Which R1Pro hand/arm should execute the primitive.",
}

_XYZ_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
    "description": "Target position as [x, y, z] in the selected frame.",
}

_QUAT_SCHEMA: dict[str, Any] = {
    "type": ["array", "null"],
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
    "description": "Quaternion [x, y, z, w].",
}

_VECTOR_SCHEMA: dict[str, Any] = {
    "type": ["array", "null"],
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

_FRAME_SCHEMA: dict[str, Any] = {
    "type": "string",
    "default": "world",
    "description": "Coordinate frame understood by the BEHAVIOR env server.",
}


def _planner_spec(
    name: str,
    description: str,
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
    one_of: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    spec = {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
    }
    if one_of is not None:
        spec["input_schema"]["oneOf"] = one_of
    return spec


INSPECT_POST_PICK_STATE_SPEC: dict[str, Any] = _planner_spec(
    "inspect_post_pick_state",
    (
        "Bind the post-pick phase to a real robot-motion checkpoint and inspect "
        "the current held-object state without advancing physics. The env reads "
        "held_hand and press_hand dynamically from the checkpoint and returns "
        "current held/radio transforms and task radio-face priors. Callers submit "
        "button geometry goals; the runtime alone derives held-EEF candidates."
    ),
    {
        "checkpoint_name": {
            "type": "string",
            "minLength": 1,
            "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
            "default": "state_checkpoint_1",
        },
    },
)

_POST_PICK_CAMERA_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": ["head", "held_wrist", "press_wrist"],
    "description": (
        "Dynamic post-pick camera role. held_wrist and press_wrist are resolved "
        "from state_checkpoint_1; literal left/right is not accepted."
    ),
}

POST_PICK_OBSERVE_SPEC: dict[str, Any] = _planner_spec(
    "observe",
    (
        "Capture one synchronized BEHAVIOR RGB-D frame from head or a dynamic "
        "checkpoint wrist role without advancing physics."
    ),
    {"camera": _POST_PICK_CAMERA_SCHEMA},
    required=["camera"],
)

POST_PICK_PIXEL_TO_WORLD_SPEC: dict[str, Any] = _planner_spec(
    "pixel_to_world",
    (
        "Convert a pixel from a fresh head/held-wrist/press-wrist frame into a "
        "3D point using the env-side depth and calibrated intrinsics."
    ),
    {
        "camera": _POST_PICK_CAMERA_SCHEMA,
        "frame_id": {"type": "string"},
        "u": {"type": "integer", "description": "Pixel column coordinate."},
        "v": {"type": "integer", "description": "Pixel row coordinate."},
        "depth_window_px": {"type": "integer", "default": 7, "minimum": 1},
        "output_frame": {"type": "string", "default": "world"},
    },
    required=["camera", "frame_id", "u", "v"],
)

DECLARE_BUTTON_VISIBILITY_SPEC: dict[str, Any] = _planner_spec(
    "declare_button_visibility",
    (
        "Apply the hard visual gate to one fresh radio frame. A negative "
        "declaration must classify clear_slotted_back_face, side_port, or "
        "ambiguous and omit coordinates. A positive declaration requires the "
        "complete black-disk, white-ring, red-center signature plus bbox and center."
    ),
    {
        "camera": {
            "type": "string",
            "enum": ["head", "held_wrist", "press_wrist"],
        },
        "frame_id": {"type": "string", "minLength": 1},
        "button_visible": {"type": "boolean"},
        "positive_signature": {
            "type": ["object", "null"],
            "default": None,
            "properties": {
                "red_front_face": {"type": "boolean"},
                "black_round_or_oval_disk": {"type": "boolean"},
                "white_outer_ring": {"type": "boolean"},
                "red_center_bump": {"type": "boolean"},
            },
            "required": [
                "red_front_face",
                "black_round_or_oval_disk",
                "white_outer_ring",
                "red_center_bump",
            ],
            "additionalProperties": False,
        },
        "negative_case": {
            "type": ["string", "null"],
            "enum": [
                "clear_slotted_back_face",
                "side_port",
                "ambiguous",
                None,
            ],
            "default": None,
        },
        "bbox_xyxy": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "default": None,
        },
        "center_uv": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
            "default": None,
        },
    },
    required=["camera", "frame_id", "button_visible"],
    one_of=[
        {
            "required": ["negative_case"],
            "properties": {
                "button_visible": {"const": False},
                "positive_signature": {"type": "null"},
                "negative_case": {
                    "type": "string",
                    "enum": [
                        "clear_slotted_back_face",
                        "side_port",
                        "ambiguous",
                    ],
                },
                "bbox_xyxy": {"type": "null"},
                "center_uv": {"type": "null"},
            },
        },
        {
            "required": ["positive_signature", "bbox_xyxy", "center_uv"],
            "properties": {
                "button_visible": {"const": True},
                "positive_signature": {"type": "object"},
                "negative_case": {"type": "null"},
                "bbox_xyxy": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "center_uv": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
        },
    ],
)

PROJECT_BUTTON_SPEC: dict[str, Any] = _planner_spec(
    "project_button",
    (
        "Depth-project the center authorized by a successful button visibility "
        "gate. The env binds gate_id to its exact fresh frame and rejects stale "
        "or NOT_VISIBLE gates."
    ),
    {
        "gate_id": {"type": "string", "minLength": 1},
        "depth_window_px": {"type": "integer", "default": 7, "minimum": 1},
    },
    required=["gate_id"],
)

EVALUATE_PREPRESS_GEOMETRY_SPEC: dict[str, Any] = _planner_spec(
    "evaluate_prepress_geometry",
    (
        "Evaluate the projected button against the dynamically selected press "
        "hand approach line. This computes line distance, outward-normal "
        "opposition angle and axial standoff for the current state; it neither "
        "returns an EEF target nor moves or presses."
    ),
    {
        "projection_id": {"type": "string", "minLength": 1},
        "max_line_distance_m": {
            "type": "number",
            "default": 0.010,
            "exclusiveMinimum": 0.0,
            "maximum": 0.010,
        },
        "max_opposition_angle_deg": {
            "type": "number",
            "default": 15.0,
            "exclusiveMinimum": 0.0,
            "maximum": 15.0,
        },
        "min_axial_standoff_m": {
            "type": "number",
            "default": 0.03,
            "minimum": 0.03,
            "maximum": 0.06,
        },
        "max_axial_standoff_m": {
            "type": "number",
            "default": 0.06,
            "exclusiveMinimum": 0.03,
            "maximum": 0.06,
        },
    },
    required=["projection_id"],
)

PREPRESS_MOVE_TO_SPEC: dict[str, Any] = _planner_spec(
    "move_to",
    (
        "Plan and optionally execute one CuRobo motion selected from runtime-"
        "generated EEF candidates for a button-space goal. "
        "This is the dedicated post-pick move_to, not the generic planner tool: "
        "left/right hand and literal EEF xyz/quaternion arguments are not "
        "accepted. For held_button_alignment, the env turns a desired button "
        "translation/view/face relation into radio poses, derives held EEF "
        "candidates through the live grasp transform, and lets CuRobo select a "
        "reachable collision-free trajectory. For press_staging, the env derives "
        "non-contact press EEF candidates from a fresh press-wrist projection. "
        "The held gripper is forced closed at every waypoint regardless of which "
        "role moves. "
        "Held-object stability and three-view evidence are checked at trajectory end. "
        "Execution requires a matching one-use plan_only certificate for the "
        "exact current gate, projection, role, checkpoint, env step, button goal, "
        "selected candidate, and trajectory."
    ),
    {
        "role": {
            "type": "string",
            "enum": ["held", "press"],
            "default": "held",
            "description": (
                "Dynamic checkpoint role to move. This never accepts a hard-coded "
                "left or right hand."
            ),
        },
        "button_goal": {
            "type": "object",
            "oneOf": [
                {
                    "required": [
                        "kind",
                        "toward_robot_m",
                        "head_view",
                        "face_toward",
                    ],
                    "properties": {
                        "kind": {"const": "held_button_alignment"},
                        "alignment_phase": {
                            "type": "string",
                            "enum": ["joint", "position_first", "normal_refine"],
                            "default": "joint",
                            "description": (
                                "position_first preserves the current radio "
                                "orientation while centering the button; "
                                "normal_refine adjusts the button normal after "
                                "the position move; joint keeps legacy combined "
                                "alignment behavior."
                            ),
                        },
                        "toward_robot_m": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 0.30,
                        },
                        "head_view": {"const": "side"},
                        "face_toward": {"const": "press"},
                        "side_view_tolerance_deg": {
                            "type": "number",
                            "default": 15.0,
                            "exclusiveMinimum": 0.0,
                            "maximum": 30.0,
                        },
                        "face_toward_tolerance_deg": {
                            "type": "number",
                            "default": 30.0,
                            "exclusiveMinimum": 0.0,
                            "maximum": 45.0,
                        },
                        "position_slack_m": {
                            "type": "number",
                            "default": 0.04,
                            "minimum": 0.0,
                            "maximum": 0.10,
                        },
                        "head_target_uv": {
                            "type": ["array", "null"],
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                            "default": None,
                            "description": (
                                "Optional button-center pixel goal in the fresh "
                                "head frame. The runtime preserves the projected "
                                "button depth, constructs a desired button-space "
                                "translation, and only then derives held EEF "
                                "candidates from the live grasp transform."
                            ),
                        },
                        "head_target_radius_px": {
                            "type": "number",
                            "default": 60.0,
                            "minimum": 0.0,
                            "maximum": 160.0,
                            "description": (
                                "Allowed circular image-space neighborhood around "
                                "head_target_uv. The runtime samples button-center "
                                "positions in this region and CuRobo selects a "
                                "reachable collision-free candidate."
                            ),
                        },
                        "minimum_table_clearance_m": {
                            "type": "number",
                            "default": 0.12,
                            "minimum": 0.08,
                            "maximum": 0.25,
                            "description": (
                                "Minimum radio-to-table air gap enforced while "
                                "constructing the button goal. XY may change so "
                                "CuRobo can find a feasible raised trajectory."
                            ),
                        },
                        "candidate_budget": {
                            "type": "integer",
                            "default": 12,
                            "minimum": 1,
                            "maximum": 32,
                        },
                    },
                    "additionalProperties": False,
                },
                {
                    "required": ["kind", "projection_id"],
                    "properties": {
                        "kind": {"const": "press_staging"},
                        "projection_id": {"type": "string", "minLength": 1},
                        "alignment_phase": {
                            "type": "string",
                            "enum": ["final", "observation"],
                            "default": "final",
                            "description": (
                                "final aligns the press EEF at 0.03--0.06 m; "
                                "observation aligns the real wrist-camera "
                                "optical axis at a farther non-contact pose."
                            ),
                        },
                        "standoff_m": {
                            "type": "number",
                            "default": 0.055,
                            "minimum": 0.03,
                            "maximum": 0.25,
                            "description": (
                                "Button-normal standoff. final accepts only "
                                "0.03--0.06 m. After a certified close-pose "
                                "planning failure, observation may use up to "
                                "0.25 m and does not authorize state 2."
                            ),
                        },
                        "candidate_budget": {
                            "type": "integer",
                            "default": 8,
                            "minimum": 1,
                            "maximum": 16,
                        },
                    },
                    "additionalProperties": False,
                },
            ],
        },
        "plan_only": {"type": "boolean", "default": False},
        "timeout_s": {
            "type": "number",
            "default": 90.0,
            "exclusiveMinimum": 0.0,
        },
    },
    required=["role", "button_goal"],
    one_of=[
        {
            "properties": {
                "role": {"const": "held"},
                "button_goal": {
                    "properties": {"kind": {"const": "held_button_alignment"}}
                },
            }
        },
        {
            "properties": {
                "role": {"const": "press"},
                "button_goal": {
                    "properties": {"kind": {"const": "press_staging"}}
                },
            }
        },
    ],
)

PREPRESS_ROTATE_WRIST_SPEC: dict[str, Any] = _planner_spec(
    "rotate_wrist",
    (
        "Plan and optionally execute a wrist-only orientation change for the "
        "selected dynamic role bound by inspect_post_pick_state. No literal "
        "hand argument is accepted. The selected EEF position is retained, the "
        "radio remains attached to the held role for collision checking, and "
        "the held gripper is forced closed throughout every executed trajectory."
    ),
    {
        "role": {
            "type": "string",
            "enum": ["held", "press"],
            "default": "held",
            "description": "Dynamic checkpoint role whose wrist is rotated.",
        },
        "target_quat_xyzw": _QUAT_SCHEMA,
        "relative_axis_angle": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": "[axis_x, axis_y, axis_z, angle_rad].",
        },
        "frame": {
            "type": "string",
            "enum": ["world", "eef"],
            "default": "eef",
        },
        "plan_only": {"type": "boolean", "default": False},
        "timeout_s": {
            "type": "number",
            "default": 90.0,
            "exclusiveMinimum": 0.0,
        },
    },
    one_of=[
        {
            "required": ["target_quat_xyzw"],
            "properties": {
                "target_quat_xyzw": {"type": "array"},
                "relative_axis_angle": {"type": "null"},
            },
        },
        {
            "required": ["relative_axis_angle"],
            "properties": {
                "target_quat_xyzw": {"type": "null"},
                "relative_axis_angle": {"type": "array"},
            },
        },
    ],
)

POST_PICK_SAVE_ROBOT_STATE_CHECKPOINT_SPEC: dict[str, Any] = {
    "name": "save_robot_state_checkpoint",
    "description": (
        "Save only state_checkpoint_1 or state_checkpoint_2 as a robot-motion "
        "checkpoint. An existing state_checkpoint_1 is never overwritten. "
        "state_checkpoint_2 is committed only through the current button, "
        "geometry, motion, and three-view pre-press evidence gate. This never "
        "serializes or restores simulator state."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "checkpoint_name": {
                "type": "string",
                "const": "state_checkpoint_2",
                "default": "state_checkpoint_2",
            },
            "stage": {
                "type": "string",
                "const": "pre_press_alignment",
                "default": "pre_press_alignment",
            },
            "visual_review": {"const": True, "default": True},
        },
        "required": ["checkpoint_name", "stage"],
        "additionalProperties": False,
    },
}

POST_PICK_RESTORE_ROBOT_STATE_CHECKPOINT_SPEC: dict[str, Any] = {
    **RESTORE_ROBOT_STATE_CHECKPOINT_SPEC,
    "description": (
        "Plan and execute a CuRobo robot-motion restore to this run's exact "
        "state_checkpoint_1 or state_checkpoint_2 JSON. This is never a reset, "
        "teleport, simulator snapshot load, or scene restore."
    ),
    "input_schema": {
        **RESTORE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"],
        "properties": {
            **RESTORE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"]["properties"],
            "checkpoint_name": {
                "type": "string",
                "enum": ["state_checkpoint_1", "state_checkpoint_2"],
                "default": "state_checkpoint_1",
            },
        },
    },
}

SAVE_PREPRESS_CHECKPOINT_SPEC: dict[str, Any] = _planner_spec(
    "save_prepress_checkpoint",
    (
        "Save state_checkpoint_2 only after the env has accepted the button hard "
        "gate and pre-press geometry. This is a robot-motion checkpoint, never a "
        "simulator snapshot, and does not press the button."
    ),
    {
        "checkpoint_name": {
            "type": "string",
            "minLength": 1,
            "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
            "default": "state_checkpoint_2",
        },
        "stage": {
            "type": "string",
            "minLength": 1,
            "default": "pre_press_alignment",
        },
        "visual_review": {"type": "boolean", "default": True},
    },
)


PLANNER_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "observe": _planner_spec(
        "observe",
        (
            "Capture one synchronized BEHAVIOR RGB-D frame from the selected "
            "camera. Returns RGB, frame_id, image size, and camera metadata."
        ),
        {
            "camera": {
                "type": "string",
                "enum": ["head", "left_wrist", "right_wrist"],
                "description": "Camera to observe.",
            },
        },
        required=["camera"],
    ),
    "pixel_to_world": _planner_spec(
        "pixel_to_world",
        (
            "Convert a pixel from a previously observed frame into a 3D point "
            "using the env-side depth frame and calibrated camera intrinsics."
        ),
        {
            "camera": {
                "type": "string",
                "enum": ["head", "left_wrist", "right_wrist"],
            },
            "frame_id": {"type": "string"},
            "u": {
                "type": "integer",
                "description": "Pixel column coordinate.",
            },
            "v": {
                "type": "integer",
                "description": "Pixel row coordinate.",
            },
            "depth_window_px": {
                "type": "integer",
                "default": 7,
                "minimum": 1,
            },
            "output_frame": {
                "type": "string",
                "default": "world",
            },
        },
        required=["camera", "frame_id", "u", "v"],
    ),
    "navigate_to": _planner_spec(
        "navigate_to",
        (
            "Move the mobile base to a collision-free stand-off pose near a "
            "target, ensuring the named hand has a feasible collision-free IK "
            "solution after arrival. Requires a fresh observe afterwards."
        ),
        {
            "hand": _HAND_SCHEMA,
            "target_xyz": _XYZ_SCHEMA,
            "frame": _FRAME_SCHEMA,
            "standoff_m": {"type": "number", "default": 0.85, "minimum": 0.0},
            "timeout_s": {"type": "number", "default": 90, "minimum": 0.0},
        },
        required=["hand", "target_xyz"],
    ),
    "move_to": _planner_spec(
        "move_to",
        (
            "Plan and optionally execute a collision-aware cuRobo arm motion "
            "for the selected R1Pro hand. Primitive success never implies "
            "official BEHAVIOR task success."
        ),
        {
            "hand": _HAND_SCHEMA,
            "target_xyz": _XYZ_SCHEMA,
            "frame": _FRAME_SCHEMA,
            "target_quat_xyzw": _QUAT_SCHEMA,
            "plan_only": {"type": "boolean", "default": False},
            "position_tolerance_m": {
                "type": "number",
                "default": 0.02,
                "minimum": 0.0,
            },
            "orientation_tolerance_rad": {
                "type": "number",
                "default": 0.087,
                "minimum": 0.0,
            },
            "timeout_s": {"type": "number", "default": 45, "minimum": 0.0},
        },
        required=["hand", "target_xyz"],
    ),
    "pick": _planner_spec(
        "pick",
        (
            "Execute a guarded grasp with the chosen hand from a depth-derived "
            "target point, using cuRobo to reach pre-grasp and guarded contact "
            "for the final approach."
        ),
        {
            "hand": _HAND_SCHEMA,
            "target_xyz": _XYZ_SCHEMA,
            "approach_vector": _VECTOR_SCHEMA,
            "grasp_quat_xyzw": _QUAT_SCHEMA,
            "pregrasp_offset_m": {
                "type": "number",
                "default": 0.08,
                "minimum": 0.0,
            },
            "lift_m": {"type": "number", "default": 0.08, "minimum": 0.0},
            "timeout_s": {"type": "number", "default": 90, "minimum": 0.0},
        },
        required=["hand", "target_xyz"],
    ),
    "rotate_wrist": _planner_spec(
        "rotate_wrist",
        (
            "Rotate the selected wrist to an absolute quaternion or by a "
            "relative axis-angle command. Provide exactly one rotation form."
        ),
        {
            "hand": _HAND_SCHEMA,
            "target_quat_xyzw": _QUAT_SCHEMA,
            "relative_axis_angle": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "[axis_x, axis_y, axis_z, angle_rad].",
            },
            "frame": {
                "type": "string",
                "enum": ["world", "eef"],
                "default": "world",
            },
            "timeout_s": {"type": "number", "default": 45, "minimum": 0.0},
        },
        required=["hand"],
        one_of=[
            {
                "required": ["target_quat_xyzw"],
                "properties": {
                    "target_quat_xyzw": {"type": "array"},
                    "relative_axis_angle": {"type": "null"},
                },
            },
            {
                "required": ["relative_axis_angle"],
                "properties": {
                    "target_quat_xyzw": {"type": "null"},
                    "relative_axis_angle": {"type": "array"},
                },
            },
        ],
    ),
    "press": _planner_spec(
        "press",
        (
            "Approach and press a target with guarded contact checks. Any "
            "unexpected contact before the target neighborhood aborts."
        ),
        {
            "hand": _HAND_SCHEMA,
            "target_xyz": _XYZ_SCHEMA,
            "press_direction": _VECTOR_SCHEMA,
            "approach_distance_m": {
                "type": "number",
                "default": 0.04,
                "minimum": 0.0,
            },
            "press_depth_m": {
                "type": "number",
                "default": 0.012,
                "minimum": 0.0,
            },
            "timeout_s": {"type": "number", "default": 60, "minimum": 0.0},
        },
        required=["hand", "target_xyz"],
    ),
    "release": _planner_spec(
        "release",
        (
            "Open the selected gripper and optionally retreat along a vector. "
            "Returns primitive status and official task_success separately."
        ),
        {
            "hand": _HAND_SCHEMA,
            "opening": {"type": "number", "default": 1.0, "minimum": 0.0},
            "retreat_vector": _VECTOR_SCHEMA,
            "retreat_m": {"type": "number", "default": 0.03, "minimum": 0.0},
            "timeout_s": {"type": "number", "default": 30, "minimum": 0.0},
        },
        required=["hand"],
    ),
}


if tuple(PLANNER_TOOL_SPECS) != PLANNER_TOOL_NAMES:
    raise ValueError("planner tool schema order must match PLANNER_TOOL_NAMES")

PUBLIC_PRIMITIVE_ENTRYPOINTS: dict[str, str] = {
    "run_full_task": "BehaviorPrimitives.run_full_task",
    **{
        name: f"BehaviorPrimitives.{name}"
        for name in PLANNER_TOOL_NAMES
    },
    "pi0_pick": "BehaviorPrimitives.pi0_pick",
    "pi0_navigate_to": "BehaviorPrimitives.pi0_navigate_to",
    "pi0_nav_pick": "BehaviorPrimitives.pi0_nav_pick",
    **{
        name: f"BehaviorPrimitives.{name}"
        for name in POST_PICK_TOOL_NAMES
        if name not in PLANNER_TOOL_NAMES
        and name not in ROBOT_STATE_CHECKPOINT_TOOL_NAMES
    },
    "save_robot_state_checkpoint": (
        "BehaviorPrimitives.save_robot_state_checkpoint"
    ),
    "restore_robot_state_checkpoint": (
        "BehaviorPrimitives.restore_robot_state_checkpoint"
    ),
}


__all__ = [
    "ACTION_DIM",
    "CAMERA_KEYS",
    "CONTROL_MODES",
    "DEFAULT_ACTION_CHUNK",
    "ENV_ACTION_SEGMENTS",
    "ENV_WIRE_SCHEMA",
    "FULL_TASK_VLA_MODE",
    "HYBRID_TOOL_NAMES",
    "HYBRID_VLM_PI0_MODE",
    "DECLARE_BUTTON_VISIBILITY_SPEC",
    "EVALUATE_PREPRESS_GEOMETRY_SPEC",
    "INSPECT_POST_PICK_STATE_SPEC",
    "POST_PICK_OBSERVE_SPEC",
    "POST_PICK_PIXEL_TO_WORLD_SPEC",
    "POST_PICK_RESTORE_ROBOT_STATE_CHECKPOINT_SPEC",
    "POST_PICK_SAVE_ROBOT_STATE_CHECKPOINT_SPEC",
    "PREPRESS_MOVE_TO_SPEC",
    "PREPRESS_ROTATE_WRIST_SPEC",
    "PI0_PICK_SPEC",
    "PI0_NAV_PICK_SPEC",
    "PI0_NAV_PICK_VLA_MODE",
    "PI0_NAVIGATE_TO_SPEC",
    "PI0_PICK_VLA_MODE",
    "PLANNER_TOOLS_MODE",
    "PLANNER_TOOL_NAMES",
    "PLANNER_TOOL_SPECS",
    "POST_PICK_TOOL_NAMES",
    "POLICY_STATE_SEGMENTS",
    "PROJECT_BUTTON_SPEC",
    "PUBLIC_PRIMITIVE_ENTRYPOINTS",
    "RAW_PROPRIO_SEGMENTS",
    "RESTORE_ROBOT_STATE_CHECKPOINT_SPEC",
    "ROBOT_STATE_CHECKPOINT_TOOL_NAMES",
    "RUN_FULL_TASK_SPEC",
    "SAVE_ROBOT_STATE_CHECKPOINT_SPEC",
    "SAVE_PREPRESS_CHECKPOINT_SPEC",
    "VLA_WIRE_SCHEMA",
    "VLA_CONTROL_MODES",
    "extract_policy_state",
    "segment_ranges",
    "validate_action_chunk",
    "validate_policy_state",
]
