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
CONTROL_MODES = (FULL_TASK_VLA_MODE, PLANNER_TOOLS_MODE, PI0_PICK_VLA_MODE)
VLA_CONTROL_MODES = (FULL_TASK_VLA_MODE, PI0_PICK_VLA_MODE)
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
        "action chunk. The loop stops at the first local gripper-closure "
        "candidate, an "
        "official environment stop, the configured horizon, or an error. "
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
                    "Maximum selected-hand compact gripper opening for a local "
                    "grasp candidate. Closure alone is not success."
                ),
            },
            "required_closed_chunks": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Consecutive completed chunks satisfying closure.",
            },
        },
        "required": ["hand", "instruction"],
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
}


__all__ = [
    "ACTION_DIM",
    "CAMERA_KEYS",
    "CONTROL_MODES",
    "DEFAULT_ACTION_CHUNK",
    "ENV_ACTION_SEGMENTS",
    "ENV_WIRE_SCHEMA",
    "FULL_TASK_VLA_MODE",
    "PI0_PICK_SPEC",
    "PI0_PICK_VLA_MODE",
    "PLANNER_TOOLS_MODE",
    "PLANNER_TOOL_NAMES",
    "PLANNER_TOOL_SPECS",
    "POLICY_STATE_SEGMENTS",
    "PUBLIC_PRIMITIVE_ENTRYPOINTS",
    "RAW_PROPRIO_SEGMENTS",
    "RUN_FULL_TASK_SPEC",
    "VLA_WIRE_SCHEMA",
    "VLA_CONTROL_MODES",
    "extract_policy_state",
    "segment_ranges",
    "validate_action_chunk",
    "validate_policy_state",
]
