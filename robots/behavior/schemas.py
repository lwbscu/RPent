"""Validated BEHAVIOR/R1Pro observation and action contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from robots.behavior.task_specs import (
    BehaviorTaskSpec,
    ReleaseVisualPolicy,
    get_task_spec,
)

ACTION_DIM = 23
DEFAULT_ACTION_CHUNK = 32
# Runtime-owned hard deadline. It is deliberately absent from the public tool
# schema so the planner cannot shorten or extend CuRobo execution.
ROTATE_WRIST_RUNTIME_TIMEOUT_S = 30.0
# Dashboard jog amounts are a server-owned safety contract.  They are
# intentionally not represented in the browser request schema and must not be
# overridden by callers at the HTTP or env-RPC boundaries.
BASE_TRANSLATION_STEP_M = 0.05
BASE_ROTATION_STEP_RAD = math.radians(5.0)
EEF_TRANSLATION_STEP_M = 0.03
TORSO_VERTICAL_STEP_M = 0.03
WRIST_ROTATION_STEP_RAD = math.radians(5.0)

DASHBOARD_CONTROL_TARGETS = ("chassis", "left_arm", "right_arm")
DASHBOARD_CONTROL_ACTIONS = (
    "forward",
    "backward",
    "turn_left",
    "turn_right",
    "up",
    "down",
    "rotate_left",
    "rotate_right",
    "open",
    "close",
    "observe",
)
DASHBOARD_CONTROL_CAMERAS = ("head", "left_wrist", "right_wrist")
CAMERA_KEYS = ("main", "left_wrist", "right_wrist")
_PUBLIC_TOOL_CONTRACT_V1 = (
    "pi0_nav_pick",
    "observe",
    "pixel_to_world",
    "move_to",
    "rotate_wrist",
    "close",
    "open",
    "press",
    "save_robot_state_checkpoint",
)
PUBLIC_TOOL_CONTRACTS: dict[int, tuple[str, ...]] = {
    1: _PUBLIC_TOOL_CONTRACT_V1,
    2: _PUBLIC_TOOL_CONTRACT_V1 + ("navigate_to",),
}
CURRENT_PUBLIC_TOOL_CONTRACT_VERSION = 2
BEHAVIOR_TOOL_NAMES = PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
if tuple(PUBLIC_TOOL_CONTRACTS) != (1, 2):
    raise ValueError("BEHAVIOR public tool contract versions must be contiguous")
if PUBLIC_TOOL_CONTRACTS[2][:-1] != PUBLIC_TOOL_CONTRACTS[1]:
    raise ValueError("BEHAVIOR public tool contract v2 must preserve the v1 prefix")
if len(BEHAVIOR_TOOL_NAMES) != 10 or len(set(BEHAVIOR_TOOL_NAMES)) != 10:
    raise ValueError("the current BEHAVIOR toolkit must expose 10 unique primitives")
FRAME_REVIEW_ASSESSMENTS = (
    "target_bearing_surface_confirmed",
    "opposite_surface_confirmed",
    "side_or_indeterminate",
)


def validate_dashboard_manual_command(
    *,
    target: Any,
    action: Any,
    camera: Any,
) -> dict[str, str]:
    """Validate the semantic-only manual-control RPC contract.

    Fixed motion amounts and gripper openings are deliberately absent.  This
    validator is shared by the in-process primitive, RPC client, and simulator
    facade so no caller can smuggle browser-controlled distances or angles
    through a looser internal layer.
    """

    if not isinstance(target, str) or target not in DASHBOARD_CONTROL_TARGETS:
        raise ValueError("target must be chassis, left_arm, or right_arm")
    if not isinstance(action, str) or action not in DASHBOARD_CONTROL_ACTIONS:
        raise ValueError("unsupported dashboard manual action")
    if not isinstance(camera, str) or camera not in DASHBOARD_CONTROL_CAMERAS:
        raise ValueError("camera must be head, left_wrist, or right_wrist")
    if target == "chassis" and action in {
        "rotate_left",
        "rotate_right",
        "open",
        "close",
    }:
        raise ValueError(f"{action} is available for arm control only")
    return {"target": target, "action": action, "camera": camera}


def _dashboard_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def validate_dashboard_prepare_request(
    *,
    target: Any,
    action: Any,
    predecessor_plan_id: Any = None,
    background: Any = False,
) -> dict[str, Any]:
    """Validate one internal, motion-only Dashboard planning request."""

    command = validate_dashboard_manual_command(
        target=target,
        action=action,
        camera="head",
    )
    if command["action"] == "observe":
        raise ValueError("observe must use dashboard_capture_views")
    if type(background) is not bool:
        raise TypeError("background must be boolean")
    predecessor = (
        None
        if predecessor_plan_id is None
        else _dashboard_identifier(
            predecessor_plan_id,
            name="predecessor_plan_id",
        )
    )
    if background and predecessor is None:
        raise ValueError(
            "background planning requires a predecessor_plan_id"
        )
    return {
        "target": command["target"],
        "action": command["action"],
        "predecessor_plan_id": predecessor,
        "background": background,
    }


def validate_dashboard_plan_id(value: Any) -> str:
    """Validate one opaque internal Dashboard plan identifier."""

    return _dashboard_identifier(value, name="plan_id")


def validate_dashboard_command_id(value: Any) -> str:
    """Validate one opaque command id used for permit and exactly-once replay."""

    return _dashboard_identifier(value, name="command_id")


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
        if (
            indices.start is None
            or indices.stop is None
            or indices.step not in (None, 1)
        ):
            raise ValueError(f"{name}.{segment} must be a contiguous slice")
        covered.extend(range(indices.start, indices.stop))
    if covered != list(range(ACTION_DIM)):
        raise ValueError(
            f"{name} must cover 0..{ACTION_DIM - 1} exactly, got {covered}"
        )


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
        raise ValueError(
            f"compact policy state must be [{ACTION_DIM}], got {array.shape}"
        )
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


def validate_action_chunk(
    actions: Any, *, max_horizon: int | None = None
) -> np.ndarray:
    """Validate a finite ``[T,23]`` R1Pro env-action chunk."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] < 1:
        raise ValueError(
            f"BEHAVIOR actions must be [T,{ACTION_DIM}], got {array.shape}"
        )
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

PI0_NAV_PICK_SPEC: dict[str, Any] = {
    "name": "pi0_nav_pick",
    "description": (
        "Invoke a Pi0.5/VLA skill supporting navigation, grasping, and "
        "pressing using up to the LLM-requested number of [32,23] action chunks. "
        "chunks is a positive requested work bound with no fixed maximum, not a "
        "per-tool quota. Without raw official success or an allowed terminal "
        "exception, an admitted invocation executes exactly the requested number "
        "of complete 32-action chunks. Raw info.done.success stops physical task "
        "execution at the successful environment step, including mid-chunk, seals "
        "an immutable official-success receipt, and permits no later action, "
        "prediction, public-tool call, observation, capability read, or other "
        "task RPC. Only no-action VLA disable/health, environment "
        "freeze/finalize, transport shutdown, and artifact sealing remain "
        "allowlisted. That "
        "receipt-bound partial chunk is a normal successful terminal outcome and "
        "is not counted as a complete chunk. Attachment, held-object, and "
        "multiple-attachment observations alone never shorten the requested work. "
        "Other early returns require a real environment termination/truncation or "
        "a fail-closed runtime safety/infrastructure error. "
        "When one or more objects are already attached, cite exactly one fresh "
        "public observe "
        "frame in current_object_visual_check before this invocation; this replaces "
        "any unconditional held-object rejection. "
        "If the head-camera view does not show either hand clearly, or a hand is "
        "visibly far from the image center, the skill may use pose correction to "
        "re-center the relevant hand before continuing. "
        "Two consecutive complete runtime-accepted regressions—successful "
        "selected-attached-hand "
        "rotate, fresh target-surface review, this skill executing at least one "
        "complete chunk and handing control back, then a distinct fresh "
        "opposite-surface review—disable only pi0_nav_pick for the remainder of "
        "the current attempt; all other public tools remain available. "
        "primitive_success reports only local skill execution; task_success "
        "independently reports the official info.done.success bit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Exact VLA task language used unchanged for every complete "
                    "action chunk, whether for navigation, grasping, or pressing."
                ),
            },
            "chunks": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Positive upper bound on [32,23] Pi0 action chunks selected by "
                    "the LLM. In the absence of official success or an allowed "
                    "terminal exception, exactly this many complete chunks execute."
                ),
            },
            "current_object_visual_check": {
                "type": "object",
                "description": (
                    "Required whenever runtime reports one or more current "
                    "attachments. It binds this invocation to one fresh public "
                    "observe frame that the LLM used to review the current "
                    "task-object configuration."
                ),
                "properties": {
                    "camera": {
                        "type": "string",
                        "enum": ["head", "left_wrist", "right_wrist"],
                    },
                    "frame_id": {"type": "string", "minLength": 1},
                    "assessment": {
                        "type": "string",
                        "const": "current_task_object_configuration_reviewed",
                    },
                },
                "required": ["camera", "frame_id", "assessment"],
                "additionalProperties": False,
            },
        },
        "required": ["instruction", "chunks"],
        "additionalProperties": False,
    },
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


_CAMERA_ROLE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": ["head", "left_wrist", "right_wrist"],
    "description": (
        "Current public physical camera. Wrist names identify the robot's "
        "anatomical left and right hands."
    ),
}

_ANALYTIC_HAND_SCHEMA: dict[str, Any] = {
    "type": "string",
    "enum": ["left", "right"],
    "description": (
        "Robot anatomical hand selected for this analytic primitive. Every hand "
        "selection requires a fresh LLM visual_hand_check."
    ),
}

_VISUAL_HAND_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "LLM visual confirmation required for every analytic primitive hand. "
        "The LLM must inspect the RGB content of a fresh head observe call from "
        "the current episode. left/right mean the robot's anatomical sides, never "
        "the left or right side of the image. selected_hand must equal hand."
    ),
    "properties": {
        "camera": {
            "type": "string",
            "const": "head",
        },
        "frame_id": {
            "type": "string",
            "minLength": 1,
            "description": "Fresh frame_id returned by the cited public observe call.",
        },
        "selected_hand": {
            "type": "string",
            "enum": ["left", "right"],
            "description": "Physical hand selected by the LLM from the cited frame.",
        },
        "assessment": {
            "type": "string",
            "const": "selected_hand_visually_confirmed",
        },
    },
    "required": ["camera", "frame_id", "selected_hand", "assessment"],
    "additionalProperties": False,
}

_RELEASE_VISUAL_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Optional task-specific visual release evidence. Runtime decides when this "
        "evidence is required. It is bound to a fresh public head observation and "
        "must identify the same anatomical hand selected by the open call."
    ),
    "properties": {
        "camera": {
            "type": "string",
            "const": "head",
        },
        "frame_id": {
            "type": "string",
            "minLength": 1,
        },
        "selected_hand": {
            "type": "string",
            "enum": ["left", "right"],
        },
        "assessment": {
            "type": "string",
            "const": "attached_object_fully_inside_receptacle_opening",
        },
    },
    "required": ["camera", "frame_id", "selected_hand", "assessment"],
    "additionalProperties": False,
}


def _bind_visual_checks_to_hand(
    spec: dict[str, Any],
    *,
    include_release_visual_check: bool = False,
) -> None:
    """Require every visual hand declaration to match the requested hand."""

    def branch(hand: str) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "hand": {"const": hand},
            "visual_hand_check": {
                "properties": {
                    "selected_hand": {"const": hand},
                },
            },
        }
        if include_release_visual_check:
            properties["release_visual_check"] = {
                "properties": {
                    "selected_hand": {"const": hand},
                },
            }
        return {"properties": properties}

    spec["input_schema"]["allOf"] = [
        {
            "oneOf": [
                branch("left"),
                branch("right"),
            ]
        }
    ]


OBSERVE_SPEC: dict[str, Any] = _planner_spec(
    "observe",
    (
        "Capture one fresh synchronized public RGB-D observation, or submit an LLM "
        "review or selected-pixel depth probe for a previously returned current "
        "frame, without advancing physics. Omit frame_review and depth_probe to "
        "capture. After inspecting that returned RGB, call observe again with the "
        "same camera and exactly one of frame_review or depth_probe; this read-only "
        "follow-up does not capture or refresh an image. Runtime verifies frame "
        "provenance and freshness, not the semantic truth of an LLM assessment. "
        "A frame review must consume the immediately preceding, same-camera capture "
        "exactly once. "
        "Two consecutive complete runtime-accepted selected-attached-hand "
        "rotate/Pi0/fresh-opposite-surface regression cycles disable only "
        "pi0_nav_pick for the remainder of the current attempt; all other public "
        "tools remain available."
    ),
    {
        "camera": _CAMERA_ROLE_SCHEMA,
        "frame_review": {
            "type": "object",
            "description": (
                "Optional LLM assessment of a fresh frame returned by an earlier "
                "observe call. Include it only after inspecting that RGB; otherwise "
                "omit it to capture a new observation."
            ),
            "properties": {
                "frame_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Fresh frame_id returned by the earlier observe call."
                    ),
                },
                "assessment": {
                    "type": "string",
                    "enum": list(FRAME_REVIEW_ASSESSMENTS),
                    "description": (
                        "Task-surface assessment made by the LLM from the cited "
                        "frame under the selected task-specific visual prior."
                    ),
                },
            },
            "required": ["frame_id", "assessment"],
            "additionalProperties": False,
        },
        "depth_probe": {
            "type": "object",
            "description": (
                "Optional read-only metric-depth probe at one pixel selected by the "
                "LLM after inspecting the immediately preceding fresh RGB frame. "
                "The runtime measures the selected pixel; it does not verify that "
                "the pixel semantically belongs to the claimed target."
            ),
            "properties": {
                "frame_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Fresh frame_id returned by the immediately preceding "
                        "same-camera observe call."
                    ),
                },
                "u": {
                    "type": "integer",
                    "description": "LLM-selected RGB pixel column coordinate.",
                },
                "v": {
                    "type": "integer",
                    "description": "LLM-selected RGB pixel row coordinate.",
                },
                "depth_window_px": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 31,
                    "description": "Local depth sampling window size.",
                },
                "assessment": {
                    "type": "string",
                    "const": "target_point_visually_confirmed",
                    "description": (
                        "LLM confirmation that it selected the intended visible "
                        "target point in the cited RGB frame."
                    ),
                },
            },
            "required": [
                "frame_id",
                "u",
                "v",
                "depth_window_px",
                "assessment",
            ],
            "additionalProperties": False,
        },
    },
    required=["camera"],
)
OBSERVE_SPEC["input_schema"]["allOf"] = [
    {"not": {"required": ["frame_review", "depth_probe"]}}
]

PIXEL_TO_WORLD_SPEC: dict[str, Any] = _planner_spec(
    "pixel_to_world",
    (
        "Back-project one pixel from a fresh public RGB-D frame. Returns a world "
        "point, camera-facing surface normal, confidence, and frame-bound projection_id."
    ),
    {
        "camera": _CAMERA_ROLE_SCHEMA,
        "frame_id": {"type": "string", "minLength": 1},
        "u": {"type": "integer", "description": "Pixel column coordinate."},
        "v": {"type": "integer", "description": "Pixel row coordinate."},
        "depth_window_px": {
            "type": "integer",
            "default": 7,
            "minimum": 1,
            "maximum": 31,
        },
    },
    required=["camera", "frame_id", "u", "v"],
)

_NAVIGATION_VISUAL_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "LLM confirmation that a fresh public head frame depicts the intended "
        "base-navigation target. The frame_id must bind to projection_id at runtime."
    ),
    "properties": {
        "camera": {
            "type": "string",
            "const": "head",
        },
        "frame_id": {
            "type": "string",
            "minLength": 1,
        },
        "assessment": {
            "type": "string",
            "const": "navigation_target_visually_confirmed",
        },
    },
    "required": ["camera", "frame_id", "assessment"],
    "additionalProperties": False,
}

_RELATIVE_NAVIGATION_MOTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "One base-relative motion. Translation follows the robot body's heading "
        "at call start; rotation turns the base and the body together in place."
    ),
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["translation", "rotation"],
        },
        "direction": {
            "type": "string",
            "enum": ["forward", "backward", "left", "right"],
        },
        "distance_m": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.5,
        },
        "angle_deg": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 180.0,
        },
    },
    "oneOf": [
        {
            "properties": {
                "kind": {"const": "translation"},
                "direction": {"enum": ["forward", "backward"]},
            },
            "required": ["kind", "direction", "distance_m"],
            "not": {"required": ["angle_deg"]},
        },
        {
            "properties": {
                "kind": {"const": "rotation"},
                "direction": {"enum": ["left", "right"]},
            },
            "required": ["kind", "direction", "angle_deg"],
            "not": {"required": ["distance_m"]},
        },
    ],
    "additionalProperties": False,
}

NAVIGATE_TO_SPEC: dict[str, Any] = _planner_spec(
    "navigate_to",
    (
        "Move only the robot base while holding trunk, both arms, both grippers, "
        "and both attachment identities fixed relative to the base, so the body "
        "moves and rotates together with it. Use either a fresh head-camera "
        "projection, or one explicit relative translation (forward/backward) or "
        "in-place rotation (left/right). The two modes are mutually exclusive."
    ),
    {
        "projection_id": {
            "type": "string",
            "minLength": 1,
        },
        "navigation_visual_check": _NAVIGATION_VISUAL_CHECK_SCHEMA,
        "relative_motion": _RELATIVE_NAVIGATION_MOTION_SCHEMA,
        "standoff_m": {
            "type": "number",
            "default": 0.85,
            "minimum": 0.45,
            "maximum": 1.50,
        },
        "max_travel_m": {
            "type": "number",
            "default": 1.0,
            "exclusiveMinimum": 0.0,
            "maximum": 1.50,
        },
        "timeout_s": {
            "type": "number",
            "default": 300.0,
            "exclusiveMinimum": 0.0,
        },
    },
    one_of=[
        {
            "required": ["projection_id", "navigation_visual_check"],
            "not": {"required": ["relative_motion"]},
        },
        {
            "required": ["relative_motion"],
            "not": {
                "anyOf": [
                    {"required": ["projection_id"]},
                    {"required": ["navigation_visual_check"]},
                    {"required": ["standoff_m"]},
                    {"required": ["max_travel_m"]},
                ]
            },
        },
    ],
)


def validate_relative_navigation_motion(value: Any) -> dict[str, Any]:
    """Validate and normalize one explicit base-relative navigation motion."""

    if not isinstance(value, Mapping):
        raise ValueError("relative_motion must be an object")
    motion = dict(value)
    kind = motion.get("kind")
    direction = motion.get("direction")
    if kind == "translation":
        expected = {"kind", "direction", "distance_m"}
        allowed_directions = {"forward", "backward"}
        amount_name = "distance_m"
        maximum = 1.5
    elif kind == "rotation":
        expected = {"kind", "direction", "angle_deg"}
        allowed_directions = {"left", "right"}
        amount_name = "angle_deg"
        maximum = 180.0
    else:
        raise ValueError("relative_motion.kind must be translation or rotation")
    if set(motion) != expected:
        raise ValueError(f"relative_motion.{kind} requires exactly {sorted(expected)}")
    if direction not in allowed_directions:
        raise ValueError(f"relative_motion.direction is invalid for {kind}")
    amount = motion[amount_name]
    if isinstance(amount, bool) or not isinstance(
        amount, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"relative_motion.{amount_name} must be a finite number")
    amount = float(amount)
    if not np.isfinite(amount) or amount <= 0.0 or amount > maximum:
        raise ValueError(f"relative_motion.{amount_name} must be within (0,{maximum}]")
    return {
        "kind": str(kind),
        "direction": str(direction),
        amount_name: amount,
    }


_MOVE_TARGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "projection_id": {"type": "string", "minLength": 1},
        "standoff_m": {"type": "number", "minimum": 0.0},
        "delta_xyz": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        "frame": {
            "type": "string",
            "enum": ["world", "eef"],
        },
    },
    "oneOf": [
        {
            "required": ["projection_id"],
            "not": {
                "anyOf": [
                    {"required": ["delta_xyz"]},
                    {"required": ["frame"]},
                ]
            },
        },
        {
            "required": ["delta_xyz", "frame"],
            "not": {
                "anyOf": [
                    {"required": ["projection_id"]},
                    {"required": ["standoff_m"]},
                ]
            },
        },
    ],
    "additionalProperties": False,
}

MOVE_TO_SPEC: dict[str, Any] = _planner_spec(
    "move_to",
    (
        "Execute one R1Pro whole-body 21-DOF CuRobo joint motion using either a "
        "fresh projection or a relative translation. hand selects only the target "
        "EEF; it does not select an isolated arm-only embodiment. The planner may "
        "coordinate the base, trunk, and both arms, and includes objects held by "
        "either hand in collision checking."
    ),
    {
        "hand": _ANALYTIC_HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "target": _MOVE_TARGET_SCHEMA,
        "position_tolerance_m": {
            "type": "number",
            "default": 0.02,
            "exclusiveMinimum": 0.0,
        },
        "max_travel_m": {
            "type": "number",
            "default": 0.25,
            "exclusiveMinimum": 0.0,
        },
        "timeout_s": {
            "type": "number",
            "default": 240.0,
            "exclusiveMinimum": 0.0,
        },
    },
    required=["hand", "visual_hand_check", "target"],
)
_bind_visual_checks_to_hand(MOVE_TO_SPEC)

ROTATE_WRIST_SPEC: dict[str, Any] = _planner_spec(
    "rotate_wrist",
    (
        "Execute one R1Pro whole-body 21-DOF CuRobo joint motion that changes the "
        "target EEF orientation while approximately preserving its position. hand "
        "selects only the target EEF; the planner may coordinate the base, trunk, "
        "and both arms and includes objects held by either hand in collision "
        "checking. Every selected hand must cite one "
        "fresh head observe frame in visual_hand_check. "
        "left/right mean the robot's anatomical sides, not image sides. "
        "Planning and execution use a runtime-owned 30-second hard deadline; "
        "no caller timeout or step budget is accepted."
    ),
    {
        "hand": _ANALYTIC_HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "relative_axis_angle": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": (
                "Relative rotation [axis_x, axis_y, axis_z, angle_rad], using "
                "the right-hand rule. The axis is expressed in frame."
            ),
        },
        "frame": {
            "type": "string",
            "enum": ["world", "eef"],
            "default": "eef",
            "description": (
                "Axis frame for relative_axis_angle only; eef is usually easiest "
                "for an object-relative wrist adjustment."
            ),
        },
    },
    required=["hand", "visual_hand_check", "relative_axis_angle"],
)
_bind_visual_checks_to_hand(ROTATE_WRIST_SPEC)


def _gripper_spec(
    name: str,
    verb: str,
    *,
    release_visual_policy: ReleaseVisualPolicy | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "hand": _ANALYTIC_HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "timeout_s": {
            "type": "number",
            "default": 30.0,
            "exclusiveMinimum": 0.0,
        },
    }
    if release_visual_policy is not None:
        release_visual_check = deepcopy(_RELEASE_VISUAL_CHECK_SCHEMA)
        release_visual_check["properties"]["camera"]["const"] = (
            release_visual_policy.camera
        )
        release_visual_check["properties"]["assessment"]["const"] = (
            release_visual_policy.assessment
        )
        properties["release_visual_check"] = release_visual_check
    spec = _planner_spec(
        name,
        (
            f"{verb} only the gripper on a visually confirmed anatomical hand. "
            "The other gripper, both arms, base, and trunk remain isolated."
        ),
        properties,
        required=["hand", "visual_hand_check"],
    )
    _bind_visual_checks_to_hand(
        spec,
        include_release_visual_check=release_visual_policy is not None,
    )
    return spec


CLOSE_SPEC: dict[str, Any] = _gripper_spec("close", "Close")
OPEN_SPEC: dict[str, Any] = _gripper_spec("open", "Open")

PRESS_SPEC: dict[str, Any] = _planner_spec(
    "press",
    (
        "Execute a press against a fresh projected target using R1Pro whole-body "
        "21-DOF CuRobo joint planning. hand selects only the target EEF; the planner "
        "may coordinate the base, trunk, and both arms and includes objects held by "
        "either hand in collision checking."
    ),
    {
        "hand": _ANALYTIC_HAND_SCHEMA,
        "visual_hand_check": _VISUAL_HAND_CHECK_SCHEMA,
        "projection_id": {"type": "string", "minLength": 1},
        "travel_m": {
            "type": "number",
            "default": 0.03,
            "exclusiveMinimum": 0.0,
        },
        "timeout_s": {
            "type": "number",
            "default": 300.0,
            "exclusiveMinimum": 0.0,
        },
    },
    required=["hand", "visual_hand_check", "projection_id", "travel_m"],
)
_bind_visual_checks_to_hand(PRESS_SPEC)

SAVE_ROBOT_STATE_CHECKPOINT_SPEC: dict[str, Any] = {
    "name": "save_robot_state_checkpoint",
    "description": (
        "Capture synchronized public RGB-D as a read-only visual anchor for LLM "
        "review. It never stores simulator state or authorizes physical action. "
        "When a terminal_failure declaration is bound to a fresh observed frame, "
        "the runtime seals the current attempt as failed and stops further actions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "semantic_label": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
                "description": "Optional human-readable purpose for this visual anchor.",
            },
            "terminal_failure": {
                "type": "object",
                "description": (
                    "Optional condition-5 declaration made only after visually "
                    "verifying the task-relevant radio is lying flat. The cited "
                    "fresh frame is runtime-validated and the attempt ends with "
                    "task_success=false."
                ),
                "properties": {
                    "condition": {
                        "type": "string",
                        "enum": ["radio_tipped_flat"],
                    },
                    "cause": {
                        "type": "string",
                        "enum": [
                            "knocked_over_by_robot_hand",
                            "dropped_out_of_gripper",
                        ],
                    },
                    "camera": _CAMERA_ROLE_SCHEMA,
                    "frame_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Frame ID from the fresh observe result that visually "
                            "established the terminal failure."
                        ),
                    },
                },
                "required": ["condition", "cause", "camera", "frame_id"],
                "additionalProperties": False,
            },
        },
        "required": [],
        "additionalProperties": False,
    },
}

PUBLIC_PRIMITIVE_ENTRYPOINTS: dict[str, str] = {
    "pi0_nav_pick": "BehaviorPrimitives.pi0_nav_pick",
    "observe": "BehaviorPrimitives.observe",
    "pixel_to_world": "BehaviorPrimitives.pixel_to_world",
    "move_to": "BehaviorPrimitives.move_to",
    "rotate_wrist": "BehaviorPrimitives.rotate_wrist",
    "close": "BehaviorPrimitives.close",
    "open": "BehaviorPrimitives.open",
    "press": "BehaviorPrimitives.press",
    "save_robot_state_checkpoint": "BehaviorPrimitives.save_robot_state_checkpoint",
    "navigate_to": "BehaviorPrimitives.navigate_to",
}
if tuple(PUBLIC_PRIMITIVE_ENTRYPOINTS) != BEHAVIOR_TOOL_NAMES:
    raise ValueError("public primitive entrypoints must match BEHAVIOR_TOOL_NAMES")


def behavior_tool_specs_for_task(
    task: str | BehaviorTaskSpec,
) -> dict[str, dict[str, Any]]:
    """Return the fixed tool surface with task-scoped optional contracts."""

    task_spec = get_task_spec(task) if isinstance(task, str) else task
    specs = {
        "pi0_nav_pick": deepcopy(PI0_NAV_PICK_SPEC),
        "observe": deepcopy(OBSERVE_SPEC),
        "pixel_to_world": deepcopy(PIXEL_TO_WORLD_SPEC),
        "move_to": deepcopy(MOVE_TO_SPEC),
        "rotate_wrist": deepcopy(ROTATE_WRIST_SPEC),
        "close": deepcopy(CLOSE_SPEC),
        "open": deepcopy(OPEN_SPEC),
        "press": deepcopy(PRESS_SPEC),
        "save_robot_state_checkpoint": deepcopy(SAVE_ROBOT_STATE_CHECKPOINT_SPEC),
        "navigate_to": deepcopy(NAVIGATE_TO_SPEC),
    }
    if task_spec.release_visual_policy is not None:
        specs["open"] = _gripper_spec(
            "open",
            "Open",
            release_visual_policy=task_spec.release_visual_policy,
        )
    if task_spec.surface_review_policy is None:
        observe = specs["observe"]
        observe["description"] = (
            "Capture one fresh synchronized public RGB-D observation, or submit "
            "one LLM-selected depth_probe for the immediately preceding current "
            "same-camera frame, without advancing physics. Omit depth_probe to "
            "capture a fresh frame."
        )
        observe["input_schema"]["properties"].pop("frame_review", None)
        pi0 = specs["pi0_nav_pick"]
        pi0["description"] = pi0["description"].replace(
            "Two consecutive complete runtime-accepted regressions—successful "
            "selected-attached-hand rotate, fresh target-surface review, this skill "
            "executing "
            "at least one complete chunk and handing control back, then a distinct "
            "fresh opposite-surface review—disable only pi0_nav_pick for the "
            "remainder of the current attempt; all other public tools remain "
            "available. ",
            "",
        )
    if task_spec.terminal_failure_policy is None:
        checkpoint = specs["save_robot_state_checkpoint"]
        checkpoint["description"] = (
            "Capture synchronized public RGB-D as a read-only visual anchor for "
            "LLM review. It never stores simulator state, authorizes physical "
            "action, or declares a task terminal condition."
        )
        checkpoint["input_schema"]["properties"].pop("terminal_failure", None)
    if tuple(specs) != BEHAVIOR_TOOL_NAMES:
        raise RuntimeError("task-scoped public primitive schema order mismatch")
    return specs


__all__ = [
    "ACTION_DIM",
    "BASE_ROTATION_STEP_RAD",
    "BASE_TRANSLATION_STEP_M",
    "BEHAVIOR_TOOL_NAMES",
    "CAMERA_KEYS",
    "CLOSE_SPEC",
    "CURRENT_PUBLIC_TOOL_CONTRACT_VERSION",
    "DEFAULT_ACTION_CHUNK",
    "DASHBOARD_CONTROL_ACTIONS",
    "DASHBOARD_CONTROL_CAMERAS",
    "DASHBOARD_CONTROL_TARGETS",
    "EEF_TRANSLATION_STEP_M",
    "ENV_ACTION_SEGMENTS",
    "ENV_WIRE_SCHEMA",
    "FRAME_REVIEW_ASSESSMENTS",
    "MOVE_TO_SPEC",
    "NAVIGATE_TO_SPEC",
    "OBSERVE_SPEC",
    "OPEN_SPEC",
    "PI0_NAV_PICK_SPEC",
    "PIXEL_TO_WORLD_SPEC",
    "POLICY_STATE_SEGMENTS",
    "PRESS_SPEC",
    "PUBLIC_PRIMITIVE_ENTRYPOINTS",
    "PUBLIC_TOOL_CONTRACTS",
    "RAW_PROPRIO_SEGMENTS",
    "ROTATE_WRIST_SPEC",
    "SAVE_ROBOT_STATE_CHECKPOINT_SPEC",
    "VLA_WIRE_SCHEMA",
    "TORSO_VERTICAL_STEP_M",
    "WRIST_ROTATION_STEP_RAD",
    "behavior_tool_specs_for_task",
    "extract_policy_state",
    "segment_ranges",
    "validate_action_chunk",
    "validate_dashboard_command_id",
    "validate_dashboard_manual_command",
    "validate_dashboard_plan_id",
    "validate_dashboard_prepare_request",
    "validate_policy_state",
    "validate_relative_navigation_motion",
]
