"""Validated BEHAVIOR/R1Pro observation and action contracts."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

ACTION_DIM = 23
DEFAULT_ACTION_CHUNK = 32
CAMERA_KEYS = ("main", "left_wrist", "right_wrist")

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


__all__ = [
    "ACTION_DIM",
    "CAMERA_KEYS",
    "DEFAULT_ACTION_CHUNK",
    "ENV_ACTION_SEGMENTS",
    "ENV_WIRE_SCHEMA",
    "POLICY_STATE_SEGMENTS",
    "RAW_PROPRIO_SEGMENTS",
    "RUN_FULL_TASK_SPEC",
    "VLA_WIRE_SCHEMA",
    "extract_policy_state",
    "segment_ranges",
    "validate_action_chunk",
    "validate_policy_state",
]
