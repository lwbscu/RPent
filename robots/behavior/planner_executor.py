"""Environment-side BEHAVIOR planner primitives backed by RGB-D and cuRobo."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import signal
import threading
import time
import types
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.camera_geometry import (
    CameraGeometryError,
    FrameCache,
    backproject_pixel_to_world,
    canonical_camera,
    hand_geometry_sync_certificate_is_valid,
    validated_rigid_transform,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    BASE_ROTATION_STEP_RAD,
    BASE_TRANSLATION_STEP_M,
    EEF_TRANSLATION_STEP_M,
    ENV_ACTION_SEGMENTS,
    TORSO_VERTICAL_STEP_M,
    WRIST_ROTATION_STEP_RAD,
    validate_action_chunk,
    validate_relative_navigation_motion,
)

LEFT_EEF_LINK = "left_eef_link"
RIGHT_EEF_LINK = "right_eef_link"
EEF_LINK_BY_HAND = {"left": LEFT_EEF_LINK, "right": RIGHT_EEF_LINK}
GRIPPER_COMMAND_BY_OPENING = {"open": 1.0, "closed": -1.0}
LOCAL_GUARDED_IK_SEEDS = 16
ARM_WAYPOINT_TOLERANCE_RAD = 0.02
WHOLE_BODY_BASE_XY_WAYPOINT_TOLERANCE_M = 0.02
WHOLE_BODY_BASE_YAW_WAYPOINT_TOLERANCE_RAD = math.radians(1.0)
WHOLE_BODY_ARTICULATION_WAYPOINT_TOLERANCE_RAD = 0.02
WHOLE_BODY_DENSE_COLLISION_STEP = 0.0075
WHOLE_BODY_EXECUTION_BASE_XY_STEP_M = WHOLE_BODY_DENSE_COLLISION_STEP
WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD = math.radians(1.0)
WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD = 0.030
BASE_EXECUTION_XY_STEP_M = 0.015
BASE_EXECUTION_YAW_STEP_RAD = math.radians(1.5)
DASHBOARD_BASE_EXECUTION_XY_STEP_M = WHOLE_BODY_DENSE_COLLISION_STEP
DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD = math.radians(1.0)
WHOLE_BODY_TOTAL_DEADLINE_S = 240.0
WHOLE_BODY_PLANNING_DEADLINE_S = 60.0
WHOLE_BODY_FAST_TRAJOPT_DEADLINE_S = 12.0
WHOLE_BODY_EXECUTION_DEADLINE_S = 180.0
WHOLE_BODY_SEARCH_PROFILE_DEFAULT = "default"
WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG = "dashboard_jog"
WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S = 12.0
WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S = 12.0
WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S = 4.0
WHOLE_BODY_DASHBOARD_JOG_LOCAL_IK_DEADLINE_S = 0.5
WHOLE_BODY_DASHBOARD_JOG_REPLAN_POSITION_IMPROVEMENT_M = 0.010
DASHBOARD_PREPARED_BASE_PLANNING_PROFILE = "dashboard_base_fast"
DASHBOARD_PREPARED_TORSO_PLANNING_PROFILE = "dashboard_torso_fast"
PREPARED_DASHBOARD_EEF_EXECUTION_POLICY = (
    "prepared_dashboard_eef_first_sample_transient_v1"
)
PREPARED_DASHBOARD_BASE_EXECUTION_POLICY = (
    "prepared_dashboard_base_terminal_tilt_settle_v1"
)
PREPARED_DASHBOARD_TORSO_EXECUTION_POLICY = (
    "prepared_dashboard_torso_certificate_v1"
)
RESET_IDENTITY_WARMUP_PROFILE = "reset_identity_warmup"
RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S = 120.0
WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M = 0.002
WHOLE_BODY_REPLAN_TRACKING_IMPROVEMENT_RATIO = 0.05
WHOLE_BODY_EEF_START_FK_TOLERANCE_M = 0.001
WHOLE_BODY_EEF_TERMINAL_TOLERANCE_M = 0.005
WHOLE_BODY_EEF_SHORT_TARGET_DISTANCE_M = 0.036
WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M = 0.005
WHOLE_BODY_EEF_SHORT_MAX_LATERAL_M = 0.005
WHOLE_BODY_EEF_SHORT_MAX_CARTESIAN_STEP_M = 0.0022
WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M = 0.0029
WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M = 0.006
WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD = math.radians(1.0)
WHOLE_BODY_EEF_SHORT_MAX_REVERSE_STEP_M = 0.001
WHOLE_BODY_EEF_SHORT_MAX_CUMULATIVE_EXCESS_M = 0.010
WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD = math.radians(1.25)
WHOLE_BODY_EEF_WRIST_MAX_REVERSE_PROGRESS_RAD = math.radians(0.25)
WHOLE_BODY_EEF_WRIST_MAX_CUMULATIVE_EXCESS_RAD = math.radians(1.0)
WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M = 0.005
WHOLE_BODY_EEF_LIVE_WAYPOINT_ORIENTATION_TOLERANCE_RAD = math.radians(1.0)
WHOLE_BODY_EEF_WAYPOINT_SETTLE_POSITION_TOLERANCE_M = 0.002
WHOLE_BODY_EEF_SHORT_WAYPOINT_SETTLE_POSITION_TOLERANCE_M = 0.004
WHOLE_BODY_EEF_WAYPOINT_SETTLE_ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M = 0.001
WHOLE_BODY_EEF_NUMERICAL_MARGIN_M = 1e-6
WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M = (
    WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M
    + WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
)
WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M = (
    WHOLE_BODY_EEF_SHORT_MAX_REVERSE_STEP_M
    + WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
)
DYNAMICS_NOMINAL_SAMPLE_RATE_HZ = 60.0
BASE_TRANSLATION_STATIONARY_MAX_ACTUAL_VELOCITY_M_S = 0.001
BASE_YAW_STATIONARY_MAX_ACTUAL_VELOCITY_RAD_S = 0.001
WHOLE_BODY_ARTICULATION_STATIONARY_MAX_ACTUAL_VELOCITY_RAD_S = 0.005
WHOLE_BODY_ARTICULATION_STATIONARY_MAX_STEP_RAD = (
    WHOLE_BODY_ARTICULATION_STATIONARY_MAX_ACTUAL_VELOCITY_RAD_S
    / DYNAMICS_NOMINAL_SAMPLE_RATE_HZ
)
NAVIGATION_TERMINAL_STATIONARY_POLICY = (
    "base_groups_stationary_all_groups_limits_v2"
)
WHOLE_BODY_WAYPOINT_STATIONARY_POLICY = "r1pro_grouped_stationary_60hz_v2"
NAVIGATION_TERMINAL_SETTLE_MAX_HOLDS = 240
NAVIGATION_TERMINAL_SETTLE_DEADLINE_S = 5.0
TRACKING_HARD_BASE_XY_ERROR_M = 0.05
TRACKING_HARD_BASE_YAW_ERROR_RAD = math.radians(5.0)
TRACKING_HARD_ARTICULATION_ERROR_RAD = 0.15
EEF_TERMINAL_POSITION_TOLERANCE_M = 0.015
EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
BASE_TERMINAL_POSITION_TOLERANCE_M = 0.020
BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
TERMINAL_COMMAND_LIMIT = 6
MANUAL_EEF_FALLBACK_OFFSETS_M = (0.0025, -0.0025, 0.005, -0.005)
WRIST_POSITION_DRIFT_LIMIT_M = 0.005
TORSO_LINK_NAME = "torso_link4"
TORSO_ACTIVE_JOINT_NAMES = (
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
)
TORSO_LOCKED_JOINT_NAME = "torso_joint4"
TORSO_LINK_TERMINAL_TOLERANCE_M = 0.010
TORSO_LINK_MAX_XY_DRIFT_M = 0.005
TORSO_LINK_MAX_ORIENTATION_DRIFT_RAD = math.radians(1.0)
TORSO_LINK_MAX_Z_STEP_M = 0.010
TORSO_JOINT_EXECUTION_STEP_RAD = 0.0075
BASE_ACTIVE_JOINT_NAMES = (
    "base_footprint_x_joint",
    "base_footprint_y_joint",
    "base_footprint_rz_joint",
)
PRESS_EEF_TO_CONTACT_OFFSET_M = 0.026
GRIPPER_CLOSE_COARSE_COMMAND_STEP = 0.05
GRIPPER_CLOSE_FINE_COMMAND_STEP = 0.00625
GRIPPER_CONTACT_SETTLE_STEPS = 10
LOCKED_BASE_XY_MAX_DRIFT_M = 0.01
LOCKED_BASE_Z_MAX_DRIFT_M = 0.01
LOCKED_BASE_RPY_MAX_DRIFT_RAD = math.radians(1.0)
DASHBOARD_BASE_TERMINAL_SETTLE_RPY_MAX_RAD = math.radians(1.5)
LOCKED_ARTICULATION_MAX_DRIFT_RAD = 0.01
DASHBOARD_BASE_INTERMEDIATE_TRUNK_MAX_DRIFT_RAD = 0.015
LOCKED_GRIPPER_COMMAND_MAX_DRIFT = 1e-6
TRUNK_ASSIST_MAX_STEP_RAD = 0.01
TRUNK_ASSIST_MAX_TOTAL_RAD = 0.12
_MOTION_SCOPES = frozenset({"arm_only", "arm_with_trunk", "whole_body", "gripper_only"})
WHOLE_BODY_ACTIVE_JOINT_NAMES = (
    "base_footprint_x_joint",
    "base_footprint_y_joint",
    "base_footprint_rz_joint",
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
    *tuple(f"left_arm_joint{i}" for i in range(1, 8)),
    *tuple(f"right_arm_joint{i}" for i in range(1, 8)),
)
DYNAMICS_VELOCITY_GROUP_JOINTS = {
    "base_translation": BASE_ACTIVE_JOINT_NAMES[:2],
    "base_yaw": BASE_ACTIVE_JOINT_NAMES[2:],
    "articulation": WHOLE_BODY_ACTIVE_JOINT_NAMES[3:],
}
DYNAMICS_VELOCITY_GROUP_UNITS = {
    "base_translation": "m/s",
    "base_yaw": "rad/s",
    "articulation": "rad/s",
}
WHOLE_BODY_LOCKED_JOINT_NAMES = (
    "base_footprint_z_joint",
    "base_footprint_rx_joint",
    "base_footprint_ry_joint",
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
)
MAX_BASE_STATION_SHORTLIST = 9
MAX_BASE_PLAN_CANDIDATES = 6
BASE_PLAN_ATTEMPT_TIMEOUT_S = 8.0
_ATTACHMENT_UNSET = object()


def _whole_body_search_profile(profile: str) -> dict[str, Any]:
    """Resolve one internal-only whole-body solver policy."""

    normalized = str(profile)
    if normalized == WHOLE_BODY_SEARCH_PROFILE_DEFAULT:
        return {
            "name": WHOLE_BODY_SEARCH_PROFILE_DEFAULT,
            "planning_deadline_s": WHOLE_BODY_PLANNING_DEADLINE_S,
            "fast_trajopt_deadline_s": WHOLE_BODY_FAST_TRAJOPT_DEADLINE_S,
            "max_attempts": 5,
            "ik_fail_return": 5,
            "finetune_attempts": 2,
            "first_safe_candidate": False,
        }
    if normalized == WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG:
        return {
            "name": WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
            "planning_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S
            ),
            "fast_trajopt_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
            ),
            "local_ik_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_LOCAL_IK_DEADLINE_S
            ),
            "max_attempts": 3,
            "ik_fail_return": 3,
            "finetune_attempts": 1,
            "first_safe_candidate": True,
        }
    if normalized == RESET_IDENTITY_WARMUP_PROFILE:
        return {
            "name": RESET_IDENTITY_WARMUP_PROFILE,
            "planning_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S
            ),
            "fast_trajopt_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
            ),
            "max_attempts": 3,
            "ik_fail_return": 3,
            "finetune_attempts": 1,
            "first_safe_candidate": True,
        }
    raise ValueError(f"unsupported whole-body search profile {profile!r}")


class _WholeBodyCertificationError(RuntimeError):
    """A shared cuRobo certification failure that invalidates the generator."""


@contextmanager
def _wall_clock_deadline(timeout_s: float, operation: str):
    """Interrupt a synchronous planner call at its public wall-clock bound.

    BEHAVIOR's env server dispatches planner tools on its main thread, where a
    POSIX interval timer can bound Python and native-extension calls without a
    worker that could keep mutating CUDA state after the RPC has returned.  A
    non-main-thread call fails closed because Python cannot safely deliver the
    timer there.
    """

    seconds = float(timeout_s)
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise TimeoutError(f"{operation} deadline is already exhausted")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            f"{operation} requires main-thread dispatch for bounded timeout"
        )
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_remaining, previous_interval = signal.getitimer(signal.ITIMER_REAL)
    effective_seconds = (
        min(seconds, previous_remaining) if previous_remaining > 0.0 else seconds
    )
    started = time.monotonic()

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(
            f"{operation} exceeded wall-clock deadline {effective_seconds:.3f}s"
        )

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, effective_seconds)
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_remaining > 0.0:
            remaining = previous_remaining - elapsed
            if remaining <= 0.0:
                if callable(previous_handler):
                    previous_handler(signal.SIGALRM, None)
                else:
                    raise TimeoutError("outer planner deadline exceeded")
            else:
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_interval)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return repr(value)


def _artifact_jsonable(value: Any) -> Any:
    """Serialize diagnostics without copying image or other binary payloads."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary_omitted": True, "size_bytes": len(value)}
    if isinstance(value, dict):
        return {
            str(key): _artifact_jsonable(item)
            for key, item in value.items()
            if not str(key).startswith("_image")
            and not str(key).startswith("_depth_image")
        }
    if isinstance(value, (list, tuple)):
        return [_artifact_jsonable(item) for item in value]
    return _jsonable(value)


def _guarded_waypoint_distances(total_distance_m: float) -> list[float]:
    """Return an approach split into at most two-millimetre Cartesian segments."""

    total = max(0.0, float(total_distance_m))
    steps = max(1, int(math.ceil(total / 0.002)))
    return [total * index / steps for index in range(1, steps + 1)]


def _terminally_smoothed_joint_trajectory(
    trajectory: Any,
    *,
    max_collinear_tail_segments: int = 12,
    zero_velocity_steps: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ease a guarded path to zero velocity without changing its geometry.

    The dense guarded path normally ends with many collinear interpolation
    samples.  Replace only that final straight suffix with a triangular
    (linearly decreasing) command-spacing profile, then append exact endpoint
    repeats.  Every replacement sample lies on the original final joint-space
    line segment.
    """

    q = np.asarray(_jsonable(trajectory), dtype=np.float64)
    original_waypoints = int(len(q)) if q.ndim >= 1 else 0
    if q.ndim != 2 or q.shape[0] < 1 or not np.isfinite(q).all():
        raise ValueError(f"joint trajectory must be finite [T,D], got {q.shape}")
    tail_limit = max(1, int(max_collinear_tail_segments))
    zero_steps = max(1, int(zero_velocity_steps))
    if len(q) == 1:
        result = np.repeat(q, zero_steps + 1, axis=0)
        return result.astype(np.float32), {
            "method": "collinear_terminal_triangular_ease_out",
            "path_geometry": "original_joint_polyline",
            "original_waypoints": 1,
            "smoothed_waypoints": int(len(result)),
            "collinear_tail_segments": 0,
            "terminal_zero_delta_steps": zero_steps,
            "terminal_max_command_acceleration_proxy_rad_s2": 0.0,
        }

    deltas = np.diff(q, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    nonzero = np.flatnonzero(delta_norms > 1e-10)
    if not nonzero.size:
        result = np.vstack([q[:1], np.repeat(q[-1:], zero_steps, axis=0)])
        return result.astype(np.float32), {
            "method": "collinear_terminal_triangular_ease_out",
            "path_geometry": "original_joint_polyline",
            "original_waypoints": int(len(q)),
            "smoothed_waypoints": int(len(result)),
            "collinear_tail_segments": 0,
            "terminal_zero_delta_steps": zero_steps,
            "terminal_max_command_acceleration_proxy_rad_s2": 0.0,
        }

    final_segment = int(nonzero[-1])
    # Ignore any pre-existing endpoint repeats; fresh repeats are added below.
    q = q[: final_segment + 2]
    deltas = np.diff(q, axis=0)
    reference = deltas[-1] / max(float(np.linalg.norm(deltas[-1])), 1e-12)
    tail_start = final_segment
    while tail_start > 0 and final_segment - tail_start + 1 < tail_limit:
        candidate = deltas[tail_start - 1]
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm <= 1e-10:
            break
        candidate_direction = candidate / candidate_norm
        if float(np.dot(candidate_direction, reference)) < 1.0 - 1e-5:
            break
        tail_start -= 1

    tail_segments = final_segment - tail_start + 1
    tail_vector = q[-1] - q[tail_start]
    # With M=2L+1, the first triangular increment is close to the incoming
    # nominal increment while every subsequent increment decreases linearly.
    # A minimum of 12 samples also leaves a useful terminal ramp for a short
    # final IK edge.
    eased_steps = max(12, 2 * tail_segments + 1)
    weights = np.arange(eased_steps, 0, -1, dtype=np.float64)
    fractions = np.cumsum(weights) / float(np.sum(weights))
    eased = q[tail_start] + fractions[:, None] * tail_vector
    eased[-1] = q[-1]
    result = np.vstack(
        [q[: tail_start + 1], eased, np.repeat(q[-1:], zero_steps, axis=0)]
    )
    result_deltas = np.diff(result, axis=0)
    # This is a conservative one-sample command proxy, not a substitute for
    # the runtime robot-velocity feedback and its unchanged 15 rad/s^2 limit.
    acceleration_proxy = (
        float(np.max(np.abs(np.diff(result_deltas, axis=0)))) * 60.0 * 60.0
        if len(result_deltas) >= 2
        else 0.0
    )
    return result.astype(np.float32), {
        "method": "collinear_terminal_triangular_ease_out",
        "path_geometry": "original_joint_polyline",
        "original_waypoints": original_waypoints,
        "smoothed_waypoints": int(len(result)),
        "collinear_tail_segments": int(tail_segments),
        "terminal_ease_out_steps": int(eased_steps),
        "terminal_zero_delta_steps": zero_steps,
        "terminal_max_command_acceleration_proxy_rad_s2": acceleration_proxy,
    }


def _attachment_identity_status(
    attached_obj: Any,
    expected_attachment: Any,
    *,
    hand: str,
) -> tuple[bool, dict[str, Any]]:
    """Compare an assisted-grasp collision body by stable root identity."""

    eef_link = EEF_LINK_BY_HAND[_normalize_hand(hand)]
    actual_root = attached_obj.get(eef_link) if isinstance(attached_obj, dict) else None
    expected_root = (
        expected_attachment.get(eef_link)
        if isinstance(expected_attachment, dict)
        else None
    )

    def root_path(root: Any) -> str | None:
        value = str(getattr(root, "prim_path", "")).rstrip("/")
        return value or None

    actual_path = root_path(actual_root)
    expected_path = root_path(expected_root)
    matches = bool(
        actual_root is not None
        and expected_root is not None
        and (
            actual_root is expected_root
            or (
                actual_path is not None
                and expected_path is not None
                and actual_path == expected_path
            )
        )
    )
    return matches, {
        "expected_available": expected_root is not None,
        "actual_available": actual_root is not None,
        "identity_kind": (
            "prim_path"
            if expected_path is not None
            else "exact_python_object_reference"
        ),
        "matches": matches,
    }


def _attachment_state_status(
    actual: Any,
    expected: Any,
    *,
    hand: str,
) -> tuple[bool, dict[str, Any]]:
    """Compare an attachment without exposing its private scene identity."""

    if actual is None or expected is None:
        matches = actual is None and expected is None
        return matches, {
            "expected_attached": expected is not None,
            "actual_attached": actual is not None,
            "matches": matches,
        }
    matches, _identity = _attachment_identity_status(
        actual,
        expected,
        hand=hand,
    )
    return matches, {
        "expected_attached": True,
        "actual_attached": True,
        "matches": matches,
    }


def _apply_fixed_trajectory_hold_segments(
    action: Any,
    hold_reference: Any,
    *,
    hand: str | None,
    motion_scope: str = "arm_only",
) -> np.ndarray:
    """Freeze non-active 23D controller segments to one trajectory-start hold."""

    out = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
    hold = np.asarray(hold_reference, dtype=np.float32).reshape(ACTION_DIM)
    if motion_scope not in _MOTION_SCOPES:
        raise ValueError(f"unsupported analytic motion scope {motion_scope!r}")
    if motion_scope == "whole_body":
        locked_segments = ("left_gripper", "right_gripper")
    elif hand is None:
        locked_segments = (
            "trunk",
            "left_arm",
            "right_arm",
            "left_gripper",
            "right_gripper",
        )
    else:
        active = _normalize_hand(hand)
        inactive = "right" if active == "left" else "left"
        locked_segments = (
            ("base", f"{inactive}_arm", "left_gripper", "right_gripper")
            if motion_scope == "arm_with_trunk"
            else (
                "base",
                "trunk",
                f"{inactive}_arm",
                "left_gripper",
                "right_gripper",
            )
        )
    for segment_name in locked_segments:
        segment = ENV_ACTION_SEGMENTS[segment_name]
        out[segment] = hold[segment]
    if hand is not None and motion_scope == "arm_with_trunk":
        trunk = ENV_ACTION_SEGMENTS["trunk"]
        out[trunk.stop - 1] = hold[trunk.stop - 1]
    return out


def _apply_single_arm_isolation_mask(
    action: Any,
    hold_reference: Any,
    *,
    hand: str,
    gripper_only: bool,
    motion_scope: str = "arm_only",
) -> np.ndarray:
    """Keep every controller outside the selected analytic scope at its hold."""

    if motion_scope not in _MOTION_SCOPES:
        raise ValueError(f"unsupported analytic motion scope {motion_scope!r}")
    if motion_scope == "whole_body":
        raise ValueError("whole-body motion must not use a single-arm isolation mask")
    expected_scope = "gripper_only" if bool(gripper_only) else motion_scope
    if bool(gripper_only) != (expected_scope == "gripper_only"):
        raise ValueError("gripper_only and motion_scope disagree")
    selected = _normalize_hand(hand)
    inactive = "right" if selected == "left" else "left"
    out = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
    hold = np.asarray(hold_reference, dtype=np.float32).reshape(ACTION_DIM)
    locked_segments = (
        (
            "base",
            "trunk",
            "left_arm",
            "right_arm",
            f"{inactive}_gripper",
        )
        if bool(gripper_only)
        else (
            ("base", f"{inactive}_arm", "left_gripper", "right_gripper")
            if motion_scope == "arm_with_trunk"
            else (
                "base",
                "trunk",
                f"{inactive}_arm",
                "left_gripper",
                "right_gripper",
            )
        )
    )
    for segment_name in locked_segments:
        segment = ENV_ACTION_SEGMENTS[segment_name]
        out[segment] = hold[segment]
    if motion_scope == "arm_with_trunk":
        trunk = ENV_ACTION_SEGMENTS["trunk"]
        out[trunk.stop - 1] = hold[trunk.stop - 1]
    return out


def official_task_success(info: Any) -> bool:
    """Read only BEHAVIOR's official success bit."""
    if not isinstance(info, dict):
        return False
    done = info.get("done")
    if not isinstance(done, dict):
        return False
    value = done.get("success", False)
    return isinstance(value, (bool, np.bool_)) and bool(value)


_TERMINAL_STEP_STOP_REASONS = frozenset(
    {
        "official_task_success",
        "environment_terminated",
        "environment_truncated",
    }
)


def _terminal_step_outcome(
    receipt: dict[str, bool | int],
) -> tuple[bool, str] | None:
    """Return the highest-priority terminal outcome from one executed env step."""

    if receipt.get("raw_success") is True:
        return True, "official_task_success"
    if receipt.get("terminated") is True:
        return False, "environment_terminated"
    if receipt.get("truncated") is True:
        return False, "environment_truncated"
    return None


def primitive_result(
    *,
    primitive_success: bool,
    task_success: bool,
    stop_reason: str,
    recoverable: bool,
    suggested_next_tool: str | None = None,
    metrics: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared planner result envelope."""
    metric_values = _jsonable(metrics or {})
    diagnostic_values = _jsonable(diagnostics or {})
    joint_margin = metric_values.get("joint_margin")
    return {
        "primitive_success": bool(primitive_success),
        "task_success": bool(task_success),
        "_finish": bool(task_success),
        "official_success_source": 'info["done"]["success"]',
        "stop_reason": str(stop_reason),
        "recoverable": bool(recoverable),
        "position_error_m": metric_values.get("final_position_error_m"),
        "orientation_error_rad": metric_values.get("final_orientation_error_rad"),
        "joint_margin": joint_margin,
        "elapsed_s": metric_values.get("elapsed_s"),
        "trace": diagnostic_values.get("trace", []),
        "trace_artifact": diagnostic_values.get("trace_artifact"),
        "metrics": metric_values,
        "diagnostics": diagnostic_values,
    }


def _strip_public_flow_advice(value: Any) -> Any:
    """Remove planner-prescribed ordering and phase structure from public output."""

    if isinstance(value, dict):
        return {
            key: _strip_public_flow_advice(item)
            for key, item in value.items()
            if key != "suggested_next_tool" and "stage" not in key.lower()
        }
    if isinstance(value, list):
        return [_strip_public_flow_advice(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_public_flow_advice(item) for item in value)
    return value


def _planner_tool(name: str):
    """Normalize every public planner call and persist its complete envelope."""

    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            started = time.monotonic()
            try:
                bound = inspect.signature(fn).bind_partial(self, *args, **kwargs)
                timeout_s = bound.arguments.get("timeout_s")
                if timeout_s is None:
                    result = fn(self, *args, **kwargs)
                else:
                    public_deadline_s = float(timeout_s)
                    if name == "move_to":
                        public_deadline_s = min(
                            public_deadline_s,
                            WHOLE_BODY_TOTAL_DEADLINE_S,
                        )
                    with _wall_clock_deadline(
                        public_deadline_s,
                        f"planner tool {name}",
                    ):
                        result = fn(self, *args, **kwargs)
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"planner tool {name} returned {type(result)!r}, expected dict"
                    )
            except Exception as exc:
                result = self._exception_result(
                    exc,
                    suggested_next_tool=None,
                )
            metrics = result.setdefault("metrics", {})
            if isinstance(metrics, dict):
                metrics.setdefault("elapsed_s", round(time.monotonic() - started, 3))
                result["elapsed_s"] = metrics["elapsed_s"]
            result = _strip_public_flow_advice(result)
            result["tool_artifact"] = self._persist_tool_artifact(
                tool=name,
                args=args,
                kwargs=kwargs,
                result=result,
            )
            return result

        return wrapped

    return decorate


def _normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if value not in EEF_LINK_BY_HAND:
        raise ValueError("hand must be 'left' or 'right'")
    return value


def _as_xyz(value: Any, *, name: str = "target_xyz") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _quat_xyzw(value: Any | None) -> np.ndarray | None:
    if value is None:
        return None
    quat = np.asarray(value, dtype=np.float64).reshape(-1)
    if quat.shape != (4,):
        raise ValueError("quaternion must contain exactly 4 xyzw values")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8 or not math.isfinite(norm):
        raise ValueError("quaternion has zero or invalid norm")
    return quat / norm


def _axis_angle_to_quat_xyzw(axis_angle: Any) -> np.ndarray:
    vec = np.asarray(axis_angle, dtype=np.float64).reshape(-1)
    if vec.shape != (4,):
        raise ValueError(
            "relative_axis_angle must contain [axis_x, axis_y, axis_z, angle_rad]"
        )
    axis = vec[:3]
    angle = float(vec[3])
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        if abs(angle) <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        raise ValueError("relative_axis_angle axis cannot be zero for nonzero angle")
    axis = axis / norm
    if abs(angle) <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return np.concatenate([axis * math.sin(angle * 0.5), [math.cos(angle * 0.5)]])


def _quat_multiply_xyzw(a: Any, b: Any) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    out = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )
    return out / max(float(np.linalg.norm(out)), 1e-12)


def _quat_rotate_vector_xyzw(quaternion: Any, vector: Any) -> np.ndarray:
    quat = _quat_xyzw(quaternion)
    assert quat is not None
    x, y, z, w = quat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ np.asarray(vector, dtype=np.float64).reshape(3)


def _quat_angle_error_rad(a: Any | None, b: Any | None) -> float | None:
    if a is None or b is None:
        return None
    qa = _quat_xyzw(a)
    qb = _quat_xyzw(b)
    assert qa is not None and qb is not None
    dot = abs(float(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return 2.0 * math.acos(dot)


def _eef_pose_path_admission_report(
    positions: Any,
    quaternions_xyzw: Any,
    *,
    call_start_xyz: Any,
    target_xyz: Any,
    target_quat_xyzw: Any | None,
    short_cartesian_step_limit_m: float = (
        WHOLE_BODY_EEF_SHORT_MAX_CARTESIAN_STEP_M
    ),
) -> dict[str, Any]:
    """Recompute the short-jog Cartesian/orientation contract from EEF poses."""

    start = _as_xyz(call_start_xyz, name="call_start_xyz")
    target = _as_xyz(target_xyz)
    target_quat = _quat_xyzw(target_quat_xyzw)
    points = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions_xyzw, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 1
        or quaternions.shape != (len(points), 4)
        or not np.isfinite(points).all()
        or not np.isfinite(quaternions).all()
    ):
        raise ValueError("EEF path poses must be finite [T,3] and [T,4] arrays")
    quat_norms = np.linalg.norm(quaternions, axis=1)
    if np.any(quat_norms <= 1e-9):
        raise ValueError("EEF path contains an invalid quaternion")
    normalized_quats = quaternions / quat_norms[:, None]

    start_fk_error = float(np.linalg.norm(points[0] - start))
    terminal_error = float(np.linalg.norm(points[-1] - target))
    target_errors = np.linalg.norm(points - target, axis=1)
    relative = points - points[0]
    start_excursions = np.linalg.norm(relative, axis=1)
    cartesian_steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_cartesian_path = float(np.sum(cartesian_steps))
    direct = target - points[0]
    direct_distance = float(np.linalg.norm(direct))
    if direct_distance > 1e-9:
        axis = direct / direct_distance
        along = relative @ axis
        lateral = np.linalg.norm(relative - along[:, None] * axis, axis=1)
    else:
        along = np.zeros((len(points),), dtype=np.float64)
        lateral = start_excursions
    along_steps = np.diff(along)
    max_excursion = float(np.max(start_excursions, initial=0.0))
    max_lateral = float(np.max(lateral, initial=0.0))
    min_along = float(np.min(along, initial=0.0))
    max_along = float(np.max(along, initial=0.0))
    max_reverse_step = float(np.max(-along_steps, initial=0.0))
    max_cartesian_step = float(np.max(cartesian_steps, initial=0.0))

    max_orientation_error = 0.0
    start_orientation_error = 0.0
    terminal_orientation_error = 0.0
    max_orientation_step = 0.0
    max_orientation_reverse_progress = 0.0
    cumulative_orientation_path = 0.0
    intentional_orientation_change = False
    if target_quat is not None:
        dots = np.clip(
            np.abs(normalized_quats @ target_quat),
            0.0,
            1.0,
        )
        orientation_errors = 2.0 * np.arccos(dots)
        max_orientation_error = float(np.max(orientation_errors, initial=0.0))
        start_orientation_error = float(orientation_errors[0])
        terminal_orientation_error = float(orientation_errors[-1])
        intentional_orientation_change = (
            start_orientation_error
            > WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9
        )
        if len(normalized_quats) > 1:
            adjacent_dots = np.clip(
                np.abs(
                    np.sum(
                        normalized_quats[:-1] * normalized_quats[1:],
                        axis=1,
                    )
                ),
                0.0,
                1.0,
            )
            orientation_steps = 2.0 * np.arccos(adjacent_dots)
            cumulative_orientation_path = float(np.sum(orientation_steps))
            max_orientation_step = float(
                np.max(orientation_steps, initial=0.0)
            )
            max_orientation_reverse_progress = float(
                np.max(np.diff(orientation_errors), initial=0.0)
            )

    short_target = (
        direct_distance <= WHOLE_BODY_EEF_SHORT_TARGET_DISTANCE_M + 1e-9
    )
    short_step_limit = float(short_cartesian_step_limit_m)
    if not math.isfinite(short_step_limit) or short_step_limit <= 0.0:
        raise ValueError("short EEF Cartesian step limit must be positive")
    all_target_step_limit = (
        max(WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M, short_step_limit)
        if short_target
        else WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M
    )
    max_excursion_limit = (
        direct_distance + WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M
    )
    checks = {
        "start_fk_matches_call_start": (
            start_fk_error <= WHOLE_BODY_EEF_START_FK_TOLERANCE_M + 1e-9
        ),
        "terminal_reaches_target": (
            terminal_error <= WHOLE_BODY_EEF_TERMINAL_TOLERANCE_M + 1e-9
        ),
        "all_target_cartesian_step": (
            max_cartesian_step <= all_target_step_limit + 1e-9
        ),
        "short_target_start_excursion": (
            not short_target or max_excursion <= max_excursion_limit + 1e-9
        ),
        "short_target_segment_lateral": (
            not short_target
            or max_lateral <= WHOLE_BODY_EEF_SHORT_MAX_LATERAL_M + 1e-9
        ),
        "short_target_no_reverse_or_overshoot": (
            not short_target
            or (
                min_along >= -WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M - 1e-9
                and max_along
                <= direct_distance + WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M + 1e-9
            )
        ),
        "short_target_cartesian_step": (
            not short_target
            or max_cartesian_step
            <= short_step_limit + 1e-9
        ),
        "short_target_axis_monotonic": (
            not short_target
            or direct_distance <= 1e-9
            or max_reverse_step
            <= WHOLE_BODY_EEF_SHORT_MAX_REVERSE_STEP_M + 1e-9
        ),
        "short_target_cumulative_path": (
            not short_target
            or cumulative_cartesian_path
            <= direct_distance
            + WHOLE_BODY_EEF_SHORT_MAX_CUMULATIVE_EXCESS_M
            + 1e-9
        ),
        "short_target_orientation_preserved": (
            not short_target
            or target_quat is None
            or intentional_orientation_change
            or max_orientation_error
            <= WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9
        ),
        "short_target_rotation_terminal": (
            not short_target
            or target_quat is None
            or not intentional_orientation_change
            or terminal_orientation_error
            <= WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9
        ),
        "short_target_rotation_step": (
            not short_target
            or target_quat is None
            or not intentional_orientation_change
            or max_orientation_step
            <= WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD + 1e-9
        ),
        "short_target_rotation_monotonic": (
            not short_target
            or target_quat is None
            or not intentional_orientation_change
            or max_orientation_reverse_progress
            <= WHOLE_BODY_EEF_WRIST_MAX_REVERSE_PROGRESS_RAD + 1e-9
        ),
        "short_target_rotation_cumulative_path": (
            not short_target
            or target_quat is None
            or not intentional_orientation_change
            or cumulative_orientation_path
            <= start_orientation_error
            + WHOLE_BODY_EEF_WRIST_MAX_CUMULATIVE_EXCESS_RAD
            + 1e-9
        ),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "waypoint_count": int(len(points)),
        "start_fk_error_m": start_fk_error,
        "terminal_error_m": terminal_error,
        "call_start_to_target_distance_m": direct_distance,
        "short_target": short_target,
        "max_start_excursion_m": max_excursion,
        "max_start_excursion_limit_m": max_excursion_limit,
        "max_segment_lateral_m": max_lateral,
        "min_segment_along_m": min_along,
        "max_segment_along_m": max_along,
        "max_cartesian_step_m": max_cartesian_step,
        "all_target_cartesian_step_limit_m": all_target_step_limit,
        "short_target_cartesian_step_limit_m": short_step_limit,
        "max_reverse_step_m": max_reverse_step,
        "cumulative_cartesian_path_m": cumulative_cartesian_path,
        "max_orientation_error_rad": max_orientation_error,
        "intentional_orientation_change": intentional_orientation_change,
        "start_orientation_error_rad": start_orientation_error,
        "terminal_orientation_error_rad": terminal_orientation_error,
        "max_orientation_step_rad": max_orientation_step,
        "max_orientation_reverse_progress_rad": (
            max_orientation_reverse_progress
        ),
        "cumulative_orientation_path_rad": cumulative_orientation_path,
        "nominal_max_target_error_m": float(
            np.max(target_errors, initial=0.0)
        ),
    }


def _segment_array(value: Any) -> np.ndarray:
    array = np.asarray(_jsonable(value), dtype=np.int64).reshape(-1)
    return array


class RealCuroboBackend:
    """Lazy OmniGibson/cuRobo adapter used only inside the real env process."""

    def __init__(
        self, env_facade: Any, *, output_dir: str | Path | None = None
    ) -> None:
        self.env_facade = env_facade
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("/tmp") / "rpent_behavior_planner" / str(os.getpid())
        )
        self._robot: Any | None = None
        self._torch: Any | None = None
        self._curobo_cls: Any | None = None
        self._embodiment_cls: Any | None = None
        self._generators: dict[str, Any] = {}
        self._invalid_generators: set[str] = set()
        self._config_paths: dict[str, Path] = {}
        self._base_workspace_limit_m: float | None = None
        self._last_base_candidate_summary: dict[str, Any] = {}
        self._attached_objects_by_hand: dict[str, Any] = {}
        self._last_actual_velocity: np.ndarray | None = None
        self._last_actual_velocity_step: int | None = None
        # Phase one has been the stable branch for the R1Pro contact corridor
        # in the real challenge scene.  Subsequent calls still rotate through
        # deterministic phases, so a failed branch remains bounded and
        # independently retryable.
        self._guarded_seed_counter = 1

    def on_runtime_state_changed(self) -> None:
        """Invalidate controller-dependent feedback after a q-state change.

        A controller reload does not mutate scene mesh geometry.  Generator
        obstacle baselines therefore remain valid and every subsequent plan
        still refreshes all robot-relative poses before collision checking.
        Keeping the baseline also preserves reset-time identity warmup for the
        first Dashboard jog instead of forcing another multi-second full world
        rebuild.  Actual generator quarantine/rebuild and scene reset create a
        fresh wrapper, while topology/storage/scale mismatches independently
        fail back to a cleared full rebuild.
        """

        self._attached_objects_by_hand.clear()
        self._last_actual_velocity = None
        self._last_actual_velocity_step = None
        self._base_workspace_limit_m = None
        self._guarded_seed_counter = 1

    @staticmethod
    def _generator_key(*, kind: str, hand: str = "left") -> str:
        hand_kinds = {"arm", "attached_arm", "arm_with_trunk", "whole_body"}
        return f"{kind}:{_normalize_hand(hand) if kind in hand_kinds else 'left'}"

    @staticmethod
    def _mesh_topology_component(
        value: Any,
        *,
        full_digest: bool,
        exact_values: bool = False,
    ) -> tuple[Any, ...]:
        """Fingerprint mesh storage, optionally including exact values.

        OmniGibson's rigid ``GeomPrim.points`` and ``faces`` are
        ``functools.cached_property`` tensors.  Their object identity plus the
        Torch mutation version is therefore an exact, constant-time topology
        revision for the lifetime of that wrapper: replacement changes the
        identity and in-place edits increment ``_version``.  Non-Torch storage
        has no equivalent mutation counter, so it continues to use an exact
        content digest.

        ``full_digest`` validates the storage contents separately, but must
        not change the shape of the topology key.  Otherwise the first full
        snapshot can never compare equal to the following pose-only snapshot.
        """

        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.is_tensor(value):
            component: tuple[Any, ...] = (
                "torch",
                str(value.dtype),
                str(value.device),
                tuple(int(size) for size in value.shape),
                tuple(int(stride) for stride in value.stride()),
                int(getattr(value, "_version")),
            )
            if not exact_values:
                return (*component, id(value))
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
            if array.dtype == object:
                raise RuntimeError("collision mesh topology is not numeric")
            component = (
                "numpy",
                str(array.dtype),
                tuple(int(size) for size in array.shape),
                tuple(int(stride) for stride in array.strides),
            )
        contiguous = np.ascontiguousarray(array)
        return (
            *component,
            hashlib.sha256(contiguous.tobytes()).hexdigest(),
        )

    @staticmethod
    def _mesh_content_digest(value: Any) -> str:
        """Return an exact geometry digest for full-refresh validation only."""

        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.dtype == object:
            raise RuntimeError("collision mesh geometry is not numeric")
        contiguous = np.ascontiguousarray(array)
        return hashlib.sha256(contiguous.tobytes()).hexdigest()

    @staticmethod
    def _physics_tensor_array(value: Any) -> np.ndarray:
        """Copy one Physics Tensor result to a numeric host array."""

        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            to_numpy = getattr(value, "numpy", None)
            array = to_numpy() if callable(to_numpy) else value
        result = np.asarray(array)
        if result.dtype == object:
            raise RuntimeError("Physics Tensor transforms are not numeric")
        return result

    @classmethod
    def _collision_link_pose_batch(
        cls,
        *,
        physics_sim_view: Any,
        requested_prim_paths: tuple[str, ...],
        pose_batch_cache: Mapping[str, Any] | None,
        create: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Read exact dynamic-link poses through one validated Physics Tensor view."""

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Physics Tensor collision pose reads require main-thread dispatch"
            )
        import omnigibson as og

        if bool(getattr(og.sim, "currently_stepping", False)):
            raise RuntimeError(
                "Physics Tensor collision pose reads are unavailable during a physics step"
            )
        if physics_sim_view is None:
            raise RuntimeError("Physics Tensor simulation view is unavailable")
        paths = tuple(str(path) for path in requested_prim_paths)
        if not paths or paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise RuntimeError(
                "Physics Tensor collision link paths must be non-empty, sorted, and unique"
            )

        if pose_batch_cache is None:
            if not create:
                raise RuntimeError("Physics Tensor collision pose cache is unavailable")
            create_view = getattr(physics_sim_view, "create_rigid_body_view", None)
            if not callable(create_view):
                raise RuntimeError(
                    "Physics Tensor rigid-body view API is unavailable"
                )
            view = create_view(list(paths))
            cached_mapping = None
        else:
            if pose_batch_cache.get("physics_sim_view") is not physics_sim_view:
                raise RuntimeError("Physics Tensor simulation view identity changed")
            if tuple(pose_batch_cache.get("prim_paths", ())) != paths:
                raise RuntimeError("Physics Tensor collision link path registry changed")
            view = pose_batch_cache.get("view")
            cached_mapping = pose_batch_cache.get("index_by_prim_path")
            if not isinstance(cached_mapping, Mapping):
                raise RuntimeError(
                    "Physics Tensor rigid-body view row mapping is unavailable"
                )

        check = getattr(view, "check", None)
        if not callable(check) or not bool(check()):
            raise RuntimeError("Physics Tensor rigid-body view is invalid")
        try:
            count = int(view.count)
            actual_paths = tuple(str(path) for path in view.prim_paths)
        except Exception as exc:
            raise RuntimeError(
                "Physics Tensor rigid-body view registry is unavailable"
            ) from exc
        if (
            count != len(paths)
            or len(actual_paths) != count
            or len(actual_paths) != len(set(actual_paths))
            or set(actual_paths) != set(paths)
        ):
            raise RuntimeError(
                "Physics Tensor rigid-body view did not bind the exact requested paths"
            )
        index_by_prim_path = {
            prim_path: index for index, prim_path in enumerate(actual_paths)
        }
        if cached_mapping is not None and dict(cached_mapping) != index_by_prim_path:
            raise RuntimeError("Physics Tensor rigid-body view row mapping changed")

        get_transforms = getattr(view, "get_transforms", None)
        if not callable(get_transforms):
            raise RuntimeError("Physics Tensor transform read API is unavailable")
        transforms = cls._physics_tensor_array(get_transforms())
        if transforms.shape != (count, 7):
            raise RuntimeError(
                "Physics Tensor transforms must have exact shape [N,7]"
            )
        transforms = np.asarray(transforms, dtype=np.float64)
        if not np.isfinite(transforms).all():
            raise RuntimeError("Physics Tensor transforms are not finite")
        quaternion_norms = np.linalg.norm(transforms[:, 3:], axis=1)
        if not np.allclose(
            quaternion_norms,
            np.ones(count, dtype=np.float64),
            rtol=0.0,
            atol=1e-3,
        ):
            raise RuntimeError(
                "Physics Tensor transforms contain a non-unit quaternion"
            )

        poses_by_path = {
            path: transforms[index_by_prim_path[path]].copy() for path in paths
        }
        cache = {
            # Strong references bind the view to the exact simulator instance
            # and prevent identity reuse while the fast-refresh state is live.
            "physics_sim_view": physics_sim_view,
            "prim_paths": paths,
            "view": view,
            "index_by_prim_path": index_by_prim_path,
        }
        return poses_by_path, cache

    @classmethod
    def _current_collision_mesh_snapshot(
        cls,
        generator: Any,
        ignore_objects: Any = None,
        *,
        full_digest: bool = False,
        kinematic_cache: Mapping[str, Any] | None = None,
        pose_batch_cache: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read every live collision mesh and its fresh robot-relative pose."""

        snapshot_started_ns = time.monotonic_ns()
        import omnigibson as og
        import omnigibson.lazy as lazy
        import omnigibson.utils.transform_utils as transform_utils

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "collision mesh snapshot requires main-thread dispatch"
            )
        if bool(getattr(og.sim, "currently_stepping", False)):
            raise RuntimeError(
                "collision mesh snapshot is unavailable during a physics step"
            )
        pose_elapsed_ns = 0
        ignored = set(ignore_objects or ())
        robot = generator.robot
        robot_pose_started_ns = time.monotonic_ns()
        try:
            robot_transform = transform_utils.pose_inv(
                transform_utils.pose2mat(
                    robot.root_link.get_position_orientation()
                )
            )
        finally:
            pose_elapsed_ns += time.monotonic_ns() - robot_pose_started_ns
        meshes: list[dict[str, Any]] = []
        storage_refs: list[tuple[str, Any, Any]] = []
        next_kinematic_cache: dict[str, dict[str, Any]] = {}
        link_pose_cache: dict[int, Any] = {}
        scalar_link_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        batch_link_poses: dict[int, np.ndarray] = {}

        content_digests: list[str] = []
        topology_elapsed_ns = 0

        link_entries: list[tuple[Any, Any, Mapping[str, Any]]] = []
        dynamic_links_by_path: dict[str, Any] = {}
        batch_setup_error: str | None = None

        scene = getattr(robot, "scene", None)
        objects = getattr(scene, "objects", None)
        if objects is None:
            raise RuntimeError("robot scene collision objects are unavailable")
        for obj in objects:
            if obj is robot or obj in ignored or bool(getattr(obj, "visual_only", False)):
                continue
            links = getattr(obj, "links", None)
            if not isinstance(links, Mapping):
                raise RuntimeError("collision object links are unavailable")
            for link in links.values():
                collision_meshes = getattr(link, "collision_meshes", None)
                if not isinstance(collision_meshes, Mapping):
                    raise RuntimeError("collision link mesh registry is unavailable")
                link_entries.append((obj, link, collision_meshes))
                if not collision_meshes:
                    continue
                if getattr(obj, "kinematic_only", None) is not False:
                    continue
                link_path = str(getattr(link, "prim_path", ""))
                if not link_path:
                    batch_setup_error = (
                        "dynamic collision link prim path is unavailable"
                    )
                    continue
                existing = dynamic_links_by_path.get(link_path)
                if existing is not None and existing is not link:
                    batch_setup_error = (
                        "dynamic collision link prim paths are not unique"
                    )
                    continue
                dynamic_links_by_path[link_path] = link

        dynamic_paths = tuple(sorted(dynamic_links_by_path))
        next_pose_batch_cache: dict[str, Any] | None = None
        batch_fallback_reason = batch_setup_error
        batch_read_count = 0
        if not full_digest and dynamic_paths and batch_fallback_reason is None:
            if pose_batch_cache is None:
                batch_fallback_reason = (
                    "Physics Tensor collision pose cache is unavailable"
                )
            else:
                batch_pose_started_ns = time.monotonic_ns()
                try:
                    poses_by_path, next_pose_batch_cache = (
                        cls._collision_link_pose_batch(
                            physics_sim_view=getattr(
                                og.sim,
                                "physics_sim_view",
                                None,
                            ),
                            requested_prim_paths=dynamic_paths,
                            pose_batch_cache=pose_batch_cache,
                            create=False,
                        )
                    )
                    batch_link_poses = {
                        id(dynamic_links_by_path[path]): pose
                        for path, pose in poses_by_path.items()
                    }
                    batch_read_count = len(poses_by_path)
                except Exception as exc:
                    batch_fallback_reason = f"{type(exc).__name__}: {exc}"
                    next_pose_batch_cache = None
                    batch_link_poses.clear()
                finally:
                    pose_elapsed_ns += (
                        time.monotonic_ns() - batch_pose_started_ns
                    )

        def pose_to_transform(pose: np.ndarray) -> Any:
            position: Any = pose[:3]
            orientation: Any = pose[3:]
            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None and torch.is_tensor(robot_transform):
                position = torch.as_tensor(
                    position,
                    dtype=robot_transform.dtype,
                    device=robot_transform.device,
                )
                orientation = torch.as_tensor(
                    orientation,
                    dtype=robot_transform.dtype,
                    device=robot_transform.device,
                )
            return transform_utils.pose2mat((position, orientation))

        def link_world_transform(link: Any) -> Any:
            nonlocal pose_elapsed_ns
            key = id(link)
            transform = link_pose_cache.get(key)
            if transform is not None:
                return transform
            pose_started_ns = time.monotonic_ns()
            batch_pose = batch_link_poses.get(key)
            if batch_pose is not None:
                transform = pose_to_transform(batch_pose)
            else:
                scalar_pose = link.get_position_orientation()
                transform = transform_utils.pose2mat(scalar_pose)
                scalar_link_poses[key] = (
                    np.asarray(_jsonable(scalar_pose[0]), dtype=np.float64),
                    np.asarray(_jsonable(scalar_pose[1]), dtype=np.float64),
                )
            pose_elapsed_ns += time.monotonic_ns() - pose_started_ns
            link_pose_cache[key] = transform
            return transform

        def append_mesh(mesh: Any, *, static_floor: bool = False) -> None:
            nonlocal topology_elapsed_ns
            topology_started_ns = time.monotonic_ns()
            name = str(getattr(mesh, "name", ""))
            pose = np.asarray(_jsonable(getattr(mesh, "pose", None)), dtype=np.float64)
            if (
                not name
                or pose.shape != (7,)
                or not np.isfinite(pose).all()
                or float(np.linalg.norm(pose[3:])) <= 1e-9
            ):
                raise RuntimeError("collision mesh name or pose is unavailable")
            if static_floor:
                topology = (
                    name,
                    "static_floor",
                    cls._mesh_topology_component(
                        mesh.scale,
                        full_digest=True,
                        exact_values=True,
                    ),
                    cls._mesh_content_digest(mesh.vertices),
                    cls._mesh_content_digest(mesh.faces),
                )
                if full_digest:
                    content_digests.append(
                        hashlib.sha256(
                            repr(
                                (
                                    cls._mesh_content_digest(mesh.vertices),
                                    cls._mesh_content_digest(mesh.faces),
                                )
                            ).encode("utf-8")
                        ).hexdigest()
                    )
            else:
                # Rigid OmniGibson mesh vertices/faces are cached-property
                # tensors.  Identity + Torch _version detects replacement and
                # in-place edits without copying all scene geometry back to
                # CPU on every 5 cm / 5 degree Dashboard jog.  The first full
                # refresh still records an exact content digest, while
                # non-Torch providers remain exact-content keyed.
                vertices = mesh.vertices
                faces = mesh.faces
                topology = (
                    name,
                    cls._mesh_topology_component(
                        vertices,
                        full_digest=True,
                        exact_values=False,
                    ),
                    cls._mesh_topology_component(
                        faces,
                        full_digest=True,
                        exact_values=False,
                    ),
                    cls._mesh_topology_component(
                        mesh.scale,
                        full_digest=True,
                        exact_values=True,
                    ),
                )
                if full_digest:
                    content_digests.append(
                        hashlib.sha256(
                            repr(
                                (
                                    cls._mesh_content_digest(vertices),
                                    cls._mesh_content_digest(faces),
                                )
                            ).encode("utf-8")
                        ).hexdigest()
                    )
                storage_refs.append((name, vertices, faces))
            meshes.append(
                {
                    "name": name,
                    "pose": pose.tolist(),
                    "topology": topology,
                }
            )
            topology_elapsed_ns += time.monotonic_ns() - topology_started_ns

        floor_plane = og.sim.floor_plane
        if floor_plane is not None:
            prim = getattr(floor_plane, "prim", None)
            children = list(prim.GetChildren()) if prim is not None else []
            mesh_reader = getattr(
                lazy.curobo.util.usd_helper,
                "get_mesh_attrs",
                None,
            )
            xform_cache = getattr(
                getattr(generator, "usd_help", None),
                "_xform_cache",
                None,
            )
            if len(children) != 1 or not callable(mesh_reader) or xform_cache is None:
                raise RuntimeError("unsupported floor plane collision mesh API")
            floor_prim = children[0]
            # The floor is only one mesh, so verify its exact immutable
            # geometry even on a pose-only refresh.  This closes the otherwise
            # silent case where a same-name floor changes vertices in place.
            floor_mesh = mesh_reader(
                floor_prim,
                cache=xform_cache,
                transform=robot_transform.numpy(),
            )
            append_mesh(floor_mesh, static_floor=True)

        for _obj, link, collision_meshes in link_entries:
            for collision_mesh in collision_meshes.values():
                if str(getattr(collision_mesh, "geom_type", "")) != "Mesh":
                    raise RuntimeError(
                        "unsupported non-mesh collision geometry "
                        f"{getattr(collision_mesh, 'prim_path', None)!r}"
                    )
                name = str(collision_mesh.prim_path)
                cached = (
                    kinematic_cache.get(name)
                    if isinstance(kinematic_cache, Mapping)
                    else None
                )
                link_pose = link_world_transform(link)
                pose_started_ns = time.monotonic_ns()
                if full_digest or not isinstance(cached, Mapping):
                    obj_pose = transform_utils.pose2mat(
                        collision_mesh.get_position_orientation()
                    )
                    mesh_in_link = transform_utils.pose_inv(link_pose) @ obj_pose
                else:
                    if (
                        cached.get("link") is not link
                        or cached.get("mesh") is not collision_mesh
                    ):
                        raise RuntimeError(
                            "collision mesh kinematic registry changed"
                        )
                    mesh_in_link = cached.get("mesh_in_link")
                    if mesh_in_link is None:
                        raise RuntimeError(
                            "collision mesh local transform is unavailable"
                        )
                    obj_pose = link_pose @ mesh_in_link
                pose = robot_transform @ obj_pose
                position, orientation_xyzw = transform_utils.mat2pose(pose)
                orientation_wxyz = orientation_xyzw[[3, 0, 1, 2]]
                pose_elapsed_ns += time.monotonic_ns() - pose_started_ns
                next_kinematic_cache[name] = {
                    "link": link,
                    "mesh": collision_mesh,
                    "mesh_in_link": mesh_in_link,
                }
                append_mesh(
                    types.SimpleNamespace(
                        name=name,
                        pose=np.concatenate(
                            [
                                np.asarray(_jsonable(position), dtype=np.float64),
                                np.asarray(
                                    _jsonable(orientation_wxyz),
                                    dtype=np.float64,
                                ),
                            ]
                        ),
                        vertices=collision_mesh.points,
                        faces=collision_mesh.faces,
                        scale=collision_mesh.get_world_scale(),
                    )
                )

        rebuild_pose_batch_cache = (
            dynamic_paths
            and batch_setup_error is None
            and (full_digest or pose_batch_cache is None)
        )
        if rebuild_pose_batch_cache:
            batch_pose_started_ns = time.monotonic_ns()
            try:
                poses_by_path, candidate_cache = cls._collision_link_pose_batch(
                    physics_sim_view=getattr(
                        og.sim,
                        "physics_sim_view",
                        None,
                    ),
                    requested_prim_paths=dynamic_paths,
                    pose_batch_cache=None,
                    create=True,
                )
                for path, batch_pose in poses_by_path.items():
                    link = dynamic_links_by_path[path]
                    scalar_pose = scalar_link_poses.get(id(link))
                    if scalar_pose is None:
                        raise RuntimeError(
                            "authoritative scalar dynamic-link pose is unavailable"
                        )
                    scalar_position, scalar_quaternion = scalar_pose
                    batch_position = batch_pose[:3]
                    batch_quaternion = batch_pose[3:]
                    normalized_scalar = scalar_quaternion / np.linalg.norm(
                        scalar_quaternion
                    )
                    normalized_batch = batch_quaternion / np.linalg.norm(
                        batch_quaternion
                    )
                    if not np.allclose(
                        scalar_position,
                        batch_position,
                        rtol=1e-5,
                        atol=1e-4,
                    ) or not math.isclose(
                        abs(float(np.dot(normalized_scalar, normalized_batch))),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-4,
                    ):
                        raise RuntimeError(
                            f"Physics Tensor shadow pose parity failed for {path!r}"
                        )
                next_pose_batch_cache = candidate_cache
                batch_read_count = len(poses_by_path)
                batch_fallback_reason = None
            except Exception as exc:
                batch_fallback_reason = f"{type(exc).__name__}: {exc}"
                next_pose_batch_cache = None
            finally:
                pose_elapsed_ns += time.monotonic_ns() - batch_pose_started_ns

        meshes.sort(key=lambda item: item["name"])
        names = [item["name"] for item in meshes]
        if len(names) != len(set(names)):
            raise RuntimeError("collision mesh names are not unique")
        if not dynamic_paths:
            pose_read_mode = "scalar_only"
        elif rebuild_pose_batch_cache and batch_fallback_reason is None:
            pose_read_mode = (
                "scalar_authoritative_shadow_batch"
                if full_digest
                else "scalar_authoritative_shadow_batch_rebuild"
            )
        elif full_digest:
            pose_read_mode = "scalar_authoritative_fallback"
        elif batch_fallback_reason is None:
            pose_read_mode = (
                "batch_dynamic_scalar_residual"
                if scalar_link_poses
                else "batch_dynamic"
            )
        else:
            pose_read_mode = "scalar_fallback"
        pose_read_metrics = {
            "mode": pose_read_mode,
            "count": int(batch_read_count),
            "fallback": batch_fallback_reason is not None,
            "fallback_reason": batch_fallback_reason,
            "dynamic_link_count": len(dynamic_paths),
            "scalar_link_count": len(scalar_link_poses),
        }
        return {
            "topology": tuple(item["topology"] for item in meshes),
            "poses": tuple((item["name"], item["pose"]) for item in meshes),
            "mesh_count": len(meshes),
            # Strong references prevent Python object-id ABA while the fast
            # topology token is live.  They are intentionally private to the
            # in-process refresh wrapper and never enter public artifacts.
            "storage_refs": tuple(storage_refs),
            "kinematic_cache": next_kinematic_cache,
            "pose_batch_cache": next_pose_batch_cache,
            "pose_read": pose_read_metrics,
            "snapshot_timings": {
                "pose_collection_s": round(pose_elapsed_ns / 1_000_000_000, 6),
                "topology_collection_s": round(
                    topology_elapsed_ns / 1_000_000_000,
                    6,
                ),
                "total_s": round(
                    (time.monotonic_ns() - snapshot_started_ns) / 1_000_000_000,
                    6,
                ),
                "unique_link_count": len(link_pose_cache),
            },
            "content_sha256": (
                hashlib.sha256(
                    repr(tuple(content_digests)).encode("utf-8")
                ).hexdigest()
                if full_digest
                else None
            ),
        }

    @staticmethod
    def _obstacle_refresh_metrics(generator: Any) -> dict[str, Any] | None:
        metrics = getattr(generator, "_rpent_obstacle_refresh_metrics", None)
        return deepcopy(metrics) if isinstance(metrics, dict) else None

    def _install_fast_obstacle_refresh(
        self,
        generator: Any,
        *,
        snapshot_provider: Any = None,
        world_collision_type: Any = None,
        pose_factory: Any = None,
        pose_batch_factory: Any = None,
    ) -> None:
        """Wrap OG obstacle rebuilds with an exact CuRobo mesh-pose fast path."""

        if getattr(generator, "_rpent_fast_obstacle_refresh_installed", False):
            return
        original_update = generator.update_obstacles
        provider = snapshot_provider or self._current_collision_mesh_snapshot
        if world_collision_type is None:
            from curobo.geom.sdf.world_mesh import WorldMeshCollision

            world_collision_type = WorldMeshCollision
        if pose_factory is None:
            from curobo.types.math import Pose

            def pose_factory(pose: Any, tensor_args: Any) -> Any:
                return Pose.from_list(
                    pose,
                    tensor_args=tensor_args,
                )

            def pose_batch_factory(poses: Any, tensor_args: Any) -> Any:
                return Pose.from_batch_list(
                    poses,
                    tensor_args=tensor_args,
                )

        state: dict[str, Any] = {
            "topology": None,
            "storage_refs": None,
            "kinematic_cache": None,
            "pose_batch_cache": None,
            "count": 0,
            "initialized": False,
        }
        refresh_lock = threading.RLock()

        try:
            provider_parameters = inspect.signature(provider).parameters
        except (TypeError, ValueError):
            provider_parameters = {}
        provider_accepts_full_digest = (
            "full_digest" in provider_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in provider_parameters.values()
            )
        )
        provider_accepts_kinematic_cache = (
            "kinematic_cache" in provider_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in provider_parameters.values()
            )
        )
        provider_accepts_pose_batch_cache = (
            "pose_batch_cache" in provider_parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in provider_parameters.values()
            )
        )

        def capture_snapshot(
            ignore_objects: Any,
            *,
            full_digest: bool,
        ) -> dict[str, Any]:
            kwargs: dict[str, Any] = {}
            if provider_accepts_full_digest:
                kwargs["full_digest"] = full_digest
            if provider_accepts_kinematic_cache:
                kwargs["kinematic_cache"] = state.get("kinematic_cache")
            if provider_accepts_pose_batch_cache:
                kwargs["pose_batch_cache"] = state.get("pose_batch_cache")
            return provider(generator, ignore_objects, **kwargs)

        def motion_generators() -> list[Any]:
            by_id = {id(value): value for value in generator.mg.values()}
            return list(by_id.values())

        def reset_all_graph_buffers() -> int:
            resets = [
                getattr(
                    getattr(value, "graph_planner", None),
                    "reset_buffer",
                    None,
                )
                for value in motion_generators()
            ]
            if not all(callable(reset) for reset in resets):
                raise RuntimeError(
                    "CuRobo graph planner reset API is unavailable"
                )
            for reset in resets:
                reset()
            return len(resets)

        def clear_collision_world() -> None:
            motion_gen_values = motion_generators()
            checkers = {
                id(getattr(value, "world_coll_checker", None)): getattr(
                    value,
                    "world_coll_checker",
                    None,
                )
                for value in motion_gen_values
            }
            if len(checkers) != 1:
                raise RuntimeError("generator does not share one collision world")
            clear_world = getattr(motion_gen_values[0], "clear_world_cache", None)
            if not callable(clear_world):
                raise RuntimeError("CuRobo collision cache clear API is unavailable")
            clear_world()

        def call_full(ignore_objects: Any) -> None:
            if ignore_objects is None:
                original_update()
            else:
                original_update(ignore_objects=ignore_objects)

        def record(
            *,
            started_ns: int,
            mode: str,
            reason: str,
            mesh_count: int | None,
            fallback: bool,
            fallback_reason: str | None,
            graph_reset_count: int,
            pose_update_mode: str | None = None,
            pose_read: Mapping[str, Any] | None = None,
            snapshot_timings: Mapping[str, Any] | None = None,
            error: str | None = None,
        ) -> None:
            payload = {
                "clock": "time.monotonic",
                "completed_monotonic_ns": time.monotonic_ns(),
                "count": int(state["count"]),
                "mode": str(mode),
                "reason": str(reason),
                "mesh_count": mesh_count,
                "elapsed_s": round(
                    max(0, time.monotonic_ns() - started_ns) / 1_000_000_000,
                    6,
                ),
                "fallback": bool(fallback),
                "fallback_reason": fallback_reason,
                "fresh_robot_relative_poses": True,
                "stale_pose_ttl_s": None,
                "collision_checks_skipped": False,
                "topology_verified": mode == "pose_only",
                "graph_buffer_reset": graph_reset_count > 0,
                "graph_buffer_reset_count": int(graph_reset_count),
                "pose_update_mode": pose_update_mode,
                "pose_read": (
                    _jsonable(dict(pose_read))
                    if isinstance(pose_read, Mapping)
                    else None
                ),
                "snapshot_timings": (
                    _jsonable(dict(snapshot_timings))
                    if isinstance(snapshot_timings, Mapping)
                    else None
                ),
            }
            if error is not None:
                payload["error"] = str(error)
            generator._rpent_obstacle_refresh_metrics = payload

        def full_refresh(
            *,
            started_ns: int,
            snapshot: dict[str, Any] | None,
            ignore_objects: Any,
            reason: str,
            fallback: bool,
            fallback_reason: str | None,
            clear_cache: bool = False,
        ) -> None:
            try:
                if (
                    isinstance(snapshot, dict)
                    and snapshot.get("content_sha256") is None
                ):
                    snapshot = capture_snapshot(
                        ignore_objects,
                        full_digest=True,
                    )
                if clear_cache:
                    clear_collision_world()
                call_full(ignore_objects)
                graph_reset_count = reset_all_graph_buffers()
            except Exception as exc:
                state["topology"] = None
                state["storage_refs"] = None
                state["kinematic_cache"] = None
                state["pose_batch_cache"] = None
                record(
                    started_ns=started_ns,
                    mode="full",
                    reason=reason,
                    mesh_count=(
                        int(snapshot["mesh_count"])
                        if isinstance(snapshot, dict)
                        else None
                    ),
                    fallback=fallback,
                    fallback_reason=fallback_reason,
                    graph_reset_count=0,
                    pose_update_mode="full_rebuild",
                    pose_read=(
                        snapshot.get("pose_read")
                        if isinstance(snapshot, dict)
                        else None
                    ),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            state["topology"] = (
                snapshot["topology"] if isinstance(snapshot, dict) else None
            )
            state["storage_refs"] = (
                snapshot.get("storage_refs")
                if isinstance(snapshot, dict)
                else None
            )
            state["kinematic_cache"] = (
                snapshot.get("kinematic_cache")
                if isinstance(snapshot, dict)
                else None
            )
            state["pose_batch_cache"] = (
                snapshot.get("pose_batch_cache")
                if isinstance(snapshot, dict)
                else None
            )
            state["initialized"] = True
            record(
                started_ns=started_ns,
                mode="full",
                reason=reason,
                mesh_count=(
                    int(snapshot["mesh_count"])
                    if isinstance(snapshot, dict)
                    else None
                ),
                fallback=fallback,
                fallback_reason=fallback_reason,
                graph_reset_count=graph_reset_count,
                pose_update_mode="full_rebuild",
                pose_read=(
                    snapshot.get("pose_read")
                    if isinstance(snapshot, dict)
                    else None
                ),
                snapshot_timings=(
                    snapshot.get("snapshot_timings")
                    if isinstance(snapshot, dict)
                    else None
                ),
            )

        def fast_update_locked(ignore_objects: Any = None) -> None:
            started_ns = time.monotonic_ns()
            state["count"] += 1
            snapshot: dict[str, Any] | None = None
            snapshot_error: str | None = None
            try:
                snapshot = capture_snapshot(
                    ignore_objects,
                    full_digest=(
                        state["topology"] is None or ignore_objects is not None
                    ),
                )
                if not isinstance(snapshot, dict):
                    raise RuntimeError("collision mesh snapshot is not a mapping")
            except Exception as exc:
                snapshot_error = f"{type(exc).__name__}: {exc}"

            if not state["initialized"]:
                full_refresh(
                    started_ns=started_ns,
                    snapshot=snapshot,
                    ignore_objects=ignore_objects,
                    reason="first_refresh",
                    fallback=False,
                    fallback_reason=snapshot_error,
                )
                return
            if state["topology"] is None:
                full_refresh(
                    started_ns=started_ns,
                    snapshot=snapshot,
                    ignore_objects=ignore_objects,
                    reason="fallback_full",
                    fallback=True,
                    fallback_reason=(
                        snapshot_error
                        or "fast refresh topology is unavailable"
                    ),
                )
                return
            if ignore_objects is not None:
                full_refresh(
                    started_ns=started_ns,
                    snapshot=snapshot,
                    ignore_objects=ignore_objects,
                    reason="ignore_objects",
                    fallback=False,
                    fallback_reason=snapshot_error,
                )
                return
            if snapshot is None:
                full_refresh(
                    started_ns=started_ns,
                    snapshot=None,
                    ignore_objects=None,
                    reason="fallback_full",
                    fallback=True,
                    fallback_reason=snapshot_error,
                )
                return
            if snapshot["topology"] != state["topology"]:
                full_refresh(
                    started_ns=started_ns,
                    snapshot=snapshot,
                    ignore_objects=None,
                    reason="topology_change",
                    fallback=False,
                    fallback_reason=None,
                    clear_cache=True,
                )
                return
            baseline_refs = state.get("storage_refs")
            current_refs = snapshot.get("storage_refs")
            if baseline_refs is not None or current_refs is not None:
                if (
                    not isinstance(baseline_refs, tuple)
                    or not isinstance(current_refs, tuple)
                    or len(baseline_refs) != len(current_refs)
                    or any(
                        str(old[0]) != str(new[0])
                        or old[1] is not new[1]
                        or old[2] is not new[2]
                        for old, new in zip(
                            baseline_refs,
                            current_refs,
                            strict=True,
                        )
                    )
                ):
                    full_refresh(
                        started_ns=started_ns,
                        snapshot=snapshot,
                        ignore_objects=None,
                        reason="mesh_storage_replaced",
                        fallback=False,
                        fallback_reason=None,
                        clear_cache=True,
                    )
                    return

            try:
                motion_gen_values = motion_generators()
                checkers = {
                    id(getattr(value, "world_coll_checker", None)): getattr(
                        value,
                        "world_coll_checker",
                        None,
                    )
                    for value in motion_gen_values
                }
                if len(checkers) != 1:
                    raise RuntimeError(
                        "generator does not share one collision world"
                    )
                checker = next(iter(checkers.values()))
                if not isinstance(checker, world_collision_type):
                    raise RuntimeError(
                        "collision world is not WorldMeshCollision"
                    )
                update_pose = getattr(checker, "update_obstacle_pose", None)
                env_mesh_names = getattr(checker, "_env_mesh_names", None)
                env_n_mesh = getattr(checker, "_env_n_mesh", None)
                if (
                    not callable(update_pose)
                    or env_mesh_names is None
                    or env_n_mesh is None
                ):
                    raise RuntimeError(
                        "WorldMeshCollision pose-update API is unavailable"
                    )
                mesh_count = int(_jsonable(env_n_mesh[0]))
                loaded_names = tuple(
                    sorted(str(name) for name in env_mesh_names[0][:mesh_count])
                )
                current_names = tuple(
                    sorted(str(name) for name, _pose in snapshot["poses"])
                )
                if loaded_names != current_names:
                    raise RuntimeError(
                        "WorldMeshCollision mesh registry topology changed"
                    )
                graph_planners = [
                    getattr(value, "graph_planner", None)
                    for value in motion_gen_values
                ]
                if not all(
                    callable(getattr(planner, "reset_buffer", None))
                    for planner in graph_planners
                ):
                    raise RuntimeError("CuRobo graph planner reset API is unavailable")
                pose_update_mode = "per_obstacle"
                update_mesh_pose = getattr(checker, "update_mesh_pose", None)
                world_model = getattr(checker, "world_model", None)
                if (
                    callable(update_mesh_pose)
                    and callable(pose_batch_factory)
                    and world_model is not None
                ):
                    import torch

                    loaded_order = tuple(
                        str(name) for name in env_mesh_names[0][:mesh_count]
                    )
                    loaded_index = {
                        name: index for index, name in enumerate(loaded_order)
                    }
                    if len(loaded_index) != mesh_count:
                        raise RuntimeError(
                            "WorldMeshCollision mesh registry contains duplicate names"
                        )
                    pose_rows = [
                        list(pose) for _name, pose in snapshot["poses"]
                    ]
                    pose_names = [
                        str(name) for name, _pose in snapshot["poses"]
                    ]
                    pose_indices = torch.as_tensor(
                        [loaded_index[name] for name in pose_names],
                        device=checker.tensor_args.device,
                        dtype=torch.int64,
                    )
                    update_mesh_pose(
                        w_obj_pose=pose_batch_factory(
                            pose_rows,
                            checker.tensor_args,
                        ),
                        env_obj_idx=pose_indices,
                        env_idx=0,
                    )
                    cpu_world = (
                        world_model[0]
                        if isinstance(world_model, list)
                        else world_model
                    )
                    cpu_objects = list(getattr(cpu_world, "objects", ()))
                    cpu_by_name = {
                        str(getattr(obstacle, "name", "")): obstacle
                        for obstacle in cpu_objects
                    }
                    if any(name not in cpu_by_name for name in pose_names):
                        raise RuntimeError(
                            "CuRobo CPU world model mesh registry changed"
                        )
                    for name, pose in zip(
                        pose_names,
                        pose_rows,
                        strict=True,
                    ):
                        cpu_by_name[name].pose = list(pose)
                    pose_update_mode = "batch_tensor_and_linear_cpu"
                else:
                    poses = [
                        (
                            str(name),
                            pose_factory(pose, checker.tensor_args),
                        )
                        for name, pose in snapshot["poses"]
                    ]
                    for name, pose in poses:
                        update_pose(
                            name=name,
                            w_obj_pose=pose,
                            env_idx=0,
                            update_cpu_reference=True,
                        )
                graph_reset_count = reset_all_graph_buffers()
            except Exception as exc:
                full_refresh(
                    started_ns=started_ns,
                    snapshot=snapshot,
                    ignore_objects=None,
                    reason="fallback_full",
                    fallback=True,
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                    clear_cache=True,
                )
                return
            state["topology"] = snapshot["topology"]
            state["storage_refs"] = snapshot.get("storage_refs")
            state["kinematic_cache"] = snapshot.get("kinematic_cache")
            state["pose_batch_cache"] = snapshot.get("pose_batch_cache")
            record(
                started_ns=started_ns,
                mode="pose_only",
                reason="unchanged_rigid_mesh_topology",
                mesh_count=int(snapshot["mesh_count"]),
                fallback=False,
                fallback_reason=None,
                graph_reset_count=graph_reset_count,
                pose_update_mode=pose_update_mode,
                pose_read=snapshot.get("pose_read"),
                snapshot_timings=snapshot.get("snapshot_timings"),
            )

        def fast_update(_generator: Any, ignore_objects: Any = None) -> None:
            with refresh_lock:
                fast_update_locked(ignore_objects)

        def invalidate_fast_refresh() -> None:
            with refresh_lock:
                state["topology"] = None
                state["storage_refs"] = None
                state["kinematic_cache"] = None
                state["pose_batch_cache"] = None
                state["initialized"] = False

        generator.update_obstacles = types.MethodType(fast_update, generator)
        generator._rpent_invalidate_obstacle_refresh = invalidate_fast_refresh
        generator._rpent_fast_obstacle_refresh_installed = True

    def _record_generator_recovery(self, event: dict[str, Any]) -> None:
        path = self.output_dir / "planner_generator_recovery.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_artifact_jsonable(event), sort_keys=True) + "\n")

    def _record_base_phase(self, event: dict[str, Any]) -> None:
        path = self.output_dir / "planner_base_phases.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"monotonic_ns": time.monotonic_ns(), **event}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_artifact_jsonable(payload), sort_keys=True) + "\n")

    def _quarantine_generator(
        self,
        *,
        kind: str,
        hand: str = "left",
        reason: str,
    ) -> dict[str, Any]:
        """Discard a generator whose temporary planner state is uncertain."""

        key = self._generator_key(kind=kind, hand=hand)
        discarded = self._generators.pop(key, None)
        self._invalid_generators.add(key)
        event = {
            "event": "generator_quarantined",
            "generator_key": key,
            "reason": str(reason),
            "monotonic_ns": time.monotonic_ns(),
            "requires_fresh_rebuild": True,
        }
        self._record_generator_recovery(event)
        del discarded
        gc.collect()
        return event

    def plan_base_identity_trajectory(
        self,
        *,
        timeout_s: float = RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
    ) -> dict[str, Any]:
        """Plan and collision-certify BASE at its current pose without acting."""

        warmup_wall_clock_s = min(
            float(timeout_s),
            RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
        )
        if not math.isfinite(warmup_wall_clock_s) or warmup_wall_clock_s <= 0.0:
            raise ValueError("BASE identity warmup timeout must be positive")
        robot = self._find_robot()
        current = np.asarray(
            self._base_xy_yaw(robot), dtype=np.float64
        ).reshape(-1)
        if current.shape not in {(3,), (4,)} or not np.isfinite(current).all():
            raise RuntimeError("R1Pro BASE pose unavailable during warmup")
        result = self._compute_base_plan(
            target_xyyaw=current[:3],
            timeout_s=warmup_wall_clock_s,
            skip_obstacle_update=False,
            attempt_timeout_cap_s=BASE_PLAN_ATTEMPT_TIMEOUT_S,
            solver_timeout_cap_s=BASE_PLAN_ATTEMPT_TIMEOUT_S - 2.0,
            planning_profile=RESET_IDENTITY_WARMUP_PROFILE,
            wall_clock_timeout_s=warmup_wall_clock_s,
        )
        if isinstance(result, dict):
            metrics = dict(result.get("metrics") or {})
            metrics.update(
                {
                    "identity_plan": True,
                    "env_actions_sent": 0,
                    "simulator_advanced": False,
                }
            )
            result["metrics"] = metrics
        return result

    def _warmup_dashboard_feedback(self) -> dict[str, Any]:
        """Compile every read-only BASE post-step safety path without acting."""

        started = time.monotonic()
        current_q = self.get_joint_positions()
        expected_attachments = {
            hand: self.get_attached_object(hand) for hand in ("left", "right")
        }
        isolation_reference = self.capture_navigation_isolation_reference()
        gripper_commands = isolation_reference.get("gripper_commands")
        if not isinstance(gripper_commands, dict) or any(
            side not in gripper_commands
            or not math.isfinite(float(gripper_commands[side]))
            for side in ("left", "right")
        ):
            raise RuntimeError(
                "Dashboard read-only safety feedback warmup has no gripper latch "
                "reference"
            )
        # ``navigation_isolation_report`` only needs a valid packed action to
        # verify that both gripper commands match their immutable call-start
        # latches.  Do not call ``joint_target_to_action`` here: its official
        # R1Pro adapter installation intentionally mutates the live robot and
        # belongs to the first admitted controller action, not a read-only
        # reset warmup.
        identity_action = np.zeros(ACTION_DIM, dtype=np.float32)
        for side in ("left", "right"):
            identity_action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = float(
                gripper_commands[side]
            )
        identity_action = validate_action_chunk(
            identity_action.reshape(1, ACTION_DIM)
        )[0]
        isolation = self.navigation_isolation_report(
            action=identity_action,
            reference=isolation_reference,
        )
        contact_baseline = self.capture_whole_body_contact_baseline(
            expected_attachments_by_hand=expected_attachments,
        )
        contact = self.whole_body_contact_report(
            baseline=contact_baseline,
            expected_attachments_by_hand=expected_attachments,
            allowed_expected_contact=None,
        )
        tracking = self.joint_tracking_report(current_q, hand=None)
        hard_tracking, hard_tracking_available = (
            _tracking_hard_deviation_report(tracking)
        )
        base_pose = np.asarray(self.get_base_pose(), dtype=np.float64).reshape(-1)
        checks = {
            "isolation_available": isolation.get("available") is True,
            "isolation_ok": isolation.get("ok") is True,
            "contact_baseline_available": contact_baseline.get("available") is True,
            "contact_available": contact.get("available") is True,
            "contact_clear": contact.get("unexpected_contact") is False,
            "tracking_available": tracking.get("available") is True,
            "tracking_hard_deviation_available": hard_tracking_available,
            "tracking_within_hard_limits": (
                hard_tracking.get("hard_deviation") is False
            ),
            "base_pose_finite": bool(
                base_pose.shape == (3,) and np.isfinite(base_pose).all()
            ),
        }
        checks = {name: bool(value) for name, value in checks.items()}
        if not all(checks.values()):
            raise RuntimeError(
                "Dashboard read-only safety feedback warmup failed closed: "
                f"{checks!r}"
            )
        return {
            "query": "read_only_post_step_safety_feedback",
            "ok": True,
            "checks": checks,
            "env_actions_sent": 0,
            "simulator_advanced": False,
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    def warmup(self) -> dict[str, Any]:
        """Compile BASE and whole-body planner kernels used by analytic primitives.

        This runs before any public planner-tool deadline starts.  It performs
        one BASE identity query plus one selected-EEF collision-certified
        identity query per hand, but never executes an action or advances the
        simulator.
        """
        started = time.monotonic()
        report: dict[str, Any] = {
            "status": "running",
            "stages": {},
            "identity_warmup": {
                "env_actions_sent": 0,
                "simulator_advanced": False,
                "base": None,
                "torso": None,
                "hands": [],
                "dashboard_feedback": None,
            },
        }
        path = self.output_dir / "planner_curobo_warmup.json"
        report["artifact"] = str(path)

        def save() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        def stage(name: str, call: Any) -> Any:
            stage_started = time.monotonic()
            try:
                with _wall_clock_deadline(
                    RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S,
                    f"reset identity warmup stage {name}",
                ):
                    value = call()
            except Exception as exc:
                report["stages"][name] = {
                    "ok": False,
                    "elapsed_s": round(time.monotonic() - stage_started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["status"] = "error"
                report["elapsed_s"] = round(time.monotonic() - started, 3)
                save()
                raise
            report["stages"][name] = {
                "ok": True,
                "elapsed_s": round(time.monotonic() - stage_started, 3),
                "result": _artifact_jsonable(value),
            }
            return value

        base_plan = stage(
            "base_current_pose_identity_trajectory",
            lambda: self.plan_base_identity_trajectory(
                timeout_s=RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S
            ),
        )
        if not bool(base_plan.get("ok", False)):
            report["status"] = "error"
            report["elapsed_s"] = round(time.monotonic() - started, 3)
            save()
            raise RuntimeError(
                "BASE current-pose cuRobo warmup failed closed: "
                f"{base_plan.get('stop_reason', 'unknown')}"
            )
        base_metrics = (
            dict(base_plan.get("metrics"))
            if isinstance(base_plan.get("metrics"), dict)
            else {}
        )
        report["identity_warmup"]["base"] = {
            "query": "current_pose_identity_trajectory",
            "ok": True,
            "stop_reason": base_plan.get("stop_reason"),
            "trajectory_waypoints": base_metrics.get("trajectory_waypoints"),
            "collision_admitted": (
                base_metrics.get("collision_admission", {}).get("admitted") is True
            ),
        }
        report["identity_warmup"]["dashboard_feedback"] = stage(
            "dashboard_read_only_safety_feedback",
            self._warmup_dashboard_feedback,
        )
        torso_pose = self.get_torso_pose()
        torso_plan = stage(
            "torso_current_pose_identity_trajectory",
            lambda: self.plan_torso_trajectory(
                target_z_m=float(torso_pose[0][2]),
                timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                start_torso_pose=torso_pose,
            ),
        )
        if not bool(torso_plan.get("ok", False)):
            report["status"] = "error"
            report["elapsed_s"] = round(time.monotonic() - started, 3)
            save()
            raise RuntimeError(
                "torso current-pose CuRobo warmup failed closed: "
                f"{torso_plan.get('stop_reason', 'unknown')}"
            )
        torso_metrics = (
            dict(torso_plan.get("metrics"))
            if isinstance(torso_plan.get("metrics"), dict)
            else {}
        )
        report["identity_warmup"]["torso"] = {
            "query": "current_pose_identity_trajectory",
            "ok": True,
            "stop_reason": torso_plan.get("stop_reason"),
            "trajectory_waypoints": int(
                len(
                    np.asarray(
                        torso_plan.get("joint_trajectory"),
                        dtype=np.float32,
                    )
                )
            ),
            "collision_admitted": (
                torso_metrics.get("collision_admission", {}).get("admitted")
                is True
            ),
            "active_dof_count": torso_metrics.get("active_dof_count"),
            "env_actions_sent": 0,
            "simulator_advanced": False,
        }

        for hand in ("left", "right"):
            def warm_whole_body(hand: str = hand) -> dict[str, Any]:
                eef_pose = self.get_eef_pose(hand)
                if eef_pose is None:
                    raise RuntimeError(
                        f"R1Pro {hand} EEF pose unavailable during warmup"
                    )
                return self.plan_whole_body_trajectory(
                    hand=hand,
                    target_xyz=np.asarray(eef_pose[0], dtype=np.float64),
                    target_quat_xyzw=np.asarray(
                        eef_pose[1], dtype=np.float64
                    ),
                    timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                    search_profile=RESET_IDENTITY_WARMUP_PROFILE,
                    wall_clock_timeout_s=(
                        RESET_IDENTITY_WARMUP_STAGE_DEADLINE_S
                    ),
                )

            whole_body_plan = stage(
                f"whole_body_{hand}_identity_trajectory",
                warm_whole_body,
            )
            if not bool(whole_body_plan.get("ok", False)):
                report["status"] = "error"
                report["elapsed_s"] = round(time.monotonic() - started, 3)
                save()
                raise RuntimeError(
                    f"{hand} whole-body cuRobo warmup failed closed: "
                    f"{whole_body_plan.get('stop_reason', 'unknown')}"
                )
            identity_metrics = (
                whole_body_plan.get("metrics")
                if isinstance(whole_body_plan.get("metrics"), dict)
                else {}
            )
            report["identity_warmup"]["hands"].append(
                {
                    "hand": hand,
                    "query": "identity_trajectory",
                    "ok": whole_body_plan.get("ok") is True,
                    "stop_reason": whole_body_plan.get("stop_reason"),
                    "trajectory_waypoints": identity_metrics.get("trajectory_waypoints"),
                    "collision_admitted": (
                        identity_metrics.get("collision_admission", {}).get("admitted")
                        is True
                    ),
                    "active_dof_count": identity_metrics.get("active_dof_count"),
                    "selected_eef_goal_count": identity_metrics.get(
                        "selected_eef_goal_count"
                    ),
                }
            )
        report["status"] = "complete"
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        save()
        return report

    def warmup_attached_arm(
        self,
        *,
        hand: str,
        expected_attached_root: Any,
    ) -> dict[str, Any]:
        """Warm the attachment-aware held-arm planner without executing motion."""

        hand = _normalize_hand(hand)
        started = time.monotonic()
        path = self.output_dir / "planner_attached_arm_warmup.json"
        report: dict[str, Any] = {
            "status": "running",
            "generator_kind": "attached_arm",
            "held_hand": hand,
            "base_generator_warmed": False,
            "unrelated_press_arm_generator_warmed": False,
            "stages": {},
            "warnings": [],
        }

        def save() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        robot = self._find_robot()
        pre_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float32
        ).reshape(-1)
        try:
            attached = self.get_attached_object(hand)
            if not isinstance(attached, dict) or not attached:
                raise RuntimeError(
                    f"{hand} attached-arm warmup requires the held collision body"
                )
            expected_path = str(
                getattr(expected_attached_root, "prim_path", "")
            ).rstrip("/")
            root_matches_expected = any(
                value is expected_attached_root
                or (
                    bool(expected_path)
                    and str(getattr(value, "prim_path", "")).rstrip("/")
                    == expected_path
                )
                for value in attached.values()
            )
            if not root_matches_expected:
                raise RuntimeError(
                    f"{hand} attached body does not match the selected object"
                )
            report["attached_collision_body"] = {
                "available": True,
                "root_matches_expected_object": True,
                "eef_links": sorted(str(key) for key in attached),
                "prim_paths": sorted(
                    str(getattr(value, "prim_path", "")) for value in attached.values()
                ),
            }
            generator = self._generator(kind="attached_arm", hand=hand)
            target_xyz, target_quat = self._curobo_eef_poses(
                generator, pre_q.reshape(1, -1)
            )
            if (
                target_xyz.shape != (1, 3)
                or target_quat.shape != (1, 4)
                or not np.isfinite(target_xyz).all()
                or not np.isfinite(target_quat).all()
            ):
                raise RuntimeError(
                    f"R1Pro {hand} CuRobo EEF pose unavailable during warmup"
                )
            report["stages"]["current_q_curobo_fk"] = {
                "ok": True,
                "target_xyz": target_xyz[0].tolist(),
                "target_quat_xyzw": target_quat[0].tolist(),
            }
            identity_plan = self.plan_attached_arm_trajectory(
                hand=hand,
                target_xyz=target_xyz[0],
                target_quat_xyzw=target_quat[0],
                timeout_s=120.0,
                attached_obj=attached,
            )
            report["stages"]["current_pose_attached_ik"] = {
                "ok": bool(identity_plan.get("ok", False)),
                "diagnostic_only": True,
                "stop_reason": identity_plan.get("stop_reason"),
                "metrics": _artifact_jsonable(identity_plan.get("metrics", {})),
            }
            if not bool(identity_plan.get("ok", False)):
                report["warnings"].append(
                    f"{hand} identity IK diagnostic did not return a solution: "
                    f"{identity_plan.get('stop_reason', 'unknown')}"
                )
            q_path = np.asarray(
                _jsonable(identity_plan.get("joint_trajectory")), dtype=np.float32
            )
            q_path_valid = bool(
                q_path.ndim == 2
                and q_path.shape[0] >= 1
                and q_path.shape[1] == pre_q.size
                and np.isfinite(q_path).all()
            )
            if not q_path_valid:
                report["warnings"].append(
                    "identity IK diagnostic returned no usable joint path"
                )
            post_q = np.asarray(
                _jsonable(robot.get_joint_positions()), dtype=np.float32
            ).reshape(-1)
            if post_q.shape != pre_q.shape or not np.isfinite(post_q).all():
                raise RuntimeError("attached-arm warmup changed the robot q layout")
            pose_jump = float(np.max(np.abs(post_q - pre_q)))
            report["robot_q_pose_jump_max"] = pose_jump
            if pose_jump > 1e-6:
                raise RuntimeError("attached-arm warmup moved the robot")
            if q_path_valid:
                first_delta = float(np.max(np.abs(q_path[0] - pre_q)))
                terminal_delta = float(np.max(np.abs(q_path[-1] - pre_q)))
                path_delta = float(np.max(np.abs(q_path - pre_q.reshape(1, -1))))
                report["stages"]["identity_ik_path"] = {
                    "ok": True,
                    "diagnostic_only": True,
                    "trajectory_waypoints": int(q_path.shape[0]),
                    "first_max_joint_delta_rad": first_delta,
                    "terminal_max_joint_delta_rad": terminal_delta,
                    "path_max_joint_delta_rad": path_delta,
                    "robot_q_pose_jump_max": pose_jump,
                }
            else:
                report["stages"]["identity_ik_path"] = {
                    "ok": False,
                    "diagnostic_only": True,
                    "trajectory_waypoints": 0,
                    "robot_q_pose_jump_max": pose_jump,
                }
        except Exception as exc:
            report["status"] = "error"
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["elapsed_s"] = round(time.monotonic() - started, 3)
            save()
            raise
        report["status"] = "complete"
        report["target_plan_validation"] = (
            "warmup identity results are diagnostic only; public motion uses fresh "
            "no-collision-admission IK"
        )
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        report["artifact"] = str(path)
        save()
        return report

    def _lazy_imports(self) -> None:
        if self._curobo_cls is not None:
            return
        import torch
        from omnigibson.action_primitives.curobo import (
            CuRoboEmbodimentSelection,
            CuRoboMotionGenerator,
        )

        self._torch = torch
        self._curobo_cls = CuRoboMotionGenerator
        self._embodiment_cls = CuRoboEmbodimentSelection

    def _find_robot(self) -> Any:
        if self._robot is not None:
            return self._robot
        candidates = [
            self.env_facade,
            getattr(self.env_facade, "_env", None),
            getattr(getattr(self.env_facade, "_env", None), "_env", None),
            getattr(getattr(self.env_facade, "_env", None), "_direct_process", None),
            getattr(
                getattr(
                    getattr(self.env_facade, "_env", None), "_direct_process", None
                ),
                "env",
                None,
            ),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            robots = getattr(candidate, "robots", None)
            if robots:
                self._robot = robots[0]
                return self._robot
            scene = getattr(candidate, "scene", None)
            robots = getattr(scene, "robots", None) if scene is not None else None
            if robots:
                self._robot = robots[0]
                return self._robot
        try:
            import omnigibson as og

            if og.sim.scenes and og.sim.scenes[0].robots:
                self._robot = og.sim.scenes[0].robots[0]
                return self._robot
        except Exception:
            pass
        raise RuntimeError("could not locate the R1Pro robot in the BEHAVIOR env")

    def _asset_curobo_dir(self, robot: Any) -> Path:
        curobo_path = getattr(robot, "curobo_path", None)
        if isinstance(curobo_path, dict):
            for value in curobo_path.values():
                path = Path(str(value))
                if path.is_file():
                    return path.parent
        elif curobo_path:
            path = Path(str(curobo_path))
            if path.is_file():
                return path.parent
        roots = [
            os.environ.get("OMNIGIBSON_ASSET_PATH"),
            os.environ.get("BEHAVIOR_ASSET_PATH"),
            os.environ.get("RPENT_RLINF_ROOT"),
            os.environ.get("RLINF_REPO_PATH"),
        ]
        for root in roots:
            if not root:
                continue
            base = Path(root).expanduser()
            candidates = [
                base
                / ".venv-behavior"
                / "BEHAVIOR-1K"
                / "datasets"
                / "omnigibson-robot-assets"
                / "models"
                / "r1pro"
                / "curobo",
                base
                / "datasets"
                / "omnigibson-robot-assets"
                / "models"
                / "r1pro"
                / "curobo",
            ]
            for candidate in candidates:
                if (candidate / "r1pro_description_curobo_arm.yaml").is_file():
                    return candidate
        raise RuntimeError("could not locate R1Pro cuRobo YAML assets")

    def _hand_config_path(
        self,
        hand: str,
        *,
        lock_trunk: bool = False,
        motion_scope: str = "arm_only",
    ) -> Path:
        hand = _normalize_hand(hand)
        if motion_scope not in {"arm_only", "arm_with_trunk"}:
            raise ValueError(f"unsupported arm motion scope {motion_scope!r}")
        cache_key = (
            f"{hand}:arm_with_trunk"
            if motion_scope == "arm_with_trunk"
            else (f"{hand}:attached" if lock_trunk else hand)
        )
        if cache_key in self._config_paths:
            return self._config_paths[cache_key]
        robot = self._find_robot()
        source = self._asset_curobo_dir(robot) / "r1pro_description_curobo_arm.yaml"
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required to generate hand-specific cuRobo config"
            ) from exc

        with source.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kinematics = cfg["robot_cfg"]["kinematics"]
        eef_link = self._eef_link_name(robot, hand)
        kinematics["ee_link"] = eef_link
        lock_joints = dict(kinematics.get("lock_joints") or {})
        inactive = "right" if hand == "left" else "left"
        inactive_eef_link = self._eef_link_name(robot, inactive)
        kinematics["link_names"] = [inactive_eef_link]
        for joint in [f"{inactive}_arm_joint{i}" for i in range(1, 8)]:
            self._validate_joint_name(robot, joint)
            lock_joints.setdefault(joint, None)
        for side in ("left", "right"):
            for suffix in ("finger_joint1", "finger_joint2"):
                joint = f"{side}_gripper_{suffix}"
                self._validate_joint_name(robot, joint)
                lock_joints.setdefault(joint, None)
        for joint in (
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_z_joint",
            "base_footprint_rx_joint",
            "base_footprint_ry_joint",
            "base_footprint_rz_joint",
        ):
            self._validate_joint_name(robot, joint)
            lock_joints.setdefault(joint, None)
        trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
        joint_names = list((getattr(robot, "joints", {}) or {}).keys())
        if len(trunk_indices) != 4 or max(trunk_indices, default=-1) >= len(
            joint_names
        ):
            raise RuntimeError("R1Pro arm trunk joint indices unavailable")
        locked_trunk_indices = (
            trunk_indices[-1:] if motion_scope == "arm_with_trunk" else trunk_indices
        )
        for index in locked_trunk_indices:
            joint = joint_names[index]
            self._validate_joint_name(robot, joint)
            lock_joints.setdefault(joint, None)
        self._validate_lock_joint_names(robot, lock_joints)
        kinematics["lock_joints"] = dict(sorted(lock_joints.items()))
        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = (
            "_arm_with_trunk"
            if motion_scope == "arm_with_trunk"
            else ("_attached" if lock_trunk else "")
        )
        out = out_dir / f"r1pro_description_curobo_arm_{hand}{suffix}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._config_paths[cache_key] = out
        return out

    def _whole_body_config_path(self, hand: str) -> Path:
        """Generate a live-regularized 21-DOF R1Pro cuRobo configuration."""

        hand = _normalize_hand(hand)
        cache_key = f"{hand}:whole_body"
        if cache_key in self._config_paths:
            return self._config_paths[cache_key]
        robot = self._find_robot()
        source = self._asset_curobo_dir(robot) / "r1pro_description_curobo_default.yaml"
        if not source.is_file():
            raise RuntimeError(f"official R1Pro whole-body config is missing: {source}")
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required to generate whole-body cuRobo config"
            ) from exc

        with source.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        kinematics = cfg["robot_cfg"]["kinematics"]
        cspace = kinematics["cspace"]
        joint_names = [str(name) for name in cspace["joint_names"]]
        if len(joint_names) != 28 or len(set(joint_names)) != 28:
            raise RuntimeError(
                "official R1Pro whole-body config must expose 28 unique joints"
            )
        for joint_name in joint_names:
            self._validate_joint_name(robot, joint_name)
        active = [name for name in joint_names if name in WHOLE_BODY_ACTIVE_JOINT_NAMES]
        locked = [name for name in joint_names if name in WHOLE_BODY_LOCKED_JOINT_NAMES]
        if (
            len(active) != 21
            or set(active) != set(WHOLE_BODY_ACTIVE_JOINT_NAMES)
            or len(locked) != 7
            or set(locked) != set(WHOLE_BODY_LOCKED_JOINT_NAMES)
        ):
            raise RuntimeError(
                f"invalid R1Pro whole-body joint partition: active={active!r}, "
                f"locked={locked!r}"
            )

        kinematics["ee_link"] = self._eef_link_name(robot, hand)
        # The non-selected EEF is intentionally unconstrained.  Both arms stay
        # in the 21-DOF optimization and both attachment bodies remain in the
        # collision model, but no inactive-link world pose is submitted.
        kinematics["link_names"] = []
        kinematics["lock_joints"] = dict.fromkeys(WHOLE_BODY_LOCKED_JOINT_NAMES)
        self._validate_lock_joint_names(robot, kinematics["lock_joints"])

        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        robot_joint_names = list((getattr(robot, "joints", {}) or {}).keys())
        if len(q) != len(robot_joint_names):
            raise RuntimeError("R1Pro joint-position/name layout is inconsistent")
        q_by_name = {
            str(name): float(q[index]) for index, name in enumerate(robot_joint_names)
        }
        if any(name not in q_by_name for name in joint_names):
            raise RuntimeError("R1Pro live retract configuration is incomplete")
        cspace["retract_config"] = [q_by_name[name] for name in joint_names]

        weights = [1.0] * len(joint_names)
        cspace["cspace_distance_weight"] = weights
        cspace["null_space_weight"] = weights

        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"r1pro_description_curobo_whole_body_{hand}.yaml"
        with out.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
        self._config_paths[cache_key] = out
        return out

    def _torso_config_path(self) -> Path:
        """Generate the torso-link-only R1Pro CuRobo configuration."""

        cache_key = "torso"
        if cache_key in self._config_paths:
            return self._config_paths[cache_key]
        robot = self._find_robot()
        source = self._asset_curobo_dir(robot) / "r1pro_description_curobo_default.yaml"
        if not source.is_file():
            raise RuntimeError(f"official R1Pro whole-body config is missing: {source}")
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required to generate the torso CuRobo config"
            ) from exc

        with source.open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        kinematics = cfg["robot_cfg"]["kinematics"]
        cspace = kinematics["cspace"]
        joint_names = tuple(str(name) for name in cspace["joint_names"])
        if len(joint_names) != 28 or len(set(joint_names)) != 28:
            raise RuntimeError(
                "official R1Pro torso config must expose 28 unique joints"
            )
        for joint_name in joint_names:
            self._validate_joint_name(robot, joint_name)
        links = getattr(robot, "links", {}) or {}
        if TORSO_LINK_NAME not in links:
            raise RuntimeError(f"R1Pro link {TORSO_LINK_NAME!r} is unavailable")
        if not set(TORSO_ACTIVE_JOINT_NAMES).issubset(joint_names):
            raise RuntimeError("R1Pro torso active joints are unavailable")
        locked_names = tuple(
            name for name in joint_names if name not in TORSO_ACTIVE_JOINT_NAMES
        )
        if (
            len(locked_names) != 25
            or TORSO_LOCKED_JOINT_NAME not in locked_names
        ):
            raise RuntimeError("R1Pro torso locked-joint partition is invalid")
        kinematics["ee_link"] = TORSO_LINK_NAME
        kinematics["link_names"] = []
        kinematics["lock_joints"] = dict.fromkeys(locked_names)
        self._validate_lock_joint_names(robot, kinematics["lock_joints"])

        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        robot_names = tuple(str(name) for name in (getattr(robot, "joints", {}) or {}))
        if (
            len(robot_names) != len(set(robot_names))
            or set(robot_names) != set(joint_names)
            or q.shape != (len(robot_names),)
        ):
            raise RuntimeError("R1Pro live torso retract layout is inconsistent")
        q_by_name = {
            name: float(q[index]) for index, name in enumerate(robot_names)
        }
        cspace["retract_config"] = [
            q_by_name[name] for name in joint_names
        ]
        cspace["cspace_distance_weight"] = [1.0] * len(joint_names)
        cspace["null_space_weight"] = [1.0] * len(joint_names)

        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "r1pro_description_curobo_torso.yaml"
        with out.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
        self._config_paths[cache_key] = out
        return out

    def _base_config_path(self) -> Path:
        key = "base"
        if key in self._config_paths:
            return self._config_paths[key]
        robot = self._find_robot()
        source = self._asset_curobo_dir(robot) / "r1pro_description_curobo_base.yaml"
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "PyYAML is required to generate base cuRobo config"
            ) from exc
        with source.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        kinematics = cfg["robot_cfg"]["kinematics"]
        lock_joints = dict(kinematics.get("lock_joints") or {})
        self._validate_lock_joint_names(robot, lock_joints)
        kinematics["lock_joints"] = dict(sorted(lock_joints.items()))
        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "r1pro_description_curobo_base_runtime.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._config_paths[key] = out
        return out

    def _eef_link_name(self, robot: Any, hand: str) -> str:
        expected = EEF_LINK_BY_HAND[_normalize_hand(hand)]
        eef_names = getattr(robot, "eef_link_names", None)
        if isinstance(eef_names, dict):
            value = eef_names.get(hand)
            if value:
                expected = str(value)
        canonical = EEF_LINK_BY_HAND[_normalize_hand(hand)]
        if expected != canonical:
            raise RuntimeError(
                f"R1Pro {hand} EEF link drifted from the BEHAVIOR collision "
                f"contract: runtime={expected!r} expected={canonical!r}"
            )
        links = getattr(robot, "links", {}) or {}
        if expected not in links:
            raise RuntimeError(
                f"R1Pro {hand} EEF link {expected!r} not found in robot links"
            )
        return expected

    def _validate_joint_name(self, robot: Any, joint_name: str) -> None:
        joints = getattr(robot, "joints", {}) or {}
        if joint_name not in joints:
            raise RuntimeError(f"R1Pro cuRobo lock joint {joint_name!r} not found")

    def _validate_lock_joint_names(
        self, robot: Any, lock_joints: dict[str, Any]
    ) -> None:
        for joint in lock_joints:
            self._validate_joint_name(robot, str(joint))

    def _generator(self, *, kind: str, hand: str = "left") -> Any:
        self._lazy_imports()
        robot = self._find_robot()
        key = self._generator_key(kind=kind, hand=hand)
        if key in self._generators:
            return self._generators[key]
        if kind in {"arm", "attached_arm", "arm_with_trunk", "whole_body", "torso"}:
            if kind == "whole_body":
                config_path = self._whole_body_config_path(hand)
            elif kind == "torso":
                config_path = self._torso_config_path()
            else:
                motion_scope = (
                    "arm_with_trunk" if kind == "arm_with_trunk" else "arm_only"
                )
                config_path = (
                    self._hand_config_path(
                        hand,
                        lock_trunk=False,
                        motion_scope=motion_scope,
                    )
                    if motion_scope == "arm_with_trunk"
                    else self._hand_config_path(
                        hand,
                        lock_trunk=kind == "attached_arm",
                    )
                )
            robot_cfg_path: Any = str(config_path)
            use_default_embodiment_only = True
        elif kind == "base":
            robot_cfg_path = dict(getattr(robot, "curobo_path", {}) or {})
            if not robot_cfg_path:
                raise RuntimeError(
                    "R1Pro does not expose official cuRobo embodiment configs"
                )
            emb_sel = self._embodiment_cls.BASE
            robot_cfg_path[emb_sel] = str(self._base_config_path())
            use_default_embodiment_only = False
        else:
            raise ValueError(f"unknown cuRobo generator kind {kind!r}")
        assert self._curobo_cls is not None
        generator_cls = self._curobo_cls
        if kind in {"base", "whole_body", "torso"}:
            workspace_limit_m = self._base_prismatic_workspace_limit(robot)
            parent_cls = generator_cls

            class _SceneWorkspaceCuroboMotionGenerator(parent_cls):
                """Official OG generator with scene-sized virtual x/y bounds."""

                def update_joint_limits(inner_self, robot_cfg_obj, inner_emb_sel):
                    super().update_joint_limits(robot_cfg_obj, inner_emb_sel)
                    joint_limits = (
                        robot_cfg_obj.kinematics.kinematics_config.joint_limits
                    )
                    for joint_name in inner_self.robot.base_joint_names:
                        if joint_name not in joint_limits.joint_names:
                            continue
                        joint = inner_self.robot.joints[joint_name]
                        if str(getattr(joint, "axis", "")).upper() not in {"X", "Y"}:
                            continue
                        index = joint_limits.joint_names.index(joint_name)
                        joint_limits.position[0][index] = -workspace_limit_m
                        joint_limits.position[1][index] = workspace_limit_m

                def plan_batch(
                    inner_self,
                    start_state,
                    goal_pose,
                    plan_config,
                    link_poses=None,
                    emb_sel=None,
                ):
                    """Apply one Behavior-owned per-query MotionGen policy."""

                    override = getattr(inner_self, "_rpent_plan_override", None)
                    if isinstance(override, dict):
                        plan_config.enable_graph = bool(
                            override.get("enable_graph", False)
                        )
                        plan_config.enable_graph_attempt = None
                        plan_config.timeout = float(override["timeout_s"])
                        if bool(override.get("position_only", False)):
                            from curobo.rollout.cost.pose_cost import PoseCostMetric

                            plan_config.pose_cost_metric = PoseCostMetric(
                                reach_partial_pose=True,
                                reach_vec_weight=inner_self._tensor_args.to_device(
                                    [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
                                ),
                            )
                    kwargs = {"link_poses": link_poses}
                    if emb_sel is not None:
                        kwargs["emb_sel"] = emb_sel
                    return super().plan_batch(
                        start_state,
                        goal_pose,
                        plan_config,
                        **kwargs,
                    )

            generator_cls = _SceneWorkspaceCuroboMotionGenerator
        generator = generator_cls(
            robot,
            robot_cfg_path=robot_cfg_path,
            motion_cfg_kwargs={
                "trajopt_tsteps": 32,
                "num_trajopt_seeds": 4,
                "num_graph_seeds": 4,
                "finetune_trajopt_iters": 100,
                "self_collision_check": kind in {
                    "arm_with_trunk",
                    "whole_body",
                    "torso",
                },
            },
            batch_size=2,
            use_cuda_graph=False,
            use_default_embodiment_only=use_default_embodiment_only,
        )
        if kind in {"base", "whole_body", "torso"}:
            self._install_fast_obstacle_refresh(generator)
        self._generators[key] = generator
        if key in self._invalid_generators:
            self._invalid_generators.remove(key)
            self._record_generator_recovery(
                {
                    "event": "generator_rebuilt",
                    "generator_key": key,
                    "monotonic_ns": time.monotonic_ns(),
                    "fresh_instance": True,
                }
            )
        return generator

    def _base_prismatic_workspace_limit(self, robot: Any) -> float:
        """Cover the loaded scene while preserving OG's official BASE model.

        OmniGibson 3.7.2 unconditionally clamps the virtual holonomic x/y
        joints to +/-5 m.  Public BEHAVIOR scenes may reset the robot outside
        that interval (instance 211 starts at x=5.213 m), which makes even the
        current BASE pose an impossible IK goal.  Only these two virtual
        workspace bounds are widened; all physical, velocity, acceleration,
        and locked-joint limits remain official.
        """

        if self._base_workspace_limit_m is not None:
            return self._base_workspace_limit_m
        coordinates: list[float] = []
        try:
            base = self._base_xy_yaw(robot)
            coordinates.extend([abs(float(base[0])), abs(float(base[1]))])
        except Exception:
            pass
        try:
            scene_objects = self._scene(robot).objects
        except Exception:
            scene_objects = ()
        if isinstance(scene_objects, Mapping):
            scene_objects = scene_objects.values()
        for obj in scene_objects:
            try:
                position, _orientation = obj.get_position_orientation()
                xy = np.asarray(_jsonable(position), dtype=np.float64).reshape(-1)
                if xy.size >= 2 and np.isfinite(xy[:2]).all():
                    coordinates.extend([abs(float(xy[0])), abs(float(xy[1]))])
            except Exception:
                continue
        # Two metres covers object extents and a safe approach outside the
        # scene envelope without creating an unbounded optimization domain.
        self._base_workspace_limit_m = max(5.0, max(coordinates, default=5.0) + 2.0)
        return self._base_workspace_limit_m

    def get_eef_pose(self, hand: str) -> tuple[np.ndarray, np.ndarray] | None:
        robot = self._find_robot()
        link_name = EEF_LINK_BY_HAND[_normalize_hand(hand)]
        link = getattr(robot, "links", {}).get(link_name)
        if link is None:
            return None
        pos, quat = link.get_position_orientation()
        return np.asarray(_jsonable(pos), dtype=np.float64), np.asarray(
            _jsonable(quat), dtype=np.float64
        )

    def get_eef_to_fingertip_length(self, hand: str) -> float:
        """Return the R1Pro contact offset along the EEF local +Z axis."""

        hand = _normalize_hand(hand)
        robot = self._find_robot()
        by_hand = getattr(robot, "eef_to_fingertip_lengths", None)
        values = None if by_hand is None else by_hand.get(hand)
        lengths = [] if not values else [float(value) for value in values.values()]
        if not lengths or not all(
            np.isfinite(value) and value > 0 for value in lengths
        ):
            raise RuntimeError(f"R1Pro {hand} fingertip offset is unavailable")
        return float(np.mean(lengths))

    def get_assisted_grasp_outward_ray_geometry(self, hand: str) -> dict[str, Any]:
        """Resolve OG's outward assisted-grasp ray plane in EEF local +Z.

        OG stores each ray endpoint in its finger link's local frame.  Compute
        the live world positions exactly as OG does, transform them into the
        selected EEF frame, and select the positive-Z start/end pair.  No link
        or prim identity leaves this backend method.
        """

        hand = _normalize_hand(hand)
        robot = self._find_robot()
        eef_pose = self.get_eef_pose(hand)
        if eef_pose is None:
            raise RuntimeError(f"R1Pro {hand} EEF pose unavailable for AG ray")
        eef_position = np.asarray(eef_pose[0], dtype=np.float64).reshape(3)
        eef_quat = _quat_xyzw(eef_pose[1])
        assert eef_quat is not None
        world_to_eef_quat = np.array(
            [-eef_quat[0], -eef_quat[1], -eef_quat[2], eef_quat[3]],
            dtype=np.float64,
        )
        start_by_hand = getattr(robot, "assisted_grasp_start_points", None)
        end_by_hand = getattr(robot, "assisted_grasp_end_points", None)
        start_points = None if start_by_hand is None else start_by_hand.get(hand)
        end_points = None if end_by_hand is None else end_by_hand.get(hand)
        if not start_points or not end_points:
            raise RuntimeError(f"R1Pro {hand} assisted-grasp rays are unavailable")

        def points_in_eef(points: Any) -> np.ndarray:
            transformed = []
            for point in points:
                link_name = getattr(point, "link_name", None)
                local_position = getattr(point, "position", None)
                link = getattr(robot, "links", {}).get(link_name)
                if link is None or local_position is None:
                    raise RuntimeError("assisted-grasp ray endpoint link unavailable")
                link_position, link_quat = link.get_position_orientation()
                world_position = np.asarray(
                    _jsonable(link_position), dtype=np.float64
                ).reshape(3) + _quat_rotate_vector_xyzw(
                    link_quat,
                    np.asarray(_jsonable(local_position), dtype=np.float64).reshape(3),
                )
                transformed.append(
                    _quat_rotate_vector_xyzw(
                        world_to_eef_quat,
                        world_position - eef_position,
                    )
                )
            result = np.asarray(transformed, dtype=np.float64).reshape(-1, 3)
            if not np.isfinite(result).all():
                raise RuntimeError("assisted-grasp ray geometry is non-finite")
            return result

        start_eef = points_in_eef(start_points)
        end_eef = points_in_eef(end_points)
        start_outward = start_eef[int(np.argmax(start_eef[:, 2]))]
        end_outward = end_eef[int(np.argmax(end_eef[:, 2]))]
        start_inward_z = float(np.min(start_eef[:, 2]))
        end_inward_z = float(np.min(end_eef[:, 2]))
        start_offset = float(start_outward[2])
        end_offset = float(end_outward[2])
        plane_mismatch = abs(start_offset - end_offset)
        # Align the furthest positive-Z endpoint to the guarded penetration
        # plane. Using the mean would let the more outward endpoint penetrate
        # by an additional half of the start/end plane mismatch.
        offset = max(start_offset, end_offset)
        ray_span = float(np.linalg.norm(end_outward - start_outward))
        if (
            start_offset <= 0.0
            or end_offset <= 0.0
            or start_inward_z >= 0.0
            or end_inward_z >= 0.0
            or plane_mismatch > 0.002
            or ray_span <= 1e-6
        ):
            raise RuntimeError("assisted-grasp outward ray plane is invalid")
        return {
            "available": True,
            "outward_offset_m": offset,
            "start_outward_offset_m": start_offset,
            "end_outward_offset_m": end_offset,
            "plane_mismatch_m": plane_mismatch,
            "outward_offset_selection": "positive_z_endpoint_max",
            "ray_span_m": ray_span,
            "start_point_count": int(len(start_eef)),
            "end_point_count": int(len(end_eef)),
            "frame": "eef_local_positive_z",
        }

    def check_arm_reachability(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        base_xyyaw: np.ndarray | None = None,
        timeout_s: float = 5.0,
        skip_obstacle_update: bool = False,
        attached_obj: Any = _ATTACHMENT_UNSET,
    ) -> tuple[bool, str, dict[str, Any]]:
        del skip_obstacle_update
        if float(timeout_s) <= 0.0:
            return False, "timeout", {"timeout_s": float(timeout_s)}
        try:
            result = self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=min(5.0, float(timeout_s)),
                ik_only=True,
                base_xyyaw=base_xyyaw,
                attached_obj=(
                    self.get_attached_object(hand)
                    if attached_obj is _ATTACHMENT_UNSET
                    else attached_obj
                ),
            )
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="arm",
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return (
                False,
                "timeout" if isinstance(exc, TimeoutError) else "planner_unavailable",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "generator_quarantine": quarantine,
                },
            )
        metrics = dict(result.get("metrics", {}))
        current = self.get_eef_pose(hand)
        if current is not None:
            metrics["eef_target_distance_m"] = float(
                np.linalg.norm(target_xyz - current[0])
            )
        if result.get("ok"):
            return True, "reachable_candidate", metrics
        reason = str(result.get("stop_reason", "unreachable"))
        if (
            reason == "unreachable"
            and current is not None
            and metrics.get("eef_target_distance_m", 0.0) > 1.0
        ):
            reason = "navigation_required"
        return False, reason, metrics

    def check_candidate_arm_reachability(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        base_xyyaw: np.ndarray,
        timeout_s: float,
        skip_obstacle_update: bool = False,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Try a bounded, deterministic set of wrist poses.

        A station is not rejected merely because the robot's reset wrist
        orientation cannot reach the point.  Every accepted pose is still an
        official cuRobo IK result for the requested hand and candidate BASE
        state.
        """

        started = time.monotonic()
        current_eef = self.get_eef_pose(hand)
        if current_eef is None:
            return False, "planner_unavailable", {"error": "EEF pose unavailable"}
        robot = self._find_robot()
        current_base = self._base_xy_yaw(robot)
        delta_yaw = _wrap_angle(float(base_xyyaw[2]) - float(current_base[2]))
        natural = _quat_multiply_xyzw(_yaw_to_quat_xyzw(delta_yaw), current_eef[1])
        orientation_specs = [
            ("preserve_relative", None),
            ("local_x_plus_90", [1.0, 0.0, 0.0, math.pi / 2.0]),
            ("local_x_minus_90", [1.0, 0.0, 0.0, -math.pi / 2.0]),
            ("local_y_plus_90", [0.0, 1.0, 0.0, math.pi / 2.0]),
            ("local_y_minus_90", [0.0, 1.0, 0.0, -math.pi / 2.0]),
            ("local_z_plus_90", [0.0, 0.0, 1.0, math.pi / 2.0]),
            ("local_z_minus_90", [0.0, 0.0, 1.0, -math.pi / 2.0]),
            ("local_x_180", [1.0, 0.0, 0.0, math.pi]),
        ]
        attempts: list[dict[str, Any]] = []
        obstacle_update_pending = not bool(skip_obstacle_update)
        for name, relative_axis_angle in orientation_specs:
            remaining_s = float(timeout_s) - (time.monotonic() - started)
            if remaining_s <= 0.0:
                break
            quat = (
                natural
                if relative_axis_angle is None
                else _quat_multiply_xyzw(
                    natural, _axis_angle_to_quat_xyzw(relative_axis_angle)
                )
            )
            attempt_started = time.monotonic()
            reachable, reason, metrics = self.check_arm_reachability(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=quat,
                base_xyyaw=base_xyyaw,
                timeout_s=min(4.0, remaining_s),
                skip_obstacle_update=not obstacle_update_pending,
            )
            obstacle_update_pending = False
            attempts.append(
                {
                    "orientation": name,
                    "target_quat_xyzw": quat.tolist(),
                    "reachable": bool(reachable),
                    "reason": reason,
                    "elapsed_s": round(time.monotonic() - attempt_started, 3),
                    "metrics": metrics,
                }
            )
            if reachable:
                return (
                    True,
                    "reachable_candidate",
                    {
                        "reachability_stage": ("candidate_multi_orientation_ik"),
                        "selected_orientation": name,
                        "selected_target_quat_xyzw": quat.tolist(),
                        "attempts": attempts,
                    },
                )
            if reason in {"timeout", "planner_unavailable"}:
                break
        return (
            False,
            "unreachable",
            {
                "reachability_stage": "candidate_multi_orientation_ik",
                "attempts": attempts,
            },
        )

    def _all_attached_objects(
        self,
        *,
        selected_hand: str,
        selected_expected: Any = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Snapshot and merge both assisted-grasp collision bodies."""

        selected_hand = _normalize_hand(selected_hand)
        merged: dict[str, Any] = {}
        by_hand: dict[str, Any] = {}
        for side in ("left", "right"):
            live = self.get_attached_object(side)
            by_hand[side] = live
            if isinstance(live, dict):
                merged.update(live)
        roots = list(merged.values())
        root_paths = [str(getattr(root, "prim_path", "")).rstrip("/") for root in roots]
        if len(roots) != len(
            {path or f"object:{id(root)}" for path, root in zip(root_paths, roots)}
        ):
            raise RuntimeError(
                "the same collision body cannot be attached to both R1Pro EEFs"
            )
        if selected_expected is not None:
            matches, identity = _attachment_state_status(
                by_hand[selected_hand],
                selected_expected,
                hand=selected_hand,
            )
            if not matches:
                raise RuntimeError(
                    "selected attachment changed before whole-body planning: "
                    f"{identity!r}"
                )
        return (merged or None), by_hand

    @contextmanager
    def _whole_body_plan_policy(
        self,
        generator: Any,
        *,
        enable_graph: bool,
        position_only: bool,
        timeout_s: float,
    ):
        """Install and always clear one Behavior-owned MotionGen policy."""

        if hasattr(generator, "_rpent_plan_override"):
            raise RuntimeError("whole-body MotionGen policy is already active")
        generator._rpent_plan_override = {
            "enable_graph": bool(enable_graph),
            "position_only": bool(position_only),
            "timeout_s": float(timeout_s),
        }
        try:
            yield
        finally:
            try:
                delattr(generator, "_rpent_plan_override")
            except AttributeError:
                pass

    def plan_whole_body_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
        search_profile: str = WHOLE_BODY_SEARCH_PROFILE_DEFAULT,
        start_q: Any | None = None,
        start_eef_pose: Any | None = None,
        background: bool = False,
        wall_clock_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Plan and independently certify one selected-EEF 21-DOF trajectory."""

        hand = _normalize_hand(hand)
        profile = _whole_body_search_profile(search_profile)
        started = time.monotonic()
        hard_timeout_s = min(
            float(timeout_s),
            float(profile["planning_deadline_s"]),
        )
        wall_clock_timeout = (
            hard_timeout_s
            if wall_clock_timeout_s is None
            else float(wall_clock_timeout_s)
        )
        if (
            not math.isfinite(wall_clock_timeout)
            or wall_clock_timeout <= 0.0
            or wall_clock_timeout + 1e-12 < hard_timeout_s
        ):
            raise ValueError(
                "whole-body wall-clock timeout must be finite, positive, and "
                "at least the solver planning budget"
            )
        try:
            deadline_scope = (
                nullcontext()
                if background
                else _wall_clock_deadline(
                    wall_clock_timeout,
                    f"{hand} whole-body planning transaction",
                )
            )
            with deadline_scope:
                result = self._compute_whole_body_plan(
                    hand=hand,
                    target_xyz=target_xyz,
                    target_quat_xyzw=target_quat_xyzw,
                    timeout_s=hard_timeout_s,
                    selected_expected_attachment=attached_obj,
                    search_profile=str(profile["name"]),
                    start_q=start_q,
                    start_eef_pose=start_eef_pose,
                )
            metrics = (
                dict(result.get("metrics"))
                if isinstance(result.get("metrics"), dict)
                else {}
            )
            metrics["planning_elapsed_s"] = round(
                time.monotonic() - started, 3
            )
            metrics["deadline_enforcement"] = {
                "solver_timeout_enforced": True,
                "hard_wall_clock_enforced": not bool(background),
                "hard_wall_clock_deadline_s": (
                    wall_clock_timeout if not background else None
                ),
                "soft_deadline_s": (
                    wall_clock_timeout if background else None
                ),
                "solver_planning_budget_s": hard_timeout_s,
            }
            result["metrics"] = metrics
            return result
        except Exception as exc:
            quarantine = self._quarantine_generator(
                kind="whole_body",
                hand=hand,
                reason=f"{type(exc).__name__}: {exc}",
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "motion_scope": "whole_body",
                    "generator_kind": "whole_body",
                    "planning_elapsed_s": round(
                        time.monotonic() - started, 3
                    ),
                    "deadline_enforcement": {
                        "solver_timeout_enforced": True,
                        "hard_wall_clock_enforced": not bool(background),
                        "hard_wall_clock_deadline_s": (
                            wall_clock_timeout if not background else None
                        ),
                        "soft_deadline_s": (
                            wall_clock_timeout if background else None
                        ),
                        "solver_planning_budget_s": hard_timeout_s,
                    },
                    "generator_quarantine": quarantine,
                    "collision_admission": {
                        "available": False,
                        "admitted": False,
                        "reason": "whole-body trajectory unavailable",
                    },
                },
            }

    def get_torso_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the live torso_link4 world pose."""

        robot = self._find_robot()
        link = (getattr(robot, "links", {}) or {}).get(TORSO_LINK_NAME)
        if link is None:
            raise RuntimeError(f"R1Pro link {TORSO_LINK_NAME!r} is unavailable")
        position, quaternion = link.get_position_orientation()
        position_array = np.asarray(
            _jsonable(position), dtype=np.float64
        ).reshape(-1)
        quaternion_array = _quat_xyzw(quaternion)
        if (
            position_array.shape != (3,)
            or not np.isfinite(position_array).all()
            or quaternion_array is None
        ):
            raise RuntimeError("R1Pro torso_link4 pose feedback is invalid")
        return position_array, quaternion_array

    def _torso_joint_layouts(
        self,
        *,
        generator: Any | None = None,
        robot: Any | None = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return validated live and CuRobo torso joint layouts.

        Torso plans cross the simulator boundary in the live ``robot.joints``
        order required by ``q_to_action``.  CuRobo may expose the same joints in
        a different order, so every FK/planner boundary must reorder explicitly
        instead of changing the public trajectory layout.
        """

        generator = self._generator(kind="torso") if generator is None else generator
        robot = self._find_robot() if robot is None else robot
        live_names = tuple(
            str(name) for name in (getattr(robot, "joints", {}) or {})
        )
        generator_names = tuple(
            str(name) for name in getattr(generator, "robot_joint_names", ())
        )
        if (
            len(live_names) != 28
            or len(live_names) != len(set(live_names))
            or len(generator_names) != 28
            or len(generator_names) != len(set(generator_names))
            or set(live_names) != set(generator_names)
        ):
            raise RuntimeError("R1Pro torso full joint layout is invalid")
        return live_names, generator_names

    def _torso_path_to_full_joint_trajectory(
        self,
        generator: Any,
        robot: Any,
        path: Any,
        *,
        start_q: np.ndarray,
    ) -> np.ndarray:
        """Merge a torso-only CuRobo path while preserving all locked joints."""

        robot_names, full_names = self._torso_joint_layouts(
            generator=generator,
            robot=robot,
        )
        start = np.asarray(start_q, dtype=np.float32).reshape(-1)
        if start.shape != (len(robot_names),):
            raise RuntimeError("R1Pro torso path full joint layout is invalid")
        active_names = tuple(
            str(name)
            for name in generator.mg[
                self._embodiment_cls.DEFAULT
            ].kinematics.joint_names
        )
        if (
            len(active_names) != len(TORSO_ACTIVE_JOINT_NAMES)
            or set(active_names) != set(TORSO_ACTIVE_JOINT_NAMES)
        ):
            raise RuntimeError("R1Pro torso generator active joint set is invalid")
        path_names = tuple(
            str(name) for name in (getattr(path, "joint_names", None) or ())
        )
        values = np.asarray(
            _jsonable(getattr(path, "position", None)), dtype=np.float32
        )
        if (
            not path_names
            or len(path_names) != len(set(path_names))
            or values.size == 0
            or values.shape[-1] != len(path_names)
            or not np.isfinite(values).all()
        ):
            raise RuntimeError("R1Pro torso CuRobo path is invalid")
        if not set(TORSO_ACTIVE_JOINT_NAMES).issubset(path_names):
            raise RuntimeError("R1Pro torso path omitted an active joint")
        unknown = set(path_names) - set(full_names)
        if unknown:
            raise RuntimeError(
                f"R1Pro torso path contains unknown joints: {sorted(unknown)!r}"
            )
        values = values.reshape(-1, len(path_names))
        full_index = {name: index for index, name in enumerate(robot_names)}
        merged = np.broadcast_to(start, (len(values), len(start))).copy()
        for source_index, name in enumerate(path_names):
            if name in TORSO_ACTIVE_JOINT_NAMES:
                merged[:, full_index[name]] = values[:, source_index]
        locked_names = set(full_names) - set(TORSO_ACTIVE_JOINT_NAMES)
        locked_indices = [full_index[name] for name in locked_names]
        if not np.array_equal(
            merged[:, locked_indices],
            np.broadcast_to(
                start[locked_indices], (len(merged), len(locked_indices))
            ),
        ):
            raise RuntimeError("R1Pro torso path changed a locked joint")
        return merged

    def torso_jog_capability(self) -> dict[str, Any]:
        """Verify the read-only torso planner and controller contract."""

        try:
            robot = self._find_robot()
            pose = self.get_torso_pose()
            generator = self._generator(kind="torso")
            active_names = tuple(
                str(name)
                for name in generator.mg[
                    self._embodiment_cls.DEFAULT
                ].kinematics.joint_names
            )
            trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
            robot_joint_names = tuple(
                str(name) for name in (getattr(robot, "joints", {}) or {})
            )
            expected_trunk_indices = [
                robot_joint_names.index(f"torso_joint{index}")
                for index in range(1, 5)
            ]
            controllers = getattr(robot, "controllers", {}) or {}
            trunk_controller = controllers.get("trunk")
            trunk_controller_indices = _indices(
                getattr(trunk_controller, "dof_idx", [])
            )
            controller_layout = tuple(
                (str(name), int(getattr(controller, "command_dim", -1)))
                for name, controller in controllers.items()
            )
            checks = {
                "target_link_available": TORSO_LINK_NAME
                in (getattr(robot, "links", {}) or {}),
                "pose_available": (
                    np.asarray(pose[0]).shape == (3,)
                    and np.asarray(pose[1]).shape == (4,)
                ),
                "active_joint_set_exact": active_names
                == TORSO_ACTIVE_JOINT_NAMES,
                "trunk_indices_exact": (
                    trunk_indices == expected_trunk_indices
                ),
                "trunk_controller_indices_exact": (
                    trunk_controller_indices == expected_trunk_indices
                ),
                "trunk_controller_absolute_4d": bool(
                    trunk_controller is not None
                    and "JointController"
                    in {
                        cls.__name__
                        for cls in type(trunk_controller).__mro__
                    }
                    and int(getattr(trunk_controller, "command_dim", -1)) == 4
                    and not bool(
                        getattr(trunk_controller, "use_delta_commands", False)
                    )
                ),
                "controller_layout_23d_exact": controller_layout
                == (
                    ("base", 3),
                    ("trunk", 4),
                    ("arm_left", 7),
                    ("gripper_left", 1),
                    ("arm_right", 7),
                    ("gripper_right", 1),
                ),
                "q_to_action_callable": callable(
                    getattr(robot, "q_to_action", None)
                ),
                "plan_callable": callable(
                    getattr(self, "plan_torso_trajectory", None)
                ),
            }
            return {
                "available": bool(all(checks.values())),
                "verified": bool(all(checks.values())),
                "target_link": TORSO_LINK_NAME,
                "active_joint_names": list(active_names),
                "locked_joint_count": 25,
                "checks": {key: bool(value) for key, value in checks.items()},
                "env_actions_sent": 0,
                "simulator_advanced": False,
            }
        except Exception as exc:
            return {
                "available": False,
                "verified": False,
                "target_link": TORSO_LINK_NAME,
                "reason": f"{type(exc).__name__}: {exc}",
                "env_actions_sent": 0,
                "simulator_advanced": False,
            }

    def plan_torso_trajectory(
        self,
        *,
        target_z_m: float,
        timeout_s: float,
        start_q: Any | None = None,
        start_torso_pose: Any | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        """Plan and collision-certify one torso_link4 world-Z motion."""

        started = time.monotonic()
        target_z = float(target_z_m)
        if not math.isfinite(target_z):
            raise ValueError("torso target Z must be finite")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("torso planning timeout must be finite and positive")
        timeout = min(timeout, WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S)
        try:
            deadline_scope = (
                nullcontext()
                if background
                else _wall_clock_deadline(
                    timeout, "Dashboard torso planning transaction"
                )
            )
            with deadline_scope:
                generator = self._generator(kind="torso")
                robot = self._find_robot()
                torch = self._torch
                if torch is None:
                    import torch as torch  # type: ignore[no-redef]
                start_q_array = np.asarray(
                    _jsonable(
                        robot.get_joint_positions()
                        if start_q is None
                        else start_q
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                robot_names, generator_names = self._torso_joint_layouts(
                    generator=generator,
                    robot=robot,
                )
                if (
                    start_q_array.shape != (len(robot_names),)
                    or not np.isfinite(start_q_array).all()
                ):
                    raise RuntimeError("R1Pro torso start joint state is invalid")
                start_q_generator = _reorder_joint_trajectory(
                    start_q_array.reshape(1, -1),
                    source_names=robot_names,
                    target_names=generator_names,
                )[0]
                if start_torso_pose is None:
                    start_position, start_quaternion = self.get_torso_pose()
                else:
                    start_position = np.asarray(
                        start_torso_pose[0], dtype=np.float64
                    ).reshape(3)
                    start_quaternion = _quat_xyzw(start_torso_pose[1])
                    if start_quaternion is None:
                        raise RuntimeError("R1Pro predicted torso pose is invalid")
                target_position = np.asarray(
                    [start_position[0], start_position[1], target_z],
                    dtype=np.float64,
                )
                attachments, attachments_by_hand = self._all_attached_objects(
                    selected_hand="left",
                )
                attachment_scales = (
                    {str(link): 1.0 for link in attachments}
                    if isinstance(attachments, dict)
                    else None
                )
                batch_size = max(int(generator.batch_size), 1)
                target_positions = (
                    torch.as_tensor(target_position, dtype=torch.float32)
                    .reshape(1, 3)
                    .repeat(batch_size, 1)
                )
                target_quaternions = (
                    torch.as_tensor(start_quaternion, dtype=torch.float32)
                    .reshape(1, 4)
                    .repeat(batch_size, 1)
                )
                generator.update_obstacles()
                with self._whole_body_plan_policy(
                    generator,
                    enable_graph=False,
                    position_only=False,
                    timeout_s=min(
                        timeout,
                        WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S,
                    ),
                ):
                    successes, paths = generator.compute_trajectories(
                        target_positions,
                        target_quaternions,
                        initial_joint_pos=torch.as_tensor(
                            start_q_generator, dtype=torch.float32
                        ),
                        max_attempts=2,
                        timeout=min(
                            timeout,
                            WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S,
                        ),
                        ik_fail_return=2,
                        enable_finetune_trajopt=True,
                        finetune_attempts=2,
                        return_full_result=False,
                        success_ratio=1.0 / batch_size,
                        attached_obj=attachments,
                        attached_obj_scale=attachment_scales,
                        ik_only=False,
                        is_local=False,
                        skip_obstacle_update=True,
                        ik_world_collision_check=True,
                    )
                success_indices = np.flatnonzero(
                    np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
                )
                if not len(success_indices):
                    return {
                        "ok": False,
                        "stop_reason": "unreachable",
                        "metrics": {
                            "motion_scope": "torso_only",
                            "env_actions_sent": 0,
                        },
                    }
                selected: dict[str, Any] | None = None
                candidate_rejections: list[dict[str, Any]] = []
                for candidate_index in success_indices:
                    raw = self._torso_path_to_full_joint_trajectory(
                        generator,
                        robot,
                        paths[int(candidate_index)],
                        start_q=start_q_array,
                    )
                    with_start = (
                        raw.copy()
                        if np.allclose(
                            raw[0], start_q_array, atol=1e-7, rtol=0.0
                        )
                        else np.vstack([start_q_array.reshape(1, -1), raw])
                    )
                    dense = _interpolate_joint_trajectory(
                        with_start,
                        max_inter_dist=TORSO_JOINT_EXECUTION_STEP_RAD,
                    )
                    dense_generator = _reorder_joint_trajectory(
                        dense,
                        source_names=robot_names,
                        target_names=generator_names,
                    )
                    positions, quaternions = self._curobo_eef_poses(
                        generator, dense_generator
                    )
                    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
                    quaternions = np.asarray(
                        quaternions, dtype=np.float64
                    ).reshape(-1, 4)
                    if (
                        positions.shape != (len(dense), 3)
                        or quaternions.shape != (len(dense), 4)
                        or not np.isfinite(positions).all()
                        or not np.isfinite(quaternions).all()
                    ):
                        candidate_rejections.append(
                            {
                                "candidate_index": int(candidate_index),
                                "reason": "torso_fk_invalid",
                            }
                        )
                        continue
                    z_steps = np.diff(positions[:, 2])
                    orientation_errors = np.asarray(
                        [
                            _quat_angle_error_rad(quaternion, start_quaternion)
                            for quaternion in quaternions
                        ],
                        dtype=np.float64,
                    )
                    path_checks = {
                        "start_pose_matches": float(
                            np.linalg.norm(positions[0] - start_position)
                        )
                        <= WHOLE_BODY_EEF_START_FK_TOLERANCE_M,
                        "terminal_z_matches": abs(
                            float(positions[-1, 2] - target_z)
                        )
                        <= TORSO_LINK_TERMINAL_TOLERANCE_M,
                        "xy_locked": float(
                            np.max(
                                np.linalg.norm(
                                    positions[:, :2] - start_position[:2],
                                    axis=1,
                                )
                            )
                        )
                        <= TORSO_LINK_MAX_XY_DRIFT_M,
                        "z_step_bounded": (
                            not len(z_steps)
                            or float(np.max(np.abs(z_steps)))
                            <= TORSO_LINK_MAX_Z_STEP_M
                        ),
                        "z_direction_monotonic": (
                            _torso_z_path_direction_admitted(
                                positions[:, 2],
                                start_z_m=float(start_position[2]),
                                target_z_m=target_z,
                            )
                        ),
                        "orientation_locked": bool(
                            np.isfinite(orientation_errors).all()
                            and float(
                                np.max(orientation_errors, initial=0.0)
                            )
                            <= TORSO_LINK_MAX_ORIENTATION_DRIFT_RAD
                        ),
                    }
                    if not all(path_checks.values()):
                        candidate_rejections.append(
                            {
                                "candidate_index": int(candidate_index),
                                "reason": "torso_path_checks_failed",
                                "path_checks": {
                                    key: bool(value)
                                    for key, value in path_checks.items()
                                },
                            }
                        )
                        continue
                    collision_flags = generator.check_collisions(
                        torch.as_tensor(
                            dense_generator, dtype=torch.float32
                        ),
                        initial_joint_pos=torch.as_tensor(
                            start_q_generator, dtype=torch.float32
                        ),
                        self_collision_check=True,
                        skip_obstacle_update=True,
                        attached_obj=attachments,
                        attached_obj_scale=attachment_scales,
                    )
                    collision_array = np.asarray(
                        _jsonable(collision_flags), dtype=bool
                    ).reshape(-1)
                    if (
                        collision_array.shape != (len(dense),)
                        or bool(collision_array.any())
                    ):
                        candidate_rejections.append(
                            {
                                "candidate_index": int(candidate_index),
                                "reason": "torso_collision_check_failed",
                                "collision_shape": list(
                                    collision_array.shape
                                ),
                                "expected_shape": [len(dense)],
                                "colliding_waypoint_count": int(
                                    collision_array.sum()
                                ),
                            }
                        )
                        continue
                    if len(dense) > 1:
                        execution = np.ascontiguousarray(
                            dense[1:], dtype=np.float32
                        )
                        execution_positions = np.ascontiguousarray(
                            positions[1:], dtype=np.float32
                        )
                    else:
                        # A current-pose identity query is a mandatory reset-time
                        # planner warmup.  Preserve one certificate-bound no-op
                        # target without executing it.
                        execution = np.ascontiguousarray(
                            dense.copy(), dtype=np.float32
                        )
                        execution_positions = np.ascontiguousarray(
                            positions.copy(), dtype=np.float32
                        )
                    certificate = {
                        "schema_version": 1,
                        "kind": "torso_link4_world_z",
                        "trajectory_sha256": hashlib.sha256(
                            execution.tobytes()
                        ).hexdigest(),
                        "start_q_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                start_q_array, dtype=np.float32
                            ).tobytes()
                        ).hexdigest(),
                        "joint_name_layout_sha256": _joint_name_layout_sha256(
                            robot_names
                        ),
                        "q_dimension": int(execution.shape[1]),
                        "active_joint_names": list(TORSO_ACTIVE_JOINT_NAMES),
                        "locked_joint_count": 25,
                        "target_link": TORSO_LINK_NAME,
                        "target_z_m": target_z,
                        "start_position": start_position.tolist(),
                        "start_quaternion_xyzw": (
                            np.asarray(
                                start_quaternion, dtype=np.float64
                            ).tolist()
                        ),
                        "terminal_position": positions[-1].tolist(),
                        "terminal_quaternion_xyzw": quaternions[-1].tolist(),
                        "execution_torso_positions": (
                            execution_positions.astype(float).tolist()
                        ),
                        "execution_torso_positions_sha256": hashlib.sha256(
                            execution_positions.tobytes()
                        ).hexdigest(),
                        "execution_torso_quaternions_xyzw": (
                            (
                                quaternions[1:]
                                if len(dense) > 1
                                else quaternions
                            )
                            .astype(float)
                            .tolist()
                        ),
                        "execution_torso_quaternions_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                (
                                    quaternions[1:]
                                    if len(dense) > 1
                                    else quaternions
                                ),
                                dtype=np.float32,
                            ).tobytes()
                        ).hexdigest(),
                        "world_collision_check": True,
                        "self_collision_check": True,
                        "post_interpolation_check": True,
                        "collision_free_waypoints": int(len(dense)),
                        "path_checks": {
                            key: bool(value) for key, value in path_checks.items()
                        },
                        "attachment_hand_count": 2,
                    }
                    selected = {
                        "joint_trajectory": execution,
                        "certificate": certificate,
                        "terminal_pose": (
                            positions[-1].copy(),
                            quaternions[-1].copy(),
                        ),
                    }
                    break
                if selected is None:
                    return {
                        "ok": False,
                        "stop_reason": "collision_admission_failed",
                        "metrics": {
                            "motion_scope": "torso_only",
                            "env_actions_sent": 0,
                            "candidate_rejections": candidate_rejections,
                        },
                    }
                return {
                    "ok": True,
                    "joint_trajectory": selected["joint_trajectory"],
                    "torso_certificate": selected["certificate"],
                    "torso_goal": {
                        "target_z_m": target_z,
                        "target_xyz": selected["terminal_pose"][0].tolist(),
                    },
                    "expected_attachments_by_hand": attachments_by_hand,
                    "metrics": {
                        "motion_scope": "torso_only",
                        "generator_kind": "torso",
                        "active_dof_count": 3,
                        "active_joint_names": list(TORSO_ACTIVE_JOINT_NAMES),
                        "locked_joint_count": 25,
                        "target_link": TORSO_LINK_NAME,
                        "collision_admission": {
                            "available": True,
                            "admitted": True,
                            "world_collision_check": True,
                            "self_collision_check": True,
                            "obstacle_update": True,
                            "full_trajectory": True,
                            "post_interpolation_check": True,
                        },
                        "planning_elapsed_s": round(
                            time.monotonic() - started, 3
                        ),
                        "env_actions_sent": 0,
                    },
                }
        except Exception as exc:
            return {
                "ok": False,
                "stop_reason": (
                    "timeout" if isinstance(exc, TimeoutError) else "planner_unavailable"
                ),
                "metrics": {
                    "motion_scope": "torso_only",
                    "error": f"{type(exc).__name__}: {exc}",
                    "planning_elapsed_s": round(time.monotonic() - started, 3),
                    "env_actions_sent": 0,
                },
            }

    def _compute_whole_body_plan(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        selected_expected_attachment: Any = None,
        search_profile: str = WHOLE_BODY_SEARCH_PROFILE_DEFAULT,
        start_q: Any | None = None,
        start_eef_pose: Any | None = None,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        profile = _whole_body_search_profile(search_profile)
        inactive = "right" if hand == "left" else "left"
        generator = self._generator(kind="whole_body", hand=hand)
        robot = self._find_robot()
        embodiment = self._embodiment_cls.DEFAULT
        planner_joint_names = tuple(
            str(name) for name in generator.mg[embodiment].kinematics.joint_names
        )
        if len(planner_joint_names) != 21 or set(planner_joint_names) != set(
            WHOLE_BODY_ACTIVE_JOINT_NAMES
        ):
            raise RuntimeError(
                "whole-body generator active joints do not match the 21-DOF contract: "
                f"{planner_joint_names!r}"
            )
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]

        selected_pose = (
            start_eef_pose
            if start_eef_pose is not None
            else self.get_eef_pose(hand)
        )
        if selected_pose is None:
            raise RuntimeError(f"R1Pro {hand} EEF pose is unavailable")
        selected_target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        attachments, attachments_by_hand = self._all_attached_objects(
            selected_hand=hand,
            selected_expected=selected_expected_attachment,
        )
        selected_attachment_present = attachments_by_hand[hand] is not None
        if target_quat_xyzw is not None:
            selected_quat = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
            orientation_mode = "explicit_hard_target"
            position_only = False
        elif selected_attachment_present:
            selected_quat = np.asarray(selected_pose[1], dtype=np.float64).reshape(4)
            orientation_mode = "preserve_call_start_world_orientation"
            position_only = False
        else:
            selected_quat = np.asarray(selected_pose[1], dtype=np.float64).reshape(4)
            orientation_mode = "position_only_orientation_free"
            position_only = True
        if not np.isfinite(selected_target).all() or not np.isfinite(
            selected_quat
        ).all():
            raise RuntimeError("whole-body selected EEF target must be finite")

        selected_link = self._eef_link_name(robot, hand)
        batch_size = max(int(generator.batch_size), 1)
        target_positions = {
            selected_link: torch.as_tensor(selected_target, dtype=torch.float32)
            .reshape(1, 3)
            .repeat(batch_size, 1)
        }
        target_quaternions = {
            selected_link: torch.as_tensor(selected_quat, dtype=torch.float32)
            .reshape(1, 4)
            .repeat(batch_size, 1)
        }
        start_q_array = np.asarray(
            _jsonable(
                robot.get_joint_positions()
                if start_q is None
                else start_q
            ),
            dtype=np.float32,
        ).reshape(-1)
        initial_joint_pos = torch.as_tensor(start_q_array, dtype=torch.float32)
        robot_joint_names = list((getattr(robot, "joints", {}) or {}).keys())
        if len(robot_joint_names) != len(start_q_array):
            raise RuntimeError(
                "R1Pro live joint names do not match the whole-body start state"
            )
        start_by_name = {
            str(name): float(start_q_array[index])
            for index, name in enumerate(robot_joint_names)
        }
        active_retract = torch.as_tensor(
            [start_by_name[name] for name in planner_joint_names],
            dtype=torch.float32,
        )
        retract_tensors: dict[int, Any] = {}
        for rollout in generator.mg[embodiment].get_all_rollout_instances():
            dynamics = getattr(rollout, "dynamics_model", None)
            retract = getattr(dynamics, "retract_config", None)
            if retract is None:
                continue
            if tuple(retract.shape) != tuple(active_retract.shape):
                raise RuntimeError(
                    "whole-body rollout retract shape does not match 21 active joints"
                )
            retract.copy_(active_retract.to(device=retract.device, dtype=retract.dtype))
            retract_tensors[id(retract)] = retract
        kinematics_config = generator.mg[embodiment].kinematics.kinematics_config
        cspace_retract = getattr(
            getattr(kinematics_config, "cspace", None),
            "retract_config",
            None,
        )
        if cspace_retract is not None:
            if tuple(cspace_retract.shape) != tuple(active_retract.shape):
                raise RuntimeError(
                    "whole-body kinematics retract shape does not match 21 active joints"
                )
            cspace_retract.copy_(
                active_retract.to(
                    device=cspace_retract.device,
                    dtype=cspace_retract.dtype,
                )
            )
            retract_tensors[id(cspace_retract)] = cspace_retract
        if not retract_tensors:
            raise RuntimeError("whole-body live retract synchronization is unavailable")
        attachment_scales = (
            {str(link): 1.0 for link in attachments}
            if isinstance(attachments, dict)
            else None
        )
        config_path = self._whole_body_config_path(hand)
        metrics: dict[str, Any] = {
            "motion_scope": "whole_body",
            "generator_kind": "whole_body",
            "active_dof_count": 21,
            "active_joint_names": list(WHOLE_BODY_ACTIVE_JOINT_NAMES),
            "locked_joint_names": list(WHOLE_BODY_LOCKED_JOINT_NAMES),
            "curobo_config": str(config_path),
            "curobo_config_sha256": hashlib.sha256(
                config_path.read_bytes()
            ).hexdigest(),
            "eef_targets": {
                hand: {
                    "role": "requested_target",
                    "link": selected_link,
                    "xyz": selected_target.tolist(),
                    "quat_xyzw": selected_quat.tolist(),
                    "orientation_mode": orientation_mode,
                    "orientation_constrained": not position_only,
                },
                inactive: {
                    "role": "unconstrained",
                    "target_submitted": False,
                    "world_pose_gate": False,
                },
            },
            "selected_eef_goal_count": 1,
            "inactive_eef_goal_count": 0,
            "attachment_hand_count": 2,
            "attachments_by_hand": {
                side: {"available": attachments_by_hand[side] is not None}
                for side in ("left", "right")
            },
            "live_retract_synchronization": {
                "available": True,
                "active_dof_count": 21,
                "updated_tensor_count": len(retract_tensors),
                "source": "call_start_robot_joint_positions",
            },
            "collision_admission": {
                "available": False,
                "admitted": False,
                "world_collision_check": True,
                "self_collision_check": True,
                "obstacle_update": True,
                "full_trajectory": True,
                "post_interpolation_check": False,
                "source": "CuRoboMotionGenerator.compute_trajectories+check_collisions",
            },
            "planning_policy": {
                "search_profile": str(profile["name"]),
                "fast_trajopt_s": float(
                    profile["fast_trajopt_deadline_s"]
                ),
                "local_ik_s": (
                    float(profile["local_ik_deadline_s"])
                    if "local_ik_deadline_s" in profile
                    else None
                ),
                "local_ik_enabled": bool(
                    profile["name"] == WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
                    and not position_only
                ),
                "graph_fallback": True,
                "planning_hard_limit_s": float(
                    profile["planning_deadline_s"]
                ),
                "compute_max_attempts": int(profile["max_attempts"]),
                "ik_fail_return": int(profile["ik_fail_return"]),
                "finetune_attempts": int(profile["finetune_attempts"]),
                "candidate_certification": (
                    "first_fully_certified"
                    if bool(profile["first_safe_candidate"])
                    else "certify_all_solver_successes_then_rank"
                ),
                "single_arm_fallback": False,
                "probe_env_actions": 0,
            },
            "solver_stages": [],
            "candidate_audit": [],
        }

        full_joint_names = tuple(str(name) for name in generator.robot_joint_names)
        full_index = {name: index for index, name in enumerate(full_joint_names)}
        active_indices = [full_index[name] for name in WHOLE_BODY_ACTIVE_JOINT_NAMES]
        planning_deadline = time.monotonic() + min(
            float(timeout_s), float(profile["planning_deadline_s"])
        )
        world_refresh_started = time.monotonic()
        generator.update_obstacles()
        world_refresh_elapsed_s = max(
            0.0,
            time.monotonic() - world_refresh_started,
        )
        obstacle_refresh = RealCuroboBackend._obstacle_refresh_metrics(generator)
        if obstacle_refresh is None:
            raise RuntimeError(
                "whole-body obstacle refresh metrics are unavailable"
            )
        metrics["obstacle_refresh"] = obstacle_refresh
        metrics["world_refresh_count"] = 1
        metrics["world_refresh_s"] = world_refresh_elapsed_s
        safe_candidates: list[dict[str, Any]] = []
        any_solver_success = False
        any_eef_path_admitted = False
        collision_check_performed_candidate_count = 0
        eef_fk_cache: dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        eef_fk_cache_metrics = {
            "schema_version": 1,
            "scope": "single_candidate",
            "identity": "float32-c-order-sha256+array_equal",
            "queries": 0,
            "hits": 0,
            "backend_calls": 0,
            "backend_elapsed_s": 0.0,
        }
        metrics["selected_eef_fk_cache"] = eef_fk_cache_metrics

        def exact_eef_fk(
            q_trajectory: Any,
        ) -> tuple[np.ndarray, np.ndarray, bool]:
            """Reuse FK only for an exact canonical full-q path in this call."""

            q_canonical = np.ascontiguousarray(
                np.asarray(
                    _jsonable(q_trajectory),
                    dtype=np.float32,
                )
            )
            q_digest = hashlib.sha256(q_canonical.tobytes()).hexdigest()
            eef_fk_cache_metrics["queries"] += 1
            cached = eef_fk_cache.get(q_digest)
            if (
                cached is not None
                and cached[0].shape == q_canonical.shape
                and np.array_equal(cached[0], q_canonical)
            ):
                eef_fk_cache_metrics["hits"] += 1
                return cached[1].copy(), cached[2].copy(), True
            fk_started = time.monotonic()
            positions, quaternions = self._curobo_eef_poses(
                generator,
                q_canonical,
            )
            eef_fk_cache_metrics["backend_elapsed_s"] += max(
                0.0,
                time.monotonic() - fk_started,
            )
            eef_fk_cache_metrics["backend_calls"] += 1
            positions_array = np.asarray(positions).copy()
            quaternions_array = np.asarray(quaternions).copy()
            eef_fk_cache[q_digest] = (
                q_canonical.copy(),
                positions_array.copy(),
                quaternions_array.copy(),
            )
            return positions_array, quaternions_array, False

        def dense_eef_certificate(
            q_path_with_start: np.ndarray,
            *,
            initial_joint_step: float = WHOLE_BODY_DENSE_COLLISION_STEP,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            """Densify planning-only until all Cartesian step gates are decidable."""

            joint_step = min(
                float(initial_joint_step),
                WHOLE_BODY_DENSE_COLLISION_STEP,
            )
            if not math.isfinite(joint_step) or joint_step <= 0.0:
                raise ValueError(
                    "whole-body dense certification step must be positive"
                )
            attempts: list[dict[str, Any]] = []
            dense = np.asarray(q_path_with_start, dtype=np.float32)
            report: dict[str, Any] = {}
            for attempt in range(8):
                dense = _interpolate_joint_trajectory(
                    q_path_with_start,
                    max_inter_dist=joint_step,
                )
                positions, quaternions, _reused = exact_eef_fk(dense)
                report = self._whole_body_eef_path_report(
                    generator,
                    dense,
                    call_start_xyz=selected_pose[0],
                    target_xyz=selected_target,
                    target_quat_xyzw=(
                        None if position_only else selected_quat
                    ),
                    precomputed_eef_poses=(positions, quaternions),
                )
                report["checks"]["all_target_orientation_step"] = (
                    report.get("max_orientation_step_rad", float("inf"))
                    <= WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD + 1e-9
                )
                report["admitted"] = all(report["checks"].values())
                failed = sorted(
                    name
                    for name, passed in dict(
                        report.get("checks") or {}
                    ).items()
                    if not passed
                )
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "joint_step_rad": joint_step,
                        "dense_waypoints": int(len(dense)),
                        "failed_checks": failed,
                    }
                )
                if report.get("admitted") is True or not (
                    failed
                    and set(failed).issubset(
                        {
                            "all_target_cartesian_step",
                            "all_target_orientation_step",
                            "short_target_cartesian_step",
                            "short_target_rotation_step",
                        }
                    )
                ):
                    break
                joint_step *= 0.5
            report["adaptive_densification"] = {
                "planning_only": True,
                "attempts": attempts,
                "final_joint_step_rad": attempts[-1]["joint_step_rad"],
                "probe_env_actions": 0,
            }
            return dense, report

        solver_stages = []
        if (
            profile["name"] == WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
            and not position_only
        ):
            solver_stages.append(
                {
                    "name": "local_ik",
                    "enable_graph": False,
                    "ik_only": True,
                    "max_attempts": 1,
                    "ik_fail_return": 1,
                    "finetune_attempts": 0,
                }
            )
        solver_stages.extend(
            [
                {
                    "name": "fast_trajopt",
                    "enable_graph": False,
                    "ik_only": False,
                    "max_attempts": int(profile["max_attempts"]),
                    "ik_fail_return": int(profile["ik_fail_return"]),
                    "finetune_attempts": int(profile["finetune_attempts"]),
                },
                {
                    "name": "graph_trajopt",
                    "enable_graph": True,
                    "ik_only": False,
                    "max_attempts": int(profile["max_attempts"]),
                    "ik_fail_return": int(profile["ik_fail_return"]),
                    "finetune_attempts": int(profile["finetune_attempts"]),
                },
            ]
        )
        for stage in solver_stages:
            stage_name = str(stage["name"])
            enable_graph = bool(stage["enable_graph"])
            ik_only = bool(stage["ik_only"])
            remaining = planning_deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if ik_only:
                stage_budget = min(
                    float(profile["local_ik_deadline_s"]),
                    remaining,
                )
            elif not enable_graph:
                stage_budget = min(
                    float(profile["fast_trajopt_deadline_s"]),
                    remaining,
                )
            else:
                stage_budget = remaining
            stage_started = time.monotonic()
            stage_policy = (
                nullcontext()
                if ik_only
                else self._whole_body_plan_policy(
                    generator,
                    enable_graph=enable_graph,
                    position_only=position_only,
                    timeout_s=stage_budget,
                )
            )
            with stage_policy:
                successes, paths = generator.compute_trajectories(
                    target_positions,
                    target_quaternions,
                    initial_joint_pos=initial_joint_pos,
                    max_attempts=int(stage["max_attempts"]),
                    timeout=stage_budget,
                    ik_fail_return=int(stage["ik_fail_return"]),
                    enable_finetune_trajopt=not ik_only,
                    finetune_attempts=int(stage["finetune_attempts"]),
                    return_full_result=False,
                    success_ratio=1.0 / batch_size,
                    attached_obj=attachments,
                    attached_obj_scale=attachment_scales,
                    ik_only=ik_only,
                    is_local=False,
                    skip_obstacle_update=True,
                    ik_world_collision_check=True,
                )
            solver_finished = time.monotonic()
            success_array = np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
            success_indices = np.flatnonzero(success_array)
            any_solver_success = any_solver_success or bool(success_indices.size)
            stage_report = {
                "name": stage_name,
                "enable_graph": enable_graph,
                "ik_only": ik_only,
                "timeout_s": float(stage_budget),
                "elapsed_s": round(solver_finished - stage_started, 3),
                "successes": success_array.tolist(),
                "successful_candidate_indices": success_indices.tolist(),
                "certified_candidate_indices": [],
            }
            certificate_started = solver_finished
            for candidate_index in success_indices:
                eef_fk_cache.clear()
                audit: dict[str, Any] = {
                    "stage": stage_name,
                    "candidate_index": int(candidate_index),
                    "solver_success": True,
                    "certified": False,
                }
                candidate_started = time.monotonic()
                certificate_timing = {
                    "clock": "time.monotonic",
                    "source_dense_certificate_s": 0.0,
                    "subset_selection_s": 0.0,
                    "collision_recertification_s": 0.0,
                    "collision_check_s": 0.0,
                    "total_s": 0.0,
                }
                try:
                    candidate_q, path_merge = (
                        self._whole_body_path_to_full_joint_trajectory(
                            generator,
                            robot,
                            paths[int(candidate_index)],
                            start_q=start_q_array,
                        )
                    )
                    candidate_q, yaw_continuity = (
                        _canonicalize_whole_body_base_yaw_trajectory(
                            candidate_q,
                            joint_names=full_joint_names,
                            call_start_q=start_q_array,
                        )
                    )
                    path_merge["base_yaw_continuity"] = yaw_continuity
                    if (
                        candidate_q.ndim != 2
                        or candidate_q.shape[1] != start_q_array.shape[0]
                    ):
                        raise RuntimeError(
                            "whole-body candidate shape does not match R1Pro q"
                        )
                    if np.allclose(
                        candidate_q[0],
                        start_q_array,
                        atol=1e-7,
                        rtol=0.0,
                    ):
                        candidate_with_start = candidate_q.copy()
                        candidate_with_start[0] = start_q_array
                    else:
                        candidate_with_start = np.vstack(
                            [start_q_array.reshape(1, -1), candidate_q]
                        )
                    source_dense_started = time.monotonic()
                    dense_q, eef_path_report = dense_eef_certificate(
                        candidate_with_start
                    )
                    certificate_timing["source_dense_certificate_s"] += max(
                        0.0,
                        time.monotonic() - source_dense_started,
                    )
                    if (
                        len(dense_q) < 1
                        or not np.isfinite(dense_q).all()
                    ):
                        raise RuntimeError(
                            "whole-body candidate interpolation is empty or non-finite"
                        )
                    path_merge["path_variant"] = "solver_trajectory"
                    audit["solver_trajectory_eef_path"] = eef_path_report
                    if eef_path_report.get("admitted") is not True:
                        # CuRobo's endpoint-only whole-body objective may reach a
                        # local 3 cm goal through a large Cartesian loop.  The
                        # solver endpoint is still useful: construct one direct
                        # call-start -> endpoint 21-DOF candidate, then subject
                        # the *derived* trajectory to the same FK, yaw, dense
                        # collision, attachment, and certificate gates.  This
                        # is never a single-arm or unchecked execution fallback.
                        if (
                            eef_path_report.get("short_target") is True
                            and len(candidate_q) >= 1
                        ):
                            direct_candidate = np.vstack(
                                [
                                    start_q_array.reshape(1, -1),
                                    candidate_q[-1].reshape(1, -1),
                                ]
                            )
                            direct_candidate, direct_yaw = (
                                _canonicalize_whole_body_base_yaw_trajectory(
                                    direct_candidate,
                                    joint_names=full_joint_names,
                                    call_start_q=start_q_array,
                                )
                            )
                            candidate_with_start = direct_candidate
                            source_dense_started = time.monotonic()
                            dense_q, eef_path_report = dense_eef_certificate(
                                candidate_with_start
                            )
                            certificate_timing[
                                "source_dense_certificate_s"
                            ] += max(
                                0.0,
                                time.monotonic() - source_dense_started,
                            )
                            path_merge = {
                                **path_merge,
                                "path_variant": (
                                    "call_start_to_solver_endpoint"
                                ),
                                "base_yaw_continuity": direct_yaw,
                                "solver_trajectory_replaced": True,
                            }
                            audit["direct_endpoint_eef_path"] = eef_path_report
                    audit["selected_eef_path"] = eef_path_report
                    if eef_path_report.get("admitted") is not True:
                        audit["rejection_reason"] = "selected_eef_path_rejected"
                        metrics["candidate_audit"].append(audit)
                        continue
                    any_eef_path_admitted = True
                    execution_cap_refinements: list[dict[str, Any]] = []
                    for cap_refinement in range(8):
                        source_dense_q = np.asarray(dense_q, dtype=np.float32)
                        source_positions, source_quaternions, source_fk_reused = (
                            exact_eef_fk(source_dense_q)
                        )
                        source_positions = np.asarray(
                            source_positions, dtype=np.float32
                        ).reshape(-1, 3)
                        source_quaternions = np.asarray(
                            source_quaternions, dtype=np.float32
                        ).reshape(-1, 4)
                        if (
                            source_positions.shape != (len(source_dense_q), 3)
                            or source_quaternions.shape
                            != (len(source_dense_q), 4)
                            or not np.isfinite(source_positions).all()
                            or not np.isfinite(source_quaternions).all()
                            or hashlib.sha256(
                                np.ascontiguousarray(
                                    source_positions, dtype=np.float32
                                ).tobytes()
                            ).hexdigest()
                            != eef_path_report.get("positions_sha256")
                            or hashlib.sha256(
                                np.ascontiguousarray(
                                    source_quaternions, dtype=np.float32
                                ).tobytes()
                            ).hexdigest()
                            != eef_path_report.get("quaternions_sha256")
                        ):
                            raise _WholeBodyCertificationError(
                                "whole-body source dense EEF lineage is invalid"
                            )
                        source_step_report = (
                            _whole_body_execution_step_report(
                                source_dense_q,
                                source_positions,
                                source_quaternions,
                                joint_names=full_joint_names,
                                short_target=bool(
                                    eef_path_report["short_target"]
                                ),
                            )
                        )
                        try:
                            subset_started = time.monotonic()
                            proposed_indices = (
                                _whole_body_execution_subset_indices(
                                    source_dense_q,
                                    source_positions,
                                    source_quaternions,
                                    joint_names=full_joint_names,
                                    short_target=bool(
                                        eef_path_report["short_target"]
                                    ),
                                )
                            )
                            certificate_timing["subset_selection_s"] += max(
                                0.0,
                                time.monotonic() - subset_started,
                            )
                            break
                        except RuntimeError as exc:
                            if (
                                "no execution subset satisfying step limits"
                                not in str(exc)
                                or cap_refinement >= 7
                            ):
                                raise
                            prior_step = float(
                                eef_path_report[
                                    "adaptive_densification"
                                ]["final_joint_step_rad"]
                            )
                            refined_step = prior_step * 0.5
                            execution_cap_refinements.append(
                                {
                                    "attempt": cap_refinement + 1,
                                    "source_dense_waypoints": int(
                                        len(source_dense_q)
                                    ),
                                    "joint_step_rad": prior_step,
                                    "failed_checks": sorted(
                                        name
                                        for name, passed in dict(
                                            source_step_report.get(
                                                "checks"
                                            )
                                            or {}
                                        ).items()
                                        if not passed
                                    ),
                                    "refined_joint_step_rad": refined_step,
                                }
                            )
                            source_dense_started = time.monotonic()
                            dense_q, eef_path_report = dense_eef_certificate(
                                candidate_with_start,
                                initial_joint_step=refined_step,
                            )
                            certificate_timing[
                                "source_dense_certificate_s"
                            ] += max(
                                0.0,
                                time.monotonic() - source_dense_started,
                            )
                            audit["selected_eef_path"] = eef_path_report
                            if eef_path_report.get("admitted") is not True:
                                raise _WholeBodyCertificationError(
                                    "execution-cap source refinement rejected "
                                    "the selected EEF path"
                                )
                    else:
                        raise _WholeBodyCertificationError(
                            "whole-body execution-cap densification exhausted"
                        )
                    audit["execution_cap_densification"] = {
                        "planning_only": True,
                        "refinement_count": len(execution_cap_refinements),
                        "attempts": execution_cap_refinements,
                        "final_joint_step_rad": float(
                            eef_path_report["adaptive_densification"][
                                "final_joint_step_rad"
                            ]
                        ),
                        "exact_path_fk_recomputed": not source_fk_reused,
                        "exact_path_fk_reused": source_fk_reused,
                        "exact_path_fk_identity": (
                            "float32-c-order-sha256+array_equal"
                        ),
                        "collision_recertified_after_subset": True,
                        "probe_env_actions": 0,
                    }
                    full_indices = np.arange(len(source_dense_q), dtype=np.int64)
                    index_attempts = [proposed_indices]
                    if not np.array_equal(proposed_indices, full_indices):
                        index_attempts.append(full_indices)

                    certified_execution: dict[str, Any] | None = None
                    subset_attempts: list[dict[str, Any]] = []
                    colliding_count = 0
                    for source_indices in index_attempts:
                        execution_with_start = source_dense_q[source_indices]
                        execution_positions = source_positions[source_indices]
                        execution_quaternions = source_quaternions[source_indices]
                        execution_step_report = (
                            _whole_body_execution_step_report(
                                execution_with_start,
                                execution_positions,
                                execution_quaternions,
                                joint_names=full_joint_names,
                                short_target=bool(
                                    eef_path_report["short_target"]
                                ),
                            )
                        )
                        execution_path_report = _eef_pose_path_admission_report(
                            execution_positions,
                            execution_quaternions,
                            call_start_xyz=selected_pose[0],
                            target_xyz=selected_target,
                            target_quat_xyzw=(
                                None if position_only else selected_quat
                            ),
                            short_cartesian_step_limit_m=(
                                WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
                            ),
                        )
                        attempt_audit: dict[str, Any] = {
                            "source_dense_indices": source_indices.tolist(),
                            "execution_waypoints": int(len(source_indices) - 1),
                            "execution_step_admitted": (
                                execution_step_report.get("admitted") is True
                            ),
                            "execution_path_admitted": (
                                execution_path_report.get("admitted") is True
                            ),
                        }
                        if (
                            execution_step_report.get("admitted") is not True
                            or execution_path_report.get("admitted") is not True
                        ):
                            attempt_audit["rejection_reason"] = (
                                "execution_subset_admission_failed"
                            )
                            subset_attempts.append(attempt_audit)
                            continue

                        collision_recertification_started = time.monotonic()
                        collision_dense_q, collision_eef_path_report = (
                            dense_eef_certificate(execution_with_start)
                        )
                        (
                            collision_positions,
                            collision_quaternions,
                            collision_fk_reused,
                        ) = (
                            exact_eef_fk(collision_dense_q)
                        )
                        collision_positions = np.asarray(
                            collision_positions, dtype=np.float32
                        ).reshape(-1, 3)
                        collision_quaternions = np.asarray(
                            collision_quaternions, dtype=np.float32
                        ).reshape(-1, 4)
                        collision_step_report = (
                            _whole_body_execution_step_report(
                                collision_dense_q,
                                collision_positions,
                                collision_quaternions,
                                joint_names=full_joint_names,
                                short_target=bool(
                                    eef_path_report["short_target"]
                                ),
                            )
                        )
                        collision_fk_lineage_matches = (
                            hashlib.sha256(
                                np.ascontiguousarray(
                                    collision_positions, dtype=np.float32
                                ).tobytes()
                            ).hexdigest()
                            == collision_eef_path_report.get(
                                "positions_sha256"
                            )
                            and hashlib.sha256(
                                np.ascontiguousarray(
                                    collision_quaternions, dtype=np.float32
                                ).tobytes()
                            ).hexdigest()
                            == collision_eef_path_report.get(
                                "quaternions_sha256"
                            )
                        )
                        attempt_audit.update(
                            {
                                "collision_dense_waypoints": int(
                                    len(collision_dense_q)
                                ),
                                "collision_eef_path_admitted": (
                                    collision_eef_path_report.get("admitted")
                                    is True
                                ),
                                "collision_step_admitted": (
                                    collision_step_report.get("admitted") is True
                                ),
                                "collision_fk_lineage_matches": (
                                    collision_fk_lineage_matches
                                ),
                                "collision_fk_reused": collision_fk_reused,
                            }
                        )
                        if (
                            collision_eef_path_report.get("admitted") is not True
                            or collision_step_report.get("admitted") is not True
                            or not collision_fk_lineage_matches
                        ):
                            attempt_audit["rejection_reason"] = (
                                "shortcut_dense_admission_failed"
                            )
                            subset_attempts.append(attempt_audit)
                            continue
                        collision_execution_indices = (
                            _whole_body_ordered_row_indices(
                                collision_dense_q, execution_with_start
                            )
                        )
                        certificate_timing[
                            "collision_recertification_s"
                        ] += max(
                            0.0,
                            time.monotonic()
                            - collision_recertification_started,
                        )
                        try:
                            collision_check_started = time.monotonic()
                            collision_flags = generator.check_collisions(
                                torch.as_tensor(
                                    collision_dense_q, dtype=torch.float32
                                ),
                                initial_joint_pos=torch.as_tensor(
                                    start_q_array, dtype=torch.float32
                                ),
                                self_collision_check=True,
                                skip_obstacle_update=True,
                                attached_obj=attachments,
                                attached_obj_scale=attachment_scales,
                            )
                            certificate_timing["collision_check_s"] += max(
                                0.0,
                                time.monotonic() - collision_check_started,
                            )
                            collision_check_performed_candidate_count += 1
                        except TimeoutError:
                            raise
                        except Exception as exc:
                            raise _WholeBodyCertificationError(
                                "whole-body collision certification backend failed"
                            ) from exc
                        collision_array = np.asarray(
                            _jsonable(collision_flags), dtype=bool
                        ).reshape(-1)
                        if collision_array.shape != (len(collision_dense_q),):
                            raise _WholeBodyCertificationError(
                                "whole-body collision checker returned an invalid "
                                "waypoint count"
                            )
                        colliding_count = int(
                            np.count_nonzero(collision_array)
                        )
                        attempt_audit["colliding_waypoint_count"] = (
                            colliding_count
                        )
                        if colliding_count:
                            attempt_audit["rejection_reason"] = "collision"
                            subset_attempts.append(attempt_audit)
                            continue
                        attempt_audit["certified"] = True
                        subset_attempts.append(attempt_audit)
                        certified_execution = {
                            "joint_trajectory": execution_with_start[1:],
                            "source_dense_trajectory": source_dense_q,
                            "execution_source_dense_indices": source_indices,
                            "dense_collision_trajectory": collision_dense_q,
                            "execution_collision_dense_indices": (
                                collision_execution_indices
                            ),
                            "execution_step_report": execution_step_report,
                            "execution_eef_path": execution_path_report,
                            "dense_collision_eef_path": (
                                collision_eef_path_report
                            ),
                            "dense_collision_positions": collision_positions,
                            "dense_collision_quaternions": collision_quaternions,
                            "collision_densification_joint_step_rad": (
                                collision_eef_path_report[
                                    "adaptive_densification"
                                ]["final_joint_step_rad"]
                            ),
                        }
                        break
                    audit["execution_subset_attempts"] = subset_attempts
                    if certified_execution is None:
                        audit["colliding_waypoint_count"] = sum(
                            int(
                                attempt.get(
                                    "colliding_waypoint_count", 0
                                )
                            )
                            for attempt in subset_attempts
                        )
                        audit["rejection_reason"] = (
                            "execution_subset_collision_admission_failed"
                        )
                        metrics["candidate_audit"].append(audit)
                        continue
                    execution_q = np.asarray(
                        certified_execution["joint_trajectory"],
                        dtype=np.float32,
                    )
                    dense_q = np.asarray(
                        certified_execution["dense_collision_trajectory"],
                        dtype=np.float32,
                    )
                    path_length = float(
                        np.sum(
                            np.linalg.norm(
                                np.diff(
                                    candidate_with_start[:, active_indices],
                                    axis=0,
                                ),
                                axis=1,
                            )
                        )
                    )
                    audit.update(
                        {
                            "path_joint_merge": path_merge,
                            "source_dense_waypoints": int(len(source_dense_q)),
                            "dense_collision_waypoints": int(len(dense_q)),
                            "execution_waypoints": int(len(execution_q)),
                            "full_21d_path_length": path_length,
                            "colliding_waypoint_count": colliding_count,
                        }
                    )
                    audit["certified"] = True
                    stage_report["certified_candidate_indices"].append(
                        int(candidate_index)
                    )
                    safe_candidates.append(
                        {
                            "stage": stage_name,
                            "candidate_index": int(candidate_index),
                            **certified_execution,
                            "path_joint_merge": path_merge,
                            "full_21d_path_length": path_length,
                            "selected_eef_path": (
                                certified_execution[
                                    "dense_collision_eef_path"
                                ]
                            ),
                        }
                    )
                except TimeoutError:
                    raise
                except _WholeBodyCertificationError:
                    raise
                except Exception as exc:
                    audit["rejection_reason"] = f"{type(exc).__name__}: {exc}"
                finally:
                    certificate_timing["total_s"] = max(
                        0.0,
                        time.monotonic() - candidate_started,
                    )
                    audit["certificate_timing_s"] = certificate_timing
                metrics["candidate_audit"].append(audit)
                if (
                    bool(profile["first_safe_candidate"])
                    and audit.get("certified") is True
                ):
                    stage_report["certification_short_circuit"] = {
                        "enabled": True,
                        "candidate_index": int(candidate_index),
                        "remaining_successes_not_certified": int(
                            np.count_nonzero(success_indices > candidate_index)
                        ),
                    }
                    break
            stage_finished = time.monotonic()
            stage_report["timing_s"] = {
                "clock": "time.monotonic",
                "solver_s": max(0.0, solver_finished - stage_started),
                "certificate_s": max(
                    0.0,
                    stage_finished - certificate_started,
                ),
                "total_s": max(0.0, stage_finished - stage_started),
            }
            metrics["solver_stages"].append(stage_report)
            if safe_candidates:
                break

        if not safe_candidates:
            metrics["collision_admission"]["post_interpolation_check"] = bool(
                collision_check_performed_candidate_count
            )
            metrics["collision_admission"][
                "collision_check_performed_candidate_count"
            ] = collision_check_performed_candidate_count
            metrics["collision_admission"]["colliding_waypoint_count"] = (
                sum(
                    int(candidate.get("colliding_waypoint_count", 0))
                    for candidate in metrics["candidate_audit"]
                )
                if collision_check_performed_candidate_count
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    (
                        "eef_path_admission_failed"
                        if not any_eef_path_admitted
                        else "collision_admission_failed"
                    )
                    if any_solver_success
                    else "unreachable"
                ),
                "metrics": metrics,
            }

        safe_candidates.sort(
            key=lambda item: (
                len(item["joint_trajectory"]),
                item["full_21d_path_length"],
                item["candidate_index"],
            )
        )
        selected = safe_candidates[0]
        q_traj = np.asarray(selected["joint_trajectory"], dtype=np.float32)
        source_dense_q = np.asarray(
            selected["source_dense_trajectory"], dtype=np.float32
        )
        source_dense_indices = np.asarray(
            selected["execution_source_dense_indices"], dtype=np.int64
        )
        dense_q = np.asarray(
            selected["dense_collision_trajectory"], dtype=np.float32
        )
        collision_dense_indices = np.asarray(
            selected["execution_collision_dense_indices"], dtype=np.int64
        )
        dense_eef_positions = np.asarray(
            selected["dense_collision_positions"], dtype=np.float32
        ).reshape(-1, 3)
        dense_eef_quaternions = np.asarray(
            selected["dense_collision_quaternions"], dtype=np.float32
        ).reshape(-1, 4)
        execution_eef_positions = dense_eef_positions[
            collision_dense_indices[1:]
        ].copy()
        execution_eef_quaternions = dense_eef_quaternions[
            collision_dense_indices[1:]
        ].copy()
        if (
            execution_eef_positions.shape != (len(q_traj), 3)
            or execution_eef_quaternions.shape != (len(q_traj), 4)
            or source_dense_indices.shape != (len(q_traj) + 1,)
            or collision_dense_indices.shape != (len(q_traj) + 1,)
            or dense_eef_positions.shape != (len(dense_q), 3)
            or dense_eef_quaternions.shape != (len(dense_q), 4)
            or not np.isfinite(execution_eef_positions).all()
            or not np.isfinite(execution_eef_quaternions).all()
            or not np.isfinite(dense_eef_positions).all()
            or not np.isfinite(dense_eef_quaternions).all()
            or not np.array_equal(
                source_dense_q[source_dense_indices[1:]], q_traj
            )
            or not np.array_equal(
                dense_q[collision_dense_indices[1:]], q_traj
            )
            or not np.array_equal(
                dense_eef_positions[collision_dense_indices[1:]],
                execution_eef_positions,
            )
            or not np.array_equal(
                dense_eef_quaternions[collision_dense_indices[1:]],
                execution_eef_quaternions,
            )
        ):
            raise _WholeBodyCertificationError(
                "whole-body execution EEF waypoint lineage is invalid"
            )
        metrics["path_joint_merge"] = selected["path_joint_merge"]
        metrics["selected_eef_path"] = selected["selected_eef_path"]

        post_attachments, post_by_hand = self._all_attached_objects(
            selected_hand=hand,
            selected_expected=selected_expected_attachment,
        )
        del post_attachments
        for side in ("left", "right"):
            matches, identity = _attachment_state_status(
                post_by_hand[side], attachments_by_hand[side], hand=side
            )
            if not matches:
                metrics["attachment_identity_changed"] = {
                    "hand": side,
                    **identity,
                }
                return {
                    "ok": False,
                    "stop_reason": "attachment_identity_mismatch",
                    "metrics": metrics,
                }

        trajectory_bytes = np.ascontiguousarray(q_traj, dtype=np.float32).tobytes()
        source_dense_bytes = np.ascontiguousarray(
            source_dense_q, dtype=np.float32
        ).tobytes()
        dense_bytes = np.ascontiguousarray(dense_q, dtype=np.float32).tobytes()
        certificate = {
            "schema_version": 3,
            "collision_lineage_mode": (
                "source_dense_execution_subset_recertified_v1"
            ),
            "q_encoding": "float32-c-order",
            "trajectory_sha256": hashlib.sha256(trajectory_bytes).hexdigest(),
            "execution_trajectory_sha256": hashlib.sha256(
                trajectory_bytes
            ).hexdigest(),
            "start_q_sha256": hashlib.sha256(
                np.ascontiguousarray(start_q_array, dtype=np.float32).tobytes()
            ).hexdigest(),
            "waypoint_count": int(len(q_traj)),
            "execution_waypoint_count": int(len(q_traj)),
            "execution_includes_start": False,
            "q_dimension": int(q_traj.shape[1]),
            "joint_name_layout_sha256": _joint_name_layout_sha256(
                full_joint_names
            ),
            "active_dof_count": 21,
            "selected_eef_goal_count": 1,
            "inactive_eef_goal_count": 0,
            "attachment_hand_count": 2,
            "world_collision_check": True,
            "self_collision_check": True,
            "post_interpolation_check": True,
            "collision_free_waypoints": int(len(dense_q)),
            "dense_collision_waypoint_count": int(len(dense_q)),
            "dense_collision_checked_waypoint_count": int(len(dense_q)),
            "dense_collision_includes_start": True,
            "dense_collision_trajectory_sha256": hashlib.sha256(
                dense_bytes
            ).hexdigest(),
            "dense_collision_trajectory": dense_q.tolist(),
            "collision_densification_method": "joint_linear_max_inter_dist",
            "collision_densification_joint_step_rad": float(
                selected["collision_densification_joint_step_rad"]
            ),
            "source_dense_waypoint_count": int(len(source_dense_q)),
            "source_dense_includes_start": True,
            "source_dense_trajectory_sha256": hashlib.sha256(
                source_dense_bytes
            ).hexdigest(),
            "source_dense_trajectory": source_dense_q.tolist(),
            "execution_source_dense_indices": source_dense_indices.tolist(),
            "execution_source_dense_indices_sha256": (
                _whole_body_indices_sha256(source_dense_indices)
            ),
            "execution_source_terminal_index": int(source_dense_indices[-1]),
            "execution_collision_dense_indices": (
                collision_dense_indices.tolist()
            ),
            "execution_collision_dense_indices_sha256": (
                _whole_body_indices_sha256(collision_dense_indices)
            ),
            "execution_collision_terminal_index": int(
                collision_dense_indices[-1]
            ),
            "selected_hand": hand,
            "selected_target_xyz_sha256": _whole_body_target_sha256(
                selected_target
            ),
            "selected_target_quat_xyzw_sha256": (
                None
                if position_only
                else hashlib.sha256(
                    np.ascontiguousarray(
                        selected_quat, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
            ),
            "selected_eef_path_admitted": True,
            "selected_eef_short_target": selected["selected_eef_path"][
                "short_target"
            ],
            "selected_eef_positions_sha256": selected["selected_eef_path"][
                "positions_sha256"
            ],
            "selected_eef_dense_positions_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    dense_eef_positions, dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "selected_eef_dense_quaternions_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    dense_eef_quaternions, dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "selected_eef_dense_positions": dense_eef_positions.tolist(),
            "selected_eef_dense_quaternions_xyzw": (
                dense_eef_quaternions.tolist()
            ),
            "selected_eef_nominal_max_target_error_m": selected[
                "selected_eef_path"
            ]["nominal_max_target_error_m"],
            "selected_eef_execution_positions_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    execution_eef_positions, dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "selected_eef_execution_quaternions_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    execution_eef_quaternions, dtype=np.float32
                ).tobytes()
            ).hexdigest(),
            "selected_eef_execution_positions": execution_eef_positions.tolist(),
            "selected_eef_execution_quaternions_xyzw": (
                execution_eef_quaternions.tolist()
            ),
            "selected_eef_live_waypoint_position_tolerance_m": (
                WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M
            ),
            "selected_eef_live_waypoint_orientation_tolerance_rad": (
                WHOLE_BODY_EEF_LIVE_WAYPOINT_ORIENTATION_TOLERANCE_RAD
            ),
            "selected_eef_controller_response_margin_m": (
                WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M
            ),
            "selected_eef_numerical_margin_m": (
                WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
            ),
            "selected_eef_prospective_guard_margin_m": (
                WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
            ),
            "execution_base_xy_step_limit_m": (
                WHOLE_BODY_EXECUTION_BASE_XY_STEP_M
            ),
            "execution_base_yaw_step_limit_rad": (
                WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD
            ),
            "execution_articulation_step_limit_rad": (
                WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD
            ),
            "terminal_command_limit": TERMINAL_COMMAND_LIMIT,
            "terminal_eef_position_tolerance_m": (
                EEF_TERMINAL_POSITION_TOLERANCE_M
            ),
            "terminal_eef_orientation_tolerance_rad": (
                EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD
            ),
        }
        metrics["selected_solver_stage"] = selected["stage"]
        metrics["selected_full_trajectory_candidate"] = selected["candidate_index"]
        metrics["full_trajectory_selection"] = (
            "first_fully_certified_candidate"
            if bool(profile["first_safe_candidate"])
            else "fewest_execution_waypoints_then_shortest_full_21d_path"
        )
        metrics["selected_full_21d_path_length"] = selected[
            "full_21d_path_length"
        ]
        metrics["trajectory_waypoints"] = int(len(q_traj))
        metrics["source_dense_waypoints"] = int(len(source_dense_q))
        metrics["execution_source_dense_indices"] = (
            source_dense_indices.tolist()
        )
        metrics["execution_collision_dense_indices"] = (
            collision_dense_indices.tolist()
        )
        metrics["execution_step_admission"] = selected[
            "execution_step_report"
        ]
        metrics["execution_eef_path"] = selected["execution_eef_path"]
        metrics["execution_mode"] = "online_robot_q_to_action"
        metrics["collision_admission"].update(
            {
                "available": True,
                "admitted": True,
                "post_interpolation_check": True,
                "colliding_waypoint_count": 0,
                "collision_free_waypoints": int(len(dense_q)),
            }
        )
        metrics["whole_body_certificate"] = certificate
        return {
            "ok": True,
            "joint_trajectory": q_traj,
            "metrics": metrics,
            "whole_body_certificate": certificate,
            "expected_attachments_by_hand": attachments_by_hand,
        }

    def _whole_body_path_to_full_joint_trajectory(
        self,
        generator: Any,
        robot: Any,
        path: Any,
        *,
        start_q: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Merge active or already-augmented cuRobo paths into full R1Pro q.

        OmniGibson's ``path_to_joint_trajectory(get_full_js=True)`` delegates
        to cuRobo ``get_full_js``.  Some supported cuRobo builds already
        return locked joints in a trajectory; asking them to append locks a
        second time raises ``lock_joints is also listed in self.joint_names``.
        Name-based merging accepts either backend representation while never
        allowing a planned value to change the seven locked R1Pro joints.
        """

        full_names = tuple(str(name) for name in generator.robot_joint_names)
        robot_names = tuple(str(name) for name in (getattr(robot, "joints", {}) or {}))
        start = np.asarray(start_q, dtype=np.float32).reshape(-1)
        if (
            len(full_names) != len(set(full_names))
            or full_names != robot_names
            or start.size != len(full_names)
        ):
            raise RuntimeError(
                "R1Pro whole-body path merge requires the exact full joint layout"
            )

        path_names = tuple(
            str(name) for name in (getattr(path, "joint_names", None) or ())
        )
        if not path_names or len(path_names) != len(set(path_names)):
            raise RuntimeError(
                "whole-body cuRobo path has missing or duplicate joint names"
            )
        unknown = sorted(set(path_names) - set(full_names))
        if unknown:
            raise RuntimeError(
                f"whole-body cuRobo path contains unknown joints: {unknown}"
            )

        values = np.asarray(
            _jsonable(getattr(path, "position", None)), dtype=np.float32
        )
        if values.size == 0 or values.shape[-1] != len(path_names):
            raise RuntimeError(
                "whole-body cuRobo path width does not match its joint names: "
                f"shape={values.shape}, names={len(path_names)}"
            )
        values = values.reshape(-1, len(path_names))
        if not np.isfinite(values).all():
            raise RuntimeError("whole-body cuRobo path contains non-finite values")

        active_names = tuple(
            str(name)
            for name in generator.mg[
                self._embodiment_cls.DEFAULT
            ].kinematics.joint_names
        )
        if (
            len(active_names) != 21
            or set(active_names) != set(WHOLE_BODY_ACTIVE_JOINT_NAMES)
        ):
            raise RuntimeError(
                "whole-body cuRobo path active names violate the 21-DOF contract"
            )
        missing_active = sorted(set(active_names) - set(path_names))
        if missing_active:
            raise RuntimeError(
                f"whole-body cuRobo path is missing active joints: {missing_active}"
            )

        full_index = {name: index for index, name in enumerate(full_names)}
        merged = np.broadcast_to(start, (len(values), len(start))).copy()
        locked_entries_ignored: list[str] = []
        active_entries_written: list[str] = []
        for source_index, name in enumerate(path_names):
            if name in WHOLE_BODY_LOCKED_JOINT_NAMES:
                locked_entries_ignored.append(name)
                continue
            if name not in WHOLE_BODY_ACTIVE_JOINT_NAMES:
                raise RuntimeError(
                    f"whole-body cuRobo path contains unclassified joint {name!r}"
                )
            merged[:, full_index[name]] = values[:, source_index]
            active_entries_written.append(name)

        if set(active_entries_written) != set(WHOLE_BODY_ACTIVE_JOINT_NAMES):
            raise RuntimeError(
                "whole-body cuRobo path did not write all 21 active joints"
            )
        locked_indices = [
            full_index[name] for name in WHOLE_BODY_LOCKED_JOINT_NAMES
        ]
        expected_locked = np.broadcast_to(
            start[locked_indices], (len(merged), len(locked_indices))
        )
        if not np.array_equal(merged[:, locked_indices], expected_locked):
            raise RuntimeError("whole-body path merge changed a locked joint")
        return merged, {
            "method": "joint_name_merge_preserving_call_start_locked_joints",
            "source_joint_count": len(path_names),
            "waypoint_count": len(values),
            "active_joint_count": len(active_entries_written),
            "locked_joint_count": len(WHOLE_BODY_LOCKED_JOINT_NAMES),
            "locked_source_entries_ignored": len(locked_entries_ignored),
            "source_representation": (
                "active_only"
                if not locked_entries_ignored
                else "already_augmented_full"
            ),
        }

    def plan_arm_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        """Solve an arm IK target and interpolate it without safety admission."""

        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=True,
                attached_obj=attached_obj,
                return_ik_solution=True,
            )
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="arm",
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ik_only": False,
                    "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
                    "generator_quarantine": quarantine,
                },
            }

    def plan_arm_with_trunk_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        """Plan one collision-admitted selected-arm plus trunk1-3 trajectory."""

        try:
            result = self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=False,
                attached_obj=attached_obj,
                generator_kind="arm_with_trunk",
            )
            if not result.get("ok"):
                return result
            return self._certify_arm_with_trunk_trajectory(result, hand=hand)
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="arm_with_trunk",
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "motion_scope": "arm_with_trunk",
                    "generator_kind": "arm_with_trunk",
                    "generator_quarantine": quarantine,
                    "collision_admission": {
                        "available": False,
                        "admitted": False,
                        "reason": "collision-admitted trajectory unavailable",
                    },
                },
            }

    def _certify_arm_with_trunk_trajectory(
        self,
        result: dict[str, Any],
        *,
        hand: str,
    ) -> dict[str, Any]:
        """Fail closed unless collision admission and bounded trunk motion exist."""

        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("arm-with-trunk planner metrics are unavailable")
        collision = metrics.get("collision_admission")
        if (
            not isinstance(collision, dict)
            or collision.get("available") is not True
            or collision.get("admitted") is not True
            or collision.get("world_collision_check") is not True
            or collision.get("self_collision_check") is not True
            or collision.get("obstacle_update") is not True
            or collision.get("full_trajectory") is not True
        ):
            raise RuntimeError(
                "arm-with-trunk collision admission certificate is unavailable"
            )
        trajectory = np.asarray(
            _jsonable(result.get("joint_trajectory")), dtype=np.float64
        )
        robot = self._find_robot()
        current = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
        if (
            trajectory.ndim != 2
            or trajectory.shape[0] < 1
            or trajectory.shape[1] != current.size
            or len(trunk_indices) != 4
            or max(trunk_indices, default=-1) >= current.size
            or not np.isfinite(trajectory).all()
            or not np.isfinite(current).all()
        ):
            raise RuntimeError("arm-with-trunk joint trajectory is invalid")
        active_trunk = trunk_indices[:3]
        locked_trunk = trunk_indices[3]
        with_start = np.vstack([current.reshape(1, -1), trajectory])
        adjacent = np.abs(np.diff(with_start[:, active_trunk], axis=0))
        total = np.max(
            np.abs(trajectory[:, active_trunk] - current[active_trunk]), axis=0
        )
        locked_delta = float(
            np.max(np.abs(trajectory[:, locked_trunk] - current[locked_trunk]))
        )
        max_adjacent = float(np.max(adjacent))
        if max_adjacent > TRUNK_ASSIST_MAX_STEP_RAD + 1e-9:
            raise RuntimeError("arm-with-trunk trajectory exceeds per-step trunk limit")
        if np.any(total > TRUNK_ASSIST_MAX_TOTAL_RAD + 1e-9):
            raise RuntimeError("arm-with-trunk trajectory exceeds total trunk limit")
        if locked_delta > 1e-6 + 1e-12:
            raise RuntimeError("arm-with-trunk trajectory changed locked trunk4")
        certified = dict(result)
        certified_metrics = dict(metrics)
        certified_metrics.update(
            {
                "motion_scope": "arm_with_trunk",
                "active_controller_segments": [
                    "trunk[0:3]",
                    f"{_normalize_hand(hand)}_arm",
                ],
                "trunk_motion_limits": {
                    "available": True,
                    "ok": True,
                    "active_trunk_joint_count": 3,
                    "locked_trunk_joint_count": 1,
                    "max_adjacent_delta_rad": max_adjacent,
                    "max_adjacent_delta_limit_rad": TRUNK_ASSIST_MAX_STEP_RAD,
                    "max_total_delta_rad_by_joint": total.tolist(),
                    "max_total_delta_limit_rad": TRUNK_ASSIST_MAX_TOTAL_RAD,
                    "locked_trunk4_delta_rad": locked_delta,
                },
            }
        )
        certified["metrics"] = certified_metrics
        return certified

    def plan_attached_arm_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray,
        timeout_s: float,
        attached_obj: Any,
    ) -> dict[str, Any]:
        """Solve held-arm IK with base, trunk and press arm locked."""

        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=True,
                attached_obj=attached_obj,
                return_ik_solution=True,
                generator_kind="attached_arm",
            )
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="attached_arm",
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "generator_kind": "attached_arm",
                    "generator_quarantine": quarantine,
                },
            }

    def plan_guarded_ik_step(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
        contact_target_xyz: np.ndarray | None = None,
        lock_trunk: bool = False,
        full_solution: bool = False,
    ) -> dict[str, Any]:
        """Solve one bounded Cartesian contact step without safety admission."""

        generator_kind = "attached_arm" if lock_trunk else "arm"
        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=True,
                attached_obj=attached_obj,
                return_ik_solution=True,
                guarded_contact_target_xyz=contact_target_xyz,
                guarded_full_solution=bool(full_solution),
                generator_kind=generator_kind,
            )
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind=generator_kind,
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ik_only": True,
                    "guarded_contact_step": True,
                    "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
                    "generator_quarantine": quarantine,
                },
            }

    def plan_guarded_ik_path(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
        contact_target_xyz: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Plan one Cartesian-consistent guarded path to a contact target.

        The terminal pose is solved by cuRobo IK once.  The resulting joint
        path is then densified until forward kinematics proves every EEF step
        is at most roughly 2 mm and follows the requested approach corridor.
        """

        try:
            return self._compute_guarded_waypoint_path(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                attached_obj=attached_obj,
                contact_target_xyz=contact_target_xyz,
            )
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="arm",
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "ik_only": True,
                    "guarded_cartesian_path": True,
                    "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
                    "generator_quarantine": quarantine,
                },
            }

    def _compute_guarded_waypoint_path(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any,
        contact_target_xyz: np.ndarray | None,
    ) -> dict[str, Any]:
        """Batch-solve Cartesian guard points and certify the connected path."""

        started = time.monotonic()
        if contact_target_xyz is None:
            raise RuntimeError("guarded contact path requires a resolved target point")
        hand = _normalize_hand(hand)
        generator = self._generator(kind="arm", hand=hand)
        robot = self._find_robot()
        current_pose = self.get_eef_pose(hand)
        if current_pose is None:
            raise RuntimeError(f"R1Pro {hand} EEF pose unavailable for guarded path")
        start = np.asarray(current_pose[0], dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        total = float(np.linalg.norm(target - start))
        # IK layers regularize branch selection; they are not controller
        # waypoints.  Solve a sparse local graph, then independently densify
        # and FK-certify the selected joint path below so every executed EEF
        # step remains at most 2.2 mm along the guarded corridor.
        solver_layer_step_m = 0.008
        solver_layers = max(1, int(math.ceil(total / solver_layer_step_m)))
        distances = [
            total * float(index) / float(solver_layers)
            for index in range(1, solver_layers + 1)
        ]
        if not distances:
            distances = [0.0]
        positions = np.stack(
            [
                target
                if total <= 1e-9
                else start + (target - start) * (distance / total)
                for distance in distances
            ],
            axis=0,
        )
        target_quat = (
            np.asarray(current_pose[1], dtype=np.float64)
            if target_quat_xyzw is None
            else np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
        )
        quaternions = np.repeat(target_quat.reshape(1, 4), len(positions), axis=0)
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        original_solve_ik_batch = generator.solve_ik_batch
        seed_phase = int(self._guarded_seed_counter)
        self._guarded_seed_counter += 1
        guarded_selection_state: dict[str, Any] = {
            "previous": None,
            "graph_independent_seeds": True,
            "seed_phase": seed_phase,
        }

        def local_seeded_solve_ik_batch(
            curobo_generator: Any,
            start_state: Any,
            goal_pose: Any,
            plan_config: Any,
            link_poses: Any = None,
            emb_sel: Any = self._embodiment_cls.DEFAULT,
        ) -> tuple[Any, Any, list[Any]]:
            return self._solve_local_seeded_ik_batch(
                curobo_generator,
                start_state,
                goal_pose,
                plan_config,
                link_poses=link_poses,
                emb_sel=emb_sel,
                selection_state=guarded_selection_state,
            )

        generator.solve_ik_batch = types.MethodType(
            local_seeded_solve_ik_batch, generator
        )
        try:
            with _wall_clock_deadline(float(timeout_s), f"{hand} guarded cuRobo"):
                successes, _paths = generator.compute_trajectories(
                    torch.as_tensor(positions, dtype=torch.float32),
                    torch.as_tensor(quaternions, dtype=torch.float32),
                    max_attempts=5,
                    timeout=min(float(timeout_s), 8.0),
                    ik_fail_return=5,
                    enable_finetune_trajopt=False,
                    finetune_attempts=0,
                    return_full_result=False,
                    success_ratio=1.0 / max(int(generator.batch_size), 1),
                    attached_obj=attached_obj,
                    ik_only=True,
                    is_local=False,
                    skip_obstacle_update=False,
                    ik_world_collision_check=False,
                )
        finally:
            generator.solve_ik_batch = original_solve_ik_batch
        success_array = np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
        metrics: dict[str, Any] = {
            "successes": success_array.tolist(),
            "ik_only": True,
            "guarded_cartesian_path": True,
            "guarded_cartesian_targets": int(len(positions)),
            "guarded_cartesian_target_distances_m": distances,
            "guarded_ik_solver_layer_step_limit_m": solver_layer_step_m,
            "guarded_execution_cartesian_step_limit_m": 0.0022,
            "guarded_cartesian_solver_batches": int(
                math.ceil(len(positions) / max(int(generator.batch_size), 1))
            ),
            "planner_seed_count_per_target": LOCAL_GUARDED_IK_SEEDS,
            "guarded_seed_phase": seed_phase,
            "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
            "curobo_config": str(self._hand_config_path(hand, lock_trunk=True)),
            "collision_semantics": "not_used_for_admission",
            "collision_admission_enabled": False,
            "attached_collision_body": {"available": attached_obj is not None},
        }
        if len(success_array) != len(positions) or not bool(success_array.all()):
            return {"ok": False, "stop_reason": "unreachable", "metrics": metrics}
        current_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float32
        ).reshape(1, -1)
        remaining_s = float(timeout_s) - (time.monotonic() - started)
        if remaining_s <= 0.0:
            raise TimeoutError("guarded candidate graph exceeded timeout")
        with _wall_clock_deadline(remaining_s, f"{hand} guarded final-state graph"):
            raw_q_waypoints, graph_report = self._select_guarded_candidate_path(
                generator,
                current_q=current_q,
                candidate_q_sets=guarded_selection_state.get("candidate_q_sets", [])[
                    : len(positions)
                ],
                candidate_selectable_masks=guarded_selection_state.get(
                    "candidate_selectable_masks", []
                )[: len(positions)],
            )
        graph_report["method"] = "sequential_seeded_candidates+cartesian_graph"
        metrics["guarded_candidate_graph"] = graph_report
        if raw_q_waypoints is None:
            return {
                "ok": False,
                "stop_reason": "guarded_cartesian_path_unreachable",
                "metrics": metrics,
            }
        raw_adjacent = np.diff(np.vstack([current_q, raw_q_waypoints]), axis=0)
        raw_max_joint_delta = float(np.max(np.abs(raw_adjacent)))
        q_waypoints = raw_q_waypoints
        metrics["guarded_raw_max_adjacent_joint_delta"] = raw_max_joint_delta
        trajectory = None
        cartesian_attempts: list[dict[str, Any]] = []
        safe_guarded_candidates: list[
            tuple[int, float, str, np.ndarray, dict[str, Any]]
        ] = []
        for waypoint_mode, candidate_goals in (
            ("direct_terminal_ik", q_waypoints[-1:]),
            ("layered_candidate_graph", q_waypoints),
        ):
            candidate_trajectory, cartesian_report = (
                self._cartesian_certified_guarded_trajectory(
                    generator,
                    current_q=current_q,
                    q_goal=candidate_goals,
                    expected_distance_m=total,
                    time_dilate_contact=(
                        float(
                            np.linalg.norm(
                                np.asarray(contact_target_xyz, dtype=np.float64)
                                - target
                            )
                        )
                        > 0.005
                    ),
                )
            )
            attempt = {
                "waypoint_mode": waypoint_mode,
                "cartesian_report": cartesian_report,
            }
            if candidate_trajectory is None:
                attempt["accepted"] = False
                attempt["reason"] = "cartesian_path_invalid"
                cartesian_attempts.append(attempt)
                continue
            attempt["accepted"] = True
            joint_path_length = float(
                np.linalg.norm(
                    np.diff(np.vstack([current_q, candidate_trajectory]), axis=0),
                    axis=1,
                ).sum()
            )
            attempt["execution_waypoints"] = int(len(candidate_trajectory))
            attempt["joint_path_length_rad"] = joint_path_length
            cartesian_attempts.append(attempt)
            safe_guarded_candidates.append(
                (
                    int(len(candidate_trajectory)),
                    joint_path_length,
                    waypoint_mode,
                    candidate_trajectory,
                    cartesian_report,
                )
            )
        metrics["guarded_cartesian_attempts"] = cartesian_attempts
        if not safe_guarded_candidates:
            return {
                "ok": False,
                "stop_reason": "guarded_cartesian_path_invalid",
                "metrics": metrics,
            }
        (
            _selected_waypoints,
            _selected_joint_path_length,
            selected_mode,
            trajectory,
            selected_cartesian_report,
        ) = min(
            safe_guarded_candidates,
            key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
        )
        metrics["guarded_cartesian_path_report"] = selected_cartesian_report
        metrics["guarded_selected_waypoint_mode"] = selected_mode
        metrics["guarded_path_selection"] = (
            "minimum_execution_waypoints_then_joint_path_length"
        )
        metrics["guarded_interpolated_waypoints"] = int(len(trajectory))
        metrics["execution_mode"] = "online_robot_q_to_action"
        return {
            "ok": True,
            "joint_trajectory": trajectory,
            "reverse_joint_trajectory": np.vstack(
                [current_q.reshape(1, -1), trajectory]
            )[::-1].copy(),
            "metrics": metrics,
        }

    def _compute_arm_plan(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        ik_only: bool,
        base_xyyaw: np.ndarray | None = None,
        attached_obj: Any = None,
        return_ik_solution: bool = False,
        guarded_contact_target_xyz: np.ndarray | None = None,
        guarded_full_solution: bool = False,
        generator_kind: str = "arm",
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        if generator_kind not in {"arm", "attached_arm", "arm_with_trunk"}:
            raise ValueError(f"invalid arm generator kind {generator_kind!r}")
        trunk_assist = generator_kind == "arm_with_trunk"
        if trunk_assist and bool(ik_only):
            raise ValueError("arm-with-trunk planning requires a full trajectory")
        generator = self._generator(kind=generator_kind, hand=hand)
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        if target_quat_xyzw is None:
            current_eef = self.get_eef_pose(hand)
            if current_eef is None:
                raise RuntimeError(
                    f"R1Pro {hand} EEF pose unavailable for position-only target"
                )
            current_eef_quat = np.asarray(current_eef[1], dtype=np.float64)
            if base_xyyaw is not None:
                robot = self._find_robot()
                current_base = self._base_xy_yaw(robot)
                delta_yaw = _wrap_angle(float(base_xyyaw[2]) - float(current_base[2]))
                target_quat = _quat_multiply_xyzw(
                    _yaw_to_quat_xyzw(delta_yaw), current_eef_quat
                )
                orientation_mode = "preserve_eef_orientation_relative_to_candidate_base"
            else:
                target_quat = current_eef_quat
                orientation_mode = "preserve_current_eef_world_orientation"
        else:
            target_quat = target_quat_xyzw
            orientation_mode = "explicit_target"
        planner_target = np.asarray(target_xyz, dtype=np.float64)
        robot = self._find_robot()
        initial_joint_pos = (
            self._initial_joint_pos_for_base_candidate(robot, base_xyyaw)
            if base_xyyaw is not None
            else None
        )
        if initial_joint_pos is not None:
            initial_joint_pos = torch.as_tensor(initial_joint_pos, dtype=torch.float32)
        reachability_stage = "ik_exists"
        batch_size = max(int(generator.batch_size), 1)
        planner_targets = (
            torch.as_tensor(
                planner_target,
                dtype=torch.float32,
            )
            .reshape(1, 3)
            .repeat(batch_size, 1)
        )
        planner_quats = (
            torch.as_tensor(
                target_quat,
                dtype=torch.float32,
            )
            .reshape(1, 4)
            .repeat(batch_size, 1)
        )
        attachment_world_refreshed = attached_obj is not None
        with _wall_clock_deadline(float(timeout_s), f"{hand} ARM cuRobo"):
            if attachment_world_refreshed and not trunk_assist:
                # CuRobo resolves attached collision-mesh names through its
                # world registry even when collision costs are not used for
                # admission. A fresh generator has no world model until this
                # registry is populated. Refresh it only for attachment
                # bookkeeping; the solver call below still disables collision
                # admission and skips its own obstacle update.
                generator.update_obstacles()
            successes, paths = generator.compute_trajectories(
                planner_targets,
                planner_quats,
                initial_joint_pos=initial_joint_pos,
                max_attempts=5,
                timeout=min(float(timeout_s), 8.0),
                ik_fail_return=5,
                enable_finetune_trajopt=not bool(ik_only),
                finetune_attempts=0 if ik_only else 2,
                return_full_result=False,
                success_ratio=1.0 / batch_size,
                attached_obj=attached_obj,
                ik_only=bool(ik_only),
                is_local=False,
                skip_obstacle_update=not trunk_assist,
                ik_world_collision_check=trunk_assist,
            )
        success_array = np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
        success_indices = np.flatnonzero(success_array)
        metrics = {
            "successes": success_array.tolist(),
            "ik_only": bool(ik_only),
            "is_local": False,
            "reachability_stage": reachability_stage,
            "candidate_base_xyyaw": base_xyyaw.tolist()
            if base_xyyaw is not None
            else None,
            "collision_semantics": (
                "curobo_world_and_self_collision_admitted"
                if trunk_assist
                else "not_used_for_admission"
            ),
            "curobo_config": str(
                self._hand_config_path(
                    hand,
                    lock_trunk=False,
                    motion_scope="arm_with_trunk",
                )
                if trunk_assist
                else self._hand_config_path(
                    hand,
                    lock_trunk=generator_kind == "attached_arm",
                )
            ),
            "generator_kind": generator_kind,
            "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
            "attached_collision_body": {"available": attached_obj is not None},
            "success_ratio": 1.0 / batch_size,
            "planner_seed_count": batch_size,
            "orientation_mode": orientation_mode,
            "world_mesh_registry_refreshed": attachment_world_refreshed,
            "collision_admission_enabled": trunk_assist,
            "obstacle_update": trunk_assist or attachment_world_refreshed,
            "motion_scope": "arm_with_trunk" if trunk_assist else "arm_only",
            "collision_admission": {
                "available": bool(trunk_assist),
                "admitted": bool(trunk_assist and success_indices.size > 0),
                "world_collision_check": bool(trunk_assist),
                "self_collision_check": bool(trunk_assist),
                "obstacle_update": bool(trunk_assist),
                "full_trajectory": bool(trunk_assist and not ik_only),
                "source": "CuRoboMotionGenerator.compute_trajectories",
            },
        }
        if success_indices.size == 0:
            return {
                "ok": False,
                "stop_reason": "unreachable",
                "metrics": metrics,
            }
        if ik_only:
            metrics["reachable_by_ik"] = True
            if return_ik_solution:
                path = paths[int(success_indices[0])]
                q_goal, merge_report = self._merge_ik_solution_into_full_q(
                    generator,
                    robot,
                    path,
                )
                metrics["ik_solution_merge"] = merge_report
                current_q = np.asarray(
                    _jsonable(robot.get_joint_positions()), dtype=np.float32
                ).reshape(1, -1)
                max_joint_delta = float(np.max(np.abs(q_goal - current_q)))
                joint_step = 0.004 if guarded_contact_target_xyz is not None else 0.0075
                trajectory = _interpolate_joint_trajectory(
                    np.vstack([current_q, q_goal]),
                    max_inter_dist=joint_step,
                )[1:]
                metrics["maximum_ik_joint_delta_rad"] = max_joint_delta
                metrics["interpolated_waypoints"] = int(len(trajectory))
                metrics["interpolation_joint_step_rad"] = joint_step
                metrics["guarded_contact_step"] = guarded_contact_target_xyz is not None
                metrics["guarded_full_solution_requested"] = bool(guarded_full_solution)
                metrics["execution_mode"] = "online_robot_q_to_action"
                return {
                    "ok": True,
                    "joint_trajectory": trajectory,
                    "metrics": metrics,
                }
            return {"ok": True, "metrics": metrics}
        selected_index = int(success_indices[0])
        q_traj = generator.path_to_joint_trajectory(
            paths[selected_index], get_full_js=True
        )
        q_traj = _interpolate_joint_trajectory(q_traj, max_inter_dist=0.0075)
        metrics["selected_full_trajectory_candidate"] = selected_index
        metrics["full_trajectory_selection"] = "first_solver_success"
        metrics["trajectory_waypoints"] = int(len(q_traj))
        metrics["execution_mode"] = "online_robot_q_to_action"
        return {
            "ok": True,
            "joint_trajectory": q_traj,
            "metrics": metrics,
        }

    def _solve_local_seeded_ik_batch(
        self,
        generator: Any,
        start_state: Any,
        goal_pose: Any,
        _plan_config: Any,
        *,
        link_poses: Any = None,
        emb_sel: Any,
        selection_state: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, list[Any]]:
        """Use cuRobo's public IK API with the current state as seed/retract.

        This is the documented cuRobo servoing configuration: the current
        active-joint state regularizes and seeds each 2 mm guarded step.  It
        prevents a valid Cartesian target from selecting a distant IK branch.
        """

        solver = generator.mg[emb_sel].ik_solver
        previous = (
            selection_state.get("previous") if selection_state is not None else None
        )
        if previous is None:
            previous = start_state.position[0:1]
        previous_full = (
            selection_state.get("previous_full")
            if selection_state is not None
            else None
        )
        if selection_state is not None and previous_full is None:
            previous_full = np.asarray(
                _jsonable(self._find_robot().get_joint_positions()),
                dtype=np.float32,
            ).reshape(-1)
        paths = []
        success_flags = []
        last_result = None
        for index in range(int(goal_pose.batch)):
            single_link_poses = (
                {name: pose[index] for name, pose in link_poses.items()}
                if link_poses
                else None
            )
            # This pinned cuRobo build's get_seed() consumes
            # (batch, seeds, dof), despite an older solve_single docstring
            # describing the first two axes in the opposite order.
            seed_config = previous.unsqueeze(1).repeat(1, LOCAL_GUARDED_IK_SEEDS, 1)
            seed_rows = previous.new_tensor(
                np.arange(1, LOCAL_GUARDED_IK_SEEDS, dtype=np.float32)
            ).reshape(-1, 1)
            seed_columns = previous.new_tensor(
                np.arange(1, previous.shape[-1] + 1, dtype=np.float32)
            ).reshape(1, -1)
            seed_scales = previous.new_tensor(
                [0.0025, 0.005, 0.01, 0.02], dtype=previous.dtype
            )[
                previous.new_tensor(
                    np.arange(LOCAL_GUARDED_IK_SEEDS - 1, dtype=np.int64)
                ).long()
                % 4
            ].reshape(-1, 1)
            seed_phase = float(
                selection_state.get("seed_phase", 0)
                if selection_state is not None
                else 0
            )
            local_offsets = (
                seed_scales
                * (
                    seed_rows * seed_columns * math.sqrt(2.0)
                    + seed_phase * math.sqrt(5.0)
                ).sin()
            )
            seed_config[0, 1:, :] += local_offsets
            result = solver.solve_single(
                goal_pose[index],
                retract_config=previous,
                seed_config=seed_config,
                return_seeds=LOCAL_GUARDED_IK_SEEDS,
                num_seeds=LOCAL_GUARDED_IK_SEEDS,
                use_nn_seed=False,
                link_poses=single_link_poses,
            )
            last_result = result
            seed_success = result.success.reshape(-1)
            solution_state = result.js_solution
            solution_position = getattr(solution_state, "position", solution_state)
            solution_names = list(getattr(solution_state, "joint_names", None) or [])
            current_names = list(getattr(start_state, "joint_names", None) or [])
            if solution_names and current_names:
                solution_index = [solution_names.index(name) for name in current_names]
                solution_position = solution_position[..., solution_index]
            solution_position = solution_position.reshape(-1, previous.shape[-1])
            selectable = seed_success.clone()
            candidate_full: list[np.ndarray] = []
            if selection_state is not None:
                for seed_index in range(int(seed_success.shape[0])):
                    q_candidate, _merge_report = self._merge_ik_solution_into_full_q(
                        generator,
                        self._find_robot(),
                        result.js_solution[0, seed_index],
                    )
                    candidate = q_candidate.reshape(-1)
                    candidate_full.append(candidate)
                selection_state.setdefault("candidate_q_sets", []).append(
                    np.asarray(candidate_full, dtype=np.float32).tolist()
                )
                selection_state.setdefault("candidate_selectable_masks", []).append(
                    np.asarray(_jsonable(selectable), dtype=bool).tolist()
                )
            joint_delta = (solution_position - previous[0]).abs().amax(dim=-1)
            joint_delta[~selectable].fill_(float("inf"))
            graph_independent = bool(
                selection_state is not None
                and selection_state.get("graph_independent_seeds", False)
            )
            effective_success = selectable
            success = bool(effective_success.any())
            minimum_error_index = int(joint_delta.argmin().item())
            if graph_independent:
                graph_delta = (solution_position - previous[0]).abs().amax(dim=-1)
                graph_delta[~selectable].fill_(float("inf"))
                minimum_error_index = int(graph_delta.argmin().item())
            paths.append(result.js_solution[0, minimum_error_index])
            success_flags.append(effective_success.any())
            if success and not graph_independent:
                previous = (
                    solution_position[minimum_error_index].reshape(1, -1).detach()
                )
                if selection_state is not None:
                    previous_full = candidate_full[minimum_error_index]
                    selection_state.setdefault("selected_q_path", []).append(
                        previous_full.tolist()
                    )
        if selection_state is not None:
            selection_state["previous"] = previous
            selection_state["previous_full"] = previous_full
        if last_result is None:
            raise RuntimeError("guarded IK batch contained no goals")
        success = last_result.success.new_tensor(
            success_flags, dtype=last_result.success.dtype
        )
        return last_result, success, paths

    def _merge_ik_solution_into_full_q(
        self,
        generator: Any,
        robot: Any,
        path: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Merge an IK-only solution by name while preserving locked joints.

        OG 3.7.2's ``path_to_joint_trajectory(get_full_js=True)`` asks cuRobo
        to append lock joints even when this fixed cuRobo build already returns
        them in ``js_solution``.  Name-based merging supports both active-only
        and already-augmented results without ever changing a locked joint.
        """

        full_names = [str(name) for name in generator.robot_joint_names]
        current = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float32
        ).reshape(-1)
        if current.size != len(full_names):
            raise RuntimeError(
                "robot joint state and CuRobo robot_joint_names size mismatch: "
                f"{current.size} != {len(full_names)}"
            )
        path_names = [str(name) for name in (getattr(path, "joint_names", None) or [])]
        if not path_names or len(set(path_names)) != len(path_names):
            raise RuntimeError("IK solution has missing or duplicate joint names")
        values = np.asarray(
            _jsonable(getattr(path, "position", None)), dtype=np.float32
        )
        if values.size == 0 or values.shape[-1] != len(path_names):
            raise RuntimeError(
                "IK solution position width does not match its joint names: "
                f"shape={values.shape}, names={len(path_names)}"
            )
        values = values.reshape(-1, len(path_names))[-1]
        lock_names = {
            str(name)
            for name in generator.mg[
                self._embodiment_cls.DEFAULT
            ].kinematics.kinematics_config.lock_jointstate.joint_names
        }
        full_index = {name: index for index, name in enumerate(full_names)}
        unknown = sorted(set(path_names) - set(full_names))
        if unknown:
            raise RuntimeError(f"IK solution contains unknown joints: {unknown}")
        merged = current.copy()
        active_written = []
        locked_preserved = []
        for name, value in zip(path_names, values, strict=True):
            if name in lock_names:
                locked_preserved.append(name)
                continue
            merged[full_index[name]] = float(value)
            active_written.append(name)
        if not active_written:
            raise RuntimeError("IK solution did not contain any active joints")
        return merged.reshape(1, -1), {
            "method": "joint_name_merge_preserving_current_locked_joints",
            "solution_joint_count": len(path_names),
            "active_joint_count": len(active_written),
            "locked_joint_count": len(lock_names),
            "locked_solution_entries_ignored": len(locked_preserved),
            "active_joint_names": active_written,
        }

    def _merge_base_ik_solution_into_full_q(
        self,
        generator: Any,
        robot: Any,
        path: Any,
        *,
        call_start_q: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Merge one BASE IK result without augmenting its lock joints twice."""

        full_names = tuple(str(name) for name in generator.robot_joint_names)
        robot_names = tuple(str(name) for name in (getattr(robot, "joints", {}) or {}))
        start = np.asarray(call_start_q, dtype=np.float32).reshape(-1)
        if (
            len(full_names) != 28
            or len(set(full_names)) != 28
            or full_names != robot_names
            or start.size != len(full_names)
        ):
            raise RuntimeError(
                "R1Pro BASE IK merge requires the exact 28-joint live layout"
            )
        active_names = tuple(
            str(name)
            for name in generator.mg[
                self._embodiment_cls.BASE
            ].kinematics.joint_names
        )
        lock_names = tuple(
            str(name)
            for name in generator.mg[
                self._embodiment_cls.BASE
            ].kinematics.kinematics_config.lock_jointstate.joint_names
        )
        if (
            active_names != BASE_ACTIVE_JOINT_NAMES
            or len(lock_names) != 25
            or set(active_names).intersection(lock_names)
            or set(active_names).union(lock_names) != set(full_names)
        ):
            raise RuntimeError(
                "R1Pro BASE runtime active/locked joint partition is invalid"
            )
        path_names = tuple(
            str(name) for name in (getattr(path, "joint_names", None) or ())
        )
        if len(path_names) != len(set(path_names)):
            raise RuntimeError("BASE IK solution contains duplicate joint names")
        representation = (
            "active_only"
            if set(path_names) == set(active_names) and len(path_names) == 3
            else (
                "already_full"
                if set(path_names) == set(full_names) and len(path_names) == 28
                else None
            )
        )
        if representation is None:
            raise RuntimeError(
                "BASE IK solution must contain exactly active-only or full joints"
            )
        values = np.asarray(
            _jsonable(getattr(path, "position", None)), dtype=np.float32
        )
        if values.size == 0 or values.shape[-1] != len(path_names):
            raise RuntimeError(
                "BASE IK solution width does not match its joint names: "
                f"shape={values.shape}, names={len(path_names)}"
            )
        values = values.reshape(-1, len(path_names))[-1]
        if not np.isfinite(values).all():
            raise RuntimeError("BASE IK solution contains non-finite values")
        source_index = {name: index for index, name in enumerate(path_names)}
        full_index = {name: index for index, name in enumerate(full_names)}
        merged = start.copy()
        for name in active_names:
            merged[full_index[name]] = float(values[source_index[name]])
        return merged.reshape(1, -1), {
            "method": "joint_name_merge_preserving_call_start_locked_joints",
            "source_representation": representation,
            "source_joint_count": len(path_names),
            "active_joint_names": list(active_names),
            "active_joint_count": len(active_names),
            "locked_joint_names": list(lock_names),
            "locked_joint_count": len(lock_names),
            "locked_solution_entries_ignored": (
                len(lock_names) if representation == "already_full" else 0
            ),
            "get_full_js_called": False,
        }

    def _cartesian_certified_guarded_trajectory(
        self,
        generator: Any,
        *,
        current_q: np.ndarray,
        q_goal: np.ndarray,
        expected_distance_m: float,
        time_dilate_contact: bool = False,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Densify an IK solution and certify its Cartesian approach path."""

        # Press contact remains intentionally time-dilated.  Pick approach can
        # start from a 0.012 rad stride because the
        # FK gate below independently rejects any result whose Cartesian EEF
        # increments exceed 2.2 mm.  This avoids needless duplicate simulator
        # setpoints while preserving the actual guarded-motion contract.
        joint_step_rad = 0.006 if time_dilate_contact else 0.012
        contact_joint_step_rad = 0.002
        contact_zone_m = 0.006
        report: dict[str, Any] = {}
        for _attempt in range(5):
            full = _interpolate_joint_trajectory(
                np.vstack([current_q, q_goal]),
                max_inter_dist=joint_step_rad,
            )
            positions = self._curobo_eef_positions(generator, full)
            remaining_to_contact = np.linalg.norm(positions - positions[-1], axis=1)
            contact_indices = np.flatnonzero(remaining_to_contact <= contact_zone_m)
            if time_dilate_contact and contact_indices.size:
                contact_start = max(0, int(contact_indices[0]) - 1)
                if contact_start < len(full) - 1:
                    fine = _interpolate_joint_trajectory(
                        full[contact_start:],
                        max_inter_dist=contact_joint_step_rad,
                    )
                    full = np.vstack([full[:contact_start], fine])
                    positions = self._curobo_eef_positions(generator, full)
            original_waypoints = int(len(full))
            terminal_trim_error_m = 0.0
            terminal_nullspace_trimmed = False
            if not time_dilate_contact and len(full) > 2:
                terminal_position = positions[-1].copy()
                terminal_errors = np.linalg.norm(
                    positions - terminal_position,
                    axis=1,
                )
                terminal_indices = np.flatnonzero(terminal_errors <= 0.0015)
                if terminal_indices.size:
                    terminal_index = int(terminal_indices[0])
                    if 0 < terminal_index < len(full) - 1:
                        terminal_trim_error_m = float(terminal_errors[terminal_index])
                        full = full[: terminal_index + 1]
                        positions = positions[: terminal_index + 1]
                        terminal_nullspace_trimmed = True
            full, terminal_smoothing = _terminally_smoothed_joint_trajectory(full)
            # Recompute FK after smoothing.  The selected result below is then
            # passed through the existing target-excluded world+self check in
            # _compute_guarded_waypoint_path, including every ease-out and
            # exact endpoint sample.
            positions = self._curobo_eef_positions(generator, full)
            segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            max_cartesian_step_m = float(segment_lengths.max(initial=0.0))
            chord = positions[-1] - positions[0]
            chord_length = float(np.linalg.norm(chord))
            if chord_length > 1e-9:
                axis = chord / chord_length
                relative = positions - positions[0]
                along = relative @ axis
                lateral = np.linalg.norm(relative - along[:, None] * axis, axis=1)
                max_lateral_deviation_m = float(lateral.max(initial=0.0))
                min_progress_delta_m = float(np.diff(along).min(initial=0.0))
            else:
                max_lateral_deviation_m = 0.0
                min_progress_delta_m = 0.0
            distance_error_m = abs(chord_length - float(expected_distance_m))
            report = {
                "available": True,
                "certified": False,
                "certification_frame": "curobo_base_link",
                "waypoints_including_start": int(len(full)),
                "waypoints_before_terminal_nullspace_trim": original_waypoints,
                "terminal_nullspace_trimmed": terminal_nullspace_trimmed,
                "terminal_trim_error_m": terminal_trim_error_m,
                "terminal_trim_tolerance_m": 0.0015,
                "execution_waypoints": int(max(0, len(full) - 1)),
                "joint_step_rad": joint_step_rad,
                "contact_joint_step_rad": contact_joint_step_rad,
                "contact_zone_m": contact_zone_m,
                "contact_time_dilated": bool(time_dilate_contact),
                "terminal_smoothing": terminal_smoothing,
                "max_cartesian_step_m": max_cartesian_step_m,
                "max_cartesian_step_limit_m": 0.0022,
                "max_lateral_deviation_m": max_lateral_deviation_m,
                "max_lateral_deviation_limit_m": 0.002,
                "minimum_progress_delta_m": min_progress_delta_m,
                "minimum_progress_delta_limit_m": -0.0005,
                "cartesian_chord_length_m": chord_length,
                "expected_distance_m": float(expected_distance_m),
                "distance_consistency_error_m": distance_error_m,
                "distance_consistency_limit_m": 0.01,
            }
            certified = bool(
                max_cartesian_step_m <= 0.0022
                and max_lateral_deviation_m <= 0.002
                and min_progress_delta_m >= -0.0005
                and distance_error_m <= 0.01
            )
            report["certified"] = certified
            if certified:
                return full[1:].astype(np.float32, copy=False), report
            if max_cartesian_step_m > 0.0022:
                joint_step_rad *= 0.5
                continue
            break
        return None, report

    def _curobo_eef_positions(self, generator: Any, q_trajectory: Any) -> np.ndarray:
        """Return cuRobo end-effector positions for full robot q samples."""

        positions, _quaternions = self._curobo_eef_poses(generator, q_trajectory)
        return positions

    def _curobo_eef_poses(
        self, generator: Any, q_trajectory: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cuRobo EEF positions and xyzw quaternions for full q samples."""

        from omnigibson import lazy

        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        emb_sel = self._embodiment_cls.DEFAULT
        q_tensor = generator._tensor_args.to_device(
            torch.as_tensor(
                np.asarray(_jsonable(q_trajectory), dtype=np.float32),
                dtype=torch.float32,
            )
        )
        cu_js = lazy.curobo.types.state.JointState(
            position=q_tensor,
            joint_names=generator.robot_joint_names,
        ).get_ordered_joint_state(generator.mg[emb_sel].kinematics.joint_names)
        state = generator.mg[emb_sel].kinematics.compute_kinematics(cu_js)
        positions = np.asarray(_jsonable(state.ee_position), dtype=np.float64).reshape(
            -1, 3
        )
        quaternions_wxyz = np.asarray(
            _jsonable(state.ee_quaternion), dtype=np.float64
        ).reshape(-1, 4)
        quaternions_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
        return positions, quaternions_xyzw

    def whole_body_eef_poses(
        self,
        hand: str,
        q_trajectory: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return selected-EEF FK for a certificate-bound full-q trajectory."""

        generator = self._generator(
            kind="whole_body",
            hand=_normalize_hand(hand),
        )
        return self._curobo_eef_poses(generator, q_trajectory)

    def torso_poses(
        self,
        q_trajectory: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return torso_link4 FK for a live-order full-q trajectory."""

        generator = self._generator(kind="torso")
        live_names, generator_names = self._torso_joint_layouts(
            generator=generator,
        )
        generator_trajectory = _reorder_joint_trajectory(
            q_trajectory,
            source_names=live_names,
            target_names=generator_names,
        )
        return self._curobo_eef_poses(generator, generator_trajectory)

    def torso_joint_names(self) -> tuple[str, ...]:
        """Return the live full-q layout shared by planning and execution."""

        live_names, _generator_names = self._torso_joint_layouts()
        return live_names

    def whole_body_joint_names(self, hand: str) -> tuple[str, ...]:
        """Return the ordered full-q layout shared by planning and execution."""

        generator = self._generator(
            kind="whole_body",
            hand=_normalize_hand(hand),
        )
        generator_names = tuple(
            str(name) for name in generator.robot_joint_names
        )
        robot = self._find_robot()
        live_names = tuple(
            str(name)
            for name in (getattr(robot, "joints", {}) or {}).keys()
        )
        live_q = np.asarray(
            _jsonable(robot.get_joint_positions()),
            dtype=np.float32,
        ).reshape(-1)
        if (
            live_names != generator_names
            or len(live_names) != len(live_q)
            or not np.isfinite(live_q).all()
        ):
            raise RuntimeError(
                "live R1Pro q/action joint layout diverged from whole-body generator"
            )
        return live_names

    def _whole_body_eef_path_report(
        self,
        generator: Any,
        dense_q: Any,
        *,
        call_start_xyz: Any,
        target_xyz: Any,
        target_quat_xyzw: Any | None = None,
        precomputed_eef_poses: tuple[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """Certify the selected EEF Cartesian path before admitting actions."""

        start = np.asarray(call_start_xyz, dtype=np.float64).reshape(3)
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        target_quat = (
            None
            if target_quat_xyzw is None
            else np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
        )
        q = np.asarray(_jsonable(dense_q), dtype=np.float32)
        try:
            positions, quaternions = (
                self._curobo_eef_poses(generator, q)
                if precomputed_eef_poses is None
                else precomputed_eef_poses
            )
            positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
            quaternions = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
            if (
                q.ndim != 2
                or len(q) < 1
                or positions.shape != (len(q), 3)
                or quaternions.shape != (len(q), 4)
                or not np.isfinite(q).all()
                or not np.isfinite(positions).all()
                or not np.isfinite(quaternions).all()
                or not np.isfinite(start).all()
                or not np.isfinite(target).all()
                or (target_quat is not None and not np.isfinite(target_quat).all())
            ):
                raise RuntimeError("selected EEF FK path is invalid")
            start_fk_error = float(np.linalg.norm(positions[0] - start))
            terminal_error = float(np.linalg.norm(positions[-1] - target))
            target_errors = np.linalg.norm(positions - target, axis=1)
            start_excursions = np.linalg.norm(positions - positions[0], axis=1)
            cartesian_steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            cumulative_cartesian_path = float(np.sum(cartesian_steps))
            direct = target - positions[0]
            direct_distance = float(np.linalg.norm(direct))
            if direct_distance > 1e-9:
                axis = direct / direct_distance
                relative = positions - positions[0]
                along = relative @ axis
                lateral = np.linalg.norm(
                    relative - along[:, None] * axis,
                    axis=1,
                )
            else:
                along = np.zeros((len(positions),), dtype=np.float64)
                lateral = start_excursions
            max_excursion = float(np.max(start_excursions, initial=0.0))
            max_lateral = float(np.max(lateral, initial=0.0))
            min_along = float(np.min(along, initial=0.0))
            max_along = float(np.max(along, initial=0.0))
            along_steps = np.diff(along)
            max_reverse_step = float(
                np.max(-along_steps, initial=0.0)
            )
            max_cartesian_step = float(np.max(cartesian_steps, initial=0.0))
            max_orientation_error = 0.0
            start_orientation_error = 0.0
            terminal_orientation_error = 0.0
            max_orientation_step = 0.0
            max_orientation_reverse_progress = 0.0
            cumulative_orientation_path = 0.0
            intentional_orientation_change = False
            if target_quat is not None:
                target_norm = float(np.linalg.norm(target_quat))
                quat_norms = np.linalg.norm(quaternions, axis=1)
                if target_norm <= 1e-9 or np.any(quat_norms <= 1e-9):
                    raise RuntimeError("selected EEF FK quaternion path is invalid")
                normalized_target = target_quat / target_norm
                normalized_quats = quaternions / quat_norms[:, None]
                dots = np.clip(
                    np.abs(normalized_quats @ normalized_target),
                    0.0,
                    1.0,
                )
                orientation_errors = 2.0 * np.arccos(dots)
                max_orientation_error = float(
                    np.max(orientation_errors, initial=0.0)
                )
                start_orientation_error = float(orientation_errors[0])
                terminal_orientation_error = float(orientation_errors[-1])
                intentional_orientation_change = (
                    start_orientation_error
                    > WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9
                )
                if len(normalized_quats) > 1:
                    adjacent_dots = np.clip(
                        np.abs(
                            np.sum(
                                normalized_quats[:-1] * normalized_quats[1:],
                                axis=1,
                            )
                        ),
                        0.0,
                        1.0,
                    )
                    orientation_steps = 2.0 * np.arccos(adjacent_dots)
                    cumulative_orientation_path = float(
                        np.sum(orientation_steps)
                    )
                    max_orientation_step = float(
                        np.max(orientation_steps, initial=0.0)
                    )
                    max_orientation_reverse_progress = float(
                        np.max(np.diff(orientation_errors), initial=0.0)
                    )
            short_target = direct_distance <= (
                WHOLE_BODY_EEF_SHORT_TARGET_DISTANCE_M + 1e-9
            )
            max_excursion_limit = (
                direct_distance + WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M
            )
            checks = {
                "start_fk_matches_call_start": (
                    start_fk_error <= WHOLE_BODY_EEF_START_FK_TOLERANCE_M + 1e-9
                ),
                "terminal_reaches_target": (
                    terminal_error <= WHOLE_BODY_EEF_TERMINAL_TOLERANCE_M + 1e-9
                ),
                "all_target_cartesian_step": (
                    max_cartesian_step
                    <= WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M + 1e-9
                ),
                "short_target_start_excursion": (
                    not short_target
                    or max_excursion
                    <= max_excursion_limit + 1e-9
                ),
                "short_target_segment_lateral": (
                    not short_target
                    or max_lateral <= WHOLE_BODY_EEF_SHORT_MAX_LATERAL_M + 1e-9
                ),
                "short_target_no_reverse_or_overshoot": (
                    not short_target
                    or (
                        min_along
                        >= -WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M - 1e-9
                        and max_along
                        <= direct_distance
                        + WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M
                        + 1e-9
                    )
                ),
                "short_target_cartesian_step": (
                    not short_target
                    or max_cartesian_step
                    <= WHOLE_BODY_EEF_SHORT_MAX_CARTESIAN_STEP_M + 1e-9
                ),
                "short_target_axis_monotonic": (
                    not short_target
                    or direct_distance <= 1e-9
                    or max_reverse_step
                    <= WHOLE_BODY_EEF_SHORT_MAX_REVERSE_STEP_M + 1e-9
                ),
                "short_target_cumulative_path": (
                    not short_target
                    or cumulative_cartesian_path
                    <= direct_distance
                    + WHOLE_BODY_EEF_SHORT_MAX_CUMULATIVE_EXCESS_M
                    + 1e-9
                ),
                "short_target_orientation_preserved": (
                    not short_target
                    or target_quat is None
                    or intentional_orientation_change
                    or max_orientation_error
                    <= WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9
                ),
                "short_target_rotation_terminal": (
                    not short_target
                    or target_quat is None
                    or not intentional_orientation_change
                    or terminal_orientation_error
                    <= WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9
                ),
                "short_target_rotation_step": (
                    not short_target
                    or target_quat is None
                    or not intentional_orientation_change
                    or max_orientation_step
                    <= WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD + 1e-9
                ),
                "short_target_rotation_monotonic": (
                    not short_target
                    or target_quat is None
                    or not intentional_orientation_change
                    or max_orientation_reverse_progress
                    <= WHOLE_BODY_EEF_WRIST_MAX_REVERSE_PROGRESS_RAD + 1e-9
                ),
                "short_target_rotation_cumulative_path": (
                    not short_target
                    or target_quat is None
                    or not intentional_orientation_change
                    or cumulative_orientation_path
                    <= start_orientation_error
                    + WHOLE_BODY_EEF_WRIST_MAX_CUMULATIVE_EXCESS_RAD
                    + 1e-9
                ),
            }
            shared_admission = _eef_pose_path_admission_report(
                positions,
                quaternions,
                call_start_xyz=start,
                target_xyz=target,
                target_quat_xyzw=target_quat,
            )
            if checks != shared_admission["checks"]:
                raise RuntimeError(
                    "planner and runtime EEF admission checks diverged"
                )
            checks = dict(shared_admission["checks"])
            positions_f32 = np.ascontiguousarray(positions, dtype=np.float32)
            quaternions_f32 = np.ascontiguousarray(quaternions, dtype=np.float32)
            return {
                "available": True,
                "admitted": all(checks.values()),
                "checks": checks,
                "waypoint_count": int(len(positions)),
                "start_fk_error_m": start_fk_error,
                "start_fk_tolerance_m": WHOLE_BODY_EEF_START_FK_TOLERANCE_M,
                "terminal_error_m": terminal_error,
                "terminal_tolerance_m": WHOLE_BODY_EEF_TERMINAL_TOLERANCE_M,
                "call_start_to_target_distance_m": direct_distance,
                "short_target": short_target,
                "short_target_distance_limit_m": (
                    WHOLE_BODY_EEF_SHORT_TARGET_DISTANCE_M
                ),
                "max_start_excursion_m": max_excursion,
                "max_start_excursion_limit_m": max_excursion_limit,
                "max_segment_lateral_m": max_lateral,
                "max_segment_lateral_limit_m": (
                    WHOLE_BODY_EEF_SHORT_MAX_LATERAL_M
                ),
                "min_segment_along_m": min_along,
                "min_segment_along_limit_m": (
                    -WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M
                ),
                "max_segment_along_m": max_along,
                "max_segment_along_limit_m": (
                    direct_distance + WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M
                ),
                "max_cartesian_step_m": max_cartesian_step,
                "all_target_cartesian_step_limit_m": (
                    WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M
                ),
                "max_cartesian_step_limit_m": (
                    WHOLE_BODY_EEF_SHORT_MAX_CARTESIAN_STEP_M
                ),
                "max_reverse_step_m": max_reverse_step,
                "max_reverse_step_limit_m": (
                    WHOLE_BODY_EEF_SHORT_MAX_REVERSE_STEP_M
                ),
                "cumulative_cartesian_path_m": cumulative_cartesian_path,
                "cumulative_cartesian_path_limit_m": (
                    direct_distance
                    + WHOLE_BODY_EEF_SHORT_MAX_CUMULATIVE_EXCESS_M
                ),
                "max_orientation_error_rad": max_orientation_error,
                "max_orientation_error_limit_rad": (
                    WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD
                ),
                "intentional_orientation_change": (
                    intentional_orientation_change
                ),
                "start_orientation_error_rad": start_orientation_error,
                "terminal_orientation_error_rad": terminal_orientation_error,
                "max_orientation_step_rad": max_orientation_step,
                "max_orientation_step_limit_rad": (
                    WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD
                ),
                "max_orientation_reverse_progress_rad": (
                    max_orientation_reverse_progress
                ),
                "max_orientation_reverse_progress_limit_rad": (
                    WHOLE_BODY_EEF_WRIST_MAX_REVERSE_PROGRESS_RAD
                ),
                "cumulative_orientation_path_rad": cumulative_orientation_path,
                "cumulative_orientation_path_limit_rad": (
                    start_orientation_error
                    + WHOLE_BODY_EEF_WRIST_MAX_CUMULATIVE_EXCESS_RAD
                ),
                "nominal_max_target_error_m": float(
                    np.max(target_errors, initial=0.0)
                ),
                "positions_sha256": hashlib.sha256(
                    positions_f32.tobytes()
                ).hexdigest(),
                "quaternions_sha256": hashlib.sha256(
                    quaternions_f32.tobytes()
                ).hexdigest(),
            }
        except Exception as exc:
            return {
                "available": False,
                "admitted": False,
                "checks": {},
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def _select_guarded_candidate_path(
        self,
        generator: Any,
        *,
        current_q: np.ndarray,
        candidate_q_sets: Any,
        candidate_selectable_masks: Any,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Find a minimum-motion layered IK path with Cartesian-consistent edges."""

        candidates = np.asarray(candidate_q_sets, dtype=np.float32)
        selectable = np.asarray(candidate_selectable_masks, dtype=bool)
        if candidates.ndim != 3 or selectable.shape != candidates.shape[:2]:
            raise RuntimeError(
                "guarded candidate graph shape mismatch: "
                f"candidates={candidates.shape} selectable={selectable.shape}"
            )
        layer_count, candidate_count, _dof = candidates.shape
        previous_qs = np.asarray(current_q, dtype=np.float32).reshape(1, -1)
        previous_costs = np.asarray([0.0], dtype=np.float64)
        backpointers: list[np.ndarray] = []
        layer_reports = []
        for layer in range(layer_count):
            edges: list[tuple[int, int, int, int, float]] = []
            edge_waypoints = []
            offset = 0
            branch_rejected_edges = 0
            for previous_index, previous in enumerate(previous_qs):
                if not np.isfinite(previous_costs[previous_index]):
                    continue
                for candidate_index in range(candidate_count):
                    if not selectable[layer, candidate_index]:
                        continue
                    candidate = candidates[layer, candidate_index]
                    if float(np.max(np.abs(candidate - previous))) > 2.0:
                        branch_rejected_edges += 1
                        continue
                    segment = _interpolate_joint_trajectory(
                        np.vstack([previous, candidate]),
                        max_inter_dist=0.01,
                    )
                    start_offset = offset
                    offset += int(len(segment))
                    edge_waypoints.append(segment)
                    edge_cost = float(np.linalg.norm(candidate - previous))
                    edges.append(
                        (
                            previous_index,
                            candidate_index,
                            start_offset,
                            offset,
                            edge_cost,
                        )
                    )
            if not edge_waypoints:
                return None, {
                    "available": True,
                    "selected": False,
                    "reason": "no_selectable_candidate_edges",
                    "failed_layer": layer,
                    "layers": layer_reports,
                }
            eef_positions = self._curobo_eef_positions(
                generator, np.concatenate(edge_waypoints, axis=0)
            )
            next_costs = np.full(candidate_count, np.inf, dtype=np.float64)
            next_backpointer = np.full(candidate_count, -1, dtype=np.int64)
            accepted_edges = 0
            cartesian_rejected_edges = 0
            for (
                previous_index,
                candidate_index,
                start_offset,
                end_offset,
                edge_cost,
            ) in edges:
                edge_positions = eef_positions[start_offset:end_offset]
                chord = edge_positions[-1] - edge_positions[0]
                chord_length = float(np.linalg.norm(chord))
                if chord_length > 1e-9:
                    axis = chord / chord_length
                    relative = edge_positions - edge_positions[0]
                    along = relative @ axis
                    lateral = np.linalg.norm(relative - along[:, None] * axis, axis=1)
                    max_lateral = float(lateral.max(initial=0.0))
                    min_progress = float(np.diff(along).min(initial=0.0))
                else:
                    max_lateral = 0.0
                    min_progress = 0.0
                if max_lateral > 0.002 or min_progress < -0.0005:
                    cartesian_rejected_edges += 1
                    continue
                accepted_edges += 1
                cost = previous_costs[previous_index] + edge_cost
                if cost < next_costs[candidate_index]:
                    next_costs[candidate_index] = cost
                    next_backpointer[candidate_index] = previous_index
            layer_reports.append(
                {
                    "layer": layer,
                    "candidate_count": int(selectable[layer].sum()),
                    "tested_edges": len(edges),
                    "branch_rejected_edges": branch_rejected_edges,
                    "accepted_edges": accepted_edges,
                    "cartesian_rejected_edges": cartesian_rejected_edges,
                    "reachable_candidates": int(np.isfinite(next_costs).sum()),
                }
            )
            if not bool(np.isfinite(next_costs).any()):
                return None, {
                    "available": True,
                    "selected": False,
                    "reason": "no_cartesian_consistent_path",
                    "failed_layer": layer,
                    "layers": layer_reports,
                }
            backpointers.append(next_backpointer)
            previous_qs = candidates[layer]
            previous_costs = next_costs
        selected_indices = [int(np.argmin(previous_costs))]
        for layer in range(layer_count - 1, 0, -1):
            selected_indices.append(int(backpointers[layer][selected_indices[-1]]))
        selected_indices.reverse()
        selected = np.stack(
            [candidates[layer, index] for layer, index in enumerate(selected_indices)],
            axis=0,
        )
        return selected, {
            "available": True,
            "selected": True,
            "selected_candidate_indices": selected_indices,
            "path_cost_l2": float(np.min(previous_costs)),
            "layers": layer_reports,
        }

    def plan_base_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        standoff_m: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        started = time.monotonic()
        deadline = started + float(timeout_s)
        try:
            robot = self._find_robot()
            current = self._base_xy_yaw(robot)
            candidates = self._ranked_base_candidates(
                robot,
                hand=hand,
                target_xyz=target_xyz,
                standoff_m=float(standoff_m),
                deadline=deadline,
            )
            candidate_selection_elapsed_s = round(time.monotonic() - started, 3)
            if not candidates:
                summary = dict(self._last_base_candidate_summary)
                return {
                    "ok": False,
                    "stop_reason": "navigation_unreachable",
                    "metrics": {
                        "candidate_count": 0,
                        "reason": "no traversable IK-reachable station",
                        "candidate_summary": summary,
                    },
                }
            candidate_trace = [
                {
                    "xyyaw": item["xyyaw"].tolist(),
                    "geodesic_distance_m": item.get("geodesic_distance_m"),
                    "path_rank_distance_m": item.get("path_rank_distance_m"),
                    "reachability_target_xyz": item.get("reachability_target_xyz"),
                    "reachability_reason": item.get("reachability_reason"),
                    "reachability_stage": item.get("reachability_stage"),
                    "reachability_target_quat_xyzw": (
                        item.get("reachability", {}).get("selected_target_quat_xyzw")
                    ),
                }
                for item in candidates[:8]
            ]
            base_plan_attempts = []
            base_plan = None
            best = None
            best_candidate = None
            for rank, candidate in enumerate(candidates):
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    break
                attempt_started = time.monotonic()
                attempt = self._compute_base_plan(
                    target_xyyaw=candidate["xyyaw"],
                    timeout_s=remaining_s,
                    skip_obstacle_update=True,
                )
                base_plan_attempts.append(
                    {
                        "rank": rank,
                        "xyyaw": candidate["xyyaw"].tolist(),
                        "ok": bool(attempt.get("ok")),
                        "stop_reason": attempt.get("stop_reason"),
                        "elapsed_s": round(time.monotonic() - attempt_started, 3),
                        "metrics": attempt.get("metrics", {}),
                    }
                )
                if attempt.get("ok"):
                    best = _canonical_base_xyyaw(candidate["xyyaw"])
                    best_candidate = candidate
                    base_plan = attempt
                    break
            if base_plan is None or best is None or best_candidate is None:
                timed_out = time.monotonic() - started >= float(timeout_s)
                return {
                    "ok": False,
                    "stop_reason": "timeout" if timed_out else "base_plan_failed",
                    "metrics": {
                        "candidate_count": len(candidates),
                        "candidate_trace": candidate_trace,
                        "base_plan_attempts": base_plan_attempts,
                        "current_base": current.tolist(),
                        "elapsed_s": round(time.monotonic() - started, 3),
                    },
                }
            metrics = {
                **base_plan.get("metrics", {}),
                "candidate_count": len(candidates),
                "candidate_selection_elapsed_s": candidate_selection_elapsed_s,
                "candidate_trace": candidate_trace,
                "base_plan_attempts": base_plan_attempts,
                "post_base_reachability_required": True,
                "post_base_reachability_target_xyz": best_candidate[
                    "reachability_target_xyz"
                ],
                "requested_surface_target_xyz": target_xyz.tolist(),
                "base_goal": best.tolist(),
                "current_base": current.tolist(),
            }
            return {
                "ok": True,
                "joint_trajectory": base_plan["joint_trajectory"],
                "base_goal": best.tolist(),
                "reachability_target_xyz": best_candidate["reachability_target_xyz"],
                "reachability_target_quat_xyzw": best_candidate.get(
                    "reachability", {}
                ).get("selected_target_quat_xyzw"),
                "metrics": metrics,
            }
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="base",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout"
                    if isinstance(exc, TimeoutError)
                    else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
                    "generator_quarantine": quarantine,
                },
            }

    def plan_navigation_trajectory(
        self,
        *,
        target_xyz: Any,
        standoff_m: float,
        max_travel_m: float,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Plan one bounded BASE-only stage along OG's robot-eroded A* path."""

        started = time.monotonic()
        target = _as_xyz(target_xyz)
        standoff = float(standoff_m)
        max_travel = float(max_travel_m)
        timeout = float(timeout_s)
        if not math.isfinite(standoff) or standoff <= 0.0:
            raise ValueError("standoff_m must be finite and positive")
        if not math.isfinite(max_travel) or max_travel <= 0.0:
            raise ValueError("max_travel_m must be finite and positive")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = started + timeout

        robot = self._find_robot()
        scene = self._scene(robot)
        trav_map = getattr(scene, "trav_map", None)
        if trav_map is None or not callable(
            getattr(trav_map, "get_shortest_path", None)
        ):
            return {
                "ok": False,
                "stop_reason": "navigation_planner_unavailable",
                "metrics": {
                    "navigation_path": {
                        "source": "official_robot_eroded_traversability",
                        "entire_path_requested": True,
                        "full_path_used": False,
                        "dynamic_world_collision_admission": False,
                    }
                },
            }
        current = self._base_xy_yaw(robot)
        floor = self._current_floor(scene, current)
        candidates = [
            candidate
            for candidate in _base_candidates(target, standoff_m=standoff)
            if self._candidate_is_traversable(trav_map, candidate, floor=floor)
        ]
        path_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            if time.monotonic() >= deadline:
                raise TimeoutError("navigation candidate search timed out")
            path_value, geodesic_value = trav_map.get_shortest_path(
                floor,
                current[:2],
                candidate[:2],
                entire_path=True,
                robot=robot,
            )
            if path_value is None or geodesic_value is None:
                continue
            path = np.asarray(_jsonable(path_value), dtype=np.float64)
            if (
                path.ndim != 2
                or path.shape[1] != 2
                or len(path) < 1
                or not np.isfinite(path).all()
            ):
                continue
            if float(np.linalg.norm(path[0] - current[:2])) > 1e-9:
                path = np.vstack([current[:2], path])
            if float(np.linalg.norm(path[-1] - candidate[:2])) > 1e-9:
                path = np.vstack([path, candidate[:2]])
            geodesic = float(geodesic_value)
            if not math.isfinite(geodesic) or geodesic < 0.0:
                continue
            path_candidates.append(
                {
                    "station": candidate,
                    "path": path,
                    "geodesic_distance_m": geodesic,
                }
            )
        if not path_candidates:
            return {
                "ok": False,
                "stop_reason": "navigation_unreachable",
                "metrics": {
                    "candidate_count": len(candidates),
                    "navigation_path": {
                        "source": "official_robot_eroded_traversability",
                        "entire_path_requested": True,
                        "full_path_used": False,
                        "dynamic_world_collision_admission": False,
                    },
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }
        path_candidates.sort(
            key=lambda item: (
                float(item["geodesic_distance_m"]),
                float(np.linalg.norm(item["station"][:2] - current[:2])),
            )
        )
        selected = path_candidates[0]
        bounded_path, planned_travel, full_length, truncated = _bounded_polyline_prefix(
            selected["path"],
            max_travel_m=max_travel,
        )

        start_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if start_q.size < 1 or not np.isfinite(start_q).all():
            raise RuntimeError("R1Pro joint feedback is unavailable for navigation")
        base_indices = _indices(getattr(robot, "base_idx", []))
        if len(base_indices) != 6:
            raise RuntimeError("R1Pro six-axis virtual base indices are unavailable")
        q_waypoints = [start_q.copy()]
        prior_yaw = float(start_q[base_indices[5]])
        for point in bounded_path[1:]:
            toward_target = target[:2] - point
            desired_yaw = (
                prior_yaw
                if float(np.linalg.norm(toward_target)) <= 1e-9
                else math.atan2(float(toward_target[1]), float(toward_target[0]))
            )
            desired_yaw = prior_yaw + _wrap_angle(desired_yaw - prior_yaw)
            q = start_q.copy()
            q[base_indices[0]] = float(point[0])
            q[base_indices[1]] = float(point[1])
            q[base_indices[5]] = desired_yaw
            q_waypoints.append(q)
            prior_yaw = desired_yaw
        if len(q_waypoints) == 1:
            q = start_q.copy()
            desired_yaw = math.atan2(
                float(target[1] - current[1]),
                float(target[0] - current[0]),
            )
            q[base_indices[5]] = prior_yaw + _wrap_angle(desired_yaw - prior_yaw)
            q_waypoints.append(q)
        q_path = _interpolate_joint_trajectory(
            np.asarray(q_waypoints, dtype=np.float64),
            max_inter_dist=0.01,
        )[1:].astype(np.float32)
        if q_path.ndim != 2 or len(q_path) < 1 or not np.isfinite(q_path).all():
            raise RuntimeError("navigation q trajectory is unavailable")
        base_goal = np.asarray(
            [
                q_path[-1, base_indices[0]],
                q_path[-1, base_indices[1]],
                _wrap_angle(float(q_path[-1, base_indices[5]])),
            ],
            dtype=np.float64,
        )
        try:
            (
                q_path,
                collision_metrics,
                expected_attachments_by_hand,
            ) = self._certify_base_trajectory(
                q_path,
                start_q=start_q,
                base_goal_xyyaw=base_goal,
                skip_obstacle_update=False,
            )
        except _WholeBodyCertificationError as exc:
            return {
                "ok": False,
                "stop_reason": "collision_admission_failed",
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidate_count": len(candidates),
                    "reachable_candidate_count": len(path_candidates),
                    "env_actions_sent": 0,
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }
        metrics = {
            **collision_metrics,
            "candidate_count": len(candidates),
            "reachable_candidate_count": len(path_candidates),
            "selected_geodesic_distance_m": float(selected["geodesic_distance_m"]),
            "base_goal": base_goal.tolist(),
            "navigation_path": {
                "source": "official_robot_eroded_traversability",
                "entire_path_requested": True,
                "full_path_used": not truncated,
                "dynamic_world_collision_admission": True,
                "returned_waypoint_count": int(len(selected["path"])),
                "bounded_waypoint_count": int(len(bounded_path)),
                "execution_waypoint_count": int(len(q_path)),
                "full_path_length_m": full_length,
                "bounded_stage": {
                    "max_travel_m": max_travel,
                    "planned_travel_m": planned_travel,
                    "truncated": bool(truncated),
                },
            },
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        self._record_base_phase(
            {
                "phase": "navigation_path_selected",
                "metrics": metrics,
            }
        )
        return {
            "ok": True,
            "joint_trajectory": q_path,
            "base_goal": base_goal,
            "expected_attachments_by_hand": expected_attachments_by_hand,
            "metrics": metrics,
        }

    def _certify_base_trajectory(
        self,
        joint_trajectory: Any,
        *,
        start_q: Any,
        base_goal_xyyaw: Any,
        skip_obstacle_update: bool,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """Independently certify a full-q BASE path before any env action.

        Traversability is a useful static-map pre-filter, but it cannot replace
        cuRobo's live world, self-collision, and attachment checks.  The
        certificate is bound to the exact execution trajectory and call-start
        robot state.
        """

        robot = self._find_robot()
        generator = self._generator(kind="base")
        q_start = np.asarray(_jsonable(start_q), dtype=np.float32).reshape(-1)
        q_path = np.asarray(_jsonable(joint_trajectory), dtype=np.float32)
        base_goal = _canonical_base_xyyaw(base_goal_xyyaw)
        if (
            q_start.size < 1
            or q_path.ndim != 2
            or q_path.shape[1] != q_start.size
            or len(q_path) < 1
            or not np.isfinite(q_start).all()
            or not np.isfinite(q_path).all()
            or base_goal.shape != (3,)
            or not np.isfinite(base_goal).all()
        ):
            raise RuntimeError("BASE trajectory certification received invalid q")
        base_idx = _indices(getattr(robot, "base_idx", []))
        if len(base_idx) != 6:
            raise RuntimeError("R1Pro six-axis virtual base indices are unavailable")
        movable = {base_idx[0], base_idx[1], base_idx[5]}
        locked = [index for index in range(q_start.size) if index not in movable]
        if locked and not np.allclose(
            q_path[:, locked],
            q_start[locked],
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError("BASE trajectory changed a locked non-base joint")
        terminal_base = np.asarray(
            [
                q_path[-1, base_idx[0]],
                q_path[-1, base_idx[1]],
                q_path[-1, base_idx[5]],
            ],
            dtype=np.float64,
        )
        if (
            not np.allclose(
                terminal_base[:2],
                base_goal[:2],
                atol=1e-6,
                rtol=0.0,
            )
            or abs(_wrap_angle(float(terminal_base[2] - base_goal[2])))
            > 1e-6
        ):
            raise RuntimeError(
                "BASE trajectory terminal q does not match its canonical goal"
            )

        with_start = (
            q_path
            if np.allclose(q_path[0], q_start, atol=1e-7, rtol=0.0)
            else np.vstack([q_start.reshape(1, -1), q_path])
        )
        dense = _interpolate_joint_trajectory(
            with_start,
            max_inter_dist=WHOLE_BODY_DENSE_COLLISION_STEP,
        )
        attachments, attachments_by_hand = self._all_attached_objects(
            selected_hand="left"
        )
        attachment_scales = (
            {str(link): 1.0 for link in attachments}
            if isinstance(attachments, dict)
            else None
        )
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        try:
            collision_flags = generator.check_collisions(
                torch.as_tensor(dense, dtype=torch.float32),
                initial_joint_pos=torch.as_tensor(q_start, dtype=torch.float32),
                self_collision_check=True,
                skip_obstacle_update=bool(skip_obstacle_update),
                attached_obj=attachments,
                attached_obj_scale=attachment_scales,
            )
        except Exception as exc:
            raise _WholeBodyCertificationError(
                "BASE full-trajectory collision certification failed"
            ) from exc
        collision_array = np.asarray(
            _jsonable(collision_flags), dtype=bool
        ).reshape(-1)
        if collision_array.shape != (len(dense),):
            raise _WholeBodyCertificationError(
                "BASE collision checker returned an invalid waypoint count"
            )
        colliding = int(np.count_nonzero(collision_array))
        certificate = {
            "schema_version": 1,
            "trajectory_sha256": hashlib.sha256(
                np.ascontiguousarray(q_path, dtype=np.float32).tobytes()
            ).hexdigest(),
            "start_q_sha256": hashlib.sha256(
                np.ascontiguousarray(q_start, dtype=np.float32).tobytes()
            ).hexdigest(),
            "waypoint_count": int(len(q_path)),
            "dense_collision_waypoint_count": int(len(dense)),
            "world_collision_check": True,
            "self_collision_check": True,
            "post_interpolation_check": True,
            "attachment_hand_count": 2,
            "colliding_waypoint_count": colliding,
            "base_goal_xyyaw_sha256": _whole_body_target_sha256(base_goal),
            "terminal_q_sha256": hashlib.sha256(
                np.ascontiguousarray(q_path[-1], dtype=np.float32).tobytes()
            ).hexdigest(),
            "terminal_command_limit": TERMINAL_COMMAND_LIMIT,
            "terminal_position_tolerance_m": BASE_TERMINAL_POSITION_TOLERANCE_M,
            "terminal_orientation_tolerance_rad": (
                BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD
            ),
        }
        metrics = {
            "collision_admission": {
                "available": True,
                "admitted": colliding == 0,
                "world_collision_check": True,
                "self_collision_check": True,
                "obstacle_update": not bool(skip_obstacle_update),
                "full_trajectory": True,
                "post_interpolation_check": True,
                "colliding_waypoint_count": colliding,
                "dense_collision_waypoint_count": int(len(dense)),
                "source": "CuRoboMotionGenerator.check_collisions",
            },
            "base_trajectory_certificate": certificate,
            "attachments_by_hand": {
                side: {"available": attachments_by_hand[side] is not None}
                for side in ("left", "right")
            },
        }
        obstacle_refresh = RealCuroboBackend._obstacle_refresh_metrics(generator)
        if obstacle_refresh is not None:
            metrics["obstacle_refresh"] = obstacle_refresh
            metrics["world_refresh_count"] = (
                0 if bool(skip_obstacle_update) else 1
            )
            metrics["world_refresh_s"] = float(
                obstacle_refresh.get("elapsed_s", 0.0)
            )
        if colliding:
            raise _WholeBodyCertificationError(
                f"BASE dense trajectory contains {colliding} colliding waypoints"
            )
        return q_path, metrics, attachments_by_hand

    def plan_relative_navigation_trajectory(
        self,
        *,
        relative_motion: Any,
        timeout_s: float,
        start_q: Any | None = None,
        start_base_xyyaw: Any | None = None,
        background: bool = False,
        base_attempt_timeout_cap_s: float | None = None,
        base_solver_timeout_cap_s: float | None = None,
        base_planning_profile: str | None = None,
    ) -> dict[str, Any]:
        """Plan one exact straight translation or in-place BASE rotation."""

        started = time.monotonic()
        motion = validate_relative_navigation_motion(relative_motion)
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        robot = self._find_robot()
        scene = self._scene(robot)
        trav_map = getattr(scene, "trav_map", None)
        floor_maps = getattr(trav_map, "floor_map", None)
        erode = getattr(trav_map, "_erode_trav_map", None)
        world_to_map = getattr(trav_map, "world_to_map", None)
        if floor_maps is None or not callable(erode) or not callable(world_to_map):
            return {
                "ok": False,
                "stop_reason": "relative_navigation_traversability_unavailable",
                "metrics": {"elapsed_s": round(time.monotonic() - started, 3)},
            }

        current = np.asarray(
            self._base_xy_yaw(robot)
            if start_base_xyyaw is None
            else _jsonable(start_base_xyyaw),
            dtype=np.float64,
        ).reshape(-1)
        if current.shape not in {(3,), (4,)} or not np.isfinite(current).all():
            raise ValueError("relative navigation start BASE pose is invalid")
        floor = self._current_floor(scene, current)
        if floor < 0 or floor >= len(floor_maps):
            return {
                "ok": False,
                "stop_reason": "relative_navigation_traversability_unavailable",
                "metrics": {"elapsed_s": round(time.monotonic() - started, 3)},
            }
        try:
            source_map = floor_maps[floor]
            try:
                import torch
            except ImportError:
                source_tensor = np.asarray(_jsonable(source_map)).copy()
            else:
                source_tensor = (
                    torch.clone(source_map)
                    if torch.is_tensor(source_map)
                    else torch.as_tensor(np.asarray(_jsonable(source_map))).clone()
                )
            eroded_map = np.asarray(
                _jsonable(erode(source_tensor, robot=robot)),
                dtype=np.float64,
            )
        except Exception as exc:
            return {
                "ok": False,
                "stop_reason": "relative_navigation_traversability_unavailable",
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }
        if (
            eroded_map.ndim != 2
            or eroded_map.size < 1
            or not np.isfinite(eroded_map).all()
        ):
            return {
                "ok": False,
                "stop_reason": "relative_navigation_traversability_unavailable",
                "metrics": {"elapsed_s": round(time.monotonic() - started, 3)},
            }

        start_q_array = np.asarray(
            _jsonable(
                robot.get_joint_positions()
                if start_q is None
                else start_q
            ),
            dtype=np.float64,
        ).reshape(-1)
        base_indices = _indices(getattr(robot, "base_idx", []))
        if (
            start_q_array.size < 1
            or not np.isfinite(start_q_array).all()
            or len(base_indices) != 6
        ):
            raise RuntimeError("R1Pro base joint feedback is unavailable")
        q_pose = np.asarray(
            [
                start_q_array[base_indices[0]],
                start_q_array[base_indices[1]],
                start_q_array[base_indices[5]],
            ],
            dtype=np.float64,
        )
        if (
            float(np.linalg.norm(q_pose[:2] - current[:2])) > 0.02 + 1e-9
            or abs(_wrap_angle(float(q_pose[2] - current[2])))
            > math.radians(1.0) + 1e-9
        ):
            return {
                "ok": False,
                "stop_reason": "relative_navigation_pose_inconsistent",
                "metrics": {"elapsed_s": round(time.monotonic() - started, 3)},
            }

        base_goal = q_pose.copy()
        if motion["kind"] == "translation":
            signed_distance = float(motion["distance_m"]) * (
                1.0 if motion["direction"] == "forward" else -1.0
            )
            heading = np.asarray(
                [math.cos(float(current[2])), math.sin(float(current[2]))],
                dtype=np.float64,
            )
            base_goal[:2] = current[:2] + signed_distance * heading
            start_cell = np.asarray(
                _jsonable(world_to_map(current[:2])), dtype=np.int64
            ).reshape(-1)
            end_cell = np.asarray(
                _jsonable(world_to_map(base_goal[:2])), dtype=np.int64
            ).reshape(-1)
            if start_cell.shape != (2,) or end_cell.shape != (2,):
                return {
                    "ok": False,
                    "stop_reason": "relative_navigation_traversability_unavailable",
                    "metrics": {"elapsed_s": round(time.monotonic() - started, 3)},
                }
            checked_cells = _supercover_grid_cells(start_cell, end_cell)
        else:
            signed_angle = math.radians(float(motion["angle_deg"])) * (
                1.0 if motion["direction"] == "left" else -1.0
            )
            base_goal[2] = float(q_pose[2]) + signed_angle
            start_cell = np.asarray(
                _jsonable(world_to_map(current[:2])), dtype=np.int64
            ).reshape(-1)
            if start_cell.shape != (2,):
                return {
                    "ok": False,
                    "stop_reason": "relative_navigation_traversability_unavailable",
                    "metrics": {"elapsed_s": round(time.monotonic() - started, 3)},
                }
            checked_cells = [(int(start_cell[0]), int(start_cell[1]))]

        base_goal = _canonical_base_xyyaw(base_goal)
        if any(
            row < 0
            or column < 0
            or row >= eroded_map.shape[0]
            or column >= eroded_map.shape[1]
            or eroded_map[row, column] <= 0.0
            for row, column in checked_cells
        ):
            return {
                "ok": False,
                "stop_reason": "relative_navigation_untraversable",
                "metrics": {
                    "checked_cell_count": len(checked_cells),
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }

        remaining_s = timeout - (time.monotonic() - started)
        if remaining_s <= 0.0:
            return {
                "ok": False,
                "stop_reason": "timeout",
                "metrics": {
                    "env_actions_sent": 0,
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }
        base_plan_kwargs: dict[str, Any] = {
            "target_xyyaw": base_goal,
            "timeout_s": remaining_s,
            "skip_obstacle_update": False,
            "start_q": start_q_array,
            "background": background,
        }
        if base_attempt_timeout_cap_s is not None:
            base_plan_kwargs["attempt_timeout_cap_s"] = float(
                base_attempt_timeout_cap_s
            )
        if base_solver_timeout_cap_s is not None:
            base_plan_kwargs["solver_timeout_cap_s"] = float(
                base_solver_timeout_cap_s
            )
        if base_planning_profile is not None:
            base_plan_kwargs["planning_profile"] = str(
                base_planning_profile
            )
        exact_plan = self._compute_base_plan(**base_plan_kwargs)
        if exact_plan.get("ok") is not True:
            return {
                "ok": False,
                "stop_reason": str(
                    exact_plan.get("stop_reason", "relative_navigation_unreachable")
                ),
                "metrics": {
                    **dict(exact_plan.get("metrics") or {}),
                    "relative_motion": dict(motion),
                    "checked_cell_count": len(checked_cells),
                    "env_actions_sent": 0,
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            }
        q_path = np.asarray(
            _jsonable(exact_plan.get("joint_trajectory")), dtype=np.float32
        )
        if (
            q_path.ndim != 2
            or q_path.shape[1] != start_q_array.size
            or len(q_path) < 1
            or not np.isfinite(q_path).all()
        ):
            raise RuntimeError("exact relative BASE planner returned invalid q")
        exact_metrics = dict(exact_plan.get("metrics") or {})
        base_solver_deadline = dict(
            exact_metrics.get("deadline_enforcement") or {}
        )
        metrics = {
            **exact_metrics,
            "relative_motion": dict(motion),
            "base_goal": [
                float(base_goal[0]),
                float(base_goal[1]),
                float(base_goal[2]),
            ],
            "navigation_path": {
                "source": "official_robot_eroded_traversability",
                "straight_relative_motion": True,
                "dynamic_world_collision_admission": True,
                "checked_cell_count": len(checked_cells),
                "execution_waypoint_count": int(len(q_path)),
            },
            "elapsed_s": round(time.monotonic() - started, 3),
            "base_solver_deadline_enforcement": base_solver_deadline,
            "deadline_enforcement": {
                "solver_timeout_enforced": True,
                "hard_wall_clock_enforced": not bool(background),
                "soft_deadline_s": timeout if background else None,
            },
        }
        self._record_base_phase(
            {
                "phase": "relative_navigation_admitted",
                "metrics": metrics,
            }
        )
        return {
            "ok": True,
            "joint_trajectory": q_path.astype(np.float32),
            "base_goal": np.asarray(
                [base_goal[0], base_goal[1], base_goal[2]],
                dtype=np.float64,
            ),
            "expected_attachments_by_hand": exact_plan.get(
                "expected_attachments_by_hand"
            ),
            "metrics": metrics,
        }

    def _ranked_base_candidates(
        self,
        robot: Any,
        *,
        hand: str,
        target_xyz: np.ndarray,
        standoff_m: float,
        deadline: float,
    ) -> list[dict[str, Any]]:
        scene = self._scene(robot)
        trav_map = getattr(scene, "trav_map", None)
        if trav_map is None:
            raise RuntimeError("BEHAVIOR scene has no traversability map")
        current = self._base_xy_yaw(robot)
        floor = self._current_floor(scene, current)
        generated = _base_candidates(target_xyz, standoff_m=standoff_m)
        pixel_traversable = [
            candidate
            for candidate in generated
            if self._candidate_is_traversable(trav_map, candidate, floor=floor)
        ]
        pixel_traversable.sort(
            key=lambda candidate: (
                float(np.linalg.norm(candidate[:2] - current[:2])),
                abs(_wrap_angle(candidate[2] - current[2])),
            )
        )
        shortlisted = pixel_traversable[:MAX_BASE_STATION_SHORTLIST]
        reachability_attempts = 0
        self._last_base_candidate_summary = {
            "generated_count": len(generated),
            "pixel_traversable_count": len(pixel_traversable),
            "traversable_count": 0,
            "reachability_checked_count": 0,
            "reachable_count": 0,
            "shortlisted_count": len(shortlisted),
            "candidate_batch_size": len(shortlisted),
            "candidate_limit": MAX_BASE_PLAN_CANDIDATES,
        }
        self._record_base_phase(
            {
                "phase": "candidate_search_started",
                "summary": dict(self._last_base_candidate_summary),
                "remaining_s": float(deadline) - time.monotonic(),
            }
        )
        path_started = time.monotonic()
        connected, traversability_method = self._connected_base_candidates(
            trav_map,
            robot=robot,
            floor=floor,
            current_xy=current[:2],
            candidates=shortlisted,
            deadline=deadline,
        )
        connected.sort(
            key=lambda item: (
                float(
                    item.get("geodesic_distance_m")
                    if item.get("geodesic_distance_m") is not None
                    else item.get("path_rank_distance_m", float("inf"))
                ),
                float(np.linalg.norm(item["xyyaw"][:2] - current[:2])),
                abs(_wrap_angle(float(item["xyyaw"][2] - current[2]))),
            )
        )
        self._last_base_candidate_summary["traversable_count"] = len(connected)
        self._record_base_phase(
            {
                "phase": "traversability_complete",
                "elapsed_s": round(time.monotonic() - path_started, 3),
                "method": traversability_method,
                "connected": [item["xyyaw"].tolist() for item in connected],
                "remaining_s": float(deadline) - time.monotonic(),
            }
        )
        ranked: list[dict[str, Any]] = []
        for item in connected:
            if len(ranked) >= MAX_BASE_PLAN_CANDIDATES:
                break
            remaining_s = float(deadline) - time.monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError("BASE candidate reachability deadline exceeded")
            reachability_target = self._candidate_reachability_target(
                target_xyz,
                item["xyyaw"],
            )
            reach_started = time.monotonic()
            reachable, reason, reach_metrics = self.check_candidate_arm_reachability(
                hand=hand,
                target_xyz=reachability_target,
                base_xyyaw=item["xyyaw"],
                timeout_s=min(12.0, remaining_s),
                skip_obstacle_update=reachability_attempts > 0,
            )
            reachability_attempts += 1
            self._record_base_phase(
                {
                    "phase": "candidate_arm_reachability",
                    "attempt": reachability_attempts,
                    "candidate_xyyaw": item["xyyaw"].tolist(),
                    "reachable": bool(reachable),
                    "reason": reason,
                    "metrics": reach_metrics,
                    "elapsed_s": round(time.monotonic() - reach_started, 3),
                    "remaining_s": float(deadline) - time.monotonic(),
                }
            )
            self._last_base_candidate_summary["reachability_checked_count"] = (
                reachability_attempts
            )
            if not reachable:
                continue
            ranked.append(
                {
                    **item,
                    "reachability_target_xyz": reachability_target.tolist(),
                    "reachability_reason": reason,
                    "reachability": reach_metrics,
                    "reachability_stage": reach_metrics.get("reachability_stage"),
                }
            )
            self._last_base_candidate_summary["reachable_count"] = len(ranked)
        self._record_base_phase(
            {
                "phase": "candidate_search_complete",
                "summary": dict(self._last_base_candidate_summary),
                "remaining_s": float(deadline) - time.monotonic(),
            }
        )
        return ranked

    def _connected_base_candidates(
        self,
        trav_map: Any,
        *,
        robot: Any,
        floor: int,
        current_xy: np.ndarray,
        candidates: list[np.ndarray],
        deadline: float,
    ) -> tuple[list[dict[str, Any]], str]:
        """Reuse one official robot-radius erosion for a batch of A* goals."""

        if hasattr(trav_map, "_erode_trav_map") and hasattr(trav_map, "floor_map"):
            import cv2
            import torch

            eroded = trav_map._erode_trav_map(
                torch.clone(trav_map.floor_map[floor]), robot=robot
            )
            _count, labels = cv2.connectedComponents(
                np.asarray(_jsonable(eroded), dtype=np.uint8), connectivity=4
            )
            source_map = np.asarray(
                _jsonable(trav_map.world_to_map(current_xy)), dtype=np.int64
            ).reshape(2)
            source_label = int(labels[int(source_map[0]), int(source_map[1])])
            if source_label == 0:
                raise RuntimeError(
                    "current BASE pose is outside the robot-eroded traversability map"
                )
            connected: list[dict[str, Any]] = []
            for candidate in candidates:
                if float(deadline) - time.monotonic() <= 0.0:
                    raise TimeoutError("BASE candidate reachability deadline exceeded")
                target_map = np.asarray(
                    _jsonable(trav_map.world_to_map(candidate[:2])),
                    dtype=np.int64,
                ).reshape(2)
                if int(labels[int(target_map[0]), int(target_map[1])]) != source_label:
                    continue
                connected.append(
                    {
                        "xyyaw": candidate,
                        "geodesic_distance_m": None,
                        "path_rank_distance_m": float(
                            np.linalg.norm(candidate[:2] - current_xy)
                        ),
                        "traversability_component_id": source_label,
                    }
                )
            return connected, "official_eroded_map_connected_component"

        connected = []
        for candidate in candidates:
            if float(deadline) - time.monotonic() <= 0.0:
                raise TimeoutError("BASE candidate reachability deadline exceeded")
            path, distance = trav_map.get_shortest_path(
                floor,
                current_xy,
                candidate[:2],
                entire_path=True,
                robot=robot,
            )
            if path is not None and distance is not None:
                connected.append(
                    {
                        "xyyaw": candidate,
                        "geodesic_distance_m": float(distance),
                    }
                )
        return connected, "official_get_shortest_path_per_candidate"

    @staticmethod
    def _candidate_reachability_target(
        surface_target_xyz: np.ndarray,
        candidate_xyyaw: np.ndarray,
        *,
        clearance_m: float = 0.15,
    ) -> np.ndarray:
        """Return a precontact target between base and surface."""

        target = np.asarray(surface_target_xyz, dtype=np.float64).reshape(3)
        toward_surface = target[:2] - np.asarray(candidate_xyyaw, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(toward_surface))
        if norm <= 1e-9:
            return target.copy()
        result = target.copy()
        result[:2] -= toward_surface / norm * float(clearance_m)
        return result

    def _scene(self, robot: Any) -> Any:
        scene = getattr(robot, "scene", None)
        if scene is not None:
            return scene
        candidates = [
            self.env_facade,
            getattr(self.env_facade, "_env", None),
            getattr(getattr(self.env_facade, "_env", None), "_direct_process", None),
            getattr(
                getattr(
                    getattr(self.env_facade, "_env", None), "_direct_process", None
                ),
                "env",
                None,
            ),
        ]
        for candidate in candidates:
            scene = getattr(candidate, "scene", None)
            if scene is not None:
                return scene
        raise RuntimeError("could not locate BEHAVIOR scene for traversability map")

    def _current_floor(self, scene: Any, current_xyyaw: np.ndarray) -> int:
        floor_heights = getattr(getattr(scene, "trav_map", None), "floor_heights", None)
        if floor_heights is None:
            return 0
        heights = np.asarray(_jsonable(floor_heights), dtype=np.float64).reshape(-1)
        if heights.size == 0:
            return 0
        z = float(current_xyyaw[3]) if current_xyyaw.shape[0] > 3 else 0.0
        return int(np.argmin(np.abs(heights - z)))

    def _candidate_is_traversable(
        self,
        trav_map: Any,
        candidate: np.ndarray,
        *,
        floor: int,
    ) -> bool:
        try:
            map_xy = np.asarray(
                trav_map.world_to_map(candidate[:2]), dtype=np.int64
            ).reshape(2)
            floor_maps = getattr(trav_map, "floor_map", None)
            if floor_maps is None or floor < 0 or floor >= len(floor_maps):
                return False
            trav = floor_maps[floor]
            array = np.asarray(_jsonable(trav))
            row, column = int(map_xy[0]), int(map_xy[1])
            if (
                array.ndim != 2
                or row < 0
                or column < 0
                or row >= array.shape[0]
                or column >= array.shape[1]
            ):
                return False
            return bool(array[row, column])
        except Exception:
            return False

    def _compute_base_plan(
        self,
        *,
        target_xyyaw: np.ndarray,
        timeout_s: float,
        skip_obstacle_update: bool = False,
        start_q: Any | None = None,
        background: bool = False,
        attempt_timeout_cap_s: float = BASE_PLAN_ATTEMPT_TIMEOUT_S,
        solver_timeout_cap_s: float | None = None,
        planning_profile: str = "default",
        wall_clock_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        target_xyyaw = _canonical_base_xyyaw(target_xyyaw)
        analytic_dashboard_fast = (
            planning_profile == DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
        )
        generator = self._generator(kind="base")
        prior_obstacle_refresh = (
            RealCuroboBackend._obstacle_refresh_metrics(generator) or {}
        )
        prior_obstacle_refresh_count = int(
            prior_obstacle_refresh.get("count", 0)
        )
        emb_sel = self._embodiment_cls.BASE
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        robot = self._find_robot()
        current_q = np.asarray(
            _jsonable(
                robot.get_joint_positions()
                if start_q is None
                else start_q
            ),
            dtype=np.float32,
        ).reshape(1, -1)
        pos, quat = self._base_target_world_pose(robot, target_xyyaw)
        batch_size = max(int(generator.batch_size), 1)
        planner_targets = (
            torch.as_tensor(pos, dtype=torch.float32)
            .reshape(1, 3)
            .repeat(
                batch_size,
                1,
            )
        )
        planner_quats = (
            torch.as_tensor(quat, dtype=torch.float32)
            .reshape(1, 4)
            .repeat(
                batch_size,
                1,
            )
        )
        attempt_timeout_cap = float(attempt_timeout_cap_s)
        if not math.isfinite(attempt_timeout_cap) or attempt_timeout_cap <= 0.0:
            raise ValueError(
                "BASE attempt timeout cap must be finite and positive"
            )
        attempt_timeout_budget_s = min(float(timeout_s), attempt_timeout_cap)
        if solver_timeout_cap_s is None:
            solver_timeout_cap = max(
                0.1,
                BASE_PLAN_ATTEMPT_TIMEOUT_S - 2.0,
            )
            attempt_timeout_s = max(0.1, attempt_timeout_budget_s - 2.0)
        else:
            solver_timeout_cap = float(solver_timeout_cap_s)
            if (
                not math.isfinite(solver_timeout_cap)
                or solver_timeout_cap <= 0.0
            ):
                raise ValueError(
                    "BASE solver timeout cap must be finite and positive"
                )
            attempt_timeout_s = min(
                attempt_timeout_budget_s,
                solver_timeout_cap,
            )
        wall_clock_timeout = (
            attempt_timeout_budget_s
            if wall_clock_timeout_s is None
            else float(wall_clock_timeout_s)
        )
        if (
            not math.isfinite(wall_clock_timeout)
            or wall_clock_timeout <= 0.0
            or wall_clock_timeout + 1e-12 < attempt_timeout_s
        ):
            raise ValueError(
                "BASE wall-clock timeout must be finite, positive, and "
                "at least the solver timeout"
            )
        deadline_scope = (
            nullcontext()
            if background
            else _wall_clock_deadline(
                wall_clock_timeout,
                "BASE cuRobo candidate",
            )
        )
        with deadline_scope:
            if analytic_dashboard_fast:
                successes, paths = np.zeros((0,), dtype=bool), []
            else:
                successes, paths = generator.compute_trajectories(
                    planner_targets,
                    planner_quats,
                    initial_joint_pos=torch.as_tensor(
                        current_q[0], dtype=torch.float32
                    ),
                    max_attempts=5,
                    timeout=attempt_timeout_s,
                    ik_fail_return=5,
                    enable_finetune_trajopt=False,
                    finetune_attempts=0,
                    return_full_result=False,
                    success_ratio=1.0 / batch_size,
                    ik_only=True,
                    skip_obstacle_update=bool(skip_obstacle_update),
                    ik_world_collision_check=True,
                    emb_sel=emb_sel,
                )
        success_array = np.asarray(_jsonable(successes), dtype=bool).reshape(-1)
        success_indices = np.flatnonzero(success_array)
        metrics = {
            "successes": success_array.tolist(),
            "ik_only": not analytic_dashboard_fast,
            "solver_invoked": not analytic_dashboard_fast,
            "curobo_config": str(self._base_config_path()),
            "curobo_api": (
                "CuRoboMotionGenerator.check_collisions"
                if analytic_dashboard_fast
                else "CuRoboMotionGenerator.compute_trajectories"
            ),
            "success_ratio": 1.0 / batch_size,
            "planner_seed_count": 0 if analytic_dashboard_fast else batch_size,
            "max_attempts": 0 if analytic_dashboard_fast else 5,
            "ik_fail_return": 0 if analytic_dashboard_fast else 5,
            "attempt_timeout_s": attempt_timeout_s,
            "solver_timeout_s": attempt_timeout_s,
            "attempt_timeout_budget_s": attempt_timeout_budget_s,
            "attempt_timeout_cap_s": attempt_timeout_cap,
            "solver_timeout_cap_s": solver_timeout_cap,
            "planning_profile": str(planning_profile),
            "base_prismatic_workspace_limit_m": self._base_workspace_limit_m,
            "base_prismatic_workspace_limit_source": "scene_envelope_plus_2m",
            "collision_admission_enabled": True,
            "obstacle_update": not bool(skip_obstacle_update),
            "deadline_enforcement": {
                "solver_timeout_enforced": True,
                "hard_wall_clock_enforced": not bool(background),
                "hard_wall_clock_deadline_s": (
                    wall_clock_timeout if not background else None
                ),
                "soft_deadline_s": (
                    wall_clock_timeout if background else None
                ),
            },
        }
        obstacle_refresh = RealCuroboBackend._obstacle_refresh_metrics(generator)
        if obstacle_refresh is not None:
            metrics["obstacle_refresh"] = obstacle_refresh
            metrics["world_refresh_count"] = max(
                0,
                int(obstacle_refresh.get("count", 0))
                - prior_obstacle_refresh_count,
            )
            metrics["world_refresh_s"] = float(
                obstacle_refresh.get("elapsed_s", 0.0)
            )
        base_indices = _indices(getattr(robot, "base_idx", []))
        if len(base_indices) != 6:
            raise RuntimeError("R1Pro six-axis virtual base indices are unavailable")
        if analytic_dashboard_fast:
            q_goal = current_q.copy()
            q_goal[0, base_indices[0]] = float(target_xyyaw[0])
            q_goal[0, base_indices[1]] = float(target_xyyaw[1])
            q_goal[0, base_indices[5]] = float(current_q[0, base_indices[5]]) + (
                _wrap_angle(
                    float(target_xyyaw[2])
                    - float(current_q[0, base_indices[5]])
                )
            )
            metrics["base_goal_construction"] = {
                "method": "analytic_full_q_base_assignment",
                "active_base_indices": [
                    int(base_indices[0]),
                    int(base_indices[1]),
                    int(base_indices[5]),
                ],
                "nonbase_locked_to_call_start": True,
                "continuous_shortest_yaw_arc": True,
            }
        else:
            if success_indices.size == 0:
                return {
                    "ok": False,
                    "stop_reason": "base_plan_failed",
                    "metrics": metrics,
                }
            path = paths[int(success_indices[0])]
            q_goal, merge_report = self._merge_base_ik_solution_into_full_q(
                generator,
                robot,
                path,
                call_start_q=current_q[0],
            )
            metrics["base_ik_solution_merge"] = merge_report
        if q_goal.ndim != 2 or q_goal.shape[1] != current_q.shape[1]:
            raise RuntimeError(
                "BASE IK goal does not match the live robot joint layout: "
                f"goal={q_goal.shape} current={current_q.shape}"
            )
        execution_with_start, resampling_metrics = (
            _minimum_jerk_base_execution_trajectory(
                np.vstack([current_q, q_goal[-1:]]),
                base_indices=base_indices,
                max_xy_step_m=(
                    DASHBOARD_BASE_EXECUTION_XY_STEP_M
                    if analytic_dashboard_fast
                    else BASE_EXECUTION_XY_STEP_M
                ),
                max_yaw_step_rad=(
                    DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD
                    if analytic_dashboard_fast
                    else BASE_EXECUTION_YAW_STEP_RAD
                ),
            )
        )
        execution_q_traj = execution_with_start[1:]
        try:
            (
                execution_q_traj,
                collision_metrics,
                expected_attachments_by_hand,
            ) = self._certify_base_trajectory(
                execution_q_traj,
                start_q=current_q[0],
                base_goal_xyyaw=target_xyyaw,
                # compute_trajectories above already refreshed the world when
                # this call owns obstacle admission.
                skip_obstacle_update=(
                    bool(skip_obstacle_update)
                    if analytic_dashboard_fast
                    else True
                ),
            )
        except _WholeBodyCertificationError as exc:
            return {
                "ok": False,
                "stop_reason": "collision_admission_failed",
                "metrics": {
                    **metrics,
                    "error": f"{type(exc).__name__}: {exc}",
                    "env_actions_sent": 0,
                },
            }
        metrics.update(
            {
                **collision_metrics,
                "trajectory_waypoints": int(len(execution_q_traj)),
                "execution_waypoints": int(len(execution_q_traj)),
                "execution_resampling": {
                    **resampling_metrics,
                    "source": (
                        "analytic_full_q_minimum_jerk_execution_resampling"
                        if analytic_dashboard_fast
                        else "base_ik_minimum_jerk_execution_resampling"
                    ),
                    "controller": "official_position_holonomic_base",
                    "planner_base_isaac_kp": 2_000_000.0,
                    "planner_base_isaac_kd": 100_000.0,
                },
            }
        )
        obstacle_refresh = RealCuroboBackend._obstacle_refresh_metrics(generator)
        if obstacle_refresh is not None:
            metrics["obstacle_refresh"] = obstacle_refresh
            metrics["world_refresh_count"] = max(
                0,
                int(obstacle_refresh.get("count", 0))
                - prior_obstacle_refresh_count,
            )
            metrics["world_refresh_s"] = float(
                obstacle_refresh.get("elapsed_s", 0.0)
            )
        return {
            "ok": True,
            "joint_trajectory": execution_q_traj,
            "base_goal": target_xyyaw.copy(),
            "expected_attachments_by_hand": expected_attachments_by_hand,
            "metrics": metrics,
        }

    def _base_target_world_pose(
        self,
        robot: Any,
        target_xyyaw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        base_idx = _indices(getattr(robot, "base_idx", []))
        if len(base_idx) < 6:
            raise RuntimeError("R1Pro six-axis virtual base joint indices unavailable")
        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        pos = np.array(
            [target_xyyaw[0], target_xyyaw[1], q[base_idx[2]]],
            dtype=np.float64,
        )
        quat = _intrinsic_rpy_to_quat_xyzw(
            float(q[base_idx[3]]),
            float(q[base_idx[4]]),
            float(target_xyyaw[2]),
        )
        return pos, quat

    def _initial_joint_pos_for_base_candidate(
        self, robot: Any, base_xyyaw: np.ndarray
    ) -> np.ndarray:
        q = (
            np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
            .reshape(-1)
            .copy()
        )
        base_idx = _indices(getattr(robot, "base_idx", []))
        if len(base_idx) >= 6:
            q[base_idx[0]] = float(base_xyyaw[0])
            q[base_idx[1]] = float(base_xyyaw[1])
            q[base_idx[5]] = float(base_xyyaw[2])
            return q
        control_idx = _indices(getattr(robot, "base_control_idx", []))
        if len(control_idx) >= 3:
            q[control_idx[0]] = float(base_xyyaw[0])
            q[control_idx[1]] = float(base_xyyaw[1])
            q[control_idx[2]] = float(base_xyyaw[2])
            return q
        raise RuntimeError("R1Pro base DOF indices unavailable for candidate IK")

    def _base_xy_yaw(self, robot: Any) -> np.ndarray:
        try:
            link = robot.links[getattr(robot, "base_footprint_link_name", "base_link")]
            pos, quat = link.get_position_orientation()
            pos = np.asarray(_jsonable(pos), dtype=np.float64)
            quat = np.asarray(_jsonable(quat), dtype=np.float64)
            yaw = _yaw_from_quat_xyzw(quat)
            return np.array([pos[0], pos[1], yaw, pos[2]], dtype=np.float64)
        except Exception:
            qpos = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
            idx = _indices(getattr(robot, "base_control_idx", [0, 1, 5]))
            values = qpos[idx[:3]]
            return np.asarray([values[0], values[1], values[2], 0.0], dtype=np.float64)

    def get_base_pose(self) -> np.ndarray:
        return self._base_xy_yaw(self._find_robot())[:3]

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(
            _jsonable(self._find_robot().get_joint_positions()),
            dtype=np.float32,
        ).reshape(-1)

    def joint_tracking_report(
        self, target_q: Any, *, hand: str | None
    ) -> dict[str, Any]:
        """Report closed-loop waypoint tracking against controlled R1Pro joints."""
        robot = self._find_robot()
        target = np.asarray(_jsonable(target_q), dtype=np.float64).reshape(-1)
        current = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if target.shape != current.shape:
            return {
                "available": False,
                "reason": f"joint shape mismatch target={target.shape} current={current.shape}",
                "reached": False,
            }
        if not np.isfinite(target).all() or not np.isfinite(current).all():
            return {
                "available": False,
                "reason": "joint tracking state contains NaN or infinity",
                "reached": False,
            }
        base_idx = _indices(getattr(robot, "base_control_idx", []))
        trunk_idx = _indices(getattr(robot, "trunk_control_idx", []))
        arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
        if hand is None:
            left_idx = _indices(arm_control_idx.get("left", []))
            right_idx = _indices(arm_control_idx.get("right", []))
            active_idx = base_idx[:3] + trunk_idx + left_idx + right_idx
            if (
                len(base_idx) < 3
                or len(trunk_idx) != 4
                or len(left_idx) != 7
                or len(right_idx) != 7
                or len(active_idx) != 21
                or len(set(active_idx)) != 21
            ):
                return {
                    "available": False,
                    "reason": "R1Pro 21-DOF controlled joint indices unavailable",
                    "reached": False,
                }
            articulation_idx = trunk_idx + left_idx + right_idx
        else:
            selected_idx = _indices(arm_control_idx.get(hand, []))
            articulation_idx = trunk_idx + selected_idx
        if len(base_idx) < 3 or not articulation_idx:
            return {
                "available": False,
                "reason": "R1Pro controlled joint indices unavailable",
                "reached": False,
            }
        diff = target - current
        tracking_diff = diff.copy()
        tracking_diff[base_idx[2]] = _wrap_angle(
            float(tracking_diff[base_idx[2]])
        )
        base_xy_error = float(np.max(np.abs(diff[base_idx[:2]])))
        base_yaw_error = abs(float(tracking_diff[base_idx[2]]))
        articulation_error = float(np.max(np.abs(diff[sorted(set(articulation_idx))])))
        if hand is None:
            normalized_components = np.concatenate(
                [
                    diff[base_idx[:2]]
                    / WHOLE_BODY_BASE_XY_WAYPOINT_TOLERANCE_M,
                    np.asarray(
                        [
                            float(tracking_diff[base_idx[2]])
                            / WHOLE_BODY_BASE_YAW_WAYPOINT_TOLERANCE_RAD
                        ],
                        dtype=np.float64,
                    ),
                    diff[articulation_idx]
                    / WHOLE_BODY_ARTICULATION_WAYPOINT_TOLERANCE_RAD,
                ]
            )
            reached = bool(
                base_xy_error <= WHOLE_BODY_BASE_XY_WAYPOINT_TOLERANCE_M
                and base_yaw_error
                <= WHOLE_BODY_BASE_YAW_WAYPOINT_TOLERANCE_RAD
                and articulation_error
                <= WHOLE_BODY_ARTICULATION_WAYPOINT_TOLERANCE_RAD
            )
        else:
            normalized_components = np.asarray(
                diff[sorted(set(articulation_idx))] / ARM_WAYPOINT_TOLERANCE_RAD,
                dtype=np.float64,
            )
            reached = articulation_error <= ARM_WAYPOINT_TOLERANCE_RAD
        return {
            "available": True,
            "reached": bool(reached),
            "max_articulation_error_rad": articulation_error,
            "articulation_waypoint_tolerance_rad": ARM_WAYPOINT_TOLERANCE_RAD,
            "max_base_xy_error_m": base_xy_error,
            "base_yaw_error_rad": base_yaw_error,
            "base_waypoint_xy_tolerance_m": (
                WHOLE_BODY_BASE_XY_WAYPOINT_TOLERANCE_M
                if hand is None
                else 0.01
            ),
            "base_waypoint_yaw_tolerance_rad": (
                WHOLE_BODY_BASE_YAW_WAYPOINT_TOLERANCE_RAD
            ),
            "active_dof_count": 21 if hand is None else len(articulation_idx),
            "normalized_21d_tracking_error": (
                float(np.max(np.abs(normalized_components)))
                if hand is None
                else None
            ),
            "active_joint_l2_error": float(
                np.linalg.norm(
                    tracking_diff[
                        active_idx if hand is None else articulation_idx
                    ]
                )
            ),
        }

    def capture_trajectory_hold_reference(
        self,
        *,
        hand: str | None,
        motion_scope: str = "arm_only",
    ) -> dict[str, Any]:
        """Capture fixed world/joint targets and gripper commands once.

        The reference intentionally contains q-space values, not a packed 23D
        action.  In particular, an ARM trajectory's six virtual base joints are
        fixed in world coordinates and converted through ``q_to_action`` again
        at every control step as the robot root frame changes.
        """

        if motion_scope not in _MOTION_SCOPES:
            raise ValueError(f"unsupported analytic motion scope {motion_scope!r}")
        robot = self._find_robot()
        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
        base_indices: list[int] = []
        articulation_indices: list[int] = []
        if motion_scope == "whole_body":
            if hand is not None:
                raise RuntimeError(
                    "whole-body trajectory hold reference requires hand=None"
                )
            base_all = _indices(getattr(robot, "base_idx", []))
            gripper_control_idx = getattr(robot, "gripper_control_idx", {}) or {}
            if len(base_all) != 6:
                raise RuntimeError(
                    "R1Pro whole-body trajectory requires six virtual base joints"
                )
            base_indices = base_all[2:5]
            for side in ("left", "right"):
                indices = _indices(gripper_control_idx.get(side, []))
                if len(indices) != 2:
                    raise RuntimeError(
                        f"R1Pro {side} gripper joint indices are unavailable"
                    )
                articulation_indices.extend(indices)
            scope = "whole_body_21dof_locks_base_z_roll_pitch_and_both_finger_pairs"
        elif hand is None:
            scope = "base_trajectory_locks_trunk_and_both_arms"
            articulation_indices.extend(
                _indices(getattr(robot, "trunk_control_idx", []))
            )
            for side in ("left", "right"):
                articulation_indices.extend(_indices(arm_control_idx.get(side, [])))
        else:
            hand = _normalize_hand(hand)
            inactive = "right" if hand == "left" else "left"
            scope = (
                f"{hand}_arm_with_trunk_trajectory_locks_full_base_trunk4_and_"
                f"{inactive}_arm"
                if motion_scope == "arm_with_trunk"
                else (f"{hand}_arm_trajectory_locks_full_base_trunk_and_{inactive}_arm")
            )
            base_indices = _indices(getattr(robot, "base_idx", []))
            if len(base_indices) != 6:
                raise RuntimeError(
                    "R1Pro ARM trajectory requires all six virtual base joint indices"
                )
            trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
            if len(trunk_indices) != 4:
                raise RuntimeError("R1Pro trajectory requires four trunk joints")
            articulation_indices.extend(
                trunk_indices[-1:]
                if motion_scope == "arm_with_trunk"
                else trunk_indices
            )
            articulation_indices.extend(_indices(arm_control_idx.get(inactive, [])))
        base_indices = list(dict.fromkeys(base_indices))
        articulation_indices = sorted(set(articulation_indices))
        q_indices = base_indices + [
            index for index in articulation_indices if index not in base_indices
        ]
        if (
            not q_indices
            or min(q_indices) < 0
            or max(q_indices) >= len(q)
            or not np.isfinite(q[q_indices]).all()
        ):
            raise RuntimeError("R1Pro trajectory hold joint indices unavailable")
        locked_joint_reference = {
            "base_indices": base_indices,
            "base_values": q[base_indices].copy(),
            "articulation_indices": articulation_indices,
            "articulation_values": q[articulation_indices].copy(),
            "scope": scope,
            "base_xy_threshold_m": LOCKED_BASE_XY_MAX_DRIFT_M,
            "base_z_threshold_m": LOCKED_BASE_Z_MAX_DRIFT_M,
            "base_rpy_threshold_rad": LOCKED_BASE_RPY_MAX_DRIFT_RAD,
            "articulation_threshold_rad": LOCKED_ARTICULATION_MAX_DRIFT_RAD,
        }
        return {
            "hand": hand,
            "q_indices": q_indices,
            "q_values": q[q_indices].copy(),
            "gripper_commands": {
                side: self._gripper_latch(side) for side in ("left", "right")
            },
            "locked_joint_reference": locked_joint_reference,
            "scope": scope,
            "motion_scope": motion_scope,
        }

    def capture_navigation_isolation_reference(self) -> dict[str, Any]:
        """Capture immutable non-BASE and attachment state for one navigation call."""

        robot = self._find_robot()
        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        base_indices = _indices(getattr(robot, "base_idx", []))
        trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
        arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
        arm_indices = {
            hand: _indices(arm_control_idx.get(hand, [])) for hand in ("left", "right")
        }
        if (
            len(base_indices) != 6
            or len(trunk_indices) != 4
            or any(len(arm_indices[hand]) != 7 for hand in ("left", "right"))
        ):
            raise RuntimeError(
                "R1Pro navigation isolation joint indices are unavailable"
            )
        selected_indices = (
            base_indices[2:5]
            + trunk_indices
            + arm_indices["left"]
            + arm_indices["right"]
        )
        if (
            min(selected_indices) < 0
            or max(selected_indices) >= len(q)
            or not np.isfinite(q[selected_indices]).all()
        ):
            raise RuntimeError(
                "R1Pro navigation isolation joint feedback is unavailable"
            )
        return {
            "mode": "base_only",
            "base_z_index": base_indices[2],
            "base_roll_pitch_indices": base_indices[3:5],
            "trunk_indices": trunk_indices,
            "arm_indices": arm_indices,
            "q_reference": q.copy(),
            "gripper_commands": {
                hand: self._gripper_latch(hand) for hand in ("left", "right")
            },
            "attachments": {
                hand: self.get_attached_object(hand) for hand in ("left", "right")
            },
        }

    def navigation_isolation_report(
        self,
        *,
        action: Any,
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        """Verify actual BASE-only isolation after one admitted action."""

        thresholds = {
            "base_z_m": LOCKED_BASE_Z_MAX_DRIFT_M,
            "base_roll_pitch_rad": LOCKED_BASE_RPY_MAX_DRIFT_RAD,
            "articulation_rad": LOCKED_ARTICULATION_MAX_DRIFT_RAD,
            "gripper_command": LOCKED_GRIPPER_COMMAND_MAX_DRIFT,
        }
        try:
            if not isinstance(reference, dict) or reference.get("mode") != "base_only":
                raise RuntimeError("navigation isolation reference is invalid")
            robot = self._find_robot()
            current_q = np.asarray(
                _jsonable(robot.get_joint_positions()), dtype=np.float64
            ).reshape(-1)
            expected_q = np.asarray(
                _jsonable(reference.get("q_reference")), dtype=np.float64
            ).reshape(-1)
            if (
                current_q.shape != expected_q.shape
                or current_q.size < 1
                or not np.isfinite(current_q).all()
                or not np.isfinite(expected_q).all()
            ):
                raise RuntimeError("navigation joint feedback is unavailable")
            base_z_index = int(reference["base_z_index"])
            base_roll_pitch_indices = _indices(reference["base_roll_pitch_indices"])
            trunk_indices = _indices(reference["trunk_indices"])
            raw_arm_indices = reference.get("arm_indices")
            if not isinstance(raw_arm_indices, dict):
                raise RuntimeError("navigation arm indices are unavailable")
            arm_indices = {
                hand: _indices(raw_arm_indices.get(hand)) for hand in ("left", "right")
            }
            all_indices = (
                [base_z_index]
                + base_roll_pitch_indices
                + trunk_indices
                + arm_indices["left"]
                + arm_indices["right"]
            )
            if (
                len(base_roll_pitch_indices) != 2
                or len(trunk_indices) != 4
                or any(len(arm_indices[hand]) != 7 for hand in ("left", "right"))
                or min(all_indices) < 0
                or max(all_indices) >= len(current_q)
            ):
                raise RuntimeError("navigation isolation indices are invalid")
            packed = validate_action_chunk(
                np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM)
            )[0]
            gripper_commands = reference.get("gripper_commands")
            if not isinstance(gripper_commands, dict) or any(
                hand not in gripper_commands
                or not math.isfinite(float(gripper_commands[hand]))
                for hand in ("left", "right")
            ):
                raise RuntimeError(
                    "navigation gripper command reference is unavailable"
                )
            attachments = reference.get("attachments")
            if not isinstance(attachments, dict):
                raise RuntimeError("navigation attachment reference is unavailable")

            base_z_drift = abs(
                float(current_q[base_z_index] - expected_q[base_z_index])
            )
            base_roll_pitch_drift = float(
                np.max(
                    np.abs(
                        current_q[base_roll_pitch_indices]
                        - expected_q[base_roll_pitch_indices]
                    )
                )
            )
            trunk_drift = float(
                np.max(np.abs(current_q[trunk_indices] - expected_q[trunk_indices]))
            )
            arm_drifts = {
                hand: float(
                    np.max(
                        np.abs(
                            current_q[arm_indices[hand]] - expected_q[arm_indices[hand]]
                        )
                    )
                )
                for hand in ("left", "right")
            }
            gripper_drifts = {
                hand: abs(
                    float(packed[ENV_ACTION_SEGMENTS[f"{hand}_gripper"]][0])
                    - float(gripper_commands[hand])
                )
                for hand in ("left", "right")
            }
            attachment_checks: dict[str, bool] = {}
            for hand in ("left", "right"):
                actual = self.get_attached_object(hand)
                matches, _identity = _attachment_state_status(
                    actual,
                    attachments.get(hand),
                    hand=hand,
                )
                attachment_checks[hand] = bool(matches)
            checks = {
                "base_z_locked": (base_z_drift <= LOCKED_BASE_Z_MAX_DRIFT_M + 1e-9),
                "base_roll_pitch_locked": (
                    base_roll_pitch_drift <= LOCKED_BASE_RPY_MAX_DRIFT_RAD + 1e-9
                ),
                "trunk_locked": (
                    trunk_drift <= LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
                ),
                "left_arm_locked": (
                    arm_drifts["left"] <= LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
                ),
                "right_arm_locked": (
                    arm_drifts["right"] <= LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
                ),
                "left_gripper_command_locked": (
                    gripper_drifts["left"] <= LOCKED_GRIPPER_COMMAND_MAX_DRIFT + 1e-12
                ),
                "right_gripper_command_locked": (
                    gripper_drifts["right"] <= LOCKED_GRIPPER_COMMAND_MAX_DRIFT + 1e-12
                ),
                "left_attachment_identity_unchanged": attachment_checks["left"],
                "right_attachment_identity_unchanged": attachment_checks["right"],
            }
            return {
                "available": True,
                "ok": all(checks.values()),
                "mode": "base_only",
                "checks": checks,
                "max_observed": {
                    "base_z_drift_m": base_z_drift,
                    "base_roll_pitch_drift_rad": base_roll_pitch_drift,
                    "trunk_drift_rad": trunk_drift,
                    "left_arm_drift_rad": arm_drifts["left"],
                    "right_arm_drift_rad": arm_drifts["right"],
                    "left_gripper_command_drift": gripper_drifts["left"],
                    "right_gripper_command_drift": gripper_drifts["right"],
                },
                "thresholds": thresholds,
            }
        except Exception as exc:
            return {
                "available": False,
                "ok": False,
                "mode": "base_only",
                "checks": {},
                "max_observed": {},
                "thresholds": thresholds,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def capture_locked_joint_reference(self, *, hand: str | None) -> dict[str, Any]:
        """Compatibility entry point for callers that only monitor drift."""

        return self.capture_trajectory_hold_reference(hand=hand)[
            "locked_joint_reference"
        ]

    def locked_joint_drift_report(self, *, reference: dict[str, Any]) -> dict[str, Any]:
        """Measure actual drift of the trajectory-start locked DOFs."""

        robot = self._find_robot()
        base_indices = _indices(reference.get("base_indices"))
        base_expected = np.asarray(
            _jsonable(reference.get("base_values", [])), dtype=np.float64
        ).reshape(-1)
        articulation_indices = _indices(reference.get("articulation_indices"))
        articulation_expected = np.asarray(
            _jsonable(reference.get("articulation_values", [])), dtype=np.float64
        ).reshape(-1)
        current_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        all_indices = base_indices + articulation_indices
        valid_base = not base_indices or (
            len(base_indices) == 6 and len(base_expected) == 6
        )
        valid_articulation = len(articulation_indices) == len(articulation_expected)
        if not all_indices or not valid_base or not valid_articulation:
            return {
                "available": False,
                "ok": False,
                "reason": "locked joint reference is invalid",
            }
        if (
            min(all_indices) < 0
            or max(all_indices) >= len(current_q)
            or not np.isfinite(base_expected).all()
            or not np.isfinite(articulation_expected).all()
            or not np.isfinite(current_q[all_indices]).all()
        ):
            return {
                "available": False,
                "ok": False,
                "reason": "locked joint feedback is invalid",
            }
        base_xy_drift_m = None
        base_z_drift_m = None
        base_rpy_drift_rad = None
        if base_indices:
            base_xy_drift_m = float(
                np.max(np.abs(current_q[base_indices[:2]] - base_expected[:2]))
            )
            base_z_drift_m = abs(float(current_q[base_indices[2]] - base_expected[2]))
            base_rpy_drift_rad = float(
                max(
                    abs(_wrap_angle(float(current_q[index] - base_expected[offset])))
                    for offset, index in enumerate(base_indices[3:6], start=3)
                )
            )
        articulation_drift_rad = (
            float(
                np.max(np.abs(current_q[articulation_indices] - articulation_expected))
            )
            if articulation_indices
            else None
        )
        base_xy_threshold_m = float(
            reference.get("base_xy_threshold_m", LOCKED_BASE_XY_MAX_DRIFT_M)
        )
        base_z_threshold_m = float(
            reference.get("base_z_threshold_m", LOCKED_BASE_Z_MAX_DRIFT_M)
        )
        base_rpy_threshold_rad = float(
            reference.get("base_rpy_threshold_rad", LOCKED_BASE_RPY_MAX_DRIFT_RAD)
        )
        articulation_threshold_rad = float(
            reference.get(
                "articulation_threshold_rad", LOCKED_ARTICULATION_MAX_DRIFT_RAD
            )
        )
        ok = all(
            value is None or value <= threshold + 1e-9
            for value, threshold in (
                (base_xy_drift_m, base_xy_threshold_m),
                (base_z_drift_m, base_z_threshold_m),
                (base_rpy_drift_rad, base_rpy_threshold_rad),
                (articulation_drift_rad, articulation_threshold_rad),
            )
        )
        return {
            "available": True,
            "ok": bool(ok),
            "base_xy_drift_m": base_xy_drift_m,
            "base_xy_threshold_m": base_xy_threshold_m,
            "base_z_drift_m": base_z_drift_m,
            "base_z_threshold_m": base_z_threshold_m,
            "base_rpy_drift_rad": base_rpy_drift_rad,
            "base_rpy_threshold_rad": base_rpy_threshold_rad,
            "articulation_drift_rad": articulation_drift_rad,
            "articulation_threshold_rad": articulation_threshold_rad,
            "locked_joint_count": int(len(all_indices)),
            "scope": reference.get("scope"),
        }

    def locked_gripper_command_report(
        self, *, action: Any, reference: dict[str, Any]
    ) -> dict[str, Any]:
        """Check fixed controller commands, never compliant gripper joint pose."""

        commands = reference.get("gripper_commands")
        if not isinstance(commands, dict):
            return {
                "available": False,
                "ok": False,
                "reason": "fixed gripper command reference is unavailable",
            }
        packed = np.asarray(_jsonable(action), dtype=np.float64).reshape(-1)
        if packed.shape != (ACTION_DIM,) or any(
            side not in commands or not math.isfinite(float(commands[side]))
            for side in ("left", "right")
        ):
            return {
                "available": False,
                "ok": False,
                "reason": "fixed gripper command feedback is invalid",
            }
        drifts = {
            side: abs(
                float(packed[ENV_ACTION_SEGMENTS[f"{side}_gripper"]][0])
                - float(commands[side])
            )
            for side in ("left", "right")
        }
        maximum = max(drifts.values())
        return {
            "available": True,
            "ok": maximum <= LOCKED_GRIPPER_COMMAND_MAX_DRIFT,
            "left_command_drift": drifts["left"],
            "right_command_drift": drifts["right"],
            "max_command_drift": maximum,
            "command_drift_threshold": LOCKED_GRIPPER_COMMAND_MAX_DRIFT,
            "semantics": "fixed_controller_command_not_physical_joint_pose",
        }

    def capture_single_arm_isolation_reference(
        self,
        *,
        hand: str,
        gripper_only: bool,
        motion_scope: str = "arm_only",
    ) -> dict[str, Any]:
        """Capture one immutable reference for a complete analytic primitive."""

        if motion_scope not in _MOTION_SCOPES:
            raise ValueError(f"unsupported analytic motion scope {motion_scope!r}")
        expected_scope = "gripper_only" if bool(gripper_only) else motion_scope
        if bool(gripper_only) != (expected_scope == "gripper_only"):
            raise ValueError("gripper_only and motion_scope disagree")
        mode = (
            "gripper_only"
            if bool(gripper_only)
            else ("arm_motion" if motion_scope == "arm_only" else motion_scope)
        )
        selected = _normalize_hand(hand)
        inactive = "right" if selected == "left" else "left"
        robot = self._find_robot()
        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        base_indices = _indices(getattr(robot, "base_idx", []))
        trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
        arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
        locked_arm_sides = ("left", "right") if bool(gripper_only) else (inactive,)
        arm_indices = {
            side: _indices(arm_control_idx.get(side, [])) for side in locked_arm_sides
        }
        locked_trunk_indices = (
            trunk_indices[-1:] if motion_scope == "arm_with_trunk" else trunk_indices
        )
        all_indices = [
            *base_indices,
            *locked_trunk_indices,
            *(index for side in locked_arm_sides for index in arm_indices[side]),
        ]
        if (
            len(base_indices) != 6
            or len(trunk_indices) != 4
            or any(len(arm_indices[side]) != 7 for side in locked_arm_sides)
            or not all_indices
            or min(all_indices) < 0
            or max(all_indices) >= len(q)
            or not np.isfinite(q[all_indices]).all()
        ):
            raise RuntimeError(
                "R1Pro single-arm isolation joint feedback is unavailable"
            )
        eef_sides = (
            ()
            if motion_scope == "arm_with_trunk"
            else (("left", "right") if bool(gripper_only) else (inactive,))
        )
        eef_poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in eef_sides:
            pose = self.get_eef_pose(side)
            if pose is None:
                raise RuntimeError(
                    f"R1Pro {side} EEF feedback is unavailable for isolation"
                )
            position = np.asarray(_jsonable(pose[0]), dtype=np.float64).reshape(-1)
            orientation = _quat_xyzw(pose[1])
            if position.shape != (3,) or not np.isfinite(position).all():
                raise RuntimeError(
                    f"R1Pro {side} EEF feedback is invalid for isolation"
                )
            assert orientation is not None
            eef_poses[side] = (position.copy(), orientation.copy())
        locked_gripper_sides = (
            ("left", "right") if not bool(gripper_only) else (inactive,)
        )
        gripper_commands = {
            side: self._gripper_latch(side) for side in locked_gripper_sides
        }
        if any(not math.isfinite(float(value)) for value in gripper_commands.values()):
            raise RuntimeError("R1Pro gripper command latch is unavailable")
        return {
            "selected_hand": selected,
            "mode": mode,
            "motion_scope": expected_scope,
            "base_indices": base_indices,
            "base_values": q[base_indices].copy(),
            "trunk_indices": trunk_indices,
            "trunk_values": q[trunk_indices].copy(),
            "locked_trunk_indices": locked_trunk_indices,
            "locked_trunk_values": q[locked_trunk_indices].copy(),
            "arm_indices": arm_indices,
            "arm_values": {
                side: q[arm_indices[side]].copy() for side in locked_arm_sides
            },
            "eef_poses": eef_poses,
            "locked_gripper_sides": locked_gripper_sides,
            "gripper_commands": gripper_commands,
            "inactive_hand": inactive,
            "inactive_attachment": self.get_attached_object(inactive),
            "selected_attachment": (
                None if bool(gripper_only) else self.get_attached_object(selected)
            ),
            "reference_origin": "primitive_call_start",
            "inactive_eef_check": (
                "not_applicable_shared_trunk"
                if motion_scope == "arm_with_trunk"
                else "initial_world_pose"
            ),
        }

    def single_arm_isolation_report(
        self,
        *,
        hand: str,
        action: Any,
        reference: dict[str, Any],
        gripper_only: bool,
        motion_scope: str = "arm_only",
    ) -> dict[str, Any]:
        """Validate actual locked feedback after one analytic action step."""

        if motion_scope not in _MOTION_SCOPES:
            raise ValueError(f"unsupported analytic motion scope {motion_scope!r}")
        selected = _normalize_hand(hand)
        mode = (
            "gripper_only"
            if bool(gripper_only)
            else ("arm_motion" if motion_scope == "arm_only" else motion_scope)
        )
        expected_scope = "gripper_only" if bool(gripper_only) else motion_scope
        unavailable = {
            "available": False,
            "ok": False,
            "selected_hand": selected,
            "mode": mode,
            "context_id": reference.get("context_id")
            if isinstance(reference, dict)
            else None,
            "reference_origin": reference.get(
                "reference_origin", "primitive_call_start"
            )
            if isinstance(reference, dict)
            else "primitive_call_start",
            "checks": {},
            "max_observed": {},
            "thresholds": {
                "base_xy_m": LOCKED_BASE_XY_MAX_DRIFT_M,
                "base_z_m": LOCKED_BASE_Z_MAX_DRIFT_M,
                "base_rpy_rad": LOCKED_BASE_RPY_MAX_DRIFT_RAD,
                "articulation_rad": LOCKED_ARTICULATION_MAX_DRIFT_RAD,
                "inactive_eef_position_m": 0.01,
                "inactive_eef_orientation_rad": math.radians(1.0),
                "gripper_command": LOCKED_GRIPPER_COMMAND_MAX_DRIFT,
            },
        }
        if (
            not isinstance(reference, dict)
            or reference.get("selected_hand") != selected
            or reference.get("mode") != mode
            or reference.get("motion_scope") != expected_scope
            or "inactive_attachment" not in reference
            or (not bool(gripper_only) and "selected_attachment" not in reference)
        ):
            unavailable["reason"] = "isolation reference does not match action scope"
            return unavailable
        try:
            robot = self._find_robot()
            current_q = np.asarray(
                _jsonable(robot.get_joint_positions()), dtype=np.float64
            ).reshape(-1)
            base_indices = _indices(reference.get("base_indices"))
            base_expected = np.asarray(
                _jsonable(reference.get("base_values", [])), dtype=np.float64
            ).reshape(-1)
            trunk_indices = _indices(reference.get("locked_trunk_indices"))
            trunk_expected = np.asarray(
                _jsonable(reference.get("locked_trunk_values", [])), dtype=np.float64
            ).reshape(-1)
            arm_indices = reference.get("arm_indices")
            arm_values = reference.get("arm_values")
            if (
                len(base_indices) != 6
                or base_expected.shape != (6,)
                or len(trunk_indices) != (1 if motion_scope == "arm_with_trunk" else 4)
                or trunk_expected.shape
                != ((1,) if motion_scope == "arm_with_trunk" else (4,))
                or not isinstance(arm_indices, dict)
                or not isinstance(arm_values, dict)
            ):
                raise RuntimeError("locked joint reference is incomplete")
            locked_arm_sides = tuple(str(side) for side in arm_indices)
            if not locked_arm_sides:
                raise RuntimeError("locked arm reference is empty")
            locked_indices = [*base_indices, *trunk_indices]
            arm_drifts: dict[str, float] = {}
            for side in locked_arm_sides:
                indices = _indices(arm_indices.get(side))
                expected = np.asarray(
                    _jsonable(arm_values.get(side, [])), dtype=np.float64
                ).reshape(-1)
                if len(indices) != 7 or expected.shape != (7,):
                    raise RuntimeError(f"{side} locked arm reference is incomplete")
                locked_indices.extend(indices)
                arm_drifts[side] = float(np.max(np.abs(current_q[indices] - expected)))
            if (
                min(locked_indices) < 0
                or max(locked_indices) >= len(current_q)
                or not np.isfinite(current_q[locked_indices]).all()
                or not np.isfinite(base_expected).all()
                or not np.isfinite(trunk_expected).all()
            ):
                raise RuntimeError("locked joint feedback is invalid")
            base_xy_drift = float(
                np.max(np.abs(current_q[base_indices[:2]] - base_expected[:2]))
            )
            base_z_drift = abs(float(current_q[base_indices[2]] - base_expected[2]))
            base_rpy_drift = max(
                abs(_wrap_angle(float(current_q[index] - base_expected[offset])))
                for offset, index in enumerate(base_indices[3:6], start=3)
            )
            trunk_drift = float(
                np.max(np.abs(current_q[trunk_indices] - trunk_expected))
            )
            joints_ok = bool(
                base_xy_drift <= LOCKED_BASE_XY_MAX_DRIFT_M + 1e-9
                and base_z_drift <= LOCKED_BASE_Z_MAX_DRIFT_M + 1e-9
                and base_rpy_drift <= LOCKED_BASE_RPY_MAX_DRIFT_RAD + 1e-9
                and trunk_drift <= LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
                and all(
                    drift <= LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
                    for drift in arm_drifts.values()
                )
            )

            eef_reports: dict[str, dict[str, Any]] = {}
            eef_poses = reference.get("eef_poses")
            if motion_scope != "arm_with_trunk":
                if not isinstance(eef_poses, dict) or not eef_poses:
                    raise RuntimeError("inactive EEF reference is unavailable")
                for side, expected_pose in eef_poses.items():
                    live_pose = self.get_eef_pose(str(side))
                    if live_pose is None:
                        raise RuntimeError(
                            f"{side} inactive EEF feedback is unavailable"
                        )
                    expected_position = np.asarray(
                        _jsonable(expected_pose[0]), dtype=np.float64
                    ).reshape(3)
                    position_drift = float(
                        np.linalg.norm(
                            np.asarray(
                                _jsonable(live_pose[0]), dtype=np.float64
                            ).reshape(3)
                            - expected_position
                        )
                    )
                    orientation_drift = _quat_angle_error_rad(
                        live_pose[1],
                        expected_pose[1],
                    )
                    if orientation_drift is None:
                        raise RuntimeError(
                            f"{side} inactive EEF orientation is unavailable"
                        )
                    eef_reports[str(side)] = {
                        "position_drift_m": position_drift,
                        "orientation_drift_rad": orientation_drift,
                        "ok": bool(
                            position_drift <= 0.01 + 1e-9
                            and orientation_drift <= math.radians(1.0) + 1e-9
                        ),
                    }

            packed = np.asarray(_jsonable(action), dtype=np.float64).reshape(-1)
            locked_gripper_sides = tuple(reference.get("locked_gripper_sides", ()))
            commands = reference.get("gripper_commands")
            if (
                packed.shape != (ACTION_DIM,)
                or not locked_gripper_sides
                or not isinstance(commands, dict)
            ):
                raise RuntimeError("locked gripper command feedback is unavailable")
            gripper_drifts = {
                str(side): abs(
                    float(packed[ENV_ACTION_SEGMENTS[f"{side}_gripper"]][0])
                    - float(commands[side])
                )
                for side in locked_gripper_sides
            }
            grippers_ok = all(
                drift <= LOCKED_GRIPPER_COMMAND_MAX_DRIFT + 1e-12
                for drift in gripper_drifts.values()
            )
            inactive = str(reference.get("inactive_hand"))
            attachment_ok, attachment_report = _attachment_state_status(
                self.get_attached_object(inactive),
                reference.get("inactive_attachment"),
                hand=inactive,
            )
            selected_attachment_ok = True
            selected_attachment_report: dict[str, Any] | None = None
            if not bool(gripper_only):
                selected_attachment_ok, selected_attachment_report = (
                    _attachment_state_status(
                        self.get_attached_object(selected),
                        reference.get("selected_attachment"),
                        hand=selected,
                    )
                )
        except Exception as exc:
            unavailable["reason"] = f"{type(exc).__name__}: {exc}"
            return unavailable

        checks = {
            "locked_joints": {
                "ok": joints_ok,
                "base_xy_drift_m": base_xy_drift,
                "base_z_drift_m": base_z_drift,
                "base_rpy_drift_rad": base_rpy_drift,
                "trunk_drift_rad": trunk_drift,
                "locked_arm_drift_rad": arm_drifts,
            },
            "locked_gripper_commands": {
                "ok": grippers_ok,
                "command_drift": gripper_drifts,
            },
            "inactive_attachment": {
                "ok": attachment_ok,
                "hand": inactive,
                **attachment_report,
            },
        }
        if motion_scope != "arm_with_trunk":
            checks["inactive_eef"] = {
                "ok": all(bool(report["ok"]) for report in eef_reports.values()),
                "hands": eef_reports,
            }
        if not bool(gripper_only):
            assert selected_attachment_report is not None
            checks["selected_attachment"] = {
                "ok": selected_attachment_ok,
                "hand": selected,
                **selected_attachment_report,
            }
        return {
            "available": True,
            "ok": all(bool(check["ok"]) for check in checks.values()),
            "selected_hand": selected,
            "mode": mode,
            "motion_scope": expected_scope,
            "context_id": reference.get("context_id"),
            "reference_origin": reference.get(
                "reference_origin", "primitive_call_start"
            ),
            "checks": checks,
            "max_observed": {
                "base_xy_m": base_xy_drift,
                "base_z_m": base_z_drift,
                "base_rpy_rad": base_rpy_drift,
                "trunk_rad": trunk_drift,
                "inactive_arm_rad": max(arm_drifts.values()),
                "inactive_eef_position_m": (
                    max(
                        float(report["position_drift_m"])
                        for report in eef_reports.values()
                    )
                    if eef_reports
                    else None
                ),
                "inactive_eef_orientation_rad": (
                    max(
                        float(report["orientation_drift_rad"])
                        for report in eef_reports.values()
                    )
                    if eef_reports
                    else None
                ),
                "gripper_command": max(gripper_drifts.values()),
            },
            "thresholds": unavailable["thresholds"],
            "inactive_eef_check": (
                "not_applicable_shared_trunk"
                if motion_scope == "arm_with_trunk"
                else "initial_world_pose"
            ),
        }

    def single_arm_isolation_hold_action(
        self,
        *,
        reference: dict[str, Any],
    ) -> np.ndarray:
        """Repack the original locked q targets in the current robot root frame."""

        robot = self._find_robot()
        target_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float32
        ).reshape(-1)
        groups: list[tuple[list[int], np.ndarray]] = [
            (
                _indices(reference.get("base_indices")),
                np.asarray(
                    _jsonable(reference.get("base_values", [])), dtype=np.float32
                ).reshape(-1),
            ),
        ]
        groups.append(
            (
                _indices(reference.get("locked_trunk_indices")),
                np.asarray(
                    _jsonable(reference.get("locked_trunk_values", [])),
                    dtype=np.float32,
                ).reshape(-1),
            )
        )
        arm_indices = reference.get("arm_indices")
        arm_values = reference.get("arm_values")
        if not isinstance(arm_indices, dict) or not isinstance(arm_values, dict):
            raise RuntimeError("single-arm isolation arm reference is unavailable")
        for side, raw_indices in arm_indices.items():
            groups.append(
                (
                    _indices(raw_indices),
                    np.asarray(
                        _jsonable(arm_values.get(side, [])), dtype=np.float32
                    ).reshape(-1),
                )
            )
        for indices, values in groups:
            if (
                not indices
                or len(indices) != len(values)
                or min(indices) < 0
                or max(indices) >= len(target_q)
                or not np.isfinite(values).all()
            ):
                raise RuntimeError("single-arm isolation q reference is invalid")
            target_q[indices] = values
        action = self.joint_target_to_action(target_q, hand=None)
        commands = reference.get("gripper_commands")
        if not isinstance(commands, dict):
            raise RuntimeError("single-arm isolation gripper reference is unavailable")
        for side, command in commands.items():
            action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = float(command)
        return validate_action_chunk(
            np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM)
        )[0]

    def q_trajectory_to_actions(self, q_traj: Any, *, hand: str | None) -> np.ndarray:
        robot = self._find_robot()
        q = np.asarray(_jsonable(q_traj), dtype=np.float64)
        if q.ndim != 2:
            raise RuntimeError(f"cuRobo q trajectory must be [T,D], got {q.shape}")
        fixed_reference = self.capture_trajectory_hold_reference(hand=hand)
        actions = []
        q_names = list(getattr(robot, "joints", {}).keys())
        for row in q:
            action = self.joint_target_to_action(
                row, hand=hand, fixed_reference=fixed_reference
            )
            actions.append(action)

        try:
            if q_names:
                names_path = (
                    self.output_dir
                    / "planner_curobo_configs"
                    / "last_q_joint_names.json"
                )
                names_path.parent.mkdir(parents=True, exist_ok=True)
                names_path.write_text(json.dumps(q_names, indent=2), encoding="utf-8")
        except Exception:
            pass
        return validate_action_chunk(np.stack(actions, axis=0))

    def joint_target_to_action(
        self,
        q: Any,
        *,
        hand: str | None,
        fixed_reference: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Use R1Pro q_to_action while preserving the 23D smooth-gripper ABI."""
        robot = self._find_robot()
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        target_values = np.asarray(_jsonable(q), dtype=np.float32).reshape(-1).copy()
        fixed_gripper_commands: dict[str, Any] | None = None
        if fixed_reference is not None:
            reference_hand = fixed_reference.get("hand")
            normalized_hand = None if hand is None else _normalize_hand(hand)
            if reference_hand != normalized_hand:
                raise RuntimeError(
                    "trajectory hold reference does not match active embodiment"
                )
            indices = _indices(fixed_reference.get("q_indices"))
            values = np.asarray(
                _jsonable(fixed_reference.get("q_values", [])), dtype=np.float32
            ).reshape(-1)
            if (
                not indices
                or len(indices) != len(values)
                or min(indices) < 0
                or max(indices) >= len(target_values)
                or not np.isfinite(values).all()
            ):
                raise RuntimeError("trajectory hold q-space reference is invalid")
            target_values[indices] = values
            commands = fixed_reference.get("gripper_commands")
            if not isinstance(commands, dict) or any(
                side not in commands or not math.isfinite(float(commands[side]))
                for side in ("left", "right")
            ):
                raise RuntimeError("trajectory gripper command reference is invalid")
            fixed_gripper_commands = commands
        target = torch.as_tensor(target_values, dtype=torch.float32)
        expected_layout = (
            ("base", 3),
            ("trunk", 4),
            ("arm_left", 7),
            ("gripper_left", 1),
            ("arm_right", 7),
            ("gripper_right", 1),
        )
        controllers = list(getattr(robot, "controllers", {}).items())
        actual_layout = tuple(
            (str(name), int(controller.command_dim)) for name, controller in controllers
        )
        if actual_layout != expected_layout:
            raise RuntimeError(
                "R1Pro controller layout does not match the 23D env action contract: "
                f"expected={expected_layout!r} actual={actual_layout!r}"
            )

        self._ensure_23d_q_to_action(robot)
        robot._rpent_gripper_latches = {
            side: float(fixed_gripper_commands[side])
            if fixed_gripper_commands is not None
            else self._gripper_latch(side)
            for side in ("left", "right")
        }

        # This official conversion performs the required world joint target to
        # current-root-local base command conversion. It is intentionally called
        # online for every controller step because the root frame keeps changing.
        action = np.asarray(
            _jsonable(robot.q_to_action(target)), dtype=np.float32
        ).reshape(-1)
        if action.shape != (ACTION_DIM,):
            raise RuntimeError(
                f"R1Pro controller packer returned {action.shape}, expected [{ACTION_DIM}]"
            )
        if fixed_reference is not None:
            for side in ("left", "right"):
                action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = float(
                    fixed_gripper_commands[side]
                )
            return action
        hold = self.hold_action() if hand is not None else action
        return self._apply_latches_and_inactive_segments(action, hold, hand=hand)

    @staticmethod
    def _ensure_23d_q_to_action(robot: Any) -> None:
        """Extend OG q_to_action only for its 1D smooth gripper controllers.

        OmniGibson's official cuRobo example configures grippers as absolute
        ``JointController`` instances, so its stock ``q_to_action`` rejects the
        RLinf BEHAVIOR config's 1D ``MultiFingerGripperController(mode=smooth)``.
        The rest of the conversion, especially holonomic world-to-local base
        handling, stays byte-for-byte equivalent to the official method.  The
        two gripper controller slots are filled from explicit command latches;
        no joint-space target is collapsed into a potentially unsafe grasp.
        """
        if getattr(robot, "_rpent_q_to_action_23d", False):
            return

        controllers = getattr(robot, "controllers", {}) or {}
        gripper_names = []
        for name, controller in controllers.items():
            mro_names = {cls.__name__ for cls in type(controller).__mro__}
            if "MultiFingerGripperController" not in mro_names:
                continue
            if name not in {"gripper_left", "gripper_right"}:
                raise RuntimeError(
                    "unsupported MultiFingerGripperController outside R1Pro grippers: "
                    f"{name!r}"
                )
            if (
                int(getattr(controller, "command_dim", -1)) != 1
                or getattr(controller, "_mode", None) != "smooth"
            ):
                raise RuntimeError(
                    "R1Pro 23D q_to_action adapter requires 1D smooth grippers; "
                    f"controller={name!r} command_dim="
                    f"{getattr(controller, 'command_dim', None)!r} "
                    f"mode={getattr(controller, '_mode', None)!r}"
                )
            gripper_names.append(name)
        if not gripper_names:
            return
        if set(gripper_names) != {"gripper_left", "gripper_right"}:
            raise RuntimeError(
                "R1Pro 23D q_to_action adapter requires both smooth grippers"
            )

        def q_to_action_23d(bound_robot: Any, q: Any) -> Any:
            import omnigibson.utils.transform_utils as transform_utils
            import torch as th
            from omnigibson.utils.geometry_utils import wrap_angle

            actions = []
            latches = getattr(bound_robot, "_rpent_gripper_latches", {})
            for name, controller in bound_robot.controllers.items():
                mro_names = {cls.__name__ for cls in type(controller).__mro__}
                if "MultiFingerGripperController" in mro_names:
                    side = name.removeprefix("gripper_")
                    latch = float(latches.get(side, 1.0))
                    actions.append(
                        th.as_tensor([latch], dtype=q.dtype, device=q.device)
                    )
                    continue
                if (
                    "JointController" not in mro_names
                    and "HolonomicBaseJointController" not in mro_names
                ) or bool(getattr(controller, "use_delta_commands", False)):
                    raise RuntimeError(
                        "q_to_action requires absolute joint controllers; "
                        f"controller={name!r} type={type(controller).__name__!r}"
                    )
                command = q[controller.dof_idx]
                if "HolonomicBaseJointController" in mro_names:
                    current_rz = bound_robot.get_joint_positions()[
                        bound_robot.base_idx
                    ][5]
                    delta_q = wrap_angle(command[2] - current_rz)
                    body_pose = bound_robot.get_position_orientation()
                    canonical_pos = th.tensor(
                        [command[0], command[1], body_pose[0][2]],
                        dtype=th.float32,
                        device=q.device,
                    )
                    local_pos = transform_utils.relative_pose_transform(
                        canonical_pos,
                        th.tensor(
                            [0.0, 0.0, 0.0, 1.0],
                            dtype=th.float32,
                            device=q.device,
                        ),
                        *body_pose,
                    )[0]
                    command = th.stack((local_pos[0], local_pos[1], delta_q))
                actions.append(controller._reverse_preprocess_command(command))
            action = th.cat(actions, dim=0)
            if int(action.shape[0]) != int(bound_robot.action_dim):
                raise RuntimeError(
                    "R1Pro q_to_action adapter returned an invalid action size: "
                    f"{int(action.shape[0])} != {int(bound_robot.action_dim)}"
                )
            return action

        robot._rpent_original_q_to_action = robot.q_to_action
        robot.q_to_action = types.MethodType(q_to_action_23d, robot)
        robot._rpent_q_to_action_23d = True

    def hold_action(self, hand: str | None = None) -> np.ndarray:
        del hand
        robot = self._find_robot()
        q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
        action = self.joint_target_to_action(q, hand=None)
        for side in ("left", "right"):
            action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = self._gripper_latch(side)
        return validate_action_chunk(action.reshape(1, ACTION_DIM))[0]

    def velocity_base_hold_action(self) -> np.ndarray:
        """Freeze current joints while commanding zero velocity to the Pi0 base."""

        robot = self._find_robot()
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        q = torch.as_tensor(robot.get_joint_positions(), dtype=torch.float32).reshape(
            -1
        )
        actions = []
        expected_layout = (
            ("base", 3),
            ("trunk", 4),
            ("arm_left", 7),
            ("gripper_left", 1),
            ("arm_right", 7),
            ("gripper_right", 1),
        )
        controllers = list(getattr(robot, "controllers", {}).items())
        actual_layout = tuple(
            (str(name), int(controller.command_dim)) for name, controller in controllers
        )
        if actual_layout != expected_layout:
            raise RuntimeError(
                "velocity-base hold requires the canonical R1Pro controller layout"
            )
        for name, controller in controllers:
            mro_names = {cls.__name__ for cls in type(controller).__mro__}
            if name == "base":
                if (
                    "HolonomicBaseJointController" not in mro_names
                    or str(getattr(controller, "motor_type", "")) != "velocity"
                ):
                    raise RuntimeError(
                        "velocity-base hold requires the Pi0 velocity base controller"
                    )
                command = torch.zeros(
                    int(controller.command_dim), dtype=q.dtype, device=q.device
                )
                actions.append(controller._reverse_preprocess_command(command))
                continue
            if name in {"gripper_left", "gripper_right"}:
                side = name.removeprefix("gripper_")
                actions.append(
                    torch.as_tensor(
                        [self._gripper_latch(side)], dtype=q.dtype, device=q.device
                    )
                )
                continue
            if "JointController" not in mro_names or bool(
                getattr(controller, "use_delta_commands", False)
            ):
                raise RuntimeError(
                    f"velocity-base hold cannot freeze controller {name!r}"
                )
            command = q[controller.dof_idx]
            actions.append(controller._reverse_preprocess_command(command))
        action = torch.cat(actions).detach().cpu().numpy().astype(np.float32)
        return validate_action_chunk(action.reshape(1, ACTION_DIM))[0]

    def _apply_latches_and_inactive_segments(
        self,
        action: np.ndarray,
        hold: np.ndarray,
        *,
        hand: str | None,
    ) -> np.ndarray:
        _verify_env_action_segments()
        out = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
        hold = np.asarray(hold, dtype=np.float32).reshape(ACTION_DIM)
        if hand is None:
            for segment in ("trunk", "left_arm", "right_arm"):
                out[ENV_ACTION_SEGMENTS[segment]] = hold[ENV_ACTION_SEGMENTS[segment]]
        else:
            hand = _normalize_hand(hand)
            out[ENV_ACTION_SEGMENTS["base"]] = hold[ENV_ACTION_SEGMENTS["base"]]
            inactive = "right" if hand == "left" else "left"
            out[ENV_ACTION_SEGMENTS[f"{inactive}_arm"]] = hold[
                ENV_ACTION_SEGMENTS[f"{inactive}_arm"]
            ]
        for side in ("left", "right"):
            out[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = self._gripper_latch(side)
        return out

    def _gripper_latch(self, hand: str) -> float:
        value = getattr(self.env_facade, "_gripper_latch", {}).get(hand, 1.0)
        return float(value)

    def joint_margin(self) -> float | None:
        report = self.joint_margin_report()
        return report.get("min_normalized_margin") if report.get("available") else None

    def joint_margin_report(self) -> dict[str, Any]:
        robot = self._find_robot()
        try:
            q_normalized = np.asarray(
                _jsonable(robot.get_joint_positions(normalized=True)), dtype=np.float64
            )
            q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
            position_limits = robot.control_limits["position"]
            lower = np.asarray(_jsonable(position_limits[0]), dtype=np.float64)
            upper = np.asarray(_jsonable(position_limits[1]), dtype=np.float64)
            controlled = _indices(getattr(robot, "trunk_control_idx", []))
            arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
            for side in ("left", "right"):
                controlled.extend(_indices(arm_control_idx.get(side, [])))
            controlled = sorted(set(controlled))
            if len(controlled) != 18:
                raise RuntimeError(
                    "expected 18 trunk+arm controlled joints for R1Pro, "
                    f"got {len(controlled)}"
                )
            relevant_normalized = q_normalized[controlled]
            relevant = q[controlled]
            relevant_lower = lower[controlled]
            relevant_upper = upper[controlled]
            if not all(
                np.isfinite(values).all()
                for values in (
                    relevant_normalized,
                    relevant,
                    relevant_lower,
                    relevant_upper,
                )
            ):
                raise RuntimeError("trunk/arm joint limits or state are non-finite")
            ranges = relevant_upper - relevant_lower
            if np.any(ranges <= 0.0):
                raise RuntimeError("trunk/arm joint position range is invalid")
            raw_margins = np.minimum(
                relevant - relevant_lower,
                relevant_upper - relevant,
            )
            range_fractions = raw_margins / ranges
            normalized_margin = float(np.min(1.0 - np.abs(relevant_normalized)))
            raw_threshold = 0.05
            range_threshold = 0.03
            per_joint_ok = (raw_margins >= raw_threshold) | (
                range_fractions >= range_threshold
            )
            dof_names = list(getattr(robot, "dof_names_ordered", []))
            names = [
                str(dof_names[index]) if index < len(dof_names) else f"dof_{index}"
                for index in controlled
            ]
            limiting_index = int(np.argmin(range_fractions))
            return {
                "available": True,
                "min_normalized_margin": normalized_margin,
                "min_raw_margin_joint_units": float(np.min(raw_margins)),
                "min_range_fraction": float(np.min(range_fractions)),
                "limiting_joint": names[limiting_index],
                "threshold_normalized": range_threshold,
                "threshold_raw_rad": raw_threshold,
                "threshold_range_fraction": range_threshold,
                "policy": "each_joint_raw_0.05_or_range_fraction_0.03",
                "ok": bool(np.all(per_joint_ok)),
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_normalized_margin": None,
                "min_raw_margin_joint_units": None,
                "min_range_fraction": None,
                "threshold_normalized": 0.03,
                "threshold_raw_rad": 0.05,
                "threshold_range_fraction": 0.03,
                "policy": "each_joint_raw_0.05_or_range_fraction_0.03",
                "ok": None,
            }

    def dynamics_report(self) -> dict[str, Any]:
        """Measure actual controlled-joint velocity and acceleration limits."""
        try:
            robot = self._find_robot()
            velocity = np.asarray(
                _jsonable(robot.get_joint_velocities()), dtype=np.float64
            ).reshape(-1)
            velocity_limits = np.asarray(
                _jsonable(robot.control_limits["velocity"][1]), dtype=np.float64
            ).reshape(-1)
            base_controlled = _indices(getattr(robot, "base_control_idx", []))
            trunk_controlled = _indices(getattr(robot, "trunk_control_idx", []))
            arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
            left_arm_controlled = _indices(arm_control_idx.get("left", []))
            right_arm_controlled = _indices(arm_control_idx.get("right", []))
            source_groups = (
                base_controlled,
                trunk_controlled,
                left_arm_controlled,
                right_arm_controlled,
            )
            controlled_unsorted = [
                index for group in source_groups for index in group
            ]
            if (
                velocity.shape != velocity_limits.shape
                or [len(group) for group in source_groups] != [3, 4, 7, 7]
                or len(controlled_unsorted) != 21
                or len(set(controlled_unsorted)) != 21
                or min(controlled_unsorted, default=-1) < 0
                or max(controlled_unsorted, default=-1) >= velocity.size
            ):
                raise RuntimeError("controlled velocity limits unavailable")
            controlled = sorted(controlled_unsorted)
            actual = np.abs(velocity[controlled])
            limits = np.abs(velocity_limits[controlled])
            if (
                not np.isfinite(actual).all()
                or not np.isfinite(limits).all()
                or bool(np.any(limits <= 0.0))
            ):
                raise RuntimeError("invalid actual velocity or control limit")
            velocity_ratio = float(np.max(actual / np.maximum(limits, 1e-9)))
            velocity_peak_index = int(np.argmax(actual))
            dof_names = list(getattr(robot, "dof_names_ordered", []))
            controlled_names = [
                str(dof_names[idx]) if idx < len(dof_names) else f"dof_{idx}"
                for idx in controlled
            ]
            expected_controlled_names = set(WHOLE_BODY_ACTIVE_JOINT_NAMES)
            if (
                len(controlled_names) != len(expected_controlled_names)
                or set(controlled_names) != expected_controlled_names
            ):
                raise RuntimeError(
                    "R1Pro dynamics grouping requires the exact 21 controlled "
                    "base, trunk, and arm joints"
                )
            controlled_name_to_offset = {
                name: index for index, name in enumerate(controlled_names)
            }

            def velocity_group(
                *,
                names: tuple[str, ...],
                unit: str,
            ) -> dict[str, Any]:
                offsets = np.asarray(
                    [controlled_name_to_offset[name] for name in names],
                    dtype=np.int64,
                )
                group_actual = actual[offsets]
                group_limits = limits[offsets]
                peak_index = int(np.argmax(group_actual))
                group_ratio = float(
                    np.max(group_actual / np.maximum(group_limits, 1e-9))
                )
                return {
                    "available": True,
                    "unit": unit,
                    "dof_count": int(len(names)),
                    "joint_names": list(names),
                    "dof_indices": [
                        int(controlled[controlled_name_to_offset[name]])
                        for name in names
                    ],
                    "max_actual_velocity": float(group_actual[peak_index]),
                    "max_actual_velocity_joint": names[peak_index],
                    "max_velocity_limit": float(np.max(group_limits)),
                    "max_velocity_ratio": group_ratio,
                    "within_control_limit": bool(group_ratio <= 1.0 + 1e-3),
                }

            velocity_groups = {
                "base_translation": velocity_group(
                    names=DYNAMICS_VELOCITY_GROUP_JOINTS["base_translation"],
                    unit=DYNAMICS_VELOCITY_GROUP_UNITS["base_translation"],
                ),
                "base_yaw": velocity_group(
                    names=DYNAMICS_VELOCITY_GROUP_JOINTS["base_yaw"],
                    unit=DYNAMICS_VELOCITY_GROUP_UNITS["base_yaw"],
                ),
                "articulation": velocity_group(
                    names=DYNAMICS_VELOCITY_GROUP_JOINTS["articulation"],
                    unit=DYNAMICS_VELOCITY_GROUP_UNITS["articulation"],
                ),
            }
            step = int(getattr(self.env_facade, "_env_steps", -1))
            acceleration_max = None
            acceleration_peak_joint = None
            acceleration_limit = 15.0  # official R1Pro cuRobo cspace limit
            acceleration_ratio = None
            sample_dt_s = None
            if (
                self._last_actual_velocity is not None
                and self._last_actual_velocity_step is not None
                and step > self._last_actual_velocity_step
            ):
                elapsed_steps = step - self._last_actual_velocity_step
                dt_s = float(elapsed_steps) / 60.0
                sample_dt_s = dt_s
                acceleration = np.abs(
                    (velocity[controlled] - self._last_actual_velocity) / dt_s
                )
                acceleration_peak_index = int(np.argmax(acceleration))
                acceleration_max = float(acceleration[acceleration_peak_index])
                acceleration_peak_joint = controlled_names[acceleration_peak_index]
                acceleration_ratio = acceleration_max / acceleration_limit
            self._last_actual_velocity = velocity[controlled].copy()
            self._last_actual_velocity_step = step
            return {
                "available": True,
                "ok": bool(
                    all(
                        group.get("within_control_limit") is True
                        for group in velocity_groups.values()
                    )
                    and (acceleration_ratio is None or acceleration_ratio <= 1.0 + 1e-3)
                ),
                "max_actual_velocity": float(np.max(actual)),
                "max_actual_velocity_joint": controlled_names[velocity_peak_index],
                "max_velocity_limit": float(np.max(limits)),
                "max_velocity_ratio": velocity_ratio,
                "max_actual_acceleration": acceleration_max,
                "max_actual_acceleration_joint": acceleration_peak_joint,
                "max_acceleration_limit": acceleration_limit,
                "max_acceleration_ratio": acceleration_ratio,
                "acceleration_available": acceleration_ratio is not None,
                "sample_dt_s": sample_dt_s,
                "velocity_groups": velocity_groups,
                "group_policy": "r1pro_controlled_joint_velocity_groups_v1",
                "control_limit_ratio_tolerance": 1e-3,
                "mixed_unit_top_level_fields_are_diagnostics_only": True,
                "nominal_sample_rate_hz": DYNAMICS_NOMINAL_SAMPLE_RATE_HZ,
                "source": (
                    "robot.get_joint_velocities+control_limits+"
                    "curobo_cspace_grouped"
                ),
            }
        except Exception as exc:
            return {
                "available": False,
                "ok": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def contact_report(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray | None = None,
        allowed_contact_distance_m: float = 0.025,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        robot = self._find_robot()
        finder = getattr(robot, "_find_gripper_contacts", None)
        if finder is None:
            return {
                "available": False,
                "reason": "gripper_contact_api_unavailable",
                "unexpected_contact": False,
                "expected_contact": False,
            }
        try:
            contacts, contact_links = finder(arm=hand, return_contact_positions=True)
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "unexpected_contact": False,
                "expected_contact": False,
            }
        target_object = (
            self._target_object_for_point(target_xyz)
            if target_xyz is not None
            else None
        )
        target_root = (
            str(getattr(target_object, "prim_path", "")).rstrip("/")
            if target_object is not None
            else ""
        )
        finger_paths = {
            str(getattr(link, "prim_path", "")).rstrip("/")
            for link in getattr(robot, "finger_links", {}).get(hand, ())
        }
        finger_slots = {
            path: f"finger_{index}" for index, path in enumerate(sorted(finger_paths))
        }
        target_finger_paths: set[str] = set()
        if isinstance(contact_links, dict) and target_root:
            for contacted_path, robot_links in contact_links.items():
                contacted = str(contacted_path).rstrip("/")
                if contacted == target_root or contacted.startswith(f"{target_root}/"):
                    target_finger_paths.update(
                        str(getattr(link, "prim_path", link)).rstrip("/")
                        for link in robot_links
                        if str(getattr(link, "prim_path", link)).rstrip("/")
                        in finger_paths
                    )
        target_two_finger_contact = len(target_finger_paths) >= 2
        raycast_available = False
        raycast_target_match = False
        raycast_error = None
        raycast = getattr(robot, "_find_gripper_raycast_collisions", None)
        if target_two_finger_contact and callable(raycast):
            try:
                raycast_hits = raycast(arm=hand)
                raycast_available = True
                raycast_target_match = any(
                    str(hit).rstrip("/") == target_root
                    or str(hit).rstrip("/").startswith(f"{target_root}/")
                    for hit in raycast_hits
                )
            except Exception as exc:
                raycast_error = f"{type(exc).__name__}: {exc}"
        grasp_counter = None
        try:
            grasp_counter = getattr(robot, "_ag_grasp_counter", {}).get(hand)
        except Exception:
            grasp_counter = None
        points = []
        target_points = []
        target_contact_count = 0
        unexpected_contact_count = 0
        for item in contacts:
            if isinstance(item, tuple) and len(item) >= 2:
                try:
                    contact_path = str(item[0]).rstrip("/")
                    points.append(
                        np.asarray(_jsonable(item[1]), dtype=np.float64).reshape(3)
                    )
                    if target_root and (
                        contact_path == target_root
                        or contact_path.startswith(f"{target_root}/")
                    ):
                        target_contact_count += 1
                        target_points.append(points[-1])
                    else:
                        unexpected_contact_count += 1
                except Exception:
                    unexpected_contact_count += 1
        min_distance = None
        if points and target_xyz is not None:
            target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
            min_distance = float(
                min(np.linalg.norm(point - target) for point in points)
            )
        min_target_distance = None
        max_target_distance = None
        if target_points and target_xyz is not None:
            target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
            target_distances = [
                float(np.linalg.norm(point - target)) for point in target_points
            ]
            min_target_distance = min(target_distances)
            max_target_distance = max(target_distances)
        target_contacts_in_neighborhood = bool(
            target_points
            and max_target_distance is not None
            and max_target_distance <= float(allowed_contact_distance_m)
        )
        far_target_contact_count = (
            sum(
                float(np.linalg.norm(point - np.asarray(target_xyz, dtype=np.float64)))
                > float(allowed_contact_distance_m)
                for point in target_points
            )
            if target_xyz is not None
            else len(target_points)
        )
        expected = bool(
            target_object is not None
            and target_contact_count > 0
            and target_contacts_in_neighborhood
            and unexpected_contact_count == 0
        )
        unexpected = bool(unexpected_contact_count > 0 or far_target_contact_count > 0)
        return {
            "available": True,
            "contact_count": int(len(points)),
            "target_object_resolved": target_object is not None,
            "target_contact_count": int(target_contact_count),
            "unexpected_contact_count": int(unexpected_contact_count),
            "min_contact_target_distance_m": min_distance,
            "min_target_contact_distance_m": min_target_distance,
            "max_target_contact_distance_m": max_target_distance,
            "far_target_contact_count": int(far_target_contact_count),
            "target_finger_contact_count": int(len(target_finger_paths)),
            "target_finger_contact_slots": sorted(
                finger_slots[path] for path in target_finger_paths
            ),
            "target_finger_paths": sorted(target_finger_paths),
            "target_root": target_root or None,
            "target_two_finger_contact": target_two_finger_contact,
            "assisted_grasp_raycast_available": raycast_available,
            "assisted_grasp_raycast_target_match": raycast_target_match,
            "assisted_grasp_raycast_error": raycast_error,
            "official_assisted_grasp_counter": grasp_counter,
            "official_assisted_grasp_window_s": 1.0 / 30.0,
            "allowed_contact_distance_m": float(allowed_contact_distance_m),
            "unexpected_contact": unexpected,
            "expected_contact": expected,
        }

    @staticmethod
    def _canonical_contact_pairs(contacts: Any) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for contact in contacts or ():
            body0 = str(getattr(contact, "body0", "")).rstrip("/")
            body1 = str(getattr(contact, "body1", "")).rstrip("/")
            if not body0 or not body1:
                raise RuntimeError("contact feedback omitted a body path")
            pairs.add(tuple(sorted((body0, body1))))
        return pairs

    def _whole_body_contact_pairs(
        self,
        expected_attachments_by_hand: dict[str, Any],
    ) -> set[tuple[str, str]]:
        robot = self._find_robot()
        contact_list = getattr(robot, "contact_list", None)
        if not callable(contact_list):
            raise RuntimeError("R1Pro whole-body contact API is unavailable")
        contacts = list(contact_list())
        seen_roots: set[int] = set()
        for side in ("left", "right"):
            attachment = expected_attachments_by_hand.get(side)
            if not isinstance(attachment, dict):
                continue
            for root in attachment.values():
                if id(root) in seen_roots:
                    continue
                seen_roots.add(id(root))
                root_contacts = getattr(root, "contact_list", None)
                if not callable(root_contacts):
                    raise RuntimeError(
                        f"{side} attachment contact API is unavailable"
                    )
                contacts.extend(root_contacts())
        return self._canonical_contact_pairs(contacts)

    @staticmethod
    def _is_r1pro_wheel_floor_support_pair(pair: tuple[str, str]) -> bool:
        """Recognize only the R1Pro rolling-support contact topology."""

        if len(pair) != 2:
            return False
        wheel_links = {f"wheel_motor_link{index}" for index in range(4)}
        wheel_path = next(
            (
                path
                for path in pair
                if path.rsplit("/", 1)[-1] in wheel_links
                and any(
                    component.startswith("controllable__r1pro__")
                    for component in path.split("/")
                )
            ),
            None,
        )
        floor_path = next(
            (
                path
                for path in pair
                if path.endswith("/base_link")
                and any(
                    component.startswith("floors_")
                    for component in path.split("/")
                )
            ),
            None,
        )
        return wheel_path is not None and floor_path is not None

    def capture_whole_body_contact_baseline(
        self,
        *,
        expected_attachments_by_hand: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture existing contacts without advancing the simulator."""

        try:
            pairs = self._whole_body_contact_pairs(expected_attachments_by_hand)
            support_pairs = {
                pair for pair in pairs if self._is_r1pro_wheel_floor_support_pair(pair)
            }
            monitored_pairs = pairs - support_pairs
            return {
                "available": True,
                "pairs": [list(pair) for pair in sorted(pairs)],
                "continuous_pairs": [
                    list(pair) for pair in sorted(monitored_pairs)
                ],
                "support_pairs": [list(pair) for pair in sorted(support_pairs)],
                "pair_count": len(pairs),
                "monitored_pair_count": len(monitored_pairs),
                "support_pair_count": len(support_pairs),
                "policy": (
                    "new_or_reappearing_non_support_contact_pair_is_unexpected"
                ),
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def whole_body_contact_report(
        self,
        *,
        baseline: dict[str, Any],
        expected_attachments_by_hand: dict[str, Any],
        allowed_expected_contact: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reject any contact pair not present at trajectory start."""

        if not isinstance(baseline, dict) or baseline.get("available") is not True:
            return {
                "available": False,
                "reason": "whole-body contact baseline unavailable",
                "unexpected_contact": False,
            }
        try:
            original_baseline_pairs = {
                tuple(str(value) for value in pair)
                for pair in baseline.get("pairs", ())
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
            baseline_pairs = {
                tuple(str(value) for value in pair)
                for pair in baseline.get(
                    "continuous_pairs",
                    baseline.get("pairs", ()),
                )
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
            current_pairs = self._whole_body_contact_pairs(
                expected_attachments_by_hand
            )
            support_pairs = {
                pair
                for pair in current_pairs
                if self._is_r1pro_wheel_floor_support_pair(pair)
            }
            monitored_current_pairs = current_pairs - support_pairs
            new_pairs = monitored_current_pairs - baseline_pairs
            allowed_expected_pairs: set[tuple[str, str]] = set()
            if (
                isinstance(allowed_expected_contact, dict)
                and allowed_expected_contact.get("expected_contact") is True
            ):
                target_root = str(
                    allowed_expected_contact.get("target_root") or ""
                ).rstrip("/")
                finger_paths = {
                    str(path).rstrip("/")
                    for path in allowed_expected_contact.get(
                        "target_finger_paths", ()
                    )
                    if str(path).rstrip("/")
                }
                if target_root and finger_paths:
                    for pair in new_pairs:
                        finger_path = next(
                            (path for path in pair if path in finger_paths),
                            None,
                        )
                        target_path = next(
                            (
                                path
                                for path in pair
                                if path == target_root
                                or path.startswith(f"{target_root}/")
                            ),
                            None,
                        )
                        if finger_path is not None and target_path is not None:
                            allowed_expected_pairs.add(pair)
            unexpected = sorted(new_pairs - allowed_expected_pairs)
            continuous_pairs = baseline_pairs & monitored_current_pairs
            baseline["continuous_pairs"] = [
                list(pair) for pair in sorted(continuous_pairs)
            ]
            return {
                "available": True,
                "unexpected_contact": bool(unexpected),
                "unexpected_pairs": [list(pair) for pair in unexpected],
                "allowed_expected_contact_pairs": [
                    list(pair) for pair in sorted(allowed_expected_pairs)
                ],
                "current_pairs": [list(pair) for pair in sorted(current_pairs)],
                "monitored_current_pairs": [
                    list(pair) for pair in sorted(monitored_current_pairs)
                ],
                "allowed_support_pairs": [
                    list(pair) for pair in sorted(support_pairs)
                ],
                "original_baseline_pairs": [
                    list(pair) for pair in sorted(original_baseline_pairs)
                ],
                "continuous_baseline_pairs": [
                    list(pair) for pair in sorted(continuous_pairs)
                ],
                "current_pair_count": len(current_pairs),
                "baseline_pair_count": len(baseline_pairs),
                "original_baseline_pair_count": len(original_baseline_pairs),
                "continuous_baseline_pair_count": len(continuous_pairs),
                "allowed_support_pair_count": len(support_pairs),
                "policy": (
                    "new_or_reappearing_non_support_contact_pair_is_unexpected"
                ),
                "support_policy": (
                    "r1pro_wheel_motor_link0-3_to_behavior_floors_base_link"
                ),
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "unexpected_contact": False,
            }

    def _target_object_for_point(
        self,
        target_xyz: np.ndarray | None,
        *,
        padding_m: float = 0.04,
    ) -> Any | None:
        """Resolve the smallest physical scene body containing a contact point."""

        if target_xyz is None:
            return None
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        robot = self._find_robot()
        candidates: list[tuple[float, float, Any]] = []
        scene_objects = getattr(robot.scene, "objects", ())
        if isinstance(scene_objects, dict):
            scene_objects = scene_objects.values()
        for obj in scene_objects:
            if obj is robot or bool(getattr(obj, "visual_only", False)):
                continue
            bbox = getattr(obj, "get_base_aligned_bbox", None)
            if not callable(bbox):
                continue
            try:
                center, _quat, extent, _center_local = bbox(xy_aligned=True)
                center_array = np.asarray(_jsonable(center), dtype=np.float64).reshape(
                    3
                )
                extent_array = np.asarray(_jsonable(extent), dtype=np.float64).reshape(
                    3
                )
                outside = np.maximum(
                    np.abs(target - center_array) - extent_array * 0.5,
                    0.0,
                )
                outside_distance = float(np.linalg.norm(outside))
                if outside_distance <= float(padding_m):
                    volume = float(np.prod(np.maximum(extent_array, 1e-6)))
                    candidates.append((outside_distance, volume, obj))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        nearest_distance = candidates[0][0]
        if (
            sum(
                abs(distance - nearest_distance) <= 1e-4
                for distance, _volume, _obj in candidates
            )
            != 1
        ):
            return None
        return candidates[0][2]

    def get_attached_object(self, hand: str) -> Any:
        hand = _normalize_hand(hand)
        robot = self._find_robot()
        obj = None
        try:
            obj = getattr(robot, "_ag_obj_in_hand", {}).get(hand)
        except Exception:
            obj = None
        if obj is None:
            self._attached_objects_by_hand.pop(hand, None)
            return None
        root_link = getattr(obj, "root_link", None)
        if root_link is None:
            raise RuntimeError("assisted-grasp object has no root_link collision body")
        self._attached_objects_by_hand[hand] = root_link
        return {EEF_LINK_BY_HAND[hand]: root_link}

    def clear_attached_object(self, hand: str) -> None:
        self._attached_objects_by_hand.pop(_normalize_hand(hand), None)


def _indices(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [
            int(x) for x in np.asarray(_jsonable(value), dtype=np.int64).reshape(-1)
        ]
    except Exception:
        return []


def _verify_env_action_segments() -> None:
    covered = []
    for segment in ENV_ACTION_SEGMENTS.values():
        covered.extend(range(segment.start, segment.stop))
    if covered != list(range(ACTION_DIM)):
        raise RuntimeError(
            "ENV_ACTION_SEGMENTS no longer covers the 23D env action exactly"
        )


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _canonical_base_xyyaw(value: Any) -> np.ndarray:
    """Return one finite BASE pose with yaw canonicalized to [-pi, pi)."""

    base_xyyaw = np.asarray(_jsonable(value), dtype=np.float64).reshape(-1)
    if base_xyyaw.shape != (3,) or not np.isfinite(base_xyyaw).all():
        raise ValueError("BASE xyyaw must contain three finite values")
    canonical = base_xyyaw.copy()
    yaw = float(canonical[2])
    canonical[2] = (
        _wrap_angle(yaw) if yaw < -math.pi or yaw >= math.pi else yaw
    )
    return canonical


def _supercover_grid_cells(
    start_row_col: Any,
    end_row_col: Any,
) -> list[tuple[int, int]]:
    """Return every grid cell touched by a segment between two map cells."""

    start = np.asarray(start_row_col, dtype=np.int64).reshape(2)
    end = np.asarray(end_row_col, dtype=np.int64).reshape(2)
    row, column = int(start[0]), int(start[1])
    end_row, end_column = int(end[0]), int(end[1])
    delta_column = end_column - column
    delta_row = end_row - row
    count_column = abs(delta_column)
    count_row = abs(delta_row)
    step_column = 0 if delta_column == 0 else (1 if delta_column > 0 else -1)
    step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
    advanced_column = 0
    advanced_row = 0
    cells: list[tuple[int, int]] = [(row, column)]
    while advanced_column < count_column or advanced_row < count_row:
        decision = (
            (1 + 2 * advanced_column) * count_row
            - (1 + 2 * advanced_row) * count_column
        )
        if decision == 0:
            next_column = column + step_column
            next_row = row + step_row
            # A corner crossing touches both orthogonal neighbours as well as
            # the diagonal destination; checking all three is conservative.
            cells.append((row, next_column))
            cells.append((next_row, column))
            column = next_column
            row = next_row
            advanced_column += 1
            advanced_row += 1
        elif decision < 0:
            column += step_column
            advanced_column += 1
        else:
            row += step_row
            advanced_row += 1
        cells.append((row, column))
    return list(dict.fromkeys(cells))


def _yaw_from_quat_xyzw(quat: Any) -> float:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_to_quat_xyzw(yaw: float) -> np.ndarray:
    half = float(yaw) * 0.5
    return np.array([0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float64)


def _quat_to_intrinsic_rpy(quat: Any) -> tuple[float, float, float]:
    normalized = _quat_xyzw(quat)
    assert normalized is not None
    x, y, z, w = normalized
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    return roll, pitch, _yaw_from_quat_xyzw(normalized)


def _intrinsic_rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.asarray(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _base_candidates(target_xyz: np.ndarray, *, standoff_m: float) -> list[np.ndarray]:
    candidates = []
    for angle in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        xy = target_xyz[:2] - standoff_m * np.array([math.cos(angle), math.sin(angle)])
        yaw = math.atan2(target_xyz[1] - xy[1], target_xyz[0] - xy[0])
        candidates.append(np.array([xy[0], xy[1], yaw], dtype=np.float64))
    return candidates


def _bounded_polyline_prefix(
    points_xy: Any,
    *,
    max_travel_m: float,
) -> tuple[np.ndarray, float, float, bool]:
    """Return an exact bounded prefix of a finite world-XY polyline."""

    points = np.asarray(_jsonable(points_xy), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1:
        raise ValueError(f"navigation path must be finite [N,2], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("navigation path contains NaN or infinity")
    limit = float(max_travel_m)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_travel_m must be finite and positive")

    compact = [points[0].copy()]
    for point in points[1:]:
        if float(np.linalg.norm(point - compact[-1])) > 1e-9:
            compact.append(point.copy())
    path = np.asarray(compact, dtype=np.float64)
    if len(path) == 1:
        return path, 0.0, 0.0, False

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    full_length = float(np.sum(segment_lengths))
    if full_length <= limit + 1e-9:
        return path, full_length, full_length, False

    prefix = [path[0].copy()]
    travelled = 0.0
    for start, end, segment_length in zip(
        path[:-1],
        path[1:],
        segment_lengths,
        strict=True,
    ):
        remaining = limit - travelled
        if remaining <= 1e-9:
            break
        if segment_length <= remaining + 1e-9:
            prefix.append(end.copy())
            travelled += float(segment_length)
            continue
        fraction = remaining / float(segment_length)
        prefix.append(start + fraction * (end - start))
        travelled = limit
        break
    return np.asarray(prefix, dtype=np.float64), travelled, full_length, True


class PlannerExecutor:
    """Executes planner tool requests inside the BEHAVIOR env process."""

    def __init__(
        self,
        *,
        env: Any,
        frame_cache: FrameCache,
        output_dir: str | Path | None = None,
        backend: Any | None = None,
        max_stall_steps: int = 20,
    ) -> None:
        self.env = env
        self.frame_cache = frame_cache
        self.output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path("/tmp") / "rpent_behavior_planner" / str(os.getpid())
        )
        self.backend = (
            backend
            if backend is not None
            else RealCuroboBackend(env, output_dir=self.output_dir)
        )
        self.max_stall_steps = int(max_stall_steps)
        self.last_info: Any = None
        self._trace_counter = 0
        self._isolation_context_counter = 0
        self._active_isolation_report: dict[str, Any] | None = None
        self._last_guarded_retreat_paths: dict[str, np.ndarray] = {}
        self._prepared_motion_lock = threading.RLock()
        self._prepared_motions: dict[str, dict[str, Any]] = {}
        self._warmup_report: dict[str, Any] | None = None

    def on_runtime_state_changed(self) -> None:
        """Reset executor-local state after a controller or q-state change."""

        self.last_info = None
        self._active_isolation_report = None
        self._last_guarded_retreat_paths.clear()
        with self._prepared_motion_lock:
            for entry in self._prepared_motions.values():
                if entry.get("status") in {"prepared", "executing"}:
                    entry["status"] = "invalidated"
                    entry["invalidated_reason"] = "runtime_state_changed"
        changed = getattr(self.backend, "on_runtime_state_changed", None)
        if callable(changed):
            changed()

    def warmup(self) -> dict[str, Any]:
        warmup = getattr(self.backend, "warmup", None)
        if not callable(warmup):
            raise RuntimeError("planner backend does not implement safety warmup")
        report = dict(warmup())
        self._warmup_report = deepcopy(report)
        return report

    def warmup_attached_arm(
        self,
        *,
        hand: str,
        expected_attached_root: Any,
    ) -> dict[str, Any]:
        warmup = getattr(self.backend, "warmup_attached_arm", None)
        if not callable(warmup):
            raise RuntimeError(
                "planner backend does not implement attached-arm safety warmup"
            )
        return dict(warmup(hand=hand, expected_attached_root=expected_attached_root))

    def _capture_single_arm_isolation(
        self,
        *,
        hand: str,
        gripper_only: bool,
        reference_origin: str,
        motion_scope: str = "arm_only",
    ) -> dict[str, Any] | None:
        capture = getattr(
            self.backend,
            "capture_single_arm_isolation_reference",
            None,
        )
        if not callable(capture):
            return None
        try:
            kwargs = {
                "hand": _normalize_hand(hand),
                "gripper_only": bool(gripper_only),
            }
            if motion_scope == "arm_with_trunk":
                kwargs["motion_scope"] = motion_scope
            reference = capture(**kwargs)
        except Exception:
            return None
        if not isinstance(reference, dict):
            return None
        self._isolation_context_counter += 1
        result = dict(reference)
        result["context_id"] = (
            f"single-arm-isolation-{self._isolation_context_counter:06d}"
        )
        result["reference_origin"] = str(reference_origin)
        return result

    @staticmethod
    def _merge_isolation_report(
        aggregate: dict[str, Any] | None,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain latest checks plus truthful all-step isolation evidence."""

        merged = dict(report)
        report_checks = dict(report.get("checks") or {})
        report_failed_checks = {
            str(name) for name, passed in report_checks.items() if passed is not True
        }
        if not isinstance(aggregate, dict):
            merged["checks_performed"] = 1
            merged["all_steps_strict_ok"] = report.get("ok") is True
            merged["failed_checks_observed"] = sorted(report_failed_checks)
            return merged
        maxima = dict(aggregate.get("max_observed", {}))
        for key, value in dict(report.get("max_observed", {})).items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                prior = maxima.get(key)
                maxima[key] = (
                    float(value)
                    if not isinstance(prior, (int, float))
                    else max(float(prior), float(value))
                )
        merged["max_observed"] = maxima
        merged["checks_performed"] = int(aggregate.get("checks_performed", 0)) + 1
        merged["all_steps_strict_ok"] = bool(
            aggregate.get("all_steps_strict_ok", aggregate.get("ok") is True)
            and report.get("ok") is True
        )
        merged["failed_checks_observed"] = sorted(
            {
                str(name)
                for name in aggregate.get("failed_checks_observed", ())
            }
            | report_failed_checks
        )
        return merged

    @_planner_tool("observe")
    def observe(self, camera: str) -> dict[str, Any]:
        payload = self.frame_cache.observe_payload(canonical_camera(camera))
        payload.update(
            primitive_result(
                primitive_success=True,
                task_success=self._task_success(),
                stop_reason="observed",
                recoverable=True,
                suggested_next_tool="pixel_to_world",
                metrics={
                    "camera": payload.get("camera"),
                    "frame_id": payload.get("frame_id"),
                    "step_index": payload.get("step_index"),
                },
            )
        )
        return payload

    @_planner_tool("pixel_to_world")
    def pixel_to_world(
        self,
        *,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        try:
            if u is None or v is None:
                raise CameraGeometryError("both u=column and v=row are required")
            frame = self.frame_cache.get_current(
                canonical_camera(camera), str(frame_id)
            )
            projection = backproject_pixel_to_world(
                frame,
                u=u,
                v=v,
                depth_window_px=int(depth_window_px),
                output_frame=output_frame,
            )
            metrics = {
                "camera": frame.camera,
                "frame_id": frame.frame_id,
                "step_index": frame.step_index,
                "confidence": projection["confidence"],
                "reprojection_error_px": projection["reprojection_error_px"],
                "depth": projection["depth"],
            }
            return primitive_result(
                primitive_success=True,
                task_success=self._task_success(),
                stop_reason="projected",
                recoverable=True,
                suggested_next_tool="move_to",
                metrics=metrics,
                diagnostics={
                    "xyz": projection["xyz"],
                    "surface_normal": projection["surface_normal"],
                    "output_frame": output_frame,
                },
            )
        except Exception as exc:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="projection_failed",
                recoverable=True,
                suggested_next_tool="observe",
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )

    @_planner_tool("navigate_to")
    def navigate_to(
        self,
        *,
        target_xyz: Any | None = None,
        relative_motion: Any = None,
        standoff_m: float | None = None,
        max_travel_m: float | None = None,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Move only BASE toward a target or by one explicit relative motion."""

        started = time.monotonic()
        timeout = self._validated_timeout(timeout_s)
        if self._task_success():
            return primitive_result(
                primitive_success=True,
                task_success=True,
                stop_reason="task_success",
                recoverable=False,
                suggested_next_tool=None,
                metrics={"env_actions_sent": 0},
            )
        relative_mode = relative_motion is not None
        if relative_mode:
            if any(
                value is not None
                for value in (target_xyz, standoff_m, max_travel_m)
            ):
                raise ValueError(
                    "relative_motion is mutually exclusive with target navigation"
                )
            motion = validate_relative_navigation_motion(relative_motion)
            plan_navigation = getattr(
                self.backend,
                "plan_relative_navigation_trajectory",
                None,
            )
            planner_kwargs = {
                "relative_motion": motion,
                "timeout_s": timeout,
            }
            unavailable_reason = "relative_navigation_planner_unavailable"
        else:
            if target_xyz is None:
                raise ValueError("target_xyz is required for projection navigation")
            target = _as_xyz(target_xyz)
            standoff = float(0.85 if standoff_m is None else standoff_m)
            max_travel = float(1.0 if max_travel_m is None else max_travel_m)
            if not math.isfinite(standoff) or standoff <= 0.0:
                raise ValueError("standoff_m must be finite and positive")
            if not math.isfinite(max_travel) or max_travel <= 0.0:
                raise ValueError("max_travel_m must be finite and positive")
            plan_navigation = getattr(
                self.backend,
                "plan_navigation_trajectory",
                None,
            )
            planner_kwargs = {
                "target_xyz": target,
                "standoff_m": standoff,
                "max_travel_m": max_travel,
                "timeout_s": timeout,
            }
            unavailable_reason = "navigation_planner_unavailable"
        if not callable(plan_navigation):
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason=unavailable_reason,
                recoverable=True,
                suggested_next_tool="observe",
                metrics={
                    "navigation_isolation": {
                        "available": True,
                        "ok": True,
                        "mode": "base_only",
                        "checks": {},
                        "max_observed": {},
                        "checks_performed": 0,
                        "reason": "no navigation action was admitted",
                    }
                },
            )
        plan = plan_navigation(**planner_kwargs)
        if not isinstance(plan, dict):
            raise RuntimeError("navigation planner returned a non-mapping result")
        plan_metrics = (
            dict(plan.get("metrics")) if isinstance(plan.get("metrics"), dict) else {}
        )
        planning_elapsed = plan_metrics.pop("elapsed_s", None)
        if isinstance(planning_elapsed, (int, float)):
            plan_metrics["planning_elapsed_s"] = float(planning_elapsed)
        if plan.get("ok") is not True:
            stop_reason = str(plan.get("stop_reason", "navigation_unreachable"))
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason=stop_reason,
                recoverable=stop_reason
                not in {"navigation_planner_unavailable", "planner_unavailable"},
                suggested_next_tool="observe",
                metrics={
                    **plan_metrics,
                    "navigation_isolation": {
                        "available": True,
                        "ok": True,
                        "mode": "base_only",
                        "checks": {},
                        "max_observed": {},
                        "checks_performed": 0,
                        "reason": "no navigation action was admitted",
                    },
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
            )
        trajectory = np.asarray(
            _jsonable(plan.get("joint_trajectory")), dtype=np.float32
        )
        try:
            base_goal = _canonical_base_xyyaw(plan.get("base_goal"))
        except ValueError as exc:
            raise RuntimeError(
                "navigation planner omitted a finite BASE goal"
            ) from exc
        if (
            trajectory.ndim != 2
            or len(trajectory) < 1
            or not np.isfinite(trajectory).all()
        ):
            raise RuntimeError("navigation planner omitted a finite q trajectory")
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0.0:
            raise TimeoutError("navigation planning consumed the tool deadline")
        return self._execute_navigation_trajectory(
            trajectory,
            base_goal_xyyaw=base_goal,
            timeout_s=remaining,
            plan_metrics=plan_metrics,
            expected_attachments_by_hand=plan.get(
                "expected_attachments_by_hand"
            ),
        )

    @staticmethod
    def _manual_result_metrics(
        result: dict[str, Any],
        **metrics: Any,
    ) -> dict[str, Any]:
        out = dict(result)
        out["metrics"] = {**dict(result.get("metrics") or {}), **metrics}
        return out

    def _wrist_camera_rotation_calibration(self, hand: str) -> dict[str, Any]:
        """Resolve one same-step wrist-camera axis in its captured EEF frame."""

        hand = _normalize_hand(hand)
        camera = f"{hand}_wrist"
        try:
            frame = self.frame_cache.latest(camera)
            frame = self.frame_cache.get_current(camera, frame.frame_id)
            if not isinstance(frame.capture_group_id, str) or not frame.capture_group_id:
                raise RuntimeError("wrist calibration requires an atomic capture group")
            current_env_step = int(getattr(self.env, "_env_steps", -1))
            if current_env_step != int(frame.step_index):
                raise RuntimeError(
                    "wrist calibration frame is not from the current simulator step"
                )
            metadata = (
                frame.capture_metadata
                if isinstance(frame.capture_metadata, dict)
                else {}
            )
            sync_certificate = metadata.get("hand_geometry_sync_certificate")
            if not hand_geometry_sync_certificate_is_valid(
                sync_certificate,
                hand=hand,
                env_step=int(frame.step_index),
            ):
                raise RuntimeError(
                    "wrist hand-geometry sync certificate is invalid"
                )
            hand_certificate = sync_certificate["hands"][hand]
            camera_pose_lineage = metadata.get("camera_pose_lineage")
            selected_lineage = (
                camera_pose_lineage.get(camera)
                if isinstance(camera_pose_lineage, dict)
                else None
            )
            if (
                not isinstance(selected_lineage, dict)
                or selected_lineage.get("render_bound") is not True
                or selected_lineage.get("source")
                != hand_certificate.get("camera_pose_source")
                or int(selected_lineage.get("env_step", -1))
                != int(frame.step_index)
                or int(selected_lineage.get("render_sync_iterations", -1))
                != int(sync_certificate["render_sync_iterations"])
            ):
                raise RuntimeError(
                    "wrist camera pose lineage does not match its hand certificate"
                )
            references = metadata.get("r1pro_hand_reference_transforms")
            hand_references = (
                references.get("hands")
                if isinstance(references, dict)
                else None
            )
            selected_references = (
                hand_references.get(hand)
                if isinstance(hand_references, dict)
                else None
            )
            if (
                not isinstance(references, dict)
                or references.get("available") is not True
                or references.get("source")
                != "capture_time_live_r1pro_link_transforms"
                or int(references.get("env_step", -1))
                != int(frame.step_index)
                or not isinstance(selected_references, dict)
            ):
                raise RuntimeError(
                    "captured wrist EEF transform is unavailable"
                )
            world_from_eef = validated_rigid_transform(
                selected_references.get("grip_point"),
                name=f"{hand} captured EEF world transform",
            )
            world_from_camera = validated_rigid_transform(
                frame.camera_to_world,
                name=f"{hand} captured wrist-camera world transform",
            )
            camera_rotation = np.asarray(
                world_from_camera[:3, :3], dtype=np.float64
            )
            # USD cameras use +X right, +Y up and -Z forward.  Local +Z
            # therefore points toward the viewer; by the right-hand rule a
            # positive turn around that screen normal is visually
            # counterclockwise.
            screen_normal_world = camera_rotation @ np.array(
                [0.0, 0.0, 1.0], dtype=np.float64
            )
            screen_normal_eef = (
                world_from_eef[:3, :3].T @ screen_normal_world
            )
            norm = float(np.linalg.norm(screen_normal_eef))
            if norm <= 1e-9 or not math.isfinite(norm):
                raise RuntimeError("wrist camera screen-normal axis is invalid")
            screen_normal_eef /= norm
            visual_validation = {
                "geometry_verified": True,
                "release_admission": False,
                "real_visual_probe_required": True,
                "hand": hand,
                "visual_ccw_angle_sign": 1.0,
                "camera_convention": (
                    "USD_-Z_view,+X_right,+Y_up;positive_about_+Z_is_"
                    "screen_counterclockwise"
                ),
                "source": (
                    "same_step_render_bound_RGBD+captured_EEF+"
                    "official_R1Pro_fixed_extrinsics"
                ),
                "hand_geometry_sync_certificate": sync_certificate,
            }
            return {
                "available": True,
                "verified": True,
                "release_admission": False,
                "real_visual_probe_required": True,
                "hand": hand,
                "camera": camera,
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "step_index": int(frame.step_index),
                "screen_normal_axis_eef": screen_normal_eef.tolist(),
                "visual_ccw_angle_sign": float(
                    visual_validation["visual_ccw_angle_sign"]
                ),
                "source": (
                    "same_step_RGBD_camera_to_world+captured_EEF_pose+"
                    "synchronized_official_R1Pro_hand_geometry"
                ),
                "visual_validation": dict(visual_validation),
            }
        except Exception as exc:
            return {
                "available": False,
                "verified": False,
                "hand": hand,
                "camera": camera,
                "reason": f"{type(exc).__name__}: {exc}",
            }

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        """Return fail-closed manual-control capabilities without moving the sim."""

        eef_available = callable(
            getattr(self.backend, "plan_whole_body_trajectory", None)
        )
        wrist_geometry = {
            hand: self._wrist_camera_rotation_calibration(hand)
            for hand in ("left", "right")
        }
        # Geometry is necessary for a controlled planning-only probe, but it
        # cannot by itself certify the controller's observed image direction.
        wrist = {"left": False, "right": False}
        torso_report = _call_optional(self.backend, "torso_jog_capability")
        warmup = (
            self._warmup_report
            if isinstance(self._warmup_report, dict)
            else {}
        )
        identity_warmup = (
            warmup.get("identity_warmup")
            if isinstance(warmup.get("identity_warmup"), dict)
            else {}
        )
        torso_warmup = (
            identity_warmup.get("torso")
            if isinstance(identity_warmup.get("torso"), dict)
            else {}
        )
        torso_warmup_verified = bool(
            warmup.get("status") == "complete"
            and torso_warmup.get("ok") is True
            and torso_warmup.get("collision_admitted") is True
            and int(torso_warmup.get("active_dof_count", -1)) == 3
            and int(torso_warmup.get("env_actions_sent", -1)) == 0
            and torso_warmup.get("simulator_advanced") is False
        )
        torso_available = bool(
            isinstance(torso_report, dict)
            and torso_report.get("verified") is True
            and callable(getattr(self.backend, "plan_torso_trajectory", None))
            and callable(getattr(self, "_execute_torso_trajectory", None))
            and torso_warmup_verified
        )
        return {
            "base": callable(
                getattr(self.backend, "plan_relative_navigation_trajectory", None)
            ),
            "eef": {"left": eef_available, "right": eef_available},
            "torso": {
                "available": torso_available,
                "reason": (
                    None
                    if torso_available
                    else (
                        "torso_link4 CuRobo/controller/warmup capability "
                        "is unverified"
                    )
                ),
                "backend_report": torso_report,
                "warmup_verified": torso_warmup_verified,
            },
            "wrist": wrist,
            "wrist_geometry": wrist_geometry,
            "gripper": callable(getattr(self.backend, "hold_action", None)),
        }

    @staticmethod
    def _dashboard_base_motion(action: str) -> dict[str, Any]:
        if action in {"forward", "backward"}:
            return {
                "kind": "translation",
                "direction": action,
                "distance_m": BASE_TRANSLATION_STEP_M,
            }
        if action in {"turn_left", "turn_right"}:
            return {
                "kind": "rotation",
                "direction": action.removeprefix("turn_"),
                "angle_deg": math.degrees(BASE_ROTATION_STEP_RAD),
            }
        raise ValueError(
            "BASE prepared motion must be forward, backward, turn_left, or turn_right"
        )

    def _prepared_predicted_eef_pose(
        self,
        *,
        hand: str,
        start_q: np.ndarray,
        predecessor: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted = (
            predecessor.get("predicted_terminal")
            if isinstance(predecessor, dict)
            else None
        )
        eef_by_hand = (
            predicted.get("eef_by_hand")
            if isinstance(predicted, dict)
            else None
        )
        cached = (
            eef_by_hand.get(hand)
            if isinstance(eef_by_hand, dict)
            else None
        )
        if isinstance(cached, dict):
            return (
                np.asarray(cached["xyz"], dtype=np.float64).reshape(3),
                np.asarray(cached["quat_xyzw"], dtype=np.float64).reshape(4),
            )
        poses = _call_optional_kw(
            self.backend,
            "whole_body_eef_poses",
            hand=hand,
            q_trajectory=np.asarray(start_q, dtype=np.float32).reshape(1, -1),
        )
        if isinstance(poses, (tuple, list)) and len(poses) == 2:
            positions = np.asarray(poses[0], dtype=np.float64).reshape(-1, 3)
            quaternions = np.asarray(poses[1], dtype=np.float64).reshape(-1, 4)
            if len(positions) == 1 and len(quaternions) == 1:
                quat = _quat_xyzw(quaternions[0])
                if quat is not None and np.isfinite(positions[0]).all():
                    return positions[0], quat
        live = _call_optional_arg(self.backend, "get_eef_pose", hand)
        if predecessor is None and isinstance(live, (tuple, list)) and len(live) == 2:
            quat = _quat_xyzw(live[1])
            position = np.asarray(live[0], dtype=np.float64).reshape(-1)
            if quat is not None and position.shape == (3,) and np.isfinite(
                position
            ).all():
                return position, quat
        raise RuntimeError(
            f"{hand} predicted EEF start pose is unavailable for prepared motion"
        )

    def _prepared_predicted_torso_pose(
        self,
        *,
        start_q: np.ndarray,
        predecessor: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted = (
            predecessor.get("predicted_terminal")
            if isinstance(predecessor, dict)
            else None
        )
        cached = (
            predicted.get("torso_link4")
            if isinstance(predicted, dict)
            else None
        )
        if isinstance(cached, dict):
            position = np.asarray(
                cached.get("xyz"), dtype=np.float64
            ).reshape(-1)
            quaternion = _quat_xyzw(cached.get("quat_xyzw"))
            if (
                position.shape == (3,)
                and np.isfinite(position).all()
                and quaternion is not None
            ):
                pose_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        np.concatenate([position, quaternion]),
                        dtype=np.float32,
                    ).tobytes()
                ).hexdigest()
                if cached.get("pose_sha256") != pose_digest:
                    raise RuntimeError(
                        "prepared torso predecessor pose digest changed"
                    )
                return position, quaternion
        poses = _call_optional_kw(
            self.backend,
            "torso_poses",
            q_trajectory=np.asarray(start_q, dtype=np.float32).reshape(1, -1),
        )
        if isinstance(poses, (tuple, list)) and len(poses) == 2:
            positions = np.asarray(poses[0], dtype=np.float64).reshape(-1, 3)
            quaternions = np.asarray(
                poses[1], dtype=np.float64
            ).reshape(-1, 4)
            if len(positions) == 1 and len(quaternions) == 1:
                quaternion = _quat_xyzw(quaternions[0])
                if quaternion is not None and np.isfinite(positions[0]).all():
                    return positions[0], quaternion
        live = _call_optional(self.backend, "get_torso_pose")
        if predecessor is None and isinstance(live, (tuple, list)) and len(live) == 2:
            position = np.asarray(live[0], dtype=np.float64).reshape(-1)
            quaternion = _quat_xyzw(live[1])
            if (
                position.shape == (3,)
                and np.isfinite(position).all()
                and quaternion is not None
            ):
                return position, quaternion
        raise RuntimeError(
            "torso_link4 predicted start pose is unavailable for prepared motion"
        )

    def _prepared_motion_safety_certificate_summary(
        self,
        *,
        motion_kind: str,
        plan: dict[str, Any],
        start_q: np.ndarray,
        joint_trajectory: np.ndarray,
        hand: str | None,
        target_xyz: np.ndarray | None,
        target_quat_xyzw: np.ndarray | None,
        target_torso_z_m: float | None,
    ) -> dict[str, Any]:
        """Verify and expose a bounded collision-certificate receipt."""

        q_path = np.ascontiguousarray(joint_trajectory, dtype=np.float32)
        start = np.ascontiguousarray(start_q, dtype=np.float32).reshape(-1)
        metrics = plan.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        collision = metrics.get("collision_admission")
        collision = collision if isinstance(collision, dict) else {}
        plan_sha256 = hashlib.sha256(q_path.tobytes()).hexdigest()
        start_sha256 = hashlib.sha256(start.tobytes()).hexdigest()

        try:
            if motion_kind == "torso":
                certificate = plan.get("torso_certificate")
                certificate = (
                    certificate if isinstance(certificate, dict) else {}
                )
                names_value = _call_optional(
                    self.backend,
                    "torso_joint_names",
                )
                names = tuple(str(name) for name in names_value)
                path_checks = certificate.get("path_checks")
                path_checks = (
                    path_checks if isinstance(path_checks, dict) else {}
                )
                required_path_checks = (
                    "start_pose_matches",
                    "terminal_z_matches",
                    "xy_locked",
                    "z_step_bounded",
                    "z_direction_monotonic",
                    "orientation_locked",
                )
                collision_free_waypoints = int(
                    certificate.get("collision_free_waypoints", -1)
                )
                checks = {
                    "schema": certificate.get("schema_version") == 1,
                    "kind": (
                        certificate.get("kind")
                        == "torso_link4_world_z"
                    ),
                    "target_link": (
                        certificate.get("target_link") == TORSO_LINK_NAME
                    ),
                    "plan_digest_bound": (
                        certificate.get("trajectory_sha256")
                        == plan_sha256
                    ),
                    "start_digest_bound": (
                        certificate.get("start_q_sha256")
                        == start_sha256
                    ),
                    "joint_layout_bound": (
                        len(names) == q_path.shape[1]
                        and len(names) == len(set(names))
                        and certificate.get("joint_name_layout_sha256")
                        == _joint_name_layout_sha256(names)
                    ),
                    "q_dimension": (
                        int(certificate.get("q_dimension", -1))
                        == q_path.shape[1]
                    ),
                    "active_joint_set": (
                        certificate.get("active_joint_names")
                        == list(TORSO_ACTIVE_JOINT_NAMES)
                    ),
                    "locked_joint_count": (
                        int(certificate.get("locked_joint_count", -1))
                        == 25
                    ),
                    "target_z_bound": (
                        target_torso_z_m is not None
                        and math.isclose(
                            float(
                                certificate.get(
                                    "target_z_m",
                                    math.nan,
                                )
                            ),
                            float(target_torso_z_m),
                            rel_tol=0.0,
                            abs_tol=1e-9,
                        )
                    ),
                    "world_collision_check": (
                        certificate.get("world_collision_check") is True
                        and collision.get("world_collision_check") is True
                    ),
                    "self_collision_check": (
                        certificate.get("self_collision_check") is True
                        and collision.get("self_collision_check") is True
                    ),
                    "post_interpolation_check": (
                        certificate.get("post_interpolation_check") is True
                        and collision.get("post_interpolation_check") is True
                    ),
                    "collision_admitted": (
                        collision.get("available") is True
                        and collision.get("admitted") is True
                        and collision.get("full_trajectory") is True
                    ),
                    "collision_waypoints": (
                        collision_free_waypoints >= len(q_path)
                    ),
                    "dual_attachment_collision": (
                        int(
                            certificate.get(
                                "attachment_hand_count",
                                -1,
                            )
                        )
                        == 2
                    ),
                    "path_checks": (
                        all(
                            path_checks.get(name) is True
                            for name in required_path_checks
                        )
                    ),
                }
                summary = {
                    "schema_version": 1,
                    "motion_kind": "torso",
                    "certificate_schema_version": 1,
                    "certificate_kind": "torso_link4_world_z",
                    "plan_sha256": plan_sha256,
                    "start_q_sha256": start_sha256,
                    "q_dimension": int(q_path.shape[1]),
                    "execution_waypoint_count": int(len(q_path)),
                    "collision_free_waypoint_count": (
                        collision_free_waypoints
                    ),
                    "active_dof_count": len(TORSO_ACTIVE_JOINT_NAMES),
                    "locked_joint_count": 25,
                    "attachment_hand_count": 2,
                    "checks": checks,
                }
            elif motion_kind == "eef":
                certificate = plan.get("whole_body_certificate")
                certificate = (
                    certificate if isinstance(certificate, dict) else {}
                )
                if (
                    hand not in {"left", "right"}
                    or target_xyz is None
                    or target_quat_xyzw is None
                ):
                    raise ValueError(
                        "prepared EEF safety certificate context is incomplete"
                    )
                target_xyz_sha256 = _whole_body_target_sha256(
                    target_xyz
                )
                target_quat_sha256 = hashlib.sha256(
                    np.ascontiguousarray(
                        target_quat_xyzw,
                        dtype=np.float32,
                    ).tobytes()
                ).hexdigest()
                dense_waypoint_count = int(
                    certificate.get(
                        "dense_collision_waypoint_count",
                        -1,
                    )
                )
                collision_free_waypoints = int(
                    certificate.get("collision_free_waypoints", -1)
                )
                dense_checked_waypoints = int(
                    certificate.get(
                        "dense_collision_checked_waypoint_count",
                        -1,
                    )
                )
                checks = {
                    "schema": certificate.get("schema_version") == 3,
                    "collision_lineage": (
                        certificate.get("collision_lineage_mode")
                        == (
                            "source_dense_execution_subset_"
                            "recertified_v1"
                        )
                        and certificate.get("q_encoding")
                        == "float32-c-order"
                    ),
                    "plan_digest_bound": (
                        certificate.get("trajectory_sha256")
                        == plan_sha256
                        and certificate.get(
                            "execution_trajectory_sha256"
                        )
                        == plan_sha256
                    ),
                    "start_digest_bound": (
                        certificate.get("start_q_sha256")
                        == start_sha256
                    ),
                    "waypoint_count": (
                        int(certificate.get("waypoint_count", -1))
                        == len(q_path)
                        and int(
                            certificate.get(
                                "execution_waypoint_count",
                                -1,
                            )
                        )
                        == len(q_path)
                        and certificate.get("execution_includes_start")
                        is False
                    ),
                    "q_dimension": (
                        int(certificate.get("q_dimension", -1))
                        == q_path.shape[1]
                    ),
                    "whole_body_dof_contract": (
                        int(
                            certificate.get(
                                "active_dof_count",
                                -1,
                            )
                        )
                        == 21
                        and int(
                            certificate.get(
                                "selected_eef_goal_count",
                                -1,
                            )
                        )
                        == 1
                        and int(
                            certificate.get(
                                "inactive_eef_goal_count",
                                -1,
                            )
                        )
                        == 0
                    ),
                    "selected_hand": (
                        certificate.get("selected_hand") == hand
                    ),
                    "target_position_bound": (
                        certificate.get("selected_target_xyz_sha256")
                        == target_xyz_sha256
                    ),
                    "target_orientation_bound": (
                        certificate.get(
                            "selected_target_quat_xyzw_sha256"
                        )
                        == target_quat_sha256
                    ),
                    "selected_eef_path_admitted": (
                        certificate.get("selected_eef_path_admitted")
                        is True
                    ),
                    "world_collision_check": (
                        certificate.get("world_collision_check") is True
                        and collision.get("world_collision_check") is True
                    ),
                    "self_collision_check": (
                        certificate.get("self_collision_check") is True
                        and collision.get("self_collision_check") is True
                    ),
                    "post_interpolation_check": (
                        certificate.get("post_interpolation_check") is True
                        and collision.get("post_interpolation_check") is True
                    ),
                    "collision_admitted": (
                        collision.get("available") is True
                        and collision.get("admitted") is True
                        and collision.get("full_trajectory") is True
                        and int(
                            collision.get(
                                "colliding_waypoint_count",
                                -1,
                            )
                        )
                        == 0
                    ),
                    "dense_collision_waypoints": (
                        dense_waypoint_count >= len(q_path) + 1
                        and collision_free_waypoints
                        == dense_waypoint_count
                        and dense_checked_waypoints
                        == dense_waypoint_count
                    ),
                    "dual_attachment_collision": (
                        int(
                            certificate.get(
                                "attachment_hand_count",
                                -1,
                            )
                        )
                        == 2
                    ),
                }
                summary = {
                    "schema_version": 1,
                    "motion_kind": "eef",
                    "certificate_schema_version": 3,
                    "certificate_kind": "whole_body",
                    "plan_sha256": plan_sha256,
                    "start_q_sha256": start_sha256,
                    "q_dimension": int(q_path.shape[1]),
                    "execution_waypoint_count": int(len(q_path)),
                    "collision_free_waypoint_count": (
                        collision_free_waypoints
                    ),
                    "active_dof_count": 21,
                    "selected_eef_goal_count": 1,
                    "inactive_eef_goal_count": 0,
                    "attachment_hand_count": 2,
                    "selected_hand": hand,
                    "checks": checks,
                }
            else:
                raise ValueError(
                    f"unsupported prepared safety motion kind {motion_kind!r}"
                )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"prepared {motion_kind} safety certificate is malformed"
            ) from exc

        failed_checks = [
            name for name, passed in checks.items() if passed is not True
        ]
        if failed_checks:
            raise RuntimeError(
                f"prepared {motion_kind} safety certificate failed: "
                + ", ".join(failed_checks)
            )
        summary["admitted"] = True
        return summary

    def prepare_dashboard_motion(
        self,
        target: str,
        action: str,
        predecessor_plan_id: str | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        """Plan one repeatable Dashboard motion and cache its exact trajectory."""

        target = str(target)
        action = str(action)
        predecessor_plan_id = (
            None
            if predecessor_plan_id is None
            else str(predecessor_plan_id).strip()
        )
        if action in {"observe", "open", "close"}:
            raise ValueError(f"{action} is one-shot and cannot enter the prepared queue")
        if target not in {"chassis", "left_arm", "right_arm"}:
            raise ValueError(f"unsupported Dashboard motion target {target!r}")

        with self._prepared_motion_lock:
            predecessor = (
                self._prepared_motions.get(predecessor_plan_id)
                if predecessor_plan_id is not None
                else None
            )
            if predecessor_plan_id is not None and predecessor is None:
                raise KeyError(f"unknown predecessor plan {predecessor_plan_id!r}")
            if predecessor is not None and predecessor.get("status") in {
                "discarded",
                "invalidated",
                "failed",
            }:
                raise RuntimeError("predecessor plan is not usable")
            predecessor_terminal = (
                predecessor.get("predicted_terminal")
                if isinstance(predecessor, dict)
                else None
            )

        start_q_value = (
            predecessor_terminal.get("joint_positions")
            if isinstance(predecessor_terminal, dict)
            else _call_optional(self.backend, "get_joint_positions")
        )
        start_q = np.asarray(
            _jsonable(start_q_value), dtype=np.float32
        ).reshape(-1)
        if start_q.size < 1 or not np.isfinite(start_q).all():
            raise RuntimeError("prepared motion start joint state is unavailable")
        base_pose_value = (
            predecessor_terminal.get("base_xyyaw")
            if isinstance(predecessor_terminal, dict)
            else _call_optional(self.backend, "get_base_pose")
        )
        base_pose = np.asarray(
            _jsonable(base_pose_value), dtype=np.float64
        ).reshape(-1)
        if base_pose.shape != (3,) or not np.isfinite(base_pose).all():
            raise RuntimeError("prepared motion start BASE pose is unavailable")

        started = time.monotonic()
        motion_kind: str
        hand: str | None = None
        plan: dict[str, Any]
        target_xyz: np.ndarray | None = None
        target_quat: np.ndarray | None = None
        target_torso_z: float | None = None
        wrist_calibration: dict[str, Any] | None = None
        requested_wrist_rotation_rad: float | None = None
        if target == "chassis" and action in {"up", "down"}:
            motion_kind = "torso"
            start_torso_position, start_torso_quaternion = (
                self._prepared_predicted_torso_pose(
                    start_q=start_q,
                    predecessor=predecessor,
                )
            )
            target_torso_z = float(
                start_torso_position[2]
                + (
                    TORSO_VERTICAL_STEP_M
                    if action == "up"
                    else -TORSO_VERTICAL_STEP_M
                )
            )
            planner = getattr(self.backend, "plan_torso_trajectory", None)
            if not callable(planner):
                raise RuntimeError("torso_planner_unavailable")
            plan = planner(
                target_z_m=target_torso_z,
                timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                start_q=start_q,
                start_torso_pose=(
                    start_torso_position,
                    start_torso_quaternion,
                ),
                background=bool(background),
            )
        elif target == "chassis":
            motion_kind = "base"
            planner = getattr(
                self.backend, "plan_relative_navigation_trajectory", None
            )
            if not callable(planner):
                raise RuntimeError("relative_navigation_planner_unavailable")
            base_planner_kwargs = {
                "relative_motion": self._dashboard_base_motion(action),
                "timeout_s": WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                "start_q": start_q,
                "start_base_xyyaw": base_pose,
                "background": bool(background),
                "base_attempt_timeout_cap_s": (
                    WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
                ),
                "base_solver_timeout_cap_s": (
                    WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
                ),
                "base_planning_profile": (
                    DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
                ),
            }
            if background:
                plan = planner(**base_planner_kwargs)
            else:
                with _wall_clock_deadline(
                    WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                    "prepared Dashboard BASE planning transaction",
                ):
                    plan = planner(**base_planner_kwargs)
        else:
            motion_kind = "eef"
            hand = "left" if target == "left_arm" else "right"
            start_xyz, start_quat = self._prepared_predicted_eef_pose(
                hand=hand,
                start_q=start_q,
                predecessor=predecessor,
            )
            if action in {"rotate_left", "rotate_right"}:
                wrist_calibration = self._wrist_camera_rotation_calibration(hand)
                if wrist_calibration.get("verified") is not True:
                    raise RuntimeError("wrist_calibration_unavailable")
                axis_eef = np.asarray(
                    wrist_calibration["screen_normal_axis_eef"],
                    dtype=np.float64,
                ).reshape(3)
                requested_wrist_rotation_rad = (
                    float(wrist_calibration["visual_ccw_angle_sign"])
                    * WRIST_ROTATION_STEP_RAD
                    * (1.0 if action == "rotate_left" else -1.0)
                )
                target_xyz = start_xyz.copy()
                target_quat = _quat_multiply_xyzw(
                    start_quat,
                    _axis_angle_to_quat_xyzw(
                        [*axis_eef, requested_wrist_rotation_rad]
                    ),
                )
            else:
                local_by_action = {
                    "forward": np.array([1.0, 0.0, 0.0]),
                    "backward": np.array([-1.0, 0.0, 0.0]),
                    "turn_left": np.array([0.0, 1.0, 0.0]),
                    "turn_right": np.array([0.0, -1.0, 0.0]),
                    "up": np.array([0.0, 0.0, 1.0]),
                    "down": np.array([0.0, 0.0, -1.0]),
                }
                if action not in local_by_action:
                    raise ValueError(
                        f"unsupported repeatable arm action {action!r}"
                    )
                local_delta = local_by_action[action] * EEF_TRANSLATION_STEP_M
                yaw = float(base_pose[2])
                world_delta = np.asarray(
                    [
                        math.cos(yaw) * local_delta[0]
                        - math.sin(yaw) * local_delta[1],
                        math.sin(yaw) * local_delta[0]
                        + math.cos(yaw) * local_delta[1],
                        local_delta[2],
                    ],
                    dtype=np.float64,
                )
                target_xyz = start_xyz + world_delta
                target_quat = start_quat
            planner = getattr(self.backend, "plan_whole_body_trajectory", None)
            if not callable(planner):
                raise RuntimeError("whole_body_planner_unavailable")
            plan = planner(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat,
                timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                search_profile=WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG,
                start_q=start_q,
                start_eef_pose=(start_xyz, start_quat),
                background=bool(background),
            )
        elapsed = time.monotonic() - started
        if (
            not background
            and elapsed > WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S + 1e-9
        ):
            raise TimeoutError(
                "prepared Dashboard motion exceeded its hard deadline"
            )
        if not isinstance(plan, dict) or plan.get("ok") is not True:
            reason = (
                plan.get("stop_reason", "planning_failed")
                if isinstance(plan, dict)
                else "planner_returned_non_mapping"
            )
            raise RuntimeError(f"prepared Dashboard motion failed closed: {reason}")
        q_path = np.asarray(
            _jsonable(plan.get("joint_trajectory")), dtype=np.float32
        )
        if (
            q_path.ndim != 2
            or len(q_path) < 1
            or q_path.shape[1] != start_q.size
            or not np.isfinite(q_path).all()
        ):
            raise RuntimeError("prepared Dashboard plan omitted a finite trajectory")
        safety_certificate = (
            self._prepared_motion_safety_certificate_summary(
                motion_kind=motion_kind,
                plan=plan,
                start_q=start_q,
                joint_trajectory=q_path,
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat,
                target_torso_z_m=target_torso_z,
            )
            if motion_kind in {"eef", "torso"}
            else None
        )
        if motion_kind == "base":
            plan["base_goal"] = _canonical_base_xyyaw(
                plan.get("base_goal")
            )
        predicted_q = np.ascontiguousarray(q_path[-1], dtype=np.float32)
        predicted_base = (
            _canonical_base_xyyaw(plan.get("base_goal"))
            if motion_kind == "base"
            else np.asarray(
                [
                    predicted_q[0],
                    predicted_q[1],
                    predicted_q[5],
                ],
                dtype=np.float64,
            )
        )
        eef_by_hand: dict[str, Any] = {}
        for side in ("left", "right"):
            pose_value = _call_optional_kw(
                self.backend,
                "whole_body_eef_poses",
                hand=side,
                q_trajectory=predicted_q.reshape(1, -1),
            )
            if isinstance(pose_value, (tuple, list)) and len(pose_value) == 2:
                positions = np.asarray(pose_value[0], dtype=np.float64).reshape(-1, 3)
                quaternions = np.asarray(
                    pose_value[1], dtype=np.float64
                ).reshape(-1, 4)
                if len(positions) == 1 and len(quaternions) == 1:
                    eef_by_hand[side] = {
                        "xyz": positions[0].tolist(),
                        "quat_xyzw": quaternions[0].tolist(),
                    }
        if hand is not None and hand not in eef_by_hand:
            assert target_xyz is not None and target_quat is not None
            eef_by_hand[hand] = {
                "xyz": target_xyz.tolist(),
                "quat_xyzw": target_quat.tolist(),
            }
        torso_terminal: dict[str, Any] | None = None
        torso_pose_value = _call_optional_kw(
            self.backend,
            "torso_poses",
            q_trajectory=predicted_q.reshape(1, -1),
        )
        if (
            isinstance(torso_pose_value, (tuple, list))
            and len(torso_pose_value) == 2
        ):
            torso_positions = np.asarray(
                torso_pose_value[0], dtype=np.float64
            ).reshape(-1, 3)
            torso_quaternions = np.asarray(
                torso_pose_value[1], dtype=np.float64
            ).reshape(-1, 4)
            if len(torso_positions) == 1 and len(torso_quaternions) == 1:
                torso_quaternion = _quat_xyzw(torso_quaternions[0])
                if (
                    torso_quaternion is not None
                    and np.isfinite(torso_positions[0]).all()
                ):
                    torso_terminal = {
                        "xyz": torso_positions[0].tolist(),
                        "quat_xyzw": torso_quaternion.tolist(),
                        "pose_sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                np.concatenate(
                                    [
                                        torso_positions[0],
                                        torso_quaternion,
                                    ]
                                ),
                                dtype=np.float32,
                            ).tobytes()
                        ).hexdigest(),
                    }
        if motion_kind == "torso" and torso_terminal is None:
            raise RuntimeError(
                "prepared torso plan omitted predicted torso_link4 FK"
            )

        plan_id = os.urandom(16).hex()
        predicted_terminal = {
            "joint_positions": predicted_q.tolist(),
            "joint_positions_sha256": hashlib.sha256(
                predicted_q.tobytes()
            ).hexdigest(),
            "base_xyyaw": predicted_base.tolist(),
            "eef_by_hand": eef_by_hand,
            "torso_link4": torso_terminal,
        }
        plan_metrics = (
            dict(plan.get("metrics"))
            if isinstance(plan.get("metrics"), dict)
            else {}
        )
        planning_profile = WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
        if motion_kind == "torso":
            planning_profile = DASHBOARD_PREPARED_TORSO_PLANNING_PROFILE
        fast_solver_deadline_s = (
            WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
        )
        deadline_enforcement = {
            "solver_timeout_enforced": True,
            "hard_wall_clock_enforced": not bool(background),
            "hard_wall_clock_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S
                if not background
                else None
            ),
            "soft_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S
                if background
                else None
            ),
            "soft_deadline_exceeded": bool(
                background
                and elapsed > WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S
            ),
        }
        if motion_kind == "base":
            expected_fast_cap = (
                WHOLE_BODY_DASHBOARD_JOG_FAST_TRAJOPT_DEADLINE_S
            )
            try:
                attempt_cap = float(
                    plan_metrics["attempt_timeout_cap_s"]
                )
                solver_cap = float(
                    plan_metrics["solver_timeout_cap_s"]
                )
                actual_attempt = float(
                    plan_metrics["attempt_timeout_budget_s"]
                )
                actual_solver = float(plan_metrics["solver_timeout_s"])
                plan_deadline = plan_metrics[
                    "base_solver_deadline_enforcement"
                ]
                if not isinstance(plan_deadline, dict):
                    raise TypeError("BASE solver deadline must be a mapping")
                inner_hard_deadline = plan_deadline.get(
                    "hard_wall_clock_deadline_s"
                )
                inner_soft_deadline = plan_deadline.get("soft_deadline_s")
                if inner_hard_deadline is not None:
                    inner_hard_deadline = float(inner_hard_deadline)
                if inner_soft_deadline is not None:
                    inner_soft_deadline = float(inner_soft_deadline)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "prepared BASE plan omitted deadline metrics"
                ) from exc
            planning_profile = str(
                plan_metrics.get("planning_profile", "")
            )
            deadline_contract_ok = bool(
                math.isclose(
                    attempt_cap,
                    expected_fast_cap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    solver_cap,
                    expected_fast_cap,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and 0.0 < actual_attempt <= expected_fast_cap
                and 0.0 < actual_solver <= expected_fast_cap
                and actual_solver <= actual_attempt + 1e-12
                and planning_profile
                == DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
                and plan_deadline.get("solver_timeout_enforced") is True
                and plan_deadline.get("hard_wall_clock_enforced")
                == (not bool(background))
                and (
                    (
                        not background
                        and inner_hard_deadline is not None
                        and math.isclose(
                            inner_hard_deadline,
                            actual_attempt,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                        and inner_soft_deadline is None
                    )
                    or (
                        background
                        and inner_hard_deadline is None
                        and inner_soft_deadline is not None
                        and math.isclose(
                            inner_soft_deadline,
                            actual_attempt,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                )
            )
            if not deadline_contract_ok:
                raise RuntimeError(
                    "prepared BASE plan violated its 4s solver deadline contract"
                )
            fast_solver_deadline_s = solver_cap
            deadline_enforcement.update(
                {
                    "solver_timeout_enforced": True,
                    "base_attempt_timeout_budget_s": actual_attempt,
                    "base_solver_timeout_s": actual_solver,
                    "base_attempt_timeout_cap_s": attempt_cap,
                    "base_solver_timeout_cap_s": solver_cap,
                    "base_solver_deadline_enforcement": dict(plan_deadline),
                }
            )
        execution_policy = {
            "base": PREPARED_DASHBOARD_BASE_EXECUTION_POLICY,
            "eef": PREPARED_DASHBOARD_EEF_EXECUTION_POLICY,
            "torso": PREPARED_DASHBOARD_TORSO_EXECUTION_POLICY,
        }[motion_kind]
        bounded_plan_metrics = {
            key: deepcopy(plan_metrics[key])
            for key in ("obstacle_refresh", "selected_solver_stage")
            if key in plan_metrics
        }
        solver_stages = plan_metrics.get("solver_stages")
        if isinstance(solver_stages, list):
            bounded_plan_metrics["solver_stages"] = [
                {
                    key: deepcopy(stage[key])
                    for key in (
                        "name",
                        "enable_graph",
                        "ik_only",
                        "timeout_s",
                        "elapsed_s",
                        "successes",
                        "successful_candidate_indices",
                        "certified_candidate_indices",
                    )
                    if isinstance(stage, dict) and key in stage
                }
                for stage in solver_stages
                if isinstance(stage, dict)
            ]
        metadata = {
            "schema_version": 1,
            "plan_id": plan_id,
            "target": target,
            "action": action,
            "predecessor_plan_id": predecessor_plan_id,
            "motion_kind": motion_kind,
            "status": "prepared",
            "predicted_terminal": predicted_terminal,
            "planning_profile": planning_profile,
            "execution_policy": execution_policy,
            "planning_deadline_s": (
                WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S
            ),
            "fast_solver_deadline_s": fast_solver_deadline_s,
            "background": bool(background),
            "deadline_enforcement": deadline_enforcement,
            "planning_elapsed_s": elapsed,
            "plan_metrics": bounded_plan_metrics,
            "picklable": True,
        }
        if safety_certificate is not None:
            metadata["safety_certificate"] = safety_certificate
        if requested_wrist_rotation_rad is not None:
            metadata.update(
                {
                    "requested_rotation_rad": (
                        requested_wrist_rotation_rad
                    ),
                    "visual_direction": (
                        "counterclockwise"
                        if action == "rotate_left"
                        else "clockwise"
                    ),
                    "calibration": deepcopy(wrist_calibration),
                }
            )
        entry = {
            **metadata,
            "metadata": metadata,
            "status": "prepared",
            "hand": hand,
            "execution_policy": execution_policy,
            "start_q": start_q.copy(),
            "joint_trajectory": np.ascontiguousarray(q_path, dtype=np.float32),
            "plan": plan,
            "target_xyz": None if target_xyz is None else target_xyz.copy(),
            "target_quat_xyzw": (
                None if target_quat is None else target_quat.copy()
            ),
            "target_torso_z_m": target_torso_z,
            "start_eef_xyz": (
                None
                if hand is None or target_xyz is None
                else start_xyz.copy()
            ),
            "requested_wrist_rotation_rad": requested_wrist_rotation_rad,
            "wrist_calibration": deepcopy(wrist_calibration),
            "predicted_terminal": predicted_terminal,
            "execution_command_id": None,
            "execution_result": None,
            "execution_event": threading.Event(),
        }
        with self._prepared_motion_lock:
            self._prepared_motions[plan_id] = entry
        return deepcopy(metadata)

    def execute_dashboard_motion(
        self,
        plan_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        """Consume one cached Dashboard plan with exactly-once command binding."""

        plan_id = str(plan_id).strip()
        command_id = str(command_id).strip()
        if not plan_id or not command_id:
            raise ValueError("plan_id and command_id are required")
        wait_event: threading.Event | None = None
        with self._prepared_motion_lock:
            entry = self._prepared_motions.get(plan_id)
            if entry is None:
                raise KeyError(f"unknown prepared plan {plan_id!r}")
            existing_command_id = entry.get("execution_command_id")
            if existing_command_id is not None:
                if existing_command_id != command_id:
                    raise RuntimeError(
                        "prepared plan is already bound to a different command"
                    )
                if isinstance(entry.get("execution_result"), dict):
                    return deepcopy(entry["execution_result"])
                wait_event = entry["execution_event"]
            else:
                if entry.get("status") != "prepared":
                    raise RuntimeError(
                        f"prepared plan is not executable: {entry.get('status')}"
                    )
                predecessor_id = entry.get("predecessor_plan_id")
                predecessor = (
                    self._prepared_motions.get(predecessor_id)
                    if predecessor_id
                    else None
                )
                if predecessor_id and (
                    predecessor is None
                    or predecessor.get("status") != "completed"
                    or dict(predecessor.get("execution_result") or {}).get(
                        "primitive_success"
                    )
                    is not True
                ):
                    raise RuntimeError(
                        "prepared predecessor has not completed successfully"
                    )
                entry["execution_command_id"] = command_id
                entry["status"] = "executing"
        if wait_event is not None:
            wait_event.wait()
            with self._prepared_motion_lock:
                result = self._prepared_motions[plan_id].get("execution_result")
                if not isinstance(result, dict):
                    raise RuntimeError("prepared motion execution has no terminal result")
                return deepcopy(result)

        try:
            with self._prepared_motion_lock:
                entry = self._prepared_motions[plan_id]
                q_path = np.ascontiguousarray(
                    entry["joint_trajectory"], dtype=np.float32
                )
                plan = entry["plan"]
                planned_start_q = np.ascontiguousarray(
                    entry["start_q"], dtype=np.float32
                )
            if hashlib.sha256(q_path.tobytes()).hexdigest() != hashlib.sha256(
                np.ascontiguousarray(
                    np.asarray(plan.get("joint_trajectory"), dtype=np.float32)
                ).tobytes()
            ).hexdigest():
                raise RuntimeError("prepared plan integrity check failed")
            motion_kind = str(entry["motion_kind"])
            if motion_kind == "base":
                certificate = dict(plan.get("metrics") or {}).get(
                    "base_trajectory_certificate"
                )
            elif motion_kind == "torso":
                certificate = plan.get("torso_certificate")
            else:
                certificate = plan.get("whole_body_certificate")
            certificate_start_digest = (
                certificate.get("start_q_sha256")
                if isinstance(certificate, dict)
                else None
            )
            planned_start_digest = hashlib.sha256(
                planned_start_q.tobytes()
            ).hexdigest()
            live_start_value: Any | None = None
            live_start_error: str | None = None
            try:
                live_start_value = _call_optional(
                    self.backend,
                    "get_joint_positions",
                )
            except Exception as exc:
                live_start_error = f"{type(exc).__name__}: {exc}"
            live_start_q: np.ndarray | None = None
            if live_start_value is not None:
                try:
                    live_start_q = np.ascontiguousarray(
                        np.asarray(
                            _jsonable(live_start_value),
                            dtype=np.float32,
                        ).reshape(-1)
                    )
                except (TypeError, ValueError) as exc:
                    live_start_error = f"{type(exc).__name__}: {exc}"
            live_start_available = live_start_q is not None
            live_start_shape_match = bool(
                live_start_q is not None
                and live_start_q.shape == planned_start_q.shape
            )
            live_start_finite = bool(
                live_start_q is not None
                and np.isfinite(live_start_q).all()
            )
            live_start_digest = (
                hashlib.sha256(live_start_q.tobytes()).hexdigest()
                if live_start_q is not None
                and live_start_shape_match
                and live_start_finite
                else None
            )
            planned_start_certificate_bound = bool(
                isinstance(certificate_start_digest, str)
                and certificate_start_digest == planned_start_digest
            )
            prepared_start_strict_equal = bool(
                planned_start_certificate_bound
                and live_start_digest == certificate_start_digest
            )
            max_abs_joint_drift = (
                float(
                    np.max(
                        np.abs(
                            live_start_q.astype(np.float64)
                            - planned_start_q.astype(np.float64)
                        )
                    )
                )
                if live_start_q is not None
                and live_start_shape_match
                and live_start_finite
                else None
            )
            start_metrics = {
                "live_start_checked": True,
                "live_start_equality_skipped": False,
                "prepared_start_strict_float32_equal": (
                    prepared_start_strict_equal
                ),
                "prepared_start_comparison": "float32-c-order-sha256",
                "prepared_start_tolerance": 0.0,
                "prepared_start_q_dimension": int(planned_start_q.size),
                "planned_start_q_sha256": planned_start_digest,
                "certificate_start_q_sha256": certificate_start_digest,
                "live_start_q_sha256": live_start_digest,
                "live_start_available": live_start_available,
                "live_start_shape_match": live_start_shape_match,
                "live_start_finite": live_start_finite,
                "max_abs_joint_drift": max_abs_joint_drift,
            }
            if live_start_error is not None:
                start_metrics["live_start_error"] = live_start_error
            if not prepared_start_strict_equal:
                result = primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="prepared_start_drift",
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics={
                        "prepared_plan_reused": True,
                        "prepared_plan_id": plan_id,
                        "prepared_command_id": command_id,
                        "predecessor_plan_id": entry.get(
                            "predecessor_plan_id"
                        ),
                        "replan_required": True,
                        "env_actions_sent": 0,
                        "env_actions_confirmed": 0,
                        "partial_motion": False,
                        "post_stop_env_actions": 0,
                        "collision_revalidated": False,
                        "collision_revalidation_skipped": True,
                        **start_metrics,
                    },
                )
            elif motion_kind == "base":
                assert live_start_q is not None
                plan_metrics = dict(plan.get("metrics") or {})
                result = self._execute_navigation_trajectory(
                    q_path,
                    base_goal_xyyaw=_canonical_base_xyyaw(
                        plan["base_goal"]
                    ),
                    timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                    plan_metrics=plan_metrics,
                    expected_attachments_by_hand=plan.get(
                        "expected_attachments_by_hand"
                    ),
                    prepared_start_q=live_start_q,
                    execution_policy=str(entry["execution_policy"]),
                )
            elif motion_kind == "torso":
                assert live_start_q is not None
                result = self._execute_torso_trajectory(
                    q_path,
                    target_z_m=float(entry["target_torso_z_m"]),
                    torso_certificate=plan["torso_certificate"],
                    expected_attachments_by_hand=plan.get(
                        "expected_attachments_by_hand"
                    ),
                    timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                    prepared_start_q=live_start_q,
                )
            else:
                assert live_start_q is not None
                wrist_rotation = entry.get("action") in {
                    "rotate_left",
                    "rotate_right",
                }
                result = self._execute_actions(
                    None,
                    hand=str(entry["hand"]),
                    target_xyz=np.asarray(
                        entry["target_xyz"], dtype=np.float64
                    ).reshape(3),
                    target_quat_xyzw=np.asarray(
                        entry["target_quat_xyzw"], dtype=np.float64
                    ).reshape(4),
                    position_tolerance_m=(
                        WRIST_POSITION_DRIFT_LIMIT_M
                        if wrist_rotation
                        else EEF_TERMINAL_POSITION_TOLERANCE_M
                    ),
                    orientation_tolerance_rad=(
                        math.radians(1.0)
                        if wrist_rotation
                        else EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD
                    ),
                    timeout_s=WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
                    require_pose=True,
                    hold_steps_required=0,
                    joint_trajectory=q_path,
                    expected_attachments_by_hand=plan.get(
                        "expected_attachments_by_hand"
                    ),
                    motion_scope="whole_body",
                    whole_body_certificate=plan.get("whole_body_certificate"),
                    prepared_start_q=live_start_q,
                    execution_policy=str(entry["execution_policy"]),
                )
                if wrist_rotation:
                    final_position_drift_m: float | None = None
                    if result.get("task_success") is not True:
                        final_pose = _call_optional_arg(
                            self.backend,
                            "get_eef_pose",
                            str(entry["hand"]),
                        )
                        if (
                            isinstance(final_pose, (tuple, list))
                            and len(final_pose) == 2
                        ):
                            try:
                                final_position = np.asarray(
                                    final_pose[0], dtype=np.float64
                                ).reshape(3)
                                start_position = np.asarray(
                                    entry["start_eef_xyz"],
                                    dtype=np.float64,
                                ).reshape(3)
                                if np.isfinite(final_position).all():
                                    final_position_drift_m = float(
                                        np.linalg.norm(
                                            final_position - start_position
                                        )
                                    )
                            except (TypeError, ValueError):
                                final_position_drift_m = None
                        if result.get("primitive_success") is True and (
                            final_position_drift_m is None
                            or final_position_drift_m
                            > WRIST_POSITION_DRIFT_LIMIT_M + 1e-9
                        ):
                            result = primitive_result(
                                primitive_success=False,
                                task_success=self._task_success(),
                                stop_reason=(
                                    "pose_feedback_unavailable"
                                    if final_position_drift_m is None
                                    else "wrist_position_drift"
                                ),
                                recoverable=True,
                                suggested_next_tool="observe",
                                metrics={
                                    **dict(result.get("metrics") or {}),
                                    "partial_motion": True,
                                },
                                diagnostics=result.get("diagnostics"),
                            )
                    result = self._manual_result_metrics(
                        result,
                        manual_primitive="jog_wrist",
                        manual_action=str(entry["action"]),
                        hand=str(entry["hand"]),
                        requested_rotation_rad=entry[
                            "requested_wrist_rotation_rad"
                        ],
                        visual_direction=(
                            "counterclockwise"
                            if entry["action"] == "rotate_left"
                            else "clockwise"
                        ),
                        calibration=deepcopy(
                            entry["wrist_calibration"]
                        ),
                        final_position_drift_m=final_position_drift_m,
                        position_drift_limit_m=WRIST_POSITION_DRIFT_LIMIT_M,
                        fixed_server_step=True,
                    )
            if prepared_start_strict_equal:
                result = self._manual_result_metrics(
                    result,
                    prepared_plan_reused=True,
                    prepared_plan_id=plan_id,
                    prepared_command_id=command_id,
                    replan_required=False,
                    collision_revalidated=False,
                    collision_revalidation_skipped=True,
                    tracking_policy="wide_hard_deviation",
                    **start_metrics,
                )
        except Exception as exc:
            result = primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="prepared_execution_error",
                recoverable=False,
                suggested_next_tool="observe",
                metrics={
                    "prepared_plan_reused": True,
                    "prepared_plan_id": plan_id,
                    "prepared_command_id": command_id,
                    "post_stop_env_actions": 0,
                },
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )
        with self._prepared_motion_lock:
            entry = self._prepared_motions[plan_id]
            entry["execution_result"] = deepcopy(result)
            entry["status"] = (
                "completed"
                if result.get("primitive_success") is True
                or result.get("task_success") is True
                else "failed"
            )
            entry["execution_event"].set()
        return deepcopy(result)

    def discard_dashboard_motion(self, plan_id: str) -> dict[str, Any]:
        """Discard one unexecuted cached plan and invalidate its descendants."""

        plan_id = str(plan_id).strip()
        if not plan_id:
            raise ValueError("plan_id is required")
        with self._prepared_motion_lock:
            entry = self._prepared_motions.get(plan_id)
            if entry is None:
                return {
                    "schema_version": 1,
                    "plan_id": plan_id,
                    "discarded": False,
                    "status": "unknown",
                }
            if entry.get("status") == "executing":
                raise RuntimeError("cannot discard an executing prepared plan")
            if entry.get("status") in {"completed", "failed"}:
                return {
                    "schema_version": 1,
                    "plan_id": plan_id,
                    "discarded": False,
                    "status": str(entry["status"]),
                    "reason": "prepared plan was already consumed",
                }
            already_discarded = entry.get("status") == "discarded"
            invalidated_descendants: list[str] = []
            if not already_discarded:
                entry["status"] = "discarded"
                entry["joint_trajectory"] = None
                entry["plan"] = None
                invalidated_ancestors = {plan_id}
                changed = True
                while changed:
                    changed = False
                    for descendant_id, descendant in self._prepared_motions.items():
                        if (
                            descendant.get("predecessor_plan_id")
                            in invalidated_ancestors
                            and descendant.get("status") == "prepared"
                        ):
                            descendant["status"] = "invalidated"
                            descendant["invalidated_reason"] = (
                                "predecessor_discarded"
                            )
                            descendant["joint_trajectory"] = None
                            descendant["plan"] = None
                            invalidated_ancestors.add(descendant_id)
                            invalidated_descendants.append(descendant_id)
                            changed = True
            return {
                "schema_version": 1,
                "plan_id": plan_id,
                "discarded": True,
                "already_discarded": already_discarded,
                "status": "discarded",
                "invalidated_descendant_plan_ids": invalidated_descendants,
            }

    def jog_base(
        self,
        action: str,
        *,
        timeout_s: float = WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
    ) -> dict[str, Any]:
        """Execute exactly one fixed-size body-local BASE jog."""

        action = str(action)
        if action in {"forward", "backward"}:
            motion = {
                "kind": "translation",
                "direction": action,
                "distance_m": BASE_TRANSLATION_STEP_M,
            }
            requested_step: dict[str, Any] = {
                "translation_m": BASE_TRANSLATION_STEP_M,
                "frame": "base_call_start",
            }
        elif action in {"turn_left", "turn_right"}:
            motion = {
                "kind": "rotation",
                "direction": action.removeprefix("turn_"),
                "angle_deg": math.degrees(BASE_ROTATION_STEP_RAD),
            }
            requested_step = {
                "rotation_rad": BASE_ROTATION_STEP_RAD,
                "frame": "base_call_start",
            }
        else:
            raise ValueError(
                "base jog action must be forward, backward, turn_left, or turn_right"
            )
        result = self.navigate_to(relative_motion=motion, timeout_s=timeout_s)
        return self._manual_result_metrics(
            result,
            manual_primitive="jog_base",
            manual_action=action,
            requested_step=requested_step,
            fixed_server_step=True,
        )

    def jog_eef(
        self,
        hand: str,
        action: str,
        *,
        timeout_s: float = WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
    ) -> dict[str, Any]:
        """Execute one fixed 3 cm selected-EEF jog with bounded compensation."""

        hand = _normalize_hand(hand)
        action = str(action)
        local_by_action = {
            "forward": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "backward": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "turn_left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "turn_right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
            "up": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "down": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        }
        if action not in local_by_action:
            raise ValueError(
                "EEF jog action must be forward, backward, turn_left, "
                "turn_right, up, or down"
            )
        started = time.monotonic()
        total_timeout = min(
            self._validated_timeout(timeout_s),
            WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S,
        )
        remaining_planning_budget = (
            WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S
        )
        remaining_execution_budget = WHOLE_BODY_EXECUTION_DEADLINE_S
        with _wall_clock_deadline(
            total_timeout,
            "EEF jog call-start pose transaction",
        ):
            base_pose = _call_optional(self.backend, "get_base_pose")
            eef_pose = _call_optional_arg(self.backend, "get_eef_pose", hand)
        if base_pose is None or eef_pose is None:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="pose_feedback_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                metrics={"manual_primitive": "jog_eef", "env_actions_sent": 0},
            )
        base = np.asarray(base_pose, dtype=np.float64).reshape(-1)
        current_position = np.asarray(eef_pose[0], dtype=np.float64).reshape(3)
        current_quat = _quat_xyzw(eef_pose[1])
        if (
            base.shape != (3,)
            or not np.isfinite(base).all()
            or not np.isfinite(current_position).all()
            or current_quat is None
        ):
            raise ValueError("call-start base/EEF pose is invalid")
        local_requested = local_by_action[action] * EEF_TRANSLATION_STEP_M

        def local_to_world(delta: np.ndarray) -> np.ndarray:
            yaw = float(base[2])
            return np.array(
                [
                    math.cos(yaw) * delta[0] - math.sin(yaw) * delta[1],
                    math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1],
                    delta[2],
                ],
                dtype=np.float64,
            )

        requested_world = local_to_world(local_requested)
        strict_target = current_position + requested_world
        attempts: list[dict[str, Any]] = []

        def execute_candidate(
            target: np.ndarray,
            offset: np.ndarray,
        ) -> dict[str, Any]:
            nonlocal remaining_execution_budget
            nonlocal remaining_planning_budget
            remaining = total_timeout - (time.monotonic() - started)
            if remaining <= 0.0:
                raise TimeoutError("EEF jog deadline exhausted")
            if remaining_planning_budget <= 0.0:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="planning_budget_exhausted",
                    recoverable=True,
                    suggested_next_tool=None,
                    metrics={
                        "planning_spent_s": (
                            WHOLE_BODY_DASHBOARD_JOG_PLANNING_DEADLINE_S
                        ),
                        "execution_spent_s": (
                            WHOLE_BODY_EXECUTION_DEADLINE_S
                            - remaining_execution_budget
                        ),
                        "env_actions_sent": 0,
                    },
                )
            try:
                with _wall_clock_deadline(
                    remaining,
                    "EEF jog candidate transaction",
                ):
                    candidate = self._move_to_whole_body_impl(
                        hand=hand,
                        target_xyz=target,
                        frame="world",
                        target_quat_xyzw=current_quat,
                        plan_only=False,
                        position_tolerance_m=WRIST_POSITION_DRIFT_LIMIT_M,
                        orientation_tolerance_rad=math.radians(1.0),
                        timeout_s=remaining,
                        planning_budget_s=remaining_planning_budget,
                        execution_budget_s=max(
                            remaining_execution_budget,
                            1e-9,
                        ),
                        allow_replan_checkpoint=True,
                        search_profile=(
                            WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
                        ),
                        replan_checkpoint_position_improvement_m=(
                            WHOLE_BODY_DASHBOARD_JOG_REPLAN_POSITION_IMPROVEMENT_M
                        ),
                    )
            except Exception as exc:
                candidate = self._exception_result(
                    exc,
                    suggested_next_tool=None,
                )
            candidate_metrics = dict(candidate.get("metrics") or {})
            planning_spent = float(
                candidate_metrics.get("planning_spent_s", 0.0) or 0.0
            )
            execution_spent = float(
                candidate_metrics.get("execution_spent_s", 0.0) or 0.0
            )
            remaining_planning_budget = max(
                0.0,
                remaining_planning_budget - max(planning_spent, 0.0),
            )
            remaining_execution_budget = max(
                0.0,
                remaining_execution_budget - max(execution_spent, 0.0),
            )
            attempts.append(
                {
                    "target_xyz": target.tolist(),
                    "fallback_offset": offset.tolist(),
                    "primitive_success": candidate.get("primitive_success"),
                    "task_success": candidate.get("task_success"),
                    "stop_reason": candidate.get("stop_reason"),
                    "env_actions_sent": int(
                        candidate_metrics.get(
                            "env_actions_sent", 0
                        )
                    ),
                    "planning_spent_s": planning_spent,
                    "execution_spent_s": execution_spent,
                    "remaining_planning_budget_s": (
                        remaining_planning_budget
                    ),
                    "remaining_execution_budget_s": (
                        remaining_execution_budget
                    ),
                }
            )
            return candidate

        def explicitly_unreachable(candidate: dict[str, Any]) -> bool:
            metrics = dict(candidate.get("metrics") or {})
            if int(metrics.get("env_actions_sent", 0)) != 0:
                return False
            if (
                candidate.get("partial_motion") is True
                or dict(candidate.get("metrics") or {}).get("partial_motion") is True
            ):
                return False
            if str(candidate.get("stop_reason")) == "unreachable":
                return True
            if str(candidate.get("stop_reason")) != "replan_no_progress":
                return False
            rounds = metrics.get("replan_rounds")
            return bool(
                isinstance(rounds, list)
                and rounds
                and all(
                    isinstance(item, dict)
                    and item.get("plan_ok") is False
                    and item.get("plan_stop_reason") == "unreachable"
                    for item in rounds
                )
            )

        def manual_result(
            candidate: dict[str, Any],
            *,
            target: np.ndarray,
            offset: np.ndarray,
            **extra: Any,
        ) -> dict[str, Any]:
            return self._manual_result_metrics(
                candidate,
                manual_primitive="jog_eef",
                requested_delta=local_requested.tolist(),
                requested_delta_frame="base_call_start",
                requested_delta_world=requested_world.tolist(),
                actual_target=target.tolist(),
                fallback_offset=offset.tolist(),
                candidate_attempts=attempts,
                fixed_server_step=True,
                **extra,
            )

        selected_target = strict_target
        selected_offset = np.zeros(3, dtype=np.float64)
        executed = execute_candidate(selected_target, selected_offset)
        if (
            executed.get("task_success") is True
            or executed.get("primitive_success") is True
        ):
            return manual_result(
                executed,
                target=selected_target,
                offset=selected_offset,
            )
        if not explicitly_unreachable(executed):
            return manual_result(
                executed,
                target=strict_target,
                offset=np.zeros(3, dtype=np.float64),
            )

        command_axis = int(np.argmax(np.abs(local_requested)))
        orthogonal_axes = [axis for axis in range(3) if axis != command_axis]
        last_result = executed
        last_target = strict_target
        last_offset = np.zeros(3, dtype=np.float64)
        for axis in orthogonal_axes:
            for amount in MANUAL_EEF_FALLBACK_OFFSETS_M:
                local_offset = np.zeros(3, dtype=np.float64)
                local_offset[axis] = amount
                world_offset = local_to_world(local_offset)
                target = strict_target + world_offset
                candidate = execute_candidate(target, world_offset)
                last_result = candidate
                last_target = target
                last_offset = world_offset
                if (
                    candidate.get("task_success") is True
                    or candidate.get("primitive_success") is True
                ):
                    return manual_result(
                        candidate,
                        target=target,
                        offset=world_offset,
                    )
                if not explicitly_unreachable(candidate):
                    return manual_result(
                        candidate,
                        target=target,
                        offset=world_offset,
                    )
        return manual_result(
            last_result,
            target=last_target,
            offset=last_offset,
            fallback_exhausted=True,
        )

    def _execute_torso_trajectory(
        self,
        joint_trajectory: Any,
        *,
        target_z_m: float,
        torso_certificate: dict[str, Any],
        expected_attachments_by_hand: dict[str, Any],
        timeout_s: float,
        prepared_start_q: Any | None = None,
    ) -> dict[str, Any]:
        """Execute one certified torso-only trajectory through the trunk controller."""

        started = time.monotonic()
        q_path = np.ascontiguousarray(
            np.asarray(_jsonable(joint_trajectory), dtype=np.float32)
        )
        target_z = float(target_z_m)
        if (
            q_path.ndim != 2
            or len(q_path) < 1
            or not np.isfinite(q_path).all()
            or not math.isfinite(target_z)
            or not isinstance(torso_certificate, dict)
            or not isinstance(expected_attachments_by_hand, dict)
            or set(expected_attachments_by_hand) != {"left", "right"}
        ):
            raise ValueError("torso execution contract is incomplete")
        current_q = np.ascontiguousarray(
            np.asarray(
                _jsonable(
                    prepared_start_q
                    if prepared_start_q is not None
                    else _call_optional(self.backend, "get_joint_positions")
                ),
                dtype=np.float32,
            ).reshape(-1)
        )
        trajectory_digest = hashlib.sha256(q_path.tobytes()).hexdigest()
        start_digest = hashlib.sha256(current_q.tobytes()).hexdigest()
        names_value = _call_optional(self.backend, "torso_joint_names")
        try:
            names = tuple(str(name) for name in names_value)
        except TypeError:
            names = ()
        path_checks = torso_certificate.get("path_checks")
        required_path_checks = {
            "start_pose_matches",
            "terminal_z_matches",
            "xy_locked",
            "z_step_bounded",
            "z_direction_monotonic",
            "orientation_locked",
        }
        if (
            current_q.shape != (q_path.shape[1],)
            or not np.isfinite(current_q).all()
            or len(names) != q_path.shape[1]
            or len(names) != len(set(names))
            or torso_certificate.get("kind") != "torso_link4_world_z"
            or torso_certificate.get("target_link") != TORSO_LINK_NAME
            or torso_certificate.get("trajectory_sha256") != trajectory_digest
            or torso_certificate.get("start_q_sha256") != start_digest
            or int(torso_certificate.get("q_dimension", -1))
            != q_path.shape[1]
            or torso_certificate.get("joint_name_layout_sha256")
            != _joint_name_layout_sha256(names)
            or torso_certificate.get("world_collision_check") is not True
            or torso_certificate.get("self_collision_check") is not True
            or torso_certificate.get("post_interpolation_check") is not True
            or int(torso_certificate.get("attachment_hand_count", -1)) != 2
            or torso_certificate.get("active_joint_names")
            != list(TORSO_ACTIVE_JOINT_NAMES)
            or int(torso_certificate.get("locked_joint_count", -1)) != 25
            or not isinstance(path_checks, dict)
            or not required_path_checks.issubset(path_checks)
            or any(path_checks.get(name) is not True for name in required_path_checks)
            or not math.isclose(
                float(torso_certificate.get("target_z_m", math.nan)),
                target_z,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            return self._execution_result(
                primitive_success=False,
                stop_reason="torso_certificate_invalid",
                recoverable=False,
                suggested_next_tool=None,
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics={"env_actions_sent": 0},
            )
        nominal_positions = np.asarray(
            torso_certificate.get("execution_torso_positions"),
            dtype=np.float64,
        )
        nominal_quaternions = np.asarray(
            torso_certificate.get("execution_torso_quaternions_xyzw"),
            dtype=np.float64,
        )
        certificate_start_position = np.asarray(
            torso_certificate.get("start_position"),
            dtype=np.float64,
        ).reshape(-1)
        certificate_start_quaternion = _quat_xyzw(
            torso_certificate.get("start_quaternion_xyzw")
        )
        if (
            nominal_positions.shape != (len(q_path), 3)
            or nominal_quaternions.shape != (len(q_path), 4)
            or certificate_start_position.shape != (3,)
            or certificate_start_quaternion is None
            or not np.isfinite(nominal_positions).all()
            or not np.isfinite(nominal_quaternions).all()
            or not np.isfinite(certificate_start_position).all()
            or hashlib.sha256(
                np.ascontiguousarray(
                    nominal_positions, dtype=np.float32
                ).tobytes()
            ).hexdigest()
            != torso_certificate.get("execution_torso_positions_sha256")
            or hashlib.sha256(
                np.ascontiguousarray(
                    nominal_quaternions, dtype=np.float32
                ).tobytes()
            ).hexdigest()
            != torso_certificate.get("execution_torso_quaternions_sha256")
        ):
            return self._execution_result(
                primitive_success=False,
                stop_reason="torso_certificate_invalid",
                recoverable=False,
                suggested_next_tool=None,
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics={"env_actions_sent": 0},
            )
        index = {name: offset for offset, name in enumerate(names)}
        if not set(TORSO_ACTIVE_JOINT_NAMES).issubset(index):
            raise RuntimeError("R1Pro torso execution active joints are unavailable")
        locked_indices = [
            offset
            for name, offset in index.items()
            if name not in TORSO_ACTIVE_JOINT_NAMES
        ]
        live_start_pose = _call_optional(self.backend, "get_torso_pose")
        try:
            live_start_position = np.asarray(
                live_start_pose[0], dtype=np.float64
            ).reshape(3)
            live_start_quaternion = _quat_xyzw(live_start_pose[1])
        except (TypeError, ValueError, IndexError):
            live_start_position = np.asarray([], dtype=np.float64)
            live_start_quaternion = None
        start_position_error = (
            float(
                np.linalg.norm(
                    live_start_position - certificate_start_position
                )
            )
            if live_start_position.shape == (3,)
            and np.isfinite(live_start_position).all()
            else math.inf
        )
        start_orientation_error = _quat_angle_error_rad(
            live_start_quaternion,
            certificate_start_quaternion,
        )
        if (
            start_position_error > WHOLE_BODY_EEF_START_FK_TOLERANCE_M
            or start_orientation_error is None
            or start_orientation_error > TORSO_LINK_MAX_ORIENTATION_DRIFT_RAD
        ):
            return self._execution_result(
                primitive_success=False,
                stop_reason="torso_start_pose_drift",
                recoverable=True,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=start_position_error,
                final_ori_err=start_orientation_error,
                held_steps=0,
                started=started,
                extra_metrics={
                    "env_actions_sent": 0,
                    "torso_start_position_error_m": start_position_error,
                    "torso_start_orientation_error_rad": start_orientation_error,
                },
            )
        hold_reference = _call_optional_kw(
            self.backend,
            "capture_trajectory_hold_reference",
            hand=None,
        )
        gripper_commands = (
            hold_reference.get("gripper_commands")
            if isinstance(hold_reference, dict)
            else None
        )
        if (
            len(locked_indices) != 25
            or not isinstance(gripper_commands, dict)
            or set(gripper_commands) != {"left", "right"}
        ):
            raise RuntimeError("R1Pro torso locked action reference is unavailable")
        fixed_reference = {
            "hand": None,
            "q_indices": locked_indices,
            "q_values": current_q[locked_indices].copy(),
            "gripper_commands": {
                side: float(gripper_commands[side])
                for side in ("left", "right")
            },
        }
        for side in ("left", "right"):
            live = _call_optional_arg(self.backend, "get_attached_object", side)
            matches, identity = _attachment_state_status(
                live, expected_attachments_by_hand[side], hand=side
            )
            if not matches:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="attachment_identity_mismatch",
                    recoverable=False,
                    suggested_next_tool="observe",
                    executed=0,
                    trace=[],
                    final_pos_err=None,
                    final_ori_err=None,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        "whole_body_attachment_preflight": {
                            "hand": side,
                            **identity,
                        },
                        "env_actions_sent": 0,
                    },
                )
        contact_baseline = _call_optional_kw(
            self.backend,
            "capture_whole_body_contact_baseline",
            expected_attachments_by_hand=expected_attachments_by_hand,
        )
        if (
            not isinstance(contact_baseline, dict)
            or contact_baseline.get("available") is not True
        ):
            return self._execution_result(
                primitive_success=False,
                stop_reason="contact_feedback_unavailable",
                recoverable=False,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics={"env_actions_sent": 0},
            )
        deadline = started + self._validated_timeout(timeout_s)
        trace: list[dict[str, Any]] = []
        executed = 0
        command_index = 0
        terminal_repeats = 0
        final_error: float | None = None
        final_orientation_error: float | None = None
        stop_reason = "target_tolerance_not_met"
        while command_index < len(q_path) or terminal_repeats < TERMINAL_COMMAND_LIMIT:
            if time.monotonic() >= deadline:
                stop_reason = "timeout"
                break
            path_index = min(command_index, len(q_path) - 1)
            target_q = q_path[path_index]
            action = self.backend.joint_target_to_action(
                target_q,
                hand=None,
                fixed_reference=fixed_reference,
            )
            action = validate_action_chunk(
                np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM)
            )[0]
            step_receipt = self._step_env_action(action)
            executed += 1
            terminal_outcome = _terminal_step_outcome(step_receipt)
            if terminal_outcome is not None:
                primitive_success, stop_reason = terminal_outcome
                return self._execution_result(
                    primitive_success=primitive_success,
                    stop_reason=stop_reason,
                    recoverable=False,
                    suggested_next_tool=None,
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_error,
                    final_ori_err=final_orientation_error,
                    held_steps=terminal_repeats,
                    started=started,
                    extra_metrics={
                        "manual_primitive": "jog_torso",
                        "target_link": TORSO_LINK_NAME,
                        "env_actions_sent": executed,
                    },
                )
            failure_reason: str | None = None
            failure_metrics: dict[str, Any] = {}
            for side in ("left", "right"):
                live = _call_optional_arg(self.backend, "get_attached_object", side)
                matches, identity = _attachment_state_status(
                    live, expected_attachments_by_hand[side], hand=side
                )
                if not matches:
                    failure_reason = "attachment_identity_mismatch"
                    failure_metrics["attachment_identity"] = {
                        "hand": side,
                        **identity,
                    }
                    break
            contact = _call_optional_kw(
                self.backend,
                "whole_body_contact_report",
                baseline=contact_baseline,
                expected_attachments_by_hand=expected_attachments_by_hand,
                allowed_expected_contact=None,
            )
            if failure_reason is None and (
                not isinstance(contact, dict)
                or contact.get("available") is not True
            ):
                failure_reason = "contact_feedback_unavailable"
            elif (
                failure_reason is None
                and contact.get("unexpected_contact") is True
            ):
                failure_reason = "collision_detected"
                failure_metrics["whole_body_contact"] = contact
            tracking = _call_optional_kw(
                self.backend,
                "joint_tracking_report",
                target_q=target_q,
                hand=None,
            )
            hard_tracking, hard_available = _tracking_hard_deviation_report(
                tracking if isinstance(tracking, dict) else {}
            )
            if failure_reason is None and not hard_available:
                failure_reason = "joint_feedback_unavailable"
            elif (
                failure_reason is None
                and hard_tracking.get("hard_deviation") is True
            ):
                failure_reason = "joint_tracking_deviation"
                failure_metrics["tracking_hard_deviation"] = hard_tracking
            live_q = np.asarray(
                _jsonable(_call_optional(self.backend, "get_joint_positions")),
                dtype=np.float64,
            ).reshape(-1)
            locked_drift = (
                float(
                    np.max(
                        np.abs(
                            live_q[locked_indices]
                            - current_q[locked_indices].astype(np.float64)
                        )
                    )
                )
                if live_q.shape == current_q.shape and np.isfinite(live_q).all()
                else math.inf
            )
            if (
                failure_reason is None
                and locked_drift > LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
            ):
                failure_reason = "torso_isolation_violation"
                failure_metrics["locked_joint_drift"] = locked_drift
            torso_pose = _call_optional(self.backend, "get_torso_pose")
            try:
                torso_position = np.asarray(
                    torso_pose[0], dtype=np.float64
                ).reshape(3)
                torso_quaternion = _quat_xyzw(torso_pose[1])
            except (TypeError, ValueError, IndexError):
                torso_position = np.asarray([], dtype=np.float64)
                torso_quaternion = None
            if torso_position.shape != (3,) or not np.isfinite(
                torso_position
            ).all() or torso_quaternion is None:
                if failure_reason is None:
                    failure_reason = "pose_feedback_unavailable"
                final_error = None
                final_orientation_error = None
                pose_error = math.inf
                xy_drift = math.inf
                orientation_drift = math.inf
                nominal_orientation_error = math.inf
            else:
                final_error = abs(float(torso_position[2] - target_z))
                final_orientation_error = _quat_angle_error_rad(
                    torso_quaternion,
                    certificate_start_quaternion,
                )
                pose_error = float(
                    np.linalg.norm(
                        torso_position - nominal_positions[path_index]
                    )
                )
                xy_drift = float(
                    np.linalg.norm(
                        torso_position[:2] - certificate_start_position[:2]
                    )
                )
                orientation_drift = float(
                    final_orientation_error
                    if final_orientation_error is not None
                    else math.inf
                )
                nominal_orientation_error_value = _quat_angle_error_rad(
                    torso_quaternion,
                    nominal_quaternions[path_index],
                )
                nominal_orientation_error = (
                    math.inf
                    if nominal_orientation_error_value is None
                    else float(nominal_orientation_error_value)
                )
                if (
                    failure_reason is None
                    and pose_error
                    > TORSO_LINK_TERMINAL_TOLERANCE_M + 0.005
                ):
                    failure_reason = "torso_tracking_deviation"
                    failure_metrics["torso_pose_error_m"] = pose_error
                elif (
                    failure_reason is None
                    and xy_drift > TORSO_LINK_MAX_XY_DRIFT_M + 1e-9
                ):
                    failure_reason = "torso_xy_drift"
                    failure_metrics["torso_xy_drift_m"] = xy_drift
                elif (
                    failure_reason is None
                    and (
                        orientation_drift
                        > TORSO_LINK_MAX_ORIENTATION_DRIFT_RAD + 1e-9
                        or nominal_orientation_error
                        > TORSO_LINK_MAX_ORIENTATION_DRIFT_RAD + 1e-9
                    )
                ):
                    failure_reason = "torso_orientation_drift"
                    failure_metrics.update(
                        {
                            "torso_orientation_drift_rad": orientation_drift,
                            "torso_nominal_orientation_error_rad": (
                                nominal_orientation_error
                            ),
                        }
                    )
            trace.append(
                {
                    "step": executed,
                    "waypoint_index": path_index,
                    "terminal_repeat": command_index >= len(q_path),
                    "final_position_error_m": final_error,
                    "locked_joint_drift": locked_drift,
                    "torso_pose_error_m": pose_error,
                    "torso_xy_drift_m": xy_drift,
                    "torso_orientation_drift_rad": orientation_drift,
                    "torso_nominal_orientation_error_rad": (
                        nominal_orientation_error
                    ),
                    "failure_reason": failure_reason,
                }
            )
            if failure_reason is not None:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason=failure_reason,
                    recoverable=False,
                    suggested_next_tool="observe",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_error,
                    final_ori_err=final_orientation_error,
                    held_steps=terminal_repeats,
                    started=started,
                    extra_metrics={
                        **failure_metrics,
                        "manual_primitive": "jog_torso",
                        "target_link": TORSO_LINK_NAME,
                        "partial_motion": True,
                        "env_actions_sent": executed,
                        "post_stop_env_actions": 0,
                    },
                )
            if command_index < len(q_path):
                command_index += 1
            else:
                terminal_repeats += 1
            if (
                command_index >= len(q_path)
                and final_error is not None
                and final_error <= TORSO_LINK_TERMINAL_TOLERANCE_M
                and xy_drift <= TORSO_LINK_MAX_XY_DRIFT_M
                and orientation_drift <= TORSO_LINK_MAX_ORIENTATION_DRIFT_RAD
            ):
                return self._execution_result(
                    primitive_success=True,
                    stop_reason="reached",
                    recoverable=True,
                    suggested_next_tool=None,
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_error,
                    final_ori_err=final_orientation_error,
                    held_steps=terminal_repeats,
                    started=started,
                    extra_metrics={
                        "manual_primitive": "jog_torso",
                        "target_link": TORSO_LINK_NAME,
                        "target_z_m": target_z,
                        "env_actions_sent": executed,
                        "post_stop_env_actions": 0,
                        "torso_certificate_verified_before_first_action": True,
                        "dual_attachment_checked_each_nonterminal_step": True,
                        "unexpected_contact_checked_each_nonterminal_step": True,
                    },
                )
        return self._execution_result(
            primitive_success=False,
            stop_reason=stop_reason,
            recoverable=True,
            suggested_next_tool=None,
            executed=executed,
            trace=trace,
            final_pos_err=final_error,
            final_ori_err=final_orientation_error,
            held_steps=terminal_repeats,
            started=started,
            extra_metrics={
                "manual_primitive": "jog_torso",
                "target_link": TORSO_LINK_NAME,
                "target_z_m": target_z,
                "env_actions_sent": executed,
                "post_stop_env_actions": 0,
            },
        )

    def jog_torso(
        self,
        action: str,
        *,
        timeout_s: float = WHOLE_BODY_TOTAL_DEADLINE_S,
    ) -> dict[str, Any]:
        """Move torso_link4 by one collision-certified world-Z step."""

        action = str(action)
        if action not in {"up", "down"}:
            raise ValueError("torso jog action must be up or down")
        capability = _call_optional(self.backend, "torso_jog_capability")
        delta = TORSO_VERTICAL_STEP_M if action == "up" else -TORSO_VERTICAL_STEP_M
        if (
            not isinstance(capability, dict)
            or capability.get("verified") is not True
        ):
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="torso_control_unsupported",
                recoverable=True,
                suggested_next_tool=None,
                metrics={
                    "manual_primitive": "jog_torso",
                    "manual_action": action,
                    "target_link": TORSO_LINK_NAME,
                    "requested_delta_z_m": delta,
                    "capability": capability,
                    "env_actions_sent": 0,
                    "fail_closed": True,
                },
            )
        current_pose = _call_optional(self.backend, "get_torso_pose")
        if not isinstance(current_pose, (tuple, list)) or len(current_pose) != 2:
            raise RuntimeError("torso_link4 pose feedback is unavailable")
        target_z = float(np.asarray(current_pose[0], dtype=np.float64).reshape(3)[2]) + delta
        expected_attachments = {
            side: _call_optional_arg(self.backend, "get_attached_object", side)
            for side in ("left", "right")
        }
        plan = _call_optional_kw(
            self.backend,
            "plan_torso_trajectory",
            target_z_m=target_z,
            timeout_s=min(
                float(timeout_s), WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S
            ),
            start_torso_pose=current_pose,
        )
        if not isinstance(plan, dict) or plan.get("ok") is not True:
            reason = (
                str(plan.get("stop_reason", "unreachable"))
                if isinstance(plan, dict)
                else "planner_unavailable"
            )
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason=reason,
                recoverable=True,
                suggested_next_tool=None,
                metrics={
                    **(
                        dict(plan.get("metrics") or {})
                        if isinstance(plan, dict)
                        else {}
                    ),
                    "manual_primitive": "jog_torso",
                    "manual_action": action,
                    "target_link": TORSO_LINK_NAME,
                    "requested_delta_z_m": delta,
                    "env_actions_sent": 0,
                },
            )
        result = self._execute_torso_trajectory(
            plan["joint_trajectory"],
            target_z_m=target_z,
            torso_certificate=plan["torso_certificate"],
            expected_attachments_by_hand=plan.get(
                "expected_attachments_by_hand", expected_attachments
            ),
            timeout_s=min(
                float(timeout_s), WHOLE_BODY_DASHBOARD_JOG_TOTAL_DEADLINE_S
            ),
        )
        metrics = dict(plan.get("metrics") or {})
        metrics.update(
            {
                "manual_primitive": "jog_torso",
                "manual_action": action,
                "target_link": TORSO_LINK_NAME,
                "requested_delta_z_m": delta,
                "actual_target": {
                    "torso_link4_world_z_m": target_z
                },
                "fixed_server_step": True,
            }
        )
        return self._manual_result_metrics(result, **metrics)

    def jog_wrist(
        self,
        hand: str,
        action: str,
        *,
        timeout_s: float = WHOLE_BODY_TOTAL_DEADLINE_S,
    ) -> dict[str, Any]:
        """Rotate one wrist by a calibrated visual 5 degrees."""

        hand = _normalize_hand(hand)
        action = str(action)
        if action not in {"rotate_left", "rotate_right"}:
            raise ValueError("wrist jog action must be rotate_left or rotate_right")
        calibration = self._wrist_camera_rotation_calibration(hand)
        if calibration.get("verified") is not True:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="wrist_calibration_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                metrics={
                    "manual_primitive": "jog_wrist",
                    "calibration": calibration,
                    "env_actions_sent": 0,
                },
            )
        current = _call_optional_arg(self.backend, "get_eef_pose", hand)
        if current is None:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="pose_feedback_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                metrics={"manual_primitive": "jog_wrist", "env_actions_sent": 0},
            )
        start_position = np.asarray(current[0], dtype=np.float64).reshape(3)
        start_quat = _quat_xyzw(current[1])
        assert start_quat is not None
        axis_eef = np.asarray(
            calibration["screen_normal_axis_eef"], dtype=np.float64
        ).reshape(3)
        counterclockwise = action == "rotate_left"
        signed_angle = (
            float(calibration["visual_ccw_angle_sign"])
            * WRIST_ROTATION_STEP_RAD
            * (1.0 if counterclockwise else -1.0)
        )
        target_quat = _quat_multiply_xyzw(
            start_quat,
            _axis_angle_to_quat_xyzw([*axis_eef, signed_angle]),
        )
        result = self.move_to(
            hand=hand,
            target_xyz=start_position,
            target_quat_xyzw=target_quat,
            plan_only=False,
            position_tolerance_m=WRIST_POSITION_DRIFT_LIMIT_M,
            orientation_tolerance_rad=math.radians(1.0),
            timeout_s=timeout_s,
        )
        drift_m: float | None = None
        if result.get("task_success") is not True:
            final = _call_optional_arg(self.backend, "get_eef_pose", hand)
            if final is not None:
                try:
                    final_position = np.asarray(
                        final[0], dtype=np.float64
                    ).reshape(-1)
                except (TypeError, ValueError):
                    final_position = np.asarray([], dtype=np.float64)
                if final_position.shape == (3,) and np.isfinite(
                    final_position
                ).all():
                    drift_m = float(
                        np.linalg.norm(final_position - start_position)
                    )
        if result.get("task_success") is not True and result.get(
            "primitive_success"
        ) is True and (
            drift_m is None or drift_m > WRIST_POSITION_DRIFT_LIMIT_M + 1e-9
        ):
            result = primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason=(
                    "pose_feedback_unavailable"
                    if drift_m is None
                    else "wrist_position_drift"
                ),
                recoverable=True,
                suggested_next_tool="observe",
                metrics={
                    **dict(result.get("metrics") or {}),
                    "partial_motion": True,
                },
                diagnostics=result.get("diagnostics"),
            )
        return self._manual_result_metrics(
            result,
            manual_primitive="jog_wrist",
            manual_action=action,
            hand=hand,
            requested_rotation_rad=signed_angle,
            visual_direction=(
                "counterclockwise" if counterclockwise else "clockwise"
            ),
            calibration=calibration,
            final_position_drift_m=drift_m,
            position_drift_limit_m=WRIST_POSITION_DRIFT_LIMIT_M,
            fixed_server_step=True,
        )

    def rotate_wrist_step(
        self,
        hand: str,
        action: str,
        *,
        timeout_s: float = WHOLE_BODY_TOTAL_DEADLINE_S,
    ) -> dict[str, Any]:
        """Compatibility spelling for the Dashboard wrist jog."""

        return self.jog_wrist(hand, action, timeout_s=timeout_s)

    def set_gripper(
        self,
        hand: str,
        opening: float,
        *,
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        """Set and latch one gripper without retreat or repeated RPC calls."""

        hand = _normalize_hand(hand)
        value = float(opening)
        if not math.isfinite(value) or value not in {0.0, 1.0}:
            raise ValueError("opening must be exactly 0.0 (close) or 1.0 (open)")
        expected_attachment = (
            _call_optional_arg(self.backend, "get_attached_object", hand)
            if value == 0.0
            else None
        )
        result = self._gripper_command(
            hand,
            opening=value,
            timeout_s=self._validated_timeout(timeout_s),
            expected_attachment=expected_attachment,
            require_attachment=expected_attachment is not None,
        )
        latch = getattr(self.env, "_gripper_latch", None)
        return self._manual_result_metrics(
            result,
            manual_primitive="set_gripper",
            hand=hand,
            opening=value,
            retreat_executed=False,
            network_primitive_calls=1,
            held_object_close=expected_attachment is not None,
            gripper_latch=(
                float(latch.get(hand))
                if isinstance(latch, dict) and hand in latch
                else None
            ),
        )

    def _move_to_whole_body_impl(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str,
        target_quat_xyzw: Any | None,
        plan_only: bool,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        timeout_s: float,
        planning_budget_s: float = WHOLE_BODY_PLANNING_DEADLINE_S,
        execution_budget_s: float = WHOLE_BODY_EXECUTION_DEADLINE_S,
        allow_replan_checkpoint: bool = False,
        search_profile: str = WHOLE_BODY_SEARCH_PROFILE_DEFAULT,
        replan_checkpoint_position_improvement_m: float = (
            WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
        ),
    ) -> dict[str, Any]:
        started = time.monotonic()
        profile = _whole_body_search_profile(search_profile)
        total_budget = min(
            self._validated_timeout(timeout_s), WHOLE_BODY_TOTAL_DEADLINE_S
        )
        planning_budget = min(
            self._validated_timeout(planning_budget_s),
            float(profile["planning_deadline_s"]),
            total_budget,
        )
        execution_budget = min(
            self._validated_timeout(execution_budget_s),
            WHOLE_BODY_EXECUTION_DEADLINE_S,
            total_budget,
        )
        deadline = started + total_budget
        hand = _normalize_hand(hand)
        target = self._world_target(target_xyz, frame=frame)
        explicit_quat = _quat_xyzw(target_quat_xyzw)
        expected_attachments = {
            side: _call_optional_arg(self.backend, "get_attached_object", side)
            for side in ("left", "right")
        }
        effective_quat = explicit_quat
        if effective_quat is None and expected_attachments[hand] is not None:
            call_start_pose = _call_optional_arg(self.backend, "get_eef_pose", hand)
            if call_start_pose is None:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="pose_feedback_unavailable",
                    recoverable=False,
                    suggested_next_tool="observe",
                    metrics={"env_actions_sent": 0},
                )
            effective_quat = _quat_xyzw(call_start_pose[1])

        if self._task_success():
            return primitive_result(
                primitive_success=True,
                task_success=True,
                stop_reason="official_task_success",
                recoverable=False,
                suggested_next_tool=None,
                metrics={
                    "motion_scope": "whole_body",
                    "env_actions_sent": 0,
                    "planning_attempts": 0,
                    "post_success_env_actions": 0,
                },
            )

        plan_whole_body = getattr(self.backend, "plan_whole_body_trajectory", None)
        if not callable(plan_whole_body):
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="whole_body_planner_unavailable",
                recoverable=True,
                suggested_next_tool=None,
                metrics={
                    "motion_scope": "whole_body",
                    "active_dof_count": 21,
                },
            )

        current_pose = _call_optional_arg(self.backend, "get_eef_pose", hand)
        if current_pose is not None and not plan_only:
            initial_position_error = float(
                np.linalg.norm(
                    np.asarray(current_pose[0], dtype=np.float64).reshape(3) - target
                )
            )
            initial_orientation_error = (
                _quat_angle_error_rad(current_pose[1], effective_quat)
                if effective_quat is not None
                else None
            )
            if (
                initial_position_error <= float(position_tolerance_m)
                and (
                    initial_orientation_error is None
                    or initial_orientation_error <= float(orientation_tolerance_rad)
                )
            ):
                return primitive_result(
                    primitive_success=True,
                    task_success=self._task_success(),
                    stop_reason="reached",
                    recoverable=True,
                    suggested_next_tool=None,
                    metrics={
                        "motion_scope": "whole_body",
                        "active_dof_count": 21,
                        "env_actions_sent": 0,
                        "planning_attempts": 0,
                        "final_position_error_m": initial_position_error,
                        "final_orientation_error_rad": initial_orientation_error,
                        "local_target_already_satisfied": True,
                    },
                )

        planning_spent_s = 0.0
        execution_spent_s = 0.0
        previous_checkpoint_position_error: float | None = None
        planning_retry_used = False
        cumulative_env_actions_sent = 0
        cumulative_partial_motion = False
        rounds: list[dict[str, Any]] = []
        last_plan_metrics: dict[str, Any] = {}
        while True:
            if self._task_success():
                return primitive_result(
                    primitive_success=True,
                    task_success=True,
                    stop_reason="official_task_success",
                    recoverable=False,
                    suggested_next_tool=None,
                    metrics={
                        **last_plan_metrics,
                        "motion_scope": "whole_body",
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                        "post_success_env_actions": 0,
                    },
                )
            for side in ("left", "right"):
                live = _call_optional_arg(self.backend, "get_attached_object", side)
                matches, identity = _attachment_state_status(
                    live, expected_attachments[side], hand=side
                )
                if not matches:
                    return primitive_result(
                        primitive_success=False,
                        task_success=self._task_success(),
                        stop_reason="attachment_identity_mismatch",
                        recoverable=False,
                        suggested_next_tool="observe",
                        metrics={
                            **last_plan_metrics,
                            "attachment_identity": {"hand": side, **identity},
                            "planning_spent_s": planning_spent_s,
                            "execution_spent_s": execution_spent_s,
                            "replan_rounds": rounds,
                            "env_actions_sent": cumulative_env_actions_sent,
                            "partial_motion": cumulative_partial_motion,
                        },
                    )

            total_remaining = deadline - time.monotonic()
            plan_allowance = min(
                planning_budget - planning_spent_s,
                total_remaining,
            )
            if plan_allowance <= 0.0:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="planning_budget_exhausted",
                    recoverable=True,
                    suggested_next_tool=None,
                    metrics={
                        **last_plan_metrics,
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "total_deadline_s": total_budget,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                )

            plan_started = time.monotonic()
            try:
                with _wall_clock_deadline(
                    plan_allowance,
                    "whole-body planning transaction",
                ):
                    plan = plan_whole_body(
                        hand=hand,
                        target_xyz=target,
                        target_quat_xyzw=effective_quat,
                        timeout_s=plan_allowance,
                        attached_obj=expected_attachments[hand],
                        search_profile=str(profile["name"]),
                    )
            except TimeoutError as exc:
                planning_spent_s += time.monotonic() - plan_started
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="planning_budget_exhausted",
                    recoverable=True,
                    suggested_next_tool=None,
                    metrics={
                        **last_plan_metrics,
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "total_deadline_s": total_budget,
                        "planning_hard_limit_s": float(
                            profile["planning_deadline_s"]
                        ),
                        "call_planning_budget_s": planning_budget,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                    diagnostics={"error": f"{type(exc).__name__}: {exc}"},
                )
            planning_elapsed = time.monotonic() - plan_started
            planning_spent_s += planning_elapsed
            plan_metrics = (
                dict(plan.get("metrics"))
                if isinstance(plan.get("metrics"), dict)
                else {}
            )
            last_plan_metrics = plan_metrics
            round_report: dict[str, Any] = {
                "round": len(rounds) + 1,
                "planning_allowance_s": plan_allowance,
                "planning_elapsed_s": planning_elapsed,
                "plan_ok": plan.get("ok") is True,
                "plan_stop_reason": plan.get("stop_reason"),
            }
            rounds.append(round_report)
            if not plan.get("ok"):
                stop_reason = str(plan.get("stop_reason", "whole_body_plan_failed"))
                eligible_plan_failure = stop_reason in {
                    "unreachable",
                    "timeout",
                    "collision_admission_failed",
                    "eef_path_admission_failed",
                }
                if eligible_plan_failure and not planning_retry_used:
                    # The first qualifying planning failure gets one fresh,
                    # unconditional retry without moving the simulator.
                    # A second planning-only failure exhausts that independent
                    # retry; execution checkpoint progress is not consulted.
                    planning_retry_used = True
                    round_report["eligible_plan_failure"] = {
                        "stop_reason": stop_reason,
                        "env_actions_sent": 0,
                        "unconditional_replan": True,
                    }
                    continue
                if eligible_plan_failure and planning_retry_used:
                    round_report["eligible_plan_failure"] = {
                        "stop_reason": stop_reason,
                        "env_actions_sent": 0,
                        "unconditional_replan": False,
                        "physical_progress_available": False,
                    }
                    return primitive_result(
                        primitive_success=False,
                        task_success=self._task_success(),
                        stop_reason="replan_no_progress",
                        recoverable=True,
                        suggested_next_tool=None,
                        metrics={
                            **plan_metrics,
                            "planning_spent_s": planning_spent_s,
                            "execution_spent_s": execution_spent_s,
                            "replan_rounds": rounds,
                            "last_plan_stop_reason": stop_reason,
                            "post_stop_action_policy": "no_additional_env_action",
                            "post_stop_env_actions": 0,
                            "env_actions_sent": cumulative_env_actions_sent,
                            "partial_motion": cumulative_partial_motion,
                        },
                        diagnostics=plan,
                    )
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=stop_reason,
                    recoverable=stop_reason
                    in {
                        "unreachable",
                        "timeout",
                        "planner_unavailable",
                        "collision_admission_failed",
                        "eef_path_admission_failed",
                    },
                    suggested_next_tool=None,
                    metrics={
                        **plan_metrics,
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                    diagnostics=plan,
                )

            metrics = plan.get("metrics")
            collision = (
                metrics.get("collision_admission")
                if isinstance(metrics, dict)
                else None
            )
            certificate = plan.get("whole_body_certificate")
            if (
                not isinstance(collision, dict)
                or collision.get("available") is not True
                or collision.get("admitted") is not True
                or collision.get("world_collision_check") is not True
                or collision.get("self_collision_check") is not True
                or collision.get("obstacle_update") is not True
                or collision.get("full_trajectory") is not True
                or collision.get("post_interpolation_check") is not True
                or not isinstance(certificate, dict)
            ):
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="collision_admission_unavailable",
                    recoverable=False,
                    suggested_next_tool=None,
                    metrics={
                        **dict(metrics or {}),
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                    diagnostics=plan,
                )
            for side in ("left", "right"):
                live = _call_optional_arg(self.backend, "get_attached_object", side)
                matches, identity = _attachment_state_status(
                    live, expected_attachments[side], hand=side
                )
                if not matches:
                    return primitive_result(
                        primitive_success=False,
                        task_success=self._task_success(),
                        stop_reason="attachment_identity_mismatch",
                        recoverable=False,
                        suggested_next_tool="observe",
                        metrics={
                            **dict(metrics or {}),
                            "attachment_identity": {"hand": side, **identity},
                            "planning_spent_s": planning_spent_s,
                            "execution_spent_s": execution_spent_s,
                            "replan_rounds": rounds,
                            "env_actions_sent": cumulative_env_actions_sent,
                            "partial_motion": cumulative_partial_motion,
                        },
                    )
            if plan_only:
                return primitive_result(
                    primitive_success=True,
                    task_success=self._task_success(),
                    stop_reason="plan_ready",
                    recoverable=True,
                    metrics={
                        **dict(metrics or {}),
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": 0.0,
                        "replan_rounds": rounds,
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                )
            trajectory = plan.get("joint_trajectory")
            if trajectory is None:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="whole_body_trajectory_unavailable",
                    recoverable=False,
                    metrics={
                        **dict(metrics or {}),
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                )

            total_remaining = deadline - time.monotonic()
            execution_allowance = min(
                execution_budget - execution_spent_s,
                total_remaining,
            )
            if execution_allowance <= 0.0:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="execution_budget_exhausted",
                    recoverable=True,
                    suggested_next_tool=None,
                    metrics={
                        **dict(metrics or {}),
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "partial_motion": cumulative_partial_motion,
                    },
                )

            execution_started = time.monotonic()
            execution_action_counter = {"attempted": 0, "confirmed": 0}
            try:
                with _wall_clock_deadline(
                    execution_allowance,
                    "whole-body execution transaction",
                ):
                    execution = self._execute_actions(
                        None,
                        hand=hand,
                        target_xyz=target,
                        target_quat_xyzw=effective_quat,
                        position_tolerance_m=float(position_tolerance_m),
                        orientation_tolerance_rad=float(
                            orientation_tolerance_rad
                        ),
                        timeout_s=execution_allowance,
                        require_pose=True,
                        hold_steps_required=1,
                        joint_trajectory=trajectory,
                        expected_attachments_by_hand=expected_attachments,
                        motion_scope="whole_body",
                        whole_body_certificate=certificate,
                        action_counter=execution_action_counter,
                        allow_replan_checkpoint=allow_replan_checkpoint,
                        replan_checkpoint_position_improvement_m=(
                            replan_checkpoint_position_improvement_m
                        ),
                    )
            except TimeoutError as exc:
                execution_elapsed = time.monotonic() - execution_started
                execution_spent_s += execution_elapsed
                timed_out_action_attempts = int(
                    execution_action_counter["attempted"]
                )
                timed_out_confirmed_actions = int(
                    execution_action_counter["confirmed"]
                )
                cumulative_env_actions_sent += timed_out_action_attempts
                cumulative_partial_motion = bool(
                    cumulative_partial_motion or timed_out_action_attempts > 0
                )
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="execution_budget_exhausted",
                    recoverable=False,
                    suggested_next_tool=None,
                    metrics={
                        **dict(metrics or {}),
                        "planning_spent_s": planning_spent_s,
                        "execution_spent_s": execution_spent_s,
                        "replan_rounds": rounds,
                        "total_deadline_s": total_budget,
                        "execution_hard_limit_s": (
                            WHOLE_BODY_EXECUTION_DEADLINE_S
                        ),
                        "call_execution_budget_s": execution_budget,
                        "env_actions_sent": cumulative_env_actions_sent,
                        "env_actions_sent_exact": (
                            timed_out_action_attempts
                            == timed_out_confirmed_actions
                        ),
                        "env_actions_confirmed": (
                            cumulative_env_actions_sent
                            - timed_out_action_attempts
                            + timed_out_confirmed_actions
                        ),
                        "timed_out_round_action_attempts": (
                            timed_out_action_attempts
                        ),
                        "timed_out_round_confirmed_actions": (
                            timed_out_confirmed_actions
                        ),
                        "partial_motion": cumulative_partial_motion,
                    },
                    diagnostics={"error": f"{type(exc).__name__}: {exc}"},
                )
            execution_elapsed = time.monotonic() - execution_started
            execution_spent_s += execution_elapsed
            execution_metrics = (
                dict(execution.get("metrics"))
                if isinstance(execution.get("metrics"), dict)
                else {}
            )
            round_env_actions = max(
                int(execution_metrics.get("env_actions_sent", 0) or 0),
                int(execution_metrics.get("executed_waypoints", 0) or 0),
            )
            cumulative_env_actions_sent += round_env_actions
            cumulative_partial_motion = bool(
                cumulative_partial_motion
                or execution_metrics.get("partial_motion") is True
                or (
                    execution.get("primitive_success") is not True
                    and round_env_actions > 0
                )
            )
            round_report.update(
                {
                    "execution_allowance_s": execution_allowance,
                    "execution_elapsed_s": execution_elapsed,
                    "execution_stop_reason": execution.get("stop_reason"),
                    "primitive_success": execution.get("primitive_success"),
                    "task_success": execution.get("task_success"),
                    "env_actions_sent": round_env_actions,
                }
            )
            combined_metrics = {
                **dict(metrics or {}),
                **execution_metrics,
                "motion_scope": "whole_body",
                "planning_spent_s": planning_spent_s,
                "execution_spent_s": execution_spent_s,
                "total_deadline_s": total_budget,
                "planning_hard_limit_s": float(profile["planning_deadline_s"]),
                "execution_hard_limit_s": WHOLE_BODY_EXECUTION_DEADLINE_S,
                "call_planning_budget_s": planning_budget,
                "call_execution_budget_s": execution_budget,
                "whole_body_search_profile": str(profile["name"]),
                "replan_checkpoint_trigger_position_improvement_m": float(
                    replan_checkpoint_position_improvement_m
                ),
                "replan_rounds": rounds,
                "env_actions_sent": cumulative_env_actions_sent,
                "partial_motion": bool(
                    cumulative_partial_motion
                    and execution.get("primitive_success") is not True
                    and execution.get("task_success") is not True
                ),
                "motion_occurred_before_replan": cumulative_partial_motion,
                "whole_body_execution": {
                    "available": True,
                    "ok": bool(execution.get("primitive_success")),
                    "collision_certificate_verified_before_first_action": True,
                    "dual_attachment_checked_each_nonterminal_step": True,
                    "unexpected_contact_checked_each_nonterminal_step": True,
                    "raw_success_checked_after_each_action": True,
                    "raw_success_preempts_post_step_safety_checks": True,
                    "feedback_gated_waypoints": True,
                },
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            if execution.get("primitive_success") is True or execution.get(
                "task_success"
            ) is True:
                return primitive_result(
                    primitive_success=bool(execution["primitive_success"]),
                    task_success=self._task_success(),
                    stop_reason=str(execution["stop_reason"]),
                    recoverable=bool(execution["recoverable"]),
                    suggested_next_tool=execution.get("suggested_next_tool"),
                    metrics=combined_metrics,
                    diagnostics=execution.get("diagnostics"),
                )

            stop_reason = str(execution.get("stop_reason", "execution_failed"))
            replan_checkpoint = execution_metrics.get(
                "whole_body_replan_checkpoint"
            )
            checkpoint_progress_proved = False
            if stop_reason == "whole_body_replan_checkpoint":
                checkpoint_commanded_index = (
                    replan_checkpoint.get("commanded_index")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_post_step_index = (
                    replan_checkpoint.get("post_step_index")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_start_error = (
                    replan_checkpoint.get("call_start_position_error_m")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_final_error = (
                    replan_checkpoint.get("final_position_error_m")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_improvement = (
                    replan_checkpoint.get("position_improvement_m")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_minimum_improvement = (
                    replan_checkpoint.get("minimum_position_improvement_m")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_trigger_improvement = (
                    replan_checkpoint.get("trigger_position_improvement_m")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_q_sha256 = (
                    replan_checkpoint.get("q_sha256")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_trajectory_sha256 = (
                    replan_checkpoint.get("trajectory_sha256")
                    if isinstance(replan_checkpoint, dict)
                    else None
                )
                checkpoint_q_matches_trajectory = False
                try:
                    checkpoint_trajectory = np.ascontiguousarray(
                        trajectory, dtype=np.float32
                    )
                    if (
                        isinstance(checkpoint_commanded_index, int)
                        and checkpoint_trajectory.ndim == 2
                        and 0
                        <= checkpoint_commanded_index
                        < checkpoint_trajectory.shape[0] - 1
                    ):
                        checkpoint_q_matches_trajectory = (
                            hashlib.sha256(
                                np.ascontiguousarray(
                                    checkpoint_trajectory[
                                        checkpoint_commanded_index
                                    ],
                                    dtype=np.float32,
                                ).tobytes()
                            ).hexdigest()
                            == checkpoint_q_sha256
                        )
                except (TypeError, ValueError):
                    checkpoint_q_matches_trajectory = False
                checkpoint_progress_proved = bool(
                    isinstance(replan_checkpoint, dict)
                    and allow_replan_checkpoint
                    and round_env_actions > 0
                    and all(
                        value is None for value in expected_attachments.values()
                    )
                    and replan_checkpoint.get("eef_pose_settled") is True
                    and replan_checkpoint.get("joint_waypoint_reached") is True
                    and isinstance(checkpoint_commanded_index, int)
                    and checkpoint_commanded_index >= 0
                    and isinstance(checkpoint_post_step_index, int)
                    and checkpoint_post_step_index
                    == checkpoint_commanded_index + 1
                    and isinstance(checkpoint_q_sha256, str)
                    and len(checkpoint_q_sha256) == 64
                    and checkpoint_q_matches_trajectory
                    and isinstance(checkpoint_trajectory_sha256, str)
                    and checkpoint_trajectory_sha256
                    == certificate.get("trajectory_sha256")
                    and isinstance(checkpoint_start_error, (int, float))
                    and math.isfinite(float(checkpoint_start_error))
                    and isinstance(checkpoint_final_error, (int, float))
                    and math.isfinite(float(checkpoint_final_error))
                    and isinstance(checkpoint_improvement, (int, float))
                    and math.isfinite(float(checkpoint_improvement))
                    and abs(
                        float(checkpoint_start_error)
                        - float(checkpoint_final_error)
                        - float(checkpoint_improvement)
                    )
                    <= 1e-9
                    and float(checkpoint_improvement)
                    + WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                    >= WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
                    and isinstance(
                        checkpoint_minimum_improvement, (int, float)
                    )
                    and math.isclose(
                        float(checkpoint_minimum_improvement),
                        WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and isinstance(
                        checkpoint_trigger_improvement, (int, float)
                    )
                    and math.isclose(
                        float(checkpoint_trigger_improvement),
                        float(replan_checkpoint_position_improvement_m),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    and float(checkpoint_improvement)
                    + WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                    >= float(replan_checkpoint_position_improvement_m)
                    and isinstance(
                        execution_metrics.get("final_position_error_m"),
                        (int, float),
                    )
                    and math.isfinite(
                        float(execution_metrics["final_position_error_m"])
                    )
                    and abs(
                        float(checkpoint_final_error)
                        - float(execution_metrics["final_position_error_m"])
                    )
                    <= 1e-9
                    and execution_metrics.get("post_stop_env_actions") == 0
                )
                if not checkpoint_progress_proved:
                    return primitive_result(
                        primitive_success=False,
                        task_success=self._task_success(),
                        stop_reason="whole_body_replan_checkpoint_invalid",
                        recoverable=False,
                        suggested_next_tool="observe",
                        metrics={
                            **combined_metrics,
                            "post_stop_action_policy": (
                                "no_additional_env_action"
                            ),
                            "post_stop_env_actions": 0,
                        },
                        diagnostics=execution.get("diagnostics"),
                    )
                round_report["whole_body_replan_checkpoint"] = dict(
                    replan_checkpoint
                )
            if stop_reason != "whole_body_replan_checkpoint":
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=stop_reason,
                    recoverable=bool(execution.get("recoverable", False)),
                    suggested_next_tool=execution.get("suggested_next_tool"),
                    metrics=combined_metrics,
                    diagnostics=execution.get("diagnostics"),
                )

            position_error = execution_metrics.get("final_position_error_m")
            current_failure = {
                "position_error_m": (
                    float(position_error)
                    if isinstance(position_error, (int, float))
                    and math.isfinite(float(position_error))
                    else None
                ),
            }
            round_report["eligible_replan_failure"] = current_failure
            current_position_error = current_failure["position_error_m"]
            assert current_position_error is not None
            if previous_checkpoint_position_error is None:
                previous_checkpoint_position_error = current_position_error
                round_report["eligible_replan_failure"] = {
                    **current_failure,
                    "first_checkpoint": True,
                }
                continue

            position_improved = bool(
                previous_checkpoint_position_error
                - current_position_error
                + WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                >= WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
            )
            round_report["eligible_replan_failure"] = {
                **current_failure,
                "first_checkpoint": False,
            }
            round_report["replan_progress"] = {
                "baseline_available": True,
                "position_improved_at_least_2mm": position_improved,
                "fresh_plan_tracking_comparison_used": False,
            }
            if not position_improved:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="replan_no_progress",
                    recoverable=True,
                    suggested_next_tool=None,
                    metrics={
                        **combined_metrics,
                        "post_stop_action_policy": "no_additional_env_action",
                        "post_stop_env_actions": 0,
                    },
                    diagnostics=execution.get("diagnostics"),
                )
            previous_checkpoint_position_error = current_position_error

    @_planner_tool("move_to")
    def move_to(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        target_quat_xyzw: Any | None = None,
        plan_only: bool = False,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = math.radians(5.0),
        timeout_s: float = WHOLE_BODY_TOTAL_DEADLINE_S,
    ) -> dict[str, Any]:
        try:
            return self._move_to_whole_body_impl(
                hand=hand,
                target_xyz=target_xyz,
                frame=frame,
                target_quat_xyzw=target_quat_xyzw,
                plan_only=plan_only,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
                timeout_s=timeout_s,
                allow_replan_checkpoint=True,
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool=None)

    def _move_to_composite_stage(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: Any | None,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        timeout_s: float,
        hold_steps_required: int,
        contact_target_xyz: np.ndarray | None = None,
        expected_attachment: Any = None,
        require_attachment: bool = False,
        isolation_reference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a collision-certified whole-body stage inside a composite tool."""

        started = time.monotonic()
        try:
            del isolation_reference
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            target = _as_xyz(target_xyz)
            quat = _quat_xyzw(target_quat_xyzw)
            reach_metrics: dict[str, Any] = {
                "reachability_stage": "whole_body_full_trajectory",
                "redundant_ik_probe_skipped": True,
            }
            expected_attachments = {
                side: _call_optional_arg(self.backend, "get_attached_object", side)
                for side in ("left", "right")
            }
            if require_attachment:
                selected_matches, _identity = _attachment_state_status(
                    expected_attachments[hand],
                    expected_attachment,
                    hand=hand,
                )
                if not selected_matches:
                    raise RuntimeError(
                        "selected attachment changed before composite whole-body stage"
                    )
            plan_whole_body = getattr(self.backend, "plan_whole_body_trajectory", None)
            if not callable(plan_whole_body):
                raise RuntimeError("whole-body cuRobo planner is unavailable")
            plan = plan_whole_body(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                timeout_s=self._remaining_s(deadline),
                attached_obj=expected_attachments[hand],
            )
            if not plan.get("ok"):
                stop_reason = str(plan.get("stop_reason", "arm_plan_failed"))
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=stop_reason,
                    recoverable=stop_reason in {"unreachable", "planner_unavailable"},
                    suggested_next_tool=None,
                    metrics={**reach_metrics, **plan.get("metrics", {})},
                    diagnostics=plan,
                )
            execution = self._execute_actions(
                validate_action_chunk(plan["actions"])
                if plan.get("actions") is not None
                else None,
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                position_tolerance_m=float(position_tolerance_m),
                orientation_tolerance_rad=float(orientation_tolerance_rad),
                timeout_s=self._remaining_s(deadline),
                require_pose=True,
                hold_steps_required=max(1, int(hold_steps_required)),
                contact_target_xyz=contact_target_xyz,
                joint_trajectory=plan.get("joint_trajectory"),
                expected_attachment=expected_attachment,
                require_attachment=require_attachment,
                expected_attachments_by_hand=expected_attachments,
                motion_scope="whole_body",
                whole_body_certificate=plan.get("whole_body_certificate"),
            )
            metrics = {
                **reach_metrics,
                **plan.get("metrics", {}),
                **execution["metrics"],
                "composite_intermediate_stage": True,
                "elapsed_s": round(time.monotonic() - started, 3),
            }
            return primitive_result(
                primitive_success=execution["primitive_success"],
                task_success=self._task_success(),
                stop_reason=execution["stop_reason"],
                recoverable=execution["recoverable"],
                suggested_next_tool=execution.get("suggested_next_tool"),
                metrics=metrics,
                diagnostics=execution["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool=None)

    @_planner_tool("rotate_wrist")
    def rotate_wrist(
        self,
        *,
        hand: str,
        target_quat_xyzw: Any | None = None,
        relative_axis_angle: Any | None = None,
        frame: str = "world",
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        try:
            hand = _normalize_hand(hand)
            if (target_quat_xyzw is None) == (relative_axis_angle is None):
                raise ValueError(
                    "provide exactly one of target_quat_xyzw or relative_axis_angle"
                )
            current = self.backend.get_eef_pose(hand)
            if current is None:
                raise RuntimeError("cannot read current EEF pose for rotate_wrist")
            position, current_quat = current
            if target_quat_xyzw is not None:
                target_quat = _quat_xyzw(target_quat_xyzw)
            else:
                rel = _axis_angle_to_quat_xyzw(relative_axis_angle)
                if frame == "eef":
                    target_quat = _quat_multiply_xyzw(current_quat, rel)
                elif frame == "world":
                    target_quat = _quat_multiply_xyzw(rel, current_quat)
                else:
                    raise ValueError("frame must be 'world' or 'eef'")
            return self.move_to(
                hand=hand,
                target_xyz=position,
                target_quat_xyzw=target_quat,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="move_to")

    @_planner_tool("press")
    def press(
        self,
        *,
        hand: str,
        target_xyz: Any,
        press_direction: Any | None = None,
        travel_m: float,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            target = _as_xyz(target_xyz)
            direction = _approach_vector(
                press_direction if press_direction is not None else [0, 0, -1]
            )
            travel = float(travel_m)
            if not np.isfinite(travel) or travel <= 0.0:
                raise ValueError("travel_m must be finite and positive")
            isolation_reference = None
            contact = (
                target - direction * PRESS_EEF_TO_CONTACT_OFFSET_M + direction * travel
            )
            guarded_transition_m = travel + 0.001
            pre = contact - direction * guarded_transition_m
            move = self._move_to_composite_stage(
                hand=hand,
                target_xyz=pre,
                target_quat_xyzw=None,
                position_tolerance_m=0.002,
                orientation_tolerance_rad=0.087,
                timeout_s=min(self._remaining_s(deadline), 40.0),
                hold_steps_required=1,
                contact_target_xyz=target,
                isolation_reference=isolation_reference,
            )
            if move.get("stop_reason") in _TERMINAL_STEP_STOP_REASONS:
                move.setdefault("metrics", {}).update(
                    {
                        "motion_scope": "whole_body",
                        "whole_body_execution": {
                            "available": True,
                            "ok": True,
                            "raw_success_checked_after_each_action": True,
                        },
                    }
                )
                return move
            if not move["primitive_success"]:
                return move
            guarded_press = self._guarded_incremental_move(
                hand=hand,
                target_xyz=contact,
                target_quat_xyzw=None,
                direction=direction,
                position_tolerance_m=0.012,
                timeout_s=self._remaining_s(deadline),
                require_expected_contact=True,
                contact_target_xyz=target,
                stop_on_expected_contact=True,
                isolation_reference=isolation_reference,
            )
            if guarded_press.get("stop_reason") in _TERMINAL_STEP_STOP_REASONS:
                guarded_press.setdefault("metrics", {}).update(
                    {
                        "motion_scope": "whole_body",
                        "whole_body_execution": {
                            "available": True,
                            "ok": True,
                            "raw_success_checked_after_each_action": True,
                        },
                    }
                )
                return guarded_press
            guarded_metrics = dict(guarded_press["metrics"])
            guarded_metrics.setdefault(
                "single_arm_isolation",
                move.get("metrics", {}).get("single_arm_isolation"),
            )
            return primitive_result(
                primitive_success=guarded_press["primitive_success"],
                task_success=self._task_success(),
                stop_reason="pressed"
                if guarded_press["primitive_success"]
                else guarded_press["stop_reason"],
                recoverable=guarded_press["recoverable"],
                suggested_next_tool=None
                if guarded_press["primitive_success"]
                else "observe",
                metrics={
                    **guarded_metrics,
                    "motion_scope": "whole_body",
                    "whole_body_execution": {
                        "available": True,
                        "ok": True,
                        "collision_certificate_verified_before_each_guarded_action": True,
                        "dual_attachment_checked_each_nonterminal_step": True,
                        "raw_success_checked_after_each_action": True,
                        "raw_success_preempts_post_step_safety_checks": True,
                    },
                    "precontact_motion": move.get("metrics", {}),
                    "requested_travel_m": travel,
                },
                diagnostics=guarded_press["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    def _guarded_incremental_move(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: Any | None,
        direction: np.ndarray,
        allow_expected_contact: bool | None = None,
        position_tolerance_m: float,
        timeout_s: float,
        require_expected_contact: bool = False,
        contact_target_xyz: np.ndarray | None = None,
        stop_on_expected_contact: bool = False,
        eef_to_contact_vector: np.ndarray | None = None,
        allowed_contact_distance_m: float = 0.025,
        isolation_reference: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del allow_expected_contact, eef_to_contact_vector, isolation_reference
        started = time.monotonic()
        quat = _quat_xyzw(target_quat_xyzw)
        expected_attachments = {
            side: _call_optional_arg(self.backend, "get_attached_object", side)
            for side in ("left", "right")
        }
        current = self.backend.get_eef_pose(hand)
        if current is None:
            return self._execution_result(
                primitive_success=False,
                stop_reason="pose_feedback_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
            )
        start = np.asarray(current[0], dtype=np.float64)
        target = np.asarray(target_xyz, dtype=np.float64)
        contact_target = (
            target
            if contact_target_xyz is None
            else np.asarray(contact_target_xyz, dtype=np.float64).reshape(3)
        )
        allowed_contact_distance = float(allowed_contact_distance_m)
        if not np.isfinite(allowed_contact_distance) or allowed_contact_distance <= 0.0:
            raise ValueError("allowed_contact_distance_m must be finite and positive")
        total = float(np.linalg.norm(target - start))
        nominal_waypoint_distances = _guarded_waypoint_distances(total)
        nominal_steps = len(nominal_waypoint_distances)
        steps = nominal_steps
        max_guarded_iterations = max(steps * 4, steps + 8)
        guard_metrics = {
            "guarded_step_m": 0.002,
            "guarded_coarse_step_m": 0.002,
            "guarded_fine_distance_m": total,
            "guarded_total_distance_m": total,
            "guarded_waypoints": steps,
            "guarded_max_feedback_iterations": max_guarded_iterations,
            "guarded_execution_mode": "receding_horizon_cartesian_ik",
            "guarded_physical_contact_query_interval_steps": 1,
        }
        if steps > max(1, int(float(timeout_s) * 120)):
            return self._execution_result(
                primitive_success=False,
                stop_reason="timeout",
                recoverable=True,
                suggested_next_tool="move_to",
                executed=0,
                trace=[],
                final_pos_err=total,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics=guard_metrics,
            )
        trace: list[dict[str, Any]] = []
        executed = 0
        final_pos_err = total
        final_ori_err: float | None = None
        expected_contact_seen = False
        for index in range(1, max_guarded_iterations + 1):
            contact = self._contact_report(
                hand=hand,
                target_xyz=contact_target,
                allowed_contact_distance_m=allowed_contact_distance,
            )
            if contact.get("available") is False:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="contact_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="observe",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        **guard_metrics,
                        "contact_report": contact,
                    },
                )
            expected_contact_seen = expected_contact_seen or bool(
                contact.get("expected_contact", False)
            )
            if stop_on_expected_contact and expected_contact_seen:
                return self._execution_result(
                    primitive_success=True,
                    stop_reason="guarded_reached",
                    recoverable=True,
                    suggested_next_tool=None,
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        **guard_metrics,
                        "guarded_direction": direction.tolist(),
                        "guarded_feedback_iterations": index,
                        "expected_contact_seen": True,
                        "contact_report": contact,
                    },
                )
            live_pose = self.backend.get_eef_pose(hand)
            live_target_error = (
                float(
                    np.linalg.norm(np.asarray(live_pose[0], dtype=np.float64) - target)
                )
                if live_pose is not None
                else float("inf")
            )
            terminal_tolerance_m = max(0.0015, min(float(position_tolerance_m), 0.003))
            terminal_reached = bool(live_target_error <= terminal_tolerance_m)
            if time.monotonic() - started > float(timeout_s) and not terminal_reached:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="timeout",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=live_target_error,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics=guard_metrics,
                )
            if terminal_reached:
                final_pos_err = live_target_error
                trace.append(
                    {
                        "event": "terminal_target_reached",
                        "env_action_sent": False,
                        "post_stop_action_policy": "no_additional_env_action",
                        "position_error_m": live_target_error,
                    }
                )
                break
            else:
                assert live_pose is not None
                live_position = np.asarray(live_pose[0], dtype=np.float64)
                remaining_vector = target - live_position
                remaining_distance = float(np.linalg.norm(remaining_vector))
                waypoint = live_position + remaining_vector * min(
                    1.0, 0.002 / max(remaining_distance, 1e-9)
                )
                guarded_plan = getattr(self.backend, "plan_whole_body_trajectory", None)
                if not callable(guarded_plan):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="whole_body_planner_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=live_target_error,
                        final_ori_err=final_ori_err,
                        held_steps=0,
                        started=started,
                        extra_metrics=guard_metrics,
                    )
                guarded_plan_kwargs = {
                    "hand": hand,
                    "target_xyz": waypoint,
                    "target_quat_xyzw": quat,
                    "timeout_s": min(
                        2.0,
                        max(
                            0.25,
                            float(timeout_s) - (time.monotonic() - started),
                        ),
                    ),
                    "attached_obj": expected_attachments[hand],
                }
                plan = guarded_plan(**guarded_plan_kwargs)
                reverse_path = plan.get("reverse_joint_trajectory")
                if reverse_path is not None:
                    self._last_guarded_retreat_paths[hand] = np.asarray(
                        _jsonable(reverse_path), dtype=np.float32
                    )
                guard_metrics["guarded_plan_metrics"] = plan.get("metrics", {})
                guarded_plan_attempts = [
                    {
                        "attempt": 1,
                        "ok": bool(plan.get("ok", False)),
                        "stop_reason": plan.get("stop_reason"),
                    }
                ]
                guard_metrics["guarded_plan_attempts"] = guarded_plan_attempts
                cartesian_report = plan.get("metrics", {}).get(
                    "guarded_cartesian_path_report"
                )
                if isinstance(cartesian_report, dict):
                    guard_metrics["guarded_last_plan_execution_waypoints"] = int(
                        cartesian_report.get("execution_waypoints", steps)
                    )
                    guard_metrics["guarded_cartesian_path_report"] = cartesian_report
            if not plan.get("ok"):
                live_after_failure = self.backend.get_eef_pose(hand)
                final_target_error = (
                    float(
                        np.linalg.norm(
                            np.asarray(live_after_failure[0], dtype=np.float64) - target
                        )
                    )
                    if live_after_failure is not None
                    else final_pos_err
                )
                return self._execution_result(
                    primitive_success=False,
                    stop_reason=str(plan.get("stop_reason", "guarded_plan_failed")),
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_target_error,
                    final_ori_err=final_ori_err,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        **guard_metrics,
                        **plan.get("metrics", {}),
                    },
                )
            execution = self._execute_actions(
                validate_action_chunk(plan["actions"])
                if plan.get("actions") is not None
                else None,
                hand=hand,
                target_xyz=waypoint,
                target_quat_xyzw=quat,
                position_tolerance_m=max(
                    0.0015,
                    min(float(position_tolerance_m), 0.0015),
                ),
                orientation_tolerance_rad=0.087,
                timeout_s=(
                    min(
                        16.0,
                        max(
                            0.5,
                            float(timeout_s) - (time.monotonic() - started),
                        ),
                    )
                ),
                require_pose=True,
                hold_steps_required=5,
                contact_target_xyz=contact_target,
                stop_on_expected_contact=stop_on_expected_contact,
                joint_trajectory=plan.get("joint_trajectory"),
                allowed_contact_distance_m=allowed_contact_distance,
                expected_attachments_by_hand=(
                    expected_attachments
                    if plan.get("joint_trajectory") is not None
                    else None
                ),
                motion_scope=(
                    "whole_body"
                    if plan.get("joint_trajectory") is not None
                    else "arm_only"
                ),
                whole_body_certificate=(
                    plan.get("whole_body_certificate")
                    if plan.get("joint_trajectory") is not None
                    else None
                ),
            )
            if execution.get("stop_reason") in _TERMINAL_STEP_STOP_REASONS:
                return execution
            executed += int(execution["metrics"].get("executed_waypoints", 0))
            trace.extend(execution["diagnostics"].get("trace", []))
            final_pos_err = execution["metrics"].get("final_position_error_m")
            final_ori_err = execution["metrics"].get("final_orientation_error_rad")
            if not execution["primitive_success"]:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason=str(execution["stop_reason"]),
                    recoverable=bool(execution["recoverable"]),
                    suggested_next_tool=str(
                        execution.get("suggested_next_tool") or "observe"
                    ),
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=int(execution["metrics"].get("held_steps", 0)),
                    started=started,
                    extra_metrics={
                        **execution.get("metrics", {}),
                        **guard_metrics,
                    },
                )
            after_contact = self._contact_report(
                hand=hand,
                target_xyz=contact_target,
                allowed_contact_distance_m=allowed_contact_distance,
            )
            expected_contact_seen = expected_contact_seen or bool(
                after_contact.get("expected_contact", False)
            )
            if stop_on_expected_contact and expected_contact_seen:
                break
        else:
            live_pose = self.backend.get_eef_pose(hand)
            final_target_error = (
                float(
                    np.linalg.norm(np.asarray(live_pose[0], dtype=np.float64) - target)
                )
                if live_pose is not None
                else final_pos_err
            )
            return self._execution_result(
                primitive_success=False,
                stop_reason="stalled_tracking",
                recoverable=True,
                suggested_next_tool="observe",
                executed=executed,
                trace=trace,
                final_pos_err=final_target_error,
                final_ori_err=final_ori_err,
                held_steps=0,
                started=started,
                extra_metrics={
                    **guard_metrics,
                    "guarded_feedback_iterations": max_guarded_iterations,
                },
            )
        if require_expected_contact and not expected_contact_seen:
            return self._execution_result(
                primitive_success=False,
                stop_reason="expected_contact_not_confirmed",
                recoverable=True,
                suggested_next_tool="observe",
                executed=executed,
                trace=trace,
                final_pos_err=final_pos_err,
                final_ori_err=final_ori_err,
                held_steps=0,
                started=started,
                extra_metrics={
                    **guard_metrics,
                    "expected_contact_seen": False,
                },
            )
        return self._execution_result(
            primitive_success=True,
            stop_reason="guarded_reached",
            recoverable=True,
            suggested_next_tool=None,
            executed=executed,
            trace=trace,
            final_pos_err=final_pos_err,
            final_ori_err=final_ori_err,
            held_steps=0,
            started=started,
            extra_metrics={
                **guard_metrics,
                "guarded_direction": direction.tolist(),
                "guarded_feedback_iterations": index,
                "expected_contact_seen": expected_contact_seen,
            },
        )

    def _gripper_command(
        self,
        hand: str,
        *,
        opening: float,
        timeout_s: float,
        contact_target_xyz: np.ndarray | None = None,
        allow_expected_contact: bool | None = None,
        hold_steps_required: int = 1,
        stop_on_attachment: bool = False,
        eef_to_contact_vector: np.ndarray | None = None,
        expected_attachment: Any = None,
        require_attachment: bool = False,
    ) -> dict[str, Any]:
        del allow_expected_contact, eef_to_contact_vector
        command = 1.0 if float(opening) >= 0.5 else -1.0
        latch = getattr(self.env, "_gripper_latch", None)
        current_command = (
            float(latch.get(hand, 1.0)) if isinstance(latch, dict) else 1.0
        )
        hold = _call_optional_arg(self.backend, "hold_action", hand)
        if hold is None:
            hold = np.zeros((ACTION_DIM,), dtype=np.float32)
        if command < current_command:
            command_stages = []
            stage_start = current_command
            if stage_start > 0.0:
                coarse_target = max(0.0, command)
                coarse_intervals = max(
                    1,
                    int(
                        math.ceil(
                            abs(coarse_target - stage_start)
                            / GRIPPER_CLOSE_COARSE_COMMAND_STEP
                        )
                    ),
                )
                command_stages.append(
                    np.linspace(
                        stage_start,
                        coarse_target,
                        coarse_intervals + 1,
                        dtype=np.float32,
                    )[1:]
                )
                stage_start = coarse_target
            if command < stage_start:
                fine_intervals = max(
                    1,
                    int(
                        math.ceil(
                            abs(command - stage_start) / GRIPPER_CLOSE_FINE_COMMAND_STEP
                        )
                    ),
                )
                command_stages.append(
                    np.linspace(
                        stage_start,
                        command,
                        fine_intervals + 1,
                        dtype=np.float32,
                    )[1:]
                )
            gripper_commands = np.concatenate(command_stages)
        else:
            # Release keeps the existing short command window; OG performs
            # its own bounded gradual assisted-grasp release.
            gripper_commands = np.repeat(np.float32(command), 3)
        actions = np.repeat(
            np.asarray(hold, dtype=np.float32).reshape(1, ACTION_DIM),
            len(gripper_commands),
            axis=0,
        )
        segment = ENV_ACTION_SEGMENTS[f"{hand}_gripper"]
        actions[:, segment] = gripper_commands[:, None]
        execution = self._execute_actions(
            validate_action_chunk(actions),
            hand=hand,
            target_xyz=None,
            target_quat_xyzw=None,
            position_tolerance_m=0.0,
            orientation_tolerance_rad=0.0,
            timeout_s=timeout_s,
            require_pose=False,
            contact_target_xyz=contact_target_xyz,
            hold_steps_required=max(1, int(hold_steps_required)),
            stop_on_attachment=stop_on_attachment,
            static_gripper_only=True,
            expected_attachment=expected_attachment,
            require_attachment=require_attachment,
            gripper_contact_settle_steps=(
                GRIPPER_CONTACT_SETTLE_STEPS if command < current_command else 0
            ),
        )
        if execution.get("stop_reason") in _TERMINAL_STEP_STOP_REASONS:
            return execution
        execution.setdefault("metrics", {})["gripper_command_profile"] = {
            "start": current_command,
            "target": command,
            "planned_steps": int(len(gripper_commands)),
            "max_command_step": GRIPPER_CLOSE_COARSE_COMMAND_STEP,
            "coarse_max_command_step": GRIPPER_CLOSE_COARSE_COMMAND_STEP,
            "fine_max_command_step": GRIPPER_CLOSE_FINE_COMMAND_STEP,
            "fine_region_command_upper_bound": 0.0,
            "two_finger_contact_settle_steps": (
                GRIPPER_CONTACT_SETTLE_STEPS if command < current_command else 0
            ),
        }
        if command > 0 and execution["primitive_success"]:
            clear_attached = getattr(self.backend, "clear_attached_object", None)
            if callable(clear_attached):
                clear_attached(hand)
        return primitive_result(
            primitive_success=execution["primitive_success"],
            task_success=self._task_success(),
            stop_reason="gripper_commanded"
            if execution["primitive_success"]
            else execution["stop_reason"],
            recoverable=execution["recoverable"],
            suggested_next_tool=None,
            metrics=execution["metrics"],
            diagnostics=execution["diagnostics"],
        )

    def _execute_navigation_trajectory(
        self,
        joint_trajectory: Any,
        *,
        base_goal_xyyaw: np.ndarray,
        timeout_s: float,
        plan_metrics: dict[str, Any],
        expected_attachments_by_hand: dict[str, Any] | None = None,
        prepared_start_q: Any | None = None,
        execution_policy: str = "default",
    ) -> dict[str, Any]:
        """Execute a BASE-only q path with fail-closed per-step isolation."""

        started = time.monotonic()
        q_path = np.asarray(
            _jsonable(joint_trajectory), dtype=np.float32
        ).copy()
        if q_path.ndim != 2 or len(q_path) < 1 or not np.isfinite(q_path).all():
            raise ValueError(
                f"navigation joint trajectory must be finite [T,D], got {q_path.shape}"
            )
        base_goal = _canonical_base_xyyaw(base_goal_xyyaw)
        certificate = plan_metrics.get("base_trajectory_certificate")
        current_q = (
            prepared_start_q
            if prepared_start_q is not None
            else _call_optional(self.backend, "get_joint_positions")
        )
        current_q = (
            None
            if current_q is None
            else np.asarray(_jsonable(current_q), dtype=np.float32).reshape(-1)
        )
        trajectory_digest = hashlib.sha256(
            np.ascontiguousarray(q_path, dtype=np.float32).tobytes()
        ).hexdigest()
        terminal_q_digest = hashlib.sha256(
            np.ascontiguousarray(q_path[-1], dtype=np.float32).tobytes()
        ).hexdigest()
        current_digest = (
            hashlib.sha256(
                np.ascontiguousarray(current_q, dtype=np.float32).tobytes()
            ).hexdigest()
            if isinstance(current_q, np.ndarray)
            else None
        )
        certificate_ok = bool(
            isinstance(certificate, dict)
            and isinstance(current_q, np.ndarray)
            and current_q.size == q_path.shape[1]
            and certificate.get("trajectory_sha256") == trajectory_digest
            and certificate.get("start_q_sha256") == current_digest
            and certificate.get("waypoint_count") == int(len(q_path))
            and certificate.get("world_collision_check") is True
            and certificate.get("self_collision_check") is True
            and certificate.get("post_interpolation_check") is True
            and certificate.get("attachment_hand_count") == 2
            and certificate.get("colliding_waypoint_count") == 0
            and certificate.get("terminal_q_sha256") == terminal_q_digest
            and certificate.get("base_goal_xyyaw_sha256")
            == _whole_body_target_sha256(base_goal)
            and certificate.get("terminal_command_limit")
            == TERMINAL_COMMAND_LIMIT
            and certificate.get("terminal_position_tolerance_m")
            == BASE_TERMINAL_POSITION_TOLERANCE_M
            and certificate.get("terminal_orientation_tolerance_rad")
            == BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD
        )
        if not certificate_ok:
            return self._execution_result(
                primitive_success=False,
                stop_reason="navigation_collision_certificate_unavailable",
                recoverable=False,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                started=started,
                extra_metrics={
                    **plan_metrics,
                    "certificate_verified_before_first_action": False,
                },
            )
        if not isinstance(expected_attachments_by_hand, dict) or set(
            expected_attachments_by_hand
        ) != {"left", "right"}:
            return self._execution_result(
                primitive_success=False,
                stop_reason="attachment_identity_unavailable",
                recoverable=False,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                started=started,
                extra_metrics={
                    **plan_metrics,
                    "certificate_verified_before_first_action": True,
                },
            )
        for side in ("left", "right"):
            live = _call_optional_arg(self.backend, "get_attached_object", side)
            matches, identity = _attachment_state_status(
                live, expected_attachments_by_hand[side], hand=side
            )
            if not matches:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="attachment_identity_mismatch",
                    recoverable=False,
                    suggested_next_tool="observe",
                    executed=0,
                    trace=[],
                    final_pos_err=None,
                    final_ori_err=None,
                    started=started,
                    extra_metrics={
                        **plan_metrics,
                        "certificate_verified_before_first_action": True,
                        "whole_body_attachment": {"hand": side, **identity},
                    },
                )
        contact_baseline = _call_optional_kw(
            self.backend,
            "capture_whole_body_contact_baseline",
            expected_attachments_by_hand=expected_attachments_by_hand,
        )
        if (
            not isinstance(contact_baseline, dict)
            or contact_baseline.get("available") is not True
        ):
            return self._execution_result(
                primitive_success=False,
                stop_reason="contact_feedback_unavailable",
                recoverable=False,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                started=started,
                extra_metrics={
                    **plan_metrics,
                    "whole_body_contact_baseline": contact_baseline,
                },
            )
        deadline = time.monotonic() + self._validated_timeout(timeout_s)
        capture_hold = getattr(
            self.backend,
            "capture_trajectory_hold_reference",
            None,
        )
        capture_isolation = getattr(
            self.backend,
            "capture_navigation_isolation_reference",
            None,
        )
        report_isolation = getattr(
            self.backend,
            "navigation_isolation_report",
            None,
        )
        if not all(
            callable(fn) for fn in (capture_hold, capture_isolation, report_isolation)
        ):
            return self._execution_result(
                primitive_success=False,
                stop_reason="navigation_isolation_feedback_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                started=started,
                extra_metrics={
                    **plan_metrics,
                    "navigation_isolation": {
                        "available": False,
                        "ok": False,
                        "mode": "base_only",
                        "checks": {},
                        "max_observed": {},
                        "checks_performed": 0,
                        "reason": "navigation isolation backend is unavailable",
                    },
                },
            )
        try:
            hold_reference = capture_hold(hand=None)
            isolation_reference = capture_isolation()
        except Exception as exc:
            return self._execution_result(
                primitive_success=False,
                stop_reason="navigation_isolation_feedback_unavailable",
                recoverable=True,
                suggested_next_tool="observe",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                started=started,
                extra_metrics={
                    **plan_metrics,
                    "navigation_isolation": {
                        "available": False,
                        "ok": False,
                        "mode": "base_only",
                        "checks": {},
                        "max_observed": {},
                        "checks_performed": 0,
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                },
            )
        if not isinstance(hold_reference, dict) or not isinstance(
            isolation_reference, dict
        ):
            raise RuntimeError("navigation isolation references are invalid")

        trace: list[dict[str, Any]] = []
        aggregate_isolation: dict[str, Any] | None = None
        executed = 0
        waypoint_index = 0
        final_pos_err: float | None = None
        final_ori_err: float | None = None
        terminal_commands_sent = 0
        dashboard_terminal_tilt_policy = bool(
            prepared_start_q is not None
            and execution_policy
            == PREPARED_DASHBOARD_BASE_EXECUTION_POLICY
            and plan_metrics.get("planning_profile")
            == DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
        )
        terminal_tilt_settle_pending = False
        terminal_tilt_settle_attempted = False
        terminal_tilt_settle_metrics = {
            "enabled": dashboard_terminal_tilt_policy,
            "eligible": False,
            "hold_sent": False,
            "maximum_holds": 1,
            "initial_base_roll_pitch_drift_rad": None,
            "settled_base_roll_pitch_drift_rad": None,
            "settled": False,
            "normal_limit_rad": LOCKED_BASE_RPY_MAX_DRIFT_RAD,
            "transient_limit_rad": (
                DASHBOARD_BASE_TERMINAL_SETTLE_RPY_MAX_RAD
            ),
            "drift_frame": "relative_to_navigation_isolation_reference",
            "collision_revalidated_for_transient": False,
            "nominal_certificate_covers_transient_tilt": False,
            "intermediate_transient_count": 0,
            "maximum_intermediate_base_roll_pitch_drift_rad": None,
            "normal_articulation_limit_rad": (
                LOCKED_ARTICULATION_MAX_DRIFT_RAD
            ),
            "transient_trunk_limit_rad": (
                DASHBOARD_BASE_INTERMEDIATE_TRUNK_MAX_DRIFT_RAD
            ),
            "intermediate_trunk_transient_count": 0,
            "maximum_intermediate_trunk_drift_rad": None,
        }
        navigation_check_names = {
            "base_z_locked",
            "base_roll_pitch_locked",
            "trunk_locked",
            "left_arm_locked",
            "right_arm_locked",
            "left_gripper_command_locked",
            "right_gripper_command_locked",
            "left_attachment_identity_unchanged",
            "right_attachment_identity_unchanged",
        }
        feedback_latency: dict[str, dict[str, float | int]] = {
            phase: {"count": 0, "total_s": 0.0, "max_s": 0.0}
            for phase in (
                "attachment_identity",
                "navigation_isolation",
                "whole_body_contact",
                "joint_tracking",
                "base_pose",
            )
        }

        def record_feedback_latency(phase: str, phase_started: float) -> None:
            elapsed = max(0.0, time.monotonic() - phase_started)
            sample = feedback_latency[phase]
            sample["count"] = int(sample["count"]) + 1
            sample["total_s"] = float(sample["total_s"]) + elapsed
            sample["max_s"] = max(float(sample["max_s"]), elapsed)

        def finish(
            *,
            primitive_success: bool,
            stop_reason: str,
            recoverable: bool,
            terminal_step_receipt: dict[str, bool | int] | None = None,
        ) -> dict[str, Any]:
            isolation = (
                aggregate_isolation
                if isinstance(aggregate_isolation, dict)
                else {
                    "available": False,
                    "ok": False,
                    "mode": "base_only",
                    "checks": {},
                    "max_observed": {},
                    "checks_performed": 0,
                    "reason": "no navigation action was verified",
                }
            )
            return self._execution_result(
                primitive_success=primitive_success,
                stop_reason=stop_reason,
                recoverable=recoverable,
                suggested_next_tool=(
                    None if primitive_success or not recoverable else "observe"
                ),
                executed=executed,
                trace=trace,
                final_pos_err=final_pos_err,
                final_ori_err=final_ori_err,
                started=started,
                extra_metrics={
                    **plan_metrics,
                    "final_yaw_error_rad": final_ori_err,
                    "certificate_verified_before_first_action": True,
                    "navigation_isolation": isolation,
                    "navigation_terminal": {
                        "command_limit": TERMINAL_COMMAND_LIMIT,
                        "commands_sent": terminal_commands_sent,
                        "position_tolerance_m": (
                            BASE_TERMINAL_POSITION_TOLERANCE_M
                        ),
                        "orientation_tolerance_rad": (
                            BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD
                        ),
                        "terminal_q_sha256": terminal_q_digest,
                    },
                    "navigation_feedback_latency": {
                        "clock": "time.monotonic",
                        "phases": deepcopy(feedback_latency),
                    },
                    "dashboard_base_terminal_tilt_settle": (
                        terminal_tilt_settle_metrics
                    ),
                    "live_start_equality_checked": (
                        prepared_start_q is None
                    ),
                    "collision_revalidated": False,
                    **(
                        {"terminal_step_receipt": dict(terminal_step_receipt)}
                        if terminal_step_receipt is not None
                        else {}
                    ),
                },
            )

        max_steps = max(len(q_path) - 1, 0) + TERMINAL_COMMAND_LIMIT
        while executed < max_steps:
            now = time.monotonic()
            if now >= deadline:
                return finish(
                    primitive_success=False,
                    stop_reason="timeout",
                    recoverable=True,
                )
            target_q = q_path[min(waypoint_index, len(q_path) - 1)]
            terminal_command = waypoint_index == len(q_path) - 1
            terminal_tilt_settle_command = bool(
                terminal_command and terminal_tilt_settle_pending
            )
            if terminal_tilt_settle_command:
                terminal_tilt_settle_pending = False
            if terminal_command:
                if terminal_commands_sent >= TERMINAL_COMMAND_LIMIT:
                    return finish(
                        primitive_success=False,
                        stop_reason="target_tolerance_not_met",
                        recoverable=True,
                    )
            try:
                action = self.backend.joint_target_to_action(
                    target_q,
                    hand=None,
                    fixed_reference=hold_reference,
                )
                action = validate_action_chunk(
                    np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM)
                )[0]
            except Exception as exc:
                trace.append(
                    {
                        "step": executed,
                        "waypoint_index": waypoint_index,
                        "action_conversion_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return finish(
                    primitive_success=False,
                    stop_reason="navigation_isolation_feedback_unavailable",
                    recoverable=True,
                )

            if time.monotonic() >= deadline:
                return finish(
                    primitive_success=False,
                    stop_reason="timeout",
                    recoverable=True,
                )
            step_receipt = self._step_env_action(action)
            executed += 1
            if terminal_tilt_settle_command:
                terminal_tilt_settle_metrics["hold_sent"] = True
            if terminal_command:
                terminal_commands_sent += 1
            terminal_outcome = _terminal_step_outcome(step_receipt)
            if terminal_outcome is not None:
                primitive_success, stop_reason = terminal_outcome
                trace.append(
                    {
                        "step": executed,
                        "waypoint_index": waypoint_index,
                        "terminal_command": terminal_command,
                        "terminal_command_index": (
                            terminal_commands_sent if terminal_command else None
                        ),
                        "step_receipt": dict(step_receipt),
                    }
                )
                return finish(
                    primitive_success=primitive_success,
                    stop_reason=stop_reason,
                    recoverable=False,
                    terminal_step_receipt=step_receipt,
                )
            attachment_started = time.monotonic()
            attachment_mismatch: tuple[str, dict[str, Any]] | None = None
            try:
                for side in ("left", "right"):
                    live = _call_optional_arg(
                        self.backend, "get_attached_object", side
                    )
                    matches, identity = _attachment_state_status(
                        live, expected_attachments_by_hand[side], hand=side
                    )
                    if not matches:
                        attachment_mismatch = (side, identity)
                        break
            finally:
                record_feedback_latency(
                    "attachment_identity", attachment_started
                )
            if attachment_mismatch is not None:
                side, identity = attachment_mismatch
                trace.append(
                    {
                        "step": executed,
                        "waypoint_index": waypoint_index,
                        "whole_body_attachment": {
                            "hand": side,
                            **identity,
                        },
                    }
                )
                return finish(
                    primitive_success=False,
                    stop_reason="attachment_identity_mismatch",
                    recoverable=False,
                )
            isolation_started = time.monotonic()
            try:
                isolation_report = report_isolation(
                    action=action,
                    reference=isolation_reference,
                )
            finally:
                record_feedback_latency(
                    "navigation_isolation", isolation_started
                )
            if not isinstance(isolation_report, dict):
                isolation_report = {
                    "available": False,
                    "ok": False,
                    "mode": "base_only",
                    "checks": {},
                    "max_observed": {},
                    "reason": "navigation isolation reporter returned no mapping",
                }
            aggregate_isolation = self._merge_isolation_report(
                aggregate_isolation, isolation_report
            )
            trace_entry: dict[str, Any] = {
                "step": executed,
                "waypoint_index": waypoint_index,
                "terminal_command": terminal_command,
                "terminal_command_index": (
                    terminal_commands_sent if terminal_command else None
                ),
                "navigation_isolation": isolation_report,
                "dashboard_base_terminal_tilt_settle_command": (
                    terminal_tilt_settle_command
                ),
            }
            if isolation_report.get("available") is not True:
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="navigation_isolation_feedback_unavailable",
                    recoverable=True,
                )
            isolation_checks = dict(isolation_report.get("checks") or {})
            failed_isolation_checks = {
                name
                for name, passed in isolation_checks.items()
                if passed is not True
            }
            roll_pitch_drift_raw = dict(
                isolation_report.get("max_observed") or {}
            ).get("base_roll_pitch_drift_rad")
            roll_pitch_threshold_raw = dict(
                isolation_report.get("thresholds") or {}
            ).get("base_roll_pitch_rad")
            roll_pitch_drift = (
                float(roll_pitch_drift_raw)
                if isinstance(roll_pitch_drift_raw, (int, float))
                and not isinstance(roll_pitch_drift_raw, bool)
                and math.isfinite(float(roll_pitch_drift_raw))
                else None
            )
            roll_pitch_threshold_matches = bool(
                isinstance(roll_pitch_threshold_raw, (int, float))
                and not isinstance(roll_pitch_threshold_raw, bool)
                and math.isfinite(float(roll_pitch_threshold_raw))
                and math.isclose(
                    float(roll_pitch_threshold_raw),
                    LOCKED_BASE_RPY_MAX_DRIFT_RAD,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            trunk_drift_raw = dict(
                isolation_report.get("max_observed") or {}
            ).get("trunk_drift_rad")
            articulation_threshold_raw = dict(
                isolation_report.get("thresholds") or {}
            ).get("articulation_rad")
            trunk_drift = (
                float(trunk_drift_raw)
                if isinstance(trunk_drift_raw, (int, float))
                and not isinstance(trunk_drift_raw, bool)
                and math.isfinite(float(trunk_drift_raw))
                else None
            )
            articulation_threshold_matches = bool(
                isinstance(articulation_threshold_raw, (int, float))
                and not isinstance(articulation_threshold_raw, bool)
                and math.isfinite(float(articulation_threshold_raw))
                and math.isclose(
                    float(articulation_threshold_raw),
                    LOCKED_ARTICULATION_MAX_DRIFT_RAD,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            terminal_tilt_eligible = bool(
                dashboard_terminal_tilt_policy
                and terminal_command
                and terminal_commands_sent == 1
                and not terminal_tilt_settle_attempted
                and set(isolation_checks) == navigation_check_names
                and failed_isolation_checks == {"base_roll_pitch_locked"}
                and roll_pitch_threshold_matches
                and roll_pitch_drift is not None
                and roll_pitch_drift > LOCKED_BASE_RPY_MAX_DRIFT_RAD + 1e-9
                and roll_pitch_drift
                <= DASHBOARD_BASE_TERMINAL_SETTLE_RPY_MAX_RAD + 1e-9
            )
            intermediate_tilt_eligible = bool(
                dashboard_terminal_tilt_policy
                and not terminal_command
                and set(isolation_checks) == navigation_check_names
                and failed_isolation_checks == {"base_roll_pitch_locked"}
                and roll_pitch_threshold_matches
                and roll_pitch_drift is not None
                and roll_pitch_drift > LOCKED_BASE_RPY_MAX_DRIFT_RAD + 1e-9
                and roll_pitch_drift
                <= DASHBOARD_BASE_TERMINAL_SETTLE_RPY_MAX_RAD + 1e-9
            )
            intermediate_trunk_eligible = bool(
                dashboard_terminal_tilt_policy
                and not terminal_command
                and set(isolation_checks) == navigation_check_names
                and failed_isolation_checks == {"trunk_locked"}
                and articulation_threshold_matches
                and trunk_drift is not None
                and trunk_drift > LOCKED_ARTICULATION_MAX_DRIFT_RAD + 1e-9
                and trunk_drift
                <= DASHBOARD_BASE_INTERMEDIATE_TRUNK_MAX_DRIFT_RAD + 1e-9
            )
            if (
                terminal_tilt_settle_command
                and isolation_report.get("ok") is True
                and (
                    set(isolation_checks) != navigation_check_names
                    or roll_pitch_drift is None
                )
            ):
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="navigation_isolation_feedback_unavailable",
                    recoverable=True,
                )
            if (
                isolation_report.get("ok") is not True
                and not terminal_tilt_eligible
                and not intermediate_tilt_eligible
                and not intermediate_trunk_eligible
            ):
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="navigation_isolation_violation",
                    recoverable=True,
                )
            if terminal_tilt_eligible:
                trace_entry["navigation_isolation_deferred"] = (
                    "dashboard_base_terminal_roll_pitch_settle"
                )
            elif intermediate_tilt_eligible:
                trace_entry["navigation_isolation_deferred"] = (
                    "dashboard_base_intermediate_roll_pitch_transient"
                )
                terminal_tilt_settle_metrics["intermediate_transient_count"] = (
                    int(
                        terminal_tilt_settle_metrics[
                            "intermediate_transient_count"
                        ]
                    )
                    + 1
                )
                prior_intermediate_drift = terminal_tilt_settle_metrics[
                    "maximum_intermediate_base_roll_pitch_drift_rad"
                ]
                terminal_tilt_settle_metrics[
                    "maximum_intermediate_base_roll_pitch_drift_rad"
                ] = (
                    roll_pitch_drift
                    if prior_intermediate_drift is None
                    else max(float(prior_intermediate_drift), roll_pitch_drift)
                )
            elif intermediate_trunk_eligible:
                trace_entry["navigation_isolation_deferred"] = (
                    "dashboard_base_intermediate_trunk_transient"
                )
                terminal_tilt_settle_metrics[
                    "intermediate_trunk_transient_count"
                ] = (
                    int(
                        terminal_tilt_settle_metrics[
                            "intermediate_trunk_transient_count"
                        ]
                    )
                    + 1
                )
                prior_trunk_drift = terminal_tilt_settle_metrics[
                    "maximum_intermediate_trunk_drift_rad"
                ]
                terminal_tilt_settle_metrics[
                    "maximum_intermediate_trunk_drift_rad"
                ] = (
                    trunk_drift
                    if prior_trunk_drift is None
                    else max(float(prior_trunk_drift), trunk_drift)
                )
            contact_started = time.monotonic()
            try:
                contact = _call_optional_kw(
                    self.backend,
                    "whole_body_contact_report",
                    baseline=contact_baseline,
                    expected_attachments_by_hand=expected_attachments_by_hand,
                    allowed_expected_contact=None,
                )
            finally:
                record_feedback_latency(
                    "whole_body_contact", contact_started
                )
            trace_entry["whole_body_contact"] = contact
            if (
                not isinstance(contact, dict)
                or contact.get("available") is not True
            ):
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="contact_feedback_unavailable",
                    recoverable=False,
                )
            if contact.get("unexpected_contact") is True:
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="unexpected_contact",
                    recoverable=False,
                )

            tracking_started = time.monotonic()
            try:
                tracking = _call_optional_kw(
                    self.backend,
                    "joint_tracking_report",
                    target_q=target_q,
                    hand=None,
                )
            finally:
                record_feedback_latency("joint_tracking", tracking_started)
            if not isinstance(tracking, dict) or tracking.get("available") is not True:
                trace_entry["joint_tracking"] = tracking
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="joint_tracking_feedback_unavailable",
                    recoverable=True,
                )
            trace_entry["joint_tracking"] = tracking
            hard_tracking, hard_tracking_available = (
                _tracking_hard_deviation_report(tracking)
            )
            trace_entry["tracking_hard_deviation"] = hard_tracking
            if not hard_tracking_available:
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="joint_tracking_feedback_unavailable",
                    recoverable=True,
                )
            if hard_tracking["hard_deviation"] is True:
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="tracking_hard_deviation",
                    recoverable=False,
                )
            if (
                terminal_tilt_eligible or terminal_tilt_settle_command
            ) and tracking.get("reached") is not True:
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="target_tolerance_not_met",
                    recoverable=True,
                )
            base_pose_started = time.monotonic()
            try:
                base_pose = _call_optional(self.backend, "get_base_pose")
            finally:
                record_feedback_latency("base_pose", base_pose_started)
            if base_pose is None:
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="pose_feedback_unavailable",
                    recoverable=True,
                )
            base_pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
            if base_pose.shape != (3,) or not np.isfinite(base_pose).all():
                trace.append(trace_entry)
                return finish(
                    primitive_success=False,
                    stop_reason="pose_feedback_unavailable",
                    recoverable=True,
                )
            final_pos_err = float(np.linalg.norm(base_pose[:2] - base_goal[:2]))
            final_ori_err = abs(_wrap_angle(float(base_pose[2]) - float(base_goal[2])))

            trace.append(trace_entry)
            if not terminal_command:
                waypoint_index += 1
                continue
            terminal_pose_reached = bool(
                final_pos_err <= BASE_TERMINAL_POSITION_TOLERANCE_M + 1e-9
                and final_ori_err
                <= BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD + 1e-9
            )
            if terminal_tilt_eligible:
                if not terminal_pose_reached:
                    return finish(
                        primitive_success=False,
                        stop_reason="navigation_isolation_violation",
                        recoverable=True,
                    )
                terminal_tilt_settle_attempted = True
                terminal_tilt_settle_pending = True
                terminal_tilt_settle_metrics.update(
                    {
                        "eligible": True,
                        "initial_base_roll_pitch_drift_rad": (
                            roll_pitch_drift
                        ),
                    }
                )
                continue
            if terminal_tilt_settle_command:
                terminal_tilt_settle_metrics.update(
                    {
                        "settled_base_roll_pitch_drift_rad": (
                            roll_pitch_drift
                        ),
                        "settled": bool(
                            terminal_pose_reached
                            and roll_pitch_drift is not None
                            and roll_pitch_drift
                            <= LOCKED_BASE_RPY_MAX_DRIFT_RAD + 1e-9
                        ),
                    }
                )
                if not terminal_tilt_settle_metrics["settled"]:
                    return finish(
                        primitive_success=False,
                        stop_reason="navigation_isolation_violation",
                        recoverable=True,
                    )
            if terminal_pose_reached:
                return finish(
                    primitive_success=True,
                    stop_reason="reached",
                    recoverable=True,
                )
        return finish(
            primitive_success=False,
            stop_reason="target_tolerance_not_met",
            recoverable=True,
        )

    def _execute_actions(
        self,
        actions: np.ndarray | None,
        *,
        hand: str,
        target_xyz: np.ndarray | None,
        target_quat_xyzw: np.ndarray | None,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        timeout_s: float,
        require_pose: bool,
        base_goal_xyyaw: np.ndarray | None = None,
        hold_steps_required: int = 10,
        contact_target_xyz: np.ndarray | None = None,
        allow_expected_contact: bool | None = None,
        stop_on_expected_contact: bool = False,
        stop_on_attachment: bool = False,
        static_gripper_only: bool = False,
        joint_trajectory: Any | None = None,
        eef_to_contact_vector: np.ndarray | None = None,
        expected_attachment: Any = None,
        expected_attachments_by_hand: dict[str, Any] | None = None,
        require_attachment: bool = False,
        gripper_contact_settle_steps: int = 0,
        allowed_contact_distance_m: float = 0.025,
        isolation_reference: dict[str, Any] | None = None,
        motion_scope: str = "arm_only",
        whole_body_certificate: dict[str, Any] | None = None,
        action_counter: dict[str, int] | None = None,
        allow_replan_checkpoint: bool = False,
        replan_checkpoint_position_improvement_m: float = (
            WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
        ),
        prepared_start_q: Any | None = None,
        execution_policy: str = "default",
    ) -> dict[str, Any]:
        del allow_expected_contact, eef_to_contact_vector
        started = time.monotonic()
        execution_policy = str(execution_policy)
        if execution_policy not in {
            "default",
            PREPARED_DASHBOARD_EEF_EXECUTION_POLICY,
        }:
            raise ValueError(
                f"unsupported whole-body execution policy {execution_policy!r}"
            )
        if (
            execution_policy == PREPARED_DASHBOARD_EEF_EXECUTION_POLICY
            and motion_scope != "whole_body"
        ):
            raise ValueError(
                "prepared Dashboard EEF execution policy requires whole_body scope"
            )
        checkpoint_trigger_position_improvement_m = float(
            replan_checkpoint_position_improvement_m
        )
        if (
            not math.isfinite(checkpoint_trigger_position_improvement_m)
            or checkpoint_trigger_position_improvement_m
            < WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
        ):
            raise ValueError(
                "replan checkpoint trigger must be finite and at least "
                f"{WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M} m"
            )
        if action_counter is not None:
            action_counter.setdefault("attempted", 0)
            action_counter.setdefault("confirmed", 0)
        if motion_scope not in _MOTION_SCOPES:
            raise ValueError(f"unsupported analytic motion scope {motion_scope!r}")
        if bool(static_gripper_only) != (motion_scope == "gripper_only"):
            if bool(static_gripper_only):
                motion_scope = "gripper_only"
            elif motion_scope == "gripper_only":
                raise ValueError("gripper-only scope requires static gripper motion")
        if (actions is None) == (joint_trajectory is None):
            raise ValueError("provide exactly one of actions or joint_trajectory")
        isolation_required = base_goal_xyyaw is None and motion_scope != "whole_body"
        isolation_gripper_only = bool(static_gripper_only)
        if isolation_required and isolation_reference is None:
            isolation_reference = self._capture_single_arm_isolation(
                hand=hand,
                gripper_only=isolation_gripper_only,
                reference_origin="primitive_call_start",
                motion_scope=motion_scope,
            )
        if isolation_required:
            expected_mode = (
                "gripper_only"
                if isolation_gripper_only
                else ("arm_motion" if motion_scope == "arm_only" else motion_scope)
            )
            if (
                not isinstance(isolation_reference, dict)
                or isolation_reference.get("selected_hand") != _normalize_hand(hand)
                or isolation_reference.get("mode") != expected_mode
            ):
                self._active_isolation_report = {
                    "available": False,
                    "ok": False,
                    "selected_hand": _normalize_hand(hand),
                    "mode": expected_mode,
                    "context_id": isolation_reference.get("context_id")
                    if isinstance(isolation_reference, dict)
                    else None,
                    "reference_origin": isolation_reference.get(
                        "reference_origin", "primitive_call_start"
                    )
                    if isinstance(isolation_reference, dict)
                    else "primitive_call_start",
                    "reason": "isolation reference unavailable or mismatched",
                    "checks": {},
                    "max_observed": {},
                    "checks_performed": 0,
                }
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="single_arm_isolation_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="observe",
                    executed=0,
                    trace=[],
                    final_pos_err=None,
                    final_ori_err=None,
                    held_steps=0,
                    started=started,
                )
            self._active_isolation_report = {
                "available": True,
                "ok": True,
                "selected_hand": _normalize_hand(hand),
                "mode": expected_mode,
                "context_id": isolation_reference.get("context_id"),
                "reference_origin": isolation_reference.get(
                    "reference_origin", "primitive_call_start"
                ),
                "checks": {},
                "max_observed": {},
                "checks_performed": 0,
            }
        else:
            self._active_isolation_report = None
        if (stop_on_attachment or require_attachment) and expected_attachment is None:
            raise ValueError(
                "attachment monitoring requires a locked expected attachment root"
            )
        if require_attachment:
            attached_at_execution_start = _call_optional_arg(
                self.backend, "get_attached_object", hand
            )
            attachment_matches, attachment_identity = _attachment_identity_status(
                attached_at_execution_start,
                expected_attachment,
                hand=hand,
            )
            if not attachment_matches:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason=(
                        "attachment_lost"
                        if attached_at_execution_start is None
                        else "attachment_identity_mismatch"
                    ),
                    recoverable=True,
                    suggested_next_tool="observe",
                    executed=0,
                    trace=[],
                    final_pos_err=None,
                    final_ori_err=None,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        "attachment_identity": attachment_identity,
                        "attachment_preflight_checked": True,
                        "attached_collision_body": {
                            "available": True,
                            "required_during_execution": True,
                        },
                    },
                )
        whole_body_contact_baseline: dict[str, Any] | None = None
        if motion_scope == "whole_body":
            if not isinstance(expected_attachments_by_hand, dict) or set(
                expected_attachments_by_hand
            ) != {"left", "right"}:
                raise ValueError(
                    "whole-body execution requires a two-hand attachment snapshot"
                )
            for side in ("left", "right"):
                live = _call_optional_arg(self.backend, "get_attached_object", side)
                matches, identity = _attachment_state_status(
                    live,
                    expected_attachments_by_hand[side],
                    hand=side,
                )
                if not matches:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="attachment_identity_mismatch",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=0,
                        trace=[],
                        final_pos_err=None,
                        final_ori_err=None,
                        held_steps=0,
                        started=started,
                        extra_metrics={
                            "whole_body_attachment_preflight": {
                                "hand": side,
                                **identity,
                            }
                        },
                    )
            whole_body_contact_baseline = _call_optional_kw(
                self.backend,
                "capture_whole_body_contact_baseline",
                expected_attachments_by_hand=expected_attachments_by_hand,
            )
            if (
                not isinstance(whole_body_contact_baseline, dict)
                or whole_body_contact_baseline.get("available") is not True
            ):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="contact_feedback_unavailable",
                    recoverable=False,
                    suggested_next_tool="observe",
                    executed=0,
                    trace=[],
                    final_pos_err=None,
                    final_ori_err=None,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        "whole_body_contact_baseline": (
                            whole_body_contact_baseline
                        )
                    },
                )
        action_chunk = validate_action_chunk(actions) if actions is not None else None
        q_chunk = (
            np.asarray(_jsonable(joint_trajectory), dtype=np.float32)
            if joint_trajectory is not None
            else None
        )
        if q_chunk is not None and (
            q_chunk.ndim != 2
            or q_chunk.shape[0] < 1
            or not np.isfinite(q_chunk).all()
        ):
            raise ValueError(
                f"joint trajectory must be a finite [T,D] array, got {q_chunk.shape}"
            )
        whole_body_eef_guard: dict[str, Any] | None = None

        def eef_guard_snapshot() -> dict[str, Any] | None:
            if whole_body_eef_guard is None:
                return None
            return {
                key: _jsonable(value)
                for key, value in whole_body_eef_guard.items()
                if key
                not in {
                    "nominal_positions",
                    "nominal_quaternions",
                    "call_start_xyz",
                    "call_start_quat_xyzw",
                    "target_xyz",
                    "direct_axis",
                    "previous_live_position",
                    "previous_live_quat_xyzw",
                }
            }

        if motion_scope == "whole_body":
            if not isinstance(whole_body_certificate, dict):
                raise ValueError(
                    "whole-body execution requires a collision certificate"
                )
            if q_chunk is None:
                raise ValueError(
                    "whole-body execution requires a finite joint trajectory"
                )
            digest = hashlib.sha256(
                np.ascontiguousarray(q_chunk, dtype=np.float32).tobytes()
            ).hexdigest()
            current_q = (
                prepared_start_q
                if prepared_start_q is not None
                else _call_optional(self.backend, "get_joint_positions")
            )
            current_q_array = (
                np.asarray(current_q, dtype=np.float32).reshape(-1)
                if current_q is not None
                else None
            )
            current_q_digest = (
                hashlib.sha256(
                    np.ascontiguousarray(
                        current_q_array,
                        dtype=np.float32,
                    ).tobytes()
                ).hexdigest()
                if current_q_array is not None
                and current_q_array.shape == (q_chunk.shape[1],)
                and np.isfinite(current_q_array).all()
                else None
            )
            if current_q_array is not None and int(q_chunk.shape[1]) != int(
                current_q_array.shape[0]
            ):
                raise ValueError(
                    "whole-body joint trajectory dimension "
                    f"{q_chunk.shape[1]} does not match live robot dimension "
                    f"{current_q_array.shape[0]}"
                )
            joint_name_layout = _call_optional_arg(
                self.backend,
                "whole_body_joint_names",
                _normalize_hand(hand),
            )
            joint_name_layout_digest = (
                _joint_name_layout_sha256(joint_name_layout)
                if isinstance(joint_name_layout, (tuple, list))
                and len(joint_name_layout) == q_chunk.shape[1]
                else None
            )
            source_dense_q = np.asarray(
                whole_body_certificate.get("source_dense_trajectory"),
                dtype=np.float32,
            )
            certified_dense_q = np.asarray(
                whole_body_certificate.get("dense_collision_trajectory"),
                dtype=np.float32,
            )
            source_dense_row_count = (
                int(source_dense_q.shape[0]) if source_dense_q.ndim == 2 else 0
            )
            certified_dense_row_count = (
                int(certified_dense_q.shape[0])
                if certified_dense_q.ndim == 2
                else 0
            )

            def certified_indices(name: str) -> np.ndarray:
                raw = whole_body_certificate.get(name)
                if (
                    not isinstance(raw, (tuple, list))
                    or not raw
                    or any(type(value) is not int for value in raw)
                ):
                    return np.zeros((0,), dtype=np.int64)
                return np.asarray(raw, dtype=np.int64).reshape(-1)

            source_dense_indices = certified_indices(
                "execution_source_dense_indices"
            )
            collision_dense_indices = certified_indices(
                "execution_collision_dense_indices"
            )
            densification_step_raw = whole_body_certificate.get(
                "collision_densification_joint_step_rad"
            )
            densification_step = (
                float(densification_step_raw)
                if isinstance(densification_step_raw, (int, float))
                and not isinstance(densification_step_raw, bool)
                else float("nan")
            )
            allowed_densification_steps = tuple(
                WHOLE_BODY_DENSE_COLLISION_STEP * (0.5**attempt)
                for attempt in range(8)
            )
            densification_step_allowed = (
                math.isfinite(densification_step)
                and any(
                    abs(densification_step - allowed) <= 1e-12
                    for allowed in allowed_densification_steps
                )
            )
            reconstructed_dense_q: np.ndarray | None = None
            reconstructed_collision_indices = np.zeros((0,), dtype=np.int64)
            if (
                current_q_array is not None
                and source_dense_q.ndim == 2
                and certified_dense_q.ndim == 2
                and source_dense_q.shape[1:] == (q_chunk.shape[1],)
                and certified_dense_q.shape[1:] == (q_chunk.shape[1],)
                and len(source_dense_q) >= 2
                and len(certified_dense_q) >= 2
                and np.isfinite(source_dense_q).all()
                and np.isfinite(certified_dense_q).all()
                and densification_step_allowed
            ):
                execution_with_start = np.vstack(
                    [current_q_array.reshape(1, -1), q_chunk]
                ).astype(np.float32)
                reconstructed_dense_q = _interpolate_joint_trajectory(
                    execution_with_start,
                    max_inter_dist=densification_step,
                )
                try:
                    reconstructed_collision_indices = (
                        _whole_body_ordered_row_indices(
                            reconstructed_dense_q, execution_with_start
                        )
                    )
                except (RuntimeError, ValueError):
                    reconstructed_collision_indices = np.zeros(
                        (0,), dtype=np.int64
                    )
            source_indices_valid = (
                source_dense_indices.shape == (len(q_chunk) + 1,)
                and source_dense_indices[0] == 0
                and source_dense_indices[-1] == source_dense_row_count - 1
                and bool(np.all(np.diff(source_dense_indices) > 0))
            )
            collision_indices_valid = (
                collision_dense_indices.shape == (len(q_chunk) + 1,)
                and collision_dense_indices[0] == 0
                and collision_dense_indices[-1]
                == certified_dense_row_count - 1
                and bool(np.all(np.diff(collision_dense_indices) > 0))
            )
            lineage_checks = {
                "schema": whole_body_certificate.get("schema_version") == 3,
                "lineage_mode": whole_body_certificate.get(
                    "collision_lineage_mode"
                )
                == "source_dense_execution_subset_recertified_v1",
                "q_encoding": whole_body_certificate.get("q_encoding")
                == "float32-c-order",
                "execution_digest": (
                    whole_body_certificate.get("trajectory_sha256") == digest
                    and whole_body_certificate.get(
                        "execution_trajectory_sha256"
                    )
                    == digest
                ),
                "live_start": (
                    current_q_digest is not None
                    and whole_body_certificate.get("start_q_sha256")
                    == current_q_digest
                ),
                "execution_count": (
                    whole_body_certificate.get("waypoint_count")
                    == int(len(q_chunk))
                    and whole_body_certificate.get(
                        "execution_waypoint_count"
                    )
                    == int(len(q_chunk))
                    and whole_body_certificate.get("execution_includes_start")
                    is False
                ),
                "q_dimension": whole_body_certificate.get("q_dimension")
                == int(q_chunk.shape[1]),
                "joint_layout": (
                    joint_name_layout_digest is not None
                    and whole_body_certificate.get(
                        "joint_name_layout_sha256"
                    )
                    == joint_name_layout_digest
                ),
                "active_dof_count": whole_body_certificate.get(
                    "active_dof_count"
                )
                == 21,
                "selected_eef_goal_count": whole_body_certificate.get(
                    "selected_eef_goal_count"
                )
                == 1,
                "inactive_eef_goal_count": whole_body_certificate.get(
                    "inactive_eef_goal_count"
                )
                == 0,
                "attachment_hand_count": whole_body_certificate.get(
                    "attachment_hand_count"
                )
                == 2,
                "world_collision_check": whole_body_certificate.get(
                    "world_collision_check"
                )
                is True,
                "self_collision_check": whole_body_certificate.get(
                    "self_collision_check"
                )
                is True,
                "post_interpolation_check": whole_body_certificate.get(
                    "post_interpolation_check"
                )
                is True,
                "densification_method": whole_body_certificate.get(
                    "collision_densification_method"
                )
                == "joint_linear_max_inter_dist",
                "densification_step": densification_step_allowed,
                "source_dense_shape": (
                    source_dense_q.ndim == 2
                    and source_dense_q.shape[1:] == (q_chunk.shape[1],)
                    and len(source_dense_q) >= 2
                    and np.isfinite(source_dense_q).all()
                ),
                "source_dense_count": whole_body_certificate.get(
                    "source_dense_waypoint_count"
                )
                == source_dense_row_count,
                "source_dense_start": (
                    current_q_array is not None
                    and source_dense_q.ndim == 2
                    and len(source_dense_q)
                    and np.array_equal(source_dense_q[0], current_q_array)
                    and whole_body_certificate.get(
                        "source_dense_includes_start"
                    )
                    is True
                ),
                "source_dense_digest": (
                    source_dense_q.ndim == 2
                    and whole_body_certificate.get(
                        "source_dense_trajectory_sha256"
                    )
                    == hashlib.sha256(
                        np.ascontiguousarray(
                            source_dense_q, dtype=np.float32
                        ).tobytes()
                    ).hexdigest()
                ),
                "source_indices": (
                    source_indices_valid
                    and whole_body_certificate.get(
                        "execution_source_dense_indices_sha256"
                    )
                    == _whole_body_indices_sha256(source_dense_indices)
                    and whole_body_certificate.get(
                        "execution_source_terminal_index"
                    )
                    == source_dense_row_count - 1
                ),
                "source_subset_q": (
                    source_indices_valid
                    and np.array_equal(
                        source_dense_q[source_dense_indices[1:]], q_chunk
                    )
                ),
                "dense_shape": (
                    certified_dense_q.ndim == 2
                    and certified_dense_q.shape[1:] == (q_chunk.shape[1],)
                    and len(certified_dense_q) >= 2
                    and np.isfinite(certified_dense_q).all()
                ),
                "dense_count": (
                    whole_body_certificate.get("collision_free_waypoints")
                    == certified_dense_row_count
                    and whole_body_certificate.get(
                        "dense_collision_waypoint_count"
                    )
                    == certified_dense_row_count
                    and whole_body_certificate.get(
                        "dense_collision_checked_waypoint_count"
                    )
                    == certified_dense_row_count
                    and whole_body_certificate.get(
                        "dense_collision_includes_start"
                    )
                    is True
                ),
                "dense_digest": (
                    certified_dense_q.ndim == 2
                    and whole_body_certificate.get(
                        "dense_collision_trajectory_sha256"
                    )
                    == hashlib.sha256(
                        np.ascontiguousarray(
                            certified_dense_q, dtype=np.float32
                        ).tobytes()
                    ).hexdigest()
                ),
                "dense_reconstruction": (
                    reconstructed_dense_q is not None
                    and np.array_equal(
                        certified_dense_q, reconstructed_dense_q
                    )
                ),
                "collision_indices": (
                    collision_indices_valid
                    and whole_body_certificate.get(
                        "execution_collision_dense_indices_sha256"
                    )
                    == _whole_body_indices_sha256(collision_dense_indices)
                    and whole_body_certificate.get(
                        "execution_collision_terminal_index"
                    )
                    == certified_dense_row_count - 1
                    and np.array_equal(
                        collision_dense_indices,
                        reconstructed_collision_indices,
                    )
                ),
                "collision_subset_q": (
                    collision_indices_valid
                    and np.array_equal(
                        certified_dense_q[collision_dense_indices[1:]],
                        q_chunk,
                    )
                ),
            }
            if not all(lineage_checks.values()):
                failed_lineage_checks = sorted(
                    name
                    for name, passed in lineage_checks.items()
                    if not passed
                )
                raise RuntimeError(
                    "whole-body trajectory does not match its collision "
                    f"certificate: {failed_lineage_checks}"
                )
            if target_xyz is not None:
                certified_positions = np.asarray(
                    whole_body_certificate.get(
                        "selected_eef_execution_positions"
                    ),
                    dtype=np.float32,
                )
                certified_quaternions = np.asarray(
                    whole_body_certificate.get(
                        "selected_eef_execution_quaternions_xyzw"
                    ),
                    dtype=np.float32,
                )
                certified_dense_positions = np.asarray(
                    whole_body_certificate.get(
                        "selected_eef_dense_positions"
                    ),
                    dtype=np.float32,
                )
                certified_dense_quaternions = np.asarray(
                    whole_body_certificate.get(
                        "selected_eef_dense_quaternions_xyzw"
                    ),
                    dtype=np.float32,
                )
                if (
                    certified_positions.shape != (len(q_chunk), 3)
                    or certified_quaternions.shape != (len(q_chunk), 4)
                    or certified_dense_positions.shape
                    != (len(certified_dense_q), 3)
                    or certified_dense_quaternions.shape
                    != (len(certified_dense_q), 4)
                    or not np.isfinite(certified_positions).all()
                    or not np.isfinite(certified_quaternions).all()
                    or not np.isfinite(certified_dense_positions).all()
                    or not np.isfinite(certified_dense_quaternions).all()
                ):
                    raise RuntimeError(
                        "whole-body EEF certificate payload shape or finiteness "
                        "is invalid"
                    )
                positions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        certified_positions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                quaternions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        certified_quaternions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                dense_positions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        certified_dense_positions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                dense_quaternions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        certified_dense_quaternions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                recomputed = _call_optional_kw(
                    self.backend,
                    "whole_body_eef_poses",
                    hand=hand,
                    q_trajectory=q_chunk,
                )
                if not isinstance(recomputed, (tuple, list)) or len(recomputed) != 2:
                    raise RuntimeError(
                        "whole-body EEF certificate FK verification is unavailable"
                    )
                recomputed_positions = np.asarray(
                    recomputed[0], dtype=np.float32
                ).reshape(-1, 3)
                recomputed_quaternions = np.asarray(
                    recomputed[1], dtype=np.float32
                ).reshape(-1, 4)
                recomputed_positions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        recomputed_positions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                recomputed_quaternions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        recomputed_quaternions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                recomputed_dense = _call_optional_kw(
                    self.backend,
                    "whole_body_eef_poses",
                    hand=hand,
                    q_trajectory=certified_dense_q,
                )
                if (
                    not isinstance(recomputed_dense, (tuple, list))
                    or len(recomputed_dense) != 2
                ):
                    raise RuntimeError(
                        "whole-body dense EEF certificate FK verification "
                        "is unavailable"
                    )
                recomputed_dense_positions = np.asarray(
                    recomputed_dense[0], dtype=np.float32
                ).reshape(-1, 3)
                recomputed_dense_quaternions = np.asarray(
                    recomputed_dense[1], dtype=np.float32
                ).reshape(-1, 4)
                recomputed_dense_positions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        recomputed_dense_positions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                recomputed_dense_quaternions_digest = hashlib.sha256(
                    np.ascontiguousarray(
                        recomputed_dense_quaternions, dtype=np.float32
                    ).tobytes()
                ).hexdigest()
                target_array = np.asarray(
                    target_xyz, dtype=np.float64
                ).reshape(-1)
                target_quat_array = (
                    _quat_xyzw(target_quat_xyzw)
                    if target_quat_xyzw is not None
                    else None
                )
                target_quat_digest = (
                    hashlib.sha256(
                        np.ascontiguousarray(
                            target_quat_array, dtype=np.float32
                        ).tobytes()
                    ).hexdigest()
                    if target_quat_array is not None
                    else None
                )
                certified_terminal_position_error = (
                    float(
                        np.linalg.norm(
                            certified_positions[-1].astype(np.float64)
                            - target_array
                        )
                    )
                    if target_array.shape == (3,) and len(certified_positions)
                    else float("inf")
                )
                certified_terminal_orientation_error = (
                    _quat_angle_error_rad(
                        certified_quaternions[-1],
                        target_quat_array,
                    )
                    if target_quat_array is not None
                    and len(certified_quaternions)
                    else 0.0
                )
                assert current_q_array is not None
                execution_q_with_start = np.vstack(
                    [current_q_array.reshape(1, -1), q_chunk]
                ).astype(np.float32)
                execution_positions_with_start = np.vstack(
                    [
                        certified_dense_positions[0].reshape(1, 3),
                        certified_positions,
                    ]
                )
                execution_quaternions_with_start = np.vstack(
                    [
                        certified_dense_quaternions[0].reshape(1, 4),
                        certified_quaternions,
                    ]
                )
                certified_short_target = whole_body_certificate.get(
                    "selected_eef_short_target"
                )
                execution_step_report = _whole_body_execution_step_report(
                    execution_q_with_start,
                    execution_positions_with_start,
                    execution_quaternions_with_start,
                    joint_names=tuple(str(name) for name in joint_name_layout),
                    short_target=bool(certified_short_target),
                )
                dense_step_report = _whole_body_execution_step_report(
                    certified_dense_q,
                    certified_dense_positions,
                    certified_dense_quaternions,
                    joint_names=tuple(str(name) for name in joint_name_layout),
                    short_target=bool(certified_short_target),
                )
                certificate_checks = {
                    "schema": whole_body_certificate.get("schema_version") == 3,
                    "hand": whole_body_certificate.get("selected_hand")
                    == _normalize_hand(hand),
                    "target_xyz": whole_body_certificate.get(
                        "selected_target_xyz_sha256"
                    )
                    == _whole_body_target_sha256(target_xyz),
                    "target_quat": whole_body_certificate.get(
                        "selected_target_quat_xyzw_sha256"
                    )
                    == target_quat_digest,
                    "admitted": whole_body_certificate.get(
                        "selected_eef_path_admitted"
                    )
                    is True,
                    "short_target_type": isinstance(
                        whole_body_certificate.get("selected_eef_short_target"),
                        bool,
                    ),
                    "position_shape": certified_positions.shape
                    == (len(q_chunk), 3),
                    "quaternion_shape": certified_quaternions.shape
                    == (len(q_chunk), 4),
                    "position_finite": bool(
                        np.isfinite(certified_positions).all()
                    ),
                    "quaternion_finite": bool(
                        np.isfinite(certified_quaternions).all()
                    ),
                    "position_digest": whole_body_certificate.get(
                        "selected_eef_execution_positions_sha256"
                    )
                    == positions_digest,
                    "quaternion_digest": whole_body_certificate.get(
                        "selected_eef_execution_quaternions_sha256"
                    )
                    == quaternions_digest,
                    "recomputed_position_shape": recomputed_positions.shape
                    == certified_positions.shape,
                    "recomputed_quaternion_shape": recomputed_quaternions.shape
                    == certified_quaternions.shape,
                    "recomputed_position_digest": (
                        recomputed_positions_digest == positions_digest
                    ),
                    "recomputed_quaternion_digest": (
                        recomputed_quaternions_digest == quaternions_digest
                    ),
                    "dense_position_shape": certified_dense_positions.shape
                    == (len(certified_dense_q), 3),
                    "dense_quaternion_shape": certified_dense_quaternions.shape
                    == (len(certified_dense_q), 4),
                    "dense_position_finite": bool(
                        np.isfinite(certified_dense_positions).all()
                    ),
                    "dense_quaternion_finite": bool(
                        np.isfinite(certified_dense_quaternions).all()
                    ),
                    "dense_position_digest": whole_body_certificate.get(
                        "selected_eef_dense_positions_sha256"
                    )
                    == dense_positions_digest,
                    "dense_quaternion_digest": whole_body_certificate.get(
                        "selected_eef_dense_quaternions_sha256"
                    )
                    == dense_quaternions_digest,
                    "dense_position_path_digest": whole_body_certificate.get(
                        "selected_eef_positions_sha256"
                    )
                    == dense_positions_digest,
                    "recomputed_dense_position_shape": (
                        recomputed_dense_positions.shape
                        == certified_dense_positions.shape
                    ),
                    "recomputed_dense_quaternion_shape": (
                        recomputed_dense_quaternions.shape
                        == certified_dense_quaternions.shape
                    ),
                    "recomputed_dense_position_digest": (
                        recomputed_dense_positions_digest
                        == dense_positions_digest
                    ),
                    "recomputed_dense_quaternion_digest": (
                        recomputed_dense_quaternions_digest
                        == dense_quaternions_digest
                    ),
                    "execution_dense_position_subset": np.array_equal(
                        certified_dense_positions[
                            collision_dense_indices[1:]
                        ],
                        certified_positions,
                    ),
                    "execution_dense_quaternion_subset": np.array_equal(
                        certified_dense_quaternions[
                            collision_dense_indices[1:]
                        ],
                        certified_quaternions,
                    ),
                    "execution_step_admitted": (
                        execution_step_report.get("admitted") is True
                    ),
                    "dense_step_admitted": (
                        dense_step_report.get("admitted") is True
                    ),
                    "terminal_position": certified_terminal_position_error
                    <= WHOLE_BODY_EEF_TERMINAL_TOLERANCE_M + 1e-9,
                    "terminal_orientation": certified_terminal_orientation_error
                    <= WHOLE_BODY_EEF_SHORT_MAX_ORIENTATION_ERROR_RAD + 1e-9,
                    "position_tolerance": whole_body_certificate.get(
                        "selected_eef_live_waypoint_position_tolerance_m"
                    )
                    == WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M,
                    "orientation_tolerance": whole_body_certificate.get(
                        "selected_eef_live_waypoint_orientation_tolerance_rad"
                    )
                    == WHOLE_BODY_EEF_LIVE_WAYPOINT_ORIENTATION_TOLERANCE_RAD,
                    "controller_response_margin": whole_body_certificate.get(
                        "selected_eef_controller_response_margin_m"
                    )
                    == WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M,
                    "numerical_margin": whole_body_certificate.get(
                        "selected_eef_numerical_margin_m"
                    )
                    == WHOLE_BODY_EEF_NUMERICAL_MARGIN_M,
                    "prospective_guard_margin": whole_body_certificate.get(
                        "selected_eef_prospective_guard_margin_m"
                    )
                    == WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M,
                    "execution_base_xy_step_limit": (
                        whole_body_certificate.get(
                            "execution_base_xy_step_limit_m"
                        )
                        == WHOLE_BODY_EXECUTION_BASE_XY_STEP_M
                    ),
                    "execution_base_yaw_step_limit": (
                        whole_body_certificate.get(
                            "execution_base_yaw_step_limit_rad"
                        )
                        == WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD
                    ),
                    "execution_articulation_step_limit": (
                        whole_body_certificate.get(
                            "execution_articulation_step_limit_rad"
                        )
                        == WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD
                    ),
                    "terminal_command_limit": whole_body_certificate.get(
                        "terminal_command_limit"
                    )
                    == TERMINAL_COMMAND_LIMIT,
                    "terminal_eef_position_tolerance": (
                        whole_body_certificate.get(
                            "terminal_eef_position_tolerance_m"
                        )
                        == EEF_TERMINAL_POSITION_TOLERANCE_M
                    ),
                    "terminal_eef_orientation_tolerance": (
                        whole_body_certificate.get(
                            "terminal_eef_orientation_tolerance_rad"
                        )
                        == EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD
                    ),
                }
                if not all(certificate_checks.values()):
                    failed_checks = sorted(
                        name
                        for name, passed in certificate_checks.items()
                        if not passed
                    )
                    raise RuntimeError(
                        "whole-body EEF path does not match its collision "
                        f"certificate: {failed_checks}"
                    )
                pose = (
                    (
                        certified_dense_positions[0],
                        certified_dense_quaternions[0],
                    )
                    if prepared_start_q is not None
                    else _call_optional_arg(self.backend, "get_eef_pose", hand)
                )
                if not isinstance(pose, (tuple, list)) or len(pose) != 2:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="pose_feedback_unavailable",
                        recoverable=False,
                        suggested_next_tool="observe",
                        executed=0,
                        trace=[],
                        final_pos_err=None,
                        final_ori_err=None,
                        held_steps=0,
                        started=started,
                    )
                call_start_xyz = np.asarray(pose[0], dtype=np.float64).reshape(-1)
                call_start_quat = _quat_xyzw(pose[1])
                if (
                    call_start_xyz.shape != (3,)
                    or target_array.shape != (3,)
                    or not np.isfinite(call_start_xyz).all()
                    or not np.isfinite(target_array).all()
                    or call_start_quat is None
                ):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="pose_feedback_unavailable",
                        recoverable=False,
                        suggested_next_tool="observe",
                        executed=0,
                        trace=[],
                        final_pos_err=None,
                        final_ori_err=None,
                        held_steps=0,
                        started=started,
                    )
                direct = target_array - call_start_xyz
                direct_distance = float(np.linalg.norm(direct))
                direct_axis = (
                    direct / direct_distance
                    if direct_distance > 1e-9
                    else np.zeros(3, dtype=np.float64)
                )
                certified_path_with_start = np.vstack(
                    [
                        call_start_xyz.reshape(1, 3),
                        certified_positions.astype(np.float64),
                    ]
                )
                certified_quats_with_start = np.vstack(
                    [
                        call_start_quat.reshape(1, 4),
                        certified_quaternions.astype(np.float64),
                    ]
                )
                runtime_admission = _eef_pose_path_admission_report(
                    certified_path_with_start,
                    certified_quats_with_start,
                    call_start_xyz=call_start_xyz,
                    target_xyz=target_array,
                    target_quat_xyzw=target_quat_array,
                    short_cartesian_step_limit_m=(
                        WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
                    ),
                )
                dense_runtime_admission = _eef_pose_path_admission_report(
                    certified_dense_positions,
                    certified_dense_quaternions,
                    call_start_xyz=call_start_xyz,
                    target_xyz=target_array,
                    target_quat_xyzw=target_quat_array,
                )
                runtime_short_target = bool(runtime_admission["short_target"])
                lineage_checks = {
                    "short_target_matches_live_call_start": (
                        certified_short_target is runtime_short_target
                    ),
                    "dense_short_target_matches_live_call_start": (
                        bool(dense_runtime_admission["short_target"])
                        is runtime_short_target
                    ),
                    "runtime_eef_path_admitted": (
                        runtime_admission.get("admitted") is True
                    ),
                    "runtime_dense_eef_path_admitted": (
                        dense_runtime_admission.get("admitted") is True
                    ),
                }
                if not all(lineage_checks.values()):
                    failed_lineage_checks = sorted(
                        name
                        for name, passed in lineage_checks.items()
                        if not passed
                    )
                    failed_runtime_checks = sorted(
                        name
                        for report_name, report in (
                            ("execution", runtime_admission),
                            ("dense", dense_runtime_admission),
                        )
                        for name, passed in dict(
                            report.get("checks") or {}
                        ).items()
                        if not passed
                    )
                    raise RuntimeError(
                        "whole-body runtime EEF/collision lineage rejected "
                        f"{failed_lineage_checks}; path={failed_runtime_checks}"
                    )
                whole_body_eef_guard = {
                    "available": True,
                    "execution_policy": execution_policy,
                    "first_sample_axis_reverse_transient_enabled": bool(
                        execution_policy
                        == PREPARED_DASHBOARD_EEF_EXECUTION_POLICY
                        and contact_target_xyz is None
                        and not stop_on_expected_contact
                        and not stop_on_attachment
                        and not require_attachment
                        and expected_attachment is None
                        and not any(
                            value is not None
                            for value in expected_attachments_by_hand.values()
                        )
                    ),
                    "certificate_schema_version": 3,
                    "collision_lineage_mode": (
                        "source_dense_execution_subset_recertified_v1"
                    ),
                    "selected_hand": _normalize_hand(hand),
                    "selected_target_xyz_sha256": _whole_body_target_sha256(
                        target_array
                    ),
                    "selected_eef_execution_positions_sha256": positions_digest,
                    "selected_eef_execution_quaternions_sha256": (
                        quaternions_digest
                    ),
                    "nominal_positions": certified_positions,
                    "nominal_quaternions": certified_quaternions,
                    "call_start_xyz": call_start_xyz,
                    "call_start_quat_xyzw": call_start_quat,
                    "target_xyz": target_array,
                    "direct_axis": direct_axis,
                    "direct_distance_m": direct_distance,
                    "short_target": runtime_short_target,
                    "runtime_admission": runtime_admission,
                    "dense_runtime_admission": dense_runtime_admission,
                    "execution_step_admission": execution_step_report,
                    "dense_step_admission": dense_step_report,
                    "controller_response_margin_m": (
                        WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M
                    ),
                    "numerical_margin_m": (
                        WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                    ),
                    "prospective_guard_margin_m": (
                        WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
                    ),
                    "axis_reverse_defer_ledger": {},
                    "pending_axis_reverse_settle_repeat": None,
                    "replan_checkpoint_trigger_position_improvement_m": (
                        checkpoint_trigger_position_improvement_m
                    ),
                    "replan_checkpoint_outer_minimum_improvement_m": (
                        WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
                    ),
                    "terminal_command_limit": TERMINAL_COMMAND_LIMIT,
                    "terminal_commands_sent": 0,
                    "live_start_equality_checked": (
                        prepared_start_q is None
                    ),
                    "collision_revalidated": False,
                    "live_waypoint_position_tolerance_m": (
                        WHOLE_BODY_EEF_LIVE_WAYPOINT_POSITION_TOLERANCE_M
                    ),
                    "live_waypoint_orientation_tolerance_rad": (
                        WHOLE_BODY_EEF_LIVE_WAYPOINT_ORIENTATION_TOLERANCE_RAD
                    ),
                    "waypoint_settle_position_tolerance_m": (
                        (
                            WHOLE_BODY_EEF_SHORT_WAYPOINT_SETTLE_POSITION_TOLERANCE_M
                            if runtime_short_target
                            else WHOLE_BODY_EEF_WAYPOINT_SETTLE_POSITION_TOLERANCE_M
                        )
                    ),
                    "waypoint_settle_orientation_tolerance_rad": (
                        WHOLE_BODY_EEF_WAYPOINT_SETTLE_ORIENTATION_TOLERANCE_RAD
                    ),
                    "settle_waypoint_index": None,
                    "settle_previous_normalized_error": None,
                    "corridor_margin_m": WHOLE_BODY_EEF_SHORT_CORRIDOR_MARGIN_M,
                    "max_observed_live_waypoint_position_error_m": 0.0,
                    "max_observed_live_waypoint_orientation_error_rad": 0.0,
                    "max_observed_live_lateral_m": 0.0,
                    "max_observed_live_start_excursion_m": 0.0,
                    "max_observed_live_reverse_step_m": 0.0,
                    "cumulative_observed_live_path_m": 0.0,
                    "previous_live_position": call_start_xyz.copy(),
                    "previous_live_along_m": 0.0,
                    "previous_live_quat_xyzw": call_start_quat.copy(),
                    "previous_live_target_orientation_error_rad": (
                        float(
                            runtime_admission[
                                "start_orientation_error_rad"
                            ]
                        )
                    ),
                    "max_observed_live_orientation_step_rad": 0.0,
                    "max_observed_live_orientation_reverse_progress_rad": 0.0,
                    "cumulative_observed_live_orientation_path_rad": 0.0,
                }
        planned_steps = (
            action_chunk.shape[0] if action_chunk is not None else q_chunk.shape[0]
        )
        trajectory_hold_reference: np.ndarray | None = None
        trajectory_fixed_reference: dict[str, Any] | None = None
        trajectory_uses_fixed_reference = False
        trajectory_hand = (
            None
            if base_goal_xyyaw is not None or motion_scope == "whole_body"
            else hand
        )
        if q_chunk is not None:
            trajectory_fixed_reference = _call_optional_kw(
                self.backend,
                "capture_trajectory_hold_reference",
                hand=trajectory_hand,
                **(
                    {"motion_scope": motion_scope} if motion_scope != "arm_only" else {}
                ),
            )
            if isinstance(trajectory_fixed_reference, dict):
                try:
                    converter_parameters = inspect.signature(
                        self.backend.joint_target_to_action
                    ).parameters
                except (AttributeError, TypeError, ValueError):
                    converter_parameters = {}
                trajectory_uses_fixed_reference = (
                    "fixed_reference" in converter_parameters
                )
            else:
                # Compatibility path for small test backends. Real execution
                # always uses the q-space reference above.
                hold_value = _call_optional_arg(
                    self.backend, "hold_action", trajectory_hand
                )
                if hold_value is not None:
                    try:
                        trajectory_hold_reference = validate_action_chunk(
                            np.asarray(hold_value, dtype=np.float32).reshape(
                                1, ACTION_DIM
                            )
                        )[0]
                    except Exception:
                        trajectory_hold_reference = None
        deadline = time.monotonic() + float(timeout_s)
        best_error = float("inf")
        stalled_steps = 0
        held_steps = 0
        executed = 0
        trace: list[dict[str, Any]] = []
        final_pos_err: float | None = None
        final_ori_err: float | None = None
        hold_action: np.ndarray | None = None
        previous_action: np.ndarray | None = None
        contact_stop_active = False
        attachment_seen = False
        attachment_confirmation_steps = 0
        attachment_endpoint_held_steps = 0
        attachment_identity: dict[str, Any] | None = None
        gripper_contact_settle_remaining = 0
        gripper_contact_settle_executed = 0
        gripper_contact_settle_started = False
        gripper_contact_hold_action: np.ndarray | None = None
        index = 0
        waypoint_attempts = 0
        terminal_commands_sent = 0
        final_waypoint_tracking: dict[str, Any] | None = None
        max_steps = (
            max(planned_steps - 1, 0)
            + TERMINAL_COMMAND_LIMIT
            + int(hold_steps_required)
            if whole_body_eef_guard is not None
            else (
                planned_steps * (self.max_stall_steps + 1)
                + int(hold_steps_required)
                + self.max_stall_steps
            )
        )
        while executed < max_steps:
            whole_body_commanded_waypoint_index = (
                min(index, planned_steps - 1)
                if whole_body_eef_guard is not None
                else None
            )
            whole_body_waypoint_command = None
            terminal_command = bool(
                whole_body_commanded_waypoint_index is not None
                and whole_body_commanded_waypoint_index == planned_steps - 1
            )
            if terminal_command and terminal_commands_sent >= TERMINAL_COMMAND_LIMIT:
                break
            if (
                whole_body_commanded_waypoint_index is not None
                and q_chunk is not None
            ):
                commanded_q = np.ascontiguousarray(
                    q_chunk[whole_body_commanded_waypoint_index],
                    dtype=np.float32,
                )
                commanded_q_sha256 = hashlib.sha256(
                    commanded_q.tobytes()
                ).hexdigest()
                pending_axis_reverse_repeat = (
                    whole_body_eef_guard.get(
                        "pending_axis_reverse_settle_repeat"
                    )
                    if whole_body_eef_guard is not None
                    else None
                )
                if pending_axis_reverse_repeat is not None and (
                    not isinstance(pending_axis_reverse_repeat, dict)
                    or pending_axis_reverse_repeat.get("index")
                    != whole_body_commanded_waypoint_index
                    or pending_axis_reverse_repeat.get("q_sha256")
                    != commanded_q_sha256
                ):
                    raise RuntimeError(
                        "prepared Dashboard EEF transient repeat lost exact-q lineage"
                    )
                whole_body_waypoint_command = {
                    "index": whole_body_commanded_waypoint_index,
                    "q_sha256": commanded_q_sha256,
                    "trajectory_sha256": (
                        whole_body_certificate.get("trajectory_sha256")
                        if isinstance(whole_body_certificate, dict)
                        else None
                    ),
                    "settle_repeat": (
                        (
                            terminal_command
                            and terminal_commands_sent > 0
                        )
                        or pending_axis_reverse_repeat is not None
                    ),
                    "terminal_command": terminal_command,
                    "terminal_command_index": (
                        terminal_commands_sent + 1
                        if terminal_command
                        else None
                    ),
                }
            attachment_confirmed = False
            if time.monotonic() > deadline:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="timeout",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "final_joint_tracking": final_waypoint_tracking,
                        "trajectory_complete": index >= planned_steps,
                    },
                )
            if gripper_contact_settle_remaining > 0:
                assert gripper_contact_hold_action is not None
                action = gripper_contact_hold_action
                gripper_contact_settle_remaining -= 1
                gripper_contact_settle_executed += 1
            elif stop_on_attachment and attachment_seen and previous_action is not None:
                # Once OG creates the exact expected constraint, preserve the
                # current opening instead of continuing to squeeze the body.
                action = previous_action
            elif index < planned_steps:
                if action_chunk is not None:
                    action = action_chunk[index]
                    index += 1
                else:
                    if trajectory_uses_fixed_reference:
                        action = self.backend.joint_target_to_action(
                            q_chunk[index],
                            hand=trajectory_hand,
                            fixed_reference=trajectory_fixed_reference,
                        )
                    else:
                        action = self.backend.joint_target_to_action(
                            q_chunk[index], hand=trajectory_hand
                        )
            else:
                if q_chunk is not None:
                    if trajectory_uses_fixed_reference:
                        action = self.backend.joint_target_to_action(
                            q_chunk[-1],
                            hand=trajectory_hand,
                            fixed_reference=trajectory_fixed_reference,
                        )
                    else:
                        action = self.backend.joint_target_to_action(
                            q_chunk[-1], hand=trajectory_hand
                        )
                else:
                    if hold_action is None:
                        if static_gripper_only and previous_action is not None:
                            # A gripper confirmation window must preserve the
                            # exact accepted endpoint command. A generic robot
                            # hold can contain a stale gripper value and would
                            # otherwise reopen the fingers during validation.
                            hold_action = previous_action.copy()
                        else:
                            hold_action = _call_optional_arg(
                                self.backend, "hold_action", hand
                            )
                            if hold_action is None:
                                hold_action = np.zeros((ACTION_DIM,), dtype=np.float32)
                    action = hold_action
            action = validate_action_chunk(
                np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM)
            )[0]
            if q_chunk is not None and not trajectory_uses_fixed_reference:
                assert trajectory_hold_reference is not None
                action = _apply_fixed_trajectory_hold_segments(
                    action,
                    trajectory_hold_reference,
                    hand=trajectory_hand,
                    motion_scope=motion_scope,
                )
            if isolation_required:
                assert isolation_reference is not None
                isolation_hold = _call_optional_kw(
                    self.backend,
                    "single_arm_isolation_hold_action",
                    reference=isolation_reference,
                )
                if isolation_hold is None:
                    isolation_hold = _call_optional_arg(
                        self.backend,
                        "hold_action",
                        hand,
                    )
                try:
                    isolation_hold = validate_action_chunk(
                        np.asarray(isolation_hold, dtype=np.float32).reshape(
                            1, ACTION_DIM
                        )
                    )[0]
                except Exception:
                    assert self._active_isolation_report is not None
                    self._active_isolation_report.update(
                        {
                            "available": False,
                            "ok": False,
                            "reason": "single-arm isolation hold action unavailable",
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="single_arm_isolation_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                    )
                action = _apply_single_arm_isolation_mask(
                    action,
                    isolation_hold,
                    hand=hand,
                    gripper_only=isolation_gripper_only,
                    motion_scope=motion_scope,
                )
                action = validate_action_chunk(action.reshape(1, ACTION_DIM))[0]
            previous_action = action.copy()
            if action_counter is not None:
                action_counter["attempted"] += 1
            step_receipt = self._step_env_action(action)
            executed += 1
            if terminal_command:
                terminal_commands_sent += 1
                assert whole_body_eef_guard is not None
                whole_body_eef_guard["terminal_commands_sent"] = (
                    terminal_commands_sent
                )
            if action_counter is not None:
                action_counter["confirmed"] += 1
            if static_gripper_only:
                # The env receipt proves this exact command was executed once.
                # Commit the latch before interpreting a terminal raw-success
                # receipt so the successful final frame preserves the accepted
                # gripper pressure without sending a post-success hold action.
                latch = getattr(self.env, "_gripper_latch", None)
                if isinstance(latch, dict):
                    gripper_segment = ENV_ACTION_SEGMENTS[f"{hand}_gripper"]
                    latch[hand] = float(action[gripper_segment][0])
            terminal_outcome = _terminal_step_outcome(step_receipt)
            if terminal_outcome is not None:
                primitive_success, stop_reason = terminal_outcome
                trace.append(
                    {
                        "step": executed,
                        "step_receipt": dict(step_receipt),
                        "whole_body_waypoint_command": whole_body_waypoint_command,
                    }
                )
                return self._execution_result(
                    primitive_success=primitive_success,
                    stop_reason=stop_reason,
                    recoverable=False,
                    suggested_next_tool=None,
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "terminal_step_receipt": dict(step_receipt),
                    },
                )
            if isolation_required:
                assert isolation_reference is not None
                isolation_kwargs = {
                    "hand": hand,
                    "action": action,
                    "reference": isolation_reference,
                    "gripper_only": isolation_gripper_only,
                }
                if motion_scope == "arm_with_trunk":
                    isolation_kwargs["motion_scope"] = motion_scope
                isolation_report = _call_optional_kw(
                    self.backend,
                    "single_arm_isolation_report",
                    **isolation_kwargs,
                )
                if not isinstance(isolation_report, dict):
                    isolation_report = {
                        "available": False,
                        "ok": False,
                        "selected_hand": _normalize_hand(hand),
                        "mode": (
                            "gripper_only"
                            if isolation_gripper_only
                            else (
                                "arm_motion"
                                if motion_scope == "arm_only"
                                else motion_scope
                            )
                        ),
                        "context_id": isolation_reference.get("context_id"),
                        "reference_origin": isolation_reference.get(
                            "reference_origin", "primitive_call_start"
                        ),
                        "reason": "single-arm isolation report unavailable",
                        "checks": {},
                        "max_observed": {},
                    }
                self._active_isolation_report = self._merge_isolation_report(
                    self._active_isolation_report,
                    isolation_report,
                )
                if not bool(isolation_report.get("available", False)):
                    trace.append(
                        {
                            "step": executed,
                            "single_arm_isolation": isolation_report,
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="single_arm_isolation_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                    )
                if not bool(isolation_report.get("ok", False)):
                    trace.append(
                        {
                            "step": executed,
                            "single_arm_isolation": isolation_report,
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="single_arm_isolation_violation",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                    )
            contact: dict[str, Any] | None = None
            if motion_scope == "whole_body":
                assert expected_attachments_by_hand is not None
                for side in ("left", "right"):
                    live = _call_optional_arg(self.backend, "get_attached_object", side)
                    matches, identity = _attachment_state_status(
                        live,
                        expected_attachments_by_hand[side],
                        hand=side,
                    )
                    if not matches:
                        trace.append(
                            {
                                "step": executed,
                                "whole_body_attachment": {
                                    "hand": side,
                                    **identity,
                                },
                                "whole_body_waypoint_command": (
                                    whole_body_waypoint_command
                                ),
                            }
                        )
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="attachment_identity_mismatch",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={
                                "whole_body_attachment": {
                                    "hand": side,
                                    **identity,
                                }
                            },
                        )
                if stop_on_expected_contact and contact_target_xyz is not None:
                    contact = self._contact_report(
                        hand=hand,
                        target_xyz=contact_target_xyz,
                        allowed_contact_distance_m=max(
                            float(allowed_contact_distance_m),
                            float(position_tolerance_m) * 2.0,
                        ),
                    )
                    if not bool(contact.get("available", False)):
                        trace.append(
                            {
                                "step": executed,
                                "contact_report": contact,
                                "whole_body_waypoint_command": (
                                    whole_body_waypoint_command
                                ),
                            }
                        )
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="contact_feedback_unavailable",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={"contact_report": contact},
                        )
                whole_body_contact = _call_optional_kw(
                    self.backend,
                    "whole_body_contact_report",
                    baseline=whole_body_contact_baseline,
                    expected_attachments_by_hand=expected_attachments_by_hand,
                    allowed_expected_contact=(
                        contact
                        if isinstance(contact, dict)
                        and contact.get("expected_contact") is True
                        else None
                    ),
                )
                if not isinstance(whole_body_contact, dict) or not bool(
                    whole_body_contact.get("available", False)
                ):
                    trace.append(
                        {
                            "step": executed,
                            "whole_body_contact": whole_body_contact,
                            "whole_body_waypoint_command": (
                                whole_body_waypoint_command
                            ),
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="contact_feedback_unavailable",
                        recoverable=False,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "whole_body_contact": whole_body_contact,
                        },
                    )
                if bool(whole_body_contact.get("unexpected_contact", False)):
                    trace.append(
                        {
                            "step": executed,
                            "whole_body_contact": whole_body_contact,
                            "whole_body_waypoint_command": (
                                whole_body_waypoint_command
                            ),
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="unexpected_contact",
                        recoverable=False,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "whole_body_contact": whole_body_contact,
                        },
                    )
                if (
                    isinstance(contact, dict)
                    and contact.get("expected_contact") is True
                ):
                    trace.append(
                        {
                            "step": executed,
                            "contact_report": contact,
                            "whole_body_contact": whole_body_contact,
                            "whole_body_waypoint_command": (
                                whole_body_waypoint_command
                            ),
                            "expected_contact_stop": True,
                        }
                    )
                    return self._execution_result(
                        primitive_success=True,
                        stop_reason="reached",
                        recoverable=True,
                        suggested_next_tool=None,
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=0,
                        started=started,
                        extra_metrics={
                            "contact_report": contact,
                            "whole_body_contact": whole_body_contact,
                            "expected_contact_seen": True,
                            "whole_body_eef_path_guard": eef_guard_snapshot(),
                        },
                    )
            if stop_on_attachment or require_attachment:
                attached_now = _call_optional_arg(
                    self.backend, "get_attached_object", hand
                )
                attachment_confirmed, attachment_identity = _attachment_identity_status(
                    attached_now,
                    expected_attachment,
                    hand=hand,
                )
                if attached_now is not None and not attachment_confirmed:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="attachment_identity_mismatch",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "attachment_identity": attachment_identity,
                            "attachment_confirmation_steps": (
                                attachment_confirmation_steps
                            ),
                            "attachment_endpoint_held_steps": (
                                attachment_endpoint_held_steps
                            ),
                        },
                    )
                if require_attachment and not attachment_confirmed:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="attachment_lost",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "attachment_identity": attachment_identity,
                            "attachment_confirmation_steps": (
                                attachment_confirmation_steps
                            ),
                            "attachment_endpoint_held_steps": (
                                attachment_endpoint_held_steps
                            ),
                        },
                    )
                if stop_on_attachment:
                    if attachment_confirmed:
                        attachment_seen = True
                        attachment_confirmation_steps += 1
                    elif attachment_seen:
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="attachment_lost",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={
                                "attachment_identity": attachment_identity,
                                "attachment_confirmation_steps": (
                                    attachment_confirmation_steps
                                ),
                            },
                        )
                    else:
                        attachment_confirmation_steps = 0
                elif require_attachment:
                    attachment_seen = True
                    attachment_confirmation_steps += 1
            waypoint_tracking: dict[str, Any] | None = None
            waypoint_advanced = False
            waypoint_stalled = False
            whole_body_joint_waypoint_reached = False
            whole_body_eef_waypoint_trace: dict[str, Any] | None = None
            whole_body_eef_settle_no_progress = False
            whole_body_replan_checkpoint: dict[str, Any] | None = None
            if q_chunk is not None and index < planned_steps:
                waypoint_attempts += 1
                waypoint_tracking = _call_optional_kw(
                    self.backend,
                    "joint_tracking_report",
                    target_q=q_chunk[index],
                    hand=(
                        None
                        if base_goal_xyyaw is not None or motion_scope == "whole_body"
                        else hand
                    ),
                )
                if not isinstance(waypoint_tracking, dict) or not bool(
                    waypoint_tracking.get("available", False)
                ):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="joint_tracking_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"joint_tracking": waypoint_tracking},
                    )
                final_waypoint_tracking = waypoint_tracking
                if whole_body_eef_guard is not None:
                    hard_tracking, hard_tracking_available = (
                        _tracking_hard_deviation_report(waypoint_tracking)
                    )
                    if not hard_tracking_available:
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="joint_tracking_feedback_unavailable",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={
                                "tracking_hard_deviation": hard_tracking,
                            },
                        )
                    if hard_tracking["hard_deviation"] is True:
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="tracking_hard_deviation",
                            recoverable=False,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={
                                "tracking_hard_deviation": hard_tracking,
                                "whole_body_waypoint_command": (
                                    whole_body_waypoint_command
                                ),
                            },
                        )
                if bool(waypoint_tracking.get("reached", False)):
                    if whole_body_eef_guard is not None:
                        # A loose 21-DOF joint-space tolerance must not advance
                        # a whole-body path while the selected EEF is still
                        # lagging its collision-certified Cartesian waypoint.
                        # The post-step EEF feedback below is the second half of
                        # this combined settle gate.
                        whole_body_joint_waypoint_reached = True
                    else:
                        index += 1
                        waypoint_advanced = True
                        waypoint_attempts = 0
                elif (
                    whole_body_eef_guard is None
                    and waypoint_attempts >= self.max_stall_steps
                ):
                    waypoint_stalled = True
            if base_goal_xyyaw is not None:
                base_pose = _call_optional(self.backend, "get_base_pose")
                if base_pose is None:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="pose_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                    )
                try:
                    base_pose = np.asarray(
                        base_pose, dtype=np.float64
                    ).reshape(-1)
                except (TypeError, ValueError):
                    base_pose = np.asarray([], dtype=np.float64)
                if base_pose.shape != (3,) or not np.isfinite(base_pose).all():
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="pose_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                    )
                final_pos_err = float(
                    np.linalg.norm(base_pose[:2] - base_goal_xyyaw[:2])
                )
                final_ori_err = abs(
                    _wrap_angle(float(base_pose[2]) - float(base_goal_xyyaw[2]))
                )
            else:
                pose = (
                    self.backend.get_eef_pose(hand)
                    if hasattr(self.backend, "get_eef_pose")
                    else None
                )
                if pose is None:
                    if require_pose:
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="pose_feedback_unavailable",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                        )
                    final_pos_err = None
                    final_ori_err = None
                else:
                    try:
                        pos = np.asarray(pose[0], dtype=np.float64).reshape(-1)
                        quat = _quat_xyzw(pose[1])
                    except (TypeError, ValueError):
                        pos = np.asarray([], dtype=np.float64)
                        quat = None
                    if (
                        pos.shape != (3,)
                        or not np.isfinite(pos).all()
                        or (
                            (
                                require_pose
                                or motion_scope == "whole_body"
                                or target_quat_xyzw is not None
                            )
                            and quat is None
                        )
                    ):
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="pose_feedback_unavailable",
                            recoverable=True,
                            suggested_next_tool="observe",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                        )
                    if target_xyz is not None:
                        final_pos_err = float(np.linalg.norm(pos - target_xyz))
                    if target_quat_xyzw is not None:
                        final_ori_err = _quat_angle_error_rad(quat, target_quat_xyzw)
            if (
                motion_scope == "whole_body"
                and whole_body_eef_guard is not None
                and pose is not None
                and whole_body_commanded_waypoint_index is not None
            ):
                live_position = np.asarray(pose[0], dtype=np.float64).reshape(3)
                live_quat = _quat_xyzw(pose[1])
                nominal_position = np.asarray(
                    whole_body_eef_guard["nominal_positions"][
                        whole_body_commanded_waypoint_index
                    ],
                    dtype=np.float64,
                ).reshape(3)
                nominal_quat = np.asarray(
                    whole_body_eef_guard["nominal_quaternions"][
                        whole_body_commanded_waypoint_index
                    ],
                    dtype=np.float64,
                ).reshape(4)
                configured_position_settle_tolerance = float(
                    whole_body_eef_guard[
                        "waypoint_settle_position_tolerance_m"
                    ]
                )
                prospective_next_waypoint_step = 0.0
                if (
                    whole_body_commanded_waypoint_index + 1
                    < len(whole_body_eef_guard["nominal_positions"])
                ):
                    next_nominal_position = np.asarray(
                        whole_body_eef_guard["nominal_positions"][
                            whole_body_commanded_waypoint_index + 1
                        ],
                        dtype=np.float64,
                    ).reshape(3)
                    prospective_next_waypoint_step = float(
                        np.linalg.norm(
                            next_nominal_position - nominal_position
                        )
                    )
                effective_position_settle_tolerance = min(
                    configured_position_settle_tolerance,
                    max(
                        0.0,
                        float(
                            whole_body_eef_guard[
                                "live_waypoint_position_tolerance_m"
                            ]
                        )
                        - prospective_next_waypoint_step
                        - WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M,
                    ),
                )
                waypoint_position_error = float(
                    np.linalg.norm(live_position - nominal_position)
                )
                waypoint_orientation_error = (
                    _quat_angle_error_rad(live_quat, nominal_quat)
                    if live_quat is not None
                    else float("inf")
                )
                relative = live_position - np.asarray(
                    whole_body_eef_guard["call_start_xyz"],
                    dtype=np.float64,
                )
                start_excursion = float(np.linalg.norm(relative))
                direct_axis = np.asarray(
                    whole_body_eef_guard["direct_axis"],
                    dtype=np.float64,
                )
                along = float(relative @ direct_axis)
                lateral = float(
                    np.linalg.norm(relative - along * direct_axis)
                )
                previous_live_position = np.asarray(
                    whole_body_eef_guard["previous_live_position"],
                    dtype=np.float64,
                ).reshape(3)
                live_step_distance = float(
                    np.linalg.norm(live_position - previous_live_position)
                )
                previous_live_along = float(
                    whole_body_eef_guard["previous_live_along_m"]
                )
                live_reverse_step = max(0.0, previous_live_along - along)
                cumulative_live_path = (
                    float(
                        whole_body_eef_guard[
                            "cumulative_observed_live_path_m"
                        ]
                    )
                    + live_step_distance
                )
                previous_live_quat = np.asarray(
                    whole_body_eef_guard["previous_live_quat_xyzw"],
                    dtype=np.float64,
                ).reshape(4)
                live_orientation_step = float(
                    _quat_angle_error_rad(previous_live_quat, live_quat) or 0.0
                )
                live_target_orientation_error = float(
                    _quat_angle_error_rad(live_quat, target_quat_xyzw) or 0.0
                )
                previous_target_orientation_error = float(
                    whole_body_eef_guard[
                        "previous_live_target_orientation_error_rad"
                    ]
                )
                live_orientation_reverse_progress = max(
                    0.0,
                    live_target_orientation_error
                    - previous_target_orientation_error,
                )
                cumulative_live_orientation_path = (
                    float(
                        whole_body_eef_guard[
                            "cumulative_observed_live_orientation_path_rad"
                        ]
                    )
                    + live_orientation_step
                )
                direct_distance = float(
                    whole_body_eef_guard["direct_distance_m"]
                )
                corridor_margin = float(
                    whole_body_eef_guard["corridor_margin_m"]
                )
                whole_body_eef_guard[
                    "max_observed_live_waypoint_position_error_m"
                ] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_waypoint_position_error_m"
                        ]
                    ),
                    waypoint_position_error,
                )
                whole_body_eef_guard[
                    "max_observed_live_waypoint_orientation_error_rad"
                ] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_waypoint_orientation_error_rad"
                        ]
                    ),
                    waypoint_orientation_error,
                )
                whole_body_eef_guard["max_observed_live_lateral_m"] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_lateral_m"
                        ]
                    ),
                    lateral,
                )
                whole_body_eef_guard[
                    "max_observed_live_start_excursion_m"
                ] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_start_excursion_m"
                        ]
                    ),
                    start_excursion,
                )
                whole_body_eef_guard[
                    "max_observed_live_reverse_step_m"
                ] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_reverse_step_m"
                        ]
                    ),
                    live_reverse_step,
                )
                whole_body_eef_guard[
                    "cumulative_observed_live_path_m"
                ] = cumulative_live_path
                whole_body_eef_guard["previous_live_position"] = (
                    live_position.copy()
                )
                whole_body_eef_guard["previous_live_along_m"] = along
                whole_body_eef_guard["previous_live_quat_xyzw"] = (
                    live_quat.copy()
                )
                whole_body_eef_guard[
                    "previous_live_target_orientation_error_rad"
                ] = live_target_orientation_error
                whole_body_eef_guard[
                    "max_observed_live_orientation_step_rad"
                ] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_orientation_step_rad"
                        ]
                    ),
                    live_orientation_step,
                )
                whole_body_eef_guard[
                    "max_observed_live_orientation_reverse_progress_rad"
                ] = max(
                    float(
                        whole_body_eef_guard[
                            "max_observed_live_orientation_reverse_progress_rad"
                        ]
                    ),
                    live_orientation_reverse_progress,
                )
                whole_body_eef_guard[
                    "cumulative_observed_live_orientation_path_rad"
                ] = cumulative_live_orientation_path
                intentional_rotation = bool(
                    dict(
                        whole_body_eef_guard["runtime_admission"]
                    ).get("intentional_orientation_change")
                )
                rotation_path_limit = (
                    float(
                        dict(
                            whole_body_eef_guard["runtime_admission"]
                        ).get("start_orientation_error_rad", 0.0)
                    )
                    + WHOLE_BODY_EEF_WRIST_MAX_CUMULATIVE_EXCESS_RAD
                )
                violations = {
                    "waypoint_position": waypoint_position_error
                    > TRACKING_HARD_BASE_XY_ERROR_M + 1e-9,
                    "waypoint_orientation": waypoint_orientation_error
                    > TRACKING_HARD_BASE_YAW_ERROR_RAD + 1e-9,
                    "start_excursion": bool(
                        whole_body_eef_guard["short_target"]
                        and start_excursion
                        > direct_distance + corridor_margin + 1e-9
                    ),
                    "lateral": bool(
                        whole_body_eef_guard["short_target"]
                        and lateral > corridor_margin + 1e-9
                    ),
                    "reverse_or_overshoot": (
                        bool(whole_body_eef_guard["short_target"])
                        and (
                            along < -corridor_margin - 1e-9
                            or along
                            > direct_distance + corridor_margin + 1e-9
                        )
                    ),
                    "axis_monotonic": bool(
                        whole_body_eef_guard["short_target"]
                        and direct_distance > 1e-9
                        and live_reverse_step
                        > WHOLE_BODY_EEF_SHORT_MAX_REVERSE_STEP_M + 1e-9
                    ),
                    "cumulative_path": bool(
                        whole_body_eef_guard["short_target"]
                        and cumulative_live_path
                        > direct_distance
                        + WHOLE_BODY_EEF_SHORT_MAX_CUMULATIVE_EXCESS_M
                        + 1e-9
                    ),
                    "orientation_step": bool(
                        intentional_rotation
                        and live_orientation_step
                        > WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD + 1e-9
                    ),
                    "orientation_monotonic": bool(
                        intentional_rotation
                        and live_orientation_reverse_progress
                        > WHOLE_BODY_EEF_WRIST_MAX_REVERSE_PROGRESS_RAD + 1e-9
                    ),
                    "orientation_cumulative_path": bool(
                        intentional_rotation
                        and cumulative_live_orientation_path
                        > rotation_path_limit + 1e-9
                    ),
                }
                eef_pose_settled = bool(
                    final_pos_err is not None
                    and final_pos_err
                    <= min(
                        float(position_tolerance_m),
                        EEF_TERMINAL_POSITION_TOLERANCE_M,
                    )
                    + 1e-9
                    and (
                        final_ori_err is None
                        or final_ori_err
                        <= min(
                            float(orientation_tolerance_rad),
                            EEF_TERMINAL_ORIENTATION_TOLERANCE_RAD,
                        )
                        + 1e-9
                    )
                )
                waypoint_settled = eef_pose_settled
                waypoint_q_sha256 = (
                    str(whole_body_waypoint_command["q_sha256"])
                    if whole_body_waypoint_command is not None
                    else ""
                )
                fatal_violations = dict(violations)
                deferred_violations: dict[str, str] = {}
                waypoint_axis_reverse_deferred = False
                axis_reverse_defer_key = None
                axis_reverse_settle_repeat_command = bool(
                    whole_body_waypoint_command is not None
                    and whole_body_waypoint_command.get("settle_repeat") is True
                    and whole_body_eef_guard.get(
                        "pending_axis_reverse_settle_repeat"
                    )
                    is not None
                )
                if axis_reverse_settle_repeat_command:
                    whole_body_eef_guard[
                        "pending_axis_reverse_settle_repeat"
                    ] = None
                violation_names = {
                    name for name, violated in violations.items() if violated
                }
                axis_reverse_defer_ledger = whole_body_eef_guard[
                    "axis_reverse_defer_ledger"
                ]
                if (
                    whole_body_eef_guard[
                        "first_sample_axis_reverse_transient_enabled"
                    ]
                    and executed == 1
                    and whole_body_commanded_waypoint_index == 0
                    and waypoint_attempts == 1
                    and not axis_reverse_defer_ledger
                    and not axis_reverse_settle_repeat_command
                    and violation_names == {"axis_monotonic"}
                    and live_reverse_step
                    <= WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M
                    + 1e-9
                ):
                    fatal_violations["axis_monotonic"] = False
                    deferred_violations["axis_monotonic"] = (
                        "prepared_dashboard_first_sample_controller_transient"
                    )
                    waypoint_axis_reverse_deferred = True
                    axis_reverse_defer_key = (
                        f"{whole_body_commanded_waypoint_index}:"
                        f"{waypoint_q_sha256}"
                    )
                    axis_reverse_defer_ledger[axis_reverse_defer_key] = {
                        "index": whole_body_commanded_waypoint_index,
                        "q_sha256": waypoint_q_sha256,
                        "executed_sample": executed,
                        "reverse_step_m": live_reverse_step,
                        "maximum_reverse_step_m": (
                            WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M
                        ),
                        "exact_same_q_repeat_required": True,
                    }
                    whole_body_eef_guard[
                        "pending_axis_reverse_settle_repeat"
                    ] = {
                        "index": whole_body_commanded_waypoint_index,
                        "q_sha256": waypoint_q_sha256,
                    }
                normalized_settle_error = max(
                    waypoint_position_error
                    / max(effective_position_settle_tolerance, 1e-12),
                    waypoint_orientation_error
                    / float(
                        whole_body_eef_guard[
                            "waypoint_settle_orientation_tolerance_rad"
                        ]
                    ),
                )
                previous_settle_index = whole_body_eef_guard.get(
                    "settle_waypoint_index"
                )
                previous_normalized_settle_error = whole_body_eef_guard.get(
                    "settle_previous_normalized_error"
                )
                settle_progress_ok = True
                if whole_body_joint_waypoint_reached and not eef_pose_settled:
                    if (
                        previous_settle_index
                        == whole_body_commanded_waypoint_index
                        and isinstance(
                            previous_normalized_settle_error,
                            (int, float),
                        )
                    ):
                        settle_progress_ok = bool(
                            normalized_settle_error
                            <= float(previous_normalized_settle_error) * 0.95
                            + 1e-9
                        )
                        whole_body_eef_settle_no_progress = (
                            not settle_progress_ok
                        )
                    if settle_progress_ok:
                        whole_body_eef_guard["settle_waypoint_index"] = (
                            whole_body_commanded_waypoint_index
                        )
                        whole_body_eef_guard[
                            "settle_previous_normalized_error"
                        ] = normalized_settle_error
                whole_body_eef_guard["last_waypoint"] = {
                    "index": whole_body_commanded_waypoint_index,
                    "joint_waypoint_reached": whole_body_joint_waypoint_reached,
                    "waypoint_settled": waypoint_settled,
                    "eef_pose_settled": eef_pose_settled,
                    "dynamics_gate_used": False,
                    "waypoint_attempts": waypoint_attempts,
                    "normalized_settle_error": normalized_settle_error,
                    "settle_progress_ok": settle_progress_ok,
                    "position_error_m": waypoint_position_error,
                    "orientation_error_rad": waypoint_orientation_error,
                    "configured_position_settle_tolerance_m": (
                        configured_position_settle_tolerance
                    ),
                    "effective_position_settle_tolerance_m": (
                        effective_position_settle_tolerance
                    ),
                    "controller_response_margin_m": (
                        WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M
                    ),
                    "numerical_margin_m": (
                        WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                    ),
                    "prospective_guard_margin_m": (
                        WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
                    ),
                    "prospective_next_waypoint_step_m": (
                        prospective_next_waypoint_step
                    ),
                    "prospective_position_bound_m": (
                        waypoint_position_error
                        + prospective_next_waypoint_step
                        + WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
                    ),
                    "start_excursion_m": start_excursion,
                    "along_m": along,
                    "lateral_m": lateral,
                    "step_distance_m": live_step_distance,
                    "reverse_step_m": live_reverse_step,
                    "cumulative_path_m": cumulative_live_path,
                    "orientation_step_rad": live_orientation_step,
                    "target_orientation_error_rad": (
                        live_target_orientation_error
                    ),
                    "orientation_reverse_progress_rad": (
                        live_orientation_reverse_progress
                    ),
                    "cumulative_orientation_path_rad": (
                        cumulative_live_orientation_path
                    ),
                    "violations": violations,
                    "fatal_violations": fatal_violations,
                    "deferred_violations": deferred_violations,
                    "initial_axis_reverse_deferred": (
                        waypoint_axis_reverse_deferred
                    ),
                    "axis_reverse_deferred": (
                        waypoint_axis_reverse_deferred
                    ),
                    "axis_reverse_defer_key": (
                        axis_reverse_defer_key
                    ),
                    "axis_reverse_settle_repeat_command": (
                        axis_reverse_settle_repeat_command
                    ),
                    "initial_axis_reverse_transient_max_m": (
                        WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M
                    ),
                }
                whole_body_eef_waypoint_trace = {
                    "commanded_index": whole_body_commanded_waypoint_index,
                    "post_step_index": index,
                    "q_sha256": (
                        whole_body_waypoint_command["q_sha256"]
                        if whole_body_waypoint_command is not None
                        else None
                    ),
                    "trajectory_sha256": (
                        whole_body_waypoint_command["trajectory_sha256"]
                        if whole_body_waypoint_command is not None
                        else None
                    ),
                    "settle_repeat": (
                        bool(whole_body_waypoint_command["settle_repeat"])
                        if whole_body_waypoint_command is not None
                        else False
                    ),
                    "joint_waypoint_reached": whole_body_joint_waypoint_reached,
                    "waypoint_settled": waypoint_settled,
                    "eef_pose_settled": eef_pose_settled,
                    "dynamics_gate_used": False,
                    "waypoint_attempts": waypoint_attempts,
                    "normalized_settle_error": normalized_settle_error,
                    "settle_progress_ok": settle_progress_ok,
                    "position_error_m": waypoint_position_error,
                    "orientation_error_rad": waypoint_orientation_error,
                    "configured_position_settle_tolerance_m": (
                        configured_position_settle_tolerance
                    ),
                    "effective_position_settle_tolerance_m": (
                        effective_position_settle_tolerance
                    ),
                    "controller_response_margin_m": (
                        WHOLE_BODY_EEF_CONTROLLER_RESPONSE_MARGIN_M
                    ),
                    "numerical_margin_m": (
                        WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                    ),
                    "prospective_guard_margin_m": (
                        WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
                    ),
                    "prospective_next_waypoint_step_m": (
                        prospective_next_waypoint_step
                    ),
                    "prospective_position_bound_m": (
                        waypoint_position_error
                        + prospective_next_waypoint_step
                        + WHOLE_BODY_EEF_PROSPECTIVE_GUARD_MARGIN_M
                    ),
                    "live_position": live_position.tolist(),
                    "nominal_position": nominal_position.tolist(),
                    "along_m": along,
                    "lateral_m": lateral,
                    "step_distance_m": live_step_distance,
                    "reverse_step_m": live_reverse_step,
                    "cumulative_path_m": cumulative_live_path,
                    "violations": dict(violations),
                    "fatal_violations": dict(fatal_violations),
                    "deferred_violations": dict(deferred_violations),
                    "initial_axis_reverse_deferred": (
                        waypoint_axis_reverse_deferred
                    ),
                    "axis_reverse_deferred": (
                        waypoint_axis_reverse_deferred
                    ),
                    "axis_reverse_defer_key": (
                        axis_reverse_defer_key
                    ),
                    "axis_reverse_settle_repeat_command": (
                        axis_reverse_settle_repeat_command
                    ),
                    "initial_axis_reverse_transient_max_m": (
                        WHOLE_BODY_EEF_FIRST_SAMPLE_REVERSE_TRANSIENT_MAX_M
                    ),
                }
                if any(fatal_violations.values()):
                    trace.append(
                        {
                            "step": executed,
                            "position_error_m": final_pos_err,
                            "orientation_error_rad": final_ori_err,
                            "joint_tracking": waypoint_tracking,
                            "whole_body_eef_path_guard": eef_guard_snapshot(),
                            "whole_body_waypoint_command": (
                                whole_body_waypoint_command
                            ),
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="eef_path_divergence",
                        recoverable=False,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "whole_body_eef_path_guard": eef_guard_snapshot(),
                        },
                    )
                intermediate_command = bool(
                    whole_body_commanded_waypoint_index < planned_steps - 1
                )
                if (
                    not waypoint_axis_reverse_deferred
                    and (intermediate_command or waypoint_settled)
                ):
                    checkpoint_position_improvement = (
                        float(whole_body_eef_guard["direct_distance_m"])
                        - float(final_pos_err)
                        if final_pos_err is not None
                        and math.isfinite(float(final_pos_err))
                        else None
                    )
                    checkpoint_eligible = bool(
                        allow_replan_checkpoint
                        and intermediate_command
                        and eef_pose_settled
                        and checkpoint_position_improvement is not None
                        and checkpoint_position_improvement
                        + WHOLE_BODY_EEF_NUMERICAL_MARGIN_M
                        >= checkpoint_trigger_position_improvement_m
                        and contact_target_xyz is None
                        and not stop_on_expected_contact
                        and not stop_on_attachment
                        and not require_attachment
                        and expected_attachment is None
                        and not any(
                            value is not None
                            for value in (
                                expected_attachments_by_hand or {}
                            ).values()
                        )
                    )
                    index += 1
                    waypoint_advanced = True
                    waypoint_attempts = 0
                    whole_body_eef_guard["settle_waypoint_index"] = None
                    whole_body_eef_guard[
                        "settle_previous_normalized_error"
                    ] = None
                    if checkpoint_eligible:
                        whole_body_replan_checkpoint = {
                            "commanded_index": (
                                whole_body_commanded_waypoint_index
                            ),
                            "post_step_index": index,
                            "q_sha256": waypoint_q_sha256,
                            "trajectory_sha256": (
                                whole_body_waypoint_command[
                                    "trajectory_sha256"
                                ]
                            ),
                            "call_start_position_error_m": float(
                                whole_body_eef_guard["direct_distance_m"]
                            ),
                            "final_position_error_m": float(final_pos_err),
                            "position_improvement_m": float(
                                checkpoint_position_improvement
                            ),
                            "minimum_position_improvement_m": (
                                WHOLE_BODY_REPLAN_POSITION_IMPROVEMENT_M
                            ),
                            "trigger_position_improvement_m": (
                                checkpoint_trigger_position_improvement_m
                            ),
                            "eef_pose_settled": eef_pose_settled,
                            "dynamics_gate_used": False,
                            "joint_waypoint_reached": (
                                whole_body_joint_waypoint_reached
                            ),
                        }
                assert whole_body_eef_waypoint_trace is not None
                whole_body_eef_waypoint_trace["post_step_index"] = index
            if contact_target_xyz is not None and (
                stop_on_expected_contact or stop_on_attachment
            ):
                if contact is None:
                    contact = self._contact_report(
                        hand=hand,
                        target_xyz=contact_target_xyz,
                        allowed_contact_distance_m=max(
                            float(allowed_contact_distance_m),
                            float(position_tolerance_m) * 2.0,
                        ),
                    )
                if not bool(contact.get("available", False)):
                    trace.append(
                        {
                            "step": executed,
                            "contact_report": contact,
                            "whole_body_waypoint_command": (
                                whole_body_waypoint_command
                            ),
                        }
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="contact_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"contact_report": contact},
                    )
                expected_contact = bool(contact.get("expected_contact", False))
                if stop_on_expected_contact and expected_contact:
                    trace.append(
                        {
                            "step": executed,
                            "position_error_m": final_pos_err,
                            "orientation_error_rad": final_ori_err,
                            "joint_tracking": waypoint_tracking,
                            "contact_report": contact,
                            "whole_body_waypoint_command": (
                                whole_body_waypoint_command
                            ),
                            "expected_contact_stop": True,
                        }
                    )
                    return self._execution_result(
                        primitive_success=True,
                        stop_reason="reached",
                        recoverable=True,
                        suggested_next_tool=None,
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=0,
                        started=started,
                        extra_metrics={
                            "contact_report": contact,
                            "expected_contact_seen": True,
                            "whole_body_eef_path_guard": eef_guard_snapshot(),
                        },
                    )
                if (
                    stop_on_attachment
                    and int(gripper_contact_settle_steps) > 0
                    and not gripper_contact_settle_started
                    and bool(contact.get("target_two_finger_contact", False))
                ):
                    # Hold the exact two-finger+raycast candidate for a bounded
                    # confirmation window.
                    gripper_contact_settle_started = True
                    gripper_contact_settle_remaining = max(
                        GRIPPER_CONTACT_SETTLE_STEPS,
                        int(gripper_contact_settle_steps),
                    )
                    gripper_contact_hold_action = action.copy()
            trace.append(
                {
                    "step": executed,
                    "position_error_m": final_pos_err,
                    "orientation_error_rad": final_ori_err,
                    "joint_tracking": waypoint_tracking,
                    "waypoint_attempts": waypoint_attempts,
                    "contact_stop_active": contact_stop_active,
                    "attachment_confirmed": attachment_confirmed,
                    "attachment_confirmation_steps": attachment_confirmation_steps,
                    "attachment_identity": attachment_identity,
                    "contact_report": contact,
                    "gripper_contact_settle_started": (gripper_contact_settle_started),
                    "gripper_contact_settle_remaining": (
                        gripper_contact_settle_remaining
                    ),
                    "gripper_command": float(
                        action[ENV_ACTION_SEGMENTS[f"{hand}_gripper"]][0]
                    ),
                    "single_arm_isolation": (
                        self._active_isolation_report if isolation_required else None
                    ),
                    "whole_body_eef_waypoint_tracking": (
                        whole_body_eef_waypoint_trace
                    ),
                    "hold_step": index >= planned_steps,
                }
            )
            if whole_body_replan_checkpoint is not None:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="whole_body_replan_checkpoint",
                    recoverable=True,
                    suggested_next_tool=None,
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "final_joint_tracking": final_waypoint_tracking,
                        "trajectory_complete": False,
                        "whole_body_replan_checkpoint": (
                            whole_body_replan_checkpoint
                        ),
                        "whole_body_eef_path_guard": eef_guard_snapshot(),
                        "post_stop_action_policy": (
                            "fresh_plan_before_next_env_action"
                        ),
                        "post_stop_env_actions": 0,
                    },
                )
            if stop_on_attachment:
                if attachment_confirmation_steps >= int(hold_steps_required):
                    return self._execution_result(
                        primitive_success=True,
                        stop_reason="reached",
                        recoverable=True,
                        suggested_next_tool=None,
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=attachment_confirmation_steps,
                        started=started,
                        extra_metrics={
                            "attachment_confirmed_during_execution": True,
                            "attachment_confirmation_steps": (
                                attachment_confirmation_steps
                            ),
                            "attachment_identity": attachment_identity,
                            "gripper_contact_settle_started": (
                                gripper_contact_settle_started
                            ),
                            "gripper_contact_settle_steps_executed": (
                                gripper_contact_settle_executed
                            ),
                            "whole_body_eef_path_guard": eef_guard_snapshot(),
                        },
                    )
                # A gripper close succeeds only after the same target root has
                # remained attached for the complete confirmation window.
                continue
            if waypoint_stalled:
                settle_failure = bool(
                    whole_body_eef_guard is not None
                    and whole_body_joint_waypoint_reached
                    and not waypoint_advanced
                )
                return self._execution_result(
                    primitive_success=False,
                    stop_reason=(
                        "eef_settle_no_progress"
                        if whole_body_eef_settle_no_progress
                        else (
                            "eef_settle_exhausted"
                            if settle_failure
                            else "stalled_tracking"
                        )
                    ),
                    recoverable=not settle_failure,
                    suggested_next_tool=(
                        "observe" if settle_failure else "move_to"
                    ),
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "final_joint_tracking": final_waypoint_tracking,
                        "trajectory_complete": False,
                        "waypoint_attempts": waypoint_attempts,
                    },
                )
            position_ok = contact_stop_active or (
                final_pos_err is None or final_pos_err <= position_tolerance_m
            )
            orientation_ok = (
                final_ori_err is None or final_ori_err <= orientation_tolerance_rad
            )
            trajectory_complete = index >= planned_steps
            if trajectory_complete and position_ok and orientation_ok:
                held_steps += 1
                if require_attachment:
                    attachment_endpoint_held_steps += 1
                if held_steps >= int(hold_steps_required):
                    return self._execution_result(
                        primitive_success=True,
                        stop_reason="reached",
                        recoverable=True,
                        suggested_next_tool=None,
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "final_joint_tracking": final_waypoint_tracking,
                            "trajectory_complete": True,
                            "attachment_confirmation_steps": (
                                attachment_confirmation_steps
                            ),
                            "attachment_endpoint_held_steps": (
                                attachment_endpoint_held_steps
                            ),
                            "attachment_identity": attachment_identity,
                            "gripper_contact_settle_started": (
                                gripper_contact_settle_started
                            ),
                            "gripper_contact_settle_steps_executed": (
                                gripper_contact_settle_executed
                            ),
                            "whole_body_eef_path_guard": eef_guard_snapshot(),
                        },
                    )
                continue
            if not trajectory_complete and position_ok and orientation_ok:
                # The final EEF goal may already be satisfied while a validated
                # trajectory still contains waypoints. Do not misclassify that
                # as stalled, and do not finish before consuming the full path.
                held_steps = 0
                stalled_steps = 0
                continue
            held_steps = 0
            attachment_endpoint_held_steps = 0
            if waypoint_tracking is not None:
                error = float(waypoint_tracking.get("max_articulation_error_rad", 0.0))
                error += float(waypoint_tracking.get("max_base_xy_error_m", 0.0))
                error += float(waypoint_tracking.get("base_yaw_error_rad", 0.0))
            else:
                error = final_pos_err if final_pos_err is not None else 0.0
                if final_ori_err is not None:
                    error += final_ori_err
            progress_feedback_available = (
                waypoint_tracking is not None
                or base_goal_xyyaw is not None
                or require_pose
            )
            if not progress_feedback_available:
                stalled_steps = 0
            elif waypoint_advanced:
                best_error = float("inf")
                stalled_steps = 0
            elif error + 1e-9 < best_error:
                best_error = error
                stalled_steps = 0
            else:
                # Count every consecutive simulation step that fails to set a
                # new best error, including regressions, as stalled tracking.
                stalled_steps += 1
            if stalled_steps >= self.max_stall_steps:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="stalled_tracking",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "final_joint_tracking": final_waypoint_tracking,
                        "trajectory_complete": index >= planned_steps,
                    },
                )
            if index >= planned_steps and stalled_steps >= max(
                1, self.max_stall_steps // 2
            ):
                break
        return self._execution_result(
            primitive_success=False,
            stop_reason=(
                "grasp_not_confirmed"
                if stop_on_attachment and not attachment_seen
                else (
                    "attachment_confirmation_incomplete"
                    if stop_on_attachment
                    else "target_tolerance_not_met"
                )
            ),
            recoverable=True,
            suggested_next_tool="move_to",
            executed=executed,
            trace=trace,
            final_pos_err=final_pos_err,
            final_ori_err=final_ori_err,
            held_steps=held_steps,
            started=started,
            extra_metrics={
                "final_joint_tracking": final_waypoint_tracking,
                "trajectory_complete": index >= planned_steps,
                "attachment_confirmation_steps": attachment_confirmation_steps,
                "attachment_endpoint_held_steps": attachment_endpoint_held_steps,
                "attachment_identity": attachment_identity,
                "gripper_contact_settle_started": gripper_contact_settle_started,
                "gripper_contact_settle_steps_executed": (
                    gripper_contact_settle_executed
                ),
                "whole_body_eef_path_guard": eef_guard_snapshot(),
            },
        )

    def _execution_result(
        self,
        *,
        primitive_success: bool,
        stop_reason: str,
        recoverable: bool,
        suggested_next_tool: str | None,
        executed: int,
        trace: list[dict[str, Any]],
        final_pos_err: float | None,
        final_ori_err: float | None,
        held_steps: int = 0,
        started: float | None = None,
        extra_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = {
            "executed_waypoints": int(executed),
            "env_actions_sent": int(executed),
            "final_position_error_m": final_pos_err,
            "final_orientation_error_rad": final_ori_err,
            "held_steps": int(held_steps),
            "elapsed_s": round(time.monotonic() - started, 3)
            if started is not None
            else None,
        }
        metrics.update(extra_metrics or {})
        metrics.setdefault(
            "partial_motion",
            bool(not primitive_success and int(executed) > 0),
        )
        if not primitive_success:
            metrics.setdefault("post_stop_action_policy", "no_additional_env_action")
            metrics.setdefault("post_stop_env_actions", 0)
            if int(executed) > 0:
                metrics.setdefault(
                    "passive_hold_policy",
                    "position_controller_last_target_remains_latched",
                )
        if isinstance(self._active_isolation_report, dict):
            metrics.setdefault(
                "single_arm_isolation",
                self._active_isolation_report,
            )
        trace_artifact = self._persist_trace_artifact(trace, stop_reason=stop_reason)
        return primitive_result(
            primitive_success=primitive_success,
            task_success=self._task_success(),
            stop_reason=stop_reason,
            recoverable=recoverable,
            suggested_next_tool=suggested_next_tool,
            metrics=metrics,
            diagnostics={
                "trace": trace[-50:],
                "trace_artifact": trace_artifact,
                "trace_steps_persisted": len(trace),
            },
        )

    def _persist_trace_artifact(
        self,
        trace: list[dict[str, Any]],
        *,
        stop_reason: str,
    ) -> str | None:
        """Persist the complete controller trace without privileged object state."""
        try:
            self._trace_counter += 1
            trace_dir = self.output_dir / "planner_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            safe_reason = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in str(stop_reason)
            )[:64]
            path = trace_dir / (
                f"{time.time_ns()}_{self._trace_counter:06d}_{safe_reason or 'result'}.json"
            )
            payload = {
                "schema_version": 1,
                "stop_reason": str(stop_reason),
                "trace_steps": len(trace),
                "trace": _jsonable(trace),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return str(path)
        except Exception:
            return None

    def _persist_tool_artifact(
        self,
        *,
        tool: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        """Persist every planner invocation, including validation failures."""
        try:
            self._trace_counter += 1
            artifact_dir = self.output_dir / "planner_tool_artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            stop_reason = str(result.get("stop_reason", "unknown"))
            safe_reason = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in stop_reason
            )[:64]
            path = artifact_dir / (
                f"{time.time_ns()}_{self._trace_counter:06d}_{tool}_"
                f"{safe_reason or 'result'}.json"
            )
            payload = {
                "schema_version": 1,
                "tool": str(tool),
                "inputs": {
                    "args": _artifact_jsonable(args),
                    "kwargs": _artifact_jsonable(kwargs),
                },
                "result": _artifact_jsonable(result),
            }
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            return str(path)
        except Exception:
            return None

    def _contact_report(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray | None,
        allowed_contact_distance_m: float,
    ) -> dict[str, Any]:
        report = _call_optional_kw(
            self.backend,
            "contact_report",
            hand=hand,
            target_xyz=target_xyz,
            allowed_contact_distance_m=allowed_contact_distance_m,
        )
        if isinstance(report, dict):
            return report
        return {
            "available": False,
            "reason": "contact_report_unavailable",
            "unexpected_contact": False,
            "expected_contact": False,
        }

    def _step_env_action(self, action: np.ndarray) -> dict[str, bool | int]:
        step = getattr(self.env, "planner_step", None)
        if not callable(step):
            step = self.env.chunk_step
        ret = step(np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM))
        if not isinstance(ret, tuple) or len(ret) < 5:
            raise RuntimeError(
                "planner action did not return an execution receipt from the env"
            )
        self.last_info = ret[4]
        info = ret[4]
        rpent = info.get("_rpent") if isinstance(info, dict) else None
        if not isinstance(rpent, dict) or "executed_steps" not in rpent:
            raise RuntimeError("planner action execution receipt is missing")
        executed_steps = rpent["executed_steps"]
        if (
            isinstance(executed_steps, bool)
            or not isinstance(executed_steps, (int, np.integer))
            or int(executed_steps) != 1
        ):
            raise RuntimeError(
                "planner action was not executed exactly once by the env"
            )
        return {
            "executed_steps": int(executed_steps),
            "raw_success": official_task_success(info),
            "terminated": bool(np.asarray(ret[2]).any()),
            "truncated": bool(np.asarray(ret[3]).any()),
        }

    def _world_target(self, target_xyz: Any, *, frame: str) -> np.ndarray:
        if str(frame) != "world":
            raise ValueError("planner currently requires frame='world'")
        return _as_xyz(target_xyz)

    def _check_arm_reachability(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = _ATTACHMENT_UNSET,
    ) -> tuple[bool, str, dict[str, Any]]:
        fn = self.backend.check_arm_reachability
        kwargs: dict[str, Any] = {
            "hand": hand,
            "target_xyz": target_xyz,
            "target_quat_xyzw": target_quat_xyzw,
        }
        try:
            parameters = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "timeout_s" in parameters:
            kwargs["timeout_s"] = min(5.0, float(timeout_s))
        if "attached_obj" in parameters:
            kwargs["attached_obj"] = attached_obj
        return fn(**kwargs)

    def _task_success(self) -> bool:
        if bool(getattr(self.env, "_official_success_latched", False)):
            return True
        return official_task_success(
            self.last_info or getattr(self.env, "_last_info", None)
        )

    @staticmethod
    def _validated_timeout(timeout_s: Any) -> float:
        value = float(timeout_s)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("timeout_s must be a finite positive number")
        return value

    @staticmethod
    def _remaining_s(deadline: float) -> float:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("planner tool deadline exceeded")
        return remaining

    def _exception_result(
        self, exc: Exception, *, suggested_next_tool: str | None
    ) -> dict[str, Any]:
        return primitive_result(
            primitive_success=False,
            task_success=self._task_success(),
            stop_reason="timeout" if isinstance(exc, TimeoutError) else "error",
            recoverable=True,
            suggested_next_tool=suggested_next_tool,
            diagnostics={"error": f"{type(exc).__name__}: {exc}"},
        )


def _call_optional(obj: Any, name: str) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _call_optional_arg(obj: Any, name: str, *args: Any) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:
        return None


def _call_optional_kw(obj: Any, name: str, **kwargs: Any) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(**kwargs)
    except Exception:
        return None


def _interpolate_joint_trajectory(
    trajectory: Any,
    *,
    max_inter_dist: float,
) -> np.ndarray:
    q = np.asarray(_jsonable(trajectory), dtype=np.float32)
    if q.ndim != 2 or q.shape[0] < 1:
        raise ValueError(f"joint trajectory must be [T,D], got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError("joint trajectory contains NaN or infinity")
    if max_inter_dist <= 0:
        raise ValueError("max_inter_dist must be positive")
    if q.shape[0] == 1:
        return q.copy()

    interpolated = []
    for start, end in zip(q[:-1], q[1:], strict=True):
        intervals = max(
            1,
            int(math.ceil(float(np.max(np.abs(end - start))) / max_inter_dist)),
        )
        for index in range(intervals):
            alpha = index / intervals
            interpolated.append(start + (end - start) * alpha)
    interpolated.append(q[-1])
    result = np.stack(interpolated, axis=0).astype(np.float32, copy=False)
    if result.shape[0] > 1:
        max_delta = float(np.max(np.abs(np.diff(result, axis=0))))
        if max_delta > max_inter_dist + 1e-6:
            raise RuntimeError(
                f"interpolated joint delta {max_delta} exceeds {max_inter_dist}"
            )
    return result


def _minimum_jerk_base_execution_trajectory(
    trajectory: Any,
    *,
    base_indices: tuple[int, ...] | list[int],
    max_xy_step_m: float = BASE_EXECUTION_XY_STEP_M,
    max_yaw_step_rad: float = BASE_EXECUTION_YAW_STEP_RAD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample a BASE-only q path with a bounded quintic motion profile.

    Each input edge follows a straight world-XY segment and the shortest yaw
    arc. The quintic minimum-jerk profile has zero analytic velocity and
    acceleration at both endpoints, avoiding the full one-step velocity
    reversal produced by uniform linear interpolation. Non-BASE joints must
    already be locked and remain bitwise stable in the returned path.
    """

    q = np.asarray(_jsonable(trajectory), dtype=np.float64)
    indices = tuple(int(index) for index in base_indices)
    xy_limit = float(max_xy_step_m)
    yaw_limit = float(max_yaw_step_rad)
    if q.ndim != 2 or q.shape[0] < 1 or not np.isfinite(q).all():
        raise ValueError(f"BASE trajectory must be finite [T,D], got {q.shape}")
    if (
        len(indices) != 6
        or len(set(indices)) != 6
        or min(indices) < 0
        or max(indices) >= q.shape[1]
    ):
        raise ValueError("BASE resampling requires six unique in-range base indices")
    if (
        not math.isfinite(xy_limit)
        or not math.isfinite(yaw_limit)
        or xy_limit <= 0.0
        or yaw_limit <= 0.0
    ):
        raise ValueError("BASE resampling step limits must be finite and positive")
    if len(q) == 1:
        return q.astype(np.float32), {
            "method": "quintic_minimum_jerk",
            "profile": "10u^3-15u^4+6u^5",
            "path_geometry": "straight_xy_shortest_yaw_arc",
            "original_waypoints": 1,
            "resampled_waypoints": 1,
            "execution_waypoints": 0,
            "segment_intervals": [],
            "xy_step_limit_m": xy_limit,
            "yaw_step_limit_rad": yaw_limit,
            "measured_max_xy_step_m": 0.0,
            "measured_max_yaw_step_rad": 0.0,
            "analytic_endpoint_velocity_zero": True,
            "analytic_endpoint_acceleration_zero": True,
        }

    x_index, y_index, yaw_index = indices[0], indices[1], indices[5]
    active = {x_index, y_index, yaw_index}
    locked = [index for index in range(q.shape[1]) if index not in active]
    rows = [q[0].copy()]
    segment_intervals: list[int] = []
    minimum_intervals = 4

    for input_start, input_end in zip(q[:-1], q[1:], strict=True):
        start = rows[-1]
        if locked and not np.allclose(
            input_end[locked],
            input_start[locked],
            atol=1e-7,
            rtol=0.0,
        ):
            raise ValueError("BASE resampling received a changed locked joint")
        end = input_end.copy()
        end[locked] = start[locked]
        end[yaw_index] = start[yaw_index] + _wrap_angle(
            float(input_end[yaw_index] - input_start[yaw_index])
        )
        xy_distance = float(
            np.linalg.norm(end[[x_index, y_index]] - start[[x_index, y_index]])
        )
        yaw_distance = abs(_wrap_angle(float(end[yaw_index] - start[yaw_index])))
        intervals = max(
            minimum_intervals,
            int(math.ceil(xy_distance / xy_limit)),
            int(math.ceil(yaw_distance / yaw_limit)),
        )

        for _attempt in range(10000):
            u = np.arange(1, intervals + 1, dtype=np.float64) / intervals
            progress = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            segment = start + progress[:, None] * (end - start)
            segment[-1] = end
            with_start = np.vstack([start, segment])
            xy_steps = np.linalg.norm(
                np.diff(with_start[:, [x_index, y_index]], axis=0),
                axis=1,
            )
            yaw_steps = np.asarray(
                [
                    abs(_wrap_angle(float(delta)))
                    for delta in np.diff(with_start[:, yaw_index])
                ],
                dtype=np.float64,
            )
            if (
                float(np.max(xy_steps, initial=0.0)) <= xy_limit + 1e-9
                and float(np.max(yaw_steps, initial=0.0)) <= yaw_limit + 1e-9
            ):
                break
            intervals += 1
        else:
            raise RuntimeError("minimum-jerk BASE resampling did not converge")

        rows.extend(segment)
        segment_intervals.append(intervals)

    result = np.stack(rows, axis=0)
    result[:, locked] = q[0, locked]
    execution_result = result.astype(np.float32)
    xy_steps = np.linalg.norm(
        np.diff(
            execution_result[:, [x_index, y_index]].astype(np.float64),
            axis=0,
        ),
        axis=1,
    )
    yaw_steps = np.asarray(
        [
            abs(_wrap_angle(float(delta)))
            for delta in np.diff(execution_result[:, yaw_index].astype(np.float64))
        ],
        dtype=np.float64,
    )
    measured_xy = float(np.max(xy_steps, initial=0.0))
    measured_yaw = float(np.max(yaw_steps, initial=0.0))
    if measured_xy > xy_limit + 1e-9 or measured_yaw > yaw_limit + 1e-9:
        raise RuntimeError("minimum-jerk BASE resampling exceeded a step limit")
    if locked and not bool(
        np.all(execution_result[:, locked] == execution_result[0, locked])
    ):
        raise RuntimeError("minimum-jerk BASE resampling changed a locked joint")

    return execution_result, {
        "method": "quintic_minimum_jerk",
        "profile": "10u^3-15u^4+6u^5",
        "path_geometry": "straight_xy_shortest_yaw_arc",
        "original_waypoints": int(len(q)),
        "resampled_waypoints": int(len(execution_result)),
        "execution_waypoints": int(len(execution_result) - 1),
        "segment_intervals": segment_intervals,
        "minimum_intervals_per_segment": minimum_intervals,
        "xy_step_limit_m": xy_limit,
        "yaw_step_limit_rad": yaw_limit,
        "measured_max_xy_step_m": measured_xy,
        "measured_max_yaw_step_rad": measured_yaw,
        "analytic_endpoint_velocity_zero": True,
        "analytic_endpoint_acceleration_zero": True,
    }


def _canonicalize_whole_body_base_yaw_trajectory(
    trajectory: Any,
    *,
    joint_names: tuple[str, ...] | list[str],
    call_start_q: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep whole-body base yaw on the call-start shortest-arc branch."""

    q = np.asarray(_jsonable(trajectory), dtype=np.float32)
    names = tuple(str(name) for name in joint_names)
    start = np.asarray(_jsonable(call_start_q), dtype=np.float32).reshape(-1)
    if (
        q.ndim != 2
        or q.shape[0] < 1
        or q.shape[1] != len(names)
        or start.shape != (len(names),)
        or len(names) != len(set(names))
        or "base_footprint_rz_joint" not in names
        or not np.isfinite(q).all()
        or not np.isfinite(start).all()
    ):
        raise ValueError(
            "whole-body yaw canonicalization requires a finite full-q trajectory"
        )

    yaw_index = names.index("base_footprint_rz_joint")
    result = q.copy()
    raw_yaw = result[:, yaw_index].astype(np.float64, copy=True)
    canonical_yaw = np.empty_like(raw_yaw)
    previous_yaw = float(start[yaw_index])
    branch_correction_count = 0
    for waypoint_index, waypoint_yaw in enumerate(raw_yaw):
        continuous_yaw = previous_yaw + _wrap_angle(
            float(waypoint_yaw - previous_yaw)
        )
        canonical_yaw[waypoint_index] = continuous_yaw
        if abs(continuous_yaw - float(waypoint_yaw)) > 1e-6:
            branch_correction_count += 1
        previous_yaw = continuous_yaw
    result[:, yaw_index] = canonical_yaw.astype(np.float32)

    raw_with_start = np.concatenate(
        [
            np.asarray([float(start[yaw_index])], dtype=np.float64),
            raw_yaw,
        ]
    )
    canonical_with_start = np.concatenate(
        [
            np.asarray([float(start[yaw_index])], dtype=np.float64),
            canonical_yaw,
        ]
    )
    canonical_yaw_steps = np.abs(np.diff(canonical_with_start))
    if bool(np.any(canonical_yaw_steps > math.pi + 1e-6)):
        raise RuntimeError(
            "whole-body path yaw continuity exceeded the shortest-arc bound"
        )
    wrapped_equivalence_error = np.asarray(
        [
            abs(_wrap_angle(float(raw - canonical)))
            for raw, canonical in zip(raw_yaw, canonical_yaw, strict=True)
        ],
        dtype=np.float64,
    )
    if float(np.max(wrapped_equivalence_error, initial=0.0)) > 1e-5:
        raise RuntimeError("whole-body yaw canonicalization changed a target pose")

    return result, {
        "policy": "call_start_then_previous_plus_wrapped_delta",
        "branch_correction_count": branch_correction_count,
        "raw_max_step_rad": float(
            np.max(np.abs(np.diff(raw_with_start)), initial=0.0)
        ),
        "canonical_max_step_rad": float(
            np.max(canonical_yaw_steps, initial=0.0)
        ),
        "wrapped_equivalence_max_error_rad": float(
            np.max(wrapped_equivalence_error, initial=0.0)
        ),
    }


def _grouped_stationary_dynamics_status(
    report: Any,
    *,
    stationary_thresholds: dict[str, float],
    policy: str,
) -> tuple[dict[str, Any], str | None]:
    """Validate grouped R1Pro dynamics and derive an auditable stationary gate."""

    if not isinstance(report, dict) or report.get("available") is not True:
        return (
            report
            if isinstance(report, dict)
            else {
                "available": False,
                "ok": None,
                "reason": "dynamics report unavailable",
            },
            "dynamics_feedback_unavailable",
        )
    if report.get("ok") is not True:
        return dict(report), "dynamics_limit_violation"
    groups = report.get("velocity_groups")
    if not isinstance(groups, dict) or set(groups) != set(
        DYNAMICS_VELOCITY_GROUP_JOINTS
    ):
        return dict(report), "dynamics_feedback_unavailable"

    stationary_groups: dict[str, Any] = {}
    grouped_dof_indices: list[int] = []
    for group_name, expected_names in DYNAMICS_VELOCITY_GROUP_JOINTS.items():
        group = groups.get(group_name)
        if not isinstance(group, dict) or group.get("available") is not True:
            return dict(report), "dynamics_feedback_unavailable"
        actual_raw = group.get("max_actual_velocity")
        limit_raw = group.get("max_velocity_limit")
        ratio_raw = group.get("max_velocity_ratio")
        peak_joint = group.get("max_actual_velocity_joint")
        joint_names = group.get("joint_names")
        dof_indices = group.get("dof_indices")
        within_control_limit = group.get("within_control_limit")
        if (
            not isinstance(actual_raw, (int, float))
            or isinstance(actual_raw, bool)
            or not math.isfinite(float(actual_raw))
            or float(actual_raw) < 0.0
            or not isinstance(limit_raw, (int, float))
            or isinstance(limit_raw, bool)
            or not math.isfinite(float(limit_raw))
            or float(limit_raw) <= 0.0
            or not isinstance(ratio_raw, (int, float))
            or isinstance(ratio_raw, bool)
            or not math.isfinite(float(ratio_raw))
            or float(ratio_raw) < 0.0
            or not isinstance(peak_joint, str)
            or peak_joint not in expected_names
            or not isinstance(joint_names, list)
            or tuple(str(name) for name in joint_names) != expected_names
            or not isinstance(dof_indices, list)
            or len(dof_indices) != len(expected_names)
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                for index in dof_indices
            )
            or len(set(dof_indices)) != len(dof_indices)
            or group.get("dof_count") != len(expected_names)
            or group.get("unit") != DYNAMICS_VELOCITY_GROUP_UNITS[group_name]
            or not isinstance(within_control_limit, bool)
        ):
            return dict(report), "dynamics_feedback_unavailable"
        if (
            within_control_limit is not True
            or float(ratio_raw) > 1.0 + 1e-3
        ):
            return dict(report), "dynamics_limit_violation"
        grouped_dof_indices.extend(dof_indices)
        threshold = stationary_thresholds.get(group_name)
        stationary_groups[group_name] = {
            "used_for_stationary": threshold is not None,
            "max_actual_velocity": float(actual_raw),
            "max_actual_velocity_joint": peak_joint,
            "stationary_threshold": threshold,
            "stationary": (
                None
                if threshold is None
                else float(actual_raw) <= float(threshold)
            ),
        }
    if (
        len(grouped_dof_indices) != len(WHOLE_BODY_ACTIVE_JOINT_NAMES)
        or len(set(grouped_dof_indices)) != len(grouped_dof_indices)
    ):
        return dict(report), "dynamics_feedback_unavailable"

    stationary = all(
        group["stationary"] is True
        for group in stationary_groups.values()
        if group["used_for_stationary"]
    )
    enriched = {
        **report,
        "stationary": stationary,
        "stationary_groups": stationary_groups,
        "stationary_policy": policy,
    }
    return enriched, None if stationary else "robot_not_stationary"


def _tracking_hard_deviation_report(
    report: Any,
) -> tuple[dict[str, Any], bool]:
    """Validate the wide runtime tracking envelope used after every command."""

    if not isinstance(report, dict) or report.get("available") is not True:
        return {
            "available": False,
            "hard_deviation": None,
            "reason": "joint tracking feedback unavailable",
        }, False
    try:
        base_xy = float(report.get("max_base_xy_error_m", float("inf")))
        base_yaw = float(report.get("base_yaw_error_rad", float("inf")))
        articulation = float(
            report.get("max_articulation_error_rad", float("inf"))
        )
    except (TypeError, ValueError):
        base_xy = base_yaw = articulation = float("inf")
    finite = all(math.isfinite(value) for value in (base_xy, base_yaw, articulation))
    checks = {
        "base_xy": finite
        and base_xy <= TRACKING_HARD_BASE_XY_ERROR_M,
        "base_yaw": finite
        and base_yaw <= TRACKING_HARD_BASE_YAW_ERROR_RAD,
        "articulation": finite
        and articulation <= TRACKING_HARD_ARTICULATION_ERROR_RAD,
    }
    return {
        "available": finite,
        "hard_deviation": not all(checks.values()),
        "checks": checks,
        "max_base_xy_error_m": base_xy,
        "base_yaw_error_rad": base_yaw,
        "max_articulation_error_rad": articulation,
        "limits": {
            "base_xy_m": TRACKING_HARD_BASE_XY_ERROR_M,
            "base_yaw_rad": TRACKING_HARD_BASE_YAW_ERROR_RAD,
            "articulation_rad": TRACKING_HARD_ARTICULATION_ERROR_RAD,
        },
    }, finite


def _whole_body_target_sha256(target_xyz: Any) -> str:
    target = np.asarray(_jsonable(target_xyz), dtype=np.float32).reshape(3)
    if not np.isfinite(target).all():
        raise ValueError("whole-body selected EEF target must be finite")
    return hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()


def _joint_name_layout_sha256(joint_names: Any) -> str:
    names = tuple(str(name) for name in joint_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("whole-body joint-name layout must be non-empty and unique")
    payload = json.dumps(
        list(names),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reorder_joint_trajectory(
    trajectory: Any,
    *,
    source_names: Any,
    target_names: Any,
) -> np.ndarray:
    """Return a finite q trajectory in an explicitly requested joint order."""

    source = tuple(str(name) for name in source_names)
    target = tuple(str(name) for name in target_names)
    values = np.asarray(_jsonable(trajectory), dtype=np.float32)
    if (
        not source
        or len(source) != len(set(source))
        or len(target) != len(set(target))
        or set(source) != set(target)
        or values.ndim != 2
        or values.shape[1] != len(source)
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("joint trajectory name-order mapping is invalid")
    source_index = {name: index for index, name in enumerate(source)}
    return np.ascontiguousarray(
        values[:, [source_index[name] for name in target]],
        dtype=np.float32,
    )


def _torso_z_path_direction_admitted(
    z_values: Any,
    *,
    start_z_m: float,
    target_z_m: float,
) -> bool:
    """Validate torso Z direction without inventing a direction for an identity plan."""

    values = np.asarray(_jsonable(z_values), dtype=np.float64).reshape(-1)
    start_z = float(start_z_m)
    target_z = float(target_z_m)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or not math.isfinite(start_z)
        or not math.isfinite(target_z)
    ):
        return False
    delta = target_z - start_z
    if abs(delta) <= 1e-6:
        return bool(
            np.max(np.abs(values - start_z), initial=0.0)
            <= TORSO_LINK_TERMINAL_TOLERANCE_M
        )
    direction = math.copysign(1.0, delta)
    return bool(np.all(direction * np.diff(values) >= -1e-6))


def _whole_body_indices_sha256(indices: Any) -> str:
    values = np.asarray(indices, dtype="<i8").reshape(-1)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _whole_body_ordered_row_indices(
    dense_trajectory: Any,
    execution_with_start: Any,
) -> np.ndarray:
    """Bind every execution anchor to an exact row in a denser q polyline."""

    dense = np.asarray(_jsonable(dense_trajectory), dtype=np.float32)
    anchors = np.asarray(_jsonable(execution_with_start), dtype=np.float32)
    if (
        dense.ndim != 2
        or anchors.ndim != 2
        or dense.shape[1:] != anchors.shape[1:]
        or len(dense) < 1
        or len(anchors) < 1
        or not np.isfinite(dense).all()
        or not np.isfinite(anchors).all()
    ):
        raise ValueError("whole-body row mapping requires finite compatible q paths")
    if not np.array_equal(dense[0], anchors[0]):
        raise RuntimeError("whole-body dense path does not start at its first anchor")
    if not np.array_equal(dense[-1], anchors[-1]):
        raise RuntimeError("whole-body dense path does not end at its last anchor")

    indices = [0]
    cursor = 1
    for anchor_index, anchor in enumerate(anchors[1:], start=1):
        if anchor_index == len(anchors) - 1:
            match = len(dense) - 1
            if match < cursor or not np.array_equal(dense[match], anchor):
                raise RuntimeError(
                    "whole-body terminal anchor is absent from the dense path"
                )
        else:
            matches = np.flatnonzero(
                np.all(dense[cursor:-1] == anchor.reshape(1, -1), axis=1)
            )
            if not len(matches):
                raise RuntimeError(
                    "whole-body execution anchor is absent from the dense path"
                )
            match = cursor + int(matches[0])
        indices.append(match)
        cursor = match + 1
    result = np.asarray(indices, dtype=np.int64)
    if (
        result.shape != (len(anchors),)
        or result[0] != 0
        or result[-1] != len(dense) - 1
        or (len(result) > 1 and bool(np.any(np.diff(result) <= 0)))
    ):
        raise RuntimeError("whole-body dense anchor mapping is not strictly ordered")
    return result


def _whole_body_execution_step_report(
    trajectory_with_start: Any,
    positions: Any,
    quaternions_xyzw: Any,
    *,
    joint_names: tuple[str, ...] | list[str],
    short_target: bool,
) -> dict[str, Any]:
    """Verify command-to-command q, EEF translation, and orientation limits."""

    q = np.asarray(_jsonable(trajectory_with_start), dtype=np.float32)
    xyz = np.asarray(_jsonable(positions), dtype=np.float64)
    quaternions = np.asarray(_jsonable(quaternions_xyzw), dtype=np.float64)
    names = tuple(str(name) for name in joint_names)
    if (
        q.ndim != 2
        or q.shape[0] < 1
        or q.shape[1] != len(names)
        or xyz.shape != (len(q), 3)
        or quaternions.shape != (len(q), 4)
        or len(names) != len(set(names))
        or not np.isfinite(q).all()
        or not np.isfinite(xyz).all()
        or not np.isfinite(quaternions).all()
    ):
        raise ValueError(
            "whole-body execution step admission requires finite aligned q/EEF paths"
        )
    missing = sorted(
        (set(WHOLE_BODY_ACTIVE_JOINT_NAMES) | set(WHOLE_BODY_LOCKED_JOINT_NAMES))
        - set(names)
    )
    if missing:
        raise ValueError(
            f"whole-body execution step admission is missing joints: {missing}"
        )

    name_to_index = {name: index for index, name in enumerate(names)}
    q64 = q.astype(np.float64)
    q_steps = np.diff(q64, axis=0)
    x_steps = np.abs(q_steps[:, name_to_index["base_footprint_x_joint"]])
    y_steps = np.abs(q_steps[:, name_to_index["base_footprint_y_joint"]])
    yaw_steps = np.asarray(
        [
            abs(_wrap_angle(float(value)))
            for value in q_steps[:, name_to_index["base_footprint_rz_joint"]]
        ],
        dtype=np.float64,
    )
    articulation_indices = [
        name_to_index[name]
        for name in WHOLE_BODY_ACTIVE_JOINT_NAMES
        if name
        not in {
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_rz_joint",
        }
    ]
    articulation_steps = (
        np.abs(q_steps[:, articulation_indices])
        if articulation_indices
        else np.zeros((max(len(q) - 1, 0), 0), dtype=np.float64)
    )
    locked_indices = [
        name_to_index[name] for name in WHOLE_BODY_LOCKED_JOINT_NAMES
    ]
    locked_delta = (
        np.abs(q64[:, locked_indices] - q64[0, locked_indices])
        if locked_indices
        else np.zeros((len(q), 0), dtype=np.float64)
    )
    eef_steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    quat_norms = np.linalg.norm(quaternions, axis=1)
    if bool(np.any(quat_norms <= 1e-9)):
        raise ValueError("whole-body execution path contains an invalid quaternion")
    normalized_quats = quaternions / quat_norms[:, None]
    adjacent_dots = (
        np.clip(
            np.abs(np.sum(normalized_quats[:-1] * normalized_quats[1:], axis=1)),
            0.0,
            1.0,
        )
        if len(normalized_quats) > 1
        else np.zeros((0,), dtype=np.float64)
    )
    orientation_steps = 2.0 * np.arccos(adjacent_dots)
    eef_step_limit = (
        WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
        if bool(short_target)
        else WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M
    )
    measured = {
        "max_base_x_step_m": float(np.max(x_steps, initial=0.0)),
        "max_base_y_step_m": float(np.max(y_steps, initial=0.0)),
        "max_base_yaw_step_rad": float(np.max(yaw_steps, initial=0.0)),
        "max_articulation_step_rad": float(
            np.max(articulation_steps, initial=0.0)
        ),
        "max_locked_joint_drift": float(np.max(locked_delta, initial=0.0)),
        "max_eef_cartesian_step_m": float(np.max(eef_steps, initial=0.0)),
        "max_eef_orientation_step_rad": float(
            np.max(orientation_steps, initial=0.0)
        ),
    }
    checks = {
        "base_x_step": (
            measured["max_base_x_step_m"]
            <= WHOLE_BODY_EXECUTION_BASE_XY_STEP_M + 1e-9
        ),
        "base_y_step": (
            measured["max_base_y_step_m"]
            <= WHOLE_BODY_EXECUTION_BASE_XY_STEP_M + 1e-9
        ),
        "base_yaw_step": (
            measured["max_base_yaw_step_rad"]
            <= WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD + 1e-9
        ),
        "articulation_step": (
            measured["max_articulation_step_rad"]
            <= WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD + 1e-9
        ),
        "locked_joint_stability": (
            measured["max_locked_joint_drift"] <= 1e-7
        ),
        "eef_cartesian_step": (
            measured["max_eef_cartesian_step_m"] <= eef_step_limit + 1e-9
        ),
        "eef_orientation_step": (
            measured["max_eef_orientation_step_rad"]
            <= WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD + 1e-9
        ),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        **measured,
        "base_xy_step_limit_m": WHOLE_BODY_EXECUTION_BASE_XY_STEP_M,
        "base_yaw_step_limit_rad": WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD,
        "articulation_step_limit_rad": (
            WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD
        ),
        "eef_cartesian_step_limit_m": eef_step_limit,
        "eef_orientation_step_limit_rad": (
            WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD
        ),
    }


def _whole_body_execution_subset_indices(
    dense_trajectory: Any,
    positions: Any,
    quaternions_xyzw: Any,
    *,
    joint_names: tuple[str, ...] | list[str],
    short_target: bool,
) -> np.ndarray:
    """Choose the fewest ordered dense anchors satisfying every execution edge cap."""

    q = np.asarray(_jsonable(dense_trajectory), dtype=np.float32)
    xyz = np.asarray(_jsonable(positions), dtype=np.float64)
    quaternions = np.asarray(_jsonable(quaternions_xyzw), dtype=np.float64)
    names = tuple(str(name) for name in joint_names)
    if len(q) < 2:
        raise RuntimeError("whole-body execution subset requires a moving dense path")
    _whole_body_execution_step_report(
        q,
        xyz,
        quaternions,
        joint_names=names,
        short_target=short_target,
    )

    name_to_index = {name: index for index, name in enumerate(names)}
    q64 = q.astype(np.float64)
    articulation_indices = [
        name_to_index[name]
        for name in WHOLE_BODY_ACTIVE_JOINT_NAMES
        if name
        not in {
            "base_footprint_x_joint",
            "base_footprint_y_joint",
            "base_footprint_rz_joint",
        }
    ]
    locked_indices = [
        name_to_index[name] for name in WHOLE_BODY_LOCKED_JOINT_NAMES
    ]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    normalized_quaternions = quaternions / quaternion_norms[:, None]
    eef_step_limit = (
        WHOLE_BODY_EEF_SHORT_EXECUTION_STEP_M
        if bool(short_target)
        else WHOLE_BODY_EEF_MAX_CARTESIAN_STEP_M
    )
    edge_count = np.full((len(q),), -1, dtype=np.int64)
    predecessor = np.full((len(q),), -1, dtype=np.int64)
    edge_count[0] = 0
    for end in range(1, len(q)):
        deltas = q64[:end] - q64[end]
        yaw_deltas = deltas[
            :,
            name_to_index["base_footprint_rz_joint"],
        ]
        yaw_steps = np.fromiter(
            (
                abs(_wrap_angle(float(value)))
                for value in yaw_deltas
            ),
            dtype=np.float64,
            count=end,
        )
        quaternion_dots = np.clip(
            np.abs(
                np.sum(
                    normalized_quaternions[:end]
                    * normalized_quaternions[end],
                    axis=1,
                )
            ),
            0.0,
            1.0,
        )
        admitted = (
            (edge_count[:end] >= 0)
            & (
                np.abs(
                    deltas[
                        :,
                        name_to_index["base_footprint_x_joint"],
                    ]
                )
                <= WHOLE_BODY_EXECUTION_BASE_XY_STEP_M + 1e-9
            )
            & (
                np.abs(
                    deltas[
                        :,
                        name_to_index["base_footprint_y_joint"],
                    ]
                )
                <= WHOLE_BODY_EXECUTION_BASE_XY_STEP_M + 1e-9
            )
            & (
                yaw_steps
                <= WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD + 1e-9
            )
            & (
                np.max(
                    np.abs(deltas[:, articulation_indices]),
                    axis=1,
                )
                <= WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD + 1e-9
            )
            & (
                np.max(
                    np.abs(deltas[:, locked_indices]),
                    axis=1,
                )
                <= 1e-7
            )
            & (
                np.linalg.norm(xyz[:end] - xyz[end], axis=1)
                <= eef_step_limit + 1e-9
            )
            & (
                2.0 * np.arccos(quaternion_dots)
                <= WHOLE_BODY_EEF_WRIST_MAX_ORIENTATION_STEP_RAD + 1e-9
            )
        )
        admitted_starts = np.flatnonzero(admitted)
        if admitted_starts.size:
            admitted_edge_counts = edge_count[admitted_starts]
            minimum_edge_count = int(np.min(admitted_edge_counts))
            best_start = int(
                admitted_starts[
                    np.flatnonzero(
                        admitted_edge_counts == minimum_edge_count
                    )[-1]
                ]
            )
            edge_count[end] = minimum_edge_count + 1
            predecessor[end] = best_start
    if predecessor[-1] < 0:
        raise RuntimeError(
            "whole-body dense path has no execution subset satisfying step limits"
        )

    reverse_indices = [len(q) - 1]
    cursor = len(q) - 1
    while cursor > 0:
        cursor = int(predecessor[cursor])
        if cursor < 0:
            raise RuntimeError("whole-body execution subset predecessor is invalid")
        reverse_indices.append(cursor)
    indices = np.asarray(list(reversed(reverse_indices)), dtype=np.int64)
    if (
        indices[0] != 0
        or indices[-1] != len(q) - 1
        or bool(np.any(np.diff(indices) <= 0))
    ):
        raise RuntimeError("whole-body execution subset is not a terminal ordered path")
    return indices


def _interpolate_whole_body_execution_trajectory(
    trajectory: Any,
    *,
    joint_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Densify a full-q path with the approved whole-body group step caps."""

    q = np.asarray(_jsonable(trajectory), dtype=np.float32)
    names = tuple(str(name) for name in joint_names)
    if (
        q.ndim != 2
        or q.shape[0] < 1
        or q.shape[1] != len(names)
        or len(names) != len(set(names))
    ):
        raise ValueError(
            "whole-body execution trajectory requires one unique name per q column"
        )
    if not np.isfinite(q).all():
        raise ValueError("whole-body execution trajectory contains NaN or infinity")
    missing = sorted(
        (set(WHOLE_BODY_ACTIVE_JOINT_NAMES) | set(WHOLE_BODY_LOCKED_JOINT_NAMES))
        - set(names)
    )
    if missing:
        raise ValueError(f"whole-body execution trajectory is missing joints: {missing}")

    caps = np.full((len(names),), np.inf, dtype=np.float64)
    for index, name in enumerate(names):
        if name in WHOLE_BODY_LOCKED_JOINT_NAMES:
            if np.max(np.abs(q[:, index] - q[0, index])) > 1e-7:
                raise RuntimeError(
                    f"whole-body execution path changed locked joint {name!r}"
                )
        elif name in {
            "base_footprint_x_joint",
            "base_footprint_y_joint",
        }:
            caps[index] = WHOLE_BODY_EXECUTION_BASE_XY_STEP_M
        elif name == "base_footprint_rz_joint":
            caps[index] = WHOLE_BODY_EXECUTION_BASE_YAW_STEP_RAD
        elif name in WHOLE_BODY_ACTIVE_JOINT_NAMES:
            caps[index] = WHOLE_BODY_EXECUTION_ARTICULATION_STEP_RAD
        else:
            raise RuntimeError(f"unclassified R1Pro joint {name!r}")

    interpolated: list[np.ndarray] = []
    for start, end in zip(q[:-1], q[1:], strict=True):
        ratios = np.abs(end.astype(np.float64) - start.astype(np.float64)) / caps
        intervals = max(1, int(math.ceil(float(np.max(ratios)))))
        for index in range(intervals):
            alpha = index / intervals
            interpolated.append(start + (end - start) * alpha)
    interpolated.append(q[-1])
    result = np.stack(interpolated, axis=0).astype(np.float32, copy=False)
    if len(result) > 1:
        deltas = np.abs(np.diff(result.astype(np.float64), axis=0))
        finite_caps = np.isfinite(caps)
        if bool(np.any(deltas[:, finite_caps] > caps[finite_caps] + 1e-6)):
            raise RuntimeError("whole-body execution interpolation exceeded a group cap")
    return result


def _retime_joint_trajectory(
    trajectory: Any,
    *,
    sample_dt_s: float,
    max_command_velocity: float,
    max_command_acceleration: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Time-stretch a joint path without leaving its line segments.

    cuRobo paths can contain a repeated waypoint at a stitching boundary.  At
    the 60 Hz controller that can command an abrupt stop close to the official
    acceleration limit, leaving no margin for tracking dynamics.  Linear
    interpolation over a longer duration preserves the geometric path.
    """

    q = np.asarray(_jsonable(trajectory), dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 1 or not np.isfinite(q).all():
        raise ValueError(f"joint trajectory must be finite [T,D], got {q.shape}")
    dt = float(sample_dt_s)
    velocity_limit = float(max_command_velocity)
    acceleration_limit = float(max_command_acceleration)
    if dt <= 0 or velocity_limit <= 0 or acceleration_limit <= 0:
        raise ValueError("retiming dt and dynamics limits must be positive")

    def dynamics(values: np.ndarray) -> tuple[float, float]:
        if len(values) < 2:
            return 0.0, 0.0
        velocity = np.diff(values, axis=0) / dt
        max_velocity = float(np.max(np.abs(velocity)))
        acceleration = (
            np.diff(velocity, axis=0) / dt
            if len(velocity) >= 2
            else np.zeros((0, values.shape[1]), dtype=np.float64)
        )
        max_acceleration = (
            float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
        )
        return max_velocity, max_acceleration

    original_velocity, original_acceleration = dynamics(q)
    if len(q) == 1:
        return q.astype(np.float32), {
            "method": "piecewise_linear_time_stretch",
            "original_waypoints": 1,
            "retimed_waypoints": 1,
            "duration_scale": 1.0,
            "sample_dt_s": dt,
            "max_command_velocity": 0.0,
            "max_command_acceleration": 0.0,
            "velocity_limit": velocity_limit,
            "acceleration_limit": acceleration_limit,
        }

    scale = max(
        1.0,
        original_velocity / velocity_limit,
        math.sqrt(original_acceleration / acceleration_limit)
        if original_acceleration > 0
        else 1.0,
    )
    source_time = np.arange(len(q), dtype=np.float64)
    retimed = q
    retimed_velocity = original_velocity
    retimed_acceleration = original_acceleration
    for _attempt in range(8):
        waypoint_count = int(math.ceil((len(q) - 1) * scale)) + 1
        target_time = np.linspace(0.0, float(len(q) - 1), waypoint_count)
        retimed = np.column_stack(
            [
                np.interp(target_time, source_time, q[:, joint])
                for joint in range(q.shape[1])
            ]
        )
        retimed_velocity, retimed_acceleration = dynamics(retimed)
        if (
            retimed_velocity <= velocity_limit + 1e-6
            and retimed_acceleration <= acceleration_limit + 1e-6
        ):
            break
        required = max(
            retimed_velocity / velocity_limit,
            math.sqrt(retimed_acceleration / acceleration_limit),
            1.05,
        )
        scale *= required * 1.02
        if scale > 16.0:
            raise RuntimeError("joint trajectory retiming exceeded 16x duration")
    else:
        raise RuntimeError("joint trajectory retiming did not converge")

    retimed[0] = q[0]
    retimed[-1] = q[-1]
    return retimed.astype(np.float32), {
        "method": "piecewise_linear_time_stretch",
        "path_geometry": "original_joint_polyline",
        "original_waypoints": int(len(q)),
        "retimed_waypoints": int(len(retimed)),
        "duration_scale": float((len(retimed) - 1) / max(len(q) - 1, 1)),
        "sample_dt_s": dt,
        "original_max_command_velocity": original_velocity,
        "original_max_command_acceleration": original_acceleration,
        "max_command_velocity": retimed_velocity,
        "max_command_acceleration": retimed_acceleration,
        "velocity_limit": velocity_limit,
        "acceleration_limit": acceleration_limit,
    }


def _approach_vector(value: Any | None) -> np.ndarray:
    if value is None:
        vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        vec = _as_xyz(value, name="approach_vector")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        raise ValueError("approach/press vector cannot be zero")
    return vec / norm


__all__ = [
    "LEFT_EEF_LINK",
    "RIGHT_EEF_LINK",
    "PlannerExecutor",
    "RealCuroboBackend",
    "official_task_success",
    "primitive_result",
]
