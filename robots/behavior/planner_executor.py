"""Environment-side BEHAVIOR planner primitives backed by RGB-D and cuRobo."""

from __future__ import annotations

import gc
import inspect
import json
import math
import os
import signal
import threading
import time
import types
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.camera_geometry import (
    CameraGeometryError,
    FrameCache,
    backproject_pixel_to_world,
    canonical_camera,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    ENV_ACTION_SEGMENTS,
    validate_action_chunk,
)

LEFT_EEF_LINK = "left_eef_link"
RIGHT_EEF_LINK = "right_eef_link"
EEF_LINK_BY_HAND = {"left": LEFT_EEF_LINK, "right": RIGHT_EEF_LINK}
GRIPPER_COMMAND_BY_OPENING = {"open": 1.0, "closed": -1.0}
CUROBO_COLLISION_ACTIVATION_DISTANCE_M = 0.005
LOCAL_GUARDED_IK_SEEDS = 16
ARM_WAYPOINT_TOLERANCE_RAD = 0.015
PICK_GUARDED_OVERTRAVEL_M = 0.001
GUARDED_TARGET_NEIGHBORHOOD_M = 0.03
PRESS_EEF_TO_CONTACT_OFFSET_M = 0.026
GRIPPER_CLOSE_COARSE_COMMAND_STEP = 0.05
GRIPPER_CLOSE_FINE_COMMAND_STEP = 0.00625
GRIPPER_CONTACT_SETTLE_STEPS = 10
LOCKED_BASE_XY_MAX_DRIFT_M = 0.01
LOCKED_BASE_Z_MAX_DRIFT_M = 0.01
LOCKED_BASE_RPY_MAX_DRIFT_RAD = math.radians(1.0)
LOCKED_ARTICULATION_MAX_DRIFT_RAD = 0.01
LOCKED_GRIPPER_COMMAND_MAX_DRIFT = 1e-6
PREPRESS_WARMUP_ENDPOINT_MAX_JOINT_DELTA_RAD = 0.05
PREPRESS_WARMUP_PATH_MAX_JOINT_DELTA_RAD = 0.10
MAX_BASE_STATION_SHORTLIST = 9
MAX_BASE_PLAN_CANDIDATES = 6
BASE_PLAN_ATTEMPT_TIMEOUT_S = 8.0
_ATTACHMENT_UNSET = object()


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


def _collision_margin_report_values(colliding: bool) -> dict[str, Any]:
    """Expose the world-clearance lower bound certified by OG's checker.

    OmniGibson's public ``check_collisions`` only returns whether any sphere is
    inside cuRobo's activation zone.  A clear result therefore certifies a
    lower bound equal to that configured zone; a violation is reported as zero
    rather than inventing an unavailable penetration distance.
    """

    return {
        "min_margin_m": (0.0 if colliding else CUROBO_COLLISION_ACTIVATION_DISTANCE_M),
        "margin_available": True,
        "margin_semantics": "curobo_world_activation_distance_lower_bound",
        "margin_scope": "world_only",
        "collision_activation_distance_m": CUROBO_COLLISION_ACTIVATION_DISTANCE_M,
    }


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
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
    line segment, so the caller can run the usual full world+self collision
    recheck on the returned samples.
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
            "full_collision_recheck_required": True,
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
            "full_collision_recheck_required": True,
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
        "full_collision_recheck_required": True,
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


def _apply_fixed_trajectory_hold_segments(
    action: Any,
    hold_reference: Any,
    *,
    hand: str | None,
) -> np.ndarray:
    """Freeze non-active 23D controller segments to one trajectory-start hold."""

    out = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
    hold = np.asarray(hold_reference, dtype=np.float32).reshape(ACTION_DIM)
    if hand is None:
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
            "base",
            f"{inactive}_arm",
            "left_gripper",
            "right_gripper",
        )
    for segment_name in locked_segments:
        segment = ENV_ACTION_SEGMENTS[segment_name]
        out[segment] = hold[segment]
    return out


def official_task_success(info: Any) -> bool:
    """Read only BEHAVIOR's official success bit."""
    if not isinstance(info, dict):
        return False
    done = info.get("done")
    return bool(done.get("success", False)) if isinstance(done, dict) else False


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
    collision_margin = metric_values.get("collision_margin")
    if collision_margin is None:
        collision_margin = metric_values.get("collision_margin_m")
    if collision_margin is None and isinstance(
        metric_values.get("collision_report"), dict
    ):
        collision_margin = metric_values["collision_report"].get("min_margin_m")
    return {
        "primitive_success": bool(primitive_success),
        "task_success": bool(task_success),
        "_finish": bool(task_success),
        "official_success_source": 'info["done"]["success"]',
        "stop_reason": str(stop_reason),
        "recoverable": bool(recoverable),
        "suggested_next_tool": suggested_next_tool,
        "position_error_m": metric_values.get("final_position_error_m"),
        "orientation_error_rad": metric_values.get("final_orientation_error_rad"),
        "joint_margin": joint_margin,
        "collision_margin_m": collision_margin,
        "elapsed_s": metric_values.get("elapsed_s"),
        "trace": diagnostic_values.get("trace", []),
        "trace_artifact": diagnostic_values.get("trace_artifact"),
        "metrics": metric_values,
        "diagnostics": diagnostic_values,
    }


def _planner_tool(
    name: str,
    *,
    suggested_next_tool: str | None,
):
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
                    with _wall_clock_deadline(float(timeout_s), f"planner tool {name}"):
                        result = fn(self, *args, **kwargs)
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"planner tool {name} returned {type(result)!r}, expected dict"
                    )
            except Exception as exc:
                result = self._exception_result(
                    exc,
                    suggested_next_tool=suggested_next_tool,
                )
            metrics = result.setdefault("metrics", {})
            if isinstance(metrics, dict):
                metrics.setdefault("elapsed_s", round(time.monotonic() - started, 3))
                result["elapsed_s"] = metrics["elapsed_s"]
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


def _segment_array(value: Any) -> np.ndarray:
    array = np.asarray(_jsonable(value), dtype=np.int64).reshape(-1)
    return array


class RealCuroboBackend:
    """Lazy OmniGibson/cuRobo adapter used only inside the real env process."""

    def __init__(
        self, env_facade: Any, *, output_dir: str | Path | None = None
    ) -> None:
        self.env_facade = env_facade
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._robot: Any | None = None
        self._torch: Any | None = None
        self._curobo_cls: Any | None = None
        self._embodiment_cls: Any | None = None
        self._generators: dict[str, Any] = {}
        self._invalid_generators: set[str] = set()
        self._recovering_generators: set[str] = set()
        self._config_paths: dict[str, Path] = {}
        self._base_workspace_limit_m: float | None = None
        self._last_base_candidate_summary: dict[str, Any] = {}
        self._attached_objects_by_hand: dict[str, Any] = {}
        self._active_generator: Any | None = None
        self._last_collision_step = -1
        self._collision_check_interval_steps = 4
        self._last_obstacle_update_step = -1
        self._base_obstacle_world_step = -1
        self._last_collision_report: dict[str, Any] = {
            "available": False,
            "reason": "not_checked",
            "min_margin_m": None,
        }
        self._last_actual_velocity: np.ndarray | None = None
        self._last_actual_velocity_step: int | None = None
        # Phase one has been the stable branch for the R1Pro contact corridor
        # in the real challenge scene.  Subsequent calls still rotate through
        # deterministic phases, so a failed branch remains bounded and
        # independently retryable.
        self._guarded_seed_counter = 1

    def on_simulator_state_restored(self) -> None:
        """Invalidate live feedback caches after an acceptance-only restore."""

        self._attached_objects_by_hand.clear()
        self._active_generator = None
        self._last_collision_step = -1
        self._last_obstacle_update_step = -1
        self._base_obstacle_world_step = -1
        self._last_collision_report = {
            "available": False,
            "reason": "simulator_state_restored",
            "min_margin_m": None,
        }
        self._last_actual_velocity = None
        self._last_actual_velocity_step = None
        self._base_workspace_limit_m = None
        self._guarded_seed_counter = 1

    @staticmethod
    def _generator_key(*, kind: str, hand: str = "left") -> str:
        return f"{kind}:{_normalize_hand(hand) if kind in {'arm', 'prepress_arm'} else 'left'}"

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
        """Discard a generator whose temporary OG collision state is uncertain."""

        key = self._generator_key(kind=kind, hand=hand)
        discarded = self._generators.pop(key, None)
        if discarded is not None and self._active_generator is discarded:
            self._active_generator = None
        self._invalid_generators.add(key)
        if kind == "base":
            self._base_obstacle_world_step = -1
        event = {
            "event": "generator_quarantined",
            "generator_key": key,
            "reason": str(reason),
            "monotonic_ns": time.monotonic_ns(),
            "requires_rebuild_and_warmup": True,
        }
        self._record_generator_recovery(event)
        del discarded
        gc.collect()
        return event

    def _warmup_rebuilt_generator(self, *, kind: str, hand: str) -> dict[str, Any]:
        """Prove a fresh generator safe before returning it to movement RPCs."""

        report: dict[str, Any] = {"kind": kind, "hand": hand, "checks": {}}
        if kind == "base":
            robot = self._find_robot()
            current = self._base_xy_yaw(robot)
            result = self._compute_base_plan(
                target_xyyaw=current[:3],
                timeout_s=120.0,
            )
        else:
            generator = self._generators[self._generator_key(kind=kind, hand=hand)]
            robot = self._find_robot()
            current_q = np.asarray(
                _jsonable(robot.get_joint_positions()), dtype=np.float32
            ).reshape(1, -1)
            collision = self._check_q_trajectory_collisions(
                generator,
                current_q,
                skip_obstacle_update=False,
            )
            report["checks"]["current_q_combined_collision"] = collision
            if not bool(collision.get("available", False)) or bool(
                collision.get("colliding", True)
            ):
                raise RuntimeError("fresh ARM current-q collision warmup failed")
            target_xyz, target_quat = self._curobo_eef_poses(generator, current_q)
            report["checks"]["current_q_fk"] = {
                "available": True,
                "target_xyz": target_xyz[0].tolist(),
                "target_quat_xyzw": target_quat[0].tolist(),
            }
            planner_checks = (
                (
                    "collision_ik",
                    lambda: self._compute_arm_plan(
                        hand=hand,
                        target_xyz=target_xyz[0],
                        target_quat_xyzw=target_quat[0],
                        timeout_s=30.0,
                        ik_only=True,
                        generator_kind=kind,
                    ),
                ),
                (
                    "full_trajectory",
                    lambda: self._compute_arm_plan(
                        hand=hand,
                        target_xyz=target_xyz[0],
                        target_quat_xyzw=target_quat[0],
                        timeout_s=120.0,
                        ik_only=False,
                        generator_kind=kind,
                    ),
                ),
                (
                    "guarded_ik",
                    lambda: self._compute_arm_plan(
                        hand=hand,
                        target_xyz=target_xyz[0],
                        target_quat_xyzw=target_quat[0],
                        timeout_s=60.0,
                        ik_only=True,
                        ik_world_collision_check=False,
                        return_ik_solution=True,
                        guarded_contact_target_xyz=None,
                        generator_kind=kind,
                    ),
                ),
            )
            result = {"ok": True}
            for check_name, check in planner_checks:
                result = check()
                report["checks"][check_name] = _artifact_jsonable(result)
                if not bool(result.get("ok", False)):
                    break
        if not bool(result.get("ok", False)):
            raise RuntimeError(
                f"fresh {kind}:{hand} generator warmup failed: "
                f"{result.get('stop_reason', 'unknown')}"
            )
        report["ok"] = True
        return report

    def warmup(self) -> dict[str, Any]:
        """Compile planner kernels and freeze the post-reset obstacle world.

        This runs before any public planner-tool deadline starts.  It performs
        real collision-checked queries at the current robot state, but never
        executes an action or advances the simulator.
        """
        started = time.monotonic()
        robot = self._find_robot()
        current_base = self._base_xy_yaw(robot)
        report: dict[str, Any] = {
            "status": "running",
            "current_base_xyyaw": current_base[:3].tolist(),
            "stages": {},
        }
        path = self.output_dir / "planner_curobo_warmup.json"

        def save() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        def stage(name: str, call: Any) -> Any:
            stage_started = time.monotonic()
            try:
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

        stage(
            "base_collision_world_batch1",
            lambda: self._candidate_base_collision_reports(
                robot, [current_base[:3].copy()]
            ),
        )
        base_plan = stage(
            "base_full_trajectory",
            lambda: self._compute_base_plan(
                target_xyyaw=current_base[:3], timeout_s=60.0
            ),
        )
        if not bool(base_plan.get("ok", False)):
            report["status"] = "error"
            report["elapsed_s"] = round(time.monotonic() - started, 3)
            save()
            raise RuntimeError(
                "BASE cuRobo warmup failed closed: "
                f"{base_plan.get('stop_reason', 'unknown')}"
            )
        current_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float32
        ).reshape(-1)
        stage(
            "base_full_path_collision_batch16",
            lambda: self._check_q_trajectory_collisions(
                self._generator(kind="base"),
                np.stack([current_q] * 16),
                skip_obstacle_update=True,
            ),
        )
        for hand in ("left", "right"):
            eef_pose = self.get_eef_pose(hand)
            if eef_pose is None:
                raise RuntimeError(f"R1Pro {hand} EEF pose unavailable during warmup")
            arm_plan = stage(
                f"arm_{hand}_collision_ik",
                lambda hand=hand, eef_pose=eef_pose: self._compute_arm_plan(
                    hand=hand,
                    target_xyz=np.asarray(eef_pose[0], dtype=np.float64),
                    target_quat_xyzw=np.asarray(eef_pose[1], dtype=np.float64),
                    timeout_s=30.0,
                    ik_only=True,
                ),
            )
            if not bool(arm_plan.get("ok", False)):
                report["status"] = "error"
                report["elapsed_s"] = round(time.monotonic() - started, 3)
                save()
                raise RuntimeError(
                    f"{hand} ARM cuRobo warmup failed closed: "
                    f"{arm_plan.get('stop_reason', 'unknown')}"
                )
            arm_trajectory = stage(
                f"arm_{hand}_full_trajectory",
                lambda hand=hand, eef_pose=eef_pose: self._compute_arm_plan(
                    hand=hand,
                    target_xyz=np.asarray(eef_pose[0], dtype=np.float64),
                    target_quat_xyzw=np.asarray(eef_pose[1], dtype=np.float64),
                    timeout_s=120.0,
                    ik_only=False,
                ),
            )
            if not bool(arm_trajectory.get("ok", False)):
                report["status"] = "error"
                report["elapsed_s"] = round(time.monotonic() - started, 3)
                save()
                raise RuntimeError(
                    f"{hand} ARM trajectory warmup failed closed: "
                    f"{arm_trajectory.get('stop_reason', 'unknown')}"
                )
            guarded_plan = stage(
                f"arm_{hand}_guarded_local_ik",
                lambda hand=hand, eef_pose=eef_pose: self.plan_guarded_ik_step(
                    hand=hand,
                    target_xyz=np.asarray(eef_pose[0], dtype=np.float64),
                    target_quat_xyzw=np.asarray(eef_pose[1], dtype=np.float64),
                    timeout_s=60.0,
                    contact_target_xyz=None,
                ),
            )
            if not bool(guarded_plan.get("ok", False)):
                report["status"] = "error"
                report["elapsed_s"] = round(time.monotonic() - started, 3)
                save()
                raise RuntimeError(
                    f"{hand} guarded local IK warmup failed closed: "
                    f"{guarded_plan.get('stop_reason', 'unknown')}"
                )
        report["status"] = "complete"
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        save()
        report["artifact"] = str(path)
        return report

    def warmup_prepress(
        self,
        *,
        hand: str,
        expected_attached_root: Any,
        ignore_collision_checks: bool = False,
    ) -> dict[str, Any]:
        """Warm only the attachment-aware held-arm planner used pre-press."""

        hand = _normalize_hand(hand)
        started = time.monotonic()
        path = self.output_dir / "planner_prepress_warmup.json"
        report: dict[str, Any] = {
            "status": "running",
            "generator_kind": "prepress_arm",
            "held_hand": hand,
            "base_generator_warmed": False,
            "unrelated_press_arm_generator_warmed": False,
            "stages": {},
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
                    f"{hand} pre-press warmup requires the held collision body"
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
                    f"{hand} pre-press attached body is not the selected radio"
                )
            report["attached_collision_body"] = {
                "available": True,
                "root_matches_expected_radio": True,
                "eef_links": sorted(str(key) for key in attached),
                "prim_paths": sorted(
                    str(getattr(value, "prim_path", "")) for value in attached.values()
                ),
            }
            if ignore_collision_checks:
                skipped = {
                    "ok": True,
                    "collision_checks_skipped": True,
                    "authorization": "stage3_post_press_debug_mirror_restore",
                }
                for stage_name in (
                    "current_q_attached_combined_collision",
                    "current_pose_attached_full_trajectory",
                    "identity_neighborhood_connected_path",
                ):
                    report["stages"][stage_name] = dict(skipped)
                report["robot_q_pose_jump_max"] = 0.0
                report["status"] = "complete"
                report["target_plan_validation"] = (
                    "collision warmup skipped only for an explicitly restored "
                    "stage-3 press mirror; direct stage-3 motion retains joint, "
                    "dynamics, contact, and locked-segment guards"
                )
                report["elapsed_s"] = round(time.monotonic() - started, 3)
                report["artifact"] = str(path)
                save()
                return report
            generator = self._generator(kind="prepress_arm", hand=hand)
            current_collision = self._check_q_trajectory_collisions(
                generator,
                pre_q.reshape(1, -1),
                attached_obj=attached,
                skip_obstacle_update=False,
            )
            report["stages"]["current_q_attached_combined_collision"] = {
                "ok": bool(
                    current_collision.get("available", False)
                    and not current_collision.get("colliding", True)
                ),
                "result": _artifact_jsonable(current_collision),
            }
            if not bool(current_collision.get("available", False)) or bool(
                current_collision.get("colliding", True)
            ):
                raise RuntimeError(
                    f"{hand} pre-press current-q attached collision check failed"
                )
            eef_pose = self.get_eef_pose(hand)
            if eef_pose is None:
                raise RuntimeError(f"R1Pro {hand} EEF pose unavailable during warmup")
            full_plan = self._compute_arm_plan(
                hand=hand,
                target_xyz=np.asarray(eef_pose[0], dtype=np.float64),
                target_quat_xyzw=np.asarray(eef_pose[1], dtype=np.float64),
                timeout_s=120.0,
                ik_only=False,
                attached_obj=attached,
                generator_kind="prepress_arm",
            )
            report["stages"]["current_pose_attached_full_trajectory"] = {
                "ok": bool(full_plan.get("ok", False)),
                "stop_reason": full_plan.get("stop_reason"),
                "metrics": _artifact_jsonable(full_plan.get("metrics", {})),
            }
            if not bool(full_plan.get("ok", False)):
                raise RuntimeError(
                    f"{hand} pre-press trajectory warmup failed: "
                    f"{full_plan.get('stop_reason', 'unknown')}"
                )
            q_path = np.asarray(
                _jsonable(full_plan.get("joint_trajectory")), dtype=np.float32
            )
            if (
                q_path.ndim != 2
                or q_path.shape[0] < 1
                or q_path.shape[1] != pre_q.size
                or not np.isfinite(q_path).all()
            ):
                raise RuntimeError("pre-press warmup trajectory q layout is invalid")
            first_delta = float(np.max(np.abs(q_path[0] - pre_q)))
            terminal_delta = float(np.max(np.abs(q_path[-1] - pre_q)))
            path_delta = float(np.max(np.abs(q_path - pre_q.reshape(1, -1))))
            connected_collision = self._check_q_trajectory_collisions(
                generator,
                np.vstack([pre_q.reshape(1, -1), q_path]),
                attached_obj=attached,
                skip_obstacle_update=False,
            )
            identity_ok = bool(
                first_delta <= PREPRESS_WARMUP_ENDPOINT_MAX_JOINT_DELTA_RAD
                and terminal_delta <= PREPRESS_WARMUP_ENDPOINT_MAX_JOINT_DELTA_RAD
                and path_delta <= PREPRESS_WARMUP_PATH_MAX_JOINT_DELTA_RAD
                and connected_collision.get("available", False)
                and not connected_collision.get("colliding", True)
            )
            report["stages"]["identity_neighborhood_connected_path"] = {
                "ok": identity_ok,
                "trajectory_waypoints": int(q_path.shape[0]),
                "first_max_joint_delta_rad": first_delta,
                "terminal_max_joint_delta_rad": terminal_delta,
                "path_max_joint_delta_rad": path_delta,
                "endpoint_max_joint_delta_rad": (
                    PREPRESS_WARMUP_ENDPOINT_MAX_JOINT_DELTA_RAD
                ),
                "path_max_joint_delta_limit_rad": (
                    PREPRESS_WARMUP_PATH_MAX_JOINT_DELTA_RAD
                ),
                "connected_attached_collision": _artifact_jsonable(
                    connected_collision
                ),
            }
            if not identity_ok:
                raise RuntimeError(
                    f"{hand} pre-press identity warmup path is unsafe"
                )
            post_q = np.asarray(
                _jsonable(robot.get_joint_positions()), dtype=np.float32
            ).reshape(-1)
            if post_q.shape != pre_q.shape or not np.isfinite(post_q).all():
                raise RuntimeError("pre-press warmup changed the robot q layout")
            pose_jump = float(np.max(np.abs(post_q - pre_q)))
            report["robot_q_pose_jump_max"] = pose_jump
            if pose_jump > 1e-6:
                raise RuntimeError("pre-press warmup moved the robot")
        except Exception as exc:
            report["status"] = "error"
            report["error"] = f"{type(exc).__name__}: {exc}"
            report["elapsed_s"] = round(time.monotonic() - started, 3)
            save()
            raise
        report["status"] = "complete"
        report["target_plan_validation"] = (
            "full attached-radio plan and collision recheck required again "
            "by every move_to"
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

    def _hand_config_path(self, hand: str, *, lock_trunk: bool = False) -> Path:
        hand = _normalize_hand(hand)
        cache_key = f"{hand}:prepress" if lock_trunk else hand
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
        if lock_trunk:
            trunk_indices = _indices(getattr(robot, "trunk_control_idx", []))
            joint_names = list((getattr(robot, "joints", {}) or {}).keys())
            if len(trunk_indices) != 4 or max(trunk_indices, default=-1) >= len(
                joint_names
            ):
                raise RuntimeError("R1Pro pre-press trunk joint indices unavailable")
            for index in trunk_indices:
                joint = joint_names[index]
                self._validate_joint_name(robot, joint)
                lock_joints.setdefault(joint, None)
        self._validate_lock_joint_names(robot, lock_joints)
        kinematics["lock_joints"] = dict(sorted(lock_joints.items()))
        out_dir = self.output_dir / "planner_curobo_configs"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_prepress" if lock_trunk else ""
        out = out_dir / f"r1pro_description_curobo_arm_{hand}{suffix}.yaml"
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
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

    def _probe_generator_lock_resolution(
        self,
        generator: Any,
        *,
        kind: str,
        hand: str,
        emb_sel: Any | None = None,
    ) -> None:
        """Fail early if official null lock_joints were not resolved by OG/cuRobo."""
        update = getattr(generator, "update_locked_joints", None)
        if update is not None:
            try:
                from omnigibson import lazy

                if emb_sel is None:
                    emb_sel = (
                        self._embodiment_cls.DEFAULT
                        if self._embodiment_cls is not None
                        else None
                    )
                robot = self._find_robot()
                torch = self._torch
                if torch is None:
                    import torch as torch  # type: ignore[no-redef]
                batch_size = int(getattr(generator, "batch_size", 1))
                q_pos = torch.stack([robot.get_joint_positions()] * batch_size, dim=0)
                zeros = torch.zeros_like(q_pos)
                cu_joint_state = lazy.curobo.types.state.JointState(
                    position=generator._tensor_args.to_device(q_pos),
                    velocity=generator._tensor_args.to_device(zeros),
                    acceleration=generator._tensor_args.to_device(zeros),
                    jerk=generator._tensor_args.to_device(zeros),
                    joint_names=generator.robot_joint_names,
                )
                update(cu_joint_state, emb_sel)
                kc = generator.mg[emb_sel].kinematics.kinematics_config
                positions = np.asarray(
                    _jsonable(kc.lock_jointstate.position), dtype=np.float64
                )
                if not np.isfinite(positions).all():
                    raise RuntimeError(
                        "resolved lock joint positions contain NaN or infinity"
                    )
                return
            except Exception as exc:
                raise RuntimeError(
                    f"failed to verify cuRobo {kind}:{hand} null lock_joints resolution"
                ) from exc
        lock_attrs = ("lock_joints", "_lock_joints", "locked_joints", "_locked_joints")
        for attr in lock_attrs:
            if not hasattr(generator, attr):
                continue
            value = getattr(generator, attr)
            if value is None:
                continue
            flat = _jsonable(value)
            if _contains_none(flat):
                raise RuntimeError(
                    f"cuRobo {kind}:{hand} lock_joints still contain null after runtime parsing"
                )
            return

    def _generator(self, *, kind: str, hand: str = "left") -> Any:
        self._lazy_imports()
        robot = self._find_robot()
        key = self._generator_key(kind=kind, hand=hand)
        if key in self._generators:
            return self._generators[key]
        if kind in {"arm", "prepress_arm"}:
            config_path = (
                self._hand_config_path(hand, lock_trunk=True)
                if kind == "prepress_arm"
                else self._hand_config_path(hand)
            )
            robot_cfg_path: Any = str(config_path)
            use_default_embodiment_only = True
            emb_sel = self._embodiment_cls.DEFAULT
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
        if kind == "base":
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

            generator_cls = _SceneWorkspaceCuroboMotionGenerator
        generator = generator_cls(
            robot,
            robot_cfg_path=robot_cfg_path,
            motion_cfg_kwargs={
                "trajopt_tsteps": 32,
                "num_trajopt_seeds": 4,
                "num_graph_seeds": 4,
                "finetune_trajopt_iters": 100,
            },
            batch_size=2,
            use_cuda_graph=False,
            use_default_embodiment_only=use_default_embodiment_only,
            collision_activation_distance=CUROBO_COLLISION_ACTIVATION_DISTANCE_M,
        )
        self._probe_generator_lock_resolution(
            generator,
            kind=kind,
            hand=hand,
            emb_sel=emb_sel,
        )
        self._generators[key] = generator
        if key in self._invalid_generators:
            self._recovering_generators.add(key)
            recovery_started = time.monotonic()
            try:
                recovery_report = self._warmup_rebuilt_generator(kind=kind, hand=hand)
            except BaseException as exc:
                self._generators.pop(key, None)
                self._record_generator_recovery(
                    {
                        "event": "generator_recovery_failed",
                        "generator_key": key,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_s": round(time.monotonic() - recovery_started, 3),
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                raise RuntimeError(
                    f"{key} generator recovery failed closed: {exc}"
                ) from exc
            else:
                self._invalid_generators.remove(key)
                self._record_generator_recovery(
                    {
                        "event": "generator_recovered",
                        "generator_key": key,
                        "elapsed_s": round(time.monotonic() - recovery_started, 3),
                        "monotonic_ns": time.monotonic_ns(),
                        "fresh_instance": True,
                        "warmup": _artifact_jsonable(recovery_report),
                    }
                )
            finally:
                self._recovering_generators.discard(key)
        return generator

    def _base_prismatic_workspace_limit(self, robot: Any) -> float:
        """Cover the loaded scene while preserving OG's official BASE model.

        OmniGibson 3.7.2 unconditionally clamps the virtual holonomic x/y
        joints to +/-5 m.  Public BEHAVIOR scenes may reset the robot outside
        that interval (instance 211 starts at x=5.213 m), which makes even the
        current BASE pose an impossible IK goal.  Only these two virtual
        workspace bounds are widened; all physical, velocity, acceleration,
        collision, and locked-joint limits remain official.
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
                skip_obstacle_update=skip_obstacle_update,
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
        """Try a bounded, deterministic set of collision-free wrist poses.

        A station is not rejected merely because the robot's reset wrist
        orientation cannot reach the point.  Every accepted pose is still an
        official cuRobo world-collision IK result for the requested hand and
        candidate BASE state.
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
                        "reachability_stage": (
                            "candidate_world_collision_multi_orientation_ik"
                        ),
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
                "reachability_stage": "candidate_world_collision_multi_orientation_ik",
                "attempts": attempts,
            },
        )

    def plan_arm_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=False,
                attached_obj=attached_obj,
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

    def plan_prepress_arm_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray,
        timeout_s: float,
        attached_obj: Any,
    ) -> dict[str, Any]:
        """Plan held-arm-only motion with base, trunk and press arm locked."""

        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=False,
                attached_obj=attached_obj,
                generator_kind="prepress_arm",
            )
        except Exception as exc:
            quarantine = (
                self._quarantine_generator(
                    kind="prepress_arm",
                    hand=hand,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(exc, TimeoutError)
                else None
            )
            return {
                "ok": False,
                "stop_reason": (
                    "timeout" if isinstance(exc, TimeoutError) else "planner_unavailable"
                ),
                "metrics": {
                    "error": f"{type(exc).__name__}: {exc}",
                    "generator_kind": "prepress_arm",
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
        ignore_collision_checks: bool = False,
        full_solution: bool = False,
    ) -> dict[str, Any]:
        """Solve one bounded Cartesian contact step with collision-free IK.

        World collision is deliberately deferred to the 2 mm online guarded
        controller, where actual contacts can be classified against the target
        neighborhood. Self-collision remains mandatory in the IK solution and
        large joint-branch jumps are rejected.
        """

        try:
            return self._compute_arm_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                timeout_s=timeout_s,
                ik_only=True,
                attached_obj=attached_obj,
                ik_world_collision_check=False,
                return_ik_solution=True,
                guarded_contact_target_xyz=contact_target_xyz,
                guarded_full_solution=bool(full_solution),
                # Direct stage-3 presses must move the press arm relative to
                # the held object.  The normal arm generator may recruit the
                # trunk, which moves both hands together and leaves the
                # fingertip-to-button error unchanged.  The prepress variant
                # locks the trunk so each receding-horizon step is press-arm
                # motion only.
                generator_kind=(
                    "prepress_arm" if ignore_collision_checks else "arm"
                ),
                ignore_collision_checks=bool(ignore_collision_checks),
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
        """Plan one Cartesian-certified guarded path to a contact target.

        The terminal pose is solved by cuRobo IK once.  The resulting joint
        path is then densified until forward kinematics proves every EEF step
        is at most roughly 2 mm, follows the requested approach corridor, and
        is collision-free with only the resolved contact body exempted.
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

    def certify_attached_joint_trajectory(
        self,
        *,
        hand: str,
        joint_trajectory: Any,
        attached_obj: Any,
        timeout_s: float,
    ) -> dict[str, Any]:
        """Fail-closed full-path certification with the held body attached."""

        hand = _normalize_hand(hand)
        generator = self._generator(kind="arm", hand=hand)
        self._active_generator = generator
        with _wall_clock_deadline(
            float(timeout_s), f"{hand} attached retreat certification"
        ):
            report = self._check_q_trajectory_collisions(
                generator,
                joint_trajectory,
                attached_obj=attached_obj,
                skip_obstacle_update=False,
            )
        available = bool(report.get("available", False))
        colliding = bool(report.get("colliding", True))
        return {
            "ok": available and not colliding,
            "stop_reason": (
                "certified"
                if available and not colliding
                else (
                    "attached_retreat_collision"
                    if available
                    else "collision_check_unavailable"
                )
            ),
            "metrics": {
                "method": "reverse_guarded_path_full_world+self_recheck",
                "checked_waypoints": int(
                    np.asarray(_jsonable(joint_trajectory)).shape[0]
                ),
                "attached_collision_body": {"available": attached_obj is not None},
                "collision_report": report,
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
        self._active_generator = generator
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
            "attached_obj": attached_obj,
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
            "curobo_config": str(self._hand_config_path(hand)),
            "collision_semantics": (
                "guarded_online_world_contact+self_collision_checked_path"
            ),
            "attached_collision_body": {"available": attached_obj is not None},
            "candidate_self_collision_reports": guarded_selection_state.get(
                "candidate_self_collision_reports", []
            ),
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
                attached_obj=attached_obj,
            )
        graph_report["method"] = (
            "sequential_seeded_candidates+final_state_collision_graph"
        )
        metrics["guarded_candidate_graph"] = graph_report
        if raw_q_waypoints is None:
            return {
                "ok": False,
                "stop_reason": "guarded_self_collision_path_unreachable",
                "metrics": metrics,
            }
        raw_adjacent = np.diff(np.vstack([current_q, raw_q_waypoints]), axis=0)
        raw_max_joint_delta = float(np.max(np.abs(raw_adjacent)))
        q_waypoints = raw_q_waypoints
        metrics["guarded_raw_max_adjacent_joint_delta"] = raw_max_joint_delta
        metrics["guarded_edge_collision_sample_steps_rad"] = [
            0.01,
            0.005,
            0.0025,
            0.00125,
        ]
        trajectory = None
        cartesian_attempts: list[dict[str, Any]] = []
        collision_report: dict[str, Any] | None = None
        safe_guarded_candidates: list[
            tuple[int, float, str, np.ndarray, dict[str, Any], dict[str, Any]]
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
            collision_report = self._check_q_target_excluded_collisions(
                generator,
                candidate_trajectory,
                target_xyz=np.asarray(contact_target_xyz, dtype=np.float64),
                attached_obj=attached_obj,
            )
            attempt["collision_report"] = collision_report
            if not bool(collision_report.get("available", False)):
                metrics["guarded_cartesian_attempts"] = cartesian_attempts + [attempt]
                return {
                    "ok": False,
                    "stop_reason": "collision_check_unavailable",
                    "metrics": metrics,
                }
            if bool(collision_report.get("colliding", False)):
                attempt["accepted"] = False
                attempt["reason"] = "guarded_trajectory_collision"
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
                    collision_report,
                )
            )
        metrics["guarded_cartesian_attempts"] = cartesian_attempts
        if not safe_guarded_candidates:
            metrics["guarded_target_excluded_collision_report"] = collision_report
            return {
                "ok": False,
                "stop_reason": (
                    "guarded_trajectory_collision"
                    if any(
                        attempt.get("reason") == "guarded_trajectory_collision"
                        for attempt in cartesian_attempts
                    )
                    else "guarded_cartesian_path_invalid"
                ),
                "metrics": metrics,
            }
        (
            _selected_waypoints,
            _selected_joint_path_length,
            selected_mode,
            trajectory,
            selected_cartesian_report,
            selected_collision_report,
        ) = min(
            safe_guarded_candidates,
            key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
        )
        metrics["guarded_cartesian_path_report"] = selected_cartesian_report
        metrics["guarded_selected_waypoint_mode"] = selected_mode
        metrics["guarded_path_selection"] = (
            "minimum_execution_waypoints_then_joint_path_length_after_full_recheck"
        )
        metrics["guarded_target_excluded_collision_report"] = selected_collision_report
        metrics["guarded_interpolated_waypoints"] = int(len(trajectory))
        metrics["execution_mode"] = "online_robot_q_to_action"
        return {
            "ok": True,
            "joint_trajectory": trajectory,
            # The descent was certified from current_q through trajectory.
            # A pick may reverse it only after a second full-path check with
            # the newly held object added to the collision model.
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
        skip_obstacle_update: bool = False,
        ik_world_collision_check: bool = True,
        return_ik_solution: bool = False,
        guarded_contact_target_xyz: np.ndarray | None = None,
        guarded_full_solution: bool = False,
        generator_kind: str = "arm",
        ignore_collision_checks: bool = False,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        if generator_kind not in {"arm", "prepress_arm"}:
            raise ValueError(f"invalid arm generator kind {generator_kind!r}")
        generator = self._generator(kind=generator_kind, hand=hand)
        self._active_generator = generator
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
        reachability_stage = (
            "candidate_world_collision_ik_with_initial_base"
            if base_xyyaw is not None and ik_only
            else (
                "world_collision_ik" if ik_only else "world_collision_full_trajectory"
            )
        )
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
        original_solve_ik_batch = None
        if return_ik_solution:
            original_solve_ik_batch = generator.solve_ik_batch
            guarded_selection_state: dict[str, Any] = {
                "previous": None,
                "attached_obj": attached_obj,
                "ignore_collision_checks": bool(ignore_collision_checks),
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
            with _wall_clock_deadline(float(timeout_s), f"{hand} ARM cuRobo"):
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
                    skip_obstacle_update=bool(skip_obstacle_update),
                    ik_world_collision_check=bool(ik_world_collision_check),
                )
        finally:
            if original_solve_ik_batch is not None:
                generator.solve_ik_batch = original_solve_ik_batch
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
                "guarded_online_world_contact+self_collision_checked_ik"
                if ik_only and not ik_world_collision_check
                else (
                    "candidate_world_collision_checked"
                    if base_xyyaw is not None and ik_only
                    else "world_collision_checked"
                )
            ),
            "curobo_config": str(
                self._hand_config_path(hand, lock_trunk=True)
                if generator_kind == "prepress_arm"
                else self._hand_config_path(hand)
            ),
            "generator_kind": generator_kind,
            "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
            "attached_collision_body": {"available": attached_obj is not None},
            "success_ratio": 1.0 / batch_size,
            "planner_seed_count": batch_size,
            "orientation_mode": orientation_mode,
            "obstacle_update": not bool(skip_obstacle_update),
        }
        if success_indices.size == 0:
            return {
                "ok": False,
                "stop_reason": "unreachable",
                "metrics": metrics,
            }
        if ik_only:
            metrics["reachable_by_collision_free_ik"] = bool(ik_world_collision_check)
            metrics["reachable_by_candidate_world_collision_ik"] = (
                base_xyyaw is not None
            )
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
                metrics["guarded_max_joint_delta"] = max_joint_delta
                metrics["guarded_max_joint_delta_limit"] = 0.35
                if max_joint_delta > 0.35:
                    return {
                        "ok": False,
                        "stop_reason": "guarded_ik_branch_jump",
                        "metrics": metrics,
                    }
                if guarded_full_solution:
                    guarded_start_pose = self.get_eef_pose(hand)
                    if guarded_start_pose is None:
                        raise RuntimeError(
                            f"R1Pro {hand} EEF pose unavailable for guarded path"
                        )
                    guarded_trajectory, cartesian_report = (
                        self._cartesian_certified_guarded_trajectory(
                            generator,
                            current_q=current_q,
                            q_goal=q_goal,
                            expected_distance_m=float(
                                np.linalg.norm(planner_target - guarded_start_pose[0])
                            ),
                        )
                    )
                    metrics["guarded_cartesian_path_report"] = cartesian_report
                    if guarded_trajectory is None:
                        return {
                            "ok": False,
                            "stop_reason": "guarded_cartesian_path_invalid",
                            "metrics": metrics,
                        }
                    collision_report = (
                        {
                            "available": True,
                            "colliding": False,
                            "collision_checks_skipped": True,
                            "authorization": "explicit_stage3_direct_press",
                        }
                        if ignore_collision_checks
                        else self._check_q_target_excluded_collisions(
                            generator,
                            guarded_trajectory,
                            target_xyz=np.asarray(
                                guarded_contact_target_xyz, dtype=np.float64
                            ),
                            attached_obj=attached_obj,
                        )
                    )
                    metrics["guarded_target_excluded_collision_report"] = (
                        collision_report
                    )
                    if not bool(collision_report.get("available", False)):
                        return {
                            "ok": False,
                            "stop_reason": "collision_check_unavailable",
                            "metrics": metrics,
                        }
                    if bool(collision_report.get("colliding", False)):
                        return {
                            "ok": False,
                            "stop_reason": "guarded_trajectory_collision",
                            "metrics": metrics,
                        }
                    metrics["guarded_contact_step"] = True
                    metrics["guarded_cartesian_path"] = True
                    metrics["guarded_interpolated_waypoints"] = int(
                        len(guarded_trajectory)
                    )
                    metrics["execution_mode"] = "online_robot_q_to_action"
                    return {
                        "ok": True,
                        "joint_trajectory": guarded_trajectory,
                        "metrics": metrics,
                    }
                current_eef = self.get_eef_pose(hand)
                cartesian_step_m = (
                    float(np.linalg.norm(planner_target - current_eef[0]))
                    if current_eef is not None
                    else 0.002
                )
                requested_trust_region_rad = min(
                    0.04,
                    max(0.01, 0.02 * cartesian_step_m / 0.004),
                )
                metrics["guarded_cartesian_requested_step_m"] = cartesian_step_m
                collision_attempts = []
                trust_region_rad = requested_trust_region_rad
                guarded_trajectory = None
                collision_report = None
                while True:
                    trust_scale = min(
                        1.0,
                        trust_region_rad / max(max_joint_delta, 1e-9),
                    )
                    trusted_goal = current_q + (q_goal - current_q) * trust_scale
                    candidate_trajectory = _interpolate_joint_trajectory(
                        np.vstack([current_q, trusted_goal]),
                        max_inter_dist=0.004,
                    )[1:]
                    if ignore_collision_checks:
                        candidate_report = {
                            "available": True,
                            "colliding": False,
                            "collision_checks_skipped": True,
                            "authorization": "explicit_stage3_direct_press",
                        }
                    elif guarded_contact_target_xyz is not None:
                        candidate_report = self._check_q_target_excluded_collisions(
                            generator,
                            candidate_trajectory,
                            target_xyz=guarded_contact_target_xyz,
                            attached_obj=attached_obj,
                        )
                    else:
                        candidate_report = self._check_q_combined_collisions(
                            generator,
                            candidate_trajectory,
                            attached_obj=attached_obj,
                        )
                    collision_attempts.append(
                        {
                            "trust_region_rad": trust_region_rad,
                            "trust_scale": trust_scale,
                            "waypoints": int(len(candidate_trajectory)),
                            "collision_report": candidate_report,
                        }
                    )
                    if not bool(candidate_report.get("available", False)):
                        metrics["guarded_collision_backoff_attempts"] = (
                            collision_attempts
                        )
                        return {
                            "ok": False,
                            "stop_reason": "collision_check_unavailable",
                            "metrics": metrics,
                        }
                    if not bool(candidate_report.get("colliding", False)):
                        guarded_trajectory = candidate_trajectory
                        collision_report = candidate_report
                        break
                    if trust_region_rad <= 0.0100001:
                        break
                    trust_region_rad = max(0.01, trust_region_rad * 0.5)
                metrics["guarded_collision_backoff_attempts"] = collision_attempts
                metrics["guarded_interpolated_waypoints"] = int(
                    len(guarded_trajectory) if guarded_trajectory is not None else 0
                )
                metrics["guarded_max_joint_step"] = 0.004
                metrics["guarded_joint_trust_region_requested_rad"] = (
                    requested_trust_region_rad
                )
                metrics["guarded_joint_trust_region_rad"] = trust_region_rad
                metrics["guarded_joint_trust_scale"] = min(
                    1.0,
                    trust_region_rad / max(max_joint_delta, 1e-9),
                )
                if guarded_contact_target_xyz is not None:
                    metrics["guarded_target_excluded_collision_report"] = (
                        collision_report or collision_attempts[-1]["collision_report"]
                    )
                else:
                    metrics["combined_collision_report"] = (
                        collision_report or collision_attempts[-1]["collision_report"]
                    )
                if guarded_trajectory is None or collision_report is None:
                    return {
                        "ok": False,
                        "stop_reason": (
                            "guarded_trajectory_collision"
                            if guarded_contact_target_xyz is not None
                            else "self_collision"
                        ),
                        "metrics": metrics,
                    }
                metrics["guarded_contact_step"] = True
                metrics["execution_mode"] = "online_robot_q_to_action"
                return {
                    "ok": True,
                    "joint_trajectory": guarded_trajectory,
                    "metrics": metrics,
                }
            return {"ok": True, "metrics": metrics}
        current_q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float32
        ).reshape(-1)
        candidate_reports = []
        safe_candidates: list[
            tuple[float, int, np.ndarray, dict[str, Any], dict[str, Any]]
        ] = []
        for success_index in success_indices:
            candidate_index = int(success_index)
            try:
                candidate_q = generator.path_to_joint_trajectory(
                    paths[candidate_index], get_full_js=True
                )
                candidate_q = _interpolate_joint_trajectory(
                    candidate_q, max_inter_dist=0.0075
                )
                candidate_q, retime_report = _retime_joint_trajectory(
                    candidate_q,
                    sample_dt_s=1.0 / 60.0,
                    max_command_velocity=3.0,
                    max_command_acceleration=7.5,
                )
                candidate_collision = self._check_q_trajectory_collisions(
                    generator,
                    candidate_q,
                    attached_obj=attached_obj,
                    skip_obstacle_update=True,
                )
            except Exception as exc:
                candidate_reports.append(
                    {
                        "candidate_index": candidate_index,
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            terminal_l2 = float(
                np.linalg.norm(
                    np.asarray(candidate_q[-1], dtype=np.float32).reshape(-1)
                    - current_q
                )
            )
            candidate_reports.append(
                {
                    "candidate_index": candidate_index,
                    "available": bool(candidate_collision.get("available", False)),
                    "colliding": bool(candidate_collision.get("colliding", True)),
                    "terminal_joint_l2": terminal_l2,
                    "trajectory_waypoints": int(len(candidate_q)),
                    "execution_retime": retime_report,
                    "collision_report": candidate_collision,
                }
            )
            if bool(candidate_collision.get("available", False)) and not bool(
                candidate_collision.get("colliding", True)
            ):
                safe_candidates.append(
                    (
                        terminal_l2,
                        candidate_index,
                        np.asarray(candidate_q, dtype=np.float32),
                        candidate_collision,
                        retime_report,
                    )
                )
        metrics["full_trajectory_candidate_reports"] = candidate_reports
        if not safe_candidates:
            return {
                "ok": False,
                "stop_reason": "trajectory_collision"
                if any(report.get("available") for report in candidate_reports)
                else "collision_check_unavailable",
                "metrics": metrics,
            }
        _cost, selected_index, q_traj, collision_report, retime_report = min(
            safe_candidates, key=lambda item: (item[0], item[1])
        )
        metrics["selected_full_trajectory_candidate"] = selected_index
        metrics["full_trajectory_selection"] = (
            "minimum_terminal_joint_l2_after_full_collision_recheck"
        )
        metrics["collision_report"] = collision_report
        metrics["trajectory_waypoints"] = int(len(q_traj))
        metrics["execution_retime"] = retime_report
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
            graph_selectable = seed_success.clone()
            candidate_full: list[np.ndarray] = []
            if selection_state is not None:
                edge_segments = []
                edge_offsets = []
                offset = 0
                assert previous_full is not None
                preliminary_delta = np.asarray(
                    _jsonable((solution_position - previous[0]).abs().amax(dim=-1)),
                    dtype=np.float64,
                ).reshape(-1)
                preliminary_success = np.asarray(
                    _jsonable(seed_success), dtype=bool
                ).reshape(-1)
                checked_indices = [
                    int(candidate_index)
                    for candidate_index in np.argsort(preliminary_delta)
                    if preliminary_success[candidate_index]
                ][:8]
                for seed_index in range(int(seed_success.shape[0])):
                    q_candidate, _merge_report = self._merge_ik_solution_into_full_q(
                        generator,
                        self._find_robot(),
                        result.js_solution[0, seed_index],
                    )
                    candidate = q_candidate.reshape(-1)
                    candidate_full.append(candidate)
                    if seed_index not in checked_indices:
                        edge_offsets.append((-1, -1))
                        continue
                    segment = np.concatenate(
                        [
                            _interpolate_joint_trajectory(
                                np.vstack([previous_full, candidate]),
                                max_inter_dist=sample_step,
                            )[1:]
                            for sample_step in (0.01, 0.005, 0.0025, 0.00125)
                        ],
                        axis=0,
                    )
                    edge_segments.append(segment)
                    edge_offsets.append((offset, offset + len(segment)))
                    offset += len(segment)
                ignore_collisions = bool(
                    selection_state.get("ignore_collision_checks", False)
                )
                edge_report = (
                    {
                        "available": True,
                        "colliding": False,
                        "colliding_mask": [False] * int(offset),
                        "collision_waypoints": 0,
                        "checked_waypoints": 0,
                        "collision_checks_skipped": True,
                        "authorization": "explicit_stage3_direct_press",
                    }
                    if ignore_collisions
                    else (
                        self._check_q_self_collisions(
                            generator,
                            np.concatenate(edge_segments, axis=0),
                            attached_obj=selection_state.get("attached_obj"),
                        )
                        if edge_segments
                        else {
                            "available": True,
                            "colliding": False,
                            "colliding_mask": [],
                            "collision_waypoints": 0,
                            "checked_waypoints": 0,
                            "reason": "no_successful_ik_candidates",
                        }
                    )
                )
                if not bool(edge_report.get("available", False)):
                    raise RuntimeError(
                        "guarded IK edge self-collision check unavailable: "
                        f"{edge_report.get('reason', 'unknown')}"
                    )
                edge_mask = np.asarray(
                    edge_report.get("colliding_mask"), dtype=bool
                ).reshape(-1)
                collision_candidates = (
                    np.zeros(len(edge_offsets), dtype=bool)
                    if ignore_collisions
                    else np.asarray(
                        [
                            True if start < 0 else edge_mask[start:end].any()
                            for start, end in edge_offsets
                        ],
                        dtype=bool,
                    )
                )
                collision_tensor = selectable.new_tensor(
                    collision_candidates, dtype=selectable.dtype
                )
                selectable &= ~collision_tensor
                selection_state.setdefault("candidate_q_sets", []).append(
                    np.asarray(candidate_full, dtype=np.float32).tolist()
                )
                selection_state.setdefault("candidate_selectable_masks", []).append(
                    np.asarray(_jsonable(graph_selectable), dtype=bool).tolist()
                )
                selection_state.setdefault(
                    "candidate_self_collision_reports", []
                ).append(
                    {
                        "candidate_count": int(len(collision_candidates)),
                        "ik_seed_strategy": (
                            "current_q_plus_15_deterministic_local_seeds"
                        ),
                        "ik_seed_max_perturbation_rad": 0.02,
                        "ik_successful_candidates": int(
                            np.asarray(_jsonable(graph_selectable), dtype=bool).sum()
                        ),
                        "densely_checked_candidates": len(checked_indices),
                        "pruned_far_candidates": int(
                            len(collision_candidates) - len(checked_indices)
                        ),
                        "edge_joint_sample_steps_rad": [
                            0.01,
                            0.005,
                            0.0025,
                            0.00125,
                        ],
                        "self_colliding_candidates": int(collision_candidates.sum()),
                        "targets_without_self_free_solution": int(
                            not bool(selectable.any())
                        ),
                        "graph_candidate_semantics": (
                            "all_ik_successes; final_state_graph_rechecks_each_edge"
                        ),
                        "checked_edge_waypoints": int(len(edge_mask)),
                    }
                )
            joint_delta = (solution_position - previous[0]).abs().amax(dim=-1)
            joint_delta[~selectable].fill_(float("inf"))
            graph_independent = bool(
                selection_state is not None
                and selection_state.get("graph_independent_seeds", False)
            )
            effective_success = graph_selectable if graph_independent else selectable
            success = bool(effective_success.any())
            minimum_error_index = int(joint_delta.argmin().item())
            if graph_independent:
                graph_delta = (solution_position - previous[0]).abs().amax(dim=-1)
                graph_delta[~graph_selectable].fill_(float("inf"))
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

    def _select_guarded_candidate_path(
        self,
        generator: Any,
        *,
        current_q: np.ndarray,
        candidate_q_sets: Any,
        candidate_selectable_masks: Any,
        attached_obj: Any = None,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Find a minimum-motion layered IK path with collision-free edges."""

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
            collision = self._check_q_self_collisions(
                generator,
                np.concatenate(edge_waypoints, axis=0),
                attached_obj=attached_obj,
            )
            if not bool(collision.get("available", False)):
                raise RuntimeError(
                    "guarded candidate edge collision check unavailable: "
                    f"{collision.get('reason', 'unknown')}"
                )
            collision_mask = np.asarray(
                collision.get("colliding_mask"), dtype=bool
            ).reshape(-1)
            eef_positions = self._curobo_eef_positions(
                generator, np.concatenate(edge_waypoints, axis=0)
            )
            next_costs = np.full(candidate_count, np.inf, dtype=np.float64)
            next_backpointer = np.full(candidate_count, -1, dtype=np.int64)
            clear_edges = 0
            cartesian_rejected_edges = 0
            for (
                previous_index,
                candidate_index,
                start_offset,
                end_offset,
                edge_cost,
            ) in edges:
                if bool(collision_mask[start_offset:end_offset].any()):
                    continue
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
                clear_edges += 1
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
                    "collision_free_edges": clear_edges,
                    "cartesian_rejected_edges": cartesian_rejected_edges,
                    "reachable_candidates": int(np.isfinite(next_costs).sum()),
                    "checked_edge_waypoints": int(len(collision_mask)),
                }
            )
            if not bool(np.isfinite(next_costs).any()):
                return None, {
                    "available": True,
                    "selected": False,
                    "reason": "no_self_collision_free_path",
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
                collision_only = bool(
                    summary.get("traversable_count", 0) > 0
                    and summary.get("collision_colliding_count", 0)
                    == summary.get("traversable_count", 0)
                )
                collision_unavailable = bool(
                    summary.get("collision_unavailable_count", 0) > 0
                    and summary.get("collision_free_count", 0) == 0
                )
                return {
                    "ok": False,
                    "stop_reason": (
                        "navigation_collision"
                        if collision_only
                        else (
                            "planner_unavailable"
                            if collision_unavailable
                            else "navigation_unreachable"
                        )
                    ),
                    "metrics": {
                        "candidate_count": 0,
                        "reason": "no traversable collision-free reachable station",
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
                    "base_collision_report": item.get("base_collision_report"),
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
                    best = candidate["xyyaw"]
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
            "collision_free_count": 0,
            "collision_colliding_count": 0,
            "collision_unavailable_count": 0,
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
        collision_started = time.monotonic()
        collision_reports = self._candidate_base_collision_reports(
            robot,
            [item["xyyaw"] for item in connected],
        )
        self._last_base_candidate_summary["collision_free_count"] = sum(
            bool(report.get("available", False))
            and not bool(report.get("colliding", False))
            for report in collision_reports
        )
        self._last_base_candidate_summary["collision_colliding_count"] = sum(
            bool(report.get("available", False))
            and bool(report.get("colliding", False))
            for report in collision_reports
        )
        self._last_base_candidate_summary["collision_unavailable_count"] = sum(
            not bool(report.get("available", False)) for report in collision_reports
        )
        self._record_base_phase(
            {
                "phase": "candidate_collision_complete",
                "elapsed_s": round(time.monotonic() - collision_started, 3),
                "summary": dict(self._last_base_candidate_summary),
                "remaining_s": float(deadline) - time.monotonic(),
            }
        )
        ranked: list[dict[str, Any]] = []
        for item, base_collision in zip(connected, collision_reports):
            if len(ranked) >= MAX_BASE_PLAN_CANDIDATES:
                break
            if not bool(base_collision.get("available", False)) or bool(
                base_collision.get("colliding", False)
            ):
                continue
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
                    "base_collision_report": base_collision,
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
        """Return a collision-free precontact target between base and surface."""

        target = np.asarray(surface_target_xyz, dtype=np.float64).reshape(3)
        toward_surface = target[:2] - np.asarray(candidate_xyyaw, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(toward_surface))
        if norm <= 1e-9:
            return target.copy()
        result = target.copy()
        result[:2] -= toward_surface / norm * float(clearance_m)
        return result

    def _candidate_base_collision_reports(
        self,
        robot: Any,
        candidates: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Validate all traversable BASE stations in one official OG query."""

        if not candidates:
            return []
        try:
            generator = self._generator(kind="base")
            step = int(getattr(self.env_facade, "_env_steps", -1))
            refresh_obstacles = step < 0 or self._base_obstacle_world_step != step
            q = np.stack(
                [
                    self._initial_joint_pos_for_base_candidate(robot, candidate)
                    for candidate in candidates
                ],
                axis=0,
            )
            colliding = generator.check_collisions(
                q,
                self_collision_check=False,
                skip_obstacle_update=not refresh_obstacles,
                attached_obj=None,
            )
            if refresh_obstacles:
                self._base_obstacle_world_step = step
            collision_array = np.asarray(_jsonable(colliding), dtype=bool).reshape(-1)
            if collision_array.size != len(candidates):
                raise RuntimeError(
                    "BASE collision result count does not match candidate count"
                )
        except TimeoutError:
            # An interrupted CUDA collision-world update invalidates the
            # generator.  Propagate so the owning BASE plan quarantines it;
            # never downgrade a hard timeout to an ordinary unavailable
            # candidate and reuse uncertain planner state.
            raise
        except Exception as exc:
            return [
                {
                    "available": False,
                    "colliding": None,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "checked_waypoints": 0,
                    "candidate_xyyaw": candidate.tolist(),
                }
                for candidate in candidates
            ]
        return [
            {
                "available": True,
                "colliding": bool(is_colliding),
                "collision_waypoints": int(bool(is_colliding)),
                "checked_waypoints": 1,
                **_collision_margin_report_values(bool(is_colliding)),
                "self_collision_check": False,
                "obstacle_world_refreshed": bool(refresh_obstacles),
                "candidate_xyyaw": candidate.tolist(),
            }
            for candidate, is_colliding in zip(candidates, collision_array)
        ]

    def _candidate_base_collision_report(
        self,
        robot: Any,
        candidate: np.ndarray,
    ) -> dict[str, Any]:
        """Check the full robot at a candidate base state, failing closed."""
        return self._candidate_base_collision_reports(robot, [candidate])[0]

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
    ) -> dict[str, Any]:
        generator = self._generator(kind="base")
        self._active_generator = generator
        emb_sel = self._embodiment_cls.BASE
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        robot = self._find_robot()
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
        hard_attempt_timeout_s = min(float(timeout_s), BASE_PLAN_ATTEMPT_TIMEOUT_S)
        attempt_timeout_s = max(0.1, hard_attempt_timeout_s - 2.0)
        with _wall_clock_deadline(hard_attempt_timeout_s, "BASE cuRobo candidate"):
            full_results = generator.compute_trajectories(
                planner_targets,
                planner_quats,
                max_attempts=5,
                timeout=attempt_timeout_s,
                ik_fail_return=5,
                enable_finetune_trajopt=True,
                finetune_attempts=1,
                return_full_result=True,
                success_ratio=1.0 / batch_size,
                ik_only=False,
                skip_obstacle_update=bool(skip_obstacle_update),
                emb_sel=emb_sel,
            )
        success_chunks = []
        paths = []
        result_statuses = []
        for result in full_results:
            result_success = np.asarray(
                _jsonable(result.success),
                dtype=bool,
            ).reshape(-1)
            success_chunks.append(result_success)
            if result_success.any():
                result_paths = result.get_paths()
                if result_paths is not None:
                    paths.extend(list(result_paths))
            result_statuses.append(
                {
                    "success": result_success.tolist(),
                    "status": str(getattr(result, "status", "unavailable")),
                    "valid_query": str(getattr(result, "valid_query", "unavailable")),
                }
            )
        success_array = (
            np.concatenate(success_chunks)
            if success_chunks
            else np.zeros((0,), dtype=bool)
        )
        success_indices = np.flatnonzero(success_array)
        metrics = {
            "successes": success_array.tolist(),
            "ik_only": False,
            "curobo_config": str(self._base_config_path()),
            "curobo_api": "CuRoboMotionGenerator.compute_trajectories",
            "success_ratio": 1.0 / batch_size,
            "planner_seed_count": batch_size,
            "max_attempts": 5,
            "ik_fail_return": 5,
            "attempt_timeout_s": attempt_timeout_s,
            "hard_attempt_timeout_s": hard_attempt_timeout_s,
            "motion_gen_results": result_statuses,
            "base_prismatic_workspace_limit_m": self._base_workspace_limit_m,
            "base_prismatic_workspace_limit_source": "scene_envelope_plus_2m",
            "obstacle_update": not bool(skip_obstacle_update),
        }
        if success_indices.size == 0:
            return {"ok": False, "stop_reason": "base_plan_failed", "metrics": metrics}
        path = paths[int(success_indices[0])]
        q_traj = generator.path_to_joint_trajectory(
            path,
            get_full_js=True,
            emb_sel=emb_sel,
        )
        dense_q_traj = _interpolate_joint_trajectory(q_traj, max_inter_dist=0.01)
        collision_report = self._check_q_trajectory_collisions(
            generator,
            dense_q_traj,
            skip_obstacle_update=True,
        )
        # CuRobo already time-parameterizes against the simulator step.  Do not
        # decimate this path: position-drive setpoint jumps reintroduce unsafe
        # acceleration even though the dense path itself passed collision and
        # dynamics validation.
        execution_q_traj = dense_q_traj.copy()
        metrics.update(
            {
                "trajectory_waypoints": int(len(dense_q_traj)),
                "dense_collision_check_waypoints": int(len(dense_q_traj)),
                "execution_waypoints": int(len(execution_q_traj)),
                "execution_resampling": {
                    "source": "dense_curobo_path",
                    "dense_max_joint_step": 0.01,
                    "execution_max_joint_step": 0.01,
                    "stride": 1,
                    "controller": "official_position_holonomic_base",
                    "planner_base_isaac_kp": 2_000_000.0,
                    "planner_base_isaac_kd": 100_000.0,
                },
                "collision_report": collision_report,
            }
        )
        if not bool(collision_report.get("available", False)):
            return {
                "ok": False,
                "stop_reason": "collision_check_unavailable",
                "metrics": metrics,
            }
        if bool(collision_report.get("colliding", False)):
            return {
                "ok": False,
                "stop_reason": "trajectory_collision",
                "metrics": metrics,
            }
        return {
            "ok": True,
            "joint_trajectory": execution_q_traj,
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
        base_idx = _indices(getattr(robot, "base_control_idx", []))
        articulation_idx = _indices(getattr(robot, "trunk_control_idx", []))
        arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
        if hand is None:
            for side in ("left", "right"):
                articulation_idx.extend(_indices(arm_control_idx.get(side, [])))
        else:
            articulation_idx.extend(_indices(arm_control_idx.get(hand, [])))
        if len(base_idx) < 3 or not articulation_idx:
            return {
                "available": False,
                "reason": "R1Pro controlled joint indices unavailable",
                "reached": False,
            }
        diff = target - current
        base_xy_error = float(np.max(np.abs(diff[base_idx[:2]])))
        base_yaw_error = abs(_wrap_angle(float(diff[base_idx[2]])))
        articulation_error = float(np.max(np.abs(diff[sorted(set(articulation_idx))])))
        if hand is None:
            reached = base_xy_error <= 0.01 and base_yaw_error <= math.radians(1.0)
        else:
            reached = articulation_error <= ARM_WAYPOINT_TOLERANCE_RAD
        return {
            "available": True,
            "reached": bool(reached),
            "max_articulation_error_rad": articulation_error,
            "articulation_waypoint_tolerance_rad": ARM_WAYPOINT_TOLERANCE_RAD,
            "max_base_xy_error_m": base_xy_error,
            "base_yaw_error_rad": base_yaw_error,
            "base_waypoint_xy_tolerance_m": 0.01,
            "base_waypoint_yaw_tolerance_rad": math.radians(1.0),
        }

    def capture_trajectory_hold_reference(self, *, hand: str | None) -> dict[str, Any]:
        """Capture fixed world/joint targets and gripper commands once.

        The reference intentionally contains q-space values, not a packed 23D
        action.  In particular, an ARM trajectory's six virtual base joints are
        fixed in world coordinates and converted through ``q_to_action`` again
        at every control step as the robot root frame changes.
        """

        robot = self._find_robot()
        q = np.asarray(
            _jsonable(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
        base_indices: list[int] = []
        articulation_indices: list[int] = []
        if hand is None:
            scope = "base_trajectory_locks_trunk_and_both_arms"
            articulation_indices.extend(
                _indices(getattr(robot, "trunk_control_idx", []))
            )
            for side in ("left", "right"):
                articulation_indices.extend(_indices(arm_control_idx.get(side, [])))
        else:
            hand = _normalize_hand(hand)
            inactive = "right" if hand == "left" else "left"
            scope = f"{hand}_arm_trajectory_locks_full_base_and_{inactive}_arm"
            base_indices = _indices(getattr(robot, "base_idx", []))
            if len(base_indices) != 6:
                raise RuntimeError(
                    "R1Pro ARM trajectory requires all six virtual base joint indices"
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

    def _check_q_trajectory_collisions(
        self,
        generator: Any,
        q_traj: Any,
        *,
        attached_obj: Any = None,
        skip_obstacle_update: bool = False,
    ) -> dict[str, Any]:
        if not hasattr(generator, "check_collisions"):
            report = {
                "available": False,
                "reason": "check_collisions_unavailable",
                "min_margin_m": None,
            }
            self._last_collision_report = report
            return report
        try:
            torch = self._torch
            if torch is None:
                import torch as torch  # type: ignore[no-redef]
            q_tensor = torch.as_tensor(
                np.asarray(_jsonable(q_traj), dtype=np.float32),
                dtype=torch.float32,
            )
            world_collision_chunks = []
            waypoint_count = int(q_tensor.shape[0])
            for start in range(0, waypoint_count, 16):
                world_collision_chunks.append(
                    generator.check_collisions(
                        q_tensor[start : start + 16],
                        self_collision_check=False,
                        skip_obstacle_update=bool(skip_obstacle_update) or start > 0,
                        attached_obj=attached_obj,
                    )
                )
            world_colliding = np.concatenate(
                [
                    np.asarray(_jsonable(chunk), dtype=bool).reshape(-1)
                    for chunk in world_collision_chunks
                ]
            )
            self_report = self._check_q_self_collisions(
                generator,
                q_tensor,
                attached_obj=attached_obj,
            )
            if not bool(self_report.get("available", False)):
                raise RuntimeError(
                    f"self collision check unavailable: {self_report.get('reason')}"
                )
        except TimeoutError:
            raise
        except Exception as exc:
            report = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_margin_m": None,
            }
            self._last_collision_report = report
            return report
        world_array = np.asarray(_jsonable(world_colliding), dtype=bool)
        world_hit = bool(world_array.any())
        self_hit = bool(self_report.get("colliding", False))
        combined_hit = world_hit or self_hit
        world_report = {
            "available": True,
            "colliding": world_hit,
            "collision_waypoints": int(world_array.sum()),
            "checked_waypoints": int(world_array.size),
            **_collision_margin_report_values(world_hit),
        }
        report = {
            "available": True,
            "colliding": combined_hit,
            "world_colliding": world_hit,
            "self_colliding": self_hit,
            "collision_waypoints": max(
                int(world_array.sum()),
                int(self_report.get("collision_waypoints", 0)),
            ),
            "checked_waypoints": int(world_array.size),
            **_collision_margin_report_values(world_hit),
            "min_margin_m": (
                0.0 if combined_hit else CUROBO_COLLISION_ACTIVATION_DISTANCE_M
            ),
            "margin_semantics": (
                "world_activation_lower_bound_with_self_collision_boolean"
            ),
            "collision_margin_scope": "world_activation_lower_bound_only",
            "combined_clearance_certified": False,
            "world_collision_report": world_report,
            "self_collision_report": self_report,
            "attached_collision_body": {"available": attached_obj is not None},
            "obstacle_world_refreshed": not bool(skip_obstacle_update),
        }
        self._last_collision_report = report
        step = int(getattr(self.env_facade, "_env_steps", -1))
        self._last_collision_step = step
        if not bool(skip_obstacle_update):
            self._last_obstacle_update_step = step
        return report

    def _check_q_self_collisions(
        self,
        generator: Any,
        q_traj: Any,
        *,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        """Run OG/cuRobo's robot-sphere self-collision constraint only."""

        attached_info = None
        try:
            torch = self._torch
            if torch is None:
                import torch as torch  # type: ignore[no-redef]

            emb_sel = self._embodiment_cls.DEFAULT
            robot = self._find_robot()
            q_tensor = generator._tensor_args.to_device(
                torch.as_tensor(
                    np.asarray(_jsonable(q_traj), dtype=np.float32),
                    dtype=torch.float32,
                )
            )
            current_q = robot.get_joint_positions().unsqueeze(0)
            current_state = generator._tensor_args.to_device(current_q)
            from omnigibson import lazy

            zeros = torch.zeros_like(current_state)
            cu_current = lazy.curobo.types.state.JointState(
                position=current_state,
                velocity=generator._tensor_args.to_device(zeros),
                acceleration=generator._tensor_args.to_device(zeros),
                jerk=generator._tensor_args.to_device(zeros),
                joint_names=generator.robot_joint_names,
            )
            generator.update_locked_joints(cu_current, emb_sel)
            cu_js = lazy.curobo.types.state.JointState(
                position=q_tensor,
                joint_names=generator.robot_joint_names,
            ).get_ordered_joint_state(generator.mg[emb_sel].kinematics.joint_names)
            attached_info = generator._attach_objects_to_robot(
                attached_obj=attached_obj,
                attached_obj_scale=None,
                cu_js_batch=cu_js,
                emb_sel=emb_sel,
            )
            spheres = generator.mg[emb_sel].compute_kinematics(cu_js).robot_spheres
            spheres = spheres.unsqueeze(dim=1)
            with torch.no_grad():
                distances = (
                    generator.mg[emb_sel]
                    .rollout_fn.robot_self_collision_constraint.forward(spheres)
                    .squeeze(1)
                )
                colliding = distances > 0.0
            array = np.asarray(_jsonable(colliding), dtype=bool).reshape(-1)
            return {
                "available": True,
                "colliding": bool(array.any()),
                "colliding_mask": array.tolist(),
                "collision_waypoints": int(array.sum()),
                "checked_waypoints": int(array.size),
                "min_margin_m": None,
                "margin_available": False,
                "margin_semantics": "self_collision_boolean_only",
                "collision_scope": "self_only",
                "attached_collision_body": {"available": attached_obj is not None},
            }
        except TimeoutError:
            raise
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_margin_m": None,
                "collision_scope": "self_only",
            }
        finally:
            if attached_info is not None:
                try:
                    generator._detach_objects_from_robot(attached_info, emb_sel)
                except Exception:
                    pass

    def _check_q_target_excluded_collisions(
        self,
        generator: Any,
        q_traj: Any,
        *,
        target_xyz: np.ndarray,
        attached_obj: Any = None,
    ) -> dict[str, Any]:
        """Check a guarded configuration with only its resolved target removed.

        World collision and self collision are queried independently with the
        official OG/cuRobo constraints, then combined explicitly.  This avoids
        losing attribution in OG's mixed ``self_collision_check=True`` result.
        The normal world is restored before this method returns, including all
        fail-closed error paths.
        """

        target_object = self._target_object_for_point(target_xyz)
        if target_object is None:
            return {
                "available": False,
                "reason": "target_collision_body_unresolved",
                "target_object_resolved": False,
                "normal_world_restored": True,
            }
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        q_tensor = torch.as_tensor(
            np.asarray(_jsonable(q_traj), dtype=np.float32),
            dtype=torch.float32,
        )
        self_report = self._check_q_self_collisions(
            generator,
            q_tensor,
            attached_obj=attached_obj,
        )
        if not bool(self_report.get("available", False)):
            return {
                "available": False,
                "reason": self_report.get("reason", "self_collision_check_unavailable"),
                "target_object_resolved": True,
                "normal_world_restored": True,
                "self_collision_report": self_report,
            }
        self_array = np.asarray(self_report.get("colliding_mask"), dtype=bool).reshape(
            -1
        )
        if self_array.shape != (int(q_tensor.shape[0]),):
            return {
                "available": False,
                "reason": "self_collision_mask_shape_mismatch",
                "target_object_resolved": True,
                "normal_world_restored": True,
                "self_collision_report": self_report,
            }
        query_error: Exception | None = None
        restore_error: Exception | None = None
        world_array = np.ones((int(q_tensor.shape[0]),), dtype=bool)
        try:
            generator.update_obstacles(ignore_objects=[target_object])
            world_array = np.asarray(
                _jsonable(
                    generator.check_collisions(
                        q_tensor,
                        self_collision_check=False,
                        skip_obstacle_update=True,
                        attached_obj=attached_obj,
                    )
                ),
                dtype=bool,
            ).reshape(-1)
        except TimeoutError:
            raise
        except Exception as exc:
            query_error = exc
        finally:
            try:
                generator.update_obstacles()
            except Exception as exc:
                restore_error = exc
        if query_error is not None or restore_error is not None:
            error = restore_error or query_error
            assert error is not None
            return {
                "available": False,
                "reason": f"{type(error).__name__}: {error}",
                "target_object_resolved": True,
                "normal_world_restored": restore_error is None,
            }
        world_hit = bool(world_array.any())
        combined_array = world_array | self_array
        combined_hit = bool(combined_array.any())
        return {
            "available": True,
            "colliding": combined_hit,
            "target_object_resolved": True,
            "world_without_target_colliding": world_hit,
            "self_or_unrelated_world_colliding": combined_hit,
            "self_colliding": bool(self_array.any()),
            "collision_waypoints": int(combined_array.sum()),
            "checked_waypoints": int(combined_array.size),
            "normal_world_restored": True,
            "target_only_activation_verified": not combined_hit,
            "self_collision_report": self_report,
            "world_without_target_report": {
                "available": True,
                "colliding": world_hit,
                "collision_waypoints": int(world_array.sum()),
                "checked_waypoints": int(world_array.size),
                **_collision_margin_report_values(world_hit),
            },
            "combined_target_excluded_report": {
                "available": True,
                "colliding": combined_hit,
                "collision_waypoints": int(combined_array.sum()),
                "checked_waypoints": int(combined_array.size),
                "margin_available": False,
                "margin_semantics": (
                    "combined_self_and_unrelated_world_collision_boolean"
                ),
            },
        }

    def _check_q_combined_collisions(
        self,
        generator: Any,
        q_traj: Any,
        *,
        attached_obj: Any = None,
        skip_obstacle_update: bool = False,
    ) -> dict[str, Any]:
        """Use OG's public combined world+self collision query."""

        try:
            torch = self._torch
            if torch is None:
                import torch as torch  # type: ignore[no-redef]
            q_tensor = torch.as_tensor(
                np.asarray(_jsonable(q_traj), dtype=np.float32),
                dtype=torch.float32,
            )
            collision = generator.check_collisions(
                q_tensor,
                self_collision_check=True,
                skip_obstacle_update=bool(skip_obstacle_update),
                attached_obj=attached_obj,
            )
            array = np.asarray(_jsonable(collision), dtype=bool).reshape(-1)
            hit = bool(array.any())
            return {
                "available": True,
                "colliding": hit,
                "collision_waypoints": int(array.sum()),
                "checked_waypoints": int(array.size),
                "collision_scope": "combined_world_and_self",
                "min_margin_m": (
                    0.0 if hit else CUROBO_COLLISION_ACTIVATION_DISTANCE_M
                ),
                "margin_available": True,
                "margin_semantics": (
                    "combined_boolean_with_world_activation_lower_bound"
                ),
                "collision_activation_distance_m": (
                    CUROBO_COLLISION_ACTIVATION_DISTANCE_M
                ),
                "obstacle_world_refreshed": not bool(skip_obstacle_update),
            }
        except TimeoutError:
            raise
        except Exception as exc:
            return {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "collision_scope": "combined_world_and_self",
                "min_margin_m": None,
            }

    def collision_report(self, *, force: bool = False) -> dict[str, Any]:
        step = int(getattr(self.env_facade, "_env_steps", -1))
        if (
            not force
            and self._last_collision_report.get("available")
            and step >= 0
            and self._last_collision_step >= 0
            and step - self._last_collision_step < self._collision_check_interval_steps
        ):
            return dict(self._last_collision_report)
        try:
            generator = self._active_generator or self._generator(
                kind="arm", hand="left"
            )
            robot = self._find_robot()
            q = robot.get_joint_positions().reshape(1, -1)
            attached: dict[str, Any] = {}
            for side in ("left", "right"):
                item = self.get_attached_object(side)
                if item:
                    attached.update(item)
            refresh_obstacles = (
                self._last_obstacle_update_step < 0
                or step - self._last_obstacle_update_step
                >= self._collision_check_interval_steps
            )
            combined = self._check_q_combined_collisions(
                generator,
                q,
                attached_obj=attached or None,
                skip_obstacle_update=not refresh_obstacles,
            )
            if not bool(combined.get("available", False)):
                self._last_collision_report = combined
            elif bool(combined.get("colliding", False)):
                # Keep the common clear-state query to one official combined
                # world+self kernel per waypoint.  Only a hit pays for the
                # separate world/self attribution needed by guarded target
                # contact handling; both paths remain fail closed.
                self._check_q_trajectory_collisions(
                    generator,
                    q,
                    attached_obj=attached or None,
                    skip_obstacle_update=not refresh_obstacles,
                )
            else:
                self._last_collision_report = {
                    **combined,
                    "world_colliding": False,
                    "self_colliding": False,
                    "combined_clearance_certified": False,
                    "attached_collision_body": {
                        "available": bool(attached),
                    },
                }
                self._last_collision_step = step
                if refresh_obstacles:
                    self._last_obstacle_update_step = step
        except TimeoutError:
            raise
        except Exception as exc:
            self._last_collision_report = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "min_margin_m": None,
            }
            self._last_collision_step = step
        return dict(self._last_collision_report)

    def collision_margin(self) -> float | None:
        return self._last_collision_report.get("min_margin_m")

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
            controlled = _indices(getattr(robot, "base_control_idx", []))
            controlled.extend(_indices(getattr(robot, "trunk_control_idx", [])))
            arm_control_idx = getattr(robot, "arm_control_idx", {}) or {}
            for side in ("left", "right"):
                controlled.extend(_indices(arm_control_idx.get(side, [])))
            controlled = sorted(set(controlled))
            if not controlled or velocity.shape != velocity_limits.shape:
                raise RuntimeError("controlled velocity limits unavailable")
            actual = np.abs(velocity[controlled])
            limits = np.abs(velocity_limits[controlled])
            if not np.isfinite(actual).all() or not np.isfinite(limits).all():
                raise RuntimeError("non-finite actual velocity or limit")
            velocity_ratio = float(np.max(actual / np.maximum(limits, 1e-9)))
            velocity_peak_index = int(np.argmax(actual))
            dof_names = list(getattr(robot, "dof_names_ordered", []))
            controlled_names = [
                str(dof_names[idx]) if idx < len(dof_names) else f"dof_{idx}"
                for idx in controlled
            ]
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
                    velocity_ratio <= 1.0 + 1e-3
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
                "sample_dt_s": sample_dt_s,
                "source": "robot.get_joint_velocities+control_limits+curobo_cspace",
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

    def guarded_contact_safety_report(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        allowed_contact_distance_m: float = 0.025,
    ) -> dict[str, Any]:
        """Verify a guarded collision is attributable only to its target body.

        The target is removed only for this diagnostic world query. The normal
        cuRobo world is restored before returning; self collision is checked
        independently and can never be exempted.
        """

        target_object = self._target_object_for_point(target_xyz)
        if target_object is None:
            return {
                "available": False,
                "reason": "target_collision_body_unresolved",
                "target_object_resolved": False,
            }
        generator = self._active_generator or self._generator(kind="arm", hand=hand)
        robot = self._find_robot()
        q = robot.get_joint_positions().reshape(1, -1)
        attached: dict[str, Any] = {}
        for side in ("left", "right"):
            item = self.get_attached_object(side)
            if item:
                attached.update(item)
        collision_report = self._check_q_target_excluded_collisions(
            generator,
            q,
            target_xyz=target_xyz,
            attached_obj=attached or None,
        )
        if not bool(collision_report.get("available", False)):
            return {
                "available": False,
                "reason": collision_report.get(
                    "reason", "target_excluded_collision_check_unavailable"
                ),
                "target_object_resolved": True,
                "target_excluded_collision_report": collision_report,
                "normal_world_restored": collision_report.get(
                    "normal_world_restored", False
                ),
            }
        contact = self.contact_report(
            hand=hand,
            target_xyz=target_xyz,
            allowed_contact_distance_m=allowed_contact_distance_m,
        )
        return {
            "available": bool(contact.get("available", False)),
            "target_object_resolved": True,
            "self_colliding": bool(collision_report.get("colliding", True)),
            "world_without_target_colliding": bool(
                collision_report.get("world_without_target_colliding", True)
            ),
            "target_excluded_collision_report": collision_report,
            "world_without_target_report": collision_report.get(
                "world_without_target_report"
            ),
            "contact_report": contact,
            "target_only_activation_verified": bool(
                collision_report.get("target_only_activation_verified", False)
            ),
        }

    def resolve_target_attachment(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
    ) -> Any:
        """Lock the RGB-D target's exact root collision body before grasping."""

        hand = _normalize_hand(hand)
        target_object = self._target_object_for_point(target_xyz)
        if target_object is None:
            return None
        root_link = getattr(target_object, "root_link", None)
        if root_link is None:
            raise RuntimeError("pick target has no root_link collision body")
        return {EEF_LINK_BY_HAND[hand]: root_link}

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


def _contains_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_none(item) for item in value)
    return False


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
        self._last_guarded_retreat_paths: dict[str, np.ndarray] = {}

    def on_simulator_state_restored(self) -> None:
        """Reset executor-local state without exposing restore over planner RPC."""

        self.last_info = None
        self._last_guarded_retreat_paths.clear()
        restored = getattr(self.backend, "on_simulator_state_restored", None)
        if callable(restored):
            restored()

    def warmup(self) -> dict[str, Any]:
        warmup = getattr(self.backend, "warmup", None)
        if not callable(warmup):
            raise RuntimeError("planner backend does not implement safety warmup")
        return dict(warmup())

    def warmup_prepress(
        self,
        *,
        hand: str,
        expected_attached_root: Any,
        ignore_collision_checks: bool = False,
    ) -> dict[str, Any]:
        warmup = getattr(self.backend, "warmup_prepress", None)
        if not callable(warmup):
            raise RuntimeError(
                "planner backend does not implement pre-press safety warmup"
            )
        return dict(
            warmup(
                hand=hand,
                expected_attached_root=expected_attached_root,
                ignore_collision_checks=bool(ignore_collision_checks),
            )
        )

    @_planner_tool("observe", suggested_next_tool="observe")
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

    @_planner_tool("pixel_to_world", suggested_next_tool="observe")
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

    @_planner_tool("navigate_to", suggested_next_tool="observe")
    def navigate_to(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        standoff_m: float = 0.85,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            target = self._world_target(target_xyz, frame=frame)
            plan = self.backend.plan_base_trajectory(
                hand=hand,
                target_xyz=target,
                standoff_m=float(standoff_m),
                timeout_s=self._remaining_s(deadline),
            )
            if not plan.get("ok"):
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=str(plan.get("stop_reason", "base_plan_failed")),
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics=plan.get("metrics", {}),
                    diagnostics=plan,
                )
            actions = plan.get("actions")
            execution = self._execute_actions(
                validate_action_chunk(actions) if actions is not None else None,
                hand=hand,
                target_xyz=None,
                target_quat_xyzw=None,
                position_tolerance_m=0.05,
                orientation_tolerance_rad=math.radians(5.0),
                timeout_s=self._remaining_s(deadline),
                require_pose=False,
                base_goal_xyyaw=np.asarray(plan.get("base_goal"), dtype=np.float64),
                joint_trajectory=plan.get("joint_trajectory"),
            )
            if not bool(execution.get("primitive_success", False)):
                metrics = {
                    **plan.get("metrics", {}),
                    **execution.get("metrics", {}),
                }
                metrics["elapsed_s"] = round(time.monotonic() - started, 3)
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=str(execution.get("stop_reason", "execution_failed")),
                    recoverable=bool(execution.get("recoverable", True)),
                    suggested_next_tool=execution.get("suggested_next_tool", "observe"),
                    metrics=metrics,
                    diagnostics=execution.get("diagnostics", {}),
                )
            reachability_target = np.asarray(
                plan.get("reachability_target_xyz", target),
                dtype=np.float64,
            ).reshape(3)
            reachability_quat = _quat_xyzw(plan.get("reachability_target_quat_xyzw"))
            reachable, reason, reach_metrics = self._check_arm_reachability(
                hand=hand,
                target_xyz=reachability_target,
                target_quat_xyzw=reachability_quat,
                timeout_s=self._remaining_s(deadline),
            )
            self._remaining_s(deadline)
            metrics = {
                **plan.get("metrics", {}),
                **execution["metrics"],
                **reach_metrics,
            }
            metrics["elapsed_s"] = round(time.monotonic() - started, 3)
            metrics["post_base_reachability_stage"] = reach_metrics.get(
                "reachability_stage"
            )
            metrics["post_base_reachability_target_xyz"] = reachability_target.tolist()
            success = bool(execution["primitive_success"] and reachable)
            return primitive_result(
                primitive_success=success,
                task_success=self._task_success(),
                stop_reason="arrived" if success else reason,
                recoverable=True,
                suggested_next_tool="observe",
                metrics=metrics,
                diagnostics=execution["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    @_planner_tool("move_to", suggested_next_tool="navigate_to")
    def move_to(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        target_quat_xyzw: Any | None = None,
        plan_only: bool = False,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = 0.087,
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            # Lock the assisted-grasp collision body exactly once. Reachability,
            # full planning, and execution must reason about the same object;
            # later reads are identity checks, never silent reference updates.
            attached_obj = _call_optional_arg(self.backend, "get_attached_object", hand)
            expected_attachment = attached_obj
            require_attachment = attached_obj is not None
            attached_collision_body_metrics = {
                "attached_collision_body": {
                    "available": require_attachment,
                    "identity_locked_at_call_start": require_attachment,
                    "used_for_reachability": require_attachment,
                    "used_for_full_trajectory_plan": require_attachment,
                    "required_during_execution": require_attachment,
                }
            }
            target = self._world_target(target_xyz, frame=frame)
            quat = _quat_xyzw(target_quat_xyzw)
            reachable, reason, reach_metrics = self._check_arm_reachability(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                timeout_s=self._remaining_s(deadline),
                attached_obj=attached_obj,
            )
            self._remaining_s(deadline)
            if not reachable:
                suggested = "navigate_to" if reason == "navigation_required" else None
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=reason,
                    recoverable=True,
                    suggested_next_tool=suggested,
                    metrics={**reach_metrics, **attached_collision_body_metrics},
                )
            plan = self.backend.plan_arm_trajectory(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                timeout_s=self._remaining_s(deadline),
                attached_obj=attached_obj,
            )
            if not plan.get("ok"):
                stop_reason = str(plan.get("stop_reason", "arm_plan_failed"))
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=stop_reason,
                    recoverable=stop_reason in {"unreachable", "planner_unavailable"},
                    suggested_next_tool="navigate_to"
                    if stop_reason == "unreachable"
                    else None,
                    metrics={
                        **reach_metrics,
                        **plan.get("metrics", {}),
                        **attached_collision_body_metrics,
                    },
                    diagnostics=plan,
                )
            if plan_only:
                return primitive_result(
                    primitive_success=True,
                    task_success=self._task_success(),
                    stop_reason="plan_ready",
                    recoverable=True,
                    metrics={
                        **reach_metrics,
                        **plan.get("metrics", {}),
                        **attached_collision_body_metrics,
                        "elapsed_s": round(time.monotonic() - started, 3),
                    },
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
                joint_trajectory=plan.get("joint_trajectory"),
                expected_attachment=expected_attachment,
                require_attachment=require_attachment,
            )
            metrics = {
                **reach_metrics,
                **plan.get("metrics", {}),
                **execution["metrics"],
                **attached_collision_body_metrics,
            }
            metrics["elapsed_s"] = round(time.monotonic() - started, 3)
            return primitive_result(
                primitive_success=execution["primitive_success"],
                task_success=self._task_success(),
                stop_reason=execution["stop_reason"],
                recoverable=execution["recoverable"],
                suggested_next_tool=execution["suggested_next_tool"],
                metrics=metrics,
                diagnostics=execution["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="navigate_to")

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
        allow_target_activation: bool = False,
        expected_attachment: Any = None,
        require_attachment: bool = False,
    ) -> dict[str, Any]:
        """Run a collision-checked intermediate move inside a composite tool."""

        started = time.monotonic()
        try:
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            target = _as_xyz(target_xyz)
            quat = _quat_xyzw(target_quat_xyzw)
            # Composite stages immediately request a full collision-checked
            # trajectory.  A separate IK reachability probe here duplicated
            # cuRobo work without adding a safety check; the full planner
            # already performs IK, trajopt, interpolation, and complete path
            # collision certification under the same hard deadline.
            reach_metrics: dict[str, Any] = {
                "reachability_stage": "full_trajectory_direct",
                "redundant_ik_probe_skipped": True,
            }
            plan = self.backend.plan_arm_trajectory(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=quat,
                timeout_s=self._remaining_s(deadline),
                attached_obj=(
                    expected_attachment
                    if require_attachment
                    else _call_optional_arg(self.backend, "get_attached_object", hand)
                ),
            )
            if not plan.get("ok"):
                stop_reason = str(plan.get("stop_reason", "arm_plan_failed"))
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=stop_reason,
                    recoverable=stop_reason in {"unreachable", "planner_unavailable"},
                    suggested_next_tool=(
                        "navigate_to" if stop_reason == "unreachable" else None
                    ),
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
                allow_expected_contact=False,
                allow_guarded_goal_world_collision=allow_target_activation,
                joint_trajectory=plan.get("joint_trajectory"),
                expected_attachment=expected_attachment,
                require_attachment=require_attachment,
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
                suggested_next_tool=execution["suggested_next_tool"],
                metrics=metrics,
                diagnostics=execution["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="navigate_to")

    @_planner_tool("pick", suggested_next_tool="observe")
    def pick(
        self,
        *,
        hand: str,
        target_xyz: Any,
        approach_vector: Any | None = None,
        grasp_quat_xyzw: Any | None = None,
        pregrasp_offset_m: float = 0.08,
        lift_m: float = 0.08,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            self._last_guarded_retreat_paths.pop(hand, None)
            target = _as_xyz(target_xyz)
            resolve_target_attachment = getattr(
                self.backend, "resolve_target_attachment", None
            )
            expected_attachment = (
                resolve_target_attachment(hand=hand, target_xyz=target)
                if callable(resolve_target_attachment)
                else None
            )
            if expected_attachment is None:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="target_attachment_unresolved",
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics={
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "target_root_identity_locked": False,
                    },
                )
            approach = _approach_vector(approach_vector)
            effective_grasp_quat = _quat_xyzw(grasp_quat_xyzw)
            if effective_grasp_quat is None:
                current_pose = self.backend.get_eef_pose(hand)
                if current_pose is None:
                    raise RuntimeError("cannot validate pick approach without EEF pose")
                effective_grasp_quat = _quat_xyzw(current_pose[1])
            assert effective_grasp_quat is not None
            finger_axis_world = _quat_rotate_vector_xyzw(
                effective_grasp_quat, [0.0, 0.0, 1.0]
            )
            axis_alignment = float(np.dot(finger_axis_world, approach))
            if axis_alignment < math.cos(math.radians(5.0)):
                raise ValueError(
                    "pick approach_vector must align with grasp EEF local +Z"
                )
            fingertip_offset_value = _call_optional_arg(
                self.backend, "get_eef_to_fingertip_length", hand
            )
            fingertip_offset_m = (
                0.0 if fingertip_offset_value is None else float(fingertip_offset_value)
            )
            if not np.isfinite(fingertip_offset_m) or fingertip_offset_m < 0:
                raise RuntimeError("invalid EEF-to-fingertip offset")
            ray_geometry = _call_optional_arg(
                self.backend,
                "get_assisted_grasp_outward_ray_geometry",
                hand,
            )
            if not isinstance(ray_geometry, dict) or not bool(
                ray_geometry.get("available", False)
            ):
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="assisted_grasp_ray_geometry_unavailable",
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics={
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "target_root_identity_locked": True,
                    },
                )
            guarded_overtravel_m = PICK_GUARDED_OVERTRAVEL_M
            start_ray_offset_m = float(
                ray_geometry.get("start_outward_offset_m", float("nan"))
            )
            end_ray_offset_m = float(
                ray_geometry.get("end_outward_offset_m", float("nan"))
            )
            assisted_grasp_ray_offset_m = max(
                start_ray_offset_m,
                end_ray_offset_m,
            )
            if (
                not np.isfinite(
                    [
                        assisted_grasp_ray_offset_m,
                        start_ray_offset_m,
                        end_ray_offset_m,
                    ]
                ).all()
                or assisted_grasp_ray_offset_m <= guarded_overtravel_m
            ):
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="assisted_grasp_ray_geometry_unavailable",
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics={
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "target_root_identity_locked": True,
                    },
                )
            # target_xyz is the public RGB-D surface point. Place OG's actual
            # outward (+Z) assisted-grasp ray plane one millimetre through that
            # surface. The fingertip tip is diagnostic only and does not define
            # the attachment ray's longitudinal plane.
            eef_to_contact_vector = finger_axis_world * (
                assisted_grasp_ray_offset_m - guarded_overtravel_m
            )
            grasp_eef_target = target - eef_to_contact_vector
            guarded_transition_m = float(pregrasp_offset_m)
            if not np.isfinite(guarded_transition_m) or guarded_transition_m <= 0:
                raise ValueError("pregrasp_offset_m must be finite and positive")
            pregrasp = grasp_eef_target - approach * guarded_transition_m
            move = self._move_to_composite_stage(
                hand=hand,
                target_xyz=pregrasp,
                target_quat_xyzw=effective_grasp_quat,
                position_tolerance_m=0.002,
                orientation_tolerance_rad=0.087,
                timeout_s=self._remaining_s(deadline),
                hold_steps_required=1,
            )
            if not move["primitive_success"]:
                move["suggested_next_tool"] = (
                    move.get("suggested_next_tool") or "move_to"
                )
                return move
            guarded = self._guarded_incremental_move(
                hand=hand,
                target_xyz=grasp_eef_target,
                target_quat_xyzw=effective_grasp_quat,
                direction=approach,
                allow_expected_contact=True,
                position_tolerance_m=0.015,
                timeout_s=self._remaining_s(deadline),
                terminal_hold_steps_required=1,
                contact_target_xyz=target,
                eef_to_contact_vector=eef_to_contact_vector,
            )
            if not guarded["primitive_success"]:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=guarded["stop_reason"],
                    recoverable=guarded["recoverable"],
                    suggested_next_tool="observe",
                    metrics={
                        **guarded["metrics"],
                        "pick_stages": {
                            "pregrasp_move": move.get("metrics", {}),
                        },
                    },
                    diagnostics=guarded["diagnostics"],
                )
            close = self._gripper_command(
                hand,
                opening=0.0,
                timeout_s=self._remaining_s(deadline),
                contact_target_xyz=target,
                allow_expected_contact=True,
                hold_steps_required=10,
                stop_on_attachment=True,
                eef_to_contact_vector=eef_to_contact_vector,
                expected_attachment=expected_attachment,
            )
            close.setdefault("metrics", {})["pick_stages"] = {
                "pregrasp_move": move.get("metrics", {}),
                "guarded_approach": guarded.get("metrics", {}),
            }
            if not close["primitive_success"]:
                return close
            attached_obj = _call_optional_arg(self.backend, "get_attached_object", hand)
            attachment_matches, attachment_identity = _attachment_identity_status(
                attached_obj,
                expected_attachment,
                hand=hand,
            )
            if not attachment_matches:
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason=(
                        "attachment_identity_mismatch"
                        if attached_obj is not None
                        else "grasp_not_confirmed"
                    ),
                    recoverable=True,
                    suggested_next_tool="observe",
                    metrics={
                        **close.get("metrics", {}),
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "attached_collision_body": {"available": False},
                        "attachment_identity": attachment_identity,
                    },
                    diagnostics=close.get("diagnostics", {}),
                )
            lift_target = grasp_eef_target + np.array(
                [0.0, 0.0, float(lift_m)], dtype=np.float64
            )
            reverse_path = self._last_guarded_retreat_paths.pop(hand, None)
            reverse_path_available = reverse_path is not None
            reverse_endpoint_matches_lift = bool(
                np.linalg.norm(pregrasp - lift_target) <= 0.01
            )
            if not reverse_endpoint_matches_lift:
                reverse_path = None
            certify_retreat = getattr(
                self.backend, "certify_attached_joint_trajectory", None
            )
            retreat_certification: dict[str, Any] | None = None
            if reverse_path is not None and callable(certify_retreat):
                retreat_certification = certify_retreat(
                    hand=hand,
                    joint_trajectory=reverse_path,
                    attached_obj=attached_obj,
                    timeout_s=min(self._remaining_s(deadline), 8.0),
                )
            if retreat_certification is not None and retreat_certification.get("ok"):
                current_pose = self.backend.get_eef_pose(hand)
                lift_quat = None if current_pose is None else current_pose[1]
                lift = self._execute_actions(
                    None,
                    hand=hand,
                    target_xyz=lift_target,
                    target_quat_xyzw=lift_quat,
                    position_tolerance_m=0.02,
                    orientation_tolerance_rad=0.087,
                    timeout_s=self._remaining_s(deadline),
                    require_pose=True,
                    hold_steps_required=10,
                    runtime_collision_interval_steps=1,
                    joint_trajectory=reverse_path,
                    expected_attachment=expected_attachment,
                    require_attachment=True,
                )
                lift.setdefault("metrics", {})["lift_execution"] = {
                    "method": "reverse_guarded_path",
                    "full_attached_path_rechecked": True,
                    "runtime_collision_interval_steps": 1,
                    "physical_contact_query_interval_steps": 1,
                    "certification": retreat_certification.get("metrics", {}),
                }
            else:
                lift = self._move_to_composite_stage(
                    hand=hand,
                    target_xyz=lift_target,
                    target_quat_xyzw=None,
                    position_tolerance_m=0.02,
                    orientation_tolerance_rad=0.087,
                    timeout_s=min(self._remaining_s(deadline), 30.0),
                    hold_steps_required=10,
                    expected_attachment=expected_attachment,
                    require_attachment=True,
                )
                lift.setdefault("metrics", {})["lift_execution"] = {
                    "method": "fresh_curobo_trajectory",
                    "reverse_guarded_path_available": reverse_path_available,
                    "reverse_endpoint_matches_lift": reverse_endpoint_matches_lift,
                    "reverse_guarded_path_certification": retreat_certification,
                }
            lift["stop_reason"] = (
                "picked" if lift["primitive_success"] else lift["stop_reason"]
            )
            lift["metrics"]["attached_collision_body"] = {
                "available": attached_obj is not None
            }
            lift["metrics"]["attachment_identity"] = attachment_identity
            lift["metrics"]["target_root_identity_locked"] = True
            lift["metrics"]["pick_stages"] = {
                "pregrasp_move": move.get("metrics", {}),
                "guarded_approach": guarded.get("metrics", {}),
                "gripper_close": close.get("metrics", {}),
            }
            lift["metrics"]["pick_contact_geometry"] = {
                "target_xyz": target.tolist(),
                "grasp_eef_target_xyz": grasp_eef_target.tolist(),
                "eef_to_fingertip_offset_m": fingertip_offset_m,
                "assisted_grasp_outward_ray_offset_m": (assisted_grasp_ray_offset_m),
                "assisted_grasp_ray_geometry": ray_geometry,
                "assisted_grasp_ray_endpoint_penetration_m": {
                    "start": guarded_overtravel_m
                    - (assisted_grasp_ray_offset_m - start_ray_offset_m),
                    "end": guarded_overtravel_m
                    - (assisted_grasp_ray_offset_m - end_ray_offset_m),
                },
                "maximum_ray_endpoint_penetration_m": guarded_overtravel_m,
                "outward_ray_to_fingertip_delta_m": (
                    assisted_grasp_ray_offset_m - fingertip_offset_m
                ),
                "guarded_overtravel_m": guarded_overtravel_m,
                "approach_vector": approach.tolist(),
                "finger_axis_world": finger_axis_world.tolist(),
                "axis_alignment_cosine": axis_alignment,
                "reverse_endpoint_matches_lift": reverse_endpoint_matches_lift,
            }
            return lift
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    @_planner_tool("rotate_wrist", suggested_next_tool="observe")
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

    @_planner_tool("press", suggested_next_tool="observe")
    def press(
        self,
        *,
        hand: str,
        target_xyz: Any,
        press_direction: Any | None = None,
        approach_distance_m: float = 0.04,
        press_depth_m: float = 0.012,
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
            contact = (
                target
                - direction * PRESS_EEF_TO_CONTACT_OFFSET_M
                + direction * float(press_depth_m)
            )
            guarded_transition_m = min(
                float(approach_distance_m) + float(press_depth_m),
                max(
                    0.012,
                    float(press_depth_m)
                    + CUROBO_COLLISION_ACTIVATION_DISTANCE_M
                    + 0.001,
                ),
            )
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
                allow_target_activation=True,
            )
            if not move["primitive_success"]:
                move.setdefault("metrics", {})["press_stages"] = {}
                return move
            # Configure the fingertip only after reaching the certified
            # pre-contact pose.  Closing at the previous tool's terminal pose
            # can itself enter the newly activated press target's collision
            # zone before guarded motion begins.
            close = self._gripper_command(
                hand,
                opening=0.0,
                timeout_s=min(self._remaining_s(deadline), 10.0),
            )
            if not close["primitive_success"]:
                close.setdefault("metrics", {})["press_stages"] = {
                    "precontact_move": move.get("metrics", {}),
                }
                return close
            guarded_press = self._guarded_incremental_move(
                hand=hand,
                target_xyz=contact,
                target_quat_xyzw=None,
                direction=direction,
                allow_expected_contact=True,
                position_tolerance_m=0.012,
                timeout_s=self._remaining_s(deadline),
                require_expected_contact=True,
                contact_target_xyz=target,
                stop_on_expected_contact=True,
                eef_to_contact_vector=direction * PRESS_EEF_TO_CONTACT_OFFSET_M,
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
                    **guarded_press["metrics"],
                    "press_stages": {
                        "gripper_close": close.get("metrics", {}),
                        "precontact_move": move.get("metrics", {}),
                    },
                },
                diagnostics=guarded_press["diagnostics"],
            )
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    @_planner_tool("release", suggested_next_tool="observe")
    def release(
        self,
        *,
        hand: str,
        opening: float = 1.0,
        retreat_vector: Any | None = None,
        retreat_m: float = 0.03,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        started = time.monotonic()
        try:
            deadline = started + self._validated_timeout(timeout_s)
            hand = _normalize_hand(hand)
            attached_before_release = _call_optional_arg(
                self.backend, "get_attached_object", hand
            )
            release_pose = self.backend.get_eef_pose(hand)
            release = self._gripper_command(
                hand,
                opening=float(opening),
                timeout_s=min(self._remaining_s(deadline), 15.0),
                contact_target_xyz=(
                    np.asarray(release_pose[0], dtype=np.float64)
                    if attached_before_release is not None and release_pose is not None
                    else None
                ),
                allow_expected_contact=attached_before_release is not None,
            )
            if not release["primitive_success"]:
                return release
            if (
                _call_optional_arg(self.backend, "get_attached_object", hand)
                is not None
            ):
                return primitive_result(
                    primitive_success=False,
                    task_success=self._task_success(),
                    stop_reason="release_not_confirmed",
                    recoverable=True,
                    suggested_next_tool="release",
                    metrics={
                        **release.get("metrics", {}),
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "attached_collision_body": {"available": True},
                    },
                    diagnostics=release.get("diagnostics", {}),
                )
            if retreat_vector is None or float(retreat_m) <= 0:
                release["stop_reason"] = "released"
                return release
            current = self.backend.get_eef_pose(hand)
            if current is None:
                return release
            direction = _approach_vector(retreat_vector)
            target = current[0] + direction * float(retreat_m)
            retreat = self.move_to(
                hand=hand,
                target_xyz=target,
                timeout_s=min(self._remaining_s(deadline), 20.0),
            )
            retreat["stop_reason"] = (
                "released" if retreat["primitive_success"] else retreat["stop_reason"]
            )
            return retreat
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="move_to")

    def _guarded_incremental_move(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: Any | None,
        direction: np.ndarray,
        allow_expected_contact: bool,
        position_tolerance_m: float,
        timeout_s: float,
        require_expected_contact: bool = False,
        contact_target_xyz: np.ndarray | None = None,
        terminal_hold_steps_required: int = 10,
        stop_on_expected_contact: bool = False,
        eef_to_contact_vector: np.ndarray | None = None,
        ignore_collision_checks: bool = False,
        allowed_contact_distance_m: float = 0.025,
    ) -> dict[str, Any]:
        started = time.monotonic()
        quat = _quat_xyzw(target_quat_xyzw)
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
        if not 0.025 <= allowed_contact_distance <= 0.05:
            raise ValueError("allowed_contact_distance_m must lie within [0.025, 0.05]")
        total = float(np.linalg.norm(target - start))
        guarded_path_planner = getattr(self.backend, "plan_guarded_ik_path", None)
        use_certified_path = callable(guarded_path_planner) and not bool(
            ignore_collision_checks
        )
        nominal_waypoint_distances = _guarded_waypoint_distances(total)
        nominal_steps = len(nominal_waypoint_distances)
        steps = 1 if use_certified_path else nominal_steps
        max_guarded_iterations = 1 if use_certified_path else max(steps * 4, steps + 8)
        guard_metrics = {
            "guarded_step_m": 0.002,
            "guarded_coarse_step_m": 0.002,
            "guarded_fine_distance_m": total,
            "guarded_total_distance_m": total,
            "guarded_waypoints": steps,
            "guarded_max_feedback_iterations": max_guarded_iterations,
            "guarded_execution_mode": (
                "single_batch_curobo_cartesian_fk_certified_path"
                if use_certified_path
                else "receding_horizon_cartesian_ik"
            ),
            "guarded_preexecution_collision_recheck": (
                "every_dense_joint_waypoint_world+self_target_excluded"
                if use_certified_path
                else "per_step_online"
            ),
            "guarded_runtime_collision_query_interval_steps": 1,
            "guarded_physical_contact_query_interval_steps": 1,
            "collision_checks_skipped": bool(ignore_collision_checks),
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
            if self._contact_is_abort(
                contact,
                hand=hand,
                target_xyz=contact_target,
                allow_expected_contact=allow_expected_contact,
                eef_to_contact_vector=eef_to_contact_vector,
            ):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="unexpected_contact",
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
            live_pose = self.backend.get_eef_pose(hand)
            live_target_error = (
                float(
                    np.linalg.norm(np.asarray(live_pose[0], dtype=np.float64) - target)
                )
                if live_pose is not None
                else float("inf")
            )
            terminal_tolerance_m = max(0.0015, min(float(position_tolerance_m), 0.003))
            terminal_hold = bool(live_target_error <= terminal_tolerance_m)
            if time.monotonic() - started > float(timeout_s) and not terminal_hold:
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
            if terminal_hold:
                hold = _call_optional_arg(self.backend, "hold_action", hand)
                if hold is None:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="hold_action_unavailable",
                        recoverable=True,
                        suggested_next_tool="observe",
                        executed=executed,
                        trace=trace,
                        final_pos_err=live_target_error,
                        final_ori_err=final_ori_err,
                        held_steps=0,
                        started=started,
                    )
                plan = {
                    "ok": True,
                    "actions": np.repeat(
                        np.asarray(hold, dtype=np.float32).reshape(1, ACTION_DIM),
                        10,
                        axis=0,
                    ),
                    "metrics": {"terminal_hold_without_replanning": True},
                }
                waypoint = target
            else:
                assert live_pose is not None
                live_position = np.asarray(live_pose[0], dtype=np.float64)
                remaining_vector = target - live_position
                remaining_distance = float(np.linalg.norm(remaining_vector))
                waypoint = (
                    target
                    if use_certified_path
                    else live_position
                    + remaining_vector * min(1.0, 0.002 / max(remaining_distance, 1e-9))
                )
                guarded_plan = (
                    guarded_path_planner
                    if use_certified_path
                    else getattr(
                        self.backend,
                        "plan_guarded_ik_step",
                        self.backend.plan_arm_trajectory,
                    )
                )
                guarded_plan_kwargs = {
                    "hand": hand,
                    "target_xyz": waypoint,
                    "target_quat_xyzw": quat,
                    "timeout_s": min(
                        (
                            30.0
                            if use_certified_path
                            else (15.0 if ignore_collision_checks else 2.0)
                        ),
                        max(
                            0.25,
                            float(timeout_s) - (time.monotonic() - started),
                        ),
                    ),
                    "attached_obj": _call_optional_arg(
                        self.backend, "get_attached_object", hand
                    ),
                }
                if "contact_target_xyz" in inspect.signature(guarded_plan).parameters:
                    guarded_plan_kwargs["contact_target_xyz"] = contact_target
                if "ignore_collision_checks" in inspect.signature(
                    guarded_plan
                ).parameters:
                    guarded_plan_kwargs["ignore_collision_checks"] = bool(
                        ignore_collision_checks
                    )
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
                retryable_guarded_failures = {
                    "unreachable",
                    "guarded_self_collision_path_unreachable",
                    "guarded_trajectory_collision",
                    "guarded_cartesian_path_invalid",
                }
                while (
                    use_certified_path
                    and len(guarded_plan_attempts) < 4
                    and not plan.get("ok")
                    and plan.get("stop_reason") in retryable_guarded_failures
                ):
                    retry_remaining_s = float(timeout_s) - (time.monotonic() - started)
                    if retry_remaining_s <= 1.0:
                        break
                    guarded_plan_kwargs["timeout_s"] = min(30.0, retry_remaining_s)
                    plan = guarded_plan(**guarded_plan_kwargs)
                    reverse_path = plan.get("reverse_joint_trajectory")
                    if reverse_path is not None:
                        self._last_guarded_retreat_paths[hand] = np.asarray(
                            _jsonable(reverse_path), dtype=np.float32
                        )
                    guard_metrics["guarded_plan_metrics"] = plan.get("metrics", {})
                    guarded_plan_attempts.append(
                        {
                            "attempt": len(guarded_plan_attempts) + 1,
                            "ok": bool(plan.get("ok", False)),
                            "stop_reason": plan.get("stop_reason"),
                        }
                    )
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
            hold_required = (
                max(1, int(terminal_hold_steps_required))
                if terminal_hold or use_certified_path
                else 5
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
                    min(
                        float(position_tolerance_m),
                        0.003 if terminal_hold or use_certified_path else 0.0015,
                    ),
                ),
                orientation_tolerance_rad=0.087,
                timeout_s=(
                    max(0.5, float(timeout_s) - (time.monotonic() - started))
                    if use_certified_path
                    else min(
                        16.0,
                        max(
                            0.5,
                            float(timeout_s) - (time.monotonic() - started),
                        ),
                    )
                ),
                require_pose=True,
                hold_steps_required=hold_required,
                contact_target_xyz=contact_target,
                allow_expected_contact=allow_expected_contact,
                allow_guarded_goal_world_collision=allow_expected_contact,
                eef_to_contact_vector=eef_to_contact_vector,
                stop_on_expected_contact=stop_on_expected_contact,
                runtime_collision_interval_steps=1,
                joint_trajectory=plan.get("joint_trajectory"),
                ignore_collision_checks=bool(ignore_collision_checks),
                allowed_contact_distance_m=allowed_contact_distance,
            )
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
            if terminal_hold or use_certified_path:
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
                held_steps=max(1, int(terminal_hold_steps_required)),
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
            held_steps=max(1, int(terminal_hold_steps_required)),
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
        allow_expected_contact: bool = False,
        hold_steps_required: int = 1,
        stop_on_attachment: bool = False,
        eef_to_contact_vector: np.ndarray | None = None,
        expected_attachment: Any = None,
    ) -> dict[str, Any]:
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
            allow_expected_contact=allow_expected_contact,
            allow_guarded_goal_world_collision=allow_expected_contact,
            hold_steps_required=max(1, int(hold_steps_required)),
            stop_on_attachment=stop_on_attachment,
            static_gripper_only=True,
            eef_to_contact_vector=eef_to_contact_vector,
            expected_attachment=expected_attachment,
            gripper_contact_settle_steps=(
                GRIPPER_CONTACT_SETTLE_STEPS if command < current_command else 0
            ),
        )
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
        allow_expected_contact: bool = False,
        allow_guarded_goal_world_collision: bool = False,
        stop_on_expected_contact: bool = False,
        stop_on_attachment: bool = False,
        static_gripper_only: bool = False,
        runtime_collision_interval_steps: int = 1,
        joint_trajectory: Any | None = None,
        eef_to_contact_vector: np.ndarray | None = None,
        expected_attachment: Any = None,
        require_attachment: bool = False,
        gripper_contact_settle_steps: int = 0,
        ignore_collision_checks: bool = False,
        allowed_contact_distance_m: float = 0.025,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if (actions is None) == (joint_trajectory is None):
            raise ValueError("provide exactly one of actions or joint_trajectory")
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
        action_chunk = validate_action_chunk(actions) if actions is not None else None
        q_chunk = (
            np.asarray(_jsonable(joint_trajectory), dtype=np.float32)
            if joint_trajectory is not None
            else None
        )
        if q_chunk is not None and (q_chunk.ndim != 2 or q_chunk.shape[0] < 1):
            raise ValueError(f"joint trajectory must be [T,D], got {q_chunk.shape}")
        planned_steps = (
            action_chunk.shape[0] if action_chunk is not None else q_chunk.shape[0]
        )
        trajectory_hold_reference: np.ndarray | None = None
        trajectory_fixed_reference: dict[str, Any] | None = None
        trajectory_uses_fixed_reference = False
        locked_joint_reference: dict[str, Any] | None = None
        trajectory_hand = None if base_goal_xyyaw is not None else hand
        if q_chunk is not None:
            trajectory_fixed_reference = _call_optional_kw(
                self.backend,
                "capture_trajectory_hold_reference",
                hand=trajectory_hand,
            )
            if isinstance(trajectory_fixed_reference, dict):
                locked_joint_reference = trajectory_fixed_reference.get(
                    "locked_joint_reference"
                )
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
                locked_joint_reference = _call_optional_kw(
                    self.backend,
                    "capture_locked_joint_reference",
                    hand=trajectory_hand,
                )
            if (
                not isinstance(locked_joint_reference, dict)
                or (
                    trajectory_fixed_reference is not None
                    and not trajectory_uses_fixed_reference
                )
                or (
                    trajectory_fixed_reference is None
                    and trajectory_hold_reference is None
                )
            ):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="locked_joint_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=0,
                    trace=[],
                    final_pos_err=None,
                    final_ori_err=None,
                    held_steps=0,
                    started=started,
                    extra_metrics={
                        "trajectory_hold_reference_available": (
                            trajectory_fixed_reference is not None
                            or trajectory_hold_reference is not None
                        ),
                        "q_space_hold_reference": (
                            trajectory_fixed_reference is not None
                        ),
                        "online_fixed_reference_conversion": (
                            trajectory_uses_fixed_reference
                        ),
                        "locked_joint_reference_available": isinstance(
                            locked_joint_reference, dict
                        ),
                    },
                )
        runtime_collision_interval_steps = max(1, int(runtime_collision_interval_steps))
        dynamics = (
            _action_dynamics_report(action_chunk)
            if action_chunk is not None
            else {
                "ok": True,
                "mode": "online_controller_commands",
                "max_velocity_command_delta": 0.0,
                "max_acceleration_command_delta": 0.0,
                "velocity_limit": 5.0,
                "acceleration_limit": 10.0,
            }
        )
        if not dynamics["ok"]:
            return self._execution_result(
                primitive_success=False,
                stop_reason=str(dynamics["stop_reason"]),
                recoverable=True,
                suggested_next_tool="move_to",
                executed=0,
                trace=[],
                final_pos_err=None,
                final_ori_err=None,
                held_steps=0,
                started=started,
                extra_metrics={"dynamics": dynamics},
            )
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
        previous_delta: np.ndarray | None = None
        contact_stop_active = False
        attachment_seen = False
        attachment_confirmation_steps = 0
        attachment_endpoint_held_steps = 0
        attachment_identity: dict[str, Any] | None = None
        locked_joint_report: dict[str, Any] | None = None
        locked_joint_peaks: dict[str, float | None] = {
            "base_xy_drift_m": None,
            "base_z_drift_m": None,
            "base_rpy_drift_rad": None,
            "articulation_drift_rad": None,
        }
        locked_gripper_command_report: dict[str, Any] | None = None
        gripper_contact_settle_remaining = 0
        gripper_contact_settle_executed = 0
        gripper_contact_settle_started = False
        gripper_contact_hold_action: np.ndarray | None = None
        static_guarded_collision_cache: tuple[bool, dict[str, Any]] | None = None
        index = 0
        max_steps = (
            planned_steps * (self.max_stall_steps + 1)
            + int(hold_steps_required)
            + self.max_stall_steps
        )
        while executed < max_steps:
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
                )
            if trajectory_uses_fixed_reference:
                locked_gripper_command_report = _call_optional_kw(
                    self.backend,
                    "locked_gripper_command_report",
                    action=action,
                    reference=trajectory_fixed_reference,
                )
                if not isinstance(locked_gripper_command_report, dict) or not bool(
                    locked_gripper_command_report.get("available", False)
                ):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="locked_gripper_command_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "locked_gripper_command_report": (
                                locked_gripper_command_report
                            )
                        },
                    )
                if locked_gripper_command_report.get("ok") is False:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="locked_gripper_command_drift",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "locked_gripper_command_report": (
                                locked_gripper_command_report
                            )
                        },
                    )
            if static_gripper_only:
                latch = getattr(self.env, "_gripper_latch", None)
                if isinstance(latch, dict):
                    gripper_segment = ENV_ACTION_SEGMENTS[f"{hand}_gripper"]
                    latch[hand] = float(action[gripper_segment][0])
            if previous_action is not None:
                delta = action - previous_action
                max_velocity = float(np.max(np.abs(delta)))
                dynamics["max_velocity_command_delta"] = max(
                    float(dynamics["max_velocity_command_delta"]),
                    max_velocity,
                )
                if previous_delta is not None:
                    max_acceleration = float(np.max(np.abs(delta - previous_delta)))
                    dynamics["max_acceleration_command_delta"] = max(
                        float(dynamics["max_acceleration_command_delta"]),
                        max_acceleration,
                    )
                    if max_acceleration > float(dynamics["acceleration_limit"]):
                        return self._execution_result(
                            primitive_success=False,
                            stop_reason="acceleration_limit",
                            recoverable=True,
                            suggested_next_tool="move_to",
                            executed=executed,
                            trace=trace,
                            final_pos_err=final_pos_err,
                            final_ori_err=final_ori_err,
                            held_steps=held_steps,
                            started=started,
                            extra_metrics={"dynamics": dynamics},
                        )
                previous_delta = delta
                if max_velocity > float(dynamics["velocity_limit"]):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="velocity_limit",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"dynamics": dynamics},
                    )
            previous_action = action.copy()
            joint_report = self._joint_margin_report()
            if not bool(joint_report.get("available", False)):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="joint_limit_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"joint_margin": joint_report},
                )
            if joint_report.get("ok") is False:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="joint_limit_margin",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"joint_margin": joint_report},
                )
            collision_report = (
                {
                    "available": True,
                    "colliding": False,
                    "collision_checks_skipped": True,
                    "authorization": "explicit_stage3_direct_press",
                }
                if ignore_collision_checks
                else self._collision_report()
            )
            if not bool(collision_report.get("available", False)):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="collision_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"collision_report": collision_report},
                )
            collision_margin = collision_report.get("min_margin_m")
            collision_detected = bool(collision_report.get("colliding", False)) or (
                collision_margin is not None and float(collision_margin) < 0.0
            )
            if static_gripper_only and static_guarded_collision_cache is not None:
                guarded_collision_allowed, guarded_collision_contact = (
                    static_guarded_collision_cache
                )
            else:
                guarded_collision_allowed, guarded_collision_contact = (
                    self._guarded_target_collision_allowed(
                        hand=hand,
                        target_xyz=contact_target_xyz,
                        enabled=(
                            collision_detected and allow_guarded_goal_world_collision
                        ),
                        eef_to_contact_vector=eef_to_contact_vector,
                    )
                )
                if static_gripper_only and guarded_collision_allowed:
                    static_guarded_collision_cache = (
                        guarded_collision_allowed,
                        guarded_collision_contact,
                    )
            if collision_detected and not guarded_collision_allowed:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="unexpected_collision",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "collision_report": collision_report,
                        "guarded_collision_contact": guarded_collision_contact,
                    },
                )
            self._step_env_action(action)
            executed += 1
            # Validate the state produced by this command. A pre-step margin
            # cannot authorize success after a waypoint enters the soft-limit
            # guard band.
            joint_report = self._joint_margin_report()
            if not bool(joint_report.get("available", False)):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="joint_limit_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"joint_margin": joint_report},
                )
            if joint_report.get("ok") is False:
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="joint_limit_margin",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"joint_margin": joint_report},
                )
            if locked_joint_reference is not None:
                locked_joint_report = _call_optional_kw(
                    self.backend,
                    "locked_joint_drift_report",
                    reference=locked_joint_reference,
                )
                if not isinstance(locked_joint_report, dict) or not bool(
                    locked_joint_report.get("available", False)
                ):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="locked_joint_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "locked_joint_report": locked_joint_report,
                            "locked_joint_peaks": locked_joint_peaks,
                        },
                    )
                for field in locked_joint_peaks:
                    value = locked_joint_report.get(field)
                    if value is not None:
                        previous_peak = locked_joint_peaks[field]
                        locked_joint_peaks[field] = (
                            float(value)
                            if previous_peak is None
                            else max(previous_peak, float(value))
                        )
                if locked_joint_report.get("ok") is False:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="locked_joint_drift",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={
                            "locked_joint_report": locked_joint_report,
                            "locked_joint_peaks": locked_joint_peaks,
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
            # Validate the state produced by this waypoint, not only the state
            # that preceded it. Force a new world/self-collision query so the
            # final waypoint cannot succeed on a cached pre-action report.
            collision_report = (
                {
                    "available": True,
                    "colliding": False,
                    "collision_checks_skipped": True,
                    "authorization": "explicit_stage3_direct_press",
                }
                if ignore_collision_checks
                else self._collision_report(
                    force=(
                        q_chunk is not None
                        or (
                            not static_gripper_only
                            and runtime_collision_interval_steps == 1
                        )
                        or executed == 1
                        or (
                            not static_gripper_only
                            and executed % runtime_collision_interval_steps == 0
                        )
                        or (q_chunk is not None and index >= planned_steps - 1)
                        or (static_gripper_only and executed == planned_steps)
                    )
                )
            )
            if not bool(collision_report.get("available", False)):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="collision_feedback_unavailable",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={"collision_report": collision_report},
                )
            if static_gripper_only and static_guarded_collision_cache is not None:
                guarded_collision_allowed, guarded_collision_contact = (
                    static_guarded_collision_cache
                )
            else:
                guarded_collision_allowed, guarded_collision_contact = (
                    self._guarded_target_collision_allowed(
                        hand=hand,
                        target_xyz=contact_target_xyz,
                        enabled=(
                            bool(collision_report.get("colliding", False))
                            and allow_guarded_goal_world_collision
                        ),
                        eef_to_contact_vector=eef_to_contact_vector,
                    )
                )
                if static_gripper_only and guarded_collision_allowed:
                    static_guarded_collision_cache = (
                        guarded_collision_allowed,
                        guarded_collision_contact,
                    )
            if (
                bool(collision_report.get("colliding", False))
                and not guarded_collision_allowed
            ):
                return self._execution_result(
                    primitive_success=False,
                    stop_reason="unexpected_collision",
                    recoverable=True,
                    suggested_next_tool="move_to",
                    executed=executed,
                    trace=trace,
                    final_pos_err=final_pos_err,
                    final_ori_err=final_ori_err,
                    held_steps=held_steps,
                    started=started,
                    extra_metrics={
                        "collision_report": collision_report,
                        "guarded_collision_contact": guarded_collision_contact,
                    },
                )
            actual_dynamics = self._actual_dynamics_report()
            if actual_dynamics is not None:
                dynamics["actual"] = actual_dynamics
                dynamics["actual_peak"] = _accumulate_actual_dynamics_peak(
                    dynamics.get("actual_peak"),
                    actual_dynamics,
                )
                if not bool(actual_dynamics.get("available", False)):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="dynamics_feedback_unavailable",
                        recoverable=True,
                        suggested_next_tool="move_to",
                        executed=executed,
                        trace=trace,
                        final_pos_err=final_pos_err,
                        final_ori_err=final_ori_err,
                        held_steps=held_steps,
                        started=started,
                        extra_metrics={"dynamics": dynamics},
                    )
                if actual_dynamics.get("ok") is False:
                    terminal_guarded_oscillation = bool(
                        q_chunk is not None
                        and index >= planned_steps
                        and contact_target_xyz is not None
                    )
                    gripper_contact_impact = bool(
                        static_gripper_only and contact_target_xyz is not None
                    )
                    terminal_contact = (
                        self._contact_report(
                            hand=hand,
                            target_xyz=contact_target_xyz,
                            allowed_contact_distance_m=max(
                                float(allowed_contact_distance_m),
                                float(position_tolerance_m) * 2.0,
                            ),
                        )
                        if terminal_guarded_oscillation or gripper_contact_impact
                        else None
                    )
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason=(
                            "guarded_terminal_oscillation"
                            if terminal_guarded_oscillation
                            else (
                                "gripper_contact_dynamics_limit"
                                if gripper_contact_impact
                                else "actual_dynamics_limit"
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
                            "dynamics": dynamics,
                            "terminal_hold_actual_dynamics_violation": (
                                terminal_guarded_oscillation
                            ),
                            "gripper_contact_actual_dynamics_violation": (
                                gripper_contact_impact
                            ),
                            "terminal_contact_report": terminal_contact,
                            "actual_acceleration_limit_rad_s2": 15.0,
                        },
                    )
            waypoint_tracking: dict[str, Any] | None = None
            waypoint_advanced = False
            if q_chunk is not None and index < planned_steps:
                waypoint_tracking = _call_optional_kw(
                    self.backend,
                    "joint_tracking_report",
                    target_q=q_chunk[index],
                    hand=None if base_goal_xyyaw is not None else hand,
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
                if bool(waypoint_tracking.get("reached", False)):
                    index += 1
                    waypoint_advanced = True
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
                base_pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
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
                    pos, quat = pose
                    if target_xyz is not None:
                        final_pos_err = float(np.linalg.norm(pos - target_xyz))
                    if target_quat_xyzw is not None:
                        final_ori_err = _quat_angle_error_rad(quat, target_quat_xyzw)
            if contact_target_xyz is not None:
                contact = self._contact_report(
                    hand=hand,
                    target_xyz=contact_target_xyz,
                    allowed_contact_distance_m=max(
                        float(allowed_contact_distance_m),
                        float(position_tolerance_m) * 2.0,
                    ),
                )
                if not bool(contact.get("available", False)):
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
                if self._contact_is_abort(
                    contact,
                    hand=hand,
                    target_xyz=contact_target_xyz,
                    allow_expected_contact=allow_expected_contact,
                    eef_to_contact_vector=eef_to_contact_vector,
                ):
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="unexpected_contact",
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
                if contact_stop_active and not expected_contact:
                    return self._execution_result(
                        primitive_success=False,
                        stop_reason="expected_contact_lost",
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
                if stop_on_expected_contact and expected_contact:
                    contact_stop_active = True
                    index = planned_steps
                if (
                    stop_on_attachment
                    and int(gripper_contact_settle_steps) > 0
                    and not gripper_contact_settle_started
                    and bool(contact.get("target_two_finger_contact", False))
                ):
                    # Hold this exact two-finger+raycast candidate for at least
                    # ten 60 Hz steps while normal per-step safety checks continue.
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
                    "joint_margin": joint_report,
                    "collision_report": collision_report,
                    "guarded_collision_contact": guarded_collision_contact,
                    "actual_dynamics": actual_dynamics,
                    "joint_tracking": waypoint_tracking,
                    "contact_stop_active": contact_stop_active,
                    "attachment_confirmed": attachment_confirmed,
                    "attachment_confirmation_steps": attachment_confirmation_steps,
                    "attachment_identity": attachment_identity,
                    "contact_report": contact
                    if contact_target_xyz is not None
                    else None,
                    "gripper_contact_settle_started": (gripper_contact_settle_started),
                    "gripper_contact_settle_remaining": (
                        gripper_contact_settle_remaining
                    ),
                    "gripper_command": float(
                        action[ENV_ACTION_SEGMENTS[f"{hand}_gripper"]][0]
                    ),
                    "locked_joint_report": locked_joint_report,
                    "locked_joint_peaks": locked_joint_peaks,
                    "locked_gripper_command_report": (locked_gripper_command_report),
                    "hold_step": index >= planned_steps,
                }
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
                            "locked_joint_report": locked_joint_report,
                            "locked_joint_peaks": locked_joint_peaks,
                            "locked_gripper_command_report": (
                                locked_gripper_command_report
                            ),
                        },
                    )
                # A gripper close succeeds only after the same target root has
                # remained attached for the complete confirmation window.
                continue
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
                            "dynamics": dynamics,
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
                            "locked_joint_report": locked_joint_report,
                            "locked_joint_peaks": locked_joint_peaks,
                            "locked_gripper_command_report": (
                                locked_gripper_command_report
                            ),
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
                    extra_metrics={"dynamics": dynamics},
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
                "dynamics": dynamics,
                "attachment_confirmation_steps": attachment_confirmation_steps,
                "attachment_endpoint_held_steps": attachment_endpoint_held_steps,
                "attachment_identity": attachment_identity,
                "gripper_contact_settle_started": gripper_contact_settle_started,
                "gripper_contact_settle_steps_executed": (
                    gripper_contact_settle_executed
                ),
                "locked_joint_report": locked_joint_report,
                "locked_joint_peaks": locked_joint_peaks,
                "locked_gripper_command_report": locked_gripper_command_report,
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
        joint_report = self._joint_margin_report()
        collision_report = self._collision_report()
        metrics = {
            "executed_waypoints": int(executed),
            "final_position_error_m": final_pos_err,
            "final_orientation_error_rad": final_ori_err,
            "held_steps": int(held_steps),
            "elapsed_s": round(time.monotonic() - started, 3)
            if started is not None
            else None,
            "joint_margin": joint_report,
            "collision_report": collision_report,
            "collision_margin": collision_report.get("min_margin_m"),
        }
        metrics.update(extra_metrics or {})
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

    def _joint_margin_report(self) -> dict[str, Any]:
        report = _call_optional(self.backend, "joint_margin_report")
        if isinstance(report, dict):
            return report
        margin = _call_optional(self.backend, "joint_margin")
        if margin is None:
            return {
                "available": False,
                "reason": "joint_margin_unavailable",
                "min_normalized_margin": None,
                "threshold_normalized": 0.03,
                "threshold_raw_rad": 0.05,
                "ok": None,
            }
        return {
            "available": True,
            "min_normalized_margin": float(margin),
            "threshold_normalized": 0.03,
            "threshold_raw_rad": 0.05,
            "ok": bool(float(margin) >= 0.03),
        }

    def _collision_report(self, *, force: bool = False) -> dict[str, Any]:
        fn = getattr(self.backend, "collision_report", None)
        report = None
        if callable(fn):
            try:
                parameters = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                parameters = {}
            try:
                report = fn(force=force) if "force" in parameters else fn()
            except TimeoutError:
                raise
            except Exception:
                report = None
        if isinstance(report, dict):
            return report
        margin = _call_optional(self.backend, "collision_margin")
        return {
            "available": margin is not None,
            "reason": None if margin is not None else "collision_margin_unavailable",
            "colliding": bool(margin is not None and float(margin) < 0.0),
            "min_margin_m": float(margin) if margin is not None else None,
            "margin_available": margin is not None,
        }

    def _actual_dynamics_report(self) -> dict[str, Any] | None:
        fn = getattr(self.backend, "dynamics_report", None)
        if not callable(fn):
            return None
        try:
            report = fn()
        except Exception as exc:
            return {
                "available": False,
                "ok": None,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        return (
            report
            if isinstance(report, dict)
            else {
                "available": False,
                "ok": None,
                "reason": "dynamics_report returned a non-dict value",
            }
        )

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

    def _contact_is_abort(
        self,
        contact: dict[str, Any],
        *,
        hand: str,
        target_xyz: np.ndarray,
        allow_expected_contact: bool,
        eef_to_contact_vector: np.ndarray | None = None,
    ) -> bool:
        if bool(contact.get("unexpected_contact", False)):
            return True
        if not bool(contact.get("expected_contact", False)):
            return False
        if not allow_expected_contact:
            return True
        current = (
            self.backend.get_eef_pose(hand)
            if hasattr(self.backend, "get_eef_pose")
            else None
        )
        if current is None:
            return True
        contact_point = np.asarray(current[0], dtype=np.float64)
        if eef_to_contact_vector is not None:
            contact_point = contact_point + np.asarray(
                eef_to_contact_vector, dtype=np.float64
            ).reshape(3)
        distance = float(np.linalg.norm(contact_point - target_xyz))
        allowed = contact.get("allowed_contact_distance_m", 0.025)
        try:
            allowed = float(allowed)
        except Exception:
            allowed = 0.025
        return distance > max(allowed, GUARDED_TARGET_NEIGHBORHOOD_M)

    def _guarded_target_collision_allowed(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray | None,
        enabled: bool,
        eef_to_contact_vector: np.ndarray | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Allow only a guarded-step world activation near its target point."""

        if not enabled or target_xyz is None:
            return False, None
        current = self.backend.get_eef_pose(hand)
        if current is None:
            return False, {"available": False, "reason": "eef_pose_unavailable"}
        allowed_distance = GUARDED_TARGET_NEIGHBORHOOD_M
        contact_point = np.asarray(current[0], dtype=np.float64)
        if eef_to_contact_vector is not None:
            contact_point = contact_point + np.asarray(
                eef_to_contact_vector, dtype=np.float64
            ).reshape(3)
        distance = float(
            np.linalg.norm(contact_point - np.asarray(target_xyz, dtype=np.float64))
        )
        safety = _call_optional_kw(
            self.backend,
            "guarded_contact_safety_report",
            hand=hand,
            target_xyz=np.asarray(target_xyz, dtype=np.float64),
            allowed_contact_distance_m=allowed_distance,
        )
        if not isinstance(safety, dict):
            safety = {
                "available": False,
                "reason": "guarded_contact_safety_report_unavailable",
            }
        contact = safety.get("contact_report")
        if not isinstance(contact, dict):
            contact = {
                "available": False,
                "reason": "target_identity_contact_report_unavailable",
                "unexpected_contact": False,
                "expected_contact": False,
            }
        contact_count = int(contact.get("contact_count", 0) or 0)
        target_contact_safe = bool(
            contact.get("expected_contact", False)
            and not contact.get("unexpected_contact", False)
        )
        activation_without_contact_safe = bool(
            contact_count == 0 and safety.get("target_only_activation_verified", False)
        )
        allowed = bool(
            safety.get("available", False)
            and safety.get("target_object_resolved", False)
            and not safety.get("self_colliding", True)
            and not safety.get("world_without_target_colliding", True)
            and not contact.get("unexpected_contact", False)
            and (target_contact_safe or activation_without_contact_safe)
            and distance <= allowed_distance
            and not self._contact_is_abort(
                contact,
                hand=hand,
                target_xyz=np.asarray(target_xyz, dtype=np.float64),
                allow_expected_contact=True,
                eef_to_contact_vector=eef_to_contact_vector,
            )
        )
        return allowed, {
            "safety_report": safety,
            "contact_report": contact,
            "eef_target_distance_m": distance,
            "eef_contact_target_distance_m": distance,
            "guarded_target_neighborhood_m": allowed_distance,
            "world_collision_exception_allowed": allowed,
            "exception_scope": (
                "current_2mm_guarded_step_target_body_only;self_never_exempt"
            ),
            "target_contact_identity_confirmed": target_contact_safe,
            "target_activation_without_physical_contact": (
                activation_without_contact_safe
            ),
        }

    def _step_env_action(self, action: np.ndarray) -> None:
        step = getattr(self.env, "planner_step", None)
        if not callable(step):
            step = self.env.chunk_step
        ret = step(np.asarray(action, dtype=np.float32).reshape(1, ACTION_DIM))
        if isinstance(ret, tuple) and len(ret) >= 5:
            self.last_info = ret[4]

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


def _accumulate_actual_dynamics_peak(
    previous: Any,
    current: dict[str, Any],
) -> dict[str, Any]:
    """Retain episode-local observed dynamics maxima and their source joints."""
    peak = dict(previous) if isinstance(previous, dict) else {}
    peak["available"] = bool(current.get("available", False))
    peak["ok"] = bool(peak.get("ok", True) and current.get("ok", False))
    peak["samples"] = int(peak.get("samples", 0)) + 1
    peak["source"] = current.get("source")
    for value_key, joint_key in (
        ("max_actual_velocity", "max_actual_velocity_joint"),
        ("max_actual_acceleration", "max_actual_acceleration_joint"),
    ):
        value = current.get(value_key)
        if value is None:
            continue
        old = peak.get(value_key)
        if old is None or float(value) > float(old):
            peak[value_key] = float(value)
            peak[joint_key] = current.get(joint_key)
    for value_key in ("max_velocity_ratio", "max_acceleration_ratio"):
        value = current.get(value_key)
        if value is not None and (
            peak.get(value_key) is None or float(value) > float(peak[value_key])
        ):
            peak[value_key] = float(value)
    for key in (
        "max_velocity_limit",
        "max_acceleration_limit",
        "sample_dt_s",
    ):
        if current.get(key) is not None:
            peak[key] = current[key]
    return peak


def _action_dynamics_report(actions: np.ndarray) -> dict[str, Any]:
    chunk = validate_action_chunk(actions)
    if chunk.shape[0] < 2:
        return {
            "ok": True,
            "max_velocity_command_delta": 0.0,
            "max_acceleration_command_delta": 0.0,
            "velocity_limit": 5.0,
            "acceleration_limit": 10.0,
        }
    velocity = np.diff(chunk, axis=0)
    acceleration = (
        np.diff(velocity, axis=0)
        if velocity.shape[0] >= 2
        else np.zeros((0, ACTION_DIM))
    )
    max_velocity = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    max_acceleration = float(np.max(np.abs(acceleration))) if acceleration.size else 0.0
    velocity_limit = 5.0
    acceleration_limit = 10.0
    if max_velocity > velocity_limit:
        return {
            "ok": False,
            "stop_reason": "velocity_limit",
            "max_velocity_command_delta": max_velocity,
            "max_acceleration_command_delta": max_acceleration,
            "velocity_limit": velocity_limit,
            "acceleration_limit": acceleration_limit,
        }
    if max_acceleration > acceleration_limit:
        return {
            "ok": False,
            "stop_reason": "acceleration_limit",
            "max_velocity_command_delta": max_velocity,
            "max_acceleration_command_delta": max_acceleration,
            "velocity_limit": velocity_limit,
            "acceleration_limit": acceleration_limit,
        }
    return {
        "ok": True,
        "max_velocity_command_delta": max_velocity,
        "max_acceleration_command_delta": max_acceleration,
        "velocity_limit": velocity_limit,
        "acceleration_limit": acceleration_limit,
    }


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


def _retime_joint_trajectory(
    trajectory: Any,
    *,
    sample_dt_s: float,
    max_command_velocity: float,
    max_command_acceleration: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Time-stretch a certified joint path without leaving its line segments.

    cuRobo paths can contain a repeated waypoint at a stitching boundary.  At
    the 60 Hz controller that can command an abrupt stop close to the official
    acceleration limit, leaving no margin for tracking dynamics.  Linear
    interpolation over a longer duration preserves the geometric path; callers
    must still collision-check the returned samples in full.
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
        "full_collision_recheck_required": True,
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
