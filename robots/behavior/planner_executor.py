"""Environment-side BEHAVIOR planner primitives backed by RGB-D and cuRobo."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
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
    validated_rigid_transform,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    BASE_ROTATION_STEP_RAD,
    BASE_TRANSLATION_STEP_M,
    DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT,
    DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    DASHBOARD_EXECUTION_MAX_WAYPOINTS,
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
WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG = "dashboard_jog"
DASHBOARD_PREPARED_BASE_PLANNING_PROFILE = "dashboard_base_fast"
DASHBOARD_PREPARED_TORSO_PLANNING_PROFILE = "dashboard_torso_fast"
PREPARED_DASHBOARD_EEF_EXECUTION_POLICY = "prepared_dashboard_eef_goal_only_v1"
PREPARED_DASHBOARD_BASE_EXECUTION_POLICY = "prepared_dashboard_base_goal_only_v1"
PREPARED_DASHBOARD_TORSO_EXECUTION_POLICY = "prepared_dashboard_torso_goal_only_v1"
TORSO_LINK_NAME = "torso_link4"
TORSO_ACTIVE_JOINT_NAMES = (
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
)
TORSO_LOCKED_JOINT_NAME = "torso_joint4"
PRESS_EEF_TO_CONTACT_OFFSET_M = 0.026
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
WHOLE_BODY_LOCKED_JOINT_NAMES = (
    "base_footprint_z_joint",
    "base_footprint_rx_joint",
    "base_footprint_ry_joint",
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
)


class CuroboPlanningError(RuntimeError):
    """Structured CuRobo planning failure propagated across controller layers."""

    def __init__(self, stop_reason: str, error: str | None = None) -> None:
        self.stop_reason = str(stop_reason)
        self.error = None if error is None else str(error)
        super().__init__(self.error or self.stop_reason)


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


def official_task_success(info: Any) -> bool:
    """Read only BEHAVIOR's official success bit."""
    if not isinstance(info, dict):
        return False
    done = info.get("done")
    if not isinstance(done, dict):
        return False
    value = done.get("success", False)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _terminal_step_outcome(
    receipt: dict[str, Any],
) -> tuple[bool, str] | None:
    """Return the highest-priority terminal outcome from one executed env step."""

    if receipt.get("raw_success") is True:
        return True, "official_task_success"
    if receipt.get("terminated") is True:
        return False, "environment_terminated"
    if receipt.get("truncated") is True:
        return False, "environment_truncated"
    return None


def _deterministic_execution_waypoints(
    trajectory: Any,
    *,
    max_waypoints: int = DASHBOARD_EXECUTION_MAX_WAYPOINTS,
) -> tuple[np.ndarray, list[int]]:
    """Select at most ``max_waypoints`` command rows and always keep the end."""

    rows = np.asarray(_jsonable(trajectory), dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] < 1:
        raise ValueError(f"command trajectory must be [T,D], got {rows.shape}")
    limit = int(max_waypoints)
    if limit < 1:
        raise ValueError("execution waypoint limit must be positive")
    if len(rows) <= limit:
        indices = np.arange(len(rows), dtype=np.int64)
    else:
        indices = np.rint(
            np.linspace(0, len(rows) - 1, num=limit, dtype=np.float64)
        ).astype(np.int64)
        indices[-1] = len(rows) - 1
        indices = np.unique(indices)
    return np.ascontiguousarray(rows[indices], dtype=np.float32), [
        int(index) for index in indices
    ]


def _curobo_status_labels(value: Any) -> list[str]:
    """Flatten CuRobo status values without depending on one installed version."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        labels: list[str] = []
        for item in value:
            labels.extend(_curobo_status_labels(item))
        return labels
    name = getattr(value, "name", None)
    raw_value = getattr(value, "value", None)
    if isinstance(name, str) and name:
        return [name]
    if isinstance(raw_value, str) and raw_value:
        return [raw_value]
    rendered = str(value).strip()
    return [rendered] if rendered else []


def _curobo_failure_from_status(
    statuses: Any,
    *,
    elapsed_s: float,
) -> tuple[str, str, list[str]]:
    """Preserve CuRobo's joint-limit, convergence, timeout, and solver classes."""

    labels = _curobo_status_labels(statuses)
    normalized = " ".join(labels).upper()
    if "JOINT_LIMIT" in normalized or "JOINT LIMIT" in normalized:
        stop_reason = "joint_limits"
    elif "TIMEOUT" in normalized or "TIMED OUT" in normalized:
        stop_reason = "timeout"
    elif any(
        token in normalized
        for token in (
            "IK_FAIL",
            "IK FAIL",
            "TRAJOPT_FAIL",
            "TRAJOPT FAIL",
            "FINETUNE_TRAJOPT_FAIL",
            "FINETUNE TRAJOPT FAIL",
        )
    ):
        stop_reason = "unreachable"
    elif not labels and elapsed_s >= DASHBOARD_CUROBO_PLAN_TIMEOUT_S:
        stop_reason = "timeout"
    else:
        stop_reason = "solver_error"
    detail = ", ".join(labels) if labels else "no status"
    return (
        stop_reason,
        f"CuRobo planning failed ({detail})",
        labels,
    )


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
    return {
        "primitive_success": bool(primitive_success),
        "task_success": bool(task_success),
        "_finish": bool(task_success),
        "official_success_source": 'info["done"]["success"]',
        "stop_reason": str(stop_reason),
        "recoverable": bool(recoverable),
        "elapsed_s": metric_values.get("elapsed_s"),
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
        self._generator_lock = threading.RLock()
        self._curobo_root_world_poses: dict[
            tuple[int, str], tuple[np.ndarray, np.ndarray]
        ] = {}
        self._invalid_generators: set[str] = set()
        self._config_paths: dict[str, Path] = {}
        self._base_workspace_limit_m: float | None = None

    def on_runtime_state_changed(self) -> None:
        """Invalidate live-pose-derived generator state after a q-state change."""

        self._base_workspace_limit_m = None

    @staticmethod
    def _generator_key(*, kind: str, hand: str = "left") -> str:
        hand_kinds = {"arm", "attached_arm", "arm_with_trunk", "whole_body"}
        return f"{kind}:{_normalize_hand(hand) if kind in hand_kinds else 'left'}"

    def _record_generator_recovery(self, event: dict[str, Any]) -> None:
        path = self.output_dir / "planner_generator_recovery.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_artifact_jsonable(event), sort_keys=True) + "\n")

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
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Plan BASE at its current pose without sending an Env action."""

        del timeout_s
        robot = self._find_robot()
        current = np.asarray(self._base_xy_yaw(robot), dtype=np.float64).reshape(-1)
        if current.shape not in {(3,), (4,)} or not np.isfinite(current).all():
            raise RuntimeError("R1Pro BASE pose unavailable during warmup")
        result = self._compute_base_plan(
            target_xyyaw=current[:3],
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

    def warmup(self) -> dict[str, Any]:
        """Build all Dashboard generators and run identity plans without acting."""

        started = time.monotonic()
        base = self.plan_base_identity_trajectory()
        if base.get("ok") is not True:
            raise RuntimeError(
                f"BASE identity planning failed: {base.get('stop_reason', 'unknown')}"
            )

        torso_pose = self.get_torso_pose()
        torso = self.plan_torso_trajectory(
            target_z_m=float(torso_pose[0][2]),
            timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
            start_torso_pose=torso_pose,
        )
        if torso.get("ok") is not True:
            raise RuntimeError(
                f"torso identity planning failed: {torso.get('stop_reason', 'unknown')}"
            )

        hands: dict[str, dict[str, Any]] = {}
        for hand in ("left", "right"):
            eef_pose = self.get_eef_pose(hand)
            if eef_pose is None:
                raise RuntimeError(f"{hand} EEF pose is unavailable")
            result = self.plan_whole_body_trajectory(
                hand=hand,
                target_xyz=np.asarray(eef_pose[0], dtype=np.float64),
                target_quat_xyzw=np.asarray(eef_pose[1], dtype=np.float64),
                timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                start_eef_pose=eef_pose,
            )
            if result.get("ok") is not True:
                raise RuntimeError(
                    f"{hand} identity planning failed: "
                    f"{result.get('stop_reason', 'unknown')}"
                )
            hands[hand] = {
                "ok": True,
                "trajectory_waypoints": int(
                    len(np.asarray(result["joint_trajectory"]))
                ),
            }

        report = {
            "status": "complete",
            "identity_warmup": {
                "base": {"ok": True},
                "torso": {
                    "ok": True,
                    "active_dof_count": len(TORSO_ACTIVE_JOINT_NAMES),
                    "trajectory_waypoints": int(
                        len(np.asarray(torso["joint_trajectory"]))
                    ),
                    "env_actions_sent": 0,
                    "simulator_advanced": False,
                },
                "hands": hands,
                "env_actions_sent": 0,
                "simulator_advanced": False,
            },
            "threaded_predicted_planning_ready": (
                self.threaded_predicted_planning_ready()
            ),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        path = self.output_dir / "planner_curobo_warmup.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["artifact"] = str(path)
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
        # The non-selected EEF is intentionally unconstrained while both arms
        # stay in the 21-DOF optimization.
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
        if len(locked_names) != 25 or TORSO_LOCKED_JOINT_NAME not in locked_names:
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
        q_by_name = {name: float(q[index]) for index, name in enumerate(robot_names)}
        cspace["retract_config"] = [q_by_name[name] for name in joint_names]
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

    @staticmethod
    def _disable_curobo_collision_costs(generator: Any) -> None:
        """Disable every CuRobo world/self collision term for this simulator mode."""

        for motion_generator in (getattr(generator, "mg", {}) or {}).values():
            rollouts = motion_generator.get_all_rollout_instances()
            for rollout in rollouts:
                for attribute in (
                    "primitive_collision_cost",
                    "primitive_collision_constraint",
                    "robot_self_collision_cost",
                    "robot_self_collision_constraint",
                ):
                    cost = getattr(rollout, attribute, None)
                    disable = getattr(cost, "disable_cost", None)
                    if callable(disable):
                        disable()

    def threaded_predicted_planning_ready(self) -> bool:
        """Return whether all generator root frames were snapshotted on-main."""

        required_keys = (
            self._generator_key(kind="base"),
            self._generator_key(kind="torso"),
            self._generator_key(kind="whole_body", hand="left"),
            self._generator_key(kind="whole_body", hand="right"),
        )
        for key in required_keys:
            generator = self._generators.get(key)
            if generator is None:
                return False
            if not any(
                cache_key[0] == id(generator)
                for cache_key in self._curobo_root_world_poses
            ):
                return False
        return True

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
                "self_collision_check": False,
            },
            batch_size=2,
            use_cuda_graph=False,
            use_default_embodiment_only=use_default_embodiment_only,
        )
        self._disable_curobo_collision_costs(generator)
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

    def _goal_only_curobo_trajectory(
        self,
        *,
        generator: Any,
        robot: Any,
        target_positions: Any,
        target_quaternions: Any,
        start_q: Any,
        embodiment: Any,
        position_only: bool,
    ) -> dict[str, Any]:
        """Run one joint-limits-and-goal fast-trajopt query."""

        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        start = np.asarray(_jsonable(start_q), dtype=np.float32).reshape(-1)
        batch_size = max(int(generator.batch_size), 1)
        self._disable_curobo_collision_costs(generator)
        started = time.monotonic()
        try:
            with self._whole_body_plan_policy(
                generator,
                enable_graph=False,
                position_only=position_only,
                timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
            ):
                raw_results = generator.compute_trajectories(
                    target_positions,
                    target_quaternions,
                    initial_joint_pos=torch.as_tensor(start, dtype=torch.float32),
                    is_local=True,
                    max_attempts=1,
                    timeout=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                    ik_fail_return=1,
                    enable_finetune_trajopt=True,
                    finetune_attempts=1,
                    return_full_result=True,
                    success_ratio=1.0 / batch_size,
                    attached_obj=None,
                    attached_obj_scale=None,
                    skip_obstacle_update=True,
                    ik_only=False,
                    ik_world_collision_check=False,
                    emb_sel=embodiment,
                )
        except Exception as exc:
            elapsed = time.monotonic() - started
            is_timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower()
            return {
                "ok": False,
                "stop_reason": "timeout" if is_timeout else "solver_error",
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {
                    "solver": "fast_trajopt",
                    "solver_timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                    "solver_elapsed_s": round(elapsed, 3),
                    "enable_graph": False,
                    "max_attempts": 1,
                    "curobo_statuses": [],
                },
            }
        elapsed = time.monotonic() - started
        result_batches = (
            list(raw_results)
            if isinstance(raw_results, (list, tuple))
            else [raw_results]
        )
        status_values = [getattr(result, "status", None) for result in result_batches]
        successful_paths: list[Any] = []
        success_values: list[bool] = []
        for result in result_batches:
            batch_success = np.asarray(
                _jsonable(getattr(result, "success", False)),
                dtype=bool,
            ).reshape(-1)
            success_values.extend(bool(value) for value in batch_success)
            if not np.any(batch_success):
                continue
            get_paths = getattr(result, "get_paths", None)
            if not callable(get_paths):
                raise RuntimeError("CuRobo result omitted get_paths()")
            batch_paths = list(get_paths())
            for index in np.flatnonzero(batch_success):
                if int(index) >= len(batch_paths):
                    raise RuntimeError(
                        "CuRobo result path count does not match success flags"
                    )
                successful_paths.append(batch_paths[int(index)])
        success_array = np.asarray(success_values, dtype=bool)
        status_labels = _curobo_status_labels(status_values)
        metrics = {
            "solver": "fast_trajopt",
            "solver_timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
            "solver_elapsed_s": round(elapsed, 3),
            "enable_graph": False,
            "max_attempts": 1,
            "successes": success_array.tolist(),
            "curobo_statuses": status_labels,
        }
        if not successful_paths:
            stop_reason, error, status_labels = _curobo_failure_from_status(
                status_values,
                elapsed_s=elapsed,
            )
            metrics["curobo_statuses"] = status_labels
            return {
                "ok": False,
                "stop_reason": stop_reason,
                "error": error,
                "metrics": metrics,
            }

        path = successful_paths[0]
        full_names = tuple(str(name) for name in generator.robot_joint_names)
        robot_names = tuple(str(name) for name in (getattr(robot, "joints", {}) or {}))
        path_names = tuple(
            str(name) for name in (getattr(path, "joint_names", None) or ())
        )
        if not path_names:
            raise RuntimeError("CuRobo path omitted joint names")
        values = np.asarray(
            _jsonable(getattr(path, "position", None)), dtype=np.float32
        ).reshape(-1, len(path_names))
        active_names = tuple(
            str(name) for name in generator.mg[embodiment].kinematics.joint_names
        )
        if (
            full_names != robot_names
            or start.shape != (len(full_names),)
            or values.shape[0] < 1
            or not set(active_names).issubset(path_names)
        ):
            raise RuntimeError("CuRobo path does not match the R1Pro joint layout")
        full_index = {name: index for index, name in enumerate(full_names)}
        merged = np.broadcast_to(start, (len(values), len(start))).copy()
        for source_index, name in enumerate(path_names):
            if name in active_names:
                merged[:, full_index[name]] = values[:, source_index]
        # OmniGibson obtains this JointState from MotionGenResult.get_paths(),
        # whose interpolated path starts at t=0.  That first row is the supplied
        # start state and is never an executable Env action.
        command_rows = merged[1:]
        if len(command_rows) < 1:
            return {
                "ok": False,
                "stop_reason": "unreachable",
                "error": "CuRobo returned no executable waypoint",
                "metrics": metrics,
            }
        selected, source_indices = _deterministic_execution_waypoints(command_rows)
        metrics.update(
            {
                "solver_waypoints": int(len(merged)),
                "geometric_waypoints": int(len(selected)),
                "selected_source_indices": source_indices,
                "start_state_row_removed": True,
            }
        )
        action_rows = self.joint_trajectory_to_actions(
            selected,
            start_q=start,
        )
        return {
            "ok": True,
            "joint_trajectory": selected,
            "action_trajectory": action_rows,
            "metrics": metrics,
        }

    def _world_pose_in_curobo_root(
        self,
        *,
        generator: Any,
        robot: Any,
        embodiment: Any,
        position: Any,
        quaternion_xyzw: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use one main-thread root snapshot for pure predicted-start planning."""

        key = (id(generator), str(embodiment))
        root_pose = self._curobo_root_world_poses.get(key)
        if root_pose is None:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError(
                    "CuRobo root frame must be snapshotted on the env main thread"
                )
            root_name = generator.base_link[embodiment]
            root_link = (getattr(robot, "links", {}) or {}).get(root_name)
            if root_link is None:
                raise RuntimeError(f"CuRobo root link {root_name!r} is unavailable")
            root_position, root_quaternion = root_link.get_position_orientation()
            root_pose = (
                np.asarray(_jsonable(root_position), dtype=np.float64).reshape(3),
                np.asarray(_jsonable(root_quaternion), dtype=np.float64).reshape(4),
            )
            self._curobo_root_world_poses[key] = (
                root_pose[0].copy(),
                root_pose[1].copy(),
            )
        root_position, root_quaternion = root_pose
        inverse_root_quaternion = np.asarray(
            [
                -root_quaternion[0],
                -root_quaternion[1],
                -root_quaternion[2],
                root_quaternion[3],
            ],
            dtype=np.float64,
        )
        local_position = _quat_rotate_vector_xyzw(
            inverse_root_quaternion,
            np.asarray(position, dtype=np.float64).reshape(3) - root_position,
        )
        local_quaternion = _quat_multiply_xyzw(
            inverse_root_quaternion,
            np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4),
        )
        return local_position, local_quaternion

    def _compute_whole_body_goal_only_plan(
        self,
        *,
        hand: str,
        target_xyz: Any,
        target_quat_xyzw: Any | None,
        start_q: Any | None,
        start_eef_pose: Any | None,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        with self._generator_lock:
            generator = self._generator(kind="whole_body", hand=hand)
            robot = self._find_robot()
            start = np.asarray(
                _jsonable(robot.get_joint_positions() if start_q is None else start_q),
                dtype=np.float32,
            ).reshape(-1)
            selected_pose = (
                start_eef_pose
                if start_eef_pose is not None
                else self.get_eef_pose(hand)
            )
            if selected_pose is None:
                raise RuntimeError(f"R1Pro {hand} EEF pose is unavailable")
            target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
            position_only = target_quat_xyzw is None
            quaternion = np.asarray(
                selected_pose[1] if target_quat_xyzw is None else target_quat_xyzw,
                dtype=np.float64,
            ).reshape(4)
            torch = self._torch
            if torch is None:
                import torch as torch  # type: ignore[no-redef]
            batch_size = max(int(generator.batch_size), 1)
            link = self._eef_link_name(robot, hand)
            local_target, local_quaternion = self._world_pose_in_curobo_root(
                generator=generator,
                robot=robot,
                embodiment=self._embodiment_cls.DEFAULT,
                position=target,
                quaternion_xyzw=quaternion,
            )
            result = self._goal_only_curobo_trajectory(
                generator=generator,
                robot=robot,
                target_positions={
                    link: torch.as_tensor(local_target, dtype=torch.float32)
                    .reshape(1, 3)
                    .repeat(batch_size, 1)
                },
                target_quaternions={
                    link: torch.as_tensor(local_quaternion, dtype=torch.float32)
                    .reshape(1, 4)
                    .repeat(batch_size, 1)
                },
                start_q=start,
                embodiment=self._embodiment_cls.DEFAULT,
                position_only=position_only,
            )
            if result.get("ok") is True:
                result["metrics"] = {
                    **dict(result.get("metrics") or {}),
                    "motion_scope": "whole_body",
                    "active_dof_count": 21,
                }
            return result

    def plan_whole_body_trajectory(
        self,
        *,
        hand: str,
        target_xyz: np.ndarray,
        target_quat_xyzw: np.ndarray | None,
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
        start_q: Any | None = None,
        start_eef_pose: Any | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        """Plan one selected-EEF 21-DOF fast-trajopt trajectory."""

        hand = _normalize_hand(hand)
        del timeout_s, background
        started = time.monotonic()
        try:
            return self._compute_whole_body_goal_only_plan(
                hand=hand,
                target_xyz=target_xyz,
                target_quat_xyzw=target_quat_xyzw,
                start_q=start_q,
                start_eef_pose=start_eef_pose,
            )
        except Exception as exc:
            return {
                "ok": False,
                "stop_reason": (
                    "timeout" if isinstance(exc, TimeoutError) else "solver_error"
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {
                    "solver": "fast_trajopt",
                    "solver_timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                    "planning_elapsed_s": round(time.monotonic() - started, 3),
                },
            }

    def get_torso_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the live torso_link4 world pose."""

        robot = self._find_robot()
        link = (getattr(robot, "links", {}) or {}).get(TORSO_LINK_NAME)
        if link is None:
            raise RuntimeError(f"R1Pro link {TORSO_LINK_NAME!r} is unavailable")
        position, quaternion = link.get_position_orientation()
        position_array = np.asarray(_jsonable(position), dtype=np.float64).reshape(-1)
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
        live_names = tuple(str(name) for name in (getattr(robot, "joints", {}) or {}))
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

    def torso_jog_capability(self) -> dict[str, Any]:
        """Report whether the torso generator and action converter are available."""

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
            checks = {
                "target_link_available": TORSO_LINK_NAME
                in (getattr(robot, "links", {}) or {}),
                "pose_available": (
                    np.asarray(pose[0]).shape == (3,)
                    and np.asarray(pose[1]).shape == (4,)
                ),
                "active_joint_set_exact": active_names == TORSO_ACTIVE_JOINT_NAMES,
                "action_prepack_callable": callable(
                    getattr(self, "joint_trajectory_to_actions", None)
                ),
                "plan_callable": callable(getattr(self, "plan_torso_trajectory", None)),
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
        """Plan one torso_link4 world-Z fast-trajopt motion."""

        started = time.monotonic()
        target_z = float(target_z_m)
        if not math.isfinite(target_z):
            raise ValueError("torso target Z must be finite")
        timeout = float(timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("torso planning timeout must be finite and positive")
        try:
            with self._generator_lock:
                generator = self._generator(kind="torso")
                robot = self._find_robot()
                start = np.asarray(
                    _jsonable(
                        robot.get_joint_positions() if start_q is None else start_q
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                if start_torso_pose is None:
                    start_position, start_quaternion = self.get_torso_pose()
                else:
                    start_position = np.asarray(
                        start_torso_pose[0], dtype=np.float64
                    ).reshape(3)
                    start_quaternion = np.asarray(
                        start_torso_pose[1], dtype=np.float64
                    ).reshape(4)
                target_position = np.asarray(
                    [start_position[0], start_position[1], target_z],
                    dtype=np.float64,
                )
                torch = self._torch
                if torch is None:
                    import torch as torch  # type: ignore[no-redef]
                batch_size = max(int(generator.batch_size), 1)
                local_target, local_quaternion = self._world_pose_in_curobo_root(
                    generator=generator,
                    robot=robot,
                    embodiment=self._embodiment_cls.DEFAULT,
                    position=target_position,
                    quaternion_xyzw=start_quaternion,
                )
                result = self._goal_only_curobo_trajectory(
                    generator=generator,
                    robot=robot,
                    target_positions=torch.as_tensor(local_target, dtype=torch.float32)
                    .reshape(1, 3)
                    .repeat(batch_size, 1),
                    target_quaternions=torch.as_tensor(
                        local_quaternion, dtype=torch.float32
                    )
                    .reshape(1, 4)
                    .repeat(batch_size, 1),
                    start_q=start,
                    embodiment=self._embodiment_cls.DEFAULT,
                    position_only=False,
                )
                if result.get("ok") is True:
                    result["metrics"] = {
                        **dict(result.get("metrics") or {}),
                        "motion_scope": "torso",
                        "active_dof_count": len(TORSO_ACTIVE_JOINT_NAMES),
                    }
                return result
        except Exception as exc:
            return {
                "ok": False,
                "stop_reason": (
                    "timeout" if isinstance(exc, TimeoutError) else "solver_error"
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": {
                    "solver": "fast_trajopt",
                    "solver_timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                    "planning_elapsed_s": round(time.monotonic() - started, 3),
                },
            }

    def _curobo_eef_poses(
        self,
        generator: Any,
        q_trajectory: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run CuRobo FK for predicted terminal poses."""

        from omnigibson import lazy

        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        embodiment = self._embodiment_cls.DEFAULT
        q_tensor = generator._tensor_args.to_device(
            torch.as_tensor(
                np.asarray(_jsonable(q_trajectory), dtype=np.float32),
                dtype=torch.float32,
            )
        )
        joint_state = lazy.curobo.types.state.JointState(
            position=q_tensor,
            joint_names=generator.robot_joint_names,
        ).get_ordered_joint_state(generator.mg[embodiment].kinematics.joint_names)
        state = generator.mg[embodiment].kinematics.compute_kinematics(joint_state)
        positions = np.asarray(_jsonable(state.ee_position), dtype=np.float64).reshape(
            -1, 3
        )
        quaternions_wxyz = np.asarray(
            _jsonable(state.ee_quaternion), dtype=np.float64
        ).reshape(-1, 4)
        return positions, quaternions_wxyz[:, [1, 2, 3, 0]]

    def whole_body_eef_poses(
        self,
        hand: str,
        q_trajectory: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return selected-EEF FK for one predicted full-q trajectory."""

        with self._generator_lock:
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

        with self._generator_lock:
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
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        current = self.get_base_pose()
        delta = target[:2] - current[:2]
        distance = float(np.linalg.norm(delta))
        direction = (
            delta / distance
            if distance > 1e-9
            else np.asarray([math.cos(current[2]), math.sin(current[2])])
        )
        goal_xy = target[:2] - direction * float(standoff_m)
        goal = _canonical_base_xyyaw(
            np.asarray(
                [
                    goal_xy[0],
                    goal_xy[1],
                    math.atan2(float(direction[1]), float(direction[0])),
                ]
            )
        )
        plan = self._compute_base_plan(
            target_xyyaw=goal,
        )
        plan["metrics"] = {
            **dict(plan.get("metrics") or {}),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        return plan

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
        robot = self._find_robot()
        current = self._base_xy_yaw(robot)
        delta = target[:2] - current[:2]
        target_distance = float(np.linalg.norm(delta))
        direction = (
            delta / target_distance
            if target_distance > 1e-9
            else np.asarray([math.cos(current[2]), math.sin(current[2])])
        )
        desired_travel = max(0.0, target_distance - standoff)
        travel = min(desired_travel, max_travel)
        goal_xy = current[:2] + direction * travel
        goal = _canonical_base_xyyaw(
            np.asarray(
                [
                    goal_xy[0],
                    goal_xy[1],
                    math.atan2(float(direction[1]), float(direction[0])),
                ]
            )
        )
        plan = self._compute_base_plan(
            target_xyyaw=goal,
        )
        plan["metrics"] = {
            **dict(plan.get("metrics") or {}),
            "planned_travel_m": travel,
            "requested_standoff_m": standoff,
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        return plan

    def plan_relative_navigation_trajectory(
        self,
        *,
        relative_motion: Any,
        timeout_s: float,
        start_q: Any | None = None,
        start_base_xyyaw: Any | None = None,
        background: bool = False,
        base_planning_profile: str | None = None,
    ) -> dict[str, Any]:
        """Plan one exact straight translation or in-place BASE rotation."""

        started = time.monotonic()
        motion = validate_relative_navigation_motion(relative_motion)
        del timeout_s, background
        robot = self._find_robot()
        start = np.asarray(
            _jsonable(robot.get_joint_positions() if start_q is None else start_q),
            dtype=np.float32,
        ).reshape(-1)
        base_indices = _indices(getattr(robot, "base_idx", []))
        if len(base_indices) != 6:
            raise RuntimeError("R1Pro base joint layout is unavailable")
        current = np.asarray(
            (
                [
                    start[base_indices[0]],
                    start[base_indices[1]],
                    start[base_indices[5]],
                ]
                if start_base_xyyaw is None
                else _jsonable(start_base_xyyaw)
            ),
            dtype=np.float64,
        ).reshape(-1)
        if current.shape not in {(3,), (4,)}:
            raise ValueError("relative navigation start BASE pose is invalid")
        base_goal = np.asarray(current[:3], dtype=np.float64).copy()
        if motion["kind"] == "translation":
            signed_distance = float(motion["distance_m"]) * (
                1.0 if motion["direction"] == "forward" else -1.0
            )
            base_goal[:2] += signed_distance * np.asarray(
                [math.cos(float(current[2])), math.sin(float(current[2]))],
                dtype=np.float64,
            )
        else:
            signed_angle = math.radians(float(motion["angle_deg"])) * (
                1.0 if motion["direction"] == "left" else -1.0
            )
            base_goal[2] += signed_angle
        base_goal = _canonical_base_xyyaw(base_goal)
        result = self._compute_base_plan(
            target_xyyaw=base_goal,
            start_q=start,
            planning_profile=(
                DASHBOARD_PREPARED_BASE_PLANNING_PROFILE
                if base_planning_profile is None
                else str(base_planning_profile)
            ),
        )
        metrics = {
            **dict(result.get("metrics") or {}),
            "relative_motion": dict(motion),
            "base_goal": base_goal.tolist(),
            "elapsed_s": round(time.monotonic() - started, 3),
        }
        if result.get("ok") is not True:
            return {
                "ok": False,
                "stop_reason": str(result.get("stop_reason", "unreachable")),
                "error": result.get("error"),
                "metrics": metrics,
            }
        return {
            "ok": True,
            "joint_trajectory": result["joint_trajectory"],
            "action_trajectory": result["action_trajectory"],
            "base_goal": base_goal,
            "metrics": metrics,
        }

    def _compute_base_plan(
        self,
        *,
        target_xyyaw: np.ndarray,
        start_q: Any | None = None,
        planning_profile: str = "default",
    ) -> dict[str, Any]:
        target_xyyaw = _canonical_base_xyyaw(target_xyyaw)
        with self._generator_lock:
            generator = self._generator(kind="base")
            robot = self._find_robot()
            start = np.asarray(
                _jsonable(robot.get_joint_positions() if start_q is None else start_q),
                dtype=np.float32,
            ).reshape(-1)
            base_indices = _indices(getattr(robot, "base_idx", []))
            if len(base_indices) != 6:
                raise RuntimeError(
                    "R1Pro six-axis virtual base indices are unavailable"
                )
            target_position = np.asarray(
                [
                    target_xyyaw[0],
                    target_xyyaw[1],
                    start[base_indices[2]],
                ],
                dtype=np.float64,
            )
            target_quaternion = _intrinsic_rpy_to_quat_xyzw(
                float(start[base_indices[3]]),
                float(start[base_indices[4]]),
                float(target_xyyaw[2]),
            )
            torch = self._torch
            if torch is None:
                import torch as torch  # type: ignore[no-redef]
            batch_size = max(int(generator.batch_size), 1)
            local_target, local_quaternion = self._world_pose_in_curobo_root(
                generator=generator,
                robot=robot,
                embodiment=self._embodiment_cls.BASE,
                position=target_position,
                quaternion_xyzw=target_quaternion,
            )
            result = self._goal_only_curobo_trajectory(
                generator=generator,
                robot=robot,
                target_positions=torch.as_tensor(local_target, dtype=torch.float32)
                .reshape(1, 3)
                .repeat(batch_size, 1),
                target_quaternions=torch.as_tensor(
                    local_quaternion, dtype=torch.float32
                )
                .reshape(1, 4)
                .repeat(batch_size, 1),
                start_q=start,
                embodiment=self._embodiment_cls.BASE,
                position_only=False,
            )
            result["base_goal"] = target_xyyaw.copy()
            result["metrics"] = {
                **dict(result.get("metrics") or {}),
                "planning_profile": str(planning_profile),
                "motion_scope": "base",
                "active_dof_count": 3,
            }
            return result

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

    def joint_trajectory_to_actions(
        self,
        q_trajectory: Any,
        *,
        start_q: Any,
    ) -> np.ndarray:
        """Pre-pack CuRobo waypoints without reading simulator state at execution."""

        robot = self._find_robot()
        torch = self._torch
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
        targets = np.asarray(_jsonable(q_trajectory), dtype=np.float32).reshape(
            -1, len(robot.joints)
        )
        previous = torch.as_tensor(
            np.asarray(_jsonable(start_q), dtype=np.float32).reshape(-1),
            dtype=torch.float32,
        )
        base_indices = _indices(getattr(robot, "base_idx", []))
        if len(base_indices) != 6:
            raise RuntimeError("R1Pro six-axis virtual base layout is unavailable")
        action_rows: list[np.ndarray] = []
        for target_value in targets:
            target = torch.as_tensor(target_value, dtype=torch.float32)
            commands = []
            for name, controller in robot.controllers.items():
                mro_names = {cls.__name__ for cls in type(controller).__mro__}
                if "MultiFingerGripperController" in mro_names:
                    side = name.removeprefix("gripper_")
                    commands.append(
                        torch.as_tensor(
                            [self._gripper_latch(side)],
                            dtype=target.dtype,
                            device=target.device,
                        )
                    )
                    continue
                if (
                    "JointController" not in mro_names
                    and "HolonomicBaseJointController" not in mro_names
                ) or bool(getattr(controller, "use_delta_commands", False)):
                    raise RuntimeError(
                        "CuRobo action packing requires absolute joint controllers; "
                        f"controller={name!r} type={type(controller).__name__!r}"
                    )
                command = target[controller.dof_idx]
                if "HolonomicBaseJointController" in mro_names:
                    current_yaw = previous[base_indices[5]]
                    world_delta = torch.stack(
                        (
                            command[0] - previous[base_indices[0]],
                            command[1] - previous[base_indices[1]],
                        )
                    )
                    cosine = torch.cos(current_yaw)
                    sine = torch.sin(current_yaw)
                    local_x = cosine * world_delta[0] + sine * world_delta[1]
                    local_y = -sine * world_delta[0] + cosine * world_delta[1]
                    yaw_delta = command[2] - current_yaw
                    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
                    command = torch.stack((local_x, local_y, yaw_delta))
                commands.append(controller._reverse_preprocess_command(command))
            action = np.asarray(
                _jsonable(torch.cat(commands, dim=0)), dtype=np.float32
            ).reshape(ACTION_DIM)
            action_rows.append(action)
            previous = target
        return validate_action_chunk(np.stack(action_rows, axis=0))

    def action_for_control_target(self, target_q: Any) -> np.ndarray:
        """Encode one absolute q target against the current live controller state."""

        live_q = self.get_joint_positions()
        target = np.asarray(_jsonable(target_q), dtype=np.float32).reshape(-1)
        if target.shape != live_q.shape:
            raise RuntimeError(
                "control target does not match the live R1Pro joint layout"
            )
        return self.joint_trajectory_to_actions(
            target.reshape(1, -1),
            start_q=live_q,
        )[0]

    def hold_action(self, hand: str | None = None) -> np.ndarray:
        del hand
        robot = self._find_robot()
        q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float32)
        action = self.joint_trajectory_to_actions(
            q.reshape(1, -1),
            start_q=q,
        )[0]
        for side in ("left", "right"):
            action[ENV_ACTION_SEGMENTS[f"{side}_gripper"]] = self._gripper_latch(side)
        return validate_action_chunk(action.reshape(1, ACTION_DIM))[0]

    def _gripper_latch(self, hand: str) -> float:
        value = getattr(self.env_facade, "_gripper_latch", {}).get(hand, 1.0)
        return float(value)


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
    """Return one BASE pose with yaw canonicalized to [-pi, pi)."""

    base_xyyaw = np.asarray(_jsonable(value), dtype=np.float64).reshape(-1)
    if base_xyyaw.shape != (3,):
        raise ValueError("BASE xyyaw must contain three values")
    canonical = base_xyyaw.copy()
    yaw = float(canonical[2])
    canonical[2] = _wrap_angle(yaw) if yaw < -math.pi or yaw >= math.pi else yaw
    return canonical


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
        self._prepared_motion_lock = threading.RLock()
        self._prepared_motions: dict[str, dict[str, Any]] = {}
        self._warmup_report: dict[str, Any] | None = None

    def on_runtime_state_changed(self) -> None:
        """Reset executor-local state after a controller or q-state change."""

        self.last_info = None
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
            raise RuntimeError("planner backend does not implement generator warmup")
        report = dict(warmup())
        self._warmup_report = deepcopy(report)
        return report

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
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Plan one BASE motion and execute at most three CuRobo waypoints."""

        del timeout_s
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
                value is not None for value in (target_xyz, standoff_m, max_travel_m)
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
                "timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
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
                "timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
            }
            unavailable_reason = "navigation_planner_unavailable"
        if not callable(plan_navigation):
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason=unavailable_reason,
                recoverable=False,
                diagnostics={"error": "BASE CuRobo planner is unavailable"},
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
                recoverable=True,
                metrics=plan_metrics,
                diagnostics=(
                    {"error": plan.get("error")} if plan.get("error") else None
                ),
            )
        result = self._execute_goal_only_waypoints(
            actions=plan.get("action_trajectory"),
            joint_targets=plan.get("joint_trajectory"),
        )
        result["metrics"] = {
            **plan_metrics,
            **dict(result.get("metrics") or {}),
        }
        return result

    @staticmethod
    def _manual_result_metrics(
        result: dict[str, Any],
        **metrics: Any,
    ) -> dict[str, Any]:
        out = dict(result)
        out["metrics"] = {**dict(result.get("metrics") or {}), **metrics}
        return out

    def _wrist_camera_rotation_calibration(self, hand: str) -> dict[str, Any]:
        """Resolve one same-step wrist-camera axis in the live EEF frame."""

        hand = _normalize_hand(hand)
        camera = f"{hand}_wrist"
        try:
            frame = self.frame_cache.latest(camera)
            if (
                not isinstance(frame.capture_group_id, str)
                or not frame.capture_group_id
            ):
                raise RuntimeError("wrist calibration requires an atomic capture group")
            current_env_step = int(getattr(self.env, "_env_steps", -1))
            if current_env_step != int(frame.step_index):
                raise RuntimeError(
                    "wrist calibration frame is not from the current simulator step"
                )
            # Pixel/depth projection uses the wall-clock FrameCache TTL, but this
            # rigid camera-to-EEF axis remains valid while the simulator step is
            # unchanged.  Requiring the exact current env step prevents mixing
            # an old wrist frame with a newly moved EEF without adding a
            # time-based capability failure while the simulator is idle.
            world_from_camera = validated_rigid_transform(
                frame.camera_to_world,
                name=f"{hand} captured wrist-camera world transform",
            )
            live_eef_pose = _call_optional_arg(
                self.backend,
                "get_eef_pose",
                hand,
            )
            if live_eef_pose is None:
                raise RuntimeError("live wrist EEF pose is unavailable")
            live_eef_quaternion = _quat_xyzw(live_eef_pose[1])
            assert live_eef_quaternion is not None
            camera_rotation = np.asarray(world_from_camera[:3, :3], dtype=np.float64)
            # USD cameras use +X right, +Y up and -Z forward.  Local +Z
            # therefore points toward the viewer; by the right-hand rule a
            # positive turn around that screen normal is visually
            # counterclockwise.
            screen_normal_world = camera_rotation @ np.array(
                [0.0, 0.0, 1.0], dtype=np.float64
            )
            inverse_eef_quaternion = np.asarray(
                [
                    -live_eef_quaternion[0],
                    -live_eef_quaternion[1],
                    -live_eef_quaternion[2],
                    live_eef_quaternion[3],
                ],
                dtype=np.float64,
            )
            screen_normal_eef = _quat_rotate_vector_xyzw(
                inverse_eef_quaternion,
                screen_normal_world,
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
                "source": "same_step_wrist_camera+live_EEF_pose",
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
                "source": ("same_step_RGBD_camera_to_world+live_EEF_pose"),
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
        wrist = {
            hand: bool(
                wrist_geometry[hand].get("available") is True
                and wrist_geometry[hand].get("verified") is True
            )
            for hand in ("left", "right")
        }
        torso_report = _call_optional(self.backend, "torso_jog_capability")
        warmup = self._warmup_report if isinstance(self._warmup_report, dict) else {}
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
            and int(identity_warmup.get("env_actions_sent", -1)) == 0
            and identity_warmup.get("simulator_advanced") is False
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
                        "torso_link4 CuRobo/controller/warmup capability is unverified"
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
            predicted.get("eef_by_hand") if isinstance(predicted, dict) else None
        )
        cached = eef_by_hand.get(hand) if isinstance(eef_by_hand, dict) else None
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
                return positions[0], quaternions[0]
        live = _call_optional_arg(self.backend, "get_eef_pose", hand)
        if predecessor is None and isinstance(live, (tuple, list)) and len(live) == 2:
            quat = _quat_xyzw(live[1])
            position = np.asarray(live[0], dtype=np.float64).reshape(-1)
            if (
                quat is not None
                and position.shape == (3,)
                and np.isfinite(position).all()
            ):
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
        cached = predicted.get("torso_link4") if isinstance(predicted, dict) else None
        if isinstance(cached, dict):
            position = np.asarray(cached.get("xyz"), dtype=np.float64).reshape(-1)
            quaternion = np.asarray(cached.get("quat_xyzw"), dtype=np.float64).reshape(
                -1
            )
            if position.shape == (3,) and quaternion.shape == (4,):
                return position, quaternion
        poses = _call_optional_kw(
            self.backend,
            "torso_poses",
            q_trajectory=np.asarray(start_q, dtype=np.float32).reshape(1, -1),
        )
        if isinstance(poses, (tuple, list)) and len(poses) == 2:
            positions = np.asarray(poses[0], dtype=np.float64).reshape(-1, 3)
            quaternions = np.asarray(poses[1], dtype=np.float64).reshape(-1, 4)
            if len(positions) == 1 and len(quaternions) == 1:
                return positions[0], quaternions[0]
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
            None if predecessor_plan_id is None else str(predecessor_plan_id).strip()
        )
        if action in {"observe", "open", "close"}:
            raise ValueError(
                f"{action} is one-shot and cannot enter the prepared queue"
            )
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
        start_q = np.asarray(_jsonable(start_q_value), dtype=np.float32).reshape(-1)
        if start_q.size < 1:
            raise RuntimeError("prepared motion start joint state is unavailable")
        base_pose_value = (
            predecessor_terminal.get("base_xyyaw")
            if isinstance(predecessor_terminal, dict)
            else _call_optional(self.backend, "get_base_pose")
        )
        base_pose = np.asarray(_jsonable(base_pose_value), dtype=np.float64).reshape(-1)
        if base_pose.shape != (3,):
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
                + (TORSO_VERTICAL_STEP_M if action == "up" else -TORSO_VERTICAL_STEP_M)
            )
            planner = getattr(self.backend, "plan_torso_trajectory", None)
            if not callable(planner):
                raise RuntimeError("torso_planner_unavailable")
            plan = planner(
                target_z_m=target_torso_z,
                timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                start_q=start_q,
                start_torso_pose=(
                    start_torso_position,
                    start_torso_quaternion,
                ),
                background=bool(background),
            )
        elif target == "chassis":
            motion_kind = "base"
            planner = getattr(self.backend, "plan_relative_navigation_trajectory", None)
            if not callable(planner):
                raise RuntimeError("relative_navigation_planner_unavailable")
            base_planner_kwargs = {
                "relative_motion": self._dashboard_base_motion(action),
                "timeout_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                "start_q": start_q,
                "start_base_xyyaw": base_pose,
                "background": bool(background),
                "base_planning_profile": (DASHBOARD_PREPARED_BASE_PLANNING_PROFILE),
            }
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
                predecessor_calibration = (
                    predecessor.get("wrist_calibration")
                    if isinstance(predecessor, dict)
                    and predecessor.get("target") == target
                    and predecessor.get("action") == action
                    else None
                )
                wrist_calibration = (
                    deepcopy(predecessor_calibration)
                    if isinstance(predecessor_calibration, dict)
                    else self._wrist_camera_rotation_calibration(hand)
                )
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
                    _axis_angle_to_quat_xyzw([*axis_eef, requested_wrist_rotation_rad]),
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
                    raise ValueError(f"unsupported repeatable arm action {action!r}")
                local_delta = local_by_action[action] * EEF_TRANSLATION_STEP_M
                yaw = float(base_pose[2])
                world_delta = np.asarray(
                    [
                        math.cos(yaw) * local_delta[0] - math.sin(yaw) * local_delta[1],
                        math.sin(yaw) * local_delta[0] + math.cos(yaw) * local_delta[1],
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
                timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
                start_q=start_q,
                start_eef_pose=(start_xyz, start_quat),
                background=bool(background),
            )
        elapsed = time.monotonic() - started
        if not isinstance(plan, dict) or plan.get("ok") is not True:
            reason = (
                plan.get("stop_reason", "planning_failed")
                if isinstance(plan, dict)
                else "planner_returned_non_mapping"
            )
            error = (
                plan.get("error")
                if isinstance(plan, dict) and plan.get("error") is not None
                else None
            )
            raise CuroboPlanningError(str(reason), error)
        q_path = np.asarray(_jsonable(plan.get("joint_trajectory")), dtype=np.float32)
        action_path = validate_action_chunk(plan.get("action_trajectory"))
        if (
            q_path.ndim != 2
            or len(q_path) < 1
            or q_path.shape[1] != start_q.size
            or len(action_path) != len(q_path)
        ):
            raise RuntimeError(
                "prepared Dashboard plan omitted aligned q/action trajectories"
            )
        if motion_kind == "base":
            plan["base_goal"] = _canonical_base_xyyaw(plan.get("base_goal"))
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
                quaternions = np.asarray(pose_value[1], dtype=np.float64).reshape(-1, 4)
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
        if isinstance(torso_pose_value, (tuple, list)) and len(torso_pose_value) == 2:
            torso_positions = np.asarray(torso_pose_value[0], dtype=np.float64).reshape(
                -1, 3
            )
            torso_quaternions = np.asarray(
                torso_pose_value[1], dtype=np.float64
            ).reshape(-1, 4)
            if len(torso_positions) == 1 and len(torso_quaternions) == 1:
                torso_terminal = {
                    "xyz": torso_positions[0].tolist(),
                    "quat_xyzw": torso_quaternions[0].tolist(),
                }
        if motion_kind == "torso" and torso_terminal is None:
            raise RuntimeError("prepared torso plan omitted predicted torso_link4 FK")

        plan_id = os.urandom(16).hex()
        predicted_terminal = {
            "joint_positions": predicted_q.tolist(),
            "base_xyyaw": predicted_base.tolist(),
            "eef_by_hand": eef_by_hand,
            "torso_link4": torso_terminal,
        }
        plan_metrics = (
            dict(plan.get("metrics")) if isinstance(plan.get("metrics"), dict) else {}
        )
        planning_profile = str(
            plan_metrics.get(
                "planning_profile",
                (
                    DASHBOARD_PREPARED_TORSO_PLANNING_PROFILE
                    if motion_kind == "torso"
                    else WHOLE_BODY_SEARCH_PROFILE_DASHBOARD_JOG
                ),
            )
        )
        deadline_enforcement = {
            "solver_timeout_enforced": True,
            "hard_wall_clock_enforced": False,
            "hard_wall_clock_deadline_s": None,
            "soft_deadline_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
            "soft_deadline_exceeded": bool(elapsed > DASHBOARD_CUROBO_PLAN_TIMEOUT_S),
        }
        execution_policy = {
            "base": PREPARED_DASHBOARD_BASE_EXECUTION_POLICY,
            "eef": PREPARED_DASHBOARD_EEF_EXECUTION_POLICY,
            "torso": PREPARED_DASHBOARD_TORSO_EXECUTION_POLICY,
        }[motion_kind]
        bounded_plan_metrics = deepcopy(plan_metrics)
        prediction_epoch = (
            str(predecessor.get("prediction_epoch"))
            if isinstance(predecessor, dict) and predecessor.get("prediction_epoch")
            else os.urandom(16).hex()
        )
        prediction_sequence = (
            int(predecessor.get("prediction_sequence", 0)) + 1
            if isinstance(predecessor, dict)
            else 0
        )
        metadata = {
            "schema_version": 1,
            "plan_id": plan_id,
            "target": target,
            "action": action,
            "predecessor_plan_id": predecessor_plan_id,
            "motion_kind": motion_kind,
            "status": "prepared",
            "prediction_epoch": prediction_epoch,
            "prediction_sequence": prediction_sequence,
            "predicted_start_digest": hashlib.sha256(
                np.ascontiguousarray(start_q, dtype=np.float32).tobytes()
            ).hexdigest(),
            "predicted_terminal": predicted_terminal,
            "planning_profile": planning_profile,
            "execution_policy": execution_policy,
            "planning_deadline_s": (DASHBOARD_CUROBO_PLAN_TIMEOUT_S),
            "fast_solver_deadline_s": DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
            "background": bool(background),
            "deadline_enforcement": deadline_enforcement,
            "planning_elapsed_s": elapsed,
            "plan_metrics": bounded_plan_metrics,
            "picklable": True,
        }
        if requested_wrist_rotation_rad is not None:
            metadata.update(
                {
                    "requested_rotation_rad": (requested_wrist_rotation_rad),
                    "visual_direction": (
                        "counterclockwise" if action == "rotate_left" else "clockwise"
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
            "action_trajectory": np.ascontiguousarray(action_path, dtype=np.float32),
            "plan": plan,
            "target_xyz": None if target_xyz is None else target_xyz.copy(),
            "target_quat_xyzw": (None if target_quat is None else target_quat.copy()),
            "target_torso_z_m": target_torso_z,
            "start_eef_xyz": (
                None if hand is None or target_xyz is None else start_xyz.copy()
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

    def threaded_predicted_planning_ready(self) -> bool:
        """Expose the backend's off-main predicted-planning readiness."""

        ready = getattr(self.backend, "threaded_predicted_planning_ready", None)
        return bool(callable(ready) and ready())

    def prepare_predicted_dashboard_motion(
        self,
        target: str,
        action: str,
        predecessor_plan_id: str,
    ) -> dict[str, Any]:
        """Plan one predicted successor without reading live simulator state."""

        predecessor_plan_id = str(predecessor_plan_id).strip()
        if not predecessor_plan_id:
            raise ValueError("predicted planning requires a predecessor_plan_id")
        if not self.threaded_predicted_planning_ready():
            raise RuntimeError("threaded predicted planning is not ready")
        return self.prepare_dashboard_motion(
            target,
            action,
            predecessor_plan_id=predecessor_plan_id,
            background=True,
        )

    def execute_dashboard_motion(
        self,
        plan_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        """Execute one cached predicted plan without live-start admission."""

        plan_id = str(plan_id).strip()
        command_id = str(command_id).strip()
        if not plan_id or not command_id:
            raise ValueError("plan_id and command_id are required")
        with self._prepared_motion_lock:
            entry = self._prepared_motions.get(plan_id)
            if entry is None:
                raise KeyError(f"unknown prepared plan {plan_id!r}")
            if entry.get("status") in {"discarded", "invalidated"}:
                raise RuntimeError("prepared plan is no longer usable")
            if entry.get("status") == "executing":
                raise RuntimeError("prepared plan is already executing")
            predecessor_id = entry.get("predecessor_plan_id")
            predecessor = (
                self._prepared_motions.get(predecessor_id) if predecessor_id else None
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
            action_path = np.ascontiguousarray(
                entry["action_trajectory"], dtype=np.float32
            )
            joint_path = np.ascontiguousarray(
                entry["joint_trajectory"], dtype=np.float32
            )
            entry["execution_command_id"] = command_id
            entry["execution_event"] = threading.Event()
            entry["status"] = "executing"
        result = self._execute_goal_only_waypoints(
            actions=action_path,
            joint_targets=joint_path,
        )
        result = self._manual_result_metrics(
            result,
            prepared_plan_reused=True,
            prepared_plan_id=plan_id,
            prepared_command_id=command_id,
            predecessor_plan_id=entry.get("predecessor_plan_id"),
            prediction_epoch=entry.get("prediction_epoch"),
            prediction_sequence=entry.get("prediction_sequence"),
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
                entry["action_trajectory"] = None
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
                            descendant["invalidated_reason"] = "predecessor_discarded"
                            descendant["joint_trajectory"] = None
                            descendant["action_trajectory"] = None
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
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
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
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Execute one fixed selected-EEF translation from its live start pose."""

        del timeout_s
        hand = _normalize_hand(hand)
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
        base_pose = _call_optional(self.backend, "get_base_pose")
        eef_pose = _call_optional_arg(self.backend, "get_eef_pose", hand)
        if base_pose is None or eef_pose is None:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="pose_feedback_unavailable",
                recoverable=True,
                metrics={"env_actions_sent": 0},
            )
        base = np.asarray(base_pose, dtype=np.float64).reshape(3)
        current_position = np.asarray(eef_pose[0], dtype=np.float64).reshape(3)
        current_quat = _quat_xyzw(eef_pose[1])
        local_delta = local_by_action[action] * EEF_TRANSLATION_STEP_M
        yaw = float(base[2])
        world_delta = np.asarray(
            [
                math.cos(yaw) * local_delta[0] - math.sin(yaw) * local_delta[1],
                math.sin(yaw) * local_delta[0] + math.cos(yaw) * local_delta[1],
                local_delta[2],
            ],
            dtype=np.float64,
        )
        result = self.move_to(
            hand=hand,
            target_xyz=current_position + world_delta,
            target_quat_xyzw=current_quat,
        )
        return self._manual_result_metrics(
            result,
            manual_primitive="jog_eef",
            manual_action=action,
            requested_delta=local_delta.tolist(),
            requested_delta_frame="base_call_start",
            requested_delta_world=world_delta.tolist(),
            fixed_server_step=True,
        )

    def _execute_torso_trajectory(
        self,
        action_trajectory: Any,
        joint_trajectory: Any,
    ) -> dict[str, Any]:
        """Execute one torso-only trajectory through the minimal executor."""

        return self._execute_goal_only_waypoints(
            actions=action_trajectory,
            joint_targets=joint_trajectory,
        )

    def jog_torso(
        self,
        action: str,
        *,
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Move torso_link4 by one fixed world-Z step."""

        del timeout_s
        action = str(action)
        if action not in {"up", "down"}:
            raise ValueError("torso jog action must be up or down")
        delta = TORSO_VERTICAL_STEP_M if action == "up" else -TORSO_VERTICAL_STEP_M
        current_pose = _call_optional(self.backend, "get_torso_pose")
        if not isinstance(current_pose, (tuple, list)) or len(current_pose) != 2:
            raise RuntimeError("torso_link4 pose feedback is unavailable")
        target_z = (
            float(np.asarray(current_pose[0], dtype=np.float64).reshape(3)[2]) + delta
        )
        plan = _call_optional_kw(
            self.backend,
            "plan_torso_trajectory",
            target_z_m=target_z,
            timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
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
            plan["action_trajectory"],
            plan["joint_trajectory"],
        )
        metrics = dict(plan.get("metrics") or {})
        metrics.update(
            {
                "manual_primitive": "jog_torso",
                "manual_action": action,
                "target_link": TORSO_LINK_NAME,
                "requested_delta_z_m": delta,
                "actual_target": {"torso_link4_world_z_m": target_z},
                "fixed_server_step": True,
            }
        )
        return self._manual_result_metrics(result, **metrics)

    def jog_wrist(
        self,
        hand: str,
        action: str,
        *,
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Rotate one wrist by a calibrated visual 5 degrees."""

        del timeout_s
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
        )
        return self._manual_result_metrics(
            result,
            manual_primitive="jog_wrist",
            manual_action=action,
            hand=hand,
            requested_rotation_rad=signed_angle,
            visual_direction=("counterclockwise" if counterclockwise else "clockwise"),
            calibration=calibration,
            fixed_server_step=True,
        )

    def rotate_wrist_step(
        self,
        hand: str,
        action: str,
        *,
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Compatibility spelling for the Dashboard wrist jog."""

        return self.jog_wrist(hand, action, timeout_s=timeout_s)

    def set_gripper(
        self,
        hand: str,
        opening: float,
        *,
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Set and latch one gripper without retreat or repeated RPC calls."""

        hand = _normalize_hand(hand)
        value = float(opening)
        if not math.isfinite(value) or value not in {0.0, 1.0}:
            raise ValueError("opening must be exactly 0.0 (close) or 1.0 (open)")
        del timeout_s
        result = self._gripper_command(hand, opening=value)
        latch = getattr(self.env, "_gripper_latch", None)
        return self._manual_result_metrics(
            result,
            manual_primitive="set_gripper",
            hand=hand,
            opening=value,
            retreat_executed=False,
            network_primitive_calls=1,
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
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        target = self._world_target(target_xyz, frame=frame)
        explicit_quat = _quat_xyzw(target_quat_xyzw)
        planner = getattr(self.backend, "plan_whole_body_trajectory", None)
        if not callable(planner):
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="solver_error",
                recoverable=False,
                diagnostics={"error": "whole-body planner is unavailable"},
            )
        plan = planner(
            hand=hand,
            target_xyz=target,
            target_quat_xyzw=explicit_quat,
            timeout_s=DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
        )
        if not isinstance(plan, dict) or plan.get("ok") is not True:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason=(
                    str(plan.get("stop_reason", "unreachable"))
                    if isinstance(plan, dict)
                    else "solver_error"
                ),
                recoverable=True,
                metrics=(
                    dict(plan.get("metrics") or {}) if isinstance(plan, dict) else {}
                ),
                diagnostics=(
                    {"error": plan.get("error")}
                    if isinstance(plan, dict) and plan.get("error")
                    else None
                ),
            )
        if plan_only:
            return primitive_result(
                primitive_success=True,
                task_success=self._task_success(),
                stop_reason="plan_ready",
                recoverable=True,
                metrics=dict(plan.get("metrics") or {}),
            )
        result = self._execute_goal_only_waypoints(
            actions=plan["action_trajectory"],
            joint_targets=plan["joint_trajectory"],
        )
        result_metrics = dict(plan.get("metrics") or {})
        result_metrics.update(dict(result.get("metrics") or {}))
        result["metrics"] = result_metrics
        return result

    @_planner_tool("move_to")
    def move_to(
        self,
        *,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        target_quat_xyzw: Any | None = None,
        plan_only: bool = False,
    ) -> dict[str, Any]:
        try:
            return self._move_to_whole_body_impl(
                hand=hand,
                target_xyz=target_xyz,
                frame=frame,
                target_quat_xyzw=target_quat_xyzw,
                plan_only=plan_only,
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
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
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
            del timeout_s
            return self.move_to(
                hand=hand,
                target_xyz=position,
                target_quat_xyzw=target_quat,
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
        timeout_s: float = DASHBOARD_CUROBO_PLAN_TIMEOUT_S,
    ) -> dict[str, Any]:
        try:
            del timeout_s
            hand = _normalize_hand(hand)
            target = _as_xyz(target_xyz)
            direction = _approach_vector(
                press_direction if press_direction is not None else [0, 0, -1]
            )
            travel = float(travel_m)
            if not np.isfinite(travel) or travel <= 0.0:
                raise ValueError("travel_m must be finite and positive")
            endpoint = (
                target - direction * PRESS_EEF_TO_CONTACT_OFFSET_M + direction * travel
            )
            result = self.move_to(
                hand=hand,
                target_xyz=endpoint,
            )
            result["metrics"] = {
                **dict(result.get("metrics") or {}),
                "requested_travel_m": travel,
            }
            return result
        except Exception as exc:
            return self._exception_result(exc, suggested_next_tool="observe")

    def _gripper_command(
        self,
        hand: str,
        *,
        opening: float,
    ) -> dict[str, Any]:
        hand = _normalize_hand(hand)
        command = 1.0 if float(opening) >= 0.5 else -1.0
        hold = _call_optional_arg(self.backend, "hold_action", hand)
        if hold is None:
            hold = np.zeros((ACTION_DIM,), dtype=np.float32)
        action = np.asarray(hold, dtype=np.float32).reshape(ACTION_DIM).copy()
        action[ENV_ACTION_SEGMENTS[f"{hand}_gripper"]] = command
        return self._execute_goal_only_waypoints(
            actions=validate_action_chunk(action.reshape(1, ACTION_DIM)),
        )

    def _execute_goal_only_waypoints(
        self,
        *,
        actions: Any,
        joint_targets: Any | None = None,
    ) -> dict[str, Any]:
        """Apply each of at most three geometric rows over five control cycles."""

        started = time.monotonic()
        if self._task_success():
            return primitive_result(
                primitive_success=True,
                task_success=True,
                stop_reason="official_task_success",
                recoverable=False,
                metrics={
                    "geometric_waypoints": 0,
                    "control_cycles_per_waypoint": (
                        DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT
                    ),
                    "env_actions_sent": 0,
                    "executed_control_cycles": 0,
                    "elapsed_s": 0.0,
                },
            )
        action_chunk = validate_action_chunk(actions)
        selected, source_indices = _deterministic_execution_waypoints(action_chunk)
        action_rows = list(selected)
        geometric_waypoints = len(action_rows)
        selected_joint_targets: np.ndarray | None = None
        if joint_targets is not None:
            target_rows = np.asarray(_jsonable(joint_targets), dtype=np.float32)
            if target_rows.ndim != 2 or len(target_rows) != len(action_chunk):
                raise ValueError(
                    "joint targets must align one-to-one with geometric actions"
                )
            selected_joint_targets = np.ascontiguousarray(
                target_rows[source_indices], dtype=np.float32
            )
            action_for_target = getattr(self.backend, "action_for_control_target", None)
            if not callable(action_for_target):
                raise RuntimeError("backend action_for_control_target is unavailable")
        else:
            action_for_target = None

        sent = 0
        executed_control_cycles = 0
        try:
            for waypoint_index, action in enumerate(action_rows):
                target_q = (
                    None
                    if selected_joint_targets is None
                    else selected_joint_targets[waypoint_index]
                )
                for _ in range(DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT):
                    cycle_action = (
                        np.asarray(action, dtype=np.float32).copy()
                        if target_q is None
                        else validate_action_chunk(
                            np.asarray(
                                action_for_target(target_q),
                                dtype=np.float32,
                            ).reshape(1, ACTION_DIM)
                        )[0]
                    )
                    sent += 1
                    receipt = self._step_env_action(cycle_action)
                    executed_control_cycles += 1
                    terminal = _terminal_step_outcome(receipt)
                    if terminal is None:
                        continue
                    primitive_success, stop_reason = terminal
                    metrics: dict[str, Any] = {
                        "geometric_waypoints": geometric_waypoints,
                        "control_cycles_per_waypoint": (
                            DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT
                        ),
                        "env_actions_sent": sent,
                        "executed_control_cycles": executed_control_cycles,
                        "selected_source_indices": source_indices,
                        "elapsed_s": round(time.monotonic() - started, 3),
                        "terminal_step_receipt": receipt,
                    }
                    if isinstance(receipt.get("terminal_capture"), dict):
                        metrics["terminal_capture"] = deepcopy(
                            receipt["terminal_capture"]
                        )
                    result = primitive_result(
                        primitive_success=primitive_success,
                        task_success=stop_reason == "official_task_success",
                        stop_reason=stop_reason,
                        recoverable=False,
                        metrics=metrics,
                    )
                    if isinstance(receipt.get("terminal_capture"), dict):
                        result["terminal_capture"] = deepcopy(
                            receipt["terminal_capture"]
                        )
                    return result
        except Exception as exc:
            return primitive_result(
                primitive_success=False,
                task_success=self._task_success(),
                stop_reason="rpc_error",
                recoverable=False,
                metrics={
                    "geometric_waypoints": geometric_waypoints,
                    "control_cycles_per_waypoint": (
                        DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT
                    ),
                    "env_actions_sent": sent,
                    "executed_control_cycles": executed_control_cycles,
                    "selected_source_indices": source_indices,
                    "elapsed_s": round(time.monotonic() - started, 3),
                },
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )
        return primitive_result(
            primitive_success=True,
            task_success=False,
            stop_reason="reached",
            recoverable=True,
            metrics={
                "geometric_waypoints": geometric_waypoints,
                "control_cycles_per_waypoint": (DASHBOARD_CONTROL_CYCLES_PER_WAYPOINT),
                "env_actions_sent": sent,
                "executed_control_cycles": executed_control_cycles,
                "selected_source_indices": source_indices,
                "elapsed_s": round(time.monotonic() - started, 3),
            },
        )

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

    def _step_env_action(self, action: np.ndarray) -> dict[str, Any]:
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
        receipt: dict[str, Any] = {
            "raw_success": official_task_success(info),
            "terminated": bool(np.asarray(ret[2]).any()),
            "truncated": bool(np.asarray(ret[3]).any()),
        }
        if isinstance(info, dict):
            rpent = info.get("_rpent")
            for payload in (info, rpent):
                if not isinstance(payload, dict):
                    continue
                terminal_capture = payload.get("terminal_capture")
                if isinstance(terminal_capture, dict):
                    receipt["terminal_capture"] = deepcopy(terminal_capture)
                    break
        return receipt

    def _world_target(self, target_xyz: Any, *, frame: str) -> np.ndarray:
        if str(frame) != "world":
            raise ValueError("planner currently requires frame='world'")
        return _as_xyz(target_xyz)

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


def _reorder_joint_trajectory(
    trajectory: Any,
    *,
    source_names: Any,
    target_names: Any,
) -> np.ndarray:
    """Return a q trajectory in an explicitly requested joint order."""

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
    ):
        raise RuntimeError("joint trajectory name-order mapping is invalid")
    source_index = {name: index for index, name in enumerate(source)}
    return np.ascontiguousarray(
        values[:, [source_index[name] for name in target]],
        dtype=np.float32,
    )


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
