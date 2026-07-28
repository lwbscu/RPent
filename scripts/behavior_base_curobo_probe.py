#!/usr/bin/env python3
"""R1Pro BASE cuRobo probe; planning-only unless --execute-sim-jogs is set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robots.behavior.env_server import (  # noqa: E402
    BehaviorEnvFacade,
    _flush_shutdown_artifacts,
    _load_env_config,
)
from robots.behavior.planner_executor import (  # noqa: E402
    _artifact_jsonable,
    _jsonable,
)
from robots.behavior.schemas import (  # noqa: E402
    BASE_ROTATION_STEP_RAD,
    BASE_TRANSLATION_STEP_M,
    EEF_TRANSLATION_STEP_M,
    TORSO_VERTICAL_STEP_M,
    WRIST_ROTATION_STEP_RAD,
)
from robots.behavior.task_specs import (  # noqa: E402
    BehaviorTaskSpec,
    get_task_spec,
)

# Keep the standalone probe importable without loading the runtime/VLA HTTP stack.
# The production runtime enforces the same immutable native simulator seed.
BEHAVIOR_NATIVE_ENV_SEED = 0


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-instance-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task-name",
        default="turning_on_radio",
        help="Supported BEHAVIOR TaskSpec name used to resolve the native identity.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=BEHAVIOR_NATIVE_ENV_SEED,
        help=(
            "Native simulator RNG seed; BEHAVIOR probes require the official "
            f"fixed value {BEHAVIOR_NATIVE_ENV_SEED}."
        ),
    )
    parser.add_argument("--activity-instance-id", type=int, default=211)
    parser.add_argument("--compatible-only", action="store_true")
    parser.add_argument(
        "--execute-sim-jogs",
        action="store_true",
        help=(
            "After the zero-action planning preflight, explicitly exercise "
            "Dashboard base, arm, wrist, torso, gripper, and synchronized "
            "three-camera paths in BEHAVIOR/OmniGibson. Never use this flag "
            "on a real robot."
        ),
    )
    parser.add_argument("--probe-arm-candidates", action="store_true")
    return parser.parse_args()


def _resolve_probe_identity(
    task_name: str,
    activity_instance_id: int,
) -> tuple[BehaviorTaskSpec, int]:
    """Resolve one public instance exclusively through its task-scoped mapping."""

    task_spec = get_task_spec(str(task_name))
    instance_id = int(activity_instance_id)
    public_seed = task_spec.public_seed_for_instance(instance_id)
    if public_seed is None:
        classification = task_spec.classify_instance(instance_id)
        raise ValueError(
            f"{task_spec.task_name} instance {instance_id} is "
            f"{classification.kind}, not a mapped public instance"
        )
    if task_spec.instance_for_public_seed(public_seed) != instance_id:
        raise RuntimeError("TaskSpec public-seed mapping is not bijective")
    return task_spec, public_seed


def _target_pose_from_base_joints(robot: Any, xyyaw: np.ndarray):
    import torch
    from omnigibson.utils import transform_utils as transform

    base_joints = robot.get_joint_positions()[robot.base_idx]
    pos = torch.tensor(
        [float(xyyaw[0]), float(xyyaw[1]), float(base_joints[2])],
        dtype=torch.float32,
    )
    euler = torch.tensor(
        [float(base_joints[3]), float(base_joints[4]), float(xyyaw[2])],
        dtype=torch.float32,
    )
    return pos, transform.mat2quat(transform.euler_intrinsic2mat(euler))


def _summarize_results(results: Any) -> dict[str, Any]:
    if isinstance(results, tuple):
        successes, _paths = results
        return {"successes": _jsonable(successes)}
    return {
        "batches": [
            {
                "success": _jsonable(result.success),
                "status": str(getattr(result, "status", "unavailable")),
                "valid_query": _jsonable(
                    getattr(result, "valid_query", "unavailable")
                ),
            }
            for result in results
        ]
    }


def _attachment_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"attached": False, "links": [], "roots": []}
    roots = []
    for root in value.values():
        roots.append(
            {
                "prim_path": str(getattr(root, "prim_path", "")) or None,
                "name": str(getattr(root, "name", "")) or None,
                "type": type(root).__name__,
            }
        )
    return {
        "attached": bool(value),
        "links": sorted(str(link) for link in value),
        "roots": roots,
    }


def _safe_probe_value(call: Any) -> tuple[Any, str | None]:
    try:
        return call(), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _official_task_success_latched(
    env: Any,
    result: dict[str, Any] | None = None,
) -> bool:
    """Use only the raw env success bit, its latch, or the returned task_success."""

    info = getattr(env, "_last_info", None)
    info = info if isinstance(info, dict) else {}
    done = info.get("done")
    done = done if isinstance(done, dict) else {}
    result = result if isinstance(result, dict) else {}
    return bool(
        done.get("success") is True
        or getattr(env, "_official_success_latched", False)
        or result.get("task_success") is True
    )


def _quat_multiply_xyzw(left: Any, right: Any) -> np.ndarray:
    left_q = np.asarray(left, dtype=np.float64).reshape(4)
    right_q = np.asarray(right, dtype=np.float64).reshape(4)
    if not np.isfinite(left_q).all() or not np.isfinite(right_q).all():
        raise ValueError("quaternion must be finite")
    lx, ly, lz, lw = left_q
    rx, ry, rz, rw = right_q
    product = np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(product))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion product is degenerate")
    return product / norm


def _axis_angle_quat_xyzw(axis: Any, angle_rad: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float64).reshape(3)
    angle = float(angle_rad)
    norm = float(np.linalg.norm(axis_array))
    if (
        not np.isfinite(axis_array).all()
        or not math.isfinite(angle)
        or not math.isfinite(norm)
        or norm <= 1e-12
    ):
        raise ValueError("axis-angle rotation is invalid")
    unit = axis_array / norm
    half = angle / 2.0
    return np.asarray(
        [*(unit * math.sin(half)), math.cos(half)],
        dtype=np.float64,
    )


def _capture_lineage(env: Any) -> dict[str, Any]:
    cameras: dict[str, Any] = {}
    groups: set[str] = set()
    steps: set[int] = set()
    for camera in ("head", "left_wrist", "right_wrist"):
        try:
            frame = env._frame_cache.latest(camera)
            cameras[camera] = {
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "simulator_step": int(frame.step_index),
                "timestamp_s": float(frame.timestamp_s),
            }
            if isinstance(frame.capture_group_id, str):
                groups.add(frame.capture_group_id)
            steps.add(int(frame.step_index))
        except Exception as exc:
            cameras[camera] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "cameras": cameras,
        "capture_group_ids": sorted(groups),
        "simulator_steps": sorted(steps),
        "synchronized": len(groups) == 1 and len(steps) == 1,
    }


def _runtime_snapshot(env: Any, backend: Any) -> dict[str, Any]:
    robot, robot_error = _safe_probe_value(backend._find_robot)
    eef = {}
    for hand in ("left", "right"):
        pose, error = _safe_probe_value(lambda hand=hand: backend.get_eef_pose(hand))
        eef[hand] = (
            {"error": error}
            if error is not None
            else (
                None
                if pose is None
                else {
                    "position": _jsonable(pose[0]),
                    "quaternion_xyzw": _jsonable(pose[1]),
                }
            )
        )
    torso_pose: dict[str, Any] | None = None
    if robot is not None:
        torso_link = (getattr(robot, "links", {}) or {}).get("torso_link4")
        if torso_link is not None:
            value, error = _safe_probe_value(torso_link.get_position_orientation)
            torso_pose = (
                {"link": "torso_link4", "error": error}
                if error is not None
                else {
                    "link": "torso_link4",
                    "position": _jsonable(value[0]),
                    "quaternion_xyzw": _jsonable(value[1]),
                }
            )
    joint_state: dict[str, Any]
    if robot is None:
        joint_state = {"error": robot_error or "robot unavailable"}
    else:
        joint_positions, error = _safe_probe_value(robot.get_joint_positions)
        if error is not None:
            joint_state = {"error": error}
        else:
            q = np.asarray(_jsonable(joint_positions), dtype=np.float32)
            joint_state = {
                "shape": list(q.shape),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(q).tobytes()
                ).hexdigest(),
            }
    latch = getattr(env, "_gripper_latch", {})
    latch = latch if isinstance(latch, dict) else {}
    attachments: dict[str, Any] = {}
    for hand in ("left", "right"):
        attached, error = _safe_probe_value(
            lambda hand=hand: backend.get_attached_object(hand)
        )
        attachments[hand] = (
            {"attached": None, "error": error}
            if error is not None
            else {
                **_attachment_summary(attached),
                "error": None,
            }
        )
    base_pose, base_error = _safe_probe_value(backend.get_base_pose)
    return {
        "env_steps": int(getattr(env, "_env_steps", -1)),
        "base_world_xyyaw": (
            {"error": base_error}
            if base_error is not None
            else _jsonable(base_pose)
        ),
        "eef_world_pose": eef,
        "torso_world_pose": torso_pose,
        "joint_state": joint_state,
        "gripper_latch": {
            hand: (
                float(latch[hand])
                if hand in latch and latch[hand] is not None
                else None
            )
            for hand in ("left", "right")
        },
        "attachments": attachments,
        "capture_lineage": _capture_lineage(env),
        "official_success_latched": bool(
            getattr(env, "_official_success_latched", False)
        ),
        "controller": {
            "mode": str(getattr(env, "_controller_mode", "unavailable")),
            "state": str(getattr(env, "_controller_state", "unavailable")),
            "base_controller_mode": str(
                getattr(env, "_base_controller_mode", "unavailable")
            ),
        },
    }


def _result_evidence(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"result": _artifact_jsonable(result)}
    metrics = dict(result.get("metrics") or {})
    return {
        "primitive_success": result.get("primitive_success"),
        "task_success": result.get("task_success"),
        "stop_reason": result.get("stop_reason"),
        "recoverable": result.get("recoverable"),
        "env_actions_sent": metrics.get("env_actions_sent"),
        "partial_motion": metrics.get("partial_motion"),
        "collision_certificate": (
            metrics.get("base_trajectory_certificate")
            or metrics.get("whole_body_certificate")
            or result.get("torso_certificate")
        ),
        "collision_admission": metrics.get("collision_admission"),
        "tracking": {
            "final_joint_tracking": metrics.get("final_joint_tracking"),
            "whole_body_execution": metrics.get("whole_body_execution"),
            "navigation_isolation": metrics.get("navigation_isolation"),
            "tracking_validation": metrics.get("tracking_validation"),
            "tracking_error": metrics.get("tracking_error"),
            "tracking_error_max": metrics.get("tracking_error_max"),
            "final_position_error_m": metrics.get("final_position_error_m"),
            "final_orientation_error_rad": metrics.get(
                "final_orientation_error_rad"
            ),
        },
        "capture_lineage": {
            "capture_group_id": result.get("capture_group_id"),
            "simulator_step": result.get("simulator_step"),
            "frame_ids": result.get("frame_ids"),
            "camera_keys": sorted(
                str(camera)
                for camera in dict(result.get("_frames_bytes") or {})
            ),
        },
        "result": _artifact_jsonable(result),
    }


def _execute_manual_probe_case(
    env: Any,
    backend: Any,
    *,
    name: str,
    target: str,
    action: str,
    camera: str,
    expect_fail_closed: bool = False,
) -> dict[str, Any]:
    before = _runtime_snapshot(env, backend)
    started = time.monotonic()
    try:
        result = env.dashboard_manual_command(
            target=target,
            action=action,
            camera=camera,
        )
        error = None
    except Exception as exc:
        result = None
        error = f"{type(exc).__name__}: {exc}"
    after = _runtime_snapshot(env, backend)
    env_step_delta = int(after["env_steps"]) - int(before["env_steps"])
    fail_closed_observed = bool(
        expect_fail_closed
        and env_step_delta == 0
        and (
            error is not None
            or (
                isinstance(result, dict)
                and result.get("primitive_success") is False
            )
        )
    )
    return {
        "name": name,
        "target": target,
        "action": action,
        "camera": camera,
        "expected_fail_closed": bool(expect_fail_closed),
        "fail_closed_observed": fail_closed_observed,
        "elapsed_s": round(time.monotonic() - started, 3),
        "env_step_delta": env_step_delta,
        "before": before,
        "after": after,
        "error": error,
        "evidence": _result_evidence(result),
        "real_robot_deployment_allowed": False,
    }


def _plan_only_arm_probe(
    env: Any,
    backend: Any,
    *,
    hand: str,
) -> dict[str, Any]:
    before = _runtime_snapshot(env, backend)
    started = time.monotonic()
    try:
        base = np.asarray(backend.get_base_pose(), dtype=np.float64).reshape(3)
        eef_position, eef_quat = backend.get_eef_pose(hand)
        eef_position = np.asarray(eef_position, dtype=np.float64).reshape(3)
        requested_delta_world = np.array(
            [
                math.cos(float(base[2])) * EEF_TRANSLATION_STEP_M,
                math.sin(float(base[2])) * EEF_TRANSLATION_STEP_M,
                0.0,
            ],
            dtype=np.float64,
        )
        target = eef_position + requested_delta_world
        result = env._planner.move_to(
            hand=hand,
            target_xyz=target,
            frame="world",
            target_quat_xyzw=eef_quat,
            plan_only=True,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=math.radians(1.0),
            timeout_s=60.0,
        )
        error = None
    except Exception as exc:
        requested_delta_world = None
        target = None
        result = None
        error = f"{type(exc).__name__}: {exc}"
    after = _runtime_snapshot(env, backend)
    env_step_delta = int(after["env_steps"]) - int(before["env_steps"])
    return {
        "name": f"{hand}_arm_forward_3cm_plan_only",
        "hand": hand,
        "requested_delta_base": [EEF_TRANSLATION_STEP_M, 0.0, 0.0],
        "requested_delta_world": _jsonable(requested_delta_world),
        "target_world_xyz": _jsonable(target),
        "elapsed_s": round(time.monotonic() - started, 3),
        "env_step_delta": env_step_delta,
        "zero_action_verified": env_step_delta == 0,
        "before": before,
        "after": after,
        "error": error,
        "evidence": _result_evidence(result),
        "real_robot_deployment_allowed": False,
    }


def _plan_only_torso_probe(
    env: Any,
    backend: Any,
    *,
    action: str,
) -> dict[str, Any]:
    """Plan one torso world-Z jog without admitting it for public execution."""

    action = str(action)
    if action not in {"up", "down"}:
        raise ValueError("planning-only torso action must be up or down")
    if _official_task_success_latched(env):
        return {
            **_skipped_after_success(
                name=f"torso_{action}_3cm_plan_only",
                target="chassis",
                action=action,
                camera="head",
            ),
            "mode": "planning_only",
            "release_admission": False,
        }
    before = _runtime_snapshot(env, backend)
    started = time.monotonic()
    delta_z = TORSO_VERTICAL_STEP_M * (1.0 if action == "up" else -1.0)
    try:
        start_pose = backend.get_torso_pose()
        start_position = np.asarray(start_pose[0], dtype=np.float64).reshape(3)
        target_z = float(start_position[2] + delta_z)
        result = backend.plan_torso_trajectory(
            target_z_m=target_z,
            timeout_s=60.0,
            start_torso_pose=start_pose,
        )
        error = None
    except Exception as exc:
        start_position = None
        target_z = None
        result = None
        error = f"{type(exc).__name__}: {exc}"
    after = _runtime_snapshot(env, backend)
    env_step_delta = int(after["env_steps"]) - int(before["env_steps"])
    metrics = (
        dict(result.get("metrics") or {}) if isinstance(result, dict) else {}
    )
    planner_actions = metrics.get("env_actions_sent")
    before_digest = before["joint_state"].get("sha256")
    zero_action = bool(
        env_step_delta == 0
        and (planner_actions is None or planner_actions == 0)
        and isinstance(before_digest, str)
        and after["joint_state"].get("sha256") == before_digest
    )
    return {
        "name": f"torso_{action}_3cm_plan_only",
        "mode": "planning_only",
        "target": "chassis",
        "action": action,
        "target_link": "torso_link4",
        "requested_delta_z_m": delta_z,
        "start_world_xyz": _jsonable(start_position),
        "target_world_z_m": target_z,
        "elapsed_s": round(time.monotonic() - started, 3),
        "env_step_delta": env_step_delta,
        "planner_env_actions_sent": planner_actions,
        "zero_action_verified": zero_action,
        "plan_admitted": bool(
            isinstance(result, dict) and result.get("ok") is True
        ),
        # A collision-certified plan is not a public-control calibration.
        "release_admission": False,
        "release_admission_reason": (
            "planning-only evidence cannot enable a public motion capability"
        ),
        "before": before,
        "after": after,
        "error": error,
        "evidence": _result_evidence(result),
        "real_robot_deployment_allowed": False,
    }


def _plan_only_wrist_probe(
    env: Any,
    backend: Any,
    *,
    hand: str,
    action: str,
) -> dict[str, Any]:
    """Plan one camera-relative wrist turn with zero simulator actions."""

    hand = str(hand)
    action = str(action)
    if hand not in {"left", "right"}:
        raise ValueError("planning-only wrist hand must be left or right")
    if action not in {"rotate_left", "rotate_right"}:
        raise ValueError(
            "planning-only wrist action must be rotate_left or rotate_right"
        )
    visual_direction = (
        "counterclockwise" if action == "rotate_left" else "clockwise"
    )
    if _official_task_success_latched(env):
        return {
            **_skipped_after_success(
                name=f"{hand}_wrist_visual_{visual_direction}_5deg_plan_only",
                target=f"{hand}_arm",
                action=action,
                camera=f"{hand}_wrist",
            ),
            "mode": "planning_only",
            "release_admission": False,
        }
    before = _runtime_snapshot(env, backend)
    started = time.monotonic()
    calibration: dict[str, Any] | None = None
    start_position: np.ndarray | None = None
    start_quaternion: np.ndarray | None = None
    signed_angle: float | None = None
    target_quaternion: np.ndarray | None = None
    result: dict[str, Any] | None = None
    try:
        calibration = env._planner._wrist_camera_rotation_calibration(hand)
        if not isinstance(calibration, dict) or calibration.get("verified") is not True:
            raise RuntimeError("wrist planning calibration is unavailable")
        start_position, start_quaternion = backend.get_eef_pose(hand)
        start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
        start_quaternion = np.asarray(
            start_quaternion, dtype=np.float64
        ).reshape(4)
        signed_angle = (
            float(calibration["visual_ccw_angle_sign"])
            * WRIST_ROTATION_STEP_RAD
            * (1.0 if action == "rotate_left" else -1.0)
        )
        target_quaternion = _quat_multiply_xyzw(
            start_quaternion,
            _axis_angle_quat_xyzw(
                calibration["screen_normal_axis_eef"],
                signed_angle,
            ),
        )
        result = env._planner.move_to(
            hand=hand,
            target_xyz=start_position,
            frame="world",
            target_quat_xyzw=target_quaternion,
            plan_only=True,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=math.radians(1.0),
            timeout_s=60.0,
        )
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    after = _runtime_snapshot(env, backend)
    env_step_delta = int(after["env_steps"]) - int(before["env_steps"])
    metrics = (
        dict(result.get("metrics") or {}) if isinstance(result, dict) else {}
    )
    planner_actions = metrics.get("env_actions_sent")
    before_digest = before["joint_state"].get("sha256")
    zero_action = bool(
        env_step_delta == 0
        and (planner_actions is None or planner_actions == 0)
        and isinstance(before_digest, str)
        and after["joint_state"].get("sha256") == before_digest
    )
    return {
        "name": f"{hand}_wrist_visual_{visual_direction}_5deg_plan_only",
        "mode": "planning_only",
        "target": f"{hand}_arm",
        "hand": hand,
        "action": action,
        "camera": f"{hand}_wrist",
        "visual_direction": visual_direction,
        "requested_rotation_rad": signed_angle,
        "start_world_xyz": _jsonable(start_position),
        "start_quaternion_xyzw": _jsonable(start_quaternion),
        "target_quaternion_xyzw": _jsonable(target_quaternion),
        "calibration": _artifact_jsonable(calibration),
        "elapsed_s": round(time.monotonic() - started, 3),
        "env_step_delta": env_step_delta,
        "planner_env_actions_sent": planner_actions,
        "zero_action_verified": zero_action,
        "plan_admitted": bool(
            isinstance(result, dict)
            and result.get("primitive_success") is True
        ),
        # Per-hand live image evidence must separately admit this capability.
        "release_admission": False,
        "release_admission_reason": (
            "planning-only camera geometry is not a real visual rotation probe"
        ),
        "before": before,
        "after": after,
        "error": error,
        "evidence": _result_evidence(result),
        "real_robot_deployment_allowed": False,
    }


def _skipped_after_success(
    *,
    name: str,
    target: str,
    action: str,
    camera: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "target": target,
        "action": action,
        "camera": camera,
        "skipped": True,
        "skip_reason": "official_task_success_latched",
        "env_step_delta": 0,
        "no_follow_up_rpc": True,
        "real_robot_deployment_allowed": False,
    }


def _probe_call(
    generator: Any,
    *,
    robot: Any,
    emb_sel: Any,
    name: str,
    target_xyyaw: np.ndarray,
    ik_only: bool,
    world_collision: bool,
    exact_starter_budget: bool = False,
) -> dict[str, Any]:
    import torch

    pos, quat = _target_pose_from_base_joints(robot, target_xyyaw)
    batch_size = int(generator.batch_size)
    target_pos = {
        robot.base_footprint_link_name: torch.stack([pos] * batch_size)
    }
    target_quat = {
        robot.base_footprint_link_name: torch.stack([quat] * batch_size)
    }
    started = time.monotonic()
    try:
        results = generator.compute_trajectories(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=None,
            is_local=False,
            max_attempts=(math.ceil(100 / batch_size) if exact_starter_budget else 5),
            timeout=(60.0 if exact_starter_budget else 10.0),
            ik_fail_return=(50 if exact_starter_budget else 5),
            enable_finetune_trajopt=not ik_only,
            finetune_attempts=(1 if not ik_only else 0),
            return_full_result=True,
            success_ratio=1.0 / batch_size,
            attached_obj=None,
            attached_obj_scale=None,
            motion_constraint=None,
            skip_obstacle_update=True,
            ik_only=ik_only,
            ik_world_collision_check=world_collision,
            emb_sel=emb_sel,
        )
        outcome = {"ok": True, **_summarize_results(results)}
    except Exception as exc:
        outcome = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "name": name,
        "target_xyyaw": target_xyyaw.tolist(),
        "target_position": _jsonable(pos),
        "target_quaternion_xyzw": _jsonable(quat),
        "ik_only": ik_only,
        "world_collision": world_collision,
        "exact_starter_budget": exact_starter_budget,
        "elapsed_s": round(time.monotonic() - started, 3),
        **outcome,
    }


def main() -> None:
    args = _args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "base_curobo_probe.json"
    events_path = output_dir / "base_curobo_probe.events.jsonl"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "stage": "initialize_report",
        "cases": [],
        "process": {
            "pid": os.getpid(),
            "pgid": _safe_probe_value(lambda: os.getpgid(0))[0],
            "sid": _safe_probe_value(lambda: os.getsid(0))[0],
            "ports": [],
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "commit": None,
        "worktree_dirty": None,
        "requested_configuration": {
            "task_name": args.task_name,
            "activity_instance_id": int(args.activity_instance_id),
            "activity_instance_dir": str(
                Path(args.activity_instance_dir).expanduser().resolve()
            ),
            "native_env_seed": int(args.seed),
            "attempt_index": 1,
        },
        "configuration": None,
        "planning_only": not bool(args.execute_sim_jogs),
        "simulation_execution_explicitly_enabled": bool(args.execute_sim_jogs),
        "structured_log_path": str(events_path),
    }

    def save() -> None:
        temporary = report_path.with_suffix(f"{report_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(report_path)

    def log_event(event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "timestamp_unix_s": time.time(),
            "pid": os.getpid(),
            **_artifact_jsonable(fields),
        }
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def record_failure(stage: str, exc: Exception) -> None:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        report.update(
            {
                "status": "failed",
                "stage": stage,
                "failed_stage": stage,
                "failure": failure,
            }
        )
        save()
        log_event("probe_failed", stage=stage, failure=failure)
        print(
            json.dumps(
                {
                    "event": "probe_failed",
                    "stage": stage,
                    "failure": failure,
                    "report_path": str(report_path),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    lifecycle_sealed = False

    def seal_lifecycle(facade: BehaviorEnvFacade | None) -> None:
        """Seal Python-owned artifacts without invoking a motion primitive."""

        nonlocal lifecycle_sealed
        if lifecycle_sealed:
            return
        lifecycle_sealed = True
        before_steps = (
            int(getattr(facade, "_env_steps", -1)) if facade is not None else None
        )
        cleanup: dict[str, Any] = {
            "method": (
                "BehaviorEnvFacade.shutdown"
                if facade is not None
                else "no_facade_constructed"
            ),
            "gripper_close_called": False,
            "env_steps_before": before_steps,
            "env_steps_after": before_steps,
            "env_step_delta": 0,
            "shutdown_called": False,
            "artifacts_flushed": False,
        }
        cleanup_error: dict[str, Any] | None = None
        if facade is not None:
            try:
                shutdown = getattr(facade, "shutdown", None)
                if not callable(shutdown):
                    raise RuntimeError(
                        "BehaviorEnvFacade lifecycle shutdown is unavailable"
                    )
                shutdown()
                cleanup["shutdown_called"] = True
                after_steps = int(getattr(facade, "_env_steps", -1))
                cleanup["env_steps_after"] = after_steps
                cleanup["env_step_delta"] = after_steps - int(before_steps)
                if cleanup["env_step_delta"] != 0:
                    raise RuntimeError(
                        "BehaviorEnvFacade.shutdown advanced the simulator"
                    )
            except Exception as exc:
                cleanup_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
        try:
            _flush_shutdown_artifacts(output_dir)
            cleanup["artifacts_flushed"] = True
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
        report["cleanup"] = cleanup
        if cleanup_error is not None:
            report["cleanup_error"] = cleanup_error
        save()
        log_event("probe_lifecycle_sealed", cleanup=cleanup)

    save()
    log_event(
        "probe_report_initialized",
        report_path=str(report_path),
        requested_configuration=report["requested_configuration"],
    )
    env: BehaviorEnvFacade | None = None
    construction_stage = "resolve_task_identity"
    try:
        if int(args.seed) != BEHAVIOR_NATIVE_ENV_SEED:
            raise ValueError(
                "BEHAVIOR probe native env seed must be "
                f"{BEHAVIOR_NATIVE_ENV_SEED}, got {args.seed}"
            )
        task_spec, public_seed = _resolve_probe_identity(
            args.task_name,
            args.activity_instance_id,
        )
        env_args = SimpleNamespace(
            suite="behavior_2025_challenge",
            task=task_spec.task_index,
            task_name=task_spec.task_name,
            activity_definition_id=task_spec.activity_definition_id,
            activity_instance_id=args.activity_instance_id,
            activity_instance_dir=args.activity_instance_dir,
            scene_model=task_spec.scene_model,
            seed=BEHAVIOR_NATIVE_ENV_SEED,
            public_seed=public_seed,
            attempt_index=1,
            max_episode_steps=24756,
            output_dir=str(output_dir),
            config_path=None,
            controller_mode="hybrid",
        )
        meta = vars(env_args).copy()
        report.update(
            {
                "stage": "identity_resolved",
                "configuration": meta,
                "task_identity": {
                    "task_name": task_spec.task_name,
                    "task_index": task_spec.task_index,
                    "activity_definition_id": task_spec.activity_definition_id,
                    "activity_instance_id": int(args.activity_instance_id),
                    "public_seed": public_seed,
                    "recipe_tag": task_spec.tag(public_seed),
                    "mapping_version": task_spec.mapping_version,
                    "classification": asdict(
                        task_spec.classify_instance(int(args.activity_instance_id))
                    ),
                },
            }
        )
        commit, commit_error = _safe_probe_value(
            lambda: subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        )
        worktree, worktree_error = _safe_probe_value(
            lambda: bool(
                subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                    text=True,
                ).strip()
            )
        )
        report["commit"] = commit
        report["worktree_dirty"] = worktree
        report["metadata_errors"] = {
            "commit": commit_error,
            "worktree_dirty": worktree_error,
        }
        save()
        log_event(
            "task_identity_resolved",
            task_identity=report["task_identity"],
            configuration=meta,
        )
        construction_stage = "load_env_config"
        report["stage"] = construction_stage
        save()
        cfg = _load_env_config(env_args)
        construction_stage = "construct_facade"
        report["stage"] = construction_stage
        save()
        env = BehaviorEnvFacade(
            cfg=cfg,
            meta=meta,
            output_dir=output_dir,
        )
    except Exception as exc:
        record_failure(construction_stage, exc)
        seal_lifecycle(None)
        raise

    try:
        report["stage"] = "reset"
        save()
        env.reset()
        if _official_task_success_latched(env):
            report["integration_preflight"] = {
                "mode": "planning_only",
                "skipped": True,
                "skip_reason": "official_task_success_latched",
                "post_success_follow_up_rpc_count": 0,
                "zero_action_verified": True,
                "real_robot_deployment_allowed": False,
            }
            report["status"] = "complete"
            report["stage"] = "complete"
            save()
            log_event(
                "probe_complete",
                report_path=str(report_path),
                official_success_stop=True,
            )
            return
        report["stage"] = "planning_probe"
        backend = env._planner.backend
        robot = backend._find_robot()
        current = backend.get_base_pose()
        q = np.asarray(_jsonable(robot.get_joint_positions()), dtype=np.float64)
        from omnigibson.action_primitives.curobo import (
            CuRoboEmbodimentSelection,
            CuRoboMotionGenerator,
        )
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            m as starter_macros,
        )

        emb_sel = CuRoboEmbodimentSelection.BASE
        if args.compatible_only:
            generator = backend._generator(kind="base")
            report["generator_mode"] = "rpent_scene_workspace_official_base"
            report["base_prismatic_workspace_limit_m"] = (
                backend._base_workspace_limit_m
            )
        else:
            generator = CuRoboMotionGenerator(
                robot=robot,
                batch_size=3,
                collision_activation_distance=(
                    starter_macros.DEFAULT_COLLISION_ACTIVATION_DISTANCE
                ),
            )
            report["generator_mode"] = "official_starter_exact"
        generator.update_obstacles()
        kinematics = generator.mg[emb_sel].kinematics
        report.update(
            {
                "status": "running",
                "current_base_world_xyyaw": current.tolist(),
                "base_idx": _jsonable(robot.base_idx),
                "base_control_idx": _jsonable(robot.base_control_idx),
                "current_base_joint_values": _jsonable(
                    robot.get_joint_positions()[robot.base_idx]
                ),
                "robot_joint_names": list(generator.robot_joint_names),
                "base_link": str(generator.base_link[emb_sel]),
                "eef_link": str(generator.ee_link[emb_sel]),
                "kinematics_joint_names": list(kinematics.joint_names),
                "kinematics_joint_limits": _jsonable(
                    kinematics.kinematics_config.joint_limits.position
                ),
                "current_q_shape": list(q.shape),
            }
        )
        save()
        targets = [
            ("current_ik_collision", current.copy(), True, True, False),
            ("current_ik_no_world_collision", current.copy(), True, False, False),
            ("x_plus_5cm_ik_collision", current + [0.05, 0.0, 0.0], True, True, False),
            ("y_plus_5cm_ik_collision", current + [0.0, 0.05, 0.0], True, True, False),
            ("yaw_plus_3deg_ik_collision", current + [0.0, 0.0, math.radians(3)], True, True, False),
            ("x_plus_5cm_full", current + [0.05, 0.0, 0.0], False, True, False),
            ("yaw_plus_3deg_full", current + [0.0, 0.0, math.radians(3)], False, True, False),
            (
                "known_candidate_exact_starter_full",
                np.asarray([5.6047677074, 5.1226747401, -0.2617993878]),
                False,
                True,
                True,
            ),
        ]
        if args.compatible_only:
            targets = [
                target
                for target in targets
                if target[0]
                in {
                    "current_ik_collision",
                    "x_plus_5cm_ik_collision",
                    "x_plus_5cm_full",
                    "yaw_plus_3deg_full",
                    "known_candidate_exact_starter_full",
                }
            ]
            targets[-1] = (*targets[-1][:-1], False)
        for name, target, ik_only, world_collision, exact_budget in targets:
            report["cases"].append(
                _probe_call(
                    generator,
                    robot=robot,
                    emb_sel=emb_sel,
                    name=name,
                    target_xyyaw=np.asarray(target, dtype=np.float64),
                    ik_only=ik_only,
                    world_collision=world_collision,
                    exact_starter_budget=exact_budget,
                )
            )
            save()
        report["formal_relative_jog_plans"] = []
        formal_jogs = (
            (
                "forward",
                {
                    "kind": "translation",
                    "direction": "forward",
                    "distance_m": BASE_TRANSLATION_STEP_M,
                },
            ),
            (
                "backward",
                {
                    "kind": "translation",
                    "direction": "backward",
                    "distance_m": BASE_TRANSLATION_STEP_M,
                },
            ),
            (
                "turn_left",
                {
                    "kind": "rotation",
                    "direction": "left",
                    "angle_deg": math.degrees(BASE_ROTATION_STEP_RAD),
                },
            ),
            (
                "turn_right",
                {
                    "kind": "rotation",
                    "direction": "right",
                    "angle_deg": math.degrees(BASE_ROTATION_STEP_RAD),
                },
            ),
        )
        preflight_start = _runtime_snapshot(env, backend)
        for action, relative_motion in formal_jogs:
            plan_started = time.monotonic()
            before_step = int(getattr(env, "_env_steps", -1))
            plan = backend.plan_relative_navigation_trajectory(
                relative_motion=relative_motion,
                timeout_s=45.0,
            )
            after_step = int(getattr(env, "_env_steps", -1))
            report["formal_relative_jog_plans"].append(
                {
                    "action": action,
                    "relative_motion": relative_motion,
                    "elapsed_s": round(time.monotonic() - plan_started, 3),
                    "ok": bool(plan.get("ok", False)),
                    "stop_reason": plan.get("stop_reason"),
                    "base_goal": _jsonable(plan.get("base_goal")),
                    "trajectory_waypoints": (
                        len(plan["joint_trajectory"])
                        if plan.get("joint_trajectory") is not None
                        else 0
                    ),
                    "collision_admission": _jsonable(
                        dict(plan.get("metrics") or {}).get(
                            "collision_admission"
                        )
                    ),
                    "base_trajectory_certificate": _jsonable(
                        dict(plan.get("metrics") or {}).get(
                            "base_trajectory_certificate"
                        )
                    ),
                    "env_step_delta": after_step - before_step,
                    "zero_action_verified": after_step == before_step,
                }
            )
            save()
        arm_preflight = [
            _plan_only_arm_probe(env, backend, hand=hand)
            for hand in ("left", "right")
        ]
        torso_preflight = [
            _plan_only_torso_probe(env, backend, action=action)
            for action in ("up", "down")
        ]
        wrist_sync_before_step = int(getattr(env, "_env_steps", -1))
        env._refresh_observation_without_step(
            synchronize_hand_geometry=True
        )
        wrist_sync_after_step = int(getattr(env, "_env_steps", -1))
        if wrist_sync_after_step != wrist_sync_before_step:
            raise RuntimeError(
                "wrist planning capture advanced the simulator"
            )
        wrist_preflight = {
            hand: [
                _plan_only_wrist_probe(
                    env,
                    backend,
                    hand=hand,
                    action=action,
                )
                for action in ("rotate_left", "rotate_right")
            ]
            for hand in ("left", "right")
        }
        capabilities = env.dashboard_control_capabilities()
        preflight_end = _runtime_snapshot(env, backend)
        report["integration_preflight"] = {
            "mode": "planning_only",
            "before": preflight_start,
            "after": preflight_end,
            "base": report["formal_relative_jog_plans"],
            "arm": arm_preflight,
            "wrist": {
                "planning_only": wrist_preflight,
                "left": _artifact_jsonable(
                    dict(capabilities.get("planner") or {})
                    .get("wrist", {})
                    .get("left")
                ),
                "right": _artifact_jsonable(
                    dict(capabilities.get("planner") or {})
                    .get("wrist", {})
                    .get("right")
                ),
                "left_available": bool(
                    dict(
                        capabilities.get("wrist_rotation_available") or {}
                    ).get("left")
                ),
                "right_available": bool(
                    dict(
                        capabilities.get("wrist_rotation_available") or {}
                    ).get("right")
                ),
                "planning_phase_public_capability_blocked": all(
                    not bool(
                        dict(
                            capabilities.get("wrist_rotation_available") or {}
                        ).get(hand)
                    )
                    for hand in ("left", "right")
                ),
                "env_step_delta": 0,
                "render_sync_env_step_delta": (
                    wrist_sync_after_step - wrist_sync_before_step
                ),
                "release_admission": False,
                "admission_policy": (
                    "planning-only camera geometry remains blocked from public "
                    "release until each hand has an independent real visual "
                    "rotation probe"
                ),
            },
            "torso": {
                "planning_only": torso_preflight,
                "available": bool(capabilities.get("torso_available")),
                "planner": _artifact_jsonable(
                    dict(capabilities.get("planner") or {}).get("torso")
                ),
                "env_step_delta": 0,
                "release_admission": False,
                "admission_policy": (
                    "planning-only evidence does not bypass the Dashboard "
                    "capability gate"
                ),
            },
            "gripper": {
                "available": _artifact_jsonable(
                    capabilities.get("gripper_available")
                ),
                "latch": preflight_end["gripper_latch"],
                "attachments": preflight_end["attachments"],
                "env_step_delta": 0,
                "planning_only_action": "none",
            },
            "three_camera_sync": preflight_end["capture_lineage"],
            "control_capabilities": _artifact_jsonable(capabilities),
            "env_step_delta": (
                int(preflight_end["env_steps"])
                - int(preflight_start["env_steps"])
            ),
            "planning_admission": {
                "torso_all_plans_admitted": all(
                    bool(item.get("plan_admitted"))
                    for item in torso_preflight
                ),
                "wrist_all_plans_admitted": all(
                    bool(item.get("plan_admitted"))
                    for hand_cases in wrist_preflight.values()
                    for item in hand_cases
                ),
                "public_release_admitted": False,
            },
            "zero_action_verified": (
                int(preflight_end["env_steps"])
                == int(preflight_start["env_steps"])
                and all(
                    bool(item.get("zero_action_verified"))
                    for item in report["formal_relative_jog_plans"]
                )
                and all(
                    bool(item.get("zero_action_verified"))
                    for item in arm_preflight
                )
                and all(
                    bool(item.get("zero_action_verified"))
                    for item in torso_preflight
                )
                and all(
                    bool(item.get("zero_action_verified"))
                    for hand_cases in wrist_preflight.values()
                    for item in hand_cases
                )
            ),
            "real_robot_deployment_allowed": False,
        }
        save()
        if args.probe_arm_candidates:
            surface = np.asarray([6.4258046597, 4.9026785517, 0.9806372991])
            known_candidate = np.asarray([5.6047677074, 5.1226747401, -0.2617993878])
            direction = known_candidate[:2] - surface[:2]
            direction /= np.linalg.norm(direction)
            report["arm_candidate_probes"] = []
            probe_index = 0
            for standoff_m in (0.85, 0.75, 0.65):
                candidate = np.asarray(
                    [
                        surface[0] + direction[0] * standoff_m,
                        surface[1] + direction[1] * standoff_m,
                        math.atan2(-direction[1], -direction[0]),
                    ]
                )
                for clearance_m in (0.08, 0.15, 0.25):
                    target = backend._candidate_reachability_target(
                        surface,
                        candidate,
                        clearance_m=clearance_m,
                    )
                    started = time.monotonic()
                    reachable, reason, metrics = backend.check_arm_reachability(
                        hand="left",
                        target_xyz=target,
                        target_quat_xyzw=None,
                        base_xyyaw=candidate,
                        timeout_s=8.0,
                        skip_obstacle_update=probe_index > 0,
                    )
                    probe_index += 1
                    report["arm_candidate_probes"].append(
                        {
                            "standoff_m": standoff_m,
                            "clearance_m": clearance_m,
                            "candidate_xyyaw": candidate.tolist(),
                            "target_xyz": target.tolist(),
                            "reachable": reachable,
                            "reason": reason,
                            "elapsed_s": round(time.monotonic() - started, 3),
                            "metrics": metrics,
                        }
                    )
                    save()
        if args.execute_sim_jogs:
            if (
                env._meta.get("suite") != "behavior_2025_challenge"
                or env._controller_mode != "hybrid"
            ):
                raise RuntimeError(
                    "--execute-sim-jogs requires BEHAVIOR hybrid simulation"
                )
            execution_capabilities = env.dashboard_control_capabilities()
            wrist_available = dict(
                execution_capabilities.get("wrist_rotation_available") or {}
            )
            release_motion_gate = {
                "torso_available": (
                    execution_capabilities.get("torso_available") is True
                ),
                "left_wrist_available": wrist_available.get("left") is True,
                "right_wrist_available": wrist_available.get("right") is True,
            }
            if not all(release_motion_gate.values()):
                report["integration_execution"] = {
                    "mode": "explicit_behavior_simulation",
                    "blocked": True,
                    "block_reason": (
                        "torso/wrist public release capability is not admitted; "
                        "planning-only evidence cannot bridge the real visual "
                        "probe gate"
                    ),
                    "release_motion_gate": release_motion_gate,
                    "capabilities": _artifact_jsonable(execution_capabilities),
                    "ordered_cases": [],
                    "env_step_delta": 0,
                    "post_success_follow_up_rpc_count": 0,
                    "real_robot_deployment_allowed": False,
                }
                report["status"] = "complete"
                report["stage"] = "complete"
                save()
                log_event(
                    "probe_complete",
                    report_path=str(report_path),
                    execution_blocked=True,
                )
                return
            report["integration_execution"] = {
                "mode": "explicit_behavior_simulation",
                "ordered_cases": [],
                "real_robot_deployment_allowed": False,
            }
            case_specs: list[dict[str, Any]] = [
                {
                    "name": "torso_link4_world_up_3cm",
                    "target": "chassis",
                    "action": "up",
                    "camera": "head",
                    "capability": "torso_available",
                },
                {
                    "name": "torso_link4_world_down_3cm",
                    "target": "chassis",
                    "action": "down",
                    "camera": "head",
                    "capability": "torso_available",
                },
                {
                    "name": "left_wrist_visual_counterclockwise_5deg",
                    "target": "left_arm",
                    "action": "rotate_left",
                    "camera": "left_wrist",
                    "capability": ("wrist_rotation_available", "left"),
                },
                {
                    "name": "left_wrist_visual_clockwise_5deg",
                    "target": "left_arm",
                    "action": "rotate_right",
                    "camera": "left_wrist",
                    "capability": ("wrist_rotation_available", "left"),
                },
                {
                    "name": "right_wrist_visual_counterclockwise_5deg",
                    "target": "right_arm",
                    "action": "rotate_left",
                    "camera": "right_wrist",
                    "capability": ("wrist_rotation_available", "right"),
                },
                {
                    "name": "right_wrist_visual_clockwise_5deg",
                    "target": "right_arm",
                    "action": "rotate_right",
                    "camera": "right_wrist",
                    "capability": ("wrist_rotation_available", "right"),
                },
                {
                    "name": "base_forward_5cm",
                    "target": "chassis",
                    "action": "forward",
                    "camera": "head",
                    "capability": "base_available",
                },
                {
                    "name": "left_arm_forward_3cm",
                    "target": "left_arm",
                    "action": "forward",
                    "camera": "left_wrist",
                    "capability": ("eef_available", "left"),
                },
            ]
            left_attached = bool(
                preflight_end["attachments"]["left"].get("attached")
            )
            case_specs.append(
                {
                    "name": (
                        "left_gripper_hold_close"
                        if left_attached
                        else "left_gripper_open"
                    ),
                    "target": "left_arm",
                    "action": "close" if left_attached else "open",
                    "camera": "left_wrist",
                    "capability": ("gripper_available", "left"),
                }
            )
            case_specs.append(
                {
                    "name": "synchronized_three_camera_observe",
                    "target": "chassis",
                    "action": "observe",
                    "camera": "head",
                    "capability": "observe_available",
                }
            )

            success_latched = _official_task_success_latched(env)
            for spec in case_specs:
                if success_latched:
                    case = _skipped_after_success(
                        name=str(spec["name"]),
                        target=str(spec["target"]),
                        action=str(spec["action"]),
                        camera=str(spec["camera"]),
                    )
                    report["integration_execution"]["ordered_cases"].append(case)
                    save()
                    continue
                live_capabilities = env.dashboard_control_capabilities()
                capability = spec["capability"]
                if isinstance(capability, tuple):
                    capability_available = bool(
                        dict(live_capabilities.get(capability[0]) or {}).get(
                            capability[1]
                        )
                    )
                else:
                    capability_available = bool(
                        live_capabilities.get(capability)
                    )
                unavailable_expected = bool(
                    spec["action"] in {"rotate_left", "rotate_right", "up", "down"}
                    and not capability_available
                )
                case = _execute_manual_probe_case(
                    env,
                    backend,
                    name=str(spec["name"]),
                    target=str(spec["target"]),
                    action=str(spec["action"]),
                    camera=str(spec["camera"]),
                    expect_fail_closed=unavailable_expected,
                )
                case["capability_available_before"] = capability_available
                case["capabilities_before"] = _artifact_jsonable(live_capabilities)
                if unavailable_expected:
                    case["disposition"] = (
                        "unavailable_zero_action_fail_closed"
                        if case.get("fail_closed_observed")
                        else "fail_closed_contract_violation"
                    )
                else:
                    case["disposition"] = (
                        "executed"
                        if case.get("error") is None
                        else "execution_rejected"
                    )
                report["integration_execution"]["ordered_cases"].append(case)
                save()
                evidence = case.get("evidence")
                success_latched = _official_task_success_latched(
                    env,
                    (
                        {"task_success": evidence.get("task_success")}
                        if isinstance(evidence, dict)
                        else None
                    ),
                )
            execution_cases = report["integration_execution"]["ordered_cases"]
            camera_case = next(
                (
                    item
                    for item in execution_cases
                    if item.get("name") == "synchronized_three_camera_observe"
                ),
                None,
            )
            camera_evidence = (
                dict(camera_case.get("evidence") or {})
                if isinstance(camera_case, dict)
                else {}
            )
            returned_lineage = dict(
                camera_evidence.get("capture_lineage") or {}
            )
            report["integration_execution"].update(
                {
                    "official_success_latched": bool(
                        getattr(env, "_official_success_latched", False)
                    ),
                    "post_success_follow_up_rpc_count": 0,
                    "three_camera_contract": {
                        "returned_camera_keys": returned_lineage.get(
                            "camera_keys"
                        ),
                        "capture_group_id": returned_lineage.get(
                            "capture_group_id"
                        ),
                        "simulator_step": returned_lineage.get(
                            "simulator_step"
                        ),
                        "complete": (
                            returned_lineage.get("camera_keys")
                            == ["head", "left_wrist", "right_wrist"]
                            and isinstance(
                                returned_lineage.get("capture_group_id"), str
                            )
                            and isinstance(
                                returned_lineage.get("simulator_step"), int
                            )
                        ),
                    },
                }
            )
            save()
        report["status"] = "complete"
        report["stage"] = "complete"
        save()
        log_event("probe_complete", report_path=str(report_path))
    except Exception as exc:
        record_failure(str(report.get("stage") or "runtime_probe"), exc)
        raise
    finally:
        seal_lifecycle(env)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(exit_code)
