#!/usr/bin/env python3
"""Restore the real-radio station and visually audit one public planner action."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ in {None, ""}:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import numpy as np  # noqa: E402

from robots.behavior.env_server import BehaviorEnvFacade, _load_env_config  # noqa: E402
from robots.behavior.snapshot_manifest import (  # noqa: E402
    SNAPSHOT_MANIFEST_KIND,
    build_snapshot_manifest,
    write_snapshot_manifest,
)


def _load_observer_geometry_function() -> Any:
    """Load the sibling module without colliding with ROS's ``scripts`` package."""

    path = Path(__file__).resolve().with_name("behavior_radio_observer_geometry.py")
    spec = importlib.util.spec_from_file_location(
        "_rpent_behavior_radio_observer_geometry",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load observer geometry module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.opposite_wrist_observer_pose


opposite_wrist_observer_pose = _load_observer_geometry_function()

RADIO_BDDL_NAME = "radio_receiver.n.01_1"
POST_PICK_VISUAL_OFFSET_WORLD_M = (0.10, -0.17, 0.20)
PRESS_VIEW_OFFSET_WORLD_M = (0.0, 0.0, 0.20)
HELD_ROTATION_AXIS_ANGLE_RAD = (0.0, 0.0, 1.0, float(np.deg2rad(15.0)))
RADIO_NEAR_NOMINAL_STANDOFF_M = 0.85
OBSERVER_LOOK_OFFSET_WORLD_M = (0.0, 0.0, -0.10)
OBSERVER_CAMERA_OFFSET_WORLD_M = (0.0, -0.45, 0.02)
OBSERVER_MOVE_TIMEOUT_S = 45.0
PUBLIC_CHAIN_OPERATIONS = frozenset(
    {"held_chain", "press_preview", "press", "navigate", "observer_preview"}
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-instance-dir", required=True)
    parser.add_argument("--activity-instance-id", type=int, default=211)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--operation",
        choices=(
            "inspect",
            "plan_pick",
            "preview_pick",
            "pick",
            "held_chain",
            "press_preview",
            "press",
            "navigate",
            "observer_preview",
        ),
        default="inspect",
    )
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument(
        "--camera",
        choices=("head", "left_wrist", "right_wrist"),
        default="head",
    )
    # Selected by direct visual inspection of the restored head RGB image.
    # u is column, v is row.  This pixel is on the real red radio, away from
    # the bright circular table reflection to its right.
    parser.add_argument("--u", type=int, default=360)
    parser.add_argument("--v", type=int, default=532)
    parser.add_argument("--handle-u1", type=int, default=348)
    parser.add_argument("--handle-v1", type=int, default=518)
    parser.add_argument("--handle-u2", type=int, default=373)
    parser.add_argument("--handle-v2", type=int, default=548)
    parser.add_argument("--target-depth-window-px", type=int, default=7)
    parser.add_argument("--handle-depth-window-px", type=int, default=5)
    parser.add_argument("--target-z-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--reuse-public-geometry-json",
        help="prior acceptance JSON whose public RGB-D target and grasp are reused",
    )
    parser.add_argument("--timeout-s", type=float, default=90.0)
    parser.add_argument("--pick-timeout-s", type=float)
    parser.add_argument("--rotate-timeout-s", type=float, default=45.0)
    parser.add_argument("--release-timeout-s", type=float, default=30.0)
    parser.add_argument("--press-timeout-s", type=float, default=60.0)
    parser.add_argument("--navigate-timeout-s", type=float, default=90.0)
    parser.add_argument("--navigate-standoff-m", type=float, default=1.0)
    parser.add_argument(
        "--button-camera",
        choices=("left_wrist", "right_wrist"),
    )
    parser.add_argument("--button-u", type=int)
    parser.add_argument("--button-v", type=int)
    parser.add_argument("--button-depth-window-px", type=int, default=7)
    parser.add_argument("--button-selection-source-json")
    parser.add_argument(
        "--visual-observer-camera",
        choices=("left_wrist", "right_wrist"),
        help=(
            "opposite wrist camera used for observer_preview or pre-pick visual setup"
        ),
    )
    parser.add_argument(
        "--observer-look-offset-world-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=OBSERVER_LOOK_OFFSET_WORLD_M,
    )
    parser.add_argument(
        "--observer-camera-offset-world-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=OBSERVER_CAMERA_OFFSET_WORLD_M,
    )
    parser.add_argument(
        "--observer-move-timeout-s",
        type=float,
        default=OBSERVER_MOVE_TIMEOUT_S,
        help="bounded timeout for the observer-hand move_to (default: 45s)",
    )
    parser.add_argument(
        "--post-pick-visual-validation",
        action="store_true",
        help="after a successful pick, move the held radio to a fixed public visual pose",
    )
    parser.add_argument(
        "--post-pick-visual-validation-timeout-s",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--post-pick-visual-offset-world-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=POST_PICK_VISUAL_OFFSET_WORLD_M,
        help="public world-frame offset added to the public radio target",
    )
    return parser.parse_args()


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
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
            if not str(key).startswith("_image")
            and not str(key).startswith("_depth_image")
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_radio_near_snapshot_manifest(snapshot: Path) -> dict[str, Any]:
    """Fail closed unless *snapshot* is the trusted radio-near bootstrap state."""

    snapshot = Path(snapshot).expanduser().resolve()
    manifest_path = Path(f"{snapshot}.manifest.json")
    if not snapshot.is_file():
        raise RuntimeError(f"radio-near simulator snapshot is missing: {snapshot}")
    if not manifest_path.is_file():
        raise RuntimeError(
            f"radio-near simulator snapshot manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid radio-near simulator snapshot manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("radio-near simulator snapshot manifest must be an object")
    if manifest.get("kind") != SNAPSHOT_MANIFEST_KIND:
        raise RuntimeError("radio-near simulator snapshot manifest has the wrong kind")

    task = manifest.get("task")
    if not isinstance(task, dict):
        raise RuntimeError("radio-near simulator snapshot manifest lacks task binding")
    required_task_fields = {
        "control_mode": "planner_tools",
        "activity_instance_id": 211,
        "seed": 211,
    }
    for field, expected in required_task_fields.items():
        if task.get(field) != expected:
            raise RuntimeError(
                f"radio-near simulator snapshot task {field} mismatch: "
                f"manifest={task.get(field)!r} required={expected!r}"
            )

    invariants = manifest.get("invariants")
    if not isinstance(invariants, dict):
        raise RuntimeError("radio-near simulator snapshot manifest lacks invariants")
    if invariants.get("head_radio_visible") is not True:
        raise RuntimeError("radio-near snapshot must have head_radio_visible=true")
    if invariants.get("radio_toggled_on") is not False:
        raise RuntimeError("radio-near snapshot must have radio_toggled_on=false")

    attachments = invariants.get("attachments")
    if not isinstance(attachments, dict):
        raise RuntimeError("radio-near snapshot lacks attachment invariants")
    for hand in ("left", "right"):
        attachment = attachments.get(hand)
        if not isinstance(attachment, dict):
            raise RuntimeError(f"radio-near snapshot lacks {hand} attachment invariant")
        if (
            attachment.get("backend") is not False
            or attachment.get("simulator") is not False
        ):
            raise RuntimeError(
                f"radio-near snapshot {hand} hand must have no attachment"
            )

    navigation_error = invariants.get("radio_navigation_error")
    if not isinstance(navigation_error, dict):
        raise RuntimeError("radio-near snapshot lacks radio_navigation_error")
    navigation_limits = {
        "position_m": 0.05,
        "orientation_rad": float(np.deg2rad(5.0)),
    }
    for field, limit in navigation_limits.items():
        value = navigation_error.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(
                f"radio-near snapshot navigation {field} must be numeric"
            )
        value = float(value)
        if not np.isfinite(value) or value < 0.0 or value > limit:
            raise RuntimeError(
                f"radio-near snapshot navigation {field} exceeds limit: "
                f"value={value!r} limit={limit!r}"
            )
    return manifest


def _require_cli_snapshot_identity(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> None:
    task = manifest.get("task")
    if not isinstance(task, dict):
        raise RuntimeError("radio-near simulator snapshot manifest lacks task binding")
    for cli_field, manifest_field in (
        ("activity_instance_id", "activity_instance_id"),
        ("seed", "seed"),
    ):
        cli_value = int(getattr(args, cli_field))
        manifest_value = task.get(manifest_field)
        if cli_value != manifest_value:
            raise ValueError(
                f"CLI {cli_field} does not match snapshot manifest: "
                f"cli={cli_value!r} manifest={manifest_value!r}"
            )


def _load_bound_public_geometry(
    geometry_path: Path, *, snapshot: Path
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load only geometry whose public transcript is bound to this snapshot."""

    geometry_path = Path(geometry_path).expanduser().resolve()
    try:
        source = json.loads(geometry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid public geometry report: {exc}") from exc
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise RuntimeError("public geometry report has an unsupported schema")
    if source.get("operation") not in {
        "inspect",
        "plan_pick",
        "preview_pick",
        "pick",
    }:
        raise RuntimeError("public geometry report is not from a legacy operation")
    snapshot_record = source.get("snapshot")
    if not isinstance(snapshot_record, dict):
        raise RuntimeError("public geometry report lacks snapshot binding")
    manifest_path = Path(f"{Path(snapshot).expanduser().resolve()}.manifest.json")
    expected_binding = {
        "sha256": _sha256(Path(snapshot).expanduser().resolve()),
        "manifest_sha256": _sha256(manifest_path),
    }
    for field, expected in expected_binding.items():
        if snapshot_record.get(field) != expected:
            raise RuntimeError(
                f"public geometry snapshot {field} mismatch: "
                f"report={snapshot_record.get(field)!r} current={expected!r}"
            )
    visual_selection = source.get("public_visual_selection")
    if not isinstance(visual_selection, dict) or (
        visual_selection.get("private_truth_used_for_tool_target") is not False
    ):
        raise RuntimeError(
            "public geometry report lacks a non-privileged selection binding"
        )
    selection_camera = visual_selection.get("camera")
    selection_u = visual_selection.get("u_column")
    selection_v = visual_selection.get("v_row")
    if (
        selection_camera not in {"head", "left_wrist", "right_wrist"}
        or isinstance(selection_u, bool)
        or not isinstance(selection_u, int)
        or isinstance(selection_v, bool)
        or not isinstance(selection_v, int)
        or not isinstance(visual_selection.get("selection_method"), str)
        or not visual_selection["selection_method"]
    ):
        raise RuntimeError("public geometry report visual selection schema is invalid")

    target = np.asarray(source.get("public_target_xyz"), dtype=np.float64).reshape(-1)
    grasp_geometry = source.get("public_grasp_geometry")
    if target.size != 3 or not np.isfinite(target).all():
        raise RuntimeError("public geometry report target must be a finite xyz")
    if not isinstance(grasp_geometry, dict):
        raise RuntimeError("public geometry report lacks grasp geometry")
    grasp_quat = np.asarray(
        grasp_geometry.get("grasp_quat_xyzw"), dtype=np.float64
    ).reshape(-1)
    handle_points = np.asarray(
        grasp_geometry.get("handle_points_world"), dtype=np.float64
    )
    if (
        grasp_quat.size != 4
        or not np.isfinite(grasp_quat).all()
        or float(np.linalg.norm(grasp_quat)) <= 1e-9
        or handle_points.shape != (2, 3)
        or not np.isfinite(handle_points).all()
        or grasp_geometry.get("approach_vector") != [0.0, 0.0, -1.0]
    ):
        raise RuntimeError("public geometry report grasp fields are invalid")

    calls = source.get("public_tool_calls")
    if not isinstance(calls, list):
        raise RuntimeError("public geometry report lacks a public tool transcript")
    observe_index = None
    observed_frame_id = None
    primary_projection_index = None
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        call_arguments = call.get("call_arguments")
        if call.get("tool") == "observe" and isinstance(result, dict):
            if (
                isinstance(call_arguments, dict)
                and call_arguments.get("camera") == selection_camera
                and result.get("primitive_success") is True
                and result.get("frame_id")
            ):
                observe_index = index
                observed_frame_id = result["frame_id"]
                continue
        if call.get("tool") != "pixel_to_world" or not isinstance(result, dict):
            continue
        diagnostics = result.get("diagnostics")
        metrics = result.get("metrics")
        if (
            result.get("primitive_success") is True
            and isinstance(diagnostics, dict)
            and isinstance(metrics, dict)
            and isinstance(call_arguments, dict)
            and call_arguments.get("camera") == selection_camera
            and call_arguments.get("frame_id") == observed_frame_id
            and call_arguments.get("u") == selection_u
            and call_arguments.get("v") == selection_v
            and call_arguments.get("output_frame") == "world"
            and diagnostics.get("output_frame") == "world"
            and metrics.get("frame_id") == observed_frame_id
        ):
            xyz = np.asarray(diagnostics.get("xyz"), dtype=np.float64).reshape(-1)
            if xyz.size == 3 and np.array_equal(xyz, target):
                primary_projection_index = index
                break
    if (
        observe_index is None
        or primary_projection_index is None
        or observe_index >= primary_projection_index
    ):
        raise RuntimeError(
            "public geometry report lacks an ordered observe+world pixel_to_world transcript"
        )
    evidence = {
        "path": str(geometry_path),
        "sha256": _sha256(geometry_path),
        "snapshot_binding": expected_binding,
        "observe_call_index": observe_index,
        "primary_projection_call_index": primary_projection_index,
        "observed_frame_id": observed_frame_id,
        "source_public_geometry_only": True,
        "transcript_validated": True,
    }
    normalized_geometry = dict(grasp_geometry)
    normalized_geometry["grasp_quat_xyzw"] = (
        grasp_quat / np.linalg.norm(grasp_quat)
    ).tolist()
    return target.reshape(3), normalized_geometry, evidence


def _load_reviewed_button_selection(
    source_path: Path,
    *,
    snapshot: Path,
    hand: str,
    camera: str,
    setup_target_xyz: Any,
    cli_u: int,
    cli_v: int,
) -> dict[str, Any]:
    """Bind press pixels to a manually reviewed press_preview artifact."""

    source_path = Path(source_path).expanduser().resolve()
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid press_preview source report: {exc}") from exc
    if not isinstance(source, dict) or source.get("operation") != "press_preview":
        raise RuntimeError("button selection source must be a press_preview report")
    if source.get("hand") != hand:
        raise RuntimeError("press_preview hand does not match the press hand")
    if source.get("execution_status") != "runtime_press_preview_complete":
        raise RuntimeError("press_preview source did not complete successfully")
    snapshot_record = source.get("snapshot")
    if not isinstance(snapshot_record, dict):
        raise RuntimeError("press_preview source lacks snapshot binding")
    expected_snapshot = {
        "sha256": _sha256(Path(snapshot).expanduser().resolve()),
        "manifest_sha256": _sha256(
            Path(f"{Path(snapshot).expanduser().resolve()}.manifest.json")
        ),
    }
    for field, expected in expected_snapshot.items():
        if snapshot_record.get(field) != expected:
            raise RuntimeError(f"press_preview snapshot {field} mismatch")
    setup = source.get("press_view_setup")
    if not isinstance(setup, dict) or setup.get("button_camera") != camera:
        raise RuntimeError("press_preview button camera does not match")
    prior_target = np.asarray(setup.get("target_xyz"), dtype=np.float64).reshape(-1)
    current_target = np.asarray(setup_target_xyz, dtype=np.float64).reshape(-1)
    if (
        prior_target.size != 3
        or current_target.size != 3
        or not np.array_equal(prior_target, current_target)
    ):
        raise RuntimeError("press_preview setup target does not exactly match")
    preview_policy = source.get("press_preview_policy")
    if not isinstance(preview_policy, dict) or (
        preview_policy.get("visual_candidate_only") is not True
        or preview_policy.get("automatically_authorizes_press") is not False
    ):
        raise RuntimeError(
            "press_preview source lacks fail-closed authorization policy"
        )

    decision_candidates = (
        source_path.parent / "press_preview" / "visual_review" / "leader_decision.json",
        source_path.parent / "leader_decision.json",
    )
    decision_path = next((path for path in decision_candidates if path.is_file()), None)
    if decision_path is None:
        raise RuntimeError("press_preview leader_decision.json is missing")
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid press_preview leader decision: {exc}") from exc
    findings = decision.get("visual_findings") if isinstance(decision, dict) else None
    reviewed = (
        decision.get("reviewed_button_selection")
        if isinstance(decision, dict)
        else None
    )
    if (
        not isinstance(decision, dict)
        or decision.get("accepted") is not True
        or not isinstance(findings, dict)
        or findings.get("button_visible") is not True
        or not isinstance(reviewed, dict)
    ):
        raise RuntimeError(
            "leader decision does not explicitly accept a visible button"
        )
    expected_review = {
        "hand": hand,
        "camera": camera,
        "u_column": int(cli_u),
        "v_row": int(cli_v),
    }
    for field, expected in expected_review.items():
        if reviewed.get(field) != expected:
            raise RuntimeError(f"reviewed button {field} does not match CLI selection")
    return {
        "source_report_path": str(source_path),
        "source_report_sha256": _sha256(source_path),
        "leader_decision_path": str(decision_path),
        "leader_decision_sha256": _sha256(decision_path),
        "snapshot_binding": expected_snapshot,
        "setup_target_xyz": current_target.tolist(),
        "reviewed_button_selection": expected_review,
        "button_visible": True,
        "manual_review_required": True,
    }


def _validate_post_pick_visual_validation_args(args: argparse.Namespace) -> None:
    offset = np.asarray(
        getattr(
            args,
            "post_pick_visual_offset_world_m",
            POST_PICK_VISUAL_OFFSET_WORLD_M,
        ),
        dtype=np.float64,
    ).reshape(-1)
    if offset.size != 3 or not np.isfinite(offset).all():
        raise ValueError("post-pick visual offset must contain three finite values")
    if not bool(args.post_pick_visual_validation):
        return
    if args.operation != "pick":
        raise ValueError("post-pick visual validation requires operation=pick")
    timeout_s = float(args.post_pick_visual_validation_timeout_s)
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(
            "post-pick visual validation timeout must be finite and positive"
        )


def _positive_finite(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _effective_pick_timeout_s(args: argparse.Namespace) -> float:
    value = args.pick_timeout_s
    if value is None:
        value = args.timeout_s
    return _positive_finite("pick timeout", value)


def _finite_world_offset(name: str, value: Any) -> list[float]:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector.tolist()


def _observer_hand_for_camera(camera: str) -> str:
    mapping = {"left_wrist": "left", "right_wrist": "right"}
    try:
        return mapping[camera]
    except KeyError as exc:
        raise ValueError(f"unsupported visual observer camera: {camera!r}") from exc


def _observer_locking_report(
    *, action_hand: str, observer_hand: str, operation: str
) -> dict[str, Any]:
    """Describe executor locking without claiming a fixed observer world pose."""

    during_pick = operation == "pick"
    return {
        "executor_lock_scope": "non_active_arm_and_gripper_joints_only",
        "during_observer_move_non_active_arm_gripper": action_hand,
        "during_pick_non_active_arm_gripper": (observer_hand if during_pick else None),
        "shared_trunk_remains_active": True,
        "observer_wrist_camera_world_pose_may_drift_during_pick": during_pick,
        "fixed_observer_world_pose_claimed": False,
    }


def _validate_public_operation_args(args: argparse.Namespace) -> None:
    _positive_finite("legacy operation timeout", args.timeout_s)
    if args.pick_timeout_s is not None:
        _positive_finite("pick timeout", args.pick_timeout_s)
    for field, label in (
        ("rotate_timeout_s", "rotate timeout"),
        ("release_timeout_s", "release timeout"),
        ("press_timeout_s", "press timeout"),
        ("navigate_timeout_s", "navigate timeout"),
    ):
        _positive_finite(label, getattr(args, field))
    _positive_finite(
        "observer move timeout",
        getattr(args, "observer_move_timeout_s", OBSERVER_MOVE_TIMEOUT_S),
    )
    observer_camera = getattr(args, "visual_observer_camera", None)
    if args.operation == "observer_preview" and observer_camera is None:
        raise ValueError("observer_preview requires --visual-observer-camera")
    if observer_camera is not None:
        if args.operation not in {"observer_preview", "pick"}:
            raise ValueError(
                "--visual-observer-camera is only valid for observer_preview or pick"
            )
        expected_camera = "left_wrist" if args.hand == "right" else "right_wrist"
        if observer_camera != expected_camera:
            raise ValueError(
                "visual observer must be the action hand's opposite wrist: "
                f"hand={args.hand} required={expected_camera}"
            )
        _finite_world_offset(
            "observer look offset",
            getattr(
                args,
                "observer_look_offset_world_m",
                OBSERVER_LOOK_OFFSET_WORLD_M,
            ),
        )
        _finite_world_offset(
            "observer camera offset",
            getattr(
                args,
                "observer_camera_offset_world_m",
                OBSERVER_CAMERA_OFFSET_WORLD_M,
            ),
        )
    fresh_geometry_required = (
        args.operation in PUBLIC_CHAIN_OPERATIONS or observer_camera is not None
    )
    if not fresh_geometry_required:
        return
    if args.camera != "head":
        raise ValueError(f"{args.operation} requires camera=head for radio geometry")
    if args.reuse_public_geometry_json:
        raise ValueError(
            f"{args.operation} requires fresh head RGB-D; geometry reuse is forbidden"
        )
    if args.operation == "navigate":
        standoff_m = _positive_finite("navigate standoff", args.navigate_standoff_m)
        if abs(standoff_m - RADIO_NEAR_NOMINAL_STANDOFF_M) < 0.10 - 1e-9:
            raise ValueError(
                "navigate standoff must differ from radio-near nominal 0.85m "
                "by at least 0.10m"
            )
    if args.operation in {"press_preview", "press"}:
        expected_button_camera = f"{args.hand}_wrist"
        if args.button_camera != expected_button_camera:
            raise ValueError(
                f"{args.operation} requires --button-camera {expected_button_camera} "
                "for the selected hand"
            )
        if args.operation == "press" and (
            args.button_u is None or args.button_v is None
        ):
            raise ValueError("press requires both --button-u and --button-v")
        if args.operation == "press" and not args.button_selection_source_json:
            raise ValueError("press requires --button-selection-source-json")
        if int(args.button_depth_window_px) < 1:
            raise ValueError("button depth window must be at least one pixel")


def _video_frame_boundary(env: Any) -> int:
    frames = getattr(env, "_video_frames", None)
    if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)):
        raise RuntimeError("video frame counter unavailable for public tool boundary")
    if int(frames) < 0:
        raise RuntimeError("video frame counter cannot be negative")
    return int(frames)


def _call_public_tool_with_boundary(
    env: Any,
    *,
    tool: str,
    tool_role: str,
    call_kwargs: dict[str, Any],
    public_tool_calls: list[dict[str, Any]],
    phase_boundaries: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call one public facade tool and bind it to the continuous MP4 timeline."""

    if tool_role not in {
        "setup",
        "targeting",
        "subject",
        "cleanup",
        "post_pick_visual_validation",
    }:
        raise ValueError(f"invalid public tool role: {tool_role}")
    method = getattr(env, tool, None)
    if not callable(method):
        raise RuntimeError(f"public tool unavailable: {tool}")
    before = _video_frame_boundary(env)
    sequence = len(phase_boundaries)
    try:
        result = method(**call_kwargs)
        if not isinstance(result, dict):
            raise RuntimeError(f"public tool {tool} returned a non-object result")
    except Exception as exc:
        after = _video_frame_boundary(env)
        boundary = {
            "sequence": sequence,
            "tool": tool,
            "tool_role": tool_role,
            "before_video_frame": before,
            "after_video_frame": after,
            "outcome": "exception",
        }
        phase_boundaries.append(boundary)
        public_tool_calls.append(
            {
                "tool": tool,
                "tool_role": tool_role,
                "call_arguments": _jsonable(call_kwargs),
                "phase_boundary": boundary,
                "error": f"{type(exc).__name__}: {exc}",
                **({} if metadata is None else _jsonable(metadata)),
            }
        )
        raise
    after = _video_frame_boundary(env)
    boundary = {
        "sequence": sequence,
        "tool": tool,
        "tool_role": tool_role,
        "before_video_frame": before,
        "after_video_frame": after,
        "outcome": "returned",
    }
    motion_tools = {
        "navigate_to",
        "move_to",
        "pick",
        "rotate_wrist",
        "press",
        "release",
    }
    if tool in motion_tools and after <= before:
        boundary["outcome"] = "video_frame_not_advanced"
    phase_boundaries.append(boundary)
    public_tool_calls.append(
        {
            "tool": tool,
            "tool_role": tool_role,
            "call_arguments": _jsonable(call_kwargs),
            "phase_boundary": boundary,
            "result": _jsonable(result),
            **({} if metadata is None else _jsonable(metadata)),
        }
    )
    if tool in motion_tools and after <= before:
        raise RuntimeError(
            f"motion tool {tool} did not advance the continuous MP4 frame counter"
        )
    return result


def _probe_public_env_health_after_failure(
    env: Any,
    *,
    public_tool_calls: list[dict[str, Any]],
    phase_boundaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Issue exactly one public observe probe without masking the prior failure."""

    probe_kwargs = {"camera": "head"}
    try:
        _video_frame_boundary(env)
    except RuntimeError:
        # reset / video startup can fail before a frame counter is usable.  The
        # public RPC must still be attempted once, but no fabricated video
        # boundary is attached to it.
        try:
            result = env.observe(**probe_kwargs)
            if not isinstance(result, dict):
                raise RuntimeError("public health observe returned a non-object result")
        except Exception as exc:
            public_tool_calls.append(
                {
                    "tool": "observe",
                    "tool_role": "cleanup",
                    "call_arguments": probe_kwargs,
                    "visual_role": "post_failure_rpc_health_probe",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return {
                "attempted": True,
                "responsive": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        public_tool_calls.append(
            {
                "tool": "observe",
                "tool_role": "cleanup",
                "call_arguments": probe_kwargs,
                "visual_role": "post_failure_rpc_health_probe",
                "result": _jsonable(result),
            }
        )
    else:
        try:
            result = _call_public_tool_with_boundary(
                env,
                tool="observe",
                tool_role="cleanup",
                call_kwargs=probe_kwargs,
                public_tool_calls=public_tool_calls,
                phase_boundaries=phase_boundaries,
                metadata={"visual_role": "post_failure_rpc_health_probe"},
            )
        except Exception as exc:
            return {
                "attempted": True,
                "responsive": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "attempted": True,
        "responsive": True,
        "primitive_success": result.get("primitive_success"),
        "stop_reason": result.get("stop_reason"),
    }


def _require_primitive_success(result: dict[str, Any], *, tool: str) -> None:
    if result.get("primitive_success") is not True:
        raise RuntimeError(f"{tool} failed: {result.get('stop_reason')}")


def _run_opposite_wrist_observer_setup(
    env: Any,
    *,
    action_hand: str,
    observer_camera: str,
    public_surface_xyz: Any,
    look_offset_world_m: Any,
    camera_offset_world_m: Any,
    timeout_s: float,
    output_dir: Path,
    public_tool_calls: list[dict[str, Any]],
    phase_boundaries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Place the opposite wrist using public geometry, then capture its RGB-D."""

    expected_camera = "left_wrist" if action_hand == "right" else "right_wrist"
    if observer_camera != expected_camera:
        raise ValueError(
            "visual observer must be the action hand's opposite wrist: "
            f"hand={action_hand} required={expected_camera}"
        )
    observer_hand = _observer_hand_for_camera(observer_camera)
    move_timeout_s = _positive_finite("observer move timeout", timeout_s)
    geometry = opposite_wrist_observer_pose(
        public_surface_xyz,
        look_offset_world_m=_finite_world_offset(
            "observer look offset", look_offset_world_m
        ),
        camera_offset_world_m=_finite_world_offset(
            "observer camera offset", camera_offset_world_m
        ),
    )
    move_result = _call_public_tool_with_boundary(
        env,
        tool="move_to",
        tool_role="setup",
        call_kwargs={
            "hand": observer_hand,
            "target_xyz": geometry["eef_position_world_xyz"],
            "frame": "world",
            "target_quat_xyzw": geometry["eef_quat_xyzw"],
            "plan_only": False,
            "position_tolerance_m": 0.02,
            "orientation_tolerance_rad": 0.087,
            "timeout_s": move_timeout_s,
        },
        public_tool_calls=public_tool_calls,
        phase_boundaries=phase_boundaries,
        metadata={
            "visual_role": "opposite_wrist_observer_pose_setup",
            "action_hand": action_hand,
            "observer_camera": observer_camera,
            "geometry": geometry,
        },
    )
    _require_primitive_success(move_result, tool="opposite wrist observer move_to")
    observed = _call_public_tool_with_boundary(
        env,
        tool="observe",
        tool_role="targeting",
        call_kwargs={"camera": observer_camera},
        public_tool_calls=public_tool_calls,
        phase_boundaries=phase_boundaries,
        metadata={
            "visual_role": "opposite_wrist_observer_rgbd",
            "action_hand": action_hand,
            "observer_hand": observer_hand,
        },
    )
    _require_primitive_success(observed, tool="opposite wrist observer observe")
    _save_public_rgbd_frame(
        env,
        camera=observer_camera,
        observed=observed,
        output_dir=output_dir,
        prefix="visual_observer_after_setup",
    )
    report = {
        "action_hand": action_hand,
        "observer_hand": observer_hand,
        "observer_camera": observer_camera,
        "geometry": geometry,
        "move_timeout_s": move_timeout_s,
        "move_result": _jsonable(move_result),
        "observed_frame_id": observed.get("frame_id"),
        "provenance": {
            "surface_source": "fresh_head_rgbd_pixel_to_world",
            "eef_pose_source": (
                "public_surface_plus_offsets_and_official_r1pro_fixed_camera_extrinsic"
            ),
            "uses_private_truth": False,
            "uses_backend_pose": False,
        },
    }
    return move_result, observed, report


def _negative_public_surface_normal(projection: dict[str, Any]) -> list[float]:
    diagnostics = projection.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("output_frame") != "world":
        raise RuntimeError("button projection surface normal must be in world frame")
    normal_value = (
        diagnostics.get("surface_normal") if isinstance(diagnostics, dict) else None
    )
    try:
        normal = np.asarray(normal_value, dtype=np.float64).reshape(3)
    except Exception as exc:
        raise RuntimeError("button projection lacks a 3D surface normal") from exc
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal).all() or not np.isfinite(norm) or norm <= 1e-9:
        raise RuntimeError("button projection surface normal is invalid")
    return (-normal / norm).tolist()


def _run_post_pick_visual_validation(
    env: Any,
    *,
    hand: str,
    public_target_xyz: Any,
    grasp_quat_xyzw: Any,
    timeout_s: float,
    public_tool_calls: list[dict[str, Any]],
    phase_boundaries: list[dict[str, Any]],
    offset_world_m: Any = POST_PICK_VISUAL_OFFSET_WORLD_M,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move a successful public grasp to one fixed, non-privileged visual pose."""

    source_target = np.asarray(public_target_xyz, dtype=np.float64).reshape(3)
    grasp_quat = np.asarray(grasp_quat_xyzw, dtype=np.float64).reshape(4)
    if not np.isfinite(source_target).all() or not np.isfinite(grasp_quat).all():
        raise RuntimeError("post-pick public target or grasp quaternion is non-finite")
    offset = np.asarray(offset_world_m, dtype=np.float64).reshape(-1)
    if offset.size != 3 or not np.isfinite(offset).all():
        raise RuntimeError("post-pick visual offset must contain three finite values")
    target = source_target + offset
    derivation = {
        "source": "public_target_xyz",
        "offset_world_m": offset.tolist(),
        "formula": "public_target_xyz + fixed_world_offset",
        "uses_private_truth": False,
        "uses_backend_pose": False,
    }
    result = _call_public_tool_with_boundary(
        env,
        tool="move_to",
        tool_role="post_pick_visual_validation",
        call_kwargs={
            "hand": hand,
            "target_xyz": target.tolist(),
            "frame": "world",
            "target_quat_xyzw": grasp_quat.tolist(),
            "plan_only": False,
            "position_tolerance_m": 0.02,
            "orientation_tolerance_rad": 0.087,
            "timeout_s": float(timeout_s),
        },
        public_tool_calls=public_tool_calls,
        phase_boundaries=phase_boundaries,
        metadata={
            "target_xyz": target.tolist(),
            "target_derivation": derivation,
        },
    )
    return result, {
        "enabled": True,
        "tool_role": "post_pick_visual_validation",
        "public_source_target_xyz": source_target.tolist(),
        "target_xyz": target.tolist(),
        "target_quat_xyzw": grasp_quat.tolist(),
        "offset_world_m": offset.tolist(),
        "target_derivation": derivation,
        "timeout_s": float(timeout_s),
        "result": result,
    }


def _save_frame(frame: Any, path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.asarray(frame.rgb, dtype=np.uint8)[..., :3])


def _save_public_rgbd_frame(
    env: Any,
    *,
    camera: str,
    observed: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> None:
    frame = env._frame_cache.get_current(camera, observed["frame_id"])
    _save_frame(frame, output_dir / f"{prefix}_{camera}.png")
    np.save(output_dir / f"{prefix}_{camera}_depth_m.npy", frame.depth_m)
    (output_dir / f"{prefix}_{camera}_depth_visualization.png").write_bytes(
        observed["_depth_image_bytes"]
    )


def _rgb_frame(value: Any) -> np.ndarray:
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(f"video frame must be HxWxC with C>=3, got {frame.shape}")
    frame = frame[..., :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _three_camera_panels(mosaic: np.ndarray) -> dict[str, np.ndarray]:
    """Split the env video's 2x2 head/left/right/blank mosaic."""

    frame = _rgb_frame(mosaic)
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        raise RuntimeError(
            f"three-camera mosaic must have even dimensions, got {width}x{height}"
        )
    half_h, half_w = height // 2, width // 2
    return {
        "head": frame[:half_h, :half_w],
        "left_wrist": frame[:half_h, half_w:],
        "right_wrist": frame[half_h:, :half_w],
    }


def _write_png_artifact(path: Path, image: np.ndarray) -> dict[str, Any]:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, _rgb_frame(image))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(np.asarray(image).shape),
    }


def _write_visual_review_artifacts(
    *,
    video_path: Path,
    output_dir: Path,
    phase_frames: dict[str, tuple[int, np.ndarray]],
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    """Persist first/middle/last three-camera frames for visual review.

    This function deliberately records no visual-success boolean.  The images
    are evidence for a later human or VLM review, not an automatic acceptance
    signal.
    """

    expected_phases = ("first", "middle", "last")
    if tuple(phase_frames) != expected_phases:
        raise RuntimeError(
            f"phase frames must be ordered as {expected_phases}, got {tuple(phase_frames)}"
        )
    if int(frame_count) < 1:
        raise RuntimeError("video contains no frames")
    fps_value = float(fps)
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        raise RuntimeError(f"video FPS must be finite and positive, got {fps}")

    review_dir = output_dir / "visual_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    phase_metadata: dict[str, Any] = {}
    contact_rows = []
    panel_shape: tuple[int, ...] | None = None
    for phase in expected_phases:
        frame_index, mosaic = phase_frames[phase]
        mosaic = _rgb_frame(mosaic)
        panels = _three_camera_panels(mosaic)
        if panel_shape is None:
            panel_shape = panels["head"].shape
        if any(panel.shape != panel_shape for panel in panels.values()):
            raise RuntimeError("three-camera panels do not share one shape")
        mosaic_artifact = _write_png_artifact(
            review_dir / f"{phase}_mosaic.png", mosaic
        )
        camera_artifacts = {
            camera: _write_png_artifact(review_dir / f"{phase}_{camera}.png", panel)
            for camera, panel in panels.items()
        }
        phase_metadata[phase] = {
            "frame_index": int(frame_index),
            "mosaic": mosaic_artifact,
            "cameras": camera_artifacts,
        }
        contact_rows.append(
            np.concatenate(
                [panels["head"], panels["left_wrist"], panels["right_wrist"]],
                axis=1,
            )
        )
    contact_sheet = np.concatenate(contact_rows, axis=0)
    contact_sheet_artifact = _write_png_artifact(
        review_dir / "contact_sheet.png", contact_sheet
    )
    return {
        "path": str(video_path),
        "exists": video_path.is_file(),
        "bytes": video_path.stat().st_size if video_path.is_file() else 0,
        "sha256": _sha256(video_path) if video_path.is_file() else None,
        "fps": fps_value,
        "frames": int(frame_count),
        "duration_s": float(frame_count) / fps_value,
        "source_layout": "2x2:head,left_wrist/right_wrist,blank",
        "extracted_phases": phase_metadata,
        "contact_sheet": {
            **contact_sheet_artifact,
            "layout": "rows=first,middle,last;columns=head,left_wrist,right_wrist",
        },
        "visual_review_required": True,
        "visual_decision": None,
        "automatic_visual_success_declared": False,
    }


def _phase_boundary_frame_indices(
    phase_boundaries: list[dict[str, Any]], *, frame_count: int
) -> list[dict[str, Any]]:
    if int(frame_count) < 1:
        raise RuntimeError("phase extraction requires at least one video frame")
    indexed = []
    for expected_sequence, boundary in enumerate(phase_boundaries):
        if not isinstance(boundary, dict):
            raise RuntimeError("phase boundary must be an object")
        if boundary.get("sequence") != expected_sequence:
            raise RuntimeError("phase boundary sequence is not contiguous")
        before_count = boundary.get("before_video_frame")
        after_count = boundary.get("after_video_frame")
        if (
            isinstance(before_count, bool)
            or isinstance(after_count, bool)
            or not isinstance(before_count, int)
            or not isinstance(after_count, int)
            or before_count < 1
            or after_count < before_count
            or after_count > int(frame_count)
        ):
            raise RuntimeError(
                "phase boundary video frame is outside the finalized MP4: "
                f"boundary={boundary!r} frame_count={frame_count}"
            )
        before_index = before_count - 1
        after_index = after_count - 1
        indexed.append(
            {
                **boundary,
                "frame_indices": {
                    "before": before_index,
                    "mid": (before_index + after_index) // 2,
                    "after": after_index,
                },
            }
        )
    return indexed


def _write_phase_boundary_visual_artifacts(
    *,
    video_path: Path,
    output_dir: Path,
    phase_boundaries: list[dict[str, Any]],
    frame_count: int,
) -> list[dict[str, Any]]:
    import imageio.v2 as imageio

    indexed = _phase_boundary_frame_indices(phase_boundaries, frame_count=frame_count)
    if not indexed:
        return []
    requested_indices = sorted(
        {index for boundary in indexed for index in boundary["frame_indices"].values()}
    )
    reader = imageio.get_reader(str(video_path))
    try:
        decoded = {
            index: _rgb_frame(reader.get_data(index)).copy()
            for index in requested_indices
        }
    finally:
        reader.close()
    phase_root = output_dir / "visual_review" / "phase_boundaries"
    artifacts = []
    for boundary in indexed:
        sequence = int(boundary["sequence"])
        tool = str(boundary["tool"])
        role = str(boundary["tool_role"])
        phases = {}
        for phase_name, frame_index in boundary["frame_indices"].items():
            mosaic = decoded[frame_index]
            panels = _three_camera_panels(mosaic)
            stem = f"{sequence:03d}_{role}_{tool}_{phase_name}"
            phases[phase_name] = {
                "frame_index": int(frame_index),
                "mosaic": _write_png_artifact(
                    phase_root / f"{stem}_mosaic.png", mosaic
                ),
                "cameras": {
                    camera: _write_png_artifact(
                        phase_root / f"{stem}_{camera}.png", panel
                    )
                    for camera, panel in panels.items()
                },
            }
        artifacts.append(
            {
                "sequence": sequence,
                "tool": tool,
                "tool_role": role,
                "before_video_frame": int(boundary["before_video_frame"]),
                "after_video_frame": int(boundary["after_video_frame"]),
                "outcome": boundary["outcome"],
                "phases": phases,
            }
        )
    return artifacts


def _extract_visual_review_artifacts(
    video_path: Path,
    output_dir: Path,
    *,
    phase_boundaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Decode a finalized MP4 and write deterministic visual-review evidence."""

    import imageio.v2 as imageio

    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError("video artifact is missing or empty")
    reader = imageio.get_reader(str(video_path))
    first: np.ndarray | None = None
    last: np.ndarray | None = None
    frame_count = 0
    metadata: dict[str, Any] = {}
    try:
        metadata = dict(reader.get_meta_data() or {})
        for frame_count, frame in enumerate(reader, start=1):
            rgb = _rgb_frame(frame)
            if first is None:
                first = rgb.copy()
            last = rgb.copy()
    finally:
        reader.close()
    if first is None or last is None or frame_count < 1:
        raise RuntimeError("video decoder returned no frames")
    middle_index = (frame_count - 1) // 2
    reader = imageio.get_reader(str(video_path))
    try:
        middle = _rgb_frame(reader.get_data(middle_index)).copy()
    finally:
        reader.close()
    fps = float(metadata.get("fps") or 0.0)
    visual_review = _write_visual_review_artifacts(
        video_path=video_path,
        output_dir=output_dir,
        phase_frames={
            "first": (0, first),
            "middle": (middle_index, middle),
            "last": (frame_count - 1, last),
        },
        frame_count=frame_count,
        fps=fps,
    )
    visual_review["phase_boundary_frames"] = _write_phase_boundary_visual_artifacts(
        video_path=video_path,
        output_dir=output_dir,
        phase_boundaries=list(phase_boundaries or []),
        frame_count=frame_count,
    )
    return visual_review


def _require_public_projection_quality(
    projection: dict[str, Any], *, visual_role: str
) -> None:
    """Gate a visual target using only fields exposed by pixel_to_world."""

    if projection.get("primitive_success") is not True:
        raise RuntimeError(
            f"{visual_role} pixel_to_world failed: {projection.get('stop_reason')}"
        )
    metrics = projection.get("metrics", {})
    depth = metrics.get("depth", {})
    checks = {
        "confidence_at_least_0_80": float(metrics.get("confidence", 0.0)) >= 0.80,
        "round_trip_at_most_1px": float(
            metrics.get("reprojection_error_px", float("inf"))
        )
        <= 1.0,
        "valid_ratio_at_least_0_90": float(depth.get("valid_ratio", 0.0)) >= 0.90,
        "center_cluster_ratio_at_least_0_60": float(depth.get("cluster_ratio", 0.0))
        >= 0.60,
        "depth_mad_at_most_0_006m": float(depth.get("mad_m", float("inf"))) <= 0.006,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{visual_role} public RGB-D quality rejected: {checks}")


def _matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized xyzw quaternion."""

    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            quat = np.array(
                [
                    0.25 * scale,
                    (m[0, 1] + m[1, 0]) / scale,
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[2, 1] - m[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            quat = np.array(
                [
                    (m[0, 1] + m[1, 0]) / scale,
                    0.25 * scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    (m[0, 2] - m[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            quat = np.array(
                [
                    (m[0, 2] + m[2, 0]) / scale,
                    (m[1, 2] + m[2, 1]) / scale,
                    0.25 * scale,
                    (m[1, 0] - m[0, 1]) / scale,
                ]
            )
    return quat / np.linalg.norm(quat)


def _top_down_grasp_quaternion(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Align R1Pro finger opening perpendicular to an RGB-D handle axis."""

    tangent = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    tangent[2] = 0.0
    tangent /= np.linalg.norm(tangent)
    local_y_world = np.array([tangent[1], -tangent[0], 0.0], dtype=np.float64)
    local_z_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    local_x_world = np.cross(local_y_world, local_z_world)
    rotation = np.column_stack((local_x_world, local_y_world, local_z_world))
    if np.linalg.det(rotation) < 0.999:
        raise RuntimeError("computed grasp frame is not a proper rotation")
    return _matrix_to_quaternion_xyzw(rotation)


def _radio_mask(
    sensor: Any, radio_object_name: str
) -> tuple[np.ndarray, dict[str, Any]]:
    private_obs, private_info = sensor.get_obs()
    labels = {
        int(key): str(value) for key, value in private_info["seg_instance_id"].items()
    }
    segment = f"/{radio_object_name.strip().lower()}/"
    ids = {
        key
        for key, label in labels.items()
        if segment in f"/{label.strip('/').lower()}/"
    }
    segmentation = private_obs["seg_instance_id"]
    try:
        import torch

        if torch.is_tensor(segmentation):
            segmentation = segmentation.detach().cpu().numpy()
    except Exception:
        pass
    return np.isin(np.asarray(segmentation).squeeze(), sorted(ids)), private_obs


def _private_audit(env: BehaviorEnvFacade, radio: Any) -> dict[str, Any]:
    from omnigibson.object_states.toggle import ToggledOn

    wrapped = getattr(radio, "wrapped_obj", radio)
    robot = env._planner.backend._find_robot()
    assisted = getattr(robot, "_ag_obj_in_hand", {})
    attached: dict[str, Any] = {}
    for hand in ("left", "right"):
        backend_obj = env._planner.backend.get_attached_object(hand)
        simulator_obj = assisted.get(hand)
        attached[hand] = {
            "backend_name": getattr(backend_obj, "name", None),
            "simulator_name": getattr(simulator_obj, "name", None),
            "is_exact_radio": bool(backend_obj is wrapped or simulator_obj is wrapped),
        }
    position, orientation = radio.get_position_orientation()
    return {
        "purpose": "acceptance_audit_only_not_public_tool_payload",
        "radio_simulator_name": str(getattr(wrapped, "name", "")),
        "radio_position_world": _jsonable(position),
        "radio_orientation_xyzw": _jsonable(orientation),
        "radio_toggled_on": bool(radio.states[ToggledOn].get_value()),
        "attachments": attached,
        "eef": {
            hand: _jsonable(env._planner.backend.get_eef_pose(hand))
            for hand in ("left", "right")
        },
    }


def _collect_posthoc_private_evidence(
    env: BehaviorEnvFacade,
    *,
    args: argparse.Namespace,
    action_dir: Path,
    source_snapshot: Path,
) -> dict[str, Any]:
    """Record private truth after public calls without affecting execution.

    This evidence is deliberately best-effort: missing task objects, sensors, or
    snapshot manifests are recorded as audit errors and never raised back into
    the public action flow.
    """

    evidence: dict[str, Any] = {
        "purpose": "posthoc_acceptance_audit_only_not_execution_gate",
        "phase": "after_all_public_tool_calls",
        "available": False,
        "does_not_gate_execution": True,
    }
    try:
        radio = env._env.omnigibson_env.task.object_scope.get(RADIO_BDDL_NAME)
        if radio is None or not radio.exists:
            raise RuntimeError("real task radio is missing during posthoc audit")
        radio_object = getattr(radio, "wrapped_obj", radio)
        radio_name = str(getattr(radio_object, "name", ""))
        evidence.update(_private_audit(env, radio))
        evidence.update(
            {
                "available": True,
                "phase": "after_all_public_tool_calls",
                "does_not_gate_execution": True,
            }
        )

        visual_audit: dict[str, Any] = {
            "sensor_frame_timing": "captured_after_all_public_tool_calls",
            "same_frame_as_public_target_observation": False,
            "does_not_gate_execution": True,
        }
        try:
            mask, _private_sensor_obs = _radio_mask(
                env._sensor_for_camera(args.camera), radio_name
            )
            mask_path = action_dir / "private_radio_mask_after_public_tools.png"
            mask_rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)
            _write_png_artifact(mask_path, mask_rgb)

            def mask_value(u: int, v: int) -> bool | None:
                if 0 <= int(v) < mask.shape[0] and 0 <= int(u) < mask.shape[1]:
                    return bool(mask[int(v), int(u)])
                return None

            visual_audit.update(
                {
                    "mask_path": str(mask_path),
                    "mask_sha256": _sha256(mask_path),
                    "radio_mask_pixels": int(np.count_nonzero(mask)),
                    "current_frame_selected_pixels": {
                        "primary": mask_value(args.u, args.v),
                        "handle_endpoint_1": mask_value(args.handle_u1, args.handle_v1),
                        "handle_endpoint_2": mask_value(args.handle_u2, args.handle_v2),
                    },
                    "selected_pixel_values_are_not_target_validation": True,
                }
            )
        except Exception as exc:
            visual_audit["error"] = f"{type(exc).__name__}: {exc}"
        evidence["visual_target_audit"] = visual_audit

        manifest_path = Path(f"{source_snapshot}.manifest.json")
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                before_position = manifest.get("invariants", {}).get(
                    "radio_position_world"
                )
                after_position = evidence.get("radio_position_world")
                baseline: dict[str, Any] = {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "radio_position_world": before_position,
                }
                if before_position is not None and after_position is not None:
                    before = np.asarray(before_position, dtype=np.float64).reshape(3)
                    after = np.asarray(after_position, dtype=np.float64).reshape(3)
                    baseline["radio_lift_m"] = float(after[2] - before[2])
                evidence["source_snapshot_baseline"] = baseline
            except Exception as exc:
                evidence["source_snapshot_baseline_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
    except Exception as exc:
        evidence["error"] = f"{type(exc).__name__}: {exc}"
    return evidence


def main() -> None:
    args = _args()
    _validate_post_pick_visual_validation_args(args)
    _validate_public_operation_args(args)
    if abs(float(args.target_z_offset_m)) > 1e-9:
        raise ValueError(
            "target-z-offset-m must remain zero; contact targets come directly "
            "from public RGB-D"
        )
    repo = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    action_dir = output_dir / args.operation
    snapshot = Path(args.snapshot).expanduser().resolve()
    radio_near_manifest = _require_radio_near_snapshot_manifest(snapshot)
    _require_cli_snapshot_identity(args, radio_near_manifest)
    os.environ["RPENT_BEHAVIOR_RESTORE_STATE"] = str(snapshot)
    env_args = SimpleNamespace(
        suite="behavior_2025_challenge",
        task=0,
        task_name="turning_on_radio",
        activity_definition_id=0,
        activity_instance_id=args.activity_instance_id,
        activity_instance_dir=args.activity_instance_dir,
        scene_model="house_double_floor_lower",
        seed=args.seed,
        max_episode_steps=24756,
        output_dir=str(output_dir),
        config_path=None,
        control_mode="planner_tools",
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pending_visual_review",
        "execution_status": "initializing",
        "visual_review": {
            "required": True,
            "decision": None,
            "automatic_success_declared": False,
        },
        "operation": args.operation,
        "hand": args.hand,
        "post_pick_visual_validation": {
            "enabled": bool(args.post_pick_visual_validation),
            "timeout_s": float(args.post_pick_visual_validation_timeout_s),
            "offset_world_m": [
                float(value) for value in args.post_pick_visual_offset_world_m
            ],
        },
        "public_operation_configuration": {
            "pick_timeout_s": _effective_pick_timeout_s(args),
            "rotate_timeout_s": float(args.rotate_timeout_s),
            "release_timeout_s": float(args.release_timeout_s),
            "press_timeout_s": float(args.press_timeout_s),
            "navigate_timeout_s": float(args.navigate_timeout_s),
            "navigate_standoff_m": float(args.navigate_standoff_m),
            "button_camera": args.button_camera,
            "button_u_column": args.button_u,
            "button_v_row": args.button_v,
            "button_selection_source_json": args.button_selection_source_json,
            "visual_observer_camera": args.visual_observer_camera,
            "observer_look_offset_world_m": [
                float(value) for value in args.observer_look_offset_world_m
            ],
            "observer_camera_offset_world_m": [
                float(value) for value in args.observer_camera_offset_world_m
            ],
            "observer_move_timeout_s": float(args.observer_move_timeout_s),
        },
        "process": {
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "sid": os.getsid(0),
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ports": [],
        },
        "commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            ).strip()
        ),
        "configuration": vars(env_args),
        "snapshot": {
            "path": str(snapshot),
            "sha256": _sha256(snapshot),
            "manifest_path": str(Path(f"{snapshot}.manifest.json")),
            "manifest_sha256": _sha256(Path(f"{snapshot}.manifest.json")),
            "radio_near_gate": {
                "passed": True,
                "kind": radio_near_manifest["kind"],
                "task": {
                    field: radio_near_manifest["task"][field]
                    for field in (
                        "control_mode",
                        "activity_instance_id",
                        "seed",
                    )
                },
                "invariants": {
                    field: radio_near_manifest["invariants"][field]
                    for field in (
                        "head_radio_visible",
                        "attachments",
                        "radio_toggled_on",
                        "radio_navigation_error",
                    )
                },
            },
        },
        "public_visual_selection": {
            "camera": args.camera,
            "u_column": int(args.u),
            "v_row": int(args.v),
            "selection_method": (
                "reuse_prior_public_rgbd_world_geometry"
                if args.reuse_public_geometry_json
                else "manual_visual_review_of_real_radio_rgb"
            ),
            "private_truth_used_for_tool_target": False,
        },
        "public_tool_calls": [],
        "phase_boundaries": [],
    }
    report_path = output_dir / "radio_tool_acceptance.json"

    def save() -> None:
        report_path.write_text(
            json.dumps(_jsonable(report), indent=2), encoding="utf-8"
        )

    save()
    env: BehaviorEnvFacade | None = None
    try:
        env = BehaviorEnvFacade(
            cfg=_load_env_config(env_args),
            meta=vars(env_args),
            output_dir=output_dir,
            control_mode="planner_tools",
        )
        _observation, reset_info = env.reset()
        report["reset"] = {
            "env_meta": env.get_env_meta(),
            "raw_official_success": bool(
                isinstance(reset_info, dict)
                and isinstance(reset_info.get("done"), dict)
                and reset_info["done"].get("success", False)
            ),
        }
        env.start_video_segment(action_dir / "episode.mp4")

        new_public_operation = (
            args.operation in PUBLIC_CHAIN_OPERATIONS
            or args.visual_observer_camera is not None
        )
        if new_public_operation:
            observed = _call_public_tool_with_boundary(
                env,
                tool="observe",
                tool_role="setup",
                call_kwargs={"camera": "head"},
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={"visual_role": "radio_geometry"},
            )
            _require_primitive_success(observed, tool="radio geometry observe")
        else:
            observed = env.observe(args.camera)
            report["public_tool_calls"].append(
                {
                    "tool": "observe",
                    "tool_role": "targeting",
                    "call_arguments": {"camera": args.camera},
                    "visual_role": "radio_geometry",
                    "result": _jsonable(observed),
                }
            )
        _save_public_rgbd_frame(
            env,
            camera=args.camera,
            observed=observed,
            output_dir=action_dir,
            prefix="before",
        )
        if args.reuse_public_geometry_json:
            geometry_path = Path(args.reuse_public_geometry_json).expanduser().resolve()
            (
                target,
                public_grasp_geometry,
                report["public_geometry_reuse"],
            ) = _load_bound_public_geometry(geometry_path, snapshot=snapshot)
            grasp_quat = np.asarray(
                public_grasp_geometry["grasp_quat_xyzw"], dtype=np.float64
            ).reshape(4)
            report["public_target_xyz"] = target.tolist()
            report["public_grasp_geometry"] = public_grasp_geometry
        else:
            projection_kwargs = {
                "camera": args.camera,
                "frame_id": observed["frame_id"],
                "u": int(args.u),
                "v": int(args.v),
                "depth_window_px": int(args.target_depth_window_px),
                "output_frame": "world",
            }
            if new_public_operation:
                projection = _call_public_tool_with_boundary(
                    env,
                    tool="pixel_to_world",
                    tool_role="setup",
                    call_kwargs=projection_kwargs,
                    public_tool_calls=report["public_tool_calls"],
                    phase_boundaries=report["phase_boundaries"],
                    metadata={"visual_role": "radio_primary_target"},
                )
            else:
                projection = env.pixel_to_world(**projection_kwargs)
                report["public_tool_calls"].append(
                    {
                        "tool": "pixel_to_world",
                        "tool_role": "targeting",
                        "call_arguments": _jsonable(projection_kwargs),
                        "visual_role": "radio_primary_target",
                        "result": _jsonable(projection),
                    }
                )
            _require_public_projection_quality(projection, visual_role="primary_target")
            target = np.asarray(
                projection["diagnostics"]["xyz"], dtype=np.float64
            ).reshape(3)
            target[2] += float(args.target_z_offset_m)
            report["public_target_xyz"] = target.tolist()
            if args.operation not in {"navigate", "observer_preview"}:
                handle_points = []
                for label, u, v in (
                    ("handle_endpoint_1", args.handle_u1, args.handle_v1),
                    ("handle_endpoint_2", args.handle_u2, args.handle_v2),
                ):
                    endpoint_kwargs = {
                        "camera": args.camera,
                        "frame_id": observed["frame_id"],
                        "u": int(u),
                        "v": int(v),
                        "depth_window_px": int(args.handle_depth_window_px),
                        "output_frame": "world",
                    }
                    if new_public_operation:
                        endpoint = _call_public_tool_with_boundary(
                            env,
                            tool="pixel_to_world",
                            tool_role="setup",
                            call_kwargs=endpoint_kwargs,
                            public_tool_calls=report["public_tool_calls"],
                            phase_boundaries=report["phase_boundaries"],
                            metadata={
                                "visual_role": label,
                                "u_column": int(u),
                                "v_row": int(v),
                            },
                        )
                    else:
                        endpoint = env.pixel_to_world(**endpoint_kwargs)
                        report["public_tool_calls"].append(
                            {
                                "tool": "pixel_to_world",
                                "tool_role": "targeting",
                                "call_arguments": _jsonable(endpoint_kwargs),
                                "visual_role": label,
                                "u_column": int(u),
                                "v_row": int(v),
                                "result": _jsonable(endpoint),
                            }
                        )
                    _require_public_projection_quality(endpoint, visual_role=label)
                    handle_points.append(
                        np.asarray(endpoint["diagnostics"]["xyz"], dtype=np.float64)
                    )
                grasp_quat = _top_down_grasp_quaternion(*handle_points)
                report["public_grasp_geometry"] = {
                    "handle_points_world": [point.tolist() for point in handle_points],
                    "grasp_quat_xyzw": grasp_quat.tolist(),
                    "approach_vector": [0.0, 0.0, -1.0],
                    "derivation": "head_rgbd_handle_axis_and_known_r1pro_eef_axes",
                }

        action_result: dict[str, Any] | None = None
        post_pick_validation_result: dict[str, Any] | None = None
        observer_setup_result: dict[str, Any] | None = None
        if args.visual_observer_camera is not None:
            (
                observer_setup_result,
                _observer_frame,
                report["visual_observer_setup"],
            ) = _run_opposite_wrist_observer_setup(
                env,
                action_hand=args.hand,
                observer_camera=args.visual_observer_camera,
                public_surface_xyz=target,
                look_offset_world_m=args.observer_look_offset_world_m,
                camera_offset_world_m=args.observer_camera_offset_world_m,
                timeout_s=float(args.observer_move_timeout_s),
                output_dir=action_dir,
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
            )
            report["visual_observer_setup"]["observer_locking"] = (
                _observer_locking_report(
                    action_hand=args.hand,
                    observer_hand=_observer_hand_for_camera(
                        args.visual_observer_camera
                    ),
                    operation=args.operation,
                )
            )

        if args.operation == "observer_preview":
            action_result = observer_setup_result
        elif args.operation in {"plan_pick", "preview_pick"}:
            fingertip_offset = env._planner.backend.get_eef_to_fingertip_length(
                args.hand
            )
            action_result = env.move_to(
                hand=args.hand,
                target_xyz=(
                    target + np.array([0.0, 0.0, float(fingertip_offset) + 0.08])
                ).tolist(),
                frame="world",
                target_quat_xyzw=grasp_quat.tolist(),
                plan_only=args.operation == "plan_pick",
                timeout_s=float(args.timeout_s),
            )
            report["public_tool_calls"].append(
                {
                    "tool": "move_to",
                    "plan_only": args.operation == "plan_pick",
                    "result": action_result,
                }
            )
        elif args.operation == "pick":
            action_result = _call_public_tool_with_boundary(
                env,
                tool="pick",
                tool_role="subject",
                call_kwargs={
                    "hand": args.hand,
                    "target_xyz": target.tolist(),
                    "approach_vector": [0.0, 0.0, -1.0],
                    "grasp_quat_xyzw": grasp_quat.tolist(),
                    "pregrasp_offset_m": 0.08,
                    "lift_m": 0.08,
                    "timeout_s": _effective_pick_timeout_s(args),
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={
                    "visual_role": "radio_pick_subject",
                    "observer_camera": args.visual_observer_camera,
                    "observer_non_active_arm_gripper_joints_locked": (
                        args.visual_observer_camera is not None
                    ),
                    "shared_trunk_remains_active": True,
                    "observer_wrist_camera_world_pose_may_drift": (
                        args.visual_observer_camera is not None
                    ),
                    "fixed_observer_world_pose_claimed": False,
                },
            )
            if (
                args.post_pick_visual_validation
                and action_result.get("primitive_success") is True
            ):
                (
                    post_pick_validation_result,
                    report["post_pick_visual_validation"],
                ) = _run_post_pick_visual_validation(
                    env,
                    hand=args.hand,
                    public_target_xyz=target,
                    grasp_quat_xyzw=grasp_quat,
                    timeout_s=float(args.post_pick_visual_validation_timeout_s),
                    public_tool_calls=report["public_tool_calls"],
                    phase_boundaries=report["phase_boundaries"],
                    offset_world_m=args.post_pick_visual_offset_world_m,
                )
        elif args.operation == "held_chain":
            pick_result = _call_public_tool_with_boundary(
                env,
                tool="pick",
                tool_role="setup",
                call_kwargs={
                    "hand": args.hand,
                    "target_xyz": target.tolist(),
                    "approach_vector": [0.0, 0.0, -1.0],
                    "grasp_quat_xyzw": grasp_quat.tolist(),
                    "pregrasp_offset_m": 0.08,
                    "lift_m": 0.08,
                    "timeout_s": _effective_pick_timeout_s(args),
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={"visual_role": "held_chain_pick_setup"},
            )
            _require_primitive_success(pick_result, tool="held_chain setup pick")
            rotate_result = _call_public_tool_with_boundary(
                env,
                tool="rotate_wrist",
                tool_role="subject",
                call_kwargs={
                    "hand": args.hand,
                    "target_quat_xyzw": None,
                    "relative_axis_angle": list(HELD_ROTATION_AXIS_ANGLE_RAD),
                    "frame": "eef",
                    "timeout_s": float(args.rotate_timeout_s),
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={
                    "visual_role": "held_rotation_subject",
                    "rotation_degrees": 15.0,
                },
            )
            _require_primitive_success(rotate_result, tool="held_chain rotate_wrist")
            release_result = _call_public_tool_with_boundary(
                env,
                tool="release",
                tool_role="subject",
                call_kwargs={
                    "hand": args.hand,
                    "opening": 1.0,
                    "retreat_vector": [0.0, 0.0, 1.0],
                    "retreat_m": 0.03,
                    "timeout_s": float(args.release_timeout_s),
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={"visual_role": "held_release_subject"},
            )
            _require_primitive_success(release_result, tool="held_chain release")
            action_result = release_result
            report["held_chain"] = {
                "pick_setup": pick_result,
                "rotate_wrist_subject": rotate_result,
                "release_subject": release_result,
                "continuous_video_segment": str(action_dir / "episode.mp4"),
            }
        elif args.operation in {"press_preview", "press"}:
            view_target = target + np.asarray(
                PRESS_VIEW_OFFSET_WORLD_M, dtype=np.float64
            )
            view_derivation = {
                "source": "public_radio_target_xyz",
                "offset_world_m": list(PRESS_VIEW_OFFSET_WORLD_M),
                "formula": "public_radio_target_xyz + fixed_noncontact_view_offset",
                "orientation_source": "public_head_rgbd_grasp_quaternion",
                "orientation_is_calibrated_wrist_look_at": False,
                "orientation_limitation": (
                    "visual candidate only; no wrist-camera look-at extrinsic was invented"
                ),
                "uses_private_truth": False,
                "uses_backend_pose": False,
            }
            setup_result = _call_public_tool_with_boundary(
                env,
                tool="move_to",
                tool_role="setup",
                call_kwargs={
                    "hand": args.hand,
                    "target_xyz": view_target.tolist(),
                    "frame": "world",
                    "target_quat_xyzw": grasp_quat.tolist(),
                    "plan_only": False,
                    "position_tolerance_m": 0.02,
                    "orientation_tolerance_rad": 0.087,
                    "timeout_s": float(args.timeout_s),
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={
                    "visual_role": "press_noncontact_view_setup",
                    "target_derivation": view_derivation,
                },
            )
            _require_primitive_success(setup_result, tool="press view setup move_to")
            button_observed = _call_public_tool_with_boundary(
                env,
                tool="observe",
                tool_role="subject",
                call_kwargs={"camera": args.button_camera},
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={"visual_role": "button_wrist_rgbd"},
            )
            _require_primitive_success(button_observed, tool="button wrist observe")
            _save_public_rgbd_frame(
                env,
                camera=args.button_camera,
                observed=button_observed,
                output_dir=action_dir,
                prefix="button_after_setup",
            )
            report["press_view_setup"] = {
                "result": setup_result,
                "target_xyz": view_target.tolist(),
                "target_quat_xyzw": grasp_quat.tolist(),
                "target_derivation": view_derivation,
                "button_camera": args.button_camera,
                "button_frame_id": button_observed["frame_id"],
                "orientation_provenance": {
                    "source": "public_head_rgbd_grasp_quaternion",
                    "calibrated_wrist_look_at": False,
                },
            }
            report["press_preview_policy"] = {
                "visual_candidate_only": True,
                "automatically_authorizes_press": False,
                "manual_button_visibility_and_pixel_review_required": True,
            }
            action_result = setup_result
            if args.operation == "press":
                report["button_selection_review"] = _load_reviewed_button_selection(
                    Path(args.button_selection_source_json),
                    snapshot=snapshot,
                    hand=args.hand,
                    camera=args.button_camera,
                    setup_target_xyz=view_target,
                    cli_u=int(args.button_u),
                    cli_v=int(args.button_v),
                )
                button_projection = _call_public_tool_with_boundary(
                    env,
                    tool="pixel_to_world",
                    tool_role="subject",
                    call_kwargs={
                        "camera": args.button_camera,
                        "frame_id": button_observed["frame_id"],
                        "u": int(args.button_u),
                        "v": int(args.button_v),
                        "depth_window_px": int(args.button_depth_window_px),
                        "output_frame": "world",
                    },
                    public_tool_calls=report["public_tool_calls"],
                    phase_boundaries=report["phase_boundaries"],
                    metadata={"visual_role": "button_target"},
                )
                _require_public_projection_quality(
                    button_projection, visual_role="button_target"
                )
                button_target = np.asarray(
                    button_projection["diagnostics"]["xyz"], dtype=np.float64
                ).reshape(3)
                press_direction = _negative_public_surface_normal(button_projection)
                press_result = _call_public_tool_with_boundary(
                    env,
                    tool="press",
                    tool_role="subject",
                    call_kwargs={
                        "hand": args.hand,
                        "target_xyz": button_target.tolist(),
                        "press_direction": press_direction,
                        "approach_distance_m": 0.04,
                        "press_depth_m": 0.012,
                        "timeout_s": float(args.press_timeout_s),
                    },
                    public_tool_calls=report["public_tool_calls"],
                    phase_boundaries=report["phase_boundaries"],
                    metadata={
                        "visual_role": "button_press_subject",
                        "target_derivation": {
                            "source": "fresh_public_wrist_rgbd_pixel_to_world",
                            "camera": args.button_camera,
                            "u_column": int(args.button_u),
                            "v_row": int(args.button_v),
                            "press_direction_formula": "negative_normalized_surface_normal",
                            "uses_private_truth": False,
                            "uses_backend_pose": False,
                        },
                    },
                )
                _require_primitive_success(press_result, tool="press")
                action_result = press_result
                report["press_subject"] = {
                    "projection": button_projection,
                    "target_xyz": button_target.tolist(),
                    "press_direction": press_direction,
                    "result": press_result,
                }
        elif args.operation == "navigate":
            navigate_result = _call_public_tool_with_boundary(
                env,
                tool="navigate_to",
                tool_role="subject",
                call_kwargs={
                    "hand": args.hand,
                    "target_xyz": target.tolist(),
                    "frame": "world",
                    "standoff_m": float(args.navigate_standoff_m),
                    "timeout_s": float(args.navigate_timeout_s),
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={
                    "visual_role": "radio_navigation_subject",
                    "target_derivation": {
                        "source": "fresh_head_rgbd_pixel_to_world",
                        "u_column": int(args.u),
                        "v_row": int(args.v),
                        "uses_private_truth": False,
                    },
                },
            )
            _require_primitive_success(navigate_result, tool="navigate_to")
            post_navigate_observed = _call_public_tool_with_boundary(
                env,
                tool="observe",
                tool_role="targeting",
                call_kwargs={"camera": "head"},
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={"visual_role": "post_navigation_radio_reobserve"},
            )
            _require_primitive_success(
                post_navigate_observed, tool="post-navigation head observe"
            )
            post_navigate_projection = _call_public_tool_with_boundary(
                env,
                tool="pixel_to_world",
                tool_role="targeting",
                call_kwargs={
                    "camera": "head",
                    "frame_id": post_navigate_observed["frame_id"],
                    "u": int(args.u),
                    "v": int(args.v),
                    "depth_window_px": int(args.target_depth_window_px),
                    "output_frame": "world",
                },
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
                metadata={"visual_role": "post_navigation_radio_reprojection"},
            )
            _require_public_projection_quality(
                post_navigate_projection,
                visual_role="post_navigation_radio_target",
            )
            action_result = navigate_result
            report["navigate_closed_loop"] = {
                "navigate_result": navigate_result,
                "post_observe_frame_id": post_navigate_observed["frame_id"],
                "post_projection": post_navigate_projection,
                "initial_public_target_xyz": target.tolist(),
                "post_public_target_xyz": post_navigate_projection["diagnostics"][
                    "xyz"
                ],
            }

        if args.operation == "preview_pick" and action_result is not None:
            wrist_camera = f"{args.hand}_wrist"
            wrist_observed = env.observe(wrist_camera)
            report["public_tool_calls"].append(
                {"tool": "observe", "result": _jsonable(wrist_observed)}
            )
            wrist_frame = env._frame_cache.get_current(
                wrist_camera, wrist_observed["frame_id"]
            )
            _save_frame(wrist_frame, action_dir / f"after_{wrist_camera}.png")
            np.save(
                action_dir / f"after_{wrist_camera}_depth_m.npy",
                wrist_frame.depth_m,
            )
            (action_dir / f"after_{wrist_camera}_depth_visualization.png").write_bytes(
                wrist_observed["_depth_image_bytes"]
            )

        env._refresh_observation_without_step()
        assert env._last_observation is not None
        env._append_video(env._last_observation)
        final_frame = env._frame_cache.latest("head")
        _save_frame(final_frame, action_dir / "after_head.png")

        report["audit_private_after"] = _collect_posthoc_private_evidence(
            env,
            args=args,
            action_dir=action_dir,
            source_snapshot=snapshot,
        )
        report["private_visual_target_audit"] = report["audit_private_after"].get(
            "visual_target_audit",
            {
                "available": False,
                "does_not_gate_execution": True,
            },
        )

        if (
            action_result is not None
            and action_result.get("primitive_success") is not True
        ):
            raise RuntimeError(
                f"{args.operation} failed: {action_result.get('stop_reason')}"
            )
        if (
            post_pick_validation_result is not None
            and post_pick_validation_result.get("primitive_success") is not True
        ):
            raise RuntimeError(
                "post-pick visual validation move failed: "
                f"{post_pick_validation_result.get('stop_reason')}"
            )
        if args.operation == "preview_pick":
            import torch

            state = env.dump_simulator_state(serialized=True).detach().cpu()
            pregrasp_snapshot = output_dir / "radio_pregrasp_state.pt"
            torch.save(state, pregrasp_snapshot)
            try:
                import omnigibson as og

                omnigibson_version = getattr(og, "__version__", None)
            except Exception:
                omnigibson_version = None
            status_porcelain = subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True
            )
            tracked_diff = subprocess.check_output(
                ["git", "-C", str(repo), "diff", "--binary", "HEAD"]
            )
            manifest = build_snapshot_manifest(
                pregrasp_snapshot,
                serialized_elements=int(state.numel()),
                serialized_dtype=str(state.dtype),
                serialized_shape=list(state.shape),
                serialized_finite=bool(torch.isfinite(state).all().item()),
                meta=vars(env_args),
                source={
                    "commit": report["commit"],
                    "worktree_dirty": report["worktree_dirty"],
                    "status_porcelain_sha256": hashlib.sha256(
                        status_porcelain.encode("utf-8")
                    ).hexdigest(),
                    "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
                    "generator": str(Path(__file__).resolve()),
                    "source_snapshot": str(snapshot),
                },
                omnigibson_version=(
                    None if omnigibson_version is None else str(omnigibson_version)
                ),
                invariants={
                    "stage": "right_or_left_pregrasp_after_public_rgbd_targeting",
                    "hand": args.hand,
                    "radio_bddl_name": RADIO_BDDL_NAME,
                    "private_posthoc_audit": report["audit_private_after"],
                    "public_target_xyz": report["public_target_xyz"],
                    "public_grasp_geometry": report["public_grasp_geometry"],
                    "move_position_error_m": action_result.get("position_error_m"),
                    "move_orientation_error_rad": action_result.get(
                        "orientation_error_rad"
                    ),
                    "visual_review_required": True,
                },
            )
            manifest_path = write_snapshot_manifest(pregrasp_snapshot, manifest)
            report["generated_pregrasp_snapshot"] = {
                "path": str(pregrasp_snapshot),
                "sha256": _sha256(pregrasp_snapshot),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "serialized_elements": int(state.numel()),
            }
        if args.operation == "pick":
            other_hand = "left" if args.hand == "right" else "right"
            attachments = report["audit_private_after"].get("attachments", {})
            active_attachment = attachments.get(args.hand, {})
            inactive_attachment = attachments.get(other_hand, {})
            baseline = report["audit_private_after"].get("source_snapshot_baseline", {})
            report["private_pick_audit"] = {
                "purpose": "posthoc_record_only_not_acceptance_decision",
                "does_not_gate_execution": True,
                "active_hand_exact_radio": active_attachment.get("is_exact_radio"),
                "inactive_hand_empty": (
                    None
                    if not inactive_attachment
                    else (
                        inactive_attachment.get("backend_name") is None
                        and inactive_attachment.get("simulator_name") is None
                    )
                ),
                "radio_lift_m": baseline.get("radio_lift_m"),
                "radio_toggled_on": report["audit_private_after"].get(
                    "radio_toggled_on"
                ),
                "visual_review_required": True,
                "acceptance_decision": None,
            }
        report["execution_status"] = {
            "inspect": "capture_complete",
            "plan_pick": "plan_complete",
            "preview_pick": "runtime_preview_complete",
            "pick": "runtime_pick_complete",
            "held_chain": "runtime_held_chain_complete",
            "press_preview": "runtime_press_preview_complete",
            "press": "runtime_press_complete",
            "navigate": "runtime_navigate_complete",
            "observer_preview": "runtime_observer_preview_complete",
        }[args.operation]
        report["status"] = "pending_visual_review"
        save()
    except Exception as exc:
        report["status"] = "execution_failed"
        report["execution_status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if env is not None:
            report["failure_health_probe"] = _probe_public_env_health_after_failure(
                env,
                public_tool_calls=report["public_tool_calls"],
                phase_boundaries=report["phase_boundaries"],
            )
        save()
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                report["close_error"] = f"{type(exc).__name__}: {exc}"
                if report.get("execution_status") != "failed":
                    report["execution_status"] = "close_failed"
        video = action_dir / "episode.mp4"
        basic_video = {
            "path": str(video),
            "exists": video.is_file(),
            "bytes": video.stat().st_size if video.is_file() else 0,
            "sha256": _sha256(video) if video.is_file() else None,
        }
        try:
            report["video"] = _extract_visual_review_artifacts(
                video,
                action_dir,
                phase_boundaries=report["phase_boundaries"],
            )
        except Exception as exc:
            report["video"] = {
                **basic_video,
                "visual_review_required": True,
                "visual_decision": None,
                "automatic_visual_success_declared": False,
                "artifact_error": f"{type(exc).__name__}: {exc}",
            }
            if report.get("execution_status") != "failed":
                report["execution_status"] = "visual_artifact_failed"
        if report.get("status") != "execution_failed":
            report["status"] = "pending_visual_review"
        save()


if __name__ == "__main__":
    main()
