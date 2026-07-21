"""OmniGibson/R1Pro process for the BEHAVIOR RPent runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import threading
import time
from concurrent.futures import Future
from copy import deepcopy
from pathlib import Path
from queue import Empty, Queue
from typing import Any

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import numpy as np

from robots.behavior.camera_geometry import (
    CameraGeometryError,
    CameraIntrinsics,
    FrameCache,
    camera_point_from_pixel,
    canonical_camera,
    load_camera_correction_profiles,
    transform_point,
)
from robots.behavior.planner_executor import (
    EEF_LINK_BY_HAND,
    PRESS_EEF_TO_CONTACT_OFFSET_M,
    PlannerExecutor,
)
from robots.behavior.post_pick_debug_mirror import (
    DEBUG_MIRROR_CHECKPOINT_NAME,
    DEBUG_MIRROR_SCENE_NAME,
    build_debug_mirror_manifest,
    validate_debug_mirror_bundle,
    write_debug_mirror_manifest,
)
from robots.behavior.prepress import (
    BUTTON_FACE_CLASS,
    CLEAR_SLOTTED_BACK_FACE_CLASS,
    PREPRESS_AXIAL_STANDOFF_MAX_M,
    PREPRESS_AXIAL_STANDOFF_MIN_M,
    PREPRESS_LINE_DISTANCE_MAX_M,
    PREPRESS_OPPOSITION_ANGLE_MAX_DEG,
    PRESS_STAGING_AXIAL_STANDOFF_MAX_M,
    RADIO_LOCAL_BUTTON_CENTER_M,
    RADIO_LOCAL_BUTTON_FACE_NORMAL,
    RADIO_LOCAL_UP_AXIS,
    authorize_prepress_motion,
    direct_back_to_front_alignment,
    evaluate_geometry,
    gate_token,
    generate_button_goal_pose_candidates,
    generate_press_staging_pose_candidates,
    pose_matrix_xyzw,
    quat_multiply_xyzw,
    quat_rotate_xyzw,
    validate_button_declaration,
)
from robots.behavior.schemas import (
    CONTROL_MODES,
    ENV_ACTION_SEGMENTS,
    FULL_TASK_VLA_MODE,
    HYBRID_VLM_PI0_MODE,
    PI0_NAV_PICK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOLS_MODE,
    POLICY_STATE_SEGMENTS,
    extract_policy_state,
    segment_ranges,
    validate_action_chunk,
)
from robots.behavior.snapshot_manifest import validate_snapshot_manifest
from rpent.rpc_driver.socket import SocketRpcServer
from rpent.utils.config import get_repo_root, get_rlinf_repo_path
from rpent.utils.logging import get_logger

logger = get_logger("behavior_env_server")
RPENT_ROOT = get_repo_root()
RLINF_ROOT = get_rlinf_repo_path() or (RPENT_ROOT.parent / "RLinf_agentic_push")
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

_SHARED_ENV_RPC_METHODS = frozenset({"get_env_meta", "reset"})
_PLANNER_ENV_RPC_METHODS = frozenset(
    {
        "observe",
        "pixel_to_world",
        "navigate_to",
        "move_to",
        "pick",
        "rotate_wrist",
        "press",
        "release",
    }
)
_ENV_RPC_METHODS_BY_MODE = {
    FULL_TASK_VLA_MODE: frozenset({"chunk_step"}),
    PI0_PICK_VLA_MODE: frozenset({"pi0_chunk_step"}),
    PLANNER_TOOLS_MODE: _PLANNER_ENV_RPC_METHODS,
    HYBRID_VLM_PI0_MODE: (_PLANNER_ENV_RPC_METHODS - {"pick", "navigate_to"})
    | frozenset(
        {
            "current_observation",
            "pi0_chunk_step",
            "pi0_navigate_to_chunk_step",
            "restore_robot_state_checkpoint",
            "save_robot_state_checkpoint",
        }
    ),
    PI0_NAV_PICK_VLA_MODE: frozenset(
        {
            "current_observation",
            "declare_button_visibility",
            "evaluate_prepress_geometry",
            "finalize_paused_runtime",
            "inspect_post_pick_state",
            "inspect_toggle_geometry",
            "pixel_to_world",
            "prepress_move_to",
            "prepress_rotate_wrist",
            "observe",
            "pi0_nav_pick_chunk_step",
            "restore_robot_state_checkpoint",
            "save_post_pick_debug_mirror",
            "save_prepress_checkpoint",
            "save_robot_state_checkpoint",
            "press",
            "post_pick_close_press_gripper",
            "post_pick_direct_align",
            "post_pick_direct_advance",
            "post_pick_direct_press",
            "post_pick_direct_finger_toggle",
            "post_pick_recenter_held_button",
            "post_pick_visual_servo_align",
            "post_success_hold_frames",
            "post_success_retreat_and_open",
        }
    ),
}

_PLANNER_CONTROL_MODES = frozenset(
    {PLANNER_TOOLS_MODE, HYBRID_VLM_PI0_MODE, PI0_NAV_PICK_VLA_MODE}
)
_PLANNER_POSITION_START_MODES = frozenset({PLANNER_TOOLS_MODE, HYBRID_VLM_PI0_MODE})
_AUDIT_VIDEO_MODES = frozenset(
    {
        PLANNER_TOOLS_MODE,
        PI0_PICK_VLA_MODE,
        HYBRID_VLM_PI0_MODE,
        PI0_NAV_PICK_VLA_MODE,
    }
)

RESTORE_RENDER_SETTLE_FRAMES = 3
HANDOFF_VALIDATION_FRAMES = 8
HANDOFF_GRIPPER_OPENING_MAX = 0.045
HANDOFF_RADIO_LIFT_MIN_M = 0.04
HANDOFF_POST_RELOAD_LIFT_TOLERANCE_M = 0.002
HANDOFF_SUPPORT_GAP_MIN_M = 0.03
HANDOFF_WINDOW_MOTION_MIN_M = 0.008
HANDOFF_RELATIVE_DRIFT_MAX_M = 0.015
HANDOFF_COMOTION_RESIDUAL_MAX_M = 0.005
HANDOFF_MASK_GAP_MIN_PX = 3.0
HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD = 0.15
HANDOFF_ARTICULATION_ERROR_MAX_RAD = 0.05
POST_PICK_DEBUG_SAVE_POLICY = "debug_save_physics_warnings_non_blocking"
_STRICT_GRASP_WARNING_CODES = frozenset(
    {
        "controller_reload_pose_jump",
        "controller_hold_settling",
        "episode_status_during_handoff",
        "post_reload_grasp_not_strict",
        "held_object_stability_not_strict",
        "current_grasp_validator_unavailable",
        "current_grasp_not_strict",
        "post_pick_handoff_diagnostic_failed",
        "insufficient_handoff_horizon",
    }
)

_HANDOFF_VLA_ACTIVE = "VLA_ACTIVE"
_HANDOFF_CHECKPOINTING = "CHECKPOINTING"
_HANDOFF_CONTROLLER_RELOAD = "CONTROLLER_RELOAD"
_HANDOFF_STABLE_VALIDATION = "STABLE_VALIDATION"
_HANDOFF_PAUSED = "PAUSED"
_HANDOFF_FAILED = "FAILED"
_HANDOFF_OFFICIAL_SUCCESS = "OFFICIAL_SUCCESS"


def _post_pick_warning(
    code: str,
    message: str,
    *,
    metrics: Any | None = None,
) -> dict[str, Any]:
    """Build one wire-stable diagnostic that never decides persistence."""

    warning = {"code": str(code), "message": str(message)}
    if metrics is not None:
        warning["metrics"] = _wire_safe(metrics)
    return warning


def _handoff_controller_hold_warning(metrics: dict[str, Any]) -> bool:
    """Controller settling is diagnostic-only after the reload jump gate."""

    return bool(
        float(metrics["base_xy_error_m"]) > 0.02
        or float(metrics["base_yaw_error_rad"]) > 0.03
        or float(metrics["articulation_error_rad"]) > HANDOFF_ARTICULATION_ERROR_MAX_RAD
    )


def _handoff_held_object_stable(
    *, relative_drift_m: float, angular_drift_rad: float
) -> bool:
    """Hard handoff gate after reload: the held radio must remain stable."""

    return bool(
        float(relative_drift_m) <= HANDOFF_RELATIVE_DRIFT_MAX_M
        and float(angular_drift_rad) <= HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD
    )


def _post_reload_grasp_stable(
    *, selected: dict[str, Any], other: dict[str, Any]
) -> bool:
    """Allow millimetric controller settling after a strict Pi0 grasp pass."""

    criteria = selected.get("criteria")
    if not isinstance(criteria, dict):
        return False
    lift = float(selected.get("radio_lift_m", -np.inf))
    required = {
        "opening_strict",
        "support_gap",
        "selected_attachment_or_two_finger_contact",
        "other_hand_no_assisted_attachment",
        "other_hand_no_backend_attachment",
        "other_hand_no_radio_contact",
    }
    return bool(
        lift >= HANDOFF_RADIO_LIFT_MIN_M - HANDOFF_POST_RELOAD_LIFT_TOLERANCE_M
        and all(bool(criteria.get(name)) for name in required)
        and not bool(other.get("instantaneous_pass"))
    )


def _numpy_tree(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, dict):
        return {key: _numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_numpy_tree(item) for item in value)
    return value


def _settle_visual_pipeline_after_restore(
    simulator: Any, *, render_iterations: int = RESTORE_RENDER_SETTLE_FRAMES
) -> None:
    """Advance Kit's async renderer without advancing physics after restore."""

    iterations = int(render_iterations)
    if iterations < 3:
        raise ValueError("restored RGB-D synchronization requires at least 3 renders")
    for _ in range(iterations):
        simulator.render()


def _wire_safe(value: Any) -> Any:
    """Keep only wire-stable builtins and arrays at the simulator boundary."""
    value = _numpy_tree(value)
    if isinstance(value, np.generic):
        return _wire_safe(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return _wire_safe(value.tolist())
        return value
    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if key is None or isinstance(key, (str, bytes, bool, int, float)):
                safe_key = key
            else:
                safe_key = (
                    f"<key:{index}:{type(key).__module__}.{type(key).__qualname__}>"
                )
            result[safe_key] = _wire_safe(item)
        return result
    if isinstance(value, list):
        return [_wire_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_wire_safe(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return [_wire_safe(item) for item in value]
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    return f"<unserializable:{type(value).__module__}.{type(value).__qualname__}>"


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            default=lambda item: (
                item.tolist()
                if isinstance(item, np.ndarray)
                else item.item()
                if isinstance(item, np.generic)
                else repr(item)
            ),
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_scene_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _minimum_mask_gap_px(first: np.ndarray, second: np.ndarray) -> float:
    """Return exact Euclidean pixel separation, failing closed on empty masks."""

    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    if first_mask.shape != second_mask.shape or first_mask.ndim != 2:
        raise ValueError("instance masks must be equal-shape 2D arrays")
    if not first_mask.any() or not second_mask.any():
        raise ValueError("instance mask is empty")
    if np.logical_and(first_mask, second_mask).any():
        return 0.0
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception as exc:  # pragma: no cover - production dependency probe
        raise RuntimeError("scipy distance transform is unavailable") from exc
    distances = distance_transform_edt(~second_mask)
    gap = float(np.min(distances[first_mask]))
    if not np.isfinite(gap):
        raise RuntimeError("instance mask gap is non-finite")
    return gap


def _quaternion_angle_rad(first_xyzw: Any, second_xyzw: Any) -> float:
    first = np.asarray(first_xyzw, dtype=np.float64).reshape(4)
    second = np.asarray(second_xyzw, dtype=np.float64).reshape(4)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("quaternion contains NaN or Inf")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-9 or second_norm <= 1e-9:
        raise ValueError("quaternion norm is zero")
    dot = float(np.dot(first / first_norm, second / second_norm))
    return float(2.0 * np.arccos(np.clip(abs(dot), 0.0, 1.0)))


def _single_observation(obs: dict[str, Any]) -> dict[str, Any]:
    obs = _numpy_tree(obs)
    descriptions = obs.get("task_descriptions")
    return {
        "main_images": obs["main_images"][0],
        "wrist_images": obs["wrist_images"][0],
        "states": obs["states"][0],
        "task_descriptions": (
            descriptions[0] if isinstance(descriptions, list) else descriptions
        ),
    }


def _first_env_value(value: Any) -> Any:
    value = _numpy_tree(value)
    if isinstance(value, list):
        if not value:
            return None
        return value[0]
    if isinstance(value, tuple):
        if not value:
            return None
        return value[0]
    if isinstance(value, dict):
        return value
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return value


def _iter_sensor_payloads(value: Any, path: tuple[str, ...] = ()):
    if not isinstance(value, dict):
        return
    if any(key in value for key in ("rgb", "depth", "depth_linear")):
        yield "::".join(path), value
    for key, item in value.items():
        if isinstance(item, dict):
            yield from _iter_sensor_payloads(item, (*path, str(key)))


def _payload_camera_name(path: str) -> str | None:
    lowered = path.lower()
    if "zed_link" in lowered or "head" in lowered:
        return "head"
    if "left_realsense" in lowered or "left_wrist" in lowered:
        return "left_wrist"
    if "right_realsense" in lowered or "right_wrist" in lowered:
        return "right_wrist"
    return None


def _payload_depth(payload: dict[str, Any]) -> Any | None:
    for key in ("depth_linear", "depth", "depths"):
        if key in payload:
            return payload[key]
    return None


def _payload_rgb(payload: dict[str, Any]) -> Any | None:
    for key in ("rgb", "rgba", "image", "rgb_image"):
        if key in payload:
            return payload[key]
    return None


def _matrix_from_pose(position: Any, orientation_xyzw: Any) -> np.ndarray:
    pos = np.asarray(_numpy_tree(position), dtype=np.float64).reshape(3)
    quat = np.asarray(_numpy_tree(orientation_xyzw), dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise CameraGeometryError("camera orientation quaternion has zero norm")
    x, y, z, w = quat / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def _payload_matrix(
    payload: dict[str, Any], names: tuple[str, ...]
) -> np.ndarray | None:
    for name in names:
        if name not in payload:
            continue
        try:
            matrix = np.asarray(_numpy_tree(payload[name]), dtype=np.float64)
        except Exception:
            continue
        if matrix.shape == (4, 4) and np.isfinite(matrix).all():
            return matrix
        if matrix.shape == (3, 4) and np.isfinite(matrix).all():
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = matrix
            return out
    return None


def _payload_intrinsics(
    payload: dict[str, Any],
    *,
    rgb_shape: tuple[int, ...],
) -> CameraIntrinsics | None:
    for name in ("intrinsics", "intrinsic_matrix", "camera_intrinsics", "K"):
        if name not in payload:
            continue
        try:
            intrinsics = CameraIntrinsics.from_matrix(
                payload[name],
                width=int(rgb_shape[1]),
                height=int(rgb_shape[0]),
            )
            intrinsics.validate()
            return intrinsics
        except Exception:
            continue
    return None


def _sensor_intrinsics(
    sensor: Any, *, rgb_shape: tuple[int, ...]
) -> CameraIntrinsics | None:
    for name in ("intrinsic_matrix", "camera_intrinsics", "K"):
        try:
            value = getattr(sensor, name)
        except Exception:
            continue
        try:
            intrinsics = CameraIntrinsics.from_matrix(
                value,
                width=int(rgb_shape[1]),
                height=int(rgb_shape[0]),
            )
            intrinsics.validate()
            return intrinsics
        except Exception:
            pass
    for name in ("get_intrinsic_matrix", "get_camera_intrinsics"):
        fn = getattr(sensor, name, None)
        if fn is None:
            continue
        try:
            intrinsics = CameraIntrinsics.from_matrix(
                fn(),
                width=int(rgb_shape[1]),
                height=int(rgb_shape[0]),
            )
            intrinsics.validate()
            return intrinsics
        except Exception:
            pass
    width = int(rgb_shape[1])
    height = int(rgb_shape[0])

    def scalar(property_name: str, usd_attribute: str) -> float | None:
        try:
            value = getattr(sensor, property_name)
            number = float(np.asarray(_numpy_tree(value)).reshape(()))
            if np.isfinite(number):
                return number
        except Exception:
            pass
        getter = getattr(sensor, "get_attribute", None)
        if getter is None:
            return None
        try:
            number = float(np.asarray(_numpy_tree(getter(usd_attribute))).reshape(()))
            return number if np.isfinite(number) else None
        except Exception:
            return None

    focal_length = scalar("focal_length", "focalLength")
    horizontal_aperture = scalar("horizontal_aperture", "horizontalAperture")
    vertical_aperture = scalar("vertical_aperture", "verticalAperture")
    horizontal_offset = scalar("horizontal_aperture_offset", "horizontalApertureOffset")
    vertical_offset = scalar("vertical_aperture_offset", "verticalApertureOffset")
    if horizontal_offset not in (None, 0.0) or vertical_offset not in (None, 0.0):
        return None
    if (
        focal_length is not None
        and horizontal_aperture is not None
        and vertical_aperture is not None
        and focal_length > 0
        and horizontal_aperture > 0
        and vertical_aperture > 0
    ):
        intrinsics = CameraIntrinsics(
            fx=focal_length * width / horizontal_aperture,
            fy=focal_length * height / vertical_aperture,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )
        intrinsics.validate()
        return intrinsics
    return None


def _sensor_camera_to_world(sensor: Any) -> np.ndarray | None:
    # Kit publishes the render-synchronous world-to-camera matrix here.  A
    # sensor pose getter can already reflect a newer articulation state because
    # the renderer is asynchronous, so prefer this exact frame metadata when
    # available (the first query may legitimately be all zeros).
    try:
        parameters = getattr(sensor, "camera_parameters", None)
        view = parameters.get("cameraViewTransform") if parameters is not None else None
        matrix = np.asarray(_numpy_tree(view), dtype=np.float64).reshape(4, 4)
        if np.isfinite(matrix).all() and not np.allclose(matrix, 0.0):
            return np.linalg.inv(matrix.T)
    except Exception:
        pass
    for name in ("camera_to_world", "camera_to_world_matrix"):
        try:
            value = getattr(sensor, name)
        except Exception:
            continue
        try:
            matrix = np.asarray(_numpy_tree(value), dtype=np.float64)
            if matrix.shape == (4, 4) and np.isfinite(matrix).all():
                return matrix
        except Exception:
            pass
    for name in ("get_position_orientation", "get_world_pose"):
        fn = getattr(sensor, name, None)
        if fn is None:
            continue
        try:
            position, orientation = fn()
            return _matrix_from_pose(position, orientation)
        except Exception:
            pass
    return None


def _scalar_bool(value: Any) -> bool:
    try:
        import torch

        if torch.is_tensor(value):
            return bool(value.detach().cpu().any().item())
    except Exception:
        pass
    import numpy as np

    array = np.asarray(value)
    return bool(array.any()) if array.size else False


def _raw_success(info: Any) -> bool:
    return bool(
        isinstance(info, dict)
        and isinstance(info.get("done"), dict)
        and info["done"].get("success", False)
    )


def _raw_done(info: Any) -> bool:
    if not isinstance(info, dict) or not isinstance(info.get("done"), dict):
        return False
    conditions = info["done"].get("termination_conditions")
    if not isinstance(conditions, dict):
        return False
    return any(
        isinstance(value, dict) and bool(value.get("done", False))
        for value in conditions.values()
    )


def _bootstrap_template_path(
    instance_dir: Path,
    *,
    scene_model: str,
    task_name: str,
    activity_definition_id: int,
) -> Path:
    """Resolve the full scene template used before applying a tro-state instance."""
    template_path = instance_dir.parent / (
        f"{scene_model}_task_{task_name}_{activity_definition_id}_0_template.json"
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            f"BEHAVIOR bootstrap scene template not found: {template_path}"
        )
    return template_path


def _load_env_config(args: argparse.Namespace) -> Any:
    from omegaconf import OmegaConf

    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else RLINF_ROOT
        / "examples"
        / "embodiment"
        / "config"
        / "env"
        / "behavior_r1pro.yaml"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"BEHAVIOR env config not found: {config_path}")
    instance_dir = Path(args.activity_instance_dir).expanduser().resolve()
    if not instance_dir.is_dir():
        raise FileNotFoundError(
            f"BEHAVIOR instance directory not found: {instance_dir}"
        )

    cfg = OmegaConf.load(config_path)
    action_frequency = int(cfg.omni_config.env.action_frequency)
    if action_frequency != 60:
        raise ValueError(
            "BEHAVIOR video contract requires 60 Hz env actions, "
            f"got {action_frequency}"
        )
    cfg.seed = int(args.seed)
    cfg.total_num_envs = 1
    cfg.num_env_subprocess = 1
    cfg.direct_omnigibson_env = True
    cfg.auto_reset = False
    cfg.ignore_terminations = False
    cfg.skip_intermediate_obs_in_chunk = False
    cfg.max_episode_steps = int(args.max_episode_steps)
    cfg.max_steps_per_rollout_epoch = int(args.max_episode_steps)
    cfg.video_cfg.save_video = False
    cfg.action_trace_path = str(Path(args.output_dir) / "behavior_action_trace.jsonl")
    cfg.action_trace_interval = 1
    for robot_cfg in cfg.omni_config.robots:
        modalities = list(robot_cfg.get("obs_modalities") or [])
        for modality in ("rgb", "depth", "proprio"):
            if modality not in modalities:
                modalities.append(modality)
        robot_cfg.obs_modalities = modalities
    _configure_control_mode(cfg, getattr(args, "control_mode", None))

    task = cfg.omni_config.task
    task.activity_name = str(args.task_name)
    task.activity_definition_id = int(args.activity_definition_id)
    task.activity_instance_id = int(args.activity_instance_id)
    task.activity_instance_dir = str(instance_dir)
    task.instance_file_format = "tro_state"
    task.instance_resample_mode = "disabled"
    task.online_object_sampling = False
    task.predefined_problem = None
    task.use_presampled_robot_pose = True
    task.termination_config.max_steps = int(args.max_episode_steps)
    scene = cfg.omni_config.scene
    scene.scene_model = str(args.scene_model)
    # Official public instances are tro-state deltas, not complete scene templates.
    # Build the object scope from instance 0, then let ActivityInstanceLoader apply
    # the requested instance id immediately before the first reset.
    scene.scene_file = str(
        _bootstrap_template_path(
            instance_dir,
            scene_model=str(args.scene_model),
            task_name=str(args.task_name),
            activity_definition_id=int(args.activity_definition_id),
        )
    )
    scene.scene_instance = None
    return cfg


def _configure_control_mode(cfg: Any, control_mode: str | None) -> None:
    """Start planner surfaces in position mode; nav-pick starts as raw Pi0."""
    if control_mode not in _PLANNER_POSITION_START_MODES:
        return
    robots = list(cfg.omni_config.robots)
    if len(robots) != 1 or str(robots[0].type) != "R1Pro":
        raise ValueError("planner control requires exactly one R1Pro robot")
    base = robots[0].controller_config.base
    if str(base.name) != "HolonomicBaseJointController":
        raise ValueError(
            "planner control requires OmniGibson HolonomicBaseJointController"
        )
    base.motor_type = "position"
    base.command_input_limits = None
    base.command_output_limits = None
    base.use_impedances = False
    base.isaac_kp = 2_000_000.0
    base.isaac_kd = 100_000.0


def _resize_video_tile(rgb: Any, *, height: int, width: int) -> np.ndarray:
    """Resize one RGB camera image for a synchronized audit mosaic."""

    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] < 3:
        raise RuntimeError(f"video camera frame must be HxWxC, got {array.shape}")
    array = np.ascontiguousarray(array[..., :3])
    if array.shape[:2] == (int(height), int(width)):
        return array
    from PIL import Image

    return np.asarray(
        Image.fromarray(array, mode="RGB").resize(
            (int(width), int(height)),
            resample=Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )


class BehaviorEnvFacade:
    """Single-env raw-info facade with streaming 15 FPS video."""

    def __init__(
        self,
        *,
        cfg: Any,
        meta: dict[str, Any],
        output_dir: Path,
        control_mode: str | None = None,
        debug_mirror_path: str | Path | None = None,
    ) -> None:
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        self._env = BehaviorEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            record_metrics=False,
        )
        self._meta = dict(meta)
        self._control_mode = control_mode
        self._debug_mirror_path = (
            Path(debug_mirror_path).expanduser().resolve()
            if debug_mirror_path is not None
            else None
        )
        self._output_dir = Path(output_dir)
        self._done = False
        self._env_steps = 0
        self._video_path = (
            output_dir / "pi0_nav_pick_episode.mp4"
            if control_mode == PI0_NAV_PICK_VLA_MODE
            else output_dir / "episode.mp4"
        )
        self._video_writer = None
        self._video_frames = 0
        self._video_error: str | None = None
        self._video_sealed = False
        self._video_source_shapes: dict[str, list[int]] = {}
        self._planner_video_interval_steps = max(
            1, int(os.environ.get("RPENT_PLANNER_VIDEO_INTERVAL_STEPS", "4"))
        )
        correction_path = os.environ.get("RPENT_BEHAVIOR_CAMERA_CORRECTION_PROFILE")
        self._frame_cache = FrameCache(
            max_frames_per_camera=8,
            ttl_s=60.0,
            correction_profiles=load_camera_correction_profiles(correction_path),
        )
        self._planner = (
            PlannerExecutor(
                env=self,
                frame_cache=self._frame_cache,
                output_dir=output_dir,
            )
            if self._control_mode in _PLANNER_CONTROL_MODES
            else None
        )
        self._last_observation: dict[str, Any] | None = None
        self._last_info: Any = None
        self._gripper_latch = {"left": 1.0, "right": 1.0}
        self._restored_state: dict[str, Any] | None = None
        self._live_observation_counter = 0
        self._base_controller_mode = (
            "velocity"
            if self._control_mode == PI0_NAV_PICK_VLA_MODE
            else "position"
            if self._control_mode in _PLANNER_POSITION_START_MODES
            else None
        )
        self._action_source = (
            "pi0_vla" if self._control_mode == PI0_NAV_PICK_VLA_MODE else None
        )
        self._vla_actions_enabled = self._control_mode == PI0_NAV_PICK_VLA_MODE
        self._handoff_state = (
            _HANDOFF_VLA_ACTIVE if self._control_mode == PI0_NAV_PICK_VLA_MODE else None
        )
        self._held_hand: str | None = None
        self._reset_completed = False
        self._handoff_validator_frames: list[dict[str, Any]] = []
        self._handoff_target_objects: tuple[Any, Any] | None = None
        self._initial_radio_position: np.ndarray | None = None
        self._last_instance_id_masks: dict[str, np.ndarray] = {}
        self._last_capture_step: int | None = None
        self._validator_trace_path = output_dir / "handoff_validator_trace.json"
        self._state_checkpoint_path = (
            output_dir / "state_checkpoints" / "state_checkpoint_1.json"
        )
        self._paused_runtime_path = output_dir / "paused_runtime.json"
        self._prepress_context: dict[str, Any] | None = None
        self._prepress_gate: dict[str, Any] | None = None
        self._prepress_projection: dict[str, Any] | None = None
        self._prepress_geometry: dict[str, Any] | None = None
        self._prepress_motion: dict[str, Any] | None = None
        self._prepress_plan_certificate: dict[str, Any] | None = None
        self._prepress_coarse_flip_used = False
        self._prepress_round = 0
        self._last_handoff_failure: dict[str, Any] | None = None

    def _robot(self) -> Any | None:
        candidates = [
            self._env,
            getattr(self._env, "_env", None),
            getattr(self._env, "_direct_process", None),
            getattr(getattr(self._env, "_direct_process", None), "env", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            robots = getattr(candidate, "robots", None)
            if robots:
                return robots[0]
            envs = getattr(candidate, "envs", None)
            if envs:
                env = envs[0]
                seen: set[int] = set()
                while env is not None and id(env) not in seen:
                    seen.add(id(env))
                    robots = getattr(env, "robots", None)
                    if robots:
                        return robots[0]
                    env = getattr(env, "env", None) or getattr(env, "_env", None)
        return None

    def _require_planner(self) -> PlannerExecutor:
        if self._control_mode not in _PLANNER_CONTROL_MODES or self._planner is None:
            raise RuntimeError(
                "planner primitives are unavailable outside planner-enabled modes"
            )
        return self._planner

    @staticmethod
    def _object_position(obj: Any) -> np.ndarray:
        getter = getattr(obj, "get_position_orientation", None)
        if not callable(getter):
            raise RuntimeError("scene object has no pose API")
        position, _orientation = getter()
        value = np.asarray(_numpy_tree(position), dtype=np.float64).reshape(3)
        if not np.isfinite(value).all():
            raise RuntimeError("scene object position is non-finite")
        return value

    @staticmethod
    def _object_pose(obj: Any) -> tuple[np.ndarray, np.ndarray]:
        getter = getattr(obj, "get_position_orientation", None)
        if not callable(getter):
            raise RuntimeError("scene object has no pose API")
        position, orientation = getter()
        position_array = np.asarray(_numpy_tree(position), dtype=np.float64).reshape(3)
        orientation_array = np.asarray(
            _numpy_tree(orientation), dtype=np.float64
        ).reshape(4)
        if (
            not np.isfinite(position_array).all()
            or not np.isfinite(orientation_array).all()
        ):
            raise RuntimeError("scene object pose is non-finite")
        norm = float(np.linalg.norm(orientation_array))
        if norm <= 1e-9:
            raise RuntimeError("scene object orientation is invalid")
        return position_array, orientation_array / norm

    @staticmethod
    def _relative_pose(
        parent_position: np.ndarray,
        parent_quat_xyzw: np.ndarray,
        child_position: np.ndarray,
        child_quat_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return child pose in parent coordinates using xyzw quaternions."""

        def multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            x1, y1, z1, w1 = first
            x2, y2, z2, w2 = second
            return np.asarray(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ],
                dtype=np.float64,
            )

        inverse = np.asarray(
            [
                -parent_quat_xyzw[0],
                -parent_quat_xyzw[1],
                -parent_quat_xyzw[2],
                parent_quat_xyzw[3],
            ],
            dtype=np.float64,
        )
        delta = child_position - parent_position
        rotated = multiply(multiply(inverse, np.r_[delta, 0.0]), parent_quat_xyzw)[:3]
        relative_quat = multiply(inverse, child_quat_xyzw)
        relative_quat /= np.linalg.norm(relative_quat)
        return rotated, relative_quat

    @staticmethod
    def _object_vertical_bounds(obj: Any) -> tuple[float, float]:
        bbox = getattr(obj, "get_base_aligned_bbox", None)
        if not callable(bbox):
            raise RuntimeError("scene object has no base-aligned bounding box")
        center, _quat, extent, _center_local = bbox(xy_aligned=True)
        center_array = np.asarray(_numpy_tree(center), dtype=np.float64).reshape(3)
        extent_array = np.asarray(_numpy_tree(extent), dtype=np.float64).reshape(3)
        if not np.isfinite(center_array).all() or not np.isfinite(extent_array).all():
            raise RuntimeError("scene object bounding box is non-finite")
        if np.any(extent_array <= 0.0):
            raise RuntimeError("scene object bounding box extent is not positive")
        return (
            float(center_array[2] - 0.5 * extent_array[2]),
            float(center_array[2] + 0.5 * extent_array[2]),
        )

    def _resolve_handoff_targets(self) -> tuple[Any, Any]:
        if self._handoff_target_objects is not None:
            return self._handoff_target_objects
        instance_dir = Path(str(self._meta["activity_instance_dir"]))
        instance_path = instance_dir / (
            f"{self._meta['scene_model']}_task_{self._meta['task_name']}_"
            f"{int(self._meta['activity_definition_id'])}_"
            f"{int(self._meta['activity_instance_id'])}_template-tro_state.json"
        )
        if not instance_path.is_file():
            raise RuntimeError(f"handoff instance state is missing: {instance_path}")
        instance = json.loads(instance_path.read_text(encoding="utf-8"))
        object_names = [str(name) for name in instance if str(name) != "robot_poses"]
        radio_names = [name for name in object_names if "radio_receiver" in name]
        table_names = [name for name in object_names if name.startswith("table.")]
        if len(radio_names) != 1 or len(table_names) != 1:
            raise RuntimeError(
                "handoff requires exactly one radio_receiver and one table in "
                f"the task instance, got radio={radio_names!r} table={table_names!r}"
            )
        robot = self._robot()
        omni_env = getattr(self._env, "omnigibson_env", None)
        task = getattr(omni_env, "task", None)
        object_scope = getattr(task, "object_scope", None)
        scene = getattr(omni_env, "scene", None)
        if scene is None:
            scene = getattr(robot, "scene", None) if robot is not None else None
        objects = getattr(scene, "objects", None)
        if isinstance(objects, dict):
            objects = list(objects.values())
        else:
            objects = list(objects or ())

        def resolve(spec_name: str) -> Any:
            if isinstance(object_scope, dict):
                scoped = object_scope.get(spec_name)
                if scoped is not None:
                    return scoped
            wanted = _normalized_scene_name(spec_name)
            matches = []
            for obj in objects:
                names = {
                    _normalized_scene_name(getattr(obj, "name", "")),
                    _normalized_scene_name(
                        str(getattr(obj, "prim_path", "")).rstrip("/").split("/")[-1]
                    ),
                }
                if wanted in names:
                    matches.append(obj)
            if len(matches) != 1:
                raise RuntimeError(
                    f"handoff scene object {spec_name!r} resolved to {len(matches)} bodies"
                )
            return matches[0]

        radio = resolve(radio_names[0])
        table = resolve(table_names[0])
        self._object_position(radio)
        self._object_vertical_bounds(radio)
        self._object_vertical_bounds(table)
        self._handoff_target_objects = (radio, table)
        return radio, table

    @staticmethod
    def _object_mask(instance_ids: np.ndarray, obj: Any) -> np.ndarray:
        try:
            from omnigibson.sensors.vision_sensor import VisionSensor
        except Exception as exc:
            raise RuntimeError("OmniGibson instance registry is unavailable") from exc
        target_root = str(getattr(obj, "prim_path", "")).rstrip("/")
        if not target_root:
            raise RuntimeError("mask target has no prim_path")
        registry = dict(getattr(VisionSensor, "INSTANCE_ID_REGISTRY", {}))
        matching_ids = [
            int(instance_id)
            for instance_id, label in registry.items()
            if str(label).rstrip("/") == target_root
            or str(label).rstrip("/").startswith(f"{target_root}/")
        ]
        if not matching_ids:
            raise RuntimeError(f"instance registry has no labels for {target_root}")
        return np.isin(np.asarray(instance_ids), np.asarray(matching_ids))

    def _radio_table_mask_report(self, radio: Any, table: Any) -> dict[str, Any]:
        if self._last_capture_step != self._env_steps:
            raise RuntimeError(
                "instance masks are not synchronized to the current step"
            )
        if set(self._last_instance_id_masks) != {
            "head",
            "left_wrist",
            "right_wrist",
        }:
            raise RuntimeError("current three-camera instance masks are unavailable")
        gaps: dict[str, float | None] = {}
        passing_cameras = []
        errors: dict[str, str] = {}
        for camera, instance_ids in self._last_instance_id_masks.items():
            try:
                radio_mask = self._object_mask(instance_ids, radio)
                table_mask = self._object_mask(instance_ids, table)
                gap = _minimum_mask_gap_px(radio_mask, table_mask)
            except Exception as exc:
                gaps[camera] = None
                errors[camera] = f"{type(exc).__name__}: {exc}"
                continue
            gaps[camera] = gap
            if gap >= HANDOFF_MASK_GAP_MIN_PX:
                passing_cameras.append(camera)
        return {
            "available": any(value is not None for value in gaps.values()),
            "passed": bool(passing_cameras),
            "minimum_required_px": HANDOFF_MASK_GAP_MIN_PX,
            "gaps_px": gaps,
            "passing_cameras": passing_cameras,
            "errors": errors,
        }

    def _attachment_matches(self, hand: str, target: Any) -> tuple[bool, bool]:
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        assisted = getattr(robot, "_ag_obj_in_hand", {})
        assisted_obj = assisted.get(hand) if isinstance(assisted, dict) else None
        target_root = str(getattr(target, "prim_path", "")).rstrip("/")
        assisted_root = str(getattr(assisted_obj, "prim_path", "")).rstrip("/")
        assisted_match = bool(
            assisted_obj is not None
            and target_root
            and (
                assisted_root == target_root
                or assisted_root.startswith(f"{target_root}/")
            )
        )
        backend_attachment = self._require_planner().backend.get_attached_object(hand)
        backend_match = False
        if isinstance(backend_attachment, dict):
            for attached in backend_attachment.values():
                root = str(getattr(attached, "prim_path", "")).rstrip("/")
                backend_match = backend_match or bool(
                    target_root
                    and (root == target_root or root.startswith(f"{target_root}/"))
                )
        return assisted_match, backend_match

    def _hand_target_contact_report(
        self, hand: str, radio_position: np.ndarray
    ) -> dict[str, Any]:
        report = self._require_planner().backend.contact_report(
            hand=hand,
            target_xyz=np.asarray(radio_position, dtype=np.float64),
        )
        if not isinstance(report, dict) or not bool(report.get("available", False)):
            raise RuntimeError(
                f"{hand} gripper contact report is unavailable: {report!r}"
            )
        count = report.get("target_contact_count")
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise RuntimeError(f"{hand} target contact count is invalid")
        return {
            "available": True,
            "target_contact_count": int(count),
            "target_two_finger_contact": bool(
                report.get("target_two_finger_contact", False)
            ),
        }

    def _handoff_validator_frame(self, observation: dict[str, Any]) -> dict[str, Any]:
        radio, table = self._resolve_handoff_targets()
        radio_position = self._object_position(radio)
        if self._initial_radio_position is None:
            raise RuntimeError("initial radio position was not captured at reset")
        radio_bottom_z, _radio_top_z = self._object_vertical_bounds(radio)
        _table_bottom_z, table_top_z = self._object_vertical_bounds(table)
        mask_report = self._radio_table_mask_report(radio, table)
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        eef_links = getattr(robot, "eef_links", {})
        per_hand: dict[str, Any] = {}
        attachment_by_hand: dict[str, tuple[bool, bool]] = {}
        contact_by_hand: dict[str, dict[str, Any]] = {}
        eef_by_hand: dict[str, np.ndarray] = {}
        for hand in ("left", "right"):
            link = eef_links.get(hand) if isinstance(eef_links, dict) else None
            if link is None or not callable(
                getattr(link, "get_position_orientation", None)
            ):
                raise RuntimeError(f"{hand} EEF pose is unavailable")
            eef_position, _orientation = link.get_position_orientation()
            eef = np.asarray(_numpy_tree(eef_position), dtype=np.float64).reshape(3)
            if not np.isfinite(eef).all():
                raise RuntimeError(f"{hand} EEF position is non-finite")
            eef_by_hand[hand] = eef
            attachment_by_hand[hand] = self._attachment_matches(hand, radio)
            contact_by_hand[hand] = self._hand_target_contact_report(
                hand, radio_position
            )
        radio_lift = float(radio_position[2] - self._initial_radio_position[2])
        support_gap = float(radio_bottom_z - table_top_z)
        for hand in ("left", "right"):
            other = "right" if hand == "left" else "left"
            opening = self._validated_selected_gripper_opening(
                observation=observation,
                hand=hand,
            )
            selected_assisted, selected_backend = attachment_by_hand[hand]
            selected_contact = contact_by_hand[hand]
            other_assisted, other_backend = attachment_by_hand[other]
            criteria = {
                "opening_strict": bool(opening < HANDOFF_GRIPPER_OPENING_MAX),
                "radio_lift": bool(radio_lift >= HANDOFF_RADIO_LIFT_MIN_M),
                "support_gap": bool(support_gap >= HANDOFF_SUPPORT_GAP_MIN_M),
                "selected_attachment_or_two_finger_contact": bool(
                    selected_assisted
                    or selected_backend
                    or selected_contact["target_two_finger_contact"]
                ),
                "other_hand_no_assisted_attachment": not other_assisted,
                "other_hand_no_backend_attachment": not other_backend,
                "other_hand_no_radio_contact": (
                    contact_by_hand[other]["target_contact_count"] == 0
                ),
            }
            per_hand[hand] = {
                "opening": float(opening),
                "opening_max_exclusive": HANDOFF_GRIPPER_OPENING_MAX,
                "radio_position": radio_position.astype(float).tolist(),
                "eef_position": eef_by_hand[hand].astype(float).tolist(),
                "radio_lift_m": radio_lift,
                "radio_lift_min_inclusive_m": HANDOFF_RADIO_LIFT_MIN_M,
                "support_gap_m": support_gap,
                "support_gap_min_inclusive_m": HANDOFF_SUPPORT_GAP_MIN_M,
                "selected_assisted_attachment": selected_assisted,
                "selected_backend_attachment": selected_backend,
                "selected_contact": selected_contact,
                "other_contact": contact_by_hand[other],
                "criteria": criteria,
                "instantaneous_pass": all(criteria.values()),
            }
        frame = {
            "total_env_steps": int(self._env_steps),
            "capture_step": self._last_capture_step,
            "mask_report": mask_report,
            "per_hand": per_hand,
        }
        if not all(np.isfinite(value) for value in (radio_lift, support_gap)):
            raise RuntimeError("handoff validator geometry is non-finite")
        return frame

    @staticmethod
    def _handoff_window_metrics(
        frames: list[dict[str, Any]], hand: str
    ) -> dict[str, Any]:
        radio = np.asarray(
            [frame["per_hand"][hand]["radio_position"] for frame in frames],
            dtype=np.float64,
        )
        eef = np.asarray(
            [frame["per_hand"][hand]["eef_position"] for frame in frames],
            dtype=np.float64,
        )
        if radio.shape != (HANDOFF_VALIDATION_FRAMES, 3) or eef.shape != (
            HANDOFF_VALIDATION_FRAMES,
            3,
        ):
            raise RuntimeError("handoff validator window shape is invalid")
        if not np.isfinite(radio).all() or not np.isfinite(eef).all():
            raise RuntimeError("handoff validator window contains NaN or Inf")
        radio_motion = float(np.linalg.norm(radio[-1] - radio[0]))
        eef_motion = float(np.linalg.norm(eef[-1] - eef[0]))
        relative = radio - eef
        relative_drift = np.linalg.norm(relative - relative[0], axis=1)
        max_relative_drift = float(np.max(relative_drift))
        comotion = np.linalg.norm(np.diff(radio, axis=0) - np.diff(eef, axis=0), axis=1)
        mean_comotion = float(np.mean(comotion))
        criteria = {
            "current_instantaneous_frame": bool(
                frames[-1]["per_hand"][hand]["instantaneous_pass"]
            ),
            "held_opening_all_frames": all(
                bool(
                    frame["per_hand"][hand]
                    .get("criteria", {})
                    .get("opening_strict", False)
                )
                for frame in frames
            ),
            "radio_window_motion": radio_motion >= HANDOFF_WINDOW_MOTION_MIN_M,
            "eef_window_motion": eef_motion >= HANDOFF_WINDOW_MOTION_MIN_M,
            "relative_drift": max_relative_drift <= HANDOFF_RELATIVE_DRIFT_MAX_M,
            "mean_comotion_residual": (
                mean_comotion <= HANDOFF_COMOTION_RESIDUAL_MAX_M
            ),
        }
        return {
            "radio_displacement_m": radio_motion,
            "eef_displacement_m": eef_motion,
            "minimum_displacement_m": HANDOFF_WINDOW_MOTION_MIN_M,
            "max_relative_drift_m": max_relative_drift,
            "max_relative_drift_limit_m": HANDOFF_RELATIVE_DRIFT_MAX_M,
            "mean_comotion_residual_m": mean_comotion,
            "mean_comotion_residual_limit_m": HANDOFF_COMOTION_RESIDUAL_MAX_M,
            "criteria": criteria,
            "passed": all(criteria.values()),
        }

    def _update_handoff_validator(self, observation: dict[str, Any]) -> dict[str, Any]:
        try:
            frame = self._handoff_validator_frame(observation)
        except Exception as exc:
            frame = {
                "total_env_steps": int(self._env_steps),
                "capture_step": self._last_capture_step,
                "error": f"{type(exc).__name__}: {exc}",
                "per_hand": {
                    hand: {"instantaneous_pass": False, "criteria": {}}
                    for hand in ("left", "right")
                },
            }
        self._handoff_validator_frames.append(frame)
        window = self._handoff_validator_frames[-HANDOFF_VALIDATION_FRAMES:]
        per_hand: dict[str, Any] = {}
        if len(window) == HANDOFF_VALIDATION_FRAMES:
            for hand in ("left", "right"):
                try:
                    per_hand[hand] = self._handoff_window_metrics(window, hand)
                except Exception as exc:
                    per_hand[hand] = {
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        else:
            per_hand = {
                hand: {
                    "passed": False,
                    "frames_collected": len(window),
                    "frames_required": HANDOFF_VALIDATION_FRAMES,
                }
                for hand in ("left", "right")
            }
        passed_hands = [hand for hand in ("left", "right") if per_hand[hand]["passed"]]
        held_hand = passed_hands[0] if len(passed_hands) == 1 else None
        result = {
            "frames_required": HANDOFF_VALIDATION_FRAMES,
            "frames_collected": len(window),
            "unique_pass_required": True,
            "passed_hands": passed_hands,
            "held_hand": held_hand,
            "local_grasp_success": held_hand is not None,
            "current": frame,
            "per_hand": per_hand,
        }
        _write_json_atomic(
            self._validator_trace_path,
            {
                "schema_version": 1,
                "frames": self._handoff_validator_frames,
                "latest": result,
            },
        )
        return result

    @staticmethod
    def _validated_handoff_hands(*, held_hand: str, press_hand: str) -> tuple[str, str]:
        if held_hand not in {"left", "right"}:
            raise ValueError("held_hand must be 'left' or 'right'")
        if press_hand not in {"left", "right"}:
            raise ValueError("press_hand must be 'left' or 'right'")
        if held_hand == press_hand:
            raise ValueError("held_hand and press_hand must be different")
        return held_hand, press_hand

    def _robot_state_checkpoint_payload(
        self,
        *,
        checkpoint_name: str,
        stage: str,
        held_hand: str,
        press_hand: str,
        object_name: str,
        require_current_grasp: bool,
        validation_evidence: dict[str, Any] | None,
    ) -> dict[str, Any]:
        held_hand, press_hand = self._validated_handoff_hands(
            held_hand=held_hand, press_hand=press_hand
        )
        radio, _table = self._resolve_handoff_targets()
        resolved_name = str(getattr(radio, "name", ""))
        if not object_name or _normalized_scene_name(object_name) not in {
            _normalized_scene_name(resolved_name),
            _normalized_scene_name("radio_receiver"),
            _normalized_scene_name("radio"),
        }:
            raise ValueError(
                f"object_name {object_name!r} does not identify the task radio"
            )
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(q).all():
            raise RuntimeError("robot joint motion state is invalid")
        robot_position, robot_orientation = self._object_pose(robot)
        radio_position, radio_orientation = self._object_pose(radio)
        eef_links = getattr(robot, "eef_links", {})
        held_link = eef_links.get(held_hand) if isinstance(eef_links, dict) else None
        press_link = eef_links.get(press_hand) if isinstance(eef_links, dict) else None
        if held_link is None or press_link is None:
            raise RuntimeError("held or press EEF link is unavailable")
        eef_position, eef_orientation = self._object_pose(held_link)
        press_eef_position, press_eef_orientation = self._object_pose(press_link)
        relative_position, relative_orientation = self._relative_pose(
            eef_position,
            eef_orientation,
            radio_position,
            radio_orientation,
        )
        base_indices = list(
            np.asarray(_numpy_tree(getattr(robot, "base_control_idx", [])), dtype=int)
        )
        trunk_indices = list(
            np.asarray(_numpy_tree(getattr(robot, "trunk_control_idx", [])), dtype=int)
        )
        arm_indices = getattr(robot, "arm_control_idx", {}) or {}
        left_indices = list(
            np.asarray(_numpy_tree(arm_indices.get("left", [])), dtype=int)
        )
        right_indices = list(
            np.asarray(_numpy_tree(arm_indices.get("right", [])), dtype=int)
        )
        if not (
            len(base_indices) == 3
            and len(trunk_indices) == 4
            and len(left_indices) == 7
            and len(right_indices) == 7
        ):
            raise RuntimeError("R1Pro controlled joint layout is unavailable")
        controlled_indices = base_indices + trunk_indices + left_indices + right_indices
        if min(controlled_indices) < 0 or max(controlled_indices) >= len(q):
            raise RuntimeError("R1Pro controlled joint indices are invalid")
        observation = self._last_observation
        if observation is None:
            raise RuntimeError("checkpoint requires a current public observation")
        openings = {
            side: self._validated_selected_gripper_opening(
                observation=observation, hand=side
            )
            for side in ("left", "right")
        }
        evidence = validation_evidence
        if evidence is None and self._handoff_validator_frames:
            evidence = {
                "latest_frame": self._handoff_validator_frames[-1],
                "validator_trace_path": str(self._validator_trace_path),
            }
        nonblocking_post_pick_save = bool(
            self._control_mode == PI0_NAV_PICK_VLA_MODE
            and checkpoint_name == "state_checkpoint_1"
        )
        checkpoint_warnings = list(
            (evidence or {}).get("post_pick_warnings", [])
            if isinstance(evidence, dict)
            else []
        )
        checkpoint_grasp_pass = not require_current_grasp
        current_validation: dict[str, Any] | None = None
        if require_current_grasp:
            settling_aware = False
            try:
                current_validation = self._handoff_validator_frame(observation)
                selected = current_validation["per_hand"][held_hand]
                other = current_validation["per_hand"][press_hand]
                stable_frames = (
                    (evidence or {}).get("stable_hold_frames", [])
                    if isinstance(evidence, dict)
                    else []
                )
                settling_aware = bool(
                    nonblocking_post_pick_save
                    and stage == "post_pi0_nav_pick"
                    and isinstance(stable_frames, list)
                    and len(stable_frames) == HANDOFF_VALIDATION_FRAMES
                )
                checkpoint_grasp_pass = bool(
                    _post_reload_grasp_stable(selected=selected, other=other)
                    if settling_aware
                    else bool(selected.get("instantaneous_pass", False))
                    and not bool(other.get("instantaneous_pass", False))
                )
            except Exception as exc:
                if not nonblocking_post_pick_save:
                    raise
                checkpoint_grasp_pass = False
                checkpoint_warnings.append(
                    _post_pick_warning(
                        "current_grasp_validator_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            if not checkpoint_grasp_pass:
                if not nonblocking_post_pick_save:
                    raise RuntimeError(
                        "checkpoint current-grasp validation did not uniquely pass held_hand"
                    )
                checkpoint_warnings.append(
                    _post_pick_warning(
                        "current_grasp_not_strict",
                        "checkpoint current-grasp validator did not uniquely pass held_hand",
                        metrics=current_validation,
                    )
                )
            evidence = {
                **(evidence or {}),
                "checkpoint_current_validation": current_validation,
                "checkpoint_current_validation_policy": (
                    "post_reload_2mm_lift_settling_tolerance"
                    if settling_aware
                    else "strict_instantaneous"
                ),
            }
        held_validation = (
            current_validation["per_hand"][held_hand]
            if isinstance(current_validation, dict)
            else {}
        )
        press_validation = (
            current_validation["per_hand"][press_hand]
            if isinstance(current_validation, dict)
            else {}
        )
        lift_delta = (
            float(radio_position[2] - self._initial_radio_position[2])
            if (self._initial_radio_position is not None)
            else None
        )
        return {
            "schema_version": 1,
            "kind": "robot_motion_checkpoint",
            "not_simulator_restore": True,
            "checkpoint_name": checkpoint_name,
            "stage": stage,
            "strict_local_grasp_success": bool(checkpoint_grasp_pass),
            "usable_post_pick_saved": bool(nonblocking_post_pick_save),
            "save_policy": (
                POST_PICK_DEBUG_SAVE_POLICY
                if nonblocking_post_pick_save
                else "strict_checkpoint_validation"
            ),
            "warnings": _wire_safe(checkpoint_warnings),
            "env_step": int(self._env_steps),
            "held_hand": held_hand,
            "press_hand": press_hand,
            "object_name": resolved_name or object_name,
            "robot": {
                "base_xy_yaw": q[base_indices].astype(float).tolist(),
                "trunk_q": q[trunk_indices].astype(float).tolist(),
                "left_arm_q": q[left_indices].astype(float).tolist(),
                "right_arm_q": q[right_indices].astype(float).tolist(),
                "left_gripper": {
                    "command": float(self._gripper_latch["left"]),
                    "opening": float(openings["left"]),
                    "opening_threshold_exclusive": HANDOFF_GRIPPER_OPENING_MAX,
                    "close_command": -1.0,
                },
                "right_gripper": {
                    "command": float(self._gripper_latch["right"]),
                    "opening": float(openings["right"]),
                    "opening_threshold_exclusive": HANDOFF_GRIPPER_OPENING_MAX,
                    "close_command": -1.0,
                },
                "left_gripper_command": float(self._gripper_latch["left"]),
                "left_gripper_opening_m": float(openings["left"]),
                "right_gripper_command": float(self._gripper_latch["right"]),
                "right_gripper_opening_m": float(openings["right"]),
                "gripper_opening_threshold_exclusive_m": HANDOFF_GRIPPER_OPENING_MAX,
                "held_gripper_close_command": -1.0,
                "q_space_target": {
                    "indices": [int(index) for index in controlled_indices],
                    "values": q[controlled_indices].astype(float).tolist(),
                },
            },
            "poses": {
                "held_eef_pose_world": {
                    "position": eef_position.astype(float).tolist(),
                    "quat_xyzw": eef_orientation.astype(float).tolist(),
                },
                "press_eef_pose_world": {
                    "position": press_eef_position.astype(float).tolist(),
                    "quat_xyzw": press_eef_orientation.astype(float).tolist(),
                },
                "object_pose_world": {
                    "position": radio_position.astype(float).tolist(),
                    "quat_xyzw": radio_orientation.astype(float).tolist(),
                },
                "object_pose_in_held_eef": {
                    "position": relative_position.astype(float).tolist(),
                    "quat_xyzw": relative_orientation.astype(float).tolist(),
                },
                "robot_pose_world": {
                    "position": robot_position.astype(float).tolist(),
                    "quat_xyzw": robot_orientation.astype(float).tolist(),
                },
            },
            "validation": {
                "require_current_grasp": bool(require_current_grasp),
                "strict_local_grasp_success": bool(checkpoint_grasp_pass),
                "usable_post_pick_saved": bool(nonblocking_post_pick_save),
                "save_policy": (
                    POST_PICK_DEBUG_SAVE_POLICY
                    if nonblocking_post_pick_save
                    else "strict_checkpoint_validation"
                ),
                "warnings": _wire_safe(checkpoint_warnings),
                "held_gripper_closed": bool(
                    held_validation.get("criteria", {}).get("opening_strict", False)
                ),
                "held_object_stable": bool(checkpoint_grasp_pass),
                "inactive_hand_clear": bool(
                    not press_validation.get("instantaneous_pass", False)
                    and held_validation.get("criteria", {}).get(
                        "other_hand_no_assisted_attachment", False
                    )
                    and held_validation.get("criteria", {}).get(
                        "other_hand_no_backend_attachment", False
                    )
                    and held_validation.get("criteria", {}).get(
                        "other_hand_no_radio_contact", False
                    )
                ),
                "lift_delta_m": lift_delta,
                "official_task_success": bool(_raw_success(self._last_info)),
                "validator_trace_path": str(self._validator_trace_path),
                "evidence_summary": _wire_safe(
                    {
                        "stable_post_reload_frame_count": len(
                            (evidence or {}).get("stable_hold_frames", [])
                        ),
                        "strict_held_hand": (evidence or {})
                        .get("strict_vla_window", {})
                        .get("held_hand"),
                        "controller_reload": (evidence or {}).get(
                            "controller_reload", {}
                        ),
                    }
                ),
            },
            "visual_evidence": {},
        }

    def save_robot_state_checkpoint(
        self,
        *,
        checkpoint_name: str,
        stage: str,
        held_hand: str,
        press_hand: str,
        object_name: str,
        require_current_grasp: bool = True,
        visual_review: bool = True,
    ) -> dict[str, Any]:
        """Atomically save robot motion state, never a whole-scene simulator dump."""

        if getattr(self, "_control_mode", None) == PI0_NAV_PICK_VLA_MODE:
            if checkpoint_name != "state_checkpoint_2":
                raise RuntimeError(
                    "post-pick state_checkpoint_1 is handoff-owned and immutable"
                )
            if stage != "pre_press_alignment":
                raise ValueError("state_checkpoint_2 stage must be pre_press_alignment")
            if require_current_grasp is not True or visual_review is not True:
                raise ValueError(
                    "state_checkpoint_2 requires current grasp and visual review"
                )
            return self.save_prepress_checkpoint(
                checkpoint_name="state_checkpoint_2",
                stage="pre_press_alignment",
                visual_review=True,
            )

        if not isinstance(checkpoint_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", checkpoint_name
        ):
            raise ValueError("checkpoint_name is not a safe artifact stem")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(require_current_grasp, bool) or not isinstance(
            visual_review, bool
        ):
            raise ValueError("require_current_grasp and visual_review must be booleans")
        self._state_checkpoint_path = (
            self._output_dir / "state_checkpoints" / f"{checkpoint_name}.json"
        )
        payload = self._robot_state_checkpoint_payload(
            checkpoint_name=checkpoint_name,
            stage=stage,
            held_hand=held_hand,
            press_hand=press_hand,
            object_name=object_name,
            require_current_grasp=require_current_grasp,
            validation_evidence=None,
        )
        self._state_checkpoint_path.unlink(missing_ok=True)
        self._state_checkpoint_path.with_suffix(
            self._state_checkpoint_path.suffix + ".tmp"
        ).unlink(missing_ok=True)
        visual_paths = (
            self._checkpoint_visual_evidence(
                checkpoint_name=checkpoint_name,
                held_hand=held_hand,
                press_hand=press_hand,
            )
            if visual_review
            else {}
        )
        payload["visual_evidence"] = visual_paths
        _write_json_atomic(self._state_checkpoint_path, payload)
        return {
            "_finish": False,
            "primitive_success": True,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "saved_robot_state_checkpoint",
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": str(self._state_checkpoint_path),
            "state_checkpoint_path": str(self._state_checkpoint_path),
            "sha256": _sha256_file(self._state_checkpoint_path),
            "held_hand": held_hand,
            "press_hand": press_hand,
            "object_name": payload["object_name"],
            "total_env_steps": int(self._env_steps),
            "visual_evidence": visual_paths,
        }

    def _checkpoint_visual_evidence(
        self, *, checkpoint_name: str, held_hand: str, press_hand: str
    ) -> dict[str, str]:
        review_dir = self._output_dir / "visual_review" / checkpoint_name
        visual_paths: dict[str, str] = {}
        for label, camera in {
            "head": "head",
            "held_wrist": f"{held_hand}_wrist",
            "press_wrist": f"{press_hand}_wrist",
        }.items():
            observed = self._require_planner().observe(camera)
            image = observed.get("_image_bytes")
            if not isinstance(image, bytes):
                raise RuntimeError(f"checkpoint visual evidence missing {camera} PNG")
            image_path = review_dir / f"{label}.png"
            _write_bytes_atomic(image_path, image)
            visual_paths[label] = str(image_path)
        return visual_paths

    def restore_robot_state_checkpoint(
        self,
        *,
        checkpoint_name: str,
        checkpoint_path: str | None,
        mode: str,
        keep_held_gripper_closed: bool,
        require_object_still_held: bool,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Restore robot q-state through certified q-space controller motion."""

        if self._base_controller_mode != "position":
            raise RuntimeError("robot checkpoint restore requires position controllers")
        if mode != "plan_and_execute":
            raise ValueError("restore mode must be 'plan_and_execute'")
        if (
            keep_held_gripper_closed is not True
            or require_object_still_held is not True
        ):
            raise ValueError(
                "restore requires held-gripper closure and held-object validation"
            )
        if not isinstance(checkpoint_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", checkpoint_name
        ):
            raise ValueError("checkpoint_name is not a safe artifact stem")
        if getattr(
            self, "_control_mode", None
        ) == PI0_NAV_PICK_VLA_MODE and checkpoint_name not in {
            "state_checkpoint_1",
            "state_checkpoint_2",
        }:
            raise ValueError(
                "post-pick restore only accepts state_checkpoint_1 or "
                "state_checkpoint_2"
            )
        timeout = float(timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        deadline = time.monotonic() + timeout
        checkpoint_root = (self._output_dir / "state_checkpoints").resolve()
        expected_candidate = checkpoint_root / f"{checkpoint_name}.json"
        supplied_candidate = (
            expected_candidate
            if checkpoint_path is None
            else Path(checkpoint_path).expanduser()
        )
        if expected_candidate.is_symlink() or supplied_candidate.is_symlink():
            raise ValueError("checkpoint_path may not be a symbolic link")
        expected_path = expected_candidate.resolve()
        path = supplied_candidate.resolve()
        if path != expected_path or path.parent != checkpoint_root:
            raise ValueError(
                "checkpoint_path must resolve to this run's named checkpoint"
            )
        if not path.is_file():
            raise RuntimeError(f"robot checkpoint is missing: {path}")
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("kind") != "robot_motion_checkpoint"
            or checkpoint.get("not_simulator_restore") is not True
        ):
            raise RuntimeError("robot checkpoint schema is invalid")
        if checkpoint.get("checkpoint_name") != checkpoint_name:
            raise RuntimeError("checkpoint_name does not match checkpoint JSON")
        held_hand, press_hand = self._validated_handoff_hands(
            held_hand=str(checkpoint.get("held_hand")),
            press_hand=str(checkpoint.get("press_hand")),
        )
        robot_state = checkpoint.get("robot")
        poses = checkpoint.get("poses")
        if not isinstance(robot_state, dict) or not isinstance(poses, dict):
            raise RuntimeError("robot checkpoint payload is incomplete")
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        current_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        q_target = robot_state.get("q_space_target")
        if not isinstance(q_target, dict):
            raise RuntimeError("checkpoint q_space_target is missing")
        target_indices = np.asarray(q_target.get("indices"), dtype=int).reshape(-1)
        target_values = np.asarray(q_target.get("values"), dtype=np.float64).reshape(-1)
        target_q = current_q.copy()
        if (
            len(target_indices) != len(target_values)
            or len(target_indices) != 21
            or len(set(target_indices.tolist())) != len(target_indices)
            or np.any(target_indices < 0)
            or np.any(target_indices >= len(current_q))
        ):
            raise RuntimeError("checkpoint controlled q-space layout is invalid")
        expected_controlled: list[int] = []
        for values in (
            getattr(robot, "base_control_idx", []),
            getattr(robot, "trunk_control_idx", []),
            (getattr(robot, "arm_control_idx", {}) or {}).get("left", []),
            (getattr(robot, "arm_control_idx", {}) or {}).get("right", []),
        ):
            expected_controlled.extend(
                int(value)
                for value in np.asarray(_numpy_tree(values), dtype=int).reshape(-1)
            )
        if expected_controlled and expected_controlled != target_indices.tolist():
            raise RuntimeError("checkpoint q-space indices do not match loaded R1Pro")
        target_q[target_indices] = target_values
        yaw_index = int(target_indices[2])
        yaw_delta = float(
            np.arctan2(
                np.sin(target_values[2] - current_q[yaw_index]),
                np.cos(target_values[2] - current_q[yaw_index]),
            )
        )
        target_q[yaw_index] = current_q[yaw_index] + yaw_delta
        if (
            target_q.shape != current_q.shape
            or not np.isfinite(target_q).all()
            or not np.isfinite(current_q).all()
        ):
            raise RuntimeError("checkpoint q-state does not match the loaded R1Pro")
        steps = max(
            2,
            int(np.ceil(float(np.max(np.abs(target_q - current_q))) / 0.01)) + 1,
        )
        if steps > 600:
            raise RuntimeError("checkpoint q-space path exceeds the bounded horizon")
        q_path = np.linspace(current_q, target_q, steps, dtype=np.float64)
        planner = self._require_planner()
        backend = planner.backend
        generator_getter = getattr(backend, "_generator", None)
        collision_checker = getattr(backend, "_check_q_trajectory_collisions", None)
        if not callable(generator_getter) or not callable(collision_checker):
            raise RuntimeError("cuRobo q-space certification API is unavailable")
        generator = generator_getter(kind="arm", hand=held_hand)
        radio, _table = self._resolve_handoff_targets()
        attachment_getter = getattr(backend, "get_attached_object", None)
        attached_obj = (
            attachment_getter(held_hand) if callable(attachment_getter) else radio
        )
        if callable(attachment_getter):
            root_link = getattr(radio, "root_link", None)
            if root_link is None:
                raise RuntimeError(
                    "radio root link is unavailable for attached planning"
                )
            target_root = str(getattr(root_link, "prim_path", "")).rstrip("/")
            attached_matches = bool(
                isinstance(attached_obj, dict)
                and attached_obj
                and any(
                    value is root_link
                    or (
                        target_root
                        and str(getattr(value, "prim_path", "")).rstrip("/")
                        == target_root
                    )
                    for value in attached_obj.values()
                )
            )
            if not attached_matches:
                attached_obj = {EEF_LINK_BY_HAND[held_hand]: root_link}
        collision = collision_checker(generator, q_path, attached_obj=attached_obj)
        trace_path = self._output_dir / "state_checkpoints" / "restore_trace.json"
        restore_video_path: Path | None = None

        def restore_result(
            *,
            primitive_success: bool,
            stop_reason: str,
            trace: list[dict[str, Any]],
            metrics: dict[str, Any],
        ) -> dict[str, Any]:
            _write_json_atomic(trace_path, trace)
            if restore_video_path is not None:
                self._finalize_video_segment()
                self._video_sealed = True
            public_metrics = {
                key: metrics.get(key)
                for key in (
                    "joint_error_max_rad",
                    "base_error_m",
                    "base_yaw_error_rad",
                    "held_object_drift_m",
                    "held_gripper_opening_m",
                )
            }
            return {
                "_finish": False,
                "primitive_success": bool(primitive_success),
                "task_success": bool(_raw_success(self._last_info)),
                "official_success_source": 'info["done"]["success"]',
                "stop_reason": stop_reason,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": str(path),
                "held_hand": held_hand,
                "press_hand": press_hand,
                "object_name": checkpoint.get("object_name"),
                "executed_steps": len(trace),
                "total_env_steps": int(self._env_steps),
                "restore_trace_path": str(trace_path),
                "video_path": (
                    str(restore_video_path) if restore_video_path is not None else None
                ),
                "metrics": _wire_safe(public_metrics),
                "cuRobo_q_space_collision_report": _wire_safe(collision),
            }

        if not isinstance(collision, dict) or not bool(
            collision.get("available", False)
        ):
            return restore_result(
                primitive_success=False,
                stop_reason="curobo_collision_check_unavailable",
                trace=[],
                metrics={},
            )
        if time.monotonic() >= deadline:
            return restore_result(
                primitive_success=False,
                stop_reason="timeout",
                trace=[],
                metrics={},
            )
        if bool(collision.get("colliding", True)):
            return restore_result(
                primitive_success=False,
                stop_reason="curobo_path_in_collision",
                trace=[],
                metrics={},
            )

        eef_links = getattr(robot, "eef_links", {})
        held_link = eef_links.get(held_hand) if isinstance(eef_links, dict) else None
        if held_link is None:
            raise RuntimeError(f"{held_hand} EEF link is unavailable")
        start_eef_position, start_eef_orientation = self._object_pose(held_link)
        start_radio_position, start_radio_orientation = self._object_pose(radio)
        current_relative_position, current_relative_orientation = self._relative_pose(
            start_eef_position,
            start_eef_orientation,
            start_radio_position,
            start_radio_orientation,
        )
        object_world = poses.get("object_pose_world")
        expected_relative = poses.get("object_pose_in_held_eef")
        if not isinstance(object_world, dict):
            raise RuntimeError("checkpoint object pose is missing")
        expected_relative_position = (
            np.asarray(expected_relative.get("position"), dtype=np.float64).reshape(3)
            if isinstance(expected_relative, dict)
            else current_relative_position.copy()
        )
        expected_relative_orientation = (
            np.asarray(expected_relative.get("quat_xyzw"), dtype=np.float64).reshape(4)
            if isinstance(expected_relative, dict)
            else current_relative_orientation.copy()
        )
        if not np.isfinite(expected_relative_position).all():
            raise RuntimeError("checkpoint relative object pose is invalid")
        preflight_relative_drift = float(
            np.linalg.norm(current_relative_position - expected_relative_position)
        )
        preflight_angular_drift = _quaternion_angle_rad(
            current_relative_orientation, expected_relative_orientation
        )
        checkpoint_height = float(object_world.get("position", [0, 0, np.nan])[2])
        if not np.isfinite(checkpoint_height):
            raise RuntimeError("checkpoint target height is invalid")
        if self._last_observation is None:
            raise RuntimeError("restore preflight has no synchronized observation")
        preflight_opening = self._validated_selected_gripper_opening(
            observation=self._last_observation, hand=held_hand
        )
        preflight_held_attachment = self._attachment_matches(held_hand, radio)
        preflight_held_contact = self._hand_target_contact_report(
            held_hand, start_radio_position
        )
        preflight_press_attachment = self._attachment_matches(press_hand, radio)
        preflight_press_contact = self._hand_target_contact_report(
            press_hand, start_radio_position
        )
        if (
            preflight_relative_drift > HANDOFF_RELATIVE_DRIFT_MAX_M
            or preflight_angular_drift > HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD
        ):
            return restore_result(
                primitive_success=False,
                stop_reason="held_object_drift",
                trace=[],
                metrics={"held_object_drift_m": preflight_relative_drift},
            )
        if preflight_opening >= HANDOFF_GRIPPER_OPENING_MAX:
            return restore_result(
                primitive_success=False,
                stop_reason="held_object_lost",
                trace=[],
                metrics={"held_gripper_opening_m": preflight_opening},
            )
        if isinstance(expected_relative, dict) and not (
            preflight_held_attachment[0]
            or preflight_held_attachment[1]
            or preflight_held_contact["target_contact_count"] > 0
        ):
            return restore_result(
                primitive_success=False,
                stop_reason="held_object_lost",
                trace=[],
                metrics={"held_gripper_opening_m": preflight_opening},
            )
        if isinstance(expected_relative, dict) and (
            preflight_press_attachment[0]
            or preflight_press_attachment[1]
            or preflight_press_contact["target_contact_count"] != 0
        ):
            return restore_result(
                primitive_success=False,
                stop_reason="press_hand_contacted_object",
                trace=[],
                metrics={"held_gripper_opening_m": preflight_opening},
            )
        self._gripper_latch[held_hand] = -1.0
        if hasattr(self, "_video_path"):
            restore_video_path = (
                self._output_dir
                / f"curobo_checkpoint_restore_{checkpoint_name}_episode.mp4"
            )
            self.start_video_segment(restore_video_path)
        trace: list[dict[str, Any]] = []
        official_success = False
        stop_reason: str | None = None
        for index, waypoint in enumerate(q_path[1:], start=1):
            if time.monotonic() >= deadline:
                stop_reason = "timeout"
                break
            restore_horizon = int(
                getattr(self, "_meta", {}).get("max_episode_steps", 2**63 - 1)
            )
            if self._env_steps >= restore_horizon:
                stop_reason = "horizon"
                break
            action = np.asarray(
                backend.joint_target_to_action(waypoint, hand=None), dtype=np.float32
            ).reshape(23)
            action[ENV_ACTION_SEGMENTS[f"{held_hand}_gripper"]] = -1.0
            action[ENV_ACTION_SEGMENTS[f"{press_hand}_gripper"]] = float(
                self._gripper_latch[press_hand]
            )
            step_obs, _reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    __import__("torch").as_tensor(action).reshape(1, 23),
                    need_obs=True,
                )
            )
            self._env_steps += 1
            step_info = _numpy_tree(step_infos[0])
            self._last_info = step_info
            official_success = official_success or _raw_success(step_info)
            observation = _single_observation(self._env._wrap_obs(step_obs))
            self._last_observation = observation
            self._record_rgbd_frames(step_obs, observation)
            self._append_video(observation)
            eef_position, eef_orientation = self._object_pose(held_link)
            radio_position, radio_orientation = self._object_pose(radio)
            relative_position, relative_orientation = self._relative_pose(
                eef_position, eef_orientation, radio_position, radio_orientation
            )
            relative_drift = float(
                np.linalg.norm(relative_position - expected_relative_position)
            )
            angular_drift = _quaternion_angle_rad(
                relative_orientation, expected_relative_orientation
            )
            assisted, backend_attached = self._attachment_matches(press_hand, radio)
            contact = self._hand_target_contact_report(press_hand, radio_position)
            held_assisted, held_backend_attached = self._attachment_matches(
                held_hand, radio
            )
            held_contact = self._hand_target_contact_report(held_hand, radio_position)
            held_opening = self._validated_selected_gripper_opening(
                observation=observation, hand=held_hand
            )
            sample = {
                "waypoint": index,
                "total_env_steps": int(self._env_steps),
                "relative_drift_m": relative_drift,
                "held_object_angular_drift_rad": angular_drift,
                "radio_height_m": float(radio_position[2]),
                "press_hand_assisted_attachment": assisted,
                "press_hand_backend_attachment": backend_attached,
                "press_hand_contact_count": contact["target_contact_count"],
                "held_hand_assisted_attachment": held_assisted,
                "held_hand_backend_attachment": held_backend_attached,
                "held_hand_contact_count": held_contact["target_contact_count"],
                "held_gripper_opening_m": float(held_opening),
                "official_task_success": bool(_raw_success(step_info)),
            }
            trace.append(sample)
            if (
                relative_drift > HANDOFF_RELATIVE_DRIFT_MAX_M
                or angular_drift > HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD
            ):
                stop_reason = "held_object_drift"
                break
            if float(radio_position[2]) < checkpoint_height - 0.02:
                stop_reason = "held_object_lost"
                break
            if held_opening >= HANDOFF_GRIPPER_OPENING_MAX:
                stop_reason = "held_gripper_opened"
                break
            if not (
                held_assisted
                or held_backend_attached
                or held_contact["target_contact_count"] > 0
            ):
                stop_reason = "held_object_lost"
                break
            if assisted or backend_attached or contact["target_contact_count"] != 0:
                stop_reason = "press_hand_contacted_object"
                break
            if official_success:
                break
            if _scalar_bool(step_term) or _scalar_bool(step_trunc):
                stop_reason = "episode_stopped"
                break
        final_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        base_error = float(
            np.linalg.norm(final_q[target_indices[:2]] - target_values[:2])
        )
        base_yaw_error = float(
            abs(
                np.arctan2(
                    np.sin(final_q[target_indices[2]] - target_values[2]),
                    np.cos(final_q[target_indices[2]] - target_values[2]),
                )
            )
        )
        joint_error = float(
            np.max(np.abs(final_q[target_indices[3:]] - target_values[3:]))
        )
        metrics = {
            "joint_error_max_rad": joint_error,
            "base_error_m": base_error,
            "base_yaw_error_rad": base_yaw_error,
            "held_object_drift_m": (
                float(trace[-1]["relative_drift_m"]) if trace else None
            ),
            "held_gripper_opening_m": (
                float(trace[-1]["held_gripper_opening_m"]) if trace else None
            ),
        }
        success = bool(
            stop_reason is None
            and joint_error <= 0.02
            and base_error <= 0.02
            and base_yaw_error <= 0.03
        )
        if stop_reason is None and not success:
            stop_reason = "joint_target_not_reached"
        if success and checkpoint_name == "state_checkpoint_1":
            self._prepress_coarse_flip_used = False
        return restore_result(
            primitive_success=success,
            stop_reason=(
                "restored_robot_state_checkpoint" if success else str(stop_reason)
            ),
            trace=trace,
            metrics=metrics,
        )

    def _reload_base_controller_position(self) -> dict[str, Any]:
        if self._base_controller_mode != "velocity":
            raise RuntimeError(
                "base controller handoff did not start from velocity mode"
            )
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        reload_controllers = getattr(robot, "reload_controllers", None)
        original_config = getattr(robot, "_controller_config", None)
        if not callable(reload_controllers) or original_config is None:
            raise RuntimeError("R1Pro controller reload API is unavailable")
        config = deepcopy(original_config)
        base = (
            config.get("base")
            if isinstance(config, dict)
            else getattr(config, "base", None)
        )
        if base is None:
            raise RuntimeError("R1Pro base controller config is unavailable")

        def assign(name: str, value: Any) -> None:
            if isinstance(base, dict):
                base[name] = value
            else:
                setattr(base, name, value)

        assign("motor_type", "position")
        assign("command_input_limits", None)
        assign("command_output_limits", None)
        assign("use_impedances", False)
        assign("isaac_kp", 2_000_000.0)
        assign("isaac_kd", 100_000.0)
        reload_controllers(config)
        expected_layout = (
            ("base", 3),
            ("trunk", 4),
            ("arm_left", 7),
            ("gripper_left", 1),
            ("arm_right", 7),
            ("gripper_right", 1),
        )
        actual_layout = tuple(
            (str(name), int(controller.command_dim))
            for name, controller in getattr(robot, "controllers", {}).items()
        )
        if actual_layout != expected_layout:
            raise RuntimeError(
                f"controller reload changed the 23D layout: {actual_layout!r}"
            )
        base_controller = getattr(robot, "controllers", {}).get("base")
        motor_type = getattr(base_controller, "motor_type", None)
        if motor_type is not None and not str(motor_type).lower().endswith("position"):
            raise RuntimeError(
                f"base controller reload did not enter position mode: {motor_type!r}"
            )
        self._base_controller_mode = "position"
        return {
            "from": "velocity",
            "to": "position",
            "controller_layout": [list(item) for item in actual_layout],
        }

    @staticmethod
    def _base_controller_signature(robot: Any) -> dict[str, Any]:
        controllers = getattr(robot, "controllers", {})
        controller = controllers.get("base") if isinstance(controllers, dict) else None
        config = getattr(robot, "_controller_config", None)
        base_config = (
            config.get("base")
            if isinstance(config, dict)
            else getattr(config, "base", None)
        )

        def configured(name: str) -> Any:
            if isinstance(base_config, dict):
                return base_config.get(name)
            return getattr(base_config, name, None)

        dof_idx = getattr(controller, "dof_idx", None)
        if dof_idx is None:
            dof_idx = getattr(controller, "control_idx", None)
        return {
            "class": (
                f"{type(controller).__module__}.{type(controller).__qualname__}"
                if controller is not None
                else None
            ),
            "command_dim": int(getattr(controller, "command_dim", -1)),
            "motor_type": str(getattr(controller, "motor_type", "")),
            "dof_idx": np.asarray(_numpy_tree(dof_idx), dtype=int).reshape(-1).tolist()
            if dof_idx is not None
            else None,
            "config": {
                "motor_type": str(configured("motor_type")),
                "command_input_limits": _wire_safe(configured("command_input_limits")),
                "command_output_limits": _wire_safe(
                    configured("command_output_limits")
                ),
                "use_impedances": _wire_safe(configured("use_impedances")),
                "isaac_kp": _wire_safe(configured("isaac_kp")),
                "isaac_kd": _wire_safe(configured("isaac_kd")),
            },
        }

    def run_controller_switch_smoke(self) -> dict[str, Any]:
        """Exercise the real velocity-to-position handoff without invoking Pi0."""

        import torch

        if self._control_mode != PI0_NAV_PICK_VLA_MODE:
            raise RuntimeError("controller switch smoke requires pi0_nav_pick_vla mode")
        if self._reset_completed:
            raise RuntimeError("controller switch smoke must own the initial reset")
        self.reset()
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        self.start_video_segment(
            self._output_dir / "controller_switch_smoke_episode.mp4"
        )
        pre_reload_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if not np.isfinite(pre_reload_q).all():
            raise RuntimeError("controller smoke initial q-state is invalid")
        reload_report = self._reload_base_controller_position()
        post_reload_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if (
            post_reload_q.shape != pre_reload_q.shape
            or not np.isfinite(post_reload_q).all()
        ):
            raise RuntimeError("controller smoke post-reload q-state is invalid")

        base_indices = np.asarray(
            _numpy_tree(getattr(robot, "base_control_idx", [])), dtype=int
        )
        trunk_indices = np.asarray(
            _numpy_tree(getattr(robot, "trunk_control_idx", [])), dtype=int
        )
        arm_indices = getattr(robot, "arm_control_idx", {}) or {}
        left_indices = np.asarray(_numpy_tree(arm_indices.get("left", [])), dtype=int)
        right_indices = np.asarray(_numpy_tree(arm_indices.get("right", [])), dtype=int)
        if not (
            base_indices.shape == (3,)
            and trunk_indices.shape == (4,)
            and left_indices.shape == (7,)
            and right_indices.shape == (7,)
        ):
            raise RuntimeError("controller smoke R1Pro joint layout is invalid")
        articulation_indices = np.r_[trunk_indices, left_indices, right_indices]

        hold = np.asarray(
            self._require_planner().backend.hold_action(), dtype=np.float32
        ).reshape(-1)
        if hold.shape != (23,) or not np.isfinite(hold).all():
            raise RuntimeError("controller smoke hold must be one finite 23D action")
        stable_frames: list[dict[str, Any]] = []
        for hold_index in range(HANDOFF_VALIDATION_FRAMES):
            if self._env_steps >= int(self._meta["max_episode_steps"]):
                raise RuntimeError("controller smoke exhausted the episode horizon")
            step_obs, _reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    torch.as_tensor(hold, dtype=torch.float32).reshape(1, 23),
                    need_obs=True,
                )
            )
            self._env_steps += 1
            self._last_info = _numpy_tree(step_infos[0])
            if _scalar_bool(step_term) or _scalar_bool(step_trunc):
                raise RuntimeError(
                    "controller smoke episode stopped during stable hold"
                )
            observation = _single_observation(self._env._wrap_obs(step_obs))
            self._last_observation = observation
            self._record_rgbd_frames(step_obs, observation)
            self._append_video(observation)
            current_q = np.asarray(
                _numpy_tree(robot.get_joint_positions()), dtype=np.float64
            ).reshape(-1)
            base_error = float(
                np.linalg.norm(
                    current_q[base_indices[:2]] - pre_reload_q[base_indices[:2]]
                )
            )
            yaw_error = float(
                abs(
                    np.arctan2(
                        np.sin(
                            current_q[base_indices[2]] - pre_reload_q[base_indices[2]]
                        ),
                        np.cos(
                            current_q[base_indices[2]] - pre_reload_q[base_indices[2]]
                        ),
                    )
                )
            )
            articulation_error = float(
                np.max(
                    np.abs(
                        current_q[articulation_indices]
                        - pre_reload_q[articulation_indices]
                    )
                )
            )
            if base_error > 0.02 or yaw_error > 0.03 or articulation_error > 0.02:
                raise RuntimeError("controller smoke current-target hold drifted")
            stable_frames.append(
                {
                    "hold_index": hold_index + 1,
                    "env_step": int(self._env_steps),
                    "base_error_m": base_error,
                    "base_yaw_error_rad": yaw_error,
                    "joint_error_max_rad": articulation_error,
                }
            )

        pre_warmup_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        self._require_planner().on_simulator_state_restored()
        warmup = self._require_planner().warmup()
        post_warmup_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        warmup_pose_jump = float(np.max(np.abs(post_warmup_q - pre_warmup_q)))
        if warmup_pose_jump > 1e-6:
            raise RuntimeError("cuRobo smoke warmup moved the robot")
        self._finalize_video_segment()
        self._video_sealed = True
        result = {
            "schema_version": 1,
            "success": True,
            "control_mode": PI0_NAV_PICK_VLA_MODE,
            "controller_reload": reload_report,
            "first_command": "current_target_hold",
            "stable_hold_frames": stable_frames,
            "stable_hold_frames_required": HANDOFF_VALIDATION_FRAMES,
            "planner_warmup": _wire_safe(warmup),
            "warmup_pose_jump_max": warmup_pose_jump,
            "total_env_steps": int(self._env_steps),
            "video_path": str(self._video_path),
        }
        result_path = self._output_dir / "controller_switch_smoke.json"
        result["result_path"] = str(result_path)
        _write_json_atomic(result_path, result)
        return result

    def _persist_pi0_nav_pick_views(
        self, *, chunk_index: int, validator: dict[str, Any]
    ) -> dict[str, Any]:
        root = (
            self._output_dir
            / "visual_review"
            / "pi0_nav_pick"
            / f"chunk_{chunk_index:04d}"
        )
        views: dict[str, Any] = {}
        capture_group_id: str | None = None
        for camera in ("head", "left_wrist", "right_wrist"):
            payload = self._require_planner().observe(camera)
            image = payload.get("_image_bytes")
            if not isinstance(image, bytes):
                raise RuntimeError(f"pi0_nav_pick {camera} review PNG is unavailable")
            group = payload.get("capture_group")
            group_id = str(group.get("id")) if isinstance(group, dict) else None
            if capture_group_id is None:
                capture_group_id = group_id
            elif capture_group_id != group_id:
                raise RuntimeError("pi0_nav_pick review views are not synchronized")
            image_path = root / f"{camera}.png"
            _write_bytes_atomic(image_path, image)
            views[camera] = {
                "path": str(image_path),
                "frame_id": payload.get("frame_id"),
            }
        metadata = {
            "chunk_index": int(chunk_index),
            "env_step": int(self._env_steps),
            "capture_group_id": capture_group_id,
            "validator_summary": _wire_safe(
                {
                    key: validator.get(key)
                    for key in (
                        "frames_collected",
                        "passed_hands",
                        "held_hand",
                        "local_grasp_success",
                        "per_hand",
                    )
                }
            ),
            "views": views,
        }
        metadata_path = root / "metadata.json"
        metadata["metadata_path"] = str(metadata_path)
        _write_json_atomic(metadata_path, metadata)
        return metadata

    def _complete_pi0_nav_pick_handoff(
        self, *, held_hand: str, validator: dict[str, Any]
    ) -> dict[str, Any]:
        import torch

        held_hand, press_hand = self._validated_handoff_hands(
            held_hand=held_hand,
            press_hand="right" if held_hand == "left" else "left",
        )
        self._state_checkpoint_path = (
            self._output_dir / "state_checkpoints" / "state_checkpoint_1.json"
        )
        generated = (
            self._state_checkpoint_path,
            self._paused_runtime_path,
        )
        state_pending = self._state_checkpoint_path.with_suffix(".json.pending")
        handoff_env_steps = 0
        stable_frames: list[dict[str, Any]] = []
        post_pick_warnings: list[dict[str, Any]] = []
        latest_metrics: dict[str, Any] | None = None
        reload_report: dict[str, Any] | None = None
        diagnostics_path = self._output_dir / "handoff_failure_diagnostics.json"
        for path in generated:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
        state_pending.unlink(missing_ok=True)
        try:
            remaining_horizon = max(
                0, int(self._meta["max_episode_steps"]) - int(self._env_steps)
            )
            validation_frame_budget = min(HANDOFF_VALIDATION_FRAMES, remaining_horizon)
            if validation_frame_budget < HANDOFF_VALIDATION_FRAMES:
                post_pick_warnings.append(
                    _post_pick_warning(
                        "insufficient_handoff_horizon",
                        "fewer than eight post-reload diagnostic hold frames remain",
                        metrics={
                            "available_frames": validation_frame_budget,
                            "requested_frames": HANDOFF_VALIDATION_FRAMES,
                        },
                    )
                )
            self._handoff_state = _HANDOFF_CHECKPOINTING
            self._gripper_latch[held_hand] = -1.0
            robot = self._robot()
            if robot is None:
                raise RuntimeError("R1Pro robot is unavailable")
            pre_reload_q = np.asarray(
                _numpy_tree(robot.get_joint_positions()), dtype=np.float64
            ).reshape(-1)
            if not np.isfinite(pre_reload_q).all():
                raise RuntimeError("pre-reload R1Pro q-state is invalid")
            state_payload = self._robot_state_checkpoint_payload(
                checkpoint_name="state_checkpoint_1",
                stage="post_pick_pre_controller_reload",
                held_hand=held_hand,
                press_hand=press_hand,
                object_name="radio",
                require_current_grasp=True,
                validation_evidence={"strict_vla_window": validator},
            )
            try:
                state_payload["visual_evidence"] = self._checkpoint_visual_evidence(
                    checkpoint_name="state_checkpoint_1",
                    held_hand=held_hand,
                    press_hand=press_hand,
                )
            except Exception as exc:
                post_pick_warnings.append(
                    _post_pick_warning(
                        "visual_evidence_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                state_payload["visual_evidence"] = {}
            state_payload["warnings"] = _wire_safe(post_pick_warnings)
            _write_json_atomic(state_pending, state_payload)
            self._finalize_video_segment()
            self._video_sealed = True
            self.start_video_segment(
                self._output_dir / "curobo_checkpoint_restore_episode.mp4"
            )
            self._handoff_state = _HANDOFF_CONTROLLER_RELOAD
            reload_report = self._reload_base_controller_position()
            post_reload_q = np.asarray(
                _numpy_tree(robot.get_joint_positions()), dtype=np.float64
            ).reshape(-1)
            if (
                post_reload_q.shape != pre_reload_q.shape
                or not np.isfinite(post_reload_q).all()
            ):
                raise RuntimeError("post-reload R1Pro q-state is invalid")
            base_indices = np.asarray(
                _numpy_tree(getattr(robot, "base_control_idx", [])), dtype=int
            )
            trunk_indices = np.asarray(
                _numpy_tree(getattr(robot, "trunk_control_idx", [])), dtype=int
            )
            arm_indices = getattr(robot, "arm_control_idx", {}) or {}
            left_indices = np.asarray(
                _numpy_tree(arm_indices.get("left", [])), dtype=int
            )
            right_indices = np.asarray(
                _numpy_tree(arm_indices.get("right", [])), dtype=int
            )
            if not (
                base_indices.shape == (3,)
                and trunk_indices.shape == (4,)
                and left_indices.shape == (7,)
                and right_indices.shape == (7,)
            ):
                raise RuntimeError("post-reload R1Pro joint index layout is invalid")
            base_xy_jump = float(
                np.max(
                    np.abs(
                        post_reload_q[base_indices[:2]] - pre_reload_q[base_indices[:2]]
                    )
                )
            )
            base_yaw_jump = float(
                abs(post_reload_q[base_indices[2]] - pre_reload_q[base_indices[2]])
            )
            articulation_indices = np.r_[trunk_indices, left_indices, right_indices]
            articulation_jump = float(
                np.max(
                    np.abs(
                        post_reload_q[articulation_indices]
                        - pre_reload_q[articulation_indices]
                    )
                )
            )
            if base_xy_jump > 0.02 or base_yaw_jump > 0.03 or articulation_jump > 0.02:
                post_pick_warnings.append(
                    _post_pick_warning(
                        "controller_reload_pose_jump",
                        "controller reload exceeded strict pose-jump diagnostics",
                        metrics={
                            "base_xy_jump_m": base_xy_jump,
                            "base_yaw_jump_rad": base_yaw_jump,
                            "articulation_jump_rad": articulation_jump,
                            "limits": {
                                "base_xy_jump_m": 0.02,
                                "base_yaw_jump_rad": 0.03,
                                "articulation_jump_rad": 0.02,
                            },
                        },
                    )
                )
            hold = np.asarray(
                self._require_planner().backend.hold_action(), dtype=np.float32
            ).reshape(-1)
            if hold.shape != (23,) or not np.isfinite(hold).all():
                raise RuntimeError(
                    "first post-reload hold is not one finite 23D action"
                )
            hold[ENV_ACTION_SEGMENTS[f"{held_hand}_gripper"]] = -1.0
            hold[ENV_ACTION_SEGMENTS[f"{press_hand}_gripper"]] = float(
                self._gripper_latch[press_hand]
            )
            self._handoff_state = _HANDOFF_STABLE_VALIDATION
            radio, _table = self._resolve_handoff_targets()
            eef_links = getattr(robot, "eef_links", {})
            held_link = (
                eef_links.get(held_hand) if isinstance(eef_links, dict) else None
            )
            if held_link is None:
                raise RuntimeError(f"{held_hand} EEF link is unavailable")
            eef_position, eef_orientation = self._object_pose(held_link)
            radio_position, radio_orientation = self._object_pose(radio)
            initial_relative_position, initial_relative_orientation = (
                self._relative_pose(
                    eef_position, eef_orientation, radio_position, radio_orientation
                )
            )
            for hold_index in range(validation_frame_budget):
                step_obs, _reward, step_term, step_trunc, step_infos = (
                    self._env._direct_process.step_env(
                        torch.as_tensor(hold, dtype=torch.float32).reshape(1, 23),
                        need_obs=True,
                    )
                )
                self._env_steps += 1
                handoff_env_steps += 1
                self._last_info = _numpy_tree(step_infos[0])
                observation = _single_observation(self._env._wrap_obs(step_obs))
                self._last_observation = observation
                self._record_rgbd_frames(step_obs, observation)
                frame = self._handoff_validator_frame(observation)
                selected = frame["per_hand"][held_hand]
                other = frame["per_hand"][press_hand]
                current_q = np.asarray(
                    _numpy_tree(robot.get_joint_positions()), dtype=np.float64
                ).reshape(-1)
                eef_position, eef_orientation = self._object_pose(held_link)
                radio_position, radio_orientation = self._object_pose(radio)
                relative_position, relative_orientation = self._relative_pose(
                    eef_position, eef_orientation, radio_position, radio_orientation
                )
                relative_drift = float(
                    np.linalg.norm(relative_position - initial_relative_position)
                )
                angular_drift = _quaternion_angle_rad(
                    relative_orientation, initial_relative_orientation
                )
                latest_metrics = {
                    "hold_index": hold_index + 1,
                    "env_step": int(self._env_steps),
                    "base_xy_error_m": float(
                        np.max(
                            np.abs(
                                current_q[base_indices[:2]]
                                - pre_reload_q[base_indices[:2]]
                            )
                        )
                    ),
                    "base_yaw_error_rad": float(
                        abs(current_q[base_indices[2]] - pre_reload_q[base_indices[2]])
                    ),
                    "articulation_error_rad": float(
                        np.max(
                            np.abs(
                                current_q[articulation_indices]
                                - pre_reload_q[articulation_indices]
                            )
                        )
                    ),
                    "object_relative_drift_m": relative_drift,
                    "held_object_angular_drift_rad": angular_drift,
                    "limits": {
                        "base_xy_error_m": 0.02,
                        "base_yaw_error_rad": 0.03,
                        "articulation_error_rad": HANDOFF_ARTICULATION_ERROR_MAX_RAD,
                        "object_relative_drift_m": HANDOFF_RELATIVE_DRIFT_MAX_M,
                        "held_object_angular_drift_rad": HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD,
                    },
                    "validator": frame,
                    "episode_status": {
                        "terminated": bool(_scalar_bool(step_term)),
                        "truncated": bool(_scalar_bool(step_trunc)),
                        "done": bool(_raw_done(self._last_info)),
                        "official_task_success": bool(_raw_success(self._last_info)),
                    },
                }
                latest_metrics["controller_hold_warning"] = (
                    _handoff_controller_hold_warning(latest_metrics)
                )
                if latest_metrics["controller_hold_warning"]:
                    post_pick_warnings.append(
                        _post_pick_warning(
                            "controller_hold_settling",
                            "post-reload controller settling exceeded diagnostic limits",
                            metrics=latest_metrics,
                        )
                    )
                if any(latest_metrics["episode_status"].values()):
                    post_pick_warnings.append(
                        _post_pick_warning(
                            "episode_status_during_handoff",
                            "episode stopped before eight-frame diagnostic hold completed",
                            metrics=latest_metrics["episode_status"],
                        )
                    )
                    stable_frames.append(latest_metrics)
                    self._append_video(observation)
                    break
                if not _post_reload_grasp_stable(
                    selected=selected,
                    other=other,
                ):
                    post_pick_warnings.append(
                        _post_pick_warning(
                            "post_reload_grasp_not_strict",
                            "post-reload strict grasp diagnostics did not pass",
                            metrics=frame,
                        )
                    )
                if not _handoff_held_object_stable(
                    relative_drift_m=relative_drift,
                    angular_drift_rad=angular_drift,
                ):
                    post_pick_warnings.append(
                        _post_pick_warning(
                            "held_object_stability_not_strict",
                            "held-radio drift exceeded strict diagnostic thresholds",
                            metrics={
                                "relative_drift_m": relative_drift,
                                "angular_drift_rad": angular_drift,
                            },
                        )
                    )
                stable_frames.append(latest_metrics)
                self._append_video(observation)
            try:
                warmup = self._prepare_prepress_planner_readiness(held_hand=held_hand)
            except Exception as exc:
                warmup = {"status": "warning", "error": f"{type(exc).__name__}: {exc}"}
                post_pick_warnings.append(
                    _post_pick_warning(
                        "prepress_planner_readiness_failed",
                        warmup["error"],
                    )
                )
            try:
                verification_views = self._checkpoint_visual_evidence(
                    checkpoint_name="restore_verification",
                    held_hand=held_hand,
                    press_hand=press_hand,
                )
            except Exception as exc:
                verification_views = {}
                post_pick_warnings.append(
                    _post_pick_warning(
                        "restore_verification_visual_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            verification_path = self._output_dir / "restore_verification.json"
            _write_json_atomic(
                verification_path,
                {
                    "controller_reload": reload_report,
                    "planner_warmup": _wire_safe(warmup),
                    "stable_hold_frames": stable_frames,
                    "visual_evidence": verification_views,
                    "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                    "strict_local_grasp_success": not bool(post_pick_warnings),
                    "usable_post_pick_saved": True,
                    "warnings": _wire_safe(post_pick_warnings),
                    "official_task_success": bool(_raw_success(self._last_info)),
                },
            )
            # The authoritative checkpoint is the actual stable post-reload
            # observation, not the pre-reload target.  This keeps the robot-
            # motion checkpoint aligned with the PAUSED state while the strict
            # pre/post metrics above still guard controller-switch continuity.
            final_state_payload = self._robot_state_checkpoint_payload(
                checkpoint_name="state_checkpoint_1",
                stage="post_pi0_nav_pick",
                held_hand=held_hand,
                press_hand=press_hand,
                object_name="radio",
                require_current_grasp=True,
                validation_evidence={
                    "strict_vla_window": validator,
                    "controller_reload": reload_report,
                    "stable_hold_frames": stable_frames,
                    "post_pick_warnings": post_pick_warnings,
                },
            )
            try:
                final_state_payload["visual_evidence"] = (
                    self._checkpoint_visual_evidence(
                        checkpoint_name="state_checkpoint_1",
                        held_hand=held_hand,
                        press_hand=press_hand,
                    )
                )
            except Exception as exc:
                post_pick_warnings.append(
                    _post_pick_warning(
                        "state1_visual_evidence_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                final_state_payload["visual_evidence"] = {}
            final_state_payload["strict_local_grasp_success"] = bool(
                final_state_payload.get("strict_local_grasp_success")
                and bool(validator.get("local_grasp_success"))
                and not any(
                    warning.get("code") in _STRICT_GRASP_WARNING_CODES
                    for warning in post_pick_warnings
                    if isinstance(warning, dict)
                )
            )
            final_state_payload["usable_post_pick_saved"] = True
            final_state_payload["save_policy"] = POST_PICK_DEBUG_SAVE_POLICY
            final_state_payload["warnings"] = _wire_safe(post_pick_warnings)
            final_state_payload["validation"].update(
                {
                    "strict_local_grasp_success": final_state_payload[
                        "strict_local_grasp_success"
                    ],
                    "usable_post_pick_saved": True,
                    "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                    "warnings": _wire_safe(post_pick_warnings),
                }
            )
            _write_json_atomic(state_pending, final_state_payload)
            os.replace(state_pending, self._state_checkpoint_path)
            self._held_hand = held_hand
            self._action_source = "curobo"
            self._finalize_video_segment()
            self._video_sealed = True
            self._handoff_state = _HANDOFF_PAUSED
            paused = {
                "schema_version": 1,
                "control_mode": PI0_NAV_PICK_VLA_MODE,
                "handoff_state": _HANDOFF_PAUSED,
                "env_pid": os.getpid(),
                "env_step": int(self._env_steps),
                "held_hand": held_hand,
                "press_hand": press_hand,
                "action_source": self._action_source,
                "vla_actions_enabled": True,
                "vla_action_gate_confirmed": False,
                "lifecycle_finalized": False,
                "state_checkpoint_path": str(self._state_checkpoint_path),
                "state_checkpoint_sha256": _sha256_file(self._state_checkpoint_path),
                "video_path": str(self._video_path),
                "pi0_nav_pick_video_path": str(
                    self._output_dir / "pi0_nav_pick_episode.mp4"
                ),
                "curobo_restore_video_path": str(
                    self._output_dir / "curobo_checkpoint_restore_episode.mp4"
                ),
                "restore_verification_path": str(verification_path),
                "strict_local_grasp_success": bool(
                    final_state_payload["strict_local_grasp_success"]
                ),
                "usable_post_pick_saved": True,
                "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                "warnings": _wire_safe(post_pick_warnings),
            }
            _write_json_atomic(self._paused_runtime_path, paused)
            return {
                "held_hand": held_hand,
                "press_hand": press_hand,
                "state_checkpoint_path": str(self._state_checkpoint_path),
                "paused_runtime_path": str(self._paused_runtime_path),
                "controller_reload": reload_report,
                "handoff_env_steps": handoff_env_steps,
                "strict_local_grasp_success": bool(
                    final_state_payload["strict_local_grasp_success"]
                ),
                "usable_post_pick_saved": True,
                "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                "warnings": _wire_safe(post_pick_warnings),
            }
        except Exception as exc:
            self._vla_actions_enabled = False
            post_pick_warnings.append(
                _post_pick_warning(
                    "post_pick_handoff_diagnostic_failed",
                    f"{type(exc).__name__}: {exc}",
                    metrics={
                        "controller_reload": reload_report,
                        "latest_metrics": latest_metrics,
                    },
                )
            )
            diagnostics = {
                "schema_version": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "held_hand": held_hand,
                "press_hand": press_hand,
                "env_step": int(self._env_steps),
                "handoff_env_steps": int(handoff_env_steps),
                "strict_vla_validator": _wire_safe(validator),
                "controller_reload": _wire_safe(reload_report),
                "stable_frames": _wire_safe(stable_frames),
                "latest_metrics": _wire_safe(latest_metrics),
                "warnings": _wire_safe(post_pick_warnings),
            }
            _write_json_atomic(diagnostics_path, diagnostics)
            self._last_handoff_failure = {
                **diagnostics,
                "handoff_failure_diagnostics_path": str(diagnostics_path),
            }
            if state_pending.is_file() and not self._state_checkpoint_path.exists():
                # The pre-controller snapshot is already a complete robot-motion
                # payload.  Operational handoff/readiness/validator failures do
                # not revoke its value as a development restart artifact.
                fallback = json.loads(state_pending.read_text(encoding="utf-8"))
                if not isinstance(fallback, dict):
                    raise RuntimeError("pending state_checkpoint_1 payload is invalid")
                fallback.update(
                    {
                        "stage": "post_pick_debug_fallback",
                        "strict_local_grasp_success": False,
                        "usable_post_pick_saved": True,
                        "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                        "warnings": _wire_safe(post_pick_warnings),
                        "snapshot_phase": "pre_controller_reload",
                    }
                )
                validation = fallback.get("validation")
                if not isinstance(validation, dict):
                    raise RuntimeError(
                        "pending state_checkpoint_1 validation schema is invalid"
                    )
                validation.update(
                    {
                        "strict_local_grasp_success": False,
                        "usable_post_pick_saved": True,
                        "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                        "warnings": _wire_safe(post_pick_warnings),
                    }
                )
                _write_json_atomic(state_pending, fallback)
                os.replace(state_pending, self._state_checkpoint_path)
                self._held_hand = held_hand
                self._action_source = (
                    "curobo" if reload_report is not None else "pi0_vla"
                )
                self._handoff_state = _HANDOFF_PAUSED
                paused = {
                    "schema_version": 1,
                    "control_mode": PI0_NAV_PICK_VLA_MODE,
                    "handoff_state": _HANDOFF_PAUSED,
                    "controller_handoff_pass": False,
                    "env_pid": os.getpid(),
                    "env_step": int(self._env_steps),
                    "held_hand": held_hand,
                    "press_hand": press_hand,
                    "action_source": self._action_source,
                    "vla_actions_enabled": False,
                    "vla_action_gate_confirmed": False,
                    "lifecycle_finalized": False,
                    "state_checkpoint_path": str(self._state_checkpoint_path),
                    "state_checkpoint_sha256": _sha256_file(
                        self._state_checkpoint_path
                    ),
                    "strict_local_grasp_success": False,
                    "usable_post_pick_saved": True,
                    "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                    "warnings": _wire_safe(post_pick_warnings),
                    "handoff_failure_diagnostics_path": str(diagnostics_path),
                }
                _write_json_atomic(self._paused_runtime_path, paused)
                return {
                    "held_hand": held_hand,
                    "press_hand": press_hand,
                    "state_checkpoint_path": str(self._state_checkpoint_path),
                    "paused_runtime_path": str(self._paused_runtime_path),
                    "controller_reload": reload_report,
                    "handoff_env_steps": handoff_env_steps,
                    "strict_local_grasp_success": False,
                    "usable_post_pick_saved": True,
                    "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
                    "warnings": _wire_safe(post_pick_warnings),
                }
            self._handoff_state = _HANDOFF_FAILED
            # A committed state checkpoint is immutable evidence even when a
            # later paused-runtime or mirror write fails.  Only incomplete
            # temporary files are safe to discard here.
            for path in generated:
                path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
            state_pending.unlink(missing_ok=True)
            raise

    def _prepare_prepress_planner_readiness(
        self,
        *,
        held_hand: str,
        ignore_collision_checks: bool = False,
    ) -> dict[str, Any]:
        """Prepare only the planner generator used after the pick handoff.

        A post-pick current-q query intentionally contains gripper/radio contact.
        The generic planner warmup includes the unrelated BASE generator and
        treats that contact as a world collision, so it cannot be a valid
        readiness gate for the held-arm-only pre-press phase.  Target-specific
        plan and attached-radio full-path collision validation remain mandatory
        in ``prepress_move_to`` before any waypoint can execute.
        """

        if held_hand not in {"left", "right"}:
            raise ValueError(f"invalid held_hand {held_hand!r}")
        planner = self._require_planner()
        planner.on_simulator_state_restored()
        radio, _table = self._resolve_handoff_targets()
        expected_root = getattr(radio, "root_link", None)
        if expected_root is None:
            raise RuntimeError("radio root link is unavailable for pre-press warmup")
        readiness = planner.warmup_prepress(
            hand=held_hand,
            expected_attached_root=expected_root,
            ignore_collision_checks=bool(ignore_collision_checks),
        )
        stages = readiness.get("stages")
        required_stages = (
            "current_q_attached_combined_collision",
            "current_pose_attached_full_trajectory",
            "identity_neighborhood_connected_path",
        )
        valid = bool(
            readiness.get("status") == "complete"
            and readiness.get("generator_kind") == "prepress_arm"
            and readiness.get("held_hand") == held_hand
            and readiness.get("base_generator_warmed") is False
            and readiness.get("unrelated_press_arm_generator_warmed") is False
            and isinstance(stages, dict)
            and all(
                isinstance(stages.get(name), dict) and stages[name].get("ok") is True
                for name in required_stages
            )
            and readiness.get("attached_collision_body", {}).get(
                "root_matches_expected_radio"
            )
            is True
            and isinstance(readiness.get("robot_q_pose_jump_max"), (int, float))
            and not isinstance(readiness.get("robot_q_pose_jump_max"), bool)
            and float(readiness["robot_q_pose_jump_max"]) <= 1e-6
        )
        if not valid:
            raise RuntimeError("pre-press planner readiness report is invalid")
        return readiness

    def finalize_paused_runtime(self, vla_status: dict[str, Any]) -> dict[str, Any]:
        if self._control_mode != PI0_NAV_PICK_VLA_MODE:
            raise RuntimeError(
                "paused runtime finalization is exclusive to pi0_nav_pick"
            )
        if (
            self._handoff_state != _HANDOFF_PAUSED
            or not self._paused_runtime_path.is_file()
        ):
            raise RuntimeError("runtime is not in a checkpointed PAUSED state")
        if (
            not isinstance(vla_status, dict)
            or vla_status.get("actions_enabled") is not False
        ):
            raise RuntimeError("VLA action disable confirmation is missing")
        health = vla_status.get("healthz")
        if not isinstance(health, dict) or health.get("actions_enabled") is not False:
            raise RuntimeError("VLA health did not confirm actions_enabled=false")
        pid = health.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("VLA health pid is invalid")
        endpoint = vla_status.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise RuntimeError(
                "VLA endpoint is required for paused runtime finalization"
            )
        paused = json.loads(self._paused_runtime_path.read_text(encoding="utf-8"))
        paused.update(
            {
                "vla_pid": pid,
                "vla_endpoint": endpoint,
                "vla_actions_enabled": False,
                "vla_action_gate_confirmed": True,
                "vla_status": _wire_safe(vla_status),
                "lifecycle_finalized": True,
                "finalized_at_unix_s": time.time(),
            }
        )
        _write_json_atomic(self._paused_runtime_path, paused)
        self._vla_actions_enabled = False
        return {
            "paused_runtime_path": str(self._paused_runtime_path),
            "handoff_state": self._handoff_state,
            "vla_actions_enabled": False,
            "lifecycle_finalized": True,
            "env_pid": os.getpid(),
            "vla_pid": pid,
            "vla_endpoint": endpoint,
        }

    def _sensor_for_camera(self, camera: str) -> Any | None:
        robot = self._robot()
        sensors = getattr(robot, "sensors", None) if robot is not None else None
        if not sensors:
            return None
        camera = canonical_camera(camera)
        needles = {
            "head": ("zed", "head"),
            "left_wrist": ("left_realsense", "left_wrist"),
            "right_wrist": ("right_realsense", "right_wrist"),
        }[camera]
        for name, sensor in sensors.items():
            lowered = str(name).lower()
            prim_path = str(getattr(sensor, "prim_path", "")).lower()
            if any(needle in lowered or needle in prim_path for needle in needles):
                return sensor
        return None

    def _camera_to_world(
        self,
        *,
        camera: str,
        payload: dict[str, Any],
        sensor: Any | None,
    ) -> np.ndarray:
        # Explicit Kit view matrices and the sensor's render annotator are tied
        # to the pixels just returned. Pose-like payload fields may already
        # reflect a newer articulation state, especially for wrist cameras.
        view = _payload_matrix(
            payload, ("view_matrix", "view_transform", "world_to_camera")
        )
        if view is not None:
            return np.linalg.inv(view.T)
        if sensor is not None:
            sensor_matrix = _sensor_camera_to_world(sensor)
            if sensor_matrix is not None:
                return sensor_matrix
        direct = _payload_matrix(
            payload,
            (
                "camera_to_world",
                "cam2world",
                "camera_to_world_matrix",
                "world_from_camera",
            ),
        )
        if direct is not None:
            return direct
        pose = payload.get("pose") or payload.get("camera_pose")
        if isinstance(pose, dict) and "position" in pose and "orientation" in pose:
            return _matrix_from_pose(pose["position"], pose["orientation"])
        raise CameraGeometryError(f"camera pose unavailable for {camera}")

    def _record_rgbd_frames(
        self, raw_observations: Any, observation: dict[str, Any]
    ) -> None:
        raw = _first_env_value(raw_observations)
        if raw is None:
            return
        payloads: dict[str, dict[str, Any]] = {}
        for path, payload in _iter_sensor_payloads(raw):
            camera = _payload_camera_name(path)
            if camera is not None:
                payloads[camera] = payload
        expected_cameras = ("head", "left_wrist", "right_wrist")
        try:
            missing = [camera for camera in expected_cameras if camera not in payloads]
            if missing:
                raise CameraGeometryError(
                    f"same-step RGB-D capture missing cameras: {missing}"
                )
            frames: dict[str, dict[str, Any]] = {}
            for camera in expected_cameras:
                payload = payloads[camera]
                rgb = _payload_rgb(payload)
                depth = _payload_depth(payload)
                if rgb is None or depth is None:
                    raise CameraGeometryError(
                        f"same-step capture missing RGB or depth for {camera}"
                    )
                sensor = self._sensor_for_camera(camera)
                rgb_array = np.asarray(_numpy_tree(rgb))
                intrinsics = _payload_intrinsics(
                    payload, rgb_shape=rgb_array.shape
                ) or (
                    _sensor_intrinsics(sensor, rgb_shape=rgb_array.shape)
                    if sensor is not None
                    else None
                )
                if intrinsics is None:
                    raise CameraGeometryError(
                        f"verified camera intrinsics unavailable for {camera}"
                    )
                frames[camera] = {
                    "rgb": rgb_array,
                    "depth_m": depth,
                    "intrinsics": intrinsics,
                    "camera_to_world": self._camera_to_world(
                        camera=camera,
                        payload=payload,
                        sensor=sensor,
                    ),
                }
            compact_proprio = extract_policy_state(observation["states"])
            self._frame_cache.add_capture_group(
                frames=frames,
                step_index=self._env_steps,
                capture_metadata={
                    "proprio": {
                        "values": compact_proprio.astype(float).tolist(),
                        "dimension": int(compact_proprio.size),
                        "layout": "POLICY_STATE_SEGMENTS",
                        "segments": segment_ranges(POLICY_STATE_SEGMENTS),
                    }
                },
            )
            instance_masks: dict[str, np.ndarray] = {}
            for camera in expected_cameras:
                raw_mask = payloads[camera].get("seg_instance_id")
                if raw_mask is None:
                    instance_masks = {}
                    break
                mask = np.asarray(_numpy_tree(raw_mask))
                if mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = mask[..., 0]
                if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
                    instance_masks = {}
                    break
                instance_masks[camera] = np.ascontiguousarray(mask)
            self._last_instance_id_masks = instance_masks
            self._last_capture_step = self._env_steps if instance_masks else None
        except Exception:
            # Atomicity is intentional: never publish one camera from a newer
            # simulator step beside two cameras from an older step.
            logger.exception(
                "failed to cache atomic BEHAVIOR RGB-D capture group at sim step %s",
                self._env_steps,
            )

    def _append_video(self, observation: dict[str, Any]) -> None:
        if self._video_error is not None or self._video_sealed:
            return
        try:
            import imageio.v2 as imageio
            import numpy as np

            head = np.asarray(observation["main_images"], dtype=np.uint8)
            if head.ndim != 3 or head.shape[2] < 3:
                raise RuntimeError(
                    f"planner head video must be HxWxC, got {head.shape}"
                )
            self._video_source_shapes["head"] = list(head.shape)
            frame = head[..., :3]
            if self._control_mode in _AUDIT_VIDEO_MODES:
                wrists = np.asarray(observation["wrist_images"], dtype=np.uint8)
                if wrists.ndim != 4 or wrists.shape[0] != 2:
                    raise RuntimeError(
                        "planner video requires synchronized left/right wrist RGB"
                    )
                height, width = head.shape[:2]
                self._video_source_shapes["left_wrist"] = list(wrists[0].shape)
                self._video_source_shapes["right_wrist"] = list(wrists[1].shape)
                left_wrist = _resize_video_tile(wrists[0], height=height, width=width)
                right_wrist = _resize_video_tile(wrists[1], height=height, width=width)
                frame = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
                frame[:height, :width] = head[..., :3]
                frame[:height, width:] = left_wrist
                frame[height:, :width] = right_wrist
            self._video_source_shapes["output"] = list(frame.shape)
            # Open the encoder only after a complete frame has validated.  A
            # malformed first capture must not leave an empty MP4 handle.
            if self._video_writer is None:
                self._video_path.parent.mkdir(parents=True, exist_ok=True)
                self._video_writer = imageio.get_writer(self._video_path, fps=15)
            self._video_writer.append_data(frame)
            self._video_frames += 1
        except Exception as exc:
            self._video_error = f"{type(exc).__name__}: {exc}"
            logger.exception("failed to append BEHAVIOR video frame")

    def _finalize_video_segment(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        video_meta = {
            "path": str(self._video_path),
            "fps": 15,
            "sample_every_env_steps": self._planner_video_interval_steps,
            "frames": self._video_frames,
            "error": self._video_error,
            "source_shapes": dict(self._video_source_shapes),
            "layout": (
                "2x2:head,left_wrist/right_wrist,blank"
                if self._control_mode in _AUDIT_VIDEO_MODES
                else "head"
            ),
        }
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            self._video_path.with_name(f"{self._video_path.stem}_meta.json"),
            video_meta,
        )
        _write_json_atomic(self._video_path.parent / "video_meta.json", video_meta)

    def start_video_segment(self, path: str | Path) -> None:
        """Rotate acceptance video without exposing a new planner RPC method."""

        self._finalize_video_segment()
        self._video_path = Path(path).expanduser().resolve()
        self._video_writer = None
        self._video_sealed = False
        self._video_frames = 0
        self._video_error = None
        self._video_source_shapes = {}
        if self._last_observation is not None:
            self._append_video(self._last_observation)

    def dump_simulator_state(self, *, serialized: bool = True) -> Any:
        """Capture the complete in-process simulator state for test isolation."""

        import omnigibson as og

        return og.sim.dump_state(serialized=bool(serialized))

    @staticmethod
    def _scene_assisted_grasp_entries(value: Any) -> list[dict[str, Any]]:
        """Collect assisted-grasp mappings from a decoded official scene JSON."""

        entries: list[dict[str, Any]] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "ag_obj_constraint_params" and isinstance(item, dict):
                    entries.append(item)
                entries.extend(BehaviorEnvFacade._scene_assisted_grasp_entries(item))
        elif isinstance(value, list):
            for item in value:
                entries.extend(BehaviorEnvFacade._scene_assisted_grasp_entries(item))
        return entries

    def save_post_pick_debug_mirror(self) -> dict[str, Any]:
        """Save a debug-only official scene JSON that preserves assisted grasp.

        This is intentionally an internal env RPC rather than an LLM tool.  It
        is never a substitute for ``state_checkpoint_1.json`` and can only be
        produced after the formal post-pick handoff has been finalized.
        """

        if self._control_mode != PI0_NAV_PICK_VLA_MODE:
            raise RuntimeError("post-pick debug mirror requires pi0_nav_pick_vla")
        mirror_warnings: list[dict[str, Any]] = []
        if (
            self._handoff_state != _HANDOFF_PAUSED
            or self._action_source != "curobo"
            or self._vla_actions_enabled is not False
            or not self._paused_runtime_path.is_file()
        ):
            mirror_warnings.append(
                _post_pick_warning(
                    "runtime_not_strict_paused_curobo",
                    "debug mirror source is not a strict finalized PAUSED/CuRobo runtime",
                    metrics={
                        "handoff_state": self._handoff_state,
                        "action_source": self._action_source,
                        "vla_actions_enabled": self._vla_actions_enabled,
                        "paused_runtime_exists": self._paused_runtime_path.is_file(),
                    },
                )
            )
        if not self._paused_runtime_path.is_file():
            raise RuntimeError("post-pick debug mirror paused runtime JSON is missing")
        paused = json.loads(self._paused_runtime_path.read_text(encoding="utf-8"))
        if (
            paused.get("lifecycle_finalized") is not True
            or paused.get("vla_action_gate_confirmed") is not True
        ):
            mirror_warnings.append(
                _post_pick_warning(
                    "vla_gate_not_confirmed",
                    "debug mirror source lacks a finalized VLA action-gate confirmation",
                )
            )
        checkpoint_path, checkpoint, checkpoint_sha = self._read_post_pick_checkpoint()
        if paused.get("state_checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("paused runtime and state_checkpoint_1 SHA differ")
        mirror_warnings.extend(
            warning
            for warning in checkpoint.get("warnings", [])
            if isinstance(warning, dict)
        )
        held = checkpoint["held_hand"]
        press = checkpoint["press_hand"]
        radio, _table = self._resolve_handoff_targets()
        robot = self._robot()
        if robot is None:
            raise RuntimeError("post-pick debug mirror robot is unavailable")
        constraint_params = getattr(robot, "_ag_obj_constraint_params", {})
        held_params = (
            constraint_params.get(held, {})
            if isinstance(constraint_params, dict)
            else {}
        )
        press_params = (
            constraint_params.get(press, {})
            if isinstance(constraint_params, dict)
            else {}
        )
        radio_prim_path = str(getattr(radio, "prim_path", ""))
        radio_root_link = getattr(radio, "root_link", None)
        radio_root_prim_path = str(getattr(radio_root_link, "prim_path", ""))
        assisted_grasp_prim_path = str(
            held_params.get("ag_obj_prim_path", "")
            if isinstance(held_params, dict)
            else ""
        )
        if not radio_prim_path:
            raise RuntimeError("post-pick debug mirror radio prim path is unavailable")
        manifest_assisted_grasp_prim_path = (
            assisted_grasp_prim_path or radio_root_prim_path or radio_prim_path
        )
        source_attachment = self._attachment_matches(held, radio)
        press_attachment = self._attachment_matches(press, radio)
        source_attachment_bound = bool(
            source_attachment[0]
            and isinstance(held_params, dict)
            and bool(assisted_grasp_prim_path)
            and assisted_grasp_prim_path in {radio_prim_path, radio_root_prim_path}
        )
        press_clear = bool(
            not press_attachment[0]
            and not press_attachment[1]
            and not bool(press_params)
        )
        if not source_attachment_bound:
            mirror_warnings.append(
                _post_pick_warning(
                    "source_assisted_grasp_not_bound",
                    "source runtime does not have an exact held-hand radio binding",
                )
            )
        if not press_clear:
            mirror_warnings.append(
                _post_pick_warning(
                    "press_hand_not_clear",
                    "press hand has radio attachment/contact bookkeeping at mirror save",
                )
            )
        opening: float | None = None
        if self._last_observation is None:
            mirror_warnings.append(
                _post_pick_warning(
                    "mirror_observation_unavailable",
                    "current public observation is unavailable for gripper diagnostics",
                )
            )
        else:
            try:
                opening = self._validated_selected_gripper_opening(
                    observation=self._last_observation, hand=held
                )
            except Exception as exc:
                mirror_warnings.append(
                    _post_pick_warning(
                        "held_gripper_opening_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        held_opening_closed = bool(
            opening is not None and opening < HANDOFF_GRIPPER_OPENING_MAX
        )
        if not held_opening_closed:
            mirror_warnings.append(
                _post_pick_warning(
                    "held_gripper_not_strictly_closed",
                    "held gripper opening does not pass the strict diagnostic threshold",
                    metrics={"opening_m": opening},
                )
            )
        held_latch_closed = bool(float(self._gripper_latch.get(held, 1.0)) == -1.0)
        if not held_latch_closed:
            mirror_warnings.append(
                _post_pick_warning(
                    "held_latch_not_closed",
                    "live held-hand command latch is not exactly close",
                )
            )
        if self._initial_radio_position is None:
            current_radio_position, _ = self._object_pose(radio)
            initial_radio_position = current_radio_position.astype(float).tolist()
            mirror_warnings.append(
                _post_pick_warning(
                    "initial_radio_position_unavailable",
                    "using current radio position as the debug reference",
                )
            )
        else:
            initial_radio_position = self._initial_radio_position.astype(float).tolist()

        bundle = self._output_dir / "debug_mirror_post_pick"
        pending_bundle = self._output_dir / ".debug_mirror_post_pick.pending"
        if bundle.exists() or pending_bundle.exists():
            raise RuntimeError("post-pick debug mirror bundle already exists")
        pending_bundle.mkdir(parents=True)
        scene_path = pending_bundle / DEBUG_MIRROR_SCENE_NAME
        bundled_checkpoint = pending_bundle / DEBUG_MIRROR_CHECKPOINT_NAME
        shutil.copyfile(checkpoint_path, bundled_checkpoint)

        import omnigibson as og

        try:
            if len(og.sim.scenes) != 1:
                raise RuntimeError("post-pick debug mirror requires exactly one scene")
            # OmniGibson's current assisted-grasp dump uses a shallow copy and
            # rewrites contact_pos into scene coordinates.  Preserve the live
            # robot bookkeeping around the official scene save.
            assisted_grasp_backup = deepcopy(robot._ag_obj_constraint_params)
            try:
                og.sim.save([str(scene_path)])
            finally:
                robot._ag_obj_constraint_params = assisted_grasp_backup
            if not scene_path.is_file():
                raise RuntimeError("OmniGibson did not create the debug mirror scene")
            scene_payload = json.loads(scene_path.read_text(encoding="utf-8"))
            grasp_entries = self._scene_assisted_grasp_entries(
                scene_payload.get("state")
            )
            matched = [
                entry
                for entry in grasp_entries
                if isinstance(entry.get(held), dict)
                and entry[held].get("ag_obj_prim_path") == assisted_grasp_prim_path
                and not entry.get(press)
            ]
            attachment_serialized = bool(len(matched) == 1 and len(grasp_entries) == 1)
            if source_attachment_bound and not attachment_serialized:
                raise RuntimeError(
                    "official scene JSON lost the source held-radio grasp binding"
                )
            if not attachment_serialized:
                mirror_warnings.append(
                    _post_pick_warning(
                        "assisted_grasp_not_serialized",
                        "diagnostic mirror does not contain one exact held-radio grasp",
                    )
                )
            live_attachment_unchanged = bool(
                self._attachment_matches(held, radio) == (True, True)
            )
            if source_attachment_bound and not live_attachment_unchanged:
                raise RuntimeError("scene save changed the live radio attachment")
            if not live_attachment_unchanged:
                mirror_warnings.append(
                    _post_pick_warning(
                        "live_attachment_not_strict_after_save",
                        "live attachment diagnostics are not strict after scene save",
                    )
                )

            controllers = getattr(robot, "controllers", {})
            layout = [
                [str(name), int(controller.command_dim)]
                for name, controller in controllers.items()
            ]
            base_motor_type = str(getattr(controllers.get("base"), "motor_type", ""))
            try:
                base_controller_signature = self._base_controller_signature(robot)
            except Exception as exc:
                base_controller_signature = {}
                mirror_warnings.append(
                    _post_pick_warning(
                        "base_controller_signature_unavailable",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            position_controller_ready = bool(base_motor_type == "position")
            if not position_controller_ready:
                mirror_warnings.append(
                    _post_pick_warning(
                        "base_controller_not_position",
                        "source mirror controller is not confirmed position-base",
                        metrics={"base_motor_type": base_motor_type},
                    )
                )
            strict_local_grasp_success = bool(
                checkpoint.get("strict_local_grasp_success", False)
                and source_attachment_bound
                and press_clear
                and held_opening_closed
                and held_latch_closed
            )
            payload = build_debug_mirror_manifest(
                scene_path=scene_path,
                checkpoint_path=bundled_checkpoint,
                meta=self._meta,
                held_hand=held,
                press_hand=press,
                object_name=checkpoint["object_name"],
                object_prim_path=radio_prim_path,
                assisted_grasp_prim_path=manifest_assisted_grasp_prim_path,
                source_env_step=self._env_steps,
                initial_radio_position=initial_radio_position,
                gripper_latches={
                    "left": float(self._gripper_latch["left"]),
                    "right": float(self._gripper_latch["right"]),
                },
                controller_layout=layout,
                base_motor_type=base_motor_type,
                base_controller_signature=base_controller_signature,
                official_task_success=bool(_raw_success(self._last_info)),
                strict_local_grasp_success=strict_local_grasp_success,
                validation={
                    "source_assisted_grasp_bound": source_attachment_bound,
                    "press_hand_clear": press_clear,
                    "held_gripper_closed": held_opening_closed,
                    "held_latch_closed": held_latch_closed,
                    "attachment_serialized": attachment_serialized,
                    "live_attachment_unchanged": live_attachment_unchanged,
                    "position_controller_ready": position_controller_ready,
                    "restore_eligible": bool(
                        attachment_serialized
                        and source_attachment_bound
                        and position_controller_ready
                    ),
                },
                warnings=_wire_safe(mirror_warnings),
            )
            write_debug_mirror_manifest(pending_bundle, payload)
            # Integrity, task binding, role identity, checkpoint schema, and
            # hashes are hard requirements even though physical diagnostics
            # are warning-only.
            validate_debug_mirror_bundle(scene_path, meta=self._meta)
            os.replace(pending_bundle, bundle)
        except BaseException:
            shutil.rmtree(pending_bundle, ignore_errors=True)
            raise
        scene_path = bundle / DEBUG_MIRROR_SCENE_NAME
        bundled_checkpoint = bundle / DEBUG_MIRROR_CHECKPOINT_NAME
        manifest_path = bundle / "debug_mirror_post_pick.manifest.json"
        return {
            "debug_only": True,
            "not_robot_motion_checkpoint": True,
            "scene_path": str(scene_path),
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(bundled_checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "strict_local_grasp_success": bool(payload["strict_local_grasp_success"]),
            "usable_post_pick_saved": True,
            "save_policy": POST_PICK_DEBUG_SAVE_POLICY,
            "warnings": _wire_safe(mirror_warnings),
            "restore_eligible": bool(payload["validation"].get("restore_eligible")),
            "held_hand": held,
            "press_hand": press,
            "object_name": checkpoint["object_name"],
            "source_env_step": int(self._env_steps),
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
        }

    def restore_simulator_state(
        self,
        state: Any,
        *,
        serialized: bool = True,
        gripper_latches: dict[str, float] | None = None,
    ) -> None:
        """Restore a trusted in-process snapshot and invalidate stale caches."""

        from copy import deepcopy

        import omnigibson as og

        restore_state = deepcopy(state)
        if serialized:
            template = og.sim.dump_state(serialized=True)
            if int(restore_state.numel()) != int(template.numel()):
                raise RuntimeError(
                    "simulator snapshot size does not match loaded scene: "
                    f"snapshot={int(restore_state.numel())} "
                    f"scene={int(template.numel())}"
                )
            restore_state = restore_state.to(
                device=template.device, dtype=template.dtype
            )
        og.sim.load_state(restore_state, serialized=bool(serialized))
        # Set the command latch before OmniGibson's mandatory propagation
        # step, so the first subsequent facade action cannot reopen the held
        # gripper and release the restored assisted grasp.
        self._gripper_latch = (
            {"left": 1.0, "right": 1.0}
            if gripper_latches is None
            else {
                "left": float(gripper_latches["left"]),
                "right": float(gripper_latches["right"]),
            }
        )
        # OmniGibson documents that one physics update is required after load
        # for spatial object states to become current.
        og.sim.step_physics()
        self._done = False
        self._last_info = None
        self._frame_cache.clear()
        if self._planner is not None:
            self._planner.on_simulator_state_restored()
        # Kit camera annotators trail articulation changes.  OmniGibson's own
        # camera wrapper documents that at least three render calls are needed
        # before RGB/depth/cameraViewTransform describe the restored pose.
        # Rendering does not advance task physics or alter official success.
        _settle_visual_pipeline_after_restore(og.sim)
        self._refresh_observation_without_step()
        if self._last_observation is not None:
            self._append_video(self._last_observation)

    def _restore_post_pick_debug_mirror(self, configured_path: Path) -> dict[str, Any]:
        """Load a trusted post-pick scene into the already-created task env."""

        if self._control_mode != PI0_NAV_PICK_VLA_MODE:
            raise RuntimeError("post-pick debug mirror requires pi0_nav_pick_vla")
        manifest, scene_path, bundled_checkpoint = validate_debug_mirror_bundle(
            configured_path, meta=self._meta
        )
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        if not isinstance(scene, dict) or not isinstance(scene.get("state"), dict):
            raise RuntimeError("post-pick debug mirror scene state is invalid")
        if self._state_checkpoint_path.exists():
            raise RuntimeError("debug mirror output state_checkpoint_1 already exists")
        import omnigibson as og

        if len(og.sim.scenes) != 1:
            raise RuntimeError(
                "post-pick debug mirror requires exactly one loaded scene"
            )
        current_scene = og.sim.scenes[0]
        objects_info = scene.get("objects_info")
        scene_object_init = (
            objects_info.get("init_info") if isinstance(objects_info, dict) else None
        )
        if not isinstance(scene_object_init, dict):
            raise RuntimeError("post-pick debug mirror lacks object registry binding")
        current_object_names = set(
            current_scene.object_registry.get_dict("name").keys()
        )
        if set(scene_object_init) != current_object_names:
            raise RuntimeError("post-pick debug mirror object registry mismatch")
        state_registry = scene["state"].get("registry")
        scene_systems = (
            state_registry.get("system_registry")
            if isinstance(state_registry, dict)
            else None
        )
        if not isinstance(scene_systems, dict) or set(scene_systems) != set(
            current_scene.active_systems
        ):
            raise RuntimeError("post-pick debug mirror system registry mismatch")
        roles = manifest["roles"]
        held = roles["held_hand"]
        press = roles["press_hand"]
        grasp_entries = self._scene_assisted_grasp_entries(scene["state"])
        matching_grasps = [
            entry
            for entry in grasp_entries
            if isinstance(entry.get(held), dict)
            and entry[held].get("ag_obj_prim_path") == roles["assisted_grasp_prim_path"]
            and not entry.get(press)
        ]
        if len(matching_grasps) != 1 or len(grasp_entries) != 1:
            raise RuntimeError("post-pick debug mirror lacks the bound radio grasp")
        initial_radio, _table = self._resolve_handoff_targets()
        if str(getattr(initial_radio, "prim_path", "")) != roles["object_prim_path"]:
            raise RuntimeError("post-pick debug mirror radio prim path mismatch")

        reload_report = self._reload_base_controller_position()
        expected_layout = manifest["controller"].get("layout")
        if reload_report.get("controller_layout") != expected_layout:
            raise RuntimeError("post-pick debug mirror controller layout mismatch")
        robot_before_load = self._robot()
        controllers = getattr(robot_before_load, "controllers", {})
        current_motor_type = str(getattr(controllers.get("base"), "motor_type", ""))
        if current_motor_type != manifest["controller"].get("base_motor_type"):
            raise RuntimeError("post-pick debug mirror base motor type mismatch")
        if self._base_controller_signature(robot_before_load) != manifest[
            "controller"
        ].get("base_controller_signature"):
            raise RuntimeError(
                "post-pick debug mirror base controller signature mismatch"
            )
        from omnigibson.utils.python_utils import recursively_convert_to_torch

        scene_state = recursively_convert_to_torch(scene["state"])
        latches = manifest["controller"]["gripper_latches"]
        latches[held] = -1.0
        self.restore_simulator_state(
            {0: scene_state},
            serialized=False,
            gripper_latches=latches,
        )
        self._state_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_temporary = self._state_checkpoint_path.with_suffix(".json.tmp")
        shutil.copyfile(bundled_checkpoint, checkpoint_temporary)
        os.replace(checkpoint_temporary, self._state_checkpoint_path)

        self._initial_radio_position = np.asarray(
            manifest["source"]["initial_radio_position"], dtype=np.float64
        )
        if (
            self._initial_radio_position.shape != (3,)
            or not np.isfinite(self._initial_radio_position).all()
        ):
            raise RuntimeError("debug mirror initial radio position is invalid")
        radio, _table = self._resolve_handoff_targets()
        robot = self._robot()
        if robot is None:
            raise RuntimeError("debug mirror robot is unavailable after restore")
        restored_params = getattr(robot, "_ag_obj_constraint_params", {})
        restored_constraints = getattr(robot, "_ag_obj_constraints", {})
        held_params = (
            restored_params.get(held, {}) if isinstance(restored_params, dict) else {}
        )
        press_params = (
            restored_params.get(press, {}) if isinstance(restored_params, dict) else {}
        )
        held_constraint = (
            restored_constraints.get(held)
            if isinstance(restored_constraints, dict)
            else None
        )
        press_constraint = (
            restored_constraints.get(press)
            if isinstance(restored_constraints, dict)
            else None
        )
        constraint_valid = held_constraint is not None and (
            not hasattr(held_constraint, "IsValid") or bool(held_constraint.IsValid())
        )
        restored_held_attachment = self._attachment_matches(held, radio)
        restored_press_attachment = self._attachment_matches(press, radio)
        if (
            not restored_held_attachment[0]
            or restored_press_attachment[0]
            or restored_press_attachment[1]
            or str(getattr(radio, "prim_path", "")) != roles["object_prim_path"]
            or held_params.get("ag_obj_prim_path") != roles["assisted_grasp_prim_path"]
            or bool(press_params)
            or not constraint_valid
            or press_constraint is not None
        ):
            raise RuntimeError("debug mirror did not rebuild the bound assisted grasp")
        self._held_hand = held
        self._action_source = "curobo"
        # The runtime provider disables and health-checks the VLA before it
        # enters env.reset() for a mirror startup.  Toolkit creation still
        # happens only after finalize_paused_runtime binds that proof.
        self._vla_actions_enabled = False
        self._handoff_state = _HANDOFF_STABLE_VALIDATION
        self._prepress_context = {
            "checkpoint_path": str(self._state_checkpoint_path),
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "checkpoint": json.loads(
                self._state_checkpoint_path.read_text(encoding="utf-8")
            ),
            "held_hand": held,
            "press_hand": press,
            "object_name": roles["object_name"],
            "bound_env_step": int(self._env_steps),
        }
        import torch

        initial_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        hold = np.asarray(
            self._require_planner().backend.hold_action(), dtype=np.float32
        ).reshape(-1)
        if hold.shape != (23,) or not np.isfinite(hold).all():
            raise RuntimeError("debug mirror first hold action is invalid")
        hold[ENV_ACTION_SEGMENTS[f"{held}_gripper"]] = -1.0
        hold[ENV_ACTION_SEGMENTS[f"{press}_gripper"]] = float(
            self._gripper_latch[press]
        )
        hold_trace: list[dict[str, Any]] = []
        source_already_successful = bool(
            manifest.get("source", {}).get("official_task_success", False)
        )
        for hold_index in range(HANDOFF_VALIDATION_FRAMES):
            step_obs, _reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    torch.as_tensor(hold, dtype=torch.float32).reshape(1, 23),
                    need_obs=True,
                )
            )
            self._env_steps += 1
            self._last_info = _numpy_tree(step_infos[0])
            observation = _single_observation(self._env._wrap_obs(step_obs))
            self._last_observation = observation
            self._record_rgbd_frames(step_obs, observation)
            self._append_video(observation)
            stability = self._prepress_stability_snapshot()
            current_q = np.asarray(
                _numpy_tree(robot.get_joint_positions()), dtype=np.float64
            ).reshape(-1)
            frame = {
                "hold_index": hold_index + 1,
                "env_step": int(self._env_steps),
                "robot_q_drift_max": float(np.max(np.abs(current_q - initial_q))),
                "stability": _wire_safe(stability),
                "held_constraint_present": bool(
                    getattr(robot, "_ag_obj_constraints", {}).get(held) is not None
                ),
                "press_constraint_absent": bool(
                    getattr(robot, "_ag_obj_constraints", {}).get(press) is None
                ),
                "episode_status": {
                    "terminated": bool(_scalar_bool(step_term)),
                    "truncated": bool(_scalar_bool(step_trunc)),
                    "done": bool(_raw_done(self._last_info)),
                    "official_task_success": bool(_raw_success(self._last_info)),
                },
            }
            hold_trace.append(frame)
            if any(frame["episode_status"].values()):
                if (
                    source_already_successful
                    and frame["episode_status"]["official_task_success"]
                    and not frame["episode_status"]["truncated"]
                ):
                    # A success mirror is used only for the bounded post-success
                    # hold/retreat/open cleanup.  The simulator necessarily
                    # reports termination on its first controlled frame, so
                    # one stable frame is the complete restore validation.
                    held_stable_for_cleanup = bool(
                        float(stability["relative_drift_m"])
                        <= HANDOFF_RELATIVE_DRIFT_MAX_M
                        and float(stability["angular_drift_rad"])
                        <= HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD
                        and float(stability["air_gap_m"])
                        >= HANDOFF_SUPPORT_GAP_MIN_M
                        and float(stability["held_gripper_opening_m"])
                        < HANDOFF_GRIPPER_OPENING_MAX
                        and any(stability["held_attachment"])
                        and not any(stability["press_attachment"])
                    )
                    if (
                        not held_stable_for_cleanup
                        or not frame["held_constraint_present"]
                        or not frame["press_constraint_absent"]
                    ):
                        raise RuntimeError(
                            "successful debug mirror failed stable cleanup restore"
                        )
                    break
                raise RuntimeError(
                    "debug mirror episode stopped during close-hold validation"
                )
            if (
                not stability["stable"]
                or not frame["held_constraint_present"]
                or not frame["press_constraint_absent"]
            ):
                raise RuntimeError("debug mirror grasp failed close-hold validation")
        self._prepress_context = None
        self._handoff_state = _HANDOFF_PAUSED
        self._reset_completed = True
        self._done = bool(_raw_success(self._last_info))
        warmup = self._prepare_prepress_planner_readiness(
            held_hand=held,
            ignore_collision_checks=True,
        )
        visual_evidence = self._checkpoint_visual_evidence(
            checkpoint_name="debug_mirror_restore",
            held_hand=held,
            press_hand=press,
        )
        verification_path = self._output_dir / "debug_mirror_restore_verification.json"
        verification = {
            "schema_version": 1,
            "debug_only": True,
            "not_official_episode_resume": True,
            "source_env_step": int(manifest["source"]["env_step"]),
            "restored_env_step": int(self._env_steps),
            "held_hand": held,
            "press_hand": press,
            "object_name": roles["object_name"],
            "controller_reload": reload_report,
            "stability": _wire_safe(stability),
            "mandatory_uncontrolled_physics_steps": 1,
            "first_controlled_action": hold.astype(float).tolist(),
            "close_hold_trace": hold_trace,
            "planner_warmup": _wire_safe(warmup),
            "visual_evidence": visual_evidence,
            "official_task_success": bool(_raw_success(self._last_info)),
        }
        _write_json_atomic(verification_path, verification)
        paused = {
            "schema_version": 1,
            "control_mode": PI0_NAV_PICK_VLA_MODE,
            "handoff_state": _HANDOFF_PAUSED,
            "env_pid": os.getpid(),
            "env_step": int(self._env_steps),
            "source_env_step": int(manifest["source"]["env_step"]),
            "held_hand": held,
            "press_hand": press,
            "action_source": "curobo",
            "vla_actions_enabled": False,
            "vla_action_gate_confirmed": False,
            "lifecycle_finalized": False,
            "debug_mirror_restore": True,
            "not_official_episode_resume": True,
            "debug_mirror_scene_path": str(scene_path),
            "state_checkpoint_path": str(self._state_checkpoint_path),
            "state_checkpoint_sha256": manifest["checkpoint"]["sha256"],
            "restore_verification_path": str(verification_path),
        }
        _write_json_atomic(self._paused_runtime_path, paused)
        self._restored_state = {
            "kind": "post_pick_debug_mirror",
            "debug_only": True,
            "scene_path": str(scene_path),
            "manifest_path": str(
                scene_path.parent / "debug_mirror_post_pick.manifest.json"
            ),
            "source_env_step": int(manifest["source"]["env_step"]),
        }
        return verification

    def reset(self) -> tuple[dict[str, Any], Any]:
        if self._control_mode == PI0_NAV_PICK_VLA_MODE and (
            bool(getattr(self, "_reset_completed", False))
            or self._handoff_state
            in {
                _HANDOFF_CHECKPOINTING,
                _HANDOFF_CONTROLLER_RELOAD,
                _HANDOFF_STABLE_VALIDATION,
                _HANDOFF_PAUSED,
                _HANDOFF_FAILED,
                _HANDOFF_OFFICIAL_SUCCESS,
            }
        ):
            raise RuntimeError("pi0_nav_pick permits exactly one initial reset")
        started_at = time.monotonic()
        logger.info("BEHAVIOR reset started on thread %s", threading.get_ident())
        raw_observations, infos = self._env.env_reset()
        observation = _single_observation(self._env._wrap_obs(raw_observations))
        self._done = False
        self._env_steps = 0
        self._last_observation = observation
        self._last_info = _numpy_tree(infos[0])
        if self._control_mode == PI0_NAV_PICK_VLA_MODE:
            self._video_sealed = False
            self._base_controller_mode = "velocity"
            self._action_source = "pi0_vla"
            self._vla_actions_enabled = True
            self._handoff_state = _HANDOFF_VLA_ACTIVE
            self._held_hand = None
            self._handoff_validator_frames.clear()
            self._handoff_target_objects = None
            self._initial_radio_position = None
        restore_path_value = os.environ.get("RPENT_BEHAVIOR_RESTORE_STATE")
        if self._debug_mirror_path is not None and restore_path_value:
            raise RuntimeError(
                "post-pick debug mirror and acceptance restore are mutually exclusive"
            )
        if self._debug_mirror_path is not None:
            self._restore_post_pick_debug_mirror(self._debug_mirror_path)
            observation = self._last_observation
            assert observation is not None
        elif restore_path_value:
            if self._control_mode not in _AUDIT_VIDEO_MODES:
                raise RuntimeError(
                    "RPENT_BEHAVIOR_RESTORE_STATE is acceptance-only and "
                    "forbidden outside planner, pi0_pick, or hybrid modes"
                )
            import torch

            restore_path = Path(restore_path_value).expanduser().resolve()
            if not restore_path.is_file():
                raise RuntimeError(
                    f"configured simulator restore state is missing: {restore_path}"
                )
            state = torch.load(
                restore_path,
                map_location="cpu",
                weights_only=True,
            )
            if not torch.is_tensor(state) or state.ndim != 1:
                raise RuntimeError(
                    "simulator restore artifact must contain one serialized tensor"
                )
            state_finite = bool(torch.isfinite(state).all().item())
            if not state_finite:
                raise RuntimeError("simulator restore tensor contains NaN or Inf")
            restore_manifest = validate_snapshot_manifest(
                restore_path,
                serialized_elements=int(state.numel()),
                serialized_dtype=str(state.dtype),
                serialized_shape=list(state.shape),
                serialized_finite=state_finite,
                meta={**self._meta, "control_mode": self._control_mode},
            )
            self.restore_simulator_state(state, serialized=True)
            observation = self._last_observation
            assert observation is not None
            self._last_info = _numpy_tree(infos[0])
            self._restored_state = {
                "path": str(restore_path),
                "sha256": restore_manifest["state"]["sha256"],
                "elements": int(state.numel()),
                "manifest_path": str(Path(f"{restore_path}.manifest.json")),
                "manifest_schema_version": restore_manifest["schema_version"],
            }
        else:
            self._record_rgbd_frames(raw_observations, observation)
            self._append_video(observation)
            self._restored_state = None
        if self._debug_mirror_path is None and self._control_mode in {
            PI0_NAV_PICK_VLA_MODE,
            HYBRID_VLM_PI0_MODE,
        }:
            radio, _table = self._resolve_handoff_targets()
            self._initial_radio_position = self._object_position(radio).copy()
        if self._control_mode in _PLANNER_POSITION_START_MODES:
            warmup = self._require_planner().warmup()
            logger.info(
                "BEHAVIOR planner cuRobo warmup completed in %.1fs artifact=%s",
                float(warmup.get("elapsed_s", 0.0)),
                warmup.get("artifact"),
            )
        logger.info(
            "BEHAVIOR reset completed in %.1fs on thread %s",
            time.monotonic() - started_at,
            threading.get_ident(),
        )
        if self._control_mode == PI0_NAV_PICK_VLA_MODE:
            self._reset_completed = True
        return observation, _numpy_tree(
            self._last_info if self._debug_mirror_path is not None else infos[0]
        )

    def chunk_step(self, actions: Any) -> tuple[Any, Any, bool, bool, Any]:
        return self._step_action_chunk(actions, observe_final=True)

    def pi0_chunk_step(
        self,
        actions: Any,
        *,
        hand: str,
        gripper_closed_threshold: float = 0.045,
        required_closed_steps: int = 3,
        stop_on_candidate: bool = False,
    ) -> tuple[Any, Any, bool, bool, Any]:
        """Execute a Pi0 chunk while monitoring the selected physical gripper.

        This Pi0-only RPC samples the actual selected gripper joints and the
        public raw proprio at every simulator step.  It fails closed if the two
        same-step openings disagree.  The configured number of consecutive
        closed steps (three by default) establishes only a local closure
        *candidate*, never a grasp or task success.
        """

        if hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        threshold = float(gripper_closed_threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("gripper_closed_threshold must be finite and non-negative")
        if (
            isinstance(required_closed_steps, bool)
            or int(required_closed_steps) != required_closed_steps
            or int(required_closed_steps) <= 0
        ):
            raise ValueError("required_closed_steps must be a positive integer")
        if not isinstance(stop_on_candidate, bool):
            raise ValueError("stop_on_candidate must be a boolean")
        return self._step_action_chunk(
            actions,
            observe_final=True,
            local_gripper_monitor={
                "hand": hand,
                "threshold": threshold,
                "required_closed_steps": int(required_closed_steps),
                "stop_on_candidate": stop_on_candidate,
            },
        )

    def pi0_nav_pick_chunk_step(
        self,
        actions: Any,
        *,
        chunk_index: int,
    ) -> tuple[Any, Any, bool, bool, Any]:
        """Execute an unmodified complete Pi0 chunk with strict local handoff."""

        if self._control_mode != PI0_NAV_PICK_VLA_MODE:
            raise RuntimeError(
                "pi0_nav_pick is available only in pi0_nav_pick_vla mode"
            )
        if self._handoff_state != _HANDOFF_VLA_ACTIVE or not self._vla_actions_enabled:
            raise RuntimeError("Pi0 actions are gated after handoff or failure")
        if (
            isinstance(chunk_index, bool)
            or int(chunk_index) != chunk_index
            or int(chunk_index) <= 0
        ):
            raise ValueError("chunk_index must be a positive integer")
        action_array = validate_action_chunk(actions, max_horizon=32)
        if action_array.shape != (32, 23):
            raise ValueError(
                f"pi0_nav_pick requires one complete [32,23] chunk, got {action_array.shape}"
            )
        try:
            ret = self._step_action_chunk(
                action_array,
                observe_final=True,
                pi0_nav_pick=True,
            )
            observation, reward, terminated, truncated, info = ret
            rpent = info.get("_rpent") if isinstance(info, dict) else None
            monitor = (
                rpent.get("pi0_nav_pick_monitor") if isinstance(rpent, dict) else None
            )
            if not isinstance(monitor, dict):
                raise RuntimeError("pi0_nav_pick monitor was not produced")
            monitor["visual_review"] = self._persist_pi0_nav_pick_views(
                chunk_index=int(chunk_index), validator=monitor
            )
            result = (observation, reward, terminated, truncated, info)
            pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            return result
        except Exception:
            self._vla_actions_enabled = False
            self._finalize_video_segment()
            self._video_sealed = True
            if self._state_checkpoint_path.is_file():
                self._handoff_state = _HANDOFF_PAUSED
            else:
                self._handoff_state = _HANDOFF_FAILED
            # Never revoke already-committed development evidence because a
            # later visual/tool-envelope step failed.
            self._state_checkpoint_path.with_suffix(".json.tmp").unlink(missing_ok=True)
            self._paused_runtime_path.with_suffix(".json.tmp").unlink(missing_ok=True)
            raise

    def pi0_navigate_to_chunk_step(
        self,
        actions: Any,
        *,
        segment_index: int,
        chunk_index: int,
    ) -> tuple[Any, Any, bool, bool, Any]:
        """Execute up to eight adapted 23D Pi0 navigation actions."""

        if (
            isinstance(segment_index, bool)
            or int(segment_index) != segment_index
            or int(segment_index) <= 0
        ):
            raise ValueError("segment_index must be a positive integer")
        if (
            isinstance(chunk_index, bool)
            or int(chunk_index) != chunk_index
            or int(chunk_index) <= 0
        ):
            raise ValueError("chunk_index must be a positive integer")
        action_array = validate_action_chunk(actions, max_horizon=8)
        ret = self._step_action_chunk(
            action_array,
            observe_final=True,
            pi0_navigate_to=True,
        )
        observation, reward, terminated, truncated, info = ret
        artifacts = self._persist_pi0_navigate_to_views(
            segment_index=int(segment_index),
            chunk_index=int(chunk_index),
        )
        public_info = dict(info) if isinstance(info, dict) else {"raw": info}
        rpent = public_info.setdefault("_rpent", {})
        if not isinstance(rpent, dict):
            raise RuntimeError("pi0_navigate_to accounting envelope is invalid")
        monitor = rpent.setdefault("pi0_navigate_to_monitor", {})
        if not isinstance(monitor, dict):
            raise RuntimeError("pi0_navigate_to monitor envelope is invalid")
        monitor["visual_review"] = artifacts
        result = (observation, reward, terminated, truncated, public_info)
        pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    def planner_step(self, action: Any) -> tuple[Any, Any, bool, bool, Any]:
        """Execute one internal planner action without forcing RGB-D rendering.

        The same official info and termination path as ``chunk_step`` is used;
        synchronized observations and video are sampled every four env steps by
        default (configurable only for internal acceptance runs);
        a later public ``observe`` can refresh all three cameras at the same
        simulator step without executing an action.
        This method is intentionally absent from the RPC allowlist.
        """
        action_array = validate_action_chunk(action)
        if action_array.shape[0] != 1:
            raise ValueError("planner_step requires exactly one 23D action")
        return self._step_action_chunk(action_array, observe_final=False)

    def _step_action_chunk(
        self,
        actions: Any,
        *,
        observe_final: bool,
        local_gripper_monitor: dict[str, Any] | None = None,
        pi0_navigate_to: bool = False,
        pi0_nav_pick: bool = False,
    ) -> tuple[Any, Any, bool, bool, Any]:
        import torch

        post_success_cleanup = bool(
            getattr(self, "_post_success_cleanup_active", False)
        )
        if self._done and not post_success_cleanup:
            raise RuntimeError("env.chunk_step called after episode stop")
        action_array = validate_action_chunk(actions)
        if pi0_navigate_to:
            if self._control_mode != HYBRID_VLM_PI0_MODE:
                raise RuntimeError("pi0_navigate_to is available only in hybrid mode")
            if action_array.shape[0] > 8:
                raise ValueError(
                    "pi0_navigate_to executes at most eight actions per chunk"
                )
            if local_gripper_monitor is not None:
                raise ValueError("pi0_navigate_to cannot monitor or modify a gripper")
        if pi0_nav_pick:
            if self._control_mode != PI0_NAV_PICK_VLA_MODE:
                raise RuntimeError(
                    "pi0_nav_pick is available only in its isolated mode"
                )
            if local_gripper_monitor is not None or pi0_navigate_to:
                raise ValueError("pi0_nav_pick cannot combine action adapters")
        action_tensor = torch.as_tensor(action_array, dtype=torch.float32).reshape(
            1, action_array.shape[0], action_array.shape[1]
        )
        final_observation = self._last_observation
        final_reward = None
        official_info: Any = {}
        terminated = False
        truncated = False
        executed_steps = 0
        monitor_result: dict[str, Any] | None = None
        monitor_candidate_env_step: int | None = None
        monitor_closed_streak = 0
        monitor_min_opening = float("inf")
        monitor_opening: float | None = None
        monitor_hand = (
            str(local_gripper_monitor["hand"])
            if local_gripper_monitor is not None
            else None
        )
        monitor_threshold = (
            float(local_gripper_monitor["threshold"])
            if local_gripper_monitor is not None
            else None
        )
        monitor_stop_on_candidate = bool(
            local_gripper_monitor is not None
            and local_gripper_monitor["stop_on_candidate"]
        )
        monitor_required_closed_steps = (
            int(local_gripper_monitor["required_closed_steps"])
            if local_gripper_monitor is not None
            else None
        )
        nav_checks: list[dict[str, Any]] = []
        nav_stop_reason: str | None = None
        nav_raw_base_actions: list[list[float]] = []
        nav_position_deltas: list[list[float]] = []
        nav_input_clip_count = 0
        nav_delta_clip_count = 0
        handoff_validator: dict[str, Any] = {
            "local_grasp_success": False,
            "held_hand": None,
            "per_hand": {},
            "current": {},
        }
        handoff_artifacts: dict[str, Any] = {}
        handoff_stop_reason: str | None = None
        for step_index in range(action_tensor.shape[1]):
            if pi0_nav_pick and self._env_steps >= int(self._meta["max_episode_steps"]):
                break
            is_last_action = step_index == action_tensor.shape[1] - 1
            observation_interval = (
                4 if bool(observe_final) else self._planner_video_interval_steps
            )
            need_observation = (
                pi0_nav_pick
                or local_gripper_monitor is not None
                or (bool(observe_final) and is_last_action)
                or (self._env_steps + 1) % observation_interval == 0
            )
            step_action = action_tensor[:, step_index]
            if pi0_navigate_to:
                raw = np.asarray(
                    step_action.detach().cpu().numpy(), dtype=np.float32
                ).reshape(23)
                base = ENV_ACTION_SEGMENTS["base"]
                raw_base = np.asarray(raw[base], dtype=np.float32)
                clipped_input = np.clip(raw_base, -1.0, 1.0)
                nav_input_clip_count += int(np.count_nonzero(clipped_input != raw_base))
                scaled_delta = (
                    clipped_input
                    * np.asarray([0.75, 0.75, 1.0], dtype=np.float32)
                    / 60.0
                )
                delta = np.clip(scaled_delta, -0.01, 0.01)
                nav_delta_clip_count += int(np.count_nonzero(delta != scaled_delta))
                # Keep the policy's trunk and both arm commands so its
                # receding visual/proprio trajectory advances beyond the
                # repeated eight-step startup motion. Only the position-base
                # segment needs a controller-specific adapter.
                adapted = raw.copy()
                # HolonomicBaseJointController's position input is a local
                # per-step delta, not an absolute base pose. Never add a hold
                # residual to the normalized VLA delta.
                adapted[base] = delta
                # Navigation may move the whole body but may never grasp. Lock
                # both grippers to the exact latches inherited from the current
                # episode, ignoring all predicted gripper values.
                for side in ("left", "right"):
                    gripper = ENV_ACTION_SEGMENTS[f"{side}_gripper"]
                    adapted[gripper] = float(self._gripper_latch[side])
                step_action = torch.as_tensor(
                    adapted,
                    dtype=step_action.dtype,
                    device=step_action.device,
                ).reshape(1, 23)
                nav_raw_base_actions.append(raw_base.astype(float).tolist())
                nav_position_deltas.append(delta.astype(float).tolist())
            if (
                getattr(self, "_control_mode", None) == HYBRID_VLM_PI0_MODE
                and local_gripper_monitor is not None
            ):
                # Hybrid runs the official position-base planner controller. Pi0.5
                # was trained with a velocity-base action, so its first three
                # outputs must never be reinterpreted as position targets. Build a
                # fresh position-controller hold command at every executed step.
                hold_action = np.asarray(
                    self._require_planner().backend.hold_action(), dtype=np.float32
                ).reshape(-1)
                if hold_action.shape != (23,) or not np.isfinite(hold_action).all():
                    raise RuntimeError(
                        "hybrid planner hold action must be one finite 23D vector"
                    )
                base = ENV_ACTION_SEGMENTS["base"]
                step_action = step_action.clone()
                step_action[:, base] = torch.as_tensor(
                    hold_action[base],
                    dtype=step_action.dtype,
                    device=step_action.device,
                )
            step_obs, step_reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    step_action,
                    need_obs=need_observation,
                )
            )
            self._env_steps += 1
            executed_steps += 1
            if pi0_nav_pick:
                executed_action = np.asarray(
                    step_action.detach().cpu().numpy(), dtype=np.float32
                ).reshape(23)
                for side in ("left", "right"):
                    gripper = ENV_ACTION_SEGMENTS[f"{side}_gripper"]
                    self._gripper_latch[side] = float(executed_action[gripper][0])
            if (
                getattr(self, "_control_mode", None) == HYBRID_VLM_PI0_MODE
                and local_gripper_monitor is not None
            ):
                # Planner commands after Pi0 must hold the exact most recently
                # executed gripper command for both hands. Update only after the
                # simulator step returns, so an unexecuted action is never latched.
                executed_action = np.asarray(
                    step_action.detach().cpu().numpy(), dtype=np.float32
                ).reshape(23)
                for side in ("left", "right"):
                    gripper = ENV_ACTION_SEGMENTS[f"{side}_gripper"]
                    self._gripper_latch[side] = float(executed_action[gripper][0])
            step_info = step_infos[0]
            official_info = step_info
            final_reward = step_reward[0]
            terminated = terminated or _scalar_bool(step_term) or _raw_done(step_info)
            truncated = truncated or _scalar_bool(step_trunc)
            self._last_info = _numpy_tree(step_info)
            if need_observation:
                if step_obs is None:
                    raise RuntimeError(
                        "BEHAVIOR requested observation but received None"
                    )
                final_observation = _single_observation(self._env._wrap_obs(step_obs))
                self._last_observation = final_observation
                self._record_rgbd_frames(step_obs, final_observation)
                if self._env_steps % observation_interval == 0:
                    self._append_video(final_observation)
            if (
                pi0_nav_pick
                and not _raw_success(step_info)
                and not terminated
                and not truncated
            ):
                if not need_observation or final_observation is None:
                    raise RuntimeError(
                        "pi0_nav_pick validation requires same-step observations"
                    )
                handoff_validator = self._update_handoff_validator(final_observation)
                if bool(handoff_validator["local_grasp_success"]):
                    held = handoff_validator.get("held_hand")
                    if held not in {"left", "right"}:
                        raise RuntimeError(
                            "strict handoff did not select exactly one hand"
                        )
                    remaining = int(self._meta["max_episode_steps"]) - int(
                        self._env_steps
                    )
                    if remaining < HANDOFF_VALIDATION_FRAMES:
                        handoff_stop_reason = "insufficient_handoff_horizon"
                        handoff_validator = {
                            **handoff_validator,
                            "handoff_failed": True,
                        }
                        self._handoff_state = _HANDOFF_FAILED
                        self._vla_actions_enabled = False
                    else:
                        try:
                            handoff_artifacts = self._complete_pi0_nav_pick_handoff(
                                held_hand=str(held), validator=handoff_validator
                            )
                        except Exception as exc:
                            logger.exception("pi0_nav_pick handoff failed closed")
                            handoff_stop_reason = (
                                f"handoff_failed:{type(exc).__name__}: {exc}"
                            )
                            failure = getattr(self, "_last_handoff_failure", None) or {}
                            handoff_artifacts = {
                                "handoff_env_steps": int(
                                    failure.get("handoff_env_steps", 0)
                                ),
                                "handoff_failure_diagnostics_path": failure.get(
                                    "handoff_failure_diagnostics_path"
                                ),
                            }
                            handoff_validator = {
                                **handoff_validator,
                                "handoff_failed": True,
                                "handoff_failure_diagnostics_path": failure.get(
                                    "handoff_failure_diagnostics_path"
                                ),
                            }
                            self._handoff_state = _HANDOFF_FAILED
                            self._vla_actions_enabled = False
                    break
            if local_gripper_monitor is not None:
                if not need_observation or final_observation is None:
                    raise RuntimeError(
                        "Pi0 gripper monitoring requires a same-step public observation"
                    )
                assert monitor_hand is not None
                assert monitor_threshold is not None
                assert monitor_required_closed_steps is not None
                monitor_opening = self._validated_selected_gripper_opening(
                    observation=final_observation,
                    hand=monitor_hand,
                )
                monitor_min_opening = min(monitor_min_opening, monitor_opening)
                if monitor_opening <= monitor_threshold:
                    monitor_closed_streak += 1
                else:
                    monitor_closed_streak = 0
                if (
                    monitor_candidate_env_step is None
                    and monitor_closed_streak >= monitor_required_closed_steps
                ):
                    monitor_candidate_env_step = self._env_steps
                monitor_result = {
                    "hand": monitor_hand,
                    "opening": monitor_opening,
                    "min_opening": monitor_min_opening,
                    "closed_streak": monitor_closed_streak,
                    "candidate": monitor_candidate_env_step is not None,
                    "candidate_env_step": monitor_candidate_env_step,
                    "executed_steps": executed_steps,
                }
                if (
                    monitor_stop_on_candidate
                    and monitor_candidate_env_step is not None
                    and not _raw_success(step_info)
                    and not terminated
                    and not truncated
                ):
                    # Synchronize the returned RGB/proprio with the stopped
                    # articulation without taking an additional physics step.
                    self._refresh_observation_without_step()
                    final_observation = self._last_observation
                    if final_observation is None:
                        raise RuntimeError(
                            "Pi0 candidate refresh produced no observation"
                        )
                    if self._env_steps % observation_interval != 0:
                        self._append_video(final_observation)
                    break
            if pi0_navigate_to and executed_steps % 4 == 0:
                backend = self._require_planner().backend
                try:
                    collision = backend.collision_report(force=True)
                except Exception as exc:
                    collision = {
                        "available": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                try:
                    dynamics = backend.dynamics_report()
                except Exception as exc:
                    dynamics = {
                        "available": False,
                        "ok": None,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                check = {
                    "executed_steps": executed_steps,
                    "total_env_steps": self._env_steps,
                    "collision": _wire_safe(collision),
                    "dynamics": _wire_safe(dynamics),
                }
                nav_checks.append(check)
                if not bool(collision.get("available", False)):
                    nav_stop_reason = "collision_feedback_unavailable"
                elif bool(collision.get("colliding", True)):
                    nav_stop_reason = "collision"
                elif not bool(dynamics.get("available", False)):
                    nav_stop_reason = "dynamics_feedback_unavailable"
                elif not bool(dynamics.get("ok", False)):
                    nav_stop_reason = "dynamics_violation"
                if nav_stop_reason is not None:
                    self._refresh_observation_without_step()
                    final_observation = self._last_observation
                    if final_observation is None:
                        raise RuntimeError(
                            "pi0_navigate_to safety stop produced no final observation"
                        )
                    self._append_video(final_observation)
                    break
            if _raw_success(step_info) or terminated or truncated:
                break

        if final_observation is None:
            raise RuntimeError("BEHAVIOR action chunk executed zero steps")
        task_success = _raw_success(official_info)
        self._done = bool(
            (task_success or terminated or truncated) and not post_success_cleanup
        )
        returned_info = _wire_safe(official_info)
        if not isinstance(returned_info, dict):
            returned_info = {"raw": returned_info}
        handoff_env_steps = int(handoff_artifacts.get("handoff_env_steps", 0))
        returned_info["_rpent"] = {"executed_steps": int(executed_steps)}
        if pi0_nav_pick:
            returned_info["_rpent"]["handoff_env_steps"] = handoff_env_steps
        if getattr(self, "_control_mode", None) in {
            HYBRID_VLM_PI0_MODE,
            PI0_NAV_PICK_VLA_MODE,
        }:
            returned_info["_rpent"]["total_env_steps"] = self._env_steps
        if monitor_result is not None:
            returned_info["_rpent"]["local_gripper_monitor"] = monitor_result
        if pi0_navigate_to:
            returned_info["_rpent"]["pi0_navigate_to_monitor"] = {
                "executed_steps": executed_steps,
                "safety_check_interval_steps": 4,
                "checks": nav_checks,
                "safety_stop": nav_stop_reason is not None,
                "stop_reason": nav_stop_reason,
                "raw_base_actions": nav_raw_base_actions,
                "position_deltas": nav_position_deltas,
                "input_clip_count": nav_input_clip_count,
                "delta_clip_count": nav_delta_clip_count,
                "delta_scale_per_second": [0.75, 0.75, 1.0],
                "controller_hz": 60,
                "per_axis_delta_limit": 0.01,
                "execution_mode": "adapted_23d_vla_receding_horizon",
                "predicted_action_dim": 23,
                "vla_passthrough_segments": [
                    "trunk",
                    "left_arm",
                    "right_arm",
                ],
                "adapted_segments": ["base"],
                "held_segments": ["left_gripper", "right_gripper"],
                "gripper_latches_unchanged": True,
                "grasp_allowed": False,
            }
        if pi0_nav_pick:
            local_success = bool(handoff_validator.get("local_grasp_success", False))
            if not local_success and task_success:
                self._handoff_state = _HANDOFF_OFFICIAL_SUCCESS
                self._vla_actions_enabled = False
            elif not local_success and (
                terminated
                or truncated
                or self._env_steps >= int(self._meta["max_episode_steps"])
                or handoff_stop_reason is not None
            ):
                self._handoff_state = _HANDOFF_FAILED
                self._vla_actions_enabled = False
            current = handoff_validator.get("current")
            current_per_hand = (
                current.get("per_hand") if isinstance(current, dict) else {}
            )
            returned_info["_rpent"]["pi0_nav_pick_monitor"] = {
                "executed_steps": int(executed_steps),
                "handoff_env_steps": handoff_env_steps,
                "total_env_steps": int(self._env_steps),
                "local_grasp_success": local_success,
                "held_hand": handoff_validator.get("held_hand"),
                "per_hand": _wire_safe(handoff_validator.get("per_hand", {})),
                "current_criteria": _wire_safe(
                    {
                        hand: value.get("criteria", {})
                        for hand, value in current_per_hand.items()
                        if isinstance(value, dict)
                    }
                ),
                "validator_trace_path": str(self._validator_trace_path),
                "state_checkpoint_path": (
                    handoff_artifacts.get("state_checkpoint_path")
                    if local_success
                    else None
                ),
                "handoff_state": self._handoff_state,
                "action_source": self._action_source,
                "vla_actions_enabled": bool(self._vla_actions_enabled),
                "paused_runtime_path": (
                    handoff_artifacts.get("paused_runtime_path")
                    if local_success
                    else None
                ),
                "strict_local_grasp_success": bool(
                    handoff_artifacts.get("strict_local_grasp_success", local_success)
                ),
                "usable_post_pick_saved": bool(
                    handoff_artifacts.get("usable_post_pick_saved", False)
                ),
                "save_policy": handoff_artifacts.get("save_policy"),
                "warnings": _wire_safe(handoff_artifacts.get("warnings", [])),
                "stop_reason": handoff_stop_reason,
                "handoff_failure_diagnostics_path": handoff_artifacts.get(
                    "handoff_failure_diagnostics_path"
                ),
            }
            if not local_success and (
                task_success
                or terminated
                or truncated
                or self._env_steps >= int(self._meta["max_episode_steps"])
            ):
                self._handoff_state = (
                    _HANDOFF_OFFICIAL_SUCCESS if task_success else _HANDOFF_FAILED
                )
                self._vla_actions_enabled = False
                self._finalize_video_segment()
                self._video_sealed = True
            elif not local_success and handoff_stop_reason is not None:
                self._finalize_video_segment()
                self._video_sealed = True
        result = _wire_safe(
            (
                final_observation,
                _numpy_tree(final_reward),
                terminated,
                truncated,
                returned_info,
            )
        )
        # Catch simulator-owned objects here so the transport can return a
        # useful RPC error instead of closing the socket without a frame.
        pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    def _persist_pi0_navigate_to_views(
        self,
        *,
        segment_index: int,
        chunk_index: int,
    ) -> dict[str, Any]:
        """Atomically persist one same-capture three-camera navigation audit."""

        root = (
            self._output_dir
            / "visual_review"
            / "pi0_navigate_to"
            / f"segment_{segment_index:04d}"
            / f"chunk_{chunk_index:04d}"
        )
        views: dict[str, Any] = {}
        capture_group_id: str | None = None
        for camera in ("head", "left_wrist", "right_wrist"):
            payload = self._require_planner().observe(camera)
            image = payload.get("_image_bytes")
            if not isinstance(image, bytes):
                raise RuntimeError(
                    f"pi0_navigate_to {camera} audit view is not PNG bytes"
                )
            group = payload.get("capture_group")
            group_id = str(group.get("id")) if isinstance(group, dict) else None
            if capture_group_id is None:
                capture_group_id = group_id
            elif group_id != capture_group_id:
                raise RuntimeError(
                    "pi0_navigate_to audit views are not from one capture group"
                )
            image_path = root / f"{camera}.png"
            _write_bytes_atomic(image_path, image)
            views[camera] = {
                "rgb_path": str(image_path),
                "frame_id": payload.get("frame_id"),
                "capture_group": group,
            }
        metadata = {
            "segment_index": segment_index,
            "chunk_index": chunk_index,
            "total_env_steps": int(self._env_steps),
            "capture_group_id": capture_group_id,
            "views": views,
        }
        metadata_path = root / "metadata.json"
        metadata["metadata_path"] = str(metadata_path)
        _write_json_atomic(metadata_path, metadata)
        return metadata

    def _physical_gripper_opening(self, hand: str) -> float:
        """Return the physical two-finger joint opening without a sensor join."""

        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot unavailable for Pi0 gripper monitoring")
        control_indices = getattr(robot, "gripper_control_idx", None)
        if not isinstance(control_indices, dict) or hand not in control_indices:
            raise RuntimeError(f"R1Pro {hand} gripper control indices are unavailable")
        indices = np.asarray(
            _numpy_tree(control_indices[hand]), dtype=np.int64
        ).reshape(-1)
        if indices.size != 2 or np.any(indices < 0):
            raise RuntimeError(
                f"R1Pro {hand} gripper must contain exactly two joint indices"
            )
        qpos = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if int(indices.max()) >= qpos.size:
            raise RuntimeError(f"R1Pro {hand} gripper joint index is out of bounds")
        actual_opening = float(qpos[indices].sum())
        if not np.isfinite(actual_opening):
            raise RuntimeError(f"R1Pro {hand} physical gripper opening is not finite")
        return actual_opening

    def _validated_selected_gripper_opening(
        self,
        *,
        observation: dict[str, Any],
        hand: str,
    ) -> float:
        """Return physical opening after matching same-step public proprio."""

        actual_opening = self._physical_gripper_opening(hand)

        compact = extract_policy_state(observation["states"])
        public_values = compact[POLICY_STATE_SEGMENTS[f"{hand}_gripper"]]
        if public_values.shape != (1,) or not np.isfinite(public_values[0]):
            raise RuntimeError(
                f"R1Pro {hand} public proprio gripper opening is invalid"
            )
        public_opening = float(public_values[0])
        if not np.isclose(
            actual_opening,
            public_opening,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                "same-step physical/public gripper opening mismatch: "
                f"hand={hand!r}, physical={actual_opening:.9g}, "
                f"public={public_opening:.9g}"
            )
        return actual_opening

    def get_env_meta(self) -> dict[str, Any]:
        return {
            **self._meta,
            "control_mode": self._control_mode,
            "restored_state": self._restored_state,
        }

    def _info_with_accounting(self) -> dict[str, Any]:
        info = _wire_safe(self._last_info)
        if not isinstance(info, dict):
            info = {"raw": info}
        rpent = info.get("_rpent")
        if not isinstance(rpent, dict):
            rpent = {}
            info["_rpent"] = rpent
        rpent["total_env_steps"] = int(self._env_steps)
        return info

    def current_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return current RGB/proprio without a reset or a physics action."""

        if self._last_observation is None:
            raise RuntimeError("current observation is unavailable before reset")
        self._refresh_observation_without_step()
        observation = self._last_observation
        if observation is None:
            raise RuntimeError("current observation refresh produced no observation")
        self._append_video(observation)
        result = _wire_safe((observation, self._info_with_accounting()))
        pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    def _planner_result_with_accounting(self, result: dict[str, Any]) -> dict[str, Any]:
        public = dict(result)
        if (
            self._control_mode == HYBRID_VLM_PI0_MODE
            and public.get("suggested_next_tool") == "navigate_to"
        ):
            public["suggested_next_tool"] = "pi0_navigate_to"
        public["total_env_steps"] = int(self._env_steps)
        return public

    def _persist_live_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Lightweight unit facades built with ``__new__`` predate live review.
        # Production construction always supplies the output directory.
        if not hasattr(self, "_output_dir"):
            return payload
        image = payload.get("_image_bytes")
        if not isinstance(image, bytes):
            raise RuntimeError("observe payload did not contain PNG bytes")
        self._live_observation_counter = (
            int(getattr(self, "_live_observation_counter", 0)) + 1
        )
        camera = canonical_camera(str(payload.get("camera", "head")))
        stem = f"{self._live_observation_counter:06d}_{camera}"
        live_dir = self._output_dir / "visual_review" / "live"
        image_path = live_dir / f"{stem}.png"
        metadata_path = live_dir / f"{stem}.json"
        metadata = {
            key: value for key, value in payload.items() if not str(key).startswith("_")
        }
        metadata.update(
            {
                "rgb_path": str(image_path),
                "metadata_path": str(metadata_path),
                "total_env_steps": int(self._env_steps),
            }
        )
        _write_bytes_atomic(image_path, image)
        _write_json_atomic(metadata_path, metadata)
        public = dict(payload)
        public["visual_review"] = {
            "rgb_path": str(image_path),
            "metadata_path": str(metadata_path),
            "live_dir": str(live_dir),
        }
        return public

    def _refresh_observation_without_step(self) -> None:
        """Capture current synchronized sensors without advancing simulation time.

        Planning can legitimately take longer than the RGB-D cache TTL while no
        controller waypoint is executed.  A later ``observe`` must therefore
        obtain a new capture (and new frame ids) at the same simulator step,
        rather than renewing or returning the expired capture.
        """

        omni_env = self._env.omnigibson_env
        raw_observation, _sensor_info = omni_env.get_obs()
        observation = _single_observation(self._env._wrap_obs([raw_observation]))
        self._last_observation = observation
        self._record_rgbd_frames([raw_observation], observation)

    def _resolve_post_pick_camera(self, camera: str) -> str:
        """Resolve dynamic wrist roles without exposing literal hands publicly."""

        requested = str(camera)
        if requested not in {"held_wrist", "press_wrist"}:
            return canonical_camera(requested)
        context = self._prepress_context
        if not isinstance(context, dict):
            raise RuntimeError("dynamic wrist cameras require inspect_post_pick_state")
        role = "held_hand" if requested == "held_wrist" else "press_hand"
        return canonical_camera(f"{context[role]}_wrist")

    def observe(self, camera: str) -> dict[str, Any]:
        requested_camera = str(camera)
        resolved_camera = self._resolve_post_pick_camera(requested_camera)
        try:
            frame = self._frame_cache.latest(resolved_camera)
            self._frame_cache.get_current(resolved_camera, frame.frame_id)
            # Planner warmup can consume nearly the entire cache TTL without
            # advancing simulation.  Do not hand a VLM a frame that is valid
            # at observe() time but likely to expire before pixel_to_world().
            if time.monotonic() - frame.timestamp_s > 5.0:
                self._refresh_observation_without_step()
        except CameraGeometryError:
            self._refresh_observation_without_step()
            refreshed = self._frame_cache.latest(resolved_camera)
            self._frame_cache.get_current(resolved_camera, refreshed.frame_id)
        payload = self._require_planner().observe(resolved_camera)
        payload = self._persist_live_observation(payload)
        payload["camera"] = requested_camera
        payload["resolved_camera"] = resolved_camera
        certificate = getattr(self, "_prepress_plan_certificate", None)
        if isinstance(certificate, dict):
            gate_binding = certificate.get("gate_binding")
            capture_group = payload.get("capture_group") or {}
            if (
                isinstance(gate_binding, dict)
                and gate_binding.get("capture_group_id") is not None
                and gate_binding.get("capture_group_id") != capture_group.get("id")
            ):
                self._prepress_plan_certificate = None
        motion = getattr(self, "_prepress_motion", None)
        if (
            isinstance(motion, dict)
            and motion.get("primitive_success") is True
            and motion.get("end_env_step") == self._env_steps
        ):
            frame = self._frame_cache.get_current(
                resolved_camera, str(payload.get("frame_id"))
            )
            if int(frame.step_index) != int(self._env_steps):
                raise RuntimeError("post-motion public review frame is stale")
            reviews = motion.setdefault("public_visual_review", {})
            reviews[resolved_camera] = {
                "camera": requested_camera,
                "resolved_camera": resolved_camera,
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "env_step": int(self._env_steps),
                "image_path": (payload.get("visual_review") or {}).get("rgb_path"),
            }
            context = self._prepress_context or {}
            required = {
                "head",
                f"{context.get('held_hand')}_wrist",
                f"{context.get('press_hand')}_wrist",
            }
            selected = [reviews.get(name) for name in required]
            capture_group_ids = {
                item.get("capture_group_id")
                for item in selected
                if isinstance(item, dict)
            }
            motion["three_view_observed_by_vlm"] = bool(
                all(isinstance(item, dict) for item in selected)
                and all(item.get("env_step") == self._env_steps for item in selected)
                and None not in capture_group_ids
                and len(capture_group_ids) == 1
            )
            self._persist_prepress_motion_record(motion)
        return self._planner_result_with_accounting(payload)

    def _pixel_to_world_raw(
        self,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        result = self._require_planner().pixel_to_world(
            camera=camera,
            frame_id=frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
            output_frame=output_frame,
        )
        return self._planner_result_with_accounting(result)

    def _record_button_projection(
        self,
        *,
        gate: dict[str, Any],
        projected: dict[str, Any],
    ) -> dict[str, Any]:
        if gate.get("face_class") != BUTTON_FACE_CLASS:
            raise RuntimeError(
                "only a BUTTON_FACE gate may authorize button projection"
            )
        if not projected.get("primitive_success"):
            return projected
        diagnostics = projected.get("diagnostics", {})
        xyz, normal = diagnostics.get("xyz"), diagnostics.get("surface_normal")
        if xyz is None or normal is None:
            raise RuntimeError("button projection omitted xyz or surface normal")
        gate_id = str(gate["gate_id"])
        projection = {
            "projection_id": "button_projection_"
            + hashlib.sha256(f"{gate_id}:{xyz}:{normal}".encode()).hexdigest()[:20],
            "gate_id": gate_id,
            "camera": gate["camera"],
            "resolved_camera": gate["resolved_camera"],
            "frame_id": gate["frame_id"],
            "capture_group_id": gate["capture_group_id"],
            "env_step": int(self._env_steps),
            "button_center_world": xyz,
            "button_normal_world": normal,
            "projection_metrics": projected.get("metrics", {}),
        }
        self._prepress_projection = projection
        self._prepress_geometry = None
        return {
            **projected,
            "stop_reason": "button_projected",
            **_wire_safe(projection),
            "total_env_steps": int(self._env_steps),
        }

    def pixel_to_world(
        self,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        requested_camera = str(camera)
        resolved_camera = self._resolve_post_pick_camera(requested_camera)
        projected = self._pixel_to_world_raw(
            camera=resolved_camera,
            frame_id=frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
            output_frame=output_frame,
        )
        projected["camera"] = requested_camera
        projected["resolved_camera"] = resolved_camera
        gate = self._prepress_gate
        if (
            isinstance(gate, dict)
            and gate.get("button_visible") is True
            and gate.get("face_class") == BUTTON_FACE_CLASS
            and gate.get("env_step") == self._env_steps
            and gate.get("camera") == requested_camera
            and gate.get("resolved_camera") == resolved_camera
            and gate.get("frame_id") == str(frame_id)
            and output_frame == "world"
        ):
            center = np.asarray(gate.get("center_uv"), dtype=np.float64).reshape(2)
            requested = np.asarray([u, v], dtype=np.float64).reshape(2)
            if np.max(np.abs(center - requested)) <= 0.5:
                return self._record_button_projection(
                    gate=gate,
                    projected=projected,
                )
        return projected

    def _read_post_pick_checkpoint(
        self,
    ) -> tuple[Path, dict[str, Any], str]:
        path = self._output_dir / "state_checkpoints" / "state_checkpoint_1.json"
        root = path.parent.resolve()
        if path.is_symlink() or path.resolve().parent != root or not path.is_file():
            raise RuntimeError("this run's state_checkpoint_1.json is unavailable")
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("kind") != "robot_motion_checkpoint"
            or payload.get("not_simulator_restore") is not True
            or payload.get("checkpoint_name") != "state_checkpoint_1"
        ):
            raise RuntimeError("state_checkpoint_1 robot-motion schema is invalid")
        held, press = self._validated_handoff_hands(
            held_hand=str(payload.get("held_hand")),
            press_hand=str(payload.get("press_hand")),
        )
        if not str(payload.get("object_name", "")):
            raise RuntimeError("state_checkpoint_1 object binding is missing")
        payload["held_hand"], payload["press_hand"] = held, press
        return path, payload, hashlib.sha256(raw).hexdigest()

    def _load_post_pick_checkpoint(self) -> tuple[Path, dict[str, Any]]:
        path, payload, _sha256 = self._read_post_pick_checkpoint()
        return path, payload

    def _assert_prepress_checkpoint_bound(self) -> tuple[Path, dict[str, Any]]:
        context = self._prepress_context
        if not isinstance(context, dict):
            raise RuntimeError("inspect_post_pick_state must run first")
        path, checkpoint, sha256 = self._read_post_pick_checkpoint()
        if str(path) != context.get("checkpoint_path") or sha256 != context.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError("bound state_checkpoint_1 changed during pre-press")
        return path, checkpoint

    def _prepress_stability_snapshot(self) -> dict[str, Any]:
        context = self._prepress_context
        if not isinstance(context, dict):
            raise RuntimeError("inspect_post_pick_state must run first")
        held, press = context["held_hand"], context["press_hand"]
        radio, table = self._resolve_handoff_targets()
        robot = self._robot()
        if robot is None or self._last_observation is None:
            raise RuntimeError("current robot observation is unavailable")
        link = robot.eef_links[held]
        press_link = robot.eef_links[press]
        eef_pos, eef_quat = self._object_pose(link)
        press_eef_pos, press_eef_quat = self._object_pose(press_link)
        radio_pos, radio_quat = self._object_pose(radio)
        rel_pos, rel_quat = self._relative_pose(
            eef_pos, eef_quat, radio_pos, radio_quat
        )
        expected = context["checkpoint"]["poses"]["object_pose_in_held_eef"]
        expected_pos = np.asarray(expected["position"], dtype=np.float64)
        expected_quat = np.asarray(expected["quat_xyzw"], dtype=np.float64)
        radio_bottom, _ = self._object_vertical_bounds(radio)
        _, table_top = self._object_vertical_bounds(table)
        opening = self._validated_selected_gripper_opening(
            observation=self._last_observation, hand=held
        )
        held_attachment = self._attachment_matches(held, radio)
        press_attachment = self._attachment_matches(press, radio)
        held_contact = self._hand_target_contact_report(held, radio_pos)
        press_contact = self._hand_target_contact_report(press, radio_pos)
        result = {
            "relative_position_m": rel_pos.tolist(),
            "relative_quat_xyzw": rel_quat.tolist(),
            "relative_drift_m": float(np.linalg.norm(rel_pos - expected_pos)),
            "angular_drift_rad": _quaternion_angle_rad(rel_quat, expected_quat),
            "radio_position_world": radio_pos.tolist(),
            "radio_quat_xyzw": radio_quat.tolist(),
            "held_eef_position_world": eef_pos.tolist(),
            "held_eef_quat_xyzw": eef_quat.tolist(),
            "press_eef_position_world": press_eef_pos.tolist(),
            "press_eef_quat_xyzw": press_eef_quat.tolist(),
            "radio_height_m": float(radio_pos[2]),
            "air_gap_m": float(radio_bottom - table_top),
            "held_gripper_opening_m": float(opening),
            "held_attachment": list(held_attachment),
            "held_contact_count": int(held_contact["target_contact_count"]),
            "held_two_finger_contact": bool(held_contact["target_two_finger_contact"]),
            "press_attachment": list(press_attachment),
            "press_contact_count": int(press_contact["target_contact_count"]),
        }
        result["stable"] = bool(
            result["relative_drift_m"] <= HANDOFF_RELATIVE_DRIFT_MAX_M
            and result["angular_drift_rad"] <= HANDOFF_RELATIVE_ANGULAR_DRIFT_MAX_RAD
            and result["air_gap_m"] >= HANDOFF_SUPPORT_GAP_MIN_M
            and opening < HANDOFF_GRIPPER_OPENING_MAX
            and (any(held_attachment) or held_contact["target_two_finger_contact"])
            and not any(press_attachment)
            and press_contact["target_contact_count"] == 0
        )
        return result

    @staticmethod
    def _prepress_search_rotations(
        _stability: dict[str, Any],
    ) -> list[dict[str, Any]]:
        angle = float(np.deg2rad(15.0))
        candidates = []
        for axis_name, axis in (
            ("local_x", np.array([1.0, 0.0, 0.0])),
            ("local_y", np.array([0.0, 1.0, 0.0])),
            ("local_z", np.array([0.0, 0.0, 1.0])),
        ):
            for sign in (-1.0, 1.0):
                candidates.append(
                    {
                        "label": (
                            f"{axis_name}_{'plus' if sign > 0 else 'minus'}_15deg"
                        ),
                        "tool": "rotate_wrist",
                        "role": "held",
                        "relative_axis_angle": [
                            *axis.tolist(),
                            sign * angle,
                        ],
                        "frame": "eef",
                        "plan_only": True,
                    }
                )
        return candidates

    def inspect_post_pick_state(
        self, checkpoint_name: str = "state_checkpoint_1"
    ) -> dict[str, Any]:
        if checkpoint_name != "state_checkpoint_1":
            raise ValueError("pre-press must bind state_checkpoint_1")
        if (
            self._handoff_state != _HANDOFF_PAUSED
            or self._action_source != "curobo"
            or self._vla_actions_enabled is not False
            or not self._paused_runtime_path.is_file()
        ):
            raise RuntimeError("pre-press requires a formally finalized PAUSED handoff")
        paused_runtime = json.loads(
            self._paused_runtime_path.read_text(encoding="utf-8")
        )
        if (
            not isinstance(paused_runtime, dict)
            or paused_runtime.get("handoff_state") != _HANDOFF_PAUSED
            or paused_runtime.get("action_source") != "curobo"
            or paused_runtime.get("vla_actions_enabled") is not False
            or paused_runtime.get("vla_action_gate_confirmed") is not True
            or paused_runtime.get("lifecycle_finalized") is not True
        ):
            raise RuntimeError("paused_runtime.json is not formally finalized")
        path, checkpoint, checkpoint_sha256 = self._read_post_pick_checkpoint()
        self._refresh_observation_without_step()
        self._prepress_context = {
            "checkpoint_path": str(path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint": checkpoint,
            "held_hand": checkpoint["held_hand"],
            "press_hand": checkpoint["press_hand"],
            "object_name": checkpoint["object_name"],
            "bound_env_step": int(self._env_steps),
        }
        self._prepress_gate = self._prepress_projection = None
        self._prepress_geometry = self._prepress_motion = None
        self._prepress_plan_certificate = None
        stability = self._prepress_stability_snapshot()
        transform_context = {
            "convention": (
                "T_A_B maps coordinates from frame B into frame A; matrices "
                "are row-major homogeneous 4x4 transforms"
            ),
            "T_world_held_current": pose_matrix_xyzw(
                stability["held_eef_position_world"],
                stability["held_eef_quat_xyzw"],
            ).tolist(),
            "T_world_press_current": pose_matrix_xyzw(
                stability["press_eef_position_world"],
                stability["press_eef_quat_xyzw"],
            ).tolist(),
            "T_world_radio_current": pose_matrix_xyzw(
                stability["radio_position_world"],
                stability["radio_quat_xyzw"],
            ).tolist(),
            "T_held_radio_current": pose_matrix_xyzw(
                stability["relative_position_m"],
                stability["relative_quat_xyzw"],
            ).tolist(),
            "candidate_generation": (
                "The runtime applies T_held_radio_current internally to derive "
                "held-EEF candidates from button/radio geometry goals; callers "
                "must not submit a literal held EEF pose."
            ),
            "radio_pose_prior": {
                "scope": "turning_on_radio radio object only",
                "local_button_center_m": list(RADIO_LOCAL_BUTTON_CENTER_M),
                "local_button_face_outward_normal": list(
                    RADIO_LOCAL_BUTTON_FACE_NORMAL
                ),
                "local_back_face_outward_normal": [
                    -value for value in RADIO_LOCAL_BUTTON_FACE_NORMAL
                ],
                "local_upright_axis": list(RADIO_LOCAL_UP_AXIS),
                "use": (
                    "coarse back-to-front reveal only; a fresh positive "
                    "button gate must refine and validate the final pose"
                ),
            },
        }
        return {
            "_finish": False,
            "primitive_success": bool(stability["stable"]),
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "post_pick_state_inspected"
            if stability["stable"]
            else "post_pick_state_unstable",
            "checkpoint_path": str(path),
            "checkpoint_sha256": self._prepress_context["checkpoint_sha256"],
            "held_hand": checkpoint["held_hand"],
            "press_hand": checkpoint["press_hand"],
            "object_name": checkpoint["object_name"],
            "press_approach_axis_local": [0.0, 0.0, 1.0],
            "search_rotate_wrist_candidates": self._prepress_search_rotations(
                stability
            ),
            "handoff_state": self._handoff_state,
            "action_source": self._action_source,
            "vla_actions_enabled": bool(self._vla_actions_enabled),
            "paused_runtime_path": str(self._paused_runtime_path),
            "stability": _wire_safe(stability),
            "transform_context": _wire_safe(transform_context),
            "total_env_steps": int(self._env_steps),
        }

    def declare_button_visibility(
        self,
        camera: str,
        frame_id: str,
        button_visible: bool,
        positive_signature: dict[str, bool] | None = None,
        negative_case: str | None = None,
        bbox_xyxy: Any | None = None,
        center_uv: Any | None = None,
    ) -> dict[str, Any]:
        self._assert_prepress_checkpoint_bound()
        requested_camera = str(camera)
        resolved_camera = self._resolve_post_pick_camera(requested_camera)
        frame = self._frame_cache.get_current(resolved_camera, str(frame_id))
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError(
                "button declaration frame is not from the current env step"
            )
        declaration = validate_button_declaration(
            button_visible=bool(button_visible),
            positive_signature=positive_signature,
            negative_case=negative_case,
            bbox_xyxy=bbox_xyxy,
            center_uv=center_uv,
            image_width=frame.intrinsics.width,
            image_height=frame.intrinsics.height,
        )
        gate = {
            **declaration,
            "camera": requested_camera,
            "resolved_camera": resolved_camera,
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "capture_step": int(frame.step_index),
            "env_step": int(self._env_steps),
        }
        if declaration["face_class"] == BUTTON_FACE_CLASS:
            gate["gate_id"] = gate_token(gate)
        gate["motion_policy"] = {
            "direct_back_to_front": bool(
                declaration["face_class"] == CLEAR_SLOTTED_BACK_FACE_CLASS
            ),
            "button_visible_micro_adjust": bool(
                declaration["face_class"] == BUTTON_FACE_CLASS
            ),
            "other_views_bounded_search_only": bool(
                declaration["face_class"]
                not in {BUTTON_FACE_CLASS, CLEAR_SLOTTED_BACK_FACE_CLASS}
            ),
        }
        self._prepress_gate = gate
        self._prepress_projection = self._prepress_geometry = None
        self._prepress_plan_certificate = None
        return {
            "_finish": False,
            "primitive_success": True,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "button_visible"
            if gate["button_visible"]
            else "button_not_visible",
            **_wire_safe(gate),
            "total_env_steps": int(self._env_steps),
        }

    def project_button(self, gate_id: str, depth_window_px: int = 7) -> dict[str, Any]:
        self._assert_prepress_checkpoint_bound()
        gate = self._prepress_gate
        if (
            not isinstance(gate, dict)
            or not gate.get("button_visible")
            or gate.get("face_class") != BUTTON_FACE_CLASS
        ):
            raise RuntimeError("a current VISIBLE button gate is required")
        if gate.get("gate_id") != gate_id or gate.get("env_step") != self._env_steps:
            raise RuntimeError("button gate is stale or does not match gate_id")
        frame = self._frame_cache.get_current(gate["resolved_camera"], gate["frame_id"])
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError(
                "button projection frame is not from the current env step"
            )
        if frame.capture_group_id != gate.get("capture_group_id"):
            raise RuntimeError("button gate capture group changed")
        u, v = gate["center_uv"]
        projected = self._pixel_to_world_raw(
            gate["resolved_camera"],
            gate["frame_id"],
            u,
            v,
            depth_window_px,
            "world",
        )
        projected["camera"] = gate["camera"]
        projected["resolved_camera"] = gate["resolved_camera"]
        return self._record_button_projection(gate=gate, projected=projected)

    def evaluate_prepress_geometry(
        self,
        projection_id: str,
        max_line_distance_m: float = 0.010,
        max_opposition_angle_deg: float = 15.0,
        min_axial_standoff_m: float = 0.03,
        max_axial_standoff_m: float = 0.06,
    ) -> dict[str, Any]:
        self._assert_prepress_checkpoint_bound()
        if (
            float(max_line_distance_m) > PREPRESS_LINE_DISTANCE_MAX_M
            or float(max_opposition_angle_deg) > PREPRESS_OPPOSITION_ANGLE_MAX_DEG
            or float(min_axial_standoff_m) < PREPRESS_AXIAL_STANDOFF_MIN_M
            or float(max_axial_standoff_m) > PREPRESS_AXIAL_STANDOFF_MAX_M
        ):
            raise ValueError("pre-press geometry thresholds may only be tightened")
        projection = getattr(self, "_prepress_projection", None)
        context = self._prepress_context
        if not isinstance(projection, dict) or not isinstance(context, dict):
            raise RuntimeError("a current button projection is required")
        if (
            projection.get("projection_id") != projection_id
            or projection.get("env_step") != self._env_steps
        ):
            raise RuntimeError(
                "button projection is stale or does not match projection_id"
            )
        pose = self._require_planner().backend.get_eef_pose(context["press_hand"])
        if pose is None:
            raise RuntimeError("press-hand EEF pose is unavailable")
        press_position, press_quat = pose
        press_direction = quat_rotate_xyzw(press_quat, [0.0, 0.0, 1.0])
        geometry = evaluate_geometry(
            button_center_world=projection["button_center_world"],
            button_normal_world=projection["button_normal_world"],
            press_eef_position_world=press_position,
            press_direction_world=press_direction,
            max_line_distance_m=max_line_distance_m,
            max_opposition_angle_deg=max_opposition_angle_deg,
            min_axial_standoff_m=min_axial_standoff_m,
            max_axial_standoff_m=max_axial_standoff_m,
        )
        geometry.update(
            {
                "projection_id": projection_id,
                "env_step": int(self._env_steps),
                "press_hand": context["press_hand"],
                "press_eef_pose_world": {
                    "position": np.asarray(press_position).tolist(),
                    "quat_xyzw": np.asarray(press_quat).tolist(),
                },
            }
        )
        stability = self._prepress_stability_snapshot()
        geometry["radio_held_stability"] = stability
        geometry["geometry_pass"] = bool(
            geometry["geometry_pass"] and stability["stable"]
        )
        motion = self._prepress_motion
        if (
            geometry["geometry_pass"]
            and isinstance(motion, dict)
            and motion.get("end_env_step") == self._env_steps
            and motion.get("three_view_observed_by_vlm") is True
        ):
            motion["visual_review_verdict"] = (
                "passed_by_post_motion_button_gate_and_geometry"
            )
            motion["visual_review_pass"] = True
            self._persist_prepress_motion_record(motion)
        self._prepress_geometry = geometry
        return {
            "_finish": False,
            "primitive_success": bool(geometry["geometry_pass"]),
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "prepress_geometry_aligned"
            if geometry["geometry_pass"]
            else "prepress_geometry_not_aligned",
            "geometry": _wire_safe(geometry),
            "total_env_steps": int(self._env_steps),
        }

    def _capture_prepress_views(
        self, *, round_index: int
    ) -> tuple[dict[str, str], str | None]:
        context = self._prepress_context or {}
        review_dir = (
            self._output_dir / "visual_review" / "prepress" / f"round_{round_index:03d}"
        )
        paths: dict[str, str] = {}
        groups: set[str] = set()
        for label, camera in (
            ("head", "head"),
            ("held_wrist", f"{context.get('held_hand')}_wrist"),
            ("press_wrist", f"{context.get('press_hand')}_wrist"),
        ):
            observed = self.observe(camera)
            image = observed.get("_image_bytes")
            if not isinstance(image, bytes):
                raise RuntimeError(f"pre-press review missing {camera} PNG")
            group = (observed.get("capture_group") or {}).get("id")
            frame_id = observed.get("frame_id")
            frame = self._frame_cache.get_current(camera, str(frame_id))
            if int(frame.step_index) != int(self._env_steps):
                raise RuntimeError(
                    f"pre-press review {camera} is not from the current env step"
                )
            if not isinstance(group, str) or not group:
                raise RuntimeError(f"pre-press review {camera} omitted capture group")
            groups.add(group)
            target = review_dir / f"{label}.png"
            _write_bytes_atomic(target, image)
            paths[label] = str(target)
        if len(groups) != 1:
            raise RuntimeError(
                "pre-press three-view review is not capture synchronized"
            )
        return paths, next(iter(groups))

    @staticmethod
    def _persist_prepress_motion_record(motion: dict[str, Any]) -> None:
        trace_path = Path(str(motion.get("trace_path", "")))
        if not trace_path.is_file():
            return
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload["motion"] = _wire_safe(motion)
        _write_json_atomic(trace_path, payload)

    def _prepress_gate_binding(
        self, *, allow_expired: bool = False
    ) -> dict[str, Any] | None:
        gate = self._prepress_gate
        if not isinstance(gate, dict) or gate.get("env_step") != self._env_steps:
            return None
        binding = {
            key: gate.get(key)
            for key in (
                "face_class",
                "button_visible",
                "frame_id",
                "capture_group_id",
                "camera",
                "resolved_camera",
                "env_step",
                "gate_id",
            )
        }
        resolved_camera = binding.get("resolved_camera")
        frame_id = binding.get("frame_id")
        capture_group_id = binding.get("capture_group_id")
        if not (resolved_camera and frame_id and capture_group_id):
            return None
        try:
            current = (
                self._frame_cache.latest(str(resolved_camera))
                if allow_expired
                else self._frame_cache.get_current(str(resolved_camera), str(frame_id))
            )
        except CameraGeometryError:
            return None
        if (
            int(current.step_index) != int(self._env_steps)
            or current.frame_id != frame_id
            or current.capture_group_id != capture_group_id
        ):
            return None
        projection = self._prepress_projection
        if (
            isinstance(projection, dict)
            and projection.get("env_step") == self._env_steps
            and projection.get("gate_id") == gate.get("gate_id")
            and projection.get("frame_id") == gate.get("frame_id")
            and projection.get("capture_group_id") == capture_group_id
        ):
            binding["projection_id"] = projection.get("projection_id")
        return binding

    def _prepress_plan_signature(
        self,
        *,
        role: str,
        target: np.ndarray,
        quaternion: np.ndarray,
        gate_binding: dict[str, Any] | None,
        goal_binding: dict[str, Any] | None = None,
    ) -> str:
        context = self._prepress_context or {}
        payload = {
            "checkpoint_sha256": context.get("checkpoint_sha256"),
            "env_step": int(self._env_steps),
            "role": role,
            "target_xyz": np.asarray(target, dtype=np.float64).tolist(),
            "target_quat_xyzw": np.asarray(quaternion, dtype=np.float64).tolist(),
            "gate_binding": gate_binding,
            "goal_binding": goal_binding,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _prepress_move_to_candidate(
        self,
        target_xyz: Any,
        target_quat_xyzw: Any,
        plan_only: bool = False,
        timeout_s: float = 90.0,
        role: str = "held",
        *,
        goal_binding: dict[str, Any] | None = None,
        candidate_metadata: dict[str, Any] | None = None,
        press_observation_rotation: bool = False,
    ) -> dict[str, Any]:
        """Plan one resolved EEF candidate while closing the held gripper."""

        self._assert_prepress_checkpoint_bound()
        context = self._prepress_context
        assert isinstance(context, dict)
        if role not in {"held", "press"}:
            raise ValueError("pre-press role must be 'held' or 'press'")
        if press_observation_rotation and role != "press":
            raise ValueError(
                "press_observation_rotation is valid only for role='press'"
            )
        if press_observation_rotation and goal_binding is not None:
            raise ValueError(
                "press observation rotation cannot carry a button goal binding"
            )
        if self._base_controller_mode != "position":
            raise RuntimeError("pre-press motion requires position-base controllers")
        target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        quat = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
        if (
            not np.isfinite(target).all()
            or not np.isfinite(quat).all()
            or np.linalg.norm(quat) <= 1e-9
        ):
            raise ValueError("pre-press target pose is invalid")
        quat /= np.linalg.norm(quat)
        preflight_stability = self._prepress_stability_snapshot()
        if not preflight_stability["stable"]:
            raise RuntimeError("radio is not stably held before pre-press move_to")
        held, press = context["held_hand"], context["press_hand"]
        active = held if role == "held" else press
        locked = press if role == "held" else held
        gate = self._prepress_gate
        face_class = (
            gate.get("face_class")
            if isinstance(gate, dict) and gate.get("env_step") == self._env_steps
            else None
        )
        gate_binding = self._prepress_gate_binding(
            allow_expired=goal_binding is not None
        )
        if press_observation_rotation:
            # Acquiring a press-wrist view must not depend on the very button
            # gate/projection that the observation rotation is intended to
            # make possible.  The motion remains bound to the exact role,
            # target pose, checkpoint, env step, and certified trajectory.
            face_class = None
            gate_binding = None
        if face_class is not None and gate_binding is None:
            return {
                "_finish": False,
                "primitive_success": False,
                "task_success": bool(_raw_success(self._last_info)),
                "official_success_source": 'info["done"]["success"]',
                "stop_reason": "visual_gate_stale_before_motion",
                "held_hand": held,
                "press_hand": press,
                "active_role": role,
                "active_hand": active,
                "total_env_steps": int(self._env_steps),
            }
        if role == "press" and not press_observation_rotation:
            projection = self._prepress_projection
            if not (
                face_class == BUTTON_FACE_CLASS
                and isinstance(gate_binding, dict)
                and isinstance(projection, dict)
                and projection.get("env_step") == self._env_steps
                and projection.get("gate_id") == gate.get("gate_id")
                and projection.get("frame_id") == gate.get("frame_id")
                and projection.get("capture_group_id") == gate.get("capture_group_id")
            ):
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": "press_staging_requires_fresh_button_projection",
                    "held_hand": held,
                    "press_hand": press,
                    "active_role": role,
                    "active_hand": active,
                    "total_env_steps": int(self._env_steps),
                }
        direct_alignment = None
        if role == "held" and face_class == CLEAR_SLOTTED_BACK_FACE_CLASS:
            if bool(getattr(self, "_prepress_coarse_flip_used", False)):
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": "coarse_back_to_front_flip_already_used",
                    "held_hand": held,
                    "press_hand": press,
                    "active_role": role,
                    "active_hand": active,
                    "total_env_steps": int(self._env_steps),
                }
            world_held = pose_matrix_xyzw(
                preflight_stability["held_eef_position_world"],
                preflight_stability["held_eef_quat_xyzw"],
            )
            world_radio = pose_matrix_xyzw(
                preflight_stability["radio_position_world"],
                preflight_stability["radio_quat_xyzw"],
            )
            held_to_radio = np.linalg.inv(world_held) @ world_radio
            direct_alignment = direct_back_to_front_alignment(
                target_held_pose=pose_matrix_xyzw(target, quat),
                held_to_radio_transform=held_to_radio,
                press_eef_position_world=preflight_stability[
                    "press_eef_position_world"
                ],
            )
        if goal_binding is None:
            authorization = authorize_prepress_motion(
                role=role,
                current_xyz=preflight_stability[f"{role}_eef_position_world"],
                current_quat_xyzw=preflight_stability[f"{role}_eef_quat_xyzw"],
                target_xyz=target,
                target_quat_xyzw=quat,
                face_class=face_class,
                direct_back_alignment=direct_alignment,
                press_observation_rotation=press_observation_rotation,
            )
        else:
            current_xyz = np.asarray(
                preflight_stability[f"{role}_eef_position_world"],
                dtype=np.float64,
            )
            authorization = {
                "allowed": True,
                "policy": "button_goal_candidate",
                "role": role,
                "face_class": face_class or "BUTTON_GOAL",
                "translation_m": float(np.linalg.norm(target - current_xyz)),
                "rotation_rad": _quaternion_angle_rad(
                    preflight_stability[f"{role}_eef_quat_xyzw"], quat
                ),
                "z_increase_m": float(target[2] - current_xyz[2]),
                "requires_plan_only_first": True,
                "goal_binding": _wire_safe(goal_binding),
                "candidate_id": (candidate_metadata or {}).get("candidate_id"),
            }
        if not authorization["allowed"]:
            return {
                "_finish": False,
                "primitive_success": False,
                "task_success": bool(_raw_success(self._last_info)),
                "official_success_source": 'info["done"]["success"]',
                "stop_reason": "search_motion_too_large_for_visual_gate",
                "held_hand": held,
                "press_hand": press,
                "active_role": role,
                "active_hand": active,
                "target_pose": {
                    "position": target.tolist(),
                    "quat_xyzw": quat.tolist(),
                },
                "motion_authorization": _wire_safe(authorization),
                "total_env_steps": int(self._env_steps),
            }
        plan_signature = self._prepress_plan_signature(
            role=role,
            target=target,
            quaternion=quat,
            gate_binding=gate_binding,
            goal_binding=goal_binding,
        )
        certificate: dict[str, Any] | None = None
        if not plan_only:
            certificate = getattr(self, "_prepress_plan_certificate", None)
            self._prepress_plan_certificate = None
            if (
                not isinstance(certificate, dict)
                or certificate.get("signature") != plan_signature
            ):
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": "matching_plan_only_certificate_required",
                    "held_hand": held,
                    "press_hand": press,
                    "active_role": role,
                    "active_hand": active,
                    "target_pose": {
                        "position": target.tolist(),
                        "quat_xyzw": quat.tolist(),
                    },
                    "motion_authorization": _wire_safe(authorization),
                    "total_env_steps": int(self._env_steps),
                }
            if authorization.get("policy") == "direct_back_to_front":
                self._prepress_coarse_flip_used = True
        planner = self._require_planner()
        backend = planner.backend
        radio, _table = self._resolve_handoff_targets()
        root = getattr(radio, "root_link", None)
        if root is None:
            raise RuntimeError("radio root link is unavailable for attached collision")
        reported_attached = backend.get_attached_object(held)
        target_root = str(getattr(root, "prim_path", "")).rstrip("/")
        expected_attachment_link = EEF_LINK_BY_HAND[held]
        if reported_attached is not None:
            reported_keys = (
                set(reported_attached) if isinstance(reported_attached, dict) else set()
            )
            reported_root = (
                reported_attached.get(expected_attachment_link)
                if isinstance(reported_attached, dict)
                else None
            )
            reported_root_path = str(getattr(reported_root, "prim_path", "")).rstrip(
                "/"
            )
            if reported_keys != {expected_attachment_link} or not (
                reported_root is root or reported_root_path == target_root
            ):
                raise RuntimeError(
                    "radio attachment is not anchored exclusively to held EEF"
                )
        attached = {expected_attachment_link: root}
        if plan_only:
            plan = backend.plan_prepress_arm_trajectory(
                hand=active,
                target_xyz=target,
                target_quat_xyzw=quat,
                timeout_s=float(timeout_s),
                attached_obj=attached,
            )
            if not plan.get("ok"):
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": str(plan.get("stop_reason", "curobo_plan_failed")),
                    "held_hand": held,
                    "press_hand": press,
                    "active_role": role,
                    "active_hand": active,
                    "target_pose": {
                        "position": target.tolist(),
                        "quat_xyzw": quat.tolist(),
                    },
                    "motion_authorization": _wire_safe(authorization),
                    "planner_metrics": _wire_safe(plan.get("metrics", {})),
                    "total_env_steps": int(self._env_steps),
                }
            q_path = np.asarray(plan.get("joint_trajectory"), dtype=np.float64)
        else:
            assert isinstance(certificate, dict)
            q_path = np.asarray(certificate.get("_joint_trajectory"), dtype=np.float64)
        robot = self._robot()
        if robot is None or q_path.ndim != 2:
            raise RuntimeError("pre-press cuRobo trajectory is invalid")
        initial_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if q_path.shape[1] != initial_q.size or not np.isfinite(q_path).all():
            raise RuntimeError("pre-press cuRobo trajectory q layout is invalid")
        if not plan_only:
            assert isinstance(certificate, dict)
            certified_start = np.asarray(
                certificate.get("_start_joint_positions"), dtype=np.float64
            ).reshape(-1)
            trajectory_digest = hashlib.sha256(
                np.ascontiguousarray(q_path).tobytes()
            ).hexdigest()
            if (
                certified_start.shape != initial_q.shape
                or not np.allclose(certified_start, initial_q, atol=1e-6, rtol=0.0)
                or certificate.get("trajectory_sha256") != trajectory_digest
            ):
                raise RuntimeError(
                    "certified pre-press trajectory or start state changed"
                )
        arm_idx = getattr(robot, "arm_control_idx", {}) or {}
        base_lock_indices = list(
            np.asarray(_numpy_tree(getattr(robot, "base_idx", [])), dtype=int).reshape(
                -1
            )
        )
        trunk_lock_indices = list(
            np.asarray(
                _numpy_tree(getattr(robot, "trunk_control_idx", [])), dtype=int
            ).reshape(-1)
        )
        inactive_arm_indices = list(
            np.asarray(_numpy_tree(arm_idx.get(locked, [])), dtype=int).reshape(-1)
        )
        if len(base_lock_indices) != 6:
            raise RuntimeError("pre-press requires all six virtual base joints locked")
        if len(trunk_lock_indices) != 4:
            raise RuntimeError("pre-press requires all four trunk joints locked")
        if len(inactive_arm_indices) != 7:
            raise RuntimeError(
                "pre-press requires all seven inactive arm joints locked"
            )
        raw_locked_indices = (
            base_lock_indices + trunk_lock_indices + inactive_arm_indices
        )
        if (
            len(set(raw_locked_indices)) != len(raw_locked_indices)
            or min(raw_locked_indices, default=-1) < 0
            or max(raw_locked_indices, default=-1) >= initial_q.size
        ):
            raise RuntimeError("pre-press locked joint indices are invalid")
        locked_indices = sorted(raw_locked_indices)
        planned_locked_drift = float(
            np.max(np.abs(q_path[:, locked_indices] - initial_q[locked_indices]))
        )
        if planned_locked_drift > 1e-6:
            raise RuntimeError(
                "pre-press plan changed locked base, trunk, or inactive arm"
            )
        generator = backend._generator(kind="prepress_arm", hand=active)
        collision = backend._check_q_trajectory_collisions(
            generator, q_path, attached_obj=attached
        )
        if not collision.get("available") or collision.get("colliding", True):
            raise RuntimeError(
                "pre-press full attached-object path is not collision-free"
            )
        self._assert_prepress_checkpoint_bound()
        self._prepress_round = int(getattr(self, "_prepress_round", 0)) + 1
        round_index = self._prepress_round
        trace_path = (
            self._output_dir
            / "state_checkpoints"
            / f"prepress_motion_round_{round_index:03d}_trace.json"
        )
        video_path = (
            self._output_dir / f"curobo_prepress_round_{round_index:03d}_episode.mp4"
        )
        if plan_only:
            refreshed_gate_binding = self._prepress_gate_binding(
                allow_expired=goal_binding is not None
            )
            if gate_binding is not None and refreshed_gate_binding != gate_binding:
                self._prepress_plan_certificate = None
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": "visual_gate_expired_during_plan_only",
                    "active_role": role,
                    "active_hand": active,
                    "target_pose": {
                        "position": target.tolist(),
                        "quat_xyzw": quat.tolist(),
                    },
                    "total_env_steps": int(self._env_steps),
                }
            trajectory_digest = hashlib.sha256(
                np.ascontiguousarray(q_path).tobytes()
            ).hexdigest()
            self._prepress_plan_certificate = {
                "signature": plan_signature,
                "env_step": int(self._env_steps),
                "role": role,
                "target_pose": {
                    "position": target.tolist(),
                    "quat_xyzw": quat.tolist(),
                },
                "gate_binding": gate_binding,
                "goal_binding": deepcopy(goal_binding),
                "selected_candidate": deepcopy(candidate_metadata),
                "trajectory_sha256": trajectory_digest,
                "collision_report": _wire_safe(collision),
                "_joint_trajectory": q_path.copy(),
                "_start_joint_positions": initial_q.copy(),
            }
            public_certificate = {
                key: value
                for key, value in self._prepress_plan_certificate.items()
                if not key.startswith("_")
            }
            return {
                "_finish": False,
                "primitive_success": True,
                "task_success": bool(_raw_success(self._last_info)),
                "official_success_source": 'info["done"]["success"]',
                "stop_reason": "prepress_plan_certified",
                "plan_only": True,
                "active_role": role,
                "active_hand": active,
                "locked_hand": locked,
                "waypoints": int(len(q_path)),
                "target_pose": {
                    "position": target.tolist(),
                    "quat_xyzw": quat.tolist(),
                },
                "motion_authorization": _wire_safe(authorization),
                "plan_certificate": _wire_safe(public_certificate),
                "button_goal": _wire_safe(goal_binding),
                "selected_candidate": _wire_safe(candidate_metadata),
                "cuRobo_collision_report": _wire_safe(collision),
                "total_env_steps": int(self._env_steps),
            }
        self._gripper_latch[held] = -1.0
        self.start_video_segment(video_path)
        trace: list[dict[str, Any]] = []
        start_step = int(self._env_steps)
        stop_reason: str | None = None
        deadline = time.monotonic() + float(timeout_s)
        for index, waypoint in enumerate(q_path, start=1):
            if time.monotonic() >= deadline:
                stop_reason = "timeout"
                break
            action = np.asarray(
                backend.joint_target_to_action(waypoint, hand=None), dtype=np.float32
            ).reshape(23)
            action[ENV_ACTION_SEGMENTS[f"{held}_gripper"]] = -1.0
            action[ENV_ACTION_SEGMENTS[f"{press}_gripper"]] = float(
                self._gripper_latch[press]
            )
            step_obs, _reward, term, trunc, infos = self._env._direct_process.step_env(
                __import__("torch").as_tensor(action).reshape(1, 23), need_obs=True
            )
            self._env_steps += 1
            self._last_info = _numpy_tree(infos[0])
            observation = _single_observation(self._env._wrap_obs(step_obs))
            self._last_observation = observation
            self._record_rgbd_frames(step_obs, observation)
            self._append_video(observation)
            sample = {
                "waypoint": index,
                "env_step": int(self._env_steps),
                "held_gripper_command": -1.0,
                "official_task_success": bool(_raw_success(self._last_info)),
            }
            trace.append(sample)
            _write_json_atomic(trace_path, trace)
            if _scalar_bool(term) or _scalar_bool(trunc) or _raw_done(self._last_info):
                stop_reason = "episode_stopped"
            if stop_reason is not None:
                break
        self._finalize_video_segment()
        self._video_sealed = True
        visual_paths, capture_group = self._capture_prepress_views(
            round_index=round_index
        )
        self._assert_prepress_checkpoint_bound()
        final_stability = self._prepress_stability_snapshot()
        final_q = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(1, -1)
        actual_locked_drift = float(
            np.max(np.abs(final_q[0, locked_indices] - initial_q[locked_indices]))
        )
        endpoint_collision = backend._check_q_trajectory_collisions(
            generator, final_q, attached_obj=attached
        )
        final_pose = backend.get_eef_pose(active)
        position_error = (
            float(np.linalg.norm(np.asarray(final_pose[0]) - target))
            if final_pose
            else float("inf")
        )
        orientation_error = (
            _quaternion_angle_rad(final_pose[1], quat) if final_pose else float("inf")
        )
        success = bool(
            stop_reason is None
            and position_error <= 0.02
            and orientation_error <= 0.087
            and final_stability["stable"]
            and actual_locked_drift <= 0.02
            and endpoint_collision.get("available") is True
            and endpoint_collision.get("colliding") is False
        )
        if stop_reason is None and not final_stability["stable"]:
            stop_reason = "endpoint_radio_hold_or_press_clear_failed"
        if stop_reason is None and actual_locked_drift > 0.02:
            stop_reason = "locked_chain_drift"
        if stop_reason is None and (
            not endpoint_collision.get("available")
            or endpoint_collision.get("colliding", True)
        ):
            stop_reason = "endpoint_collision"
        if stop_reason is None and not success:
            stop_reason = "active_eef_target_not_reached"
        motion = {
            "motion_id": f"prepress_motion_{round_index:03d}",
            "round": round_index,
            "primitive_success": bool(success),
            "stop_reason": f"{role}_hand_pose_reached" if success else stop_reason,
            "held_hand": held,
            "press_hand": press,
            "active_role": role,
            "active_hand": active,
            "locked_hand": locked,
            "target_pose": {"position": target.tolist(), "quat_xyzw": quat.tolist()},
            "button_goal": _wire_safe(goal_binding),
            "selected_candidate": _wire_safe(candidate_metadata),
            "motion_authorization": _wire_safe(authorization),
            "start_env_step": start_step,
            "end_env_step": int(self._env_steps),
            "executed_steps": len(trace),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "planned_locked_drift_rad": planned_locked_drift,
            "actual_locked_drift": actual_locked_drift,
            "collision_report": collision,
            "endpoint_collision_report": endpoint_collision,
            "trace_path": str(trace_path),
            "video_path": str(video_path),
            "visual_review_paths": visual_paths,
            "capture_group_id": capture_group,
            "public_visual_review": {},
            "three_view_observed_by_vlm": False,
            "visual_review_pass": False,
            "visual_review_verdict": "pending_post_motion_button_gate",
            "final_stability": final_stability,
            "next_search_rotate_wrist_candidates": self._prepress_search_rotations(
                final_stability
            ),
        }
        self._prepress_motion = motion
        self._prepress_plan_certificate = None
        self._prepress_gate = self._prepress_projection = self._prepress_geometry = None
        _write_json_atomic(trace_path, {"motion": motion, "waypoints": trace})
        return {
            "_finish": False,
            "primitive_success": bool(success),
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": motion["stop_reason"],
            **_wire_safe(motion),
            "total_env_steps": int(self._env_steps),
        }

    def prepress_move_to(
        self,
        button_goal: dict[str, Any],
        plan_only: bool = False,
        timeout_s: float = 90.0,
        role: str = "held",
    ) -> dict[str, Any]:
        """Resolve a button-space goal, search EEF candidates, then move once."""

        self._assert_prepress_checkpoint_bound()
        if role not in {"held", "press"}:
            raise ValueError("pre-press role must be 'held' or 'press'")
        if not isinstance(button_goal, dict):
            raise ValueError("button_goal must be a dictionary")
        expected_kind = "held_button_alignment" if role == "held" else "press_staging"
        if button_goal.get("kind") != expected_kind:
            raise ValueError(
                f"role={role!r} requires button_goal.kind={expected_kind!r}"
            )
        canonical_goal = json.loads(
            json.dumps(button_goal, sort_keys=True, separators=(",", ":"))
        )
        if role == "held":
            numeric_defaults = {
                "side_view_tolerance_deg": 15.0,
                "face_toward_tolerance_deg": 30.0,
                "position_slack_m": 0.04,
                "minimum_table_clearance_m": 0.12,
            }
            if (
                canonical_goal.get("head_view") != "side"
                or canonical_goal.get("face_toward") != "press"
            ):
                raise ValueError(
                    "held_button_alignment requires head_view='side' and "
                    "face_toward='press'"
                )
            for name, default in numeric_defaults.items():
                canonical_goal.setdefault(name, default)
            canonical_goal.setdefault("candidate_budget", 12)
            if "toward_robot_m" not in canonical_goal:
                raise ValueError("held_button_alignment requires toward_robot_m")
            numeric_bounds = {
                "toward_robot_m": (0.0, 0.30),
                "side_view_tolerance_deg": (1e-9, 30.0),
                "face_toward_tolerance_deg": (1e-9, 45.0),
                "position_slack_m": (0.0, 0.10),
                "minimum_table_clearance_m": (0.08, 0.25),
            }
            for name, (lower, upper) in numeric_bounds.items():
                value = canonical_goal.get(name)
                if isinstance(value, bool):
                    raise ValueError(f"button_goal.{name} must be numeric")
                number = float(value)
                if not np.isfinite(number) or not lower <= number <= upper:
                    raise ValueError(
                        f"button_goal.{name} must be in [{lower}, {upper}]"
                    )
                canonical_goal[name] = number
        else:
            canonical_goal.setdefault("standoff_m", 0.055)
            canonical_goal.setdefault("candidate_budget", 8)
            projection_id = canonical_goal.get("projection_id")
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("press_staging requires projection_id")
            alignment_phase = canonical_goal.get("alignment_phase", "final")
            if alignment_phase not in {"final", "observation"}:
                raise ValueError(
                    "button_goal.alignment_phase must be final or observation"
                )
            standoff = canonical_goal.get("standoff_m")
            if isinstance(standoff, bool):
                raise ValueError("button_goal.standoff_m must be numeric")
            standoff = float(standoff)
            standoff_max = (
                PREPRESS_AXIAL_STANDOFF_MAX_M
                if alignment_phase == "final"
                else PRESS_STAGING_AXIAL_STANDOFF_MAX_M
            )
            if (
                not np.isfinite(standoff)
                or not PREPRESS_AXIAL_STANDOFF_MIN_M
                <= standoff
                <= standoff_max
            ):
                raise ValueError(
                    "button_goal.standoff_m must be in "
                    f"[0.03, {standoff_max:.2f}] for {alignment_phase} alignment"
                )
            canonical_goal["alignment_phase"] = alignment_phase
            canonical_goal["standoff_m"] = standoff
        budget = canonical_goal.get("candidate_budget")
        max_budget = 32 if role == "held" else 16
        if isinstance(budget, bool) or not isinstance(budget, int):
            raise ValueError("button_goal.candidate_budget must be an integer")
        if not 1 <= budget <= max_budget:
            raise ValueError(
                f"button_goal.candidate_budget must be in [1, {max_budget}]"
            )
        goal_digest = hashlib.sha256(
            json.dumps(canonical_goal, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        if not plan_only:
            certificate = getattr(self, "_prepress_plan_certificate", None)
            if not isinstance(certificate, dict):
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": "matching_button_goal_plan_required",
                    "active_role": role,
                    "total_env_steps": int(self._env_steps),
                }
            goal_binding = certificate.get("goal_binding")
            candidate = certificate.get("selected_candidate")
            target_pose = certificate.get("target_pose")
            if (
                not isinstance(goal_binding, dict)
                or goal_binding.get("goal_digest") != goal_digest
                or goal_binding.get("role") != role
                or not isinstance(candidate, dict)
                or not isinstance(target_pose, dict)
            ):
                self._prepress_plan_certificate = None
                return {
                    "_finish": False,
                    "primitive_success": False,
                    "task_success": bool(_raw_success(self._last_info)),
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": "button_goal_changed_after_plan_only",
                    "active_role": role,
                    "total_env_steps": int(self._env_steps),
                }
            return self._prepress_move_to_candidate(
                role=role,
                target_xyz=target_pose["position"],
                target_quat_xyzw=target_pose["quat_xyzw"],
                plan_only=False,
                timeout_s=timeout_s,
                goal_binding=goal_binding,
                candidate_metadata=candidate,
            )

        self._prepress_plan_certificate = None
        context = self._prepress_context
        assert isinstance(context, dict)
        stability = self._prepress_stability_snapshot()
        if not stability["stable"]:
            raise RuntimeError("radio is not stably held before button-goal planning")
        held, press = context["held_hand"], context["press_hand"]
        alignment_phase = canonical_goal.get("alignment_phase", "joint")
        if role == "held" and alignment_phase == "normal_refine":
            frame = self._frame_cache.latest("head")
            if int(frame.step_index) != int(self._env_steps):
                raise RuntimeError(
                    "normal_refine requires a fresh current head observation"
                )
            gate = {
                "camera": "head",
                "resolved_camera": "head",
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "env_step": int(self._env_steps),
                "gate_id": hashlib.sha256(
                    f"normal_refine:{frame.frame_id}:{self._env_steps}".encode()
                ).hexdigest()[:24],
                "face_class": "MODEL_PRIOR_BUTTON_FACE",
                "button_visible": False,
                "provenance": "radio_local_button_model_after_position_first",
            }
            projection = None
        else:
            gate = self._prepress_gate
            if not isinstance(gate, dict) or gate.get("env_step") != self._env_steps:
                raise RuntimeError("a current button visual gate is required")
            frame = self._frame_cache.get_current(
                str(gate["resolved_camera"]), str(gate["frame_id"])
            )
            projection = self._prepress_projection
        attempts: list[dict[str, Any]] = []

        if role == "held":
            if gate.get("camera") != "head" or gate.get("resolved_camera") != "head":
                raise RuntimeError("held button alignment requires a fresh head gate")
            if alignment_phase == "normal_refine":
                radio_rotation = pose_matrix_xyzw(
                    [0.0, 0.0, 0.0], stability["radio_quat_xyzw"]
                )[:3, :3]
                button_center = (
                    np.asarray(stability["radio_position_world"], dtype=np.float64)
                    + radio_rotation
                    @ np.asarray(RADIO_LOCAL_BUTTON_CENTER_M, dtype=np.float64)
                ).tolist()
                button_normal = (
                    radio_rotation
                    @ np.asarray(RADIO_LOCAL_BUTTON_FACE_NORMAL, dtype=np.float64)
                ).tolist()
                geometry_source = "radio_local_button_prior_normal_refine"
            elif gate.get("face_class") == BUTTON_FACE_CLASS:
                if not (
                    isinstance(projection, dict)
                    and projection.get("env_step") == self._env_steps
                    and projection.get("gate_id") == gate.get("gate_id")
                    and projection.get("frame_id") == gate.get("frame_id")
                    and projection.get("capture_group_id")
                    == gate.get("capture_group_id")
                ):
                    raise RuntimeError(
                        "visible-button alignment requires a fresh head projection"
                    )
                button_center = projection["button_center_world"]
                button_normal = projection["button_normal_world"]
                geometry_source = "fresh_head_button_projection"
            elif gate.get("face_class") == CLEAR_SLOTTED_BACK_FACE_CLASS:
                radio_rotation = pose_matrix_xyzw(
                    [0.0, 0.0, 0.0], stability["radio_quat_xyzw"]
                )[:3, :3]
                button_center = (
                    np.asarray(stability["radio_position_world"], dtype=np.float64)
                    + radio_rotation
                    @ np.asarray(RADIO_LOCAL_BUTTON_CENTER_M, dtype=np.float64)
                ).tolist()
                button_normal = (
                    radio_rotation
                    @ np.asarray(RADIO_LOCAL_BUTTON_FACE_NORMAL, dtype=np.float64)
                ).tolist()
                geometry_source = "clear_slotted_opposite_face_prior"
            else:
                raise RuntimeError(
                    "held button alignment requires BUTTON_FACE or clear slotted back face"
                )

            robot = self._robot()
            if robot is None:
                raise RuntimeError("robot pose is unavailable for toward_robot goal")
            robot_position, _robot_quat = self._object_pose(robot)
            button_center_array = np.asarray(button_center, dtype=np.float64)
            chest_direction = (
                np.asarray(robot_position, dtype=np.float64) - button_center_array
            )
            chest_direction[2] = 0.0
            chest_norm = float(np.linalg.norm(chest_direction))
            if chest_norm <= 1e-9:
                raise RuntimeError("button is coincident with robot XY reference")
            chest_direction /= chest_norm
            head_axis = frame.camera_to_world[:3, :3] @ np.asarray(
                [0.0, 0.0, -1.0], dtype=np.float64
            )
            head_target_uv = canonical_goal.get("head_target_uv")
            if head_target_uv is None:
                translation_direction = chest_direction
                translation_m = float(canonical_goal["toward_robot_m"])
                nominal_button = (
                    button_center_array + translation_m * translation_direction
                )
                translation_source = "toward_robot"
            else:
                if gate.get("face_class") != BUTTON_FACE_CLASS:
                    raise RuntimeError(
                        "head_target_uv requires a directly visible button projection"
                    )
                u, v = (float(head_target_uv[0]), float(head_target_uv[1]))
                if not (0.0 <= u < frame.intrinsics.width) or not (
                    0.0 <= v < frame.intrinsics.height
                ):
                    raise RuntimeError("head_target_uv is outside the head image")
                world_to_camera = np.linalg.inv(frame.camera_to_world)
                current_button_camera = transform_point(
                    world_to_camera, button_center_array
                )
                depth_m = -float(current_button_camera[2])
                target_button_camera = camera_point_from_pixel(
                    frame.intrinsics,
                    u=int(round(u)),
                    v=int(round(v)),
                    depth_m=depth_m,
                )
                if frame.correction_profile is not None:
                    target_button_camera = (
                        frame.correction_profile.apply_camera_point(
                            target_button_camera
                        )
                    )
                nominal_button = transform_point(
                    frame.camera_to_world, target_button_camera
                )
                pixel_translation = nominal_button - button_center_array
                translation_m = float(np.linalg.norm(pixel_translation))
                if translation_m <= 1e-9:
                    translation_direction = chest_direction
                    translation_m = 0.0
                else:
                    translation_direction = pixel_translation / translation_m
                translation_source = "head_button_center_pixel"
            to_press = (
                np.asarray(stability["press_eef_position_world"], dtype=np.float64)
                - nominal_button
            )
            to_press_norm = float(np.linalg.norm(to_press))
            if to_press_norm <= 1e-9:
                raise RuntimeError("button goal is coincident with press EEF")
            to_press /= to_press_norm
            side_normal = to_press - np.dot(to_press, head_axis) * head_axis
            side_norm = float(np.linalg.norm(side_normal))
            if side_norm <= 1e-9:
                fallback = np.asarray(button_normal, dtype=np.float64)
                side_normal = fallback - np.dot(fallback, head_axis) * head_axis
                side_norm = float(np.linalg.norm(side_normal))
            if side_norm <= 1e-9:
                for basis in np.eye(3):
                    side_normal = basis - np.dot(basis, head_axis) * head_axis
                    side_norm = float(np.linalg.norm(side_normal))
                    if side_norm > 1e-9:
                        break
            side_normal /= side_norm
            if np.dot(side_normal, to_press) < 0.0:
                side_normal = -side_normal
            slack = float(canonical_goal.get("position_slack_m", 0.04))
            budget = int(canonical_goal.get("candidate_budget", 12))
            clearance_lift = max(
                0.0,
                float(canonical_goal.get("minimum_table_clearance_m", 0.12))
                - float(stability["air_gap_m"]),
            )
            lateral = np.asarray(
                [-translation_direction[1], translation_direction[0], 0.0],
                dtype=np.float64,
            )
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm <= 1e-9:
                lateral = np.asarray(
                    [-chest_direction[1], chest_direction[0], 0.0],
                    dtype=np.float64,
                )
                lateral_norm = float(np.linalg.norm(lateral))
            lateral /= max(lateral_norm, 1e-9)
            if head_target_uv is None:
                position_options = [
                    np.asarray([0.0, 0.0, clearance_lift]),
                    slack * lateral + np.asarray([0.0, 0.0, clearance_lift]),
                    -slack * lateral + np.asarray([0.0, 0.0, clearance_lift]),
                    np.asarray([0.0, 0.0, clearance_lift + slack]),
                    np.asarray([0.0, 0.0, max(0.0, clearance_lift - slack)]),
                ]
            else:
                radius_px = float(canonical_goal.get("head_target_radius_px", 60.0))
                diagonal_px = radius_px / np.sqrt(2.0)
                pixel_offsets = [
                    (0.0, 0.0),
                    (-radius_px, 0.0),
                    (radius_px, 0.0),
                    (0.0, -radius_px),
                    (0.0, radius_px),
                    (-diagonal_px, -diagonal_px),
                    (-diagonal_px, diagonal_px),
                    (diagonal_px, -diagonal_px),
                    (diagonal_px, diagonal_px),
                ]
                position_options = []
                for du, dv in pixel_offsets:
                    sample_u = int(round(float(head_target_uv[0]) + du))
                    sample_v = int(round(float(head_target_uv[1]) + dv))
                    if not (0 <= sample_u < frame.intrinsics.width) or not (
                        0 <= sample_v < frame.intrinsics.height
                    ):
                        continue
                    sample_camera = camera_point_from_pixel(
                        frame.intrinsics,
                        u=sample_u,
                        v=sample_v,
                        depth_m=depth_m,
                    )
                    if frame.correction_profile is not None:
                        sample_camera = frame.correction_profile.apply_camera_point(
                            sample_camera
                        )
                    sample_world = transform_point(frame.camera_to_world, sample_camera)
                    position_options.append(
                        sample_world
                        - nominal_button
                        + np.asarray([0.0, 0.0, clearance_lift])
                    )
                if not position_options:
                    raise RuntimeError("head_target_uv neighborhood is outside the image")
            alignment_phase = canonical_goal.get("alignment_phase", "joint")
            roll_options = (
                [0.0]
                if alignment_phase == "position_first"
                else [0.0, -30.0, 30.0, -60.0, 60.0, -90.0, 90.0, 180.0]
            )
            roll_count = min(
                len(roll_options),
                8 if alignment_phase == "normal_refine" else 3,
                budget,
            )
            position_count = max(1, min(len(position_options), budget // roll_count))
            while position_count * roll_count > budget:
                position_count -= 1
            internal_goal = {
                "chest_direction_world": translation_direction.tolist(),
                "chest_translation_m": translation_m,
                "position_perturbations_world_m": [
                    value.tolist() for value in position_options[:position_count]
                ],
                "orientation_perturbations_world_axis_angle": [
                    [*side_normal.tolist(), float(np.deg2rad(value))]
                    for value in roll_options[:roll_count]
                ],
                "normal_blend_factors": [1.0],
                "orientation_goal": (
                    "preserve_current"
                    if alignment_phase == "position_first"
                    else "side_to_press"
                ),
                "max_position_perturbation_m": max(
                    float(np.linalg.norm(value)) for value in position_options
                ),
                "max_orientation_perturbation_rad": float(np.pi),
                "max_face_to_press_angle_deg": float(
                    canonical_goal.get("face_toward_tolerance_deg", 30.0)
                ),
                "max_press_approach_opposition_angle_deg": 180.0,
                "target_head_side_angle_deg": 90.0,
                "max_head_side_error_deg": float(
                    canonical_goal.get("side_view_tolerance_deg", 15.0)
                ),
                "chest_translation_tolerance_m": (
                    max(
                        0.005,
                        max(float(np.linalg.norm(value)) for value in position_options)
                        + 0.005,
                    )
                    if head_target_uv is not None
                    else 0.005
                ),
                "max_candidates": budget,
            }
            generated = generate_button_goal_pose_candidates(
                world_held_transform=pose_matrix_xyzw(
                    stability["held_eef_position_world"],
                    stability["held_eef_quat_xyzw"],
                ),
                world_radio_transform=pose_matrix_xyzw(
                    stability["radio_position_world"],
                    stability["radio_quat_xyzw"],
                ),
                held_to_radio_transform=pose_matrix_xyzw(
                    stability["relative_position_m"],
                    stability["relative_quat_xyzw"],
                ),
                button_center_world=button_center,
                button_normal_world=button_normal,
                world_press_transform=pose_matrix_xyzw(
                    stability["press_eef_position_world"],
                    stability["press_eef_quat_xyzw"],
                ),
                goal=internal_goal,
                head_optical_axis_world=head_axis,
            )
            candidates = [
                {
                    **candidate,
                    "target_pose": candidate["target_held_eef_pose"],
                    "head_target_uv": head_target_uv,
                    "head_target_radius_px": canonical_goal.get(
                        "head_target_radius_px"
                    ),
                    "alignment_phase": alignment_phase,
                    "translation_source": translation_source,
                }
                for candidate in generated["candidates"]
                if candidate["eligible"]
            ][:budget]
        else:
            if not (
                gate.get("face_class") == BUTTON_FACE_CLASS
                and isinstance(projection, dict)
                and projection.get("projection_id")
                == canonical_goal.get("projection_id")
                and projection.get("env_step") == self._env_steps
                and projection.get("gate_id") == gate.get("gate_id")
                and projection.get("camera") == "press_wrist"
                and projection.get("resolved_camera") == f"{press}_wrist"
            ):
                raise RuntimeError(
                    "press staging requires a fresh dynamic press-wrist button projection"
                )
            budget = int(canonical_goal.get("candidate_budget", 8))
            world_press = pose_matrix_xyzw(
                stability["press_eef_position_world"],
                stability["press_eef_quat_xyzw"],
            )
            alignment_phase = canonical_goal.get("alignment_phase", "final")
            generated = generate_press_staging_pose_candidates(
                button_center_world=projection["button_center_world"],
                button_normal_world=projection["button_normal_world"],
                world_press_transform=world_press,
                standoff_m=float(canonical_goal.get("standoff_m", 0.055)),
                max_candidates=budget,
                alignment_phase=alignment_phase,
                eef_to_camera_transform=(
                    np.linalg.inv(world_press) @ frame.camera_to_world
                    if alignment_phase == "observation"
                    else None
                ),
            )
            geometry_source = (
                "fresh_press_wrist_projection_camera_centered_observation"
                if alignment_phase == "observation"
                else "fresh_press_wrist_button_projection"
            )
            candidates = [
                {
                    **candidate,
                    "target_pose": candidate["target_press_eef_pose"],
                }
                for candidate in generated["candidates"]
                if candidate["eligible"]
            ][:budget]

        grasp_binding = {
            "radio_pose_world": stability["radio_position_world"],
            "radio_quat_xyzw": stability["radio_quat_xyzw"],
            "held_to_radio_position": stability["relative_position_m"],
            "held_to_radio_quat_xyzw": stability["relative_quat_xyzw"],
        }
        goal_binding = {
            "resolver_version": 1,
            "role": role,
            "goal": canonical_goal,
            "goal_digest": goal_digest,
            "geometry_source": geometry_source,
            "gate_id": gate.get("gate_id"),
            "projection_id": (
                projection.get("projection_id")
                if isinstance(projection, dict)
                else None
            ),
            "frame_id": gate.get("frame_id"),
            "capture_group_id": gate.get("capture_group_id"),
            "env_step": int(self._env_steps),
            "grasp_transform_sha256": hashlib.sha256(
                json.dumps(
                    grasp_binding, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        deadline = time.monotonic() + float(timeout_s)
        for candidate in candidates:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                attempts.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "primitive_success": False,
                        "stop_reason": "candidate_search_timeout",
                    }
                )
                break
            target_pose = candidate["target_pose"]
            result = self._prepress_move_to_candidate(
                role=role,
                target_xyz=target_pose["position"],
                target_quat_xyzw=target_pose["quat_xyzw"],
                plan_only=True,
                timeout_s=remaining,
                goal_binding=goal_binding,
                candidate_metadata=candidate,
            )
            attempts.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "primitive_success": bool(result.get("primitive_success")),
                    "stop_reason": result.get("stop_reason"),
                    "planner_metrics": result.get("planner_metrics"),
                }
            )
            if result.get("primitive_success"):
                certificate = self._prepress_plan_certificate
                if isinstance(certificate, dict):
                    certificate["candidate_attempts"] = deepcopy(attempts)
                result.update(
                    {
                        "button_goal": _wire_safe(canonical_goal),
                        "goal_binding": _wire_safe(goal_binding),
                        "candidate_attempts": _wire_safe(attempts),
                        "selected_candidate": _wire_safe(candidate),
                    }
                )
                return result

        self._prepress_plan_certificate = None
        return {
            "_finish": False,
            "primitive_success": False,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "button_goal_unreachable",
            "active_role": role,
            "held_hand": held,
            "press_hand": press,
            "button_goal": _wire_safe(canonical_goal),
            "goal_binding": _wire_safe(goal_binding),
            "candidate_count": len(candidates),
            "candidate_attempts": _wire_safe(attempts),
            "total_env_steps": int(self._env_steps),
        }

    def prepress_rotate_wrist(
        self,
        target_quat_xyzw: Any | None = None,
        relative_axis_angle: Any | None = None,
        frame: str = "eef",
        plan_only: bool = False,
        timeout_s: float = 90.0,
        role: str = "held",
    ) -> dict[str, Any]:
        """Rotate one dynamic role through the dedicated pre-press planner."""

        self._assert_prepress_checkpoint_bound()
        if role not in {"held", "press"}:
            raise ValueError("pre-press role must be 'held' or 'press'")
        if (target_quat_xyzw is None) == (relative_axis_angle is None):
            raise ValueError(
                "rotate_wrist requires exactly one of target_quat_xyzw or "
                "relative_axis_angle"
            )
        if frame not in {"world", "eef"}:
            raise ValueError("frame must be 'world' or 'eef'")
        if not isinstance(plan_only, bool):
            raise ValueError("plan_only must be boolean")
        current = self._prepress_stability_snapshot()
        position_key = f"{role}_eef_position_world"
        quat_key = f"{role}_eef_quat_xyzw"
        position = np.asarray(current[position_key], dtype=np.float64).reshape(3)
        current_quat = np.asarray(current[quat_key], dtype=np.float64).reshape(4)
        current_quat /= max(float(np.linalg.norm(current_quat)), 1e-12)
        if target_quat_xyzw is not None:
            target_quat = np.asarray(target_quat_xyzw, dtype=np.float64).reshape(4)
            if (
                not np.isfinite(target_quat).all()
                or float(np.linalg.norm(target_quat)) <= 1e-9
            ):
                raise ValueError("target_quat_xyzw is invalid")
            target_quat /= np.linalg.norm(target_quat)
        else:
            command = np.asarray(relative_axis_angle, dtype=np.float64).reshape(4)
            axis, angle = command[:3], float(command[3])
            axis_norm = float(np.linalg.norm(axis))
            if not np.isfinite(command).all() or axis_norm <= 1e-9:
                raise ValueError("relative_axis_angle is invalid")
            if role == "press" and abs(angle) > float(np.pi) + 1e-9:
                raise ValueError(
                    "press observation rotation angle must not exceed pi radians"
                )
            axis /= axis_norm
            half = 0.5 * angle
            delta = np.r_[axis * np.sin(half), np.cos(half)]
            target_quat = (
                quat_multiply_xyzw(delta, current_quat)
                if frame == "world"
                else quat_multiply_xyzw(current_quat, delta)
            )
        result = self._prepress_move_to_candidate(
            role=role,
            target_xyz=position.tolist(),
            target_quat_xyzw=target_quat.tolist(),
            plan_only=plan_only,
            timeout_s=timeout_s,
            press_observation_rotation=role == "press",
        )
        result["rotation_request"] = {
            "role": role,
            "target_quat_xyzw": target_quat.tolist(),
            "relative_axis_angle": (
                None
                if relative_axis_angle is None
                else np.asarray(relative_axis_angle, dtype=np.float64).tolist()
            ),
            "frame": frame,
        }
        return result

    def save_prepress_checkpoint(
        self,
        checkpoint_name: str = "state_checkpoint_2",
        stage: str = "pre_press_alignment",
        visual_review: bool = True,
        user_review_override: bool = False,
        review_note: str | None = None,
    ) -> dict[str, Any]:
        """Atomically save checkpoint2 only after the complete current gate chain."""

        if checkpoint_name != "state_checkpoint_2" or stage != "pre_press_alignment":
            raise ValueError("pre-press finalizer only writes state_checkpoint_2")
        if visual_review is not True:
            raise ValueError("state_checkpoint_2 requires visual review")
        path1, checkpoint1 = self._assert_prepress_checkpoint_bound()
        sha1 = _sha256_file(path1)
        context, gate, projection = (
            self._prepress_context,
            self._prepress_gate,
            self._prepress_projection,
        )
        geometry, motion = self._prepress_geometry, self._prepress_motion
        current_step = int(self._env_steps)
        if user_review_override:
            if not isinstance(review_note, str) or not review_note.strip():
                raise ValueError("user_review_override requires a non-empty review_note")
            if not all(
                isinstance(value, dict)
                for value in (context, gate, projection, geometry)
            ):
                raise RuntimeError(
                    "user-reviewed pre-press gate, projection, and geometry are incomplete"
                )
            assert isinstance(context, dict)
            assert isinstance(gate, dict)
            assert isinstance(projection, dict)
            assert isinstance(geometry, dict)
            frame = self._frame_cache.get_current(
                gate["resolved_camera"], gate["frame_id"]
            )
            image_height, image_width = frame.rgb.shape[:2]
            center = np.asarray(gate.get("center_uv"), dtype=np.float64).reshape(2)
            image_center = np.asarray(
                [(image_width - 1) / 2.0, (image_height - 1) / 2.0],
                dtype=np.float64,
            )
            radial_error_px = float(np.linalg.norm(center - image_center))
            relaxed_center_limit_px = 0.15 * float(min(image_width, image_height))
            measured = geometry.get("geometry", geometry)
            relaxed_geometry_pass = bool(
                0.03 - 1e-9
                <= float(measured.get("axial_standoff_m", -np.inf))
                <= 0.12 + 1e-9
            )
            if not (
                gate.get("button_visible") is True
                and gate.get("face_class") == BUTTON_FACE_CLASS
                and gate.get("env_step") == current_step
                and projection.get("gate_id") == gate.get("gate_id")
                and projection.get("env_step") == current_step
                and geometry.get("projection_id") == projection.get("projection_id")
                and geometry.get("env_step") == current_step
                and relaxed_geometry_pass
            ):
                raise RuntimeError(
                    "user-reviewed pre-press evidence is stale or outside relaxed bounds"
                )
            stability = self._prepress_stability_snapshot()
            if not stability["stable"] or int(stability["press_contact_count"]) != 0:
                raise RuntimeError(
                    "user-reviewed checkpoint requires stable held radio and zero press contact"
                )
            visual_paths = self._checkpoint_visual_evidence(
                checkpoint_name=checkpoint_name,
                held_hand=context["held_hand"],
                press_hand=context["press_hand"],
            )
            payload = self._robot_state_checkpoint_payload(
                checkpoint_name=checkpoint_name,
                stage=stage,
                held_hand=context["held_hand"],
                press_hand=context["press_hand"],
                object_name=context["object_name"],
                require_current_grasp=True,
                validation_evidence={"prepress_stability": stability},
            )
            payload["prepress"] = {
                "source_checkpoint_path": str(path1),
                "source_checkpoint_sha256": sha1,
                "button_gate": gate,
                "button_projection": projection,
                "geometry": geometry,
                "motion": motion,
                "user_review_override": {
                    "accepted": True,
                    "review_note": review_note.strip(),
                    "radial_error_px": radial_error_px,
                    "center_limit_px": relaxed_center_limit_px,
                    "line_distance_limit_m": None,
                    "line_distance_policy": "not_enforced_by_user_visual_override",
                    "radial_center_policy": "not_enforced_by_user_visual_override",
                    "visual_contact_policy": (
                        "fully_closed_press_fingertips_near_red_button_center_"
                        "and_inside_black_disk"
                    ),
                    "opposition_angle_limit_deg": None,
                    "opposition_angle_policy": (
                        "not_enforced_by_user_visual_override"
                    ),
                    "relaxed_geometry_pass": relaxed_geometry_pass,
                    "strict_geometry_pass": bool(
                        measured.get("geometry_pass", False)
                    ),
                },
            }
            payload["visual_evidence"] = visual_paths
            path2 = self._output_dir / "state_checkpoints" / f"{checkpoint_name}.json"
            _write_json_atomic(path2, payload)
            if _sha256_file(path1) != sha1:
                path2.unlink(missing_ok=True)
                raise RuntimeError("state_checkpoint_1 changed during checkpoint2 save")
            return {
                "_finish": False,
                "primitive_success": True,
                "task_success": bool(_raw_success(self._last_info)),
                "official_success_source": 'info["done"]["success"]',
                "stop_reason": "saved_user_reviewed_prepress_checkpoint",
                "state_checkpoint_1_path": str(path1),
                "state_checkpoint_1_sha256": sha1,
                "state_checkpoint_2_path": str(path2),
                "state_checkpoint_2_sha256": _sha256_file(path2),
                "held_hand": checkpoint1["held_hand"],
                "press_hand": checkpoint1["press_hand"],
                "object_name": checkpoint1["object_name"],
                "visual_review": visual_paths,
                "geometry": _wire_safe(geometry),
                "user_review_override": payload["prepress"]["user_review_override"],
                "total_env_steps": current_step,
            }
        if not all(
            isinstance(value, dict)
            for value in (context, gate, projection, geometry, motion)
        ):
            raise RuntimeError(
                "pre-press gate, projection, geometry, motion, and review are incomplete"
            )
        assert (
            isinstance(context, dict)
            and isinstance(gate, dict)
            and isinstance(projection, dict)
        )
        assert isinstance(geometry, dict) and isinstance(motion, dict)
        if not (
            gate.get("button_visible") is True
            and gate.get("face_class") == BUTTON_FACE_CLASS
            and gate.get("env_step") == current_step
            and projection.get("gate_id") == gate.get("gate_id")
            and projection.get("env_step") == current_step
            and geometry.get("projection_id") == projection.get("projection_id")
            and geometry.get("env_step") == current_step
            and geometry.get("geometry_pass") is True
            and motion.get("primitive_success") is True
            and motion.get("end_env_step") == current_step
            and motion.get("three_view_observed_by_vlm") is True
            and motion.get("visual_review_pass") is True
        ):
            raise RuntimeError(
                "latest pre-press evidence chain did not pass at current env step"
            )
        stability = self._prepress_stability_snapshot()
        if not stability["stable"]:
            raise RuntimeError("radio is not stably held at checkpoint2 finalization")
        payload = self._robot_state_checkpoint_payload(
            checkpoint_name=checkpoint_name,
            stage=stage,
            held_hand=context["held_hand"],
            press_hand=context["press_hand"],
            object_name=context["object_name"],
            require_current_grasp=True,
            validation_evidence={"prepress_stability": stability},
        )
        payload["prepress"] = {
            "source_checkpoint_path": str(path1),
            "source_checkpoint_sha256": sha1,
            "button_gate": gate,
            "button_projection": projection,
            "geometry": geometry,
            "motion": motion,
        }
        payload["visual_evidence"] = dict(motion["visual_review_paths"])
        path2 = self._output_dir / "state_checkpoints" / f"{checkpoint_name}.json"
        _write_json_atomic(path2, payload)
        if _sha256_file(path1) != sha1:
            path2.unlink(missing_ok=True)
            raise RuntimeError("state_checkpoint_1 changed during checkpoint2 save")
        return {
            "_finish": False,
            "primitive_success": True,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "saved_prepress_checkpoint",
            "state_checkpoint_1_path": str(path1),
            "state_checkpoint_1_sha256": sha1,
            "state_checkpoint_2_path": str(path2),
            "state_checkpoint_2_sha256": _sha256_file(path2),
            "held_hand": checkpoint1["held_hand"],
            "press_hand": checkpoint1["press_hand"],
            "object_name": checkpoint1["object_name"],
            "visual_review": motion["visual_review_paths"],
            "video_path": motion["video_path"],
            "trace_path": motion["trace_path"],
            "geometry": _wire_safe(geometry),
            "total_env_steps": current_step,
        }

    def navigate_to(
        self,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        standoff_m: float = 0.85,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        result = self._require_planner().navigate_to(
            hand=hand,
            target_xyz=target_xyz,
            frame=frame,
            standoff_m=standoff_m,
            timeout_s=timeout_s,
        )
        return self._planner_result_with_accounting(result)

    def move_to(
        self,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        target_quat_xyzw: Any | None = None,
        plan_only: bool = False,
        position_tolerance_m: float = 0.02,
        orientation_tolerance_rad: float = 0.087,
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        result = self._require_planner().move_to(
            hand=hand,
            target_xyz=target_xyz,
            frame=frame,
            target_quat_xyzw=target_quat_xyzw,
            plan_only=plan_only,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            timeout_s=timeout_s,
        )
        return self._planner_result_with_accounting(result)

    def pick(
        self,
        hand: str,
        target_xyz: Any,
        approach_vector: Any | None = None,
        grasp_quat_xyzw: Any | None = None,
        pregrasp_offset_m: float = 0.08,
        lift_m: float = 0.08,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        result = self._require_planner().pick(
            hand=hand,
            target_xyz=target_xyz,
            approach_vector=approach_vector,
            grasp_quat_xyzw=grasp_quat_xyzw,
            pregrasp_offset_m=pregrasp_offset_m,
            lift_m=lift_m,
            timeout_s=timeout_s,
        )
        return self._planner_result_with_accounting(result)

    def rotate_wrist(
        self,
        hand: str,
        target_quat_xyzw: Any | None = None,
        relative_axis_angle: Any | None = None,
        frame: str = "world",
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        result = self._require_planner().rotate_wrist(
            hand=hand,
            target_quat_xyzw=target_quat_xyzw,
            relative_axis_angle=relative_axis_angle,
            frame=frame,
            timeout_s=timeout_s,
        )
        return self._planner_result_with_accounting(result)

    def press(
        self,
        hand: str,
        target_xyz: Any,
        press_direction: Any | None = None,
        approach_distance_m: float = 0.04,
        press_depth_m: float = 0.012,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        result = self._require_planner().press(
            hand=hand,
            target_xyz=target_xyz,
            press_direction=press_direction,
            approach_distance_m=approach_distance_m,
            press_depth_m=press_depth_m,
            timeout_s=timeout_s,
        )
        return self._planner_result_with_accounting(result)

    def post_pick_close_press_gripper(self, timeout_s: float = 30.0) -> dict[str, Any]:
        """Close the dynamic press gripper before any stage-3 press motion."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        if _raw_success(self._last_info):
            raise RuntimeError("press gripper close must precede official task success")
        stability_before = self._prepress_stability_snapshot()
        if not stability_before["stable"]:
            raise RuntimeError("radio is not stably held before press-gripper close")
        planner = self._require_planner()
        result = planner._gripper_command(
            hand,
            opening=0.0,
            timeout_s=float(timeout_s),
        )
        opening = self._physical_gripper_opening(hand)
        latch = float(self._gripper_latch[hand])
        stability_after = self._prepress_stability_snapshot()
        closed = bool(
            result.get("primitive_success")
            and latch <= -0.99
            and opening <= 0.003
            and stability_after["stable"]
        )
        # Closing advances physics, so every prior camera/depth binding is stale.
        self._prepress_gate = self._prepress_projection = None
        self._prepress_geometry = None
        self._prepress_plan_certificate = None
        return {
            "_finish": False,
            "primitive_success": closed,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": (
                "press_gripper_closed"
                if closed
                else result.get("stop_reason", "press_gripper_close_not_confirmed")
            ),
            "press_hand": hand,
            "press_gripper_command": latch,
            "press_gripper_opening_m": float(opening),
            "planner_result": _wire_safe(result),
            "radio_held_stability_before": _wire_safe(stability_before),
            "radio_held_stability_after": _wire_safe(stability_after),
            "fresh_press_wrist_projection_required": True,
            "total_env_steps": int(self._env_steps),
        }

    def post_pick_direct_align(
        self,
        *,
        projection_id: str,
        standoff_m: float = 0.04,
        max_travel_m: float = 0.075,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Align the closed press hand to a fresh button projection without contact."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        projection = self._prepress_projection
        if not isinstance(projection, dict):
            raise RuntimeError("fresh press-wrist projection is required")
        if (
            projection.get("projection_id") != projection_id
            or projection.get("env_step") != int(self._env_steps)
            or projection.get("camera") != "press_wrist"
        ):
            raise RuntimeError("direct alignment projection is stale or not press-wrist bound")
        standoff = float(standoff_m)
        travel_limit = float(max_travel_m)
        if not 0.03 <= standoff <= 0.06:
            raise ValueError("standoff_m must lie within [0.03, 0.06]")
        if not 0.01 <= travel_limit <= 0.08:
            raise ValueError("max_travel_m must lie within [0.01, 0.08]")
        opening = self._physical_gripper_opening(hand)
        latch = float(self._gripper_latch[hand])
        if latch > -0.99 or opening > 0.003:
            raise RuntimeError("press-hand gripper is not confirmed fully closed")
        stability_before = self._prepress_stability_snapshot()
        if (
            not stability_before["stable"]
            or int(stability_before["press_contact_count"]) != 0
        ):
            raise RuntimeError(
                "direct alignment requires stable radio and zero press-hand contact"
            )
        planner = self._require_planner()
        current = planner.backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("press-hand EEF pose is unavailable")
        generated = generate_press_staging_pose_candidates(
            button_center_world=projection["button_center_world"],
            button_normal_world=projection["button_normal_world"],
            world_press_transform=pose_matrix_xyzw(current[0], current[1]),
            standoff_m=standoff,
            max_candidates=8,
            alignment_phase="final",
        )
        candidates = [
            candidate
            for candidate in generated["candidates"]
            if candidate.get("eligible") is True
        ]
        if not candidates:
            raise RuntimeError("no eligible direct press-alignment candidate")
        selected = min(
            candidates,
            key=lambda candidate: _quaternion_angle_rad(
                current[1], candidate["target_press_eef_pose"]["quat_xyzw"]
            ),
        )
        target_pose = selected["target_press_eef_pose"]
        target = np.asarray(target_pose["position"], dtype=np.float64).reshape(3)
        start = np.asarray(current[0], dtype=np.float64).reshape(3)
        delta = target - start
        travel = float(np.linalg.norm(delta))
        if not np.isfinite(travel) or travel > travel_limit + 1e-9:
            raise RuntimeError(
                f"direct alignment travel {travel:.6f} m exceeds limit "
                f"{travel_limit:.6f} m"
            )
        direction = delta / max(travel, 1e-12)
        button = np.asarray(
            projection["button_center_world"], dtype=np.float64
        ).reshape(3)
        result = planner._guarded_incremental_move(
            hand=hand,
            target_xyz=target,
            target_quat_xyzw=target_pose["quat_xyzw"],
            direction=direction,
            allow_expected_contact=False,
            position_tolerance_m=0.003,
            timeout_s=float(timeout_s),
            require_expected_contact=False,
            contact_target_xyz=button,
            terminal_hold_steps_required=4,
            stop_on_expected_contact=False,
            ignore_collision_checks=True,
        )
        return {
            **self._planner_result_with_accounting(result),
            "_finish": False,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "press_hand": hand,
            "projection_id": projection_id,
            "standoff_m": standoff,
            "direct_travel_m": travel,
            "selected_candidate": _wire_safe(selected),
            "collision_checks_skipped": True,
            "collision_skip_scope": "curobo_precheck_and_runtime_world+self_reports",
            "press_gripper_command": latch,
            "press_gripper_opening_m": float(opening),
            "total_env_steps": int(self._env_steps),
        }

    def post_pick_direct_press(
        self,
        *,
        projection_id: str,
        press_depth_m: float = 0.012,
        max_travel_m: float = 0.075,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Directly press from a fresh wrist projection with collision gates off."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        projection = self._prepress_projection
        geometry = self._prepress_geometry
        if not isinstance(projection, dict) or not isinstance(geometry, dict):
            raise RuntimeError("fresh press-wrist projection and geometry are required")
        if (
            projection.get("projection_id") != projection_id
            or projection.get("env_step") != int(self._env_steps)
            or geometry.get("projection_id") != projection_id
            or geometry.get("env_step") != int(self._env_steps)
            or projection.get("camera") != "press_wrist"
        ):
            raise RuntimeError("direct press projection is stale or not press-wrist bound")
        checkpoint2_path = (
            self._output_dir / "state_checkpoints" / "state_checkpoint_2.json"
        )
        if not checkpoint2_path.is_file():
            raise RuntimeError("state_checkpoint_2 must be saved before direct press")
        checkpoint2 = json.loads(checkpoint2_path.read_text(encoding="utf-8"))
        checkpoint2_projection = (
            checkpoint2.get("prepress", {}).get("button_projection", {})
            if isinstance(checkpoint2, dict)
            else {}
        )
        if checkpoint2_projection.get("projection_id") != projection_id:
            raise RuntimeError("state_checkpoint_2 is not bound to the fresh projection")
        press_depth = float(press_depth_m)
        travel_limit = float(max_travel_m)
        if not 0.004 <= press_depth <= 0.03:
            raise ValueError("press_depth_m must lie within [0.004, 0.03]")
        if not 0.03 <= travel_limit <= 0.08:
            raise ValueError("max_travel_m must lie within [0.03, 0.08]")
        opening = self._physical_gripper_opening(hand)
        latch = float(self._gripper_latch[hand])
        if latch > -0.99 or opening > 0.003:
            raise RuntimeError("press-hand gripper is not confirmed fully closed")
        stability_before = self._prepress_stability_snapshot()
        if not stability_before["stable"]:
            raise RuntimeError("radio is not stably held before direct press")
        button = np.asarray(
            projection["button_center_world"], dtype=np.float64
        ).reshape(3)
        outward = np.asarray(
            projection["button_normal_world"], dtype=np.float64
        ).reshape(3)
        outward_norm = float(np.linalg.norm(outward))
        if not np.isfinite(outward).all() or outward_norm < 1e-9:
            raise RuntimeError("fresh button normal is invalid")
        outward /= outward_norm
        planner = self._require_planner()
        current = planner.backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("press-hand EEF pose is unavailable")
        target = (
            button
            - direction * PRESS_EEF_TO_CONTACT_OFFSET_M
            + direction * press_depth
        )
        travel = float(np.linalg.norm(target - np.asarray(current[0], dtype=np.float64)))
        if not np.isfinite(travel) or travel > travel_limit + 1e-9:
            raise RuntimeError(
                f"direct press travel {travel:.6f} m exceeds limit {travel_limit:.6f} m"
            )
        result = planner._guarded_incremental_move(
            hand=hand,
            target_xyz=target,
            target_quat_xyzw=current[1],
            direction=direction,
            allow_expected_contact=True,
            position_tolerance_m=0.012,
            timeout_s=float(timeout_s),
            require_expected_contact=True,
            contact_target_xyz=button,
            terminal_hold_steps_required=4,
            stop_on_expected_contact=False,
            eef_to_contact_vector=direction * PRESS_EEF_TO_CONTACT_OFFSET_M,
            ignore_collision_checks=True,
        )
        return {
            **self._planner_result_with_accounting(result),
            "_finish": False,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "press_hand": hand,
            "projection_id": projection_id,
            "button_center_world": button.tolist(),
            "button_normal_world": outward.tolist(),
            "press_direction_world": direction.tolist(),
            "press_depth_m": press_depth,
            "direct_travel_m": travel,
            "collision_checks_skipped": True,
            "collision_skip_scope": "curobo_precheck_and_runtime_world+self_reports",
            "radio_held_stability_before": _wire_safe(stability_before),
            "press_gripper_command": latch,
            "press_gripper_opening_m": float(opening),
            "total_env_steps": int(self._env_steps),
        }

    def post_pick_direct_advance(
        self,
        *,
        projection_id: str,
        advance_m: float = 0.03,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Advance the closed press hand along its live camera-reviewed axis."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        projection = self._prepress_projection
        if not isinstance(projection, dict):
            raise RuntimeError("fresh press-wrist projection is required")
        if (
            projection.get("projection_id") != projection_id
            or projection.get("env_step") != int(self._env_steps)
            or projection.get("camera") != "press_wrist"
        ):
            raise RuntimeError("direct advance projection is stale or not press-wrist bound")
        checkpoint2_path = (
            self._output_dir / "state_checkpoints" / "state_checkpoint_2.json"
        )
        if not checkpoint2_path.is_file():
            raise RuntimeError("state_checkpoint_2 must be saved before direct advance")
        checkpoint2 = json.loads(checkpoint2_path.read_text(encoding="utf-8"))
        checkpoint2_projection = checkpoint2.get("prepress", {}).get(
            "button_projection", {}
        )
        if checkpoint2_projection.get("projection_id") != projection_id:
            raise RuntimeError("state_checkpoint_2 is not bound to the fresh projection")
        advance = float(advance_m)
        if not 0.004 <= advance <= 0.10:
            raise ValueError("advance_m must lie within [0.004, 0.10]")
        opening = self._physical_gripper_opening(hand)
        latch = float(self._gripper_latch[hand])
        if latch > -0.99 or opening > 0.003:
            raise RuntimeError("press-hand gripper is not confirmed fully closed")
        stability_before = self._prepress_stability_snapshot()
        if not stability_before["stable"]:
            raise RuntimeError("radio is not stably held before direct advance")
        outward = np.asarray(
            projection["button_normal_world"], dtype=np.float64
        ).reshape(3)
        outward_norm = float(np.linalg.norm(outward))
        if not np.isfinite(outward).all() or outward_norm < 1e-9:
            raise RuntimeError("fresh button normal is invalid")
        outward /= outward_norm
        button = np.asarray(
            projection["button_center_world"], dtype=np.float64
        ).reshape(3)
        planner = self._require_planner()
        current = planner.backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("press-hand EEF pose is unavailable")
        direction = np.asarray(
            quat_rotate_xyzw(current[1], [0.0, 0.0, 1.0]), dtype=np.float64
        ).reshape(3)
        direction_norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or direction_norm < 1e-9:
            raise RuntimeError("press-hand live approach axis is invalid")
        direction /= direction_norm
        target = np.asarray(current[0], dtype=np.float64) + direction * advance
        result = planner._guarded_incremental_move(
            hand=hand,
            target_xyz=target,
            target_quat_xyzw=current[1],
            direction=direction,
            allow_expected_contact=True,
            position_tolerance_m=0.012,
            timeout_s=float(timeout_s),
            require_expected_contact=True,
            contact_target_xyz=button,
            terminal_hold_steps_required=4,
            stop_on_expected_contact=False,
            eef_to_contact_vector=direction * PRESS_EEF_TO_CONTACT_OFFSET_M,
            ignore_collision_checks=True,
        )
        return {
            **self._planner_result_with_accounting(result),
            "_finish": False,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "press_hand": hand,
            "projection_id": projection_id,
            "button_center_world": button.tolist(),
            "button_normal_world": outward.tolist(),
            "press_direction_world": direction.tolist(),
            "press_direction_source": "live_press_eef_local_positive_z",
            "relative_advance_m": advance,
            "collision_checks_skipped": True,
            "collision_skip_scope": "curobo_precheck_and_runtime_world+self_reports",
            "radio_held_stability_before": _wire_safe(stability_before),
            "press_gripper_command": latch,
            "press_gripper_opening_m": float(opening),
            "total_env_steps": int(self._env_steps),
        }

    def post_pick_recenter_held_button(
        self,
        *,
        target_finger_standoff_m: float = 0.04,
        max_held_travel_m: float = 0.08,
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        """Translate the held radio so its red marker lies on the press axis."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        held, press = checkpoint["held_hand"], checkpoint["press_hand"]
        standoff = float(target_finger_standoff_m)
        travel_limit = float(max_held_travel_m)
        if not 0.03 <= standoff <= 0.07:
            raise ValueError("target_finger_standoff_m must lie within [0.03, 0.07]")
        if not 0.02 <= travel_limit <= 0.10:
            raise ValueError("max_held_travel_m must lie within [0.02, 0.10]")
        press_opening = self._physical_gripper_opening(press)
        press_latch = float(self._gripper_latch[press])
        if press_latch > -0.99 or press_opening > 0.003:
            raise RuntimeError("press-hand gripper is not confirmed fully closed")
        stability_before = self._prepress_stability_snapshot()
        if (
            not stability_before["stable"]
            or int(stability_before["press_contact_count"]) != 0
        ):
            raise RuntimeError(
                "held-button recenter requires stable radio and zero press contact"
            )
        geometry_before = self._live_toggle_geometry(press)
        selected = min(
            geometry_before["fingers"],
            key=lambda finger: finger["center_distance_to_marker_m"],
        )
        marker = np.asarray(
            geometry_before["marker_position_world"], dtype=np.float64
        ).reshape(3)
        finger = np.asarray(selected["position_world"], dtype=np.float64).reshape(3)
        planner = self._require_planner()
        press_pose = planner.backend.get_eef_pose(press)
        held_pose = planner.backend.get_eef_pose(held)
        if press_pose is None or held_pose is None:
            raise RuntimeError("held or press EEF pose is unavailable")
        press_axis = np.asarray(
            quat_rotate_xyzw(press_pose[1], [0.0, 0.0, 1.0]), dtype=np.float64
        ).reshape(3)
        press_axis /= max(float(np.linalg.norm(press_axis)), 1e-12)
        desired_marker = finger + press_axis * standoff
        translation = desired_marker - marker
        travel = float(np.linalg.norm(translation))
        if not np.isfinite(travel) or travel > travel_limit + 1e-9:
            raise RuntimeError(
                f"held-button recenter travel {travel:.6f} m exceeds limit "
                f"{travel_limit:.6f} m"
            )
        held_start = np.asarray(held_pose[0], dtype=np.float64).reshape(3)
        attached_obj = planner.backend.get_attached_object(held)
        plan = planner.backend.plan_guarded_ik_step(
            hand=held,
            target_xyz=held_start + translation,
            target_quat_xyzw=held_pose[1],
            timeout_s=float(timeout_s),
            attached_obj=attached_obj,
            contact_target_xyz=marker,
            ignore_collision_checks=True,
            full_solution=True,
        )
        if not plan.get("ok") or plan.get("joint_trajectory") is None:
            raise RuntimeError(
                "held-button recenter IK failed: "
                f"{plan.get('stop_reason', 'unknown')}"
            )
        result = planner._execute_actions(
            None,
            hand=held,
            target_xyz=held_start + translation,
            target_quat_xyzw=np.asarray(held_pose[1], dtype=np.float64),
            position_tolerance_m=0.004,
            orientation_tolerance_rad=0.087,
            timeout_s=float(timeout_s),
            require_pose=True,
            hold_steps_required=4,
            joint_trajectory=plan["joint_trajectory"],
            ignore_collision_checks=True,
        )
        # Any motion invalidates every prior camera/depth binding.
        self._prepress_gate = self._prepress_projection = None
        self._prepress_geometry = None
        self._prepress_plan_certificate = None
        stability_after = self._prepress_stability_snapshot()
        geometry_after = self._live_toggle_geometry(press)
        selected_after = min(
            geometry_after["fingers"],
            key=lambda item: item["center_distance_to_marker_m"],
        )
        marker_after = np.asarray(
            geometry_after["marker_position_world"], dtype=np.float64
        ).reshape(3)
        finger_after = np.asarray(
            selected_after["position_world"], dtype=np.float64
        ).reshape(3)
        offset_after = marker_after - finger_after
        axial_after = float(np.dot(offset_after, press_axis))
        lateral_after = float(
            np.linalg.norm(offset_after - press_axis * axial_after)
        )
        return {
            **self._planner_result_with_accounting(result),
            "_finish": False,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "held_hand": held,
            "press_hand": press,
            "selected_press_finger": {
                key: _wire_safe(value)
                for key, value in selected_after.items()
                if key != "link"
            },
            "press_axis_world": press_axis.tolist(),
            "desired_marker_world": desired_marker.tolist(),
            "held_translation_world": translation.tolist(),
            "held_travel_m": travel,
            "marker_to_press_axis_lateral_error_m": lateral_after,
            "marker_axial_standoff_from_finger_m": axial_after,
            "target_finger_standoff_m": standoff,
            "collision_checks_skipped": True,
            "press_gripper_command": press_latch,
            "press_gripper_opening_m": float(press_opening),
            "radio_held_stability_before": _wire_safe(stability_before),
            "radio_held_stability_after": _wire_safe(stability_after),
            "fresh_press_wrist_projection_required": True,
            "total_env_steps": int(self._env_steps),
        }

    def _live_toggle_geometry(self, hand: str) -> dict[str, Any]:
        """Return the live toggle marker and press-finger geometry."""

        from omnigibson.object_states.toggle import ToggledOn

        radio, _table = self._resolve_handoff_targets()
        toggle = getattr(radio, "states", {}).get(ToggledOn)
        if toggle is None or toggle.visual_marker is None:
            raise RuntimeError("radio ToggledOn visual marker is unavailable")
        marker_position, _marker_quat = toggle.visual_marker.get_position_orientation()
        marker = np.asarray(_numpy_tree(marker_position), dtype=np.float64).reshape(3)
        marker_radius = float(
            np.min(
                np.asarray(
                    _numpy_tree(toggle.visual_marker.extent * toggle.visual_marker.scale),
                    dtype=np.float64,
                )
            )
        )
        robot = self._robot()
        if robot is None:
            raise RuntimeError("robot is unavailable for toggle geometry")
        finger_map = getattr(robot, "finger_links", {}) or {}
        links = finger_map.get(hand, []) if isinstance(finger_map, dict) else []
        if not links:
            raise RuntimeError(f"{hand} finger links are unavailable")
        fingers = []
        for index, link in enumerate(links):
            position, _quat = self._object_pose(link)
            position = np.asarray(position, dtype=np.float64).reshape(3)
            fingers.append(
                {
                    "index": index,
                    "name": str(getattr(link, "name", f"{hand}_finger_{index}")),
                    "prim_path": str(getattr(link, "prim_path", "")),
                    "position_world": position,
                    "center_distance_to_marker_m": float(
                        np.linalg.norm(position - marker)
                    ),
                    "link": link,
                }
            )
        contact_objects = getattr(type(toggle), "_finger_contact_objs", set()) or set()
        return {
            "radio": radio,
            "toggle": toggle,
            "marker_position_world": marker,
            "marker_radius_m": marker_radius,
            "fingers": fingers,
            "radio_in_global_finger_contact_set": radio in contact_objects,
        }

    def inspect_toggle_geometry(self) -> dict[str, Any]:
        """Inspect the exact physical trigger geometry for the held radio."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        geometry = self._live_toggle_geometry(hand)
        toggle = geometry["toggle"]
        fingers = [
            {key: _wire_safe(value) for key, value in finger.items() if key != "link"}
            for finger in geometry["fingers"]
        ]
        return {
            "_finish": False,
            "primitive_success": True,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "press_hand": hand,
            "toggled_on": bool(toggle.get_value()),
            "robot_can_toggle_steps": int(toggle.robot_can_toggle_steps),
            "required_toggle_steps": 5,
            "marker_position_world": geometry["marker_position_world"].tolist(),
            "marker_radius_m": float(geometry["marker_radius_m"]),
            "radio_in_global_finger_contact_set": bool(
                geometry["radio_in_global_finger_contact_set"]
            ),
            "press_fingers": fingers,
            "total_env_steps": int(self._env_steps),
        }

    def post_pick_direct_finger_toggle(
        self,
        *,
        projection_id: str,
        penetration_m: float = 0.008,
        max_travel_m: float = 0.15,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """Move the nearest closed press finger through the exact toggle marker."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        projection = self._prepress_projection
        if not isinstance(projection, dict) or (
            projection.get("projection_id") != projection_id
            or projection.get("env_step") != int(self._env_steps)
            or projection.get("camera") != "press_wrist"
        ):
            raise RuntimeError("fresh press-wrist projection is required")
        checkpoint2_path = (
            self._output_dir / "state_checkpoints" / "state_checkpoint_2.json"
        )
        if not checkpoint2_path.is_file():
            raise RuntimeError("state_checkpoint_2 must be saved before toggle press")
        checkpoint2 = json.loads(checkpoint2_path.read_text(encoding="utf-8"))
        checkpoint2_projection = checkpoint2.get("prepress", {}).get(
            "button_projection", {}
        )
        if checkpoint2_projection.get("projection_id") != projection_id:
            raise RuntimeError("state_checkpoint_2 is not bound to the fresh projection")
        penetration = float(penetration_m)
        travel_limit = float(max_travel_m)
        if not 0.0 <= penetration <= 0.03:
            raise ValueError("penetration_m must lie within [0.0, 0.03]")
        if not 0.02 <= travel_limit <= 0.20:
            raise ValueError("max_travel_m must lie within [0.02, 0.20]")
        opening = self._physical_gripper_opening(hand)
        latch = float(self._gripper_latch[hand])
        if latch > -0.99 or opening > 0.003:
            raise RuntimeError("press-hand gripper is not confirmed fully closed")
        stability_before = self._prepress_stability_snapshot()
        if not stability_before["stable"]:
            raise RuntimeError("radio is not stably held before toggle press")
        geometry_before = self._live_toggle_geometry(hand)
        selected = min(
            geometry_before["fingers"],
            key=lambda finger: finger["center_distance_to_marker_m"],
        )
        marker = np.asarray(
            geometry_before["marker_position_world"], dtype=np.float64
        ).reshape(3)
        finger_position = np.asarray(
            selected["position_world"], dtype=np.float64
        ).reshape(3)
        planner = self._require_planner()
        current = planner.backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("press-hand EEF pose is unavailable")
        current_eef = np.asarray(current[0], dtype=np.float64).reshape(3)
        press_axis = np.asarray(
            quat_rotate_xyzw(current[1], [0.0, 0.0, 1.0]), dtype=np.float64
        ).reshape(3)
        press_axis /= max(float(np.linalg.norm(press_axis)), 1e-12)
        target_finger = marker + press_axis * penetration
        translation = target_finger - finger_position
        travel = float(np.linalg.norm(translation))
        if not np.isfinite(travel) or travel > travel_limit + 1e-9:
            raise RuntimeError(
                f"finger-to-toggle travel {travel:.6f} m exceeds limit "
                f"{travel_limit:.6f} m"
            )
        eef_to_finger = finger_position - current_eef
        plan = planner.backend.plan_guarded_ik_step(
            hand=hand,
            target_xyz=current_eef + translation,
            target_quat_xyzw=current[1],
            timeout_s=float(timeout_s),
            contact_target_xyz=marker,
            ignore_collision_checks=True,
            full_solution=True,
        )
        if not plan.get("ok"):
            raise RuntimeError(
                "full finger-center IK failed: "
                f"{plan.get('stop_reason', 'unknown')}"
            )
        q_path = plan.get("joint_trajectory")
        if q_path is None:
            raise RuntimeError("full finger-center IK omitted a joint trajectory")
        result = planner._execute_actions(
            None,
            hand=hand,
            target_xyz=current_eef + translation,
            target_quat_xyzw=np.asarray(current[1], dtype=np.float64),
            position_tolerance_m=0.003,
            orientation_tolerance_rad=0.087,
            timeout_s=float(timeout_s),
            require_pose=True,
            hold_steps_required=8,
            contact_target_xyz=marker,
            allow_expected_contact=True,
            allow_guarded_goal_world_collision=True,
            stop_on_expected_contact=False,
            joint_trajectory=q_path,
            eef_to_contact_vector=eef_to_finger,
            ignore_collision_checks=True,
        )
        held_frames = 0
        hold_action = np.asarray(
            planner.backend.hold_action(hand), dtype=np.float32
        ).reshape(23)
        hold_action[ENV_ACTION_SEGMENTS[f"{hand}_gripper"]] = -1.0
        while held_frames < 12 and not _raw_success(self._last_info):
            self.planner_step(hold_action.reshape(1, 23))
            held_frames += 1
        geometry_after = self._live_toggle_geometry(hand)
        toggle_after = geometry_after["toggle"]
        return {
            **self._planner_result_with_accounting(result),
            "_finish": False,
            "primitive_success": bool(_raw_success(self._last_info)),
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "press_hand": hand,
            "projection_id": projection_id,
            "selected_finger": {
                key: _wire_safe(value)
                for key, value in selected.items()
                if key != "link"
            },
            "marker_position_world": marker.tolist(),
            "marker_radius_m": float(geometry_before["marker_radius_m"]),
            "eef_to_finger_vector_world": eef_to_finger.tolist(),
            "penetration_m": penetration,
            "direct_travel_m": travel,
            "guarded_center_target": True,
            "post_contact_hold_frames": held_frames,
            "toggled_on": bool(toggle_after.get_value()),
            "robot_can_toggle_steps": int(toggle_after.robot_can_toggle_steps),
            "radio_in_global_finger_contact_set": bool(
                geometry_after["radio_in_global_finger_contact_set"]
            ),
            "collision_checks_skipped": True,
            "press_gripper_command": latch,
            "press_gripper_opening_m": float(opening),
            "total_env_steps": int(self._env_steps),
        }

    def post_pick_visual_servo_align(
        self,
        *,
        projection_id: str,
        desired_uv: Any,
        max_travel_m: float = 0.12,
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        """Translate the closed press hand so the button reaches a wrist pixel."""

        _path, checkpoint = self._assert_prepress_checkpoint_bound()
        hand = checkpoint["press_hand"]
        projection = self._prepress_projection
        gate = self._prepress_gate
        if not isinstance(projection, dict) or not isinstance(gate, dict):
            raise RuntimeError("fresh press-wrist gate and projection are required")
        if (
            projection.get("projection_id") != projection_id
            or projection.get("env_step") != int(self._env_steps)
            or projection.get("camera") != "press_wrist"
            or projection.get("frame_id") != gate.get("frame_id")
        ):
            raise RuntimeError("visual-servo projection is stale or not press-wrist bound")
        uv = np.asarray(desired_uv, dtype=np.float64).reshape(2)
        frame = self._frame_cache.get_current(
            projection["resolved_camera"], projection["frame_id"]
        )
        if (
            not np.isfinite(uv).all()
            or not 0.0 <= uv[0] < frame.intrinsics.width
            or not 0.0 <= uv[1] < frame.intrinsics.height
        ):
            raise ValueError("desired_uv is outside the press-wrist image")
        travel_limit = float(max_travel_m)
        if not 0.01 <= travel_limit <= 0.15:
            raise ValueError("max_travel_m must lie within [0.01, 0.15]")
        opening = self._physical_gripper_opening(hand)
        latch = float(self._gripper_latch[hand])
        if latch > -0.99 or opening > 0.003:
            raise RuntimeError("press-hand gripper is not confirmed fully closed")
        stability_before = self._prepress_stability_snapshot()
        if (
            not stability_before["stable"]
            or int(stability_before["press_contact_count"]) != 0
        ):
            raise RuntimeError(
                "visual servo requires stable radio and zero press-hand contact"
            )
        button = np.asarray(
            projection["button_center_world"], dtype=np.float64
        ).reshape(3)
        world_to_camera = np.linalg.inv(frame.camera_to_world)
        button_camera = transform_point(world_to_camera, button)
        depth_m = -float(button_camera[2])
        if not np.isfinite(depth_m) or depth_m <= 0.0:
            raise RuntimeError("projected button is not in front of the wrist camera")
        desired_camera = camera_point_from_pixel(
            frame.intrinsics,
            u=int(round(float(uv[0]))),
            v=int(round(float(uv[1]))),
            depth_m=depth_m,
        )
        if frame.correction_profile is not None:
            desired_camera = frame.correction_profile.apply_camera_point(
                desired_camera
            )
        desired_sample_world = transform_point(
            frame.camera_to_world, desired_camera
        )
        camera_translation = button - desired_sample_world
        travel = float(np.linalg.norm(camera_translation))
        if not np.isfinite(travel) or travel > travel_limit + 1e-9:
            raise RuntimeError(
                f"visual-servo travel {travel:.6f} m exceeds limit "
                f"{travel_limit:.6f} m"
            )
        planner = self._require_planner()
        current = planner.backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("press-hand EEF pose is unavailable")
        direction = camera_translation / max(travel, 1e-12)
        target = np.asarray(current[0], dtype=np.float64) + camera_translation
        result = planner._guarded_incremental_move(
            hand=hand,
            target_xyz=target,
            target_quat_xyzw=current[1],
            direction=direction,
            allow_expected_contact=False,
            position_tolerance_m=0.003,
            timeout_s=float(timeout_s),
            require_expected_contact=False,
            contact_target_xyz=button,
            terminal_hold_steps_required=4,
            stop_on_expected_contact=False,
            ignore_collision_checks=True,
        )
        return {
            **self._planner_result_with_accounting(result),
            "_finish": False,
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "press_hand": hand,
            "projection_id": projection_id,
            "source_uv": _wire_safe(gate.get("center_uv")),
            "desired_uv": uv.tolist(),
            "button_depth_m": depth_m,
            "camera_translation_world": camera_translation.tolist(),
            "direct_travel_m": travel,
            "collision_checks_skipped": True,
            "collision_skip_scope": "curobo_precheck_and_runtime_world+self_reports",
            "press_gripper_command": latch,
            "press_gripper_opening_m": float(opening),
            "total_env_steps": int(self._env_steps),
        }

    def post_success_hold_frames(self, frames: int = 4) -> dict[str, Any]:
        """Advance a bounded number of stationary frames after official success."""

        if isinstance(frames, bool) or not 1 <= int(frames) <= 16:
            raise ValueError("frames must be an integer in [1, 16]")
        if not _raw_success(self._last_info):
            raise RuntimeError("post-success hold requires official task success")
        self._post_success_cleanup_active = True
        self._done = False
        executed = 0
        try:
            for _ in range(int(frames)):
                action = np.asarray(
                    self._require_planner().backend.hold_action(), dtype=np.float32
                ).reshape(23)
                self.planner_step(action.reshape(1, 23))
                executed += 1
        finally:
            self._post_success_cleanup_active = False
            self._done = bool(_raw_success(self._last_info))
        return {
            "_finish": False,
            "primitive_success": executed == int(frames),
            "task_success": bool(_raw_success(self._last_info)),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "post_success_frames_held",
            "executed_frames": executed,
            "total_env_steps": int(self._env_steps),
        }

    def post_success_retreat_and_open(
        self,
        *,
        hand: str,
        retreat_direction: Any,
        retreat_m: float = 0.05,
        opening: float = 1.0,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Retreat the dynamic press hand first, then open its gripper."""

        _path, checkpoint, _sha256 = self._read_post_pick_checkpoint()
        if hand != checkpoint["press_hand"]:
            raise ValueError("cleanup hand must match dynamic press_hand")
        if not _raw_success(self._last_info):
            raise RuntimeError("post-success cleanup requires official task success")
        direction = np.asarray(retreat_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or norm < 1e-9:
            raise ValueError("retreat_direction must be one finite non-zero vector")
        direction /= norm
        distance = float(retreat_m)
        if not 0.01 <= distance <= 0.10:
            raise ValueError("retreat_m must lie within [0.01, 0.10]")
        planner = self._require_planner()
        current = planner.backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("cannot read press EEF pose for post-success retreat")
        target = np.asarray(current[0], dtype=np.float64) + direction * distance
        toggle_geometry = self._live_toggle_geometry(hand)
        marker = np.asarray(
            toggle_geometry["marker_position_world"], dtype=np.float64
        ).reshape(3)
        selected_finger = min(
            toggle_geometry["fingers"],
            key=lambda item: item["center_distance_to_marker_m"],
        )
        finger_position = np.asarray(
            selected_finger["position_world"], dtype=np.float64
        ).reshape(3)
        eef_to_finger = finger_position - np.asarray(
            current[0], dtype=np.float64
        ).reshape(3)
        marker_to_finger = finger_position - marker
        marker_to_finger /= max(float(np.linalg.norm(marker_to_finger)), 1e-12)
        contact_surface_target = marker + marker_to_finger * float(
            toggle_geometry["marker_radius_m"]
        )
        self._post_success_cleanup_active = True
        self._done = False
        try:
            retreat = planner._guarded_incremental_move(
                hand=hand,
                target_xyz=target,
                target_quat_xyzw=current[1],
                direction=direction,
                allow_expected_contact=True,
                position_tolerance_m=0.006,
                timeout_s=float(timeout_s),
                require_expected_contact=False,
                contact_target_xyz=contact_surface_target,
                terminal_hold_steps_required=2,
                stop_on_expected_contact=False,
                eef_to_contact_vector=eef_to_finger,
                allowed_contact_distance_m=0.04,
            )
            if not retreat.get("primitive_success"):
                return {
                    **retreat,
                    "task_success": bool(_raw_success(self._last_info)),
                    "cleanup_stage": "retreat",
                    "total_env_steps": int(self._env_steps),
                }
            opened = planner.release(
                hand=hand,
                opening=float(opening),
                retreat_vector=None,
                retreat_m=0.0,
                timeout_s=min(30.0, float(timeout_s)),
            )
            return {
                "_finish": False,
                "primitive_success": bool(opened.get("primitive_success")),
                "task_success": bool(_raw_success(self._last_info)),
                "official_success_source": 'info["done"]["success"]',
                "stop_reason": (
                    "post_success_retreated_and_opened"
                    if opened.get("primitive_success")
                    else opened.get("stop_reason", "gripper_open_failed")
                ),
                "retreat": _wire_safe(retreat),
                "gripper_open": _wire_safe(opened),
                "retreat_direction": direction.tolist(),
                "retreat_m": distance,
                "initial_toggle_marker_world": marker.tolist(),
                "initial_toggle_contact_surface_world": (
                    contact_surface_target.tolist()
                ),
                "total_env_steps": int(self._env_steps),
            }
        finally:
            self._post_success_cleanup_active = False
            self._done = bool(_raw_success(self._last_info))

    def release(
        self,
        hand: str,
        opening: float = 1.0,
        retreat_vector: Any | None = None,
        retreat_m: float = 0.03,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        result = self._require_planner().release(
            hand=hand,
            opening=opening,
            retreat_vector=retreat_vector,
            retreat_m=retreat_m,
            timeout_s=timeout_s,
        )
        return self._planner_result_with_accounting(result)

    def close(self) -> None:
        try:
            self._finalize_video_segment()
        finally:
            self._env.close()


_INITIAL_PPID = os.getppid()


def _start_parent_watchdog(
    server: SocketRpcServer,
    shutdown_event: threading.Event,
    env: BehaviorEnvFacade,
) -> None:
    def watch() -> None:
        while not shutdown_event.wait(2.0):
            ppid = os.getppid()
            if ppid != _INITIAL_PPID or ppid == 1:
                failed_runtime_path = env._output_dir / "failed_runtime.json"
                if failed_runtime_path.is_file():
                    try:
                        failed_runtime = json.loads(
                            failed_runtime_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        failed_runtime = {}
                    if (
                        failed_runtime.get("env_pid") == os.getpid()
                        and failed_runtime.get("vla_gate_confirmed") is True
                        and failed_runtime.get("lifecycle_finalized") is True
                    ):
                        continue
                if (
                    env._handoff_state == _HANDOFF_PAUSED
                    and env._paused_runtime_path.is_file()
                ):
                    try:
                        paused = json.loads(
                            env._paused_runtime_path.read_text(encoding="utf-8")
                        )
                    except Exception:
                        paused = {}
                    if paused.get("lifecycle_finalized") is True:
                        continue
                shutdown_event.set()
                threading.Thread(target=server.shutdown, daemon=True).start()
                return

    threading.Thread(target=watch, daemon=True).start()


class _MainThreadDispatcher:
    """Execute simulator RPCs on the thread that created OmniGibson."""

    def __init__(
        self,
        env: BehaviorEnvFacade,
        shutdown_event: threading.Event,
    ) -> None:
        self._env = env
        self._shutdown_event = shutdown_event
        self._calls: Queue[tuple[str, tuple, dict, Future]] = Queue()

    def submit(self, method: str, args: tuple, kwargs: dict) -> Any:
        future: Future = Future()
        self._calls.put((method, args, kwargs, future))
        return future.result()

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method.startswith("env."):
            env_method = method.removeprefix("env.")
            mode = getattr(self._env, "_control_mode", None)
            allowed = _SHARED_ENV_RPC_METHODS | _ENV_RPC_METHODS_BY_MODE.get(
                mode, frozenset()
            )
            if env_method not in allowed:
                raise ValueError(f"unknown BEHAVIOR env RPC method: {method!r}")
            return getattr(self._env, env_method)(*args, **kwargs)
        if method == "shutdown":
            self._shutdown_event.set()
            return {"ok": True}
        raise ValueError(f"unknown RPC method: {method!r}")

    def process_next(self, *, timeout_s: float = 0.5) -> bool:
        try:
            method, args, kwargs, future = self._calls.get(timeout=timeout_s)
        except Empty:
            return False
        try:
            result = self._dispatch(method, args, kwargs)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return True

    def run(self) -> None:
        while not self._shutdown_event.is_set():
            self.process_next()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--activity-definition-id", type=int, required=True)
    parser.add_argument("--activity-instance-id", type=int, required=True)
    parser.add_argument("--activity-instance-dir", required=True)
    parser.add_argument("--scene-model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--control-mode", choices=CONTROL_MODES, required=True)
    parser.add_argument("--transport-host", default="127.0.0.1")
    parser.add_argument("--transport-port", type=int, default=0)
    parser.add_argument(
        "--post-pick-debug-mirror",
        help=(
            "Trusted debug_mirror_post_pick.scene.json (or its bundle directory) "
            "to load after the normal task reset. This is not an official episode resume."
        ),
    )
    parser.add_argument(
        "--controller-switch-smoke",
        action="store_true",
        help=(
            "Run one real velocity-to-position controller reload, eight-step "
            "current-target hold, and cuRobo warmup, then exit without RPC/VLA."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "suite": args.suite,
        "task": args.task,
        "task_name": args.task_name,
        "activity_definition_id": args.activity_definition_id,
        "activity_instance_id": args.activity_instance_id,
        "activity_instance_dir": str(Path(args.activity_instance_dir).resolve()),
        "scene_model": args.scene_model,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
    }
    env = BehaviorEnvFacade(
        cfg=_load_env_config(args),
        meta=meta,
        output_dir=output_dir,
        control_mode=args.control_mode,
        debug_mirror_path=args.post_pick_debug_mirror,
    )
    if args.controller_switch_smoke:
        try:
            result = env.run_controller_switch_smoke()
            print(
                json.dumps(
                    {
                        "event": "controller_switch_smoke_complete",
                        "result_path": result["result_path"],
                    }
                ),
                flush=True,
            )
        finally:
            env.close()
        return
    shutdown_event = threading.Event()
    dispatcher = _MainThreadDispatcher(env, shutdown_event)
    server = SocketRpcServer(
        (args.transport_host, args.transport_port),
        dispatcher.submit,
    )
    bound_host, bound_port = server.server_address
    print(
        json.dumps(
            {
                "event": "transport_ready",
                "kind": "socket",
                "host": "127.0.0.1" if bound_host == "0.0.0.0" else bound_host,
                "port": bound_port,
            }
        ),
        flush=True,
    )
    _start_parent_watchdog(server, shutdown_event, env)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        dispatcher.run()
    finally:
        try:
            env.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
