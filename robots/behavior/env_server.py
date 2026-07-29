"""OmniGibson/R1Pro process for the BEHAVIOR RPent runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import secrets
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

import numpy as np

from robots.behavior.camera_geometry import (
    HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M,
    HAND_GEOMETRY_ROTATION_TOLERANCE_DEG,
    HAND_GEOMETRY_SYNC_RENDER_ITERATIONS,
    HAND_GEOMETRY_TRANSLATION_TOLERANCE_M,
    CameraGeometryError,
    CameraIntrinsics,
    FrameCache,
    camera_point_from_pixel,
    canonical_camera,
    frame_bound_hand_distance_report,
    hand_geometry_sync_certificate_is_valid,
    load_camera_correction_profiles,
    r1pro_wrist_camera_reference_transforms,
    rigid_transform_residual,
    robust_depth_sample,
    validated_rigid_transform,
)
from robots.behavior.planner_executor import (
    CuroboPlanningError,
    PlannerExecutor,
    _quat_rotate_vector_xyzw,
)
from robots.behavior.schemas import (
    BASE_ROTATION_STEP_RAD,
    BASE_TRANSLATION_STEP_M,
    EEF_TRANSLATION_STEP_M,
    ENV_ACTION_SEGMENTS,
    POLICY_STATE_SEGMENTS,
    RAW_PROPRIO_SEGMENTS,
    TORSO_VERTICAL_STEP_M,
    WRIST_ROTATION_STEP_RAD,
    extract_policy_state,
    segment_ranges,
    validate_action_chunk,
    validate_dashboard_command_id,
    validate_dashboard_manual_command,
    validate_dashboard_plan_id,
    validate_dashboard_prepare_request,
    validate_relative_navigation_motion,
)
from robots.behavior.task_specs import (
    TURNING_ON_RADIO_TASK_SPEC,
    BehaviorTaskSpec,
    resolve_task_spec,
)
from rpent.utils.config import get_repo_root, get_rlinf_repo_path
from rpent.utils.logging import get_logger
from rpent.utils.socket_rpc import SocketRpcServer

logger = get_logger("behavior_env_server")
RPENT_ROOT = get_repo_root()
RLINF_ROOT = get_rlinf_repo_path() or (RPENT_ROOT.parent / "RLinf_agentic_push")
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

_ENV_RPC_METHODS = frozenset(
    {
        "get_env_meta",
        "guard_tool_call",
        "reset",
        "prepare_vla_invocation",
        "current_observation",
        "dashboard_control_capabilities",
        "dashboard_prepare_manual_command",
        "dashboard_execute_prepared_command",
        "dashboard_discard_prepared_command",
        "dashboard_capture_views",
        "dashboard_manual_command",
        "finalize_paused_runtime",
        "observe",
        "pixel_to_world",
        "navigate_to",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "pi0_nav_pick_chunk_step",
        "save_robot_state_checkpoint",
    }
)

_CONTROLLER_VLA = "vla"
_CONTROLLER_PLANNER = "planner"
_CONTROLLER_SWITCHING = "switching"
_CONTROLLER_FAILED = "failed"
_CONTROLLER_FROZEN = "frozen"
_BEHAVIOR_ROBOT_COLLISION_GROUP = "rpent_behavior_r1pro"
_BEHAVIOR_NONFLOOR_COLLISION_GROUP = "rpent_behavior_nonfloor_world"
_R1PRO_FLOOR_SUPPORT_LINK_NAMES = frozenset(
    {
        "wheel_motor_link1",
        "wheel_motor_link2",
        "wheel_motor_link3",
    }
)
_HAND_GEOMETRY_SYNC_RENDER_ITERATIONS = HAND_GEOMETRY_SYNC_RENDER_ITERATIONS
_HAND_GEOMETRY_TRANSLATION_TOLERANCE_M = (
    HAND_GEOMETRY_TRANSLATION_TOLERANCE_M
)
_HAND_GEOMETRY_ROTATION_TOLERANCE_DEG = (
    HAND_GEOMETRY_ROTATION_TOLERANCE_DEG
)
_HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M = (
    HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M
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


def _canonical_json_bytes(value: Any) -> bytes:
    """Encode runtime receipt material deterministically without repr fallbacks."""

    def encode_unknown(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, bytes):
            return {"__bytes_hex__": item.hex()}
        raise TypeError(
            "unsupported runtime receipt value: "
            f"{type(item).__module__}.{type(item).__qualname__}"
        )

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=encode_unknown,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    """Return only OmniGibson metric optical-axis depth.

    ``depth_linear`` is ``distance_to_image_plane``.  OmniGibson's distinct
    ``depth`` modality is ``distance_to_camera`` (Euclidean ray range), so
    falling back to it would corrupt pinhole back-projection by treating range
    as optical-axis Z.
    """

    return payload.get("depth_linear")


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


def _sensor_render_camera_to_world(sensor: Any) -> np.ndarray | None:
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
    return None


def _sensor_camera_to_world(sensor: Any) -> np.ndarray | None:
    render_matrix = _sensor_render_camera_to_world(sensor)
    if render_matrix is not None:
        return render_matrix
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
    if not isinstance(info, dict) or not isinstance(info.get("done"), dict):
        return False
    value = info["done"].get("success")
    return bool(value) if isinstance(value, (bool, np.bool_)) else False


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


def _strict_wire_bool(value: Any) -> bool | None:
    """Return a strict scalar boolean without accepting truthy payloads."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _classify_pi0_terminal_step(
    *,
    info: Any,
    raw_terminated: Any,
    raw_truncated: Any,
) -> dict[str, Any]:
    """Separate predicate-success termination from real Pi0 stop conditions.

    OmniGibson's BEHAVIOR ``PredicateGoal`` is itself a success termination
    condition, so a normal raw-success step reports ``terminated=True``.  That
    aggregate flag cannot be used directly when an admitted Pi0 invocation must
    finish its requested complete chunks.  A success-only predicate termination
    is therefore soft inside the active invocation.  Timeout, failure
    conditions, truncation, and malformed/inconsistent terminal envelopes remain
    hard and fail closed.
    """

    terminated_flag = _scalar_bool(raw_terminated)
    truncated_flag = _scalar_bool(raw_truncated)
    raw_success = _raw_success(info)
    done = info.get("done") if isinstance(info, dict) else None
    conditions = done.get("termination_conditions") if isinstance(done, dict) else None

    active_success: list[str] = []
    active_failure: list[str] = []
    active_timeout: list[str] = []
    malformed = False
    conditions_present = isinstance(conditions, dict) and bool(conditions)
    if conditions_present:
        for name, payload in conditions.items():
            if not isinstance(name, str) or not name or not isinstance(payload, dict):
                malformed = True
                continue
            condition_done = _strict_wire_bool(payload.get("done"))
            condition_success = _strict_wire_bool(payload.get("success"))
            if condition_done is None or condition_success is None:
                malformed = True
                continue
            if not condition_done:
                if condition_success:
                    malformed = True
                continue
            if name == "timeout":
                if condition_success:
                    malformed = True
                active_timeout.append(name)
            elif condition_success:
                active_success.append(name)
            else:
                active_failure.append(name)
    elif raw_success or terminated_flag or truncated_flag:
        malformed = True

    info_done = bool(active_success or active_failure or active_timeout)
    if conditions_present and info_done != bool(terminated_flag or truncated_flag):
        malformed = True
    if raw_success != bool(active_success):
        malformed = True

    hard_truncated = bool(truncated_flag or active_timeout)
    hard_terminated = bool(active_failure)
    soft_success_termination = bool(
        raw_success
        and active_success
        and terminated_flag
        and not truncated_flag
        and not active_failure
        and not active_timeout
        and not malformed
    )
    if malformed:
        hard_terminated = True
        soft_success_termination = False
    elif terminated_flag and not soft_success_termination:
        hard_terminated = True
    if hard_truncated:
        soft_success_termination = False

    if malformed:
        reason = "malformed_or_inconsistent_terminal_envelope"
    elif hard_truncated:
        reason = "hard_truncation"
    elif hard_terminated:
        reason = "hard_termination"
    elif soft_success_termination:
        reason = "soft_predicate_success"
    else:
        reason = "running"
    return {
        "raw_terminated": bool(terminated_flag),
        "raw_truncated": bool(truncated_flag),
        "raw_success": bool(raw_success),
        "info_done": bool(info_done),
        "soft_success_termination": bool(soft_success_termination),
        "hard_terminated": bool(hard_terminated),
        "hard_truncated": bool(hard_truncated),
        "terminal_envelope_malformed": bool(malformed),
        "terminal_classification_reason": reason,
        "active_success_conditions": active_success,
        "active_failure_conditions": active_failure,
        "active_timeout_conditions": active_timeout,
    }


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


def _resolve_env_task_identity(
    meta: dict[str, Any],
) -> tuple[BehaviorTaskSpec, tuple[str, int, int]]:
    """Validate the task-scoped native identity before constructing the env."""

    task_spec = resolve_task_spec(
        task_name=str(meta["task_name"]),
        task_index=int(meta["task"]),
    )
    activity_definition_id = int(meta["activity_definition_id"])
    if activity_definition_id != task_spec.activity_definition_id:
        raise ValueError(
            f"{task_spec.task_name} requires activity_definition_id "
            f"{task_spec.activity_definition_id}, got {activity_definition_id}"
        )
    scene_model = str(meta["scene_model"])
    if scene_model != task_spec.scene_model:
        raise ValueError(
            f"{task_spec.task_name} requires scene_model {task_spec.scene_model!r}, "
            f"got {scene_model!r}"
        )
    activity_instance_id = int(meta["activity_instance_id"])
    if activity_instance_id <= 0:
        raise ValueError("activity_instance_id must be positive")
    public_seed = int(meta["public_seed"])
    if public_seed < 0:
        raise ValueError("public_seed must be non-negative")
    mapped_public_seed = task_spec.public_seed_for_instance(activity_instance_id)
    if mapped_public_seed is not None and mapped_public_seed != public_seed:
        raise ValueError(
            f"{task_spec.task_name} instance {activity_instance_id} is public "
            f"s{mapped_public_seed}, not s{public_seed}"
        )
    instance_dir = Path(str(meta["activity_instance_dir"])).expanduser()
    if instance_dir.name != task_spec.state_dir_name:
        raise ValueError(
            f"{task_spec.task_name} requires state directory "
            f"{task_spec.state_dir_name!r}, got {instance_dir.name!r}"
        )
    return task_spec, (
        task_spec.task_name,
        activity_definition_id,
        activity_instance_id,
    )


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
    ) -> None:
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        task_spec, task_identity = _resolve_env_task_identity(meta)
        self._env = BehaviorEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
            record_metrics=False,
        )
        robot = self._robot()
        if robot is None or not self._is_r1pro_robot(robot):
            raise RuntimeError(
                "BEHAVIOR collision configuration requires the live R1Pro robot"
            )
        self._configure_behavior_robot_collisions(robot)
        self._meta = dict(meta)
        self._controller_mode = str(meta.get("controller_mode", "hybrid"))
        if self._controller_mode not in {"hybrid", "pi0_nav_pick_only"}:
            raise ValueError("unsupported BEHAVIOR controller mode")
        self._task_spec = task_spec
        self._task_identity = task_identity
        expected_run_nonce = meta.get("expected_run_nonce")
        if expected_run_nonce is not None and (
            not isinstance(expected_run_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", expected_run_nonce) is None
        ):
            raise ValueError("expected_run_nonce must be 32 lowercase hex characters")
        self._run_nonce = (
            expected_run_nonce
            if isinstance(expected_run_nonce, str)
            else secrets.token_hex(16)
        )
        trace_writer = getattr(self._env, "_write_trace_record", None)
        if not callable(trace_writer):
            # Official RLinf keeps the trace file on its in-process
            # BehaviorProcess in direct mode.
            trace_writer = getattr(
                getattr(self._env, "_direct_process", None),
                "_write_trace_record",
                None,
            )
        if isinstance(expected_run_nonce, str):
            if not callable(trace_writer):
                raise RuntimeError(
                    "expected run nonce requires an action-trace binding writer"
                )
            trace_writer(
                {
                    "event": "rpent_run_binding",
                    "run_nonce": self._run_nonce,
                    "attempt_index": int(meta["attempt_index"]),
                }
            )
        self._attempt_index = int(meta["attempt_index"])
        if self._attempt_index < 1:
            raise ValueError("attempt_index must be positive")
        self._attempt_nonce = secrets.token_hex(16)
        self._active_vla_invocation: str | None = None
        self._active_vla_call_index: int | None = None
        self._pending_vla_baseline_internal_authorization = False
        self._public_capture_sequence = 0
        self._latest_unconsumed_public_capture_receipt: dict[str, Any] | None = None
        self._motion_in_flight = False
        self._threaded_predicted_planning_dispatcher_ready = False
        self._dashboard_planning_admitted = False
        self._dashboard_execute_receipts: dict[
            str,
            tuple[str, dict[str, Any]],
        ] = {}
        self._dashboard_env_step_latency: dict[str, Any] | None = None
        self._official_success_latched = False
        self._official_success_receipt: dict[str, Any] | None = None
        self._official_success_receipt_path = (
            Path(output_dir) / "official_success_receipt.json"
        )
        self._terminal_failure_receipt: dict[str, Any] | None = None
        self._terminal_failure_receipt_path = (
            Path(output_dir) / "terminal_failure_receipt.json"
        )
        self._output_dir = Path(output_dir)
        self._done = False
        self._env_steps = 0
        self._video_path = output_dir / "episode.mp4"
        self._video_writer = None
        self._video_frames = 0
        self._video_error: str | None = None
        self._video_sealed = False
        self._video_source_shapes: dict[str, list[int]] = {}
        self._video_queue: Queue[Any] = Queue(maxsize=64)
        self._video_queue_stop = object()
        self._video_queue_closed = False
        self._video_frames_dropped = 0
        self._video_worker = threading.Thread(
            target=self._video_writer_loop,
            name="behavior-mp4-writer",
            daemon=True,
        )
        self._video_worker.start()
        self._planner_video_interval_steps = max(
            1, int(os.environ.get("RPENT_PLANNER_VIDEO_INTERVAL_STEPS", "4"))
        )
        correction_path = os.environ.get("RPENT_BEHAVIOR_CAMERA_CORRECTION_PROFILE")
        self._frame_cache = FrameCache(
            max_frames_per_camera=8,
            ttl_s=60.0,
            correction_profiles=load_camera_correction_profiles(correction_path),
        )
        self._planner = PlannerExecutor(
            env=self,
            frame_cache=self._frame_cache,
            output_dir=output_dir,
        )
        self._last_observation: dict[str, Any] | None = None
        self._last_info: Any = None
        self._gripper_latch = {"left": 1.0, "right": 1.0}
        self._live_observation_counter = 0
        self._visual_checkpoint_counter = 0
        self._base_controller_mode = "velocity"
        self._velocity_controller_config: Any = None
        self._action_source = "pi0_vla"
        self._vla_actions_enabled = True
        self._controller_state = _CONTROLLER_VLA
        self._motion_frozen = False
        self._reset_completed = False
        self._last_capture_step: int | None = None
        self._public_observed_frame_ids: set[str] = set()
        self._latest_public_head_frame_id: str | None = None
        self._latest_public_observation_lineage: dict[str, Any] | None = None
        self._projection_receipts: dict[str, dict[str, Any]] = {}

    def _active_task_spec(self) -> BehaviorTaskSpec:
        """Return the selected task contract.

        The fallback exists only for lightweight ``__new__`` unit facades;
        production construction always resolves and cross-checks ``meta``.
        """

        return getattr(self, "_task_spec", TURNING_ON_RADIO_TASK_SPEC)

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

    @staticmethod
    def _is_r1pro_robot(robot: Any) -> bool:
        robot_class = type(robot).__name__.strip().lower()
        robot_name = str(getattr(robot, "name", "")).strip().lower()
        return bool(
            robot_class in {"r1", "r1pro"}
            or "r1pro" in robot_name
            or robot_name == "r1"
        )

    @staticmethod
    def _collision_relation_paths(relation: Any) -> set[str]:
        get_targets = getattr(relation, "GetTargets", None)
        if not callable(get_targets):
            raise RuntimeError("PhysX collision relationship is unavailable")
        paths: set[str] = set()
        for target in get_targets():
            value = getattr(target, "pathString", None)
            path = str(value if value is not None else target).strip()
            if not path.startswith("/"):
                raise RuntimeError(
                    f"PhysX collision relationship returned an invalid path: {path!r}"
                )
            paths.add(path)
        return paths

    @classmethod
    def _collision_group_includes(cls, group: Any) -> set[str]:
        collection = getattr(group, "GetCollidersCollectionAPI", None)
        if not callable(collection):
            raise RuntimeError("PhysX collision group collection API is unavailable")
        includes = getattr(collection(), "GetIncludesRel", None)
        if not callable(includes):
            raise RuntimeError("PhysX collision group includes API is unavailable")
        return cls._collision_relation_paths(includes())

    @classmethod
    def _collision_group_filters(cls, group: Any) -> set[str]:
        filtered_groups = getattr(group, "GetFilteredGroupsRel", None)
        if not callable(filtered_groups):
            raise RuntimeError("PhysX collision group filter API is unavailable")
        return cls._collision_relation_paths(filtered_groups())

    def _apply_behavior_robot_collision_policy(
        self,
        robot: Any,
        *,
        scene: Any,
        collision_api: Any,
        ground_categories: frozenset[str],
    ) -> dict[str, int]:
        """Keep wheel-floor support while filtering all other robot collisions."""

        if not self._is_r1pro_robot(robot):
            raise RuntimeError("collision policy requires the live R1Pro robot")
        links = getattr(robot, "links", None)
        if not isinstance(links, dict) or not links:
            raise RuntimeError("R1Pro links are unavailable for collision policy")
        support_link_names = frozenset(
            str(name).strip()
            for name in getattr(robot, "floor_touching_base_link_names", ())
        )
        if support_link_names != _R1PRO_FLOOR_SUPPORT_LINK_NAMES:
            raise RuntimeError(
                "R1Pro floor-support link identity is not the verified "
                "three-wheel set"
            )
        missing_support_links = support_link_names.difference(links)
        if missing_support_links:
            raise RuntimeError(
                "R1Pro floor-support links are unavailable: "
                f"{sorted(missing_support_links)!r}"
            )
        scene_objects = getattr(scene, "objects", None)
        if scene_objects is None:
            raise RuntimeError("BEHAVIOR scene objects are unavailable")
        scene_objects = list(scene_objects)
        if not scene_objects:
            raise RuntimeError("BEHAVIOR scene has no objects for collision policy")
        if not isinstance(ground_categories, frozenset) or not ground_categories:
            raise RuntimeError("OmniGibson ground categories are unavailable")

        try:
            robot.self_collisions = False
        except Exception as exc:
            raise RuntimeError(
                "failed to disable R1Pro PhysX self collision"
            ) from exc

        support_link_paths: set[str] = set()
        support_collision_mesh_count = 0
        disabled_non_support_link_count = 0
        disabled_non_support_mesh_count = 0
        for link_name, link in links.items():
            enable_collisions = getattr(link, "enable_collisions", None)
            disable_collisions = getattr(link, "disable_collisions", None)
            collision_meshes = getattr(link, "collision_meshes", None)
            link_path = str(getattr(link, "prim_path", "")).strip()
            if (
                not callable(enable_collisions)
                or not callable(disable_collisions)
                or not isinstance(collision_meshes, dict)
                or not link_path.startswith("/")
            ):
                raise RuntimeError(
                    f"R1Pro link {link_name!r} lacks the OmniGibson collision API"
                )
            if link_name in support_link_names:
                enable_collisions()
                support_link_paths.add(link_path)
                support_collision_mesh_count += len(collision_meshes)
                disabled_meshes = [
                    str(mesh_name)
                    for mesh_name, mesh in collision_meshes.items()
                    if not bool(getattr(mesh, "collision_enabled", False))
                ]
                if disabled_meshes:
                    raise RuntimeError(
                        f"R1Pro support link {link_name!r} retained disabled "
                        f"collision meshes: {disabled_meshes!r}"
                    )
            else:
                disable_collisions()
                disabled_non_support_link_count += 1
                disabled_non_support_mesh_count += len(collision_meshes)
                enabled_meshes = [
                    str(mesh_name)
                    for mesh_name, mesh in collision_meshes.items()
                    if bool(getattr(mesh, "collision_enabled", False))
                ]
                if enabled_meshes:
                    raise RuntimeError(
                        f"R1Pro non-support link {link_name!r} retained enabled "
                        f"collision meshes: {enabled_meshes!r}"
                    )
        if support_collision_mesh_count < 1:
            raise RuntimeError("R1Pro floor-support links have no collision meshes")
        if len(support_link_paths) != len(support_link_names):
            raise RuntimeError("R1Pro support-link prim paths are not unique")
        if bool(robot.self_collisions):
            raise RuntimeError("R1Pro PhysX self collision remained enabled")

        robot_path = str(getattr(robot, "prim_path", "")).rstrip("/")
        ground_paths: set[str] = set()
        nonfloor_paths: set[str] = set()
        ground_collision_mesh_count = 0
        for obj in scene_objects:
            obj_path = str(getattr(obj, "prim_path", "")).rstrip("/")
            if not obj_path.startswith("/"):
                raise RuntimeError("BEHAVIOR scene object has no absolute prim path")
            if obj is robot or obj_path == robot_path:
                continue
            category = str(getattr(obj, "category", "")).strip().lower()
            if category in ground_categories:
                ground_paths.add(obj_path)
                object_links = getattr(obj, "links", None)
                if isinstance(object_links, dict):
                    for ground_link in object_links.values():
                        collision_meshes = getattr(
                            ground_link, "collision_meshes", None
                        )
                        if isinstance(collision_meshes, dict):
                            ground_collision_mesh_count += sum(
                                bool(getattr(mesh, "collision_enabled", False))
                                for mesh in collision_meshes.values()
                            )
            else:
                nonfloor_paths.add(obj_path)
        if not ground_paths:
            raise RuntimeError(
                "BEHAVIOR scene contains no OmniGibson ground-category object"
            )
        if ground_collision_mesh_count < 1:
            raise RuntimeError(
                "BEHAVIOR ground objects contain no enabled collision mesh"
            )
        if ground_paths & nonfloor_paths:
            raise RuntimeError("ground and non-floor collision sets overlap")

        active_groups = getattr(collision_api, "ACTIVE_COLLISION_GROUPS", None)
        if not isinstance(active_groups, dict):
            raise RuntimeError("OmniGibson CollisionAPI registry is unavailable")
        group_names = (
            _BEHAVIOR_ROBOT_COLLISION_GROUP,
            _BEHAVIOR_NONFLOOR_COLLISION_GROUP,
        )
        if any(name in active_groups for name in group_names):
            raise RuntimeError("BEHAVIOR collision groups already exist")
        collision_api.create_collision_group(
            _BEHAVIOR_ROBOT_COLLISION_GROUP,
            filter_self_collisions=True,
        )
        collision_api.create_collision_group(
            _BEHAVIOR_NONFLOOR_COLLISION_GROUP,
            filter_self_collisions=False,
        )
        for link_path in sorted(support_link_paths):
            collision_api.add_to_collision_group(
                _BEHAVIOR_ROBOT_COLLISION_GROUP,
                link_path,
            )
        for object_path in sorted(nonfloor_paths):
            collision_api.add_to_collision_group(
                _BEHAVIOR_NONFLOOR_COLLISION_GROUP,
                object_path,
            )
        collision_api.add_group_filter(
            _BEHAVIOR_ROBOT_COLLISION_GROUP,
            _BEHAVIOR_NONFLOOR_COLLISION_GROUP,
        )

        robot_group = active_groups.get(_BEHAVIOR_ROBOT_COLLISION_GROUP)
        nonfloor_group = active_groups.get(_BEHAVIOR_NONFLOOR_COLLISION_GROUP)
        if robot_group is None or nonfloor_group is None:
            raise RuntimeError("BEHAVIOR collision groups were not created")
        if self._collision_group_includes(robot_group) != support_link_paths:
            raise RuntimeError("R1Pro collision group membership verification failed")
        if self._collision_group_includes(nonfloor_group) != nonfloor_paths:
            raise RuntimeError(
                "non-floor world collision group membership verification failed"
            )
        robot_group_path = (
            f"/World/collision_groups/{_BEHAVIOR_ROBOT_COLLISION_GROUP}"
        )
        nonfloor_group_path = (
            f"/World/collision_groups/{_BEHAVIOR_NONFLOOR_COLLISION_GROUP}"
        )
        expected_filters = {robot_group_path, nonfloor_group_path}
        if self._collision_group_filters(robot_group) != expected_filters:
            raise RuntimeError("R1Pro collision group filter verification failed")
        if self._collision_group_filters(nonfloor_group):
            raise RuntimeError("non-floor collision group unexpectedly filters itself")

        logger.info(
            "configured R1Pro collision policy: %d support links, "
            "%d support meshes, %d disabled non-support links, "
            "%d ground objects, %d non-floor objects",
            len(support_link_paths),
            support_collision_mesh_count,
            disabled_non_support_link_count,
            len(ground_paths),
            len(nonfloor_paths),
        )
        return {
            "support_links": len(support_link_paths),
            "support_collision_meshes": support_collision_mesh_count,
            "disabled_non_support_links": disabled_non_support_link_count,
            "disabled_non_support_collision_meshes": (
                disabled_non_support_mesh_count
            ),
            "ground_objects": len(ground_paths),
            "ground_collision_meshes": ground_collision_mesh_count,
            "nonfloor_objects": len(nonfloor_paths),
        }

    def _configure_behavior_robot_collisions(self, robot: Any) -> None:
        """Apply the collision policy in a stopped-simulator transaction."""

        try:
            import omnigibson as og
            from omnigibson.utils.usd_utils import CollisionAPI
        except Exception as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "OmniGibson is unavailable for R1Pro collision configuration"
            ) from exc
        simulator = getattr(og, "sim", None)
        if simulator is None:
            raise RuntimeError(
                "OmniGibson simulator is unavailable for collision configuration"
            )
        is_playing = getattr(simulator, "is_playing", None)
        is_paused = getattr(simulator, "is_paused", None)
        is_stopped = getattr(simulator, "is_stopped", None)
        stop = getattr(simulator, "stop", None)
        play = getattr(simulator, "play", None)
        pause = getattr(simulator, "pause", None)
        required = (is_playing, is_paused, is_stopped, stop, play, pause)
        if not all(callable(method) for method in required):
            raise RuntimeError(
                "OmniGibson simulator lifecycle API is incomplete"
            )
        was_playing = bool(is_playing())
        was_paused = bool(is_paused())
        was_stopped = bool(is_stopped())
        if sum((was_playing, was_paused, was_stopped)) != 1:
            raise RuntimeError(
                "OmniGibson simulator reported an invalid lifecycle state"
            )
        configuration_complete = False
        try:
            if not was_stopped:
                stop()
            if not bool(is_stopped()):
                raise RuntimeError(
                    "OmniGibson simulator did not stop for collision configuration"
                )
            omni_env = getattr(self._env, "omnigibson_env", None)
            scene = getattr(omni_env, "scene", None)
            if scene is None:
                raise RuntimeError("BEHAVIOR OmniGibson scene is unavailable")
            self._apply_behavior_robot_collision_policy(
                robot,
                scene=scene,
                collision_api=CollisionAPI,
                # The user authorized only robot-floor support.  Carpets,
                # lawns, driveways, walls, furniture and task objects remain
                # in the filtered non-floor world group.
                ground_categories=frozenset({"floors"}),
            )
            configuration_complete = True
        except Exception as exc:
            raise RuntimeError(
                "BEHAVIOR R1Pro collision configuration failed"
            ) from exc
        finally:
            # A partially built filter must never be resumed.  Keeping the
            # simulator stopped makes a failed constructor inert until its
            # owning process exits.
            if configuration_complete and (was_playing or was_paused):
                try:
                    # OmniGibson.play() rebuilds the Physics Tensor simulation
                    # view and all initialized object handles before it returns.
                    play()
                    if was_paused:
                        pause()
                except Exception as exc:
                    raise RuntimeError(
                        "failed to restore OmniGibson after R1Pro collision "
                        "configuration"
                    ) from exc
        if was_playing and not bool(is_playing()):
            raise RuntimeError(
                "OmniGibson simulator did not resume after collision configuration"
            )
        if was_paused and not bool(is_paused()):
            raise RuntimeError(
                "OmniGibson simulator pause state was not restored"
            )
        if was_stopped and not bool(is_stopped()):
            raise RuntimeError(
                "OmniGibson simulator stopped state was not preserved"
            )

    def _require_planner(self) -> PlannerExecutor:
        if self._planner is None:
            raise RuntimeError("planner primitives are unavailable")
        return self._planner

    def _assert_rpc_lifecycle(self, method: str) -> None:
        """Enforce live lifecycle facts without prescribing a tool order."""

        if bool(getattr(self, "_official_success_latched", False)) or _raw_success(
            getattr(self, "_last_info", None)
        ):
            if method == "finalize_paused_runtime":
                return
            raise RuntimeError(
                "raw task success is terminal; no further RPC is allowed"
            )
        if method in {
            "get_env_meta",
            "guard_tool_call",
            "dashboard_control_capabilities",
        }:
            return
        if method == "reset":
            if bool(getattr(self, "_reset_completed", False)):
                raise RuntimeError("the BEHAVIOR environment may be reset only once")
            return
        if method == "prepare_vla_invocation":
            if not bool(getattr(self, "_reset_completed", False)):
                raise RuntimeError(f"{method} requires the initialized episode")
            return
        if not bool(getattr(self, "_reset_completed", False)):
            raise RuntimeError("physical tools require the initialized episode")
        if bool(getattr(self, "_motion_frozen", False)):
            raise RuntimeError("terminal state is frozen; no further RPCs are allowed")
        if method == "finalize_paused_runtime":
            if getattr(self, "_controller_state", None) not in {
                _CONTROLLER_VLA,
                _CONTROLLER_PLANNER,
            }:
                raise RuntimeError("controller cannot be transferred to planner")
            return
        if method == "pi0_nav_pick_chunk_step":
            if (
                getattr(self, "_controller_state", None) != _CONTROLLER_VLA
                or getattr(self, "_vla_actions_enabled", None) is not True
                or getattr(self, "_action_source", None) != "pi0_vla"
            ):
                raise RuntimeError("VLA controller is not active")
            return

    def guard_tool_call(
        self, *, name: str, input_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Return read-only runtime facts used for side-effect-free rejection."""

        del input_dict
        failed: list[str] = []
        if not self._reset_completed:
            failed.append("episode_not_initialized")
        if self._official_success_latched or _raw_success(self._last_info):
            failed.append("official_success_latched")
        if getattr(self, "_terminal_failure_receipt", None) is not None:
            failed.append("terminal_failure_latched")
        if name == "pi0_nav_pick":
            if self._controller_state not in {
                _CONTROLLER_VLA,
                _CONTROLLER_PLANNER,
                _CONTROLLER_FAILED,
            }:
                failed.append("controller_not_vla_rearmable")
        failed = list(dict.fromkeys(failed))
        return {
            "primitive_success": not failed,
            "task_success": bool(
                self._official_success_latched or _raw_success(self._last_info)
            ),
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": "guard_passed" if not failed else "precondition_rejected",
            "failed_preconditions": failed,
            "attempt_index": self._attempt_index,
            "attempt_nonce": self._attempt_nonce,
            "total_env_steps": int(self._env_steps),
        }

    def _seal_attempt_receipt(self, value: dict[str, Any]) -> dict[str, Any]:
        """Bind one public receipt to this attempt and add a stable digest."""

        receipt = {
            "schema_version": 1,
            "run_nonce": str(getattr(self, "_run_nonce", "unbound")),
            "attempt_nonce": str(getattr(self, "_attempt_nonce", "unbound")),
            "attempt_index": int(getattr(self, "_attempt_index", 1)),
            **value,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        return receipt

    def _attempt_receipt_is_current(self, receipt: Any) -> bool:
        if not isinstance(receipt, dict):
            return False
        try:
            return bool(
                receipt.get("run_nonce") == getattr(self, "_run_nonce", None)
                and receipt.get("attempt_nonce")
                == getattr(self, "_attempt_nonce", None)
                and int(receipt.get("attempt_index", -1))
                == int(getattr(self, "_attempt_index", -2))
                and isinstance(receipt.get("receipt_sha256"), str)
                and receipt.get("receipt_sha256")
                == hashlib.sha256(
                    _canonical_json_bytes(
                        {
                            key: value
                            for key, value in receipt.items()
                            if key != "receipt_sha256"
                        }
                    )
                ).hexdigest()
            )
        except (TypeError, ValueError):
            return False

    def _clear_active_vla_invocation_state(self) -> None:
        """Remove invocation-local authority after finalization or failure."""

        self._active_vla_invocation = None
        self._active_vla_call_index = None
        self._pending_vla_baseline_internal_authorization = False

    def _register_public_capture(
        self,
        *,
        requested_camera: str,
        resolved_camera: str,
        frame: Any,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        """Create the sole one-use frame-review authority for a fresh capture."""

        if self._active_task_spec().surface_review_policy is None:
            self._latest_unconsumed_public_capture_receipt = None
            return {}
        self._public_capture_sequence = (
            int(getattr(self, "_public_capture_sequence", 0)) + 1
        )
        receipt = self._seal_attempt_receipt(
            {
                "kind": "public_observe_capture",
                "requested_camera": requested_camera,
                "resolved_camera": resolved_camera,
                "frame_id": str(frame.frame_id),
                "capture_group_id": str(frame.capture_group_id),
                "env_step": int(self._env_steps),
                "capture_sequence": int(self._public_capture_sequence),
                "rgb_sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )
        self._latest_unconsumed_public_capture_receipt = receipt
        return receipt

    @staticmethod
    def _metric_depth_sha256(frame: Any) -> str:
        """Hash the immutable, normalized metric depth samples for one frame."""

        if not hasattr(frame, "depth_m"):
            raise CameraGeometryError("metric depth is unavailable for RGB-D lineage")
        depth = np.ascontiguousarray(np.asarray(frame.depth_m, dtype=np.dtype("<f8")))
        material = (
            json.dumps(
                {
                    "shape": list(depth.shape),
                    "dtype": str(depth.dtype),
                    "source_modality": "depth_linear",
                    "measurement": "distance_to_image_plane",
                    "unit": "m",
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\0"
            + depth.tobytes(order="C")
        )
        return hashlib.sha256(material).hexdigest()

    @staticmethod
    def _frame_geometry_sha256(frame: Any) -> str:
        """Bind every transform used by frame-bound depth and hand distances."""

        intrinsics = getattr(frame, "intrinsics", None)
        if intrinsics is None:
            raise CameraGeometryError("camera intrinsics are unavailable")
        camera_to_world = validated_rigid_transform(
            getattr(frame, "camera_to_world", None),
            name="frame camera-to-world transform",
        )
        correction = getattr(frame, "correction_profile", None)
        if correction is None:
            raise CameraGeometryError("camera correction profile is unavailable")
        correction_transform = validated_rigid_transform(
            getattr(correction, "raw_to_corrected_camera", None),
            name="camera correction transform",
        )
        metadata = getattr(frame, "capture_metadata", None)
        hand_references = (
            metadata.get("r1pro_hand_reference_transforms")
            if isinstance(metadata, dict)
            else None
        )
        sync_certificate = (
            metadata.get("hand_geometry_sync_certificate")
            if isinstance(metadata, dict)
            else None
        )
        render_sync_iterations = (
            metadata.get("render_sync_iterations")
            if isinstance(metadata, dict)
            else None
        )
        camera_pose_lineage = (
            metadata.get("camera_pose_lineage") if isinstance(metadata, dict) else None
        )
        material = {
            "schema_version": 1,
            "camera": str(frame.camera),
            "frame_id": str(frame.frame_id),
            "capture_group_id": str(frame.capture_group_id),
            "env_step": int(frame.step_index),
            "intrinsics": {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.cx),
                "cy": float(intrinsics.cy),
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
            },
            "camera_to_world": camera_to_world.tolist(),
            "correction": {
                "enabled": bool(correction.enabled),
                "raw_to_corrected_camera": correction_transform.tolist(),
            },
            "r1pro_hand_reference_transforms": hand_references,
            "hand_geometry_sync_certificate": sync_certificate,
            "camera_pose_lineage": camera_pose_lineage,
            "render_sync_iterations": render_sync_iterations,
        }
        return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()

    @staticmethod
    def _hand_geometry_sync_certificate_is_valid(
        certificate: Any,
        *,
        hand: str,
        env_step: int,
    ) -> bool:
        """Recompute certificate admission from sealed numeric residuals."""

        return hand_geometry_sync_certificate_is_valid(
            certificate,
            hand=hand,
            env_step=env_step,
        )

    def _register_public_observation_lineage(
        self,
        *,
        requested_camera: str,
        resolved_camera: str,
        frame: Any,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        """Bind depth probes to the immediately preceding public RGB-D frame.

        This receipt is deliberately independent of the Radio-only, one-use
        surface-review receipt.  A read-only depth probe therefore neither
        consumes nor renews Radio visual authorization.
        """

        try:
            depth_sha256: str | None = self._metric_depth_sha256(frame)
            geometry_sha256: str | None = self._frame_geometry_sha256(frame)
        except CameraGeometryError:
            # Lightweight compatibility facades may expose RGB-only frames.
            # They remain observable but cannot authorize any depth probe.
            depth_sha256 = None
            geometry_sha256 = None
        receipt = self._seal_attempt_receipt(
            {
                "kind": "public_observe_rgbd_capture",
                "requested_camera": str(requested_camera),
                "resolved_camera": str(resolved_camera),
                "frame_id": str(frame.frame_id),
                "capture_group_id": str(frame.capture_group_id),
                "env_step": int(self._env_steps),
                "rgb_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "depth_sha256": depth_sha256,
                "geometry_sha256": geometry_sha256,
                "depth_source_modality": "depth_linear",
                "depth_measurement": "distance_to_image_plane",
                "depth_unit": "m",
            }
        )
        self._latest_public_observation_lineage = receipt
        return receipt

    @classmethod
    def _public_visual_receipt(cls, value: Any) -> Any:
        """Remove private dynamic-side lineage from Agent-visible receipts."""

        if isinstance(value, dict):
            return {
                key: cls._public_visual_receipt(item)
                for key, item in value.items()
                if key not in {"resolved_camera", "resolved_hand"}
            }
        if isinstance(value, list):
            return [cls._public_visual_receipt(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._public_visual_receipt(item) for item in value)
        return _wire_safe(value)

    def _record_frame_review_cycle(
        self,
        *,
        requested_camera: str,
        resolved_camera: str,
        frame: Any,
        capture_receipt: dict[str, Any],
        assessment: str,
    ) -> dict[str, Any]:
        """Record one visual assessment without physical-state admission."""

        surface_policy = self._active_task_spec().surface_review_policy
        if surface_policy is None:
            raise ValueError(
                f"{self._active_task_spec().task_name} does not define frame review"
            )
        review_receipt = self._seal_attempt_receipt(
            {
                "kind": "public_frame_review",
                "env_step": int(self._env_steps),
                "camera": requested_camera,
                "resolved_camera": resolved_camera,
                "frame_id": str(frame.frame_id),
                "capture_group_id": str(frame.capture_group_id),
                "capture_sequence": int(capture_receipt["capture_sequence"]),
                "capture_receipt_sha256": capture_receipt["receipt_sha256"],
                "rgb_sha256": capture_receipt["rgb_sha256"],
                "assessment": assessment,
            }
        )
        return {
            "accepted": True,
            "frame_review_receipt": review_receipt,
            "capture_receipt": capture_receipt,
        }

    def _visual_hand_authorization(
        self,
        *,
        selected_hand: str,
        visual_hand_check: Any,
    ) -> dict[str, Any]:
        """Bind an LLM-confirmed physical hand to the latest public head frame."""

        if not isinstance(visual_hand_check, dict):
            raise ValueError("visual_hand_check is required for analytic primitives")
        required = {"camera", "frame_id", "selected_hand", "assessment"}
        if set(visual_hand_check) != required:
            raise ValueError(
                "visual_hand_check requires exactly camera, frame_id, "
                "selected_hand, and assessment"
            )
        camera = str(visual_hand_check["camera"])
        if camera != "head":
            raise ValueError("visual_hand_check.camera must be head")
        checked_hand = str(visual_hand_check["selected_hand"])
        if checked_hand != selected_hand:
            raise ValueError(
                "visual_hand_check.selected_hand must match the resolved physical hand"
            )
        assessment = str(visual_hand_check["assessment"])
        if assessment != "selected_hand_visually_confirmed":
            raise ValueError(
                "visual_hand_check.assessment must be selected_hand_visually_confirmed"
            )
        frame_id = str(visual_hand_check["frame_id"]).strip()
        if not frame_id:
            raise ValueError("visual_hand_check.frame_id must be non-empty")
        resolved_camera = canonical_camera(camera)
        frame = self._frame_cache.get_current(resolved_camera, frame_id)
        if frame.frame_id not in getattr(self, "_public_observed_frame_ids", set()):
            raise RuntimeError(
                "visual_hand_check must reference a public observe result"
            )
        if frame.frame_id != getattr(self, "_latest_public_head_frame_id", None):
            raise RuntimeError(
                "visual_hand_check must reference the latest public head observation"
            )
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError("visual_hand_check must reference the current env step")
        if not isinstance(frame.capture_group_id, str) or not frame.capture_group_id:
            raise RuntimeError(
                "visual_hand_check must reference a synchronized public observe capture"
            )
        return self._seal_attempt_receipt(
            {
                "kind": "visual_hand_authorization",
                "source": "llm_fresh_public_head_observation",
                "env_step": int(self._env_steps),
                "camera": camera,
                "resolved_camera": resolved_camera,
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "selected_hand": selected_hand,
                "assessment": "selected_hand_visually_confirmed",
            }
        )

    def _navigation_visual_authorization(
        self,
        *,
        projection_receipt: Any,
        navigation_visual_check: Any,
    ) -> dict[str, Any]:
        """Bind a base-navigation projection to the latest public head frame."""

        if not self._projection_receipt_is_fresh(projection_receipt):
            raise RuntimeError("fresh projection receipt is required")
        if not isinstance(navigation_visual_check, dict):
            raise ValueError("navigation_visual_check is required for navigate_to")
        required = {"camera", "frame_id", "assessment"}
        if set(navigation_visual_check) != required:
            raise ValueError(
                "navigation_visual_check requires exactly camera, frame_id, "
                "and assessment"
            )
        camera = str(navigation_visual_check["camera"])
        if camera != "head":
            raise ValueError("navigation_visual_check.camera must be head")
        assessment = str(navigation_visual_check["assessment"])
        if assessment != "navigation_target_visually_confirmed":
            raise ValueError(
                "navigation_visual_check.assessment must be "
                "navigation_target_visually_confirmed"
            )
        frame_id = str(navigation_visual_check["frame_id"]).strip()
        if not frame_id:
            raise ValueError("navigation_visual_check.frame_id must be non-empty")
        resolved_camera = canonical_camera(camera)
        frame = self._frame_cache.get_current(resolved_camera, frame_id)
        if frame.frame_id not in getattr(self, "_public_observed_frame_ids", set()):
            raise RuntimeError(
                "navigation_visual_check must reference a public observe result"
            )
        if frame.frame_id != getattr(self, "_latest_public_head_frame_id", None):
            raise RuntimeError(
                "navigation_visual_check must reference the latest public head "
                "observation"
            )
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError(
                "navigation_visual_check must reference the current env step"
            )
        if not isinstance(frame.capture_group_id, str) or not frame.capture_group_id:
            raise RuntimeError(
                "navigation_visual_check must reference a synchronized public "
                "observe capture"
            )
        if (
            projection_receipt.get("camera") != camera
            or projection_receipt.get("resolved_camera") != resolved_camera
            or projection_receipt.get("frame_id") != frame.frame_id
            or projection_receipt.get("capture_group_id") != frame.capture_group_id
        ):
            raise RuntimeError(
                "navigation projection and visual check must reference the same "
                "fresh public head frame"
            )
        projection_id = projection_receipt.get("projection_id")
        if not isinstance(projection_id, str) or not projection_id:
            raise RuntimeError("navigation projection receipt has no identity")
        return self._seal_attempt_receipt(
            {
                "kind": "navigation_visual_authorization",
                "source": "llm_fresh_public_head_observation",
                "env_step": int(self._env_steps),
                "camera": camera,
                "resolved_camera": resolved_camera,
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "projection_id": projection_id,
                "assessment": "navigation_target_visually_confirmed",
            }
        )

    def _authorize_analytic_hand(
        self,
        hand: str,
        visual_hand_check: Any,
    ) -> tuple[str, str, dict[str, Any]]:
        """Bind one explicit anatomical hand to public visual evidence."""

        selected_hand = str(hand)
        if selected_hand not in {"left", "right"}:
            raise ValueError("hand must be left or right")
        evidence = self._visual_hand_authorization(
            selected_hand=selected_hand,
            visual_hand_check=visual_hand_check,
        )
        return selected_hand, "llm_visual_hand_selection", evidence

    def _projection_receipt_is_fresh(self, receipt: Any) -> bool:
        if not isinstance(receipt, dict):
            return False
        return bool(
            receipt.get("run_nonce") == self._run_nonce
            and receipt.get("attempt_nonce") == self._attempt_nonce
            and int(receipt.get("env_step", -1)) == int(self._env_steps)
            and isinstance(receipt.get("projection_id"), str)
        )

    def _switch_controller(
        self,
        target: str,
    ) -> dict[str, Any]:
        """Switch between the BEHAVIOR VLA and planner controller configs."""

        if target not in {_CONTROLLER_VLA, _CONTROLLER_PLANNER}:
            raise ValueError("target controller must be vla or planner")
        if self._official_success_latched:
            raise RuntimeError("controller switch rejected after official success")
        if self._controller_state == target:
            return {"from": target, "to": target, "changed": False}
        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot is unavailable")
        reload_controllers = getattr(robot, "reload_controllers", None)
        if not callable(reload_controllers):
            raise RuntimeError("R1Pro controller reload API is unavailable")
        self._controller_state = _CONTROLLER_SWITCHING
        try:
            if target == _CONTROLLER_PLANNER:
                report = self._reload_base_controller_position()
            else:
                if self._velocity_controller_config is None:
                    raise RuntimeError(
                        "initial velocity controller config is unavailable"
                    )
                reload_controllers(deepcopy(self._velocity_controller_config))
                self._base_controller_mode = "velocity"
                report = {"from": "position", "to": "velocity"}
            self._require_planner().on_runtime_state_changed()
            self._controller_state = target
            self._action_source = "pi0_vla" if target == _CONTROLLER_VLA else "planner"
            self._vla_actions_enabled = target == _CONTROLLER_VLA
            return {**report, "changed": True}
        except Exception:
            self._controller_state = _CONTROLLER_FAILED
            raise

    def _capture_official_success(self, info: Any) -> dict[str, Any] | None:
        """Persist the first raw success without freezing an active Pi0 call."""

        if self._official_success_receipt is not None:
            self._official_success_latched = True
            return self._official_success_receipt
        if not _raw_success(info):
            return None
        raw_done = _wire_safe(info.get("done")) if isinstance(info, dict) else None
        receipt = {
            "schema_version": 1,
            "source": 'info["done"]["success"]',
            "run_nonce": self._run_nonce,
            "attempt_nonce": self._attempt_nonce,
            "attempt_index": int(self._attempt_index),
            "env_step": int(self._env_steps),
            "raw_done": raw_done,
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            _canonical_json_bytes(receipt)
        ).hexdigest()
        _write_json_atomic(self._official_success_receipt_path, receipt)
        self._official_success_receipt = receipt
        self._official_success_latched = True
        return receipt

    def _freeze_official_success_runtime(self) -> None:
        """Freeze physical control only after a successful invocation boundary."""

        if not self._official_success_latched or not isinstance(
            self._official_success_receipt, dict
        ):
            raise RuntimeError(
                "official-success runtime freeze requires an immutable receipt"
            )
        self._motion_frozen = True
        self._controller_state = _CONTROLLER_FROZEN
        self._action_source = "frozen"
        self._vla_actions_enabled = False
        self._done = True
        try:
            self._finalize_video_segment()
            self._video_sealed = bool(
                self._video_error is None
                and self._video_path.is_file()
                and self._video_path.stat().st_size > 0
            )
        except Exception:
            logger.exception("best-effort success video sealing failed")

    def _latch_official_success(self, info: Any) -> dict[str, Any] | None:
        """Preserve raw success and immediately freeze outside active Pi0 work."""

        if self._official_success_receipt is None and not _raw_success(info):
            return None
        receipt = self._capture_official_success(info)
        if receipt is None:
            return None
        self._freeze_official_success_runtime()
        return receipt

    def save_robot_state_checkpoint(
        self,
        *,
        semantic_label: str | None = None,
        terminal_failure: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Persist one synchronized RGB-D visual anchor without simulator state."""

        if semantic_label is not None:
            if not isinstance(semantic_label, str) or not semantic_label.strip():
                raise ValueError("semantic_label must be a non-empty string")
            semantic_label = semantic_label.strip()
            if len(semantic_label) > 128:
                raise ValueError("semantic_label must contain at most 128 characters")
        failure_declaration: dict[str, str] | None = None
        evidence_frame = None
        evidence_camera = None
        if terminal_failure is not None:
            terminal_policy = self._active_task_spec().terminal_failure_policy
            if terminal_policy is None:
                raise ValueError(
                    f"{self._active_task_spec().task_name} does not define a visual "
                    "terminal-failure policy"
                )
            if not isinstance(terminal_failure, dict):
                raise ValueError("terminal_failure must be an object")
            required = {"condition", "cause", "camera", "frame_id"}
            if set(terminal_failure) != required:
                raise ValueError(
                    "terminal_failure requires exactly condition, cause, camera, "
                    "and frame_id"
                )
            failure_declaration = {
                key: str(terminal_failure[key]).strip() for key in sorted(required)
            }
            if failure_declaration["condition"] != terminal_policy.condition:
                raise ValueError(
                    f"terminal_failure.condition must be {terminal_policy.condition}"
                )
            if failure_declaration["cause"] not in terminal_policy.causes:
                raise ValueError("terminal_failure.cause is invalid")
            if failure_declaration["camera"] not in terminal_policy.cameras:
                raise ValueError("terminal_failure.camera is invalid")
            if not failure_declaration["frame_id"]:
                raise ValueError("terminal_failure.frame_id must be non-empty")
            evidence_camera = self._resolve_camera_role(failure_declaration["camera"])
            evidence_frame = self._frame_cache.get_current(
                evidence_camera,
                failure_declaration["frame_id"],
            )
            if evidence_frame.frame_id not in getattr(
                self, "_public_observed_frame_ids", set()
            ):
                raise RuntimeError(
                    "terminal failure evidence must reference a public observe result"
                )
            if int(evidence_frame.step_index) != int(self._env_steps):
                raise RuntimeError(
                    "terminal failure evidence frame is not from the current env step"
                )
        if failure_declaration is None:
            self._refresh_observation_without_step()
        self._visual_checkpoint_counter = (
            int(getattr(self, "_visual_checkpoint_counter", 0)) + 1
        )
        visual_checkpoint_id = (
            f"visual_checkpoint_{self._visual_checkpoint_counter:03d}"
        )
        root = self._output_dir / "visual_checkpoints" / visual_checkpoint_id
        cameras: dict[str, dict[str, Any]] = {}
        public_images: dict[str, dict[str, Any]] = {}
        capture_group_ids: set[str] = set()
        for camera in ("head", "left_wrist", "right_wrist"):
            payload = self._frame_cache.observe_payload(camera)
            rgb = payload.get("_image_bytes")
            depth = payload.get("_depth_image_bytes")
            if not isinstance(rgb, bytes) or not isinstance(depth, bytes):
                raise RuntimeError(f"visual checkpoint omitted RGB-D for {camera}")
            group = payload.get("capture_group")
            group_id = group.get("id") if isinstance(group, dict) else None
            if not isinstance(group_id, str) or not group_id:
                raise RuntimeError("visual checkpoint capture group is unavailable")
            capture_group_ids.add(group_id)
            frame = self._frame_cache.get_current(camera, str(payload.get("frame_id")))
            if int(frame.step_index) != int(self._env_steps):
                raise RuntimeError("visual checkpoint frame is not current")
            rgb_path = root / f"{camera}_rgb.png"
            depth_path = root / f"{camera}_depth.png"
            _write_bytes_atomic(rgb_path, rgb)
            _write_bytes_atomic(depth_path, depth)
            cameras[camera] = {
                "camera": camera,
                "frame_id": frame.frame_id,
                "capture_group_id": group_id,
                "capture_env_step": int(frame.step_index),
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
            }
            public_images[camera] = {
                **cameras[camera],
                "_image_bytes": rgb,
                "_depth_image_bytes": depth,
            }
        if len(capture_group_ids) != 1:
            raise RuntimeError("visual checkpoint cameras are not synchronized")
        capture_group_id = next(iter(capture_group_ids))
        if (
            failure_declaration is not None
            and evidence_frame is not None
            and capture_group_id != evidence_frame.capture_group_id
        ):
            raise RuntimeError(
                "terminal failure checkpoint does not match the cited capture group"
            )
        metadata_path = root / "metadata.json"
        metadata = {
            "schema_version": 1,
            "kind": "visual_checkpoint",
            "visual_checkpoint_id": visual_checkpoint_id,
            "semantic_label": semantic_label,
            "capture_group_id": capture_group_id,
            "env_step": int(self._env_steps),
            "cameras": cameras,
            "contains_simulator_state": False,
            "authorizes_motion": False,
            "terminal_failure": failure_declaration,
        }
        _write_json_atomic(metadata_path, metadata)
        task_success = bool(_raw_success(self._last_info))
        terminal_failure_receipt = None
        stop_reason = "saved_visual_checkpoint"
        if task_success:
            self._latch_official_success(self._last_info)
            stop_reason = "task_success"
        elif failure_declaration is not None:
            if evidence_frame is None or evidence_camera is None:
                raise RuntimeError("terminal failure evidence binding is unavailable")
            terminal_failure_receipt = {
                "schema_version": 1,
                "source": "llm_fresh_visual_observation",
                "condition": failure_declaration["condition"],
                "cause": failure_declaration["cause"],
                "camera": failure_declaration["camera"],
                "frame_id": evidence_frame.frame_id,
                "capture_group_id": evidence_frame.capture_group_id,
                "env_step": int(evidence_frame.step_index),
                "visual_checkpoint_id": visual_checkpoint_id,
                "visual_checkpoint_capture_group_id": capture_group_id,
                "visual_checkpoint_metadata_sha256": hashlib.sha256(
                    metadata_path.read_bytes()
                ).hexdigest(),
                "images_sha256": {
                    camera: {
                        "rgb": hashlib.sha256(
                            Path(values["rgb_path"]).read_bytes()
                        ).hexdigest(),
                        "depth": hashlib.sha256(
                            Path(values["depth_path"]).read_bytes()
                        ).hexdigest(),
                    }
                    for camera, values in cameras.items()
                },
                "run_nonce": self._run_nonce,
                "attempt_nonce": self._attempt_nonce,
                "attempt_index": int(self._attempt_index),
                "task_success": False,
                "official_success_source": 'info["done"]["success"]',
            }
            terminal_failure_receipt["receipt_sha256"] = hashlib.sha256(
                _canonical_json_bytes(terminal_failure_receipt)
            ).hexdigest()
            _write_json_atomic(
                self._terminal_failure_receipt_path,
                terminal_failure_receipt,
            )
            self._terminal_failure_receipt = terminal_failure_receipt
            self._motion_frozen = True
            self._controller_state = _CONTROLLER_FROZEN
            self._action_source = "frozen"
            self._vla_actions_enabled = False
            self._done = True
            stop_reason = terminal_policy.condition
            try:
                self._finalize_video_segment()
                self._video_sealed = bool(
                    self._video_error is None
                    and self._video_path.is_file()
                    and self._video_path.stat().st_size > 0
                )
            except Exception:
                logger.exception("best-effort terminal failure video sealing failed")
        return {
            "_finish": bool(task_success or terminal_failure_receipt is not None),
            "primitive_success": True,
            "task_success": task_success,
            "official_success_source": 'info["done"]["success"]',
            "stop_reason": stop_reason,
            "runner_termination_reason": (
                "official_task_success"
                if task_success
                else terminal_policy.runner_reason
                if terminal_failure_receipt is not None
                else None
            ),
            "visual_checkpoint_id": visual_checkpoint_id,
            "semantic_label": semantic_label,
            "capture_group_id": capture_group_id,
            "env_step": int(self._env_steps),
            "metadata_path": str(metadata_path),
            "cameras": cameras,
            "images": public_images,
            "terminal_failure_receipt": terminal_failure_receipt,
            "total_env_steps": int(self._env_steps),
        }

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
        self._base_controller_mode = "position"
        return {
            "from": "velocity",
            "to": "position",
        }

    def _persist_pi0_nav_pick_views(
        self, *, chunk_index: int, validator: dict[str, Any]
    ) -> dict[str, Any]:
        root = (
            self._output_dir
            / "vla_calls"
            / f"call_{int(self._active_vla_call_index or 0):03d}"
            / "visual_review"
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
                        "local_grasp_success",
                        "capability",
                    )
                }
            ),
            "views": views,
        }
        metadata_path = root / "metadata.json"
        metadata["metadata_path"] = str(metadata_path)
        _write_json_atomic(metadata_path, metadata)
        return {
            "capture_group_id": capture_group_id,
            "metadata_path": str(metadata_path),
            "views": views,
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
        matrix, _source, _render_bound = self._camera_to_world_with_source(
            camera=camera,
            payload=payload,
            sensor=sensor,
        )
        return matrix

    def _camera_to_world_with_source(
        self,
        *,
        camera: str,
        payload: dict[str, Any],
        sensor: Any | None,
    ) -> tuple[np.ndarray, str, bool]:
        """Resolve camera pose while preserving pixel-binding provenance."""

        # Explicit Kit view matrices and the sensor's render annotator are tied
        # to the pixels just returned. Pose-like payload fields may already
        # reflect a newer articulation state, especially for wrist cameras.
        for name in ("view_matrix", "view_transform", "world_to_camera"):
            view = _payload_matrix(payload, (name,))
            if view is not None:
                return (
                    np.linalg.inv(view.T),
                    f"payload_{name}",
                    True,
                )
        if sensor is not None:
            render_matrix = _sensor_render_camera_to_world(sensor)
            if render_matrix is not None:
                return render_matrix, "sensor_cameraViewTransform", True
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
            return direct, "payload_pose_matrix_fallback", False
        if sensor is not None:
            sensor_matrix = _sensor_camera_to_world(sensor)
            if sensor_matrix is not None:
                return sensor_matrix, "live_sensor_pose_fallback", False
        pose = payload.get("pose") or payload.get("camera_pose")
        if isinstance(pose, dict) and "position" in pose and "orientation" in pose:
            return (
                _matrix_from_pose(pose["position"], pose["orientation"]),
                "payload_pose_fallback",
                False,
            )
        raise CameraGeometryError(f"camera pose unavailable for {camera}")

    @staticmethod
    def _live_link_world_transform(link: Any, *, reference: str) -> np.ndarray:
        getter = getattr(link, "get_position_orientation", None)
        if not callable(getter):
            raise CameraGeometryError(f"{reference} live pose is unavailable")
        position, orientation = getter()
        return validated_rigid_transform(
            _matrix_from_pose(position, orientation),
            name=f"{reference} live world transform",
        )

    def _capture_r1pro_hand_reference_transforms(
        self,
        *,
        camera_to_world_by_camera: dict[str, Any],
        camera_pose_lineage_by_camera: dict[str, Any],
        raw_proprio: Any,
        render_sync_iterations: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Snapshot both hands' complete live link transforms for one RGB-D group.

        These matrices remain simulator-internal. Public probes expose only
        frame-bound distances and a digest, never simulator prim paths.
        """

        references_unavailable = {
            "schema_version": 1,
            "available": False,
            "reason": "hand_geometry_sync_certificate_unavailable",
            "env_step": int(self._env_steps),
            "source": "capture_time_live_r1pro_link_transforms",
            "hands": {},
        }
        certificate_unavailable = {
            "schema_version": 1,
            "available": False,
            "synchronized": False,
            "reason": "r1pro_live_hand_geometry_unavailable",
            "env_step": int(self._env_steps),
            "render_sync_iterations": int(render_sync_iterations),
            "translation_tolerance_m": _HAND_GEOMETRY_TRANSLATION_TOLERANCE_M,
            "rotation_tolerance_deg": _HAND_GEOMETRY_ROTATION_TOLERANCE_DEG,
            "finger_joint_tolerance_m": (_HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M),
            "source": "render_sync_plus_official_r1pro_fixed_extrinsics",
            "hands": {},
        }
        robot = self._robot()
        if robot is None:
            return references_unavailable, certificate_unavailable
        links = getattr(robot, "links", None)
        eef_names = getattr(robot, "eef_link_names", None)
        finger_names = getattr(robot, "finger_link_names", None)
        gripper_indices = getattr(robot, "gripper_control_idx", None)
        if (
            not isinstance(links, dict)
            or not isinstance(eef_names, dict)
            or not isinstance(finger_names, dict)
            or not isinstance(gripper_indices, dict)
        ):
            return references_unavailable, certificate_unavailable

        raw = np.asarray(_numpy_tree(raw_proprio), dtype=np.float64).reshape(-1)
        qpos = np.asarray(
            _numpy_tree(robot.get_joint_positions()), dtype=np.float64
        ).reshape(-1)
        if (
            raw.size < RAW_PROPRIO_SEGMENTS["right_gripper"].stop
            or not np.isfinite(raw).all()
            or not np.isfinite(qpos).all()
        ):
            return references_unavailable, certificate_unavailable

        hands: dict[str, Any] = {}
        hand_certificates: dict[str, Any] = {}
        expected = r1pro_wrist_camera_reference_transforms()
        try:
            for hand in ("left", "right"):
                palm_name = f"{hand}_gripper_link"
                grip_name = str(eef_names[hand])
                current_finger_names = list(finger_names[hand])
                if len(current_finger_names) != 2:
                    raise CameraGeometryError(
                        f"{hand} hand does not expose exactly two finger roots"
                    )
                if (
                    palm_name not in links
                    or grip_name not in links
                    or any(name not in links for name in current_finger_names)
                ):
                    raise CameraGeometryError(
                        f"{hand} hand reference links are unavailable"
                    )
                palm = self._live_link_world_transform(
                    links[palm_name],
                    reference=f"{hand} palm",
                )
                grip_point = self._live_link_world_transform(
                    links[grip_name],
                    reference=f"{hand} grip point",
                )
                finger_roots = [
                    self._live_link_world_transform(
                        links[name],
                        reference=f"{hand} finger root {index + 1}",
                    )
                    for index, name in enumerate(current_finger_names)
                ]
                camera_to_world = validated_rigid_transform(
                    camera_to_world_by_camera[f"{hand}_wrist"],
                    name=f"{hand} wrist camera-to-world transform",
                )
                camera_pose_lineage = camera_pose_lineage_by_camera.get(f"{hand}_wrist")
                camera_pose_render_bound = bool(
                    isinstance(camera_pose_lineage, dict)
                    and camera_pose_lineage.get("render_bound") is True
                    and int(camera_pose_lineage.get("env_step", -1))
                    == int(self._env_steps)
                    and int(camera_pose_lineage.get("render_sync_iterations", -1))
                    >= _HAND_GEOMETRY_SYNC_RENDER_ITERATIONS
                    and camera_pose_lineage.get("source")
                    in {
                        "payload_view_matrix",
                        "payload_view_transform",
                        "payload_world_to_camera",
                        "sensor_cameraViewTransform",
                    }
                )
                palm_residual = rigid_transform_residual(
                    np.linalg.inv(palm) @ camera_to_world,
                    expected["palm_from_camera"],
                )
                grip_residual = rigid_transform_residual(
                    np.linalg.inv(grip_point) @ camera_to_world,
                    expected["grip_point_from_camera"],
                )
                indices = np.asarray(
                    _numpy_tree(gripper_indices[hand]), dtype=np.int64
                ).reshape(-1)
                if (
                    indices.shape != (2,)
                    or np.any(indices < 0)
                    or int(indices.max()) >= qpos.size
                ):
                    raise CameraGeometryError(
                        f"{hand} gripper joint indices are unavailable"
                    )
                live_finger_q = qpos[indices]
                capture_finger_q = raw[RAW_PROPRIO_SEGMENTS[f"{hand}_gripper"]]
                if capture_finger_q.shape != (2,):
                    raise CameraGeometryError(
                        f"{hand} capture finger joints are unavailable"
                    )
                finger_error = float(np.max(np.abs(live_finger_q - capture_finger_q)))
                palm_passed = bool(
                    palm_residual["translation_error_m"]
                    <= _HAND_GEOMETRY_TRANSLATION_TOLERANCE_M
                    and palm_residual["rotation_error_deg"]
                    <= _HAND_GEOMETRY_ROTATION_TOLERANCE_DEG
                )
                grip_passed = bool(
                    grip_residual["translation_error_m"]
                    <= _HAND_GEOMETRY_TRANSLATION_TOLERANCE_M
                    and grip_residual["rotation_error_deg"]
                    <= _HAND_GEOMETRY_ROTATION_TOLERANCE_DEG
                )
                finger_passed = bool(
                    finger_error <= _HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M
                )
                hand_certificates[hand] = {
                    "passed": bool(
                        camera_pose_render_bound
                        and palm_passed
                        and grip_passed
                        and finger_passed
                    ),
                    "camera_pose_source": (
                        camera_pose_lineage.get("source")
                        if isinstance(camera_pose_lineage, dict)
                        else None
                    ),
                    "camera_pose_render_bound": camera_pose_render_bound,
                    "palm_from_camera": {
                        **palm_residual,
                        "passed": palm_passed,
                    },
                    "grip_point_from_camera": {
                        **grip_residual,
                        "passed": grip_passed,
                    },
                    "finger_joint_capture_match": {
                        "max_abs_error_m": finger_error,
                        "passed": finger_passed,
                    },
                }
                hands[hand] = {
                    "palm": palm.tolist(),
                    "grip_point": grip_point.tolist(),
                    # R1Pro declares finger_link_names in finger1/finger2 order.
                    # Their live link origins track the current articulation.
                    "finger_roots": [transform.tolist() for transform in finger_roots],
                }
        except (CameraGeometryError, KeyError, TypeError, ValueError):
            return references_unavailable, certificate_unavailable

        synchronized = bool(
            int(render_sync_iterations) >= _HAND_GEOMETRY_SYNC_RENDER_ITERATIONS
            and all(certificate["passed"] for certificate in hand_certificates.values())
        )
        certificate = {
            "schema_version": 1,
            "available": True,
            "synchronized": synchronized,
            "reason": None if synchronized else "hand_geometry_sync_residual_failed",
            "env_step": int(self._env_steps),
            "render_sync_iterations": int(render_sync_iterations),
            "translation_tolerance_m": _HAND_GEOMETRY_TRANSLATION_TOLERANCE_M,
            "rotation_tolerance_deg": _HAND_GEOMETRY_ROTATION_TOLERANCE_DEG,
            "finger_joint_tolerance_m": (_HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M),
            "source": "render_sync_plus_official_r1pro_fixed_extrinsics",
            "hands": hand_certificates,
        }
        if not synchronized:
            return references_unavailable, certificate
        return (
            {
                "schema_version": 1,
                "available": True,
                "env_step": int(self._env_steps),
                "source": "capture_time_live_r1pro_link_transforms",
                "hands": hands,
            },
            certificate,
        )

    def _record_rgbd_frames(
        self,
        raw_observations: Any,
        observation: dict[str, Any],
        *,
        strict: bool = False,
        synchronize_hand_geometry: bool = False,
        render_sync_iterations: int = 0,
    ) -> None:
        raw = _first_env_value(raw_observations)
        if raw is None:
            if strict:
                raise RuntimeError(
                    "canonical RGB-D refresh returned no raw observation"
                )
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
            camera_pose_lineage: dict[str, dict[str, Any]] = {}
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
                (
                    camera_to_world,
                    camera_pose_source,
                    camera_pose_render_bound,
                ) = self._camera_to_world_with_source(
                    camera=camera,
                    payload=payload,
                    sensor=sensor,
                )
                frames[camera] = {
                    "rgb": rgb_array,
                    "depth_m": depth,
                    "intrinsics": intrinsics,
                    "camera_to_world": camera_to_world,
                }
                camera_pose_lineage[camera] = {
                    "source": camera_pose_source,
                    "render_bound": bool(camera_pose_render_bound),
                    "env_step": int(self._env_steps),
                    "render_sync_iterations": int(render_sync_iterations),
                }
            raw_proprio = np.asarray(
                _numpy_tree(observation["states"]), dtype=np.float64
            ).reshape(-1)
            compact_proprio = extract_policy_state(raw_proprio)
            if synchronize_hand_geometry:
                (
                    hand_reference_transforms,
                    hand_geometry_sync_certificate,
                ) = self._capture_r1pro_hand_reference_transforms(
                    camera_to_world_by_camera={
                        camera: frames[camera]["camera_to_world"]
                        for camera in expected_cameras
                    },
                    camera_pose_lineage_by_camera=camera_pose_lineage,
                    raw_proprio=raw_proprio,
                    render_sync_iterations=render_sync_iterations,
                )
            else:
                hand_reference_transforms = {
                    "schema_version": 1,
                    "available": False,
                    "reason": "render_sync_not_requested",
                    "env_step": int(self._env_steps),
                    "source": "capture_time_live_r1pro_link_transforms",
                    "hands": {},
                }
                hand_geometry_sync_certificate = {
                    "schema_version": 1,
                    "available": False,
                    "synchronized": False,
                    "reason": "render_sync_not_requested",
                    "env_step": int(self._env_steps),
                    "render_sync_iterations": int(render_sync_iterations),
                    "translation_tolerance_m": (_HAND_GEOMETRY_TRANSLATION_TOLERANCE_M),
                    "rotation_tolerance_deg": (_HAND_GEOMETRY_ROTATION_TOLERANCE_DEG),
                    "finger_joint_tolerance_m": (
                        _HAND_GEOMETRY_FINGER_JOINT_TOLERANCE_M
                    ),
                    "source": ("render_sync_plus_official_r1pro_fixed_extrinsics"),
                    "hands": {},
                }
            self._frame_cache.add_capture_group(
                frames=frames,
                step_index=self._env_steps,
                capture_metadata={
                    "proprio": {
                        "values": compact_proprio.astype(float).tolist(),
                        "dimension": int(compact_proprio.size),
                        "layout": "POLICY_STATE_SEGMENTS",
                        "segments": segment_ranges(POLICY_STATE_SEGMENTS),
                    },
                    "r1pro_hand_reference_transforms": hand_reference_transforms,
                    "hand_geometry_sync_certificate": (hand_geometry_sync_certificate),
                    "camera_pose_lineage": camera_pose_lineage,
                    "render_sync_iterations": int(render_sync_iterations),
                },
            )
            self._last_capture_step = self._env_steps
        except Exception as exc:
            # Atomicity is intentional: never publish one camera from a newer
            # simulator step beside two cameras from an older step.
            logger.exception(
                "failed to cache atomic BEHAVIOR RGB-D capture group at sim step %s",
                self._env_steps,
            )
            if strict:
                cache = getattr(self, "_frame_cache", None)
                if cache is not None:
                    cache.clear()
                self._last_capture_step = None
                raise RuntimeError(
                    "canonical RGB-D capture group validation failed"
                ) from exc

    def _video_writer_loop(self) -> None:
        """Resize and encode queued capture groups away from the Env thread."""

        import imageio.v2 as imageio

        while True:
            item = self._video_queue.get()
            try:
                if item is self._video_queue_stop:
                    break
                observation = item
                head = np.asarray(
                    observation["main_images"], dtype=np.uint8
                )
                if head.ndim != 3 or head.shape[2] < 3:
                    raise RuntimeError(
                        f"planner head video must be HxWxC, got {head.shape}"
                    )
                wrists = np.asarray(
                    observation["wrist_images"], dtype=np.uint8
                )
                if wrists.ndim != 4 or wrists.shape[0] != 2:
                    raise RuntimeError(
                        "planner video requires synchronized left/right wrist RGB"
                    )
                height, width = head.shape[:2]
                left_wrist = _resize_video_tile(
                    wrists[0], height=height, width=width
                )
                right_wrist = _resize_video_tile(
                    wrists[1], height=height, width=width
                )
                frame = np.zeros(
                    (height * 2, width * 2, 3), dtype=np.uint8
                )
                frame[:height, :width] = head[..., :3]
                frame[:height, width:] = left_wrist
                frame[height:, :width] = right_wrist
                self._video_source_shapes.update(
                    {
                        "head": list(head.shape),
                        "left_wrist": list(wrists[0].shape),
                        "right_wrist": list(wrists[1].shape),
                        "output": list(frame.shape),
                    }
                )
                if self._video_writer is None:
                    self._video_path.parent.mkdir(parents=True, exist_ok=True)
                    self._video_writer = imageio.get_writer(
                        self._video_path, fps=15
                    )
                self._video_writer.append_data(frame)
                self._video_frames += 1
            except Exception as exc:
                self._video_error = f"{type(exc).__name__}: {exc}"
                logger.exception("failed to append BEHAVIOR video frame")
            finally:
                self._video_queue.task_done()
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception as exc:
                self._video_error = f"{type(exc).__name__}: {exc}"
                logger.exception("failed to close BEHAVIOR video writer")
            finally:
                self._video_writer = None

    def _append_video(self, observation: dict[str, Any]) -> None:
        if self._video_error is not None or self._video_sealed:
            return
        try:
            if self._video_queue_closed:
                return
            queued = {
                "main_images": np.asarray(
                    observation["main_images"], dtype=np.uint8
                ).copy(),
                "wrist_images": np.asarray(
                    observation["wrist_images"], dtype=np.uint8
                ).copy(),
            }
            self._video_queue.put_nowait(queued)
        except Full:
            self._video_frames_dropped += 1
        except Exception as exc:
            self._video_error = f"{type(exc).__name__}: {exc}"
            logger.exception("failed to queue BEHAVIOR video frame")

    def _finalize_video_segment(self) -> None:
        if not self._video_queue_closed:
            self._video_queue_closed = True
            self._video_queue.put(self._video_queue_stop)
        worker = getattr(self, "_video_worker", None)
        if isinstance(worker, threading.Thread) and worker.is_alive():
            worker.join(timeout=30.0)
            if worker.is_alive() and self._video_error is None:
                self._video_error = "video writer did not stop within 30 seconds"
        video_meta = {
            "path": str(self._video_path),
            "fps": 15,
            "sample_every_env_steps": self._planner_video_interval_steps,
            "frames": self._video_frames,
            "dropped_frames": self._video_frames_dropped,
            "error": self._video_error,
            "source_shapes": dict(self._video_source_shapes),
            "layout": "2x2:head,left_wrist/right_wrist,blank",
        }
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            self._video_path.with_name(f"{self._video_path.stem}_meta.json"),
            video_meta,
        )
        _write_json_atomic(self._video_path.parent / "video_meta.json", video_meta)

    def reset(self) -> tuple[dict[str, Any], Any]:
        if bool(getattr(self, "_reset_completed", False)):
            raise RuntimeError("the env process permits one lifecycle initialization")
        started_at = time.monotonic()
        logger.info("BEHAVIOR reset started on thread %s", threading.get_ident())
        raw_observations, infos = self._env.env_reset()
        observation = _single_observation(self._env._wrap_obs(raw_observations))
        self._done = False
        self._env_steps = 0
        self._last_observation = observation
        self._last_info = _numpy_tree(infos[0])
        self._video_sealed = False
        self._base_controller_mode = "velocity"
        robot = self._robot()
        controller_config = (
            None if robot is None else getattr(robot, "_controller_config", None)
        )
        if controller_config is None:
            raise RuntimeError("initial velocity controller config is unavailable")
        self._velocity_controller_config = deepcopy(controller_config)
        self._action_source = "pi0_vla"
        self._vla_actions_enabled = True
        self._controller_state = _CONTROLLER_VLA
        self._public_capture_sequence = 0
        self._latest_unconsumed_public_capture_receipt = None
        self._clear_active_vla_invocation_state()
        self._motion_frozen = False
        self._dashboard_planning_admitted = False
        self._dashboard_execute_receipts = {}
        self._dashboard_env_step_latency = None
        self._projection_receipts.clear()
        self._public_observed_frame_ids.clear()
        self._latest_public_head_frame_id = None
        self._latest_public_observation_lineage = None
        self._official_success_latched = False
        self._official_success_receipt = None
        self._terminal_failure_receipt = None
        restore_path_value = os.environ.get("RPENT_BEHAVIOR_RESTORE_STATE")
        if restore_path_value:
            raise RuntimeError(
                "simulator-state restore is unsupported; the outer Explore "
                "harness must start a fresh env process"
            )
        self._record_rgbd_frames(raw_observations, observation, strict=True)
        self._append_video(observation)
        self._planner_warmup_report = None
        if self._controller_mode == "hybrid":
            planner_warmup = self._planner.warmup()
            if planner_warmup.get("status") != "complete":
                raise RuntimeError(
                    "hybrid R1Pro cuRobo warmup did not complete successfully"
                )
            self._planner_warmup_report = planner_warmup
        logger.info(
            "BEHAVIOR reset completed in %.1fs on thread %s",
            time.monotonic() - started_at,
            threading.get_ident(),
        )
        self._reset_completed = True
        return observation, self._info_with_accounting()

    def prepare_vla_invocation(
        self,
        *,
        invocation_id: str,
        call_index: int,
        vla_status: dict[str, Any] | None,
        current_object_visual_check: dict[str, Any] | None = None,
        baseline_internal_authorization: bool = False,
    ) -> dict[str, Any]:
        """Re-arm the VLA controller without external physical-state gates."""

        if not isinstance(invocation_id, str) or not invocation_id:
            raise ValueError("invocation_id must be a non-empty string")
        if isinstance(call_index, bool) or int(call_index) < 1:
            raise ValueError("call_index must be positive")
        if not isinstance(baseline_internal_authorization, bool):
            raise TypeError("baseline_internal_authorization must be boolean")
        del current_object_visual_check
        if self._official_success_latched or _raw_success(self._last_info):
            receipt = self._latch_official_success(self._last_info)
            self._clear_active_vla_invocation_state()
            return {
                "primitive_success": False,
                "task_success": True,
                "official_success_source": 'info["done"]["success"]',
                "official_success_receipt": _wire_safe(receipt),
                "stop_reason": "official_success_latched",
                "failed_preconditions": ["official_success_latched"],
                "attempt_index": self._attempt_index,
                "attempt_nonce": self._attempt_nonce,
                "total_env_steps": int(self._env_steps),
                "vla_actions_enabled": False,
            }
        if vla_status is None:
            if self._active_vla_invocation is not None:
                self._clear_active_vla_invocation_state()
            controller = self._switch_controller(_CONTROLLER_VLA)
            self._active_vla_invocation = invocation_id
            self._active_vla_call_index = int(call_index)
            self._pending_vla_baseline_internal_authorization = bool(
                baseline_internal_authorization
            )
            return {
                "primitive_success": True,
                "task_success": False,
                "official_success_source": 'info["done"]["success"]',
                "failed_preconditions": [],
                "attempt_index": self._attempt_index,
                "attempt_nonce": self._attempt_nonce,
                "total_env_steps": int(self._env_steps),
                "stop_reason": "vla_rearm_preflight_passed",
                "invocation_id": invocation_id,
                "call_index": int(call_index),
                "vla_actions_enabled": bool(self._vla_actions_enabled),
                "controller_transition": controller,
                "baseline_internal_authorization": bool(
                    baseline_internal_authorization
                ),
            }
        facts = {
            "primitive_success": True,
            "task_success": bool(self._official_success_latched),
            "official_success_source": 'info["done"]["success"]',
            "failed_preconditions": [],
            "attempt_index": self._attempt_index,
            "attempt_nonce": self._attempt_nonce,
            "total_env_steps": int(self._env_steps),
        }
        if self._active_vla_invocation != invocation_id:
            raise RuntimeError("VLA re-arm confirmation does not match preflight")
        if self._active_vla_call_index != int(call_index):
            raise RuntimeError(
                "VLA re-arm confirmation call index does not match preflight"
            )
        pending_is_baseline = bool(self._pending_vla_baseline_internal_authorization)
        if pending_is_baseline != baseline_internal_authorization:
            raise RuntimeError("VLA re-arm authorization mode changed after preflight")
        if (
            not isinstance(vla_status, dict)
            or vla_status.get("actions_enabled") is not True
        ):
            raise RuntimeError("VLA server enable confirmation is invalid")
        self._controller_state = _CONTROLLER_VLA
        self._action_source = "pi0_vla"
        self._vla_actions_enabled = True
        return {
            **facts,
            "stop_reason": "vla_runtime_rearmed",
            "invocation_id": invocation_id,
            "call_index": int(call_index),
            "vla_actions_enabled": True,
            "baseline_internal_authorization": bool(baseline_internal_authorization),
        }

    def pi0_nav_pick_chunk_step(
        self,
        actions: Any,
        *,
        chunk_index: int,
    ) -> tuple[Any, Any, bool, bool, Any]:
        """Execute one admitted Pi0 chunk, stopping on the exact success step."""

        if self._active_vla_invocation is None:
            raise RuntimeError("Pi0 chunk requires prepare_vla_invocation")
        if self._official_success_latched or _raw_success(self._last_info):
            raise RuntimeError(
                "raw task success is terminal; no further Pi0 chunk is admissible"
            )
        if self._controller_state != _CONTROLLER_VLA or not self._vla_actions_enabled:
            raise RuntimeError("VLA controller is not active")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, (int, np.integer))
            or int(chunk_index) < 1
        ):
            raise ValueError("chunk_index must be positive")
        action_array = validate_action_chunk(actions, max_horizon=32)
        if action_array.shape != (32, 23):
            raise ValueError(
                f"pi0_nav_pick requires one complete [32,23] chunk, got {action_array.shape}"
            )
        self._motion_in_flight = True
        try:
            ret = self._step_action_chunk(
                action_array,
                observe_final=True,
                pi0_nav_pick=True,
            )
            pickle.dumps(ret, protocol=pickle.HIGHEST_PROTOCOL)
            return ret
        except Exception:
            self._vla_actions_enabled = False
            self._clear_active_vla_invocation_state()
            if self._official_success_latched or _raw_success(self._last_info):
                self._latch_official_success(self._last_info)
            else:
                self._controller_state = _CONTROLLER_FAILED
                self._action_source = "failed"
            self._finalize_video_segment()
            self._video_sealed = bool(
                self._video_error is None
                and self._video_path.is_file()
                and self._video_path.stat().st_size > 0
            )
            raise
        finally:
            self._motion_in_flight = False

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
        pi0_nav_pick: bool = False,
    ) -> tuple[Any, Any, bool, bool, Any]:
        """Send one 23D chunk and stop only on raw environment lifecycle facts."""

        import torch

        if self._done:
            raise RuntimeError("env.chunk_step called after episode stop")
        action_array = validate_action_chunk(actions)
        action_tensor = torch.as_tensor(action_array, dtype=torch.float32).reshape(
            1, action_array.shape[0], action_array.shape[1]
        )
        final_observation = self._last_observation
        final_reward: Any = None
        official_info: Any = {}
        terminated = False
        truncated = False
        executed_steps = 0
        terminal_capture: dict[str, Any] | None = None
        terminal_capture_error: str | None = None
        for step_index in range(action_tensor.shape[1]):
            is_last = step_index == action_tensor.shape[1] - 1
            interval = self._planner_video_interval_steps
            record_observation = bool(
                observe_final
                and (
                    is_last
                    or (
                        pi0_nav_pick
                        and (self._env_steps + 1) % interval == 0
                    )
                )
            )
            need_observation = record_observation
            step_action = action_tensor[:, step_index]
            step_latency = getattr(
                self,
                "_dashboard_env_step_latency",
                None,
            )
            env_step_started = (
                time.monotonic() if isinstance(step_latency, dict) else None
            )
            step_obs, step_reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    step_action,
                    need_obs=need_observation,
                )
            )
            if isinstance(step_latency, dict) and env_step_started is not None:
                env_step_elapsed_s = max(
                    0.0,
                    time.monotonic() - env_step_started,
                )
                count = int(step_latency.get("count", 0)) + 1
                total_s = float(step_latency.get("total_s", 0.0))
                total_s += env_step_elapsed_s
                previous_min = step_latency.get("min_s")
                previous_max = step_latency.get("max_s")
                step_latency.update(
                    {
                        "count": count,
                        "total_s": total_s,
                        "min_s": (
                            env_step_elapsed_s
                            if previous_min is None
                            else min(float(previous_min), env_step_elapsed_s)
                        ),
                        "max_s": (
                            env_step_elapsed_s
                            if previous_max is None
                            else max(float(previous_max), env_step_elapsed_s)
                        ),
                    }
                )
            self._env_steps += 1
            executed_steps += 1
            executed = np.asarray(_numpy_tree(step_action), dtype=np.float32).reshape(
                23
            )
            for side in ("left", "right"):
                segment = ENV_ACTION_SEGMENTS[f"{side}_gripper"]
                self._gripper_latch[side] = float(executed[segment][0])
            official_info = step_infos[0]
            self._last_info = _numpy_tree(official_info)
            final_reward = step_reward[0]
            raw_success_this_step = _raw_success(official_info)
            terminated = bool(terminated or _scalar_bool(step_term))
            truncated = bool(truncated or _scalar_bool(step_trunc))
            if need_observation:
                if step_obs is None:
                    raise RuntimeError("requested observation was not returned")
                final_observation = _single_observation(self._env._wrap_obs(step_obs))
                self._last_observation = final_observation
                self._record_rgbd_frames(step_obs, final_observation)
                self._append_video(final_observation)
            if (
                (raw_success_this_step or terminated or truncated)
                and not need_observation
            ):
                try:
                    # This is an in-process render/get_obs on the same Env RPC,
                    # not a second task RPC and never an additional env action.
                    self._refresh_observation_without_step()
                    final_observation = self._last_observation
                    if final_observation is not None:
                        self._append_video(final_observation)
                except Exception as exc:
                    terminal_capture_error = f"{type(exc).__name__}: {exc}"
                    logger.exception("same-step terminal camera refresh failed")
            if raw_success_this_step or terminated or truncated:
                frame_cache = getattr(self, "_frame_cache", None)
                if frame_cache is not None:
                    try:
                        terminal_capture = self._cached_capture_group()
                    except Exception as exc:
                        terminal_capture_error = f"{type(exc).__name__}: {exc}"
                        # Raw success and raw lifecycle facts remain
                        # authoritative even if terminal image packaging
                        # fails. Never issue a post-terminal retry RPC.
                        logger.exception(
                            "same-step terminal capture packaging failed"
                        )
            if raw_success_this_step:
                self._latch_official_success(self._last_info)
                break
            if terminated or truncated:
                break
        if final_observation is None:
            raise RuntimeError("BEHAVIOR action chunk executed zero steps")
        if not self._official_success_latched:
            self._done = bool(terminated or truncated)
        returned_info = _wire_safe(official_info)
        if not isinstance(returned_info, dict):
            returned_info = {"raw": returned_info}
        envelope = returned_info.setdefault("_rpent", {})
        if not isinstance(envelope, dict):
            raise RuntimeError("runtime accounting envelope is invalid")
        envelope.update(
            {
                "executed_steps": int(executed_steps),
                "total_env_steps": int(self._env_steps),
            }
        )
        if terminal_capture is not None:
            envelope["terminal_capture"] = terminal_capture
        if terminal_capture_error is not None:
            envelope["terminal_capture_error"] = terminal_capture_error
        if pi0_nav_pick:
            envelope["pi0_nav_pick_monitor"] = {
                "executed_steps": int(executed_steps),
                "total_env_steps": int(self._env_steps),
                "official_success_receipt": _wire_safe(self._official_success_receipt),
                "stop_reason": (
                    "official_task_success"
                    if self._official_success_latched
                    else "environment_terminated"
                    if terminated
                    else "environment_truncated"
                    if truncated
                    else "chunk_complete"
                ),
            }
        result = _wire_safe(
            (
                final_observation,
                _numpy_tree(final_reward),
                terminated,
                truncated,
                returned_info,
            )
        )
        pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        return result

    def get_env_meta(self) -> dict[str, Any]:
        """Return only public benchmark identity, never native simulator bindings."""

        return {
            key: _wire_safe(self._meta.get(key))
            for key in (
                "suite",
                "task",
                "task_name",
                "public_seed",
                "max_episode_steps",
            )
        } | {"run_nonce": self._run_nonce}

    def _info_with_accounting(self, source_info: Any = None) -> dict[str, Any]:
        info = _wire_safe(self._last_info if source_info is None else source_info)
        if not isinstance(info, dict):
            info = {"raw": info}
        rpent = info.get("_rpent")
        if not isinstance(rpent, dict):
            rpent = {}
            info["_rpent"] = rpent
        rpent["total_env_steps"] = int(self._env_steps)
        rpent["global_env_steps"] = int(self._env_steps)
        rpent["run_nonce"] = str(getattr(self, "_run_nonce", "unbound"))
        rpent["attempt_index"] = int(getattr(self, "_attempt_index", 1))
        rpent["attempt_nonce"] = str(getattr(self, "_attempt_nonce", "unbound"))
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

    @staticmethod
    def _dashboard_capability_flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            for key in ("available", "supported", "verified", "ok"):
                flag = value.get(key)
                if isinstance(flag, bool):
                    return flag
        return False

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        """Return fail-closed manual-control capabilities for this live env."""

        planner = getattr(self, "_planner", None)
        planner_capability_method = getattr(
            planner, "dashboard_control_capabilities", None
        )
        planner_report: dict[str, Any] = {}
        planner_available = callable(planner_capability_method)
        if planner_available:
            try:
                value = planner_capability_method()
                if isinstance(value, dict):
                    planner_report = _wire_safe(value)
                else:
                    planner_available = False
            except Exception:
                logger.exception("failed to read dashboard planner capabilities")
                planner_available = False

        reset_completed = bool(getattr(self, "_reset_completed", False))
        official_success = bool(
            getattr(self, "_official_success_latched", False)
            or _raw_success(getattr(self, "_last_info", None))
        )
        stopped = bool(
            getattr(self, "_done", False)
            or getattr(self, "_motion_frozen", False)
            or getattr(self, "_terminal_failure_receipt", None) is not None
            or official_success
        )
        controller_state = str(getattr(self, "_controller_state", "unavailable"))
        controller_transfer_available = controller_state in {
            _CONTROLLER_VLA,
            _CONTROLLER_PLANNER,
        }
        hybrid_mode = getattr(self, "_controller_mode", None) == "hybrid"
        suite_is_behavior = str(getattr(self, "_meta", {}).get("suite", "")).startswith(
            "behavior"
        )
        task_spec = getattr(self, "_task_spec", None)
        task_identity = getattr(self, "_task_identity", None)
        task_identity_verified = bool(
            isinstance(task_spec, BehaviorTaskSpec)
            and isinstance(task_identity, tuple)
            and len(task_identity) == 3
            and task_identity[0] == task_spec.task_name
            and isinstance(task_identity[1], int)
            and not isinstance(task_identity[1], bool)
            and task_identity[1] == task_spec.activity_definition_id
            and isinstance(task_identity[2], int)
            and not isinstance(task_identity[2], bool)
            and task_identity[2] > 0
        )
        omni_env = getattr(getattr(self, "_env", None), "omnigibson_env", None)
        robot = self._robot() if reset_completed else None
        robot_class = type(robot).__name__.strip().lower() if robot is not None else ""
        robot_name = str(getattr(robot, "name", "")).strip().lower()
        robot_is_r1pro = bool(
            robot is not None
            and (
                robot_class in {"r1", "r1pro"}
                or "r1pro" in robot_name
                or robot_name == "r1"
            )
        )
        simulation_identity = bool(
            suite_is_behavior
            and task_identity_verified
            and omni_env is not None
            and robot_is_r1pro
        )
        position_control_ready = bool(
            controller_transfer_available
            and (
                getattr(self, "_base_controller_mode", None) == "position"
                or (
                    controller_state == _CONTROLLER_VLA
                    and callable(getattr(robot, "reload_controllers", None))
                )
            )
        )
        common_motion = bool(
            reset_completed
            and simulation_identity
            and hybrid_mode
            and planner_available
            and not stopped
        )
        observe_available = bool(
            reset_completed
            and simulation_identity
            and planner is not None
            and not stopped
        )

        def planner_flag(name: str) -> bool:
            return self._dashboard_capability_flag(planner_report.get(name))

        base_available = bool(common_motion and planner_flag("base"))
        eef_flag = planner_report.get("eef")
        eef_available = {
            hand: bool(
                common_motion
                and (
                    self._dashboard_capability_flag(eef_flag.get(hand))
                    if isinstance(eef_flag, dict)
                    else self._dashboard_capability_flag(eef_flag)
                )
            )
            for hand in ("left", "right")
        }
        warmup_report = getattr(self, "_planner_warmup_report", None)
        warmup_identity = (
            warmup_report.get("identity_warmup")
            if isinstance(warmup_report, dict)
            and isinstance(warmup_report.get("identity_warmup"), dict)
            else {}
        )
        torso_warmup = (
            warmup_identity.get("torso")
            if isinstance(warmup_identity.get("torso"), dict)
            else {}
        )
        torso_warmup_verified = bool(
            isinstance(warmup_report, dict)
            and warmup_report.get("status") == "complete"
            and torso_warmup.get("ok") is True
            and int(torso_warmup.get("active_dof_count", -1)) == 3
            and int(torso_warmup.get("env_actions_sent", -1)) == 0
            and torso_warmup.get("simulator_advanced") is False
        )
        torso_available = bool(
            common_motion
            and planner_flag("torso")
            and torso_warmup_verified
        )
        wrist_flag = planner_report.get("wrist")
        wrist_available = {
            hand: bool(
                common_motion
                and (
                    self._dashboard_capability_flag(wrist_flag.get(hand))
                    if isinstance(wrist_flag, dict)
                    else self._dashboard_capability_flag(wrist_flag)
                )
            )
            for hand in ("left", "right")
        }
        gripper_flag = planner_report.get("gripper")
        gripper_available = {
            hand: bool(
                common_motion
                and (
                    self._dashboard_capability_flag(gripper_flag.get(hand))
                    if isinstance(gripper_flag, dict)
                    else self._dashboard_capability_flag(gripper_flag)
                )
            )
            for hand in ("left", "right")
        }
        motion_available = bool(
            base_available
            or torso_available
            or any(eef_available.values())
            or any(wrist_available.values())
            or any(gripper_available.values())
        )
        threaded_ready = getattr(
            planner,
            "threaded_predicted_planning_ready",
            None,
        )
        threaded_predicted_planning = bool(
            getattr(
                self,
                "_threaded_predicted_planning_dispatcher_ready",
                False,
            )
            and callable(threaded_ready)
            and threaded_ready() is True
        )
        unavailable_reason: str | None = None
        if stopped:
            unavailable_reason = "run_finished"
        elif not reset_completed:
            unavailable_reason = "episode_not_initialized"
        elif not simulation_identity:
            unavailable_reason = "behavior_omnigibson_r1pro_unverified"
        elif not hybrid_mode:
            unavailable_reason = "manual_motion_requires_hybrid_mode"
        elif not planner_available:
            unavailable_reason = "planner_unavailable"
        elif not motion_available:
            unavailable_reason = "manual_motion_capabilities_unverified"

        return {
            "motion_available": motion_available,
            "observe_available": observe_available,
            "unavailable_reason": unavailable_reason,
            "simulation_identity": (
                "behavior_omnigibson_r1pro" if simulation_identity else None
            ),
            "planner_available": planner_available,
            "position_control_ready": position_control_ready,
            "current_base_controller_mode": getattr(
                self, "_base_controller_mode", None
            ),
            "controller_state": controller_state,
            "base_available": base_available,
            "eef_available": eef_available,
            "torso_available": torso_available,
            "torso_warmup_verified": torso_warmup_verified,
            "wrist_rotation_available": wrist_available,
            "gripper_available": gripper_available,
            "threaded_predicted_planning": threaded_predicted_planning,
            "steps": {
                "base_translation_m": BASE_TRANSLATION_STEP_M,
                "base_rotation_rad": BASE_ROTATION_STEP_RAD,
                "eef_translation_m": EEF_TRANSLATION_STEP_M,
                "torso_vertical_m": TORSO_VERTICAL_STEP_M,
                "wrist_rotation_rad": WRIST_ROTATION_STEP_RAD,
            },
            "planner": planner_report,
        }

    def _cached_capture_group(self) -> dict[str, Any]:
        """Serialize the already-published same-step three-camera capture."""

        frames_bytes: dict[str, bytes] = {}
        frame_ids: dict[str, str] = {}
        group_ids: set[str] = set()
        simulator_steps: set[int] = set()
        for camera in ("head", "left_wrist", "right_wrist"):
            payload = self._frame_cache.observe_payload(camera)
            image = payload.get("_image_bytes")
            group = payload.get("capture_group")
            group_id = group.get("id") if isinstance(group, dict) else None
            sim_step = group.get("sim_step") if isinstance(group, dict) else None
            frame_id = payload.get("frame_id")
            if not isinstance(image, bytes):
                raise RuntimeError(f"dashboard capture omitted PNG bytes for {camera}")
            if not isinstance(group_id, str) or not group_id:
                raise RuntimeError(
                    f"dashboard capture omitted capture_group_id for {camera}"
                )
            if isinstance(sim_step, bool) or not isinstance(
                sim_step, (int, np.integer)
            ):
                raise RuntimeError(
                    f"dashboard capture omitted simulator step for {camera}"
                )
            if not isinstance(frame_id, str) or not frame_id:
                raise RuntimeError(f"dashboard capture omitted frame id for {camera}")
            frames_bytes[camera] = image
            frame_ids[camera] = frame_id
            group_ids.add(group_id)
            simulator_steps.add(int(sim_step))
        if len(group_ids) != 1 or len(simulator_steps) != 1:
            raise RuntimeError(
                "dashboard capture cameras do not share one capture group and step"
            )
        capture_group_id = next(iter(group_ids))
        simulator_step = next(iter(simulator_steps))
        if simulator_step != int(self._env_steps):
            raise RuntimeError(
                "dashboard capture is not from the current simulator step"
            )
        return {
            "_frames_bytes": frames_bytes,
            "frame_ids": frame_ids,
            "capture_group_id": capture_group_id,
            "simulator_step": simulator_step,
        }

    def _dashboard_capture_group(self) -> dict[str, Any]:
        """Capture and serialize one fresh, atomic three-camera simulator view."""

        previous_group_ids: set[str] = set()
        for camera in ("head", "left_wrist", "right_wrist"):
            try:
                prior = self._frame_cache.latest(camera)
            except Exception:
                continue
            if isinstance(prior.capture_group_id, str):
                previous_group_ids.add(prior.capture_group_id)

        self._refresh_observation_without_step()
        capture = self._cached_capture_group()
        if capture["capture_group_id"] in previous_group_ids:
            raise RuntimeError(
                "dashboard capture refresh did not publish a fresh capture group"
            )
        observation = self._last_observation
        if not isinstance(observation, dict):
            raise RuntimeError(
                "dashboard capture refresh produced no recordable observation"
            )
        self._append_video(observation)
        return capture

    @staticmethod
    def _dashboard_requested_step(target: str, action: str) -> dict[str, Any]:
        if action == "observe":
            return {"camera_refresh": True}
        if target == "chassis":
            if action in {"forward", "backward"}:
                return {
                    "frame": "base",
                    "distance_m": BASE_TRANSLATION_STEP_M,
                    "direction": action,
                }
            if action in {"turn_left", "turn_right"}:
                return {
                    "frame": "base",
                    "angle_rad": BASE_ROTATION_STEP_RAD,
                    "direction": action,
                }
            return {
                "frame": "world",
                "torso_delta_z_m": (
                    TORSO_VERTICAL_STEP_M if action == "up" else -TORSO_VERTICAL_STEP_M
                ),
            }
        if action in {
            "forward",
            "backward",
            "turn_left",
            "turn_right",
            "up",
            "down",
        }:
            return {
                "frame": "base",
                "eef_translation_m": EEF_TRANSLATION_STEP_M,
                "direction": action,
            }
        if action in {"rotate_left", "rotate_right"}:
            return {
                "frame": "wrist_camera",
                "angle_rad": WRIST_ROTATION_STEP_RAD,
                "visual_direction": (
                    "counterclockwise" if action == "rotate_left" else "clockwise"
                ),
            }
        return {"opening": 1.0 if action == "open" else 0.0}

    @staticmethod
    def _dashboard_motion_capability_available(
        capabilities: dict[str, Any],
        *,
        target: str,
        action: str,
    ) -> bool:
        if target == "chassis":
            return bool(
                capabilities.get("torso_available")
                if action in {"up", "down"}
                else capabilities.get("base_available")
            )
        hand = "left" if target == "left_arm" else "right"
        if action in {"rotate_left", "rotate_right"}:
            return bool(
                dict(capabilities.get("wrist_rotation_available") or {}).get(
                    hand
                )
            )
        if action in {"open", "close"}:
            return bool(
                dict(capabilities.get("gripper_available") or {}).get(hand)
            )
        return bool(
            dict(capabilities.get("eef_available") or {}).get(hand)
        )

    def dashboard_prepare_manual_command(
        self,
        *,
        target: str,
        action: str,
        predecessor_plan_id: str | None = None,
        background: bool = False,
        planning_only_probe: bool = False,
    ) -> dict[str, Any]:
        """Prepare motion on the Env RPC FIFO."""

        method_started = time.monotonic()
        start_env_step = int(getattr(self, "_env_steps", -1))
        controller_switch_elapsed_s = 0.0
        controller_state_before = str(getattr(self, "_controller_state", "unavailable"))
        controller_switch_attempted = False
        controller_switch_changed = False
        request = validate_dashboard_prepare_request(
            target=target,
            action=action,
            predecessor_plan_id=predecessor_plan_id,
            background=background,
            planning_only_probe=planning_only_probe,
        )
        target = request["target"]
        action = request["action"]
        planning_only_probe = request.get("planning_only_probe") is True
        if self._official_success_latched or _raw_success(self._last_info):
            raise RuntimeError("raw task success is terminal")
        if action in {"open", "close"}:
            raise ValueError(
                "open and close are one-shot commands, not prepared motions"
            )
        if request["background"]:
            if not bool(getattr(self, "_dashboard_planning_admitted", False)):
                raise RuntimeError(
                    "background planning requires a foreground-admitted predecessor"
                )
        else:
            capabilities = self.dashboard_control_capabilities()
            if capabilities.get("motion_available") is not True:
                raise RuntimeError(
                    str(
                        capabilities.get("unavailable_reason")
                        or "manual motion unavailable"
                    )
                )
            capability_available = self._dashboard_motion_capability_available(
                capabilities,
                target=target,
                action=action,
            )
            if not capability_available and planning_only_probe:
                planner_report = (
                    dict(capabilities.get("planner") or {})
                    if isinstance(capabilities.get("planner"), dict)
                    else {}
                )
                if target == "chassis":
                    torso = planner_report.get("torso")
                    capability_available = bool(
                        isinstance(torso, dict)
                        and isinstance(torso.get("backend_report"), dict)
                        and torso["backend_report"].get("verified") is True
                    )
                else:
                    hand = "left" if target == "left_arm" else "right"
                    geometry = planner_report.get("wrist_geometry")
                    hand_geometry = (
                        geometry.get(hand)
                        if isinstance(geometry, dict)
                        else None
                    )
                    capability_available = bool(
                        isinstance(hand_geometry, dict)
                        and hand_geometry.get("verified") is True
                    )
            if not capability_available:
                raise RuntimeError(
                    f"{target}/{action} capability is unavailable or unsupported"
                )
            controller_switch_attempted = True
            controller_switch_started = time.monotonic()
            transition = self._switch_controller(_CONTROLLER_PLANNER)
            controller_switch_elapsed_s = max(
                0.0,
                time.monotonic() - controller_switch_started,
            )
            controller_switch_changed = transition.get("changed") is True
            self._dashboard_planning_admitted = True

        planner = self._require_planner()
        prepare = getattr(
            planner,
            (
                "prepare_predicted_dashboard_motion"
                if request["background"]
                else "prepare_dashboard_motion"
            ),
            None,
        )
        if not callable(prepare):
            raise RuntimeError("Dashboard prepared-motion planner is unavailable")
        prepare_started = time.monotonic()
        prepare_preflight_s = max(
            0.0,
            prepare_started - method_started - controller_switch_elapsed_s,
        )
        try:
            if request["background"]:
                result = prepare(
                    target,
                    action,
                    predecessor_plan_id=request["predecessor_plan_id"],
                )
            else:
                result = prepare(
                    target,
                    action,
                    predecessor_plan_id=request["predecessor_plan_id"],
                    background=False,
                )
        except CuroboPlanningError as exc:
            result = {
                "status": "failed",
                "target": target,
                "action": action,
                "predecessor_plan_id": request["predecessor_plan_id"],
                "background": request["background"],
                "primitive_success": False,
                "stop_reason": exc.stop_reason,
                "error": exc.error or str(exc),
            }
        prepare_finished = time.monotonic()
        planner_prepare_s = max(0.0, prepare_finished - prepare_started)
        if self._official_success_latched or _raw_success(self._last_info):
            raise RuntimeError(
                "raw task success became terminal during Dashboard planning"
            )
        finish_env_step = int(getattr(self, "_env_steps", -1))
        if planning_only_probe and finish_env_step != start_env_step:
            raise RuntimeError(
                "planning-only Dashboard probe advanced the simulator"
            )
        if not isinstance(result, dict):
            raise RuntimeError(
                "prepare_dashboard_motion returned a non-mapping result"
            )
        if result.get("status") == "failed":
            postcheck_finished = time.monotonic()
            response = {
                **_wire_safe(result),
                "source": "dashboard_prepare",
                "requested_step": self._dashboard_requested_step(target, action),
                "elapsed_s": planner_prepare_s,
                "planning_only_probe": planning_only_probe,
                "env_step_delta": finish_env_step - start_env_step,
                "zero_action_verified": finish_env_step == start_env_step,
                "latency_metrics": {
                    "schema_version": 1,
                    "clock": "time.monotonic",
                    "operation": "dashboard_prepare",
                    "phases_s": {
                        "prepare_preflight": prepare_preflight_s,
                        "controller_switch": controller_switch_elapsed_s,
                        "planner_prepare": planner_prepare_s,
                        "prepare_postcheck": max(
                            0.0, postcheck_finished - prepare_finished
                        ),
                    },
                    "controller_switch": {
                        "attempted": controller_switch_attempted,
                        "changed": controller_switch_changed,
                        "from_state": controller_state_before,
                        "to_state": str(
                            getattr(self, "_controller_state", "unavailable")
                        ),
                    },
                    "total_s": max(0.0, postcheck_finished - method_started),
                },
            }
            pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
            return response
        plan_id = validate_dashboard_plan_id(result.get("plan_id"))
        for field, expected in (
            ("target", target),
            ("action", action),
            ("predecessor_plan_id", request["predecessor_plan_id"]),
            ("background", request["background"]),
        ):
            if result.get(field) != expected:
                raise RuntimeError(
                    f"prepared plan returned mismatched {field}"
                )
        if result.get("status") != "prepared":
            raise RuntimeError("prepared plan omitted status='prepared'")

        legacy_elapsed_s = max(0.0, time.monotonic() - prepare_started)
        response = {
            **_wire_safe(result),
            "plan_id": plan_id,
            "source": "dashboard_prepare",
            "requested_step": self._dashboard_requested_step(target, action),
            "elapsed_s": legacy_elapsed_s,
            "planning_only_probe": planning_only_probe,
            "env_step_delta": finish_env_step - start_env_step,
            "zero_action_verified": bool(
                planning_only_probe and finish_env_step == start_env_step
            ),
        }
        postcheck_finished = time.monotonic()
        prepare_postcheck_s = max(
            0.0,
            postcheck_finished - prepare_finished,
        )
        response["latency_metrics"] = {
            "schema_version": 1,
            "clock": "time.monotonic",
            "operation": "dashboard_prepare",
            "phases_s": {
                "prepare_preflight": prepare_preflight_s,
                "controller_switch": controller_switch_elapsed_s,
                "planner_prepare": planner_prepare_s,
                "prepare_postcheck": prepare_postcheck_s,
            },
            "controller_switch": {
                "attempted": controller_switch_attempted,
                "changed": controller_switch_changed,
                "from_state": controller_state_before,
                "to_state": str(getattr(self, "_controller_state", "unavailable")),
            },
            "total_s": (
                prepare_preflight_s
                + controller_switch_elapsed_s
                + planner_prepare_s
                + prepare_postcheck_s
            ),
        }
        pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        return response

    def dashboard_execute_prepared_command(
        self,
        *,
        plan_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        """Execute one prepared motion request."""

        method_started = time.monotonic()
        plan_id = validate_dashboard_plan_id(plan_id)
        command_id = validate_dashboard_command_id(command_id)
        if self._official_success_latched or _raw_success(self._last_info):
            raise RuntimeError("raw task success is terminal")
        planner = self._require_planner()
        execute = getattr(planner, "execute_dashboard_motion", None)
        if not callable(execute):
            raise RuntimeError("Dashboard prepared-motion execution is unavailable")

        execute_started = time.monotonic()
        execute_preflight_s = max(0.0, execute_started - method_started)
        env_step_latency: dict[str, Any] = {
            "count": 0,
            "total_s": 0.0,
            "min_s": None,
            "max_s": None,
        }
        self._dashboard_env_step_latency = env_step_latency
        self._motion_in_flight = True
        try:
            result = execute(plan_id, command_id)
        finally:
            execute_finished = time.monotonic()
            self._dashboard_env_step_latency = None
            self._motion_in_flight = False
        planner_execute_s = max(0.0, execute_finished - execute_started)
        if not isinstance(result, dict):
            raise RuntimeError(
                "execute_dashboard_motion returned a non-mapping result"
            )
        result = self._planner_public_result(result)
        legacy_elapsed_s = max(0.0, time.monotonic() - execute_started)
        response = {
            **result,
            "source": "dashboard_execute",
            "plan_id": plan_id,
            "command_id": command_id,
            "elapsed_s": legacy_elapsed_s,
        }
        postcheck_finished = time.monotonic()
        execute_postcheck_s = max(
            0.0,
            postcheck_finished - execute_finished,
        )
        env_step_count = int(env_step_latency["count"])
        env_step_total_s = float(env_step_latency["total_s"])
        response["latency_metrics"] = {
            "schema_version": 1,
            "clock": "time.monotonic",
            "operation": "dashboard_execute",
            "phases_s": {
                "execute_preflight": execute_preflight_s,
                "planner_execute": planner_execute_s,
                "execute_postcheck": execute_postcheck_s,
            },
            "env_step_aggregate": {
                "boundary": "env._direct_process.step_env",
                "count": env_step_count,
                "total_s": env_step_total_s,
                "min_s": env_step_latency["min_s"],
                "max_s": env_step_latency["max_s"],
                "mean_s": env_step_total_s / env_step_count if env_step_count else None,
            },
            "total_s": execute_preflight_s + planner_execute_s + execute_postcheck_s,
        }
        pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        return response

    def dashboard_discard_prepared_command(
        self,
        *,
        plan_id: str,
    ) -> dict[str, Any]:
        """Discard one unneeded prepared plan without any simulator action."""

        plan_id = validate_dashboard_plan_id(plan_id)
        discard = getattr(
            self._require_planner(),
            "discard_dashboard_motion",
            None,
        )
        if not callable(discard):
            raise RuntimeError("Dashboard prepared-plan discard is unavailable")
        result = discard(plan_id)
        if not isinstance(result, dict):
            raise RuntimeError(
                "discard_dashboard_motion returned a non-mapping result"
            )
        result_plan_id = str(result.get("plan_id") or "").strip()
        if (
            result_plan_id != plan_id
            or result.get("discarded") is not True
            or result.get("status") != "discarded"
        ):
            raise RuntimeError(
                "discard_dashboard_motion returned an invalid discard receipt"
            )
        response = {
            **_wire_safe(result),
            "source": "dashboard_discard",
        }
        pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        return response

    def dashboard_capture_views(
        self,
        *,
        command_id: str,
    ) -> dict[str, Any]:
        """Capture one complete camera group while physical control is idle."""

        command_id = validate_dashboard_command_id(command_id)
        if self._motion_in_flight:
            raise RuntimeError("Dashboard capture requires an idle controller")
        capabilities = self.dashboard_control_capabilities()
        if capabilities.get("observe_available") is not True:
            raise RuntimeError(
                str(
                    capabilities.get("unavailable_reason")
                    or "camera refresh unavailable"
                )
            )
        capture = self._dashboard_capture_group()
        response = {
            **capture,
            "source": "dashboard_capture",
            "command_id": command_id,
        }
        pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        return response

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
    ) -> dict[str, Any]:
        """Run one legacy one-shot command; only observe captures cameras."""

        command = validate_dashboard_manual_command(
            target=target,
            action=action,
            camera=camera,
        )
        target = command["target"]
        action = command["action"]
        camera = command["camera"]
        started = time.monotonic()
        capabilities = self.dashboard_control_capabilities()
        capture: dict[str, Any] = {}

        if action == "observe":
            if capabilities.get("observe_available") is not True:
                raise RuntimeError(
                    str(
                        capabilities.get("unavailable_reason")
                        or "camera refresh unavailable"
                    )
                )
            capture = self._dashboard_capture_group()
            result = self._require_planner().observe(camera)
            if not isinstance(result, dict):
                raise RuntimeError("observe returned a non-mapping planner result")
            result = self._planner_public_result(result)
            # The Dashboard consumes the explicit physical-camera map below;
            # retain no selected-camera legacy aliases in the new contract.
            result.pop("_image_bytes", None)
            result.pop("_depth_image_bytes", None)
            primitive = "observe"
        else:
            if capabilities.get("motion_available") is not True:
                raise RuntimeError(
                    str(
                        capabilities.get("unavailable_reason")
                        or "manual motion unavailable"
                    )
                )
            hand = "left" if target == "left_arm" else "right"
            capability_ok = False
            if target == "chassis":
                capability_ok = bool(
                    capabilities.get("torso_available")
                    if action in {"up", "down"}
                    else capabilities.get("base_available")
                )
            elif action in {"rotate_left", "rotate_right"}:
                capability_ok = bool(
                    dict(capabilities.get("wrist_rotation_available") or {}).get(hand)
                )
            elif action in {"open", "close"}:
                capability_ok = bool(
                    dict(capabilities.get("gripper_available") or {}).get(hand)
                )
            else:
                capability_ok = bool(
                    dict(capabilities.get("eef_available") or {}).get(hand)
                )
            if not capability_ok:
                raise RuntimeError(
                    f"{target}/{action} capability is unavailable or unverified"
                )

            self._switch_controller(_CONTROLLER_PLANNER)
            planner = self._require_planner()
            self._motion_in_flight = True
            try:
                if target == "chassis" and action in {
                    "forward",
                    "backward",
                    "turn_left",
                    "turn_right",
                }:
                    result = planner.jog_base(action)
                    primitive = "navigate_to"
                elif target == "chassis":
                    result = planner.jog_torso(action)
                    primitive = "jog_torso"
                elif action in {
                    "forward",
                    "backward",
                    "turn_left",
                    "turn_right",
                    "up",
                    "down",
                }:
                    result = planner.jog_eef(hand, action)
                    primitive = "move_to"
                elif action in {"rotate_left", "rotate_right"}:
                    result = planner.jog_wrist(hand, action)
                    primitive = "rotate_wrist"
                else:
                    result = planner.set_gripper(
                        hand,
                        1.0 if action == "open" else 0.0,
                    )
                    primitive = "set_gripper"
            finally:
                self._motion_in_flight = False
            if not isinstance(result, dict):
                raise RuntimeError(f"{primitive} returned a non-mapping planner result")
            result = self._planner_public_result(result)

        response = {
            **result,
            **capture,
            "source": "dashboard_manual",
            "target": target,
            "action": action,
            "camera": camera,
            "primitive": primitive,
            "primitive_detail": (
                "relative jog" if primitive == "navigate_to" else None
            ),
            "requested_step": self._dashboard_requested_step(target, action),
            "elapsed_s": max(0.0, time.monotonic() - started),
        }
        metrics = response.get("metrics")
        metrics = dict(metrics) if isinstance(metrics, dict) else {}
        response["partial_motion"] = bool(
            response.get("partial_motion") or metrics.get("partial_motion")
        )
        for field in ("actual_target", "fallback_offset", "requested_delta"):
            if field not in response and field in metrics:
                response[field] = metrics[field]
        # Refresh the cached admission state from facts after controller transfer
        # and execution; the HTTP controller does not poll the simulator.
        if response.get("task_success") is True:
            # Do not query planner/backend capabilities after the success step,
            # even within this RPC. Publish a locally-derived terminal snapshot.
            terminal_capabilities = dict(capabilities)
            terminal_capabilities.update(
                {
                    "motion_available": False,
                    "observe_available": False,
                    "unavailable_reason": "official_success_latched",
                    "base_available": False,
                    "eef_available": {"left": False, "right": False},
                    "torso_available": False,
                    "wrist_rotation_available": {
                        "left": False,
                        "right": False,
                    },
                    "gripper_available": {"left": False, "right": False},
                }
            )
            response["control_capabilities"] = terminal_capabilities
        else:
            response["control_capabilities"] = capabilities
        pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        return response

    def _planner_result_with_accounting(self, result: dict[str, Any]) -> dict[str, Any]:
        public = self._strip_flow_advice(result)
        public["total_env_steps"] = int(self._env_steps)
        public["global_env_steps"] = int(self._env_steps)
        public["run_nonce"] = str(getattr(self, "_run_nonce", "unbound"))
        public["attempt_index"] = int(getattr(self, "_attempt_index", 1))
        public["attempt_nonce"] = str(getattr(self, "_attempt_nonce", "unbound"))
        return public

    @classmethod
    def _strip_flow_advice(cls, value: Any) -> Any:
        """Remove ordering and stage advice from Agent-visible tool results."""

        if isinstance(value, dict):
            return {
                key: cls._strip_flow_advice(item)
                for key, item in value.items()
                if key != "suggested_next_tool" and "stage" not in key.lower()
            }
        if isinstance(value, list):
            return [cls._strip_flow_advice(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._strip_flow_advice(item) for item in value)
        return value

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

    @staticmethod
    def _render_only_for_hand_geometry() -> None:
        """Advance Kit rendering once without stepping physics.

        Calling ``Environment.render`` is insufficient in headless runs because
        it may return before invoking Kit when no external debug sensor exists.
        OmniGibson's own reset synchronization calls the simulator directly.
        """

        try:
            import omnigibson as og
        except Exception as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "OmniGibson render-only synchronization is unavailable"
            ) from exc
        simulator = getattr(og, "sim", None)
        render = getattr(simulator, "render", None)
        if not callable(render):
            raise RuntimeError(
                "OmniGibson simulator render-only synchronization is unavailable"
            )
        render()

    def _refresh_observation_without_step(
        self,
        *,
        synchronize_hand_geometry: bool = False,
    ) -> None:
        """Capture current synchronized sensors without advancing simulation time.

        Planning can legitimately take longer than the RGB-D cache TTL while no
        controller waypoint is executed.  A later ``observe`` must therefore
        obtain a new capture (and new frame ids) at the same simulator step,
        rather than renewing or returning the expired capture.
        """

        omni_env = self._env.omnigibson_env
        render_sync_iterations = 0
        if synchronize_hand_geometry:
            for _ in range(_HAND_GEOMETRY_SYNC_RENDER_ITERATIONS):
                self._render_only_for_hand_geometry()
                render_sync_iterations += 1
        else:
            # ``get_obs()`` can return the previous render target after a
            # controller-only Env step.  A render-only Kit update refreshes all
            # three camera sensors without advancing physics or sending another
            # Env action, so the terminal capture and MP4 reflect the state that
            # just executed rather than a newly labelled copy of stale pixels.
            self._render_only_for_hand_geometry()
            render_sync_iterations = 1
        raw_observation, _sensor_info = omni_env.get_obs()
        observation = _single_observation(self._env._wrap_obs([raw_observation]))
        self._last_observation = observation
        self._record_rgbd_frames(
            [raw_observation],
            observation,
            strict=synchronize_hand_geometry,
            synchronize_hand_geometry=synchronize_hand_geometry,
            render_sync_iterations=render_sync_iterations,
        )

    def _resolve_camera_role(self, camera: str) -> str:
        requested = str(camera)
        if requested not in {"head", "left_wrist", "right_wrist"}:
            raise ValueError("camera must be head, left_wrist, or right_wrist")
        return canonical_camera(requested)

    def _decorate_public_observation(
        self,
        *,
        payload: dict[str, Any],
        requested_camera: str,
        resolved_camera: str,
        current_frame: Any,
    ) -> dict[str, Any]:
        payload["camera"] = requested_camera
        payload.pop("visual_review", None)
        capture_group = payload.get("capture_group")
        if isinstance(capture_group, dict):
            capture_group.pop("cameras", None)
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            metrics["camera"] = requested_camera
        payload["camera_metadata"] = {
            "camera": requested_camera,
            "frame_id": current_frame.frame_id,
            "capture_group_id": current_frame.capture_group_id,
            "capture_env_step": int(current_frame.step_index),
            "current_env_step": int(self._env_steps),
            "fresh": bool(int(current_frame.step_index) == int(self._env_steps)),
            "frame_age_s": float(
                max(0.0, time.monotonic() - current_frame.timestamp_s)
            ),
            "frame_ttl_s": float(self._frame_cache.ttl_s),
            "width": int(current_frame.intrinsics.width),
            "height": int(current_frame.intrinsics.height),
            "intrinsics": {
                "fx": float(current_frame.intrinsics.fx),
                "fy": float(current_frame.intrinsics.fy),
                "cx": float(current_frame.intrinsics.cx),
                "cy": float(current_frame.intrinsics.cy),
            },
            "camera_to_world": np.asarray(
                current_frame.camera_to_world, dtype=np.float64
            ).tolist(),
        }
        return payload

    def _review_public_observation(
        self,
        *,
        requested_camera: str,
        frame_review: dict[str, Any],
    ) -> dict[str, Any]:
        """Review an existing public frame without refreshing any sensor."""

        surface_policy = self._active_task_spec().surface_review_policy
        if surface_policy is None:
            raise ValueError(
                f"{self._active_task_spec().task_name} does not define frame review"
            )
        if not isinstance(frame_review, dict):
            raise ValueError("frame_review must be an object")
        if set(frame_review) != {"frame_id", "assessment"}:
            raise ValueError("frame_review requires exactly frame_id and assessment")
        frame_id = str(frame_review["frame_id"]).strip()
        if not frame_id:
            raise ValueError("frame_review.frame_id must be a non-empty string")
        assessment = str(frame_review["assessment"])
        if assessment not in {
            surface_policy.target_assessment,
            surface_policy.opposite_assessment,
            surface_policy.indeterminate_assessment,
        }:
            raise ValueError("frame_review.assessment is invalid")
        resolved_camera = self._resolve_camera_role(requested_camera)
        capture_receipt = getattr(
            self, "_latest_unconsumed_public_capture_receipt", None
        )
        if (
            not self._attempt_receipt_is_current(capture_receipt)
            or capture_receipt.get("kind") != "public_observe_capture"
            or capture_receipt.get("requested_camera") != requested_camera
            or capture_receipt.get("resolved_camera") != resolved_camera
            or capture_receipt.get("frame_id") != frame_id
            or int(capture_receipt.get("env_step", -1)) != int(self._env_steps)
            or int(capture_receipt.get("capture_sequence", -1))
            != int(getattr(self, "_public_capture_sequence", 0))
        ):
            raise RuntimeError(
                "frame_review must consume the immediately preceding, "
                "same-camera public capture"
            )
        frame = self._frame_cache.get_current(resolved_camera, frame_id)
        if frame.frame_id not in getattr(self, "_public_observed_frame_ids", set()):
            raise RuntimeError(
                "frame_review must reference a frame returned by public observe"
            )
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError("frame_review must reference the current env step")
        if not isinstance(frame.capture_group_id, str) or not frame.capture_group_id:
            raise RuntimeError(
                "frame_review must reference a synchronized public capture"
            )
        if capture_receipt.get("capture_group_id") != frame.capture_group_id:
            raise RuntimeError("frame_review capture-group lineage changed")
        payload = self._require_planner().observe(resolved_camera)
        if str(payload.get("frame_id", "")) != frame.frame_id:
            raise RuntimeError("frame_review cache head changed during review")
        image_bytes = payload.get("_image_bytes")
        if not isinstance(image_bytes, bytes) or hashlib.sha256(
            image_bytes
        ).hexdigest() != capture_receipt.get("rgb_sha256"):
            raise RuntimeError("frame_review RGB digest does not match its capture")
        self._latest_unconsumed_public_capture_receipt = None
        payload = self._decorate_public_observation(
            payload=payload,
            requested_camera=requested_camera,
            resolved_camera=resolved_camera,
            current_frame=frame,
        )
        payload["frame_review"] = self._public_visual_receipt(
            self._record_frame_review_cycle(
                requested_camera=requested_camera,
                resolved_camera=resolved_camera,
                frame=frame,
                capture_receipt=capture_receipt,
                assessment=assessment,
            )
        )
        return self._planner_result_with_accounting(payload)

    def _probe_public_observation(
        self,
        *,
        requested_camera: str,
        depth_probe: dict[str, Any],
    ) -> dict[str, Any]:
        """Measure one LLM-selected pixel in the latest public RGB-D frame.

        This is a read-only follow-up to ``observe``: it neither refreshes
        sensors nor advances physics, registers a new public capture, consumes
        the Radio surface-review receipt, or creates motion authority.
        """

        if not isinstance(depth_probe, dict):
            raise ValueError("depth_probe must be an object")
        required = {
            "frame_id",
            "u",
            "v",
            "depth_window_px",
            "assessment",
        }
        if set(depth_probe) != required:
            raise ValueError(
                "depth_probe requires exactly frame_id, u, v, "
                "depth_window_px, and assessment"
            )
        frame_id = depth_probe["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("depth_probe.frame_id must be a non-empty string")
        frame_id = frame_id.strip()
        coordinates: dict[str, int] = {}
        for field in ("u", "v"):
            value = depth_probe[field]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"depth_probe.{field} must be an integer pixel coordinate"
                )
            coordinates[field] = int(value)
        depth_window_px = depth_probe["depth_window_px"]
        if (
            isinstance(depth_window_px, bool)
            or not isinstance(depth_window_px, (int, np.integer))
            or not 1 <= int(depth_window_px) <= 31
        ):
            raise ValueError("depth_probe.depth_window_px must be an integer in [1,31]")
        depth_window_px = int(depth_window_px)
        if depth_probe["assessment"] != "target_point_visually_confirmed":
            raise ValueError(
                "depth_probe.assessment must be target_point_visually_confirmed"
            )

        resolved_camera = self._resolve_camera_role(requested_camera)
        capture_receipt = getattr(self, "_latest_public_observation_lineage", None)
        if (
            not self._attempt_receipt_is_current(capture_receipt)
            or capture_receipt.get("kind") != "public_observe_rgbd_capture"
            or capture_receipt.get("requested_camera") != requested_camera
            or capture_receipt.get("resolved_camera") != resolved_camera
            or capture_receipt.get("frame_id") != frame_id
            or int(capture_receipt.get("env_step", -1)) != int(self._env_steps)
            or capture_receipt.get("depth_source_modality") != "depth_linear"
            or capture_receipt.get("depth_measurement") != "distance_to_image_plane"
            or capture_receipt.get("depth_unit") != "m"
            or not isinstance(capture_receipt.get("depth_sha256"), str)
            or not isinstance(capture_receipt.get("geometry_sha256"), str)
        ):
            raise RuntimeError(
                "depth_probe must reference the immediately preceding, "
                "same-camera public RGB-D capture"
            )

        # get_current validates latest-frame identity, current capture group,
        # and TTL without refreshing the cache.
        frame = self._frame_cache.get_current(resolved_camera, frame_id)
        if frame.frame_id not in getattr(self, "_public_observed_frame_ids", set()):
            raise RuntimeError(
                "depth_probe must reference a frame returned by public observe"
            )
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError("depth_probe must reference the current env step")
        if not isinstance(frame.capture_group_id, str) or not frame.capture_group_id:
            raise RuntimeError(
                "depth_probe must reference a synchronized public capture"
            )
        if capture_receipt.get("capture_group_id") != frame.capture_group_id:
            raise RuntimeError("depth_probe capture-group lineage changed")
        if capture_receipt.get("depth_sha256") != self._metric_depth_sha256(frame):
            raise RuntimeError("depth_probe metric-depth lineage changed")
        geometry_sha256 = self._frame_geometry_sha256(frame)
        if capture_receipt.get("geometry_sha256") != geometry_sha256:
            raise RuntimeError("depth_probe frame-geometry lineage changed")

        stats = robust_depth_sample(
            frame,
            u=coordinates["u"],
            v=coordinates["v"],
            window_px=depth_window_px,
        )
        optical_axis_depth_m = float(stats["depth_m"])
        camera_point = camera_point_from_pixel(
            frame.intrinsics,
            u=coordinates["u"],
            v=coordinates["v"],
            depth_m=optical_axis_depth_m,
        )
        camera_range_m = float(np.linalg.norm(camera_point))
        if not np.isfinite(camera_range_m) or camera_range_m <= 0.0:
            raise CameraGeometryError("depth_probe produced an invalid camera range")
        effective_camera_point = frame.correction_profile.apply_camera_point(
            camera_point
        )
        if (
            np.asarray(effective_camera_point).shape != (3,)
            or not np.isfinite(effective_camera_point).all()
        ):
            raise CameraGeometryError(
                "depth_probe produced an invalid effective camera point"
            )

        hand_geometry: dict[str, Any] = {
            "available": False,
            "reason": "requires_resolved_wrist_camera",
            "geometry_sha256": geometry_sha256,
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "env_step": int(self._env_steps),
            "source": "frame_bound_live_r1pro_link_transforms",
            "target_point_camera_frame": "effective_usd_camera",
            "camera_axes": "+X right,+Y up,-Z forward",
            "distance_computation_frame": "world",
            "guidance_only": True,
            "semantic_target_verified": False,
        }
        distance_report: dict[str, Any] | None = None
        if resolved_camera in {"left_wrist", "right_wrist"}:
            resolved_hand = resolved_camera.removesuffix("_wrist")
            sync_certificate = frame.capture_metadata.get(
                "hand_geometry_sync_certificate"
            )
            camera_pose_lineage = frame.capture_metadata.get("camera_pose_lineage")
            selected_pose_lineage = (
                camera_pose_lineage.get(resolved_camera)
                if isinstance(camera_pose_lineage, dict)
                else None
            )
            selected_hand_certificate = (
                sync_certificate.get("hands", {}).get(resolved_hand)
                if isinstance(sync_certificate, dict)
                else None
            )
            if not self._hand_geometry_sync_certificate_is_valid(
                sync_certificate,
                hand=resolved_hand,
                env_step=int(frame.step_index),
            ) or not (
                isinstance(selected_pose_lineage, dict)
                and isinstance(selected_hand_certificate, dict)
                and selected_pose_lineage.get("render_bound") is True
                and selected_pose_lineage.get("source")
                == selected_hand_certificate.get("camera_pose_source")
                and int(selected_pose_lineage.get("env_step", -1))
                == int(frame.step_index)
                and int(selected_pose_lineage.get("render_sync_iterations", -1))
                == int(sync_certificate["render_sync_iterations"])
            ):
                raise RuntimeError(
                    "depth_probe wrist hand-geometry sync certificate is unavailable"
                )
            references = frame.capture_metadata.get("r1pro_hand_reference_transforms")
            if (
                not isinstance(references, dict)
                or references.get("available") is not True
                or int(references.get("env_step", -1)) != int(frame.step_index)
                or references.get("source") != "capture_time_live_r1pro_link_transforms"
                or not isinstance(references.get("hands"), dict)
                or resolved_hand not in references["hands"]
            ):
                raise RuntimeError(
                    "depth_probe wrist hand-reference geometry is unavailable"
                )
            distance_report = frame_bound_hand_distance_report(
                frame,
                raw_target_point_camera_xyz_m=camera_point,
                hand_reference_transforms_world=references["hands"][resolved_hand],
            )
            if not np.allclose(
                np.asarray(distance_report["target_point_camera_xyz_m"]),
                np.asarray(effective_camera_point),
                atol=1e-12,
                rtol=0.0,
            ):
                raise RuntimeError(
                    "depth_probe camera-point geometry changed during measurement"
                )
            hand_geometry = {
                **hand_geometry,
                "available": True,
                "reason": None,
                "resolved_hand": resolved_hand,
                "sync_certificate": {
                    "synchronized": True,
                    "render_sync_iterations": int(
                        sync_certificate["render_sync_iterations"]
                    ),
                    "translation_tolerance_m": float(
                        sync_certificate["translation_tolerance_m"]
                    ),
                    "rotation_tolerance_deg": float(
                        sync_certificate["rotation_tolerance_deg"]
                    ),
                    "finger_joint_tolerance_m": float(
                        sync_certificate["finger_joint_tolerance_m"]
                    ),
                    "selected_hand_passed": True,
                    "camera_pose_source": sync_certificate["hands"][resolved_hand][
                        "camera_pose_source"
                    ],
                    "camera_pose_render_bound": True,
                },
                "target_to_finger_roots_individual_m": list(
                    distance_report["target_to_finger_roots_individual_m"]
                ),
            }
        relative_dispersion = float(stats["mad_m"]) / max(optical_axis_depth_m, 1e-6)
        confidence = max(
            0.0,
            min(
                1.0,
                float(stats["cluster_ratio"])
                * (1.0 - min(1.0, relative_dispersion / 0.04)),
            ),
        )
        quality = {
            "mad_m": float(stats["mad_m"]),
            "valid_ratio": float(stats["valid_ratio"]),
            "cluster_ratio": float(stats["cluster_ratio"]),
            "sample_count": int(stats["sample_count"]),
            "valid_count": int(stats["valid_count"]),
            "cluster_count": int(stats["cluster_count"]),
            "confidence": float(confidence),
        }
        probe_receipt = self._seal_attempt_receipt(
            {
                "kind": "public_observe_depth_probe",
                "capture_receipt_sha256": capture_receipt["receipt_sha256"],
                "requested_camera": requested_camera,
                "resolved_camera": resolved_camera,
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "env_step": int(self._env_steps),
                "pixel": dict(coordinates),
                "depth_window_px": depth_window_px,
                "source": "llm_selected_pixel",
                "measurement": ("first_visible_surface_at_llm_selected_pixel"),
                "optical_axis_depth_m": optical_axis_depth_m,
                "camera_range_m": camera_range_m,
                "target_point_camera_xyz_m": np.asarray(
                    effective_camera_point, dtype=np.float64
                ).tolist(),
                **(
                    {
                        key: float(distance_report[key])
                        for key in (
                            "target_to_palm_m",
                            "target_to_grip_point_m",
                            "target_to_finger_roots_m",
                        )
                    }
                    if distance_report is not None
                    else {}
                ),
                "hand_geometry": hand_geometry,
                "quality": quality,
                "semantic_target_verified": False,
                "motion_authorization": False,
            }
        )

        # Re-encode the already cached frame so the LLM receives the exact RGB-D
        # pair it selected.  PlannerExecutor.observe only reads FrameCache.
        payload = self._require_planner().observe(resolved_camera)
        if str(payload.get("frame_id", "")) != frame.frame_id:
            raise RuntimeError("depth_probe cache head changed during sampling")
        image_bytes = payload.get("_image_bytes")
        if not isinstance(image_bytes, bytes) or hashlib.sha256(
            image_bytes
        ).hexdigest() != capture_receipt.get("rgb_sha256"):
            raise RuntimeError("depth_probe RGB digest does not match its capture")
        payload = self._decorate_public_observation(
            payload=payload,
            requested_camera=requested_camera,
            resolved_camera=resolved_camera,
            current_frame=frame,
        )
        payload["depth_probe"] = {
            "source": "llm_selected_pixel",
            "measurement": "first_visible_surface_at_llm_selected_pixel",
            "pixel": dict(coordinates),
            "depth_window_px": depth_window_px,
            "optical_axis_depth_m": optical_axis_depth_m,
            "camera_range_m": camera_range_m,
            "target_point_camera_xyz_m": np.asarray(
                effective_camera_point, dtype=np.float64
            ).tolist(),
            **(
                {
                    key: float(distance_report[key])
                    for key in (
                        "target_to_palm_m",
                        "target_to_grip_point_m",
                        "target_to_finger_roots_m",
                    )
                }
                if distance_report is not None
                else {}
            ),
            "hand_geometry": hand_geometry,
            "quality": quality,
            "semantic_target_verified": False,
            "motion_authorization": False,
            "lineage": {
                "camera": requested_camera,
                "frame_id": frame.frame_id,
                "capture_group_id": frame.capture_group_id,
                "env_step": int(self._env_steps),
                "receipt_sha256": probe_receipt["receipt_sha256"],
            },
        }
        return self._planner_result_with_accounting(payload)

    def observe(
        self,
        camera: str,
        frame_review: dict[str, Any] | None = None,
        depth_probe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        requested_camera = str(camera)
        if frame_review is not None and depth_probe is not None:
            raise ValueError("frame_review and depth_probe are mutually exclusive")
        if frame_review is not None:
            return self._review_public_observation(
                requested_camera=requested_camera,
                frame_review=frame_review,
            )
        if depth_probe is not None:
            return self._probe_public_observation(
                requested_camera=requested_camera,
                depth_probe=depth_probe,
            )
        resolved_camera = self._resolve_camera_role(requested_camera)
        if resolved_camera in {"left_wrist", "right_wrist"}:
            # A wrist hand-distance chain is admitted only from a fresh
            # render-synchronized capture with a fixed-extrinsic certificate.
            # Action-loop frames are intentionally never upgraded in place.
            self._refresh_observation_without_step(synchronize_hand_geometry=True)
        else:
            try:
                frame = self._frame_cache.latest(resolved_camera)
                self._frame_cache.get_current(resolved_camera, frame.frame_id)
                if (
                    int(frame.step_index) != int(self._env_steps)
                    or time.monotonic() - frame.timestamp_s > 5.0
                ):
                    self._refresh_observation_without_step()
            except CameraGeometryError:
                self._refresh_observation_without_step()
        refreshed = self._frame_cache.latest(resolved_camera)
        self._frame_cache.get_current(resolved_camera, refreshed.frame_id)
        payload = self._require_planner().observe(resolved_camera)
        image_bytes = payload.get("_image_bytes")
        if not isinstance(image_bytes, bytes):
            raise RuntimeError("observe payload did not contain PNG bytes")
        payload = self._persist_live_observation(payload)
        current_frame = self._frame_cache.get_current(
            resolved_camera, str(payload.get("frame_id"))
        )
        observed_frame_ids = getattr(self, "_public_observed_frame_ids", None)
        if not isinstance(observed_frame_ids, set):
            observed_frame_ids = set()
            self._public_observed_frame_ids = observed_frame_ids
        observed_frame_ids.add(current_frame.frame_id)
        if requested_camera == "head" and resolved_camera == canonical_camera("head"):
            self._latest_public_head_frame_id = current_frame.frame_id
        self._register_public_observation_lineage(
            requested_camera=requested_camera,
            resolved_camera=resolved_camera,
            frame=current_frame,
            image_bytes=image_bytes,
        )
        capture_receipt = self._register_public_capture(
            requested_camera=requested_camera,
            resolved_camera=resolved_camera,
            frame=current_frame,
            image_bytes=image_bytes,
        )
        payload = self._decorate_public_observation(
            payload=payload,
            requested_camera=requested_camera,
            resolved_camera=resolved_camera,
            current_frame=current_frame,
        )
        payload["capture_receipt"] = self._public_visual_receipt(capture_receipt)
        return self._planner_result_with_accounting(payload)

    def _pixel_to_world_raw(
        self,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
    ) -> dict[str, Any]:
        result = self._require_planner().pixel_to_world(
            camera=camera,
            frame_id=frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
            output_frame="world",
        )
        return self._planner_result_with_accounting(result)

    def pixel_to_world(
        self,
        camera: str,
        frame_id: str,
        u: Any,
        v: Any,
        depth_window_px: int = 7,
    ) -> dict[str, Any]:
        requested_camera = str(camera)
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        for field, value in (("u", u), ("v", v)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field} must be an integer pixel coordinate")
        if (
            isinstance(depth_window_px, bool)
            or not isinstance(depth_window_px, (int, np.integer))
            or int(depth_window_px) < 1
        ):
            raise ValueError("depth_window_px must be a positive integer")
        resolved_camera = self._resolve_camera_role(requested_camera)
        frame = self._frame_cache.get_current(resolved_camera, frame_id.strip())
        if frame.frame_id not in getattr(self, "_public_observed_frame_ids", set()):
            raise RuntimeError("pixel_to_world requires a public observe frame")
        if int(frame.step_index) != int(self._env_steps):
            raise RuntimeError("pixel_to_world frame is not from the current env step")
        projected = self._pixel_to_world_raw(
            camera=resolved_camera,
            frame_id=frame.frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
        )
        projected_metrics = projected.get("metrics")
        if isinstance(projected_metrics, dict):
            projected_metrics["camera"] = requested_camera
        if projected.get("primitive_success") is not True:
            return {**projected, "camera": requested_camera}
        diagnostics = projected.get("diagnostics")
        metrics = projected.get("metrics")
        if not isinstance(diagnostics, dict) or not isinstance(metrics, dict):
            raise RuntimeError("projection omitted diagnostics or metrics")
        point = np.asarray(diagnostics.get("xyz"), dtype=np.float64).reshape(3)
        normal = np.asarray(
            diagnostics.get("surface_normal"), dtype=np.float64
        ).reshape(3)
        norm = float(np.linalg.norm(normal))
        if (
            not np.isfinite(point).all()
            or not np.isfinite(normal).all()
            or norm <= 1e-9
        ):
            raise RuntimeError("projection point or surface normal is invalid")
        normal /= norm
        camera_origin = np.asarray(frame.camera_to_world[:3, 3], dtype=np.float64)
        if float(np.dot(normal, camera_origin - point)) < 0.0:
            normal = -normal
        receipt = {
            "run_nonce": self._run_nonce,
            "attempt_nonce": self._attempt_nonce,
            "env_step": int(self._env_steps),
            "camera": requested_camera,
            "resolved_camera": resolved_camera,
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "world_point": point.tolist(),
            "camera_facing_normal": normal.tolist(),
            "confidence": float(metrics.get("confidence", 0.0)),
        }
        receipt["projection_id"] = (
            "projection_"
            + hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()[:24]
        )
        self._projection_receipts[receipt["projection_id"]] = receipt
        return {
            **projected,
            "camera": requested_camera,
            "frame_id": frame.frame_id,
            "capture_group_id": frame.capture_group_id,
            "world_point": receipt["world_point"],
            "camera_facing_normal": receipt["camera_facing_normal"],
            "confidence": receipt["confidence"],
            "projection_id": receipt["projection_id"],
        }

    def _planner_public_result(self, result: dict[str, Any]) -> dict[str, Any]:
        public = self._planner_result_with_accounting(result)
        public["task_success"] = bool(self._official_success_latched)
        public["official_success_source"] = 'info["done"]["success"]'
        if self._official_success_latched:
            if not isinstance(self._official_success_receipt, dict):
                raise RuntimeError(
                    "official success is latched without its immutable receipt"
                )
            public["official_success_receipt"] = _wire_safe(
                self._official_success_receipt
            )
        return public

    def finalize_paused_runtime(self, vla_status: dict[str, Any]) -> dict[str, Any]:
        """Confirm external VLA disable and transfer ownership to the planner."""

        if self._official_success_latched or _raw_success(self._last_info):
            receipt = self._latch_official_success(self._last_info)
            self._clear_active_vla_invocation_state()
            return {
                "controller_state": self._controller_state,
                "vla_actions_enabled": False,
                "lifecycle_finalized": True,
                "task_success": True,
                "official_success_source": 'info["done"]["success"]',
                "official_success_receipt": _wire_safe(receipt),
            }
        try:
            if (
                not isinstance(vla_status, dict)
                or vla_status.get("actions_enabled") is not False
            ):
                raise RuntimeError("VLA action disable confirmation is missing")
            health = vla_status.get("healthz")
            if (
                not isinstance(health, dict)
                or health.get("actions_enabled") is not False
            ):
                raise RuntimeError("VLA health did not confirm actions_enabled=false")
            transition = self._switch_controller(_CONTROLLER_PLANNER)
        except Exception:
            self._clear_active_vla_invocation_state()
            raise
        self._vla_actions_enabled = False
        self._clear_active_vla_invocation_state()
        return {
            "controller_state": self._controller_state,
            "controller_transition": transition,
            "vla_actions_enabled": False,
            "lifecycle_finalized": True,
            "task_success": False,
            "official_success_source": 'info["done"]["success"]',
            "env_pid": os.getpid(),
            "vla_pid": health.get("pid"),
            "vla_endpoint": vla_status.get("endpoint"),
        }

    def _analytic_public_result(
        self,
        result: dict[str, Any],
        *,
        primitive: str,
        requested_hand: str,
        resolved_hand: str,
        hand_selection_source: str,
        visual_hand_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach only request-selection context to the CuRobo result."""

        del primitive
        public = self._planner_public_result(result)
        public.update(
            {
                "requested_hand": requested_hand,
                "resolved_hand": resolved_hand,
                "hand_selection_source": hand_selection_source,
                "visual_hand_evidence": visual_hand_evidence,
            }
        )
        return public

    def _motion_target(
        self,
        *,
        hand: str,
        target: dict[str, Any],
    ) -> tuple[np.ndarray, str | None]:
        if not isinstance(target, dict):
            raise ValueError("target must be an object")
        target = dict(target)
        allowed = {"projection_id", "standoff_m", "delta_xyz", "frame"}
        unknown = set(target).difference(allowed)
        if unknown:
            raise ValueError(f"target contains unsupported fields: {sorted(unknown)}")
        projection_id = target.get("projection_id")
        delta_xyz = target.get("delta_xyz")
        if (projection_id is None) == (delta_xyz is None):
            raise ValueError(
                "target requires exactly one of projection_id or delta_xyz"
            )
        current = self._require_planner().backend.get_eef_pose(hand)
        if current is None:
            raise RuntimeError("current EEF pose is unavailable")
        current_position = np.asarray(current[0], dtype=np.float64).reshape(3)
        if projection_id is not None:
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("target.projection_id must be a non-empty string")
            if "frame" in target or "delta_xyz" in target:
                raise ValueError("projection targets cannot contain delta_xyz or frame")
            receipt = self._projection_receipts.get(str(projection_id))
            if not self._projection_receipt_is_fresh(receipt):
                raise RuntimeError("fresh projection receipt is required")
            point = np.asarray(receipt["world_point"], dtype=np.float64).reshape(3)
            normal = np.asarray(
                receipt["camera_facing_normal"], dtype=np.float64
            ).reshape(3)
            standoff = float(target.get("standoff_m", 0.0))
            if not np.isfinite(standoff) or standoff < 0.0:
                raise ValueError("target.standoff_m must be finite and non-negative")
            resolved = point + normal * standoff
            receipt_id = str(projection_id)
        else:
            if "standoff_m" in target or "projection_id" in target:
                raise ValueError(
                    "relative targets cannot contain projection_id or standoff_m"
                )
            frame = str(target.get("frame", ""))
            if frame not in {"world", "eef"}:
                raise ValueError("target.frame must be 'world' or 'eef'")
            delta = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
            if delta.shape != (3,) or not np.isfinite(delta).all():
                raise ValueError("target.delta_xyz must contain three finite values")
            if frame == "eef":
                delta = _quat_rotate_vector_xyzw(current[1], delta)
            resolved = current_position + delta
            receipt_id = None
        return resolved, receipt_id

    def navigate_to(
        self,
        *,
        projection_id: str | None = None,
        navigation_visual_check: Any = None,
        relative_motion: Any = None,
        standoff_m: float | None = None,
    ) -> dict[str, Any]:
        """Resolve one base goal and delegate its execution to CuRobo."""

        projection_mode = relative_motion is None
        navigation_visual_evidence: dict[str, Any] | None = None
        target_xyz: np.ndarray | None = None
        normalized_motion: dict[str, Any] | None = None
        if projection_mode:
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("projection_id must be a non-empty string")
            projection_id = projection_id.strip()
            standoff = float(0.85 if standoff_m is None else standoff_m)
            if not np.isfinite(standoff) or not 0.45 <= standoff <= 1.50:
                raise ValueError("standoff_m must be finite and within [0.45,1.50]")
            projection_receipt = self._projection_receipts.get(projection_id)
            if not self._projection_receipt_is_fresh(projection_receipt):
                raise RuntimeError("fresh projection receipt is required")
            navigation_visual_evidence = self._navigation_visual_authorization(
                projection_receipt=projection_receipt,
                navigation_visual_check=navigation_visual_check,
            )
            target_xyz = np.asarray(
                projection_receipt.get("world_point"), dtype=np.float64
            ).reshape(-1)
            if target_xyz.shape != (3,) or not np.isfinite(target_xyz).all():
                raise RuntimeError("navigation projection point is invalid")
        else:
            if any(
                value is not None
                for value in (
                    projection_id,
                    navigation_visual_check,
                    standoff_m,
                )
            ):
                raise ValueError(
                    "relative_motion is mutually exclusive with projection "
                    "navigation arguments"
                )
            normalized_motion = validate_relative_navigation_motion(relative_motion)

        self._switch_controller(_CONTROLLER_PLANNER)
        planner = self._require_planner()
        if projection_mode:
            result = planner.navigate_to(
                target_xyz=target_xyz,
                standoff_m=standoff,
            )
        else:
            result = planner.navigate_to(relative_motion=normalized_motion)
        public = self._planner_public_result(result)
        if projection_mode:
            public.update(
                {
                    "requested_projection_id": projection_id,
                    "navigation_visual_evidence": self._public_visual_receipt(
                        navigation_visual_evidence
                    ),
                }
            )
        else:
            public["requested_relative_motion"] = dict(normalized_motion or {})
        public["_finish"] = bool(public["task_success"])
        return public

    def move_to(
        self,
        *,
        hand: str,
        target: dict[str, Any],
        visual_hand_check: Any,
    ) -> dict[str, Any]:
        requested_hand = str(hand)
        hand, selection_source, visual_hand_evidence = self._authorize_analytic_hand(
            requested_hand, visual_hand_check
        )
        target, receipt_id = self._motion_target(
            hand=hand,
            target=target,
        )
        self._switch_controller(_CONTROLLER_PLANNER)
        result = self._require_planner().move_to(
            hand=hand,
            target_xyz=target,
        )
        del receipt_id
        return self._analytic_public_result(
            result,
            primitive="move_to",
            requested_hand=requested_hand,
            resolved_hand=hand,
            hand_selection_source=selection_source,
            visual_hand_evidence=visual_hand_evidence,
        )

    def rotate_wrist(
        self,
        *,
        hand: str,
        relative_axis_angle: Any,
        frame: str = "eef",
        visual_hand_check: Any,
    ) -> dict[str, Any]:
        requested_hand = str(hand)
        hand, selection_source, visual_hand_evidence = self._authorize_analytic_hand(
            requested_hand, visual_hand_check
        )
        if frame not in {"world", "eef"}:
            raise ValueError("frame must be world or eef")
        orientation = np.asarray(relative_axis_angle, dtype=np.float64).reshape(-1)
        if orientation.shape != (4,) or not np.isfinite(orientation).all():
            raise ValueError("relative_axis_angle must contain four finite values")
        if (
            abs(float(orientation[3])) > 1e-12
            and float(np.linalg.norm(orientation[:3])) <= 1e-12
        ):
            raise ValueError("relative axis must be nonzero for a nonzero angle")
        self._switch_controller(_CONTROLLER_PLANNER)
        result = self._require_planner().rotate_wrist(
            hand=hand,
            target_quat_xyzw=None,
            relative_axis_angle=orientation.tolist(),
            frame=frame,
        )
        return self._analytic_public_result(
            result,
            primitive="rotate_wrist",
            requested_hand=requested_hand,
            resolved_hand=hand,
            hand_selection_source=selection_source,
            visual_hand_evidence=visual_hand_evidence,
        )

    def _set_gripper(
        self,
        *,
        hand: str,
        opening: float,
        visual_hand_check: Any,
        release_visual_check: Any = None,
    ) -> dict[str, Any]:
        requested_hand = str(hand)
        hand, selection_source, visual_hand_evidence = self._authorize_analytic_hand(
            requested_hand, visual_hand_check
        )
        del release_visual_check
        self._switch_controller(_CONTROLLER_PLANNER)
        result = self._require_planner()._gripper_command(
            hand,
            opening=float(opening),
        )
        return self._analytic_public_result(
            result,
            primitive="close" if float(opening) < 0.5 else "open",
            requested_hand=requested_hand,
            resolved_hand=hand,
            hand_selection_source=selection_source,
            visual_hand_evidence=visual_hand_evidence,
        )

    def close(
        self,
        *,
        hand: str,
        visual_hand_check: Any,
    ) -> dict[str, Any]:
        return self._set_gripper(
            hand=hand,
            opening=0.0,
            visual_hand_check=visual_hand_check,
        )

    def open(
        self,
        *,
        hand: str,
        visual_hand_check: Any,
        release_visual_check: Any = None,
    ) -> dict[str, Any]:
        return self._set_gripper(
            hand=hand,
            opening=1.0,
            visual_hand_check=visual_hand_check,
            release_visual_check=release_visual_check,
        )

    def press(
        self,
        *,
        hand: str,
        projection_id: str,
        travel_m: float,
        visual_hand_check: Any,
    ) -> dict[str, Any]:
        requested_hand = str(hand)
        hand, selection_source, visual_hand_evidence = self._authorize_analytic_hand(
            requested_hand, visual_hand_check
        )
        if not isinstance(projection_id, str) or not projection_id.strip():
            raise ValueError("projection_id must be a non-empty string")
        travel = float(travel_m)
        if not np.isfinite(travel) or travel <= 0.0:
            raise ValueError("travel_m must be finite and positive")
        receipt = self._projection_receipts.get(str(projection_id))
        if not self._projection_receipt_is_fresh(receipt):
            raise RuntimeError("fresh projection receipt is required")
        self._switch_controller(_CONTROLLER_PLANNER)
        point = np.asarray(receipt["world_point"], dtype=np.float64).reshape(3)
        outward = np.asarray(receipt["camera_facing_normal"], dtype=np.float64).reshape(
            3
        )
        result = self._require_planner().press(
            hand=hand,
            target_xyz=point,
            press_direction=-outward,
            travel_m=travel,
        )
        return self._analytic_public_result(
            result,
            primitive="press",
            requested_hand=requested_hand,
            resolved_hand=hand,
            hand_selection_source=selection_source,
            visual_hand_evidence=visual_hand_evidence,
        )

    def shutdown(self) -> None:
        # The process owns exactly one attempt. OmniGibson / Kit teardown has
        # repeatedly segfaulted inside Replicator plugin destruction after all
        # task artifacts were already complete. Seal Python-owned artifacts
        # here; main() closes the RPC server, fsyncs receipts, and uses
        # os._exit(0), letting the OS release the GPU context without invoking
        # the broken native destructor chain.
        self._finalize_video_segment()


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
        self._predicted_planner = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rpent-curobo-predicted",
        )
        ready = getattr(
            getattr(env, "_planner", None),
            "threaded_predicted_planning_ready",
            None,
        )
        env._threaded_predicted_planning_dispatcher_ready = callable(ready)

    def submit(self, method: str, args: tuple, kwargs: dict) -> Any:
        if (
            method == "env.dashboard_prepare_manual_command"
            and kwargs.get("background") is True
        ):
            return self.submit_dashboard_prepare(method, args, kwargs)
        future: Future = Future()
        self._calls.put((method, args, kwargs, future))
        return future.result()

    def submit_dashboard_prepare(
        self,
        method: str,
        args: tuple,
        kwargs: dict,
    ) -> Any:
        """Run only pure predicted-start CuRobo planning off the env FIFO."""

        if method != "env.dashboard_prepare_manual_command":
            raise ValueError(
                f"unknown concurrent BEHAVIOR env RPC method: {method!r}"
            )
        if kwargs.get("background") is not True:
            return self.submit(method, args, kwargs)
        ready = getattr(
            getattr(self._env, "_planner", None),
            "threaded_predicted_planning_ready",
            None,
        )
        if (
            not bool(
                getattr(
                    self._env,
                    "_threaded_predicted_planning_dispatcher_ready",
                    False,
                )
            )
            or not callable(ready)
            or ready() is not True
        ):
            raise RuntimeError(
                "pure predicted CuRobo planning worker is not ready"
            )

        def plan() -> Any:
            self._env._assert_rpc_lifecycle(
                "dashboard_prepare_manual_command"
            )
            if ready() is not True:
                raise RuntimeError(
                    "pure predicted CuRobo planning readiness changed"
                )
            return self._env.dashboard_prepare_manual_command(*args, **kwargs)

        return self._predicted_planner.submit(plan).result()

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method.startswith("env."):
            env_method = method.removeprefix("env.")
            if env_method not in _ENV_RPC_METHODS:
                raise ValueError(f"unknown BEHAVIOR env RPC method: {method!r}")
            self._env._assert_rpc_lifecycle(env_method)
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

    def close(self) -> None:
        self._env._threaded_predicted_planning_dispatcher_ready = False
        self._predicted_planner.shutdown(wait=True, cancel_futures=True)


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
    parser.add_argument("--public-seed", type=int, required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument(
        "--expected-run-nonce",
        default=os.environ.get("RPENT_BEHAVIOR_EXPECTED_RUN_NONCE"),
    )
    parser.add_argument(
        "--controller-mode",
        choices=("hybrid", "pi0_nav_pick_only"),
        default="hybrid",
    )
    parser.add_argument("--max-episode-steps", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path")
    parser.add_argument("--transport-host", default="127.0.0.1")
    parser.add_argument("--transport-port", type=int, default=0)
    return parser.parse_args()


def _flush_shutdown_artifacts(output_dir: Path) -> None:
    """Durably flush the sealed attempt before bypassing native destructors."""

    receipt = {
        "schema_version": 1,
        "status": "sealed",
        "exit_strategy": "controlled_fast_exit_without_native_destructor",
        "pid": os.getpid(),
        "video_path": str(output_dir / "episode.mp4"),
        "action_trace_path": str(output_dir / "behavior_action_trace.jsonl"),
        "sealed_at_unix_s": time.time(),
    }
    _write_json_atomic(output_dir / "env_shutdown_receipt.json", receipt)
    for path in (
        output_dir / "episode.mp4",
        output_dir / "episode_meta.json",
        output_dir / "video_meta.json",
        output_dir / "behavior_action_trace.jsonl",
        output_dir / "env_shutdown_receipt.json",
    ):
        if not path.is_file():
            continue
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sys.stdout.flush()
    sys.stderr.flush()


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
        "public_seed": args.public_seed,
        "attempt_index": args.attempt_index,
        "expected_run_nonce": args.expected_run_nonce,
        "controller_mode": args.controller_mode,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
    }
    env = BehaviorEnvFacade(
        cfg=_load_env_config(args),
        meta=meta,
        output_dir=output_dir,
    )
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
            dispatcher.close()
        finally:
            try:
                env.shutdown()
            finally:
                server.shutdown()
                server.server_close()
    _flush_shutdown_artifacts(output_dir)


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # OmniGibson / Kit native module destructors have repeatedly
            # segfaulted after a fully sealed graceful shutdown. This process
            # owns one attempt, so bypass Python/native teardown only after
            # main() has sealed Python-owned artifacts and closed the socket
            # server, without entering the broken native destructor chain.
            os._exit(exit_code)
