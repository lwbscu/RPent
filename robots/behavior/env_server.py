"""OmniGibson/R1Pro process for the BEHAVIOR RPent runtime."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import threading
import time
from concurrent.futures import Future
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
    canonical_camera,
    load_camera_correction_profiles,
)
from robots.behavior.planner_executor import PlannerExecutor
from robots.behavior.schemas import (
    CONTROL_MODES,
    FULL_TASK_VLA_MODE,
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
_ENV_RPC_METHODS_BY_MODE = {
    FULL_TASK_VLA_MODE: frozenset({"chunk_step"}),
    PI0_PICK_VLA_MODE: frozenset({"pi0_chunk_step"}),
    PLANNER_TOOLS_MODE: frozenset(
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
    ),
}

RESTORE_RENDER_SETTLE_FRAMES = 3


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
            number = float(
                np.asarray(_numpy_tree(getter(usd_attribute))).reshape(())
            )
            return number if np.isfinite(number) else None
        except Exception:
            return None

    focal_length = scalar("focal_length", "focalLength")
    horizontal_aperture = scalar("horizontal_aperture", "horizontalAperture")
    vertical_aperture = scalar("vertical_aperture", "verticalAperture")
    horizontal_offset = scalar(
        "horizontal_aperture_offset", "horizontalApertureOffset"
    )
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
    """Keep VLA controllers intact and use OG's cuRobo BASE controller in planner mode."""
    if control_mode != PLANNER_TOOLS_MODE:
        return
    robots = list(cfg.omni_config.robots)
    if len(robots) != 1 or str(robots[0].type) != "R1Pro":
        raise ValueError("planner_tools requires exactly one R1Pro robot")
    base = robots[0].controller_config.base
    if str(base.name) != "HolonomicBaseJointController":
        raise ValueError(
            "planner_tools requires OmniGibson HolonomicBaseJointController"
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
        self._done = False
        self._env_steps = 0
        self._video_path = output_dir / "episode.mp4"
        self._video_writer = None
        self._video_frames = 0
        self._video_error: str | None = None
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
            if self._control_mode == PLANNER_TOOLS_MODE
            else None
        )
        self._last_observation: dict[str, Any] | None = None
        self._last_info: Any = None
        self._gripper_latch = {"left": 1.0, "right": 1.0}
        self._restored_state: dict[str, Any] | None = None

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
        if self._control_mode != PLANNER_TOOLS_MODE or self._planner is None:
            raise RuntimeError(
                "planner primitives are unavailable outside planner_tools mode"
            )
        return self._planner

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
        except Exception:
            # Atomicity is intentional: never publish one camera from a newer
            # simulator step beside two cameras from an older step.
            logger.exception(
                "failed to cache atomic BEHAVIOR RGB-D capture group at sim step %s",
                self._env_steps,
            )

    def _append_video(self, observation: dict[str, Any]) -> None:
        if self._video_error is not None:
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
            if self._control_mode in (PLANNER_TOOLS_MODE, PI0_PICK_VLA_MODE):
                wrists = np.asarray(observation["wrist_images"], dtype=np.uint8)
                if wrists.ndim != 4 or wrists.shape[0] != 2:
                    raise RuntimeError(
                        "planner video requires synchronized left/right wrist RGB"
                    )
                height, width = head.shape[:2]
                self._video_source_shapes["left_wrist"] = list(wrists[0].shape)
                self._video_source_shapes["right_wrist"] = list(wrists[1].shape)
                left_wrist = _resize_video_tile(
                    wrists[0], height=height, width=width
                )
                right_wrist = _resize_video_tile(
                    wrists[1], height=height, width=width
                )
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
                if self._control_mode in (PLANNER_TOOLS_MODE, PI0_PICK_VLA_MODE)
                else "head"
            ),
        }
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        (self._video_path.parent / "video_meta.json").write_text(
            json.dumps(video_meta, indent=2), encoding="utf-8"
        )

    def start_video_segment(self, path: str | Path) -> None:
        """Rotate acceptance video without exposing a new planner RPC method."""

        self._finalize_video_segment()
        self._video_path = Path(path).expanduser().resolve()
        self._video_writer = None
        self._video_frames = 0
        self._video_error = None
        self._video_source_shapes = {}
        if self._last_observation is not None:
            self._append_video(self._last_observation)

    def dump_simulator_state(self, *, serialized: bool = True) -> Any:
        """Capture the complete in-process simulator state for test isolation."""

        import omnigibson as og

        return og.sim.dump_state(serialized=bool(serialized))

    def restore_simulator_state(
        self, state: Any, *, serialized: bool = True
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
        # OmniGibson documents that one physics update is required after load
        # for spatial object states to become current.
        og.sim.step_physics()
        self._done = False
        self._last_info = None
        self._gripper_latch = {"left": 1.0, "right": 1.0}
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

    def reset(self) -> tuple[dict[str, Any], Any]:
        started_at = time.monotonic()
        logger.info("BEHAVIOR reset started on thread %s", threading.get_ident())
        raw_observations, infos = self._env.env_reset()
        observation = _single_observation(self._env._wrap_obs(raw_observations))
        self._done = False
        self._env_steps = 0
        self._last_observation = observation
        self._last_info = _numpy_tree(infos[0])
        restore_path_value = os.environ.get("RPENT_BEHAVIOR_RESTORE_STATE")
        if restore_path_value:
            if self._control_mode not in (PLANNER_TOOLS_MODE, PI0_PICK_VLA_MODE):
                raise RuntimeError(
                    "RPENT_BEHAVIOR_RESTORE_STATE is acceptance-only and "
                    "forbidden outside planner_tools or pi0_pick_vla"
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
                "manifest_path": str(
                    Path(f"{restore_path}.manifest.json")
                ),
                "manifest_schema_version": restore_manifest["schema_version"],
            }
        else:
            self._record_rgbd_frames(raw_observations, observation)
            self._append_video(observation)
            self._restored_state = None
        if self._control_mode == PLANNER_TOOLS_MODE:
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
        return observation, _numpy_tree(infos[0])

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
            raise ValueError(
                "gripper_closed_threshold must be finite and non-negative"
            )
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
    ) -> tuple[Any, Any, bool, bool, Any]:
        import torch

        if self._done:
            raise RuntimeError("env.chunk_step called after episode stop")
        action_array = validate_action_chunk(actions)
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
        for step_index in range(action_tensor.shape[1]):
            is_last_action = step_index == action_tensor.shape[1] - 1
            observation_interval = (
                4 if bool(observe_final) else self._planner_video_interval_steps
            )
            need_observation = (
                local_gripper_monitor is not None
                or (bool(observe_final) and is_last_action)
                or (self._env_steps + 1) % observation_interval == 0
            )
            step_obs, step_reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    action_tensor[:, step_index],
                    need_obs=need_observation,
                )
            )
            self._env_steps += 1
            executed_steps += 1
            step_info = step_infos[0]
            official_info = step_info
            final_reward = step_reward[0]
            terminated = terminated or _scalar_bool(step_term) or _raw_done(step_info)
            truncated = truncated or _scalar_bool(step_trunc)
            self._last_info = _numpy_tree(step_info)
            if need_observation:
                if step_obs is None:
                    raise RuntimeError("BEHAVIOR requested observation but received None")
                final_observation = _single_observation(self._env._wrap_obs(step_obs))
                self._last_observation = final_observation
                self._record_rgbd_frames(step_obs, final_observation)
                if self._env_steps % observation_interval == 0:
                    self._append_video(final_observation)
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
            if _raw_success(step_info) or terminated or truncated:
                break

        if final_observation is None:
            raise RuntimeError("BEHAVIOR action chunk executed zero steps")
        task_success = _raw_success(official_info)
        self._done = task_success or terminated or truncated
        returned_info = _wire_safe(official_info)
        if not isinstance(returned_info, dict):
            returned_info = {"raw": returned_info}
        returned_info["_rpent"] = {"executed_steps": executed_steps}
        if monitor_result is not None:
            returned_info["_rpent"]["local_gripper_monitor"] = monitor_result
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

    def _validated_selected_gripper_opening(
        self,
        *,
        observation: dict[str, Any],
        hand: str,
    ) -> float:
        """Return physical opening after matching same-step public proprio."""

        robot = self._robot()
        if robot is None:
            raise RuntimeError("R1Pro robot unavailable for Pi0 gripper monitoring")
        control_indices = getattr(robot, "gripper_control_idx", None)
        if not isinstance(control_indices, dict) or hand not in control_indices:
            raise RuntimeError(
                f"R1Pro {hand} gripper control indices are unavailable"
            )
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
            raise RuntimeError(
                f"R1Pro {hand} physical gripper opening is not finite"
            )

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

    def observe(self, camera: str) -> dict[str, Any]:
        camera = canonical_camera(camera)
        try:
            frame = self._frame_cache.latest(camera)
            self._frame_cache.get_current(camera, frame.frame_id)
            # Planner warmup can consume nearly the entire cache TTL without
            # advancing simulation.  Do not hand a VLM a frame that is valid
            # at observe() time but likely to expire before pixel_to_world().
            if time.monotonic() - frame.timestamp_s > 5.0:
                self._refresh_observation_without_step()
        except CameraGeometryError:
            self._refresh_observation_without_step()
            refreshed = self._frame_cache.latest(camera)
            self._frame_cache.get_current(camera, refreshed.frame_id)
        return self._require_planner().observe(camera)

    def pixel_to_world(
        self,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        return self._require_planner().pixel_to_world(
            camera=camera,
            frame_id=frame_id,
            u=u,
            v=v,
            depth_window_px=depth_window_px,
            output_frame=output_frame,
        )

    def navigate_to(
        self,
        hand: str,
        target_xyz: Any,
        frame: str = "world",
        standoff_m: float = 0.85,
        timeout_s: float = 90.0,
    ) -> dict[str, Any]:
        return self._require_planner().navigate_to(
            hand=hand,
            target_xyz=target_xyz,
            frame=frame,
            standoff_m=standoff_m,
            timeout_s=timeout_s,
        )

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
        return self._require_planner().move_to(
            hand=hand,
            target_xyz=target_xyz,
            frame=frame,
            target_quat_xyzw=target_quat_xyzw,
            plan_only=plan_only,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            timeout_s=timeout_s,
        )

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
        return self._require_planner().pick(
            hand=hand,
            target_xyz=target_xyz,
            approach_vector=approach_vector,
            grasp_quat_xyzw=grasp_quat_xyzw,
            pregrasp_offset_m=pregrasp_offset_m,
            lift_m=lift_m,
            timeout_s=timeout_s,
        )

    def rotate_wrist(
        self,
        hand: str,
        target_quat_xyzw: Any | None = None,
        relative_axis_angle: Any | None = None,
        frame: str = "world",
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        return self._require_planner().rotate_wrist(
            hand=hand,
            target_quat_xyzw=target_quat_xyzw,
            relative_axis_angle=relative_axis_angle,
            frame=frame,
            timeout_s=timeout_s,
        )

    def press(
        self,
        hand: str,
        target_xyz: Any,
        press_direction: Any | None = None,
        approach_distance_m: float = 0.04,
        press_depth_m: float = 0.012,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        return self._require_planner().press(
            hand=hand,
            target_xyz=target_xyz,
            press_direction=press_direction,
            approach_distance_m=approach_distance_m,
            press_depth_m=press_depth_m,
            timeout_s=timeout_s,
        )

    def release(
        self,
        hand: str,
        opening: float = 1.0,
        retreat_vector: Any | None = None,
        retreat_m: float = 0.03,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return self._require_planner().release(
            hand=hand,
            opening=opening,
            retreat_vector=retreat_vector,
            retreat_m=retreat_m,
            timeout_s=timeout_s,
        )

    def close(self) -> None:
        try:
            self._finalize_video_segment()
        finally:
            self._env.close()


_INITIAL_PPID = os.getppid()


def _start_parent_watchdog(
    server: SocketRpcServer,
    shutdown_event: threading.Event,
) -> None:
    def watch() -> None:
        while not shutdown_event.wait(2.0):
            ppid = os.getppid()
            if ppid != _INITIAL_PPID or ppid == 1:
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
    _start_parent_watchdog(server, shutdown_event)
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
