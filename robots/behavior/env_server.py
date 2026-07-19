"""OmniGibson/R1Pro process for the BEHAVIOR RPent runtime."""
from __future__ import annotations

import argparse
import json
import os
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
from robots.behavior.schemas import validate_action_chunk
from rpent.rpc_driver.socket import SocketRpcServer
from rpent.utils.config import get_repo_root, get_rlinf_repo_path
from rpent.utils.logging import get_logger

logger = get_logger("behavior_env_server")
RPENT_ROOT = get_repo_root()
RLINF_ROOT = get_rlinf_repo_path() or (RPENT_ROOT.parent / "RLinf_agentic_push")
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))


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


def _payload_matrix(payload: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
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
            return CameraIntrinsics.from_matrix(
                payload[name],
                width=int(rgb_shape[1]),
                height=int(rgb_shape[0]),
            )
        except Exception:
            continue
    return None


def _sensor_intrinsics(sensor: Any, *, rgb_shape: tuple[int, ...]) -> CameraIntrinsics | None:
    for name in ("intrinsic_matrix", "camera_intrinsics", "K"):
        try:
            value = getattr(sensor, name)
        except Exception:
            continue
        try:
            return CameraIntrinsics.from_matrix(
                value,
                width=int(rgb_shape[1]),
                height=int(rgb_shape[0]),
            )
        except Exception:
            pass
    for name in ("get_intrinsic_matrix", "get_camera_intrinsics"):
        fn = getattr(sensor, name, None)
        if fn is None:
            continue
        try:
            return CameraIntrinsics.from_matrix(
                fn(),
                width=int(rgb_shape[1]),
                height=int(rgb_shape[0]),
            )
        except Exception:
            pass
    return None


def _sensor_camera_to_world(sensor: Any) -> np.ndarray | None:
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
            "BEHAVIOR bootstrap scene template not found: " f"{template_path}"
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
        raise FileNotFoundError(f"BEHAVIOR instance directory not found: {instance_dir}")

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


class BehaviorEnvFacade:
    """Single-env raw-info facade with streaming 15 FPS video."""

    def __init__(self, *, cfg: Any, meta: dict[str, Any], output_dir: Path) -> None:
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
        self._done = False
        self._env_steps = 0
        self._video_path = output_dir / "episode.mp4"
        self._video_writer = None
        self._video_frames = 0
        self._video_error: str | None = None
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
        view = _payload_matrix(payload, ("view_matrix", "view_transform", "world_to_camera"))
        if view is not None:
            return np.linalg.inv(view.T)
        if sensor is not None:
            sensor_matrix = _sensor_camera_to_world(sensor)
            if sensor_matrix is not None:
                return sensor_matrix
        raise CameraGeometryError(f"camera pose unavailable for {camera}")

    def _record_rgbd_frames(self, raw_observations: Any, observation: dict[str, Any]) -> None:
        del observation
        raw = _first_env_value(raw_observations)
        if raw is None:
            return
        payloads: dict[str, dict[str, Any]] = {}
        for path, payload in _iter_sensor_payloads(raw):
            camera = _payload_camera_name(path)
            if camera is not None:
                payloads[camera] = payload
        for camera, payload in payloads.items():
            rgb = _payload_rgb(payload)
            depth = _payload_depth(payload)
            if rgb is None or depth is None:
                continue
            sensor = self._sensor_for_camera(camera)
            try:
                rgb_array = np.asarray(_numpy_tree(rgb))
                intrinsics = (
                    _payload_intrinsics(payload, rgb_shape=rgb_array.shape)
                    or (
                        _sensor_intrinsics(sensor, rgb_shape=rgb_array.shape)
                        if sensor is not None
                        else None
                    )
                )
                if intrinsics is None:
                    raise CameraGeometryError(
                        f"verified camera intrinsics unavailable for {camera}"
                    )
                self._frame_cache.add(
                    camera=camera,
                    rgb=rgb_array,
                    depth_m=depth,
                    intrinsics=intrinsics,
                    camera_to_world=self._camera_to_world(
                        camera=camera,
                        payload=payload,
                        sensor=sensor,
                    ),
                    step_index=self._env_steps,
                )
            except Exception:
                logger.exception("failed to cache BEHAVIOR RGB-D frame for %s", camera)

    def _append_video(self, observation: dict[str, Any]) -> None:
        if self._video_error is not None:
            return
        try:
            import imageio.v2 as imageio
            import numpy as np

            if self._video_writer is None:
                self._video_path.parent.mkdir(parents=True, exist_ok=True)
                self._video_writer = imageio.get_writer(self._video_path, fps=15)
            frame = np.asarray(observation["main_images"], dtype=np.uint8)
            self._video_writer.append_data(frame)
            self._video_frames += 1
        except Exception as exc:
            self._video_error = f"{type(exc).__name__}: {exc}"
            logger.exception("failed to append BEHAVIOR video frame")

    def reset(self) -> tuple[dict[str, Any], Any]:
        started_at = time.monotonic()
        logger.info("BEHAVIOR reset started on thread %s", threading.get_ident())
        raw_observations, infos = self._env.env_reset()
        observation = _single_observation(self._env._wrap_obs(raw_observations))
        self._done = False
        self._env_steps = 0
        self._last_observation = observation
        self._last_info = _numpy_tree(infos[0])
        self._record_rgbd_frames(raw_observations, observation)
        self._append_video(observation)
        logger.info(
            "BEHAVIOR reset completed in %.1fs on thread %s",
            time.monotonic() - started_at,
            threading.get_ident(),
        )
        return observation, _numpy_tree(infos[0])

    def chunk_step(self, actions: Any) -> tuple[Any, Any, bool, bool, Any]:
        import torch

        if self._done:
            raise RuntimeError("env.chunk_step called after episode stop")
        action_array = validate_action_chunk(actions)
        action_tensor = torch.as_tensor(action_array, dtype=torch.float32).reshape(
            1, action_array.shape[0], action_array.shape[1]
        )
        final_observation = None
        final_reward = None
        official_info: Any = {}
        terminated = False
        truncated = False
        executed_steps = 0
        for step_index in range(action_tensor.shape[1]):
            step_obs, step_reward, step_term, step_trunc, step_infos = (
                self._env._direct_process.step_env(
                    action_tensor[:, step_index],
                    need_obs=True,
                )
            )
            self._env_steps += 1
            executed_steps += 1
            step_info = step_infos[0]
            official_info = step_info
            final_reward = step_reward[0]
            terminated = terminated or _scalar_bool(step_term) or _raw_done(step_info)
            truncated = truncated or _scalar_bool(step_trunc)
            final_observation = _single_observation(self._env._wrap_obs(step_obs))
            self._last_observation = final_observation
            self._last_info = _numpy_tree(step_info)
            self._record_rgbd_frames(step_obs, final_observation)
            if self._env_steps % 4 == 0:
                self._append_video(final_observation)
            if _raw_success(step_info) or terminated or truncated:
                break

        if final_observation is None:
            raise RuntimeError("BEHAVIOR action chunk executed zero steps")
        task_success = _raw_success(official_info)
        self._done = task_success or terminated or truncated
        returned_info = _numpy_tree(official_info)
        if not isinstance(returned_info, dict):
            returned_info = {"raw": returned_info}
        returned_info["_rpent"] = {"executed_steps": executed_steps}
        return (
            final_observation,
            _numpy_tree(final_reward),
            terminated,
            truncated,
            returned_info,
        )

    def get_env_meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def observe(self, camera: str) -> dict[str, Any]:
        return self._planner.observe(camera)

    def pixel_to_world(
        self,
        camera: str,
        frame_id: str,
        u: Any = None,
        v: Any = None,
        depth_window_px: int = 7,
        output_frame: str = "world",
    ) -> dict[str, Any]:
        return self._planner.pixel_to_world(
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
        return self._planner.navigate_to(
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
        return self._planner.move_to(
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
        return self._planner.pick(
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
        return self._planner.rotate_wrist(
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
        return self._planner.press(
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
        return self._planner.release(
            hand=hand,
            opening=opening,
            retreat_vector=retreat_vector,
            retreat_m=retreat_m,
            timeout_s=timeout_s,
        )

    def close(self) -> None:
        try:
            if self._video_writer is not None:
                self._video_writer.close()
                self._video_writer = None
        finally:
            self._env.close()
        video_meta = {
            "path": str(self._video_path),
            "fps": 15,
            "sample_every_env_steps": 4,
            "frames": self._video_frames,
            "error": self._video_error,
        }
        (self._video_path.parent / "video_meta.json").write_text(
            json.dumps(video_meta, indent=2), encoding="utf-8"
        )


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
            return getattr(self._env, method.removeprefix("env."))(*args, **kwargs)
        if method in {
            "observe",
            "pixel_to_world",
            "navigate_to",
            "move_to",
            "pick",
            "rotate_wrist",
            "press",
            "release",
        }:
            return getattr(self._env, method)(*args, **kwargs)
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
