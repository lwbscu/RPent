# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPent facade for the RLinf-native real YAM runtime."""

from __future__ import annotations

import sys
import threading
import time
import uuid
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from robots.yam.cameras import YamRgbdCameraRig
from robots.yam.contracts import MODEL_SPEC, YAM_CAMERA_NAMES, validate_actions
from robots.yam.geometry import (
    ARM_JOINT_INDICES,
    YamGeometry,
    apply_previous_command_slew,
    enforce_hard_limits,
    joint_limits_from_config,
)
from robots.yam.hardware_ownership import HardwareLease
from robots.yam.operator_control import read_receipt
from rpent.utils.config import get_rlinf_repo_path


class YamAgentEnv:
    """Gym-style YAM facade for RPent.

    The constructor opens no hardware. The first ``reset()``, ``observe()``, or
    mutating command starts cameras first, then connects the single RLinf
    follower writer. Reset means "begin a new RPent episode from the current
    manually prepared scene"; it does not fold, park, home, or otherwise move
    the robot.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        runtime: Any | None = None,
        cameras: Any | None = None,
        kinematics: Any | None = None,
    ) -> None:
        self.config = dict(config)
        self.task_name = str(
            self.config.get("task_name", self.config.get("task_language", ""))
        )
        self.task_language = str(
            self.config.get(
                "task_language",
                self.config.get("task_description", self.task_name),
            )
        )
        self.step_lim = int(self.config.get("max_episode_steps", 1000))
        self.control_hz = float(MODEL_SPEC.control_hz)
        self.max_joint_delta_per_step = self.config.get(
            "max_joint_delta_per_step", 0.05
        )
        if self.max_joint_delta_per_step is not None:
            self.max_joint_delta_per_step = float(self.max_joint_delta_per_step)
        self.lower, self.upper = joint_limits_from_config(self.config)
        self.max_tracking_error_rad = self.config.get("max_tracking_error_rad")
        if self.max_tracking_error_rad is not None:
            self.max_tracking_error_rad = float(self.max_tracking_error_rad)
            if (
                not np.isfinite(self.max_tracking_error_rad)
                or self.max_tracking_error_rad <= 0
            ):
                raise ValueError("max_tracking_error_rad must be finite and positive")
        self.geometry = YamGeometry(self.config, kinematics=kinematics)
        self._runtime = runtime
        self._owns_runtime = runtime is None
        self._hardware_lease: HardwareLease | None = None
        self._cameras = cameras
        self._owns_cameras = cameras is None
        self._started = False
        self._startup_failed = False
        self._closed = False
        self._lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._next_tick_s: float | None = None
        self._previous_command: np.ndarray | None = None
        self._take_action_cnt = 0
        self._actual_seed = int(self.config.get("seed", 0))
        self._episode_id = str(self.config.get("episode_id", uuid.uuid4().hex))
        self._operator_success: bool | None = None
        self._operator_success_source: str | None = None
        self._operator_ready_receipt: dict[str, Any] | None = None
        self._operator_success_receipt: dict[str, Any] | None = None
        self._terminal_event: str | None = None
        self._last_stop_hold_s: float | None = None
        self._consumed_receipt_ids: set[str] = set()
        self.operator_receipt_path = self.config.get("operator_receipt_path")
        self.last_obs: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Consume an operator ready receipt and begin a new episode.

        This is bookkeeping only: it never homes, folds, parks, or otherwise
        moves the robot. The ready receipt must be written by the local operator
        for the currently advertised pending ``episode_id``.
        """
        options = dict(options or {})
        with self._lock:
            self._ensure_started()
            if "operator_ready_receipt" in options:
                raise ValueError("reset options must not carry operator receipts")
            if "episode_id" in options:
                raise ValueError("reset options must not override YAM episode_id")
            ready = self._consume_ready_receipt_locked()
            if ready is None:
                self._runtime.hold()
                self._last_stop_hold_s = time.time()
                raise RuntimeError(
                    "YAM reset requires a local operator ready receipt matching "
                    f"pending episode_id={self._episode_id} at "
                    f"{self.operator_receipt_path!r}; call observe/status, write "
                    "event='ready' with operator_control.write_receipt, then reset"
                )
            reset_request_episode_id = self._episode_id
            self._begin_episode(seed=seed)
            ready = dict(ready)
            ready["reset_request_episode_id"] = reset_request_episode_id
            ready["ready_for_episode_id"] = self._episode_id
            self._operator_ready_receipt = ready
            self._stop_requested.clear()
            qpos = self._read_qpos()
            self._previous_command = qpos.copy()
            obs, info = self._observe_locked()
            return obs, info

    def observe(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            self._ensure_started()
            return self._observe_locked(require_valid_wrist_projection=True)

    def step(
        self,
        action: Any,
        *,
        action_type: Literal["qpos"] = "qpos",
        expected_episode_id: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if np.asarray(action).shape not in {(14,), (1, 14)}:
            raise ValueError("step requires a single qpos14 action")
        return self.chunk_step(
            action,
            action_type=action_type,
            return_all_frames=False,
            expected_episode_id=expected_episode_id,
        )

    def chunk_step(
        self,
        actions: Any,
        action_type: Literal["qpos"] = "qpos",
        return_all_frames: bool = False,
        expected_episode_id: str | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        array = validate_actions(actions, action_type=action_type)
        per_step: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        executed = 0
        last_result: Any = None
        with self._lock:
            self._ensure_started()
            self._require_expected_episode_locked(expected_episode_id)
            try:
                for requested in array:
                    self._poll_operator_receipt_locked()
                    self._require_expected_episode_locked(expected_episode_id)
                    if self._terminal_event is not None:
                        self._runtime.hold()
                        self._last_stop_hold_s = time.time()
                        break
                    if self._stop_requested.is_set():
                        self._runtime.hold()
                        self._last_stop_hold_s = time.time()
                        break
                    self._require_ready_for_motion()
                    if self._take_action_cnt >= self.step_lim:
                        self._runtime.hold()
                        self._last_stop_hold_s = time.time()
                        break
                    accepted, clipped = self._prepare_action(requested)
                    self._pace()
                    self._require_expected_episode_locked(expected_episode_id)
                    if self._stop_requested.is_set():
                        self._runtime.hold()
                        self._last_stop_hold_s = time.time()
                        break
                    self._check_measured_transition(accepted)
                    result = self._runtime.command(accepted)
                    last_result = result
                    if getattr(result, "rejection_reason", None):
                        self._runtime.hold()
                        self._stop_requested.set()
                        raise RuntimeError(
                            "YAM command rejected: "
                            + str(getattr(result, "rejection_reason"))
                        )
                    accepted_qpos = np.asarray(result.accepted, dtype=np.float64)
                    self._previous_command = accepted_qpos.copy()
                    self._take_action_cnt += 1
                    executed += 1
                    obs, info = self._observe_locked()
                    step_info = {
                        "requested_action": np.asarray(requested, dtype=np.float64),
                        "accepted_action": accepted_qpos,
                        "action_clipped": bool(
                            clipped or getattr(result, "clipped", False)
                        ),
                        "action_rejected": getattr(result, "rejection_reason", None),
                        "episode_status": info["episode_status"],
                        "robot_state": info["robot_state"],
                    }
                    per_step.append(step_info)
                    if return_all_frames:
                        frames.append(obs)
                    if self._terminal_event is not None:
                        self._runtime.hold()
                        self._last_stop_hold_s = time.time()
                        break
                    if self._take_action_cnt >= self.step_lim:
                        self._runtime.hold()
                        self._last_stop_hold_s = time.time()
                        break
                obs, info = self._observe_locked()
            except Exception:
                self._stop_requested.set()
                self._hold_after_failure()
                raise
        terminated = bool(self._terminal_event == "success")
        truncated = bool(
            self._terminal_event in {"failure", "abort"}
            or self._stop_requested.is_set()
            or self._take_action_cnt >= self.step_lim
        )
        reward = 1.0 if terminated else 0.0
        info.update(
            {
                "requested_actions": int(len(array)),
                "executed_actions": int(executed),
                "per_step": per_step,
                "stop_requested": self._stop_requested.is_set(),
            }
        )
        if last_result is not None:
            info["accepted_action"] = np.asarray(last_result.accepted, dtype=np.float64)
            info["action_clipped"] = bool(getattr(last_result, "clipped", False))
            info["action_rejected"] = getattr(last_result, "rejection_reason", None)
        if return_all_frames:
            obs = frames if frames else obs
        return obs, reward, terminated, truncated, info

    def render_camera(self, camera_name: str, depth: bool = False) -> Any:
        with self._lock:
            self._ensure_started()
            if self.last_obs is None:
                self._observe_locked(require_valid_wrist_projection=True)
            view = self.last_obs["views"][camera_name]
            if depth:
                return view["rgb"].copy(), view["depth"].copy()
            return view["rgb"].copy()

    def get_camera_meta(self, camera_name: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            if self.last_obs is None:
                self._observe_locked(require_valid_wrist_projection=True)
            return deepcopy(self.last_obs["views"][camera_name]["camera_meta"])

    def get_task_language(self) -> str:
        return self.task_language

    def plan_arm_path(
        self, arm: Literal["left", "right"], target_pose: Any
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_started()
            qpos = self._read_qpos()
            return self.geometry.plan_arm_path(arm, target_pose, current_qpos14=qpos)

    def request_stop(self) -> None:
        """Request stop without waiting for the dispatch lock.

        If the env is idle and the lock is available, issue a hold immediately
        through the same serialized runtime path. If a chunk is active, this call
        only sets the event; the active writer observes it at the next boundary
        and holds.
        """
        self._stop_requested.set()
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return
        try:
            if self._started and self._runtime is not None:
                self._runtime.hold()
                self._last_stop_hold_s = time.time()
        finally:
            self._lock.release()

    def _begin_episode(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        self._take_action_cnt = 0
        self._next_tick_s = None
        self._actual_seed = self._actual_seed if seed is None else int(seed)
        self._episode_id = uuid.uuid4().hex
        self._operator_success = None
        self._operator_success_source = None
        self._operator_success_receipt = None
        self._operator_ready_receipt = None
        self._terminal_event = None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            errors: list[Exception] = []
            if self._runtime is not None:
                try:
                    self._runtime.close()
                except Exception as error:
                    errors.append(error)
            if self._cameras is not None and self._owns_cameras:
                try:
                    self._cameras.close()
                except Exception as error:
                    errors.append(error)
            self._started = False
            if errors:
                self._startup_failed = True
                raise RuntimeError(
                    "failed to fully close YAM agent env: "
                    + "; ".join(str(error) for error in errors)
                ) from errors[0]
            if self._hardware_lease is not None:
                self._hardware_lease.close()
                self._hardware_lease = None
            self._closed = True

    def _ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("cannot use a closed YamAgentEnv")
        if self._startup_failed:
            raise RuntimeError(
                "YAM startup failed; operator intervention is required before restarting"
            )
        if self._started:
            return
        if self._cameras is None:
            self._cameras = YamRgbdCameraRig(
                self.config, calibration=self.geometry.calibration
            )
        try:
            if self._owns_runtime and self._hardware_lease is None:
                lease = HardwareLease(
                    [
                        self._device_config("left_follower", "can_left").channel,
                        self._device_config("right_follower", "can_right").channel,
                    ]
                )
                try:
                    lease.acquire()
                    lease.check_subscriptions()
                except BaseException:
                    lease.close()
                    raise
                self._hardware_lease = lease
            self._cameras.open()
            if self._runtime is None:
                self._runtime = self._build_runtime()
            self._runtime.connect_followers()
            self._runtime.hold()
            qpos = self._read_qpos()
            self._previous_command = qpos.copy()
            self._started = True
        except Exception:
            self._startup_failed = True
            runtime_closed = self._runtime is None
            if self._runtime is not None:
                try:
                    self._runtime.close()
                    runtime_closed = True
                except Exception:
                    pass
            if self._cameras is not None and self._owns_cameras:
                try:
                    self._cameras.close()
                except Exception:
                    pass
            if runtime_closed and self._hardware_lease is not None:
                self._hardware_lease.close()
                self._hardware_lease = None
            raise

    def _build_runtime(self) -> Any:
        rlinf_root = get_rlinf_repo_path()
        if rlinf_root is not None and str(rlinf_root) not in sys.path:
            sys.path.insert(0, str(rlinf_root))
        try:
            from rlinf.envs.realworld.yam.config import DualYamJointEnvConfig
            from rlinf.envs.realworld.yam.control_runtime import YamControlRuntime
            from rlinf.envs.realworld.yam.i2rt_backend import I2RTYamBackendFactory
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "missing dependency while importing RLinf YAM runtime. "
                "Set RPENT_RLINF_ROOT or RLINF_REPO_PATH to the RLinf checkout "
                "and install the pinned YAM/i2rt dependencies on the robot host."
            ) from error
        runtime_config = DualYamJointEnvConfig(
            task_description=self.task_language,
            step_frequency=self.control_hz,
            max_joint_delta=max(float(self.max_joint_delta_per_step or 0.05), 1e-6),
            # RPent validates hard limits, command slew and the measured-to-target
            # path. A second per-joint measured-position clip would reshape that
            # checked path when joints have different tracking errors.
            enforce_runtime_joint_limits=False,
            joint_limit_min=self.lower.tolist(),
            joint_limit_max=self.upper.tolist(),
            feedback_timeout_s=float(self.config.get("feedback_timeout_s", 0.25)),
        )
        hardware = SimpleNamespace(
            left_follower=self._device_config("left_follower", "can_left"),
            right_follower=self._device_config("right_follower", "can_right"),
            cameras=[],
        )
        return YamControlRuntime(runtime_config, hardware, I2RTYamBackendFactory())

    def _device_config(self, role: str, default_channel: str) -> SimpleNamespace:
        devices = dict(self.config.get("devices", self.config.get("robot", {})))
        values = dict(devices.get(role, {}))
        return SimpleNamespace(
            channel=str(values.get("channel", default_channel)),
            arm_type=str(values.get("arm_type", "yam")),
            gripper_type=str(values.get("gripper_type", "flexible_4310")),
            ee_mass=values.get("ee_mass"),
            gripper_limits=values.get("gripper_limits"),
            gravity_comp_factor=values.get(
                "gravity_comp_factor",
                [1.0, 1.1, 1.1, 1.2, 1.0, 1.0],
            ),
            grav_comp_kd=values.get("grav_comp_kd"),
            coulomb_friction=values.get("coulomb_friction"),
            use_coulomb_friction=bool(values.get("use_coulomb_friction", True)),
            enable_auto_recovery=bool(values.get("enable_auto_recovery", False)),
        )

    def _prepare_action(self, action: Any) -> tuple[np.ndarray, bool]:
        target = enforce_hard_limits(action, self.lower, self.upper, name="action")
        previous = (
            self._previous_command
            if self._previous_command is not None
            else self._read_qpos()
        )
        accepted, clipped = apply_previous_command_slew(
            target, previous, self.max_joint_delta_per_step
        )
        accepted = enforce_hard_limits(
            accepted, self.lower, self.upper, name="accepted_action"
        )
        table_guard = self.geometry.check_qpos_transition(previous, accepted)
        if not table_guard["ok"]:
            self._runtime.hold()
            raise RuntimeError(
                "YAM qpos motion rejected by table guard: "
                + str(table_guard.get("reason"))
            )
        return accepted, clipped

    def _check_measured_transition(self, target: np.ndarray) -> None:
        """Check the actual feedback-to-command segment immediately before sending."""
        measured = enforce_hard_limits(self._read_qpos(), self.lower, self.upper)
        if self.max_tracking_error_rad is not None:
            error = np.max(
                np.abs(target[ARM_JOINT_INDICES] - measured[ARM_JOINT_INDICES])
            )
            if error > self.max_tracking_error_rad:
                raise RuntimeError(
                    f"YAM tracking error {error:.4f} rad exceeds configured bound"
                )
        check = self.geometry.check_qpos_transition(measured, target)
        if not check["ok"]:
            raise RuntimeError(
                f"YAM measured trajectory rejected: {check.get('reason')}"
            )

    def _read_qpos(self) -> np.ndarray:
        state = self._runtime.read_state()
        return np.asarray(state.as_vector(), dtype=np.float64).reshape(14)

    def _observe_locked(
        self, *, require_valid_wrist_projection: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._poll_operator_receipt_locked()
        deadline = time.monotonic() + float(
            self.config.get(
                "projection_observe_timeout_s",
                self.config.get("camera_frame_timeout_s", 1.0),
            )
        )
        while True:
            obs, info = self._capture_observation_once_locked(
                wait_for_fresh_frame=require_valid_wrist_projection
            )
            if not require_valid_wrist_projection:
                return obs, info
            if self._wrist_projection_ready(obs):
                return obs, info
            if self._wrist_projection_has_configuration_error(obs):
                return obs, info
            if time.monotonic() >= deadline:
                return obs, info
            time.sleep(0.005)

    def _capture_observation_once_locked(
        self, *, wait_for_fresh_frame: bool = True
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        qpos_before, timestamps_before = self._read_qpos_with_timestamps()
        # A motion chunk consumes the background cache without waiting for
        # a new camera frame on every command. Explicit observe waits for
        # a bounded stationary snapshot before exposing wrist world points.
        snapshot = self._cameras.snapshot(
            not_before_monotonic_s=(
                timestamps_before["host_after_monotonic_s"]
                if wait_for_fresh_frame
                else None
            )
        )
        qpos_after, timestamps_after = self._read_qpos_with_timestamps()
        qpos = enforce_hard_limits(
            qpos_after, self.lower, self.upper, name="observed_qpos"
        )
        qpos_delta = qpos_after - qpos_before
        qpos_delta_linf = float(np.max(np.abs(qpos_delta)))
        qpos_static_tolerance = float(
            self.config.get("qpos_static_tolerance_rad", 1e-3)
        )
        camera_timeout_s = float(self.config.get("camera_frame_timeout_s", 1.0))
        now_monotonic_s = time.monotonic()
        views = {}
        for name, frame in snapshot["views"].items():
            metadata_error = None
            try:
                camera_meta = self.geometry.camera_meta_for_qpos(
                    name, frame.camera_meta, qpos
                )
            except Exception as error:
                if name not in {"left", "right"}:
                    raise
                metadata_error = str(error)
                camera_meta = dict(frame.camera_meta)
                camera_meta["cam2world_source"] = "unavailable_missing_wrist_handeye"
                camera_meta["cam2world_cv"] = np.full((4, 4), np.nan, dtype=np.float64)
            timestamps = dict(camera_meta.get("timestamps", {}))
            host_before_monotonic_s = timestamps.get("host_before_monotonic_s")
            host_after_monotonic_s = timestamps.get("host_after_monotonic_s")
            frame_age_s = (
                None
                if host_after_monotonic_s is None
                else float(now_monotonic_s - float(host_after_monotonic_s))
            )
            stale_for_projection = frame_age_s is None or frame_age_s > camera_timeout_s
            frame_bracketed = (
                host_before_monotonic_s is not None
                and host_after_monotonic_s is not None
                and timestamps_before["host_after_monotonic_s"]
                <= float(host_before_monotonic_s)
                and float(host_after_monotonic_s)
                <= timestamps_after["host_before_monotonic_s"]
            )
            moving_for_projection = (
                name in {"left", "right"} and qpos_delta_linf > qpos_static_tolerance
            )
            if name == "top":
                projection_valid = metadata_error is None and not stale_for_projection
            else:
                projection_valid = (
                    metadata_error is None
                    and not stale_for_projection
                    and frame_bracketed
                    and not moving_for_projection
                )
            projection_limitation = None
            if metadata_error is not None:
                projection_limitation = (
                    "missing_or_invalid_wrist_handeye: " + metadata_error
                )
            elif stale_for_projection:
                projection_limitation = (
                    "cached RGBD frame is stale or lacks host timestamp"
                )
            elif name != "top" and not frame_bracketed:
                projection_limitation = (
                    "camera frame host capture interval is not bracketed by qpos-before "
                    "and qpos-after host timestamps"
                )
            elif moving_for_projection:
                projection_limitation = (
                    "wrist cam2world uses current FK for a cached frame while joints moved "
                    f"{qpos_delta_linf:.6g} rad; require <= {qpos_static_tolerance:.6g}"
                )
            camera_meta.update(
                {
                    "frame_age_s": frame_age_s,
                    "projection_valid": bool(projection_valid),
                    "projection_limitation": projection_limitation,
                }
            )
            views[name] = {
                "rgb": frame.rgb.copy(),
                "depth": frame.depth.copy(),
                "camera_meta": camera_meta,
            }
        obs = {
            "frames": {name: views[name]["rgb"] for name in YAM_CAMERA_NAMES},
            "state": {"joint_position": qpos.astype(np.float64)},
            "views": views,
            "snapshot_id": str(snapshot["snapshot_id"]),
        }
        robot_state = self.geometry.robot_state(qpos)
        info = {
            "episode_status": self._episode_status(),
            "robot_state": robot_state,
            "commanded_qpos": self._previous_command.copy(),
        }
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def _episode_status(self) -> dict[str, Any]:
        return {
            "eval_success": bool(self._operator_success is True),
            "eval_success_source": self._operator_success_source or "operator_unset",
            "terminal_event": self._terminal_event,
            "stop_requested": self._stop_requested.is_set(),
            "take_action_cnt": int(self._take_action_cnt),
            "step_lim": int(self.step_lim),
            "actual_seed": int(self._actual_seed),
            "episode_id": self._episode_id,
            "ready_for_motion": self._operator_ready_receipt is not None,
            "operator_ready_receipt": (
                None
                if self._operator_ready_receipt is None
                else dict(self._operator_ready_receipt)
            ),
            "operator_success_receipt": (
                None
                if self._operator_success_receipt is None
                else dict(self._operator_success_receipt)
            ),
            "last_stop_hold_s": self._last_stop_hold_s,
        }

    def _pace(self) -> None:
        period_s = 1.0 / self.control_hz
        now = time.perf_counter()
        if self._next_tick_s is not None and now < self._next_tick_s:
            time.sleep(self._next_tick_s - now)
        self._next_tick_s = time.perf_counter() + period_s

    def _hold_after_failure(self) -> None:
        if self._runtime is None:
            return
        try:
            self._runtime.hold()
            self._last_stop_hold_s = time.time()
        except Exception:
            try:
                self._runtime.emergency_hold()
            except Exception:
                pass

    def _read_qpos_with_timestamps(self) -> tuple[np.ndarray, dict[str, float]]:
        host_before_time_s = time.time()
        host_before_monotonic_s = time.monotonic()
        state = self._runtime.read_state()
        host_after_monotonic_s = time.monotonic()
        host_after_time_s = time.time()
        return (
            np.asarray(state.as_vector(), dtype=np.float64).reshape(14),
            {
                "left": float(state.left.timestamp_s),
                "right": float(state.right.timestamp_s),
                "host_before_time_s": host_before_time_s,
                "host_after_time_s": host_after_time_s,
                "host_before_monotonic_s": host_before_monotonic_s,
                "host_after_monotonic_s": host_after_monotonic_s,
            },
        )

    @staticmethod
    def _wrist_projection_ready(obs: dict[str, Any]) -> bool:
        return all(
            bool(obs["views"][name]["camera_meta"].get("projection_valid"))
            for name in ("left", "right")
        )

    @staticmethod
    def _wrist_projection_has_configuration_error(obs: dict[str, Any]) -> bool:
        for name in ("left", "right"):
            limitation = str(
                obs["views"][name]["camera_meta"].get("projection_limitation") or ""
            )
            if "missing_or_invalid_wrist_handeye" in limitation:
                return True
        return False

    def _require_expected_episode_locked(self, expected_episode_id: str | None) -> None:
        if expected_episode_id is None:
            return
        if str(expected_episode_id) == self._episode_id:
            return
        self._runtime.hold()
        self._last_stop_hold_s = time.time()
        raise RuntimeError(
            "YAM expected_episode_id mismatch: expected "
            f"{expected_episode_id!r}, current {self._episode_id!r}"
        )

    def _require_ready_for_motion(self) -> None:
        if self.config.get("require_table_guard", False) and not (
            self.geometry.table_guard_configured
        ):
            raise RuntimeError(
                "YAM motion requires verified table geometry in the site config"
            )
        if self._operator_ready_receipt is None:
            self._runtime.hold()
            self._last_stop_hold_s = time.time()
            raise RuntimeError(
                "YAM motion requires an operator ready receipt matching "
                f"episode_id={self._episode_id} at {self.operator_receipt_path!r}"
            )

    def _validate_operator_receipt(
        self,
        receipt: dict[str, Any],
        *,
        event: Literal["ready", "success", "failure", "abort"],
    ) -> dict[str, Any]:
        if not isinstance(receipt, dict):
            raise TypeError("operator receipt must be a dict")
        normalized = dict(receipt)
        if normalized.get("event") != event:
            raise ValueError(f"operator receipt event must be {event!r}")
        if normalized.get("episode_id") != self._episode_id:
            raise ValueError(
                "operator receipt episode_id does not match current episode"
            )
        if normalized.get("source") != "operator_local":
            raise ValueError("operator receipt source must be operator_local")
        request_id = str(normalized.get("request_id", ""))
        if not request_id:
            raise ValueError("operator receipt requires request_id")
        receipt_key = f"{event}:{request_id}"
        if receipt_key in self._consumed_receipt_ids:
            raise ValueError("operator receipt request_id has already been consumed")
        self._consumed_receipt_ids.add(receipt_key)
        return normalized

    def _consume_ready_receipt_locked(self) -> dict[str, Any] | None:
        receipt = read_receipt(self.operator_receipt_path)
        if receipt is None or receipt.get("episode_id") != self._episode_id:
            return None
        if receipt.get("event") != "ready":
            return None
        key = f"ready:{receipt['request_id']}"
        if key in self._consumed_receipt_ids:
            return None
        ready = self._validate_operator_receipt(receipt, event="ready")
        self._operator_ready_receipt = ready
        return dict(ready)

    def _poll_operator_receipt_locked(self) -> dict[str, Any] | None:
        receipt = read_receipt(self.operator_receipt_path)
        if receipt is None or receipt.get("episode_id") != self._episode_id:
            return None
        key = f"{receipt['event']}:{receipt['request_id']}"
        if key in self._consumed_receipt_ids:
            return None
        event = receipt["event"]
        if event == "ready":
            return None
        if event in {"success", "failure", "abort"}:
            self._consume_terminal_receipt(receipt)
            self._runtime.hold()
            self._last_stop_hold_s = time.time()
            return receipt
        return None

    def _consume_terminal_receipt(self, receipt: dict[str, Any]) -> None:
        event = str(receipt.get("event"))
        if event not in {"success", "failure", "abort"}:
            raise ValueError("terminal receipt event must be success/failure/abort")
        terminal = self._validate_operator_receipt(
            receipt,
            event=event,  # type: ignore[arg-type]
        )
        self._terminal_event = event
        self._operator_success = event == "success"
        self._operator_success_source = str(terminal["source"])
        self._operator_success_receipt = {
            **terminal,
            "eval_success": event == "success",
        }
        if event == "abort":
            self._stop_requested.set()


__all__ = ["YamAgentEnv"]
