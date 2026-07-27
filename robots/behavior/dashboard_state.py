"""Thread-safe, environment-aware state for the shared live dashboard."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

_BEHAVIOR_TOOLS = frozenset(
    {
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
        "navigate_to",
    }
)
_PUBLIC_RUN_METADATA = frozenset(
    {
        "planner",
        "model",
        "reasoning-effort",
        "task-name",
        "task-language",
        "task-index",
        "activity-definition-id",
        "activity-instance-id",
        "public-instance-id",
        "candidate-instance-id",
        "public-seed",
        "public-seed-max",
        "scene-model",
        "behavior-phase",
        "job-id",
        "campaign-position",
        "max-episode-steps",
        "max-tool-calls",
        "max-wall-clock-s",
        "public-tool-contract-version",
        "public-tool-count",
        "eval-cohort",
        "controller",
        "llm-enabled",
        "cuda-device",
        "health-status",
        "health-checked-at",
    }
)
_PRIVATE_KEYS = frozenset(
    {
        "gt",
        "stage",
        "ground_truth_pose",
        "suggested_next_tool",
        "suggested_tool",
        "strategy",
        "hidden_object_path",
        "path",
        "rgb_path",
        "image_path",
        "image_cam_path",
        "overlay_path",
    }
)
_BEHAVIOR_CAMERA_ALIASES = {
    "main": "head",
    "agent": "head",
    "head": "head",
    "left": "left_wrist",
    "left_wrist": "left_wrist",
    "right": "right_wrist",
    "right_wrist": "right_wrist",
}
_UNSET = object()


def _json_safe(value: Any) -> Any:
    """Return a bounded JSON-safe value without private/runtime-only fields."""

    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            if (
                name.startswith("_")
                or lowered in _PRIVATE_KEYS
                or lowered.startswith("ground_truth")
                or lowered.startswith("suggested_")
            ):
                continue
            public[name] = _json_safe(item)
        return public
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_image(path: Any) -> bytes | None:
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except (OSError, TypeError, ValueError):
        return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class State:
    """Thread-safe dashboard state for one LIBERO or BEHAVIOR run."""

    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        output_dir: str,
        video_path: str,
        suite: str = "",
        task: int = 0,
        seed: int = 0,
        environment: str | None = None,
        identity: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        frame_roles: Mapping[str, Any] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        inferred = "behavior" if str(suite).startswith("behavior") else "libero"
        self.environment = str(environment or inferred).strip().lower()
        if self.environment not in {"behavior", "libero"}:
            self.environment = inferred
        self.run_id = str(run_id)
        self.name = str(name)
        self.suite = str(suite)
        self.task = int(task)
        self.seed = int(seed)
        self.output_dir = Path(output_dir)
        self._job_output_dir = self.output_dir.resolve(strict=False)
        self.video_path = Path(video_path)
        self.identity = _json_safe(
            dict(
                identity
                or {
                    "suite": self.suite,
                    "task": self.task,
                    "seed": self.seed,
                }
            )
        )

        if frame_roles is None:
            frame_kinds = (
                ("head", "left_wrist", "right_wrist")
                if self.environment == "behavior"
                else ("agent", "camera")
            )
        elif isinstance(frame_roles, Mapping):
            frame_kinds = tuple(str(kind) for kind in frame_roles)
        else:
            frame_kinds = tuple(str(kind) for kind in frame_roles)
        if not frame_kinds:
            frame_kinds = ("agent", "camera")

        self._lock = threading.RLock()
        self._state = "running"
        self._terminated = False
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        self._timeline_revision = 0
        self._next_timeline_id = 1
        self._pending_tool_args: dict[str, list[dict[str, Any]]] = {}
        self._event_targets: dict[str, list[int]] = {}
        self._metadata: dict[str, Any] = {}
        self._frames_png: dict[str, bytes | None] = dict.fromkeys(frame_kinds)
        self._frame_indices: dict[str, int] = dict.fromkeys(frame_kinds, -1)
        self._frame_revisions: dict[str, int] = dict.fromkeys(frame_kinds, 0)
        self._frame_idx = -1
        self._capture_group_id: str | int | None = None
        self._simulator_step: int | None = None
        self._last_selected_camera = (
            "head" if self.environment == "behavior" else frame_kinds[0]
        )
        self._control_controller: Any = None
        self._control_snapshot: dict[str, Any] = {
            "available": False,
            "motion_available": False,
            "observe_available": False,
            "busy": False,
            "owner": None,
            "phase": "idle",
            "selected_camera": self._last_selected_camera,
            "unavailable_reason": "controller_not_bound",
        }
        self._video_generation = 0
        self._sealed_video_generation: int | None = None
        self._sealed_episode_replays: dict[int, dict[str, Any]] = {}
        self._progress: dict[str, Any] = {
            "attempt_index": 1,
            "attempts_completed": 0,
            "attempt_outcome": None,
            "job_unlimited": True,
            "official_task_success": False,
            "workflow_complete": False,
            "publication_complete": False,
            "artifact_seal_complete": False,
            "artifact_seal_warning": False,
            "cumulative_env_steps": 0,
            "cumulative_tool_calls": 0,
            "cumulative_vla_chunks": 0,
            "total_env_steps": 0,
            "global_tool_calls": 0,
            "global_vla_chunks": 0,
            "global_vla_invocations": 0,
            "elapsed_wall_clock_s": 0.0,
            "max_episode_steps": None,
            "max_tool_calls": 350 if self.environment == "behavior" else None,
            "max_wall_clock_s": None,
        }
        if metadata:
            self.set_metadata(dict(metadata))

    # -- public event inputs ----------------------------------------------

    def on_event(self, event: dict[str, Any]) -> None:
        if not isinstance(event, dict):
            return
        event_type = str(event.get("type") or "")
        tool_name = str(event.get("tool") or "")
        if self._ignore_tool(tool_name) and event_type in {
            "tool_call",
            "tool_result",
            "tool_progress",
        }:
            return
        safe_event = _json_safe(event)
        if not isinstance(safe_event, dict):
            return
        with self._lock:
            if event_type == "official_success":
                if self._trusted_attempt_event(event, required_key="task_success"):
                    self._progress["official_task_success"] = True
                    if isinstance(event.get("workflow_complete"), bool):
                        self._progress["workflow_complete"] = event["workflow_complete"]
                    if isinstance(event.get("artifact_seal_complete"), bool):
                        self._progress["artifact_seal_complete"] = event[
                            "artifact_seal_complete"
                        ]
                    if isinstance(event.get("publication_complete"), bool):
                        self._progress["publication_complete"] = event[
                            "publication_complete"
                        ]
                    self._terminated = True
                    if self._timeline:
                        self._timeline[-1]["terminated"] = True
                self._update_artifact_warning_locked()
            elif event_type == "publication_complete":
                if self._trusted_attempt_event(
                    event, required_key="publication_complete"
                ):
                    self._progress["publication_complete"] = True
                self._update_artifact_warning_locked()
            elif event_type == "workflow_complete":
                if self._trusted_attempt_event(event, required_key="workflow_complete"):
                    self._progress["workflow_complete"] = True
                    self._progress["artifact_seal_complete"] = bool(
                        event.get("artifact_seal_complete", True)
                    )
                self._update_artifact_warning_locked()

            if event_type == "tool_call" and tool_name:
                args = event.get("args")
                safe_args = _json_safe(args) if isinstance(args, dict) else {}
                targets = self._event_targets.get(tool_name, [])
                if targets:
                    index = targets.pop(0)
                    if 0 <= index < len(self._timeline):
                        self._timeline[index]["args"] = safe_args
                        self._timeline_revision += 1
                else:
                    self._pending_tool_args.setdefault(tool_name, []).append(safe_args)
            self._events.append(safe_event)

    def on_usage(self, *, inp: int, out: int, tool_calls: int) -> None:
        with self._lock:
            self._usage = {
                "in": int(inp),
                "out": int(out),
                "tool_calls": int(tool_calls),
            }

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> int:
        name = str(name)
        if self._ignore_tool(name):
            return -1
        with self._lock:
            pending = self._pending_tool_args.get(name, [])
            event_args = pending.pop(0) if pending else None
            args = event_args if event_args is not None else _json_safe(arguments)
            timeline_id = self._next_timeline_id
            self._next_timeline_id += 1
            ordinal = len(self._timeline) + 1
            item = {
                "timeline_id": timeline_id,
                "ordinal": ordinal,
                "step": ordinal,
                "env_step": None,
                "action": name,
                "args": args if isinstance(args, dict) else {},
                "result": {},
                "elapsed_s": 0.0,
                "terminated": bool(
                    self.environment == "behavior"
                    and self._progress["official_task_success"]
                ),
                "has_action_video": False,
                "status": "running",
                "_started_at": time.monotonic(),
            }
            self._timeline.append(item)
            if event_args is None:
                self._event_targets.setdefault(name, []).append(len(self._timeline) - 1)
            self._timeline_revision += 1
            return timeline_id

    def on_tool_progress(self, name: str, result: dict[str, Any]) -> None:
        name = str(name)
        if self._ignore_tool(name) or not isinstance(result, dict):
            return
        self._ingest_frames(result)
        safe_result = _json_safe(result)
        if not isinstance(safe_result, dict):
            safe_result = {}
        for frame_key in ("visual_review", "views", "images"):
            safe_result.pop(frame_key, None)
        event = {"type": "tool_progress", "tool": name, **safe_result}
        with self._lock:
            index = self._running_index_locked(name)
            if index is None:
                self.on_tool_start(name, {})
                index = len(self._timeline) - 1
            item = self._timeline[index]
            env_step = self._env_step(result)
            if env_step is not None:
                item["env_step"] = env_step
            item["result"] = safe_result
            self._events.append(event)
            self._update_progress_from_payload_locked(result)
            self._timeline_revision += 1

    def on_tool_result(self, name: str, result: Any) -> None:
        name = str(name)
        if self._ignore_tool(name) or not isinstance(result, dict):
            return
        self._ingest_frames(result)
        log = result.get("log")
        command = log.get("command") if isinstance(log, dict) else None
        log_result = log.get("result") if isinstance(log, dict) else None
        safe_result = _json_safe(log_result if log_result is not None else result)
        if not isinstance(safe_result, dict):
            safe_result = {}

        with self._lock:
            index = self._running_index_locked(name)
            if index is None:
                pending = self._pending_tool_args.get(name, [])
                event_args = pending.pop(0) if pending else None
                args = (
                    event_args
                    if event_args is not None
                    else {
                        key: _json_safe(value)
                        for key, value in command.items()
                        if key != "action"
                    }
                    if isinstance(command, dict) and command.get("action") == name
                    else {}
                )
                timeline_id = self._next_timeline_id
                self._next_timeline_id += 1
                ordinal = len(self._timeline) + 1
                item = {
                    "timeline_id": timeline_id,
                    "ordinal": ordinal,
                    "step": ordinal,
                    "env_step": None,
                    "action": name,
                    "args": args,
                    "result": {},
                    "elapsed_s": None,
                    "terminated": False,
                    "has_action_video": False,
                    "status": "running",
                    "_started_at": time.monotonic(),
                }
                self._timeline.append(item)
                index = len(self._timeline) - 1
                if event_args is None:
                    self._event_targets.setdefault(name, []).append(index)

            item = self._timeline[index]
            env_step = self._env_step(result)
            item["env_step"] = env_step
            item["result"] = safe_result
            reported_elapsed = log.get("elapsed_s") if isinstance(log, dict) else None
            if reported_elapsed is None:
                reported_elapsed = result.get("elapsed_s")
            if isinstance(reported_elapsed, (int, float)) and not isinstance(
                reported_elapsed, bool
            ):
                elapsed = max(0.0, float(reported_elapsed))
            else:
                elapsed = max(0.0, time.monotonic() - item["_started_at"])
            item["elapsed_s"] = elapsed

            if self.environment == "behavior":
                terminated = bool(self._progress["official_task_success"])
                safe_rejected = (
                    terminated
                    and result.get("stop_reason") == "precondition_rejected"
                    and "official_success_latched"
                    in (
                        result.get("failed_preconditions")
                        if isinstance(result.get("failed_preconditions"), list)
                        else []
                    )
                )
                item["status"] = (
                    "safe_rejected"
                    if safe_rejected
                    else "failed"
                    if self._tool_failed(result)
                    else "completed"
                )
                item["terminated"] = terminated
            else:
                terminated = bool(result.get("libero_terminated"))
                self._terminated = self._terminated or terminated
                item["terminated"] = terminated
                item["status"] = "failed" if self._tool_failed(result) else "completed"

            item["has_action_video"] = self._action_video_path_locked(item).exists()
            self._update_progress_from_payload_locked(result)
            self._timeline_revision += 1

    def on_job_progress(self, progress: dict[str, Any]) -> None:
        if not isinstance(progress, dict):
            return
        mapping = {
            "attempts": "attempts_completed",
            "env_steps": "cumulative_env_steps",
            "tool_calls": "cumulative_tool_calls",
            "vla_chunks": "cumulative_vla_chunks",
            "wall_clock_s": "elapsed_wall_clock_s",
        }
        with self._lock:
            for source, target in mapping.items():
                value = progress.get(source)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._progress[target] = value

    def on_frame(self, kind: str, image: bytes, *, env_step: Any = None) -> None:
        if not isinstance(image, bytes):
            return
        physical = self._physical_camera(kind, frame_id=None)
        if not physical:
            return
        with self._lock:
            if physical not in self._frames_png:
                return
            self._frames_png[physical] = bytes(image)
            step = int(env_step) if _is_int(env_step) else -1
            self._frame_indices[physical] = step
            self._frame_revisions[physical] += 1
            self._capture_group_id = None
            self._simulator_step = step if step >= 0 else None
            self._frame_idx += 1

    def on_frame_group(
        self,
        frames: Mapping[str, Any],
        *,
        capture_group_id: str | int,
        simulator_step: int,
    ) -> bool:
        """Atomically publish one complete, same-step BEHAVIOR capture group."""

        expected = {"head", "left_wrist", "right_wrist"}
        if (
            self.environment != "behavior"
            or set(frames) != expected
            or not all(isinstance(frames[camera], bytes) for camera in expected)
            or not isinstance(capture_group_id, (str, int))
            or isinstance(capture_group_id, bool)
            or capture_group_id == ""
            or not _is_int(simulator_step)
            or int(simulator_step) < 0
        ):
            return False
        with self._lock:
            if not expected.issubset(self._frames_png):
                return False
            if (
                self._simulator_step is not None
                and int(simulator_step) < self._simulator_step
            ):
                return False
            if self._capture_group_id == capture_group_id:
                if self._simulator_step != int(simulator_step):
                    return False
                return all(
                    self._frames_png[camera] == bytes(frames[camera])
                    for camera in expected
                )
            for camera in ("head", "left_wrist", "right_wrist"):
                self._frames_png[camera] = bytes(frames[camera])
                self._frame_indices[camera] = int(simulator_step)
                self._frame_revisions[camera] += 1
            self._capture_group_id = capture_group_id
            self._simulator_step = int(simulator_step)
            self._frame_idx += 1
            return True

    # -- Dashboard manual-control binding --------------------------------

    def bind_controller(self, controller: Any) -> None:
        snapshot_callback = getattr(controller, "snapshot", None)
        if not callable(snapshot_callback):
            raise TypeError("controller must provide snapshot()")
        snapshot = snapshot_callback()
        if not isinstance(snapshot, Mapping):
            raise TypeError("controller snapshot must be a mapping")
        with self._lock:
            if self._control_controller not in (None, controller):
                raise RuntimeError("a different Dashboard controller is already bound")
            self._control_controller = controller
            self._control_snapshot = dict(_json_safe(snapshot))

    def unbind_controller(self, controller: Any = None) -> None:
        with self._lock:
            if (
                controller is not None
                and self._control_controller is not controller
            ):
                return
            self._control_controller = None
            selected_camera = self._last_selected_camera
            self._control_snapshot = {
                "available": False,
                "motion_available": False,
                "observe_available": False,
                "busy": False,
                "owner": None,
                "phase": "idle",
                "selected_camera": selected_camera,
                "unavailable_reason": "controller_not_bound",
            }

    def control_controller(self) -> Any:
        """Return the binding only; callers must invoke it after this lock exits."""

        with self._lock:
            return self._control_controller

    def update_control_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        safe = _json_safe(snapshot)
        if not isinstance(safe, dict):
            return
        with self._lock:
            safe["selected_camera"] = self._last_selected_camera
            self._control_snapshot = safe

    def control_admission_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "official_task_success": bool(
                    self._progress["official_task_success"]
                ),
            }

    def set_selected_camera(self, camera: str) -> None:
        camera = str(camera)
        if (
            self.environment == "behavior"
            and camera not in {"head", "left_wrist", "right_wrist"}
        ):
            raise ValueError("invalid BEHAVIOR camera")
        with self._lock:
            if camera not in self._frames_png:
                raise ValueError("camera is not available for this run")
            self._last_selected_camera = camera
            self._control_snapshot["selected_camera"] = camera

    def on_manual_command_start(self, command: Mapping[str, Any]) -> int:
        """Add a manual Timeline row without changing Agent tool accounting."""

        command_id = str(command.get("command_id") or "")
        if not command_id:
            raise ValueError("manual command_id is required")
        target = str(command.get("target") or "")
        manual_action = str(command.get("action") or "")
        primitive = (
            "navigate_to"
            if target == "chassis"
            and manual_action
            in {"forward", "backward", "turn_left", "turn_right"}
            else manual_action
        )
        with self._lock:
            timeline_id = self._next_timeline_id
            self._next_timeline_id += 1
            ordinal = len(self._timeline) + 1
            item = {
                "timeline_id": timeline_id,
                "ordinal": ordinal,
                "step": ordinal,
                "env_step": None,
                "source": "dashboard_manual",
                "action": primitive,
                "manual_action": manual_action,
                "target": target,
                "command_id": command_id,
                "lease_id": str(command.get("lease_id") or ""),
                "sequence": command.get("sequence"),
                "primitive": primitive,
                "args": {
                    "target": target,
                    "action": manual_action,
                    "camera": str(command.get("camera") or ""),
                    "requested_step": self._manual_requested_step(
                        target, manual_action
                    ),
                    "detail": (
                        "relative jog"
                        if primitive == "navigate_to"
                        else None
                    ),
                },
                "result": {},
                "elapsed_s": 0.0,
                "primitive_success": None,
                "stop_reason": None,
                "capture_group_id": None,
                "partial_motion": False,
                "terminated": bool(self._progress["official_task_success"]),
                "has_action_video": False,
                "status": "running",
                "_started_at": time.monotonic(),
            }
            self._timeline.append(item)
            self._timeline_revision += 1
            return timeline_id

    def on_manual_command_result(
        self,
        command: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        official_success_latched: bool,
    ) -> None:
        """Atomically merge a manual receipt, final frames and Timeline result."""

        if not isinstance(result, Mapping):
            return
        safe_result = _json_safe(result)
        if not isinstance(safe_result, dict):
            safe_result = {}
        command_id = str(command.get("command_id") or "")
        frames = result.get("_frames_bytes")
        capture_group_id = result.get("capture_group_id")
        simulator_step = result.get(
            "simulator_step",
            self._env_step(dict(result)),
        )
        with self._lock:
            frame_group_committed = bool(
                isinstance(frames, Mapping)
                and set(frames) == {"head", "left_wrist", "right_wrist"}
                and self.on_frame_group(
                    frames,
                    capture_group_id=capture_group_id,
                    simulator_step=simulator_step,
                )
            )
            item = next(
                (
                    candidate
                    for candidate in reversed(self._timeline)
                    if candidate.get("source") == "dashboard_manual"
                    and candidate.get("command_id") == command_id
                ),
                None,
            )
            if item is None:
                return
            env_step = self._env_step(dict(result))
            if env_step is not None:
                item["env_step"] = env_step
            primitive = result.get("primitive_used") or result.get("primitive")
            if primitive:
                item["primitive"] = str(primitive)
                if item.get("action") != "navigate_to":
                    item["action"] = str(primitive)
            item["result"] = safe_result
            elapsed = result.get("elapsed_s")
            item["elapsed_s"] = (
                max(0.0, float(elapsed))
                if isinstance(elapsed, (int, float))
                and not isinstance(elapsed, bool)
                else max(0.0, time.monotonic() - item["_started_at"])
            )
            failed = self._tool_failed(dict(result))
            item["primitive_success"] = not failed
            item["stop_reason"] = result.get("stop_reason")
            metrics = result.get("metrics")
            item["partial_motion"] = bool(
                result.get("partial_motion")
                or (
                    isinstance(metrics, Mapping)
                    and metrics.get("partial_motion")
                )
            )
            item["capture_group_id"] = (
                capture_group_id
                if frame_group_committed
                and capture_group_id == self._capture_group_id
                else None
            )
            item["status"] = "failed" if failed else "completed"
            if official_success_latched:
                self._progress["official_task_success"] = True
                self._terminated = True
                self._control_snapshot["available"] = False
                self._control_snapshot["motion_available"] = False
                self._control_snapshot["observe_available"] = False
                self._control_snapshot["success_latched"] = True
                self._control_snapshot["unavailable_reason"] = (
                    "official_success_latched"
                )
            item["terminated"] = bool(self._progress["official_task_success"])
            self._update_progress_from_payload_locked(dict(result))
            self._timeline_revision += 1

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        """Attach public run identity and budget metadata."""

        if not isinstance(metadata, dict):
            return
        with self._lock:
            for key, value in metadata.items():
                if key not in _PUBLIC_RUN_METADATA or value in (None, ""):
                    continue
                self._metadata[key] = _json_safe(value)
            budget_map = {
                "max-episode-steps": "max_episode_steps",
                "max-tool-calls": "max_tool_calls",
                "max-wall-clock-s": "max_wall_clock_s",
            }
            for source, target in budget_map.items():
                value = self._metadata.get(source)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._progress[target] = value

    def set_budget_limits(
        self,
        *,
        max_episode_steps: int | None | object = _UNSET,
        max_tool_calls: int | None | object = _UNSET,
        max_wall_clock_s: int | float | None | object = _UNSET,
    ) -> None:
        """Set explicit run budgets, including an intentional unlimited value."""

        updates = {
            "max_episode_steps": max_episode_steps,
            "max_tool_calls": max_tool_calls,
            "max_wall_clock_s": max_wall_clock_s,
        }
        with self._lock:
            for key, value in updates.items():
                if value is _UNSET:
                    continue
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise ValueError(f"{key} must be non-negative or None")
                self._progress[key] = value

    # -- attempt and completion lifecycle ---------------------------------

    def begin_attempt(
        self,
        *,
        attempt_index: int,
        output_dir: str | Path,
        video_path: str | Path,
    ) -> None:
        with self._lock:
            self.output_dir = Path(output_dir)
            self.video_path = Path(video_path)
            self._state = "running"
            self._terminated = False
            self._sealed_video_generation = None
            self._video_generation += 1
            for kind in self._frames_png:
                self._frames_png[kind] = None
                self._frame_indices[kind] = -1
                self._frame_revisions[kind] = 0
            self._frame_idx = -1
            self._capture_group_id = None
            self._simulator_step = None
            self._last_selected_camera = (
                "head"
                if self.environment == "behavior"
                else next(iter(self._frames_png))
            )
            self._control_snapshot.update(
                {
                    "selected_camera": self._last_selected_camera,
                    "command_id": None,
                    "lease_id": None,
                    "target": None,
                    "action": None,
                    "phase": "idle",
                    "error": None,
                    "stop_reason": None,
                    "success_latched": False,
                }
            )
            self._progress.update(
                {
                    "attempt_index": int(attempt_index),
                    "attempt_outcome": None,
                    "official_task_success": False,
                    "workflow_complete": False,
                    "publication_complete": False,
                    "artifact_seal_complete": False,
                    "artifact_seal_warning": False,
                    "total_env_steps": 0,
                    "global_tool_calls": 0,
                    "global_vla_chunks": 0,
                    "global_vla_invocations": 0,
                    "elapsed_wall_clock_s": 0.0,
                }
            )
            self._timeline_revision += 1

    def end_attempt(self, *, attempt_index: int, outcome: str) -> None:
        with self._lock:
            if int(attempt_index) != int(self._progress["attempt_index"]):
                return
            self._progress["attempts_completed"] = max(
                int(self._progress["attempts_completed"]),
                int(attempt_index),
            )
            self._progress["attempt_outcome"] = str(outcome)
            video_path = self._trusted_video_path_locked(self.video_path)
            if (
                video_path is not None
                and self._video_generation not in self._sealed_episode_replays
            ):
                self._sealed_episode_replays[self._video_generation] = {
                    "generation": self._video_generation,
                    "attempt_index": int(attempt_index),
                    "outcome": str(outcome),
                    "path": video_path,
                }

    def mark_done(self, terminated: bool | None = None) -> None:
        with self._lock:
            now = time.monotonic()
            for item in self._timeline:
                if item["status"] == "running":
                    item["status"] = "interrupted"
                    item["elapsed_s"] = max(0.0, now - item["_started_at"])
                    self._timeline_revision += 1
            self._state = "done"
            self._control_snapshot.update(
                {
                    "available": False,
                    "motion_available": False,
                    "observe_available": False,
                    "unavailable_reason": "run_finished",
                }
            )
            if self.environment == "behavior":
                self._terminated = bool(self._progress["official_task_success"])
            elif terminated is not None:
                self._terminated = self._terminated or bool(terminated)
            self._sealed_video_generation = (
                self._video_generation
                if self._trusted_video_path_locked(self.video_path) is not None
                else None
            )
            self._update_artifact_warning_locked()

    # -- public snapshots --------------------------------------------------

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events[max(0, int(since)) :])

    def frame(self, kind: str) -> bytes | None:
        with self._lock:
            requested = str(kind)
            if self.environment == "behavior" and requested in {"agent", "camera"}:
                requested = "head"
            return self._frames_png.get(requested)

    def action_video_path(self, step: int) -> Path | None:
        with self._lock:
            for item in self._timeline:
                if int(item.get("step", -1)) != int(step):
                    continue
                path = self._action_video_path_locked(item)
                return path if path.exists() else None
        return None

    def video_path_for_generation(self, generation: int) -> Path | None:
        with self._lock:
            generation = int(generation)
            replay = self._sealed_episode_replays.get(generation)
            if replay is not None:
                return self._trusted_video_path_locked(
                    replay["path"], require_current_output=False
                )
            if generation != self._video_generation:
                return None
            if self._sealed_video_generation != self._video_generation:
                return None
            return self._trusted_video_path_locked(self.video_path)

    def has_video(self) -> bool:
        with self._lock:
            return (
                self._state == "done"
                and self._sealed_video_generation == self._video_generation
                and self._trusted_video_path_locked(self.video_path) is not None
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_publication_amendment_locked()
            return {
                "state": self._state,
                "terminated": self._terminated,
                "environment": self.environment,
                "usage": dict(self._usage),
                "has_video": self.has_video(),
                "video_generation": self._video_generation,
                "episode_replays": self._episode_replays_locked(),
                "frame_idx": self._frame_idx,
                "frame_indices": dict(self._frame_indices),
                "frame_revisions": dict(self._frame_revisions),
                "frame_kinds": list(self._frames_png),
                "capture_group_id": self._capture_group_id,
                "simulator_step": self._simulator_step,
                "last_selected_camera": self._last_selected_camera,
                "control": dict(self._control_snapshot),
                "n_steps": len(self._timeline),
                "timeline_revision": self._timeline_revision,
                "progress": dict(self._progress),
            }

    def run_info(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_publication_amendment_locked()
            return {
                "id": self.run_id,
                "name": self.name,
                "environment": self.environment,
                "suite": self.suite,
                "task": self.task,
                "seed": self.seed,
                "identity": dict(self.identity),
                "state": self._state,
                "n_steps": len(self._timeline),
                "timeline_revision": self._timeline_revision,
                "video_generation": self._video_generation,
                "progress": dict(self._progress),
            }

    def run_detail(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_publication_amendment_locked()
            timeline = [
                self._public_timeline_item_locked(item) for item in self._timeline
            ]
            return {
                "state": self._state,
                "terminated": self._terminated,
                "environment": self.environment,
                "suite": self.suite,
                "name": self.name,
                "task": self.task,
                "seed": self.seed,
                "identity": dict(self.identity),
                "usage": dict(self._usage),
                "timeline": timeline,
                "timeline_revision": self._timeline_revision,
                "has_video": self.has_video(),
                "video_generation": self._video_generation,
                "episode_replays": self._episode_replays_locked(),
                "frame_idx": self._frame_idx,
                "frame_indices": dict(self._frame_indices),
                "frame_revisions": dict(self._frame_revisions),
                "frame_kinds": list(self._frames_png),
                "capture_group_id": self._capture_group_id,
                "simulator_step": self._simulator_step,
                "last_selected_camera": self._last_selected_camera,
                "control": dict(self._control_snapshot),
                "metadata": dict(self._metadata),
                "progress": dict(self._progress),
            }

    # -- internals ---------------------------------------------------------

    def _trusted_video_path_locked(
        self,
        path: str | Path,
        *,
        require_current_output: bool = True,
    ) -> Path | None:
        candidate = Path(path)
        try:
            if candidate.is_symlink() or not candidate.is_file():
                return None
            resolved = candidate.resolve(strict=True)
            job_root = self._job_output_dir
            current_output = self.output_dir.resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        if not (resolved == job_root or job_root in resolved.parents):
            return None
        if require_current_output and not (
            resolved == current_output or current_output in resolved.parents
        ):
            return None
        return resolved

    def _episode_replays_locked(self) -> list[dict[str, Any]]:
        replays: list[dict[str, Any]] = []
        for generation in sorted(self._sealed_episode_replays):
            replay = self._sealed_episode_replays[generation]
            if (
                self._trusted_video_path_locked(
                    replay["path"], require_current_output=False
                )
                is None
            ):
                continue
            replays.append(
                {
                    "generation": int(replay["generation"]),
                    "attempt_index": int(replay["attempt_index"]),
                    "outcome": str(replay["outcome"]),
                }
            )
        return replays

    def _ignore_tool(self, name: str) -> bool:
        return (
            self.environment == "behavior"
            and bool(name)
            and name not in _BEHAVIOR_TOOLS
        )

    def _trusted_attempt_event(
        self, event: dict[str, Any], *, required_key: str
    ) -> bool:
        attempt = event.get("attempt_index")
        return (
            _is_int(attempt)
            and int(attempt) == int(self._progress["attempt_index"])
            and event.get(required_key) is True
        )

    def _running_index_locked(self, name: str) -> int | None:
        for index, item in enumerate(self._timeline):
            if item["action"] == name and item["status"] == "running":
                return index
        return None

    @staticmethod
    def _tool_failed(result: dict[str, Any]) -> bool:
        if result.get("primitive_success") is False or result.get("success") is False:
            return True
        if (
            result.get("error") not in (None, "", False)
            or result.get("capture_error") not in (None, "", False)
        ):
            return True
        return str(result.get("stop_reason") or "") in {
            "handoff_failed",
            "precondition_rejected",
            "isolation_failure",
            "tool_error",
        }

    @staticmethod
    def _env_step(result: dict[str, Any]) -> int | None:
        for key in ("env_step", "total_env_steps", "step"):
            value = result.get(key)
            if _is_int(value):
                return int(value)
        return None

    def _physical_camera(self, camera: Any, *, frame_id: Any) -> str | None:
        candidate = str(camera or "")
        if self.environment != "behavior":
            return candidate if candidate in self._frames_png else None
        if isinstance(frame_id, str):
            prefix = frame_id.split(":", 1)[0]
            if prefix in {"head", "left_wrist", "right_wrist"}:
                return prefix
        return _BEHAVIOR_CAMERA_ALIASES.get(candidate)

    def _ingest_frames(self, result: dict[str, Any]) -> None:
        env_step = self._env_step(result)
        inline = result.get("_frames_bytes")
        if isinstance(inline, dict) and set(inline) == {
            "head",
            "left_wrist",
            "right_wrist",
        }:
            self.on_frame_group(
                inline,
                capture_group_id=result.get("capture_group_id"),
                simulator_step=result.get("simulator_step", env_step),
            )
            return
        direct = (
            result.get("_image_bytes")
            if isinstance(result.get("_image_bytes"), bytes)
            else _read_image(result.get("overlay_path") or result.get("image_path"))
        )
        if isinstance(direct, bytes):
            camera = self._physical_camera(
                result.get("resolved_camera") or result.get("camera") or "agent",
                frame_id=result.get("frame_id"),
            )
            if camera:
                self.on_frame(camera, direct, env_step=env_step)

        camera_image = (
            result.get("_image_cam_bytes")
            if isinstance(result.get("_image_cam_bytes"), bytes)
            else _read_image(result.get("image_cam_path"))
        )
        if isinstance(camera_image, bytes) and self.environment == "libero":
            self.on_frame("camera", camera_image, env_step=env_step)

        if isinstance(inline, dict):
            for camera, image in inline.items():
                if isinstance(image, bytes):
                    physical = self._physical_camera(camera, frame_id=None)
                    if physical:
                        self.on_frame(physical, image, env_step=env_step)

        containers: list[dict[str, Any]] = []
        for key in ("views", "images"):
            value = result.get(key)
            if isinstance(value, dict):
                containers.append(value)
        visual_review = result.get("visual_review")
        if isinstance(visual_review, dict):
            for key in ("views", "images"):
                value = visual_review.get(key)
                if isinstance(value, dict):
                    containers.append(value)
        for views in containers:
            for camera, view in views.items():
                if not isinstance(view, dict):
                    continue
                image = (
                    view.get("_image_bytes")
                    if isinstance(view.get("_image_bytes"), bytes)
                    else _read_image(
                        view.get("rgb_path")
                        or view.get("path")
                        or view.get("image_path")
                    )
                )
                if not isinstance(image, bytes):
                    continue
                physical = self._physical_camera(
                    camera,
                    frame_id=view.get("frame_id"),
                )
                if physical:
                    self.on_frame(physical, image, env_step=env_step)

    @staticmethod
    def _manual_requested_step(target: str, action: str) -> dict[str, Any]:
        if action == "observe":
            return {"kind": "camera_refresh"}
        if target == "chassis":
            if action in {"forward", "backward"}:
                return {
                    "kind": "translation",
                    "meters": 0.05 if action == "forward" else -0.05,
                }
            if action in {"turn_left", "turn_right"}:
                return {
                    "kind": "rotation",
                    "degrees": 5.0 if action == "turn_left" else -5.0,
                }
            if action in {"up", "down"}:
                return {
                    "kind": "torso_vertical",
                    "meters": 0.03 if action == "up" else -0.03,
                }
        if action in {
            "forward",
            "backward",
            "turn_left",
            "turn_right",
            "up",
            "down",
        }:
            signs = {
                "forward": (0.03, 0.0, 0.0),
                "backward": (-0.03, 0.0, 0.0),
                "turn_left": (0.0, 0.03, 0.0),
                "turn_right": (0.0, -0.03, 0.0),
                "up": (0.0, 0.0, 0.03),
                "down": (0.0, 0.0, -0.03),
            }
            return {"kind": "eef_translation", "delta_m": list(signs[action])}
        if action in {"rotate_left", "rotate_right"}:
            return {
                "kind": "wrist_rotation",
                "visual_degrees": 5.0 if action == "rotate_left" else -5.0,
            }
        if action in {"open", "close"}:
            return {"kind": "gripper", "opening": 1.0 if action == "open" else 0.0}
        return {"kind": "unknown"}

    def _update_progress_from_payload_locked(self, payload: dict[str, Any]) -> None:
        fields = {
            "attempt_index": "attempt_index",
            "total_env_steps": "total_env_steps",
            "global_tool_calls": "global_tool_calls",
            "global_vla_chunks": "global_vla_chunks",
            "global_vla_invocations": "global_vla_invocations",
            "global_elapsed_wall_clock_s": "elapsed_wall_clock_s",
        }
        for source, target in fields.items():
            value = payload.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._progress[target] = value

    def _action_video_path_locked(self, item: dict[str, Any]) -> Path:
        return (
            self.output_dir
            / "action_videos"
            / f"step_{int(item['step']):02d}_{item['action']}.mp4"
        )

    def _public_timeline_item_locked(self, item: dict[str, Any]) -> dict[str, Any]:
        public = {key: value for key, value in item.items() if not key.startswith("_")}
        if item["status"] == "running":
            public["elapsed_s"] = max(0.0, time.monotonic() - item["_started_at"])
        return public

    def _update_artifact_warning_locked(self) -> None:
        self._progress["artifact_seal_warning"] = bool(
            self._progress["official_task_success"]
            and not self._progress["workflow_complete"]
        )

    def _refresh_publication_amendment_locked(self) -> None:
        if self.environment != "behavior":
            return
        amendment_path = self._job_root_locked() / "publication_amendment.json"
        try:
            amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(amendment, dict):
            return
        task_name = self._metadata.get("task-name")
        public_seed = self._metadata.get("public-seed", self.seed)
        expected_tag = (
            f"{task_name}_s{public_seed}" if isinstance(task_name, str) else self.name
        )
        if (
            amendment.get("job_id") != self._metadata.get("job-id")
            or amendment.get("tag") != expected_tag
            or amendment.get("public_seed") != public_seed
            or amendment.get("attempt_index") != self._progress["attempt_index"]
            or amendment.get("success_source") != 'info["done"]["success"]'
            or amendment.get("task_success") is not True
            or amendment.get("publication_complete") is not True
        ):
            return
        recipe = self._job_root_locked() / f"recipe_{expected_tag}.jsonl"
        memory = self._job_root_locked() / "memory" / f"{task_name}.md"
        provenance = self._job_root_locked() / "memory" / f"{task_name}_provenance.json"
        expected = {
            recipe: amendment.get("recipe_sha256"),
            memory: amendment.get("memory_sha256"),
            provenance: amendment.get("provenance_sha256"),
        }
        for path, digest in expected.items():
            if not isinstance(digest, str):
                return
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                return
            if actual != digest:
                return
        attempt_dir = (
            self._job_root_locked()
            / "attempts"
            / expected_tag
            / f"attempt_{int(self._progress['attempt_index']):03d}"
        )
        receipt_path = attempt_dir / "official_success_receipt.json"
        try:
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(receipt, dict) or not isinstance(provenance_payload, dict):
            return
        receipt_digest = receipt.get("receipt_sha256")
        unsigned_receipt = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        canonical_receipt = json.dumps(
            unsigned_receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        official_binding = provenance_payload.get("official_success_receipt")
        if (
            receipt.get("source") != 'info["done"]["success"]'
            or not isinstance(receipt.get("raw_done"), dict)
            or receipt["raw_done"].get("success") is not True
            or receipt_digest != hashlib.sha256(canonical_receipt).hexdigest()
            or provenance_payload.get("source") != "raw_official_success_v1"
            or provenance_payload.get("success_source") != 'info["done"]["success"]'
            or provenance_payload.get("source_tag") != expected_tag
            or provenance_payload.get("job_id") != self._metadata.get("job-id")
            or provenance_payload.get("attempt_index")
            != self._progress["attempt_index"]
            or provenance_payload.get("task_success") is not True
            or not isinstance(official_binding, dict)
            or official_binding.get("receipt_sha256") != receipt_digest
            or official_binding.get("file_sha256")
            != hashlib.sha256(receipt_bytes).hexdigest()
        ):
            return
        self._progress["official_task_success"] = True
        self._progress["publication_complete"] = True
        self._progress["artifact_seal_complete"] = bool(
            amendment.get("artifact_seal_complete")
        )
        self._terminated = True
        self._update_artifact_warning_locked()

    def _job_root_locked(self) -> Path:
        job_id = self._metadata.get("job-id")
        if isinstance(job_id, str) and self.run_id.endswith(f"/{job_id}"):
            # Serial Explore points output_dir at the current attempt.
            for candidate in (self.output_dir, *self.output_dir.parents):
                if (candidate / "session_manifest.json").exists():
                    return candidate
        return self.output_dir


__all__ = ["State"]
