"""Append-only dashboard event sink for isolated child processes."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

_BEHAVIOR_CAMERA_ALIASES = {
    "main": "head",
    "agent": "head",
    "head": "head",
    "left": "left_wrist",
    "left_wrist": "left_wrist",
    "right": "right_wrist",
    "right_wrist": "right_wrist",
}


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe dashboard value without private capabilities."""

    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


_PATH_KEYS = frozenset(
    {
        "path",
        "rgb_path",
        "image_path",
        "image_cam_path",
        "overlay_path",
    }
)


def _wire_safe(value: Any) -> Any:
    """Sanitize payloads and remove all filesystem path capabilities."""

    safe = _json_safe(value)
    if isinstance(safe, dict):
        return {
            key: _wire_safe(item)
            for key, item in safe.items()
            if key.lower() not in _PATH_KEYS
        }
    if isinstance(safe, list):
        return [_wire_safe(item) for item in safe]
    return safe


def strip_dashboard_frame_sources(value: Any) -> Any:
    """Remove binary/path frame capabilities while retaining public metadata.

    Parent relays use a separate, hash-bound ``frame`` channel for image bytes.
    This helper makes it safe to forward the corresponding tool payload
    without re-reading an untrusted path in the dashboard process.
    """

    return _wire_safe(value)


class FileDashboardSink:
    """Persist JSONL events and hash-bound frame files beside one attempt."""

    def __init__(self, event_path: str | Path) -> None:
        self.event_path = Path(event_path)
        self.root = self.event_path.parent
        self.frame_dir = self.root / "dashboard_frames"
        self._lock = threading.Lock()
        self._frame_index = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def on_event(self, event: dict[str, Any]) -> None:
        self._append("event", _wire_safe(event))

    def on_usage(self, *, inp: int, out: int, tool_calls: int) -> None:
        self._append(
            "usage",
            {"inp": int(inp), "out": int(out), "tool_calls": int(tool_calls)},
        )

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self._append(
            "tool_start",
            {"name": str(name), "arguments": _wire_safe(arguments)},
        )

    def on_tool_progress(self, name: str, result: dict[str, Any]) -> None:
        self._persist_frames(result)
        self._append(
            "tool_progress",
            {"name": str(name), "result": _wire_safe(result)},
        )

    def on_tool_result(self, name: str, result: Any) -> None:
        if not isinstance(result, dict):
            result = {"result": result}
        self._persist_frames(result)
        self._append(
            "tool_result",
            {"name": str(name), "result": _wire_safe(result)},
        )

    def set_metadata(self, metadata: dict[str, Any]) -> None:
        self._append("metadata", _wire_safe(metadata))

    def on_frame(self, kind: str, image: bytes, *, env_step: Any = None) -> None:
        if isinstance(image, bytes):
            self._store_frame(str(kind), image, env_step=env_step)

    def on_job_progress(self, progress: dict[str, Any]) -> None:
        self._append("job_progress", _wire_safe(progress))

    def begin_attempt(self, **payload: Any) -> None:
        self._append("attempt_begin", _wire_safe(payload))

    def end_attempt(self, **payload: Any) -> None:
        self._append("attempt_end", _wire_safe(payload))

    def mark_done(self, terminated: bool | None = None) -> None:
        self._append("done", {"terminated": terminated})

    def _append(self, channel: str, payload: Any) -> None:
        record = {"channel": str(channel), "payload": payload}
        encoded = json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=True,
        )
        with self._lock:
            descriptor = os.open(
                self.event_path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o600,
            )
            try:
                os.write(descriptor, (encoded + "\n").encode("utf-8"))
            finally:
                os.close(descriptor)

    def _persist_frames(self, payload: dict[str, Any]) -> None:
        env_step = _env_step(payload)
        direct = payload.get("_image_bytes")
        if isinstance(direct, bytes):
            camera = _physical_camera(
                payload.get("resolved_camera") or payload.get("camera") or "head",
                payload.get("frame_id"),
            )
            self._store_frame(camera, direct, env_step=env_step)

        inline = payload.get("_frames_bytes")
        if isinstance(inline, Mapping):
            for camera, image in inline.items():
                if isinstance(image, bytes):
                    self._store_frame(
                        _physical_camera(camera, None),
                        image,
                        env_step=env_step,
                    )

        containers: list[Mapping[str, Any]] = []
        for key in ("views", "images"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
        review = payload.get("visual_review")
        if isinstance(review, Mapping):
            for key in ("views", "images"):
                value = review.get(key)
                if isinstance(value, Mapping):
                    containers.append(value)

        for views in containers:
            for camera, view in views.items():
                if not isinstance(view, Mapping):
                    continue
                image = view.get("_image_bytes")
                if not isinstance(image, bytes):
                    image = self._read_contained_image(
                        view.get("rgb_path")
                        or view.get("path")
                        or view.get("image_path")
                    )
                if isinstance(image, bytes):
                    self._store_frame(
                        _physical_camera(camera, view.get("frame_id")),
                        image,
                        env_step=env_step,
                    )

    def _read_contained_image(self, value: Any) -> bytes | None:
        if not isinstance(value, (str, os.PathLike)):
            return None
        candidate = Path(value)
        try:
            root = self.root.resolve(strict=True)
            if candidate.is_absolute():
                resolved = candidate.resolve(strict=True)
            else:
                resolved = (self.root / candidate).resolve(strict=True)
            resolved.relative_to(root)
            cursor = self.root
            for part in resolved.relative_to(root).parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    return None
            return resolved.read_bytes()
        except (OSError, ValueError):
            return None

    def _store_frame(self, camera: str, image: bytes, *, env_step: Any) -> None:
        if camera not in {"head", "left_wrist", "right_wrist", "agent", "camera"}:
            return
        digest = hashlib.sha256(image).hexdigest()
        with self._lock:
            self._frame_index += 1
            frame_index = self._frame_index
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{frame_index:06d}_{camera}_{digest[:12]}.png"
            path = self.frame_dir / filename
            if not path.exists():
                descriptor = os.open(
                    path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                try:
                    os.write(descriptor, image)
                finally:
                    os.close(descriptor)
        self._append(
            "frame",
            {
                "camera": camera,
                "env_step": int(env_step) if _is_int(env_step) else None,
                "relative_path": path.relative_to(self.root).as_posix(),
                "sha256": digest,
            },
        )


def _physical_camera(camera: Any, frame_id: Any) -> str:
    if isinstance(frame_id, str):
        prefix = frame_id.split(":", 1)[0]
        if prefix in {"head", "left_wrist", "right_wrist"}:
            return prefix
    raw = str(camera or "head")
    return _BEHAVIOR_CAMERA_ALIASES.get(raw, raw)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _env_step(payload: Mapping[str, Any]) -> int | None:
    for key in ("env_step", "total_env_steps", "step"):
        value = payload.get(key)
        if _is_int(value):
            return int(value)
    return None


__all__ = ["FileDashboardSink", "strip_dashboard_frame_sources"]
