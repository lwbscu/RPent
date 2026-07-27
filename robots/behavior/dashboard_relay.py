"""Incrementally relay one isolated child dashboard sink into a live State."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from robots.behavior.dashboard_sink import strip_dashboard_frame_sources

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UNTRUSTED_LIFECYCLE_EVENTS = frozenset(
    {"official_success", "workflow_complete", "publication_complete"}
)


class DashboardEventRelay:
    """Tail a :class:`FileDashboardSink` without trusting child capabilities.

    The relay reads appended bytes once, retains a partial final line, and
    validates every frame using path containment, symlink rejection, and its
    claimed SHA-256. Child lifecycle events are deliberately ignored: only the
    campaign controller may publish official success or artifact state after it
    validates the run artifacts.
    """

    def __init__(
        self,
        event_path: str | os.PathLike[str],
        dashboard: Any,
        *,
        allowed_tools: Iterable[str] | None = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.event_path = Path(event_path)
        self.root = self.event_path.parent
        self.dashboard = dashboard
        self.allowed_tools = (
            frozenset(str(name) for name in allowed_tools)
            if allowed_tools is not None
            else None
        )
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self._offset = 0
        self._partial = b""
        self._file_identity: tuple[int, int] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"dashboard-file-relay-{self.event_path.parent.name}",
            daemon=True,
        )
        self._violations: list[str] = []
        self._records_relayed = 0

    @property
    def violations(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._violations)

    @property
    def records_relayed(self) -> int:
        with self._lock:
            return self._records_relayed

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, timeout_s: float | None = None) -> None:
        self._stop.set()
        deadline = (
            None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        )
        join_timeout = (
            max(5.0, self.poll_interval_s * 4)
            if timeout_s is None
            else max(0.0, deadline - time.monotonic())
        )
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            self._violation("dashboard event relay did not stop before deadline")
        self.drain(deadline_monotonic=deadline)
        if self._partial.strip():
            self._violation("dashboard event sink ends with a partial record")

    def drain(self, *, deadline_monotonic: float | None = None) -> None:
        """Relay all currently complete records."""

        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            self._violation("dashboard event drain deadline exhausted")
            return
        try:
            descriptor = _open_regular_nonblocking_no_follow(self.event_path)
            event_stat = os.fstat(descriptor)
            file_size = event_stat.st_size
        except FileNotFoundError:
            return
        except OSError as error:
            self._violation(f"dashboard event sink is unsafe: {error}")
            return
        try:
            identity = (event_stat.st_dev, event_stat.st_ino)
            if self._file_identity is None:
                self._file_identity = identity
            elif self._file_identity != identity:
                self._violation("dashboard event sink identity changed")
                return
            if file_size < self._offset:
                self._violation("dashboard event sink was truncated")
                return
            os.lseek(descriptor, self._offset, os.SEEK_SET)
            chunks: list[bytes] = []
            while True:
                if (
                    deadline_monotonic is not None
                    and time.monotonic() >= deadline_monotonic
                ):
                    self._violation("dashboard event drain deadline exhausted")
                    return
                try:
                    part = os.read(descriptor, 1024 * 1024)
                except BlockingIOError:
                    break
                if not part:
                    break
                chunks.append(part)
            chunk = b"".join(chunks)
            self._offset += len(chunk)
        finally:
            os.close(descriptor)
        if not chunk:
            return
        payload = self._partial + chunk
        lines = payload.split(b"\n")
        self._partial = lines.pop()
        for raw_line in lines:
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._violation("dashboard event sink contains invalid JSON")
                continue
            if not isinstance(record, dict):
                self._violation("dashboard event sink record is not an object")
                continue
            self._relay_record(record)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            self.drain()

    def _relay_record(self, record: dict[str, Any]) -> None:
        channel = record.get("channel")
        payload = record.get("payload")
        if not isinstance(channel, str) or not isinstance(payload, dict):
            self._violation("dashboard event sink record has invalid channel payload")
            return
        if channel == "event":
            event_type = str(payload.get("type") or "")
            if event_type in {
                "tool_call",
                "tool_result",
                "tool_progress",
                "tool_use",
            } and not self._tool_allowed(
                str(payload.get("tool") or payload.get("name") or "")
            ):
                return
            if event_type not in _UNTRUSTED_LIFECYCLE_EVENTS:
                self.dashboard.on_event(payload)
        elif channel == "usage":
            self.dashboard.on_usage(
                inp=_public_int(payload.get("inp")),
                out=_public_int(payload.get("out")),
                tool_calls=_public_int(payload.get("tool_calls")),
            )
        elif channel in {"tool_start", "tool_progress", "tool_result"}:
            name = str(payload.get("name") or "")
            if not self._tool_allowed(name):
                return
            if channel == "tool_start":
                arguments = payload.get("arguments")
                self.dashboard.on_tool_start(
                    name,
                    arguments if isinstance(arguments, dict) else {},
                )
            else:
                result = payload.get("result")
                safe = strip_dashboard_frame_sources(
                    result if isinstance(result, dict) else {}
                )
                callback = (
                    self.dashboard.on_tool_progress
                    if channel == "tool_progress"
                    else self.dashboard.on_tool_result
                )
                callback(name, safe)
        elif channel == "frame":
            self._relay_frame(payload)
        elif channel == "metadata":
            self.dashboard.set_metadata(payload)
        elif channel in {"attempt_begin", "attempt_end", "done", "job_progress"}:
            # The paired parent owns attempt/result lifecycle. It intentionally
            # does not trust child copies of these state transitions.
            pass
        else:
            self._violation(f"dashboard event sink channel is not allowed: {channel}")
            return
        with self._lock:
            self._records_relayed += 1

    def _tool_allowed(self, name: str) -> bool:
        if not name:
            self._violation("dashboard event sink tool name is empty")
            return False
        if self.allowed_tools is not None and name not in self.allowed_tools:
            self._violation(f"dashboard event sink tool is not allowed: {name}")
            return False
        return True

    def _relay_frame(self, payload: dict[str, Any]) -> None:
        callback = getattr(self.dashboard, "on_frame", None)
        relative_value = payload.get("relative_path")
        claimed_sha256 = payload.get("sha256")
        if (
            not callable(callback)
            or not isinstance(relative_value, str)
            or not relative_value
            or not isinstance(claimed_sha256, str)
            or _SHA256_RE.fullmatch(claimed_sha256) is None
        ):
            self._violation("dashboard frame receipt is invalid")
            return
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            self._violation("dashboard frame path escapes its attempt")
            return
        try:
            root = self.root.resolve(strict=True)
            cursor = self.root
            for part in relative.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError("dashboard frame path contains a symlink")
            resolved = (self.root / relative).resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                raise ValueError("dashboard frame is not a file")
            image = resolved.read_bytes()
        except (OSError, ValueError) as error:
            self._violation(f"dashboard frame is not contained: {error}")
            return
        if hashlib.sha256(image).hexdigest() != claimed_sha256:
            self._violation("dashboard frame SHA-256 mismatch")
            return
        callback(
            str(payload.get("camera") or "head"),
            image,
            env_step=payload.get("env_step"),
        )

    def _violation(self, message: str) -> None:
        with self._lock:
            if message not in self._violations:
                self._violations.append(message)


def _public_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _open_regular_nonblocking_no_follow(path: Path) -> int:
    absolute = path.absolute()
    parts = absolute.parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    directory_fd = os.open(parts[0], directory_flags)
    try:
        for part in parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | os.O_NONBLOCK
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    finally:
        os.close(directory_fd)
    try:
        file_stat = os.fstat(descriptor)
    except OSError:
        os.close(descriptor)
        raise
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(descriptor)
        raise OSError(f"dashboard event sink is not regular: {path}")
    return descriptor


__all__ = ["DashboardEventRelay"]
