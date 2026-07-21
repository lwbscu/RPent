"""Atomic, redacted lifecycle manifest for a BEHAVIOR runtime session."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit

from robots.behavior.schemas import VLA_CONTROL_MODES
from rpent.utils.redaction import redact_command as _redact_command
from rpent.utils.redaction import redact_text as _redact_text

MANIFEST_FILENAME = "run_manifest.json"


def utc_timestamp() -> str:
    """Return a stable UTC timestamp suitable for machine-readable artifacts."""

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def redact_text(value: str) -> str:
    """Redact common credential assignments and URL userinfo from text."""

    return _redact_text(value)


def redact_command(command: Iterable[object] | str | None) -> list[str] | None:
    """Return an argv-shaped command with credential-bearing values removed."""

    return _redact_command(command)


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _source_identity(repo_root: Path) -> dict[str, Any]:
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    worktree = _git_value(repo_root, "rev-parse", "--show-toplevel")
    status = _git_value(repo_root, "status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": commit,
        "worktree": worktree or str(repo_root.resolve()),
        "worktree_dirty": status is not None,
    }


def _port_from_url(endpoint: str | None) -> int | None:
    if not endpoint:
        return None
    try:
        return urlsplit(endpoint).port
    except ValueError:
        return None


def process_identity(proc: Any) -> tuple[int | None, int | None]:
    """Return PID/PGID only when they can be safely identified."""

    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return None, None
    try:
        pgid = os.getpgid(pid)
    except (OSError, TypeError, ValueError):
        pgid = None
    return pid, pgid


def _proc_identity(pid: int | None) -> tuple[int | None, int | None]:
    """Return session and start ticks for a currently live Linux process."""

    if not isinstance(pid, int) or pid <= 0:
        return None, None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        sid = int(fields[3])
        start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError):
        return None, None
    return sid, start_ticks


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass
class RunManifest:
    """In-memory manifest whose every transition is atomically persisted."""

    path: Path
    data: dict[str, Any]
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def start(
        cls,
        output_dir: str | Path,
        args: Any,
        *,
        repo_root: str | Path,
    ) -> "RunManifest":
        output_dir = Path(output_dir)
        mode = str(args.behavior_control_mode)
        started_at = utc_timestamp()
        managed_env = not bool(args.no_driver)
        env_process: dict[str, Any] = {
            "managed": managed_env,
            "pid": None,
            "pgid": None,
            "sid": None,
            "start_ticks": None,
            "host": None if managed_env else str(args.env_endpoint),
            "port": None if managed_env else int(args.env_port),
            "command": None,
            "started_at": None,
            "stopped_at": None,
            "returncode": None,
        }
        data: dict[str, Any] = {
            "schema_version": 1,
            **_source_identity(Path(repo_root)),
            "control_mode": mode,
            "stage3_press_enabled": bool(getattr(args, "behavior_stage3_press", False)),
            "task": {
                "suite": args.suite,
                "task": int(args.task),
                "task_name": args.task_name,
                "activity_definition_id": int(args.activity_definition_id),
                "activity_instance_id": int(args.activity_instance_id),
                "activity_instance_dir": str(
                    Path(args.activity_instance_dir).expanduser().resolve()
                ),
                "scene_model": args.scene_model,
                "seed": int(args.seed),
                "max_episode_steps": int(args.max_episode_steps),
            },
            "gpu": None if args.cuda_device is None else str(args.cuda_device),
            "status": "starting",
            "started_at": started_at,
            "stopped_at": None,
            "processes": {"env": env_process},
        }
        if mode in VLA_CONTROL_MODES:
            managed_vla = not bool(args.no_driver) and not bool(args.vla_endpoint)
            data["checkpoint"] = (
                str(Path(args.policy_checkpoint).expanduser().resolve())
                if args.policy_checkpoint
                else None
            )
            data["processes"]["vla"] = {
                "managed": managed_vla,
                "pid": None,
                "pgid": None,
                "sid": None,
                "start_ticks": None,
                "host": None if managed_vla else args.vla_endpoint,
                "port": None if managed_vla else _port_from_url(args.vla_endpoint),
                "command": None,
                "started_at": None,
                "stopped_at": None,
                "returncode": None,
            }
        manifest = cls(path=output_dir / MANIFEST_FILENAME, data=data)
        manifest._write()
        return manifest

    def _write(self) -> None:
        with self._lock:
            _atomic_write_json(self.path, self.data)

    def _update(self, update: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            update(self.data)
            _atomic_write_json(self.path, self.data)

    def process_started(
        self,
        name: str,
        proc: Any,
        *,
        command: Iterable[object] | str,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        pid, pgid = process_identity(proc)
        sid, start_ticks = _proc_identity(pid)

        def update(data: dict[str, Any]) -> None:
            process = data["processes"][name]
            process.update(
                {
                    "managed": True,
                    "pid": pid,
                    "pgid": pgid,
                    "sid": sid,
                    "start_ticks": start_ticks,
                    "host": host,
                    "port": port,
                    "command": redact_command(command),
                    "started_at": utc_timestamp(),
                    "stopped_at": None,
                    "returncode": None,
                }
            )

        self._update(update)

    def process_endpoint(self, name: str, *, host: str, port: int) -> None:
        self._update(
            lambda data: data["processes"][name].update(
                {"host": str(host), "port": int(port)}
            )
        )

    def process_stopped(self, name: str, proc: Any) -> None:
        returncode = getattr(proc, "returncode", None)
        if returncode is None:
            poll = getattr(proc, "poll", None)
            if callable(poll):
                try:
                    returncode = poll()
                except Exception:
                    returncode = None

        def update(data: dict[str, Any]) -> None:
            process = data.get("processes", {}).get(name)
            if (
                process is None
                or not process.get("managed")
                or process.get("started_at") is None
            ):
                return
            process["returncode"] = returncode
            process["stopped_at"] = utc_timestamp()

        self._update(update)

    def running(self) -> None:
        self._update(lambda data: data.update({"status": "running"}))

    def stopping(self) -> None:
        self._update(lambda data: data.update({"status": "stopping"}))

    def finish(self, *, status: str, error: BaseException | None = None) -> None:
        if status not in {"stopped", "failed"}:
            raise ValueError(f"invalid final manifest status: {status}")

        def update(data: dict[str, Any]) -> None:
            data["status"] = status
            data["stopped_at"] = utc_timestamp()
            if error is not None:
                data["error"] = {
                    "type": type(error).__name__,
                    "message": redact_text(str(error)),
                }

        self._update(update)


__all__ = [
    "MANIFEST_FILENAME",
    "RunManifest",
    "process_identity",
    "redact_command",
    "redact_text",
    "utc_timestamp",
]
