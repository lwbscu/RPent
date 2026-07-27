"""Atomic, redacted lifecycle manifest for a BEHAVIOR runtime session."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import urlsplit

from robots.behavior.redaction import redact_command as _redact_command
from robots.behavior.redaction import redact_text as _redact_text
from robots.behavior.schemas import (
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    PUBLIC_TOOL_CONTRACTS,
)

MANIFEST_FILENAME = "run_manifest.json"
LEGACY_RUN_MANIFEST_SCHEMA_VERSION = 5
RUN_MANIFEST_SCHEMA_VERSION = 6
PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION = 5


def resolve_run_manifest_public_tool_contract(
    manifest: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Resolve a run manifest's public-tool ABI without ambiguous inference.

    Historical schema-5 manifests predate the explicit version field and are
    therefore accepted only as the exact v1 nine-tool surface.  New schema-6
    manifests must declare the current version explicitly and match its exact,
    ordered tool list.
    """

    schema_version = manifest.get("schema_version")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("run manifest protocol is missing")
    declared_version = protocol.get("public_tool_contract_version")
    declared_tools = protocol.get("public_primitives")

    if schema_version == LEGACY_RUN_MANIFEST_SCHEMA_VERSION:
        if declared_version is not None:
            raise ValueError(
                "legacy run manifest must not declare a public-tool contract version"
            )
        version = 1
    elif schema_version == RUN_MANIFEST_SCHEMA_VERSION:
        if (
            isinstance(declared_version, bool)
            or not isinstance(declared_version, int)
            or declared_version != CURRENT_PUBLIC_TOOL_CONTRACT_VERSION
        ):
            raise ValueError(
                "current run manifest must declare the current public-tool "
                "contract version"
            )
        version = declared_version
    else:
        raise ValueError(f"unsupported run manifest schema: {schema_version!r}")

    expected_tools = PUBLIC_TOOL_CONTRACTS[version]
    if tuple(declared_tools or ()) != expected_tools:
        raise ValueError(
            f"run manifest public primitives do not match contract v{version}"
        )
    return version, expected_tools


def pi0_nav_pick_exact_chunk_contract() -> dict[str, Any]:
    """Return the public ABI for one bounded Pi0 invocation."""

    return {
        "call_artifact_schema_version": PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION,
        "chunks_argument": {
            "name": "chunks",
            "required": True,
            "minimum": 1,
            "maximum": None,
        },
        "action_shape": [32, 23],
        "normal_completion": "exact_requested_chunks",
        "raw_success_behavior": "stop_after_success_env_step",
        "official_success_completion": {
            "task_success": True,
            "primitive_success": True,
            "stop_reason": "official_task_success",
            "exact_requested_chunks_completed": (
                "false_unless_success_step_completes_exact_request"
            ),
            "post_success_env_actions": 0,
        },
        "partial_chunk_exceptions": [
            "official_task_success",
            "terminated",
            "truncated",
            "safety_failure",
            "infrastructure_failure",
        ],
    }


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
        phase = str(args.behavior_phase)
        candidate_instance_id = getattr(args, "behavior_candidate_instance_id", None)
        max_attempts = (
            1 if candidate_instance_id is not None or phase != "explore" else None
        )
        attempt_index = int(getattr(args, "behavior_attempt_index", 1) or 1)
        job_id = getattr(args, "behavior_job_id", None)
        task_identity = {
            "task_name": args.task_name,
            "activity_definition_id": int(args.activity_definition_id),
            "activity_instance_id": int(args.activity_instance_id),
        }
        source_snapshot_binding = getattr(
            args, "_behavior_source_snapshot_binding", None
        )
        source_snapshot_as_dict = getattr(source_snapshot_binding, "as_dict", None)
        data: dict[str, Any] = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            **_source_identity(Path(repo_root)),
            "source_snapshot": (
                source_snapshot_as_dict() if callable(source_snapshot_as_dict) else None
            ),
            "protocol": {
                "behavior_phase": phase,
                "public_seed": int(args.public_seed),
                "recipe_tag": f"{args.task_name}_s{int(args.public_seed)}",
                "public_tool_contract_version": (CURRENT_PUBLIC_TOOL_CONTRACT_VERSION),
                "public_primitives": list(
                    PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
                ),
                "agent_finish_registered": False,
                "mapping_version": str(args._behavior_mapping_version),
                "task_spec": args._behavior_task_spec_binding,
                "prompt": args._behavior_prompt_binding,
                "task_identity": task_identity,
                "instance_classification": getattr(
                    args, "_behavior_instance_classification", None
                ),
                **(
                    {"campaign_kind": "candidate_instance_explore"}
                    if candidate_instance_id is not None
                    else {}
                ),
                "attempts": {
                    "initial_attempt_index": attempt_index,
                    "max_attempts": max_attempts,
                    "reset_registered": False,
                },
                "pi0_nav_pick_contract": pi0_nav_pick_exact_chunk_contract(),
            },
            "job": (
                {
                    "job_id": str(job_id),
                    "attempt_index": attempt_index,
                    "vla_binding_sha256": hashlib.sha256(
                        str(args.behavior_vla_binding_id).encode("utf-8")
                    ).hexdigest(),
                    "prior_attempt_summaries": getattr(
                        args, "_behavior_prior_attempt_summaries_input", None
                    ),
                    "child_dashboard_disabled": not bool(
                        getattr(args, "dashboard", False)
                    ),
                    **(
                        {
                            "candidate_campaign_id": str(
                                args.behavior_candidate_campaign_id
                            ),
                            "candidate_state_sha256": str(
                                args.behavior_candidate_state_sha256
                            ),
                        }
                        if candidate_instance_id is not None
                        else {}
                    ),
                }
                if job_id is not None
                else None
            ),
            "task": {
                "suite": args.suite,
                "task": int(args.task),
                "task_name": args.task_name,
                "task_language": None,
                "public_seed": int(args.public_seed),
                "max_episode_steps": int(args.max_episode_steps),
            },
            "native_binding": {
                "activity_definition_id": int(args.activity_definition_id),
                "activity_instance_id": int(args.activity_instance_id),
                "activity_instance_dir": str(
                    Path(args.activity_instance_dir).expanduser().resolve()
                ),
                "scene_model": args.scene_model,
                "env_seed": int(args.seed),
                **(
                    {"state_sha256": str(args.behavior_candidate_state_sha256)}
                    if candidate_instance_id is not None
                    else {}
                ),
            },
            # Retained at top level for compatibility with existing artifact
            # readers; protocol.task_identity is the authoritative binding.
            "task_identity": task_identity,
            "budgets": {
                "max_episode_steps": int(args.max_episode_steps),
                "max_tool_calls": int(args.max_tool_calls),
                "max_wall_clock_s": int(args.max_wall_clock_s),
                "max_attempts": max_attempts,
            },
            "planner": {
                "backend": getattr(args, "planner", None),
                "model": getattr(args, "model", None),
                "reasoning_effort": getattr(args, "reasoning_effort", None),
            },
            "dashboard": {
                "enabled": bool(getattr(args, "dashboard", False)),
                "auto_start": bool(getattr(args, "dashboard_auto_start", False)),
            },
            "frozen_eval_inputs": getattr(args, "_behavior_frozen_inputs", None),
            "reviewed_repo_memory": getattr(args, "_behavior_repo_memory_input", None),
            "reviewed_recipe_catalog": getattr(
                args, "_behavior_recipe_catalog_input", None
            ),
            "resource_source": getattr(args, "_behavior_resource_source", None),
            "policy_checkpoint": args._behavior_policy_checkpoint_binding,
            "gpu": None if args.cuda_device is None else str(args.cuda_device),
            "status": "starting",
            "started_at": started_at,
            "stopped_at": None,
            "processes": {"env": env_process},
        }
        managed_vla = not bool(args.no_driver) and not bool(args.vla_endpoint)
        data["checkpoint"] = args._behavior_policy_checkpoint_binding["resolved_path"]
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

    def set_task_language(self, task_language: str) -> None:
        """Record the environment-authoritative natural-language instruction."""

        value = str(task_language).strip()
        if not value:
            raise ValueError("task_language must be non-empty")

        def update(data: dict[str, Any]) -> None:
            data["task"]["task_language"] = value

        self._update(update)

    def _write(self) -> None:
        with self._lock:
            _atomic_write_json(self.path, self.data)
            _atomic_write_json(self.path.parent / "session_manifest.json", self.data)

    def _update(self, update: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            update(self.data)
            _atomic_write_json(self.path, self.data)
            _atomic_write_json(self.path.parent / "session_manifest.json", self.data)

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
    "LEGACY_RUN_MANIFEST_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "RUN_MANIFEST_SCHEMA_VERSION",
    "RunManifest",
    "process_identity",
    "redact_command",
    "redact_text",
    "resolve_run_manifest_public_tool_contract",
    "utc_timestamp",
]
