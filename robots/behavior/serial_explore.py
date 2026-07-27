"""Job-level fresh-environment Explore harness for BEHAVIOR.

One job owns the VLA process and Dashboard.  Every attempt is a fresh
``robots.behavior.cli`` subprocess, Codex invocation, environment process, and
episode.  There is deliberately no Agent-facing reset and no job-wide budget.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from robots.behavior.dashboard_sink import strip_dashboard_frame_sources
from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    prepare_local_dataset_resources,
    prepare_pinned_dataset_resources,
    verify_pinned_dataset_resources,
    write_dataset_resource_binding,
)
from robots.behavior.memory_snapshot import load_behavior_memory_snapshot
from robots.behavior.policy_checkpoint import SHARED_POLICY_CHECKPOINT_PATH
from robots.behavior.publication import (
    canonical_bundle_id,
    validate_canonical_publication_root,
    validate_forensic_publication_binding,
)
from robots.behavior.recipe_catalog import load_behavior_recipe_catalog
from robots.behavior.run_manifest import (
    PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION,
    RUN_MANIFEST_SCHEMA_VERSION,
    pi0_nav_pick_exact_chunk_contract,
    resolve_run_manifest_public_tool_contract,
)
from robots.behavior.runtime import (
    BEHAVIOR_NATIVE_ENV_SEED,
    _expected_shared_policy_checkpoint_binding,
    _terminate_process,
    start_vla_server,
)
from robots.behavior.schemas import (
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    PUBLIC_TOOL_CONTRACTS,
)
from robots.behavior.serial_eval import (
    _checkout_identity,
    _gpu_lock_path,
    _manifest_owned_groups,
    _manifest_unverified_groups,
    _terminate_manifest_processes,
    _terminate_top_process,
)
from robots.behavior.task_specs import (
    TURNING_ON_RADIO_TASK_SPEC,
    BehaviorTaskSpec,
    get_task_spec,
)
from robots.behavior.terminal_success import (
    summarize_action_trace_success,
    validate_terminal_success_receipt,
)
from robots.behavior.toolkit import (
    RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE,
    BehaviorToolkit,
)
from robots.behavior.vla_client import BehaviorVLAClient

# Compatibility export for existing test/support callers.  The source of truth
# remains the versioned schema registry above.
BEHAVIOR_TOOL_NAMES = PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
PUBLIC_SEED = 0
NATIVE_INSTANCE = TURNING_ON_RADIO_TASK_SPEC.instance_for_public_seed(PUBLIC_SEED)
CANDIDATE_MAPPING_VERSION = TURNING_ON_RADIO_TASK_SPEC.candidate_mapping_version
TASK_NAME = TURNING_ON_RADIO_TASK_SPEC.task_name
TAG = "turning_on_radio_s0"
ATTEMPT_ENV_STEPS = 24_756
ATTEMPT_TOOL_CALLS = 350
ATTEMPT_WALL_CLOCK_S = 43_200
ATTEMPT_PROCESS_TIMEOUT_S = 46_800
SUMMARY_MAX_ITEMS = 8
SUMMARY_MAX_CHARS = 16_000
_OFFICIAL_SUCCESS_SOURCE = 'info["done"]["success"]'
_SYMBOLIC_PUBLIC_TOOL_DENYLIST = frozenset(
    tool_name for contract in PUBLIC_TOOL_CONTRACTS.values() for tool_name in contract
)
_EPOCH_PREDECESSOR_MAX_BYTES = 1024 * 1024

_FORBIDDEN_SUMMARY = re.compile(
    r"(?ix)"
    r"(?:\b(?:pixel|frame(?:_id)?|bbox|row|col|qpos|proprio|pose|xyz|"
    r"coordinate|left|right|seed|instance|checkpoint[_ /-]?path)\b|"
    r"\b[xyzuv]\s*[:=]\s*[-+]?\d+(?:\.\d+)?|"
    r"(?:^|\s)/(?:home|tmp|mnt)/|"
    r"\[[^\]]*[-+]?\d+(?:\.\d+)?\s*,[^\]]*\])"
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_epoch_predecessor_binding_file(
    path_value: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read one stable regular JSON object without following symlinks."""

    path = Path(path_value).expanduser().absolute()
    relative_parts = path.parts[1:]
    if not relative_parts:
        raise ValueError("epoch predecessor binding must name a regular file")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(path.anchor, directory_flags)
        for component in relative_parts[:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            relative_parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ValueError(
            "epoch predecessor binding must be a readable non-symlink regular file"
        ) from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("epoch predecessor binding must be a regular file")
        if before.st_size > _EPOCH_PREDECESSOR_MAX_BYTES:
            raise ValueError("epoch predecessor binding exceeds the size limit")
        chunks: list[bytes] = []
        remaining = _EPOCH_PREDECESSOR_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    stable_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(raw) != before.st_size or stable_before != stable_after:
        raise ValueError("epoch predecessor binding changed while being read")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "epoch predecessor binding must be one strict UTF-8 JSON object"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("epoch predecessor binding must be one JSON object")
    return {
        "binding_file_sha256": hashlib.sha256(raw).hexdigest(),
        "binding": value,
    }


def _process_record(proc: subprocess.Popen[Any], endpoint: str) -> dict[str, Any]:
    pid = int(proc.pid)
    try:
        pgid = os.getpgid(pid)
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        sid = int(fields[3])
        start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError):
        pgid = sid = start_ticks = None
    return {
        "managed": True,
        "pid": pid,
        "pgid": pgid,
        "sid": sid,
        "start_ticks": start_ticks,
        "endpoint": endpoint,
        "started_at": _utc_now(),
        "stopped_at": None,
        "returncode": None,
    }


def sanitize_prior_attempt_summaries(
    summaries: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return bounded, de-instantiated summaries safe for a fresh prompt."""

    sanitized: list[dict[str, Any]] = []
    for item in list(summaries)[-SUMMARY_MAX_ITEMS:]:
        index = item.get("attempt_index")
        outcome = str(item.get("outcome") or "task_failed").strip()
        raw = str(item.get("summary") or "Attempt ended without official success")
        clauses = re.split(r"(?<=[.!?。！？])\s+|\n+", raw)
        kept = [
            clause.strip()
            for clause in clauses
            if clause.strip() and not _FORBIDDEN_SUMMARY.search(clause)
        ]
        text = (
            " ".join(kept) or "Vary the semantic strategy and re-ground all evidence."
        )
        sanitized.append(
            {
                "attempt_index": int(index),
                "outcome": re.sub(r"[^a-z0-9_-]", "_", outcome.lower())[:64],
                "summary": text[:2000],
            }
        )
    while len(json.dumps(sanitized, ensure_ascii=False)) > SUMMARY_MAX_CHARS:
        if len(sanitized) > 1:
            sanitized.pop(0)
        else:
            sanitized[0]["summary"] = sanitized[0]["summary"][:8000]
            break
    return sanitized


def _prior_summaries_binding(
    summaries: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sanitized = sanitize_prior_attempt_summaries(summaries)
    canonical = json.dumps(
        sanitized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sanitized, {
        "count": len(sanitized),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


@dataclass(frozen=True)
class ExploreConfig:
    output_root: Path
    repo_root: Path
    python: Path
    behavior_repo: Path
    behavior_python: Path
    policy_checkpoint: Path
    policy_checkpoint_binding: dict[str, Any] | None = None
    task_name: str = TASK_NAME
    public_seed: int = PUBLIC_SEED
    cuda_device: str = "7"
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    max_turns: int = 300
    max_tool_calls: int = ATTEMPT_TOOL_CALLS
    planner_timeout_s: int = ATTEMPT_WALL_CLOCK_S
    attempt_timeout_s: int = ATTEMPT_PROCESS_TIMEOUT_S
    max_wall_clock_s: int = ATTEMPT_WALL_CLOCK_S
    vla_ready_timeout_s: int = 1800
    min_free_disk_gb: float = 10.0
    dashboard: bool = False
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    dashboard_language: str = "en"
    resume: bool = False
    candidate_instance_id: int | None = None
    candidate_state_sha256: str | None = None
    max_attempts: int | None = None
    initial_prior_summaries: tuple[dict[str, Any], ...] = ()
    recipe_catalog_sha256: str | None = None
    epoch_predecessor_binding: dict[str, Any] | None = None
    resource_binding: DatasetResourceBinding | None = None


def _task_spec(config: ExploreConfig) -> BehaviorTaskSpec:
    return get_task_spec(str(getattr(config, "task_name", TASK_NAME)))


def _tag(config: ExploreConfig) -> str:
    return _task_spec(config).tag(config.public_seed)


def _native_instance(config: ExploreConfig) -> int:
    return (
        int(config.candidate_instance_id)
        if config.candidate_instance_id is not None
        else _task_spec(config).instance_for_public_seed(
            config.public_seed,
            phase="explore",
        )
    )


def _job_tag(config: ExploreConfig) -> str:
    if config.candidate_instance_id is None:
        return _tag(config)
    return (
        f"{_task_spec(config).task_name}_candidate_i{int(config.candidate_instance_id)}"
    )


def _checkpoint_binding(config: ExploreConfig) -> dict[str, Any]:
    binding = config.policy_checkpoint_binding
    if not isinstance(binding, dict):
        raise ValueError("policy_checkpoint_binding is required")
    if binding.get("resolved_path") != str(config.policy_checkpoint.resolve()):
        raise ValueError("policy checkpoint path differs from its immutable binding")
    return json.loads(json.dumps(binding, sort_keys=True))


def _expected_job_checkpoint_binding(path: str | Path) -> dict[str, Any]:
    """Bind the fixed path; the once-per-Job VLA startup performs the full hash."""

    try:
        requested = Path(path).expanduser().resolve(strict=True)
        expected = _expected_shared_policy_checkpoint_binding()
    except OSError as error:
        raise ValueError(
            f"shared BEHAVIOR policy checkpoint is unavailable: {error}"
        ) from error
    if str(requested) != expected["resolved_path"]:
        raise ValueError(
            "BEHAVIOR Explore requires the shared policy checkpoint "
            f"{expected['resolved_path']}; got {requested}"
        )
    return expected


def _checkpoint_binding_path(config: ExploreConfig) -> Path:
    return config.output_root / "policy_checkpoint_binding.json"


def _resource_binding(config: ExploreConfig) -> DatasetResourceBinding:
    binding = config.resource_binding
    if not isinstance(binding, DatasetResourceBinding):
        raise ValueError("pinned BEHAVIOR resource binding is required")
    if binding.subtree != "behavior":
        raise ValueError("BEHAVIOR resource binding must select the behavior subtree")
    return binding


def _resource_source_path(config: ExploreConfig) -> Path:
    return config.output_root / "resource_source.json"


def _ensure_resource_source_file(config: ExploreConfig) -> Path:
    """Persist and locally revalidate the once-per-Job resource binding."""

    binding = verify_pinned_dataset_resources(_resource_binding(config))
    path = _resource_source_path(config)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Job resource source must be a regular non-symlink file")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Job resource source is unreadable") from error
        if existing != binding.as_dict():
            raise RuntimeError(
                "Job resource source differs from the pinned preflight binding"
            )
        return path
    write_dataset_resource_binding(binding, path)
    return path


def _ensure_checkpoint_binding_file(config: ExploreConfig) -> Path:
    """Persist the Job-preflight binding once for fresh attempt children."""

    path = _checkpoint_binding_path(config)
    binding = _checkpoint_binding(config)
    if path.exists():
        if path.is_symlink() or not path.is_file() or _read_json(path) != binding:
            raise RuntimeError(
                "Job policy checkpoint binding differs from the validated preflight"
            )
        return path
    _atomic_json(path, binding)
    return path


def _validate_config(config: ExploreConfig) -> None:
    spec = _task_spec(config)
    spec.instance_for_public_seed(config.public_seed, phase="explore")
    _checkpoint_binding(config)
    _resource_binding(config)
    if not isinstance(config.initial_prior_summaries, tuple):
        raise ValueError("initial_prior_summaries must be an immutable tuple")
    if config.candidate_instance_id is None:
        if config.candidate_state_sha256 is not None:
            raise ValueError("candidate state SHA requires a candidate instance")
    else:
        if (
            config.candidate_instance_id <= 0
            or spec.classify_instance(config.candidate_instance_id).kind != "candidate"
        ):
            raise ValueError(
                "candidate instance must be positive and non-public for this task"
            )
        if (
            not isinstance(config.candidate_state_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", config.candidate_state_sha256) is None
        ):
            raise ValueError("candidate instance requires its state-file SHA256")
        if config.max_attempts is None:
            raise ValueError("candidate Explore requires a finite max_attempts")
        if config.attempt_timeout_s > 7200:
            raise ValueError("candidate attempt timeout may not exceed two hours")
        if config.planner_timeout_s > 7200:
            raise ValueError("candidate planner timeout may not exceed two hours")
        if config.max_wall_clock_s > 7200:
            raise ValueError("candidate wall-clock budget may not exceed two hours")
    if config.max_attempts is not None and config.max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if (
        config.recipe_catalog_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", config.recipe_catalog_sha256) is None
    ):
        raise ValueError("recipe catalog SHA256 must be 64 lowercase hex characters")
    if config.epoch_predecessor_binding is not None and not isinstance(
        config.epoch_predecessor_binding, dict
    ):
        raise ValueError("epoch_predecessor_binding must be a JSON object")
    try:
        json.dumps(config.epoch_predecessor_binding, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "epoch_predecessor_binding must be JSON serializable"
        ) from error


@dataclass
class AttemptExecution:
    exit_code: int | None
    timed_out: bool
    final_result: dict[str, Any] | None
    forced_cleanup: dict[str, tuple[int, ...]]
    alive_after_cleanup: dict[str, tuple[int, ...]]
    ambiguous_groups: dict[str, tuple[int, ...]]


@dataclass
class ExploreDependencies:
    start_vla: Callable[[ExploreConfig, Path], tuple[str, subprocess.Popen[Any]]]
    stop_vla: Callable[[subprocess.Popen[Any] | None], None]
    bind_attempt_vla: Callable[[str, str], dict[str, Any]]
    run_attempt: Callable[[tuple[str, ...], Path, Path, int], AttemptExecution]
    free_disk_bytes: Callable[[Path], int]
    start_dashboard: Callable[
        [ExploreConfig, str], tuple[Any | None, Any | None, str | None]
    ]
    owns_vla: bool = True


class _AttemptDashboardRelay:
    """Tail child artifacts into the one job-owned in-memory Dashboard."""

    def __init__(self, attempt_dir: Path, dashboard: Any) -> None:
        self._attempt_dir = attempt_dir
        self._dashboard = dashboard
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"behavior-dashboard-relay-{attempt_dir.name}",
            daemon=True,
        )
        self._tool_records = 0
        self._dashboard_event_records = 0
        self._event_sink_active = False
        self._pi0_progress_chunk = -1
        self._pi0_progress_started = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._drain()

    @staticmethod
    def _json_lines(path: Path) -> list[dict[str, Any]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records

    def _drain_pi0_progress(self) -> None:
        states_paths = sorted(
            (self._attempt_dir / "vla_calls").glob("call_*/pi0_nav_pick_states.json")
        )
        if not states_paths:
            return
        payload = _read_json(states_paths[-1])
        states: Any = payload
        if payload is None:
            try:
                states = json.loads(states_paths[-1].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return
        if not isinstance(states, list):
            return
        for item in states:
            if not isinstance(item, dict):
                continue
            chunk = item.get("chunk")
            if isinstance(chunk, bool) or not isinstance(chunk, int):
                continue
            if chunk <= self._pi0_progress_chunk:
                continue
            monitor = item.get("pi0_nav_pick_monitor")
            review = monitor.get("visual_review") if isinstance(monitor, dict) else None
            self._dashboard.on_tool_progress(
                "pi0_nav_pick",
                strip_dashboard_frame_sources(
                    {
                        "chunk_index": chunk,
                        "env_step": item.get("total_env_steps"),
                        "visual_review": review,
                        "total_env_steps": item.get("total_env_steps"),
                    }
                ),
            )
            self._pi0_progress_started = True
            self._pi0_progress_chunk = chunk

    def _drain_dashboard_events(self) -> None:
        path = self._attempt_dir / "dashboard_events.jsonl"
        if path.exists():
            self._event_sink_active = True
        records = self._json_lines(path)
        for record in records[self._dashboard_event_records :]:
            channel = record.get("channel")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if channel == "event":
                # Child artifacts are not an authority for job-level success,
                # sealing, or publication.  The parent emits these events only
                # after validating final_result plus the immutable receipt and
                # artifact bindings.
                if payload.get("type") not in {
                    "official_success",
                    "workflow_complete",
                    "publication_complete",
                }:
                    self._dashboard.on_event(payload)
            elif channel == "usage":
                self._dashboard.on_usage(
                    inp=int(payload.get("inp") or 0),
                    out=int(payload.get("out") or 0),
                    tool_calls=int(payload.get("tool_calls") or 0),
                )
            elif channel == "tool_start":
                self._dashboard.on_tool_start(
                    str(payload.get("name") or "unknown"),
                    payload.get("arguments")
                    if isinstance(payload.get("arguments"), dict)
                    else {},
                )
            elif channel == "tool_progress":
                self._dashboard.on_tool_progress(
                    str(payload.get("name") or "unknown"),
                    strip_dashboard_frame_sources(
                        payload.get("result")
                        if isinstance(payload.get("result"), dict)
                        else {}
                    ),
                )
            elif channel == "tool_result":
                self._dashboard.on_tool_result(
                    str(payload.get("name") or "unknown"),
                    strip_dashboard_frame_sources(
                        payload.get("result")
                        if isinstance(payload.get("result"), dict)
                        else {}
                    ),
                )
            elif channel == "frame":
                self._relay_frame(payload)
            elif channel == "metadata":
                self._dashboard.set_metadata(payload)
        self._dashboard_event_records = len(records)

    def _relay_frame(self, payload: dict[str, Any]) -> None:
        callback = getattr(self._dashboard, "on_frame", None)
        relative_value = payload.get("relative_path")
        claimed_sha256 = payload.get("sha256")
        if (
            not callable(callback)
            or not isinstance(relative_value, str)
            or not relative_value
            or not isinstance(claimed_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", claimed_sha256) is None
        ):
            return
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            return
        root = self._attempt_dir.resolve()
        candidate = self._attempt_dir / relative
        cursor = self._attempt_dir
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            image = resolved.read_bytes()
        except (OSError, ValueError):
            return
        if hashlib.sha256(image).hexdigest() != claimed_sha256:
            return
        callback(
            str(payload.get("camera") or "head"),
            image,
            env_step=payload.get("env_step"),
        )

    def _drain(self) -> None:
        self._drain_dashboard_events()
        if self._event_sink_active:
            return
        self._drain_pi0_progress()
        records = self._json_lines(self._attempt_dir / "behavior_tool_trace.jsonl")
        for record in records[self._tool_records :]:
            name = str(record.get("tool") or "unknown")
            arguments = record.get("input")
            result = record.get("result")
            if not (name == "pi0_nav_pick" and self._pi0_progress_started):
                self._dashboard.on_tool_start(
                    name, arguments if isinstance(arguments, dict) else {}
                )
            self._dashboard.on_tool_result(
                name,
                strip_dashboard_frame_sources(
                    result if isinstance(result, dict) else {}
                ),
            )
        self._tool_records = len(records)

    def _run(self) -> None:
        while not self._stop.wait(0.5):
            self._drain()


def _claim_job_lock(root: Path) -> tuple[int, Path]:
    """Take a non-blocking lock for one exact Explore job root."""

    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.serial-explore.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise RuntimeError(f"another Explore job owns {lock_path}")
    return descriptor, lock_path


def _claim_gpu_lock(cuda_device: str) -> tuple[int, Path]:
    lock_path = _gpu_lock_path(cuda_device)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"explore_pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise RuntimeError(f"another BEHAVIOR job owns GPU lock {lock_path}")
    return descriptor, lock_path


def _release_job_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def build_attempt_argv(
    config: ExploreConfig,
    *,
    job_id: str,
    attempt_index: int,
    output_dir: Path,
    summaries_path: Path,
    vla_endpoint: str,
    vla_binding_id: str,
    reviewed_memory_snapshot_sha256: str,
    recipe_catalog_sha256: str,
) -> tuple[str, ...]:
    """Build one fresh Agent/env worker invocation."""

    spec = _task_spec(config)
    argv = [
        str(config.python),
        "-m",
        "robots.behavior.cli",
        "--env",
        "behavior",
        "--planner",
        "codex",
        "--model",
        config.model,
        "--reasoning-effort",
        config.reasoning_effort,
        "--suite",
        "behavior_2025_challenge",
        "--task",
        str(spec.task_index),
        "--task-name",
        spec.task_name,
        "--activity-definition-id",
        str(spec.activity_definition_id),
        "--activity-instance-id",
        str(_native_instance(config)),
        "--scene-model",
        spec.scene_model,
        "--public-seed",
        str(config.public_seed),
        "--seed",
        str(BEHAVIOR_NATIVE_ENV_SEED),
        "--behavior-phase",
        "explore",
        "--behavior-job-id",
        job_id,
        "--behavior-attempt-index",
        str(attempt_index),
        "--behavior-prior-attempt-summaries-file",
        str(summaries_path),
        "--behavior-vla-binding-id",
        vla_binding_id,
        "--behavior-policy-checkpoint-binding-file",
        str(_checkpoint_binding_path(config)),
        "--behavior-reviewed-memory-snapshot-sha256",
        reviewed_memory_snapshot_sha256,
        "--behavior-recipe-catalog-sha256",
        recipe_catalog_sha256,
        "--behavior-resource-root",
        str(_resource_binding(config).root),
        "--behavior-resource-source-file",
        str(_resource_source_path(config)),
        "--behavior-repo",
        str(config.behavior_repo),
        "--behavior-python",
        str(config.behavior_python),
        "--policy-checkpoint",
        str(config.policy_checkpoint),
        "--cuda-device",
        config.cuda_device,
        "--vla-endpoint",
        vla_endpoint,
        "--max-episode-steps",
        str(ATTEMPT_ENV_STEPS),
        "--max-tool-calls",
        str(config.max_tool_calls),
        "--max-wall-clock-s",
        str(config.max_wall_clock_s),
        "--max-turns",
        str(config.max_turns),
        "--planner-timeout-s",
        str(config.planner_timeout_s),
        "--output-dir",
        str(output_dir),
    ]
    if config.dashboard:
        argv.append("--behavior-dashboard-event-sink")
    if config.candidate_instance_id is not None:
        argv.extend(
            [
                "--behavior-candidate-campaign-id",
                job_id,
                "--behavior-candidate-instance-id",
                str(config.candidate_instance_id),
                "--behavior-candidate-state-sha256",
                str(config.candidate_state_sha256),
            ]
        )
    if "--dashboard" in argv or "--dashboard-auto-start" in argv:
        raise AssertionError("attempt child must not own a Dashboard server")
    return tuple(argv)


def _default_start_vla(
    config: ExploreConfig, output_dir: Path
) -> tuple[str, subprocess.Popen[Any]]:
    args = argparse.Namespace(
        behavior_python=str(config.behavior_python),
        behavior_repo=str(config.behavior_repo),
        policy_checkpoint=str(config.policy_checkpoint),
        seed=BEHAVIOR_NATIVE_ENV_SEED,
        vla_port=0,
        vla_ready_timeout_s=config.vla_ready_timeout_s,
        cuda_device=config.cuda_device,
        _behavior_policy_checkpoint_binding=_checkpoint_binding(config),
    )
    return start_vla_server(args, output_dir=output_dir)


def _default_stop_vla(proc: subprocess.Popen[Any] | None) -> None:
    _terminate_process(proc)


def _default_bind_attempt_vla(
    endpoint: str,
    binding_id: str,
) -> dict[str, Any]:
    client = BehaviorVLAClient(endpoint)
    try:
        disabled = client.disable_actions()
        if disabled.get("actions_enabled") is not False:
            raise RuntimeError("persistent VLA action gate did not disable")
        return client.bind_actions(binding_id)
    finally:
        client.close()


def _default_run_attempt(
    argv: tuple[str, ...], output_dir: Path, log_path: Path, timeout_s: int
) -> AttemptExecution:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"attempt output is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        output_dir.rmdir()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv,
            cwd=configured_repo_root(argv),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_manifest_processes(output_dir)
            _terminate_top_process(process)
        except BaseException:
            _terminate_manifest_processes(output_dir)
            _terminate_top_process(process)
            raise
    forced = _manifest_owned_groups(output_dir)
    if forced:
        _terminate_manifest_processes(output_dir)
    alive = _manifest_owned_groups(output_dir)
    ambiguous = _manifest_unverified_groups(output_dir)
    return AttemptExecution(
        exit_code=process.returncode,
        timed_out=timed_out,
        final_result=_read_json(output_dir / "final_result.json"),
        forced_cleanup=forced,
        alive_after_cleanup=alive,
        ambiguous_groups=ambiguous,
    )


def configured_repo_root(argv: tuple[str, ...]) -> Path:
    """Return the checkout containing this module for child execution."""

    del argv
    return Path(__file__).resolve().parents[2]


def _default_start_dashboard(
    config: ExploreConfig, job_id: str
) -> tuple[Any | None, Any | None, str | None]:
    if not config.dashboard:
        return None, None, None
    from robots.behavior.dashboard_server import DashboardServer
    from robots.behavior.dashboard_state import State

    server = DashboardServer(
        host=config.dashboard_host,
        port=config.dashboard_port,
        runs_dir=str(config.output_root),
        language=config.dashboard_language,
    )
    state = State(
        run_id=f"behavior/{job_id}",
        name=_job_tag(config),
        suite="behavior_2025_challenge",
        task=_task_spec(config).task_index,
        seed=config.public_seed,
        output_dir=str(config.output_root),
        video_path=str(config.output_root / "episode.mp4"),
    )
    spec = _task_spec(config)
    state.set_metadata(
        {
            "planner": "codex",
            "model": config.model,
            "reasoning-effort": config.reasoning_effort,
            "task-name": spec.task_name,
            "task-language": spec.task_language,
            "activity-definition-id": spec.activity_definition_id,
            "activity-instance-id": _native_instance(config),
            "public-instance-id": spec.instance_for_public_seed(
                config.public_seed,
                phase="explore",
            ),
            "public-seed-max": max(spec.public_seed_to_instance),
            "scene-model": spec.scene_model,
            "behavior-phase": "explore",
            "public-seed": config.public_seed,
            "job-id": job_id,
            "candidate-instance-id": config.candidate_instance_id,
            "max-episode-steps": ATTEMPT_ENV_STEPS,
            "max-tool-calls": config.max_tool_calls,
            "max-wall-clock-s": config.max_wall_clock_s,
            "public-tool-contract-version": (CURRENT_PUBLIC_TOOL_CONTRACT_VERSION),
            "public-tool-count": len(
                PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
            ),
        }
    )
    server.register(state)
    server.arm_auto_start(
        {
            "job-id": job_id,
            "task-name": spec.task_name,
            "public-seed": config.public_seed,
        }
    )
    return server, state, server.start()


def default_dependencies() -> ExploreDependencies:
    return ExploreDependencies(
        start_vla=_default_start_vla,
        stop_vla=_default_stop_vla,
        bind_attempt_vla=_default_bind_attempt_vla,
        run_attempt=_default_run_attempt,
        free_disk_bytes=lambda path: shutil.disk_usage(path).free,
        start_dashboard=_default_start_dashboard,
    )


def _reviewed_memory_binding(config: ExploreConfig) -> dict[str, Any]:
    reviewed_memory = load_behavior_memory_snapshot(
        _resource_binding(config).root / "memory"
    )
    selection = reviewed_memory.select_task(_task_spec(config).task_name)
    return {
        "snapshot_sha256": reviewed_memory.snapshot_sha256,
        "manifest": asdict(reviewed_memory.manifest_binding),
        "files": {
            name: asdict(metadata) for name, metadata in reviewed_memory.files.items()
        },
        "selection": selection.public_binding,
    }


def _plain_binding(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_binding(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_binding(item) for item in value]
    return value


def _reviewed_recipe_catalog_binding(config: ExploreConfig) -> dict[str, Any]:
    catalog = load_behavior_recipe_catalog(_resource_binding(config).root / "recipes")
    if (
        config.recipe_catalog_sha256 is not None
        and catalog.catalog_sha256 != config.recipe_catalog_sha256
    ):
        raise RuntimeError(
            "reviewed BEHAVIOR Recipe Catalog SHA256 differs from the pinned epoch"
        )
    selection = catalog.select(_task_spec(config).task_name, "explore")
    selection_binding = _plain_binding(selection.public_binding)
    selected_entries = selection_binding.get("selected_entries")
    if not isinstance(selected_entries, list) or any(
        not isinstance(entry, dict)
        or entry.get("provenance_class")
        not in {"canonical_public_explore", "candidate_explore_reviewed"}
        for entry in selected_entries
    ):
        raise RuntimeError("Explore Recipe Catalog selection has invalid provenance")
    return {
        "catalog_sha256": catalog.catalog_sha256,
        "manifest": _plain_binding(catalog.manifest_binding),
        "files": _plain_binding(catalog.files),
        "selection": selection_binding,
        "selected_ids": list(selection.selected_ids),
    }


def _runtime_recipe_catalog_binding(
    binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "catalog_sha256": binding["catalog_sha256"],
        "selection": binding["selection"],
    }


def _new_manifest(config: ExploreConfig, job_id: str) -> dict[str, Any]:
    spec = _task_spec(config)
    candidate = config.candidate_instance_id is not None
    _, inherited_prior_binding = _prior_summaries_binding(
        config.initial_prior_summaries
    )
    return {
        "schema_version": 1,
        "job_id": job_id,
        "status": "starting",
        "started_at": _utc_now(),
        "finished_at": None,
        "protocol": {
            "behavior_phase": "explore",
            "task_index": spec.task_index,
            "task_name": spec.task_name,
            "task_language": spec.task_language,
            "public_seed": config.public_seed,
            "recipe_tag": _tag(config),
            "task_spec": {
                "task_index": spec.task_index,
                "task_name": spec.task_name,
                "task_language": spec.task_language,
                "prompt_profile_id": spec.prompt_profile_id,
                "activity_definition_id": spec.activity_definition_id,
                "scene_model": spec.scene_model,
                "public_seed_to_instance": {
                    str(seed): instance_id
                    for seed, instance_id in spec.public_seed_to_instance.items()
                },
                "mapping_version": spec.mapping_version,
                "candidate_mapping_version": spec.candidate_mapping_version,
                "explore_public_seeds": list(spec.explore_public_seeds),
                "eval_public_seeds": list(spec.eval_public_seeds),
            },
            **(
                {
                    "campaign_kind": "candidate_instance_explore",
                    "job_tag": _job_tag(config),
                }
                if candidate
                else {}
            ),
            "attempt_protocol": "fresh_env_and_agent_process",
            "agent_invocation_per_attempt": True,
            "reset_registered": False,
            "agent_finish_registered": False,
            "public_tool_contract_version": (CURRENT_PUBLIC_TOOL_CONTRACT_VERSION),
            "public_primitives": list(
                PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
            ),
            "pi0_nav_pick_contract": pi0_nav_pick_exact_chunk_contract(),
            "persistent_vla": True,
            "recipe_catalog_consumer": "explore",
            "inherited_prior_summaries": inherited_prior_binding,
            "automatic_retry": config.max_attempts is None or config.max_attempts > 1,
            "total_limits": {
                "attempts": config.max_attempts,
                "env_steps": None,
                "tool_calls": None,
                "wall_clock_s": (
                    config.attempt_timeout_s * config.max_attempts
                    if config.max_attempts is not None
                    else None
                ),
            },
            "per_attempt_limits": {
                "env_steps": ATTEMPT_ENV_STEPS,
                "tool_calls": config.max_tool_calls,
                "wall_clock_s": config.max_wall_clock_s,
            },
        },
        "native_binding": {
            "mapping_version": (
                spec.candidate_mapping_version if candidate else spec.mapping_version
            ),
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": _native_instance(config),
            "scene_model": spec.scene_model,
            "env_seed": BEHAVIOR_NATIVE_ENV_SEED,
            **({"state_sha256": config.candidate_state_sha256} if candidate else {}),
        },
        "task_identity": {
            "task_name": spec.task_name,
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": _native_instance(config),
        },
        "policy_checkpoint": _checkpoint_binding(config),
        "planner": {
            "backend": "codex",
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
        },
        "source": _checkout_identity(config.repo_root),
        "resource_source": _resource_binding(config).as_dict(),
        "reviewed_repo_memory": _reviewed_memory_binding(config),
        "reviewed_recipe_catalog": _reviewed_recipe_catalog_binding(config),
        "epoch_predecessor": (
            json.loads(
                json.dumps(
                    config.epoch_predecessor_binding,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            if config.epoch_predecessor_binding is not None
            else None
        ),
        "processes": {"vla": None},
        "attempts": [],
        "cumulative": {
            "attempts": 0,
            "env_steps": 0,
            "vla_chunks": 0,
            "vla_invocations": 0,
            "tool_calls": 0,
            "wall_clock_s": 0.0,
        },
        "task_success": None,
        "artifact_seal_complete": False,
        # Compatibility alias only. Raw success and publication never consult it.
        "workflow_complete": False,
        "publication_complete": False,
        "terminal_failure": None,
        "dashboard_url": None,
        "blocked_reason": None,
    }


def _attempt_summary(
    final_result: dict[str, Any] | None,
    *,
    raw_success: bool = False,
) -> str:
    if raw_success:
        return (
            "Fresh public evidence and raw official success confirmed completion "
            "of the task interaction."
        )
    if not isinstance(final_result, dict):
        return "Attempt produced no valid final result."
    summary = final_result.get("agent_summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    error = final_result.get("error")
    if isinstance(error, str) and error.strip():
        return f"Infrastructure or planner error: {error.strip()}"
    return "Attempt ended without raw official success."


def _exact_pi0_call_artifacts_valid(
    attempt_dir: Path,
    tool_trace: list[dict[str, Any]],
) -> bool:
    """Validate bounded-chunk accounting before success publication."""

    executed = [
        record
        for record in tool_trace
        if record.get("tool") == "pi0_nav_pick"
        and not (
            isinstance(record.get("result"), dict)
            and record["result"].get("stop_reason") == "precondition_rejected"
        )
    ]
    call_root = attempt_dir / "vla_calls"
    call_dirs = (
        sorted(
            path
            for path in call_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and re.fullmatch(r"call_\d{3}", path.name)
        )
        if call_root.is_dir()
        else []
    )
    if [path.name for path in call_dirs] != [
        f"call_{index:03d}" for index in range(1, len(call_dirs) + 1)
    ]:
        return False
    if len(call_dirs) != len(executed):
        return False
    for index, (call_dir, trace_record) in enumerate(
        zip(call_dirs, executed, strict=True),
        start=1,
    ):
        call = _read_json(call_dir / "pi0_nav_pick_call.json")
        result_path = call_dir / "pi0_nav_pick_result.json"
        result = _read_json(result_path)
        trace_input = trace_record.get("input")
        trace_result = trace_record.get("result")
        if (
            not isinstance(call, dict)
            or call.get("schema_version") != PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION
            or call.get("name") != "pi0_nav_pick"
            or call.get("status") != "completed"
            or call.get("global_vla_invocations") != index
            or not result_path.is_file()
            or result_path.is_symlink()
            or call.get("result_path") != str(result_path)
            or call.get("result_sha256")
            != hashlib.sha256(result_path.read_bytes()).hexdigest()
            or not isinstance(result, dict)
            or not isinstance(trace_input, dict)
            or not isinstance(trace_result, dict)
        ):
            return False
        requested = call.get("requested_chunks")
        chunks_used = result.get("chunks_used")
        full_chunks = result.get("full_chunks_executed")
        vla_steps = result.get("vla_env_steps_used")
        terminated = result.get("terminated")
        truncated = result.get("truncated")
        stop_reason = result.get("stop_reason")
        task_success = result.get("task_success")
        exact_requested = result.get("exact_requested_chunks_completed")
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested < 1
            or trace_input.get("instruction") != call.get("instruction")
            or isinstance(trace_input.get("chunks"), bool)
            or trace_input.get("chunks") != requested
            or result.get("requested_chunks") != requested
            or isinstance(chunks_used, bool)
            or not isinstance(chunks_used, int)
            or isinstance(full_chunks, bool)
            or not isinstance(full_chunks, int)
            or isinstance(vla_steps, bool)
            or not isinstance(vla_steps, int)
            or result.get("run_nonce") != call.get("run_nonce")
            or result.get("attempt_nonce") != call.get("attempt_nonce")
            or result.get("attempt_index") != call.get("attempt_index")
            or not isinstance(task_success, bool)
            or not isinstance(exact_requested, bool)
        ):
            return False
        exact = bool(
            task_success is False
            and chunks_used == requested
            and full_chunks == requested
            and vla_steps == requested * 32
            and exact_requested is True
            and terminated is False
            and truncated is False
            and stop_reason == "requested_chunks_completed"
        )
        success_receipt = result.get("official_success_receipt")
        root_receipt = _read_json(attempt_dir / "official_success_receipt.json")
        success_validation = validate_terminal_success_receipt(
            tool_name="pi0_nav_pick",
            step=trace_record.get("step"),
            result=result,
            output_dir=attempt_dir,
        )
        official_success_binding = (
            _official_success_binding(attempt_dir) if task_success is True else None
        )
        success_receipt_bound = bool(
            success_validation.valid
            and isinstance(success_receipt, dict)
            and success_receipt == root_receipt
            and trace_result.get("official_success_receipt") == success_receipt
            and success_receipt.get("run_nonce") == call.get("run_nonce")
            and success_receipt.get("attempt_nonce") == call.get("attempt_nonce")
            and success_receipt.get("attempt_index") == call.get("attempt_index")
            and isinstance(official_success_binding, dict)
            and success_receipt.get("run_nonce")
            == official_success_binding.get("run_nonce")
            and success_receipt.get("attempt_nonce")
            == official_success_binding.get("attempt_nonce")
            and success_receipt.get("attempt_index")
            == official_success_binding.get("attempt_index")
            and success_receipt.get("env_step")
            == official_success_binding.get("env_step")
            and success_receipt.get("receipt_sha256")
            == official_success_binding.get("receipt_sha256")
        )
        completed_work_accounting = bool(
            full_chunks == chunks_used and vla_steps == full_chunks * 32
        )
        partial_work_accounting = bool(
            full_chunks == chunks_used - 1
            and full_chunks * 32 + 1 <= vla_steps <= chunks_used * 32 - 1
        )
        expected_exact_requested = bool(
            chunks_used == requested
            and full_chunks == requested
            and vla_steps == requested * 32
        )
        success_terminal = bool(
            index == len(call_dirs)
            and trace_record is tool_trace[-1]
            and task_success is True
            and 1 <= chunks_used <= requested
            and terminated is False
            and truncated is False
            and stop_reason == "official_task_success"
            and (completed_work_accounting or partial_work_accounting)
            and exact_requested is expected_exact_requested
            and success_receipt_bound
        )
        hard_terminal = bool(
            index == len(call_dirs)
            and task_success is False
            and 1 <= chunks_used <= requested
            and exact_requested is False
            and (
                (
                    terminated is True
                    and truncated is False
                    and stop_reason == "terminated"
                )
                or (
                    terminated is False
                    and truncated is True
                    and stop_reason == "truncated"
                )
            )
            and (completed_work_accounting or partial_work_accounting)
        )
        if not (exact or success_terminal or hard_terminal):
            return False
        if (
            call.get("task_success") is not task_success
            or call.get("exact_requested_chunks_completed") is not exact_requested
        ):
            return False
        for field in (
            "requested_chunks",
            "exact_requested_chunks_completed",
            "chunks_used",
            "full_chunks_executed",
            "vla_env_steps_used",
            "task_success",
            "terminated",
            "truncated",
            "stop_reason",
        ):
            if trace_result.get(field) != result.get(field):
                return False
    return True


def _exact_pi0_attempt_artifacts_valid(attempt_dir: Path) -> bool:
    """Validate bounded Pi0 artifacts without independently declaring success."""

    tool_path = attempt_dir / "behavior_tool_trace.jsonl"
    try:
        tool_bytes = _read_contained_regular_file(attempt_dir, tool_path)
        tool_trace = [
            json.loads(line)
            for line in tool_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, UnicodeError):
        return False
    if not tool_trace or not all(isinstance(record, dict) for record in tool_trace):
        return False
    return _exact_pi0_call_artifacts_valid(attempt_dir, tool_trace)


def _normalized_action_env_step(record: dict[str, Any]) -> int | None:
    """Normalize action-trace lineage to the receipt's one-based env step."""

    def non_negative_int(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    has_env_step = "env_step" in record
    env_step = non_negative_int(record.get("env_step"))
    action_step = non_negative_int(record.get("step"))
    if has_env_step:
        return env_step
    env_idx = record.get("env_idx")
    if (
        isinstance(env_idx, bool)
        or not isinstance(env_idx, int)
        or env_idx != 0
        or action_step is None
    ):
        return None
    return action_step + 1


def _official_success_binding(attempt_dir: Path) -> dict[str, Any] | None:
    """Validate the runtime-owned raw-success receipt and its trace lineage."""

    receipt_path = attempt_dir / "official_success_receipt.json"
    action_path = attempt_dir / "behavior_action_trace.jsonl"
    tool_path = attempt_dir / "behavior_tool_trace.jsonl"
    try:
        receipt_bytes = _read_contained_regular_file(attempt_dir, receipt_path)
        action_bytes = _read_contained_regular_file(attempt_dir, action_path)
        tool_bytes = _read_contained_regular_file(attempt_dir, tool_path)
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        tool_trace = [
            json.loads(line)
            for line in tool_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        action_trace = [
            json.loads(line)
            for line in action_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(receipt, dict) or not tool_trace or not action_trace:
        return None
    schema_version = receipt.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        return None
    env_step = receipt.get("env_step")
    validation = validate_terminal_success_receipt(
        tool_name="runtime_owned_receipt",
        step=max(1, int(env_step)) if isinstance(env_step, int) else 1,
        result={
            "task_success": True,
            "official_success_receipt": receipt,
        },
        output_dir=attempt_dir,
    )
    if not validation.valid:
        return None
    try:
        attempt_index = int(attempt_dir.name.removeprefix("attempt_"))
    except ValueError:
        return None
    receipt_attempt_index = receipt.get("attempt_index")
    if (
        isinstance(receipt_attempt_index, bool)
        or not isinstance(receipt_attempt_index, int)
        or receipt_attempt_index != attempt_index
    ):
        return None

    def raw_done_success(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        done = record.get("info_done")
        return bool(isinstance(done, dict) and done.get("success") is True)

    raw_success_steps = []
    for record in action_trace:
        if not raw_done_success(record):
            continue
        normalized_step = _normalized_action_env_step(record)
        if normalized_step is None:
            return None
        raw_success_steps.append(normalized_step)
    if not raw_success_steps or receipt.get("env_step") != raw_success_steps[0]:
        return None
    receipt_bound_tool = False
    for record in tool_trace:
        if not isinstance(record, dict):
            continue
        result = record.get("result")
        if not isinstance(result, dict) or result.get("task_success") is not True:
            continue
        if (
            result.get("attempt_nonce") != receipt["attempt_nonce"]
            or result.get("run_nonce") != receipt["run_nonce"]
        ):
            continue
        trace_validation = validate_terminal_success_receipt(
            tool_name=str(record.get("tool") or ""),
            step=record.get("step"),
            result=result,
            output_dir=attempt_dir,
        )
        if trace_validation.valid:
            receipt_bound_tool = True
            break
    if not receipt_bound_tool:
        return None
    return {
        "source": _OFFICIAL_SUCCESS_SOURCE,
        "run_nonce": receipt["run_nonce"],
        "attempt_nonce": receipt["attempt_nonce"],
        "attempt_index": attempt_index,
        "env_step": receipt["env_step"],
        "receipt_sha256": receipt["receipt_sha256"],
        "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "source_artifacts_sha256": {
            "official_success_receipt": hashlib.sha256(receipt_bytes).hexdigest(),
            "behavior_action_trace": hashlib.sha256(action_bytes).hexdigest(),
            "behavior_tool_trace": hashlib.sha256(tool_bytes).hexdigest(),
        },
    }


def _terminal_failure_binding(
    attempt_dir: Path,
    *,
    task_name: str = TASK_NAME,
) -> dict[str, Any] | None:
    """Validate a fresh-frame-bound, runtime-sealed visual task failure."""

    policy = get_task_spec(task_name).terminal_failure_policy
    if policy is None:
        return None
    receipt_path = attempt_dir / "terminal_failure_receipt.json"
    tool_path = attempt_dir / "behavior_tool_trace.jsonl"
    action_path = attempt_dir / "behavior_action_trace.jsonl"
    final_path = attempt_dir / "final_result.json"
    try:
        receipt_bytes = _read_contained_regular_file(attempt_dir, receipt_path)
        tool_bytes = _read_contained_regular_file(attempt_dir, tool_path)
        action_bytes = _read_contained_regular_file(attempt_dir, action_path)
        final_result = _read_json_contained(attempt_dir, final_path)
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        tool_trace = [
            json.loads(line)
            for line in tool_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
        action_trace = [
            json.loads(line)
            for line in action_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, UnicodeError):
        return None
    if (
        not isinstance(receipt, dict)
        or not tool_trace
        or final_result.get("task_success") is not False
        or receipt.get("source") != "llm_fresh_visual_observation"
        or receipt.get("condition") != policy.condition
        or receipt.get("cause") not in set(policy.causes)
        or receipt.get("camera") not in set(policy.cameras)
        or receipt.get("task_success") is not False
        or receipt.get("official_success_source") != _OFFICIAL_SUCCESS_SOURCE
    ):
        return None
    try:
        attempt_index = int(attempt_dir.name.removeprefix("attempt_"))
    except ValueError:
        return None
    if (
        receipt.get("attempt_index") != attempt_index
        or not isinstance(receipt.get("env_step"), int)
        or isinstance(receipt.get("env_step"), bool)
        or receipt["env_step"] < 0
        or not isinstance(receipt.get("run_nonce"), str)
        or not receipt["run_nonce"]
        or not isinstance(receipt.get("attempt_nonce"), str)
        or not receipt["attempt_nonce"]
    ):
        return None
    receipt_material = dict(receipt)
    declared_receipt_sha256 = receipt_material.pop("receipt_sha256", None)
    computed_receipt_sha256 = hashlib.sha256(
        json.dumps(
            receipt_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if declared_receipt_sha256 != computed_receipt_sha256:
        return None
    visual_checkpoint_id = receipt.get("visual_checkpoint_id")
    if (
        not isinstance(visual_checkpoint_id, str)
        or re.fullmatch(r"visual_checkpoint_[0-9]{3,}", visual_checkpoint_id) is None
    ):
        return None
    metadata_path = (
        attempt_dir / "visual_checkpoints" / visual_checkpoint_id / "metadata.json"
    )
    try:
        metadata_bytes = _read_contained_regular_file(attempt_dir, metadata_path)
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError, UnicodeError):
        return None
    if (
        not isinstance(metadata, dict)
        or hashlib.sha256(metadata_bytes).hexdigest()
        != receipt.get("visual_checkpoint_metadata_sha256")
        or metadata.get("visual_checkpoint_id") != visual_checkpoint_id
        or metadata.get("capture_group_id")
        != receipt.get("visual_checkpoint_capture_group_id")
        or metadata.get("capture_group_id") != receipt.get("capture_group_id")
        or metadata.get("terminal_failure")
        != {
            "camera": receipt["camera"],
            "cause": receipt["cause"],
            "condition": receipt["condition"],
            "frame_id": receipt.get("frame_id"),
        }
    ):
        return None
    cited_frames = [
        camera_record
        for camera_record in metadata.get("cameras", {}).values()
        if isinstance(camera_record, dict)
        and camera_record.get("frame_id") == receipt.get("frame_id")
        and camera_record.get("capture_group_id") == receipt.get("capture_group_id")
        and camera_record.get("capture_env_step") == receipt.get("env_step")
    ]
    if len(cited_frames) != 1:
        return None
    images_sha256 = receipt.get("images_sha256")
    cameras = metadata.get("cameras")
    if (
        not isinstance(images_sha256, dict)
        or set(images_sha256) != {"head", "left_wrist", "right_wrist"}
        or not isinstance(cameras, dict)
        or set(cameras) != {"head", "left_wrist", "right_wrist"}
    ):
        return None
    try:
        for camera in ("head", "left_wrist", "right_wrist"):
            declared = images_sha256[camera]
            camera_record = cameras[camera]
            if not isinstance(declared, dict) or set(declared) != {"rgb", "depth"}:
                return None
            if not isinstance(camera_record, dict):
                return None
            for kind, path_key in (("rgb", "rgb_path"), ("depth", "depth_path")):
                image_path = Path(str(camera_record[path_key]))
                image_bytes = _read_contained_regular_file(attempt_dir, image_path)
                if hashlib.sha256(image_bytes).hexdigest() != declared[kind]:
                    return None
    except (KeyError, OSError, RuntimeError):
        return None

    def raw_done_success(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        done = record.get("info_done")
        return bool(isinstance(done, dict) and done.get("success") is True)

    if any(raw_done_success(record) for record in action_trace):
        return None
    if any(
        isinstance(record, dict)
        and isinstance(record.get("result"), dict)
        and record["result"].get("task_success") is True
        for record in tool_trace
    ):
        return None
    final_tool = tool_trace[-1]
    final_tool_result = (
        final_tool.get("result") if isinstance(final_tool, dict) else None
    )
    if (
        not isinstance(final_tool_result, dict)
        or final_tool.get("tool") != "save_robot_state_checkpoint"
        or final_tool_result.get("_finish") is not True
        or final_tool_result.get("task_success") is not False
        or final_tool_result.get("stop_reason") != policy.condition
        or final_tool_result.get("runner_termination_reason") != policy.runner_reason
        or final_tool_result.get("terminal_failure_receipt") != receipt
    ):
        return None
    return {
        "source": "llm_fresh_visual_observation",
        "condition": policy.condition,
        "cause": receipt["cause"],
        "camera": receipt["camera"],
        "frame_id": receipt["frame_id"],
        "env_step": receipt["env_step"],
        "attempt_index": attempt_index,
        "run_nonce": receipt["run_nonce"],
        "attempt_nonce": receipt["attempt_nonce"],
        "receipt_sha256": declared_receipt_sha256,
        "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "visual_checkpoint_metadata_sha256": receipt[
            "visual_checkpoint_metadata_sha256"
        ],
        "behavior_action_trace_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "behavior_tool_trace_sha256": hashlib.sha256(tool_bytes).hexdigest(),
    }


def _runtime_attempt_result(
    attempt_dir: Path,
    final_result: dict[str, Any] | None,
    *,
    tag: str = TAG,
) -> dict[str, Any] | None:
    """Resolve only runtime-owned official-success artifacts, never Agent prose."""

    candidates = [
        final_result,
        _read_json(attempt_dir / "behavior_result.json"),
        _read_json(attempt_dir / f"{tag}.json"),
    ]
    valid = [item for item in candidates if isinstance(item, dict)]
    binding = _official_success_binding(attempt_dir)
    if binding is not None:
        merged = dict(final_result or {})
        for item in valid:
            merged.update(item)
        merged.update(
            {
                "task_success": True,
                "official_success_source": _OFFICIAL_SUCCESS_SOURCE,
                "official_success_receipt": binding,
                "recovered_from_runtime_receipt": not bool(
                    final_result and final_result.get("task_success") is True
                ),
            }
        )
        return merged
    for item in valid:
        if item.get("task_success") is not True:
            return item
    if valid:
        return {
            "task_success": False,
            "artifact_seal_complete": False,
            # Compatibility alias only; not a task-success gate.
            "workflow_complete": False,
            "error": "unverifiable task_success=true was rejected",
            "official_success_source": None,
        }
    return None


def _attempt_run_identity_valid(
    run_manifest: Any,
    *,
    task_spec: BehaviorTaskSpec,
    public_seed: int,
    recipe_tag: str,
    native_instance: int,
) -> bool:
    """Bind one child attempt to the exact task cell requested by its Job."""

    if not isinstance(run_manifest, dict):
        return False
    try:
        contract_version, _ = resolve_run_manifest_public_tool_contract(run_manifest)
    except ValueError:
        return False
    if (
        run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION
        or contract_version != CURRENT_PUBLIC_TOOL_CONTRACT_VERSION
    ):
        return False
    protocol = run_manifest.get("protocol")
    protocol_spec = protocol.get("task_spec") if isinstance(protocol, dict) else None
    protocol_identity = (
        protocol.get("task_identity") if isinstance(protocol, dict) else None
    )
    prompt = protocol.get("prompt") if isinstance(protocol, dict) else None
    task = run_manifest.get("task")
    task_identity = run_manifest.get("task_identity")
    native = run_manifest.get("native_binding")
    if not all(
        isinstance(value, dict)
        for value in (
            protocol,
            protocol_spec,
            protocol_identity,
            prompt,
            task,
            task_identity,
            native,
        )
    ):
        return False

    def exact_int(value: Any, expected: int) -> bool:
        return (
            not isinstance(value, bool) and isinstance(value, int) and value == expected
        )

    for identity in (protocol_identity, task_identity):
        if (
            identity.get("task_name") != task_spec.task_name
            or not exact_int(
                identity.get("activity_definition_id"),
                task_spec.activity_definition_id,
            )
            or not exact_int(
                identity.get("activity_instance_id"),
                native_instance,
            )
        ):
            return False
    if not all(
        (
            protocol.get("behavior_phase") == "explore",
            exact_int(protocol.get("public_seed"), public_seed),
            protocol.get("recipe_tag") == recipe_tag,
            exact_int(task.get("task"), task_spec.task_index),
            task.get("task_name") == task_spec.task_name,
            exact_int(task.get("public_seed"), public_seed),
            exact_int(protocol_spec.get("task_index"), task_spec.task_index),
            protocol_spec.get("task_name") == task_spec.task_name,
            exact_int(
                protocol_spec.get("activity_definition_id"),
                task_spec.activity_definition_id,
            ),
            protocol_spec.get("prompt_profile_id") == task_spec.prompt_profile_id,
            prompt.get("profile_id") == task_spec.prompt_profile_id,
            exact_int(
                native.get("activity_definition_id"),
                task_spec.activity_definition_id,
            ),
            exact_int(native.get("activity_instance_id"), native_instance),
            exact_int(native.get("env_seed"), BEHAVIOR_NATIVE_ENV_SEED),
        )
    ):
        return False
    for key, expected in (
        ("task_index", task_spec.task_index),
        ("task_name", task_spec.task_name),
    ):
        if key not in protocol:
            continue
        if key == "task_index":
            if not exact_int(protocol[key], expected):
                return False
        elif protocol[key] != expected:
            return False
    return True


def _artifact_seal_complete(final_result: dict[str, Any] | None) -> bool:
    """Return an independent best-effort artifact diagnostic.

    Runtime cleanup is a conservative fallback for runtimes that have not yet
    emitted the explicit field. This value never participates in raw-success
    verification, retry control, or publication eligibility.
    """

    if not isinstance(final_result, dict):
        return False
    explicit = final_result.get("artifact_seal_complete")
    if isinstance(explicit, bool):
        return explicit
    return final_result.get("runtime_cleanup") == "complete"


def _usage_from_attempt(
    attempt_dir: Path,
    final_result: dict[str, Any] | None,
    *,
    elapsed_s: float,
    tag: str = TAG,
) -> dict[str, Any]:
    audit = _read_json(attempt_dir / f"{tag}.json") or {}
    return {
        "env_steps": int(audit.get("total_env_steps") or 0),
        "vla_chunks": int(audit.get("global_vla_chunks") or 0),
        "vla_invocations": int(audit.get("global_vla_invocations") or 0),
        "tool_calls": int(audit.get("global_tool_calls") or 0),
        "wall_clock_s": float((final_result or {}).get("elapsed_s") or elapsed_s),
    }


def _ensure_failed_archive(
    tag_root: Path,
    attempt_dir: Path,
    record: dict[str, Any],
    *,
    task_spec: BehaviorTaskSpec,
    public_seed: int,
    recipe_tag: str,
) -> None:
    """Create missing failure audit files without overwriting child artifacts."""

    stem = f"attempt_{int(record['attempt_index']):03d}_failed"
    audit_path = tag_root / f"{stem}.json"
    trace_path = tag_root / f"{stem}.jsonl"
    if not audit_path.exists():
        _atomic_json(
            audit_path,
            {
                "schema_version": 1,
                "attempt_index": int(record["attempt_index"]),
                "task_name": task_spec.task_name,
                "activity_definition_id": task_spec.activity_definition_id,
                "activity_instance_id": int(
                    (record.get("task_identity") or {}).get(
                        "activity_instance_id",
                        task_spec.instance_for_public_seed(public_seed),
                    )
                ),
                "public_seed": public_seed,
                "recipe_tag": recipe_tag,
                "task_success": False,
                "workflow_complete": False,
                "official_success_source": 'info["done"]["success"]',
                "outcome": record["outcome"],
                "summary": record["summary"],
                "terminal_failure": record.get("terminal_failure"),
                "subprocess_exit_code": record["subprocess_exit_code"],
                "timed_out": record["timed_out"],
                "outer_harness_fallback": True,
            },
        )
    if trace_path.exists():
        return
    child_trace = attempt_dir / "behavior_tool_trace.jsonl"
    if child_trace.is_file():
        shutil.copy2(child_trace, trace_path)
    else:
        _append_jsonl(
            trace_path,
            {
                "attempt_index": int(record["attempt_index"]),
                "outcome": record["outcome"],
                "summary": record["summary"],
                "terminal_failure": record.get("terminal_failure"),
                "outer_harness_fallback": True,
            },
        )


def _validate_resume_manifest(
    config: ExploreConfig,
    manifest: dict[str, Any],
) -> None:
    spec = _task_spec(config)
    protocol = manifest.get("protocol") or {}
    native = manifest.get("native_binding") or {}
    planner = manifest.get("planner") or {}
    expected_native = {
        "mapping_version": (
            spec.candidate_mapping_version
            if config.candidate_instance_id is not None
            else spec.mapping_version
        ),
        "activity_definition_id": spec.activity_definition_id,
        "activity_instance_id": _native_instance(config),
        "scene_model": spec.scene_model,
        "env_seed": BEHAVIOR_NATIVE_ENV_SEED,
    }
    if config.candidate_instance_id is not None:
        expected_native["state_sha256"] = config.candidate_state_sha256
    if (
        protocol.get("task_index") != spec.task_index
        or protocol.get("task_name") != spec.task_name
        or protocol.get("public_seed") != config.public_seed
        or protocol.get("recipe_tag") != _tag(config)
        or (protocol.get("task_spec") or {})
        != {
            "task_index": spec.task_index,
            "task_name": spec.task_name,
            "task_language": spec.task_language,
            "prompt_profile_id": spec.prompt_profile_id,
            "activity_definition_id": spec.activity_definition_id,
            "scene_model": spec.scene_model,
            "public_seed_to_instance": {
                str(seed): instance_id
                for seed, instance_id in spec.public_seed_to_instance.items()
            },
            "mapping_version": spec.mapping_version,
            "candidate_mapping_version": spec.candidate_mapping_version,
            "explore_public_seeds": list(spec.explore_public_seeds),
            "eval_public_seeds": list(spec.eval_public_seeds),
        }
    ):
        raise RuntimeError("resume manifest does not describe the Explore public cell")
    declared_contract_version = protocol.get("public_tool_contract_version")
    if declared_contract_version is None:
        source_contract_version = 1
    elif (
        isinstance(declared_contract_version, bool)
        or not isinstance(declared_contract_version, int)
        or declared_contract_version not in PUBLIC_TOOL_CONTRACTS
    ):
        raise RuntimeError(
            "resume manifest has an invalid public-tool contract version"
        )
    else:
        source_contract_version = declared_contract_version
    if source_contract_version != CURRENT_PUBLIC_TOOL_CONTRACT_VERSION:
        raise RuntimeError(
            "unfinished public-tool contract v1 Explore Jobs cannot resume "
            "under the v2 runtime"
        )
    expected_public_tools = PUBLIC_TOOL_CONTRACTS[source_contract_version]
    if tuple(protocol.get("public_primitives") or ()) != expected_public_tools:
        raise RuntimeError(
            "resume manifest public primitives do not match its declared contract"
        )
    if protocol.get("agent_finish_registered") is not False:
        raise RuntimeError("resume manifest unexpectedly exposes Agent finish")
    if protocol.get("pi0_nav_pick_contract") != pi0_nav_pick_exact_chunk_contract():
        raise RuntimeError("resume manifest Pi0 exact-chunk contract mismatch")
    if any(native.get(key) != value for key, value in expected_native.items()):
        raise RuntimeError("resume manifest native binding does not match this harness")
    if (
        planner.get("model") != config.model
        or planner.get("reasoning_effort") != config.reasoning_effort
    ):
        raise RuntimeError("resume planner identity differs from the original job")
    if manifest.get("policy_checkpoint") != _checkpoint_binding(config):
        raise RuntimeError(
            "resume shared policy checkpoint differs from the original job"
        )
    if manifest.get("resource_source") != _resource_binding(config).as_dict():
        raise RuntimeError(
            "resume pinned BEHAVIOR resources differ from the original job"
        )
    current_source = _checkout_identity(config.repo_root)
    previous_source = manifest.get("source") or {}
    for key in ("commit", "dirty_content_sha256"):
        if previous_source.get(key) != current_source.get(key):
            raise RuntimeError("resume checkout content differs from the original job")
    if manifest.get("reviewed_repo_memory") != _reviewed_memory_binding(config):
        raise RuntimeError(
            "resume reviewed BEHAVIOR Global Memory differs from the original job"
        )
    if manifest.get("reviewed_recipe_catalog") != (
        _reviewed_recipe_catalog_binding(config)
    ):
        raise RuntimeError(
            "resume reviewed BEHAVIOR Recipe Catalog differs from the original job"
        )
    expected_predecessor = (
        json.loads(
            json.dumps(
                config.epoch_predecessor_binding,
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        if config.epoch_predecessor_binding is not None
        else None
    )
    if manifest.get("epoch_predecessor") != expected_predecessor:
        raise RuntimeError("resume epoch predecessor binding differs from original job")


def _promote_success(
    job_root: Path,
    attempt_dir: Path,
    publication_eligible: bool,
    *,
    task_name: str = TASK_NAME,
) -> bool:
    del attempt_dir
    if not publication_eligible:
        return False
    try:
        publish_existing_success(job_root, task_name=task_name)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, UnicodeError):
        return False
    return True


def _safe_relative(root: Path, path: Path) -> Path:
    root = root.resolve(strict=True)
    candidate = path.absolute()
    relative = candidate.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"symlink is forbidden in publication path: {cursor}")
    resolved_parent = candidate.parent.resolve(strict=True)
    resolved_parent.relative_to(root)
    return relative


def _read_contained_regular_file(root: Path, path: Path) -> bytes:
    _safe_relative(root, path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"publication source is not a regular file: {path}")
    resolved = path.resolve(strict=True)
    resolved.relative_to(root.resolve(strict=True))
    return resolved.read_bytes()


def _read_json_contained(root: Path, path: Path) -> dict[str, Any]:
    value = json.loads(_read_contained_regular_file(root, path).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _read_json_lines_contained(root: Path, path: Path) -> list[dict[str, Any]]:
    records = []
    for line in _read_contained_regular_file(root, path).decode("utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON objects in {path}")
            records.append(value)
    return records


def _validate_publication_provenance(
    provenance: Any,
    *,
    task_spec: BehaviorTaskSpec,
    public_seed: int,
    recipe_tag: str,
    recipe: bytes,
    memory: bytes,
    job_id: str,
    attempt_index: int,
    official_binding: dict[str, Any],
    source_artifacts_sha256: dict[str, str],
) -> None:
    if not isinstance(provenance, dict):
        raise RuntimeError("publication provenance is not a JSON object")
    expected = {
        "task": task_spec.task_name,
        "public_seed": public_seed,
        "source_tag": recipe_tag,
        "success_source": 'info["done"]["success"]',
        "job_id": job_id,
        "attempt_index": attempt_index,
        "attempt_nonce": official_binding["attempt_nonce"],
        "task_success": True,
        "recipe_sha256": hashlib.sha256(recipe).hexdigest(),
        "memory_sha256": hashlib.sha256(memory).hexdigest(),
        "official_success_receipt": {
            key: official_binding[key]
            for key in (
                "source",
                "run_nonce",
                "attempt_nonce",
                "attempt_index",
                "env_step",
                "receipt_sha256",
                "file_sha256",
            )
        },
        "source_artifacts_sha256": source_artifacts_sha256,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise RuntimeError("publication provenance binding or digest mismatch")
    if provenance.get("source") != RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE:
        raise RuntimeError("publication provenance source mismatch")


def _task_memory_files_sha256(
    reviewed_repo_memory: dict[str, Any],
) -> dict[str, str]:
    """Return only the selected task-memory leaf hashes for publication."""

    selection = reviewed_repo_memory.get("selection")
    selected_files = selection.get("files") if isinstance(selection, dict) else None
    task_directory = (
        selection.get("task_directory") if isinstance(selection, dict) else None
    )
    if (
        not isinstance(task_directory, str)
        or not task_directory
        or not isinstance(selected_files, dict)
        or not selected_files
    ):
        raise RuntimeError(
            "reviewed Global Memory lacks a task-scoped publication selection"
        )
    result: dict[str, str] = {}
    task_prefix = f"{task_directory}/"
    for name, metadata in selected_files.items():
        digest = metadata.get("sha256") if isinstance(metadata, dict) else None
        if (
            not isinstance(name, str)
            or not name.startswith(task_prefix)
            or metadata.get("relative_path") != name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError(
                "reviewed Global Memory task selection has invalid file hashes"
            )
        result[name] = digest
    return result


def _install_derived_file(root: Path, path: Path, content: bytes) -> None:
    """Install a derived artifact once, or verify an identical prior result."""

    root = root.resolve(strict=True)
    path = path.absolute()
    path.relative_to(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_relative(root, path)
    if path.is_symlink():
        raise RuntimeError(f"refusing to replace publication symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite non-identical artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise RuntimeError(
                    f"concurrent non-identical publication artifact: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _commit_publication_group(
    root: Path,
    payloads: dict[str, bytes],
    *,
    amendment: bytes | None = None,
) -> str:
    """Stage a complete immutable bundle, then expose canonical files."""

    root = root.resolve(strict=True)
    bundle_id = canonical_bundle_id(payloads)
    bundles = root / ".publication_bundles"
    bundles.mkdir(exist_ok=True)
    _safe_relative(root, bundles)
    final_bundle = bundles / bundle_id
    if final_bundle.exists():
        if final_bundle.is_symlink() or not final_bundle.is_dir():
            raise RuntimeError("publication bundle path is not a safe directory")
        for name, content in payloads.items():
            if (
                _read_contained_regular_file(final_bundle, final_bundle / name)
                != content
            ):
                raise RuntimeError("existing publication bundle is non-identical")
    else:
        stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=bundles))
        try:
            for name, content in payloads.items():
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            try:
                os.rename(stage, final_bundle)
            except FileExistsError:
                pass
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    if final_bundle.is_symlink() or not final_bundle.is_dir():
        raise RuntimeError("publication bundle path is not a safe directory")
    for name, content in payloads.items():
        if _read_contained_regular_file(final_bundle, final_bundle / name) != content:
            raise RuntimeError("committed publication bundle is non-identical")
    for name, content in payloads.items():
        _install_derived_file(root, root / name, content)
    if amendment is not None:
        _install_derived_file(root, root / "publication_amendment.json", amendment)
    return bundle_id


def _validate_symbolic_publication(recipe: bytes, memory: bytes) -> None:
    """Reject replay programs, run-specific geometry, and prompt steering."""

    combined = recipe + b"\n" + memory
    if re.search(rb"/(?:home|tmp|mnt)/", combined):
        raise RuntimeError("publication contains an absolute runtime path")
    text = combined.decode("utf-8", errors="strict")
    quoted_tool_names = "|".join(
        re.escape(name) for name in sorted(_SYMBOLIC_PUBLIC_TOOL_DENYLIST)
    )
    forbidden_text = re.compile(
        r"(?i)(?:"
        rf"`(?:{quoted_tool_names})`|"
        r"\b(?:pi0_nav_pick|pixel_to_world|navigate_to|"
        r"save_robot_state_checkpoint|"
        r"vla instruction|camera order|fixed hand|tool sequence)\b|"
        r"\b(?:first|second|third|next|then|finally|subsequently|afterwards)\b|"
        r"\bstep\s*(?:#\s*)?\d+\b|"
        r"\b(?:left|right)[-_ ]?(?:hand|arm|wrist)\b|"
        r"\b[xyzuv]\s*[:=]\s*[-+]?\d+(?:\.\d+)?"
        r")"
    )
    if forbidden_text.search(text):
        raise RuntimeError("publication contains tool, order, or spatial steering")
    records = [
        json.loads(line) for line in recipe.decode("utf-8").splitlines() if line.strip()
    ]
    forbidden_keys = {
        "pixel",
        "row",
        "col",
        "frame_id",
        "xyz",
        "pose",
        "qpos",
        "checkpoint_path",
        "activity_instance_id",
        "native_instance",
        "seed",
        "held_hand",
        "press_hand",
        "tool",
        "tool_name",
        "tools",
        "tool_sequence",
        "sequence",
        "steps",
        "instruction",
        "vla_instruction",
        "camera_order",
        "fixed_hand",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in forbidden_keys:
                    raise RuntimeError(f"publication contains forbidden field: {key}")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(records)


def _owned_process_is_alive(record: Any) -> bool:
    if not isinstance(record, dict) or record.get("managed") is not True:
        return False
    pid = record.get("pid")
    start_ticks = record.get("start_ticks")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not isinstance(start_ticks, int)
        or isinstance(start_ticks, bool)
    ):
        raise RuntimeError("attempt process identity is incomplete")
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return False
    try:
        raw = stat_path.read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        current_start_ticks = int(fields[19])
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError("attempt process identity is ambiguous") from error
    return current_start_ticks == start_ticks


def publish_existing_success(
    job_root: str | Path,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Idempotently derive publication artifacts without changing an attempt.

    This entry accepts only a runtime-owned raw-success receipt and writes a
    separate, hash-bound publication amendment at the Job root.
    """

    requested_root = Path(job_root).expanduser().absolute()
    if requested_root.is_symlink():
        raise RuntimeError("offline publication Job root must not be a symlink")
    root = requested_root.resolve(strict=True)
    lock_path = root / ".publication.lock"
    if lock_path.is_symlink():
        raise RuntimeError("offline publication lock must not be a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another offline publisher owns this Job") from error
        return _publish_existing_success_locked(root, task_name=task_name)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _forensic_receipt_payload(
    *,
    root: Path,
    manifest: dict[str, Any],
    attempt_index: int,
    binding: dict[str, Any],
) -> dict[str, Any]:
    """Build a trace-bound receipt without claiming runtime provenance."""

    job_id = manifest.get("job_id")
    publication_identity = (
        {
            "job_id": job_id,
            "attempt_index": attempt_index,
        }
        if isinstance(job_id, str) and job_id
        else {
            "job_root_path": str(root),
            "attempt_index": attempt_index,
        }
    )
    success_evidence = {
        "source": "behavior_action_trace",
        "field_path": binding["field_path"],
        "first_success_step": binding["first_success_step"],
        "action_trace_sha256": binding["action_trace_sha256"],
    }
    receipt = {
        "schema_version": 1,
        "receipt_type": "forensic_action_trace_receipt",
        "source": "behavior_action_trace",
        "job_id": job_id,
        "job_root_path": str(root),
        "attempt_index": attempt_index,
        "field_path": binding["field_path"],
        "first_success_step": binding["first_success_step"],
        "action_trace_sha256": binding["action_trace_sha256"],
        "publication_identity": publication_identity,
        "official_success_binding": success_evidence,
        "created_at": _utc_now(),
    }
    unsigned = json.dumps(
        receipt,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return receipt


def _forensic_summary_target(root: Path, path: Path) -> Path:
    """Validate one correction target and reject legacy fixed temp paths."""

    _safe_relative(root, path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RuntimeError(f"forensic summary target is unsafe: {path}")
    legacy_temp = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(legacy_temp):
        raise RuntimeError(
            f"forensic correction refuses a pre-existing legacy temp path: "
            f"{legacy_temp}"
        )
    _safe_relative(root, legacy_temp)
    return path


def _forensic_atomic_json(
    root: Path,
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically write JSON through a random, exclusive, no-follow temp file."""

    _forensic_summary_target(root, path)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    parent = path.parent
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_descriptor = os.open(parent, directory_flags)
    temporary_name = f".{path.name}.forensic-{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            temporary_name,
            file_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        created = True
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        created = False
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def _forensic_attempts(
    root: Path,
    *,
    recipe_tag: str,
) -> list[tuple[int, Path, dict[str, Any], bytes]]:
    """Return attempt directories whose canonical action trace proves success."""

    tag_root = root / "attempts" / recipe_tag
    _safe_relative(root, tag_root)
    if tag_root.is_symlink() or not tag_root.is_dir():
        raise RuntimeError("forensic correction attempt root is unsafe or missing")
    successful = []
    for attempt_dir in sorted(tag_root.iterdir()):
        match = re.fullmatch(r"attempt_(\d{3})", attempt_dir.name)
        if match is None or not attempt_dir.is_dir():
            continue
        if attempt_dir.is_symlink():
            raise RuntimeError("forensic correction attempt must not be a symlink")
        action_path = attempt_dir / "behavior_action_trace.jsonl"
        if not action_path.exists():
            continue
        action_bytes = _read_contained_regular_file(attempt_dir, action_path)
        binding = summarize_action_trace_success(action_bytes)
        if binding is not None:
            successful.append((int(match.group(1)), attempt_dir, binding, action_bytes))
    return successful


def _forensic_publication_complete(
    root: Path,
    attempt_dir: Path,
    binding: dict[str, Any],
) -> bool:
    """Keep task-success correction independent from publication readiness."""

    try:
        validation = validate_forensic_publication_binding(
            root,
            attempt_dir,
            binding,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return getattr(validation, "complete", False) is True


def _require_stable_forensic_action_trace(
    attempt_dir: Path,
    expected_bytes: bytes,
    expected_sha256: str,
) -> None:
    action_path = attempt_dir / "behavior_action_trace.jsonl"
    current = _read_contained_regular_file(attempt_dir, action_path)
    if (
        current != expected_bytes
        or hashlib.sha256(current).hexdigest() != expected_sha256
    ):
        raise RuntimeError("forensic action trace changed during correction")


def _correct_existing_success_locked(
    root: Path,
    *,
    task_name: str | None,
) -> dict[str, Any]:
    manifest_path = root / "session_manifest.json"
    manifest = _read_json_contained(root, manifest_path)
    if manifest.get("status") in {"starting", "running"}:
        raise RuntimeError("forensic correction refuses an active Job")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("forensic correction requires Job protocol identity")
    manifest_task_name = protocol.get("task_name")
    if not isinstance(manifest_task_name, str):
        raise RuntimeError("forensic correction task identity is missing")
    if task_name is not None and task_name != manifest_task_name:
        raise RuntimeError("forensic correction task identity mismatch")
    try:
        task_spec = get_task_spec(manifest_task_name)
    except ValueError as error:
        raise RuntimeError(
            "forensic correction task identity is unsupported"
        ) from error
    public_seed = protocol.get("public_seed")
    if isinstance(public_seed, bool) or not isinstance(public_seed, int):
        raise RuntimeError("forensic correction public seed is invalid")
    recipe_tag = task_spec.tag(public_seed)
    if protocol.get("recipe_tag") != recipe_tag:
        raise RuntimeError("forensic correction recipe tag identity mismatch")

    successful = _forensic_attempts(root, recipe_tag=recipe_tag)
    if len(successful) != 1:
        raise RuntimeError(
            "forensic correction requires exactly one canonical successful attempt"
        )
    attempt_index, attempt_dir, trace_binding, action_trace_bytes = successful[0]
    binding = json.loads(json.dumps(trace_binding))
    if (
        binding.get("source") != "behavior_action_trace"
        or binding.get("field_path") != "info_done.success"
    ):
        raise RuntimeError("forensic correction requires canonical info_done.success")

    attempts = manifest.get("attempts")
    if not isinstance(attempts, list):
        raise RuntimeError("forensic correction attempts ledger is invalid")
    matching_records = [
        (index, item)
        for index, item in enumerate(attempts)
        if isinstance(item, dict) and item.get("attempt_index") == attempt_index
    ]
    if len(matching_records) > 1:
        raise RuntimeError("forensic correction attempt ledger is ambiguous")
    if matching_records:
        manifest_attempt_list_index, attempt_record = matching_records[0]
    else:
        manifest_attempt_list_index = len(attempts)
        attempt_record = {
            "attempt_index": attempt_index,
            "output_dir": str(attempt_dir),
        }
        attempts.append(attempt_record)

    receipt_path = attempt_dir / "official_success_receipt.json"
    manifest_target = _forensic_summary_target(root, manifest_path)
    summary_paths = (
        attempt_dir / "final_result.json",
        attempt_dir / "behavior_result.json",
        attempt_dir / f"{recipe_tag}.json",
    )
    for path in summary_paths:
        _forensic_summary_target(root, path)
    _forensic_summary_target(root, receipt_path)
    failed_paths = tuple(
        attempt_dir.parent / f"attempt_{attempt_index:03d}_failed{suffix}"
        for suffix in (".json", ".jsonl")
    )
    for failed_path in failed_paths:
        _safe_relative(root, failed_path)
        if failed_path.is_symlink() or (
            failed_path.exists() and not failed_path.is_file()
        ):
            raise RuntimeError("forensic failed archive target is unsafe")

    create_receipt = not receipt_path.exists()
    if receipt_path.exists():
        receipt_bytes = _read_contained_regular_file(attempt_dir, receipt_path)
        try:
            receipt = json.loads(receipt_bytes.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "existing official success receipt is not valid JSON"
            ) from error
        if not isinstance(receipt, dict):
            raise RuntimeError(
                "existing official success receipt must be a JSON object"
            )
    else:
        receipt = _forensic_receipt_payload(
            root=root,
            manifest=manifest,
            attempt_index=attempt_index,
            binding=binding,
        )
        receipt_bytes = (
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    binding["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    publication_complete = _forensic_publication_complete(
        root,
        attempt_dir,
        binding,
    )
    _require_stable_forensic_action_trace(
        attempt_dir,
        action_trace_bytes,
        binding["action_trace_sha256"],
    )

    corrected_attempt = dict(attempt_record)
    corrected_attempt.update(
        {
            "outcome": "official_success",
            "task_success": True,
            "summary": "Official raw success confirmed from bound action trace.",
            "terminal_failure": None,
            "publication_eligible": publication_complete,
            "official_success_binding": binding,
            "official_success_receipt": receipt,
        }
    )
    attempts[manifest_attempt_list_index] = corrected_attempt
    corrected_manifest = dict(manifest)
    corrected_manifest.update(
        {
            "status": "succeeded",
            "task_success": True,
            "publication_complete": publication_complete,
            "official_success_binding": binding,
            "official_success_receipt": receipt,
            "attempts": attempts,
        }
    )

    common_updates = {
        "task_success": True,
        "success": True,
        "official_success_source": "behavior_action_trace",
        "official_success_binding": binding,
        "official_success_receipt": receipt,
    }
    corrected_summaries: list[tuple[Path, dict[str, Any]]] = []
    for path in summary_paths:
        current = _read_json_contained(root, path) if path.exists() else {}
        corrected = dict(current)
        corrected.update(common_updates)
        corrected_summaries.append((path, corrected))

    if create_receipt:
        _forensic_atomic_json(root, receipt_path, receipt)
    for path, corrected in corrected_summaries:
        _forensic_atomic_json(root, path, corrected)
    _forensic_atomic_json(root, manifest_target, corrected_manifest)

    if _read_contained_regular_file(attempt_dir, receipt_path) != receipt_bytes:
        raise RuntimeError("forensic receipt verification failed")
    for path, _ in corrected_summaries:
        verified = _read_json_contained(root, path)
        if (
            verified.get("task_success") is not True
            or verified.get("success") is not True
            or verified.get("official_success_binding") != binding
            or verified.get("official_success_receipt") != receipt
        ):
            raise RuntimeError(f"forensic summary verification failed: {path}")
    verified_manifest = _read_json_contained(root, manifest_path)
    verified_attempts = verified_manifest.get("attempts")
    if (
        verified_manifest.get("status") != "succeeded"
        or verified_manifest.get("task_success") is not True
        or verified_manifest.get("publication_complete") is not publication_complete
        or verified_manifest.get("official_success_binding") != binding
        or verified_manifest.get("official_success_receipt") != receipt
        or not isinstance(verified_attempts, list)
        or verified_attempts[manifest_attempt_list_index].get("outcome")
        != "official_success"
    ):
        raise RuntimeError("forensic session manifest verification failed")

    _require_stable_forensic_action_trace(
        attempt_dir,
        action_trace_bytes,
        binding["action_trace_sha256"],
    )
    for failed_path in failed_paths:
        failed_path.unlink(missing_ok=True)
    return corrected_manifest


def correct_existing_success(
    job_root: str | Path,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Correct summary lifecycle state from one immutable canonical action trace."""

    requested_root = Path(job_root).expanduser().absolute()
    if requested_root.is_symlink():
        raise RuntimeError("forensic correction Job root must not be a symlink")
    root = requested_root.resolve(strict=True)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(root, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns this forensic Job root") from error
        return _correct_existing_success_locked(root, task_name=task_name)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _publish_existing_success_locked(
    root: Path,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    """Publish from one immutable source snapshot while holding the Job lock."""

    manifest_path = root / "session_manifest.json"
    manifest_bytes = _read_contained_regular_file(root, manifest_path)
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("task_success") is not True:
        raise RuntimeError("offline publication requires a successful Job manifest")
    if (
        manifest.get("status") != "succeeded"
        or not isinstance(manifest.get("finished_at"), str)
        or manifest.get("blocked_reason") is not None
    ):
        raise RuntimeError("offline publication requires a cleanly terminated Job")
    job_id = str(manifest.get("job_id") or "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", job_id):
        raise RuntimeError("offline publication Job identity is invalid")
    protocol = manifest.get("protocol") or {}
    manifest_task_name = protocol.get("task_name")
    if not isinstance(manifest_task_name, str):
        raise RuntimeError("offline publication task identity is missing")
    if task_name is not None and manifest_task_name != task_name:
        raise RuntimeError("offline publication task identity mismatch")
    try:
        task_spec = get_task_spec(manifest_task_name)
    except ValueError as error:
        raise RuntimeError(
            "offline publication task identity is unsupported"
        ) from error
    public_seed = protocol.get("public_seed")
    if (
        isinstance(public_seed, bool)
        or not isinstance(public_seed, int)
        or public_seed not in task_spec.explore_public_seeds
    ):
        raise RuntimeError("offline publication requires a task Explore public seed")
    recipe_tag = task_spec.tag(public_seed)
    if (
        protocol.get("behavior_phase") != "explore"
        or protocol.get("task_index") != task_spec.task_index
        or protocol.get("recipe_tag") != recipe_tag
    ):
        raise RuntimeError("offline publication requires the canonical Explore cell")
    native_binding = manifest.get("native_binding") or {}
    expected_instance = task_spec.instance_for_public_seed(
        public_seed,
        phase="explore",
    )
    if any(
        native_binding.get(key) != value
        for key, value in {
            "mapping_version": task_spec.mapping_version,
            "activity_definition_id": task_spec.activity_definition_id,
            "activity_instance_id": expected_instance,
            "scene_model": task_spec.scene_model,
            "env_seed": BEHAVIOR_NATIVE_ENV_SEED,
        }.items()
    ):
        raise RuntimeError("offline publication native task identity mismatch")
    successful = [
        item
        for item in manifest.get("attempts", [])
        if isinstance(item, dict) and item.get("task_success") is True
    ]
    if len(successful) != 1:
        raise RuntimeError(
            "offline publication requires exactly one successful attempt"
        )
    successful_record = successful[0]
    if successful_record.get("outcome") != "official_success" or successful_record.get(
        "forced_cleanup_groups"
    ) not in ({}, None):
        raise RuntimeError("successful attempt cleanup or outcome is not canonical")
    attempt_index = int(successful[0]["attempt_index"])
    attempt_dir_lexical = (
        root / "attempts" / recipe_tag / f"attempt_{attempt_index:03d}"
    )
    _safe_relative(root, attempt_dir_lexical)
    if attempt_dir_lexical.is_symlink() or not attempt_dir_lexical.is_dir():
        raise RuntimeError("successful attempt directory is unsafe or missing")
    attempt_dir = attempt_dir_lexical.resolve(strict=True)
    attempt_dir.relative_to(root)
    declared_output = successful_record.get("output_dir")
    if (
        not isinstance(declared_output, str)
        or Path(declared_output).expanduser().resolve() != attempt_dir
    ):
        raise RuntimeError("successful attempt output binding mismatch")
    final_path = attempt_dir / "final_result.json"
    trace_path = attempt_dir / "behavior_tool_trace.jsonl"
    action_trace_path = attempt_dir / "behavior_action_trace.jsonl"
    receipt_path = attempt_dir / "official_success_receipt.json"
    run_manifest_path = attempt_dir / "run_manifest.json"
    final_bytes = _read_contained_regular_file(attempt_dir, final_path)
    trace_bytes = _read_contained_regular_file(attempt_dir, trace_path)
    action_trace_bytes = _read_contained_regular_file(attempt_dir, action_trace_path)
    receipt_bytes = _read_contained_regular_file(attempt_dir, receipt_path)
    run_manifest_bytes = _read_contained_regular_file(attempt_dir, run_manifest_path)
    final_result = json.loads(final_bytes.decode("utf-8"))
    if (
        not isinstance(final_result, dict)
        or final_result.get("task_success") is not True
        or final_result.get("official_success_source") != _OFFICIAL_SUCCESS_SOURCE
    ):
        raise RuntimeError("successful attempt lacks verified runtime raw success")
    final_job = final_result.get("job") or {}
    if (
        final_job.get("job_id") != job_id
        or final_job.get("attempt_index") != attempt_index
        or final_result.get("runtime_cleanup") != "complete"
    ):
        raise RuntimeError("successful attempt Job or cleanup binding mismatch")
    run_manifest = json.loads(run_manifest_bytes.decode("utf-8"))
    if not isinstance(run_manifest, dict):
        raise RuntimeError("successful attempt run manifest is invalid")
    run_job = run_manifest.get("job") or {}
    if run_job.get("job_id") != job_id or run_job.get("attempt_index") != attempt_index:
        raise RuntimeError("successful attempt run-manifest binding mismatch")
    reviewed_repo_memory = manifest.get("reviewed_repo_memory")
    if (
        not isinstance(reviewed_repo_memory, dict)
        or not isinstance(reviewed_repo_memory.get("snapshot_sha256"), str)
        or run_manifest.get("reviewed_repo_memory") != reviewed_repo_memory
    ):
        raise RuntimeError("successful attempt reviewed Global Memory binding mismatch")
    reviewed_recipe_catalog = manifest.get("reviewed_recipe_catalog")
    if reviewed_recipe_catalog is not None:
        if (
            not isinstance(reviewed_recipe_catalog, dict)
            or not isinstance(reviewed_recipe_catalog.get("catalog_sha256"), str)
            or run_manifest.get("reviewed_recipe_catalog")
            != _runtime_recipe_catalog_binding(reviewed_recipe_catalog)
        ):
            raise RuntimeError("successful attempt Recipe Catalog binding mismatch")
    env_process = (run_manifest.get("processes") or {}).get("env")
    if (
        run_manifest.get("status") != "stopped"
        or not isinstance(env_process, dict)
        or not isinstance(env_process.get("stopped_at"), str)
        or _owned_process_is_alive(env_process)
    ):
        raise RuntimeError("successful attempt environment is not fully stopped")
    owned_groups = _manifest_owned_groups(attempt_dir)
    unverified_groups = _manifest_unverified_groups(attempt_dir)
    if owned_groups or unverified_groups:
        raise RuntimeError(
            "successful attempt still owns or ambiguously references processes"
        )
    trace = []
    for line in trace_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise RuntimeError("successful attempt tool trace is invalid")
        trace.append(record)
    if not trace:
        raise RuntimeError("successful attempt lacks a tool trace")
    official_binding = _official_success_binding(attempt_dir)
    if official_binding is None:
        raise RuntimeError("successful attempt lacks a bound raw-success receipt")
    attempt_protocol = run_manifest.get("protocol")
    if (
        not isinstance(attempt_protocol, dict)
        or attempt_protocol.get("pi0_nav_pick_contract")
        != pi0_nav_pick_exact_chunk_contract()
        or not _exact_pi0_call_artifacts_valid(attempt_dir, trace)
    ):
        raise RuntimeError(
            "successful attempt lacks publication-eligible exact Pi0 artifacts"
        )
    attempt_nonce = official_binding["attempt_nonce"]
    source = RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE
    source_artifacts_sha256 = {
        "official_success_receipt": hashlib.sha256(receipt_bytes).hexdigest(),
        "behavior_action_trace": hashlib.sha256(action_trace_bytes).hexdigest(),
        "behavior_tool_trace": hashlib.sha256(trace_bytes).hexdigest(),
        "final_result": hashlib.sha256(final_bytes).hexdigest(),
        "run_manifest": hashlib.sha256(run_manifest_bytes).hexdigest(),
        "session_manifest": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    stage = Path(tempfile.mkdtemp(prefix=".publication-", dir=root))
    try:
        fake = object.__new__(BehaviorToolkit)
        fake._tool_trace = trace
        fake._task_spec = task_spec
        fake._primitives = SimpleNamespace(
            attempt_index=attempt_index,
            output_dir=stage,
            job_id=job_id,
            attempt_nonce=attempt_nonce,
            run_nonce=official_binding["run_nonce"],
            public_seed=public_seed,
        )
        records = fake._symbolic_recipe()
        BehaviorToolkit.validate_symbolic_publication(records)
        recipe_path = stage / f"recipe_{recipe_tag}.jsonl"
        BehaviorToolkit._write_json_atomic(recipe_path, records, json_lines=True)
        fake._publish_task_memory(
            recipe_tag=recipe_tag,
            recipe_path=recipe_path,
            official_success_receipt={
                key: official_binding[key]
                for key in (
                    "source",
                    "run_nonce",
                    "attempt_nonce",
                    "attempt_index",
                    "env_step",
                    "receipt_sha256",
                    "file_sha256",
                )
            },
            source_artifacts_sha256=source_artifacts_sha256,
        )
        staged_memory = stage / "memory" / f"{task_spec.task_name}.md"
        staged_provenance = stage / "memory" / f"{task_spec.task_name}_provenance.json"
        recipe_bytes = recipe_path.read_bytes()
        memory_bytes = staged_memory.read_bytes()
        _validate_symbolic_publication(recipe_bytes, memory_bytes)
        provenance = json.loads(staged_provenance.read_text(encoding="utf-8"))
        provenance.pop("workflow_complete", None)
        provenance["schema_version"] = 3
        provenance["derived_offline"] = True
        provenance.update(
            {
                "task": task_spec.task_name,
                "public_seed": public_seed,
                "source_tag": recipe_tag,
                "source": source,
                "success_source": _OFFICIAL_SUCCESS_SOURCE,
                "job_id": job_id,
                "attempt_index": attempt_index,
                "attempt_nonce": attempt_nonce,
                "task_success": True,
                "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
                "memory_sha256": hashlib.sha256(memory_bytes).hexdigest(),
                "official_success_receipt": {
                    key: official_binding[key]
                    for key in (
                        "source",
                        "run_nonce",
                        "attempt_nonce",
                        "attempt_index",
                        "env_step",
                        "receipt_sha256",
                        "file_sha256",
                    )
                },
                "source_artifacts_sha256": source_artifacts_sha256,
                "global_memory_snapshot_sha256": reviewed_repo_memory[
                    "snapshot_sha256"
                ],
                "global_memory_files_sha256": _task_memory_files_sha256(
                    reviewed_repo_memory
                ),
            }
        )
        provenance_bytes = (
            json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _validate_publication_provenance(
            provenance,
            task_spec=task_spec,
            public_seed=public_seed,
            recipe_tag=recipe_tag,
            recipe=recipe_bytes,
            memory=memory_bytes,
            job_id=job_id,
            attempt_index=attempt_index,
            official_binding=official_binding,
            source_artifacts_sha256=source_artifacts_sha256,
        )
        publication_payloads = {
            recipe_path.name: recipe_bytes,
            f"memory/{staged_memory.name}": memory_bytes,
            f"memory/{staged_provenance.name}": provenance_bytes,
        }
        bundle_id = canonical_bundle_id(publication_payloads)
        amendment = {
            "schema_version": 2,
            "kind": "posthoc_publication_override",
            "job_id": job_id,
            "task": task_spec.task_name,
            "tag": recipe_tag,
            "public_seed": public_seed,
            "success_source": 'info["done"]["success"]',
            "task_success": True,
            "publication_complete": True,
            "publication_source": source,
            "attempt_index": attempt_index,
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "memory_sha256": hashlib.sha256(memory_bytes).hexdigest(),
            "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            "bundle_id": bundle_id,
            "global_memory_snapshot_sha256": reviewed_repo_memory["snapshot_sha256"],
            "original_attempt_immutable": True,
            "artifact_seal_complete": bool(manifest.get("artifact_seal_complete")),
            "overlay_semantics": {
                "overrides": {"publication_complete": True},
                "preserves_session_manifest": {
                    "task_success": True,
                    "artifact_seal_complete": bool(
                        manifest.get("artifact_seal_complete")
                    ),
                    "workflow_complete": bool(manifest.get("workflow_complete")),
                    "publication_complete": bool(manifest.get("publication_complete")),
                },
            },
        }
        amendment_bytes = (
            json.dumps(amendment, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        source_snapshot = {
            manifest_path: manifest_bytes,
            final_path: final_bytes,
            trace_path: trace_bytes,
            action_trace_path: action_trace_bytes,
            receipt_path: receipt_bytes,
            run_manifest_path: run_manifest_bytes,
        }
        for source_path, expected_bytes in source_snapshot.items():
            source_root = root if source_path == manifest_path else attempt_dir
            if _read_contained_regular_file(source_root, source_path) != expected_bytes:
                raise RuntimeError(
                    "offline publication source changed during derivation"
                )
        _commit_publication_group(
            root,
            publication_payloads,
            amendment=amendment_bytes,
        )
        validate_canonical_publication_root(
            root,
            expected_provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
            expected_job_id=job_id,
            task_name=task_spec.task_name,
            task_index=task_spec.task_index,
            public_seed=public_seed,
        )
        return amendment
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _append_attempt_trace(
    job_root: Path,
    attempt_dir: Path,
    attempt_index: int,
    *,
    tag: str = TAG,
) -> None:
    source = attempt_dir / "behavior_tool_trace.jsonl"
    target = job_root / "traces" / f"{tag}.jsonl"
    if not source.is_file():
        return
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            record["attempt_index"] = attempt_index
            _append_jsonl(target, record)


def run_explore_job(
    config: ExploreConfig,
    *,
    dependencies: ExploreDependencies | None = None,
) -> dict[str, Any]:
    """Run indefinitely until raw official success or a safety block."""

    _validate_config(config)
    spec = _task_spec(config)
    tag = _tag(config)
    deps = dependencies or default_dependencies()
    root = config.output_root.resolve()
    manifest_path = root / "session_manifest.json"
    gpu_lock_descriptor, gpu_lock_path = _claim_gpu_lock(config.cuda_device)
    try:
        lock_descriptor, lock_path = _claim_job_lock(root)
    except BaseException:
        _release_job_lock(gpu_lock_descriptor)
        raise
    try:
        if config.resume:
            manifest = _read_json(manifest_path)
            if not isinstance(manifest, dict) or manifest.get("task_success") is True:
                raise RuntimeError("--resume requires one unfinished session manifest")
            if isinstance(manifest.get("terminal_failure"), dict):
                raise RuntimeError(
                    "--resume is forbidden after a terminal visual task failure"
                )
            _validate_resume_manifest(config, manifest)
            job_id = str(manifest["job_id"])
            prior = [
                item
                for item in manifest.get("attempts", [])
                if isinstance(item, dict) and item.get("task_success") is False
            ]
            attempt_index = int(manifest.get("cumulative", {}).get("attempts", 0)) + 1
        else:
            if root.exists() and any(root.iterdir()):
                raise RuntimeError(f"output root must be absent or empty: {root}")
            root.mkdir(parents=True, exist_ok=True)
            job_id = (
                f"behavior-explore-{time.strftime('%Y%m%dT%H%M%S')}-"
                f"{secrets.token_hex(4)}"
            )
            manifest = _new_manifest(config, job_id)
            prior = sanitize_prior_attempt_summaries(config.initial_prior_summaries)
            attempt_index = 1
            _atomic_json(manifest_path, manifest)
        _ensure_resource_source_file(config)
        _ensure_checkpoint_binding_file(config)
        dashboard_server, dashboard, dashboard_url = deps.start_dashboard(
            config, job_id
        )
    except BaseException:
        _release_job_lock(lock_descriptor)
        _release_job_lock(gpu_lock_descriptor)
        raise
    manifest["locks"] = {
        "job": str(lock_path),
        "gpu": str(gpu_lock_path),
    }
    manifest["dashboard_url"] = dashboard_url
    vla_proc: subprocess.Popen[Any] | None = None
    vla_endpoint: str | None = None
    try:
        vla_root = root / "vla"
        vla_root.mkdir(parents=True, exist_ok=True)
        vla_endpoint, vla_proc = deps.start_vla(config, vla_root)
        manifest["processes"]["vla"] = _process_record(vla_proc, vla_endpoint)
        if not deps.owns_vla:
            manifest["processes"]["vla"]["managed"] = False
            manifest["processes"]["vla"]["ownership"] = "borrowed"
        manifest["status"] = "running"
        _atomic_json(manifest_path, manifest)

        while True:
            current_resource_binding = verify_pinned_dataset_resources(
                _resource_binding(config)
            )
            if current_resource_binding.as_dict() != manifest.get("resource_source"):
                raise RuntimeError(
                    "pinned BEHAVIOR resource source changed during Explore"
                )
            _ensure_resource_source_file(config)
            current_reviewed_memory = _reviewed_memory_binding(config)
            if current_reviewed_memory != manifest.get("reviewed_repo_memory"):
                raise RuntimeError(
                    "reviewed BEHAVIOR Global Memory changed during Explore"
                )
            current_recipe_catalog = _reviewed_recipe_catalog_binding(config)
            if current_recipe_catalog != manifest.get("reviewed_recipe_catalog"):
                raise RuntimeError(
                    "reviewed BEHAVIOR Recipe Catalog changed during Explore"
                )
            free = deps.free_disk_bytes(root)
            if free < int(config.min_free_disk_gb * 1024**3):
                manifest["status"] = "blocked"
                manifest["task_success"] = None
                manifest["blocked_reason"] = "insufficient_disk_space"
                _atomic_json(manifest_path, manifest)
                return manifest
            if vla_proc.poll() is not None:
                manifest["processes"]["vla"]["stopped_at"] = _utc_now()
                manifest["processes"]["vla"]["returncode"] = vla_proc.returncode
                _append_jsonl(
                    root / "events.jsonl",
                    {
                        "type": "vla_exited",
                        "at": _utc_now(),
                        "attempt_index": attempt_index,
                    },
                )
                manifest["status"] = "blocked"
                manifest["task_success"] = None
                manifest["blocked_reason"] = "persistent_vla_exited"
                _atomic_json(manifest_path, manifest)
                return manifest
            vla_binding_id = f"{job_id}.a{attempt_index}.{secrets.token_hex(8)}"
            bound = deps.bind_attempt_vla(vla_endpoint, vla_binding_id)
            expected_binding_digest = hashlib.sha256(
                vla_binding_id.encode("utf-8")
            ).hexdigest()
            if (
                bound.get("actions_enabled") is not False
                or bound.get("binding_digest") != expected_binding_digest
            ):
                raise RuntimeError("persistent VLA attempt binding was not confirmed")

            attempt_tag = _job_tag(config)
            tag_root = root / "attempts" / attempt_tag
            attempt_dir = tag_root / f"attempt_{attempt_index:03d}"
            summary_file = (
                root / "prior_attempt_summaries" / f"attempt_{attempt_index:03d}.json"
            )
            summary_payload = {
                "job_id": job_id,
                "next_attempt_index": attempt_index,
                "lineage_scope": (
                    "campaign_prior"
                    if attempt_index == 1 and config.initial_prior_summaries
                    else "same_job_prior"
                ),
                "summaries": sanitize_prior_attempt_summaries(prior),
            }
            _atomic_json(summary_file, summary_payload)
            argv = build_attempt_argv(
                config,
                job_id=job_id,
                attempt_index=attempt_index,
                output_dir=attempt_dir,
                summaries_path=summary_file,
                vla_endpoint=vla_endpoint,
                vla_binding_id=vla_binding_id,
                reviewed_memory_snapshot_sha256=current_reviewed_memory[
                    "snapshot_sha256"
                ],
                recipe_catalog_sha256=current_recipe_catalog["catalog_sha256"],
            )
            if dashboard is not None:
                dashboard.on_job_progress(
                    {
                        **manifest["cumulative"],
                        "job_unlimited": config.max_attempts is None,
                    }
                )
                dashboard.begin_attempt(
                    attempt_index=attempt_index,
                    output_dir=attempt_dir,
                    video_path=attempt_dir / "episode.mp4",
                )
                dashboard.on_event(
                    {"type": "env_restart_started", "attempt_index": attempt_index}
                )
            attempt_started_at = _utc_now()
            _append_jsonl(
                root / "events.jsonl",
                {
                    "type": "attempt_started",
                    "at": attempt_started_at,
                    "attempt_index": attempt_index,
                },
            )
            _append_jsonl(
                root / "events.jsonl",
                {
                    "type": "env_restart_started",
                    "at": attempt_started_at,
                    "attempt_index": attempt_index,
                },
            )
            relay = (
                _AttemptDashboardRelay(attempt_dir, dashboard)
                if dashboard is not None
                else None
            )
            if relay is not None:
                relay.start()
            started = time.monotonic()
            try:
                execution = deps.run_attempt(
                    argv,
                    attempt_dir,
                    root / "launcher_logs" / f"attempt_{attempt_index:03d}.log",
                    config.attempt_timeout_s,
                )
            finally:
                if relay is not None:
                    relay.stop()
            elapsed = time.monotonic() - started
            if execution.alive_after_cleanup or execution.ambiguous_groups:
                manifest["status"] = "blocked"
                manifest["task_success"] = None
                manifest["blocked_reason"] = "attempt_process_ownership_unverified"
                _atomic_json(manifest_path, manifest)
                return manifest

            final_result = _runtime_attempt_result(
                attempt_dir,
                execution.final_result,
                tag=tag,
            )
            _ensure_resource_source_file(config)
            attempt_run_manifest = _read_json(attempt_dir / "run_manifest.json")
            if not isinstance(attempt_run_manifest, dict) or attempt_run_manifest.get(
                "resource_source"
            ) != manifest.get("resource_source"):
                raise RuntimeError(
                    "attempt resource source differs from the pinned Job binding"
                )
            attempt_identity_valid = _attempt_run_identity_valid(
                attempt_run_manifest,
                task_spec=spec,
                public_seed=config.public_seed,
                recipe_tag=tag,
                native_instance=_native_instance(config),
            )
            attempt_protocol = (
                attempt_run_manifest.get("protocol")
                if isinstance(attempt_run_manifest, dict)
                else None
            )
            exact_chunk_contract_valid = bool(
                isinstance(attempt_protocol, dict)
                and attempt_protocol.get("pi0_nav_pick_contract")
                == pi0_nav_pick_exact_chunk_contract()
            )
            exact_pi0_artifacts_valid = bool(
                exact_chunk_contract_valid
                and _exact_pi0_attempt_artifacts_valid(attempt_dir)
            )
            raw_success = bool(
                attempt_identity_valid
                and isinstance(final_result, dict)
                and final_result.get("task_success") is True
            )
            publication_eligible = bool(raw_success and exact_pi0_artifacts_valid)
            vla_exited = vla_proc.poll() is not None
            terminal_failure = (
                None
                if raw_success or vla_exited or not attempt_identity_valid
                else _terminal_failure_binding(
                    attempt_dir,
                    task_name=spec.task_name,
                )
            )
            artifact_seal_complete = bool(
                attempt_identity_valid
                and exact_pi0_artifacts_valid
                and _artifact_seal_complete(final_result)
            )
            attempt_task_success: bool | None = (
                True
                if raw_success
                else None
                if (
                    vla_exited
                    or not exact_chunk_contract_valid
                    or not attempt_identity_valid
                )
                else False
                if terminal_failure is not None
                or (
                    isinstance(final_result, dict)
                    and final_result.get("task_success") is False
                    and not execution.timed_out
                    and execution.exit_code in {0, None}
                )
                else None
            )
            outcome = (
                "official_success"
                if raw_success
                else "run_error"
                if (
                    vla_exited
                    or not exact_chunk_contract_valid
                    or not attempt_identity_valid
                )
                else "visual_terminal_failure"
                if terminal_failure is not None
                else "run_error"
                if execution.timed_out
                or execution.exit_code not in {0, None}
                or not isinstance(final_result, dict)
                else "task_failed"
            )
            usage = _usage_from_attempt(
                attempt_dir,
                final_result,
                elapsed_s=elapsed,
                tag=tag,
            )
            summary_source = (
                "Fresh task-specific visual evidence verified the configured "
                "terminal-failure condition after robot interaction; the runtime "
                "sealed the attempt as a terminal task failure."
                if terminal_failure is not None
                else (
                    "Infrastructure error: child attempt task identity mismatch."
                    if not attempt_identity_valid
                    else (
                        _attempt_summary(final_result, raw_success=raw_success)
                        if exact_chunk_contract_valid or raw_success
                        else "Infrastructure error: child Pi0 exact-chunk contract mismatch."
                    )
                )
            )
            sanitized_summary = sanitize_prior_attempt_summaries(
                [
                    {
                        "attempt_index": attempt_index,
                        "outcome": outcome,
                        "summary": summary_source,
                    }
                ]
            )[0]["summary"]
            attempt_prompt_binding = (
                attempt_protocol.get("prompt")
                if isinstance(attempt_protocol, dict)
                else None
            )
            record = {
                "attempt_index": attempt_index,
                "task_identity": {
                    "task_name": spec.task_name,
                    "activity_definition_id": spec.activity_definition_id,
                    "activity_instance_id": _native_instance(config),
                },
                "started_at": attempt_started_at,
                "elapsed_s": round(elapsed, 3),
                "outcome": outcome,
                "task_success": attempt_task_success,
                "terminal_failure": terminal_failure,
                "artifact_seal_complete": artifact_seal_complete,
                "attempt_identity_valid": attempt_identity_valid,
                "exact_pi0_artifacts_valid": exact_pi0_artifacts_valid,
                "publication_eligible": publication_eligible,
                # Compatibility alias only; not a task-success gate.
                "workflow_complete": artifact_seal_complete,
                "summary": sanitized_summary,
                "subprocess_exit_code": execution.exit_code,
                "timed_out": execution.timed_out,
                "vla_binding_sha256": expected_binding_digest,
                "prompt": (
                    json.loads(
                        json.dumps(
                            attempt_prompt_binding,
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                    )
                    if isinstance(attempt_prompt_binding, dict)
                    else None
                ),
                "forced_cleanup_groups": {
                    key: list(value) for key, value in execution.forced_cleanup.items()
                },
                "output_dir": str(attempt_dir),
                "usage": usage,
            }
            manifest["attempts"].append(record)
            manifest["cumulative"]["attempts"] += 1
            for key in (
                "env_steps",
                "vla_chunks",
                "vla_invocations",
                "tool_calls",
                "wall_clock_s",
            ):
                manifest["cumulative"][key] += usage[key]
            if attempt_task_success is False:
                _ensure_failed_archive(
                    tag_root,
                    attempt_dir,
                    record,
                    task_spec=spec,
                    public_seed=config.public_seed,
                    recipe_tag=tag,
                )
            _append_attempt_trace(root, attempt_dir, attempt_index, tag=attempt_tag)
            _append_jsonl(root / "events.jsonl", {"type": "attempt_finished", **record})
            _append_jsonl(
                root / "events.jsonl",
                {
                    "type": "env_restart_completed",
                    "at": _utc_now(),
                    "attempt_index": attempt_index,
                    "outcome": outcome,
                },
            )
            if dashboard is not None:
                dashboard.end_attempt(attempt_index=attempt_index, outcome=outcome)
                dashboard.on_job_progress(
                    {
                        **manifest["cumulative"],
                        "job_unlimited": config.max_attempts is None,
                    }
                )
                dashboard.on_event(
                    {"type": "env_restart_completed", "attempt_index": attempt_index}
                )
            if not attempt_identity_valid:
                manifest["status"] = "blocked"
                manifest["task_success"] = None
                manifest["artifact_seal_complete"] = False
                manifest["workflow_complete"] = False
                manifest["publication_complete"] = False
                manifest["blocked_reason"] = "attempt_task_identity_mismatch"
                manifest["finished_at"] = _utc_now()
                _atomic_json(manifest_path, manifest)
                return manifest
            if vla_exited and not raw_success:
                process = manifest.get("processes", {}).get("vla")
                if isinstance(process, dict):
                    process["stopped_at"] = _utc_now()
                    process["returncode"] = vla_proc.returncode
                manifest["status"] = "blocked"
                manifest["task_success"] = None
                manifest["blocked_reason"] = "persistent_vla_exited"
                manifest["finished_at"] = _utc_now()
                _append_jsonl(
                    root / "events.jsonl",
                    {
                        "type": "vla_exited",
                        "at": _utc_now(),
                        "attempt_index": attempt_index,
                    },
                )
                _atomic_json(manifest_path, manifest)
                return manifest
            if raw_success:
                manifest["task_success"] = True
                manifest["artifact_seal_complete"] = artifact_seal_complete
                manifest["workflow_complete"] = artifact_seal_complete
                manifest["publication_complete"] = False
                manifest["status"] = "succeeded"
                manifest["finished_at"] = _utc_now()
                if vla_proc is not None and deps.owns_vla:
                    deps.stop_vla(vla_proc)
                    process = manifest.get("processes", {}).get("vla")
                    if isinstance(process, dict):
                        process["stopped_at"] = _utc_now()
                        process["returncode"] = vla_proc.poll()
                    vla_proc = None
                _atomic_json(manifest_path, manifest)
                publication_complete = (
                    False
                    if config.candidate_instance_id is not None
                    else _promote_success(
                        root,
                        attempt_dir,
                        publication_eligible,
                        task_name=spec.task_name,
                    )
                )
                # The session manifest is intentionally immutable after its
                # SHA is bound into publication provenance.  The amendment is
                # the authoritative publication overlay.
                manifest["publication_complete"] = publication_complete
                if dashboard is not None:
                    dashboard.on_event(
                        {
                            "type": "official_success",
                            "attempt_index": attempt_index,
                            "task_success": True,
                            "artifact_seal_complete": artifact_seal_complete,
                            # Compatibility alias for older Dashboard clients.
                            "workflow_complete": artifact_seal_complete,
                            "publication_complete": publication_complete,
                        }
                    )
                    dashboard.mark_done(True)
                return manifest
            if terminal_failure is not None:
                manifest["task_success"] = False
                manifest["artifact_seal_complete"] = artifact_seal_complete
                manifest["workflow_complete"] = artifact_seal_complete
                manifest["publication_complete"] = False
                manifest["status"] = "completed_without_success"
                manifest["finished_at"] = _utc_now()
                manifest["blocked_reason"] = None
                manifest["terminal_failure"] = terminal_failure
                _atomic_json(manifest_path, manifest)
                if dashboard is not None:
                    dashboard.on_event(
                        {
                            "type": "visual_terminal_failure",
                            "attempt_index": attempt_index,
                            "task_success": False,
                            "terminal_failure": terminal_failure,
                        }
                    )
                    dashboard.mark_done(False)
                return manifest
            if attempt_task_success is False:
                prior.append(record)
            if config.max_attempts is not None and attempt_index >= config.max_attempts:
                manifest["finished_at"] = _utc_now()
                attempts = [
                    item
                    for item in manifest.get("attempts", [])
                    if isinstance(item, dict)
                ]
                if attempts and all(
                    item.get("task_success") is False for item in attempts
                ):
                    manifest["task_success"] = False
                    manifest["status"] = "completed_without_success"
                    manifest["blocked_reason"] = None
                else:
                    manifest["task_success"] = None
                    manifest["status"] = "blocked"
                    manifest["blocked_reason"] = "attempt_outcome_unknown"
                _atomic_json(manifest_path, manifest)
                if dashboard is not None and manifest["task_success"] is False:
                    dashboard.mark_done(False)
                return manifest
            attempt_index += 1
            _atomic_json(manifest_path, manifest)
    except KeyboardInterrupt:
        manifest["status"] = "stopped_by_operator"
        manifest["task_success"] = None
        manifest["finished_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        raise
    except Exception as error:
        manifest["status"] = "blocked"
        manifest["task_success"] = None
        manifest["blocked_reason"] = f"{type(error).__name__}: {error}"
        _atomic_json(manifest_path, manifest)
        return manifest
    finally:
        if vla_proc is not None and deps.owns_vla:
            deps.stop_vla(vla_proc)
            process = manifest.get("processes", {}).get("vla")
            if isinstance(process, dict):
                process["stopped_at"] = _utc_now()
                process["returncode"] = vla_proc.poll()
            _atomic_json(manifest_path, manifest)
        del dashboard_server
        _release_job_lock(lock_descriptor)
        _release_job_lock(gpu_lock_descriptor)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unlimited fresh-env BEHAVIOR Explore attempts until success."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo", default=None)
    parser.add_argument("--behavior-python", default=None)
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
        help="shared BEHAVIOR Pi0.5 checkpoint; task-specific SFTs are rejected",
    )
    parser.add_argument(
        "--task-name",
        choices=("turning_on_radio", "picking_up_trash"),
        default=TASK_NAME,
    )
    parser.add_argument("--public-seed", type=int, default=PUBLIC_SEED)
    parser.add_argument("--cuda-device", default="7")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="xhigh",
    )
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--max-tool-calls", type=int, default=ATTEMPT_TOOL_CALLS)
    parser.add_argument("--planner-timeout-s", type=int, default=ATTEMPT_WALL_CLOCK_S)
    parser.add_argument(
        "--attempt-timeout-s", type=int, default=ATTEMPT_PROCESS_TIMEOUT_S
    )
    parser.add_argument("--vla-ready-timeout-s", type=int, default=1800)
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-auto-start", action="store_true")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--dashboard-language", choices=("en", "zh-cn"), default="en")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--behavior-resource-revision",
        default=None,
        help=(
            "requested RLinf/RPent-memory revision; resolve and pin it once "
            "before reading BEHAVIOR memory or recipes"
        ),
    )
    parser.add_argument(
        "--behavior-resource-local",
        default=None,
        help=(
            "local closed-manifest BEHAVIOR resource root; copy it into an "
            "immutable content-addressed snapshot instead of contacting HF"
        ),
    )
    parser.add_argument(
        "--behavior-resource-cache",
        default=None,
        help=(
            "versioned resource cache root (default: <repo-root>/resources/.snapshots)"
        ),
    )
    parser.add_argument(
        "--behavior-resource-offline",
        action="store_true",
        default=None,
        help=("forbid network resolution/download; HF_HUB_OFFLINE=1 also enables it"),
    )
    parser.add_argument(
        "--behavior-recipe-catalog-sha256",
        default=None,
        help="optional epoch-pinned reviewed Recipe Catalog SHA256",
    )
    parser.add_argument(
        "--epoch-predecessor-binding-file",
        default=None,
        help="optional JSON object binding this Explore Job to a prior epoch",
    )
    offline_action = parser.add_mutually_exclusive_group()
    offline_action.add_argument(
        "--publish-existing-success",
        action="store_true",
        help=(
            "derive hash-bound recipe/memory from an already successful Job "
            "without changing its original attempt"
        ),
    )
    offline_action.add_argument(
        "--correct-existing-success",
        action="store_true",
        help=(
            "correct historical summary lifecycle state from canonical "
            "behavior_action_trace info_done.success evidence"
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else None
    args = _parse_args(arguments)
    explicit_arguments = arguments if arguments is not None else sys.argv[1:]
    explicit_task_name = (
        args.task_name
        if any(
            value == "--task-name" or value.startswith("--task-name=")
            for value in explicit_arguments
        )
        else None
    )
    if args.correct_existing_success:
        corrected = correct_existing_success(
            args.output_root,
            task_name=explicit_task_name,
        )
        print(json.dumps(corrected, indent=2, sort_keys=True))
        return 0
    if args.publish_existing_success:
        amendment = publish_existing_success(
            args.output_root,
            task_name=explicit_task_name,
        )
        print(json.dumps(amendment, indent=2, sort_keys=True))
        return 0
    if args.dashboard_auto_start and not args.dashboard:
        raise SystemExit("--dashboard-auto-start requires --dashboard")
    if args.cuda_device != "7":
        raise SystemExit("formal BEHAVIOR Explore is restricted to GPU7")
    repo_root = Path(args.repo_root).expanduser().resolve()
    resource_cache = (
        Path(args.behavior_resource_cache).expanduser().resolve()
        if args.behavior_resource_cache
        else repo_root / "resources" / ".snapshots"
    )
    if args.behavior_resource_local and args.behavior_resource_revision:
        raise SystemExit(
            "--behavior-resource-local cannot be combined with "
            "--behavior-resource-revision"
        )
    try:
        if args.behavior_resource_local:
            resource_binding = prepare_local_dataset_resources(
                "behavior",
                source_root=Path(args.behavior_resource_local)
                .expanduser()
                .resolve(strict=True),
                cache_root=resource_cache,
            )
        else:
            resource_binding = prepare_pinned_dataset_resources(
                "behavior",
                requested_revision=args.behavior_resource_revision,
                cache_root=resource_cache,
                offline=args.behavior_resource_offline,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"invalid pinned BEHAVIOR resources: {error}") from error
    epoch_predecessor_binding = None
    if args.epoch_predecessor_binding_file:
        try:
            epoch_predecessor_binding = _read_epoch_predecessor_binding_file(
                args.epoch_predecessor_binding_file
            )
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
    behavior_repo = (
        Path(
            args.behavior_repo
            or os.environ.get("RPENT_RLINF_ROOT")
            or repo_root.parent / "RLinf_agentic_push"
        )
        .expanduser()
        .resolve()
    )
    behavior_python = (
        Path(
            args.behavior_python or behavior_repo / ".venv-behavior" / "bin" / "python"
        )
        .expanduser()
        .absolute()
    )
    try:
        task_spec = get_task_spec(args.task_name)
        task_spec.instance_for_public_seed(args.public_seed, phase="explore")
        checkpoint_binding = _expected_job_checkpoint_binding(args.policy_checkpoint)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    config = ExploreConfig(
        output_root=Path(args.output_root).expanduser().resolve(),
        repo_root=repo_root,
        # Preserve virtualenv launcher symlinks. Resolving either executable to
        # /usr/bin/python drops its venv site-packages in spawned VLA/attempt
        # processes.
        python=Path(args.python).expanduser().absolute(),
        behavior_repo=behavior_repo,
        behavior_python=behavior_python,
        policy_checkpoint=Path(checkpoint_binding["resolved_path"]),
        policy_checkpoint_binding=checkpoint_binding,
        task_name=task_spec.task_name,
        public_seed=args.public_seed,
        cuda_device=args.cuda_device,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        planner_timeout_s=args.planner_timeout_s,
        attempt_timeout_s=args.attempt_timeout_s,
        vla_ready_timeout_s=args.vla_ready_timeout_s,
        min_free_disk_gb=args.min_free_disk_gb,
        dashboard=args.dashboard,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        dashboard_language=args.dashboard_language,
        resume=args.resume,
        recipe_catalog_sha256=args.behavior_recipe_catalog_sha256,
        epoch_predecessor_binding=epoch_predecessor_binding,
        resource_binding=resource_binding,
    )
    manifest = run_explore_job(config)
    if args.dashboard:
        print(f"Dashboard remains available at {manifest.get('dashboard_url')}")
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass
    if manifest.get("task_success") is True:
        return 0
    return 2 if manifest.get("status") == "blocked" else 1


__all__ = [
    "AttemptExecution",
    "ExploreConfig",
    "ExploreDependencies",
    "build_attempt_argv",
    "correct_existing_success",
    "main",
    "publish_existing_success",
    "run_explore_job",
    "sanitize_prior_attempt_summaries",
]


if __name__ == "__main__":
    raise SystemExit(main())
