"""Paired GPU7 agentic and GPU6 pure-VLA BEHAVIOR Eval supervisor.

The supervisor owns two persistent VLA servers and two independent live
Dashboards. For each task-local Eval seed it launches one single-seed runner per
lane, waits for both, then advances to the next pair. All runnable Python comes
from one validated read-only source snapshot.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping

from robots.behavior.dashboard_relay import DashboardEventRelay
from robots.behavior.dashboard_server import DashboardServer
from robots.behavior.dashboard_state import State
from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    prepare_local_dataset_resources,
    verify_pinned_dataset_resources,
)
from robots.behavior.llm_preflight import (
    network_environment_binding,
    run_llm_proxy_preflight,
)
from robots.behavior.policy_checkpoint import (
    SHARED_POLICY_CHECKPOINT_PATH,
)
from robots.behavior.publication import (
    PublicationValidationError,
    ValidatedBehaviorPublication,
    validate_canonical_publication_root,
)
from robots.behavior.runtime import (
    BEHAVIOR_NATIVE_ENV_SEED,
    _expected_shared_policy_checkpoint_binding,
    _terminate_process,
    prepare_campaign_runtime_isolation,
    start_vla_server,
    validate_campaign_runtime_isolation,
)
from robots.behavior.schemas import (
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    PUBLIC_TOOL_CONTRACTS,
)
from robots.behavior.serial_eval import _gpu_lock_path, _terminate_manifest_processes
from robots.behavior.serial_vla_eval import (
    CHUNKS_PER_CALL as PURE_VLA_CHUNKS_PER_CALL,
)
from robots.behavior.source_snapshot import validate_source_snapshot
from robots.behavior.task_specs import PICKING_UP_TRASH_TASK_SPEC, get_task_spec
from robots.behavior.terminal_success import summarize_action_trace_success
from robots.behavior.vla_client import BehaviorVLAClient

PAIRED_EVAL_SCHEMA_VERSION = 1
DEFAULT_AGENTIC_GPU = "7"
DEFAULT_BASELINE_GPU = "6"
DEFAULT_AGENTIC_DASHBOARD_PORT = 8766
DEFAULT_BASELINE_DASHBOARD_PORT = 8767
PROTECTED_EXISTING_DASHBOARD_PORT = 8765
DEFAULT_ACTION_DEADLINE_S = 6900
DEFAULT_CLEANUP_DEADLINE_S = 7080
DEFAULT_INSTANCE_TIMEOUT_S = 7200
POST_SUCCESS_CLEANUP_DEADLINE_S = 180
DEFAULT_MONITOR_INTERVAL_S = 20 * 60
DEFAULT_MONITOR_WINDOW_S = 2 * 60 * 60
DEFAULT_MIN_FREE_DISK_BYTES = 30 * 1024**3
DEFAULT_MIN_FREE_RUNTIME_BYTES = 20 * 1024**3
CANONICAL_AGENTIC_SOURCE_PUBLIC_SEED = 3
INSTANCE_CHILD_PROCESS_SCHEMA_VERSION = 1
EXTERNAL_RUNTIME_OWNER_SCHEMA_VERSION = 1
EXTERNAL_RUNTIME_OWNER_FILENAME = "external_runtime_owner.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ArmSpec:
    """Immutable public configuration for one side of the paired campaign."""

    name: str
    gpu: str
    dashboard_port: int
    output_root: Path
    runner_script: Path
    run_id: str
    llm_enabled: bool
    controller: str
    allowed_tools: tuple[str, ...]


@dataclass(frozen=True)
class PairDeadlineBinding:
    """One immutable absolute deadline origin shared by both lanes of a pair."""

    started_monotonic_ns: int
    action_deadline_monotonic_ns: int
    cleanup_deadline_monotonic_ns: int
    hard_deadline_monotonic_ns: int


@dataclass
class OwnedChild:
    """One exact runner process identity authorized for later cleanup."""

    arm: ArmSpec
    public_seed: int
    output_root: Path
    entry_output_dir: Path
    event_path: Path
    log_path: Path
    argv: tuple[str, ...]
    process: subprocess.Popen[Any]
    pid: int
    pgid: int
    sid: int
    start_ticks: int
    started_monotonic: float
    source_snapshot_root: Path
    source_snapshot_binding_sha256: str
    action_deadline_s: int
    cleanup_deadline_s: int
    instance_timeout_s: int
    started_monotonic_ns: int
    action_deadline_monotonic_ns: int
    cleanup_deadline_monotonic_ns: int
    hard_deadline_monotonic_ns: int
    action_deadline_monotonic: float
    cleanup_deadline_monotonic: float
    hard_deadline_monotonic: float
    expected_run_nonce: str | None
    timed_out: bool = False
    identity_ambiguous: bool = False
    action_cleanup_started: bool = False
    forced_cleanup_started: bool = False
    cleanup_verified: bool = False
    action_deadline_exhausted: bool = False
    hard_deadline_exhausted: bool = False
    safety_errors: list[str] | None = None
    peer_abort_reason: str | None = None
    post_run_vla_quiesced: bool = False
    post_run_vla_health: dict[str, Any] | None = None
    official_success_observed_monotonic: float | None = None

    @property
    def argv_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(list(self.argv), separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass
class LiveArm:
    spec: ArmSpec
    server: DashboardServer
    url: str
    state: State | None = None
    relay: DashboardEventRelay | None = None
    child: OwnedChild | None = None
    vla_endpoint: str | None = None
    vla_process: subprocess.Popen[Any] | None = None
    vla_pgid: int | None = None
    vla_sid: int | None = None
    vla_start_ticks: int | None = None
    vla_disabled_health: dict[str, Any] | None = None
    disabled_reason: str | None = None
    gpu_ownership_violations: list[str] = dataclass_field(default_factory=list)
    runtime_violations: list[str] = dataclass_field(default_factory=list)


@dataclass
class PairMonitorState:
    """Health-monitoring state owned by exactly one paired seed."""

    started: list[float | None] = dataclass_field(default_factory=lambda: [None])
    sampled_offsets: set[int] = dataclass_field(default_factory=set)
    previous: dict[str, dict[str, Any]] = dataclass_field(default_factory=dict)
    start_evidence: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class ValidatedRuntimeBase:
    """An existing external parent directory approved for transient runtime data."""

    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class OwnedRuntimeRoot:
    """One unique campaign-owned child below a caller-owned runtime base."""

    base: Path
    root: Path
    base_device: int
    base_inode: int
    root_device: int
    root_inode: int
    output_root: Path
    output_device: int
    output_inode: int
    source_snapshot_root: Path
    source_snapshot_binding_sha256: str
    owner_token: str
    created_at: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_file_no_follow(path: Path) -> tuple[bytes, os.stat_result]:
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
        file_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(parts[-1], file_flags, dir_fd=directory_fd)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(f"file is not regular: {path}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field) for field in identity_fields
    ):
        raise RuntimeError(f"file changed while reading: {path}")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise RuntimeError(f"file size changed while reading: {path}")
    return payload, after


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_external_runtime_base(
    runtime_base: str | os.PathLike[str],
    *,
    output_root: Path,
    protected_paths: Iterable[Path],
    minimum_free_bytes: int = DEFAULT_MIN_FREE_RUNTIME_BYTES,
) -> ValidatedRuntimeBase:
    """Validate a caller-owned parent without creating or deleting it."""

    lexical = Path(runtime_base).expanduser().absolute()
    try:
        lexical_stat = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"--runtime-base is unavailable: {error}") from error
    if stat.S_ISLNK(lexical_stat.st_mode):
        raise ValueError("--runtime-base must not be a symlink")
    if not stat.S_ISDIR(lexical_stat.st_mode):
        raise ValueError("--runtime-base must be an existing directory")
    canonical = lexical.resolve(strict=True)
    canonical_stat = os.stat(canonical, follow_symlinks=False)
    if not stat.S_ISDIR(canonical_stat.st_mode):
        raise ValueError("--runtime-base must resolve to a directory")
    if not os.access(canonical, os.W_OK | os.X_OK):
        raise ValueError("--runtime-base must be writable and searchable")

    output_parent = output_root.parent.resolve(strict=True)
    canonical_output = output_parent / output_root.name
    if canonical_stat.st_dev == os.stat(output_parent).st_dev:
        raise ValueError(
            "--runtime-base must be on a different filesystem from --output-root"
        )
    if shutil.disk_usage(canonical).free < minimum_free_bytes:
        raise ValueError("--runtime-base has less than 20 GiB free")

    protected = (
        canonical_output,
        *(path.resolve(strict=True) for path in protected_paths),
    )
    for path in protected:
        if _paths_overlap(canonical, path):
            raise ValueError(
                "--runtime-base must not contain or be contained by formal inputs "
                f"or outputs: {path}"
            )
    return ValidatedRuntimeBase(
        path=canonical,
        device=canonical_stat.st_dev,
        inode=canonical_stat.st_ino,
    )


def _external_runtime_owner_document(
    owned: OwnedRuntimeRoot,
    runtime_isolation: Mapping[str, Any],
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for name, isolation in sorted(runtime_isolation.items()):
        expected_root = owned.root / name
        isolation_root = Path(isolation.root).resolve(strict=True)
        if isolation_root != expected_root.resolve(strict=True):
            raise RuntimeError(f"{name} runtime isolation escaped the owned root")
        arms[name] = {
            "root": str(isolation_root),
            "namespace": str(isolation.namespace),
            "cuda_device": str(isolation.cuda_device),
            "binding_sha256": str(isolation.binding_sha256),
            "binding": isolation.as_dict(),
        }
    payload: dict[str, Any] = {
        "schema_version": EXTERNAL_RUNTIME_OWNER_SCHEMA_VERSION,
        "kind": "rpent_paired_eval_external_runtime",
        "created_at": owned.created_at,
        "owner_token": owned.owner_token,
        "runtime_base": {
            "path": str(owned.base),
            "device": owned.base_device,
            "inode": owned.base_inode,
        },
        "runtime_root": {
            "path": str(owned.root),
            "device": owned.root_device,
            "inode": owned.root_inode,
        },
        "output_root": {
            "path": str(owned.output_root),
            "device": owned.output_device,
            "inode": owned.output_inode,
        },
        "source_snapshot": {
            "root": str(owned.source_snapshot_root),
            "binding_sha256": owned.source_snapshot_binding_sha256,
        },
        "arms": arms,
    }
    return {**payload, "binding_sha256": _canonical_json_sha256(payload)}


def _write_external_runtime_owner(
    owned: OwnedRuntimeRoot,
    runtime_isolation: Mapping[str, Any],
) -> dict[str, Any]:
    document = _external_runtime_owner_document(owned, runtime_isolation)
    marker = owned.root / EXTERNAL_RUNTIME_OWNER_FILENAME
    _atomic_json(marker, document)
    marker.chmod(0o600)
    evidence_dir = owned.output_root / "runtime_bindings"
    evidence_dir.mkdir(mode=0o700, exist_ok=True)
    evidence = evidence_dir / EXTERNAL_RUNTIME_OWNER_FILENAME
    _atomic_json(evidence, document)
    evidence.chmod(0o600)
    return document


def _create_owned_runtime_root(
    runtime_base: ValidatedRuntimeBase,
    *,
    output_root: Path,
    source_snapshot_root: Path,
    source_snapshot_binding_sha256: str,
) -> tuple[OwnedRuntimeRoot, dict[str, Any]]:
    """Create one unique 0700 child; the caller-owned base is never a delete target."""

    current_base = os.stat(runtime_base.path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current_base.st_mode)
        or current_base.st_dev != runtime_base.device
        or current_base.st_ino != runtime_base.inode
    ):
        raise RuntimeError("--runtime-base identity changed before allocation")
    if shutil.disk_usage(runtime_base.path).free < DEFAULT_MIN_FREE_RUNTIME_BYTES:
        raise RuntimeError("--runtime-base has less than 20 GiB free")
    created = Path(
        tempfile.mkdtemp(prefix="rpent-paired-eval-", dir=str(runtime_base.path))
    )
    created.chmod(0o700)
    root_stat = os.stat(created, follow_symlinks=False)
    output_stat = os.stat(output_root, follow_symlinks=False)
    owned = OwnedRuntimeRoot(
        base=runtime_base.path,
        root=created,
        base_device=runtime_base.device,
        base_inode=runtime_base.inode,
        root_device=root_stat.st_dev,
        root_inode=root_stat.st_ino,
        output_root=output_root.resolve(strict=True),
        output_device=output_stat.st_dev,
        output_inode=output_stat.st_ino,
        source_snapshot_root=source_snapshot_root.resolve(strict=True),
        source_snapshot_binding_sha256=source_snapshot_binding_sha256,
        owner_token=secrets.token_hex(32),
        created_at=_utc_now(),
    )
    try:
        document = _write_external_runtime_owner(owned, {})
    except Exception:
        shutil.rmtree(created)
        raise
    return owned, document


def _external_runtime_cleanup_errors(
    owned: OwnedRuntimeRoot,
    *,
    owner_document: Mapping[str, Any],
    runtime_isolation: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        base_stat = os.stat(owned.base, follow_symlinks=False)
        if (
            not stat.S_ISDIR(base_stat.st_mode)
            or base_stat.st_dev != owned.base_device
            or base_stat.st_ino != owned.base_inode
        ):
            errors.append("runtime_base_identity_changed")
    except OSError as error:
        errors.append(f"runtime_base_unavailable: {error}")
    try:
        root_stat = os.stat(owned.root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_dev != owned.root_device
            or root_stat.st_ino != owned.root_inode
            or owned.root.parent != owned.base
        ):
            errors.append("runtime_root_identity_changed")
    except OSError as error:
        errors.append(f"runtime_root_unavailable: {error}")
    try:
        output_stat = os.stat(owned.output_root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(output_stat.st_mode)
            or output_stat.st_dev != owned.output_device
            or output_stat.st_ino != owned.output_inode
        ):
            errors.append("output_root_identity_changed")
    except OSError as error:
        errors.append(f"output_root_unavailable: {error}")

    try:
        validate_source_snapshot(
            owned.source_snapshot_root,
            owned.source_snapshot_binding_sha256,
        )
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"source_snapshot_binding_changed: {error}")

    try:
        marker_bytes, _ = _read_regular_file_no_follow(
            owned.root / EXTERNAL_RUNTIME_OWNER_FILENAME
        )
        marker = json.loads(marker_bytes)
        if marker != dict(owner_document):
            errors.append("runtime_owner_marker_changed")
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"runtime_owner_marker_invalid: {error}")

    document_without_sha = dict(owner_document)
    recorded_sha = document_without_sha.pop("binding_sha256", None)
    if (
        not isinstance(recorded_sha, str)
        or _SHA256_RE.fullmatch(recorded_sha) is None
        or _canonical_json_sha256(document_without_sha) != recorded_sha
    ):
        errors.append("runtime_owner_binding_invalid")

    documented_arms = owner_document.get("arms")
    if not isinstance(documented_arms, dict) or set(documented_arms) != set(
        runtime_isolation
    ):
        errors.append("runtime_owner_arm_set_changed")
        documented_arms = {}
    for name, expected in runtime_isolation.items():
        try:
            actual = validate_campaign_runtime_isolation(
                expected.root,
                expected.binding_sha256,
            )
            documented = documented_arms.get(name)
            if (
                actual.as_dict() != expected.as_dict()
                or not isinstance(documented, dict)
                or documented.get("namespace") != expected.namespace
                or documented.get("cuda_device") != expected.cuda_device
                or documented.get("binding_sha256") != expected.binding_sha256
                or documented.get("root") != str(expected.root)
                or documented.get("binding") != expected.as_dict()
            ):
                errors.append(f"{name}_runtime_binding_changed")
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{name}_runtime_binding_invalid: {error}")
    return errors


def _delete_owned_runtime_root(
    owned: OwnedRuntimeRoot,
    *,
    owner_document: Mapping[str, Any],
    runtime_isolation: Mapping[str, Any],
    processes_stopped: bool,
) -> dict[str, Any]:
    """Delete only the verified owned child, retaining it on any ambiguity."""

    result: dict[str, Any] = {
        "status": "runtime_cleanup_pending",
        "attempted_at": _utc_now(),
        "runtime_base": str(owned.base),
        "runtime_root": str(owned.root),
        "runtime_root_device": owned.root_device,
        "runtime_root_inode": owned.root_inode,
        "owner_binding_sha256": owner_document.get("binding_sha256"),
        "errors": [],
    }
    if not processes_stopped:
        result["errors"] = ["owned_process_cleanup_unverified"]
        return result
    errors = _external_runtime_cleanup_errors(
        owned,
        owner_document=owner_document,
        runtime_isolation=runtime_isolation,
    )
    if errors:
        result["errors"] = errors
        return result

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )

    def remove_contents(directory_fd: int) -> None:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISDIR(entry_stat.st_mode):
                    os.unlink(entry.name, dir_fd=directory_fd)
                    continue
                if entry_stat.st_dev != owned.root_device:
                    raise RuntimeError(
                        f"runtime directory crosses a filesystem: {entry.name}"
                    )
                child_fd = os.open(entry.name, directory_flags, dir_fd=directory_fd)
                try:
                    child_stat = os.fstat(child_fd)
                    if (
                        child_stat.st_dev != entry_stat.st_dev
                        or child_stat.st_ino != entry_stat.st_ino
                    ):
                        raise RuntimeError(
                            f"runtime directory changed during deletion: {entry.name}"
                        )
                    remove_contents(child_fd)
                    current = os.stat(
                        entry.name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                    if (
                        current.st_dev != child_stat.st_dev
                        or current.st_ino != child_stat.st_ino
                    ):
                        raise RuntimeError(
                            f"runtime directory changed before rmdir: {entry.name}"
                        )
                    os.rmdir(entry.name, dir_fd=directory_fd)
                finally:
                    os.close(child_fd)

    try:
        base_fd = os.open(owned.base, directory_flags)
        try:
            base_stat = os.fstat(base_fd)
            if (
                base_stat.st_dev != owned.base_device
                or base_stat.st_ino != owned.base_inode
            ):
                raise RuntimeError("runtime base changed before deletion")
            root_fd = os.open(owned.root.name, directory_flags, dir_fd=base_fd)
            try:
                root_stat = os.fstat(root_fd)
                if (
                    root_stat.st_dev != owned.root_device
                    or root_stat.st_ino != owned.root_inode
                ):
                    raise RuntimeError("runtime root changed before deletion")
                remove_contents(root_fd)
                current_root = os.stat(
                    owned.root.name,
                    dir_fd=base_fd,
                    follow_symlinks=False,
                )
                if (
                    current_root.st_dev != root_stat.st_dev
                    or current_root.st_ino != root_stat.st_ino
                ):
                    raise RuntimeError("runtime root changed before rmdir")
                os.rmdir(owned.root.name, dir_fd=base_fd)
                os.close(root_fd)
                root_fd = -1
            finally:
                if root_fd >= 0:
                    os.close(root_fd)
            try:
                os.stat(owned.root.name, dir_fd=base_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("runtime root still exists after deletion")
        finally:
            os.close(base_fd)
    except (OSError, RuntimeError) as error:
        result["errors"] = [f"runtime_root_delete_failed: {error}"]
        return result
    result["status"] = "deleted"
    result["completed_at"] = _utc_now()
    return result


def _external_runtime_active_errors(
    owned: OwnedRuntimeRoot,
    *,
    owner_document: Mapping[str, Any],
    runtime_isolation: Mapping[str, Any],
) -> list[str]:
    errors = _external_runtime_cleanup_errors(
        owned,
        owner_document=owner_document,
        runtime_isolation=runtime_isolation,
    )
    try:
        if shutil.disk_usage(owned.base).free < DEFAULT_MIN_FREE_RUNTIME_BYTES:
            errors.append("runtime_base_free_space_below_20_gib")
    except OSError as error:
        errors.append(f"runtime_base_capacity_unavailable: {error}")
    return errors


def _partition_runtime_errors(
    errors: Iterable[str],
    *,
    lane_names: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Separate shared runtime ownership failures from one-lane bindings."""

    names = tuple(str(name) for name in lane_names)
    shared: list[str] = []
    by_lane = {name: [] for name in names}
    for error in (str(item) for item in errors):
        matched = next(
            (name for name in names if error.startswith(f"{name}_runtime_")),
            None,
        )
        if matched is None:
            shared.append(error)
        else:
            by_lane[matched].append(error)
    return shared, by_lane


def _record_external_runtime_cleanup(
    *,
    output_root: Path,
    manifest_path: Path,
    cleanup: Mapping[str, Any],
) -> None:
    _atomic_json(output_root / "runtime_cleanup.json", cleanup)
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict):
        return
    external_runtime = manifest.setdefault("external_runtime", {})
    if not isinstance(external_runtime, dict):
        external_runtime = {}
        manifest["external_runtime"] = external_runtime
    external_runtime["cleanup"] = dict(cleanup)
    pending = cleanup.get("status") in {
        "cleanup_in_progress",
        "runtime_cleanup_pending",
    }
    manifest["runtime_cleanup_pending"] = pending
    if pending:
        reason = "runtime_cleanup_pending"
        errors = cleanup.get("errors")
        if isinstance(errors, list) and errors:
            reason += ": " + "; ".join(str(error) for error in errors)
        previous = manifest.get("blocked_reason")
        manifest["blocked_reason"] = reason if not previous else f"{previous}; {reason}"
        manifest["status"] = "blocked"
        manifest["finished_at"] = _utc_now()
    else:
        previous = manifest.get("blocked_reason")
        if isinstance(previous, str):
            marker = "runtime_cleanup_pending"
            if previous.startswith(marker):
                manifest["blocked_reason"] = None
            else:
                prefix, separator, _suffix = previous.partition(f"; {marker}")
                if separator:
                    manifest["blocked_reason"] = prefix or None
    _atomic_json(manifest_path, manifest)


def validate_dashboard_endpoints(
    *,
    host: str,
    agentic_port: int,
    baseline_port: int,
) -> None:
    """Reject any endpoint that could collide with the protected s3 monitor."""

    if host != "127.0.0.1":
        raise ValueError("paired Eval Dashboards must bind 127.0.0.1")
    ports = (agentic_port, baseline_port)
    if any(isinstance(port, bool) or not 1 <= int(port) <= 65535 for port in ports):
        raise ValueError("paired Eval Dashboard ports must be in 1..65535")
    if agentic_port == baseline_port:
        raise ValueError("paired Eval Dashboard ports must be distinct")
    if PROTECTED_EXISTING_DASHBOARD_PORT in ports:
        raise ValueError("port 8765 is protected and cannot be used by paired Eval")


def validate_deadlines(
    *,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
) -> None:
    if not (
        0
        < int(action_deadline_s)
        < int(cleanup_deadline_s)
        < int(instance_timeout_s)
        <= DEFAULT_INSTANCE_TIMEOUT_S
    ):
        raise ValueError(
            "deadlines must satisfy 0 < action < cleanup < instance <= 7200"
        )


def _pair_deadline_binding(
    *,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
    started_monotonic_ns: int | None = None,
) -> PairDeadlineBinding:
    """Sample or validate the single monotonic origin for one paired seed."""

    validate_deadlines(
        action_deadline_s=action_deadline_s,
        cleanup_deadline_s=cleanup_deadline_s,
        instance_timeout_s=instance_timeout_s,
    )
    started = (
        time.monotonic_ns()
        if started_monotonic_ns is None
        else int(started_monotonic_ns)
    )
    if started < 0:
        raise ValueError("pair monotonic deadline origin must be non-negative")
    return PairDeadlineBinding(
        started_monotonic_ns=started,
        action_deadline_monotonic_ns=(started + int(action_deadline_s) * 1_000_000_000),
        cleanup_deadline_monotonic_ns=(
            started + int(cleanup_deadline_s) * 1_000_000_000
        ),
        hard_deadline_monotonic_ns=(started + int(instance_timeout_s) * 1_000_000_000),
    )


def _lexical_executable_path(value: str | os.PathLike[str], *, label: str) -> Path:
    """Validate an executable while preserving its lexical symlink identity."""

    path = Path(value).expanduser().absolute()
    if not path.exists():
        raise SystemExit(f"{label} does not exist: {path}")
    if not path.is_file():
        raise SystemExit(f"{label} is not a file: {path}")
    if not os.access(path, os.X_OK):
        raise SystemExit(f"{label} is not executable: {path}")
    return path


def build_agentic_runner_argv(
    *,
    python: Path,
    snapshot_root: Path,
    output_root: Path,
    public_seed: int,
    vla_endpoint: str,
    behavior_repo: Path,
    behavior_python: Path,
    checkpoint: Path,
    frozen_publication_root: Path,
    frozen_provenance_sha256: str,
    behavior_resource_local: Path,
    behavior_resource_cache: Path,
    source_binding_sha256: str,
    dashboard_host: str,
    dashboard_port: int,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
    instance_started_monotonic_ns: int,
    action_deadline_monotonic_ns: int,
    cleanup_deadline_monotonic_ns: int,
    hard_deadline_monotonic_ns: int,
    model: str,
    reasoning_effort: str,
    expected_run_nonce: str,
    runtime_isolation_root: Path | None = None,
    runtime_isolation_binding_sha256: str | None = None,
) -> tuple[str, ...]:
    argv = [
        str(python),
        str(snapshot_root / "scripts" / "run_behavior_serial_eval.py"),
        "--output-root",
        str(output_root),
        "--cuda-device",
        DEFAULT_AGENTIC_GPU,
        "--task-name",
        PICKING_UP_TRASH_TASK_SPEC.task_name,
        "--public-seed",
        str(public_seed),
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--planner-timeout-s",
        str(action_deadline_s),
        "--max-wall-clock-s",
        str(action_deadline_s),
        "--cleanup-deadline-s",
        str(cleanup_deadline_s),
        "--instance-timeout-s",
        str(instance_timeout_s),
        "--instance-started-monotonic-ns",
        str(instance_started_monotonic_ns),
        "--action-deadline-monotonic-ns",
        str(action_deadline_monotonic_ns),
        "--cleanup-deadline-monotonic-ns",
        str(cleanup_deadline_monotonic_ns),
        "--hard-deadline-monotonic-ns",
        str(hard_deadline_monotonic_ns),
        "--repo-root",
        str(snapshot_root),
        "--python",
        str(python),
        "--behavior-repo",
        str(behavior_repo),
        "--behavior-python",
        str(behavior_python),
        "--policy-checkpoint",
        str(checkpoint),
        "--behavior-frozen-publication-root",
        str(frozen_publication_root),
        "--behavior-frozen-provenance-sha256",
        frozen_provenance_sha256,
        "--behavior-resource-local",
        str(behavior_resource_local),
        "--behavior-resource-cache",
        str(behavior_resource_cache),
        "--vla-endpoint",
        vla_endpoint,
        "--dashboard-event-sink",
        "--dashboard-host",
        dashboard_host,
        "--dashboard-port",
        str(dashboard_port),
        "--dashboard-language",
        "en",
        "--source-snapshot-root",
        str(snapshot_root),
        "--source-snapshot-binding-sha256",
        source_binding_sha256,
        "--external-gpu-lock-owned",
    ]
    if re.fullmatch(r"[0-9a-f]{32}", expected_run_nonce) is None:
        raise ValueError("expected run nonce must be 32 lowercase hex characters")
    argv.extend(["--expected-run-nonce", expected_run_nonce])
    if runtime_isolation_root is not None:
        if runtime_isolation_binding_sha256 is None:
            raise ValueError("runtime isolation root requires its binding SHA-256")
        argv.extend(
            [
                "--runtime-isolation-root",
                str(runtime_isolation_root),
                "--runtime-isolation-binding-sha256",
                runtime_isolation_binding_sha256,
            ]
        )
    return tuple(argv)


def build_baseline_runner_argv(
    *,
    python: Path,
    snapshot_root: Path,
    output_root: Path,
    public_seed: int,
    vla_endpoint: str,
    behavior_repo: Path,
    behavior_python: Path,
    checkpoint: Path,
    source_binding_sha256: str,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
    instance_started_monotonic_ns: int,
    action_deadline_monotonic_ns: int,
    cleanup_deadline_monotonic_ns: int,
    hard_deadline_monotonic_ns: int,
    runtime_isolation_root: Path | None = None,
    runtime_isolation_binding_sha256: str | None = None,
    expected_run_nonce: str | None = None,
) -> tuple[str, ...]:
    argv = [
        str(python),
        str(snapshot_root / "scripts" / "run_behavior_serial_vla_eval.py"),
        "--output-root",
        str(output_root),
        "--cuda-device",
        DEFAULT_BASELINE_GPU,
        "--task-name",
        PICKING_UP_TRASH_TASK_SPEC.task_name,
        "--public-seed",
        str(public_seed),
        "--repo-root",
        str(snapshot_root),
        "--python",
        str(python),
        "--behavior-repo",
        str(behavior_repo),
        "--behavior-python",
        str(behavior_python),
        "--policy-checkpoint",
        str(checkpoint),
        "--vla-endpoint",
        vla_endpoint,
        "--chunks-per-call",
        str(PURE_VLA_CHUNKS_PER_CALL),
        "--action-deadline-s",
        str(action_deadline_s),
        "--cleanup-deadline-s",
        str(cleanup_deadline_s),
        "--instance-timeout-s",
        str(instance_timeout_s),
        "--instance-started-monotonic-ns",
        str(instance_started_monotonic_ns),
        "--action-deadline-monotonic-ns",
        str(action_deadline_monotonic_ns),
        "--cleanup-deadline-monotonic-ns",
        str(cleanup_deadline_monotonic_ns),
        "--hard-deadline-monotonic-ns",
        str(hard_deadline_monotonic_ns),
        "--source-snapshot-root",
        str(snapshot_root),
        "--source-snapshot-binding-sha256",
        source_binding_sha256,
        "--external-gpu-lock-owned",
    ]
    if expected_run_nonce is not None:
        if re.fullmatch(r"[0-9a-f]{32}", expected_run_nonce) is None:
            raise ValueError("expected run nonce must be 32 lowercase hex characters")
        argv.extend(["--expected-run-nonce", expected_run_nonce])
    if runtime_isolation_root is not None:
        if runtime_isolation_binding_sha256 is None:
            raise ValueError("runtime isolation root requires its binding SHA-256")
        argv.extend(
            [
                "--runtime-isolation-root",
                str(runtime_isolation_root),
                "--runtime-isolation-binding-sha256",
                runtime_isolation_binding_sha256,
            ]
        )
    return tuple(argv)


def _proc_start_ticks(pid: int) -> int | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _proc_identity(pid: int) -> dict[str, int] | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def _proc_executable(pid: int) -> Path | None:
    try:
        return (Path("/proc") / str(pid) / "exe").resolve(strict=True)
    except (OSError, RuntimeError):
        return None


def _resolve_executable_reference(value: str) -> Path | None:
    candidate = Path(value)
    if not candidate.is_absolute():
        located = shutil.which(value)
        if located is None:
            return None
        candidate = Path(located)
    try:
        resolved = candidate.resolve(strict=True)
        resolved_stat = resolved.stat()
    except OSError:
        return None
    if not stat.S_ISREG(resolved_stat.st_mode) or not os.access(resolved, os.X_OK):
        return None
    return resolved


def _bound_nested_python(child: OwnedChild) -> tuple[str, Path] | None:
    """Return the single Python launcher bound by the outer runner argv."""

    positions = [index for index, value in enumerate(child.argv) if value == "--python"]
    if len(positions) != 1 or positions[0] + 1 >= len(child.argv):
        return None
    lexical = str(child.argv[positions[0] + 1])
    resolved = _resolve_executable_reference(lexical)
    if resolved is None:
        return None
    return lexical, resolved


def _nested_cmdline_matches_receipt(
    child: OwnedChild,
    *,
    command: list[str],
    expected_sha256: str,
    proc_executable: Path,
) -> bool:
    """Match exact argv, permitting only an equivalent Python argv[0] spelling."""

    if not command:
        return False
    bound_python = _bound_nested_python(child)
    actual_argv0 = _resolve_executable_reference(command[0])
    if (
        bound_python is None
        or actual_argv0 is None
        or actual_argv0 != proc_executable
        or bound_python[1] != proc_executable
    ):
        return False
    candidates = (command[0], bound_python[0])
    return any(
        hashlib.sha256(
            json.dumps(
                [argv0, *command[1:]],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        == expected_sha256
        for argv0 in candidates
    )


def _identity_is_live(identity: Mapping[str, Any]) -> bool:
    try:
        pid = int(identity["pid"])
        expected = {
            "pid": pid,
            "pgid": int(identity["pgid"]),
            "sid": int(identity["sid"]),
            "start_ticks": int(identity["start_ticks"]),
        }
    except (KeyError, TypeError, ValueError):
        return False
    current = _proc_identity(pid)
    return current is not None and all(
        current.get(field) == value for field, value in expected.items()
    )


def _instance_child_process_path(child: OwnedChild) -> Path:
    return child.output_root / "instance_child_process.json"


def _load_instance_child_process(
    child: OwnedChild,
) -> tuple[dict[str, Any] | None, str | None]:
    """Load and validate the exact nested instance-runner process contract."""

    path = _instance_child_process_path(child)
    try:
        raw, path_stat = _read_regular_file_no_follow(path)
        value = json.loads(raw)
    except FileNotFoundError:
        return None, "instance_child_process.json is missing"
    except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"instance_child_process.json is invalid: {error}"
    if not isinstance(value, dict):
        return None, "instance_child_process.json is not a regular JSON object"
    required = {
        "schema_version",
        "state",
        "pid",
        "pgid",
        "sid",
        "start_ticks",
        "runner_pid",
        "runner_pgid",
        "runner_sid",
        "runner_start_ticks",
        "task_name",
        "public_seed",
        "activity_instance_id",
        "entry_output_dir",
        "source_snapshot_root",
        "source_snapshot_binding_sha256",
        "cuda_device",
        "argv_sha256",
        "started_at",
        "updated_at",
        "action_deadline_s",
        "cleanup_deadline_s",
        "instance_timeout_s",
        "started_monotonic_ns",
        "action_deadline_monotonic_ns",
        "cleanup_deadline_monotonic_ns",
        "hard_deadline_monotonic_ns",
    }
    optional = {"returncode", "timed_out", "expected_run_nonce"}
    if not required.issubset(value) or not set(value).issubset(required | optional):
        return None, "instance_child_process.json fields do not match schema"
    if value.get(
        "schema_version"
    ) != INSTANCE_CHILD_PROCESS_SCHEMA_VERSION or value.get("state") not in {
        "running",
        "exited",
    }:
        return None, "instance child schema_version/state is invalid"
    integer_fields = (
        "pid",
        "pgid",
        "sid",
        "start_ticks",
        "runner_pid",
        "runner_pgid",
        "runner_sid",
        "runner_start_ticks",
        "public_seed",
        "activity_instance_id",
        "action_deadline_s",
        "cleanup_deadline_s",
        "instance_timeout_s",
        "started_monotonic_ns",
        "action_deadline_monotonic_ns",
        "cleanup_deadline_monotonic_ns",
        "hard_deadline_monotonic_ns",
    )
    if any(
        isinstance(value.get(field), bool)
        or not isinstance(value.get(field), int)
        or int(value[field]) < 0
        for field in integer_fields
    ):
        return None, "instance child integer identity is invalid"
    expected_instance = PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(
        child.public_seed,
        phase="eval",
    )
    expected_scalars = {
        "runner_pid": child.pid,
        "runner_pgid": child.pgid,
        "runner_sid": child.sid,
        "runner_start_ticks": child.start_ticks,
        "task_name": PICKING_UP_TRASH_TASK_SPEC.task_name,
        "public_seed": child.public_seed,
        "activity_instance_id": expected_instance,
        "entry_output_dir": str(child.entry_output_dir.resolve()),
        "source_snapshot_root": str(child.source_snapshot_root.resolve()),
        "source_snapshot_binding_sha256": child.source_snapshot_binding_sha256,
        "cuda_device": child.arm.gpu,
        "action_deadline_s": child.action_deadline_s,
        "cleanup_deadline_s": child.cleanup_deadline_s,
        "instance_timeout_s": child.instance_timeout_s,
        "started_monotonic_ns": child.started_monotonic_ns,
        "action_deadline_monotonic_ns": child.action_deadline_monotonic_ns,
        "cleanup_deadline_monotonic_ns": child.cleanup_deadline_monotonic_ns,
        "hard_deadline_monotonic_ns": child.hard_deadline_monotonic_ns,
    }
    child_expected_run_nonce = getattr(child, "expected_run_nonce", None)
    if child_expected_run_nonce is not None:
        expected_scalars["expected_run_nonce"] = child_expected_run_nonce
    if any(
        value.get(field) != expected for field, expected in expected_scalars.items()
    ):
        return None, "instance child binding differs from paired runner identity"
    if (
        _SHA256_RE.fullmatch(str(value.get("argv_sha256") or "")) is None
        or not isinstance(value.get("started_at"), str)
        or not value["started_at"]
        or not isinstance(value.get("updated_at"), str)
        or not value["updated_at"]
    ):
        return None, "instance child argv/timestamp binding is invalid"
    if "timed_out" in value and type(value["timed_out"]) is not bool:
        return None, "instance child timed_out must be boolean"
    if "returncode" in value and (
        value["returncode"] is not None
        and (
            isinstance(value["returncode"], bool)
            or not isinstance(value["returncode"], int)
        )
    ):
        return None, "instance child returncode is invalid"
    if value["pid"] != value["pgid"] or value["pid"] != value["sid"]:
        return None, "instance child must own a dedicated process session"
    if value["state"] == "running":
        current = _proc_identity(int(value["pid"]))
        if current is None or not _identity_is_live(value):
            return None, "running instance child process identity does not match /proc"
        # While the outer runner is alive, require the original direct
        # parent/child lineage. If the outer runner has already exited, init may
        # have reparented the still-running nested worker; in that case the
        # nested PID/PGID/SID/start-ticks and argv digest remain the cleanup
        # authority.
        if owned_child_identity_matches(child) and current["ppid"] != child.pid:
            return None, "instance child is not a direct child of the bound runner"
        try:
            raw_cmdline = (Path("/proc") / str(value["pid"]) / "cmdline").read_bytes()
            command = [
                os.fsdecode(item)
                for item in raw_cmdline.rstrip(b"\0").split(b"\0")
                if item
            ]
        except OSError as error:
            return None, f"instance child cmdline is unavailable: {error}"
        executable_before = _proc_executable(int(value["pid"]))
        final_identity = _proc_identity(int(value["pid"]))
        executable_after = _proc_executable(int(value["pid"]))
        if (
            final_identity != current
            or executable_before is None
            or executable_after != executable_before
        ):
            return None, "instance child /proc identity changed during validation"
        if not _nested_cmdline_matches_receipt(
            child,
            command=command,
            expected_sha256=str(value["argv_sha256"]),
            proc_executable=executable_before,
        ):
            return None, "instance child argv SHA-256 does not match /proc"
    if value["state"] == "exited" and _identity_is_live(value):
        return None, "instance child declares exited while exact identity is alive"
    return value, None


def _signal_instance_child(
    child: OwnedChild,
    *,
    sig: signal.Signals,
) -> tuple[str, str | None]:
    """Signal the bound nested child and distinguish sent, gone, and error."""

    identity, error = _load_instance_child_process(child)
    if error is not None:
        return "error", error
    assert identity is not None
    if identity["state"] == "exited":
        return "already_exited", None
    try:
        os.killpg(int(identity["pgid"]), sig)
    except ProcessLookupError:
        if _identity_is_live(identity):
            return "error", "instance child disappeared during signal"
        return "already_exited", None
    except OSError as error:
        return "error", f"instance child signal failed: {error}"
    return "sent", None


def _spawn_owned_child(
    *,
    arm: ArmSpec,
    public_seed: int,
    output_root: Path,
    entry_output_dir: Path,
    event_path: Path,
    log_path: Path,
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
    source_snapshot_binding_sha256: str,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
    started_monotonic_ns: int,
    action_deadline_monotonic_ns: int,
    cleanup_deadline_monotonic_ns: int,
    hard_deadline_monotonic_ns: int,
    expected_run_nonce: str | None = None,
) -> OwnedChild:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_stream.close()
    pid = int(process.pid)
    try:
        pgid = os.getpgid(pid)
        sid = os.getsid(pid)
    except OSError:
        process.terminate()
        process.wait(timeout=30)
        raise RuntimeError(f"{arm.name} child exited before identity capture")
    start_ticks = _proc_start_ticks(pid)
    if pgid != pid or sid != pid or start_ticks is None:
        process.terminate()
        process.wait(timeout=30)
        raise RuntimeError(f"{arm.name} child has no dedicated process session")
    started_monotonic = started_monotonic_ns / 1_000_000_000
    return OwnedChild(
        arm=arm,
        public_seed=public_seed,
        output_root=output_root,
        entry_output_dir=entry_output_dir,
        event_path=event_path,
        log_path=log_path,
        argv=argv,
        process=process,
        pid=pid,
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        started_monotonic=started_monotonic,
        source_snapshot_root=cwd.resolve(),
        source_snapshot_binding_sha256=source_snapshot_binding_sha256,
        action_deadline_s=action_deadline_s,
        cleanup_deadline_s=cleanup_deadline_s,
        instance_timeout_s=instance_timeout_s,
        started_monotonic_ns=started_monotonic_ns,
        action_deadline_monotonic_ns=action_deadline_monotonic_ns,
        cleanup_deadline_monotonic_ns=cleanup_deadline_monotonic_ns,
        hard_deadline_monotonic_ns=hard_deadline_monotonic_ns,
        action_deadline_monotonic=action_deadline_monotonic_ns / 1_000_000_000,
        cleanup_deadline_monotonic=cleanup_deadline_monotonic_ns / 1_000_000_000,
        hard_deadline_monotonic=hard_deadline_monotonic_ns / 1_000_000_000,
        expected_run_nonce=expected_run_nonce,
        safety_errors=[],
    )


def owned_child_identity_matches(child: OwnedChild) -> bool:
    """Verify the exact leader identity before authorizing a group signal."""

    if child.process.poll() is not None:
        return False
    try:
        return (
            os.getpgid(child.pid) == child.pgid
            and os.getsid(child.pid) == child.sid
            and child.pid == child.pgid == child.sid
            and _proc_start_ticks(child.pid) == child.start_ticks
            and child.pgid != os.getpgrp()
        )
    except OSError:
        return False


def _vla_identity_matches(live: LiveArm) -> bool:
    process = live.vla_process
    if (
        process is None
        or process.poll() is not None
        or live.vla_pgid is None
        or live.vla_sid is None
        or live.vla_start_ticks is None
    ):
        return False
    try:
        return (
            process.pid == live.vla_pgid == live.vla_sid
            and os.getpgid(process.pid) == live.vla_pgid
            and os.getsid(process.pid) == live.vla_sid
            and _proc_start_ticks(process.pid) == live.vla_start_ticks
            and live.vla_pgid != os.getpgrp()
        )
    except OSError:
        return False


def _prepare_persistent_vla_for_runner(
    live: LiveArm,
    *,
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Disable a verified persistent VLA before any runner may bind it."""

    if live.vla_endpoint is None or not _vla_identity_matches(live):
        raise RuntimeError(f"{live.spec.name} VLA identity is not verified")
    client = BehaviorVLAClient(live.vla_endpoint)
    try:
        before = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
        if before.get("config_name") != "pi05_behavior":
            raise RuntimeError(
                f"{live.spec.name} VLA health has unexpected config_name"
            )
        disabled = client.disable_actions(timeout_ms=5000)
        if disabled.get("actions_enabled") is not False:
            raise RuntimeError(
                f"{live.spec.name} VLA did not acknowledge action disable"
            )
        after = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
    finally:
        client.close()
    if after.get("config_name") != "pi05_behavior":
        raise RuntimeError(f"{live.spec.name} VLA health has unexpected config_name")
    if after.get("actions_enabled") is not False:
        raise RuntimeError(
            f"{live.spec.name} VLA health did not confirm actions_enabled=false"
        )
    live.vla_disabled_health = dict(after)
    return dict(after)


def _quiesce_persistent_vla_after_runner(
    live: LiveArm,
    child: OwnedChild,
    *,
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an owned persistent VLA to a verified idle gate after one runner."""

    if child.process.poll() is None:
        raise RuntimeError(f"{live.spec.name} runner is still active")
    if not child.cleanup_verified or child.identity_ambiguous:
        raise RuntimeError(f"{live.spec.name} runner cleanup is not verified")
    if live.vla_endpoint is None or not _vla_identity_matches(live):
        raise RuntimeError(f"{live.spec.name} VLA identity is not verified")
    expected_pid = int(live.vla_process.pid) if live.vla_process is not None else None
    client = BehaviorVLAClient(live.vla_endpoint)
    try:
        before = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
        if (
            before.get("config_name") != "pi05_behavior"
            or before.get("pid") != expected_pid
        ):
            raise RuntimeError(
                f"{live.spec.name} VLA pre-quiescence identity is unexpected"
            )
        disabled = client.disable_actions(timeout_ms=5000)
        if disabled.get("actions_enabled") is not False:
            raise RuntimeError(
                f"{live.spec.name} VLA did not acknowledge post-run action disable"
            )
        after = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
    finally:
        client.close()
    if not _vla_identity_matches(live):
        raise RuntimeError(f"{live.spec.name} VLA identity changed while quiescing")
    if (
        after.get("config_name") != "pi05_behavior"
        or after.get("pid") != expected_pid
        or after.get("actions_enabled") is not False
    ):
        raise RuntimeError(f"{live.spec.name} VLA post-run idle health is unexpected")
    child.post_run_vla_quiesced = True
    child.post_run_vla_health = dict(after)
    live.vla_disabled_health = dict(after)
    return dict(after)


def _formal_vla_log_dir(
    isolation: Any,
    *,
    formal_output_root: Path,
) -> Path:
    """Keep the VLA log while proving every runtime/cache path is external."""

    isolation_root = Path(isolation.root).resolve(strict=True)
    formal_root = formal_output_root.resolve()
    if (
        isolation_root == formal_root
        or isolation_root in formal_root.parents
        or formal_root in isolation_root.parents
    ):
        raise RuntimeError("external runtime isolation overlaps formal Eval output")
    paths = isolation.payload.get("paths")
    if not isinstance(paths, Mapping):
        raise RuntimeError("external runtime isolation has no bound path map")
    for name in (
        "omnigibson_appdata",
        "xdg_cache",
        "xdg_config",
        "xdg_data",
        "ov_cache",
        "omni_user",
        "tmp",
        "endpoints",
        "logs",
    ):
        value = paths.get(name)
        if not isinstance(value, str):
            raise RuntimeError(f"external runtime isolation lacks {name}")
        try:
            Path(value).resolve(strict=True).relative_to(isolation_root)
        except ValueError as error:
            raise RuntimeError(
                f"runtime/cache path escapes external isolation: {name}"
            ) from error
    log_root = formal_root / "launcher_logs" / "vla"
    log_root.mkdir(parents=True)
    return log_root.resolve(strict=True)


def _terminate_verified_vla(live: LiveArm) -> bool:
    """Terminate a VLA only while its full captured identity still matches."""

    process = live.vla_process
    if process is None or process.poll() is not None:
        return True
    if not _vla_identity_matches(live):
        live.disabled_reason = "vla_cleanup_identity_ambiguous"
        return False
    _terminate_process(process)
    return process.poll() is not None


def _remaining_before_hard_deadline(
    child: OwnedChild,
    *,
    deadline_monotonic: float | None = None,
) -> float:
    deadline = child.hard_deadline_monotonic
    if deadline_monotonic is not None:
        deadline = min(deadline, deadline_monotonic)
    return max(0.0, deadline - time.monotonic())


def _note_child_safety_error(child: OwnedChild, message: str) -> None:
    child.identity_ambiguous = True
    if child.safety_errors is None:
        child.safety_errors = []
    if message not in child.safety_errors:
        child.safety_errors.append(message)


def _disable_lane_after_error(
    live: LiveArm,
    child: OwnedChild,
    message: str,
) -> None:
    """Latch a lane-local blocker without aborting the independent peer."""

    _note_child_safety_error(child, message)
    live.disabled_reason = message


def _manifest_cleanup_within_deadline(
    child: OwnedChild,
    *,
    deadline_monotonic: float | None = None,
) -> bool:
    remaining = _remaining_before_hard_deadline(
        child,
        deadline_monotonic=deadline_monotonic,
    )
    effective_deadline_ns = min(
        child.hard_deadline_monotonic_ns,
        int(deadline_monotonic * 1_000_000_000)
        if deadline_monotonic is not None
        else child.hard_deadline_monotonic_ns,
    )
    if remaining <= 0:
        return not bool(
            _terminate_manifest_processes(
                child.entry_output_dir,
                timeout_s=0,
                hard_deadline_monotonic_ns=effective_deadline_ns,
            )
        )
    # The imported helper has both a TERM wait and a final KILL wait. Giving it
    # at most half the remaining budget keeps both phases inside the hard edge.
    timeout_s = min(10.0, max(0.0, remaining / 2.0))
    return not bool(
        _terminate_manifest_processes(
            child.entry_output_dir,
            timeout_s=timeout_s,
            hard_deadline_monotonic_ns=effective_deadline_ns,
        )
    )


def terminate_owned_child(
    child: OwnedChild,
    *,
    timeout_s: float = 30.0,
    deadline_monotonic: float | None = None,
) -> bool:
    """Stop nested/env/outer identities without waiting beyond the hard edge."""

    del timeout_s  # Absolute instance deadlines are the sole time authority.

    def cleanup_manifest() -> bool:
        if deadline_monotonic is None:
            return _manifest_cleanup_within_deadline(child)
        return _manifest_cleanup_within_deadline(
            child,
            deadline_monotonic=deadline_monotonic,
        )

    nested, nested_error = _load_instance_child_process(child)
    if nested_error is not None:
        _note_child_safety_error(child, nested_error)
    elif nested is not None and nested["state"] == "running":
        signal_status, error = _signal_instance_child(child, sig=signal.SIGTERM)
        if signal_status == "error":
            _note_child_safety_error(child, str(error))

    manifest_clean = cleanup_manifest()

    if child.process.poll() is None:
        if not owned_child_identity_matches(child):
            _note_child_safety_error(child, "outer runner identity is ambiguous")
        else:
            try:
                os.killpg(child.pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            remaining = _remaining_before_hard_deadline(
                child,
                deadline_monotonic=deadline_monotonic,
            )
            if remaining > 0:
                try:
                    child.process.wait(timeout=min(10.0, remaining))
                except subprocess.TimeoutExpired:
                    pass

    if (
        nested is not None
        and nested["state"] == "running"
        and _identity_is_live(nested)
    ):
        if (
            _remaining_before_hard_deadline(
                child,
                deadline_monotonic=deadline_monotonic,
            )
            > 0
        ):
            signal_status, error = _signal_instance_child(child, sig=signal.SIGKILL)
            if signal_status == "error":
                _note_child_safety_error(child, str(error))

    if child.process.poll() is None and owned_child_identity_matches(child):
        try:
            os.killpg(child.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        remaining = _remaining_before_hard_deadline(
            child,
            deadline_monotonic=deadline_monotonic,
        )
        if remaining > 0:
            try:
                child.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
    elif child.process.poll() is None:
        _note_child_safety_error(child, "outer runner identity changed before SIGKILL")

    if (
        _remaining_before_hard_deadline(
            child,
            deadline_monotonic=deadline_monotonic,
        )
        > 0
    ):
        manifest_clean = cleanup_manifest() and manifest_clean
    nested_alive = bool(nested is not None and _identity_is_live(nested))
    return (
        manifest_clean
        and not nested_alive
        and child.process.poll() is not None
        and not child.identity_ambiguous
    )


def _snapshot_child_environment(
    *,
    snapshot_root: Path,
    arm: ArmSpec,
    runtime_environment: Mapping[str, str],
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["RPENT_REPO_ROOT"] = str(snapshot_root)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(snapshot_root)
        if not existing
        else os.pathsep.join((str(snapshot_root), existing))
    )
    environment.update(
        {str(key): str(value) for key, value in runtime_environment.items()}
    )
    if environment.get("CUDA_VISIBLE_DEVICES") != arm.gpu:
        raise RuntimeError("runtime isolation GPU does not match campaign arm")
    return environment


@contextlib.contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            stream.bind((host, int(port)))
        except OSError:
            return False
    return True


def _port_accepts_connection(host: str, port: int) -> bool:
    """Observe a protected endpoint without binding, restarting, or signaling it."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.settimeout(1.0)
        return stream.connect_ex((host, int(port))) == 0


def _http_json(url: str, *, timeout_s: float = 2.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return None
    return payload if isinstance(payload, dict) else None


def _gpu_compute_processes() -> dict[str, list[dict[str, Any]]]:
    """Return physical GPU-indexed compute processes without signaling anything."""

    try:
        gpu_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.splitlines()
        uuid_to_index = {}
        for row in gpu_rows:
            index, uuid = (item.strip() for item in row.split(",", 1))
            if (
                not index.isdigit()
                or not uuid
                or uuid in uuid_to_index
                or index in uuid_to_index.values()
            ):
                raise ValueError(f"malformed GPU identity row: {row!r}")
            uuid_to_index[uuid] = index
        process_rows = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RuntimeError(f"GPU compute-process probe failed: {error}") from error
    processes: dict[str, list[dict[str, Any]]] = {
        index: [] for index in uuid_to_index.values()
    }
    for row in process_rows:
        parts = [item.strip() for item in row.split(",", 3)]
        if len(parts) != 4 or parts[0] not in uuid_to_index:
            raise RuntimeError(f"GPU process probe contains malformed row: {row!r}")
        try:
            pid = int(parts[1])
            used_memory_mib = int(parts[3])
        except ValueError as error:
            raise RuntimeError(
                f"GPU process probe contains invalid numeric fields: {row!r}"
            ) from error
        if pid <= 0 or used_memory_mib < 0 or not parts[2]:
            raise RuntimeError(f"GPU process probe contains invalid identity: {row!r}")
        start_ticks = _proc_start_ticks(pid)
        if start_ticks is None:
            raise RuntimeError(
                f"GPU process probe PID identity is unavailable: {row!r}"
            )
        processes.setdefault(uuid_to_index[parts[0]], []).append(
            {
                "pid": pid,
                "process_name": parts[2],
                "used_memory_mib": used_memory_mib,
                "start_ticks": start_ticks,
            }
        )
    return processes


def _pid_descends_from(pid: int, ancestors: set[int]) -> bool:
    current_pid = int(pid)
    seen: set[int] = set()
    for _ in range(64):
        if current_pid in ancestors:
            return True
        if current_pid <= 1 or current_pid in seen:
            return False
        seen.add(current_pid)
        current = _proc_identity(current_pid)
        if current is None:
            return False
        current_pid = current["ppid"]
    return False


def _unknown_gpu_processes(
    live: LiveArm,
    gpu_processes: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return GPU PIDs not attributable to exact campaign-owned process trees."""

    roots: set[int] = set()
    retired_env_identity: tuple[int, int] | None = None
    if _vla_identity_matches(live) and live.vla_process is not None:
        roots.add(int(live.vla_process.pid))
    child = live.child
    if child is not None and owned_child_identity_matches(child):
        roots.add(child.pid)
    if child is not None:
        nested, nested_error = _load_instance_child_process(child)
        if nested_error is None and nested is not None and nested["state"] == "running":
            roots.add(int(nested["pid"]))
        env = _recorded_env_health(child.entry_output_dir, owner_child=child)
        if (
            isinstance(env, dict)
            and env.get("ownership_valid") is True
            and isinstance(env.get("pid"), int)
        ):
            roots.add(int(env["pid"]))
        if (
            child.process.poll() is not None
            and bool(getattr(child, "cleanup_verified", False))
            and not bool(getattr(child, "identity_ambiguous", False))
            and nested_error is None
            and nested is not None
            and nested["state"] == "exited"
            and isinstance(env, dict)
            and env.get("identity_valid") is True
            and env.get("exact_identity_live") is False
            and isinstance(env.get("pid"), int)
            and not isinstance(env.get("pid"), bool)
            and isinstance(env.get("start_ticks"), int)
            and not isinstance(env.get("start_ticks"), bool)
        ):
            retired_env_identity = (int(env["pid"]), int(env["start_ticks"]))
    unknown: list[dict[str, Any]] = []
    for process in gpu_processes.get(live.spec.gpu, ()):
        pid = process.get("pid")
        expected_start_ticks = process.get("start_ticks")
        current_start_ticks = (
            _proc_start_ticks(pid)
            if isinstance(pid, int) and not isinstance(pid, bool)
            else None
        )
        if current_start_ticks is None and retired_env_identity == (
            pid,
            expected_start_ticks,
        ):
            # nvidia-smi can retain one stale row after an exact, run-bound
            # environment PID has exited.  The sealed PID/start-ticks pair
            # proves this is the retired owned process, not a reused PID.
            continue
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or isinstance(expected_start_ticks, bool)
            or not isinstance(expected_start_ticks, int)
            or current_start_ticks != expected_start_ticks
            or not _pid_descends_from(pid, roots)
        ):
            unknown.append(dict(process))
    return unknown


def _new_dashboard_state(
    *,
    arm: ArmSpec,
    public_seed: int,
    entry_output_dir: Path,
    action_deadline_s: int,
) -> State:
    spec = PICKING_UP_TRASH_TASK_SPEC
    instance_id = spec.instance_for_public_seed(public_seed, phase="eval")
    state = State(
        run_id=arm.run_id,
        name=f"{arm.name}_{spec.tag(public_seed)}",
        suite="behavior_2025_challenge",
        task=spec.task_index,
        seed=public_seed,
        environment="behavior",
        identity={
            "task_name": spec.task_name,
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": instance_id,
            "public_seed": public_seed,
            "cohort": arm.name,
        },
        output_dir=str(entry_output_dir),
        video_path=str(entry_output_dir / "episode.mp4"),
    )
    state.set_metadata(
        {
            "planner": "codex" if arm.llm_enabled else "disabled",
            "model": "gpt-5.5" if arm.llm_enabled else None,
            "reasoning-effort": "xhigh" if arm.llm_enabled else None,
            "task-name": spec.task_name,
            "task-language": spec.task_language,
            "task-index": spec.task_index,
            "activity-definition-id": spec.activity_definition_id,
            "activity-instance-id": instance_id,
            "public-instance-id": instance_id,
            "public-seed": public_seed,
            "public-seed-max": max(spec.public_seed_to_instance),
            "scene-model": spec.scene_model,
            "behavior-phase": "eval",
            "job-id": f"paired-eval-{arm.name}-s{public_seed}",
            "max-episode-steps": 24756,
            "max-tool-calls": 350 if arm.llm_enabled else None,
            "max-wall-clock-s": action_deadline_s,
            "public-tool-contract-version": (
                CURRENT_PUBLIC_TOOL_CONTRACT_VERSION if arm.llm_enabled else None
            ),
            "public-tool-count": len(arm.allowed_tools),
            "eval-cohort": arm.name,
            "controller": arm.controller,
            "llm-enabled": arm.llm_enabled,
            "cuda-device": arm.gpu,
            "health-status": "starting",
            "health-checked-at": _utc_now(),
        }
    )
    state.begin_attempt(
        attempt_index=1,
        output_dir=entry_output_dir,
        video_path=entry_output_dir / "episode.mp4",
    )
    if not arm.llm_enabled:
        state.set_budget_limits(max_tool_calls=None)
        state.on_usage(inp=0, out=0, tool_calls=0)
    return state


def _dashboard_event_relay(
    *,
    live: LiveArm,
    event_path: Path,
    state: State,
) -> DashboardEventRelay:
    return DashboardEventRelay(
        event_path,
        state,
        allowed_tools=live.spec.allowed_tools,
    )


def _read_result_record(
    runner_root: Path,
    entry_output_dir: Path,
) -> dict[str, Any] | None:
    candidates = (
        runner_root / "eval_results.jsonl",
        runner_root / "baseline_eval_results.jsonl",
        runner_root / "vla_eval_results.jsonl",
    )
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    for path in (
        entry_output_dir / "baseline_result.json",
        entry_output_dir / "final_result.json",
    ):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _recorded_env_health(
    entry_output_dir: Path,
    *,
    owner_child: OwnedChild | None = None,
) -> dict[str, Any] | None:
    """Read, but never signal, one run-bound environment process identity."""

    identities: list[dict[str, Any]] = []
    try:
        manifest_bytes, _ = _read_regular_file_no_follow(
            entry_output_dir / "run_manifest.json"
        )
        manifest = json.loads(manifest_bytes)
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        manifest = None
    if isinstance(manifest, dict):
        processes = manifest.get("processes")
        env = processes.get("env") if isinstance(processes, dict) else None
        if isinstance(env, dict):
            identities.append(env)
    try:
        baseline_bytes, _ = _read_regular_file_no_follow(
            entry_output_dir / "baseline_owned_processes.json"
        )
        baseline = json.loads(baseline_bytes)
    except (
        OSError,
        RuntimeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        baseline = None
    if isinstance(baseline, dict) and isinstance(baseline.get("env"), dict):
        identities.append(baseline["env"])
    if not identities:
        return None
    identity = identities[-1]
    try:
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
        sid = int(identity["sid"])
        start_ticks = int(identity["start_ticks"])
    except (KeyError, TypeError, ValueError):
        return {"identity_valid": False, "alive": None}
    current = _proc_identity(pid)
    receipt_identity_valid = pid == pgid == sid and start_ticks > 0
    exact_identity_live = bool(
        receipt_identity_valid
        and current is not None
        and current["pgid"] == pgid
        and current["sid"] == sid
        and current["start_ticks"] == start_ticks
    )
    lineage_roots: set[int] = set()
    cleanly_orphaned_from_completed_owner = False
    if owner_child is not None:
        if owned_child_identity_matches(owner_child):
            lineage_roots.add(owner_child.pid)
        nested, nested_error = _load_instance_child_process(owner_child)
        if nested_error is None and nested is not None and nested["state"] == "running":
            lineage_roots.add(int(nested["pid"]))
        cleanly_orphaned_from_completed_owner = bool(
            owner_child.process.poll() is not None
            and nested_error is None
            and nested is not None
            and nested["state"] == "exited"
        )
    lineage_valid = bool(lineage_roots and _pid_descends_from(pid, lineage_roots))
    ownership_valid = bool(
        exact_identity_live
        and (
            owner_child is None
            or lineage_valid
            or cleanly_orphaned_from_completed_owner
        )
    )
    return {
        "pid": pid,
        "pgid": pgid,
        "sid": sid,
        "start_ticks": start_ticks,
        "identity_valid": receipt_identity_valid,
        "exact_identity_live": exact_identity_live,
        "lineage_valid": lineage_valid,
        "ownership_valid": ownership_valid,
        "alive": exact_identity_live,
    }


def _strict_action_trace_summary(
    entry_output_dir: Path,
    *,
    expected_run_nonce: str | None = None,
) -> dict[str, Any]:
    """Read one immutable trace and delegate success semantics to the shared parser."""

    path = entry_output_dir / "behavior_action_trace.jsonl"
    try:
        payload, _path_stat = _read_regular_file_no_follow(path)
    except (OSError, RuntimeError) as error:
        return {
            "valid": False,
            "error": f"behavior action trace is unavailable: {error}",
            "action_trace_sha256": None,
            "official_success_binding": None,
        }
    digest = hashlib.sha256(payload).hexdigest()
    binding_count = 0
    bound_run_nonce: str | None = None
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {
                "valid": False,
                "error": f"behavior action trace has malformed JSON at line {line_number}",
                "action_trace_sha256": digest,
                "official_success_binding": None,
            }
        if not isinstance(record, dict):
            return {
                "valid": False,
                "error": (
                    "behavior action trace has a non-object record at "
                    f"line {line_number}"
                ),
                "action_trace_sha256": digest,
                "official_success_binding": None,
            }
        if record.get("event") == "rpent_run_binding":
            binding_count += 1
            bound_run_nonce = record.get("run_nonce")
            if (
                record.get("attempt_index") != 1
                or not isinstance(bound_run_nonce, str)
                or re.fullmatch(r"[0-9a-f]{32}", bound_run_nonce) is None
            ):
                return {
                    "valid": False,
                    "error": "behavior action trace has an invalid run nonce binding",
                    "action_trace_sha256": digest,
                    "official_success_binding": None,
                }
    if expected_run_nonce is not None and (
        binding_count != 1 or bound_run_nonce != expected_run_nonce
    ):
        return {
            "valid": False,
            "error": "behavior action trace run nonce binding mismatch",
            "action_trace_sha256": digest,
            "official_success_binding": None,
        }
    binding = summarize_action_trace_success(payload)
    if binding is not None and binding.get("action_trace_sha256") != digest:
        return {
            "valid": False,
            "error": "shared success summarizer returned a mismatched trace SHA-256",
            "action_trace_sha256": digest,
            "official_success_binding": None,
        }
    if binding is not None and expected_run_nonce is not None:
        binding = {**binding, "run_nonce": bound_run_nonce}
    return {
        "valid": True,
        "error": None,
        "action_trace_sha256": digest,
        "official_success_binding": binding,
    }


def _stop_relay_within_child_deadline(
    live: LiveArm,
    child: OwnedChild | None,
) -> None:
    if live.relay is None:
        return
    timeout_s = (
        _remaining_before_hard_deadline(child)
        if child is not None and hasattr(child, "hard_deadline_monotonic")
        else None
    )
    live.relay.stop(timeout_s=timeout_s)


def _release_child_if_cleanup_verified(live: LiveArm) -> bool:
    """Drop an owned-child reference only after exact cleanup is complete."""

    child = live.child
    if child is None:
        return True
    if not child.cleanup_verified or child.process.poll() is None:
        return False
    live.child = None
    return True


def _is_sealed_agentic_task_failure(record: Mapping[str, Any] | None) -> bool:
    """Accept Agentic's aggregate exit 1 only for a canonical inner task failure."""

    return bool(
        isinstance(record, Mapping)
        and record.get("outcome") == "task_failed"
        and record.get("task_success") is False
        and record.get("raw_official_success") is False
        and record.get("raw_official_success_binding") is None
        and record.get("first_raw_success_env_step") is None
        and record.get("timed_out") is False
        and isinstance(record.get("subprocess_exit_code"), int)
        and not isinstance(record.get("subprocess_exit_code"), bool)
        and record.get("subprocess_exit_code") == 0
        and record.get("instance_state_binding_valid") is True
        and record.get("validation_errors") == []
        and record.get("infrastructure_error") is None
        and record.get("artifact_seal_complete") is True
        and record.get("workflow_complete") is True
    )


def _cleanup_spawned_pair_on_barrier_failure(
    live_arms: Iterable[LiveArm],
    *,
    public_seed: int,
) -> tuple[str | None, tuple[str, ...]]:
    """Enforce all-or-nothing launch and clean any already-started peer."""

    arms = tuple(live_arms)
    unavailable = [
        live
        for live in arms
        if live.child is None or live.child.public_seed != public_seed
    ]
    if not unavailable:
        return None, ()
    reason = "pair_barrier_failed: " + "; ".join(
        f"{live.spec.name}={live.disabled_reason or 'lane_unavailable'}"
        for live in unavailable
    )
    errors: list[str] = [reason]
    for live in arms:
        child = live.child
        if child is None:
            continue
        _stop_relay_within_child_deadline(live, child)
        cleanup_complete = terminate_owned_child(child)
        child.cleanup_verified = cleanup_complete
        if not cleanup_complete:
            errors.append(f"{live.spec.name}_pair_barrier_cleanup_unverified")
    return reason, tuple(errors)


def _finish_dashboard_arm(live: LiveArm, child: OwnedChild) -> dict[str, Any]:
    _stop_relay_within_child_deadline(live, child)
    record = _read_result_record(child.output_root, child.entry_output_dir)
    runner_outcome = (
        str(
            record.get("outcome")
            or (
                "passed"
                if record.get("task_success") is True
                else "task_failed"
                if record.get("task_success") is False
                else "run_error"
            )
        )
        if record
        else "run_error"
    )
    trace_summary = _strict_action_trace_summary(
        child.entry_output_dir,
        expected_run_nonce=getattr(child, "expected_run_nonce", None),
    )
    raw_success_binding = trace_summary["official_success_binding"]
    raw_success = raw_success_binding is not None
    timed_out = bool(child.timed_out or (record and record.get("timed_out") is True))
    infrastructure_error = (
        record.get("infrastructure_error") if isinstance(record, dict) else None
    )
    sealed_agentic_task_failure = bool(
        live.spec.llm_enabled
        and child.process.returncode == 1
        and _is_sealed_agentic_task_failure(record)
    )
    gpu_ownership_error = bool(live.gpu_ownership_violations)
    cleanup_identity_error = bool(
        getattr(child, "identity_ambiguous", False)
        or getattr(child, "cleanup_verified", True) is False
    )
    peer_abort_reason = getattr(child, "peer_abort_reason", None)
    child_safety_errors = tuple(getattr(child, "safety_errors", None) or ())
    action_deadline_exhausted = bool(getattr(child, "action_deadline_exhausted", False))
    deadline_cleanup_artifact: dict[str, Any] | None = None
    if action_deadline_exhausted and (
        runner_outcome in {"run_error", "incomplete", "not_run"}
        or child.process.returncode not in {0, None}
        or not isinstance(record, Mapping)
        or (
            isinstance(record, Mapping)
            and record.get("validation_errors") not in (None, (), [], "")
        )
    ):
        deadline_cleanup_artifact = {
            "runner_outcome": runner_outcome,
            "returncode": child.process.returncode,
            "result_record_missing": not isinstance(record, Mapping),
            "validation_errors": (
                record.get("validation_errors") if isinstance(record, Mapping) else None
            ),
        }
    success_infrastructure_errors: list[dict[str, Any]] = []
    artifact_seal_complete = bool(
        isinstance(record, Mapping)
        and record.get("artifact_seal_complete") is True
        and child.process.returncode == 0
        and not bool(getattr(child, "forced_cleanup_started", False))
    )
    if raw_success and not artifact_seal_complete:
        success_infrastructure_errors.append(
            {
                "reason": "artifact_seal_incomplete",
                "result_record_missing": not isinstance(record, Mapping),
                "returncode": child.process.returncode,
                "forced_cleanup": bool(getattr(child, "forced_cleanup_started", False)),
                "recorded_artifact_seal_complete": (
                    record.get("artifact_seal_complete")
                    if isinstance(record, Mapping)
                    else None
                ),
            }
        )
    if cleanup_identity_error:
        success_infrastructure_errors.append(
            {
                "reason": "cleanup_or_identity_unverified",
                "identity_ambiguous": bool(getattr(child, "identity_ambiguous", False)),
                "cleanup_verified": bool(getattr(child, "cleanup_verified", False)),
            }
        )
    if gpu_ownership_error:
        success_infrastructure_errors.append(
            {
                "reason": "gpu_ownership_violation",
                "violations": list(live.gpu_ownership_violations),
            }
        )
    if peer_abort_reason:
        success_infrastructure_errors.append(
            {
                "reason": "peer_infrastructure_abort",
                "detail": str(peer_abort_reason),
            }
        )
    if child_safety_errors:
        success_infrastructure_errors.append(
            {
                "reason": "child_safety_errors",
                "errors": list(child_safety_errors),
            }
        )
    if trace_summary["valid"] is not True:
        task_success = None
        score = "unknown"
    elif raw_success:
        task_success = True
        score = "success"
        if success_infrastructure_errors:
            if infrastructure_error not in (None, "", False):
                success_infrastructure_errors.insert(
                    0,
                    {
                        "reason": "runner_infrastructure_error",
                        "detail": infrastructure_error,
                    },
                )
            infrastructure_error = (
                success_infrastructure_errors[0]
                if len(success_infrastructure_errors) == 1
                else {
                    "reason": "multiple_infrastructure_errors",
                    "errors": success_infrastructure_errors,
                }
            )
    elif cleanup_identity_error:
        task_success = None
        score = "unknown"
        infrastructure_error = {
            "reason": "cleanup_or_identity_unverified",
            "identity_ambiguous": bool(getattr(child, "identity_ambiguous", False)),
            "cleanup_verified": bool(getattr(child, "cleanup_verified", False)),
        }
    elif gpu_ownership_error:
        task_success = None
        score = "unknown"
        infrastructure_error = {
            "reason": "gpu_ownership_violation",
            "violations": list(live.gpu_ownership_violations),
        }
    elif peer_abort_reason:
        task_success = None
        score = "unknown"
        infrastructure_error = {
            "reason": "peer_infrastructure_abort",
            "detail": str(peer_abort_reason),
        }
    elif child_safety_errors:
        task_success = None
        score = "unknown"
        infrastructure_error = {
            "reason": "child_safety_errors",
            "errors": list(child_safety_errors),
        }
    elif infrastructure_error not in (None, "", False):
        task_success = None
        score = "unknown"
    elif timed_out:
        task_success = False
        score = "failure"
    elif runner_outcome in {
        "run_error",
        "incomplete",
        "not_run",
    } or (
        child.process.returncode not in {0, None} and not sealed_agentic_task_failure
    ):
        task_success = None
        score = "unknown"
    else:
        task_success = False
        score = "failure"
    outcome = (
        "passed"
        if score == "success"
        else "run_error"
        if score == "unknown"
        else "timed_out"
        if timed_out
        else "task_failed"
    )
    if live.state is not None:
        if task_success is True:
            workflow_trusted = infrastructure_error in (None, "", False)
            live.state.on_event(
                {
                    "type": "official_success",
                    "attempt_index": 1,
                    "task_success": True,
                    "artifact_seal_complete": bool(
                        workflow_trusted
                        and record
                        and record.get("artifact_seal_complete")
                    ),
                    "workflow_complete": bool(
                        workflow_trusted
                        and record
                        and record.get("artifact_seal_complete")
                    ),
                    "publication_complete": False,
                }
            )
        live.state.end_attempt(attempt_index=1, outcome=outcome)
        live.state.mark_done(task_success is True)
    return {
        "outcome": outcome,
        "runner_outcome": runner_outcome,
        "task_success": task_success,
        "score": score,
        "raw_success_confirmed": raw_success,
        "raw_official_success_binding": raw_success_binding,
        "artifact_seal_complete": artifact_seal_complete,
        "action_trace_binding": {
            "source": "behavior_action_trace",
            "field_path": "info_done.success",
            "sha256": trace_summary["action_trace_sha256"],
            "valid": trace_summary["valid"],
            "error": trace_summary["error"],
        },
        "result": record,
        "returncode": child.process.returncode,
        "timed_out": timed_out,
        "action_deadline_exhausted": action_deadline_exhausted,
        "deadline_cleanup_artifact": deadline_cleanup_artifact,
        "post_run_vla_quiesced": bool(getattr(child, "post_run_vla_quiesced", False)),
        "post_run_vla_health": getattr(child, "post_run_vla_health", None),
        "infrastructure_error": infrastructure_error,
        "peer_abort_reason": peer_abort_reason,
        "identity_ambiguous": child.identity_ambiguous,
        "safety_errors": list(child_safety_errors),
        "absolute_deadlines": {
            "started_monotonic_ns": getattr(child, "started_monotonic_ns", None),
            "action_deadline_monotonic_ns": getattr(
                child, "action_deadline_monotonic_ns", None
            ),
            "cleanup_deadline_monotonic_ns": getattr(
                child, "cleanup_deadline_monotonic_ns", None
            ),
            "hard_deadline_monotonic_ns": getattr(
                child, "hard_deadline_monotonic_ns", None
            ),
        },
        "relay_violations": list(live.relay.violations if live.relay else ()),
        "records_relayed": live.relay.records_relayed if live.relay else 0,
    }


def _enforce_instance_state_result_binding(
    *,
    arm_result: dict[str, Any],
    entry_output_dir: Path,
    expected_sha256: str,
) -> str | None:
    """Bind the frozen state without erasing trace-confirmed raw success."""

    candidates: list[Mapping[str, Any]] = []
    result = arm_result.get("result")
    if isinstance(result, Mapping):
        candidates.append(result)
        nested = result.get("result")
        if isinstance(nested, Mapping):
            candidates.append(nested)
    for path in (
        entry_output_dir / "final_result.json",
        entry_output_dir / "baseline_result.json",
    ):
        try:
            payload, _ = _read_regular_file_no_follow(path)
            value = json.loads(payload)
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            continue
        if isinstance(value, Mapping):
            candidates.append(value)
    recorded = next(
        (
            str(candidate["instance_state_sha256"])
            for candidate in candidates
            if _SHA256_RE.fullmatch(str(candidate.get("instance_state_sha256") or ""))
        ),
        None,
    )
    if recorded == expected_sha256:
        arm_result["instance_state_sha256"] = recorded
        return None
    reason = (
        "instance_state_sha256_missing"
        if recorded is None
        else "instance_state_sha256_mismatch"
    )
    if (
        arm_result.get("raw_success_confirmed") is True
        and arm_result.get("task_success") is True
    ):
        arm_result.update(
            {
                "infrastructure_error": reason,
                "instance_state_sha256": recorded,
                "expected_instance_state_sha256": expected_sha256,
            }
        )
        return reason
    arm_result.update(
        {
            "outcome": "run_error",
            "task_success": None,
            "score": "unknown",
            "infrastructure_error": reason,
            "instance_state_sha256": recorded,
            "expected_instance_state_sha256": expected_sha256,
        }
    )
    return reason


def _pair_infrastructure_blockers(
    pair_record: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return fail-closed campaign blockers from one completed pair."""

    arms = pair_record.get("arms")
    if not isinstance(arms, Mapping) or not arms:
        return ("pair_result_arms_missing",)
    expected_arms = {"agentic", "pi0_nav_pick_only"}
    actual_arms = {str(name) for name in arms}
    blockers: list[str] = []
    if actual_arms != expected_arms:
        blockers.append(
            "pair_result_arm_set_invalid: "
            f"expected={sorted(expected_arms)!r} actual={sorted(actual_arms)!r}"
        )
    for arm_name, value in arms.items():
        if not isinstance(value, Mapping):
            blockers.append(f"{arm_name}_pair_result_invalid")
            continue
        outcome = str(value.get("outcome") or "")
        score = str(value.get("score") or "")
        task_success = value.get("task_success")
        infrastructure_error = value.get("infrastructure_error")
        canonical = bool(
            (outcome == "passed" and score == "success" and task_success is True)
            or (
                outcome == "task_failed"
                and score == "failure"
                and task_success is False
            )
            or (outcome == "timed_out" and score == "failure" and task_success is False)
        )
        canonical = canonical and infrastructure_error in (None, "", False)
        if canonical:
            continue
        reason = (
            infrastructure_error
            or value.get("reason")
            or (
                "noncanonical_pair_result="
                f"{outcome or '<missing>'}/{score or '<missing>'}/"
                f"{task_success!r}"
            )
        )
        blockers.append(f"{arm_name}_infrastructure_unknown: {reason}")
    return tuple(blockers)


def _blocked_remaining_pair_record(
    *,
    public_seed: int,
    live_arms: Iterable[LiveArm],
    reason: str,
) -> dict[str, Any]:
    """Build one non-executed result after a campaign-level blocker."""

    return {
        "schema_version": 1,
        "public_seed": public_seed,
        "activity_instance_id": (
            PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(
                public_seed,
                phase="eval",
            )
        ),
        "arms": {
            live.spec.name: {
                "outcome": "not_run",
                "task_success": None,
                "score": "unknown",
                "reason": reason,
                "infrastructure_error": {
                    "reason": "campaign_blocked_by_prior_pair",
                    "detail": reason,
                },
            }
            for live in live_arms
        },
    }


def _pair_campaign_transition(
    *,
    public_seeds: tuple[int, ...],
    pair_index: int,
    pair_record: Mapping[str, Any],
    pair_safety_errors: Iterable[str],
    live_arms: Iterable[LiveArm],
) -> tuple[str | None, tuple[dict[str, Any], ...]]:
    """Stop only for shared safety failures, never for one arm's outcome."""

    # Agentic and Pure-VLA are independent lanes. A task, transport, or runner
    # outcome on one lane is preserved in that arm's record and never aborts
    # the other lane or later work. Only shared/ownership safety failures stop
    # the campaign.
    errors = tuple(pair_safety_errors)
    if not errors:
        return None, ()
    reason = "; ".join(str(error) for error in errors)
    remaining = tuple(
        _blocked_remaining_pair_record(
            public_seed=public_seed,
            live_arms=live_arms,
            reason=reason,
        )
        for public_seed in public_seeds[pair_index + 1 :]
    )
    return reason, remaining


def _health_record(
    *,
    live: LiveArm,
    offset_s: int,
    gpu_processes: Mapping[str, list[dict[str, Any]]],
    free_disk_bytes: int,
    previous: Mapping[str, Any] | None = None,
    gpu_probe_error: str | None = None,
    external_runtime_health: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    child = live.child
    state_snapshot = live.state.snapshot() if live.state is not None else None
    dashboard_health = _http_json(f"{live.url}/healthz")
    run_health = (
        _http_json(f"{live.url}/api/run?run={live.spec.run_id}")
        if dashboard_health is not None
        else None
    )
    vla_health = (
        _http_json(f"{live.vla_endpoint}/healthz")
        if live.vla_endpoint is not None
        else None
    )
    event_size = None
    if child is not None:
        try:
            event_size = child.event_path.stat().st_size
        except OSError:
            event_size = None
    warnings: list[str] = []
    if dashboard_health is None or run_health is None:
        warnings.append("dashboard_http_unhealthy")
    if gpu_probe_error is not None:
        warnings.append("gpu_compute_process_probe_failed")
    if vla_health is None:
        warnings.append("vla_health_unavailable")
    if live.vla_process is not None and not _vla_identity_matches(live):
        warnings.append("vla_process_identity_ambiguous")
    if (
        child is not None
        and child.process.poll() is None
        and not owned_child_identity_matches(child)
    ):
        warnings.append("child_process_identity_ambiguous")
    env_process = (
        _recorded_env_health(child.entry_output_dir, owner_child=child)
        if child
        else None
    )
    if (
        child is not None
        and child.process.poll() is None
        and env_process is not None
        and env_process.get("alive") is False
    ):
        warnings.append("env_process_not_alive")
    if (
        offset_s > 0
        and child is not None
        and child.process.poll() is None
        and env_process is None
    ):
        warnings.append("env_process_identity_unavailable")
    arm_gpu_processes = list(gpu_processes.get(live.spec.gpu, ()))
    unknown_gpu_processes = _unknown_gpu_processes(live, gpu_processes)
    if live.vla_process is not None and not arm_gpu_processes:
        warnings.append("gpu_compute_process_unavailable")
    if unknown_gpu_processes:
        warnings.append("unknown_gpu_compute_process")
        violation = "unknown_gpu_compute_pids=" + ",".join(
            str(item.get("pid")) for item in unknown_gpu_processes
        )
        if violation not in live.gpu_ownership_violations:
            live.gpu_ownership_violations.append(violation)
    if gpu_probe_error is not None:
        violation = f"gpu_probe_failed={gpu_probe_error}"
        if violation not in live.gpu_ownership_violations:
            live.gpu_ownership_violations.append(violation)
    if free_disk_bytes < DEFAULT_MIN_FREE_DISK_BYTES:
        warnings.append("free_disk_below_required_threshold")
    runtime_errors = (
        tuple(str(error) for error in external_runtime_health.get("errors", ()))
        if isinstance(external_runtime_health, Mapping)
        else ()
    )
    if runtime_errors:
        warnings.append("external_runtime_unhealthy")
        for error in runtime_errors:
            if error not in live.runtime_violations:
                live.runtime_violations.append(error)
    trace_health = (
        _strict_action_trace_summary(
            child.entry_output_dir,
            expected_run_nonce=getattr(child, "expected_run_nonce", None),
        )
        if child is not None
        else None
    )
    raw_success_binding = (
        trace_health.get("official_success_binding")
        if isinstance(trace_health, Mapping)
        else None
    )
    if raw_success_binding is not None and live.state is not None:
        live.state.on_event(
            {
                "type": "official_success",
                "attempt_index": 1,
                "task_success": True,
                "artifact_seal_complete": False,
                "workflow_complete": False,
                "publication_complete": False,
            }
        )
        state_snapshot = live.state.snapshot()
    if (
        isinstance(state_snapshot, Mapping)
        and state_snapshot.get("publication_complete") is True
    ):
        warnings.append("eval_publication_must_remain_false")
    current_progress_identity = {
        "public_seed": child.public_seed if child is not None else None,
        "frame_idx": (
            state_snapshot.get("frame_idx")
            if isinstance(state_snapshot, dict)
            else None
        ),
        "timeline_revision": (
            state_snapshot.get("timeline_revision")
            if isinstance(state_snapshot, dict)
            else None
        ),
        "event_sink_size_bytes": event_size,
    }
    if (
        offset_s > 0
        and child is not None
        and child.process.poll() is None
        and isinstance(previous, Mapping)
        and previous.get("public_seed") == child.public_seed
        and all(
            previous.get(key) == current_progress_identity.get(key)
            for key in ("frame_idx", "timeline_revision", "event_sink_size_bytes")
        )
    ):
        warnings.append("no_progress_since_previous_sample")
    record = {
        "schema_version": 1,
        "checked_at": _utc_now(),
        "offset_s": int(offset_s),
        "cohort": live.spec.name,
        "gpu": live.spec.gpu,
        "public_seed": child.public_seed if child is not None else None,
        "child": (
            {
                "pid": child.pid,
                "pgid": child.pgid,
                "sid": child.sid,
                "start_ticks": child.start_ticks,
                "alive": child.process.poll() is None,
                "identity_matches": owned_child_identity_matches(child),
                "argv_sha256": child.argv_sha256,
            }
            if child is not None
            else None
        ),
        "vla": {
            "endpoint": live.vla_endpoint,
            "pid": live.vla_process.pid if live.vla_process is not None else None,
            "pgid": live.vla_pgid,
            "sid": live.vla_sid,
            "start_ticks": live.vla_start_ticks,
            "alive": bool(
                live.vla_process is not None and live.vla_process.poll() is None
            ),
            "identity_matches": _vla_identity_matches(live),
            "healthz": vla_health is not None,
        },
        "env": env_process,
        "dashboard": {
            "url": live.url,
            "healthz": dashboard_health is not None,
            "run_api": run_health is not None,
        },
        "progress": state_snapshot,
        "event_sink_size_bytes": event_size,
        "gpu_compute_processes": arm_gpu_processes,
        "unknown_gpu_compute_processes": unknown_gpu_processes,
        "gpu_probe_error": gpu_probe_error,
        "free_disk_bytes": int(free_disk_bytes),
        "external_runtime": (
            dict(external_runtime_health)
            if isinstance(external_runtime_health, Mapping)
            else None
        ),
        "raw_official_success": {
            "confirmed": raw_success_binding is not None,
            "binding": raw_success_binding,
            "action_trace_sha256": (
                trace_health.get("action_trace_sha256")
                if isinstance(trace_health, Mapping)
                else None
            ),
            "trace_valid": (
                trace_health.get("valid") if isinstance(trace_health, Mapping) else None
            ),
            "trace_error": (
                trace_health.get("error") if isinstance(trace_health, Mapping) else None
            ),
        },
        "eval_publication_complete": False,
        "warnings": warnings,
        "healthy": not warnings,
        "progress_identity": current_progress_identity,
    }
    if live.state is not None:
        status = "ok" if not warnings else "warning"
        live.state.set_metadata(
            {
                "health-status": status,
                "health-checked-at": record["checked_at"],
            }
        )
        live.state.on_event(
            {
                "type": "meta",
                "tag": "health",
                "text": (
                    f"t+{int(offset_s // 60)}m health {status}"
                    + (f": {', '.join(warnings)}" if warnings else "")
                ),
            }
        )
    return record


def _sample_health(
    *,
    live_arms: Iterable[LiveArm],
    offset_s: int,
    output_root: Path,
    previous: dict[str, dict[str, Any]],
    owned_runtime_root: OwnedRuntimeRoot | None = None,
    runtime_owner_document: Mapping[str, Any] | None = None,
    runtime_isolation: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    live_arms = tuple(live_arms)
    try:
        gpu_processes = _gpu_compute_processes()
        gpu_probe_error = None
    except RuntimeError as error:
        gpu_processes = {}
        gpu_probe_error = str(error)
    free_disk_bytes = shutil.disk_usage(output_root).free
    external_runtime_health: dict[str, Any] | None = None
    runtime_errors: tuple[str, ...] = ()
    shared_runtime_errors: list[str] = []
    lane_runtime_errors: dict[str, list[str]] = {
        live.spec.name: [] for live in live_arms
    }
    if (
        owned_runtime_root is not None
        and runtime_owner_document is not None
        and runtime_isolation is not None
    ):
        runtime_errors = tuple(
            _external_runtime_active_errors(
                owned_runtime_root,
                owner_document=runtime_owner_document,
                runtime_isolation=runtime_isolation,
            )
        )
        shared_runtime_errors, lane_runtime_errors = _partition_runtime_errors(
            runtime_errors,
            lane_names=(live.spec.name for live in live_arms),
        )
        try:
            runtime_free_bytes = shutil.disk_usage(owned_runtime_root.base).free
        except OSError:
            runtime_free_bytes = None
        external_runtime_health = {
            "runtime_base": str(owned_runtime_root.base),
            "runtime_root": str(owned_runtime_root.root),
            "runtime_base_device": owned_runtime_root.base_device,
            "runtime_base_inode": owned_runtime_root.base_inode,
            "runtime_root_device": owned_runtime_root.root_device,
            "runtime_root_inode": owned_runtime_root.root_inode,
            "owner_binding_sha256": runtime_owner_document.get("binding_sha256"),
            "arm_binding_sha256": {
                name: isolation.binding_sha256
                for name, isolation in sorted(runtime_isolation.items())
            },
            "free_bytes": runtime_free_bytes,
            "errors": list(runtime_errors),
        }
    for live in live_arms:
        live_external_runtime_health = external_runtime_health
        if external_runtime_health is not None:
            live_external_runtime_health = {
                **external_runtime_health,
                "errors": [
                    *shared_runtime_errors,
                    *lane_runtime_errors.get(live.spec.name, ()),
                ],
            }
        record = _health_record(
            live=live,
            offset_s=offset_s,
            gpu_processes=gpu_processes,
            free_disk_bytes=free_disk_bytes,
            previous=previous.get(live.spec.name),
            gpu_probe_error=gpu_probe_error,
            external_runtime_health=live_external_runtime_health,
        )
        _append_jsonl(live.spec.output_root / "health_checks.jsonl", record)
        previous[live.spec.name] = dict(record["progress_identity"])
    return tuple(shared_runtime_errors)


def _publication_binding(
    publication: ValidatedBehaviorPublication,
) -> dict[str, Any]:
    return {
        "root": str(publication.root),
        "task_name": publication.identity.task_spec.task_name,
        "task_index": publication.identity.task_spec.task_index,
        "public_seed": publication.identity.public_seed,
        "activity_instance_id": publication.identity.native_instance,
        "tag": publication.identity.tag,
        "bundle_id": publication.bundle_id,
        "provenance_sha256": hashlib.sha256(publication.provenance_bytes).hexdigest(),
        "manifest_binding": publication.manifest_binding,
        "files": publication.files,
    }


def _validate_agentic_publication(
    *,
    root: Path,
    provenance_sha256: str,
) -> dict[str, Any]:
    publication = validate_canonical_publication_root(
        root,
        expected_provenance_sha256=provenance_sha256,
        task_name=PICKING_UP_TRASH_TASK_SPEC.task_name,
        task_index=PICKING_UP_TRASH_TASK_SPEC.task_index,
        public_seed=CANONICAL_AGENTIC_SOURCE_PUBLIC_SEED,
    )
    if (
        publication.identity.public_seed != CANONICAL_AGENTIC_SOURCE_PUBLIC_SEED
        or publication.identity.native_instance
        != PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(
            CANONICAL_AGENTIC_SOURCE_PUBLIC_SEED,
            phase="explore",
        )
    ):
        raise PublicationValidationError(
            "agentic Eval input is not the canonical picking_up_trash s3 publication"
        )
    return _publication_binding(publication)


def _regular_file_fingerprint(path: Path) -> dict[str, Any]:
    payload, _file_stat = _read_regular_file_no_follow(path)
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _freeze_public_instance_states(
    *,
    behavior_repo: Path,
    public_seeds: Iterable[int],
) -> dict[str, dict[str, Any]]:
    spec = PICKING_UP_TRASH_TASK_SPEC
    state_dir = (
        behavior_repo
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
        / "scenes"
        / spec.scene_model
        / "json"
        / spec.state_dir_name
    )
    resolved_dir = state_dir.resolve(strict=True)
    if state_dir.is_symlink() or not resolved_dir.is_dir():
        raise RuntimeError("BEHAVIOR instance state directory is not safe")
    bindings: dict[str, dict[str, Any]] = {}
    for public_seed in public_seeds:
        phase = (
            "explore"
            if public_seed in spec.explore_public_seeds
            else "eval"
            if public_seed in spec.eval_public_seeds
            else "invalid"
        )
        instance_id = spec.instance_for_public_seed(public_seed, phase=phase)
        matches = sorted(
            state_dir.glob(
                f"*_{spec.activity_definition_id}_{instance_id}_template-tro_state.json"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one state for {spec.task_name} s{public_seed}, "
                f"found {len(matches)}"
            )
        fingerprint = _regular_file_fingerprint(matches[0])
        Path(fingerprint["path"]).relative_to(resolved_dir)
        bindings[str(public_seed)] = {
            "task_name": spec.task_name,
            "task_index": spec.task_index,
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": instance_id,
            "scene_model": spec.scene_model,
            "public_seed": public_seed,
            "phase": phase,
            **fingerprint,
        }
    return bindings


def _verify_public_instance_states(
    bindings: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    for public_seed, expected in bindings.items():
        try:
            current = _regular_file_fingerprint(Path(str(expected["path"])))
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"s{public_seed}_state_unavailable: {error}")
            continue
        for field in ("path", "size_bytes", "sha256"):
            if current.get(field) != expected.get(field):
                errors.append(f"s{public_seed}_state_changed")
                break
    return errors


def _gpu_ownership_errors(live_arms: Iterable[LiveArm]) -> list[str]:
    arms = tuple(live_arms)
    errors: list[str] = []
    try:
        processes = _gpu_compute_processes()
    except RuntimeError as error:
        processes = {}
        violation = f"gpu_probe_failed={error}"
        errors.append(f"gpu_compute_process_probe_failed: {error}")
        for live in arms:
            if violation not in live.gpu_ownership_violations:
                live.gpu_ownership_violations.append(violation)
    for live in arms:
        if (
            live.disabled_reason == "agentic_llm_preflight_failed"
            and live.vla_process is None
            and live.child is None
        ):
            continue
        errors.extend(
            f"{live.spec.name}_{violation}"
            for violation in live.gpu_ownership_violations
        )
        unknown = _unknown_gpu_processes(live, processes)
        if unknown:
            violation = "unknown_gpu_compute_pids=" + ",".join(
                str(item.get("pid")) for item in unknown
            )
            if violation not in live.gpu_ownership_violations:
                live.gpu_ownership_violations.append(violation)
            errors.append(f"GPU{live.spec.gpu}_{violation}")
    return errors


def _validate_shared_invariants(
    *,
    snapshot_root: Path,
    source_binding_sha256: str,
    resource_binding: DatasetResourceBinding,
    expected_resource_binding: Mapping[str, Any],
    frozen_publication_root: Path,
    frozen_provenance_sha256: str,
    expected_publication_binding: Mapping[str, Any],
    instance_state_bindings: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    min_free_disk_bytes: int,
    validate_agentic_publication: bool = True,
) -> list[str]:
    errors: list[str] = []
    try:
        validate_source_snapshot(snapshot_root, source_binding_sha256)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"source_snapshot_changed: {error}")
    try:
        current_resources = verify_pinned_dataset_resources(resource_binding)
    except (OSError, RuntimeError, ValueError) as error:
        errors.append(f"behavior_resources_changed: {error}")
    else:
        if current_resources.as_dict() != dict(expected_resource_binding):
            errors.append("behavior_resources_changed")
    if validate_agentic_publication:
        try:
            current_publication = _validate_agentic_publication(
                root=frozen_publication_root,
                provenance_sha256=frozen_provenance_sha256,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"agentic_publication_changed: {error}")
        else:
            if current_publication != dict(expected_publication_binding):
                errors.append("agentic_publication_changed")
    errors.extend(_verify_public_instance_states(instance_state_bindings))
    try:
        if shutil.disk_usage(output_root).free < min_free_disk_bytes:
            errors.append("free_disk_below_required_threshold")
    except OSError as error:
        errors.append(f"free_disk_unavailable: {error}")
    return errors


def _completed_child_infrastructure_reason(
    live: LiveArm,
    child: OwnedChild,
) -> str | None:
    """Classify only completed results that are unambiguously infrastructure-unknown.

    This deliberately does not treat raw success, a valid task failure, or a
    healthy timeout as a peer-abort reason.  It is used while the other lane is
    still running, so an uncertain result must remain non-actionable.
    """

    if child.process.poll() is None:
        return None
    entry_output_dir = getattr(child, "entry_output_dir", None)
    output_root = getattr(child, "output_root", None)
    if not isinstance(entry_output_dir, Path) or not isinstance(output_root, Path):
        return "runner output binding is missing"
    trace_summary = _strict_action_trace_summary(
        entry_output_dir,
        expected_run_nonce=getattr(child, "expected_run_nonce", None),
    )
    if trace_summary["valid"] is not True:
        return str(trace_summary["error"] or "invalid official-success action trace")
    if trace_summary["official_success_binding"] is not None:
        return None
    safety_errors = tuple(getattr(child, "safety_errors", None) or ())
    if safety_errors:
        return f"child safety errors: {list(safety_errors)!r}"
    if not bool(getattr(child, "cleanup_verified", False)):
        return "cleanup_unverified"
    if bool(getattr(child, "identity_ambiguous", False)):
        return "cleanup_or_identity_unverified"
    if live.gpu_ownership_violations:
        return "gpu_ownership_violation"

    record = _read_result_record(output_root, entry_output_dir)
    infrastructure_error = (
        record.get("infrastructure_error") if isinstance(record, Mapping) else None
    )
    if infrastructure_error not in (None, "", False):
        return f"runner infrastructure error: {infrastructure_error}"
    timed_out = bool(
        child.timed_out
        or (isinstance(record, Mapping) and record.get("timed_out") is True)
    )
    if timed_out:
        return None
    if not isinstance(record, Mapping):
        return "runner result is missing"

    runner_outcome = str(
        record.get("outcome")
        or (
            "passed"
            if record.get("task_success") is True
            else "task_failed"
            if record.get("task_success") is False
            else "run_error"
        )
    )
    if runner_outcome in {"run_error", "incomplete", "not_run"}:
        return f"runner outcome is {runner_outcome}"
    sealed_agentic_task_failure = bool(
        live.spec.llm_enabled
        and child.process.returncode == 1
        and _is_sealed_agentic_task_failure(record)
    )
    if child.process.returncode not in {0, None} and not sealed_agentic_task_failure:
        return f"runner exited with returncode {child.process.returncode}"
    return None


def _abort_running_peer_after_infrastructure_exit(
    *,
    failed_live: LiveArm,
    reason: str,
    live_arms: tuple[LiveArm, LiveArm],
) -> tuple[str, ...]:
    """Stop the still-running peer at the owned cleanup boundary."""

    errors: list[str] = []
    detail = f"{failed_live.spec.name}: {reason}"
    for peer in live_arms:
        if peer is failed_live:
            continue
        peer_child = peer.child
        if peer_child is None or peer_child.process.poll() is not None:
            continue
        if getattr(peer_child, "peer_abort_reason", None):
            continue
        peer_child.peer_abort_reason = detail
        cleanup_complete = terminate_owned_child(peer_child)
        peer_child.cleanup_verified = cleanup_complete
        if not cleanup_complete:
            errors.extend(
                f"{peer.spec.name}: {message}"
                for message in (
                    peer_child.safety_errors or ["peer_abort_cleanup_unverified"]
                )
            )
    return tuple(errors)


def _trusted_monitor_activity(live_arms: Iterable[LiveArm]) -> dict[str, Any] | None:
    """Return the first filesystem-bound runtime activity eligible for t=0."""

    for live in live_arms:
        child = live.child
        if child is None:
            continue
        for kind, path in (
            ("action_trace", child.entry_output_dir / "behavior_action_trace.jsonl"),
            ("tool_trace", child.entry_output_dir / "behavior_tool_trace.jsonl"),
            ("dashboard_event", child.event_path),
        ):
            try:
                path_stat = os.stat(path, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(path_stat.st_mode) and path_stat.st_size > 0:
                return {
                    "cohort": live.spec.name,
                    "public_seed": child.public_seed,
                    "source": kind,
                    "path": str(path),
                    "size_bytes": path_stat.st_size,
                }
    return None


def _new_pair_monitor_state() -> PairMonitorState:
    """Allocate fresh monitoring state; no field may leak across public seeds."""

    return PairMonitorState()


def _wait_for_pair(
    *,
    live_arms: tuple[LiveArm, ...],
    instance_timeout_s: int,
    monitor_started: float | list[float | None],
    monitor_interval_s: int,
    monitor_window_s: int,
    sampled_offsets: set[int],
    output_root: Path,
    health_previous: dict[str, dict[str, Any]],
    monitor_start_evidence: dict[str, Any] | None = None,
    owned_runtime_root: OwnedRuntimeRoot | None = None,
    runtime_owner_document: Mapping[str, Any] | None = None,
    runtime_isolation: Mapping[str, Any] | None = None,
    checkpoint_binding: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    safety_errors: list[str] = []
    while True:
        now = time.monotonic()
        all_finished = True
        for live in live_arms:
            child = live.child
            if child is None:
                continue
            official_success_observed_monotonic = getattr(
                child,
                "official_success_observed_monotonic",
                None,
            )
            entry_output_dir = getattr(child, "entry_output_dir", None)
            if official_success_observed_monotonic is None and isinstance(
                entry_output_dir, Path
            ):
                success_summary = _strict_action_trace_summary(
                    entry_output_dir,
                    expected_run_nonce=getattr(child, "expected_run_nonce", None),
                )
                if success_summary.get("official_success_binding") is not None:
                    official_success_observed_monotonic = now
                    child.official_success_observed_monotonic = now
            success_cleanup_deadline = (
                None
                if official_success_observed_monotonic is None
                else official_success_observed_monotonic
                + POST_SUCCESS_CLEANUP_DEADLINE_S
            )
            if child.process.poll() is not None:
                if not child.cleanup_verified:
                    cleanup_complete = (
                        terminate_owned_child(child)
                        if success_cleanup_deadline is None
                        else terminate_owned_child(
                            child,
                            deadline_monotonic=success_cleanup_deadline,
                        )
                    )
                    child.cleanup_verified = cleanup_complete
                    if not cleanup_complete:
                        detail = "; ".join(
                            child.safety_errors or ["cleanup_unverified"]
                        )
                        _disable_lane_after_error(
                            live,
                            child,
                            f"{live.spec.name}_cleanup_unverified: {detail}",
                        )
                if (
                    child.cleanup_verified
                    and not child.identity_ambiguous
                    and live.vla_process is not None
                    and not bool(getattr(child, "post_run_vla_quiesced", False))
                ):
                    try:
                        if checkpoint_binding is None:
                            raise RuntimeError(
                                "post-run VLA quiescence lacks checkpoint binding"
                            )
                        _quiesce_persistent_vla_after_runner(
                            live,
                            child,
                            checkpoint_binding=checkpoint_binding,
                        )
                    except Exception as error:
                        message = (
                            f"{live.spec.name}_post_run_vla_quiescence: "
                            f"{type(error).__name__}: {error}"
                        )
                        _disable_lane_after_error(live, child, message)
                continue
            if (
                success_cleanup_deadline is not None
                and now >= success_cleanup_deadline
                and not child.forced_cleanup_started
            ):
                child.forced_cleanup_started = True
                cleanup_complete = terminate_owned_child(
                    child,
                    deadline_monotonic=success_cleanup_deadline,
                )
                child.cleanup_verified = cleanup_complete
                if not cleanup_complete:
                    _disable_lane_after_error(
                        live,
                        child,
                        f"{live.spec.name}_post_success_cleanup_unverified",
                    )
                continue
            if now >= child.hard_deadline_monotonic:
                child.timed_out = True
                child.hard_deadline_exhausted = True
                cleanup_complete = terminate_owned_child(child)
                child.cleanup_verified = cleanup_complete
                if not cleanup_complete:
                    _disable_lane_after_error(
                        live,
                        child,
                        f"{live.spec.name}_hard_deadline_survivor_or_identity_ambiguity",
                    )
                continue
            all_finished = False
            if (
                now >= child.action_deadline_monotonic
                and not child.action_cleanup_started
            ):
                child.action_cleanup_started = True
                signal_status, error = _signal_instance_child(
                    child,
                    sig=signal.SIGTERM,
                )
                if signal_status == "error":
                    _disable_lane_after_error(
                        live,
                        child,
                        f"{live.spec.name}_action_deadline_cleanup: {error}",
                    )
                elif signal_status == "sent":
                    child.timed_out = True
                    child.action_deadline_exhausted = True
            if (
                now >= child.cleanup_deadline_monotonic
                and not child.forced_cleanup_started
            ):
                child.forced_cleanup_started = True
                child.timed_out = True
                if not terminate_owned_child(child):
                    _disable_lane_after_error(
                        live,
                        child,
                        f"{live.spec.name}_cleanup_not_complete_before_hard_deadline",
                    )
            now = time.monotonic()
            if now >= child.hard_deadline_monotonic:
                child.timed_out = True
                child.hard_deadline_exhausted = True
                cleanup_complete = terminate_owned_child(child)
                child.cleanup_verified = cleanup_complete
                if not cleanup_complete:
                    _disable_lane_after_error(
                        live,
                        child,
                        f"{live.spec.name}_hard_deadline_survivor_or_identity_ambiguity",
                    )
        monitor_epoch = (
            monitor_started[0] if isinstance(monitor_started, list) else monitor_started
        )
        if monitor_epoch is None:
            evidence = _trusted_monitor_activity(live_arms)
            if evidence is not None and isinstance(monitor_started, list):
                monitor_epoch = now
                monitor_started[0] = now
                if monitor_start_evidence is not None:
                    monitor_start_evidence.update(
                        {
                            **evidence,
                            "started_at": _utc_now(),
                            "started_monotonic": now,
                        }
                    )
        if monitor_epoch is not None:
            elapsed = max(0.0, now - monitor_epoch)
            due_offsets = [0]
            due_offsets.extend(
                range(monitor_interval_s, monitor_window_s + 1, monitor_interval_s)
            )
            for offset in due_offsets:
                if offset not in sampled_offsets and elapsed >= offset:
                    runtime_errors = _sample_health(
                        live_arms=live_arms,
                        offset_s=offset,
                        output_root=output_root,
                        previous=health_previous,
                        owned_runtime_root=owned_runtime_root,
                        runtime_owner_document=runtime_owner_document,
                        runtime_isolation=runtime_isolation,
                    )
                    safety_errors.extend(
                        f"external_runtime_health: {error}"
                        for error in (runtime_errors or ())
                    )
                    sampled_offsets.add(offset)
        if all_finished:
            return tuple(safety_errors)
        remaining_hard = [
            live.child.hard_deadline_monotonic - time.monotonic()
            for live in live_arms
            if live.child is not None and live.child.process.poll() is None
        ]
        sleep_s = min(0.25, max(0.0, min(remaining_hard, default=0.25)))
        if sleep_s > 0:
            time.sleep(sleep_s)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired picking_up_trash s10-s19 Eval: GPU7 gpt-5.5/xhigh "
            "agentic versus GPU6 pi0_nav_pick-only."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--runtime-base",
        required=True,
        help=(
            "Existing external parent filesystem for one campaign-owned transient "
            "runtime child; formal artifacts remain under --output-root."
        ),
    )
    parser.add_argument("--source-snapshot-root", required=True)
    parser.add_argument("--source-snapshot-binding-sha256", required=True)
    parser.add_argument("--behavior-frozen-publication-root", required=True)
    parser.add_argument("--behavior-frozen-provenance-sha256", required=True)
    parser.add_argument("--behavior-resource-local", required=True)
    parser.add_argument(
        "--task-name",
        default=PICKING_UP_TRASH_TASK_SPEC.task_name,
    )
    parser.add_argument("--public-seed", action="append", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo", required=True)
    parser.add_argument("--behavior-python", required=True)
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="xhigh",
    )
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument(
        "--agentic-dashboard-port",
        type=int,
        default=DEFAULT_AGENTIC_DASHBOARD_PORT,
    )
    parser.add_argument(
        "--baseline-dashboard-port",
        type=int,
        default=DEFAULT_BASELINE_DASHBOARD_PORT,
    )
    parser.add_argument(
        "--action-deadline-s",
        type=int,
        default=DEFAULT_ACTION_DEADLINE_S,
        help="Stop admitting robot actions after this many seconds (default: 6900).",
    )
    parser.add_argument(
        "--cleanup-deadline-s",
        type=int,
        default=DEFAULT_CLEANUP_DEADLINE_S,
        help="Begin forced cleanup after this many seconds (default: 7080).",
    )
    parser.add_argument(
        "--instance-timeout-s",
        type=int,
        default=DEFAULT_INSTANCE_TIMEOUT_S,
        help="Hard per-instance deadline in seconds (default: 7200).",
    )
    parser.add_argument(
        "--monitor-interval-s",
        type=int,
        default=DEFAULT_MONITOR_INTERVAL_S,
    )
    parser.add_argument(
        "--monitor-window-s",
        type=int,
        default=DEFAULT_MONITOR_WINDOW_S,
    )
    parser.add_argument(
        "--min-free-disk-bytes",
        type=int,
        default=DEFAULT_MIN_FREE_DISK_BYTES,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.task_name != PICKING_UP_TRASH_TASK_SPEC.task_name:
        raise SystemExit("paired Eval currently supports only picking_up_trash")
    try:
        validate_dashboard_endpoints(
            host=args.dashboard_host,
            agentic_port=args.agentic_dashboard_port,
            baseline_port=args.baseline_dashboard_port,
        )
        validate_deadlines(
            action_deadline_s=args.action_deadline_s,
            cleanup_deadline_s=args.cleanup_deadline_s,
            instance_timeout_s=args.instance_timeout_s,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if (
        args.monitor_interval_s <= 0
        or args.monitor_window_s < args.monitor_interval_s
        or args.monitor_window_s % args.monitor_interval_s
    ):
        raise SystemExit(
            "monitor window must be a positive multiple of monitor interval"
        )
    if args.min_free_disk_bytes < 0:
        raise SystemExit("--min-free-disk-bytes must be non-negative")
    if args.model != "gpt-5.5" or args.reasoning_effort != "xhigh":
        raise SystemExit("agentic paired Eval requires gpt-5.5/xhigh")

    snapshot_root = Path(args.source_snapshot_root).expanduser().resolve()
    source_binding_sha256 = str(args.source_snapshot_binding_sha256).lower()
    if _SHA256_RE.fullmatch(source_binding_sha256) is None:
        raise SystemExit("--source-snapshot-binding-sha256 must be SHA-256")
    try:
        source_binding = validate_source_snapshot(
            snapshot_root,
            source_binding_sha256,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"invalid source snapshot: {error}") from error
    module_root = Path(__file__).resolve().parents[2]
    if module_root != snapshot_root:
        raise SystemExit(
            "paired Eval supervisor must itself run from --source-snapshot-root"
        )

    spec = get_task_spec(args.task_name)
    public_seeds = (
        spec.eval_public_seeds if args.public_seed is None else tuple(args.public_seed)
    )
    try:
        if not public_seeds or len(public_seeds) != len(set(public_seeds)):
            raise ValueError("paired Eval seeds must be non-empty and unique")
        for public_seed in public_seeds:
            spec.instance_for_public_seed(public_seed, phase="eval")
    except ValueError as error:
        raise SystemExit(str(error)) from error

    output_root = Path(args.output_root).expanduser().absolute()
    if output_root.exists():
        raise SystemExit("--output-root must not already exist")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    python = _lexical_executable_path(args.python, label="--python")
    behavior_repo = Path(args.behavior_repo).expanduser().resolve(strict=True)
    behavior_python = _lexical_executable_path(
        args.behavior_python,
        label="--behavior-python",
    )
    checkpoint = Path(args.policy_checkpoint).expanduser().resolve(strict=True)
    frozen_publication_root = (
        Path(args.behavior_frozen_publication_root).expanduser().absolute()
    )
    behavior_resource_local = (
        Path(args.behavior_resource_local).expanduser().resolve(strict=True)
    )
    try:
        runtime_base = _validate_external_runtime_base(
            args.runtime_base,
            output_root=output_root,
            protected_paths=tuple(
                path
                for path in (
                    snapshot_root,
                    frozen_publication_root,
                    behavior_resource_local,
                    checkpoint,
                    behavior_repo,
                )
                if path.exists()
            ),
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"invalid external runtime base: {error}") from error
    agentic_input_error: str | None = None
    try:
        agentic_publication_binding = _validate_agentic_publication(
            root=frozen_publication_root,
            provenance_sha256=args.behavior_frozen_provenance_sha256,
        )
    except (OSError, RuntimeError, ValueError) as error:
        agentic_input_error = (
            f"agentic_frozen_publication_invalid: {type(error).__name__}: {error}"
        )
        agentic_publication_binding = {"validation_error": agentic_input_error}
    try:
        instance_state_bindings = _freeze_public_instance_states(
            behavior_repo=behavior_repo,
            public_seeds=spec.eval_public_seeds,
        )
        state_errors = _verify_public_instance_states(instance_state_bindings)
        if state_errors:
            raise RuntimeError("; ".join(state_errors))
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"invalid shared Eval instance states: {error}") from error
    if shutil.disk_usage(output_root.parent).free < args.min_free_disk_bytes:
        raise SystemExit("insufficient free disk before paired Eval")

    # Each persistent VLA validates the same checkpoint once at startup. The
    # controller uses the expected binding without adding a third 7.2GB hash.
    checkpoint_binding = _expected_shared_policy_checkpoint_binding()
    if str(checkpoint) != checkpoint_binding["resolved_path"]:
        raise SystemExit("paired Eval requires the shared Pi0.5 checkpoint")
    output_root.mkdir(mode=0o700)
    agentic_preflight_path = output_root / "agentic_llm_preflight.json"
    try:
        agentic_llm_preflight = run_llm_proxy_preflight(
            python=python,
            repo_root=snapshot_root,
            model=args.model,
            timeout_s=60,
            environment=os.environ.copy(),
        )
    except Exception as error:
        agentic_llm_preflight = {
            "schema_version": 1,
            "kind": "agentic_llm_proxy_preflight",
            "status": "failed",
            "valid_response": False,
            "failure_reason": (f"preflight_exception: {type(error).__name__}: {error}"),
        }
    _atomic_json(agentic_preflight_path, agentic_llm_preflight)
    agentic_preflight_passed = agentic_llm_preflight.get("status") in {
        "passed",
        "degraded",
    }
    job_lane_disabled: dict[str, str] = {}
    if not agentic_preflight_passed:
        job_lane_disabled["agentic"] = "agentic_llm_preflight_failed"
    elif agentic_input_error is not None:
        job_lane_disabled["agentic"] = agentic_input_error
    lane_ports = {
        "agentic": args.agentic_dashboard_port,
        "pi0_nav_pick_only": args.baseline_dashboard_port,
    }
    for lane_name, dashboard_port in lane_ports.items():
        if lane_name in job_lane_disabled:
            continue
        if not _port_is_free(args.dashboard_host, dashboard_port):
            job_lane_disabled[lane_name] = (
                f"dashboard_port_unavailable:{dashboard_port}"
            )
    try:
        initial_gpu_processes = _gpu_compute_processes()
    except RuntimeError as error:
        raise SystemExit(f"GPU ownership preflight failed closed: {error}") from error
    lane_gpus = {
        "agentic": DEFAULT_AGENTIC_GPU,
        "pi0_nav_pick_only": DEFAULT_BASELINE_GPU,
    }
    for lane_name, gpu in lane_gpus.items():
        if lane_name in job_lane_disabled:
            continue
        if gpu not in initial_gpu_processes:
            job_lane_disabled[lane_name] = f"GPU{gpu}_not_enumerated"
        elif initial_gpu_processes.get(gpu):
            job_lane_disabled[lane_name] = f"GPU{gpu}_already_has_compute_processes"
    agentic_lane_admitted = "agentic" not in job_lane_disabled
    baseline_lane_admitted = "pi0_nav_pick_only" not in job_lane_disabled
    shared_lock_path = output_root.parent / f".{output_root.name}.paired-eval.lock"
    lane_lock_paths = {
        "agentic": (
            _gpu_lock_path(DEFAULT_AGENTIC_GPU),
            Path("/tmp") / f"rpent-dashboard-{args.agentic_dashboard_port}.lock",
        ),
        "pi0_nav_pick_only": (
            _gpu_lock_path(DEFAULT_BASELINE_GPU),
            Path("/tmp") / f"rpent-dashboard-{args.baseline_dashboard_port}.lock",
        ),
    }
    locks: list[BinaryIO] = []
    live_arms: list[LiveArm] = []
    manifest_path = output_root / "paired_eval_manifest.json"
    results_path = output_root / "paired_eval_results.jsonl"
    interrupted = False
    blocked_reason: str | None = None
    health_samples_by_public_seed: dict[str, list[int]] = {}
    monitor_evidence_by_public_seed: dict[str, dict[str, Any] | None] = {}
    exit_code: int | None = None
    intended_final_status: str | None = None
    owned_runtime_root: OwnedRuntimeRoot | None = None
    runtime_owner_document: dict[str, Any] | None = None
    protected_dashboard_observation = {
        "host": args.dashboard_host,
        "port": PROTECTED_EXISTING_DASHBOARD_PORT,
        "accepting_connections_before_campaign": _port_accepts_connection(
            args.dashboard_host,
            PROTECTED_EXISTING_DASHBOARD_PORT,
        ),
        "observed_at": _utc_now(),
        "policy": "observe_only_never_bind_restart_or_signal",
    }

    agentic_spec = ArmSpec(
        name="agentic",
        gpu=DEFAULT_AGENTIC_GPU,
        dashboard_port=args.agentic_dashboard_port,
        output_root=output_root / "agentic",
        runner_script=snapshot_root / "scripts" / "run_behavior_serial_eval.py",
        run_id="behavior/paired-eval-agentic",
        llm_enabled=True,
        controller="gpt-5.5/xhigh hybrid",
        allowed_tools=PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION],
    )
    baseline_spec = ArmSpec(
        name="pi0_nav_pick_only",
        gpu=DEFAULT_BASELINE_GPU,
        dashboard_port=args.baseline_dashboard_port,
        output_root=output_root / "pi0_nav_pick_only",
        runner_script=snapshot_root / "scripts" / "run_behavior_serial_vla_eval.py",
        run_id="behavior/paired-eval-pi0-only",
        llm_enabled=False,
        controller="pi0_nav_pick-only",
        allowed_tools=("pi0_nav_pick",),
    )
    runtime_isolation: dict[str, Any] = {}

    try:
        stream = shared_lock_path.open("w+b")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.close()
            raise SystemExit(
                f"paired Eval lock is already held: {shared_lock_path}"
            ) from error
        locks.append(stream)
        for lane_name, lane_paths in lane_lock_paths.items():
            if lane_name in job_lane_disabled:
                continue
            lane_lock_error: str | None = None
            lane_locks: list[BinaryIO] = []
            for lock_path in lane_paths:
                lane_stream: BinaryIO | None = None
                try:
                    lane_stream = lock_path.open("w+b")
                    fcntl.flock(lane_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (BlockingIOError, OSError) as error:
                    if lane_stream is not None:
                        lane_stream.close()
                    lane_lock_error = (
                        f"{lane_name}_lock_unavailable:{lock_path}: "
                        f"{type(error).__name__}: {error}"
                    )
                    break
                lane_locks.append(lane_stream)
            if lane_lock_error is not None:
                for lane_stream in lane_locks:
                    lane_stream.close()
                job_lane_disabled[lane_name] = lane_lock_error
            else:
                locks.extend(lane_locks)
        try:
            locked_gpu_processes = _gpu_compute_processes()
        except RuntimeError as error:
            raise SystemExit(
                f"GPU ownership probe failed after locks: {error}"
            ) from error
        for lane_name, gpu in lane_gpus.items():
            if lane_name in job_lane_disabled:
                continue
            if gpu not in locked_gpu_processes:
                job_lane_disabled[lane_name] = f"GPU{gpu}_not_enumerated_after_lock"
            elif locked_gpu_processes.get(gpu):
                job_lane_disabled[lane_name] = (
                    f"GPU{gpu}_gained_compute_processes_before_startup"
                )
        agentic_lane_admitted = "agentic" not in job_lane_disabled
        baseline_lane_admitted = "pi0_nav_pick_only" not in job_lane_disabled

        try:
            owned_runtime_root, runtime_owner_document = _create_owned_runtime_root(
                runtime_base,
                output_root=output_root,
                source_snapshot_root=snapshot_root,
                source_snapshot_binding_sha256=source_binding_sha256,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(
                f"failed to allocate external runtime root: {error}"
            ) from error

        try:
            resource_binding = prepare_local_dataset_resources(
                "behavior",
                source_root=behavior_resource_local,
                cache_root=output_root / "frozen_resource_cache",
            )
            resource_source_binding = resource_binding.as_dict()
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"invalid frozen BEHAVIOR resources: {error}") from error

        startup_input_errors = _validate_shared_invariants(
            snapshot_root=snapshot_root,
            source_binding_sha256=source_binding_sha256,
            resource_binding=resource_binding,
            expected_resource_binding=resource_source_binding,
            frozen_publication_root=frozen_publication_root,
            frozen_provenance_sha256=args.behavior_frozen_provenance_sha256,
            expected_publication_binding=agentic_publication_binding,
            instance_state_bindings=instance_state_bindings,
            output_root=output_root,
            min_free_disk_bytes=args.min_free_disk_bytes,
            validate_agentic_publication=False,
        )
        if startup_input_errors:
            raise SystemExit(
                "paired Eval frozen inputs changed before VLA startup: "
                + "; ".join(startup_input_errors)
            )

        arms_to_start = tuple(
            arm
            for arm in (agentic_spec, baseline_spec)
            if arm.name not in job_lane_disabled
        )
        for arm in arms_to_start:
            server: DashboardServer | None = None
            try:
                arm.output_root.mkdir()
                isolation = prepare_campaign_runtime_isolation(
                    owned_runtime_root.root / arm.name,
                    namespace=f"paired-eval-{arm.name}",
                    cuda_device=arm.gpu,
                    behavior_python=behavior_python,
                )
                runtime_isolation[arm.name] = isolation
                runtime_owner_document = _write_external_runtime_owner(
                    owned_runtime_root,
                    runtime_isolation,
                )
                server = DashboardServer(
                    host=args.dashboard_host,
                    port=arm.dashboard_port,
                    runs_dir=str(arm.output_root),
                    language="en",
                )
                server.arm_auto_start(
                    {
                        "environment": "behavior",
                        "behavior-phase": "eval",
                        "task-name": spec.task_name,
                        "eval-cohort": arm.name,
                        "cuda-device": arm.gpu,
                    }
                )
                url = server.start()
            except Exception as error:
                if server is not None:
                    try:
                        server.stop()
                    except RuntimeError:
                        pass
                job_lane_disabled[arm.name] = (
                    f"{arm.name}_startup_failed: {type(error).__name__}: {error}"
                )
                continue
            assert server is not None
            live = LiveArm(spec=arm, server=server, url=url)
            live.disabled_reason = job_lane_disabled.get(arm.name)
            live_arms.append(live)

        runtime_startup_errors = _external_runtime_active_errors(
            owned_runtime_root,
            owner_document=runtime_owner_document,
            runtime_isolation=runtime_isolation,
        )
        shared_runtime_errors, lane_runtime_errors = _partition_runtime_errors(
            runtime_startup_errors,
            lane_names=(arm.name for arm in arms_to_start),
        )
        for lane_name, errors in lane_runtime_errors.items():
            if errors:
                job_lane_disabled[lane_name] = "; ".join(errors)
        if shared_runtime_errors:
            raise RuntimeError(
                "external runtime changed before VLA startup: "
                + "; ".join(shared_runtime_errors)
            )
        agentic_lane_admitted = "agentic" not in job_lane_disabled
        baseline_lane_admitted = "pi0_nav_pick_only" not in job_lane_disabled

        # Use the same checked binding for both VLA readiness gates. Each VLA
        # remains controller-owned and is never stopped by either runner.
        for live in live_arms:
            if live.disabled_reason is not None:
                continue
            startup_gpu_errors = _gpu_ownership_errors((live,))
            if startup_gpu_errors:
                live.disabled_reason = (
                    "gpu_ownership_changed_before_vla_startup: "
                    + "; ".join(startup_gpu_errors)
                )
                continue
            vla_root = _formal_vla_log_dir(
                runtime_isolation[live.spec.name],
                formal_output_root=live.spec.output_root,
            )
            runtime_environment = _snapshot_child_environment(
                snapshot_root=snapshot_root,
                arm=live.spec,
                runtime_environment=runtime_isolation[live.spec.name].environment(),
            )
            try:
                with _temporary_environment(runtime_environment):
                    endpoint, process = start_vla_server(
                        argparse.Namespace(
                            behavior_python=str(behavior_python),
                            behavior_repo=str(behavior_repo),
                            policy_checkpoint=str(checkpoint),
                            seed=BEHAVIOR_NATIVE_ENV_SEED,
                            vla_port=0,
                            vla_ready_timeout_s=1800,
                            cuda_device=live.spec.gpu,
                            _behavior_policy_checkpoint_binding=checkpoint_binding,
                        ),
                        output_dir=vla_root,
                    )
            except Exception as error:
                live.disabled_reason = (
                    f"persistent_vla_start_failed: {type(error).__name__}: {error}"
                )
                continue
            live.vla_endpoint = endpoint
            live.vla_process = process
            try:
                live.vla_pgid = os.getpgid(process.pid)
                live.vla_sid = os.getsid(process.pid)
            except OSError as error:
                _terminate_verified_vla(live)
                live.disabled_reason = (
                    f"{live.spec.name} VLA exited before identity capture"
                    f": {type(error).__name__}: {error}"
                )
                continue
            live.vla_start_ticks = _proc_start_ticks(process.pid)
            if not _vla_identity_matches(live):
                _terminate_verified_vla(live)
                live.disabled_reason = (
                    f"{live.spec.name} VLA has no dedicated verified session"
                )
                continue
            try:
                _prepare_persistent_vla_for_runner(
                    live,
                    checkpoint_binding=checkpoint_binding,
                )
            except Exception as error:
                _terminate_verified_vla(live)
                live.disabled_reason = (
                    "persistent_vla_disable_gate_failed: "
                    f"{type(error).__name__}: {error}"
                )

        manifest: dict[str, Any] = {
            "schema_version": PAIRED_EVAL_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "status": "running",
            "task": {
                "task_name": spec.task_name,
                "task_index": spec.task_index,
                "task_language": spec.task_language,
                "public_seeds": list(public_seeds),
                "public_seed_to_instance": {
                    str(seed): spec.instance_for_public_seed(seed, phase="eval")
                    for seed in public_seeds
                },
            },
            "source_snapshot": source_binding.as_dict(),
            "resource_source": resource_source_binding,
            "agentic_frozen_publication": agentic_publication_binding,
            "instance_states": instance_state_bindings,
            "policy_checkpoint": checkpoint_binding,
            "external_runtime": {
                "owner_binding": runtime_owner_document,
                "cleanup": {
                    "status": "active",
                    "runtime_base": str(owned_runtime_root.base),
                    "runtime_root": str(owned_runtime_root.root),
                },
            },
            "protocol": {
                "pair_barrier": False,
                "arms_execute_independently": True,
                "max_parallel": 2,
                "max_parallel_per_arm": 1,
                "max_attempts_per_instance": 1,
                "automatic_retry": False,
                "action_deadline_s": args.action_deadline_s,
                "cleanup_deadline_s": args.cleanup_deadline_s,
                "instance_timeout_s": args.instance_timeout_s,
                "monitor_offsets_s": [
                    0,
                    *range(
                        args.monitor_interval_s,
                        args.monitor_window_s + 1,
                        args.monitor_interval_s,
                    ),
                ],
                "monitor_epoch_policy": (
                    "first_bound_action_trace_tool_trace_or_dashboard_event"
                ),
                "protected_dashboard_port": PROTECTED_EXISTING_DASHBOARD_PORT,
                "protected_dashboard_observation": protected_dashboard_observation,
                "timeout_score": "failure_with_timed_out_true",
                "infrastructure_error_score": "unknown",
                "pure_vla_chunks_per_call": PURE_VLA_CHUNKS_PER_CALL,
                "official_success_action_policy": "stop_after_success_env_step",
            },
            "agentic_llm_preflight": agentic_llm_preflight,
            "lane_blockers": dict(job_lane_disabled),
            "topology": (
                "dual_independent"
                if agentic_lane_admitted and baseline_lane_admitted
                else "agentic_only"
                if agentic_lane_admitted
                else "pure_only"
                if baseline_lane_admitted
                else "no_lane_admitted"
            ),
            "agentic_admission": (
                "admitted" if agentic_lane_admitted else "blocked_lane_gate"
            ),
            "pure_admission": (
                "admitted" if baseline_lane_admitted else "blocked_lane_gate"
            ),
            "arms": {
                live.spec.name: {
                    "gpu": live.spec.gpu,
                    "dashboard_url": live.url,
                    "dashboard_port": live.spec.dashboard_port,
                    "run_id": live.spec.run_id,
                    "controller": live.spec.controller,
                    "llm_enabled": live.spec.llm_enabled,
                    "allowed_tools": list(live.spec.allowed_tools),
                    **(
                        {"chunks_per_call": PURE_VLA_CHUNKS_PER_CALL}
                        if live.spec.name == "pi0_nav_pick_only"
                        else {}
                    ),
                    "output_root": str(live.spec.output_root),
                    "vla_endpoint": live.vla_endpoint,
                    "vla_pid": (
                        live.vla_process.pid if live.vla_process is not None else None
                    ),
                    "vla_pgid": live.vla_pgid,
                    "vla_sid": live.vla_sid,
                    "vla_start_ticks": live.vla_start_ticks,
                    "vla_disabled_health": live.vla_disabled_health,
                    "runtime_isolation": runtime_isolation[live.spec.name].as_dict(),
                    "resource_binding": {
                        "manifest_sha256": resource_binding.manifest_sha256,
                        "root": str(resource_binding.root),
                        "control_input": live.spec.llm_enabled,
                        "purpose": (
                            "frozen_eval_input"
                            if live.spec.llm_enabled
                            else "audit_only_excluded_from_baseline_control"
                        ),
                    },
                }
                for live in live_arms
            },
            "pairs": [],
        }
        if not agentic_lane_admitted:
            manifest["arms"]["agentic"] = {
                "gpu": agentic_spec.gpu,
                "dashboard_url": None,
                "dashboard_port": agentic_spec.dashboard_port,
                "run_id": agentic_spec.run_id,
                "controller": agentic_spec.controller,
                "llm_enabled": True,
                "allowed_tools": list(agentic_spec.allowed_tools),
                "output_root": str(agentic_spec.output_root),
                "vla_endpoint": None,
                "admission": "blocked_lane_gate",
                "reason": job_lane_disabled["agentic"],
                "attempt_started": False,
            }
        if not baseline_lane_admitted:
            manifest["arms"]["pi0_nav_pick_only"] = {
                "gpu": baseline_spec.gpu,
                "dashboard_url": None,
                "dashboard_port": baseline_spec.dashboard_port,
                "run_id": baseline_spec.run_id,
                "controller": baseline_spec.controller,
                "llm_enabled": False,
                "allowed_tools": list(baseline_spec.allowed_tools),
                "chunks_per_call": PURE_VLA_CHUNKS_PER_CALL,
                "output_root": str(baseline_spec.output_root),
                "vla_endpoint": None,
                "admission": "blocked_lane_gate",
                "reason": job_lane_disabled["pi0_nav_pick_only"],
                "attempt_started": False,
            }
        _atomic_json(manifest_path, manifest)
        for pair_index, public_seed in enumerate(public_seeds):
            pair_monitor = _new_pair_monitor_state()
            shared_errors = _validate_shared_invariants(
                snapshot_root=snapshot_root,
                source_binding_sha256=source_binding_sha256,
                resource_binding=resource_binding,
                expected_resource_binding=resource_source_binding,
                frozen_publication_root=frozen_publication_root,
                frozen_provenance_sha256=args.behavior_frozen_provenance_sha256,
                expected_publication_binding=agentic_publication_binding,
                instance_state_bindings=instance_state_bindings,
                output_root=output_root,
                min_free_disk_bytes=args.min_free_disk_bytes,
                validate_agentic_publication=False,
            )
            runtime_errors = _external_runtime_active_errors(
                owned_runtime_root,
                owner_document=runtime_owner_document,
                runtime_isolation=runtime_isolation,
            )
            shared_runtime_errors, lane_runtime_errors = _partition_runtime_errors(
                runtime_errors,
                lane_names=(live.spec.name for live in live_arms),
            )
            shared_errors.extend(shared_runtime_errors)
            for live in live_arms:
                if errors := lane_runtime_errors.get(live.spec.name):
                    reason = "; ".join(errors)
                    live.disabled_reason = reason
                    job_lane_disabled[live.spec.name] = reason
            for live in live_arms:
                try:
                    current_isolation = validate_campaign_runtime_isolation(
                        runtime_isolation[live.spec.name].root,
                        runtime_isolation[live.spec.name].binding_sha256,
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    reason = f"{live.spec.name}_runtime_isolation_changed: {error}"
                    live.disabled_reason = reason
                    job_lane_disabled[live.spec.name] = reason
                else:
                    if (
                        current_isolation.as_dict()
                        != runtime_isolation[live.spec.name].as_dict()
                    ):
                        reason = f"{live.spec.name}_runtime_isolation_changed"
                        live.disabled_reason = reason
                        job_lane_disabled[live.spec.name] = reason
            if shared_errors:
                blocked_reason = "; ".join(shared_errors)
                break
            pair_started = _utc_now()
            pair_deadlines = _pair_deadline_binding(
                action_deadline_s=args.action_deadline_s,
                cleanup_deadline_s=args.cleanup_deadline_s,
                instance_timeout_s=args.instance_timeout_s,
            )
            agentic_expected_run_nonce = secrets.token_hex(16)
            pure_expected_run_nonce = secrets.token_hex(16)
            instance_id = spec.instance_for_public_seed(public_seed, phase="eval")
            for live in live_arms:
                job_disabled_reason = job_lane_disabled.get(live.spec.name)
                if job_disabled_reason is not None:
                    live.disabled_reason = job_disabled_reason
                elif live.vla_process is not None:
                    live.disabled_reason = None
                runner_root = (
                    live.spec.output_root / f"pair_{pair_index:02d}_s{public_seed}"
                )
                entry_output_dir = runner_root / spec.tag(public_seed)
                event_path = entry_output_dir / "dashboard_events.jsonl"
                state = _new_dashboard_state(
                    arm=live.spec,
                    public_seed=public_seed,
                    entry_output_dir=entry_output_dir,
                    action_deadline_s=args.action_deadline_s,
                )
                live.server.register(state)
                live.state = state
                if live.disabled_reason is not None:
                    continue
                if live.spec.llm_enabled:
                    try:
                        current_agentic_publication = _validate_agentic_publication(
                            root=frozen_publication_root,
                            provenance_sha256=(args.behavior_frozen_provenance_sha256),
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        live.disabled_reason = (
                            "agentic_publication_changed: "
                            f"{type(error).__name__}: {error}"
                        )
                        continue
                    if current_agentic_publication != dict(agentic_publication_binding):
                        live.disabled_reason = "agentic_publication_changed"
                        continue
                if (
                    live.vla_process is None
                    or live.vla_process.poll() is not None
                    or live.vla_endpoint is None
                ):
                    live.disabled_reason = "persistent_vla_unavailable"
                    continue
                lane_start_errors = _gpu_ownership_errors((live,))
                if lane_start_errors:
                    live.disabled_reason = (
                        "gpu_ownership_changed_before_runner_spawn: "
                        + "; ".join(lane_start_errors)
                    )
                    continue
                try:
                    _prepare_persistent_vla_for_runner(
                        live,
                        checkpoint_binding=checkpoint_binding,
                    )
                except Exception as error:
                    live.disabled_reason = (
                        "persistent_vla_disable_gate_failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
                if live.spec.llm_enabled:
                    runner_argv = build_agentic_runner_argv(
                        python=python,
                        snapshot_root=snapshot_root,
                        output_root=runner_root,
                        public_seed=public_seed,
                        vla_endpoint=live.vla_endpoint,
                        behavior_repo=behavior_repo,
                        behavior_python=behavior_python,
                        checkpoint=checkpoint,
                        frozen_publication_root=frozen_publication_root,
                        frozen_provenance_sha256=(
                            args.behavior_frozen_provenance_sha256
                        ),
                        behavior_resource_local=resource_binding.root,
                        behavior_resource_cache=(
                            agentic_spec.output_root / "resource_cache"
                        ),
                        source_binding_sha256=source_binding_sha256,
                        dashboard_host=args.dashboard_host,
                        dashboard_port=live.spec.dashboard_port,
                        action_deadline_s=args.action_deadline_s,
                        cleanup_deadline_s=args.cleanup_deadline_s,
                        instance_timeout_s=args.instance_timeout_s,
                        instance_started_monotonic_ns=(
                            pair_deadlines.started_monotonic_ns
                        ),
                        action_deadline_monotonic_ns=(
                            pair_deadlines.action_deadline_monotonic_ns
                        ),
                        cleanup_deadline_monotonic_ns=(
                            pair_deadlines.cleanup_deadline_monotonic_ns
                        ),
                        hard_deadline_monotonic_ns=(
                            pair_deadlines.hard_deadline_monotonic_ns
                        ),
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        expected_run_nonce=agentic_expected_run_nonce,
                        runtime_isolation_root=runtime_isolation[live.spec.name].root,
                        runtime_isolation_binding_sha256=runtime_isolation[
                            live.spec.name
                        ].binding_sha256,
                    )
                else:
                    runner_argv = build_baseline_runner_argv(
                        python=python,
                        snapshot_root=snapshot_root,
                        output_root=runner_root,
                        public_seed=public_seed,
                        vla_endpoint=live.vla_endpoint,
                        behavior_repo=behavior_repo,
                        behavior_python=behavior_python,
                        checkpoint=checkpoint,
                        source_binding_sha256=source_binding_sha256,
                        action_deadline_s=args.action_deadline_s,
                        cleanup_deadline_s=args.cleanup_deadline_s,
                        instance_timeout_s=args.instance_timeout_s,
                        instance_started_monotonic_ns=(
                            pair_deadlines.started_monotonic_ns
                        ),
                        action_deadline_monotonic_ns=(
                            pair_deadlines.action_deadline_monotonic_ns
                        ),
                        cleanup_deadline_monotonic_ns=(
                            pair_deadlines.cleanup_deadline_monotonic_ns
                        ),
                        hard_deadline_monotonic_ns=(
                            pair_deadlines.hard_deadline_monotonic_ns
                        ),
                        runtime_isolation_root=runtime_isolation[live.spec.name].root,
                        runtime_isolation_binding_sha256=runtime_isolation[
                            live.spec.name
                        ].binding_sha256,
                        expected_run_nonce=pure_expected_run_nonce,
                    )
                environment = _snapshot_child_environment(
                    snapshot_root=snapshot_root,
                    arm=live.spec,
                    runtime_environment=runtime_isolation[live.spec.name].environment(),
                )
                if live.spec.llm_enabled and network_environment_binding(
                    environment
                ) != agentic_llm_preflight.get("network_environment"):
                    live.disabled_reason = "agentic_llm_preflight_environment_changed"
                    state.set_metadata(
                        {
                            "health-status": "warning",
                            "health-checked-at": _utc_now(),
                        }
                    )
                    state.on_event(
                        {
                            "type": "meta",
                            "tag": "health",
                            "text": live.disabled_reason,
                        }
                    )
                    continue
                try:
                    child = _spawn_owned_child(
                        arm=live.spec,
                        public_seed=public_seed,
                        output_root=runner_root,
                        entry_output_dir=entry_output_dir,
                        event_path=event_path,
                        log_path=(
                            live.spec.output_root
                            / "launcher_logs"
                            / f"s{public_seed}.log"
                        ),
                        argv=runner_argv,
                        cwd=snapshot_root,
                        environment=environment,
                        source_snapshot_binding_sha256=source_binding_sha256,
                        action_deadline_s=args.action_deadline_s,
                        cleanup_deadline_s=args.cleanup_deadline_s,
                        instance_timeout_s=args.instance_timeout_s,
                        started_monotonic_ns=pair_deadlines.started_monotonic_ns,
                        action_deadline_monotonic_ns=(
                            pair_deadlines.action_deadline_monotonic_ns
                        ),
                        cleanup_deadline_monotonic_ns=(
                            pair_deadlines.cleanup_deadline_monotonic_ns
                        ),
                        hard_deadline_monotonic_ns=(
                            pair_deadlines.hard_deadline_monotonic_ns
                        ),
                        expected_run_nonce=(
                            agentic_expected_run_nonce
                            if live.spec.llm_enabled
                            else pure_expected_run_nonce
                        ),
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    live.disabled_reason = (
                        f"runner_launch_failed: {type(error).__name__}: {error}"
                    )
                    state.set_metadata(
                        {
                            "health-status": "warning",
                            "health-checked-at": _utc_now(),
                        }
                    )
                    state.on_event(
                        {
                            "type": "meta",
                            "tag": "health",
                            "text": live.disabled_reason,
                        }
                    )
                    continue
                relay = _dashboard_event_relay(
                    live=live,
                    event_path=event_path,
                    state=state,
                )
                live.child = child
                live.relay = relay
                relay.start()

            pair_safety_errors = _wait_for_pair(
                live_arms=tuple(live_arms),
                instance_timeout_s=args.instance_timeout_s,
                monitor_started=pair_monitor.started,
                monitor_interval_s=args.monitor_interval_s,
                monitor_window_s=args.monitor_window_s,
                sampled_offsets=pair_monitor.sampled_offsets,
                output_root=output_root,
                health_previous=pair_monitor.previous,
                monitor_start_evidence=pair_monitor.start_evidence,
                owned_runtime_root=owned_runtime_root,
                runtime_owner_document=runtime_owner_document,
                runtime_isolation=runtime_isolation,
                checkpoint_binding=checkpoint_binding,
            )
            for live in live_arms:
                if live.disabled_reason is not None:
                    job_lane_disabled[live.spec.name] = live.disabled_reason
            pair_record: dict[str, Any] = {
                "schema_version": 1,
                "pair_index": pair_index,
                "public_seed": public_seed,
                "activity_instance_id": instance_id,
                "started_at": pair_started,
                "finished_at": _utc_now(),
                "monitoring": {
                    "health_samples_completed": sorted(pair_monitor.sampled_offsets),
                    "monitor_start_evidence": (
                        dict(pair_monitor.start_evidence)
                        if pair_monitor.start_evidence
                        else None
                    ),
                },
                "arms": {},
            }
            health_samples_by_public_seed[str(public_seed)] = sorted(
                pair_monitor.sampled_offsets
            )
            monitor_evidence_by_public_seed[str(public_seed)] = (
                dict(pair_monitor.start_evidence)
                if pair_monitor.start_evidence
                else None
            )
            for live in live_arms:
                _gpu_ownership_errors((live,))
            for live in live_arms:
                child = live.child
                if child is None or child.public_seed != public_seed:
                    preflight_blocked = (
                        live.disabled_reason == "agentic_llm_preflight_failed"
                    )
                    arm_result = {
                        "outcome": ("not_admitted" if preflight_blocked else "not_run"),
                        "attempt_started": False,
                        "task_success": None,
                        "raw_official_success": None,
                        "score": None if preflight_blocked else "unknown",
                        "reason": live.disabled_reason or "lane_unavailable",
                        "infrastructure_error": (
                            {
                                "reason": live.disabled_reason,
                                "preflight_receipt": str(agentic_preflight_path),
                            }
                            if live.disabled_reason == "agentic_llm_preflight_failed"
                            else None
                        ),
                    }
                    if live.state is not None:
                        live.state.end_attempt(attempt_index=1, outcome="not_run")
                        live.state.mark_done(False)
                else:
                    arm_result = _finish_dashboard_arm(live, child)
                    state_binding_error = _enforce_instance_state_result_binding(
                        arm_result=arm_result,
                        entry_output_dir=child.entry_output_dir,
                        expected_sha256=str(
                            instance_state_bindings[str(public_seed)]["sha256"]
                        ),
                    )
                    if state_binding_error is not None:
                        lane_reason = f"{live.spec.name}_{state_binding_error}"
                        live.disabled_reason = lane_reason
                        job_lane_disabled[live.spec.name] = lane_reason
                pair_record["arms"][live.spec.name] = arm_result
                _release_child_if_cleanup_verified(live)
                live.relay = None
            if not agentic_lane_admitted:
                pair_record["arms"]["agentic"] = {
                    "outcome": "not_admitted",
                    "attempt_started": False,
                    "task_success": None,
                    "raw_official_success": None,
                    "score": None,
                    "reason": job_lane_disabled["agentic"],
                    "infrastructure_error": {
                        "reason": job_lane_disabled["agentic"],
                        "preflight_receipt": str(agentic_preflight_path),
                    },
                }
            if not baseline_lane_admitted:
                pair_record["arms"]["pi0_nav_pick_only"] = {
                    "outcome": "not_admitted",
                    "attempt_started": False,
                    "task_success": None,
                    "raw_official_success": None,
                    "score": None,
                    "reason": job_lane_disabled["pi0_nav_pick_only"],
                    "infrastructure_error": {
                        "reason": job_lane_disabled["pi0_nav_pick_only"],
                    },
                }
            (
                pair_blocked_reason,
                blocked_remaining_records,
            ) = _pair_campaign_transition(
                public_seeds=public_seeds,
                pair_index=pair_index,
                pair_record=pair_record,
                pair_safety_errors=pair_safety_errors,
                live_arms=live_arms,
            )
            _append_jsonl(results_path, pair_record)
            manifest["pairs"].append(pair_record)
            manifest["lane_blockers"] = dict(job_lane_disabled)
            _atomic_json(manifest_path, manifest)
            if pair_blocked_reason is not None:
                blocked_reason = pair_blocked_reason
                for remaining_record in blocked_remaining_records:
                    _append_jsonl(results_path, remaining_record)
                break

        intended_final_status = "blocked" if blocked_reason else "completed"
        manifest["status"] = (
            "blocked" if intended_final_status == "blocked" else "finishing"
        )
        manifest["intended_final_status"] = intended_final_status
        manifest["blocked_reason"] = blocked_reason
        manifest["finishing_at"] = _utc_now()
        manifest["health_samples_completed"] = sorted(
            {
                offset
                for offsets in health_samples_by_public_seed.values()
                for offset in offsets
            }
        )
        manifest["health_samples_completed_by_public_seed"] = (
            health_samples_by_public_seed
        )
        manifest["monitor_start_evidence"] = next(
            (
                evidence
                for evidence in monitor_evidence_by_public_seed.values()
                if evidence is not None
            ),
            None,
        )
        manifest["monitor_start_evidence_by_public_seed"] = (
            monitor_evidence_by_public_seed
        )
        _atomic_json(manifest_path, manifest)
        exit_code = 1 if blocked_reason else 0
    except KeyboardInterrupt:
        interrupted = True
        exit_code = 130
    finally:
        process_cleanup_errors: list[str] = []
        for live in live_arms:
            if live.relay is not None:
                try:
                    _stop_relay_within_child_deadline(live, live.child)
                except Exception as error:
                    process_cleanup_errors.append(
                        f"{live.spec.name}_relay_cleanup_error: "
                        f"{type(error).__name__}: {error}"
                    )
            if live.child is not None:
                child = live.child
                if child.process.poll() is None or not child.cleanup_verified:
                    try:
                        child.cleanup_verified = terminate_owned_child(child)
                    except Exception as error:
                        process_cleanup_errors.append(
                            f"{live.spec.name}_runner_cleanup_error: "
                            f"{type(error).__name__}: {error}"
                        )
                if (
                    child.process.poll() is None
                    or not child.cleanup_verified
                    or child.identity_ambiguous
                ):
                    process_cleanup_errors.append(
                        f"{live.spec.name}_runner_cleanup_unverified"
                    )
            try:
                vla_clean = _terminate_verified_vla(live)
            except Exception as error:
                vla_clean = False
                process_cleanup_errors.append(
                    f"{live.spec.name}_vla_cleanup_error: "
                    f"{type(error).__name__}: {error}"
                )
            if not vla_clean:
                process_cleanup_errors.append(
                    f"{live.spec.name}_vla_cleanup_unverified"
                )
            try:
                live.server.stop()
            except RuntimeError:
                pass
        if manifest_path.exists() and interrupted:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
            if isinstance(manifest, dict):
                manifest["status"] = "operator_stop"
                manifest["finished_at"] = _utc_now()
                _atomic_json(manifest_path, manifest)
        if owned_runtime_root is not None and runtime_owner_document is not None:
            cleaning_record = {
                "status": "cleanup_in_progress",
                "attempted_at": _utc_now(),
                "runtime_base": str(owned_runtime_root.base),
                "runtime_root": str(owned_runtime_root.root),
                "runtime_root_device": owned_runtime_root.root_device,
                "runtime_root_inode": owned_runtime_root.root_inode,
                "owner_binding_sha256": runtime_owner_document.get("binding_sha256"),
                "errors": list(process_cleanup_errors),
            }
            _record_external_runtime_cleanup(
                output_root=output_root,
                manifest_path=manifest_path,
                cleanup=cleaning_record,
            )
            cleanup_record = _delete_owned_runtime_root(
                owned_runtime_root,
                owner_document=runtime_owner_document,
                runtime_isolation=runtime_isolation,
                processes_stopped=not process_cleanup_errors,
            )
            if process_cleanup_errors:
                cleanup_record["errors"] = [
                    *process_cleanup_errors,
                    *list(cleanup_record.get("errors", [])),
                ]
            _record_external_runtime_cleanup(
                output_root=output_root,
                manifest_path=manifest_path,
                cleanup=cleanup_record,
            )
            if cleanup_record["status"] != "deleted":
                exit_code = 1
            elif manifest_path.exists() and not interrupted:
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {}
                if isinstance(manifest, dict):
                    manifest["status"] = intended_final_status or "blocked"
                    manifest.pop("intended_final_status", None)
                    manifest["runtime_cleanup_pending"] = False
                    manifest["finished_at"] = _utc_now()
                    _atomic_json(manifest_path, manifest)
        for stream in reversed(locks):
            try:
                fcntl.flock(stream, fcntl.LOCK_UN)
            finally:
                stream.close()
    return 1 if exit_code is None else exit_code


__all__ = [
    "ArmSpec",
    "DEFAULT_AGENTIC_DASHBOARD_PORT",
    "DEFAULT_BASELINE_DASHBOARD_PORT",
    "OwnedChild",
    "build_agentic_runner_argv",
    "build_baseline_runner_argv",
    "main",
    "owned_child_identity_matches",
    "terminate_owned_child",
    "validate_dashboard_endpoints",
    "validate_deadlines",
]
