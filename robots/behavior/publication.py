"""Fail-closed validation for canonical BEHAVIOR Explore publications.

The validator accepts one visible Job root rather than three independently
selected prompt files.  Every input is opened relative to that root without
following symlinks, and the canonical prompt artifacts are tied back to the
immutable raw-success evidence and hidden publication bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    ResourceManifestError,
)
from robots.behavior.run_manifest import (
    resolve_run_manifest_public_tool_contract,
)
from robots.behavior.schemas import (
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    PUBLIC_TOOL_CONTRACTS,
)
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
    BehaviorTaskSpec,
    get_task_spec,
    resolve_task_spec,
)

RAW_SUCCESS_SOURCE = 'info["done"]["success"]'
PUBLICATION_SOURCE = "raw_official_success_v1"
NATIVE_ENV_SEED = 0

PUBLIC_TOOLS = PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]

AMENDMENT_RELATIVE = "publication_amendment.json"
SESSION_MANIFEST_RELATIVE = "session_manifest.json"

SOURCE_ARTIFACT_FILENAMES = {
    "official_success_receipt": "official_success_receipt.json",
    "behavior_action_trace": "behavior_action_trace.jsonl",
    "behavior_tool_trace": "behavior_tool_trace.jsonl",
    "final_result": "final_result.json",
    "run_manifest": "run_manifest.json",
}
SOURCE_ARTIFACT_NAMES = frozenset({*SOURCE_ARTIFACT_FILENAMES, "session_manifest"})

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_CONTROL_FILE_LIMIT = 2 * 1024 * 1024
_TRACE_FILE_LIMIT = 256 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)


class PublicationValidationError(ValueError):
    """The requested Job root is not one complete canonical publication."""


@dataclass(frozen=True)
class BehaviorPublicationIdentity:
    """Task-scoped canonical Explore publication identity and paths."""

    task_spec: BehaviorTaskSpec
    public_seed: int
    native_instance: int
    tag: str
    recipe_relative: str
    memory_relative: str
    provenance_relative: str

    @property
    def core_payload_paths(self) -> tuple[str, str, str]:
        return (
            self.recipe_relative,
            self.memory_relative,
            self.provenance_relative,
        )


def resolve_publication_identity(
    *,
    task_name: str,
    public_seed: int = 0,
    task_index: int | None = None,
) -> BehaviorPublicationIdentity:
    """Resolve one canonical task-local Explore publication identity.

    A task name is always required.  A supplied task index is cross-checked
    against it, and the TaskSpec supplies the activity definition, public
    mapping, native instance, tag, and publication namespace.
    """

    spec = (
        get_task_spec(task_name)
        if task_index is None
        else resolve_task_spec(task_name=task_name, task_index=task_index)
    )
    instance_id = spec.instance_for_public_seed(public_seed, phase="explore")
    tag = spec.tag(public_seed)
    return BehaviorPublicationIdentity(
        task_spec=spec,
        public_seed=public_seed,
        native_instance=instance_id,
        tag=tag,
        recipe_relative=f"recipe_{tag}.jsonl",
        memory_relative=f"memory/{spec.task_name}.md",
        provenance_relative=f"memory/{spec.task_name}_provenance.json",
    )


# Compatibility aliases for the established Radio s0 publication API.  New
# task-aware callers pass task_name/task_index/public_seed to the resolver or
# validator instead of importing these constants as global task identity.
_DEFAULT_IDENTITY = resolve_publication_identity(
    task_name=TURNING_ON_RADIO_TASK_SPEC.task_name,
    task_index=TURNING_ON_RADIO_TASK_SPEC.task_index,
)
TASK_NAME = _DEFAULT_IDENTITY.task_spec.task_name
PUBLIC_SEED = _DEFAULT_IDENTITY.public_seed
TAG = _DEFAULT_IDENTITY.tag
MAPPING_VERSION = _DEFAULT_IDENTITY.task_spec.mapping_version
NATIVE_INSTANCE = _DEFAULT_IDENTITY.native_instance
RECIPE_RELATIVE = _DEFAULT_IDENTITY.recipe_relative
MEMORY_RELATIVE = _DEFAULT_IDENTITY.memory_relative
PROVENANCE_RELATIVE = _DEFAULT_IDENTITY.provenance_relative
CORE_PAYLOAD_PATHS = _DEFAULT_IDENTITY.core_payload_paths


@dataclass(frozen=True)
class ValidatedBehaviorPublication:
    """A stable, fully validated publication snapshot."""

    root: Path
    identity: BehaviorPublicationIdentity
    bundle_id: str
    recipe_bytes: bytes
    memory_bytes: bytes
    provenance_bytes: bytes
    amendment: dict[str, Any]
    provenance: dict[str, Any]
    files: dict[str, dict[str, Any]]
    manifest_binding: dict[str, Any]


@dataclass(frozen=True)
class ForensicPublicationValidation:
    """Fail-closed qualification result for one run-local publication."""

    complete: bool
    reason: str
    identity_tier: str | None = None
    files: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class _ReadRecord:
    relative_path: str
    content: bytes
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def identity(self) -> tuple[int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size,
            "sha256": self.sha256,
        }


def _fail(message: str) -> None:
    raise PublicationValidationError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HASH_PATTERN.fullmatch(value) is not None


def _validate_resource_source_binding(
    value: object,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate a complete immutable dataset binding without reopening its cache."""

    try:
        binding = DatasetResourceBinding.from_dict(value)
    except (ResourceManifestError, TypeError, ValueError) as error:
        raise PublicationValidationError(
            f"{label} is not a complete pinned dataset resource binding"
        ) from error
    if binding.subtree != "behavior" or not binding.files:
        _fail(f"{label} is not a non-empty BEHAVIOR resource binding")
    return binding.as_dict()


def _relative_parts(relative: str) -> tuple[str, ...]:
    path = Path(relative)
    parts = path.parts
    if (
        not relative
        or path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in parts)
    ):
        _fail(f"publication path is not canonical and relative: {relative!r}")
    return tuple(parts)


def canonical_bundle_id(payloads: Mapping[str, bytes]) -> str:
    """Return the canonical bundle digest used by the Explore publisher."""

    if not isinstance(payloads, Mapping) or not payloads:
        _fail("publication bundle payloads must be a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(payloads):
        if not isinstance(name, str):
            _fail("publication bundle names must be strings")
        _relative_parts(name)
        content = payloads[name]
        if not isinstance(content, bytes):
            _fail(f"publication bundle payload must be bytes: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


class _RootReader:
    """Open and recheck files through one non-symlink Job-root descriptor."""

    def __init__(self, root: str | Path):
        requested = Path(root).expanduser().absolute()
        if requested.name.startswith(".") or ".publication_bundles" in requested.parts:
            _fail("publication root must be the visible Job root")
        try:
            root_lstat = requested.lstat()
            resolved = requested.resolve(strict=True)
        except OSError as error:
            raise PublicationValidationError(
                f"publication root is unavailable: {requested}"
            ) from error
        if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
            _fail("publication root must be a non-symlink directory")
        if resolved != requested:
            _fail("publication root or one of its ancestors is a symlink")
        try:
            descriptor = os.open(
                requested,
                _OPEN_FLAGS | _DIRECTORY | _NOFOLLOW,
            )
        except OSError as error:
            raise PublicationValidationError(
                "publication root cannot be opened without following symlinks"
            ) from error
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode):
            os.close(descriptor)
            _fail("publication root descriptor is not a directory")
        if (opened.st_dev, opened.st_ino) != (
            root_lstat.st_dev,
            root_lstat.st_ino,
        ):
            os.close(descriptor)
            _fail("publication root changed while it was opened")
        self.root = requested
        self._descriptor = descriptor
        self._root_identity = (opened.st_dev, opened.st_ino)
        self._records: dict[str, _ReadRecord] = {}

    def __enter__(self) -> "_RootReader":
        return self

    def __exit__(self, *_args: object) -> None:
        os.close(self._descriptor)

    def _open_directory(self, parts: tuple[str, ...]) -> int:
        current = os.dup(self._descriptor)
        try:
            for part in parts:
                next_descriptor = os.open(
                    part,
                    _OPEN_FLAGS | _DIRECTORY | _NOFOLLOW,
                    dir_fd=current,
                )
                metadata = os.fstat(next_descriptor)
                if not stat.S_ISDIR(metadata.st_mode):
                    os.close(next_descriptor)
                    _fail(f"publication path component is not a directory: {part}")
                os.close(current)
                current = next_descriptor
            return current
        except OSError as error:
            os.close(current)
            raise PublicationValidationError(
                "publication path contains a missing, non-directory, or "
                "symlinked component"
            ) from error
        except BaseException:
            os.close(current)
            raise

    def _read_uncached(self, relative: str, *, limit: int) -> _ReadRecord:
        parts = _relative_parts(relative)
        parent = self._open_directory(parts[:-1])
        try:
            descriptor = os.open(
                parts[-1],
                _OPEN_FLAGS | _NOFOLLOW,
                dir_fd=parent,
            )
        except OSError as error:
            os.close(parent)
            raise PublicationValidationError(
                f"publication file is missing, unreadable, or a symlink: {relative}"
            ) from error
        os.close(parent)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                _fail(f"publication input is not a regular file: {relative}")
            if before.st_size > limit:
                _fail(f"publication input exceeds its size limit: {relative}")
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining:
                chunk = os.read(descriptor, min(_READ_CHUNK, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            content = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if len(content) > limit:
            _fail(f"publication input exceeds its size limit: {relative}")
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity or len(content) != before.st_size:
            _fail(f"publication input changed while being read: {relative}")
        return _ReadRecord(
            relative_path=relative,
            content=content,
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            ctime_ns=before.st_ctime_ns,
        )

    def read(self, relative: str, *, limit: int = _CONTROL_FILE_LIMIT) -> bytes:
        existing = self._records.get(relative)
        if existing is not None:
            if existing.size > limit:
                _fail(f"publication input exceeds its size limit: {relative}")
            return existing.content
        record = self._read_uncached(relative, limit=limit)
        self._records[relative] = record
        return record.content

    def require_entries(self, relative: str, expected: set[str]) -> None:
        parts = _relative_parts(relative)
        try:
            descriptor = self._open_directory(parts)
            try:
                actual = set(os.listdir(descriptor))
            finally:
                os.close(descriptor)
        except OSError as error:
            raise PublicationValidationError(
                f"publication directory is missing or unsafe: {relative}"
            ) from error
        if actual != expected:
            _fail(
                f"publication directory entries are non-canonical: {relative}: "
                f"{sorted(actual)!r}"
            )

    def verify_stable(self) -> None:
        try:
            current_root = self.root.lstat()
        except OSError as error:
            raise PublicationValidationError(
                "publication root disappeared during validation"
            ) from error
        if (
            stat.S_ISLNK(current_root.st_mode)
            or (
                current_root.st_dev,
                current_root.st_ino,
            )
            != self._root_identity
        ):
            _fail("publication root changed during validation")
        for relative, expected in tuple(self._records.items()):
            limit = (
                _TRACE_FILE_LIMIT
                if relative.endswith(
                    ("behavior_action_trace.jsonl", "behavior_tool_trace.jsonl")
                )
                else _CONTROL_FILE_LIMIT
            )
            actual = self._read_uncached(relative, limit=limit)
            if (
                actual.identity != expected.identity
                or actual.content != expected.content
            ):
                _fail(f"publication input changed during validation: {relative}")

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {
            relative: record.metadata()
            for relative, record in sorted(self._records.items())
        }


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains a duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=build_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PublicationValidationError(
            f"{label} must be strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _json_lines(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise PublicationValidationError(
            f"{label} must be strict UTF-8 JSONL"
        ) from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:

            def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, item in pairs:
                    if key in result:
                        _fail(
                            f"{label} line {line_number} contains "
                            f"a duplicate JSON key: {key}"
                        )
                    result[key] = item
                return result

            value = json.loads(line, object_pairs_hook=build_object)
        except json.JSONDecodeError as error:
            raise PublicationValidationError(
                f"{label} contains invalid JSON at line {line_number}"
            ) from error
        if not isinstance(value, dict):
            _fail(f"{label} line {line_number} is not a JSON object")
        records.append(value)
    if not records:
        _fail(f"{label} must contain at least one record")
    return records


def _validate_task_publication_content(
    *,
    identity: BehaviorPublicationIdentity,
    recipe_bytes: bytes,
    memory_bytes: bytes,
) -> None:
    """Reject cross-task symbolic content from a task-local namespace."""

    recipe_records = _json_lines(recipe_bytes, label="symbolic recipe")
    for record in recipe_records:
        declared_task = record.get("task")
        if declared_task is not None and declared_task != identity.task_spec.task_name:
            _fail("symbolic recipe declares a different BEHAVIOR task")
    try:
        memory_text = memory_bytes.decode("utf-8", errors="strict")
        recipe_text = recipe_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PublicationValidationError(
            "symbolic recipe and Task Memory must be strict UTF-8"
        ) from error
    if "\x00" in recipe_text or "\x00" in memory_text:
        _fail("symbolic recipe and Task Memory must not contain NUL bytes")
    combined = f"{recipe_text}\n{memory_text}".lower()
    other_task_names = {
        TURNING_ON_RADIO_TASK_SPEC.task_name,
        PICKING_UP_TRASH_TASK_SPEC.task_name,
    } - {identity.task_spec.task_name}
    if any(other_name in combined for other_name in other_task_names):
        _fail("publication contains a different task namespace")
    if identity.task_spec.surface_review_policy is None:
        radio_only_markers = (
            "radio_tipped_flat",
            "button-face",
            "button face",
            "control face",
            "opposite face",
            "radio receiver",
        )
        if any(marker in combined for marker in radio_only_markers):
            _fail("publication contains Radio-only task policy")


def _raw_done_success(record: Mapping[str, Any]) -> bool:
    done = record.get("info_done")
    return isinstance(done, dict) and done.get("success") is True


def _action_record_matches_receipt_env_step(
    record: Mapping[str, Any],
    *,
    receipt_env_step: int,
) -> bool:
    """Match current or legacy action-trace lineage without integer coercion."""

    if "env_idx" in record:
        env_idx = record["env_idx"]
        if not isinstance(env_idx, int) or isinstance(env_idx, bool) or env_idx != 0:
            return False

    if "env_step" in record:
        env_step = record["env_step"]
        return (
            isinstance(env_step, int)
            and not isinstance(env_step, bool)
            and env_step == receipt_env_step
        )

    env_idx = record.get("env_idx")
    step = record.get("step")
    return (
        isinstance(env_idx, int)
        and not isinstance(env_idx, bool)
        and env_idx == 0
        and isinstance(step, int)
        and not isinstance(step, bool)
        and step >= 0
        and step + 1 == receipt_env_step
    )


def _receipt_from_result(result: Mapping[str, Any]) -> dict[str, Any] | None:
    direct = result.get("official_success_receipt")
    if isinstance(direct, dict):
        return direct
    monitor = result.get("pi0_nav_pick_monitor")
    if isinstance(monitor, dict) and isinstance(
        monitor.get("official_success_receipt"), dict
    ):
        return monitor["official_success_receipt"]
    for info_name in ("info", "last_info"):
        info = result.get(info_name)
        runtime = info.get("_rpent") if isinstance(info, dict) else None
        if isinstance(runtime, dict) and isinstance(
            runtime.get("official_success_receipt"), dict
        ):
            return runtime["official_success_receipt"]
        nested_monitor = (
            runtime.get("pi0_nav_pick_monitor") if isinstance(runtime, dict) else None
        )
        if isinstance(nested_monitor, dict) and isinstance(
            nested_monitor.get("official_success_receipt"), dict
        ):
            return nested_monitor["official_success_receipt"]
    return None


def _validate_no_live_managed_processes(
    processes: Any,
    *,
    label: str,
) -> None:
    if not isinstance(processes, dict):
        _fail(f"{label} lacks process records")
    for name, process in processes.items():
        if not isinstance(process, dict):
            _fail(f"{label} process record is invalid: {name}")
        if process.get("managed") is not True:
            continue
        if not isinstance(process.get("stopped_at"), str) or not process["stopped_at"]:
            _fail(f"{label} managed process is not stopped: {name}")
        pid = process.get("pid")
        start_ticks = process.get("start_ticks")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(start_ticks, int)
            or isinstance(start_ticks, bool)
        ):
            continue
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            current_start_ticks = int(fields[19])
        except (OSError, ValueError, IndexError):
            continue
        if current_start_ticks == start_ticks:
            _fail(f"{label} still owns a live process: {name}")


def _resolve_session_public_tool_contract(
    session: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Resolve the outer Explore session's source tool surface.

    Session manifests retain their independent schema-1 lifecycle.  Missing
    tool-contract metadata is interpreted only as historical v1; new sessions
    declare their version explicitly.
    """

    if session.get("schema_version") != 1:
        _fail("session manifest schema is unsupported")
    protocol = session.get("protocol")
    if not isinstance(protocol, Mapping):
        _fail("session manifest protocol is missing")
    declared_version = protocol.get("public_tool_contract_version")
    if declared_version is None:
        version = 1
    elif (
        isinstance(declared_version, bool)
        or not isinstance(declared_version, int)
        or declared_version not in PUBLIC_TOOL_CONTRACTS
    ):
        _fail("session manifest public-tool contract version is invalid")
    else:
        version = declared_version
    expected_tools = PUBLIC_TOOL_CONTRACTS[version]
    if tuple(protocol.get("public_primitives") or ()) != expected_tools:
        _fail(f"session manifest public primitives do not match contract v{version}")
    return version, expected_tools


def _validate_session_manifest(
    session: dict[str, Any],
    *,
    root: Path,
    job_id: str,
    attempt_index: int,
    identity: BehaviorPublicationIdentity,
) -> tuple[dict[str, Any], int]:
    source_tool_contract_version, _ = _resolve_session_public_tool_contract(session)
    if (
        session.get("job_id") != job_id
        or session.get("status") != "succeeded"
        or not isinstance(session.get("finished_at"), str)
        or session.get("blocked_reason") is not None
        or session.get("task_success") is not True
        or session.get("publication_complete") is not False
    ):
        _fail("session manifest is not one immutable successful Explore Job")
    for field in (
        "artifact_seal_complete",
        "workflow_complete",
        "publication_complete",
    ):
        if not isinstance(session.get(field), bool):
            _fail(f"session manifest has an invalid boolean field: {field}")
    protocol = session.get("protocol")
    declared_task_name = (
        protocol.get("task_name") if isinstance(protocol, dict) else None
    )
    declared_task_index = (
        protocol.get("task_index") if isinstance(protocol, dict) else None
    )
    if (
        not isinstance(protocol, dict)
        or protocol.get("behavior_phase") != "explore"
        or protocol.get("public_seed") != identity.public_seed
        or protocol.get("recipe_tag") != identity.tag
        or (
            declared_task_name is not None
            and declared_task_name != identity.task_spec.task_name
        )
        or (
            declared_task_index is not None
            and declared_task_index != identity.task_spec.task_index
        )
        or (declared_task_name is None) != (declared_task_index is None)
        or (
            identity.task_spec.task_name != TASK_NAME
            and (declared_task_name is None or declared_task_index is None)
        )
        or protocol.get("reset_registered") is not False
        or protocol.get("agent_finish_registered") is not False
    ):
        _fail("session manifest does not describe the canonical Explore protocol")
    native = session.get("native_binding")
    if (
        not isinstance(native, dict)
        or native.get("mapping_version") != identity.task_spec.mapping_version
        or native.get("activity_definition_id")
        != identity.task_spec.activity_definition_id
        or native.get("activity_instance_id") != identity.native_instance
        or native.get("env_seed") != NATIVE_ENV_SEED
    ):
        _fail("session manifest native binding is not canonical")
    reviewed_memory = session.get("reviewed_repo_memory")
    if not isinstance(reviewed_memory, dict) or not _is_sha256(
        reviewed_memory.get("snapshot_sha256")
    ):
        _fail("session manifest lacks a valid reviewed Global Memory snapshot")
    reviewed_files = reviewed_memory.get("files")
    if not isinstance(reviewed_files, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(metadata, dict)
        or not _is_sha256(metadata.get("sha256"))
        for name, metadata in reviewed_files.items()
    ):
        _fail("session manifest reviewed Global Memory files are invalid")
    processes = session.get("processes")
    if (
        not isinstance(processes, dict)
        or not isinstance(processes.get("vla"), dict)
        or processes["vla"].get("managed") is not True
    ):
        _fail("session manifest lacks its Job-owned persistent VLA process")
    attempts = session.get("attempts")
    if not isinstance(attempts, list):
        _fail("session manifest attempt records are invalid")
    successful = [
        record
        for record in attempts
        if isinstance(record, dict) and record.get("task_success") is True
    ]
    if len(successful) != 1:
        _fail("session manifest must contain exactly one successful attempt")
    success = successful[0]
    expected_attempt = root / "attempts" / identity.tag / f"attempt_{attempt_index:03d}"
    declared_output = success.get("output_dir")
    try:
        declared_resolved = (
            Path(declared_output).expanduser().resolve(strict=True)
            if isinstance(declared_output, str)
            else None
        )
        expected_resolved = expected_attempt.resolve(strict=True)
    except OSError as error:
        raise PublicationValidationError(
            "session manifest successful attempt path is unavailable"
        ) from error
    if (
        success.get("attempt_index") != attempt_index
        or success.get("outcome") != "official_success"
        or success.get("forced_cleanup_groups") not in ({}, None)
        or declared_resolved != expected_resolved
    ):
        _fail("session manifest successful attempt binding is invalid")
    _validate_no_live_managed_processes(
        processes,
        label="session manifest",
    )
    return reviewed_memory, source_tool_contract_version


def _validate_run_manifest(
    run_manifest: dict[str, Any],
    *,
    job_id: str,
    attempt_index: int,
    reviewed_memory: dict[str, Any],
    resource_source: dict[str, Any] | None,
    identity: BehaviorPublicationIdentity,
) -> int:
    try:
        source_tool_contract_version, _ = resolve_run_manifest_public_tool_contract(
            run_manifest
        )
    except ValueError as error:
        raise PublicationValidationError(
            f"attempt run manifest public-tool contract is invalid: {error}"
        ) from error
    job = run_manifest.get("job")
    protocol = run_manifest.get("protocol")
    native = run_manifest.get("native_binding")
    task = run_manifest.get("task")
    if (
        run_manifest.get("status") != "stopped"
        or not isinstance(run_manifest.get("stopped_at"), str)
        or not isinstance(job, dict)
        or job.get("job_id") != job_id
        or job.get("attempt_index") != attempt_index
        or run_manifest.get("reviewed_repo_memory") != reviewed_memory
        or (
            resource_source is not None
            and run_manifest.get("resource_source") != resource_source
        )
    ):
        _fail("attempt run manifest lifecycle or Job binding is invalid")
    attempts = protocol.get("attempts") if isinstance(protocol, dict) else None
    task_spec = protocol.get("task_spec") if isinstance(protocol, dict) else None
    prompt = protocol.get("prompt") if isinstance(protocol, dict) else None
    if (
        not isinstance(protocol, dict)
        or protocol.get("behavior_phase") != "explore"
        or protocol.get("public_seed") != identity.public_seed
        or protocol.get("recipe_tag") != identity.tag
        or protocol.get("agent_finish_registered") is not False
        or not isinstance(attempts, dict)
        or attempts.get("initial_attempt_index") != attempt_index
        or attempts.get("max_attempts") is not None
        or attempts.get("reset_registered") is not False
    ):
        _fail("attempt run manifest protocol binding is invalid")
    if (
        not isinstance(task_spec, dict)
        or task_spec.get("task_name") != identity.task_spec.task_name
        or task_spec.get("prompt_profile_id") != identity.task_spec.prompt_profile_id
        or not isinstance(prompt, dict)
        or prompt.get("profile_id") != identity.task_spec.prompt_profile_id
        or not _is_sha256(prompt.get("rendered_system_sha256"))
        or not _is_sha256(prompt.get("rendered_user_sha256"))
        or not _is_sha256(prompt.get("combined_sha256"))
    ):
        _fail("attempt run manifest task prompt binding is invalid")
    if (
        not isinstance(native, dict)
        or native.get("activity_definition_id")
        != identity.task_spec.activity_definition_id
        or native.get("activity_instance_id") != identity.native_instance
        or native.get("env_seed") != NATIVE_ENV_SEED
        or not isinstance(task, dict)
        or task.get("suite") != "behavior_2025_challenge"
        or task.get("task") != identity.task_spec.task_index
        or task.get("task_name") != identity.task_spec.task_name
        or task.get("public_seed") != identity.public_seed
    ):
        _fail("attempt run manifest task or native binding is invalid")
    if run_manifest.get("frozen_eval_inputs") is not None:
        _fail("Explore attempt unexpectedly consumed frozen Eval inputs")
    processes = run_manifest.get("processes")
    if (
        not isinstance(processes, dict)
        or not isinstance(processes.get("env"), dict)
        or processes["env"].get("managed") is not True
        or not isinstance(processes.get("vla"), dict)
        or processes["vla"].get("managed") is not False
    ):
        _fail("attempt run manifest process ownership is invalid")
    _validate_no_live_managed_processes(
        processes,
        label="attempt run manifest",
    )
    return source_tool_contract_version


def _forensic_source_identity(
    document: Mapping[str, Any],
    *,
    root: Path,
    job_id: str | None,
    attempt_index: int,
) -> tuple[str, tuple[Any, ...]] | None:
    declared = document.get("publication_identity")
    if not isinstance(declared, dict):
        return None
    identity = declared
    keys = set(identity)
    declared_attempt = identity.get("attempt_index")
    if (
        not isinstance(declared_attempt, int)
        or isinstance(declared_attempt, bool)
        or declared_attempt != attempt_index
    ):
        return None

    declared_job = identity.get("job_id")
    declared_root = identity.get("job_root_path")
    nonce = identity.get("nonce")
    attempt_nonce = identity.get("attempt_nonce")
    if keys in (
        {"job_id", "attempt_index", "nonce"},
        {"job_id", "attempt_index", "attempt_nonce"},
    ):
        effective_nonce = nonce if "nonce" in identity else attempt_nonce
        if (
            not isinstance(declared_job, str)
            or _JOB_ID_PATTERN.fullmatch(declared_job) is None
            or job_id is None
            or declared_job != job_id
            or not isinstance(effective_nonce, str)
            or not effective_nonce
        ):
            return None
        return (
            "job_attempt_nonce",
            (declared_job, attempt_index, effective_nonce),
        )
    if keys == {"job_id", "attempt_index"}:
        if (
            not isinstance(declared_job, str)
            or _JOB_ID_PATTERN.fullmatch(declared_job) is None
            or job_id is None
            or declared_job != job_id
        ):
            return None
        return ("job_attempt", (declared_job, attempt_index))
    if keys != {"job_root_path", "attempt_index"}:
        return None
    if not isinstance(declared_root, str):
        return None
    try:
        canonical_declared_root = Path(declared_root).expanduser().resolve(strict=True)
    except OSError:
        return None
    if canonical_declared_root != root:
        return None
    return ("job_root_attempt", (str(root), attempt_index))


def _forensic_success_evidence(
    document: Mapping[str, Any],
) -> tuple[str, int, str] | None:
    declared = document.get("official_success_binding")
    if not isinstance(declared, dict):
        return None
    field_path = declared.get("field_path")
    first_success_step = declared.get("first_success_step")
    action_trace_sha256 = declared.get("action_trace_sha256")
    if (
        field_path != "info_done.success"
        or not isinstance(first_success_step, int)
        or isinstance(first_success_step, bool)
        or first_success_step < 0
        or not _is_sha256(action_trace_sha256)
    ):
        return None
    return (field_path, first_success_step, action_trace_sha256)


def _forensic_trace_success_step(raw: bytes) -> int | None:
    records = _json_lines(raw, label="forensic behavior action trace")
    for record in records:
        if not _raw_done_success(record):
            continue
        step = record.get("step")
        if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
            return step
        env_step = record.get("env_step")
        if (
            isinstance(env_step, int)
            and not isinstance(env_step, bool)
            and env_step >= 0
        ):
            return env_step
        return None
    return None


def _forensic_validation(
    job_root: str | Path,
    attempt_dir: str | Path,
    official_success_binding: Mapping[str, Any],
) -> ForensicPublicationValidation:
    expected_evidence = _forensic_success_evidence(
        {"official_success_binding": dict(official_success_binding)}
    )
    if (
        official_success_binding.get("source") != "behavior_action_trace"
        or expected_evidence is None
    ):
        _fail("official success binding is not current action-trace evidence")

    with _RootReader(job_root) as reader:
        attempt = Path(attempt_dir).expanduser().absolute()
        try:
            resolved_attempt = attempt.resolve(strict=True)
            attempt_relative_path = resolved_attempt.relative_to(reader.root)
        except (OSError, ValueError) as error:
            raise PublicationValidationError(
                "forensic attempt is outside the visible Job root"
            ) from error
        if resolved_attempt != attempt:
            _fail("forensic attempt or one of its ancestors is a symlink")

        session = _json_object(
            reader.read(SESSION_MANIFEST_RELATIVE),
            label="forensic session manifest",
        )
        declared_job_id = session.get("job_id")
        job_id = (
            declared_job_id
            if isinstance(declared_job_id, str)
            and _JOB_ID_PATTERN.fullmatch(declared_job_id) is not None
            else None
        )
        protocol = session.get("protocol")
        if not isinstance(protocol, dict):
            _fail("forensic session manifest lacks a task protocol")
        task_name = protocol.get("task_name")
        public_seed = protocol.get("public_seed")
        task_index = protocol.get("task_index")
        if (
            not isinstance(task_name, str)
            or not isinstance(public_seed, int)
            or isinstance(public_seed, bool)
            or public_seed < 0
            or not isinstance(task_index, int)
            or isinstance(task_index, bool)
            or task_index < 0
        ):
            _fail("forensic session manifest lacks a task-scoped public seed")
        try:
            identity = resolve_publication_identity(
                task_name=task_name,
                task_index=task_index,
                public_seed=public_seed,
            )
        except (KeyError, ValueError) as error:
            raise PublicationValidationError(
                "forensic session manifest task identity is invalid"
            ) from error

        attempt_match = re.fullmatch(r"attempt_([0-9]{3,})", attempt.name)
        if attempt_match is None:
            _fail("forensic attempt directory name is invalid")
        attempt_index = int(attempt_match.group(1))
        if attempt_index < 1:
            _fail("forensic attempt index must be positive")
        expected_attempt_relative = (
            Path("attempts") / identity.tag / f"attempt_{attempt_index:03d}"
        )
        if attempt_relative_path != expected_attempt_relative:
            _fail("forensic attempt path does not match the task identity")
        attempt_relative = expected_attempt_relative.as_posix()

        attempt_recipe_relative = (
            f"{attempt_relative}/{Path(identity.recipe_relative).name}"
        )
        try:
            (reader.root / attempt_recipe_relative).lstat()
        except FileNotFoundError:
            recipe_relative = identity.recipe_relative
        else:
            recipe_relative = attempt_recipe_relative
        receipt_relative = f"{attempt_relative}/official_success_receipt.json"
        action_relative = f"{attempt_relative}/behavior_action_trace.jsonl"

        recipe_bytes = reader.read(recipe_relative)
        memory_bytes = reader.read(identity.memory_relative)
        provenance_bytes = reader.read(identity.provenance_relative)
        receipt_bytes = reader.read(receipt_relative)
        action_bytes = reader.read(action_relative, limit=_TRACE_FILE_LIMIT)
        provenance = _json_object(
            provenance_bytes,
            label="forensic publication provenance",
        )
        receipt = _json_object(
            receipt_bytes,
            label="forensic official success receipt",
        )
        _validate_task_publication_content(
            identity=identity,
            recipe_bytes=recipe_bytes,
            memory_bytes=memory_bytes,
        )

        provenance_identity = _forensic_source_identity(
            provenance,
            root=reader.root,
            job_id=job_id,
            attempt_index=attempt_index,
        )
        receipt_identity = _forensic_source_identity(
            receipt,
            root=reader.root,
            job_id=job_id,
            attempt_index=attempt_index,
        )
        if (
            provenance_identity is None
            or receipt_identity is None
            or provenance_identity != receipt_identity
        ):
            _fail("forensic publication identity is incomplete or inconsistent")

        if (
            _forensic_success_evidence(provenance) != expected_evidence
            or _forensic_success_evidence(receipt) != expected_evidence
        ):
            _fail("forensic publication raw-success evidence is incomplete")
        action_sha256 = hashlib.sha256(action_bytes).hexdigest()
        if (
            action_sha256 != expected_evidence[2]
            or _forensic_trace_success_step(action_bytes) != expected_evidence[1]
        ):
            _fail("forensic publication action trace does not match its binding")

        recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
        memory_sha256 = hashlib.sha256(memory_bytes).hexdigest()
        receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        if (
            provenance.get("recipe_sha256") != recipe_sha256
            or provenance.get("memory_sha256") != memory_sha256
            or provenance.get("receipt_sha256") != receipt_sha256
        ):
            _fail("forensic publication payload hashes are incomplete")
        if (
            provenance.get("task") is not None
            and provenance.get("task") != identity.task_spec.task_name
        ):
            _fail("forensic publication provenance declares a different task")

        reader.verify_stable()
        files = {
            "recipe": {
                "relative_path": recipe_relative,
                "sha256": recipe_sha256,
            },
            "memory": {
                "relative_path": identity.memory_relative,
                "sha256": memory_sha256,
            },
            "provenance": {
                "relative_path": identity.provenance_relative,
                "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
            },
            "receipt": {
                "relative_path": receipt_relative,
                "sha256": receipt_sha256,
            },
        }
        return ForensicPublicationValidation(
            complete=True,
            reason="run-local publication is bound to raw action-trace success",
            identity_tier=provenance_identity[0],
            files=files,
        )


def validate_forensic_publication_binding(
    job_root: str | Path,
    attempt_dir: str | Path,
    official_success_binding: Mapping[str, Any],
) -> ForensicPublicationValidation:
    """Qualify a run-local forensic publication without blocking correction.

    This intentionally does not reuse or relax canonical publication rules.
    Missing, unsafe, ambiguous, or inconsistently bound artifacts simply make
    publication incomplete; action-trace task-success correction remains
    independent.
    """

    try:
        return _forensic_validation(
            job_root,
            attempt_dir,
            official_success_binding,
        )
    except Exception as error:
        return ForensicPublicationValidation(
            complete=False,
            reason=str(error) or error.__class__.__name__,
        )


def validate_canonical_publication_root(
    root: str | Path,
    *,
    expected_provenance_sha256: str | None = None,
    expected_job_id: str | None = None,
    task_name: str | None = None,
    task_index: int | None = None,
    public_seed: int | None = None,
) -> ValidatedBehaviorPublication:
    """Validate and snapshot one complete canonical task-local publication."""

    if expected_provenance_sha256 is not None:
        expected_provenance_sha256 = str(expected_provenance_sha256).lower()
        if not _is_sha256(expected_provenance_sha256):
            _fail("expected provenance SHA256 must be 64 hexadecimal characters")
    if expected_job_id is not None and not _JOB_ID_PATTERN.fullmatch(
        str(expected_job_id)
    ):
        _fail("expected Job ID has an invalid format")
    with _RootReader(root) as reader:
        session_bytes = reader.read(SESSION_MANIFEST_RELATIVE)
        session = _json_object(session_bytes, label="session manifest")
        protocol = session.get("protocol")
        declared_task_name = (
            protocol.get("task_name") if isinstance(protocol, dict) else None
        )
        declared_task_index = (
            protocol.get("task_index") if isinstance(protocol, dict) else None
        )
        declared_public_seed = (
            protocol.get("public_seed") if isinstance(protocol, dict) else None
        )
        resolved_name = (
            task_name
            if task_name is not None
            else declared_task_name
            if isinstance(declared_task_name, str)
            else TASK_NAME
        )
        resolved_index = (
            task_index
            if task_index is not None
            else declared_task_index
            if isinstance(declared_task_index, int)
            and not isinstance(declared_task_index, bool)
            else None
        )
        resolved_public_seed = (
            public_seed
            if public_seed is not None
            else declared_public_seed
            if isinstance(declared_public_seed, int)
            and not isinstance(declared_public_seed, bool)
            else PUBLIC_SEED
        )
        try:
            identity = resolve_publication_identity(
                task_name=resolved_name,
                task_index=resolved_index,
                public_seed=resolved_public_seed,
            )
        except ValueError as error:
            raise PublicationValidationError(
                f"invalid BEHAVIOR publication task identity: {error}"
            ) from error

        recipe_bytes = reader.read(identity.recipe_relative)
        memory_bytes = reader.read(identity.memory_relative)
        provenance_bytes = reader.read(identity.provenance_relative)
        amendment_bytes = reader.read(AMENDMENT_RELATIVE)

        payloads = {
            identity.recipe_relative: recipe_bytes,
            identity.memory_relative: memory_bytes,
            identity.provenance_relative: provenance_bytes,
        }
        bundle_id = canonical_bundle_id(payloads)
        bundle_root = f".publication_bundles/{bundle_id}"
        reader.require_entries(
            bundle_root,
            {Path(identity.recipe_relative).name, "memory"},
        )
        reader.require_entries(
            f"{bundle_root}/memory",
            {
                Path(identity.memory_relative).name,
                Path(identity.provenance_relative).name,
            },
        )
        for relative, expected in payloads.items():
            bundled = reader.read(f"{bundle_root}/{relative}")
            if bundled != expected:
                _fail(f"hidden bundle differs from canonical artifact: {relative}")

        amendment = _json_object(amendment_bytes, label="publication amendment")
        provenance = _json_object(provenance_bytes, label="publication provenance")
        _validate_task_publication_content(
            identity=identity,
            recipe_bytes=recipe_bytes,
            memory_bytes=memory_bytes,
        )

        provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
        if (
            expected_provenance_sha256 is not None
            and provenance_sha256 != expected_provenance_sha256
        ):
            _fail("publication provenance SHA256 does not match its trusted pin")
        recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
        memory_sha256 = hashlib.sha256(memory_bytes).hexdigest()
        job_id = provenance.get("job_id")
        attempt_index = provenance.get("attempt_index")
        if (
            provenance.get("schema_version") not in {2, 3}
            or provenance.get("derived_offline") is not True
            or provenance.get("task") != identity.task_spec.task_name
            or provenance.get("public_seed") != identity.public_seed
            or provenance.get("source_tag") != identity.tag
            or (
                provenance.get("task_index") is not None
                and provenance.get("task_index") != identity.task_spec.task_index
            )
            or (
                provenance.get("activity_definition_id") is not None
                and provenance.get("activity_definition_id")
                != identity.task_spec.activity_definition_id
            )
            or (
                provenance.get("activity_instance_id") is not None
                and provenance.get("activity_instance_id") != identity.native_instance
            )
            or (
                identity.task_spec.task_name != TASK_NAME
                and (
                    provenance.get("task_index") is None
                    or provenance.get("activity_definition_id") is None
                    or provenance.get("activity_instance_id") is None
                )
            )
            or provenance.get("source") != PUBLICATION_SOURCE
            or provenance.get("success_source") != RAW_SUCCESS_SOURCE
            or provenance.get("task_success") is not True
            or not isinstance(job_id, str)
            or _JOB_ID_PATTERN.fullmatch(job_id) is None
            or not isinstance(attempt_index, int)
            or isinstance(attempt_index, bool)
            or attempt_index < 1
            or not isinstance(provenance.get("attempt_nonce"), str)
            or not provenance["attempt_nonce"]
            or provenance.get("recipe_sha256") != recipe_sha256
            or provenance.get("memory_sha256") != memory_sha256
        ):
            _fail("publication provenance identity or core hashes are invalid")
        if expected_job_id is not None and job_id != expected_job_id:
            _fail("publication Job ID does not match the expected Job")

        expected_amendment = {
            "schema_version": 2,
            "kind": "posthoc_publication_override",
            "job_id": job_id,
            "tag": identity.tag,
            "public_seed": identity.public_seed,
            "success_source": RAW_SUCCESS_SOURCE,
            "task_success": True,
            "publication_complete": True,
            "publication_source": PUBLICATION_SOURCE,
            "attempt_index": attempt_index,
            "recipe_sha256": recipe_sha256,
            "memory_sha256": memory_sha256,
            "provenance_sha256": provenance_sha256,
            "original_attempt_immutable": True,
            "bundle_id": bundle_id,
        }
        if any(
            amendment.get(key) != value for key, value in expected_amendment.items()
        ):
            _fail("publication amendment identity, bundle, or hashes are invalid")

        reviewed_memory, source_tool_contract_version = _validate_session_manifest(
            session,
            root=reader.root,
            job_id=job_id,
            attempt_index=attempt_index,
            identity=identity,
        )
        resource_source = None
        if provenance["schema_version"] == 3:
            resource_source = _validate_resource_source_binding(
                session.get("resource_source"),
                label="session manifest resource source",
            )
        global_memory_sha256 = reviewed_memory["snapshot_sha256"]
        if provenance["schema_version"] == 3:
            selection = reviewed_memory.get("selection")
            selected_files = (
                selection.get("files") if isinstance(selection, dict) else None
            )
            expected_task_directory = identity.task_spec.task_name
            roles = selection.get("roles") if isinstance(selection, dict) else None
            target_prior = (
                roles.get("target_prior") if isinstance(roles, dict) else None
            )
            explore_experience = (
                roles.get("explore_experience") if isinstance(roles, dict) else None
            )
            additional = (
                roles.get("additional_expert_knowledge")
                if isinstance(roles, dict)
                else None
            )
            if (
                not isinstance(selection, dict)
                or selection.get("task_name") != identity.task_spec.task_name
                or selection.get("task_directory") != expected_task_directory
                or not _is_sha256(selection.get("selection_sha256"))
                or not _is_sha256(selection.get("prompt_sha256"))
                or not isinstance(selected_files, dict)
                or not selected_files
                or set(roles or ())
                != {
                    "target_prior",
                    "explore_experience",
                    "additional_expert_knowledge",
                }
                or not isinstance(target_prior, str)
                or not isinstance(explore_experience, str)
                or not isinstance(additional, list)
                or any(not isinstance(name, str) for name in additional)
            ):
                _fail(
                    "publication lacks a task-scoped reviewed Global Memory selection"
                )
            global_memory_files_sha256: dict[str, str] = {}
            task_prefix = f"{expected_task_directory}/"
            role_paths = [target_prior, explore_experience, *additional]
            if len(set(role_paths)) != len(role_paths) or set(role_paths) != set(
                selected_files
            ):
                _fail(
                    "publication reviewed Global Memory roles do not match "
                    "the task selection"
                )
            for name, metadata in selected_files.items():
                full_metadata = reviewed_memory["files"].get(name)
                if (
                    not isinstance(name, str)
                    or not name.startswith(task_prefix)
                    or not isinstance(metadata, dict)
                    or set(metadata) != {"relative_path", "size_bytes", "sha256"}
                    or metadata.get("relative_path") != name
                    or not isinstance(metadata.get("size_bytes"), int)
                    or isinstance(metadata.get("size_bytes"), bool)
                    or metadata["size_bytes"] < 0
                    or not _is_sha256(metadata.get("sha256"))
                    or not isinstance(full_metadata, dict)
                    or full_metadata.get("relative_path") != name
                    or full_metadata.get("size_bytes") != metadata["size_bytes"]
                    or full_metadata.get("sha256") != metadata["sha256"]
                    or full_metadata.get("included_in_prompt") is not True
                ):
                    _fail(
                        "publication reviewed Global Memory selection is not "
                        "strictly task-scoped"
                    )
                global_memory_files_sha256[name] = metadata["sha256"]
        else:
            # Schema v2 is retained only for validation of immutable historical
            # publications produced before task-scoped metadata projection.
            global_memory_files_sha256 = {
                name: metadata["sha256"]
                for name, metadata in reviewed_memory["files"].items()
            }
        if (
            provenance.get("global_memory_snapshot_sha256") != global_memory_sha256
            or provenance.get("global_memory_files_sha256")
            != global_memory_files_sha256
            or amendment.get("global_memory_snapshot_sha256") != global_memory_sha256
        ):
            _fail("publication does not bind the reviewed Global Memory snapshot")
        if amendment.get("artifact_seal_complete") != session.get(
            "artifact_seal_complete"
        ):
            _fail("publication amendment artifact-seal binding is invalid")
        overlay = amendment.get("overlay_semantics")
        expected_preserved = {
            "task_success": True,
            "artifact_seal_complete": session["artifact_seal_complete"],
            "workflow_complete": session["workflow_complete"],
            "publication_complete": False,
        }
        if (
            not isinstance(overlay, dict)
            or overlay.get("overrides") != {"publication_complete": True}
            or overlay.get("preserves_session_manifest") != expected_preserved
        ):
            _fail("publication amendment overlay semantics are invalid")

        source_hashes = provenance.get("source_artifacts_sha256")
        if (
            not isinstance(source_hashes, dict)
            or set(source_hashes) != SOURCE_ARTIFACT_NAMES
            or any(not _is_sha256(value) for value in source_hashes.values())
        ):
            _fail("publication provenance must declare exactly six source hashes")

        attempt_relative = f"attempts/{identity.tag}/attempt_{attempt_index:03d}"
        source_bytes: dict[str, bytes] = {"session_manifest": session_bytes}
        for name, filename in SOURCE_ARTIFACT_FILENAMES.items():
            limit = (
                _TRACE_FILE_LIMIT
                if name in {"behavior_action_trace", "behavior_tool_trace"}
                else _CONTROL_FILE_LIMIT
            )
            source_bytes[name] = reader.read(
                f"{attempt_relative}/{filename}",
                limit=limit,
            )
        for name, content in source_bytes.items():
            if hashlib.sha256(content).hexdigest() != source_hashes[name]:
                _fail(f"publication source artifact hash mismatch: {name}")

        run_manifest = _json_object(
            source_bytes["run_manifest"],
            label="attempt run manifest",
        )
        run_tool_contract_version = _validate_run_manifest(
            run_manifest,
            job_id=job_id,
            attempt_index=attempt_index,
            reviewed_memory=reviewed_memory,
            resource_source=resource_source,
            identity=identity,
        )
        if run_tool_contract_version != source_tool_contract_version:
            _fail("session and attempt public-tool contract versions differ")
        source_public_tools = PUBLIC_TOOL_CONTRACTS[source_tool_contract_version]

        receipt_bytes = source_bytes["official_success_receipt"]
        receipt = _json_object(receipt_bytes, label="official success receipt")
        receipt_binding = provenance.get("official_success_receipt")
        if not isinstance(receipt_binding, dict) or set(receipt_binding) != {
            "source",
            "run_nonce",
            "attempt_nonce",
            "attempt_index",
            "env_step",
            "receipt_sha256",
            "file_sha256",
        }:
            _fail("publication receipt binding schema is invalid")
        unsigned_receipt = dict(receipt)
        claimed_receipt_sha256 = unsigned_receipt.pop("receipt_sha256", None)
        actual_receipt_sha256 = hashlib.sha256(
            json.dumps(
                unsigned_receipt,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if (
            receipt.get("schema_version") != 1
            or receipt.get("source") != RAW_SUCCESS_SOURCE
            or not isinstance(receipt.get("raw_done"), dict)
            or receipt["raw_done"].get("success") is not True
            or not isinstance(receipt.get("run_nonce"), str)
            or not receipt["run_nonce"]
            or receipt.get("attempt_nonce") != provenance["attempt_nonce"]
            or receipt.get("attempt_index") != attempt_index
            or not isinstance(receipt.get("env_step"), int)
            or isinstance(receipt.get("env_step"), bool)
            or receipt["env_step"] < 0
            or claimed_receipt_sha256 != actual_receipt_sha256
        ):
            _fail("official success receipt is invalid")
        expected_receipt_binding = {
            "source": receipt["source"],
            "run_nonce": receipt["run_nonce"],
            "attempt_nonce": receipt["attempt_nonce"],
            "attempt_index": receipt["attempt_index"],
            "env_step": receipt["env_step"],
            "receipt_sha256": receipt["receipt_sha256"],
            "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        }
        if receipt_binding != expected_receipt_binding:
            _fail("publication receipt binding does not match the receipt file")

        action_records = _json_lines(
            source_bytes["behavior_action_trace"],
            label="behavior action trace",
        )
        if not any(
            _action_record_matches_receipt_env_step(
                record,
                receipt_env_step=receipt["env_step"],
            )
            and _raw_done_success(record)
            for record in action_records
        ):
            _fail("action trace lacks raw success at the receipt environment step")

        tool_records = _json_lines(
            source_bytes["behavior_tool_trace"],
            label="behavior tool trace",
        )
        for record in tool_records:
            if record.get("tool") not in source_public_tools:
                _fail("tool trace contains a non-public primitive")
        matching_tool_receipt = False
        for record in tool_records:
            result = record.get("result")
            if not isinstance(result, dict) or result.get("task_success") is not True:
                continue
            if (
                result.get("attempt_nonce") != receipt["attempt_nonce"]
                or result.get("run_nonce") != receipt["run_nonce"]
                or result.get("official_success_source")
                not in {
                    RAW_SUCCESS_SOURCE,
                    f"{RAW_SUCCESS_SOURCE} via task_success",
                }
                or _receipt_from_result(result) != receipt
            ):
                continue
            matching_tool_receipt = True
            break
        if not matching_tool_receipt:
            _fail("tool trace lacks a success record bound to the exact receipt")

        final_result = _json_object(
            source_bytes["final_result"],
            label="final result",
        )
        final_job = final_result.get("job")
        if (
            final_result.get("task_success") is not True
            or final_result.get("official_success_source") != RAW_SUCCESS_SOURCE
            or final_result.get("runtime_cleanup") != "complete"
            or not isinstance(final_job, dict)
            or final_job.get("job_id") != job_id
            or final_job.get("attempt_index") != attempt_index
        ):
            _fail("final result is not bound to the canonical successful attempt")

        reader.verify_stable()
        files = reader.metadata()
        manifest_binding = {
            "schema_version": 2,
            "task_name": identity.task_spec.task_name,
            "task_index": identity.task_spec.task_index,
            "activity_definition_id": identity.task_spec.activity_definition_id,
            "activity_instance_id": identity.native_instance,
            "source_public_seed": identity.public_seed,
            "source": PUBLICATION_SOURCE,
            "source_tag": identity.tag,
            "mapping_version": identity.task_spec.mapping_version,
            "job_id": job_id,
            "attempt_index": attempt_index,
            "bundle_id": bundle_id,
            "source_public_tool_contract_version": source_tool_contract_version,
            "source_public_primitives": list(source_public_tools),
            "source_session_manifest_schema_version": session["schema_version"],
            "source_run_manifest_schema_version": run_manifest["schema_version"],
            "amendment_sha256": hashlib.sha256(amendment_bytes).hexdigest(),
            "provenance_sha256": provenance_sha256,
            "recipe_sha256": recipe_sha256,
            "memory_sha256": memory_sha256,
            "global_memory_snapshot_sha256": global_memory_sha256,
            "source_artifacts_sha256": dict(source_hashes),
        }
        return ValidatedBehaviorPublication(
            root=reader.root,
            identity=identity,
            bundle_id=bundle_id,
            recipe_bytes=recipe_bytes,
            memory_bytes=memory_bytes,
            provenance_bytes=provenance_bytes,
            amendment=amendment,
            provenance=provenance,
            files=files,
            manifest_binding=manifest_binding,
        )


__all__ = [
    "BehaviorPublicationIdentity",
    "ForensicPublicationValidation",
    "PublicationValidationError",
    "ValidatedBehaviorPublication",
    "canonical_bundle_id",
    "resolve_publication_identity",
    "validate_canonical_publication_root",
    "validate_forensic_publication_binding",
]
