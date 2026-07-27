"""EnvSpec lifecycle hooks for the BEHAVIOR/R1Pro plugin."""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from robots.behavior.dashboard_sink import FileDashboardSink
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.memory_snapshot import (
    BehaviorMemorySnapshotError,
    load_behavior_memory_snapshot,
)
from robots.behavior.policy_checkpoint import (
    POLICY_CHECKPOINT_BINDING_SCHEMA_VERSION,
    SHARED_POLICY_CHECKPOINT_PATH,
    SHARED_POLICY_PROFILE,
)
from robots.behavior.prompt_bundle import (
    BEHAVIOR_PHASES,
    build_prompt_context,
    system_prompt,
    user_prompt,
)
from robots.behavior.publication import (
    PublicationValidationError,
    validate_canonical_publication_root,
)
from robots.behavior.recipe_catalog import (
    BehaviorRecipeCatalogError,
    load_behavior_recipe_catalog,
)
from robots.behavior.redaction import redact_text
from robots.behavior.run_manifest import RunManifest, process_identity, redact_command
from robots.behavior.schemas import DEFAULT_ACTION_CHUNK
from robots.behavior.source_snapshot import (
    SourceSnapshotBinding,
    validate_source_snapshot,
)
from robots.behavior.spec import BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S, RunConfig
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
    BehaviorTaskSpec,
    resolve_task_spec,
)
from robots.behavior.vla_client import BehaviorVLAClient
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.logging import get_logger
from rpent.utils.socket_rpc import SocketRpcClient

logger = get_logger("behavior_runtime")
RESOURCE_POLICY = "frozen_local"

# Transitional imports for callers that have not moved to ``task_specs`` yet.
# Runtime decisions below never use these aliases.
TURNING_ON_RADIO_PUBLIC_SEED_MAP_VERSION = TURNING_ON_RADIO_TASK_SPEC.mapping_version
TURNING_ON_RADIO_CANDIDATE_MAPPING_VERSION = (
    TURNING_ON_RADIO_TASK_SPEC.candidate_mapping_version
)
TURNING_ON_RADIO_PUBLIC_SEED_TO_INSTANCE = dict(
    TURNING_ON_RADIO_TASK_SPEC.public_seed_to_instance
)
TURNING_ON_RADIO_ACTIVITY_DEFINITION_ID = (
    TURNING_ON_RADIO_TASK_SPEC.activity_definition_id
)
BEHAVIOR_NATIVE_ENV_SEED = 0

DEFAULT_MAX_EPISODE_STEPS = 24756
DEFAULT_MAX_TOOL_CALLS = 350
DEFAULT_MAX_WALL_CLOCK_S = 43200

_FROZEN_INPUT_MAX_BYTES = 2 * 1024 * 1024
_PRIOR_ATTEMPT_SUMMARIES_MAX_CHARS = 16_000
_PRIOR_ATTEMPT_SUMMARIES_MAX_ITEMS = 8
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_FORBIDDEN_PRIOR_SUMMARY_PATTERN = re.compile(
    r"(?i)(?:pixel|coord(?:inate)?s?|frame|fixed[-_ ]?hand|"
    r"activity[-_ ]?(?:definition|instance)[-_ ]?id|"
    r"native[-_ ]?(?:binding|id|instance|seed))"
)

_ACTIVE_RUN_MANIFEST: contextvars.ContextVar[RunManifest | None] = (
    contextvars.ContextVar("behavior_run_manifest", default=None)
)
_RUNTIME_NAMESPACE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RUNTIME_ISOLATION_FILENAME = "runtime_isolation.json"
_RUNTIME_ISOLATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CampaignRuntimeIsolation:
    """Private OmniGibson/Isaac writable state for one campaign arm."""

    root: Path
    namespace: str
    cuda_device: str
    binding_sha256: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {**self.payload, "binding_sha256": self.binding_sha256}

    def environment(self) -> dict[str, str]:
        paths = self.payload["paths"]
        return {
            "CUDA_VISIBLE_DEVICES": self.cuda_device,
            "OMNIGIBSON_APPDATA_PATH": str(paths["omnigibson_appdata"]),
            "XDG_CACHE_HOME": str(paths["xdg_cache"]),
            "XDG_CONFIG_HOME": str(paths["xdg_config"]),
            "XDG_DATA_HOME": str(paths["xdg_data"]),
            "OV_CACHE_DIR": str(paths["ov_cache"]),
            "OMNI_USER_FOLDER": str(paths["omni_user"]),
            "ISAAC_PATH": str(paths["isaac_root"]),
            "EXP_PATH": str(paths["experience"]),
            "TMPDIR": str(paths["tmp"]),
            "RPENT_BEHAVIOR_RUNTIME_NAMESPACE": self.namespace,
            "RPENT_BEHAVIOR_ENDPOINT_DIR": str(paths["endpoints"]),
            "RPENT_BEHAVIOR_LOG_DIR": str(paths["logs"]),
        }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _runtime_isolation_binding_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _discover_isaac_root(behavior_python: Path) -> Path:
    """Locate the Isaac Sim package without importing its native runtime."""

    command = [
        str(behavior_python),
        "-c",
        (
            "import importlib.util, pathlib; "
            "spec=importlib.util.find_spec('isaacsim'); "
            "assert spec is not None and spec.origin is not None; "
            "print(pathlib.Path(spec.origin).resolve().parent)"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot locate the Isaac Sim package root") from error
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Isaac Sim package discovery returned an invalid path")
    root = Path(lines[0]).resolve(strict=True)
    if not (root / "apps").is_dir() or not (root / "VERSION").is_file():
        raise RuntimeError(f"Isaac Sim package root is incomplete: {root}")
    return root


def _validated_behavior_python_path(
    behavior_python: str | os.PathLike[str],
) -> Path:
    """Return an absolute executable path without dereferencing its final symlink."""

    path = Path(behavior_python).expanduser().absolute()
    if not path.exists() or not path.is_file():
        raise ValueError("behavior_python must be an existing executable file")
    if not os.access(path, os.X_OK):
        raise ValueError("behavior_python must be executable")
    return path


def _write_runtime_isolation_binding(
    path: Path,
    binding: CampaignRuntimeIsolation,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json_bytes(binding.as_dict()) + b"\n")
    os.replace(temporary, path)


def _validate_runtime_isolation_payload(
    root: Path,
    value: Any,
    *,
    expected_binding_sha256: str | None = None,
) -> CampaignRuntimeIsolation:
    if not isinstance(value, dict):
        raise RuntimeError("campaign runtime isolation binding must be an object")
    recorded = value.get("binding_sha256")
    payload = {key: item for key, item in value.items() if key != "binding_sha256"}
    actual = _runtime_isolation_binding_sha256(payload)
    if (
        payload.get("schema_version") != _RUNTIME_ISOLATION_SCHEMA_VERSION
        or payload.get("kind") != "behavior_campaign_runtime_isolation"
        or not isinstance(recorded, str)
        or actual != recorded
        or (expected_binding_sha256 is not None and recorded != expected_binding_sha256)
    ):
        raise RuntimeError("campaign runtime isolation binding SHA-256 mismatch")
    namespace = payload.get("namespace")
    cuda_device = payload.get("cuda_device")
    paths = payload.get("paths")
    if (
        not isinstance(namespace, str)
        or _RUNTIME_NAMESPACE_PATTERN.fullmatch(namespace) is None
        or not isinstance(cuda_device, str)
        or re.fullmatch(r"[0-9]+", cuda_device) is None
        or not isinstance(paths, dict)
    ):
        raise RuntimeError("campaign runtime isolation identity is invalid")
    required_directories = {
        "omnigibson_appdata",
        "xdg_cache",
        "xdg_config",
        "xdg_data",
        "ov_cache",
        "omni_user",
        "isaac_root",
        "experience",
        "tmp",
        "endpoints",
        "logs",
    }
    if not required_directories.issubset(paths):
        raise RuntimeError("campaign runtime isolation paths are incomplete")
    for name in required_directories:
        path = Path(str(paths[name])).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"campaign runtime isolation path escapes root: {name}"
            ) from error
        if not path.is_dir():
            raise RuntimeError(f"campaign runtime isolation directory missing: {name}")
    return CampaignRuntimeIsolation(
        root=root,
        namespace=namespace,
        cuda_device=cuda_device,
        binding_sha256=recorded,
        payload=payload,
    )


def validate_campaign_runtime_isolation(
    runtime_root: str | os.PathLike[str],
    expected_binding_sha256: str | None = None,
) -> CampaignRuntimeIsolation:
    """Validate an already prepared campaign arm's private runtime layout."""

    root = Path(runtime_root).expanduser().resolve(strict=True)
    binding_path = root / _RUNTIME_ISOLATION_FILENAME
    try:
        value = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "campaign runtime isolation binding is unreadable"
        ) from error
    return _validate_runtime_isolation_payload(
        root,
        value,
        expected_binding_sha256=expected_binding_sha256,
    )


def prepare_campaign_runtime_isolation(
    runtime_root: str | os.PathLike[str],
    namespace: str,
    cuda_device: str,
    *,
    behavior_python: str | os.PathLike[str] | None = None,
    isaac_root: str | os.PathLike[str] | None = None,
) -> CampaignRuntimeIsolation:
    """Create disjoint OmniGibson/Isaac/cache/log state for one GPU arm."""

    name = str(namespace)
    device = str(cuda_device)
    if _RUNTIME_NAMESPACE_PATTERN.fullmatch(name) is None:
        raise ValueError("runtime namespace contains unsupported characters")
    if re.fullmatch(r"[0-9]+", device) is None:
        raise ValueError("cuda_device must be one decimal GPU ordinal")
    validated_behavior_python = (
        _validated_behavior_python_path(behavior_python)
        if behavior_python is not None
        else None
    )
    root = Path(runtime_root).expanduser().absolute()
    marker = root / _RUNTIME_ISOLATION_FILENAME
    if marker.exists():
        existing = validate_campaign_runtime_isolation(root)
        if existing.namespace != name or existing.cuda_device != device:
            raise RuntimeError(
                "existing campaign runtime isolation has a different identity"
            )
        return existing
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("campaign runtime isolation root must be new or empty")

    if isaac_root is None:
        if validated_behavior_python is None:
            raise ValueError("behavior_python or isaac_root is required")
        source_isaac = _discover_isaac_root(validated_behavior_python)
    else:
        source_isaac = Path(isaac_root).expanduser().resolve(strict=True)
    if not (source_isaac / "apps").is_dir() or not (source_isaac / "VERSION").is_file():
        raise RuntimeError(f"Isaac Sim package root is incomplete: {source_isaac}")

    root.mkdir(parents=True, exist_ok=True)
    private_isaac = root / "isaac"
    private_isaac.mkdir()
    try:
        for entry in source_isaac.iterdir():
            if entry.name == "apps":
                continue
            os.symlink(
                entry, private_isaac / entry.name, target_is_directory=entry.is_dir()
            )
        shutil.copytree(source_isaac / "apps", private_isaac / "apps")
        paths = {
            "omnigibson_appdata": root / "omnigibson_appdata",
            "xdg_cache": root / "xdg" / "cache",
            "xdg_config": root / "xdg" / "config",
            "xdg_data": root / "xdg" / "data",
            "ov_cache": root / "ov_cache",
            "omni_user": root / "omni_user",
            "isaac_root": private_isaac,
            "experience": private_isaac / "apps",
            "tmp": root / "tmp",
            "endpoints": root / "endpoints",
            "logs": root / "logs",
        }
        for path in paths.values():
            if path == private_isaac or path == private_isaac / "apps":
                continue
            path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _RUNTIME_ISOLATION_SCHEMA_VERSION,
            "kind": "behavior_campaign_runtime_isolation",
            "namespace": name,
            "cuda_device": device,
            "paths": {key: str(path.resolve()) for key, path in paths.items()},
        }
        binding = CampaignRuntimeIsolation(
            root=root.resolve(),
            namespace=name,
            cuda_device=device,
            binding_sha256=_runtime_isolation_binding_sha256(payload),
            payload=payload,
        )
        _write_runtime_isolation_binding(marker, binding)
        return validate_campaign_runtime_isolation(root, binding.binding_sha256)
    except BaseException:
        # This root was required to be new/empty, so cleanup cannot remove user
        # data or another campaign's state.
        shutil.rmtree(root, ignore_errors=True)
        raise


def _git_worktree_dirty(repo_root: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=normal",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "formal BEHAVIOR source is neither a clean Git checkout nor "
            "an explicit validated source snapshot"
        ) from error
    return bool(completed.stdout)


def _validate_runtime_source_identity(
    args: argparse.Namespace,
) -> SourceSnapshotBinding | None:
    root_value = getattr(args, "behavior_source_snapshot_root", None)
    sha_value = getattr(args, "behavior_source_snapshot_binding_sha256", None)
    if bool(root_value) != bool(sha_value):
        raise ValueError(
            "--behavior-source-snapshot-root and "
            "--behavior-source-snapshot-binding-sha256 must be supplied together"
        )
    repo_root = get_repo_root().resolve()
    if root_value:
        binding = validate_source_snapshot(str(root_value), str(sha_value))
        if binding.snapshot_root != repo_root:
            raise ValueError(
                "RPENT_REPO_ROOT must equal the validated source snapshot root"
            )
        runtime_module = Path(__file__).resolve()
        try:
            runtime_module.relative_to(binding.snapshot_root)
        except ValueError as error:
            raise ValueError(
                "robots.behavior.runtime was not imported from the source snapshot"
            ) from error
        args._behavior_source_snapshot_binding = binding
        return binding
    if (
        str(getattr(args, "behavior_phase", "")) == "eval"
        and (
            getattr(args, "behavior_job_id", None) is not None
            or getattr(args, "behavior_policy_checkpoint_binding_file", None)
            is not None
        )
        and _git_worktree_dirty(repo_root)
    ):
        raise ValueError(
            "formal BEHAVIOR Eval refuses a dirty worktree; run from an explicit "
            "hash-sealed source snapshot"
        )
    args._behavior_source_snapshot_binding = None
    return None


def _task_spec_binding(spec: BehaviorTaskSpec) -> dict[str, Any]:
    """Return the immutable task identity recorded by every runtime artifact."""

    return {
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


def _expected_shared_policy_checkpoint_binding() -> dict[str, Any]:
    """Return the immutable shared binding without re-hashing model files."""

    resolved_path = str(SHARED_POLICY_CHECKPOINT_PATH.resolve(strict=True))
    payload = {
        "schema_version": POLICY_CHECKPOINT_BINDING_SCHEMA_VERSION,
        "profile_id": SHARED_POLICY_PROFILE.profile_id,
        "resolved_path": resolved_path,
        "files": {
            item.relative_path: {
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
            }
            for item in SHARED_POLICY_PROFILE.files
        },
    }
    return {
        **payload,
        "binding_sha256": hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _read_prevalidated_policy_checkpoint_binding(
    path_value: str,
) -> dict[str, Any]:
    """Read the serial Job's once-validated checkpoint identity."""

    raw, _metadata = _read_regular_file(
        path_value,
        label="prevalidated policy checkpoint binding",
    )
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "prevalidated policy checkpoint binding must be strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("prevalidated policy checkpoint binding must be a JSON object")
    expected = _expected_shared_policy_checkpoint_binding()
    if value != expected:
        raise ValueError(
            "prevalidated policy checkpoint binding does not match the shared policy"
        )
    return value


def _behavior_resource_root(args: argparse.Namespace) -> Path:
    """Return the already prepared, immutable BEHAVIOR resource subtree."""

    value = getattr(args, "_behavior_resource_root", None)
    source = getattr(args, "_behavior_resource_source", None)
    if value is None or not isinstance(source, dict):
        raise ValueError(
            "BEHAVIOR resources must be prepared before configuration parsing"
        )
    root = Path(value).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("prepared BEHAVIOR resource root must be a directory")
    return root


def _read_reviewed_repo_memory(args: argparse.Namespace) -> None:
    """Freeze the reviewed external Global Memory before Agent startup."""

    try:
        snapshot = load_behavior_memory_snapshot(
            _behavior_resource_root(args) / "memory"
        )
    except BehaviorMemorySnapshotError as error:
        raise ValueError(f"invalid reviewed BEHAVIOR Global Memory: {error}") from error
    expected = getattr(args, "behavior_reviewed_memory_snapshot_sha256", None)
    if expected is not None and str(expected).lower() != snapshot.snapshot_sha256:
        raise ValueError("reviewed BEHAVIOR Global Memory snapshot SHA256 mismatch")
    try:
        selection = snapshot.select_task(args.task_name)
    except BehaviorMemorySnapshotError as error:
        raise ValueError(f"invalid reviewed BEHAVIOR task memory: {error}") from error
    args._behavior_target_prior = selection.target_prior_text
    args._behavior_explore_experience = selection.explore_experience_text
    args._behavior_additional_expert_knowledge = (
        selection.additional_expert_knowledge_text
    )
    # Retained as the task-scoped combined view for callers that audit the
    # selected input without rendering a prompt.
    args._behavior_repo_memory_contents = selection.prompt_text
    args._behavior_repo_memory_input = {
        "snapshot_sha256": snapshot.snapshot_sha256,
        "manifest": asdict(snapshot.manifest_binding),
        "files": {name: asdict(metadata) for name, metadata in snapshot.files.items()},
        "selection": selection.public_binding,
    }
    # The Agent sees only its task-scoped selection. The complete closed
    # snapshot and dataset source remain in the run artifacts for audit.
    args._behavior_repo_memory_prompt_input = selection.public_binding


def _read_reviewed_recipe_catalog(args: argparse.Namespace) -> None:
    """Freeze and select reviewed Recipe priors before Agent startup."""

    catalog_root = _behavior_resource_root(args) / "recipes"
    catalog = load_behavior_recipe_catalog(catalog_root)
    expected = getattr(args, "behavior_recipe_catalog_sha256", None)
    if expected is not None and str(expected).lower() != catalog.catalog_sha256:
        raise BehaviorRecipeCatalogError(
            "reviewed BEHAVIOR Recipe Catalog SHA256 mismatch"
        )
    consumer = "explore" if args.behavior_phase == "explore" else "formal_eval"
    selection = catalog.select(args.task_name, consumer)
    prompt_text = str(selection.prompt_text).strip()
    if not prompt_text:
        prompt_text = (
            "No reviewed Recipe Catalog prior is selected for this task and consumer."
        )
    args._behavior_recipe_priors = prompt_text
    args._behavior_recipe_catalog_input = {
        "catalog_sha256": catalog.catalog_sha256,
        "selection": selection.public_binding,
    }
    args._behavior_recipe_selected_ids = list(selection.selected_ids)


def _read_regular_file(path_value: str, *, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read one bounded regular file without following a terminal symlink."""

    path = Path(path_value).expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"{label} must be a readable non-symlink regular file"
        ) from error
    try:
        metadata_before = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata_before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        if metadata_before.st_size > _FROZEN_INPUT_MAX_BYTES:
            raise ValueError(f"{label} exceeds {_FROZEN_INPUT_MAX_BYTES} bytes")
        chunks: list[bytes] = []
        bytes_remaining = _FROZEN_INPUT_MAX_BYTES + 1
        while bytes_remaining:
            chunk = os.read(file_descriptor, min(64 * 1024, bytes_remaining))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_remaining -= len(chunk)
        raw = b"".join(chunks)
        metadata_after = os.fstat(file_descriptor)
    finally:
        os.close(file_descriptor)
    if len(raw) > _FROZEN_INPUT_MAX_BYTES:
        raise ValueError(f"{label} exceeds {_FROZEN_INPUT_MAX_BYTES} bytes")
    stable_fields_before = (
        metadata_before.st_dev,
        metadata_before.st_ino,
        metadata_before.st_size,
        metadata_before.st_mtime_ns,
        metadata_before.st_ctime_ns,
    )
    stable_fields_after = (
        metadata_after.st_dev,
        metadata_after.st_ino,
        metadata_after.st_size,
        metadata_after.st_mtime_ns,
        metadata_after.st_ctime_ns,
    )
    if (
        len(raw) != metadata_before.st_size
        or stable_fields_before != stable_fields_after
    ):
        raise ValueError(f"{label} changed while being read")
    return raw, {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _sanitize_frozen_text(raw: bytes, *, label: str) -> str:
    """Return strict UTF-8 prompt text with native-binding lines removed."""

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be strict UTF-8 text") from error
    if "\x00" in text:
        raise ValueError(f"{label} must not contain NUL bytes")
    native_markers = (
        "activity_instance_id",
        "activity-instance-id",
        "activity_definition_id",
        "activity-definition-id",
        "native_instance",
        "native binding",
    )
    lines = [
        line
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if not any(marker in line.lower() for marker in native_markers)
    ]
    return "\n".join(lines).strip()


def _read_frozen_eval_inputs(args: argparse.Namespace) -> None:
    """Validate one complete Explore Job publication for frozen Eval prompts."""

    try:
        publication = validate_canonical_publication_root(
            args.behavior_frozen_publication_root,
            expected_provenance_sha256=args.behavior_frozen_provenance_sha256,
            task_name=args.task_name,
            task_index=int(args.task),
        )
    except PublicationValidationError as error:
        raise ValueError(f"invalid frozen BEHAVIOR publication: {error}") from error
    task_spec = getattr(args, "_behavior_task_spec", None)
    if task_spec is None:
        task_spec = resolve_task_spec(
            task_name=str(args.task_name),
            task_index=int(args.task),
        )
    source_public_seed = publication.identity.public_seed
    source_tag = publication.identity.tag
    if source_public_seed not in task_spec.explore_public_seeds:
        raise ValueError(
            "frozen BEHAVIOR publication source is outside the selected task's "
            "Explore partition"
        )
    if (
        publication.manifest_binding.get("source_public_seed") != source_public_seed
        or publication.manifest_binding.get("source_tag") != source_tag
        or source_tag != task_spec.tag(source_public_seed)
    ):
        raise ValueError("frozen BEHAVIOR publication source identity is inconsistent")
    args._behavior_frozen_contents = {
        "recipe": _sanitize_frozen_text(
            publication.recipe_bytes, label="frozen recipe"
        ),
        "memory": _sanitize_frozen_text(
            publication.memory_bytes, label="frozen memory"
        ),
    }
    args._behavior_frozen_inputs = {
        **publication.manifest_binding,
        "files": publication.files,
    }
    args._behavior_frozen_source_public_seed = source_public_seed
    args._behavior_frozen_source_tag = source_tag


def _read_prior_attempt_summaries(args: argparse.Namespace) -> None:
    """Read the job-owned, bounded summaries passed to one fresh child."""

    raw, metadata = _read_regular_file(
        args.behavior_prior_attempt_summaries_file,
        label="prior attempt summaries",
    )
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prior attempt summaries must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("prior attempt summaries must be a JSON object")
    if payload.get("job_id") != args.behavior_job_id:
        raise ValueError("prior attempt summaries job binding mismatch")
    if payload.get("next_attempt_index") != args.behavior_attempt_index:
        raise ValueError("prior attempt summaries attempt binding mismatch")
    lineage_scope = payload.get("lineage_scope", "same_job_prior")
    if lineage_scope not in {"same_job_prior", "campaign_prior"}:
        raise ValueError("prior attempt summaries lineage scope is invalid")
    if lineage_scope == "campaign_prior" and args.behavior_attempt_index != 1:
        raise ValueError("campaign prior summaries are valid only for attempt 1")
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("prior attempt summaries must contain a summaries list")
    if len(summaries) > _PRIOR_ATTEMPT_SUMMARIES_MAX_ITEMS:
        raise ValueError("prior attempt summaries may contain at most 8 entries")
    rendered: list[str] = []
    previous_index = 0
    for item in summaries:
        if not isinstance(item, dict):
            raise ValueError("each prior attempt summary must be an object")
        index = item.get("attempt_index")
        outcome = item.get("outcome")
        summary = item.get("summary")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index <= previous_index
            or (
                lineage_scope == "same_job_prior"
                and index >= args.behavior_attempt_index
            )
        ):
            raise ValueError("prior attempt summary indices must be ordered and prior")
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("prior attempt summary outcome must be non-empty")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("prior attempt summary text must be non-empty")
        label = "Campaign evidence" if lineage_scope == "campaign_prior" else "Attempt"
        line = f"{label} {index} ({outcome.strip()}): {summary.strip()}"
        if _FORBIDDEN_PRIOR_SUMMARY_PATTERN.search(line):
            raise ValueError("prior attempt summaries contain disallowed spatial data")
        rendered.append(line)
        previous_index = index
    text = "\n".join(rendered)
    if len(text) > _PRIOR_ATTEMPT_SUMMARIES_MAX_CHARS:
        raise ValueError("prior attempt summaries exceed 16000 characters")
    args._behavior_prior_attempt_summaries = text or None
    args._behavior_prior_attempt_summaries_input = {
        **metadata,
        "count": len(summaries),
        "lineage_scope": lineage_scope,
        "max_count": _PRIOR_ATTEMPT_SUMMARIES_MAX_ITEMS,
        "max_chars": _PRIOR_ATTEMPT_SUMMARIES_MAX_CHARS,
    }


@contextlib.contextmanager
def _defer_termination_signals():
    """Make child spawn plus manifest registration one interrupt-safe section."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    deferred: list[int] = []
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def defer(signum, _frame):
        deferred.append(int(signum))

    for signum in previous:
        signal.signal(signum, defer)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if deferred:
        signum = deferred[0]
        handler = previous[signum]
        if handler == signal.SIG_IGN:
            return
        if callable(handler):
            handler(signum, None)
            return
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _env_rpc_client(*, host: str, port: int):
    """Build the current BEHAVIOR env transport without endpoint side files."""

    if not host:
        raise ValueError("BEHAVIOR env host must be non-empty")
    if int(port) <= 0:
        raise ValueError("BEHAVIOR env port must be positive")
    return SocketRpcClient(str(host), int(port))


def _external_env_rpc_client(args: argparse.Namespace):
    """Build an explicit external env client.

    The legacy BEHAVIOR CLI represents the socket host and port separately.
    A protocol URL is accepted for migration convenience, but HTTP is not
    silently selected from a bare host.
    """

    endpoint = str(args.env_endpoint)
    if "://" not in endpoint:
        return _env_rpc_client(host=endpoint, port=int(args.env_port))
    protocol, _, address = endpoint.partition("://")
    host, separator, port_text = address.rpartition(":")
    if not separator or not host:
        raise ValueError(
            "--env-endpoint must be a host with --env-port or protocol://host:port"
        )
    port = int(port_text)
    if protocol == "socket":
        return _env_rpc_client(host=host, port=port)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}")
    raise ValueError("--env-endpoint protocol must be socket or http")


def _managed_env_rpc_client(proc: subprocess.Popen):
    host = getattr(proc, "_rpent_transport_host", None)
    port = getattr(proc, "_rpent_transport_port", None)
    if not isinstance(host, str) or not isinstance(port, int):
        raise RuntimeError("managed BEHAVIOR env transport metadata is unavailable")
    return _env_rpc_client(host=host, port=port)


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return "<log unavailable>"


def _record_owned_process_group(proc: subprocess.Popen) -> int | None:
    """Mark the dedicated process group created for this exact Popen handle."""

    pid, pgid = process_identity(proc)
    if pid is None or pgid != pid or pgid == os.getpgrp():
        return None
    proc._rpent_owned_pgid = pgid
    try:
        proc._rpent_owned_sid = os.getsid(pid)
    except OSError:
        proc._rpent_owned_sid = None
    return pgid


def _owned_process_group_alive(proc: subprocess.Popen) -> bool:
    """Confirm a recorded dedicated PGID still has members in its session."""
    pgid = getattr(proc, "_rpent_owned_pgid", None)
    sid = getattr(proc, "_rpent_owned_sid", None)
    if not isinstance(pgid, int) or not isinstance(sid, int):
        return False
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            process_group = int(fields[2])
            session = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
        if process_group == pgid and session == sid:
            return True
    return False


def _terminate_process(proc: subprocess.Popen | None, *, timeout: float = 15.0) -> None:
    """Terminate an owned subprocess and its dedicated process group."""

    if proc is None:
        return
    pid, pgid = process_identity(proc)
    recorded_pgid = getattr(proc, "_rpent_owned_pgid", None)
    group_alive = _owned_process_group_alive(proc)
    leader_alive = proc.poll() is None
    owns_process_group = (
        isinstance(recorded_pgid, int)
        and recorded_pgid != os.getpgrp()
        and (
            (leader_alive and pgid == recorded_pgid)
            or (not leader_alive and group_alive and pgid in {None, recorded_pgid})
        )
    )
    if owns_process_group:
        try:
            os.killpg(recorded_pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        if proc.poll() is not None:
            return
        if pid is not None:
            logger.warning(
                "refusing group signal for pid=%s pgid=%s; using direct terminate",
                pid,
                pgid,
            )
        proc.terminate()
    deadline = time.monotonic() + float(timeout)
    if owns_process_group:
        while _owned_process_group_alive(proc) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _owned_process_group_alive(proc):
            try:
                os.killpg(recorded_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    else:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)


def _runtime_env(args: argparse.Namespace) -> dict[str, str]:
    repo_root = get_repo_root()
    behavior_root = Path(args.behavior_repo).expanduser().resolve()
    env = os.environ.copy()
    env["RPENT_REPO_ROOT"] = str(repo_root)
    env["RPENT_RLINF_ROOT"] = str(behavior_root)
    env["RLINF_REPO_PATH"] = str(behavior_root)
    if getattr(args, "policy_checkpoint", None):
        env["PI05_CHECKPOINT_PATH"] = str(Path(args.policy_checkpoint).resolve())
    if args.cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    isolation = getattr(args, "_behavior_runtime_isolation", None)
    if isolation is not None:
        if not isinstance(isolation, CampaignRuntimeIsolation):
            raise RuntimeError("invalid BEHAVIOR campaign runtime isolation")
        if isolation.cuda_device != str(args.cuda_device):
            raise RuntimeError("runtime isolation GPU does not match --cuda-device")
        env.update(isolation.environment())
    env["RPENT_BEHAVIOR_CONTROLLER_MODE"] = str(
        getattr(args, "behavior_controller_mode", "hybrid")
    )
    env.setdefault("OMNIGIBSON_HEADLESS", "1")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    existing = env.get("PYTHONPATH")
    parts = [str(repo_root), str(behavior_root)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _pump_ready_events(
    proc: subprocess.Popen,
    log_path: Path,
    events: "queue.Queue[dict[str, Any]]",
) -> None:
    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict) and event.get("event") == "transport_ready":
                events.put(event)


def start_env_server(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> subprocess.Popen:
    script = get_repo_root() / "robots" / "behavior" / "env_server.py"
    command = [
        str(Path(args.behavior_python).expanduser().absolute()),
        str(script),
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--task-name",
        args.task_name,
        "--activity-definition-id",
        str(args.activity_definition_id),
        "--activity-instance-id",
        str(args.activity_instance_id),
        "--activity-instance-dir",
        str(Path(args.activity_instance_dir).resolve()),
        "--scene-model",
        args.scene_model,
        "--seed",
        str(args.seed),
        "--public-seed",
        str(args.public_seed),
        "--attempt-index",
        str(int(getattr(args, "behavior_attempt_index", 1) or 1)),
        "--controller-mode",
        str(getattr(args, "behavior_controller_mode", "hybrid")),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--output-dir",
        str(output_dir),
        "--transport-host",
        "127.0.0.1",
        "--transport-port",
        "0",
    ]
    if args.behavior_config:
        command.extend(["--config-path", str(Path(args.behavior_config).resolve())])
    log_path = output_dir / "env_server.log"
    logger.info(
        "BEHAVIOR env server cmd: %s",
        shlex.join(redact_command(command) or []),
    )
    manifest = _ACTIVE_RUN_MANIFEST.get()
    proc = None
    try:
        with _defer_termination_signals():
            proc = subprocess.Popen(
                command,
                cwd=Path(args.behavior_repo).resolve(),
                env=_runtime_env(args),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            _record_owned_process_group(proc)
            if manifest is not None:
                manifest.process_started("env", proc, command=command)
    except BaseException:
        _terminate_process(proc)
        if manifest is not None:
            manifest.process_stopped("env", proc)
        raise
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    threading.Thread(
        target=_pump_ready_events,
        args=(proc, log_path, events),
        daemon=True,
    ).start()
    deadline = time.time() + args.env_ready_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            if manifest is not None:
                manifest.process_stopped("env", proc)
            raise RuntimeError(
                "BEHAVIOR env server exited before ready:\n" + _tail(log_path)
            )
        try:
            event = events.get(timeout=2.0)
        except queue.Empty:
            continue
        if event.get("kind") == "socket" and event.get("host") and event.get("port"):
            proc._rpent_transport_host = str(event["host"])
            proc._rpent_transport_port = int(event["port"])
            if manifest is not None:
                try:
                    manifest.process_endpoint(
                        "env", host=str(event["host"]), port=int(event["port"])
                    )
                except BaseException:
                    _terminate_process(proc)
                    raise
            logger.info(
                "BEHAVIOR env server ready at %s:%s", event["host"], event["port"]
            )
            return proc
    _terminate_process(proc)
    if manifest is not None:
        manifest.process_stopped("env", proc)
    raise TimeoutError(
        f"BEHAVIOR env server not ready after {args.env_ready_timeout_s}s:\n"
        + _tail(log_path)
    )


def stop_env_server(proc: subprocess.Popen | None, *, output_dir: Path) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            _managed_env_rpc_client(proc).call("shutdown", timeout_s=30.0)
        except Exception:
            logger.exception("BEHAVIOR env graceful shutdown failed")
        try:
            proc.wait(timeout=120.0)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
    # The Isaac/OG leader can exit before one of its descendants.  The exact
    # PGID+SID captured at launch remains the ownership boundary in that case.
    if _owned_process_group_alive(proc):
        _terminate_process(proc)
    returncode = proc.poll()
    if returncode not in {None, 0}:
        raise RuntimeError(
            "BEHAVIOR env server did not exit cleanly "
            f"(returncode={returncode}):\n" + _tail(Path(output_dir) / "env_server.log")
        )


def start_vla_server(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[str, subprocess.Popen]:
    host = "127.0.0.1"
    port = args.vla_port or _free_port(host)
    script = get_repo_root() / "robots" / "behavior" / "vla_server.py"
    command = [
        str(Path(args.behavior_python).expanduser().absolute()),
        str(script),
        "--host",
        host,
        "--port",
        str(port),
        "--checkpoint",
        str(Path(args.policy_checkpoint).resolve()),
        "--seed",
        str(args.seed),
    ]
    log_path = output_dir / "vla_server.log"
    logger.info(
        "BEHAVIOR VLA server cmd: %s",
        shlex.join(redact_command(command) or []),
    )
    manifest = _ACTIVE_RUN_MANIFEST.get()
    proc = None
    try:
        with log_path.open("a", encoding="utf-8") as log:
            with _defer_termination_signals():
                proc = subprocess.Popen(
                    command,
                    cwd=Path(args.behavior_repo).resolve(),
                    env=_runtime_env(args),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _record_owned_process_group(proc)
                if manifest is not None:
                    manifest.process_started(
                        "vla", proc, command=command, host=host, port=int(port)
                    )
    except BaseException:
        _terminate_process(proc)
        if manifest is not None:
            manifest.process_stopped("vla", proc)
        raise
    base_url = f"http://{host}:{port}"
    client = None
    ready = False
    deadline = time.time() + args.vla_ready_timeout_s
    try:
        client = BehaviorVLAClient(base_url)
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    "BEHAVIOR VLA server exited before ready:\n" + _tail(log_path)
                )
            try:
                metadata = client.healthz(
                    timeout_ms=2000,
                    expected_checkpoint_binding=(
                        args._behavior_policy_checkpoint_binding
                    ),
                )
                if metadata.get("config_name") != "pi05_behavior":
                    raise RuntimeError(f"unexpected VLA metadata: {metadata!r}")
                logger.info("BEHAVIOR VLA server ready at %s", base_url)
                ready = True
                break
            except Exception:
                time.sleep(2.0)
        if not ready:
            raise TimeoutError(
                f"BEHAVIOR VLA server not ready after {args.vla_ready_timeout_s}s:\n"
                + _tail(log_path)
            )
    except BaseException:
        _terminate_process(proc)
        if manifest is not None:
            manifest.process_stopped("vla", proc)
        raise
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except BaseException:
                if ready:
                    _terminate_process(proc)
                    if manifest is not None:
                        manifest.process_stopped("vla", proc)
                    raise
                logger.exception("BEHAVIOR temporary VLA health client close failed")
    return base_url, proc


@dataclass
class BehaviorRuntimeResources:
    """Only the runtime resources owned by one EnvSpec initialization."""

    output_dir: Path
    toolkit: Any = None
    env_proc: subprocess.Popen | None = None
    vla_proc: subprocess.Popen | None = None
    manifest: RunManifest | None = None
    env_rpc_client: Any = None
    command_arbiter: Any = None
    success_latch: Any = None
    dashboard_controller: Any = None
    dashboard_state: Any = None
    _closed: bool = False
    _control_closed: bool = False
    _control_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )

    def attach_dashboard_control(
        self,
        *,
        toolkit: Any,
        controller: Any,
        dashboard_state: Any,
    ) -> None:
        """Bind the one runtime-owned manual controller exactly once."""

        with self._control_lock:
            if self._closed or self._control_closed:
                raise RuntimeError("BEHAVIOR runtime control is already closed")
            if self.dashboard_controller is not None:
                if (
                    self.dashboard_controller is controller
                    and self.toolkit is toolkit
                    and self.dashboard_state is dashboard_state
                ):
                    return
                raise RuntimeError("BEHAVIOR runtime control is already bound")
            self.toolkit = toolkit
            self.dashboard_controller = controller
            self.dashboard_state = dashboard_state

    def quiesce_control(
        self,
        *,
        timeout_s: float = BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S,
    ) -> None:
        """Stop admission, drain the current safe step, then unbind control."""

        with self._control_lock:
            if self._control_closed:
                return
            controller = self.dashboard_controller
            arbiter = self.command_arbiter
            dashboard = self.dashboard_state

            for owner in (controller, arbiter):
                quiesce = getattr(owner, "quiesce", None)
                if callable(quiesce):
                    quiesce()

            deadline = time.monotonic() + max(0.0, float(timeout_s))
            for label, owner in (("controller", controller), ("arbiter", arbiter)):
                drain = getattr(owner, "drain", None)
                if not callable(drain):
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                if drain(remaining) is not True:
                    raise TimeoutError(
                        f"BEHAVIOR dashboard {label} did not drain before cleanup"
                    )

            unbind = getattr(dashboard, "unbind_controller", None)
            if callable(unbind) and controller is not None:
                unbind(controller)

            close_controller = getattr(controller, "close", None)
            if callable(close_controller):
                close_controller(timeout_s=max(0.0, deadline - time.monotonic()))
            close_arbiter = getattr(arbiter, "close", None)
            if callable(close_arbiter):
                close_arbiter(timeout_s=max(0.0, deadline - time.monotonic()))
            self._control_closed = True

    def runner_continuation_state(self) -> dict[str, Any]:
        """Combine toolkit facts with owned child-process liveness."""

        if self.toolkit is None:
            snapshot: dict[str, Any] = {}
        else:
            snapshot = self.toolkit.runner_continuation_state()
        failures: list[dict[str, Any]] = []
        for name, process in (("env", self.env_proc), ("vla", self.vla_proc)):
            if process is None:
                continue
            returncode = process.poll()
            if returncode is not None:
                failures.append(
                    {
                        "process": name,
                        "pid": getattr(process, "pid", None),
                        "returncode": int(returncode),
                    }
                )
        result = dict(snapshot)
        result["unrecoverable_infrastructure_termination"] = bool(
            result.get("unrecoverable_infrastructure_termination") is True
            or failures
            or self._closed
        )
        result["infrastructure_failures"] = failures
        return result

    def close(self) -> None:
        """Single-flight release of controller, transports, and processes."""

        with self._control_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._closed:
            return
        # Never tear down the simulator transport while one validated
        # discrete CuRobo step is still executing.
        self.quiesce_control()
        self._closed = True
        failure: BaseException | None = None
        if self.manifest is not None:
            try:
                stopping = getattr(self.manifest, "stopping", None)
                if callable(stopping):
                    stopping()
            except BaseException as error:
                failure = error
        try:
            stop_env_server(self.env_proc, output_dir=self.output_dir)
        except BaseException as error:
            if failure is None:
                failure = error
        finally:
            if self.manifest is not None:
                try:
                    self.manifest.process_stopped("env", self.env_proc)
                except BaseException as error:
                    if failure is None:
                        failure = error
        try:
            _terminate_process(self.vla_proc)
        except BaseException as error:
            if failure is None:
                failure = error
        finally:
            if self.manifest is not None and "vla" in self.manifest.data["processes"]:
                try:
                    self.manifest.process_stopped("vla", self.vla_proc)
                except BaseException as error:
                    if failure is None:
                        failure = error
        client_close = getattr(self.env_rpc_client, "close", None)
        if callable(client_close):
            try:
                client_close()
            except BaseException as error:
                if failure is None:
                    failure = error
        if self.manifest is not None:
            try:
                self.manifest.finish(
                    status="failed" if failure is not None else "stopped",
                    error=failure,
                )
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure

    def stop(self) -> None:
        """Release owned resources through the shared RuntimeResource API."""

        self.close()


class _BehaviorConfig:
    """Task-aware validation and derivation used by the EnvSpec hooks."""

    name = "behavior"
    default_planner = "codex"

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--suite", default="behavior_2025_challenge")
        parser.add_argument("--task", type=int, default=0)
        parser.add_argument("--task-name", default="turning_on_radio")
        parser.add_argument("--activity-definition-id", type=int, default=0)
        parser.add_argument("--activity-instance-id", type=int, default=None)
        parser.add_argument("--activity-instance-dir", default=None)
        parser.add_argument("--scene-model", default="house_double_floor_lower")
        parser.add_argument(
            "--public-seed",
            type=int,
            default=1,
            help=(
                "Task-local public seed; TaskSpec defines the disjoint Explore "
                "and formal Eval partitions."
            ),
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=BEHAVIOR_NATIVE_ENV_SEED,
            help="Native simulator seed. Formal BEHAVIOR runs require 0.",
        )
        parser.add_argument(
            "--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS
        )
        parser.add_argument(
            "--max-tool-calls", type=int, default=DEFAULT_MAX_TOOL_CALLS
        )
        parser.add_argument(
            "--max-wall-clock-s", type=int, default=DEFAULT_MAX_WALL_CLOCK_S
        )
        parser.add_argument("--behavior-repo", default=None)
        parser.add_argument("--behavior-python", default=None)
        parser.add_argument("--behavior-config", default=None)
        parser.add_argument(
            "--policy-checkpoint",
            default=str(SHARED_POLICY_CHECKPOINT_PATH),
            help=(
                "Shared BEHAVIOR Pi0.5 base checkpoint. Task-specific SFT "
                "checkpoints are not supported."
            ),
        )
        parser.add_argument(
            "--behavior-phase",
            choices=BEHAVIOR_PHASES,
            default="eval",
            help="Prompt/evaluation protocol. It never changes the 9-primitive surface.",
        )
        parser.add_argument("--behavior-frozen-publication-root", default=None)
        parser.add_argument("--behavior-frozen-provenance-sha256", default=None)
        parser.add_argument(
            "--behavior-reviewed-memory-snapshot-sha256",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-recipe-catalog-sha256",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-resource-revision",
            default=None,
            help=(
                "Optional pinned RLinf/RPent-memory revision used to materialize "
                "the BEHAVIOR resource snapshot (default: main)."
            ),
        )
        parser.add_argument(
            "--behavior-resource-local",
            default=None,
            help=(
                "Local reviewed BEHAVIOR resource source to seal into the "
                "versioned cache instead of resolving HuggingFace."
            ),
        )
        parser.add_argument(
            "--behavior-resource-cache",
            default=str(get_repo_root() / "resources" / ".snapshots"),
            help="Versioned local cache for pinned BEHAVIOR resources.",
        )
        parser.add_argument(
            "--behavior-resource-offline",
            action="store_true",
            default=None,
            help=(
                "Use an already materialized full-commit resource snapshot; "
                "also enabled by HF_HUB_OFFLINE=1."
            ),
        )
        parser.add_argument(
            "--behavior-resource-root",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-resource-source-file",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument("--cuda-device", default="7")
        parser.add_argument("--no-driver", action="store_true")
        parser.add_argument("--env-endpoint", default="127.0.0.1")
        parser.add_argument("--env-port", type=int, default=0)
        parser.add_argument("--vla-endpoint", default=None)
        parser.add_argument("--vla-port", type=int, default=0)
        parser.add_argument("--env-ready-timeout-s", type=int, default=1800)
        parser.add_argument("--vla-ready-timeout-s", type=int, default=1800)
        parser.add_argument(
            "--behavior-job-id",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-attempt-index",
            type=int,
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-prior-attempt-summaries-file",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-vla-binding-id",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-dashboard-event-sink",
            action="store_true",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-policy-checkpoint-binding-file",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-candidate-campaign-id",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-candidate-instance-id",
            type=int,
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-candidate-state-sha256",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-source-snapshot-root",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-source-snapshot-binding-sha256",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-runtime-isolation-root",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-runtime-namespace",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-runtime-isolation-binding-sha256",
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--behavior-controller-mode",
            choices=("hybrid", "pi0_nav_pick_only"),
            default="hybrid",
            help=argparse.SUPPRESS,
        )

    def _resolve_paths(self, args: argparse.Namespace) -> None:
        task_spec = getattr(args, "_behavior_task_spec", None)
        if task_spec is None:
            task_spec = resolve_task_spec(
                task_name=str(args.task_name),
                task_index=int(args.task),
            )
        if not args.behavior_repo:
            args.behavior_repo = os.environ.get("RPENT_RLINF_ROOT") or os.environ.get(
                "RLINF_REPO_PATH"
            )
        if not args.behavior_repo:
            args.behavior_repo = str(get_repo_root().parent / "RLinf_agentic_push")
        root = Path(args.behavior_repo).expanduser().resolve()
        args.behavior_repo = str(root)
        if not args.behavior_python:
            args.behavior_python = str(root / ".venv-behavior" / "bin" / "python")
        if not args.activity_instance_dir:
            args.activity_instance_dir = str(
                root
                / ".venv-behavior"
                / "BEHAVIOR-1K"
                / "datasets"
                / "2025-challenge-task-instances"
                / "scenes"
                / task_spec.scene_model
                / "json"
                / task_spec.state_dir_name
            )
        # Both supported tasks use one immutable general policy. Do not inherit
        # PI05_CHECKPOINT_PATH: a previous task's ambient SFT must not leak into
        # a new BEHAVIOR Job.
        if not args.policy_checkpoint:
            args.policy_checkpoint = str(SHARED_POLICY_CHECKPOINT_PATH)

    def validate_args(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> None:
        if getattr(args, "planner", None) != "codex":
            parser.error(
                "BEHAVIOR requires --planner codex so the fixed public toolkit "
                "is the only execution surface"
            )
        if args.suite != "behavior_2025_challenge":
            parser.error("formal BEHAVIOR runs require --suite behavior_2025_challenge")
        try:
            task_spec = resolve_task_spec(
                task_name=str(args.task_name),
                task_index=int(args.task),
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
        args._behavior_task_spec = task_spec
        args._behavior_task_spec_binding = _task_spec_binding(task_spec)
        self._resolve_paths(args)
        if int(args.activity_definition_id) != task_spec.activity_definition_id:
            parser.error(
                f"{task_spec.task_name} requires --activity-definition-id "
                f"{task_spec.activity_definition_id}"
            )
        if str(args.scene_model) != task_spec.scene_model:
            parser.error(
                f"{task_spec.task_name} requires --scene-model {task_spec.scene_model}"
            )
        reviewed_memory_pin = getattr(
            args, "behavior_reviewed_memory_snapshot_sha256", None
        )
        if (
            not isinstance(reviewed_memory_pin, str)
            or re.fullmatch(r"[0-9a-f]{64}", reviewed_memory_pin) is None
        ):
            parser.error(
                "formal BEHAVIOR runs require "
                "--behavior-reviewed-memory-snapshot-sha256"
            )
        try:
            _read_reviewed_repo_memory(args)
        except ValueError as error:
            parser.error(str(error))
        if int(args.seed) != BEHAVIOR_NATIVE_ENV_SEED:
            parser.error(
                "native --seed is fixed at 0; select a harness case with --public-seed"
            )
        try:
            mapped_instance = task_spec.instance_for_public_seed(
                int(args.public_seed),
                phase=args.behavior_phase,
            )
        except ValueError as error:
            parser.error(str(error))
        recipe_catalog_pin = getattr(args, "behavior_recipe_catalog_sha256", None)
        if (
            not isinstance(recipe_catalog_pin, str)
            or re.fullmatch(r"[0-9a-f]{64}", recipe_catalog_pin) is None
        ):
            parser.error(
                "BEHAVIOR Explore/Eval runs require --behavior-recipe-catalog-sha256"
            )
        try:
            _read_reviewed_recipe_catalog(args)
        except (OSError, BehaviorRecipeCatalogError) as error:
            parser.error(f"invalid reviewed BEHAVIOR Recipe Catalog: {error}")
        candidate_values = (
            args.behavior_candidate_campaign_id,
            args.behavior_candidate_instance_id,
            args.behavior_candidate_state_sha256,
        )
        candidate_requested = any(value is not None for value in candidate_values)
        if candidate_requested and not all(
            value is not None for value in candidate_values
        ):
            parser.error("candidate Explore binding arguments must be complete")
        if candidate_requested:
            if (
                args.behavior_phase != "explore"
                or int(args.public_seed) not in task_spec.explore_public_seeds
            ):
                parser.error(
                    "candidate binding is restricted to this task's Explore seed"
                )
            if (
                args.behavior_candidate_campaign_id != args.behavior_job_id
                or not _JOB_ID_PATTERN.fullmatch(
                    str(args.behavior_candidate_campaign_id)
                )
            ):
                parser.error("candidate campaign must match the owning Explore job")
            candidate_instance = int(args.behavior_candidate_instance_id)
            if candidate_instance <= 0:
                parser.error("candidate instance must be positive")
            classification = task_spec.classify_instance(candidate_instance)
            if classification.kind != "candidate":
                parser.error(
                    "candidate instance is already part of this task's public mapping"
                )
            if (
                args.activity_instance_id is not None
                and int(args.activity_instance_id) != candidate_instance
            ):
                parser.error("--activity-instance-id conflicts with candidate binding")
            if (
                re.fullmatch(r"[0-9a-f]{64}", str(args.behavior_candidate_state_sha256))
                is None
            ):
                parser.error("candidate state SHA256 is invalid")
            args.activity_instance_id = candidate_instance
            args._behavior_mapping_version = task_spec.candidate_mapping_version
            args._behavior_instance_classification = asdict(classification)
        else:
            if (
                args.activity_instance_id is not None
                and int(args.activity_instance_id) != mapped_instance
            ):
                parser.error(
                    "--activity-instance-id conflicts with the authoritative "
                    f"public-seed mapping (s{args.public_seed} -> {mapped_instance})"
                )
            args.activity_instance_id = mapped_instance
            args._behavior_mapping_version = task_spec.mapping_version
            args._behavior_instance_classification = asdict(
                task_spec.classify_instance(mapped_instance)
            )
        if str(args.cuda_device) != "7":
            parser.error("BEHAVIOR evaluation is restricted to --cuda-device 7")
        budget_values = {
            "--max-episode-steps": args.max_episode_steps,
            "--max-tool-calls": args.max_tool_calls,
            "--max-wall-clock-s": args.max_wall_clock_s,
        }
        for name, value in budget_values.items():
            if value <= 0:
                parser.error(f"{name} must be positive")
        if args.env_ready_timeout_s <= 0 or args.vla_ready_timeout_s <= 0:
            parser.error("runtime ready timeouts must be positive")
        if getattr(args, "model", None) is None:
            args.model = "gpt-5.5"
        elif str(args.model) != "gpt-5.5":
            parser.error("formal BEHAVIOR runs require --model gpt-5.5")
        if getattr(args, "reasoning_effort", None) is None:
            args.reasoning_effort = "xhigh"
        elif str(args.reasoning_effort) != "xhigh":
            parser.error("formal BEHAVIOR runs require --reasoning-effort xhigh")
        frozen_values = (
            args.behavior_frozen_publication_root,
            args.behavior_frozen_provenance_sha256,
        )
        args._behavior_frozen_contents = None
        args._behavior_frozen_inputs = None
        args._behavior_frozen_source_public_seed = None
        args._behavior_frozen_source_tag = None
        args._behavior_prior_attempt_summaries = None
        args._behavior_prior_attempt_summaries_input = None
        job_values = (
            args.behavior_job_id,
            args.behavior_attempt_index,
            args.behavior_prior_attempt_summaries_file,
            args.behavior_vla_binding_id,
        )
        dashboard_event_sink = bool(
            getattr(args, "behavior_dashboard_event_sink", False)
        )
        if dashboard_event_sink:
            if getattr(args, "dashboard", False):
                parser.error(
                    "--behavior-dashboard-event-sink cannot be combined with "
                    "--dashboard"
                )
            if args.behavior_phase == "explore" and args.behavior_job_id is None:
                parser.error(
                    "--behavior-dashboard-event-sink is restricted to a serial "
                    "Explore attempt"
                )
            if args.behavior_phase == "eval":
                if not getattr(args, "output_dir", None):
                    parser.error(
                        "formal serial Eval dashboard event sink requires --output-dir"
                    )
                if not all(value is not None for value in frozen_values):
                    parser.error(
                        "formal serial Eval dashboard event sink requires complete "
                        "frozen publication and provenance inputs"
                    )
        if args.behavior_phase == "explore":
            if any(value is not None for value in frozen_values):
                parser.error("Explore rejects all --behavior-frozen-* arguments")
            if not all(value is not None for value in job_values):
                parser.error(
                    "Explore must run under serial_explore and requires its internal "
                    "job, attempt, summary, and VLA-binding arguments"
                )
            if not _JOB_ID_PATTERN.fullmatch(str(args.behavior_job_id)):
                parser.error("--behavior-job-id has an invalid format")
            if args.behavior_attempt_index <= 0:
                parser.error("--behavior-attempt-index must be positive")
            if not _JOB_ID_PATTERN.fullmatch(str(args.behavior_vla_binding_id)):
                parser.error("--behavior-vla-binding-id has an invalid format")
            try:
                _read_prior_attempt_summaries(args)
            except (OSError, ValueError) as error:
                parser.error(str(error))
        elif not all(value is not None for value in frozen_values):
            if any(value is not None for value in job_values):
                parser.error("Eval rejects internal Explore job arguments")
            parser.error(
                "Eval requires --behavior-frozen-publication-root and "
                "--behavior-frozen-provenance-sha256"
            )
        else:
            if any(value is not None for value in job_values):
                parser.error("Eval rejects internal Explore job arguments")
            try:
                _read_frozen_eval_inputs(args)
            except (OSError, ValueError) as error:
                parser.error(str(error))
            if dashboard_event_sink and not isinstance(
                args._behavior_frozen_inputs, dict
            ):
                parser.error(
                    "formal serial Eval dashboard event sink requires validated "
                    "frozen publication inputs"
                )
        requested_checkpoint = Path(args.policy_checkpoint).expanduser()
        try:
            requested_checkpoint = requested_checkpoint.resolve(strict=True)
            expected_checkpoint = SHARED_POLICY_CHECKPOINT_PATH.resolve(strict=True)
        except OSError as error:
            parser.error(f"shared BEHAVIOR policy checkpoint is unavailable: {error}")
        if requested_checkpoint != expected_checkpoint:
            parser.error(
                "BEHAVIOR tasks require the shared base checkpoint "
                f"{expected_checkpoint}; task-specific SFT checkpoints are forbidden"
            )
        binding_file = getattr(
            args,
            "behavior_policy_checkpoint_binding_file",
            None,
        )
        if binding_file is not None:
            if args.behavior_phase == "explore" and args.behavior_job_id is None:
                parser.error(
                    "prevalidated checkpoint binding is restricted to serial Explore"
                )
            if args.behavior_phase == "eval":
                binding_path = Path(binding_file).expanduser().absolute()
                eval_output_dir = getattr(args, "output_dir", None)
                if not eval_output_dir:
                    parser.error(
                        "formal serial Eval checkpoint binding requires --output-dir"
                    )
                eval_output_parent = (
                    Path(eval_output_dir).expanduser().absolute().parent
                )
                if (
                    binding_path.name != "policy_checkpoint_binding.json"
                    or binding_path.parent != eval_output_parent
                ):
                    parser.error(
                        "formal serial Eval checkpoint binding must be the "
                        "campaign-root policy_checkpoint_binding.json"
                    )
            try:
                checkpoint_binding_value = _read_prevalidated_policy_checkpoint_binding(
                    binding_file
                )
            except (OSError, ValueError) as error:
                parser.error(str(error))
        else:
            try:
                checkpoint_binding_value = _expected_shared_policy_checkpoint_binding()
            except OSError as error:
                parser.error(str(error))
        args.policy_checkpoint = checkpoint_binding_value["resolved_path"]
        args._behavior_policy_checkpoint_binding = checkpoint_binding_value
        try:
            _validate_runtime_source_identity(args)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        if args.no_driver:
            if args.env_port <= 0:
                parser.error("--no-driver requires --env-port")
            if not args.vla_endpoint:
                parser.error("--no-driver requires --vla-endpoint")
            return

        root = Path(args.behavior_repo)
        python = Path(args.behavior_python)
        instance_dir = Path(args.activity_instance_dir)
        if not root.is_dir():
            parser.error(f"--behavior-repo does not exist: {root}")
        if not python.is_file():
            parser.error(f"--behavior-python does not exist: {python}")
        if not instance_dir.is_dir():
            parser.error(f"--activity-instance-dir does not exist: {instance_dir}")
        instance_matches = list(
            instance_dir.glob(f"*_{args.activity_instance_id}_template-tro_state.json")
        )
        if len(instance_matches) != 1:
            parser.error(
                "expected exactly one tro_state file for activity instance "
                f"{args.activity_instance_id} in {instance_dir}, got {len(instance_matches)}"
            )
        if candidate_requested:
            state_sha256 = hashlib.sha256(instance_matches[0].read_bytes()).hexdigest()
            if state_sha256 != args.behavior_candidate_state_sha256:
                parser.error("candidate state file SHA256 mismatch")

    def recipe_tag(self, args: argparse.Namespace) -> str:
        task_spec = getattr(args, "_behavior_task_spec", None)
        if task_spec is None:
            task_spec = resolve_task_spec(
                task_name=str(args.task_name),
                task_index=int(args.task),
            )
        return task_spec.tag(int(args.public_seed))

    def dashboard_state(self, args: argparse.Namespace, *, output_dir: Path) -> Any:
        from robots.behavior.dashboard_state import State

        task_spec = getattr(args, "_behavior_task_spec", None)
        if task_spec is None:
            task_spec = resolve_task_spec(
                task_name=str(args.task_name),
                task_index=int(args.task),
            )
        state = State(
            run_id=f"behavior/{output_dir.name}",
            name=self.recipe_tag(args),
            suite=args.suite,
            task=int(args.task),
            seed=int(args.public_seed),
            output_dir=str(output_dir),
            video_path=str(output_dir / "episode.mp4"),
        )
        return state

    def prompt_vars(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
        recipe_tag: str,
    ) -> dict[str, Any]:
        task_spec = getattr(args, "_behavior_task_spec", None)
        if task_spec is None:
            task_spec = resolve_task_spec(
                task_name=str(args.task_name),
                task_index=int(args.task),
            )
        task_language = str(getattr(args, "_behavior_task_language", "")).strip()
        if not task_language:
            raise RuntimeError(
                "BEHAVIOR task language is unavailable before prompt rendering"
            )
        frozen_contents = getattr(args, "_behavior_frozen_contents", None) or {}
        frozen_manifest = getattr(args, "_behavior_frozen_inputs", None)
        frozen_source_public_seed = getattr(
            args, "_behavior_frozen_source_public_seed", None
        )
        frozen_source_tag = getattr(args, "_behavior_frozen_source_tag", None)
        prompt_memory_manifest = None
        if frozen_manifest is not None:
            if (
                frozen_manifest.get("source_public_seed") != frozen_source_public_seed
                or frozen_manifest.get("source_tag") != frozen_source_tag
                or isinstance(frozen_source_public_seed, bool)
                or not isinstance(frozen_source_public_seed, int)
                or frozen_source_public_seed not in task_spec.explore_public_seeds
                or frozen_source_tag != task_spec.tag(frozen_source_public_seed)
            ):
                raise RuntimeError(
                    "validated frozen Eval source identity is unavailable or "
                    "inconsistent"
                )
            prompt_memory_manifest = {
                key: frozen_manifest[key]
                for key in (
                    "schema_version",
                    "source_public_seed",
                    "source",
                    "source_tag",
                    "mapping_version",
                    "bundle_id",
                    "amendment_sha256",
                    "provenance_sha256",
                    "recipe_sha256",
                    "memory_sha256",
                    "global_memory_snapshot_sha256",
                    "source_artifacts_sha256",
                )
                if key in frozen_manifest
            }
            public_frozen_paths = {
                f"recipe_{frozen_source_tag}.jsonl",
                f"memory/{args.task_name}.md",
                f"memory/{args.task_name}_provenance.json",
                "publication_amendment.json",
            }
            prompt_memory_manifest["files"] = {
                name: {
                    "size_bytes": metadata["size_bytes"],
                    "sha256": metadata["sha256"],
                }
                for name, metadata in frozen_manifest["files"].items()
                if name in public_frozen_paths
            }
        return {
            "suite": args.suite,
            "task": args.task,
            "task_name": args.task_name,
            "task_language": task_language,
            "public_seed": args.public_seed,
            "seed": args.public_seed,
            "max_episode_steps": args.max_episode_steps,
            "max_tool_calls": args.max_tool_calls,
            "max_wall_clock_s": args.max_wall_clock_s,
            "output_dir": output_dir,
            "recipe_tag": recipe_tag,
            **build_prompt_context(
                phase=args.behavior_phase,
                task_name=args.task_name,
                task_language=task_language,
                public_seed=args.public_seed,
                recipe_tag=recipe_tag,
                output_dir=str(output_dir),
                max_session_steps=args.max_episode_steps,
                global_tool_budget=args.max_tool_calls,
                wall_clock_seconds=args.max_wall_clock_s,
                target_prior=args._behavior_target_prior,
                reviewed_explore_experience=args._behavior_explore_experience,
                additional_expert_knowledge=(
                    args._behavior_additional_expert_knowledge
                ),
                reviewed_repo_memory_manifest=json.dumps(
                    args._behavior_repo_memory_prompt_input,
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                reviewed_recipe_priors=args._behavior_recipe_priors,
                reviewed_recipe_selection_manifest=json.dumps(
                    args._behavior_recipe_catalog_input["selection"],
                    sort_keys=True,
                    ensure_ascii=False,
                ),
                robust_recipe=frozen_contents.get("recipe"),
                task_memory=frozen_contents.get("memory"),
                memory_manifest=(
                    json.dumps(
                        prompt_memory_manifest,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                    if prompt_memory_manifest is not None
                    else None
                ),
                source_public_seed=frozen_source_public_seed,
                source_recipe_tag=frozen_source_tag,
                attempt_index=int(getattr(args, "behavior_attempt_index", 1) or 1),
                job_id=getattr(args, "behavior_job_id", None),
                prior_attempt_summaries=getattr(
                    args, "_behavior_prior_attempt_summaries", None
                ),
            ),
        }

    def _expected_meta(self, args: argparse.Namespace) -> dict[str, Any]:
        # env.get_env_meta intentionally exposes only the public benchmark
        # identity. Native simulator bindings belong in the run manifest and
        # must not leak into the Agent-facing RPC surface.
        return {
            "suite": args.suite,
            "task": args.task,
            "task_name": args.task_name,
            "public_seed": args.public_seed,
            "max_episode_steps": args.max_episode_steps,
        }


class _ArgumentErrorSink:
    """Argparse-compatible validation sink for EnvSpec.parse_config."""

    @staticmethod
    def error(message: str) -> None:
        raise ValueError(message)


def add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register BEHAVIOR flags on the shared CLI parser."""

    del use_dashboard
    specs = (TURNING_ON_RADIO_TASK_SPEC, PICKING_UP_TRASH_TASK_SPEC)
    parser.set_defaults(
        planner="codex",
        public_seed_max=max(TURNING_ON_RADIO_TASK_SPEC.public_seed_to_instance),
        dashboard_task_options=tuple(
            {
                "task_index": spec.task_index,
                "task_name": spec.task_name,
                "scene_model": spec.scene_model,
                "public_seed_max": max(spec.public_seed_to_instance),
                "public_instances": tuple(
                    spec.public_seed_to_instance[seed]
                    for seed in sorted(spec.public_seed_to_instance)
                ),
            }
            for spec in specs
        ),
    )
    _BehaviorConfig().add_cli_args(parser)


def prepare_resources(args: argparse.Namespace) -> Any:
    """Prepare one pinned BEHAVIOR resource snapshot before prompt parsing."""

    from robots.behavior.resources import prepare_behavior_resources

    return prepare_behavior_resources(args)


def parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate BEHAVIOR arguments and derive one immutable run identity."""

    provider = _BehaviorConfig()
    provider.validate_args(args, _ArgumentErrorSink())  # type: ignore[arg-type]
    task_spec = args._behavior_task_spec
    # Prompt rendering happens before the environment starts.  The TaskSpec
    # language is the pinned expected value; init_runtime verifies the reset
    # response matches it exactly before any planner action is possible.
    args._behavior_task_language = task_spec.task_language
    args.public_seed_max = max(task_spec.public_seed_to_instance)
    recipe_tag = provider.recipe_tag(args)
    output_dir_value = getattr(args, "output_dir", None)
    if output_dir_value is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir_value = get_repo_root() / "logs" / f"{timestamp}_{recipe_tag}"
    output_dir = Path(output_dir_value)
    if getattr(args, "behavior_dashboard_event_sink", False):
        dashboard_state = FileDashboardSink(output_dir / "dashboard_events.jsonl")
    elif getattr(args, "dashboard", False):
        dashboard_state = provider.dashboard_state(args, output_dir=output_dir)
    else:
        dashboard_state = None
    args._behavior_dashboard_state = dashboard_state
    prompt_vars = provider.prompt_vars(
        args,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
    )
    prompt_bundle = PromptBundle(system=system_prompt, user=user_prompt)
    rendered_system = prompt_bundle.render("system", variables=prompt_vars)
    rendered_user = prompt_bundle.render("user", variables=prompt_vars)
    combined_prompt = json.dumps(
        {
            "system": rendered_system,
            "user": rendered_user,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    args._behavior_prompt_binding = {
        "profile_id": task_spec.prompt_profile_id,
        "rendered_system_sha256": hashlib.sha256(
            rendered_system.encode("utf-8")
        ).hexdigest(),
        "rendered_user_sha256": hashlib.sha256(
            rendered_user.encode("utf-8")
        ).hexdigest(),
        "combined_sha256": hashlib.sha256(combined_prompt.encode("utf-8")).hexdigest(),
    }
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        dashboard_state=dashboard_state,
        task_desc={
            "suite": args.suite,
            "task": int(args.task),
            "task_name": args.task_name,
            "public_seed": int(args.public_seed),
            "activity_definition_id": int(args.activity_definition_id),
            "activity_instance_id": int(args.activity_instance_id),
        },
    )


def _task_language_from_reset(initial_observation: dict[str, Any]) -> str:
    value = initial_observation.get("task_descriptions")
    if isinstance(value, (list, tuple)):
        value = next(
            (item for item in value if isinstance(item, str) and item.strip()),
            None,
        )
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            "environment reset did not return authoritative task_descriptions"
        )
    return value.strip()


def init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[BehaviorRuntimeResources], dict[str, Any]]:
    """Start one serial BEHAVIOR runtime and return generic CLI resources."""

    provider = _BehaviorConfig()
    provider._resolve_paths(args)
    task_spec = getattr(args, "_behavior_task_spec", None)
    if task_spec is None:
        task_spec = resolve_task_spec(
            task_name=str(args.task_name),
            task_index=int(args.task),
        )
        args._behavior_task_spec = task_spec
        args._behavior_task_spec_binding = _task_spec_binding(task_spec)
    if args.activity_instance_id is None:
        args.activity_instance_id = task_spec.instance_for_public_seed(
            int(args.public_seed),
            phase=args.behavior_phase,
        )
    args._behavior_mapping_version = (
        task_spec.mapping_version
        if getattr(args, "behavior_candidate_instance_id", None) is None
        else task_spec.candidate_mapping_version
    )
    if not hasattr(args, "_behavior_policy_checkpoint_binding"):
        checkpoint_binding = _expected_shared_policy_checkpoint_binding()
        args.policy_checkpoint = checkpoint_binding["resolved_path"]
        args._behavior_policy_checkpoint_binding = checkpoint_binding

    _validate_runtime_source_identity(args)
    isolation_root = getattr(args, "behavior_runtime_isolation_root", None)
    isolation_namespace = getattr(args, "behavior_runtime_namespace", None)
    isolation_expected = getattr(
        args, "behavior_runtime_isolation_binding_sha256", None
    )
    if bool(isolation_root) != bool(isolation_namespace):
        raise ValueError(
            "--behavior-runtime-isolation-root and "
            "--behavior-runtime-namespace must be supplied together"
        )
    runtime_isolation: CampaignRuntimeIsolation | None = None
    if isolation_root:
        marker = Path(str(isolation_root)).expanduser() / _RUNTIME_ISOLATION_FILENAME
        if marker.is_file():
            runtime_isolation = validate_campaign_runtime_isolation(
                str(isolation_root),
                (
                    str(isolation_expected)
                    if isinstance(isolation_expected, str) and isolation_expected
                    else None
                ),
            )
        else:
            runtime_isolation = prepare_campaign_runtime_isolation(
                str(isolation_root),
                str(isolation_namespace),
                str(args.cuda_device),
                behavior_python=str(args.behavior_python),
            )
            if (
                isinstance(isolation_expected, str)
                and isolation_expected
                and runtime_isolation.binding_sha256 != isolation_expected
            ):
                raise RuntimeError(
                    "new campaign runtime isolation binding differs from expected"
                )
        if runtime_isolation.cuda_device != str(args.cuda_device):
            raise RuntimeError("campaign runtime isolation GPU mismatch")
    elif str(getattr(args, "behavior_controller_mode", "hybrid")) == (
        "pi0_nav_pick_only"
    ):
        raise ValueError("pure VLA baseline requires campaign runtime isolation")
    args._behavior_runtime_isolation = runtime_isolation

    output_dir = Path(output_dir)
    manifest = RunManifest.start(output_dir, args, repo_root=get_repo_root())
    runtime_manifest_values = {
        "runtime_isolation": (
            runtime_isolation.as_dict() if runtime_isolation is not None else None
        ),
        "controller_mode": str(getattr(args, "behavior_controller_mode", "hybrid")),
    }
    manifest_update = getattr(manifest, "_update", None)
    if callable(manifest_update):
        manifest_update(lambda data: data.update(runtime_manifest_values))
    else:
        manifest.data.update(runtime_manifest_values)
    manifest_token = _ACTIVE_RUN_MANIFEST.set(manifest)
    env_proc = None
    vla_proc = None
    env_rpc_client = None
    model = None
    vla_endpoint = args.vla_endpoint
    try:
        if args.no_driver:
            env_rpc_client = _external_env_rpc_client(args)
        else:
            env_proc = start_env_server(args, output_dir=output_dir)
            env_rpc_client = _managed_env_rpc_client(env_proc)
            if not vla_endpoint:
                vla_endpoint, vla_proc = start_vla_server(
                    args,
                    output_dir=output_dir,
                )
        env = BehaviorEnvClient(
            env_rpc_client,
            expected_meta=provider._expected_meta(args),
        )
        model = BehaviorVLAClient(
            str(vla_endpoint),
            binding_id=getattr(args, "behavior_vla_binding_id", None),
        )
        env.vla_endpoint = str(vla_endpoint)
        if args.no_driver:
            model.wait_for_healthz(
                timeout_s=30.0,
                expected_checkpoint_binding=args._behavior_policy_checkpoint_binding,
            )
        else:
            model.healthz(
                timeout_ms=5000,
                expected_checkpoint_binding=args._behavior_policy_checkpoint_binding,
            )
        initial_observation, initial_info = env.reset()
        task_language = _task_language_from_reset(initial_observation)
        if task_language != task_spec.task_language:
            raise RuntimeError(
                "environment task language does not match the selected TaskSpec"
            )
        args._behavior_task_language = task_language
        manifest.set_task_language(task_language)
        dashboard = getattr(args, "_behavior_dashboard_state", None)
        if dashboard is not None:
            dashboard.set_metadata({"task-language": task_language})
        from robots.behavior.dashboard_control import (
            BehaviorCommandArbiter,
            BehaviorRawSuccessLatch,
        )

        success_latch = BehaviorRawSuccessLatch()
        command_arbiter = BehaviorCommandArbiter(success_latch=success_latch)
        resources = BehaviorRuntimeResources(
            output_dir=output_dir,
            env_proc=env_proc,
            vla_proc=vla_proc,
            manifest=manifest,
            env_rpc_client=env_rpc_client,
            command_arbiter=command_arbiter,
            success_latch=success_latch,
            dashboard_state=dashboard,
        )
        manifest.running()
        primitives_kwargs = {
            "env": env,
            "model": model,
            "task_name": args.task_name,
            "behavior_phase": args.behavior_phase,
            "public_seed": args.public_seed,
            "initial_attempt_index": int(
                getattr(args, "behavior_attempt_index", 1) or 1
            ),
            "job_id": getattr(args, "behavior_job_id", None),
            "max_episode_steps": args.max_episode_steps,
            "max_tool_calls": (
                None
                if str(getattr(args, "behavior_controller_mode", "hybrid"))
                == "pi0_nav_pick_only"
                else args.max_tool_calls
            ),
            "max_wall_clock_s": args.max_wall_clock_s,
            "pure_vla_baseline": bool(
                str(getattr(args, "behavior_controller_mode", "hybrid"))
                == "pi0_nav_pick_only"
            ),
            "action_horizon": DEFAULT_ACTION_CHUNK,
            "output_dir": output_dir,
            "video_path": output_dir / "episode.mp4",
            "initial_observation": initial_observation,
            "initial_info": initial_info,
            "_dashboard_runtime_resource": resources,
            "_dashboard_command_arbiter": command_arbiter,
            "_dashboard_success_latch": success_latch,
            "_dashboard_motion_allowed": bool(
                str(getattr(args, "behavior_controller_mode", "hybrid")) == "hybrid"
                and str(getattr(args, "suite", "")).startswith("behavior")
            ),
            "_dashboard_observe_allowed": bool(
                str(getattr(args, "suite", "")).startswith("behavior")
            ),
            "_dashboard_control_unavailable_reason": (
                "Pure-VLA controller ownership disables manual motion."
                if str(getattr(args, "behavior_controller_mode", "hybrid"))
                == "pi0_nav_pick_only"
                else None
            ),
        }
        return [resources], primitives_kwargs
    except BaseException as error:
        model_close = getattr(model, "close", None)
        if callable(model_close):
            try:
                model_close()
            except BaseException:
                logger.exception(
                    "BEHAVIOR VLA client cleanup after start failure failed"
                )
        try:
            stop_env_server(env_proc, output_dir=output_dir)
        except BaseException:
            logger.exception("BEHAVIOR env cleanup after start failure failed")
        finally:
            try:
                manifest.process_stopped("env", env_proc)
            except BaseException:
                logger.exception("BEHAVIOR env manifest stop update failed")
        client_close = getattr(env_rpc_client, "close", None)
        if callable(client_close):
            try:
                client_close()
            except BaseException:
                logger.exception(
                    "BEHAVIOR env client cleanup after start failure failed"
                )
        try:
            _terminate_process(vla_proc)
        except BaseException:
            logger.exception("BEHAVIOR VLA cleanup after start failure failed")
        finally:
            if "vla" in manifest.data["processes"]:
                try:
                    manifest.process_stopped("vla", vla_proc)
                except BaseException:
                    logger.exception("BEHAVIOR VLA manifest stop update failed")
        try:
            manifest.finish(status="failed", error=error)
        except BaseException:
            logger.exception("BEHAVIOR failed-session manifest update failed")
        raise
    finally:
        _ACTIVE_RUN_MANIFEST.reset(manifest_token)


def run_planner(
    *,
    planner: Any,
    system_prompt: str,
    user_message: str,
    toolkit: Any,
    max_turns: int,
    input_queue: Any = None,
    args: argparse.Namespace | None = None,
    run_config: RunConfig | None = None,
    runtime_resources: Any = (),
):
    """Run BEHAVIOR continuation until a trusted terminal condition."""

    if input_queue is not None:
        raise ValueError("interactive steering is not supported for BEHAVIOR runs")
    del args, run_config
    from robots.behavior.continuation import run_behavior_planner_continuation

    runtime = next(
        (
            resource
            for resource in runtime_resources
            if isinstance(resource, BehaviorRuntimeResources)
        ),
        None,
    )
    if runtime is not None:
        runtime.toolkit = toolkit
    return run_behavior_planner_continuation(
        task_name=toolkit._task_spec.task_name,
        planner=planner,
        system_prompt=system_prompt,
        initial_user_message=user_message,
        toolkit=toolkit,
        max_turns=max_turns,
        output_dir=toolkit._primitives.output_dir,
        runtime=runtime,
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


def _outcome_value(outcome: Any, name: str, default: Any = None) -> Any:
    if isinstance(outcome, dict):
        return outcome.get(name, default)
    return getattr(outcome, name, default)


def _transcript_value(value: Any) -> Any:
    """Remove inline images while retaining the planner transcript shape."""

    if isinstance(value, list):
        return [_transcript_value(item) for item in value]
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url"}:
            return {
                "type": value.get("type"),
                "_omitted_for_transcript": True,
            }
        return {str(key): _transcript_value(item) for key, item in value.items()}
    return value


def finalize_run(outcome: Any) -> dict[str, Any]:
    """Atomically write the cleanup-bound BEHAVIOR terminal result."""

    args = _outcome_value(outcome, "args")
    run_config = _outcome_value(outcome, "run_config")
    toolkit = _outcome_value(outcome, "toolkit")
    if args is None or run_config is None:
        raise TypeError("BEHAVIOR finalization requires args and run_config")
    output_dir = Path(run_config.output_dir)
    finish_result = _outcome_value(outcome, "finish_result")
    error = _outcome_value(outcome, "error")
    cleanup = _outcome_value(outcome, "cleanup")
    if isinstance(cleanup, dict):
        cleanup_complete = cleanup.get("complete") is True
        cleanup_error = next(
            (
                entry.get("error")
                for entry in [
                    cleanup.get("toolkit"),
                    *(cleanup.get("runtime_resources") or []),
                ]
                if isinstance(entry, dict) and entry.get("error")
            ),
            None,
        )
    else:
        cleanup_complete = cleanup is True
        cleanup_error = None
    runtime_cleanup = "complete" if cleanup_complete else "error"

    toolkit_state = (
        toolkit.runner_continuation_state()
        if toolkit is not None
        and callable(getattr(toolkit, "runner_continuation_state", None))
        else {}
    )
    raw_success = toolkit_state.get("raw_official_success_verified") is True
    finish_success = (
        finish_result.get("task_success") if isinstance(finish_result, dict) else None
    )
    task_success: bool | None
    if raw_success:
        task_success = True
    elif finish_success is False:
        task_success = False
    else:
        task_success = None
    redacted_error = redact_text(str(error)) if error else None
    if runtime_cleanup != "complete" and redacted_error is None:
        redacted_error = redact_text(str(cleanup_error or "runtime cleanup failed"))
    recipe_path = None
    if runtime_cleanup == "complete" and redacted_error is None and toolkit is not None:
        try:
            recipe_path = toolkit.write_recipe(run_config.recipe_tag)
            if not isinstance(outcome, dict):
                outcome.recipe_path = recipe_path
        except BaseException as recipe_error:
            try:
                (output_dir / f"recipe_{run_config.recipe_tag}.jsonl").unlink(
                    missing_ok=True
                )
            except OSError:
                logger.exception(
                    "failed to remove incomplete BEHAVIOR recipe publication"
                )
            redacted_error = redact_text(
                f"{type(recipe_error).__name__}: {recipe_error}"
            )
            logger.error("BEHAVIOR final publication failed: %s", redacted_error)
    run_status = (
        "error"
        if redacted_error is not None
        else "completed"
        if task_success is not None
        else "incomplete"
    )
    transcript_path = _outcome_value(outcome, "transcript_path")
    if transcript_path is None:
        transcript_path = output_dir / f"transcript_{run_config.recipe_tag}.json"
    transcript = {
        **dict(_outcome_value(outcome, "task_desc", {}) or {}),
        "model": getattr(args, "model", None),
        "elapsed_s": round(
            float(_outcome_value(outcome, "elapsed_s", 0.0)),
            3,
        ),
        "finish": finish_result,
        "error": redacted_error,
        "stats": _outcome_value(outcome, "stats", {}),
        "messages": _transcript_value(_outcome_value(outcome, "messages", [])),
    }
    _atomic_json(Path(transcript_path), transcript)
    required_artifacts = [
        output_dir / "run_manifest.json",
        output_dir / "session_manifest.json",
        output_dir / "behavior_result.json",
        output_dir / "behavior_tool_trace.jsonl",
        output_dir / "episode.mp4",
    ]
    if task_success is True:
        required_artifacts.extend(
            [
                output_dir / "official_success_receipt.json",
                output_dir / "behavior_action_trace.jsonl",
            ]
        )
        if getattr(args, "behavior_phase", None) == "explore":
            required_artifacts.append(
                output_dir / f"recipe_{run_config.recipe_tag}.jsonl"
            )
    artifact_seal_complete = bool(
        runtime_cleanup == "complete"
        and all(path.is_file() for path in required_artifacts)
    )
    result = {
        "schema_version": 1,
        "run_status": run_status,
        "task_success": task_success,
        "official_success_source": (
            'info["done"]["success"]' if task_success is not None else None
        ),
        "planner": {
            "backend": getattr(args, "planner", None),
            "model": getattr(args, "model", None),
            "reasoning_effort": getattr(args, "reasoning_effort", None),
        },
        "finish": finish_result,
        "error": redacted_error,
        "elapsed_s": round(float(_outcome_value(outcome, "elapsed_s", 0.0)), 3),
        "runtime_cleanup": runtime_cleanup,
        "artifact_seal_complete": artifact_seal_complete,
        "job": {
            "job_id": getattr(args, "behavior_job_id", None),
            "attempt_index": int(getattr(args, "behavior_attempt_index", 1) or 1),
        },
        "artifacts": {
            "transcript": str(transcript_path) if transcript_path else None,
            "recipe": str(recipe_path) if recipe_path else None,
            "official_success_receipt": (
                str(output_dir / "official_success_receipt.json")
                if task_success is True
                else None
            ),
        },
    }
    messages = _outcome_value(outcome, "messages", [])
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                result["agent_summary"] = redact_text(content.strip())
                break
    _atomic_json(output_dir / "final_result.json", result)
    dashboard = getattr(run_config, "dashboard_state", None)
    on_event = getattr(dashboard, "on_event", None)
    if raw_success and callable(on_event):
        try:
            on_event(
                {
                    "type": "official_success",
                    "attempt_index": int(
                        getattr(args, "behavior_attempt_index", 1) or 1
                    ),
                    "task_success": True,
                }
            )
        except BaseException:
            logger.exception("BEHAVIOR Dashboard official-success event failed")
    return result


__all__ = [
    "BEHAVIOR_NATIVE_ENV_SEED",
    "BehaviorRuntimeResources",
    "CampaignRuntimeIsolation",
    "RESOURCE_POLICY",
    "_expected_shared_policy_checkpoint_binding",
    "_read_prior_attempt_summaries",
    "_terminate_process",
    "add_cli_args",
    "finalize_run",
    "init_runtime",
    "parse_config",
    "prepare_campaign_runtime_isolation",
    "run_planner",
    "start_vla_server",
    "validate_campaign_runtime_isolation",
]
