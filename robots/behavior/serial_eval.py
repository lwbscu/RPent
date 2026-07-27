"""Reproducible, strictly serial BEHAVIOR public-instance evaluation."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from robots.behavior.dataset_resources import (
    prepare_local_dataset_resources,
    prepare_pinned_dataset_resources,
    verify_pinned_dataset_resources,
    write_dataset_resource_binding,
)
from robots.behavior.memory_snapshot import (
    BehaviorMemorySnapshotError,
    load_behavior_memory_snapshot,
)
from robots.behavior.policy_checkpoint import (
    SHARED_POLICY_CHECKPOINT_PATH,
)
from robots.behavior.publication import (
    PublicationValidationError,
    ValidatedBehaviorPublication,
    validate_canonical_publication_root,
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
    validate_campaign_runtime_isolation,
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
)
from robots.behavior.terminal_success import (
    summarize_action_trace_success,
    validate_terminal_success_receipt,
)
from robots.behavior.vla_client import BehaviorVLAClient

TURNING_ON_RADIO_TASK_ID = TURNING_ON_RADIO_TASK_SPEC.task_index
TURNING_ON_RADIO_TASK_NAME = TURNING_ON_RADIO_TASK_SPEC.task_name
TURNING_ON_RADIO_PUBLIC_IDS = (
    242,
    295,
    211,
    203,
    109,
    181,
    197,
    187,
    214,
    139,
    185,
    102,
    246,
    105,
    271,
    119,
    220,
    224,
    212,
    298,
)
PICKING_UP_TRASH_PUBLIC_IDS = tuple(
    PICKING_UP_TRASH_TASK_SPEC.public_seed_to_instance[public_seed]
    for public_seed in sorted(PICKING_UP_TRASH_TASK_SPEC.public_seed_to_instance)
)
TEST_INSTANCES_SHA256 = (
    "5cd78301ddc764158a20d4cf8c134afb2cb3bbf1f0611aa55aee34873b5b4d23"
)
EVAL_PUBLIC_SEEDS = TURNING_ON_RADIO_TASK_SPEC.eval_public_seeds
DEFAULT_MAX_WALL_CLOCK_S = 6900
DEFAULT_CLEANUP_DEADLINE_S = 7080
DEFAULT_INSTANCE_TIMEOUT_S = 7200
MAX_INSTANCE_TIMEOUT_S = 7200
INSTANCE_CHILD_PROCESS_FILENAME = "instance_child_process.json"
FORCED_MANAGED_CLEANUP_FILENAME = "runner_forced_cleanup.json"
_EXPECTED_RUN_NONCE_ENV = "RPENT_BEHAVIOR_EXPECTED_RUN_NONCE"
_RUN_NONCE_RE = re.compile(r"[0-9a-f]{32}")

_PINNED_CSV_PUBLIC_IDS = {
    TURNING_ON_RADIO_TASK_NAME: TURNING_ON_RADIO_PUBLIC_IDS,
    PICKING_UP_TRASH_TASK_SPEC.task_name: PICKING_UP_TRASH_PUBLIC_IDS,
}


@dataclass(frozen=True)
class EvalEntry:
    """One immutable public instance and its exact launch argv."""

    split_position: int
    csv_position: int
    activity_instance_id: int
    public_seed: int
    seed: int
    output_dir: Path
    argv: tuple[str, ...]
    checkpoint: Path
    cuda_device: str
    instance_state_path: Path
    instance_state_sha256: str
    frozen_publication_binding: dict[str, Any] | None = None
    reviewed_repo_memory_binding: dict[str, Any] | None = None
    reviewed_recipe_catalog_binding: dict[str, Any] | None = None
    policy_checkpoint_binding: dict[str, Any] | None = None
    resource_source_binding: dict[str, Any] | None = None
    source_snapshot_binding: dict[str, Any] | None = None
    runtime_isolation_binding: dict[str, Any] | None = None
    task_name: str = TURNING_ON_RADIO_TASK_NAME

    def __post_init__(self) -> None:
        spec = get_task_spec(self.task_name)
        expected_instance = spec.instance_for_public_seed(
            self.public_seed,
            phase="eval",
        )
        if self.activity_instance_id != expected_instance:
            raise ValueError(
                "Eval entry task/public/native identity mismatch: "
                f"{spec.task_name} s{self.public_seed} requires "
                f"instance {expected_instance}, got {self.activity_instance_id}"
            )


@dataclass(frozen=True)
class InstanceDeadlineBinding:
    """One monotonic clock binding shared by the paired and nested runners."""

    started_monotonic_ns: int
    action_deadline_monotonic_ns: int
    cleanup_deadline_monotonic_ns: int
    hard_deadline_monotonic_ns: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _validate_instance_deadline_binding(
    binding: InstanceDeadlineBinding,
    *,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
) -> None:
    values = tuple(binding.as_dict().values())
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("absolute monotonic deadlines must be positive integers")
    started = binding.started_monotonic_ns
    expected = (
        started + action_deadline_s * 1_000_000_000,
        started + cleanup_deadline_s * 1_000_000_000,
        started + instance_timeout_s * 1_000_000_000,
    )
    actual = (
        binding.action_deadline_monotonic_ns,
        binding.cleanup_deadline_monotonic_ns,
        binding.hard_deadline_monotonic_ns,
    )
    if actual != expected:
        raise ValueError(
            "absolute monotonic deadline differences must exactly match the "
            "configured action, cleanup, and hard timeout seconds"
        )


def _new_instance_deadline_binding(
    *,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
) -> InstanceDeadlineBinding:
    """Sample the standalone start once and derive every deadline from it."""

    started = time.monotonic_ns()
    binding = InstanceDeadlineBinding(
        started_monotonic_ns=started,
        action_deadline_monotonic_ns=(started + action_deadline_s * 1_000_000_000),
        cleanup_deadline_monotonic_ns=(started + cleanup_deadline_s * 1_000_000_000),
        hard_deadline_monotonic_ns=(started + instance_timeout_s * 1_000_000_000),
    )
    _validate_instance_deadline_binding(
        binding,
        action_deadline_s=action_deadline_s,
        cleanup_deadline_s=cleanup_deadline_s,
        instance_timeout_s=instance_timeout_s,
    )
    return binding


def _validate_frozen_source_identity(
    publication: ValidatedBehaviorPublication,
    task_spec: BehaviorTaskSpec,
) -> tuple[int, str]:
    """Return one canonical Explore source identity for frozen formal Eval."""

    source_public_seed = publication.identity.public_seed
    source_tag = publication.identity.tag
    if source_public_seed not in task_spec.explore_public_seeds:
        raise PublicationValidationError(
            "source public seed is outside the selected task's Explore partition"
        )
    if (
        publication.identity.task_spec != task_spec
        or publication.manifest_binding.get("source_public_seed") != source_public_seed
        or publication.manifest_binding.get("source_tag") != source_tag
        or source_tag != task_spec.tag(source_public_seed)
    ):
        raise PublicationValidationError("source identity is inconsistent")
    return source_public_seed, source_tag


def _task_spec_binding(task_spec: BehaviorTaskSpec) -> dict[str, Any]:
    """Return the full task mapping and phase partition sealed by an Eval plan."""

    return {
        "task_index": task_spec.task_index,
        "task_name": task_spec.task_name,
        "task_language": task_spec.task_language,
        "prompt_profile_id": task_spec.prompt_profile_id,
        "activity_definition_id": task_spec.activity_definition_id,
        "scene_model": task_spec.scene_model,
        "public_seed_to_instance": {
            str(seed): instance_id
            for seed, instance_id in task_spec.public_seed_to_instance.items()
        },
        "mapping_version": task_spec.mapping_version,
        "candidate_mapping_version": task_spec.candidate_mapping_version,
        "explore_public_seeds": list(task_spec.explore_public_seeds),
        "eval_public_seeds": list(task_spec.eval_public_seeds),
    }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_binding(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_binding(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_binding(item) for item in value]
    return value


def _reviewed_recipe_catalog_binding(
    resource_root: Path,
    *,
    task_name: str = TURNING_ON_RADIO_TASK_NAME,
) -> dict[str, Any]:
    spec = get_task_spec(task_name)
    catalog = load_behavior_recipe_catalog(resource_root / "recipes")
    selection = catalog.select(spec.task_name, "formal_eval")
    selection_binding = _plain_binding(selection.public_binding)
    selected_entries = selection_binding.get("selected_entries")
    if not isinstance(selected_entries, list) or any(
        not isinstance(entry, dict)
        or entry.get("provenance_class") != "canonical_public_explore"
        for entry in selected_entries
    ):
        raise RuntimeError(
            "formal Eval Recipe Catalog selection contains non-canonical provenance"
        )
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_turning_on_radio_instances(
    csv_path: Path,
    *,
    expected_sha256: str = TEST_INSTANCES_SHA256,
) -> tuple[int, ...]:
    """Backward-compatible Radio CSV reader."""

    return read_task_instances(
        csv_path,
        task_name=TURNING_ON_RADIO_TASK_NAME,
        expected_sha256=expected_sha256,
    )


def read_task_instances(
    csv_path: Path,
    *,
    task_name: str,
    expected_sha256: str = TEST_INSTANCES_SHA256,
) -> tuple[int, ...]:
    """Read one task's authoritative ordered public-instance row."""

    spec = get_task_spec(task_name)
    actual_sha256 = _sha256(csv_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "test_instances.csv SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    matches: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if (
                row.get("Task ID") == str(spec.task_index)
                and row.get("Task") == spec.task_name
            ):
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {spec.task_name} row in test_instances.csv"
        )
    raw_ids = matches[0].get("Public Test Instance IDs", "")
    try:
        instance_ids = tuple(int(item.strip()) for item in raw_ids.split(","))
    except ValueError as error:
        raise RuntimeError("invalid public instance ID list") from error
    expected_ids = _PINNED_CSV_PUBLIC_IDS[spec.task_name]
    if instance_ids != expected_ids:
        raise RuntimeError(
            f"{spec.task_name} public IDs differ from the pinned ordered protocol"
        )
    if len(instance_ids) != len(set(instance_ids)):
        raise RuntimeError(f"{spec.task_name} public IDs contain duplicates")
    return instance_ids


def select_instances(instance_ids: tuple[int, ...], split: str) -> tuple[int, ...]:
    """Select a protocol slice without reordering raw instance IDs."""

    if instance_ids != TURNING_ON_RADIO_PUBLIC_IDS:
        raise RuntimeError("instance order is not the pinned CSV order")
    if split == "official_first10":
        return instance_ids[:10]
    if split == "holdback_last10":
        return instance_ids[10:]
    if split == "all_public":
        return instance_ids
    raise ValueError(f"unknown split: {split}")


def _git(repo_root: Path, *arguments: str, check: bool = True) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        if check:
            raise
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _file_fingerprint(path: Path) -> dict[str, Any]:
    requested = path.expanduser().absolute()
    resolved = requested.resolve()
    stat = resolved.stat()
    return {
        "path": str(requested),
        "resolved_path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(resolved),
    }


def _checkout_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    top = _git(resolved, "rev-parse", "--show-toplevel", check=False)
    commit = _git(resolved, "rev-parse", "HEAD", check=False)
    status = _git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        check=False,
    )
    if top is None or commit is None or status is None:
        raise RuntimeError(f"required checkout is not readable as git: {resolved}")
    top_path = Path(top).resolve()
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=resolved,
            check=True,
            capture_output=True,
            timeout=120,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=resolved,
            check=True,
            capture_output=True,
            timeout=120,
        ).stdout.split(b"\0")
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"cannot fingerprint checkout content: {resolved}"
        ) from error
    content_digest = hashlib.sha256(diff)
    for raw_relative in sorted(item for item in untracked if item):
        relative = os.fsdecode(raw_relative)
        candidate = top_path / relative
        content_digest.update(raw_relative)
        content_digest.update(b"\0")
        if candidate.is_symlink():
            content_digest.update(b"symlink\0")
            content_digest.update(os.fsencode(os.readlink(candidate)))
        elif candidate.is_file():
            content_digest.update(b"file\0")
            with candidate.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    content_digest.update(chunk)
        else:
            content_digest.update(b"other\0")
    return {
        "path": str(resolved),
        "toplevel": str(top_path),
        "commit": commit,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_content_sha256": content_digest.hexdigest(),
    }


def source_identity(repo_root: Path) -> dict[str, Any]:
    """Require a clean, committed checkout and return its immutable identity."""

    commit = _git(repo_root, "rev-parse", "HEAD")
    branch = _git(repo_root, "branch", "--show-current")
    status = _git(repo_root, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("formal serial evaluation requires a clean worktree")
    return {
        "commit": commit,
        "branch": branch,
        "worktree": str(repo_root.resolve()),
        "worktree_dirty": False,
    }


def _validated_source_snapshot(
    snapshot_root: Path,
    *,
    expected_binding_sha256: str,
) -> Any:
    """Load the source-snapshot validator lazily for compatibility."""

    from robots.behavior.source_snapshot import validate_source_snapshot

    expected = str(expected_binding_sha256).strip().lower()
    binding = validate_source_snapshot(
        snapshot_root,
        expected_binding_sha256=expected,
    )
    if (
        Path(binding.snapshot_root).resolve() != snapshot_root.resolve()
        or binding.binding_sha256 != expected
    ):
        raise RuntimeError("validated source snapshot identity is inconsistent")
    return binding


def _validate_entry_python(python: Path, *, repo_root: Path) -> None:
    """Fail before plan creation unless the frozen RPent SDK Python is usable."""

    completed = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import httpx, openai_codex; "
                "import robots.behavior.cli; "
                "import robots.behavior.runtime"
            ),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = detail[-1] if detail else f"exit {completed.returncode}"
        raise RuntimeError(
            f"RPent entry Python dependency preflight failed: {python}: {suffix}"
        )


def _normalize_cuda_device(cuda_device: str) -> str:
    raw = str(cuda_device).strip()
    if not re.fullmatch(r"[0-9]+", raw):
        raise ValueError("cuda_device must be one decimal GPU ordinal")
    return str(int(raw))


def _validate_deadline_budget(
    *,
    planner_timeout_s: int,
    max_wall_clock_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
) -> None:
    """Validate one absolute per-instance deadline budget.

    The planner and runtime stop admitting work at or before the action budget.
    The launcher starts owned-process cleanup at the cleanup deadline and
    retains the remaining interval solely for bounded termination.  The hard
    watchdog is never allowed to exceed two hours.
    """

    values = (
        planner_timeout_s,
        max_wall_clock_s,
        cleanup_deadline_s,
        instance_timeout_s,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("Eval deadline values must be positive integers")
    if not (
        planner_timeout_s
        <= max_wall_clock_s
        < cleanup_deadline_s
        < instance_timeout_s
        <= MAX_INSTANCE_TIMEOUT_S
    ):
        raise ValueError(
            "Eval deadlines must satisfy planner <= wall-clock < cleanup "
            "< hard timeout <= 7200 seconds"
        )


def _validate_external_vla(
    endpoint: str,
    *,
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one externally owned VLA without assuming process ownership."""

    client = BehaviorVLAClient(str(endpoint))
    try:
        metadata = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
    finally:
        client.close()
    if (
        metadata.get("config_name") != "pi05_behavior"
        or metadata.get("actions_enabled") is not False
    ):
        raise RuntimeError(f"unexpected external VLA metadata: {metadata!r}")
    return {
        "endpoint": str(endpoint),
        "config_name": metadata["config_name"],
        "checkpoint_binding": _plain_binding(metadata.get("checkpoint_binding")),
        "managed": False,
    }


def _disable_external_vla_actions(
    endpoint: str,
    *,
    checkpoint_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an externally owned VLA to checkpoint-bound safe idle.

    Unlike the strict pre-run admission check, the first health sample may
    report ``actions_enabled=true`` while an admitted child is winding down.
    Checkpoint/config identity is still validated before the action gate is
    changed, and the final health sample must report exact boolean ``false``.
    """

    client = BehaviorVLAClient(str(endpoint))
    try:
        before = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
        if before.get("config_name") != "pi05_behavior":
            raise RuntimeError(f"unexpected external VLA metadata: {before!r}")
        disabled = client.disable_actions(timeout_ms=5000)
        if disabled.get("actions_enabled") is not False:
            raise RuntimeError(
                f"external VLA did not confirm action disable: {disabled!r}"
            )
        after = client.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=checkpoint_binding,
        )
    finally:
        client.close()
    if (
        after.get("config_name") != "pi05_behavior"
        or after.get("actions_enabled") is not False
    ):
        raise RuntimeError(f"unexpected external VLA metadata after disable: {after!r}")
    return {
        "endpoint": str(endpoint),
        "config_name": after["config_name"],
        "checkpoint_binding": _plain_binding(after.get("checkpoint_binding")),
        "managed": False,
        "actions_enabled": False,
    }


def _is_external_action_deadline_sigterm(
    *,
    returncode: int | None,
    externally_bound_deadline: bool,
    deadline_binding: InstanceDeadlineBinding,
    observed_at_monotonic_ns: int,
) -> bool:
    """Recognize the paired supervisor's exact action-deadline termination."""

    return bool(
        externally_bound_deadline
        and returncode == -int(signal.SIGTERM)
        and observed_at_monotonic_ns >= deadline_binding.action_deadline_monotonic_ns
    )


def _gpu_lock_path(cuda_device: str) -> Path:
    normalized = _normalize_cuda_device(cuda_device)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return Path("/tmp") / f"rpent-behavior-eval-gpu-{digest}.lock"


def _eval_lock_paths(
    output_root: Path,
    cuda_device: str,
    external_gpu_lock_owned: bool,
) -> tuple[Path, ...]:
    """Return the exact locks owned by this serial evaluator."""

    output_lock = output_root.parent / f".{output_root.name}.lock"
    if external_gpu_lock_owned:
        return (output_lock,)
    return (_gpu_lock_path(cuda_device), output_lock)


def _verify_input_fingerprints(
    *,
    repo_root: Path,
    source: dict[str, Any],
    global_inputs: dict[str, dict[str, Any]],
    entry: EvalEntry,
    source_snapshot_root: Path | None = None,
    source_snapshot_binding_sha256: str | None = None,
    runtime_isolation_root: Path | None = None,
    runtime_isolation_binding_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if source_snapshot_root is None:
        try:
            current_source = source_identity(repo_root)
        except Exception as error:
            errors.append(
                f"source identity unavailable: {type(error).__name__}: {error}"
            )
        else:
            if current_source != source:
                errors.append("RPent source identity changed after plan creation")
    else:
        try:
            current_snapshot = _validated_source_snapshot(
                source_snapshot_root,
                expected_binding_sha256=str(source_snapshot_binding_sha256),
            )
            current_binding = _plain_binding(current_snapshot.as_dict())
        except Exception as error:
            errors.append(
                f"source snapshot identity unavailable: {type(error).__name__}: {error}"
            )
        else:
            if current_binding != entry.source_snapshot_binding:
                errors.append("RPent source snapshot changed after plan creation")
    if runtime_isolation_root is not None:
        try:
            current_isolation = validate_campaign_runtime_isolation(
                runtime_isolation_root,
                runtime_isolation_binding_sha256,
            )
        except Exception as error:
            errors.append(
                "campaign runtime isolation unavailable: "
                f"{type(error).__name__}: {error}"
            )
        else:
            if current_isolation.as_dict() != entry.runtime_isolation_binding:
                errors.append("campaign runtime isolation changed after plan creation")
    for label, expected in global_inputs.items():
        try:
            if label.endswith("_checkout"):
                actual = _checkout_identity(Path(expected["path"]))
            else:
                actual = _file_fingerprint(Path(expected["path"]))
        except Exception as error:
            errors.append(
                f"{label} fingerprint unavailable: {type(error).__name__}: {error}"
            )
            continue
        if actual != expected:
            errors.append(f"{label} changed after plan creation")
    try:
        actual_state = _file_fingerprint(entry.instance_state_path)
    except Exception as error:
        errors.append(
            f"instance state fingerprint unavailable: {type(error).__name__}: {error}"
        )
    else:
        if actual_state["sha256"] != entry.instance_state_sha256:
            errors.append("instance state changed after plan creation")
    return errors


def _observed_instance_state_sha256(
    entry: EvalEntry,
) -> tuple[str | None, str | None]:
    """Re-read the exact public state and compare it with the admitted binding."""

    try:
        if entry.instance_state_path.is_symlink():
            raise RuntimeError("instance state path became a symlink")
        observed = _sha256(entry.instance_state_path)
    except (OSError, RuntimeError) as error:
        return None, f"instance state binding unavailable: {error}"
    if observed != entry.instance_state_sha256:
        return observed, "instance state SHA-256 changed after admission"
    return observed, None


def _resolve_eval_task_success(
    *,
    raw_success_confirmed: bool,
    final_result: Mapping[str, Any] | None,
    infrastructure_error: str | None,
) -> bool | None:
    """Keep raw action-trace success latched independently of infrastructure."""

    if raw_success_confirmed:
        return True
    if infrastructure_error is not None or final_result is None:
        return None
    value = final_result.get("task_success")
    return value if isinstance(value, bool) else None


def build_entry_argv(
    *,
    python: Path,
    repo_root: Path,
    output_dir: Path,
    behavior_repo: Path,
    behavior_python: Path,
    checkpoint: Path,
    activity_instance_id: int,
    public_seed: int,
    cuda_device: str,
    model: str | None,
    reasoning_effort: str,
    max_turns: int,
    planner_timeout_s: int,
    frozen_publication_root: Path,
    frozen_provenance_sha256: str,
    reviewed_memory_snapshot_sha256: str,
    recipe_catalog_sha256: str,
    max_wall_clock_s: int = DEFAULT_MAX_WALL_CLOCK_S,
    dashboard_event_sink: bool = False,
    task_name: str = TURNING_ON_RADIO_TASK_NAME,
    policy_checkpoint_binding_file: Path | None = None,
    vla_endpoint: str | None = None,
    resource_root: Path | None = None,
    resource_source_file: Path | None = None,
    source_snapshot_root: Path | None = None,
    source_snapshot_binding_sha256: str | None = None,
    runtime_isolation_root: Path | None = None,
    runtime_isolation_binding_sha256: str | None = None,
    runtime_isolation_namespace: str | None = None,
) -> tuple[str, ...]:
    """Build one fixed fresh-process Codex SDK invocation."""

    spec = get_task_spec(task_name)
    expected_instance = spec.instance_for_public_seed(public_seed, phase="eval")
    if activity_instance_id != expected_instance:
        raise ValueError(
            f"{spec.task_name} s{public_seed} requires instance "
            f"{expected_instance}, got {activity_instance_id}"
        )
    argv = [
        str(python),
        "-m",
        "robots.behavior.cli",
        "--env",
        "behavior",
        "--planner",
        "codex",
        "--suite",
        "behavior_2025_challenge",
        "--task",
        str(spec.task_index),
        "--task-name",
        spec.task_name,
        "--activity-definition-id",
        str(spec.activity_definition_id),
        "--activity-instance-id",
        str(activity_instance_id),
        "--scene-model",
        spec.scene_model,
        "--seed",
        str(BEHAVIOR_NATIVE_ENV_SEED),
        "--public-seed",
        str(public_seed),
        "--max-episode-steps",
        "24756",
        "--max-tool-calls",
        "350",
        "--max-wall-clock-s",
        str(max_wall_clock_s),
        "--behavior-repo",
        str(behavior_repo),
        "--behavior-python",
        str(behavior_python),
        "--policy-checkpoint",
        str(checkpoint),
        "--behavior-phase",
        "eval",
        "--behavior-frozen-publication-root",
        str(frozen_publication_root),
        "--behavior-frozen-provenance-sha256",
        frozen_provenance_sha256,
        "--behavior-reviewed-memory-snapshot-sha256",
        reviewed_memory_snapshot_sha256,
        "--behavior-recipe-catalog-sha256",
        recipe_catalog_sha256,
        "--reasoning-effort",
        reasoning_effort,
        "--cuda-device",
        str(cuda_device),
        "--max-turns",
        str(max_turns),
        "--planner-timeout-s",
        str(planner_timeout_s),
        "--output-dir",
        str(output_dir),
    ]
    if policy_checkpoint_binding_file is not None:
        argv.extend(
            [
                "--behavior-policy-checkpoint-binding-file",
                str(policy_checkpoint_binding_file),
            ]
        )
    if vla_endpoint is not None:
        argv.extend(["--vla-endpoint", vla_endpoint])
    if dashboard_event_sink:
        argv.append("--behavior-dashboard-event-sink")
    if (resource_root is None) != (resource_source_file is None):
        raise ValueError(
            "resource_root and resource_source_file must be provided together"
        )
    if resource_root is not None and resource_source_file is not None:
        argv.extend(
            [
                "--behavior-resource-root",
                str(resource_root),
                "--behavior-resource-source-file",
                str(resource_source_file),
            ]
        )
    if (source_snapshot_root is None) != (source_snapshot_binding_sha256 is None):
        raise ValueError(
            "source_snapshot_root and source_snapshot_binding_sha256 must be "
            "provided together"
        )
    if source_snapshot_root is not None and source_snapshot_binding_sha256 is not None:
        argv.extend(
            [
                "--behavior-source-snapshot-root",
                str(source_snapshot_root),
                "--behavior-source-snapshot-binding-sha256",
                source_snapshot_binding_sha256,
            ]
        )
    runtime_isolation_values = (
        runtime_isolation_root,
        runtime_isolation_binding_sha256,
        runtime_isolation_namespace,
    )
    if any(value is not None for value in runtime_isolation_values) and not all(
        value is not None for value in runtime_isolation_values
    ):
        raise ValueError(
            "runtime isolation root, binding SHA256, and namespace must be "
            "provided together"
        )
    if all(value is not None for value in runtime_isolation_values):
        argv.extend(
            [
                "--behavior-runtime-isolation-root",
                str(runtime_isolation_root),
                "--behavior-runtime-isolation-binding-sha256",
                str(runtime_isolation_binding_sha256),
                "--behavior-runtime-namespace",
                str(runtime_isolation_namespace),
            ]
        )
    if model:
        argv.extend(["--model", model])
    if "--no-driver" in argv or "--env-port" in argv:
        raise AssertionError("formal serial entries must own fresh env servers")
    if repo_root.resolve() != Path(repo_root).resolve():
        raise AssertionError("repo_root must be resolved")
    return tuple(argv)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return []
    return [value for value in values if isinstance(value, dict)]


def _raw_success_env_steps(action_trace_bytes: bytes) -> tuple[int, ...]:
    """Return receipt-style env steps from canonical zero-based trace steps.

    ``summarize_action_trace_success`` is the sole format authority and only
    recognizes ``info_done.success``.  Action trace ``step`` is zero-based,
    while runtime receipts identify the post-action environment step as
    ``step + 1``.
    """

    summary = summarize_action_trace_success(action_trace_bytes)
    if summary is None:
        return ()
    first_success_step = summary.get("first_success_step")
    if (
        isinstance(first_success_step, bool)
        or not isinstance(first_success_step, int)
        or first_success_step < 0
    ):
        return ()
    return (first_success_step + 1,)


def _bound_action_trace_success(
    action_trace_bytes: bytes,
    *,
    expected_run_nonce: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Bind raw success to the paired supervisor's one exact run nonce."""

    summary = summarize_action_trace_success(action_trace_bytes)
    if expected_run_nonce is None:
        return summary, None
    if _RUN_NONCE_RE.fullmatch(expected_run_nonce) is None:
        return None, "expected run nonce is invalid"
    binding_count = 0
    bound_run_nonce: str | None = None
    for line_number, line in enumerate(action_trace_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, f"action trace has malformed JSON at line {line_number}"
        if not isinstance(record, dict):
            return None, f"action trace has a non-object record at line {line_number}"
        if record.get("event") != "rpent_run_binding":
            continue
        binding_count += 1
        bound_run_nonce = record.get("run_nonce")
        if (
            record.get("attempt_index") != 1
            or not isinstance(bound_run_nonce, str)
            or _RUN_NONCE_RE.fullmatch(bound_run_nonce) is None
        ):
            return None, "action trace has an invalid run nonce binding"
    if binding_count != 1 or bound_run_nonce != expected_run_nonce:
        return None, "action trace run nonce binding mismatch"
    if summary is not None:
        summary = {**summary, "run_nonce": bound_run_nonce}
    return summary, None


def _proc_stat(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, Any] | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        fields = raw[raw.rfind(")") + 2 :].split()
        return {
            "pid": int(pid),
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None


def _owned_group_members(
    process: dict[str, Any], *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    """Find live members of one manifest-bound dedicated process session."""

    if process.get("managed") is not True:
        return ()
    pid = process.get("pid")
    pgid = process.get("pgid")
    sid = process.get("sid")
    start_ticks = process.get("start_ticks")
    if not all(isinstance(value, int) and value > 0 for value in (pid, pgid, sid)):
        return ()
    if not isinstance(start_ticks, int) or start_ticks <= 0:
        return ()
    if pid != pgid or pid != sid:
        return ()
    leader = _proc_stat(pid, proc_root=proc_root)
    if (
        leader is None
        or leader["state"] == "Z"
        or leader["pgid"] != pgid
        or leader["sid"] != sid
        or leader["start_ticks"] != start_ticks
    ):
        return ()
    return _matching_group_members(
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        proc_root=proc_root,
    )


def _matching_group_members(
    *, pgid: int, sid: int, start_ticks: int, proc_root: Path
) -> tuple[int, ...]:
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return ()
    members: list[int] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = _proc_stat(int(entry.name), proc_root=proc_root)
        if (
            stat is not None
            and stat["state"] != "Z"
            and stat["pgid"] == pgid
            and stat["sid"] == sid
            and stat["start_ticks"] >= start_ticks
        ):
            members.append(stat["pid"])
    return tuple(sorted(members))


def _unverified_group_members(
    process: dict[str, Any], *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    """Report possible recycled/leaderless members but never authorize signals."""

    pid = process.get("pid")
    pgid = process.get("pgid")
    sid = process.get("sid")
    start_ticks = process.get("start_ticks")
    if process.get("managed") is not True or not all(
        isinstance(value, int) and value > 0 for value in (pid, pgid, sid, start_ticks)
    ):
        return ()
    if pid != pgid or pid != sid or _owned_group_members(process, proc_root=proc_root):
        return ()
    return _matching_group_members(
        pgid=pgid,
        sid=sid,
        start_ticks=start_ticks,
        proc_root=proc_root,
    )


def _manifest_owned_groups(output_dir: Path) -> dict[str, tuple[int, ...]]:
    manifest = _read_json(output_dir / "run_manifest.json") or {}
    alive: dict[str, tuple[int, ...]] = {}
    for role, process in (manifest.get("processes") or {}).items():
        if not isinstance(process, dict):
            continue
        members = _owned_group_members(process)
        if members:
            alive[str(role)] = members
    return alive


def _manifest_unverified_groups(output_dir: Path) -> dict[str, tuple[int, ...]]:
    manifest = _read_json(output_dir / "run_manifest.json") or {}
    alive: dict[str, tuple[int, ...]] = {}
    for role, process in (manifest.get("processes") or {}).items():
        if not isinstance(process, dict):
            continue
        members = _unverified_group_members(process)
        if members:
            alive[str(role)] = members
    return alive


def _terminate_manifest_processes(
    output_dir: Path,
    *,
    timeout_s: float = 30.0,
    hard_deadline_monotonic_ns: int | None = None,
) -> dict[str, tuple[int, ...]]:
    """Stop only manifest-bound dedicated groups and report any survivors."""

    manifest = _read_json(output_dir / "run_manifest.json") or {}
    records = {
        str(role): process
        for role, process in (manifest.get("processes") or {}).items()
        if isinstance(process, dict)
    }
    groups: dict[int, dict[str, Any]] = {}
    for process in records.values():
        members = _owned_group_members(process)
        pgid = process.get("pgid")
        if members and isinstance(pgid, int) and pgid != os.getpgrp():
            groups[pgid] = process
    for pgid, process in tuple(groups.items()):
        if not _owned_group_members(process):
            groups.pop(pgid, None)
            continue
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(0.0, timeout_s)
    if hard_deadline_monotonic_ns is not None:
        deadline = min(deadline, hard_deadline_monotonic_ns / 1_000_000_000)
    while groups and time.monotonic() < deadline:
        groups = {
            pgid: process
            for pgid, process in groups.items()
            if _owned_group_members(process)
        }
        if groups:
            time.sleep(0.1)
    for pgid, process in tuple(groups.items()):
        if not _owned_group_members(process):
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    kill_deadline = (
        deadline
        if hard_deadline_monotonic_ns is not None
        else time.monotonic() + min(5.0, max(0.0, timeout_s))
    )
    while groups and time.monotonic() < kill_deadline:
        groups = {
            pgid: process
            for pgid, process in groups.items()
            if _owned_group_members(process)
        }
        if groups:
            time.sleep(0.1)
    return _manifest_owned_groups(output_dir)


def _write_verified_forced_cleanup_receipt(
    output_dir: Path,
    *,
    forced_groups: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Seal parent-owned proof without rewriting the crashed child manifest."""

    manifest_path = output_dir / "run_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise RuntimeError("forced-cleanup manifest is not a JSON object")
    if _manifest_owned_groups(output_dir):
        raise RuntimeError("cannot seal forced cleanup while managed groups are alive")
    if _manifest_unverified_groups(output_dir):
        raise RuntimeError("cannot seal forced cleanup with ambiguous group identity")
    process_bindings: dict[str, dict[str, Any]] = {}
    for role in sorted(forced_groups):
        process = (manifest.get("processes") or {}).get(role)
        if not isinstance(process, dict) or process.get("managed") is not True:
            raise RuntimeError(f"forced-cleanup role is not manifest-managed: {role}")
        process_bindings[role] = {
            field: process.get(field)
            for field in ("pid", "pgid", "sid", "start_ticks", "started_at")
        }
    receipt = {
        "schema_version": 1,
        "status": "verified",
        "completed_at": _utc_now(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "forced_groups": {
            role: list(forced_groups[role]) for role in sorted(forced_groups)
        },
        "process_bindings": process_bindings,
        "survivors": {},
        "ambiguous_groups": {},
    }
    _atomic_json(output_dir / FORCED_MANAGED_CLEANUP_FILENAME, receipt)
    return receipt


def _verified_forced_cleanup_receipt(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    receipt = _read_json(output_dir / FORCED_MANAGED_CLEANUP_FILENAME)
    if receipt is None:
        return None
    manifest_path = output_dir / "run_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError:
        return None
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "verified"
        or not isinstance(receipt.get("completed_at"), str)
        or not receipt["completed_at"]
        or receipt.get("manifest_path") != str(manifest_path.resolve())
        or receipt.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
        or receipt.get("survivors") != {}
        or receipt.get("ambiguous_groups") != {}
    ):
        return None
    bindings = receipt.get("process_bindings")
    forced = receipt.get("forced_groups")
    if not isinstance(bindings, dict) or not isinstance(forced, dict) or not forced:
        return None
    processes = manifest.get("processes")
    if not isinstance(processes, Mapping) or set(bindings) != set(forced):
        return None
    for role, binding in bindings.items():
        process = processes.get(role)
        members = forced.get(role)
        if (
            not isinstance(process, Mapping)
            or process.get("managed") is not True
            or not isinstance(binding, Mapping)
            or not isinstance(members, list)
            or not members
            or any(
                binding.get(field) != process.get(field)
                for field in ("pid", "pgid", "sid", "start_ticks", "started_at")
            )
        ):
            return None
    if _manifest_owned_groups(output_dir) or _manifest_unverified_groups(output_dir):
        return None
    return receipt


def _contained_output_file(output_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    if candidate.is_symlink():
        return None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output_dir.resolve())
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def _symbolic_recipe_exposes_checkpoint_identity(path: Path) -> bool:
    """Reject checkpoint names/paths while allowing generic strategy language."""

    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return True

    def exposes(value: Any, *, key: str | None = None) -> bool:
        normalized_key = str(key or "").lower()
        if normalized_key in {"checkpoint_name", "checkpoint_path"} or (
            normalized_key.endswith("_checkpoint_name")
            or normalized_key.endswith("_checkpoint_path")
        ):
            return True
        if isinstance(value, dict):
            return any(exposes(item, key=str(name)) for name, item in value.items())
        if isinstance(value, list):
            return any(exposes(item) for item in value)
        if not isinstance(value, str):
            return False
        lowered = value.lower()
        return bool(
            "/state_checkpoints/" in lowered
            or re.search(
                r"(?:^|[/\\])(?:state|tmp)_checkpoint_[a-z0-9_.-]+"
                r"(?:\.json)?(?:$|[/\\])",
                lowered,
            )
        )

    return any(exposes(record) for record in records)


def validate_instance_result(
    entry: EvalEntry,
    *,
    source_commit: str | None,
    subprocess_exit_code: int | None,
    timed_out: bool,
    expected_run_nonce: str | None = None,
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Classify one run from bound raw artifacts, never from exit code alone."""

    spec = get_task_spec(entry.task_name)
    recipe_tag = spec.tag(entry.public_seed)
    errors: list[str] = []
    final_result = _read_json(entry.output_dir / "final_result.json")
    behavior_result = _read_json(entry.output_dir / "behavior_result.json")
    manifest = _read_json(entry.output_dir / "run_manifest.json")
    _observed_state_sha256, state_binding_error = _observed_instance_state_sha256(entry)
    if state_binding_error is not None:
        errors.append(state_binding_error)
    if timed_out:
        errors.append("top-level RPent process timed out")
    if manifest is None:
        errors.append("missing or invalid run_manifest.json")
    else:
        try:
            run_tool_contract_version, _ = resolve_run_manifest_public_tool_contract(
                manifest
            )
        except ValueError:
            run_tool_contract_version = None
            errors.append("manifest public-tool contract is invalid")
        task = manifest.get("task") if isinstance(manifest.get("task"), dict) else {}
        expected = {
            "suite": "behavior_2025_challenge",
            "task": spec.task_index,
            "task_name": spec.task_name,
            "task_language": spec.task_language,
            "public_seed": entry.public_seed,
            "max_episode_steps": 24756,
        }
        for field, value in expected.items():
            if task.get(field) != value:
                errors.append(f"manifest task binding mismatch: {field}")
        native = (
            manifest.get("native_binding")
            if isinstance(manifest.get("native_binding"), dict)
            else {}
        )
        expected_native = {
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": entry.activity_instance_id,
            "activity_instance_dir": str(entry.instance_state_path.parent.resolve()),
            "scene_model": spec.scene_model,
            "env_seed": entry.seed,
        }
        for field, value in expected_native.items():
            if native.get(field) != value:
                errors.append(f"manifest native binding mismatch: {field}")
        if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
            errors.append("manifest schema version mismatch")
        protocol = (
            manifest.get("protocol")
            if isinstance(manifest.get("protocol"), dict)
            else {}
        )
        if protocol.get("behavior_phase") != "eval":
            errors.append("manifest BEHAVIOR phase is not eval")
        if protocol.get("public_seed") != entry.public_seed:
            errors.append("manifest public seed mismatch")
        if protocol.get("recipe_tag") != recipe_tag:
            errors.append("manifest recipe tag mismatch")
        if protocol.get("mapping_version") != spec.mapping_version:
            errors.append("manifest public-seed mapping version mismatch")
        if protocol.get("task_spec") != _task_spec_binding(spec):
            errors.append("manifest full TaskSpec binding mismatch")
        expected_task_identity = {
            "task_name": spec.task_name,
            "activity_definition_id": spec.activity_definition_id,
            "activity_instance_id": entry.activity_instance_id,
        }
        if protocol.get("task_identity") != expected_task_identity:
            errors.append("manifest composite task identity mismatch")
        if run_tool_contract_version != CURRENT_PUBLIC_TOOL_CONTRACT_VERSION:
            errors.append("manifest public-tool contract is not current")
        if protocol.get("agent_finish_registered") is not False:
            errors.append("formal Eval manifest unexpectedly registers Agent finish")
        if protocol.get("pi0_nav_pick_contract") != pi0_nav_pick_exact_chunk_contract():
            errors.append("manifest Pi0 exact-chunk contract mismatch")
        attempts = protocol.get("attempts")
        if not isinstance(attempts, dict) or attempts.get("max_attempts") != 1:
            errors.append("formal Eval manifest attempt policy mismatch")
        if isinstance(attempts, dict) and attempts.get("reset_registered") is not False:
            errors.append("formal Eval manifest unexpectedly registers reset")
        if entry.source_snapshot_binding is None:
            if manifest.get("commit") != source_commit:
                errors.append("manifest source commit mismatch")
            if manifest.get("worktree_dirty") is not False:
                errors.append("manifest source worktree is dirty")
        elif manifest.get("source_snapshot") != entry.source_snapshot_binding:
            errors.append("manifest sealed source snapshot binding mismatch")
        if (
            entry.runtime_isolation_binding is not None
            and manifest.get("runtime_isolation") != entry.runtime_isolation_binding
        ):
            errors.append("manifest campaign runtime isolation binding mismatch")
        forced_cleanup_receipt = _verified_forced_cleanup_receipt(
            entry.output_dir,
            manifest,
        )
        externally_stopped_roles = (
            set(forced_cleanup_receipt["process_bindings"])
            if forced_cleanup_receipt is not None
            else set()
        )
        if manifest.get("status") != "stopped" and forced_cleanup_receipt is None:
            errors.append("manifest lifecycle did not stop cleanly")
        if manifest.get("checkpoint") != str(entry.checkpoint.resolve()):
            errors.append("manifest policy checkpoint mismatch")
        if (
            entry.policy_checkpoint_binding is not None
            and manifest.get("policy_checkpoint") != entry.policy_checkpoint_binding
        ):
            errors.append("manifest shared checkpoint binding mismatch")
        if (
            entry.resource_source_binding is not None
            and manifest.get("resource_source") != entry.resource_source_binding
        ):
            errors.append("manifest pinned resource source binding mismatch")
        if manifest.get("gpu") != entry.cuda_device:
            errors.append("manifest GPU binding mismatch")
        if (
            entry.frozen_publication_binding is not None
            and manifest.get("frozen_eval_inputs") != entry.frozen_publication_binding
        ):
            errors.append("manifest frozen publication binding mismatch")
        if (
            entry.reviewed_repo_memory_binding is not None
            and manifest.get("reviewed_repo_memory")
            != entry.reviewed_repo_memory_binding
        ):
            errors.append("manifest reviewed Global Memory binding mismatch")
        if entry.reviewed_recipe_catalog_binding is not None and manifest.get(
            "reviewed_recipe_catalog"
        ) != _runtime_recipe_catalog_binding(entry.reviewed_recipe_catalog_binding):
            errors.append("manifest reviewed Recipe Catalog binding mismatch")
        for role, process in (manifest.get("processes") or {}).items():
            if not isinstance(process, dict):
                errors.append(f"invalid process manifest for {role}")
                continue
            stopped_at = process.get("stopped_at")
            if (
                process.get("managed") is True
                and role not in externally_stopped_roles
                and (not isinstance(stopped_at, str) or not stopped_at)
            ):
                errors.append(f"managed {role} process lacks stopped_at")
            members = _owned_group_members(process)
            if members:
                errors.append(
                    f"managed {role} process group is still alive: {list(members)}"
                )
            unverified = _unverified_group_members(process)
            if unverified:
                errors.append(
                    f"managed {role} group identity is ambiguous: {list(unverified)}"
                )
    legacy_state_artifacts = sorted(
        str(path)
        for path in (entry.output_dir / "state_checkpoints").rglob("*")
        if path.is_file()
    )
    if legacy_state_artifacts:
        errors.append("forbidden simulator-state checkpoint artifacts are present")

    def call_result_file(value: Any) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(entry.output_dir.resolve())
        except (OSError, ValueError):
            return None
        if path.is_symlink() or not resolved.is_file():
            return None
        return resolved

    tool_trace = _read_jsonl_records(entry.output_dir / "behavior_tool_trace.jsonl")
    action_trace_path = entry.output_dir / "behavior_action_trace.jsonl"
    try:
        action_trace_bytes = action_trace_path.read_bytes()
    except OSError:
        action_trace_bytes = b""
    raw_success_summary, nonce_binding_error = _bound_action_trace_success(
        action_trace_bytes,
        expected_run_nonce=expected_run_nonce,
    )
    if nonce_binding_error is not None:
        errors.append(nonce_binding_error)
    raw_success_steps = _raw_success_env_steps(action_trace_bytes)
    if isinstance(raw_success_summary, Mapping) and raw_success_summary.get(
        "first_success_step"
    ) != raw_success_summary.get("last_trace_step"):
        errors.append("action trace contains an action after official task success")
    executed_pi0_records = [
        record
        for record in tool_trace
        if record.get("tool") == "pi0_nav_pick"
        and not (
            isinstance(record.get("result"), dict)
            and record["result"].get("stop_reason") == "precondition_rejected"
        )
    ]
    successful_tool_indices = [
        index
        for index, record in enumerate(tool_trace)
        if isinstance(record.get("result"), dict)
        and record["result"].get("task_success") is True
    ]
    if successful_tool_indices and successful_tool_indices[0] != len(tool_trace) - 1:
        errors.append("tool trace contains a call after official task success")

    vla_root = entry.output_dir / "attempts" / recipe_tag / "attempt_001" / "vla_calls"
    call_dirs: list[Path] = []
    if vla_root.is_dir():
        call_dirs = sorted(
            path
            for path in vla_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and re.fullmatch(r"call_\d{3}", path.name)
        )
    expected_names = [f"call_{index:03d}" for index in range(1, len(call_dirs) + 1)]
    if [path.name for path in call_dirs] != expected_names:
        errors.append("VLA call artifact sequence is not contiguous")
    total_vla_chunks = 0
    total_vla_env_steps = 0
    previous_episode_env_steps = 0
    for call_index, call_dir in enumerate(call_dirs, start=1):
        call_record = _read_json(call_dir / "pi0_nav_pick_call.json")
        if call_record is None:
            errors.append(f"missing or invalid VLA call record: {call_dir.name}")
            continue
        claimed = call_record.get("claimed_at_unix_s")
        completed = call_record.get("completed_at_unix_s")
        result_path = call_result_file(call_record.get("result_path"))
        expected_result_path = call_result_file(
            str(call_dir / "pi0_nav_pick_result.json")
        )
        instruction = call_record.get("instruction")
        requested_chunks = call_record.get("requested_chunks")
        trace_record = (
            executed_pi0_records[call_index - 1]
            if call_index <= len(executed_pi0_records)
            else None
        )
        trace_input = (
            trace_record.get("input") if isinstance(trace_record, dict) else None
        )
        trace_result = (
            trace_record.get("result") if isinstance(trace_record, dict) else None
        )
        schema_version = call_record.get("schema_version")
        if schema_version != PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION:
            errors.append(
                f"unsupported VLA call artifact schema_version "
                f"{schema_version!r}: {call_dir.name}"
            )
            continue
        if (
            call_record.get("name") != "pi0_nav_pick"
            or "max_chunks" in call_record
            or "max_total_vla_chunks" in call_record
            or isinstance(requested_chunks, bool)
            or not isinstance(requested_chunks, int)
            or requested_chunks < 1
            or not isinstance(call_record.get("request_id"), str)
            or re.fullmatch(r"[0-9a-f]{64}", call_record["request_id"]) is None
            or call_record.get("status") != "completed"
            or not isinstance(instruction, str)
            or not instruction.strip()
            or not isinstance(call_record.get("run_nonce"), str)
            or not call_record["run_nonce"]
            or not isinstance(call_record.get("attempt_nonce"), str)
            or not call_record["attempt_nonce"]
            or call_record.get("attempt_index") != 1
            or call_record.get("global_vla_invocations") != call_index
            or not isinstance(claimed, (int, float))
            or isinstance(claimed, bool)
            or not math.isfinite(float(claimed))
            or float(claimed) <= 0.0
            or not isinstance(completed, (int, float))
            or isinstance(completed, bool)
            or not math.isfinite(float(completed))
            or float(completed) < float(claimed)
            or not isinstance(call_record.get("outcome"), str)
            or not call_record["outcome"]
            or not isinstance(call_record.get("task_success"), bool)
            or not isinstance(call_record.get("local_grasp_success"), bool)
            or result_path is None
            or expected_result_path is None
            or result_path != expected_result_path
            or call_record.get("result_sha256") != _sha256(result_path)
            or not isinstance(trace_input, dict)
            or trace_input.get("instruction") != instruction
            or isinstance(trace_input.get("chunks"), bool)
            or not isinstance(trace_input.get("chunks"), int)
            or trace_input.get("chunks") != requested_chunks
            or not isinstance(trace_result, dict)
        ):
            errors.append(f"invalid VLA call record: {call_dir.name}")
            continue
        result = _read_json(result_path)
        result_requested_chunks = (
            result.get("requested_chunks") if result is not None else None
        )
        chunks_used = result.get("chunks_used") if result is not None else None
        full_chunks = result.get("full_chunks_executed") if result is not None else None
        vla_env_steps = result.get("vla_env_steps_used") if result is not None else None
        env_steps_used = result.get("env_steps_used") if result is not None else None
        episode_env_steps = (
            result.get("total_env_steps") if result is not None else None
        )
        global_chunks = result.get("global_vla_chunks") if result is not None else None
        result_task_success = result.get("task_success") if result is not None else None
        result_terminated = result.get("terminated") if result is not None else None
        result_truncated = result.get("truncated") if result is not None else None
        result_stop_reason = result.get("stop_reason") if result is not None else None
        exact_requested_accounting = bool(
            result_requested_chunks == requested_chunks
            and isinstance(chunks_used, int)
            and not isinstance(chunks_used, bool)
            and chunks_used == requested_chunks
            and isinstance(full_chunks, int)
            and not isinstance(full_chunks, bool)
            and full_chunks == requested_chunks
            and isinstance(vla_env_steps, int)
            and not isinstance(vla_env_steps, bool)
            and vla_env_steps == requested_chunks * 32
            and result_stop_reason
            in (
                {"requested_chunks_completed", "official_task_success"}
                if result_task_success is True
                else {"requested_chunks_completed"}
            )
        )
        hard_terminal_reason = bool(
            (
                result_terminated is True
                and result_truncated is False
                and result_stop_reason == "terminated"
            )
            or (
                result_terminated is False
                and result_truncated is True
                and result_stop_reason == "truncated"
            )
        )
        hard_terminal_accounting = bool(
            call_index == len(call_dirs)
            and result_requested_chunks == requested_chunks
            and isinstance(chunks_used, int)
            and not isinstance(chunks_used, bool)
            and chunks_used >= 1
            and chunks_used <= requested_chunks
            and isinstance(full_chunks, int)
            and not isinstance(full_chunks, bool)
            and isinstance(vla_env_steps, int)
            and not isinstance(vla_env_steps, bool)
            and (
                (full_chunks == chunks_used and vla_env_steps == full_chunks * 32)
                or (
                    full_chunks == chunks_used - 1
                    and full_chunks * 32 + 1 <= vla_env_steps <= chunks_used * 32 - 1
                )
            )
            and hard_terminal_reason
        )
        official_success_accounting = bool(
            call_index == len(call_dirs)
            and result_task_success is True
            and result_requested_chunks == requested_chunks
            and isinstance(chunks_used, int)
            and not isinstance(chunks_used, bool)
            and 1 <= chunks_used <= requested_chunks
            and isinstance(full_chunks, int)
            and not isinstance(full_chunks, bool)
            and isinstance(vla_env_steps, int)
            and not isinstance(vla_env_steps, bool)
            and (
                (full_chunks == chunks_used and vla_env_steps == full_chunks * 32)
                or (
                    full_chunks == chunks_used - 1
                    and full_chunks * 32 + 1 <= vla_env_steps <= chunks_used * 32 - 1
                )
            )
            and result_stop_reason == "official_task_success"
            and result.get("exact_requested_chunks_completed") is False
            and result.get("primitive_success") is True
            and not result.get("error")
        )
        raw_success_receipt_valid = True
        if result_task_success is True and result is not None:
            receipt = result.get("official_success_receipt")
            receipt_validation = validate_terminal_success_receipt(
                tool_name="pi0_nav_pick",
                step=(
                    trace_record.get("step") if isinstance(trace_record, dict) else None
                ),
                result=result,
                output_dir=entry.output_dir,
            )
            raw_success_receipt_valid = bool(
                receipt_validation.valid
                and isinstance(receipt, dict)
                and receipt.get("run_nonce") == call_record.get("run_nonce")
                and receipt.get("attempt_nonce") == call_record.get("attempt_nonce")
                and receipt.get("attempt_index") == call_record.get("attempt_index")
                and isinstance(receipt.get("env_step"), int)
                and not isinstance(receipt.get("env_step"), bool)
                and raw_success_steps
                and receipt.get("env_step") == raw_success_steps[0]
                and trace_result.get("official_success_receipt") == receipt
            )
        elif result is not None and result.get("official_success_receipt") is not None:
            raw_success_receipt_valid = False
        trace_bound_fields = (
            "requested_chunks",
            "chunks_used",
            "full_chunks_executed",
            "vla_env_steps_used",
            "task_success",
            "terminated",
            "truncated",
            "stop_reason",
            "global_vla_invocations",
            "global_vla_chunks",
            "total_env_steps",
        )
        if (
            result is None
            or "max_chunks" in result
            or "max_total_vla_chunks" in result
            or isinstance(result_requested_chunks, bool)
            or not isinstance(result_requested_chunks, int)
            or result_requested_chunks != requested_chunks
            or isinstance(chunks_used, bool)
            or not isinstance(chunks_used, int)
            or chunks_used < 0
            or isinstance(full_chunks, bool)
            or not isinstance(full_chunks, int)
            or isinstance(vla_env_steps, bool)
            or not isinstance(vla_env_steps, int)
            or not (
                exact_requested_accounting
                or hard_terminal_accounting
                or official_success_accounting
            )
            or isinstance(env_steps_used, bool)
            or not isinstance(env_steps_used, int)
            or env_steps_used < vla_env_steps
            or isinstance(episode_env_steps, bool)
            or not isinstance(episode_env_steps, int)
            or episode_env_steps < previous_episode_env_steps
            or episode_env_steps < env_steps_used
            or result.get("action_horizon") != 32
            or result.get("required_action_shape") != [32, 23]
            or result.get("global_vla_invocations") != call_index
            or not isinstance(result_task_success, bool)
            or result_task_success is not call_record.get("task_success")
            or not isinstance(result_terminated, bool)
            or not isinstance(result_truncated, bool)
            or not isinstance(result_stop_reason, str)
            or not result_stop_reason
            or result_stop_reason != call_record.get("outcome")
            or result.get("run_nonce") != call_record.get("run_nonce")
            or result.get("attempt_nonce") != call_record.get("attempt_nonce")
            or result.get("attempt_index") != call_record.get("attempt_index")
            or any(
                trace_result.get(field) != result.get(field)
                for field in trace_bound_fields
            )
            or not raw_success_receipt_valid
        ):
            errors.append(f"invalid exact-N chunk accounting: {call_dir.name}")
            continue
        total_vla_chunks += chunks_used
        total_vla_env_steps += vla_env_steps
        previous_episode_env_steps = episode_env_steps
        if global_chunks != total_vla_chunks:
            errors.append(f"invalid cumulative VLA chunk accounting: {call_dir.name}")
        if episode_env_steps < total_vla_env_steps:
            errors.append(f"invalid episode environment total: {call_dir.name}")
    tool_names = [record.get("tool") for record in tool_trace]
    trace_vla_invocations: list[int] = []
    trace_vla_chunks: list[int] = []
    trace_env_steps: list[int] = []
    if not tool_names:
        errors.append("missing or invalid behavior_tool_trace.jsonl")
    else:
        allowed_tools = set(PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION])
        if "reset" in tool_names:
            errors.append("formal Eval trace contains reset")
        if any(name not in allowed_tools for name in tool_names):
            errors.append("tool trace contains an unregistered BEHAVIOR tool")
        artifact_bound_calls = sum(
            1
            for record in tool_trace
            if record.get("tool") == "pi0_nav_pick"
            and not (
                isinstance(record.get("result"), dict)
                and record["result"].get("stop_reason") == "precondition_rejected"
            )
        )
        if len(call_dirs) != artifact_bound_calls:
            errors.append(
                "VLA artifact count does not match executed pi0_nav_pick calls"
            )
        for record in tool_trace:
            result = record.get("result")
            invocation_value = (
                result.get("global_vla_invocations")
                if isinstance(result, dict)
                else None
            )
            chunk_value = (
                result.get("global_vla_chunks") if isinstance(result, dict) else None
            )
            env_step_value = (
                result.get("total_env_steps") if isinstance(result, dict) else None
            )
            if (
                not isinstance(invocation_value, int)
                or isinstance(invocation_value, bool)
                or invocation_value < 0
            ):
                errors.append("tool trace lacks a valid global VLA invocation counter")
                trace_vla_invocations = []
                break
            if (
                not isinstance(chunk_value, int)
                or isinstance(chunk_value, bool)
                or chunk_value < 0
            ):
                errors.append("tool trace lacks a valid global VLA chunk counter")
                trace_vla_chunks = []
                break
            if (
                not isinstance(env_step_value, int)
                or isinstance(env_step_value, bool)
                or env_step_value < 0
            ):
                errors.append("tool trace lacks a valid environment-step counter")
                trace_env_steps = []
                break
            trace_vla_invocations.append(invocation_value)
            trace_vla_chunks.append(chunk_value)
            trace_env_steps.append(env_step_value)
        if trace_vla_invocations and any(
            current < previous
            for previous, current in zip(
                trace_vla_invocations, trace_vla_invocations[1:]
            )
        ):
            errors.append("global VLA invocation counter regressed in tool trace")
        if trace_vla_invocations and trace_vla_invocations[-1] != len(call_dirs):
            errors.append("global VLA invocation counter does not match VLA artifacts")
        if trace_vla_chunks and any(
            current < previous
            for previous, current in zip(trace_vla_chunks, trace_vla_chunks[1:])
        ):
            errors.append("global VLA chunk counter regressed in tool trace")
        if trace_vla_chunks and trace_vla_chunks[-1] != total_vla_chunks:
            errors.append("global VLA chunk counter does not match VLA artifacts")
        if trace_env_steps and any(
            current < previous
            for previous, current in zip(trace_env_steps, trace_env_steps[1:])
        ):
            errors.append("environment-step counter regressed in tool trace")
        if trace_env_steps and trace_env_steps[-1] < total_vla_env_steps:
            errors.append("environment-step total is smaller than executed VLA actions")
    if final_result is None:
        errors.append("missing or invalid final_result.json")
        claimed_recipe = (
            behavior_result.get("recipe_path")
            if isinstance(behavior_result, dict)
            else None
        )
        if claimed_recipe not in {None, ""} or any(
            path.is_file() for path in entry.output_dir.glob("recipe_*.jsonl")
        ):
            errors.append("non-complete run published a symbolic recipe")
        memory_root = entry.output_dir / "memory"
        if memory_root.is_dir() and any(
            path.is_file() and not path.is_symlink() for path in memory_root.rglob("*")
        ):
            errors.append("non-complete run published task memory")
        if timed_out or subprocess_exit_code not in {0, None}:
            return "run_error", errors, None
        return "incomplete", errors, None
    if subprocess_exit_code not in {0, None}:
        errors.append("top-level RPent process returned nonzero")
    if final_result.get("runtime_cleanup") != "complete":
        errors.append("runtime cleanup did not complete")
    if (
        final_result.get("error") is not None
        or final_result.get("run_status") == "error"
    ):
        errors.append("RPent final_result reports an execution error")
    task_success = final_result.get("task_success")
    if task_success not in {True, False, None}:
        errors.append("task_success is not boolean or null")
    expected_source = (
        'info["done"]["success"]' if task_success in {True, False} else None
    )
    if final_result.get("official_success_source") != expected_source:
        errors.append("invalid official success source")
    root_success_receipt = _read_json(
        entry.output_dir / "official_success_receipt.json"
    )
    valid_official_receipt = next(
        (
            result.get("official_success_receipt")
            for record in reversed(tool_trace)
            if isinstance(record, dict)
            and isinstance((result := record.get("result")), dict)
            and isinstance(result.get("official_success_receipt"), dict)
            and validate_terminal_success_receipt(
                tool_name=str(record.get("tool", "")),
                step=record.get("step"),
                result=result,
                output_dir=entry.output_dir,
            ).valid
            and result.get("official_success_receipt") == root_success_receipt
            and raw_success_steps
            and result["official_success_receipt"].get("env_step")
            == raw_success_steps[0]
            and result.get("run_nonce")
            == result["official_success_receipt"].get("run_nonce")
            and (
                expected_run_nonce is None
                or result["official_success_receipt"].get("run_nonce")
                == expected_run_nonce
            )
            and result.get("attempt_nonce")
            == result["official_success_receipt"].get("attempt_nonce")
        ),
        None,
    )
    raw_success_verified = bool(
        task_success is True and valid_official_receipt is not None
    )
    if behavior_result is None:
        errors.append("missing or invalid behavior_result.json")
    else:
        if (
            behavior_result.get("task_success") is not task_success
            or behavior_result.get("success") is not task_success
        ):
            errors.append("behavior_result task success mismatch")
        result_vla_invocations = behavior_result.get("global_vla_invocations")
        if (
            not isinstance(result_vla_invocations, int)
            or isinstance(result_vla_invocations, bool)
            or result_vla_invocations < 0
        ):
            errors.append("behavior_result has an invalid global VLA invocation count")
        elif result_vla_invocations != len(call_dirs):
            errors.append(
                "behavior_result global VLA invocation count does not match artifacts"
            )
    if task_success is True:
        if not raw_success_verified:
            errors.append("successful run lacks a valid raw official-success receipt")
    audit = _read_json(entry.output_dir / f"{recipe_tag}.json")
    if audit is None:
        errors.append("missing or invalid BEHAVIOR audit")
    else:
        audit_vla_invocations = audit.get("global_vla_invocations")
        audit_vla_chunks = audit.get("global_vla_chunks")
        audit_env_steps = audit.get("total_env_steps")
        if (
            not isinstance(audit_vla_invocations, int)
            or isinstance(audit_vla_invocations, bool)
            or audit_vla_invocations < 0
        ):
            errors.append("BEHAVIOR audit has an invalid global VLA invocation count")
        elif audit_vla_invocations != len(call_dirs):
            errors.append(
                "BEHAVIOR audit global VLA invocation count does not match artifacts"
            )
        if (
            not isinstance(audit_vla_chunks, int)
            or isinstance(audit_vla_chunks, bool)
            or audit_vla_chunks < 0
        ):
            errors.append("BEHAVIOR audit has an invalid global VLA chunk count")
        elif audit_vla_chunks != total_vla_chunks:
            errors.append(
                "BEHAVIOR audit global VLA chunk count does not match artifacts"
            )
        if (
            not isinstance(audit_env_steps, int)
            or isinstance(audit_env_steps, bool)
            or audit_env_steps < 0
        ):
            errors.append("BEHAVIOR audit has an invalid environment-step total")
        elif audit_env_steps < total_vla_env_steps:
            errors.append(
                "BEHAVIOR audit environment-step total is smaller than VLA actions"
            )
        if trace_env_steps and audit_env_steps != trace_env_steps[-1]:
            errors.append(
                "BEHAVIOR audit environment-step total does not match tool trace"
            )
    claimed_recipe = (
        behavior_result.get("recipe_path")
        if isinstance(behavior_result, dict)
        else None
    )
    published_recipes = {
        path.resolve()
        for path in entry.output_dir.glob("recipe_*.jsonl")
        if path.is_file() and not path.is_symlink()
    }
    if claimed_recipe not in {None, ""}:
        claimed_path = _contained_output_file(entry.output_dir, claimed_recipe)
        if claimed_path is None:
            errors.append("behavior_result published an invalid symbolic recipe path")
        else:
            published_recipes.add(claimed_path)
    memory_root = entry.output_dir / "memory"
    published_memory = bool(
        memory_root.is_dir()
        and any(
            path.is_file() and not path.is_symlink() for path in memory_root.rglob("*")
        )
    )
    if published_recipes:
        errors.append("formal Eval published a symbolic recipe")
    if published_memory:
        errors.append("formal Eval published task memory")
    if any(
        _symbolic_recipe_exposes_checkpoint_identity(path) for path in published_recipes
    ):
        errors.append("symbolic recipe exposes a robot checkpoint name or path")
    if errors:
        if (
            timed_out
            or subprocess_exit_code not in {0, None}
            or final_result.get("error")
        ):
            return "run_error", errors, final_result
        return "incomplete", errors, final_result
    if task_success is True:
        return "passed", errors, final_result
    if task_success is False:
        return "task_failed", errors, final_result
    return "incomplete", ["official task_success is missing"], final_result


def _terminate_top_process(
    process: subprocess.Popen[Any],
    *,
    timeout_s: float = 60.0,
    identity: Mapping[str, Any] | None = None,
    hard_deadline_monotonic_ns: int | None = None,
) -> None:
    if process.poll() is not None:
        return
    bound_identity = (
        dict(identity) if identity is not None else _capture_process_identity(process)
    )
    if bound_identity is None:
        return
    deadline = time.monotonic() + max(0.0, timeout_s)
    if hard_deadline_monotonic_ns is not None:
        deadline = min(deadline, hard_deadline_monotonic_ns / 1_000_000_000)
    if not _signal_bound_process_group(
        process,
        bound_identity,
        signal.SIGTERM,
    ):
        return
    try:
        process.wait(timeout=min(30.0, max(0.0, deadline - time.monotonic())))
        return
    except subprocess.TimeoutExpired:
        pass
    if not _signal_bound_process_group(
        process,
        bound_identity,
        signal.SIGKILL,
    ):
        return
    remaining = max(0.0, deadline - time.monotonic())
    if remaining <= 0.0:
        return
    try:
        process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        return


def _capture_process_identity(
    process: subprocess.Popen[Any],
) -> dict[str, Any] | None:
    stat = _proc_stat(int(process.pid))
    if (
        stat is None
        or stat["state"] == "Z"
        or stat["pid"] != stat["pgid"]
        or stat["pid"] != stat["sid"]
        or stat["start_ticks"] <= 0
        or stat["pgid"] == os.getpgrp()
    ):
        return None
    return {
        "pid": stat["pid"],
        "pgid": stat["pgid"],
        "sid": stat["sid"],
        "start_ticks": stat["start_ticks"],
    }


def _process_identity_matches(
    process: subprocess.Popen[Any],
    identity: Mapping[str, Any],
) -> bool:
    if process.poll() is not None:
        return False
    expected = (
        identity.get("pid"),
        identity.get("pgid"),
        identity.get("sid"),
        identity.get("start_ticks"),
    )
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in expected
    ):
        return False
    pid, pgid, sid, start_ticks = expected
    if process.pid != pid or pid != pgid or pid != sid or pgid == os.getpgrp():
        return False
    current = _proc_stat(pid)
    return bool(
        current is not None
        and current["state"] != "Z"
        and current["pid"] == pid
        and current["pgid"] == pgid
        and current["sid"] == sid
        and current["start_ticks"] == start_ticks
    )


def _signal_bound_process_group(
    process: subprocess.Popen[Any],
    identity: Mapping[str, Any],
    sig: signal.Signals,
) -> bool:
    """Signal only after a fresh four-field identity revalidation."""

    if not _process_identity_matches(process, identity):
        return False
    try:
        os.killpg(int(identity["pgid"]), sig)
    except ProcessLookupError:
        return False
    return True


def _instance_child_process_payload(
    entry: EvalEntry,
    process: subprocess.Popen[Any],
    *,
    state: str,
    started_at: str,
    action_deadline_s: int,
    cleanup_deadline_s: int,
    instance_timeout_s: int,
    source_snapshot_root: Path | None,
    source_snapshot_binding_sha256: str | None,
    deadline_binding: InstanceDeadlineBinding,
    expected_run_nonce: str | None = None,
) -> dict[str, Any]:
    if state not in {"running", "exited"}:
        raise ValueError("instance child process state is invalid")
    child = _proc_stat(process.pid)
    runner = _proc_stat(os.getpid())
    if (
        child is None
        or child["pid"] != child["pgid"]
        or child["pid"] != child["sid"]
        or child["start_ticks"] <= 0
        or runner is None
        or runner["start_ticks"] <= 0
    ):
        raise RuntimeError("could not bind the nested instance child process")
    argv_sha256 = hashlib.sha256(
        json.dumps(
            list(entry.argv),
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    spec = get_task_spec(entry.task_name)
    payload = {
        "schema_version": 1,
        "state": state,
        "pid": child["pid"],
        "pgid": child["pgid"],
        "sid": child["sid"],
        "start_ticks": child["start_ticks"],
        "runner_pid": runner["pid"],
        "runner_pgid": runner["pgid"],
        "runner_sid": runner["sid"],
        "runner_start_ticks": runner["start_ticks"],
        "task_name": spec.task_name,
        "public_seed": entry.public_seed,
        "activity_instance_id": entry.activity_instance_id,
        "entry_output_dir": str(entry.output_dir.resolve()),
        "source_snapshot_root": (
            str(source_snapshot_root.resolve())
            if source_snapshot_root is not None
            else None
        ),
        "source_snapshot_binding_sha256": source_snapshot_binding_sha256,
        "cuda_device": entry.cuda_device,
        "argv_sha256": argv_sha256,
        "started_at": started_at,
        "updated_at": _utc_now(),
        "action_deadline_s": action_deadline_s,
        "cleanup_deadline_s": cleanup_deadline_s,
        "instance_timeout_s": instance_timeout_s,
        **deadline_binding.as_dict(),
    }
    if expected_run_nonce is not None:
        if _RUN_NONCE_RE.fullmatch(expected_run_nonce) is None:
            raise ValueError("expected run nonce must be 32 lowercase hex characters")
        payload["expected_run_nonce"] = expected_run_nonce
    return payload


def _write_instance_child_process_receipt(
    path: Path,
    payload: dict[str, Any],
) -> None:
    if path.is_symlink():
        raise RuntimeError("instance child process receipt must not be a symlink")
    _atomic_json(path, payload)


def _remaining_whole_seconds(deadline_monotonic_ns: int, now_ns: int) -> int:
    """Floor a remaining budget so a nested duration cannot exceed its deadline."""

    return max(0, (deadline_monotonic_ns - now_ns) // 1_000_000_000)


def _replace_argv_int_option(
    argv: tuple[str, ...],
    option: str,
    value: int,
) -> tuple[str, ...]:
    updated = list(argv)
    try:
        index = updated.index(option)
    except ValueError as error:
        raise RuntimeError(f"nested Eval argv is missing {option}") from error
    if index + 1 >= len(updated):
        raise RuntimeError(f"nested Eval argv has no value for {option}")
    updated[index + 1] = str(value)
    return tuple(updated)


def _admit_entry_action_budget(
    entry: EvalEntry,
    *,
    deadline_binding: InstanceDeadlineBinding,
    configured_planner_timeout_s: int,
    admitted_at_monotonic_ns: int,
) -> tuple[EvalEntry, dict[str, int]]:
    action_remaining_s = _remaining_whole_seconds(
        deadline_binding.action_deadline_monotonic_ns,
        admitted_at_monotonic_ns,
    )
    cleanup_remaining_s = _remaining_whole_seconds(
        deadline_binding.cleanup_deadline_monotonic_ns,
        admitted_at_monotonic_ns,
    )
    hard_remaining_s = _remaining_whole_seconds(
        deadline_binding.hard_deadline_monotonic_ns,
        admitted_at_monotonic_ns,
    )
    if action_remaining_s <= 0:
        raise TimeoutError("absolute action deadline expired before child admission")
    planner_remaining_s = min(configured_planner_timeout_s, action_remaining_s)
    argv = _replace_argv_int_option(
        entry.argv,
        "--max-wall-clock-s",
        action_remaining_s,
    )
    argv = _replace_argv_int_option(
        argv,
        "--planner-timeout-s",
        planner_remaining_s,
    )
    admitted = {
        "admitted_at_monotonic_ns": admitted_at_monotonic_ns,
        "planner_timeout_s": planner_remaining_s,
        "max_wall_clock_s": action_remaining_s,
        "cleanup_remaining_s": cleanup_remaining_s,
        "instance_timeout_remaining_s": hard_remaining_s,
    }
    return replace(entry, argv=argv), admitted


def _run_entry(
    entry: EvalEntry,
    *,
    repo_root: Path,
    log_stream: BinaryIO,
    timeout_s: int,
    cleanup_deadline_s: int,
    action_deadline_s: int,
    source_snapshot_root: Path | None,
    source_snapshot_binding_sha256: str | None,
    deadline_binding: InstanceDeadlineBinding,
    externally_bound_deadline: bool = False,
    expected_run_nonce: str | None = None,
) -> tuple[int | None, bool]:
    if entry.output_dir.exists() and any(entry.output_dir.iterdir()):
        raise RuntimeError(f"entry output directory is not empty: {entry.output_dir}")
    if entry.output_dir.exists():
        entry.output_dir.rmdir()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["RPENT_REPO_ROOT"] = str(repo_root)
    if expected_run_nonce is not None:
        if _RUN_NONCE_RE.fullmatch(expected_run_nonce) is None:
            raise ValueError("expected run nonce must be 32 lowercase hex characters")
        environment[_EXPECTED_RUN_NONCE_ENV] = expected_run_nonce
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else str(repo_root) + os.pathsep + existing_pythonpath
    )
    child_started_at = _utc_now()
    process = subprocess.Popen(
        entry.argv,
        cwd=repo_root,
        env=environment,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    receipt_path = entry.output_dir.parent / INSTANCE_CHILD_PROCESS_FILENAME
    child_identity = _capture_process_identity(process)
    if child_identity is None:
        raise RuntimeError("could not bind the nested instance child identity")
    try:
        running_receipt = _instance_child_process_payload(
            entry,
            process,
            state="running",
            started_at=child_started_at,
            action_deadline_s=action_deadline_s,
            cleanup_deadline_s=cleanup_deadline_s,
            instance_timeout_s=timeout_s,
            source_snapshot_root=source_snapshot_root,
            source_snapshot_binding_sha256=source_snapshot_binding_sha256,
            deadline_binding=deadline_binding,
            expected_run_nonce=expected_run_nonce,
        )
        _write_instance_child_process_receipt(receipt_path, running_receipt)
    except BaseException:
        _terminate_top_process(
            process,
            timeout_s=30.0,
            identity=child_identity,
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        raise
    returncode: int | None = None
    timed_out = False
    try:
        returncode = process.wait(
            timeout=max(
                0.0,
                (deadline_binding.cleanup_deadline_monotonic_ns - time.monotonic_ns())
                / 1_000_000_000,
            )
        )
        timed_out = _is_external_action_deadline_sigterm(
            returncode=returncode,
            externally_bound_deadline=externally_bound_deadline,
            deadline_binding=deadline_binding,
            observed_at_monotonic_ns=time.monotonic_ns(),
        )
        return returncode, timed_out
    except subprocess.TimeoutExpired:
        remaining = max(
            0.0,
            (deadline_binding.hard_deadline_monotonic_ns - time.monotonic_ns())
            / 1_000_000_000,
        )
        _terminate_manifest_processes(
            entry.output_dir,
            timeout_s=min(30.0, remaining / 3.0),
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        remaining = max(
            0.0,
            (deadline_binding.hard_deadline_monotonic_ns - time.monotonic_ns())
            / 1_000_000_000,
        )
        _terminate_top_process(
            process,
            timeout_s=min(30.0, remaining / 2.0),
            identity=running_receipt,
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        remaining = max(
            0.0,
            (deadline_binding.hard_deadline_monotonic_ns - time.monotonic_ns())
            / 1_000_000_000,
        )
        _terminate_manifest_processes(
            entry.output_dir,
            timeout_s=min(30.0, remaining),
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        returncode = process.returncode
        timed_out = True
        return returncode, timed_out
    except BaseException:
        remaining = max(
            0.0,
            (deadline_binding.hard_deadline_monotonic_ns - time.monotonic_ns())
            / 1_000_000_000,
        )
        _terminate_manifest_processes(
            entry.output_dir,
            timeout_s=min(30.0, remaining / 3.0),
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        remaining = max(
            0.0,
            (deadline_binding.hard_deadline_monotonic_ns - time.monotonic_ns())
            / 1_000_000_000,
        )
        _terminate_top_process(
            process,
            timeout_s=min(30.0, remaining / 2.0),
            identity=running_receipt,
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        remaining = max(
            0.0,
            (deadline_binding.hard_deadline_monotonic_ns - time.monotonic_ns())
            / 1_000_000_000,
        )
        _terminate_manifest_processes(
            entry.output_dir,
            timeout_s=min(30.0, remaining),
            hard_deadline_monotonic_ns=(deadline_binding.hard_deadline_monotonic_ns),
        )
        raise
    finally:
        final_receipt = {
            **running_receipt,
            "state": "exited" if process.poll() is not None else "running",
            "updated_at": _utc_now(),
            "returncode": returncode,
            "timed_out": timed_out,
        }
        _write_instance_child_process_receipt(receipt_path, final_receipt)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one BEHAVIOR task's formal public Eval instances serially with "
            "fresh Codex/env processes, one controller-owned VLA, and no "
            "automatic retry."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cuda-device", required=True)
    parser.add_argument("--task-name", default=TURNING_ON_RADIO_TASK_NAME)
    parser.add_argument(
        "--public-seed",
        action="append",
        type=int,
        default=None,
        help=(
            "task-local Eval public seed; repeat to run a subset in the given "
            "order (default: every Eval seed registered by TaskSpec)"
        ),
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="xhigh",
    )
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--planner-timeout-s",
        type=int,
        default=DEFAULT_MAX_WALL_CLOCK_S,
    )
    parser.add_argument(
        "--max-wall-clock-s",
        type=int,
        default=DEFAULT_MAX_WALL_CLOCK_S,
        help="child planner/runtime action budget in seconds",
    )
    parser.add_argument(
        "--cleanup-deadline-s",
        type=int,
        default=DEFAULT_CLEANUP_DEADLINE_S,
        help=(
            "absolute seconds after child launch when owned-process cleanup must begin"
        ),
    )
    parser.add_argument(
        "--instance-timeout-s",
        type=int,
        default=DEFAULT_INSTANCE_TIMEOUT_S,
        help="absolute per-instance hard watchdog; cannot exceed 7200 seconds",
    )
    parser.add_argument("--instance-started-monotonic-ns", type=int, default=None)
    parser.add_argument("--action-deadline-monotonic-ns", type=int, default=None)
    parser.add_argument("--cleanup-deadline-monotonic-ns", type=int, default=None)
    parser.add_argument("--hard-deadline-monotonic-ns", type=int, default=None)
    parser.add_argument("--expected-run-nonce", default=None)
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help=(
            "externally owned persistent VLA endpoint; validate it but never stop it"
        ),
    )
    parser.add_argument(
        "--dashboard-event-sink",
        action="store_true",
        help=(
            "emit child Dashboard events for an externally owned campaign "
            "Dashboard relay"
        ),
    )
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8766)
    parser.add_argument(
        "--dashboard-language",
        choices=("en", "zh-cn"),
        default="en",
    )
    parser.add_argument("--source-snapshot-root", default=None)
    parser.add_argument("--source-snapshot-binding-sha256", default=None)
    parser.add_argument(
        "--external-gpu-lock-owned",
        action="store_true",
        help=(
            "declare that an external paired-campaign supervisor owns the "
            "GPU lock for this entire invocation"
        ),
    )
    parser.add_argument("--runtime-isolation-root", default=None)
    parser.add_argument("--runtime-isolation-binding-sha256", default=None)
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo", default=None)
    parser.add_argument("--behavior-python", default=None)
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
    )
    parser.add_argument("--test-instances-csv", default=None)
    parser.add_argument("--behavior-frozen-publication-root", required=True)
    parser.add_argument("--behavior-frozen-provenance-sha256", required=True)
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
            "local reviewed BEHAVIOR resource source to seal into the versioned "
            "cache instead of resolving HuggingFace"
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
        "--expected-csv-sha256",
        default=TEST_INSTANCES_SHA256,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _external_instance_deadline_binding(
    args: argparse.Namespace,
) -> InstanceDeadlineBinding | None:
    values = (
        args.instance_started_monotonic_ns,
        args.action_deadline_monotonic_ns,
        args.cleanup_deadline_monotonic_ns,
        args.hard_deadline_monotonic_ns,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "all four absolute monotonic deadline flags must be provided together"
        )
    binding = InstanceDeadlineBinding(
        started_monotonic_ns=int(args.instance_started_monotonic_ns),
        action_deadline_monotonic_ns=int(args.action_deadline_monotonic_ns),
        cleanup_deadline_monotonic_ns=int(args.cleanup_deadline_monotonic_ns),
        hard_deadline_monotonic_ns=int(args.hard_deadline_monotonic_ns),
    )
    _validate_instance_deadline_binding(
        binding,
        action_deadline_s=args.max_wall_clock_s,
        cleanup_deadline_s=args.cleanup_deadline_s,
        instance_timeout_s=args.instance_timeout_s,
    )
    return binding


def _validate_runtime_output_separation(
    *,
    runtime_isolation_root: Path,
    output_root: Path,
) -> None:
    """Reject any runtime/output identity that could mix cache and evidence."""

    runtime = runtime_isolation_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    if runtime == output or runtime in output.parents or output in runtime.parents:
        raise ValueError(
            "formal Eval runtime isolation root and output root must be "
            "disjoint paths with no ancestor relationship"
        )


def main(argv: Iterable[str] | None = None) -> int:
    """Create an immutable plan, then execute each cell in its single attempt."""

    args = _parse_args(argv)
    if args.max_turns <= 0:
        raise SystemExit("turn limit must be positive")
    try:
        _validate_deadline_budget(
            planner_timeout_s=args.planner_timeout_s,
            max_wall_clock_s=args.max_wall_clock_s,
            cleanup_deadline_s=args.cleanup_deadline_s,
            instance_timeout_s=args.instance_timeout_s,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        external_instance_deadlines = _external_instance_deadline_binding(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if external_instance_deadlines is not None and (
        not args.external_gpu_lock_owned or len(args.public_seed or ()) != 1
    ):
        raise SystemExit(
            "externally supplied monotonic deadlines require one paired-supervisor "
            "public seed and external GPU-lock ownership"
        )
    if args.expected_run_nonce is not None and (
        not isinstance(args.expected_run_nonce, str)
        or _RUN_NONCE_RE.fullmatch(args.expected_run_nonce) is None
    ):
        raise SystemExit("--expected-run-nonce must be 32 lowercase hex characters")
    if external_instance_deadlines is not None and args.expected_run_nonce is None:
        raise SystemExit("paired-supervisor Agentic Eval requires --expected-run-nonce")
    if not (1 <= args.dashboard_port <= 65535):
        raise SystemExit("--dashboard-port must be between 1 and 65535")
    if (args.source_snapshot_root is None) != (
        args.source_snapshot_binding_sha256 is None
    ):
        raise SystemExit(
            "--source-snapshot-root and --source-snapshot-binding-sha256 must "
            "be provided together"
        )
    if (args.runtime_isolation_root is None) != (
        args.runtime_isolation_binding_sha256 is None
    ):
        raise SystemExit(
            "--runtime-isolation-root and "
            "--runtime-isolation-binding-sha256 must be provided together"
        )
    if args.external_gpu_lock_owned and not (
        args.vla_endpoint and args.source_snapshot_root and args.runtime_isolation_root
    ):
        raise SystemExit(
            "--external-gpu-lock-owned requires an external VLA, sealed source "
            "snapshot, and bound campaign runtime isolation"
        )
    if args.runtime_isolation_root is None:
        raise SystemExit(
            "formal BEHAVIOR Eval requires --runtime-isolation-root and "
            "--runtime-isolation-binding-sha256"
        )
    output_root = Path(args.output_root).expanduser().resolve()
    try:
        _validate_runtime_output_separation(
            runtime_isolation_root=Path(args.runtime_isolation_root),
            output_root=output_root,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        task_spec = get_task_spec(args.task_name)
        task_spec_binding = _task_spec_binding(task_spec)
        eval_public_seeds = (
            task_spec.eval_public_seeds
            if args.public_seed is None
            else tuple(args.public_seed)
        )
        if not eval_public_seeds:
            raise ValueError("at least one formal Eval public seed is required")
        if len(eval_public_seeds) != len(set(eval_public_seeds)):
            raise ValueError("formal Eval public seeds must not contain duplicates")
        for public_seed in eval_public_seeds:
            task_spec.instance_for_public_seed(public_seed, phase="eval")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        cuda_device = _normalize_cuda_device(args.cuda_device)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if cuda_device != "7":
        raise SystemExit("formal BEHAVIOR Eval is restricted to GPU7")
    configured_repo_root = Path(args.repo_root).expanduser().resolve()
    source_snapshot = None
    source_snapshot_binding: dict[str, Any] | None = None
    if args.source_snapshot_root is not None:
        try:
            source_snapshot = _validated_source_snapshot(
                Path(args.source_snapshot_root).expanduser().resolve(),
                expected_binding_sha256=args.source_snapshot_binding_sha256,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(
                f"invalid sealed RPent source snapshot: {error}"
            ) from error
        repo_root = Path(source_snapshot.snapshot_root).resolve()
        source_snapshot_binding = _plain_binding(source_snapshot.as_dict())
        source = {
            "mode": "sealed_snapshot",
            "snapshot": source_snapshot_binding,
        }
    else:
        repo_root = configured_repo_root
        source = source_identity(repo_root)
    runtime_isolation = None
    runtime_isolation_binding: dict[str, Any] | None = None
    if args.runtime_isolation_root is not None:
        try:
            runtime_isolation = validate_campaign_runtime_isolation(
                Path(args.runtime_isolation_root).expanduser().resolve(),
                str(args.runtime_isolation_binding_sha256),
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise SystemExit(f"invalid campaign runtime isolation: {error}") from error
        if runtime_isolation.cuda_device != cuda_device:
            raise SystemExit(
                "campaign runtime isolation GPU does not match --cuda-device"
            )
        runtime_isolation_binding = _plain_binding(runtime_isolation.as_dict())
    resource_cache = (
        Path(args.behavior_resource_cache).expanduser().resolve()
        if args.behavior_resource_cache
        else configured_repo_root / "resources" / ".snapshots"
    )
    if (
        args.behavior_resource_local is not None
        and args.behavior_resource_revision is not None
    ):
        raise SystemExit(
            "--behavior-resource-local and --behavior-resource-revision are "
            "mutually exclusive"
        )
    try:
        resource_binding = (
            prepare_local_dataset_resources(
                "behavior",
                source_root=(Path(args.behavior_resource_local).expanduser().resolve()),
                cache_root=resource_cache,
            )
            if args.behavior_resource_local is not None
            else prepare_pinned_dataset_resources(
                "behavior",
                requested_revision=args.behavior_resource_revision,
                cache_root=resource_cache,
                offline=args.behavior_resource_offline,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"invalid frozen BEHAVIOR resources: {error}") from error
    resource_root = resource_binding.root
    resource_source_binding = resource_binding.as_dict()
    try:
        reviewed_memory = load_behavior_memory_snapshot(resource_root / "memory")
    except BehaviorMemorySnapshotError as error:
        raise SystemExit(f"invalid reviewed BEHAVIOR Global Memory: {error}") from error
    reviewed_memory_binding = {
        "snapshot_sha256": reviewed_memory.snapshot_sha256,
        "manifest": asdict(reviewed_memory.manifest_binding),
        "files": {
            name: asdict(metadata) for name, metadata in reviewed_memory.files.items()
        },
        "selection": reviewed_memory.select_task(task_spec.task_name).public_binding,
    }
    try:
        reviewed_recipe_catalog_binding = _reviewed_recipe_catalog_binding(
            resource_root,
            task_name=task_spec.task_name,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(
            f"invalid reviewed BEHAVIOR Recipe Catalog: {error}"
        ) from error
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
    entry_python = Path(args.python).expanduser().absolute()
    try:
        checkpoint = Path(args.policy_checkpoint).expanduser().resolve(strict=True)
        checkpoint_binding = _expected_shared_policy_checkpoint_binding()
    except OSError as error:
        raise SystemExit(str(error)) from error
    if str(checkpoint) != checkpoint_binding["resolved_path"]:
        raise SystemExit(
            "formal BEHAVIOR Eval requires the shared policy checkpoint "
            f"{checkpoint_binding['resolved_path']}; got {checkpoint}"
        )
    checkpoint = Path(checkpoint_binding["resolved_path"])
    frozen_publication_root = (
        Path(args.behavior_frozen_publication_root).expanduser().absolute()
    )
    try:
        frozen_publication = validate_canonical_publication_root(
            frozen_publication_root,
            expected_provenance_sha256=args.behavior_frozen_provenance_sha256,
            task_name=task_spec.task_name,
            task_index=task_spec.task_index,
        )
        frozen_source_public_seed, frozen_source_tag = _validate_frozen_source_identity(
            frozen_publication, task_spec
        )
    except PublicationValidationError as error:
        raise SystemExit(f"invalid frozen BEHAVIOR publication: {error}") from error
    metadata_root = (
        behavior_repo
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
    )
    csv_path = (
        Path(
            args.test_instances_csv or metadata_root / "metadata" / "test_instances.csv"
        )
        .expanduser()
        .resolve()
    )
    instance_dir = (
        metadata_root
        / "scenes"
        / task_spec.scene_model
        / "json"
        / task_spec.state_dir_name
    )
    for required in (
        repo_root / "pyproject.toml",
        entry_python,
        behavior_python,
        checkpoint / "model.safetensors",
        checkpoint
        / "assets"
        / "behavior-1k"
        / "2025-challenge-demos"
        / "norm_stats.json",
        frozen_publication_root,
        csv_path,
        instance_dir,
    ):
        if not required.exists():
            raise SystemExit(f"required path is missing: {required}")

    try:
        _validate_entry_python(entry_python, repo_root=repo_root)
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    public_ids = read_task_instances(
        csv_path,
        task_name=task_spec.task_name,
        expected_sha256=args.expected_csv_sha256,
    )
    selected = tuple(
        task_spec.instance_for_public_seed(public_seed, phase="eval")
        for public_seed in eval_public_seeds
    )
    missing_public_ids = sorted(set(selected).difference(public_ids))
    if missing_public_ids:
        raise SystemExit(
            "public-seed mapping contains IDs absent from test_instances.csv: "
            + ", ".join(map(str, missing_public_ids))
        )
    behavior_dataset_repo = behavior_repo / ".venv-behavior" / "BEHAVIOR-1K"
    global_inputs = {
        "test_instances_csv": _file_fingerprint(csv_path),
        "rpent_python": _file_fingerprint(entry_python),
        "behavior_python": _file_fingerprint(behavior_python),
        "behavior_checkout": _checkout_identity(behavior_repo),
        "behavior_dataset_checkout": _checkout_identity(behavior_dataset_repo),
        **{
            f"frozen_publication:{relative}": _file_fingerprint(
                frozen_publication_root / relative
            )
            for relative in frozen_publication.files
        },
        **{
            f"recipe_catalog:{relative}": _file_fingerprint(
                resource_root / "recipes" / relative
            )
            for relative in (
                "catalog_manifest.json",
                *reviewed_recipe_catalog_binding["files"],
            )
        },
        "resource_manifest": _file_fingerprint(resource_root / "manifest.json"),
        **{
            f"resource:{file.path}": _file_fingerprint(resource_root / file.path)
            for file in resource_binding.files
        },
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock_path = _gpu_lock_path(cuda_device)
    lock_paths = _eval_lock_paths(
        output_root,
        cuda_device,
        args.external_gpu_lock_owned,
    )
    lock_streams: list[Any] = []
    vla_proc: subprocess.Popen[Any] | None = None
    try:
        for lock_path in lock_paths:
            stream = lock_path.open("w", encoding="utf-8")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise SystemExit(
                    f"another serial evaluator owns {lock_path}"
                ) from error
            lock_streams.append(stream)
        if output_root.exists() and any(output_root.iterdir()):
            raise SystemExit(f"--output-root must be absent or empty: {output_root}")
        output_root.mkdir(parents=True, exist_ok=True)
        verify_pinned_dataset_resources(resource_binding)
        resource_source_file = output_root / "resource_source.json"
        write_dataset_resource_binding(resource_binding, resource_source_file)
        global_inputs["resource_source"] = _file_fingerprint(resource_source_file)
        checkpoint_binding_path = output_root / "policy_checkpoint_binding.json"
        _atomic_json(checkpoint_binding_path, checkpoint_binding)
        global_inputs["policy_checkpoint_binding"] = _file_fingerprint(
            checkpoint_binding_path
        )
        external_vla_binding: dict[str, Any] | None = None
        if args.vla_endpoint:
            vla_endpoint = str(args.vla_endpoint)
            try:
                external_vla_binding = _validate_external_vla(
                    vla_endpoint,
                    checkpoint_binding=checkpoint_binding,
                )
            except Exception as error:
                raise SystemExit(f"invalid external Eval VLA: {error}") from error
        else:
            assert runtime_isolation is not None
            vla_root = output_root / "launcher_logs" / "vla"
            vla_root.mkdir(parents=True)
            vla_endpoint, vla_proc = start_vla_server(
                argparse.Namespace(
                    behavior_python=str(behavior_python),
                    behavior_repo=str(behavior_repo),
                    policy_checkpoint=str(checkpoint),
                    seed=BEHAVIOR_NATIVE_ENV_SEED,
                    vla_port=0,
                    vla_ready_timeout_s=1800,
                    cuda_device=cuda_device,
                    _behavior_policy_checkpoint_binding=checkpoint_binding,
                    _behavior_runtime_isolation=runtime_isolation,
                ),
                output_dir=vla_root,
            )

        entries: list[EvalEntry] = []
        for split_position, public_seed in enumerate(eval_public_seeds):
            instance_id = task_spec.instance_for_public_seed(
                public_seed,
                phase="eval",
            )
            csv_position = public_ids.index(instance_id)
            if public_ids[csv_position] != instance_id:
                raise RuntimeError("CSV position and raw instance ID binding diverged")
            state_matches = sorted(
                instance_dir.glob(
                    f"*_{task_spec.activity_definition_id}_{instance_id}"
                    "_template-tro_state.json"
                )
            )
            if len(state_matches) != 1:
                raise RuntimeError(
                    f"expected one tro-state for instance {instance_id}, "
                    f"found {len(state_matches)}"
                )
            instance_state_path = state_matches[0].resolve()
            output_dir = output_root / task_spec.tag(public_seed)
            entry_argv = build_entry_argv(
                python=entry_python,
                repo_root=repo_root,
                output_dir=output_dir,
                behavior_repo=behavior_repo,
                behavior_python=behavior_python,
                checkpoint=checkpoint,
                activity_instance_id=instance_id,
                public_seed=public_seed,
                cuda_device=cuda_device,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_turns=args.max_turns,
                planner_timeout_s=args.planner_timeout_s,
                max_wall_clock_s=args.max_wall_clock_s,
                frozen_publication_root=frozen_publication_root,
                frozen_provenance_sha256=args.behavior_frozen_provenance_sha256,
                reviewed_memory_snapshot_sha256=reviewed_memory.snapshot_sha256,
                recipe_catalog_sha256=reviewed_recipe_catalog_binding["catalog_sha256"],
                task_name=task_spec.task_name,
                policy_checkpoint_binding_file=checkpoint_binding_path,
                vla_endpoint=vla_endpoint,
                resource_root=resource_root,
                resource_source_file=resource_source_file,
                dashboard_event_sink=args.dashboard_event_sink,
                source_snapshot_root=(
                    Path(source_snapshot.snapshot_root)
                    if source_snapshot is not None
                    else None
                ),
                source_snapshot_binding_sha256=(
                    source_snapshot.binding_sha256
                    if source_snapshot is not None
                    else None
                ),
                runtime_isolation_root=(
                    runtime_isolation.root if runtime_isolation is not None else None
                ),
                runtime_isolation_binding_sha256=(
                    runtime_isolation.binding_sha256
                    if runtime_isolation is not None
                    else None
                ),
                runtime_isolation_namespace=(
                    runtime_isolation.namespace
                    if runtime_isolation is not None
                    else None
                ),
            )
            entries.append(
                EvalEntry(
                    split_position=split_position,
                    csv_position=csv_position,
                    activity_instance_id=instance_id,
                    public_seed=public_seed,
                    seed=BEHAVIOR_NATIVE_ENV_SEED,
                    output_dir=output_dir,
                    argv=entry_argv,
                    checkpoint=checkpoint,
                    cuda_device=cuda_device,
                    instance_state_path=instance_state_path,
                    instance_state_sha256=_sha256(instance_state_path),
                    frozen_publication_binding={
                        **frozen_publication.manifest_binding,
                        "files": frozen_publication.files,
                    },
                    reviewed_repo_memory_binding=reviewed_memory_binding,
                    reviewed_recipe_catalog_binding=(reviewed_recipe_catalog_binding),
                    policy_checkpoint_binding=checkpoint_binding,
                    resource_source_binding=resource_source_binding,
                    source_snapshot_binding=source_snapshot_binding,
                    runtime_isolation_binding=runtime_isolation_binding,
                    task_name=task_spec.task_name,
                )
            )

        plan_path = output_root / "eval_plan.json"
        results_path = output_root / "eval_results.jsonl"
        plan = {
            "schema_version": 1,
            "created_at": _utc_now(),
            "protocol": {
                "behavior_phase": "eval",
                "task_id": task_spec.task_index,
                "task_name": task_spec.task_name,
                "task_language": task_spec.task_language,
                "activity_definition_id": task_spec.activity_definition_id,
                "scene_model": task_spec.scene_model,
                "task_spec": task_spec_binding,
                "public_tool_contract_version": (CURRENT_PUBLIC_TOOL_CONTRACT_VERSION),
                "public_primitives": list(
                    PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
                ),
                "public_seeds": list(eval_public_seeds),
                "mapping_version": task_spec.mapping_version,
                "native_env_seed": BEHAVIOR_NATIVE_ENV_SEED,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "cuda_device": cuda_device,
                "gpu_lock": str(gpu_lock_path),
                "gpu_lock_owner": (
                    "external_paired_supervisor"
                    if args.external_gpu_lock_owned
                    else "serial_eval"
                ),
                "serial_eval_gpu_lock_acquired": not args.external_gpu_lock_owned,
                "max_parallel": 1,
                "max_attempts_per_instance": 1,
                "automatic_retry": False,
                "fresh_top_level_process_per_instance": True,
                "fresh_codex_context_per_instance": True,
                "persistent_campaign_vla": True,
                "fresh_vla_process_per_instance": False,
                "persistent_vla_endpoint": vla_endpoint,
                "persistent_vla_managed": vla_proc is not None,
                "persistent_vla_pid": (
                    int(vla_proc.pid) if vla_proc is not None else None
                ),
                "external_vla_binding": external_vla_binding,
                "cross_instance_adaptation": False,
                "recipe_catalog_consumer": "formal_eval",
                "candidate_recipe_entries_allowed": False,
                "success_field": (
                    'final_result.task_success from info["done"]["success"]'
                ),
                "visual_checkpoints": {
                    "optional": True,
                    "contents": "public synchronized RGB-D and capture lineage",
                    "physical_side_effect": False,
                    "success_gate": False,
                    "restore_supported": False,
                    "symbolic_recipe_identity_allowed": False,
                },
                "official_success_receipt_required": True,
                "artifact_seal_is_success_gate": False,
                "publication_is_success_gate": False,
                "eval_publication_enabled": False,
                "planner_timeout_s": args.planner_timeout_s,
                "max_wall_clock_s": args.max_wall_clock_s,
                "cleanup_deadline_s": args.cleanup_deadline_s,
                "instance_timeout_s": args.instance_timeout_s,
                "hard_instance_timeout_max_s": MAX_INSTANCE_TIMEOUT_S,
                "externally_bound_instance_deadlines": (
                    external_instance_deadlines.as_dict()
                    if external_instance_deadlines is not None
                    else None
                ),
                "dashboard": {
                    "event_sink": args.dashboard_event_sink,
                    "host": args.dashboard_host,
                    "port": args.dashboard_port,
                    "language": args.dashboard_language,
                    "owned_by_serial_eval": False,
                },
            },
            "source": source,
            "source_snapshot": source_snapshot_binding,
            "runtime_isolation": runtime_isolation_binding,
            "resource_source": resource_source_binding,
            "reviewed_repo_memory": reviewed_memory_binding,
            "reviewed_recipe_catalog": reviewed_recipe_catalog_binding,
            "frozen_publication": {
                **frozen_publication.manifest_binding,
                "files": frozen_publication.files,
            },
            "policy_checkpoint": checkpoint_binding,
            "input_fingerprints": global_inputs,
            "entries": [
                {
                    "split_position": entry.split_position,
                    "csv_position": entry.csv_position,
                    "public_seed": entry.public_seed,
                    "recipe_tag": task_spec.tag(entry.public_seed),
                    "task_name": task_spec.task_name,
                    "activity_definition_id": task_spec.activity_definition_id,
                    "activity_instance_id": entry.activity_instance_id,
                    "native_env_seed": entry.seed,
                    "output_dir": str(entry.output_dir),
                    "argv": list(entry.argv),
                    "deadlines": {
                        "planner_timeout_s": args.planner_timeout_s,
                        "max_wall_clock_s": args.max_wall_clock_s,
                        "cleanup_deadline_s": args.cleanup_deadline_s,
                        "instance_timeout_s": args.instance_timeout_s,
                    },
                    "dashboard_event_path": (
                        str(entry.output_dir / "dashboard_events.jsonl")
                        if args.dashboard_event_sink
                        else None
                    ),
                    "source_snapshot": entry.source_snapshot_binding,
                    "runtime_isolation": entry.runtime_isolation_binding,
                    "expected_run_nonce": args.expected_run_nonce,
                    "instance_state": {
                        "path": str(entry.instance_state_path),
                        "sha256": entry.instance_state_sha256,
                    },
                }
                for entry in entries
            ],
        }
        _atomic_json(plan_path, plan)

        launcher_logs = output_root / "launcher_logs"
        launcher_logs.mkdir(exist_ok=True)
        abort_remaining = False
        interrupted = False
        for entry in entries:
            if abort_remaining:
                _append_jsonl(
                    results_path,
                    {
                        "task_name": task_spec.task_name,
                        "activity_definition_id": task_spec.activity_definition_id,
                        "activity_instance_id": entry.activity_instance_id,
                        "public_seed": entry.public_seed,
                        "csv_position": entry.csv_position,
                        "outcome": "not_run",
                        "reason": (
                            "serial ownership or immutable inputs could not be "
                            "guaranteed after prior run"
                        ),
                    },
                )
                continue
            started_at = _utc_now()
            started = time.monotonic()
            instance_deadlines = (
                external_instance_deadlines
                if external_instance_deadlines is not None
                else _new_instance_deadline_binding(
                    action_deadline_s=args.max_wall_clock_s,
                    cleanup_deadline_s=args.cleanup_deadline_s,
                    instance_timeout_s=args.instance_timeout_s,
                )
            )
            admitted_deadlines: dict[str, int] | None = None
            admitted_entry: EvalEntry | None = None
            log_path = (
                launcher_logs
                / f"csv{entry.csv_position:02d}_i{entry.activity_instance_id}.log"
            )
            exit_code: int | None = None
            timed_out = False
            deadline_induced_timeout = False
            launch_error: str | None = None
            try:
                if vla_proc is not None and vla_proc.poll() is not None:
                    launch_error = (
                        "persistent Eval VLA exited; checkpoint will not be "
                        "rehashed or restarted"
                    )
                    abort_remaining = True
                elif vla_proc is None:
                    _validate_external_vla(
                        vla_endpoint,
                        checkpoint_binding=checkpoint_binding,
                    )
                fingerprint_errors = _verify_input_fingerprints(
                    repo_root=repo_root,
                    source=source,
                    global_inputs=global_inputs,
                    entry=entry,
                    source_snapshot_root=(
                        Path(source_snapshot.snapshot_root)
                        if source_snapshot is not None
                        else None
                    ),
                    source_snapshot_binding_sha256=(
                        source_snapshot.binding_sha256
                        if source_snapshot is not None
                        else None
                    ),
                    runtime_isolation_root=(
                        runtime_isolation.root
                        if runtime_isolation is not None
                        else None
                    ),
                    runtime_isolation_binding_sha256=(
                        runtime_isolation.binding_sha256
                        if runtime_isolation is not None
                        else None
                    ),
                )
                current_resource_binding = verify_pinned_dataset_resources(
                    resource_binding
                )
                try:
                    current_task_spec_binding = _task_spec_binding(
                        get_task_spec(task_spec.task_name)
                    )
                except ValueError:
                    current_task_spec_binding = None
                if current_task_spec_binding != task_spec_binding:
                    fingerprint_errors.append(
                        "BEHAVIOR TaskSpec changed after plan creation"
                    )
                if current_resource_binding.as_dict() != resource_source_binding:
                    fingerprint_errors.append(
                        "pinned BEHAVIOR resource source changed after plan creation"
                    )
                current_reviewed_memory = load_behavior_memory_snapshot(
                    resource_root / "memory"
                )
                current_reviewed_binding = {
                    "snapshot_sha256": current_reviewed_memory.snapshot_sha256,
                    "manifest": asdict(current_reviewed_memory.manifest_binding),
                    "files": {
                        name: asdict(metadata)
                        for name, metadata in current_reviewed_memory.files.items()
                    },
                    "selection": current_reviewed_memory.select_task(
                        task_spec.task_name
                    ).public_binding,
                }
                if current_reviewed_binding != reviewed_memory_binding:
                    fingerprint_errors.append(
                        "reviewed BEHAVIOR Global Memory changed after plan creation"
                    )
                current_publication = validate_canonical_publication_root(
                    frozen_publication_root,
                    expected_provenance_sha256=(args.behavior_frozen_provenance_sha256),
                    task_name=task_spec.task_name,
                    task_index=task_spec.task_index,
                    public_seed=frozen_source_public_seed,
                )
                current_frozen_source = _validate_frozen_source_identity(
                    current_publication,
                    task_spec,
                )
                if (
                    current_frozen_source
                    != (frozen_source_public_seed, frozen_source_tag)
                    or current_publication.manifest_binding
                    != frozen_publication.manifest_binding
                    or current_publication.files != frozen_publication.files
                ):
                    fingerprint_errors.append(
                        "frozen BEHAVIOR publication changed after plan creation"
                    )
                current_recipe_catalog = _reviewed_recipe_catalog_binding(
                    resource_root,
                    task_name=task_spec.task_name,
                )
                if current_recipe_catalog != reviewed_recipe_catalog_binding:
                    fingerprint_errors.append(
                        "reviewed BEHAVIOR Recipe Catalog changed after plan creation"
                    )
            except BaseException as error:
                fingerprint_errors = []
                launch_error = f"{type(error).__name__}: {error}"
                interrupted = not isinstance(error, Exception)
                abort_remaining = True
            if launch_error is not None:
                pass
            elif fingerprint_errors:
                launch_error = "; ".join(fingerprint_errors)
                abort_remaining = True
            else:
                try:
                    admitted_entry, admitted_deadlines = _admit_entry_action_budget(
                        entry,
                        deadline_binding=instance_deadlines,
                        configured_planner_timeout_s=args.planner_timeout_s,
                        admitted_at_monotonic_ns=time.monotonic_ns(),
                    )
                    with log_path.open("wb") as log_stream:
                        exit_code, timed_out = _run_entry(
                            admitted_entry,
                            repo_root=repo_root,
                            log_stream=log_stream,
                            timeout_s=args.instance_timeout_s,
                            cleanup_deadline_s=args.cleanup_deadline_s,
                            action_deadline_s=args.max_wall_clock_s,
                            source_snapshot_root=(
                                source_snapshot.snapshot_root
                                if source_snapshot is not None
                                else None
                            ),
                            source_snapshot_binding_sha256=(
                                source_snapshot.binding_sha256
                                if source_snapshot is not None
                                else None
                            ),
                            deadline_binding=instance_deadlines,
                            externally_bound_deadline=(
                                external_instance_deadlines is not None
                            ),
                            expected_run_nonce=args.expected_run_nonce,
                        )
                    deadline_induced_timeout = bool(
                        timed_out and external_instance_deadlines is not None
                    )
                    if vla_proc is not None and vla_proc.poll() is not None:
                        launch_error = (
                            "persistent Eval VLA exited during the instance; "
                            "remaining instances are blocked"
                        )
                        abort_remaining = True
                except BaseException as error:
                    launch_error = f"{type(error).__name__}: {error}"
                    if isinstance(error, TimeoutError):
                        timed_out = True
                    interrupted = not isinstance(error, Exception)
                    abort_remaining = interrupted or vla_proc is None

            alive_before_cleanup = _manifest_owned_groups(entry.output_dir)
            alive_after_cleanup: dict[str, tuple[int, ...]] = {}
            if alive_before_cleanup:
                alive_after_cleanup = _terminate_manifest_processes(
                    entry.output_dir,
                    timeout_s=(10.0 if timed_out or launch_error is not None else 30.0),
                    hard_deadline_monotonic_ns=(
                        instance_deadlines.hard_deadline_monotonic_ns
                    ),
                )
                if launch_error is None and not (
                    deadline_induced_timeout and not alive_after_cleanup
                ):
                    launch_error = (
                        "managed process groups required forced cleanup: "
                        + ", ".join(sorted(alive_before_cleanup))
                    )
            if alive_after_cleanup:
                abort_remaining = True
            ambiguous_groups = _manifest_unverified_groups(entry.output_dir)
            if ambiguous_groups:
                abort_remaining = True
                if launch_error is None:
                    launch_error = (
                        "managed process identity became ambiguous; refusing to "
                        "signal or continue serial evaluation: "
                        + ", ".join(sorted(ambiguous_groups))
                    )
            forced_cleanup_receipt: dict[str, Any] | None = None
            if (
                alive_before_cleanup
                and not alive_after_cleanup
                and not ambiguous_groups
            ):
                try:
                    forced_cleanup_receipt = _write_verified_forced_cleanup_receipt(
                        entry.output_dir,
                        forced_groups=alive_before_cleanup,
                    )
                except BaseException as error:
                    receipt_error = (
                        "failed to seal verified forced-cleanup receipt: "
                        f"{type(error).__name__}: {error}"
                    )
                    launch_error = (
                        receipt_error
                        if launch_error is None
                        else f"{launch_error}; {receipt_error}"
                    )
                    interrupted = interrupted or not isinstance(error, Exception)
                    abort_remaining = True
            if vla_proc is None and admitted_entry is not None:
                try:
                    _disable_external_vla_actions(
                        vla_endpoint,
                        checkpoint_binding=checkpoint_binding,
                    )
                except BaseException as error:
                    external_vla_error = (
                        "external VLA safe-idle reset failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    launch_error = (
                        external_vla_error
                        if launch_error is None
                        else f"{launch_error}; {external_vla_error}"
                    )
                    interrupted = interrupted or not isinstance(error, Exception)
                    abort_remaining = True
            outcome, validation_errors, final_result = validate_instance_result(
                entry,
                source_commit=(
                    str(source["commit"]) if source_snapshot is None else None
                ),
                subprocess_exit_code=exit_code,
                timed_out=timed_out,
                expected_run_nonce=args.expected_run_nonce,
            )
            observed_instance_state_sha256, instance_state_binding_error = (
                _observed_instance_state_sha256(entry)
            )
            try:
                action_trace_bytes = (
                    entry.output_dir / "behavior_action_trace.jsonl"
                ).read_bytes()
            except OSError:
                action_trace_bytes = b""
            raw_success_summary, nonce_binding_error = _bound_action_trace_success(
                action_trace_bytes,
                expected_run_nonce=args.expected_run_nonce,
            )
            raw_success_steps = _raw_success_env_steps(action_trace_bytes)
            if launch_error is not None:
                outcome = "run_error"
                validation_errors.insert(0, launch_error)
            if instance_state_binding_error is not None:
                outcome = "run_error"
                validation_errors.insert(0, instance_state_binding_error)
            if nonce_binding_error is not None:
                outcome = "run_error"
                validation_errors.insert(0, nonce_binding_error)
            infrastructure_errors = tuple(
                error
                for error in (
                    launch_error,
                    instance_state_binding_error,
                    nonce_binding_error,
                )
                if error is not None
            )
            infrastructure_error = (
                "; ".join(infrastructure_errors) if infrastructure_errors else None
            )
            record = {
                "schema_version": 1,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "elapsed_s": round(time.monotonic() - started, 3),
                "split_position": entry.split_position,
                "csv_position": entry.csv_position,
                "task_name": task_spec.task_name,
                "activity_definition_id": task_spec.activity_definition_id,
                "activity_instance_id": entry.activity_instance_id,
                "public_seed": entry.public_seed,
                "native_env_seed": entry.seed,
                "instance_state_sha256": observed_instance_state_sha256,
                "instance_state_binding_valid": (
                    instance_state_binding_error is None
                    and observed_instance_state_sha256 == entry.instance_state_sha256
                ),
                "attempt": 1,
                "subprocess_exit_code": exit_code,
                "timed_out": timed_out,
                "raw_official_success": raw_success_summary is not None,
                "raw_official_success_binding": raw_success_summary,
                "expected_run_nonce": args.expected_run_nonce,
                "first_raw_success_env_step": (
                    raw_success_steps[0] if raw_success_steps else None
                ),
                "timed_out_without_raw_success": bool(
                    timed_out and raw_success_summary is None
                ),
                "instance_deadlines": instance_deadlines.as_dict(),
                "admitted_deadlines": admitted_deadlines,
                "outcome": outcome,
                "task_success": _resolve_eval_task_success(
                    raw_success_confirmed=raw_success_summary is not None,
                    final_result=final_result,
                    infrastructure_error=infrastructure_error,
                ),
                "infrastructure_error": infrastructure_error,
                "validation_errors": validation_errors,
                "forced_cleanup_groups": {
                    role: list(members)
                    for role, members in alive_before_cleanup.items()
                },
                "alive_managed_groups": {
                    role: list(members) for role, members in alive_after_cleanup.items()
                },
                "ambiguous_managed_groups": {
                    role: list(members) for role, members in ambiguous_groups.items()
                },
                "forced_cleanup_receipt": forced_cleanup_receipt,
                "artifact_seal_complete": (
                    bool(final_result.get("artifact_seal_complete"))
                    if isinstance(final_result, dict)
                    and isinstance(final_result.get("artifact_seal_complete"), bool)
                    else False
                ),
                # Compatibility alias only. It is not consulted by result
                # classification and is never a task-success gate.
                "workflow_complete": (
                    bool(final_result.get("artifact_seal_complete"))
                    if isinstance(final_result, dict)
                    and isinstance(final_result.get("artifact_seal_complete"), bool)
                    else False
                ),
                "output_dir": str(entry.output_dir),
                "launcher_log": str(log_path),
            }
            _append_jsonl(results_path, record)

        results = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = {
            "schema_version": 1,
            "finished_at": _utc_now(),
            "task_name": task_spec.task_name,
            "task_index": task_spec.task_index,
            "task_spec": task_spec_binding,
            "public_seeds": list(eval_public_seeds),
            "gpu_lock": {
                "path": str(gpu_lock_path),
                "owner": (
                    "external_paired_supervisor"
                    if args.external_gpu_lock_owned
                    else "serial_eval"
                ),
            },
            "runtime_isolation": runtime_isolation_binding,
            "plan_path": str(plan_path),
            "results_path": str(results_path),
            "interrupted": interrupted,
            "raw_official_success_count": sum(
                item.get("raw_official_success") is True for item in results
            ),
            "timed_out_without_raw_success_count": sum(
                item.get("timed_out_without_raw_success") is True for item in results
            ),
            "counts": {
                outcome: sum(item.get("outcome") == outcome for item in results)
                for outcome in (
                    "passed",
                    "task_failed",
                    "run_error",
                    "incomplete",
                    "not_run",
                )
            },
        }
        _atomic_json(output_root / "eval_summary.json", summary)
        if interrupted:
            return 130
        return 0 if all(item.get("outcome") == "passed" for item in results) else 1
    finally:
        if vla_proc is not None:
            _terminate_process(vla_proc)
        for stream in reversed(lock_streams):
            try:
                fcntl.flock(stream, fcntl.LOCK_UN)
            finally:
                stream.close()


__all__ = [
    "EvalEntry",
    "PICKING_UP_TRASH_PUBLIC_IDS",
    "TEST_INSTANCES_SHA256",
    "TURNING_ON_RADIO_PUBLIC_IDS",
    "build_entry_argv",
    "main",
    "read_task_instances",
    "read_turning_on_radio_instances",
    "select_instances",
    "validate_instance_result",
]
