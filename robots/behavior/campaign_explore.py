"""Finite, resumable task-scoped BEHAVIOR Explore campaign.

The selected task's Explore public instance uses one finite canonical Explore
Job; all other requested instances use one finite candidate Job.  A single
campaign-owned Dashboard is reused across Jobs, and only de-instantiated
summaries are inherited by later Jobs.  Job knowledge is sealed inside the
task-bound Campaign; repository memory and recipes require a separate reviewed
promotion step.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from robots.behavior.candidate_explore import (
    CANDIDATE_ATTEMPT_TIMEOUT_S,
    CANDIDATE_MAX_WALL_CLOCK_S,
    CANDIDATE_PLANNER_TIMEOUT_S,
    _publish_candidate,
    build_candidate_config,
)
from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    prepare_pinned_dataset_resources,
    verify_pinned_dataset_resources,
)
from robots.behavior.memory_snapshot import load_behavior_memory_snapshot
from robots.behavior.policy_checkpoint import SHARED_POLICY_CHECKPOINT_PATH
from robots.behavior.publication import validate_canonical_publication_root
from robots.behavior.serial_explore import (
    ATTEMPT_TOOL_CALLS,
    ExploreConfig,
    ExploreDependencies,
    _expected_job_checkpoint_binding,
    _gpu_lock_path,
    _manifest_owned_groups,
    _manifest_unverified_groups,
    _official_success_binding,
    _owned_process_is_alive,
    _validate_symbolic_publication,
    default_dependencies,
    run_explore_job,
    sanitize_prior_attempt_summaries,
)
from robots.behavior.task_specs import TURNING_ON_RADIO_TASK_SPEC, get_task_spec

_LEGACY_RADIO_INSTANCE_ORDER = (
    TURNING_ON_RADIO_TASK_SPEC.instance_for_public_seed(0, phase="explore"),
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
DEFAULT_INSTANCE_ORDER = tuple(
    instance_id
    for instance_id in _LEGACY_RADIO_INSTANCE_ORDER
    if (
        TURNING_ON_RADIO_TASK_SPEC.classify_instance(instance_id).kind
        in {"explore", "candidate"}
    )
)
_EVIDENCE_NAMES = (
    "session_manifest.json",
    "behavior_action_trace.jsonl",
    "behavior_tool_trace.jsonl",
    "final_result.json",
    "run_manifest.json",
)
_TERMINAL_JOB_STATUSES = {"completed", "succeeded", "failed", "infra_unknown"}
_BLOCKING_REASONS = {
    "attempt_process_ownership_unverified",
    "insufficient_disk_space",
    "persistent_vla_exited",
}
_CAMPAIGN_LOCK_PREFIX = "rpent_behavior_campaign_supervisor_gpu"


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append_unique_jsonl(
    path: Path,
    payload: dict[str, Any],
    *,
    identity_key: str,
) -> None:
    identity = payload.get(identity_key)
    if identity is None:
        raise ValueError(f"JSONL payload is missing {identity_key}")
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"existing JSONL is invalid: {path}") from error
    matches = [
        record
        for record in existing
        if isinstance(record, dict) and record.get(identity_key) == identity
    ]
    if matches:
        if len(matches) != 1 or matches[0] != payload:
            raise RuntimeError(f"JSONL identity collision: {identity}")
        return
    _append_jsonl(path, payload)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_bound_json_file(path: Path) -> tuple[Path, dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("bound JSON must be a non-symlink regular file")
    resolved = path.resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise ValueError("bound JSON cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("bound JSON must be a regular file")
        chunks: list[bytes] = []
        for chunk in iter(lambda: os.read(descriptor, 64 * 1024), b""):
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity_before != identity_after:
        raise ValueError("bound JSON changed while it was read")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"bound JSON contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("bound JSON must be strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("bound JSON must contain one JSON object")
    return resolved, payload, hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign_checkpoint_binding(config: CampaignConfig) -> dict[str, Any]:
    """Return the shared immutable checkpoint binding without hashing model bytes."""

    expected = _expected_job_checkpoint_binding(config.policy_checkpoint)
    if (
        config.policy_checkpoint_binding is not None
        and config.policy_checkpoint_binding != expected
    ):
        raise ValueError("campaign checkpoint binding differs from the shared profile")
    return expected


def _campaign_resource_binding(config: CampaignConfig) -> DatasetResourceBinding:
    """Revalidate the parent-prepared resource snapshot without any download."""

    binding = config.resource_binding
    if not isinstance(binding, DatasetResourceBinding):
        raise ValueError("campaign requires a pinned BEHAVIOR resource binding")
    if binding.subtree != "behavior":
        raise ValueError("campaign resource binding must select behavior")
    verified = verify_pinned_dataset_resources(binding)
    if verified.as_dict() != binding.as_dict():
        raise ValueError("campaign BEHAVIOR resource binding changed")
    return binding


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _claim_campaign_lock(cuda_device: str) -> tuple[int, Path]:
    path = Path("/tmp") / f"{_CAMPAIGN_LOCK_PREFIX}{cuda_device}.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        payload = (
            json.dumps(
                {
                    "pid": os.getpid(),
                    "cuda_device": cuda_device,
                    "claimed_at": _utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        os.ftruncate(descriptor, 0)
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.write(descriptor, payload)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise RuntimeError(
            f"another BEHAVIOR campaign already supervises GPU{cuda_device}"
        ) from None
    return descriptor, path


def _release_campaign_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _assert_job_gpu_lock_available(cuda_device: str) -> None:
    path = _gpu_lock_path(cuda_device)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        raise RuntimeError(
            f"another standalone BEHAVIOR job already owns GPU{cuda_device}"
        ) from None
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _state_file(state_dir: Path, instance_id: int) -> Path:
    matches = list(state_dir.glob(f"*_{int(instance_id)}_template-tro_state.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one state file for instance {instance_id}, "
            f"got {len(matches)}"
        )
    path = matches[0].resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("instance state must be a non-symlink regular file")
    return path


def _source_tree_sha256(
    repo_root: Path,
    relative_roots: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    roots = tuple(repo_root / relative for relative in relative_roots)
    files = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".md", ".html"}
    )
    for path in files:
        relative = path.relative_to(repo_root)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _recipe_catalog_binding(config: CampaignConfig) -> dict[str, Any] | None:
    catalog_root = _campaign_resource_binding(config).root / "recipes"
    manifest_path = catalog_root / "catalog_manifest.json"
    declared_sha256 = config.recipe_catalog_sha256
    if declared_sha256 is None and not manifest_path.exists():
        return None
    if (
        not isinstance(declared_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None
    ):
        raise ValueError("a reviewed recipe catalog requires its exact catalog SHA256")
    from robots.behavior.recipe_catalog import load_behavior_recipe_catalog

    catalog = load_behavior_recipe_catalog(catalog_root)
    if catalog.catalog_sha256 != declared_sha256:
        raise ValueError("reviewed recipe catalog SHA256 does not match")
    return {
        "root": str(catalog_root.resolve(strict=True)),
        "catalog_sha256": catalog.catalog_sha256,
        "manifest_sha256": catalog.manifest_binding.manifest_sha256,
        "declared_catalog_sha256": (catalog.manifest_binding.declared_catalog_sha256),
    }


def _epoch_predecessor_binding(config: CampaignConfig) -> dict[str, Any] | None:
    path = config.predecessor_epoch_boundary
    declared_sha256 = config.predecessor_epoch_boundary_sha256
    if path is None and declared_sha256 is None:
        return None
    if path is None or declared_sha256 is None:
        raise ValueError(
            "predecessor epoch boundary path and SHA256 must be provided together"
        )
    if (
        not isinstance(declared_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None
    ):
        raise ValueError("predecessor epoch boundary SHA256 is invalid")
    try:
        resolved, payload, actual_sha256 = _read_bound_json_file(path)
    except ValueError as error:
        raise ValueError(f"predecessor epoch boundary is invalid: {error}") from error
    if actual_sha256 != declared_sha256:
        raise ValueError("predecessor epoch boundary SHA256 does not match")
    sha_fields = (
        "predecessor_campaign_manifest_sha256",
        "predecessor_configuration_sha256",
        "predecessor_source_tree_sha256",
        "predecessor_reviewed_memory_snapshot_sha256",
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "behavior_campaign_epoch_boundary"
        or payload.get("resume_predecessor_campaign") is not False
        or any(
            not isinstance(payload.get(field), str)
            or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None
            for field in sha_fields
        )
        or not isinstance(payload.get("sealed_completed_prefix"), int)
        or isinstance(payload.get("sealed_completed_prefix"), bool)
        or payload["sealed_completed_prefix"] < 0
        or not isinstance(payload.get("sealed_instance_order"), list)
        or not isinstance(payload.get("remaining_instance_order"), list)
        or not isinstance(payload.get("reason"), str)
        or not payload["reason"].strip()
    ):
        raise ValueError("predecessor epoch boundary schema is invalid")
    sealed = payload["sealed_instance_order"]
    remaining = payload["remaining_instance_order"]
    if (
        any(not isinstance(value, int) or isinstance(value, bool) for value in sealed)
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in remaining
        )
        or len(sealed) != payload["sealed_completed_prefix"]
        or len(set(sealed)) != len(sealed)
        or len(set(remaining)) != len(remaining)
        or set(sealed).intersection(remaining)
        or tuple(remaining) != config.instance_ids
    ):
        raise ValueError(
            "predecessor epoch boundary does not match the new instance order"
        )
    return {
        "path": str(resolved),
        "sha256": actual_sha256,
        "payload_sha256": _canonical_sha256(payload),
        "predecessor_campaign_manifest_sha256": payload[
            "predecessor_campaign_manifest_sha256"
        ],
        "predecessor_configuration_sha256": payload["predecessor_configuration_sha256"],
        "predecessor_source_tree_sha256": payload["predecessor_source_tree_sha256"],
        "predecessor_reviewed_memory_snapshot_sha256": payload[
            "predecessor_reviewed_memory_snapshot_sha256"
        ],
        "sealed_completed_prefix": payload["sealed_completed_prefix"],
        "sealed_instance_order": sealed,
        "remaining_instance_order": remaining,
        "reason": payload["reason"],
        "resume_predecessor_campaign": False,
    }


def _campaign_memory_binding(config: CampaignConfig) -> dict[str, Any]:
    memory_root = _campaign_resource_binding(config).root / "memory"
    if not memory_root.is_dir():
        return {
            "snapshot_sha256": None,
            "selection": None,
        }
    memory_snapshot = load_behavior_memory_snapshot(memory_root)
    return {
        "snapshot_sha256": memory_snapshot.snapshot_sha256,
        "selection": memory_snapshot.select_task(config.task_name).public_binding,
    }


def _campaign_configuration_binding(config: CampaignConfig) -> dict[str, Any]:
    state_hashes = {
        str(instance_id): _sha256(_state_file(config.state_dir, instance_id))
        for instance_id in config.instance_ids
    }
    memory_binding = _campaign_memory_binding(config)
    memory_sha256 = memory_binding["snapshot_sha256"]
    memory_selection = memory_binding["selection"]
    return {
        "schema_version": 2,
        "task_identity": {
            "task_name": config.task_name,
            "task_index": get_task_spec(config.task_name).task_index,
            "activity_definition_id": get_task_spec(
                config.task_name
            ).activity_definition_id,
        },
        "public_seed": config.public_seed,
        "instance_order": list(config.instance_ids),
        "state_sha256": state_hashes,
        "source_tree_sha256": _source_tree_sha256(
            config.repo_root,
            ("robots/behavior", "robots/libero", "rpent", "scripts"),
        ),
        "behavior_source_tree_sha256": _source_tree_sha256(
            config.behavior_repo,
            ("rlinf", "examples/embodiment/behavior_primitives", "tools"),
        ),
        "reviewed_memory_snapshot_sha256": memory_sha256,
        "reviewed_memory_selection": memory_selection,
        "reviewed_recipe_catalog": _recipe_catalog_binding(config),
        "resource_source": _campaign_resource_binding(config).as_dict(),
        "epoch_predecessor": _epoch_predecessor_binding(config),
        "policy_checkpoint": _campaign_checkpoint_binding(config),
        "behavior_repo": str(config.behavior_repo.resolve(strict=True)),
        "python": str(config.python),
        "behavior_python": str(config.behavior_python),
        "cuda_device": config.cuda_device,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "max_turns": config.max_turns,
        "max_tool_calls": config.max_tool_calls,
        "planner_timeout_s": config.planner_timeout_s,
        "attempt_timeout_s": config.attempt_timeout_s,
        "max_wall_clock_s": config.max_wall_clock_s,
        "vla_ready_timeout_s": config.vla_ready_timeout_s,
        "min_free_disk_gb": config.min_free_disk_gb,
    }


def _attempt_dir(job_root: Path, manifest: dict[str, Any]) -> Path | None:
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    record = attempts[-1]
    value = record.get("output_dir") if isinstance(record, dict) else None
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = job_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(job_root.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_dir() and not resolved.is_symlink() else None


def _fallback_result(error: str, *, task_success: bool | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_outer_fallback": True,
        "task_success": task_success,
        "official_success_source": None,
        "publication_complete": False,
        "error": error,
        "sealed_at": _utc_now(),
    }


def _seal_job_evidence(
    job_root: Path,
    manifest: dict[str, Any],
    *,
    error: str | None = None,
) -> Path | None:
    """Ensure every Job root has the five required evidence names."""

    job_root.mkdir(parents=True, exist_ok=True)
    session_path = job_root / "session_manifest.json"
    if not session_path.is_file():
        _atomic_json(session_path, manifest)
    attempt = _attempt_dir(job_root, manifest)
    for name in ("behavior_action_trace.jsonl", "behavior_tool_trace.jsonl"):
        target = job_root / name
        source = attempt / name if attempt is not None else None
        if source is not None and source.is_file() and not source.is_symlink():
            _atomic_bytes(target, source.read_bytes())
        elif not target.exists():
            _atomic_bytes(target, b"")
    for name in ("final_result.json", "run_manifest.json"):
        target = job_root / name
        source = attempt / name if attempt is not None else None
        if source is not None and source.is_file() and not source.is_symlink():
            _atomic_bytes(target, source.read_bytes())
        elif not target.exists():
            _atomic_json(
                target,
                _fallback_result(
                    error or "runtime did not produce complete root evidence"
                ),
            )
    return attempt


def _classify_outcome(
    manifest: dict[str, Any],
    attempt: Path | None,
) -> tuple[bool | None, str, dict[str, Any] | None]:
    binding = _official_success_binding(attempt) if attempt is not None else None
    if binding is not None:
        return True, "official_success", binding
    trace_state = (
        _raw_success_trace_state(attempt)
        if attempt is not None
        else "integrity_unknown"
    )
    if trace_state == "success":
        return None, "success_integrity_unknown", None
    if trace_state == "integrity_unknown":
        return None, "trace_integrity_unknown", None
    attempts = manifest.get("attempts")
    record = (
        attempts[-1]
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict)
        else {}
    )
    outcome = str(record.get("outcome") or "")
    status = str(manifest.get("status") or "")
    blocked_reason = str(manifest.get("blocked_reason") or "")
    if (
        status in {"blocked", "stopped_by_operator", "outer_harness_failure"}
        or outcome == "run_error"
        or record.get("timed_out") is True
        or (
            record.get("subprocess_exit_code") not in {0, None}
            and outcome not in {"task_failed", "visual_terminal_failure"}
        )
    ):
        return None, "infrastructure_unknown", None
    if outcome in {"task_failed", "visual_terminal_failure"}:
        return False, outcome, None
    if isinstance(manifest.get("terminal_failure"), dict):
        return False, "visual_terminal_failure", None
    if blocked_reason:
        return None, "infrastructure_unknown", None
    return None, "infrastructure_unknown", None


def _raw_success_trace_state(attempt: Path) -> str:
    action_path = attempt / "behavior_action_trace.jsonl"
    if not action_path.is_file() or action_path.is_symlink():
        return "integrity_unknown"
    try:
        lines = action_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return "integrity_unknown"
    saw_valid_record = False
    saw_malformed_record = False
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            saw_malformed_record = True
            continue
        if not isinstance(record, dict):
            saw_malformed_record = True
            continue
        saw_valid_record = True
        done = record.get("info_done")
        if isinstance(done, dict) and done.get("success") is True:
            return "success"
        info = record.get("info")
        if (
            isinstance(info, dict)
            and isinstance(info.get("done"), dict)
            and info["done"].get("success") is True
        ):
            return "success"
    if saw_valid_record and not saw_malformed_record:
        return "clean_no_success"
    return "integrity_unknown"


def _success_recipe_lesson(job_root: Path) -> str | None:
    pair = _published_recipe_pair(job_root)
    if pair is None:
        return None
    path = pair[0]
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    lessons: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        kind = record.get("kind")
        if kind == "semantic_goal" and isinstance(record.get("goal"), str):
            lessons.append(record["goal"])
        elif kind == "evidence_contract" and isinstance(
            record.get("requirements"), str
        ):
            lessons.append(record["requirements"])
        elif kind == "failed_attempt_evidence" and isinstance(
            record.get("lesson"), str
        ):
            lessons.append(record["lesson"])
    if lessons:
        return " Reviewed symbolic-recipe lesson: " + " ".join(lessons)
    return None


def _summary_for_job(
    manifest: dict[str, Any],
    *,
    job_root: Path,
    sequence_index: int,
    task_success: bool | None,
    outcome: str,
    publication_complete: bool,
) -> dict[str, Any]:
    attempts = manifest.get("attempts")
    record = (
        attempts[-1]
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict)
        else {}
    )
    if task_success is True:
        summary = (
            "Fresh public evidence and raw official success confirmed a valid "
            "task interaction. Preserve only semantic control cues."
        )
        if publication_complete:
            summary += _success_recipe_lesson(job_root) or ""
    else:
        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            if task_success is False:
                summary = (
                    "The episode ended without raw official success. Re-ground the "
                    "visible task control and vary the semantic interaction hypothesis."
                )
            else:
                summary = (
                    "Infrastructure ended before a trustworthy task outcome was "
                    "established; do not infer a task strategy from this run."
                )
    sanitized = sanitize_prior_attempt_summaries(
        (
            {
                "attempt_index": sequence_index,
                "outcome": outcome,
                "summary": summary,
            },
        )
    )
    if not sanitized:
        raise RuntimeError("job summary sanitization produced no record")
    if not _anonymous_publication_safe(
        b"",
        sanitized[0]["summary"].encode("utf-8"),
    ):
        sanitized[0]["summary"] = (
            "Re-ground fresh semantic evidence and revise the interaction hypothesis."
            if task_success is not True
            else "Raw official success confirmed a fresh semantic interaction."
        )
    return sanitized[0]


def _artifact_hashes(job_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in _EVIDENCE_NAMES:
        path = job_root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required Job evidence is missing: {name}")
        hashes[name] = _sha256(path)
    return hashes


def _safe_copy(source: Path, target: Path) -> None:
    if source.is_file() and not source.is_symlink():
        _atomic_bytes(target, source.read_bytes())


def _write_or_verify_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != payload:
            raise RuntimeError(f"existing receipt differs: {path}")
        return
    _atomic_json(path, payload)


def _anonymous_publication_safe(recipe: bytes, memory: bytes) -> bool:
    try:
        text = (recipe + b"\n" + memory).decode("utf-8", errors="strict")
    except UnicodeError:
        return False
    if re.search(r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", text):
        return False
    instance_pattern = "|".join(str(value) for value in _LEGACY_RADIO_INSTANCE_ORDER)
    return re.search(rf"(?<!\d)(?:{instance_pattern})(?!\d)", text) is None


def _candidate_publication_pair(job_root: Path) -> tuple[Path, Path] | None:
    publication = job_root / "candidate_publication"
    if publication.is_symlink() or not publication.is_dir():
        return None
    recipe = publication / "recipe.jsonl"
    memory = publication / "task_memory.md"
    provenance_path = publication / "provenance.json"
    amendment_path = publication / "amendment.json"
    if any(
        path.is_symlink() or not path.is_file()
        for path in (recipe, memory, provenance_path, amendment_path)
    ):
        return None
    provenance = _read_json(provenance_path)
    amendment = _read_json(amendment_path)
    session = _read_json(job_root / "session_manifest.json")
    if not all(isinstance(item, dict) for item in (provenance, amendment, session)):
        return None
    assert provenance is not None and amendment is not None and session is not None
    attempt = _attempt_dir(job_root, session)
    binding = _official_success_binding(attempt) if attempt is not None else None
    if binding is None:
        return None
    source_paths = {
        "official_success_receipt": attempt / "official_success_receipt.json",
        "behavior_action_trace": job_root / "behavior_action_trace.jsonl",
        "behavior_tool_trace": job_root / "behavior_tool_trace.jsonl",
        "final_result": job_root / "final_result.json",
        "run_manifest": job_root / "run_manifest.json",
        "session_manifest": job_root / "session_manifest.json",
    }
    try:
        source_hashes = {
            name: _sha256(path)
            for name, path in source_paths.items()
            if path.is_file() and not path.is_symlink()
        }
        recipe_bytes = recipe.read_bytes()
        memory_bytes = memory.read_bytes()
    except OSError:
        return None
    if set(source_hashes) != set(source_paths):
        return None
    native_binding = session.get("native_binding")
    protocol = session.get("protocol")
    planner = session.get("planner")
    state_binding = provenance.get("state")
    task_name = (
        protocol.get("task_name")
        if isinstance(protocol, dict) and isinstance(protocol.get("task_name"), str)
        else TURNING_ON_RADIO_TASK_SPEC.task_name
    )
    task_spec = get_task_spec(task_name)
    expected_task_identity = {
        "task_name": task_spec.task_name,
        "task_index": task_spec.task_index,
        "activity_definition_id": task_spec.activity_definition_id,
        "activity_instance_id": native_binding.get("activity_instance_id")
        if isinstance(native_binding, dict)
        else None,
    }
    if (
        not isinstance(native_binding, dict)
        or not isinstance(planner, dict)
        or not isinstance(state_binding, dict)
        or provenance.get("candidate_only") is not True
        or provenance.get("eligible_for_formal_eval") is not False
        or provenance.get("review_required") is not True
        or provenance.get("task_identity") != expected_task_identity
        or provenance.get("source_tag")
        != task_spec.tag(
            int(protocol.get("public_seed", 0)) if isinstance(protocol, dict) else 0
        )
        or provenance.get("activity_instance_id")
        != native_binding.get("activity_instance_id")
        or state_binding.get("sha256") != native_binding.get("state_sha256")
        or provenance.get("planner") != planner
        or provenance.get("raw_success") != binding
        or provenance.get("source_artifacts_sha256") != source_hashes
        or provenance.get("recipe_sha256") != _sha256(recipe)
        or provenance.get("memory_sha256") != _sha256(memory)
        or amendment.get("candidate_only") is not True
        or amendment.get("eligible_for_formal_eval") is not False
        or amendment.get("review_required") is not True
        or amendment.get("publication_complete") is not True
        or amendment.get("provenance_sha256") != _sha256(provenance_path)
        or amendment.get("recipe_sha256") != _sha256(recipe)
        or amendment.get("memory_sha256") != _sha256(memory)
    ):
        return None
    try:
        _validate_symbolic_publication(recipe_bytes, memory_bytes)
    except (UnicodeError, RuntimeError, json.JSONDecodeError):
        return None
    if not _anonymous_publication_safe(recipe_bytes, memory_bytes):
        return None
    return recipe, memory


def _published_recipe_pair(job_root: Path) -> tuple[Path, Path] | None:
    if (job_root / "candidate_publication").exists():
        return _candidate_publication_pair(job_root)
    try:
        publication = validate_canonical_publication_root(job_root)
        recipe = job_root / publication.identity.recipe_relative
        memory = job_root / publication.identity.memory_relative
        recipe_bytes = recipe.read_bytes()
        memory_bytes = memory.read_bytes()
    except (OSError, UnicodeError, ValueError, RuntimeError, json.JSONDecodeError):
        return None
    if not publication.manifest_binding or not _anonymous_publication_safe(
        recipe_bytes, memory_bytes
    ):
        return None
    return recipe, memory


def _write_knowledge(
    campaign_root: Path,
    job_root: Path,
    *,
    task_name: str = TURNING_ON_RADIO_TASK_SPEC.task_name,
    summary: dict[str, Any],
    task_success: bool | None,
    outcome: str,
    publication_complete: bool,
    official_success_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write anonymous, content-addressed Job knowledge before queue advance."""

    if not _anonymous_publication_safe(
        b"",
        str(summary.get("summary") or "").encode("utf-8"),
    ):
        raise RuntimeError("Job summary failed anonymous knowledge validation")
    if (task_success is True) is not isinstance(official_success_binding, dict):
        raise RuntimeError("official-success knowledge binding mismatch")
    recipe_pair = (
        _published_recipe_pair(job_root)
        if task_success is True and publication_complete
        else None
    )
    recipe_eligible = recipe_pair is not None
    publication_payload_sha256 = (
        {
            "recipe.jsonl": _sha256(recipe_pair[0]),
            "task_memory.md": _sha256(recipe_pair[1]),
        }
        if recipe_pair is not None
        else None
    )
    task_spec = get_task_spec(task_name)
    hashes = _artifact_hashes(job_root)
    material = {
        "task_identity": {
            "task_name": task_spec.task_name,
            "task_index": task_spec.task_index,
            "activity_definition_id": task_spec.activity_definition_id,
        },
        "task_success": task_success,
        "outcome": outcome,
        "summary": summary,
        "artifact_sha256": hashes,
        "publication_payload_sha256": publication_payload_sha256,
        "official_success_receipt": official_success_binding,
    }
    digest = _canonical_sha256(material)
    knowledge_root = campaign_root / "knowledge" / task_spec.task_name
    receipt = {
        "schema_version": 1,
        "task_identity": {
            "task_name": task_spec.task_name,
            "task_index": task_spec.task_index,
            "activity_definition_id": task_spec.activity_definition_id,
        },
        "knowledge_id": digest,
        "task_success": task_success,
        "outcome": outcome,
        "official_success_checked": True,
        "publication_complete": bool(recipe_eligible),
        "artifact_sha256": hashes,
        "publication_payload_sha256": publication_payload_sha256,
        "official_success_receipt": official_success_binding,
        "summary": summary,
        "eligible_for_success_publication": recipe_eligible,
        "created_at": _utc_now(),
    }
    receipt_path = knowledge_root / "receipts" / f"{digest}.json"
    _write_or_verify_json(receipt_path, receipt)

    if recipe_eligible:
        destination = knowledge_root / "recipes" / digest
        destination.mkdir(parents=True, exist_ok=True)
        assert recipe_pair is not None
        _safe_copy(recipe_pair[0], destination / "recipe.jsonl")
        _safe_copy(recipe_pair[1], destination / "task_memory.md")
        _write_or_verify_json(destination / "knowledge_receipt.json", receipt)
    elif task_success is True:
        destination = knowledge_root / "success_unpublished" / digest
        destination.mkdir(parents=True, exist_ok=True)
        _write_or_verify_json(destination / "knowledge_receipt.json", receipt)
    else:
        destination = knowledge_root / "failure_pool" / digest
        destination.mkdir(parents=True, exist_ok=True)
        _write_or_verify_json(destination / "failure_evidence_binding.json", receipt)
        label = (
            "task-level failure" if task_success is False else "infrastructure unknown"
        )
        _atomic_bytes(
            destination / "failure_summary.md",
            (
                "# Anonymous failure summary\n\n"
                f"- Outcome: {label}.\n"
                f"- Reviewed lesson: {summary['summary']}\n"
                "- No official-success recipe is eligible for publication.\n"
            ).encode("utf-8"),
        )
        hypothesis = {
            "hypothesis": "The current semantic interaction was not confirmed.",
            "evidence": outcome,
            "reuse": summary["summary"],
            "ban": (
                "Do not encode runtime paths, geometry, fixed physical sides, "
                "or an executable action sequence."
            ),
        }
        _atomic_bytes(
            destination / "failed_hypotheses.jsonl",
            (json.dumps(hypothesis, sort_keys=True) + "\n").encode("utf-8"),
        )
        _atomic_bytes(
            destination / "reusable_negative_lessons.md",
            (
                "# Reusable negative lesson candidate\n\n"
                f"- {summary['summary']}\n"
                "- This is advisory evidence, not a success recipe.\n"
            ).encode("utf-8"),
        )
    reviewer_record = {
        "knowledge_id": digest,
        "task_name": task_spec.task_name,
        "task_success": task_success,
        "outcome": outcome,
        "summary": summary["summary"],
    }
    _append_unique_jsonl(
        knowledge_root / "reviewer" / "campaign_pattern_candidates.jsonl",
        reviewer_record,
        identity_key="knowledge_id",
    )
    return receipt


@dataclass(frozen=True)
class CampaignConfig:
    output_root: Path
    repo_root: Path
    python: Path
    behavior_repo: Path
    behavior_python: Path
    policy_checkpoint: Path
    state_dir: Path
    resource_binding: DatasetResourceBinding
    policy_checkpoint_binding: dict[str, Any] | None = None
    task_name: str = TURNING_ON_RADIO_TASK_SPEC.task_name
    public_seed: int = 0
    instance_ids: tuple[int, ...] = DEFAULT_INSTANCE_ORDER
    cuda_device: str = "7"
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    max_turns: int = 1000
    max_tool_calls: int = ATTEMPT_TOOL_CALLS
    planner_timeout_s: int = CANDIDATE_PLANNER_TIMEOUT_S
    attempt_timeout_s: int = CANDIDATE_ATTEMPT_TIMEOUT_S
    max_wall_clock_s: int = CANDIDATE_MAX_WALL_CLOCK_S
    vla_ready_timeout_s: int = 1800
    min_free_disk_gb: float = 10.0
    dashboard: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8765
    recipe_catalog_sha256: str | None = None
    predecessor_epoch_boundary: Path | None = None
    predecessor_epoch_boundary_sha256: str | None = None
    resume: bool = False


@dataclass(frozen=True)
class JobExecution:
    manifest: dict[str, Any]
    publication_error: str | None = None


JobRunner = Callable[[ExploreConfig, Path, bool, ExploreDependencies], JobExecution]


class _CampaignDashboard:
    def __init__(self, config: CampaignConfig) -> None:
        from robots.behavior.dashboard_server import DashboardServer
        from robots.behavior.dashboard_state import State

        self._config = config
        self._server = DashboardServer(
            host=config.dashboard_host,
            port=config.dashboard_port,
            runs_dir=str(config.output_root),
            language="en",
        )
        task_spec = get_task_spec(config.task_name)
        self._state = State(
            run_id="behavior/campaign",
            name=f"{task_spec.task_name} campaign",
            suite="behavior_2025_challenge",
            task=task_spec.task_index,
            seed=config.public_seed,
            output_dir=str(config.output_root),
            video_path=str(config.output_root / "episode.mp4"),
        )
        self._server.register(self._state)
        self.url = self._server.start()
        self.sequence_index = 0

    def start_job(
        self,
        job_config: ExploreConfig,
        job_id: str,
    ) -> tuple[Any, Any, str]:
        task_spec = get_task_spec(job_config.task_name)
        instance_id = (
            task_spec.instance_for_public_seed(
                job_config.public_seed,
                phase="explore",
            )
            if job_config.candidate_instance_id is None
            else int(job_config.candidate_instance_id)
        )
        self._state.set_metadata(
            {
                "planner": "codex",
                "model": self._config.model,
                "reasoning-effort": self._config.reasoning_effort,
                "behavior-phase": "explore",
                "task-name": task_spec.task_name,
                "task-index": task_spec.task_index,
                "activity-definition-id": task_spec.activity_definition_id,
                "public-seed": job_config.public_seed,
                "public-instance-id": task_spec.instance_for_public_seed(
                    job_config.public_seed,
                    phase="explore",
                ),
                "candidate-instance-id": instance_id,
                "job-id": job_id,
                "max-tool-calls": job_config.max_tool_calls,
                "max-wall-clock-s": job_config.max_wall_clock_s,
            }
        )
        self._server.arm_auto_start(
            {
                "job-id": job_id,
                "activity-instance-id": instance_id,
                "campaign-position": self.sequence_index,
            }
        )
        return self._server, self._state, self.url


def _default_job_runner(
    config: ExploreConfig,
    state_file: Path,
    canonical: bool,
    dependencies: ExploreDependencies,
) -> JobExecution:
    manifest = run_explore_job(config, dependencies=dependencies)
    publication_error = None
    attempt = _seal_job_evidence(config.output_root, manifest)
    binding = _official_success_binding(attempt) if attempt is not None else None
    if not canonical and binding is not None and attempt is not None:
        try:
            amendment = _publish_candidate(
                config,
                state_file=state_file,
                attempt_dir=attempt,
                binding=binding,
            )
            manifest["publication_complete"] = bool(
                amendment.get("publication_complete")
            )
        except Exception as error:
            publication_error = f"{type(error).__name__}: {error}"
            manifest["publication_complete"] = False
    return JobExecution(manifest=manifest, publication_error=publication_error)


def _new_campaign_manifest(
    config: CampaignConfig, dashboard_url: str | None
) -> dict[str, Any]:
    configuration = _campaign_configuration_binding(config)
    task_spec = get_task_spec(config.task_name)
    return {
        "schema_version": 2,
        "kind": "behavior_candidate_explore_campaign",
        "campaign_id": (
            f"behavior-campaign-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{secrets.token_hex(4)}"
        ),
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "pid": os.getpid(),
        "instances": list(config.instance_ids),
        "task_identity": {
            "task_name": task_spec.task_name,
            "task_index": task_spec.task_index,
            "activity_definition_id": task_spec.activity_definition_id,
        },
        "public_seed": config.public_seed,
        "planner": {
            "backend": "codex",
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
        },
        "cuda_device": config.cuda_device,
        "configuration_binding": configuration,
        "configuration_sha256": _canonical_sha256(configuration),
        "resource_source": configuration["resource_source"],
        "reviewed_recipe_catalog": configuration["reviewed_recipe_catalog"],
        "epoch_predecessor": configuration["epoch_predecessor"],
        "dashboard_url": dashboard_url,
        "current_index": 0,
        "jobs": [
            {
                "sequence_index": index,
                "task_identity": {
                    "task_name": task_spec.task_name,
                    "activity_definition_id": task_spec.activity_definition_id,
                    "activity_instance_id": instance_id,
                },
                "activity_instance_id": instance_id,
                "status": "pending",
                "task_success": None,
                "knowledge_complete": False,
            }
            for index, instance_id in enumerate(config.instance_ids, start=1)
        ],
        "rolling_summary_count": 0,
        "counts": {"success": 0, "task_failure": 0, "infra_unknown": 0},
    }


def _validate_campaign_config(config: CampaignConfig) -> None:
    _campaign_resource_binding(config)
    task_spec = get_task_spec(config.task_name)
    task_spec.instance_for_public_seed(config.public_seed, phase="explore")
    if config.cuda_device != "7":
        raise ValueError("BEHAVIOR campaign is restricted to GPU7")
    if config.model != "gpt-5.5" or config.reasoning_effort != "xhigh":
        raise ValueError("BEHAVIOR campaign requires gpt-5.5 with xhigh reasoning")
    if not config.instance_ids or len(set(config.instance_ids)) != len(
        config.instance_ids
    ):
        raise ValueError("campaign instances must be non-empty and unique")
    for value in (
        config.planner_timeout_s,
        config.attempt_timeout_s,
        config.max_wall_clock_s,
    ):
        if int(value) <= 0 or int(value) > 7200:
            raise ValueError("each campaign Job budget must be in 1..7200 seconds")
    _campaign_checkpoint_binding(config)
    if config.state_dir.resolve().name != task_spec.state_dir_name:
        raise ValueError("campaign state directory does not match the selected task")
    _recipe_catalog_binding(config)
    _epoch_predecessor_binding(config)
    for instance_id in config.instance_ids:
        classification = task_spec.classify_instance(instance_id)
        if classification.kind == "eval":
            raise ValueError(
                "public Eval instances are forbidden in candidate campaigns"
            )
        _state_file(config.state_dir, instance_id)


def _build_job_config(
    campaign: CampaignConfig,
    *,
    job_root: Path,
    instance_id: int,
    state_file: Path,
    prior: tuple[dict[str, Any], ...],
) -> ExploreConfig:
    task_spec = get_task_spec(campaign.task_name)
    canonical_instance = task_spec.instance_for_public_seed(
        campaign.public_seed,
        phase="explore",
    )
    checkpoint_binding = _campaign_checkpoint_binding(campaign)
    if instance_id != canonical_instance:
        config = build_candidate_config(
            output_root=job_root,
            repo_root=campaign.repo_root,
            python=campaign.python,
            behavior_repo=campaign.behavior_repo,
            behavior_python=campaign.behavior_python,
            policy_checkpoint=campaign.policy_checkpoint,
            policy_checkpoint_binding=checkpoint_binding,
            candidate_instance_id=instance_id,
            candidate_state_file=state_file,
            resource_binding=campaign.resource_binding,
            task_name=task_spec.task_name,
            public_seed=campaign.public_seed,
            cuda_device=campaign.cuda_device,
            model=campaign.model,
            reasoning_effort=campaign.reasoning_effort,
            max_turns=campaign.max_turns,
            max_tool_calls=campaign.max_tool_calls,
            planner_timeout_s=campaign.planner_timeout_s,
            attempt_timeout_s=campaign.attempt_timeout_s,
            max_wall_clock_s=campaign.max_wall_clock_s,
            vla_ready_timeout_s=campaign.vla_ready_timeout_s,
            min_free_disk_gb=campaign.min_free_disk_gb,
            dashboard=campaign.dashboard,
            dashboard_host=campaign.dashboard_host,
            dashboard_port=campaign.dashboard_port,
        )
        return replace(
            config,
            initial_prior_summaries=prior,
            recipe_catalog_sha256=campaign.recipe_catalog_sha256,
            epoch_predecessor_binding=_epoch_predecessor_binding(campaign),
        )
    return ExploreConfig(
        output_root=job_root,
        repo_root=campaign.repo_root,
        python=campaign.python,
        behavior_repo=campaign.behavior_repo,
        behavior_python=campaign.behavior_python,
        policy_checkpoint=campaign.policy_checkpoint,
        policy_checkpoint_binding=checkpoint_binding,
        task_name=task_spec.task_name,
        public_seed=campaign.public_seed,
        cuda_device="7",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        max_turns=campaign.max_turns,
        max_tool_calls=campaign.max_tool_calls,
        planner_timeout_s=campaign.planner_timeout_s,
        attempt_timeout_s=campaign.attempt_timeout_s,
        max_wall_clock_s=campaign.max_wall_clock_s,
        vla_ready_timeout_s=campaign.vla_ready_timeout_s,
        min_free_disk_gb=campaign.min_free_disk_gb,
        dashboard=campaign.dashboard,
        dashboard_host=campaign.dashboard_host,
        dashboard_port=campaign.dashboard_port,
        dashboard_language="en",
        resume=False,
        max_attempts=1,
        initial_prior_summaries=prior,
        recipe_catalog_sha256=campaign.recipe_catalog_sha256,
        epoch_predecessor_binding=_epoch_predecessor_binding(campaign),
        resource_binding=campaign.resource_binding,
    )


def _blocking_manifest(manifest: dict[str, Any]) -> bool:
    reason = str(manifest.get("blocked_reason") or "")
    return any(token in reason for token in _BLOCKING_REASONS)


def _job_processes_quiescent(
    session: dict[str, Any],
    attempt: Path | None,
) -> bool:
    vla_process = session.get("processes", {}).get("vla")
    try:
        if isinstance(vla_process, dict):
            if _owned_process_is_alive(vla_process) or _process_group_has_live_member(
                vla_process
            ):
                return False
        if attempt is not None and (
            _manifest_owned_groups(attempt) or _manifest_unverified_groups(attempt)
        ):
            return False
    except RuntimeError:
        return False
    return True


def _process_group_has_live_member(record: dict[str, Any]) -> bool:
    if record.get("managed") is not True:
        return False
    pgid = record.get("pgid")
    sid = record.get("sid")
    if (
        not isinstance(pgid, int)
        or isinstance(pgid, bool)
        or pgid <= 0
        or not isinstance(sid, int)
        or isinstance(sid, bool)
        or sid <= 0
    ):
        raise RuntimeError("managed process group identity is incomplete")
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            state = fields[0]
            current_pgid = int(fields[2])
            current_sid = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
        if state != "Z" and current_pgid == pgid and current_sid == sid:
            return True
    return False


def _validate_completed_job_knowledge(
    campaign_root: Path,
    job_record: dict[str, Any],
    *,
    task_name: str = TURNING_ON_RADIO_TASK_SPEC.task_name,
) -> bool:
    task_spec = get_task_spec(task_name)
    knowledge_root = campaign_root / "knowledge" / task_spec.task_name
    knowledge_id = job_record.get("knowledge_id")
    output_dir = job_record.get("output_dir")
    if not isinstance(knowledge_id, str) or len(knowledge_id) != 64:
        return False
    if not isinstance(output_dir, str) or not output_dir:
        return False
    try:
        job_root = Path(output_dir).resolve(strict=True)
        job_root.relative_to(campaign_root.resolve(strict=True))
        hashes = _artifact_hashes(job_root)
    except (OSError, RuntimeError, ValueError):
        return False
    session = _read_json(job_root / "session_manifest.json")
    if not isinstance(session, dict):
        return False
    attempt = _attempt_dir(job_root, session)
    if not _job_processes_quiescent(session, attempt):
        return False
    receipt_path = knowledge_root / "receipts" / f"{knowledge_id}.json"
    receipt = _read_json(receipt_path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("task_identity")
        != {
            "task_name": task_spec.task_name,
            "task_index": task_spec.task_index,
            "activity_definition_id": task_spec.activity_definition_id,
        }
        or receipt.get("knowledge_id") != knowledge_id
        or receipt.get("artifact_sha256") != hashes
        or receipt.get("task_success") is not job_record.get("task_success")
        or receipt.get("outcome") != job_record.get("outcome")
        or not isinstance(receipt.get("summary"), dict)
    ):
        return False
    expected_knowledge_id = _canonical_sha256(
        {
            "task_identity": receipt.get("task_identity"),
            "task_success": receipt.get("task_success"),
            "outcome": receipt.get("outcome"),
            "summary": receipt.get("summary"),
            "artifact_sha256": hashes,
            "publication_payload_sha256": receipt.get("publication_payload_sha256"),
            "official_success_receipt": receipt.get("official_success_receipt"),
        }
    )
    if knowledge_id != expected_knowledge_id:
        return False
    eligible = receipt.get("eligible_for_success_publication") is True
    if receipt.get("task_success") is True:
        current_binding = (
            _official_success_binding(attempt) if attempt is not None else None
        )
        if (
            current_binding is None
            or current_binding != receipt.get("official_success_receipt")
            or current_binding != job_record.get("official_success_receipt")
        ):
            return False
    elif receipt.get("official_success_receipt") is not None:
        return False
    if job_record.get("publication_complete") is not eligible:
        return False
    if eligible:
        recipe_root = knowledge_root / "recipes" / knowledge_id
        recipe = recipe_root / "recipe.jsonl"
        memory = recipe_root / "task_memory.md"
        destination_receipt = recipe_root / "knowledge_receipt.json"
        try:
            _validate_symbolic_publication(recipe.read_bytes(), memory.read_bytes())
        except (OSError, UnicodeError, RuntimeError, json.JSONDecodeError):
            return False
        if receipt.get("publication_payload_sha256") != {
            "recipe.jsonl": _sha256(recipe),
            "task_memory.md": _sha256(memory),
        }:
            return False
        pair = _published_recipe_pair(job_root)
        if pair is None or {
            "recipe.jsonl": _sha256(pair[0]),
            "task_memory.md": _sha256(pair[1]),
        } != receipt.get("publication_payload_sha256"):
            return False
    elif receipt.get("task_success") is True:
        if receipt.get("publication_payload_sha256") is not None:
            return False
        success_root = knowledge_root / "success_unpublished" / knowledge_id
        destination_receipt = success_root / "knowledge_receipt.json"
    else:
        if receipt.get("publication_payload_sha256") is not None:
            return False
        failure_root = knowledge_root / "failure_pool" / knowledge_id
        destination_receipt = failure_root / "failure_evidence_binding.json"
    if _read_json(destination_receipt) != receipt:
        return False
    reviewer_record = {
        "knowledge_id": knowledge_id,
        "task_name": task_spec.task_name,
        "task_success": receipt.get("task_success"),
        "outcome": receipt.get("outcome"),
        "summary": receipt.get("summary", {}).get("summary"),
    }
    for reviewer_path in (
        knowledge_root / "reviewer" / "campaign_pattern_candidates.jsonl",
    ):
        try:
            records = [
                json.loads(line)
                for line in reviewer_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        matches = [
            record
            for record in records
            if isinstance(record, dict) and record.get("knowledge_id") == knowledge_id
        ]
        if matches != [reviewer_record]:
            return False
    return True


def _validate_resume_integrity(
    campaign_root: Path,
    manifest: dict[str, Any],
) -> str | None:
    identity = manifest.get("task_identity")
    task_name = (
        identity.get("task_name")
        if isinstance(identity, dict) and isinstance(identity.get("task_name"), str)
        else TURNING_ON_RADIO_TASK_SPEC.task_name
    )
    knowledge_root = campaign_root / "knowledge" / get_task_spec(task_name).task_name
    summaries_path = knowledge_root / "rolling_summaries.json"
    completed_jobs = [
        job
        for job in manifest.get("jobs", [])
        if isinstance(job, dict)
        and job.get("status") in _TERMINAL_JOB_STATUSES
        and job.get("knowledge_complete") is True
    ]
    if completed_jobs and not summaries_path.exists():
        return "rolling_summary_integrity_mismatch"
    if summaries_path.exists():
        payload = _read_json(summaries_path)
        if not isinstance(payload, dict):
            return "rolling_summary_integrity_mismatch"
        summaries = sanitize_prior_attempt_summaries(
            item for item in payload.get("summaries", []) if isinstance(item, dict)
        )
        if payload.get("sha256") != _canonical_sha256(summaries):
            return "rolling_summary_integrity_mismatch"
        expected_summaries = []
        for job in completed_jobs:
            receipt = _read_json(
                knowledge_root / "receipts" / f"{job.get('knowledge_id')}.json"
            )
            if not isinstance(receipt, dict) or not isinstance(
                receipt.get("summary"), dict
            ):
                return "rolling_summary_integrity_mismatch"
            expected_summaries.append(receipt["summary"])
        if summaries != sanitize_prior_attempt_summaries(expected_summaries):
            return "rolling_summary_integrity_mismatch"
        if manifest.get("rolling_summary_count") != len(summaries):
            return "rolling_summary_integrity_mismatch"
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return "campaign_job_manifest_invalid"
    if manifest.get("status") == "completed":
        if (
            len(jobs) != len(manifest.get("instances", []))
            or manifest.get("current_index") != len(jobs)
            or any(
                not isinstance(job, dict)
                or job.get("status") not in _TERMINAL_JOB_STATUSES
                or job.get("knowledge_complete") is not True
                for job in jobs
            )
        ):
            return "campaign_completion_closure_mismatch"
        expected_counts = {
            "success": sum(job.get("task_success") is True for job in jobs),
            "task_failure": sum(job.get("task_success") is False for job in jobs),
            "infra_unknown": sum(job.get("task_success") is None for job in jobs),
        }
        if manifest.get("counts") != expected_counts:
            return "campaign_completion_closure_mismatch"
    for job in jobs:
        if (
            isinstance(job, dict)
            and job.get("status") in _TERMINAL_JOB_STATUSES
            and job.get("knowledge_complete") is True
            and not _validate_completed_job_knowledge(
                campaign_root,
                job,
                task_name=task_name,
            )
        ):
            return "completed_job_knowledge_integrity_mismatch"
    return None


def _run_campaign_locked(
    config: CampaignConfig,
    *,
    job_runner: JobRunner = _default_job_runner,
    dependencies: ExploreDependencies | None = None,
) -> dict[str, Any]:
    """Run each requested instance once, summarize it, then continue."""

    _validate_campaign_config(config)
    _assert_job_gpu_lock_available(config.cuda_device)
    root = config.output_root.expanduser().resolve()
    manifest_path = root / "campaign_manifest.json"
    if config.resume:
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError("--resume requires campaign_manifest.json")
        task_spec = get_task_spec(config.task_name)
        if (
            manifest.get("task_identity")
            != {
                "task_name": task_spec.task_name,
                "task_index": task_spec.task_index,
                "activity_definition_id": task_spec.activity_definition_id,
            }
            or manifest.get("public_seed") != config.public_seed
        ):
            raise RuntimeError("resume task identity does not match the campaign")
        if tuple(manifest.get("instances", ())) != config.instance_ids:
            raise RuntimeError("resume instance order does not match the campaign")
        configuration = _campaign_configuration_binding(config)
        if manifest.get("configuration_binding") != configuration or manifest.get(
            "configuration_sha256"
        ) != _canonical_sha256(configuration):
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = "campaign_configuration_binding_mismatch"
            manifest["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        integrity_error = _validate_resume_integrity(root, manifest)
        if integrity_error is not None:
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = integrity_error
            manifest["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        if manifest.get("status") == "completed":
            return manifest
        manifest["status"] = "running"
        manifest["pid"] = os.getpid()
    else:
        allowed_bootstrap = (
            dependencies is not None
            and dependencies.owns_vla is False
            and root.is_dir()
            and {path.name for path in root.iterdir()} <= {"vla"}
        )
        if root.exists() and any(root.iterdir()) and not allowed_bootstrap:
            raise RuntimeError(f"campaign root must be absent or empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        manifest = {}

    dashboard = _CampaignDashboard(config) if config.dashboard else None
    dashboard_url = dashboard.url if dashboard is not None else None
    if not manifest:
        manifest = _new_campaign_manifest(config, dashboard_url)
    else:
        manifest["dashboard_url"] = dashboard_url
    _atomic_json(manifest_path, manifest)

    summaries_path = root / "knowledge" / config.task_name / "rolling_summaries.json"
    prior_payload = _read_json(summaries_path) or {"summaries": []}
    prior = sanitize_prior_attempt_summaries(
        item for item in prior_payload.get("summaries", []) if isinstance(item, dict)
    )
    declared_prior_sha = prior_payload.get("sha256")
    if declared_prior_sha is not None and declared_prior_sha != _canonical_sha256(
        prior
    ):
        manifest["status"] = "blocked"
        manifest["blocked_reason"] = "rolling_summary_integrity_mismatch"
        manifest["finished_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        return manifest

    base_dependencies = dependencies or default_dependencies()
    for sequence_index, instance_id in enumerate(config.instance_ids, start=1):
        job_record = manifest["jobs"][sequence_index - 1]
        if (
            job_record.get("status") in _TERMINAL_JOB_STATUSES
            and job_record.get("knowledge_complete") is True
        ):
            continue
        try:
            current_resource_source = _campaign_resource_binding(config).as_dict()
        except (OSError, RuntimeError, ValueError) as error:
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = f"resource_source_binding_mismatch: {error}"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        if current_resource_source != manifest["configuration_binding"].get(
            "resource_source"
        ):
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = "resource_source_binding_mismatch"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        try:
            current_memory = _campaign_memory_binding(config)
        except (OSError, RuntimeError, ValueError) as error:
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = f"reviewed_memory_binding_mismatch: {error}"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        expected_memory = {
            "snapshot_sha256": manifest["configuration_binding"].get(
                "reviewed_memory_snapshot_sha256"
            ),
            "selection": manifest["configuration_binding"].get(
                "reviewed_memory_selection"
            ),
        }
        if current_memory != expected_memory:
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = "reviewed_memory_binding_mismatch"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        try:
            current_catalog_binding = _recipe_catalog_binding(config)
        except (OSError, RuntimeError, ValueError) as error:
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = (
                f"reviewed_recipe_catalog_binding_mismatch: {error}"
            )
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        if current_catalog_binding != manifest["configuration_binding"].get(
            "reviewed_recipe_catalog"
        ):
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = "reviewed_recipe_catalog_binding_mismatch"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        try:
            current_predecessor = _epoch_predecessor_binding(config)
        except (OSError, RuntimeError, ValueError) as error:
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = f"epoch_predecessor_binding_mismatch: {error}"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        if current_predecessor != manifest["configuration_binding"].get(
            "epoch_predecessor"
        ):
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = "epoch_predecessor_binding_mismatch"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        manifest["current_index"] = sequence_index
        job_record["status"] = "running"
        job_record["started_at"] = _utc_now()
        _atomic_json(manifest_path, manifest)

        state_file = _state_file(config.state_dir, instance_id)
        job_root = root / "jobs" / f"{sequence_index:03d}_instance_{instance_id}"
        if job_root.exists() and any(job_root.iterdir()):
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = "attempt_process_ownership_unverified"
            manifest["finished_at"] = _utc_now()
            job_record["status"] = "blocked"
            job_record["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest
        else:
            try:
                _assert_job_gpu_lock_available(config.cuda_device)
            except RuntimeError:
                manifest["status"] = "blocked"
                manifest["blocked_reason"] = "gpu_job_lock_conflict"
                manifest["finished_at"] = _utc_now()
                job_record["status"] = "blocked"
                job_record["finished_at"] = _utc_now()
                _atomic_json(manifest_path, manifest)
                return manifest
            prior_tuple = tuple(prior)
            job_config = _build_job_config(
                config,
                job_root=job_root,
                instance_id=instance_id,
                state_file=state_file,
                prior=prior_tuple,
            )
            if dashboard is not None:
                dashboard.sequence_index = sequence_index
                dependencies = replace(
                    base_dependencies,
                    start_dashboard=dashboard.start_job,
                )
            else:
                dependencies = base_dependencies
            try:
                execution = job_runner(
                    job_config,
                    state_file,
                    instance_id
                    == get_task_spec(config.task_name).instance_for_public_seed(
                        config.public_seed,
                        phase="explore",
                    ),
                    dependencies,
                )
            except KeyboardInterrupt:
                manifest["status"] = "stopped_by_operator"
                manifest["finished_at"] = _utc_now()
                job_record["status"] = "stopped_by_operator"
                _atomic_json(manifest_path, manifest)
                raise
            except BaseException as error:
                if "owns GPU lock" in str(error):
                    manifest["status"] = "blocked"
                    manifest["blocked_reason"] = "gpu_job_lock_conflict"
                    manifest["finished_at"] = _utc_now()
                    job_record["status"] = "blocked"
                    job_record["finished_at"] = _utc_now()
                    _atomic_json(manifest_path, manifest)
                    return manifest
                execution = JobExecution(
                    manifest={
                        "schema_version": 1,
                        "status": "outer_harness_failure",
                        "attempts": [],
                        "task_success": None,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    publication_error=f"{type(error).__name__}: {error}",
                )

        child_manifest = execution.manifest
        attempt = _seal_job_evidence(
            job_root,
            child_manifest,
            error=execution.publication_error,
        )
        if not _job_processes_quiescent(child_manifest, attempt):
            child_manifest["status"] = "blocked"
            child_manifest["blocked_reason"] = "attempt_process_ownership_unverified"
        task_success, outcome, binding = _classify_outcome(child_manifest, attempt)
        publication_complete = bool(
            child_manifest.get("publication_complete")
            and _published_recipe_pair(job_root) is not None
        )
        summary = _summary_for_job(
            child_manifest,
            job_root=job_root,
            sequence_index=sequence_index,
            task_success=task_success,
            outcome=outcome,
            publication_complete=publication_complete,
        )
        receipt = _write_knowledge(
            root,
            job_root,
            task_name=config.task_name,
            summary=summary,
            task_success=task_success,
            outcome=outcome,
            publication_complete=publication_complete,
            official_success_binding=binding,
        )
        prior.append(summary)
        prior = sanitize_prior_attempt_summaries(prior)
        _atomic_json(
            summaries_path,
            {
                "schema_version": 1,
                "summaries": prior,
                "sha256": _canonical_sha256(prior),
                "updated_at": _utc_now(),
            },
        )

        job_record.update(
            {
                "status": (
                    "succeeded"
                    if task_success is True
                    else "failed"
                    if task_success is False
                    else "infra_unknown"
                ),
                "task_success": task_success,
                "outcome": outcome,
                "finished_at": _utc_now(),
                "output_dir": str(job_root),
                "state_sha256": _sha256(state_file),
                "official_success_receipt": binding,
                "publication_complete": publication_complete,
                "publication_error": execution.publication_error,
                "knowledge_complete": True,
                "knowledge_id": receipt["knowledge_id"],
            }
        )
        count_key = (
            "success"
            if task_success is True
            else "task_failure"
            if task_success is False
            else "infra_unknown"
        )
        manifest["counts"][count_key] += 1
        manifest["rolling_summary_count"] = len(prior)
        _atomic_json(manifest_path, manifest)
        _append_jsonl(
            root / "events.jsonl",
            {
                "type": "campaign_job_completed",
                "sequence_index": sequence_index,
                "activity_instance_id": instance_id,
                "task_success": task_success,
                "outcome": outcome,
                "knowledge_id": receipt["knowledge_id"],
                "at": _utc_now(),
            },
        )
        if _blocking_manifest(child_manifest):
            manifest["status"] = "blocked"
            manifest["blocked_reason"] = child_manifest.get("blocked_reason")
            manifest["finished_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
            return manifest

    manifest["status"] = "completed"
    manifest["current_index"] = len(config.instance_ids)
    manifest["finished_at"] = _utc_now()
    _atomic_json(manifest_path, manifest)
    return manifest


def run_campaign(
    config: CampaignConfig,
    *,
    job_runner: JobRunner = _default_job_runner,
) -> dict[str, Any]:
    """Hold the GPU-level supervisor lock while running the serial campaign."""

    _campaign_resource_binding(config)
    descriptor, _ = _claim_campaign_lock(config.cuda_device)
    owner_dependencies: ExploreDependencies | None = None
    persistent_vla: Any = None
    try:
        if job_runner is _default_job_runner:
            _validate_campaign_config(config)
            _assert_job_gpu_lock_available(config.cuda_device)
            root = config.output_root.expanduser().resolve()
            if not config.resume and root.exists() and any(root.iterdir()):
                raise RuntimeError(f"campaign root must be absent or empty: {root}")
            vla_root = root / "vla"
            vla_root.mkdir(parents=True, exist_ok=True)
            first_instance = config.instance_ids[0]
            bootstrap = _build_job_config(
                config,
                job_root=root / "jobs" / "_vla_bootstrap",
                instance_id=first_instance,
                state_file=_state_file(config.state_dir, first_instance),
                prior=(),
            )
            owner_dependencies = default_dependencies()
            endpoint, persistent_vla = owner_dependencies.start_vla(
                bootstrap,
                vla_root,
            )
            borrowed_dependencies = replace(
                owner_dependencies,
                start_vla=lambda _config, _output_dir: (
                    endpoint,
                    persistent_vla,
                ),
                stop_vla=lambda _proc: None,
                owns_vla=False,
            )
        else:
            borrowed_dependencies = None
        return _run_campaign_locked(
            config,
            job_runner=job_runner,
            dependencies=borrowed_dependencies,
        )
    finally:
        if owner_dependencies is not None and persistent_vla is not None:
            owner_dependencies.stop_vla(persistent_vla)
        _release_campaign_lock(descriptor)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a finite serial BEHAVIOR candidate Explore campaign."
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--instances", type=int, nargs="+", default=None)
    parser.add_argument(
        "--task-name",
        choices=("turning_on_radio", "picking_up_trash"),
        default=TURNING_ON_RADIO_TASK_SPEC.task_name,
    )
    parser.add_argument("--public-seed", type=int, default=0)
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo")
    parser.add_argument("--behavior-python")
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
        help="shared BEHAVIOR Pi0.5 checkpoint; task-specific SFTs are rejected",
    )
    parser.add_argument("--state-dir")
    parser.add_argument("--cuda-device", default="7")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--max-tool-calls", type=int, default=ATTEMPT_TOOL_CALLS)
    parser.add_argument(
        "--planner-timeout-s", type=int, default=CANDIDATE_PLANNER_TIMEOUT_S
    )
    parser.add_argument(
        "--attempt-timeout-s", type=int, default=CANDIDATE_ATTEMPT_TIMEOUT_S
    )
    parser.add_argument(
        "--max-wall-clock-s", type=int, default=CANDIDATE_MAX_WALL_CLOCK_S
    )
    parser.add_argument("--vla-ready-timeout-s", type=int, default=1800)
    parser.add_argument("--min-free-disk-gb", type=float, default=10.0)
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--recipe-catalog-sha256")
    parser.add_argument("--predecessor-epoch-boundary")
    parser.add_argument("--predecessor-epoch-boundary-sha256")
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
        help="forbid network resolution/download; HF_HUB_OFFLINE=1 also enables it",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    task_spec = get_task_spec(args.task_name)
    if args.instances is None:
        if task_spec is not TURNING_ON_RADIO_TASK_SPEC:
            raise SystemExit(
                "--instances is required for non-Radio candidate campaigns"
            )
        instance_ids = DEFAULT_INSTANCE_ORDER
    else:
        instance_ids = tuple(args.instances)
    repo_root = Path(args.repo_root).expanduser().resolve()
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
        resource_binding = prepare_pinned_dataset_resources(
            "behavior",
            requested_revision=args.behavior_resource_revision,
            cache_root=(
                Path(args.behavior_resource_cache).expanduser().resolve()
                if args.behavior_resource_cache
                else repo_root / "resources" / ".snapshots"
            ),
            offline=args.behavior_resource_offline,
        )
        checkpoint_binding = _expected_job_checkpoint_binding(args.policy_checkpoint)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    state_dir = (
        Path(
            args.state_dir
            or behavior_repo
            / ".venv-behavior"
            / "BEHAVIOR-1K"
            / "datasets"
            / "2025-challenge-task-instances"
            / "scenes"
            / task_spec.scene_model
            / "json"
            / task_spec.state_dir_name
        )
        .expanduser()
        .resolve()
    )
    config = CampaignConfig(
        output_root=Path(args.output_root),
        repo_root=repo_root,
        python=Path(args.python).expanduser().absolute(),
        behavior_repo=behavior_repo,
        behavior_python=behavior_python,
        policy_checkpoint=Path(checkpoint_binding["resolved_path"]),
        policy_checkpoint_binding=checkpoint_binding,
        state_dir=state_dir,
        resource_binding=resource_binding,
        task_name=task_spec.task_name,
        public_seed=args.public_seed,
        instance_ids=instance_ids,
        cuda_device=args.cuda_device,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        planner_timeout_s=args.planner_timeout_s,
        attempt_timeout_s=args.attempt_timeout_s,
        max_wall_clock_s=args.max_wall_clock_s,
        vla_ready_timeout_s=args.vla_ready_timeout_s,
        min_free_disk_gb=args.min_free_disk_gb,
        dashboard=args.dashboard,
        dashboard_host=args.dashboard_host,
        dashboard_port=args.dashboard_port,
        recipe_catalog_sha256=args.recipe_catalog_sha256,
        predecessor_epoch_boundary=(
            Path(args.predecessor_epoch_boundary).expanduser().absolute()
            if args.predecessor_epoch_boundary
            else None
        ),
        predecessor_epoch_boundary_sha256=(args.predecessor_epoch_boundary_sha256),
        resume=args.resume,
    )
    result = run_campaign(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.dashboard and result.get("dashboard_url"):
        print(f"Dashboard remains available at {result['dashboard_url']}")
        try:
            signal.pause()
        except KeyboardInterrupt:
            pass
    return 0 if result.get("status") == "completed" else 2


__all__ = [
    "CampaignConfig",
    "DEFAULT_INSTANCE_ORDER",
    "JobExecution",
    "main",
    "run_campaign",
]
