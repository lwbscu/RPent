"""Load a reviewed, hash-bound BEHAVIOR symbolic-recipe catalog.

The catalog is a closed set: every regular file below the catalog root must be
declared by exactly one manifest entry, and every declared payload is checked
before any text is made available to a planner.  Selection is deterministic
and consumer-aware so candidate Explore knowledge cannot leak into formal Eval.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from robots.behavior.task_specs import get_task_spec

CATALOG_MANIFEST = "catalog_manifest.json"
CATALOG_MANIFEST_KIND = "reviewed_behavior_recipe_catalog"
CATALOG_MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_CATALOG_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2})
SYMBOLIC_RECIPE_SCHEMA_VERSION = 1
PROMOTION_RECEIPT_KIND = "reviewed_behavior_recipe_promotion"
RESOURCE_PROVENANCE_KIND = "behavior_recipe_provenance"
RESOURCE_RECEIPT_KIND = "reviewed_behavior_recipe_receipt"
RAW_SUCCESS_SOURCE = 'info["done"]["success"]'
RECIPE_SOURCE = "raw_official_success_v1"

EXPLORE_CONSUMER = "explore"
FORMAL_EVAL_CONSUMER = "formal_eval"
RECIPE_CONSUMERS = frozenset({EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER})

CANONICAL_PUBLIC_PROVENANCE = "canonical_public_explore"
CANDIDATE_REVIEWED_PROVENANCE = "candidate_explore_reviewed"
RECIPE_PROVENANCE_CLASSES = frozenset(
    {CANONICAL_PUBLIC_PROVENANCE, CANDIDATE_REVIEWED_PROVENANCE}
)
SOURCE_EVIDENCE_VALIDATIONS = {
    "canonical_publication_validation": "canonical_publication_validated",
    "official_success_receipt": "raw_info_done_success_true",
}

MAX_CATALOG_ENTRIES = 64
MAX_RECIPE_FILE_BYTES = 256 * 1024
MAX_RECIPE_TOTAL_BYTES = 2 * 1024 * 1024

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_PORTABLE_COMPONENT_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,95}")
_PORTABLE_PATH_PART_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_PROMPT_FORBIDDEN_PATTERNS = (
    re.compile(r"/(?:home|tmp|mnt|workspace)/", re.IGNORECASE),
    re.compile(
        r"\b(?:instance|native_instance|native_seed|seed|job|attempt|campaign|"
        r"run_nonce|attempt_nonce|knowledge_id|artifact_sha256|env_step)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[0-9a-f]{64}\b"),
    re.compile(r"\b(?:left|right)[-_ ]?(?:hand|arm|wrist|gripper)\b", re.IGNORECASE),
    re.compile(r"\b[xyzuv]\s*[:=]\s*[-+]?\d+(?:\.\d+)?", re.IGNORECASE),
    re.compile(r"\b(?:pixel|row|col(?:umn)?)\b", re.IGNORECASE),
    re.compile(
        r"(?<![\w.])[\[(]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:\s*,\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)){1,5}\s*[\])]"
    ),
    re.compile(
        r"\b(?:close|move_to|observe|open|pi0_nav_pick|pixel_to_world|press|"
        r"rotate_wrist|save_robot_state_checkpoint|max_chunks|"
        r"max_vla_chunks_per_call|max_total_vla_chunks|call_chunk_limit)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bchunks?\s*(?:(?:=|:)\s*|\bto\s+)"
        r"(?:[a-z_][a-z0-9_]*|\d+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d+\s+(?:complete\s+)?(?:pi0(?:\.5)?[- ]?)?chunks?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"</?(?:reviewed_behavior_recipe|symbolic_recipe_jsonl|reviewed_task_memory)\b",
        re.IGNORECASE,
    ),
)


class BehaviorRecipeCatalogError(ValueError):
    """Raised when a reviewed BEHAVIOR recipe catalog is not trustworthy."""


@dataclass(frozen=True)
class RecipeFileMetadata:
    """Verified metadata for one catalog payload."""

    relative_path: str
    entry_id: str
    role: str
    size_bytes: int
    sha256: str
    included_in_prompt: bool


@dataclass(frozen=True)
class RecipeCatalogManifestBinding:
    """Identity and digest binding for ``catalog_manifest.json``."""

    schema_version: int
    kind: str
    environment: str
    manifest_sha256: str
    declared_catalog_sha256: str


@dataclass(frozen=True)
class BehaviorRecipeEntry:
    """One fully validated reviewed recipe entry."""

    entry_id: str
    task: str
    provenance_class: str
    allowed_consumers: tuple[str, ...]
    entry_sha256: str
    recipe_text: str
    task_memory_text: str
    files: Mapping[str, RecipeFileMetadata]


@dataclass(frozen=True)
class BehaviorRecipeSelection:
    """Deterministic task- and phase-filtered recipe selection."""

    task_name: str
    consumer: str
    catalog_schema_version: int
    catalog_sha256: str
    entries: tuple[BehaviorRecipeEntry, ...]
    prompt_text: str

    @property
    def selected_ids(self) -> tuple[str, ...]:
        """Return selected semantic entry IDs in deterministic order."""

        return tuple(entry.entry_id for entry in self.entries)

    @property
    def public_binding(self) -> dict[str, Any]:
        """Return a JSON-safe, path-free binding suitable for run manifests."""

        return {
            "schema_version": self.catalog_schema_version,
            "kind": "reviewed_behavior_recipe_selection",
            "environment": "behavior",
            "task": self.task_name,
            "consumer": self.consumer,
            "catalog_sha256": self.catalog_sha256,
            "selected_entry_ids": list(self.selected_ids),
            "selected_entries": [
                {
                    "entry_id": entry.entry_id,
                    "entry_sha256": entry.entry_sha256,
                    "provenance_class": entry.provenance_class,
                    "payload_sha256": {
                        "recipe": entry.files["recipe"].sha256,
                        "memory": entry.files[
                            "memory" if "memory" in entry.files else "task_memory"
                        ].sha256,
                    },
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True)
class BehaviorRecipeCatalog:
    """Immutable validated recipe catalog."""

    schema_version: int
    catalog_sha256: str
    entries: Mapping[str, BehaviorRecipeEntry]
    files: Mapping[str, RecipeFileMetadata]
    manifest_binding: RecipeCatalogManifestBinding

    def select(
        self,
        task_name: str,
        consumer: str,
    ) -> BehaviorRecipeSelection:
        """Select all eligible reviewed entries for ``task_name``.

        An empty selection is valid.  The caller may impose a stronger
        phase-specific requirement, while still retaining the catalog binding.
        """

        task = _portable_component(task_name, label="task name")
        try:
            task_spec = get_task_spec(task)
        except ValueError as error:
            raise BehaviorRecipeCatalogError(
                f"unknown BEHAVIOR recipe task: {task!r}"
            ) from error
        task = task_spec.task_name
        if consumer not in RECIPE_CONSUMERS:
            raise BehaviorRecipeCatalogError(
                f"unknown recipe catalog consumer: {consumer!r}"
            )
        selected = tuple(
            entry
            for entry in sorted(self.entries.values(), key=lambda item: item.entry_id)
            if entry.task == task and consumer in entry.allowed_consumers
        )
        sections = [
            (
                f'<reviewed_behavior_recipe id="{entry.entry_id}" '
                f'provenance="{entry.provenance_class}">\n'
                "<symbolic_recipe_jsonl>\n"
                f"{entry.recipe_text.strip()}\n"
                "</symbolic_recipe_jsonl>\n"
                "<reviewed_task_memory>\n"
                f"{entry.task_memory_text.strip()}\n"
                "</reviewed_task_memory>\n"
                "</reviewed_behavior_recipe>"
            )
            for entry in selected
        ]
        return BehaviorRecipeSelection(
            task_name=task,
            consumer=consumer,
            catalog_schema_version=self.schema_version,
            catalog_sha256=self.catalog_sha256,
            entries=selected,
            prompt_text="\n\n".join(sections),
        )


def _strict_text(raw: bytes, *, label: str) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BehaviorRecipeCatalogError(f"{label} must be strict UTF-8") from error
    if "\x00" in text:
        raise BehaviorRecipeCatalogError(f"{label} must not contain NUL bytes")
    return text


def _json_without_duplicate_keys(text: str, *, label: str) -> Any:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BehaviorRecipeCatalogError(
                    f"{label} contains duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=build_object)
    except json.JSONDecodeError as error:
        raise BehaviorRecipeCatalogError(f"{label} must be valid JSON") from error


def _portable_component(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or _PORTABLE_COMPONENT_PATTERN.fullmatch(value) is None
    ):
        raise BehaviorRecipeCatalogError(
            f"{label} must be a canonical lowercase semantic identifier"
        )
    normalized = value.replace("_", " ")
    if re.search(
        r"\b(?:instance|seed|job|attempt|campaign|nonce)\b",
        normalized,
        re.IGNORECASE,
    ):
        raise BehaviorRecipeCatalogError(f"{label} must be anonymous and semantic")
    if _SHA256_PATTERN.fullmatch(value) is not None:
        raise BehaviorRecipeCatalogError(f"{label} must not be a digest")
    return value


def _portable_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BehaviorRecipeCatalogError(f"{label} must be a non-empty path")
    if "\\" in value or "%" in value:
        raise BehaviorRecipeCatalogError(f"{label} is not a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BehaviorRecipeCatalogError(f"{label} must not contain traversal")
    if any(_PORTABLE_PATH_PART_PATTERN.fullmatch(part) is None for part in path.parts):
        raise BehaviorRecipeCatalogError(f"{label} is not a portable relative path")
    return path.as_posix()


def _read_regular_file(root: Path, relative_path: str) -> bytes:
    relative = PurePosixPath(relative_path)
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink() or not cursor.is_dir():
            raise BehaviorRecipeCatalogError(
                f"recipe catalog path parent is unsafe: {relative_path}"
            )
    path = root.joinpath(*relative.parts)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BehaviorRecipeCatalogError(
            f"catalog file must be a readable regular file: {relative_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BehaviorRecipeCatalogError(
                f"catalog file must be regular: {relative_path}"
            )
        if before.st_size > MAX_RECIPE_FILE_BYTES:
            raise BehaviorRecipeCatalogError(
                f"catalog file exceeds {MAX_RECIPE_FILE_BYTES} bytes: {relative_path}"
            )
        chunks: list[bytes] = []
        remaining = MAX_RECIPE_FILE_BYTES + 1
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
        raise BehaviorRecipeCatalogError(
            f"catalog file changed while being read: {relative_path}"
        )
    return raw


def _discover_regular_files(root: Path) -> set[str]:
    discovered: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise BehaviorRecipeCatalogError(
                f"recipe catalog directory is unreadable: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise BehaviorRecipeCatalogError(
                    f"symlink is forbidden in recipe catalog: {path}"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise BehaviorRecipeCatalogError(
                    f"non-regular entry is forbidden in recipe catalog: {path}"
                )
            relative = path.relative_to(root).as_posix()
            discovered.add(_portable_relative_path(relative, label="catalog file"))
    return discovered


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_prompt_payload(text: str, *, label: str) -> None:
    if not text.strip():
        raise BehaviorRecipeCatalogError(f"{label} must not be empty")
    for pattern in _PROMPT_FORBIDDEN_PATTERNS:
        if pattern.search(text):
            raise BehaviorRecipeCatalogError(
                f"{label} contains run-specific, non-anonymous, or prescriptive guidance"
            )


def _validate_recipe_jsonl(text: str, *, task: str, label: str) -> None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise BehaviorRecipeCatalogError(f"{label} must contain JSONL records")
    records = [
        _json_without_duplicate_keys(line, label=f"{label} line {index}")
        for index, line in enumerate(lines, start=1)
    ]
    if not all(isinstance(record, dict) for record in records):
        raise BehaviorRecipeCatalogError(f"{label} records must be JSON objects")
    first = records[0]
    required_first = {
        "schema_version": SYMBOLIC_RECIPE_SCHEMA_VERSION,
        "kind": "task_level_symbolic_recipe",
        "task": task,
        "source": RECIPE_SOURCE,
    }
    if any(first.get(key) != value for key, value in required_first.items()):
        raise BehaviorRecipeCatalogError(
            f"{label} does not declare a raw-success symbolic recipe"
        )


def _parse_file_declaration(
    *,
    schema_version: int,
    entry_id: str,
    task: str,
    role: str,
    record: Any,
) -> tuple[str, int, str]:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise BehaviorRecipeCatalogError(
            f"catalog file declaration is invalid: {entry_id}/{role}"
        )
    path = _portable_relative_path(record.get("path"), label=f"{entry_id}/{role} path")
    expected_names = (
        {
            "recipe": "recipe.jsonl",
            "task_memory": "task_memory.md",
            "promotion_receipt": "promotion_receipt.json",
        }
        if schema_version == 1
        else {
            "recipe": "recipe.jsonl",
            "memory": "memory.md",
            "provenance": "provenance.json",
            "receipt": "receipt.json",
        }
    )
    expected_path = f"{task}/{entry_id}/{expected_names[role]}"
    if path != expected_path:
        raise BehaviorRecipeCatalogError(
            f"catalog payload path is not canonical for {entry_id}/{role}"
        )
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or size > MAX_RECIPE_FILE_BYTES
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise BehaviorRecipeCatalogError(
            f"catalog file metadata is invalid: {entry_id}/{role}"
        )
    return path, size, digest


def _validate_promotion_receipt(
    receipt: Any,
    *,
    entry_id: str,
    task: str,
    provenance_class: str,
    allowed_consumers: tuple[str, ...],
    recipe_sha256: str,
    task_memory_sha256: str,
) -> None:
    base_expected_keys = {
        "schema_version",
        "kind",
        "recipe_id",
        "task",
        "provenance_class",
        "allowed_consumers",
        "review_status",
        "task_success",
        "official_success_checked",
        "official_success_source",
        "payload_sha256",
        "source_evidence",
    }
    if not isinstance(receipt, dict):
        raise BehaviorRecipeCatalogError(
            f"promotion receipt fields are not canonical: {entry_id}"
        )
    receipt_schema_version = receipt.get("schema_version")
    if receipt_schema_version not in {1, 2}:
        raise BehaviorRecipeCatalogError(
            f"promotion receipt schema is not supported: {entry_id}"
        )
    expected_keys = set(base_expected_keys)
    if receipt_schema_version == 2:
        expected_keys.add("source")
    if set(receipt) != expected_keys:
        raise BehaviorRecipeCatalogError(
            f"promotion receipt fields are not canonical: {entry_id}"
        )
    source_evidence = receipt.get("source_evidence")
    if not isinstance(source_evidence, list) or not source_evidence:
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source evidence is invalid: {entry_id}"
        )
    source_kinds: list[str] = []
    for evidence in source_evidence:
        if not isinstance(evidence, dict) or set(evidence) != {
            "kind",
            "sha256",
            "validation",
        }:
            raise BehaviorRecipeCatalogError(
                f"promotion receipt source evidence is invalid: {entry_id}"
            )
        kind = evidence.get("kind")
        digest = evidence.get("sha256")
        validation = evidence.get("validation")
        if (
            kind not in SOURCE_EVIDENCE_VALIDATIONS
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            or validation != SOURCE_EVIDENCE_VALIDATIONS[kind]
        ):
            raise BehaviorRecipeCatalogError(
                f"promotion receipt source evidence is invalid: {entry_id}"
            )
        source_kinds.append(kind)
    if source_kinds != sorted(set(source_kinds)):
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source evidence is not canonical: {entry_id}"
        )
    required_kinds = {"official_success_receipt"}
    if provenance_class == CANONICAL_PUBLIC_PROVENANCE:
        required_kinds.add("canonical_publication_validation")
    if set(source_kinds) != required_kinds:
        raise BehaviorRecipeCatalogError(
            f"promotion receipt lacks required upstream evidence: {entry_id}"
        )

    expected_values = {
        "schema_version": receipt_schema_version,
        "kind": PROMOTION_RECEIPT_KIND,
        "recipe_id": entry_id,
        "task": task,
        "provenance_class": provenance_class,
        "allowed_consumers": list(allowed_consumers),
        "review_status": "accepted",
        "task_success": True,
        "official_success_checked": True,
        "official_success_source": RAW_SUCCESS_SOURCE,
        "payload_sha256": {
            "recipe": recipe_sha256,
            "task_memory": task_memory_sha256,
        },
        "source_evidence": source_evidence,
    }
    if receipt_schema_version == 2:
        source = receipt.get("source")
        _validate_promotion_source_binding(
            source,
            entry_id=entry_id,
            task=task,
        )
        expected_values["source"] = source
    if receipt != expected_values:
        raise BehaviorRecipeCatalogError(
            f"promotion receipt binding is invalid: {entry_id}"
        )


def _validate_promotion_source_binding(
    source: Any,
    *,
    entry_id: str,
    task: str,
) -> None:
    expected_keys = {
        "job",
        "attempt",
        "task",
        "action_trace_sha256",
        "review",
    }
    if not isinstance(source, dict) or set(source) != expected_keys:
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source binding is invalid: {entry_id}"
        )
    job = source.get("job")
    attempt = source.get("attempt")
    if (
        not isinstance(job, str)
        or not job
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source binding is invalid: {entry_id}"
        )
    task_identity = source.get("task")
    if (
        not isinstance(task_identity, dict)
        or set(task_identity)
        != {"name", "activity_definition_id", "activity_instance_id"}
        or task_identity.get("name") != task
        or not isinstance(task_identity.get("activity_definition_id"), int)
        or isinstance(task_identity.get("activity_definition_id"), bool)
        or task_identity.get("activity_definition_id") < 0
        or not isinstance(task_identity.get("activity_instance_id"), int)
        or isinstance(task_identity.get("activity_instance_id"), bool)
        or task_identity.get("activity_instance_id") < 0
    ):
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source binding is invalid: {entry_id}"
        )
    action_trace_sha256 = source.get("action_trace_sha256")
    if (
        not isinstance(action_trace_sha256, str)
        or _SHA256_PATTERN.fullmatch(action_trace_sha256) is None
    ):
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source binding is invalid: {entry_id}"
        )
    review = source.get("review")
    if (
        not isinstance(review, dict)
        or set(review)
        != {
            "kind",
            "sha256",
            "scope",
            "status",
            "outer_job_reclassified",
        }
        or review.get("kind") != "behavior_attempt_raw_success_recovery"
        or not isinstance(review.get("sha256"), str)
        or _SHA256_PATTERN.fullmatch(review.get("sha256")) is None
        or review.get("scope") != "attempt_only"
        or review.get("status") != "accepted"
        or review.get("outer_job_reclassified") is not False
    ):
        raise BehaviorRecipeCatalogError(
            f"promotion receipt source binding is invalid: {entry_id}"
        )


def _validate_source_binding(source: Any, *, entry_id: str, task: str) -> None:
    expected_keys = {
        "job",
        "attempt",
        "task",
        "official_success",
        "action_trace_sha256",
        "official_success_receipt_sha256",
        "official_success_receipt_payload_sha256",
        "publication_sha256",
        "vla_calls",
    }
    if not isinstance(source, dict) or set(source) != expected_keys:
        raise BehaviorRecipeCatalogError(
            f"resource source binding fields are not canonical: {entry_id}"
        )
    job = source.get("job")
    attempt = source.get("attempt")
    if (
        not isinstance(job, str)
        or not job
        or not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise BehaviorRecipeCatalogError(
            f"resource source job or attempt is invalid: {entry_id}"
        )
    task_identity = source.get("task")
    if (
        not isinstance(task_identity, dict)
        or set(task_identity)
        != {"name", "activity_definition_id", "activity_instance_id"}
        or task_identity.get("name") != task
        or not isinstance(task_identity.get("activity_definition_id"), int)
        or isinstance(task_identity.get("activity_definition_id"), bool)
        or task_identity.get("activity_definition_id") < 0
        or not isinstance(task_identity.get("activity_instance_id"), int)
        or isinstance(task_identity.get("activity_instance_id"), bool)
        or task_identity.get("activity_instance_id") < 0
    ):
        raise BehaviorRecipeCatalogError(
            f"resource source task identity is invalid: {entry_id}"
        )
    official_success = source.get("official_success")
    if (
        not isinstance(official_success, dict)
        or set(official_success) != {"field", "value", "env_step"}
        or official_success.get("field") != RAW_SUCCESS_SOURCE
        or official_success.get("value") is not True
        or not isinstance(official_success.get("env_step"), int)
        or isinstance(official_success.get("env_step"), bool)
        or official_success.get("env_step") < 1
    ):
        raise BehaviorRecipeCatalogError(
            f"resource source lacks raw official success: {entry_id}"
        )
    for field in (
        "action_trace_sha256",
        "official_success_receipt_sha256",
        "official_success_receipt_payload_sha256",
        "publication_sha256",
    ):
        digest = source.get(field)
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise BehaviorRecipeCatalogError(
                f"resource source digest is invalid: {entry_id}/{field}"
            )
    vla_calls = source.get("vla_calls")
    if not isinstance(vla_calls, list) or not vla_calls:
        raise BehaviorRecipeCatalogError(
            f"resource source VLA telemetry is invalid: {entry_id}"
        )
    for call in vla_calls:
        if (
            not isinstance(call, dict)
            or set(call) != {"complete_chunks", "partial_steps"}
            or not isinstance(call.get("complete_chunks"), int)
            or isinstance(call.get("complete_chunks"), bool)
            or call.get("complete_chunks") < 0
            or not isinstance(call.get("partial_steps"), int)
            or isinstance(call.get("partial_steps"), bool)
            or not 0 <= call.get("partial_steps") < 32
        ):
            raise BehaviorRecipeCatalogError(
                f"resource source VLA telemetry is invalid: {entry_id}"
            )


def _validate_resource_provenance(
    provenance: Any,
    *,
    entry_id: str,
    task: str,
    source: Any,
    recipe_sha256: str,
    memory_sha256: str,
) -> None:
    expected = {
        "schema_version": 1,
        "kind": RESOURCE_PROVENANCE_KIND,
        "task": task,
        "source": source,
        "payload_sha256": {
            "recipe": recipe_sha256,
            "memory": memory_sha256,
        },
    }
    if provenance != expected:
        raise BehaviorRecipeCatalogError(
            f"resource provenance binding is invalid: {entry_id}"
        )


def _validate_resource_receipt(
    receipt: Any,
    *,
    entry_id: str,
    task: str,
    provenance_class: str,
    allowed_consumers: tuple[str, ...],
    source: Any,
    recipe_sha256: str,
    memory_sha256: str,
    provenance_sha256: str,
) -> None:
    expected = {
        "schema_version": 1,
        "kind": RESOURCE_RECEIPT_KIND,
        "recipe_id": entry_id,
        "task": task,
        "provenance_class": provenance_class,
        "allowed_consumers": list(allowed_consumers),
        "review_status": "accepted",
        "task_success": True,
        "official_success_checked": True,
        "official_success_source": RAW_SUCCESS_SOURCE,
        "source": source,
        "payload_sha256": {
            "recipe": recipe_sha256,
            "memory": memory_sha256,
            "provenance": provenance_sha256,
        },
    }
    if receipt != expected:
        raise BehaviorRecipeCatalogError(
            f"resource receipt binding is invalid: {entry_id}"
        )


def load_behavior_recipe_catalog(
    catalog_root: str | Path,
) -> BehaviorRecipeCatalog:
    """Load and validate a closed-set reviewed recipe catalog."""

    requested_root = Path(catalog_root).expanduser().absolute()
    if requested_root.is_symlink():
        raise BehaviorRecipeCatalogError("recipe catalog root must not be a symlink")
    try:
        root = requested_root.resolve(strict=True)
    except OSError as error:
        raise BehaviorRecipeCatalogError(
            "recipe catalog root does not exist"
        ) from error
    if root != requested_root:
        raise BehaviorRecipeCatalogError(
            "recipe catalog root or one of its ancestors must not be a symlink"
        )
    if not root.is_dir():
        raise BehaviorRecipeCatalogError("recipe catalog root must be a directory")

    discovered = _discover_regular_files(root)
    if CATALOG_MANIFEST not in discovered:
        raise BehaviorRecipeCatalogError(f"recipe catalog omitted {CATALOG_MANIFEST}")
    manifest_raw = _read_regular_file(root, CATALOG_MANIFEST)
    manifest_text = _strict_text(manifest_raw, label=CATALOG_MANIFEST)
    manifest = _json_without_duplicate_keys(manifest_text, label=CATALOG_MANIFEST)
    if not isinstance(manifest, dict):
        raise BehaviorRecipeCatalogError("catalog manifest must be a JSON object")
    expected_manifest_keys = {
        "schema_version",
        "kind",
        "environment",
        "entries",
        "catalog_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise BehaviorRecipeCatalogError("catalog manifest fields are not canonical")
    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in SUPPORTED_CATALOG_MANIFEST_SCHEMA_VERSIONS
        or manifest.get("kind") != CATALOG_MANIFEST_KIND
        or manifest.get("environment") != "behavior"
    ):
        raise BehaviorRecipeCatalogError("catalog manifest binding is invalid")
    declared_entries = manifest.get("entries")
    if (
        not isinstance(declared_entries, list)
        or not declared_entries
        or len(declared_entries) > MAX_CATALOG_ENTRIES
    ):
        raise BehaviorRecipeCatalogError(
            f"catalog entries must contain 1..{MAX_CATALOG_ENTRIES} records"
        )

    expected_entry_keys = {
        "id",
        "task",
        "provenance_class",
        "allowed_consumers",
        "review_status",
        "task_success",
        "official_success_checked",
        "official_success_source",
        "files",
        "entry_sha256",
    }
    declarations: list[dict[str, Any]] = []
    declared_paths: dict[str, tuple[str, str, int, str]] = {}
    entry_payload_versions: dict[str, int] = {}
    seen_ids: set[str] = set()
    last_id: str | None = None
    for raw_entry in declared_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != expected_entry_keys:
            raise BehaviorRecipeCatalogError("catalog entry fields are not canonical")
        entry_id = _portable_component(raw_entry.get("id"), label="recipe id")
        task = _portable_component(raw_entry.get("task"), label="recipe task")
        if entry_id in seen_ids:
            raise BehaviorRecipeCatalogError(f"duplicate recipe id: {entry_id}")
        if last_id is not None and entry_id <= last_id:
            raise BehaviorRecipeCatalogError(
                "catalog entries must be strictly sorted by id"
            )
        last_id = entry_id
        seen_ids.add(entry_id)

        provenance = raw_entry.get("provenance_class")
        if provenance not in RECIPE_PROVENANCE_CLASSES:
            raise BehaviorRecipeCatalogError(
                f"recipe provenance is invalid: {entry_id}"
            )
        consumers_raw = raw_entry.get("allowed_consumers")
        if (
            not isinstance(consumers_raw, list)
            or not consumers_raw
            or any(item not in RECIPE_CONSUMERS for item in consumers_raw)
            or consumers_raw != sorted(set(consumers_raw))
        ):
            raise BehaviorRecipeCatalogError(
                f"recipe consumers are invalid: {entry_id}"
            )
        consumers = tuple(consumers_raw)
        if (
            provenance == CANDIDATE_REVIEWED_PROVENANCE
            and FORMAL_EVAL_CONSUMER in consumers
        ):
            raise BehaviorRecipeCatalogError(
                f"candidate Explore recipe cannot enter formal Eval: {entry_id}"
            )
        if (
            raw_entry.get("review_status") != "accepted"
            or raw_entry.get("task_success") is not True
            or raw_entry.get("official_success_checked") is not True
            or raw_entry.get("official_success_source") != RAW_SUCCESS_SOURCE
        ):
            raise BehaviorRecipeCatalogError(
                f"recipe is not reviewed raw-official-success evidence: {entry_id}"
            )
        files = raw_entry.get("files")
        legacy_roles = {"recipe", "task_memory", "promotion_receipt"}
        resource_roles = {"recipe", "memory", "provenance", "receipt"}
        if not isinstance(files, dict):
            raise BehaviorRecipeCatalogError(
                f"recipe payload roles are invalid: {entry_id}"
            )
        if set(files) == legacy_roles:
            payload_schema_version = 1
        elif schema_version == 2 and set(files) == resource_roles:
            payload_schema_version = 2
        else:
            raise BehaviorRecipeCatalogError(
                f"recipe payload roles are invalid: {entry_id}"
            )
        expected_roles = legacy_roles if payload_schema_version == 1 else resource_roles
        entry_payload_versions[entry_id] = payload_schema_version
        normalized_files: dict[str, dict[str, Any]] = {}
        for role in sorted(expected_roles):
            path, size, digest = _parse_file_declaration(
                schema_version=payload_schema_version,
                entry_id=entry_id,
                task=task,
                role=role,
                record=files[role],
            )
            if path in declared_paths:
                raise BehaviorRecipeCatalogError(
                    f"catalog payload path is declared more than once: {path}"
                )
            declared_paths[path] = (entry_id, role, size, digest)
            normalized_files[role] = {
                "path": path,
                "size_bytes": size,
                "sha256": digest,
            }
        normalized_entry = {
            "id": entry_id,
            "task": task,
            "provenance_class": provenance,
            "allowed_consumers": list(consumers),
            "review_status": "accepted",
            "task_success": True,
            "official_success_checked": True,
            "official_success_source": RAW_SUCCESS_SOURCE,
            "files": normalized_files,
        }
        declared_entry_sha = raw_entry.get("entry_sha256")
        if (
            not isinstance(declared_entry_sha, str)
            or _SHA256_PATTERN.fullmatch(declared_entry_sha) is None
            or declared_entry_sha != _canonical_json_sha256(normalized_entry)
        ):
            raise BehaviorRecipeCatalogError(
                f"recipe entry SHA256 mismatch: {entry_id}"
            )
        normalized_entry["entry_sha256"] = declared_entry_sha
        declarations.append(normalized_entry)

    expected_discovered = set(declared_paths) | {CATALOG_MANIFEST}
    if discovered != expected_discovered:
        extra = sorted(discovered - expected_discovered)
        missing = sorted(expected_discovered - discovered)
        raise BehaviorRecipeCatalogError(
            f"recipe catalog file set mismatch; extra={extra}, missing={missing}"
        )

    actual_catalog_sha = _canonical_json_sha256(
        [
            {
                "entry_id": entry["id"],
                "entry_sha256": entry["entry_sha256"],
            }
            for entry in declarations
        ]
    )
    declared_catalog_sha = manifest.get("catalog_sha256")
    if (
        not isinstance(declared_catalog_sha, str)
        or _SHA256_PATTERN.fullmatch(declared_catalog_sha) is None
        or declared_catalog_sha != actual_catalog_sha
    ):
        raise BehaviorRecipeCatalogError("aggregate recipe catalog SHA256 mismatch")

    raw_files: dict[str, bytes] = {}
    total_bytes = len(manifest_raw)
    for path, (
        _entry_id,
        _role,
        declared_size,
        declared_digest,
    ) in declared_paths.items():
        raw = _read_regular_file(root, path)
        total_bytes += len(raw)
        if total_bytes > MAX_RECIPE_TOTAL_BYTES:
            raise BehaviorRecipeCatalogError(
                f"recipe catalog exceeds {MAX_RECIPE_TOTAL_BYTES} total bytes"
            )
        _strict_text(raw, label=path)
        if len(raw) != declared_size:
            raise BehaviorRecipeCatalogError(f"catalog file size mismatch: {path}")
        if hashlib.sha256(raw).hexdigest() != declared_digest:
            raise BehaviorRecipeCatalogError(f"catalog file SHA256 mismatch: {path}")
        raw_files[path] = raw

    metadata: dict[str, RecipeFileMetadata] = {}
    entries: dict[str, BehaviorRecipeEntry] = {}
    for declaration in declarations:
        entry_id = declaration["id"]
        role_metadata: dict[str, RecipeFileMetadata] = {}
        payload_schema_version = entry_payload_versions[entry_id]
        ordered_roles = (
            ("promotion_receipt", "recipe", "task_memory")
            if payload_schema_version == 1
            else ("provenance", "receipt", "recipe", "memory")
        )
        for role in ordered_roles:
            record = declaration["files"][role]
            item = RecipeFileMetadata(
                relative_path=record["path"],
                entry_id=entry_id,
                role=role,
                size_bytes=record["size_bytes"],
                sha256=record["sha256"],
                included_in_prompt=role in {"recipe", "task_memory", "memory"},
            )
            metadata[item.relative_path] = item
            role_metadata[role] = item
        recipe_text = _strict_text(
            raw_files[role_metadata["recipe"].relative_path],
            label=role_metadata["recipe"].relative_path,
        )
        memory_role = "task_memory" if payload_schema_version == 1 else "memory"
        memory_text = _strict_text(
            raw_files[role_metadata[memory_role].relative_path],
            label=role_metadata[memory_role].relative_path,
        )
        _validate_prompt_payload(recipe_text, label=f"{entry_id} recipe")
        _validate_prompt_payload(memory_text, label=f"{entry_id} task memory")
        _validate_recipe_jsonl(
            recipe_text,
            task=declaration["task"],
            label=f"{entry_id} recipe",
        )
        if payload_schema_version == 1:
            receipt_text = _strict_text(
                raw_files[role_metadata["promotion_receipt"].relative_path],
                label=role_metadata["promotion_receipt"].relative_path,
            )
            receipt = _json_without_duplicate_keys(
                receipt_text,
                label=f"{entry_id} promotion receipt",
            )
            _validate_promotion_receipt(
                receipt,
                entry_id=entry_id,
                task=declaration["task"],
                provenance_class=declaration["provenance_class"],
                allowed_consumers=tuple(declaration["allowed_consumers"]),
                recipe_sha256=role_metadata["recipe"].sha256,
                task_memory_sha256=role_metadata["task_memory"].sha256,
            )
        else:
            provenance_text = _strict_text(
                raw_files[role_metadata["provenance"].relative_path],
                label=role_metadata["provenance"].relative_path,
            )
            provenance = _json_without_duplicate_keys(
                provenance_text,
                label=f"{entry_id} provenance",
            )
            if not isinstance(provenance, dict):
                raise BehaviorRecipeCatalogError(
                    f"resource provenance must be a JSON object: {entry_id}"
                )
            source = provenance.get("source")
            _validate_source_binding(
                source,
                entry_id=entry_id,
                task=declaration["task"],
            )
            _validate_resource_provenance(
                provenance,
                entry_id=entry_id,
                task=declaration["task"],
                source=source,
                recipe_sha256=role_metadata["recipe"].sha256,
                memory_sha256=role_metadata["memory"].sha256,
            )
            receipt_text = _strict_text(
                raw_files[role_metadata["receipt"].relative_path],
                label=role_metadata["receipt"].relative_path,
            )
            receipt = _json_without_duplicate_keys(
                receipt_text,
                label=f"{entry_id} receipt",
            )
            _validate_resource_receipt(
                receipt,
                entry_id=entry_id,
                task=declaration["task"],
                provenance_class=declaration["provenance_class"],
                allowed_consumers=tuple(declaration["allowed_consumers"]),
                source=source,
                recipe_sha256=role_metadata["recipe"].sha256,
                memory_sha256=role_metadata["memory"].sha256,
                provenance_sha256=role_metadata["provenance"].sha256,
            )
        entries[entry_id] = BehaviorRecipeEntry(
            entry_id=entry_id,
            task=declaration["task"],
            provenance_class=declaration["provenance_class"],
            allowed_consumers=tuple(declaration["allowed_consumers"]),
            entry_sha256=declaration["entry_sha256"],
            recipe_text=recipe_text,
            task_memory_text=memory_text,
            files=MappingProxyType(role_metadata),
        )

    binding = RecipeCatalogManifestBinding(
        schema_version=schema_version,
        kind=CATALOG_MANIFEST_KIND,
        environment="behavior",
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        declared_catalog_sha256=declared_catalog_sha,
    )
    return BehaviorRecipeCatalog(
        schema_version=schema_version,
        catalog_sha256=actual_catalog_sha,
        entries=MappingProxyType(entries),
        files=MappingProxyType(metadata),
        manifest_binding=binding,
    )


__all__ = [
    "CANONICAL_PUBLIC_PROVENANCE",
    "CANDIDATE_REVIEWED_PROVENANCE",
    "CATALOG_MANIFEST",
    "EXPLORE_CONSUMER",
    "FORMAL_EVAL_CONSUMER",
    "BehaviorRecipeCatalog",
    "BehaviorRecipeCatalogError",
    "BehaviorRecipeEntry",
    "BehaviorRecipeSelection",
    "RecipeCatalogManifestBinding",
    "RecipeFileMetadata",
    "load_behavior_recipe_catalog",
]
