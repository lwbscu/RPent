"""Load the reviewed, versioned BEHAVIOR Global Memory snapshot."""

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

MEMORY_ENTRYPOINT = "MEMORY.md"
MEMORY_README = "README.md"
MEMORY_MANIFEST = "snapshot_manifest.json"
MEMORY_MANIFEST_KIND = "reviewed_behavior_memory_snapshot"
MEMORY_MANIFEST_SCHEMA_VERSION = 1
MAX_MEMORY_FILES = 32
MAX_MEMORY_FILE_BYTES = 128 * 1024
MAX_MEMORY_TOTAL_BYTES = 512 * 1024
TARGET_PRIOR_FILENAME = "target_prior.md"
EXPLORE_EXPERIENCE_FILENAME = "explore_experience.md"

_ADDITIONAL_TASK_EXPERT_KNOWLEDGE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "turning_on_radio": ("control_face_target_lock.md",),
        "picking_up_trash": (),
    }
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_PORTABLE_PATH_PART_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_FORBIDDEN_PROMPT_PATTERNS = (
    re.compile(r"/(?:home|tmp|mnt)/"),
    re.compile(
        r"(?i)\b(?:activity[-_ ]?(?:definition|instance)[-_ ]?id|"
        r"native[-_ ]?(?:binding|instance|seed))\b"
    ),
    re.compile(r"(?i)\b(?:left|right)[-_ ]?(?:hand|arm|wrist|gripper)\b"),
    re.compile(r"(?i)\b[xyzuv]\s*[:=]\s*[-+]?\d+(?:\.\d+)?"),
    re.compile(r"(?i)\b(?:pixel|row|col(?:umn)?)\b"),
    re.compile(
        r"(?<![\w.])[\[(]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:\s*,\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)){1,5}\s*[\])]"
    ),
    re.compile(
        r"(?i)\b(?:close|move_to|observe|open|pi0_nav_pick|pixel_to_world|"
        r"press|rotate_wrist|save_robot_state_checkpoint|max_chunks|"
        r"max_vla_chunks_per_call|max_total_vla_chunks|call_chunk_limit)\b"
    ),
    re.compile(
        r"(?i)\bchunks?\s*(?:(?:=|:)\s*|\bto\s+)"
        r"(?:[a-z_][a-z0-9_]*|\d+)\b"
    ),
    re.compile(r"(?i)\b\d+\s+(?:complete\s+)?(?:pi0(?:\.5)?[- ]?)?chunks?\b"),
    re.compile(r"(?i)\b(?:camera|view)\s*(?:order|schedule|sequence|plan)\b"),
    re.compile(r"(?i)\b(?:once|twice|\d+\s*(?:times?|calls?|invocations?))\b"),
    re.compile(
        r"(?i)\b(?:call|invocation|retry)\s*(?:count|limit|budget)"
        r"\s*[:=]\s*\d+\b"
    ),
    re.compile(
        r"(?i)\b(?:first|second|third|next|then|finally|subsequently|"
        r"afterwards|before|after)\b"
    ),
    re.compile(
        r"(?i)</?(?:reviewed_memory_file|target_prior|explore_experience|"
        r"additional_expert_knowledge|reviewed_repo_memory_manifest)\b"
    ),
)


class BehaviorMemorySnapshotError(ValueError):
    """Raised when the reviewed BEHAVIOR memory snapshot is not trustworthy."""


@dataclass(frozen=True)
class MemoryFileMetadata:
    """Verified metadata for one file in the reviewed snapshot."""

    relative_path: str
    size_bytes: int
    sha256: str
    included_in_prompt: bool


@dataclass(frozen=True)
class MemoryManifestBinding:
    """Verified identity and digest binding of ``snapshot_manifest.json``."""

    schema_version: int
    kind: str
    environment: str
    entrypoint: str
    manifest_sha256: str
    declared_snapshot_sha256: str


@dataclass(frozen=True)
class BehaviorMemorySnapshot:
    """Immutable view of one validated reviewed-memory snapshot."""

    prompt_text: str
    snapshot_sha256: str
    files: Mapping[str, MemoryFileMetadata]
    manifest_binding: MemoryManifestBinding
    file_texts: Mapping[str, str]
    indexed_leaf_paths: tuple[str, ...]

    def select_task(self, task_name: str) -> BehaviorTaskMemorySelection:
        """Select the exact reviewed knowledge directory for one task."""

        try:
            task_spec = get_task_spec(task_name)
        except ValueError as error:
            raise BehaviorMemorySnapshotError(
                f"no reviewed memory directory is registered for task: {task_name!r}"
            ) from error
        directory = _portable_relative_path(
            task_spec.task_name, label="task memory directory"
        )
        if "/" in directory:
            raise BehaviorMemorySnapshotError(
                f"task memory directory must be one path component: {task_name!r}"
            )
        target_prior_path = f"{directory}/{TARGET_PRIOR_FILENAME}"
        explore_experience_path = f"{directory}/{EXPLORE_EXPERIENCE_FILENAME}"
        raw_additional_paths = _ADDITIONAL_TASK_EXPERT_KNOWLEDGE.get(task_name, ())
        additional_paths = tuple(
            _portable_relative_path(
                f"{directory}/{filename}",
                label="additional expert knowledge path",
            )
            for filename in raw_additional_paths
        )
        selected_paths = (
            target_prior_path,
            explore_experience_path,
            *additional_paths,
        )
        if len(set(selected_paths)) != len(selected_paths):
            raise BehaviorMemorySnapshotError(
                f"reviewed task memory route contains duplicate paths: {task_name!r}"
            )
        missing = sorted(set(selected_paths).difference(self.indexed_leaf_paths))
        if missing:
            raise BehaviorMemorySnapshotError(
                f"reviewed task memory is incomplete for {task_name!r}: {missing}"
            )
        indexed_task_paths = {
            path for path in self.indexed_leaf_paths if path.startswith(f"{directory}/")
        }
        unregistered = sorted(indexed_task_paths.difference(selected_paths))
        if unregistered:
            raise BehaviorMemorySnapshotError(
                f"reviewed task knowledge is not explicitly registered: {unregistered}"
            )
        empty = [path for path in selected_paths if not self.file_texts[path].strip()]
        if empty:
            raise BehaviorMemorySnapshotError(
                f"reviewed task memory files must be non-empty: {empty}"
            )
        prompt_paths = selected_paths
        prompt_text = _render_prompt_sections(prompt_paths, self.file_texts)
        selection_records = [
            {
                "path": path,
                "sha256": self.files[path].sha256,
                "size_bytes": self.files[path].size_bytes,
            }
            for path in prompt_paths
        ]
        selection_payload = {
            "task_name": task_name,
            "task_directory": directory,
            "files": selection_records,
        }
        selection_sha256 = hashlib.sha256(
            json.dumps(
                selection_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        additional_expert_knowledge = _render_prompt_sections(
            additional_paths, self.file_texts
        )
        if not additional_expert_knowledge:
            additional_expert_knowledge = (
                "No additional reviewed expert knowledge is registered for this task."
            )
        return BehaviorTaskMemorySelection(
            task_name=task_name,
            task_directory=directory,
            target_prior_text=self.file_texts[target_prior_path].strip(),
            explore_experience_text=self.file_texts[explore_experience_path].strip(),
            additional_expert_knowledge_text=additional_expert_knowledge,
            prompt_text=prompt_text,
            selected_paths=selected_paths,
            selection_sha256=selection_sha256,
            prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            files=MappingProxyType({path: self.files[path] for path in prompt_paths}),
        )


@dataclass(frozen=True)
class BehaviorTaskMemorySelection:
    """Task-scoped, hash-bound view of reviewed BEHAVIOR knowledge."""

    task_name: str
    task_directory: str
    target_prior_text: str
    explore_experience_text: str
    additional_expert_knowledge_text: str
    prompt_text: str
    selected_paths: tuple[str, ...]
    selection_sha256: str
    prompt_sha256: str
    files: Mapping[str, MemoryFileMetadata]

    @property
    def public_binding(self) -> dict[str, Any]:
        """Return JSON-safe provenance for the selected task knowledge."""

        return {
            "task_name": self.task_name,
            "task_directory": self.task_directory,
            "selection_sha256": self.selection_sha256,
            "prompt_sha256": self.prompt_sha256,
            "roles": {
                "target_prior": (self.selected_paths[0]),
                "explore_experience": (self.selected_paths[1]),
                "additional_expert_knowledge": list(self.selected_paths[2:]),
            },
            "files": {
                path: {
                    "relative_path": metadata.relative_path,
                    "size_bytes": metadata.size_bytes,
                    "sha256": metadata.sha256,
                }
                for path, metadata in self.files.items()
            },
        }


def _strict_text(raw: bytes, *, label: str) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BehaviorMemorySnapshotError(f"{label} must be strict UTF-8") from error
    if "\x00" in text:
        raise BehaviorMemorySnapshotError(f"{label} must not contain NUL bytes")
    return text


def _json_without_duplicate_keys(text: str) -> Any:
    def build_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BehaviorMemorySnapshotError(
                    f"snapshot manifest contains duplicate key: {key}"
                )
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=build_object)
    except json.JSONDecodeError as error:
        raise BehaviorMemorySnapshotError(
            "snapshot manifest must be valid JSON"
        ) from error


def _portable_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BehaviorMemorySnapshotError(f"{label} must be a non-empty path")
    if "\\" in value or "%" in value:
        raise BehaviorMemorySnapshotError(f"{label} is not a portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BehaviorMemorySnapshotError(f"{label} must not contain traversal")
    if any(_PORTABLE_PATH_PART_PATTERN.fullmatch(part) is None for part in path.parts):
        raise BehaviorMemorySnapshotError(f"{label} is not a portable relative path")
    return path.as_posix()


def _read_regular_file(root: Path, relative_path: str) -> bytes:
    relative = PurePosixPath(relative_path)
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if cursor.is_symlink() or not cursor.is_dir():
            raise BehaviorMemorySnapshotError(
                f"memory path parent is unsafe: {relative_path}"
            )
    path = root.joinpath(*relative.parts)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise BehaviorMemorySnapshotError(
            f"memory file must be a readable regular file: {relative_path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BehaviorMemorySnapshotError(
                f"memory file must be regular: {relative_path}"
            )
        if before.st_size > MAX_MEMORY_FILE_BYTES:
            raise BehaviorMemorySnapshotError(
                f"memory file exceeds {MAX_MEMORY_FILE_BYTES} bytes: {relative_path}"
            )
        chunks: list[bytes] = []
        remaining = MAX_MEMORY_FILE_BYTES + 1
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
        raise BehaviorMemorySnapshotError(
            f"memory file changed while being read: {relative_path}"
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
            raise BehaviorMemorySnapshotError(
                f"memory directory is unreadable: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            if entry.is_symlink():
                raise BehaviorMemorySnapshotError(
                    f"symlink is forbidden in reviewed memory: {path}"
                )
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise BehaviorMemorySnapshotError(
                    f"non-regular entry is forbidden in reviewed memory: {path}"
                )
            relative = path.relative_to(root).as_posix()
            discovered.add(_portable_relative_path(relative, label="memory file"))
    return discovered


def _aggregate_snapshot_sha256(
    records: Mapping[str, tuple[int, str]],
) -> str:
    canonical = [
        {
            "path": path,
            "sha256": records[path][1],
            "size_bytes": records[path][0],
        }
        for path in sorted(records)
    ]
    raw = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _indexed_leaf_paths(entrypoint: str) -> list[str]:
    indexed: list[str] = []
    for raw_target in _MARKDOWN_LINK_PATTERN.findall(entrypoint):
        target = raw_target.strip()
        if any(character.isspace() for character in target):
            raise BehaviorMemorySnapshotError(
                "MEMORY.md links must not contain titles or whitespace"
            )
        path = _portable_relative_path(target, label="MEMORY.md link")
        if not path.endswith(".md"):
            raise BehaviorMemorySnapshotError("MEMORY.md may index only Markdown files")
        if path in {MEMORY_ENTRYPOINT, MEMORY_README}:
            continue
        if path in indexed:
            raise BehaviorMemorySnapshotError(f"duplicate MEMORY.md leaf link: {path}")
        indexed.append(path)
    return indexed


def _validate_prompt_text(prompt_text: str) -> None:
    for pattern in _FORBIDDEN_PROMPT_PATTERNS:
        if pattern.search(prompt_text):
            raise BehaviorMemorySnapshotError(
                "reviewed memory contains run-specific or prescriptive guidance"
            )


def _render_prompt_sections(
    prompt_paths: tuple[str, ...] | list[str],
    file_texts: Mapping[str, str],
) -> str:
    return "\n\n".join(
        f'<reviewed_memory_file path="{path}">\n'
        f"{file_texts[path].strip()}\n"
        "</reviewed_memory_file>"
        for path in prompt_paths
    )


def load_behavior_memory_snapshot(
    memory_root: str | Path,
) -> BehaviorMemorySnapshot:
    """Load and validate a reviewed BEHAVIOR memory directory.

    The loader never writes to ``memory_root``. It verifies the manifest, the
    complete regular-file set, every file digest, the aggregate snapshot digest,
    and the one-to-one relationship between task leaves and ``MEMORY.md`` links.
    """

    requested_root = Path(memory_root).expanduser().absolute()
    if requested_root.is_symlink():
        raise BehaviorMemorySnapshotError("memory root must not be a symlink")
    try:
        root = requested_root.resolve(strict=True)
    except OSError as error:
        raise BehaviorMemorySnapshotError("memory root does not exist") from error
    if not root.is_dir():
        raise BehaviorMemorySnapshotError("memory root must be a directory")

    discovered = _discover_regular_files(root)
    if MEMORY_MANIFEST not in discovered:
        raise BehaviorMemorySnapshotError(f"reviewed memory omitted {MEMORY_MANIFEST}")
    if MEMORY_ENTRYPOINT not in discovered or MEMORY_README not in discovered:
        raise BehaviorMemorySnapshotError(
            f"reviewed memory requires {MEMORY_ENTRYPOINT} and {MEMORY_README}"
        )

    manifest_raw = _read_regular_file(root, MEMORY_MANIFEST)
    manifest_text = _strict_text(manifest_raw, label=MEMORY_MANIFEST)
    manifest = _json_without_duplicate_keys(manifest_text)
    if not isinstance(manifest, dict):
        raise BehaviorMemorySnapshotError("snapshot manifest must be a JSON object")
    expected_manifest_keys = {
        "schema_version",
        "kind",
        "environment",
        "entrypoint",
        "files",
        "snapshot_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise BehaviorMemorySnapshotError("snapshot manifest fields are not canonical")
    if (
        manifest.get("schema_version") != MEMORY_MANIFEST_SCHEMA_VERSION
        or manifest.get("kind") != MEMORY_MANIFEST_KIND
        or manifest.get("environment") != "behavior"
        or manifest.get("entrypoint") != MEMORY_ENTRYPOINT
    ):
        raise BehaviorMemorySnapshotError("snapshot manifest binding is invalid")

    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        raise BehaviorMemorySnapshotError(
            "snapshot manifest files must be a non-empty object"
        )
    if len(declared_files) > MAX_MEMORY_FILES:
        raise BehaviorMemorySnapshotError(
            f"reviewed memory exceeds {MAX_MEMORY_FILES} tracked files"
        )

    canonical_declarations: dict[str, tuple[int, str]] = {}
    for raw_path, record in declared_files.items():
        path = _portable_relative_path(raw_path, label="manifest file")
        if path == MEMORY_MANIFEST or not path.endswith(".md"):
            raise BehaviorMemorySnapshotError(
                "manifest may track only reviewed Markdown files"
            )
        if not isinstance(record, dict) or set(record) != {"size_bytes", "sha256"}:
            raise BehaviorMemorySnapshotError(
                f"manifest metadata is invalid for {path}"
            )
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > MAX_MEMORY_FILE_BYTES
            or not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise BehaviorMemorySnapshotError(
                f"manifest metadata is invalid for {path}"
            )
        if path in canonical_declarations:
            raise BehaviorMemorySnapshotError(f"duplicate manifest file path: {path}")
        canonical_declarations[path] = (size, digest)

    expected_discovered = set(canonical_declarations) | {MEMORY_MANIFEST}
    if discovered != expected_discovered:
        extra = sorted(discovered - expected_discovered)
        missing = sorted(expected_discovered - discovered)
        raise BehaviorMemorySnapshotError(
            f"reviewed memory file set mismatch; extra={extra}, missing={missing}"
        )

    file_bytes: dict[str, bytes] = {}
    total_bytes = len(manifest_raw)
    for path, (declared_size, declared_digest) in canonical_declarations.items():
        raw = _read_regular_file(root, path)
        total_bytes += len(raw)
        if total_bytes > MAX_MEMORY_TOTAL_BYTES:
            raise BehaviorMemorySnapshotError(
                f"reviewed memory exceeds {MAX_MEMORY_TOTAL_BYTES} total bytes"
            )
        _strict_text(raw, label=path)
        if len(raw) != declared_size:
            raise BehaviorMemorySnapshotError(f"memory file size mismatch: {path}")
        if hashlib.sha256(raw).hexdigest() != declared_digest:
            raise BehaviorMemorySnapshotError(f"memory file SHA256 mismatch: {path}")
        file_bytes[path] = raw

    declared_snapshot = manifest.get("snapshot_sha256")
    actual_snapshot = _aggregate_snapshot_sha256(canonical_declarations)
    if (
        not isinstance(declared_snapshot, str)
        or _SHA256_PATTERN.fullmatch(declared_snapshot) is None
        or declared_snapshot != actual_snapshot
    ):
        raise BehaviorMemorySnapshotError("aggregate snapshot SHA256 mismatch")

    entrypoint_text = _strict_text(
        file_bytes[MEMORY_ENTRYPOINT], label=MEMORY_ENTRYPOINT
    )
    indexed_leaves = _indexed_leaf_paths(entrypoint_text)
    declared_leaves = {
        path
        for path in canonical_declarations
        if path not in {MEMORY_ENTRYPOINT, MEMORY_README}
    }
    if set(indexed_leaves) != declared_leaves:
        unindexed = sorted(declared_leaves - set(indexed_leaves))
        missing = sorted(set(indexed_leaves) - declared_leaves)
        raise BehaviorMemorySnapshotError(
            f"MEMORY.md leaf index mismatch; unindexed={unindexed}, missing={missing}"
        )

    file_texts = {
        path: _strict_text(raw, label=path) for path, raw in file_bytes.items()
    }
    prompt_paths = [MEMORY_ENTRYPOINT, *indexed_leaves]
    for path in prompt_paths:
        _validate_prompt_text(file_texts[path])
    prompt_text = _render_prompt_sections(prompt_paths, file_texts)

    prompt_set = set(prompt_paths)
    metadata = {
        path: MemoryFileMetadata(
            relative_path=path,
            size_bytes=canonical_declarations[path][0],
            sha256=canonical_declarations[path][1],
            included_in_prompt=path in prompt_set,
        )
        for path in sorted(canonical_declarations)
    }
    binding = MemoryManifestBinding(
        schema_version=MEMORY_MANIFEST_SCHEMA_VERSION,
        kind=MEMORY_MANIFEST_KIND,
        environment="behavior",
        entrypoint=MEMORY_ENTRYPOINT,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        declared_snapshot_sha256=declared_snapshot,
    )
    return BehaviorMemorySnapshot(
        prompt_text=prompt_text,
        snapshot_sha256=actual_snapshot,
        files=MappingProxyType(metadata),
        manifest_binding=binding,
        file_texts=MappingProxyType(file_texts),
        indexed_leaf_paths=tuple(indexed_leaves),
    )


__all__ = [
    "BehaviorMemorySnapshot",
    "BehaviorMemorySnapshotError",
    "BehaviorTaskMemorySelection",
    "EXPLORE_EXPERIENCE_FILENAME",
    "MAX_MEMORY_FILE_BYTES",
    "MAX_MEMORY_FILES",
    "MAX_MEMORY_TOTAL_BYTES",
    "MemoryFileMetadata",
    "MemoryManifestBinding",
    "TARGET_PRIOR_FILENAME",
    "load_behavior_memory_snapshot",
]
