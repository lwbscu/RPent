"""Content-addressed, read-only source snapshots for formal BEHAVIOR runs.

Formal evaluation normally requires a clean Git checkout. A reviewed dirty
worktree can instead be copied once into a hash-sealed snapshot and evaluated
from that copy. Every consumer validates the snapshot before importing code.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_SNAPSHOT_FILENAME = "source_snapshot.json"
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
_EXCLUDED_PATHS = (".git/", "<git-ignored-paths>")


class SourceSnapshotError(RuntimeError):
    """Raised when a source snapshot cannot be created or validated."""


@dataclass(frozen=True)
class SourceSnapshotBinding:
    """Validated identity plus local paths that are outside the content hash."""

    snapshot_root: Path
    binding_path: Path
    binding_sha256: str
    tree_sha256: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical manifest value; local paths remain separate."""

        return {**self.payload, "binding_sha256": self.binding_sha256}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceSnapshotError(
            f"source root is not a readable Git worktree: {root}"
        ) from error


def _git_text(root: Path, *arguments: str) -> str:
    try:
        return _git_bytes(root, *arguments).decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise SourceSnapshotError("Git metadata is not valid UTF-8") from error


def _snapshot_source_paths(source_root: Path) -> tuple[str, ...]:
    raw = _git_bytes(
        source_root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    )
    paths: list[str] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        relative = os.fsdecode(item)
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SourceSnapshotError(f"unsafe Git source path: {relative!r}")
        if relative == SOURCE_SNAPSHOT_FILENAME:
            raise SourceSnapshotError(
                f"source worktree must not contain {SOURCE_SNAPSHOT_FILENAME}"
            )
        source_path = source_root / candidate
        # Tracked deletions have no runnable bytes; status_sha256 binds them.
        if source_path.is_symlink() or source_path.is_file():
            paths.append(candidate.as_posix())
    return tuple(sorted(set(paths)))


def _validate_internal_symlink(root: Path, relative: str, target: str) -> None:
    target_path = Path(target)
    if target_path.is_absolute():
        raise SourceSnapshotError(f"snapshot symlink must be relative: {relative}")
    resolved = (root / relative).parent.joinpath(target_path).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise SourceSnapshotError(
            f"snapshot symlink escapes the source root: {relative}"
        ) from error
    if not resolved.exists():
        raise SourceSnapshotError(
            f"snapshot symlink target is not included or readable: {relative}"
        )


def _make_payload_files_read_only(root: Path) -> None:
    for entry in root.rglob("*"):
        if entry.is_symlink() or not entry.is_file():
            continue
        mode = stat.S_IMODE(entry.stat().st_mode)
        entry.chmod(mode & ~0o222)


def _make_snapshot_read_only(root: Path) -> None:
    entries = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for entry in entries:
        if entry.is_symlink():
            continue
        mode = stat.S_IMODE(entry.stat().st_mode)
        entry.chmod(mode & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def _file_record(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    file_stat = path.lstat()
    mode = stat.S_IMODE(file_stat.st_mode)
    if stat.S_ISLNK(file_stat.st_mode):
        target = os.readlink(path)
        _validate_internal_symlink(root, relative, target)
        return {
            "path": relative,
            "kind": "symlink",
            "mode": mode,
            "size": len(os.fsencode(target)),
            "sha256": hashlib.sha256(os.fsencode(target)).hexdigest(),
            "target": target,
        }
    if not stat.S_ISREG(file_stat.st_mode):
        raise SourceSnapshotError(f"source entry is not a regular file: {relative}")
    return {
        "path": relative,
        "kind": "file",
        "mode": mode,
        "size": file_stat.st_size,
        "sha256": _sha256_file(path),
    }


def _tree_sha256(files: list[dict[str, Any]]) -> str:
    # Local paths, timestamps, and Git metadata intentionally do not participate.
    return hashlib.sha256(_canonical_json_bytes(files)).hexdigest()


def _binding_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _binding_from_payload(root: Path, payload: dict[str, Any]) -> SourceSnapshotBinding:
    binding_sha256 = _binding_sha256(payload)
    return SourceSnapshotBinding(
        snapshot_root=root,
        binding_path=root / SOURCE_SNAPSHOT_FILENAME,
        binding_sha256=binding_sha256,
        tree_sha256=str(payload["source_tree_sha256"]),
        payload=payload,
    )


def create_source_snapshot(
    source_root: str | os.PathLike[str],
    destination_root: str | os.PathLike[str],
) -> SourceSnapshotBinding:
    """Copy one Git worktree state into a new hash-sealed read-only directory.

    The destination must not already exist and must be outside the source
    worktree. No commit is created and the source worktree is never modified.
    """

    source = Path(source_root).expanduser().resolve(strict=True)
    destination = Path(destination_root).expanduser().absolute()
    if not source.is_dir():
        raise SourceSnapshotError(f"source root is not a directory: {source}")
    try:
        destination.resolve(strict=False).relative_to(source)
    except ValueError:
        pass
    else:
        raise SourceSnapshotError("source snapshot destination must be outside source")
    if destination.exists():
        raise SourceSnapshotError(
            f"source snapshot destination already exists: {destination}"
        )

    top = Path(_git_text(source, "rev-parse", "--show-toplevel")).resolve()
    if top != source:
        raise SourceSnapshotError(
            f"source_root must be the Git toplevel: expected {top}, got {source}"
        )
    commit = _git_text(source, "rev-parse", "HEAD")
    branch = _git_text(source, "branch", "--show-current")
    status = _git_bytes(
        source,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=normal",
    )
    relative_paths = _snapshot_source_paths(source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid():x}-{time.time_ns():x}"
    )
    temporary.mkdir(mode=0o700)
    try:
        for relative in relative_paths:
            source_path = source / relative
            target_path = temporary / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.is_symlink():
                link_target = os.readlink(source_path)
                _validate_internal_symlink(source, relative, link_target)
                os.symlink(link_target, target_path)
            else:
                shutil.copy2(source_path, target_path, follow_symlinks=False)
        _make_payload_files_read_only(temporary)
        files = [_file_record(temporary, relative) for relative in relative_paths]
        payload = {
            "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "kind": "hash_sealed_source_snapshot",
            "source_tree_sha256": _tree_sha256(files),
            "base_git": {
                "commit": commit,
                "branch": branch,
                "worktree_dirty": bool(status),
                "status_sha256": hashlib.sha256(status).hexdigest(),
            },
            "files": files,
            "excluded_paths": list(_EXCLUDED_PATHS),
        }
        binding = _binding_from_payload(destination, payload)
        (temporary / SOURCE_SNAPSHOT_FILENAME).write_bytes(
            _canonical_json_bytes(binding.as_dict()) + b"\n"
        )
        os.replace(temporary, destination)
        _make_snapshot_read_only(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_source_snapshot(destination, binding.binding_sha256)


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / SOURCE_SNAPSHOT_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SourceSnapshotError(
            f"source snapshot manifest is missing: {manifest_path}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceSnapshotError("source snapshot manifest is invalid JSON") from error
    if not isinstance(value, dict):
        raise SourceSnapshotError("source snapshot manifest must be an object")
    return value


def validate_source_snapshot(
    snapshot_root: str | os.PathLike[str],
    expected_binding_sha256: str,
) -> SourceSnapshotBinding:
    """Validate every file, tree digest, binding, and read-only mode."""

    root = Path(snapshot_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise SourceSnapshotError("source snapshot root must be a real directory")
    expected = str(expected_binding_sha256).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise SourceSnapshotError("expected source snapshot binding must be SHA-256")
    manifest = _read_manifest(root)
    recorded_binding = manifest.pop("binding_sha256", None)
    if (
        manifest.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION
        or manifest.get("kind") != "hash_sealed_source_snapshot"
        or not isinstance(recorded_binding, str)
        or _binding_sha256(manifest) != recorded_binding
        or recorded_binding != expected
    ):
        raise SourceSnapshotError("source snapshot binding SHA-256 mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SourceSnapshotError("source snapshot files must be a list")
    relative_paths: list[str] = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SourceSnapshotError("source snapshot file record is invalid")
        relative = item["path"]
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SourceSnapshotError(f"unsafe snapshot path: {relative!r}")
        relative_paths.append(relative)
    if relative_paths != sorted(set(relative_paths)):
        raise SourceSnapshotError("source snapshot file paths are not canonical")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != set(relative_paths) | {SOURCE_SNAPSHOT_FILENAME}:
        raise SourceSnapshotError("source snapshot contains unbound files")
    actual_files = [_file_record(root, relative) for relative in relative_paths]
    if actual_files != files:
        raise SourceSnapshotError("source snapshot file metadata changed")
    if _tree_sha256(actual_files) != manifest.get("source_tree_sha256"):
        raise SourceSnapshotError("source snapshot tree SHA-256 mismatch")
    for record in actual_files:
        if record["kind"] == "file":
            path = root / str(record["path"])
            if stat.S_IMODE(path.stat().st_mode) & 0o222:
                raise SourceSnapshotError(
                    f"source snapshot file is writable: {record['path']}"
                )
    if stat.S_IMODE(root.stat().st_mode) & 0o222:
        raise SourceSnapshotError("source snapshot root is writable")
    for directory in (path for path in root.rglob("*") if path.is_dir()):
        if stat.S_IMODE(directory.stat().st_mode) & 0o222:
            raise SourceSnapshotError(
                "source snapshot directory is writable: "
                f"{directory.relative_to(root).as_posix()}"
            )
    if stat.S_IMODE((root / SOURCE_SNAPSHOT_FILENAME).stat().st_mode) & 0o222:
        raise SourceSnapshotError("source snapshot manifest is writable")
    return SourceSnapshotBinding(
        snapshot_root=root,
        binding_path=root / SOURCE_SNAPSHOT_FILENAME,
        binding_sha256=recorded_binding,
        tree_sha256=str(manifest["source_tree_sha256"]),
        payload=manifest,
    )


__all__ = [
    "SOURCE_SNAPSHOT_FILENAME",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "SourceSnapshotBinding",
    "SourceSnapshotError",
    "create_source_snapshot",
    "validate_source_snapshot",
]
