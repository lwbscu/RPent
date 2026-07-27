"""Synchronize resource payloads from their HuggingFace dataset.

``ensure_resources`` preserves LIBERO's best-effort, mutable checkout behavior.
Strict consumers such as BEHAVIOR use
``prepare_pinned_dataset_resources`` instead: it resolves a mutable revision
once, materializes that exact commit in a versioned cache, and validates a
closed, hash-bound subtree manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from rpent.utils.config import get_resources_dir
from rpent.utils.logging import get_logger

RESOURCES_HF_REPO = os.environ.get("RPENT_RESOURCES_HF_REPO", "RLinf/RPent-memory")
PINNED_RESOURCE_MANIFEST = "manifest.json"
PINNED_RESOURCE_SCHEMA_VERSION = 1
_FULL_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LOCAL_RESOURCE_ID = "local"

logger = get_logger("resources")


def get_resources_cache_dir() -> Path:
    """Return the BEHAVIOR-local immutable resource cache root."""

    configured = os.environ.get("RPENT_RESOURCES_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return (cache_home / "rpent" / "resources").resolve()


class ResourcePreparationError(RuntimeError):
    """A strict dataset resource could not be prepared safely."""


class ResourceManifestError(ResourcePreparationError):
    """A strict dataset resource manifest or one of its files is invalid."""


@dataclass(frozen=True)
class DatasetResourceFile:
    """One file sealed by a strict dataset subtree manifest."""

    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "DatasetResourceFile":
        if not isinstance(value, dict) or set(value) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ResourceManifestError(
                "resource file binding must contain exactly "
                "path, size_bytes, and sha256"
            )
        path = value["path"]
        size_bytes = value["size_bytes"]
        sha256 = value["sha256"]
        if not isinstance(path, str):
            raise ResourceManifestError("resource file path must be a string")
        _validate_relative_path(path)
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ResourceManifestError(
                f"invalid size_bytes for resource file {path!r}"
            )
        if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
            raise ResourceManifestError(f"invalid sha256 for resource file {path!r}")
        return cls(path=path, size_bytes=size_bytes, sha256=sha256)


@dataclass(frozen=True)
class DatasetResourceBinding:
    """JSON-safe binding for one immutable, validated dataset subtree."""

    dataset_repo: str
    repo_type: str
    requested_revision: str
    resolved_revision: str
    subtree: str
    manifest_sha256: str
    files: tuple[DatasetResourceFile, ...]
    offline: bool
    root: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_repo": self.dataset_repo,
            "repo_type": self.repo_type,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "subtree": self.subtree,
            "manifest_sha256": self.manifest_sha256,
            "files": [item.as_dict() for item in self.files],
            "offline": self.offline,
            "root": str(self.root),
        }

    @classmethod
    def from_dict(cls, value: object) -> "DatasetResourceBinding":
        expected = {
            "dataset_repo",
            "repo_type",
            "requested_revision",
            "resolved_revision",
            "subtree",
            "manifest_sha256",
            "files",
            "offline",
            "root",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ResourceManifestError(
                "dataset resource binding has unexpected or missing fields"
            )
        strings: dict[str, str] = {}
        for key in (
            "dataset_repo",
            "repo_type",
            "requested_revision",
            "resolved_revision",
            "subtree",
            "manifest_sha256",
            "root",
        ):
            item = value[key]
            if not isinstance(item, str) or not item:
                raise ResourceManifestError(
                    f"dataset resource binding {key} must be a non-empty string"
                )
            strings[key] = item
        if not _FULL_REVISION_RE.fullmatch(strings["resolved_revision"]):
            raise ResourceManifestError(
                "dataset resource resolved_revision must be a full 40-hex "
                "immutable revision"
            )
        if not _SHA256_RE.fullmatch(strings["manifest_sha256"]):
            raise ResourceManifestError(
                "dataset resource manifest_sha256 must be lowercase SHA-256"
            )
        _validate_subtree(strings["subtree"])
        raw_files = value["files"]
        if not isinstance(raw_files, list):
            raise ResourceManifestError("dataset resource files must be a list")
        files = tuple(DatasetResourceFile.from_dict(item) for item in raw_files)
        if tuple(item.path for item in files) != tuple(
            sorted(item.path for item in files)
        ):
            raise ResourceManifestError("dataset resource files must be sorted by path")
        if len({item.path for item in files}) != len(files):
            raise ResourceManifestError(
                "dataset resource files contain duplicate paths"
            )
        offline = value["offline"]
        if not isinstance(offline, bool):
            raise ResourceManifestError(
                "dataset resource binding offline must be a boolean"
            )
        root = Path(strings["root"]).expanduser()
        if not root.is_absolute():
            raise ResourceManifestError(
                "dataset resource binding root must be absolute"
            )
        is_local = (
            strings["dataset_repo"] == LOCAL_RESOURCE_ID
            or strings["repo_type"] == LOCAL_RESOURCE_ID
        )
        if is_local:
            if (
                strings["dataset_repo"] != LOCAL_RESOURCE_ID
                or strings["repo_type"] != LOCAL_RESOURCE_ID
                or offline is not True
                or not _SHA256_RE.fullmatch(strings["requested_revision"])
                or strings["requested_revision"] != strings["manifest_sha256"]
                or strings["resolved_revision"].lower()
                != strings["manifest_sha256"][:40]
                or root.name != strings["subtree"]
                or root.parent.name != strings["manifest_sha256"]
                or root.parent.parent.name != LOCAL_RESOURCE_ID
            ):
                raise ResourceManifestError(
                    "local resource binding identity or content-addressed root "
                    "is invalid"
                )
        return cls(
            dataset_repo=strings["dataset_repo"],
            repo_type=strings["repo_type"],
            requested_revision=strings["requested_revision"],
            resolved_revision=strings["resolved_revision"].lower(),
            subtree=strings["subtree"],
            manifest_sha256=strings["manifest_sha256"],
            files=files,
            offline=offline,
            root=root,
        )


def ensure_resources(env_name: str) -> Path:
    """Sync the env's resources from HuggingFace each run; set HF_HUB_OFFLINE=1 to use the local copy only. Memory is optional."""
    resources_dir = get_resources_dir(env_name)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return resources_dir

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=RESOURCES_HF_REPO,
            repo_type="dataset",
            local_dir=str(resources_dir.parent),
            allow_patterns=[f"{env_name}/**"],
        )
    except Exception as exc:
        logger.warning(
            "could not sync '%s' from '%s': %s; continuing with local files under %s",
            env_name,
            RESOURCES_HF_REPO,
            exc,
            resources_dir,
        )

    return resources_dir


def prepare_pinned_dataset_resources(
    subtree: str,
    *,
    requested_revision: str | None = None,
    dataset_repo: str | None = None,
    repo_type: str = "dataset",
    cache_root: Path | None = None,
    offline: bool | None = None,
) -> DatasetResourceBinding:
    """Prepare and validate one immutable dataset subtree.

    Online mutable revisions (for example ``main``) are resolved exactly once
    to a full commit hash before any download. Offline operation never resolves
    a mutable name: callers must provide the full cached commit hash.
    """
    _validate_subtree(subtree)
    repo = dataset_repo or os.environ.get("RPENT_RESOURCES_HF_REPO", RESOURCES_HF_REPO)
    if not isinstance(repo, str) or not repo.strip():
        raise ResourcePreparationError("dataset_repo must be non-empty")
    repo = repo.strip()
    if not isinstance(repo_type, str) or not repo_type.strip():
        raise ResourcePreparationError("repo_type must be non-empty")
    repo_type = repo_type.strip()

    requested = (
        requested_revision
        if requested_revision is not None
        else os.environ.get("RPENT_RESOURCES_REVISION", "main")
    )
    if not isinstance(requested, str) or not requested.strip():
        raise ResourcePreparationError("requested_revision must be non-empty")
    requested = requested.strip()
    use_offline = (
        os.environ.get("HF_HUB_OFFLINE") == "1" if offline is None else offline
    )
    if not isinstance(use_offline, bool):
        raise ResourcePreparationError("offline must be a boolean")

    if _FULL_REVISION_RE.fullmatch(requested):
        resolved = requested.lower()
    elif use_offline:
        raise ResourcePreparationError(
            "offline pinned resources require a full 40-hex requested revision"
        )
    else:
        resolved = _resolve_dataset_revision(
            dataset_repo=repo,
            repo_type=repo_type,
            requested_revision=requested,
        )

    base = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else get_resources_cache_dir()
    )
    version_root = _materialized_version_root(
        base,
        dataset_repo=repo,
        repo_type=repo_type,
        resolved_revision=resolved,
    )
    resource_root = version_root / subtree

    if version_root.exists():
        if not version_root.is_dir() or version_root.is_symlink():
            raise ResourcePreparationError(
                f"pinned resource version root is not a safe directory: {version_root}"
            )
        return _binding_from_validated_root(
            root=resource_root,
            dataset_repo=repo,
            repo_type=repo_type,
            requested_revision=requested,
            resolved_revision=resolved,
            subtree=subtree,
            offline=use_offline,
        )
    if use_offline:
        raise ResourcePreparationError(
            f"offline pinned resource snapshot is not cached: {version_root}"
        )

    _download_versioned_snapshot(
        version_root=version_root,
        subtree=subtree,
        dataset_repo=repo,
        repo_type=repo_type,
        resolved_revision=resolved,
        requested_revision=requested,
    )
    return _binding_from_validated_root(
        root=resource_root,
        dataset_repo=repo,
        repo_type=repo_type,
        requested_revision=requested,
        resolved_revision=resolved,
        subtree=subtree,
        offline=use_offline,
    )


def prepare_local_dataset_resources(
    subtree: str,
    *,
    source_root: Path,
    cache_root: Path | None = None,
) -> DatasetResourceBinding:
    """Freeze one local closed resource tree into a content-addressed snapshot.

    A local source is never represented as a Hugging Face repository or
    revision.  The complete manifest SHA-256 is its immutable identity and
    cache key; the 40-hex ``resolved_revision`` is retained only for binding
    compatibility with dataset snapshots.
    """

    _validate_subtree(subtree)
    source_input = Path(source_root).expanduser()
    if source_input.is_symlink():
        raise ResourcePreparationError(
            f"local resource source must not be a symlink: {source_input}"
        )
    try:
        source = source_input.resolve(strict=True)
    except OSError as error:
        raise ResourcePreparationError(
            f"local resource source is unavailable: {source_root}"
        ) from error
    if not source.is_dir() or source.is_symlink():
        raise ResourcePreparationError(
            f"local resource source is not a safe directory: {source}"
        )

    manifest_sha256, files = _validate_resource_manifest(source, subtree=subtree)
    cache = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else get_resources_cache_dir()
    )
    version_root = cache / LOCAL_RESOURCE_ID / manifest_sha256
    resource_root = version_root / subtree
    binding = _local_binding_from_validated_root(
        root=resource_root,
        subtree=subtree,
        manifest_sha256=manifest_sha256,
        files=files,
        allow_missing=True,
    )
    if binding is not None:
        return binding

    _install_local_snapshot(
        source_root=source,
        version_root=version_root,
        subtree=subtree,
        manifest_sha256=manifest_sha256,
        files=files,
    )
    installed = _local_binding_from_validated_root(
        root=resource_root,
        subtree=subtree,
        manifest_sha256=manifest_sha256,
        files=files,
        allow_missing=False,
    )
    assert installed is not None
    return installed


def verify_pinned_dataset_resources(
    binding: DatasetResourceBinding,
) -> DatasetResourceBinding:
    """Revalidate an existing binding without network access or ref resolution."""
    if not isinstance(binding, DatasetResourceBinding):
        raise TypeError("binding must be a DatasetResourceBinding")
    # Apply the same exact-shape and local-source invariants used for a binding
    # loaded from disk.  Direct dataclass construction must not bypass them.
    binding = DatasetResourceBinding.from_dict(binding.as_dict())
    verified = _binding_from_validated_root(
        root=binding.root,
        dataset_repo=binding.dataset_repo,
        repo_type=binding.repo_type,
        requested_revision=binding.requested_revision,
        resolved_revision=binding.resolved_revision,
        subtree=binding.subtree,
        offline=binding.offline,
    )
    if verified.manifest_sha256 != binding.manifest_sha256:
        raise ResourceManifestError(
            "dataset resource manifest changed after preparation"
        )
    if verified.files != binding.files:
        raise ResourceManifestError("dataset resource files changed after preparation")
    return verified


def load_dataset_resource_binding(path: Path) -> DatasetResourceBinding:
    """Load one JSON binding file; no resource file is trusted or read yet."""
    source = Path(path)
    if source.is_symlink():
        raise ResourceManifestError(
            f"dataset resource binding source must not be a symlink: {source}"
        )
    try:
        payload = _load_json_strict(source)
    except OSError as error:
        raise ResourceManifestError(
            f"could not read dataset resource binding {source}: {error}"
        ) from error
    return DatasetResourceBinding.from_dict(payload)


def write_dataset_resource_binding(
    binding: DatasetResourceBinding,
    path: Path,
) -> Path:
    """Atomically write a JSON-safe binding for a child process or run manifest."""
    verify_pinned_dataset_resources(binding)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(binding.as_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _resolve_dataset_revision(
    *,
    dataset_repo: str,
    repo_type: str,
    requested_revision: str,
) -> str:
    try:
        from huggingface_hub import HfApi

        info = HfApi().repo_info(
            repo_id=dataset_repo,
            repo_type=repo_type,
            revision=requested_revision,
        )
    except Exception as error:
        raise ResourcePreparationError(
            f"could not resolve {repo_type} {dataset_repo!r} revision "
            f"{requested_revision!r}: {error}"
        ) from error
    resolved = getattr(info, "sha", None)
    if not isinstance(resolved, str) or not _FULL_REVISION_RE.fullmatch(resolved):
        raise ResourcePreparationError(
            f"resolved revision is not a full 40-hex commit: {resolved!r}"
        )
    return resolved.lower()


def _download_versioned_snapshot(
    *,
    version_root: Path,
    subtree: str,
    dataset_repo: str,
    repo_type: str,
    resolved_revision: str,
    requested_revision: str,
) -> None:
    version_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_revision}.",
            dir=str(version_root.parent),
        )
    )
    try:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=dataset_repo,
                repo_type=repo_type,
                revision=resolved_revision,
                local_dir=str(temporary),
                allow_patterns=[f"{subtree}/**"],
            )
        except Exception as error:
            raise ResourcePreparationError(
                f"could not materialize {repo_type} {dataset_repo!r} revision "
                f"{resolved_revision} (requested {requested_revision!r}): {error}"
            ) from error

        _binding_from_validated_root(
            root=temporary / subtree,
            dataset_repo=dataset_repo,
            repo_type=repo_type,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            subtree=subtree,
            offline=False,
        )
        try:
            temporary.rename(version_root)
        except OSError as error:
            # A concurrent process may have materialized the same immutable
            # revision. Accept it only after a complete independent validation.
            if not version_root.exists():
                raise ResourcePreparationError(
                    f"could not install pinned resource snapshot at "
                    f"{version_root}: {error}"
                ) from error
            _binding_from_validated_root(
                root=version_root / subtree,
                dataset_repo=dataset_repo,
                repo_type=repo_type,
                requested_revision=requested_revision,
                resolved_revision=resolved_revision,
                subtree=subtree,
                offline=False,
            )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _materialized_version_root(
    cache_root: Path,
    *,
    dataset_repo: str,
    repo_type: str,
    resolved_revision: str,
) -> Path:
    identity = f"{repo_type}:{dataset_repo}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", dataset_repo).strip("-") or "repo"
    return cache_root / f"{slug}-{suffix}" / resolved_revision


def _local_binding_from_validated_root(
    *,
    root: Path,
    subtree: str,
    manifest_sha256: str,
    files: tuple[DatasetResourceFile, ...],
    allow_missing: bool,
) -> DatasetResourceBinding | None:
    if not root.exists():
        if allow_missing:
            return None
        raise ResourcePreparationError(f"local resource snapshot is missing: {root}")
    if not root.is_dir() or root.is_symlink():
        raise ResourcePreparationError(
            f"local resource snapshot is not a safe directory: {root}"
        )
    actual_manifest_sha256, actual_files = _validate_resource_manifest(
        root,
        subtree=subtree,
    )
    if (
        actual_manifest_sha256 != manifest_sha256
        or actual_files != files
        or root.parent.name != manifest_sha256
        or root.parent.parent.name != LOCAL_RESOURCE_ID
    ):
        raise ResourceManifestError(
            "local resource snapshot differs from its content-addressed binding"
        )
    return DatasetResourceBinding(
        dataset_repo=LOCAL_RESOURCE_ID,
        repo_type=LOCAL_RESOURCE_ID,
        requested_revision=manifest_sha256,
        resolved_revision=manifest_sha256[:40],
        subtree=subtree,
        manifest_sha256=manifest_sha256,
        files=files,
        offline=True,
        root=root.resolve(),
    )


def _install_local_snapshot(
    *,
    source_root: Path,
    version_root: Path,
    subtree: str,
    manifest_sha256: str,
    files: tuple[DatasetResourceFile, ...],
) -> None:
    version_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{manifest_sha256}.",
            dir=str(version_root.parent),
        )
    )
    temporary_resource_root = temporary / subtree
    try:
        temporary_resource_root.mkdir()
        _copy_regular_file_no_follow(
            source_root / PINNED_RESOURCE_MANIFEST,
            temporary_resource_root / PINNED_RESOURCE_MANIFEST,
        )
        for item in files:
            source = source_root / PurePosixPath(item.path)
            destination = temporary_resource_root / PurePosixPath(item.path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _copy_regular_file_no_follow(source, destination)

        copied_manifest_sha256, copied_files = _validate_resource_manifest(
            temporary_resource_root,
            subtree=subtree,
        )
        source_manifest_sha256, source_files = _validate_resource_manifest(
            source_root,
            subtree=subtree,
        )
        if (
            copied_manifest_sha256 != manifest_sha256
            or copied_files != files
            or source_manifest_sha256 != manifest_sha256
            or source_files != files
        ):
            raise ResourceManifestError(
                "local resource source changed while its snapshot was copied"
            )
        _fsync_tree(temporary_resource_root)
        try:
            temporary.rename(version_root)
        except OSError as error:
            existing = _local_binding_from_validated_root(
                root=version_root / subtree,
                subtree=subtree,
                manifest_sha256=manifest_sha256,
                files=files,
                allow_missing=True,
            )
            if existing is None:
                raise ResourcePreparationError(
                    f"could not install local resource snapshot at "
                    f"{version_root}: {error}"
                ) from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _copy_regular_file_no_follow(source: Path, destination: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as error:
        raise ResourceManifestError(
            f"could not safely open local resource file {source}: {error}"
        ) from error
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ResourceManifestError(
                f"local resource source is not a regular file: {source}"
            )
        with os.fdopen(source_fd, "rb", closefd=False) as input_stream:
            with destination.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        after = os.fstat(source_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ResourceManifestError(
                f"local resource file changed while copied: {source}"
            )
    finally:
        os.close(source_fd)


def _fsync_tree(root: Path) -> None:
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    ):
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _binding_from_validated_root(
    *,
    root: Path,
    dataset_repo: str,
    repo_type: str,
    requested_revision: str,
    resolved_revision: str,
    subtree: str,
    offline: bool,
) -> DatasetResourceBinding:
    manifest_sha256, files = _validate_resource_manifest(root, subtree=subtree)
    return DatasetResourceBinding(
        dataset_repo=dataset_repo,
        repo_type=repo_type,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        subtree=subtree,
        manifest_sha256=manifest_sha256,
        files=files,
        offline=offline,
        root=root,
    )


def _validate_resource_manifest(
    root: Path,
    *,
    subtree: str,
) -> tuple[str, tuple[DatasetResourceFile, ...]]:
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ResourceManifestError(
            f"dataset resource subtree is not a safe directory: {root}"
        )
    manifest_path = root / PINNED_RESOURCE_MANIFEST
    if (
        not manifest_path.exists()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise ResourceManifestError(
            f"dataset resource manifest is missing or unsafe: {manifest_path}"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ResourceManifestError(
            f"could not read dataset resource manifest {manifest_path}: {error}"
        ) from error
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        payload = _json_loads_strict(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ResourceManifestError(
            f"invalid dataset resource manifest JSON: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "subtree",
        "files",
    }:
        raise ResourceManifestError(
            "dataset resource manifest must contain exactly "
            "schema_version, subtree, and files"
        )
    if payload["schema_version"] != PINNED_RESOURCE_SCHEMA_VERSION:
        raise ResourceManifestError(
            "unsupported dataset resource manifest schema_version"
        )
    if payload["subtree"] != subtree:
        raise ResourceManifestError(
            f"dataset resource manifest subtree mismatch: "
            f"{payload['subtree']!r} != {subtree!r}"
        )
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ResourceManifestError(
            "dataset resource manifest files must be a non-empty list"
        )
    files = tuple(DatasetResourceFile.from_dict(item) for item in raw_files)
    paths = tuple(item.path for item in files)
    if paths != tuple(sorted(paths)):
        raise ResourceManifestError(
            "dataset resource manifest files must be sorted by path"
        )
    if len(set(paths)) != len(paths):
        raise ResourceManifestError(
            "dataset resource manifest contains duplicate paths"
        )

    actual_paths: set[str] = set()
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ResourceManifestError(
                f"dataset resource subtree contains a symlink: {candidate}"
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ResourceManifestError(
                f"dataset resource subtree contains a special file: {candidate}"
            )
        relative = candidate.relative_to(root).as_posix()
        if relative != PINNED_RESOURCE_MANIFEST:
            actual_paths.add(relative)
    if actual_paths != set(paths):
        missing = sorted(set(paths) - actual_paths)
        unexpected = sorted(actual_paths - set(paths))
        raise ResourceManifestError(
            "dataset resource manifest closed-set mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for item in files:
        candidate = root / PurePosixPath(item.path)
        stat = candidate.stat()
        if stat.st_size != item.size_bytes:
            raise ResourceManifestError(
                f"dataset resource size mismatch for {item.path!r}: "
                f"{stat.st_size} != {item.size_bytes}"
            )
        digest = _sha256_file(candidate)
        if digest != item.sha256:
            raise ResourceManifestError(
                f"dataset resource sha256 mismatch for {item.path!r}: "
                f"{digest} != {item.sha256}"
            )
    return manifest_sha256, files


def _validate_subtree(subtree: str) -> None:
    if not isinstance(subtree, str) or not subtree:
        raise ResourcePreparationError("subtree must be a non-empty string")
    _validate_relative_path(subtree)
    if "/" in subtree:
        raise ResourcePreparationError("subtree must be one top-level directory")


def _validate_relative_path(path: str) -> None:
    if not path or "\\" in path:
        raise ResourceManifestError(f"unsafe resource relative path: {path!r}")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ResourceManifestError(f"unsafe resource relative path: {path!r}")
    if pure.as_posix() == PINNED_RESOURCE_MANIFEST:
        raise ResourceManifestError(
            f"{PINNED_RESOURCE_MANIFEST!r} must not list itself"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_loads_strict(text: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=reject_duplicates)


def _load_json_strict(path: Path) -> Any:
    try:
        return _json_loads_strict(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ResourceManifestError(
            f"dataset resource binding is not valid UTF-8: {path}"
        ) from error
    except ValueError as error:
        raise ResourceManifestError(
            f"invalid dataset resource binding JSON {path}: {error}"
        ) from error
