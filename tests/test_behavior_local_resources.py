"""Strict local frozen-resource contracts for BEHAVIOR."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from behavior_resource_fixtures import FIXTURE_RESOURCES

import robots.behavior.dataset_resources as resource_module
from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    ResourceManifestError,
    ResourcePreparationError,
    load_dataset_resource_binding,
    prepare_local_dataset_resources,
    verify_pinned_dataset_resources,
    write_dataset_resource_binding,
)
from robots.behavior.resources import prepare_behavior_resources


def _local_source(tmp_path: Path) -> Path:
    source = tmp_path / "source" / "behavior"
    shutil.copytree(FIXTURE_RESOURCES, source)
    return source


def test_local_source_is_copied_to_a_full_hash_content_addressed_snapshot(
    tmp_path: Path,
) -> None:
    source = _local_source(tmp_path)
    cache = tmp_path / "cache"

    binding = prepare_local_dataset_resources(
        "behavior",
        source_root=source,
        cache_root=cache,
    )

    assert binding.dataset_repo == "local"
    assert binding.repo_type == "local"
    assert binding.offline is True
    assert binding.requested_revision == binding.manifest_sha256
    assert binding.resolved_revision == binding.manifest_sha256[:40]
    assert binding.root == cache / "local" / binding.manifest_sha256 / "behavior"
    assert binding.root != source
    assert verify_pinned_dataset_resources(binding) == binding


def test_local_snapshot_is_independent_but_snapshot_drift_fails_closed(
    tmp_path: Path,
) -> None:
    source = _local_source(tmp_path)
    binding = prepare_local_dataset_resources(
        "behavior",
        source_root=source,
        cache_root=tmp_path / "cache",
    )
    source_target = source / "memory" / "picking_up_trash" / "target_prior.md"
    source_target.write_text("# source changed after freeze\n", encoding="utf-8")

    assert verify_pinned_dataset_resources(binding) == binding

    snapshot_target = binding.root / "memory" / "picking_up_trash" / "target_prior.md"
    snapshot_target.write_text("# snapshot changed\n", encoding="utf-8")
    with pytest.raises(
        ResourceManifestError,
        match="size mismatch|sha256 mismatch",
    ):
        verify_pinned_dataset_resources(binding)


def test_local_binding_round_trip_preserves_local_identity(
    tmp_path: Path,
) -> None:
    binding = prepare_local_dataset_resources(
        "behavior",
        source_root=_local_source(tmp_path),
        cache_root=tmp_path / "cache",
    )

    path = write_dataset_resource_binding(binding, tmp_path / "source.json")
    loaded = load_dataset_resource_binding(path)

    assert loaded == binding
    assert verify_pinned_dataset_resources(loaded) == binding


def test_local_binding_cannot_impersonate_a_dataset_or_use_a_short_cache_key(
    tmp_path: Path,
) -> None:
    binding = prepare_local_dataset_resources(
        "behavior",
        source_root=_local_source(tmp_path),
        cache_root=tmp_path / "cache",
    )
    payload = binding.as_dict()
    payload["dataset_repo"] = "RLinf/RPent-memory"
    with pytest.raises(ResourceManifestError, match="local resource binding"):
        DatasetResourceBinding.from_dict(payload)

    payload = binding.as_dict()
    payload["root"] = str(
        binding.root.parent.parent / binding.resolved_revision / "behavior"
    )
    with pytest.raises(ResourceManifestError, match="local resource binding"):
        DatasetResourceBinding.from_dict(payload)


def test_local_prepare_detects_source_change_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _local_source(tmp_path)
    original = resource_module._copy_regular_file_no_follow
    copied = 0

    def mutate_after_first_copy(source_path: Path, destination: Path) -> None:
        nonlocal copied
        original(source_path, destination)
        copied += 1
        if copied == 1:
            target = source / "memory" / "picking_up_trash" / "target_prior.md"
            target.write_text("# changed during copy\n", encoding="utf-8")

    monkeypatch.setattr(
        resource_module,
        "_copy_regular_file_no_follow",
        mutate_after_first_copy,
    )

    with pytest.raises(ResourceManifestError):
        prepare_local_dataset_resources(
            "behavior",
            source_root=source,
            cache_root=tmp_path / "cache",
        )


def test_concurrent_local_prepare_reuses_one_complete_snapshot(
    tmp_path: Path,
) -> None:
    source = _local_source(tmp_path)
    cache = tmp_path / "cache"

    def prepare() -> DatasetResourceBinding:
        return prepare_local_dataset_resources(
            "behavior",
            source_root=source,
            cache_root=cache,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = tuple(executor.map(lambda _index: prepare(), range(2)))

    assert first == second
    assert verify_pinned_dataset_resources(first) == first
    assert not tuple((cache / "local").glob(".*"))


def test_behavior_resource_hook_selects_local_and_child_reuses_snapshot(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        behavior_resource_root=None,
        behavior_resource_source_file=None,
        behavior_resource_cache=str(tmp_path / "cache"),
        behavior_resource_offline=None,
        behavior_resource_revision=None,
        behavior_resource_local=str(_local_source(tmp_path)),
    )
    parent = prepare_behavior_resources(args)
    source_file = write_dataset_resource_binding(
        parent,
        tmp_path / "resource_source.json",
    )
    child_args = argparse.Namespace(
        behavior_resource_root=str(parent.root),
        behavior_resource_source_file=str(source_file),
        behavior_resource_cache=None,
        behavior_resource_offline=None,
        behavior_resource_revision=None,
        behavior_resource_local=None,
    )

    child = prepare_behavior_resources(child_args)

    assert child == parent
    assert child_args._behavior_resource_source == parent.as_dict()
    assert json.loads(source_file.read_text(encoding="utf-8")) == parent.as_dict()


def test_behavior_resource_hook_rejects_local_with_hf_revision(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        behavior_resource_root=None,
        behavior_resource_source_file=None,
        behavior_resource_cache=str(tmp_path / "cache"),
        behavior_resource_offline=None,
        behavior_resource_revision="main",
        behavior_resource_local=str(_local_source(tmp_path)),
    )

    with pytest.raises(ResourcePreparationError, match="mutually exclusive"):
        prepare_behavior_resources(args)
