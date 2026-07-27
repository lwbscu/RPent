"""Strict external-resource contracts for BEHAVIOR."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from behavior_resource_fixtures import (
    FIXTURE_RESOURCES,
    FIXTURE_REVISION,
    fixture_resource_binding,
)

from robots.behavior import cli as cli_main
from robots.behavior.dataset_resources import (
    DatasetResourceBinding,
    ResourceManifestError,
    ResourcePreparationError,
    load_dataset_resource_binding,
    prepare_pinned_dataset_resources,
    verify_pinned_dataset_resources,
    write_dataset_resource_binding,
)
from robots.behavior.resources import prepare_behavior_resources
from robots.behavior.spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


def _install_fake_huggingface(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolved_revision: str = FIXTURE_REVISION,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeApi:
        def repo_info(self, **kwargs):
            calls.append({"operation": "resolve", **kwargs})
            return SimpleNamespace(sha=resolved_revision)

    def snapshot_download(**kwargs):
        calls.append({"operation": "download", **kwargs})
        destination = Path(str(kwargs["local_dir"])) / "behavior"
        shutil.copytree(FIXTURE_RESOURCES, destination)
        return str(destination.parent)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=snapshot_download),
    )
    return calls


def test_online_sync_resolves_once_then_downloads_the_immutable_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_huggingface(monkeypatch)

    binding = prepare_pinned_dataset_resources(
        subtree="behavior",
        requested_revision="main",
        dataset_repo="fixture/RPent-memory",
        cache_root=tmp_path,
        offline=False,
    )

    assert isinstance(binding, DatasetResourceBinding)
    assert binding.requested_revision == "main"
    assert binding.resolved_revision == FIXTURE_REVISION
    assert binding.root.name == "behavior"
    assert binding.root != FIXTURE_RESOURCES
    assert binding.files
    assert binding.as_dict()["files"] == [item.as_dict() for item in binding.files]
    assert calls[0] == {
        "operation": "resolve",
        "repo_id": "fixture/RPent-memory",
        "repo_type": "dataset",
        "revision": "main",
    }
    assert calls[1]["operation"] == "download"
    assert calls[1]["repo_id"] == "fixture/RPent-memory"
    assert calls[1]["repo_type"] == "dataset"
    assert calls[1]["revision"] == FIXTURE_REVISION
    assert calls[1]["allow_patterns"] == ["behavior/**"]
    temporary = Path(str(calls[1]["local_dir"]))
    assert temporary.parent == binding.root.parent.parent
    assert temporary.name.startswith(f".{FIXTURE_REVISION}.")


def test_offline_mode_requires_a_full_revision_and_reuses_only_that_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_huggingface(monkeypatch)
    online = prepare_pinned_dataset_resources(
        subtree="behavior",
        requested_revision=FIXTURE_REVISION,
        dataset_repo="fixture/RPent-memory",
        cache_root=tmp_path,
        offline=False,
    )

    monkeypatch.delitem(sys.modules, "huggingface_hub")
    offline = prepare_pinned_dataset_resources(
        subtree="behavior",
        requested_revision=FIXTURE_REVISION,
        dataset_repo="fixture/RPent-memory",
        cache_root=tmp_path,
        offline=True,
    )

    assert offline.root == online.root
    assert offline.files == online.files
    assert offline.offline is True
    with pytest.raises(ResourcePreparationError, match="full 40-hex"):
        prepare_pinned_dataset_resources(
            subtree="behavior",
            requested_revision="main",
            dataset_repo="fixture/RPent-memory",
            cache_root=tmp_path,
            offline=True,
        )
    with pytest.raises(ResourcePreparationError, match="not cached"):
        prepare_pinned_dataset_resources(
            subtree="behavior",
            requested_revision="b" * 40,
            dataset_repo="fixture/RPent-memory",
            cache_root=tmp_path,
            offline=True,
        )


@pytest.mark.parametrize("mutation", ("content", "unexpected"))
def test_closed_hash_binding_rejects_any_cached_resource_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _install_fake_huggingface(monkeypatch)
    binding = prepare_pinned_dataset_resources(
        subtree="behavior",
        requested_revision=FIXTURE_REVISION,
        dataset_repo="fixture/RPent-memory",
        cache_root=tmp_path,
        offline=False,
    )
    if mutation == "content":
        target = binding.root / "memory" / "picking_up_trash" / "target_prior.md"
        target.write_text("# changed\n", encoding="utf-8")
    else:
        (binding.root / "undeclared.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(
        ResourceManifestError,
        match="size mismatch|sha256 mismatch|closed-set mismatch",
    ):
        verify_pinned_dataset_resources(binding)


def test_binding_file_round_trip_remains_hash_verified(
    tmp_path: Path,
) -> None:
    binding = fixture_resource_binding()
    path = write_dataset_resource_binding(binding, tmp_path / "source.json")

    loaded = load_dataset_resource_binding(path)

    assert loaded == binding
    assert verify_pinned_dataset_resources(loaded) == binding


def test_behavior_resource_hook_installs_verified_source_and_child_reuses_it(
    tmp_path: Path,
) -> None:
    binding = fixture_resource_binding()
    source_path = write_dataset_resource_binding(binding, tmp_path / "source.json")
    args = argparse.Namespace(
        behavior_resource_root=str(binding.root),
        behavior_resource_source_file=str(source_path),
        behavior_resource_cache=None,
        behavior_resource_offline=False,
        behavior_resource_revision=FIXTURE_REVISION,
    )

    prepared = prepare_behavior_resources(args)

    assert prepared == binding
    assert args.prepared_resources == binding
    assert args._behavior_resource_root == FIXTURE_RESOURCES.resolve()
    assert args._behavior_resource_source == binding.as_dict()


def test_behavior_resource_hook_rejects_unpaired_child_source(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        behavior_resource_root=str(FIXTURE_RESOURCES),
        behavior_resource_source_file=None,
        behavior_resource_cache=str(tmp_path),
        behavior_resource_offline=True,
        behavior_resource_revision=FIXTURE_REVISION,
    )

    with pytest.raises(ResourcePreparationError, match="must be provided together"):
        prepare_behavior_resources(args)


def test_shared_cli_prepares_resources_before_parse_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def prepare(_args):
        events.append("prepare")
        return object()

    def parse(args):
        events.append("parse")
        assert args.prepared_resources is not None
        raise ValueError("stop after ordering check")

    spec = EnvSpec(
        name="behavior",
        prompts=PromptBundle(system=lambda: None, user=lambda: None),  # type: ignore[arg-type]
        add_cli_args=lambda _parser, use_dashboard: None,
        prepare_resources=prepare,
        parse_config=parse,
        init_runtime=lambda _args, _output: ([], {}),
    )
    monkeypatch.setattr(cli_main, "get_env_spec", lambda: spec)
    monkeypatch.setattr(sys, "argv", ["rpent", "--env", "behavior"])

    with pytest.raises(SystemExit):
        cli_main.main()

    assert events == ["prepare", "parse"]
