"""Hash-sealed source identity for paired BEHAVIOR Eval campaigns."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.behavior.run_manifest import RunManifest
from robots.behavior.serial_vla_eval import load_source_snapshot
from robots.behavior.source_snapshot import (
    SOURCE_SNAPSHOT_FILENAME,
    SourceSnapshotError,
    create_source_snapshot,
    validate_source_snapshot,
)


def _git(source: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )


def _dirty_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.email", "tests@example.invalid")
    _git(source, "config", "user.name", "RPent tests")
    (source / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "runner.py")
    _git(source, "commit", "-qm", "fixture")
    (source / "runner.py").write_text("VALUE = 2\n", encoding="utf-8")
    (source / "untracked.py").write_text("EXTRA = True\n", encoding="utf-8")
    return source


def test_dirty_source_snapshot_is_repeatable_and_shared_by_both_eval_arms(
    tmp_path: Path,
) -> None:
    source = _dirty_source(tmp_path)

    first = create_source_snapshot(source, tmp_path / "agentic-source")
    second = create_source_snapshot(source, tmp_path / "baseline-source")

    assert first.payload["base_git"]["worktree_dirty"] is True
    assert first.binding_sha256 == second.binding_sha256
    assert first.tree_sha256 == second.tree_sha256
    assert (
        first.payload["base_git"]["status_sha256"]
        == second.payload["base_git"]["status_sha256"]
    )
    assert first.payload["files"] == second.payload["files"]
    assert (tmp_path / "agentic-source" / "runner.py").read_text() == "VALUE = 2\n"
    assert (tmp_path / "agentic-source" / SOURCE_SNAPSHOT_FILENAME).is_file()


def test_source_snapshot_validation_rejects_content_or_closed_set_drift(
    tmp_path: Path,
) -> None:
    source = _dirty_source(tmp_path)
    snapshot = tmp_path / "snapshot"
    binding = create_source_snapshot(source, snapshot)

    runner = snapshot / "runner.py"
    runner.chmod(runner.stat().st_mode | 0o200)
    runner.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(SourceSnapshotError, match="file metadata changed"):
        validate_source_snapshot(snapshot, binding.binding_sha256)

    snapshot = tmp_path / "closed-set-snapshot"
    binding = create_source_snapshot(source, snapshot)
    snapshot.chmod(snapshot.stat().st_mode | 0o200)
    (snapshot / "unexpected.py").write_text("BAD = True\n", encoding="utf-8")
    with pytest.raises(SourceSnapshotError, match="unbound files"):
        validate_source_snapshot(snapshot, binding.binding_sha256)


def test_source_snapshot_validation_rejects_wrong_binding_and_writable_files(
    tmp_path: Path,
) -> None:
    source = _dirty_source(tmp_path)
    snapshot = tmp_path / "snapshot"
    binding = create_source_snapshot(source, snapshot)

    with pytest.raises(SourceSnapshotError, match="binding SHA-256 mismatch"):
        validate_source_snapshot(snapshot, "0" * 64)

    runner = snapshot / "runner.py"
    runner.chmod(runner.stat().st_mode | 0o200)
    with pytest.raises(SourceSnapshotError, match="metadata changed|writable"):
        validate_source_snapshot(snapshot, binding.binding_sha256)


def test_source_snapshot_does_not_modify_source_worktree(
    tmp_path: Path,
) -> None:
    source = _dirty_source(tmp_path)
    before = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=normal"],
        cwd=source,
        check=True,
        capture_output=True,
    ).stdout

    binding = create_source_snapshot(source, tmp_path / "snapshot")

    after = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=normal"],
        cwd=source,
        check=True,
        capture_output=True,
    ).stdout
    assert after == before
    assert binding.payload["base_git"]["status_sha256"]
    assert not os.access(tmp_path / "snapshot" / "runner.py", os.W_OK)


def test_pure_vla_loader_revalidates_canonical_snapshot_binding(
    tmp_path: Path,
) -> None:
    source = _dirty_source(tmp_path)
    binding = create_source_snapshot(source, tmp_path / "snapshot")

    loaded = load_source_snapshot(
        binding.snapshot_root,
        expected_binding_sha256=binding.binding_sha256,
    )

    assert loaded.binding_sha256 == binding.binding_sha256
    assert loaded.tree_sha256 == binding.tree_sha256
    assert loaded.binding == binding.as_dict()


def test_run_manifest_start_persists_exact_source_snapshot_binding(
    tmp_path: Path,
) -> None:
    source = _dirty_source(tmp_path)
    binding = create_source_snapshot(source, tmp_path / "snapshot")
    output = tmp_path / "run"
    output.mkdir()
    instance_dir = tmp_path / "instances"
    instance_dir.mkdir()
    args = SimpleNamespace(
        no_driver=True,
        env_endpoint="127.0.0.1",
        env_port=5000,
        behavior_phase="eval",
        behavior_candidate_instance_id=None,
        behavior_attempt_index=1,
        behavior_job_id=None,
        task_name="picking_up_trash",
        activity_definition_id=0,
        activity_instance_id=108,
        public_seed=10,
        _behavior_mapping_version="picking_up_trash_public_seed_v1",
        _behavior_task_spec_binding={"task_name": "picking_up_trash"},
        _behavior_prompt_binding={"sha256": "a" * 64},
        _behavior_instance_classification="public_eval",
        suite="behavior_2025_challenge",
        task=1,
        max_episode_steps=24756,
        activity_instance_dir=str(instance_dir),
        scene_model="house_double_floor_lower",
        seed=0,
        max_tool_calls=350,
        max_wall_clock_s=3300,
        planner="codex",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        dashboard=False,
        dashboard_auto_start=False,
        _behavior_frozen_inputs={"manifest_sha256": "b" * 64},
        _behavior_repo_memory_input=None,
        _behavior_recipe_catalog_input=None,
        _behavior_resource_source={"repo_type": "local"},
        _behavior_policy_checkpoint_binding={
            "resolved_path": "/checkpoint",
            "binding_sha256": "c" * 64,
        },
        _behavior_source_snapshot_binding=binding,
        cuda_device="7",
        vla_endpoint="http://127.0.0.1:6000",
    )

    manifest = RunManifest.start(output, args, repo_root=source)
    persisted = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    assert manifest.data["source_snapshot"] == binding.as_dict()
    assert persisted["source_snapshot"] == binding.as_dict()
    assert (
        json.loads((output / "session_manifest.json").read_text(encoding="utf-8"))[
            "source_snapshot"
        ]
        == binding.as_dict()
    )
