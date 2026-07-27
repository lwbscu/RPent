"""Disjoint writable runtime state for concurrent paired Eval arms."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import robots.behavior.runtime as behavior_runtime
from robots.behavior.runtime import (
    prepare_campaign_runtime_isolation,
    validate_campaign_runtime_isolation,
)


def _fake_isaac_root(tmp_path: Path) -> Path:
    root = tmp_path / "isaac-source"
    (root / "apps").mkdir(parents=True)
    (root / "apps" / "isaac-sim.exp.full.kit").write_text(
        "[package]\n",
        encoding="utf-8",
    )
    (root / "VERSION").write_text("fixture\n", encoding="utf-8")
    (root / "exts").mkdir()
    (root / "exts" / "marker.txt").write_text("shared\n", encoding="utf-8")
    return root


def test_gpu6_gpu7_campaign_runtime_paths_are_fully_disjoint(
    tmp_path: Path,
) -> None:
    isaac = _fake_isaac_root(tmp_path)
    agentic = prepare_campaign_runtime_isolation(
        tmp_path / "agentic",
        "picking_up_trash_agentic",
        "7",
        isaac_root=isaac,
    )
    baseline = prepare_campaign_runtime_isolation(
        tmp_path / "baseline",
        "picking_up_trash_pure_vla",
        "6",
        isaac_root=isaac,
    )

    agentic_env = agentic.environment()
    baseline_env = baseline.environment()
    assert agentic_env["CUDA_VISIBLE_DEVICES"] == "7"
    assert baseline_env["CUDA_VISIBLE_DEVICES"] == "6"
    for key in (
        "OMNIGIBSON_APPDATA_PATH",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "OV_CACHE_DIR",
        "OMNI_USER_FOLDER",
        "ISAAC_PATH",
        "EXP_PATH",
        "TMPDIR",
        "RPENT_BEHAVIOR_ENDPOINT_DIR",
        "RPENT_BEHAVIOR_LOG_DIR",
    ):
        assert agentic_env[key] != baseline_env[key]
        assert Path(agentic_env[key]).is_relative_to(agentic.root)
        assert Path(baseline_env[key]).is_relative_to(baseline.root)
    assert not any("LOCK" in key for key in agentic_env)
    assert not any("LOCK" in key for key in baseline_env)
    assert not (Path(agentic_env["EXP_PATH"])).is_symlink()
    assert not (Path(baseline_env["EXP_PATH"])).is_symlink()
    assert (Path(agentic_env["ISAAC_PATH"]) / "exts").is_symlink()
    assert (Path(baseline_env["ISAAC_PATH"]) / "exts").is_symlink()


def test_runtime_isolation_binding_reuses_only_exact_identity(
    tmp_path: Path,
) -> None:
    isaac = _fake_isaac_root(tmp_path)
    root = tmp_path / "runtime"
    first = prepare_campaign_runtime_isolation(
        root,
        "paired_agentic",
        "7",
        isaac_root=isaac,
    )

    second = prepare_campaign_runtime_isolation(
        root,
        "paired_agentic",
        "7",
        isaac_root=isaac,
    )
    assert second == first
    assert validate_campaign_runtime_isolation(root, first.binding_sha256) == first

    with pytest.raises(RuntimeError, match="different identity"):
        prepare_campaign_runtime_isolation(
            root,
            "paired_baseline",
            "6",
            isaac_root=isaac,
        )


def test_runtime_isolation_preserves_behavior_python_symlink_for_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isaac = _fake_isaac_root(tmp_path)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(Path(sys.executable))
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{isaac}\n",
            stderr="",
        )

    monkeypatch.setattr(behavior_runtime.subprocess, "run", fake_run)

    prepare_campaign_runtime_isolation(
        tmp_path / "runtime",
        "paired_agentic",
        "7",
        behavior_python=venv_python,
    )

    assert commands
    assert commands[0][0] == str(venv_python.absolute())
    assert commands[0][0] != str(venv_python.resolve())


@pytest.mark.parametrize("kind", ["missing", "directory", "not_executable"])
def test_runtime_isolation_rejects_invalid_behavior_python(
    tmp_path: Path,
    kind: str,
) -> None:
    behavior_python = tmp_path / "venv" / "bin" / "python"
    if kind == "directory":
        behavior_python.mkdir(parents=True)
    elif kind == "not_executable":
        behavior_python.parent.mkdir(parents=True)
        behavior_python.write_text("#!/bin/sh\n", encoding="utf-8")
        behavior_python.chmod(0o644)

    with pytest.raises(ValueError, match="behavior_python must"):
        prepare_campaign_runtime_isolation(
            tmp_path / "runtime",
            "paired_agentic",
            "7",
            behavior_python=behavior_python,
        )

    assert not (tmp_path / "runtime").exists()


def test_runtime_isolation_reuse_still_validates_supplied_behavior_python(
    tmp_path: Path,
) -> None:
    isaac = _fake_isaac_root(tmp_path)
    root = tmp_path / "runtime"
    prepare_campaign_runtime_isolation(
        root,
        "paired_agentic",
        "7",
        isaac_root=isaac,
    )

    with pytest.raises(ValueError, match="behavior_python must"):
        prepare_campaign_runtime_isolation(
            root,
            "paired_agentic",
            "7",
            behavior_python=tmp_path / "missing-python",
        )


def test_runtime_isolation_rejects_binding_or_path_tampering(
    tmp_path: Path,
) -> None:
    isaac = _fake_isaac_root(tmp_path)
    root = tmp_path / "runtime"
    binding = prepare_campaign_runtime_isolation(
        root,
        "paired_agentic",
        "7",
        isaac_root=isaac,
    )
    marker = root / "runtime_isolation.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["paths"]["tmp"] = str(tmp_path / "outside")
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="binding SHA-256 mismatch|path escapes",
    ):
        validate_campaign_runtime_isolation(root, binding.binding_sha256)
