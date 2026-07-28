"""Paired GPU7 agentic and GPU6 pure-VLA Eval supervisor contracts.

These tests exercise only pure builders and mocked lifecycle helpers. They do
not start OmniGibson, VLA servers, robot motion, or HTTP Dashboard servers.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.behavior import paired_eval
from robots.behavior.paired_eval import (
    ArmSpec,
    LiveArm,
    ValidatedRuntimeBase,
    build_agentic_runner_argv,
    build_baseline_runner_argv,
    validate_dashboard_endpoints,
    validate_deadlines,
)
from robots.behavior.source_snapshot import SourceSnapshotBinding


def _value_after(argv: tuple[str, ...], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _runner_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "python": tmp_path / "source" / ".venv" / "bin" / "python",
        "snapshot_root": tmp_path / "source",
        "output_root": tmp_path / "output",
        "public_seed": 10,
        "vla_endpoint": "http://127.0.0.1:9123",
        "behavior_repo": tmp_path / "behavior-repo",
        "behavior_python": tmp_path / "behavior-python",
        "checkpoint": tmp_path / "checkpoint",
        "source_binding_sha256": "a" * 64,
        "action_deadline_s": 6900,
        "cleanup_deadline_s": 7080,
        "instance_timeout_s": 7200,
        "instance_started_monotonic_ns": 1_000_000_000,
        "action_deadline_monotonic_ns": 6_901_000_000_000,
        "cleanup_deadline_monotonic_ns": 7_081_000_000_000,
        "hard_deadline_monotonic_ns": 7_201_000_000_000,
        "runtime_isolation_root": tmp_path / "runtime",
        "runtime_isolation_binding_sha256": "b" * 64,
    }


def _arm(
    tmp_path: Path,
    *,
    name: str,
    gpu: str,
    port: int,
    llm_enabled: bool,
) -> ArmSpec:
    return ArmSpec(
        name=name,
        gpu=gpu,
        dashboard_port=port,
        output_root=tmp_path / name,
        runner_script=tmp_path / f"{name}.py",
        run_id=f"behavior/{name}",
        llm_enabled=llm_enabled,
        controller="gpt-5.5/xhigh" if llm_enabled else "pi0_nav_pick-only",
        allowed_tools=("observe", "pi0_nav_pick") if llm_enabled else ("pi0_nav_pick",),
    )


def _external_runtime_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    paired_eval.OwnedRuntimeRoot,
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    runtime_base = tmp_path / "external-runtime-base"
    output_root = tmp_path / "formal-output"
    source_root = tmp_path / "source"
    runtime_base.mkdir()
    output_root.mkdir()
    source_root.mkdir()
    base_stat = os.stat(runtime_base)
    monkeypatch.setattr(
        paired_eval.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=64 * 1024**3,
            used=1,
            free=63 * 1024**3,
        ),
    )
    owned, _ = paired_eval._create_owned_runtime_root(
        ValidatedRuntimeBase(
            path=runtime_base.resolve(),
            device=base_stat.st_dev,
            inode=base_stat.st_ino,
        ),
        output_root=output_root,
        source_snapshot_root=source_root,
        source_snapshot_binding_sha256="a" * 64,
    )
    arm_root = owned.root / "agentic"
    arm_root.mkdir()
    binding_payload = {
        "schema_version": 1,
        "kind": "behavior_campaign_runtime_isolation",
        "namespace": "paired-eval-agentic",
        "cuda_device": "7",
        "paths": {},
    }
    isolation = SimpleNamespace(
        root=arm_root.resolve(),
        namespace="paired-eval-agentic",
        cuda_device="7",
        binding_sha256="b" * 64,
        as_dict=lambda: {**binding_payload, "binding_sha256": "b" * 64},
    )
    runtime_isolation = {"agentic": isolation}
    owner_document = paired_eval._write_external_runtime_owner(
        owned,
        runtime_isolation,
    )
    monkeypatch.setattr(
        paired_eval,
        "validate_source_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        paired_eval,
        "validate_campaign_runtime_isolation",
        lambda *_args, **_kwargs: isolation,
    )
    return owned, owner_document, runtime_isolation, output_root


def test_vla_log_dir_keeps_runtime_cache_outside_formal_output(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "external-runtime"
    formal_output = tmp_path / "formal-output"
    runtime_root.mkdir()
    formal_output.mkdir()
    path_names = (
        "omnigibson_appdata",
        "xdg_cache",
        "xdg_config",
        "xdg_data",
        "ov_cache",
        "omni_user",
        "tmp",
        "endpoints",
        "logs",
    )
    paths: dict[str, str] = {}
    for name in path_names:
        path = runtime_root / name
        path.mkdir()
        paths[name] = str(path)
    isolation = SimpleNamespace(
        root=runtime_root,
        payload={"paths": paths},
    )

    log_dir = paired_eval._formal_vla_log_dir(
        isolation,
        formal_output_root=formal_output,
    )

    assert log_dir == (formal_output / "launcher_logs" / "vla").resolve()
    assert not any((formal_output / name).exists() for name in path_names)


def test_paired_dashboard_ports_are_distinct_local_and_preserve_8765() -> None:
    validate_dashboard_endpoints(
        host="127.0.0.1",
        agentic_port=8766,
        baseline_port=8767,
    )

    for kwargs, pattern in (
        (
            {
                "host": "0.0.0.0",
                "agentic_port": 8766,
                "baseline_port": 8767,
            },
            "127.0.0.1",
        ),
        (
            {
                "host": "127.0.0.1",
                "agentic_port": 8766,
                "baseline_port": 8766,
            },
            "distinct",
        ),
        (
            {
                "host": "127.0.0.1",
                "agentic_port": 8765,
                "baseline_port": 8767,
            },
            "8765 is protected",
        ),
    ):
        with pytest.raises(ValueError, match=pattern):
            validate_dashboard_endpoints(**kwargs)


def test_paired_deadlines_leave_cleanup_margin_within_two_hours() -> None:
    validate_deadlines(
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
    )

    for values in (
        (0, 7080, 7200),
        (6900, 6900, 7200),
        (6900, 7200, 7200),
        (6900, 7080, 7201),
    ):
        with pytest.raises(ValueError, match="action < cleanup < instance <= 7200"):
            validate_deadlines(
                action_deadline_s=values[0],
                cleanup_deadline_s=values[1],
                instance_timeout_s=values[2],
            )


def test_formal_cli_defaults_to_two_hour_deadline_contract() -> None:
    args = paired_eval._parse_args(
        [
            "--output-root",
            "/formal-output",
            "--runtime-base",
            "/runtime",
            "--source-snapshot-root",
            "/source",
            "--source-snapshot-binding-sha256",
            "a" * 64,
            "--behavior-frozen-publication-root",
            "/publication",
            "--behavior-frozen-provenance-sha256",
            "b" * 64,
            "--behavior-resource-local",
            "/resources",
            "--behavior-repo",
            "/behavior",
            "--behavior-python",
            "/behavior-python",
        ]
    )

    assert (
        args.action_deadline_s,
        args.cleanup_deadline_s,
        args.instance_timeout_s,
    ) == (6900, 7080, 7200)
    assert (args.monitor_interval_s, args.monitor_window_s) == (1200, 7200)


def test_formal_cli_help_documents_two_hour_deadline_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        paired_eval._parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "default: 6900" in help_text
    assert "default: 7080" in help_text
    assert "default: 7200" in help_text


def test_runtime_base_is_required_by_formal_cli(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        paired_eval._parse_args(
            [
                "--output-root",
                "/formal-output",
                "--source-snapshot-root",
                "/source",
                "--source-snapshot-binding-sha256",
                "a" * 64,
                "--behavior-frozen-publication-root",
                "/publication",
                "--behavior-frozen-provenance-sha256",
                "b" * 64,
                "--behavior-resource-local",
                "/resources",
                "--behavior-repo",
                "/behavior",
                "--behavior-python",
                "/behavior-python",
            ]
        )

    assert "--runtime-base" in capsys.readouterr().err


def test_runtime_base_rejects_same_output_filesystem(tmp_path: Path) -> None:
    runtime_base = tmp_path / "runtime-base"
    runtime_base.mkdir()

    with pytest.raises(ValueError, match="different filesystem"):
        paired_eval._validate_external_runtime_base(
            runtime_base,
            output_root=tmp_path / "formal-output",
            protected_paths=(),
            minimum_free_bytes=0,
        )


def test_runtime_base_rejects_symlink_and_protected_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_base = tmp_path / "real-runtime-base"
    real_base.mkdir()
    linked_base = tmp_path / "runtime-link"
    linked_base.symlink_to(real_base, target_is_directory=True)
    nested_source = real_base / "nested-source"
    nested_source.mkdir()

    with pytest.raises(ValueError, match="must not be a symlink"):
        paired_eval._validate_external_runtime_base(
            linked_base,
            output_root=tmp_path / "formal-output",
            protected_paths=(),
            minimum_free_bytes=0,
        )

    original_stat = paired_eval.os.stat
    real_base_resolved = real_base.resolve()

    def cross_device_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        result = original_stat(path, *args, **kwargs)
        if Path(path) == real_base_resolved:
            values = list(result)
            values[2] = result.st_dev + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(paired_eval.os, "stat", cross_device_stat)
    with pytest.raises(ValueError, match="formal inputs or outputs"):
        paired_eval._validate_external_runtime_base(
            real_base,
            output_root=tmp_path / "formal-output",
            protected_paths=(nested_source,),
            minimum_free_bytes=0,
        )


def test_owned_runtime_child_is_unique_private_and_evidenced_in_formal_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned, document, runtime_isolation, output_root = _external_runtime_fixture(
        tmp_path,
        monkeypatch,
    )

    assert owned.root.parent == owned.base
    assert owned.root != owned.base
    assert owned.root not in output_root.parents
    assert output_root not in owned.root.parents
    assert os.stat(owned.root).st_mode & 0o777 == 0o700
    marker = owned.root / paired_eval.EXTERNAL_RUNTIME_OWNER_FILENAME
    evidence = (
        output_root / "runtime_bindings" / paired_eval.EXTERNAL_RUNTIME_OWNER_FILENAME
    )
    assert os.stat(marker).st_mode & 0o777 == 0o600
    assert json.loads(marker.read_text(encoding="utf-8")) == document
    assert json.loads(evidence.read_text(encoding="utf-8")) == document
    assert document["runtime_root"]["device"] == owned.root_device
    assert document["runtime_root"]["inode"] == owned.root_inode
    assert document["output_root"]["path"] == str(output_root.resolve())
    assert document["source_snapshot"]["binding_sha256"] == "a" * 64
    assert document["arms"]["agentic"]["namespace"] == "paired-eval-agentic"
    assert document["arms"]["agentic"]["binding_sha256"] == "b" * 64
    assert set(runtime_isolation) == {"agentic"}


def test_verified_runtime_cleanup_deletes_only_owned_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned, document, runtime_isolation, _ = _external_runtime_fixture(
        tmp_path,
        monkeypatch,
    )
    sibling = owned.base / "unrelated-campaign"
    sibling.mkdir()

    cleanup = paired_eval._delete_owned_runtime_root(
        owned,
        owner_document=document,
        runtime_isolation=runtime_isolation,
        processes_stopped=True,
    )

    assert cleanup["status"] == "deleted"
    assert not owned.root.exists()
    assert owned.base.is_dir()
    assert sibling.is_dir()


def test_runtime_cleanup_retains_root_for_live_process_or_marker_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned, document, runtime_isolation, _ = _external_runtime_fixture(
        tmp_path,
        monkeypatch,
    )

    live_cleanup = paired_eval._delete_owned_runtime_root(
        owned,
        owner_document=document,
        runtime_isolation=runtime_isolation,
        processes_stopped=False,
    )
    assert live_cleanup["status"] == "runtime_cleanup_pending"
    assert live_cleanup["errors"] == ["owned_process_cleanup_unverified"]
    assert owned.root.is_dir()

    marker = owned.root / paired_eval.EXTERNAL_RUNTIME_OWNER_FILENAME
    marker.write_text("{}\n", encoding="utf-8")
    tampered_cleanup = paired_eval._delete_owned_runtime_root(
        owned,
        owner_document=document,
        runtime_isolation=runtime_isolation,
        processes_stopped=True,
    )
    assert tampered_cleanup["status"] == "runtime_cleanup_pending"
    assert any("runtime_owner_marker" in error for error in tampered_cleanup["errors"])
    assert owned.root.is_dir()


def test_active_runtime_capacity_gate_rechecks_external_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned, document, runtime_isolation, _ = _external_runtime_fixture(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        paired_eval.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=20 * 1024**3,
            used=1,
            free=20 * 1024**3 - 1,
        ),
    )

    errors = paired_eval._external_runtime_active_errors(
        owned,
        owner_document=document,
        runtime_isolation=runtime_isolation,
    )

    assert "runtime_base_free_space_below_20_gib" in errors


def test_runtime_cleanup_delete_failure_is_persistent_blocked_forensics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned, document, runtime_isolation, output_root = _external_runtime_fixture(
        tmp_path,
        monkeypatch,
    )
    manifest_path = output_root / "paired_eval_manifest.json"
    paired_eval._atomic_json(
        manifest_path,
        {
            "status": "finishing",
            "blocked_reason": None,
            "external_runtime": {"owner_binding": document},
        },
    )

    original_rmdir = paired_eval.os.rmdir

    def failing_rmdir(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if path == owned.root.name:
            raise OSError("synthetic deletion failure")
        original_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(paired_eval.os, "rmdir", failing_rmdir)
    cleanup = paired_eval._delete_owned_runtime_root(
        owned,
        owner_document=document,
        runtime_isolation=runtime_isolation,
        processes_stopped=True,
    )
    paired_eval._record_external_runtime_cleanup(
        output_root=output_root,
        manifest_path=manifest_path,
        cleanup=cleanup,
    )

    assert cleanup["status"] == "runtime_cleanup_pending"
    assert owned.root.is_dir()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["runtime_cleanup_pending"] is True
    assert "runtime_cleanup_pending" in manifest["blocked_reason"]
    assert (
        json.loads((output_root / "runtime_cleanup.json").read_text(encoding="utf-8"))
        == cleanup
    )


@pytest.mark.parametrize(
    ("blocked_reason", "expected"),
    (
        ("runtime_cleanup_pending", None),
        (
            "agentic_infrastructure_unknown; runtime_cleanup_pending",
            "agentic_infrastructure_unknown",
        ),
        (
            "agentic_infrastructure_unknown; runtime_cleanup_pending: first; second",
            "agentic_infrastructure_unknown",
        ),
    ),
)
def test_completed_runtime_cleanup_removes_only_pending_reason(
    tmp_path: Path,
    blocked_reason: str,
    expected: str | None,
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    manifest_path = output_root / "paired_eval_manifest.json"
    paired_eval._atomic_json(
        manifest_path,
        {
            "status": "blocked",
            "blocked_reason": blocked_reason,
            "runtime_cleanup_pending": True,
        },
    )

    paired_eval._record_external_runtime_cleanup(
        output_root=output_root,
        manifest_path=manifest_path,
        cleanup={"status": "deleted", "errors": []},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_cleanup_pending"] is False
    assert manifest["blocked_reason"] == expected


def test_runner_argv_bind_same_snapshot_but_separate_gpus_and_controllers(
    tmp_path: Path,
) -> None:
    common = _runner_kwargs(tmp_path)
    agentic = build_agentic_runner_argv(
        **common,
        frozen_publication_root=tmp_path / "frozen-publication",
        frozen_provenance_sha256="c" * 64,
        behavior_resource_local=tmp_path / "resources",
        behavior_resource_cache=tmp_path / "resource-cache",
        dashboard_host="127.0.0.1",
        dashboard_port=8766,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        expected_run_nonce="d" * 32,
    )
    baseline = build_baseline_runner_argv(**common)

    assert _value_after(agentic, "--cuda-device") == "7"
    assert _value_after(baseline, "--cuda-device") == "6"
    assert _value_after(agentic, "--public-seed") == "10"
    assert _value_after(baseline, "--public-seed") == "10"
    assert _value_after(agentic, "--source-snapshot-root") == str(
        common["snapshot_root"]
    )
    assert _value_after(baseline, "--source-snapshot-root") == str(
        common["snapshot_root"]
    )
    assert _value_after(agentic, "--source-snapshot-binding-sha256") == "a" * 64
    assert _value_after(baseline, "--source-snapshot-binding-sha256") == "a" * 64
    assert _value_after(agentic, "--model") == "gpt-5.5"
    assert _value_after(agentic, "--reasoning-effort") == "xhigh"
    assert _value_after(agentic, "--expected-run-nonce") == "d" * 32
    assert "--dashboard-event-sink" in agentic
    assert "--external-gpu-lock-owned" in agentic
    assert "--chunks-per-call" in baseline
    assert _value_after(baseline, "--chunks-per-call") == "80"
    assert "--external-gpu-lock-owned" in baseline
    for option in (
        "--instance-started-monotonic-ns",
        "--action-deadline-monotonic-ns",
        "--cleanup-deadline-monotonic-ns",
        "--hard-deadline-monotonic-ns",
    ):
        assert _value_after(agentic, option) == _value_after(baseline, option)
    assert not {
        "--model",
        "--reasoning-effort",
        "--dashboard",
        "--dashboard-event-sink",
        "--behavior-frozen-publication-root",
        "--behavior-resource-local",
    }.intersection(baseline)


def test_pair_deadline_binding_samples_one_origin_for_both_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def monotonic_ns() -> int:
        nonlocal calls
        calls += 1
        return 123_456_789

    monkeypatch.setattr(paired_eval.time, "monotonic_ns", monotonic_ns)
    binding = paired_eval._pair_deadline_binding(
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
    )

    assert calls == 1
    assert binding.started_monotonic_ns == 123_456_789
    assert binding.action_deadline_monotonic_ns == 123_456_789 + 6900 * 10**9
    assert binding.cleanup_deadline_monotonic_ns == 123_456_789 + 7080 * 10**9
    assert binding.hard_deadline_monotonic_ns == 123_456_789 + 7200 * 10**9


def test_pure_action_trace_nonce_binding_is_strict(tmp_path: Path) -> None:
    entry = tmp_path / "entry"
    entry.mkdir()
    trace = entry / "behavior_action_trace.jsonl"
    expected = "e" * 32
    trace.write_text(
        json.dumps(
            {
                "event": "rpent_run_binding",
                "run_nonce": expected,
                "attempt_index": 1,
            }
        )
        + "\n"
        + json.dumps({"step": 1, "info_done": {"success": True}})
        + "\n",
        encoding="utf-8",
    )

    summary = paired_eval._strict_action_trace_summary(
        entry,
        expected_run_nonce=expected,
    )
    assert summary["valid"] is True
    assert summary["official_success_binding"]["run_nonce"] == expected
    assert (
        paired_eval._strict_action_trace_summary(
            entry,
            expected_run_nonce="f" * 32,
        )["valid"]
        is False
    )


def test_runner_argv_requires_complete_runtime_isolation_binding(
    tmp_path: Path,
) -> None:
    common = _runner_kwargs(tmp_path)
    common["runtime_isolation_binding_sha256"] = None

    with pytest.raises(ValueError, match="runtime isolation root requires"):
        build_baseline_runner_argv(**common)


def test_python_executable_validation_preserves_venv_symlink_in_both_argvs(
    tmp_path: Path,
) -> None:
    target = tmp_path / "usr-bin-python"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    python_link = tmp_path / "source-venv" / "bin" / "python"
    behavior_python_link = tmp_path / "behavior-venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    behavior_python_link.parent.mkdir(parents=True)
    python_link.symlink_to(target)
    behavior_python_link.symlink_to(target)

    python = paired_eval._lexical_executable_path(
        python_link,
        label="--python",
    )
    behavior_python = paired_eval._lexical_executable_path(
        behavior_python_link,
        label="--behavior-python",
    )
    common = _runner_kwargs(tmp_path)
    common["python"] = python
    common["behavior_python"] = behavior_python
    agentic = build_agentic_runner_argv(
        **common,
        frozen_publication_root=tmp_path / "frozen-publication",
        frozen_provenance_sha256="c" * 64,
        behavior_resource_local=tmp_path / "resources",
        behavior_resource_cache=tmp_path / "resource-cache",
        dashboard_host="127.0.0.1",
        dashboard_port=8766,
        model="gpt-5.5",
        reasoning_effort="xhigh",
        expected_run_nonce="d" * 32,
    )
    baseline = build_baseline_runner_argv(**common)

    assert python == python_link.absolute()
    assert behavior_python == behavior_python_link.absolute()
    assert python != target.resolve()
    assert behavior_python != target.resolve()
    assert agentic[0] == str(python_link.absolute())
    assert baseline[0] == str(python_link.absolute())
    assert _value_after(agentic, "--behavior-python") == str(
        behavior_python_link.absolute()
    )
    assert _value_after(baseline, "--behavior-python") == str(
        behavior_python_link.absolute()
    )


@pytest.mark.parametrize("kind", ("missing", "directory", "non_executable"))
def test_python_executable_validation_fails_closed(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    elif kind == "non_executable":
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o644)

    with pytest.raises(SystemExit, match="does not exist|not a file|not executable"):
        paired_eval._lexical_executable_path(path, label="--python")


def test_paired_manifest_serializes_real_source_snapshot_binding(
    tmp_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "tree_sha256": "b" * 64,
        "files": [],
        "source": {"commit": "c" * 40},
    }
    binding = SourceSnapshotBinding(
        snapshot_root=tmp_path / "snapshot",
        binding_path=tmp_path / "snapshot" / "source_snapshot.json",
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
        payload=payload,
    )
    manifest_path = tmp_path / "paired_eval_manifest.json"

    paired_eval._atomic_json(
        manifest_path,
        {"schema_version": 1, "source_snapshot": binding.as_dict()},
    )

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["source_snapshot"] == binding.as_dict()
    assert written["source_snapshot"]["binding_sha256"] == "a" * 64
    assert "snapshot_root" not in written["source_snapshot"]
    assert "binding_path" not in written["source_snapshot"]
    assert '"source_snapshot": source_binding.as_dict()' in inspect.getsource(
        paired_eval.main
    )


def test_pure_vla_dashboard_metadata_exposes_disabled_llm_and_one_tool(
    tmp_path: Path,
) -> None:
    state = paired_eval._new_dashboard_state(
        arm=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        public_seed=10,
        entry_output_dir=tmp_path / "entry",
        action_deadline_s=6900,
    )
    snapshot = state.snapshot()
    metadata = state._metadata

    assert state.identity["activity_instance_id"] == 108
    assert metadata["cuda-device"] == "6"
    assert metadata["controller"] == "pi0_nav_pick-only"
    assert metadata["llm-enabled"] is False
    assert metadata["planner"] == "disabled"
    assert "model" not in metadata
    assert "reasoning-effort" not in metadata
    assert metadata["public-tool-count"] == 1
    assert "max-tool-calls" not in metadata
    assert snapshot["progress"]["max_tool_calls"] is None


class _FinishingProcess:
    def __init__(self, *, running_polls: int) -> None:
        self._remaining = running_polls

    def poll(self) -> int | None:
        if self._remaining:
            self._remaining -= 1
            return None
        return 0


def test_health_monitor_samples_t0_then_every_20_minutes_for_two_hours(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]

    class _ClockProcess:
        returncode = 0

        @staticmethod
        def poll() -> int | None:
            return None if clock[0] <= 7200 else 0

    live_arms: list[LiveArm] = []
    for name, gpu, port, llm_enabled in (
        ("agentic", "7", 8766, True),
        ("pi0_nav_pick_only", "6", 8767, False),
    ):
        live = LiveArm(
            spec=_arm(
                tmp_path,
                name=name,
                gpu=gpu,
                port=port,
                llm_enabled=llm_enabled,
            ),
            server=SimpleNamespace(),
            url=f"http://127.0.0.1:{port}",
        )
        live.child = SimpleNamespace(
            process=_ClockProcess(),
            started_monotonic=0.0,
            action_deadline_monotonic=9000.0,
            cleanup_deadline_monotonic=9500.0,
            hard_deadline_monotonic=10_000.0,
            action_cleanup_started=False,
            forced_cleanup_started=False,
            cleanup_verified=False,
            safety_errors=[],
            timed_out=False,
            identity_ambiguous=False,
        )
        live_arms.append(live)

    sampled: list[int] = []

    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        paired_eval.time,
        "sleep",
        lambda _seconds: clock.__setitem__(0, clock[0] + 1200.0),
    )
    monkeypatch.setattr(paired_eval, "terminate_owned_child", lambda _child: True)
    monkeypatch.setattr(
        paired_eval,
        "_sample_health",
        lambda **kwargs: sampled.append(int(kwargs["offset_s"])),
    )

    sampled_offsets: set[int] = set()
    paired_eval._wait_for_pair(
        live_arms=(live_arms[0], live_arms[1]),
        instance_timeout_s=10_000,
        monitor_started=0.0,
        monitor_interval_s=1200,
        monitor_window_s=7200,
        sampled_offsets=sampled_offsets,
        output_root=tmp_path,
        health_previous={},
    )

    assert sampled == [0, 1200, 2400, 3600, 4800, 6000, 7200]
    assert sampled_offsets == set(sampled)


def test_each_pair_gets_fresh_monitor_epoch_offsets_and_previous_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled: list[tuple[int, int]] = []
    current_seed = [11]

    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(paired_eval.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(paired_eval, "terminate_owned_child", lambda _child: True)
    monkeypatch.setattr(
        paired_eval,
        "_completed_child_infrastructure_reason",
        lambda _live, _child: None,
    )
    monkeypatch.setattr(
        paired_eval,
        "_trusted_monitor_activity",
        lambda _arms: {
            "cohort": "agentic",
            "public_seed": current_seed[0],
            "source": "action_trace",
            "path": f"/pair/s{current_seed[0]}/behavior_action_trace.jsonl",
            "size_bytes": 1,
        },
    )
    monkeypatch.setattr(
        paired_eval,
        "_sample_health",
        lambda **kwargs: sampled.append((current_seed[0], int(kwargs["offset_s"]))),
    )

    states: list[paired_eval.PairMonitorState] = []
    for public_seed in (11, 12):
        current_seed[0] = public_seed
        state = paired_eval._new_pair_monitor_state()
        states.append(state)
        live_arms: list[LiveArm] = []
        for name, gpu, port, llm_enabled in (
            ("agentic", "7", 8766, True),
            ("pi0_nav_pick_only", "6", 8767, False),
        ):
            live = LiveArm(
                spec=_arm(
                    tmp_path,
                    name=name,
                    gpu=gpu,
                    port=port,
                    llm_enabled=llm_enabled,
                ),
                server=SimpleNamespace(),
                url=f"http://127.0.0.1:{port}",
            )
            live.child = SimpleNamespace(
                public_seed=public_seed,
                process=_FinishingProcess(running_polls=2),
                action_deadline_monotonic=6900.0,
                cleanup_deadline_monotonic=7080.0,
                hard_deadline_monotonic=7200.0,
                action_cleanup_started=False,
                forced_cleanup_started=False,
                cleanup_verified=False,
                safety_errors=[],
                timed_out=False,
                identity_ambiguous=False,
            )
            live_arms.append(live)

        paired_eval._wait_for_pair(
            live_arms=(live_arms[0], live_arms[1]),
            instance_timeout_s=7200,
            monitor_started=state.started,
            monitor_interval_s=1200,
            monitor_window_s=7200,
            sampled_offsets=state.sampled_offsets,
            output_root=tmp_path,
            health_previous=state.previous,
            monitor_start_evidence=state.start_evidence,
        )

    assert sampled == [(11, 0), (12, 0)]
    assert states[0].sampled_offsets == {0}
    assert states[1].sampled_offsets == {0}
    assert states[0].started is not states[1].started
    assert states[0].previous is not states[1].previous
    assert states[0].start_evidence["public_seed"] == 11
    assert states[1].start_evidence["public_seed"] == 12


def test_monitor_epoch_requires_bound_runtime_activity(tmp_path: Path) -> None:
    arm = _arm(
        tmp_path,
        name="agentic",
        gpu="7",
        port=8766,
        llm_enabled=True,
    )
    entry = tmp_path / "entry"
    entry.mkdir()
    child = SimpleNamespace(
        public_seed=10,
        entry_output_dir=entry,
        event_path=entry / "dashboard_events.jsonl",
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
    )

    assert paired_eval._trusted_monitor_activity((live,)) is None

    (entry / "behavior_tool_trace.jsonl").write_text("{}\n", encoding="utf-8")
    evidence = paired_eval._trusted_monitor_activity((live,))

    assert evidence is not None
    assert evidence["cohort"] == "agentic"
    assert evidence["public_seed"] == 10
    assert evidence["source"] == "tool_trace"


def test_action_deadline_latches_timeout_and_cleanup_run_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 42, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    validation_errors = ["final_result validation failed after cutoff"]
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "run_error",
                "task_success": None,
                "timed_out": False,
                "infrastructure_error": None,
                "validation_errors": validation_errors,
            }
        ),
        encoding="utf-8",
    )
    process = _Process()
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=process,
        action_deadline_monotonic=100.0,
        cleanup_deadline_monotonic=110.0,
        hard_deadline_monotonic=120.0,
        action_cleanup_started=False,
        action_deadline_exhausted=False,
        forced_cleanup_started=False,
        cleanup_verified=False,
        hard_deadline_exhausted=False,
        safety_errors=[],
        timed_out=False,
        identity_ambiguous=False,
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
    )
    idle = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )
    signals: list[int] = []

    def signal_child(
        _child: SimpleNamespace,
        *,
        sig: int,
    ) -> tuple[bool, str | None]:
        signals.append(sig)
        process.returncode = -15
        return "sent", None

    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(paired_eval.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(paired_eval, "_signal_instance_child", signal_child)
    monkeypatch.setattr(paired_eval, "terminate_owned_child", lambda _child: True)
    monkeypatch.setattr(paired_eval, "_sample_health", lambda **_kwargs: None)

    errors = paired_eval._wait_for_pair(
        live_arms=(live, idle),
        instance_timeout_s=7200,
        monitor_started=100.0,
        monitor_interval_s=1200,
        monitor_window_s=7200,
        sampled_offsets={0},
        output_root=tmp_path,
        health_previous={},
    )
    result = paired_eval._finish_dashboard_arm(live, child)
    assert errors == ()
    assert signals == [signal.SIGTERM]
    assert child.action_cleanup_started is True
    assert child.action_deadline_exhausted is True
    assert child.timed_out is True
    assert result["outcome"] == "timed_out"
    assert result["score"] == "failure"
    assert result["task_success"] is False
    assert result["timed_out"] is True
    assert result["infrastructure_error"] is None
    assert result["deadline_cleanup_artifact"] == {
        "runner_outcome": "run_error",
        "returncode": -15,
        "result_record_missing": False,
        "validation_errors": validation_errors,
    }


def test_action_deadline_already_exited_noop_does_not_latch_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 4, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "task_failed",
                "task_success": False,
                "infrastructure_error": None,
            }
        ),
        encoding="utf-8",
    )
    process = _Process()
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=process,
        action_deadline_monotonic=100.0,
        cleanup_deadline_monotonic=110.0,
        hard_deadline_monotonic=120.0,
        action_cleanup_started=False,
        action_deadline_exhausted=False,
        forced_cleanup_started=False,
        cleanup_verified=False,
        hard_deadline_exhausted=False,
        safety_errors=[],
        timed_out=False,
        identity_ambiguous=False,
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
    )
    idle = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        paired_eval.time,
        "sleep",
        lambda _seconds: setattr(process, "returncode", 0),
    )
    monkeypatch.setattr(
        paired_eval,
        "_signal_instance_child",
        lambda _child, *, sig: ("already_exited", None),
    )
    monkeypatch.setattr(paired_eval, "terminate_owned_child", lambda _child: True)
    monkeypatch.setattr(paired_eval, "_sample_health", lambda **_kwargs: None)

    errors = paired_eval._wait_for_pair(
        live_arms=(live, idle),
        instance_timeout_s=7200,
        monitor_started=100.0,
        monitor_interval_s=1200,
        monitor_window_s=7200,
        sampled_offsets={0},
        output_root=tmp_path,
        health_previous={},
    )
    result = paired_eval._finish_dashboard_arm(live, child)

    assert errors == ()
    assert child.action_cleanup_started is True
    assert child.action_deadline_exhausted is False
    assert child.timed_out is False
    assert child.safety_errors == []
    assert result["outcome"] == "task_failed"
    assert result["score"] == "failure"


def test_action_deadline_raw_success_still_wins_over_cleanup_run_error(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 42, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "run_error",
                "task_success": None,
                "infrastructure_error": "cutoff validation errors",
            }
        ),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: -15, returncode=-15),
        timed_out=True,
        action_deadline_exhausted=True,
        identity_ambiguous=False,
        cleanup_verified=True,
        safety_errors=[],
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "passed"
    assert result["score"] == "success"
    assert result["task_success"] is True
    assert result["raw_success_confirmed"] is True
    assert result["artifact_seal_complete"] is False
    assert result["infrastructure_error"]["reason"] == "multiple_infrastructure_errors"
    assert {error["reason"] for error in result["infrastructure_error"]["errors"]} == {
        "runner_infrastructure_error",
        "artifact_seal_incomplete",
    }
    assert result["infrastructure_error"]["errors"][0]["detail"] == (
        "cutoff validation errors"
    )
    assert result["deadline_cleanup_artifact"]["runner_outcome"] == "run_error"


def test_post_success_forced_cleanup_is_success_with_incomplete_seal(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s14"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 42, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: -15, returncode=-15),
        timed_out=False,
        action_deadline_exhausted=False,
        forced_cleanup_started=True,
        identity_ambiguous=False,
        cleanup_verified=True,
        safety_errors=[],
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "passed"
    assert result["task_success"] is True
    assert result["artifact_seal_complete"] is False
    assert result["infrastructure_error"]["reason"] == "artifact_seal_incomplete"
    assert result["infrastructure_error"]["forced_cleanup"] is True
    assert result["infrastructure_error"]["result_record_missing"] is True
    assert result["infrastructure_error"]["returncode"] == -15


def test_predeadline_infrastructure_exit_is_unknown_not_timeout(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 3, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "run_error",
                "task_success": None,
                "infrastructure_error": "transport unavailable",
            }
        ),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: 2, returncode=2),
        timed_out=False,
        action_deadline_exhausted=False,
        identity_ambiguous=False,
        cleanup_verified=True,
        safety_errors=[],
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "run_error"
    assert result["score"] == "unknown"
    assert result["task_success"] is None
    assert result["timed_out"] is False
    assert result["action_deadline_exhausted"] is False
    assert result["infrastructure_error"] == "transport unavailable"


@pytest.mark.parametrize("action_deadline_exhausted", (False, True))
def test_timeout_with_explicit_infrastructure_error_is_always_unknown(
    tmp_path: Path,
    action_deadline_exhausted: bool,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 5, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "run_error",
                "task_success": None,
                "timed_out": True,
                "infrastructure_error": "transport failed before signal",
            }
        ),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: 2, returncode=2),
        timed_out=True,
        action_deadline_exhausted=action_deadline_exhausted,
        identity_ambiguous=False,
        cleanup_verified=True,
        safety_errors=[],
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    early_reason = paired_eval._completed_child_infrastructure_reason(live, child)
    result = paired_eval._finish_dashboard_arm(live, child)

    assert early_reason == (
        "runner infrastructure error: transport failed before signal"
    )
    assert result["outcome"] == "run_error"
    assert result["score"] == "unknown"
    assert result["task_success"] is None
    assert result["timed_out"] is True
    assert result["infrastructure_error"] == "transport failed before signal"


def test_action_deadline_missing_final_result_is_timeout_with_forensic_artifact(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 9, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: -15, returncode=-15),
        timed_out=True,
        action_deadline_exhausted=True,
        identity_ambiguous=False,
        cleanup_verified=True,
        safety_errors=[],
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    early_reason = paired_eval._completed_child_infrastructure_reason(live, child)
    result = paired_eval._finish_dashboard_arm(live, child)

    assert early_reason is None
    assert result["outcome"] == "timed_out"
    assert result["score"] == "failure"
    assert result["task_success"] is False
    assert result["infrastructure_error"] is None
    assert result["deadline_cleanup_artifact"]["result_record_missing"] is True


def test_action_deadline_signal_failure_is_safety_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]

    class _Process:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    entry = tmp_path / "agentic" / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 5, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "run_error",
                "task_success": None,
                "infrastructure_error": None,
            }
        ),
        encoding="utf-8",
    )
    process = _Process()
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=process,
        action_deadline_monotonic=100.0,
        cleanup_deadline_monotonic=110.0,
        hard_deadline_monotonic=120.0,
        action_cleanup_started=False,
        action_deadline_exhausted=False,
        forced_cleanup_started=False,
        cleanup_verified=False,
        hard_deadline_exhausted=False,
        safety_errors=[],
        timed_out=False,
        identity_ambiguous=False,
        peer_abort_reason=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
    )
    idle = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )

    def fail_signal(
        _child: SimpleNamespace,
        *,
        sig: int,
    ) -> tuple[bool, str | None]:
        assert sig == signal.SIGTERM
        return "error", "transport failed before signal"

    def cleanup(_child: SimpleNamespace) -> bool:
        process.returncode = -15
        return True

    def advance(_seconds: float) -> None:
        clock[0] = 110.0

    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(paired_eval.time, "sleep", advance)
    monkeypatch.setattr(paired_eval, "_signal_instance_child", fail_signal)
    monkeypatch.setattr(paired_eval, "terminate_owned_child", cleanup)
    monkeypatch.setattr(paired_eval, "_sample_health", lambda **_kwargs: None)

    errors = paired_eval._wait_for_pair(
        live_arms=(live, idle),
        instance_timeout_s=7200,
        monitor_started=100.0,
        monitor_interval_s=1200,
        monitor_window_s=7200,
        sampled_offsets={0},
        output_root=tmp_path,
        health_previous={},
    )
    result = paired_eval._finish_dashboard_arm(live, child)

    assert child.action_cleanup_started is True
    assert child.action_deadline_exhausted is False
    assert child.timed_out is True
    assert child.safety_errors == [
        "agentic_action_deadline_cleanup: transport failed before signal"
    ]
    assert live.disabled_reason == child.safety_errors[0]
    assert errors == ()
    assert result["outcome"] == "run_error"
    assert result["score"] == "unknown"
    assert result["task_success"] is None
    assert result["infrastructure_error"]["reason"] in {
        "cleanup_or_identity_unverified",
        "child_safety_errors",
    }


def test_health_record_checks_dashboard_vla_gpu_and_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _arm(
        tmp_path,
        name="agentic",
        gpu="7",
        port=8766,
        llm_enabled=True,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        vla_endpoint="http://127.0.0.1:9123",
    )
    monkeypatch.setattr(paired_eval, "_http_json", lambda _url: {"status": "ok"})
    monkeypatch.setattr(paired_eval, "_unknown_gpu_processes", lambda *_args: [])

    record = paired_eval._health_record(
        live=live,
        offset_s=1200,
        gpu_processes={
            "7": [
                {
                    "pid": 123,
                    "process_name": "vla",
                    "used_memory_mib": 2048,
                }
            ]
        },
        free_disk_bytes=50 * 1024**3,
    )

    assert record["offset_s"] == 1200
    assert record["gpu"] == "7"
    assert record["dashboard"] == {
        "url": "http://127.0.0.1:8766",
        "healthz": True,
        "run_api": True,
    }
    assert record["vla"]["healthz"] is True
    assert record["gpu_compute_processes"][0]["pid"] == 123
    assert record["free_disk_bytes"] == 50 * 1024**3
    assert record["healthy"] is True


def test_health_record_binds_runtime_health_and_live_raw_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _arm(
        tmp_path,
        name="agentic",
        gpu="7",
        port=8766,
        llm_enabled=True,
    )
    entry = tmp_path / "entry"
    entry.mkdir()
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"event": "step", "step": 0, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    event_path = entry / "dashboard_events.jsonl"
    event_path.write_text("{}\n", encoding="utf-8")
    child = SimpleNamespace(
        public_seed=10,
        event_path=event_path,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: None),
        pid=123,
        pgid=123,
        sid=123,
        start_ticks=1,
        argv_sha256="d" * 64,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
        vla_endpoint="http://127.0.0.1:9123",
    )
    monkeypatch.setattr(paired_eval, "_http_json", lambda _url: {"status": "ok"})
    monkeypatch.setattr(
        paired_eval, "owned_child_identity_matches", lambda _child: True
    )
    monkeypatch.setattr(
        paired_eval,
        "_recorded_env_health",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr(paired_eval, "_unknown_gpu_processes", lambda *_args: [])

    record = paired_eval._health_record(
        live=live,
        offset_s=0,
        gpu_processes={"7": []},
        free_disk_bytes=50 * 1024**3,
        external_runtime_health={
            "runtime_base": "/dev/shm",
            "free_bytes": 400 * 1024**3,
            "owner_binding_sha256": "a" * 64,
            "errors": ["runtime_base_identity_changed"],
        },
    )

    assert record["raw_official_success"]["confirmed"] is True
    assert record["raw_official_success"]["trace_valid"] is True
    assert record["external_runtime"]["runtime_base"] == "/dev/shm"
    assert "external_runtime_unhealthy" in record["warnings"]
    assert live.runtime_violations == ["runtime_base_identity_changed"]


def test_health_record_warns_when_live_progress_does_not_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _arm(
        tmp_path,
        name="agentic",
        gpu="7",
        port=8766,
        llm_enabled=True,
    )
    entry = tmp_path / "entry"
    entry.mkdir()
    event_path = entry / "dashboard_events.jsonl"
    event_path.write_text("{}\n", encoding="utf-8")
    child = SimpleNamespace(
        public_seed=10,
        event_path=event_path,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: None),
        pid=123,
        pgid=123,
        sid=123,
        start_ticks=1,
        argv_sha256="d" * 64,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
        vla_endpoint="http://127.0.0.1:9123",
    )
    monkeypatch.setattr(paired_eval, "_http_json", lambda _url: {"status": "ok"})
    monkeypatch.setattr(
        paired_eval, "owned_child_identity_matches", lambda _child: True
    )
    monkeypatch.setattr(
        paired_eval,
        "_recorded_env_health",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr(paired_eval, "_unknown_gpu_processes", lambda *_args: [])

    record = paired_eval._health_record(
        live=live,
        offset_s=1200,
        gpu_processes={"7": []},
        free_disk_bytes=20 * 1024**3,
        previous={
            "public_seed": 10,
            "frame_idx": None,
            "timeline_revision": None,
            "event_sink_size_bytes": event_path.stat().st_size,
        },
    )

    assert "no_progress_since_previous_sample" in record["warnings"]
    assert record["healthy"] is False


def test_pair_result_timeout_is_failure_but_infrastructure_error_is_unknown(
    tmp_path: Path,
) -> None:
    arm = _arm(
        tmp_path,
        name="pi0_nav_pick_only",
        gpu="6",
        port=8767,
        llm_enabled=False,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )

    timeout_root = tmp_path / "timeout"
    timeout_root.mkdir()
    timeout_entry = timeout_root / "entry"
    timeout_entry.mkdir()
    (timeout_entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 0, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (timeout_root / "baseline_eval_results.jsonl").write_text(
        json.dumps(
            {
                "outcome": "timed_out",
                "task_success": False,
                "timed_out": True,
                "infrastructure_error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    timeout_child = SimpleNamespace(
        output_root=timeout_root,
        entry_output_dir=timeout_entry,
        process=SimpleNamespace(returncode=0),
        timed_out=True,
        identity_ambiguous=False,
    )
    timeout_result = paired_eval._finish_dashboard_arm(live, timeout_child)
    assert timeout_result["score"] == "failure"
    assert timeout_result["task_success"] is False
    assert timeout_result["timed_out"] is True

    error_root = tmp_path / "infrastructure"
    error_root.mkdir()
    error_entry = error_root / "entry"
    error_entry.mkdir()
    (error_entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 0, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (error_root / "baseline_eval_results.jsonl").write_text(
        json.dumps(
            {
                "outcome": "infrastructure_error",
                "task_success": False,
                "timed_out": False,
                "infrastructure_error": "transport unavailable",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    error_child = SimpleNamespace(
        output_root=error_root,
        entry_output_dir=error_entry,
        process=SimpleNamespace(returncode=2),
        timed_out=False,
        identity_ambiguous=False,
    )
    error_result = paired_eval._finish_dashboard_arm(live, error_child)
    assert error_result["score"] == "unknown"
    assert error_result["task_success"] is None
    assert error_result["timed_out"] is False


@pytest.mark.parametrize(
    ("identity_ambiguous", "cleanup_verified"),
    ((True, True), (False, False)),
)
def test_unverified_cleanup_or_identity_overrides_timeout_as_unknown(
    tmp_path: Path,
    identity_ambiguous: bool,
    cleanup_verified: bool,
) -> None:
    arm = _arm(
        tmp_path,
        name="pi0_nav_pick_only",
        gpu="6",
        port=8767,
        llm_enabled=False,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )
    root = tmp_path / "timeout-unverified"
    entry = root / "entry"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 0, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (root / "baseline_eval_results.jsonl").write_text(
        json.dumps(
            {
                "outcome": "timed_out",
                "task_success": False,
                "timed_out": True,
                "infrastructure_error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=root,
        entry_output_dir=entry,
        process=SimpleNamespace(returncode=0),
        timed_out=True,
        identity_ambiguous=identity_ambiguous,
        cleanup_verified=cleanup_verified,
    )

    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "run_error"
    assert result["score"] == "unknown"
    assert result["task_success"] is None
    assert result["timed_out"] is True
    assert result["infrastructure_error"] == {
        "reason": "cleanup_or_identity_unverified",
        "identity_ambiguous": identity_ambiguous,
        "cleanup_verified": cleanup_verified,
    }


def test_strict_action_trace_rejects_malformed_json_and_symlink(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    trace = malformed / "behavior_action_trace.jsonl"
    trace.write_text(
        json.dumps({"step": 0, "info_done": {"success": True}}) + "\n{broken\n",
        encoding="utf-8",
    )
    summary = paired_eval._strict_action_trace_summary(malformed)
    assert summary["valid"] is False
    assert summary["official_success_binding"] is None
    assert summary["action_trace_sha256"]
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )
    malformed_result = paired_eval._finish_dashboard_arm(
        live,
        SimpleNamespace(
            output_root=malformed,
            entry_output_dir=malformed,
            process=SimpleNamespace(returncode=0),
            timed_out=False,
            identity_ambiguous=False,
        ),
    )
    assert malformed_result["score"] == "unknown"
    assert malformed_result["task_success"] is None

    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "behavior_action_trace.jsonl").symlink_to(trace)
    linked_summary = paired_eval._strict_action_trace_summary(linked)
    assert linked_summary["valid"] is False
    assert linked_summary["official_success_binding"] is None
    assert linked_summary["action_trace_sha256"] is None


def test_valid_trace_uses_shared_success_binding_and_passed_fallback(
    tmp_path: Path,
) -> None:
    arm = _arm(
        tmp_path,
        name="agentic",
        gpu="7",
        port=8766,
        llm_enabled=True,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )
    root = tmp_path / "success"
    entry = root / "entry"
    entry.mkdir(parents=True)
    payload = (json.dumps({"step": 4, "info_done": {"success": True}}) + "\n").encode()
    (entry / "behavior_action_trace.jsonl").write_bytes(payload)
    (entry / "final_result.json").write_text(
        json.dumps({"task_success": True, "artifact_seal_complete": True}),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=root,
        entry_output_dir=entry,
        process=SimpleNamespace(returncode=0),
        timed_out=False,
        identity_ambiguous=False,
    )

    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "passed"
    assert result["runner_outcome"] == "passed"
    assert result["task_success"] is True
    assert result["raw_success_confirmed"] is True
    assert result["raw_official_success_binding"]["first_success_step"] == 4
    assert (
        result["raw_official_success_binding"]["action_trace_sha256"]
        == result["action_trace_binding"]["sha256"]
    )


@pytest.mark.parametrize("llm_enabled", [True, False])
def test_both_dashboard_relays_enforce_arm_allowed_tools(
    tmp_path: Path,
    llm_enabled: bool,
) -> None:
    arm = _arm(
        tmp_path,
        name="agentic" if llm_enabled else "pi0_nav_pick_only",
        gpu="7" if llm_enabled else "6",
        port=8766 if llm_enabled else 8767,
        llm_enabled=llm_enabled,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url=f"http://127.0.0.1:{arm.dashboard_port}",
    )
    state = paired_eval._new_dashboard_state(
        arm=arm,
        public_seed=10,
        entry_output_dir=tmp_path / "entry",
        action_deadline_s=6900,
    )
    relay = paired_eval._dashboard_event_relay(
        live=live,
        event_path=tmp_path / "entry" / "dashboard_events.jsonl",
        state=state,
    )

    assert relay.allowed_tools == frozenset(arm.allowed_tools)


def _exited_instance_child_record(child: SimpleNamespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "exited",
        "pid": 222,
        "pgid": 222,
        "sid": 222,
        "start_ticks": 333,
        "runner_pid": child.pid,
        "runner_pgid": child.pgid,
        "runner_sid": child.sid,
        "runner_start_ticks": child.start_ticks,
        "task_name": "picking_up_trash",
        "public_seed": child.public_seed,
        "activity_instance_id": 108,
        "entry_output_dir": str(child.entry_output_dir.resolve()),
        "source_snapshot_root": str(child.source_snapshot_root.resolve()),
        "source_snapshot_binding_sha256": child.source_snapshot_binding_sha256,
        "cuda_device": child.arm.gpu,
        "argv_sha256": "a" * 64,
        "started_at": "2026-07-26T00:00:00Z",
        "updated_at": "2026-07-26T00:01:00Z",
        "action_deadline_s": child.action_deadline_s,
        "cleanup_deadline_s": child.cleanup_deadline_s,
        "instance_timeout_s": child.instance_timeout_s,
        "started_monotonic_ns": child.started_monotonic_ns,
        "action_deadline_monotonic_ns": child.action_deadline_monotonic_ns,
        "cleanup_deadline_monotonic_ns": child.cleanup_deadline_monotonic_ns,
        "hard_deadline_monotonic_ns": child.hard_deadline_monotonic_ns,
    }


def test_instance_child_contract_binds_absolute_deadlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runner"
    entry = root / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    child = SimpleNamespace(
        output_root=root,
        entry_output_dir=entry,
        pid=111,
        pgid=111,
        sid=111,
        start_ticks=222,
        public_seed=10,
        source_snapshot_root=tmp_path / "snapshot",
        source_snapshot_binding_sha256="b" * 64,
        arm=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        started_monotonic_ns=1_000_000_000,
        action_deadline_monotonic_ns=6_901_000_000_000,
        cleanup_deadline_monotonic_ns=7_081_000_000_000,
        hard_deadline_monotonic_ns=7_201_000_000_000,
    )
    child.source_snapshot_root.mkdir()
    record = _exited_instance_child_record(child)
    (root / "instance_child_process.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    monkeypatch.setattr(paired_eval, "_identity_is_live", lambda _identity: False)

    loaded, error = paired_eval._load_instance_child_process(child)
    assert error is None
    assert loaded == record

    record["hard_deadline_monotonic_ns"] += 1
    (root / "instance_child_process.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    loaded, error = paired_eval._load_instance_child_process(child)
    assert loaded is None
    assert "binding differs" in str(error)


def test_nested_cleanup_order_and_waits_stay_before_hard_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []

    class _Process:
        pid = 111
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            events.append(("wait", timeout))
            raise paired_eval.subprocess.TimeoutExpired("runner", timeout)

    process = _Process()
    child = SimpleNamespace(
        process=process,
        entry_output_dir=tmp_path / "entry",
        hard_deadline_monotonic=100.0,
        pgid=111,
        identity_ambiguous=False,
        safety_errors=[],
    )
    nested = {"state": "running", "pid": 222, "pgid": 222}
    nested_alive = [True]
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 50.0)
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: (nested, None),
    )
    monkeypatch.setattr(
        paired_eval,
        "_signal_instance_child",
        lambda _child, *, sig: (
            events.append(("nested", sig)),
            nested_alive.__setitem__(0, False)
            if sig == paired_eval.signal.SIGKILL
            else None,
            ("sent", None),
        )[-1],
    )
    monkeypatch.setattr(
        paired_eval,
        "_identity_is_live",
        lambda _identity: nested_alive[0],
    )
    monkeypatch.setattr(
        paired_eval,
        "_manifest_cleanup_within_deadline",
        lambda _child: events.append(("manifest", None)) is None,
    )
    monkeypatch.setattr(
        paired_eval,
        "owned_child_identity_matches",
        lambda _child: process.returncode is None,
    )

    def killpg(_pgid: int, sig: Any) -> None:
        events.append(("outer", sig))
        if sig == paired_eval.signal.SIGKILL:
            process.returncode = -9

    monkeypatch.setattr(paired_eval.os, "killpg", killpg)

    assert paired_eval.terminate_owned_child(child) is True
    assert events[0] == ("nested", paired_eval.signal.SIGTERM)
    assert events[1] == ("manifest", None)
    assert ("outer", paired_eval.signal.SIGTERM) in events
    assert ("nested", paired_eval.signal.SIGKILL) in events
    assert ("outer", paired_eval.signal.SIGKILL) in events
    assert all(float(timeout) <= 50.0 for kind, timeout in events if kind == "wait")


def test_hard_deadline_cleanup_never_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []

    class _Process:
        pid = 111
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            waits.append(timeout)
            raise AssertionError("must not wait at the hard deadline")

    process = _Process()
    child = SimpleNamespace(
        process=process,
        entry_output_dir=tmp_path / "entry",
        hard_deadline_monotonic=100.0,
        pgid=111,
        identity_ambiguous=False,
        safety_errors=[],
    )
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: ({"state": "exited"}, None),
    )
    monkeypatch.setattr(
        paired_eval,
        "_manifest_cleanup_within_deadline",
        lambda _child: True,
    )
    monkeypatch.setattr(
        paired_eval,
        "owned_child_identity_matches",
        lambda _child: process.returncode is None,
    )

    def killpg(_pgid: int, sig: Any) -> None:
        if sig == paired_eval.signal.SIGKILL:
            process.returncode = -9

    monkeypatch.setattr(paired_eval.os, "killpg", killpg)

    assert paired_eval.terminate_owned_child(child) is True
    assert waits == []


def test_post_success_cleanup_override_never_uses_later_instance_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[float] = []
    manifest_deadlines: list[float | None] = []

    class _Process:
        pid = 111
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float) -> int:
            waits.append(timeout)
            raise AssertionError("must not wait past the success cleanup edge")

    process = _Process()
    child = SimpleNamespace(
        process=process,
        entry_output_dir=tmp_path / "entry",
        hard_deadline_monotonic=1000.0,
        hard_deadline_monotonic_ns=1_000_000_000_000,
        pgid=111,
        identity_ambiguous=False,
        safety_errors=[],
    )
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 190.0)
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: ({"state": "exited"}, None),
    )

    def manifest_cleanup(
        _child: SimpleNamespace,
        *,
        deadline_monotonic: float | None = None,
    ) -> bool:
        manifest_deadlines.append(deadline_monotonic)
        return True

    monkeypatch.setattr(
        paired_eval,
        "_manifest_cleanup_within_deadline",
        manifest_cleanup,
    )
    monkeypatch.setattr(
        paired_eval,
        "owned_child_identity_matches",
        lambda _child: process.returncode is None,
    )

    def killpg(_pgid: int, sig: Any) -> None:
        if sig == paired_eval.signal.SIGKILL:
            process.returncode = -9

    monkeypatch.setattr(paired_eval.os, "killpg", killpg)

    assert (
        paired_eval.terminate_owned_child(
            child,
            deadline_monotonic=190.0,
        )
        is True
    )
    assert waits == []
    assert manifest_deadlines == [190.0]


def test_persistent_vla_is_disabled_and_rechecked_before_any_runner_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    binding = {"binding_sha256": "a" * 64}
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        vla_endpoint="http://127.0.0.1:9123",
        vla_process=SimpleNamespace(pid=321, poll=lambda: None),
        vla_pgid=321,
        vla_sid=321,
        vla_start_ticks=987,
    )
    health = iter(
        (
            {"config_name": "pi05_behavior", "actions_enabled": True},
            {"config_name": "pi05_behavior", "actions_enabled": False},
        )
    )

    class _Client:
        def __init__(self, endpoint: str) -> None:
            assert endpoint == live.vla_endpoint

        def healthz(
            self,
            *,
            timeout_ms: int,
            expected_checkpoint_binding: dict[str, Any],
        ) -> dict[str, Any]:
            assert timeout_ms == 5000
            assert expected_checkpoint_binding is binding
            events.append("health")
            return next(health)

        @staticmethod
        def disable_actions(*, timeout_ms: int) -> dict[str, Any]:
            assert timeout_ms == 5000
            events.append("disable")
            return {"actions_enabled": False}

        @staticmethod
        def close() -> None:
            events.append("close")

    monkeypatch.setattr(paired_eval, "BehaviorVLAClient", _Client)
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: True)

    result = paired_eval._prepare_persistent_vla_for_runner(
        live,
        checkpoint_binding=binding,
    )
    events.append("spawn")

    assert result["actions_enabled"] is False
    assert live.vla_disabled_health == result
    assert events == ["health", "disable", "health", "close", "spawn"]
    main_source = inspect.getsource(paired_eval.main)
    assert main_source.index("_prepare_persistent_vla_for_runner(") < (
        main_source.index("_spawn_owned_child(")
    )


def test_persistent_vla_is_quiesced_by_owner_after_verified_runner_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    binding = {"binding_sha256": "a" * 64}
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        vla_endpoint="http://127.0.0.1:9123",
        vla_process=SimpleNamespace(pid=321, poll=lambda: None),
        vla_pgid=321,
        vla_sid=321,
        vla_start_ticks=987,
    )
    child = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: 1),
        cleanup_verified=True,
        identity_ambiguous=False,
        post_run_vla_quiesced=False,
        post_run_vla_health=None,
    )
    health = iter(
        (
            {
                "config_name": "pi05_behavior",
                "pid": 321,
                "actions_enabled": True,
            },
            {
                "config_name": "pi05_behavior",
                "pid": 321,
                "actions_enabled": False,
            },
        )
    )

    class _Client:
        def __init__(self, endpoint: str) -> None:
            assert endpoint == live.vla_endpoint

        def healthz(
            self,
            *,
            timeout_ms: int,
            expected_checkpoint_binding: dict[str, Any],
        ) -> dict[str, Any]:
            assert timeout_ms == 5000
            assert expected_checkpoint_binding is binding
            events.append("health")
            return next(health)

        @staticmethod
        def disable_actions(*, timeout_ms: int) -> dict[str, Any]:
            assert timeout_ms == 5000
            events.append("disable")
            return {"actions_enabled": False}

        @staticmethod
        def close() -> None:
            events.append("close")

    monkeypatch.setattr(paired_eval, "BehaviorVLAClient", _Client)
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: True)

    result = paired_eval._quiesce_persistent_vla_after_runner(
        live,
        child,
        checkpoint_binding=binding,
    )

    assert result["actions_enabled"] is False
    assert child.post_run_vla_quiesced is True
    assert child.post_run_vla_health == result
    assert live.vla_disabled_health == result
    assert events == ["health", "disable", "health", "close"]


def test_post_run_vla_quiescence_requires_verified_runner_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        vla_endpoint="http://127.0.0.1:9123",
        vla_process=SimpleNamespace(pid=321, poll=lambda: None),
        vla_pgid=321,
        vla_sid=321,
        vla_start_ticks=987,
    )
    child = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: 1),
        cleanup_verified=False,
        identity_ambiguous=False,
    )
    monkeypatch.setattr(
        paired_eval,
        "BehaviorVLAClient",
        lambda _endpoint: pytest.fail("VLA must not be contacted before cleanup"),
    )

    with pytest.raises(RuntimeError, match="runner cleanup is not verified"):
        paired_eval._quiesce_persistent_vla_after_runner(
            live,
            child,
            checkpoint_binding={"binding_sha256": "a" * 64},
        )


def test_post_run_vla_quiescence_rejects_changed_server_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        vla_endpoint="http://127.0.0.1:9123",
        vla_process=SimpleNamespace(pid=321, poll=lambda: None),
        vla_pgid=321,
        vla_sid=321,
        vla_start_ticks=987,
    )
    child = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: 1),
        cleanup_verified=True,
        identity_ambiguous=False,
    )
    events: list[str] = []

    class _Client:
        def __init__(self, _endpoint: str) -> None:
            pass

        @staticmethod
        def healthz(**_kwargs: Any) -> dict[str, Any]:
            events.append("health")
            return {
                "config_name": "pi05_behavior",
                "pid": 999,
                "actions_enabled": True,
            }

        @staticmethod
        def disable_actions(**_kwargs: Any) -> dict[str, Any]:
            events.append("disable")
            return {"actions_enabled": False}

        @staticmethod
        def close() -> None:
            events.append("close")

    monkeypatch.setattr(paired_eval, "BehaviorVLAClient", _Client)
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: True)

    with pytest.raises(RuntimeError, match="pre-quiescence identity"):
        paired_eval._quiesce_persistent_vla_after_runner(
            live,
            child,
            checkpoint_binding={"binding_sha256": "a" * 64},
        )
    assert events == ["health", "close"]


@pytest.mark.parametrize(
    "post_health",
    (
        {},
        {"config_name": "pi05_behavior"},
        {"config_name": "pi05_behavior", "actions_enabled": True},
        {"config_name": "pi05_behavior", "actions_enabled": None},
        {"config_name": "pi05_behavior", "actions_enabled": 0},
    ),
)
def test_persistent_vla_bad_post_disable_health_blocks_runner_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    post_health: dict[str, Any],
) -> None:
    events: list[str] = []
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
        vla_endpoint="http://127.0.0.1:9124",
        vla_process=SimpleNamespace(pid=654, poll=lambda: None),
        vla_pgid=654,
        vla_sid=654,
        vla_start_ticks=789,
    )
    health = iter(
        (
            {"config_name": "pi05_behavior", "actions_enabled": True},
            post_health,
        )
    )

    class _Client:
        def __init__(self, _endpoint: str) -> None:
            pass

        def healthz(self, **_kwargs: Any) -> dict[str, Any]:
            events.append("health")
            return next(health)

        @staticmethod
        def disable_actions(**_kwargs: Any) -> dict[str, Any]:
            events.append("disable")
            return {"actions_enabled": False}

        @staticmethod
        def close() -> None:
            events.append("close")

    monkeypatch.setattr(paired_eval, "BehaviorVLAClient", _Client)
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: True)

    with pytest.raises(RuntimeError, match="VLA health"):
        paired_eval._prepare_persistent_vla_for_runner(
            live,
            checkpoint_binding={"binding_sha256": "a" * 64},
        )

    assert live.vla_disabled_health is None
    assert "spawn" not in events


def _running_reparented_instance_record(
    child: SimpleNamespace,
    command: list[str],
) -> dict[str, Any]:
    record = _exited_instance_child_record(child)
    record["state"] = "running"
    record["argv_sha256"] = hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def _reparented_child_fixture(
    tmp_path: Path,
    *,
    command: list[str],
) -> tuple[SimpleNamespace, dict[str, Any]]:
    root = tmp_path / "runner"
    entry = root / "picking_up_trash_s10"
    entry.mkdir(parents=True)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    child = SimpleNamespace(
        output_root=root,
        entry_output_dir=entry,
        pid=111,
        pgid=111,
        sid=111,
        start_ticks=222,
        public_seed=10,
        source_snapshot_root=snapshot,
        source_snapshot_binding_sha256="b" * 64,
        arm=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        started_monotonic_ns=1_000_000_000,
        action_deadline_monotonic_ns=6_901_000_000_000,
        cleanup_deadline_monotonic_ns=7_081_000_000_000,
        hard_deadline_monotonic_ns=7_201_000_000_000,
        hard_deadline_monotonic=7201.0,
        process=SimpleNamespace(poll=lambda: 2, returncode=2),
        argv=(command[0], "outer-runner.py", "--python", command[0]),
        identity_ambiguous=False,
        safety_errors=[],
    )
    record = _running_reparented_instance_record(child, command)
    (root / "instance_child_process.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )
    return child, record


def test_reparented_nested_child_exact_receipt_allows_safe_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = [sys.executable, "-m", "robots.behavior.serial_eval", "--instance-child"]
    child, record = _reparented_child_fixture(tmp_path, command=command)
    live = [True]
    original_read_bytes = Path.read_bytes

    def proc_identity(pid: int) -> dict[str, int] | None:
        if pid != record["pid"] or not live[0]:
            return None
        return {
            "pid": record["pid"],
            "ppid": 1,
            "pgid": record["pgid"],
            "sid": record["sid"],
            "start_ticks": record["start_ticks"],
        }

    def read_bytes(path: Path) -> bytes:
        if path == Path(f"/proc/{record['pid']}/cmdline"):
            return b"\0".join(os.fsencode(item) for item in command) + b"\0"
        return original_read_bytes(path)

    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pgid: int, sig: signal.Signals) -> None:
        signals.append((pgid, sig))
        live[0] = False

    monkeypatch.setattr(paired_eval, "_proc_identity", proc_identity)
    monkeypatch.setattr(
        paired_eval,
        "_proc_executable",
        lambda pid: Path(command[0]).resolve() if pid == record["pid"] else None,
    )
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(paired_eval.os, "killpg", killpg)
    monkeypatch.setattr(
        paired_eval,
        "_manifest_cleanup_within_deadline",
        lambda _child: True,
    )
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 7200.0)
    hard_deadline = child.hard_deadline_monotonic

    assert paired_eval.terminate_owned_child(child) is True
    assert signals == [(record["pgid"], signal.SIGTERM)]
    assert child.hard_deadline_monotonic == hard_deadline
    assert child.identity_ambiguous is False


def test_nested_child_accepts_only_equivalent_python_argv0_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_python = Path(sys.executable).resolve()
    lexical_python = tmp_path / "venv-python"
    lexical_python.symlink_to(resolved_python)
    recorded_command = [
        str(lexical_python),
        "-m",
        "robots.behavior.cli",
        "--env",
        "behavior",
    ]
    proc_command = [str(resolved_python), *recorded_command[1:]]
    child, record = _reparented_child_fixture(
        tmp_path,
        command=recorded_command,
    )
    current = {
        "pid": record["pid"],
        "ppid": 1,
        "pgid": record["pgid"],
        "sid": record["sid"],
        "start_ticks": record["start_ticks"],
    }
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == Path(f"/proc/{record['pid']}/cmdline"):
            return b"\0".join(os.fsencode(item) for item in proc_command) + b"\0"
        return original_read_bytes(path)

    monkeypatch.setattr(
        paired_eval,
        "_proc_identity",
        lambda pid: current if pid == record["pid"] else None,
    )
    monkeypatch.setattr(
        paired_eval,
        "_proc_executable",
        lambda pid: resolved_python if pid == record["pid"] else None,
    )
    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    loaded, error = paired_eval._load_instance_child_process(child)

    assert error is None
    assert loaded == record


def test_nested_child_equivalent_argv0_still_rejects_argument_or_executable_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_python = Path(sys.executable).resolve()
    lexical_python = tmp_path / "venv-python"
    lexical_python.symlink_to(resolved_python)
    recorded_command = [
        str(lexical_python),
        "-m",
        "robots.behavior.cli",
        "--env",
        "behavior",
    ]
    child, record = _reparented_child_fixture(
        tmp_path,
        command=recorded_command,
    )
    current = {
        "pid": record["pid"],
        "ppid": 1,
        "pgid": record["pgid"],
        "sid": record["sid"],
        "start_ticks": record["start_ticks"],
    }
    drifted_command = [
        str(resolved_python),
        *recorded_command[1:],
        "--unexpected",
    ]
    proc_command = [drifted_command]
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == Path(f"/proc/{record['pid']}/cmdline"):
            return b"\0".join(os.fsencode(item) for item in proc_command[0]) + b"\0"
        return original_read_bytes(path)

    monkeypatch.setattr(
        paired_eval,
        "_proc_identity",
        lambda pid: current if pid == record["pid"] else None,
    )
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        paired_eval,
        "_proc_executable",
        lambda pid: resolved_python if pid == record["pid"] else None,
    )
    loaded, error = paired_eval._load_instance_child_process(child)
    assert loaded is None
    assert "argv SHA-256" in str(error)

    proc_command[0] = [str(resolved_python), *recorded_command[1:]]
    monkeypatch.setattr(
        paired_eval,
        "_proc_executable",
        lambda pid: Path("/bin/true").resolve() if pid == record["pid"] else None,
    )
    loaded, error = paired_eval._load_instance_child_process(child)
    assert loaded is None
    assert "argv SHA-256" in str(error)


@pytest.mark.parametrize("drift", ("pgid", "sid", "start_ticks", "cmdline"))
def test_reparented_nested_child_identity_or_cmdline_drift_refuses_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    command = [sys.executable, "-m", "robots.behavior.serial_eval", "--instance-child"]
    child, record = _reparented_child_fixture(tmp_path, command=command)
    current = {
        "pid": record["pid"],
        "ppid": 1,
        "pgid": record["pgid"],
        "sid": record["sid"],
        "start_ticks": record["start_ticks"],
    }
    if drift != "cmdline":
        current[drift] += 1
    actual_command = command if drift != "cmdline" else [*command, "--unexpected"]
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == Path(f"/proc/{record['pid']}/cmdline"):
            return b"\0".join(os.fsencode(item) for item in actual_command) + b"\0"
        return original_read_bytes(path)

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        paired_eval,
        "_proc_identity",
        lambda pid: current if pid == record["pid"] else None,
    )
    monkeypatch.setattr(
        paired_eval,
        "_proc_executable",
        lambda pid: Path(command[0]).resolve() if pid == record["pid"] else None,
    )
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        paired_eval.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    signal_status, error = paired_eval._signal_instance_child(
        child,
        sig=signal.SIGTERM,
    )

    assert signal_status == "error"
    assert error
    assert signals == []


def test_signal_instance_child_exited_receipt_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = SimpleNamespace()
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: ({"state": "exited"}, None),
    )
    monkeypatch.setattr(
        paired_eval.os,
        "killpg",
        lambda *_args: pytest.fail("exited receipt must not be signalled"),
    )

    signal_status, error = paired_eval._signal_instance_child(
        child,
        sig=signal.SIGTERM,
    )

    assert signal_status == "already_exited"
    assert error is None


def test_signal_instance_child_processlookup_identity_gone_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = SimpleNamespace()
    identity = {"state": "running", "pgid": 4321}
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: (identity, None),
    )
    monkeypatch.setattr(
        paired_eval.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(paired_eval, "_identity_is_live", lambda _identity: False)

    signal_status, error = paired_eval._signal_instance_child(
        child,
        sig=signal.SIGTERM,
    )

    assert signal_status == "already_exited"
    assert error is None


def test_signal_instance_child_reports_only_real_killpg_as_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = SimpleNamespace()
    identity = {"state": "running", "pgid": 4321}
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: (identity, None),
    )
    monkeypatch.setattr(
        paired_eval.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    signal_status, error = paired_eval._signal_instance_child(
        child,
        sig=signal.SIGTERM,
    )

    assert signal_status == "sent"
    assert error is None
    assert signals == [(4321, signal.SIGTERM)]


@pytest.mark.parametrize(
    ("gpu_stdout", "process_stdout", "expected"),
    (
        ("7\n", "", "GPU compute-process probe failed"),
        ("7, GPU-7\n", "GPU-X, 123, python, 100\n", "malformed row"),
        ("7, GPU-7\n", "GPU-7, not-a-pid, python, 100\n", "numeric fields"),
    ),
)
def test_gpu_process_probe_failure_or_malformed_rows_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    gpu_stdout: str,
    process_stdout: str,
    expected: str,
) -> None:
    responses = iter(
        (
            SimpleNamespace(stdout=gpu_stdout),
            SimpleNamespace(stdout=process_stdout),
        )
    )
    monkeypatch.setattr(
        paired_eval.subprocess,
        "run",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match=expected):
        paired_eval._gpu_compute_processes()


def test_gpu_probe_error_is_shared_safety_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )
    monkeypatch.setattr(
        paired_eval,
        "_gpu_compute_processes",
        lambda: (_ for _ in ()).throw(RuntimeError("probe unavailable")),
    )
    monkeypatch.setattr(paired_eval, "_unknown_gpu_processes", lambda *_args: [])

    errors = paired_eval._gpu_ownership_errors((live,))

    assert errors
    assert any("gpu_compute_process_probe_failed" in error for error in errors)


@pytest.mark.parametrize(
    ("arm_name", "receipt_name"),
    (
        ("agentic", "run_manifest.json"),
        ("pi0_nav_pick_only", "baseline_owned_processes.json"),
    ),
)
def test_exact_current_arm_env_gpu_pid_is_owned_after_clean_runner_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm_name: str,
    receipt_name: str,
) -> None:
    gpu = "7" if arm_name == "agentic" else "6"
    entry = tmp_path / arm_name / "entry"
    entry.mkdir(parents=True)
    env_identity = {
        "pid": 584525,
        "pgid": 584525,
        "sid": 584525,
        "start_ticks": 786152853,
    }
    receipt = (
        {"processes": {"env": env_identity}}
        if receipt_name == "run_manifest.json"
        else {"env": env_identity, "vla": None}
    )
    (entry / receipt_name).write_text(json.dumps(receipt), encoding="utf-8")
    child = SimpleNamespace(
        pid=111,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: 0),
        cleanup_verified=True,
        identity_ambiguous=False,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name=arm_name,
            gpu=gpu,
            port=8766 if gpu == "7" else 8767,
            llm_enabled=gpu == "7",
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1",
        child=child,
    )
    current = {**env_identity, "ppid": 1}
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: False)
    monkeypatch.setattr(
        paired_eval, "owned_child_identity_matches", lambda _child: False
    )
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: (
            {
                "state": "exited",
                "pid": 222,
            },
            None,
        ),
    )
    monkeypatch.setattr(
        paired_eval,
        "_proc_identity",
        lambda pid: current if pid == env_identity["pid"] else None,
    )
    monkeypatch.setattr(
        paired_eval,
        "_proc_start_ticks",
        lambda pid: env_identity["start_ticks"] if pid == env_identity["pid"] else None,
    )

    unknown = paired_eval._unknown_gpu_processes(
        live,
        {
            gpu: [
                {
                    **env_identity,
                    "process_name": "python",
                    "used_memory_mib": 1024,
                }
            ]
        },
    )

    assert unknown == []
    health = paired_eval._recorded_env_health(entry, owner_child=child)
    assert health is not None
    assert health["exact_identity_live"] is True
    assert health["lineage_valid"] is False
    assert health["ownership_valid"] is True


def test_exact_retired_arm_env_gpu_row_is_owned_after_clean_runner_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = tmp_path / "pi0_nav_pick_only" / "entry"
    entry.mkdir(parents=True)
    env_identity = {
        "pid": 584525,
        "pgid": 584525,
        "sid": 584525,
        "start_ticks": 786152853,
    }
    (entry / "baseline_owned_processes.json").write_text(
        json.dumps({"env": env_identity, "vla": None}),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        pid=111,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: 0),
        cleanup_verified=True,
        identity_ambiguous=False,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1",
        child=child,
    )
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: False)
    monkeypatch.setattr(
        paired_eval, "owned_child_identity_matches", lambda _child: False
    )
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: ({"state": "exited", "pid": 222}, None),
    )
    monkeypatch.setattr(paired_eval, "_proc_identity", lambda _pid: None)
    monkeypatch.setattr(paired_eval, "_proc_start_ticks", lambda _pid: None)

    unknown = paired_eval._unknown_gpu_processes(
        live,
        {
            "6": [
                {
                    **env_identity,
                    "process_name": "[No data]",
                    "used_memory_mib": 1024,
                }
            ]
        },
    )

    assert unknown == []


@pytest.mark.parametrize(
    ("cleanup_verified", "identity_ambiguous", "nested_state", "gpu_start_ticks"),
    (
        (False, False, "exited", 786152853),
        (True, True, "exited", 786152853),
        (True, False, "running", 786152853),
        (True, False, "exited", 786152854),
    ),
)
def test_retired_env_gpu_row_alias_requires_exact_clean_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_verified: bool,
    identity_ambiguous: bool,
    nested_state: str,
    gpu_start_ticks: int,
) -> None:
    entry = tmp_path / "pi0_nav_pick_only" / "entry"
    entry.mkdir(parents=True)
    env_identity = {
        "pid": 584525,
        "pgid": 584525,
        "sid": 584525,
        "start_ticks": 786152853,
    }
    (entry / "baseline_owned_processes.json").write_text(
        json.dumps({"env": env_identity, "vla": None}),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        pid=111,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: 0),
        cleanup_verified=cleanup_verified,
        identity_ambiguous=identity_ambiguous,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1",
        child=child,
    )
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: False)
    monkeypatch.setattr(
        paired_eval, "owned_child_identity_matches", lambda _child: False
    )
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: ({"state": nested_state, "pid": 222}, None),
    )
    monkeypatch.setattr(paired_eval, "_proc_identity", lambda _pid: None)
    monkeypatch.setattr(paired_eval, "_proc_start_ticks", lambda _pid: None)

    process = {
        **env_identity,
        "start_ticks": gpu_start_ticks,
        "process_name": "[No data]",
        "used_memory_mib": 1024,
    }
    unknown = paired_eval._unknown_gpu_processes(live, {"6": [process]})

    assert unknown == [process]


def test_live_runner_cannot_claim_unrelated_gpu_pid_via_env_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = tmp_path / "agentic" / "entry"
    entry.mkdir(parents=True)
    env_identity = {
        "pid": 584525,
        "pgid": 584525,
        "sid": 584525,
        "start_ticks": 786152853,
    }
    (entry / "run_manifest.json").write_text(
        json.dumps({"processes": {"env": env_identity}}),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        pid=111,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: None),
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1",
        child=child,
    )
    current = {**env_identity, "ppid": 999}
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: False)
    monkeypatch.setattr(
        paired_eval, "owned_child_identity_matches", lambda _child: True
    )
    monkeypatch.setattr(
        paired_eval,
        "_load_instance_child_process",
        lambda _child: ({"state": "running", "pid": 222}, None),
    )
    monkeypatch.setattr(
        paired_eval,
        "_proc_identity",
        lambda pid: current if pid == env_identity["pid"] else None,
    )
    monkeypatch.setattr(
        paired_eval,
        "_proc_start_ticks",
        lambda pid: env_identity["start_ticks"] if pid == env_identity["pid"] else None,
    )
    monkeypatch.setattr(paired_eval, "_pid_descends_from", lambda *_args: False)

    unknown = paired_eval._unknown_gpu_processes(
        live,
        {
            "7": [
                {
                    **env_identity,
                    "process_name": "python",
                    "used_memory_mib": 1024,
                }
            ]
        },
    )

    assert [process["pid"] for process in unknown] == [env_identity["pid"]]
    health = paired_eval._recorded_env_health(entry, owner_child=child)
    assert health is not None
    assert health["ownership_valid"] is False


def test_transient_unknown_gpu_pid_latches_arm_unknown_and_shared_safety_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = _arm(
        tmp_path,
        name="agentic",
        gpu="7",
        port=8766,
        llm_enabled=True,
    )
    live = LiveArm(
        spec=arm,
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )
    root = tmp_path / "success"
    entry = root / "entry"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 4, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps({"task_success": True, "artifact_seal_complete": True}),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=root,
        entry_output_dir=entry,
        process=SimpleNamespace(returncode=0),
        timed_out=False,
        identity_ambiguous=False,
    )
    monkeypatch.setattr(paired_eval, "_http_json", lambda _url: {"status": "ok"})
    monkeypatch.setattr(
        paired_eval,
        "_unknown_gpu_processes",
        lambda *_args: [{"pid": 999, "process_name": "foreign"}],
    )

    health = paired_eval._health_record(
        live=live,
        offset_s=1200,
        gpu_processes={"7": [{"pid": 999, "process_name": "foreign"}]},
        free_disk_bytes=50 * 1024**3,
    )
    result = paired_eval._finish_dashboard_arm(live, child)
    monkeypatch.setattr(paired_eval, "_gpu_compute_processes", lambda: {"7": []})
    monkeypatch.setattr(paired_eval, "_unknown_gpu_processes", lambda *_args: [])
    next_pair_errors = paired_eval._gpu_ownership_errors((live,))

    assert "unknown_gpu_compute_process" in health["warnings"]
    assert live.gpu_ownership_violations
    assert result["outcome"] == "passed"
    assert result["score"] == "success"
    assert result["task_success"] is True
    assert result["infrastructure_error"]["reason"] == "gpu_ownership_violation"
    assert next_pair_errors


def test_vla_cleanup_identity_drift_refuses_any_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=321, poll=lambda: None)
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        vla_process=process,
        vla_pgid=321,
        vla_sid=321,
        vla_start_ticks=987,
    )
    terminated: list[Any] = []
    monkeypatch.setattr(paired_eval, "_vla_identity_matches", lambda _live: False)
    monkeypatch.setattr(
        paired_eval,
        "_terminate_process",
        lambda proc: terminated.append(proc),
    )

    assert paired_eval._terminate_verified_vla(live) is False
    assert terminated == []
    assert live.disabled_reason == "vla_cleanup_identity_ambiguous"


def test_action_trace_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    entry = outside / "entry"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 0, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    summary = paired_eval._strict_action_trace_summary(linked / "entry")

    assert summary["valid"] is False
    assert summary["official_success_binding"] is None
    assert summary["action_trace_sha256"] is None


@pytest.mark.parametrize("sink_kind", ("fifo", "directory"))
def test_dashboard_relay_rejects_nonregular_sink_without_blocking_stop(
    tmp_path: Path,
    sink_kind: str,
) -> None:
    sink = tmp_path / "dashboard_events.jsonl"
    if sink_kind == "fifo":
        os.mkfifo(sink)
    else:
        sink.mkdir()
    relay = paired_eval.DashboardEventRelay(
        sink,
        SimpleNamespace(),
        allowed_tools=("pi0_nav_pick",),
    )
    joins: list[float] = []
    relay._thread = SimpleNamespace(
        join=lambda *, timeout: joins.append(timeout),
        is_alive=lambda: False,
    )

    relay.stop(timeout_s=0.25)

    assert len(joins) == 1
    assert 0.0 <= joins[0] <= 0.25
    assert any("not regular" in violation for violation in relay.violations)


def test_relay_stop_uses_only_remaining_absolute_hard_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float | None] = []
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        relay=SimpleNamespace(stop=lambda *, timeout_s: calls.append(timeout_s)),
    )
    child = SimpleNamespace(hard_deadline_monotonic=100.0)
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 99.75)

    paired_eval._stop_relay_within_child_deadline(live, child)

    assert calls == [0.25]
    assert child.hard_deadline_monotonic == 100.0


def test_failed_cleanup_is_not_verified_and_child_reference_is_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: 2),
        cleanup_verified=False,
        safety_errors=["nested survivor"],
        timed_out=False,
        identity_ambiguous=True,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
    )
    idle = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )
    monkeypatch.setattr(paired_eval, "terminate_owned_child", lambda _child: False)
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 10.0)

    errors = paired_eval._wait_for_pair(
        live_arms=(live, idle),
        instance_timeout_s=7200,
        monitor_started=10.0,
        monitor_interval_s=1200,
        monitor_window_s=7200,
        sampled_offsets={0},
        output_root=tmp_path,
        health_previous={},
    )

    assert errors == ()
    assert live.disabled_reason == ("agentic_cleanup_unverified: nested survivor")
    assert child.cleanup_verified is False
    assert live.child is child
    assert paired_eval._release_child_if_cleanup_verified(live) is False
    assert live.child is child


def _sealed_agentic_task_failure_record() -> dict[str, object]:
    return {
        "outcome": "task_failed",
        "task_success": False,
        "raw_official_success": False,
        "raw_official_success_binding": None,
        "first_raw_success_env_step": None,
        "timed_out": False,
        "subprocess_exit_code": 0,
        "instance_state_binding_valid": True,
        "validation_errors": [],
        "infrastructure_error": None,
        "artifact_seal_complete": True,
        "workflow_complete": True,
    }


def test_agentic_sealed_task_failure_accepts_aggregate_returncode_one(
    tmp_path: Path,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s12"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 1, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (entry.parent / "eval_results.jsonl").write_text(
        json.dumps(_sealed_agentic_task_failure_record()) + "\n",
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: 1, returncode=1),
        cleanup_verified=True,
        identity_ambiguous=False,
        timed_out=False,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    assert paired_eval._completed_child_infrastructure_reason(live, child) is None
    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "task_failed"
    assert result["runner_outcome"] == "task_failed"
    assert result["task_success"] is False
    assert result["score"] == "failure"
    assert result["returncode"] == 1
    assert result["infrastructure_error"] is None


@pytest.mark.parametrize(
    ("record_patch", "missing_field", "returncode"),
    (
        ({"artifact_seal_complete": False}, None, 1),
        ({"validation_errors": ["inner validation failed"]}, None, 1),
        ({"infrastructure_error": "inner transport failed"}, None, 1),
        ({"raw_official_success": True}, None, 1),
        ({"workflow_complete": False}, None, 1),
        ({}, "subprocess_exit_code", 1),
        ({}, "workflow_complete", 1),
        ({}, None, 2),
    ),
)
def test_agentic_noncanonical_or_infrastructure_exit_remains_unknown(
    tmp_path: Path,
    record_patch: dict[str, object],
    missing_field: str | None,
    returncode: int,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s12"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 1, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    record = _sealed_agentic_task_failure_record()
    record.update(record_patch)
    if missing_field is not None:
        record.pop(missing_field)
    (entry.parent / "eval_results.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: returncode, returncode=returncode),
        cleanup_verified=True,
        identity_ambiguous=False,
        timed_out=False,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
    )

    assert paired_eval._completed_child_infrastructure_reason(live, child) is not None
    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == "run_error"
    assert result["task_success"] is None
    assert result["score"] == "unknown"


@pytest.mark.parametrize(
    ("returncode", "expected_outcome"),
    ((0, "task_failed"), (1, "run_error")),
)
def test_pure_vla_task_failure_keeps_arm_specific_returncode_contract(
    tmp_path: Path,
    returncode: int,
    expected_outcome: str,
) -> None:
    entry = tmp_path / "pi0_nav_pick_only" / "picking_up_trash_s12"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 1, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (entry / "final_result.json").write_text(
        json.dumps(
            {
                "outcome": "task_failed",
                "task_success": False,
                "timed_out": False,
                "infrastructure_error": None,
                "artifact_seal_complete": True,
            }
        ),
        encoding="utf-8",
    )
    child = SimpleNamespace(
        output_root=entry.parent,
        entry_output_dir=entry,
        process=SimpleNamespace(poll=lambda: returncode, returncode=returncode),
        cleanup_verified=True,
        identity_ambiguous=False,
        timed_out=False,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
    )

    reason = paired_eval._completed_child_infrastructure_reason(live, child)
    result = paired_eval._finish_dashboard_arm(live, child)

    assert result["outcome"] == expected_outcome
    if returncode == 0:
        assert reason is None
        assert result["task_success"] is False
        assert result["score"] == "failure"
    else:
        assert reason == "runner exited with returncode 1"
        assert result["task_success"] is None
        assert result["score"] == "unknown"


def test_event_channel_tool_events_enforce_lane_allowlist(tmp_path: Path) -> None:
    events: list[dict[str, Any]] = []
    dashboard = SimpleNamespace(on_event=lambda payload: events.append(payload))
    relay = paired_eval.DashboardEventRelay(
        tmp_path / "dashboard_events.jsonl",
        dashboard,
        allowed_tools=("pi0_nav_pick",),
    )

    relay._relay_record(
        {
            "channel": "event",
            "payload": {"type": "tool_call", "name": "observe"},
        }
    )
    relay._relay_record(
        {
            "channel": "event",
            "payload": {"type": "tool_call", "name": "pi0_nav_pick"},
        }
    )

    assert events == [{"type": "tool_call", "name": "pi0_nav_pick"}]
    assert any("tool is not allowed: observe" in item for item in relay.violations)


def test_instance_state_result_binding_match_preserves_lane_result(
    tmp_path: Path,
) -> None:
    expected = "a" * 64
    arm_result = {
        "outcome": "passed",
        "task_success": True,
        "score": "success",
        "result": {"result": {"instance_state_sha256": expected}},
    }

    error = paired_eval._enforce_instance_state_result_binding(
        arm_result=arm_result,
        entry_output_dir=tmp_path,
        expected_sha256=expected,
    )

    assert error is None
    assert arm_result["outcome"] == "passed"
    assert arm_result["task_success"] is True
    assert arm_result["score"] == "success"
    assert arm_result["instance_state_sha256"] == expected


@pytest.mark.parametrize(
    ("recorded", "reason"),
    (
        (None, "instance_state_sha256_missing"),
        ("b" * 64, "instance_state_sha256_mismatch"),
    ),
)
def test_instance_state_result_binding_missing_or_mismatch_is_unknown(
    tmp_path: Path,
    recorded: str | None,
    reason: str,
) -> None:
    result: dict[str, Any] = {}
    if recorded is not None:
        result["instance_state_sha256"] = recorded
    arm_result = {
        "outcome": "passed",
        "task_success": True,
        "score": "success",
        "result": result,
    }

    error = paired_eval._enforce_instance_state_result_binding(
        arm_result=arm_result,
        entry_output_dir=tmp_path,
        expected_sha256="a" * 64,
    )

    assert error == reason
    assert arm_result["outcome"] == "run_error"
    assert arm_result["task_success"] is None
    assert arm_result["score"] == "unknown"
    assert arm_result["infrastructure_error"] == reason
    assert arm_result["instance_state_sha256"] == recorded
    assert arm_result["expected_instance_state_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("recorded", "reason"),
    (
        (None, "instance_state_sha256_missing"),
        ("b" * 64, "instance_state_sha256_mismatch"),
    ),
)
def test_instance_state_binding_error_preserves_trace_confirmed_raw_success(
    tmp_path: Path,
    recorded: str | None,
    reason: str,
) -> None:
    result: dict[str, Any] = {}
    if recorded is not None:
        result["instance_state_sha256"] = recorded
    arm_result = {
        "outcome": "passed",
        "task_success": True,
        "score": "success",
        "raw_success_confirmed": True,
        "result": result,
    }

    error = paired_eval._enforce_instance_state_result_binding(
        arm_result=arm_result,
        entry_output_dir=tmp_path,
        expected_sha256="a" * 64,
    )

    assert error == reason
    assert arm_result["outcome"] == "passed"
    assert arm_result["task_success"] is True
    assert arm_result["score"] == "success"
    assert arm_result["infrastructure_error"] == reason
    assert arm_result["instance_state_sha256"] == recorded
    assert arm_result["expected_instance_state_sha256"] == "a" * 64


def test_agentic_preflight_not_admitted_does_not_stop_pure_or_block_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PureProcess:
        returncode: int | None = None
        poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            if self.poll_count >= 3:
                self.returncode = 0
            return self.returncode

    pure_process = _PureProcess()
    pure_child = SimpleNamespace(
        process=pure_process,
        action_deadline_monotonic=100.0,
        cleanup_deadline_monotonic=110.0,
        hard_deadline_monotonic=120.0,
        action_cleanup_started=False,
        forced_cleanup_started=False,
        cleanup_verified=True,
        identity_ambiguous=False,
    )
    agentic = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        disabled_reason="agentic_llm_preflight_failed",
    )
    pure = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
        child=pure_child,
    )
    terminated: list[Any] = []
    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(paired_eval.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(paired_eval, "_sample_health", lambda **_kwargs: None)
    monkeypatch.setattr(
        paired_eval,
        "terminate_owned_child",
        lambda child: terminated.append(child) is None,
    )

    safety_errors = paired_eval._wait_for_pair(
        live_arms=(agentic, pure),
        instance_timeout_s=7200,
        monitor_started=10.0,
        monitor_interval_s=1200,
        monitor_window_s=7200,
        sampled_offsets={0},
        output_root=tmp_path,
        health_previous={},
    )
    pair_record = {
        "arms": {
            "agentic": {
                "outcome": "not_run",
                "task_success": None,
                "score": "unknown",
                "reason": "agentic_llm_preflight_failed",
                "infrastructure_error": {
                    "reason": "agentic_llm_preflight_failed",
                },
            },
            "pi0_nav_pick_only": {
                "outcome": "task_failed",
                "task_success": False,
                "score": "failure",
                "infrastructure_error": None,
            },
        }
    }
    reason, remaining = paired_eval._pair_campaign_transition(
        public_seeds=(10, 11),
        pair_index=0,
        pair_record=pair_record,
        pair_safety_errors=safety_errors,
        live_arms=(agentic, pure),
    )

    assert agentic.child is None
    assert agentic.disabled_reason == "agentic_llm_preflight_failed"
    assert pure_process.returncode == 0
    assert terminated == []
    assert safety_errors == ()
    assert reason is None
    assert remaining == ()


def test_pair_transition_blocks_canonical_arms_on_safety_error(
    tmp_path: Path,
) -> None:
    lives = (
        LiveArm(
            spec=_arm(
                tmp_path,
                name="agentic",
                gpu="7",
                port=8766,
                llm_enabled=True,
            ),
            server=SimpleNamespace(),
            url="http://127.0.0.1:8766",
        ),
        LiveArm(
            spec=_arm(
                tmp_path,
                name="pi0_nav_pick_only",
                gpu="6",
                port=8767,
                llm_enabled=False,
            ),
            server=SimpleNamespace(),
            url="http://127.0.0.1:8767",
        ),
    )
    canonical_success = {
        "outcome": "passed",
        "score": "success",
        "task_success": True,
        "infrastructure_error": None,
    }
    pair_record = {"arms": {live.spec.name: dict(canonical_success) for live in lives}}

    reason, remaining = paired_eval._pair_campaign_transition(
        public_seeds=(10, 11, 12),
        pair_index=0,
        pair_record=pair_record,
        pair_safety_errors=("cleanup_identity_ambiguous",),
        live_arms=lives,
    )

    assert reason == "cleanup_identity_ambiguous"
    assert [record["public_seed"] for record in remaining] == [11, 12]
    assert all(
        arm["score"] == "unknown"
        and arm["infrastructure_error"]["reason"] == "campaign_blocked_by_prior_pair"
        for record in remaining
        for arm in record["arms"].values()
    )


def test_eval_state_bindings_include_canonical_s3_and_detect_drift(
    tmp_path: Path,
) -> None:
    spec = paired_eval.PICKING_UP_TRASH_TASK_SPEC
    state_dir = (
        tmp_path
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
        / "scenes"
        / spec.scene_model
        / "json"
        / spec.state_dir_name
    )
    state_dir.mkdir(parents=True)
    paths: dict[int, Path] = {}
    for seed in (3, 10):
        instance = spec.instance_for_public_seed(
            seed,
            phase="explore" if seed == 3 else "eval",
        )
        path = (
            state_dir / f"{spec.scene_model}_task_{spec.task_name}_"
            f"{spec.activity_definition_id}_{instance}_template-tro_state.json"
        )
        path.write_text(json.dumps({"seed": seed}), encoding="utf-8")
        paths[seed] = path

    bindings = paired_eval._freeze_public_instance_states(
        behavior_repo=tmp_path,
        public_seeds=(3, 10),
    )

    assert set(bindings) == {"3", "10"}
    assert bindings["3"]["phase"] == "explore"
    assert bindings["10"]["phase"] == "eval"
    assert paired_eval._verify_public_instance_states(bindings) == []

    paths[10].write_text(json.dumps({"seed": 10, "changed": True}), encoding="utf-8")
    assert paired_eval._verify_public_instance_states(bindings) == ["s10_state_changed"]


def test_raw_success_starts_180_second_cleanup_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = tmp_path / "pure" / "picking_up_trash_s14"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"event": "step", "step": 7, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    clock = [10.0]

    class _Process:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    process = _Process()
    child = SimpleNamespace(
        process=process,
        entry_output_dir=entry,
        action_deadline_monotonic=9000.0,
        cleanup_deadline_monotonic=9500.0,
        hard_deadline_monotonic=10_000.0,
        action_cleanup_started=False,
        forced_cleanup_started=False,
        cleanup_verified=False,
        safety_errors=[],
        timed_out=False,
        identity_ambiguous=False,
        official_success_observed_monotonic=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="pi0_nav_pick_only",
            gpu="6",
            port=8767,
            llm_enabled=False,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8767",
        child=child,
    )
    cleanup_times: list[tuple[float, float | None]] = []

    def advance(_seconds: float) -> None:
        clock[0] += 90.0

    def terminate(
        _child: SimpleNamespace,
        *,
        deadline_monotonic: float | None = None,
    ) -> bool:
        cleanup_times.append((clock[0], deadline_monotonic))
        process.returncode = -15
        return True

    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(paired_eval.time, "sleep", advance)
    monkeypatch.setattr(paired_eval, "terminate_owned_child", terminate)
    monkeypatch.setattr(paired_eval, "_sample_health", lambda **_kwargs: None)

    assert (
        paired_eval._wait_for_pair(
            live_arms=(live,),
            instance_timeout_s=10_000,
            monitor_started=10.0,
            monitor_interval_s=1200,
            monitor_window_s=7200,
            sampled_offsets={0},
            output_root=tmp_path,
            health_previous={},
        )
        == ()
    )
    assert cleanup_times == [(190.0, 190.0)]
    assert child.cleanup_verified is True


def test_exited_child_cleanup_keeps_raw_success_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = tmp_path / "agentic" / "picking_up_trash_s14"
    entry.mkdir(parents=True)
    (entry / "behavior_action_trace.jsonl").write_text(
        json.dumps({"event": "step", "step": 9, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    child = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: 0, returncode=0),
        entry_output_dir=entry,
        cleanup_verified=False,
        safety_errors=[],
        identity_ambiguous=False,
        official_success_observed_monotonic=None,
    )
    live = LiveArm(
        spec=_arm(
            tmp_path,
            name="agentic",
            gpu="7",
            port=8766,
            llm_enabled=True,
        ),
        server=SimpleNamespace(),
        url="http://127.0.0.1:8766",
        child=child,
    )
    deadlines: list[float | None] = []

    def terminate(
        _child: SimpleNamespace,
        *,
        deadline_monotonic: float | None = None,
    ) -> bool:
        deadlines.append(deadline_monotonic)
        return True

    monkeypatch.setattr(paired_eval.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(paired_eval, "terminate_owned_child", terminate)

    assert (
        paired_eval._wait_for_pair(
            live_arms=(live,),
            instance_timeout_s=7200,
            monitor_started=10.0,
            monitor_interval_s=1200,
            monitor_window_s=7200,
            sampled_offsets={0},
            output_root=tmp_path,
            health_previous={},
        )
        == ()
    )
    assert deadlines == [190.0]
    assert child.cleanup_verified is True


def test_shared_gate_does_not_read_agentic_publication_for_pure_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = SimpleNamespace(as_dict=lambda: {"binding": "same"})
    monkeypatch.setattr(
        paired_eval,
        "validate_source_snapshot",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        paired_eval,
        "verify_pinned_dataset_resources",
        lambda _binding: resource,
    )
    monkeypatch.setattr(
        paired_eval,
        "_verify_public_instance_states",
        lambda _bindings: [],
    )
    monkeypatch.setattr(
        paired_eval,
        "_validate_agentic_publication",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Pure shared gate must not inspect Agentic publication")
        ),
    )

    assert (
        paired_eval._validate_shared_invariants(
            snapshot_root=tmp_path,
            source_binding_sha256="a" * 64,
            resource_binding=resource,
            expected_resource_binding={"binding": "same"},
            frozen_publication_root=tmp_path / "agentic-only",
            frozen_provenance_sha256="b" * 64,
            expected_publication_binding={},
            instance_state_bindings={},
            output_root=tmp_path,
            min_free_disk_bytes=0,
            validate_agentic_publication=False,
        )
        == []
    )
