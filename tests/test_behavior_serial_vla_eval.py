"""Pure-VLA GPU6 comparison-Eval contracts."""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.behavior import serial_vla_eval
from robots.behavior.serial_vla_eval import (
    ACTION_DEADLINE_S,
    ACTION_STEPS_PER_CALL,
    CHUNKS_PER_CALL,
    CLEANUP_DEADLINE_S,
    INSTANCE_TIMEOUT_S,
    WINDOW_COMPLETION_RESERVE_S,
    SourceSnapshot,
    WindowLoopResult,
    build_vla_eval_plan,
    run_vla_window_loop,
    validate_pure_vla_tool_trace,
)
from robots.behavior.task_specs import get_task_spec


class _FakePrimitives:
    def __init__(
        self,
        *,
        success_on_call: int | None = None,
        fail_on_call: int | None = None,
    ) -> None:
        self.total_env_steps = 0
        self.success_on_call = success_on_call
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, Any]] = []

    def pi0_nav_pick(self, *, instruction: str, chunks: int) -> dict[str, Any]:
        call_index = len(self.calls) + 1
        self.calls.append({"instruction": instruction, "chunks": chunks})
        if self.fail_on_call == call_index:
            raise RuntimeError("injected VLA transport failure")
        self.total_env_steps += ACTION_STEPS_PER_CALL
        success = self.success_on_call == call_index
        return {
            "name": "pi0_nav_pick",
            "primitive_success": True,
            "task_success": success,
            "official_success_source": "legacy_runtime_value",
            "official_success_field_path": "legacy.runtime.path",
            "official_success_receipt": (
                {
                    "official_success_source": "legacy_runtime_value",
                    "official_success_field_path": "legacy.runtime.path",
                    "raw_done": {"success": True},
                }
                if success
                else None
            ),
            "requested_chunks": CHUNKS_PER_CALL,
            "chunks_used": CHUNKS_PER_CALL,
            "full_chunks_executed": CHUNKS_PER_CALL,
            "vla_env_steps_used": ACTION_STEPS_PER_CALL,
            "action_horizon": 32,
            "exact_requested_chunks_completed": True,
            "terminated": False,
            "truncated": False,
        }


def test_state_binding_error_does_not_revoke_trace_confirmed_success() -> None:
    payload = {
        "task_success": True,
        "success": True,
        "instance_state_sha256": "b" * 64,
    }

    serial_vla_eval._bind_result_to_frozen_instance_state(
        payload,
        expected_state_sha256="a" * 64,
    )

    assert payload["task_success"] is True
    assert payload["success"] is True
    assert payload["infrastructure_error"] == (
        "instance state SHA-256 differs from the frozen parent binding"
    )
    assert payload["expected_instance_state_sha256"] == "a" * 64


def test_missing_child_state_hash_requires_parent_postrun_observation() -> None:
    payload = {"task_success": False, "success": False}

    serial_vla_eval._bind_result_to_frozen_instance_state(
        payload,
        expected_state_sha256="a" * 64,
        parent_observed_state_sha256="a" * 64,
    )

    assert payload["instance_state_sha256"] == "a" * 64
    assert payload["instance_state_binding"] == {
        "valid": True,
        "source": "parent_postrun_native_state",
        "expected_sha256": "a" * 64,
        "observed_sha256": "a" * 64,
    }
    assert payload["task_success"] is False


def test_missing_child_and_parent_state_hash_is_infrastructure_unknown() -> None:
    payload = {"task_success": False, "success": False}

    serial_vla_eval._bind_result_to_frozen_instance_state(
        payload,
        expected_state_sha256="a" * 64,
        parent_observed_state_sha256=None,
    )

    assert "instance_state_sha256" not in payload
    assert payload["instance_state_binding"]["valid"] is False
    assert payload["task_success"] is None
    assert "parent post-run verification failed" in payload["infrastructure_error"]


def test_cleanup_signal_escapes_ordinary_exception_handlers() -> None:
    assert issubclass(serial_vla_eval._InstanceDeadline, BaseException)
    assert not issubclass(serial_vla_eval._InstanceDeadline, Exception)


def test_pure_vla_module_cli_guard_invokes_main_and_import_is_inert(
    tmp_path: Path,
) -> None:
    repo_root = Path(serial_vla_eval.__file__).resolve().parents[2]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": (
            str(repo_root)
            if not existing_pythonpath
            else os.pathsep.join((str(repo_root), existing_pythonpath))
        ),
    }

    invoked = subprocess.run(
        [sys.executable, "-m", "robots.behavior.serial_vla_eval"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert invoked.returncode == 2
    assert "--output-root" in invoked.stderr
    assert "required" in invoked.stderr
    assert "Traceback" not in invoked.stderr

    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import robots.behavior.serial_vla_eval as module; "
                "print(module.__name__)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert imported.returncode == 0
    assert imported.stdout.strip() == "robots.behavior.serial_vla_eval"
    assert imported.stderr == ""
    assert not tuple(tmp_path.iterdir())


def test_pure_vla_defaults_are_exact_80_chunk_windows_with_two_hour_watchdog() -> None:
    assert CHUNKS_PER_CALL == 80
    assert ACTION_STEPS_PER_CALL == 2560
    assert WINDOW_COMPLETION_RESERVE_S == 900
    assert (ACTION_DEADLINE_S, CLEANUP_DEADLINE_S, INSTANCE_TIMEOUT_S) == (
        6900,
        7080,
        7200,
    )


def test_external_vla_requires_explicit_external_gpu_lock_ownership() -> None:
    complete = {
        "vla_endpoint": "http://127.0.0.1:9123",
        "source_snapshot_root": "/sealed/source",
        "source_snapshot_binding_sha256": "a" * 64,
        "runtime_isolation_root": "/bound/runtime",
        "runtime_isolation_binding_sha256": "b" * 64,
        "instance_started_monotonic_ns": 9_000_000_000,
        "action_deadline_monotonic_ns": (9_000_000_000 + 6900 * 1_000_000_000),
        "cleanup_deadline_monotonic_ns": (9_000_000_000 + 7080 * 1_000_000_000),
        "hard_deadline_monotonic_ns": (9_000_000_000 + 7200 * 1_000_000_000),
    }

    with pytest.raises(SystemExit, match="explicit external GPU lock ownership"):
        serial_vla_eval._validate_external_gpu_lock_contract(
            SimpleNamespace(
                external_gpu_lock_owned=False,
                **complete,
            )
        )

    serial_vla_eval._validate_external_gpu_lock_contract(
        SimpleNamespace(
            external_gpu_lock_owned=True,
            **complete,
        )
    )


@pytest.mark.parametrize(
    "missing",
    (
        "vla_endpoint",
        "source_snapshot_root",
        "source_snapshot_binding_sha256",
        "runtime_isolation_root",
        "runtime_isolation_binding_sha256",
        "instance_started_monotonic_ns",
        "action_deadline_monotonic_ns",
        "cleanup_deadline_monotonic_ns",
        "hard_deadline_monotonic_ns",
    ),
)
def test_external_gpu_lock_ownership_requires_all_bound_inputs(missing: str) -> None:
    values = {
        "vla_endpoint": "http://127.0.0.1:9123",
        "source_snapshot_root": "/sealed/source",
        "source_snapshot_binding_sha256": "a" * 64,
        "runtime_isolation_root": "/bound/runtime",
        "runtime_isolation_binding_sha256": "b" * 64,
        "instance_started_monotonic_ns": 9_000_000_000,
        "action_deadline_monotonic_ns": (9_000_000_000 + 6900 * 1_000_000_000),
        "cleanup_deadline_monotonic_ns": (9_000_000_000 + 7080 * 1_000_000_000),
        "hard_deadline_monotonic_ns": (9_000_000_000 + 7200 * 1_000_000_000),
    }
    values[missing] = None

    with pytest.raises(SystemExit, match="requires an external VLA"):
        serial_vla_eval._validate_external_gpu_lock_contract(
            SimpleNamespace(
                external_gpu_lock_owned=True,
                **values,
            )
        )


def test_external_gpu_lock_manifest_contract_is_explicit_and_unclaimed() -> None:
    contract = serial_vla_eval._gpu_lock_contract(
        args=SimpleNamespace(external_gpu_lock_owned=True),
        lock_paths=(Path("/tmp/output.lock"),),
    )

    assert contract["owner"] == "external_paired_supervisor"
    assert contract["claimed_here"] is False
    assert contract["active_lock_paths"] == ["/tmp/output.lock"]
    assert contract["path"] not in contract["active_lock_paths"]


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"actions_enabled": True},
        {"actions_enabled": None},
        {"actions_enabled": 0},
        {"actions_enabled": "false"},
    ),
)
def test_pure_vla_health_requires_actions_enabled_exactly_false(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="actions_enabled=false"):
        serial_vla_eval._require_disabled_vla_health(payload)


def test_pure_vla_health_accepts_actions_enabled_exactly_false() -> None:
    serial_vla_eval._require_disabled_vla_health({"actions_enabled": False})


@pytest.mark.parametrize(
    ("runtime_root", "binding_sha256"),
    (
        (None, None),
        ("/external/runtime", None),
        (None, "b" * 64),
    ),
)
def test_pure_vla_requires_complete_external_runtime_isolation_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_root: str | None,
    binding_sha256: str | None,
) -> None:
    output_root = tmp_path / "new-parent" / "eval"
    validation_calls: list[object] = []
    monkeypatch.setattr(
        serial_vla_eval,
        "validate_campaign_runtime_isolation",
        lambda *_args, **_kwargs: validation_calls.append(object()),
    )

    with pytest.raises(SystemExit, match="requires an external runtime isolation"):
        serial_vla_eval._load_external_runtime_isolation(
            runtime_root=runtime_root,
            binding_sha256=binding_sha256,
            output_root=output_root,
        )

    assert validation_calls == []
    assert not output_root.parent.exists()


@pytest.mark.parametrize(
    ("runtime_relative", "output_relative"),
    (
        ("runtime", "runtime"),
        ("runtime", "runtime/eval"),
        ("eval/runtime", "eval"),
    ),
)
def test_pure_vla_rejects_runtime_output_tree_overlap_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_relative: str,
    output_relative: str,
) -> None:
    runtime_root = (tmp_path / runtime_relative).resolve()
    output_root = tmp_path / output_relative
    monkeypatch.setattr(
        serial_vla_eval,
        "validate_campaign_runtime_isolation",
        lambda *_args, **_kwargs: SimpleNamespace(
            root=runtime_root,
            cuda_device="6",
            binding_sha256="b" * 64,
        ),
    )

    with pytest.raises(SystemExit, match="must be disjoint"):
        serial_vla_eval._load_external_runtime_isolation(
            runtime_root=runtime_root,
            binding_sha256="b" * 64,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_pure_vla_accepts_disjoint_bound_external_runtime_without_copying_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root = (tmp_path / "external-runtime").resolve()
    output_root = tmp_path / "outputs" / "eval"
    binding = SimpleNamespace(
        root=runtime_root,
        cuda_device="6",
        binding_sha256="b" * 64,
    )
    calls: list[tuple[object, object]] = []

    def _validate(
        root: object,
        *,
        expected_binding_sha256: object,
    ) -> object:
        calls.append((root, expected_binding_sha256))
        return binding

    monkeypatch.setattr(
        serial_vla_eval,
        "validate_campaign_runtime_isolation",
        _validate,
    )

    actual = serial_vla_eval._load_external_runtime_isolation(
        runtime_root=runtime_root,
        binding_sha256="b" * 64,
        output_root=output_root,
    )

    assert actual is binding
    assert calls == [(runtime_root, "b" * 64)]
    assert not output_root.exists()
    assert not (output_root / "_runtime_isolation").exists()


def test_managed_vla_uses_formal_launcher_log_dir_not_runtime_cache(
    tmp_path: Path,
) -> None:
    eval_output = tmp_path / "eval-output"
    eval_output.mkdir()

    actual = serial_vla_eval._vla_launcher_log_dir(eval_output)

    assert actual == eval_output / "launcher_logs" / "vla"
    assert actual.is_dir()
    assert not (eval_output / "_managed_vla").exists()
    assert not (eval_output / "_runtime_isolation").exists()
    assert not (eval_output / "xdg").exists()
    assert not (eval_output / "ov_cache").exists()
    assert not (eval_output / "omni_user").exists()
    assert not (eval_output / "isaac").exists()
    assert not (eval_output / "tmp").exists()


def test_pure_vla_source_has_no_output_local_runtime_isolation_fallback() -> None:
    source = Path(serial_vla_eval.__file__).read_text(encoding="utf-8")

    assert "prepare_campaign_runtime_isolation" not in source
    assert 'root / "_runtime_isolation"' not in source
    assert 'root / "_managed_vla"' not in source


def test_instance_deadline_contract_generates_all_absolute_deadlines_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        instance_started_monotonic_ns=None,
        action_deadline_monotonic_ns=None,
        cleanup_deadline_monotonic_ns=None,
        hard_deadline_monotonic_ns=None,
    )
    monkeypatch.setattr(serial_vla_eval.time, "monotonic_ns", lambda: 7_000_000_000)

    serial_vla_eval._bind_instance_deadline_contract(args, allow_generate=True)

    assert args.instance_started_monotonic_ns == 7_000_000_000
    assert args.action_deadline_monotonic_ns == (7_000_000_000 + 6900 * 1_000_000_000)
    assert args.cleanup_deadline_monotonic_ns == (7_000_000_000 + 7080 * 1_000_000_000)
    assert args.hard_deadline_monotonic_ns == (7_000_000_000 + 7200 * 1_000_000_000)


def test_instance_deadline_contract_rejects_partial_absolute_binding() -> None:
    args = SimpleNamespace(
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        instance_started_monotonic_ns=7_000_000_000,
        action_deadline_monotonic_ns=None,
        cleanup_deadline_monotonic_ns=None,
        hard_deadline_monotonic_ns=None,
    )

    with pytest.raises(SystemExit, match="must be provided together"):
        serial_vla_eval._bind_instance_deadline_contract(args, allow_generate=False)


@pytest.mark.parametrize(
    ("field", "offset_s"),
    (
        ("action_deadline_monotonic_ns", 6899),
        ("cleanup_deadline_monotonic_ns", 7081),
        ("hard_deadline_monotonic_ns", 7199),
    ),
)
def test_instance_deadline_contract_rejects_relative_absolute_mismatch(
    field: str,
    offset_s: int,
) -> None:
    started = 7_000_000_000
    values = {
        "action_deadline_monotonic_ns": started + 6900 * 1_000_000_000,
        "cleanup_deadline_monotonic_ns": started + 7080 * 1_000_000_000,
        "hard_deadline_monotonic_ns": started + 7200 * 1_000_000_000,
    }
    values[field] = started + offset_s * 1_000_000_000
    args = SimpleNamespace(
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        instance_started_monotonic_ns=started,
        **values,
    )

    with pytest.raises(SystemExit, match="do not match relative timeout"):
        serial_vla_eval._bind_instance_deadline_contract(args, allow_generate=False)


def test_pure_vla_plan_uses_exact_task_local_s10_s19_mapping(tmp_path: Path) -> None:
    entries = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=None,
        output_root=tmp_path,
    )

    assert [entry.public_seed for entry in entries] == list(range(10, 20))
    assert [entry.activity_instance_id for entry in entries] == [
        108,
        152,
        84,
        198,
        199,
        100,
        111,
        151,
        130,
        168,
    ]
    assert {entry.cuda_device for entry in entries} == {"6"}
    assert len({entry.output_dir for entry in entries}) == 10


def test_pure_vla_plan_rejects_wrong_gpu_or_explore_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GPU6"):
        build_vla_eval_plan(
            task_name="picking_up_trash",
            public_seeds=(10,),
            output_root=tmp_path,
            cuda_device="7",
        )
    with pytest.raises(ValueError, match="does not allow s9 in eval"):
        build_vla_eval_plan(
            task_name="picking_up_trash",
            public_seeds=(9,),
            output_root=tmp_path,
        )


def test_pure_vla_repeats_without_tool_limit_and_finishes_success_window(
    tmp_path: Path,
) -> None:
    spec = get_task_spec("picking_up_trash")
    successful_call = 9
    primitives = _FakePrimitives(success_on_call=successful_call)
    trace = tmp_path / "tool_trace.jsonl"

    result = run_vla_window_loop(
        primitives,
        instruction=spec.task_language,
        deadline_monotonic=10_000.0,
        trace_path=trace,
        clock=lambda: 0.0,
    )

    assert result.task_success is True
    assert result.stopped_reason == "official_task_success"
    assert result.tool_calls == successful_call
    assert result.vla_chunks == successful_call * CHUNKS_PER_CALL
    assert result.total_env_steps == successful_call * ACTION_STEPS_PER_CALL
    assert (
        primitives.calls
        == [{"instruction": spec.task_language, "chunks": CHUNKS_PER_CALL}]
        * successful_call
    )
    assert validate_pure_vla_tool_trace(trace) == {
        "tool_calls": successful_call,
        "vla_chunks": successful_call * CHUNKS_PER_CALL,
    }
    final_record = json.loads(trace.read_text(encoding="utf-8").splitlines()[-1])
    assert final_record["result"]["official_success_source"] == (
        "behavior_action_trace"
    )
    assert final_record["result"]["official_success_field_path"] == (
        "info_done.success"
    )
    assert final_record["result"]["official_success_receipt"]["source"] == (
        "behavior_action_trace"
    )
    assert final_record["result"]["official_success_receipt"]["field_path"] == (
        "info_done.success"
    )
    assert (
        "official_success_source"
        not in final_record["result"]["official_success_receipt"]
    )


def test_pure_vla_deadline_is_non_success_timeout_without_tool_call(
    tmp_path: Path,
) -> None:
    primitives = _FakePrimitives()

    result = run_vla_window_loop(
        primitives,
        instruction=get_task_spec("picking_up_trash").task_language,
        deadline_monotonic=299.0,
        trace_path=tmp_path / "tool_trace.jsonl",
        clock=lambda: 0.0,
    )

    assert result.task_success is False
    assert result.timed_out is True
    assert result.stopped_reason == "action_deadline_exhausted"
    assert result.infrastructure_error is None
    assert result.tool_calls == 0


def test_pure_vla_infrastructure_error_remains_unknown_not_timeout(
    tmp_path: Path,
) -> None:
    primitives = _FakePrimitives(fail_on_call=1)

    result = run_vla_window_loop(
        primitives,
        instruction=get_task_spec("picking_up_trash").task_language,
        deadline_monotonic=10_000.0,
        trace_path=tmp_path / "tool_trace.jsonl",
        clock=lambda: 0.0,
    )

    assert result.task_success is False
    assert result.timed_out is False
    assert result.stopped_reason == "infrastructure_error"
    assert "injected VLA transport failure" in str(result.infrastructure_error)


def test_pure_vla_trace_rejects_any_non_pi0_tool(tmp_path: Path) -> None:
    trace = tmp_path / "tool_trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "step": 1,
                "tool": "observe",
                "arguments": {"camera": "head"},
                "result": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-Pi0 tool"):
        validate_pure_vla_tool_trace(trace)


def test_pure_vla_action_trace_rejects_malformed_json_or_symlink(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"step":0,"info_done":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid action trace JSON"):
        serial_vla_eval._raw_success_from_action_trace(malformed)

    target = tmp_path / "outside.jsonl"
    target.write_text(
        '{"step":0,"info_done":{"success":true}}\n',
        encoding="utf-8",
    )
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        serial_vla_eval._raw_success_from_action_trace(linked)


def test_pure_vla_action_trace_requires_expected_run_nonce_binding(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "action.jsonl"
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

    receipt = serial_vla_eval._raw_success_from_action_trace(
        trace,
        expected_run_nonce=expected,
    )
    assert receipt is not None
    assert receipt["run_nonce"] == expected
    with pytest.raises(ValueError, match="differs from supervisor"):
        serial_vla_eval._raw_success_from_action_trace(
            trace,
            expected_run_nonce="f" * 32,
        )


def test_pure_vla_module_has_no_planner_prompt_or_recipe_control_imports() -> None:
    source_path = Path(serial_vla_eval.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    assert not any(name.startswith("rpent.planner") for name in imported)
    assert "robots.behavior.prompt_bundle" not in imported
    assert "robots.behavior.recipe_catalog" not in imported
    assert "robots.behavior.memory_snapshot" not in imported


def test_pure_vla_result_artifacts_never_publish_recipe_or_memory(
    tmp_path: Path,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    (entry.output_dir / "behavior_action_trace.jsonl").write_text(
        '{"event":"step","step":0,"info_done":{"success":false}}\n',
        encoding="utf-8",
    )
    source = SourceSnapshot(
        root=tmp_path / "source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )
    result = WindowLoopResult(
        task_success=False,
        official_success_receipt=None,
        tool_calls=0,
        vla_chunks=0,
        total_env_steps=0,
        stopped_reason="action_deadline_exhausted",
        timed_out=True,
        infrastructure_error=None,
        last_result=None,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "c" * 64},
        state_sha256="d" * 64,
        result=result,
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="complete",
    )

    assert payload["task_success"] is False
    assert payload["timed_out"] is True
    assert payload["publication_complete"] is False
    assert payload["publication_eligible"] is False
    assert payload["eval_published_recipe"] is False
    assert payload["eval_published_memory"] is False
    assert payload["official_success_source"] == "behavior_action_trace"
    assert payload["official_success_field_path"] == "info_done.success"
    assert payload["instance_state_sha256"] == "d" * 64
    assert not tuple(entry.output_dir.glob("recipe_*.jsonl"))
    assert not (entry.output_dir / "memory").exists()


def test_pure_vla_result_keeps_runtime_binding_and_cleanup_without_cache_copy(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=output_root,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    (entry.output_dir / "behavior_action_trace.jsonl").write_text(
        '{"event":"step","step":0,"info_done":{"success":false}}\n',
        encoding="utf-8",
    )
    runtime_root = tmp_path / "external-runtime"
    cache_file = runtime_root / "xdg" / "cache" / "cache.bin"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"runtime-cache-must-stay-external")
    runtime_receipt = {
        "kind": "behavior_campaign_runtime_isolation",
        "root": str(runtime_root),
        "binding_sha256": "b" * 64,
    }
    source = SourceSnapshot(
        root=tmp_path / "source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="c" * 64,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "d" * 64},
        runtime_isolation=SimpleNamespace(as_dict=lambda: runtime_receipt),
        state_sha256="e" * 64,
        result=WindowLoopResult(
            task_success=False,
            official_success_receipt=None,
            tool_calls=0,
            vla_chunks=0,
            total_env_steps=0,
            stopped_reason="action_deadline_exhausted",
            timed_out=True,
            infrastructure_error=None,
            last_result=None,
        ),
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="complete",
    )

    assert payload["runtime_isolation"] == runtime_receipt
    assert payload["runtime_cleanup"] == "complete"
    assert cache_file.read_bytes() == b"runtime-cache-must-stay-external"
    assert not (entry.output_dir / "_runtime_isolation").exists()
    assert not tuple(entry.output_dir.rglob("cache.bin"))


def test_pure_vla_cleanup_failure_is_infrastructure_unknown(
    tmp_path: Path,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    (entry.output_dir / "behavior_action_trace.jsonl").write_text(
        '{"event":"step","step":0,"info_done":{"success":false}}\n',
        encoding="utf-8",
    )
    source = SourceSnapshot(
        root=tmp_path / "source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "c" * 64},
        state_sha256="d" * 64,
        result=WindowLoopResult(
            task_success=False,
            official_success_receipt=None,
            tool_calls=0,
            vla_chunks=0,
            total_env_steps=0,
            stopped_reason="instance_timeout",
            timed_out=True,
            infrastructure_error=None,
            last_result=None,
        ),
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="failed: env survived",
    )

    assert payload["task_success"] is None
    assert payload["success"] is None
    assert payload["infrastructure_error"] == (
        "runtime cleanup incomplete: failed: env survived"
    )


def test_pure_vla_cleanup_failure_does_not_revoke_raw_success(
    tmp_path: Path,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    (entry.output_dir / "behavior_action_trace.jsonl").write_text(
        '{"event":"step","step":0,"info_done":{"success":true}}\n',
        encoding="utf-8",
    )
    source = SourceSnapshot(
        root=tmp_path / "source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "c" * 64},
        state_sha256="d" * 64,
        result=WindowLoopResult(
            task_success=True,
            official_success_receipt=None,
            tool_calls=0,
            vla_chunks=0,
            total_env_steps=1,
            stopped_reason="official_task_success",
            timed_out=False,
            infrastructure_error=None,
            last_result=None,
        ),
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="failed: env survived",
    )

    assert payload["task_success"] is True
    assert payload["success"] is True
    assert "runtime cleanup incomplete" in payload["infrastructure_error"]


def test_pure_vla_instance_child_record_binds_session_source_and_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    source = SourceSnapshot(
        root=tmp_path / "sealed-source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )
    process = SimpleNamespace(pid=22003, returncode=None, poll=lambda: None)
    monkeypatch.setattr(
        serial_vla_eval,
        "_owned_process_payload",
        lambda _process: {
            "pid": 22003,
            "pgid": 22003,
            "sid": 22003,
            "start_ticks": 12345,
            "recorded_at": "ignored",
        },
    )
    monkeypatch.setattr(serial_vla_eval.os, "getpid", lambda: 22000)
    monkeypatch.setattr(serial_vla_eval.os, "getpgid", lambda _pid: 22000)
    monkeypatch.setattr(serial_vla_eval.os, "getsid", lambda _pid: 22000)
    monkeypatch.setattr(serial_vla_eval, "_process_start_ticks", lambda _pid: 67890)
    monkeypatch.setattr(
        serial_vla_eval,
        "_utc_now",
        lambda: "2026-07-26T00:00:00.000Z",
    )
    monkeypatch.setattr(serial_vla_eval.time, "monotonic_ns", lambda: 9_000_000_000)
    argv = ("python", "-m", "robots.behavior.serial_vla_eval")
    running = serial_vla_eval._running_instance_child_process_record(
        process=process,
        argv=argv,
        entry=entry,
        source=source,
        args=SimpleNamespace(
            expected_run_nonce="e" * 32,
            action_deadline_s=6900,
            cleanup_deadline_s=7080,
            instance_timeout_s=7200,
            instance_started_monotonic_ns=9_000_000_000,
            action_deadline_monotonic_ns=(9_000_000_000 + 6900 * 1_000_000_000),
            cleanup_deadline_monotonic_ns=(9_000_000_000 + 7080 * 1_000_000_000),
            hard_deadline_monotonic_ns=(9_000_000_000 + 7200 * 1_000_000_000),
        ),
    )

    assert running["state"] == "running"
    assert running["pid"] == running["pgid"] == running["sid"] == 22003
    assert running["start_ticks"] == 12345
    assert running["runner_pid"] == 22000
    assert running["runner_start_ticks"] == 67890
    assert running["source_snapshot_root"] == str(source.root.absolute())
    assert running["source_snapshot_binding_sha256"] == "a" * 64
    assert running["argv_sha256"] == serial_vla_eval._argv_sha256(argv)
    assert (
        running["action_deadline_s"],
        running["cleanup_deadline_s"],
        running["instance_timeout_s"],
    ) == (6900, 7080, 7200)
    assert running["started_monotonic_ns"] == 9_000_000_000
    assert running["action_deadline_monotonic_ns"] == (
        9_000_000_000 + 6900 * 1_000_000_000
    )
    assert running["cleanup_deadline_monotonic_ns"] == (
        9_000_000_000 + 7080 * 1_000_000_000
    )
    assert running["hard_deadline_monotonic_ns"] == (
        9_000_000_000 + 7200 * 1_000_000_000
    )

    process.returncode = 0
    process.poll = lambda: 0
    finished = serial_vla_eval._finished_instance_child_process_record(
        running,
        process=process,
        timed_out=False,
    )
    assert finished["state"] == "exited"
    assert finished["returncode"] == 0
    assert finished["timed_out"] is False


def test_pure_vla_instance_child_record_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    record = tmp_path / "instance_child_process.json"
    record.symlink_to(target)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        serial_vla_eval._write_instance_child_process_record(
            record,
            {"state": "running"},
        )


def test_pure_vla_child_argv_forwards_absolute_deadlines_unchanged(
    tmp_path: Path,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    absolute_deadlines = {
        "instance_started_monotonic_ns": 7_000_000_001,
        "action_deadline_monotonic_ns": 6_907_000_000_001,
        "cleanup_deadline_monotonic_ns": 7_087_000_000_001,
        "hard_deadline_monotonic_ns": 7_207_000_000_001,
    }
    args = SimpleNamespace(
        expected_run_nonce="e" * 32,
        python=sys.executable,
        behavior_repo=tmp_path / "behavior-repo",
        behavior_python=sys.executable,
        policy_checkpoint=tmp_path / "checkpoint",
        chunks_per_call=CHUNKS_PER_CALL,
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        source_snapshot_root=tmp_path / "sealed-source",
        source_snapshot_binding_sha256="a" * 64,
        runtime_isolation_root=tmp_path / "runtime",
        runtime_isolation_binding_sha256="b" * 64,
        **absolute_deadlines,
    )

    argv = serial_vla_eval._child_argv(
        args=args,
        entry=entry,
        vla_endpoint="http://127.0.0.1:9123",
    )

    for flag, field in (
        ("--instance-started-monotonic-ns", "instance_started_monotonic_ns"),
        ("--action-deadline-monotonic-ns", "action_deadline_monotonic_ns"),
        ("--cleanup-deadline-monotonic-ns", "cleanup_deadline_monotonic_ns"),
        ("--hard-deadline-monotonic-ns", "hard_deadline_monotonic_ns"),
    ):
        index = argv.index(flag)
        assert argv[index + 1] == str(absolute_deadlines[field])
    chunks_index = argv.index("--chunks-per-call")
    assert argv[chunks_index + 1] == str(CHUNKS_PER_CALL)


def test_pure_vla_preserves_lexical_venv_python_symlink_in_child_argv(
    tmp_path: Path,
) -> None:
    real_python = tmp_path / "real-python"
    real_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    real_python.chmod(0o755)
    venv_python = tmp_path / ".venv-behavior" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    validated = serial_vla_eval._validated_lexical_executable_path(
        venv_python,
        label="behavior_python",
    )
    assert validated == venv_python.absolute()
    assert validated.is_symlink()

    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path / "output",
    )[0]
    args = SimpleNamespace(
        expected_run_nonce="e" * 32,
        python=validated,
        behavior_repo=tmp_path / "behavior-repo",
        behavior_python=validated,
        policy_checkpoint=tmp_path / "checkpoint",
        chunks_per_call=CHUNKS_PER_CALL,
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        instance_started_monotonic_ns=7_000_000_001,
        action_deadline_monotonic_ns=6_907_000_000_001,
        cleanup_deadline_monotonic_ns=7_087_000_000_001,
        hard_deadline_monotonic_ns=7_207_000_000_001,
        source_snapshot_root=tmp_path / "sealed-source",
        source_snapshot_binding_sha256="a" * 64,
        runtime_isolation_root=tmp_path / "runtime",
        runtime_isolation_binding_sha256="b" * 64,
    )
    argv = serial_vla_eval._child_argv(
        args=args,
        entry=entry,
        vla_endpoint="http://127.0.0.1:9123",
    )

    assert argv[0] == str(venv_python.absolute())
    behavior_python_index = argv.index("--behavior-python")
    assert argv[behavior_python_index + 1] == str(venv_python.absolute())


@pytest.mark.parametrize("kind", ("missing", "directory", "non_executable"))
def test_pure_vla_rejects_invalid_lexical_executable(
    tmp_path: Path,
    kind: str,
) -> None:
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()
    elif kind == "non_executable":
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o644)

    with pytest.raises(ValueError, match="must be an existing executable|executable"):
        serial_vla_eval._validated_lexical_executable_path(
            path,
            label="behavior_python",
        )


def test_instance_child_identity_requires_exact_process_quadruple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=22003, poll=lambda: None)
    identity = {"pid": 22003, "pgid": 22003, "sid": 22003, "start_ticks": 12345}
    monkeypatch.setattr(serial_vla_eval.os, "getpgid", lambda _pid: 22003)
    monkeypatch.setattr(serial_vla_eval.os, "getsid", lambda _pid: 22003)
    monkeypatch.setattr(serial_vla_eval.os, "getpgrp", lambda: 21999)
    monkeypatch.setattr(serial_vla_eval, "_process_start_ticks", lambda _pid: 12345)

    assert serial_vla_eval._instance_child_identity_matches(process, identity) is True
    assert (
        serial_vla_eval._instance_child_identity_matches(
            process,
            {**identity, "start_ticks": 12346},
        )
        is False
    )


def test_terminate_process_group_identity_mismatch_has_zero_signal_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace(pid=22003, poll=lambda: None)
    identity = {"pid": 22003, "pgid": 22003, "sid": 22003, "start_ticks": 12345}
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        serial_vla_eval,
        "_instance_child_identity_matches",
        lambda _process, _identity: False,
    )
    monkeypatch.setattr(
        serial_vla_eval.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    assert (
        serial_vla_eval._terminate_process_group(
            process,
            expected_identity=identity,
        )
        is False
    )
    assert signals == []


def test_terminate_process_group_revalidates_identity_before_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        pid = 22003

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def wait(*, timeout: float) -> None:
            del timeout
            raise subprocess.TimeoutExpired(cmd="child", timeout=0)

    identity = {"pid": 22003, "pgid": 22003, "sid": 22003, "start_ticks": 12345}
    identity_checks = iter((True, False))
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        serial_vla_eval,
        "_instance_child_identity_matches",
        lambda _process, _identity: next(identity_checks),
    )
    monkeypatch.setattr(
        serial_vla_eval.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    assert (
        serial_vla_eval._terminate_process_group(
            _Process(),
            expected_identity=identity,
            timeout_s=0.0,
        )
        is False
    )
    assert signals == [(22003, signal.SIGTERM)]


def test_recorded_env_cleanup_missing_identity_fails_closed(tmp_path: Path) -> None:
    verified, forced, error = serial_vla_eval._kill_recorded_env(tmp_path)

    assert verified is False
    assert forced is False
    assert "identity is unavailable" in str(error)


def test_recorded_env_cleanup_accepts_exact_group_already_gone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "baseline_owned_processes.json").write_text(
        json.dumps(
            {
                "env": {
                    "pid": 22003,
                    "pgid": 22003,
                    "sid": 22003,
                    "start_ticks": 12345,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        serial_vla_eval.os,
        "getpgid",
        lambda _pid: (_ for _ in ()).throw(ProcessLookupError()),
    )
    monkeypatch.setattr(
        serial_vla_eval.os,
        "killpg",
        lambda _pgid, _sig: (_ for _ in ()).throw(ProcessLookupError()),
    )

    assert serial_vla_eval._kill_recorded_env(tmp_path) == (True, False, None)


def test_parent_result_amendment_updates_all_canonical_views(tmp_path: Path) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    payload = {
        "task_success": None,
        "infrastructure_error": "cleanup unverified",
    }

    serial_vla_eval._write_canonical_result_artifacts(entry, payload)

    assert (
        json.loads(
            (entry.output_dir / "baseline_result.json").read_text(encoding="utf-8")
        )
        == payload
    )
    assert (
        json.loads((entry.output_dir / "final_result.json").read_text(encoding="utf-8"))
        == payload
    )
    behavior = json.loads(
        (entry.output_dir / "behavior_result.json").read_text(encoding="utf-8")
    )
    assert behavior == {**payload, "recipe_path": None, "memory_path": None}


@pytest.mark.parametrize("trace_state", ("missing", "malformed"))
def test_pure_vla_invalid_action_trace_is_infrastructure_unknown(
    tmp_path: Path,
    trace_state: str,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    if trace_state == "malformed":
        (entry.output_dir / "behavior_action_trace.jsonl").write_text(
            '{"event":"step","step":0,"info_done":\n',
            encoding="utf-8",
        )
    source = SourceSnapshot(
        root=tmp_path / "sealed-source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )
    result = WindowLoopResult(
        task_success=False,
        official_success_receipt=None,
        tool_calls=0,
        vla_chunks=0,
        total_env_steps=0,
        stopped_reason="environment_terminated",
        timed_out=False,
        infrastructure_error=None,
        last_result=None,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "c" * 64},
        state_sha256="d" * 64,
        result=result,
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="complete",
    )

    assert payload["task_success"] is None
    assert payload["success"] is None
    assert "invalid official-success action trace" in payload["infrastructure_error"]
    assert payload["publication_complete"] is False
    assert payload["publication_eligible"] is False
    dashboard = serial_vla_eval._dashboard_terminal_state(payload)
    assert dashboard == {
        "outcome": "run_error",
        "task_success": None,
        "timed_out": False,
    }


@pytest.mark.parametrize("timed_out", (False, True))
def test_pure_vla_runtime_infrastructure_error_is_unknown_in_artifacts_and_dashboard(
    tmp_path: Path,
    timed_out: bool,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    (entry.output_dir / "behavior_action_trace.jsonl").write_text("", encoding="utf-8")
    source = SourceSnapshot(
        root=tmp_path / "sealed-source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )
    result = WindowLoopResult(
        task_success=False,
        official_success_receipt=None,
        tool_calls=1,
        vla_chunks=0,
        total_env_steps=0,
        stopped_reason="infrastructure_error",
        timed_out=timed_out,
        infrastructure_error="RpcError: injected re-arm failure",
        last_result=None,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "c" * 64},
        state_sha256="d" * 64,
        result=result,
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="complete",
    )

    assert payload["task_success"] is None
    assert payload["success"] is None
    assert payload["infrastructure_error"] == "RpcError: injected re-arm failure"
    assert serial_vla_eval._dashboard_terminal_state(payload) == {
        "outcome": "run_error",
        "task_success": None,
        "timed_out": timed_out,
    }


def test_pure_vla_dashboard_terminal_state_uses_trace_recovered_success(
    tmp_path: Path,
) -> None:
    entry = build_vla_eval_plan(
        task_name="picking_up_trash",
        public_seeds=(10,),
        output_root=tmp_path,
    )[0]
    entry.output_dir.mkdir(parents=True)
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    (entry.output_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps(
            {
                "event": "step",
                "step": 0,
                "info_done": {"success": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = SourceSnapshot(
        root=tmp_path / "sealed-source",
        binding={"kind": "hash_sealed_source_snapshot"},
        binding_sha256="a" * 64,
        tree_sha256="b" * 64,
    )
    loop_result = WindowLoopResult(
        task_success=False,
        official_success_receipt=None,
        tool_calls=0,
        vla_chunks=0,
        total_env_steps=1,
        stopped_reason="environment_terminated",
        timed_out=False,
        infrastructure_error="RpcError: transport failed after raw success",
        last_result=None,
    )

    payload = serial_vla_eval._write_result_artifacts(
        entry=entry,
        source=source,
        checkpoint_binding={"binding_sha256": "c" * 64},
        state_sha256="d" * 64,
        result=loop_result,
        started_at="2026-07-26T00:00:00.000Z",
        started_monotonic=0.0,
        runtime_cleanup="complete",
    )
    dashboard = serial_vla_eval._dashboard_terminal_state(payload)

    assert payload["task_success"] is True
    assert payload["official_success_source"] == "behavior_action_trace"
    assert payload["official_success_field_path"] == "info_done.success"
    assert payload["instance_state_sha256"] == "d" * 64
    assert payload["official_success_receipt"]["source"] == "behavior_action_trace"
    assert payload["official_success_receipt"]["field_path"] == "info_done.success"
    assert payload["infrastructure_error"] == (
        "RpcError: transport failed after raw success"
    )
    assert dashboard["task_success"] is True
    assert dashboard["outcome"] == "passed"
