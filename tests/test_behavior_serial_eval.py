from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from behavior_resource_fixtures import (
    FIXTURE_RESOURCES,
    fixture_resource_binding,
)

from robots.behavior import publication, serial_eval
from robots.behavior.run_manifest import pi0_nav_pick_exact_chunk_contract
from robots.behavior.schemas import (
    CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
    PUBLIC_TOOL_CONTRACTS,
)
from robots.behavior.serial_eval import (
    PICKING_UP_TRASH_PUBLIC_IDS,
    TURNING_ON_RADIO_PUBLIC_IDS,
    EvalEntry,
    _gpu_lock_path,
    _raw_success_env_steps,
    build_entry_argv,
    read_task_instances,
    read_turning_on_radio_instances,
    select_instances,
    validate_instance_result,
)
from robots.behavior.task_specs import get_task_spec


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_raw_success_precedes_eval_infrastructure_error() -> None:
    assert (
        serial_eval._resolve_eval_task_success(
            raw_success_confirmed=True,
            final_result={"task_success": False},
            infrastructure_error="instance state binding changed",
        )
        is True
    )
    assert (
        serial_eval._resolve_eval_task_success(
            raw_success_confirmed=False,
            final_result={"task_success": True},
            infrastructure_error="instance state binding changed",
        )
        is None
    )


def _entry(
    tmp_path: Path,
    *,
    public_seed: int | None = None,
    task_name: str = "turning_on_radio",
) -> EvalEntry:
    spec = get_task_spec(task_name)
    if public_seed is None:
        public_seed = spec.eval_public_seeds[0]
    instance_id = spec.instance_for_public_seed(public_seed, phase="eval")
    state_path = (
        tmp_path
        / spec.state_dir_name
        / f"{task_name}_{spec.activity_definition_id}_{instance_id}"
        "_template-tro_state.json"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")
    return EvalEntry(
        split_position=0,
        csv_position=0,
        activity_instance_id=instance_id,
        public_seed=public_seed,
        seed=0,
        output_dir=tmp_path / "run",
        argv=("python", "-m", "robots.behavior.cli"),
        checkpoint=tmp_path / "checkpoint",
        cuda_device="7",
        instance_state_path=state_path,
        instance_state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
        task_name=task_name,
    )


def _official_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "run_nonce": "eval-run-nonce",
        "attempt_nonce": "eval-attempt-nonce",
        "attempt_index": 1,
        "env_step": 42,
        "raw_done": {"success": True},
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    return receipt


def _write_bound_artifacts(
    entry: EvalEntry,
    *,
    success: bool,
    public_tool_contract_version: int = CURRENT_PUBLIC_TOOL_CONTRACT_VERSION,
) -> None:
    spec = get_task_spec(entry.task_name)
    recipe_tag = spec.tag(entry.public_seed)
    if public_tool_contract_version == 1:
        manifest_schema_version = 5
        public_tools = PUBLIC_TOOL_CONTRACTS[1]
    elif public_tool_contract_version == CURRENT_PUBLIC_TOOL_CONTRACT_VERSION:
        manifest_schema_version = 6
        public_tools = PUBLIC_TOOL_CONTRACTS[CURRENT_PUBLIC_TOOL_CONTRACT_VERSION]
    else:
        raise AssertionError("unsupported fixture public-tool contract")
    _write_json(
        entry.output_dir / "run_manifest.json",
        {
            "schema_version": manifest_schema_version,
            "protocol": {
                "behavior_phase": "eval",
                "public_seed": entry.public_seed,
                "recipe_tag": recipe_tag,
                "mapping_version": spec.mapping_version,
                "task_spec": serial_eval._task_spec_binding(spec),
                "task_identity": {
                    "task_name": spec.task_name,
                    "activity_definition_id": spec.activity_definition_id,
                    "activity_instance_id": entry.activity_instance_id,
                },
                **(
                    {
                        "public_tool_contract_version": (
                            CURRENT_PUBLIC_TOOL_CONTRACT_VERSION
                        )
                    }
                    if public_tool_contract_version
                    == CURRENT_PUBLIC_TOOL_CONTRACT_VERSION
                    else {}
                ),
                "public_primitives": list(public_tools),
                "agent_finish_registered": False,
                "pi0_nav_pick_contract": pi0_nav_pick_exact_chunk_contract(),
                "attempts": {
                    "initial_attempt_index": 1,
                    "max_attempts": 1,
                    "reset_registered": False,
                },
            },
            "commit": "abc",
            "worktree_dirty": False,
            "status": "stopped",
            "task": {
                "suite": "behavior_2025_challenge",
                "task": spec.task_index,
                "task_name": spec.task_name,
                "task_language": spec.task_language,
                "public_seed": entry.public_seed,
                "max_episode_steps": 24756,
            },
            "native_binding": {
                "activity_definition_id": spec.activity_definition_id,
                "activity_instance_id": entry.activity_instance_id,
                "activity_instance_dir": str(
                    entry.instance_state_path.parent.resolve()
                ),
                "scene_model": spec.scene_model,
                "env_seed": 0,
            },
            "checkpoint": str(entry.checkpoint.resolve()),
            "policy_checkpoint": entry.policy_checkpoint_binding,
            "gpu": "7",
            "processes": {
                "env": {"managed": True, "pid": None, "stopped_at": "done"},
                "vla": {"managed": True, "pid": None, "stopped_at": "done"},
            },
        },
    )
    _write_json(
        entry.output_dir / "final_result.json",
        {
            "run_status": "completed",
            "task_success": success,
            "official_success_source": 'info["done"]["success"]',
            "runtime_cleanup": "complete",
            "error": None,
        },
    )
    _write_json(
        entry.output_dir / "behavior_result.json",
        {
            "success": success,
            "task_success": success,
            "official_success_source": 'info["done"]["success"] via task_success',
            "global_vla_invocations": 0,
        },
    )
    _write_json(
        entry.output_dir / f"{recipe_tag}.json",
        {
            "schema_version": 1,
            "phase": "eval",
            "task_success": success,
            "global_vla_invocations": 0,
            "global_vla_chunks": 0,
            "total_env_steps": 0,
        },
    )
    result: dict[str, object] = {
        "primitive_success": True,
        "task_success": success,
        "official_success_source": 'info["done"]["success"]',
        "attempt_nonce": "eval-attempt-nonce",
        "run_nonce": "eval-run-nonce",
        "global_vla_invocations": 0,
        "global_vla_chunks": 0,
        "total_env_steps": 0,
    }
    if success:
        receipt = _official_receipt()
        receipt_path = entry.output_dir / "official_success_receipt.json"
        _write_json(receipt_path, receipt)
        result.update(
            {
                "official_success_receipt": receipt,
                "official_success_receipt_path": str(receipt_path),
            }
        )
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "tool": "observe",
                "input": {"camera": "head"},
                "result": result,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (entry.output_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps(
            {
                "event": "step",
                "step": 41,
                "info_done": {"success": success},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_raw_success_uses_only_current_info_done_and_receipt_step_is_one_based() -> (
    None
):
    trace = b"\n".join(
        (
            json.dumps(
                {
                    "event": "step",
                    "step": 0,
                    "info": {"done": {"success": True}},
                }
            ).encode(),
            json.dumps(
                {
                    "event": "step",
                    "step": 1,
                    "info_done": {"success": False},
                }
            ).encode(),
        )
    )

    assert _raw_success_env_steps(trace) == ()

    canonical = (
        json.dumps(
            {
                "event": "step",
                "step": 0,
                "info_done": {"success": True},
            }
        ).encode()
        + b"\n"
    )
    assert _raw_success_env_steps(canonical) == (1,)


def test_raw_success_latches_first_true_and_later_false_does_not_revoke() -> None:
    trace = b"\n".join(
        (
            json.dumps(
                {
                    "event": "step",
                    "step": 4,
                    "info_done": {"success": False},
                }
            ).encode(),
            json.dumps(
                {
                    "event": "step",
                    "step": 5,
                    "info_done": {"success": True},
                }
            ).encode(),
            json.dumps(
                {
                    "event": "step",
                    "step": 6,
                    "info_done": {"success": False},
                }
            ).encode(),
        )
    )

    assert _raw_success_env_steps(trace) == (6,)
    summary = serial_eval.summarize_action_trace_success(trace)
    assert summary is not None
    assert summary["field_path"] == "info_done.success"
    assert summary["first_success_step"] == 5
    assert summary["success_count"] == 1
    assert summary["success_later_reverted"] is True
    assert summary["last_success_step"] == 5
    assert summary["last_trace_step"] == 6
    assert summary["final_trace_success"] is False


def test_raw_success_matches_real_zero_based_trace_and_ignores_legacy_decoy() -> None:
    trace = b"\n".join(
        (
            json.dumps(
                {
                    "event": "step",
                    "step": 13817,
                    "info": {"done": {"success": True}},
                    "info_done": {"success": False},
                }
            ).encode(),
            json.dumps(
                {
                    "event": "step",
                    "step": 13818,
                    "env_step": 13819,
                    "info_done": {"success": True},
                }
            ).encode(),
        )
    )

    assert _raw_success_env_steps(trace) == (13819,)
    summary = serial_eval.summarize_action_trace_success(trace)
    assert summary is not None
    assert summary["first_success_step"] == 13818
    assert summary["field_path"] == "info_done.success"


def test_agentic_action_trace_success_requires_exact_supervisor_nonce() -> None:
    expected = "e" * 32
    trace = (
        json.dumps(
            {
                "event": "rpent_run_binding",
                "run_nonce": expected,
                "attempt_index": 1,
            }
        )
        + "\n"
        + json.dumps({"event": "step", "step": 8, "info_done": {"success": True}})
        + "\n"
    ).encode()

    summary, error = serial_eval._bound_action_trace_success(
        trace,
        expected_run_nonce=expected,
    )

    assert error is None
    assert summary is not None
    assert summary["run_nonce"] == expected
    assert serial_eval._bound_action_trace_success(
        trace,
        expected_run_nonce="f" * 32,
    ) == (None, "action trace run nonce binding mismatch")


def test_instance_child_process_payload_binds_exact_session_and_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(tmp_path)
    source = tmp_path / "sealed-source"
    source.mkdir()
    child_pid = 22001
    runner_pid = serial_eval.os.getpid()

    def fake_proc_stat(pid: int):
        if pid == child_pid:
            return {
                "pid": child_pid,
                "state": "S",
                "ppid": runner_pid,
                "pgid": child_pid,
                "sid": child_pid,
                "start_ticks": 12345,
            }
        if pid == runner_pid:
            return {
                "pid": runner_pid,
                "state": "S",
                "ppid": 1,
                "pgid": runner_pid,
                "sid": runner_pid,
                "start_ticks": 67890,
            }
        return None

    monkeypatch.setattr(serial_eval, "_proc_stat", fake_proc_stat)
    deadline_binding = serial_eval.InstanceDeadlineBinding(
        started_monotonic_ns=9_000_000_000,
        action_deadline_monotonic_ns=9_000_000_000 + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=9_000_000_000 + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=9_000_000_000 + 7200 * 1_000_000_000,
    )
    payload = serial_eval._instance_child_process_payload(
        entry,
        SimpleNamespace(pid=child_pid),
        state="running",
        started_at="2026-07-26T00:00:00.000Z",
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
        source_snapshot_root=source,
        source_snapshot_binding_sha256="a" * 64,
        deadline_binding=deadline_binding,
    )

    expected_argv_sha256 = hashlib.sha256(
        json.dumps(
            list(entry.argv),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    assert payload["state"] == "running"
    assert payload["pid"] == payload["pgid"] == payload["sid"] == child_pid
    assert payload["start_ticks"] == 12345
    assert payload["runner_pid"] == runner_pid
    assert payload["argv_sha256"] == expected_argv_sha256
    assert payload["source_snapshot_root"] == str(source.resolve())
    assert payload["source_snapshot_binding_sha256"] == "a" * 64
    assert (
        payload["action_deadline_s"],
        payload["cleanup_deadline_s"],
        payload["instance_timeout_s"],
    ) == (6900, 7080, 7200)
    assert payload["started_monotonic_ns"] == 9_000_000_000
    assert payload["action_deadline_monotonic_ns"] == (
        9_000_000_000 + 6900 * 1_000_000_000
    )
    assert payload["cleanup_deadline_monotonic_ns"] == (
        9_000_000_000 + 7080 * 1_000_000_000
    )
    assert payload["hard_deadline_monotonic_ns"] == (
        9_000_000_000 + 7200 * 1_000_000_000
    )


def test_run_entry_writes_running_then_exited_child_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(tmp_path)
    source = tmp_path / "sealed-source"
    source.mkdir()
    child_pid = 22002
    runner_pid = serial_eval.os.getpid()
    popen_kwargs: dict[str, object] = {}

    class FakeProcess:
        pid = child_pid
        returncode: int | None = None

        def wait(self, *, timeout: float):
            assert timeout <= 7080
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    process = FakeProcess()

    def fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        assert args[0] == entry.argv
        return process

    def fake_proc_stat(pid: int):
        selected = child_pid if pid == child_pid else runner_pid
        return {
            "pid": selected,
            "state": "S",
            "ppid": runner_pid if selected == child_pid else 1,
            "pgid": selected,
            "sid": selected,
            "start_ticks": 100 if selected == child_pid else 200,
        }

    deadline_binding = serial_eval.InstanceDeadlineBinding(
        started_monotonic_ns=9_000_000_000,
        action_deadline_monotonic_ns=9_000_000_000 + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=9_000_000_000 + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=9_000_000_000 + 7200 * 1_000_000_000,
    )
    monkeypatch.setattr(serial_eval.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(serial_eval, "_proc_stat", fake_proc_stat)
    monkeypatch.setattr(
        serial_eval.time,
        "monotonic_ns",
        lambda: 10_000_000_000,
    )

    result = serial_eval._run_entry(
        entry,
        repo_root=tmp_path,
        log_stream=io.BytesIO(),
        timeout_s=7200,
        cleanup_deadline_s=7080,
        action_deadline_s=6900,
        source_snapshot_root=source,
        source_snapshot_binding_sha256="b" * 64,
        deadline_binding=deadline_binding,
        expected_run_nonce="e" * 32,
    )

    receipt = json.loads(
        (
            entry.output_dir.parent / serial_eval.INSTANCE_CHILD_PROCESS_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert result == (0, False)
    assert popen_kwargs["start_new_session"] is True
    assert popen_kwargs["env"]["RPENT_BEHAVIOR_EXPECTED_RUN_NONCE"] == "e" * 32
    assert receipt["state"] == "exited"
    assert receipt["pid"] == child_pid
    assert receipt["source_snapshot_binding_sha256"] == "b" * 64
    assert receipt["expected_run_nonce"] == "e" * 32


def test_instance_child_process_receipt_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "outside.json"
    target.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / serial_eval.INSTANCE_CHILD_PROCESS_FILENAME
    receipt.symlink_to(target)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        serial_eval._write_instance_child_process_receipt(
            receipt,
            {"state": "running"},
        )


def test_standalone_deadline_binding_samples_start_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples: list[int] = []

    def monotonic_ns() -> int:
        samples.append(7_000_000_000)
        return samples[-1]

    monkeypatch.setattr(serial_eval.time, "monotonic_ns", monotonic_ns)
    binding = serial_eval._new_instance_deadline_binding(
        action_deadline_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
    )

    assert samples == [7_000_000_000]
    assert binding == serial_eval.InstanceDeadlineBinding(
        started_monotonic_ns=7_000_000_000,
        action_deadline_monotonic_ns=7_000_000_000 + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=7_000_000_000 + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=7_000_000_000 + 7200 * 1_000_000_000,
    )


def test_external_deadline_binding_requires_exact_shared_start_deltas() -> None:
    args = SimpleNamespace(
        instance_started_monotonic_ns=5_000_000_000,
        action_deadline_monotonic_ns=5_000_000_000 + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=5_000_000_000 + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=5_000_000_000 + 7200 * 1_000_000_000,
        max_wall_clock_s=6900,
        cleanup_deadline_s=7080,
        instance_timeout_s=7200,
    )
    binding = serial_eval._external_instance_deadline_binding(args)
    assert binding is not None
    assert binding.started_monotonic_ns == 5_000_000_000

    args.hard_deadline_monotonic_ns += 1
    with pytest.raises(ValueError, match="must exactly match"):
        serial_eval._external_instance_deadline_binding(args)

    args.hard_deadline_monotonic_ns = None
    with pytest.raises(ValueError, match="all four"):
        serial_eval._external_instance_deadline_binding(args)


def test_nested_child_receives_only_remaining_action_budget(tmp_path: Path) -> None:
    entry = replace(
        _entry(tmp_path),
        argv=(
            "python",
            "-m",
            "robots.behavior.cli",
            "--max-wall-clock-s",
            "6900",
            "--planner-timeout-s",
            "6900",
        ),
    )
    started = 10_000_000_000
    binding = serial_eval.InstanceDeadlineBinding(
        started_monotonic_ns=started,
        action_deadline_monotonic_ns=started + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=started + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=started + 7200 * 1_000_000_000,
    )
    admitted_entry, admitted = serial_eval._admit_entry_action_budget(
        entry,
        deadline_binding=binding,
        configured_planner_timeout_s=6900,
        admitted_at_monotonic_ns=started + 11_500_000_000,
    )

    wall_index = admitted_entry.argv.index("--max-wall-clock-s")
    planner_index = admitted_entry.argv.index("--planner-timeout-s")
    assert admitted_entry.argv[wall_index + 1] == "6888"
    assert admitted_entry.argv[planner_index + 1] == "6888"
    assert admitted == {
        "admitted_at_monotonic_ns": started + 11_500_000_000,
        "planner_timeout_s": 6888,
        "max_wall_clock_s": 6888,
        "cleanup_remaining_s": 7068,
        "instance_timeout_remaining_s": 7188,
    }
    assert entry.argv[wall_index + 1] == "6900"


def test_top_process_revalidates_identity_before_term_and_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid = 22003
    process = SimpleNamespace(
        pid=child_pid,
        poll=lambda: None,
        wait=lambda **_kwargs: (_ for _ in ()).throw(
            serial_eval.subprocess.TimeoutExpired("child", 1)
        ),
    )
    identity = {
        "pid": child_pid,
        "pgid": child_pid,
        "sid": child_pid,
        "start_ticks": 123,
    }
    reads = 0

    def proc_stat(_pid: int):
        nonlocal reads
        reads += 1
        return {
            "pid": child_pid,
            "state": "S",
            "ppid": serial_eval.os.getpid(),
            "pgid": child_pid,
            "sid": child_pid,
            "start_ticks": 123 if reads == 1 else 124,
        }

    signals: list[tuple[int, object]] = []
    monkeypatch.setattr(serial_eval, "_proc_stat", proc_stat)
    monkeypatch.setattr(serial_eval.os, "getpgrp", lambda: 999)
    monkeypatch.setattr(
        serial_eval.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )
    monkeypatch.setattr(serial_eval.time, "monotonic", lambda: 1.0)

    serial_eval._terminate_top_process(
        process,
        timeout_s=1.0,
        identity=identity,
    )

    assert signals == [(child_pid, serial_eval.signal.SIGTERM)]


def test_manifest_term_revalidates_identity_immediately_before_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = {
        "managed": True,
        "pid": 22004,
        "pgid": 22004,
        "sid": 22004,
        "start_ticks": 123,
    }
    monkeypatch.setattr(
        serial_eval,
        "_read_json",
        lambda _path: {"processes": {"env_server": process}},
    )
    checks = iter(((22004,), ()))
    monkeypatch.setattr(
        serial_eval,
        "_owned_group_members",
        lambda _process: next(checks),
    )
    monkeypatch.setattr(serial_eval, "_manifest_owned_groups", lambda _path: {})
    signals: list[tuple[int, object]] = []
    monkeypatch.setattr(
        serial_eval.os,
        "killpg",
        lambda pgid, sig: signals.append((pgid, sig)),
    )

    assert serial_eval._terminate_manifest_processes(tmp_path) == {}
    assert signals == []


def test_verified_forced_cleanup_receipt_binds_unchanged_child_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    manifest = {
        "status": "running",
        "stopped_at": None,
        "processes": {
            "env": {
                "managed": True,
                "pid": 22004,
                "pgid": 22004,
                "sid": 22004,
                "start_ticks": 123,
                "started_at": "2026-07-26T00:00:00.000Z",
                "stopped_at": None,
            }
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    monkeypatch.setattr(serial_eval, "_manifest_owned_groups", lambda _path: {})
    monkeypatch.setattr(serial_eval, "_manifest_unverified_groups", lambda _path: {})

    receipt = serial_eval._write_verified_forced_cleanup_receipt(
        output_dir,
        forced_groups={"env": (22004,)},
    )

    assert receipt["status"] == "verified"
    assert receipt["forced_groups"] == {"env": [22004]}
    assert serial_eval._verified_forced_cleanup_receipt(output_dir, manifest) == receipt
    # The parent records its own proof; it does not forge child-authored fields.
    assert (
        json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))[
            "processes"
        ]["env"]["stopped_at"]
        is None
    )


def test_forced_cleanup_receipt_rejects_manifest_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "attempt"
    output_dir.mkdir()
    manifest = {
        "status": "running",
        "processes": {
            "env": {
                "managed": True,
                "pid": 22004,
                "pgid": 22004,
                "sid": 22004,
                "start_ticks": 123,
                "started_at": "2026-07-26T00:00:00.000Z",
            }
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    monkeypatch.setattr(serial_eval, "_manifest_owned_groups", lambda _path: {})
    monkeypatch.setattr(serial_eval, "_manifest_unverified_groups", lambda _path: {})
    serial_eval._write_verified_forced_cleanup_receipt(
        output_dir,
        forced_groups={"env": (22004,)},
    )
    manifest["status"] = "stopped"
    _write_json(output_dir / "run_manifest.json", manifest)

    assert serial_eval._verified_forced_cleanup_receipt(output_dir, manifest) is None


def test_verified_parent_cleanup_closes_crashed_child_manifest_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    manifest_path = entry.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest["stopped_at"] = None
    manifest["processes"]["env"].update(
        {
            "pid": 22004,
            "pgid": 22004,
            "sid": 22004,
            "start_ticks": 123,
            "started_at": "2026-07-26T00:00:00.000Z",
            "stopped_at": None,
        }
    )
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(serial_eval, "_manifest_owned_groups", lambda _path: {})
    monkeypatch.setattr(serial_eval, "_manifest_unverified_groups", lambda _path: {})
    monkeypatch.setattr(serial_eval, "_owned_group_members", lambda _process: ())
    monkeypatch.setattr(serial_eval, "_unverified_group_members", lambda _process: ())
    serial_eval._write_verified_forced_cleanup_receipt(
        entry.output_dir,
        forced_groups={"env": (22004,)},
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "task_failed"
    assert "manifest lifecycle did not stop cleanly" not in errors
    assert "managed env process lacks stopped_at" not in errors


def _write_v5_vla_call(
    entry: EvalEntry,
    *,
    requested_chunks: int = 2,
    chunks_used: int | None = None,
    task_success: bool = False,
    include_legacy_limit: bool = False,
    full_chunks_executed: int | None = None,
    partial_final_steps: int | None = None,
    terminal_reason: str | None = None,
) -> None:
    """Add one schema-v5 call whose exact request is artifact-bound."""

    spec = get_task_spec(entry.task_name)
    recipe_tag = spec.tag(entry.public_seed)
    call_dir = (
        entry.output_dir
        / "attempts"
        / recipe_tag
        / "attempt_001"
        / "vla_calls"
        / "call_001"
    )
    result_path = call_dir / "pi0_nav_pick_result.json"
    if partial_final_steps is not None and not 1 <= partial_final_steps <= 31:
        raise ValueError("partial_final_steps must be in [1, 31]")
    if terminal_reason not in {None, "terminated", "truncated"}:
        raise ValueError("invalid terminal_reason")
    chunks_used = requested_chunks if chunks_used is None else chunks_used
    full_chunks = chunks_used if full_chunks_executed is None else full_chunks_executed
    if partial_final_steps is not None:
        full_chunks = chunks_used - 1
    vla_env_steps = (
        chunks_used * 32
        if partial_final_steps is None
        else full_chunks * 32 + partial_final_steps
    )
    terminated = terminal_reason == "terminated"
    truncated = terminal_reason == "truncated"
    exact_requested_chunks_completed = bool(
        partial_final_steps is None
        and chunks_used == requested_chunks
        and full_chunks == requested_chunks
        and vla_env_steps == requested_chunks * 32
    )
    stop_reason = (
        "official_task_success"
        if task_success
        else terminal_reason or "requested_chunks_completed"
    )
    result = {
        "name": "pi0_nav_pick",
        "primitive_success": True,
        "task_success": task_success,
        "official_success_source": 'info["done"]["success"]',
        "terminated": terminated,
        "truncated": truncated,
        "stop_reason": stop_reason,
        "requested_chunks": requested_chunks,
        "exact_requested_chunks_completed": exact_requested_chunks_completed,
        "chunks_used": chunks_used,
        "global_vla_chunks": chunks_used,
        "global_vla_invocations": 1,
        "full_chunks_executed": full_chunks,
        "env_steps_used": vla_env_steps,
        "vla_env_steps_used": vla_env_steps,
        "handoff_env_steps_used": 0,
        "total_env_steps": vla_env_steps,
        "action_horizon": 32,
        "required_action_shape": [32, 23],
        "attempt_index": 1,
        "attempt_nonce": "eval-attempt-nonce",
        "run_nonce": "eval-run-nonce",
    }
    if task_success:
        result["official_success_receipt"] = _official_receipt()
    _write_json(result_path, result)
    call_record = {
        "schema_version": 5,
        "name": "pi0_nav_pick",
        "request_id": "d" * 64,
        "status": "completed",
        "instruction": spec.task_language,
        "requested_chunks": requested_chunks,
        "run_nonce": "eval-run-nonce",
        "attempt_nonce": "eval-attempt-nonce",
        "attempt_index": 1,
        "global_vla_invocations": 1,
        "claimed_at_unix_s": 1000.0,
        "completed_at_unix_s": 1001.0,
        "outcome": stop_reason,
        "task_success": task_success,
        "exact_requested_chunks_completed": exact_requested_chunks_completed,
        "local_grasp_success": True,
        "result_path": str(result_path),
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    if include_legacy_limit:
        call_record["max_chunks"] = chunks_used
    _write_json(call_dir / "pi0_nav_pick_call.json", call_record)

    trace_result = {
        **result,
        "attempt_nonce": "eval-attempt-nonce",
        "run_nonce": "eval-run-nonce",
    }
    (entry.output_dir / "behavior_tool_trace.jsonl").write_text(
        json.dumps(
            {
                "step": 1,
                "tool": "pi0_nav_pick",
                "input": {
                    "instruction": spec.task_language,
                    "chunks": requested_chunks,
                },
                "result": trace_result,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    behavior_result_path = entry.output_dir / "behavior_result.json"
    behavior_result = json.loads(behavior_result_path.read_text(encoding="utf-8"))
    behavior_result["global_vla_invocations"] = 1
    _write_json(behavior_result_path, behavior_result)
    audit_path = entry.output_dir / f"{recipe_tag}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "global_vla_invocations": 1,
            "global_vla_chunks": chunks_used,
            "total_env_steps": vla_env_steps,
        }
    )
    _write_json(audit_path, audit)


def test_csv_protocol_preserves_authoritative_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_instances.csv"
    csv_path.write_text(
        "Task ID,Task,Public Test Instance IDs\n"
        '0,turning_on_radio,"'
        + ", ".join(str(value) for value in TURNING_ON_RADIO_PUBLIC_IDS)
        + '"\n',
        encoding="utf-8",
    )
    expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    public = read_turning_on_radio_instances(csv_path, expected_sha256=expected_hash)
    assert public == TURNING_ON_RADIO_PUBLIC_IDS
    assert select_instances(public, "official_first10") == public[:10]
    assert select_instances(public, "holdback_last10") == public[10:]


def test_trash_csv_protocol_and_task_local_public_mapping(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_instances.csv"
    csv_path.write_text(
        "Task ID,Task,Public Test Instance IDs\n"
        '1,picking_up_trash,"'
        + ", ".join(str(value) for value in PICKING_UP_TRASH_PUBLIC_IDS)
        + '"\n',
        encoding="utf-8",
    )
    expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    public = read_task_instances(
        csv_path,
        task_name="picking_up_trash",
        expected_sha256=expected_hash,
    )

    assert public == PICKING_UP_TRASH_PUBLIC_IDS
    assert public[0] == 196
    assert 242 not in public


def test_entry_argv_is_one_fresh_eval_without_retry_or_reset(tmp_path: Path) -> None:
    binding_path = tmp_path / "policy_checkpoint_binding.json"
    argv = build_entry_argv(
        python=Path("/python"),
        repo_root=tmp_path.resolve(),
        output_dir=tmp_path / "run",
        behavior_repo=tmp_path / "behavior",
        behavior_python=Path("/behavior-python"),
        checkpoint=tmp_path / "checkpoint",
        activity_instance_id=211,
        public_seed=6,
        cuda_device="7",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        max_turns=200,
        planner_timeout_s=6900,
        frozen_publication_root=tmp_path / "publication",
        frozen_provenance_sha256="a" * 64,
        reviewed_memory_snapshot_sha256="b" * 64,
        recipe_catalog_sha256="c" * 64,
        policy_checkpoint_binding_file=binding_path,
        vla_endpoint="http://127.0.0.1:9123",
    )
    assert argv[:3] == ("/python", "-m", "robots.behavior.cli")
    assert argv[argv.index("--public-seed") + 1] == "6"
    assert argv[argv.index("--seed") + 1] == "0"
    assert "--behavior-frozen-publication-root" in argv
    assert "--behavior-frozen-recipe" not in argv
    assert "--behavior-frozen-memory" not in argv
    assert "--behavior-frozen-provenance" not in argv
    assert (
        argv[argv.index("--behavior-reviewed-memory-snapshot-sha256") + 1] == "b" * 64
    )
    assert argv[argv.index("--behavior-recipe-catalog-sha256") + 1] == "c" * 64
    assert argv[argv.index("--behavior-phase") + 1] == "eval"
    assert argv[argv.index("--max-tool-calls") + 1] == "350"
    assert "--max-total-vla-chunks" not in argv
    assert "--max-vla-chunks-per-call" not in argv
    assert argv[argv.index("--behavior-policy-checkpoint-binding-file") + 1] == str(
        binding_path
    )
    assert argv[argv.index("--vla-endpoint") + 1] == "http://127.0.0.1:9123"
    assert "--reset" not in argv
    assert "--retry" not in argv


def test_trash_entry_argv_uses_task_spec_identity(tmp_path: Path) -> None:
    argv = build_entry_argv(
        python=Path("/python"),
        repo_root=tmp_path.resolve(),
        output_dir=tmp_path / "picking_up_trash_s10",
        behavior_repo=tmp_path / "behavior",
        behavior_python=Path("/behavior-python"),
        checkpoint=tmp_path / "checkpoint",
        activity_instance_id=108,
        public_seed=10,
        cuda_device="7",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        max_turns=200,
        planner_timeout_s=6900,
        frozen_publication_root=tmp_path / "publication",
        frozen_provenance_sha256="a" * 64,
        reviewed_memory_snapshot_sha256="b" * 64,
        recipe_catalog_sha256="c" * 64,
        task_name="picking_up_trash",
    )

    assert argv[argv.index("--task") + 1] == "1"
    assert argv[argv.index("--task-name") + 1] == "picking_up_trash"
    assert argv[argv.index("--activity-instance-id") + 1] == "108"
    assert argv[argv.index("--public-seed") + 1] == "10"
    assert argv[argv.index("--seed") + 1] == "0"
    assert "turning_on_radio" not in argv


def test_trash_eval_rejects_s9_at_explore_eval_boundary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not allow s9 in eval"):
        build_entry_argv(
            python=Path("/python"),
            repo_root=tmp_path.resolve(),
            output_dir=tmp_path / "picking_up_trash_s9",
            behavior_repo=tmp_path / "behavior",
            behavior_python=Path("/behavior-python"),
            checkpoint=tmp_path / "checkpoint",
            activity_instance_id=246,
            public_seed=9,
            cuda_device="7",
            model="gpt-5.5",
            reasoning_effort="xhigh",
            max_turns=200,
            planner_timeout_s=6900,
            frozen_publication_root=tmp_path / "publication",
            frozen_provenance_sha256="a" * 64,
            reviewed_memory_snapshot_sha256="b" * 64,
            recipe_catalog_sha256="c" * 64,
            task_name="picking_up_trash",
        )


def test_trash_eval_frozen_source_accepts_s1_explore_identity() -> None:
    spec = get_task_spec("picking_up_trash")
    identity = publication.resolve_publication_identity(
        task_name=spec.task_name,
        task_index=spec.task_index,
        public_seed=1,
    )
    frozen = SimpleNamespace(
        identity=identity,
        manifest_binding={
            "source_public_seed": 1,
            "source_tag": "picking_up_trash_s1",
        },
    )

    assert serial_eval._validate_frozen_source_identity(frozen, spec) == (
        1,
        "picking_up_trash_s1",
    )


@pytest.mark.parametrize(
    ("identity_seed", "manifest_seed", "manifest_tag", "message"),
    (
        (10, 10, "picking_up_trash_s10", "outside"),
        (1, 0, "picking_up_trash_s1", "inconsistent"),
        (1, 1, "picking_up_trash_s0", "inconsistent"),
    ),
)
def test_trash_eval_frozen_source_rejects_partition_or_manifest_identity_drift(
    identity_seed,
    manifest_seed,
    manifest_tag,
    message,
) -> None:
    spec = get_task_spec("picking_up_trash")
    identity = publication.resolve_publication_identity(
        task_name=spec.task_name,
        task_index=spec.task_index,
        public_seed=1,
    )
    if identity_seed != 1:
        identity = replace(
            identity,
            public_seed=identity_seed,
            native_instance=spec.public_seed_to_instance[identity_seed],
            tag=spec.tag(identity_seed),
            recipe_relative=f"recipe_{spec.tag(identity_seed)}.jsonl",
        )
    frozen = SimpleNamespace(
        identity=identity,
        manifest_binding={
            "source_public_seed": manifest_seed,
            "source_tag": manifest_tag,
        },
    )

    with pytest.raises(publication.PublicationValidationError, match=message):
        serial_eval._validate_frozen_source_identity(frozen, spec)


def test_entry_rejects_cross_task_instance_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity mismatch"):
        replace(
            _entry(tmp_path, task_name="picking_up_trash"),
            activity_instance_id=109,
        )


def test_formal_eval_recipe_selection_uses_only_formal_consumer(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    class _Catalog:
        catalog_sha256 = "c" * 64
        manifest_binding = {"manifest_sha256": "d" * 64}
        files = {"turning_on_radio/canonical/recipe.jsonl": {"sha256": "e" * 64}}

        def select(self, task_name, consumer):
            calls.append((task_name, consumer))
            return type(
                "_Selection",
                (),
                {
                    "public_binding": {
                        "consumer": consumer,
                        "selected_ids": ["canonical_control_access_v1"],
                        "selected_entries": [
                            {
                                "entry_id": "canonical_control_access_v1",
                                "provenance_class": "canonical_public_explore",
                            }
                        ],
                    },
                    "selected_ids": ("canonical_control_access_v1",),
                },
            )()

    monkeypatch.setattr(
        serial_eval,
        "load_behavior_recipe_catalog",
        lambda root: _Catalog(),
    )

    binding = serial_eval._reviewed_recipe_catalog_binding(tmp_path)

    assert calls == [("turning_on_radio", "formal_eval")]
    assert binding["selection"]["consumer"] == "formal_eval"
    assert binding["selected_ids"] == ["canonical_control_access_v1"]


def test_formal_eval_trash_recipe_selection_is_task_local(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    class _Catalog:
        catalog_sha256 = "c" * 64
        manifest_binding = {"manifest_sha256": "d" * 64}
        files = {}

        def select(self, task_name, consumer):
            calls.append((task_name, consumer))
            return type(
                "_Selection",
                (),
                {
                    "public_binding": {
                        "consumer": consumer,
                        "selected_ids": [],
                        "selected_entries": [],
                    },
                    "selected_ids": (),
                },
            )()

    monkeypatch.setattr(
        serial_eval,
        "load_behavior_recipe_catalog",
        lambda root: _Catalog(),
    )

    binding = serial_eval._reviewed_recipe_catalog_binding(
        tmp_path,
        task_name="picking_up_trash",
    )

    assert calls == [("picking_up_trash", "formal_eval")]
    assert binding["selected_ids"] == []


def test_formal_eval_rejects_candidate_recipe_selection(
    monkeypatch, tmp_path: Path
) -> None:
    class _Catalog:
        catalog_sha256 = "c" * 64
        manifest_binding = {"manifest_sha256": "d" * 64}
        files = {}

        def select(self, task_name, consumer):
            del task_name
            return type(
                "_Selection",
                (),
                {
                    "public_binding": {
                        "consumer": consumer,
                        "selected_entries": [
                            {
                                "entry_id": "reviewed_recovery_v1",
                                "provenance_class": "candidate_explore_reviewed",
                            }
                        ],
                    },
                    "selected_ids": ("reviewed_recovery_v1",),
                },
            )()

    monkeypatch.setattr(
        serial_eval,
        "load_behavior_recipe_catalog",
        lambda root: _Catalog(),
    )

    with pytest.raises(RuntimeError, match="non-canonical provenance"):
        serial_eval._reviewed_recipe_catalog_binding(tmp_path)


def test_result_rejects_child_input_binding_drift(tmp_path: Path) -> None:
    entry = replace(
        _entry(tmp_path),
        frozen_publication_binding={"bundle_id": "a" * 64},
        reviewed_repo_memory_binding={"snapshot_sha256": "b" * 64},
        reviewed_recipe_catalog_binding={
            "catalog_sha256": "c" * 64,
            "selection": {"consumer": "formal_eval"},
        },
    )
    _write_bound_artifacts(entry, success=True)

    outcome, errors, _result = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "manifest frozen publication binding mismatch" in errors
    assert "manifest reviewed Global Memory binding mismatch" in errors
    assert "manifest reviewed Recipe Catalog binding mismatch" in errors


def test_result_fails_closed_when_instance_state_sha256_changes(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    entry.instance_state_path.write_text('{"changed": true}', encoding="utf-8")

    outcome, errors, _result = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "instance state SHA-256 changed after admission" in errors


def test_eval_accepts_only_receipt_bound_raw_success(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "passed"
    assert errors == []

    (entry.output_dir / "official_success_receipt.json").unlink()
    trace_path = entry.output_dir / "behavior_tool_trace.jsonl"
    trace = json.loads(trace_path.read_text())
    trace["result"].pop("official_success_receipt")
    trace["result"].pop("official_success_receipt_path")
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "incomplete"
    assert "successful run lacks a valid raw official-success receipt" in errors


def test_eval_rejects_legacy_v1_manifest_as_current_frozen_input(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(
        entry,
        success=True,
        public_tool_contract_version=1,
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert any(
        "schema version mismatch" in error
        or "public-tool contract" in error
        or "public primitive surface" in error
        for error in errors
    )


def test_eval_false_raw_success_is_task_failed(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "task_failed"
    assert errors == []


def test_eval_accepts_unbounded_v5_call_with_exact_requested_chunk_totals(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(entry, requested_chunks=129)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "task_failed"
    assert errors == []


def test_eval_rejects_v3_call_artifact_fail_closed(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(entry)
    call_path = (
        entry.output_dir
        / "attempts"
        / get_task_spec(entry.task_name).tag(entry.public_seed)
        / "attempt_001"
        / "vla_calls"
        / "call_001"
        / "pi0_nav_pick_call.json"
    )
    call = json.loads(call_path.read_text(encoding="utf-8"))
    call["schema_version"] = 3
    _write_json(call_path, call)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "unsupported VLA call artifact schema_version 3: call_001" in errors


def test_eval_rejects_legacy_call_limit_in_v5_artifact(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(entry, include_legacy_limit=True)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "invalid VLA call record: call_001" in errors


def test_eval_rejects_incomplete_admitted_vla_chunk_accounting(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(
        entry,
        requested_chunks=2,
        chunks_used=2,
        full_chunks_executed=1,
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "invalid exact-N chunk accounting: call_001" in errors


@pytest.mark.parametrize("partial_final_steps", [1, 17, 31])
def test_eval_accepts_raw_success_in_partial_final_vla_chunk(
    tmp_path: Path,
    partial_final_steps: int,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    _write_v5_vla_call(
        entry,
        requested_chunks=2,
        task_success=True,
        chunks_used=2,
        full_chunks_executed=1,
        partial_final_steps=partial_final_steps,
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "passed"
    assert errors == []


def test_eval_accepts_raw_success_at_exact_final_action_boundary(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    _write_v5_vla_call(entry, requested_chunks=3, task_success=True)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "passed"
    assert errors == []


@pytest.mark.parametrize("terminal_reason", ["terminated", "truncated"])
@pytest.mark.parametrize("partial_final_steps", [1, 17, 31])
def test_eval_accepts_hard_terminal_partial_final_chunk(
    tmp_path: Path,
    terminal_reason: str,
    partial_final_steps: int,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(
        entry,
        requested_chunks=5,
        chunks_used=2,
        partial_final_steps=partial_final_steps,
        terminal_reason=terminal_reason,
    )

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "task_failed"
    assert errors == []


def test_eval_rejects_partial_final_vla_chunk_without_terminal_evidence(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(
        entry,
        requested_chunks=2,
        chunks_used=2,
        full_chunks_executed=1,
    )
    result_path = (
        entry.output_dir
        / "attempts"
        / get_task_spec(entry.task_name).tag(entry.public_seed)
        / "attempt_001"
        / "vla_calls"
        / "call_001"
        / "pi0_nav_pick_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["vla_env_steps_used"] = 33
    result["env_steps_used"] = 33
    result["total_env_steps"] = 33
    _write_json(result_path, result)
    call_path = result_path.with_name("pi0_nav_pick_call.json")
    call_record = json.loads(call_path.read_text(encoding="utf-8"))
    call_record["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    _write_json(call_path, call_record)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "invalid exact-N chunk accounting: call_001" in errors


@pytest.mark.parametrize("field", ["call", "result", "trace"])
def test_eval_rejects_requested_chunk_binding_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    _write_v5_vla_call(entry, requested_chunks=2)
    call_dir = (
        entry.output_dir
        / "attempts"
        / get_task_spec(entry.task_name).tag(entry.public_seed)
        / "attempt_001"
        / "vla_calls"
        / "call_001"
    )
    call_path = call_dir / "pi0_nav_pick_call.json"
    result_path = call_dir / "pi0_nav_pick_result.json"
    trace_path = entry.output_dir / "behavior_tool_trace.jsonl"
    if field == "call":
        call = json.loads(call_path.read_text(encoding="utf-8"))
        call["requested_chunks"] = 3
        _write_json(call_path, call)
    elif field == "result":
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["requested_chunks"] = 3
        _write_json(result_path, result)
        call = json.loads(call_path.read_text(encoding="utf-8"))
        call["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        _write_json(call_path, call)
    else:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["input"]["chunks"] = 3
        trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"


def test_trash_eval_result_uses_task_local_manifest_and_audit(tmp_path: Path) -> None:
    entry = _entry(tmp_path, task_name="picking_up_trash")
    _write_bound_artifacts(entry, success=False)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "task_failed"
    assert errors == []
    assert (entry.output_dir / "picking_up_trash_s10.json").is_file()
    assert not (entry.output_dir / "turning_on_radio_s10.json").exists()


def test_trash_eval_rejects_radio_manifest_identity(tmp_path: Path) -> None:
    entry = _entry(tmp_path, task_name="picking_up_trash")
    _write_bound_artifacts(entry, success=False)
    manifest_path = entry.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"]["task"] = 0
    manifest["task"]["task_name"] = "turning_on_radio"
    _write_json(manifest_path, manifest)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "manifest task binding mismatch: task" in errors
    assert "manifest task binding mismatch: task_name" in errors


def test_eval_rejects_bare_instance_identity_without_task_scope(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path, task_name="picking_up_trash")
    _write_bound_artifacts(entry, success=False)
    manifest_path = entry.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["task_identity"] = {
        "activity_definition_id": 0,
        "activity_instance_id": entry.activity_instance_id,
    }
    _write_json(manifest_path, manifest)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )

    assert outcome == "incomplete"
    assert "manifest composite task identity mismatch" in errors


@pytest.mark.parametrize(
    "legacy_name",
    [
        "finish",
        "reset",
        "inspect_post_pick_state",
        "post_pick_direct_finger_toggle",
        "post_success_hold_frames",
    ],
)
def test_eval_rejects_non_public_tools(tmp_path: Path, legacy_name: str) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=False)
    trace_path = entry.output_dir / "behavior_tool_trace.jsonl"
    trace = json.loads(trace_path.read_text())
    trace["tool"] = legacy_name
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "incomplete"
    assert any(
        "unregistered BEHAVIOR tool" in error or "contains reset" in error
        for error in errors
    )


def test_eval_never_publishes_recipe_or_memory(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    (entry.output_dir / "recipe_turning_on_radio_s1.jsonl").write_text(
        '{"kind":"semantic_goal"}\n', encoding="utf-8"
    )
    memory = entry.output_dir / "memory" / "turning_on_radio.md"
    memory.parent.mkdir()
    memory.write_text("frozen input only", encoding="utf-8")
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "incomplete"
    assert "formal Eval published a symbolic recipe" in errors
    assert "formal Eval published task memory" in errors


def test_visual_checkpoint_is_not_required_but_state_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    entry = _entry(tmp_path)
    _write_bound_artifacts(entry, success=True)
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "passed"
    assert errors == []

    forbidden = entry.output_dir / "state_checkpoints" / "state.json"
    _write_json(forbidden, {"qpos": []})
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit="abc",
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "incomplete"
    assert "forbidden simulator-state checkpoint artifacts are present" in errors


def test_gpu_lock_is_global_and_device_specific() -> None:
    assert _gpu_lock_path("7") == _gpu_lock_path("07")
    assert _gpu_lock_path("7") != _gpu_lock_path("6")


def test_external_gpu_lock_owner_leaves_only_output_lock_to_serial_runner(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "eval"
    output_lock = output_root.parent / ".eval.lock"
    assert serial_eval._eval_lock_paths(output_root, "7", True) == (output_lock,)
    assert serial_eval._eval_lock_paths(output_root, "7", False) == (
        _gpu_lock_path("7"),
        output_lock,
    )


def test_eval_cli_defaults_to_radio_and_fixed_shared_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "PI05_CHECKPOINT_PATH",
        "/home/ubuntu/lwb/Models/scilwb/pi05-turning_on_radio-sft-from-jax",
    )

    args = serial_eval._parse_args(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--cuda-device",
            "7",
            "--behavior-frozen-publication-root",
            str(tmp_path / "publication"),
            "--behavior-frozen-provenance-sha256",
            "a" * 64,
        ]
    )

    assert args.task_name == "turning_on_radio"
    assert args.public_seed is None
    assert args.policy_checkpoint == str(serial_eval.SHARED_POLICY_CHECKPOINT_PATH)
    assert args.instance_started_monotonic_ns is None
    assert args.action_deadline_monotonic_ns is None
    assert args.cleanup_deadline_monotonic_ns is None
    assert args.hard_deadline_monotonic_ns is None


def test_eval_cli_accepts_task_local_seed_subset(tmp_path: Path) -> None:
    args = serial_eval._parse_args(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--cuda-device",
            "7",
            "--task-name",
            "picking_up_trash",
            "--public-seed",
            "10",
            "--public-seed",
            "19",
            "--behavior-frozen-publication-root",
            str(tmp_path / "publication"),
            "--behavior-frozen-provenance-sha256",
            "a" * 64,
        ]
    )

    assert args.task_name == "picking_up_trash"
    assert args.public_seed == [10, 19]


def test_formal_eval_requires_external_runtime_before_any_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "not-created" / "eval-output"
    resource_cache = tmp_path / "not-created-cache"
    calls: list[str] = []
    monkeypatch.setattr(
        serial_eval,
        "prepare_pinned_dataset_resources",
        lambda *_args, **_kwargs: calls.append("resource-cache"),
    )
    monkeypatch.setattr(
        serial_eval.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("process"),
    )

    with pytest.raises(
        SystemExit,
        match="formal BEHAVIOR Eval requires --runtime-isolation-root",
    ):
        serial_eval.main(
            [
                "--output-root",
                str(output_root),
                "--cuda-device",
                "7",
                "--behavior-frozen-publication-root",
                str(tmp_path / "publication"),
                "--behavior-frozen-provenance-sha256",
                "a" * 64,
                "--behavior-resource-cache",
                str(resource_cache),
            ]
        )

    assert calls == []
    assert not output_root.parent.exists()
    assert not resource_cache.exists()


@pytest.mark.parametrize(
    "relationship",
    ("same", "runtime_parent", "output_parent"),
)
def test_formal_eval_rejects_runtime_output_path_overlap_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relationship: str,
) -> None:
    scope = tmp_path / "not-created"
    if relationship == "same":
        output_root = scope / "shared"
        runtime_root = output_root
    elif relationship == "runtime_parent":
        runtime_root = scope / "runtime"
        output_root = runtime_root / "eval-output"
    else:
        output_root = scope / "eval-output"
        runtime_root = output_root / "runtime"
    resource_cache = tmp_path / f"cache-{relationship}"
    calls: list[str] = []
    monkeypatch.setattr(
        serial_eval,
        "prepare_pinned_dataset_resources",
        lambda *_args, **_kwargs: calls.append("resource-cache"),
    )
    monkeypatch.setattr(
        serial_eval.subprocess,
        "Popen",
        lambda *_args, **_kwargs: calls.append("process"),
    )

    with pytest.raises(SystemExit, match="disjoint paths"):
        serial_eval.main(
            [
                "--output-root",
                str(output_root),
                "--cuda-device",
                "7",
                "--runtime-isolation-root",
                str(runtime_root),
                "--runtime-isolation-binding-sha256",
                "b" * 64,
                "--behavior-frozen-publication-root",
                str(tmp_path / "publication"),
                "--behavior-frozen-provenance-sha256",
                "a" * 64,
                "--behavior-resource-cache",
                str(resource_cache),
            ]
        )

    assert calls == []
    assert not scope.exists()
    assert not resource_cache.exists()


def test_eval_launcher_uses_local_resources_and_seals_plan_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "pyproject.toml").write_text("", encoding="utf-8")
    entry_python = tmp_path / "rpent-python"
    entry_python.write_text("", encoding="utf-8")
    behavior_repo = tmp_path / "behavior-repo"
    behavior_python = behavior_repo / ".venv-behavior" / "bin" / "python"
    behavior_python.parent.mkdir(parents=True)
    behavior_python.write_text("", encoding="utf-8")
    metadata_root = (
        behavior_repo
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
    )
    csv_path = metadata_root / "metadata" / "test_instances.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text(
        "Task ID,Task,Public Test Instance IDs\n"
        '1,picking_up_trash,"'
        + ", ".join(str(value) for value in PICKING_UP_TRASH_PUBLIC_IDS)
        + '"\n',
        encoding="utf-8",
    )
    state_dir = (
        metadata_root
        / "scenes"
        / "house_double_floor_lower"
        / "json"
        / get_task_spec("picking_up_trash").state_dir_name
    )
    state_dir.mkdir(parents=True)
    (state_dir / "scene_0_108_template-tro_state.json").write_text(
        "{}",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "model.safetensors").parent.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    norm_stats = (
        checkpoint
        / "assets"
        / "behavior-1k"
        / "2025-challenge-demos"
        / "norm_stats.json"
    )
    norm_stats.parent.mkdir(parents=True)
    norm_stats.write_text("{}", encoding="utf-8")
    frozen_root = tmp_path / "frozen-publication"
    frozen_root.mkdir()
    output_root = tmp_path / "eval-output"
    runtime_root = tmp_path / "runtime-isolation"
    runtime_root.mkdir()
    runtime_paths = {
        "omnigibson_appdata": runtime_root / "omnigibson_appdata",
        "xdg_cache": runtime_root / "xdg" / "cache",
        "xdg_config": runtime_root / "xdg" / "config",
        "xdg_data": runtime_root / "xdg" / "data",
        "ov_cache": runtime_root / "ov_cache",
        "omni_user": runtime_root / "omni_user",
        "isaac_root": runtime_root / "isaac",
        "experience": runtime_root / "isaac" / "apps",
        "tmp": runtime_root / "tmp",
        "endpoints": runtime_root / "endpoints",
        "logs": runtime_root / "logs",
    }
    for path in runtime_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    runtime_environment = {
        "OMNIGIBSON_APPDATA_PATH": str(runtime_paths["omnigibson_appdata"]),
        "XDG_CACHE_HOME": str(runtime_paths["xdg_cache"]),
        "XDG_CONFIG_HOME": str(runtime_paths["xdg_config"]),
        "XDG_DATA_HOME": str(runtime_paths["xdg_data"]),
        "OV_CACHE_DIR": str(runtime_paths["ov_cache"]),
        "OMNI_USER_FOLDER": str(runtime_paths["omni_user"]),
        "ISAAC_PATH": str(runtime_paths["isaac_root"]),
        "EXP_PATH": str(runtime_paths["experience"]),
        "TMPDIR": str(runtime_paths["tmp"]),
    }
    runtime_binding = {
        "schema_version": 1,
        "root": str(runtime_root.resolve()),
        "binding_sha256": "e" * 64,
        "namespace": "serial-eval-test",
        "cuda_device": "7",
        "paths": {name: str(path.resolve()) for name, path in runtime_paths.items()},
    }
    runtime_isolation = SimpleNamespace(
        root=runtime_root.resolve(),
        binding_sha256="e" * 64,
        namespace="serial-eval-test",
        cuda_device="7",
        payload=runtime_binding,
        as_dict=lambda: runtime_binding,
        environment=lambda: runtime_environment,
    )
    binding = fixture_resource_binding()
    frozen_identity = publication.resolve_publication_identity(
        task_name="picking_up_trash",
        task_index=1,
        public_seed=1,
    )
    frozen = SimpleNamespace(
        identity=frozen_identity,
        manifest_binding={
            "source_public_seed": 1,
            "source_tag": "picking_up_trash_s1",
        },
        files=(),
    )
    calls: dict[str, object] = {"publication": []}

    def fail_hf(*_args, **_kwargs):
        raise AssertionError("HuggingFace preparation must not run")

    def prepare_local(subtree, *, source_root, cache_root):
        calls["local_prepare"] = (subtree, source_root, cache_root)
        return binding

    def validate_publication(*_args, **kwargs):
        calls["publication"].append(dict(kwargs))
        return frozen

    class _VlaProcess:
        pid = 43210

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        serial_eval,
        "prepare_pinned_dataset_resources",
        fail_hf,
    )
    monkeypatch.setattr(
        serial_eval,
        "prepare_local_dataset_resources",
        prepare_local,
    )
    monkeypatch.setattr(
        serial_eval,
        "validate_canonical_publication_root",
        validate_publication,
    )
    monkeypatch.setattr(
        serial_eval,
        "_expected_shared_policy_checkpoint_binding",
        lambda: {
            "schema_version": 1,
            "profile_id": "pi05-b1kpt50-cs32",
            "resolved_path": str(checkpoint.resolve()),
            "files": {},
            "binding_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(serial_eval, "_validate_entry_python", lambda *_a, **_k: None)
    monkeypatch.setattr(
        serial_eval,
        "source_identity",
        lambda _root: {"commit": "abc", "dirty": False},
    )
    monkeypatch.setattr(
        serial_eval,
        "_checkout_identity",
        lambda _root: {"commit": "def", "dirty": False},
    )

    def start_vla(args, *, output_dir):
        calls["managed_vla_output_dir"] = output_dir
        calls["managed_vla_isolation"] = args._behavior_runtime_isolation
        calls["managed_vla_environment"] = (
            args._behavior_runtime_isolation.environment()
        )
        (output_dir / "vla_server.log").write_text("test log", encoding="utf-8")
        return "http://127.0.0.1:9123", _VlaProcess()

    monkeypatch.setattr(serial_eval, "start_vla_server", start_vla)
    monkeypatch.setattr(
        serial_eval,
        "_verify_input_fingerprints",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        serial_eval,
        "validate_campaign_runtime_isolation",
        lambda root, digest: (
            runtime_isolation
            if root == runtime_root.resolve() and digest == "e" * 64
            else (_ for _ in ()).throw(AssertionError("unexpected runtime binding"))
        ),
    )
    monkeypatch.setattr(
        serial_eval,
        "_gpu_lock_path",
        lambda device: tmp_path / f"gpu-{device}.lock",
    )
    monkeypatch.setattr(
        serial_eval,
        "_run_entry",
        lambda *_a, **_k: (0, False),
    )
    monkeypatch.setattr(serial_eval, "_terminate_process", lambda _process: None)

    exit_code = serial_eval.main(
        [
            "--output-root",
            str(output_root),
            "--repo-root",
            str(repo_root),
            "--python",
            str(entry_python),
            "--behavior-repo",
            str(behavior_repo),
            "--behavior-python",
            str(behavior_python),
            "--policy-checkpoint",
            str(checkpoint),
            "--cuda-device",
            "7",
            "--task-name",
            "picking_up_trash",
            "--public-seed",
            "10",
            "--test-instances-csv",
            str(csv_path),
            "--expected-csv-sha256",
            hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "--behavior-frozen-publication-root",
            str(frozen_root),
            "--behavior-frozen-provenance-sha256",
            "b" * 64,
            "--runtime-isolation-root",
            str(runtime_root),
            "--runtime-isolation-binding-sha256",
            "e" * 64,
            "--behavior-resource-local",
            str(FIXTURE_RESOURCES),
        ]
    )

    assert exit_code == 1
    assert calls["local_prepare"] == (
        "behavior",
        FIXTURE_RESOURCES.resolve(),
        repo_root / "resources" / ".snapshots",
    )
    assert calls["managed_vla_output_dir"] == output_root / "launcher_logs" / "vla"
    assert calls["managed_vla_isolation"] is runtime_isolation
    assert calls["managed_vla_environment"] == runtime_environment
    for value in runtime_environment.values():
        Path(value).resolve().relative_to(runtime_root.resolve())
    assert (output_root / "launcher_logs" / "vla" / "vla_server.log").is_file()
    assert not (output_root / "vla").exists()
    assert not any(
        path.name
        in {
            "_runtime_isolation",
            "runtime",
            "runtime_cache",
            "xdg",
            "ov_cache",
            "omnigibson_appdata",
            "omni_user",
            "tmp",
        }
        for path in output_root.rglob("*")
    )
    assert calls["publication"] == [
        {
            "expected_provenance_sha256": "b" * 64,
            "task_name": "picking_up_trash",
            "task_index": 1,
        },
        {
            "expected_provenance_sha256": "b" * 64,
            "task_name": "picking_up_trash",
            "task_index": 1,
            "public_seed": 1,
        },
    ]
    plan = json.loads((output_root / "eval_plan.json").read_text(encoding="utf-8"))
    assert plan["resource_source"] == binding.as_dict()
    assert plan["entries"][0]["public_seed"] == 10
    assert plan["entries"][0]["activity_instance_id"] == 108
    result = json.loads(
        (output_root / "eval_results.jsonl").read_text(encoding="utf-8")
    )
    expected_state_sha256 = hashlib.sha256(
        (state_dir / "scene_0_108_template-tro_state.json").read_bytes()
    ).hexdigest()
    assert result["instance_state_sha256"] == expected_state_sha256
    assert result["instance_state_binding_valid"] is True


@pytest.mark.parametrize(
    "legacy_flag",
    ("--max-total-vla-chunks", "--max-vla-chunks-per-call"),
)
def test_eval_cli_rejects_removed_pi0_chunk_limit_flags(
    tmp_path: Path,
    legacy_flag: str,
) -> None:
    with pytest.raises(SystemExit):
        serial_eval._parse_args(
            [
                "--output-root",
                str(tmp_path / "out"),
                "--cuda-device",
                "7",
                "--behavior-frozen-publication-root",
                str(tmp_path / "publication"),
                "--behavior-frozen-provenance-sha256",
                "a" * 64,
                legacy_flag,
                "1",
            ]
        )


def test_gpu7_eval_defaults_reserve_cleanup_inside_two_hours(
    tmp_path: Path,
) -> None:
    args = serial_eval._parse_args(
        [
            "--output-root",
            str(tmp_path / "out"),
            "--cuda-device",
            "7",
            "--behavior-frozen-publication-root",
            str(tmp_path / "publication"),
            "--behavior-frozen-provenance-sha256",
            "a" * 64,
        ]
    )

    assert args.max_wall_clock_s == 6900
    assert args.cleanup_deadline_s == 7080
    assert args.instance_timeout_s == 7200
    assert args.planner_timeout_s <= args.max_wall_clock_s


@pytest.mark.parametrize(
    "values",
    (
        (6901, 6900, 7080, 7200),
        (6900, 7080, 7080, 7200),
        (6900, 6900, 7200, 7200),
        (6900, 7080, 7201, 7201),
        (6900, 7080, 7199, 7201),
        (0, 6900, 7080, 7200),
    ),
)
def test_gpu7_eval_rejects_invalid_two_hour_deadline_order(
    values: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError):
        serial_eval._validate_deadline_budget(
            planner_timeout_s=values[0],
            max_wall_clock_s=values[1],
            cleanup_deadline_s=values[2],
            instance_timeout_s=values[3],
        )


def test_gpu7_eval_child_receives_wall_clock_and_dashboard_contract(
    tmp_path: Path,
) -> None:
    argv = build_entry_argv(
        python=Path("/python"),
        repo_root=tmp_path.resolve(),
        output_dir=tmp_path / "picking_up_trash_s10",
        behavior_repo=tmp_path / "behavior",
        behavior_python=Path("/behavior-python"),
        checkpoint=tmp_path / "checkpoint",
        activity_instance_id=108,
        public_seed=10,
        cuda_device="7",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        max_turns=200,
        planner_timeout_s=6900,
        max_wall_clock_s=6900,
        frozen_publication_root=tmp_path / "publication",
        frozen_provenance_sha256="a" * 64,
        reviewed_memory_snapshot_sha256="b" * 64,
        recipe_catalog_sha256="c" * 64,
        task_name="picking_up_trash",
        dashboard_event_sink=True,
        source_snapshot_root=tmp_path / "source-snapshot",
        source_snapshot_binding_sha256="d" * 64,
        runtime_isolation_root=tmp_path / "runtime-isolation",
        runtime_isolation_binding_sha256="e" * 64,
        runtime_isolation_namespace="paired-eval-agentic",
    )

    assert argv[argv.index("--max-wall-clock-s") + 1] == "6900"
    assert argv[argv.index("--planner-timeout-s") + 1] == "6900"
    assert "--behavior-dashboard-event-sink" in argv
    assert "--dashboard" not in argv
    assert "--dashboard-auto-start" not in argv
    assert argv[argv.index("--behavior-source-snapshot-root") + 1] == str(
        tmp_path / "source-snapshot"
    )
    assert argv[argv.index("--behavior-source-snapshot-binding-sha256") + 1] == "d" * 64
    assert argv[argv.index("--behavior-runtime-isolation-root") + 1] == str(
        tmp_path / "runtime-isolation"
    )
    assert (
        argv[argv.index("--behavior-runtime-isolation-binding-sha256") + 1] == "e" * 64
    )
    assert argv[argv.index("--behavior-runtime-namespace") + 1] == "paired-eval-agentic"


@pytest.mark.parametrize(
    "extra_args",
    (
        (
            "--source-snapshot-root",
            "/missing/source",
            "--source-snapshot-binding-sha256",
            "d" * 64,
            "--runtime-isolation-root",
            "/missing/runtime",
            "--runtime-isolation-binding-sha256",
            "e" * 64,
        ),
        (
            "--vla-endpoint",
            "http://127.0.0.1:9123",
            "--runtime-isolation-root",
            "/missing/runtime",
            "--runtime-isolation-binding-sha256",
            "e" * 64,
        ),
        (
            "--vla-endpoint",
            "http://127.0.0.1:9123",
            "--source-snapshot-root",
            "/missing/source",
            "--source-snapshot-binding-sha256",
            "d" * 64,
        ),
    ),
)
def test_gpu7_external_gpu_lock_requires_vla_source_and_runtime_isolation(
    tmp_path: Path,
    extra_args: tuple[str, ...],
) -> None:
    with pytest.raises(
        SystemExit,
        match=(
            "external-gpu-lock-owned requires an external VLA, sealed source "
            "snapshot, and bound campaign runtime isolation"
        ),
    ):
        serial_eval.main(
            [
                "--output-root",
                str(tmp_path / "out"),
                "--cuda-device",
                "7",
                "--behavior-frozen-publication-root",
                str(tmp_path / "publication"),
                "--behavior-frozen-provenance-sha256",
                "a" * 64,
                "--external-gpu-lock-owned",
                *extra_args,
            ]
        )


def test_gpu7_eval_rejects_unpaired_source_snapshot_identity(
    tmp_path: Path,
) -> None:
    kwargs = {
        "python": Path("/python"),
        "repo_root": tmp_path.resolve(),
        "output_dir": tmp_path / "picking_up_trash_s10",
        "behavior_repo": tmp_path / "behavior",
        "behavior_python": Path("/behavior-python"),
        "checkpoint": tmp_path / "checkpoint",
        "activity_instance_id": 108,
        "public_seed": 10,
        "cuda_device": "7",
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
        "max_turns": 200,
        "planner_timeout_s": 6900,
        "frozen_publication_root": tmp_path / "publication",
        "frozen_provenance_sha256": "a" * 64,
        "reviewed_memory_snapshot_sha256": "b" * 64,
        "recipe_catalog_sha256": "c" * 64,
        "task_name": "picking_up_trash",
    }

    with pytest.raises(ValueError, match="source_snapshot.*provided together"):
        build_entry_argv(
            **kwargs,
            source_snapshot_root=tmp_path / "source-snapshot",
        )
    with pytest.raises(ValueError, match="source_snapshot.*provided together"):
        build_entry_argv(
            **kwargs,
            source_snapshot_binding_sha256="d" * 64,
        )


def test_gpu7_external_vla_is_validated_and_marked_unmanaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class _Client:
        def __init__(self, endpoint):
            events.append(("init", endpoint))

        def healthz(self, *, timeout_ms, expected_checkpoint_binding):
            events.append(("healthz", timeout_ms, expected_checkpoint_binding))
            return {
                "config_name": "pi05_behavior",
                "checkpoint_binding": expected_checkpoint_binding,
                "actions_enabled": False,
            }

        def close(self):
            events.append("close")

    monkeypatch.setattr(serial_eval, "BehaviorVLAClient", _Client)
    binding = {"binding_sha256": "a" * 64}

    external = serial_eval._validate_external_vla(
        "http://127.0.0.1:9123",
        checkpoint_binding=binding,
    )

    assert external == {
        "endpoint": "http://127.0.0.1:9123",
        "config_name": "pi05_behavior",
        "checkpoint_binding": binding,
        "managed": False,
    }
    assert events == [
        ("init", "http://127.0.0.1:9123"),
        ("healthz", 5000, binding),
        "close",
    ]


@pytest.mark.parametrize("actions_enabled", [True, None, 0, 1, "false"])
def test_gpu7_external_vla_requires_exact_safe_idle_false(
    monkeypatch: pytest.MonkeyPatch,
    actions_enabled: object,
) -> None:
    events: list[object] = []

    class _Client:
        def __init__(self, endpoint):
            events.append(("init", endpoint))

        def healthz(self, *, timeout_ms, expected_checkpoint_binding):
            del timeout_ms
            return {
                "config_name": "pi05_behavior",
                "checkpoint_binding": expected_checkpoint_binding,
                "actions_enabled": actions_enabled,
            }

        def close(self):
            events.append("close")

    monkeypatch.setattr(serial_eval, "BehaviorVLAClient", _Client)
    with pytest.raises(RuntimeError, match="unexpected external VLA metadata"):
        serial_eval._validate_external_vla(
            "http://127.0.0.1:9123",
            checkpoint_binding={"binding_sha256": "a" * 64},
        )
    assert events[-1] == "close"


def test_gpu7_external_vla_requires_actions_enabled_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def __init__(self, _endpoint):
            pass

        def healthz(self, **_kwargs):
            return {"config_name": "pi05_behavior"}

        def close(self):
            pass

    monkeypatch.setattr(serial_eval, "BehaviorVLAClient", _Client)
    with pytest.raises(RuntimeError, match="unexpected external VLA metadata"):
        serial_eval._validate_external_vla(
            "http://127.0.0.1:9123",
            checkpoint_binding={"binding_sha256": "a" * 64},
        )


def test_external_vla_safe_idle_reset_validates_disables_then_revalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    binding = {"binding_sha256": "a" * 64}

    class _Client:
        def __init__(self, endpoint):
            events.append(("init", endpoint))
            self.health_calls = 0

        def healthz(self, *, timeout_ms, expected_checkpoint_binding):
            self.health_calls += 1
            events.append(
                (
                    "healthz",
                    self.health_calls,
                    timeout_ms,
                    expected_checkpoint_binding,
                )
            )
            return {
                "config_name": "pi05_behavior",
                "checkpoint_binding": expected_checkpoint_binding,
                "actions_enabled": self.health_calls == 1,
            }

        def disable_actions(self, *, timeout_ms):
            events.append(("disable_actions", timeout_ms))
            return {"actions_enabled": False}

        def close(self):
            events.append("close")

    monkeypatch.setattr(serial_eval, "BehaviorVLAClient", _Client)

    result = serial_eval._disable_external_vla_actions(
        "http://127.0.0.1:9123",
        checkpoint_binding=binding,
    )

    assert result == {
        "endpoint": "http://127.0.0.1:9123",
        "config_name": "pi05_behavior",
        "checkpoint_binding": binding,
        "managed": False,
        "actions_enabled": False,
    }
    assert events == [
        ("init", "http://127.0.0.1:9123"),
        ("healthz", 1, 5000, binding),
        ("disable_actions", 5000),
        ("healthz", 2, 5000, binding),
        "close",
    ]


def test_external_vla_safe_idle_reset_rejects_wrong_config_before_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _Client:
        def __init__(self, _endpoint):
            pass

        def healthz(self, **_kwargs):
            events.append("healthz")
            return {
                "config_name": "wrong",
                "checkpoint_binding": {"binding_sha256": "a" * 64},
                "actions_enabled": True,
            }

        def disable_actions(self, **_kwargs):
            events.append("disable_actions")
            return {"actions_enabled": False}

        def close(self):
            events.append("close")

    monkeypatch.setattr(serial_eval, "BehaviorVLAClient", _Client)
    with pytest.raises(RuntimeError, match="unexpected external VLA metadata"):
        serial_eval._disable_external_vla_actions(
            "http://127.0.0.1:9123",
            checkpoint_binding={"binding_sha256": "a" * 64},
        )
    assert events == ["healthz", "close"]


@pytest.mark.parametrize(
    ("returncode", "externally_bound", "observed_offset_ns", "expected"),
    (
        (-int(serial_eval.signal.SIGTERM), True, 0, True),
        (-int(serial_eval.signal.SIGTERM), True, -1, False),
        (-int(serial_eval.signal.SIGTERM), False, 0, False),
        (-int(serial_eval.signal.SIGKILL), True, 0, False),
        (0, True, 0, False),
    ),
)
def test_only_external_action_deadline_sigterm_latches_timeout(
    returncode: int,
    externally_bound: bool,
    observed_offset_ns: int,
    expected: bool,
) -> None:
    started = 5_000_000_000
    binding = serial_eval.InstanceDeadlineBinding(
        started_monotonic_ns=started,
        action_deadline_monotonic_ns=started + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=started + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=started + 7200 * 1_000_000_000,
    )
    assert (
        serial_eval._is_external_action_deadline_sigterm(
            returncode=returncode,
            externally_bound_deadline=externally_bound,
            deadline_binding=binding,
            observed_at_monotonic_ns=(
                binding.action_deadline_monotonic_ns + observed_offset_ns
            ),
        )
        is expected
    )


def test_run_entry_latches_external_action_deadline_sigterm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _entry(tmp_path, task_name="picking_up_trash")
    source = tmp_path / "source"
    source.mkdir()
    child_pid = 22012
    runner_pid = serial_eval.os.getpid()

    class _Process:
        pid = child_pid
        returncode: int | None = None

        def wait(self, *, timeout: float):
            assert timeout > 0
            self.returncode = -int(serial_eval.signal.SIGTERM)
            return self.returncode

        def poll(self):
            return self.returncode

    process = _Process()
    monkeypatch.setattr(
        serial_eval.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def fake_proc_stat(pid: int):
        selected = child_pid if pid == child_pid else runner_pid
        return {
            "pid": selected,
            "state": "S",
            "ppid": runner_pid if selected == child_pid else 1,
            "pgid": selected,
            "sid": selected,
            "start_ticks": 100 if selected == child_pid else 200,
        }

    monkeypatch.setattr(serial_eval, "_proc_stat", fake_proc_stat)
    started = 5_000_000_000
    binding = serial_eval.InstanceDeadlineBinding(
        started_monotonic_ns=started,
        action_deadline_monotonic_ns=started + 6900 * 1_000_000_000,
        cleanup_deadline_monotonic_ns=started + 7080 * 1_000_000_000,
        hard_deadline_monotonic_ns=started + 7200 * 1_000_000_000,
    )
    monotonic_samples = iter(
        (
            binding.action_deadline_monotonic_ns,
            binding.action_deadline_monotonic_ns,
        )
    )
    monkeypatch.setattr(
        serial_eval.time,
        "monotonic_ns",
        lambda: next(monotonic_samples),
    )

    assert serial_eval._run_entry(
        entry,
        repo_root=tmp_path,
        log_stream=io.BytesIO(),
        timeout_s=7200,
        cleanup_deadline_s=7080,
        action_deadline_s=6900,
        source_snapshot_root=source,
        source_snapshot_binding_sha256="b" * 64,
        deadline_binding=binding,
        externally_bound_deadline=True,
    ) == (-int(serial_eval.signal.SIGTERM), True)
    receipt = json.loads(
        (
            entry.output_dir.parent / serial_eval.INSTANCE_CHILD_PROCESS_FILENAME
        ).read_text(encoding="utf-8")
    )
    assert receipt["state"] == "exited"
    assert receipt["returncode"] == -int(serial_eval.signal.SIGTERM)
    assert receipt["timed_out"] is True


def test_gpu7_eval_binds_exact_source_snapshot_in_child_manifest(
    tmp_path: Path,
) -> None:
    source_snapshot = {
        "schema_version": 1,
        "kind": "hash_sealed_source_snapshot",
        "source_tree_sha256": "a" * 64,
        "binding_sha256": "b" * 64,
    }
    entry = replace(
        _entry(tmp_path, task_name="picking_up_trash"),
        source_snapshot_binding=source_snapshot,
    )
    _write_bound_artifacts(entry, success=False)
    manifest_path = entry.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_snapshot"] = json.loads(json.dumps(source_snapshot))
    _write_json(manifest_path, manifest)

    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit=None,
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "task_failed"
    assert errors == []

    manifest["source_snapshot"]["source_tree_sha256"] = "c" * 64
    _write_json(manifest_path, manifest)
    outcome, errors, _ = validate_instance_result(
        entry,
        source_commit=None,
        subprocess_exit_code=0,
        timed_out=False,
    )
    assert outcome == "incomplete"
    assert "manifest sealed source snapshot binding mismatch" in errors
