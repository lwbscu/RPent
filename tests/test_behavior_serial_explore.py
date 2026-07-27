from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from behavior_resource_fixtures import (
    FIXTURE_RESOURCES,
    fixture_resource_binding,
)

from robots.behavior import serial_explore
from robots.behavior.dashboard_sink import FileDashboardSink
from robots.behavior.dashboard_state import State
from robots.behavior.memory_snapshot import load_behavior_memory_snapshot
from robots.behavior.serial_explore import (
    TAG,
    AttemptExecution,
    ExploreConfig,
    ExploreDependencies,
    _attempt_summary,
    _new_manifest,
    _runtime_attempt_result,
    _terminal_failure_binding,
    build_attempt_argv,
    correct_existing_success,
    publish_existing_success,
    run_explore_job,
    sanitize_prior_attempt_summaries,
)


@pytest.fixture(autouse=True)
def _isolate_serial_explore_gpu_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(
        serial_explore,
        "_gpu_lock_path",
        lambda cuda_device: tmp_path / f"gpu-{cuda_device}.lock",
    )


class _FakeProcess:
    pid = 987654321

    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode


def test_attempt_summary_reports_raw_official_success():
    assert "raw official success confirmed" in _attempt_summary(
        {"task_success": True},
        raw_success=True,
    )


class _FakeDashboard:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.finished: list[tuple[int, str]] = []
        self.job_progress: list[dict] = []
        self.events: list[dict] = []
        self.done: list[bool] = []

    def begin_attempt(self, *, attempt_index, output_dir, video_path):
        del output_dir, video_path
        self.started.append(attempt_index)

    def end_attempt(self, *, attempt_index, outcome):
        self.finished.append((attempt_index, outcome))

    def on_job_progress(self, cumulative):
        self.job_progress.append(dict(cumulative))

    def on_event(self, event):
        self.events.append(dict(event))

    def on_tool_start(self, name, args):
        del name, args

    def on_tool_progress(self, name, payload):
        del name, payload

    def on_tool_result(self, name, result):
        del name, result

    def mark_done(self, value):
        self.done.append(bool(value))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(tmp_path: Path, repo_root: Path) -> ExploreConfig:
    checkpoint = (tmp_path / "checkpoint").resolve()
    return ExploreConfig(
        output_root=tmp_path / "explore" / "run-001",
        repo_root=repo_root,
        python=Path("/usr/bin/python3"),
        behavior_repo=tmp_path / "behavior-repo",
        behavior_python=tmp_path / "behavior-python",
        policy_checkpoint=checkpoint,
        policy_checkpoint_binding={
            "schema_version": 1,
            "profile_id": "pi05-b1kpt50-cs32",
            "resolved_path": str(checkpoint),
            "files": {},
            "binding_sha256": "a" * 64,
        },
        resource_binding=fixture_resource_binding(),
        dashboard=True,
    )


def test_main_preserves_virtualenv_python_symlink(monkeypatch, tmp_path):
    python_target = tmp_path / "python-target"
    python_target.write_text("", encoding="utf-8")
    python_link = tmp_path / "venv-python"
    python_link.symlink_to(python_target)
    behavior_repo = tmp_path / "behavior-repo"
    behavior_repo.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    captured = {}

    def fake_run(config):
        captured["config"] = config
        return {"status": "blocked", "task_success": False}

    monkeypatch.setattr(serial_explore, "run_explore_job", fake_run)
    monkeypatch.setattr(
        serial_explore,
        "prepare_pinned_dataset_resources",
        lambda *args, **kwargs: fixture_resource_binding(),
    )
    monkeypatch.setattr(
        serial_explore,
        "_expected_job_checkpoint_binding",
        lambda path: {
            "schema_version": 1,
            "profile_id": "pi05-b1kpt50-cs32",
            "resolved_path": str(Path(path).resolve()),
            "files": {},
            "binding_sha256": "a" * 64,
        },
    )
    exit_code = serial_explore.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--repo-root",
            str(Path(__file__).resolve().parents[1]),
            "--python",
            str(python_link),
            "--behavior-repo",
            str(behavior_repo),
            "--behavior-python",
            str(python_link),
            "--policy-checkpoint",
            str(checkpoint),
        ]
    )

    assert exit_code == 2
    assert captured["config"].python == python_link.absolute()
    assert captured["config"].behavior_python == python_link.absolute()
    assert captured["config"].dashboard_language == "en"


def test_main_uses_local_resources_without_huggingface(
    monkeypatch,
    tmp_path: Path,
) -> None:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    behavior_repo = tmp_path / "behavior-repo"
    behavior_repo.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    binding = fixture_resource_binding()
    captured: dict[str, object] = {}

    def fail_hf(*_args, **_kwargs):
        raise AssertionError("HuggingFace preparation must not run")

    def prepare_local(subtree, *, source_root, cache_root):
        captured["local_prepare"] = (subtree, source_root, cache_root)
        return binding

    def fake_run(config):
        captured["config"] = config
        return {"status": "blocked", "task_success": False}

    monkeypatch.setattr(serial_explore, "run_explore_job", fake_run)
    monkeypatch.setattr(
        serial_explore,
        "prepare_pinned_dataset_resources",
        fail_hf,
    )
    monkeypatch.setattr(
        serial_explore,
        "prepare_local_dataset_resources",
        prepare_local,
    )
    monkeypatch.setattr(
        serial_explore,
        "_expected_job_checkpoint_binding",
        lambda path: {
            "schema_version": 1,
            "profile_id": "pi05-b1kpt50-cs32",
            "resolved_path": str(Path(path).resolve()),
            "files": {},
            "binding_sha256": "a" * 64,
        },
    )

    exit_code = serial_explore.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--repo-root",
            str(Path(__file__).resolve().parents[1]),
            "--python",
            str(python),
            "--behavior-repo",
            str(behavior_repo),
            "--behavior-python",
            str(python),
            "--policy-checkpoint",
            str(checkpoint),
            "--task-name",
            "picking_up_trash",
            "--public-seed",
            "1",
            "--behavior-resource-local",
            str(FIXTURE_RESOURCES),
        ]
    )

    assert exit_code == 2
    assert captured["local_prepare"] == (
        "behavior",
        FIXTURE_RESOURCES.resolve(),
        Path(__file__).resolve().parents[1] / "resources" / ".snapshots",
    )
    config = captured["config"]
    assert isinstance(config, ExploreConfig)
    assert config.task_name == "picking_up_trash"
    assert config.public_seed == 1
    assert config.resource_binding == binding
    assert _new_manifest(config, "trash-s1-local")["resource_source"] == (
        binding.as_dict()
    )


def test_epoch_predecessor_binding_reader_rejects_symlinks(tmp_path):
    payload = {"epoch_boundary_sha256": "a" * 64, "completed_prefix": 11}
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_path = real_dir / "epoch_boundary.json"
    real_path.write_text(json.dumps(payload), encoding="utf-8")

    binding = serial_explore._read_epoch_predecessor_binding_file(real_path)

    assert binding == {
        "binding_file_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "binding": payload,
    }
    file_link = tmp_path / "file-link.json"
    file_link.symlink_to(real_path)
    with pytest.raises(ValueError, match="non-symlink regular file"):
        serial_explore._read_epoch_predecessor_binding_file(file_link)
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink regular file"):
        serial_explore._read_epoch_predecessor_binding_file(
            directory_link / "epoch_boundary.json"
        )


def _dependencies(outcomes: list[tuple[bool, bool]]):
    process = _FakeProcess()
    dashboard = _FakeDashboard()
    calls = SimpleNamespace(vla_starts=0, attempts=[], bindings=[])
    reviewed_snapshot = load_behavior_memory_snapshot(FIXTURE_RESOURCES / "memory")
    reviewed_binding = {
        "snapshot_sha256": reviewed_snapshot.snapshot_sha256,
        "manifest": asdict(reviewed_snapshot.manifest_binding),
        "files": {
            name: asdict(metadata) for name, metadata in reviewed_snapshot.files.items()
        },
        "selection": reviewed_snapshot.select_task("turning_on_radio").public_binding,
    }
    catalog_config = SimpleNamespace(
        repo_root=Path(__file__).resolve().parents[1],
        recipe_catalog_sha256=None,
        resource_binding=fixture_resource_binding(),
    )
    reviewed_catalog_binding = serial_explore._reviewed_recipe_catalog_binding(
        catalog_config
    )

    def start_vla(config, output_dir):
        del config, output_dir
        calls.vla_starts += 1
        return "http://127.0.0.1:9123", process

    def stop_vla(proc):
        assert proc is process
        process.returncode = 0

    def bind_attempt_vla(endpoint, binding_id):
        assert endpoint == "http://127.0.0.1:9123"
        calls.bindings.append(binding_id)
        return {
            "actions_enabled": False,
            "binding_digest": hashlib.sha256(binding_id.encode()).hexdigest(),
        }

    def run_attempt(argv, output_dir, log_path, timeout_s):
        del log_path, timeout_s
        index = len(calls.attempts) + 1
        calls.attempts.append((argv, output_dir))
        output_dir.mkdir(parents=True)
        success, claimed_artifact_seal_complete = outcomes[index - 1]
        job_id = argv[argv.index("--behavior-job-id") + 1]
        attempt_nonce = f"test-attempt-nonce-{index}"
        run_nonce = f"test-run-nonce-{index}"
        final = {
            "task_success": success,
            "artifact_seal_complete": claimed_artifact_seal_complete,
            "official_success_source": 'info["done"]["success"]',
            "agent_summary": f"attempt {index} summary",
            "elapsed_s": 1.25,
            "runtime_cleanup": "complete",
            "job": {
                "job_id": job_id,
                "attempt_index": index,
            },
        }
        _write_json(output_dir / "final_result.json", final)
        _write_json(
            output_dir / "run_manifest.json",
            {
                "status": "stopped",
                "stopped_at": "2026-07-23T00:00:00Z",
                "job": {"job_id": job_id, "attempt_index": index},
                "schema_version": 6,
                "protocol": {
                    "behavior_phase": "explore",
                    "public_seed": 0,
                    "recipe_tag": TAG,
                    "task_identity": {
                        "task_name": "turning_on_radio",
                        "activity_definition_id": 0,
                        "activity_instance_id": 242,
                    },
                    "public_primitives": list(serial_explore.BEHAVIOR_TOOL_NAMES),
                    "public_tool_contract_version": (
                        serial_explore.CURRENT_PUBLIC_TOOL_CONTRACT_VERSION
                    ),
                    "agent_finish_registered": False,
                    "pi0_nav_pick_contract": (
                        serial_explore.pi0_nav_pick_exact_chunk_contract()
                    ),
                    "task_spec": {
                        "task_index": 0,
                        "task_name": "turning_on_radio",
                        "activity_definition_id": 0,
                        "prompt_profile_id": "turning_on_radio",
                    },
                    "prompt": {
                        "profile_id": "turning_on_radio",
                        "rendered_system_sha256": "a" * 64,
                        "rendered_user_sha256": "b" * 64,
                        "combined_sha256": "c" * 64,
                    },
                    "attempts": {
                        "initial_attempt_index": index,
                        "max_attempts": None,
                        "reset_registered": False,
                    },
                },
                "task": {
                    "suite": "behavior_2025_challenge",
                    "task": 0,
                    "task_name": "turning_on_radio",
                    "public_seed": 0,
                },
                "task_identity": {
                    "task_name": "turning_on_radio",
                    "activity_definition_id": 0,
                    "activity_instance_id": 242,
                },
                "native_binding": {
                    "activity_definition_id": 0,
                    "activity_instance_id": 242,
                    "env_seed": 0,
                },
                "reviewed_repo_memory": reviewed_binding,
                "reviewed_recipe_catalog": (
                    serial_explore._runtime_recipe_catalog_binding(
                        reviewed_catalog_binding
                    )
                ),
                "resource_source": fixture_resource_binding().as_dict(),
                "frozen_eval_inputs": None,
                "processes": {
                    "env": {
                        "managed": True,
                        "pid": _FakeProcess.pid,
                        "start_ticks": 1,
                        "stopped_at": "2026-07-23T00:00:00Z",
                    },
                    "vla": {
                        "managed": False,
                        "pid": None,
                        "stopped_at": None,
                    },
                },
            },
        )
        _write_json(
            output_dir / f"{TAG}.json",
            {
                "task_success": success,
                "artifact_seal_complete": claimed_artifact_seal_complete,
                "official_success_source": 'info["done"]["success"]',
                "total_env_steps": index * 10,
                "global_vla_chunks": index,
                "global_tool_calls": index + 2,
            },
        )
        receipt = None
        if success:
            receipt = {
                "schema_version": 1,
                "source": 'info["done"]["success"]',
                "run_nonce": run_nonce,
                "attempt_nonce": attempt_nonce,
                "attempt_index": index,
                "env_step": index * 10,
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
            _write_json(output_dir / "official_success_receipt.json", receipt)
        (output_dir / "behavior_action_trace.jsonl").write_text(
            json.dumps({"env_step": index * 10, "info_done": {"success": success}})
            + "\n",
            encoding="utf-8",
        )
        (output_dir / "behavior_tool_trace.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "tool": "observe",
                    "attempt_index": index,
                    "task_success": success,
                    "result": {
                        "task_success": success,
                        "primitive_success": True,
                        "official_success_source": 'info["done"]["success"]',
                        "attempt_nonce": attempt_nonce,
                        "run_nonce": run_nonce,
                        **(
                            {
                                "official_success_receipt": receipt,
                                "official_success_receipt_path": str(
                                    output_dir / "official_success_receipt.json"
                                ),
                            }
                            if receipt is not None
                            else {}
                        ),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return AttemptExecution(
            exit_code=0,
            timed_out=False,
            final_result=final,
            forced_cleanup={},
            alive_after_cleanup={},
            ambiguous_groups={},
        )

    dependencies = ExploreDependencies(
        start_vla=start_vla,
        stop_vla=stop_vla,
        bind_attempt_vla=bind_attempt_vla,
        run_attempt=run_attempt,
        free_disk_bytes=lambda path: 100 * 1024**3,
        start_dashboard=lambda config, job_id: (
            object(),
            dashboard,
            "http://127.0.0.1:8765",
        ),
    )
    return dependencies, calls, dashboard


def _successful_attempt_fixture(tmp_path: Path):
    dependencies, _calls, _dashboard = _dependencies([(True, True)])
    attempt_dir = tmp_path / "attempt_001"
    execution = dependencies.run_attempt(
        ["runner", "--behavior-job-id", "test-job"],
        attempt_dir,
        tmp_path / "attempt.log",
        60.0,
    )
    assert isinstance(execution.final_result, dict)
    return attempt_dir, execution.final_result


@pytest.mark.parametrize(
    "action_record",
    (
        {"env_step": 10, "info_done": {"success": True}},
        {"env_idx": 0, "step": 9, "info_done": {"success": True}},
        {"env_step": 10, "step": 10, "info_done": {"success": True}},
    ),
    ids=(
        "env-step-schema",
        "zero-based-step-schema",
        "env-step-authoritative",
    ),
)
def test_official_success_binding_normalizes_action_trace_step(
    tmp_path,
    action_record,
):
    attempt_dir, _final_result = _successful_attempt_fixture(tmp_path)
    (attempt_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps(action_record) + "\n",
        encoding="utf-8",
    )

    binding = serial_explore._official_success_binding(attempt_dir)

    assert binding is not None
    assert binding["env_step"] == 10


@pytest.mark.parametrize(
    "action_record",
    (
        {"env_idx": 0, "step": 10, "info_done": {"success": True}},
        {"step": 9, "info_done": {"success": True}},
        {"env_idx": 1, "step": 9, "info_done": {"success": True}},
        {"env_idx": True, "step": 9, "info_done": {"success": True}},
        {
            "env_step": True,
            "env_idx": 0,
            "step": 9,
            "info_done": {"success": True},
        },
    ),
    ids=(
        "receipt-off-by-one",
        "legacy-step-missing-env-index",
        "legacy-step-wrong-env-index",
        "legacy-step-bool-env-index",
        "bool-env-step-does-not-fallback",
    ),
)
def test_official_success_binding_rejects_step_lineage_mismatch(
    tmp_path,
    action_record,
):
    attempt_dir, _final_result = _successful_attempt_fixture(tmp_path)
    (attempt_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps(action_record) + "\n",
        encoding="utf-8",
    )

    assert serial_explore._official_success_binding(attempt_dir) is None


def test_zero_based_action_trace_success_is_not_downgraded_as_unverifiable(tmp_path):
    attempt_dir, final_result = _successful_attempt_fixture(tmp_path)
    (attempt_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps({"env_idx": 0, "step": 9, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )

    resolved = _runtime_attempt_result(attempt_dir, final_result)

    assert resolved is not None
    assert resolved["task_success"] is True
    assert resolved.get("error") != "unverifiable task_success=true was rejected"


def _rewrite_signed_success_receipt(
    attempt_dir: Path,
    **updates,
) -> None:
    receipt_path = attempt_dir / "official_success_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(updates)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    _write_json(receipt_path, receipt)

    tool_path = attempt_dir / "behavior_tool_trace.jsonl"
    records = [
        json.loads(line)
        for line in tool_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        result = record.get("result")
        if isinstance(result, dict) and result.get("task_success") is True:
            result["official_success_receipt"] = receipt
    tool_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "receipt_update",
    (
        {"schema_version": 2},
        {"schema_version": True},
        {"attempt_index": True},
        {"attempt_index": "1"},
    ),
    ids=(
        "wrong-schema",
        "bool-schema",
        "bool-attempt-index",
        "string-attempt-index",
    ),
)
def test_official_success_binding_rejects_receipt_schema_and_attempt_types(
    tmp_path,
    receipt_update,
):
    attempt_dir, _final_result = _successful_attempt_fixture(tmp_path)
    _rewrite_signed_success_receipt(attempt_dir, **receipt_update)

    assert serial_explore._official_success_binding(attempt_dir) is None


def test_build_attempt_argv_is_fresh_child_without_dashboard_or_reset(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    argv = build_attempt_argv(
        _config(tmp_path, repo_root),
        job_id="job-1",
        attempt_index=7,
        output_dir=tmp_path / "attempt_007",
        summaries_path=tmp_path / "summaries.json",
        vla_endpoint="http://127.0.0.1:9123",
        vla_binding_id="job-1.a7.binding",
        reviewed_memory_snapshot_sha256=(
            "4b3a8f9bbc9b8938d5b984451b459bfdb5620597fe9eacefbf0f30eec028215b"
        ),
        recipe_catalog_sha256="c" * 64,
    )
    assert "--behavior-job-id" in argv
    assert "--behavior-attempt-index" in argv
    assert "--behavior-vla-binding-id" in argv
    assert "--behavior-reviewed-memory-snapshot-sha256" in argv
    assert argv[argv.index("--behavior-recipe-catalog-sha256") + 1] == "c" * 64
    assert "--vla-endpoint" in argv
    assert argv[argv.index("--max-tool-calls") + 1] == "350"
    assert "--max-total-vla-chunks" not in argv
    assert "--max-vla-chunks-per-call" not in argv
    assert "--behavior-dashboard-event-sink" in argv
    assert "--dashboard" not in argv
    assert "--dashboard-auto-start" not in argv
    assert "--reset" not in argv


def test_build_attempt_argv_omits_event_sink_without_parent_dashboard(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = replace(_config(tmp_path, repo_root), dashboard=False)

    argv = build_attempt_argv(
        config,
        job_id="job-1",
        attempt_index=1,
        output_dir=tmp_path / "attempt_001",
        summaries_path=tmp_path / "summaries.json",
        vla_endpoint="http://127.0.0.1:9123",
        vla_binding_id="job-1.a1.binding",
        reviewed_memory_snapshot_sha256="b" * 64,
        recipe_catalog_sha256="c" * 64,
    )

    assert "--behavior-dashboard-event-sink" not in argv


@pytest.mark.parametrize(
    ("public_seed", "native_instance"),
    ((0, 196), (1, 67), (9, 246)),
)
def test_trash_explore_resolves_task_scoped_public_seed_and_fixed_env_seed(
    tmp_path,
    public_seed,
    native_instance,
):
    repo_root = Path(__file__).resolve().parents[1]
    config = replace(
        _config(tmp_path, repo_root),
        task_name="picking_up_trash",
        public_seed=public_seed,
    )
    argv = build_attempt_argv(
        config,
        job_id="trash-job",
        attempt_index=1,
        output_dir=tmp_path / "attempt_001",
        summaries_path=tmp_path / "summaries.json",
        vla_endpoint="http://127.0.0.1:9123",
        vla_binding_id="trash-job.a1.binding",
        reviewed_memory_snapshot_sha256="b" * 64,
        recipe_catalog_sha256="c" * 64,
    )

    assert argv[argv.index("--task") + 1] == "1"
    assert argv[argv.index("--task-name") + 1] == "picking_up_trash"
    assert argv[argv.index("--activity-definition-id") + 1] == "0"
    assert argv[argv.index("--activity-instance-id") + 1] == str(native_instance)
    assert argv[argv.index("--scene-model") + 1] == "house_double_floor_lower"
    assert argv[argv.index("--public-seed") + 1] == str(public_seed)
    assert argv[argv.index("--seed") + 1] == "0"
    assert argv[argv.index("--behavior-phase") + 1] == "explore"
    assert argv[argv.index("--policy-checkpoint") + 1] == str(config.policy_checkpoint)
    assert argv[argv.index("--behavior-policy-checkpoint-binding-file") + 1] == str(
        config.output_root / "policy_checkpoint_binding.json"
    )
    assert "turning_on_radio" not in argv


def test_trash_explore_rejects_s10_at_explore_eval_boundary(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path, Path(__file__).resolve().parents[1]),
        task_name="picking_up_trash",
        public_seed=10,
    )

    with pytest.raises(ValueError, match="does not allow s10 in explore"):
        build_attempt_argv(
            config,
            job_id="trash-s10-job",
            attempt_index=1,
            output_dir=tmp_path / "attempt_001",
            summaries_path=tmp_path / "summaries.json",
            vla_endpoint="http://127.0.0.1:9123",
            vla_binding_id="trash-s10-job.a1.binding",
            reviewed_memory_snapshot_sha256="b" * 64,
            recipe_catalog_sha256="c" * 64,
        )


def test_trash_s1_explore_manifest_binds_instance67_and_env_seed_zero(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path, Path(__file__).resolve().parents[1]),
        task_name="picking_up_trash",
        public_seed=1,
    )

    manifest = _new_manifest(config, "trash-s1-job")

    assert manifest["protocol"]["task_name"] == "picking_up_trash"
    assert manifest["protocol"]["task_index"] == 1
    assert manifest["protocol"]["public_seed"] == 1
    assert manifest["protocol"]["recipe_tag"] == "picking_up_trash_s1"
    assert manifest["native_binding"] == {
        "mapping_version": "picking_up_trash_public_seed_v1",
        "activity_definition_id": 0,
        "activity_instance_id": 67,
        "scene_model": "house_double_floor_lower",
        "env_seed": 0,
    }


def test_explore_checkpoint_default_ignores_ambient_sft(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PI05_CHECKPOINT_PATH",
        "/tmp/pi05-turning_on_radio-sft-from-jax",
    )

    args = serial_explore._parse_args(["--output-root", str(tmp_path)])

    assert args.policy_checkpoint == str(serial_explore.SHARED_POLICY_CHECKPOINT_PATH)


def test_explore_tool_call_default_is_350_and_cli_override_is_preserved(tmp_path):
    default_args = serial_explore._parse_args(["--output-root", str(tmp_path)])
    override_args = serial_explore._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--max-tool-calls",
            "417",
        ]
    )

    assert default_args.max_tool_calls == 350
    assert override_args.max_tool_calls == 417


@pytest.mark.parametrize(
    "legacy_flag",
    ("--max-total-vla-chunks", "--max-vla-chunks-per-call"),
)
def test_explore_cli_rejects_removed_pi0_chunk_limit_flags(
    tmp_path: Path,
    legacy_flag: str,
) -> None:
    with pytest.raises(SystemExit):
        serial_explore._parse_args(["--output-root", str(tmp_path), legacy_flag, "1"])


@pytest.mark.parametrize(
    ("public_seed", "native_instance"),
    ((0, 196), (1, 67)),
)
def test_trash_dashboard_exposes_public_instance_and_seed_range(
    monkeypatch,
    tmp_path,
    public_seed,
    native_instance,
):
    import robots.behavior.dashboard_server as dashboard_module

    captured = {}

    class _Server:
        def __init__(self, **kwargs):
            captured["server_kwargs"] = kwargs

        def register(self, state):
            captured["state"] = state

        def arm_auto_start(self, metadata):
            captured["auto_start"] = metadata

        @staticmethod
        def start():
            return "http://127.0.0.1:8765"

    monkeypatch.setattr(dashboard_module, "DashboardServer", _Server)
    config = replace(
        _config(tmp_path, Path(__file__).resolve().parents[1]),
        task_name="picking_up_trash",
        public_seed=public_seed,
    )

    _server, state, url = serial_explore._default_start_dashboard(
        config,
        "trash-job",
    )

    assert url == "http://127.0.0.1:8765"
    detail = state.run_detail()
    assert detail["metadata"]["task-name"] == "picking_up_trash"
    assert detail["metadata"]["public-seed"] == public_seed
    assert detail["metadata"]["public-instance-id"] == native_instance
    assert detail["metadata"]["public-seed-max"] == 19
    assert detail["metadata"]["max-tool-calls"] == 350
    assert captured["server_kwargs"]["language"] == "en"


def test_explore_catalog_selection_and_epoch_predecessor_are_bound(
    monkeypatch, tmp_path
):
    calls: list[tuple[str, str]] = []

    class _Catalog:
        catalog_sha256 = "c" * 64
        manifest_binding = {"manifest_sha256": "d" * 64}
        files = {"turning_on_radio/reviewed/recipe.jsonl": {"sha256": "e" * 64}}

        def select(self, task_name, consumer):
            calls.append((task_name, consumer))
            return type(
                "_Selection",
                (),
                {
                    "public_binding": {
                        "consumer": consumer,
                        "selected_ids": ["reviewed_control_hypothesis_recovery_v1"],
                        "selected_entries": [
                            {
                                "entry_id": ("reviewed_control_hypothesis_recovery_v1"),
                                "provenance_class": ("candidate_explore_reviewed"),
                            }
                        ],
                    },
                    "selected_ids": ("reviewed_control_hypothesis_recovery_v1",),
                },
            )()

    monkeypatch.setattr(
        serial_explore,
        "load_behavior_recipe_catalog",
        lambda root: _Catalog(),
    )
    predecessor = {
        "epoch_boundary_sha256": "f" * 64,
        "completed_prefix": 11,
    }
    config = replace(
        _config(tmp_path, Path(__file__).resolve().parents[1]),
        recipe_catalog_sha256="c" * 64,
        epoch_predecessor_binding=predecessor,
    )

    binding = serial_explore._reviewed_recipe_catalog_binding(config)
    manifest = serial_explore._new_manifest(config, "job-test")
    predecessor["completed_prefix"] = 99

    assert calls == [
        ("turning_on_radio", "explore"),
        ("turning_on_radio", "explore"),
    ]
    assert binding["selection"]["consumer"] == "explore"
    assert manifest["reviewed_recipe_catalog"] == binding
    assert manifest["epoch_predecessor"] == {
        "epoch_boundary_sha256": "f" * 64,
        "completed_prefix": 11,
    }
    assert "vla_chunks" not in manifest["protocol"]["per_attempt_limits"]
    assert "vla_chunks_per_call" not in manifest["protocol"]["per_attempt_limits"]
    assert "vla_chunks" not in manifest["protocol"]["total_limits"]
    assert manifest["cumulative"]["vla_chunks"] == 0
    assert manifest["cumulative"]["vla_invocations"] == 0


def test_explore_rejects_recipe_catalog_drift(monkeypatch, tmp_path):
    catalog = SimpleNamespace(catalog_sha256="a" * 64)
    monkeypatch.setattr(
        serial_explore,
        "load_behavior_recipe_catalog",
        lambda root: catalog,
    )
    config = replace(
        _config(tmp_path, Path(__file__).resolve().parents[1]),
        recipe_catalog_sha256="b" * 64,
    )

    with pytest.raises(RuntimeError, match="differs from the pinned epoch"):
        serial_explore._reviewed_recipe_catalog_binding(config)


def test_sanitized_summaries_are_bounded_and_deinstantiated():
    summaries = [
        {
            "attempt_index": index,
            "outcome": "task_failed",
            "summary": (
                "Try a semantic side view. Reuse pixel row 12 col 99 from /tmp/secret."
            ),
        }
        for index in range(1, 12)
    ]
    result = sanitize_prior_attempt_summaries(summaries)
    assert len(result) == 8
    rendered = json.dumps(result)
    assert "pixel" not in rendered.lower()
    assert "/tmp" not in rendered
    assert len(rendered) <= 16_000


def test_initial_prior_summaries_enter_first_attempt_deinstantiated_and_bounded(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    initial = tuple(
        {
            "attempt_index": index,
            "outcome": "task_failed",
            "summary": (
                f"Instance {200 + index} used the left hand at pixel row 12. "
                f"Semantic lesson {index}: re-ground the visible control before acting."
            ),
        }
        for index in range(1, 12)
    )
    config = replace(
        _config(tmp_path, repo_root),
        initial_prior_summaries=initial,
    )
    dependencies, _calls, _dashboard = _dependencies([(True, False)])

    manifest = run_explore_job(config, dependencies=dependencies)

    summary_payload = json.loads(
        (
            config.output_root / "prior_attempt_summaries" / "attempt_001.json"
        ).read_text()
    )
    summaries = summary_payload["summaries"]
    assert len(summaries) == 8
    rendered = json.dumps(summaries, ensure_ascii=False)
    assert "instance" not in rendered.lower()
    assert "left" not in rendered.lower()
    assert "pixel" not in rendered.lower()
    assert len(rendered) <= 16_000
    canonical = json.dumps(
        summaries,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert manifest["protocol"]["inherited_prior_summaries"] == {
        "count": len(summaries),
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }
    assert set(manifest["protocol"]["inherited_prior_summaries"]) == {
        "count",
        "sha256",
    }


def test_default_initial_prior_summaries_leave_first_attempt_empty(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    config = _config(tmp_path, repo_root)
    dependencies, _calls, _dashboard = _dependencies([(True, False)])

    manifest = run_explore_job(config, dependencies=dependencies)

    assert config.initial_prior_summaries == ()
    summary_payload = json.loads(
        (
            config.output_root / "prior_attempt_summaries" / "attempt_001.json"
        ).read_text()
    )
    assert summary_payload["summaries"] == []
    assert manifest["protocol"]["inherited_prior_summaries"] == {
        "count": 0,
        "sha256": hashlib.sha256(b"[]").hexdigest(),
    }


def test_resume_preserves_manifest_prior_binding_without_reinjecting_config(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    original_prior = (
        {
            "attempt_index": 7,
            "outcome": "task_failed",
            "summary": "Re-ground the visible control before acting.",
        },
    )
    config = replace(
        _config(tmp_path, repo_root),
        initial_prior_summaries=original_prior,
        max_attempts=1,
    )
    first_dependencies, _calls, _dashboard = _dependencies([(False, False)])
    first_manifest = run_explore_job(config, dependencies=first_dependencies)
    original_binding = first_manifest["protocol"]["inherited_prior_summaries"]

    resume_config = replace(
        config,
        resume=True,
        max_attempts=2,
        initial_prior_summaries=(
            {
                "attempt_index": 99,
                "outcome": "task_failed",
                "summary": "This replacement must not be injected on resume.",
            },
        ),
    )
    resume_dependencies, _calls, _dashboard = _dependencies([(False, False)])
    resumed = run_explore_job(resume_config, dependencies=resume_dependencies)

    summary_payload = json.loads(
        (
            config.output_root / "prior_attempt_summaries" / "attempt_002.json"
        ).read_text()
    )
    assert [item["summary"] for item in summary_payload["summaries"]] == [
        "attempt 1 summary"
    ]
    assert resumed["protocol"]["inherited_prior_summaries"] == original_binding


def test_resume_accepts_current_v2_contract_and_rejects_unfinished_v1_job(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    config = replace(
        _config(tmp_path, repo_root),
        max_attempts=1,
    )
    dependencies, _calls, _dashboard = _dependencies([(False, False)])
    current = run_explore_job(config, dependencies=dependencies)

    serial_explore._validate_resume_manifest(config, current)
    assert current["protocol"]["public_tool_contract_version"] == 2
    assert current["protocol"]["public_primitives"][-1] == "navigate_to"

    legacy = json.loads(json.dumps(current))
    legacy["protocol"].pop("public_tool_contract_version")
    legacy["protocol"]["public_primitives"] = list(
        serial_explore.PUBLIC_TOOL_CONTRACTS[1]
    )
    with pytest.raises(
        RuntimeError,
        match="public-tool contract v1 Explore Jobs cannot resume",
    ):
        serial_explore._validate_resume_manifest(config, legacy)


def test_child_event_sink_relays_reasoning_tools_and_frames(tmp_path):
    attempt_dir = tmp_path / "attempt_001"
    sink = FileDashboardSink(attempt_dir / "dashboard_events.jsonl")
    sink.on_event({"type": "thinking", "text": "inspect the radio"})
    sink.on_event(
        {
            "type": "tool_call",
            "tool": "observe",
            "args": {"camera": "right_wrist"},
        }
    )
    sink.on_tool_start("observe", {"camera": "right_wrist"})
    sink.on_tool_result(
        "observe",
        {
            "camera": "right_wrist",
            "frame_id": "right_wrist:4:test",
            "total_env_steps": 4,
            "elapsed_s": 1.75,
            "_image_bytes": b"png-bytes",
        },
    )
    state = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(attempt_dir),
        video_path=str(attempt_dir / "episode.mp4"),
    )

    from robots.behavior.serial_explore import _AttemptDashboardRelay

    _AttemptDashboardRelay(attempt_dir, state)._drain()

    assert state.events_since(0)[0] == {
        "type": "thinking",
        "text": "inspect the radio",
    }
    assert state.frame("right_wrist") == b"png-bytes"
    assert state.run_detail()["frame_indices"]["right_wrist"] == 4
    assert state.run_detail()["timeline"][0]["action"] == "observe"
    assert state.run_detail()["timeline"][0]["elapsed_s"] == 1.75


def test_behavior_dashboard_serves_shared_ui_frame_aliases(tmp_path):
    behavior = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )

    behavior.on_frame("right_wrist", b"wrist-png", env_step=6)
    assert behavior.frame("agent") is None
    assert behavior.frame("camera") is None

    behavior.on_frame("head", b"head-png", env_step=7)
    assert behavior.frame("agent") == b"head-png"
    assert behavior.frame("camera") == b"head-png"
    assert behavior.frame("head") == b"head-png"
    assert behavior.frame("right_wrist") == b"wrist-png"
    assert behavior.run_detail()["frame_kinds"] == [
        "head",
        "left_wrist",
        "right_wrist",
    ]

    libero = State(
        run_id="libero/job",
        name=TAG,
        suite="libero",
        task=0,
        seed=0,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )
    libero.on_frame("agent", b"agent-png", env_step=1)
    libero.on_frame("camera", b"camera-png", env_step=2)
    assert libero.frame("agent") == b"agent-png"
    assert libero.frame("camera") == b"camera-png"


def test_child_event_relay_rejects_child_claims_of_parent_trust_state(tmp_path):
    attempt_dir = tmp_path / "attempt_001"
    sink = FileDashboardSink(attempt_dir / "dashboard_events.jsonl")
    sink.on_event(
        {
            "type": "official_success",
            "attempt_index": 1,
            "task_success": True,
        }
    )
    sink.on_event(
        {
            "type": "workflow_complete",
            "attempt_index": 1,
            "workflow_complete": True,
        }
    )
    sink.on_event(
        {
            "type": "publication_complete",
            "attempt_index": 1,
            "publication_complete": True,
        }
    )
    state = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(attempt_dir),
        video_path=str(attempt_dir / "episode.mp4"),
    )

    from robots.behavior.serial_explore import _AttemptDashboardRelay

    _AttemptDashboardRelay(attempt_dir, state)._drain()

    progress = state.run_info()["progress"]
    assert progress["official_task_success"] is False
    assert progress["workflow_complete"] is False
    assert progress["publication_complete"] is False


def test_child_progress_relay_copies_contained_literal_wrist_views(tmp_path):
    attempt_dir = tmp_path / "attempt_001"
    attempt_dir.mkdir()
    event_path = attempt_dir / "dashboard_events.jsonl"
    left_image = attempt_dir / "left.png"
    right_image = attempt_dir / "right.png"
    outside_image = tmp_path / "outside.png"
    left_image.write_bytes(b"left")
    right_image.write_bytes(b"right")
    outside_image.write_bytes(b"outside")
    sink = FileDashboardSink(event_path)

    sink.on_tool_progress(
        "pi0_nav_pick",
        {
            "env_step": 32,
            "visual_review": {
                "views": {
                    "left_wrist": {
                        "frame_id": "left_wrist:32:left",
                        "path": str(left_image),
                    },
                    "right_wrist": {
                        "frame_id": "right_wrist:32:right",
                        "path": str(right_image),
                    },
                    "head": {
                        "frame_id": "head:32:outside",
                        "path": str(outside_image),
                    },
                }
            },
        },
    )
    rendered = event_path.read_text(encoding="utf-8")
    assert str(left_image) not in rendered
    assert str(right_image) not in rendered
    assert str(outside_image) not in rendered

    state = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(attempt_dir),
        video_path=str(attempt_dir / "episode.mp4"),
    )

    from robots.behavior.serial_explore import _AttemptDashboardRelay

    _AttemptDashboardRelay(attempt_dir, state)._drain()

    assert state.frame("left_wrist") == b"left"
    assert state.frame("right_wrist") == b"right"
    assert state.frame("head") is None


def test_child_relay_uses_only_hashed_frame_channel_for_checkpoint_images(
    tmp_path,
):
    attempt_dir = tmp_path / "attempt_001"
    event_path = attempt_dir / "dashboard_events.jsonl"
    sink = FileDashboardSink(event_path)
    trusted = {
        camera: f"trusted-{camera}".encode()
        for camera in ("head", "left_wrist", "right_wrist")
    }
    sink.on_tool_result(
        "save_robot_state_checkpoint",
        {
            "env_step": 8,
            "images": {
                camera: {
                    "_image_bytes": image,
                    "rgb_path": f"/outside/{camera}.png",
                }
                for camera, image in trusted.items()
            },
        },
    )
    assert "/outside/" not in event_path.read_text(encoding="utf-8")
    external = tmp_path / "external.png"
    external.write_bytes(b"untrusted-external")
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "channel": "tool_result",
                    "payload": {
                        "name": "save_robot_state_checkpoint",
                        "result": {
                            "env_step": 8,
                            "images": {
                                camera: {"rgb_path": str(external)}
                                for camera in trusted
                            },
                        },
                    },
                }
            )
            + "\n"
        )

    state = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(attempt_dir),
        video_path=str(attempt_dir / "episode.mp4"),
    )

    from robots.behavior.serial_explore import _AttemptDashboardRelay

    _AttemptDashboardRelay(attempt_dir, state)._drain()

    for camera, image in trusted.items():
        assert state.frame(camera) == image
        assert state.run_detail()["frame_indices"][camera] == 8
    assert state.run_detail()["frame_idx"] == 2


def test_child_relay_rejects_unbound_frame_paths(tmp_path):
    attempt_dir = tmp_path / "attempt_001"
    attempt_dir.mkdir()
    trusted = attempt_dir / "trusted.png"
    trusted.write_bytes(b"trusted")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (attempt_dir / "linked.png").symlink_to(outside)
    state = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(attempt_dir),
        video_path=str(attempt_dir / "episode.mp4"),
    )

    from robots.behavior.serial_explore import _AttemptDashboardRelay

    relay = _AttemptDashboardRelay(attempt_dir, state)
    for relative_path, digest in (
        (str(outside), hashlib.sha256(outside.read_bytes()).hexdigest()),
        ("../outside.png", hashlib.sha256(outside.read_bytes()).hexdigest()),
        ("linked.png", hashlib.sha256(outside.read_bytes()).hexdigest()),
        ("trusted.png", "0" * 64),
    ):
        relay._relay_frame(
            {
                "camera": "head",
                "env_step": 1,
                "relative_path": relative_path,
                "sha256": digest,
            }
        )

    assert state.frame("head") is None
    assert state.run_detail()["frame_idx"] == -1


def test_child_relay_sanitizes_legacy_fallback_paths(tmp_path):
    attempt_dir = tmp_path / "attempt_001"
    states_dir = attempt_dir / "vla_calls" / "call_001"
    states_dir.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside-fallback")
    (states_dir / "pi0_nav_pick_states.json").write_text(
        json.dumps(
            [
                {
                    "chunk": 1,
                    "total_env_steps": 32,
                    "pi0_nav_pick_monitor": {
                        "visual_review": {"views": {"head": {"rgb_path": str(outside)}}}
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (attempt_dir / "behavior_tool_trace.jsonl").write_text(
        json.dumps(
            {
                "tool": "observe",
                "input": {"camera": "head"},
                "result": {
                    "env_step": 32,
                    "images": {"head": {"rgb_path": str(outside)}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = State(
        run_id="behavior/job",
        name=TAG,
        suite="behavior_2025_challenge",
        task=0,
        seed=0,
        output_dir=str(attempt_dir),
        video_path=str(attempt_dir / "episode.mp4"),
    )

    from robots.behavior.serial_explore import _AttemptDashboardRelay

    _AttemptDashboardRelay(attempt_dir, state)._drain()

    assert state.frame("head") is None
    assert state.run_detail()["frame_idx"] == -1


def test_failure_then_success_uses_fresh_children_and_one_persistent_vla(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, dashboard = _dependencies([(False, False), (True, True)])
    config = _config(tmp_path, repo_root)

    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["status"] == "succeeded"
    assert manifest["task_success"] is True
    # Artifact sealing is an independent best-effort axis and never a success gate.
    assert manifest["artifact_seal_complete"] is True
    assert manifest["workflow_complete"] is True
    assert manifest["publication_complete"] is True
    assert manifest["protocol"]["public_primitives"] == [
        "pi0_nav_pick",
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "close",
        "open",
        "press",
        "save_robot_state_checkpoint",
        "navigate_to",
    ]
    assert manifest["protocol"]["public_tool_contract_version"] == 2
    assert manifest["protocol"]["agent_finish_registered"] is False
    assert manifest["protocol"]["task_spec"]["prompt_profile_id"] == "turning_on_radio"
    assert manifest["attempts"][0]["prompt"]["profile_id"] == "turning_on_radio"
    assert manifest["attempts"][1]["prompt"]["profile_id"] == "turning_on_radio"
    assert manifest["cumulative"]["attempts"] == 2
    assert calls.vla_starts == 1
    assert len(calls.attempts) == 2
    assert calls.attempts[0][1] != calls.attempts[1][1]
    assert len(set(calls.bindings)) == 2
    root = config.output_root
    assert (
        json.loads(
            (root / "policy_checkpoint_binding.json").read_text(encoding="utf-8")
        )
        == config.policy_checkpoint_binding
    )
    failed = root / "attempts" / TAG / "attempt_001_failed.json"
    assert failed.is_file()
    summary_payload = json.loads(
        (root / "prior_attempt_summaries" / "attempt_002.json").read_text()
    )
    assert summary_payload["summaries"][0]["summary"] == "attempt 1 summary"
    assert (root / f"recipe_{TAG}.jsonl").is_file()
    assert (root / "memory" / "turning_on_radio.md").is_file()
    assert dashboard.started == [1, 2]
    assert dashboard.done == [True]


def test_zero_based_action_trace_success_stops_without_retry(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, dashboard = _dependencies([(True, True)])
    original = dependencies.run_attempt

    def write_current_action_trace(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        receipt = json.loads(
            (output_dir / "official_success_receipt.json").read_text(encoding="utf-8")
        )
        (output_dir / "behavior_action_trace.jsonl").write_text(
            json.dumps(
                {
                    "event": "step",
                    "env_idx": 0,
                    "step": receipt["env_step"] - 1,
                    "info_done": {"success": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return execution

    dependencies.run_attempt = write_current_action_trace
    manifest = run_explore_job(
        replace(_config(tmp_path, repo_root), max_attempts=2),
        dependencies=dependencies,
    )

    assert manifest["status"] == "succeeded"
    assert manifest["task_success"] is True
    assert len(calls.attempts) == 1
    assert manifest["attempts"][0]["outcome"] == "official_success"
    assert manifest["attempts"][0]["publication_eligible"] is True
    assert dashboard.done == [True]


def test_raw_success_with_exact_artifact_defect_stops_without_publication(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, dashboard = _dependencies([(True, True)])
    original = dependencies.run_attempt

    def add_incomplete_exact_artifact(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        (output_dir / "vla_calls" / "call_001").mkdir(parents=True)
        return execution

    dependencies.run_attempt = add_incomplete_exact_artifact
    config = replace(_config(tmp_path, repo_root), max_attempts=2)
    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["status"] == "succeeded"
    assert manifest["task_success"] is True
    assert manifest["artifact_seal_complete"] is False
    assert manifest["workflow_complete"] is False
    assert manifest["publication_complete"] is False
    assert len(calls.attempts) == 1
    assert manifest["attempts"][0]["outcome"] == "official_success"
    assert manifest["attempts"][0]["exact_pi0_artifacts_valid"] is False
    assert manifest["attempts"][0]["publication_eligible"] is False
    assert not (config.output_root / f"recipe_{TAG}.jsonl").exists()
    assert dashboard.done == [True]


@pytest.mark.parametrize(
    ("field_path", "tampered_value"),
    (
        (("task", "task_name"), "picking_up_trash"),
        (("task", "task"), 1),
        (("protocol", "task_spec", "activity_definition_id"), 1),
        (("protocol", "task_identity", "activity_definition_id"), False),
        (("native_binding", "activity_instance_id"), 67),
        (("native_binding", "env_seed"), 1),
        (("protocol", "public_seed"), 1),
        (("protocol", "recipe_tag"), "turning_on_radio_s1"),
        (("protocol", "prompt", "profile_id"), "picking_up_trash"),
    ),
    ids=(
        "task-name",
        "task-index",
        "activity-definition",
        "bool-activity-definition",
        "native-instance",
        "native-env-seed",
        "public-seed",
        "recipe-tag",
        "prompt-profile",
    ),
)
def test_raw_success_with_tampered_attempt_identity_blocks_without_retry(
    tmp_path,
    field_path,
    tampered_value,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, dashboard = _dependencies([(True, True)])
    original = dependencies.run_attempt

    def tamper_attempt_identity(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        manifest_path = output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cursor = manifest
        for field in field_path[:-1]:
            cursor = cursor[field]
        cursor[field_path[-1]] = tampered_value
        _write_json(manifest_path, manifest)
        return execution

    dependencies.run_attempt = tamper_attempt_identity
    config = replace(_config(tmp_path, repo_root), max_attempts=2)
    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["status"] == "blocked"
    assert manifest["task_success"] is None
    assert manifest["artifact_seal_complete"] is False
    assert manifest["publication_complete"] is False
    assert manifest["blocked_reason"] == "attempt_task_identity_mismatch"
    assert len(calls.attempts) == 1
    assert manifest["attempts"][0]["outcome"] == "run_error"
    assert manifest["attempts"][0]["task_success"] is None
    assert manifest["attempts"][0]["attempt_identity_valid"] is False
    assert not (config.output_root / f"recipe_{TAG}.jsonl").exists()
    assert dashboard.done == []


def test_attempt_resource_source_file_drift_blocks_before_success_publication(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, True)])
    original = dependencies.run_attempt
    config = _config(tmp_path, repo_root)

    def tamper_source_file(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        source_path = config.output_root / "resource_source.json"
        source = json.loads(source_path.read_text())
        source["requested_revision"] = "tampered"
        _write_json(source_path, source)
        return execution

    dependencies.run_attempt = tamper_source_file
    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["status"] == "blocked"
    assert manifest["task_success"] is None
    assert "Job resource source differs" in manifest["blocked_reason"]
    assert not (config.output_root / f"recipe_{TAG}.jsonl").exists()


def test_child_resource_source_mismatch_blocks_before_success_publication(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, True)])
    original = dependencies.run_attempt
    config = _config(tmp_path, repo_root)

    def tamper_child_manifest(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        manifest_path = output_dir / "run_manifest.json"
        child = json.loads(manifest_path.read_text())
        child["resource_source"]["requested_revision"] = "tampered"
        _write_json(manifest_path, child)
        return execution

    dependencies.run_attempt = tamper_child_manifest
    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["status"] == "blocked"
    assert manifest["task_success"] is None
    assert "attempt resource source differs" in manifest["blocked_reason"]
    assert not (config.output_root / f"recipe_{TAG}.jsonl").exists()


def test_raw_success_without_artifact_seal_stops_and_publishes_symbolic_inputs(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)

    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["task_success"] is True
    assert manifest["artifact_seal_complete"] is False
    assert manifest["workflow_complete"] is False
    assert manifest["publication_complete"] is True
    assert len(calls.attempts) == 1
    assert (config.output_root / f"recipe_{TAG}.jsonl").is_file()
    provenance = json.loads(
        (config.output_root / "memory" / "turning_on_radio_provenance.json").read_text()
    )
    assert provenance["source"] == "raw_official_success_v1"
    assert provenance["official_success_receipt"]["attempt_index"] == 1
    assert set(provenance["source_artifacts_sha256"]) == {
        "official_success_receipt",
        "behavior_action_trace",
        "behavior_tool_trace",
        "final_result",
        "run_manifest",
        "session_manifest",
    }


def test_offline_publication_is_idempotent_and_preserves_original_attempt(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    manifest = run_explore_job(config, dependencies=dependencies)
    assert manifest["task_success"] is True

    root = config.output_root
    attempt = root / "attempts" / TAG / "attempt_001"
    original = {
        path.relative_to(attempt): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in attempt.rglob("*")
        if path.is_file()
    }
    (root / f"recipe_{TAG}.jsonl").unlink()
    shutil.rmtree(root / "memory")

    first = publish_existing_success(root)
    second = publish_existing_success(root)

    assert first == second
    assert first["task_success"] is True
    assert first["artifact_seal_complete"] is False
    assert first["publication_source"] == "raw_official_success_v1"
    assert first["original_attempt_immutable"] is True
    assert (root / "publication_amendment.json").is_file()
    after = {
        path.relative_to(attempt): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in attempt.rglob("*")
        if path.is_file()
    }
    assert after == original


def _remove_promoted_publication(root: Path) -> None:
    (root / f"recipe_{TAG}.jsonl").unlink(missing_ok=True)
    shutil.rmtree(root / "memory", ignore_errors=True)
    (root / "publication_amendment.json").unlink(missing_ok=True)


def _forensic_job(
    tmp_path: Path,
    records: list[dict],
    *,
    receipt: dict | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "historical-job"
    tag = "picking_up_trash_s1"
    attempt = root / "attempts" / tag / "attempt_001"
    attempt.mkdir(parents=True)
    _write_json(
        root / "session_manifest.json",
        {
            "schema_version": 1,
            "job_id": "historical-job-id",
            "status": "stopped_by_operator",
            "protocol": {
                "task_name": "picking_up_trash",
                "public_seed": 1,
                "recipe_tag": tag,
            },
            "attempts": [
                {
                    "attempt_index": 1,
                    "outcome": "task_failed",
                    "task_success": False,
                    "summary": "stale lifecycle result",
                    "terminal_failure": {"condition": "stale"},
                    "artifact_seal_complete": True,
                    "publication_eligible": True,
                    "output_dir": str(attempt),
                }
            ],
            "task_success": None,
            "workflow_complete": False,
            "terminal_receipt_complete": False,
            "artifact_seal_complete": True,
            "publication_complete": True,
        },
    )
    for name in ("final_result.json", "behavior_result.json", f"{tag}.json"):
        _write_json(
            attempt / name,
            {
                "task_success": False,
                "success": False,
                "marker": name,
            },
        )
    (attempt / "behavior_action_trace.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (attempt / "behavior_tool_trace.jsonl").write_bytes(b"immutable tool trace\n")
    (attempt / "episode.mp4").write_bytes(b"immutable video")
    _write_json(
        attempt / "terminal_failure_receipt.json",
        {"immutable": "terminal receipt"},
    )
    exact_evidence = attempt / "vla_calls" / "call_001"
    exact_evidence.mkdir(parents=True)
    (exact_evidence / "pi0_nav_pick_call.json").write_bytes(b"immutable exact N")
    if receipt is not None:
        _write_json(attempt / "official_success_receipt.json", receipt)
    failed_root = attempt.parent
    _write_json(failed_root / "attempt_001_failed.json", {"stale": True})
    (failed_root / "attempt_001_failed.jsonl").write_text(
        "stale failed archive\n",
        encoding="utf-8",
    )
    return root, attempt


def _forensic_immutable_hashes(attempt: Path) -> dict[str, str]:
    names = (
        "behavior_action_trace.jsonl",
        "behavior_tool_trace.jsonl",
        "episode.mp4",
        "terminal_failure_receipt.json",
        "vla_calls/call_001/pi0_nav_pick_call.json",
    )
    return {
        name: hashlib.sha256((attempt / name).read_bytes()).hexdigest()
        for name in names
    }


def _forensic_summary_bytes(root: Path, attempt: Path) -> dict[str, bytes]:
    paths = (
        root / "session_manifest.json",
        attempt / "final_result.json",
        attempt / "behavior_result.json",
        attempt / "picking_up_trash_s1.json",
        attempt.parent / "attempt_001_failed.json",
        attempt.parent / "attempt_001_failed.jsonl",
    )
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in paths
        if path.exists()
    }


def test_correct_existing_success_from_single_trace_line_without_receipt(
    monkeypatch,
    tmp_path,
):
    root, attempt = _forensic_job(
        tmp_path,
        [{"step": 17, "info_done": {"success": True}}],
    )
    immutable_before = _forensic_immutable_hashes(attempt)
    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        lambda *_args: SimpleNamespace(complete=False),
    )

    corrected = correct_existing_success(root, task_name="picking_up_trash")

    receipt_path = attempt / "official_success_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = dict(receipt)
    claimed_self_hash = unsigned.pop("receipt_sha256")
    assert (
        claimed_self_hash
        == hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
    )
    receipt_file_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    binding = corrected["official_success_binding"]
    assert binding == {
        "source": "behavior_action_trace",
        "field_path": "info_done.success",
        "first_success_step": 17,
        "success_interval": [17, 17],
        "success_count": 1,
        "success_later_reverted": False,
        "last_success_step": 17,
        "last_trace_step": 17,
        "final_trace_success": True,
        "action_trace_sha256": immutable_before["behavior_action_trace.jsonl"],
        "receipt_sha256": receipt_file_sha256,
        "notes": [],
    }
    assert receipt["receipt_type"] == "forensic_action_trace_receipt"
    assert receipt["job_id"] == "historical-job-id"
    assert receipt["attempt_index"] == 1
    assert receipt["field_path"] == "info_done.success"
    assert receipt["first_success_step"] == 17
    assert corrected["status"] == "succeeded"
    assert corrected["task_success"] is True
    assert corrected["workflow_complete"] is False
    assert corrected["terminal_receipt_complete"] is False
    assert corrected["artifact_seal_complete"] is True
    assert corrected["publication_complete"] is False
    attempt_record = corrected["attempts"][0]
    assert attempt_record["outcome"] == "official_success"
    assert attempt_record["task_success"] is True
    assert (
        attempt_record["summary"]
        == "Official raw success confirmed from bound action trace."
    )
    assert attempt_record["terminal_failure"] is None
    assert attempt_record["artifact_seal_complete"] is True
    assert attempt_record["publication_eligible"] is False
    for name in (
        "final_result.json",
        "behavior_result.json",
        "picking_up_trash_s1.json",
    ):
        summary = json.loads((attempt / name).read_text(encoding="utf-8"))
        assert summary["task_success"] is True
        assert summary["success"] is True
        assert summary["official_success_source"] == "behavior_action_trace"
        assert summary["official_success_binding"] == binding
        assert summary["official_success_receipt"] == receipt
    assert _forensic_immutable_hashes(attempt) == immutable_before
    assert not (root / "forensic_correction.json").exists()
    assert not (attempt.parent / "attempt_001_failed.json").exists()
    assert not (attempt.parent / "attempt_001_failed.jsonl").exists()


@pytest.mark.parametrize("malicious_kind", ("symlink", "directory"))
def test_correct_existing_success_rejects_preexisting_legacy_temp_without_writes(
    monkeypatch,
    tmp_path,
    malicious_kind,
):
    root, attempt = _forensic_job(
        tmp_path,
        [{"step": 17, "info_done": {"success": True}}],
    )
    summaries_before = _forensic_summary_bytes(root, attempt)
    immutable_before = _forensic_immutable_hashes(attempt)
    legacy_temp = root / "session_manifest.json.tmp"
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside must remain unchanged")
    if malicious_kind == "symlink":
        legacy_temp.symlink_to(outside)
    else:
        legacy_temp.mkdir()
    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        lambda *_args: SimpleNamespace(complete=False),
    )

    with pytest.raises(RuntimeError, match="pre-existing legacy temp"):
        correct_existing_success(root)

    assert outside.read_bytes() == b"outside must remain unchanged"
    assert _forensic_summary_bytes(root, attempt) == summaries_before
    assert _forensic_immutable_hashes(attempt) == immutable_before
    assert not (attempt / "official_success_receipt.json").exists()
    assert (
        legacy_temp.is_symlink()
        if malicious_kind == "symlink"
        else legacy_temp.is_dir()
    )


def test_correct_existing_success_detects_trace_mutation_before_any_write(
    monkeypatch,
    tmp_path,
):
    root, attempt = _forensic_job(
        tmp_path,
        [{"step": 17, "info_done": {"success": True}}],
    )
    summaries_before = _forensic_summary_bytes(root, attempt)
    immutable_before = _forensic_immutable_hashes(attempt)
    action_path = attempt / "behavior_action_trace.jsonl"

    def mutate_trace(*_args):
        action_path.write_text(
            json.dumps({"step": 18, "info_done": {"success": False}}) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(complete=False)

    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        mutate_trace,
    )

    with pytest.raises(RuntimeError, match="action trace changed"):
        correct_existing_success(root)

    assert _forensic_summary_bytes(root, attempt) == summaries_before
    assert not (attempt / "official_success_receipt.json").exists()
    assert not (root / "forensic_correction.json").exists()
    for name, digest in immutable_before.items():
        if name == "behavior_action_trace.jsonl":
            continue
        assert hashlib.sha256((attempt / name).read_bytes()).hexdigest() == digest


def test_correct_existing_success_rechecks_trace_before_failed_archive_deletion(
    monkeypatch,
    tmp_path,
):
    root, attempt = _forensic_job(
        tmp_path,
        [{"step": 17, "info_done": {"success": True}}],
    )
    action_path = attempt / "behavior_action_trace.jsonl"
    original_writer = serial_explore._forensic_atomic_json
    writes = 0

    def mutate_after_summary_commit(root_arg, path, payload):
        nonlocal writes
        original_writer(root_arg, path, payload)
        writes += 1
        if path == root / "session_manifest.json":
            action_path.write_text(
                json.dumps({"step": 18, "info_done": {"success": False}}) + "\n",
                encoding="utf-8",
            )

    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        lambda *_args: SimpleNamespace(complete=False),
    )
    monkeypatch.setattr(
        serial_explore,
        "_forensic_atomic_json",
        mutate_after_summary_commit,
    )

    with pytest.raises(RuntimeError, match="action trace changed"):
        correct_existing_success(root)

    assert writes == 5
    assert (attempt.parent / "attempt_001_failed.json").is_file()
    assert (attempt.parent / "attempt_001_failed.jsonl").is_file()


def test_correct_existing_success_preserves_receipt_and_latches_after_reversion(
    monkeypatch,
    tmp_path,
):
    existing_receipt = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "job_id": "historical-job-id",
        "attempt_index": 1,
        "marker": "preserve exactly",
    }
    root, attempt = _forensic_job(
        tmp_path,
        [
            {"step": 8, "info_done": {"success": True}},
            {"step": 7, "info_done": {"success": True}},
            {"step": 9, "info_done": {"success": False}},
        ],
        receipt=existing_receipt,
    )
    receipt_path = attempt / "official_success_receipt.json"
    receipt_before = receipt_path.read_bytes()
    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        lambda *_args: SimpleNamespace(complete=True),
    )

    corrected = correct_existing_success(root)

    binding = corrected["official_success_binding"]
    assert corrected["task_success"] is True
    assert corrected["publication_complete"] is True
    assert corrected["attempts"][0]["publication_eligible"] is True
    assert binding["success_count"] == 2
    assert binding["success_later_reverted"] is True
    assert binding["final_trace_success"] is False
    assert "non_monotonic_step_records=1" in binding["notes"]
    assert receipt_path.read_bytes() == receipt_before
    assert binding["receipt_sha256"] == hashlib.sha256(receipt_before).hexdigest()
    assert corrected["official_success_receipt"] == existing_receipt


def test_correct_existing_success_ignores_legacy_info_done_fallback(
    monkeypatch,
    tmp_path,
):
    root, attempt = _forensic_job(
        tmp_path,
        [{"step": 17, "info": {"done": {"success": True}}}],
    )
    manifest_before = (root / "session_manifest.json").read_bytes()
    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        lambda *_args: SimpleNamespace(complete=True),
    )

    with pytest.raises(RuntimeError, match="exactly one canonical successful"):
        correct_existing_success(root)

    assert (root / "session_manifest.json").read_bytes() == manifest_before
    assert not (attempt / "official_success_receipt.json").exists()


def test_correct_existing_success_rejects_multiple_successful_attempts(
    monkeypatch,
    tmp_path,
):
    root, first_attempt = _forensic_job(
        tmp_path,
        [{"step": 1, "info_done": {"success": True}}],
    )
    second_attempt = first_attempt.parent / "attempt_002"
    second_attempt.mkdir()
    (second_attempt / "behavior_action_trace.jsonl").write_text(
        json.dumps({"step": 2, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    manifest_before = (root / "session_manifest.json").read_bytes()
    monkeypatch.setattr(
        serial_explore,
        "validate_forensic_publication_binding",
        lambda *_args: SimpleNamespace(complete=False),
    )

    with pytest.raises(RuntimeError, match="exactly one canonical successful"):
        correct_existing_success(root)

    assert (root / "session_manifest.json").read_bytes() == manifest_before
    assert not (first_attempt / "official_success_receipt.json").exists()


def test_correct_and_publish_existing_success_cli_modes_are_mutually_exclusive(
    tmp_path,
):
    with pytest.raises(SystemExit):
        serial_explore._parse_args(
            [
                "--output-root",
                str(tmp_path),
                "--correct-existing-success",
                "--publish-existing-success",
            ]
        )


def test_correct_existing_success_cli_dispatches_without_starting_runtime(
    monkeypatch,
    tmp_path,
):
    calls = []

    def correct(root, *, task_name):
        calls.append((Path(root), task_name))
        return {"status": "succeeded", "task_success": True}

    monkeypatch.setattr(serial_explore, "correct_existing_success", correct)

    exit_code = serial_explore.main(
        [
            "--output-root",
            str(tmp_path),
            "--task-name",
            "picking_up_trash",
            "--correct-existing-success",
        ]
    )

    assert exit_code == 0
    assert calls == [(tmp_path, "picking_up_trash")]


def test_offline_publication_rejects_unverified_success_source(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    run_explore_job(config, dependencies=dependencies)
    root = config.output_root
    attempt = root / "attempts" / TAG / "attempt_001"
    _remove_promoted_publication(root)

    for name in ("final_result.json", f"{TAG}.json"):
        path = attempt / name
        value = json.loads(path.read_text())
        value["official_success_source"] = "agent_claim"
        _write_json(path, value)
    trace_path = attempt / "behavior_tool_trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    records[0]["result"]["official_success_source"] = "agent_claim"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="verified runtime raw success"):
        publish_existing_success(root)
    assert not (root / f"recipe_{TAG}.jsonl").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("runtime_cleanup", "pending", "cleanup binding mismatch"),
        ("job_id", "different-job", "cleanup binding mismatch"),
    ],
)
def test_offline_publication_rejects_incomplete_cleanup_or_bad_job_binding(
    tmp_path,
    field,
    value,
    message,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    run_explore_job(config, dependencies=dependencies)
    root = config.output_root
    attempt = root / "attempts" / TAG / "attempt_001"
    _remove_promoted_publication(root)

    path = attempt / "final_result.json"
    final = json.loads(path.read_text())
    if field == "job_id":
        final["job"]["job_id"] = value
    else:
        final[field] = value
    _write_json(path, final)

    with pytest.raises(RuntimeError, match=message):
        publish_existing_success(root)
    assert not (root / f"recipe_{TAG}.jsonl").exists()


def test_offline_publication_rejects_symlinked_memory_target(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    run_explore_job(config, dependencies=dependencies)
    root = config.output_root
    _remove_promoted_publication(root)
    outside = tmp_path / "outside-publication"
    outside.mkdir()
    (root / "memory").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        publish_existing_success(root)
    assert list(outside.iterdir()) == []


def test_offline_publication_rejects_nonce_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    run_explore_job(config, dependencies=dependencies)
    root = config.output_root
    attempt = root / "attempts" / TAG / "attempt_001"
    _remove_promoted_publication(root)
    trace_path = attempt / "behavior_tool_trace.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    records[0]["result"]["attempt_nonce"] = "different-attempt"
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="bound raw-success receipt"):
        publish_existing_success(root)


@pytest.mark.parametrize(
    "group_check",
    ["_manifest_owned_groups", "_manifest_unverified_groups"],
)
def test_offline_publication_rejects_live_or_ambiguous_process_group(
    monkeypatch,
    tmp_path,
    group_check,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    run_explore_job(config, dependencies=dependencies)
    root = config.output_root
    _remove_promoted_publication(root)
    monkeypatch.setattr(
        serial_explore,
        group_check,
        lambda _attempt_dir: {"env": (12345,)},
    )

    with pytest.raises(RuntimeError, match="owns or ambiguously references"):
        publish_existing_success(root)


def test_offline_publication_rejects_source_change_during_derivation(
    monkeypatch,
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, False)])
    config = _config(tmp_path, repo_root)
    run_explore_job(config, dependencies=dependencies)
    root = config.output_root
    _remove_promoted_publication(root)
    manifest_path = root / "session_manifest.json"
    original_publish = serial_explore.BehaviorToolkit._publish_task_memory

    def mutate_source(*args, **kwargs):
        original_publish(*args, **kwargs)
        manifest = json.loads(manifest_path.read_text())
        manifest["concurrent_mutation"] = True
        _write_json(manifest_path, manifest)

    monkeypatch.setattr(
        serial_explore.BehaviorToolkit,
        "_publish_task_memory",
        mutate_source,
    )

    with pytest.raises(RuntimeError, match="source changed during derivation"):
        publish_existing_success(root)
    assert not (root / "publication_amendment.json").exists()


def test_symbolic_publication_rejects_absolute_path_hidden_in_text() -> None:
    recipe = (
        json.dumps(
            {
                "kind": "task_level_symbolic_recipe",
                "note": "read /home/ubuntu/private/runtime.json",
            }
        )
        + "\n"
    ).encode()

    with pytest.raises(RuntimeError, match="absolute runtime path"):
        serial_explore._validate_symbolic_publication(recipe, b"safe memory")


@pytest.mark.parametrize(
    ("recipe_note", "memory"),
    [
        ("First collect evidence, then act.", b"safe memory"),
        ("safe semantic invariant", b"Subsequently verify the result."),
        ("Step 2 refines the target.", b"safe memory"),
    ],
)
def test_symbolic_publication_rejects_generic_sequence_steering(
    recipe_note: str,
    memory: bytes,
) -> None:
    recipe = (
        json.dumps(
            {
                "kind": "task_level_symbolic_recipe",
                "note": recipe_note,
            }
        )
        + "\n"
    ).encode()

    with pytest.raises(RuntimeError, match="tool, order, or spatial steering"):
        serial_explore._validate_symbolic_publication(recipe, memory)


def test_job_has_no_hidden_attempt_cap(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, _dashboard = _dependencies(
        [(False, False)] * 12 + [(True, False)]
    )
    manifest = run_explore_job(_config(tmp_path, repo_root), dependencies=dependencies)

    assert manifest["task_success"] is True
    assert manifest["cumulative"]["attempts"] == 13
    assert len(calls.attempts) == 13


def test_unverified_attempt_process_group_blocks_job(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, _dashboard = _dependencies([(False, False)])
    original = dependencies.run_attempt

    def ambiguous_attempt(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        execution.ambiguous_groups = {"env": (12345,)}
        return execution

    dependencies.run_attempt = ambiguous_attempt
    manifest = run_explore_job(_config(tmp_path, repo_root), dependencies=dependencies)

    assert manifest["status"] == "blocked"
    assert manifest["task_success"] is None
    assert manifest["blocked_reason"] == "attempt_process_ownership_unverified"
    assert len(calls.attempts) == 1


def test_dead_persistent_vla_blocks_without_rehashing_checkpoint(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, _dashboard = _dependencies([(False, False), (True, False)])
    processes: list[_FakeProcess] = []

    def start_vla(config, output_dir):
        del config, output_dir
        process = _FakeProcess()
        processes.append(process)
        calls.vla_starts += 1
        return "http://127.0.0.1:9123", process

    original = dependencies.run_attempt

    def kill_after_first(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        if len(calls.attempts) == 1:
            processes[-1].returncode = 17
        return execution

    dependencies.start_vla = start_vla
    dependencies.stop_vla = lambda process: setattr(process, "returncode", 0)
    dependencies.run_attempt = kill_after_first
    manifest = run_explore_job(_config(tmp_path, repo_root), dependencies=dependencies)

    assert manifest["status"] == "blocked"
    assert manifest["task_success"] is None
    assert manifest["blocked_reason"] == "persistent_vla_exited"
    assert manifest["attempts"][0]["task_success"] is None
    assert manifest["attempts"][0]["outcome"] == "run_error"
    assert calls.vla_starts == 1
    assert len(processes) == 1


def test_borrowed_campaign_vla_is_recorded_but_never_stopped(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, _calls, _dashboard = _dependencies([(True, True)])
    process = _FakeProcess()
    stop_calls = []
    dependencies.start_vla = lambda _config, _output_dir: (
        "http://127.0.0.1:9123",
        process,
    )
    dependencies.stop_vla = lambda proc: stop_calls.append(proc)
    dependencies.owns_vla = False

    manifest = run_explore_job(_config(tmp_path, repo_root), dependencies=dependencies)

    assert manifest["task_success"] is True
    assert manifest["processes"]["vla"]["managed"] is False
    assert manifest["processes"]["vla"]["ownership"] == "borrowed"
    assert process.poll() is None
    assert stop_calls == []


def test_timeout_without_child_audit_remains_outcome_unknown(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, _dashboard = _dependencies([(False, False), (True, False)])
    original = dependencies.run_attempt

    def timeout_first(argv, output_dir, log_path, timeout_s):
        execution = original(argv, output_dir, log_path, timeout_s)
        if len(calls.attempts) == 1:
            (output_dir / "final_result.json").unlink()
            (output_dir / f"{TAG}.json").unlink()
            execution.final_result = None
            execution.timed_out = True
            execution.exit_code = -15
        return execution

    dependencies.run_attempt = timeout_first
    config = _config(tmp_path, repo_root)
    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["task_success"] is True
    first = manifest["attempts"][0]
    assert first["task_success"] is None
    assert first["timed_out"] is True
    assert not (
        config.output_root / "attempts" / TAG / "attempt_001_failed.json"
    ).exists()


def test_unverifiable_success_is_never_accepted(tmp_path):
    untrusted = {
        "task_success": True,
        "workflow_complete": True,
        "official_success_source": "agent_claim",
    }
    resolved = _runtime_attempt_result(tmp_path, untrusted)
    assert resolved is not None
    assert resolved["task_success"] is False


def test_terminal_failure_binding_requires_hashed_visuals_and_no_raw_success(
    tmp_path,
):
    attempt_dir = tmp_path / "attempt_001"
    checkpoint_id = "visual_checkpoint_001"
    checkpoint_dir = attempt_dir / "visual_checkpoints" / checkpoint_id
    checkpoint_dir.mkdir(parents=True)
    cameras = {}
    image_hashes = {}
    for camera in ("head", "left_wrist", "right_wrist"):
        rgb_path = checkpoint_dir / f"{camera}_rgb.png"
        depth_path = checkpoint_dir / f"{camera}_depth.png"
        rgb_path.write_bytes(f"{camera}-rgb".encode())
        depth_path.write_bytes(f"{camera}-depth".encode())
        cameras[camera] = {
            "camera": camera,
            "frame_id": ("head:evidence" if camera == "head" else f"{camera}:fresh"),
            "capture_group_id": "evidence-capture",
            "capture_env_step": 17,
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
        }
        image_hashes[camera] = {
            "rgb": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            "depth": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
        }
    terminal_declaration = {
        "camera": "head",
        "cause": "dropped_out_of_gripper",
        "condition": "radio_tipped_flat",
        "frame_id": "head:evidence",
    }
    metadata = {
        "schema_version": 1,
        "kind": "visual_checkpoint",
        "visual_checkpoint_id": checkpoint_id,
        "capture_group_id": "evidence-capture",
        "env_step": 17,
        "cameras": cameras,
        "terminal_failure": terminal_declaration,
    }
    metadata_path = checkpoint_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    receipt = {
        "schema_version": 1,
        "source": "llm_fresh_visual_observation",
        "condition": "radio_tipped_flat",
        "cause": "dropped_out_of_gripper",
        "camera": "head",
        "frame_id": "head:evidence",
        "capture_group_id": "evidence-capture",
        "env_step": 17,
        "visual_checkpoint_id": checkpoint_id,
        "visual_checkpoint_capture_group_id": "evidence-capture",
        "visual_checkpoint_metadata_sha256": hashlib.sha256(
            metadata_path.read_bytes()
        ).hexdigest(),
        "images_sha256": image_hashes,
        "run_nonce": "run",
        "attempt_nonce": "attempt",
        "attempt_index": 1,
        "task_success": False,
        "official_success_source": 'info["done"]["success"]',
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    _write_json(attempt_dir / "terminal_failure_receipt.json", receipt)
    _write_json(attempt_dir / "final_result.json", {"task_success": False})
    (attempt_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps({"env_step": 17, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (attempt_dir / "behavior_tool_trace.jsonl").write_text(
        json.dumps(
            {
                "step": 4,
                "tool": "save_robot_state_checkpoint",
                "result": {
                    "_finish": True,
                    "task_success": False,
                    "stop_reason": "radio_tipped_flat",
                    "runner_termination_reason": "visual_radio_tipped_flat",
                    "terminal_failure_receipt": receipt,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    binding = _terminal_failure_binding(attempt_dir)
    assert binding is not None
    assert binding["condition"] == "radio_tipped_flat"
    assert binding["attempt_index"] == 1
    assert (
        _terminal_failure_binding(
            attempt_dir,
            task_name="picking_up_trash",
        )
        is None
    )

    (attempt_dir / "behavior_action_trace.jsonl").write_text(
        json.dumps({"env_step": 17, "info_done": {"success": True}}) + "\n",
        encoding="utf-8",
    )
    assert _terminal_failure_binding(attempt_dir) is None


def test_visual_terminal_failure_stops_campaign_without_another_attempt(
    monkeypatch,
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[1]
    dependencies, calls, dashboard = _dependencies(
        [(False, True), (True, True), (True, True)]
    )
    config = replace(_config(tmp_path, repo_root), max_attempts=3)
    binding = {
        "source": "llm_fresh_visual_observation",
        "condition": "radio_tipped_flat",
        "cause": "dropped_out_of_gripper",
        "camera": "head",
        "frame_id": "head:evidence",
        "env_step": 10,
        "attempt_index": 1,
        "receipt_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        serial_explore,
        "_terminal_failure_binding",
        lambda _attempt_dir, **_kwargs: binding,
    )

    manifest = run_explore_job(config, dependencies=dependencies)

    assert manifest["status"] == "completed_without_success"
    assert manifest["task_success"] is False
    assert manifest["terminal_failure"] == binding
    assert manifest["cumulative"]["attempts"] == 1
    assert len(calls.attempts) == 1
    assert dashboard.started == [1]
    assert dashboard.done == [False]
