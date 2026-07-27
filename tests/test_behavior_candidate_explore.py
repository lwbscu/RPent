from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from behavior_resource_fixtures import fixture_resource_binding

from robots.behavior import candidate_explore
from robots.behavior.candidate_explore import (
    CANDIDATE_ATTEMPT_TIMEOUT_S,
    CANDIDATE_MAX_WALL_CLOCK_S,
    CANDIDATE_PLANNER_TIMEOUT_S,
    build_candidate_config,
    run_candidate_explore,
)
from robots.behavior.policy_checkpoint import SHARED_POLICY_CHECKPOINT_PATH
from robots.behavior.publication import (
    PublicationValidationError,
    validate_canonical_publication_root,
)
from robots.behavior.task_specs import get_task_spec


@pytest.fixture(autouse=True)
def _stub_shared_checkpoint_binding(monkeypatch):
    def expected(path):
        resolved = str(Path(path).expanduser().resolve(strict=True))
        return {
            "schema_version": 1,
            "profile_id": "pi05-b1kpt50-cs32",
            "resolved_path": resolved,
            "files": {},
            "binding_sha256": "a" * 64,
        }

    monkeypatch.setattr(candidate_explore, "_expected_job_checkpoint_binding", expected)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, *values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _candidate_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    repo_root = Path(__file__).resolve().parents[1]
    behavior_repo = tmp_path / "behavior-repo"
    behavior_repo.mkdir()
    python = tmp_path / "python"
    python.write_bytes(b"")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    state_dir = tmp_path / get_task_spec("turning_on_radio").state_dir_name
    state_dir.mkdir()
    state_file = state_dir / "scene_214_template-tro_state.json"
    state_file.write_text('{"candidate": 214}\n', encoding="utf-8")
    state_sha256 = hashlib.sha256(state_file.read_bytes()).hexdigest()
    return repo_root, behavior_repo, python, checkpoint, state_sha256


def _config(tmp_path: Path):
    repo_root, behavior_repo, python, checkpoint, state_sha256 = _candidate_inputs(
        tmp_path
    )
    state_file = (
        tmp_path
        / get_task_spec("turning_on_radio").state_dir_name
        / "scene_214_template-tro_state.json"
    )
    config = build_candidate_config(
        output_root=tmp_path / "candidate-job",
        repo_root=repo_root,
        python=python,
        behavior_repo=behavior_repo,
        behavior_python=python,
        policy_checkpoint=checkpoint,
        resource_binding=fixture_resource_binding(),
        candidate_instance_id=214,
        candidate_state_file=state_file,
    )
    assert config.candidate_state_sha256 == state_sha256
    return config, state_file


def _success_attempt(root: Path, state_sha256: str) -> dict:
    job_id = "candidate-job-214"
    attempt = root / "attempts" / "turning_on_radio_candidate_i214" / "attempt_001"
    attempt.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "run_nonce": "run-nonce-214",
        "attempt_nonce": "attempt-nonce-214",
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
        ).encode("utf-8")
    ).hexdigest()
    _json(attempt / "official_success_receipt.json", receipt)
    _jsonl(
        attempt / "behavior_action_trace.jsonl",
        {"env_step": 42, "info_done": {"success": True}},
    )
    _jsonl(
        attempt / "behavior_tool_trace.jsonl",
        {
            "step": 1,
            "tool": "press",
            "result": {
                "task_success": True,
                "official_success_source": 'info["done"]["success"]',
                "run_nonce": receipt["run_nonce"],
                "attempt_nonce": receipt["attempt_nonce"],
                "official_success_receipt": receipt,
            },
        },
    )
    final_result = {
        "schema_version": 1,
        "run_status": "completed",
        "task_success": True,
        "official_success_source": 'info["done"]["success"]',
        "runtime_cleanup": "complete",
        "job": {"job_id": job_id, "attempt_index": 1},
    }
    _json(attempt / "final_result.json", final_result)
    run_manifest = {
        "schema_version": 4,
        "status": "stopped",
        "stopped_at": "2026-07-23T00:00:00Z",
        "job": {"job_id": job_id, "attempt_index": 1},
        "native_binding": {
            "task_name": "turning_on_radio",
            "activity_definition_id": 0,
            "activity_instance_id": 214,
            "env_seed": 0,
            "state_sha256": state_sha256,
        },
        "processes": {
            "env": {"managed": True, "stopped_at": "2026-07-23T00:00:00Z"},
            "vla": {"managed": False, "stopped_at": None},
        },
    }
    _json(attempt / "run_manifest.json", run_manifest)
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "succeeded",
        "finished_at": "2026-07-23T00:00:00Z",
        "blocked_reason": None,
        "task_success": True,
        "publication_complete": False,
        "protocol": {
            "task_index": 0,
            "task_name": "turning_on_radio",
            "public_seed": 0,
        },
        "native_binding": {
            "mapping_version": "turning_on_radio_candidate_instance_v1",
            "activity_definition_id": 0,
            "activity_instance_id": 214,
            "env_seed": 0,
            "state_sha256": state_sha256,
        },
        "task_identity": {
            "task_name": "turning_on_radio",
            "activity_definition_id": 0,
            "activity_instance_id": 214,
        },
        "planner": {
            "backend": "codex",
            "model": "gpt-5.5",
            "reasoning_effort": "xhigh",
        },
        "attempts": [
            {
                "attempt_index": 1,
                "outcome": "official_success",
                "task_success": True,
                "forced_cleanup_groups": {},
                "output_dir": str(attempt),
            }
        ],
        "processes": {"vla": {"managed": True, "stopped_at": "2026-07-23T00:00:00Z"}},
    }
    _json(root / "session_manifest.json", manifest)
    return manifest


def test_candidate_config_is_finite_gpu7_gpt55_xhigh_and_english(tmp_path):
    config, _state_file = _config(tmp_path)

    assert config.candidate_instance_id == 214
    assert config.max_attempts == 1
    assert config.max_wall_clock_s == CANDIDATE_MAX_WALL_CLOCK_S == 6900
    assert config.planner_timeout_s == CANDIDATE_PLANNER_TIMEOUT_S == 6900
    assert config.attempt_timeout_s == CANDIDATE_ATTEMPT_TIMEOUT_S == 7200
    assert config.cuda_device == "7"
    assert config.model == "gpt-5.5"
    assert config.reasoning_effort == "xhigh"
    assert config.max_turns == 1000
    assert config.max_tool_calls == 350
    assert config.dashboard_language == "en"


def test_candidate_tool_call_default_is_350_and_cli_override_is_preserved(tmp_path):
    default_args = candidate_explore._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--activity-instance-id",
            "214",
        ]
    )
    override_args = candidate_explore._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--activity-instance-id",
            "214",
            "--max-tool-calls",
            "417",
        ]
    )

    assert default_args.max_tool_calls == 350
    assert override_args.max_tool_calls == 417


def test_candidate_identity_is_task_scoped_and_public_instances_are_rejected(
    tmp_path,
):
    repo_root, behavior_repo, python, checkpoint, _state_sha256 = _candidate_inputs(
        tmp_path
    )
    trash_spec = get_task_spec("picking_up_trash")
    trash_state_dir = tmp_path / trash_spec.state_dir_name
    trash_state_dir.mkdir()
    trash_candidate = trash_state_dir / "scene_242_template-tro_state.json"
    trash_candidate.write_text('{"candidate": 242}\n', encoding="utf-8")
    config = build_candidate_config(
        output_root=tmp_path / "trash-candidate",
        repo_root=repo_root,
        python=python,
        behavior_repo=behavior_repo,
        behavior_python=python,
        policy_checkpoint=checkpoint,
        resource_binding=fixture_resource_binding(),
        task_name=trash_spec.task_name,
        candidate_instance_id=242,
        candidate_state_file=trash_candidate,
    )

    assert config.task_name == "picking_up_trash"
    assert config.candidate_instance_id == 242
    assert config.public_seed == 0
    assert config.policy_checkpoint_binding["profile_id"] == "pi05-b1kpt50-cs32"

    for task_name, public_instance in (
        ("turning_on_radio", 298),
        ("picking_up_trash", 196),
        ("picking_up_trash", 67),
    ):
        spec = get_task_spec(task_name)
        state_dir = tmp_path / f"public-{task_name}" / spec.state_dir_name
        state_dir.mkdir(parents=True, exist_ok=True)
        state_file = state_dir / f"scene_{public_instance}_template-tro_state.json"
        state_file.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-public"):
            build_candidate_config(
                output_root=tmp_path / f"reject-{task_name}",
                repo_root=repo_root,
                python=python,
                behavior_repo=behavior_repo,
                behavior_python=python,
                policy_checkpoint=checkpoint,
                resource_binding=fixture_resource_binding(),
                task_name=task_name,
                candidate_instance_id=public_instance,
                candidate_state_file=state_file,
            )


def test_candidate_cli_ignores_ambient_checkpoint_path(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "PI05_CHECKPOINT_PATH",
        "/home/ubuntu/lwb/Models/scilwb/pi05-turning_on_radio-sft-from-jax",
    )

    args = candidate_explore._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--activity-instance-id",
            "214",
        ]
    )

    assert args.policy_checkpoint == str(SHARED_POLICY_CHECKPOINT_PATH)


def test_candidate_main_prepares_one_binding_and_passes_it_to_the_job(
    monkeypatch,
    tmp_path,
):
    binding = fixture_resource_binding()
    calls: list[dict] = []
    captured: dict = {}
    repo_root = tmp_path / "repo"
    behavior_repo = tmp_path / "behavior"
    checkpoint = tmp_path / "checkpoint"
    state_file = tmp_path / "candidate_state.json"
    for directory in (repo_root, behavior_repo, checkpoint):
        directory.mkdir()
    state_file.write_text("{}\n", encoding="utf-8")

    def prepare(subtree, **kwargs):
        calls.append({"subtree": subtree, **kwargs})
        return binding

    def build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_root=Path(kwargs["output_root"]))

    monkeypatch.setattr(
        candidate_explore,
        "prepare_pinned_dataset_resources",
        prepare,
    )
    monkeypatch.setattr(
        candidate_explore,
        "_resolve_state_file",
        lambda _args, _repo: state_file,
    )
    monkeypatch.setattr(candidate_explore, "build_candidate_config", build)
    monkeypatch.setattr(
        candidate_explore,
        "run_candidate_explore",
        lambda _config, *, candidate_state_file: {
            "task_success": True,
            "state_file": str(candidate_state_file),
        },
    )

    exit_code = candidate_explore.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--activity-instance-id",
            "214",
            "--repo-root",
            str(repo_root),
            "--behavior-repo",
            str(behavior_repo),
            "--policy-checkpoint",
            str(checkpoint),
            "--behavior-resource-revision",
            "a" * 40,
            "--behavior-resource-cache",
            str(tmp_path / "cache"),
            "--behavior-resource-offline",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "subtree": "behavior",
            "requested_revision": "a" * 40,
            "cache_root": (tmp_path / "cache").resolve(),
            "offline": True,
        }
    ]
    assert captured["resource_binding"] is binding


def test_candidate_resource_drift_is_rejected_before_the_runner_is_called(
    tmp_path,
):
    copied_root = tmp_path / "resource-snapshot"
    shutil.copytree(fixture_resource_binding().root, copied_root)
    binding = fixture_resource_binding(copied_root)
    repo_root, behavior_repo, python, checkpoint, _state_sha256 = _candidate_inputs(
        tmp_path
    )
    state_file = (
        tmp_path
        / get_task_spec("turning_on_radio").state_dir_name
        / "scene_214_template-tro_state.json"
    )
    config = build_candidate_config(
        output_root=tmp_path / "candidate-job",
        repo_root=repo_root,
        python=python,
        behavior_repo=behavior_repo,
        behavior_python=python,
        policy_checkpoint=checkpoint,
        resource_binding=binding,
        candidate_instance_id=214,
        candidate_state_file=state_file,
    )
    target = binding.root / "memory" / "MEMORY.md"
    target.write_text(
        target.read_text(encoding="utf-8") + "\nresource drift\n",
        encoding="utf-8",
    )
    called = False

    def runner(_config):
        nonlocal called
        called = True
        raise AssertionError("drifted resources must not reach the runtime")

    with pytest.raises(RuntimeError, match="resource"):
        run_candidate_explore(
            config,
            candidate_state_file=state_file,
            run_job=runner,
        )

    assert called is False


def test_pre_runtime_failure_still_seals_five_null_evidence_files(tmp_path):
    config, state_file = _config(tmp_path)

    def fail_before_runtime(_config):
        raise RuntimeError("VLA failed before the child runtime started")

    result = run_candidate_explore(
        config,
        candidate_state_file=state_file,
        run_job=fail_before_runtime,
    )

    assert result["task_success"] is None
    assert result["publication_complete"] is False
    for name in (
        "session_manifest.json",
        "behavior_action_trace.jsonl",
        "behavior_tool_trace.jsonl",
        "final_result.json",
        "run_manifest.json",
    ):
        assert (config.output_root / name).is_file()
        assert not (config.output_root / name).is_symlink()
    assert (config.output_root / "behavior_action_trace.jsonl").read_bytes() == b""
    assert (config.output_root / "behavior_tool_trace.jsonl").read_bytes() == b""
    final = json.loads((config.output_root / "final_result.json").read_text())
    run_manifest = json.loads((config.output_root / "run_manifest.json").read_text())
    assert final["outer_harness_fallback"] is True
    assert final["runtime_started"] is False
    assert final["task_success"] is None
    assert run_manifest["outer_harness_fallback"] is True
    assert run_manifest["task_success"] is None
    assert not (config.output_root / "candidate_publication").exists()


def test_raw_success_copies_exact_evidence_and_publishes_candidate_only(tmp_path):
    config, state_file = _config(tmp_path)
    copied: dict[str, bytes] = {}

    def succeed(_config):
        manifest = _success_attempt(
            config.output_root,
            str(config.candidate_state_sha256),
        )
        attempt = Path(manifest["attempts"][0]["output_dir"])
        for name in (
            "behavior_action_trace.jsonl",
            "behavior_tool_trace.jsonl",
            "final_result.json",
            "run_manifest.json",
        ):
            copied[name] = (attempt / name).read_bytes()
        return manifest

    result = run_candidate_explore(
        config,
        candidate_state_file=state_file,
        run_job=succeed,
    )

    assert result["task_success"] is True
    assert result["publication_complete"] is True
    for name, expected in copied.items():
        assert (config.output_root / name).read_bytes() == expected
    publication = config.output_root / "candidate_publication"
    assert {
        "recipe.jsonl",
        "task_memory.md",
        "provenance.json",
        "amendment.json",
    } <= {path.name for path in publication.iterdir()}
    provenance = json.loads((publication / "provenance.json").read_text())
    amendment = json.loads((publication / "amendment.json").read_text())
    assert provenance["candidate_only"] is True
    assert provenance["eligible_for_formal_eval"] is False
    assert provenance["activity_instance_id"] == 214
    assert provenance["state"]["sha256"] == config.candidate_state_sha256
    assert set(provenance["source_artifacts_sha256"]) == {
        "official_success_receipt",
        "behavior_action_trace",
        "behavior_tool_trace",
        "final_result",
        "run_manifest",
        "session_manifest",
    }
    assert amendment["review_required"] is True
    assert amendment["eligible_for_formal_eval"] is False
    assert "214" not in (publication / "recipe.jsonl").read_text()
    assert "214" not in (publication / "task_memory.md").read_text()
    assert not (config.output_root / "recipe_turning_on_radio_s0.jsonl").exists()
    assert not (config.output_root / "publication_amendment.json").exists()
    assert not (config.output_root / "memory").exists()
    with pytest.raises(PublicationValidationError):
        validate_canonical_publication_root(config.output_root)
    with pytest.raises(PublicationValidationError):
        validate_canonical_publication_root(publication)


def test_manifest_success_without_valid_receipt_does_not_publish(tmp_path):
    config, state_file = _config(tmp_path)

    def succeed_with_tampered_receipt(_config):
        manifest = _success_attempt(
            config.output_root,
            str(config.candidate_state_sha256),
        )
        attempt = Path(manifest["attempts"][0]["output_dir"])
        receipt_path = attempt / "official_success_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["receipt_sha256"] = "0" * 64
        _json(receipt_path, receipt)
        return manifest

    result = run_candidate_explore(
        config,
        candidate_state_file=state_file,
        run_job=succeed_with_tampered_receipt,
    )

    assert result["task_success"] is None
    assert result["publication_complete"] is False
    assert "raw-success receipt" in result["error"]
    assert not (config.output_root / "candidate_publication").exists()


def test_cross_task_success_identity_mismatch_does_not_publish(tmp_path):
    config, state_file = _config(tmp_path)

    def spoofed_success(_config):
        manifest = _success_attempt(
            config.output_root,
            str(config.candidate_state_sha256),
        )
        manifest["task_identity"]["task_name"] = "picking_up_trash"
        _json(config.output_root / "session_manifest.json", manifest)
        return manifest

    result = run_candidate_explore(
        config,
        candidate_state_file=state_file,
        run_job=spoofed_success,
    )

    assert result["task_success"] is None
    assert result["publication_complete"] is False
    assert "mismatched task identity" in result["error"]
    assert not (config.output_root / "candidate_publication").exists()
