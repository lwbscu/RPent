from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from behavior_resource_fixtures import fixture_resource_binding

from robots.behavior import campaign_explore, candidate_explore
from robots.behavior.campaign_explore import (
    DEFAULT_INSTANCE_ORDER,
    CampaignConfig,
    JobExecution,
    run_campaign,
)
from robots.behavior.policy_checkpoint import SHARED_POLICY_CHECKPOINT_PATH
from robots.behavior.runtime import _read_prior_attempt_summaries
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

    monkeypatch.setattr(campaign_explore, "_expected_job_checkpoint_binding", expected)
    monkeypatch.setattr(candidate_explore, "_expected_job_checkpoint_binding", expected)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _config(
    tmp_path: Path,
    instances: tuple[int, ...],
    *,
    task_name: str = "turning_on_radio",
) -> CampaignConfig:
    repo = tmp_path / "repo"
    behavior = tmp_path / "behavior"
    checkpoint = tmp_path / "checkpoint"
    state_dir = tmp_path / get_task_spec(task_name).state_dir_name
    for directory in (repo, behavior, checkpoint, state_dir):
        directory.mkdir(parents=True)
    for instance_id in instances:
        (state_dir / f"radio_0_{instance_id}_template-tro_state.json").write_text(
            "{}",
            encoding="utf-8",
        )
    resource_binding = fixture_resource_binding()
    catalog = json.loads(
        (resource_binding.root / "recipes" / "catalog_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    return CampaignConfig(
        output_root=tmp_path / "campaign",
        repo_root=repo,
        python=Path("/venv/bin/python"),
        behavior_repo=behavior,
        behavior_python=Path("/behavior/bin/python"),
        policy_checkpoint=checkpoint,
        state_dir=state_dir,
        resource_binding=resource_binding,
        task_name=task_name,
        instance_ids=instances,
        dashboard=False,
        recipe_catalog_sha256=catalog["catalog_sha256"],
    )


def test_campaign_tool_call_default_is_350_and_cli_override_is_preserved(tmp_path):
    config = _config(tmp_path, (242,))
    default_args = campaign_explore._parse_args(["--output-root", str(tmp_path)])
    override_args = campaign_explore._parse_args(
        [
            "--output-root",
            str(tmp_path),
            "--max-tool-calls",
            "417",
        ]
    )

    assert config.max_tool_calls == 350
    assert default_args.max_tool_calls == 350
    assert override_args.max_tool_calls == 417


def test_campaign_cli_ignores_ambient_checkpoint_and_requires_trash_instances(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "PI05_CHECKPOINT_PATH",
        "/home/ubuntu/lwb/Models/scilwb/pi05-turning_on_radio-sft-from-jax",
    )

    args = campaign_explore._parse_args(["--output-root", str(tmp_path)])

    assert args.policy_checkpoint == str(SHARED_POLICY_CHECKPOINT_PATH)
    assert args.instances is None
    with pytest.raises(SystemExit, match="--instances is required"):
        campaign_explore.main(
            [
                "--output-root",
                str(tmp_path / "trash"),
                "--task-name",
                "picking_up_trash",
            ]
        )


def test_campaign_main_prepares_resources_once_before_building_config(
    monkeypatch,
    tmp_path,
):
    binding = fixture_resource_binding()
    calls: list[dict] = []
    captured: dict = {}
    repo_root = tmp_path / "repo"
    behavior_repo = tmp_path / "behavior"
    checkpoint = tmp_path / "checkpoint"
    state_dir = tmp_path / get_task_spec("turning_on_radio").state_dir_name
    for directory in (repo_root, behavior_repo, checkpoint, state_dir):
        directory.mkdir()
    (state_dir / "scene_242_template-tro_state.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    def prepare(subtree, **kwargs):
        calls.append({"subtree": subtree, **kwargs})
        return binding

    def run(config):
        captured["config"] = config
        return {"status": "completed"}

    monkeypatch.setattr(
        campaign_explore,
        "prepare_pinned_dataset_resources",
        prepare,
    )
    monkeypatch.setattr(campaign_explore, "run_campaign", run)

    exit_code = campaign_explore.main(
        [
            "--output-root",
            str(tmp_path / "output"),
            "--instances",
            "242",
            "--repo-root",
            str(repo_root),
            "--behavior-repo",
            str(behavior_repo),
            "--policy-checkpoint",
            str(checkpoint),
            "--state-dir",
            str(state_dir),
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
    assert captured["config"].resource_binding is binding


def test_default_campaign_runner_borrows_one_supervisor_vla(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242, 214))
    process = SimpleNamespace(pid=12345, returncode=None)
    process.poll = lambda: process.returncode
    starts = []
    stops = []
    seen_processes = []

    def start_vla(_config, output_dir):
        starts.append(output_dir)
        return "http://127.0.0.1:9123", process

    dependencies = campaign_explore.default_dependencies()
    dependencies.start_vla = start_vla
    dependencies.stop_vla = lambda proc: stops.append(proc)
    monkeypatch.setattr(
        campaign_explore,
        "default_dependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)

    def runner(job_config, _state, _canonical, borrowed):
        assert borrowed.owns_vla is False
        _endpoint, seen = borrowed.start_vla(job_config, job_config.output_root / "vla")
        seen_processes.append(seen)
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task from fresh evidence.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_default_job_runner", runner)

    result = run_campaign(config, job_runner=runner)

    assert result["status"] == "completed"
    assert len(starts) == 1
    assert seen_processes == [process, process]
    assert stops == [process]


def test_trash_campaign_uses_task_local_candidate_and_failure_namespace(
    tmp_path,
    monkeypatch,
):
    config = _config(
        tmp_path,
        (196, 242),
        task_name="picking_up_trash",
    )
    canonical_flags: list[bool] = []
    checkpoint_bindings: list[dict] = []
    resource_bindings = []

    def runner(job_config, _state, canonical, _dependencies):
        canonical_flags.append(canonical)
        checkpoint_bindings.append(job_config.policy_checkpoint_binding)
        resource_bindings.append(job_config.resource_binding)
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the multi-object task from fresh evidence.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)

    assert result["task_identity"] == {
        "task_name": "picking_up_trash",
        "task_index": 1,
        "activity_definition_id": 0,
    }
    assert canonical_flags == [True, False]
    assert checkpoint_bindings[0] == checkpoint_bindings[1]
    assert checkpoint_bindings[0]["profile_id"] == "pi05-b1kpt50-cs32"
    assert resource_bindings == [config.resource_binding, config.resource_binding]
    assert all(binding is config.resource_binding for binding in resource_bindings)
    assert list(
        (
            config.output_root / "knowledge" / "picking_up_trash" / "failure_pool"
        ).iterdir()
    )
    assert not (config.output_root / "knowledge" / "turning_on_radio").exists()


def _write_attempt(
    config,
    *,
    outcome: str,
    summary: str,
    exit_code: int = 0,
) -> dict:
    attempt = config.output_root / "attempts" / "task" / "attempt_001"
    attempt.mkdir(parents=True)
    (attempt / "behavior_action_trace.jsonl").write_text(
        json.dumps({"env_step": 0, "info_done": {"success": False}}) + "\n",
        encoding="utf-8",
    )
    (attempt / "behavior_tool_trace.jsonl").write_text("", encoding="utf-8")
    _write_json(attempt / "final_result.json", {"task_success": False})
    _write_json(attempt / "run_manifest.json", {"schema_version": 1})
    return {
        "schema_version": 1,
        "status": "completed_without_success",
        "publication_complete": False,
        "attempts": [
            {
                "attempt_index": 1,
                "outcome": outcome,
                "summary": summary,
                "subprocess_exit_code": exit_code,
                "timed_out": False,
                "output_dir": str(attempt),
            }
        ],
    }


def _write_anonymous_publication(
    job_root: Path,
    manifest: dict,
    binding: dict,
) -> None:
    native_binding = manifest.get("native_binding", {})
    manifest.setdefault(
        "protocol",
        {
            "task_index": 0,
            "task_name": "turning_on_radio",
            "public_seed": 0,
        },
    )
    manifest.setdefault(
        "task_identity",
        {
            "task_name": "turning_on_radio",
            "activity_definition_id": 0,
            "activity_instance_id": native_binding.get("activity_instance_id"),
        },
    )
    attempt = campaign_explore._seal_job_evidence(job_root, manifest)
    assert attempt is not None
    _write_json(attempt / "official_success_receipt.json", binding)
    source_paths = {
        "official_success_receipt": attempt / "official_success_receipt.json",
        "behavior_action_trace": job_root / "behavior_action_trace.jsonl",
        "behavior_tool_trace": job_root / "behavior_tool_trace.jsonl",
        "final_result": job_root / "final_result.json",
        "run_manifest": job_root / "run_manifest.json",
        "session_manifest": job_root / "session_manifest.json",
    }
    source_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    session = json.loads((job_root / "session_manifest.json").read_text())
    publication = job_root / "candidate_publication"
    publication.mkdir(parents=True)
    (publication / "recipe.jsonl").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "semantic_goal",
                "goal": "Turn on the task radio from fresh public evidence.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (publication / "task_memory.md").write_text(
        "# Task memory\n\nRe-ground semantic evidence in every episode.\n",
        encoding="utf-8",
    )
    recipe_sha256 = hashlib.sha256(
        (publication / "recipe.jsonl").read_bytes()
    ).hexdigest()
    memory_sha256 = hashlib.sha256(
        (publication / "task_memory.md").read_bytes()
    ).hexdigest()
    provenance = {
        "schema_version": 1,
        "candidate_only": True,
        "eligible_for_formal_eval": False,
        "review_required": True,
        "task_identity": {
            "task_name": "turning_on_radio",
            "task_index": 0,
            "activity_definition_id": 0,
            "activity_instance_id": session["native_binding"]["activity_instance_id"],
        },
        "source_tag": "turning_on_radio_s0",
        "activity_instance_id": session["native_binding"]["activity_instance_id"],
        "state": {
            "sha256": session["native_binding"]["state_sha256"],
        },
        "planner": session["planner"],
        "raw_success": binding,
        "source_artifacts_sha256": source_hashes,
        "recipe_sha256": recipe_sha256,
        "memory_sha256": memory_sha256,
    }
    _write_json(publication / "provenance.json", provenance)
    _write_json(
        publication / "amendment.json",
        {
            "schema_version": 1,
            "candidate_only": True,
            "eligible_for_formal_eval": False,
            "review_required": True,
            "publication_complete": True,
            "provenance_sha256": hashlib.sha256(
                (publication / "provenance.json").read_bytes()
            ).hexdigest(),
            "recipe_sha256": recipe_sha256,
            "memory_sha256": memory_sha256,
        },
    )


def test_default_instance_order_matches_requested_campaign():
    assert DEFAULT_INSTANCE_ORDER == (
        242,
        214,
        139,
        185,
        102,
        246,
        105,
        271,
        119,
        220,
        224,
    )
    assert len(DEFAULT_INSTANCE_ORDER) == len(set(DEFAULT_INSTANCE_ORDER))


def test_campaign_rejects_task_local_public_eval_instance(tmp_path):
    config = _config(tmp_path, (242, 298))

    with pytest.raises(ValueError, match="public Eval instances"):
        campaign_explore._validate_campaign_config(config)


def test_campaign_runs_serial_jobs_and_inherits_anonymous_summary(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242, 214))
    calls: list[tuple[int | None, int | None, tuple[dict, ...]]] = []

    def runner(job_config, _state, canonical, _dependencies):
        calls.append(
            (
                job_config.candidate_instance_id,
                job_config.max_attempts,
                job_config.initial_prior_summaries,
            )
        )
        summary = (
            "Instance 242 used the left hand at pixel row 12. "
            "Re-ground the visible control hypothesis before committing."
        )
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary=summary,
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)

    assert result["status"] == "completed"
    assert result["counts"] == {
        "success": 0,
        "task_failure": 2,
        "infra_unknown": 0,
    }
    assert calls[0] == (None, 1, ())
    assert calls[1][0:2] == (214, 1)
    inherited = json.dumps(calls[1][2]).lower()
    assert "instance" not in inherited
    assert "left" not in inherited
    assert "pixel" not in inherited
    assert "re-ground" in inherited
    for index, instance_id in enumerate((242, 214), start=1):
        job = config.output_root / "jobs" / f"{index:03d}_instance_{instance_id}"
        assert all((job / name).is_file() for name in campaign_explore._EVIDENCE_NAMES)
    assert list(
        (
            config.output_root / "knowledge" / "turning_on_radio" / "failure_pool"
        ).iterdir()
    )
    assert not (config.repo_root / "resources" / "behavior").exists()


def test_campaign_classifies_verified_success_and_infra_unknown(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242, 214))
    attempt_paths: list[Path] = []

    def runner(job_config, _state, _canonical, _dependencies):
        if not attempt_paths:
            manifest = _write_attempt(
                job_config,
                outcome="official_success",
                summary="Fresh semantic evidence confirmed the radio control.",
            )
            manifest["status"] = "succeeded"
            attempt_paths.append(Path(manifest["attempts"][0]["output_dir"]).resolve())
            return JobExecution(manifest)
        return JobExecution(
            {
                "schema_version": 1,
                "status": "outer_harness_failure",
                "attempts": [],
                "error": "runtime unavailable",
            }
        )

    monkeypatch.setattr(
        campaign_explore,
        "_official_success_binding",
        lambda path: (
            {
                "source": 'info["done"]["success"]',
                "receipt_sha256": "a" * 64,
            }
            if path is not None and path.resolve() in attempt_paths
            else None
        ),
    )
    result = run_campaign(config, job_runner=runner)

    assert result["counts"] == {
        "success": 1,
        "task_failure": 0,
        "infra_unknown": 1,
    }
    assert [job["task_success"] for job in result["jobs"]] == [True, None]
    task_knowledge = config.output_root / "knowledge" / "turning_on_radio"
    assert list((task_knowledge / "success_unpublished").iterdir())
    assert not (task_knowledge / "recipes").exists()
    assert not (
        config.output_root
        / "knowledge"
        / "turning_on_radio"
        / "failure_pool"
        / result["jobs"][0]["knowledge_id"]
    ).exists()


def test_completed_campaign_resume_does_not_repeat_jobs(tmp_path, monkeypatch):
    config = _config(tmp_path, (242,))
    calls = 0

    def runner(job_config, _state, _canonical, _dependencies):
        nonlocal calls
        calls += 1
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the visible control.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    first = run_campaign(config, job_runner=runner)
    resumed = run_campaign(
        CampaignConfig(**{**config.__dict__, "resume": True}),
        job_runner=runner,
    )

    assert first["status"] == "completed"
    assert resumed["status"] == "completed"
    assert calls == 1


def test_raw_success_trace_without_valid_receipt_is_integrity_unknown(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        manifest = _write_attempt(
            job_config,
            outcome="task_failed",
            summary="The attempt was unsuccessful.",
        )
        attempt = Path(manifest["attempts"][0]["output_dir"])
        (attempt / "behavior_action_trace.jsonl").write_text(
            json.dumps({"env_step": 9, "info_done": {"success": True}}) + "\n",
            encoding="utf-8",
        )
        return JobExecution(manifest)

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)

    assert result["jobs"][0]["task_success"] is None
    assert result["jobs"][0]["outcome"] == "success_integrity_unknown"
    assert result["counts"]["infra_unknown"] == 1


def test_raw_success_before_truncated_trace_tail_is_integrity_unknown(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        manifest = _write_attempt(
            job_config,
            outcome="task_failed",
            summary="The attempt was unsuccessful.",
        )
        attempt = Path(manifest["attempts"][0]["output_dir"])
        (attempt / "behavior_action_trace.jsonl").write_text(
            json.dumps({"env_step": 9, "info_done": {"success": True}})
            + "\n"
            + '{"truncated":',
            encoding="utf-8",
        )
        return JobExecution(manifest)

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)

    assert result["jobs"][0]["task_success"] is None
    assert result["jobs"][0]["outcome"] == "success_integrity_unknown"


def test_malformed_only_trace_is_infrastructure_unknown(tmp_path, monkeypatch):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        manifest = _write_attempt(
            job_config,
            outcome="task_failed",
            summary="The attempt was unsuccessful.",
        )
        attempt = Path(manifest["attempts"][0]["output_dir"])
        (attempt / "behavior_action_trace.jsonl").write_text(
            '{"truncated":',
            encoding="utf-8",
        )
        return JobExecution(manifest)

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)

    assert result["jobs"][0]["task_success"] is None
    assert result["jobs"][0]["outcome"] == "trace_integrity_unknown"


def test_verified_published_success_inherits_only_success_recipe_lesson(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (214, 139))
    attempts: set[Path] = set()
    inherited: list[tuple[dict, ...]] = []

    def runner(job_config, _state, _canonical, _dependencies):
        inherited.append(job_config.initial_prior_summaries)
        if not attempts:
            manifest = _write_attempt(
                job_config,
                outcome="official_success",
                summary="Attempt ended without raw official success.",
            )
            manifest["status"] = "succeeded"
            manifest["publication_complete"] = True
            manifest["native_binding"] = {
                "activity_instance_id": job_config.candidate_instance_id,
                "state_sha256": job_config.candidate_state_sha256,
            }
            manifest["planner"] = {
                "backend": "codex",
                "model": job_config.model,
                "reasoning_effort": job_config.reasoning_effort,
            }
            attempt = Path(manifest["attempts"][0]["output_dir"]).resolve()
            attempts.add(attempt)
            _write_anonymous_publication(
                job_config.output_root,
                manifest,
                {"source": 'info["done"]["success"]'},
            )
            (job_config.output_root / "recipe_unverified.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "semantic_goal",
                        "goal": "UNVERIFIED_EXTRA_RECIPE_MUST_NOT_BE_INHERITED",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return JobExecution(manifest)
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task control.",
            )
        )

    monkeypatch.setattr(
        campaign_explore,
        "_official_success_binding",
        lambda path: (
            {"source": 'info["done"]["success"]'}
            if path is not None and path.resolve() in attempts
            else None
        ),
    )
    result = run_campaign(config, job_runner=runner)

    assert result["jobs"][0]["publication_complete"] is True
    inherited_text = json.dumps(inherited[1]).lower()
    assert "raw official success confirmed" in inherited_text
    assert "attempt ended without raw official success" not in inherited_text
    assert "symbolic-recipe lesson" in inherited_text
    assert "unverified_extra_recipe" not in inherited_text
    assert list(
        (config.output_root / "knowledge" / "turning_on_radio" / "recipes").iterdir()
    )
    assert not (config.repo_root / "resources" / "behavior").exists()


def test_resume_with_unsealed_job_blocks_without_running_it(tmp_path):
    config = _config(tmp_path, (242,))
    root = config.output_root
    root.mkdir()
    manifest = campaign_explore._new_campaign_manifest(config, None)
    _write_json(root / "campaign_manifest.json", manifest)
    unsealed = root / "jobs" / "001_instance_242"
    unsealed.mkdir(parents=True)
    (unsealed / "partial.log").write_text("still owned", encoding="utf-8")
    calls = 0

    def runner(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("unsealed Job must not be restarted")

    resumed = run_campaign(
        CampaignConfig(**{**config.__dict__, "resume": True}),
        job_runner=runner,
    )

    assert resumed["status"] == "blocked"
    assert resumed["blocked_reason"] == "attempt_process_ownership_unverified"
    assert calls == 0


def test_campaign_lock_failure_does_not_start_a_job(tmp_path, monkeypatch):
    config = _config(tmp_path, (242,))
    calls = 0

    def runner(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("runner must not start")

    def refuse(_cuda_device):
        raise RuntimeError("another campaign owns GPU7")

    monkeypatch.setattr(campaign_explore, "_claim_campaign_lock", refuse)
    try:
        run_campaign(config, job_runner=runner)
    except RuntimeError as error:
        assert "owns GPU7" in str(error)
    else:
        raise AssertionError("campaign lock refusal must propagate")
    assert calls == 0
    assert not config.output_root.exists()


def test_standalone_gpu_job_lock_blocks_campaign_before_root(tmp_path):
    config = _config(tmp_path, (242,))
    lock_path = campaign_explore._gpu_lock_path("7")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        try:
            run_campaign(config)
        except RuntimeError as error:
            assert "standalone BEHAVIOR job" in str(error)
        else:
            raise AssertionError("standalone GPU lock must block the campaign")
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert not config.output_root.exists()


def test_gpu_lock_race_between_jobs_blocks_without_marking_null(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242, 214))
    checks = 0
    runs = 0

    def check(_cuda_device):
        nonlocal checks
        checks += 1
        if checks == 3:
            raise RuntimeError("another standalone BEHAVIOR job already owns GPU7")

    def runner(job_config, _state, _canonical, _dependencies):
        nonlocal runs
        runs += 1
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task control.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_assert_job_gpu_lock_available", check)
    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)

    assert result["status"] == "blocked"
    assert result["blocked_reason"] == "gpu_job_lock_conflict"
    assert result["jobs"][0]["status"] == "failed"
    assert result["jobs"][1]["status"] == "blocked"
    assert result["jobs"][1]["task_success"] is None
    assert result["counts"]["infra_unknown"] == 0
    assert runs == 1


def test_anonymous_publication_rejects_instance_number_and_absolute_path():
    assert not campaign_explore._anonymous_publication_safe(
        b'{"kind":"lesson","text":"Use evidence from 242."}\n',
        b"# Memory\n",
    )


def test_failure_summary_is_deidentified_in_campaign_knowledge(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary=(
                    "Knowledge marker 298 from /etc/radio-state should remain private."
                ),
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    result = run_campaign(config, job_runner=runner)
    knowledge_id = result["jobs"][0]["knowledge_id"]
    reviewer_records = [
        json.loads(line)
        for line in (
            config.output_root
            / "knowledge"
            / "turning_on_radio"
            / "reviewer"
            / "campaign_pattern_candidates.jsonl"
        )
        .read_text()
        .splitlines()
        if line.strip()
    ]
    reviewer_text = "\n".join(
        json.dumps(
            {key: value for key, value in record.items() if key != "knowledge_id"},
            sort_keys=True,
        )
        for record in reviewer_records
    )
    failure = (
        config.output_root
        / "knowledge"
        / "turning_on_radio"
        / "failure_pool"
        / knowledge_id
        / "failure_summary.md"
    ).read_text()

    # The content-addressed opaque ID may coincidentally contain any decimal
    # substring; anonymity is enforced on human-readable publication fields.
    assert "298" not in reviewer_text + failure
    assert "/etc/" not in reviewer_text + failure
    assert "re-ground" in (reviewer_text + failure).lower()
    assert not (config.repo_root / "resources" / "behavior").exists()
    assert not campaign_explore._anonymous_publication_safe(
        b'{"kind":"lesson","text":"Read /etc/passwd."}\n',
        b"# Memory\n",
    )


def test_campaign_dashboard_reuses_state_and_keeps_event_cursor(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242, 214))
    config = CampaignConfig(**{**config.__dict__, "dashboard": True})
    monkeypatch.setattr(
        "rpent.dashboard.DashboardServer.start",
        lambda self: "http://127.0.0.1:8765",
    )
    dashboard = campaign_explore._CampaignDashboard(config)
    first_config = campaign_explore._build_job_config(
        config,
        job_root=config.output_root / "jobs" / "001_instance_242",
        instance_id=242,
        state_file=campaign_explore._state_file(config.state_dir, 242),
        prior=(),
    )
    _, first_state, _ = dashboard.start_job(first_config, "first")
    first_state.on_event({"type": "assistant", "text": "first job"})
    cursor = len(first_state.events_since(0))
    second_config = campaign_explore._build_job_config(
        config,
        job_root=config.output_root / "jobs" / "002_instance_214",
        instance_id=214,
        state_file=campaign_explore._state_file(config.state_dir, 214),
        prior=(),
    )
    _, second_state, _ = dashboard.start_job(second_config, "second")
    second_state.on_event({"type": "assistant", "text": "second job"})

    assert second_state is first_state
    assert second_state.events_since(cursor)[-1]["text"] == "second job"


def test_campaign_prior_summary_is_accepted_by_fresh_attempt_one(tmp_path):
    summary_path = tmp_path / "prior.json"
    _write_json(
        summary_path,
        {
            "job_id": "fresh-job",
            "next_attempt_index": 1,
            "lineage_scope": "campaign_prior",
            "summaries": [
                {
                    "attempt_index": 1,
                    "outcome": "official_success",
                    "summary": "Re-ground fresh semantic evidence.",
                }
            ],
        },
    )
    args = SimpleNamespace(
        behavior_prior_attempt_summaries_file=str(summary_path),
        behavior_job_id="fresh-job",
        behavior_attempt_index=1,
    )

    _read_prior_attempt_summaries(args)

    assert args._behavior_prior_attempt_summaries.startswith("Campaign evidence 1")
    assert args._behavior_prior_attempt_summaries_input["lineage_scope"] == (
        "campaign_prior"
    )


def test_completed_resume_rejects_missing_destination_and_rolling_file(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task control.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    completed = run_campaign(config, job_runner=runner)
    knowledge_id = completed["jobs"][0]["knowledge_id"]
    destination = (
        config.output_root
        / "knowledge"
        / "turning_on_radio"
        / "failure_pool"
        / knowledge_id
    )
    (destination / "failure_evidence_binding.json").unlink()

    resumed = run_campaign(
        CampaignConfig(**{**config.__dict__, "resume": True}),
        job_runner=runner,
    )
    assert resumed["status"] == "blocked"
    assert resumed["blocked_reason"] == "completed_job_knowledge_integrity_mismatch"

    # A separate completed campaign must also reject a missing rolling binding.
    second = _config(tmp_path / "second", (242,))
    completed_second = run_campaign(second, job_runner=runner)
    assert completed_second["status"] == "completed"
    (
        second.output_root / "knowledge" / "turning_on_radio" / "rolling_summaries.json"
    ).unlink()
    resumed_second = run_campaign(
        CampaignConfig(**{**second.__dict__, "resume": True}),
        job_runner=runner,
    )
    assert resumed_second["status"] == "blocked"
    assert resumed_second["blocked_reason"] == "rolling_summary_integrity_mismatch"


def test_resume_rejects_changed_campaign_configuration(tmp_path, monkeypatch):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task control.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    assert run_campaign(config, job_runner=runner)["status"] == "completed"
    changed = CampaignConfig(
        **{
            **config.__dict__,
            "resume": True,
            "max_tool_calls": config.max_tool_calls - 1,
        }
    )

    resumed = run_campaign(changed, job_runner=runner)

    assert resumed["status"] == "blocked"
    assert resumed["blocked_reason"] == "campaign_configuration_binding_mismatch"


def test_resume_rejects_changed_behavior_source_tree(tmp_path, monkeypatch):
    config = _config(tmp_path, (242,))

    def runner(job_config, _state, _canonical, _dependencies):
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task control.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)
    assert run_campaign(config, job_runner=runner)["status"] == "completed"
    changed_source = config.behavior_repo / "rlinf" / "changed.py"
    changed_source.parent.mkdir(parents=True)
    changed_source.write_text("SOURCE_REVISION = 2\n", encoding="utf-8")

    resumed = run_campaign(
        CampaignConfig(**{**config.__dict__, "resume": True}),
        job_runner=runner,
    )

    assert resumed["status"] == "blocked"
    assert resumed["blocked_reason"] == "campaign_configuration_binding_mismatch"


def test_candidate_publication_rejects_symlinked_provenance_or_amendment(
    tmp_path,
):
    config = _config(tmp_path, (214,))
    manifest = _write_attempt(
        campaign_explore._build_job_config(
            config,
            job_root=config.output_root / "jobs" / "001_instance_214",
            instance_id=214,
            state_file=campaign_explore._state_file(config.state_dir, 214),
            prior=(),
        ),
        outcome="official_success",
        summary="Fresh evidence confirmed the control.",
    )
    manifest["native_binding"] = {
        "activity_instance_id": 214,
        "state_sha256": hashlib.sha256(
            campaign_explore._state_file(config.state_dir, 214).read_bytes()
        ).hexdigest(),
    }
    manifest["planner"] = {
        "backend": "codex",
        "model": "gpt-5.5",
        "reasoning_effort": "xhigh",
    }
    job_root = config.output_root / "jobs" / "001_instance_214"
    binding = {"source": 'info["done"]["success"]'}
    _write_anonymous_publication(job_root, manifest, binding)
    publication = job_root / "candidate_publication"
    for name in ("provenance.json", "amendment.json"):
        original = publication / name
        external = tmp_path / f"external_{name}"
        external.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(external)
        assert campaign_explore._candidate_publication_pair(job_root) is None
        original.unlink()
        original.write_bytes(external.read_bytes())


def test_predecessor_boundary_is_strictly_bound_and_matches_new_order(tmp_path):
    config = _config(tmp_path, (102, 246))
    boundary = tmp_path / "epoch_boundary.json"
    payload = {
        "schema_version": 1,
        "kind": "behavior_campaign_epoch_boundary",
        "predecessor_campaign_manifest_sha256": "a" * 64,
        "predecessor_configuration_sha256": "b" * 64,
        "predecessor_source_tree_sha256": "c" * 64,
        "predecessor_reviewed_memory_snapshot_sha256": "d" * 64,
        "sealed_completed_prefix": 2,
        "sealed_instance_order": [242, 214],
        "remaining_instance_order": [102, 246],
        "reason": "reviewed knowledge epoch upgrade",
        "resume_predecessor_campaign": False,
    }
    _write_json(boundary, payload)
    boundary_sha256 = hashlib.sha256(boundary.read_bytes()).hexdigest()
    bound = CampaignConfig(
        **{
            **config.__dict__,
            "predecessor_epoch_boundary": boundary,
            "predecessor_epoch_boundary_sha256": boundary_sha256,
        }
    )

    binding = campaign_explore._epoch_predecessor_binding(bound)

    assert binding is not None
    assert binding["sha256"] == boundary_sha256
    assert binding["remaining_instance_order"] == [102, 246]
    assert binding["resume_predecessor_campaign"] is False
    child = campaign_explore._build_job_config(
        bound,
        job_root=bound.output_root / "jobs" / "001_instance_102",
        instance_id=102,
        state_file=campaign_explore._state_file(bound.state_dir, 102),
        prior=(),
    )
    assert child.epoch_predecessor_binding == binding

    wrong_order = CampaignConfig(**{**bound.__dict__, "instance_ids": (246, 102)})
    try:
        campaign_explore._epoch_predecessor_binding(wrong_order)
    except ValueError as error:
        assert "instance order" in str(error)
    else:
        raise AssertionError("boundary must bind the exact new instance order")


def test_predecessor_boundary_rejects_symlink_and_sha_mismatch(tmp_path):
    config = _config(tmp_path, (102,))
    boundary = tmp_path / "real_boundary.json"
    payload = {
        "schema_version": 1,
        "kind": "behavior_campaign_epoch_boundary",
        "predecessor_campaign_manifest_sha256": "a" * 64,
        "predecessor_configuration_sha256": "b" * 64,
        "predecessor_source_tree_sha256": "c" * 64,
        "predecessor_reviewed_memory_snapshot_sha256": "d" * 64,
        "sealed_completed_prefix": 1,
        "sealed_instance_order": [242],
        "remaining_instance_order": [102],
        "reason": "reviewed knowledge epoch upgrade",
        "resume_predecessor_campaign": False,
    }
    _write_json(boundary, payload)
    link = tmp_path / "boundary_link.json"
    link.symlink_to(boundary)
    digest = hashlib.sha256(boundary.read_bytes()).hexdigest()

    for path, sha256 in ((link, digest), (boundary, "f" * 64)):
        invalid = CampaignConfig(
            **{
                **config.__dict__,
                "predecessor_epoch_boundary": path,
                "predecessor_epoch_boundary_sha256": sha256,
            }
        )
        try:
            campaign_explore._epoch_predecessor_binding(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe predecessor boundary must be rejected")


def test_campaign_memory_and_catalog_are_loaded_only_from_the_bound_snapshot(
    tmp_path,
):
    config = _config(tmp_path, (242,))
    repo_local = config.repo_root / "resources" / "behavior"
    (repo_local / "memory").mkdir(parents=True)
    (repo_local / "recipes").mkdir(parents=True)
    (repo_local / "memory" / "snapshot_manifest.json").write_text(
        '{"invalid_repo_local_fallback": true}\n',
        encoding="utf-8",
    )
    (repo_local / "recipes" / "catalog_manifest.json").write_text(
        '{"invalid_repo_local_fallback": true}\n',
        encoding="utf-8",
    )
    catalog_manifest = json.loads(
        (config.resource_binding.root / "recipes" / "catalog_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    config = CampaignConfig(
        **{
            **config.__dict__,
            "recipe_catalog_sha256": catalog_manifest["catalog_sha256"],
        }
    )

    memory = campaign_explore._campaign_memory_binding(config)
    catalog = campaign_explore._recipe_catalog_binding(config)
    configuration = campaign_explore._campaign_configuration_binding(config)

    assert memory["snapshot_sha256"]
    assert catalog is not None
    assert catalog["root"] == str(
        (config.resource_binding.root / "recipes").resolve(strict=True)
    )
    assert configuration["resource_source"] == config.resource_binding.as_dict()


def test_campaign_resource_drift_blocks_before_starting_the_next_job(
    tmp_path,
    monkeypatch,
):
    copied_root = tmp_path / "resource-snapshot"
    shutil.copytree(fixture_resource_binding().root, copied_root)
    binding = fixture_resource_binding(copied_root)
    config = _config(tmp_path, (242, 214))
    config = CampaignConfig(
        **{
            **config.__dict__,
            "resource_binding": binding,
        }
    )
    runs = 0

    def runner(job_config, _state, _canonical, _dependencies):
        nonlocal runs
        runs += 1
        manifest = _write_attempt(
            job_config,
            outcome="task_failed",
            summary="Re-ground the visible control.",
        )
        if runs == 1:
            target = binding.root / "memory" / "MEMORY.md"
            target.write_text(
                target.read_text(encoding="utf-8") + "\nresource drift\n",
                encoding="utf-8",
            )
        return JobExecution(manifest)

    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)

    result = run_campaign(config, job_runner=runner)

    assert runs == 1
    assert result["status"] == "blocked"
    assert result["jobs"][0]["status"] == "failed"
    assert result["jobs"][1]["status"] == "blocked"
    assert "resource" in result["blocked_reason"]


def test_catalog_binding_is_rechecked_before_every_job(tmp_path, monkeypatch):
    config = _config(tmp_path, (242, 214))
    config = CampaignConfig(**{**config.__dict__, "recipe_catalog_sha256": "a" * 64})
    original = {
        "root": "/reviewed/catalog",
        "catalog_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "declared_catalog_sha256": "a" * 64,
    }
    changed = {**original, "catalog_sha256": "c" * 64}
    calls = 0
    runs = 0

    def binding(_config):
        nonlocal calls
        calls += 1
        return changed if calls >= 4 else original

    def runner(job_config, _state, _canonical, _dependencies):
        nonlocal runs
        runs += 1
        assert job_config.recipe_catalog_sha256 == "a" * 64
        return JobExecution(
            _write_attempt(
                job_config,
                outcome="task_failed",
                summary="Re-ground the task control.",
            )
        )

    monkeypatch.setattr(campaign_explore, "_recipe_catalog_binding", binding)
    monkeypatch.setattr(campaign_explore, "_official_success_binding", lambda _p: None)

    result = run_campaign(config, job_runner=runner)

    assert runs == 1
    assert result["jobs"][0]["status"] == "failed"
    assert result["jobs"][1]["status"] == "blocked"
    assert result["blocked_reason"] == "reviewed_recipe_catalog_binding_mismatch"
