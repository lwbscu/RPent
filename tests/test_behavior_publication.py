from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from robots.behavior import publication
from robots.behavior.publication import (
    PublicationValidationError,
    canonical_bundle_id,
    validate_canonical_publication_root,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_publication(
    root: Path,
    *,
    task_name: str = publication.TASK_NAME,
    public_seed: int = 0,
    provenance_schema_version: int = 2,
    selection_task_name: str | None = None,
    selection_file_sha256: str | None = None,
    run_resource_revision: str | None = None,
    omit_session_resource_source: bool = False,
    receipt_success: bool = True,
    receipt_env_step: int = 42,
    action_env_step: int = 42,
    action_record: dict[str, object] | None = None,
    tool_receipt_matches: bool = True,
    final_job_matches: bool = True,
    run_memory_matches: bool = True,
    provenance_memory_snapshot: str | None = None,
    amendment_memory_snapshot: str | None = None,
    task_memory_suffix: str = "Use fresh visual evidence.",
    public_tool_contract_version: int = 2,
) -> dict[str, object]:
    identity = publication.resolve_publication_identity(
        task_name=task_name,
        public_seed=public_seed,
    )
    root.mkdir(parents=True)
    job_id = "behavior-explore-test"
    attempt_index = 1
    attempt = root / "attempts" / identity.tag / f"attempt_{attempt_index:03d}"
    attempt.mkdir(parents=True)
    memory_snapshot = "9" * 64
    selected_task_name = selection_task_name or identity.task_spec.task_name
    selected_memory_path = f"{selected_task_name}/target_prior.md"
    other_memory_path = (
        "turning_on_radio/target_prior.md"
        if selected_task_name != "turning_on_radio"
        else "picking_up_trash/target_prior.md"
    )
    reviewed_memory = {
        "snapshot_sha256": memory_snapshot,
        "manifest": {
            "schema_version": 1,
            "kind": "reviewed_behavior_memory_snapshot",
        },
        "files": {
            "radio.md": {
                "relative_path": "radio.md",
                "size_bytes": 12,
                "sha256": "8" * 64,
            }
        },
    }
    memory_files_sha256 = {"radio.md": "8" * 64}
    resource_source: dict[str, object] | None = None
    run_resource_source: dict[str, object] | None = None
    if provenance_schema_version == 3:
        selected_metadata = {
            "relative_path": selected_memory_path,
            "size_bytes": 12,
            "sha256": selection_file_sha256 or "8" * 64,
        }
        reviewed_memory["files"] = {
            selected_memory_path: {
                "relative_path": selected_memory_path,
                "size_bytes": 12,
                "sha256": "8" * 64,
                "included_in_prompt": True,
            },
            other_memory_path: {
                "relative_path": other_memory_path,
                "size_bytes": 13,
                "sha256": "7" * 64,
                "included_in_prompt": False,
            },
        }
        reviewed_memory["selection"] = {
            "task_name": selected_task_name,
            "task_directory": selected_task_name,
            "selection_sha256": "5" * 64,
            "prompt_sha256": "4" * 64,
            "roles": {
                "target_prior": selected_memory_path,
                "explore_experience": selected_memory_path.replace(
                    "target_prior.md", "explore_experience.md"
                ),
                "additional_expert_knowledge": [],
            },
            "files": {selected_memory_path: selected_metadata},
        }
        experience_path = reviewed_memory["selection"]["roles"]["explore_experience"]
        experience_metadata = {
            "relative_path": experience_path,
            "size_bytes": 14,
            "sha256": "3" * 64,
        }
        reviewed_memory["files"][experience_path] = {
            **experience_metadata,
            "included_in_prompt": True,
        }
        reviewed_memory["selection"]["files"][experience_path] = experience_metadata
        memory_files_sha256 = {
            selected_memory_path: selected_metadata["sha256"],
            experience_path: experience_metadata["sha256"],
        }
        resource_source = {
            "dataset_repo": "RLinf/RPent-memory",
            "repo_type": "dataset",
            "requested_revision": "main",
            "resolved_revision": "a" * 40,
            "subtree": "behavior",
            "manifest_sha256": "2" * 64,
            "files": [
                {
                    "path": "memory/snapshot_manifest.json",
                    "size_bytes": 123,
                    "sha256": "1" * 64,
                }
            ],
            "offline": False,
            "root": str((root / "resource-cache" / "behavior").absolute()),
        }
        run_resource_source = {
            **resource_source,
            "resolved_revision": run_resource_revision or "a" * 40,
        }

    receipt: dict[str, object] = {
        "schema_version": 1,
        "source": publication.RAW_SUCCESS_SOURCE,
        "run_nonce": "run-nonce",
        "attempt_nonce": "attempt-nonce",
        "attempt_index": attempt_index,
        "env_step": receipt_env_step,
        "raw_done": {"success": receipt_success},
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    receipt_bytes = _json_bytes(receipt)
    _write(attempt / "official_success_receipt.json", receipt_bytes)

    if action_record is None:
        action_record = {
            "env_step": action_env_step,
            "info_done": {"success": True},
        }
    action_bytes = (json.dumps(action_record, sort_keys=True) + "\n").encode()
    _write(attempt / "behavior_action_trace.jsonl", action_bytes)

    embedded_receipt = dict(receipt)
    if not tool_receipt_matches:
        embedded_receipt["env_step"] = 43
    tool_bytes = (
        json.dumps(
            {
                "step": 1,
                "tool": "press",
                "result": {
                    "task_success": True,
                    "official_success_source": publication.RAW_SUCCESS_SOURCE,
                    "run_nonce": "run-nonce",
                    "attempt_nonce": "attempt-nonce",
                    "official_success_receipt": embedded_receipt,
                },
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    _write(attempt / "behavior_tool_trace.jsonl", tool_bytes)

    final_bytes = _json_bytes(
        {
            "schema_version": 1,
            "task_success": True,
            "official_success_source": publication.RAW_SUCCESS_SOURCE,
            "runtime_cleanup": "complete",
            "job": {
                "job_id": job_id if final_job_matches else "different-job",
                "attempt_index": attempt_index,
            },
        }
    )
    _write(attempt / "final_result.json", final_bytes)

    run_memory = (
        reviewed_memory
        if run_memory_matches
        else {**reviewed_memory, "snapshot_sha256": "7" * 64}
    )
    if public_tool_contract_version == 1:
        run_manifest_schema_version = 5
        public_tools = publication.PUBLIC_TOOL_CONTRACTS[1]
    elif public_tool_contract_version == 2:
        run_manifest_schema_version = 6
        public_tools = publication.PUBLIC_TOOL_CONTRACTS[2]
    else:
        raise AssertionError("test fixture supports only public-tool contracts v1/v2")
    run_manifest_bytes = _json_bytes(
        {
            "schema_version": run_manifest_schema_version,
            "status": "stopped",
            "stopped_at": "2026-07-23T00:00:01Z",
            "job": {
                "job_id": job_id,
                "attempt_index": attempt_index,
            },
            "protocol": {
                "behavior_phase": "explore",
                "public_seed": identity.public_seed,
                "recipe_tag": identity.tag,
                **(
                    {"public_tool_contract_version": public_tool_contract_version}
                    if public_tool_contract_version == 2
                    else {}
                ),
                "public_primitives": list(public_tools),
                "agent_finish_registered": False,
                "task_spec": {
                    "task_name": identity.task_spec.task_name,
                    "prompt_profile_id": identity.task_spec.prompt_profile_id,
                },
                "prompt": {
                    "profile_id": identity.task_spec.prompt_profile_id,
                    "rendered_system_sha256": "a" * 64,
                    "rendered_user_sha256": "b" * 64,
                    "combined_sha256": "c" * 64,
                },
                "attempts": {
                    "initial_attempt_index": attempt_index,
                    "max_attempts": None,
                    "reset_registered": False,
                },
            },
            "task": {
                "suite": "behavior_2025_challenge",
                "task": identity.task_spec.task_index,
                "task_name": identity.task_spec.task_name,
                "public_seed": identity.public_seed,
            },
            "native_binding": {
                "activity_definition_id": identity.task_spec.activity_definition_id,
                "activity_instance_id": identity.native_instance,
                "env_seed": 0,
            },
            "reviewed_repo_memory": run_memory,
            "resource_source": run_resource_source,
            "frozen_eval_inputs": None,
            "processes": {
                "env": {
                    "managed": True,
                    "pid": None,
                    "start_ticks": None,
                    "stopped_at": "2026-07-23T00:00:00Z",
                },
                "vla": {
                    "managed": False,
                    "pid": None,
                    "stopped_at": None,
                },
            },
        }
    )
    _write(attempt / "run_manifest.json", run_manifest_bytes)

    session = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "succeeded",
        "finished_at": "2026-07-23T00:00:02Z",
        "blocked_reason": None,
        "protocol": {
            "behavior_phase": "explore",
            "task_index": identity.task_spec.task_index,
            "task_name": identity.task_spec.task_name,
            "public_seed": identity.public_seed,
            "recipe_tag": identity.tag,
            "reset_registered": False,
            "agent_finish_registered": False,
            **(
                {"public_tool_contract_version": public_tool_contract_version}
                if public_tool_contract_version == 2
                else {}
            ),
            "public_primitives": list(public_tools),
        },
        "native_binding": {
            "mapping_version": identity.task_spec.mapping_version,
            "activity_definition_id": identity.task_spec.activity_definition_id,
            "activity_instance_id": identity.native_instance,
            "env_seed": 0,
        },
        "reviewed_repo_memory": reviewed_memory,
        "resource_source": (None if omit_session_resource_source else resource_source),
        "processes": {
            "vla": {
                "managed": True,
                "pid": None,
                "start_ticks": None,
                "stopped_at": "2026-07-23T00:00:00Z",
            }
        },
        "attempts": [
            {
                "attempt_index": attempt_index,
                "outcome": "official_success",
                "task_success": True,
                "forced_cleanup_groups": {},
                "output_dir": str(attempt.resolve()),
            }
        ],
        "task_success": True,
        "artifact_seal_complete": False,
        "workflow_complete": False,
        "publication_complete": False,
    }
    session_bytes = _json_bytes(session)
    _write(root / "session_manifest.json", session_bytes)

    source_hashes = {
        "official_success_receipt": hashlib.sha256(receipt_bytes).hexdigest(),
        "behavior_action_trace": hashlib.sha256(action_bytes).hexdigest(),
        "behavior_tool_trace": hashlib.sha256(tool_bytes).hexdigest(),
        "final_result": hashlib.sha256(final_bytes).hexdigest(),
        "run_manifest": hashlib.sha256(run_manifest_bytes).hexdigest(),
        "session_manifest": hashlib.sha256(session_bytes).hexdigest(),
    }
    recipe_bytes = (
        json.dumps(
            {
                "kind": "task_level_symbolic_recipe",
                "task": identity.task_spec.task_name,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    memory_bytes = (
        f"# {identity.task_spec.task_name} task memory\n\n{task_memory_suffix}\n"
    ).encode()
    provenance = {
        "schema_version": provenance_schema_version,
        "derived_offline": True,
        "task": identity.task_spec.task_name,
        "task_index": identity.task_spec.task_index,
        "activity_definition_id": identity.task_spec.activity_definition_id,
        "activity_instance_id": identity.native_instance,
        "public_seed": identity.public_seed,
        "source": publication.PUBLICATION_SOURCE,
        "source_tag": identity.tag,
        "success_source": publication.RAW_SUCCESS_SOURCE,
        "job_id": job_id,
        "attempt_index": attempt_index,
        "attempt_nonce": "attempt-nonce",
        "task_success": True,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "memory_sha256": hashlib.sha256(memory_bytes).hexdigest(),
        "global_memory_snapshot_sha256": (
            provenance_memory_snapshot or memory_snapshot
        ),
        "global_memory_files_sha256": memory_files_sha256,
        "official_success_receipt": {
            "source": publication.RAW_SUCCESS_SOURCE,
            "run_nonce": "run-nonce",
            "attempt_nonce": "attempt-nonce",
            "attempt_index": attempt_index,
            "env_step": receipt_env_step,
            "receipt_sha256": receipt["receipt_sha256"],
            "file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        "source_artifacts_sha256": source_hashes,
    }
    provenance_bytes = _json_bytes(provenance)
    _write(root / identity.recipe_relative, recipe_bytes)
    _write(root / identity.memory_relative, memory_bytes)
    _write(root / identity.provenance_relative, provenance_bytes)

    payloads = {
        identity.recipe_relative: recipe_bytes,
        identity.memory_relative: memory_bytes,
        identity.provenance_relative: provenance_bytes,
    }
    bundle_id = canonical_bundle_id(payloads)
    bundle_root = root / ".publication_bundles" / bundle_id
    for relative, content in payloads.items():
        _write(bundle_root / relative, content)

    amendment = {
        "schema_version": 2,
        "kind": "posthoc_publication_override",
        "job_id": job_id,
        "tag": identity.tag,
        "public_seed": identity.public_seed,
        "success_source": publication.RAW_SUCCESS_SOURCE,
        "task_success": True,
        "publication_complete": True,
        "publication_source": publication.PUBLICATION_SOURCE,
        "attempt_index": attempt_index,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "memory_sha256": hashlib.sha256(memory_bytes).hexdigest(),
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "global_memory_snapshot_sha256": (amendment_memory_snapshot or memory_snapshot),
        "bundle_id": bundle_id,
        "original_attempt_immutable": True,
        "artifact_seal_complete": False,
        "overlay_semantics": {
            "overrides": {"publication_complete": True},
            "preserves_session_manifest": {
                "task_success": True,
                "artifact_seal_complete": False,
                "workflow_complete": False,
                "publication_complete": False,
            },
        },
    }
    _write(root / "publication_amendment.json", _json_bytes(amendment))
    return {
        "job_id": job_id,
        "bundle_id": bundle_id,
        "provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
        "attempt": attempt,
        "identity": identity,
    }


def _build_forensic_publication(
    root: Path,
    *,
    identity_tier: str = "job_attempt_nonce",
) -> dict[str, object]:
    identity = publication.resolve_publication_identity(
        task_name="picking_up_trash",
        public_seed=1,
    )
    job_id = "behavior-forensic-test"
    attempt_index = 1
    attempt = root / "attempts" / identity.tag / "attempt_001"
    attempt.mkdir(parents=True)
    _write(
        root / publication.SESSION_MANIFEST_RELATIVE,
        _json_bytes(
            {
                "schema_version": 1,
                "job_id": job_id,
                "protocol": {
                    "task_index": identity.task_spec.task_index,
                    "task_name": identity.task_spec.task_name,
                    "public_seed": identity.public_seed,
                },
            }
        ),
    )

    action_bytes = (
        json.dumps(
            {"step": 7, "info_done": {"success": True}},
            sort_keys=True,
        )
        + "\n"
    ).encode()
    _write(attempt / "behavior_action_trace.jsonl", action_bytes)
    action_sha256 = hashlib.sha256(action_bytes).hexdigest()
    binding = {
        "source": "behavior_action_trace",
        "field_path": "info_done.success",
        "first_success_step": 7,
        "action_trace_sha256": action_sha256,
    }
    if identity_tier == "job_attempt_nonce":
        source_identity = {
            "job_id": job_id,
            "attempt_index": attempt_index,
            "nonce": "attempt-nonce",
        }
    elif identity_tier == "job_attempt":
        source_identity = {
            "job_id": job_id,
            "attempt_index": attempt_index,
        }
    elif identity_tier == "job_root_attempt":
        source_identity = {
            "job_root_path": str(root.resolve()),
            "attempt_index": attempt_index,
        }
    else:
        raise AssertionError(f"unsupported test identity tier: {identity_tier}")

    receipt = {
        "schema_version": 1,
        "receipt_type": "forensic_action_trace_receipt",
        "publication_identity": source_identity,
        "official_success_binding": binding,
    }
    receipt_bytes = _json_bytes(receipt)
    _write(attempt / "official_success_receipt.json", receipt_bytes)
    recipe_bytes = (
        json.dumps(
            {
                "kind": "task_level_symbolic_recipe",
                "task": identity.task_spec.task_name,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()
    _write(attempt / Path(identity.recipe_relative).name, recipe_bytes)
    memory_bytes = b"# picking_up_trash task memory\n\nUse semantic evidence.\n"
    _write(root / identity.memory_relative, memory_bytes)
    provenance = {
        "schema_version": 1,
        "task": identity.task_spec.task_name,
        "publication_identity": source_identity,
        "official_success_binding": binding,
        "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
        "memory_sha256": hashlib.sha256(memory_bytes).hexdigest(),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    _write(root / identity.provenance_relative, _json_bytes(provenance))
    return {
        "attempt": attempt,
        "binding": binding,
        "identity": identity,
        "provenance": root / identity.provenance_relative,
        "memory": root / identity.memory_relative,
        "recipe": attempt / Path(identity.recipe_relative).name,
        "receipt": attempt / "official_success_receipt.json",
        "action_trace": attempt / "behavior_action_trace.jsonl",
    }


def test_canonical_bundle_id_matches_publisher_algorithm() -> None:
    payloads = {"b": b"two", "a": b"one"}
    expected = hashlib.sha256()
    for name in ("a", "b"):
        expected.update(name.encode())
        expected.update(b"\0")
        expected.update(payloads[name])
    assert canonical_bundle_id(payloads) == expected.hexdigest()


def test_validates_complete_canonical_publication(tmp_path: Path) -> None:
    root = tmp_path / "job"
    built = _build_publication(root)

    result = validate_canonical_publication_root(
        root,
        expected_provenance_sha256=str(built["provenance_sha256"]),
        expected_job_id=str(built["job_id"]),
    )

    assert result.root == root.absolute()
    assert result.bundle_id == built["bundle_id"]
    assert result.recipe_bytes.startswith(b'{"kind"')
    assert result.memory_bytes.startswith(b"# turning_on_radio")
    assert result.provenance["derived_offline"] is True
    assert result.amendment["publication_complete"] is True
    assert result.manifest_binding["global_memory_snapshot_sha256"] == "9" * 64
    assert result.manifest_binding["bundle_id"] == result.bundle_id
    assert publication.SESSION_MANIFEST_RELATIVE in result.files
    assert (
        f".publication_bundles/{result.bundle_id}/{publication.RECIPE_RELATIVE}"
        in result.files
    )


def test_publication_validator_accepts_immutable_v1_and_current_v2_sources(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy-v1-job"
    current_root = tmp_path / "current-v2-job"
    _build_publication(legacy_root, public_tool_contract_version=1)
    _build_publication(current_root, public_tool_contract_version=2)

    legacy = validate_canonical_publication_root(legacy_root)
    current = validate_canonical_publication_root(current_root)

    legacy_run = json.loads(
        (
            legacy_root
            / "attempts"
            / legacy.identity.tag
            / "attempt_001"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    current_run = json.loads(
        (
            current_root
            / "attempts"
            / current.identity.tag
            / "attempt_001"
            / "run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert legacy_run["schema_version"] == 5
    assert "public_tool_contract_version" not in legacy_run["protocol"]
    assert (
        tuple(legacy_run["protocol"]["public_primitives"])
        == (publication.PUBLIC_TOOL_CONTRACTS[1])
    )
    assert current_run["schema_version"] == 6
    assert current_run["protocol"]["public_tool_contract_version"] == 2
    assert (
        tuple(current_run["protocol"]["public_primitives"])
        == (publication.PUBLIC_TOOL_CONTRACTS[2])
    )


def test_validates_trash_publication_with_auto_and_explicit_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-job"
    built = _build_publication(root, task_name="picking_up_trash")

    automatic = validate_canonical_publication_root(root)
    explicit = validate_canonical_publication_root(
        root,
        task_name="picking_up_trash",
        task_index=1,
        public_seed=0,
    )

    assert automatic.identity == explicit.identity == built["identity"]
    assert automatic.identity.native_instance == 196
    assert automatic.identity.tag == "picking_up_trash_s0"
    assert automatic.identity.recipe_relative == "recipe_picking_up_trash_s0.jsonl"
    assert automatic.identity.memory_relative == "memory/picking_up_trash.md"
    assert (
        automatic.identity.provenance_relative
        == "memory/picking_up_trash_provenance.json"
    )
    assert automatic.manifest_binding["task_name"] == "picking_up_trash"
    assert automatic.manifest_binding["task_index"] == 1
    assert automatic.manifest_binding["activity_instance_id"] == 196
    assert b"turning_on_radio" not in automatic.recipe_bytes
    assert b"turning_on_radio" not in automatic.memory_bytes


def test_validates_trash_s1_publication_identity_and_paths(tmp_path: Path) -> None:
    root = tmp_path / "trash-s1-job"
    built = _build_publication(
        root,
        task_name="picking_up_trash",
        public_seed=1,
    )

    automatic = validate_canonical_publication_root(root)
    explicit = validate_canonical_publication_root(
        root,
        task_name="picking_up_trash",
        task_index=1,
        public_seed=1,
    )

    assert automatic.identity == explicit.identity == built["identity"]
    assert automatic.identity.public_seed == 1
    assert automatic.identity.native_instance == 67
    assert automatic.identity.tag == "picking_up_trash_s1"
    assert automatic.identity.recipe_relative == "recipe_picking_up_trash_s1.jsonl"
    assert automatic.manifest_binding["source_public_seed"] == 1
    assert automatic.manifest_binding["source_tag"] == "picking_up_trash_s1"


def test_trash_s1_publication_rejects_inconsistent_tag(tmp_path: Path) -> None:
    root = tmp_path / "trash-s1-wrong-tag"
    _build_publication(
        root,
        task_name="picking_up_trash",
        public_seed=1,
    )
    manifest_path = root / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"]["recipe_tag"] = "picking_up_trash_s0"
    manifest_path.write_bytes(_json_bytes(manifest))

    with pytest.raises(
        PublicationValidationError,
        match="canonical Explore protocol",
    ):
        validate_canonical_publication_root(root)


def test_trash_s1_publication_rejects_inconsistent_recipe_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-s1-wrong-path"
    _build_publication(
        root,
        task_name="picking_up_trash",
        public_seed=1,
    )
    (root / "recipe_picking_up_trash_s1.jsonl").rename(
        root / "recipe_picking_up_trash_s0.jsonl"
    )

    with pytest.raises(
        PublicationValidationError,
        match="recipe_picking_up_trash_s1.jsonl",
    ):
        validate_canonical_publication_root(root)


def test_schema3_trash_publication_projects_only_task_memory_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-schema3"
    _build_publication(
        root,
        task_name="picking_up_trash",
        provenance_schema_version=3,
    )

    result = validate_canonical_publication_root(root)

    memory_hashes = result.provenance["global_memory_files_sha256"]
    assert set(memory_hashes) == {
        "picking_up_trash/target_prior.md",
        "picking_up_trash/explore_experience.md",
    }
    assert not any(name.startswith("turning_on_radio/") for name in memory_hashes)


def test_schema3_publication_rejects_selection_leaf_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-schema3-leaf-mismatch"
    _build_publication(
        root,
        task_name="picking_up_trash",
        provenance_schema_version=3,
        selection_file_sha256="6" * 64,
    )

    with pytest.raises(
        PublicationValidationError,
        match="strictly task-scoped",
    ):
        validate_canonical_publication_root(root)


def test_schema3_publication_rejects_child_resource_source_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-schema3-resource-mismatch"
    _build_publication(
        root,
        task_name="picking_up_trash",
        provenance_schema_version=3,
        run_resource_revision="b" * 40,
    )

    with pytest.raises(
        PublicationValidationError,
        match="run manifest lifecycle or Job binding",
    ):
        validate_canonical_publication_root(root)


def test_schema3_publication_requires_complete_session_resource_source(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-schema3-resource-missing"
    _build_publication(
        root,
        task_name="picking_up_trash",
        provenance_schema_version=3,
        omit_session_resource_source=True,
    )

    with pytest.raises(
        PublicationValidationError,
        match="complete pinned dataset resource binding",
    ):
        validate_canonical_publication_root(root)


def test_schema3_trash_publication_rejects_other_task_memory_selection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-schema3-cross-task"
    _build_publication(
        root,
        task_name="picking_up_trash",
        provenance_schema_version=3,
        selection_task_name="turning_on_radio",
    )

    with pytest.raises(
        PublicationValidationError,
        match="task-scoped reviewed Global Memory selection",
    ):
        validate_canonical_publication_root(root)


def test_rejects_cross_task_identity_and_radio_policy_in_trash_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "trash-job"
    _build_publication(
        root,
        task_name="picking_up_trash",
        task_memory_suffix="Reuse the radio_tipped_flat button-face rule.",
    )

    with pytest.raises(PublicationValidationError, match="Radio-only"):
        validate_canonical_publication_root(root)
    with pytest.raises(PublicationValidationError):
        validate_canonical_publication_root(
            root,
            task_name="turning_on_radio",
            task_index=0,
            public_seed=0,
        )


def test_rejects_legacy_nested_info_done_success_fallback(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(
        root,
        action_record={
            "env_idx": 0,
            "step": 41,
            "info": {"done": {"success": True}},
        },
    )

    with pytest.raises(
        PublicationValidationError,
        match="action trace lacks raw success",
    ):
        validate_canonical_publication_root(root)


def test_current_action_trace_env_step_is_authoritative(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(
        root,
        action_record={
            "env_step": 42,
            "env_idx": 0,
            "step": 999,
            "info_done": {"success": True},
        },
    )

    validate_canonical_publication_root(root)


@pytest.mark.parametrize(
    ("receipt_env_step", "action_record"),
    [
        (
            42,
            {
                "env_idx": 1,
                "step": 41,
                "info_done": {"success": True},
            },
        ),
        (
            42,
            {
                "env_idx": False,
                "step": 41,
                "info_done": {"success": True},
            },
        ),
        (
            42,
            {
                "env_idx": 0,
                "step": 40,
                "info_done": {"success": True},
            },
        ),
        (
            2,
            {
                "env_idx": 0,
                "step": True,
                "info_done": {"success": True},
            },
        ),
        (
            1,
            {
                "env_step": True,
                "info_done": {"success": True},
            },
        ),
        (
            42,
            {
                "env_step": 41,
                "env_idx": 0,
                "step": 41,
                "info_done": {"success": True},
            },
        ),
        (
            42,
            {
                "env_step": 42,
                "env_idx": 1,
                "step": 41,
                "info_done": {"success": True},
            },
        ),
        (
            42,
            {
                "env_step": 42,
                "env_idx": True,
                "step": 41,
                "info_done": {"success": True},
            },
        ),
    ],
)
def test_rejects_invalid_action_trace_lineage(
    tmp_path: Path,
    receipt_env_step: int,
    action_record: dict[str, object],
) -> None:
    root = tmp_path / "job"
    _build_publication(
        root,
        receipt_env_step=receipt_env_step,
        action_record=action_record,
    )

    with pytest.raises(
        PublicationValidationError,
        match="action trace lacks raw success",
    ):
        validate_canonical_publication_root(root)


@pytest.mark.parametrize(
    "relative",
    [
        publication.AMENDMENT_RELATIVE,
        publication.SESSION_MANIFEST_RELATIVE,
        publication.RECIPE_RELATIVE,
        publication.MEMORY_RELATIVE,
        publication.PROVENANCE_RELATIVE,
    ],
)
def test_rejects_missing_canonical_artifact(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    (root / relative).unlink()

    with pytest.raises(PublicationValidationError):
        validate_canonical_publication_root(root)


def test_rejects_missing_or_noncanonical_hidden_bundle(tmp_path: Path) -> None:
    root = tmp_path / "job"
    built = _build_publication(root)
    bundle = root / ".publication_bundles" / str(built["bundle_id"])
    shutil.rmtree(bundle)

    with pytest.raises(PublicationValidationError, match="directory"):
        validate_canonical_publication_root(root)

    _build_publication(tmp_path / "other")
    # Recreate the expected directory with one extra file: exact membership matters.
    bundle.mkdir(parents=True)
    _write(bundle / publication.RECIPE_RELATIVE, b"wrong")
    _write(bundle / publication.MEMORY_RELATIVE, b"wrong")
    _write(bundle / publication.PROVENANCE_RELATIVE, b"wrong")
    _write(bundle / "extra.txt", b"not canonical")
    with pytest.raises(PublicationValidationError, match="entries"):
        validate_canonical_publication_root(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "stopped_by_operator"),
        ("task_success", False),
    ],
)
def test_rejects_non_successful_job_manifest(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    path = root / publication.SESSION_MANIFEST_RELATIVE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_bytes(_json_bytes(manifest))

    with pytest.raises(
        PublicationValidationError,
        match="immutable successful Explore Job",
    ):
        validate_canonical_publication_root(root)


def test_rejects_incomplete_publication_amendment(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    path = root / publication.AMENDMENT_RELATIVE
    amendment = json.loads(path.read_text(encoding="utf-8"))
    amendment["publication_complete"] = False
    path.write_bytes(_json_bytes(amendment))

    with pytest.raises(PublicationValidationError, match="amendment identity"):
        validate_canonical_publication_root(root)


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    path = root / publication.AMENDMENT_RELATIVE
    raw = path.read_bytes()
    path.write_bytes(
        raw.replace(
            b'"publication_complete": true,',
            b'"publication_complete": true,\n  "publication_complete": true,',
            1,
        )
    )

    with pytest.raises(PublicationValidationError, match="duplicate JSON key"):
        validate_canonical_publication_root(root)


def test_rejects_symlinked_core_file_and_hidden_root(tmp_path: Path) -> None:
    root = tmp_path / "job"
    built = _build_publication(root)
    outside = tmp_path / "outside-memory.md"
    outside.write_bytes((root / publication.MEMORY_RELATIVE).read_bytes())
    (root / publication.MEMORY_RELATIVE).unlink()
    (root / publication.MEMORY_RELATIVE).symlink_to(outside)

    with pytest.raises(PublicationValidationError, match="symlink"):
        validate_canonical_publication_root(root)

    hidden = root / ".publication_bundles" / str(built["bundle_id"])
    with pytest.raises(PublicationValidationError, match="visible Job root"):
        validate_canonical_publication_root(hidden)


def test_rejects_symlinked_intermediate_directory_from_another_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    other = tmp_path / "other"
    _build_publication(other)
    shutil.rmtree(root / "memory")
    (root / "memory").symlink_to(other / "memory", target_is_directory=True)

    with pytest.raises(PublicationValidationError):
        validate_canonical_publication_root(root)


def test_rejects_publication_root_reached_through_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    root = real_parent / "job"
    _build_publication(root)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(PublicationValidationError, match="ancestor"):
        validate_canonical_publication_root(alias / "job")


def test_rejects_wrong_external_pin_or_job(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(root)

    with pytest.raises(PublicationValidationError, match="trusted pin"):
        validate_canonical_publication_root(
            root,
            expected_provenance_sha256="0" * 64,
        )
    with pytest.raises(PublicationValidationError, match="expected Job"):
        validate_canonical_publication_root(
            root,
            expected_job_id="different-job",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"receipt_success": False}, "receipt is invalid"),
        ({"action_env_step": 41}, "action trace lacks raw success"),
        ({"tool_receipt_matches": False}, "exact receipt"),
        ({"final_job_matches": False}, "final result"),
        ({"run_memory_matches": False}, "run manifest lifecycle"),
        (
            {"provenance_memory_snapshot": "6" * 64},
            "Global Memory snapshot",
        ),
        (
            {"amendment_memory_snapshot": "6" * 64},
            "Global Memory snapshot",
        ),
    ],
)
def test_rejects_broken_raw_success_or_memory_binding(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    root = tmp_path / "job"
    _build_publication(root, **kwargs)

    with pytest.raises(PublicationValidationError, match=message):
        validate_canonical_publication_root(root)


@pytest.mark.parametrize(
    "filename",
    [
        "official_success_receipt.json",
        "behavior_action_trace.jsonl",
        "behavior_tool_trace.jsonl",
        "final_result.json",
        "run_manifest.json",
    ],
)
def test_rejects_mutated_attempt_source_artifact(
    tmp_path: Path,
    filename: str,
) -> None:
    root = tmp_path / "job"
    built = _build_publication(root)
    attempt = built["attempt"]
    assert isinstance(attempt, Path)
    with (attempt / filename).open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(PublicationValidationError, match="hash mismatch"):
        validate_canonical_publication_root(root)


def test_rejects_session_source_mutation(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    with (root / "session_manifest.json").open("ab") as stream:
        stream.write(b" ")

    with pytest.raises(PublicationValidationError, match="hash mismatch"):
        validate_canonical_publication_root(root)


def test_rejects_change_during_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    original = publication._RootReader.verify_stable

    def mutate_then_verify(reader: publication._RootReader) -> None:
        with (root / publication.RECIPE_RELATIVE).open("ab") as stream:
            stream.write(b" ")
        original(reader)

    monkeypatch.setattr(
        publication._RootReader,
        "verify_stable",
        mutate_then_verify,
    )
    with pytest.raises(PublicationValidationError, match="changed"):
        validate_canonical_publication_root(root)


def test_rejects_oversized_control_file(tmp_path: Path) -> None:
    root = tmp_path / "job"
    _build_publication(root)
    (root / publication.MEMORY_RELATIVE).write_bytes(
        b"x" * (publication._CONTROL_FILE_LIMIT + 1)
    )

    with pytest.raises(PublicationValidationError, match="size limit"):
        validate_canonical_publication_root(root)


@pytest.mark.parametrize(
    "identity_tier",
    [
        "job_attempt_nonce",
        "job_attempt",
        "job_root_attempt",
    ],
)
def test_forensic_publication_accepts_three_strict_identity_tiers(
    tmp_path: Path,
    identity_tier: str,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root, identity_tier=identity_tier)

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is True
    assert result.identity_tier == identity_tier
    assert result.files is not None
    assert set(result.files) == {"recipe", "memory", "provenance", "receipt"}


def test_forensic_root_attempt_identity_does_not_require_session_job_id(
    tmp_path: Path,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(
        root,
        identity_tier="job_root_attempt",
    )
    session_path = root / publication.SESSION_MANIFEST_RELATIVE
    session = json.loads(session_path.read_text())
    session.pop("job_id")
    session_path.write_bytes(_json_bytes(session))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is True
    assert result.identity_tier == "job_root_attempt"


@pytest.mark.parametrize(
    "identity_tier",
    ["job_attempt_nonce", "job_attempt"],
)
def test_forensic_job_identity_requires_session_job_id(
    tmp_path: Path,
    identity_tier: str,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root, identity_tier=identity_tier)
    session_path = root / publication.SESSION_MANIFEST_RELATIVE
    session = json.loads(session_path.read_text())
    session.pop("job_id")
    session_path.write_bytes(_json_bytes(session))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False
    assert "identity" in result.reason


def test_forensic_publication_accepts_root_local_recipe_fallback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    recipe = built["recipe"]
    identity = built["identity"]
    assert isinstance(recipe, Path)
    assert isinstance(identity, publication.BehaviorPublicationIdentity)
    root_recipe = root / identity.recipe_relative
    root_recipe.write_bytes(recipe.read_bytes())
    recipe.unlink()

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is True
    assert result.files is not None
    assert result.files["recipe"]["relative_path"] == identity.recipe_relative


@pytest.mark.parametrize(
    "artifact_name",
    ["recipe", "memory", "provenance", "receipt"],
)
def test_forensic_publication_missing_artifact_is_incomplete(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    artifact = built[artifact_name]
    assert isinstance(artifact, Path)
    artifact.unlink()

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False


@pytest.mark.parametrize(
    "document_name",
    ["provenance", "receipt"],
)
@pytest.mark.parametrize(
    "missing_field",
    ["field_path", "first_success_step", "action_trace_sha256"],
)
def test_forensic_publication_requires_evidence_in_provenance_and_receipt(
    tmp_path: Path,
    document_name: str,
    missing_field: str,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    path = built[document_name]
    assert isinstance(path, Path)
    document = json.loads(path.read_text())
    document["official_success_binding"].pop(missing_field)
    path.write_bytes(_json_bytes(document))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False


@pytest.mark.parametrize(
    "missing_hash",
    ["recipe_sha256", "memory_sha256", "receipt_sha256"],
)
def test_forensic_publication_requires_all_payload_hashes(
    tmp_path: Path,
    missing_hash: str,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    provenance = built["provenance"]
    assert isinstance(provenance, Path)
    document = json.loads(provenance.read_text())
    document.pop(missing_hash)
    provenance.write_bytes(_json_bytes(document))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False
    assert "payload hashes" in result.reason


def test_forensic_publication_rejects_incomplete_identity(tmp_path: Path) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    receipt = built["receipt"]
    assert isinstance(receipt, Path)
    document = json.loads(receipt.read_text())
    document["publication_identity"].pop("attempt_index")
    receipt.write_bytes(_json_bytes(document))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False
    assert "identity" in result.reason


def test_forensic_publication_rejects_first_success_step_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    binding = dict(built["binding"])
    binding["first_success_step"] = 8

    for document_name in ("provenance", "receipt"):
        path = built[document_name]
        assert isinstance(path, Path)
        document = json.loads(path.read_text())
        document["official_success_binding"] = binding
        path.write_bytes(_json_bytes(document))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        binding,
    )

    assert result.complete is False
    assert "action trace" in result.reason


def test_forensic_publication_rejects_action_trace_hash_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    action_trace = built["action_trace"]
    assert isinstance(action_trace, Path)
    with action_trace.open("ab") as stream:
        stream.write(b'{"step":8,"info_done":{"success":false}}\n')

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False
    assert "action trace" in result.reason


def test_forensic_publication_rejects_identity_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "job"
    built = _build_forensic_publication(root)
    receipt = built["receipt"]
    assert isinstance(receipt, Path)
    document = json.loads(receipt.read_text())
    document["publication_identity"]["nonce"] = "different-attempt"
    receipt.write_bytes(_json_bytes(document))

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False
    assert "identity" in result.reason


def test_forensic_s1_shape_without_memory_or_provenance_is_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "picking_up_trash_s1_history"
    built = _build_forensic_publication(root)
    memory = built["memory"]
    provenance = built["provenance"]
    assert isinstance(memory, Path)
    assert isinstance(provenance, Path)
    memory.unlink()
    provenance.unlink()

    result = publication.validate_forensic_publication_binding(
        root,
        built["attempt"],
        built["binding"],
    )

    assert result.complete is False
