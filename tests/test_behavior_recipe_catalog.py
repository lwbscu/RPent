from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robots.behavior.recipe_catalog import (
    CANDIDATE_REVIEWED_PROVENANCE,
    CANONICAL_PUBLIC_PROVENANCE,
    EXPLORE_CONSUMER,
    FORMAL_EVAL_CONSUMER,
    BehaviorRecipeCatalogError,
    load_behavior_recipe_catalog,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "behavior_resources"
REVIEWED_RECIPES = FIXTURE_ROOT / "behavior" / "recipes"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_entry(
    root: Path,
    *,
    entry_id: str,
    provenance: str,
    consumers: list[str],
    task: str = "turning_on_radio",
    recipe_text: str | None = None,
    memory_text: str | None = None,
) -> dict:
    directory = root / task / entry_id
    directory.mkdir(parents=True)
    recipe = recipe_text or (
        '{"schema_version":1,"kind":"task_level_symbolic_recipe",'
        f'"task":"{task}","source":"raw_official_success_v1",'
        '"policy":"Adapt semantic constraints to fresh public evidence."}\n'
        '{"kind":"semantic_goal","goal":"Interact with the real physical control '
        'and require runtime-owned raw official success."}\n'
    )
    memory = memory_text or (
        "# Reviewed task memory\n\n"
        "Ground semantic identity and control evidence in the active episode.\n"
    )
    recipe_path = directory / "recipe.jsonl"
    memory_path = directory / "task_memory.md"
    recipe_path.write_text(recipe, encoding="utf-8")
    memory_path.write_text(memory, encoding="utf-8")
    recipe_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    memory_sha = hashlib.sha256(memory_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "kind": "reviewed_behavior_recipe_promotion",
        "recipe_id": entry_id,
        "task": task,
        "provenance_class": provenance,
        "allowed_consumers": consumers,
        "review_status": "accepted",
        "task_success": True,
        "official_success_checked": True,
        "official_success_source": 'info["done"]["success"]',
        "payload_sha256": {
            "recipe": recipe_sha,
            "task_memory": memory_sha,
        },
        "source_evidence": [
            *(
                [
                    {
                        "kind": "canonical_publication_validation",
                        "sha256": "1" * 64,
                        "validation": "canonical_publication_validated",
                    }
                ]
                if provenance == CANONICAL_PUBLIC_PROVENANCE
                else []
            ),
            {
                "kind": "official_success_receipt",
                "sha256": "2" * 64,
                "validation": "raw_info_done_success_true",
            },
        ],
    }
    receipt_path = directory / "promotion_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {}
    for role, path in (
        ("recipe", recipe_path),
        ("task_memory", memory_path),
        ("promotion_receipt", receipt_path),
    ):
        raw = path.read_bytes()
        files[role] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    entry = {
        "id": entry_id,
        "task": task,
        "provenance_class": provenance,
        "allowed_consumers": consumers,
        "review_status": "accepted",
        "task_success": True,
        "official_success_checked": True,
        "official_success_source": 'info["done"]["success"]',
        "files": files,
    }
    entry["entry_sha256"] = _digest(entry)
    return entry


def _write_catalog(root: Path, entries: list[dict]) -> Path:
    entries = sorted(entries, key=lambda item: item["id"])
    manifest = {
        "schema_version": 1,
        "kind": "reviewed_behavior_recipe_catalog",
        "environment": "behavior",
        "entries": entries,
        "catalog_sha256": _digest(
            [
                {
                    "entry_id": entry["id"],
                    "entry_sha256": entry["entry_sha256"],
                }
                for entry in entries
            ]
        ),
    }
    (root / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _valid_catalog(tmp_path: Path) -> Path:
    root = tmp_path / "recipes"
    root.mkdir(parents=True)
    canonical = _write_entry(
        root,
        entry_id="canonical_control_access_v1",
        provenance=CANONICAL_PUBLIC_PROVENANCE,
        consumers=[EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER],
    )
    candidate = _write_entry(
        root,
        entry_id="reviewed_control_recovery_v1",
        provenance=CANDIDATE_REVIEWED_PROVENANCE,
        consumers=[EXPLORE_CONSUMER],
    )
    return _write_catalog(root, [candidate, canonical])


def _valid_trash_v2_catalog(
    tmp_path: Path,
    *,
    official_success: bool = True,
    action_trace_sha256: str = "3" * 64,
) -> Path:
    root = tmp_path / "recipes"
    directory = root / "picking_up_trash" / "open_receptacle_collection"
    directory.mkdir(parents=True)
    recipe_path = directory / "recipe.jsonl"
    memory_path = directory / "memory.md"
    provenance_path = directory / "provenance.json"
    receipt_path = directory / "receipt.json"
    recipe_path.write_text(
        '{"schema_version":1,"kind":"task_level_symbolic_recipe",'
        '"task":"picking_up_trash","source":"raw_official_success_v1",'
        '"policy":"Adapt containment decisions to current semantic evidence."}\n'
        '{"kind":"semantic_goal","goal":"Keep a task-relevant upright receptacle '
        'stable while collecting every required soda can."}\n',
        encoding="utf-8",
    )
    memory_path.write_text(
        "# Reviewed trash memory\n\n"
        "Both hands may carry task-relevant objects. Re-identify their current "
        "contents from fresh evidence.\n",
        encoding="utf-8",
    )
    recipe_sha256 = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    memory_sha256 = hashlib.sha256(memory_path.read_bytes()).hexdigest()
    source = {
        "job": "source-job",
        "attempt": 1,
        "task": {
            "name": "picking_up_trash",
            "activity_definition_id": 0,
            "activity_instance_id": 196,
        },
        "official_success": {
            "field": 'info["done"]["success"]',
            "value": official_success,
            "env_step": 12835,
        },
        "action_trace_sha256": action_trace_sha256,
        "official_success_receipt_sha256": "4" * 64,
        "official_success_receipt_payload_sha256": "5" * 64,
        "publication_sha256": "6" * 64,
        "vla_calls": [
            {"complete_chunks": 150, "partial_steps": 0},
            {"complete_chunks": 87, "partial_steps": 0},
            {"complete_chunks": 164, "partial_steps": 3},
        ],
    }
    provenance = {
        "schema_version": 1,
        "kind": "behavior_recipe_provenance",
        "task": "picking_up_trash",
        "source": source,
        "payload_sha256": {
            "recipe": recipe_sha256,
            "memory": memory_sha256,
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance_sha256 = hashlib.sha256(provenance_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "kind": "reviewed_behavior_recipe_receipt",
        "recipe_id": "open_receptacle_collection",
        "task": "picking_up_trash",
        "provenance_class": CANONICAL_PUBLIC_PROVENANCE,
        "allowed_consumers": [EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER],
        "review_status": "accepted",
        "task_success": True,
        "official_success_checked": True,
        "official_success_source": 'info["done"]["success"]',
        "source": source,
        "payload_sha256": {
            "recipe": recipe_sha256,
            "memory": memory_sha256,
            "provenance": provenance_sha256,
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {}
    for role, path in (
        ("recipe", recipe_path),
        ("memory", memory_path),
        ("provenance", provenance_path),
        ("receipt", receipt_path),
    ):
        raw = path.read_bytes()
        files[role] = {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    entry = {
        "id": "open_receptacle_collection",
        "task": "picking_up_trash",
        "provenance_class": CANONICAL_PUBLIC_PROVENANCE,
        "allowed_consumers": [EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER],
        "review_status": "accepted",
        "task_success": True,
        "official_success_checked": True,
        "official_success_source": 'info["done"]["success"]',
        "files": files,
    }
    entry["entry_sha256"] = _digest(entry)
    manifest = {
        "schema_version": 2,
        "kind": "reviewed_behavior_recipe_catalog",
        "environment": "behavior",
        "entries": [entry],
        "catalog_sha256": _digest(
            [
                {
                    "entry_id": entry["id"],
                    "entry_sha256": entry["entry_sha256"],
                }
            ]
        ),
    }
    (root / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _load_manifest(root: Path) -> dict:
    return json.loads((root / "catalog_manifest.json").read_text(encoding="utf-8"))


def _rewrite_manifest(root: Path, manifest: dict) -> None:
    (root / "catalog_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rehash_catalog_payload(root: Path, *, entry_id: str, role: str) -> None:
    manifest = _load_manifest(root)
    entry = next(item for item in manifest["entries"] if item["id"] == entry_id)
    path = root / entry["files"][role]["path"]
    raw = path.read_bytes()
    entry["files"][role]["size_bytes"] = len(raw)
    entry["files"][role]["sha256"] = hashlib.sha256(raw).hexdigest()
    entry_without_hash = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    entry["entry_sha256"] = _digest(entry_without_hash)
    manifest["catalog_sha256"] = _digest(
        [
            {
                "entry_id": item["id"],
                "entry_sha256": item["entry_sha256"],
            }
            for item in manifest["entries"]
        ]
    )
    _rewrite_manifest(root, manifest)


def _valid_candidate_promotion_v2_catalog(tmp_path: Path) -> Path:
    root = _valid_catalog(tmp_path)
    entry_id = "reviewed_control_recovery_v1"
    receipt_path = root / "turning_on_radio" / entry_id / "promotion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema_version"] = 2
    receipt["source"] = {
        "job": "behavior-explore-source",
        "attempt": 1,
        "task": {
            "name": "turning_on_radio",
            "activity_definition_id": 0,
            "activity_instance_id": 242,
        },
        "action_trace_sha256": "3" * 64,
        "review": {
            "kind": "behavior_attempt_raw_success_recovery",
            "sha256": "4" * 64,
            "scope": "attempt_only",
            "status": "accepted",
            "outer_job_reclassified": False,
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rehash_catalog_payload(
        root,
        entry_id=entry_id,
        role="promotion_receipt",
    )
    return root


def test_catalog_loads_and_selects_by_consumer_deterministically(
    tmp_path: Path,
) -> None:
    catalog = load_behavior_recipe_catalog(_valid_catalog(tmp_path))

    explore = catalog.select("turning_on_radio", EXPLORE_CONSUMER)
    formal = catalog.select("turning_on_radio", FORMAL_EVAL_CONSUMER)

    assert explore.selected_ids == (
        "canonical_control_access_v1",
        "reviewed_control_recovery_v1",
    )
    assert formal.selected_ids == ("canonical_control_access_v1",)
    assert catalog.catalog_sha256 == catalog.manifest_binding.declared_catalog_sha256
    assert len(catalog.files) == 6
    assert "promotion_receipt.json" not in explore.prompt_text
    assert "runtime-owned raw official success" in explore.prompt_text
    assert explore.public_binding["catalog_sha256"] == catalog.catalog_sha256
    assert {
        item["provenance_class"] for item in formal.public_binding["selected_entries"]
    } == {CANONICAL_PUBLIC_PROVENANCE}
    assert all(
        "/" not in entry["entry_id"]
        for entry in explore.public_binding["selected_entries"]
    )


def test_supported_task_empty_selection_remains_catalog_bound(tmp_path: Path) -> None:
    catalog = load_behavior_recipe_catalog(_valid_catalog(tmp_path))

    selection = catalog.select("picking_up_trash", EXPLORE_CONSUMER)

    assert selection.selected_ids == ()
    assert selection.prompt_text == ""
    assert selection.task_name == "picking_up_trash"
    assert selection.public_binding["selected_entries"] == []
    assert selection.public_binding["catalog_sha256"] == catalog.catalog_sha256


def test_fixture_catalog_selects_no_trash_recipe() -> None:
    catalog = load_behavior_recipe_catalog(REVIEWED_RECIPES)

    selection = catalog.select("picking_up_trash", EXPLORE_CONSUMER)

    assert selection.selected_ids == ()
    assert selection.prompt_text == ""
    assert selection.public_binding["task"] == "picking_up_trash"
    assert selection.public_binding["consumer"] == EXPLORE_CONSUMER
    assert selection.public_binding["selected_entries"] == []
    assert selection.public_binding["catalog_sha256"] == catalog.catalog_sha256


def test_candidate_promotion_v2_binds_composite_attempt_review_source(
    tmp_path: Path,
) -> None:
    catalog = load_behavior_recipe_catalog(
        _valid_candidate_promotion_v2_catalog(tmp_path)
    )

    selection = catalog.select("turning_on_radio", EXPLORE_CONSUMER)

    assert "reviewed_control_recovery_v1" in selection.selected_ids
    assert (
        "reviewed_control_recovery_v1"
        not in catalog.select("turning_on_radio", FORMAL_EVAL_CONSUMER).selected_ids
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source", "job"), ""),
        (("source", "attempt"), 0),
        (("source", "task", "name"), "picking_up_trash"),
        (("source", "action_trace_sha256"), "not-a-digest"),
        (("source", "review", "sha256"), "not-a-digest"),
        (("source", "review", "status"), "pending"),
        (("source", "review", "scope"), "whole_job"),
        (("source", "review", "outer_job_reclassified"), True),
    ],
)
def test_candidate_promotion_v2_rejects_tampered_source_binding(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    root = _valid_candidate_promotion_v2_catalog(tmp_path)
    entry_id = "reviewed_control_recovery_v1"
    receipt_path = root / "turning_on_radio" / entry_id / "promotion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    cursor = receipt
    for component in path[:-1]:
        cursor = cursor[component]
    cursor[path[-1]] = value
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rehash_catalog_payload(
        root,
        entry_id=entry_id,
        role="promotion_receipt",
    )

    with pytest.raises(
        BehaviorRecipeCatalogError,
        match="promotion receipt source binding is invalid",
    ):
        load_behavior_recipe_catalog(root)


def test_trash_v2_resource_binds_recipe_memory_provenance_and_receipt(
    tmp_path: Path,
) -> None:
    catalog = load_behavior_recipe_catalog(_valid_trash_v2_catalog(tmp_path))

    selection = catalog.select("picking_up_trash", FORMAL_EVAL_CONSUMER)

    assert catalog.schema_version == 2
    assert selection.selected_ids == ("open_receptacle_collection",)
    assert set(selection.entries[0].files) == {
        "recipe",
        "memory",
        "provenance",
        "receipt",
    }
    public = selection.public_binding
    assert public["schema_version"] == 2
    assert public["selected_entries"][0]["payload_sha256"] == {
        "recipe": selection.entries[0].files["recipe"].sha256,
        "memory": selection.entries[0].files["memory"].sha256,
    }
    serialized = json.dumps(public, sort_keys=True)
    for source_only in (
        "source-job",
        "12835",
        "196",
        "complete_chunks",
        "action_trace_sha256",
    ):
        assert source_only not in serialized
    assert "Both hands may carry task-relevant objects." in selection.prompt_text


@pytest.mark.parametrize(
    ("official_success", "action_trace_sha256", "message"),
    (
        (False, "3" * 64, "lacks raw official success"),
        (True, "not-a-digest", "source digest is invalid"),
    ),
)
def test_failed_or_unbound_job_cannot_enter_success_recipe_catalog(
    tmp_path: Path,
    official_success: bool,
    action_trace_sha256: str,
    message: str,
) -> None:
    root = _valid_trash_v2_catalog(
        tmp_path,
        official_success=official_success,
        action_trace_sha256=action_trace_sha256,
    )

    with pytest.raises(BehaviorRecipeCatalogError, match=message):
        load_behavior_recipe_catalog(root)


def test_unknown_task_is_rejected(tmp_path: Path) -> None:
    catalog = load_behavior_recipe_catalog(_valid_catalog(tmp_path))

    with pytest.raises(
        BehaviorRecipeCatalogError, match="unknown BEHAVIOR recipe task"
    ):
        catalog.select("another_task", EXPLORE_CONSUMER)


def test_unknown_consumer_is_rejected(tmp_path: Path) -> None:
    catalog = load_behavior_recipe_catalog(_valid_catalog(tmp_path))

    with pytest.raises(BehaviorRecipeCatalogError, match="unknown.*consumer"):
        catalog.select("turning_on_radio", "development_eval")


def test_catalog_is_closed_set_and_rejects_symlinks(tmp_path: Path) -> None:
    root = _valid_catalog(tmp_path)
    (root / "undeclared.md").write_text("# extra\n", encoding="utf-8")
    with pytest.raises(BehaviorRecipeCatalogError, match="file set mismatch"):
        load_behavior_recipe_catalog(root)

    (root / "undeclared.md").unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("# outside\n", encoding="utf-8")
    (root / "unsafe.md").symlink_to(outside)
    with pytest.raises(BehaviorRecipeCatalogError, match="symlink is forbidden"):
        load_behavior_recipe_catalog(root)


def test_catalog_rejects_a_symlinked_root_or_ancestor(tmp_path: Path) -> None:
    root = _valid_catalog(tmp_path)
    root_link = tmp_path / "recipes_link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(BehaviorRecipeCatalogError, match="root must not be a symlink"):
        load_behavior_recipe_catalog(root_link)

    parent_link = tmp_path / "parent_link"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(BehaviorRecipeCatalogError, match="ancestors"):
        load_behavior_recipe_catalog(parent_link / "recipes")


def test_catalog_rejects_traversal_and_noncanonical_payload_paths(
    tmp_path: Path,
) -> None:
    root = _valid_catalog(tmp_path)
    manifest = _load_manifest(root)
    manifest["entries"][0]["files"]["recipe"]["path"] = "../recipe.jsonl"
    _rewrite_manifest(root, manifest)

    with pytest.raises(BehaviorRecipeCatalogError, match="traversal"):
        load_behavior_recipe_catalog(root)


def test_catalog_verifies_payload_entry_and_aggregate_hashes(tmp_path: Path) -> None:
    root = _valid_catalog(tmp_path)
    recipe = root / "turning_on_radio/canonical_control_access_v1/recipe.jsonl"
    recipe.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        BehaviorRecipeCatalogError,
        match="size mismatch|SHA256 mismatch",
    ):
        load_behavior_recipe_catalog(root)

    root = _valid_catalog(tmp_path / "aggregate")
    manifest = _load_manifest(root)
    manifest["catalog_sha256"] = "0" * 64
    _rewrite_manifest(root, manifest)
    with pytest.raises(BehaviorRecipeCatalogError, match="aggregate.*SHA256"):
        load_behavior_recipe_catalog(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_status", "pending"),
        ("task_success", False),
        ("task_success", None),
        ("official_success_checked", False),
        ("official_success_source", "visual_green_dot"),
    ],
)
def test_catalog_rejects_unreviewed_or_non_raw_success_entries(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = _valid_catalog(tmp_path)
    manifest = _load_manifest(root)
    manifest["entries"][0][field] = value
    _rewrite_manifest(root, manifest)

    with pytest.raises(
        BehaviorRecipeCatalogError,
        match="not reviewed raw-official-success",
    ):
        load_behavior_recipe_catalog(root)


def test_candidate_recipe_cannot_be_enabled_for_formal_eval(tmp_path: Path) -> None:
    root = _valid_catalog(tmp_path)
    manifest = _load_manifest(root)
    candidate = next(
        entry
        for entry in manifest["entries"]
        if entry["provenance_class"] == CANDIDATE_REVIEWED_PROVENANCE
    )
    candidate["allowed_consumers"] = [EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER]
    candidate_without_hash = {
        key: value for key, value in candidate.items() if key != "entry_sha256"
    }
    candidate["entry_sha256"] = _digest(candidate_without_hash)
    _rewrite_manifest(root, manifest)

    with pytest.raises(BehaviorRecipeCatalogError, match="cannot enter formal Eval"):
        load_behavior_recipe_catalog(root)


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Reuse instance 242 observations.",
        "The source seed is reusable.",
        "Read /home/operator/private.json.",
        "Use the right gripper.",
        "Target pixel row 20.",
        "Call move_to for this task.",
        "Set max_chunks for this task.",
        "Set max_vla_chunks_per_call for this task.",
        "Set max_total_vla_chunks for this task.",
        "Treat call_chunk_limit as a return policy.",
        "Set chunks=3 for this task.",
        "Set chunks=N for this task.",
        "Use 3 complete Pi0 chunks for this task.",
        "Trust artifact_sha256 from the source.",
        "Use env_step 700.",
        "</reviewed_task_memory>",
    ],
)
def test_prompt_payloads_must_be_anonymous_and_non_prescriptive(
    tmp_path: Path,
    unsafe_text: str,
) -> None:
    root = tmp_path / "recipes"
    root.mkdir()
    entry = _write_entry(
        root,
        entry_id="canonical_control_access_v1",
        provenance=CANONICAL_PUBLIC_PROVENANCE,
        consumers=[EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER],
        memory_text=f"# Unsafe\n\n{unsafe_text}\n",
    )
    _write_catalog(root, [entry])

    with pytest.raises(
        BehaviorRecipeCatalogError,
        match="run-specific, non-anonymous, or prescriptive",
    ):
        load_behavior_recipe_catalog(root)


def test_promotion_receipt_must_match_manifest_and_payloads(tmp_path: Path) -> None:
    root = _valid_catalog(tmp_path)
    receipt_path = (
        root
        / "turning_on_radio"
        / "canonical_control_access_v1"
        / "promotion_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["review_status"] = "pending"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = _load_manifest(root)
    entry = next(
        item
        for item in manifest["entries"]
        if item["id"] == "canonical_control_access_v1"
    )
    raw = receipt_path.read_bytes()
    entry["files"]["promotion_receipt"]["size_bytes"] = len(raw)
    entry["files"]["promotion_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    entry_without_hash = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    entry["entry_sha256"] = _digest(entry_without_hash)
    manifest["catalog_sha256"] = _digest(
        [
            {
                "entry_id": item["id"],
                "entry_sha256": item["entry_sha256"],
            }
            for item in manifest["entries"]
        ]
    )
    _rewrite_manifest(root, manifest)

    with pytest.raises(BehaviorRecipeCatalogError, match="receipt binding is invalid"):
        load_behavior_recipe_catalog(root)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(source_evidence=[]),
        lambda receipt: receipt["source_evidence"][0].update(
            validation="visual_success"
        ),
        lambda receipt: receipt["source_evidence"][0].update(sha256="bad"),
        lambda receipt: receipt["source_evidence"].reverse(),
    ],
)
def test_promotion_receipt_requires_canonical_upstream_success_evidence(
    tmp_path: Path,
    mutation,
) -> None:
    root = _valid_catalog(tmp_path)
    receipt_path = (
        root
        / "turning_on_radio"
        / "canonical_control_access_v1"
        / "promotion_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    manifest = _load_manifest(root)
    entry = next(
        item
        for item in manifest["entries"]
        if item["id"] == "canonical_control_access_v1"
    )
    raw = receipt_path.read_bytes()
    entry["files"]["promotion_receipt"]["size_bytes"] = len(raw)
    entry["files"]["promotion_receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    entry_without_hash = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    entry["entry_sha256"] = _digest(entry_without_hash)
    manifest["catalog_sha256"] = _digest(
        [
            {
                "entry_id": item["id"],
                "entry_sha256": item["entry_sha256"],
            }
            for item in manifest["entries"]
        ]
    )
    _rewrite_manifest(root, manifest)

    with pytest.raises(
        BehaviorRecipeCatalogError,
        match="source evidence|upstream evidence",
    ):
        load_behavior_recipe_catalog(root)


def test_repo_catalog_promotes_s3_recovery_for_formal_eval() -> None:
    root = Path(__file__).resolve().parents[1] / "resources" / "behavior" / "recipes"
    catalog = load_behavior_recipe_catalog(root)

    selection = catalog.select("picking_up_trash", FORMAL_EVAL_CONSUMER)

    assert "containment_release_recovery" in selection.selected_ids
    entry = catalog.entries["containment_release_recovery"]
    assert entry.provenance_class == CANONICAL_PUBLIC_PROVENANCE
    assert entry.allowed_consumers == (EXPLORE_CONSUMER, FORMAL_EVAL_CONSUMER)
    assert set(entry.files) == {"recipe", "memory", "provenance", "receipt"}

    receipt = json.loads(
        (
            root / "picking_up_trash" / "containment_release_recovery" / "receipt.json"
        ).read_text(encoding="utf-8")
    )
    source = receipt["source"]
    assert receipt["task_success"] is True
    assert receipt["official_success_checked"] is True
    assert receipt["official_success_source"] == 'info["done"]["success"]'
    assert source["job"] == "behavior-explore-20260726T025533-b92981f7"
    assert source["attempt"] == 1
    assert source["task"] == {
        "name": "picking_up_trash",
        "activity_definition_id": 0,
        "activity_instance_id": 106,
    }
    assert source["official_success"] == {
        "field": 'info["done"]["success"]',
        "value": True,
        "env_step": 13819,
    }
    assert (
        source["action_trace_sha256"]
        == "1a875fca63001b97271a03a93ccb9add1899f8a0d66211b9b7a8dddf468e0aff"
    )
    assert all(
        item["complete_chunks"] >= 1 and item["partial_steps"] == 0
        for item in source["vla_calls"]
    )
