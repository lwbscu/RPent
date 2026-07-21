import json
from pathlib import Path

import pytest

from robots.behavior.snapshot_manifest import (
    build_snapshot_manifest,
    snapshot_manifest_path,
    validate_snapshot_manifest,
    write_snapshot_manifest,
)


def _meta(tmp_path: Path) -> dict[str, object]:
    instance_dir = tmp_path / "instances"
    instance_dir.mkdir()
    (tmp_path / "house_double_floor_lower_task_turning_on_radio_0_0_template.json").write_text(
        '{"scene": "full"}\n', encoding="utf-8"
    )
    (instance_dir / "house_double_floor_lower_task_turning_on_radio_0_211_template-tro_state.json").write_text(
        '{"instance": 211}\n', encoding="utf-8"
    )
    return {
        "control_mode": "planner_tools",
        "suite": "behavior_2025_challenge",
        "task": 0,
        "task_name": "turning_on_radio",
        "activity_definition_id": 0,
        "activity_instance_id": 211,
        "activity_instance_dir": str(instance_dir),
        "scene_model": "house_double_floor_lower",
        "seed": 211,
        "max_episode_steps": 24756,
    }


def _write_valid(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    snapshot = tmp_path / "radio_near_state.pt"
    snapshot.write_bytes(b"serialized-state")
    meta = _meta(tmp_path)
    manifest = build_snapshot_manifest(
        snapshot,
        serialized_elements=17,
        serialized_dtype="torch.float32",
        serialized_shape=[17],
        serialized_finite=True,
        meta=meta,
        source={"commit": "abc", "worktree_dirty": False},
        omnigibson_version="3.7.2",
        invariants={"head_radio_visible": True},
    )
    write_snapshot_manifest(snapshot, manifest)
    return snapshot, meta


def test_snapshot_manifest_round_trip(tmp_path):
    snapshot, meta = _write_valid(tmp_path)

    loaded = validate_snapshot_manifest(
        snapshot,
        serialized_elements=17,
        serialized_dtype="torch.float32",
        serialized_shape=[17],
        serialized_finite=True,
        meta=meta,
    )

    assert loaded["state"]["sha256"]
    assert loaded["task"]["activity_instance_id"] == 211
    assert loaded["instance_artifact"]["path"].endswith(
        "_211_template-tro_state.json"
    )


def test_snapshot_manifest_rejects_missing_sidecar(tmp_path):
    snapshot = tmp_path / "radio_near_state.pt"
    snapshot.write_bytes(b"serialized-state")
    meta = _meta(tmp_path)

    with pytest.raises(RuntimeError, match="manifest is missing"):
        validate_snapshot_manifest(
            snapshot,
            serialized_elements=17,
            serialized_dtype="torch.float32",
            serialized_shape=[17],
            serialized_finite=True,
            meta=meta,
        )


@pytest.mark.parametrize("mutation", ["state", "task", "instance", "template"])
def test_snapshot_manifest_rejects_binding_mismatch(tmp_path, mutation):
    snapshot, meta = _write_valid(tmp_path)
    if mutation == "state":
        snapshot.write_bytes(b"different-state")
    elif mutation == "task":
        meta["seed"] = 212
    elif mutation == "instance":
        instance_path = next((tmp_path / "instances").glob("*.json"))
        instance_path.write_text('{"instance": "tampered"}\n', encoding="utf-8")
    else:
        template_path = tmp_path / (
            "house_double_floor_lower_task_turning_on_radio_0_0_template.json"
        )
        template_path.write_text('{"scene": "tampered"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="mismatch"):
        validate_snapshot_manifest(
            snapshot,
            serialized_elements=17,
            serialized_dtype="torch.float32",
            serialized_shape=[17],
            serialized_finite=True,
            meta=meta,
        )


def test_snapshot_manifest_rejects_untrusted_provenance(tmp_path):
    snapshot, meta = _write_valid(tmp_path)
    path = snapshot_manifest_path(snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["private_bootstrap_truth_used"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="bootstrap provenance"):
        validate_snapshot_manifest(
            snapshot,
            serialized_elements=17,
            serialized_dtype="torch.float32",
            serialized_shape=[17],
            serialized_finite=True,
            meta=meta,
        )


@pytest.mark.parametrize("target_mode", ["full_task_vla", "pi0_pick_vla"])
def test_planner_snapshot_rejects_cross_control_mode_restore(tmp_path, target_mode):
    snapshot, meta = _write_valid(tmp_path)
    meta["control_mode"] = target_mode

    with pytest.raises(RuntimeError, match="control_mode mismatch"):
        validate_snapshot_manifest(
            snapshot,
            serialized_elements=17,
            serialized_dtype="torch.float32",
            serialized_shape=[17],
            serialized_finite=True,
            meta=meta,
        )
