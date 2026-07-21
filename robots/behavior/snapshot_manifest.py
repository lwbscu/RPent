"""Strong binding for acceptance-only BEHAVIOR simulator snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

SNAPSHOT_MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_MANIFEST_KIND = "rpent_behavior_planner_restore"
TASK_BINDING_FIELDS = (
    "control_mode",
    "suite",
    "task",
    "task_name",
    "activity_definition_id",
    "activity_instance_id",
    "scene_model",
    "seed",
    "max_episode_steps",
)


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_manifest_path(snapshot_path: str | Path) -> Path:
    snapshot_path = Path(snapshot_path).expanduser().resolve()
    return Path(f"{snapshot_path}.manifest.json")


def resolve_activity_instance_path(meta: dict[str, Any]) -> Path:
    instance_dir = Path(str(meta["activity_instance_dir"])).expanduser().resolve()
    instance_id = int(meta["activity_instance_id"])
    matches = sorted(instance_dir.glob(f"*_{instance_id}_template-tro_state.json"))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one BEHAVIOR tro_state file for restore binding: "
            f"instance={instance_id} directory={instance_dir} matches={len(matches)}"
        )
    expected_name = (
        f"{meta['scene_model']}_task_{meta['task_name']}_"
        f"{int(meta['activity_definition_id'])}_{instance_id}_"
        "template-tro_state.json"
    )
    if matches[0].name != expected_name:
        raise RuntimeError(
            "BEHAVIOR restore tro_state basename mismatch: "
            f"expected={expected_name!r} current={matches[0].name!r}"
        )
    return matches[0].resolve()


def resolve_bootstrap_template_path(meta: dict[str, Any]) -> Path:
    instance_dir = Path(str(meta["activity_instance_dir"])).expanduser().resolve()
    template_path = instance_dir.parent / (
        f"{meta['scene_model']}_task_{meta['task_name']}_"
        f"{int(meta['activity_definition_id'])}_0_template.json"
    )
    if not template_path.is_file():
        raise RuntimeError(
            f"BEHAVIOR restore bootstrap template is missing: {template_path}"
        )
    return template_path.resolve()


def _task_binding(meta: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in TASK_BINDING_FIELDS if key not in meta]
    if missing:
        raise RuntimeError(
            "BEHAVIOR restore metadata is missing fields: " + ", ".join(missing)
        )
    return {
        "control_mode": str(meta["control_mode"]),
        "suite": str(meta["suite"]),
        "task": int(meta["task"]),
        "task_name": str(meta["task_name"]),
        "activity_definition_id": int(meta["activity_definition_id"]),
        "activity_instance_id": int(meta["activity_instance_id"]),
        "scene_model": str(meta["scene_model"]),
        "seed": int(meta["seed"]),
        "max_episode_steps": int(meta["max_episode_steps"]),
    }


def build_snapshot_manifest(
    snapshot_path: str | Path,
    *,
    serialized_elements: int,
    serialized_dtype: str,
    serialized_shape: list[int] | tuple[int, ...],
    serialized_finite: bool,
    meta: dict[str, Any],
    source: dict[str, Any],
    omnigibson_version: str | None,
    invariants: dict[str, Any],
) -> dict[str, Any]:
    """Create the complete sidecar payload for a trusted snapshot artifact."""

    snapshot_path = Path(snapshot_path).expanduser().resolve()
    if not snapshot_path.is_file():
        raise RuntimeError(f"simulator snapshot is missing: {snapshot_path}")
    if int(serialized_elements) <= 0:
        raise RuntimeError("serialized snapshot must contain at least one element")
    instance_path = resolve_activity_instance_path(meta)
    bootstrap_template_path = resolve_bootstrap_template_path(meta)
    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "kind": SNAPSHOT_MANIFEST_KIND,
        "state": {
            "file": snapshot_path.name,
            "sha256": sha256_file(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
            "serialization": "omnigibson.sim.dump_state(serialized=True)",
            "serialized_elements": int(serialized_elements),
            "tensor": {
                "dtype": str(serialized_dtype),
                "shape": [int(value) for value in serialized_shape],
                "finite": bool(serialized_finite),
            },
        },
        "task": _task_binding(meta),
        "instance_artifact": {
            "path": str(instance_path),
            "sha256": sha256_file(instance_path),
        },
        "bootstrap_template_artifact": {
            "path": str(bootstrap_template_path),
            "sha256": sha256_file(bootstrap_template_path),
        },
        "source": dict(source),
        "runtime": {"omnigibson_version": omnigibson_version},
        "acceptance_only": True,
        "private_bootstrap_truth_used": True,
        "invariants": dict(invariants),
    }


def write_snapshot_manifest(snapshot_path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically persist a snapshot sidecar next to the tensor file."""

    destination = snapshot_manifest_path(snapshot_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def validate_snapshot_manifest(
    snapshot_path: str | Path,
    *,
    serialized_elements: int,
    serialized_dtype: str,
    serialized_shape: list[int] | tuple[int, ...],
    serialized_finite: bool,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Validate state bytes and every current task/instance binding."""

    snapshot_path = Path(snapshot_path).expanduser().resolve()
    manifest_path = snapshot_manifest_path(snapshot_path)
    if not manifest_path.is_file():
        raise RuntimeError(f"simulator restore manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid simulator restore manifest: {exc}") from exc
    if manifest.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported simulator restore manifest schema")
    if manifest.get("kind") != SNAPSHOT_MANIFEST_KIND:
        raise RuntimeError("simulator restore manifest has the wrong kind")
    if manifest.get("acceptance_only") is not True:
        raise RuntimeError("simulator restore manifest is not acceptance-only")
    if manifest.get("private_bootstrap_truth_used") is not True:
        raise RuntimeError("simulator restore manifest lacks bootstrap provenance")

    state = manifest.get("state")
    if not isinstance(state, dict):
        raise RuntimeError("simulator restore manifest lacks state binding")
    expected_state = {
        "file": snapshot_path.name,
        "sha256": sha256_file(snapshot_path),
        "bytes": snapshot_path.stat().st_size,
        "serialized_elements": int(serialized_elements),
    }
    for field, expected in expected_state.items():
        if state.get(field) != expected:
            raise RuntimeError(
                f"simulator restore state {field} mismatch: "
                f"manifest={state.get(field)!r} current={expected!r}"
            )
    tensor = state.get("tensor")
    expected_tensor = {
        "dtype": str(serialized_dtype),
        "shape": [int(value) for value in serialized_shape],
        "finite": bool(serialized_finite),
    }
    if not isinstance(tensor, dict):
        raise RuntimeError("simulator restore manifest lacks tensor binding")
    for field, expected in expected_tensor.items():
        if tensor.get(field) != expected:
            raise RuntimeError(
                f"simulator restore tensor {field} mismatch: "
                f"manifest={tensor.get(field)!r} current={expected!r}"
            )
    if state.get("serialization") != "omnigibson.sim.dump_state(serialized=True)":
        raise RuntimeError("simulator restore serialization method mismatch")

    expected_task = _task_binding(meta)
    task = manifest.get("task")
    if not isinstance(task, dict):
        raise RuntimeError("simulator restore manifest lacks task binding")
    for field, expected in expected_task.items():
        if task.get(field) != expected:
            raise RuntimeError(
                f"simulator restore task {field} mismatch: "
                f"manifest={task.get(field)!r} current={expected!r}"
            )

    instance_path = resolve_activity_instance_path(meta)
    expected_instance = {
        "path": str(instance_path),
        "sha256": sha256_file(instance_path),
    }
    instance = manifest.get("instance_artifact")
    if not isinstance(instance, dict):
        raise RuntimeError("simulator restore manifest lacks instance binding")
    for field, expected in expected_instance.items():
        if instance.get(field) != expected:
            raise RuntimeError(
                f"simulator restore instance {field} mismatch: "
                f"manifest={instance.get(field)!r} current={expected!r}"
            )

    bootstrap_template_path = resolve_bootstrap_template_path(meta)
    expected_template = {
        "path": str(bootstrap_template_path),
        "sha256": sha256_file(bootstrap_template_path),
    }
    template = manifest.get("bootstrap_template_artifact")
    if not isinstance(template, dict):
        raise RuntimeError(
            "simulator restore manifest lacks bootstrap template binding"
        )
    for field, expected in expected_template.items():
        if template.get(field) != expected:
            raise RuntimeError(
                f"simulator restore bootstrap template {field} mismatch: "
                f"manifest={template.get(field)!r} current={expected!r}"
            )

    if not isinstance(manifest.get("source"), dict):
        raise RuntimeError("simulator restore manifest lacks source identity")
    if not isinstance(manifest.get("runtime"), dict):
        raise RuntimeError("simulator restore manifest lacks runtime identity")
    if not isinstance(manifest.get("invariants"), dict):
        raise RuntimeError("simulator restore manifest lacks invariants")
    return manifest


__all__ = [
    "SNAPSHOT_MANIFEST_KIND",
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "TASK_BINDING_FIELDS",
    "build_snapshot_manifest",
    "resolve_activity_instance_path",
    "resolve_bootstrap_template_path",
    "sha256_file",
    "snapshot_manifest_path",
    "validate_snapshot_manifest",
    "write_snapshot_manifest",
]
