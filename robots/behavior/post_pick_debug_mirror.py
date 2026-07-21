"""Bindings for an attachment-faithful post-pick simulator debug mirror.

This artifact is deliberately separate from ``state_checkpoint_1.json``.  The
checkpoint remains a robot-motion target; the mirror is a trusted local scene
snapshot used only to shorten later pre-press development cycles.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

DEBUG_MIRROR_SCHEMA_VERSION = 1
DEBUG_MIRROR_KIND = "rpent_behavior_post_pick_debug_mirror"
DEBUG_SAVE_POLICY = "debug_save_physics_warnings_non_blocking"
DEBUG_MIRROR_SCENE_NAME = "debug_mirror_post_pick.scene.json"
DEBUG_MIRROR_MANIFEST_NAME = "debug_mirror_post_pick.manifest.json"
DEBUG_MIRROR_CHECKPOINT_NAME = "state_checkpoint_1.json"
_TASK_FIELDS = (
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_binding(meta: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _TASK_FIELDS if field not in meta]
    if missing:
        raise RuntimeError("debug mirror metadata is missing: " + ", ".join(missing))
    return {
        "suite": str(meta["suite"]),
        "task": int(meta["task"]),
        "task_name": str(meta["task_name"]),
        "activity_definition_id": int(meta["activity_definition_id"]),
        "activity_instance_id": int(meta["activity_instance_id"]),
        "scene_model": str(meta["scene_model"]),
        "seed": int(meta["seed"]),
        "max_episode_steps": int(meta["max_episode_steps"]),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_debug_mirror_manifest(
    *,
    scene_path: str | Path,
    checkpoint_path: str | Path,
    meta: dict[str, Any],
    held_hand: str,
    press_hand: str,
    object_name: str,
    object_prim_path: str,
    assisted_grasp_prim_path: str,
    source_env_step: int,
    initial_radio_position: list[float],
    gripper_latches: dict[str, float],
    controller_layout: list[list[Any]],
    base_motor_type: str,
    base_controller_signature: dict[str, Any],
    official_task_success: bool,
    strict_local_grasp_success: bool = False,
    validation: dict[str, Any] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scene_path = Path(scene_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if scene_path.name != DEBUG_MIRROR_SCENE_NAME or not scene_path.is_file():
        raise RuntimeError("debug mirror scene path is missing or has the wrong name")
    if (
        checkpoint_path.name != DEBUG_MIRROR_CHECKPOINT_NAME
        or not checkpoint_path.is_file()
    ):
        raise RuntimeError(
            "debug mirror checkpoint copy is missing or has the wrong name"
        )
    if held_hand not in {"left", "right"} or press_hand not in {"left", "right"}:
        raise RuntimeError("debug mirror roles must be physical hands")
    if held_hand == press_hand:
        raise RuntimeError("debug mirror held and press hands must differ")
    if not object_prim_path or not assisted_grasp_prim_path:
        raise RuntimeError("debug mirror object/grasp prim binding is required")
    if int(source_env_step) < 0:
        raise RuntimeError("debug mirror source env step must be non-negative")
    if len(initial_radio_position) != 3 or any(
        not math.isfinite(float(value)) for value in initial_radio_position
    ):
        raise RuntimeError("debug mirror initial radio position is invalid")
    if not isinstance(base_controller_signature, dict):
        raise RuntimeError("debug mirror base controller signature is required")
    diagnostics = dict(validation or {})
    diagnostic_warnings = list(warnings or [])
    held_latch_closed = bool(gripper_latches.get(held_hand) == -1.0)
    diagnostics.setdefault("held_latch_closed", held_latch_closed)
    if not held_latch_closed and not any(
        item.get("code") == "held_latch_not_closed"
        for item in diagnostic_warnings
        if isinstance(item, dict)
    ):
        diagnostic_warnings.append(
            {
                "code": "held_latch_not_closed",
                "message": "source held-hand latch is not exactly close",
            }
        )
    if official_task_success and not any(
        item.get("code") == "official_task_already_successful"
        for item in diagnostic_warnings
        if isinstance(item, dict)
    ):
        diagnostic_warnings.append(
            {
                "code": "official_task_already_successful",
                "message": "official success is recorded but does not block debug save",
            }
        )
    return {
        "schema_version": DEBUG_MIRROR_SCHEMA_VERSION,
        "kind": DEBUG_MIRROR_KIND,
        "debug_only": True,
        "not_robot_motion_checkpoint": True,
        "not_official_episode_resume": True,
        "strict_local_grasp_success": bool(
            strict_local_grasp_success and held_latch_closed
        ),
        "usable_post_pick_saved": True,
        "save_policy": DEBUG_SAVE_POLICY,
        "warnings": diagnostic_warnings,
        "validation": diagnostics,
        "scene": {
            "file": scene_path.name,
            "sha256": sha256_file(scene_path),
            "bytes": scene_path.stat().st_size,
            "serialization": "omnigibson.sim.save(scene_json; state_serialized=False)",
        },
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "kind": "robot_motion_checkpoint",
        },
        "task": _task_binding(meta),
        "roles": {
            "held_hand": held_hand,
            "press_hand": press_hand,
            "object_name": str(object_name),
            "object_prim_path": str(object_prim_path),
            "assisted_grasp_prim_path": str(assisted_grasp_prim_path),
        },
        "source": {
            "env_step": int(source_env_step),
            "initial_radio_position": [
                float(value) for value in initial_radio_position
            ],
            "official_task_success": bool(official_task_success),
        },
        "controller": {
            "base_mode": str(base_motor_type or "unknown"),
            "base_motor_type": str(base_motor_type),
            "base_controller_signature": dict(base_controller_signature),
            "layout": controller_layout,
            "gripper_latches": {
                "left": float(gripper_latches["left"]),
                "right": float(gripper_latches["right"]),
            },
        },
        "restore_policy": {
            "held_gripper_closed_during_first_action": True,
            "vla_actions_disabled_before_post_pick_tools": True,
        },
    }


def write_debug_mirror_manifest(directory: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(directory).expanduser().resolve() / DEBUG_MIRROR_MANIFEST_NAME
    _atomic_json(path, payload)
    return path


def validate_debug_mirror_bundle(
    scene_path: str | Path, *, meta: dict[str, Any]
) -> tuple[dict[str, Any], Path, Path]:
    scene_path = Path(scene_path).expanduser().resolve()
    if scene_path.is_dir():
        scene_path = scene_path / DEBUG_MIRROR_SCENE_NAME
    if scene_path.name != DEBUG_MIRROR_SCENE_NAME or not scene_path.is_file():
        raise RuntimeError(f"post-pick debug mirror scene is missing: {scene_path}")
    directory = scene_path.parent
    manifest_path = directory / DEBUG_MIRROR_MANIFEST_NAME
    checkpoint_path = directory / DEBUG_MIRROR_CHECKPOINT_NAME
    if not manifest_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("post-pick debug mirror bundle is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DEBUG_MIRROR_SCHEMA_VERSION:
        raise RuntimeError("unsupported post-pick debug mirror schema")
    if manifest.get("kind") != DEBUG_MIRROR_KIND:
        raise RuntimeError("post-pick debug mirror kind mismatch")
    if manifest.get("debug_only") is not True:
        raise RuntimeError("post-pick debug mirror is not marked debug-only")
    if manifest.get("not_robot_motion_checkpoint") is not True:
        raise RuntimeError("post-pick debug mirror checkpoint semantics are ambiguous")
    if manifest.get("not_official_episode_resume") is not True:
        raise RuntimeError("post-pick debug mirror lacks non-official provenance")
    if manifest.get("usable_post_pick_saved") is not True:
        raise RuntimeError("post-pick debug mirror save status is invalid")
    if manifest.get("save_policy") != DEBUG_SAVE_POLICY:
        raise RuntimeError("post-pick debug mirror save policy mismatch")
    if not isinstance(manifest.get("strict_local_grasp_success"), bool):
        raise RuntimeError("post-pick debug mirror strict grasp status is invalid")
    if not isinstance(manifest.get("warnings"), list) or not isinstance(
        manifest.get("validation"), dict
    ):
        raise RuntimeError("post-pick debug mirror diagnostics are invalid")
    for payload, path, label in (
        (manifest.get("scene"), scene_path, "scene"),
        (manifest.get("checkpoint"), checkpoint_path, "checkpoint"),
    ):
        if not isinstance(payload, dict):
            raise RuntimeError(f"post-pick debug mirror lacks {label} binding")
        expected = {
            "file": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise RuntimeError(f"post-pick debug mirror {label} {field} mismatch")
    task = manifest.get("task")
    expected_task = _task_binding(meta)
    if not isinstance(task, dict):
        raise RuntimeError("post-pick debug mirror lacks task binding")
    for field, value in expected_task.items():
        if task.get(field) != value:
            raise RuntimeError(f"post-pick debug mirror task {field} mismatch")
    roles = manifest.get("roles")
    controller = manifest.get("controller")
    if not isinstance(roles, dict) or not isinstance(controller, dict):
        raise RuntimeError("post-pick debug mirror lacks role/controller binding")
    held, press = roles.get("held_hand"), roles.get("press_hand")
    if held not in {"left", "right"} or press not in {"left", "right"} or held == press:
        raise RuntimeError("post-pick debug mirror role binding is invalid")
    if (
        not isinstance(roles.get("object_prim_path"), str)
        or not roles.get("object_prim_path")
        or not isinstance(roles.get("assisted_grasp_prim_path"), str)
        or not roles.get("assisted_grasp_prim_path")
    ):
        raise RuntimeError("post-pick debug mirror prim binding is invalid")
    latches = controller.get("gripper_latches")
    if (
        not isinstance(latches, dict)
        or set(latches) != {"left", "right"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -1.1 <= float(value) <= 1.1
            for value in latches.values()
        )
    ):
        raise RuntimeError("post-pick debug mirror gripper latches are invalid")
    if not isinstance(controller.get("base_mode"), str):
        raise RuntimeError("post-pick debug mirror base mode is invalid")
    if not isinstance(controller.get("base_controller_signature"), dict):
        raise RuntimeError("post-pick debug mirror controller signature is missing")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("post-pick debug mirror lacks source binding")
    source_step = source.get("env_step")
    initial_radio = source.get("initial_radio_position")
    if (
        isinstance(source_step, bool)
        or not isinstance(source_step, int)
        or source_step < 0
        or not isinstance(initial_radio, list)
        or len(initial_radio) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in initial_radio
        )
        or not isinstance(source.get("official_task_success"), bool)
    ):
        raise RuntimeError("post-pick debug mirror source state is invalid")
    restore_policy = manifest.get("restore_policy")
    if not isinstance(restore_policy, dict) or restore_policy != {
        "held_gripper_closed_during_first_action": True,
        "vla_actions_disabled_before_post_pick_tools": True,
    }:
        raise RuntimeError("post-pick debug mirror restore policy is invalid")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("post-pick debug mirror checkpoint JSON is invalid") from exc
    expected_checkpoint = {
        "kind": "robot_motion_checkpoint",
        "not_simulator_restore": True,
        "checkpoint_name": "state_checkpoint_1",
        "held_hand": held,
        "press_hand": press,
        "object_name": roles.get("object_name"),
    }
    if not isinstance(checkpoint, dict):
        raise RuntimeError("post-pick debug mirror checkpoint payload is invalid")
    for field, value in expected_checkpoint.items():
        if checkpoint.get(field) != value:
            raise RuntimeError(f"post-pick debug mirror checkpoint {field} mismatch")
    return manifest, scene_path, checkpoint_path


__all__ = [
    "DEBUG_SAVE_POLICY",
    "DEBUG_MIRROR_CHECKPOINT_NAME",
    "DEBUG_MIRROR_KIND",
    "DEBUG_MIRROR_MANIFEST_NAME",
    "DEBUG_MIRROR_SCENE_NAME",
    "build_debug_mirror_manifest",
    "sha256_file",
    "validate_debug_mirror_bundle",
    "write_debug_mirror_manifest",
]
