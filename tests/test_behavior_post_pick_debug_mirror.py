import hashlib
import json

import pytest

from robots.behavior.post_pick_debug_mirror import (
    DEBUG_MIRROR_CHECKPOINT_NAME,
    DEBUG_MIRROR_MANIFEST_NAME,
    DEBUG_MIRROR_SCENE_NAME,
    build_debug_mirror_manifest,
    validate_debug_mirror_bundle,
    write_debug_mirror_manifest,
)


def _meta():
    return {
        "suite": "behavior_2025_challenge",
        "task": 0,
        "task_name": "turning_on_radio",
        "activity_definition_id": 0,
        "activity_instance_id": 211,
        "scene_model": "house_double_floor_lower",
        "seed": 211,
        "max_episode_steps": 24756,
    }


def _bundle(tmp_path):
    scene = tmp_path / DEBUG_MIRROR_SCENE_NAME
    checkpoint = tmp_path / DEBUG_MIRROR_CHECKPOINT_NAME
    scene.write_text('{"state":{"ag_obj_constraint_params":{}}}\n', encoding="utf-8")
    checkpoint.write_text(
        json.dumps(
            {
                "kind": "robot_motion_checkpoint",
                "not_simulator_restore": True,
                "checkpoint_name": "state_checkpoint_1",
                "held_hand": "right",
                "press_hand": "left",
                "object_name": "radio_89",
            }
        ),
        encoding="utf-8",
    )
    manifest = build_debug_mirror_manifest(
        scene_path=scene,
        checkpoint_path=checkpoint,
        meta=_meta(),
        held_hand="right",
        press_hand="left",
        object_name="radio_89",
        object_prim_path="/World/scene_0/radio_89",
        assisted_grasp_prim_path="/World/scene_0/radio_89/root_link",
        source_env_step=1464,
        initial_radio_position=[3.0, 4.0, 0.5],
        gripper_latches={"left": 1.0, "right": -1.0},
        controller_layout=[
            ["base", 3],
            ["trunk", 4],
            ["arm_left", 7],
            ["gripper_left", 1],
            ["arm_right", 7],
            ["gripper_right", 1],
        ],
        base_motor_type="position",
        base_controller_signature={"class": "BaseController", "command_dim": 3},
        official_task_success=False,
    )
    manifest_path = write_debug_mirror_manifest(tmp_path, manifest)
    assert manifest_path.name == DEBUG_MIRROR_MANIFEST_NAME
    return scene, checkpoint, manifest


def test_post_pick_debug_mirror_bundle_round_trip(tmp_path):
    scene, checkpoint, expected = _bundle(tmp_path)

    manifest, resolved_scene, resolved_checkpoint = validate_debug_mirror_bundle(
        tmp_path, meta=_meta()
    )

    assert manifest == expected
    assert resolved_scene == scene
    assert resolved_checkpoint == checkpoint
    assert manifest["debug_only"] is True
    assert manifest["not_robot_motion_checkpoint"] is True
    assert manifest["scene"]["serialization"].endswith("state_serialized=False)")
    assert manifest["usable_post_pick_saved"] is True
    assert manifest["save_policy"] == "debug_save_physics_warnings_non_blocking"
    assert manifest["restore_policy"]["held_gripper_closed_during_first_action"] is True


def test_post_pick_debug_mirror_rejects_tampered_scene(tmp_path):
    scene, _checkpoint, _manifest = _bundle(tmp_path)
    scene.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="scene (sha256|bytes) mismatch"):
        validate_debug_mirror_bundle(scene, meta=_meta())


def test_post_pick_debug_mirror_rejects_wrong_task_binding(tmp_path):
    scene, _checkpoint, _manifest = _bundle(tmp_path)
    changed = _meta()
    changed["activity_instance_id"] = 212

    with pytest.raises(RuntimeError, match="task activity_instance_id mismatch"):
        validate_debug_mirror_bundle(scene, meta=changed)


def test_post_pick_debug_mirror_rejects_checkpoint_role_split(tmp_path):
    scene, checkpoint, _manifest = _bundle(tmp_path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["held_hand"], payload["press_hand"] = "left", "right"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = tmp_path / DEBUG_MIRROR_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    manifest["checkpoint"]["bytes"] = checkpoint.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint held_hand mismatch"):
        validate_debug_mirror_bundle(scene, meta=_meta())


def test_post_pick_debug_mirror_records_open_dynamic_held_latch_as_warning(tmp_path):
    scene = tmp_path / DEBUG_MIRROR_SCENE_NAME
    checkpoint = tmp_path / DEBUG_MIRROR_CHECKPOINT_NAME
    scene.write_text("{}\n", encoding="utf-8")
    checkpoint.write_text("{}\n", encoding="utf-8")

    payload = build_debug_mirror_manifest(
        scene_path=scene,
        checkpoint_path=checkpoint,
        meta=_meta(),
        held_hand="left",
        press_hand="right",
        object_name="radio_89",
        object_prim_path="/World/scene_0/radio_89",
        assisted_grasp_prim_path="/World/scene_0/radio_89/root_link",
        source_env_step=10,
        initial_radio_position=[0.0, 0.0, 0.0],
        gripper_latches={"left": 1.0, "right": -1.0},
        controller_layout=[],
        base_motor_type="position",
        base_controller_signature={"class": "BaseController", "command_dim": 3},
        official_task_success=False,
        strict_local_grasp_success=True,
    )

    assert payload["strict_local_grasp_success"] is False
    assert payload["usable_post_pick_saved"] is True
    assert payload["validation"]["held_latch_closed"] is False
    assert [item["code"] for item in payload["warnings"]] == ["held_latch_not_closed"]


def test_post_pick_debug_mirror_official_success_is_recorded_not_blocking(tmp_path):
    scene = tmp_path / DEBUG_MIRROR_SCENE_NAME
    checkpoint = tmp_path / DEBUG_MIRROR_CHECKPOINT_NAME
    scene.write_text("{}\n", encoding="utf-8")
    checkpoint.write_text("{}\n", encoding="utf-8")

    payload = build_debug_mirror_manifest(
        scene_path=scene,
        checkpoint_path=checkpoint,
        meta=_meta(),
        held_hand="right",
        press_hand="left",
        object_name="radio_89",
        object_prim_path="/World/scene_0/radio_89",
        assisted_grasp_prim_path="/World/scene_0/radio_89/root_link",
        source_env_step=10,
        initial_radio_position=[0.0, 0.0, 0.0],
        gripper_latches={"left": 1.0, "right": -1.0},
        controller_layout=[],
        base_motor_type="position",
        base_controller_signature={"class": "BaseController", "command_dim": 3},
        official_task_success=True,
    )

    assert payload["usable_post_pick_saved"] is True
    assert payload["source"]["official_task_success"] is True
    assert [item["code"] for item in payload["warnings"]] == [
        "official_task_already_successful"
    ]
