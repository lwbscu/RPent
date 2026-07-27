from __future__ import annotations

import json

from robots.behavior.env_server import BehaviorEnvFacade
from robots.behavior.terminal_success import validate_terminal_success_receipt


def _success_facade(tmp_path):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._run_nonce = "run"
    facade._attempt_nonce = "attempt"
    facade._attempt_index = 1
    facade._env_steps = 4054
    facade._official_success_latched = False
    facade._official_success_receipt = None
    facade._official_success_receipt_path = tmp_path / "official_success_receipt.json"
    facade._motion_frozen = False
    facade._controller_state = "vla"
    facade._action_source = "pi0_vla"
    facade._vla_actions_enabled = True
    facade._done = False
    facade._video_error = None
    facade._video_path = tmp_path / "episode.mp4"
    facade._video_sealed = False
    facade._finalize_video_segment = lambda: None
    return facade


def test_first_raw_success_is_latched_without_extra_env_step(tmp_path):
    facade = _success_facade(tmp_path)
    receipt = facade._latch_official_success({"done": {"success": True}})
    assert facade._env_steps == 4054
    assert facade._official_success_latched is True
    assert facade._motion_frozen is True
    assert receipt["raw_done"]["success"] is True
    assert json.loads(facade._official_success_receipt_path.read_text()) == receipt


def test_later_false_info_cannot_revoke_first_raw_success(tmp_path):
    facade = _success_facade(tmp_path)
    first = facade._latch_official_success({"done": {"success": True}})
    later = facade._latch_official_success({"done": {"success": False}})
    assert later == first
    assert facade._official_success_latched is True


def test_raw_success_receipt_validates_without_hold_green_or_video(tmp_path):
    facade = _success_facade(tmp_path)
    receipt = facade._latch_official_success({"done": {"success": True}})
    validation = validate_terminal_success_receipt(
        tool_name="press",
        step=3,
        result={
            "task_success": True,
            "official_success_receipt": receipt,
        },
        output_dir=tmp_path,
    )
    assert validation.valid is True
    assert validation.terminal_image_path is None


def test_receipt_digest_and_raw_source_are_fail_closed(tmp_path):
    facade = _success_facade(tmp_path)
    receipt = dict(facade._latch_official_success({"done": {"success": True}}))
    receipt["env_step"] += 1
    invalid = validate_terminal_success_receipt(
        tool_name="press",
        step=1,
        result={"task_success": True, "official_success_receipt": receipt},
        output_dir=tmp_path,
    )
    assert invalid.valid is False
    assert "digest" in str(invalid.reason)


def test_non_vla_primitive_result_carries_nonce_bound_raw_receipt(tmp_path):
    facade = _success_facade(tmp_path)
    facade._attempt_index = 2
    facade._latch_official_success({"done": {"success": True}})
    facade._sanitized_capability_summary = lambda: {
        "attachments": {
            "available": True,
            "count": 1,
            "by_hand": {
                "left": {"attached": True},
                "right": {"attached": False},
            },
            "conflict": False,
        },
        "gripper_state": {"left": "closed", "right": "open"},
    }

    result = facade._planner_public_result(
        {
            "primitive_success": True,
            "task_success": True,
            "stop_reason": "pressed",
        }
    )

    assert result["task_success"] is True
    assert result["run_nonce"] == "run"
    assert result["attempt_nonce"] == "attempt"
    assert result["attempt_index"] == 2
    assert result["official_success_receipt"]["raw_done"]["success"] is True
    validation = validate_terminal_success_receipt(
        tool_name="press",
        step=1,
        result=result,
        output_dir=tmp_path,
    )
    assert validation.valid is True
