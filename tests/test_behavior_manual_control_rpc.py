from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import (
    _CONTROLLER_PLANNER,
    _CONTROLLER_VLA,
    BehaviorEnvFacade,
)
from robots.behavior.schemas import (
    BASE_ROTATION_STEP_RAD,
    BASE_TRANSLATION_STEP_M,
    EEF_TRANSLATION_STEP_M,
    TORSO_VERTICAL_STEP_M,
    WRIST_ROTATION_STEP_RAD,
    validate_dashboard_manual_command,
)
from robots.behavior.task_specs import TURNING_ON_RADIO_TASK_SPEC
from robots.behavior.tools import BehaviorPrimitives

_PNG = b"\x89PNG\r\n\x1a\n"
_RUN_NONCE = "1" * 32
_ATTEMPT_NONCE = "2" * 32
_ATTEMPT_INDEX = 3


class _Rpc:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, tuple(args), dict(kwargs or {}), timeout_s))
        return self.response

    def close(self):
        return None


def _manual_response():
    return {
        "primitive_success": True,
        "task_success": False,
        "stop_reason": "completed",
        "_frames_bytes": {
            "head": _PNG + b"h",
            "left_wrist": _PNG + b"l",
            "right_wrist": _PNG + b"r",
        },
        "capture_group_id": "capture:7:test",
        "simulator_step": 7,
    }


def _success_receipt():
    receipt = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "run_nonce": _RUN_NONCE,
        "attempt_nonce": _ATTEMPT_NONCE,
        "attempt_index": _ATTEMPT_INDEX,
        "env_step": 7,
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
    return receipt


def _resign(receipt):
    material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _client(response=None):
    client = BehaviorEnvClient.__new__(BehaviorEnvClient)
    client._client = _Rpc(_manual_response() if response is None else response)
    client.episode_done = False
    client.total_env_steps = 0
    client.vla_endpoint = None
    client._official_success_latched = False
    client._official_success_receipt = None
    client._expected_run_nonce = _RUN_NONCE
    client._attempt_identity = (
        _RUN_NONCE,
        _ATTEMPT_NONCE,
        _ATTEMPT_INDEX,
    )
    return client


def test_dashboard_steps_are_server_owned_constants():
    assert BASE_TRANSLATION_STEP_M == pytest.approx(0.05)
    assert BASE_ROTATION_STEP_RAD == pytest.approx(0.08726646259971647)
    assert EEF_TRANSLATION_STEP_M == pytest.approx(0.03)
    assert TORSO_VERTICAL_STEP_M == pytest.approx(0.03)
    assert WRIST_ROTATION_STEP_RAD == pytest.approx(0.08726646259971647)


@pytest.mark.parametrize(
    ("target", "action"),
    [
        ("chassis", "forward"),
        ("chassis", "turn_right"),
        ("chassis", "up"),
        ("left_arm", "rotate_left"),
        ("right_arm", "close"),
        ("right_arm", "observe"),
    ],
)
def test_manual_command_schema_accepts_only_semantic_fields(target, action):
    assert validate_dashboard_manual_command(
        target=target, action=action, camera="head"
    ) == {"target": target, "action": action, "camera": "head"}


@pytest.mark.parametrize("action", ["rotate_left", "rotate_right", "open", "close"])
def test_chassis_rejects_arm_only_actions(action):
    with pytest.raises(ValueError, match="arm control only"):
        validate_dashboard_manual_command(
            target="chassis", action=action, camera="head"
        )


def test_env_client_sends_one_semantic_only_rpc_and_validates_capture():
    client = _client()

    result = client.dashboard_manual_command(
        target="left_arm",
        action="forward",
        camera="left_wrist",
    )

    assert result["capture_group_id"] == "capture:7:test"
    assert client._client.calls == [
        (
            "env.dashboard_manual_command",
            (),
            {
                "target": "left_arm",
                "action": "forward",
                "camera": "left_wrist",
            },
            360.0,
        )
    ]


def test_env_client_rejects_partial_capture_group():
    response = _manual_response()
    del response["_frames_bytes"]["right_wrist"]
    client = _client(response)

    with pytest.raises(RuntimeError, match="atomic three-camera"):
        client.dashboard_manual_command(
            target="chassis", action="forward", camera="head"
        )


def test_env_client_preserves_raw_success_when_same_rpc_capture_failed():
    response = {
        "primitive_success": True,
        "task_success": True,
        "stop_reason": "official_task_success",
        "official_success_receipt": _success_receipt(),
        "capture_error": "RuntimeError: camera refresh failed",
    }
    client = _client(response)

    result = client.dashboard_manual_command(
        target="chassis", action="forward", camera="head"
    )

    assert result["task_success"] is True
    assert result["capture_error"]
    assert client._official_success_latched is True
    assert len(client._client.calls) == 1


def test_env_client_fake_success_receipt_hash_does_not_latch():
    response = _manual_response()
    response.update(
        {
            "task_success": True,
            "official_success_receipt": {
                **_success_receipt(),
                "receipt_sha256": "a" * 64,
            },
        }
    )
    client = _client(response)

    returned = client.dashboard_manual_command(
        target="chassis", action="forward", camera="head"
    )

    assert returned["task_success"] is True
    assert client._official_success_latched is False
    assert client._official_success_receipt is None
    assert client.episode_done is False


def test_env_client_valid_but_wrong_attempt_receipt_does_not_latch():
    wrong_receipt = _success_receipt()
    wrong_receipt["attempt_nonce"] = "3" * 32
    _resign(wrong_receipt)
    response = _manual_response()
    response.update(
        {
            "task_success": True,
            "official_success_receipt": wrong_receipt,
        }
    )
    client = _client(response)

    client.dashboard_manual_command(
        target="right_arm", action="close", camera="right_wrist"
    )

    assert client._official_success_latched is False
    assert client._official_success_receipt is None


def test_behavior_primitives_manual_method_is_not_a_public_tool():
    response = _manual_response()
    env = SimpleNamespace(
        dashboard_manual_command=lambda **kwargs: {
            **response,
            "received": kwargs,
        }
    )
    primitives = BehaviorPrimitives.__new__(BehaviorPrimitives)
    primitives.env = env

    result = primitives.dashboard_manual_command(
        target="right_arm",
        action="open",
        camera="right_wrist",
    )

    assert result["received"] == {
        "target": "right_arm",
        "action": "open",
        "camera": "right_wrist",
    }


def test_behavior_primitives_preserve_success_without_partial_frame_publish():
    result = {
        "primitive_success": True,
        "task_success": True,
        "stop_reason": "official_task_success",
        "official_success_receipt": _success_receipt(),
        "capture_error": "RuntimeError: incomplete three-camera capture",
    }
    primitives = BehaviorPrimitives.__new__(BehaviorPrimitives)
    primitives.env = SimpleNamespace(
        dashboard_manual_command=lambda **kwargs: dict(result)
    )

    returned = primitives.dashboard_manual_command(
        target="right_arm", action="close", camera="right_wrist"
    )

    assert returned == result


class _Planner:
    def __init__(self):
        self.calls = []
        self.capability_calls = 0

    def dashboard_control_capabilities(self):
        self.capability_calls += 1
        return {
            "base": True,
            "eef": {"left": True, "right": True},
            "torso": False,
            "wrist": {"left": False, "right": False},
            "gripper": {"left": True, "right": True},
        }

    def jog_base(self, action):
        self.calls.append(("jog_base", action))
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "completed",
        }

    def set_gripper(self, hand, opening):
        self.calls.append(("set_gripper", hand, opening))
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "completed",
        }

    def observe(self, camera):
        self.calls.append(("observe", camera))
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "observed",
            "_image_bytes": b"legacy-selected-image",
            "_depth_image_bytes": b"legacy-selected-depth",
        }


def _facade():
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._planner = _Planner()
    facade._meta = {"suite": "behavior_2025_challenge"}
    facade._task_spec = TURNING_ON_RADIO_TASK_SPEC
    facade._task_identity = (
        TURNING_ON_RADIO_TASK_SPEC.task_name,
        TURNING_ON_RADIO_TASK_SPEC.activity_definition_id,
        196,
    )
    facade._env = SimpleNamespace(omnigibson_env=object())
    robot = SimpleNamespace(
        name="R1Pro",
        reload_controllers=lambda config: None,
    )
    facade._robot = lambda: robot
    facade._controller_mode = "hybrid"
    facade._controller_state = _CONTROLLER_VLA
    facade._base_controller_mode = "velocity"
    facade._reset_completed = True
    facade._official_success_latched = False
    facade._last_info = {}
    facade._done = False
    facade._motion_frozen = False
    facade._motion_in_flight = False
    facade._env_steps = 7

    def switch(target):
        assert target == _CONTROLLER_PLANNER
        facade._controller_state = target
        facade._base_controller_mode = "position"
        return {"changed": True}

    facade._switch_controller = switch
    facade._planner_public_result = lambda result: {
        **result,
        "task_success": False,
        "official_success_source": 'info["done"]["success"]',
    }
    facade._dashboard_capture_group = lambda: {
        "_frames_bytes": {
            "head": _PNG + b"h",
            "left_wrist": _PNG + b"l",
            "right_wrist": _PNG + b"r",
        },
        "frame_ids": {
            "head": "h7",
            "left_wrist": "l7",
            "right_wrist": "r7",
        },
        "capture_group_id": "capture:7:manual",
        "simulator_step": 7,
    }
    return facade


def test_capabilities_accept_formal_behavior_suite_and_composite_identity():
    facade = _facade()

    capabilities = facade.dashboard_control_capabilities()

    assert capabilities["simulation_identity"] == "behavior_omnigibson_r1pro"
    assert capabilities["motion_available"] is True
    assert capabilities["observe_available"] is True


def test_capabilities_fail_closed_for_bare_or_mismatched_task_identity():
    facade = _facade()
    facade._task_identity = (
        TURNING_ON_RADIO_TASK_SPEC.task_name,
        TURNING_ON_RADIO_TASK_SPEC.activity_definition_id,
        0,
    )

    capabilities = facade.dashboard_control_capabilities()

    assert capabilities["simulation_identity"] is None
    assert capabilities["motion_available"] is False
    assert capabilities["observe_available"] is False
    assert capabilities["unavailable_reason"] == "behavior_omnigibson_r1pro_unverified"


def test_env_facade_dispatches_fixed_base_jog_and_returns_capture_same_call():
    facade = _facade()

    result = facade.dashboard_manual_command(
        target="chassis", action="forward", camera="head"
    )

    assert facade._planner.calls == [("jog_base", "forward")]
    assert result["primitive"] == "navigate_to"
    assert result["primitive_detail"] == "relative jog"
    assert result["requested_step"] == {
        "frame": "base",
        "distance_m": 0.05,
        "direction": "forward",
    }
    assert set(result["_frames_bytes"]) == {
        "head",
        "left_wrist",
        "right_wrist",
    }
    assert result["capture_group_id"] == "capture:7:manual"
    assert result["simulator_step"] == 7


def test_env_facade_gripper_opening_is_derived_server_side():
    facade = _facade()

    result = facade.dashboard_manual_command(
        target="left_arm", action="close", camera="left_wrist"
    )

    assert facade._planner.calls == [("set_gripper", "left", 0.0)]
    assert result["requested_step"] == {"opening": 0.0}


def test_env_facade_observe_uses_selected_camera_and_explicit_frame_map():
    facade = _facade()

    result = facade.dashboard_manual_command(
        target="right_arm",
        action="observe",
        camera="right_wrist",
    )

    assert facade._planner.calls == [("observe", "right_wrist")]
    assert "_image_bytes" not in result
    assert "_depth_image_bytes" not in result
    assert set(result["_frames_bytes"]) == {
        "head",
        "left_wrist",
        "right_wrist",
    }


def test_env_facade_fails_closed_for_unverified_wrist_capability():
    facade = _facade()

    with pytest.raises(RuntimeError, match="unavailable or unverified"):
        facade.dashboard_manual_command(
            target="right_arm",
            action="rotate_right",
            camera="right_wrist",
        )
    assert facade._planner.calls == []


def test_env_facade_returns_raw_success_even_if_final_capture_fails():
    facade = _facade()
    receipt = _success_receipt()

    def successful_jog(action):
        facade._planner.calls.append(("jog_base", action))
        facade._official_success_latched = True
        facade._official_success_receipt = dict(receipt)
        return {
            "primitive_success": True,
            "task_success": True,
            "stop_reason": "official_task_success",
            "official_success_receipt": dict(receipt),
        }

    facade._planner.jog_base = successful_jog
    facade._planner_public_result = lambda result: dict(result)
    facade._dashboard_capture_group = lambda: (_ for _ in ()).throw(
        RuntimeError("camera refresh failed")
    )

    result = facade.dashboard_manual_command(
        target="chassis", action="forward", camera="head"
    )

    assert result["task_success"] is True
    assert result["official_success_receipt"] == receipt
    assert result["capture_error"] == "RuntimeError: camera refresh failed"
    assert "_frames_bytes" not in result
    assert facade._planner.capability_calls == 1
    assert result["control_capabilities"]["motion_available"] is False
    assert (
        result["control_capabilities"]["unavailable_reason"]
        == "official_success_latched"
    )
