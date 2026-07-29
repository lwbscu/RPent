from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
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
    BEHAVIOR_TOOL_NAMES,
    EEF_TRANSLATION_STEP_M,
    TORSO_VERTICAL_STEP_M,
    WRIST_ROTATION_STEP_RAD,
    validate_dashboard_manual_command,
    validate_dashboard_prepare_request,
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
    }


def _capture_response():
    return {
        "_frames_bytes": {
            "head": _PNG + b"h",
            "left_wrist": _PNG + b"l",
            "right_wrist": _PNG + b"r",
        },
        "frame_ids": {
            "head": "head:7:test",
            "left_wrist": "left_wrist:7:test",
            "right_wrist": "right_wrist:7:test",
        },
        "capture_group_id": "capture:7:test",
        "simulator_step": 7,
    }


def _prepared_response(
    *,
    plan_id: str = "plan-1",
    target: str = "chassis",
    action: str = "forward",
    predecessor_plan_id: str | None = None,
    background: bool = False,
):
    return {
        "schema_version": 1,
        "plan_id": plan_id,
        "target": target,
        "action": action,
        "predecessor_plan_id": predecessor_plan_id,
        "motion_kind": "base" if target == "chassis" else "eef",
        "status": "prepared",
        "predicted_terminal": {
            "joint_positions": [0.0] * 21,
            "base_xyyaw": [0.0, 0.0, 0.0],
            "eef_by_hand": {},
        },
        "planning_profile": "dashboard_jog",
        "planning_deadline_s": 12.0,
        "fast_solver_deadline_s": 4.0,
        "background": background,
        "deadline_enforcement": {
            "solver_timeout_enforced": True,
            "hard_wall_clock_enforced": not background,
            "soft_deadline_s": 12.0 if background else None,
        },
        "picklable": True,
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


def test_prepared_command_schema_is_internal_motion_only():
    assert validate_dashboard_prepare_request(
        target="left_arm",
        action="up",
        predecessor_plan_id="plan-0",
        background=True,
    ) == {
        "target": "left_arm",
        "action": "up",
        "predecessor_plan_id": "plan-0",
        "background": True,
    }
    with pytest.raises(ValueError, match="observe"):
        validate_dashboard_prepare_request(
            target="left_arm",
            action="observe",
        )
    with pytest.raises(ValueError, match="predecessor"):
        validate_dashboard_prepare_request(
            target="left_arm",
            action="up",
            background=True,
        )
    assert validate_dashboard_prepare_request(
        target="right_arm",
        action="rotate_left",
        planning_only_probe=True,
    ) == {
        "target": "right_arm",
        "action": "rotate_left",
        "predecessor_plan_id": None,
        "background": False,
        "planning_only_probe": True,
    }
    with pytest.raises(ValueError, match="limited"):
        validate_dashboard_prepare_request(
            target="right_arm",
            action="forward",
            planning_only_probe=True,
        )
    assert {
        "dashboard_prepare_manual_command",
        "dashboard_execute_prepared_command",
        "dashboard_discard_prepared_command",
        "dashboard_capture_views",
    }.isdisjoint(BEHAVIOR_TOOL_NAMES)


def test_env_client_background_prepare_uses_dedicated_rpc_contract():
    response = _prepared_response(
        plan_id="plan-2",
        predecessor_plan_id="plan-1",
        background=True,
    )
    client = _client(response)

    returned = client.dashboard_prepare_manual_command(
        target="chassis",
        action="forward",
        predecessor_plan_id="plan-1",
        background=True,
    )

    assert returned["plan_id"] == "plan-2"
    assert returned["deadline_enforcement"] == {
        "solver_timeout_enforced": True,
        "hard_wall_clock_enforced": False,
        "soft_deadline_s": 12.0,
    }
    assert client._client.calls == [
        (
            "env.dashboard_prepare_manual_command",
            (),
            {
                "target": "chassis",
                "action": "forward",
                "predecessor_plan_id": "plan-1",
                "background": True,
            },
            72.0,
        )
    ]


def test_env_client_planning_only_probe_uses_explicit_zero_action_contract():
    client = _client(
        {
            **_prepared_response(
                plan_id="probe-plan",
                target="left_arm",
                action="rotate_left",
            ),
            "planning_only_probe": True,
            "env_step_delta": 0,
            "zero_action_verified": True,
        }
    )

    returned = client.dashboard_prepare_manual_command(
        target="left_arm",
        action="rotate_left",
        planning_only_probe=True,
    )

    assert returned["zero_action_verified"] is True
    assert client._client.calls == [
        (
            "env.dashboard_prepare_manual_command",
            (),
            {
                "target": "left_arm",
                "action": "rotate_left",
                "predecessor_plan_id": None,
                "background": False,
                "planning_only_probe": True,
            },
            72.0,
        )
    ]


def test_env_client_legacy_motion_rpc_has_no_synchronous_capture():
    client = _client()

    result = client.dashboard_manual_command(
        target="left_arm",
        action="forward",
        camera="left_wrist",
    )

    assert result["primitive_success"] is True
    assert "_frames_bytes" not in result
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


def test_env_client_rejects_partial_independent_capture_group():
    response = _capture_response()
    del response["_frames_bytes"]["right_wrist"]
    client = _client(response)

    with pytest.raises(RuntimeError, match="atomic three-camera"):
        client.dashboard_capture_views(command_id="capture-1")


def test_env_client_accepts_complete_independent_capture_group():
    client = _client(_capture_response())

    result = client.dashboard_capture_views(command_id="capture-1")

    assert set(result["_frames_bytes"]) == {
        "head",
        "left_wrist",
        "right_wrist",
    }
    assert client._client.calls == [
        (
            "env.dashboard_capture_views",
            (),
            {"command_id": "capture-1"},
            120.0,
        )
    ]


def test_env_client_raw_success_execute_blocks_capture_before_transport():
    response = {
        "primitive_success": True,
        "task_success": True,
        "stop_reason": "official_task_success",
        "official_success_receipt": _success_receipt(),
    }
    client = _client(response)

    result = client.dashboard_execute_prepared_command(
        plan_id="plan-1",
        command_id="command-1",
    )

    assert result["task_success"] is True
    assert client._official_success_latched is True
    with pytest.raises(RuntimeError, match="terminal"):
        client.dashboard_capture_views(command_id="capture-1")
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


def test_behavior_primitives_route_internal_pipeline_without_public_tools():
    calls = []
    primitives = BehaviorPrimitives.__new__(BehaviorPrimitives)
    primitives.env = SimpleNamespace(
        dashboard_prepare_manual_command=lambda **kwargs: calls.append(
            ("prepare", kwargs)
        )
        or _prepared_response(target="left_arm", action="up"),
        dashboard_execute_prepared_command=lambda **kwargs: calls.append(
            ("execute", kwargs)
        )
        or _manual_response(),
        dashboard_discard_prepared_command=lambda **kwargs: calls.append(
            ("discard", kwargs)
        )
        or {"plan_id": kwargs["plan_id"], "discarded": True},
        dashboard_capture_views=lambda **kwargs: calls.append(
            ("capture", kwargs)
        )
        or _capture_response(),
    )

    prepared = primitives.dashboard_prepare_manual_command(
        target="left_arm",
        action="up",
    )
    executed = primitives.dashboard_execute_prepared_command(
        plan_id=prepared["plan_id"],
        command_id="command-1",
    )
    discarded = primitives.dashboard_discard_prepared_command(
        plan_id=prepared["plan_id"],
    )
    captured = primitives.dashboard_capture_views(command_id="capture-1")

    assert executed["primitive_success"] is True
    assert discarded["discarded"] is True
    assert captured["capture_group_id"] == "capture:7:test"
    assert calls == [
        (
            "prepare",
            {
                "target": "left_arm",
                "action": "up",
                "predecessor_plan_id": None,
                "background": False,
            },
        ),
        (
            "execute",
            {"plan_id": "plan-1", "command_id": "command-1"},
        ),
        ("discard", {"plan_id": "plan-1"}),
        ("capture", {"command_id": "capture-1"}),
    ]

    calls.clear()
    primitives.dashboard_prepare_manual_command(
        target="right_arm",
        action="rotate_right",
        planning_only_probe=True,
    )
    assert calls == [
        (
            "prepare",
            {
                "target": "right_arm",
                "action": "rotate_right",
                "predecessor_plan_id": None,
                "background": False,
                "planning_only_probe": True,
            },
        )
    ]


def test_behavior_primitives_preserve_success_without_partial_frame_publish():
    result = {
        "primitive_success": True,
        "task_success": True,
        "stop_reason": "official_task_success",
        "official_success_receipt": _success_receipt(),
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

    def prepare_dashboard_motion(
        self,
        target,
        action,
        predecessor_plan_id=None,
        background=False,
    ):
        self.calls.append(
            (
                "prepare_dashboard_motion",
                target,
                action,
                predecessor_plan_id,
                background,
            )
        )
        return _prepared_response(
            plan_id=(
                "plan-2"
                if predecessor_plan_id is not None
                else "plan-1"
            ),
            target=target,
            action=action,
            predecessor_plan_id=predecessor_plan_id,
            background=background,
        )

    def execute_dashboard_motion(self, plan_id, command_id):
        self.calls.append(("execute_dashboard_motion", plan_id, command_id))
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "completed",
        }

    def discard_dashboard_motion(self, plan_id):
        self.calls.append(("discard_dashboard_motion", plan_id))
        return {
            "schema_version": 1,
            "plan_id": plan_id,
            "discarded": True,
            "status": "discarded",
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
    facade._dashboard_planning_admitted = False
    facade._dashboard_execute_receipts = {}
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


def test_capabilities_enable_torso_after_zero_action_identity_warmup():
    facade = _facade()
    facade._planner.dashboard_control_capabilities = lambda: {
        "base": True,
        "eef": {"left": True, "right": True},
        "torso": True,
        "wrist": {"left": False, "right": False},
        "gripper": {"left": True, "right": True},
    }
    facade._planner_warmup_report = {
        "status": "complete",
        "identity_warmup": {
            "torso": {
                "ok": True,
                "active_dof_count": 3,
                "trajectory_waypoints": 1,
                "env_actions_sent": 0,
                "simulator_advanced": False,
            },
            "env_actions_sent": 0,
            "simulator_advanced": False,
        },
    }

    capabilities = facade.dashboard_control_capabilities()

    assert capabilities["torso_available"] is True


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


def test_env_facade_foreground_prepare_preserves_identity_gate_and_switches():
    facade = _facade()

    result = facade.dashboard_prepare_manual_command(
        target="chassis",
        action="forward",
    )

    assert result["plan_id"] == "plan-1"
    assert result["source"] == "dashboard_prepare"
    assert facade._controller_state == _CONTROLLER_PLANNER
    assert facade._dashboard_planning_admitted is True
    assert facade._planner.calls == [
        ("prepare_dashboard_motion", "chassis", "forward", None, False)
    ]


@pytest.mark.parametrize(
    "invalid_receipt",
    [
        {
            "schema_version": 1,
            "plan_id": "wrong-plan",
            "discarded": True,
            "status": "discarded",
        },
        {
            "schema_version": 1,
            "plan_id": "plan-1",
            "discarded": False,
            "status": "discarded",
        },
        {
            "schema_version": 1,
            "plan_id": "plan-1",
            "discarded": True,
            "status": "prepared",
        },
    ],
)
def test_env_facade_discard_requires_an_exact_terminal_receipt(invalid_receipt):
    facade = _facade()
    facade._planner.discard_dashboard_motion = lambda _plan_id: dict(
        invalid_receipt
    )

    with pytest.raises(RuntimeError, match="invalid discard receipt"):
        facade.dashboard_discard_prepared_command(plan_id="plan-1")


def test_env_facade_discard_preserves_the_verified_plan_identity():
    facade = _facade()

    receipt = facade.dashboard_discard_prepared_command(plan_id="plan-1")

    assert receipt["plan_id"] == "plan-1"
    assert receipt["discarded"] is True
    assert receipt["status"] == "discarded"
    assert receipt["source"] == "dashboard_discard"


def test_env_facade_capture_views_returns_one_complete_independent_group():
    facade = _facade()

    result = facade.dashboard_capture_views(command_id="capture-1")

    assert result["source"] == "dashboard_capture"
    assert result["command_id"] == "capture-1"
    assert set(result["_frames_bytes"]) == {
        "head",
        "left_wrist",
        "right_wrist",
    }


def test_env_facade_legacy_motion_has_no_capture_in_same_call():
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
    assert "_frames_bytes" not in result
    assert "capture_group_id" not in result


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


def test_env_facade_raw_success_sends_zero_post_success_capture_calls():
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
    capture_calls = []
    facade._dashboard_capture_group = lambda: capture_calls.append(True)

    result = facade.dashboard_manual_command(
        target="chassis", action="forward", camera="head"
    )

    assert result["task_success"] is True
    assert result["official_success_receipt"] == receipt
    assert "_frames_bytes" not in result
    assert capture_calls == []
    assert facade._planner.capability_calls == 1
    assert result["control_capabilities"]["motion_available"] is False
    assert (
        result["control_capabilities"]["unavailable_reason"]
        == "official_success_latched"
    )
