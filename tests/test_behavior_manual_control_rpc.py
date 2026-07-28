from __future__ import annotations

import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import (
    _CONTROLLER_PLANNER,
    _CONTROLLER_VLA,
    BehaviorEnvFacade,
    _MainThreadDispatcher,
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
from robots.behavior.toolkit import BehaviorToolkit
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


def test_toolkit_pipeline_requires_reservation_and_exact_command_permits():
    permit_calls = []

    class _Arbiter:
        owner = "manual"

        def snapshot(self):
            return {
                "owner": self.owner,
                "command_id": "command-1",
            }

        def require_manual_permit(self, command_id):
            permit_calls.append(command_id)
            if self.owner != "manual" or command_id not in {
                "command-1",
                "capture-1",
            }:
                raise RuntimeError("wrong permit")

    primitives = SimpleNamespace(
        dashboard_prepare_manual_command=lambda **kwargs: _prepared_response(
            target=kwargs["target"],
            action=kwargs["action"],
        ),
        dashboard_execute_prepared_command=lambda **kwargs: _manual_response(),
        dashboard_capture_views=lambda **kwargs: _capture_response(),
    )
    toolkit = BehaviorToolkit.__new__(BehaviorToolkit)
    toolkit._closed = False
    toolkit._command_arbiter = _Arbiter()
    toolkit._success_latch = SimpleNamespace(is_latched=lambda: False)
    toolkit._manual_intervention_latch = SimpleNamespace(set=lambda: None)
    toolkit._primitives = primitives

    prepared = toolkit.dashboard_prepare_manual_command(
        target="left_arm",
        action="up",
    )
    toolkit.dashboard_execute_prepared_command(
        plan_id=prepared["plan_id"],
        command_id="command-1",
    )
    toolkit.dashboard_capture_views(command_id="capture-1")

    assert permit_calls == ["command-1", "capture-1"]
    toolkit._command_arbiter.owner = "agent"
    with pytest.raises(RuntimeError, match="manual reservation"):
        toolkit.dashboard_prepare_manual_command(
            target="left_arm",
            action="up",
        )


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


def test_env_facade_background_prepare_records_soft_solver_deadline():
    facade = _facade()
    facade._controller_state = _CONTROLLER_PLANNER
    facade._base_controller_mode = "position"
    facade._dashboard_planning_admitted = True

    result = facade.dashboard_prepare_manual_command(
        target="chassis",
        action="forward",
        predecessor_plan_id="plan-1",
        background=True,
    )

    assert result["plan_id"] == "plan-2"
    assert result["deadline_enforcement"] == {
        "solver_timeout_enforced": True,
        "hard_wall_clock_enforced": False,
        "soft_deadline_s": 12.0,
        "background_deadline_kind": "soft_solver_deadline",
    }


def test_env_facade_background_prepare_fails_if_success_latches_during_plan():
    facade = _facade()
    facade._controller_state = _CONTROLLER_PLANNER
    facade._base_controller_mode = "position"
    facade._dashboard_planning_admitted = True
    original = facade._planner.prepare_dashboard_motion

    def latch_success(*args, **kwargs):
        result = original(*args, **kwargs)
        facade._official_success_latched = True
        return result

    facade._planner.prepare_dashboard_motion = latch_success

    with pytest.raises(RuntimeError, match="became terminal"):
        facade.dashboard_prepare_manual_command(
            target="chassis",
            action="forward",
            predecessor_plan_id="plan-1",
            background=True,
        )


def test_env_facade_execute_is_exactly_once_by_command_id_without_capture():
    facade = _facade()
    facade._controller_state = _CONTROLLER_PLANNER
    facade._base_controller_mode = "position"

    first = facade.dashboard_execute_prepared_command(
        plan_id="plan-1",
        command_id="command-1",
    )
    replay = facade.dashboard_execute_prepared_command(
        plan_id="plan-1",
        command_id="command-1",
    )

    assert first == replay
    assert first is not replay
    assert "_frames_bytes" not in first
    assert facade._planner.calls == [
        ("execute_dashboard_motion", "plan-1", "command-1")
    ]
    with pytest.raises(RuntimeError, match="different plan_id"):
        facade.dashboard_execute_prepared_command(
            plan_id="plan-2",
            command_id="command-1",
        )


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


def test_main_dispatcher_rejects_background_prepare_bypass():
    calls = []
    env = SimpleNamespace()
    dispatcher = _MainThreadDispatcher(
        env,
        SimpleNamespace(is_set=lambda: False),
    )
    dispatcher.submit = lambda method, args, kwargs: calls.append(
        ("fifo", method, kwargs)
    ) or {"status": "queued"}

    foreground = dispatcher.submit_dashboard_prepare(
        "env.dashboard_prepare_manual_command",
        (),
        {"target": "chassis", "action": "forward", "background": False},
    )

    assert foreground == {"status": "queued"}
    assert calls == [
        (
            "fifo",
            "env.dashboard_prepare_manual_command",
            {"target": "chassis", "action": "forward", "background": False},
        ),
    ]
    with pytest.raises(RuntimeError, match="cannot bypass"):
        dispatcher.submit_dashboard_prepare(
            "env.dashboard_prepare_manual_command",
            (),
            {
                "target": "chassis",
                "action": "forward",
                "predecessor_plan_id": "plan-1",
                "background": True,
            },
        )
    assert len(calls) == 1
    with pytest.raises(ValueError, match="unknown concurrent"):
        dispatcher.submit_dashboard_prepare(
            "env.dashboard_capture_views",
            (),
            {"background": True},
        )


def test_main_dispatcher_serializes_background_and_foreground_prepare():
    shutdown = threading.Event()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    active = 0
    max_active = 0
    order: list[str] = []
    state_lock = threading.Lock()

    def prepare(*, action, **_kwargs):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            order.append(f"start:{action}")
        if action == "forward":
            first_started.set()
            assert release_first.wait(2.0)
        else:
            second_started.set()
        with state_lock:
            order.append(f"end:{action}")
            active -= 1
        return {"action": action}

    env = SimpleNamespace(
        _assert_rpc_lifecycle=lambda _method: None,
        dashboard_prepare_manual_command=prepare,
    )
    dispatcher = _MainThreadDispatcher(env, shutdown)
    processor = threading.Thread(target=dispatcher.run, daemon=True)
    processor.start()
    results: dict[str, dict] = {}

    def submit(name: str, *, background: bool) -> None:
        results[name] = dispatcher.submit(
            "env.dashboard_prepare_manual_command",
            (),
            {
                "target": "chassis",
                "action": name,
                "background": background,
            },
        )

    first = threading.Thread(
        target=submit,
        args=("forward",),
        kwargs={"background": False},
        daemon=True,
    )
    second = threading.Thread(
        target=submit,
        args=("backward",),
        kwargs={"background": True},
        daemon=True,
    )
    try:
        first.start()
        assert first_started.wait(1.0)
        second.start()
        assert second_started.wait(0.05) is False
        assert max_active == 1
        release_first.set()
        first.join(1.0)
        second.join(1.0)
        assert not first.is_alive()
        assert not second.is_alive()
        assert max_active == 1
        assert order == [
            "start:forward",
            "end:forward",
            "start:backward",
            "end:backward",
        ]
        assert results == {
            "forward": {"action": "forward"},
            "backward": {"action": "backward"},
        }
    finally:
        release_first.set()
        shutdown.set()
        processor.join(1.0)


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
