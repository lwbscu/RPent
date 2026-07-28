# Copyright (c) 2026 RPent contributors

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
from PIL import Image


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "behavior_dashboard_live_acceptance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "behavior_dashboard_live_acceptance",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_optional_high_dpi_reference_identity_and_geometry():
    module = _module()
    if not module.REFERENCE.exists():
        pytest.skip("external high-DPI visual reference is not installed")

    assert hashlib.sha256(module.REFERENCE.read_bytes()).hexdigest() == (
        module.REFERENCE_SHA256
    )
    with Image.open(module.REFERENCE) as reference:
        assert reference.size == (1280, 867)


def _before_rows(*, hand: str, state: str, attached: bool):
    return [
        {
            "result": {
                "capability": {
                    "attachments": {
                        "available": True,
                        "conflict": False,
                        "by_hand": {
                            "left": {"attached": attached if hand == "left" else False},
                            "right": {
                                "attached": attached if hand == "right" else False
                            },
                        },
                    },
                    "gripper_state": {
                        "left": state if hand == "left" else "open",
                        "right": state if hand == "right" else "open",
                    },
                }
            }
        }
    ]


def test_gripper_actions_run_close_then_open():
    module = _module()

    for target in module.TARGETS:
        order = module.MOTION_ORDER[target]
        assert order.index("close") < order.index("open")


def test_close_requires_initially_open_gripper():
    module = _module()

    open_checks = module._gripper_safety_checks(
        before_rows=_before_rows(hand="left", state="open", attached=False),
        target="left_arm",
        action="close",
    )
    closed_checks = module._gripper_safety_checks(
        before_rows=_before_rows(hand="left", state="closed", attached=False),
        target="left_arm",
        action="close",
    )

    assert all(open_checks.values())
    assert closed_checks["close_selected_gripper_initially_open"] is False


def test_open_requires_unattached_hand_but_allows_closed_or_open():
    module = _module()

    for state in ("closed", "open"):
        checks = module._gripper_safety_checks(
            before_rows=_before_rows(hand="right", state=state, attached=False),
            target="right_arm",
            action="open",
        )
        assert all(checks.values())
        assert checks["open_selected_gripper_initially_closed_or_open"] is True

    attached_checks = module._gripper_safety_checks(
        before_rows=_before_rows(hand="right", state="closed", attached=True),
        target="right_arm",
        action="open",
    )
    assert attached_checks["selected_hand_unattached"] is False
    unknown_checks = module._gripper_safety_checks(
        before_rows=_before_rows(hand="right", state="unknown", attached=False),
        target="right_arm",
        action="open",
    )
    assert (
        unknown_checks["open_selected_gripper_initially_closed_or_open"] is False
    )


def test_open_semantics_require_capability_latch_to_restore_open():
    module = _module()
    row = {
        "source": "dashboard_manual",
        "target": "left_arm",
        "manual_action": "open",
        "primitive": "set_gripper",
    }
    result = {
        "action": "open",
        "metrics": {
            "manual_primitive": "set_gripper",
            "hand": "left",
            "opening": 1.0,
            "gripper_latch": 1.0,
            "network_primitive_calls": 1,
            "retreat_executed": False,
        },
        "capability": {"gripper_state": {"left": "open", "right": "open"}},
    }

    checks = module._semantic_checks(
        target="left_arm",
        action="open",
        row=row,
        result=result,
    )
    assert all(checks.values())

    result["capability"]["gripper_state"]["left"] = "closed"
    checks = module._semantic_checks(
        target="left_arm",
        action="open",
        row=row,
        result=result,
    )
    assert checks["open_capability_latch_restored"] is False
    result["capability"]["gripper_state"]["left"] = "open"
    result["metrics"]["network_primitive_calls"] = 2
    checks = module._semantic_checks(
        target="left_arm",
        action="open",
        row=row,
        result=result,
    )
    assert checks["single_gripper_network_call"] is False


@pytest.mark.parametrize(
    ("action", "kind", "direction", "signed_key", "signed_value"),
    [
        ("forward", "translation", "forward", "meters", 0.05),
        ("backward", "translation", "backward", "meters", -0.05),
        ("turn_left", "rotation", "left", "degrees", 5.0),
        ("turn_right", "rotation", "right", "degrees", -5.0),
    ],
)
def test_chassis_semantics_use_timeline_request_and_current_fast_contract(
    action,
    kind,
    direction,
    signed_key,
    signed_value,
):
    module = _module()
    requested_step = {"kind": kind, signed_key: signed_value}
    row = {
        "source": "dashboard_manual",
        "target": "chassis",
        "action": "navigate_to",
        "manual_action": action,
        "primitive": "navigate_to",
        "args": {
            "action": action,
            "detail": "relative jog",
            "requested_step": requested_step,
        },
    }
    result = {
        "metrics": {
            "relative_motion": {
                "kind": kind,
                "direction": direction,
                **(
                    {"distance_m": 0.05}
                    if kind == "translation"
                    else {"angle_deg": 5.0}
                ),
            },
            "base_goal_construction": {
                "method": "analytic_full_q_base_assignment",
                "nonbase_locked_to_call_start": True,
            },
            "base_goal": [0.05, 0.0, 0.0],
            "final_position_error_m": 0.019,
            "final_yaw_error_rad": 0.04,
            "navigation_terminal": {
                "position_tolerance_m": 0.02,
                "orientation_tolerance_rad": (
                    module.BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD
                ),
            },
            "navigation_isolation": {"ok": True},
            "execution_resampling": {
                "method": "quintic_minimum_jerk",
                "measured_max_xy_step_m": 0.0075,
                "measured_max_yaw_step_rad": (
                    module.DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD
                ),
            },
        }
    }

    checks = module._semantic_checks(
        target="chassis",
        action=action,
        row=row,
        result=result,
    )
    assert all(checks.values()), checks


@pytest.mark.parametrize(("action", "expected_delta"), [("up", 0.03), ("down", -0.03)])
def test_torso_semantics_require_world_z_fixed_step(action, expected_delta):
    module = _module()
    row = {
        "source": "dashboard_manual",
        "target": "chassis",
        "manual_action": action,
    }
    result = {
        "metrics": {
            "manual_primitive": "jog_torso",
            "manual_action": action,
            "target_link": "torso_link4",
            "requested_delta_z_m": expected_delta,
            "fixed_server_step": True,
        }
    }

    checks = module._semantic_checks(
        target="chassis",
        action=action,
        row=row,
        result=result,
    )

    assert all(checks.values()), checks


@pytest.mark.parametrize("hand", ["left", "right"])
@pytest.mark.parametrize(
    ("action", "visual_direction", "signed_angle"),
    [
        ("rotate_left", "counterclockwise", math.radians(5.0)),
        ("rotate_right", "clockwise", -math.radians(5.0)),
    ],
)
def test_wrist_semantics_require_per_hand_visual_direction_and_fixed_step(
    hand,
    action,
    visual_direction,
    signed_angle,
):
    module = _module()
    target = f"{hand}_arm"
    row = {
        "source": "dashboard_manual",
        "target": target,
        "manual_action": action,
    }
    result = {
        "metrics": {
            "manual_primitive": "jog_wrist",
            "manual_action": action,
            "hand": hand,
            "visual_direction": visual_direction,
            "requested_rotation_rad": signed_angle,
            "fixed_server_step": True,
            "final_position_drift_m": 0.001,
            "position_drift_limit_m": 0.005,
            "calibration": {
                "verified": True,
                "hand": hand,
                "visual_ccw_angle_sign": 1.0,
            },
        }
    }

    checks = module._semantic_checks(
        target=target,
        action=action,
        row=row,
        result=result,
    )

    assert all(checks.values()), checks


def test_manual_quiescence_ignores_independent_capture_plane():
    module = _module()
    payload = {
        "control": {
            "busy": True,
            "owner": "manual",
            "lease_id": "retained-terminal-lease",
            "lease_status": "stopped",
            "current_command": None,
            "planning_command": None,
            "queue_depth": 0,
            "phase": "completed",
            "capture": {"phase": "started", "revision": 9, "error": None},
        }
    }
    assert module._manual_control_quiescent(payload) is True

    for field, value in (
        ("lease_status", "active"),
        ("current_command", {"command_id": "head"}),
        ("planning_command", {"command_id": "planning"}),
        ("queue_depth", 1),
    ):
        busy_payload = json.loads(json.dumps(payload))
        busy_payload["control"][field] = value
        assert module._manual_control_quiescent(busy_payload) is False
    agent_payload = json.loads(json.dumps(payload))
    agent_payload["control"]["owner"] = "agent"
    assert module._manual_control_quiescent(agent_payload) is False


def _sustain_payload(*, command_id: str, revision: int):
    capabilities = {
        "planner_available": True,
        "position_control_ready": True,
        "simulation_identity": "behavior_omnigibson_r1pro",
        "base_available": True,
        "torso_available": True,
        "eef_available": {"left": True, "right": True},
        "wrist_rotation_available": {"left": True, "right": True},
        "gripper_available": {"left": True, "right": True},
    }
    return {
        "state": "running",
        "terminated": False,
        "simulator_step": 12,
        "capture_group_id": f"group-{revision}",
        "frame_revisions": {
            "head": revision,
            "left_wrist": revision,
            "right_wrist": revision,
        },
        "frame_indices": {
            "head": 12,
            "left_wrist": 12,
            "right_wrist": 12,
        },
        "timeline_revision": revision,
        "timeline": [
            {
                "source": "dashboard_manual",
                "command_id": command_id,
                "status": "completed",
            }
        ],
        "control": {
            "control_revision": revision,
            "available": True,
            "motion_available": True,
            "observe_available": True,
            "busy": False,
            "owner": None,
            "lease_status": "stopped",
            "current_command": None,
            "planning_command": None,
            "queue_depth": 0,
            "success_latched": False,
            "capabilities": capabilities,
        },
    }


def test_final_release_oracle_requires_torso_and_both_wrists():
    module = _module()
    payload = _sustain_payload(command_id="ready", revision=3)

    assert len(module.RELEASE_EXPECTED_ENABLED) == 29
    assert len(module.RELEASE_EXPECTED_DISABLED) == 4
    assert module.RELEASE_EXPECTED_DISABLED == {
        ("chassis", "rotate_left"),
        ("chassis", "rotate_right"),
        ("chassis", "open"),
        ("chassis", "close"),
    }
    assert all(module._control_ready_checks(payload).values())

    payload["control"]["capabilities"]["torso_available"] = False
    checks = module._control_ready_checks(payload)
    assert checks["release_torso_enabled"] is False
    payload["control"]["capabilities"]["torso_available"] = True
    payload["control"]["capabilities"]["wrist_rotation_available"]["right"] = False
    checks = module._control_ready_checks(payload)
    assert checks["release_wrist_enabled"] is False


def test_planning_only_ready_requires_geometry_while_wrist_release_stays_closed():
    module = _module()
    payload = _sustain_payload(command_id="ready", revision=3)
    capabilities = payload["control"]["capabilities"]
    capabilities["wrist_rotation_available"] = {
        "left": False,
        "right": False,
    }
    capabilities["planner"] = {
        "wrist_geometry": {
            hand: {
                "verified": True,
                "release_admission": False,
                "real_visual_probe_required": True,
            }
            for hand in ("left", "right")
        }
    }

    checks = module._planning_only_ready_checks(payload)

    assert "release_wrist_enabled" not in checks
    assert all(checks.values())
    capabilities["planner"]["wrist_geometry"]["right"]["verified"] = False
    checks = module._planning_only_ready_checks(payload)
    assert checks["planning_wrist_geometry_verified"] is False


def test_planning_only_pre_capture_ready_allows_one_initial_head_frame():
    module = _module()
    payload = _sustain_payload(command_id="ready", revision=3)
    capabilities = payload["control"]["capabilities"]
    capabilities["wrist_rotation_available"] = {
        "left": False,
        "right": False,
    }
    capabilities["planner"] = {
        "wrist_geometry": {
            hand: {
                "verified": True,
                "release_admission": False,
                "real_visual_probe_required": True,
            }
            for hand in ("left", "right")
        }
    }
    payload["capture_group_id"] = None
    payload["frame_indices"].update(
        {"left_wrist": -1, "right_wrist": -1}
    )
    payload["frame_revisions"].update(
        {"left_wrist": 0, "right_wrist": 0}
    )

    pre_capture = module._planning_only_pre_capture_ready_checks(payload)
    full = module._planning_only_ready_checks(payload)

    assert all(pre_capture.values())
    assert full["capture_group_nonempty"] is False
    assert full["three_revisions_positive"] is False
    assert full["three_indices_match_simulator_step"] is False


def test_wait_for_run_ready_retries_the_supplied_checks(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    payloads = iter(({"ready": False}, {"ready": True}))
    monkeypatch.setattr(
        module,
        "_run_payload",
        lambda base_url, run_id: (
            next(payloads)
            if (base_url, run_id) == ("http://unused", "run")
            else pytest.fail("unexpected run identity")
        ),
    )

    def wait(predicate, *, timeout_s):
        assert timeout_s == 12.0
        assert predicate() is None
        return predicate()

    monkeypatch.setattr(module, "_wait", wait)

    result = module._wait_for_run_ready(
        base_url="http://unused",
        run_id="run",
        checks=lambda payload: {"ready": payload["ready"] is True},
        timeout_s=12.0,
    )

    assert result == {"ready": True}


def test_raw_success_gate_uses_latch_or_returned_task_success_not_termination():
    module = _module()

    assert module._raw_success_latched({"terminated": True}) is False
    assert (
        module._raw_success_latched(
            {"control": {"success_latched": True}},
        )
        is True
    )
    assert (
        module._raw_success_latched(
            {"control": {"success_latched": False}},
            {"task_success": True},
        )
        is True
    )


def test_button_raw_success_cutover_performs_no_frame_read_or_playback_wait(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _module()
    before = _sustain_payload(command_id="old", revision=3)
    before["timeline"] = []
    before["control"]["capture"] = {
        "phase": "idle",
        "revision": 3,
        "error": None,
    }
    row = {
        "source": "dashboard_manual",
        "target": "chassis",
        "manual_action": "observe",
        "primitive": "observe",
        "action": "observe",
        "command_id": "success-command",
        "status": "completed",
        "step": 12,
        "result": {
            "primitive_success": True,
            "task_success": True,
            "metrics": {"camera": "head"},
        },
    }
    terminal = json.loads(json.dumps(before))
    terminal["terminated"] = True
    terminal["control"]["success_latched"] = True
    terminal["timeline"] = [row]
    fetches: list[dict] = []

    class Page:
        sleeps: list[int] = []

        def sleep(self, milliseconds):
            self.sleeps.append(milliseconds)

        def evaluate(self, _expression):
            raise AssertionError("raw-success path must not wait for playback")

    page = Page()
    payload_calls = 0

    def run_payload(*_args):
        nonlocal payload_calls
        payload_calls += 1
        return before if payload_calls == 1 else terminal

    monkeypatch.setattr(module, "_run_payload", run_payload)
    monkeypatch.setattr(
        module,
        "_button_state",
        lambda *_args: {
            "aria_disabled": "false",
            "tooltip": "Observe",
            "pressed": False,
        },
    )
    monkeypatch.setattr(module, "_fetch_audit", lambda _page: fetches)

    def dispatch(_page, *, event_type, **_kwargs):
        if event_type == "pointerdown":
            fetches.append(
                {
                    "url": "/api/run/control/command",
                    "status": 202,
                    "body": {
                        "run": "run",
                        "target": "chassis",
                        "action": "observe",
                    },
                    "response": {"command_id": "success-command"},
                }
            )
        return {}

    monkeypatch.setattr(module, "_dispatch_pointer_event", dispatch)
    monkeypatch.setattr(
        module,
        "_terminal_command_row",
        lambda *_args: (terminal, row),
    )
    monkeypatch.setattr(
        module,
        "_wait",
        lambda predicate, **_kwargs: predicate(),
    )
    monkeypatch.setattr(
        module,
        "_save_frames",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("frame read after raw success")
        ),
    )

    result = module._exercise_button(
        page=page,
        base_url="http://unused",
        run_id="run",
        output_dir=tmp_path,
        target="chassis",
        action="observe",
        case_index=1,
        timeout_s=1.0,
        expected_enabled=True,
        hold_ms=90,
    )

    assert result["after"]["success_latched"] is True
    assert result["frame_evidence"]["post_success_frame_http_reads"] == 0
    assert result["checks"]["raw_success_cutover_zero_frame_reads"] is True
    assert page.sleeps == [90]


def test_sustain_uses_passive_sse_freshness_without_env_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _module()
    payloads = [_sustain_payload(command_id="baseline", revision=3)]
    calls = 0

    def run_payload(_base_url, _run_id):
        nonlocal calls
        calls += 1
        return payloads[0]

    class Page:
        events: list[str] = []

        def evaluate(self, expression):
            if "document_ready:" in expression:
                return {
                    "document_ready": True,
                    "event_source_open_or_connecting": True,
                    "current_run": "run",
                    "sse_listener_attached": True,
                    "sse_message_count": 1,
                    "sse_last_message_age_ms": 1,
                }
            if 'new PointerEvent("pointerdown"' in expression:
                self.events.append("pointerdown")
            if 'new PointerEvent("pointerup"' in expression:
                self.events.append("pointerup")
            return {}

        def sleep(self, _milliseconds):
            return None

    monkeypatch.setattr(module, "_run_payload", run_payload)
    monkeypatch.setattr(
        module,
        "_http_bytes",
        lambda *_args, **_kwargs: b"\x89PNG\r\n\x1a\nfresh",
    )
    page = Page()
    result = module._sustain_live_run(
        page=page,
        base_url="http://unused",
        run_id="run",
        output_path=tmp_path / "sustain.jsonl",
        duration_s=0.001,
        interval_s=0.25,
    )
    assert result["verdict"] == "pass"
    assert result["observe_probes"] == 0
    assert result["env_action_probes"] == 0
    assert result["passive_freshness_observed"] is True
    assert page.events == []
    assert calls >= 1


def test_sustain_rejects_fully_cached_health_without_passive_freshness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _module()
    payload = _sustain_payload(command_id="cached", revision=3)

    class CachedPage:
        def evaluate(self, expression):
            if "document_ready:" in expression:
                return {
                    "document_ready": True,
                    "event_source_open_or_connecting": True,
                    "current_run": "run",
                    "sse_listener_attached": True,
                    "sse_message_count": 0,
                    "sse_last_message_age_ms": None,
                }
            if "PointerEvent" in expression:
                raise AssertionError("sustain must not send an Env action")
            return {}

    monkeypatch.setattr(module, "_run_payload", lambda *_args: payload)
    monkeypatch.setattr(
        module,
        "_http_bytes",
        lambda *_args, **_kwargs: b"\x89PNG\r\n\x1a\ncached",
    )
    result = module._sustain_live_run(
        page=CachedPage(),
        base_url="http://unused",
        run_id="run",
        output_path=tmp_path / "cached.jsonl",
        duration_s=0.001,
        interval_s=0.25,
    )
    assert result["verdict"] == "fail"
    assert result["observe_probes"] == 0
    assert result["env_action_probes"] == 0
    assert result["passive_freshness_observed"] is False


def test_sustain_stops_all_probes_when_raw_success_is_already_latched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = _module()
    payload = _sustain_payload(command_id="success", revision=9)
    payload["terminated"] = True
    payload["control"]["success_latched"] = True
    calls = 0

    def run_payload(_base_url, _run_id):
        nonlocal calls
        calls += 1
        return payload

    class NoRequestPage:
        def evaluate(self, _expression):
            raise AssertionError("browser request after raw success")

        def sleep(self, _milliseconds):
            raise AssertionError("sleep after raw success")

    monkeypatch.setattr(module, "_run_payload", run_payload)
    monkeypatch.setattr(
        module,
        "_http_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("frame request after raw success")
        ),
    )
    result = module._sustain_live_run(
        page=NoRequestPage(),
        base_url="http://unused",
        run_id="run",
        output_path=tmp_path / "success.jsonl",
        duration_s=10.0,
        interval_s=1.0,
    )
    assert result["verdict"] == "pass"
    assert result["observe_probes"] == 0
    assert calls == 1


def test_planning_only_acceptance_runs_six_zero_action_probes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _module()
    payload = {
        "timeline_revision": 4,
        "simulator_step": 0,
        "capture_group_id": "capture:0:fresh",
        "frame_indices": {
            "head": 0,
            "left_wrist": 0,
            "right_wrist": 0,
        },
        "control": {
            "success_latched": False,
            "capabilities": {
                "torso_available": True,
                "wrist_rotation_available": {
                    "left": False,
                    "right": False,
                },
            },
        },
    }
    posts = []
    monkeypatch.setattr(
        module,
        "_run_payload",
        lambda *_args, **_kwargs: dict(payload),
    )

    def post(_base_url, path, body, *, timeout_s):
        posts.append((path, dict(body), timeout_s))
        motion_kind = (
            "torso" if body["target"] == "chassis" else "eef"
        )
        plan_id = f"plan-{len(posts)}"
        action = body["action"]
        target = body["target"]
        requested_step = (
            {
                "frame": "world",
                "torso_delta_z_m": (
                    module.TORSO_VERTICAL_STEP_M
                    if action == "up"
                    else -module.TORSO_VERTICAL_STEP_M
                ),
            }
            if target == "chassis"
            else {
                "frame": "wrist_camera",
                "angle_rad": module.WRIST_ROTATION_STEP_RAD,
                "visual_direction": (
                    "counterclockwise"
                    if action == "rotate_left"
                    else "clockwise"
                ),
            }
        )
        selected_hand = target.removesuffix("_arm")
        return {
            "ok": True,
            "planning_only": True,
            "release_admission": False,
            "target": target,
            "action": action,
            "plan_id": plan_id,
            "env_step_delta": 0,
            "zero_action_verified": True,
            "discarded": True,
            "discard_receipt": {
                "plan_id": plan_id,
                "discarded": True,
                "status": "discarded",
            },
            "prepared": {
                "target": target,
                "action": action,
                "plan_id": plan_id,
                "motion_kind": motion_kind,
                "requested_step": requested_step,
                **(
                    {}
                    if target == "chassis"
                    else {
                        "requested_rotation_rad": (
                            module.WRIST_ROTATION_STEP_RAD
                            if action == "rotate_left"
                            else -module.WRIST_ROTATION_STEP_RAD
                        ),
                        "calibration": {
                            "hand": selected_hand,
                            "capture_group_id": "capture:0:fresh",
                            "step_index": 0,
                        },
                    }
                ),
            },
            "safety_certificate": {
                "schema_version": 1,
                "motion_kind": motion_kind,
                "attachment_hand_count": 2,
                **(
                    {}
                    if target == "chassis"
                    else {"selected_hand": selected_hand}
                ),
                "checks": {
                    "world_collision_check": True,
                    "self_collision_check": True,
                    "post_interpolation_check": True,
                    "collision_admitted": True,
                    "dual_attachment_collision": True,
                },
                "admitted": True,
            },
        }

    monkeypatch.setattr(module, "_post_json", post)

    summary = module._run_planning_only_probes(
        base_url="http://unused",
        run_id="run",
        output_dir=tmp_path,
        timeout_s=270.0,
    )

    assert summary["verdict"] == "pass"
    assert summary["completed_probe_count"] == 6
    assert len(posts) == 6
    assert all(path == "/api/run/control/plan" for path, _, _ in posts)
    assert all(timeout == 270.0 for _, _, timeout in posts)
    assert all(
        probe["checks"]["safety_certificate_admitted"]
        and probe["checks"]["collision_checks_complete"]
        for probe in summary["probes"]
    )
    assert (tmp_path / "planning_only_probes.json").is_file()


def test_planning_only_capture_preflight_refreshes_three_views_without_step(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _module()
    before = {
        "simulator_step": 0,
        "capture_group_id": "old-group",
        "frame_indices": dict.fromkeys(module.CAMERAS, 0),
        "control": {"success_latched": False},
        "timeline": [],
    }
    command_id = "observe-command"
    result = {
        "primitive_success": True,
        "task_success": False,
    }
    after = {
        **before,
        "capture_group_id": "fresh-group",
        "frame_revisions": dict.fromkeys(module.CAMERAS, 2),
        "timeline": [
            {
                "source": "dashboard_manual",
                "command_id": command_id,
                "target": "chassis",
                "action": "observe",
                "status": "completed",
                "result": result,
            }
        ],
    }
    payloads = iter((before, after))
    monkeypatch.setattr(
        module,
        "_run_payload",
        lambda *_args, **_kwargs: next(payloads),
    )
    posted = []

    def post(_base_url, path, body, *, timeout_s):
        posted.append((path, dict(body), timeout_s))
        return {
            "accepted": True,
            "accepted_command_id": command_id,
        }

    monkeypatch.setattr(module, "_post_json", post)

    record = module._refresh_planning_only_capture(
        base_url="http://unused",
        run_id="run",
        timeout_s=20.0,
    )

    assert record["verdict"] == "pass"
    assert len(posted) == 1
    assert posted[0][0] == "/api/run/control/command"
    assert posted[0][1]["action"] == "observe"
    assert posted[0][1]["sequence"] == 1
    assert posted[0][2] == 20.0


def test_planning_only_acceptance_sends_no_probe_after_raw_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module = _module()
    payload = {
        "timeline_revision": 5,
        "simulator_step": 0,
        "frame_indices": {},
        "control": {"success_latched": True},
    }
    monkeypatch.setattr(
        module,
        "_run_payload",
        lambda *_args, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        module,
        "_post_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("planning request after raw success")
        ),
    )

    summary = module._run_planning_only_probes(
        base_url="http://unused",
        run_id="run",
        output_dir=tmp_path,
        timeout_s=270.0,
    )

    assert summary["verdict"] == "fail"
    assert summary["completed_probe_count"] == 0
