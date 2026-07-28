# Copyright (c) 2026 RPent contributors

from __future__ import annotations

import hashlib
import importlib.util
import json
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
    row = {"source": "dashboard_manual", "target": "left_arm"}
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
        "torso_available": False,
        "eef_available": {"left": True, "right": True},
        "wrist_rotation_available": {"left": False, "right": False},
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
