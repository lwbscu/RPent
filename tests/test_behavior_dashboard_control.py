from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

import numpy as np
import pytest

from robots.behavior.dashboard_control import (
    BehaviorCommandArbiter,
    BehaviorDashboardController,
    BehaviorRawSuccessLatch,
    ControlRequestError,
)
from robots.behavior.dashboard_server import DashboardServer
from robots.behavior.dashboard_state import State


def _state(tmp_path) -> State:
    return State(
        run_id="behavior/test",
        name="test_s1",
        suite="behavior_2025_challenge",
        task=1,
        seed=1,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )


def _receipt(
    *,
    run_nonce: str = "a" * 32,
    attempt_nonce: str = "b" * 32,
    attempt_index: int = 1,
    env_step: int = 7,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "run_nonce": run_nonce,
        "attempt_nonce": attempt_nonce,
        "attempt_index": attempt_index,
        "env_step": env_step,
        "raw_done": {"success": True},
    }
    return _resign_receipt(value)


def _resign_receipt(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    return value


class _Toolkit:
    def __init__(self, *, gate: threading.Event | None = None) -> None:
        self.gate = gate
        self.calls: list[tuple[str, str, str]] = []
        self.started = threading.Event()

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        return {"motion_available": True, "observe_available": True}

    def dashboard_manual_command(
        self, *, target: str, action: str, camera: str
    ) -> dict[str, Any]:
        self.calls.append((target, action, camera))
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(2.0)
        step = len(self.calls)
        return {
            "primitive_success": True,
            "primitive_used": "observe" if action == "observe" else "jog_base",
            "_frames_bytes": {
                "head": f"head-{step}".encode(),
                "left_wrist": f"left-{step}".encode(),
                "right_wrist": f"right-{step}".encode(),
            },
            "capture_group_id": f"group-{step}",
            "simulator_step": step,
        }


def _controller(
    state: State,
    *,
    toolkit: _Toolkit | None = None,
    lease_timeout_s: float = 1.2,
) -> tuple[BehaviorDashboardController, BehaviorCommandArbiter, _Toolkit]:
    latch = BehaviorRawSuccessLatch(
        run_nonce="a" * 32,
        attempt_nonce="b" * 32,
        attempt_index=1,
    )
    arbiter = BehaviorCommandArbiter(success_latch=latch)
    controller = BehaviorDashboardController(
        state=state,
        arbiter=arbiter,
        success_latch=latch,
        motion_available=True,
        observe_available=True,
        lease_timeout_s=lease_timeout_s,
    )
    bound = toolkit or _Toolkit()
    controller.bind_toolkit(bound)
    state.bind_controller(controller)
    controller.activate()
    return controller, arbiter, bound


def _wait_phase(controller: BehaviorDashboardController, phase: str) -> None:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if controller.snapshot()["phase"] == phase:
            return
        time.sleep(0.005)
    raise AssertionError(f"controller did not reach {phase}: {controller.snapshot()}")


def test_success_latch_requires_hash_and_attempt_binding() -> None:
    latch = BehaviorRawSuccessLatch(
        run_nonce="a" * 32,
        attempt_nonce="b" * 32,
        attempt_index=1,
    )
    assert not latch.observe({"task_success": True})
    assert not latch.observe(
        {"official_success_receipt": {**_receipt(), "receipt_sha256": "0" * 64}}
    )
    assert not latch.observe(
        {"official_success_receipt": _receipt(attempt_nonce="c" * 32)}
    )
    receipt = _receipt()
    assert latch.observe({"official_success_receipt": receipt})
    assert latch.is_latched()
    assert latch.receipt_binding()["receipt_sha256"] == receipt["receipt_sha256"]
    assert latch.observe({"task_success": False})


@pytest.mark.parametrize(
    ("mutate", "resign"),
    [
        (lambda receipt: receipt.pop("env_step"), True),
        (lambda receipt: receipt.update({"extra": "field"}), True),
        (lambda receipt: receipt.update({"schema_version": 2}), True),
        (lambda receipt: receipt.update({"schema_version": True}), True),
        (lambda receipt: receipt.update({"run_nonce": "A" * 32}), True),
        (lambda receipt: receipt.update({"attempt_nonce": "b" * 31}), True),
        (
            lambda receipt: receipt.update(
                {"raw_done": {"success": np.bool_(True)}}
            ),
            False,
        ),
    ],
)
def test_success_latch_rejects_noncanonical_receipts(mutate, resign: bool) -> None:
    latch = BehaviorRawSuccessLatch(
        run_nonce="a" * 32,
        attempt_nonce="b" * 32,
        attempt_index=1,
    )
    receipt = _receipt()
    mutate(receipt)
    if resign:
        _resign_receipt(receipt)
    assert not latch.observe({"official_success_receipt": receipt})


def test_controller_cannot_activate_with_unbound_success_latch(tmp_path) -> None:
    state = _state(tmp_path)
    latch = BehaviorRawSuccessLatch()
    controller = BehaviorDashboardController(
        state=state,
        arbiter=BehaviorCommandArbiter(success_latch=latch),
        success_latch=latch,
        motion_available=True,
        observe_available=True,
    )
    controller.bind_toolkit(_Toolkit())
    with pytest.raises(RuntimeError, match="not bound"):
        controller.activate()
    assert controller.close(1.0)


def test_capture_group_is_atomic_idempotent_and_rejects_stale(tmp_path) -> None:
    state = _state(tmp_path)
    frames = {"head": b"h", "left_wrist": b"l", "right_wrist": b"r"}
    assert state.on_frame_group(frames, capture_group_id="g1", simulator_step=5)
    first = state.snapshot()
    assert first["frame_revisions"] == {
        "head": 1,
        "left_wrist": 1,
        "right_wrist": 1,
    }
    assert state.on_frame_group(frames, capture_group_id="g1", simulator_step=5)
    assert state.snapshot()["frame_revisions"] == first["frame_revisions"]
    assert not state.on_frame_group(
        {**frames, "head": b"changed"},
        capture_group_id="g1",
        simulator_step=5,
    )
    assert not state.on_frame_group(
        frames, capture_group_id="older", simulator_step=4
    )
    assert state.snapshot()["frame_revisions"] == first["frame_revisions"]
    assert state.frame("head") == b"h"


def test_malformed_manual_group_never_falls_back_to_partial_frames(tmp_path) -> None:
    state = _state(tmp_path)
    state.on_frame_group(
        {"head": b"h0", "left_wrist": b"l0", "right_wrist": b"r0"},
        capture_group_id="initial",
        simulator_step=1,
    )
    command = {
        "command_id": "c1",
        "lease_id": "l1",
        "sequence": 1,
        "target": "chassis",
        "action": "forward",
        "camera": "head",
    }
    state.on_manual_command_start(command)
    before = state.snapshot()
    state.on_manual_command_result(
        command,
        {
            "primitive_success": False,
            "error": "Camera refresh failed",
            "_frames_bytes": {
                "head": b"h1",
                "left_wrist": b"l1",
                "right_wrist": b"r1",
            },
            "capture_group_id": "broken",
            "simulator_step": None,
        },
        official_success_latched=False,
    )
    after = state.snapshot()
    assert after["capture_group_id"] == "initial"
    assert after["frame_revisions"] == before["frame_revisions"]
    assert state.frame("head") == b"h0"


def test_idempotency_one_shot_release_and_manual_timeline(tmp_path) -> None:
    state = _state(tmp_path)
    controller, _, toolkit = _controller(state)
    try:
        command, duplicate = controller.submit(
            lease_id="observe-1",
            sequence=1,
            target="chassis",
            action="observe",
            camera="right_wrist",
        )
        assert not duplicate
        _wait_phase(controller, "completed")
        repeated, duplicate = controller.submit(
            lease_id="observe-1",
            sequence=1,
            target="chassis",
            action="observe",
            camera="right_wrist",
        )
        assert duplicate
        assert repeated.command_id == command.command_id
        assert len(toolkit.calls) == 1

        controller.submit(
            lease_id="observe-2",
            sequence=1,
            target="left_arm",
            action="observe",
            camera="head",
        )
        deadline = time.monotonic() + 2.0
        while len(toolkit.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(toolkit.calls) == 2
        detail = state.run_detail()
        manual = [
            item
            for item in detail["timeline"]
            if item.get("source") == "dashboard_manual"
        ]
        assert len(manual) == 2
        assert manual[0]["capture_group_id"] == "group-1"
        assert detail["last_selected_camera"] == "head"
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_single_flight_sequence_and_lease_fingerprint(tmp_path) -> None:
    state = _state(tmp_path)
    gate = threading.Event()
    controller, _, toolkit = _controller(state, toolkit=_Toolkit(gate=gate))
    try:
        controller.submit(
            lease_id="hold",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert toolkit.started.wait(1.0)
        with pytest.raises(ControlRequestError, match="still in flight"):
            controller.submit(
                lease_id="hold",
                sequence=2,
                target="chassis",
                action="forward",
                camera="head",
            )
        gate.set()
        _wait_phase(controller, "completed")
        with pytest.raises(ControlRequestError, match="cannot change"):
            controller.submit(
                lease_id="hold",
                sequence=2,
                target="chassis",
                action="backward",
                camera="head",
            )
        controller.submit(
            lease_id="hold",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
    finally:
        gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_agent_waiter_prevents_manual_repeat(tmp_path) -> None:
    state = _state(tmp_path)
    manual_gate = threading.Event()
    agent_gate = threading.Event()
    controller, arbiter, toolkit = _controller(
        state, toolkit=_Toolkit(gate=manual_gate)
    )

    def agent() -> None:
        with arbiter.agent_transaction():
            agent_gate.wait(2.0)

    try:
        controller.submit(
            lease_id="hold",
            sequence=1,
            target="right_arm",
            action="up",
            camera="head",
        )
        assert toolkit.started.wait(1.0)
        thread = threading.Thread(target=agent)
        thread.start()
        deadline = time.monotonic() + 1.0
        while arbiter.snapshot()["agent_waiters"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert arbiter.snapshot()["agent_waiters"] == 1
        manual_gate.set()
        deadline = time.monotonic() + 1.0
        while arbiter.snapshot()["owner"] != "agent" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert arbiter.snapshot()["owner"] == "agent"
        with pytest.raises(ControlRequestError, match="busy"):
            controller.submit(
                lease_id="hold",
                sequence=2,
                target="right_arm",
                action="up",
                camera="head",
            )
        agent_gate.set()
        thread.join(1.0)
    finally:
        manual_gate.set()
        agent_gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_heartbeat_expiry_stops_lease_and_allows_fresh_lease(tmp_path) -> None:
    state = _state(tmp_path)
    controller, _, _ = _controller(state, lease_timeout_s=0.15)
    try:
        controller.submit(
            lease_id="expiring",
            sequence=1,
            target="left_arm",
            action="forward",
            camera="head",
        )
        _wait_phase(controller, "completed")
        deadline = time.monotonic() + 1.0
        while (
            controller.snapshot().get("stop_reason") != "lease_expired"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert controller.snapshot()["stop_reason"] == "lease_expired"
        with pytest.raises(ControlRequestError, match="not active"):
            controller.heartbeat(lease_id="expiring")
        controller.submit(
            lease_id="fresh",
            sequence=1,
            target="left_arm",
            action="forward",
            camera="head",
        )
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_stop_before_worker_admission_cancels_with_zero_rpc(tmp_path) -> None:
    state = _state(tmp_path)
    latch = BehaviorRawSuccessLatch(
        run_nonce="a" * 32,
        attempt_nonce="b" * 32,
        attempt_index=1,
    )
    arbiter = BehaviorCommandArbiter(success_latch=latch)
    worker_gate = threading.Event()

    class _PausedController(BehaviorDashboardController):
        def _run_command(self, command) -> None:
            assert worker_gate.wait(2.0)
            super()._run_command(command)

    toolkit = _Toolkit()
    controller = _PausedController(
        state=state,
        arbiter=arbiter,
        success_latch=latch,
        motion_available=True,
        observe_available=True,
    )
    controller.bind_toolkit(toolkit)
    state.bind_controller(controller)
    controller.activate()
    try:
        controller.submit(
            lease_id="paused",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        controller.stop(lease_id="paused")
        worker_gate.set()
        _wait_phase(controller, "cancelled")
        assert toolkit.calls == []
        assert state.run_detail()["timeline"][-1]["status"] == "failed"
    finally:
        worker_gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_agent_owner_is_mirrored_to_state_and_terminal_precedes_release(
    tmp_path,
) -> None:
    class _RecordingState(State):
        def __init__(self, **kwargs) -> None:
            self.control_updates: list[dict[str, Any]] = []
            super().__init__(**kwargs)

        def update_control_snapshot(self, snapshot) -> None:
            self.control_updates.append(dict(snapshot))
            super().update_control_snapshot(snapshot)

    state = _RecordingState(
        run_id="behavior/test",
        name="test_s1",
        suite="behavior_2025_challenge",
        task=1,
        seed=1,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )
    controller, arbiter, _ = _controller(state)
    agent_gate = threading.Event()
    agent_entered = threading.Event()

    def agent() -> None:
        with arbiter.agent_transaction():
            agent_entered.set()
            agent_gate.wait(2.0)

    try:
        agent_thread = threading.Thread(target=agent)
        agent_thread.start()
        assert agent_entered.wait(1.0)
        assert controller.snapshot()["owner"] == "agent"
        deadline = time.monotonic() + 1.0
        while (
            state.snapshot()["control"].get("owner") != "agent"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert state.snapshot()["control"]["owner"] == "agent"
        agent_gate.set()
        agent_thread.join(1.0)

        controller.submit(
            lease_id="terminal-order",
            sequence=1,
            target="chassis",
            action="observe",
            camera="head",
        )
        _wait_phase(controller, "completed")
        terminal_manual = next(
            index
            for index, value in enumerate(state.control_updates)
            if value.get("phase") == "completed"
            and value.get("owner") == "manual"
        )
        released = next(
            index
            for index, value in enumerate(state.control_updates[terminal_manual + 1 :])
            if value.get("owner") is None
        )
        assert released >= 0
    finally:
        agent_gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_capture_error_fails_repeat_but_preserves_valid_raw_success(tmp_path) -> None:
    class _SuccessWithCaptureError(_Toolkit):
        def dashboard_manual_command(
            self, *, target: str, action: str, camera: str
        ) -> dict[str, Any]:
            self.calls.append((target, action, camera))
            return {
                "primitive_success": True,
                "task_success": True,
                "official_success_receipt": _receipt(),
                "capture_error": "Camera refresh failed",
                "stop_reason": "camera_refresh_failed",
            }

    state = _state(tmp_path)
    controller, _, toolkit = _controller(
        state, toolkit=_SuccessWithCaptureError()
    )
    before = state.snapshot()["frame_revisions"]
    try:
        controller.submit(
            lease_id="success-capture-error",
            sequence=1,
            target="chassis",
            action="observe",
            camera="head",
        )
        _wait_phase(controller, "failed")
        snapshot = controller.snapshot()
        assert snapshot["success_latched"] is True
        assert snapshot["capture_error"] == "Camera refresh failed"
        assert snapshot["available"] is False
        detail = state.run_detail()
        assert detail["progress"]["official_task_success"] is True
        assert detail["frame_revisions"] == before
        assert detail["timeline"][-1]["status"] == "failed"
        assert len(toolkit.calls) == 1
        with pytest.raises(ControlRequestError) as caught:
            controller.submit(
                lease_id="after-success",
                sequence=1,
                target="chassis",
                action="observe",
                camera="head",
            )
        assert caught.value.status_code == 410
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_receipt_conflict_failure_releases_permit_and_publishes_terminal(
    tmp_path,
) -> None:
    class _ConflictingLatch(BehaviorRawSuccessLatch):
        def is_latched(self) -> bool:
            # Simulate admission racing with the EnvClient's first receipt latch.
            return False

    class _SecondReceiptToolkit(_Toolkit):
        def dashboard_manual_command(
            self, *, target: str, action: str, camera: str
        ) -> dict[str, Any]:
            result = super().dashboard_manual_command(
                target=target,
                action=action,
                camera=camera,
            )
            result["official_success_receipt"] = _receipt(env_step=8)
            return result

    state = _state(tmp_path)
    latch = _ConflictingLatch(
        run_nonce="a" * 32,
        attempt_nonce="b" * 32,
        attempt_index=1,
    )
    assert latch.observe({"official_success_receipt": _receipt(env_step=7)})
    arbiter = BehaviorCommandArbiter(success_latch=latch)
    controller = BehaviorDashboardController(
        state=state,
        arbiter=arbiter,
        success_latch=latch,
        motion_available=True,
        observe_available=True,
    )
    controller.bind_toolkit(_SecondReceiptToolkit())
    state.bind_controller(controller)
    controller.activate()
    try:
        controller.submit(
            lease_id="receipt-conflict",
            sequence=1,
            target="chassis",
            action="observe",
            camera="head",
        )
        _wait_phase(controller, "failed")
        snapshot = controller.snapshot()
        assert snapshot["owner"] is None
        assert "receipt changed" in snapshot["error"]
        assert snapshot["worker_exception_type"] == "RuntimeError"
        assert controller.drain(0.2)
        timeline = state.run_detail()["timeline"][-1]
        assert timeline["status"] == "failed"
        assert "receipt changed" in timeline["result"]["error"]
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_state_result_exception_retries_failed_terminal_and_releases_permit(
    tmp_path,
) -> None:
    class _FailsFirstResult(State):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.result_calls = 0

        def on_manual_command_result(self, *args, **kwargs) -> None:
            self.result_calls += 1
            if self.result_calls == 1:
                raise RuntimeError("state publish boom")
            super().on_manual_command_result(*args, **kwargs)

    state = _FailsFirstResult(
        run_id="behavior/test",
        name="test_s1",
        suite="behavior_2025_challenge",
        task=1,
        seed=1,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )
    controller, arbiter, _ = _controller(state)
    try:
        controller.submit(
            lease_id="state-failure",
            sequence=1,
            target="left_arm",
            action="forward",
            camera="head",
        )
        _wait_phase(controller, "failed")
        assert state.result_calls == 2
        assert arbiter.snapshot()["owner"] is None
        assert controller.drain(0.2)
        snapshot = controller.snapshot()
        assert "state publish boom" in snapshot["error"]
        timeline = state.run_detail()["timeline"][-1]
        assert timeline["status"] == "failed"
        assert "state publish boom" in timeline["result"]["error"]
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_control_routes_validate_and_return_202(tmp_path) -> None:
    testclient = pytest.importorskip("fastapi.testclient")
    state = _state(tmp_path)
    controller, _, _ = _controller(state)
    server = DashboardServer()
    server.register(state)
    client = testclient.TestClient(server._app)
    payload = {
        "run": state.run_id,
        "lease_id": "route-1",
        "sequence": 1,
        "target": "chassis",
        "action": "observe",
        "camera": "head",
    }
    try:
        assert client.post(
            "/api/run/control/command", json={**payload, "distance": 10}
        ).status_code == 422
        assert client.post(
            "/api/run/control/command", json={**payload, "run": "missing"}
        ).status_code == 404
        response = client.post("/api/run/control/command", json=payload)
        assert response.status_code == 202
        assert response.json()["command_id"]
        assert client.get(
            "/api/run/control/state", params={"run": state.run_id}
        ).status_code == 200
        camera = client.post(
            "/api/run/control/camera",
            json={"run": state.run_id, "camera": "left_wrist"},
        )
        assert camera.status_code == 200
        assert state.snapshot()["last_selected_camera"] == "left_wrist"
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)

    unbound = _state(tmp_path / "unbound")
    server.register(unbound)
    assert client.get(
        "/api/run/control/state", params={"run": unbound.run_id}
    ).status_code == 409
    assert client.post(
        "/api/run/control/camera",
        json={"run": unbound.run_id, "camera": "head"},
    ).status_code == 409
    unbound.mark_done(False)
    assert client.post(
        "/api/run/control/camera",
        json={"run": unbound.run_id, "camera": "head"},
    ).status_code == 410
