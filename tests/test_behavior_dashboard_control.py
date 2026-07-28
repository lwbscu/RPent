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
        self.permit_command_ids: list[str] = []
        self.prepare_calls: list[dict[str, Any]] = []
        self.execute_calls: list[tuple[str, str]] = []
        self.discard_calls: list[str] = []
        self.capture_calls: list[str] = []
        self.started = threading.Event()

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        return {"motion_available": True, "observe_available": True}

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
        permit_command_id: str,
    ) -> dict[str, Any]:
        self.permit_command_ids.append(permit_command_id)
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

    def dashboard_prepare_manual_command(
        self,
        *,
        target: str,
        action: str,
        predecessor_plan_id: str | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        call = {
            "target": target,
            "action": action,
            "predecessor_plan_id": predecessor_plan_id,
            "background": background,
        }
        self.prepare_calls.append(call)
        return {"plan_id": f"plan-{len(self.prepare_calls)}"}

    def dashboard_execute_prepared_command(
        self,
        *,
        plan_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        self.permit_command_ids.append(command_id)
        self.execute_calls.append((plan_id, command_id))
        self.calls.append(("prepared", plan_id, command_id))
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(2.0)
        return {
            "primitive_success": True,
            "primitive_used": "jog_base",
        }

    def dashboard_discard_prepared_command(
        self,
        *,
        plan_id: str,
    ) -> dict[str, Any]:
        self.discard_calls.append(plan_id)
        return {"discarded": True, "plan_id": plan_id}

    def dashboard_capture_views(self, *, command_id: str) -> dict[str, Any]:
        self.capture_calls.append(command_id)
        step = len(self.capture_calls)
        return {
            "_frames_bytes": {
                "head": f"capture-head-{step}".encode(),
                "left_wrist": f"capture-left-{step}".encode(),
                "right_wrist": f"capture-right-{step}".encode(),
            },
            "capture_group_id": f"capture-group-{step}",
            "simulator_step": 100 + step,
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
        assert toolkit.permit_command_ids == [command.command_id]
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
        assert manual[0]["capture_group_id"] is None
        assert detail["capture_group_id"] == "group-2"
        assert detail["last_selected_camera"] == "head"
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_prepared_planning_metadata_is_published_without_trajectories(
    tmp_path,
) -> None:
    class _PlanningMetadataToolkit(_Toolkit):
        def dashboard_prepare_manual_command(
            self,
            *,
            target: str,
            action: str,
            predecessor_plan_id: str | None = None,
            background: bool = False,
        ) -> dict[str, Any]:
            prepared = super().dashboard_prepare_manual_command(
                target=target,
                action=action,
                predecessor_plan_id=predecessor_plan_id,
                background=background,
            )
            return {
                **prepared,
                "planning_elapsed_s": 0.125,
                "planning_profile": "dashboard_jog",
                "fast_solver_deadline_s": 4.0,
                "latency_metrics": {
                    "schema_version": 1,
                    "operation": "dashboard_prepare",
                    "phases_s": {
                        "planner_prepare": 0.1,
                    },
                },
                "obstacle_refresh": {
                    "mode": "pose_only",
                    "topology_verified": True,
                    "elapsed_s": 0.012,
                },
                "selected_solver_stage": "fast_trajopt",
                "solver_stages": [
                    {
                        "name": "fast_trajopt",
                        "elapsed_s": 0.1,
                        "joint_trajectory": list(range(100)),
                    }
                ],
                "joint_trajectory": list(range(100)),
            }

    state = _state(tmp_path)
    controller, _, _ = _controller(state, toolkit=_PlanningMetadataToolkit())
    try:
        command, duplicate = controller.submit(
            lease_id="planning-metrics",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert not duplicate
        _wait_phase(controller, "completed")

        expected = {
            "plan_id": "plan-1",
            "planning_elapsed_s": 0.125,
            "planning_profile": "dashboard_jog",
            "fast_solver_deadline_s": 4.0,
            "latency_metrics": {
                "schema_version": 1,
                "operation": "dashboard_prepare",
                "phases_s": {
                    "planner_prepare": 0.1,
                },
            },
            "obstacle_refresh": {
                "mode": "pose_only",
                "topology_verified": True,
                "elapsed_s": 0.012,
            },
            "selected_solver_stage": "fast_trajopt",
            "solver_stages": [
                {
                    "name": "fast_trajopt",
                    "elapsed_s": 0.1,
                }
            ],
        }
        assert command.planning_metadata == expected
        assert command.result is not None
        assert command.result["metrics"] == expected
        terminal = controller.snapshot()["last_terminal"]
        assert terminal["result"]["metrics"] == expected
        timeline = state.run_detail()["timeline"]
        manual = next(item for item in timeline if item["command_id"] == command.command_id)
        assert manual["result"]["metrics"] == expected
        assert "trajectory" not in json.dumps(expected)
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_unbind_rejects_stale_controller_snapshot_publication(tmp_path) -> None:
    state = _state(tmp_path)
    controller, _, _ = _controller(state)

    state.unbind_controller(controller)
    unbound = state.snapshot()["control"]
    assert unbound["unavailable_reason"] == "controller_not_bound"
    assert state.control_controller() is None

    controller.configure_capabilities(
        motion_available=False,
        observe_available=False,
        unavailable_reason="stale_controller_update",
    )
    assert state.snapshot()["control"] == unbound

    assert controller.close(2.0)
    assert state.snapshot()["control"] == unbound


def test_capture_result_rejects_stale_generation_and_unbound_controller(
    tmp_path,
) -> None:
    state = _state(tmp_path)
    controller, _, _ = _controller(state)
    frames = {
        "head": b"head-new",
        "left_wrist": b"left-new",
        "right_wrist": b"right-new",
    }
    assert state.on_dashboard_capture_result(
        {
            "_frames_bytes": frames,
            "capture_group_id": "new",
            "simulator_step": 3,
        },
        controller=controller,
        generation=2,
    )
    assert not state.on_dashboard_capture_result(
        {
            "_frames_bytes": {
                "head": b"stale",
                "left_wrist": b"stale",
                "right_wrist": b"stale",
            },
            "capture_group_id": "stale",
            "simulator_step": 4,
        },
        controller=controller,
        generation=1,
    )
    assert state.frame("head") == b"head-new"
    state.unbind_controller(controller)
    assert not state.on_dashboard_capture_result(
        {
            "_frames_bytes": frames,
            "capture_group_id": "after-unbind",
            "simulator_step": 5,
        },
        controller=controller,
        generation=3,
    )
    assert controller.close(2.0)


def test_queue_accepts_tail_without_planning_during_head_execution(tmp_path) -> None:
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
        second, duplicate = controller.submit(
            lease_id="hold",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert not duplicate
        snapshot = controller.snapshot()
        assert snapshot["current_command"]["sequence"] == 1
        assert [item["sequence"] for item in snapshot["queue"]] == [2]
        with pytest.raises(ControlRequestError, match="cannot change"):
            controller.submit(
                lease_id="hold",
                sequence=3,
                target="chassis",
                action="backward",
                camera="head",
            )
        repeated, duplicate = controller.submit(
            lease_id="hold",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert duplicate and repeated.command_id == second.command_id
        for sequence in range(3, 7):
            queued, duplicate = controller.submit(
                lease_id="hold",
                sequence=sequence,
                target="chassis",
                action="forward",
                camera="head",
            )
            assert not duplicate
            assert queued.sequence == sequence
        deadline = time.monotonic() + 0.2
        while len(toolkit.prepare_calls) == 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(toolkit.prepare_calls) == 1
        snapshot = controller.snapshot()
        assert snapshot["current_command"]["sequence"] == 1
        assert [item["sequence"] for item in snapshot["queue"]] == [2, 3, 4, 5, 6]
        gate.set()
        deadline = time.monotonic() + 2.0
        while len(toolkit.execute_calls) < 6 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(toolkit.prepare_calls) == 6
        assert len(toolkit.execute_calls) == 6
        assert all(call["background"] is False for call in toolkit.prepare_calls)
        assert all(
            call["predecessor_plan_id"] is None
            for call in toolkit.prepare_calls
        )
        assert all(
            command.result is not None
            and command.result.get("cancelled_before_execution") is not True
            for command in (
                controller._commands[("hold", sequence)]
                for sequence in range(1, 7)
            )
        )
    finally:
        gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_reserved_manual_queue_precedes_new_agent_waiter(tmp_path) -> None:
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
        controller.submit(
            lease_id="hold",
            sequence=2,
            target="right_arm",
            action="up",
            camera="head",
        )
        assert controller.snapshot()["queue_depth"] == 1
        manual_gate.set()
        deadline = time.monotonic() + 1.0
        while arbiter.snapshot()["owner"] != "agent" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert arbiter.snapshot()["owner"] == "agent"
        assert len(toolkit.execute_calls) == 2
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


def test_stop_clear_pending_preserves_accepted_head_and_clears_tail(tmp_path) -> None:
    state = _state(tmp_path)
    latch = BehaviorRawSuccessLatch(
        run_nonce="a" * 32,
        attempt_nonce="b" * 32,
        attempt_index=1,
    )
    arbiter = BehaviorCommandArbiter(success_latch=latch)
    action_gate = threading.Event()
    toolkit = _Toolkit(gate=action_gate)
    controller = BehaviorDashboardController(
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
        assert toolkit.started.wait(1.0)
        controller.submit(
            lease_id="paused",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
        stopped = controller.stop(
            lease_id="paused",
            stop_mode="clear_pending",
        )
        assert stopped["current_command"]["sequence"] == 1
        assert stopped["queue"] == []
        assert stopped["pending_cleared_count"] == 1
        action_gate.set()
        _wait_phase(controller, "completed")
        assert len(toolkit.execute_calls) == 1
        assert state.run_detail()["timeline"][-1]["status"] == "failed"
    finally:
        action_gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_agent_owner_is_mirrored_to_state_and_terminal_precedes_release(
    tmp_path,
) -> None:
    class _RecordingState(State):
        def __init__(self, **kwargs) -> None:
            self.control_updates: list[dict[str, Any]] = []
            super().__init__(**kwargs)

        def update_control_snapshot(self, snapshot, *, controller) -> bool:
            self.control_updates.append(dict(snapshot))
            return super().update_control_snapshot(
                snapshot,
                controller=controller,
            )

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


def test_capture_error_does_not_change_motion_terminal(tmp_path) -> None:
    class _CaptureError(_Toolkit):
        def dashboard_capture_views(self, *, command_id: str) -> dict[str, Any]:
            self.capture_calls.append(command_id)
            return {"capture_error": "Camera refresh failed"}

    state = _state(tmp_path)
    controller, _, toolkit = _controller(
        state, toolkit=_CaptureError()
    )
    before = state.snapshot()["frame_revisions"]
    try:
        controller.submit(
            lease_id="capture-error",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        _wait_phase(controller, "completed")
        deadline = time.monotonic() + 2.0
        while (
            controller.snapshot()["capture"]["phase"] != "failed"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        snapshot = controller.snapshot()
        assert snapshot["last_terminal"]["phase"] == "completed"
        assert snapshot["capture"]["error"] == "Camera refresh failed"
        assert snapshot["available"] is True
        detail = state.run_detail()
        assert detail["progress"]["official_task_success"] is False
        assert detail["frame_revisions"] == before
        assert detail["timeline"][-1]["status"] == "completed"
        assert len(toolkit.capture_calls) == 1
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
            self,
            *,
            target: str,
            action: str,
            camera: str,
            permit_command_id: str,
        ) -> dict[str, Any]:
            result = super().dashboard_manual_command(
                target=target,
                action=action,
                camera=camera,
                permit_command_id=permit_command_id,
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
        assert (
            snapshot["last_terminal"]["result"]["worker_exception_type"]
            == "RuntimeError"
        )
        assert controller.drain(1.0)
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
        _wait_phase(controller, "completed")
        assert state.result_calls == 1
        assert arbiter.snapshot()["owner"] is None
        assert controller.drain(1.0)
        snapshot = controller.snapshot()
        assert "state publish boom" in (
            snapshot["last_terminal"]["result"]["state_publish_error"]
        )
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_head_plus_five_pending_capacity_and_duplicate_at_capacity(tmp_path) -> None:
    state = _state(tmp_path)
    gate = threading.Event()
    controller, _, _ = _controller(state, toolkit=_Toolkit(gate=gate))
    try:
        first, _ = controller.submit(
            lease_id="capacity",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        for sequence in range(2, 7):
            controller.submit(
                lease_id="capacity",
                sequence=sequence,
                target="chassis",
                action="forward",
                camera="head",
            )
        snapshot = controller.snapshot()
        assert snapshot["current_command"]["command_id"] == first.command_id
        assert snapshot["queue_depth"] == 5
        duplicate, deduplicated = controller.submit(
            lease_id="capacity",
            sequence=6,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert deduplicated and duplicate.sequence == 6
        with pytest.raises(ControlRequestError) as caught:
            controller.submit(
                lease_id="capacity",
                sequence=7,
                target="chassis",
                action="forward",
                camera="head",
            )
        assert caught.value.code == "queue_full"
        controller.stop(lease_id="capacity", stop_mode="clear_pending")
    finally:
        gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_pending_capture_is_discarded_by_next_motion(tmp_path) -> None:
    state = _state(tmp_path)
    controller, _, toolkit = _controller(state)
    try:
        controller.submit(
            lease_id="repeat",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        _wait_phase(controller, "completed")
        assert controller.snapshot()["capture"]["phase"] == "pending"
        controller.submit(
            lease_id="repeat",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert controller.snapshot()["capture"]["phase"] == "discarded"
        assert toolkit.capture_calls == []
    finally:
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_started_capture_finishes_before_new_motion_executes(tmp_path) -> None:
    capture_gate = threading.Event()
    capture_started = threading.Event()

    class _CaptureGate(_Toolkit):
        def dashboard_capture_views(self, *, command_id: str) -> dict[str, Any]:
            capture_started.set()
            assert capture_gate.wait(2.0)
            return super().dashboard_capture_views(command_id=command_id)

    state = _state(tmp_path)
    controller, _, toolkit = _controller(state, toolkit=_CaptureGate())
    try:
        controller.submit(
            lease_id="serial",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert capture_started.wait(1.0)
        controller.submit(
            lease_id="serial",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
        time.sleep(0.05)
        assert len(toolkit.execute_calls) == 1
        capture_gate.set()
        deadline = time.monotonic() + 2.0
        while len(toolkit.execute_calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(toolkit.execute_calls) == 2
    finally:
        capture_gate.set()
        assert controller.close(2.0)
        state.unbind_controller(controller)


def test_raw_success_clears_tail_and_never_starts_capture(tmp_path) -> None:
    action_gate = threading.Event()

    class _RawSuccess(_Toolkit):
        def dashboard_execute_prepared_command(
            self,
            *,
            plan_id: str,
            command_id: str,
        ) -> dict[str, Any]:
            result = super().dashboard_execute_prepared_command(
                plan_id=plan_id,
                command_id=command_id,
            )
            result.update(
                {
                    "task_success": True,
                    "official_success_receipt": _receipt(),
                    "stop_reason": "official_task_success",
                }
            )
            return result

    state = _state(tmp_path)
    toolkit = _RawSuccess(gate=action_gate)
    controller, _, toolkit = _controller(state, toolkit=toolkit)
    try:
        controller.submit(
            lease_id="raw-success",
            sequence=1,
            target="chassis",
            action="forward",
            camera="head",
        )
        assert toolkit.started.wait(1.0)
        controller.submit(
            lease_id="raw-success",
            sequence=2,
            target="chassis",
            action="forward",
            camera="head",
        )
        action_gate.set()
        deadline = time.monotonic() + 2.0
        while (
            not controller.snapshot()["success_latched"]
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        snapshot = controller.snapshot()
        assert snapshot["success_latched"] is True
        assert snapshot["last_terminal"]["sequence"] == 1
        assert snapshot["last_terminal"]["phase"] == "completed"
        assert snapshot["queue"] == []
        assert snapshot["pending_cleared_count"] == 1
        assert toolkit.capture_calls == []
    finally:
        action_gate.set()
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
        assert response.json()["phase"] == "accepted"
        assert (
            response.json()["current_command"]["command_id"]
            == response.json()["command_id"]
        )
        assert response.json()["queue"] == []
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
    assert client.get(
        "/api/run/control/state", params={"run": unbound.run_id}
    ).status_code == 410


def test_raw_success_command_terminal_remains_readable_after_unbind(tmp_path) -> None:
    testclient = pytest.importorskip("fastapi.testclient")

    class _SuccessfulToolkit(_Toolkit):
        def dashboard_execute_prepared_command(
            self,
            *,
            plan_id: str,
            command_id: str,
        ) -> dict[str, Any]:
            result = super().dashboard_execute_prepared_command(
                plan_id=plan_id,
                command_id=command_id,
            )
            result.update(
                {
                    "task_success": True,
                    "official_success_receipt": _receipt(),
                    "stop_reason": "official_task_success",
                }
            )
            return result

    state = _state(tmp_path)
    controller, _, toolkit = _controller(
        state,
        toolkit=_SuccessfulToolkit(),
    )
    server = DashboardServer()
    server.register(state)
    client = testclient.TestClient(server._app)
    payload = {
        "run": state.run_id,
        "lease_id": "success-route",
        "sequence": 1,
        "target": "chassis",
        "action": "forward",
        "camera": "head",
    }

    accepted = client.post("/api/run/control/command", json=payload)
    assert accepted.status_code == 202
    command_id = accepted.json()["command_id"]
    deadline = time.monotonic() + 2.0
    terminal = None
    while time.monotonic() < deadline:
        response = client.get(
            "/api/run/control/state",
            params={"run": state.run_id},
        )
        if (
            response.status_code == 200
            and response.json().get("command_id") == command_id
            and response.json().get("phase") in {"completed", "failed", "cancelled"}
        ):
            terminal = response.json()
            break
        time.sleep(0.005)
    assert terminal is not None
    assert terminal["phase"] == "completed"
    assert terminal["success_latched"] is True
    assert toolkit.permit_command_ids == [command_id]
    assert toolkit.capture_calls == []

    rejected = client.post(
        "/api/run/control/command",
        json={**payload, "lease_id": "after-success"},
    )
    assert rejected.status_code == 410

    state.unbind_controller(controller)
    assert controller.close(2.0)
    persisted = client.get(
        "/api/run/control/state",
        params={"run": state.run_id},
    )
    assert persisted.status_code == 200
    assert persisted.json()["command_id"] == command_id
    assert persisted.json()["phase"] == "completed"
