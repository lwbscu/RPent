from __future__ import annotations

# This is the closed acceptance matrix for the BEHAVIOR
# joint-limits-and-goal-only execution mode.
# Do not add new collision, contact, attachment, tracking,
# pose-error, isolation, settling, or safety-gate tests
# without explicit user authorization.
import threading
import time
from typing import Any, Callable

from robots.behavior.dashboard_control import (
    BehaviorCommandArbiter,
    BehaviorDashboardController,
    BehaviorRawSuccessLatch,
)
from robots.behavior.dashboard_state import State
from robots.behavior.schemas import (
    DASHBOARD_HOLD_ARM_DELAY_S,
    DASHBOARD_PREDICTED_PLAN_DEPTH,
)


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


class _PipelineToolkit:
    def __init__(
        self,
        *,
        planning_failure: dict[str, Any] | None = None,
        prepare_gate: threading.Event | None = None,
        background_prepare_gate: threading.Event | None = None,
        execute_gate: threading.Event | None = None,
        execute_result: dict[str, Any] | None = None,
    ) -> None:
        self.planning_failure = planning_failure
        self.prepare_gate = prepare_gate
        self.background_prepare_gate = background_prepare_gate
        self.execute_gate = execute_gate
        self.execute_result = execute_result
        self.prepare_calls: list[dict[str, Any]] = []
        self.execute_calls: list[tuple[str, str]] = []
        self.discard_calls: list[str] = []
        self.capture_calls: list[str] = []
        self.prepare_started = threading.Event()
        self.background_prepare_started = threading.Event()
        self._counter = 0

    @staticmethod
    def dashboard_control_capabilities() -> dict[str, Any]:
        return {
            "motion_available": True,
            "observe_available": True,
            "threaded_predicted_planning": True,
        }

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
        permit_command_id: str,
    ) -> dict[str, Any]:
        del target, action, camera, permit_command_id
        return {"primitive_success": True}

    def dashboard_prepare_manual_command(
        self,
        *,
        target: str,
        action: str,
        predecessor_plan_id: str | None,
        permit_command_id: str,
        background: bool,
        planning_only_probe: bool = False,
    ) -> dict[str, Any]:
        del planning_only_probe
        self.prepare_calls.append(
            {
                "target": target,
                "action": action,
                "predecessor_plan_id": predecessor_plan_id,
                "permit_command_id": permit_command_id,
                "background": background,
            }
        )
        self.prepare_started.set()
        if background:
            self.background_prepare_started.set()
        if background and self.background_prepare_gate is not None:
            assert self.background_prepare_gate.wait(2.0)
        elif self.prepare_gate is not None:
            assert self.prepare_gate.wait(2.0)
        if self.planning_failure is not None:
            return dict(self.planning_failure)
        self._counter += 1
        plan_id = f"plan-{self._counter}"
        terminal = {
            "joint_positions": [float(self._counter)],
            "base_xyyaw": [float(self._counter), 0.0, 0.0],
            "eef_by_hand": {},
            "torso_link4": None,
        }
        return {
            "status": "prepared",
            "plan_id": plan_id,
            "predecessor_plan_id": predecessor_plan_id,
            "predicted_start_digest": f"start-{self._counter}",
            "predicted_terminal": terminal,
            "planning_profile": "fast_trajopt",
            "fast_solver_deadline_s": 1.0,
        }

    def dashboard_execute_prepared_command(
        self,
        *,
        plan_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        self.execute_calls.append((plan_id, command_id))
        if self.execute_gate is not None:
            assert self.execute_gate.wait(2.0)
        if self.execute_result is not None:
            return dict(self.execute_result)
        return {
            "primitive_success": True,
            "task_success": False,
            "stop_reason": "reached",
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
        index = len(self.capture_calls)
        return {
            "_frames_bytes": {
                "head": f"head-{index}".encode(),
                "left_wrist": f"left-{index}".encode(),
                "right_wrist": f"right-{index}".encode(),
            },
            "capture_group_id": f"capture-{index}",
            "simulator_step": index,
        }


def _controller(tmp_path, toolkit: _PipelineToolkit):
    state = _state(tmp_path)
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
        lease_timeout_s=3.0,
    )
    controller.bind_toolkit(toolkit)
    state.bind_controller(controller)
    controller.activate()
    return controller


def _wait_for(
    controller: BehaviorDashboardController,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_s: float = 3.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    snapshot = controller.snapshot()
    while not predicate(snapshot) and time.monotonic() < deadline:
        time.sleep(0.005)
        snapshot = controller.snapshot()
    assert predicate(snapshot), snapshot
    return snapshot


def _submit(
    controller: BehaviorDashboardController,
    *,
    sequence: int,
    lease_id: str = "hold",
):
    command, duplicate = controller.submit(
        lease_id=lease_id,
        sequence=sequence,
        target="chassis",
        action="forward",
        camera="head",
    )
    assert duplicate is False
    return command


def _stop_and_close(controller: BehaviorDashboardController, lease_id="hold"):
    snapshot = controller.snapshot()
    if snapshot["lease_status"] == "active":
        controller.stop(lease_id=lease_id)
    _wait_for(
        controller,
        lambda value: value["predicted_plan_depth"] == 0
        and value["prediction_planning"] is None,
    )
    assert controller.close(timeout_s=2.0)


def test_prediction_pipeline_reaches_depth_twenty_after_hold_arm_delay(tmp_path):
    toolkit = _PipelineToolkit()
    controller = _controller(tmp_path, toolkit)
    started = time.monotonic()
    try:
        _submit(controller, sequence=1)
        _wait_for(
            controller,
            lambda value: value["predicted_plan_depth"]
            == DASHBOARD_PREDICTED_PLAN_DEPTH,
        )

        snapshot = controller.snapshot()
        assert snapshot["hold_armed"] is True
        assert snapshot["predicted_plan_capacity"] == 20
        assert time.monotonic() - started >= DASHBOARD_HOLD_ARM_DELAY_S
        assert len(snapshot["predicted_plans"]) == 20
        assert all(
            plan["predecessor_plan_id"]
            for plan in snapshot["predicted_plans"]
        )
    finally:
        _stop_and_close(controller)


def test_consuming_one_predicted_plan_refills_the_queue(tmp_path):
    toolkit = _PipelineToolkit()
    controller = _controller(tmp_path, toolkit)
    try:
        first = _submit(controller, sequence=1)
        _wait_for(
            controller,
            lambda value: value["predicted_plan_depth"] == 20,
        )
        prepares_before = len(toolkit.prepare_calls)

        second = _submit(controller, sequence=2)
        assert second.plan_id is not None
        assert second.plan_id != first.plan_id
        _wait_for(
            controller,
            lambda value: len(toolkit.execute_calls) >= 2
            and value["predicted_plan_depth"] == 20,
        )

        assert len(toolkit.prepare_calls) >= prepares_before + 1
        assert toolkit.execute_calls[1][0] == second.plan_id
    finally:
        _stop_and_close(controller)


def test_pointer_stop_clears_all_unexecuted_predictions_and_starts_no_new_plan(
    tmp_path,
):
    toolkit = _PipelineToolkit()
    controller = _controller(tmp_path, toolkit)
    try:
        _submit(controller, sequence=1)
        _wait_for(
            controller,
            lambda value: value["predicted_plan_depth"] == 20,
        )
        controller.stop(lease_id="hold", reason="pointerup")
        stopped = _wait_for(
            controller,
            lambda value: value["predicted_plan_depth"] == 0
            and value["prediction_planning"] is None,
        )
        prepare_count = len(toolkit.prepare_calls)
        time.sleep(0.05)

        assert stopped["lease_status"] == "stopped"
        assert controller.snapshot()["predicted_plan_depth"] == 0
        assert len(toolkit.prepare_calls) == prepare_count
    finally:
        assert controller.close(timeout_s=2.0)


def test_short_press_captures_once_after_stop(tmp_path):
    toolkit = _PipelineToolkit()
    controller = _controller(tmp_path, toolkit)
    try:
        _submit(controller, sequence=1, lease_id="short")
        _wait_for(
            controller,
            lambda value: value["last_terminal"] is not None,
        )
        assert toolkit.capture_calls == []

        controller.stop(lease_id="short", reason="pointerup")
        _wait_for(
            controller,
            lambda value: value["capture"]["phase"] == "completed",
        )
        assert len(toolkit.capture_calls) == 1
    finally:
        assert controller.close(timeout_s=2.0)


def test_long_hold_does_not_capture_between_steps_and_captures_once_on_stop(
    tmp_path,
):
    toolkit = _PipelineToolkit()
    controller = _controller(tmp_path, toolkit)
    try:
        _submit(controller, sequence=1)
        _wait_for(
            controller,
            lambda value: value["predicted_plan_depth"] == 20,
        )
        _submit(controller, sequence=2)
        _wait_for(
            controller,
            lambda value: len(toolkit.execute_calls) >= 2,
        )
        assert toolkit.capture_calls == []

        controller.stop(lease_id="hold", reason="pointerup")
        _wait_for(
            controller,
            lambda value: value["capture"]["phase"] == "completed",
        )
        assert len(toolkit.capture_calls) == 1
    finally:
        assert controller.close(timeout_s=2.0)


def test_stop_during_planning_preserves_head_executes_once_and_captures_once(
    tmp_path,
):
    prepare_gate = threading.Event()
    toolkit = _PipelineToolkit(prepare_gate=prepare_gate)
    controller = _controller(tmp_path, toolkit)
    try:
        head = _submit(controller, sequence=1, lease_id="short-planning")
        assert toolkit.prepare_started.wait(1.0)
        planning = _wait_for(
            controller,
            lambda value: value["planning_command"] is not None,
        )

        stopped = controller.stop(
            lease_id="short-planning",
            reason="pointerup",
            stop_mode="clear_pending",
        )
        assert stopped["lease_status"] == "stopped"
        assert stopped["current_command"]["command_id"] == head.command_id
        assert stopped["planning_command"]["command_id"] == head.command_id
        assert stopped["queue_depth"] == 0
        assert stopped["predicted_plan_depth"] == 0
        assert planning["current_command"]["command_id"] == head.command_id

        prepare_gate.set()
        terminal = _wait_for(
            controller,
            lambda value: value["last_terminal"] is not None
            and value["capture"]["phase"] == "completed",
        )
        assert terminal["last_terminal"]["command_id"] == head.command_id
        assert terminal["last_terminal"]["phase"] == "completed"
        assert len(toolkit.execute_calls) == 1
        assert toolkit.execute_calls[0][1] == head.command_id
        assert len(toolkit.capture_calls) == 1
    finally:
        prepare_gate.set()
        assert controller.close(timeout_s=2.0)


def test_stop_during_repeat_planning_cancels_seq_two_and_discards_late_plan(
    tmp_path,
):
    toolkit = _PipelineToolkit()
    controller = _controller(tmp_path, toolkit)
    repeat_prepare_gate = threading.Event()
    try:
        first = _submit(controller, sequence=1)
        _wait_for(
            controller,
            lambda value: value["last_terminal"] is not None
            and value["last_terminal"]["command_id"] == first.command_id,
        )

        toolkit.prepare_started.clear()
        toolkit.prepare_gate = repeat_prepare_gate
        repeat = _submit(controller, sequence=2)
        assert toolkit.prepare_started.wait(1.0)
        _wait_for(
            controller,
            lambda value: value["planning_command"] is not None
            and value["planning_command"]["command_id"] == repeat.command_id,
        )

        stopped = controller.stop(
            lease_id="hold",
            reason="pointerup",
            stop_mode="clear_pending",
        )
        assert stopped["current_command"] is None
        assert stopped["queue_depth"] == 0
        assert stopped["last_terminal"]["command_id"] == repeat.command_id
        assert stopped["last_terminal"]["phase"] == "cancelled"

        repeat_prepare_gate.set()
        terminal = _wait_for(
            controller,
            lambda value: value["planning_command"] is None
            and value["prediction_planning"] is None
            and value["capture"]["phase"] == "completed",
        )
        assert terminal["last_terminal"]["command_id"] == repeat.command_id
        assert len(toolkit.execute_calls) == 1
        assert toolkit.execute_calls[0][1] == first.command_id
        assert "plan-2" in toolkit.discard_calls
        assert len(toolkit.capture_calls) == 1
    finally:
        repeat_prepare_gate.set()
        assert controller.close(timeout_s=2.0)


def test_stop_keeps_moving_head_but_clears_seq_two_plus_and_predictions(tmp_path):
    execute_gate = threading.Event()
    toolkit = _PipelineToolkit(execute_gate=execute_gate)
    controller = _controller(tmp_path, toolkit)
    try:
        head = _submit(controller, sequence=1)
        _wait_for(
            controller,
            lambda value: len(toolkit.execute_calls) == 1
            and value["predicted_plan_depth"]
            == DASHBOARD_PREDICTED_PLAN_DEPTH,
        )
        second = _submit(controller, sequence=2)
        third = _submit(controller, sequence=3)
        assert second.phase == "prepared"
        assert third.phase == "prepared"
        assert second.plan_id is not None
        assert third.plan_id is not None

        stopped = controller.stop(
            lease_id="hold",
            reason="pointerup",
            stop_mode="clear_pending",
        )
        assert stopped["current_command"]["command_id"] == head.command_id
        assert stopped["current_command"]["phase"] == "moving"
        assert stopped["queue_depth"] == 0
        assert stopped["predicted_plan_depth"] == 0
        assert stopped["pending_cleared_count"] >= 2

        drained_predictions = _wait_for(
            controller,
            lambda value: value["prediction_planning"] is None
            and value["predicted_plan_depth"] == 0,
        )
        assert drained_predictions["current_command"]["command_id"] == head.command_id
        _wait_for(
            controller,
            lambda _value: second.plan_id in toolkit.discard_calls
            and third.plan_id in toolkit.discard_calls,
        )
        execute_gate.set()
        terminal = _wait_for(
            controller,
            lambda value: value["last_terminal"] is not None
            and value["capture"]["phase"] == "completed",
        )

        assert terminal["last_terminal"]["command_id"] == head.command_id
        assert len(toolkit.execute_calls) == 1
        assert second.command_id != head.command_id
        assert third.command_id != head.command_id
        assert len(toolkit.capture_calls) == 1
    finally:
        execute_gate.set()
        assert controller.close(timeout_s=2.0)


def test_stop_capture_is_not_repeated_when_stale_prediction_finishes_late(tmp_path):
    execute_gate = threading.Event()
    background_prepare_gate = threading.Event()
    toolkit = _PipelineToolkit(
        background_prepare_gate=background_prepare_gate,
        execute_gate=execute_gate,
    )
    controller = _controller(tmp_path, toolkit)
    try:
        _submit(controller, sequence=1)
        assert toolkit.background_prepare_started.wait(1.0)
        controller.stop(
            lease_id="hold",
            reason="pointerup",
            stop_mode="clear_pending",
        )

        execute_gate.set()
        _wait_for(
            controller,
            lambda value: value["capture"]["phase"] == "completed",
        )
        assert len(toolkit.capture_calls) == 1

        background_prepare_gate.set()
        _wait_for(
            controller,
            lambda value: value["prediction_planning"] is None,
        )
        time.sleep(0.05)
        assert len(toolkit.capture_calls) == 1
    finally:
        execute_gate.set()
        background_prepare_gate.set()
        assert controller.close(timeout_s=2.0)


def test_raw_success_from_stop_preserved_head_preempts_pointerup_and_extra_capture(
    tmp_path,
):
    prepare_gate = threading.Event()
    toolkit = _PipelineToolkit(
        prepare_gate=prepare_gate,
        execute_result={
            "primitive_success": True,
            "info_done": {"success": True},
            "terminal_capture": {
                "_frames_bytes": {
                    "head": b"success-head",
                    "left_wrist": b"success-left",
                    "right_wrist": b"success-right",
                },
                "capture_group_id": "success-capture",
                "simulator_step": 17,
            },
        },
    )
    controller = _controller(tmp_path, toolkit)
    try:
        head = _submit(controller, sequence=1, lease_id="success-after-stop")
        assert toolkit.prepare_started.wait(1.0)
        controller.stop(
            lease_id="success-after-stop",
            reason="pointerup",
            stop_mode="clear_pending",
        )

        prepare_gate.set()
        terminal = _wait_for(
            controller,
            lambda value: value["success_latched"] is True
            and value["last_terminal"] is not None
            and value["capture"]["phase"] == "completed",
        )
        assert terminal["last_terminal"]["command_id"] == head.command_id
        assert terminal["last_terminal"]["task_success"] is True
        assert terminal["stop_reason"] == "official_task_success"
        assert terminal["lease_status"] == "succeeded"
        assert len(toolkit.execute_calls) == 1
        assert toolkit.capture_calls == []
    finally:
        prepare_gate.set()
        assert controller.close(timeout_s=2.0)


def test_one_second_curobo_timeout_stops_lease_without_execution(tmp_path):
    toolkit = _PipelineToolkit(
        planning_failure={
            "status": "failed",
            "stop_reason": "timeout",
            "error": "CuRobo solver timed out after 1.0 s",
        }
    )
    controller = _controller(tmp_path, toolkit)
    try:
        _submit(controller, sequence=1, lease_id="timeout")
        terminal = _wait_for(
            controller,
            lambda value: value["phase"] == "failed",
        )

        assert terminal["stop_reason"] == "timeout"
        assert terminal["lease_status"] == "stopped"
        assert terminal["predicted_plan_depth"] == 0
        assert toolkit.execute_calls == []
    finally:
        assert controller.close(timeout_s=2.0)
