from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import robots.behavior.env_server as env_server
from robots.behavior.env_server import (
    _CONTROLLER_PLANNER,
    _CONTROLLER_VLA,
    BehaviorEnvFacade,
)


def _prepared_result() -> dict[str, object]:
    return {
        "schema_version": 1,
        "plan_id": "plan-1",
        "target": "chassis",
        "action": "forward",
        "predecessor_plan_id": None,
        "background": False,
        "status": "prepared",
        "planning_deadline_s": 12.0,
        "deadline_enforcement": {
            "solver_timeout_enforced": True,
            "hard_wall_clock_enforced": True,
        },
    }


def _clock(*values: float):
    iterator = iter(values)
    return lambda: next(iterator)


def test_prepare_reports_fixed_json_safe_phase_boundaries(monkeypatch):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._controller_state = _CONTROLLER_VLA
    facade._base_controller_mode = "velocity"
    facade._dashboard_planning_admitted = False
    facade._official_success_latched = False
    facade._last_info = {}
    facade.dashboard_control_capabilities = lambda: {
        "motion_available": True,
        "base_available": True,
    }

    def switch_controller(target):
        assert target == _CONTROLLER_PLANNER
        facade._controller_state = _CONTROLLER_PLANNER
        facade._base_controller_mode = "position"
        return {"changed": True}

    facade._switch_controller = switch_controller
    facade._planner = SimpleNamespace(
        prepare_dashboard_motion=lambda *args, **kwargs: _prepared_result()
    )
    monkeypatch.setattr(
        env_server.time,
        "monotonic",
        _clock(0.0, 1.0, 3.0, 5.0, 9.0, 11.0, 12.0),
    )

    result = facade.dashboard_prepare_manual_command(
        target="chassis",
        action="forward",
    )

    assert result["elapsed_s"] == pytest.approx(6.0)
    assert result["latency_metrics"] == {
        "schema_version": 1,
        "clock": "time.monotonic",
        "operation": "dashboard_prepare",
        "phases_s": {
            "prepare_preflight": 3.0,
            "controller_switch": 2.0,
            "planner_prepare": 4.0,
            "prepare_postcheck": 3.0,
        },
        "controller_switch": {
            "attempted": True,
            "changed": True,
            "from_state": _CONTROLLER_VLA,
            "to_state": _CONTROLLER_PLANNER,
        },
        "total_s": 12.0,
    }
    json.dumps(result["latency_metrics"], allow_nan=False)


def _execute_facade(*, raw_success: bool, action_count: int):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._controller_state = _CONTROLLER_PLANNER
    facade._base_controller_mode = "position"
    facade._motion_in_flight = False
    facade._dashboard_execute_receipts = {}
    facade._dashboard_env_step_latency = None
    facade._done = False
    facade._env_steps = 0
    facade._meta = {"max_episode_steps": 20}
    facade._last_observation = {"cached": True}
    facade._last_info = {}
    facade._planner_video_interval_steps = 4
    facade._gripper_latch = {"left": 1.0, "right": 1.0}
    facade._official_success_latched = False
    direct_step_calls: list[bool] = []

    def step_env(_action, *, need_obs):
        direct_step_calls.append(bool(need_obs))
        return (
            None,
            [0.0],
            False,
            False,
            [{"done": {"success": raw_success}}],
        )

    facade._env = SimpleNamespace(_direct_process=SimpleNamespace(step_env=step_env))

    def latch_success(_info):
        facade._official_success_latched = True
        facade._done = True
        return {"source": 'info["done"]["success"]'}

    facade._latch_official_success = latch_success

    class Planner:
        calls = 0

        def execute_dashboard_motion(self, plan_id, command_id):
            assert (plan_id, command_id) == ("plan-1", "command-1")
            self.calls += 1
            facade._step_action_chunk(
                np.zeros((action_count, 23), dtype=np.float32),
                observe_final=False,
            )
            return {
                "primitive_success": True,
                "task_success": raw_success,
                "stop_reason": (
                    "official_task_success" if raw_success else "completed"
                ),
            }

    planner = Planner()
    facade._planner = planner
    facade._planner_public_result = lambda result: {
        **result,
        "task_success": bool(facade._official_success_latched),
    }
    facade._dashboard_capture_group = lambda: pytest.fail(
        "prepared execution must not capture after motion"
    )
    return facade, planner, direct_step_calls


def test_execute_aggregates_completed_direct_env_step_latency(monkeypatch):
    facade, _planner, direct_step_calls = _execute_facade(
        raw_success=False,
        action_count=3,
    )
    monkeypatch.setattr(
        env_server.time,
        "monotonic",
        _clock(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 10.0, 12.0, 13.0, 14.0),
    )

    result = facade.dashboard_execute_prepared_command(
        plan_id="plan-1",
        command_id="command-1",
    )

    assert direct_step_calls == [False, False, False]
    assert result["elapsed_s"] == pytest.approx(12.0)
    metrics = result["latency_metrics"]
    assert metrics["phases_s"] == {
        "execute_preflight": 1.0,
        "planner_execute": 11.0,
        "execute_postcheck": 2.0,
    }
    assert metrics["env_step_aggregate"] == {
        "boundary": "env._direct_process.step_env",
        "count": 3,
        "total_s": 6.0,
        "min_s": 1.0,
        "max_s": 3.0,
        "mean_s": 2.0,
    }
    assert metrics["total_s"] == pytest.approx(14.0)
    json.dumps(metrics, allow_nan=False)


def test_raw_success_stops_remaining_steps_and_replay_adds_no_action():
    facade, planner, direct_step_calls = _execute_facade(
        raw_success=True,
        action_count=3,
    )

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
    assert first["task_success"] is True
    assert planner.calls == 1
    assert direct_step_calls == [False]
    assert first["latency_metrics"]["env_step_aggregate"]["count"] == 1
