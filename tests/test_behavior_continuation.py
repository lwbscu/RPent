from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from robots.behavior.continuation import run_behavior_planner_continuation
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
)
from robots.behavior.toolkit import BehaviorToolkit
from rpent.planner.base import PlannerResult


class _Toolkit:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0
        self.identity = ("run-nonce", "attempt-nonce", "attached-radio")

    def runner_continuation_state(self):
        index = min(self.calls, len(self.states) - 1)
        return dict(self.states[index])


class _Planner:
    def __init__(self, tmp_path: Path, toolkit: _Toolkit):
        self.tmp_path = tmp_path
        self.toolkit = toolkit
        self.calls = []

    def solve(self, *, system_prompt, user_message, toolkit, max_turns):
        assert toolkit is self.toolkit
        assert toolkit.identity == (
            "run-nonce",
            "attempt-nonce",
            "attached-radio",
        )
        cycle = len(self.calls) + 1
        self.calls.append((system_prompt, user_message, max_turns))
        output = self.tmp_path / "codex.txt"
        stream = self.tmp_path / "codex.txt.stream.jsonl"
        last = self.tmp_path / "codex.txt.last"
        output.write_text(f"cycle {cycle} ordinary final\n", encoding="utf-8")
        stream.write_text(json.dumps({"cycle": cycle}) + "\n", encoding="utf-8")
        last.write_text(f"ordinary final {cycle}\n", encoding="utf-8")
        self.toolkit.calls = cycle
        return PlannerResult(
            messages=[{"role": "codex_sdk", "content": f"cycle {cycle}"}],
            stats={
                "backend": "codex_sdk",
                "turns_used": 0,
                "tool_calls": cycle,
                "total_input_tokens": 10 * cycle,
                "output_path": str(output),
                "raw_stream_path": str(stream),
                "last_message_path": str(last),
            },
        )


def _state(**updates):
    state = {
        "raw_official_success_verified": False,
        "visual_terminal_failure_verified": False,
        "visual_terminal_failure_reason": None,
        "unrecoverable_infrastructure_termination": False,
        "exhausted_budgets": [],
    }
    state.update(updates)
    return state


def test_toolkit_state_reports_pi0_chunks_as_count_not_execution_budget():
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._primitives = SimpleNamespace(
        elapsed_wall_clock_s=10.0,
        max_wall_clock_s=20.0,
        max_episode_steps=100,
        total_env_steps=50,
        _global_vla_chunks=8,
        _vla_invocations=3,
        attempt_index=3,
        attempt_nonce="attempt",
        run_nonce="run",
    )
    toolkit._tool_calls = 2
    toolkit._max_tool_calls = 10
    toolkit._terminal_failure_latched = False
    toolkit._task_spec = TURNING_ON_RADIO_TASK_SPEC
    toolkit._tool_trace = [
        {
            "result": {
                "primitive_success": False,
                "stop_reason": "unreachable",
                "error": "ordinary failure",
            }
        }
    ]
    toolkit._has_verified_raw_success = lambda: False

    state = toolkit.runner_continuation_state()

    assert state["raw_official_success_verified"] is False
    assert state["visual_terminal_failure_verified"] is False
    assert state["visual_terminal_failure_reason"] is None
    assert state["unrecoverable_infrastructure_termination"] is False
    assert state["exhausted_budgets"] == []
    assert state["global_vla_chunks"] == 8
    assert state["global_vla_invocations"] == 3


def test_natural_language_final_starts_fresh_cycle_and_preserves_runtime(tmp_path):
    toolkit = _Toolkit(
        [
            _state(),
            _state(),
            _state(raw_official_success_verified=True),
        ]
    )
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=5,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 2
    assert planner.calls[0][2] == 5
    assert planner.calls[1][2] == 4
    assert planner.calls[0][1] == "task"
    assert "fresh public `observe`" in planner.calls[1][1]
    assert planner.calls[1][1].startswith("task\n\n")
    assert "cycle 1 ordinary final" not in planner.calls[1][1]
    assert toolkit.identity == (
        "run-nonce",
        "attempt-nonce",
        "attached-radio",
    )
    assert result.finish_result["runner_termination_reason"] == (
        "official_task_success"
    )
    assert result.finish_result["task_success"] is True
    assert result.stats["turns_used"] == 2
    assert result.stats["total_input_tokens"] == 30
    assert [message["planner_cycle"] for message in result.messages] == [1, 2]
    assert (tmp_path / "planner_cycles" / "cycle_001" / "output.txt").read_text(
        encoding="utf-8"
    ) == "cycle 1 ordinary final\n"
    assert (tmp_path / "planner_cycles" / "cycle_002" / "output.txt").read_text(
        encoding="utf-8"
    ) == "cycle 2 ordinary final\n"


def test_legacy_pi0_chunk_limit_cannot_end_continuation(tmp_path):
    toolkit = _Toolkit(
        [
            _state(
                exhausted_budgets=[
                    "max_total_vla_chunks",
                    "global_vla_chunk_budget_exhausted",
                ]
            ),
            _state(raw_official_success_verified=True),
        ]
    )
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=2,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 1
    assert result.finish_result["runner_termination_reason"] == (
        "official_task_success"
    )
    assert result.finish_result["exhausted_budgets"] == []


def test_requested_chunk_completion_cannot_end_continuation_as_a_budget(tmp_path):
    toolkit = _Toolkit(
        [
            _state(exhausted_budgets=["requested_chunks_complete"]),
            _state(raw_official_success_verified=True),
        ]
    )
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=2,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 1
    assert result.finish_result["runner_termination_reason"] == (
        "official_task_success"
    )
    assert result.finish_result["exhausted_budgets"] == []


def test_turn_budget_is_cumulative_and_each_cycle_costs_at_least_one(tmp_path):
    toolkit = _Toolkit([_state()])
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=2,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 2
    assert [call[2] for call in planner.calls] == [2, 1]
    assert result.finish_result["runner_termination_reason"] == (
        "hard_execution_budget_exhausted"
    )
    assert result.finish_result["task_success"] is False
    assert result.finish_result["exhausted_budgets"] == ["max_turns"]
    assert result.error is None


def test_ordinary_planner_error_does_not_end_episode(tmp_path):
    toolkit = _Toolkit([_state(), _state(), _state(raw_official_success_verified=True)])

    class Planner(_Planner):
        attempts = 0

        def solve(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                self.calls.append(
                    ("failed", kwargs["user_message"], kwargs["max_turns"])
                )
                self.toolkit.calls = 1
                raise RuntimeError("ordinary SDK failure")
            result = super().solve(**kwargs)
            self.toolkit.calls = 2
            return result

    planner = Planner(tmp_path, toolkit)
    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=3,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 2
    assert result.finish_result["runner_termination_reason"] == (
        "official_task_success"
    )
    assert result.error is None
    assert result.stats["cycle_errors"][0]["cycle_index"] == 1


def test_structured_visual_terminal_failure_ends_without_another_cycle(tmp_path):
    toolkit = _Toolkit(
        [
            _state(),
            _state(
                visual_terminal_failure_verified=True,
                visual_terminal_failure_reason="visual_radio_tipped_flat",
            ),
        ]
    )
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=5,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 1
    assert result.finish_result["runner_termination_reason"] == (
        "visual_radio_tipped_flat"
    )
    assert result.finish_result["task_success"] is False


def test_trash_ignores_unregistered_radio_visual_terminal_state(tmp_path):
    toolkit = _Toolkit(
        [
            _state(),
            _state(
                visual_terminal_failure_verified=True,
                visual_terminal_failure_reason="visual_radio_tipped_flat",
            ),
            _state(raw_official_success_verified=True),
        ]
    )
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name=PICKING_UP_TRASH_TASK_SPEC.task_name,
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=5,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 2
    assert result.finish_result["runner_termination_reason"] == "official_task_success"
    assert result.finish_result["task_success"] is True
    assert result.finish_result["task_name"] == "picking_up_trash"


def test_structured_runtime_termination_is_an_error(tmp_path):
    toolkit = _Toolkit(
        [_state(), _state(unrecoverable_infrastructure_termination=True)]
    )
    planner = _Planner(tmp_path, toolkit)

    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=5,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 1
    assert result.finish_result["runner_termination_reason"] == (
        "unrecoverable_infrastructure_termination"
    )
    assert result.finish_result["task_success"] is None
    assert "unrecoverable infrastructure termination" in result.error


def test_operator_interrupt_is_terminal_unknown_not_task_failure(tmp_path):
    toolkit = _Toolkit([_state()])

    class Planner(_Planner):
        def solve(self, **kwargs):
            self.calls.append(
                (kwargs["system_prompt"], kwargs["user_message"], kwargs["max_turns"])
            )
            raise KeyboardInterrupt

    planner = Planner(tmp_path, toolkit)
    result = run_behavior_planner_continuation(
        task_name="turning_on_radio",
        planner=planner,
        system_prompt="system",
        initial_user_message="task",
        toolkit=toolkit,
        max_turns=5,
        output_dir=tmp_path,
    )

    assert len(planner.calls) == 1
    assert result.finish_result["runner_termination_reason"] == "operator_stop"
    assert result.finish_result["task_success"] is None
    assert result.error is None
