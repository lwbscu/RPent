import json

import numpy as np
import pytest

from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from rpent.tools import common


def _observation():
    return {
        "states": np.arange(256, dtype=np.float32),
        "task_descriptions": "turn on the radio",
    }


class _Model:
    def __init__(self, actions):
        self.actions = actions
        self.calls = 0

    def predict_action_batch(self, obs, *, mode):
        self.calls += 1
        assert obs["task_descriptions"] == "turn on the radio"
        assert mode == "eval"
        return self.actions, {"model_call": self.calls}


class _Env:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.reset_calls = 0
        self.step_calls = []

    def reset(self):
        self.reset_calls += 1
        return _observation(), {"done": {"success": False}}

    def chunk_step(self, actions):
        self.step_calls.append(np.asarray(actions).copy())
        return self.outcomes.pop(0)


def test_run_full_task_tool_is_called_once_and_stops_on_official_success(tmp_path):
    env = _Env(
        [
            (
                _observation(),
                1.0,
                False,
                False,
                {
                    "done": {"success": True},
                    "_rpent": {"executed_steps": 1},
                },
            )
        ]
    )
    model = _Model(np.zeros((2, 23), dtype=np.float32))
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "env": env,
            "model": model,
            "max_episode_steps": 20,
            "action_horizon": 2,
            "output_dir": tmp_path,
        }
    )

    tool_result = toolkit.execute_tool("run_full_task", {})

    assert tool_result.is_finish is True
    assert tool_result.result["success"] is True
    assert tool_result.result["task_success"] is True
    assert tool_result.result["stop_reason"] == "task_success"
    assert tool_result.result["env_steps_used"] == 1
    assert model.calls == 1
    assert len(env.step_calls) == 1
    assert json.loads((tmp_path / "final_result.json").read_text()) == tool_result.result
    assert json.loads((tmp_path / "raw_final_info.json").read_text())["done"]["success"] is True
    assert tool_result.result["action_trace_path"].endswith(
        "behavior_action_trace.jsonl"
    )


def test_run_full_task_short_chunks_continue_until_env_step_horizon(tmp_path):
    outcomes = [
        (
            _observation(),
            0.0,
            False,
            False,
            {"done": {"success": False}, "_rpent": {"executed_steps": 1}},
        )
        for _ in range(5)
    ]
    env = _Env(outcomes)
    model = _Model(np.zeros((1, 23), dtype=np.float32))
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=5,
        action_horizon=4,
        output_dir=tmp_path,
    )

    result = primitives.run_full_task()

    assert result["stop_reason"] == "horizon"
    assert result["env_steps_used"] == 5
    assert result["chunks_used"] == 5
    assert model.calls == 5


@pytest.mark.parametrize("executed_steps", [False, -1, 0, 2])
def test_run_full_task_rejects_invalid_executed_steps(tmp_path, executed_steps):
    env = _Env(
        [
            (
                _observation(),
                0.0,
                False,
                False,
                {
                    "done": {"success": False},
                    "_rpent": {"executed_steps": executed_steps},
                },
            )
        ]
    )
    primitives = BehaviorPrimitives(
        env=env,
        model=_Model(np.zeros((1, 23), dtype=np.float32)),
        max_episode_steps=5,
        output_dir=tmp_path,
    )

    result = primitives.run_full_task()

    assert result["stop_reason"] == "error"
    assert result["error"].startswith("RuntimeError: invalid env executed_steps")


@pytest.mark.parametrize(
    ("terminated", "truncated", "stop_reason"),
    [(True, False, "terminated"), (False, True, "truncated")],
)
def test_termination_or_truncation_without_official_done_is_not_success(
    tmp_path, terminated, truncated, stop_reason
):
    env = _Env(
        [
            (
                _observation(),
                100.0,
                terminated,
                truncated,
                {
                    "success": True,
                    "task_success": True,
                    "done": {"success": False},
                },
            )
        ]
    )
    model = _Model(np.zeros((2, 23), dtype=np.float32))
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=20,
        action_horizon=2,
        output_dir=tmp_path,
    )

    result = primitives.run_full_task()

    assert result["success"] is False
    assert result["task_success"] is False
    assert result["stop_reason"] == stop_reason
    assert result["terminated"] is terminated
    assert result["truncated"] is truncated
    assert model.calls == 1
    assert len(env.step_calls) == 1


@pytest.mark.parametrize("failure", ["shape", "nan"])
def test_malformed_action_is_rejected_before_env_step_and_writes_final_result(
    tmp_path, failure
):
    actions = np.zeros((1, 22), dtype=np.float32)
    if failure == "nan":
        actions = np.zeros((1, 23), dtype=np.float32)
        actions[0, 5] = np.nan
    env = _Env([])
    primitives = BehaviorPrimitives(
        env=env,
        model=_Model(actions),
        max_episode_steps=20,
        action_horizon=2,
        output_dir=tmp_path,
    )

    result = primitives.run_full_task()

    assert result["success"] is False
    assert result["stop_reason"] == "error"
    assert result["chunks_used"] == 0
    assert result["env_steps_used"] == 0
    assert result["error"].startswith("ValueError:")
    assert env.step_calls == []
    assert json.loads((tmp_path / "final_result.json").read_text()) == result


def test_behavior_toolkit_has_only_one_behavior_specific_tool(tmp_path):
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "env": _Env([]),
            "model": _Model(np.zeros((1, 23), dtype=np.float32)),
            "max_episode_steps": 1,
            "output_dir": tmp_path,
        }
    )
    names = [spec["name"] for spec in toolkit.get_tools_spec()]

    assert names == ["run_full_task"]
    assert not {spec["name"] for spec in common.TOOLS_SPEC}.intersection(names)
