import json
from pathlib import Path

import numpy as np
import pytest

import robots.behavior.tools as behavior_tools
from robots.behavior.schemas import (
    FULL_TASK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOL_SPECS,
    PLANNER_TOOLS_MODE,
    PUBLIC_PRIMITIVE_ENTRYPOINTS,
)
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives

EXPECTED_PLANNER_TOOLS = (
    "observe",
    "pixel_to_world",
    "navigate_to",
    "move_to",
    "pick",
    "rotate_wrist",
    "press",
    "release",
)


def _observation(
    *,
    left_gripper: float = 0.08,
    right_gripper: float = 0.08,
    task: str = "turn on the radio",
) -> dict:
    states = np.zeros(256, dtype=np.float32)
    states[193:195] = left_gripper / 2.0
    states[232:234] = right_gripper / 2.0
    return {"states": states, "task_descriptions": task}


class _FakeModel:
    def __init__(self, *action_chunks: np.ndarray):
        self._action_chunks = list(action_chunks)
        self.observations = []

    def predict_action_batch(self, obs, *, mode):
        assert mode == "eval"
        self.observations.append(dict(obs))
        return self._action_chunks.pop(0), {"model_call": len(self.observations)}


class _FakeEnv:
    def __init__(self, *outcomes, reset_observation=None, reset_info=None):
        self._outcomes = list(outcomes)
        self._reset_observation = reset_observation
        self._reset_info = reset_info
        self.actions = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        if self._reset_observation is None:
            raise AssertionError("unexpected reset")
        return self._reset_observation, self._reset_info

    def chunk_step(self, actions):
        array = np.asarray(actions).copy()
        self.actions.append(array)
        if not self._outcomes:
            raise AssertionError("unexpected env step")
        return self._outcomes.pop(0)


class _PlannerBackend:
    def __init__(self):
        self.calls = []

    def _record(self, name, kwargs):
        self.calls.append((name, kwargs))
        return {"delegated": name, "kwargs": kwargs}

    def observe(self, **kwargs):
        return self._record("observe", kwargs)

    def pixel_to_world(self, **kwargs):
        return self._record("pixel_to_world", kwargs)

    def navigate_to(self, **kwargs):
        return self._record("navigate_to", kwargs)

    def move_to(self, **kwargs):
        return self._record("move_to", kwargs)

    def pick(self, **kwargs):
        return self._record("pick", kwargs)

    def rotate_wrist(self, **kwargs):
        return self._record("rotate_wrist", kwargs)

    def press(self, **kwargs):
        return self._record("press", kwargs)

    def release(self, **kwargs):
        return self._record("release", kwargs)


def _pi0_primitives(
    tmp_path: Path,
    *,
    model: _FakeModel,
    env: _FakeEnv,
    initial_observation: dict,
    initial_info=None,
    local_grasp_validator=None,
) -> BehaviorPrimitives:
    return BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=64,
        action_horizon=32,
        output_dir=tmp_path,
        initial_observation=initial_observation,
        initial_info=(
            {"done": {"success": False}}
            if initial_info is None
            else initial_info
        ),
        local_grasp_validator=local_grasp_validator,
    )


def test_literal_planner_tool_tuple_is_closed_and_exact():
    assert PLANNER_TOOL_NAMES == EXPECTED_PLANNER_TOOLS
    assert "run_full_task" not in PLANNER_TOOL_NAMES
    assert "pi0_pick" not in PLANNER_TOOL_NAMES


def test_rotate_wrist_schema_and_public_primitive_enforce_xor():
    schema = PLANNER_TOOL_SPECS["rotate_wrist"]["input_schema"]
    assert len(schema["oneOf"]) == 2
    primitives = BehaviorPrimitives(planner_backend=_PlannerBackend())

    with pytest.raises(ValueError, match="exactly one"):
        primitives.rotate_wrist(hand="left")
    with pytest.raises(ValueError, match="exactly one"):
        primitives.rotate_wrist(
            hand="left",
            target_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            relative_axis_angle=[0.0, 0.0, 1.0, 0.1],
        )


def test_public_primitive_entrypoints_all_resolve_to_behavior_primitives():
    expected_names = ("run_full_task", *EXPECTED_PLANNER_TOOLS, "pi0_pick")

    assert tuple(PUBLIC_PRIMITIVE_ENTRYPOINTS) == expected_names
    assert PUBLIC_PRIMITIVE_ENTRYPOINTS == {
        name: f"BehaviorPrimitives.{name}" for name in expected_names
    }


def test_behavior_tools_exports_only_public_behavior_primitives_class():
    assert behavior_tools.__all__ == ["BehaviorPrimitives", "official_task_success"]
    assert behavior_tools.BehaviorPrimitives is BehaviorPrimitives
    assert "FullTaskRunner" not in behavior_tools.__all__
    assert not hasattr(behavior_tools, "FullTaskRunner")


def test_toolkit_modes_expose_exactly_one_closed_control_surface(tmp_path):
    primitives_kwargs = {
        "env": object(),
        "model": object(),
        "max_episode_steps": 1,
        "output_dir": tmp_path,
    }
    planner = _PlannerBackend()

    full = BehaviorToolkit(
        control_mode=FULL_TASK_VLA_MODE,
        primitives_kwargs=primitives_kwargs,
    )
    planner_tools = BehaviorToolkit(
        control_mode=PLANNER_TOOLS_MODE,
        planner_client=planner,
    )
    pi0 = BehaviorToolkit(
        control_mode=PI0_PICK_VLA_MODE,
        primitives_kwargs=primitives_kwargs,
    )

    assert [spec["name"] for spec in full.get_tools_spec()] == ["run_full_task"]
    assert [spec["name"] for spec in planner_tools.get_tools_spec()] == list(
        EXPECTED_PLANNER_TOOLS
    )
    assert [spec["name"] for spec in pi0.get_tools_spec()] == ["pi0_pick"]


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("observe", {"camera": "right_wrist"}),
        (
            "pixel_to_world",
            {
                "camera": "left_wrist",
                "frame_id": "left_wrist:17",
                "u": 123,
                "v": 231,
                "depth_window_px": 9,
                "output_frame": "world",
            },
        ),
        (
            "navigate_to",
            {
                "hand": "left",
                "target_xyz": [1.0, 2.0, 0.7],
                "frame": "world",
                "standoff_m": 0.7,
                "timeout_s": 88,
            },
        ),
        (
            "move_to",
            {
                "hand": "right",
                "target_xyz": [0.1, 0.2, 0.3],
                "frame": "world",
                "target_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "plan_only": True,
                "position_tolerance_m": 0.01,
                "orientation_tolerance_rad": 0.04,
                "timeout_s": 44,
            },
        ),
        (
            "pick",
            {
                "hand": "left",
                "target_xyz": [0.4, 0.5, 0.6],
                "approach_vector": [0.0, 0.0, -1.0],
                "grasp_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "pregrasp_offset_m": 0.09,
                "lift_m": 0.07,
                "timeout_s": 89,
            },
        ),
        (
            "rotate_wrist",
            {
                "hand": "right",
                "target_quat_xyzw": None,
                "relative_axis_angle": [0.0, 0.0, 1.0, 0.2],
                "frame": "eef",
                "timeout_s": 43,
            },
        ),
        (
            "press",
            {
                "hand": "left",
                "target_xyz": [0.5, 0.4, 0.3],
                "press_direction": [0.0, 0.0, -1.0],
                "approach_distance_m": 0.05,
                "press_depth_m": 0.01,
                "timeout_s": 59,
            },
        ),
        (
            "release",
            {
                "hand": "right",
                "opening": 0.9,
                "retreat_vector": [0.0, 0.0, 1.0],
                "retreat_m": 0.04,
                "timeout_s": 29,
            },
        ),
    ],
)
def test_each_planner_primitive_delegates_to_injected_backend(name, kwargs):
    backend = _PlannerBackend()
    primitives = BehaviorPrimitives(planner_backend=backend)

    result = getattr(primitives, name)(**kwargs)

    assert result == {"delegated": name, "kwargs": kwargs}
    assert backend.calls == [(name, kwargs)]


def test_pi0_pick_validates_every_action_chunk_and_keeps_prompt_local(tmp_path):
    original = _observation(task="official full task")
    first = _observation(task="official full task")
    second = _observation(task="official full task")
    model = _FakeModel(
        np.zeros((2, 23), dtype=np.float32),
        np.zeros((3, 23), dtype=np.float32),
    )
    env = _FakeEnv(
        (first, 0.0, False, False, {"done": {"success": False}}),
        (second, 0.0, False, False, {"done": {"success": False}}),
    )
    primitives = _pi0_primitives(
        tmp_path,
        model=model,
        env=env,
        initial_observation=original,
    )

    result = primitives.pi0_pick(
        hand="right",
        instruction="grasp only the red radio handle",
        max_chunks=2,
    )

    assert result["stop_reason"] == "chunk_limit"
    assert [action.shape for action in env.actions] == [(2, 23), (3, 23)]
    assert [obs["task_descriptions"] for obs in model.observations] == [
        "Use only the right hand for this local grasp. "
        "grasp only the red radio handle",
        "Use only the right hand for this local grasp. "
        "grasp only the red radio handle",
    ]
    assert result["model_instruction"].startswith("Use only the right hand")
    assert original["task_descriptions"] == "official full task"
    assert first["task_descriptions"] == "official full task"
    assert second["task_descriptions"] == "official full task"


def test_pi0_pick_closed_gripper_is_only_a_visual_candidate_by_default(tmp_path):
    initial = _observation(right_gripper=0.08)
    closed = _observation(right_gripper=0.02)
    env = _FakeEnv(
        (closed, 0.0, False, False, {"done": {"success": False}})
    )
    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(np.zeros((1, 23), dtype=np.float32)),
        env=env,
        initial_observation=initial,
    )

    result = primitives.pi0_pick(hand="right", instruction="grasp the radio")

    assert result["local_gripper_closure_detected"] is True
    assert result["primitive_success"] is False
    assert result["local_grasp_success"] is False
    assert result["local_grasp_validator_configured"] is False
    assert result["local_grasp_validator_result"] is None
    assert result["visual_verification_required"] is True
    assert result["stop_reason"] == "local_gripper_closure_detected"


def test_pi0_pick_validator_acceptance_is_required_for_local_success(tmp_path):
    initial = _observation(left_gripper=0.08)
    closed = _observation(left_gripper=0.02)
    validator_calls = []

    def validator(obs, context):
        validator_calls.append((obs, context))
        return True

    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(np.zeros((1, 23), dtype=np.float32)),
        env=_FakeEnv(
            (closed, 0.0, False, False, {"done": {"success": False}})
        ),
        initial_observation=initial,
        local_grasp_validator=validator,
    )

    result = primitives.pi0_pick(hand="left", instruction="grasp the radio")

    assert result["primitive_success"] is True
    assert result["local_grasp_success"] is True
    assert result["local_grasp_validator_result"] is True
    assert result["visual_verification_required"] is True
    assert result["stop_reason"] == "local_grasp_success"
    assert len(validator_calls) == 1
    assert validator_calls[0][1]["hand"] == "left"


def test_pi0_pick_validator_rejection_continues_to_bound(tmp_path):
    closed = _observation(right_gripper=0.02)
    validator_calls = []

    def reject(_obs, context):
        validator_calls.append(context)
        return False

    model = _FakeModel(
        np.zeros((1, 23), dtype=np.float32),
        np.zeros((1, 23), dtype=np.float32),
        np.zeros((1, 23), dtype=np.float32),
    )
    env = _FakeEnv(
        *[
            (closed, 0.0, False, False, {"done": {"success": False}})
            for _ in range(3)
        ]
    )
    primitives = _pi0_primitives(
        tmp_path,
        model=model,
        env=env,
        initial_observation=_observation(right_gripper=0.08),
        local_grasp_validator=reject,
    )

    result = primitives.pi0_pick(
        hand="right", instruction="grasp the radio", max_chunks=3
    )

    assert result["primitive_success"] is False
    assert result["local_gripper_closure_detected"] is True
    assert result["local_grasp_validator_result"] is False
    assert result["stop_reason"] == "chunk_limit"
    assert result["chunks_used"] == 3
    assert len(validator_calls) == 3


@pytest.mark.parametrize(
    ("terminated", "truncated", "stop_reason"),
    [(True, False, "terminated"), (False, True, "truncated")],
)
def test_pi0_pick_environment_stop_is_never_local_grasp_success(
    tmp_path, terminated, truncated, stop_reason
):
    closed = _observation(left_gripper=0.01)
    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(np.zeros((1, 23), dtype=np.float32)),
        env=_FakeEnv(
            (
                closed,
                7.0,
                terminated,
                truncated,
                {"done": {"success": False}},
            )
        ),
        initial_observation=_observation(left_gripper=0.08),
        local_grasp_validator=lambda *_args: True,
    )

    result = primitives.pi0_pick(hand="left", instruction="grasp the radio")

    assert result["primitive_success"] is False
    assert result["local_grasp_success"] is False
    assert result["local_gripper_closure_detected"] is False
    assert result["task_success"] is False
    assert result["terminated"] is terminated
    assert result["truncated"] is truncated
    assert result["stop_reason"] == stop_reason


def test_pi0_pick_raw_done_success_only_sets_task_success_and_finish(tmp_path):
    still_open = _observation(right_gripper=0.08)
    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(np.zeros((1, 23), dtype=np.float32)),
        env=_FakeEnv(
            (
                still_open,
                99.0,
                False,
                False,
                {
                    "done": {"success": True},
                    "task_success": False,
                    "success": False,
                },
            )
        ),
        initial_observation=_observation(right_gripper=0.08),
    )

    result = primitives.pi0_pick(hand="right", instruction="grasp the radio")

    assert result["task_success"] is True
    assert result["_finish"] is True
    assert result["primitive_success"] is False
    assert result["local_grasp_success"] is False
    assert result["stop_reason"] == "task_success"


@pytest.mark.parametrize("failure", ["shape", "nan"])
def test_pi0_pick_rejects_malformed_actions_before_env_step_and_writes_artifacts(
    tmp_path, failure
):
    actions = np.zeros((1, 22), dtype=np.float32)
    if failure == "nan":
        actions = np.zeros((1, 23), dtype=np.float32)
        actions[0, 4] = np.nan
    env = _FakeEnv()
    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(actions),
        env=env,
        initial_observation=_observation(),
    )

    result = primitives.pi0_pick(hand="left", instruction="grasp the radio")

    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["stop_reason"] == "error"
    assert result["chunks_used"] == 0
    assert result["env_steps_used"] == 0
    assert result["error"].startswith("ValueError:")
    assert env.actions == []
    states_path = Path(result["states_path"])
    result_path = Path(result["result_path"])
    assert states_path.is_file()
    assert result_path.is_file()
    assert len(json.loads(states_path.read_text(encoding="utf-8"))) == 1
    assert json.loads(result_path.read_text(encoding="utf-8")) == result
