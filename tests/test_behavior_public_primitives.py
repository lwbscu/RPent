import json
from pathlib import Path

import numpy as np
import pytest

import robots.behavior.tools as behavior_tools
from robots.behavior.schemas import (
    DECLARE_BUTTON_VISIBILITY_SPEC,
    FULL_TASK_VLA_MODE,
    PI0_NAV_PICK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOL_SPECS,
    PLANNER_TOOLS_MODE,
    POST_PICK_OBSERVE_SPEC,
    POST_PICK_PIXEL_TO_WORLD_SPEC,
    POST_PICK_RESTORE_ROBOT_STATE_CHECKPOINT_SPEC,
    POST_PICK_SAVE_ROBOT_STATE_CHECKPOINT_SPEC,
    POST_PICK_TOOL_NAMES,
    PREPRESS_MOVE_TO_SPEC,
    PREPRESS_ROTATE_WRIST_SPEC,
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
    return {
        "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 2, 2, 3), dtype=np.uint8),
        "states": states,
        "task_descriptions": task,
    }


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
            {"done": {"success": False}} if initial_info is None else initial_info
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
    expected_names = (
        "run_full_task",
        *EXPECTED_PLANNER_TOOLS,
        "pi0_pick",
        "pi0_navigate_to",
        "pi0_nav_pick",
        *[
            name
            for name in POST_PICK_TOOL_NAMES
            if name not in EXPECTED_PLANNER_TOOLS
            and name
            not in {
                "save_robot_state_checkpoint",
                "restore_robot_state_checkpoint",
            }
        ],
        "save_robot_state_checkpoint",
        "restore_robot_state_checkpoint",
    )

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
    nav_pick = BehaviorToolkit(
        control_mode=PI0_NAV_PICK_VLA_MODE,
        primitives_kwargs=primitives_kwargs,
    )

    assert [spec["name"] for spec in full.get_tools_spec()] == ["run_full_task"]
    assert [spec["name"] for spec in planner_tools.get_tools_spec()] == list(
        EXPECTED_PLANNER_TOOLS
    )
    assert [spec["name"] for spec in pi0.get_tools_spec()] == ["pi0_pick"]
    assert [spec["name"] for spec in nav_pick.get_tools_spec()] == [
        "pi0_nav_pick",
        *POST_PICK_TOOL_NAMES,
    ]
    nav_names = [spec["name"] for spec in nav_pick.get_tools_spec()]
    assert len(nav_names) == len(set(nav_names))
    assert "save_prepress_checkpoint" not in nav_names
    assert "project_button" not in nav_names


def test_nav_pick_stage2_surface_uses_existing_names_and_bound_hand_schemas():
    required = {
        "observe",
        "pixel_to_world",
        "move_to",
        "rotate_wrist",
        "save_robot_state_checkpoint",
        "restore_robot_state_checkpoint",
    }
    assert required <= set(POST_PICK_TOOL_NAMES)
    assert not {
        "prepress_move_to",
        "generic_move_to",
        "project_button",
        "save_prepress_checkpoint",
    } & set(POST_PICK_TOOL_NAMES)

    move_schema = PREPRESS_MOVE_TO_SPEC["input_schema"]
    assert "hand" not in move_schema["properties"]
    assert "target_xyz" not in move_schema["properties"]
    assert "target_quat_xyzw" not in move_schema["properties"]
    assert move_schema["properties"]["role"]["enum"] == ["held", "press"]
    assert set(move_schema["required"]) == {"role", "button_goal"}
    goal_kinds = {
        branch["properties"]["kind"]["const"]
        for branch in move_schema["properties"]["button_goal"]["oneOf"]
    }
    assert goal_kinds == {"held_button_alignment", "press_staging"}
    held_goal_schema = next(
        branch
        for branch in move_schema["properties"]["button_goal"]["oneOf"]
        if branch["properties"]["kind"]["const"] == "held_button_alignment"
    )
    assert "head_target_uv" in held_goal_schema["properties"]
    assert "head_target_radius_px" in held_goal_schema["properties"]
    assert "alignment_phase" in held_goal_schema["properties"]
    assert "plan_only" in move_schema["properties"]

    rotate_schema = PREPRESS_ROTATE_WRIST_SPEC["input_schema"]
    assert "hand" not in rotate_schema["properties"]
    assert rotate_schema["properties"]["role"]["enum"] == ["held", "press"]
    assert "plan_only" in rotate_schema["properties"]
    assert len(rotate_schema["oneOf"]) == 2

    observe_schema = POST_PICK_OBSERVE_SPEC["input_schema"]
    assert observe_schema["properties"]["camera"]["enum"] == [
        "head",
        "held_wrist",
        "press_wrist",
    ]
    pixel_schema = POST_PICK_PIXEL_TO_WORLD_SPEC["input_schema"]
    assert set(pixel_schema["required"]) == {"camera", "frame_id", "u", "v"}
    assert pixel_schema["properties"]["camera"]["enum"] == [
        "head",
        "held_wrist",
        "press_wrist",
    ]

    save_schema = POST_PICK_SAVE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"]
    assert save_schema["properties"]["checkpoint_name"]["const"] == (
        "state_checkpoint_2"
    )
    assert save_schema["properties"]["stage"]["const"] == ("pre_press_alignment")
    assert not {"held_hand", "press_hand", "object_name"} & set(
        save_schema["properties"]
    )

    restore_schema = POST_PICK_RESTORE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"]
    assert restore_schema["properties"]["checkpoint_name"]["enum"] == [
        "state_checkpoint_1",
        "state_checkpoint_2",
    ]


def test_button_visibility_schema_exposes_only_canonical_negative_face_classes():
    schema = DECLARE_BUTTON_VISIBILITY_SPEC["input_schema"]
    negative_case = schema["properties"]["negative_case"]

    assert negative_case["type"] == ["string", "null"]
    assert negative_case["enum"] == [
        "clear_slotted_back_face",
        "side_port",
        "ambiguous",
        None,
    ]
    assert not {"allow_large", "override_safety", "hand"} & set(
        PREPRESS_MOVE_TO_SPEC["input_schema"]["properties"]
    )


def test_post_pick_button_visibility_rejects_noncanonical_negative_case():
    class Env:
        def declare_button_visibility(self, **_kwargs):
            raise AssertionError("invalid declaration must not reach the env")

    primitives = BehaviorPrimitives(env=Env())
    with pytest.raises(ValueError, match="negative_case"):
        primitives.declare_button_visibility(
            camera="head",
            frame_id="head:1819:abc",
            button_visible=False,
            negative_case="red_side_slot",
        )


def test_post_pick_primitives_delegate_exact_core_signatures_without_hand():
    class Env:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def call(**kwargs):
                self.calls.append((name, kwargs))
                return {
                    "primitive_success": True,
                    "task_success": False,
                    "official_success_source": 'info["done"]["success"]',
                    "stop_reason": name,
                }

            return call

    env = Env()
    primitives = BehaviorPrimitives(env=env)
    primitives.inspect_post_pick_state()
    primitives.observe(camera="head")
    primitives.declare_button_visibility(
        camera="head",
        frame_id="head:1819:abc",
        button_visible=False,
        negative_case="clear_slotted_back_face",
    )
    primitives.pixel_to_world(
        camera="head",
        frame_id="head:1819:abc",
        u=10,
        v=12,
    )
    primitives.evaluate_prepress_geometry(projection_id="projection-1")
    primitives.prepress_move_to(
        role="press",
        button_goal={
            "kind": "press_staging",
            "projection_id": "projection-1",
            "standoff_m": 0.055,
        },
        plan_only=True,
    )
    primitives.prepress_rotate_wrist(
        role="press",
        relative_axis_angle=[0.0, 0.0, 1.0, 0.01],
        plan_only=True,
    )
    primitives.save_post_pick_robot_state_checkpoint()
    primitives.restore_robot_state_checkpoint()

    aliases = {
        "move_to": "prepress_move_to",
        "rotate_wrist": "prepress_rotate_wrist",
        "save_robot_state_checkpoint": "save_prepress_checkpoint",
    }
    expected_internal_calls = [aliases.get(name, name) for name in POST_PICK_TOOL_NAMES]
    assert [name for name, _kwargs in env.calls] == expected_internal_calls
    assert env.calls[0][1] == {"checkpoint_name": "state_checkpoint_1"}
    assert env.calls[2][1]["bbox_xyxy"] is None
    assert env.calls[2][1]["center_uv"] is None
    assert env.calls[2][1]["negative_case"] == "clear_slotted_back_face"
    assert "hand" not in env.calls[5][1]
    assert env.calls[5][1]["role"] == "press"
    assert "hand" not in env.calls[6][1]
    assert env.calls[6][1]["role"] == "press"
    assert env.calls[7][1] == {
        "checkpoint_name": "state_checkpoint_2",
        "stage": "pre_press_alignment",
        "visual_review": True,
    }


def test_pi0_nav_pick_failed_handoff_preserves_local_evidence_and_actual_steps(
    tmp_path,
):
    observation = _observation(right_gripper=0.03)
    monitor = {
        "executed_steps": 5,
        "handoff_env_steps": 0,
        "total_env_steps": 6,
        "local_grasp_success": True,
        "held_hand": "right",
        "per_hand": {"right": {"passed": True}, "left": {"passed": False}},
        "current_criteria": {"right": {"opening_strict": True}},
        "validator_trace_path": str(tmp_path / "validator.json"),
        "state_checkpoint_path": None,
        "handoff_state": "FAILED",
        "action_source": "pi0_vla",
        "vla_actions_enabled": False,
        "paused_runtime_path": None,
        "stop_reason": "handoff_failed:post_reload_hold",
    }

    class Env:
        def pi0_nav_pick_chunk_step(self, _actions, *, chunk_index):
            assert chunk_index == 1
            return (
                observation,
                0.0,
                False,
                False,
                {
                    "done": {"success": False},
                    "_rpent": {
                        "executed_steps": 5,
                        "handoff_env_steps": 0,
                        "total_env_steps": 6,
                        "pi0_nav_pick_monitor": monitor,
                    },
                },
            )

    result = BehaviorPrimitives(
        env=Env(),
        model=_FakeModel(np.zeros((32, 23), dtype=np.float32)),
        max_episode_steps=64,
        action_horizon=32,
        output_dir=tmp_path,
        initial_observation=observation,
        initial_info={
            "done": {"success": False},
            "_rpent": {"total_env_steps": 0},
        },
    ).pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert result["local_grasp_success"] is True
    assert result["held_hand"] == "right"
    assert result["primitive_success"] is False
    assert result["handoff_state"] == "FAILED"
    assert result["vla_env_steps_used"] == 5
    assert result["handoff_env_steps_used"] == 1
    assert result["total_env_steps"] == 6
    assert result["last_pi0_nav_pick_monitor"]["handoff_env_steps"] == 1
    assert result["stop_reason"] == "handoff_failed:post_reload_hold"


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
        "grasp only the red radio handle",
        "grasp only the red radio handle",
    ]
    assert result["model_instruction"] == "grasp only the red radio handle"
    assert original["task_descriptions"] == "official full task"
    assert first["task_descriptions"] == "official full task"
    assert second["task_descriptions"] == "official full task"


def test_pi0_pick_resolves_selected_hand_without_rewriting_explicit_instruction(
    tmp_path,
):
    model = _FakeModel(np.zeros((1, 23), dtype=np.float32))
    primitives = _pi0_primitives(
        tmp_path,
        model=model,
        env=_FakeEnv(
            (
                _observation(),
                0.0,
                False,
                False,
                {"done": {"success": False}},
            )
        ),
        initial_observation=_observation(),
    )

    result = primitives.pi0_pick(
        hand="left",
        instruction="Grasp the radio with the selected hand.",
        max_chunks=1,
    )

    assert result["model_instruction"] == "Grasp the radio with the left hand."
    assert model.observations[0]["task_descriptions"] == result["model_instruction"]


def test_pi0_pick_closed_gripper_is_only_a_visual_candidate_by_default(tmp_path):
    initial = _observation(right_gripper=0.08)
    closed = _observation(right_gripper=0.02)
    env = _FakeEnv(
        (closed, 0.0, False, False, {"done": {"success": False}}),
        (closed, 0.0, False, False, {"done": {"success": False}}),
    )
    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(
            np.zeros((1, 23), dtype=np.float32),
            np.zeros((1, 23), dtype=np.float32),
        ),
        env=env,
        initial_observation=initial,
    )

    result = primitives.pi0_pick(
        hand="right", instruction="grasp the radio", max_chunks=2
    )

    assert result["local_gripper_closure_detected"] is True
    assert result["primitive_success"] is False
    assert result["local_grasp_success"] is False
    assert result["local_grasp_validator_configured"] is False
    assert result["local_grasp_validator_result"] is None
    assert result["visual_verification_required"] is True
    assert result["stop_reason"] == "chunk_limit"
    assert result["chunks_used"] == 2


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
        env=_FakeEnv((closed, 0.0, False, False, {"done": {"success": False}})),
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


def test_pi0_pick_uses_pi0_step_monitor_and_sanitizes_validator_input(tmp_path):
    class _MonitoredEnv(_FakeEnv):
        def __init__(self):
            super().__init__(
                (
                    _observation(right_gripper=0.02),
                    0.0,
                    False,
                    False,
                    {
                        "done": {"success": False},
                        "obs_info": {"forbidden": "object pose"},
                        "_rpent": {
                            "executed_steps": 5,
                            "local_gripper_monitor": {
                                "hand": "right",
                                "candidate": True,
                                "opening": 0.02,
                                "minimum_opening": 0.019,
                                "closed_streak_steps": 3,
                                "candidate_env_step": 5,
                            },
                        },
                    },
                )
            )
            self.monitor_kwargs = None

        def pi0_chunk_step(self, actions, **kwargs):
            self.monitor_kwargs = kwargs
            return self.chunk_step(actions)

    validator_calls = []

    def validator(obs, context):
        validator_calls.append((obs, context))
        return True

    env = _MonitoredEnv()
    primitives = _pi0_primitives(
        tmp_path,
        model=_FakeModel(np.zeros((8, 23), dtype=np.float32)),
        env=env,
        initial_observation=_observation(right_gripper=0.08),
        local_grasp_validator=validator,
    )

    result = primitives.pi0_pick(hand="right", instruction="grasp the radio")

    assert result["local_grasp_success"] is True
    assert result["env_steps_used"] == 5
    assert env.monitor_kwargs == {
        "hand": "right",
        "gripper_closed_threshold": 0.045,
        "required_closed_steps": 3,
        "stop_on_candidate": True,
    }
    assert set(validator_calls[0][0]) == {
        "main_images",
        "wrist_images",
        "states",
    }
    assert "task_success" not in validator_calls[0][1]
    assert "info" not in validator_calls[0][1]
    assert "reward" not in validator_calls[0][1]
    assert result["last_info"]["_rpent"]["local_gripper_monitor"]["candidate"] is True


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
        *[(closed, 0.0, False, False, {"done": {"success": False}}) for _ in range(3)]
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
                    "obs_info": {"privileged_object_pose": [1.0, 2.0, 3.0]},
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
    assert result["last_info"] == {"done": {"success": True}}
    raw_info = json.loads(Path(result["raw_final_info_path"]).read_text())
    assert raw_info["obs_info"]["privileged_object_pose"] == [1.0, 2.0, 3.0]


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
