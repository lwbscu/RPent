import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.behavior.env_server import (
    BehaviorEnvFacade,
    _configure_control_mode,
    _MainThreadDispatcher,
)
from robots.behavior.prompt_bundle import mode_instructions
from robots.behavior.schemas import (
    ENV_ACTION_SEGMENTS,
    HYBRID_TOOL_NAMES,
    HYBRID_VLM_PI0_MODE,
    PI0_NAVIGATE_TO_SPEC,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOLS_MODE,
)
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives


def test_pi0_navigate_to_contract_allows_posture_but_reserves_grasp_for_pick():
    description = PI0_NAVIGATE_TO_SPEC["description"]
    prompts = mode_instructions(
        HYBRID_VLM_PI0_MODE,
        pi0_instruction="turn on the radio",
    )
    system = prompts["behavior_system_instructions"]
    user = prompts["behavior_user_instructions"]

    assert "predicted trunk and both arm segments are executed" in description
    assert "Both grippers remain locked" in description
    assert "object grasping is exclusively pi0_pick" in description
    assert "whole-body" in system and "posture" in system
    assert "object grasping is allowed only through pi0_pick" in system
    assert "keeps both grippers latched" in user
    assert "never claim or attempt a grasp with pi0_navigate_to" in user


def _observation(*, right_gripper: float = 0.08) -> dict:
    states = np.zeros(256, dtype=np.float32)
    states[193:195] = 0.04
    states[232:234] = right_gripper / 2.0
    return {
        "main_images": np.zeros((4, 6, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 4, 6, 3), dtype=np.uint8),
        "states": states,
        "task_descriptions": "turn on the radio",
    }


class _HybridEnv:
    def __init__(self):
        self.total_env_steps = 100
        self.reset_calls = 0
        self.monitor_kwargs = []

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("hybrid pi0_pick must not reset the episode")

    def current_observation(self):
        return _observation(), {
            "done": {"success": False},
            "_rpent": {"total_env_steps": self.total_env_steps},
        }

    def pi0_chunk_step(self, actions, **kwargs):
        self.monitor_kwargs.append(kwargs)
        self.total_env_steps += len(actions)
        return (
            _observation(right_gripper=0.02),
            0.0,
            False,
            False,
            {
                "done": {"success": False},
                "_rpent": {
                    "executed_steps": len(actions),
                    "total_env_steps": self.total_env_steps,
                    "local_gripper_monitor": {
                        "candidate": True,
                        "candidate_env_step": 101,
                        "opening": 0.02,
                    },
                },
            },
        )

    def pi0_navigate_to_chunk_step(
        self,
        actions,
        *,
        segment_index,
        chunk_index,
    ):
        self.total_env_steps += len(actions)
        return (
            _observation(),
            0.0,
            False,
            False,
            {
                "done": {"success": False},
                "_rpent": {
                    "executed_steps": len(actions),
                    "total_env_steps": self.total_env_steps,
                    "pi0_navigate_to_monitor": {
                        "safety_stop": False,
                        "visual_review": {
                            "segment_index": segment_index,
                            "chunk_index": chunk_index,
                            "views": {},
                        },
                    },
                },
            },
        )


class _Model:
    def predict_action_batch(self, _obs, *, mode):
        assert mode == "eval"
        return np.zeros((1, 23), dtype=np.float32), {}


class _NavigateModel:
    def __init__(self):
        self.instructions = []
        self.markers = []

    def predict_action_batch(self, obs, *, mode):
        assert mode == "eval"
        self.instructions.append(obs["task_descriptions"])
        self.markers.append(obs["marker"])
        actions = np.full((32, 23), 9.0, dtype=np.float32)
        actions[:, ENV_ACTION_SEGMENTS["base"]] = [2.0, -2.0, 0.5]
        return actions, {"source": "fake"}


class _NavigateEnv:
    def __init__(self, *, initially_successful=False):
        self.total_env_steps = 20
        self.initially_successful = initially_successful
        self.chunk_lengths = []

    def current_observation(self):
        observation = _observation()
        observation["marker"] = 0
        return observation, {
            "done": {"success": self.initially_successful},
            "_rpent": {"total_env_steps": self.total_env_steps},
        }

    def pi0_navigate_to_chunk_step(
        self,
        actions,
        *,
        segment_index,
        chunk_index,
    ):
        self.chunk_lengths.append(len(actions))
        self.total_env_steps += len(actions)
        observation = _observation()
        observation["marker"] = chunk_index
        return (
            observation,
            0.0,
            False,
            False,
            {
                "done": {"success": False},
                "_rpent": {
                    "executed_steps": len(actions),
                    "total_env_steps": self.total_env_steps,
                    "pi0_navigate_to_monitor": {
                        "safety_stop": False,
                        "stop_reason": None,
                        "visual_review": {
                            "segment_index": segment_index,
                            "chunk_index": chunk_index,
                            "views": {},
                        },
                    },
                },
            },
        )


def test_pi0_navigate_to_uses_exact_language_eight_step_chunks_and_visual_pause(
    tmp_path,
):
    env = _NavigateEnv()
    model = _NavigateModel()
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=200,
        output_dir=tmp_path,
        hybrid_mode=True,
    )
    exact_task_language = "Turn on the radio receiver that's on the table.  "

    result = primitives.pi0_navigate_to(
        instruction=exact_task_language,
        max_chunks=4,
    )

    assert env.chunk_lengths == [8, 8, 8, 8]
    assert model.instructions == [exact_task_language] * 4
    assert model.markers == [0, 1, 2, 3]
    assert result["name"] == "pi0_navigate_to"
    assert result["stop_reason"] == "visual_review_pause"
    assert result["segment_completed"] is True
    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["_finish"] is False
    assert result["env_steps_used"] == 32
    assert result["total_env_steps"] == 52
    assert len(result["visual_artifacts"]) == 4
    assert result["_image_bytes"].startswith(b"\x89PNG")
    assert result["_image_cam_bytes"].startswith(b"\x89PNG")
    assert result["_image_wrist_bytes"].startswith(b"\x89PNG")


def test_pi0_navigate_to_stops_on_fresh_official_success_without_motion(tmp_path):
    env = _NavigateEnv(initially_successful=True)
    model = _NavigateModel()
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=200,
        output_dir=tmp_path,
        hybrid_mode=True,
    )

    result = primitives.pi0_navigate_to(
        instruction="turn on the radio",
        max_chunks=4,
    )

    assert env.chunk_lengths == []
    assert model.instructions == []
    assert result["task_success"] is True
    assert result["stop_reason"] == "task_success"
    assert result["_finish"] is True
    assert result["suggested_next_tool"] is None


def test_hybrid_pi0_refreshes_same_episode_and_runs_four_post_candidate_chunks(
    tmp_path,
):
    env = _HybridEnv()
    primitives = BehaviorPrimitives(
        env=env,
        model=_Model(),
        max_episode_steps=200,
        output_dir=tmp_path,
        hybrid_mode=True,
    )

    result = primitives.pi0_pick(
        hand="right",
        instruction="grasp the radio",
        max_chunks=5,
        stop_on_closure_candidate=True,
        post_candidate_chunks=4,
    )

    assert env.reset_calls == 0
    assert len(env.monitor_kwargs) == 5
    assert all(call["stop_on_candidate"] is False for call in env.monitor_kwargs)
    assert result["stop_reason"] == "closure_candidate_visual_review"
    assert result["first_candidate_env_step"] == 101
    assert result["post_candidate_chunks_used"] == 4
    assert result["total_env_steps"] == 105
    assert result["primitive_success"] is False
    assert result["local_grasp_success"] is False
    assert result["task_success"] is False
    assert result["_finish"] is False
    assert result["visual_review_required"] is True
    assert result["_image_bytes"].startswith(b"\x89PNG")
    assert result["_image_cam_bytes"].startswith(b"\x89PNG")
    assert result["_image_wrist_bytes"].startswith(b"\x89PNG")


def test_pi0_navigate_to_then_pi0_pick_preserves_episode_and_global_horizon(tmp_path):
    env = _HybridEnv()
    primitives = BehaviorPrimitives(
        env=env,
        model=_Model(),
        max_episode_steps=200,
        output_dir=tmp_path,
        hybrid_mode=True,
    )

    nav_result = primitives.pi0_navigate_to(
        instruction="turn on the radio",
        max_chunks=1,
    )
    pick_result = primitives.pi0_pick(
        hand="right",
        instruction="turn on the radio",
        max_chunks=1,
        stop_on_closure_candidate=True,
        post_candidate_chunks=0,
    )

    assert env.reset_calls == 0
    assert nav_result["total_env_steps"] == 101
    assert pick_result["total_env_steps"] == 102


def test_hybrid_env_rpc_allows_planner_pi0_and_internal_current_observation():
    class Env:
        _control_mode = HYBRID_VLM_PI0_MODE

        @staticmethod
        def observe(camera):
            return camera

        @staticmethod
        def pi0_chunk_step(*_args, **_kwargs):
            return "pi0"

        @staticmethod
        def pi0_navigate_to_chunk_step(*_args, **_kwargs):
            return "pi0_navigate_to"

        @staticmethod
        def current_observation():
            return "current"

        @staticmethod
        def chunk_step(*_args, **_kwargs):
            return "full"

        @staticmethod
        def pick(*_args, **_kwargs):
            return "scripted-pick"

    dispatcher = _MainThreadDispatcher(Env(), threading.Event())

    assert dispatcher._dispatch("env.observe", (), {"camera": "head"}) == "head"
    assert dispatcher._dispatch("env.pi0_chunk_step", (), {}) == "pi0"
    assert (
        dispatcher._dispatch("env.pi0_navigate_to_chunk_step", (), {})
        == "pi0_navigate_to"
    )
    assert dispatcher._dispatch("env.current_observation", (), {}) == "current"
    for forbidden_method in ("env.chunk_step", "env.pick", "env.navigate_to"):
        try:
            dispatcher._dispatch(forbidden_method, (), {})
        except ValueError as error:
            assert "unknown BEHAVIOR env RPC" in str(error)
        else:
            raise AssertionError(f"hybrid must not expose {forbidden_method}")


def test_hybrid_uses_position_base_and_never_executes_pi0_base_outputs():
    base = SimpleNamespace(
        name="HolonomicBaseJointController",
        motor_type="velocity",
        command_input_limits=[[-1.0] * 3, [1.0] * 3],
        command_output_limits=[[-1.0] * 3, [1.0] * 3],
        use_impedances=True,
    )
    cfg = SimpleNamespace(
        omni_config=SimpleNamespace(
            robots=[
                SimpleNamespace(
                    type="R1Pro",
                    controller_config=SimpleNamespace(base=base),
                )
            ]
        )
    )
    _configure_control_mode(cfg, HYBRID_VLM_PI0_MODE)
    assert base.motor_type == "position"

    executed = []

    class Direct:
        @staticmethod
        def step_env(action, *, need_obs):
            assert need_obs is True
            executed.append(action.detach().cpu().numpy().reshape(23))
            return (
                {},
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                [{"done": {"success": False}}],
            )

    observation = _observation(right_gripper=0.02)
    hold = np.zeros(23, dtype=np.float32)
    hold[ENV_ACTION_SEGMENTS["base"]] = [7.0, 8.0, 9.0]
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = HYBRID_VLM_PI0_MODE
    facade._done = False
    facade._env_steps = 10
    facade._last_observation = observation
    facade._last_info = {"done": {"success": False}}
    facade._planner_video_interval_steps = 4
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(hold_action=lambda: hold.copy())
    )
    facade._env = SimpleNamespace(
        _direct_process=Direct(),
        _wrap_obs=lambda _raw: observation,
    )
    facade._record_rgbd_frames = lambda _raw, _wrapped: None
    facade._append_video = lambda _observation: None
    facade._validated_selected_gripper_opening = (
        lambda *, observation, hand: 0.02
    )
    facade._gripper_latch = {"left": 1.0, "right": 1.0}

    actions = np.full((2, 23), 0.5, dtype=np.float32)
    actions[:, ENV_ACTION_SEGMENTS["base"]] = [-0.3, 0.2, 0.1]
    actions[:, ENV_ACTION_SEGMENTS["left_gripper"]] = 0.25
    actions[:, ENV_ACTION_SEGMENTS["right_gripper"]] = -0.75
    result = facade.pi0_chunk_step(actions, hand="right")

    assert len(executed) == 2
    for action in executed:
        np.testing.assert_allclose(action[ENV_ACTION_SEGMENTS["base"]], [7, 8, 9])
    assert facade._gripper_latch == {"left": 0.25, "right": -0.75}
    assert result[4]["_rpent"]["total_env_steps"] == 12


def test_pi0_navigate_to_adapts_base_passes_posture_and_latches_grippers():
    executed = []

    class Direct:
        @staticmethod
        def step_env(action, *, need_obs):
            executed.append(action.detach().cpu().numpy().reshape(23))
            raw = {} if need_obs else None
            return (
                raw,
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                [{"done": {"success": False}}],
            )

    collision_forces = []
    dynamics_calls = []
    backend = SimpleNamespace(
        collision_report=lambda *, force: (
            collision_forces.append(force)
            or {"available": True, "colliding": False}
        ),
        dynamics_report=lambda: (
            dynamics_calls.append(True) or {"available": True, "ok": True}
        ),
    )
    observation = _observation()
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = HYBRID_VLM_PI0_MODE
    facade._done = False
    facade._env_steps = 10
    facade._last_observation = observation
    facade._last_info = {"done": {"success": False}}
    facade._planner_video_interval_steps = 4
    facade._planner = SimpleNamespace(backend=backend)
    facade._env = SimpleNamespace(
        _direct_process=Direct(),
        _wrap_obs=lambda _raw: observation,
    )
    facade._record_rgbd_frames = lambda _raw, _wrapped: None
    facade._append_video = lambda _observation: None
    facade._gripper_latch = {"left": 0.6, "right": -0.4}

    actions = np.full((8, 23), 9.0, dtype=np.float32)
    actions[:, ENV_ACTION_SEGMENTS["base"]] = [2.0, -2.0, 0.5]
    result = facade._step_action_chunk(
        actions,
        observe_final=True,
        pi0_navigate_to=True,
    )

    assert len(executed) == 8
    expected_delta = [0.01, -0.01, 0.5 / 60.0]
    for action in executed:
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS["base"]],
            expected_delta,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS["trunk"]],
            9.0,
        )
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS["left_arm"]],
            9.0,
        )
        np.testing.assert_allclose(
            action[ENV_ACTION_SEGMENTS["right_arm"]],
            9.0,
        )
        assert action[ENV_ACTION_SEGMENTS["left_gripper"]][0] == pytest.approx(0.6)
        assert action[ENV_ACTION_SEGMENTS["right_gripper"]][0] == pytest.approx(-0.4)
    assert facade._gripper_latch == {"left": 0.6, "right": -0.4}
    assert collision_forces == [True, True]
    assert len(dynamics_calls) == 2
    monitor = result[4]["_rpent"]["pi0_navigate_to_monitor"]
    assert monitor["input_clip_count"] == 16
    assert monitor["delta_clip_count"] == 16
    assert monitor["safety_stop"] is False
    assert monitor["execution_mode"] == "adapted_23d_vla_receding_horizon"
    assert monitor["predicted_action_dim"] == 23
    assert monitor["vla_passthrough_segments"] == [
        "trunk",
        "left_arm",
        "right_arm",
    ]
    assert monitor["adapted_segments"] == ["base"]
    assert monitor["held_segments"] == ["left_gripper", "right_gripper"]
    assert monitor["grasp_allowed"] is False
    assert "non_base_segments" not in monitor


@pytest.mark.parametrize(
    ("collision", "dynamics", "expected_reason"),
    [
        (
            {"available": False, "reason": "missing"},
            {"available": True, "ok": True},
            "collision_feedback_unavailable",
        ),
        (
            {"available": True, "colliding": True},
            {"available": True, "ok": True},
            "collision",
        ),
        (
            {"available": True, "colliding": False},
            {"available": False, "ok": None},
            "dynamics_feedback_unavailable",
        ),
        (
            {"available": True, "colliding": False},
            {"available": True, "ok": False},
            "dynamics_violation",
        ),
    ],
)
def test_pi0_navigate_to_fails_closed_at_four_step_safety_gate(
    collision,
    dynamics,
    expected_reason,
):
    executed = []

    class Direct:
        @staticmethod
        def step_env(action, *, need_obs):
            executed.append(action)
            return (
                {} if need_obs else None,
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                [{"done": {"success": False}}],
            )

    observation = _observation()
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = HYBRID_VLM_PI0_MODE
    facade._done = False
    facade._env_steps = 0
    facade._last_observation = observation
    facade._last_info = {"done": {"success": False}}
    facade._planner_video_interval_steps = 4
    facade._planner = SimpleNamespace(
        backend=SimpleNamespace(
            collision_report=lambda *, force: collision,
            dynamics_report=lambda: dynamics,
        )
    )
    facade._env = SimpleNamespace(
        _direct_process=Direct(),
        _wrap_obs=lambda _raw: observation,
    )
    facade._record_rgbd_frames = lambda _raw, _wrapped: None
    facade._append_video = lambda _observation: None
    facade._refresh_observation_without_step = lambda: None
    facade._gripper_latch = {"left": 1.0, "right": 1.0}

    result = facade._step_action_chunk(
        np.zeros((8, 23), dtype=np.float32),
        observe_final=True,
        pi0_navigate_to=True,
    )

    assert len(executed) == 4
    monitor = result[4]["_rpent"]["pi0_navigate_to_monitor"]
    assert monitor["safety_stop"] is True
    assert monitor["stop_reason"] == expected_reason


def test_hybrid_move_to_rewrites_navigation_suggestion_only_in_hybrid_mode():
    result = {"suggested_next_tool": "navigate_to", "task_success": False}
    hybrid = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    hybrid._control_mode = HYBRID_VLM_PI0_MODE
    hybrid._env_steps = 12
    planner = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    planner._control_mode = PLANNER_TOOLS_MODE
    planner._env_steps = 12

    assert (
        hybrid._planner_result_with_accounting(result)["suggested_next_tool"]
        == "pi0_navigate_to"
    )
    assert (
        planner._planner_result_with_accounting(result)["suggested_next_tool"]
        == "navigate_to"
    )


def test_hybrid_toolkit_surface_trace_recipe_and_live_observe_artifacts(tmp_path):
    class Planner:
        def observe(self, **_kwargs):
            return {
                "primitive_success": True,
                "task_success": False,
                "_image_bytes": b"\x89PNG\r\n",
            }

        def close(self):
            return None

    planner = Planner()
    toolkit = BehaviorToolkit(
        control_mode=HYBRID_VLM_PI0_MODE,
        primitives_kwargs={
            "env": object(),
            "model": object(),
            "max_episode_steps": 10,
            "output_dir": tmp_path,
        },
        planner_client=planner,
    )
    assert [spec["name"] for spec in toolkit.get_tools_spec()] == list(
        HYBRID_TOOL_NAMES
    )
    assert "pick" not in HYBRID_TOOL_NAMES
    assert "navigate_to" not in HYBRID_TOOL_NAMES
    assert "pi0_navigate_to" in HYBRID_TOOL_NAMES
    assert "run_full_task" not in HYBRID_TOOL_NAMES
    assert len(PLANNER_TOOL_NAMES) == 8
    assert "navigate_to" in PLANNER_TOOL_NAMES
    toolkit.execute_tool("observe", {"camera": "head"})
    recipe_path = Path(toolkit.write_recipe("hybrid") or "")
    hybrid_result = json.loads(
        (tmp_path / "hybrid_result.json").read_text(encoding="utf-8")
    )
    assert recipe_path.is_file()
    assert (tmp_path / "hybrid_tool_trace.jsonl").is_file()
    assert hybrid_result["task_success"] is False

    payload = {
        "camera": "head",
        "frame_id": "head:12:test",
        "capture_group": {"sim_step": 12},
        "_image_bytes": b"png-data",
    }
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._output_dir = tmp_path
    facade._env_steps = 12
    facade._live_observation_counter = 0
    public = facade._persist_live_observation(payload)
    assert Path(public["visual_review"]["rgb_path"]).read_bytes() == b"png-data"
    metadata = json.loads(
        Path(public["visual_review"]["metadata_path"]).read_text(encoding="utf-8")
    )
    assert metadata["frame_id"] == "head:12:test"
    assert metadata["total_env_steps"] == 12

    audit_call = 0

    def audit_observe(camera):
        nonlocal audit_call
        group = audit_call // 3 + 1
        audit_call += 1
        return {
            "camera": camera,
            "frame_id": f"{camera}:{group}",
            "capture_group": {"id": f"group-{group}", "sim_step": 12},
            "_image_bytes": f"png-{camera}-{group}".encode(),
        }

    facade._control_mode = HYBRID_VLM_PI0_MODE
    facade._planner = SimpleNamespace(observe=audit_observe)
    first = facade._persist_pi0_navigate_to_views(
        segment_index=1,
        chunk_index=1,
    )
    second = facade._persist_pi0_navigate_to_views(
        segment_index=2,
        chunk_index=1,
    )
    first_head = Path(first["views"]["head"]["rgb_path"])
    second_head = Path(second["views"]["head"]["rgb_path"])
    assert first_head != second_head
    assert first_head.read_bytes() == b"png-head-1"
    assert second_head.read_bytes() == b"png-head-2"
