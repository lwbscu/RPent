import inspect
import json
import threading
from pathlib import Path

import numpy as np
import pytest

import robots.behavior.vla_server as vla_server
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import (
    HANDOFF_ARTICULATION_ERROR_MAX_RAD,
    HANDOFF_COMOTION_RESIDUAL_MAX_M,
    HANDOFF_GRIPPER_OPENING_MAX,
    HANDOFF_POST_RELOAD_LIFT_TOLERANCE_M,
    HANDOFF_RADIO_LIFT_MIN_M,
    HANDOFF_RELATIVE_DRIFT_MAX_M,
    HANDOFF_SUPPORT_GAP_MIN_M,
    HANDOFF_VALIDATION_FRAMES,
    HANDOFF_WINDOW_MOTION_MIN_M,
    BehaviorEnvFacade,
    _handoff_controller_hold_warning,
    _handoff_held_object_stable,
    _MainThreadDispatcher,
    _post_reload_grasp_stable,
)
from robots.behavior.schemas import (
    ACTION_DIM,
    DEFAULT_ACTION_CHUNK,
    ENV_ACTION_SEGMENTS,
    FULL_TASK_VLA_MODE,
    HYBRID_TOOL_NAMES,
    HYBRID_VLM_PI0_MODE,
    PI0_NAV_PICK_SPEC,
    PI0_NAV_PICK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOLS_MODE,
    POST_PICK_TOOL_NAMES,
    RESTORE_ROBOT_STATE_CHECKPOINT_SPEC,
    SAVE_ROBOT_STATE_CHECKPOINT_SPEC,
)
from robots.behavior.toolkit import BehaviorToolkit
from robots.behavior.tools import BehaviorPrimitives
from robots.behavior.vla_client import BehaviorVLAClient


def _observation(*, marker: int = 0) -> dict:
    return {
        "main_images": np.zeros((4, 6, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 4, 6, 3), dtype=np.uint8),
        "states": np.zeros(256, dtype=np.float32),
        "task_descriptions": "Turn on the radio receiver.",
        "marker": marker,
    }


def _monitor(
    tmp_path: Path,
    *,
    total_env_steps: int,
    local_grasp_success: bool,
    held_hand: str | None,
    pose_jump_ok: bool,
    checkpoint: bool,
    executed_steps: int = DEFAULT_ACTION_CHUNK,
) -> dict:
    state_checkpoint_path = tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    paused_runtime_path = tmp_path / "paused_runtime.json"
    validator_trace_path = tmp_path / "validator_trace.jsonl"
    if checkpoint:
        state_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state_checkpoint_path.write_text("{}", encoding="utf-8")
        paused_runtime_path.write_text("{}", encoding="utf-8")
        validator_trace_path.write_text("{}\n", encoding="utf-8")
    selected = {
        "held": held_hand is not None,
        "gripper_closed": bool(local_grasp_success),
        "held_object_matches_target": bool(local_grasp_success),
        "pose_jump_ok": bool(pose_jump_ok),
    }
    return {
        "executed_steps": int(executed_steps),
        "handoff_env_steps": 0,
        "total_env_steps": int(total_env_steps),
        "local_grasp_success": bool(local_grasp_success),
        "held_hand": held_hand,
        "per_hand": {
            "left": selected if held_hand == "left" else {"held": False},
            "right": selected if held_hand == "right" else {"held": False},
        },
        "current_criteria": selected,
        "validator_trace_path": str(validator_trace_path),
        "state_checkpoint_path": (str(state_checkpoint_path) if checkpoint else None),
        "handoff_state": "PAUSED" if checkpoint else "VLA_ACTIVE",
        "action_source": "curobo" if checkpoint else "pi0_vla",
        # This is the env-side gate state at the moment the final action RPC
        # returns. The primitive then disables the independent VLA HTTP server.
        "vla_actions_enabled": True,
        "paused_runtime_path": (str(paused_runtime_path) if checkpoint else None),
    }


class _Model:
    def __init__(
        self,
        *,
        horizon: int = DEFAULT_ACTION_CHUNK,
        disable_pid: int = 123,
        health_pid: int = 123,
    ):
        self.horizon = int(horizon)
        self.calls = []
        self.actions_enabled = True
        self.disable_pid = disable_pid
        self.health_pid = health_pid
        self.endpoint = "http://127.0.0.1:45678"

    def predict_action_batch(self, observation, *, mode):
        assert mode == "eval"
        self.calls.append(
            {
                "instruction": observation["task_descriptions"],
                "marker": observation["marker"],
            }
        )
        actions = np.arange(self.horizon * ACTION_DIM, dtype=np.float32).reshape(
            self.horizon, ACTION_DIM
        )
        return actions, {"model_call": len(self.calls)}

    def disable_actions(self):
        self.actions_enabled = False
        return {
            "status": "ok",
            "pid": self.disable_pid,
            "actions_enabled": False,
        }

    def healthz(self):
        return {
            "status": "ok",
            "pid": self.health_pid,
            "actions_enabled": self.actions_enabled,
        }


class _Env:
    def __init__(self, tmp_path: Path, monitors: list[dict], *, official=False):
        self.tmp_path = tmp_path
        self.monitors = list(monitors)
        self.official = bool(official)
        self.calls = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("pi0_nav_pick must continue the initialized episode")

    def pi0_nav_pick_chunk_step(self, actions, *, chunk_index):
        array = np.asarray(actions, dtype=np.float32)
        self.calls.append((array.copy(), int(chunk_index)))
        monitor = dict(self.monitors.pop(0))
        info = {
            "done": {"success": self.official},
            "_rpent": {
                "executed_steps": monitor["executed_steps"],
                "total_env_steps": monitor["total_env_steps"],
                "pi0_nav_pick_monitor": monitor,
            },
        }
        return (
            _observation(marker=int(chunk_index)),
            0.0,
            self.official,
            False,
            info,
        )


def _primitives(tmp_path: Path, env: _Env, model: _Model) -> BehaviorPrimitives:
    return BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=DEFAULT_ACTION_CHUNK,
        output_dir=tmp_path,
        initial_observation=_observation(),
        initial_info={
            "done": {"success": False},
            "_rpent": {"total_env_steps": 0},
        },
    )


def test_pi0_nav_pick_primitive_schema_exposes_only_instruction(tmp_path):
    properties = PI0_NAV_PICK_SPEC["input_schema"]["properties"]
    assert set(properties) <= {"instruction"}
    assert not {
        "hand",
        "gripper_closed_threshold",
        "required_closed_frames",
        "max_chunks",
    } & set(properties)
    assert PI0_NAV_PICK_SPEC["input_schema"]["additionalProperties"] is False

    signature = inspect.signature(BehaviorPrimitives.pi0_nav_pick)
    public_parameters = {name for name in signature.parameters if name != "self"}
    assert public_parameters <= {"instruction"}

    toolkit = BehaviorToolkit(
        control_mode=PI0_NAV_PICK_VLA_MODE,
        primitives_kwargs={
            "env": object(),
            "model": object(),
            "max_episode_steps": 24756,
            "output_dir": tmp_path,
        },
    )
    assert [spec["name"] for spec in toolkit.get_tools_spec()] == [
        "pi0_nav_pick",
        *POST_PICK_TOOL_NAMES,
    ]


def test_checkpoint_tools_are_exposed_only_in_nav_pick_and_hybrid_modes(tmp_path):
    primitives_kwargs = {
        "env": object(),
        "model": object(),
        "max_episode_steps": 24756,
        "output_dir": tmp_path,
    }
    nav_pick_names = [
        spec["name"]
        for spec in BehaviorToolkit(
            control_mode=PI0_NAV_PICK_VLA_MODE,
            primitives_kwargs=primitives_kwargs,
        ).get_tools_spec()
    ]
    hybrid_names = [
        spec["name"]
        for spec in BehaviorToolkit(
            control_mode=HYBRID_VLM_PI0_MODE,
            primitives_kwargs=primitives_kwargs,
            planner_client=object(),
        ).get_tools_spec()
    ]
    assert nav_pick_names == [
        "pi0_nav_pick",
        *POST_PICK_TOOL_NAMES,
    ]
    assert hybrid_names == list(HYBRID_TOOL_NAMES)

    isolated_modes = {
        FULL_TASK_VLA_MODE: {"run_full_task"},
        PI0_PICK_VLA_MODE: {"pi0_pick"},
        PLANNER_TOOLS_MODE: set(PLANNER_TOOL_NAMES),
    }
    for mode, expected_names in isolated_modes.items():
        kwargs = {"control_mode": mode}
        if mode == PLANNER_TOOLS_MODE:
            kwargs["planner_client"] = object()
        else:
            kwargs["primitives_kwargs"] = primitives_kwargs
        names = {spec["name"] for spec in BehaviorToolkit(**kwargs).get_tools_spec()}
        assert names == expected_names
        assert "save_robot_state_checkpoint" not in names
        assert "restore_robot_state_checkpoint" not in names


def test_robot_state_checkpoint_schemas_are_explicit_and_path_safe():
    save = SAVE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"]
    assert set(save["properties"]) == {
        "checkpoint_name",
        "stage",
        "held_hand",
        "press_hand",
        "object_name",
        "require_current_grasp",
        "visual_review",
    }
    assert save["required"] == ["held_hand", "press_hand"]
    assert save["properties"]["checkpoint_name"]["default"] == ("state_checkpoint_1")
    assert save["properties"]["stage"]["default"] == "post_pi0_nav_pick"
    assert save["properties"]["object_name"]["default"] == "radio"
    assert save["properties"]["held_hand"]["enum"] == ["left", "right"]
    assert save["properties"]["press_hand"]["enum"] == ["left", "right"]
    assert save["additionalProperties"] is False

    restore = RESTORE_ROBOT_STATE_CHECKPOINT_SPEC["input_schema"]
    assert set(restore["properties"]) == {
        "checkpoint_name",
        "checkpoint_path",
        "mode",
        "keep_held_gripper_closed",
        "require_object_still_held",
        "timeout_s",
    }
    assert restore["properties"]["checkpoint_path"]["type"] == [
        "string",
        "null",
    ]
    assert restore["properties"]["mode"]["enum"] == ["plan_and_execute"]
    assert restore["properties"]["keep_held_gripper_closed"]["default"] is True
    assert restore["properties"]["require_object_still_held"]["default"] is True
    assert restore["additionalProperties"] is False


def test_checkpoint_primitives_forward_explicit_contract_and_isolate_official_success(
    tmp_path,
):
    checkpoint_path = tmp_path / "state_checkpoints" / "state_checkpoint_1.json"

    class Env:
        def __init__(self):
            self.save_kwargs = None
            self.restore_kwargs = None

        def save_robot_state_checkpoint(self, **kwargs):
            self.save_kwargs = kwargs
            return {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "saved_robot_state_checkpoint",
                "checkpoint_name": kwargs["checkpoint_name"],
                "checkpoint_path": str(checkpoint_path),
                "held_hand": kwargs["held_hand"],
                "press_hand": kwargs["press_hand"],
                "object_name": kwargs["object_name"],
                "info": {"done": {"success": False}},
            }

        def restore_robot_state_checkpoint(self, **kwargs):
            self.restore_kwargs = kwargs
            return {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "restored_robot_state_checkpoint",
                "checkpoint_name": kwargs["checkpoint_name"],
                "checkpoint_path": str(checkpoint_path),
                "held_hand": "right",
                "press_hand": "left",
                "object_name": "radio",
                "metrics": {
                    "joint_error_max_rad": 0.001,
                    "base_error_m": 0.002,
                    "base_yaw_error_rad": 0.003,
                    "held_object_drift_m": 0.004,
                    "held_gripper_opening_m": 0.02,
                },
                "info": {"done": {"success": False}},
            }

    env = Env()
    primitives = BehaviorPrimitives(env=env, output_dir=tmp_path)
    saved = primitives.save_robot_state_checkpoint(
        held_hand="right",
        press_hand="left",
    )
    restored = primitives.restore_robot_state_checkpoint()

    assert env.save_kwargs == {
        "checkpoint_name": "state_checkpoint_1",
        "stage": "post_pi0_nav_pick",
        "held_hand": "right",
        "press_hand": "left",
        "object_name": "radio",
        "require_current_grasp": True,
        "visual_review": True,
    }
    assert env.restore_kwargs == {
        "checkpoint_name": "state_checkpoint_1",
        "checkpoint_path": None,
        "mode": "plan_and_execute",
        "keep_held_gripper_closed": True,
        "require_object_still_held": True,
        "timeout_s": 90.0,
    }
    for result, reason in (
        (saved, "saved_robot_state_checkpoint"),
        (restored, "restored_robot_state_checkpoint"),
    ):
        assert result["primitive_success"] is True
        assert result["task_success"] is False
        assert result["_finish"] is False
        assert result["stop_reason"] == reason
        assert result["checkpoint_name"] == "state_checkpoint_1"
        assert result["held_hand"] == "right"
        assert result["press_hand"] == "left"
        assert result["object_name"] == "radio"
    assert set(restored["metrics"]) == {
        "joint_error_max_rad",
        "base_error_m",
        "base_yaw_error_rad",
        "held_object_drift_m",
        "held_gripper_opening_m",
    }


def test_checkpoint_primitives_reject_unsafe_or_non_guarded_inputs(tmp_path):
    primitives = BehaviorPrimitives(env=object(), output_dir=tmp_path)

    with pytest.raises(ValueError, match="must be different"):
        primitives.save_robot_state_checkpoint(
            held_hand="right",
            press_hand="right",
        )
    with pytest.raises(ValueError, match="keep_held_gripper_closed"):
        primitives.restore_robot_state_checkpoint(
            keep_held_gripper_closed=False,
        )
    with pytest.raises(ValueError, match="require_object_still_held"):
        primitives.restore_robot_state_checkpoint(
            require_object_still_held=False,
        )


def test_env_client_exposes_nav_pick_and_checkpoint_rpc_contracts():
    class Rpc:
        def __init__(self):
            self.calls = []

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            self.calls.append((method, args, kwargs or {}, timeout_s))
            if method == "env.get_env_meta":
                return {"control_mode": PI0_NAV_PICK_VLA_MODE}
            if method == "env.pi0_nav_pick_chunk_step":
                return (
                    _observation(),
                    0.0,
                    False,
                    False,
                    {
                        "done": {"success": False},
                        "_rpent": {"total_env_steps": DEFAULT_ACTION_CHUNK},
                    },
                )
            return {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "ok",
            }

    rpc = Rpc()
    client = BehaviorEnvClient(
        rpc,
        expected_meta={"control_mode": PI0_NAV_PICK_VLA_MODE},
    )
    actions = np.zeros((DEFAULT_ACTION_CHUNK, ACTION_DIM), dtype=np.float32)
    client.pi0_nav_pick_chunk_step(actions, chunk_index=7)
    client.save_robot_state_checkpoint(
        checkpoint_name="state_checkpoint_1",
        stage="post_pi0_nav_pick",
        held_hand="right",
        press_hand="left",
        object_name="radio",
        require_current_grasp=True,
        visual_review=True,
    )
    client.restore_robot_state_checkpoint(
        checkpoint_name="state_checkpoint_1",
        checkpoint_path=None,
        mode="plan_and_execute",
        keep_held_gripper_closed=True,
        require_object_still_held=True,
        timeout_s=90.0,
    )

    assert rpc.calls[1][0] == "env.pi0_nav_pick_chunk_step"
    assert len(rpc.calls[1][1][0]) == DEFAULT_ACTION_CHUNK
    assert rpc.calls[1][2] == {"chunk_index": 7}
    assert rpc.calls[2][0] == "env.save_robot_state_checkpoint"
    assert rpc.calls[2][2] == {
        "checkpoint_name": "state_checkpoint_1",
        "stage": "post_pi0_nav_pick",
        "held_hand": "right",
        "press_hand": "left",
        "object_name": "radio",
        "require_current_grasp": True,
        "visual_review": True,
    }
    assert rpc.calls[3][0] == "env.restore_robot_state_checkpoint"
    assert rpc.calls[3][2] == {
        "checkpoint_name": "state_checkpoint_1",
        "checkpoint_path": None,
        "mode": "plan_and_execute",
        "keep_held_gripper_closed": True,
        "require_object_still_held": True,
        "timeout_s": 90.0,
    }


def test_robot_motion_checkpoint_payload_is_json_allowlisted_without_sim_state(
    tmp_path,
):
    class Pose:
        def __init__(self, position):
            self.position = np.asarray(position, dtype=np.float64)
            self.name = "radio_receiver"
            self.prim_path = "/World/radio_receiver"

        def get_position_orientation(self):
            return self.position.copy(), np.array([0.0, 0.0, 0.0, 1.0])

    class Robot(Pose):
        base_control_idx = [0, 1, 2]
        trunk_control_idx = [3, 4, 5, 6]
        arm_control_idx = {
            "left": list(range(7, 14)),
            "right": list(range(14, 21)),
        }

        def __init__(self):
            super().__init__([1.0, 2.0, 0.0])
            self.eef_links = {
                "left": Pose([1.0, 2.0, 1.0]),
                "right": Pose([1.1, 2.0, 1.0]),
            }

        @staticmethod
        def get_joint_positions():
            return np.linspace(0.0, 0.29, 30, dtype=np.float64)

    radio = Pose([1.0, 2.0, 0.4])
    robot = Robot()
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = HYBRID_VLM_PI0_MODE
    facade._env_steps = 123
    facade._gripper_latch = {"left": 1.0, "right": -1.0}
    facade._last_observation = _observation()
    facade._last_info = {"done": {"success": False, "predicate": {"raw": 7}}}
    facade._handoff_validator_frames = []
    facade._initial_radio_position = np.array([1.0, 2.0, 0.3])
    facade._validator_trace_path = tmp_path / "handoff_validator_trace.json"
    facade._resolve_handoff_targets = lambda: (radio, object())
    facade._robot = lambda: robot
    facade._validated_selected_gripper_opening = lambda *, observation, hand: (
        0.02 if hand == "right" else 0.1
    )

    payload = facade._robot_state_checkpoint_payload(
        checkpoint_name="state_checkpoint_1",
        stage="post_pi0_nav_pick",
        held_hand="right",
        press_hand="left",
        object_name="radio",
        require_current_grasp=False,
        validation_evidence={"strict_grasp": True},
    )

    assert set(payload) == {
        "schema_version",
        "kind",
        "not_simulator_restore",
        "checkpoint_name",
        "stage",
        "strict_local_grasp_success",
        "usable_post_pick_saved",
        "save_policy",
        "warnings",
        "env_step",
        "object_name",
        "held_hand",
        "press_hand",
        "robot",
        "poses",
        "validation",
        "visual_evidence",
    }
    assert payload["schema_version"] == 1
    assert payload["kind"] == "robot_motion_checkpoint"
    assert payload["not_simulator_restore"] is True
    assert payload["checkpoint_name"] == "state_checkpoint_1"
    assert payload["env_step"] == 123
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert "simulator_state" not in serialized
    assert "dump_state" not in serialized
    assert "tensor" not in serialized
    assert "predicate" not in serialized
    official = payload["validation"].get("official_task_success")
    assert official is None or isinstance(official, bool)

    facade._control_mode = PI0_NAV_PICK_VLA_MODE
    facade._handoff_validator_frame = lambda _observation: {
        "per_hand": {
            "right": {
                "instantaneous_pass": False,
                "criteria": {"opening_strict": False},
            },
            "left": {"instantaneous_pass": False, "criteria": {}},
        }
    }
    warning_payload = facade._robot_state_checkpoint_payload(
        checkpoint_name="state_checkpoint_1",
        stage="post_pick_pre_controller_reload",
        held_hand="right",
        press_hand="left",
        object_name="radio",
        require_current_grasp=True,
        validation_evidence=None,
    )
    assert warning_payload["strict_local_grasp_success"] is False
    assert warning_payload["usable_post_pick_saved"] is True
    assert warning_payload["save_policy"] == (
        "debug_save_physics_warnings_non_blocking"
    )
    assert [item["code"] for item in warning_payload["warnings"]] == [
        "current_grasp_not_strict"
    ]

    facade._control_mode = HYBRID_VLM_PI0_MODE

    facade._output_dir = tmp_path
    facade._state_checkpoint_path = (
        tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    )
    saved = facade.save_robot_state_checkpoint(
        checkpoint_name="state_checkpoint_1",
        stage="post_pi0_nav_pick",
        held_hand="right",
        press_hand="left",
        object_name="radio",
        require_current_grasp=False,
        visual_review=False,
    )
    assert saved["primitive_success"] is True
    assert saved["task_success"] is False
    assert saved["stop_reason"] == "saved_robot_state_checkpoint"
    assert saved["checkpoint_name"] == "state_checkpoint_1"
    assert saved["checkpoint_path"] == str(
        tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    )
    assert saved["held_hand"] == "right"
    assert saved["press_hand"] == "left"
    assert saved["object_name"] == "radio_receiver"
    checkpoint = json.loads(Path(saved["checkpoint_path"]).read_text(encoding="utf-8"))
    assert checkpoint["visual_evidence"] == {}

    with pytest.raises(RuntimeError, match="immutable and already exists"):
        BehaviorPrimitives(env=facade, output_dir=tmp_path).save_robot_state_checkpoint(
            held_hand="right",
            press_hand="left",
            require_current_grasp=False,
            visual_review=False,
        )


def test_nav_pick_handoff_commits_only_json_and_restore_never_uses_simulator_state():
    handoff_source = inspect.getsource(BehaviorEnvFacade._complete_pi0_nav_pick_handoff)
    restore_source = inspect.getsource(BehaviorEnvFacade.restore_robot_state_checkpoint)

    assert "state_checkpoint_1.json" in handoff_source
    assert "checkpoint_post_pick" not in handoff_source
    assert "dump_simulator_state" not in handoff_source
    assert "restore_simulator_state" not in restore_source
    assert "torch.load" not in restore_source
    assert ".pt" not in restore_source


def test_nav_pick_handoff_prepares_only_target_specific_prepress_generator():
    calls = []

    class Planner:
        report = {
            "status": "complete",
            "generator_kind": "prepress_arm",
            "held_hand": "right",
            "base_generator_warmed": False,
            "unrelated_press_arm_generator_warmed": False,
            "attached_collision_body": {"root_matches_expected_radio": True},
            "robot_q_pose_jump_max": 0.0,
            "stages": {
                "current_q_attached_combined_collision": {"ok": True},
                "current_pose_attached_full_trajectory": {"ok": True},
                "identity_neighborhood_connected_path": {"ok": True},
            },
        }

        @staticmethod
        def on_simulator_state_restored():
            calls.append(("cache_reset",))

        @staticmethod
        def warmup_prepress(*, hand, expected_attached_root):
            calls.append(("warmup_prepress", hand, expected_attached_root))
            return dict(Planner.report)

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._control_mode = PI0_NAV_PICK_VLA_MODE
    facade._planner = Planner()
    radio_root = object()
    facade._resolve_handoff_targets = lambda: (
        type("Radio", (), {"root_link": radio_root})(),
        object(),
    )

    report = facade._prepare_prepress_planner_readiness(held_hand="right")

    assert calls == [
        ("cache_reset",),
        ("warmup_prepress", "right", radio_root),
    ]
    assert report == Planner.report

    Planner.report = {**Planner.report, "status": "error"}
    with pytest.raises(RuntimeError, match="readiness report"):
        facade._prepare_prepress_planner_readiness(held_hand="right")

    with pytest.raises(ValueError, match="held_hand"):
        facade._prepare_prepress_planner_readiness(held_hand="center")


def test_restore_checkpoint_rejects_unsafe_paths_and_non_robot_json(tmp_path):
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._output_dir = tmp_path
    facade._state_checkpoint_path = (
        tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    )
    facade._base_controller_mode = "position"

    with pytest.raises(ValueError, match="checkpoint.*path|safe"):
        facade.restore_robot_state_checkpoint(
            checkpoint_name="state_checkpoint_1",
            checkpoint_path="/tmp/outside-rpent-checkpoint.json",
            mode="plan_and_execute",
            keep_held_gripper_closed=True,
            require_object_still_held=True,
            timeout_s=90.0,
        )

    simulator_checkpoint = tmp_path / "state_checkpoints" / "checkpoint_post_pick.pt"
    simulator_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    simulator_checkpoint.write_bytes(b"not-a-robot-motion-checkpoint")
    with pytest.raises(ValueError, match="checkpoint.*path|safe"):
        facade.restore_robot_state_checkpoint(
            checkpoint_name="state_checkpoint_1",
            checkpoint_path=str(simulator_checkpoint),
            mode="plan_and_execute",
            keep_held_gripper_closed=True,
            require_object_still_held=True,
            timeout_s=90.0,
        )

    outside = tmp_path / "outside-checkpoint.json"
    outside.write_text("{}", encoding="utf-8")
    linked = tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    linked.parent.mkdir(parents=True, exist_ok=True)
    linked.symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link|resolve"):
        facade.restore_robot_state_checkpoint(
            checkpoint_name="state_checkpoint_1",
            checkpoint_path=None,
            mode="plan_and_execute",
            keep_held_gripper_closed=True,
            require_object_still_held=True,
            timeout_s=90.0,
        )
    linked.unlink()

    invalid = tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "simulator_snapshot",
                "not_simulator_restore": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checkpoint schema"):
        facade.restore_robot_state_checkpoint(
            checkpoint_name="state_checkpoint_1",
            checkpoint_path=None,
            mode="plan_and_execute",
            keep_held_gripper_closed=True,
            require_object_still_held=True,
            timeout_s=90.0,
        )


def _restore_facade(tmp_path, *, failure: str | None = None, official=False):
    class Pose:
        def __init__(self, position):
            self.position = np.asarray(position, dtype=np.float64)

        def get_position_orientation(self):
            return self.position.copy(), np.array([0.0, 0.0, 0.0, 1.0])

    class Robot:
        def __init__(self):
            self.q = np.zeros(30, dtype=np.float64)
            self.eef_links = {
                "left": Pose([0.2, 0.0, 1.0]),
                "right": Pose([0.0, 0.0, 1.0]),
            }

        def get_joint_positions(self):
            return self.q.copy()

    target = np.full(21, 0.01, dtype=np.float64)
    robot = Robot()
    radio = Pose([0.0, 0.0, 0.4])

    class Backend:
        @staticmethod
        def _generator(*, kind, hand):
            assert (kind, hand) == ("arm", "right")
            return object()

        @staticmethod
        def _check_q_trajectory_collisions(generator, q_path, *, attached_obj):
            assert generator is not None
            assert np.asarray(q_path).shape[1] == 30
            assert attached_obj is radio
            return {"available": True, "colliding": False}

        @staticmethod
        def joint_target_to_action(waypoint, *, hand):
            assert np.asarray(waypoint).shape == (30,)
            assert hand is None
            return np.zeros(ACTION_DIM, dtype=np.float32)

    class DirectProcess:
        def __init__(self):
            self.actions = []

        def step_env(self, action, *, need_obs):
            assert need_obs is True
            self.actions.append(action.detach().cpu().numpy().reshape(ACTION_DIM))
            robot.q[:21] = target
            if failure == "held_object_drift":
                radio.position[0] += 0.02
            return object(), 0.0, False, False, [{"done": {"success": bool(official)}}]

    class Env:
        def __init__(self):
            self._direct_process = DirectProcess()

        @staticmethod
        def _wrap_obs(_step_obs):
            return {
                "main_images": np.zeros((1, 4, 6, 3), dtype=np.uint8),
                "wrist_images": np.zeros((1, 2, 4, 6, 3), dtype=np.uint8),
                "states": np.zeros((1, 256), dtype=np.float32),
                "task_descriptions": ["Turn on the radio receiver."],
            }

    checkpoint = {
        "schema_version": 1,
        "kind": "robot_motion_checkpoint",
        "not_simulator_restore": True,
        "checkpoint_name": "state_checkpoint_1",
        "held_hand": "right",
        "press_hand": "left",
        "object_name": "radio",
        "robot": {
            "q_space_target": {
                "indices": list(range(21)),
                "values": target.tolist(),
            }
        },
        "poses": {
            "object_pose_world": {
                "position": [0.0, 0.0, 0.4],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "object_pose_in_held_eef": {
                "position": [0.0, 0.0, -0.6],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
    }
    checkpoint_path = tmp_path / "state_checkpoints" / "state_checkpoint_1.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    env = Env()
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._output_dir = tmp_path
    facade._base_controller_mode = "position"
    facade._env_steps = 0
    facade._last_info = {"done": {"success": False}}
    facade._last_observation = _observation()
    facade._gripper_latch = {"left": 1.0, "right": -1.0}
    facade._env = env
    facade._robot = lambda: robot
    facade._require_planner = lambda: type("Planner", (), {"backend": Backend()})()
    facade._resolve_handoff_targets = lambda: (radio, object())
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade._append_video = lambda *_args, **_kwargs: None
    facade.dump_simulator_state = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("robot checkpoint restore must not read simulator state")
    )
    facade.restore_simulator_state = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("robot checkpoint restore must not restore simulator state")
    )
    facade._validated_selected_gripper_opening = lambda *, observation, hand: (
        0.02 if hand == "right" else 0.1
    )
    facade._attachment_matches = lambda hand, _radio: (
        (hand == "right" and failure != "held_object_lost"),
        False,
    )
    facade._hand_target_contact_report = lambda hand, _position: {
        "available": True,
        "target_contact_count": int(
            hand == "left" and failure == "press_hand_contacted_object"
        ),
        "target_two_finger_contact": hand == "right",
    }
    return facade, env


def test_restore_checkpoint_executes_guarded_motion_and_reports_metrics(tmp_path):
    facade, env = _restore_facade(tmp_path)

    result = facade.restore_robot_state_checkpoint(
        checkpoint_name="state_checkpoint_1",
        checkpoint_path=None,
        mode="plan_and_execute",
        keep_held_gripper_closed=True,
        require_object_still_held=True,
        timeout_s=90.0,
    )

    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["stop_reason"] == "restored_robot_state_checkpoint"
    assert result["held_hand"] == "right"
    assert result["press_hand"] == "left"
    assert result["object_name"] == "radio"
    assert set(result["metrics"]) == {
        "joint_error_max_rad",
        "base_error_m",
        "base_yaw_error_rad",
        "held_object_drift_m",
        "held_gripper_opening_m",
    }
    assert result["metrics"] == {
        "joint_error_max_rad": 0.0,
        "base_error_m": 0.0,
        "base_yaw_error_rad": 0.0,
        "held_object_drift_m": 0.0,
        "held_gripper_opening_m": 0.02,
    }
    assert len(env._direct_process.actions) == 1
    held_gripper = env._direct_process.actions[0][ENV_ACTION_SEGMENTS["right_gripper"]]
    np.testing.assert_array_equal(held_gripper, np.array([-1.0], dtype=np.float32))


@pytest.mark.parametrize(
    "failure",
    [
        "held_object_lost",
        "held_object_drift",
        "press_hand_contacted_object",
    ],
)
def test_restore_checkpoint_returns_structured_guard_failure(tmp_path, failure):
    facade, env = _restore_facade(tmp_path, failure=failure, official=True)

    result = facade.restore_robot_state_checkpoint(
        checkpoint_name="state_checkpoint_1",
        checkpoint_path=None,
        mode="plan_and_execute",
        keep_held_gripper_closed=True,
        require_object_still_held=True,
        timeout_s=90.0,
    )

    assert result["primitive_success"] is False
    assert result["task_success"] is (failure == "held_object_drift")
    assert result["stop_reason"] == failure
    assert set(result["metrics"]) == {
        "joint_error_max_rad",
        "base_error_m",
        "base_yaw_error_rad",
        "held_object_drift_m",
        "held_gripper_opening_m",
    }
    if failure in {"held_object_lost", "press_hand_contacted_object"}:
        assert env._direct_process.actions == []


def test_pi0_nav_pick_executes_one_complete_unmodified_32_by_23_chunk(tmp_path):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=True,
        held_hand="left",
        pose_jump_ok=True,
        checkpoint=True,
    )
    env = _Env(tmp_path, [monitor])
    model = _Model()

    result = _primitives(tmp_path, env, model).pi0_nav_pick(
        instruction="Turn on the radio receiver."
    )

    assert env.reset_calls == 0
    assert len(model.calls) == 1
    assert len(env.calls) == 1
    actions, chunk_index = env.calls[0]
    expected = np.arange(DEFAULT_ACTION_CHUNK * ACTION_DIM, dtype=np.float32).reshape(
        DEFAULT_ACTION_CHUNK, ACTION_DIM
    )
    assert actions.shape == (DEFAULT_ACTION_CHUNK, ACTION_DIM)
    np.testing.assert_array_equal(actions, expected)
    assert chunk_index == 1
    assert result["chunks_used"] == 1
    assert result["env_steps_used"] == DEFAULT_ACTION_CHUNK


def test_pi0_nav_pick_rejects_short_policy_chunk_before_any_env_action(tmp_path):
    env = _Env(
        tmp_path,
        [
            _monitor(
                tmp_path,
                total_env_steps=DEFAULT_ACTION_CHUNK,
                local_grasp_success=False,
                held_hand=None,
                pose_jump_ok=True,
                checkpoint=False,
            )
        ],
    )
    result = _primitives(
        tmp_path, env, _Model(horizon=DEFAULT_ACTION_CHUNK - 1)
    ).pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert env.calls == []
    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["state_checkpoint_path"] is None
    assert "checkpoint_post_pick_path" not in result
    assert result["error"] is not None
    assert "32" in result["error"]


def test_horizon_tail_still_submits_full_chunk_and_env_may_execute_last_20_steps(
    tmp_path,
):
    max_episode_steps = 24756
    initial_total = max_episode_steps - 20
    monitor = _monitor(
        tmp_path,
        total_env_steps=max_episode_steps,
        local_grasp_success=False,
        held_hand=None,
        pose_jump_ok=True,
        checkpoint=False,
        executed_steps=20,
    )
    env = _Env(tmp_path, [monitor])
    model = _Model()
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=max_episode_steps,
        output_dir=tmp_path,
        initial_observation=_observation(),
        initial_info={
            "done": {"success": False},
            "_rpent": {"total_env_steps": initial_total},
        },
    )

    result = primitives.pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert len(model.calls) == 1
    assert len(env.calls) == 1
    assert env.calls[0][0].shape == (DEFAULT_ACTION_CHUNK, ACTION_DIM)
    assert result["env_steps_used"] == 20
    assert result["total_env_steps"] == max_episode_steps
    assert result["stop_reason"] == "horizon"
    assert result["error"] is None


def test_tail_insufficient_handoff_horizon_preserves_exact_accounting_and_stop(
    tmp_path,
):
    max_episode_steps = 24756
    initial_total = max_episode_steps - 7

    class DirectProcess:
        @staticmethod
        def step_env(_action, *, need_obs):
            assert need_obs is True
            return object(), [0.0], False, False, [{"done": {"success": False}}]

    class SimulatorEnv:
        _direct_process = DirectProcess()

        @staticmethod
        def _wrap_obs(_step_obs):
            return {
                "main_images": np.zeros((1, 4, 6, 3), dtype=np.uint8),
                "wrist_images": np.zeros((1, 2, 4, 6, 3), dtype=np.uint8),
                "states": np.zeros((1, 256), dtype=np.float32),
                "task_descriptions": ["Turn on the radio receiver."],
            }

    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._done = False
    facade._control_mode = PI0_NAV_PICK_VLA_MODE
    facade._env_steps = initial_total
    facade._meta = {"max_episode_steps": max_episode_steps}
    facade._env = SimulatorEnv()
    facade._last_observation = _observation()
    facade._last_info = {"done": {"success": False}}
    facade._planner_video_interval_steps = 4
    facade._gripper_latch = {"left": 1.0, "right": -1.0}
    facade._handoff_state = "VLA_ACTIVE"
    facade._vla_actions_enabled = True
    facade._action_source = "pi0_vla"
    facade._validator_trace_path = tmp_path / "validator_trace.json"
    facade._record_rgbd_frames = lambda *_args, **_kwargs: None
    facade._append_video = lambda *_args, **_kwargs: None
    facade._finalize_video_segment = lambda: None
    facade._complete_pi0_nav_pick_handoff = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("handoff must not start without eight remaining steps")
    )
    facade._update_handoff_validator = lambda _observation: {
        "local_grasp_success": True,
        "held_hand": "right",
        "per_hand": {"right": {"held": True}, "left": {"held": False}},
        "current": {"per_hand": {}},
    }

    class Env:
        def __init__(self):
            self.calls = 0

        def pi0_nav_pick_chunk_step(self, actions, *, chunk_index):
            self.calls += 1
            assert self.calls == 1
            assert chunk_index == 1
            return facade._step_action_chunk(
                actions,
                observe_final=True,
                pi0_nav_pick=True,
            )

    class Model(_Model):
        def predict_action_batch(self, observation, *, mode):
            assert mode == "eval"
            self.calls.append(observation["task_descriptions"])
            return (
                np.zeros(
                    (DEFAULT_ACTION_CHUNK, ACTION_DIM),
                    dtype=np.float32,
                ),
                {"model_call": len(self.calls)},
            )

    env = Env()
    model = Model()
    primitives = BehaviorPrimitives(
        env=env,
        model=model,
        max_episode_steps=max_episode_steps,
        output_dir=tmp_path,
        initial_observation=_observation(),
        initial_info={
            "done": {"success": False},
            "_rpent": {"total_env_steps": initial_total},
        },
    )

    result = primitives.pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert env.calls == 1
    assert len(model.calls) == 1
    assert result["primitive_success"] is False
    assert result["local_grasp_success"] is True
    assert result["held_hand"] == "right"
    assert result["task_success"] is False
    assert result["stop_reason"] == "insufficient_handoff_horizon"
    assert result["env_steps_used"] == 1
    assert result["vla_env_steps_used"] == 1
    assert result["handoff_env_steps_used"] == 0
    assert result["total_env_steps"] == initial_total + 1
    assert result["error"] is None
    assert result["last_pi0_nav_pick_monitor"]["stop_reason"] == (
        "insufficient_handoff_horizon"
    )


def test_optional_monitor_visual_review_and_stop_reason_survive_public_allowlist(
    tmp_path,
):
    review_root = tmp_path / "visual_review" / "pi0_nav_pick" / "chunk_0001"
    review_root.mkdir(parents=True)
    views = {}
    for frame_id, camera in enumerate(("head", "left_wrist", "right_wrist"), start=1):
        image_path = review_root / f"{camera}.png"
        image_path.write_bytes(b"png")
        views[camera] = {"path": str(image_path), "frame_id": frame_id}
    metadata_path = review_root / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")
    review = {
        "capture_group_id": "capture-group-1",
        "metadata_path": str(metadata_path),
        "views": views,
    }
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=False,
        held_hand=None,
        pose_jump_ok=True,
        checkpoint=False,
    )
    monitor.update(
        {
            "handoff_state": "FAILED",
            "vla_actions_enabled": False,
            "stop_reason": "insufficient_handoff_horizon",
            "visual_review": review,
            "private_debug": "drop-me",
        }
    )

    result = _primitives(tmp_path, _Env(tmp_path, [monitor]), _Model()).pi0_nav_pick(
        instruction="Turn on the radio receiver."
    )

    assert result["stop_reason"] == "insufficient_handoff_horizon"
    public_monitor = result["last_pi0_nav_pick_monitor"]
    assert public_monitor["stop_reason"] == "insufficient_handoff_horizon"
    assert public_monitor["visual_review"] == review
    assert "private_debug" not in public_monitor
    assert result["visual_review"]["chunk_artifacts"] == [review]
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["last_pi0_nav_pick_monitor"]["visual_review"] == review
    assert persisted["last_pi0_nav_pick_monitor"]["stop_reason"] == (
        "insufficient_handoff_horizon"
    )


def test_dynamic_left_hand_strict_success_is_local_only_and_handoffs(tmp_path):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=True,
        held_hand="left",
        pose_jump_ok=True,
        checkpoint=True,
    )
    result = _primitives(
        tmp_path, _Env(tmp_path, [monitor], official=False), _Model()
    ).pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert result["held_hand"] == "left"
    assert result["local_grasp_success"] is True
    assert result["primitive_success"] is True
    assert result["task_success"] is False
    assert result["handoff_state"] == "PAUSED"
    assert result["action_source"] == "curobo"
    assert result["state_checkpoint_path"] == monitor["state_checkpoint_path"]
    assert Path(result["state_checkpoint_path"]).is_file()
    assert result["state_checkpoint_path"].endswith(
        "/state_checkpoints/state_checkpoint_1.json"
    )
    assert "checkpoint_post_pick_path" not in result
    assert not (tmp_path / "checkpoint_post_pick.pt").exists()
    assert result["vla_endpoint"] == "http://127.0.0.1:45678"
    assert result["vla_disable_confirmation"]["pid"] == 123
    assert result["vla_health_after_disable"]["pid"] == 123
    paused = json.loads(Path(result["paused_runtime_path"]).read_text(encoding="utf-8"))
    assert paused["vla_endpoint"] == "http://127.0.0.1:45678"
    assert paused["vla_pid"] == 123


@pytest.mark.parametrize(
    ("disable_pid", "health_pid"),
    [(123, 456), (0, 0), (-1, -1)],
)
def test_vla_handoff_requires_same_positive_server_pid(
    tmp_path,
    disable_pid,
    health_pid,
):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=True,
        held_hand="right",
        pose_jump_ok=True,
        checkpoint=True,
    )
    model = _Model(disable_pid=disable_pid, health_pid=health_pid)
    result = _primitives(tmp_path, _Env(tmp_path, [monitor]), model).pi0_nav_pick(
        instruction="Turn on the radio receiver."
    )

    assert result["primitive_success"] is False
    assert result["vla_actions_disabled"] is False
    assert result["error"] is not None
    assert "pid" in result["error"].lower()


def test_official_success_never_fabricates_local_grasp_or_checkpoint(tmp_path):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=False,
        held_hand=None,
        pose_jump_ok=True,
        checkpoint=False,
    )
    result = _primitives(
        tmp_path, _Env(tmp_path, [monitor], official=True), _Model()
    ).pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert result["task_success"] is True
    assert result["local_grasp_success"] is False
    assert result["primitive_success"] is False
    assert result["state_checkpoint_path"] is None
    assert "checkpoint_post_pick_path" not in result
    assert result["handoff_state"] != "PAUSED"


def test_pose_jump_failure_never_creates_post_pick_checkpoint(tmp_path):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=False,
        held_hand="right",
        pose_jump_ok=False,
        checkpoint=False,
    )
    result = _primitives(
        tmp_path, _Env(tmp_path, [monitor], official=False), _Model()
    ).pi0_nav_pick(instruction="Turn on the radio receiver.")

    assert result["local_grasp_success"] is False
    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["state_checkpoint_path"] is None
    assert "checkpoint_post_pick_path" not in result
    assert not (tmp_path / "checkpoint_post_pick.pt").exists()
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
    assert persisted["state_checkpoint_path"] is None
    assert "checkpoint_post_pick_path" not in persisted


@pytest.mark.parametrize(
    ("handoff_state", "action_source", "expected_error"),
    [
        ("VLA_ACTIVE", "curobo", "handoff_state"),
        ("PAUSED", "pi0_vla", "action_source"),
    ],
)
def test_success_requires_exact_paused_curobo_handoff_state(
    tmp_path,
    handoff_state,
    action_source,
    expected_error,
):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=True,
        held_hand="right",
        pose_jump_ok=True,
        checkpoint=True,
    )
    monitor["handoff_state"] = handoff_state
    monitor["action_source"] = action_source

    result = _primitives(tmp_path, _Env(tmp_path, [monitor]), _Model()).pi0_nav_pick(
        instruction="Turn on the radio receiver."
    )

    assert result["primitive_success"] is False
    assert result["vla_actions_disabled"] is False
    assert result["error"] is not None
    assert expected_error in result["error"]


def test_action_source_is_fail_closed_to_pi0_vla(tmp_path):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=False,
        held_hand=None,
        pose_jump_ok=True,
        checkpoint=False,
    )
    monitor["action_source"] = "planner"

    result = _primitives(tmp_path, _Env(tmp_path, [monitor]), _Model()).pi0_nav_pick(
        instruction="Turn on the radio receiver."
    )

    assert result["primitive_success"] is False
    assert result["task_success"] is False
    assert result["error"] is not None
    assert "action_source" in result["error"]


def test_persisted_monitor_json_uses_a_public_field_allowlist(tmp_path):
    monitor = _monitor(
        tmp_path,
        total_env_steps=DEFAULT_ACTION_CHUNK,
        local_grasp_success=False,
        held_hand=None,
        pose_jump_ok=True,
        checkpoint=False,
    )
    monitor["private_scene_object_pose"] = [1.0, 2.0, 3.0]
    monitor["untrusted_debug_payload"] = {"secret": "must-not-persist"}

    result = _primitives(tmp_path, _Env(tmp_path, [monitor]), _Model()).pi0_nav_pick(
        instruction="Turn on the radio receiver."
    )
    public_monitor = result["last_pi0_nav_pick_monitor"]
    persisted = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))[
        "last_pi0_nav_pick_monitor"
    ]

    expected_fields = {
        "executed_steps",
        "handoff_env_steps",
        "total_env_steps",
        "local_grasp_success",
        "held_hand",
        "per_hand",
        "current_criteria",
        "validator_trace_path",
        "state_checkpoint_path",
        "handoff_state",
        "action_source",
        "vla_actions_enabled",
        "paused_runtime_path",
    }
    assert set(public_monitor) == expected_fields
    assert set(persisted) == expected_fields


def _validator_frames(
    *,
    held_hand: str,
    instantaneous_pass: bool = True,
    relative_jump_m: float = 0.0,
) -> list[dict]:
    frames = []
    for index in range(HANDOFF_VALIDATION_FRAMES):
        radio = np.array([0.004 * index, 0.0, 0.20], dtype=np.float64)
        frame = {
            "total_env_steps": index + 1,
            "capture_step": index + 1,
            "per_hand": {},
        }
        for hand in ("left", "right"):
            eef = radio - np.array([0.0, 0.0, 0.04], dtype=np.float64)
            if hand == held_hand and index == HANDOFF_VALIDATION_FRAMES - 1:
                eef[1] += relative_jump_m
            frame["per_hand"][hand] = {
                "radio_position": radio.tolist(),
                "eef_position": eef.tolist(),
                "instantaneous_pass": (
                    bool(instantaneous_pass) if hand == held_hand else False
                ),
                "criteria": {
                    "opening_strict": (
                        bool(instantaneous_pass) if hand == held_hand else False
                    )
                },
            }
        frames.append(frame)
    return frames


def _current_validator_frame(
    *,
    lift_m=0.04,
    support_gap_m=0.03,
    opening_m=np.nextafter(0.045, 0.0),
    selected_attachment=True,
    selected_backend=False,
    selected_two_finger=False,
    inactive_assisted=False,
    inactive_backend=False,
    inactive_contacts=0,
):
    class Pose:
        def __init__(self, position):
            self.position = np.asarray(position, dtype=np.float64)

        def get_position_orientation(self):
            return self.position.copy(), np.array([0.0, 0.0, 0.0, 1.0])

    radio = Pose([0.0, 0.0, float(lift_m)])
    table = object()
    robot = type(
        "Robot",
        (),
        {
            "eef_links": {
                "left": Pose([-0.1, 0.0, 0.2]),
                "right": Pose([0.0, 0.0, 0.2]),
            }
        },
    )()
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 17
    facade._last_capture_step = 17
    facade._initial_radio_position = np.zeros(3, dtype=np.float64)
    facade._resolve_handoff_targets = lambda: (radio, table)
    facade._robot = lambda: robot
    facade._object_vertical_bounds = lambda obj: (
        (float(support_gap_m), float(support_gap_m) + 0.02)
        if obj is radio
        else (-0.02, 0.0)
    )
    facade._radio_table_mask_report = lambda _radio, _table: {
        "passed": True,
        "gap_px": 4.0,
    }
    facade._validated_selected_gripper_opening = lambda *, observation, hand: (
        float(opening_m) if hand == "right" else 0.1
    )

    def attachment(hand, _radio):
        if hand == "right":
            return bool(selected_attachment), bool(selected_backend)
        return bool(inactive_assisted), bool(inactive_backend)

    def contact(hand, _position):
        if hand == "right":
            return {
                "available": True,
                "target_contact_count": 2 if selected_two_finger else 0,
                "target_two_finger_contact": bool(selected_two_finger),
            }
        return {
            "available": True,
            "target_contact_count": int(inactive_contacts),
            "target_two_finger_contact": False,
        }

    facade._attachment_matches = attachment
    facade._hand_target_contact_report = contact
    return facade._handoff_validator_frame(_observation())


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"selected_attachment": True, "selected_two_finger": False}, True),
        (
            {
                "selected_attachment": False,
                "selected_backend": True,
                "selected_two_finger": False,
            },
            True,
        ),
        ({"selected_attachment": False, "selected_two_finger": True}, True),
        ({"selected_attachment": False, "selected_two_finger": False}, False),
        ({"opening_m": 0.045}, False),
        ({"lift_m": np.nextafter(0.04, 0.0)}, False),
        ({"support_gap_m": np.nextafter(0.03, 0.0)}, False),
        ({"inactive_assisted": True}, False),
        ({"inactive_backend": True}, False),
        ({"inactive_contacts": 1}, False),
    ],
)
def test_current_handoff_validator_uses_confirmed_inclusive_geometry_and_grasp_evidence(
    overrides,
    expected,
):
    assert HANDOFF_RADIO_LIFT_MIN_M == 0.04
    assert HANDOFF_SUPPORT_GAP_MIN_M == 0.03
    assert HANDOFF_GRIPPER_OPENING_MAX == 0.045

    selected = _current_validator_frame(**overrides)["per_hand"]["right"]

    assert selected["instantaneous_pass"] is expected
    assert selected["criteria"]["opening_strict"] is (
        bool(overrides.get("opening_m", np.nextafter(0.045, 0.0)) < 0.045)
    )
    assert selected["criteria"]["selected_attachment_or_two_finger_contact"] is bool(
        overrides.get("selected_attachment", True)
        or overrides.get("selected_backend", False)
        or overrides.get("selected_two_finger", False)
    )
    if expected:
        assert all(selected["criteria"].values())


def test_post_reload_joint_settling_is_warning_only_for_state1_handoff():
    assert HANDOFF_ARTICULATION_ERROR_MAX_RAD == 0.05
    metrics = {
        "base_xy_error_m": 0.0,
        "base_yaw_error_rad": 0.0,
        "articulation_error_rad": np.nextafter(0.05, np.inf),
    }

    assert _handoff_controller_hold_warning(metrics) is True
    assert (
        _handoff_held_object_stable(
            relative_drift_m=HANDOFF_RELATIVE_DRIFT_MAX_M,
            angular_drift_rad=0.15,
        )
        is True
    )


def test_post_reload_held_radio_drift_remains_hard_handoff_gate():
    assert (
        _handoff_held_object_stable(
            relative_drift_m=np.nextafter(HANDOFF_RELATIVE_DRIFT_MAX_M, np.inf),
            angular_drift_rad=0.0,
        )
        is False
    )


def test_post_reload_lift_allows_two_mm_settling_only_after_strict_pick():
    assert HANDOFF_POST_RELOAD_LIFT_TOLERANCE_M == 0.002
    selected = {
        "radio_lift_m": 0.03989601135253906,
        "criteria": {
            "opening_strict": True,
            "radio_lift": False,
            "support_gap": True,
            "selected_attachment_or_two_finger_contact": True,
            "other_hand_no_assisted_attachment": True,
            "other_hand_no_backend_attachment": True,
            "other_hand_no_radio_contact": True,
        },
    }
    other = {"instantaneous_pass": False}

    assert _post_reload_grasp_stable(selected=selected, other=other) is True
    selected["radio_lift_m"] = np.nextafter(0.038, 0.0)
    assert _post_reload_grasp_stable(selected=selected, other=other) is False


def _window_frames(
    *,
    radio_motion_m=0.008,
    eef_motion_m=0.008,
    relative_y=None,
    failed_opening_frame=None,
):
    if relative_y is None:
        relative_y = [0.0] * HANDOFF_VALIDATION_FRAMES
    assert len(relative_y) == HANDOFF_VALIDATION_FRAMES
    frames = []
    for index in range(HANDOFF_VALIDATION_FRAMES):
        fraction = index / (HANDOFF_VALIDATION_FRAMES - 1)
        radio = np.array([radio_motion_m * fraction, 0.0, 0.2])
        eef = np.array([eef_motion_m * fraction, -float(relative_y[index]), 0.16])
        opening_closed = index != failed_opening_frame
        frames.append(
            {
                "total_env_steps": index + 1,
                "capture_step": index + 1,
                "per_hand": {
                    "right": {
                        "radio_position": radio.tolist(),
                        "eef_position": eef.tolist(),
                        "opening": 0.02 if opening_closed else 0.045,
                        "instantaneous_pass": opening_closed,
                        "criteria": {"opening_strict": opening_closed},
                    },
                    "left": {
                        "radio_position": radio.tolist(),
                        "eef_position": (eef + [0.1, 0.0, 0.0]).tolist(),
                        "opening": 0.1,
                        "instantaneous_pass": False,
                        "criteria": {"opening_strict": False},
                    },
                },
            }
        )
    return frames


def test_handoff_window_accepts_confirmed_motion_boundaries():
    assert HANDOFF_VALIDATION_FRAMES == 8
    assert HANDOFF_WINDOW_MOTION_MIN_M == 0.008
    assert HANDOFF_RELATIVE_DRIFT_MAX_M == 0.015
    assert HANDOFF_COMOTION_RESIDUAL_MAX_M == 0.005

    metrics = BehaviorEnvFacade._handoff_window_metrics(_window_frames(), "right")

    assert metrics["radio_displacement_m"] == pytest.approx(0.008)
    assert metrics["eef_displacement_m"] == pytest.approx(0.008)
    assert metrics["passed"] is True
    assert all(metrics["criteria"].values())


@pytest.mark.parametrize(
    ("relative_y", "metric_name", "boundary"),
    [
        (
            [0.0, 0.005, 0.01, 0.015, 0.01, 0.005, 0.0, 0.0],
            "max_relative_drift_m",
            0.015,
        ),
        (
            [0.0, 0.005, 0.0, 0.005, 0.0, 0.005, 0.0, 0.005],
            "mean_comotion_residual_m",
            0.005,
        ),
    ],
)
def test_handoff_window_accepts_inclusive_drift_and_residual_boundaries(
    relative_y,
    metric_name,
    boundary,
):
    metrics = BehaviorEnvFacade._handoff_window_metrics(
        _window_frames(relative_y=relative_y), "right"
    )

    assert metrics[metric_name] == pytest.approx(boundary)
    assert metrics["passed"] is True


@pytest.mark.parametrize(
    ("frames", "failed_criterion"),
    [
        (
            _window_frames(radio_motion_m=np.nextafter(0.008, 0.0)),
            "radio_window_motion",
        ),
        (
            _window_frames(eef_motion_m=np.nextafter(0.008, 0.0)),
            "eef_window_motion",
        ),
        (
            _window_frames(
                relative_y=[0.0, 0.005, 0.01, 0.0150001, 0.01, 0.005, 0.0, 0.0]
            ),
            "relative_drift",
        ),
        (
            _window_frames(
                relative_y=[
                    0.0,
                    0.0050001,
                    0.0,
                    0.0050001,
                    0.0,
                    0.0050001,
                    0.0,
                    0.0050001,
                ]
            ),
            "mean_comotion_residual",
        ),
        (_window_frames(failed_opening_frame=3), "held_opening_all_frames"),
    ],
)
def test_handoff_window_fails_just_outside_each_confirmed_boundary(
    frames,
    failed_criterion,
):
    metrics = BehaviorEnvFacade._handoff_window_metrics(frames, "right")

    assert metrics["passed"] is False
    assert metrics["criteria"][failed_criterion] is False


def test_official_success_does_not_override_failed_eight_frame_validator(tmp_path):
    frames = _window_frames(radio_motion_m=np.nextafter(0.008, 0.0))
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 0
    facade._last_info = {"done": {"success": True}}
    facade._handoff_validator_frames = []
    facade._validator_trace_path = tmp_path / "validator_trace.json"
    facade._handoff_validator_frame = lambda _observation: frames.pop(0)

    result = None
    for index in range(HANDOFF_VALIDATION_FRAMES):
        facade._env_steps = index + 1
        result = facade._update_handoff_validator(_observation(marker=index))

    assert result is not None
    assert result["local_grasp_success"] is False
    assert result["held_hand"] is None
    assert result["per_hand"]["right"]["criteria"]["radio_window_motion"] is False
    assert facade._last_info["done"]["success"] is True


@pytest.mark.parametrize("held_hand", ["left", "right"])
def test_handoff_validator_selects_the_unique_dynamic_held_hand(
    tmp_path,
    held_hand,
):
    frames = _validator_frames(held_hand=held_hand)
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 0
    facade._handoff_validator_frames = []
    facade._validator_trace_path = tmp_path / "validator_trace.json"
    facade._handoff_validator_frame = lambda _observation: frames.pop(0)

    result = None
    for index in range(HANDOFF_VALIDATION_FRAMES):
        facade._env_steps = index + 1
        result = facade._update_handoff_validator(_observation(marker=index))

    assert result is not None
    assert result["held_hand"] == held_hand
    assert result["local_grasp_success"] is True
    assert result["passed_hands"] == [held_hand]
    assert result["per_hand"][held_hand]["passed"] is True
    other = "right" if held_hand == "left" else "left"
    assert result["per_hand"][other]["passed"] is False
    assert facade._validator_trace_path.is_file()


@pytest.mark.parametrize(
    ("instantaneous_pass", "relative_jump_m"),
    [(False, 0.0), (True, 0.03)],
)
def test_handoff_validator_fails_closed_on_open_gripper_or_pose_jump(
    tmp_path,
    instantaneous_pass,
    relative_jump_m,
):
    frames = _validator_frames(
        held_hand="right",
        instantaneous_pass=instantaneous_pass,
        relative_jump_m=relative_jump_m,
    )
    facade = BehaviorEnvFacade.__new__(BehaviorEnvFacade)
    facade._env_steps = 0
    facade._handoff_validator_frames = []
    facade._validator_trace_path = tmp_path / "validator_trace.json"
    facade._handoff_validator_frame = lambda _observation: frames.pop(0)

    result = None
    for index in range(HANDOFF_VALIDATION_FRAMES):
        facade._env_steps = index + 1
        result = facade._update_handoff_validator(_observation(marker=index))

    assert result is not None
    assert result["held_hand"] is None
    assert result["local_grasp_success"] is False
    assert result["per_hand"]["right"]["passed"] is False


def test_pi0_nav_pick_rpc_allowlist_is_mode_exclusive_and_health_remains_available():
    class Env:
        _control_mode = PI0_NAV_PICK_VLA_MODE

        @staticmethod
        def get_env_meta():
            return {"status": "healthy", "handoff_state": "running"}

        @staticmethod
        def pi0_nav_pick_chunk_step(_actions, *, chunk_index):
            return {"chunk_index": chunk_index}

        @staticmethod
        def chunk_step(*_args, **_kwargs):
            return "forbidden-full-task"

        @staticmethod
        def pi0_chunk_step(*_args, **_kwargs):
            return "forbidden-pick"

        @staticmethod
        def pi0_navigate_to_chunk_step(*_args, **_kwargs):
            return "forbidden-nav"

        @staticmethod
        def observe(camera):
            return {"camera": camera, "frame_id": "fresh-frame"}

        @staticmethod
        def pixel_to_world(camera, frame_id, u, v):
            return {"camera": camera, "frame_id": frame_id, "xyz": [u, v, 1.0]}

        @staticmethod
        def prepress_move_to(**kwargs):
            return {"handler": "prepress_move_to", **kwargs}

        @staticmethod
        def prepress_rotate_wrist(**kwargs):
            return {"handler": "prepress_rotate_wrist", **kwargs}

        @staticmethod
        def save_robot_state_checkpoint(**kwargs):
            return {"handler": "save_robot_state_checkpoint", **kwargs}

        @staticmethod
        def restore_robot_state_checkpoint(**kwargs):
            return {"handler": "restore_robot_state_checkpoint", **kwargs}

    dispatcher = _MainThreadDispatcher(Env(), threading.Event())

    assert dispatcher._dispatch("env.get_env_meta", (), {})["status"] == "healthy"
    assert dispatcher._dispatch(
        "env.pi0_nav_pick_chunk_step", ([0.0] * ACTION_DIM,), {"chunk_index": 1}
    ) == {"chunk_index": 1}
    assert dispatcher._dispatch("env.observe", ("head",), {}) == {
        "camera": "head",
        "frame_id": "fresh-frame",
    }
    assert dispatcher._dispatch(
        "env.pixel_to_world",
        (),
        {"camera": "head", "frame_id": "fresh-frame", "u": 10, "v": 12},
    )["xyz"] == [10, 12, 1.0]
    for method in (
        "env.prepress_move_to",
        "env.prepress_rotate_wrist",
        "env.save_robot_state_checkpoint",
        "env.restore_robot_state_checkpoint",
    ):
        assert dispatcher._dispatch(method, (), {"marker": method})[
            "handler"
        ] == method.removeprefix("env.")
    for method in (
        "env.chunk_step",
        "env.pi0_chunk_step",
        "env.pi0_navigate_to_chunk_step",
        "env.pick",
    ):
        with pytest.raises(ValueError, match="unknown BEHAVIOR env RPC method"):
            dispatcher._dispatch(method, (), {})


def test_vla_server_handoff_disables_predict_but_keeps_health_available(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(vla_server, "_MODEL", object())
    monkeypatch.setattr(vla_server, "_MODEL_META", {"status": "ok"})
    monkeypatch.setattr(vla_server, "_ACTIONS_ENABLED", True)
    monkeypatch.setattr(vla_server, "build_env_observation", lambda _request: {})
    app = vla_server.build_app()
    endpoints = {
        route.path: route.endpoint
        for route in app.routes
        if getattr(route, "path", None)
    }

    before = endpoints["/healthz"]()
    disabled = endpoints["/control/disable-actions"]()
    after = endpoints["/healthz"]()

    assert before["actions_enabled"] is True
    assert disabled["actions_enabled"] is False
    assert after["status"] == "ok"
    assert after["actions_enabled"] is False
    assert after["pid"] == disabled["pid"]

    request = vla_server.PredictRequest(
        instruction="turn on the radio",
        images={},
        state=[],
    )
    with pytest.raises(HTTPException) as error:
        endpoints["/predict"](request)
    assert error.value.status_code == 409
    assert "disabled after controller handoff" in str(error.value.detail)


def test_vla_client_disable_actions_uses_control_endpoint_and_requires_confirmation():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return dict(self.payload)

    class HttpClient:
        def __init__(self, payload):
            self.payload = payload
            self.calls = []

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return Response(self.payload)

    client = BehaviorVLAClient.__new__(BehaviorVLAClient)
    client._base_url = "http://127.0.0.1:9876"
    client._client = HttpClient({"status": "ok", "pid": 42, "actions_enabled": False})

    result = client.disable_actions(timeout_ms=2500)

    assert result["actions_enabled"] is False
    assert client._client.calls == [
        (
            "http://127.0.0.1:9876/control/disable-actions",
            {"timeout": 2.5},
        )
    ]

    client._client = HttpClient({"status": "ok", "actions_enabled": True})
    with pytest.raises(RuntimeError, match="did not confirm action disable"):
        client.disable_actions()
