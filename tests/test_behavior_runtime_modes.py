import argparse
import json
import signal
import sys
import threading

import pytest

import robots.behavior
import robots.behavior.run_manifest as behavior_manifest
import robots.behavior.runtime_provider as behavior_runtime
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.env_server import _MainThreadDispatcher
from robots.behavior.run_manifest import RunManifest, redact_command
from robots.behavior.schemas import (
    FULL_TASK_VLA_MODE,
    PI0_NAV_PICK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOLS_MODE,
)
from robots.behavior.toolkit import BehaviorToolkit
from rpent.tools import common


class _PlannerClient:
    def __init__(self):
        self.calls = []

    def close(self):
        self.calls.append(("close", {}))


def test_post_pick_env_client_routes_exact_core_rpc_signatures():
    class Rpc:
        def __init__(self):
            self.calls = []

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            self.calls.append((method, args, kwargs or {}, timeout_s))
            if method == "env.get_env_meta":
                return {"control_mode": PI0_NAV_PICK_VLA_MODE}
            return {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": method,
                "total_env_steps": 1819,
            }

    rpc = Rpc()
    client = BehaviorEnvClient(
        rpc,
        expected_meta={"control_mode": PI0_NAV_PICK_VLA_MODE},
    )
    client.inspect_post_pick_state()
    client.observe(camera="head")
    client.declare_button_visibility(
        camera="head",
        frame_id="head:1819:abc",
        button_visible=False,
        negative_case="clear_slotted_back_face",
    )
    client.pixel_to_world(
        camera="head",
        frame_id="head:1819:abc",
        u=10,
        v=12,
    )
    client.evaluate_prepress_geometry(projection_id="projection-1")
    client.prepress_move_to(
        role="press",
        button_goal={
            "kind": "press_staging",
            "projection_id": "projection-1",
            "standoff_m": 0.055,
        },
        plan_only=True,
    )
    client.prepress_rotate_wrist(
        role="press",
        relative_axis_angle=[0.0, 0.0, 1.0, 0.01],
        plan_only=True,
    )
    client.save_prepress_checkpoint()
    client.save_post_pick_debug_mirror()

    assert [call[0] for call in rpc.calls[1:]] == [
        "env.inspect_post_pick_state",
        "env.observe",
        "env.declare_button_visibility",
        "env.pixel_to_world",
        "env.evaluate_prepress_geometry",
        "env.prepress_move_to",
        "env.prepress_rotate_wrist",
        "env.save_prepress_checkpoint",
        "env.save_post_pick_debug_mirror",
    ]
    assert rpc.calls[3][2]["bbox_xyxy"] is None
    assert rpc.calls[3][2]["negative_case"] == "clear_slotted_back_face"
    assert "hand" not in rpc.calls[6][2]
    assert rpc.calls[6][2]["role"] == "press"
    assert "hand" not in rpc.calls[7][2]
    assert rpc.calls[7][2]["role"] == "press"
    assert rpc.calls[8][2]["checkpoint_name"] == "state_checkpoint_2"


@pytest.mark.parametrize(
    "negative_case",
    ["clear_slotted_back_face", "side_port", "ambiguous"],
)
def test_post_pick_env_client_preserves_canonical_negative_face_class(
    negative_case,
):
    class Rpc:
        def __init__(self):
            self.calls = []

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            self.calls.append((method, args, kwargs or {}, timeout_s))
            if method == "env.get_env_meta":
                return {"control_mode": PI0_NAV_PICK_VLA_MODE}
            return {
                "primitive_success": True,
                "task_success": False,
                "stop_reason": "button_not_visible",
                "face_class": negative_case,
                "total_env_steps": 1819,
            }

    rpc = Rpc()
    client = BehaviorEnvClient(
        rpc,
        expected_meta={"control_mode": PI0_NAV_PICK_VLA_MODE},
    )

    result = client.declare_button_visibility(
        camera="held_wrist",
        frame_id="right_wrist:1819:abc",
        button_visible=False,
        negative_case=negative_case,
    )

    assert result["face_class"] == negative_case
    assert rpc.calls[-1][0] == "env.declare_button_visibility"
    assert rpc.calls[-1][2]["negative_case"] == negative_case


def test_env_rpc_dispatcher_rejects_private_and_legacy_unprefixed_methods():
    class Env:
        _control_mode = PLANNER_TOOLS_MODE

        def observe(self, camera):
            return {"camera": camera}

        def _robot(self):
            raise AssertionError("private simulator state must never be dispatched")

    dispatcher = _MainThreadDispatcher(Env(), threading.Event())

    assert dispatcher._dispatch("env.observe", (), {"camera": "head"}) == {
        "camera": "head"
    }
    with pytest.raises(ValueError, match="unknown BEHAVIOR env RPC"):
        dispatcher._dispatch("env._robot", (), {})
    with pytest.raises(ValueError, match="unknown RPC method"):
        dispatcher._dispatch("observe", (), {"camera": "head"})


@pytest.mark.parametrize(
    ("mode", "allowed_method", "rejected_method"),
    [
        (FULL_TASK_VLA_MODE, "env.chunk_step", "env.observe"),
        (PI0_PICK_VLA_MODE, "env.pi0_chunk_step", "env.chunk_step"),
        (PLANNER_TOOLS_MODE, "env.observe", "env.chunk_step"),
    ],
)
def test_env_rpc_dispatcher_isolates_control_mode_methods(
    mode, allowed_method, rejected_method
):
    class Env:
        _control_mode = mode

        @staticmethod
        def chunk_step(*_args, **_kwargs):
            return "chunk"

        @staticmethod
        def pi0_chunk_step(*_args, **_kwargs):
            return "pi0_chunk"

        @staticmethod
        def observe(*_args, **_kwargs):
            return "observe"

        @staticmethod
        def pick(*_args, **_kwargs):
            return "pick"

    dispatcher = _MainThreadDispatcher(Env(), threading.Event())

    assert dispatcher._dispatch(allowed_method, (), {}) in {
        "chunk",
        "pi0_chunk",
        "observe",
    }
    with pytest.raises(ValueError, match="unknown BEHAVIOR env RPC"):
        dispatcher._dispatch(rejected_method, (), {})


def test_env_rpc_dispatcher_keeps_chunk_interfaces_mode_private():
    class Env:
        def __init__(self, mode):
            self._control_mode = mode

        @staticmethod
        def chunk_step(*_args, **_kwargs):
            return "chunk"

        @staticmethod
        def pi0_chunk_step(*_args, **_kwargs):
            return "pi0_chunk"

    for mode, allowed, rejected in (
        (FULL_TASK_VLA_MODE, "env.chunk_step", "env.pi0_chunk_step"),
        (PI0_PICK_VLA_MODE, "env.pi0_chunk_step", "env.chunk_step"),
    ):
        dispatcher = _MainThreadDispatcher(Env(mode), threading.Event())
        dispatcher._dispatch(allowed, (), {})
        with pytest.raises(ValueError, match="unknown BEHAVIOR env RPC"):
            dispatcher._dispatch(rejected, (), {})

    planner = _MainThreadDispatcher(Env(PLANNER_TOOLS_MODE), threading.Event())
    for method in ("env.chunk_step", "env.pi0_chunk_step"):
        with pytest.raises(ValueError, match="unknown BEHAVIOR env RPC"):
            planner._dispatch(method, (), {})


for _tool_name in PLANNER_TOOL_NAMES:

    def _make(name):
        def _method(self, **kwargs):
            self.calls.append((name, kwargs))
            return {"name": name, "primitive_success": True, "task_success": False}

        return _method

    setattr(_PlannerClient, _tool_name, _make(_tool_name))


def _provider_args(tmp_path, *extra):
    provider = behavior_runtime.BehaviorRuntimeProvider()
    parser = argparse.ArgumentParser()
    provider.add_cli_args(parser)
    instance_dir = tmp_path / "instances"
    instance_dir.mkdir()
    (instance_dir / "house_242_template-tro_state.json").write_text(
        "{}", encoding="utf-8"
    )
    args = parser.parse_args(
        [
            "--behavior-repo",
            str(tmp_path),
            "--behavior-python",
            sys.executable,
            "--activity-instance-dir",
            str(instance_dir),
            *extra,
        ]
    )
    return provider, parser, args


def test_behavior_toolkit_full_task_mode_remains_default(tmp_path):
    toolkit = BehaviorToolkit(
        primitives_kwargs={
            "env": object(),
            "model": object(),
            "max_episode_steps": 1,
            "output_dir": tmp_path,
        }
    )
    names = [spec["name"] for spec in toolkit.get_tools_spec()]

    assert toolkit.control_mode == FULL_TASK_VLA_MODE
    assert names == ["run_full_task"]
    assert not {spec["name"] for spec in common.TOOLS_SPEC}.intersection(names)


def test_behavior_toolkit_planner_mode_exposes_only_planner_tools():
    planner = _PlannerClient()
    toolkit = BehaviorToolkit(control_mode=PLANNER_TOOLS_MODE, planner_client=planner)
    names = [spec["name"] for spec in toolkit.get_tools_spec()]

    assert names == list(PLANNER_TOOL_NAMES)
    assert not {spec["name"] for spec in common.TOOLS_SPEC}.intersection(names)
    result = toolkit.execute_tool(
        "move_to",
        {"hand": "left", "target_xyz": [1.0, 2.0, 3.0], "plan_only": True},
    )

    assert result.is_finish is False
    assert result.result["primitive_success"] is True
    assert result.result["task_success"] is False
    assert planner.calls == [
        (
            "move_to",
            {
                "hand": "left",
                "target_xyz": [1.0, 2.0, 3.0],
                "frame": "world",
                "target_quat_xyzw": None,
                "plan_only": True,
                "position_tolerance_m": 0.02,
                "orientation_tolerance_rad": 0.087,
                "timeout_s": 45,
            },
        )
    ]


def test_behavior_env_client_planner_methods_use_env_rpc_names():
    class Rpc:
        def __init__(self):
            self.calls = []

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            self.calls.append((method, args, kwargs or {}, timeout_s))
            if method == "env.get_env_meta":
                return {"suite": "behavior"}
            return {"name": method, "kwargs": kwargs or {}}

        def close(self):
            self.calls.append(("close", (), {}, None))

    rpc = Rpc()
    client = BehaviorEnvClient(rpc, expected_meta={"suite": "behavior"})

    assert client.observe(camera="head")["name"] == "env.observe"
    move_result = client.move_to(
        hand="right",
        target_xyz=[0.1, 0.2, 0.3],
        target_quat_xyzw=[0, 0, 0, 1],
    )

    assert move_result["kwargs"]["hand"] == "right"
    assert rpc.calls[1][0] == "env.observe"
    assert rpc.calls[2][0] == "env.move_to"
    with pytest.raises(ValueError, match="exactly one"):
        client.rotate_wrist(hand="left")


def test_behavior_env_client_pi0_chunk_uses_private_rpc_and_tracks_done():
    class Rpc:
        def __init__(self):
            self.calls = []

        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            self.calls.append((method, args, kwargs or {}, timeout_s))
            if method == "env.get_env_meta":
                return {"control_mode": PI0_PICK_VLA_MODE}
            return (
                {"states": [0.0] * 256},
                0.0,
                False,
                False,
                {"done": {"success": True}},
            )

    rpc = Rpc()
    client = BehaviorEnvClient(
        rpc,
        expected_meta={"control_mode": PI0_PICK_VLA_MODE},
    )
    actions = [[0.0] * 23]

    result = client.pi0_chunk_step(
        actions,
        hand="right",
        gripper_closed_threshold=0.04,
        required_closed_steps=3,
        stop_on_candidate=True,
    )

    assert result[-1]["done"]["success"] is True
    assert client.episode_done is True
    method, args, kwargs, _timeout = rpc.calls[1]
    assert method == "env.pi0_chunk_step"
    assert args == (actions,)
    assert kwargs == {
        "hand": "right",
        "gripper_closed_threshold": 0.04,
        "required_closed_steps": 3,
        "stop_on_candidate": True,
    }


def test_planner_mode_validation_does_not_require_checkpoint_or_vla(tmp_path):
    provider, parser, args = _provider_args(
        tmp_path,
        "--behavior-control-mode",
        "planner_tools",
        "--no-driver",
        "--env-port",
        "4321",
    )

    provider.validate_args(args, parser)


def test_planner_mode_start_does_not_create_vla_client_or_server(monkeypatch, tmp_path):
    provider, parser, args = _provider_args(
        tmp_path,
        "--behavior-control-mode",
        "planner_tools",
        "--no-driver",
        "--env-endpoint",
        "external-env",
        "--env-port",
        "4321",
    )
    provider.validate_args(args, parser)
    captured = {}

    class Env:
        def __init__(self, client, *, expected_meta):
            captured["env"] = (client, expected_meta)

        def reset(self):
            captured["reset"] = True

    def fail_vla(*args, **kwargs):
        raise AssertionError("planner mode must not start or connect to VLA")

    def get_toolkit(**kwargs):
        captured["toolkit"] = kwargs
        return _PlannerClient()

    monkeypatch.setattr(behavior_runtime, "BehaviorVLAClient", fail_vla)
    monkeypatch.setattr(behavior_runtime, "start_vla_server", fail_vla)
    monkeypatch.setattr(behavior_runtime, "BehaviorEnvClient", Env)
    monkeypatch.setattr(behavior_runtime, "create_rpc_client", lambda output_dir: "rpc")
    monkeypatch.setattr(robots.behavior, "get_toolkit", get_toolkit)

    handle = provider.start(args, output_dir=tmp_path / "run")

    assert captured["env"][0] == "rpc"
    assert captured["toolkit"]["control_mode"] == "planner_tools"
    assert captured["toolkit"]["planner_client"] is not None
    assert captured["reset"] is True
    assert "primitives_kwargs" not in captured["toolkit"]
    assert handle.vla_proc is None


def _write_checkpoint(checkpoint):
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    stats = (
        checkpoint
        / "assets"
        / "behavior-1k"
        / "2025-challenge-demos"
        / "norm_stats.json"
    )
    stats.parent.mkdir(parents=True)
    stats.write_text("{}", encoding="utf-8")


def test_full_task_no_driver_still_requires_complete_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    provider, parser, args = _provider_args(
        tmp_path,
        "--no-driver",
        "--env-port",
        "4321",
        "--vla-endpoint",
        "http://external-vla:8000",
        "--policy-checkpoint",
        str(checkpoint),
    )

    with pytest.raises(SystemExit):
        provider.validate_args(args, parser)

    _write_checkpoint(checkpoint)
    provider.validate_args(args, parser)


def test_pi0_mode_starts_vla_and_registers_only_local_primitive(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    provider, parser, args = _provider_args(
        tmp_path,
        "--behavior-control-mode",
        PI0_PICK_VLA_MODE,
        "--behavior-pi0-pick-hand",
        "left",
        "--behavior-pi0-pick-instruction",
        "grasp only the radio handle",
        "--no-driver",
        "--env-port",
        "4321",
        "--vla-endpoint",
        "http://external-vla:8000",
        "--policy-checkpoint",
        str(checkpoint),
    )
    provider.validate_args(args, parser)
    captured = {}

    class Env:
        def __init__(self, client, *, expected_meta):
            captured["env"] = (client, expected_meta)

        def reset(self):
            captured["reset"] = True
            return {"states": [0.0] * 256}, {"done": {"success": False}}

    class Model:
        def __init__(self, endpoint):
            captured["vla_endpoint"] = endpoint

        def wait_for_healthz(self, *, timeout_s):
            captured["health_timeout_s"] = timeout_s

    def get_toolkit(**kwargs):
        captured["toolkit"] = kwargs
        return _PlannerClient()

    monkeypatch.setattr(behavior_runtime, "BehaviorEnvClient", Env)
    monkeypatch.setattr(behavior_runtime, "BehaviorVLAClient", Model)
    monkeypatch.setattr(behavior_runtime, "create_rpc_client", lambda _output: "rpc")
    monkeypatch.setattr(robots.behavior, "get_toolkit", get_toolkit)

    handle = provider.start(args, output_dir=tmp_path / "pi0-run")
    prompt_vars = provider.prompt_vars(
        args,
        output_dir=tmp_path / "pi0-run",
        recipe_tag=provider.recipe_tag(args),
    )

    assert captured["env"][1]["control_mode"] == PI0_PICK_VLA_MODE
    assert captured["vla_endpoint"] == "http://external-vla:8000"
    assert captured["health_timeout_s"] == 30.0
    assert captured["reset"] is True
    assert captured["toolkit"]["control_mode"] == PI0_PICK_VLA_MODE
    assert "planner_client" not in captured["toolkit"]
    assert captured["toolkit"]["primitives_kwargs"]["initial_info"] == {
        "done": {"success": False}
    }
    assert 'hand="left"' in prompt_vars["behavior_user_instructions"]
    assert "grasp only the radio handle" in prompt_vars["behavior_user_instructions"]
    assert "max_chunks=24" in prompt_vars["behavior_user_instructions"]
    manifest = json.loads(
        (tmp_path / "pi0-run" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["control_mode"] == PI0_PICK_VLA_MODE
    assert manifest["checkpoint"] == str(checkpoint.resolve())
    assert "vla" in manifest["processes"]
    assert handle.vla_proc is None
    handle.close()


def test_pi0_mode_rejects_empty_local_instruction(tmp_path):
    provider, parser, args = _provider_args(
        tmp_path,
        "--behavior-control-mode",
        PI0_PICK_VLA_MODE,
        "--behavior-pi0-pick-instruction",
        "   ",
    )

    with pytest.raises(SystemExit):
        provider.validate_args(args, parser)


def test_run_manifest_planner_omits_vla_and_sensitive_environment(
    monkeypatch, tmp_path
):
    provider, _, args = _provider_args(
        tmp_path,
        "--behavior-control-mode",
        "planner_tools",
        "--no-driver",
        "--env-endpoint",
        "external-env",
        "--env-port",
        "4321",
        "--cuda-device",
        "7",
    )
    provider._resolve_paths(args)
    monkeypatch.setenv("EXAMPLE_API_TOKEN", "must-not-appear")

    manifest = RunManifest.start(tmp_path / "planner-run", args, repo_root=tmp_path)
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)

    assert payload["control_mode"] == "planner_tools"
    assert payload["gpu"] == "7"
    assert payload["processes"]["env"]["port"] == 4321
    assert "vla" not in serialized.lower()
    assert "checkpoint" not in payload
    assert "must-not-appear" not in serialized
    assert not list(manifest.path.parent.glob(".run_manifest.json.*.tmp"))


def test_run_manifest_full_records_redacted_owned_process(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint(checkpoint)
    provider, _, args = _provider_args(
        tmp_path,
        "--policy-checkpoint",
        str(checkpoint),
        "--cuda-device",
        "7",
    )
    provider._resolve_paths(args)
    manifest = RunManifest.start(tmp_path / "full-run", args, repo_root=tmp_path)

    class Process:
        pid = 81234
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(behavior_manifest.os, "getpgid", lambda pid: pid)
    manifest.process_started(
        "vla",
        Process(),
        command=[
            "vla-server",
            "--api-key",
            "secret-value",
            "--token=query-secret",
        ],
        host="127.0.0.1",
        port=27913,
    )
    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    process = payload["processes"]["vla"]

    assert payload["checkpoint"] == str(checkpoint.resolve())
    assert process["pid"] == 81234 and process["pgid"] == 81234
    assert process["port"] == 27913
    assert process["command"] == [
        "vla-server",
        "--api-key",
        "[REDACTED]",
        "--token=[REDACTED]",
    ]
    assert "secret-value" not in manifest.path.read_text(encoding="utf-8")
    assert redact_command("tool --password hunter2")[-1] == "[REDACTED]"


def test_behavior_owned_process_uses_new_session_and_targeted_pgid(
    monkeypatch, tmp_path
):
    _, _, args = _provider_args(tmp_path, "--behavior-control-mode", "planner_tools")
    captured = {}

    def capture(command, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(behavior_runtime.subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="captured"):
        behavior_runtime.start_env_server(args, output_dir=tmp_path)
    assert captured["start_new_session"] is True

    signals = []

    class Process:
        pid = 81235
        _rpent_owned_pgid = 81235

        def poll(self):
            return None

        def wait(self, *, timeout):
            return 0

        def terminate(self):
            raise AssertionError("dedicated process group must be targeted")

    monkeypatch.setattr(behavior_runtime.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(behavior_runtime.os, "getpgrp", lambda: 90000)
    monkeypatch.setattr(
        behavior_runtime.os,
        "killpg",
        lambda pgid, requested_signal: signals.append((pgid, requested_signal)),
    )
    behavior_runtime._terminate_process(Process())

    assert signals == [(81235, signal.SIGTERM)]


def test_behavior_stop_cleans_recorded_group_after_leader_already_exited(
    monkeypatch, tmp_path
):
    class Process:
        pid = 81236
        _rpent_owned_pgid = 81236
        _rpent_owned_sid = 81236

        def poll(self):
            return -11

    process = Process()
    terminated = []
    monkeypatch.setattr(
        behavior_runtime,
        "_owned_process_group_alive",
        lambda proc: proc is process,
    )
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        lambda proc: terminated.append(proc),
    )

    behavior_runtime.stop_env_server(process, output_dir=tmp_path)

    assert terminated == [process]


def test_failed_pi0_nav_pick_runtime_enters_mocked_resident_wait(monkeypatch, tmp_path):
    class Process:
        def __init__(self, pid):
            self.pid = pid

        @staticmethod
        def poll():
            return None

    class Model:
        def __init__(self, pid):
            self.pid = pid
            self.calls = []

        def disable_actions(self):
            self.calls.append("disable")
            return {"pid": self.pid, "actions_enabled": False}

        def healthz(self):
            self.calls.append("healthz")
            return {"pid": self.pid, "actions_enabled": False}

    class Toolkit:
        control_mode = PI0_NAV_PICK_VLA_MODE

        def __init__(self, model):
            self._primitives = type(
                "Primitives",
                (),
                {
                    "last_result": {
                        "primitive_success": False,
                        "task_success": False,
                        "result_path": str(tmp_path / "result.json"),
                        "stop_reason": "insufficient_handoff_horizon",
                    },
                    "model": model,
                },
            )()
            self.closed = False

        def close(self):
            self.closed = True

    env_proc = Process(81240)
    vla_proc = Process(81241)
    model = Model(vla_proc.pid)
    toolkit = Toolkit(model)
    waited = []
    stopped = []
    monkeypatch.setattr(
        behavior_runtime,
        "_owned_process_group_alive",
        lambda _process: True,
    )
    monkeypatch.setattr(
        behavior_runtime.BehaviorRuntimeHandle,
        "_wait_while_resident",
        lambda self: waited.append((self.env_proc.pid, self.vla_proc.pid)),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "stop_env_server",
        lambda process, *, output_dir: stopped.append(("env", process.pid)),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        lambda process: stopped.append(("vla", process.pid)),
    )
    handle = behavior_runtime.BehaviorRuntimeHandle(
        toolkit=toolkit,
        output_dir=tmp_path,
        env_proc=env_proc,
        vla_proc=vla_proc,
    )

    handle.close()

    assert waited == [(env_proc.pid, vla_proc.pid)]
    assert model.calls == ["disable", "healthz"]
    failed = json.loads((tmp_path / "failed_runtime.json").read_text(encoding="utf-8"))
    assert failed["handoff_state"] == "FAILED"
    assert failed["vla_gate_confirmed"] is True
    assert failed["stop_reason"] == "insufficient_handoff_horizon"
    assert toolkit.closed is True
    assert stopped == [("env", env_proc.pid), ("vla", vla_proc.pid)]


def test_officially_successful_pi0_nav_pick_never_enters_failed_resident_state(
    monkeypatch, tmp_path
):
    class Process:
        def __init__(self, pid):
            self.pid = pid

        @staticmethod
        def poll():
            return None

    class Model:
        @staticmethod
        def disable_actions():
            raise AssertionError("official success must not enter FAILED gating")

        @staticmethod
        def healthz():
            raise AssertionError("official success must not enter FAILED gating")

    class Toolkit:
        control_mode = PI0_NAV_PICK_VLA_MODE

        def __init__(self):
            self._primitives = type(
                "Primitives",
                (),
                {
                    "last_result": {
                        "primitive_success": False,
                        "task_success": True,
                        "result_path": str(tmp_path / "result.json"),
                        "stop_reason": "task_success",
                    },
                    "model": Model(),
                },
            )()
            self.closed = False

        def close(self):
            self.closed = True

    toolkit = Toolkit()
    waited = []
    stopped = []
    monkeypatch.setattr(
        behavior_runtime.BehaviorRuntimeHandle,
        "_wait_while_resident",
        lambda _self: waited.append(True),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "stop_env_server",
        lambda process, *, output_dir: stopped.append(("env", process.pid)),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        lambda process: stopped.append(("vla", process.pid)),
    )
    env_proc = Process(81242)
    vla_proc = Process(81243)
    handle = behavior_runtime.BehaviorRuntimeHandle(
        toolkit=toolkit,
        output_dir=tmp_path,
        env_proc=env_proc,
        vla_proc=vla_proc,
    )

    handle.close()

    assert waited == []
    assert not (tmp_path / "failed_runtime.json").exists()
    assert toolkit.closed is True
    assert stopped == [("env", env_proc.pid), ("vla", vla_proc.pid)]


def test_runtime_start_failure_atomically_marks_manifest_failed(monkeypatch, tmp_path):
    provider, parser, args = _provider_args(
        tmp_path,
        "--behavior-control-mode",
        "planner_tools",
        "--no-driver",
        "--env-port",
        "4321",
    )
    provider.validate_args(args, parser)
    monkeypatch.setattr(
        behavior_runtime,
        "create_rpc_client",
        lambda output_dir: (_ for _ in ()).throw(RuntimeError("rpc failed")),
    )
    output_dir = tmp_path / "failed-run"

    with pytest.raises(RuntimeError, match="rpc failed"):
        provider.start(args, output_dir=output_dir)

    payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["stopped_at"] is not None
    assert payload["error"] == {"message": "rpc failed", "type": "RuntimeError"}
    assert not list(output_dir.glob(".run_manifest.json.*.tmp"))
