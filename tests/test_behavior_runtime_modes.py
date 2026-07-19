import argparse
import sys

import pytest

import robots.behavior
import robots.behavior.runtime_provider as behavior_runtime
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.schemas import (
    FULL_TASK_VLA_MODE,
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
        runner_kwargs={
            "env": object(),
            "model": object(),
            "max_episode_steps": 1,
            "output_dir": tmp_path,
        }
    )
    names = [spec["name"] for spec in toolkit.get_tools_spec()]
    behavior_names = [
        name for name in names if name not in {spec["name"] for spec in common.TOOLS_SPEC}
    ]

    assert toolkit.control_mode == FULL_TASK_VLA_MODE
    assert behavior_names == ["run_full_task"]


def test_behavior_toolkit_planner_mode_exposes_only_planner_tools():
    planner = _PlannerClient()
    toolkit = BehaviorToolkit(control_mode=PLANNER_TOOLS_MODE, planner_client=planner)
    names = [spec["name"] for spec in toolkit.get_tools_spec()]
    behavior_names = [
        name for name in names if name not in {spec["name"] for spec in common.TOOLS_SPEC}
    ]

    assert behavior_names == list(PLANNER_TOOL_NAMES)
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
            {"hand": "left", "target_xyz": [1.0, 2.0, 3.0], "plan_only": True},
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
    assert "runner_kwargs" not in captured["toolkit"]
    assert handle.vla_proc is None
