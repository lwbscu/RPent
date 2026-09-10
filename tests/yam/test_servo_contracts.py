# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Server/client PI capabilities and optional diagnostics, without hardware."""

import runpy
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robots.yam.contracts import env_runtime_contract
from robots.yam.geometry import joint_limits_from_config
from robots.yam.rlinf_env import YamAgentEnv
from robots.yam.robot_spec import _build_env_runtime_kwargs

_fakes = runpy.run_path(
    str(Path(__file__).with_name("test_facade_vla_fake_integration.py"))
)
FakeCameraRig = _fakes["FakeCameraRig"]
FakeKinematics = _fakes["FakeKinematics"]
FakeRpc = _fakes["FakeRpc"]
FakeRuntime = _fakes["FakeRuntime"]
FakeYamEnv = _fakes["FakeYamEnv"]
_yam_agent_config = _fakes["_yam_agent_config"]


def _server_contract():
    metadata = env_runtime_contract(
        task_name="place_cube",
        seed=12,
        max_episode_steps=40,
        joint_servo={"enabled": True},
    )
    lower, upper = joint_limits_from_config({})
    metadata["execution"].update(
        joint_limit_min=lower.tolist(), joint_limit_max=upper.tolist()
    )
    return metadata


def _build_client(metadata):
    fake = FakeYamEnv()
    rpc = FakeRpc({"env.get_env_meta": metadata, "env.observe": fake.observe})
    args = SimpleNamespace(task_name="place_cube", seed=12, max_episode_steps=40)
    client = _build_env_runtime_kwargs(args, rpc)["env"]
    return client, rpc, fake


def test_server_servo_capabilities_reach_real_client_factory():
    metadata = _server_contract()
    client, rpc, fake = _build_client(metadata)
    assert client.execution_capabilities["joint_servo"]["enabled"] is True
    assert client.execution_capabilities["joint_servo"]["max_bias_rad"] == 0.025
    assert (
        client.execution_capabilities["joint_limit_min"]
        == metadata["execution"]["joint_limit_min"]
    )
    assert fake.reset_calls == 0
    assert all(call[0] in {"env.get_env_meta", "env.observe"} for call in rpc.calls)


def test_old_server_without_optional_servo_remains_compatible():
    client, _, _ = _build_client(
        env_runtime_contract(task_name="place_cube", seed=12, max_episode_steps=40)
    )
    assert "joint_servo" not in client.execution_capabilities


@pytest.mark.parametrize("bad_field", ["task_name", "seed", "action_layouts"])
def test_optional_capabilities_do_not_weaken_task_or_layout_validation(bad_field):
    metadata = _server_contract()
    metadata[bad_field] = {
        "task_name": "other_task",
        "seed": 13,
        "action_layouts": ["delta_pose"],
    }[bad_field]
    with pytest.raises((ValueError, AssertionError)):
        _build_client(metadata)


@pytest.mark.parametrize(
    "bad_servo", [{"enabled": "yes"}, {"enabled": True, "max_bias_rad": 0.5}]
)
def test_invalid_servo_configuration_rejected_by_server_contract(bad_servo):
    with pytest.raises(ValueError, match="joint_servo"):
        env_runtime_contract(task_name="place_cube", joint_servo=bad_servo)


@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostics_switch_controls_observation_info_only(
    tmp_path, monkeypatch, enabled
):
    from robots.yam import diagnostics

    calls = []
    payload = {"status": "available", "arms": {"left": {"status": "available"}}}

    def read(runtime, *, include_cached_motor_telemetry=False):
        assert include_cached_motor_telemetry is True
        calls.append(runtime)
        return deepcopy(payload)

    monkeypatch.setattr(diagnostics, "read_control_diagnostics", read)
    runtime = FakeRuntime()
    cameras = FakeCameraRig()
    config = _yam_agent_config(tmp_path=tmp_path)
    config["control_diagnostics"] = enabled
    env = YamAgentEnv(
        config, runtime=runtime, cameras=cameras, kinematics=FakeKinematics()
    )
    try:
        _, info = env.observe()
        assert calls == ([runtime] if enabled else [])
        if enabled:
            assert info["control_diagnostics"] == payload
        else:
            assert "control_diagnostics" not in info
        np.testing.assert_array_equal(info["commanded_qpos"], runtime.qpos)
        assert runtime.commands == []
    finally:
        env.close()


def test_diagnostics_default_off_and_invalid_value_rejected_before_startup(tmp_path):
    runtime = FakeRuntime()
    cameras = FakeCameraRig()
    config = _yam_agent_config(tmp_path=tmp_path)
    env = YamAgentEnv(
        config, runtime=runtime, cameras=cameras, kinematics=FakeKinematics()
    )
    assert env.control_diagnostics is False
    assert cameras.open_calls == 0
    assert runtime.connect_calls == 0
    env.close()
    with pytest.raises(ValueError, match="control_diagnostics"):
        YamAgentEnv(
            {**config, "control_diagnostics": "false"}, runtime=runtime, cameras=cameras
        )
    assert cameras.open_calls == 0
    assert runtime.connect_calls == 0


@pytest.mark.parametrize(
    "extension", ["unknown", "invalid_servo", "partial_bounds", "bad_bounds"]
)
def test_client_rejects_invalid_or_unknown_site_extensions_before_observe(extension):
    from robots.yam.env_client import YamEnvClient

    expected = env_runtime_contract(
        task_name="place_cube", seed=12, max_episode_steps=40
    )
    metadata = _server_contract()
    execution = metadata["execution"]
    if extension == "unknown":
        execution["disable_guards"] = True
    elif extension == "invalid_servo":
        execution["joint_servo"]["max_bias_rad"] = 1.0
    elif extension == "partial_bounds":
        execution.pop("joint_limit_min")
    else:
        execution["joint_limit_max"] = [[float("nan")] * 6] * 2
    rpc = FakeRpc({"env.get_env_meta": metadata})
    with pytest.raises(ValueError):
        YamEnvClient(rpc, expected_meta=expected)
    assert all(call[0] == "env.get_env_meta" for call in rpc.calls)


def test_server_entrypoint_publishes_configured_servo_and_actual_joint_bounds(
    tmp_path, monkeypatch
):
    import json

    from robots.yam import env_server, rlinf_env

    config = {"task_name": "place_cube", "joint_servo": {"enabled": True}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    fake_env = FakeYamEnv()
    fake_env.lower, fake_env.upper = joint_limits_from_config({})
    served = []
    monkeypatch.setattr(rlinf_env, "YamAgentEnv", lambda values: fake_env)
    monkeypatch.setattr(
        env_server.YamEnvFacade,
        "serve",
        lambda self, **kwargs: served.append(self.get_env_meta()),
    )
    monkeypatch.setattr("sys.argv", ["yam-env", "--config", str(config_path)])
    env_server.main()
    execution = served[0]["execution"]
    assert execution["joint_servo"]["enabled"] is True
    assert execution["joint_limit_min"] == fake_env.lower.tolist()
    assert execution["joint_limit_max"] == fake_env.upper.tolist()
    assert fake_env.step_actions == []
    assert fake_env.close_calls == 1


@pytest.mark.parametrize(
    "receipt",
    [{"stop_requested": True, "hold_confirmed": False}, {"stop_requested": False}, {}],
)
def test_stop_receipt_only_updates_acknowledged_stop_flag(receipt):
    client, rpc, _ = _build_client(
        env_runtime_contract(task_name="place_cube", seed=12, max_episode_steps=40)
    )
    client.last_info["episode_status"]["stop_requested"] = False
    before = deepcopy(client.last_info)
    rpc.responses["env.request_stop"] = receipt
    assert client.request_stop() == receipt
    if receipt.get("stop_requested") is True:
        before["episode_status"]["stop_requested"] = True
    assert client.last_info["episode_status"] == before["episode_status"]
    assert "hold_confirmed" not in client.last_info["episode_status"]
    np.testing.assert_array_equal(
        client.last_info["robot_state"]["qpos"], before["robot_state"]["qpos"]
    )
