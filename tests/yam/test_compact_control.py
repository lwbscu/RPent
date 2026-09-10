# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

"""Camera-free control feedback uses the same guarded writer, without hardware."""

import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from robots.yam.env_server import YamEnvFacade
from robots.yam.rlinf_env import YamAgentEnv

_helpers = runpy.run_path(str(Path(__file__).with_name("test_servo_contracts.py")))
_fakes = _helpers["_fakes"]


@pytest.fixture
def system(tmp_path):
    receipt = tmp_path / "receipt.json"
    config = _fakes["_yam_agent_config"](
        tmp_path=tmp_path, operator_receipt_path=receipt
    )
    runtime = _fakes["FakeRuntime"](accepted_updates_measured_state=True)
    cameras = _fakes["FakeCameraRig"]()
    env = YamAgentEnv(
        config, runtime=runtime, cameras=cameras, kinematics=_fakes["FakeKinematics"]()
    )
    _fakes["_observe_write_ready_reset"](env, receipt)
    try:
        yield env, runtime, cameras
    finally:
        env.close()


def test_compact_read_and_step_are_camera_free_and_keep_visual_snapshot(system):
    env, runtime, cameras = system
    old_obs, old_info = env.last_obs, env.last_info
    snapshots = cameras.snapshot_count
    runtime.qpos[0] = 0.001
    obs, info = env.read_control_state()
    assert obs["state"]["joint_position"][0] == 0.001
    assert set(obs) == {"state"}
    assert runtime.commands == []
    target = runtime.qpos.copy()
    target[0] = 0.002
    facade = YamEnvFacade(env)
    result = facade._dispatch(
        "env.control_step",
        (target,),
        {"expected_episode_id": info["episode_status"]["episode_id"]},
    )
    assert result[4]["executed_actions"] == 1
    np.testing.assert_array_equal(result[0]["state"]["joint_position"], target)
    np.testing.assert_array_equal(result[4]["commanded_qpos"], target)
    assert cameras.snapshot_count == snapshots
    assert env.last_obs is old_obs and env.last_info is old_info
    assert "env.control_step" not in facade._readonly_methods
    assert "env.read_control_state" not in facade._readonly_methods


@pytest.mark.parametrize("guard", ["epoch", "stop", "not_ready", "collision"])
def test_compact_step_keeps_existing_guards(system, monkeypatch, guard):
    env, runtime, cameras = system
    _, info = env.read_control_state()
    episode_id = info["episode_status"]["episode_id"]
    if guard == "epoch":
        episode_id = "stale"
    elif guard == "stop":
        env.request_stop()
    elif guard == "not_ready":
        env._operator_ready_receipt = None
    else:

        def reject(_action):
            raise ValueError("collision")

        monkeypatch.setattr(env, "_check_measured_transition", reject)
    if guard == "stop":
        result = env.control_step(runtime.qpos, expected_episode_id=episode_id)
        assert result[4]["executed_actions"] == 0
        assert result[3] is True
    else:
        with pytest.raises((ValueError, RuntimeError)):
            env.control_step(runtime.qpos, expected_episode_id=episode_id)
    assert runtime.commands == []


def test_compact_feedback_fault_stops_writer(system, monkeypatch):
    env, runtime, _ = system
    _, info = env.read_control_state()

    def fail():
        raise RuntimeError("feedback unavailable")

    monkeypatch.setattr(env, "read_control_state", fail)
    holds = runtime.hold_calls
    with pytest.raises(RuntimeError, match="feedback unavailable"):
        env.control_step(
            runtime.qpos, expected_episode_id=info["episode_status"]["episode_id"]
        )
    assert env._stop_requested.is_set()
    assert runtime.hold_calls > holds


def test_client_compact_opt_in_and_independent_caches():
    metadata = _helpers["_server_contract"]()
    metadata["execution"]["compact_control"] = True
    client, rpc, _ = _helpers["_build_client"](metadata)
    old_obs, old_info = client.last_obs, client.last_info
    qpos = np.zeros(14)
    qpos[[6, 13]] = 0.5
    observation = {"state": {"joint_position": qpos}}
    info = deepcopy(old_info)
    info.update(robot_state={}, commanded_qpos=qpos, executed_actions=1)
    rpc.responses["env.read_control_state"] = (observation, info)
    rpc.responses["env.control_step"] = (observation, 0.0, False, False, info)
    rpc.responses["env.request_stop"] = {"stop_requested": True}
    client.read_control_state()
    client.control_step(qpos, expected_episode_id="explicit-epoch")
    assert rpc.calls[-1][2] == {"expected_episode_id": "explicit-epoch"}
    assert client.last_control_obs is observation
    assert client.last_obs is old_obs and client.last_info is old_info
    client.request_stop()
    assert client.last_control_info["episode_status"]["stop_requested"] is True
    with pytest.raises(ValueError, match="one qpos14"):
        client.control_step(
            np.stack([qpos, qpos]), expected_episode_id="explicit-epoch"
        )


@pytest.mark.parametrize("capability", [None, False, "true", 1])
def test_absent_or_invalid_capability_does_not_enable_rpc(capability):
    metadata = _helpers["_server_contract"]()
    if capability is not None:
        metadata["execution"]["compact_control"] = capability
    if capability is not None and type(capability) is not bool:
        with pytest.raises(ValueError, match="boolean"):
            _helpers["_build_client"](metadata)
        return
    client, rpc, _ = _helpers["_build_client"](metadata)
    before = len(rpc.calls)
    with pytest.raises(RuntimeError, match="does not advertise"):
        client.read_control_state()
    assert len(rpc.calls) == before
