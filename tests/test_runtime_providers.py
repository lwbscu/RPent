import argparse
import subprocess
import sys

import pytest

import robots.behavior
import robots.behavior.runtime_provider as behavior_runtime
import robots.libero
import robots.libero.runtime_provider as libero_runtime
from rpent.cli.main import _build_argparser, _preparse_env
from rpent.envs import get_runtime_provider


class _Toolkit:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _behavior_args(*extra):
    provider = behavior_runtime.BehaviorRuntimeProvider()
    parser = argparse.ArgumentParser()
    provider.add_cli_args(parser)
    args = parser.parse_args(list(extra))
    return provider, parser, args


def _write_behavior_checkpoint(path):
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"weights")
    stats = (
        path
        / "assets"
        / "behavior-1k"
        / "2025-challenge-demos"
        / "norm_stats.json"
    )
    stats.parent.mkdir(parents=True)
    stats.write_text("{}", encoding="utf-8")


def test_get_runtime_provider_and_cli_load_behavior_arguments_dynamically():
    assert _preparse_env(["--env", "behavior", "--task-name", "folding_towels"]) == (
        "behavior"
    )
    provider = get_runtime_provider("BEHAVIOR")
    assert isinstance(provider, behavior_runtime.BehaviorRuntimeProvider)

    args = _build_argparser(provider).parse_args(
        [
            "--env",
            "behavior",
            "--task-name",
            "folding_towels",
            "--activity-instance-id",
            "17",
            "--no-driver",
            "--env-port",
            "4321",
            "--vla-endpoint",
            "http://vla.example",
        ]
    )

    assert args.env_name == "behavior"
    assert args.task_name == "folding_towels"
    assert args.activity_instance_id == 17
    assert not hasattr(args, "libero_type")


def test_behavior_attach_constructs_runtime_without_owning_external_servers(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "checkpoint"
    _write_behavior_checkpoint(checkpoint)
    provider, parser, args = _behavior_args(
        "--no-driver",
        "--env-endpoint",
        "external-env",
        "--env-port",
        "4321",
        "--vla-endpoint",
        "http://external-vla:8000",
        "--policy-checkpoint",
        str(checkpoint),
    )
    provider.validate_args(args, parser)
    endpoint_calls = []
    stop_env_calls = []
    stop_vla_calls = []
    captured = {}
    toolkit = _Toolkit()

    class Model:
        def __init__(self, endpoint):
            captured["model_endpoint"] = endpoint

        def wait_for_healthz(self, *, timeout_s):
            captured["health_timeout"] = timeout_s

    class Env:
        def __init__(self, client, *, expected_meta):
            captured["rpc_client"] = client
            captured["expected_meta"] = expected_meta

    def get_toolkit(**kwargs):
        captured["toolkit_kwargs"] = kwargs
        return toolkit

    monkeypatch.setattr(
        behavior_runtime,
        "set_socket_endpoint",
        lambda output_dir, host, port: endpoint_calls.append((output_dir, host, port)),
    )
    monkeypatch.setattr(behavior_runtime, "BehaviorVLAClient", Model)
    monkeypatch.setattr(behavior_runtime, "BehaviorEnvClient", Env)
    monkeypatch.setattr(behavior_runtime, "create_rpc_client", lambda output_dir: "rpc")
    monkeypatch.setattr(
        behavior_runtime,
        "stop_env_server",
        lambda proc, *, output_dir: stop_env_calls.append((proc, output_dir)),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        lambda proc: stop_vla_calls.append(proc),
    )
    monkeypatch.setattr(robots.behavior, "get_toolkit", get_toolkit)

    handle = provider.start(args, output_dir=tmp_path)
    handle.close()

    assert endpoint_calls == [(tmp_path, "external-env", 4321)]
    assert captured["model_endpoint"] == "http://external-vla:8000"
    assert captured["health_timeout"] == 30.0
    assert captured["rpc_client"] == "rpc"
    assert handle.env_proc is None and handle.vla_proc is None
    assert stop_env_calls == [(None, tmp_path)]
    assert stop_vla_calls == [None]
    assert toolkit.close_calls == 1


def test_behavior_start_failure_cleans_only_processes_started_by_provider(
    monkeypatch, tmp_path
):
    provider, _, args = _behavior_args()
    owned_env = object()
    owned_vla = object()
    stopped_env = []
    stopped_vla = []

    monkeypatch.setattr(
        behavior_runtime,
        "start_env_server",
        lambda args, *, output_dir: owned_env,
    )
    monkeypatch.setattr(
        behavior_runtime,
        "start_vla_server",
        lambda args, *, output_dir: ("http://owned-vla", owned_vla),
    )
    monkeypatch.setattr(
        behavior_runtime, "BehaviorVLAClient", lambda endpoint: object()
    )
    monkeypatch.setattr(
        behavior_runtime,
        "create_rpc_client",
        lambda output_dir: (_ for _ in ()).throw(RuntimeError("rpc failed")),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "stop_env_server",
        lambda proc, *, output_dir: stopped_env.append((proc, output_dir)),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        lambda proc: stopped_vla.append(proc),
    )

    with pytest.raises(RuntimeError, match="rpc failed"):
        provider.start(args, output_dir=tmp_path)

    assert stopped_env == [(owned_env, tmp_path)]
    assert stopped_vla == [owned_vla]


def test_libero_provider_attach_construction_remains_non_owning(monkeypatch, tmp_path):
    provider = get_runtime_provider("libero")
    assert isinstance(provider, libero_runtime.LiberoRuntimeProvider)
    parser = argparse.ArgumentParser()
    provider.add_cli_args(parser)
    args = parser.parse_args(
        [
            "--suite",
            "libero_object_task",
            "--task",
            "3",
            "--seed",
            "5",
            "--no-driver",
            "--env-endpoint",
            "external-env",
            "--env-port",
            "5001",
            "--vla-endpoint",
            "http://external-vla:8000",
        ]
    )
    provider.validate_args(args, parser)
    assert provider.recipe_tag(args) == "object_task_t3_s5"
    endpoint_calls = []
    stop_env_calls = []
    stop_vla_calls = []
    captured = {}
    toolkit = _Toolkit()

    class Env:
        def __init__(self, client, *, expected_meta):
            captured["env"] = (client, expected_meta)

    class Model:
        def __init__(self, endpoint):
            captured["model_endpoint"] = endpoint

    def get_toolkit(**kwargs):
        captured["toolkit_kwargs"] = kwargs
        return toolkit

    monkeypatch.setattr(
        libero_runtime,
        "set_socket_endpoint",
        lambda output_dir, host, port: endpoint_calls.append((output_dir, host, port)),
    )
    monkeypatch.setattr(libero_runtime, "create_rpc_client", lambda output_dir: "rpc")
    monkeypatch.setattr(libero_runtime, "LiberoEnvClient", Env)
    monkeypatch.setattr(libero_runtime, "VLAClient", Model)
    monkeypatch.setattr(
        libero_runtime,
        "stop_env_server",
        lambda *args, **kwargs: stop_env_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        libero_runtime,
        "stop_vla_server",
        lambda *args, **kwargs: stop_vla_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(robots.libero, "get_toolkit", get_toolkit)

    handle = provider.start(args, output_dir=tmp_path)
    handle.close()

    assert endpoint_calls == [(tmp_path, "external-env", 5001)]
    assert captured["model_endpoint"] == "http://external-vla:8000"
    assert captured["env"][0] == "rpc"
    assert captured["env"][1] == {
        "suite": "libero_object_task",
        "task": 3,
        "seed": 5,
        "max_episode_steps": 10000,
    }
    assert handle.env_proc is None and handle.vla_proc is None
    assert stop_env_calls == [] and stop_vla_calls == []
    assert toolkit.close_calls == 1


def test_libero_owned_process_is_killed_and_reaped_after_terminate_timeout():
    class Process:
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

        def wait(self, *, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("fake", timeout)
            return 0

    process = Process()
    libero_runtime._terminate_process(process, timeout=0.01)

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == 2


def test_behavior_server_command_preserves_virtualenv_python_symlink(
    monkeypatch,
    tmp_path,
):
    provider, _, args = _behavior_args()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(sys.executable)
    args.behavior_python = str(venv_python)
    args.behavior_repo = str(tmp_path)
    args.activity_instance_dir = str(tmp_path)
    args.policy_checkpoint = str(tmp_path)
    captured = {}

    def capture(command, **kwargs):
        captured["command"] = command
        raise RuntimeError("captured")

    monkeypatch.setattr(behavior_runtime.subprocess, "Popen", capture)

    with pytest.raises(RuntimeError, match="captured"):
        behavior_runtime.start_env_server(args, output_dir=tmp_path)

    assert captured["command"][0] == str(venv_python.absolute())
    assert captured["command"][0] != str(venv_python.resolve())
    mode_index = captured["command"].index("--control-mode")
    assert captured["command"][mode_index + 1] == "full_task_vla"
