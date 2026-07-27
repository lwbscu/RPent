from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "behavior_manual_debug.py"
    )
    spec = importlib.util.spec_from_file_location("behavior_manual_debug", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_startup_failure_status_is_not_overwritten_by_cleanup():
    module = _module()

    assert module._final_manifest_status("startup_failed", []) == "startup_failed"
    assert (
        module._final_manifest_status("startup_failed", ["env failed"])
        == "startup_failed_cleanup_failed"
    )
    assert module._final_manifest_status("ready", []) == "stopped"
    assert module._final_manifest_status("ready", ["env failed"]) == "cleanup_failed"


def test_verified_raw_success_latches_dashboard_terminal_state():
    module = _module()

    class State:
        def __init__(self):
            self.events = []
            self.attempts = []
            self.done = []

        def on_event(self, event):
            self.events.append(event)

        def end_attempt(self, **kwargs):
            self.attempts.append(kwargs)

        def mark_done(self, **kwargs):
            self.done.append(kwargs)

    state = State()
    toolkit = SimpleNamespace(
        runner_continuation_state=lambda: {
            "raw_official_success_verified": True,
            "attempt_index": 3,
        }
    )

    terminal = module._latch_dashboard_terminal(
        state,
        toolkit,
        SimpleNamespace(is_finish=False),
    )

    assert terminal is True
    assert state.events == [
        {
            "type": "official_success",
            "attempt_index": 3,
            "task_success": True,
            "workflow_complete": False,
            "artifact_seal_complete": False,
            "publication_complete": False,
        }
    ]
    assert state.attempts == [
        {"attempt_index": 3, "outcome": "official_task_success"}
    ]
    assert state.done == [{"terminated": True}]


def test_nonterminal_result_does_not_end_dashboard():
    module = _module()

    class State:
        def on_event(self, _event):
            raise AssertionError("nonterminal result must not emit success")

        def end_attempt(self, **_kwargs):
            raise AssertionError("nonterminal result must not end attempt")

        def mark_done(self, **_kwargs):
            raise AssertionError("nonterminal result must not mark dashboard done")

    toolkit = SimpleNamespace(
        runner_continuation_state=lambda: {
            "raw_official_success_verified": False,
            "attempt_index": 1,
        }
    )

    assert (
        module._latch_dashboard_terminal(
            State(),
            toolkit,
            SimpleNamespace(is_finish=False),
        )
        is False
    )


def test_runtime_args_include_checkpoint_bound_vla_contract(tmp_path, monkeypatch):
    module = _module()
    binding = {
        "resolved_path": str(module._POLICY_CHECKPOINT.resolve()),
        "binding_sha256": "binding",
    }
    monkeypatch.setattr(
        module,
        "_expected_shared_policy_checkpoint_binding",
        lambda: dict(binding),
    )
    cli = SimpleNamespace(
        public_seed=13,
        cuda_device="7",
        env_ready_timeout_s=1800,
        vla_ready_timeout_s=900,
    )

    args = module._runtime_args(
        cli,
        output_dir=tmp_path,
        instance_id=198,
    )

    assert args.vla_port == 0
    assert args.vla_ready_timeout_s == 900
    assert args._behavior_policy_checkpoint_binding == binding
    assert binding["resolved_path"] == str(Path(args.policy_checkpoint).resolve())


def test_start_bound_vla_binds_fresh_disabled_attempt(tmp_path, monkeypatch):
    module = _module()
    events = []
    process = SimpleNamespace(pid=17)
    binding_id = "manual:test-attempt"
    digest = hashlib.sha256(binding_id.encode("utf-8")).hexdigest()
    checkpoint_binding = {"binding_sha256": "checkpoint"}

    class Model:
        def __init__(self, endpoint):
            events.append(("client", endpoint))

        def healthz(self, **kwargs):
            events.append(("healthz", kwargs))
            return {
                "config_name": "pi05_behavior",
                "actions_enabled": False if len(events) > 4 else True,
                "binding_digest": digest if len(events) > 4 else None,
            }

        def disable_actions(self):
            events.append(("disable",))
            return {"actions_enabled": False}

        def bind_actions(self, value):
            events.append(("bind", value))
            return {
                "actions_enabled": False,
                "binding_digest": digest,
            }

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(
        module,
        "start_vla_server",
        lambda _args, *, output_dir: (
            events.append(("start", output_dir))
            or ("http://127.0.0.1:1234", process)
        ),
    )
    monkeypatch.setattr(module, "BehaviorVLAClient", Model)
    monkeypatch.setattr(
        module,
        "_terminate_process",
        lambda _process: events.append(("terminate",)),
    )
    args = SimpleNamespace(
        _behavior_policy_checkpoint_binding=checkpoint_binding,
    )

    endpoint, actual_process, model, health = module._start_bound_vla(
        args,
        output_dir=tmp_path,
        binding_id=binding_id,
    )

    assert endpoint == "http://127.0.0.1:1234"
    assert actual_process is process
    assert isinstance(model, Model)
    assert health["actions_enabled"] is False
    assert [event[0] for event in events] == [
        "start",
        "client",
        "healthz",
        "disable",
        "bind",
        "healthz",
    ]
    assert events[2][1]["expected_checkpoint_binding"] == checkpoint_binding
    assert events[4] == ("bind", binding_id)


def test_start_bound_vla_failure_closes_client_and_owned_process(
    tmp_path,
    monkeypatch,
):
    module = _module()
    events = []
    process = SimpleNamespace(pid=18)

    class Model:
        def __init__(self, _endpoint):
            pass

        def healthz(self, **_kwargs):
            events.append("healthz")
            return {
                "config_name": "pi05_behavior",
                "actions_enabled": True,
                "binding_digest": None,
            }

        def disable_actions(self):
            events.append("disable")
            return {"actions_enabled": False}

        def bind_actions(self, _value):
            events.append("bind")
            raise RuntimeError("binding failed")

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        module,
        "start_vla_server",
        lambda _args, *, output_dir: (
            "http://127.0.0.1:1235",
            process,
        ),
    )
    monkeypatch.setattr(module, "BehaviorVLAClient", Model)
    monkeypatch.setattr(
        module,
        "_terminate_process",
        lambda actual: events.append(("terminate", actual)),
    )

    with pytest.raises(RuntimeError, match="binding failed"):
        module._start_bound_vla(
            SimpleNamespace(_behavior_policy_checkpoint_binding={}),
            output_dir=tmp_path,
            binding_id="manual:failed",
        )

    assert events[-2:] == ["close", ("terminate", process)]


def test_start_bound_vla_client_constructor_failure_terminates_owned_process(
    tmp_path,
    monkeypatch,
):
    module = _module()
    process = SimpleNamespace(pid=20)
    terminated = []

    class FailingModel:
        def __init__(self, _endpoint):
            raise KeyboardInterrupt

    monkeypatch.setattr(
        module,
        "start_vla_server",
        lambda _args, *, output_dir: (
            "http://127.0.0.1:1237",
            process,
        ),
    )
    monkeypatch.setattr(module, "BehaviorVLAClient", FailingModel)
    monkeypatch.setattr(
        module,
        "_terminate_process",
        lambda actual: terminated.append(actual),
    )

    with pytest.raises(KeyboardInterrupt):
        module._start_bound_vla(
            SimpleNamespace(_behavior_policy_checkpoint_binding={}),
            output_dir=tmp_path,
            binding_id="manual:constructor-failed",
        )

    assert terminated == [process]


def test_tool_process_gate_rejects_only_dead_required_process():
    module = _module()
    alive = SimpleNamespace(poll=lambda: None)
    dead = SimpleNamespace(poll=lambda: 1)

    assert (
        module._tool_process_rejection_reason(
            "observe",
            env_proc=alive,
            vla_proc=dead,
        )
        is None
    )
    assert "VLA process exited" in module._tool_process_rejection_reason(
        "pi0_nav_pick",
        env_proc=alive,
        vla_proc=dead,
    )
    assert "env process exited" in module._tool_process_rejection_reason(
        "observe",
        env_proc=dead,
        vla_proc=alive,
    )


def test_start_vla_server_keyboard_interrupt_terminates_owned_process(
    tmp_path,
    monkeypatch,
):
    from robots.behavior import runtime

    events = []

    class Process:
        pid = 19

        @staticmethod
        def poll():
            return None

    process = Process()

    class Client:
        def __init__(self, _endpoint):
            pass

        def healthz(self, **_kwargs):
            raise KeyboardInterrupt

        def close(self):
            events.append("client_close")

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(runtime, "_record_owned_process_group", lambda _p: None)
    monkeypatch.setattr(runtime, "_runtime_env", lambda _args: {})
    monkeypatch.setattr(runtime, "_terminate_process", lambda _p: events.append("kill"))
    monkeypatch.setattr(runtime, "BehaviorVLAClient", Client)
    args = SimpleNamespace(
        vla_port=1236,
        behavior_python=sys.executable,
        policy_checkpoint=tmp_path / "checkpoint",
        seed=0,
        behavior_repo=tmp_path,
        vla_ready_timeout_s=30,
        _behavior_policy_checkpoint_binding={},
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.start_vla_server(args, output_dir=tmp_path)

    assert events == ["kill", "client_close"]


def test_start_vla_server_client_constructor_failure_terminates_owned_process(
    tmp_path,
    monkeypatch,
):
    from robots.behavior import runtime

    killed = []

    class Process:
        pid = 21

        @staticmethod
        def poll():
            return None

    process = Process()

    class FailingClient:
        def __init__(self, _endpoint):
            raise KeyboardInterrupt

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(runtime, "_record_owned_process_group", lambda _p: None)
    monkeypatch.setattr(runtime, "_runtime_env", lambda _args: {})
    monkeypatch.setattr(runtime, "_terminate_process", lambda proc: killed.append(proc))
    monkeypatch.setattr(runtime, "BehaviorVLAClient", FailingClient)
    args = SimpleNamespace(
        vla_port=1238,
        behavior_python=sys.executable,
        policy_checkpoint=tmp_path / "checkpoint",
        seed=0,
        behavior_repo=tmp_path,
        vla_ready_timeout_s=30,
        _behavior_policy_checkpoint_binding={},
    )

    with pytest.raises(KeyboardInterrupt):
        runtime.start_vla_server(args, output_dir=tmp_path)

    assert killed == [process]


def test_start_vla_server_health_client_close_failure_terminates_owned_process(
    tmp_path,
    monkeypatch,
):
    from robots.behavior import runtime

    killed = []

    class Process:
        pid = 22

        @staticmethod
        def poll():
            return None

    process = Process()

    class CloseFailingClient:
        def __init__(self, _endpoint):
            pass

        def healthz(self, **_kwargs):
            return {"config_name": "pi05_behavior"}

        def close(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(runtime, "_record_owned_process_group", lambda _p: None)
    monkeypatch.setattr(runtime, "_runtime_env", lambda _args: {})
    monkeypatch.setattr(runtime, "_terminate_process", lambda proc: killed.append(proc))
    monkeypatch.setattr(runtime, "BehaviorVLAClient", CloseFailingClient)
    args = SimpleNamespace(
        vla_port=1239,
        behavior_python=sys.executable,
        policy_checkpoint=tmp_path / "checkpoint",
        seed=0,
        behavior_repo=tmp_path,
        vla_ready_timeout_s=30,
        _behavior_policy_checkpoint_binding={},
    )

    with pytest.raises(RuntimeError, match="close failed"):
        runtime.start_vla_server(args, output_dir=tmp_path)

    assert killed == [process]
