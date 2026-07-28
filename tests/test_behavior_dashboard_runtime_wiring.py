from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robots.behavior import runtime as behavior_runtime
from robots.behavior.dashboard_control import (
    BehaviorCommandArbiter,
    BehaviorRawSuccessLatch,
)
from robots.behavior.runtime import BehaviorRuntimeResources
from robots.behavior.spec import BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S
from robots.behavior.toolkit import BehaviorToolkit

_RUN_NONCE = "a" * 32
_ATTEMPT_NONCE = "b" * 32


def _env_server_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        behavior_python="/usr/bin/python3",
        suite="behavior_2025_challenge",
        task=1,
        task_name="picking_up_trash",
        activity_definition_id=0,
        activity_instance_id=198,
        activity_instance_dir=str(tmp_path),
        scene_model="house_double_floor_lower",
        seed=0,
        public_seed=13,
        behavior_attempt_index=1,
        behavior_controller_mode="hybrid",
        max_episode_steps=100,
        behavior_config=None,
        behavior_repo=str(tmp_path),
        env_ready_timeout_s=30.0,
    )


def test_start_env_server_interrupt_before_ready_terminates_owned_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(pid=4312, stdout=iter(()), poll=lambda: None)
    terminated: list[Any] = []

    class InterruptQueue:
        def get(self, *, timeout: float) -> dict[str, Any]:
            assert timeout == 2.0
            raise KeyboardInterrupt

    monkeypatch.setattr(
        behavior_runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(behavior_runtime, "_runtime_env", lambda _args: {})
    monkeypatch.setattr(
        behavior_runtime,
        "_record_owned_process_group",
        lambda _proc: None,
    )
    monkeypatch.setattr(
        behavior_runtime.threading,
        "Thread",
        lambda **_kwargs: SimpleNamespace(start=lambda: None),
    )
    monkeypatch.setattr(behavior_runtime.queue, "Queue", InterruptQueue)
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        terminated.append,
    )

    with pytest.raises(KeyboardInterrupt):
        behavior_runtime.start_env_server(
            _env_server_args(tmp_path),
            output_dir=tmp_path,
        )

    assert terminated == [process]


def test_start_env_server_early_exit_still_terminates_process_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(pid=4313, stdout=iter(()), poll=lambda: 7)
    terminated: list[Any] = []
    monkeypatch.setattr(
        behavior_runtime.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(behavior_runtime, "_runtime_env", lambda _args: {})
    monkeypatch.setattr(
        behavior_runtime,
        "_record_owned_process_group",
        lambda _proc: None,
    )
    monkeypatch.setattr(
        behavior_runtime.threading,
        "Thread",
        lambda **_kwargs: SimpleNamespace(start=lambda: None),
    )
    monkeypatch.setattr(
        behavior_runtime,
        "_terminate_process",
        terminated.append,
    )

    with pytest.raises(RuntimeError, match="exited before ready"):
        behavior_runtime.start_env_server(
            _env_server_args(tmp_path),
            output_dir=tmp_path,
        )

    assert terminated == [process]


def _signed_success_receipt(**updates: Any) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "source": 'info["done"]["success"]',
        "run_nonce": _RUN_NONCE,
        "attempt_nonce": _ATTEMPT_NONCE,
        "attempt_index": 1,
        "env_step": 9,
        "raw_done": {"success": True},
    }
    receipt.update(updates)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _resign_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt = dict(receipt)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _publication_ready_toolkit(
    tmp_path: Path,
    *,
    manual_handler: Any,
) -> BehaviorToolkit:
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._closed = False
    toolkit._command_arbiter = SimpleNamespace(
        require_manual_permit=lambda command_id: (
            None
            if command_id == "test-command"
            else (_ for _ in ()).throw(RuntimeError("wrong manual permit"))
        )
    )
    toolkit._manual_intervention_latch = threading.Event()
    toolkit._success_latch = None
    toolkit._official_task_success = False
    toolkit._shared_success_evidence = None
    toolkit._behavior_phase = "explore"
    toolkit._tool_calls = 7
    toolkit._tool_trace = []
    toolkit._task_spec = SimpleNamespace(
        tag=lambda public_seed: f"picking_up_trash_s{public_seed}"
    )
    toolkit._primitives = SimpleNamespace(
        output_dir=tmp_path,
        public_seed=0,
        attempt_index=1,
        _vla_invocations=0,
        _global_vla_chunks=0,
        total_env_steps=0,
        dashboard_manual_command=manual_handler,
    )
    toolkit._has_verified_raw_success = lambda: True
    toolkit._symbolic_recipe = lambda: [
        {
            "schema_version": 1,
            "kind": "task_level_symbolic_recipe",
            "task": "picking_up_trash",
            "source": "raw_official_success_v1",
            "policy": "Use fresh public evidence.",
        }
    ]
    toolkit.validate_symbolic_publication = lambda records: None
    toolkit._seal_current_attempt = lambda *, result: None
    return toolkit


class _OrderedOwner:
    def __init__(self, name: str, events: list[str], *, drained: bool = True) -> None:
        self.name = name
        self.events = events
        self.drained = drained
        self.timeouts: list[float] = []

    def quiesce(self) -> None:
        self.events.append(f"{self.name}.quiesce")

    def drain(self, timeout_s: float) -> bool:
        assert timeout_s >= 0.0
        self.timeouts.append(timeout_s)
        self.events.append(f"{self.name}.drain")
        return self.drained

    def close(self, *, timeout_s: float) -> None:
        assert timeout_s >= 0.0
        self.events.append(f"{self.name}.close")


class _Dashboard:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.controller: Any = None
        self.run_id = "behavior/test"

    def bind_controller(self, controller: Any) -> None:
        assert self.controller is None
        self.controller = controller
        self.events.append("state.bind")

    def unbind_controller(self, controller: Any = None) -> None:
        assert controller is None or controller is self.controller
        self.events.append("state.unbind")
        self.controller = None

    def update_control_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        controller: Any,
    ) -> bool:
        assert isinstance(snapshot, dict)
        return controller is self.controller


def test_runtime_resource_quiesces_before_env_transport_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    controller = _OrderedOwner("controller", events)
    arbiter = _OrderedOwner("arbiter", events)
    dashboard = _Dashboard(events)
    dashboard.controller = controller
    client = SimpleNamespace(close=lambda: events.append("transport.close"))
    resource = BehaviorRuntimeResources(
        output_dir=tmp_path,
        env_rpc_client=client,
        command_arbiter=arbiter,
        success_latch=object(),
        dashboard_controller=controller,
        dashboard_state=dashboard,
    )
    monkeypatch.setattr(
        "robots.behavior.runtime.stop_env_server",
        lambda proc, *, output_dir: events.append("env.stop"),
    )
    monkeypatch.setattr(
        "robots.behavior.runtime._terminate_process",
        lambda proc: events.append("vla.stop"),
    )

    resource.close()
    resource.close()

    assert events == [
        "controller.quiesce",
        "arbiter.quiesce",
        "controller.drain",
        "arbiter.drain",
        "state.unbind",
        "controller.close",
        "arbiter.close",
        "env.stop",
        "vla.stop",
        "transport.close",
    ]
    assert controller.timeouts[0] > 360.0
    assert arbiter.timeouts[0] > 360.0


def test_runtime_resource_does_not_close_env_when_control_drain_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    controller = _OrderedOwner("controller", events, drained=False)
    resource = BehaviorRuntimeResources(
        output_dir=tmp_path,
        command_arbiter=_OrderedOwner("arbiter", events),
        dashboard_controller=controller,
        dashboard_state=_Dashboard(events),
    )
    monkeypatch.setattr(
        "robots.behavior.runtime.stop_env_server",
        lambda proc, *, output_dir: events.append("env.stop"),
    )

    with pytest.raises(TimeoutError, match="controller did not drain"):
        resource.close()

    assert "env.stop" not in events
    assert resource._closed is False


def test_runtime_resource_close_is_single_flight_across_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    entered_stop = threading.Event()
    release_stop = threading.Event()
    resource = BehaviorRuntimeResources(output_dir=tmp_path)

    def stop_env(proc: Any, *, output_dir: Path) -> None:
        del proc, output_dir
        events.append("env.stop")
        entered_stop.set()
        assert release_stop.wait(1.0)

    monkeypatch.setattr("robots.behavior.runtime.stop_env_server", stop_env)
    monkeypatch.setattr(
        "robots.behavior.runtime._terminate_process",
        lambda proc: events.append("vla.stop"),
    )
    errors: list[BaseException] = []

    def close() -> None:
        try:
            resource.close()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=close)
    second = threading.Thread(target=close)
    first.start()
    assert entered_stop.wait(1.0)
    second.start()
    time.sleep(0.02)
    release_stop.set()
    first.join(1.0)
    second.join(1.0)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert events == ["env.stop", "vla.stop"]


def test_execute_tool_holds_agent_transaction_around_complete_call() -> None:
    events: list[str] = []

    class Arbiter:
        @contextlib.contextmanager
        def agent_transaction(self):
            events.append("agent.enter")
            try:
                yield
            finally:
                events.append("agent.exit")

    toolkit = object.__new__(BehaviorToolkit)
    toolkit._command_arbiter = Arbiter()
    toolkit._execute_tool_lock = threading.Lock()
    toolkit._execute_tool_locked = lambda name, args: events.append("tool.complete")

    toolkit.execute_tool("observe", {"camera": "head"})

    assert events == ["agent.enter", "tool.complete", "agent.exit"]


def test_manual_command_uses_public_primitive_and_shared_success_latch() -> None:
    calls: list[dict[str, Any]] = []

    class Latch:
        def observe(self, result: dict[str, Any]) -> bool:
            calls.append({"latched": result})
            return True

    receipt = _signed_success_receipt()
    result = {
        "source": "dashboard_manual",
        "target": "left_arm",
        "action": "open",
        "primitive_success": True,
        "task_success": True,
        "official_success_source": 'info["done"]["success"]',
        "official_success_receipt": receipt,
    }
    primitives = SimpleNamespace(
        dashboard_manual_command=lambda **kwargs: calls.append(kwargs) or result,
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._closed = False
    toolkit._primitives = primitives
    toolkit._success_latch = Latch()
    toolkit._command_arbiter = SimpleNamespace(
        require_manual_permit=lambda command_id: calls.append(
            {"permit": command_id}
        )
    )
    toolkit._official_task_success = False

    returned = toolkit.dashboard_manual_command(
        target="left_arm",
        action="open",
        camera="head",
        permit_command_id="manual-command",
    )

    assert returned is result
    assert calls[0] == {"permit": "manual-command"}
    assert calls[1] == {
        "target": "left_arm",
        "action": "open",
        "camera": "head",
    }
    assert calls[2] == {"latched": result}
    assert toolkit._official_task_success is True


def test_capability_policy_cannot_be_overridden_by_env_claim() -> None:
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._dashboard_motion_allowed = False
    toolkit._dashboard_observe_allowed = True
    toolkit._dashboard_control_unavailable_reason = "pure VLA owns motion"
    toolkit._primitives = SimpleNamespace(
        dashboard_control_capabilities=lambda: {
            "simulation_identity": "behavior_omnigibson_r1pro",
            "motion_available": True,
            "observe_available": True,
            "planner_available": True,
            "position_control_ready": True,
        }
    )

    capability = toolkit.dashboard_control_capabilities()

    assert capability["motion_available"] is False
    assert capability["observe_available"] is True
    assert capability["motion_unavailable_reason"] == "pure VLA owns motion"


def test_manual_task_success_flag_without_receipt_does_not_latch() -> None:
    latch = BehaviorRawSuccessLatch(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._closed = False
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
        dashboard_manual_command=lambda **kwargs: {
            "primitive_success": True,
            "task_success": True,
        },
    )
    toolkit._success_latch = latch
    toolkit._command_arbiter = SimpleNamespace(
        require_manual_permit=lambda command_id: None
    )
    toolkit._official_task_success = False
    toolkit._shared_success_evidence = None

    toolkit.dashboard_manual_command(
        target="chassis",
        action="forward",
        camera="head",
        permit_command_id="manual-command",
    )

    assert latch.is_latched() is False
    assert toolkit._official_task_success is False
    assert toolkit._has_verified_raw_success() is False


def test_toolkit_accepts_only_exact_envclient_receipt_schema() -> None:
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )

    binding = toolkit._receipt_binding_from_result(
        {"official_success_receipt": _signed_success_receipt()}
    )

    assert binding is not None
    assert binding["run_nonce"] == _RUN_NONCE


def test_toolkit_rejects_missing_receipt_schema_even_with_recomputed_hash() -> None:
    receipt = _signed_success_receipt()
    receipt.pop("schema_version")
    receipt = _resign_receipt(receipt)
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )

    assert (
        toolkit._receipt_binding_from_result(
            {"official_success_receipt": receipt}
        )
        is None
    )


@pytest.mark.parametrize(
    "receipt",
    [
        _signed_success_receipt(extra="forbidden"),
        _signed_success_receipt(schema_version=2),
        _signed_success_receipt(schema_version=True),
        _signed_success_receipt(run_nonce="A" * 32),
        _signed_success_receipt(attempt_nonce="b" * 31),
    ],
    ids=[
        "extra-key",
        "wrong-schema",
        "bool-schema",
        "uppercase-run-nonce",
        "short-attempt-nonce",
    ],
)
def test_toolkit_rejects_noncanonical_success_receipts(
    receipt: dict[str, Any],
) -> None:
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )

    assert (
        toolkit._receipt_binding_from_result(
            {"official_success_receipt": receipt}
        )
        is None
    )


def test_toolkit_rejects_numpy_bool_raw_success_without_raising() -> None:
    receipt = _signed_success_receipt()
    receipt["raw_done"] = {"success": np.bool_(True)}
    receipt["receipt_sha256"] = "0" * 64
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )

    assert (
        toolkit._receipt_binding_from_result(
            {"official_success_receipt": receipt}
        )
        is None
    )


def test_toolkit_uses_digest_comparison_for_well_formed_receipt() -> None:
    receipt = _signed_success_receipt()
    receipt["env_step"] = 10
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )

    assert (
        toolkit._receipt_binding_from_result(
            {"official_success_receipt": receipt}
        )
        is None
    )


def test_manual_command_rejects_calls_without_shared_manual_permit() -> None:
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._closed = False
    toolkit._command_arbiter = SimpleNamespace(
        require_manual_permit=lambda command_id: (_ for _ in ()).throw(
            RuntimeError("Dashboard manual primitive requires its exact command permit")
        )
    )
    toolkit._primitives = SimpleNamespace(
        dashboard_manual_command=lambda **kwargs: pytest.fail(
            "manual handler must not run"
        )
    )

    with pytest.raises(RuntimeError, match="exact command permit"):
        toolkit.dashboard_manual_command(
            target="chassis",
            action="forward",
            camera="head",
            permit_command_id="borrowed-command",
        )


def test_manual_command_permit_is_bound_to_exact_current_command() -> None:
    latch = BehaviorRawSuccessLatch(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )
    arbiter = BehaviorCommandArbiter(success_latch=latch)
    calls: list[dict[str, Any]] = []
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._closed = False
    toolkit._command_arbiter = arbiter
    toolkit._manual_intervention_latch = threading.Event()
    toolkit._success_latch = latch
    toolkit._official_task_success = False
    toolkit._shared_success_evidence = None
    toolkit._primitives = SimpleNamespace(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
        dashboard_manual_command=lambda **kwargs: calls.append(kwargs)
        or {"primitive_success": True, "task_success": False},
    )

    acquired, reason = arbiter.try_acquire_manual("current-command")
    assert acquired is True
    assert reason is None
    with pytest.raises(RuntimeError, match="exact command permit"):
        toolkit.dashboard_manual_command(
            target="chassis",
            action="observe",
            camera="head",
            permit_command_id="borrowed-command",
        )
    assert calls == []

    result = toolkit.dashboard_manual_command(
        target="chassis",
        action="observe",
        camera="head",
        permit_command_id="current-command",
    )
    assert result["primitive_success"] is True
    assert len(calls) == 1

    arbiter.release_manual("current-command")
    with pytest.raises(RuntimeError, match="exact command permit"):
        toolkit.dashboard_manual_command(
            target="chassis",
            action="observe",
            camera="head",
            permit_command_id="current-command",
        )
    assert len(calls) == 1


def test_failed_manual_motion_blocks_explore_memory_publication(
    tmp_path: Path,
) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("planner rejected")

    toolkit = _publication_ready_toolkit(tmp_path, manual_handler=fail)

    with pytest.raises(RuntimeError, match="planner rejected"):
        toolkit.dashboard_manual_command(
            target="chassis",
            action="forward",
            camera="head",
            permit_command_id="test-command",
        )
    recipe_path = toolkit.write_recipe("picking_up_trash_s0")

    assert recipe_path is None
    assert toolkit._tool_calls == 7
    result = json.loads(
        (tmp_path / "behavior_result.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (tmp_path / "picking_up_trash_s0.json").read_text(encoding="utf-8")
    )
    for payload in (result, audit):
        assert payload["task_success"] is True
        assert payload["publication_eligible"] is False
        assert payload["manual_intervention"] is True
        assert payload["publication_reason"] == "dashboard_manual_intervention"
    assert not (tmp_path / "recipe_picking_up_trash_s0.jsonl").exists()


def test_partial_manual_motion_sets_monotonic_intervention_latch(
    tmp_path: Path,
) -> None:
    toolkit = _publication_ready_toolkit(
        tmp_path,
        manual_handler=lambda **kwargs: {
            "primitive_success": False,
            "partial_motion": True,
            "task_success": False,
        },
    )

    toolkit.dashboard_manual_command(
        target="left_arm",
        action="forward",
        camera="head",
        permit_command_id="test-command",
    )

    assert toolkit._manual_intervention_latch.is_set()
    assert toolkit._tool_calls == 7


def test_observe_only_keeps_explore_memory_publication_eligible(
    tmp_path: Path,
) -> None:
    toolkit = _publication_ready_toolkit(
        tmp_path,
        manual_handler=lambda **kwargs: {
            "primitive_success": True,
            "task_success": False,
        },
    )

    toolkit.dashboard_manual_command(
        target="chassis",
        action="observe",
        camera="head",
        permit_command_id="test-command",
    )
    recipe_path = toolkit.write_recipe("picking_up_trash_s0")

    assert recipe_path == str(tmp_path / "recipe_picking_up_trash_s0.jsonl")
    assert toolkit._manual_intervention_latch.is_set() is False
    assert toolkit._tool_calls == 7
    result = json.loads(
        (tmp_path / "behavior_result.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (tmp_path / "picking_up_trash_s0.json").read_text(encoding="utf-8")
    )
    for payload in (result, audit):
        assert payload["task_success"] is True
        assert payload["publication_eligible"] is True
        assert payload["manual_intervention"] is False
        assert payload["publication_reason"] == "eligible_raw_success_explore"


def test_standalone_toolkit_close_uses_full_manual_deadline() -> None:
    events: list[str] = []
    controller = _OrderedOwner("controller", events)
    arbiter = _OrderedOwner("arbiter", events)
    dashboard = _Dashboard(events)
    dashboard.controller = controller
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._close_lock = threading.RLock()
    toolkit._closed = False
    toolkit._runtime_resource = None
    toolkit._dashboard_controller = controller
    toolkit._command_arbiter = arbiter
    toolkit._dashboard = dashboard
    toolkit._primitives = SimpleNamespace(model=None, env=None)

    toolkit.close()

    assert controller.timeouts == [BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S]
    assert arbiter.timeouts == [BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S]
    assert dashboard.controller is None


@pytest.mark.parametrize(
    "failure_stage",
    ("bind_toolkit", "bind_state", "resource_attach", "activate"),
)
def test_dashboard_activation_failure_rolls_back_every_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    events: list[str] = []

    class Controller:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            events.append("controller.create")

        def bind_toolkit(self, toolkit: Any) -> None:
            del toolkit
            events.append("controller.bind_toolkit")
            if failure_stage == "bind_toolkit":
                raise RuntimeError("bind toolkit failed")

        def activate(self) -> None:
            events.append("controller.activate")
            if failure_stage == "activate":
                raise RuntimeError("activate failed")

        def close(self, *, timeout_s: float) -> None:
            assert timeout_s == 0.0
            events.append("controller.close")

    class Dashboard(_Dashboard):
        def bind_controller(self, controller: Any) -> None:
            events.append("state.bind")
            if failure_stage == "bind_state":
                raise RuntimeError("bind state failed")
            self.controller = controller

    monkeypatch.setattr(
        "robots.behavior.dashboard_control.BehaviorDashboardController",
        Controller,
    )
    dashboard = Dashboard(events)
    resource = BehaviorRuntimeResources(output_dir=tmp_path)
    original_attach = resource.attach_dashboard_control

    def attach(**kwargs: Any) -> None:
        events.append("resource.attach")
        if failure_stage == "resource_attach":
            raise RuntimeError("resource attach failed")
        original_attach(**kwargs)

    resource.attach_dashboard_control = attach  # type: ignore[method-assign]
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._dashboard = dashboard
    toolkit._runtime_resource = resource
    toolkit._command_arbiter = object()
    toolkit._success_latch = object()
    toolkit._dashboard_motion_allowed = True
    toolkit._dashboard_observe_allowed = True
    toolkit._dashboard_control_unavailable_reason = None
    toolkit._dashboard_controller = None
    resource.toolkit = toolkit

    with pytest.raises(RuntimeError):
        toolkit.activate_dashboard_control()

    assert toolkit._dashboard_controller is None
    assert dashboard.controller is None
    assert resource.dashboard_controller is None
    assert resource.toolkit is toolkit
    assert events[-1] == "controller.close"


def test_activate_binds_one_controller_to_same_runtime_objects(tmp_path: Path) -> None:
    events: list[str] = []
    dashboard = _Dashboard(events)
    latch = BehaviorRawSuccessLatch(
        run_nonce=_RUN_NONCE,
        attempt_nonce=_ATTEMPT_NONCE,
        attempt_index=1,
    )
    arbiter = BehaviorCommandArbiter(success_latch=latch)
    resource = BehaviorRuntimeResources(
        output_dir=tmp_path,
        command_arbiter=arbiter,
        success_latch=latch,
        dashboard_state=dashboard,
    )
    env = object()
    primitives = SimpleNamespace(
        env=env,
        dashboard_control_capabilities=lambda: {
            "simulation_identity": "behavior_omnigibson_r1pro",
            "motion_available": True,
            "observe_available": True,
            "planner_available": True,
            "position_control_ready": True,
        },
        dashboard_manual_command=lambda **kwargs: {
            "primitive_success": True,
            "task_success": False,
        },
    )
    toolkit = object.__new__(BehaviorToolkit)
    toolkit._dashboard = dashboard
    toolkit._primitives = primitives
    toolkit._runtime_resource = resource
    toolkit._command_arbiter = arbiter
    toolkit._success_latch = latch
    toolkit._dashboard_motion_allowed = True
    toolkit._dashboard_observe_allowed = True
    toolkit._dashboard_control_unavailable_reason = None
    toolkit._dashboard_controller = None

    controller = toolkit.activate_dashboard_control()

    assert dashboard.controller is controller
    assert controller.arbiter is arbiter
    assert controller.success_latch is latch
    assert controller._toolkit is toolkit
    assert toolkit._primitives.env is env
    assert resource.dashboard_controller is controller
    assert resource.toolkit is toolkit
    resource.quiesce_control(timeout_s=1.0)
    assert dashboard.controller is None
