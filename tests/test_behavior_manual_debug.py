from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "behavior_manual_debug.py"
    spec = importlib.util.spec_from_file_location("behavior_manual_debug", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_toolkit_injects_same_runtime_control_objects(tmp_path, monkeypatch):
    module = _module()
    captured = {}
    toolkit = object()

    def fake_get_toolkit(**kwargs):
        captured.update(kwargs)
        return toolkit

    monkeypatch.setattr(module, "get_toolkit", fake_get_toolkit)
    cli = SimpleNamespace(max_tool_calls=11, max_wall_clock_s=22.0)
    args = SimpleNamespace(
        task_name="picking_up_trash",
        public_seed=13,
        max_episode_steps=99,
    )
    env = object()
    state = object()
    resource = object()
    arbiter = object()
    latch = object()

    actual = module._build_toolkit(
        cli=cli,
        args=args,
        output_dir=tmp_path,
        env=env,
        model=None,
        initial_observation={"obs": 1},
        initial_info={"info": 1},
        state=state,
        resource=resource,
        arbiter=arbiter,
        success_latch=latch,
    )

    assert actual is toolkit
    assert captured["dashboard"] is state
    kwargs = captured["primitives_kwargs"]
    assert kwargs["env"] is env
    assert kwargs["model"] is None
    assert kwargs["_dashboard_runtime_resource"] is resource
    assert kwargs["_dashboard_command_arbiter"] is arbiter
    assert kwargs["_dashboard_success_latch"] is latch
    assert kwargs["_dashboard_motion_allowed"] is True
    assert kwargs["_dashboard_observe_allowed"] is True


def test_get_toolkit_contract_activates_and_binds_controller(tmp_path):
    module = _module()
    events = []

    class Resource:
        toolkit = None
        dashboard_controller = None

        def attach_dashboard_control(
            self,
            *,
            toolkit,
            controller,
            dashboard_state,
        ):
            events.append("resource.bind")
            self.toolkit = toolkit
            self.dashboard_controller = controller
            self.dashboard_state = dashboard_state

        def quiesce_control(self):
            events.append("resource.quiesce")

    class State:
        def bind_controller(self, controller):
            events.append("state.bind")
            self.controller = controller

        def unbind_controller(self, _controller=None):
            events.append("state.unbind")

    class Controller:
        def __init__(self):
            self.bound = None

        def bind_toolkit(self, toolkit):
            events.append("controller.bind")
            self.bound = toolkit

        def snapshot(self):
            return {"available": True}

        def activate(self):
            events.append("controller.activate")

        def quiesce(self):
            pass

        def drain(self, _timeout):
            return True

        def close(self, **_kwargs):
            pass

    class Toolkit:
        def __init__(self, *, primitives_kwargs, video_path, dashboard):
            events.append("toolkit.create")
            self._runtime_resource = primitives_kwargs[
                "_dashboard_runtime_resource"
            ]
            self._dashboard = dashboard
            self._dashboard_controller = None

        def activate_dashboard_control(self):
            controller = Controller()
            controller.bind_toolkit(self)
            self._dashboard.bind_controller(controller)
            self._dashboard_controller = controller
            self._runtime_resource.attach_dashboard_control(
                toolkit=self,
                controller=controller,
                dashboard_state=self._dashboard,
            )
            controller.activate()

        def close(self):
            pass

    resource = Resource()
    state = State()
    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr("robots.behavior.toolkit.BehaviorToolkit", Toolkit)
    try:
        toolkit = module.get_toolkit(
            primitives_kwargs={
                "_dashboard_runtime_resource": resource,
            },
            dashboard=state,
        )
    finally:
        monkeypatch.undo()

    assert resource.toolkit is toolkit
    assert resource.dashboard_controller is toolkit._dashboard_controller
    assert events == [
        "toolkit.create",
        "controller.bind",
        "state.bind",
        "resource.bind",
        "controller.activate",
    ]


def test_cleanup_order_is_toolkit_runtime_dashboard_and_never_double_stops():
    module = _module()
    events = []

    class Owner:
        def __init__(self, name):
            self.name = name
            self.calls = 0

        def close(self):
            self.calls += 1
            events.append(self.name)

    class Dashboard:
        calls = 0

        def stop(self):
            self.calls += 1
            events.append("dashboard")

    toolkit = Owner("toolkit")
    resource = Owner("runtime")
    dashboard = Dashboard()

    errors = module._cleanup_owned(
        toolkit=toolkit,
        resource=resource,
        dashboard=dashboard,
        dashboard_started=True,
        orphan_model=None,
    )

    assert errors == []
    assert events == ["toolkit", "runtime", "dashboard"]
    assert toolkit.calls == resource.calls == dashboard.calls == 1


def test_cleanup_continues_after_startup_failure_and_preserves_order():
    module = _module()
    events = []

    class Toolkit:
        def close(self):
            events.append("toolkit")
            raise RuntimeError("quiesce failed")

    class Resource:
        def close(self):
            events.append("runtime")

    class Dashboard:
        def stop(self):
            events.append("dashboard")

    errors = module._cleanup_owned(
        toolkit=Toolkit(),
        resource=Resource(),
        dashboard=Dashboard(),
        dashboard_started=True,
        orphan_model=None,
    )

    assert events == ["toolkit", "runtime", "dashboard"]
    assert len(errors) == 1
    assert errors[0].startswith("toolkit: RuntimeError:")


def test_planner_only_rejects_pi0_but_other_tools_require_live_env():
    module = _module()
    dead = SimpleNamespace(poll=lambda: 1)
    live = SimpleNamespace(poll=lambda: None)

    assert "--planner-only" in module._tool_rejection(
        "pi0_nav_pick",
        terminal=False,
        env_proc=live,
        planner_only=True,
    )
    assert module._tool_rejection(
        "observe",
        terminal=False,
        env_proc=live,
        planner_only=True,
    ) is None
    assert "env process exited" in module._tool_rejection(
        "move_to",
        terminal=False,
        env_proc=dead,
        planner_only=True,
    )
    assert "terminal state is latched" in module._tool_rejection(
        "observe",
        terminal=True,
        env_proc=live,
        planner_only=True,
    )


def test_default_mode_keeps_pi0_when_vla_is_live():
    module = _module()
    live = SimpleNamespace(poll=lambda: None)

    assert module._tool_rejection(
        "pi0_nav_pick",
        terminal=False,
        env_proc=live,
        planner_only=False,
        vla_proc=live,
    ) is None


def test_cli_defaults_to_vla_and_planner_only_is_explicit():
    module = _module()

    default = module._parser().parse_args([])
    planner_only = module._parser().parse_args(["--planner-only"])

    assert default.planner_only is False
    assert default.vla_ready_timeout_s == 1800
    assert planner_only.planner_only is True


def test_manifest_final_status_distinguishes_startup_and_cleanup_failure():
    module = _module()

    assert module._final_manifest_status("ready", []) == "stopped"
    assert module._final_manifest_status("ready", ["x"]) == "cleanup_failed"
    assert module._final_manifest_status("startup_failed", []) == "startup_failed"
    assert (
        module._final_manifest_status("startup_failed", ["x"])
        == "startup_failed_cleanup_failed"
    )
