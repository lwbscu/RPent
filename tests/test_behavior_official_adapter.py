"""Critical contracts for the BEHAVIOR-only official-RPent adapter."""

from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.behavior import cli
from robots.behavior import planner as behavior_planner
from robots.behavior.codex_planner import _codex_mcp_config_overrides
from robots.behavior.dashboard_sink import FileDashboardSink
from robots.behavior.spec import RunConfig


def test_codex_adapter_forces_tool_only_surface() -> None:
    overrides = set(
        _codex_mcp_config_overrides(
            mcp_url="http://127.0.0.1:1234/mcp/",
            base_url=None,
            tool_timeout_sec=60,
            reasoning_effort="high",
            tool_only=True,
        )
    )

    assert "mcp_servers={}" in overrides
    assert "features.shell_tool=false" in overrides
    assert "features.unified_exec=false" in overrides
    assert "features.apps=false" in overrides
    assert "features.multi_agent=false" in overrides
    assert "features.plugins=false" in overrides
    assert "features.plugin_sharing=false" in overrides
    assert "features.remote_plugin=false" in overrides
    assert "features.memories=false" in overrides
    assert "features.goals=false" in overrides
    assert "features.hooks=false" in overrides
    assert 'web_search="disabled"' in overrides
    assert "tools_view_image=false" in overrides


def test_behavior_planner_builder_never_uses_public_codex(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Codex:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(behavior_planner, "CodexPlanner", _Codex)

    built = behavior_planner.build_behavior_planner(
        "codex",
        output_dir=tmp_path,
        recipe_tag="picking_up_trash_s13",
        model="gpt-5.5",
        reasoning_effort="high",
        planner_timeout_s=60,
    )

    assert isinstance(built, _Codex)
    assert captured["tool_only"] is True
    assert captured["repo_root"] == tmp_path
    assert captured["extra_dirs"] == []
    assert captured["output_path"] == tmp_path / "codex_picking_up_trash_s13.txt"


def test_event_sink_records_done_after_finalization_without_local_dashboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    event_path = tmp_path / "dashboard_events.jsonl"
    dashboard = FileDashboardSink(event_path)

    class _Prompts:
        def render(self, _kind: str, *, variables: dict[str, object]) -> str:
            assert variables["output_dir"] == tmp_path
            return "prompt"

    class _Toolkit:
        def close(self) -> None:
            pass

    class _Planner:
        def solve(self, **_kwargs):
            return SimpleNamespace(
                finish_result=None,
                messages=[],
                stats={},
                error=None,
            )

    run_config = RunConfig(
        recipe_tag="picking_up_trash_s15",
        output_dir=tmp_path,
        prompt_vars={},
        dashboard_state=dashboard,
        task_desc={"task": "picking_up_trash"},
        resource_policy="frozen_local",
    )

    def finalize(_outcome):
        dashboard.on_event({"type": "finalized"})
        return {"run_status": "completed"}

    spec = SimpleNamespace(
        name="behavior",
        prompts=_Prompts(),
        add_cli_args=lambda _parser, use_dashboard: None,
        prepare_resources=lambda _args: object(),
        parse_config=lambda _args: run_config,
        init_runtime=lambda _args, _output: ([], {}),
        resource_policy="frozen_local",
        run_planner=None,
        finalize_run=finalize,
    )
    monkeypatch.setattr(cli, "get_env_spec", lambda: spec)
    monkeypatch.setattr(
        cli, "build_behavior_planner", lambda *_args, **_kwargs: _Planner()
    )
    monkeypatch.setattr(cli, "get_toolkit", lambda **_kwargs: _Toolkit())
    monkeypatch.setattr(
        sys,
        "argv",
        ["behavior-cli", "--env", "behavior", "--output-dir", str(tmp_path)],
    )

    assert cli.main() == 0
    records = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-2:] == [
        {"channel": "event", "payload": {"type": "finalized"}},
        {"channel": "done", "payload": {"terminated": None}},
    ]
    assert sum(record["channel"] == "done" for record in records) == 1


def test_sigterm_unwinds_through_behavior_cleanup_and_finalizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    previous = signal.getsignal(signal.SIGTERM)

    class _Prompts:
        def render(self, _kind: str, *, variables: dict[str, object]) -> str:
            assert variables["output_dir"] == tmp_path
            return "prompt"

    class _Resource:
        def stop(self) -> None:
            events.append("resource.stop")

    class _Toolkit:
        def close(self) -> None:
            events.append("toolkit.close")

    class _Planner:
        def solve(self, **_kwargs):
            os.kill(os.getpid(), signal.SIGTERM)
            raise AssertionError("SIGTERM must interrupt the planner call")

    run_config = RunConfig(
        recipe_tag="picking_up_trash_s13",
        output_dir=tmp_path,
        prompt_vars={},
        dashboard_state=None,
        task_desc={"task": "picking_up_trash"},
        resource_policy="frozen_local",
    )

    def finalize(outcome):
        events.append("finalize")
        assert outcome.cleanup["complete"] is True
        assert "termination signal" in str(outcome.error)
        return {"run_status": "completed"}

    spec = SimpleNamespace(
        name="behavior",
        prompts=_Prompts(),
        add_cli_args=lambda _parser, use_dashboard: None,
        prepare_resources=lambda _args: object(),
        parse_config=lambda _args: run_config,
        init_runtime=lambda _args, _output: ([_Resource()], {}),
        resource_policy="frozen_local",
        run_planner=None,
        finalize_run=finalize,
    )
    monkeypatch.setattr(cli, "get_env_spec", lambda: spec)
    monkeypatch.setattr(
        cli, "build_behavior_planner", lambda *_args, **_kwargs: _Planner()
    )
    monkeypatch.setattr(cli, "get_toolkit", lambda **_kwargs: _Toolkit())
    monkeypatch.setattr(
        sys,
        "argv",
        ["behavior-cli", "--env", "behavior", "--output-dir", str(tmp_path)],
    )

    assert cli.main() == 128 + signal.SIGTERM
    assert events == ["toolkit.close", "resource.stop", "finalize"]
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.parametrize("signal_phase", ["cleanup", "finalizer"])
def test_first_sigterm_during_sealing_does_not_interrupt_artifacts(
    tmp_path: Path,
    monkeypatch,
    signal_phase: str,
) -> None:
    events: list[str] = []
    previous = signal.getsignal(signal.SIGTERM)

    class _Prompts:
        def render(self, _kind: str, *, variables: dict[str, object]) -> str:
            assert variables["output_dir"] == tmp_path
            return "prompt"

    class _Resource:
        def stop(self) -> None:
            events.append("resource.stop")

    class _Toolkit:
        def close(self) -> None:
            events.append("toolkit.close.before")
            if signal_phase == "cleanup":
                os.kill(os.getpid(), signal.SIGTERM)
            events.append("toolkit.close.after")

    class _Planner:
        def solve(self, **_kwargs):
            return SimpleNamespace(
                finish_result=None,
                messages=[],
                stats={},
                error=None,
            )

    run_config = RunConfig(
        recipe_tag="picking_up_trash_s13",
        output_dir=tmp_path,
        prompt_vars={},
        dashboard_state=None,
        task_desc={"task": "picking_up_trash"},
        resource_policy="frozen_local",
    )

    def finalize(outcome):
        events.append("finalize.before")
        assert outcome.cleanup["complete"] is True
        assert outcome.error is None
        if signal_phase == "finalizer":
            os.kill(os.getpid(), signal.SIGTERM)
        events.append("finalize.after")
        return {"run_status": "completed"}

    spec = SimpleNamespace(
        name="behavior",
        prompts=_Prompts(),
        add_cli_args=lambda _parser, use_dashboard: None,
        prepare_resources=lambda _args: object(),
        parse_config=lambda _args: run_config,
        init_runtime=lambda _args, _output: ([_Resource()], {}),
        resource_policy="frozen_local",
        run_planner=None,
        finalize_run=finalize,
    )
    monkeypatch.setattr(cli, "get_env_spec", lambda: spec)
    monkeypatch.setattr(
        cli, "build_behavior_planner", lambda *_args, **_kwargs: _Planner()
    )
    monkeypatch.setattr(cli, "get_toolkit", lambda **_kwargs: _Toolkit())
    monkeypatch.setattr(
        sys,
        "argv",
        ["behavior-cli", "--env", "behavior", "--output-dir", str(tmp_path)],
    )

    assert cli.main() == 128 + signal.SIGTERM
    assert events == [
        "toolkit.close.before",
        "toolkit.close.after",
        "resource.stop",
        "finalize.before",
        "finalize.after",
    ]
    assert signal.getsignal(signal.SIGTERM) == previous
