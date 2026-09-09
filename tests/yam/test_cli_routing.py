# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.yam.evaluation import finalize_run
from rpent.robots.robot_spec import RobotSpec, RunConfig


class FakePromptBundle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def render(self, name: str, *, variables: dict) -> str:
        self.calls.append((name, dict(variables)))
        return f"{name}:{variables.get('mode')}:{variables.get('memory_profile')}"


class FakePlanner:
    def __init__(self) -> None:
        self.solve_calls = 0

    def solve(self, **kwargs):
        self.solve_calls += 1
        del kwargs
        return SimpleNamespace(
            finish_result={"status": "failure"},
            messages=[],
            stats={},
            error=None,
        )


class FakeState:
    def __init__(self, records: list[SimpleNamespace] | None = None) -> None:
        self._records = records or []

    def records(self) -> list[SimpleNamespace]:
        return list(self._records)


class FakeMemory:
    def __init__(self) -> None:
        self.merge_calls: list[dict] = []

    def merge_memory(self, **kwargs):
        self.merge_calls.append(dict(kwargs))
        return {"merged": True}


class FakeToolkit:
    def __init__(
        self,
        *,
        solved: bool = False,
        finish_record: dict | None = None,
    ) -> None:
        self.memory = FakeMemory()
        self.closed = False
        self._solved = solved
        self.recipe_tags: list[str] = []
        self.state = FakeState(
            [SimpleNamespace(command={"action": "finish"}, result=finish_record)]
            if finish_record is not None
            else []
        )

    def solved(self) -> bool:
        return self._solved

    def write_recipe(self, recipe_tag: str) -> str:
        self.recipe_tags.append(recipe_tag)
        return f"{recipe_tag}.json"

    def close(self) -> None:
        self.closed = True


def _init_output_dir(output_dir, verbose: bool = False) -> Path:
    del verbose
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fake_robot_spec(name: str, tmp_path: Path) -> RobotSpec:
    prompts = FakePromptBundle()

    def add_cli_args(parser, use_dashboard: bool) -> None:
        del use_dashboard
        if name == "yam":
            parser.add_argument("--task-name", required=True)
            parser.add_argument("--task-language", default=None)
            parser.add_argument("--seed", type=int, default=0)
            parser.add_argument("--max-episode-steps", type=int, default=1000)
            parser.add_argument("--env-endpoint")
            parser.add_argument("--without-vla", action="store_true")
        else:
            parser.add_argument("--suite", required=True)
            parser.add_argument("--task", type=int, required=True)
            parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--explore-attempts-per-session", type=int, default=5)
        parser.add_argument("--explore-sessions", type=int, default=1)
        parser.add_argument("--auto-merge-memory", action="store_true", default=True)

    def parse_config(args) -> RunConfig:
        task_name = getattr(args, "task_name", f"task_{getattr(args, 'task', 0)}")
        output_dir = Path(args.output_dir or tmp_path / f"{name}-run")
        return RunConfig(
            recipe_tag=f"{name}_{task_name}_s{args.seed}",
            output_dir=output_dir,
            prompt_vars={
                "task_name": task_name,
                "mode": "explore" if args.explore else "eval",
                "memory_profile": args.memory_profile,
                "memory_dir": args.memory_dir or str(tmp_path / name / "memory"),
                "memory_inbox": str(
                    Path(args.memory_dir or tmp_path / name / "memory")
                    / "_internal"
                    / "inbox"
                    / f"{name}_{task_name}_s{args.seed}"
                ),
            },
            task_desc={"env": name, "task_name": task_name, "seed": args.seed},
        )

    def init_runtime(args, output_dir, dashboard_events, components):
        del args, output_dir, dashboard_events, components
        return [], {"env": "fake-env"}

    return RobotSpec(
        name=name,
        supports_exploration=True,
        default_memory_profile="local" if name == "yam" else "hf",
        finalize_run=finalize_run if name == "yam" else None,
        prompts=prompts,
        add_cli_args=add_cli_args,
        parse_config=parse_config,
        init_runtime=init_runtime,
    )


def test_cli_routes_yam_explore_to_local_memory_and_sessions(
    monkeypatch,
    tmp_path,
) -> None:
    import rpent.cli.main as cli_main

    spec = _fake_robot_spec("yam", tmp_path)
    planners: list[FakePlanner] = []
    toolkits: list[FakeToolkit] = []
    get_toolkit_calls: list[dict] = []

    monkeypatch.setattr(cli_main, "enumerate_robots", lambda: ("libero", "yam"))
    monkeypatch.setattr(cli_main, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(
        cli_main,
        "init_output_dir",
        _init_output_dir,
    )
    monkeypatch.setattr(
        cli_main.MemoryManager,
        "sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote sync")),
    )

    def build_planner(*args, **kwargs):
        del args, kwargs
        planner = FakePlanner()
        planners.append(planner)
        return planner

    def get_toolkit(name, **kwargs):
        get_toolkit_calls.append({"name": name, **kwargs})
        toolkit = FakeToolkit()
        toolkits.append(toolkit)
        return toolkit

    monkeypatch.setattr(cli_main, "build_planner", build_planner)
    monkeypatch.setattr(cli_main, "get_toolkit", get_toolkit)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rpent",
            "--robot",
            "yam",
            "--task-name",
            "place_cube",
            "--seed",
            "7",
            "--env-endpoint",
            "http://127.0.0.1:8110",
            "--without-vla",
            "--explore",
            "--explore-sessions",
            "2",
            "--explore-attempts-per-session",
            "3",
            "--output-dir",
            str(tmp_path / "yam-out"),
            "--max-turns",
            "1",
        ],
    )

    assert cli_main.main() == 0

    assert len(planners) == 2
    assert len(toolkits) == 2
    assert [call["name"] for call in get_toolkit_calls] == ["yam", "yam"]
    assert [call["mode"] for call in get_toolkit_calls] == [
        "exploration",
        "exploration",
    ]
    assert [call["attempts_per_session"] for call in get_toolkit_calls] == [3, 3]
    assert [call["state_output_dir"].name for call in get_toolkit_calls] == [
        "session_001",
        "session_002",
    ]
    assert get_toolkit_calls[0]["config"].prompt_vars["memory_profile"] == "local"
    assert toolkits[0].memory.merge_calls == []
    assert toolkits[1].memory.merge_calls[0]["solved"] is False
    assert [toolkit.recipe_tags for toolkit in toolkits] == [[], []]


def test_cli_merges_yam_explore_memory_after_solved(monkeypatch, tmp_path) -> None:
    import rpent.cli.main as cli_main

    spec = _fake_robot_spec("yam", tmp_path)
    toolkits: list[FakeToolkit] = []

    monkeypatch.setattr(cli_main, "enumerate_robots", lambda: ("libero", "yam"))
    monkeypatch.setattr(cli_main, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli_main, "init_output_dir", _init_output_dir)
    monkeypatch.setattr(
        cli_main.MemoryManager,
        "sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote sync")),
    )
    monkeypatch.setattr(
        cli_main, "build_planner", lambda *args, **kwargs: FakePlanner()
    )

    def get_toolkit(name, **kwargs):
        del name, kwargs
        toolkit = FakeToolkit(solved=True)
        toolkits.append(toolkit)
        return toolkit

    monkeypatch.setattr(cli_main, "get_toolkit", get_toolkit)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rpent",
            "--robot",
            "yam",
            "--task-name",
            "place_cube",
            "--seed",
            "7",
            "--env-endpoint",
            "http://127.0.0.1:8110",
            "--without-vla",
            "--explore",
            "--explore-sessions",
            "2",
            "--output-dir",
            str(tmp_path / "yam-out"),
            "--max-turns",
            "1",
        ],
    )

    assert cli_main.main() == 0

    assert len(toolkits) == 1
    assert toolkits[0].recipe_tags == ["yam_place_cube_s7"]
    assert toolkits[0].memory.merge_calls == [
        {
            "cell_tag": "yam_place_cube_s7",
            "run_state_dir": tmp_path / "yam-out",
            "solved": True,
        }
    ]


def test_cli_yam_finalization_uses_environment_over_planner_success(
    monkeypatch,
    tmp_path,
) -> None:
    import rpent.cli.main as cli_main

    spec = _fake_robot_spec("yam", tmp_path)
    planner_build_calls: list[dict] = []
    planner_calls: list[dict] = []
    finish_record = {
        "_finish": True,
        "status": "failure",
        "requested_status": "success",
        "summary": "agent claimed success but final env check failed",
        "reason": "eval_success remained false",
        "verified_success": False,
    }
    toolkits: list[FakeToolkit] = []

    class SuccessClaimingPlanner(FakePlanner):
        def solve(self, **kwargs):
            planner_calls.append(kwargs)
            return SimpleNamespace(
                finish_result={
                    "_finish": True,
                    "status": "success",
                    "summary": "planner claimed success",
                },
                messages=[{"role": "assistant", "content": "done"}],
                stats={"tool_calls": 1},
                error=None,
            )

    monkeypatch.setattr(cli_main, "enumerate_robots", lambda: ("libero", "yam"))
    monkeypatch.setattr(cli_main, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli_main, "init_output_dir", _init_output_dir)
    monkeypatch.setattr(
        cli_main.MemoryManager,
        "sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("remote sync")),
    )

    def build_planner(*args, **kwargs):
        del args
        planner_build_calls.append(kwargs)
        return SuccessClaimingPlanner()

    monkeypatch.setattr(cli_main, "build_planner", build_planner)

    def get_toolkit(name, **kwargs):
        del name, kwargs
        toolkit = FakeToolkit(solved=False, finish_record=finish_record)
        toolkits.append(toolkit)
        return toolkit

    monkeypatch.setattr(cli_main, "get_toolkit", get_toolkit)
    output_dir = tmp_path / "yam-out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "rpent",
            "--robot",
            "yam",
            "--task-name",
            "place_cube",
            "--seed",
            "7",
            "--env-endpoint",
            "http://127.0.0.1:8110",
            "--without-vla",
            "--output-dir",
            str(output_dir),
            "--max-turns",
            "1",
        ],
    )

    assert cli_main.main() == 0

    assert planner_build_calls[0]["robot_name"] == "yam"
    assert planner_calls[0]["max_turns"] == 1
    assert planner_calls[0]["toolkit"] is toolkits[0]
    transcript_path = output_dir / "transcript_yam_place_cube_s7.json"
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["finish"]["status"] == "success"
    assert transcript["environment_success"] is False
    final = json.loads((output_dir / "result.json").read_text())
    assert final["status"] == "not_successful"
    assert final["environment_success"] is False
    assert final["planner_finish_request"]["status"] == "success"
    assert toolkits[0].recipe_tags == []
    assert toolkits[0].memory.merge_calls == []


def test_cli_keeps_libero_eval_on_hf_resource_path(monkeypatch, tmp_path) -> None:
    import rpent.cli.main as cli_main

    spec = _fake_robot_spec("libero", tmp_path)
    ensured: list[str] = []
    get_toolkit_calls: list[dict] = []

    monkeypatch.setattr(cli_main, "enumerate_robots", lambda: ("libero", "yam"))
    monkeypatch.setattr(cli_main, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(
        cli_main,
        "init_output_dir",
        _init_output_dir,
    )
    monkeypatch.setattr(
        cli_main.MemoryManager,
        "sync",
        lambda *args, **kwargs: ensured.append(kwargs["remote_repo"]),
    )
    monkeypatch.setattr(
        cli_main, "build_planner", lambda *args, **kwargs: FakePlanner()
    )

    def get_toolkit(name, **kwargs):
        toolkit = FakeToolkit()
        get_toolkit_calls.append({"name": name, "toolkit": toolkit, **kwargs})
        return toolkit

    monkeypatch.setattr(cli_main, "get_toolkit", get_toolkit)
    monkeypatch.setattr(
        "sys.argv",
        [
            "rpent",
            "--robot",
            "libero",
            "--suite",
            "libero_object",
            "--task",
            "0",
            "--seed",
            "1",
            "--output-dir",
            str(tmp_path / "libero-out"),
            "--max-turns",
            "1",
        ],
    )

    assert cli_main.main() == 0

    assert ensured == [spec.memory_repo_id]
    assert get_toolkit_calls[0]["name"] == "libero"
    assert get_toolkit_calls[0]["mode"] == "evaluation"
    assert get_toolkit_calls[0]["config"].prompt_vars["memory_profile"] == "hf"


@pytest.mark.parametrize(
    "vla_endpoint, without_vla", [(None, False), ("http://vla", True)]
)
def test_yam_spec_primitives_only_runtime_does_not_connect_vla(
    monkeypatch,
    tmp_path,
    vla_endpoint: str | None,
    without_vla: bool,
) -> None:
    import robots.yam.robot_spec as yam_robot_spec

    spec = yam_robot_spec.get_robot_spec()
    args = SimpleNamespace(
        task_name="place_cube",
        task_language=None,
        seed=7,
        max_episode_steps=1000,
        explore=False,
        explore_sessions=1,
        memory_profile=None,
        memory_dir=str(tmp_path / "memory"),
        output_dir=tmp_path / "run",
        env_endpoint="http://env",
        vla_endpoint=vla_endpoint,
        without_vla=without_vla,
    )

    endpoints: list[str] = []
    waited: list[str] = []
    monkeypatch.setattr(
        yam_robot_spec,
        "make_rpc_client",
        lambda endpoint: endpoints.append(endpoint) or f"rpc:{endpoint}",
    )

    def fake_wait_server(
        owned_daemons,
        dashboard_events,
        component,
        rpc,
        *args,
        **kwargs,
    ):
        del owned_daemons, dashboard_events, args, kwargs
        waited.append(component)
        return {component: rpc}

    monkeypatch.setattr(yam_robot_spec, "try_wait_server", fake_wait_server)

    config = spec.parse_config(args)
    daemons, runtime_kwargs = spec.init_runtime(
        args,
        tmp_path / "run",
        dashboard_events=SimpleNamespace(),
        components=None,
    )

    assert config.prompt_vars["vla_enabled"] is False
    assert config.prompt_vars["memory_profile"] == "local"
    assert Path(config.prompt_vars["memory_inbox"]) == (
        tmp_path / "memory" / "_internal" / "inbox" / "yam_place_cube_s7"
    )
    assert endpoints == ["http://env"]
    assert waited == ["env"]
    assert daemons == []
    assert runtime_kwargs == {"env": "rpc:http://env"}


@pytest.mark.parametrize(
    ("vla_enabled", "expected", "unexpected"),
    [
        (True, "pi05_act is available", "pi05_act is not available"),
        (False, "pi05_act is not available", "pi05_act is available"),
    ],
)
def test_yam_prompt_policy_text_follows_vla_enabled(
    vla_enabled: bool,
    expected: str,
    unexpected: str,
) -> None:
    from robots.yam.prompt_bundle import system_prompt

    rendered = json.dumps(
        system_prompt({"mode": "eval", "vla_enabled": vla_enabled}),
        ensure_ascii=False,
    )

    assert expected in rendered
    assert unexpected not in rendered


@pytest.mark.parametrize(
    ("environment_success", "expected"),
    [(True, "success"), (False, "not_successful"), (None, "unknown")],
)
def test_finalization_retains_actual_outcome_and_agent_error(
    tmp_path, environment_success, expected
) -> None:
    from rpent.evaluation import RunFinalizationContext

    path = finalize_run(
        RunFinalizationContext(
            output_dir=tmp_path,
            robot_name="yam",
            task_desc={"instruction": "Pepsi goes in the left bag"},
            environment_success=environment_success,
            agent_error="planner disconnected",
            elapsed_s=1.2,
            planner="api",
            model="test-planner",
            reasoning_effort="high",
            max_turns=10,
            planner_timeout_s=20,
            finish_result={"status": "success"},
            stats={},
        )
    )
    record = json.loads(path.read_text())
    assert record["status"] == expected
    assert record["environment_success"] is environment_success
    assert record["agent_error"] == "planner disconnected"
    assert record["planner_finish_request"]["status"] == "success"


def test_yam_continuation_handoff_points_to_prior_session_and_memory_inbox(
    monkeypatch,
    tmp_path,
) -> None:
    import rpent.cli.main as cli_main

    output_dir = tmp_path / "yam-out"
    prior_session = output_dir / "sessions" / "session_001"
    prior_session.mkdir(parents=True)
    memory_inbox = tmp_path / "memory" / "_internal" / "inbox" / "yam_place_cube_s7"
    args = SimpleNamespace(
        planner="codex",
        robot_name="yam",
        base_url=None,
        model=None,
        max_tokens=None,
        planner_timeout_s=None,
        reasoning_effort=None,
        claude_code_max_budget_usd=None,
        no_images=False,
    )
    monkeypatch.setattr(
        cli_main, "build_planner", lambda *args, **kwargs: FakePlanner()
    )

    _, system_prompt, message = cli_main._start_continuation_session(
        args,
        output_dir=output_dir,
        recipe_tag="yam_place_cube_s7",
        dashboard_events=SimpleNamespace(),
        prompt_bundle=FakePromptBundle(),
        prompt_vars={
            "mode": "explore",
            "memory_profile": "local",
            "memory_inbox": str(memory_inbox),
        },
        session_number=2,
        session_max=3,
    )

    assert system_prompt == "system:explore:local"
    assert str(prior_session) in message
    assert f"{memory_inbox}/wip/" in message
    assert "does not prove that the scene was reset" in message
    assert "clean scene" not in message


@pytest.mark.parametrize(
    ("event", "expected_calls"),
    [
        ("ready", []),
        ("start", ["env.reset"]),
    ],
)
def test_operator_control_ready_writes_receipt_and_start_resets(
    monkeypatch,
    tmp_path,
    capsys,
    event: str,
    expected_calls: list[str],
) -> None:
    import rpent.utils.rpc as rpc_module
    from robots.yam import operator_control

    receipt_path = tmp_path / "operator-receipt.json"
    config_path = tmp_path / "yam-config.json"
    config_path.write_text(
        json.dumps({"operator_receipt_path": str(receipt_path)}),
        encoding="utf-8",
    )

    class FakeOperatorRpc:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def call(self, method: str, *args, **kwargs):
            del args, kwargs
            self.calls.append(method)
            if method == "env.reset":
                return None, {"episode_status": {"episode_id": "started"}}
            raise AssertionError(method)

    fake_rpc = FakeOperatorRpc()
    monkeypatch.setattr(rpc_module, "make_rpc_client", lambda endpoint: fake_rpc)
    monkeypatch.setattr(
        "sys.argv",
        [
            "operator_control",
            "--config",
            str(config_path),
            "--endpoint",
            "http://env",
            "--episode-id",
            "pending-episode",
            "--event",
            event,
            "--note",
            "operator approved",
        ],
    )

    operator_control.main()

    output = json.loads(capsys.readouterr().out)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["episode_id"] == "pending-episode"
    assert receipt["event"] == "ready"
    assert fake_rpc.calls == expected_calls
    if event == "start":
        assert output["episode_id"] == "started"
    else:
        assert output["episode_id"] == "pending-episode"
