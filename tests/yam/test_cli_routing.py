# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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

    def latest_record(self):
        return self._records[-1] if self._records else None


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
            parser.add_argument("--yam-reset-on-connect", action="store_true")
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
            },
            task_desc={"env": name, "task_name": task_name, "seed": args.seed},
        )

    def init_runtime(args, output_dir, dashboard_events, components):
        del args, output_dir, dashboard_events, components
        return [], {"env": "fake-env"}

    return RobotSpec(
        name=name,
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
        cli_main,
        "ensure_resources",
        lambda robot_spec: (_ for _ in ()).throw(AssertionError(robot_spec.name)),
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
    assert [toolkit.memory.merge_calls for toolkit in toolkits] == [[], []]
    assert [toolkit.recipe_tags for toolkit in toolkits] == [[], []]


def test_cli_merges_yam_explore_memory_after_solved(monkeypatch, tmp_path) -> None:
    import rpent.cli.main as cli_main

    spec = _fake_robot_spec("yam", tmp_path)
    toolkits: list[FakeToolkit] = []

    monkeypatch.setattr(cli_main, "enumerate_robots", lambda: ("libero", "yam"))
    monkeypatch.setattr(cli_main, "get_robot_spec", lambda name: spec)
    monkeypatch.setattr(cli_main, "init_output_dir", _init_output_dir)
    monkeypatch.setattr(
        cli_main,
        "ensure_resources",
        lambda robot_spec: (_ for _ in ()).throw(AssertionError(robot_spec.name)),
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


def test_cli_yam_transcript_uses_toolkit_finish_failure_over_planner_success(
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
        cli_main,
        "ensure_resources",
        lambda robot_spec: (_ for _ in ()).throw(AssertionError(robot_spec.name)),
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
    assert transcript["finish"]["status"] == "failure"
    assert transcript["finish"]["requested_status"] == "success"
    assert transcript["finish"]["verified_success"] is False
    assert transcript["finish"]["reason"] == "eval_success remained false"
    assert (
        transcript["finish"]["summary"]
        == "agent claimed success but final env check failed"
    )
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
        cli_main,
        "ensure_resources",
        lambda robot_spec: ensured.append(robot_spec.name),
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

    assert ensured == ["libero"]
    assert get_toolkit_calls[0]["name"] == "libero"
    assert get_toolkit_calls[0]["mode"] == "evaluation"
    assert get_toolkit_calls[0]["config"].prompt_vars["memory_profile"] == "hf"
