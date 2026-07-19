import asyncio

from rpent.cerebrum.api_loop import _make_tool_function
from rpent.cerebrum.claude_code import _build_rpent_server
from rpent.cerebrum.codex import _Recorder as CodexRecorder
from rpent.dashboard.state import State
from rpent.tools.toolkit import Toolkit


class _FinishToolkit(Toolkit):
    def __init__(self):
        super().__init__()
        self.add_tool(
            "complete_episode",
            {
                "name": "complete_episode",
                "description": "Complete an episode.",
                "input_schema": {"type": "object", "properties": {}},
            },
            lambda: {"_finish": True, "task_success": True},
        )


def test_api_bridge_captures_finish_from_non_finish_tool():
    seen = []
    call = _make_tool_function(
        _FinishToolkit(), "complete_episode", on_tool_result=seen.append
    )

    call()

    assert seen[0].is_finish is True
    assert seen[0].result["task_success"] is True


def test_codex_recorder_captures_finish_from_tool_result():
    recorder = CodexRecorder(max_turns=1)
    result = _FinishToolkit().execute_tool("complete_episode", {})

    recorder.capture_tool_result(result)

    assert recorder.finish_result == {
        "_finish": True,
        "task_success": True,
    }


def test_claude_bridge_captures_finish_from_non_finish_tool():
    registered = {}

    class FakeSdk:
        @staticmethod
        def tool(name, description, input_schema):
            def decorate(fn):
                registered[name] = fn
                return fn

            return decorate

        @staticmethod
        def create_sdk_mcp_server(**kwargs):
            return kwargs

    seen = []
    _build_rpent_server(
        FakeSdk, toolkit=_FinishToolkit(), on_tool_result=seen.append
    )

    asyncio.run(registered["complete_episode"]({}))

    assert seen[0].is_finish is True


def test_dashboard_uses_task_success_without_libero_field(tmp_path):
    state = State(
        run_id="run",
        name="behavior",
        suite="behavior",
        task=0,
        seed=211,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )

    state.on_tool_result(
        "run_full_task", {"task_success": True, "libero_terminated": False}
    )
    state.mark_done()

    assert state.snapshot()["terminated"] is True


def test_dashboard_does_not_treat_libero_field_as_generic_success(tmp_path):
    state = State(
        run_id="run",
        name="behavior",
        suite="behavior",
        task=0,
        seed=211,
        output_dir=str(tmp_path),
        video_path=str(tmp_path / "episode.mp4"),
    )

    state.on_tool_result("legacy", {"libero_terminated": True})
    state.mark_done()

    assert state.snapshot()["terminated"] is False
