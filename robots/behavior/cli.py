"""Physical agent main CLI entrypoint."""

# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --env libero --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.planner`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import queue
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from robots.behavior import get_env_spec, get_toolkit
from robots.behavior.planner import build_behavior_planner
from robots.behavior.spec import RunOutcome, RuntimeResource
from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("agent")


class _TerminationRequested(RuntimeError):
    """Raised once so SIGTERM unwinds through BEHAVIOR artifact sealing."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"termination signal {signum} requested")
        self.signum = int(signum)


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {
            **{k: v for k, v in m.items() if k != "content"},
            "content": _strip_images(m.get("content")),
        }
        for m in messages
    ]


def _error_text(error: BaseException) -> str:
    """Return one stable public error string for run artifacts and logs."""
    return f"{type(error).__name__}: {error}"


def _default_finalize_run(outcome: RunOutcome) -> None:
    """Preserve the standard LIBERO transcript behavior after cleanup."""
    record = {
        **outcome.task_desc,
        "model": outcome.args.model,
        "elapsed_s": round(outcome.elapsed_s, 1),
        "finish": outcome.finish_result,
        "stats": outcome.stats,
        "messages": _serialize_messages(outcome.messages),
    }
    with outcome.transcript_path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, default=str)
        stream.write("\n")


def _cleanup_runtime(
    *,
    toolkit: Any,
    runtime_resources: list[RuntimeResource],
) -> tuple[dict[str, Any], str | None]:
    """Close the toolkit and every owned resource, retaining all failures."""
    cleanup: dict[str, Any] = {
        "toolkit": {"status": "not_started", "error": None},
        "runtime_resources": [],
        "complete": True,
    }
    first_error: str | None = None

    if toolkit is not None:
        try:
            toolkit.close()
            cleanup["toolkit"]["status"] = "complete"
        except BaseException as error:
            text = _error_text(error)
            cleanup["toolkit"] = {"status": "failed", "error": text}
            cleanup["complete"] = False
            first_error = text
            logger.error("toolkit cleanup failed: %s", text)

    for index, resource in enumerate(runtime_resources):
        resource_result = {
            "index": index,
            "type": type(resource).__name__,
            "status": "not_started",
            "error": None,
        }
        try:
            resource.stop()
            resource_result["status"] = "complete"
        except BaseException as error:
            text = _error_text(error)
            resource_result.update(status="failed", error=text)
            cleanup["complete"] = False
            if first_error is None:
                first_error = text
            logger.error("runtime resource cleanup failed: %s", text)
        cleanup["runtime_resources"].append(resource_result)
    return cleanup, first_error


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop physical agent",
    )

    ap.add_argument(
        "--env",
        dest="env_name",
        required=True,
        choices=["behavior"],
        help="Environment backend: behavior.",
    )

    # models
    ap.add_argument(
        "--planner",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' planner, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2). For claude_code/codex this "
        "overrides the backend default model.",
    )
    ap.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for planners that expose it (Codex SDK).",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Never send image bytes to the model (api planner only). "
        "Use for text-only models that reject image input "
        "(e.g. 400 \"message type 'image_url' is not supported\"); "
        "read_image then returns the file path with a notice.",
    )
    ap.add_argument(
        "--planner-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for the claude_code/codex planner "
        "subprocess. Defaults to CODEX_TIMEOUT_S (codex only), "
        "CELL_TIMEOUT_S, or 1200.",
    )
    ap.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=None,
        help="Budget passed to claude -p --max-budget-usd. "
        "Defaults to MAX_BUDGET_USD env or 10.",
    )

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start a local dashboard server for this single run.",
    )
    ap.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host. Defaults to 127.0.0.1.",
    )
    ap.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Dashboard port. 0 asks the OS for a free port.",
    )
    ap.add_argument(
        "--dashboard-language",
        choices=["en", "zh-cn"],
        default="en",
        help="Dashboard UI language. 'zh-cn' serves the Chinese "
        "variant (index.zh-cn.html); defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )
    ap.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: opens an interactive cli session.",
    )

    return ap


def main() -> int:
    parser = _build_argparser()
    # Two-phase argparse: first grab --env / --dashboard so we know which
    # env's flags to add and whether to make its required flags optional.
    early, _ = parser.parse_known_args()

    env_spec = get_env_spec()
    env_spec.add_cli_args(parser, use_dashboard=early.dashboard)
    args = parser.parse_args()

    # With --dashboard, open the launcher first: serve the start screen, then
    # block until the user clicks Run and overlay their choices onto args.
    # parse_config runs afterwards so validation + derivation see the final
    # config.
    dashboard_server = None
    dashboard_url = None
    launch_config = None
    if args.dashboard:
        from robots.behavior.dashboard_server import DashboardServer
        from rpent.dashboard.launcher import apply_to_args, defaults_from_args

        dashboard_server = DashboardServer(
            host=args.dashboard_host,
            port=args.dashboard_port,
            language=args.dashboard_language,
        )
        dashboard_url = dashboard_server.start()
        # The run directory is not final until the launcher form is submitted, so
        # print the pre-launch URL without initializing the run.log file handler.
        print(
            f"Dashboard: {dashboard_url}. "
            "Open it, adjust the run config, and click Run to start.",
            flush=True,
        )
        launch_config = dashboard_server.wait_for_launch(
            defaults=defaults_from_args(args)
        )
        apply_to_args(args, launch_config)

    # Strict environments prepare and pin dataset resources after the
    # dashboard has finalized CLI values, but before parse_config derives any
    # prompt, manifest, or runtime path from them. Default environments retain
    # their existing resource_policy lifecycle below.
    args.prepared_resources = None
    if env_spec.prepare_resources is not None:
        try:
            prepared_resources = env_spec.prepare_resources(args)
        except Exception as error:
            parser.error(f"resource preparation failed: {error}")
        if prepared_resources is not None:
            args.prepared_resources = prepared_resources

    try:
        run_config = env_spec.parse_config(args)
    except ValueError as error:
        parser.error(str(error))
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    dashboard_state = run_config.dashboard_state
    task_desc = run_config.task_desc

    # mkdir + logging wiring (env-side already picked the path).
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    # Now that output_dir is fixed, repeat launcher details into this run's log.
    if dashboard_url is not None:
        logger.info("Dashboard: %s", dashboard_url)
    if launch_config is not None:
        logger.info("launcher config applied: %s", launch_config)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    prompt_vars = {**prompt_vars, "output_dir": output_dir}
    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    toolkit = None
    runtime_resources: list[RuntimeResource] = []
    recipe_path: str | None = None
    interrupted = False
    termination_signal: int | None = None
    sealing_started = False
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _request_termination(signum: int, _frame: Any) -> None:
        nonlocal termination_signal
        if termination_signal is None:
            termination_signal = int(signum)
            if sealing_started:
                logger.warning(
                    "recorded signal %s while BEHAVIOR cleanup is sealing",
                    signum,
                )
                return
            raise _TerminationRequested(signum)
        logger.warning(
            "ignoring repeated signal %s while BEHAVIOR cleanup is sealing",
            signum,
        )

    signal.signal(signal.SIGTERM, _request_termination)
    try:
        resource_policy = run_config.resource_policy or env_spec.resource_policy
        if resource_policy != "frozen_local":
            raise ValueError(f"unknown resource policy: {resource_policy!r}")

        # --- dashboard state -----------------------------------------------
        if dashboard_state is not None and dashboard_server is not None:
            dashboard_server.register(dashboard_state)

        planner = build_behavior_planner(
            args.planner,
            output_dir=output_dir,
            recipe_tag=recipe_tag,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_tokens=args.max_tokens,
            planner_timeout_s=args.planner_timeout_s,
            claude_code_max_budget_usd=args.claude_code_max_budget_usd,
            dashboard=dashboard_state,
            no_images=args.no_images,
        )
        prompt_bundle = env_spec.prompts
        system_prompt = prompt_bundle.render("system", variables=prompt_vars)
        user_msg = prompt_bundle.render("user", variables=prompt_vars)

        if args.interactive:
            input_queue = queue.Queue()
            start_interactive_reader(
                input_queue,
                first_prompt_default=user_msg,
            )
            logger.info(
                "interactive mode on: the built-in task is pre-filled — "
                "edit it and press Enter, submit it as-is, or clear it to "
                "type your own. Once running, type to steer the agent. "
                "/help for commands."
            )
            await_first_prompt = start_first_prompt_resolver(input_queue)

        # --- initialise environment ----------------------------------------
        resources, primitives_kwargs = env_spec.init_runtime(args, output_dir)
        runtime_resources.extend(resources)

        # --- toolkit -------------------------------------------------------
        toolkit = get_toolkit(
            primitives_kwargs=primitives_kwargs,
            video_path=str(Path(output_dir) / "episode.mp4"),
            dashboard=dashboard_state,
        )

        first_user_msg: str | None = user_msg
        if await_first_prompt is not None:
            first_user_msg = await_first_prompt()
            if first_user_msg is None:
                logger.info("no task entered; ending session before start.")

        if first_user_msg is not None:
            planner_kwargs = {
                "planner": planner,
                "system_prompt": system_prompt,
                "user_message": first_user_msg,
                "toolkit": toolkit,
                "max_turns": args.max_turns,
                "input_queue": input_queue,
                "args": args,
                "run_config": run_config,
                "runtime_resources": tuple(runtime_resources),
            }
            if env_spec.run_planner is None:
                result = planner.solve(
                    system_prompt=system_prompt,
                    user_message=first_user_msg,
                    toolkit=toolkit,
                    max_turns=args.max_turns,
                    input_queue=input_queue,
                )
            else:
                result = env_spec.run_planner(**planner_kwargs)
            finish_result = result.finish_result
            messages = result.messages
            stats = result.stats
            agent_error = result.error
    except KeyboardInterrupt as error:
        interrupted = True
        agent_error = _error_text(error)
        logger.warning("run interrupted by operator")
    except _TerminationRequested as error:
        interrupted = True
        agent_error = _error_text(error)
        logger.warning("run termination deferred through artifact sealing")
    except Exception as error:
        agent_error = _error_text(error)
        logger.error("EXCEPTION in run lifecycle: %s", agent_error)
    finally:
        sealing_started = True
        # Default environments preserve the LIBERO recipe-before-close
        # lifecycle. Environments with a strict finalizer (BEHAVIOR) defer
        # publication until that hook can inspect the completed cleanup.
        if toolkit is not None and env_spec.finalize_run is None:
            try:
                recipe_path = toolkit.write_recipe(recipe_tag)
                logger.info("recipe: %s", recipe_path)
            except BaseException as error:
                if isinstance(error, KeyboardInterrupt):
                    interrupted = True
                if agent_error is None:
                    agent_error = _error_text(error)
                logger.error("failed to write recipe: %s", _error_text(error))
        cleanup, cleanup_error = _cleanup_runtime(
            toolkit=toolkit,
            runtime_resources=runtime_resources,
        )
        if agent_error is None and cleanup_error is not None:
            agent_error = cleanup_error

    elapsed = time.time() - t0
    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    outcome = RunOutcome(
        args=args,
        run_config=run_config,
        toolkit=toolkit,
        runtime_resources=tuple(runtime_resources),
        finish_result=finish_result,
        messages=messages,
        stats=stats,
        error=agent_error,
        elapsed_s=elapsed,
        transcript_path=transcript_path,
        recipe_path=recipe_path,
        cleanup=cleanup,
        task_desc=dict(task_desc),
        prompt_vars=dict(prompt_vars),
    )
    finalization_result: Any = None
    try:
        if env_spec.finalize_run is None:
            _default_finalize_run(outcome)
        else:
            finalization_result = env_spec.finalize_run(outcome)
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            interrupted = True
        if outcome.error is None:
            outcome.error = _error_text(error)
        logger.error("run finalization failed: %s", _error_text(error))

    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)
    if outcome.error:
        logger.error("error: %s", outcome.error)

    if dashboard_state is not None:
        dashboard_state.mark_done()
    signal.signal(signal.SIGTERM, previous_sigterm)
    if args.dashboard and dashboard_state is not None and termination_signal is None:
        logger.info(
            "Run finished. Dashboard still serving at %s. Press Ctrl+C to stop.",
            dashboard_url,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    if termination_signal is not None:
        return 128 + termination_signal
    if interrupted:
        return 130
    if outcome.error is not None:
        return 1
    if (
        isinstance(finalization_result, dict)
        and finalization_result.get("run_status") == "error"
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
