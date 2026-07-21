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
# rpent --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.cerebrum`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from rpent.cerebrum.base import build_cerebrum  # noqa: E402
from rpent.envs import RuntimeProvider, get_env_spec, get_runtime_provider  # noqa: E402
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir  # noqa: E402
from rpent.utils.redaction import redact_command, redact_text, redact_value

logger = get_logger("agent")


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
    if isinstance(value, str):
        return redact_text(value)
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


def _redact_argv(argv: list[str]) -> list[str]:
    """Redact credential-bearing CLI values without mutating the real argv."""

    return redact_command(argv) or []


def _redact_mapping(value):
    """Return a log-safe copy of a nested dashboard launch payload."""

    return redact_value(value)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically replace one standalone JSON artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            redact_value(payload),
            stream,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _prepare_output_dir(path: str | Path, parser: argparse.ArgumentParser) -> Path:
    """Require a fresh run directory before any environment process starts."""

    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        parser.error(f"--output-dir must be absent or empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _preparse_env(argv: list[str] | None = None) -> str:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env", dest="env_name", default="libero")
    args, _ = pre.parse_known_args(argv)
    return args.env_name


def _build_argparser(runtime_provider: RuntimeProvider) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone RPent physical-agent runner",
    )

    # models
    ap.add_argument(
        "--cerebrum",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' cerebrum, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2). For claude_code/codex this "
        "overrides the backend default model.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to the selected backend's API key env var.",
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--cerebrum-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for the claude_code/codex cerebrum "
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

    # environments
    ap.add_argument(
        "--env",
        dest="env_name",
        default="libero",
        help="Environment backend. Defaults to libero.",
    )
    runtime_provider.add_cli_args(ap)

    return ap


def main() -> int:
    runtime_provider = get_runtime_provider(_preparse_env(sys.argv[1:]))
    parser = _build_argparser(runtime_provider)
    args = parser.parse_args()

    # With --dashboard, open the launcher first: serve the start screen, then
    # block until the user clicks Run and overlay their choices onto args.
    # Everything downstream (output_dir, State, run loop) then sees final args.
    dashboard_server = None
    dashboard_url = None
    launch_config = None
    if args.dashboard:
        from rpent.dashboard import DashboardServer
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

    if args.env_name.lower() != runtime_provider.name:
        runtime_provider = get_runtime_provider(args.env_name)
    runtime_provider.validate_args(args, parser)

    env_name = args.env_name
    env_spec = get_env_spec(env_name)
    prompt_bundle = env_spec.prompts

    # resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_{runtime_provider.recipe_tag(args)}"
        )
    output_dir = _prepare_output_dir(output_dir, parser)
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    # Now that output_dir is fixed, repeat launcher details into this run's log.
    if dashboard_url is not None:
        logger.info("Dashboard: %s", dashboard_url)
    if launch_config is not None:
        logger.info("launcher config applied: %s", _redact_mapping(launch_config))
    logger.info(
        "physical agent cmd: %s",
        shlex.join(_redact_argv([sys.executable, *sys.argv])),
    )

    recipe_tag = runtime_provider.recipe_tag(args)

    dashboard_state = None
    if args.dashboard and dashboard_server is not None:
        dashboard_state = runtime_provider.dashboard_state(args, output_dir=output_dir)
        # Server is already serving the launcher; register the run so the
        # frontend can switch from the start screen to the live monitor.
        dashboard_server.register(dashboard_state)

    cerebrum = build_cerebrum(
        args.cerebrum,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        cerebrum_timeout_s=args.cerebrum_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard=dashboard_state,
    )

    prompt_vars = runtime_provider.prompt_vars(
        args,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
    )
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    runtime = None
    toolkit = None
    recipe_path = None
    try:
        runtime = runtime_provider.start(
            args,
            output_dir=output_dir,
            dashboard=dashboard_state,
        )
        toolkit = runtime.toolkit
        result = cerebrum.solve(
            system_prompt=system_prompt,
            user_message=user_msg,
            toolkit=toolkit,
            max_turns=args.max_turns,
        )
        finish_result = result.finish_result
        messages = result.messages
        stats = result.stats
        agent_error = redact_text(result.error) if result.error else None
    except Exception as e:
        agent_error = redact_text(f"{type(e).__name__}: {e}")
        logger.error("EXCEPTION in agent loop: %s", agent_error)
    finally:
        if toolkit is not None:
            try:
                recipe_path = toolkit.write_recipe(recipe_tag)
                logger.info("recipe: %s", recipe_path)
            except Exception as error:
                if agent_error is None:
                    agent_error = redact_text(f"{type(error).__name__}: {error}")
                logger.error("failed to write recipe: %s", agent_error)

        elapsed = time.time() - t0
        transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
        transcript_fields = {
            key: value
            for key, value in prompt_vars.items()
            if key not in {"output_dir", "recipe_tag"}
        }
        record = {
            **transcript_fields,
            "model": args.model,
            "elapsed_s": round(elapsed, 1),
            "finish": finish_result,
            "error": agent_error,
            "stats": stats,
            "messages": _serialize_messages(messages),
        }
        _atomic_write_json(transcript_path, record)

        task_success = toolkit.official_task_success if toolkit is not None else None
        checkpoint2 = Path(output_dir) / "state_checkpoints" / "state_checkpoint_2.json"
        paused_for_review = bool(
            env_name.lower() == "behavior"
            and checkpoint2.is_file()
            and task_success is not True
        )
        run_status = (
            "error"
            if agent_error is not None
            else "paused_for_review"
            if paused_for_review
            else "completed"
            if task_success is not None
            else "incomplete"
        )
        final_result_path = Path(output_dir) / "final_result.json"
        final_result = {
            "schema_version": 1,
            "run_status": run_status,
            "task_success": task_success,
            "official_success_source": (
                'info["done"]["success"]'
                if env_name.lower() == "behavior" and task_success is not None
                else "libero_terminated"
                if env_name.lower() == "libero" and task_success is not None
                else None
            ),
            "phase": (
                {"name": "pre_press_alignment", "success": True}
                if paused_for_review
                else None
            ),
            "cerebrum": {"backend": args.cerebrum, "model": args.model},
            "finish": finish_result,
            "error": agent_error,
            "elapsed_s": round(elapsed, 1),
            "runtime_cleanup": "pending" if runtime is not None else "not_started",
            "artifacts": {
                "transcript": str(transcript_path),
                "recipe": recipe_path,
                "checkpoint_2": str(checkpoint2) if checkpoint2.is_file() else None,
            },
        }
        _atomic_write_json(final_result_path, final_result)

        if runtime is not None:
            try:
                runtime.close()
                final_result["runtime_cleanup"] = "complete"
            except Exception as error:
                final_result["runtime_cleanup"] = "error"
                final_result["run_status"] = "error"
                final_result["error"] = redact_text(f"{type(error).__name__}: {error}")
                agent_error = final_result["error"]
                logger.error("runtime cleanup failed: %s", agent_error)
            finally:
                _atomic_write_json(final_result_path, final_result)

    elapsed = time.time() - t0

    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)
    logger.info("final result: %s", Path(output_dir) / "final_result.json")
    if agent_error:
        logger.error("error: %s", agent_error)

    if args.dashboard and dashboard_state is not None:
        dashboard_state.mark_done(final_result["task_success"])
        logger.info(
            "Run finished. Dashboard still serving at %s. Press Ctrl+C to stop.",
            dashboard_url,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 1 if agent_error else 0


if __name__ == "__main__":
    sys.exit(main())
