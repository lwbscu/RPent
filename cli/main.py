"""Physical agent main CLI entrypoint."""
from __future__ import annotations

import argparse
import json
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
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {**{k: v for k, v in m.items() if k != "content"},
         "content": _strip_images(m.get("content"))}
        for m in messages
    ]


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
    ap.add_argument("--cerebrum", default="api",
                    choices=["api", "claude_code", "codex"],
                    help="LLM backend: api | claude_code | codex.")
    ap.add_argument("--model", default=None,
                    help="Model id. For the 'api' cerebrum you need to prefix provider to the model id "
                         "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
                         "openai-chat:glm-5.2).")
    ap.add_argument("--base-url", default=None,
                    help="API base URL. Defaults to the selected backend's base URL env var.")
    ap.add_argument("--api-key", default=None,
                    help="API key. Defaults to the selected backend's API key env var.")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--cerebrum-timeout-s", type=int, default=None,
                    help="Wall-clock cap for the claude_code/codex cerebrum "
                         "subprocess. Defaults to CODEX_TIMEOUT_S (codex only), "
                         "CELL_TIMEOUT_S, or 1200.")
    ap.add_argument("--claude-code-max-budget-usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd. "
                         "Defaults to MAX_BUDGET_USD env or 10.")

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dashboard", action="store_true",
                    help="Start a local dashboard server for this single run.")
    ap.add_argument("--dashboard-host", default="127.0.0.1",
                    help="Dashboard bind host. Defaults to 127.0.0.1.")
    ap.add_argument("--dashboard-port", type=int, default=0,
                    help="Dashboard port. 0 asks the OS for a free port.")
    ap.add_argument("--dashboard-language", choices=["en", "zh-cn"], default="en",
                    help="Dashboard UI language. 'zh-cn' serves the Chinese "
                         "variant (index.zh-cn.html); defaults to English.")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG-level logging for stdout and the run.log "
                         "file. Defaults to INFO when not set.")

    # environments
    ap.add_argument("--env", dest="env_name", default="libero",
                    help="Environment backend. Defaults to libero.")
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
    if args.dashboard:
        from rpent.dashboard import DashboardServer
        from rpent.dashboard.launcher import apply_to_args, defaults_from_args

        dashboard_server = DashboardServer(
            host=args.dashboard_host, port=args.dashboard_port,
            language=args.dashboard_language,
        )
        dashboard_url = dashboard_server.start()
        print(
            f"Dashboard: {dashboard_url}. Open it, adjust the run config, and click Run to start.",
            flush=True,
        )
        launch_config = dashboard_server.wait_for_launch(
            defaults=defaults_from_args(args)
        )
        apply_to_args(args, launch_config)
        logger.info("launcher config applied: %s", launch_config)

    if args.env_name.lower() != runtime_provider.name:
        runtime_provider = get_runtime_provider(args.env_name)
    runtime_provider.validate_args(args, parser)

    env_name = args.env_name
    env_spec = get_env_spec(env_name)
    prompt_bundle = env_spec.prompts

    # resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{runtime_provider.recipe_tag(args)}"
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

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

    runtime = runtime_provider.start(
        args,
        output_dir=output_dir,
        dashboard=dashboard_state,
    )
    toolkit = runtime.toolkit

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    try:
        result = cerebrum.solve(
            system_prompt=system_prompt,
            user_message=user_msg,
            toolkit=toolkit,
            max_turns=args.max_turns,
        )
        finish_result = result.finish_result
        messages = result.messages
        stats = result.stats
        agent_error = result.error
    except Exception as e:
        logger.error("EXCEPTION in agent loop: %s", e)
    finally:
        try:
            # Agent-side: flush the episode video before the env+model
            recipe_path = toolkit.write_recipe(recipe_tag)
            logger.info("recipe: %s", recipe_path)
        finally:
            runtime.close()

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    transcript_fields = {
        k: v for k, v in prompt_vars.items()
        if k not in {"output_dir", "recipe_tag"}
    }
    record = {
        **transcript_fields,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info("usage: in=%s out=%s tool_calls=%s",
                 stats.get('total_input_tokens', '?'),
                 stats.get('total_output_tokens', '?'),
                 stats.get('tool_calls', '?'))
    logger.info("transcript: %s", transcript_path)
    if agent_error:
        logger.error("error: %s", agent_error)

    if args.dashboard and dashboard_state is not None:
        dashboard_state.mark_done()
        logger.info(
            "Run finished. Dashboard still serving at %s. Press Ctrl+C to stop.",
            dashboard_url,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
