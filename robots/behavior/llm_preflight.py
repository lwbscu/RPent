"""Isolated Agentic LLM transport preflight for formal BEHAVIOR Eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from robots.behavior.redaction import redact_text

LLM_PREFLIGHT_SCHEMA_VERSION = 1
LLM_PREFLIGHT_TIMEOUT_S = 60
LLM_PREFLIGHT_TOKEN = "RPENT_LLM_PREFLIGHT_OK"
_RESULT_PREFIX = "RPENT_LLM_PREFLIGHT_RESULT="
_TRANSIENT_MARKERS = (
    "reconnect",
    "retry",
    "response_stream_disconnected",
    "request timed out",
    "connection reset",
)
_NETWORK_ENDPOINT_KEYS = (
    "ALL_PROXY",
    "CODEX_BASE_URL",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "all_proxy",
    "https_proxy",
    "http_proxy",
    "no_proxy",
)
_NETWORK_CREDENTIAL_KEYS = ("CODEX_API_KEY", "OPENAI_API_KEY")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _unwrap(value: Any) -> Any:
    return getattr(value, "root", value)


def _extract_text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        parts: list[str] = []
        for item in value:
            item = _unwrap(item)
            text = _get(item, "text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def network_environment_binding(
    environment: Mapping[str, str],
) -> dict[str, Any]:
    """Bind network routing without persisting endpoints or credentials."""

    endpoints = {
        key: (
            hashlib.sha256(str(environment[key]).encode("utf-8")).hexdigest()
            if key in environment
            else None
        )
        for key in _NETWORK_ENDPOINT_KEYS
    }
    credentials_present = {
        key: bool(environment.get(key)) for key in _NETWORK_CREDENTIAL_KEYS
    }
    payload = {
        "endpoint_value_sha256": endpoints,
        "credentials_present": credentials_present,
    }
    return {
        **payload,
        "binding_sha256": hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _worker_result(*, model: str, workspace: Path, challenge: str) -> dict[str, Any]:
    """Run one ephemeral no-tools Codex turn and return redacted evidence."""

    import openai_codex

    overrides = [
        "mcp_servers={}",
        "features.shell_tool=false",
        "features.unified_exec=false",
        "features.apps=false",
        "features.multi_agent=false",
        "features.plugins=false",
        "features.plugin_sharing=false",
        "features.remote_plugin=false",
        "features.memories=false",
        "features.goals=false",
        "features.hooks=false",
        'web_search="disabled"',
        "tools_view_image=false",
        'model_reasoning_effort="low"',
    ]
    environment = dict(os.environ)
    base_url = environment.get("CODEX_BASE_URL")
    api_key = environment.get("CODEX_API_KEY")
    if base_url and api_key:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"
        environment["RPENT_CODEX_PROVIDER_KEY"] = api_key
        overrides.extend(
            (
                'model_provider="rpent_proxy"',
                'model_providers.rpent_proxy.name="rpent_proxy"',
                f"model_providers.rpent_proxy.base_url={json.dumps(normalized)}",
                'model_providers.rpent_proxy.wire_api="responses"',
                'model_providers.rpent_proxy.env_key="RPENT_CODEX_PROVIDER_KEY"',
            )
        )
    elif api_key:
        environment["OPENAI_API_KEY"] = api_key
    config = openai_codex.CodexConfig(
        config_overrides=tuple(overrides),
        cwd=str(workspace),
        env=environment,
        experimental_api=False,
    )
    response = ""
    completed_status = ""
    errors: list[str] = []
    forbidden_events: list[str] = []
    with openai_codex.Codex(config=config) as codex:
        thread = codex.thread_start(
            approval_mode=openai_codex.ApprovalMode.deny_all,
            base_instructions=(
                "You are a transport preflight. Do not use tools, inspect files, "
                "or infer any task context. Reply only with the requested token."
            ),
            cwd=str(workspace),
            ephemeral=True,
            model=model,
            sandbox=openai_codex.Sandbox.read_only,
        )
        expected_response = f"{LLM_PREFLIGHT_TOKEN}:{challenge}"
        turn = thread.turn(
            f"Reply exactly {expected_response}",
            approval_mode=openai_codex.ApprovalMode.deny_all,
            cwd=str(workspace),
            model=model,
            sandbox=openai_codex.Sandbox.read_only,
        )
        for event in turn.stream():
            method = str(_get(event, "method", ""))
            payload = _get(event, "payload")
            if method == "item/completed":
                item = _unwrap(_get(payload, "item"))
                item_type = str(_get(item, "type", ""))
                if item_type == "agentMessage":
                    response = str(_get(item, "text", "")).strip()
                elif item_type in {
                    "mcpToolCall",
                    "dynamicToolCall",
                    "commandExecution",
                    "fileChange",
                }:
                    forbidden_events.append(item_type)
            elif method == "turn/completed":
                turn_result = _get(payload, "turn")
                status = _get(turn_result, "status")
                completed_status = str(_get(status, "value", status))
                if error := _get(turn_result, "error"):
                    errors.append(str(_get(error, "message", error)))
            elif method in {"error", "fatal"}:
                errors.append(str(payload))
    return {
        "response": response,
        "completed_status": completed_status,
        "errors": [redact_text(item) for item in errors],
        "forbidden_events": forbidden_events,
    }


def _worker_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_worker_group(process: subprocess.Popen[str]) -> bool:
    if not _worker_group_alive(process.pid):
        return True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 5.0
    while _worker_group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if not _worker_group_alive(process.pid):
        return True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 5.0
    while _worker_group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _worker_group_alive(process.pid)


def _wait_worker_group_exit(pgid: int, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while _worker_group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _worker_group_alive(pgid)


def _worker_payload(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if not line.startswith(_RESULT_PREFIX):
            continue
        try:
            value = json.loads(line[len(_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def run_llm_proxy_preflight(
    *,
    python: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    model: str,
    timeout_s: int = LLM_PREFLIGHT_TIMEOUT_S,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one disposable no-tools model turn before simulator admission.

    A complete expected response passes. Transport retries followed by that
    response pass with a degraded warning. Timeout, zero output, a final
    transport error, or any malformed worker result fails closed.
    """

    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s <= 0:
        raise ValueError("LLM preflight timeout must be a positive integer")
    # Preserve the lexical virtual-environment entry point. Resolving its
    # symlink would silently launch the system interpreter and lose the Codex
    # SDK installed in the selected Eval environment.
    resolved_python = Path(python).expanduser().absolute()
    if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
        raise ValueError("LLM preflight python must be an executable file")
    root = Path(repo_root).expanduser().resolve(strict=True)
    started_at = _utc_now()
    started = time.monotonic()
    challenge = secrets.token_hex(16)
    expected_response = f"{LLM_PREFLIGHT_TOKEN}:{challenge}"
    child_environment = dict(os.environ if environment is None else environment)
    allowed_environment = {
        "ALL_PROXY",
        "CODEX_API_KEY",
        "CODEX_BASE_URL",
        "CODEX_BIN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "NO_PROXY",
        "OPENAI_API_KEY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "all_proxy",
        "https_proxy",
        "http_proxy",
        "no_proxy",
    }
    child_environment = {
        key: value
        for key, value in child_environment.items()
        if key in allowed_environment
    }
    network_binding = network_environment_binding(child_environment)

    with tempfile.TemporaryDirectory(prefix="rpent-llm-preflight-") as temporary:
        workspace = Path(temporary)
        isolated_home = workspace / "home"
        isolated_codex_home = workspace / "codex-home"
        isolated_home.mkdir(mode=0o700)
        isolated_codex_home.mkdir(mode=0o700)
        source_codex_home = Path(
            os.environ.get(
                "CODEX_HOME",
                str(Path(os.environ.get("HOME", "/nonexistent")) / ".codex"),
            )
        )
        source_auth = source_codex_home / "auth.json"
        if source_auth.is_file() and not source_auth.is_symlink():
            target_auth = isolated_codex_home / "auth.json"
            shutil.copyfile(source_auth, target_auth)
            target_auth.chmod(0o600)
        child_environment["HOME"] = str(isolated_home)
        child_environment["CODEX_HOME"] = str(isolated_codex_home)
        command = (
            str(resolved_python),
            "-m",
            "robots.behavior.llm_preflight",
            "--worker",
            "--model",
            str(model),
            "--workspace",
            str(workspace),
            "--challenge",
            challenge,
        )
        process = subprocess.Popen(
            command,
            cwd=root,
            env=child_environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        cleanup_required = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup_required = True
            cleanup_verified = _terminate_worker_group(process)
            stdout, stderr = process.communicate()
        else:
            cleanup_verified = _wait_worker_group_exit(process.pid, timeout_s=2.0)
            if not cleanup_verified:
                cleanup_required = True
                cleanup_verified = _terminate_worker_group(process)

    payload = _worker_payload(stdout)
    response = payload.get("response") if isinstance(payload, dict) else None
    completed_status = (
        payload.get("completed_status") if isinstance(payload, dict) else None
    )
    worker_errors = payload.get("errors") if isinstance(payload, dict) else None
    forbidden_events = (
        payload.get("forbidden_events") if isinstance(payload, dict) else None
    )
    errors = (
        [redact_text(str(item)) for item in worker_errors]
        if isinstance(worker_errors, list)
        else []
    )
    diagnostic = "\n".join((stderr, *errors)).lower()
    transient_count = sum(diagnostic.count(marker) for marker in _TRANSIENT_MARKERS)
    valid_response = bool(
        process.returncode == 0
        and not timed_out
        and isinstance(response, str)
        and response.strip() == expected_response
        and completed_status == "completed"
        and not errors
        and forbidden_events == []
        and not cleanup_required
        and cleanup_verified
    )
    status = (
        "degraded"
        if valid_response and transient_count
        else "passed"
        if valid_response
        else "failed"
    )
    failure_reason = None
    if status == "failed":
        if timed_out:
            failure_reason = "timeout"
        elif process.returncode != 0:
            failure_reason = "worker_exit_nonzero"
        elif payload is None:
            failure_reason = "missing_worker_result"
        elif forbidden_events:
            failure_reason = "forbidden_tool_or_file_event"
        elif errors:
            failure_reason = "final_transport_error"
        elif not response:
            failure_reason = "zero_output"
        elif cleanup_required or not cleanup_verified:
            failure_reason = "worker_cleanup_not_clean"
        else:
            failure_reason = "unexpected_response"
    response_bytes = response.encode("utf-8") if isinstance(response, str) else b""
    return {
        "schema_version": LLM_PREFLIGHT_SCHEMA_VERSION,
        "kind": "agentic_llm_proxy_preflight",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_s": round(max(0.0, time.monotonic() - started), 3),
        "timeout_s": timeout_s,
        "outer_invocation_count": 1,
        "model": str(model),
        "reasoning_effort": "low",
        "status": status,
        "valid_response": valid_response,
        "response_chars": len(response) if isinstance(response, str) else 0,
        "response_sha256": (
            hashlib.sha256(response_bytes).hexdigest() if response_bytes else None
        ),
        "challenge_sha256": hashlib.sha256(challenge.encode("ascii")).hexdigest(),
        "transient_transport_events": transient_count,
        "warning": (
            "valid response after transient transport events"
            if status == "degraded"
            else None
        ),
        "failure_reason": failure_reason,
        "diagnostic_errors": errors,
        "worker_returncode": process.returncode,
        "cleanup_verified": cleanup_verified,
        "network_environment": network_binding,
        "isolation": {
            "ephemeral_thread": True,
            "tools_enabled": False,
            "task_context_supplied": False,
            "environment_rpc_supplied": False,
            "frozen_memory_supplied": False,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--challenge", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.worker:
        raise SystemExit("llm_preflight is an internal worker")
    try:
        result = _worker_result(
            model=args.model,
            workspace=args.workspace.resolve(),
            challenge=args.challenge,
        )
    except BaseException as error:
        result = {
            "response": "",
            "completed_status": "",
            "errors": [
                redact_text(
                    f"preflight worker exception: {type(error).__name__}: {error}"
                )
            ],
            "forbidden_events": [],
        }
    print(_RESULT_PREFIX + json.dumps(result, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LLM_PREFLIGHT_SCHEMA_VERSION",
    "LLM_PREFLIGHT_TIMEOUT_S",
    "network_environment_binding",
    "run_llm_proxy_preflight",
]
