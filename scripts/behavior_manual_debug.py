#!/usr/bin/env python3
"""Run one BEHAVIOR env for operator-driven hybrid primitive debugging.

This is deliberately not an Eval/Explore runner and does not start an LLM. It
owns one env process, one checkpoint-bound VLA sidecar, one live Dashboard, and
a JSON-lines command loop so the surrounding Codex conversation can plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_RPENT_ROOT = Path(__file__).resolve().parents[1]
if str(_RPENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RPENT_ROOT))

from robots.behavior.dashboard_server import DashboardServer  # noqa: E402
from robots.behavior.dashboard_state import State  # noqa: E402
from robots.behavior.env_client import BehaviorEnvClient  # noqa: E402
from robots.behavior.runtime import (  # noqa: E402
    DEFAULT_ACTION_CHUNK,
    DEFAULT_MAX_EPISODE_STEPS,
    _expected_shared_policy_checkpoint_binding,
    _managed_env_rpc_client,
    _owned_process_group_alive,
    _terminate_process,
    prepare_campaign_runtime_isolation,
    start_env_server,
    start_vla_server,
    stop_env_server,
)
from robots.behavior.task_specs import PICKING_UP_TRASH_TASK_SPEC  # noqa: E402
from robots.behavior.toolkit import BehaviorToolkit  # noqa: E402
from robots.behavior.vla_client import BehaviorVLAClient  # noqa: E402

_RLINF_ROOT = Path("/home/ubuntu/lwb/Projects/RLinf_agentic_push")
_BEHAVIOR_PYTHON = _RLINF_ROOT / ".venv-behavior" / "bin" / "python"
_POLICY_CHECKPOINT = Path(
    "/home/ubuntu/lwb/Models/openpi_comet_pytorch/pi05-b1kpt50-cs32"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual hybrid BEHAVIOR debug host with an owned VLA sidecar",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ubuntu/lwb/RPent_outputs/manual_debug",
    )
    parser.add_argument("--public-seed", type=int, default=13)
    parser.add_argument("--cuda-device", default="7")
    parser.add_argument("--dashboard-host", default="0.0.0.0")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument(
        "--dashboard-language",
        choices=("en", "zh-cn"),
        default="en",
    )
    parser.add_argument("--env-ready-timeout-s", type=int, default=1800)
    parser.add_argument("--vla-ready-timeout-s", type=int, default=1800)
    parser.add_argument("--max-tool-calls", type=int, default=350)
    parser.add_argument("--max-wall-clock-s", type=float, default=43200.0)
    return parser


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _new_output_dir(root: Path, public_seed: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = root / f"picking_up_trash_s{public_seed}_manual_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def _instance_dir() -> Path:
    return (
        _RLINF_ROOT
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
        / "scenes"
        / PICKING_UP_TRASH_TASK_SPEC.scene_model
        / "json"
        / PICKING_UP_TRASH_TASK_SPEC.state_dir_name
    )


def _runtime_args(
    cli: argparse.Namespace,
    *,
    output_dir: Path,
    instance_id: int,
) -> SimpleNamespace:
    namespace = f"manual-s{cli.public_seed}-{output_dir.name[-16:]}"
    isolation = prepare_campaign_runtime_isolation(
        output_dir / "runtime",
        namespace,
        str(cli.cuda_device),
        behavior_python=str(_BEHAVIOR_PYTHON),
    )
    return SimpleNamespace(
        behavior_python=str(_BEHAVIOR_PYTHON),
        behavior_repo=str(_RLINF_ROOT),
        behavior_config=None,
        policy_checkpoint=str(_POLICY_CHECKPOINT),
        cuda_device=str(cli.cuda_device),
        _behavior_runtime_isolation=isolation,
        suite="behavior_2025_challenge",
        task=PICKING_UP_TRASH_TASK_SPEC.task_index,
        task_name=PICKING_UP_TRASH_TASK_SPEC.task_name,
        activity_definition_id=PICKING_UP_TRASH_TASK_SPEC.activity_definition_id,
        activity_instance_id=int(instance_id),
        activity_instance_dir=str(_instance_dir()),
        scene_model=PICKING_UP_TRASH_TASK_SPEC.scene_model,
        seed=0,
        public_seed=int(cli.public_seed),
        behavior_attempt_index=1,
        behavior_controller_mode="hybrid",
        max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
        env_ready_timeout_s=int(cli.env_ready_timeout_s),
        vla_port=0,
        vla_ready_timeout_s=int(cli.vla_ready_timeout_s),
        _behavior_policy_checkpoint_binding=(
            _expected_shared_policy_checkpoint_binding()
        ),
    )


def _start_bound_vla(
    runtime_args: SimpleNamespace,
    *,
    output_dir: Path,
    binding_id: str,
) -> tuple[str, Any, BehaviorVLAClient, dict[str, Any]]:
    """Start one owned VLA sidecar and bind its disabled action gate."""

    endpoint, process = start_vla_server(runtime_args, output_dir=output_dir)
    model = None
    try:
        model = BehaviorVLAClient(endpoint)
        health = model.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=(
                runtime_args._behavior_policy_checkpoint_binding
            ),
        )
        if health.get("config_name") != "pi05_behavior":
            raise RuntimeError(f"unexpected VLA metadata: {health!r}")
        model.disable_actions()
        bound = model.bind_actions(binding_id)
        expected_binding_digest = hashlib.sha256(
            binding_id.encode("utf-8")
        ).hexdigest()
        if (
            bound.get("actions_enabled") is not False
            or bound.get("binding_digest") != expected_binding_digest
        ):
            raise RuntimeError("VLA fresh binding did not remain disabled")
        health = model.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=(
                runtime_args._behavior_policy_checkpoint_binding
            ),
        )
        if (
            health.get("actions_enabled") is not False
            or health.get("binding_digest") != expected_binding_digest
        ):
            raise RuntimeError("VLA fresh binding did not preserve the idle gate")
        return endpoint, process, model, health
    except BaseException:
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        finally:
            _terminate_process(process)
        raise


def _persist_bytes(
    value: Any,
    *,
    image_dir: Path,
    call_index: int,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        image_dir.mkdir(parents=True, exist_ok=True)
        safe_path = "_".join(path) or "image"
        safe_path = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in safe_path
        )
        suffix = ".png" if "image" in safe_path or "depth" in safe_path else ".bin"
        target = image_dir / f"call_{call_index:04d}_{safe_path}{suffix}"
        target.write_bytes(bytes(value))
        return {
            "bytes": len(value),
            "saved_path": str(target.resolve()),
        }
    if isinstance(value, dict):
        return {
            str(key): _persist_bytes(
                item,
                image_dir=image_dir,
                call_index=call_index,
                path=(*path, str(key)),
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _persist_bytes(
                item,
                image_dir=image_dir,
                call_index=call_index,
                path=(*path, str(index)),
            )
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _final_manifest_status(
    primary_status: str,
    cleanup_errors: list[str],
) -> str:
    if primary_status == "startup_failed":
        return (
            "startup_failed_cleanup_failed"
            if cleanup_errors
            else "startup_failed"
        )
    return "cleanup_failed" if cleanup_errors else "stopped"


def _tool_process_rejection_reason(
    name: str,
    *,
    env_proc: Any,
    vla_proc: Any,
) -> str | None:
    """Return a zero-action process-liveness rejection for one public tool."""

    if env_proc is None or env_proc.poll() is not None:
        return "owned env process exited; no further tool is allowed"
    if (
        name == "pi0_nav_pick"
        and (vla_proc is None or vla_proc.poll() is not None)
    ):
        return "owned VLA process exited; pi0_nav_pick is unavailable"
    return None


def _latch_dashboard_terminal(state: Any, toolkit: Any, result: Any) -> bool:
    continuation = toolkit.runner_continuation_state()
    verified_success = (
        continuation.get("raw_official_success_verified") is True
    )
    terminal = bool(getattr(result, "is_finish", False) or verified_success)
    if not terminal:
        return False
    attempt_index = int(continuation.get("attempt_index", 1))
    if verified_success:
        state.on_event(
            {
                "type": "official_success",
                "attempt_index": attempt_index,
                "task_success": True,
                "workflow_complete": False,
                "artifact_seal_complete": False,
                "publication_complete": False,
            }
        )
        outcome = "official_task_success"
    else:
        outcome = "terminal_without_official_success"
    state.end_attempt(attempt_index=attempt_index, outcome=outcome)
    state.mark_done(terminated=verified_success)
    return True


def main() -> int:
    cli = _parser().parse_args()
    if cli.env_ready_timeout_s <= 0 or cli.vla_ready_timeout_s <= 0:
        raise ValueError("env and VLA ready timeouts must be positive")
    if cli.max_tool_calls <= 0 or cli.max_wall_clock_s <= 0:
        raise ValueError("manual debug budgets must be positive")
    output_dir = _new_output_dir(Path(cli.output_root).expanduser(), cli.public_seed)
    instance_id = PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(
        cli.public_seed,
        phase="eval",
    )
    runtime_args = _runtime_args(
        cli,
        output_dir=output_dir,
        instance_id=instance_id,
    )
    run_id = f"behavior-manual/{output_dir.name}"
    dashboard = DashboardServer(
        host=cli.dashboard_host,
        port=cli.dashboard_port,
        language=cli.dashboard_language,
    )
    state = State(
        run_id=run_id,
        name=f"picking_up_trash_s{cli.public_seed}_manual",
        suite=runtime_args.suite,
        task=runtime_args.task,
        seed=cli.public_seed,
        output_dir=str(output_dir),
        video_path=str(output_dir / "episode.mp4"),
    )
    state.set_metadata(
        {
            "planner": "current-codex-conversation",
            "model": "pi0.5-manual-planner",
            "task-name": runtime_args.task_name,
            "task-index": runtime_args.task,
            "activity-definition-id": runtime_args.activity_definition_id,
            "activity-instance-id": instance_id,
            "public-seed": cli.public_seed,
            "scene-model": runtime_args.scene_model,
            "controller": "hybrid",
            "llm-enabled": False,
            "cuda-device": cli.cuda_device,
            "health-status": "starting",
        }
    )
    manifest_path = output_dir / "manual_debug_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "behavior_manual_debug",
        "formal_eval": False,
        "analytic_only": False,
        "hybrid_vla_debug": True,
        "vla_started": False,
        "status": "starting",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "task_name": runtime_args.task_name,
        "task": runtime_args.task,
        "activity_definition_id": runtime_args.activity_definition_id,
        "activity_instance_id": instance_id,
        "public_seed": cli.public_seed,
        "native_seed": runtime_args.seed,
        "scene_model": runtime_args.scene_model,
        "cuda_device": cli.cuda_device,
        "started_at_unix_s": time.time(),
    }
    _write_manifest(manifest_path, manifest)

    env_proc = None
    vla_proc = None
    env_rpc_client = None
    model = None
    vla_endpoint = None
    vla_health: dict[str, Any] | None = None
    toolkit = None
    dashboard_started = False
    terminal_latched = False
    call_index = 0
    primary_status = "starting"

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    previous_sigint = signal.signal(signal.SIGINT, interrupt)
    try:
        dashboard_url = dashboard.start()
        dashboard_started = True
        dashboard.register(state)
        state.begin_attempt(
            attempt_index=1,
            output_dir=output_dir,
            video_path=output_dir / "episode.mp4",
        )
        env_proc = start_env_server(runtime_args, output_dir=output_dir)
        vla_binding_id = f"manual:{output_dir.name}"
        vla_endpoint, vla_proc, model, vla_health = _start_bound_vla(
            runtime_args,
            output_dir=output_dir,
            binding_id=vla_binding_id,
        )
        env_rpc_client = _managed_env_rpc_client(env_proc)
        env = BehaviorEnvClient(
            env_rpc_client,
            expected_meta={
                "suite": runtime_args.suite,
                "task": runtime_args.task,
                "task_name": runtime_args.task_name,
                "public_seed": runtime_args.public_seed,
                "max_episode_steps": runtime_args.max_episode_steps,
            },
        )
        env.vla_endpoint = vla_endpoint
        initial_observation, initial_info = env.reset()
        toolkit = BehaviorToolkit(
            primitives_kwargs={
                "env": env,
                "model": model,
                "task_name": runtime_args.task_name,
                "behavior_phase": "eval",
                "public_seed": runtime_args.public_seed,
                "initial_attempt_index": 1,
                "job_id": None,
                "max_episode_steps": runtime_args.max_episode_steps,
                "max_tool_calls": cli.max_tool_calls,
                "max_wall_clock_s": cli.max_wall_clock_s,
                "pure_vla_baseline": False,
                "action_horizon": DEFAULT_ACTION_CHUNK,
                "output_dir": output_dir,
                "video_path": output_dir / "episode.mp4",
                "initial_observation": initial_observation,
                "initial_info": initial_info,
            },
            video_path=output_dir / "episode.mp4",
            dashboard=state,
        )
        state.set_metadata(
            {
                "health-status": "ready",
                "health-checked-at": time.time(),
                "task-language": PICKING_UP_TRASH_TASK_SPEC.task_language,
                "vla-endpoint": vla_endpoint,
                "vla-pid": vla_proc.pid,
                "vla-actions-enabled": vla_health.get("actions_enabled"),
            }
        )
        manifest.update(
            {
                "status": "ready",
                "dashboard_url": dashboard_url,
                "dashboard_local_url": (
                    f"http://127.0.0.1:{cli.dashboard_port}"
                    if cli.dashboard_port
                    else dashboard_url
                ),
                "env_pid": env_proc.pid,
                "env_transport_host": getattr(
                    env_proc,
                    "_rpent_transport_host",
                    None,
                ),
                "env_transport_port": getattr(
                    env_proc,
                    "_rpent_transport_port",
                    None,
                ),
                "vla_started": True,
                "vla_available": True,
                "vla_endpoint": vla_endpoint,
                "vla_pid": vla_proc.pid,
                "vla_binding_id": vla_binding_id,
                "vla_checkpoint_binding_sha256": (
                    runtime_args._behavior_policy_checkpoint_binding[
                        "binding_sha256"
                    ]
                ),
                "vla_health": {
                    "status": vla_health.get("status"),
                    "config_name": vla_health.get("config_name"),
                    "action_horizon": vla_health.get("action_horizon"),
                    "action_dim": vla_health.get("action_dim"),
                    "actions_enabled": vla_health.get("actions_enabled"),
                },
                "ready_at_unix_s": time.time(),
            }
        )
        primary_status = "ready"
        _write_manifest(manifest_path, manifest)

        call_index += 1
        initial_result = toolkit.execute_tool("observe", {"camera": "head"})
        initial_public = _persist_bytes(
            initial_result.result,
            image_dir=output_dir / "manual_images",
            call_index=call_index,
        )
        terminal_latched = _latch_dashboard_terminal(
            state,
            toolkit,
            initial_result,
        )
        _emit(
            "manual_debug_ready",
            dashboard_url=dashboard_url,
            dashboard_local_url=manifest["dashboard_local_url"],
            run_id=run_id,
            output_dir=str(output_dir),
            env_pid=env_proc.pid,
            env_transport_port=manifest["env_transport_port"],
            initial_observe=initial_public,
            terminal_latched=terminal_latched,
            vla_available=True,
            vla_endpoint=vla_endpoint,
            vla_pid=vla_proc.pid,
            vla_actions_enabled=vla_health.get("actions_enabled"),
        )

        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                command = json.loads(raw_line)
                if not isinstance(command, dict):
                    raise ValueError("command must be a JSON object")
                operation = command.get("command")
                if operation == "shutdown":
                    _emit("shutdown_acknowledged")
                    break
                if operation == "status":
                    _emit(
                        "status",
                        run_id=run_id,
                        output_dir=str(output_dir),
                        env_pid=env_proc.pid,
                        env_returncode=env_proc.poll(),
                        vla_pid=vla_proc.pid,
                        vla_returncode=vla_proc.poll(),
                        vla_endpoint=vla_endpoint,
                        env_steps=env.total_env_steps,
                        terminal_latched=terminal_latched,
                        continuation=toolkit.runner_continuation_state(),
                    )
                    continue
                if operation != "tool":
                    raise ValueError(
                        "command must be one of: tool, status, shutdown"
                    )
                name = command.get("name")
                input_dict = command.get("input", {})
                if not isinstance(name, str) or not isinstance(input_dict, dict):
                    raise ValueError("tool command requires string name and object input")
                if terminal_latched:
                    _emit(
                        "tool_rejected",
                        name=name,
                        reason="terminal result is latched; no further tool is allowed",
                    )
                    continue
                process_rejection = _tool_process_rejection_reason(
                    name,
                    env_proc=env_proc,
                    vla_proc=vla_proc,
                )
                if process_rejection is not None:
                    _emit(
                        "tool_rejected",
                        name=name,
                        reason=process_rejection,
                    )
                    continue
                call_index += 1
                result = toolkit.execute_tool(name, input_dict)
                public = _persist_bytes(
                    result.result,
                    image_dir=output_dir / "manual_images",
                    call_index=call_index,
                )
                terminal_latched = _latch_dashboard_terminal(
                    state,
                    toolkit,
                    result,
                )
                _emit(
                    "tool_result",
                    call_index=call_index,
                    name=name,
                    result=public,
                    is_finish=result.is_finish,
                    terminal_latched=terminal_latched,
                )
            except Exception as exc:
                _emit(
                    "command_error",
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
    except KeyboardInterrupt:
        _emit("manual_debug_interrupted")
    except BaseException as exc:
        primary_status = "startup_failed"
        manifest["status"] = "startup_failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        _emit(
            "manual_debug_error",
            error=manifest["error"],
            traceback=traceback.format_exc(),
            output_dir=str(output_dir),
        )
        return 1
    finally:
        cleanup_errors: list[str] = []
        try:
            stop_env_server(env_proc, output_dir=output_dir)
        except BaseException as exc:
            cleanup_errors.append(f"env: {type(exc).__name__}: {exc}")
        if toolkit is not None:
            try:
                toolkit.close()
            except BaseException as exc:
                cleanup_errors.append(f"toolkit: {type(exc).__name__}: {exc}")
            finally:
                model = None
        if model is not None:
            try:
                model.close()
            except BaseException as exc:
                cleanup_errors.append(f"vla_client: {type(exc).__name__}: {exc}")
        if vla_proc is not None:
            try:
                _terminate_process(vla_proc)
                if _owned_process_group_alive(vla_proc):
                    raise RuntimeError("owned VLA process group is still alive")
            except BaseException as exc:
                cleanup_errors.append(f"vla: {type(exc).__name__}: {exc}")
        if dashboard_started:
            try:
                dashboard.stop()
            except BaseException as exc:
                cleanup_errors.append(f"dashboard: {type(exc).__name__}: {exc}")
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        manifest.update(
            {
                "status": _final_manifest_status(
                    primary_status,
                    cleanup_errors,
                ),
                "cleanup_errors": cleanup_errors,
                "env_returncode": (
                    env_proc.poll() if env_proc is not None else None
                ),
                "vla_returncode": (
                    vla_proc.poll() if vla_proc is not None else None
                ),
                "stopped_at_unix_s": time.time(),
            }
        )
        _write_manifest(manifest_path, manifest)
        _emit(
            "manual_debug_stopped",
            cleanup_errors=cleanup_errors,
            output_dir=str(output_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
