#!/usr/bin/env python3
"""Operator-driven BEHAVIOR/R1Pro debug host.

This is intentionally not an Eval/Explore runner and it does not start an LLM.
It owns one simulator env, an optional VLA sidecar, and one Dashboard. JSONL
commands on stdin are executed through the same public ``BehaviorToolkit``
instance that is bound to Dashboard manual control.
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

from robots.behavior import get_toolkit  # noqa: E402
from robots.behavior.dashboard_control import (  # noqa: E402
    BehaviorCommandArbiter,
    BehaviorRawSuccessLatch,
)
from robots.behavior.dashboard_server import DashboardServer  # noqa: E402
from robots.behavior.dashboard_state import State  # noqa: E402
from robots.behavior.env_client import BehaviorEnvClient  # noqa: E402
from robots.behavior.runtime import (  # noqa: E402
    DEFAULT_ACTION_CHUNK,
    DEFAULT_MAX_EPISODE_STEPS,
    BehaviorRuntimeResources,
    _expected_shared_policy_checkpoint_binding,
    _managed_env_rpc_client,
    prepare_campaign_runtime_isolation,
    start_env_server,
    start_vla_server,
)
from robots.behavior.task_specs import PICKING_UP_TRASH_TASK_SPEC  # noqa: E402
from robots.behavior.vla_client import BehaviorVLAClient  # noqa: E402

_RLINF_ROOT = Path("/home/ubuntu/lwb/Projects/RLinf_agentic_push")
_BEHAVIOR_PYTHON = _RLINF_ROOT / ".venv-behavior" / "bin" / "python"
_POLICY_CHECKPOINT = Path(
    "/home/ubuntu/lwb/Models/openpi_comet_pytorch/pi05-b1kpt50-cs32"
)
_DISABLED_TOOLS = frozenset({"pi0_nav_pick"})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-env manual BEHAVIOR whole-body planning host",
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
    parser.add_argument(
        "--planner-only",
        action="store_true",
        help="Do not start VLA; disable only pi0_nav_pick.",
    )
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


def _expected_env_meta(args: SimpleNamespace) -> dict[str, Any]:
    return {
        "suite": args.suite,
        "task": args.task,
        "task_name": args.task_name,
        "public_seed": args.public_seed,
        "max_episode_steps": args.max_episode_steps,
    }


def _make_state(
    *,
    output_dir: Path,
    run_id: str,
    args: SimpleNamespace,
    cli: argparse.Namespace,
) -> State:
    return State(
        run_id=run_id,
        name=f"picking_up_trash_s{args.public_seed}_manual",
        suite=args.suite,
        task=args.task,
        seed=args.public_seed,
        environment="behavior",
        output_dir=str(output_dir),
        video_path=str(output_dir / "episode.mp4"),
        identity={
            "suite": args.suite,
            "task": args.task,
            "task_name": args.task_name,
            "public_seed": args.public_seed,
            "activity_definition_id": args.activity_definition_id,
            "activity_instance_id": args.activity_instance_id,
        },
        metadata={
            "planner": "current-codex-conversation",
            "model": "none",
            "task-name": args.task_name,
            "task-index": args.task,
            "activity-definition-id": args.activity_definition_id,
            "activity-instance-id": args.activity_instance_id,
            "public-seed": args.public_seed,
            "scene-model": args.scene_model,
            "behavior-phase": "manual-debug",
            "controller": "hybrid",
            "llm-enabled": False,
            "cuda-device": cli.cuda_device,
            "health-status": "starting",
        },
    )


def _build_toolkit(
    *,
    cli: argparse.Namespace,
    args: SimpleNamespace,
    output_dir: Path,
    env: BehaviorEnvClient,
    model: Any,
    initial_observation: dict[str, Any],
    initial_info: Any,
    state: State,
    resource: BehaviorRuntimeResources,
    arbiter: BehaviorCommandArbiter,
    success_latch: BehaviorRawSuccessLatch,
) -> Any:
    """Build and automatically activate the one shared Dashboard toolkit."""

    return get_toolkit(
        primitives_kwargs={
            "env": env,
            "model": model,
            "task_name": args.task_name,
            "behavior_phase": "eval",
            "public_seed": args.public_seed,
            "initial_attempt_index": 1,
            "job_id": None,
            "max_episode_steps": args.max_episode_steps,
            "max_tool_calls": cli.max_tool_calls,
            "max_wall_clock_s": cli.max_wall_clock_s,
            "pure_vla_baseline": False,
            "action_horizon": DEFAULT_ACTION_CHUNK,
            "output_dir": output_dir,
            "video_path": output_dir / "episode.mp4",
            "initial_observation": initial_observation,
            "initial_info": initial_info,
            "_dashboard_runtime_resource": resource,
            "_dashboard_command_arbiter": arbiter,
            "_dashboard_success_latch": success_latch,
            "_dashboard_motion_allowed": True,
            "_dashboard_observe_allowed": True,
            "_dashboard_control_unavailable_reason": None,
        },
        video_path=str(output_dir / "episode.mp4"),
        dashboard=state,
    )


def _start_bound_vla(
    args: SimpleNamespace,
    *,
    output_dir: Path,
    binding_id: str,
) -> tuple[str, Any, BehaviorVLAClient, dict[str, Any]]:
    """Start an owned sidecar with a fresh, initially disabled action gate."""

    endpoint, process = start_vla_server(args, output_dir=output_dir)
    model = BehaviorVLAClient(endpoint)
    try:
        health = model.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=args._behavior_policy_checkpoint_binding,
        )
        if health.get("config_name") != "pi05_behavior":
            raise RuntimeError(f"unexpected VLA metadata: {health!r}")
        model.disable_actions()
        bound = model.bind_actions(binding_id)
        expected_digest = hashlib.sha256(binding_id.encode("utf-8")).hexdigest()
        if (
            bound.get("actions_enabled") is not False
            or bound.get("binding_digest") != expected_digest
        ):
            raise RuntimeError("VLA binding did not remain disabled")
        health = model.healthz(
            timeout_ms=5000,
            expected_checkpoint_binding=args._behavior_policy_checkpoint_binding,
        )
        if (
            health.get("actions_enabled") is not False
            or health.get("binding_digest") != expected_digest
        ):
            raise RuntimeError("VLA binding health check failed")
        return endpoint, process, model, health
    except BaseException:
        model.close()
        from robots.behavior.runtime import _terminate_process

        _terminate_process(process)
        raise


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
        return "startup_failed_cleanup_failed" if cleanup_errors else "startup_failed"
    return "cleanup_failed" if cleanup_errors else "stopped"


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
        return {"bytes": len(value), "saved_path": str(target.resolve())}
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


def _cleanup_owned(
    *,
    toolkit: Any,
    resource: BehaviorRuntimeResources,
    dashboard: Any,
    dashboard_started: bool,
    orphan_model: Any = None,
) -> list[str]:
    """Quiesce toolkit control before env transport/process and Dashboard."""

    errors: list[str] = []
    if toolkit is not None:
        try:
            toolkit.close()
        except BaseException as exc:
            errors.append(f"toolkit: {type(exc).__name__}: {exc}")
    elif orphan_model is not None:
        try:
            orphan_model.close()
        except BaseException as exc:
            errors.append(f"vla_client: {type(exc).__name__}: {exc}")
    try:
        resource.close()
    except BaseException as exc:
        errors.append(f"runtime: {type(exc).__name__}: {exc}")
    if dashboard_started:
        try:
            dashboard.stop()
        except BaseException as exc:
            errors.append(f"dashboard: {type(exc).__name__}: {exc}")
    return errors


def _tool_rejection(
    name: str,
    *,
    terminal: bool,
    env_proc: Any,
    planner_only: bool,
    vla_proc: Any = None,
) -> str | None:
    if name in _DISABLED_TOOLS and planner_only:
        return "pi0_nav_pick is disabled by --planner-only"
    if name in _DISABLED_TOOLS and (vla_proc is None or vla_proc.poll() is not None):
        return "owned VLA process exited; pi0_nav_pick is unavailable"
    if terminal:
        return "official success or terminal state is latched; no further action is allowed"
    if env_proc is None or env_proc.poll() is not None:
        return "owned env process exited; no further tool is allowed"
    return None


def _terminal_latched(toolkit: Any, result: Any = None) -> bool:
    continuation = toolkit.runner_continuation_state()
    return bool(
        continuation.get("raw_official_success_verified") is True
        or getattr(result, "is_finish", False)
    )


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
    args = _runtime_args(cli, output_dir=output_dir, instance_id=instance_id)
    run_id = f"behavior-manual/{output_dir.name}"
    state = _make_state(output_dir=output_dir, run_id=run_id, args=args, cli=cli)
    dashboard = DashboardServer(
        host=cli.dashboard_host,
        port=cli.dashboard_port,
        language=cli.dashboard_language,
    )
    success_latch = BehaviorRawSuccessLatch()
    arbiter = BehaviorCommandArbiter(success_latch=success_latch)
    resource = BehaviorRuntimeResources(
        output_dir=output_dir,
        command_arbiter=arbiter,
        success_latch=success_latch,
        dashboard_state=state,
    )

    toolkit = None
    model = None
    env_proc = None
    vla_proc = None
    vla_endpoint = None
    vla_health = None
    dashboard_started = False
    call_index = 0
    terminal = False
    exit_code = 0
    primary_status = "starting"
    manifest_path = output_dir / "manual_debug_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "kind": "behavior_manual_debug",
        "formal_eval": False,
        "planner_only": bool(cli.planner_only),
        "controller_bound": False,
        "vla_started": False,
        "status": "starting",
        "run_id": run_id,
        "output_dir": str(output_dir),
        "task_name": args.task_name,
        "task": args.task,
        "public_seed": args.public_seed,
        "activity_definition_id": args.activity_definition_id,
        "activity_instance_id": args.activity_instance_id,
        "cuda_device": args.cuda_device,
        "started_at_unix_s": time.time(),
    }
    _write_manifest(manifest_path, manifest)

    def interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    previous_sigint = signal.signal(signal.SIGINT, interrupt)
    try:
        dashboard.start()
        dashboard_started = True
        dashboard.register(state)
        state.begin_attempt(
            attempt_index=1,
            output_dir=output_dir,
            video_path=output_dir / "episode.mp4",
        )

        env_proc = start_env_server(args, output_dir=output_dir)
        resource.env_proc = env_proc
        if not cli.planner_only:
            binding_id = f"manual:{output_dir.name}"
            vla_endpoint, vla_proc, model, vla_health = _start_bound_vla(
                args,
                output_dir=output_dir,
                binding_id=binding_id,
            )
            resource.vla_proc = vla_proc
        rpc = _managed_env_rpc_client(env_proc)
        resource.env_rpc_client = rpc
        env = BehaviorEnvClient(rpc, expected_meta=_expected_env_meta(args))
        env.vla_endpoint = vla_endpoint
        initial_observation, initial_info = env.reset()
        toolkit = _build_toolkit(
            cli=cli,
            args=args,
            output_dir=output_dir,
            env=env,
            model=model,
            initial_observation=initial_observation,
            initial_info=initial_info,
            state=state,
            resource=resource,
            arbiter=arbiter,
            success_latch=success_latch,
        )
        if resource.toolkit is not toolkit or resource.dashboard_controller is None:
            raise RuntimeError("Dashboard controller did not bind to the shared toolkit")
        primary_status = "ready"
        manifest.update(
            {
                "status": "ready",
                "controller_bound": True,
                "env_pid": env_proc.pid,
                "env_transport_host": getattr(
                    env_proc, "_rpent_transport_host", None
                ),
                "env_transport_port": getattr(
                    env_proc, "_rpent_transport_port", None
                ),
                "vla_started": vla_proc is not None,
                "vla_pid": getattr(vla_proc, "pid", None),
                "vla_endpoint": vla_endpoint,
                "vla_health": vla_health,
                "ready_at_unix_s": time.time(),
            }
        )
        _write_manifest(manifest_path, manifest)

        state.set_metadata(
            {
                "health-status": "ready",
                "health-checked-at": time.time(),
                "task-language": PICKING_UP_TRASH_TASK_SPEC.task_language,
            }
        )
        call_index += 1
        initial_result = toolkit.execute_tool("observe", {"camera": "head"})
        initial_public = _persist_bytes(
            initial_result.result,
            image_dir=output_dir / "manual_images",
            call_index=call_index,
        )
        terminal = _terminal_latched(toolkit, initial_result)
        local_url = f"http://127.0.0.1:{dashboard.port}"
        _emit(
            "manual_debug_ready",
            dashboard_url=local_url,
            dashboard_bind=f"http://{cli.dashboard_host}:{dashboard.port}",
            run_id=run_id,
            output_dir=str(output_dir),
            env_pid=env_proc.pid,
            env_transport_port=getattr(env_proc, "_rpent_transport_port", None),
            available_tools=[
                spec["name"]
                for spec in toolkit.get_tools_spec()
                if not (cli.planner_only and spec["name"] in _DISABLED_TOOLS)
            ],
            initial_observe=initial_public,
            terminal_latched=terminal,
            planner_only=cli.planner_only,
            vla_pid=getattr(vla_proc, "pid", None),
            vla_endpoint=vla_endpoint,
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
                    terminal = terminal or success_latch.is_latched()
                    _emit(
                        "status",
                        run_id=run_id,
                        output_dir=str(output_dir),
                        env_pid=env_proc.pid,
                        env_returncode=env_proc.poll(),
                        vla_pid=getattr(vla_proc, "pid", None),
                        vla_returncode=(
                            vla_proc.poll() if vla_proc is not None else None
                        ),
                        env_steps=env.total_env_steps,
                        terminal_latched=terminal,
                        continuation=toolkit.runner_continuation_state(),
                        dashboard_control=toolkit.dashboard_control_capabilities(),
                    )
                    continue
                if operation != "tool":
                    raise ValueError("command must be one of: tool, status, shutdown")
                name = command.get("name")
                input_dict = command.get("input", {})
                if not isinstance(name, str) or not isinstance(input_dict, dict):
                    raise ValueError("tool command requires string name and object input")
                terminal = terminal or success_latch.is_latched()
                rejection = _tool_rejection(
                    name,
                    terminal=terminal,
                    env_proc=env_proc,
                    planner_only=cli.planner_only,
                    vla_proc=vla_proc,
                )
                if rejection is not None:
                    _emit("tool_rejected", name=name, reason=rejection)
                    continue
                call_index += 1
                result = toolkit.execute_tool(name, input_dict)
                public = _persist_bytes(
                    result.result,
                    image_dir=output_dir / "manual_images",
                    call_index=call_index,
                )
                terminal = _terminal_latched(toolkit, result)
                _emit(
                    "tool_result",
                    call_index=call_index,
                    name=name,
                    result=public,
                    is_finish=result.is_finish,
                    terminal_latched=terminal,
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
        exit_code = 1
        primary_status = "startup_failed"
        manifest["status"] = primary_status
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(manifest_path, manifest)
        _emit(
            "manual_debug_error",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            output_dir=str(output_dir),
        )
    finally:
        cleanup_errors = _cleanup_owned(
            toolkit=toolkit,
            resource=resource,
            dashboard=dashboard,
            dashboard_started=dashboard_started,
            orphan_model=model,
        )
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        if cleanup_errors:
            exit_code = 1
        manifest.update(
            {
                "status": _final_manifest_status(primary_status, cleanup_errors),
                "cleanup_errors": cleanup_errors,
                "exit_code": exit_code,
                "controller_bound": False,
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
