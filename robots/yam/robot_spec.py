# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""YAM robot extension runner hooks."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.yam.contracts import (
    MODEL_SPEC,
    YAM_CAMERA_NAMES,
    env_runtime_contract,
)
from robots.yam.prompt_bundle import system_prompt, user_prompt
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.memory import MemoryManager
from rpent.robots.prompt_bundle import PromptBundle
from rpent.robots.robot_spec import RobotSpec, RunConfig
from rpent.robots.runtime import stop_owned_daemons, try_spawn_server, try_wait_server
from rpent.utils.config import get_memory_dir, get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon


YAM_DASHBOARD_SPEC = {
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <task_name> <seed>",
        "fields": (
            {"name": "task_name"},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        "display": "{task_name} / seed {seed}",
        "output_slug": "{task_name}_s{seed}",
    },
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "unique"},
        {"name": "vla", "label": "VLA", "scope": "shared"},
    ),
    "frame_channels": tuple(
        {"name": name, "label": f"{name} camera"} for name in YAM_CAMERA_NAMES
    ),
}


def get_robot_spec() -> RobotSpec:
    return RobotSpec(
        name="yam",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        dashboard=YAM_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
    config: RunConfig,
    mode: str = "evaluation",
    attempts_per_session: int = 0,
    state_output_dir: Path | str | None = None,
):
    from robots.yam.toolkit import YamToolkit

    explore = mode == "exploration"
    memory = MemoryManager(
        root=config.prompt_vars.get("memory_dir") or get_memory_dir("yam"),
        memory_access="inbox_write" if explore else "read_only",
        inbox_cell_tag=config.recipe_tag if explore else None,
    )
    return YamToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
        memory=memory,
        mode=mode,
        attempts_per_session=attempts_per_session,
        state_output_dir=state_output_dir,
        run_output_dir=config.output_dir,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument("--task-name", required=required)
    parser.add_argument("--task-language", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--explore-attempts-per-session", type=int, default=5)
    parser.add_argument("--explore-sessions", type=int, default=1)
    parser.add_argument(
        "--auto-merge-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--env-endpoint",
        required=required and os.environ.get("YAM_ENV_ENDPOINT") is None,
        default=os.environ.get("YAM_ENV_ENDPOINT"),
        help="YAM env_server endpoint. Use socket://host:port for pickle TCP.",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=os.environ.get("YAM_VLA_ENDPOINT"),
        help="Existing Pi0.5 YAM VLA endpoint.",
    )
    parser.add_argument(
        "--vla-model-path",
        default=os.environ.get("YAM_PI05_CHECKPOINT_PATH"),
        help="Local YAM Pi0.5 checkpoint used only when no --vla-endpoint is given.",
    )
    parser.add_argument(
        "--without-vla",
        action="store_true",
        help="Run primitives-only wiring without starting or connecting a VLA server.",
    )
    parser.add_argument(
        "--yam-reset-on-connect",
        action="store_true",
        help="Explicitly reset the real robot when the client connects. Default is observe-only.",
    )
    parser.add_argument(
        "--yam-rlinf-root",
        default=os.environ.get("RPENT_RLINF_ROOT") or os.environ.get("RLINF_REPO_PATH"),
    )
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument("--vla-cuda-device", default=None)


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if not args.task_name:
        raise ValueError("--task-name is required")
    explore = bool(getattr(args, "explore", False))
    memory_profile = getattr(args, "memory_profile", None) or (
        "local" if explore else "hf"
    )
    args.memory_profile = memory_profile
    memory_dir_arg = getattr(args, "memory_dir", None)
    memory_dir = (
        Path(memory_dir_arg).expanduser().resolve()
        if memory_dir_arg
        else get_memory_dir("yam")
    )
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root() / "logs" / f"{timestamp}_yam_{args.task_name}_s{args.seed}"
        )
    recipe_tag = f"yam_{args.task_name}_s{args.seed}"
    instruction = args.task_language or args.task_name.replace("_", " ")
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=Path(output_dir),
        prompt_vars={
            "recipe_tag": recipe_tag,
            "task_name": args.task_name,
            "seed": args.seed,
            "instruction": instruction,
            "mode": "explore" if explore else "eval",
            "memory_profile": memory_profile,
            "memory_dir": str(memory_dir),
            "memory_inbox": str(memory_dir / "_inbox" / recipe_tag),
            "session_number": 1,
            "session_max": max(1, int(getattr(args, "explore_sessions", 1) or 1)),
        },
        task_desc={
            "env": "yam",
            "task_name": args.task_name,
            "requested_seed": args.seed,
            "instruction": instruction,
            "policy_name": MODEL_SPEC.policy_name,
            "action_layout": MODEL_SPEC.action_layout,
            "camera_order": list(MODEL_SPEC.camera_order),
        },
    )


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
    components: set[str] | None,
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    available = {"env", "vla"}
    selected = available if components is None else components
    if getattr(args, "without_vla", False):
        selected = set(selected) - {"vla"}
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown YAM runtime components: {sorted(unknown)}")

    owned_daemons: dict[str, ProcessDaemon] = {}
    env_pending = None
    vla_pending = None
    if "env" in selected:
        env_pending = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "env",
            lambda: _connect_external_env(args),
        )
    if "vla" in selected:
        vla_pending = try_spawn_server(
            owned_daemons,
            dashboard_events,
            "vla",
            lambda: _spawn_or_connect_vla(args, output_dir),
        )

    primitives_kwargs: dict[str, Any] = {}
    if env_pending is not None:
        env_daemon, env_rpc = env_pending
        primitives_kwargs.update(
            try_wait_server(
                owned_daemons,
                dashboard_events,
                "env",
                env_rpc,
                env_daemon,
                300.0,
                post_fn=lambda: _build_env_runtime_kwargs(args, env_rpc),
            )
        )
    if vla_pending is not None:
        vla_daemon, vla_rpc = vla_pending
        try:
            from rpent.utils.rpc import wait_for_ready

            wait_for_ready(
                vla_rpc, daemon=vla_daemon, timeout_s=900.0 if vla_daemon else 300.0
            )
            primitives_kwargs.update(_build_vla_runtime_kwargs(vla_rpc))
        except Exception as exc:
            stop_owned_daemons(owned_daemons, dashboard_events)
            dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
            raise RuntimeError(f"[vla] wait / client connect failed: {exc}") from exc
        dashboard_events.emit(RuntimeStatusEvent("vla", "ready"))
    return list(owned_daemons.values()), primitives_kwargs


def _connect_external_env(args: argparse.Namespace):
    if not args.env_endpoint:
        raise ValueError(
            "--env-endpoint is required for YAM; start env_server on yambox first"
        )
    from rpent.utils.rpc import make_rpc_client

    return None, make_rpc_client(args.env_endpoint)


def _spawn_or_connect_vla(args: argparse.Namespace, output_dir: Path):
    from rpent.utils.rpc import make_rpc_client

    if args.vla_endpoint:
        return None, make_rpc_client(args.vla_endpoint)
    if not args.vla_model_path:
        raise ValueError(
            "--vla-endpoint, --vla-model-path, or --without-vla is required"
        )
    from rpent.utils.daemon import ProcessDaemon, pick_free_port

    host, port = "127.0.0.1", pick_free_port()
    cmd = [
        sys.executable,
        "-m",
        "robots.yam.vla_server",
        "--model-path",
        str(Path(args.vla_model_path).expanduser()),
        "--host",
        host,
        "--port",
        str(port),
        "--transport",
        "http",
        "--parent-watch",
    ]
    env_overrides = {}
    if args.yam_rlinf_root:
        env_overrides["RPENT_RLINF_ROOT"] = str(
            Path(args.yam_rlinf_root).expanduser().resolve()
        )
    device = (
        args.vla_cuda_device if args.vla_cuda_device is not None else args.cuda_device
    )
    if device is not None:
        env_overrides["CUDA_VISIBLE_DEVICES"] = str(device)
    daemon = ProcessDaemon(
        "yam_pi05_vla_server",
        cmd,
        env_overrides=env_overrides,
        log_path=str(output_dir / "yam_pi05_vla_server.log"),
        cwd=str(get_repo_root()),
    )
    daemon.start()
    return daemon, make_rpc_client(f"http://{host}:{port}")


def _build_env_runtime_kwargs(args: argparse.Namespace, env_rpc: Any) -> dict[str, Any]:
    from robots.yam.env_client import YamEnvClient

    return {
        "env": YamEnvClient(
            env_rpc,
            expected_meta=env_runtime_contract(
                task_name=args.task_name,
                seed=int(args.seed),
                max_episode_steps=int(args.max_episode_steps),
            ),
            reset_on_connect=bool(args.yam_reset_on_connect),
        ),
        "seed": int(args.seed),
        "seed_mode": "label_only",
    }


def _build_vla_runtime_kwargs(vla_rpc: Any) -> dict[str, Any]:
    from robots.yam.vla_client import YamVLAClient

    return {"model": YamVLAClient(vla_rpc)}
