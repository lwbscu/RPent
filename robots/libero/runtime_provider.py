"""LIBERO runtime provider for the main CLI."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robots.libero.env_client import LiberoEnvClient
from rpent.envs.runtime import RuntimeHandle
from rpent.utils.config import get_libero_type, get_repo_root
from rpent.utils.logging import get_logger
from rpent.utils.rpc import create_rpc_client, set_socket_endpoint
from rpent.utils.vla_client import VLAClient

logger = get_logger("agent")


def _terminate_process(
    proc: subprocess.Popen | None,
    *,
    timeout: float = 10.0,
) -> None:
    """Terminate and reap a provider-owned process, killing if necessary."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def _pipe_driver_output(
    proc: subprocess.Popen,
    log_file,
    ready_events: "queue.Queue[dict]",
) -> None:
    """Copy driver stdout to log and capture machine-readable ready events."""
    assert proc.stdout is not None
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict) and event.get("event") == "transport_ready":
            ready_events.put(event)


def start_env_server(
    suite: str,
    task: int,
    seed: int,
    output_dir: str | Path,
    max_episode_steps: int = 10000,
    libero_type: str | None = None,
    cuda_device: str | None = None,
    log_path: str | None = None,
    driver_script: str | None = None,
    ready_timeout_s: float = 300.0,
) -> subprocess.Popen:
    """Launch the LIBERO env server and wait for its transport endpoint."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = str(out_dir / "env_server.log")

    env = os.environ.copy()
    env["LIBERO_TYPE"] = libero_type or get_libero_type()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    cmd = [
        sys.executable,
        driver_script or str(get_repo_root() / "robots" / "libero" / "env_server.py"),
        "--suite",
        suite,
        "--task",
        str(task),
        "--seed",
        str(seed),
        "--max-episode-steps",
        str(max_episode_steps),
        "--output-dir",
        str(out_dir),
    ]
    logger.info("env server cmd: %s", shlex.join(cmd))
    logger.info("env server log: %s", log_path)
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s  output_dir=%s",
        env.get("CUDA_VISIBLE_DEVICES"),
        out_dir,
    )
    log_f = open(log_path, "a")
    ready_events: queue.Queue[dict] = queue.Queue()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=get_repo_root(),
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=_pipe_driver_output,
        args=(proc, log_f, ready_events),
        daemon=True,
    ).start()

    logger.info("waiting for env server...")
    t0 = time.time()
    while True:
        try:
            event = ready_events.get(timeout=2.0)
        except queue.Empty:
            event = None
        if (
            event is not None
            and event.get("kind") == "socket"
            and event.get("host")
            and event.get("port")
        ):
            set_socket_endpoint(out_dir, event["host"], int(event["port"]))
            logger.info("env server ready at %s:%s", event["host"], event["port"])
            break
        if proc.poll() is not None:
            logger.error("env server EXITED before becoming ready. Last log:")
            logger.error("%s", Path(log_path).read_text()[-2000:])
            raise RuntimeError("env server exited prematurely")
        if time.time() - t0 > ready_timeout_s:
            _terminate_process(proc)
            raise RuntimeError(f"env server not ready after {ready_timeout_s}s")
    logger.info("env server ready in %.1fs", time.time() - t0)
    return proc


def stop_env_server(
    proc: subprocess.Popen,
    output_dir: str | Path,
    timeout: float = 15.0,
) -> None:
    """Ask the LIBERO env server to shut down, then kill as a fallback."""
    if proc.poll() is not None:
        return
    try:
        client = create_rpc_client(output_dir)
        client.call("shutdown", timeout_s=timeout)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process(proc, timeout=timeout)


def start_vla_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    cuda_device: str | None = None,
    log_path: str | None = None,
) -> tuple[str, subprocess.Popen]:
    """Launch the Pi0.5 VLA HTTP server in background."""
    if port == 0:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = int(s.getsockname()[1])

    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    cmd = [
        sys.executable,
        str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
        "--host",
        host,
        "--port",
        str(port),
    ]
    logger.info("vla server cmd: %s", " ".join(cmd))
    if log_path:
        log_f = open(log_path, "a")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.Popen(cmd, env=env)

    base_url = f"http://{host}:{port}"
    client = VLAClient(base_url)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            raise RuntimeError("vla server exited prematurely")
        try:
            if client.healthz():
                logger.info(
                    "vla server ready at %s after %.1fs",
                    base_url,
                    time.time() - t0,
                )
                return base_url, proc
        except Exception:
            pass
        time.sleep(2.0)
    _terminate_process(proc)
    raise RuntimeError("vla_server not ready after 300s")


def stop_vla_server(proc: subprocess.Popen | None, timeout: float = 10.0) -> None:
    """Stop a local VLA process if one was launched."""
    _terminate_process(proc, timeout=timeout)


@dataclass
class LiberoRuntimeHandle(RuntimeHandle):
    """LIBERO toolkit plus server processes owned by one run."""

    output_dir: Path
    env_proc: subprocess.Popen | None = None
    vla_proc: subprocess.Popen | None = None

    def close(self) -> None:
        try:
            self.toolkit.close()
        finally:
            if self.env_proc is not None:
                stop_env_server(self.env_proc, output_dir=self.output_dir)
            if self.vla_proc is not None:
                stop_vla_server(self.vla_proc)


class LiberoRuntimeProvider:
    """CLI/runtime adapter for the LIBERO environment."""

    name = "libero"

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--no-driver",
            action="store_true",
            help="Don't spawn driver; attach to existing output dir",
        )
        parser.add_argument(
            "--env-endpoint",
            default="127.0.0.1",
            help="Host of an existing env server to connect to; "
            "required with --no-driver.",
        )
        parser.add_argument(
            "--env-port",
            type=int,
            default=0,
            help="Port of an existing env server to connect to; "
            "required with --no-driver.",
        )
        parser.add_argument(
            "--vla-endpoint",
            default=None,
            help="Base URL of an existing vla_server "
            "(e.g. http://host:8000). If omitted with a "
            "spawned driver, a local vla_server is started; "
            "required with --no-driver.",
        )
        parser.add_argument(
            "--cuda-device",
            default=None,
            help="GPU device(s) to expose via CUDA_VISIBLE_DEVICES.",
        )
        parser.add_argument("--max-episode-steps", type=int, default=10000)
        parser.add_argument(
            "--libero-type",
            default=None,
            choices=["standard", "pro", "plus"],
            help="LIBERO variant (auto-routed from suite suffix if not set).",
        )
        parser.add_argument(
            "--suite", default=None, help="e.g. libero_object_task, libero_spatial_swap"
        )
        parser.add_argument("--task", type=int, default=None)
        parser.add_argument("--seed", type=int, default=0)

    def validate_args(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> None:
        if not args.suite:
            parser.error("--suite is required")
        if args.task is None:
            parser.error("--task is required")

    def recipe_tag(self, args: argparse.Namespace) -> str:
        return f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"

    def dashboard_state(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
    ) -> Any:
        from rpent.dashboard.state import State

        return State(
            run_id=f"{args.suite}/{output_dir.name}",
            name=self.recipe_tag(args),
            suite=args.suite,
            task=args.task,
            seed=args.seed,
            output_dir=str(output_dir),
            video_path=str(output_dir / "episode.mp4"),
        )

    def prompt_vars(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
        recipe_tag: str,
    ) -> dict[str, Any]:
        return {
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "output_dir": output_dir,
            "recipe_tag": recipe_tag,
        }

    def start(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
        dashboard: Any = None,
    ) -> LiberoRuntimeHandle:
        from robots.libero import get_toolkit

        suite = args.suite
        task = args.task
        seed = args.seed
        max_episode_steps = args.max_episode_steps
        vla_endpoint = args.vla_endpoint
        env_proc = None
        vla_proc = None

        try:
            if not args.no_driver:
                env_proc = start_env_server(
                    suite=suite,
                    task=task,
                    seed=seed,
                    output_dir=output_dir,
                    max_episode_steps=max_episode_steps,
                    cuda_device=args.cuda_device,
                    libero_type=args.libero_type or get_libero_type(),
                )
                if vla_endpoint is None:
                    vla_endpoint, vla_proc = start_vla_server(
                        cuda_device=args.cuda_device,
                        log_path=str(output_dir / "vla_server.log"),
                    )
            else:
                if args.env_port <= 0:
                    raise RuntimeError(
                        "--no-driver requires --env-port pointing at an existing driver"
                    )
                if vla_endpoint is None:
                    raise RuntimeError(
                        "--no-driver requires --vla-endpoint pointing at an existing vla_server"
                    )
                set_socket_endpoint(output_dir, args.env_endpoint, args.env_port)

            toolkit = get_toolkit(
                primitives_kwargs={
                    "env": LiberoEnvClient(
                        create_rpc_client(output_dir),
                        expected_meta={
                            "suite": suite,
                            "task": task,
                            "seed": seed,
                            "max_episode_steps": max_episode_steps,
                        },
                    ),
                    "model": VLAClient(vla_endpoint),
                },
                video_path=str(output_dir / "episode.mp4"),
                dashboard=dashboard,
            )
            return LiberoRuntimeHandle(
                toolkit=toolkit,
                output_dir=output_dir,
                env_proc=env_proc,
                vla_proc=vla_proc,
            )
        except Exception:
            if env_proc is not None:
                stop_env_server(env_proc, output_dir=output_dir)
            if vla_proc is not None:
                stop_vla_server(vla_proc)
            raise
