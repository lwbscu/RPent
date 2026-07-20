"""Runtime lifecycle for the BEHAVIOR/R1Pro plugin."""
from __future__ import annotations

import argparse
import contextvars
import json
import os
import queue
import shlex
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.prompt_bundle import mode_instructions
from robots.behavior.run_manifest import RunManifest, process_identity, redact_command
from robots.behavior.schemas import (
    CONTROL_MODES,
    DEFAULT_ACTION_CHUNK,
    FULL_TASK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOLS_MODE,
    VLA_CONTROL_MODES,
)
from robots.behavior.vla_client import BehaviorVLAClient
from rpent.envs.runtime import RuntimeHandle
from rpent.rpc_driver import create_rpc_client, set_socket_endpoint
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("behavior_runtime")

_ACTIVE_RUN_MANIFEST: contextvars.ContextVar[RunManifest | None] = (
    contextvars.ContextVar("behavior_run_manifest", default=None)
)


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return "<log unavailable>"


def _record_owned_process_group(proc: subprocess.Popen) -> int | None:
    """Mark the dedicated process group created for this exact Popen handle."""

    pid, pgid = process_identity(proc)
    if pid is None or pgid != pid or pgid == os.getpgrp():
        return None
    proc._rpent_owned_pgid = pgid
    try:
        proc._rpent_owned_sid = os.getsid(pid)
    except OSError:
        proc._rpent_owned_sid = None
    return pgid


def _owned_process_group_alive(proc: subprocess.Popen) -> bool:
    """Confirm a recorded dedicated PGID still has members in its session."""
    pgid = getattr(proc, "_rpent_owned_pgid", None)
    sid = getattr(proc, "_rpent_owned_sid", None)
    if not isinstance(pgid, int) or not isinstance(sid, int):
        return False
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            process_group = int(fields[2])
            session = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
        if process_group == pgid and session == sid:
            return True
    return False


def _terminate_process(proc: subprocess.Popen | None, *, timeout: float = 15.0) -> None:
    """Terminate an owned subprocess and its dedicated process group."""

    if proc is None:
        return
    pid, pgid = process_identity(proc)
    recorded_pgid = getattr(proc, "_rpent_owned_pgid", None)
    group_alive = _owned_process_group_alive(proc)
    leader_alive = proc.poll() is None
    owns_process_group = (
        isinstance(recorded_pgid, int)
        and recorded_pgid != os.getpgrp()
        and (
            (leader_alive and pgid == recorded_pgid)
            or (not leader_alive and group_alive and pgid in {None, recorded_pgid})
        )
    )
    if owns_process_group:
        try:
            os.killpg(recorded_pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        if proc.poll() is not None:
            return
        if pid is not None:
            logger.warning(
                "refusing group signal for pid=%s pgid=%s; using direct terminate",
                pid,
                pgid,
            )
        proc.terminate()
    deadline = time.monotonic() + float(timeout)
    if owns_process_group:
        while _owned_process_group_alive(proc) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _owned_process_group_alive(proc):
            try:
                os.killpg(recorded_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    else:
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)


def _runtime_env(args: argparse.Namespace) -> dict[str, str]:
    repo_root = get_repo_root()
    behavior_root = Path(args.behavior_repo).expanduser().resolve()
    env = os.environ.copy()
    env["RPENT_REPO_ROOT"] = str(repo_root)
    env["RPENT_RLINF_ROOT"] = str(behavior_root)
    env["RLINF_REPO_PATH"] = str(behavior_root)
    if getattr(args, "policy_checkpoint", None):
        env["PI05_CHECKPOINT_PATH"] = str(Path(args.policy_checkpoint).resolve())
    if args.cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    env.setdefault("OMNIGIBSON_HEADLESS", "1")
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    existing = env.get("PYTHONPATH")
    parts = [str(repo_root), str(behavior_root)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _pump_ready_events(
    proc: subprocess.Popen,
    log_path: Path,
    events: "queue.Queue[dict[str, Any]]",
) -> None:
    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            try:
                event = json.loads(line)
            except Exception:
                continue
            if isinstance(event, dict) and event.get("event") == "transport_ready":
                events.put(event)


def start_env_server(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> subprocess.Popen:
    script = get_repo_root() / "robots" / "behavior" / "env_server.py"
    command = [
        str(Path(args.behavior_python).expanduser().absolute()),
        str(script),
        "--suite",
        args.suite,
        "--task",
        str(args.task),
        "--task-name",
        args.task_name,
        "--activity-definition-id",
        str(args.activity_definition_id),
        "--activity-instance-id",
        str(args.activity_instance_id),
        "--activity-instance-dir",
        str(Path(args.activity_instance_dir).resolve()),
        "--scene-model",
        args.scene_model,
        "--seed",
        str(args.seed),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--control-mode",
        args.behavior_control_mode,
        "--output-dir",
        str(output_dir),
        "--transport-host",
        "127.0.0.1",
        "--transport-port",
        "0",
    ]
    if args.behavior_config:
        command.extend(["--config-path", str(Path(args.behavior_config).resolve())])
    log_path = output_dir / "env_server.log"
    logger.info(
        "BEHAVIOR env server cmd: %s",
        shlex.join(redact_command(command) or []),
    )
    proc = subprocess.Popen(
        command,
        cwd=Path(args.behavior_repo).resolve(),
        env=_runtime_env(args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    _record_owned_process_group(proc)
    manifest = _ACTIVE_RUN_MANIFEST.get()
    if manifest is not None:
        try:
            manifest.process_started("env", proc, command=command)
        except BaseException:
            _terminate_process(proc)
            raise
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    threading.Thread(
        target=_pump_ready_events,
        args=(proc, log_path, events),
        daemon=True,
    ).start()
    deadline = time.time() + args.env_ready_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            if manifest is not None:
                manifest.process_stopped("env", proc)
            raise RuntimeError(
                "BEHAVIOR env server exited before ready:\n" + _tail(log_path)
            )
        try:
            event = events.get(timeout=2.0)
        except queue.Empty:
            continue
        if event.get("kind") == "socket" and event.get("host") and event.get("port"):
            set_socket_endpoint(output_dir, str(event["host"]), int(event["port"]))
            if manifest is not None:
                try:
                    manifest.process_endpoint(
                        "env", host=str(event["host"]), port=int(event["port"])
                    )
                except BaseException:
                    _terminate_process(proc)
                    raise
            logger.info("BEHAVIOR env server ready at %s:%s", event["host"], event["port"])
            return proc
    _terminate_process(proc)
    if manifest is not None:
        manifest.process_stopped("env", proc)
    raise TimeoutError(
        f"BEHAVIOR env server not ready after {args.env_ready_timeout_s}s:\n"
        + _tail(log_path)
    )


def stop_env_server(proc: subprocess.Popen | None, *, output_dir: Path) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            create_rpc_client(output_dir).call("shutdown", timeout_s=30.0)
        except Exception:
            logger.exception("BEHAVIOR env graceful shutdown failed")
        try:
            proc.wait(timeout=120.0)
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
    # The Isaac/OG leader can exit before one of its descendants.  The exact
    # PGID+SID captured at launch remains the ownership boundary in that case.
    if _owned_process_group_alive(proc):
        _terminate_process(proc)


def start_vla_server(
    args: argparse.Namespace,
    *,
    output_dir: Path,
) -> tuple[str, subprocess.Popen]:
    host = "127.0.0.1"
    port = args.vla_port or _free_port(host)
    script = get_repo_root() / "robots" / "behavior" / "vla_server.py"
    command = [
        str(Path(args.behavior_python).expanduser().absolute()),
        str(script),
        "--host",
        host,
        "--port",
        str(port),
        "--checkpoint",
        str(Path(args.policy_checkpoint).resolve()),
        "--seed",
        str(args.seed),
    ]
    log_path = output_dir / "vla_server.log"
    logger.info(
        "BEHAVIOR VLA server cmd: %s",
        shlex.join(redact_command(command) or []),
    )
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=Path(args.behavior_repo).resolve(),
            env=_runtime_env(args),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _record_owned_process_group(proc)
    manifest = _ACTIVE_RUN_MANIFEST.get()
    if manifest is not None:
        try:
            manifest.process_started(
                "vla", proc, command=command, host=host, port=int(port)
            )
        except BaseException:
            _terminate_process(proc)
            raise
    base_url = f"http://{host}:{port}"
    client = BehaviorVLAClient(base_url)
    deadline = time.time() + args.vla_ready_timeout_s
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    "BEHAVIOR VLA server exited before ready:\n" + _tail(log_path)
                )
            try:
                metadata = client.healthz(timeout_ms=2000)
                if metadata.get("config_name") != "pi05_behavior":
                    raise RuntimeError(f"unexpected VLA metadata: {metadata!r}")
                logger.info("BEHAVIOR VLA server ready at %s", base_url)
                return base_url, proc
            except Exception:
                time.sleep(2.0)
        raise TimeoutError(
            f"BEHAVIOR VLA server not ready after {args.vla_ready_timeout_s}s:\n"
            + _tail(log_path)
        )
    except Exception:
        _terminate_process(proc)
        if manifest is not None:
            manifest.process_stopped("vla", proc)
        raise
    finally:
        client.close()


@dataclass
class BehaviorRuntimeHandle(RuntimeHandle):
    """Toolkit and only the processes spawned by this invocation."""

    output_dir: Path
    env_proc: subprocess.Popen | None = None
    vla_proc: subprocess.Popen | None = None
    manifest: RunManifest | None = None
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        failure: BaseException | None = None
        if self.manifest is not None:
            try:
                self.manifest.stopping()
            except BaseException as error:
                failure = error
        try:
            self.toolkit.close()
        except BaseException as error:
            if failure is None:
                failure = error
        try:
            stop_env_server(self.env_proc, output_dir=self.output_dir)
        except BaseException as error:
            if failure is None:
                failure = error
        finally:
            if self.manifest is not None:
                try:
                    self.manifest.process_stopped("env", self.env_proc)
                except BaseException as error:
                    if failure is None:
                        failure = error
        try:
            _terminate_process(self.vla_proc)
        except BaseException as error:
            if failure is None:
                failure = error
        finally:
            if self.manifest is not None and "vla" in self.manifest.data["processes"]:
                try:
                    self.manifest.process_stopped("vla", self.vla_proc)
                except BaseException as error:
                    if failure is None:
                        failure = error
        if self.manifest is not None:
            try:
                self.manifest.finish(
                    status="failed" if failure is not None else "stopped",
                    error=failure,
                )
            except BaseException as error:
                if failure is None:
                    failure = error
        if failure is not None:
            raise failure


class BehaviorRuntimeProvider:
    """CLI and process ownership adapter for BEHAVIOR."""

    name = "behavior"

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--suite", default="behavior_2025_challenge")
        parser.add_argument(
            "--behavior-control-mode",
            choices=CONTROL_MODES,
            default=FULL_TASK_VLA_MODE,
            help=(
                "BEHAVIOR control surface: full_task_vla runs the Pi0.5 "
                "baseline through run_full_task; planner_tools exposes only "
                "the eight env-side planner primitives and does not start a "
                "VLA server; pi0_pick_vla exposes only the local pi0_pick loop."
            ),
        )
        parser.add_argument("--task", type=int, default=0)
        parser.add_argument("--task-name", default="turning_on_radio")
        parser.add_argument("--activity-definition-id", type=int, default=0)
        parser.add_argument("--activity-instance-id", type=int, default=242)
        parser.add_argument("--activity-instance-dir", default=None)
        parser.add_argument("--scene-model", default="house_double_floor_lower")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--max-episode-steps", type=int, default=24756)
        parser.add_argument("--behavior-repo", default=None)
        parser.add_argument("--behavior-python", default=None)
        parser.add_argument("--behavior-config", default=None)
        parser.add_argument("--policy-checkpoint", default=None)
        parser.add_argument(
            "--behavior-pi0-pick-hand",
            choices=("left", "right"),
            default="right",
            help="Selected hand passed to the isolated pi0_pick tool prompt.",
        )
        parser.add_argument(
            "--behavior-pi0-pick-instruction",
            default=(
                "Grasp the radio securely with the selected hand and stop as "
                "soon as the local grasp is achieved."
            ),
            help="Local VLA instruction passed to the isolated pi0_pick tool.",
        )
        parser.add_argument(
            "--behavior-pi0-pick-max-chunks",
            type=int,
            default=24,
            help="Bounded number of Pi0.5 chunks passed to pi0_pick.",
        )
        parser.add_argument("--cuda-device", default=None)
        parser.add_argument("--no-driver", action="store_true")
        parser.add_argument("--env-endpoint", default="127.0.0.1")
        parser.add_argument("--env-port", type=int, default=0)
        parser.add_argument("--vla-endpoint", default=None)
        parser.add_argument("--vla-port", type=int, default=0)
        parser.add_argument("--env-ready-timeout-s", type=int, default=1800)
        parser.add_argument("--vla-ready-timeout-s", type=int, default=1800)

    def _resolve_paths(self, args: argparse.Namespace) -> None:
        if not args.behavior_repo:
            args.behavior_repo = os.environ.get("RPENT_RLINF_ROOT") or os.environ.get(
                "RLINF_REPO_PATH"
            )
        if not args.behavior_repo:
            args.behavior_repo = str(get_repo_root().parent / "RLinf_agentic_push")
        root = Path(args.behavior_repo).expanduser().resolve()
        args.behavior_repo = str(root)
        if not args.behavior_python:
            args.behavior_python = str(root / ".venv-behavior" / "bin" / "python")
        if not args.activity_instance_dir:
            args.activity_instance_dir = str(
                root
                / ".venv-behavior"
                / "BEHAVIOR-1K"
                / "datasets"
                / "2025-challenge-task-instances"
                / "scenes"
                / args.scene_model
                / "json"
                / f"{args.scene_model}_task_{args.task_name}_instances"
            )
        if not args.policy_checkpoint:
            args.policy_checkpoint = os.environ.get("PI05_CHECKPOINT_PATH")

    def validate_args(
        self,
        args: argparse.Namespace,
        parser: argparse.ArgumentParser,
    ) -> None:
        self._resolve_paths(args)
        if args.max_episode_steps <= 0:
            parser.error("--max-episode-steps must be positive")
        if args.env_ready_timeout_s <= 0 or args.vla_ready_timeout_s <= 0:
            parser.error("runtime ready timeouts must be positive")
        if args.behavior_control_mode not in CONTROL_MODES:
            parser.error(
                "--behavior-control-mode must be one of " + ", ".join(CONTROL_MODES)
            )
        if (
            args.behavior_control_mode == PI0_PICK_VLA_MODE
            and not str(args.behavior_pi0_pick_instruction).strip()
        ):
            parser.error("--behavior-pi0-pick-instruction must be non-empty")
        if (
            args.behavior_control_mode == PI0_PICK_VLA_MODE
            and args.behavior_pi0_pick_max_chunks <= 0
        ):
            parser.error("--behavior-pi0-pick-max-chunks must be positive")
        if args.behavior_control_mode in VLA_CONTROL_MODES:
            checkpoint = Path(args.policy_checkpoint or "")
            required_checkpoint_files = [
                checkpoint / "model.safetensors",
                checkpoint
                / "assets"
                / "behavior-1k"
                / "2025-challenge-demos"
                / "norm_stats.json",
            ]
            missing = [
                str(path) for path in required_checkpoint_files if not path.is_file()
            ]
            if missing:
                parser.error(
                    "BEHAVIOR Pi0.5 checkpoint is incomplete: " + ", ".join(missing)
                )
        if args.no_driver:
            if args.env_port <= 0:
                parser.error("--no-driver requires --env-port")
            if (
                args.behavior_control_mode in VLA_CONTROL_MODES
                and not args.vla_endpoint
            ):
                parser.error("--no-driver requires --vla-endpoint")
            return

        root = Path(args.behavior_repo)
        python = Path(args.behavior_python)
        instance_dir = Path(args.activity_instance_dir)
        if not root.is_dir():
            parser.error(f"--behavior-repo does not exist: {root}")
        if not python.is_file():
            parser.error(f"--behavior-python does not exist: {python}")
        if not instance_dir.is_dir():
            parser.error(f"--activity-instance-dir does not exist: {instance_dir}")
        instance_matches = list(
            instance_dir.glob(
                f"*_{args.activity_instance_id}_template-tro_state.json"
            )
        )
        if len(instance_matches) != 1:
            parser.error(
                "expected exactly one tro_state file for activity instance "
                f"{args.activity_instance_id} in {instance_dir}, got {len(instance_matches)}"
            )

    def recipe_tag(self, args: argparse.Namespace) -> str:
        return f"behavior_t{args.task}_i{args.activity_instance_id}_s{args.seed}"

    def dashboard_state(self, args: argparse.Namespace, *, output_dir: Path) -> Any:
        from rpent.dashboard.state import State

        return State(
            run_id=f"behavior/{output_dir.name}",
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
            "task_name": args.task_name,
            "activity_definition_id": args.activity_definition_id,
            "activity_instance_id": args.activity_instance_id,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
            "control_mode": args.behavior_control_mode,
            "output_dir": output_dir,
            "recipe_tag": recipe_tag,
            "behavior_control_mode": args.behavior_control_mode,
            "behavior_pi0_pick_hand": args.behavior_pi0_pick_hand,
            "behavior_pi0_pick_instruction": args.behavior_pi0_pick_instruction,
            "behavior_pi0_pick_max_chunks": args.behavior_pi0_pick_max_chunks,
            **mode_instructions(
                args.behavior_control_mode,
                pi0_hand=args.behavior_pi0_pick_hand,
                pi0_instruction=args.behavior_pi0_pick_instruction,
                pi0_max_chunks=args.behavior_pi0_pick_max_chunks,
            ),
        }

    def _expected_meta(self, args: argparse.Namespace) -> dict[str, Any]:
        return {
            "control_mode": args.behavior_control_mode,
            "suite": args.suite,
            "task": args.task,
            "task_name": args.task_name,
            "activity_definition_id": args.activity_definition_id,
            "activity_instance_id": args.activity_instance_id,
            "activity_instance_dir": str(Path(args.activity_instance_dir).resolve()),
            "scene_model": args.scene_model,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        }

    def start(
        self,
        args: argparse.Namespace,
        *,
        output_dir: Path,
        dashboard: Any = None,
    ) -> BehaviorRuntimeHandle:
        from robots.behavior import get_toolkit

        self._resolve_paths(args)
        output_dir = Path(output_dir)
        manifest = RunManifest.start(output_dir, args, repo_root=get_repo_root())
        manifest_token = _ACTIVE_RUN_MANIFEST.set(manifest)
        env_proc = None
        vla_proc = None
        vla_endpoint = args.vla_endpoint
        try:
            if args.no_driver:
                set_socket_endpoint(output_dir, args.env_endpoint, args.env_port)
            else:
                env_proc = start_env_server(args, output_dir=output_dir)
                if (
                    args.behavior_control_mode in VLA_CONTROL_MODES
                    and not vla_endpoint
                ):
                    vla_endpoint, vla_proc = start_vla_server(
                        args,
                        output_dir=output_dir,
                    )
            env = BehaviorEnvClient(
                create_rpc_client(output_dir),
                expected_meta=self._expected_meta(args),
            )
            if args.behavior_control_mode in VLA_CONTROL_MODES:
                model = BehaviorVLAClient(vla_endpoint)
                if args.no_driver:
                    model.wait_for_healthz(timeout_s=30.0)
            if args.behavior_control_mode == FULL_TASK_VLA_MODE:
                toolkit = get_toolkit(
                    control_mode=args.behavior_control_mode,
                    primitives_kwargs={
                        "env": env,
                        "model": model,
                        "max_episode_steps": args.max_episode_steps,
                        "action_horizon": DEFAULT_ACTION_CHUNK,
                        "output_dir": output_dir,
                        "video_path": output_dir / "episode.mp4",
                    },
                    dashboard=dashboard,
                )
            elif args.behavior_control_mode == PI0_PICK_VLA_MODE:
                initial_observation, initial_info = env.reset()
                toolkit = get_toolkit(
                    control_mode=args.behavior_control_mode,
                    primitives_kwargs={
                        "env": env,
                        "model": model,
                        "max_episode_steps": args.max_episode_steps,
                        "action_horizon": DEFAULT_ACTION_CHUNK,
                        "output_dir": output_dir,
                        "video_path": output_dir / "episode.mp4",
                        "initial_observation": initial_observation,
                        "initial_info": initial_info,
                    },
                    dashboard=dashboard,
                )
            elif args.behavior_control_mode == PLANNER_TOOLS_MODE:
                env.reset()
                toolkit = get_toolkit(
                    control_mode=args.behavior_control_mode,
                    planner_client=env,
                    dashboard=dashboard,
                )
            else:
                raise ValueError(
                    f"unknown BEHAVIOR control mode: {args.behavior_control_mode}"
                )
            manifest.running()
            return BehaviorRuntimeHandle(
                toolkit=toolkit,
                output_dir=output_dir,
                env_proc=env_proc,
                vla_proc=vla_proc,
                manifest=manifest,
            )
        except BaseException as error:
            try:
                stop_env_server(env_proc, output_dir=output_dir)
            except BaseException:
                logger.exception("BEHAVIOR env cleanup after start failure failed")
            finally:
                try:
                    manifest.process_stopped("env", env_proc)
                except BaseException:
                    logger.exception("BEHAVIOR env manifest stop update failed")
            try:
                _terminate_process(vla_proc)
            except BaseException:
                logger.exception("BEHAVIOR VLA cleanup after start failure failed")
            finally:
                if "vla" in manifest.data["processes"]:
                    try:
                        manifest.process_stopped("vla", vla_proc)
                    except BaseException:
                        logger.exception("BEHAVIOR VLA manifest stop update failed")
            try:
                manifest.finish(status="failed", error=error)
            except BaseException:
                logger.exception("BEHAVIOR failed-session manifest update failed")
            raise
        finally:
            _ACTIVE_RUN_MANIFEST.reset(manifest_token)


__all__ = ["BehaviorRuntimeProvider", "BehaviorRuntimeHandle"]
