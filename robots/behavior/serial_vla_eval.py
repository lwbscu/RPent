"""Pure-Pi0 BEHAVIOR Eval baseline with no language-model planner.

The public benchmark action in this runner is always exactly::

    pi0_nav_pick(instruction=TaskSpec.task_language, chunks=80)

There is no tool-call quota. Time, environment-step, real termination, safety,
and infrastructure boundaries remain authoritative. A raw official success
ends physical execution on that exact environment step, so the current
80-chunk request may return with precise partial accounting and no next call.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from robots.behavior.dashboard_sink import FileDashboardSink
from robots.behavior.env_client import BehaviorEnvClient
from robots.behavior.policy_checkpoint import (
    SHARED_POLICY_CHECKPOINT_PATH,
    validate_policy_checkpoint,
)
from robots.behavior.runtime import (
    BEHAVIOR_NATIVE_ENV_SEED,
    CampaignRuntimeIsolation,
    _expected_shared_policy_checkpoint_binding,
    _managed_env_rpc_client,
    _task_language_from_reset,
    _terminate_process,
    start_env_server,
    start_vla_server,
    stop_env_server,
    validate_campaign_runtime_isolation,
)
from robots.behavior.serial_eval import _checkout_identity, _gpu_lock_path
from robots.behavior.source_snapshot import validate_source_snapshot
from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    BehaviorTaskSpec,
    get_task_spec,
)
from robots.behavior.tools import BehaviorPrimitives
from robots.behavior.vla_client import BehaviorVLAClient

CHUNKS_PER_CALL = 80
ACTION_STEPS_PER_CHUNK = 32
ACTION_STEPS_PER_CALL = CHUNKS_PER_CALL * ACTION_STEPS_PER_CHUNK
ACTION_DEADLINE_S = 6900
CLEANUP_DEADLINE_S = 7080
INSTANCE_TIMEOUT_S = 7200
WINDOW_COMPLETION_RESERVE_S = 900
MAX_EPISODE_STEPS = 24_756

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUN_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_EXPECTED_RUN_NONCE_ENV = "RPENT_BEHAVIOR_EXPECTED_RUN_NONCE"
_SOURCE_BINDING_FILENAME = "source_snapshot.json"
_CAMERAS = ("head", "left_wrist", "right_wrist")
_OFFICIAL_SUCCESS_SOURCE = "behavior_action_trace"
_OFFICIAL_SUCCESS_FIELD_PATH = "info_done.success"


class Pi0WindowExecutor(Protocol):
    """The only physical capability used by :func:`run_vla_window_loop`."""

    total_env_steps: int

    def pi0_nav_pick(self, *, instruction: str, chunks: int) -> Mapping[str, Any]:
        """Execute one exact-N policy call."""


class _InstanceDeadline(BaseException):
    """Escape ordinary primitive error handlers on a parent cleanup signal.

    ``BehaviorPrimitives.pi0_nav_pick`` intentionally converts ordinary
    ``Exception`` instances into tool results.  A supervisor SIGTERM is not a
    tool failure, though: it must unwind the whole instance immediately so the
    child can seal its state binding and release its managed environment.
    """


@dataclass(frozen=True)
class SourceSnapshot:
    """One immutable source identity shared by both Eval cohorts."""

    root: Path
    binding: dict[str, Any]
    binding_sha256: str
    tree_sha256: str

    def public_binding(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "binding_file": str(self.root / _SOURCE_BINDING_FILENAME),
            "binding_sha256": self.binding_sha256,
            "tree_sha256": self.tree_sha256,
            "binding": self.binding,
        }


@dataclass(frozen=True)
class VLAEvalEntry:
    """One task-scoped public Eval identity."""

    task_name: str
    public_seed: int
    activity_instance_id: int
    output_dir: Path
    cuda_device: str

    def __post_init__(self) -> None:
        spec = get_task_spec(self.task_name)
        expected = spec.instance_for_public_seed(self.public_seed, phase="eval")
        if self.activity_instance_id != expected:
            raise ValueError(
                f"{self.task_name} s{self.public_seed} requires instance "
                f"{expected}, got {self.activity_instance_id}"
            )
        if self.cuda_device != "6":
            raise ValueError("pure-VLA BEHAVIOR Eval is restricted to GPU6")


@dataclass(frozen=True)
class WindowLoopResult:
    """Terminal facts from one no-LLM baseline instance."""

    task_success: bool
    official_success_receipt: dict[str, Any] | None
    tool_calls: int
    vla_chunks: int
    total_env_steps: int
    stopped_reason: str
    timed_out: bool
    infrastructure_error: str | None
    last_result: dict[str, Any] | None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_lexical_executable_path(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label} must be an existing executable file")
    if not os.access(path, os.X_OK):
        raise ValueError(f"{label} must be executable")
    return path


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_source_snapshot(
    root: str | Path,
    *,
    expected_binding_sha256: str,
) -> SourceSnapshot:
    """Load the canonical source binding without trusting the dirty checkout."""

    if not _SHA256_RE.fullmatch(str(expected_binding_sha256)):
        raise ValueError("source snapshot binding SHA256 must be 64 lowercase hex")
    resolved = Path(root).expanduser().resolve(strict=True)
    validated = validate_source_snapshot(
        resolved,
        expected_binding_sha256=str(expected_binding_sha256),
    )
    return SourceSnapshot(
        root=Path(validated.snapshot_root).resolve(),
        binding=dict(validated.as_dict()),
        binding_sha256=validated.binding_sha256,
        tree_sha256=validated.tree_sha256,
    )


def _assert_running_from_source_snapshot(source: SourceSnapshot) -> None:
    module_path = Path(__file__).resolve()
    try:
        module_path.relative_to(source.root)
    except ValueError as error:
        raise RuntimeError(
            "formal pure-VLA Eval parent must execute from the validated "
            "source snapshot"
        ) from error


def build_vla_eval_plan(
    *,
    task_name: str,
    public_seeds: Iterable[int] | None,
    output_root: str | Path,
    cuda_device: str = "6",
) -> tuple[VLAEvalEntry, ...]:
    """Resolve task-local s10-s19 identities without accepting native IDs."""

    spec = get_task_spec(task_name)
    seeds = (
        tuple(spec.eval_public_seeds)
        if public_seeds is None
        else tuple(int(seed) for seed in public_seeds)
    )
    if not seeds:
        raise ValueError("at least one Eval public seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Eval public seeds must not contain duplicates")
    root = Path(output_root).expanduser().resolve()
    entries: list[VLAEvalEntry] = []
    for public_seed in seeds:
        native = spec.instance_for_public_seed(public_seed, phase="eval")
        entries.append(
            VLAEvalEntry(
                task_name=spec.task_name,
                public_seed=public_seed,
                activity_instance_id=native,
                output_dir=root / spec.tag(public_seed),
                cuda_device=str(cuda_device),
            )
        )
    return tuple(entries)


def _validate_complete_window(result: Mapping[str, Any]) -> None:
    """Validate exact-80 completion or a receipt-bound official-success stop."""

    if result.get("name") != "pi0_nav_pick":
        raise RuntimeError("baseline primitive returned the wrong tool name")
    if result.get("requested_chunks") != CHUNKS_PER_CALL:
        raise RuntimeError("baseline primitive changed the requested chunk count")
    exact = result.get("exact_requested_chunks_completed")
    official_success = bool(
        result.get("task_success") is True
        and result.get("primitive_success") is True
        and result.get("stop_reason") == "official_task_success"
        and not result.get("error")
        and isinstance(result.get("official_success_receipt"), Mapping)
    )
    if exact is not True and official_success:
        chunks_used = result.get("chunks_used")
        full_chunks = result.get("full_chunks_executed")
        vla_env_steps = result.get("vla_env_steps_used")
        if (
            isinstance(chunks_used, bool)
            or not isinstance(chunks_used, int)
            or not 1 <= chunks_used <= CHUNKS_PER_CALL
            or isinstance(full_chunks, bool)
            or not isinstance(full_chunks, int)
            or full_chunks not in {chunks_used - 1, chunks_used}
            or isinstance(vla_env_steps, bool)
            or not isinstance(vla_env_steps, int)
            or not (
                (
                    full_chunks == chunks_used
                    and vla_env_steps == full_chunks * ACTION_STEPS_PER_CHUNK
                )
                or (
                    full_chunks == chunks_used - 1
                    and full_chunks * ACTION_STEPS_PER_CHUNK + 1
                    <= vla_env_steps
                    <= chunks_used * ACTION_STEPS_PER_CHUNK - 1
                )
            )
            or result.get("action_horizon") != ACTION_STEPS_PER_CHUNK
        ):
            raise RuntimeError(
                "pi0_nav_pick official-success partial accounting is invalid"
            )
        return
    exceptional = bool(
        result.get("terminated") is True
        or result.get("truncated") is True
        or result.get("error")
    )
    if exact is not True:
        if not exceptional:
            raise RuntimeError(
                "pi0_nav_pick returned a partial window without a real "
                "termination, safety, or infrastructure exception"
            )
        return
    expected = {
        "chunks_used": CHUNKS_PER_CALL,
        "full_chunks_executed": CHUNKS_PER_CALL,
        "vla_env_steps_used": ACTION_STEPS_PER_CALL,
        "action_horizon": ACTION_STEPS_PER_CHUNK,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"pi0_nav_pick exact-window mismatch: {mismatches!r}")


def _bind_result_to_frozen_instance_state(
    payload: dict[str, Any],
    *,
    expected_state_sha256: str,
    parent_observed_state_sha256: str | None = None,
) -> None:
    """Bind parent-observed frozen state without revoking raw trace success."""

    recorded_state_sha256 = payload.get("instance_state_sha256")
    if recorded_state_sha256 is None:
        if parent_observed_state_sha256 == expected_state_sha256:
            payload["instance_state_sha256"] = parent_observed_state_sha256
            payload["instance_state_binding"] = {
                "valid": True,
                "source": "parent_postrun_native_state",
                "expected_sha256": expected_state_sha256,
                "observed_sha256": parent_observed_state_sha256,
            }
            return
        if payload.get("task_success") is not True:
            payload["task_success"] = None
            payload["success"] = None
        _append_infrastructure_error(
            payload,
            "instance state SHA-256 is missing and parent post-run verification failed",
        )
        payload["instance_state_binding"] = {
            "valid": False,
            "source": "parent_postrun_native_state",
            "expected_sha256": expected_state_sha256,
            "observed_sha256": parent_observed_state_sha256,
        }
        return
    if (
        recorded_state_sha256 == expected_state_sha256
        and parent_observed_state_sha256 in {None, expected_state_sha256}
    ):
        payload["instance_state_binding"] = {
            "valid": True,
            "source": (
                "child_prelaunch_and_parent_postrun_native_state"
                if parent_observed_state_sha256 is not None
                else "child_prelaunch_frozen_native_state"
            ),
            "expected_sha256": expected_state_sha256,
            "observed_sha256": (parent_observed_state_sha256 or recorded_state_sha256),
        }
        return
    if payload.get("task_success") is not True:
        payload["task_success"] = None
        payload["success"] = None
    _append_infrastructure_error(
        payload,
        "instance state SHA-256 differs from the frozen parent binding",
    )
    payload["expected_instance_state_sha256"] = expected_state_sha256
    payload["instance_state_binding"] = {
        "valid": False,
        "source": "child_prelaunch_frozen_native_state",
        "expected_sha256": expected_state_sha256,
        "observed_sha256": parent_observed_state_sha256 or recorded_state_sha256,
    }


def run_vla_window_loop(
    primitives: Pi0WindowExecutor,
    *,
    instruction: str,
    deadline_monotonic: float,
    trace_path: Path,
    dashboard: FileDashboardSink | None = None,
    clock: Callable[[], float] = time.monotonic,
    completion_reserve_s: float = WINDOW_COMPLETION_RESERVE_S,
) -> WindowLoopResult:
    """Call only exact 80-chunk Pi0 windows until success or a trusted boundary."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty")
    if completion_reserve_s < 0:
        raise ValueError("completion_reserve_s must be non-negative")
    tool_calls = 0
    vla_chunks = 0
    task_success = False
    receipt: dict[str, Any] | None = None
    stopped_reason = "action_deadline_exhausted"
    infrastructure_error: str | None = None
    last_result: dict[str, Any] | None = None

    while clock() + completion_reserve_s <= deadline_monotonic:
        remaining_steps = MAX_EPISODE_STEPS - int(primitives.total_env_steps)
        if remaining_steps < ACTION_STEPS_PER_CALL:
            stopped_reason = "episode_step_budget_exhausted"
            break
        arguments = {"instruction": instruction, "chunks": CHUNKS_PER_CALL}
        step = tool_calls + 1
        if dashboard is not None:
            dashboard.on_tool_start("pi0_nav_pick", arguments)
        started = clock()
        try:
            raw_result = primitives.pi0_nav_pick(**arguments)
            if not isinstance(raw_result, Mapping):
                raise RuntimeError("pi0_nav_pick result must be a mapping")
            result = dict(raw_result)
            result["official_success_source"] = _OFFICIAL_SUCCESS_SOURCE
            result["official_success_field_path"] = _OFFICIAL_SUCCESS_FIELD_PATH
            raw_result_receipt = result.get("official_success_receipt")
            if isinstance(raw_result_receipt, Mapping):
                normalized_receipt = dict(raw_result_receipt)
                normalized_receipt.pop("official_success_source", None)
                normalized_receipt.pop("official_success_field_path", None)
                normalized_receipt["runtime_source"] = normalized_receipt.get("source")
                normalized_receipt["source"] = _OFFICIAL_SUCCESS_SOURCE
                normalized_receipt["field_path"] = _OFFICIAL_SUCCESS_FIELD_PATH
                result["official_success_receipt"] = normalized_receipt
            _validate_complete_window(result)
        except BaseException as error:
            if isinstance(error, _InstanceDeadline):
                raise
            elapsed = max(0.0, clock() - started)
            infrastructure_error = f"{type(error).__name__}: {error}"
            result = {
                "name": "pi0_nav_pick",
                "primitive_success": False,
                "task_success": False,
                "official_success_source": _OFFICIAL_SUCCESS_SOURCE,
                "official_success_field_path": _OFFICIAL_SUCCESS_FIELD_PATH,
                "requested_chunks": CHUNKS_PER_CALL,
                "error": infrastructure_error,
            }
            tool_calls = step
            record = {
                "step": step,
                "tool": "pi0_nav_pick",
                "arguments": arguments,
                "result": result,
                "elapsed_s": round(elapsed, 3),
            }
            _append_jsonl(trace_path, record)
            if dashboard is not None:
                dashboard.on_tool_result("pi0_nav_pick", result)
                dashboard.on_usage(inp=0, out=0, tool_calls=tool_calls)
            last_result = result
            stopped_reason = "infrastructure_error"
            break

        elapsed = max(0.0, clock() - started)
        tool_calls = step
        vla_chunks += int(result.get("full_chunks_executed", 0) or 0)
        record = {
            "step": step,
            "tool": "pi0_nav_pick",
            "arguments": arguments,
            "result": result,
            "elapsed_s": round(elapsed, 3),
        }
        _append_jsonl(trace_path, record)
        if dashboard is not None:
            dashboard.on_tool_result("pi0_nav_pick", result)
            dashboard.on_usage(inp=0, out=0, tool_calls=tool_calls)
        last_result = result
        if str(result.get("error", "")).startswith("_InstanceDeadline:"):
            raise _InstanceDeadline("instance cleanup deadline reached")

        if result.get("task_success") is True:
            task_success = True
            raw_receipt = result.get("official_success_receipt")
            if isinstance(raw_receipt, dict):
                receipt = dict(raw_receipt)
                receipt.pop("official_success_source", None)
                receipt.pop("official_success_field_path", None)
                receipt.setdefault("runtime_source", receipt.get("source"))
                receipt["source"] = _OFFICIAL_SUCCESS_SOURCE
                receipt["field_path"] = _OFFICIAL_SUCCESS_FIELD_PATH
            stopped_reason = "official_task_success"
            break
        if result.get("terminated") is True:
            stopped_reason = "environment_terminated"
            break
        if result.get("truncated") is True:
            stopped_reason = "environment_truncated"
            break
        if result.get("primitive_success") is not True:
            stopped_reason = "runtime_precondition_or_infrastructure_failure"
            infrastructure_error = str(
                result.get("error")
                or result.get("stop_reason")
                or "pi0_nav_pick did not complete normally"
            )
            break

    return WindowLoopResult(
        task_success=task_success,
        official_success_receipt=receipt,
        tool_calls=tool_calls,
        vla_chunks=vla_chunks,
        total_env_steps=int(primitives.total_env_steps),
        stopped_reason=stopped_reason,
        timed_out=stopped_reason == "action_deadline_exhausted",
        infrastructure_error=infrastructure_error,
        last_result=last_result,
    )


def validate_pure_vla_tool_trace(path: Path) -> dict[str, int]:
    """Validate that the baseline exposed no action tool other than Pi0."""

    tool_calls = 0
    chunks = 0
    success_seen = False
    if path.is_symlink():
        raise ValueError("pure-VLA tool trace must not be a symlink")
    if not path.is_file():
        raise ValueError("pure-VLA tool trace is missing")
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid tool trace JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict) or record.get("tool") != "pi0_nav_pick":
                raise ValueError("pure-VLA trace contains a non-Pi0 tool")
            if success_seen:
                raise ValueError(
                    "pure-VLA trace contains a call after official task success"
                )
            arguments = record.get("arguments")
            if (
                not isinstance(arguments, dict)
                or set(arguments) != {"instruction", "chunks"}
                or arguments.get("chunks") != CHUNKS_PER_CALL
            ):
                raise ValueError("pure-VLA trace contains an invalid Pi0 request")
            tool_calls += 1
            result = record.get("result")
            if isinstance(result, dict):
                _validate_complete_window(result)
                chunks += int(result.get("full_chunks_executed", 0) or 0)
                success_seen = result.get("task_success") is True
    return {"tool_calls": tool_calls, "vla_chunks": chunks}


def _raw_success_from_action_trace(
    path: Path,
    *,
    expected_run_nonce: str | None = None,
) -> dict[str, Any] | None:
    """Return a trace-bound raw-success receipt, including forced-cleanup runs."""

    if path.is_symlink():
        raise ValueError("behavior action trace must not be a symlink")
    if not path.is_file():
        raise ValueError("behavior action trace is missing")
    first_step: int | None = None
    last_step: int | None = None
    final_trace_step: int | None = None
    binding_count = 0
    bound_run_nonce: str | None = None
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid action trace JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError("action trace records must be objects")
            if record.get("event") == "rpent_run_binding":
                binding_count += 1
                bound_run_nonce = record.get("run_nonce")
                if (
                    record.get("attempt_index") != 1
                    or not isinstance(bound_run_nonce, str)
                    or _RUN_NONCE_RE.fullmatch(bound_run_nonce) is None
                ):
                    raise ValueError("action trace has an invalid run nonce binding")
            step = record.get("step")
            info_done = record.get("info_done")
            if isinstance(step, int) and not isinstance(step, bool):
                final_trace_step = step
            if (
                isinstance(step, int)
                and not isinstance(step, bool)
                and isinstance(info_done, dict)
                and info_done.get("success") is True
            ):
                if first_step is None:
                    first_step = step
                last_step = step
                count += 1
    if expected_run_nonce is not None:
        if _RUN_NONCE_RE.fullmatch(expected_run_nonce) is None:
            raise ValueError("expected run nonce is invalid")
        if binding_count != 1:
            raise ValueError("action trace requires exactly one run nonce binding")
        if bound_run_nonce != expected_run_nonce:
            raise ValueError("action trace run nonce differs from supervisor binding")
    if first_step is None:
        return None
    if first_step != final_trace_step:
        raise ValueError("action trace contains an action after official task success")
    return {
        "receipt_type": "forensic_action_trace_receipt",
        "source": _OFFICIAL_SUCCESS_SOURCE,
        "field_path": _OFFICIAL_SUCCESS_FIELD_PATH,
        "first_success_step": first_step,
        "last_success_step": last_step,
        "success_count": count,
        "run_nonce": bound_run_nonce,
        "action_trace_sha256": _sha256(path),
    }


def _png_bytes(value: Any) -> bytes | None:
    try:
        import imageio.v2 as imageio
        import numpy as np

        image = np.asarray(value)
        if image.ndim != 3 or image.shape[-1] not in {3, 4}:
            return None
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        buffer = io.BytesIO()
        imageio.imwrite(buffer, image, format="png")
        return buffer.getvalue()
    except Exception:
        return None


def _emit_internal_dashboard_frames(
    dashboard: FileDashboardSink,
    observation: Mapping[str, Any],
    *,
    env_step: int,
) -> None:
    """Publish internal policy inputs without creating an ``observe`` tool."""

    frames: dict[str, Any] = {}
    main = observation.get("main_images")
    if main is not None:
        frames["head"] = main
    wrists = observation.get("wrist_images")
    try:
        if wrists is not None and len(wrists) == 2:
            frames["left_wrist"] = wrists[0]
            frames["right_wrist"] = wrists[1]
    except (TypeError, IndexError):
        pass
    for camera in _CAMERAS:
        encoded = _png_bytes(frames.get(camera))
        if encoded is not None:
            dashboard.on_frame(camera, encoded, env_step=env_step)


def _expected_env_meta(
    spec: BehaviorTaskSpec,
    *,
    public_seed: int,
    expected_run_nonce: str,
) -> dict[str, Any]:
    return {
        "suite": "behavior_2025_challenge",
        "task": spec.task_index,
        "task_name": spec.task_name,
        "public_seed": public_seed,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "run_nonce": expected_run_nonce,
    }


def _validated_runtime_success_receipt(
    value: Any,
    *,
    expected_run_nonce: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("runtime official-success receipt is missing")
    receipt = dict(value)
    if (
        receipt.get("run_nonce") != expected_run_nonce
        or receipt.get("attempt_index") != 1
        or not isinstance(receipt.get("attempt_nonce"), str)
        or _RUN_NONCE_RE.fullmatch(receipt["attempt_nonce"]) is None
        or not isinstance(receipt.get("raw_done"), Mapping)
        or receipt["raw_done"].get("success") is not True
        or receipt.get("runtime_source") != 'info["done"]["success"]'
        or receipt.get("source") != _OFFICIAL_SUCCESS_SOURCE
        or receipt.get("field_path") != _OFFICIAL_SUCCESS_FIELD_PATH
        or _SHA256_RE.fullmatch(str(receipt.get("receipt_sha256") or "")) is None
    ):
        raise ValueError("runtime official-success receipt binding is invalid")
    original = dict(receipt)
    expected_sha256 = str(original.pop("receipt_sha256"))
    original["source"] = original.pop("runtime_source")
    original.pop("field_path", None)
    actual_sha256 = hashlib.sha256(
        json.dumps(
            original,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("runtime official-success receipt SHA-256 is invalid")
    return receipt


def _instance_dir(behavior_repo: Path, spec: BehaviorTaskSpec) -> Path:
    return (
        behavior_repo
        / ".venv-behavior"
        / "BEHAVIOR-1K"
        / "datasets"
        / "2025-challenge-task-instances"
        / "scenes"
        / spec.scene_model
        / "json"
        / spec.state_dir_name
    )


def _runtime_args(
    *,
    entry: VLAEvalEntry,
    behavior_repo: Path,
    behavior_python: Path,
    checkpoint: Path,
    checkpoint_binding: dict[str, Any],
    vla_endpoint: str | None,
    source: SourceSnapshot,
    runtime_isolation: CampaignRuntimeIsolation,
) -> argparse.Namespace:
    spec = get_task_spec(entry.task_name)
    return argparse.Namespace(
        suite="behavior_2025_challenge",
        task=spec.task_index,
        task_name=spec.task_name,
        activity_definition_id=spec.activity_definition_id,
        activity_instance_id=entry.activity_instance_id,
        activity_instance_dir=str(_instance_dir(behavior_repo, spec)),
        scene_model=spec.scene_model,
        seed=BEHAVIOR_NATIVE_ENV_SEED,
        public_seed=entry.public_seed,
        behavior_attempt_index=1,
        max_episode_steps=MAX_EPISODE_STEPS,
        behavior_config=None,
        behavior_repo=str(behavior_repo),
        behavior_python=str(behavior_python),
        policy_checkpoint=str(checkpoint),
        cuda_device=entry.cuda_device,
        env_ready_timeout_s=1800,
        vla_ready_timeout_s=1800,
        vla_port=0,
        vla_endpoint=vla_endpoint,
        behavior_controller_mode="pi0_nav_pick_only",
        pure_vla_baseline=True,
        _behavior_policy_checkpoint_binding=checkpoint_binding,
        behavior_source_snapshot_root=str(source.root),
        behavior_source_snapshot_binding_sha256=source.binding_sha256,
        _behavior_source_snapshot_binding=source.binding,
        _behavior_runtime_isolation=runtime_isolation,
    )


def _assert_instance_paths(
    *,
    args: argparse.Namespace,
    entry: VLAEvalEntry,
) -> str:
    instance_dir = Path(args.activity_instance_dir).resolve(strict=True)
    matches = tuple(
        instance_dir.glob(f"*_{entry.activity_instance_id}_template-tro_state.json")
    )
    if len(matches) != 1 or matches[0].is_symlink():
        raise RuntimeError(
            "expected exactly one safe native state file for "
            f"{entry.task_name} s{entry.public_seed}"
        )
    return _sha256(matches[0])


def _owned_process_payload(process: subprocess.Popen[Any]) -> dict[str, Any]:
    pid = int(process.pid)
    return {
        "pid": pid,
        "pgid": os.getpgid(pid),
        "sid": os.getsid(pid),
        "start_ticks": _process_start_ticks(pid),
        "recorded_at": _utc_now(),
    }


def _process_start_ticks(pid: int) -> int:
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    fields = raw[raw.rfind(")") + 2 :].split()
    return int(fields[19])


def _argv_sha256(argv: Iterable[str]) -> str:
    encoded = json.dumps(
        list(argv),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_instance_child_process_record(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    for candidate in (path.absolute(), temporary.absolute()):
        current = Path(candidate.anchor)
        for part in candidate.parts[1:]:
            current /= part
            if current.is_symlink():
                raise RuntimeError(
                    "instance child process record path must not be a symlink "
                    f"or contain one: {current}"
                )
    _atomic_json(path, payload)


def _running_instance_child_process_record(
    *,
    process: subprocess.Popen[Any],
    argv: tuple[str, ...],
    entry: VLAEvalEntry,
    source: SourceSnapshot,
    args: argparse.Namespace,
) -> dict[str, Any]:
    child = _owned_process_payload(process)
    if child["pid"] != child["pgid"] or child["pid"] != child["sid"]:
        raise RuntimeError("instance child has no dedicated process session")
    runner_pid = os.getpid()
    started_at = _utc_now()
    return {
        "schema_version": 1,
        "state": "running",
        "pid": child["pid"],
        "pgid": child["pgid"],
        "sid": child["sid"],
        "start_ticks": child["start_ticks"],
        "runner_pid": runner_pid,
        "runner_pgid": os.getpgid(runner_pid),
        "runner_sid": os.getsid(runner_pid),
        "runner_start_ticks": _process_start_ticks(runner_pid),
        "task_name": entry.task_name,
        "public_seed": entry.public_seed,
        "activity_instance_id": entry.activity_instance_id,
        "entry_output_dir": str(entry.output_dir.absolute()),
        "source_snapshot_root": str(source.root.absolute()),
        "source_snapshot_binding_sha256": source.binding_sha256,
        "cuda_device": entry.cuda_device,
        "expected_run_nonce": args.expected_run_nonce,
        "argv_sha256": _argv_sha256(argv),
        "started_at": started_at,
        "updated_at": started_at,
        "started_monotonic_ns": int(args.instance_started_monotonic_ns),
        "action_deadline_monotonic_ns": int(args.action_deadline_monotonic_ns),
        "cleanup_deadline_monotonic_ns": int(args.cleanup_deadline_monotonic_ns),
        "hard_deadline_monotonic_ns": int(args.hard_deadline_monotonic_ns),
        "action_deadline_s": int(args.action_deadline_s),
        "cleanup_deadline_s": int(args.cleanup_deadline_s),
        "instance_timeout_s": int(args.instance_timeout_s),
    }


def _finished_instance_child_process_record(
    running: Mapping[str, Any],
    *,
    process: subprocess.Popen[Any],
    timed_out: bool,
) -> dict[str, Any]:
    return {
        **running,
        "state": "exited" if process.poll() is not None else "running",
        "returncode": process.returncode,
        "timed_out": bool(timed_out),
        "updated_at": _utc_now(),
    }


def _dashboard_terminal_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_task_success = payload.get("task_success") is True
    timed_out = payload.get("timed_out") is True
    infrastructure_error = payload.get("infrastructure_error")
    has_infrastructure_error = infrastructure_error is not None
    task_success: bool | None = (
        True if raw_task_success else None if has_infrastructure_error else False
    )
    return {
        "outcome": (
            "passed"
            if raw_task_success
            else "run_error"
            if has_infrastructure_error
            else "timeout"
            if timed_out
            else "task_failed"
        ),
        "task_success": task_success,
        "timed_out": timed_out,
    }


def _write_result_artifacts(
    *,
    entry: VLAEvalEntry,
    source: SourceSnapshot,
    checkpoint_binding: dict[str, Any],
    runtime_isolation: CampaignRuntimeIsolation | None = None,
    state_sha256: str,
    result: WindowLoopResult,
    started_at: str,
    started_monotonic: float,
    runtime_cleanup: str,
    behavior_checkout: dict[str, Any] | None = None,
    behavior_dataset_checkout: dict[str, Any] | None = None,
    action_deadline_s: int = ACTION_DEADLINE_S,
    cleanup_deadline_s: int = CLEANUP_DEADLINE_S,
    instance_timeout_s: int = INSTANCE_TIMEOUT_S,
    expected_run_nonce: str | None = None,
) -> dict[str, Any]:
    trace_summary = validate_pure_vla_tool_trace(
        entry.output_dir / "behavior_tool_trace.jsonl"
    )
    action_trace_error: str | None = None
    try:
        trace_receipt = _raw_success_from_action_trace(
            entry.output_dir / "behavior_action_trace.jsonl",
            expected_run_nonce=expected_run_nonce,
        )
    except ValueError as error:
        trace_receipt = None
        action_trace_error = f"invalid official-success action trace: {error}"
    runtime_receipt: dict[str, Any] | None = None
    runtime_receipt_error: str | None = None
    if expected_run_nonce is not None and trace_receipt is not None:
        try:
            runtime_receipt = _validated_runtime_success_receipt(
                result.official_success_receipt,
                expected_run_nonce=expected_run_nonce,
            )
        except ValueError as error:
            runtime_receipt_error = f"invalid runtime official-success receipt: {error}"
    effective_success = trace_receipt is not None and runtime_receipt_error is None
    official_receipt = (
        {
            **trace_receipt,
            **(
                {"runtime_receipt": runtime_receipt}
                if runtime_receipt is not None
                else {}
            ),
        }
        if effective_success and trace_receipt is not None
        else None
    )
    infrastructure_errors = [
        error
        for error in (
            result.infrastructure_error,
            action_trace_error,
            runtime_receipt_error,
        )
        if error not in (None, "")
    ]
    if runtime_cleanup != "complete":
        infrastructure_errors.append(f"runtime cleanup incomplete: {runtime_cleanup}")
    infrastructure_error = (
        "; ".join(str(error) for error in infrastructure_errors)
        if infrastructure_errors
        else None
    )
    if result.task_success and trace_receipt is None:
        infrastructure_error = (
            "pi0_nav_pick reported success without raw info_done.success "
            "in the action trace"
        )
    trusted_task_success: bool | None = (
        True
        if effective_success
        else None
        if infrastructure_error is not None
        else False
    )
    payload = {
        "schema_version": 1,
        "controller": "pure_vla",
        "llm_enabled": False,
        "planner": None,
        "model": None,
        "reasoning_effort": None,
        "task_name": entry.task_name,
        "public_seed": entry.public_seed,
        "activity_instance_id": entry.activity_instance_id,
        "native_env_seed": BEHAVIOR_NATIVE_ENV_SEED,
        "cuda_device": entry.cuda_device,
        "instruction": get_task_spec(entry.task_name).task_language,
        "allowed_tools": ["pi0_nav_pick"],
        "chunks_per_call": CHUNKS_PER_CALL,
        "tool_call_limit": None,
        "deadlines": {
            "action_deadline_s": int(action_deadline_s),
            "cleanup_deadline_s": int(cleanup_deadline_s),
            "instance_timeout_s": int(instance_timeout_s),
        },
        "attempt_index": 1,
        "expected_run_nonce": expected_run_nonce,
        "retry_count": 0,
        "task_success": trusted_task_success,
        "success": trusted_task_success,
        "official_success_source": _OFFICIAL_SUCCESS_SOURCE,
        "official_success_field_path": _OFFICIAL_SUCCESS_FIELD_PATH,
        "official_success_receipt": official_receipt,
        "stopped_reason": (
            "official_task_success" if effective_success else result.stopped_reason
        ),
        "timed_out": result.timed_out,
        "infrastructure_error": infrastructure_error,
        "tool_calls": trace_summary["tool_calls"],
        "vla_chunks": trace_summary["vla_chunks"],
        "total_env_steps": result.total_env_steps,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_s": round(max(0.0, time.monotonic() - started_monotonic), 3),
        "runtime_cleanup": runtime_cleanup,
        "publication_complete": False,
        "publication_eligible": False,
        "eval_published_memory": False,
        "eval_published_recipe": False,
        "source_snapshot": source.public_binding(),
        "runtime_isolation": (
            runtime_isolation.as_dict() if runtime_isolation is not None else None
        ),
        "policy_checkpoint_binding": checkpoint_binding,
        "instance_state_sha256": state_sha256,
        "behavior_checkout": behavior_checkout,
        "behavior_dataset_checkout": behavior_dataset_checkout,
    }
    _atomic_json(entry.output_dir / "baseline_result.json", payload)
    _atomic_json(entry.output_dir / "final_result.json", payload)
    _atomic_json(
        entry.output_dir / "behavior_result.json",
        {
            **payload,
            "recipe_path": None,
            "memory_path": None,
        },
    )
    if official_receipt is not None:
        _atomic_json(
            entry.output_dir / "official_success_receipt.json",
            official_receipt,
        )
    return payload


def _write_canonical_result_artifacts(
    entry: VLAEvalEntry,
    payload: dict[str, Any],
) -> None:
    """Keep parent amendments identical across every canonical result view."""

    _atomic_json(entry.output_dir / "baseline_result.json", payload)
    _atomic_json(entry.output_dir / "final_result.json", payload)
    _atomic_json(
        entry.output_dir / "behavior_result.json",
        {
            **payload,
            "recipe_path": None,
            "memory_path": None,
        },
    )


def _append_infrastructure_error(
    payload: dict[str, Any],
    error: str,
) -> None:
    existing = payload.get("infrastructure_error")
    payload["infrastructure_error"] = (
        error if existing in (None, "") else f"{existing}; {error}"
    )
    # Raw trace success remains semantic task success even when cleanup or
    # provenance fails.  Every other case stays unknown, never a fake failure.
    if payload.get("task_success") is not True:
        payload["task_success"] = None
        payload["success"] = None


def _run_instance_child(args: argparse.Namespace) -> int:
    if args.cuda_device != "6":
        raise SystemExit("pure-VLA BEHAVIOR Eval is restricted to GPU6")
    if args.chunks_per_call != CHUNKS_PER_CALL:
        raise SystemExit(f"--chunks-per-call is fixed at {CHUNKS_PER_CALL}")
    if not (
        0
        < args.action_deadline_s
        < args.cleanup_deadline_s
        < args.instance_timeout_s
        <= INSTANCE_TIMEOUT_S
    ):
        raise SystemExit(
            "timeouts must satisfy 0 < action < cleanup < instance <= 7200"
        )
    _bind_instance_deadline_contract(args, allow_generate=False)
    source = load_source_snapshot(
        args.behavior_source_snapshot_root,
        expected_binding_sha256=args.behavior_source_snapshot_binding_sha256,
    )
    _assert_running_from_source_snapshot(source)
    spec = get_task_spec(args.task_name)
    entry = build_vla_eval_plan(
        task_name=spec.task_name,
        public_seeds=(args.public_seed,),
        output_root=Path(args.output_root).parent,
        cuda_device=args.cuda_device,
    )[0]
    requested_output = Path(args.output_root).expanduser().absolute()
    entry = VLAEvalEntry(
        task_name=entry.task_name,
        public_seed=entry.public_seed,
        activity_instance_id=entry.activity_instance_id,
        output_dir=requested_output,
        cuda_device=entry.cuda_device,
    )
    runtime_isolation = _load_external_runtime_isolation(
        runtime_root=args.behavior_runtime_isolation_root,
        binding_sha256=args.behavior_runtime_isolation_binding_sha256,
        output_root=entry.output_dir,
    )
    if entry.output_dir.exists():
        raise SystemExit(f"entry output already exists: {entry.output_dir}")
    entry.output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = Path(args.policy_checkpoint).expanduser().resolve(strict=True)
    checkpoint_binding = _expected_shared_policy_checkpoint_binding()
    if str(checkpoint) != checkpoint_binding["resolved_path"]:
        raise SystemExit("pure-VLA Eval requires the shared Pi0.5 checkpoint")
    behavior_repo = Path(args.behavior_repo).expanduser().resolve(strict=True)
    try:
        behavior_python = _validated_lexical_executable_path(
            args.behavior_python,
            label="behavior_python",
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    behavior_checkout = _checkout_identity(behavior_repo)
    behavior_dataset_checkout = _checkout_identity(
        behavior_repo / ".venv-behavior" / "BEHAVIOR-1K"
    )
    runtime_args = _runtime_args(
        entry=entry,
        behavior_repo=behavior_repo,
        behavior_python=behavior_python,
        checkpoint=checkpoint,
        checkpoint_binding=checkpoint_binding,
        vla_endpoint=args.vla_endpoint,
        source=source,
        runtime_isolation=runtime_isolation,
    )
    state_sha256 = _assert_instance_paths(args=runtime_args, entry=entry)
    dashboard = FileDashboardSink(entry.output_dir / "dashboard_events.jsonl")
    started_at = _utc_now()
    started_monotonic = args.instance_started_monotonic_ns / 1e9
    dashboard.set_metadata(
        {
            "controller": "pure_vla",
            "llm-enabled": False,
            "planner": None,
            "model": None,
            "reasoning-effort": None,
            "task-name": spec.task_name,
            "task-index": spec.task_index,
            "public-seed": entry.public_seed,
            "public-instance-id": entry.activity_instance_id,
            "activity-instance-id": entry.activity_instance_id,
            "phase": "eval",
            "cuda-device": "6",
            "max-wall-clock-s": INSTANCE_TIMEOUT_S,
            "allowed-tools": ["pi0_nav_pick"],
        }
    )
    dashboard.begin_attempt(
        attempt_index=1,
        public_seed=entry.public_seed,
        activity_instance_id=entry.activity_instance_id,
    )
    env_proc: subprocess.Popen[Any] | None = None
    env_rpc = None
    env = None
    model: BehaviorVLAClient | None = None
    result = WindowLoopResult(
        task_success=False,
        official_success_receipt=None,
        tool_calls=0,
        vla_chunks=0,
        total_env_steps=0,
        stopped_reason="infrastructure_error",
        timed_out=False,
        infrastructure_error="runtime did not initialize",
        last_result=None,
    )
    cleanup_status = "pending"
    cleanup_errors: list[str] = []
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    cleanup_signal_received = False

    def cleanup_deadline(_signum: int, _frame: Any) -> None:
        nonlocal cleanup_signal_received
        if cleanup_signal_received:
            return
        cleanup_signal_received = True
        raise _InstanceDeadline("instance cleanup deadline reached")

    signal.signal(signal.SIGTERM, cleanup_deadline)
    try:
        previous_expected_nonce = os.environ.get(_EXPECTED_RUN_NONCE_ENV)
        os.environ[_EXPECTED_RUN_NONCE_ENV] = args.expected_run_nonce
        try:
            env_proc = start_env_server(runtime_args, output_dir=entry.output_dir)
        finally:
            if previous_expected_nonce is None:
                os.environ.pop(_EXPECTED_RUN_NONCE_ENV, None)
            else:
                os.environ[_EXPECTED_RUN_NONCE_ENV] = previous_expected_nonce
        _atomic_json(
            entry.output_dir / "baseline_owned_processes.json",
            {"env": _owned_process_payload(env_proc), "vla": None},
        )
        env_rpc = _managed_env_rpc_client(env_proc)
        env = BehaviorEnvClient(
            env_rpc,
            expected_meta=_expected_env_meta(
                spec,
                public_seed=entry.public_seed,
                expected_run_nonce=args.expected_run_nonce,
            ),
        )
        model = BehaviorVLAClient(str(args.vla_endpoint))
        _require_disabled_vla_health(
            model.wait_for_healthz(
                timeout_s=120.0,
                expected_checkpoint_binding=checkpoint_binding,
            )
        )
        env.vla_endpoint = str(args.vla_endpoint)
        observation, info = env.reset()
        runtime_info = info.get("_rpent") if isinstance(info, Mapping) else None
        if (
            not isinstance(runtime_info, Mapping)
            or runtime_info.get("run_nonce") != args.expected_run_nonce
            or runtime_info.get("attempt_index") != 1
        ):
            raise RuntimeError("reset info run nonce differs from supervisor binding")
        task_language = _task_language_from_reset(observation)
        if task_language != spec.task_language:
            raise RuntimeError(
                "environment language differs from TaskSpec authoritative text"
            )
        _emit_internal_dashboard_frames(
            dashboard,
            observation,
            env_step=0,
        )

        def progress(name: str, payload: dict[str, Any]) -> None:
            dashboard.on_tool_progress(name, payload)

        try:
            primitives = BehaviorPrimitives(
                env=env,
                model=model,
                max_episode_steps=MAX_EPISODE_STEPS,
                output_dir=entry.output_dir,
                video_path=entry.output_dir / "episode.mp4",
                initial_observation=observation,
                initial_info=info,
                progress_callback=progress,
                behavior_phase="eval",
                task_name=spec.task_name,
                public_seed=entry.public_seed,
                initial_attempt_index=1,
                job_id=(
                    f"behavior-vla-baseline-{entry.public_seed}-"
                    f"{int(started_monotonic * 1000)}"
                ),
                max_tool_calls=None,
                max_wall_clock_s=max(
                    0.001,
                    _remaining_deadline_seconds(args.action_deadline_monotonic_ns),
                ),
                pure_vla_baseline=True,
            )
        except TypeError as error:
            if "pure_vla_baseline" not in str(error):
                raise
            raise RuntimeError(
                "runtime lacks the private pure_vla_baseline constructor "
                "contract; do not bypass the public attached-state guard"
            ) from error
        result = run_vla_window_loop(
            primitives,
            instruction=spec.task_language,
            deadline_monotonic=args.action_deadline_monotonic_ns / 1e9,
            trace_path=entry.output_dir / "behavior_tool_trace.jsonl",
            dashboard=dashboard,
        )
    except _InstanceDeadline:
        result = WindowLoopResult(
            task_success=result.task_success,
            official_success_receipt=result.official_success_receipt,
            tool_calls=result.tool_calls,
            vla_chunks=result.vla_chunks,
            total_env_steps=(int(env.total_env_steps) if env is not None else 0),
            stopped_reason="instance_timeout",
            timed_out=True,
            infrastructure_error=None,
            last_result=result.last_result,
        )
    except BaseException as error:
        result = WindowLoopResult(
            task_success=result.task_success,
            official_success_receipt=result.official_success_receipt,
            tool_calls=result.tool_calls,
            vla_chunks=result.vla_chunks,
            total_env_steps=(int(env.total_env_steps) if env is not None else 0),
            stopped_reason="infrastructure_error",
            timed_out=False,
            infrastructure_error=f"{type(error).__name__}: {error}",
            last_result=result.last_result,
        )
    finally:
        # A second supervisor SIGTERM must not interrupt env shutdown or result
        # sealing after the first signal has already begun the cleanup path.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if model is not None:
            try:
                model.disable_actions()
            except BaseException as error:
                cleanup_errors.append(
                    f"disable_actions failed: {type(error).__name__}: {error}"
                )
        try:
            stop_env_server(env_proc, output_dir=entry.output_dir)
        except BaseException as error:
            cleanup_errors.append(
                f"env shutdown failed: {type(error).__name__}: {error}"
            )
        model_close = getattr(model, "close", None)
        if callable(model_close):
            try:
                model_close()
            except BaseException as error:
                cleanup_errors.append(
                    f"model close failed: {type(error).__name__}: {error}"
                )
        client_close = getattr(env_rpc, "close", None)
        if callable(client_close):
            try:
                client_close()
            except BaseException as error:
                cleanup_errors.append(
                    f"env client close failed: {type(error).__name__}: {error}"
                )
        cleanup_status = (
            "complete" if not cleanup_errors else "failed: " + "; ".join(cleanup_errors)
        )

    try:
        if not (entry.output_dir / "behavior_tool_trace.jsonl").exists():
            (entry.output_dir / "behavior_tool_trace.jsonl").touch(mode=0o600)
        payload = _write_result_artifacts(
            entry=entry,
            source=source,
            checkpoint_binding=checkpoint_binding,
            runtime_isolation=runtime_isolation,
            state_sha256=state_sha256,
            result=result,
            started_at=started_at,
            started_monotonic=started_monotonic,
            runtime_cleanup=cleanup_status,
            behavior_checkout=behavior_checkout,
            behavior_dataset_checkout=behavior_dataset_checkout,
            action_deadline_s=args.action_deadline_s,
            cleanup_deadline_s=args.cleanup_deadline_s,
            instance_timeout_s=args.instance_timeout_s,
            expected_run_nonce=args.expected_run_nonce,
        )
        dashboard.end_attempt(attempt_index=1, **_dashboard_terminal_state(payload))
        dashboard.mark_done(terminated=True)
        return_code = 0 if payload["infrastructure_error"] is None else 2
    finally:
        # Keep repeat SIGTERM ignored through canonical artifact and Dashboard
        # sealing. The process exits immediately after this local restoration.
        try:
            signal.signal(signal.SIGTERM, previous_sigterm)
        except BaseException:
            pass
    return return_code


def _instance_child_identity_matches(
    process: subprocess.Popen[Any],
    identity: Mapping[str, Any],
) -> bool:
    if process.poll() is not None:
        return False
    try:
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
        sid = int(identity["sid"])
        start_ticks = int(identity["start_ticks"])
        return (
            process.pid == pid == pgid == sid
            and os.getpgid(pid) == pgid
            and os.getsid(pid) == sid
            and _process_start_ticks(pid) == start_ticks
            and pgid != os.getpgrp()
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    expected_identity: Mapping[str, Any],
    timeout_s: float = 30.0,
) -> bool:
    if process.poll() is not None:
        return True
    if not _instance_child_identity_matches(process, expected_identity):
        return False
    pgid = int(expected_identity["pgid"])
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll() is not None
    deadline = time.monotonic() + max(0.0, timeout_s)
    try:
        process.wait(timeout=min(30.0, max(0.0, deadline - time.monotonic())))
        return True
    except subprocess.TimeoutExpired:
        if not _instance_child_identity_matches(process, expected_identity):
            return False
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return process.poll() is not None
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        return False
    return True


def _kill_recorded_env(
    output_dir: Path,
    *,
    hard_deadline_monotonic_ns: int | None = None,
    timeout_s: float = 30.0,
) -> tuple[bool, bool, str | None]:
    """Converge one exactly recorded env group and verify the terminal state.

    Returns ``(verified, forced, error)``.  No signal is sent unless the
    recorded leader's PID/PGID/SID/start-ticks identity still matches.
    """

    path = output_dir / "baseline_owned_processes.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload["env"]
        pid = int(identity["pid"])
        pgid = int(identity["pgid"])
        sid = int(identity["sid"])
        start_ticks = int(identity["start_ticks"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        return False, False, f"recorded env identity is unavailable: {error}"
    if pid <= 1 or pgid != pid or sid != pid:
        return False, False, "recorded env identity is not a dedicated session"

    def identity_matches() -> bool:
        try:
            return (
                os.getpgid(pid) == pgid
                and os.getsid(pid) == sid
                and _process_start_ticks(pid) == start_ticks
                and pgid != os.getpgrp()
            )
        except (OSError, ProcessLookupError):
            return False

    def group_is_gone() -> bool:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False

    if not identity_matches():
        if group_is_gone():
            return True, False, None
        return False, False, "recorded env group identity is ambiguous"

    forced = True
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True, forced, None
    deadline = time.monotonic() + max(0.0, timeout_s)
    if hard_deadline_monotonic_ns is not None:
        deadline = min(deadline, int(hard_deadline_monotonic_ns) / 1e9)
    term_deadline = min(
        deadline,
        time.monotonic() + min(10.0, max(0.0, timeout_s) / 2.0),
    )
    while time.monotonic() < term_deadline:
        if group_is_gone():
            return True, forced, None
        time.sleep(0.1)
    if not identity_matches():
        if group_is_gone():
            return True, forced, None
        return False, forced, "recorded env identity changed before SIGKILL"
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True, forced, None
    while time.monotonic() < deadline:
        if group_is_gone():
            return True, forced, None
        time.sleep(0.1)
    if group_is_gone():
        return True, forced, None
    return False, forced, "recorded env group survived SIGKILL"


def _child_argv(
    *,
    args: argparse.Namespace,
    entry: VLAEvalEntry,
    vla_endpoint: str,
) -> tuple[str, ...]:
    return (
        str(Path(args.python).expanduser().absolute()),
        "-m",
        "robots.behavior.serial_vla_eval",
        "--instance-child",
        "--output-root",
        str(entry.output_dir),
        "--cuda-device",
        "6",
        "--task-name",
        entry.task_name,
        "--public-seed",
        str(entry.public_seed),
        "--behavior-repo",
        str(Path(args.behavior_repo).expanduser().resolve()),
        "--behavior-python",
        str(Path(args.behavior_python).expanduser().absolute()),
        "--policy-checkpoint",
        str(Path(args.policy_checkpoint).expanduser().resolve()),
        "--vla-endpoint",
        vla_endpoint,
        "--expected-run-nonce",
        args.expected_run_nonce,
        "--chunks-per-call",
        str(args.chunks_per_call),
        "--action-deadline-s",
        str(args.action_deadline_s),
        "--cleanup-deadline-s",
        str(args.cleanup_deadline_s),
        "--instance-timeout-s",
        str(args.instance_timeout_s),
        "--instance-started-monotonic-ns",
        str(args.instance_started_monotonic_ns),
        "--action-deadline-monotonic-ns",
        str(args.action_deadline_monotonic_ns),
        "--cleanup-deadline-monotonic-ns",
        str(args.cleanup_deadline_monotonic_ns),
        "--hard-deadline-monotonic-ns",
        str(args.hard_deadline_monotonic_ns),
        "--behavior-source-snapshot-root",
        str(Path(args.source_snapshot_root).expanduser().resolve()),
        "--behavior-source-snapshot-binding-sha256",
        str(args.source_snapshot_binding_sha256),
        "--behavior-runtime-isolation-root",
        str(args.runtime_isolation_root),
        "--behavior-runtime-isolation-binding-sha256",
        str(args.runtime_isolation_binding_sha256),
    )


def _update_campaign_manifest(
    *,
    root: Path,
    task_spec: BehaviorTaskSpec,
    source: SourceSnapshot,
    checkpoint_binding: dict[str, Any],
    runtime_isolation: CampaignRuntimeIsolation,
    entry_payload: dict[str, Any],
) -> None:
    path = root / "pure_vla_campaign_manifest.json"
    lock_path = root / ".pure_vla_campaign_manifest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("task_name") != task_spec.task_name
                or payload.get("source_snapshot", {}).get("binding_sha256")
                != source.binding_sha256
                or payload.get("policy_checkpoint_binding", {}).get("binding_sha256")
                != checkpoint_binding["binding_sha256"]
            ):
                raise RuntimeError("campaign manifest identity mismatch")
        else:
            payload = {
                "schema_version": 1,
                "controller": "pure_vla",
                "llm_enabled": False,
                "task_name": task_spec.task_name,
                "cuda_device": "6",
                "allowed_tools": ["pi0_nav_pick"],
                "chunks_per_call": CHUNKS_PER_CALL,
                "tool_call_limit": None,
                "deadlines": entry_payload.get("deadlines"),
                "source_snapshot": source.public_binding(),
                "policy_checkpoint_binding": checkpoint_binding,
                "runtime_isolation": runtime_isolation.as_dict(),
                "gpu_lock_contract": entry_payload.get("gpu_lock_contract"),
                "entries": [],
                "created_at": _utc_now(),
            }
        existing = {
            int(item["public_seed"])
            for item in payload["entries"]
            if isinstance(item, dict) and isinstance(item.get("public_seed"), int)
        }
        seed = int(entry_payload["public_seed"])
        if seed in existing:
            raise RuntimeError(f"campaign already contains public seed {seed}")
        payload["entries"].append(entry_payload)
        payload["updated_at"] = _utc_now()
        _atomic_json(path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _campaign_locks(
    output_root: Path,
    *,
    claim_gpu_lock: bool,
):
    """Claim the established Eval lock paths without inventing another lock."""

    paths = [output_root.parent / f".{output_root.name}.lock"]
    if claim_gpu_lock:
        paths.insert(0, _gpu_lock_path("6"))
    streams: list[Any] = []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("w", encoding="utf-8")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                stream.close()
                raise RuntimeError(f"another evaluator owns {path}") from error
            streams.append(stream)
        yield tuple(paths)
    finally:
        for stream in reversed(streams):
            try:
                fcntl.flock(stream, fcntl.LOCK_UN)
            finally:
                stream.close()


def _validate_external_gpu_lock_contract(args: argparse.Namespace) -> None:
    if args.vla_endpoint and not args.external_gpu_lock_owned:
        raise SystemExit(
            "external VLA mode requires explicit external GPU lock ownership "
            "via --external-gpu-lock-owned"
        )
    if args.external_gpu_lock_owned and not (
        args.vla_endpoint
        and args.source_snapshot_root
        and args.source_snapshot_binding_sha256
        and args.runtime_isolation_root
        and args.runtime_isolation_binding_sha256
        and all(
            getattr(args, field, None) is not None
            for field in _ABSOLUTE_DEADLINE_FIELDS
        )
    ):
        raise SystemExit(
            "--external-gpu-lock-owned requires an external VLA, sealed source "
            "snapshot, bound campaign runtime isolation, and bound absolute "
            "instance deadlines"
        )


_ABSOLUTE_DEADLINE_FIELDS = (
    "instance_started_monotonic_ns",
    "action_deadline_monotonic_ns",
    "cleanup_deadline_monotonic_ns",
    "hard_deadline_monotonic_ns",
)


def _bind_instance_deadline_contract(
    args: argparse.Namespace,
    *,
    allow_generate: bool,
) -> None:
    values = {field: getattr(args, field, None) for field in _ABSOLUTE_DEADLINE_FIELDS}
    provided = {field for field, value in values.items() if value is not None}
    if provided and len(provided) != len(values):
        raise SystemExit("absolute instance deadline fields must be provided together")
    if not provided:
        if not allow_generate:
            raise SystemExit("bound absolute instance deadlines are required")
        started = time.monotonic_ns()
        values = {
            "instance_started_monotonic_ns": started,
            "action_deadline_monotonic_ns": (
                started + int(args.action_deadline_s) * 1_000_000_000
            ),
            "cleanup_deadline_monotonic_ns": (
                started + int(args.cleanup_deadline_s) * 1_000_000_000
            ),
            "hard_deadline_monotonic_ns": (
                started + int(args.instance_timeout_s) * 1_000_000_000
            ),
        }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values.values()
    ):
        raise SystemExit("absolute instance deadlines must be positive integers")
    started = int(values["instance_started_monotonic_ns"])
    expected = {
        "action_deadline_monotonic_ns": (
            started + int(args.action_deadline_s) * 1_000_000_000
        ),
        "cleanup_deadline_monotonic_ns": (
            started + int(args.cleanup_deadline_s) * 1_000_000_000
        ),
        "hard_deadline_monotonic_ns": (
            started + int(args.instance_timeout_s) * 1_000_000_000
        ),
    }
    mismatches = {
        field: {"expected": expected_value, "actual": values[field]}
        for field, expected_value in expected.items()
        if values[field] != expected_value
    }
    if mismatches:
        raise SystemExit(
            "absolute instance deadlines do not match relative timeout contract: "
            f"{mismatches!r}"
        )
    for field, value in values.items():
        setattr(args, field, int(value))


def _remaining_deadline_seconds(deadline_monotonic_ns: int) -> float:
    return max(0.0, (int(deadline_monotonic_ns) - time.monotonic_ns()) / 1e9)


def _require_disabled_vla_health(payload: Mapping[str, Any]) -> None:
    if payload.get("actions_enabled") is not False:
        raise RuntimeError("pure-VLA serial boundary requires actions_enabled=false")


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    try:
        left.relative_to(right)
    except ValueError:
        pass
    else:
        return True
    try:
        right.relative_to(left)
    except ValueError:
        return False
    return True


def _load_external_runtime_isolation(
    *,
    runtime_root: str | os.PathLike[str] | None,
    binding_sha256: str | None,
    output_root: str | os.PathLike[str],
) -> CampaignRuntimeIsolation:
    """Validate a caller-owned runtime tree before any Eval output is written."""

    if runtime_root is None or binding_sha256 is None:
        raise SystemExit(
            "formal pure-VLA Eval requires an external runtime isolation root "
            "and binding SHA256"
        )
    runtime_isolation = validate_campaign_runtime_isolation(
        runtime_root,
        expected_binding_sha256=binding_sha256,
    )
    if runtime_isolation.cuda_device != "6":
        raise SystemExit("runtime isolation is not bound to GPU6")
    canonical_output = Path(output_root).expanduser().resolve(strict=False)
    if _paths_overlap(runtime_isolation.root, canonical_output):
        raise SystemExit(
            "runtime isolation root and Eval output root must be disjoint "
            "(neither may contain the other)"
        )
    return runtime_isolation


def _vla_launcher_log_dir(output_root: Path) -> Path:
    """Create the formal forensic-log directory, never a runtime/cache tree."""

    output_dir = output_root / "launcher_logs" / "vla"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _gpu_lock_contract(
    *,
    args: argparse.Namespace,
    lock_paths: Iterable[Path],
) -> dict[str, Any]:
    return {
        "path": str(_gpu_lock_path("6")),
        "owner": (
            "external_paired_supervisor"
            if args.external_gpu_lock_owned
            else "serial_vla_eval"
        ),
        "claimed_here": not args.external_gpu_lock_owned,
        "active_lock_paths": [str(path) for path in lock_paths],
    }


def _run_parent_locked(
    args: argparse.Namespace,
    *,
    source: SourceSnapshot,
    task_spec: BehaviorTaskSpec,
    entries: tuple[VLAEvalEntry, ...],
    root: Path,
    lock_paths: tuple[Path, ...],
    runtime_isolation: CampaignRuntimeIsolation,
) -> int:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"--output-root must be absent or empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    args.runtime_isolation_root = str(runtime_isolation.root)
    args.runtime_isolation_binding_sha256 = runtime_isolation.binding_sha256
    checkpoint = Path(args.policy_checkpoint).expanduser().resolve(strict=True)
    expected_checkpoint = _expected_shared_policy_checkpoint_binding()
    if str(checkpoint) != expected_checkpoint["resolved_path"]:
        raise SystemExit("pure-VLA Eval requires the shared Pi0.5 checkpoint")

    owned_vla: subprocess.Popen[Any] | None = None
    vla_endpoint = args.vla_endpoint
    if vla_endpoint is None:
        # Standalone mode performs the expensive content hash once per campaign.
        checkpoint_binding = validate_policy_checkpoint(checkpoint).as_dict()
        vla_args = argparse.Namespace(
            behavior_python=args.behavior_python,
            behavior_repo=args.behavior_repo,
            policy_checkpoint=str(checkpoint),
            seed=BEHAVIOR_NATIVE_ENV_SEED,
            cuda_device="6",
            vla_port=0,
            vla_ready_timeout_s=1800,
            _behavior_policy_checkpoint_binding=checkpoint_binding,
            _behavior_runtime_isolation=runtime_isolation,
            behavior_controller_mode="pi0_nav_pick_only",
        )
        vla_launcher_logs = _vla_launcher_log_dir(root)
        previous_repo_root = os.environ.get("RPENT_REPO_ROOT")
        os.environ["RPENT_REPO_ROOT"] = str(source.root)
        try:
            vla_endpoint, owned_vla = start_vla_server(
                vla_args,
                output_dir=vla_launcher_logs,
            )
        finally:
            if previous_repo_root is None:
                os.environ.pop("RPENT_REPO_ROOT", None)
            else:
                os.environ["RPENT_REPO_ROOT"] = previous_repo_root
    else:
        checkpoint_binding = expected_checkpoint
        client = BehaviorVLAClient(vla_endpoint)
        try:
            _require_disabled_vla_health(
                client.wait_for_healthz(
                    timeout_s=120.0,
                    expected_checkpoint_binding=checkpoint_binding,
                )
            )
        finally:
            client.close()

    exit_code = 0
    try:
        for entry in entries:
            entry_args = argparse.Namespace(**vars(args))
            _bind_instance_deadline_contract(entry_args, allow_generate=True)
            expected_state_sha256 = _assert_instance_paths(
                args=argparse.Namespace(
                    activity_instance_dir=str(
                        _instance_dir(
                            Path(args.behavior_repo).expanduser().resolve(strict=True),
                            task_spec,
                        )
                    )
                ),
                entry=entry,
            )
            if entry.output_dir.exists():
                raise RuntimeError(
                    f"refusing to overwrite existing Eval output: {entry.output_dir}"
                )
            argv = _child_argv(
                args=entry_args,
                entry=entry,
                vla_endpoint=str(vla_endpoint),
            )
            log_path = root / f"{task_spec.tag(entry.public_seed)}.launcher.log"
            environment = os.environ.copy()
            environment["PYTHONNOUSERSITE"] = "1"
            environment["CUDA_VISIBLE_DEVICES"] = "6"
            environment["RPENT_REPO_ROOT"] = str(source.root)
            environment["PYTHONPATH"] = os.pathsep.join(
                (
                    str(source.root),
                    str(Path(args.behavior_repo).expanduser().resolve()),
                )
            )
            environment.update(runtime_isolation.environment())
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    argv,
                    cwd=source.root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child_process_record_path = root / "instance_child_process.json"
                try:
                    child_process_record = _running_instance_child_process_record(
                        process=process,
                        argv=argv,
                        entry=entry,
                        source=source,
                        args=entry_args,
                    )
                    _write_instance_child_process_record(
                        child_process_record_path,
                        child_process_record,
                    )
                except BaseException:
                    try:
                        provisional_identity = _owned_process_payload(process)
                    except (OSError, ProcessLookupError):
                        provisional_identity = None
                    if provisional_identity is not None:
                        _terminate_process_group(
                            process,
                            expected_identity=provisional_identity,
                        )
                    raise
                timed_out = False
                child_exit: int | None = None
                try:
                    try:
                        child_exit = process.wait(
                            timeout=_remaining_deadline_seconds(
                                entry_args.cleanup_deadline_monotonic_ns
                            )
                        )
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        if _instance_child_identity_matches(
                            process,
                            child_process_record,
                        ):
                            try:
                                os.killpg(
                                    int(child_process_record["pgid"]),
                                    signal.SIGTERM,
                                )
                            except ProcessLookupError:
                                pass
                        remaining = _remaining_deadline_seconds(
                            entry_args.hard_deadline_monotonic_ns
                        )
                        try:
                            child_exit = process.wait(timeout=remaining)
                        except subprocess.TimeoutExpired:
                            _terminate_process_group(
                                process,
                                expected_identity=child_process_record,
                                timeout_s=_remaining_deadline_seconds(
                                    entry_args.hard_deadline_monotonic_ns
                                ),
                            )
                            child_exit = process.returncode
                except BaseException:
                    _terminate_process_group(
                        process,
                        expected_identity=child_process_record,
                        timeout_s=_remaining_deadline_seconds(
                            entry_args.hard_deadline_monotonic_ns
                        ),
                    )
                    _kill_recorded_env(
                        entry.output_dir,
                        hard_deadline_monotonic_ns=(
                            entry_args.hard_deadline_monotonic_ns
                        ),
                    )
                    raise
                finally:
                    _write_instance_child_process_record(
                        child_process_record_path,
                        _finished_instance_child_process_record(
                            child_process_record,
                            process=process,
                            timed_out=timed_out,
                        ),
                    )
            cleanup_verified, cleanup_forced, cleanup_error = _kill_recorded_env(
                entry.output_dir,
                hard_deadline_monotonic_ns=(entry_args.hard_deadline_monotonic_ns),
            )
            try:
                parent_observed_state_sha256 = _assert_instance_paths(
                    args=argparse.Namespace(
                        activity_instance_dir=str(
                            _instance_dir(
                                Path(args.behavior_repo)
                                .expanduser()
                                .resolve(strict=True),
                                task_spec,
                            )
                        )
                    ),
                    entry=entry,
                )
            except Exception:
                parent_observed_state_sha256 = None
            result_path = entry.output_dir / "baseline_result.json"
            child_result_present = result_path.is_file()
            if child_result_present:
                entry_payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                action_trace_error: str | None = None
                try:
                    trace_receipt = _raw_success_from_action_trace(
                        entry.output_dir / "behavior_action_trace.jsonl",
                        expected_run_nonce=entry_args.expected_run_nonce,
                    )
                except ValueError as error:
                    trace_receipt = None
                    action_trace_error = (
                        f"invalid official-success action trace: {error}"
                    )
                fallback_infrastructure_error = (
                    action_trace_error
                    if action_trace_error is not None
                    else f"child exited {child_exit} without a sealed result"
                )
                # A trace-only fallback cannot prove agreement with the runtime
                # success receipt, so the nonce-bound formal path fails closed.
                fallback_success = False
                fallback_task_success: bool | None = (
                    True
                    if fallback_success
                    else None
                    if fallback_infrastructure_error is not None
                    else False
                )
                entry_payload = {
                    "task_name": entry.task_name,
                    "public_seed": entry.public_seed,
                    "activity_instance_id": entry.activity_instance_id,
                    "task_success": fallback_task_success,
                    "success": fallback_task_success,
                    "official_success_source": _OFFICIAL_SUCCESS_SOURCE,
                    "official_success_field_path": _OFFICIAL_SUCCESS_FIELD_PATH,
                    "official_success_receipt": None,
                    "forensic_action_trace_receipt": trace_receipt,
                    "expected_run_nonce": entry_args.expected_run_nonce,
                    "timed_out": timed_out,
                    "stopped_reason": (
                        "official_task_success"
                        if fallback_success
                        else "instance_timeout"
                        if timed_out
                        else "child_process_error"
                    ),
                    "infrastructure_error": fallback_infrastructure_error,
                    "source_snapshot": source.public_binding(),
                    "policy_checkpoint_binding": checkpoint_binding,
                    "runtime_isolation": runtime_isolation.as_dict(),
                    "deadlines": {
                        "action_deadline_s": int(entry_args.action_deadline_s),
                        "cleanup_deadline_s": int(entry_args.cleanup_deadline_s),
                        "instance_timeout_s": int(entry_args.instance_timeout_s),
                    },
                    "gpu_lock_contract": _gpu_lock_contract(
                        args=args,
                        lock_paths=lock_paths,
                    ),
                    "publication_complete": False,
                }
                entry.output_dir.mkdir(parents=True, exist_ok=True)
            entry_payload["gpu_lock_contract"] = _gpu_lock_contract(
                args=args,
                lock_paths=lock_paths,
            )
            if (
                entry_payload.get("task_success") is not True
                and entry_payload.get("infrastructure_error") is not None
            ):
                entry_payload["task_success"] = None
                entry_payload["success"] = None
            _bind_result_to_frozen_instance_state(
                entry_payload,
                expected_state_sha256=expected_state_sha256,
                parent_observed_state_sha256=parent_observed_state_sha256,
            )
            entry_payload["runner_cleanup"] = {
                "verified": cleanup_verified,
                "forced": cleanup_forced,
                "error": cleanup_error,
            }
            if cleanup_forced:
                _append_infrastructure_error(
                    entry_payload,
                    "outer runner required forced cleanup of the recorded env group",
                )
            if not cleanup_verified:
                _append_infrastructure_error(
                    entry_payload,
                    cleanup_error or "outer runner cleanup could not be verified",
                )
            if timed_out:
                entry_payload["timed_out"] = True
                if (
                    entry_payload.get("task_success") is not True
                    and entry_payload.get("infrastructure_error") is None
                ):
                    entry_payload["task_success"] = False
                    entry_payload["success"] = False
                    entry_payload["stopped_reason"] = "instance_timeout"
            entry_payload["artifact_seal_complete"] = bool(
                child_result_present
                and cleanup_verified
                and not cleanup_forced
                and entry_payload.get("runtime_cleanup") == "complete"
                and entry_payload.get("infrastructure_error") is None
            )
            _write_canonical_result_artifacts(entry, entry_payload)
            _update_campaign_manifest(
                root=root,
                task_spec=task_spec,
                source=source,
                checkpoint_binding=checkpoint_binding,
                runtime_isolation=runtime_isolation,
                entry_payload=entry_payload,
            )
            if child_exit not in {0, None} or not cleanup_verified or cleanup_forced:
                exit_code = 2
            if not cleanup_verified or cleanup_forced:
                # A verified clean boundary is mandatory before the next native
                # instance can enter this serial GPU lane.
                break
    finally:
        if owned_vla is not None:
            _terminate_process(owned_vla)
    return exit_code


def _run_parent(args: argparse.Namespace) -> int:
    if args.cuda_device != "6":
        raise SystemExit("pure-VLA BEHAVIOR Eval is restricted to GPU6")
    if args.chunks_per_call != CHUNKS_PER_CALL:
        raise SystemExit(f"--chunks-per-call is fixed at {CHUNKS_PER_CALL}")
    if not (
        0
        < args.action_deadline_s
        < args.cleanup_deadline_s
        < args.instance_timeout_s
        <= INSTANCE_TIMEOUT_S
    ):
        raise SystemExit(
            "timeouts must satisfy 0 < action < cleanup < instance <= 7200"
        )
    try:
        args.python = str(
            _validated_lexical_executable_path(
                args.python,
                label="python",
            )
        )
        args.behavior_python = str(
            _validated_lexical_executable_path(
                args.behavior_python,
                label="behavior_python",
            )
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    deadline_prebound = any(
        getattr(args, field, None) is not None for field in _ABSOLUTE_DEADLINE_FIELDS
    )
    if deadline_prebound:
        _bind_instance_deadline_contract(args, allow_generate=False)
    _validate_external_gpu_lock_contract(args)
    source = load_source_snapshot(
        args.source_snapshot_root,
        expected_binding_sha256=args.source_snapshot_binding_sha256,
    )
    _assert_running_from_source_snapshot(source)
    task_spec = get_task_spec(args.task_name)
    entries = build_vla_eval_plan(
        task_name=task_spec.task_name,
        public_seeds=args.public_seed,
        output_root=args.output_root,
        cuda_device=args.cuda_device,
    )
    if deadline_prebound and len(entries) != 1:
        raise SystemExit(
            "one pre-bound absolute deadline contract requires exactly one "
            "Eval instance"
        )
    root = Path(args.output_root).expanduser().absolute()
    runtime_isolation = _load_external_runtime_isolation(
        runtime_root=args.runtime_isolation_root,
        binding_sha256=args.runtime_isolation_binding_sha256,
        output_root=root,
    )
    args.runtime_isolation_root = str(runtime_isolation.root)
    args.runtime_isolation_binding_sha256 = runtime_isolation.binding_sha256
    root.parent.mkdir(parents=True, exist_ok=True)
    # The paired supervisor explicitly owns GPU6 for external-VLA mode.
    # Standalone managed-VLA mode must claim that exact global lock itself.
    with _campaign_locks(
        root,
        claim_gpu_lock=not args.external_gpu_lock_owned,
    ) as lock_paths:
        return _run_parent_locked(
            args,
            source=source,
            task_spec=task_spec,
            entries=entries,
            root=root,
            lock_paths=lock_paths,
            runtime_isolation=runtime_isolation,
        )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the GPU6 BEHAVIOR pure-VLA Eval baseline. No LLM, prompt, "
            "recipe, memory, or non-Pi0 action tool participates in control."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cuda-device", default="6")
    parser.add_argument(
        "--task-name",
        default=PICKING_UP_TRASH_TASK_SPEC.task_name,
    )
    parser.add_argument("--public-seed", action="append", type=int, default=None)
    parser.add_argument("--repo-root", default=str(Path(__file__).parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--behavior-repo", required=True)
    parser.add_argument("--behavior-python", required=True)
    parser.add_argument(
        "--policy-checkpoint",
        default=str(SHARED_POLICY_CHECKPOINT_PATH),
    )
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--expected-run-nonce", required=True)
    parser.add_argument(
        "--external-gpu-lock-owned",
        action="store_true",
        help=(
            "declare that an external paired-campaign supervisor owns the "
            "GPU6 lock for this entire invocation"
        ),
    )
    parser.add_argument("--chunks-per-call", type=int, default=CHUNKS_PER_CALL)
    parser.add_argument(
        "--action-deadline-s",
        type=int,
        default=ACTION_DEADLINE_S,
    )
    parser.add_argument(
        "--cleanup-deadline-s",
        type=int,
        default=CLEANUP_DEADLINE_S,
    )
    parser.add_argument(
        "--instance-timeout-s",
        type=int,
        default=INSTANCE_TIMEOUT_S,
    )
    parser.add_argument("--instance-started-monotonic-ns", type=int, default=None)
    parser.add_argument("--action-deadline-monotonic-ns", type=int, default=None)
    parser.add_argument("--cleanup-deadline-monotonic-ns", type=int, default=None)
    parser.add_argument("--hard-deadline-monotonic-ns", type=int, default=None)
    parser.add_argument("--source-snapshot-root", default=None)
    parser.add_argument("--source-snapshot-binding-sha256", default=None)
    parser.add_argument("--runtime-isolation-root", default=None)
    parser.add_argument("--runtime-isolation-binding-sha256", default=None)
    parser.add_argument("--instance-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--behavior-source-snapshot-root",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--behavior-source-snapshot-binding-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--behavior-runtime-isolation-root",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--behavior-runtime-isolation-binding-sha256",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if _RUN_NONCE_RE.fullmatch(args.expected_run_nonce) is None:
        parser.error("--expected-run-nonce must be 32 lowercase hex characters")
    if args.instance_child:
        if args.public_seed is None or len(args.public_seed) != 1:
            parser.error("instance child requires exactly one --public-seed")
        args.public_seed = args.public_seed[0]
        if (
            args.vla_endpoint is None
            or args.behavior_source_snapshot_root is None
            or args.behavior_source_snapshot_binding_sha256 is None
            or args.behavior_runtime_isolation_root is None
            or args.behavior_runtime_isolation_binding_sha256 is None
        ):
            parser.error(
                "instance child requires external VLA and behavior source binding"
            )
    elif (
        args.source_snapshot_root is None or args.source_snapshot_binding_sha256 is None
    ):
        parser.error("parent requires --source-snapshot-root and its binding SHA256")
    elif (
        args.runtime_isolation_root is None
        or args.runtime_isolation_binding_sha256 is None
    ):
        parser.error(
            "parent requires an external --runtime-isolation-root and its "
            "binding SHA256"
        )
    return args


def main(argv: Iterable[str] | None = None) -> int:
    """Run one child instance or its strictly serial campaign parent."""

    args = _parse_args(argv)
    return _run_instance_child(args) if args.instance_child else _run_parent(args)


__all__ = [
    "ACTION_DEADLINE_S",
    "ACTION_STEPS_PER_CALL",
    "CHUNKS_PER_CALL",
    "CLEANUP_DEADLINE_S",
    "INSTANCE_TIMEOUT_S",
    "SourceSnapshot",
    "VLAEvalEntry",
    "WindowLoopResult",
    "build_vla_eval_plan",
    "load_source_snapshot",
    "main",
    "run_vla_window_loop",
    "validate_pure_vla_tool_trace",
]


if __name__ == "__main__":
    raise SystemExit(main())
