"""Thread-safe BEHAVIOR Dashboard manual-control coordination.

The Dashboard controller is deliberately transport agnostic.  Runtime wiring
binds it to the same public ``BehaviorToolkit`` instance used by the agent, so
manual control never creates a second environment RPC client.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from robots.behavior.schemas import (
    DASHBOARD_HOLD_ARM_DELAY_S,
    DASHBOARD_PREDICTED_PLAN_DEPTH,
)

LEASE_TIMEOUT_S = 1.2
_WATCHDOG_PERIOD_S = 0.1
_TARGETS = frozenset({"chassis", "left_arm", "right_arm"})
_CAMERAS = frozenset({"head", "left_wrist", "right_wrist"})
_MOTION_ACTIONS = frozenset(
    {
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "up",
        "down",
        "rotate_left",
        "rotate_right",
        "open",
        "close",
    }
)
_ACTIONS = _MOTION_ACTIONS | {"observe"}
_CHASSIS_ACTIONS = frozenset(
    {"forward", "backward", "turn_left", "turn_right", "up", "down", "observe"}
)
_ONE_SHOT_ACTIONS = frozenset({"open", "close", "observe"})
_QUEUE_CAPACITY = 5
_CAPTURE_IDLE_DELAY_S = 0.0
_PLANNING_METADATA_FIELDS = (
    "planning_elapsed_s",
    "planning_profile",
    "fast_solver_deadline_s",
    "fast_solver_deadline",
    "latency_metrics",
    "selected_solver_stage",
    "solver_stages",
    "predicted_terminal",
    "predicted_start_digest",
    "predicted_terminal_digest",
)
_PLANNING_TRAJECTORY_FIELD_FRAGMENT = "trajectory"
_PLANNING_METADATA_MAX_DEPTH = 4
_PLANNING_METADATA_MAX_ITEMS = 32


class ControlRequestError(RuntimeError):
    """A stable HTTP-facing control rejection."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)

    def payload(self) -> dict[str, Any]:
        return {"error": self.message, "code": self.code}


class _PreparedPlanningFailure(RuntimeError):
    def __init__(self, stop_reason: str, error: str) -> None:
        super().__init__(error)
        self.stop_reason = stop_reason


class BehaviorRawSuccessLatch:
    """Monotonic, thread-safe latch for raw BEHAVIOR success evidence."""

    def __init__(
        self,
        *,
        run_nonce: str | None = None,
        attempt_nonce: str | None = None,
        attempt_index: int | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._latched = False
        self._expected_run_nonce = run_nonce
        self._expected_attempt_nonce = attempt_nonce
        self._expected_attempt_index = attempt_index
        self._receipt_binding: dict[str, Any] | None = None

    @staticmethod
    def _receipt_from_result(result: Any) -> Mapping[str, Any] | None:
        if not isinstance(result, Mapping):
            return None
        direct = result.get("official_success_receipt")
        if isinstance(direct, Mapping):
            return direct
        for info_key in ("info", "last_info"):
            info = result.get(info_key)
            runtime = info.get("_rpent") if isinstance(info, Mapping) else None
            if not isinstance(runtime, Mapping):
                continue
            direct = runtime.get("official_success_receipt")
            if isinstance(direct, Mapping):
                return direct
            monitor = runtime.get("pi0_nav_pick_monitor")
            nested = (
                monitor.get("official_success_receipt")
                if isinstance(monitor, Mapping)
                else None
            )
            if isinstance(nested, Mapping):
                return nested
        return None

    @staticmethod
    def _raw_success_matches(result: Any, receipt: Mapping[str, Any]) -> bool:
        if not isinstance(result, Mapping):
            return False
        for info_key in ("info", "last_info"):
            info = result.get(info_key)
            done = info.get("done") if isinstance(info, Mapping) else None
            if isinstance(done, Mapping) and done.get("success") is True:
                return True
        info_done = result.get("info_done")
        if isinstance(info_done, Mapping) and info_done.get("success") is True:
            return True
        # Manual RPC results may intentionally expose only the immutable receipt.
        raw_done = receipt.get("raw_done")
        return isinstance(raw_done, Mapping) and raw_done.get("success") is True

    def bind_attempt(
        self,
        *,
        run_nonce: str,
        attempt_nonce: str,
        attempt_index: int,
    ) -> None:
        """Bind validation to the one environment attempt before execution."""

        if re.fullmatch(r"[0-9a-f]{32}", run_nonce or "") is None:
            raise ValueError("success latch run_nonce must be 32 lowercase hex")
        if re.fullmatch(r"[0-9a-f]{32}", attempt_nonce or "") is None:
            raise ValueError("success latch attempt_nonce must be 32 lowercase hex")
        if type(attempt_index) is not int:
            raise ValueError("success latch attempt_index must be an integer")
        with self._lock:
            expected = (
                self._expected_run_nonce,
                self._expected_attempt_nonce,
                self._expected_attempt_index,
            )
            incoming = (run_nonce, attempt_nonce, attempt_index)
            if any(value is not None for value in expected) and expected != incoming:
                raise RuntimeError("success latch attempt binding cannot change")
            self._expected_run_nonce = run_nonce
            self._expected_attempt_nonce = attempt_nonce
            self._expected_attempt_index = attempt_index

    def _validated_receipt(self, result: Any) -> dict[str, Any] | None:
        receipt_value = self._receipt_from_result(result)
        if not isinstance(receipt_value, Mapping):
            return None
        receipt = dict(receipt_value)
        expected_keys = {
            "schema_version",
            "source",
            "run_nonce",
            "attempt_nonce",
            "attempt_index",
            "env_step",
            "raw_done",
            "receipt_sha256",
        }
        if set(receipt) != expected_keys:
            return None
        raw_done = receipt.get("raw_done")
        if (
            type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1
            or receipt.get("source") != 'info["done"]["success"]'
            or not isinstance(raw_done, Mapping)
            or type(raw_done.get("success")) is not bool
            or raw_done["success"] is not True
            or re.fullmatch(r"[0-9a-f]{32}", str(receipt.get("run_nonce") or ""))
            is None
            or re.fullmatch(
                r"[0-9a-f]{32}", str(receipt.get("attempt_nonce") or "")
            )
            is None
            or type(receipt.get("attempt_index")) is not int
            or type(receipt.get("env_step")) is not int
            or int(receipt["env_step"]) < 0
        ):
            return None
        claimed = receipt.get("receipt_sha256")
        if (
            not isinstance(claimed, str)
            or re.fullmatch(r"[0-9a-f]{64}", claimed) is None
        ):
            return None
        unsigned = dict(receipt)
        unsigned.pop("receipt_sha256")
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        expected_hash = hashlib.sha256(canonical).hexdigest()
        if (
            not hmac.compare_digest(claimed, expected_hash)
            or not self._raw_success_matches(result, receipt)
        ):
            return None
        if (
            self._expected_run_nonce is not None
            and receipt["run_nonce"] != self._expected_run_nonce
        ):
            return None
        if (
            self._expected_attempt_nonce is not None
            and receipt["attempt_nonce"] != self._expected_attempt_nonce
        ):
            return None
        if (
            self._expected_attempt_index is not None
            and receipt["attempt_index"] != self._expected_attempt_index
        ):
            return None
        # If the result carries public attempt identity, it must agree too.
        for key in ("run_nonce", "attempt_nonce", "attempt_index"):
            if key in result and result.get(key) != receipt[key]:
                return None
        return {
            "source": receipt["source"],
            "run_nonce": receipt["run_nonce"],
            "attempt_nonce": receipt["attempt_nonce"],
            "attempt_index": int(receipt["attempt_index"]),
            "env_step": int(receipt["env_step"]),
            "receipt_sha256": claimed,
        }

    def observe_with_binding(
        self, result: Any
    ) -> tuple[bool, dict[str, Any] | None]:
        """Latch the raw lifecycle boolean; receipts are optional audit data."""

        raw_source: str | None = None
        raw_env_step: int | None = None
        if isinstance(result, Mapping):
            for info_key in ("info", "last_info"):
                info = result.get(info_key)
                done = info.get("done") if isinstance(info, Mapping) else None
                if isinstance(done, Mapping) and done.get("success") is True:
                    raw_source = f'{info_key}["done"]["success"]'
                    env_step = result.get("env_step")
                    if type(env_step) is int:
                        raw_env_step = env_step
                    break
            if raw_source is None:
                info_done = result.get("info_done")
                if (
                    isinstance(info_done, Mapping)
                    and info_done.get("success") is True
                ):
                    raw_source = 'info_done["success"]'
                    env_step = result.get("env_step")
                    if type(env_step) is int:
                        raw_env_step = env_step
        receipt_audit = self._validated_receipt(result)
        if raw_source is None and receipt_audit is not None:
            raw_source = str(receipt_audit["source"])
            raw_env_step = int(receipt_audit["env_step"])
        binding = (
            {
                "source": raw_source,
                **(
                    {"env_step": raw_env_step}
                    if raw_env_step is not None
                    else {}
                ),
                **(
                    {"receipt": receipt_audit}
                    if receipt_audit is not None
                    else {}
                ),
            }
            if raw_source is not None
            else None
        )
        with self._lock:
            if binding is not None:
                if self._receipt_binding is None:
                    self._receipt_binding = dict(binding)
                self._latched = True
            return self._latched, (
                dict(binding) if binding is not None else None
            )

    def observe(self, result: Any) -> bool:
        latched, _ = self.observe_with_binding(result)
        return latched

    def is_latched(self) -> bool:
        with self._lock:
            return self._latched

    def is_bound(self) -> bool:
        with self._lock:
            return bool(
                self._expected_run_nonce
                and self._expected_attempt_nonce
                and self._expected_attempt_index is not None
            )

    def receipt_binding(self) -> dict[str, Any] | None:
        with self._lock:
            return (
                dict(self._receipt_binding)
                if self._receipt_binding is not None
                else None
            )


class BehaviorCommandArbiter:
    """Serialize actual Env transactions without command-scoped permits."""

    def __init__(self, *, success_latch: BehaviorRawSuccessLatch | None = None) -> None:
        self.success_latch = success_latch or BehaviorRawSuccessLatch()
        self._condition = threading.Condition()
        self._owner: str | None = None
        self._command_id: str | None = None
        self._agent_waiters = 0
        self._quiescing = False
        self._listeners: list[Any] = []

    def add_listener(self, callback: Any) -> None:
        if not callable(callback):
            raise TypeError("arbiter listener must be callable")
        with self._condition:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Any) -> None:
        with self._condition:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify_listeners(self) -> None:
        with self._condition:
            listeners = tuple(self._listeners)
        for callback in listeners:
            try:
                callback()
            except Exception:
                continue

    @contextmanager
    def agent_transaction(self) -> Iterator[None]:
        with self._condition:
            self._agent_waiters += 1
            try:
                while self._owner is not None and not self._quiescing:
                    self._condition.wait()
                if self._quiescing:
                    raise RuntimeError("behavior command arbiter is quiescing")
                if self.success_latch.is_latched():
                    raise RuntimeError("raw task success is terminal")
                self._owner = "agent"
                self._command_id = None
            finally:
                self._agent_waiters -= 1
        self._notify_listeners()
        try:
            yield
        finally:
            with self._condition:
                if self._owner == "agent":
                    self._owner = None
                    self._command_id = None
                    self._condition.notify_all()
            self._notify_listeners()

    @contextmanager
    def manual_execution(self, command_id: str) -> Iterator[None]:
        """Wait for the one global Env execution slot.

        ``command_id`` is retained only for observability.  It is not a permit,
        idempotency key, or admission predicate.
        """

        command_id = str(command_id or "").strip()
        with self._condition:
            while self._owner is not None and not self._quiescing:
                self._condition.wait()
            if self._quiescing:
                raise RuntimeError("behavior command arbiter is quiescing")
            if self.success_latch.is_latched():
                raise RuntimeError("raw task success is terminal")
            self._owner = "manual"
            self._command_id = command_id or None
        self._notify_listeners()
        try:
            yield
        finally:
            with self._condition:
                if self._owner == "manual":
                    self._owner = None
                    self._command_id = None
                    self._condition.notify_all()
            self._notify_listeners()

    def try_acquire_manual(self, command_id: str) -> tuple[bool, str | None]:
        """Compatibility-only admission; does not reserve the execution slot."""

        with self._condition:
            if self._quiescing:
                return False, "controller_unavailable"
            if self.success_latch.is_latched():
                return False, "official_success_latched"
            return True, None

    def release_manual(self, command_id: str) -> None:
        """Compatibility no-op; manual ownership exists only in execution."""

    def handoff_manual(self, command_id: str, next_command_id: str) -> bool:
        """Compatibility no-op; command IDs do not own the Env mutex."""

        with self._condition:
            if self._quiescing or self.success_latch.is_latched():
                return False
            return True

    def require_manual_permit(self, command_id: str) -> None:
        """Compatibility lifecycle check; command IDs are not permits."""

        command_id = str(command_id or "").strip()
        with self._condition:
            if not command_id:
                raise RuntimeError("Dashboard command_id is required")
            if self._quiescing:
                raise RuntimeError("behavior command arbiter is quiescing")
            if self.success_latch.is_latched():
                raise RuntimeError("raw task success is terminal")

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "owner": self._owner,
                "command_id": self._command_id,
                "busy": self._owner is not None,
                "agent_waiters": self._agent_waiters,
                "quiescing": self._quiescing,
                "success_latched": self.success_latch.is_latched(),
            }

    def quiesce(self) -> None:
        with self._condition:
            self._quiescing = True
            self._condition.notify_all()
        self._notify_listeners()

    def drain(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while self._owner is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def close(self, timeout_s: float = 10.0) -> bool:
        self.quiesce()
        return self.drain(timeout_s)


@dataclass
class _Lease:
    lease_id: str
    generation: int
    deadline: float
    accepted_at: float = field(default_factory=time.monotonic)
    last_sequence: int = 0
    stopped: bool = False
    expired: bool = False
    stop_reason: str | None = None
    payload_fingerprint: tuple[str, str, str] | None = None
    hold_armed: bool = False
    compatibility_command_id: str | None = None
    executed_count: int = 0
    last_executed_plan_id: str | None = None
    terminal_capture_requested: bool = False


@dataclass
class _Command:
    command_id: str
    lease_id: str
    sequence: int
    target: str
    action: str
    camera: str
    payload_fingerprint: tuple[str, str, str]
    phase: str = "accepted"
    accepted_at: float = field(default_factory=time.monotonic)
    result: dict[str, Any] | None = None
    plan_id: str | None = None
    planning_metadata: dict[str, Any] = field(default_factory=dict)
    predecessor_plan_id: str | None = None
    predicted_start_digest: str | None = None
    predicted_terminal: dict[str, Any] | None = None
    predicted_terminal_digest: str | None = None
    timeline_started: bool = False
    acceptance_snapshot: dict[str, Any] | None = None
    initial_lease_command: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "lease_id": self.lease_id,
            "sequence": self.sequence,
            "target": self.target,
            "action": self.action,
            "camera": self.camera,
            "phase": self.phase,
            "result": self.result,
            "plan_id": self.plan_id,
            "predecessor_plan_id": self.predecessor_plan_id,
            "predicted_start_digest": self.predicted_start_digest,
            "predicted_terminal_digest": self.predicted_terminal_digest,
            "planning_metadata": dict(self.planning_metadata),
        }


@dataclass
class _PredictedPlan:
    """One non-executing trajectory prediction owned by an active hold lease."""

    plan_id: str
    lease_id: str
    lease_generation: int
    target: str
    action: str
    camera: str
    predecessor_plan_id: str
    predicted_start_digest: str
    predicted_terminal: dict[str, Any]
    predicted_terminal_digest: str
    planning_metadata: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "lease_id": self.lease_id,
            "lease_generation": self.lease_generation,
            "target": self.target,
            "action": self.action,
            "camera": self.camera,
            "predecessor_plan_id": self.predecessor_plan_id,
            "predicted_start_digest": self.predicted_start_digest,
            "predicted_terminal_digest": self.predicted_terminal_digest,
        }


@dataclass(frozen=True)
class _PredictionRequest:
    lease_id: str
    lease_generation: int
    target: str
    action: str
    camera: str
    predecessor_plan_id: str
    predicted_start_digest: str
    compatibility_command_id: str


def _prediction_digest(value: Any) -> str:
    bounded = _bounded_planning_value(value)
    encoded = json.dumps(
        bounded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _planning_metadata(
    prepared: Mapping[str, Any], *, plan_id: str
) -> dict[str, Any]:
    """Keep only bounded, JSON-safe planning receipt fields.

    Prepared plans may carry simulator-private trajectory arrays.  The control
    receipt needs planning provenance, not those large execution payloads.
    """

    metadata: dict[str, Any] = {"plan_id": plan_id}
    for field_name in _PLANNING_METADATA_FIELDS:
        value = _bounded_planning_value(prepared.get(field_name))
        if value is not None:
            metadata[field_name] = value
    return metadata


def _bounded_planning_value(value: Any, *, depth: int = 0) -> Any:
    """Return a small JSON-safe metadata value, dropping trajectory payloads."""

    if depth > _PLANNING_METADATA_MAX_DEPTH:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError):
            return None
        return value
    if isinstance(value, Mapping):
        bounded: dict[str, Any] = {}
        for key, item in list(value.items())[:_PLANNING_METADATA_MAX_ITEMS]:
            name = str(key)
            if (
                name.startswith("_")
                or _PLANNING_TRAJECTORY_FIELD_FRAGMENT in name.lower()
            ):
                continue
            safe_item = _bounded_planning_value(item, depth=depth + 1)
            if safe_item is not None:
                bounded[name] = safe_item
        return bounded
    if isinstance(value, (list, tuple)):
        bounded_items = []
        for item in value[:_PLANNING_METADATA_MAX_ITEMS]:
            safe_item = _bounded_planning_value(item, depth=depth + 1)
            if safe_item is not None:
                bounded_items.append(safe_item)
        return bounded_items
    return None


class BehaviorDashboardController:
    """Pipeline manual commands through one planner and one Env executor."""

    def __init__(
        self,
        *,
        state: Any,
        arbiter: BehaviorCommandArbiter,
        success_latch: BehaviorRawSuccessLatch | None = None,
        motion_available: bool = False,
        observe_available: bool = False,
        unavailable_reason: str = "controller_not_bound",
        lease_timeout_s: float = LEASE_TIMEOUT_S,
    ) -> None:
        self._state = state
        self.arbiter = arbiter
        self.success_latch = success_latch or arbiter.success_latch
        self._toolkit: Any = None
        self._lock = threading.RLock()
        self._submit_lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._commands: dict[tuple[str, int], _Command] = {}
        self._lease: _Lease | None = None
        self._lease_generation = 0
        self._head: _Command | None = None
        self._pending: list[_Command] = []
        self._planning_command: _Command | None = None
        self._prediction_planning: _PredictionRequest | None = None
        self._predicted_plans: list[_PredictedPlan] = []
        self._prediction_error: str | None = None
        self._plans_to_discard: list[str] = []
        self._last_terminal: dict[str, Any] | None = None
        self._pending_cleared_count = 0
        self._publication_failed = False
        self._active = False
        self._quiescing = False
        self._closed = False
        self._workers_stop = False
        self._policy_motion_available = bool(motion_available)
        self._policy_observe_available = bool(observe_available)
        self._motion_available = False
        self._observe_available = False
        self._unavailable_reason = str(unavailable_reason or "controller_unavailable")
        self._capabilities: dict[str, Any] = {}
        self._lease_timeout_s = max(0.1, float(lease_timeout_s))
        self._selected_camera = "head"
        self._control_revision = 0
        self._capture: dict[str, Any] = {
            "phase": "idle",
            "revision": 0,
            "error": None,
        }
        self._capture_ready_at = 0.0
        self._capture_command_id: str | None = None
        self._watchdog_stop = threading.Event()
        run_id = getattr(state, "run_id", "run")
        self._planner_worker = threading.Thread(
            target=self._planner_loop,
            name=f"behavior-dashboard-planner-{run_id}",
            daemon=True,
        )
        self._executor_worker = threading.Thread(
            target=self._executor_loop,
            name=f"behavior-dashboard-executor-{run_id}",
            daemon=True,
        )
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name=f"behavior-dashboard-control-{run_id}",
            daemon=True,
        )
        self.arbiter.add_listener(self._on_arbiter_change)
        self._planner_worker.start()
        self._executor_worker.start()
        self._watchdog.start()

    def bind_toolkit(self, toolkit: Any) -> None:
        required = (
            "dashboard_manual_command",
            "dashboard_prepare_manual_command",
            "dashboard_execute_prepared_command",
            "dashboard_discard_prepared_command",
            "dashboard_capture_views",
        )
        missing = [name for name in required if not callable(getattr(toolkit, name, None))]
        if missing:
            raise TypeError(
                "toolkit must provide Dashboard pipeline methods: "
                + ", ".join(missing)
            )
        capabilities_callback = getattr(toolkit, "dashboard_control_capabilities", None)
        capabilities: dict[str, Any] = {}
        if callable(capabilities_callback):
            reported = capabilities_callback()
            if isinstance(reported, Mapping):
                capabilities = dict(reported)
        with self._work:
            if self._closed:
                raise RuntimeError("dashboard controller is closed")
            self._toolkit = toolkit
            self._capabilities = capabilities
            self._recompute_capabilities_locked()
            reported_reason = capabilities.get("unavailable_reason")
            if reported_reason:
                self._unavailable_reason = str(reported_reason)
            elif self._motion_available or self._observe_available:
                self._unavailable_reason = ""
            self._touch_locked()
            self._work.notify_all()
        self._publish_snapshot()

    def configure_capabilities(
        self,
        *,
        motion_available: bool,
        observe_available: bool,
        unavailable_reason: str = "",
    ) -> None:
        with self._work:
            self._policy_motion_available = bool(motion_available)
            self._policy_observe_available = bool(observe_available)
            self._recompute_capabilities_locked()
            if unavailable_reason:
                self._unavailable_reason = str(unavailable_reason)
            elif self._motion_available or self._observe_available:
                self._unavailable_reason = ""
            self._touch_locked()
            self._work.notify_all()
        self._publish_snapshot()

    def activate(self) -> None:
        with self._work:
            if self._closed:
                raise RuntimeError("dashboard controller is closed")
            if self._toolkit is None:
                raise RuntimeError("dashboard controller toolkit is not bound")
            if not self.success_latch.is_bound():
                raise RuntimeError(
                    "dashboard success latch is not bound to an environment attempt"
                )
            self._active = True
            self._quiescing = False
            self._touch_locked()
            self._work.notify_all()
        self._publish_snapshot()

    def plan_only_probe(self, *, target: str, action: str) -> dict[str, Any]:
        """Plan and discard one torso/wrist calibration motion with zero action."""

        target = str(target).strip()
        action = str(action).strip()
        self._validate_target_action(target, action, "head")
        if not (
            (target == "chassis" and action in {"up", "down"})
            or (
                target in {"left_arm", "right_arm"}
                and action in {"rotate_left", "rotate_right"}
            )
        ):
            raise ControlRequestError(
                422,
                "invalid_planning_probe",
                "planning-only probes are limited to torso and wrist calibration",
            )
        lifecycle = self._state.control_admission_snapshot()
        if lifecycle["state"] != "running" or lifecycle["official_task_success"]:
            raise ControlRequestError(410, "run_finished", "run is already finished")

        command_id = f"planning-probe-{uuid.uuid4().hex}"
        with self._submit_lock:
            with self._work:
                if self.success_latch.is_latched():
                    raise ControlRequestError(
                        410, "run_finished", "official success latched"
                    )
                if not self._active or self._quiescing or self._closed:
                    raise ControlRequestError(
                        409,
                        "controller_unavailable",
                        "manual controller unavailable",
                    )
                if (
                    self._head is not None
                    or self._pending
                    or self._planning_command is not None
                    or self._capture["phase"] == "started"
                ):
                    raise ControlRequestError(
                        409, "controller_busy", "robot controller is busy"
                    )
                if self._capture["phase"] == "pending":
                    self._capture.update({"phase": "discarded", "error": None})
                    self._capture_ready_at = 0.0
                acquired, reason = self.arbiter.try_acquire_manual(command_id)
                if not acquired:
                    if reason == "official_success_latched":
                        raise ControlRequestError(
                            410, "run_finished", "official success latched"
                        )
                    raise ControlRequestError(
                        409,
                        reason or "controller_unavailable",
                        "robot controller is unavailable",
                    )
                self._touch_locked()
                self._work.notify_all()
            self._publish_snapshot()

            plan_id: str | None = None
            try:
                toolkit = self._require_toolkit()
                prepared = toolkit.dashboard_prepare_manual_command(
                    target=target,
                    action=action,
                    predecessor_plan_id=None,
                    permit_command_id=command_id,
                    background=False,
                    planning_only_probe=True,
                )
                if not isinstance(prepared, Mapping):
                    raise RuntimeError(
                        "planning-only probe returned a non-object result"
                    )
                if prepared.get("status") == "failed":
                    raise _PreparedPlanningFailure(
                        str(prepared.get("stop_reason") or "planning_error"),
                        str(prepared.get("error") or "CuRobo planning failed"),
                    )
                plan_id = str(prepared.get("plan_id") or "").strip()
                if not plan_id:
                    raise RuntimeError("planning-only probe omitted plan_id")
                discarded = toolkit.dashboard_discard_prepared_command(
                    plan_id=plan_id
                )
                if not isinstance(discarded, Mapping):
                    raise RuntimeError(
                        "planning-only probe discard returned a non-object result"
                    )
                discard_plan_id = str(discarded.get("plan_id") or "").strip()
                if discarded.get("discarded") is not True:
                    raise RuntimeError(
                        "planning-only probe discard was not acknowledged"
                    )
                if discard_plan_id != plan_id:
                    raise RuntimeError(
                        "planning-only probe discard plan_id mismatch"
                    )
                if (
                    prepared.get("planning_only_probe") is not True
                    or prepared.get("zero_action_verified") is not True
                    or int(prepared.get("env_step_delta", -1)) != 0
                ):
                    raise RuntimeError(
                        "planning-only probe omitted zero-action verification"
                    )
                result = {
                    "ok": True,
                    "planning_only": True,
                    "release_admission": False,
                    "target": target,
                    "action": action,
                    "command_id": command_id,
                    "plan_id": plan_id,
                    "env_step_delta": 0,
                    "zero_action_verified": True,
                    "discarded": True,
                    "discard_receipt": _bounded_planning_value(discarded),
                }
            except ControlRequestError:
                raise
            except BaseException as exc:
                raise ControlRequestError(
                    409,
                    "planning_probe_failed",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            finally:
                self.arbiter.release_manual(command_id)
                with self._work:
                    self._touch_locked()
                    self._work.notify_all()
                self._publish_snapshot()
        return result

    def submit(
        self,
        *,
        lease_id: str,
        sequence: int,
        target: str,
        action: str,
        camera: str,
    ) -> tuple[_Command, bool]:
        with self._submit_lock:
            return self._submit_serialized(
                lease_id=lease_id,
                sequence=sequence,
                target=target,
                action=action,
                camera=camera,
            )

    def _submit_serialized(
        self,
        *,
        lease_id: str,
        sequence: int,
        target: str,
        action: str,
        camera: str,
    ) -> tuple[_Command, bool]:
        lease_id = str(lease_id).strip()
        target = str(target).strip()
        action = str(action).strip()
        camera = str(camera).strip()
        if not lease_id or len(lease_id) > 128:
            raise ControlRequestError(422, "invalid_lease_id", "invalid lease_id")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ControlRequestError(422, "invalid_sequence", "sequence must be positive")
        self._validate_target_action(target, action, camera)
        fingerprint = (target, action, camera)
        lifecycle = self._state.control_admission_snapshot()
        if lifecycle["state"] != "running" or lifecycle["official_task_success"]:
            raise ControlRequestError(410, "run_finished", "run is already finished")

        command_id = uuid.uuid4().hex
        with self._work:
            self._expire_lease_locked(time.monotonic())
            if self.success_latch.is_latched():
                raise ControlRequestError(410, "run_finished", "official success latched")
            if not self._active or self._quiescing or self._closed:
                raise ControlRequestError(
                    409, "controller_unavailable", "manual controller unavailable"
                )
            if action == "observe" and not self._observe_available:
                raise ControlRequestError(
                    409,
                    "observe_unavailable",
                    self._unavailable_reason or "camera refresh unavailable",
                )
            if action != "observe" and not self._motion_available:
                raise ControlRequestError(
                    409,
                    "motion_unavailable",
                    self._unavailable_reason or "motion control unavailable",
                )

            new_lease = (
                self._lease is None
                or self._lease.stopped
                or self._lease.expired
            )
            if new_lease:
                self._clear_predictions_locked()
                self._prediction_error = None
                if self._head is not None or self._pending:
                    raise ControlRequestError(
                        409, "controller_busy", "another manual queue is active"
                    )
                self._reserve_for_new_head_locked(command_id)
                self._lease_generation += 1
                self._lease = _Lease(
                    lease_id=lease_id,
                    generation=self._lease_generation,
                    deadline=time.monotonic() + self._lease_timeout_s,
                    payload_fingerprint=fingerprint,
                    compatibility_command_id=command_id,
                )
            elif self._lease.lease_id != lease_id:
                raise ControlRequestError(
                    409, "controller_busy", "another manual control lease is active"
                )

            lease = self._lease
            assert lease is not None
            if lease.payload_fingerprint != fingerprint:
                raise ControlRequestError(
                    409,
                    "lease_command_conflict",
                    "a repeat lease cannot change target, action, or camera",
                )
            if (
                not new_lease
                and action not in _ONE_SHOT_ACTIONS
                and self._capabilities.get("threaded_predicted_planning")
                is not True
            ):
                raise ControlRequestError(
                    409,
                    "predicted_planning_unavailable",
                    "threaded predicted planning is unavailable",
                )
            if action in _ONE_SHOT_ACTIONS and lease.last_sequence > 0:
                raise ControlRequestError(
                    409,
                    "non_repeatable_action",
                    f"{action} does not support lease repetition",
                )
            if self._head is not None and len(self._pending) >= _QUEUE_CAPACITY:
                raise ControlRequestError(
                    409, "queue_full", "manual command queue is full"
                )
            if not new_lease and self._head is None:
                self._reserve_for_repeat_head_locked(lease, command_id)

            command = _Command(
                command_id=command_id,
                lease_id=lease_id,
                sequence=sequence,
                target=target,
                action=action,
                camera=camera,
                payload_fingerprint=fingerprint,
                initial_lease_command=new_lease,
            )
            # command_id/lease_id/sequence remain wire-compatibility metadata.
            # They intentionally do not provide idempotency: repeating an HTTP
            # request is allowed to enqueue and execute another robot command.
            self._commands[(lease_id, sequence)] = command
            lease.last_sequence = max(lease.last_sequence, sequence)
            lease.deadline = time.monotonic() + self._lease_timeout_s
            if (
                action not in _ONE_SHOT_ACTIONS
                and self._predicted_plans
                and self._prediction_matches_locked(
                    self._predicted_plans[0],
                    lease=lease,
                )
            ):
                predicted = self._predicted_plans.pop(0)
                command.plan_id = predicted.plan_id
                command.predecessor_plan_id = predicted.predecessor_plan_id
                command.predicted_start_digest = (
                    predicted.predicted_start_digest
                )
                command.predicted_terminal = dict(
                    predicted.predicted_terminal
                )
                command.predicted_terminal_digest = (
                    predicted.predicted_terminal_digest
                )
                command.planning_metadata = dict(predicted.planning_metadata)
                command.phase = "prepared"
            if self._head is None:
                self._head = command
            else:
                self._pending.append(command)
            if self._capture["phase"] == "pending":
                self._capture.update({"phase": "discarded", "error": None})
                self._capture_ready_at = 0.0
            self._selected_camera = camera
            self._touch_locked()
            command.acceptance_snapshot = self._snapshot_locked()

        try:
            self._state.on_manual_command_start(command.public())
        except BaseException as exc:
            self._fail_admission(command, exc)
            raise
        with self._work:
            command.timeline_started = True
            self._work.notify_all()
        self._publish_snapshot()
        return command, False

    def heartbeat(self, *, lease_id: str) -> dict[str, Any]:
        lease_id = str(lease_id).strip()
        with self._work:
            _, published = self._expire_lease_locked(time.monotonic())
            lease = self._lease
            inactive = bool(
                lease is None
                or lease.lease_id != lease_id
                or lease.stopped
                or lease.expired
            )
            if not inactive:
                assert lease is not None
                lease.deadline = time.monotonic() + self._lease_timeout_s
                self._touch_locked()
                result = self._snapshot_locked()
                self._work.notify_all()
        if not published:
            self.arbiter.quiesce()
        if inactive:
            raise ControlRequestError(
                409, "lease_expired", "control lease is not active"
            )
        self._publish_snapshot()
        return result

    def stop(
        self,
        *,
        lease_id: str,
        reason: str = "client_stop",
        stop_mode: str = "clear_pending",
    ) -> dict[str, Any]:
        lease_id = str(lease_id).strip()
        if str(stop_mode or "") != "clear_pending":
            raise ControlRequestError(
                422, "invalid_stop_mode", "stop_mode must be clear_pending"
            )
        with self._work:
            lease = self._lease
            if lease is None or lease.lease_id != lease_id:
                raise ControlRequestError(409, "unknown_lease", "control lease not found")
            if not lease.stopped:
                self._invalidate_lease_generation_locked(lease)
            lease.stopped = True
            lease.stop_reason = str(reason or "client_stop")
            # ``clear_pending`` preserves the lease's initial short-press step
            # through accepted, planning, prepared, and moving.  A repeat only
            # survives release after its Env execution has actually started;
            # accepted/planning/prepared repeats are still speculative tail.
            preserve_head = bool(
                self._head is not None
                and (
                    self._head.initial_lease_command
                    or self._head.phase == "moving"
                )
            )
            if preserve_head:
                cleared = self._clear_pending_locked(lease.stop_reason)
                exposed_terminal = False
            else:
                cleared, exposed_terminal = self._clear_unstarted_locked(
                    lease.stop_reason
                )
            self._clear_predictions_locked()
            release_command_id = (
                lease.compatibility_command_id
                if (
                    self._head is None
                    and self._planning_command is None
                    and self._prediction_planning is None
                )
                else None
            )
            if (
                self._head is None
                and self._planning_command is None
                and self._prediction_planning is None
                and not self.success_latch.is_latched()
            ):
                self._schedule_capture_locked()
            if release_command_id is not None:
                lease.compatibility_command_id = None
            self._touch_locked()
            self._work.notify_all()
            published = self._publish_cleared(
                cleared,
                expose_first_terminal=exposed_terminal,
            )
            if not published:
                self._mark_publication_failed_locked("cancel terminal rejected")
            result = self._snapshot_locked()
        if release_command_id is not None:
            self.arbiter.release_manual(release_command_id)
        if not published:
            self.arbiter.quiesce()
        self._publish_snapshot()
        return result

    def select_camera(self, camera: str) -> dict[str, Any]:
        camera = str(camera).strip()
        if camera not in _CAMERAS:
            raise ControlRequestError(422, "invalid_camera", "invalid camera")
        self._state.set_selected_camera(camera)
        with self._work:
            self._selected_camera = camera
            self._touch_locked()
            result = self._snapshot_locked()
        self._publish_snapshot()
        return result

    def state(self) -> dict[str, Any]:
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._work:
            changed, published = self._expire_lease_locked(time.monotonic())
            if changed:
                self._work.notify_all()
            result = self._snapshot_locked()
        if not published:
            self.arbiter.quiesce()
        return result

    def quiesce(self) -> None:
        release_command_id: str | None = None
        with self._work:
            self._quiescing = True
            self._active = False
            if self._lease is not None:
                if not self._lease.stopped:
                    self._invalidate_lease_generation_locked(self._lease)
                self._lease.stopped = True
                self._lease.stop_reason = "controller_quiescing"
            cleared, exposed_terminal = self._clear_unstarted_locked(
                "controller_quiescing"
            )
            self._clear_predictions_locked()
            if (
                self._lease is not None
                and self._head is None
                and self._planning_command is None
                and self._prediction_planning is None
            ):
                release_command_id = self._lease.compatibility_command_id
                self._lease.compatibility_command_id = None
            if self._capture["phase"] == "pending":
                self._capture.update({"phase": "discarded", "error": None})
                self._capture_ready_at = 0.0
            self._touch_locked()
            self._work.notify_all()
            published = self._publish_cleared(
                cleared,
                expose_first_terminal=exposed_terminal,
            )
            if not published:
                self._mark_publication_failed_locked("cancel terminal rejected")
        self.arbiter.quiesce()
        if release_command_id is not None:
            self.arbiter.release_manual(release_command_id)
        self._publish_snapshot()

    def drain(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._work:
            while (
                self._head is not None
                or self._pending
                or self._planning_command is not None
                or self._prediction_planning is not None
                or self._predicted_plans
                or self._capture["phase"] in {"pending", "started"}
                or self._plans_to_discard
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._work.wait(remaining)
        return self.arbiter.drain(max(0.0, deadline - time.monotonic()))

    def close(self, timeout_s: float = 10.0) -> bool:
        self.quiesce()
        drained = self.drain(timeout_s)
        if not drained:
            self._publish_snapshot()
            return False
        with self._work:
            self._closed = True
            self._workers_stop = True
            self._toolkit = None
            self._touch_locked()
            self._work.notify_all()
        self._watchdog_stop.set()
        join_s = min(1.0, max(0.0, float(timeout_s)))
        self._watchdog.join(timeout=join_s)
        self._planner_worker.join(timeout=join_s)
        self._executor_worker.join(timeout=join_s)
        self.arbiter.remove_listener(self._on_arbiter_change)
        self._publish_snapshot()
        return True

    def _planner_loop(self) -> None:
        while True:
            discard_plan: str | None = None
            command: _Command | None = None
            prediction: _PredictionRequest | None = None
            with self._work:
                while True:
                    if self._workers_stop:
                        return
                    if self._plans_to_discard:
                        discard_plan = self._plans_to_discard.pop(0)
                        break
                    command = self._next_unplanned_locked()
                    if command is not None:
                        self._planning_command = command
                        command.phase = "planning"
                        self._touch_locked()
                        break
                    prediction = self._next_prediction_request_locked()
                    if prediction is not None:
                        self._prediction_planning = prediction
                        self._touch_locked()
                        break
                    timeout = self._prediction_wait_timeout_locked()
                    self._work.wait(timeout)
            if discard_plan is not None:
                self._discard_plan(discard_plan)
                with self._work:
                    self._work.notify_all()
                continue
            assert command is not None or prediction is not None
            self._publish_snapshot()
            plan_id: str | None = None
            try:
                toolkit = self._require_toolkit()
                target = command.target if command is not None else prediction.target
                action = command.action if command is not None else prediction.action
                predecessor_plan_id = (
                    command.predecessor_plan_id
                    if command is not None
                    else prediction.predecessor_plan_id
                )
                compatibility_command_id = (
                    command.command_id
                    if command is not None
                    else prediction.compatibility_command_id
                )
                prepared = toolkit.dashboard_prepare_manual_command(
                    target=target,
                    action=action,
                    predecessor_plan_id=predecessor_plan_id,
                    permit_command_id=compatibility_command_id,
                    background=predecessor_plan_id is not None,
                )
                if not isinstance(prepared, Mapping):
                    raise RuntimeError("manual planner returned a non-object result")
                if prepared.get("status") == "failed":
                    raise _PreparedPlanningFailure(
                        str(prepared.get("stop_reason") or "planning_error"),
                        str(
                            prepared.get("error")
                            or "CuRobo planning failed"
                        ),
                    )
                if prepared.get("status") not in (None, "prepared"):
                    raise RuntimeError(
                        "manual planner returned an invalid status"
                    )
                plan_id = str(prepared.get("plan_id") or "").strip()
                if not plan_id:
                    raise RuntimeError("manual planner did not return plan_id")
                if prediction is not None and not isinstance(
                    _bounded_planning_value(prepared.get("predicted_terminal")),
                    dict,
                ):
                    raise RuntimeError(
                        "predicted plan omitted a bounded predicted_terminal"
                    )
            except BaseException as exc:
                if plan_id:
                    self._discard_plan(plan_id)
                if command is not None:
                    self._finish_planning_failure(command, exc)
                else:
                    assert prediction is not None
                    self._finish_prediction_failure(prediction, exc)
                continue
            discard_after_prepare = False
            with self._work:
                if command is not None:
                    if self._planning_command is command:
                        self._planning_command = None
                    if not self._command_is_active_locked(command):
                        discard_after_prepare = True
                    else:
                        command.plan_id = plan_id
                        command.planning_metadata = _planning_metadata(
                            prepared, plan_id=plan_id
                        )
                        predicted_terminal = _bounded_planning_value(
                            prepared.get("predicted_terminal")
                        )
                        if isinstance(predicted_terminal, dict):
                            command.predicted_terminal = predicted_terminal
                            command.predicted_terminal_digest = (
                                _prediction_digest(predicted_terminal)
                            )
                        predicted_start_digest = prepared.get(
                            "predicted_start_digest"
                        )
                        if isinstance(predicted_start_digest, str):
                            command.predicted_start_digest = (
                                predicted_start_digest
                            )
                        command.phase = "prepared"
                else:
                    assert prediction is not None
                    if self._prediction_planning == prediction:
                        self._prediction_planning = None
                    if not self._prediction_request_is_active_locked(prediction):
                        discard_after_prepare = True
                    else:
                        predicted_terminal = _bounded_planning_value(
                            prepared.get("predicted_terminal")
                        )
                        assert isinstance(predicted_terminal, dict)
                        prepared_start_digest = prepared.get(
                            "predicted_start_digest"
                        )
                        self._predicted_plans.append(
                            _PredictedPlan(
                                plan_id=plan_id,
                                lease_id=prediction.lease_id,
                                lease_generation=(
                                    prediction.lease_generation
                                ),
                                target=prediction.target,
                                action=prediction.action,
                                camera=prediction.camera,
                                predecessor_plan_id=(
                                    prediction.predecessor_plan_id
                                ),
                                predicted_start_digest=(
                                    prepared_start_digest
                                    if isinstance(prepared_start_digest, str)
                                    and prepared_start_digest
                                    else prediction.predicted_start_digest
                                ),
                                predicted_terminal=predicted_terminal,
                                predicted_terminal_digest=(
                                    _prediction_digest(predicted_terminal)
                                ),
                                planning_metadata=_planning_metadata(
                                    prepared,
                                    plan_id=plan_id,
                                ),
                            )
                        )
                self._touch_locked()
                self._work.notify_all()
            if discard_after_prepare:
                self._discard_plan(plan_id)
                self._release_stopped_idle_lease()
            self._publish_snapshot()

    def _executor_loop(self) -> None:
        while True:
            command: _Command | None = None
            capture: tuple[int, str] | None = None
            with self._work:
                while True:
                    if self._workers_stop:
                        return
                    head = self._head
                    if head is not None and (
                        self._command_has_manual_permit_locked(head)
                        and head.timeline_started
                        and (
                            head.action in _ONE_SHOT_ACTIONS
                            or head.plan_id
                        )
                    ):
                        command = head
                        command.phase = "moving"
                        lease = self._lease
                        if (
                            lease is not None
                            and lease.lease_id == command.lease_id
                        ):
                            # A later executed step makes any capture from an
                            # earlier step non-terminal for this lease.
                            lease.terminal_capture_requested = False
                        self._touch_locked()
                        break
                    capture = self._start_capture_if_ready_locked()
                    if capture is not None:
                        break
                    timeout = None
                    if self._capture["phase"] == "pending":
                        timeout = max(
                            0.05, self._capture_ready_at - time.monotonic()
                        )
                    self._work.wait(timeout)
            self._publish_snapshot()
            if command is not None:
                self._execute_command(command)
            elif capture is not None:
                self._execute_capture(*capture)

    def _execute_command(self, command: _Command) -> None:
        started = time.monotonic()
        success_latched = False
        manual_success_binding: dict[str, Any] | None = None
        try:
            toolkit = self._require_toolkit()
            with self.arbiter.manual_execution(command.command_id):
                if command.action in _ONE_SHOT_ACTIONS:
                    payload = toolkit.dashboard_manual_command(
                        target=command.target,
                        action=command.action,
                        camera=command.camera,
                        permit_command_id=command.command_id,
                    )
                else:
                    payload = toolkit.dashboard_execute_prepared_command(
                        plan_id=str(command.plan_id or ""),
                        command_id=command.command_id,
                    )
                if not isinstance(payload, Mapping):
                    raise RuntimeError(
                        "manual executor returned a non-object result"
                    )
                result = dict(payload)
                success_latched, manual_success_binding = (
                    self.success_latch.observe_with_binding(result)
                )
        except BaseException as exc:
            result = {
                "primitive_success": False,
                "stop_reason": "tool_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            success_latched = self.success_latch.is_latched()
            manual_success_binding = None
        self._attach_planning_metadata(command, result)
        result.setdefault("elapsed_s", max(0.0, time.monotonic() - started))
        capture_result = self._detach_capture_payload(result)
        with self._work:
            suppress_hold_capture = bool(
                self._lease is not None
                and self._lease.lease_id == command.lease_id
                and self._lease.hold_armed
                and not self.success_latch.is_latched()
            )
        if manual_success_binding is not None:
            result["task_success"] = True
            result.setdefault("stop_reason", "official_task_success")
        elif result.get("task_success") is True:
            result.update(
                {
                    "primitive_success": False,
                    "task_success": False,
                    "stop_reason": "invalid_success_receipt",
                    "error": (
                        "manual result claimed task success without a valid "
                        "raw-success receipt"
                    ),
                }
            )
        lifecycle_terminal = str(result.get("stop_reason") or "") in {
            "environment_terminated",
            "environment_truncated",
        }
        failed = self._action_failed(result)
        command.result = result
        command.phase = "failed" if failed else "completed"
        if success_latched:
            command.phase = "completed" if not failed else "failed"

        timeline_published, timeline_error = self._publish_manual_result(
            command,
            result,
            official_success_latched=manual_success_binding is not None,
        )
        if not timeline_published:
            result["state_publish_error"] = str(timeline_error)
            command.phase = "failed"
        embedded_terminal_capture = bool(
            manual_success_binding is not None or lifecycle_terminal
        )
        if capture_result is not None and (
            embedded_terminal_capture or not suppress_hold_capture
        ):
            revision = self._new_capture_revision()
            publish_error = self._publish_capture_result(
                revision, capture_result
            )
            with self._work:
                capture_error = (
                    publish_error
                    or capture_result.get("capture_error")
                    or capture_result.get("error")
                )
                self._capture.update(
                    {
                        "phase": "failed" if capture_error else "completed",
                        "revision": revision,
                        "error": (
                            str(capture_error) if capture_error else None
                        ),
                    }
                )
                self._touch_locked()
        elif embedded_terminal_capture:
            # Lifecycle terminals forbid a follow-up capture RPC.  Their one
            # terminal capture must be embedded in the same Env action response.
            with self._work:
                revision = int(self._capture["revision"]) + 1
                self._capture.update(
                    {
                        "phase": "failed",
                        "revision": revision,
                        "error": "terminal_capture_missing_from_terminal_action",
                    }
                )
                self._capture_ready_at = 0.0
                if self._lease is not None:
                    self._lease.terminal_capture_requested = True
                self._touch_locked()

        with self._work:
            if self._head is not command:
                return
            cleared: list[_Command] = []
            self._last_terminal = self._terminal_public(command)
            self._head = None
            lease = self._lease
            if success_latched:
                self._active = False
                if lease is not None:
                    if not lease.stopped:
                        self._invalidate_lease_generation_locked(lease)
                    lease.stopped = True
                    lease.stop_reason = "official_task_success"
                cleared = self._clear_pending_locked("official_task_success")
                self._clear_predictions_locked()
                if self._capture["phase"] == "pending":
                    self._capture.update({"phase": "discarded", "error": None})
                    self._capture_ready_at = 0.0
            elif failed:
                if lease is not None:
                    if not lease.stopped:
                        self._invalidate_lease_generation_locked(lease)
                    lease.stopped = True
                    lease.stop_reason = str(
                        result.get("stop_reason") or "command_error"
                    )
                cleared = self._clear_pending_locked(
                    str(result.get("stop_reason") or "command_error")
                )
                self._clear_predictions_locked()
            elif self._pending:
                self._head = self._pending.pop(0)
            elif lease is not None and command.action in _ONE_SHOT_ACTIONS:
                if not lease.stopped:
                    self._invalidate_lease_generation_locked(lease)
                lease.stopped = True
                lease.stop_reason = "command_complete"
            if lease is not None and not failed and command.plan_id:
                lease.executed_count += 1
                lease.last_executed_plan_id = command.plan_id

            next_head = self._head
            self._touch_locked()
            self._work.notify_all()

        # Publish the current terminal and every derived cancellation before
        # advancing the controller-visible queue.
        terminal_published = timeline_published and self._publish_snapshot()
        if not terminal_published:
            self._fail_closed_publication(
                owner_command_id=command.command_id,
                error=timeline_error or "control terminal snapshot was rejected",
            )
            return
        if cleared and not self._publish_cleared(cleared):
            self._fail_closed_publication(
                owner_command_id=command.command_id,
                error="queued cancellation terminal was rejected",
            )
            return

        if next_head is not None:
            with self._work:
                if (
                    self._head is not next_head
                    or next_head.phase == "cancelled"
                ):
                    next_head = None
                    handed_off = False
                else:
                    handed_off = self.arbiter.handoff_manual(
                        command.command_id, next_head.command_id
                    )
                    if handed_off and lease is not None:
                        lease.compatibility_command_id = next_head.command_id
        if next_head is not None:
            if not handed_off:
                arbiter_snapshot = self.arbiter.snapshot()
                reason = (
                    "official_task_success"
                    if arbiter_snapshot.get("success_latched") is True
                    else "controller_quiescing"
                    if arbiter_snapshot.get("quiescing") is True
                    else "controller_unavailable"
                )
                with self._work:
                    preempted: list[_Command] = []
                    if self._head is next_head:
                        preempted = [next_head, *self._pending]
                        self._head = None
                        self._pending.clear()
                        self._mark_cleared_locked(preempted, reason)
                        if lease is not None:
                            if not lease.stopped:
                                self._invalidate_lease_generation_locked(lease)
                            lease.stopped = True
                            lease.stop_reason = reason
                            lease.compatibility_command_id = None
                        self._clear_predictions_locked()
                        self._touch_locked()
                        self._work.notify_all()
                if preempted and not self._publish_cleared(
                    preempted,
                    expose_first_terminal=False,
                ):
                    self._fail_closed_publication(
                        owner_command_id=command.command_id,
                        error="preempted cancellation terminal was rejected",
                    )
                    return
                self.arbiter.release_manual(command.command_id)
        else:
            retain_hold_permit = bool(
                lease is not None
                and not lease.stopped
                and not lease.expired
                and command.action not in _ONE_SHOT_ACTIONS
                and not success_latched
                and not failed
            )
            if retain_hold_permit:
                with self._work:
                    lease.compatibility_command_id = command.command_id
                    self._work.notify_all()
            else:
                self.arbiter.release_manual(command.command_id)
                if lease is not None:
                    with self._work:
                        if lease.compatibility_command_id == command.command_id:
                            lease.compatibility_command_id = None
            with self._work:
                if (
                    not success_latched
                    and not lifecycle_terminal
                    and command.action != "observe"
                    and result.get("cancelled_before_execution") is not True
                    and not self._quiescing
                    and not self._closed
                    and not self._publication_failed
                    and (
                        failed
                        or command.action in _ONE_SHOT_ACTIONS
                        or lease is None
                        or lease.stopped
                    )
                ):
                    self._schedule_capture_locked()
                self._touch_locked()
                self._work.notify_all()
        self._publish_snapshot()

    def _execute_capture(self, revision: int, command_id: str) -> None:
        try:
            with self.arbiter.manual_execution(command_id):
                payload = self._require_toolkit().dashboard_capture_views(
                    command_id=command_id
                )
            if not isinstance(payload, Mapping):
                raise RuntimeError("Dashboard capture returned a non-object result")
            result = dict(payload)
        except BaseException as exc:
            result = {"capture_error": f"{type(exc).__name__}: {exc}"}
        publish_error = self._publish_capture_result(revision, result)
        with self._work:
            error = (
                publish_error
                or result.get("capture_error")
                or result.get("error")
            )
            self._capture.update(
                {
                    "phase": "failed" if error else "completed",
                    "revision": revision,
                    "error": str(error) if error else None,
                }
            )
            next_head = self._head
            self._capture_command_id = None
            self._touch_locked()
            self._work.notify_all()

        # Capture completion/failure is visible before the deferred motion.
        if not self._publish_snapshot():
            self._fail_closed_publication(
                owner_command_id=command_id,
                error="capture terminal snapshot was rejected",
            )
            return
        if next_head is not None:
            with self._work:
                if (
                    self._head is not next_head
                    or next_head.phase == "cancelled"
                ):
                    next_head = None
                    handed_off = False
                else:
                    handed_off = self.arbiter.handoff_manual(
                        command_id, next_head.command_id
                    )
                    if handed_off and self._lease is not None:
                        self._lease.compatibility_command_id = next_head.command_id
        if next_head is not None:
            if not handed_off:
                arbiter_snapshot = self.arbiter.snapshot()
                reason = (
                    "official_task_success"
                    if arbiter_snapshot.get("success_latched") is True
                    else "controller_quiescing"
                    if arbiter_snapshot.get("quiescing") is True
                    else "controller_unavailable"
                )
                with self._work:
                    cleared: list[_Command] = []
                    if self._head is next_head:
                        cleared = [next_head, *self._pending]
                        self._head = None
                        self._pending.clear()
                        self._mark_cleared_locked(cleared, reason)
                        if self._lease is not None:
                            if not self._lease.stopped:
                                self._invalidate_lease_generation_locked(
                                    self._lease
                                )
                            self._lease.stopped = True
                            self._lease.stop_reason = reason
                            self._lease.compatibility_command_id = None
                        self._clear_predictions_locked()
                        self._touch_locked()
                        self._work.notify_all()
                if cleared and not self._publish_cleared(
                    cleared,
                    expose_first_terminal=False,
                ):
                    self._fail_closed_publication(
                        owner_command_id=command_id,
                        error="deferred cancellation terminal was rejected",
                    )
                    return
                self.arbiter.release_manual(command_id)
        else:
            self.arbiter.release_manual(command_id)
        self._publish_snapshot()

    def _publish_capture_result(
        self, revision: int, result: Mapping[str, Any]
    ) -> str | None:
        callback = getattr(self._state, "on_dashboard_capture_result", None)
        if not callable(callback):
            return "Dashboard capture State callback is unavailable"
        try:
            accepted = callback(
                result,
                controller=self,
                generation=revision,
            )
            if accepted is not True:
                return "Dashboard capture result was rejected"
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._work:
                if self._capture.get("revision") == revision:
                    self._capture.update(
                        {
                            "phase": "failed",
                            "error": error,
                        }
                    )
                    self._touch_locked()
            return error
        return None

    def _finish_planning_failure(
        self, command: _Command, exc: BaseException
    ) -> None:
        stop_reason = self._planning_failure_reason(exc)
        result = {
            "primitive_success": False,
            "stop_reason": stop_reason,
            "error": self._planning_error_text(exc),
        }
        release_command_id: str | None = None
        inactive = False
        with self._work:
            if self._planning_command is command:
                self._planning_command = None
            if not self._command_is_active_locked(command):
                inactive = True
                self._work.notify_all()
            else:
                command.phase = "failed"
                command.result = result
                cleared = []
                if self._head is command:
                    self._head = None
                    self._last_terminal = self._terminal_public(command)
                    cleared = self._clear_pending_locked(stop_reason)
                    if self._lease is not None:
                        if not self._lease.stopped:
                            self._invalidate_lease_generation_locked(
                                self._lease
                            )
                        self._lease.stopped = True
                        self._lease.stop_reason = stop_reason
                    self._lease.compatibility_command_id = None
                    release_command_id = command.command_id
                else:
                    index = self._pending.index(command)
                    cleared = self._pending[index + 1 :]
                    del self._pending[index:]
                    self._pending_cleared_count += 1
                    self._mark_cleared_locked(cleared, stop_reason)
                    if self._lease is not None:
                        if not self._lease.stopped:
                            self._invalidate_lease_generation_locked(
                                self._lease
                            )
                        self._lease.stopped = True
                        self._lease.stop_reason = stop_reason
                self._clear_predictions_locked()
                if not self.success_latch.is_latched():
                    self._schedule_capture_locked()
                self._touch_locked()
                self._work.notify_all()
        if inactive:
            self._release_stopped_idle_lease()
            return
        timeline_published, timeline_error = self._publish_manual_result(
            command,
            result,
            official_success_latched=False,
        )
        # State and control readers must see the immutable planning terminal
        # before another execution is published.
        terminal_published = timeline_published and self._publish_snapshot()
        cleared = [item for item in cleared if item is not command]
        if terminal_published and cleared:
            terminal_published = self._publish_cleared(cleared)
        if not terminal_published:
            if release_command_id is not None:
                self._fail_closed_publication(
                    owner_command_id=release_command_id,
                    error=timeline_error
                    or "planning terminal snapshot was rejected",
                )
            else:
                with self._work:
                    self._mark_publication_failed_locked(
                        timeline_error
                        or "planning terminal snapshot was rejected"
                    )
                self.arbiter.quiesce()
            return
        if release_command_id is not None:
            self.arbiter.release_manual(release_command_id)
        self._publish_snapshot()

    def _finish_prediction_failure(
        self,
        prediction: _PredictionRequest,
        exc: BaseException,
    ) -> None:
        stop_reason = self._planning_failure_reason(exc)
        release_command_id: str | None = None
        already_stopped = False
        with self._work:
            if self._prediction_planning == prediction:
                self._prediction_planning = None
            if not self._prediction_request_is_active_locked(prediction):
                already_stopped = True
                self._work.notify_all()
                published = True
            else:
                self._prediction_error = self._planning_error_text(exc)
                lease = self._lease
                assert lease is not None
                self._invalidate_lease_generation_locked(lease)
                lease.stopped = True
                lease.stop_reason = stop_reason
                cleared, exposed_terminal = self._clear_unstarted_locked(
                    stop_reason
                )
                self._clear_predictions_locked()
                if self._head is None:
                    release_command_id = lease.compatibility_command_id
                    lease.compatibility_command_id = None
                    if not self.success_latch.is_latched():
                        self._schedule_capture_locked()
                self._touch_locked()
                self._work.notify_all()
                published = self._publish_cleared(
                    cleared,
                    expose_first_terminal=exposed_terminal,
                )
        if already_stopped:
            self._release_stopped_idle_lease()
            return
        if release_command_id is not None:
            self.arbiter.release_manual(release_command_id)
        if not published:
            self.arbiter.quiesce()
        self._publish_snapshot()

    @staticmethod
    def _planning_failure_reason(exc: BaseException) -> str:
        for attribute in ("stop_reason", "code"):
            value = getattr(exc, attribute, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        message = str(exc).lower()
        for reason in (
            "unreachable",
            "solver_timeout",
            "goal_not_converged",
            "joint_limits",
            "numerical_error",
        ):
            if reason in message:
                return reason
        return "planning_error"

    @staticmethod
    def _planning_error_text(exc: BaseException) -> str:
        if isinstance(exc, _PreparedPlanningFailure):
            return str(exc)
        return f"{type(exc).__name__}: {exc}"

    def _fail_admission(self, command: _Command, exc: BaseException) -> None:
        result = {
            "primitive_success": False,
            "stop_reason": "dashboard_state_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        release_command_id: str | None = None
        with self._work:
            command.phase = "failed"
            command.result = result
            if self._head is command:
                self._head = None
                release_command_id = command.command_id
            elif command in self._pending:
                self._pending.remove(command)
            if self._lease is not None:
                if not self._lease.stopped:
                    self._invalidate_lease_generation_locked(self._lease)
                self._lease.stopped = True
                self._lease.stop_reason = "dashboard_state_error"
            self._last_terminal = self._terminal_public(command)
            self._touch_locked()
            self._work.notify_all()
        published = self._publish_snapshot()
        if not published:
            if release_command_id is not None:
                self._fail_closed_publication(
                    owner_command_id=release_command_id,
                    error="admission failure terminal snapshot was rejected",
                )
            else:
                with self._work:
                    self._mark_publication_failed_locked(
                        "admission failure terminal snapshot was rejected"
                    )
                self.arbiter.quiesce()
            return
        if release_command_id is not None:
            self.arbiter.release_manual(release_command_id)

    def _reserve_for_new_head_locked(self, command_id: str) -> None:
        arbiter = self.arbiter.snapshot()
        if (
            self._capture["phase"] == "started"
            and self._capture_command_id
            and arbiter.get("owner") == "manual"
            and arbiter.get("command_id") == self._capture_command_id
        ):
            # The already-started capture cannot be cancelled honestly.  Admit
            # one deferred head; the single executor runs it after capture.
            return
        acquired, reason = self.arbiter.try_acquire_manual(command_id)
        if acquired:
            return
        if reason == "official_success_latched":
            raise ControlRequestError(410, "run_finished", "official success latched")
        raise ControlRequestError(
            409,
            reason or "controller_unavailable",
            "robot controller is unavailable",
        )

    def _reserve_for_repeat_head_locked(
        self,
        lease: _Lease,
        command_id: str,
    ) -> None:
        """Update the compatibility command context for an idle hold."""

        self._reserve_for_new_head_locked(command_id)
        lease.compatibility_command_id = command_id

    def _next_unplanned_locked(self) -> _Command | None:
        command = self._head
        if (
            command is None
            or not command.timeline_started
            or command.action in _ONE_SHOT_ACTIONS
            or command.plan_id
        ):
            return None
        if not self._command_has_manual_permit_locked(command):
            return None
        lease = self._lease
        command.predecessor_plan_id = (
            lease.last_executed_plan_id
            if lease is not None and lease.executed_count > 0
            else None
        )
        return command

    def _command_has_manual_permit_locked(self, _command: _Command) -> bool:
        return bool(
            self._active
            and not self._quiescing
            and not self._closed
            and not self.success_latch.is_latched()
        )

    def _prediction_wait_timeout_locked(self) -> float | None:
        lease = self._lease
        if (
            lease is None
            or lease.stopped
            or lease.expired
            or lease.payload_fingerprint is None
            or lease.payload_fingerprint[1] in _ONE_SHOT_ACTIONS
            or self._capabilities.get("threaded_predicted_planning") is not True
            or len(self._predicted_plans) >= DASHBOARD_PREDICTED_PLAN_DEPTH
        ):
            return None
        remaining = (
            lease.accepted_at + DASHBOARD_HOLD_ARM_DELAY_S - time.monotonic()
        )
        return max(0.01, remaining) if remaining > 0.0 else None

    def _next_prediction_request_locked(
        self,
    ) -> _PredictionRequest | None:
        lease = self._lease
        if (
            lease is None
            or lease.stopped
            or lease.expired
            or lease.payload_fingerprint is None
            or lease.payload_fingerprint[1] in _ONE_SHOT_ACTIONS
            or self._capabilities.get("threaded_predicted_planning") is not True
            or len(self._predicted_plans) >= DASHBOARD_PREDICTED_PLAN_DEPTH
            or self._prediction_planning is not None
            or time.monotonic()
            < lease.accepted_at + DASHBOARD_HOLD_ARM_DELAY_S
        ):
            return None
        compatibility_command_id = str(
            lease.compatibility_command_id or ""
        )
        if not compatibility_command_id:
            return None
        predecessor_plan_id: str | None = None
        predicted_start: Any = None
        if self._predicted_plans:
            tail = self._predicted_plans[-1]
            predecessor_plan_id = tail.plan_id
            predicted_start = tail.predicted_terminal
        else:
            candidates = [*self._pending]
            if self._head is not None:
                candidates.insert(0, self._head)
            planned = [item for item in candidates if item.plan_id]
            if planned:
                predecessor_plan_id = planned[-1].plan_id
                predicted_start = planned[-1].planning_metadata.get(
                    "predicted_terminal"
                )
            elif lease.last_executed_plan_id is not None:
                predecessor_plan_id = lease.last_executed_plan_id
        if predecessor_plan_id is None:
            return None
        lease.hold_armed = True
        target, action, camera = lease.payload_fingerprint
        return _PredictionRequest(
            lease_id=lease.lease_id,
            lease_generation=lease.generation,
            target=target,
            action=action,
            camera=camera,
            predecessor_plan_id=predecessor_plan_id,
            predicted_start_digest=_prediction_digest(
                predicted_start
                if predicted_start is not None
                else {"predecessor_plan_id": predecessor_plan_id}
            ),
            compatibility_command_id=compatibility_command_id,
        )

    def _prediction_request_is_active_locked(
        self,
        prediction: _PredictionRequest,
    ) -> bool:
        lease = self._lease
        return bool(
            lease is not None
            and lease.lease_id == prediction.lease_id
            and lease.generation == prediction.lease_generation
            and not lease.stopped
            and not lease.expired
            and lease.payload_fingerprint
            == (
                prediction.target,
                prediction.action,
                prediction.camera,
            )
            and len(self._predicted_plans)
            < DASHBOARD_PREDICTED_PLAN_DEPTH
        )

    @staticmethod
    def _prediction_matches_locked(
        prediction: _PredictedPlan,
        *,
        lease: _Lease,
    ) -> bool:
        return bool(
            lease.payload_fingerprint is not None
            and prediction.lease_id == lease.lease_id
            and prediction.lease_generation == lease.generation
            and (
                prediction.target,
                prediction.action,
                prediction.camera,
            )
            == lease.payload_fingerprint
        )

    def _start_capture_if_ready_locked(self) -> tuple[int, str] | None:
        if (
            self._capture["phase"] != "pending"
            or self._head is not None
            or time.monotonic() < self._capture_ready_at
            or self.success_latch.is_latched()
            or self._quiescing
            or self._closed
        ):
            return None
        command_id = f"capture-{uuid.uuid4().hex}"
        acquired, _ = self.arbiter.try_acquire_manual(command_id)
        if not acquired:
            return None
        revision = int(self._capture["revision"])
        self._capture_command_id = command_id
        self._capture.update({"phase": "started", "error": None})
        self._touch_locked()
        return revision, command_id

    def _schedule_capture_locked(self) -> None:
        lease = self._lease
        if lease is not None and lease.terminal_capture_requested:
            return
        if self._capture["phase"] in {"pending", "started"}:
            return
        if lease is not None:
            lease.terminal_capture_requested = True
        revision = int(self._capture["revision"]) + 1
        self._capture = {
            "phase": "pending",
            "revision": revision,
            "error": None,
        }
        self._capture_ready_at = time.monotonic() + _CAPTURE_IDLE_DELAY_S

    def _new_capture_revision(self) -> int:
        with self._work:
            if self._lease is not None:
                self._lease.terminal_capture_requested = True
            revision = int(self._capture["revision"]) + 1
            self._capture.update(
                {"phase": "started", "revision": revision, "error": None}
            )
            self._touch_locked()
            return revision

    def _detach_capture_payload(
        self, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        terminal_capture = result.pop("terminal_capture", None)
        terminal_capture_error = result.pop("terminal_capture_error", None)
        if isinstance(terminal_capture, Mapping):
            capture = dict(terminal_capture)
            if terminal_capture_error not in (None, "", False):
                capture.setdefault(
                    "capture_error",
                    str(terminal_capture_error),
                )
            return capture
        if terminal_capture_error not in (None, "", False):
            return {"capture_error": str(terminal_capture_error)}
        capture_keys = {
            "_frames_bytes",
            "capture_group_id",
            "simulator_step",
            "capture_error",
        }
        if not any(key in result for key in capture_keys):
            return None
        capture = {
            key: result.pop(key)
            for key in tuple(capture_keys)
            if key in result
        }
        return capture

    def _clear_pending_locked(self, reason: str) -> list[_Command]:
        cleared = list(self._pending)
        self._pending.clear()
        self._mark_cleared_locked(cleared, reason)
        return cleared

    def _clear_unstarted_locked(
        self, reason: str
    ) -> tuple[list[_Command], bool]:
        """Cancel every command that has not begun its Env execution."""

        unstarted_head = bool(
            self._head is not None and self._head.phase != "moving"
        )
        if not unstarted_head:
            return self._clear_pending_locked(reason), False
        assert self._head is not None
        cleared = [self._head, *self._pending]
        self._head = None
        self._pending.clear()
        self._mark_cleared_locked(cleared, reason)
        return cleared, True

    def _clear_predictions_locked(self) -> None:
        for prediction in self._predicted_plans:
            self._plans_to_discard.append(prediction.plan_id)
        self._predicted_plans.clear()

    def _release_stopped_idle_lease(self) -> None:
        release_command_id: str | None = None
        with self._work:
            lease = self._lease
            if (
                lease is not None
                and lease.stopped
                and self._head is None
                and self._planning_command is None
                and self._prediction_planning is None
            ):
                release_command_id = lease.compatibility_command_id
                lease.compatibility_command_id = None
                if (
                    self._capture["phase"] not in {"pending", "started"}
                    and not self.success_latch.is_latched()
                ):
                    self._schedule_capture_locked()
                self._touch_locked()
                self._work.notify_all()
        if release_command_id is not None:
            self.arbiter.release_manual(release_command_id)

    def _mark_cleared_locked(
        self, commands: list[_Command], reason: str
    ) -> None:
        for command in commands:
            if command.plan_id:
                self._plans_to_discard.append(command.plan_id)
            command.phase = "cancelled"
            command.result = {
                "primitive_success": False,
                "stop_reason": reason,
                "cancelled_before_execution": True,
            }
        self._pending_cleared_count += len(commands)

    def _publish_cleared(
        self,
        commands: list[_Command],
        *,
        expose_first_terminal: bool = False,
    ) -> bool:
        for command in commands:
            if not command.timeline_started:
                continue
            try:
                accepted = self._state.on_manual_command_result(
                    command.public(),
                    dict(command.result or {}),
                    official_success_latched=False,
                )
                if accepted is not True:
                    return False
            except BaseException:
                return False
        if expose_first_terminal and commands:
            with self._work:
                self._last_terminal = self._terminal_public(commands[0])
                self._touch_locked()
        return self._publish_snapshot()

    def _publish_manual_result(
        self,
        command: _Command,
        result: Mapping[str, Any],
        *,
        official_success_latched: bool,
    ) -> tuple[bool, str | None]:
        try:
            accepted = self._state.on_manual_command_result(
                command.public(),
                dict(result),
                official_success_latched=official_success_latched,
            )
        except BaseException as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if accepted is not True:
            return False, "Dashboard manual terminal was rejected"
        return True, None

    def _mark_publication_failed_locked(self, _error: str) -> None:
        self._publication_failed = True
        self._active = False
        self._quiescing = True
        self._unavailable_reason = "dashboard_state_publication_failed"
        if self._lease is not None:
            if not self._lease.stopped:
                self._invalidate_lease_generation_locked(self._lease)
            self._lease.stopped = True
            self._lease.stop_reason = "dashboard_state_publication_failed"
        self._touch_locked()

    def _fail_closed_publication(
        self,
        *,
        owner_command_id: str,
        error: str,
    ) -> None:
        """Disable admission when the immutable State publication fails."""

        with self._work:
            remaining: list[_Command] = []
            if self._head is not None:
                remaining.append(self._head)
                self._head = None
            remaining.extend(self._pending)
            self._pending.clear()
            self._mark_cleared_locked(
                remaining,
                "dashboard_state_publication_failed",
            )
            self._mark_publication_failed_locked(error)
            self._work.notify_all()
        # Quiescing rejects future work before cleanup releases execution.
        self.arbiter.quiesce()
        self.arbiter.release_manual(owner_command_id)
        self._publish_snapshot()

    def _discard_plan(self, plan_id: str) -> None:
        try:
            toolkit = self._require_toolkit()
            toolkit.dashboard_discard_prepared_command(plan_id=plan_id)
        except BaseException:
            pass

    def _command_is_active_locked(self, command: _Command) -> bool:
        return self._head is command or command in self._pending

    @staticmethod
    def _action_failed(result: Mapping[str, Any]) -> bool:
        return bool(
            result.get("primitive_success") is False
            or result.get("success") is False
            or result.get("terminated") is True
            or result.get("truncated") is True
            or result.get("error") not in (None, "", False)
        )

    @staticmethod
    def _attach_planning_metadata(
        command: _Command, result: dict[str, Any]
    ) -> None:
        """Publish bounded planning receipt metadata with the action terminal."""

        if not command.planning_metadata:
            return
        existing = result.get("metrics")
        metrics = dict(existing) if isinstance(existing, Mapping) else {}
        metrics.update(command.planning_metadata)
        result["metrics"] = metrics

    @staticmethod
    def _terminal_public(command: _Command) -> dict[str, Any]:
        terminal = command.public()
        result = command.result
        if isinstance(result, Mapping):
            terminal.update(
                {
                    "error": result.get("error"),
                    "stop_reason": result.get("stop_reason"),
                    "primitive_success": result.get("primitive_success"),
                    "task_success": result.get("task_success"),
                }
            )
        return terminal

    def _require_toolkit(self) -> Any:
        with self._lock:
            toolkit = self._toolkit
        if toolkit is None:
            raise RuntimeError("dashboard controller toolkit is not bound")
        return toolkit

    def _validate_target_action(self, target: str, action: str, camera: str) -> None:
        if target not in _TARGETS:
            raise ControlRequestError(422, "invalid_target", "invalid control target")
        if action not in _ACTIONS:
            raise ControlRequestError(422, "invalid_action", "invalid control action")
        if camera not in _CAMERAS:
            raise ControlRequestError(422, "invalid_camera", "invalid camera")
        if target == "chassis" and action not in _CHASSIS_ACTIONS:
            raise ControlRequestError(
                422,
                "invalid_target_action",
                f"{action} is available for arm control only",
            )

    def _expire_lease_locked(self, now: float) -> tuple[bool, bool]:
        lease = self._lease
        if (
            lease is None
            or lease.stopped
            or lease.expired
            or now <= lease.deadline
        ):
            return False, True
        lease.expired = True
        self._invalidate_lease_generation_locked(lease)
        lease.stopped = True
        lease.stop_reason = "lease_expired"
        cleared, exposed_terminal = self._clear_unstarted_locked(
            "lease_expired"
        )
        self._clear_predictions_locked()
        release_command_id = (
            lease.compatibility_command_id
            if (
                self._head is None
                and self._planning_command is None
                and self._prediction_planning is None
            )
            else None
        )
        if release_command_id is not None:
            lease.compatibility_command_id = None
            self.arbiter.release_manual(release_command_id)
        if (
            self._head is None
            and self._planning_command is None
            and self._prediction_planning is None
            and not self.success_latch.is_latched()
        ):
            self._schedule_capture_locked()
        self._touch_locked()
        published = self._publish_cleared(
            cleared,
            expose_first_terminal=exposed_terminal,
        )
        if not published:
            self._mark_publication_failed_locked("cancel terminal rejected")
        return True, published

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(_WATCHDOG_PERIOD_S):
            with self._work:
                changed, published = self._expire_lease_locked(time.monotonic())
                if changed:
                    self._work.notify_all()
            if not published:
                self.arbiter.quiesce()
            if changed:
                self._publish_snapshot()

    def _on_arbiter_change(self) -> None:
        with self._work:
            manual_result_pending = bool(
                self._head is not None and self._head.phase == "moving"
            )
            if self.success_latch.is_latched() and not manual_result_pending:
                self._active = False
                if self._lease is not None:
                    if not self._lease.stopped:
                        self._invalidate_lease_generation_locked(self._lease)
                    self._lease.stopped = True
                    self._lease.stop_reason = "official_task_success"
                cleared, _ = self._clear_unstarted_locked(
                    "official_task_success"
                )
                self._clear_predictions_locked()
                if self._capture["phase"] == "pending":
                    self._capture.update({"phase": "discarded", "error": None})
                    self._capture_ready_at = 0.0
                published = self._publish_cleared(cleared)
                if not published:
                    self._mark_publication_failed_locked(
                        "raw-success cancellation terminal rejected"
                    )
            self._touch_locked()
            self._work.notify_all()
        self._publish_snapshot()

    def _recompute_capabilities_locked(self) -> None:
        self._motion_available = bool(
            self._policy_motion_available
            and self._capabilities.get("motion_available", True) is True
        )
        self._observe_available = bool(
            self._policy_observe_available
            and self._capabilities.get("observe_available", True) is True
        )

    def _touch_locked(self) -> None:
        self._control_revision += 1

    def _invalidate_lease_generation_locked(self, lease: _Lease) -> None:
        self._lease_generation += 1
        lease.generation = self._lease_generation

    def _snapshot_locked(self) -> dict[str, Any]:
        arbiter = self.arbiter.snapshot()
        head = self._head.public() if self._head is not None else None
        planning = (
            self._planning_command.public()
            if self._planning_command is not None
            else None
        )
        prediction_planning = (
            {
                "lease_id": self._prediction_planning.lease_id,
                "lease_generation": (
                    self._prediction_planning.lease_generation
                ),
                "target": self._prediction_planning.target,
                "action": self._prediction_planning.action,
                "camera": self._prediction_planning.camera,
                "predecessor_plan_id": (
                    self._prediction_planning.predecessor_plan_id
                ),
                "predicted_start_digest": (
                    self._prediction_planning.predicted_start_digest
                ),
            }
            if self._prediction_planning is not None
            else None
        )
        queue = [command.public() for command in self._pending]
        terminal = (
            dict(self._last_terminal)
            if self._last_terminal is not None
            else None
        )
        run_success_latched = self.success_latch.is_latched()
        manual_success_terminal = bool(
            isinstance(terminal, Mapping)
            and terminal.get("task_success") is True
        )
        display = (
            head
            or (
                terminal
                if not run_success_latched or manual_success_terminal
                else None
            )
            or {}
        )
        lease = self._lease
        lease_status = (
            "expired"
            if lease is not None and lease.expired
            else "succeeded"
            if run_success_latched
            else "stopped"
            if lease is not None and lease.stopped
            else "active"
            if lease is not None
            else "idle"
        )
        result = display.get("result")
        error = (
            result.get("error")
            if isinstance(result, Mapping)
            else self._prediction_error
        )
        stop_reason = (
            "official_task_success"
            if run_success_latched
            else lease.stop_reason
            if lease is not None and lease.stop_reason
            else result.get("stop_reason")
            if isinstance(result, Mapping)
            else None
        )
        available = bool(
            self._active
            and not self._quiescing
            and not self._closed
            and (self._motion_available or self._observe_available)
            and not run_success_latched
        )
        return {
            "control_revision": self._control_revision,
            "available": available,
            "motion_available": bool(available and self._motion_available),
            "observe_available": bool(available and self._observe_available),
            "lease_status": lease_status,
            "current_command": head,
            "planning_command": planning,
            "queue": queue,
            "queue_depth": len(queue),
            "queue_capacity": _QUEUE_CAPACITY,
            "prediction_pipeline_available": bool(
                self._capabilities.get("threaded_predicted_planning") is True
            ),
            "prediction_planning": prediction_planning,
            "predicted_plans": [
                prediction.public() for prediction in self._predicted_plans
            ],
            "predicted_plan_depth": len(self._predicted_plans),
            "predicted_plan_capacity": DASHBOARD_PREDICTED_PLAN_DEPTH,
            "hold_armed": bool(lease is not None and lease.hold_armed),
            "last_terminal": terminal,
            "pending_cleared_count": self._pending_cleared_count,
            "owner": arbiter.get("owner"),
            "busy": bool(arbiter.get("busy")),
            "agent_waiters": int(arbiter.get("agent_waiters") or 0),
            "capture": dict(self._capture),
            "command_id": display.get("command_id"),
            "lease_id": display.get("lease_id") or (
                lease.lease_id if lease is not None else None
            ),
            "sequence": display.get("sequence"),
            "target": display.get("target"),
            "action": display.get("action"),
            "phase": display.get("phase", "idle"),
            "error": error,
            "capture_error": self._capture.get("error"),
            "stop_reason": stop_reason,
            "selected_camera": self._selected_camera,
            "success_latched": run_success_latched,
            "unavailable_reason": (
                self._unavailable_reason
                if not (self._motion_available or self._observe_available)
                else None
            ),
            "capabilities": dict(self._capabilities),
        }

    def _publish_snapshot(self) -> bool:
        with self._lock:
            snapshot = self._snapshot_locked()
        callback = getattr(self._state, "update_control_snapshot", None)
        if callable(callback):
            try:
                return callback(snapshot, controller=self) is True
            except BaseException:
                return False
        return False


__all__ = [
    "BehaviorCommandArbiter",
    "BehaviorDashboardController",
    "BehaviorRawSuccessLatch",
    "ControlRequestError",
    "DASHBOARD_HOLD_ARM_DELAY_S",
    "DASHBOARD_PREDICTED_PLAN_DEPTH",
    "LEASE_TIMEOUT_S",
]
