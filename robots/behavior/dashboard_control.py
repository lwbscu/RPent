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
_CAPTURE_IDLE_DELAY_S = 0.34
_PLANNING_METADATA_FIELDS = (
    "planning_elapsed_s",
    "planning_profile",
    "fast_solver_deadline_s",
    "fast_solver_deadline",
    "latency_metrics",
    "obstacle_refresh",
    "safety_certificate",
    "selected_solver_stage",
    "solver_stages",
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
        """Latch success and identify whether this exact result supplied it."""

        binding = self._validated_receipt(result)
        with self._lock:
            if binding is not None:
                if (
                    self._receipt_binding is not None
                    and self._receipt_binding != binding
                ):
                    raise RuntimeError(
                        "official success receipt changed after the first latch"
                    )
                self._receipt_binding = binding
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
    """Serialize complete Agent and Dashboard transactions.

    Agent waiters have priority over a new manual repeat.  The currently
    executing discrete manual trajectory is never interrupted by this class.
    """

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

    def try_acquire_manual(self, command_id: str) -> tuple[bool, str | None]:
        with self._condition:
            if self._quiescing:
                return False, "controller_unavailable"
            if self.success_latch.is_latched():
                return False, "official_success_latched"
            if self._owner is not None:
                return False, "controller_busy"
            if self._agent_waiters:
                return False, "agent_waiting"
            self._owner = "manual"
            self._command_id = str(command_id)
            return True, None

    def release_manual(self, command_id: str) -> None:
        with self._condition:
            if self._owner == "manual" and self._command_id == str(command_id):
                self._owner = None
                self._command_id = None
                self._condition.notify_all()

    def handoff_manual(self, command_id: str, next_command_id: str) -> bool:
        """Retain a manual reservation unless an Agent is already waiting.

        The check and permit change are one condition-locked operation.  A
        separate ``snapshot()`` check would leave a race in which an Agent
        waiter could arrive immediately before the manual tail handoff.
        """

        command_id = str(command_id or "").strip()
        next_command_id = str(next_command_id or "").strip()
        with self._condition:
            if (
                not command_id
                or not next_command_id
                or self._owner != "manual"
                or self._command_id != command_id
            ):
                raise RuntimeError("manual reservation handoff does not match owner")
            if (
                self._quiescing
                or self.success_latch.is_latched()
                or self._agent_waiters
            ):
                # Keep the old exact permit until the controller publishes the
                # deferred command's cancellation terminal.  The caller
                # explicitly releases only after that publication barrier.
                return False
            self._command_id = next_command_id
            return True

    def require_manual_permit(self, command_id: str) -> None:
        """Require the exact command-scoped manual permit.

        Checking only ``owner == "manual"`` would let an unrelated internal
        caller borrow another command's permit while its Env RPC is in flight.
        The opaque command id therefore remains part of the permit until the
        controller publishes the terminal receipt and releases it.
        """

        command_id = str(command_id or "").strip()
        with self._condition:
            if (
                not command_id
                or self._owner != "manual"
                or self._command_id != command_id
            ):
                raise RuntimeError(
                    "Dashboard manual primitive requires its exact command permit"
                )

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
    deadline: float
    last_sequence: int = 0
    stopped: bool = False
    expired: bool = False
    stop_reason: str | None = None
    payload_fingerprint: tuple[str, str, str] | None = None


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
    timeline_started: bool = False
    acceptance_snapshot: dict[str, Any] | None = None

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
            "planning_metadata": dict(self.planning_metadata),
        }


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
        self._head: _Command | None = None
        self._pending: list[_Command] = []
        self._planning_command: _Command | None = None
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
                        reason or "controller_busy",
                        (
                            "agent command is waiting"
                            if reason == "agent_waiting"
                            else "robot controller is busy"
                        ),
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
                safety_certificate = prepared.get("safety_certificate")
                safety_checks = (
                    safety_certificate.get("checks")
                    if isinstance(safety_certificate, Mapping)
                    else None
                )
                expected_motion_kind = (
                    "torso" if target == "chassis" else "eef"
                )
                if (
                    not isinstance(safety_certificate, Mapping)
                    or safety_certificate.get("admitted") is not True
                    or safety_certificate.get("motion_kind")
                    != expected_motion_kind
                    or int(
                        safety_certificate.get(
                            "attachment_hand_count",
                            -1,
                        )
                    )
                    != 2
                    or not isinstance(safety_checks, Mapping)
                    or not safety_checks
                    or any(value is not True for value in safety_checks.values())
                ):
                    raise RuntimeError(
                        "planning-only probe omitted a verified safety certificate"
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
                    "safety_certificate": _bounded_planning_value(
                        safety_certificate
                    ),
                    "prepared": _bounded_planning_value(prepared),
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
        key = (lease_id, sequence)
        fingerprint = (target, action, camera)

        with self._lock:
            existing = self._commands.get(key)
            if existing is not None:
                self._require_matching_fingerprint(existing, fingerprint)
                return existing, True
        lifecycle = self._state.control_admission_snapshot()
        if lifecycle["state"] != "running" or lifecycle["official_task_success"]:
            raise ControlRequestError(410, "run_finished", "run is already finished")

        command_id = uuid.uuid4().hex
        with self._work:
            existing = self._commands.get(key)
            if existing is not None:
                self._require_matching_fingerprint(existing, fingerprint)
                return existing, True
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
                if self._head is not None or self._pending:
                    raise ControlRequestError(
                        409, "controller_busy", "another manual queue is active"
                    )
                if sequence != 1:
                    raise ControlRequestError(
                        409,
                        "invalid_sequence",
                        "a new control lease must start at sequence 1",
                    )
                self._reserve_for_new_head_locked(command_id)
                self._lease = _Lease(
                    lease_id=lease_id,
                    deadline=time.monotonic() + self._lease_timeout_s,
                    payload_fingerprint=fingerprint,
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
            if sequence != lease.last_sequence + 1:
                raise ControlRequestError(
                    409,
                    "invalid_sequence",
                    "sequence must follow the previous accepted command",
                )
            if action in _ONE_SHOT_ACTIONS and sequence != 1:
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
                self._reserve_for_new_head_locked(command_id)

            command = _Command(
                command_id=command_id,
                lease_id=lease_id,
                sequence=sequence,
                target=target,
                action=action,
                camera=camera,
                payload_fingerprint=fingerprint,
            )
            self._commands[key] = command
            lease.last_sequence = sequence
            lease.deadline = time.monotonic() + self._lease_timeout_s
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
            lease.stopped = True
            lease.stop_reason = str(reason or "client_stop")
            cleared, exposed_terminal = self._clear_deferred_head_locked(
                lease.stop_reason
            )
            self._touch_locked()
            self._work.notify_all()
            published = self._publish_cleared(
                cleared,
                expose_first_terminal=exposed_terminal,
            )
            if not published:
                self._mark_publication_failed_locked("cancel terminal rejected")
            result = self._snapshot_locked()
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
        with self._work:
            self._quiescing = True
            self._active = False
            if self._lease is not None:
                self._lease.stopped = True
                self._lease.stop_reason = "controller_quiescing"
            cleared, exposed_terminal = self._clear_deferred_head_locked(
                "controller_quiescing"
            )
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
        self._publish_snapshot()

    def drain(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._work:
            while (
                self._head is not None
                or self._pending
                or self._planning_command is not None
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
                    self._work.wait()
            if discard_plan is not None:
                self._discard_plan(discard_plan)
                with self._work:
                    self._work.notify_all()
                continue
            assert command is not None
            self._publish_snapshot()
            try:
                toolkit = self._require_toolkit()
                prepared = toolkit.dashboard_prepare_manual_command(
                    target=command.target,
                    action=command.action,
                    predecessor_plan_id=command.predecessor_plan_id,
                    permit_command_id=command.command_id,
                    # Controller admission only plans the current FIFO head.
                    # Never re-enable the Env RPC background escape hatch here.
                    background=False,
                )
                if not isinstance(prepared, Mapping):
                    raise RuntimeError("manual planner returned a non-object result")
                plan_id = str(prepared.get("plan_id") or "").strip()
                if not plan_id:
                    raise RuntimeError("manual planner did not return plan_id")
            except BaseException as exc:
                self._finish_planning_failure(command, exc)
                continue
            discard_after_prepare = False
            with self._work:
                if self._planning_command is command:
                    self._planning_command = None
                if not self._command_is_active_locked(command):
                    discard_after_prepare = True
                else:
                    command.plan_id = plan_id
                    command.planning_metadata = _planning_metadata(
                        prepared, plan_id=plan_id
                    )
                    command.phase = "prepared"
                self._touch_locked()
                self._work.notify_all()
            if discard_after_prepare:
                self._discard_plan(plan_id)
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
                        self._command_has_exact_permit_locked(head)
                        and head.timeline_started
                        and (
                            head.action in _ONE_SHOT_ACTIONS
                            or head.plan_id
                        )
                    ):
                        command = head
                        command.phase = "moving"
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
        try:
            toolkit = self._require_toolkit()
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
                raise RuntimeError("manual executor returned a non-object result")
            result = dict(payload)
        except BaseException as exc:
            result = {
                "primitive_success": False,
                "stop_reason": "tool_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        self._attach_planning_metadata(command, result)
        result.setdefault("elapsed_s", max(0.0, time.monotonic() - started))
        capture_result = self._detach_capture_payload(result)
        try:
            success_latched, manual_success_binding = (
                self.success_latch.observe_with_binding(result)
            )
        except BaseException as exc:
            result.update(
                {
                    "primitive_success": False,
                    "stop_reason": "dashboard_control_worker_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "worker_exception_type": type(exc).__name__,
                }
            )
            success_latched = self.success_latch.is_latched()
            manual_success_binding = None
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
        if capture_result is not None and not success_latched:
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
                    lease.stopped = True
                    lease.stop_reason = "official_task_success"
                cleared = self._clear_pending_locked("official_task_success")
                if self._capture["phase"] == "pending":
                    self._capture.update({"phase": "discarded", "error": None})
                    self._capture_ready_at = 0.0
            elif failed:
                if lease is not None:
                    lease.stopped = True
                    lease.stop_reason = str(
                        result.get("stop_reason") or "command_error"
                    )
                cleared = self._clear_pending_locked(
                    str(result.get("stop_reason") or "command_error")
                )
            elif self._pending:
                self._head = self._pending.pop(0)
            elif lease is not None and command.action in _ONE_SHOT_ACTIONS:
                lease.stopped = True
                lease.stop_reason = "command_complete"

            next_head = self._head
            self._touch_locked()
            self._work.notify_all()

        # The current action and every cancellation caused by it must be
        # committed while the old exact permit is still held.
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
        if next_head is not None:
            if not handed_off:
                arbiter_snapshot = self.arbiter.snapshot()
                reason = (
                    "official_task_success"
                    if arbiter_snapshot.get("success_latched") is True
                    else "controller_quiescing"
                    if arbiter_snapshot.get("quiescing") is True
                    else "agent_waiting"
                )
                with self._work:
                    preempted: list[_Command] = []
                    if self._head is next_head:
                        preempted = [next_head, *self._pending]
                        self._head = None
                        self._pending.clear()
                        self._mark_cleared_locked(preempted, reason)
                        if lease is not None:
                            lease.stopped = True
                            lease.stop_reason = reason
                        self._touch_locked()
                        self._work.notify_all()
                if preempted and not self._publish_cleared(
                    preempted,
                    expose_first_terminal=reason == "agent_waiting",
                ):
                    self._fail_closed_publication(
                        owner_command_id=command.command_id,
                        error="preempted cancellation terminal was rejected",
                    )
                    return
                # handoff_manual deliberately retains the old permit on a
                # refusal so cancellation remains observable first.
                self.arbiter.release_manual(command.command_id)
        else:
            self.arbiter.release_manual(command.command_id)
            with self._work:
                if (
                    not success_latched
                    and command.action != "observe"
                    and result.get("cancelled_before_execution") is not True
                    and not self._quiescing
                    and not self._closed
                    and not self._publication_failed
                ):
                    self._schedule_capture_locked()
                self._touch_locked()
                self._work.notify_all()
        self._publish_snapshot()

    def _execute_capture(self, revision: int, command_id: str) -> None:
        try:
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

        # Capture completion/failure is visible before its exact permit can be
        # handed to a deferred motion or released.
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
        if next_head is not None:
            if not handed_off:
                arbiter_snapshot = self.arbiter.snapshot()
                reason = (
                    "official_task_success"
                    if arbiter_snapshot.get("success_latched") is True
                    else "controller_quiescing"
                    if arbiter_snapshot.get("quiescing") is True
                    else "agent_waiting"
                )
                with self._work:
                    cleared: list[_Command] = []
                    if self._head is next_head:
                        cleared = [next_head, *self._pending]
                        self._head = None
                        self._pending.clear()
                        self._mark_cleared_locked(cleared, reason)
                        if self._lease is not None:
                            self._lease.stopped = True
                            self._lease.stop_reason = reason
                        self._touch_locked()
                        self._work.notify_all()
                if cleared and not self._publish_cleared(
                    cleared,
                    expose_first_terminal=reason == "agent_waiting",
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
        result = {
            "primitive_success": False,
            "stop_reason": "planning_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        release_command_id: str | None = None
        with self._work:
            if self._planning_command is command:
                self._planning_command = None
            if not self._command_is_active_locked(command):
                self._work.notify_all()
                return
            command.phase = "failed"
            command.result = result
            cleared: list[_Command] = []
            if self._head is command:
                self._head = None
                self._last_terminal = self._terminal_public(command)
                cleared = self._clear_pending_locked("planning_error")
                if self._lease is not None:
                    self._lease.stopped = True
                    self._lease.stop_reason = "planning_error"
                release_command_id = command.command_id
            else:
                index = self._pending.index(command)
                cleared = self._pending[index + 1 :]
                del self._pending[index:]
                self._pending_cleared_count += 1
                self._mark_cleared_locked(cleared, "planning_error")
                if self._lease is not None:
                    self._lease.stopped = True
                    self._lease.stop_reason = "planning_error"
            self._touch_locked()
            self._work.notify_all()
        timeline_published, timeline_error = self._publish_manual_result(
            command,
            result,
            official_success_latched=False,
        )
        # State and control readers must see the immutable terminal before the
        # exact permit is released to an Agent or a fresh manual lease.
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
            # one deferred head, but neither plan nor execute it until capture
            # completion atomically hands over the exact permit.
            return
        acquired, reason = self.arbiter.try_acquire_manual(command_id)
        if acquired:
            return
        if reason == "official_success_latched":
            raise ControlRequestError(410, "run_finished", "official success latched")
        message = (
            "agent command is waiting"
            if reason == "agent_waiting"
            else "robot controller is busy"
        )
        raise ControlRequestError(409, reason or "controller_busy", message)

    def _next_unplanned_locked(self) -> _Command | None:
        # Planning and physical execution both enter the simulator backend.
        # A pending background prepare must therefore not overlap the current
        # head's physics steps.  Keep accepting the bounded tail, but promote
        # each item to head before planning it from the then-live state.
        command = self._head
        if (
            command is None
            or not command.timeline_started
            or command.action in _ONE_SHOT_ACTIONS
            or command.plan_id
        ):
            return None
        if not self._command_has_exact_permit_locked(command):
            return None
        command.predecessor_plan_id = None
        return command

    def _command_has_exact_permit_locked(self, command: _Command) -> bool:
        arbiter = self.arbiter.snapshot()
        return bool(
            arbiter.get("owner") == "manual"
            and arbiter.get("command_id") == command.command_id
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
        revision = int(self._capture["revision"]) + 1
        self._capture = {
            "phase": "pending",
            "revision": revision,
            "error": None,
        }
        self._capture_ready_at = time.monotonic() + _CAPTURE_IDLE_DELAY_S

    def _new_capture_revision(self) -> int:
        with self._work:
            revision = int(self._capture["revision"]) + 1
            self._capture.update(
                {"phase": "started", "revision": revision, "error": None}
            )
            self._touch_locked()
            return revision

    def _detach_capture_payload(
        self, result: dict[str, Any]
    ) -> dict[str, Any] | None:
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

    def _clear_deferred_head_locked(
        self, reason: str
    ) -> tuple[list[_Command], bool]:
        """Clear queued work without cancelling an exact-permit motion.

        A head accepted while capture is already running has no permit of its
        own.  It is therefore queued work, not an in-flight robot action, and
        must be cancelled together with the tail on stop/expiry/quiesce.
        """

        deferred_head = bool(
            self._head is not None
            and self._head.phase == "accepted"
            and not self._command_has_exact_permit_locked(self._head)
        )
        if not deferred_head:
            return self._clear_pending_locked(reason), False
        assert self._head is not None
        cleared = [self._head, *self._pending]
        self._head = None
        self._pending.clear()
        self._mark_cleared_locked(cleared, reason)
        return cleared, True

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
            self._lease.stopped = True
            self._lease.stop_reason = "dashboard_state_publication_failed"
        self._touch_locked()

    def _fail_closed_publication(
        self,
        *,
        owner_command_id: str,
        error: str,
    ) -> None:
        """Disable all admission before releasing a permit with no State ack."""

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
        # Quiescing is the fail-closed barrier: waiting Agent transactions and
        # every future manual admission are rejected before the old permit is
        # released for cleanup.
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

    @staticmethod
    def _require_matching_fingerprint(
        command: _Command, fingerprint: tuple[str, str, str]
    ) -> None:
        if command.payload_fingerprint != fingerprint:
            raise ControlRequestError(
                409,
                "idempotency_conflict",
                "lease_id and sequence were used with a different command",
            )

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
        lease.stopped = True
        lease.stop_reason = "lease_expired"
        cleared, exposed_terminal = self._clear_deferred_head_locked(
            "lease_expired"
        )
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
            if self.success_latch.is_latched():
                self._active = False
                if self._lease is not None:
                    self._lease.stopped = True
                    self._lease.stop_reason = "official_task_success"
                cleared, _ = self._clear_deferred_head_locked(
                    "official_task_success"
                )
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

    def _snapshot_locked(self) -> dict[str, Any]:
        arbiter = self.arbiter.snapshot()
        head = self._head.public() if self._head is not None else None
        planning = (
            self._planning_command.public()
            if self._planning_command is not None
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
        error = result.get("error") if isinstance(result, Mapping) else None
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
    "LEASE_TIMEOUT_S",
]
