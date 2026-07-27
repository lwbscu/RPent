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

    def observe(self, result: Any) -> bool:
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
            return self._latched

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
    in_flight_command_id: str | None = None
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
        }


class BehaviorDashboardController:
    """Own leases and dispatch one manual command at a time."""

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
        self._drained = threading.Condition(self._lock)
        self._commands: dict[tuple[str, int], _Command] = {}
        self._lease: _Lease | None = None
        self._active = False
        self._quiescing = False
        self._closed = False
        self._policy_motion_available = bool(motion_available)
        self._policy_observe_available = bool(observe_available)
        self._motion_available = False
        self._observe_available = False
        self._unavailable_reason = str(unavailable_reason or "controller_unavailable")
        self._capabilities: dict[str, Any] = {}
        self._lease_timeout_s = max(0.1, float(lease_timeout_s))
        self._control: dict[str, Any] = {
            "available": False,
            "motion_available": False,
            "observe_available": False,
            "busy": False,
            "owner": None,
            "command_id": None,
            "lease_id": None,
            "target": None,
            "action": None,
            "phase": "idle",
            "error": None,
            "stop_reason": None,
            "selected_camera": "head",
            "success_latched": False,
            "unavailable_reason": self._unavailable_reason,
        }
        self._watchdog_stop = threading.Event()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            name=f"behavior-dashboard-control-{getattr(state, 'run_id', 'run')}",
            daemon=True,
        )
        self.arbiter.add_listener(self._on_arbiter_change)
        self._watchdog.start()

    def bind_toolkit(self, toolkit: Any) -> None:
        command = getattr(toolkit, "dashboard_manual_command", None)
        if not callable(command):
            raise TypeError("toolkit must provide dashboard_manual_command")
        capabilities_callback = getattr(toolkit, "dashboard_control_capabilities", None)
        capabilities: dict[str, Any] = {}
        if callable(capabilities_callback):
            reported = capabilities_callback()
            if isinstance(reported, Mapping):
                capabilities = dict(reported)
        with self._lock:
            if self._closed:
                raise RuntimeError("dashboard controller is closed")
            self._toolkit = toolkit
            self._capabilities = capabilities
            dynamic_motion = capabilities.get("motion_available", True)
            dynamic_observe = capabilities.get("observe_available", True)
            self._motion_available = bool(
                self._policy_motion_available and dynamic_motion is True
            )
            self._observe_available = bool(
                self._policy_observe_available and dynamic_observe is True
            )
            reported_reason = capabilities.get("unavailable_reason")
            if reported_reason:
                self._unavailable_reason = str(reported_reason)
            elif self._motion_available or self._observe_available:
                self._unavailable_reason = ""
            self._refresh_control_locked()
        self._publish_snapshot()

    def configure_capabilities(
        self,
        *,
        motion_available: bool,
        observe_available: bool,
        unavailable_reason: str = "",
    ) -> None:
        with self._lock:
            self._policy_motion_available = bool(motion_available)
            self._policy_observe_available = bool(observe_available)
            self._motion_available = bool(
                self._policy_motion_available
                and self._capabilities.get("motion_available", True) is True
            )
            self._observe_available = bool(
                self._policy_observe_available
                and self._capabilities.get("observe_available", True) is True
            )
            if unavailable_reason:
                self._unavailable_reason = str(unavailable_reason)
            elif self._motion_available or self._observe_available:
                self._unavailable_reason = ""
            self._refresh_control_locked()
        self._publish_snapshot()

    def activate(self) -> None:
        with self._lock:
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
            self._refresh_control_locked()
        self._publish_snapshot()

    def submit(
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
                if existing.payload_fingerprint != fingerprint:
                    raise ControlRequestError(
                        409,
                        "idempotency_conflict",
                        "lease_id and sequence were used with a different command",
                    )
                return existing, True

        lifecycle = self._state.control_admission_snapshot()
        if lifecycle["state"] != "running" or lifecycle["official_task_success"]:
            raise ControlRequestError(410, "run_finished", "run is already finished")

        with self._lock:
            # Another request may have reserved the idempotency key while the
            # State lifecycle was read without holding either component lock.
            existing = self._commands.get(key)
            if existing is not None:
                if existing.payload_fingerprint != fingerprint:
                    raise ControlRequestError(
                        409,
                        "idempotency_conflict",
                        "lease_id and sequence were used with a different command",
                    )
                return existing, True
            self._expire_lease_locked(time.monotonic())
            if self.success_latch.is_latched():
                raise ControlRequestError(410, "run_finished", "official success latched")
            if not self._active or self._quiescing or self._closed:
                raise ControlRequestError(
                    409, "controller_unavailable", "manual controller unavailable"
                )
            if action == "observe":
                if not self._observe_available:
                    raise ControlRequestError(
                        409,
                        "observe_unavailable",
                        self._unavailable_reason or "camera refresh unavailable",
                    )
            elif not self._motion_available:
                raise ControlRequestError(
                    409,
                    "motion_unavailable",
                    self._unavailable_reason or "motion control unavailable",
                )
            if self._lease is None or self._lease.stopped or self._lease.expired:
                if sequence != 1:
                    raise ControlRequestError(
                        409,
                        "invalid_sequence",
                        "a new control lease must start at sequence 1",
                    )
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
            if lease.payload_fingerprint != fingerprint:
                raise ControlRequestError(
                    409,
                    "lease_command_conflict",
                    "a repeat lease cannot change target, action, or camera",
                )
            if lease.in_flight_command_id is not None:
                raise ControlRequestError(
                    409, "controller_busy", "a manual command is still in flight"
                )
            if sequence != lease.last_sequence + 1:
                raise ControlRequestError(
                    409,
                    "invalid_sequence",
                    "sequence must follow the previous completed command",
                )
            if action in _ONE_SHOT_ACTIONS and sequence != 1:
                raise ControlRequestError(
                    409,
                    "non_repeatable_action",
                    f"{action} does not support lease repetition",
                )

            command_id = uuid.uuid4().hex
            acquired, reason = self.arbiter.try_acquire_manual(command_id)
            if not acquired:
                if reason == "official_success_latched":
                    raise ControlRequestError(
                        410, "run_finished", "official success latched"
                    )
                message = (
                    "agent command is waiting"
                    if reason == "agent_waiting"
                    else "robot controller is busy"
                )
                raise ControlRequestError(409, reason or "controller_busy", message)
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
            lease.in_flight_command_id = command_id
            lease.deadline = time.monotonic() + self._lease_timeout_s
            self._control.update(
                {
                    "command_id": command_id,
                    "lease_id": lease_id,
                    "target": target,
                    "action": action,
                    "phase": "accepted",
                    "error": None,
                    "stop_reason": None,
                    "selected_camera": camera,
                }
            )
            self._refresh_control_locked()

        try:
            self._state.on_manual_command_start(command.public())
            self._publish_snapshot()
            worker = threading.Thread(
                target=self._run_command,
                args=(command,),
                name=f"behavior-dashboard-command-{command_id[:8]}",
                daemon=True,
            )
            worker.start()
        except Exception:
            with self._lock:
                if self._lease is not None:
                    self._lease.in_flight_command_id = None
                command.phase = "failed"
                command.result = {"error": "failed to start manual command"}
                self._refresh_control_locked()
            self.arbiter.release_manual(command_id)
            self._publish_snapshot()
            raise
        return command, False

    def heartbeat(self, *, lease_id: str) -> dict[str, Any]:
        lease_id = str(lease_id).strip()
        with self._lock:
            self._expire_lease_locked(time.monotonic())
            lease = self._lease
            if (
                lease is None
                or lease.lease_id != lease_id
                or lease.stopped
                or lease.expired
            ):
                raise ControlRequestError(
                    409, "lease_expired", "control lease is not active"
                )
            lease.deadline = time.monotonic() + self._lease_timeout_s
            result = self._snapshot_locked()
        self._publish_snapshot()
        return result

    def stop(self, *, lease_id: str, reason: str = "client_stop") -> dict[str, Any]:
        lease_id = str(lease_id).strip()
        with self._lock:
            lease = self._lease
            if lease is None or lease.lease_id != lease_id:
                raise ControlRequestError(409, "unknown_lease", "control lease not found")
            if not lease.stopped:
                lease.stopped = True
                lease.stop_reason = str(reason or "client_stop")
            if lease.in_flight_command_id is not None:
                self._control["phase"] = "stopping"
            self._control["stop_reason"] = lease.stop_reason
            self._control["lease_status"] = "stopped"
            self._refresh_control_locked()
            result = self._snapshot_locked()
        self._publish_snapshot()
        return result

    def select_camera(self, camera: str) -> dict[str, Any]:
        camera = str(camera).strip()
        if camera not in _CAMERAS:
            raise ControlRequestError(422, "invalid_camera", "invalid camera")
        self._state.set_selected_camera(camera)
        with self._lock:
            self._control["selected_camera"] = camera
            result = self._snapshot_locked()
        self._publish_snapshot()
        return result

    def state(self) -> dict[str, Any]:
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._expire_lease_locked(time.monotonic())
            self._refresh_control_locked()
            return self._snapshot_locked()

    def quiesce(self) -> None:
        with self._lock:
            self._quiescing = True
            self._active = False
            if self._lease is not None:
                self._lease.stopped = True
                self._lease.stop_reason = "controller_quiescing"
            self._refresh_control_locked()
        self.arbiter.quiesce()
        self._publish_snapshot()

    def drain(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._drained:
            while (
                self._lease is not None
                and self._lease.in_flight_command_id is not None
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._drained.wait(remaining)
        return self.arbiter.drain(max(0.0, deadline - time.monotonic()))

    def close(self, timeout_s: float = 10.0) -> bool:
        self.quiesce()
        drained = self.drain(timeout_s)
        if not drained:
            self._publish_snapshot()
            return False
        with self._lock:
            self._closed = True
            self._toolkit = None
            self._refresh_control_locked()
        self._watchdog_stop.set()
        self._watchdog.join(timeout=min(1.0, max(0.0, float(timeout_s))))
        self.arbiter.remove_listener(self._on_arbiter_change)
        self._publish_snapshot()
        return drained

    def _run_command(self, command: _Command) -> None:
        try:
            self._run_command_impl(command)
        except BaseException as exc:
            self._finalize_worker_failure(command, exc)

    def _run_command_impl(self, command: _Command) -> None:
        started = time.monotonic()
        result: dict[str, Any]
        with self._lock:
            lease = self._lease
            lifecycle = self._state.control_admission_snapshot()
            cancelled_reason = (
                "official_success_latched"
                if self.success_latch.is_latched()
                or lifecycle["official_task_success"]
                else "run_finished"
                if lifecycle["state"] != "running"
                else "controller_quiescing"
                if self._quiescing or self._closed or not self._active
                else lease.stop_reason
                if lease is not None and lease.stopped
                else "lease_expired"
                if lease is None
                or lease.lease_id != command.lease_id
                or lease.expired
                else None
            )
            command.phase = "cancelled" if cancelled_reason else "planning"
            self._control["phase"] = command.phase
            self._control["stop_reason"] = cancelled_reason
            self._refresh_control_locked()
        self._publish_snapshot()
        if cancelled_reason:
            result = {
                "primitive_success": False,
                "stop_reason": cancelled_reason,
                "cancelled_before_execution": True,
            }
        else:
            with self._lock:
                command.phase = "moving"
                self._control["phase"] = "moving"
                self._refresh_control_locked()
            self._publish_snapshot()
            try:
                toolkit = self._toolkit
                if toolkit is None:
                    raise RuntimeError("dashboard controller toolkit is not bound")
                payload = toolkit.dashboard_manual_command(
                    target=command.target,
                    action=command.action,
                    camera=command.camera,
                )
                result = dict(payload) if isinstance(payload, Mapping) else {
                    "primitive_success": False,
                    "error": "manual primitive returned a non-object result",
                }
            except Exception as exc:
                result = {
                    "primitive_success": False,
                    "stop_reason": "tool_error",
                    "error": str(exc),
                }
        result.setdefault("elapsed_s", max(0.0, time.monotonic() - started))
        command.result = result
        success_latched = self.success_latch.observe(result)
        failed = (
            result.get("primitive_success") is False
            or result.get("success") is False
            or result.get("error") not in (None, "", False)
            or result.get("capture_error") not in (None, "", False)
        )
        command.phase = (
            "cancelled"
            if result.get("cancelled_before_execution") is True
            else "failed"
            if failed
            else "completed"
        )
        self._state.on_manual_command_result(
            command.public(),
            result,
            official_success_latched=success_latched,
        )
        with self._drained:
            lease = self._lease
            if (
                lease is not None
                and lease.in_flight_command_id == command.command_id
            ):
                lease.in_flight_command_id = None
                if command.action in _ONE_SHOT_ACTIONS:
                    lease.stopped = True
                    lease.stop_reason = "command_complete"
                    self._control["lease_status"] = "completed"
            self._control.update(
                {
                    "phase": command.phase,
                    "error": result.get("error") or result.get("capture_error"),
                    "capture_error": result.get("capture_error"),
                    "stop_reason": (
                        lease.stop_reason
                        if lease is not None and lease.stopped
                        else result.get("stop_reason")
                        or (
                            "camera_refresh_failed"
                            if result.get("capture_error")
                            not in (None, "", False)
                            else None
                        )
                    ),
                    "success_latched": success_latched,
                }
            )
            capabilities = result.get("control_capabilities")
            if isinstance(capabilities, Mapping):
                self._capabilities.update(dict(capabilities))
                self._recompute_capabilities_locked()
            if success_latched:
                self._active = False
                if lease is not None:
                    lease.stopped = True
                    lease.stop_reason = "official_task_success"
            self._refresh_control_locked()
            self._drained.notify_all()
        self._publish_snapshot()
        self.arbiter.release_manual(command.command_id)
        self._publish_snapshot()

    def _finalize_worker_failure(
        self,
        command: _Command,
        exc: BaseException,
    ) -> None:
        """Fail closed and release the permit after any worker-side exception."""

        original_error = f"{type(exc).__name__}: {exc}"
        result = dict(command.result or {})
        prior_error = result.get("error")
        if prior_error not in (None, "", False) and str(prior_error) != original_error:
            result["prior_error"] = str(prior_error)
        result.update(
            {
                "primitive_success": False,
                "stop_reason": "dashboard_control_worker_error",
                "error": original_error,
                "worker_exception_type": type(exc).__name__,
            }
        )
        command.phase = "failed"
        command.result = result
        success_latched = self.success_latch.is_latched()

        # Retry a terminal State publication once.  Complete frame groups and
        # command IDs are idempotent, so a callback that raised after a partial
        # commit cannot duplicate robot work or frame revisions.
        try:
            self._state.on_manual_command_result(
                command.public(),
                result,
                official_success_latched=success_latched,
            )
        except BaseException as state_exc:
            result["state_publish_error"] = (
                f"{type(state_exc).__name__}: {state_exc}"
            )

        try:
            with self._drained:
                lease = self._lease
                if (
                    lease is not None
                    and lease.in_flight_command_id == command.command_id
                ):
                    lease.in_flight_command_id = None
                    lease.stopped = True
                    lease.stop_reason = "dashboard_control_worker_error"
                self._control.update(
                    {
                        "phase": "failed",
                        "error": original_error,
                        "stop_reason": "dashboard_control_worker_error",
                        "success_latched": success_latched,
                        "worker_exception_type": type(exc).__name__,
                    }
                )
                self._refresh_control_locked()
                self._drained.notify_all()
            try:
                self._publish_snapshot()
            except BaseException:
                pass
        finally:
            self.arbiter.release_manual(command.command_id)
            try:
                self._publish_snapshot()
            except BaseException:
                pass

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

    def _expire_lease_locked(self, now: float) -> bool:
        lease = self._lease
        if (
            lease is None
            or lease.stopped
            or lease.expired
            or now <= lease.deadline
        ):
            return False
        lease.expired = True
        lease.stopped = True
        lease.stop_reason = "lease_expired"
        self._control["phase"] = (
            "stopping" if lease.in_flight_command_id is not None else "cancelled"
        )
        self._control["stop_reason"] = "lease_expired"
        self._control["lease_status"] = "expired"
        self._refresh_control_locked()
        return True

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(_WATCHDOG_PERIOD_S):
            with self._lock:
                changed = self._expire_lease_locked(time.monotonic())
            if changed:
                self._publish_snapshot()

    def _on_arbiter_change(self) -> None:
        with self._lock:
            self._refresh_control_locked()
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

    def _refresh_control_locked(self) -> None:
        arbiter = self.arbiter.snapshot()
        self._control.update(
            {
                "available": bool(
                    self._active
                    and not self._quiescing
                    and not self._closed
                    and (self._motion_available or self._observe_available)
                    and not self.success_latch.is_latched()
                ),
                "motion_available": bool(
                    self._active
                    and not self._quiescing
                    and self._motion_available
                    and not self.success_latch.is_latched()
                ),
                "observe_available": bool(
                    self._active
                    and not self._quiescing
                    and self._observe_available
                    and not self.success_latch.is_latched()
                ),
                "busy": bool(arbiter["busy"]),
                "owner": arbiter["owner"],
                "success_latched": self.success_latch.is_latched(),
                "unavailable_reason": (
                    self._unavailable_reason
                    if not (self._motion_available or self._observe_available)
                    else None
                ),
                "capabilities": dict(self._capabilities),
            }
        )

    def _snapshot_locked(self) -> dict[str, Any]:
        return dict(self._control)

    def _publish_snapshot(self) -> None:
        snapshot = self.snapshot()
        callback = getattr(self._state, "update_control_snapshot", None)
        if callable(callback):
            callback(snapshot)


__all__ = [
    "BehaviorCommandArbiter",
    "BehaviorDashboardController",
    "BehaviorRawSuccessLatch",
    "ControlRequestError",
    "LEASE_TIMEOUT_S",
]
