"""BEHAVIOR-only lifecycle additions for the official Dashboard server."""

from __future__ import annotations

import socket
import time
from typing import Any

from fastapi import Body
from fastapi.responses import JSONResponse

from robots.behavior.dashboard_control import ControlRequestError
from rpent.dashboard.server import DashboardServer as OfficialDashboardServer


class DashboardServer(OfficialDashboardServer):
    """Add attach-only campaign launch and bounded stop without core patches."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._auto_started = False
        self._install_control_routes()

    def wait_for_launch(self, defaults: dict[str, Any]) -> dict[str, Any]:
        self._auto_started = False
        self._launch_event.clear()
        return super().wait_for_launch(defaults)

    def arm_auto_start(self, defaults: dict[str, Any]) -> None:
        """Publish immutable attach-only defaults for a running campaign."""

        self._launch_defaults = dict(defaults)
        self._launch_config = dict(defaults)
        self._launch_enabled = False
        self._auto_started = True
        self._launch_event.set()

    def stop(self, timeout_s: float = 10.0) -> None:
        """Request only this in-process server to stop and verify the deadline."""

        server = self._server
        if server is None:
            return
        server.should_exit = True
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        probe_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(
                    (probe_host, int(self.port)),
                    timeout=0.1,
                ):
                    pass
            except OSError:
                self._server = None
                return
            time.sleep(0.02)
        try:
            with socket.create_connection(
                (probe_host, int(self.port)),
                timeout=0.1,
            ):
                listening = True
        except OSError:
            listening = False
        if listening:
            raise RuntimeError(
                f"dashboard server did not stop on {self.host}:{self.port}"
            )
        self._server = None

    def _install_control_routes(self) -> None:
        """Attach BEHAVIOR-only routes without changing the official server."""

        def lookup_state(run_id: Any) -> Any:
            run = str(run_id or "").strip()
            if not run:
                raise ControlRequestError(422, "invalid_run", "run is required")
            state = self._runs.get(run)
            if state is None:
                raise ControlRequestError(404, "unknown_run", "unknown run")
            return state

        def state_for_run(run_id: Any) -> Any:
            state = lookup_state(run_id)
            lifecycle_callback = getattr(state, "control_admission_snapshot", None)
            lifecycle = (
                lifecycle_callback() if callable(lifecycle_callback) else {}
            )
            if (
                lifecycle.get("state") != "running"
                or lifecycle.get("official_task_success") is True
            ):
                raise ControlRequestError(
                    410, "run_finished", "run is already finished"
                )
            return state

        def controller_for_state(state: Any) -> Any:
            callback = getattr(state, "control_controller", None)
            controller = callback() if callable(callback) else None
            if controller is None:
                raise ControlRequestError(
                    409,
                    "controller_not_bound",
                    "manual controller is not bound to this run",
                )
            return controller

        def validate_payload(
            payload: Any,
            *,
            required: set[str],
            optional: set[str] | None = None,
        ) -> dict[str, Any]:
            if not isinstance(payload, dict):
                raise ControlRequestError(
                    422, "invalid_payload", "request body must be an object"
                )
            allowed = required | (optional or set())
            extra = sorted(set(payload) - allowed)
            missing = sorted(required - set(payload))
            if extra:
                raise ControlRequestError(
                    422,
                    "unexpected_fields",
                    f"unexpected fields: {', '.join(extra)}",
                )
            if missing:
                raise ControlRequestError(
                    422,
                    "missing_fields",
                    f"missing fields: {', '.join(missing)}",
                )
            return payload

        def error_response(
            exc: ControlRequestError,
            controller: Any = None,
        ) -> JSONResponse:
            payload: dict[str, Any] = {}
            snapshot = getattr(controller, "snapshot", None)
            if callable(snapshot):
                candidate = snapshot()
                if isinstance(candidate, dict):
                    payload.update(candidate)
            payload.update(exc.payload())
            return JSONResponse(payload, status_code=exc.status_code)

        @self._app.post("/api/run/control/command")
        def api_control_command(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            controller = None
            try:
                body = validate_payload(
                    payload,
                    required={
                        "run",
                        "lease_id",
                        "sequence",
                        "target",
                        "action",
                        "camera",
                    },
                )
                state = state_for_run(body["run"])
                controller = controller_for_state(state)
                command, deduplicated = controller.submit(
                    lease_id=body["lease_id"],
                    sequence=body["sequence"],
                    target=body["target"],
                    action=body["action"],
                    camera=body["camera"],
                )
                response = dict(
                    command.acceptance_snapshot
                    if not deduplicated
                    and isinstance(command.acceptance_snapshot, dict)
                    else controller.snapshot()
                )
                response.update(
                    {
                        "accepted": True,
                        "deduplicated": deduplicated,
                        "accepted_command_id": command.command_id,
                        "command_id": command.command_id,
                        "lease_id": command.lease_id,
                        "sequence": command.sequence,
                        "target": command.target,
                        "action": command.action,
                        "camera": command.camera,
                        "phase": (
                            "accepted"
                            if not deduplicated
                            else command.phase
                        ),
                    }
                )
                return JSONResponse(response, status_code=202)
            except ControlRequestError as exc:
                return error_response(exc, controller)

        @self._app.post("/api/run/control/heartbeat")
        def api_control_heartbeat(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            controller = None
            try:
                body = validate_payload(
                    payload, required={"run", "lease_id"}
                )
                state = state_for_run(body["run"])
                controller = controller_for_state(state)
                return JSONResponse(
                    controller.heartbeat(lease_id=body["lease_id"])
                )
            except ControlRequestError as exc:
                return error_response(exc, controller)

        @self._app.post("/api/run/control/stop")
        def api_control_stop(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            controller = None
            try:
                body = validate_payload(
                    payload,
                    required={"run", "lease_id"},
                    optional={"reason", "stop_mode"},
                )
                state = state_for_run(body["run"])
                controller = controller_for_state(state)
                return JSONResponse(
                    controller.stop(
                        lease_id=body["lease_id"],
                        reason=str(body.get("reason") or "client_stop"),
                        stop_mode=str(
                            body.get("stop_mode") or "clear_pending"
                        ),
                    )
                )
            except ControlRequestError as exc:
                return error_response(exc, controller)

        @self._app.post("/api/run/control/camera")
        def api_control_camera(
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            controller = None
            try:
                body = validate_payload(payload, required={"run", "camera"})
                state = state_for_run(body["run"])
                controller = controller_for_state(state)
                control = controller.select_camera(body["camera"])
                return JSONResponse({**control, "ok": True})
            except ControlRequestError as exc:
                return error_response(exc, controller)

        @self._app.get("/api/run/control/state")
        def api_control_state(run: str) -> JSONResponse:
            try:
                state = lookup_state(run)
                lifecycle_callback = getattr(
                    state, "control_admission_snapshot", None
                )
                lifecycle = (
                    lifecycle_callback() if callable(lifecycle_callback) else {}
                )
                if lifecycle.get("official_task_success") is True:
                    state_snapshot = getattr(state, "snapshot", None)
                    public_state = (
                        state_snapshot() if callable(state_snapshot) else {}
                    )
                    terminal = (
                        public_state.get("control")
                        if isinstance(public_state, dict)
                        else None
                    )
                    if (
                        isinstance(terminal, dict)
                        and terminal.get("success_latched") is True
                        and terminal.get("command_id")
                        and terminal.get("phase")
                        in {"completed", "failed", "cancelled"}
                    ):
                        return JSONResponse(terminal)
                    raise ControlRequestError(
                        410, "run_finished", "run is already finished"
                    )
                if lifecycle.get("state") != "running":
                    raise ControlRequestError(
                        410, "run_finished", "run is already finished"
                    )
                controller = controller_for_state(state)
                return JSONResponse(controller.state())
            except ControlRequestError as exc:
                return error_response(exc)


__all__ = ["DashboardServer"]
