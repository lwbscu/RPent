"""BEHAVIOR-only lifecycle additions for the official Dashboard server."""

from __future__ import annotations

import socket
import time
from typing import Any

from rpent.dashboard.server import DashboardServer as OfficialDashboardServer


class DashboardServer(OfficialDashboardServer):
    """Add attach-only campaign launch and bounded stop without core patches."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._auto_started = False

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


__all__ = ["DashboardServer"]
