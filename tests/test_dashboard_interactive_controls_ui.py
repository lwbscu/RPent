# Copyright (c) 2026 RPent contributors
"""Real-Chrome acceptance tests for the BEHAVIOR Dashboard controls.

These tests deliberately drive the rendered DOM through Chrome DevTools
Protocol.  They do not treat HTML source matching as a substitute for browser
layout, input, timing, or media behavior.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "rpent/dashboard/index.html"
REFERENCE = REPO / "tests/fixtures/dashboard_interactive_controls_reference.png"
REFERENCE_SHA256 = "9d3071e672c81845c6e88247bcf0d0642a0a9b7c5b28c887823b99d5d3a1607c"
HIGH_DPI_REFERENCE = Path("/home/ubuntu/lwb/Projects/test/image.png")
HIGH_DPI_REFERENCE_SHA256 = (
    "ef0d70afaf83b07a79e0cae37d23c90534a787c4ef8dcf8b0a0784e307369223"
)
OUTPUT_DIR = Path("/home/ubuntu/lwb/RPent_outputs/dashboard_visual_acceptance")
LIVE_SCREENSHOT = OUTPUT_DIR / "interactive_controls_live_823x526.png"
OVERLAY_SCREENSHOT = OUTPUT_DIR / "interactive_controls_reference_overlay_50pct.png"
HIGH_DPI_LIVE_SCREENSHOT = OUTPUT_DIR / "interactive_controls_live_1280x867.png"
HIGH_DPI_OVERLAY_SCREENSHOT = (
    OUTPUT_DIR / "interactive_controls_reference_overlay_50pct_1280x867.png"
)


def _wait_until(predicate, *, timeout: float = 5.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except Exception as exc:  # pragma: no cover - retained for diagnostics
            last = exc
        time.sleep(interval)
    raise AssertionError(f"condition did not become true in {timeout}s; last={last!r}")


class _MockDashboard:
    """Thread-safe state behind the Dashboard HTTP fixture."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.head_png = self._make_head_frame()
        self.high_dpi_head_png = self._make_high_dpi_head_frame()
        self.reset()

    @staticmethod
    def _make_head_frame() -> bytes:
        # The reference's center camera view is reused as deterministic mock
        # simulator RGB.  Controls remain live DOM/CSS; this only removes
        # nondeterministic scene imagery from pixel comparisons.
        with Image.open(REFERENCE) as image:
            camera = image.convert("RGB").crop((211, 0, 619, 410))
            output = io.BytesIO()
            camera.save(output, format="PNG")
            return output.getvalue()

    def _make_high_dpi_head_frame(self) -> bytes:
        if (
            not HIGH_DPI_REFERENCE.exists()
            or hashlib.sha256(HIGH_DPI_REFERENCE.read_bytes()).hexdigest()
            != HIGH_DPI_REFERENCE_SHA256
        ):
            return self.head_png
        with Image.open(HIGH_DPI_REFERENCE) as image:
            camera = image.convert("RGB").crop((325, 0, 962, 650))
            output = io.BytesIO()
            camera.save(output, format="PNG")
            return output.getvalue()

    @staticmethod
    def _control_base() -> dict[str, Any]:
        return {
            "control_revision": 0,
            "available": True,
            "motion_available": True,
            "observe_available": True,
            "busy": False,
            "owner": None,
            "command_id": None,
            "lease_id": None,
            "lease_status": "idle",
            "current_command": None,
            "planning_command": None,
            "queue": [],
            "queue_depth": 0,
            "queue_capacity": 5,
            "last_terminal": None,
            "pending_cleared_count": 0,
            "phase": "idle",
            "error": None,
            "capture": {"phase": "idle", "revision": 0, "error": None},
            "capabilities": {
                "base_available": True,
                "torso_available": True,
                "eef_available": {"left": True, "right": True},
                "wrist_rotation_available": {"left": True, "right": True},
                "gripper_available": {"left": True, "right": True},
            },
        }

    def reset(
        self,
        *,
        command_delay_s: float = 0.045,
        arbiter_release_delay_s: float = 0.0,
        stop_delay_s: float = 0.0,
        accept_delay_s: float = 0.0,
    ) -> None:
        with self.lock:
            self.command_delay_s = command_delay_s
            self.arbiter_release_delay_s = arbiter_release_delay_s
            self.stop_delay_s = stop_delay_s
            self.accept_delay_s = accept_delay_s
            self.commands: list[dict[str, Any]] = []
            self.heartbeats: list[dict[str, Any]] = []
            self.stops: list[dict[str, Any]] = []
            self.cameras: list[dict[str, Any]] = []
            self.max_in_flight = 0
            self.max_queue_depth = 0
            self.command_posts_in_flight = 0
            self.max_command_posts_in_flight = 0
            self.control_revision = 0
            self.pending_cleared_count = 0
            self.run_get_count = 0
            self.frame_get_count = 0
            self.use_high_dpi_frame = False
            self.selected_camera = "head"
            self.camera_delays: dict[str, float] = {}
            self.camera_failures: set[str] = set()
            self.camera_in_flight = 0
            self.max_camera_in_flight = 0
            self._dedup: dict[tuple[str, str, int], dict[str, Any]] = {}

    def timeline(self) -> list[dict[str, Any]]:
        rows = []
        for step in range(1, 57):
            rows.append(
                {
                    "step": step,
                    "action": "pi0_nav_pick" if step == 49 else "observe",
                    "args": {},
                    "result": ({"chunks": 20, "max_steps": 20} if step == 49 else {}),
                    "elapsed_s": 94.7 if step == 49 else 0.2,
                    "has_action_video": step == 49,
                }
            )
        return rows

    def run_payload(self) -> dict[str, Any]:
        control = self.control_state()["control"]
        return {
            "id": "mock-run",
            "suite": "behavior_2025_challenge",
            "name": "picking_up_trash",
            "task": 1,
            "seed": 13,
            "state": "running",
            "terminated": False,
            "frame_idx": 1587,
            "timeline_revision": 56,
            "frame_revisions": {
                "head": 1587,
                "left_wrist": 1587,
                "right_wrist": 1587,
            },
            "capture_group_id": "visual-fixture-1587",
            "simulator_step": 1587,
            "metadata": {
                "public-tool-count": 10,
                "public-tool-contract-version": "v2",
            },
            "timeline": self.timeline(),
            "has_video": False,
            "control": control,
            "usage": {"in": 0, "out": 0, "tool_calls": 10},
        }

    def accept(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        with self.lock:
            self.command_posts_in_flight += 1
            self.max_command_posts_in_flight = max(
                self.max_command_posts_in_flight,
                self.command_posts_in_flight,
            )
            delay = self.accept_delay_s
        try:
            time.sleep(delay)
            return self._accept_locked(body)
        finally:
            with self.lock:
                self.command_posts_in_flight -= 1

    def _accept_locked(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        now = time.monotonic()
        key = (str(body["run"]), str(body["lease_id"]), int(body["sequence"]))
        with self.lock:
            prior = self._dedup.get(key)
            if prior is not None:
                snapshot = self._control_state_locked(now)
                snapshot.update(
                    {
                        "accepted": True,
                        "deduplicated": True,
                        "command_id": prior["command_id"],
                    }
                )
                return HTTPStatus.ACCEPTED, snapshot
            active = [
                item
                for item in self.commands
                if not item["cancelled"] and item["complete_at"] > now
            ]
            pending = max(0, len(active) - 1)
            if pending >= 5:
                return HTTPStatus.CONFLICT, {
                    "error": "queue_full",
                    "detail": "manual queue is full",
                }
            start_at = max(
                now,
                max(
                    (
                        item["release_at"]
                        for item in active
                    ),
                    default=now,
                ),
            )
            command = {
                **body,
                "command_id": f"visual-command-{len(self.commands) + 1}",
                "accepted_at": now,
                "start_at": start_at,
                "complete_at": start_at + self.command_delay_s,
                "release_at": (
                    start_at
                    + self.command_delay_s
                    + self.arbiter_release_delay_s
                ),
                "cancelled": False,
            }
            self.commands.append(command)
            self._dedup[key] = command
            self.control_revision += 1
            self.max_in_flight = max(self.max_in_flight, 1)
            snapshot = self._control_state_locked(now)
            snapshot.update(
                {
                    "accepted": True,
                    "deduplicated": False,
                    "command_id": command["command_id"],
                }
            )
            return HTTPStatus.ACCEPTED, snapshot

    def control_state(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            return {"control": self._control_state_locked(now)}

    @staticmethod
    def _public_command(command: dict[str, Any], phase: str) -> dict[str, Any]:
        return {
            key: command[key]
            for key in (
                "command_id",
                "lease_id",
                "sequence",
                "target",
                "action",
                "camera",
            )
        } | {"phase": phase, "result": None, "plan_id": None}

    def _control_state_locked(self, now: float) -> dict[str, Any]:
        control = self._control_base()
        control["control_revision"] = self.control_revision
        control["selected_camera"] = self.selected_camera
        completed = []
        active = []
        for command in self.commands:
            if command["cancelled"]:
                completed.append((command, "cancelled"))
            elif now >= command["complete_at"]:
                command.setdefault("terminal_receipt_at", now)
                command.setdefault("arbiter_released_at", now)
                completed.append((command, "completed"))
            else:
                active.append(command)
        current = active[0] if active else None
        queue = active[1:]
        if current is not None:
            phase = "moving" if now >= current["start_at"] else "planning"
            public = self._public_command(current, phase)
            control.update(
                {
                    "busy": True,
                    "owner": "manual",
                    "command_id": current["command_id"],
                    "lease_id": current["lease_id"],
                    "lease_status": "active",
                    "sequence": current["sequence"],
                    "target": current["target"],
                    "action": current["action"],
                    "phase": phase,
                    "current_command": public,
                }
            )
        control["queue"] = [
            self._public_command(command, "accepted") for command in queue
        ]
        control["queue_depth"] = len(queue)
        self.max_queue_depth = max(self.max_queue_depth, len(queue))
        control["pending_cleared_count"] = self.pending_cleared_count
        if completed:
            terminal, phase = completed[-1]
            control["last_terminal"] = self._public_command(terminal, phase)
        return control

    def stop(self, body: dict[str, Any]) -> dict[str, Any]:
        time.sleep(self.stop_delay_s)
        with self.lock:
            stopped_at = time.monotonic()
            self.stops.append({**body, "at": stopped_at})
            lease_id = body.get("lease_id")
            active = [
                command
                for command in self.commands
                if command["lease_id"] == lease_id
                and not command["cancelled"]
                and command["complete_at"] > stopped_at
            ]
            for command in active[1:]:
                if not command["cancelled"]:
                    command["cancelled"] = True
                    self.pending_cleared_count += 1
            self.control_revision += 1
            return self._control_state_locked(stopped_at)

    def select_camera(
        self, body: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        camera = str(body.get("camera"))
        with self.lock:
            request = {
                **body,
                "started_at": time.monotonic(),
                "state": "pending",
            }
            self.cameras.append(request)
            self.camera_in_flight += 1
            self.max_camera_in_flight = max(
                self.max_camera_in_flight, self.camera_in_flight
            )
            delay = self.camera_delays.get(camera, 0.0)
            should_fail = camera in self.camera_failures
        try:
            time.sleep(delay)
            with self.lock:
                request["completed_at"] = time.monotonic()
                if should_fail:
                    request["state"] = "failed"
                    return HTTPStatus.SERVICE_UNAVAILABLE, {
                        "detail": f"camera {camera} unavailable"
                    }
                self.selected_camera = camera
                request["state"] = "completed"
                return HTTPStatus.OK, {"camera": camera}
        finally:
            with self.lock:
                self.camera_in_flight -= 1


class _Handler(BaseHTTPRequestHandler):
    server: "_DashboardHTTPServer"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        self._send(
            status,
            json.dumps(value, separators=(",", ":")).encode(),
            "application/json",
        )

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        mock = self.server.mock
        if parsed.path == "/":
            self._send(HTTPStatus.OK, HTML.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/api/launch/state":
            self._json({"enabled": False, "pending": False})
        elif parsed.path == "/api/runs":
            self._json({"runs": [{"id": "mock-run"}], "runs_dir": "/mock"})
        elif parsed.path == "/api/run":
            with mock.lock:
                mock.run_get_count += 1
            self._json(mock.run_payload())
        elif parsed.path == "/api/run/transcript":
            self._json({"events": []})
        elif parsed.path == "/api/run/control/state":
            self._json(mock.control_state())
        elif parsed.path == "/api/run/frame":
            with mock.lock:
                mock.frame_get_count += 1
                frame = (
                    mock.high_dpi_head_png
                    if mock.use_high_dpi_frame
                    else mock.head_png
                )
            self._send(HTTPStatus.OK, frame, "image/png")
        elif parsed.path == "/api/stream":
            # A complete SSE response is enough for these deterministic tests;
            # EventSource reconnect does not affect any assertion.
            self._send(HTTPStatus.NO_CONTENT, b"", "text/event-stream")
        elif parsed.path == "/api/run/action-video":
            self._send(HTTPStatus.NOT_FOUND, b"", "video/mp4")
        else:
            self._json({"detail": f"unknown test route {parsed.path}"}, 404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        body = self._body()
        mock = self.server.mock
        if parsed.path == "/api/run/control/command":
            status, result = mock.accept(body)
            self._json(result, status)
        elif parsed.path == "/api/run/control/heartbeat":
            with mock.lock:
                mock.heartbeats.append({**body, "at": time.monotonic()})
            self._json(mock.control_state())
        elif parsed.path == "/api/run/control/stop":
            self._json(mock.stop(body), HTTPStatus.ACCEPTED)
        elif parsed.path == "/api/run/control/camera":
            status, result = mock.select_camera(body)
            self._json(result, status)
        else:
            self._json({"detail": f"unknown test route {parsed.path}"}, 404)


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, mock: _MockDashboard):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.mock = mock


class _WebSocket:
    """Small RFC6455 client sufficient for localhost CDP JSON traffic."""

    def __init__(self, url: str):
        parsed = urlparse(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), 5)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        resource = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        request = (
            f"GET {resource} HTTP/1.1\r\n"
            f"Host: {parsed.hostname}:{parsed.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: http://127.0.0.1\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        response = self._recv_until(b"\r\n\r\n")
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"CDP websocket upgrade failed: {response!r}")

    def _recv_until(self, marker: bytes) -> bytes:
        value = bytearray()
        while marker not in value:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise EOFError("websocket closed")
            value.extend(chunk)
        return bytes(value)

    def _read_exact(self, size: int) -> bytes:
        value = bytearray()
        while len(value) < size:
            chunk = self.sock.recv(size - len(value))
            if not chunk:
                raise EOFError("websocket closed")
            value.extend(chunk)
        return bytes(value)

    def send_json(self, value: dict[str, Any]) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode()
        first = 0x81
        size = len(payload)
        if size < 126:
            header = bytes((first, 0x80 | size))
        elif size < 65536:
            header = bytes((first, 0x80 | 126)) + struct.pack("!H", size)
        else:
            header = bytes((first, 0x80 | 127)) + struct.pack("!Q", size)
        mask = secrets.token_bytes(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def receive_json(self) -> dict[str, Any]:
        while True:
            first, second = self._read_exact(2)
            opcode = first & 0x0F
            size = second & 0x7F
            if size == 126:
                size = struct.unpack("!H", self._read_exact(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self._read_exact(8))[0]
            mask = self._read_exact(4) if second & 0x80 else b""
            payload = self._read_exact(size)
            if mask:
                payload = bytes(
                    byte ^ mask[index % 4] for index, byte in enumerate(payload)
                )
            if opcode == 0x9:  # ping
                self._send_control(0xA, payload)
                continue
            if opcode == 0x8:
                raise EOFError("CDP websocket closed")
            if opcode == 0x1:
                return json.loads(payload)

    def _send_control(self, opcode: int, payload: bytes) -> None:
        mask = secrets.token_bytes(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self.sock.sendall(bytes((0x80 | opcode, 0x80 | len(payload))) + mask + masked)

    def close(self) -> None:
        try:
            self._send_control(0x8, b"")
        except OSError:
            pass
        self.sock.close()


class _ChromePage:
    def __init__(self, url: str):
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if not chrome:
            pytest.skip("system Google Chrome/Chromium is unavailable")
        self.profile = tempfile.TemporaryDirectory(prefix="rpent-dashboard-chrome-")
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--hide-scrollbars",
                "--remote-allow-origins=*",
                "--remote-debugging-port=0",
                f"--user-data-dir={self.profile.name}",
                "--window-size=1406,650",
                url,
            ],
            stdout=self.log,
            stderr=self.log,
        )
        port_file = Path(self.profile.name) / "DevToolsActivePort"
        _wait_until(port_file.exists, timeout=10)
        self.port = int(port_file.read_text().splitlines()[0])

        def page_target():
            with urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=2) as reply:
                targets = json.load(reply)
            return next(
                (
                    target
                    for target in targets
                    if target.get("type") == "page"
                    and target.get("url", "").startswith(url)
                ),
                None,
            )

        target = _wait_until(page_target, timeout=10)
        self.ws = _WebSocket(target["webSocketDebuggerUrl"])
        self.next_id = 0
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.set_device_metrics(width=1406, height=580, device_scale_factor=1)
        self.wait_ready()

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        call_id = self.next_id
        self.ws.send_json({"id": call_id, "method": method, "params": params or {}})
        while True:
            response = self.ws.receive_json()
            if response.get("id") != call_id:
                continue
            if "error" in response:
                raise RuntimeError(f"{method}: {response['error']}")
            return response.get("result", {})

    def evaluate(self, expression: str, *, await_promise: bool = True) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": await_promise,
                "returnByValue": True,
                "userGesture": True,
            },
        )
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"]
            raise AssertionError(detail.get("text", "browser evaluation failed"))
        remote = result.get("result", {})
        if remote.get("subtype") == "error":
            raise AssertionError(remote.get("description", "browser error"))
        return remote.get("value")

    def wait_ready(self) -> None:
        _wait_until(
            lambda: self.evaluate(
                "document.readyState === 'complete' && "
                "document.querySelector('#framewrap')?.classList.contains('behavior-mode')"
            ),
            timeout=10,
        )

    def reload(self) -> None:
        self.call("Page.reload", {"ignoreCache": True})
        self.wait_ready()

    def sleep(self, milliseconds: int) -> None:
        self.evaluate(f"new Promise(resolve => setTimeout(resolve, {milliseconds}))")

    def set_device_metrics(
        self, *, width: int, height: int, device_scale_factor: float
    ) -> None:
        self.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": width,
                "height": height,
                "deviceScaleFactor": device_scale_factor,
                "mobile": False,
            },
        )

    def mouse(self, selector: str, *, down: bool) -> dict[str, Any]:
        rect = self.evaluate(
            f"""
            (() => {{
              const rect = document.querySelector({json.dumps(selector)}).getBoundingClientRect();
              return {{x: rect.x, y: rect.y, width: rect.width, height: rect.height}};
            }})()
            """
        )
        self.call(
            "Input.dispatchMouseEvent",
            {
                "type": "mousePressed" if down else "mouseReleased",
                "x": rect["x"] + rect["width"] / 2,
                "y": rect["y"] + rect["height"] / 2,
                "button": "left",
                "buttons": 1 if down else 0,
                "clickCount": 1,
            },
        )
        return self.evaluate(
            f"""
            (() => {{
              const element = document.querySelector({json.dumps(selector)});
              return {{
                pressed: element.classList.contains("pressed"),
                background: getComputedStyle(element).backgroundColor,
                transform: getComputedStyle(element).transform
              }};
            }})()
            """
        )

    def layout_fixture(self) -> dict[str, Any]:
        return self.evaluate(
            """
            (() => {
              document.querySelector("header").style.display = "none";
              const main = document.querySelector("main");
              main.style.display = "block";
              main.style.width = "823px";
              main.style.height = "526px";
              document.querySelector(".col.left").style.display = "none";
              document.querySelector("#gutterV").style.display = "none";
              const right = document.querySelector(".col.right");
              right.style.width = "823px";
              right.style.height = "526px";
              right.style.setProperty("--frameh", "410px");
              renderTimeline([{
                step: 49, action: "pi0_nav_pick", args: {},
                result: {chunks: 20, max_steps: 20},
                elapsed_s: 94.7, has_action_video: false
              }], false);
              document.querySelector("#stepCount").textContent = "56";
              return true;
            })()
            """
        )

    def high_dpi_layout_fixture(self) -> dict[str, Any]:
        self.set_device_metrics(
            width=853,
            height=578,
            device_scale_factor=1.5,
        )
        return self.evaluate(
            """
            (() => {
              document.querySelector("header").style.display = "none";
              const main = document.querySelector("main");
              main.style.display = "block";
              main.style.width = `${1280 / 1.5}px`;
              main.style.height = `${867 / 1.5}px`;
              document.querySelector(".col.left").style.display = "none";
              document.querySelector("#gutterV").style.display = "none";
              const right = document.querySelector(".col.right");
              right.style.width = `${1280 / 1.5}px`;
              right.style.height = `${867 / 1.5}px`;
              right.style.setProperty("--frameh", "438px");
              renderTimeline([
                {
                  step: 49, action: "pi0_nav_pick", args: {},
                  result: {chunks: 20, max_steps: 20},
                  elapsed_s: 94.7, has_action_video: false
                },
                {
                  step: 50, action: "observe", args: {},
                  result: {}, elapsed_s: 0.19, has_action_video: false
                }
              ], false);
              document.querySelector("#stepCount").textContent = "56";
              frameIdx = -1;
              refreshFrame(1587, {source: "user"});
              return true;
            })()
            """
        )

    def screenshot_clip(
        self,
        path: Path,
        *,
        width: float = 823,
        height: float = 526,
    ) -> None:
        # Keep acceptance artifacts free of hover-only tooltips.
        self.call(
            "Input.dispatchMouseEvent",
            {"type": "mouseMoved", "x": 400, "y": 380, "button": "none"},
        )
        result = self.call(
            "Page.captureScreenshot",
            {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": False,
                "clip": {
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": height,
                    "scale": 1,
                },
            },
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(result["data"]))

    def close(self) -> None:
        self.ws.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.log.close()
        self.profile.cleanup()


@pytest.fixture(scope="module")
def visual_dashboard():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256
    mock = _MockDashboard()
    server = _DashboardHTTPServer(mock)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    page = _ChromePage(f"http://127.0.0.1:{server.server_port}/")
    try:
        yield page, mock
    finally:
        page.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def dashboard(visual_dashboard):
    page, mock = visual_dashboard
    mock.reset()
    page.reload()
    yield page, mock
    page.evaluate("stopActiveLease('test_cleanup')")
    page.sleep(120)


@pytest.fixture
def isolated_dashboard():
    """Fresh browser process for timing-sensitive arbiter admission checks."""
    mock = _MockDashboard()
    server = _DashboardHTTPServer(mock)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    page = _ChromePage(f"http://127.0.0.1:{server.server_port}/")
    try:
        yield page, mock
    finally:
        page.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _dispatch_pointer(page: _ChromePage, selector: str, event: str, _pointer_id: int):
    if event == "pointerdown":
        return page.mouse(selector, down=True)
    if event == "pointerup":
        return page.mouse(selector, down=False)
    return page.evaluate(
        f"""
        (() => {{
          const element = document.querySelector({json.dumps(selector)});
          element.dispatchEvent(new PointerEvent({json.dumps(event)}, {{
            bubbles: true, cancelable: true, pointerId: {_pointer_id},
            pointerType: "mouse", button: 0, isPrimary: true
          }}));
          return {{
            pressed: element.classList.contains("pressed"),
            background: getComputedStyle(element).backgroundColor,
            transform: getComputedStyle(element).transform
          }};
        }})()
        """
    )


def _wait_commands(mock: _MockDashboard, count: int, timeout: float = 3.0):
    return _wait_until(
        lambda: len(mock.commands) >= count and list(mock.commands),
        timeout=timeout,
    )


def _wait_stops(mock: _MockDashboard, count: int, timeout: float = 2.0):
    return _wait_until(
        lambda: len(mock.stops) >= count and list(mock.stops),
        timeout=timeout,
    )


def test_reference_fixture_has_exact_approved_identity():
    with Image.open(REFERENCE) as image:
        assert image.size == (823, 526)
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256


def test_initial_state_collapse_tabs_and_equal_targets(dashboard):
    page, _mock = dashboard
    state = page.evaluate(
        """
        (() => {
          const targets = [...document.querySelectorAll(".target-button")];
          return {
            expanded: controlsExpanded,
            target: selectedTarget,
            camera: selectedCamera,
            wrap: document.querySelector("#framewrap").className,
            tabs: [...document.querySelectorAll(".behavior-frame-tabs button")].map(
              button => ({
                camera: button.dataset.camera,
                active: button.classList.contains("active"),
                pressed: button.getAttribute("aria-pressed")
              })
            ),
            targets: targets.map(button => {
              const rect = button.getBoundingClientRect();
              const style = getComputedStyle(button);
              return {
                width: rect.width, height: rect.height,
                padding: style.padding, fontSize: style.fontSize,
                borderRadius: style.borderRadius, boxShadow: style.boxShadow
              };
            })
          };
        })()
        """
    )
    assert state["expanded"] is True
    assert state["target"] == "chassis"
    assert state["camera"] == "head"
    assert "behavior-mode" in state["wrap"]
    assert state["tabs"] == [
        {"camera": "head", "active": True, "pressed": "true"},
        {"camera": "left_wrist", "active": False, "pressed": "false"},
        {"camera": "right_wrist", "active": False, "pressed": "false"},
    ]
    widths = [target["width"] for target in state["targets"]]
    heights = [target["height"] for target in state["targets"]]
    assert max(widths) - min(widths) <= 0.5
    assert max(heights) - min(heights) <= 0.5
    for property_name in ("padding", "fontSize", "borderRadius", "boxShadow"):
        assert len({target[property_name] for target in state["targets"]}) == 1

    page.evaluate("document.querySelector('.controls-toggle').click()")
    page.sleep(60)
    collapsed = page.evaluate(
        """
        (() => {
          const wrap = document.querySelector("#framewrap");
          const stage = document.querySelector(".frame-stage").getBoundingClientRect();
          return {
            expanded: controlsExpanded,
            className: wrap.className,
            tabsVisible: getComputedStyle(document.querySelector(".behavior-frame-tabs")).display,
            railsVisible: [...document.querySelectorAll(".control-rail")].map(
              rail => getComputedStyle(rail).display
            ),
            stageWidth: stage.width,
            wrapWidth: wrap.getBoundingClientRect().width,
            aria: document.querySelector(".collapsed-toggle").getAttribute("aria-expanded")
          };
        })()
        """
    )
    assert collapsed["expanded"] is False
    assert "controls-collapsed" in collapsed["className"]
    assert collapsed["tabsVisible"] == "flex"
    assert collapsed["railsVisible"] == ["none", "none"]
    assert collapsed["stageWidth"] == pytest.approx(collapsed["wrapWidth"], abs=0.5)
    assert collapsed["aria"] == "false"

    page.evaluate("document.querySelector('.collapsed-toggle').click()")
    page.sleep(60)
    assert page.evaluate(
        "controlsExpanded && selectedTarget === 'chassis' && selectedCamera === 'head'"
    )


def test_dynamic_tooltips_pressed_style_and_capability_gate(dashboard):
    page, _mock = dashboard
    chassis = page.evaluate(
        """
        (() => {
          const get = action => {
            const button = document.querySelector(`[data-action="${action}"]`);
            return { disabled: button.getAttribute("aria-disabled"), tip: button.dataset.tooltip };
          };
          return { forward: get("forward"), rotate: get("rotate_left"), open: get("open") };
        })()
        """
    )
    assert chassis["forward"] == {
        "disabled": "false",
        "tip": "Move the chassis forward by 5 cm. Hold to continue.",
    }
    assert chassis["rotate"] == {
        "disabled": "true",
        "tip": "Available for arm control only.",
    }
    assert chassis["open"] == {
        "disabled": "true",
        "tip": "Available for arm control only.",
    }

    page.evaluate("document.querySelector('[data-target=\"left_arm\"]').click()")
    page.sleep(60)
    arm = page.evaluate(
        """
        (() => {
          const get = action => {
            const button = document.querySelector(`[data-action="${action}"]`);
            return { disabled: button.getAttribute("aria-disabled"), tip: button.dataset.tooltip };
          };
          return { forward: get("forward"), left: get("turn_left"), rotate: get("rotate_left") };
        })()
        """
    )
    assert arm["forward"]["tip"] == (
        "Move the left hand forward by 3 cm. Hold to continue."
    )
    assert arm["left"]["tip"] == (
        "Move the left hand 3 cm to the robot’s left. Hold to continue."
    )
    assert arm["rotate"] == {
        "disabled": "false",
        "tip": "Rotate the selected wrist 5° clockwise. Hold to continue.",
    }

    button = '[data-action="forward"]'
    resting = page.evaluate(
        f"getComputedStyle(document.querySelector({json.dumps(button)})).backgroundColor"
    )
    pressed = _dispatch_pointer(page, button, "pointerdown", 101)
    assert pressed["pressed"] is True
    assert pressed["background"] != resting
    assert pressed["transform"] != "none"
    released = _dispatch_pointer(page, button, "pointerup", 101)
    page.sleep(30)
    assert released["pressed"] is False


def test_short_press_once_and_long_press_fills_bounded_tail(dashboard):
    page, mock = dashboard
    selector = '[data-action="forward"]'

    # A short click submits sequence 1 immediately and clears only pending
    # tail work on release. The accepted head remains executable.
    mock.reset(command_delay_s=0.24)
    page.reload()
    _dispatch_pointer(page, selector, "pointerdown", 201)
    page.sleep(55)
    _dispatch_pointer(page, selector, "pointerup", 201)
    page.sleep(320)
    assert len(mock.commands) == 1
    assert len(mock.stops) >= 1
    command = mock.commands[0]
    stop = mock.stops[0]
    assert command["cancelled"] is False
    assert stop["reason"] == "released"
    assert stop["stop_mode"] == "clear_pending"
    assert stop["at"] < command["complete_at"]

    page.evaluate("stopActiveLease('between_press_cases')")
    page.sleep(120)
    mock.reset(command_delay_s=0.45)
    page.reload()
    _dispatch_pointer(page, selector, "pointerdown", 202)
    page.sleep(1050)
    _dispatch_pointer(page, selector, "pointerup", 202)
    page.sleep(120)
    commands = list(mock.commands)
    # Six commands proves the initial head + five-tail fill; the seventh
    # proves the same pump refilled after queue_depth dropped.
    assert len(commands) >= 7
    accepted = [item["accepted_at"] for item in commands]
    intervals_ms = [
        (later - earlier) * 1000
        for earlier, later in zip(accepted, accepted[1:], strict=False)
    ]
    assert intervals_ms[0] >= 315
    assert all(interval >= 95 for interval in intervals_ms[1:])
    assert mock.max_in_flight == 1
    assert mock.max_command_posts_in_flight == 1
    assert mock.max_queue_depth <= 5
    assert len(mock.heartbeats) >= 2
    heartbeat_intervals_ms = [
        (later["at"] - earlier["at"]) * 1000
        for earlier, later in zip(mock.heartbeats, mock.heartbeats[1:], strict=False)
    ]
    assert all(350 <= interval <= 500 for interval in heartbeat_intervals_ms)
    assert [item["sequence"] for item in commands] == list(range(1, len(commands) + 1))


def test_repeat_does_not_wait_for_terminal_and_stop_clears_only_tail(
    isolated_dashboard,
):
    page, mock = isolated_dashboard
    mock.reset(command_delay_s=1.2)
    page.reload()

    page.mouse('[data-action="forward"]', down=True)
    _wait_commands(mock, 6, timeout=2)
    first, second = mock.commands[:2]
    assert second["accepted_at"] < first["complete_at"]
    assert first.get("terminal_receipt_at") is None
    page.mouse('[data-action="forward"]', down=False)
    _wait_stops(mock, 1)
    page.sleep(100)

    assert mock.commands[0]["cancelled"] is False
    assert all(command["cancelled"] for command in mock.commands[1:6])
    assert mock.max_in_flight == 1
    assert mock.max_command_posts_in_flight == 1
    assert mock.max_queue_depth == 5


def test_release_before_first_202_retries_clear_pending_after_acceptance(
    isolated_dashboard,
):
    page, mock = isolated_dashboard
    mock.reset(command_delay_s=0.6, accept_delay_s=0.18)
    page.reload()

    page.mouse('[data-action="forward"]', down=True)
    page.sleep(25)
    page.mouse('[data-action="forward"]', down=False)

    _wait_commands(mock, 1)
    _wait_stops(mock, 2)
    assert len(mock.commands) == 1
    assert mock.commands[0]["sequence"] == 1
    assert mock.commands[0]["cancelled"] is False
    assert mock.stops[0]["at"] < mock.commands[0]["accepted_at"]
    assert mock.stops[-1]["at"] >= mock.commands[0]["accepted_at"]
    assert all(stop["stop_mode"] == "clear_pending" for stop in mock.stops)


def test_camera_change_does_not_stop_motion_and_lease_payload_is_frozen(
    dashboard,
):
    page, mock = dashboard
    mock.reset(command_delay_s=1.2)
    page.reload()

    page.mouse('[data-action="forward"]', down=True)
    _wait_commands(mock, 1)
    page.evaluate("setFrameKind('left_wrist')")
    _wait_until(
        lambda: len(mock.cameras) == 1
        and mock.cameras[0]["state"] == "completed"
    )
    _wait_commands(mock, 6, timeout=2)

    assert not mock.stops
    assert all(command["target"] == "chassis" for command in mock.commands)
    assert all(command["action"] == "forward" for command in mock.commands)
    assert all(command["camera"] == "head" for command in mock.commands)
    assert page.evaluate(
        """
        selectedCamera === "left_wrist" &&
        controlQueueDepth() === 5 &&
        document.querySelector(".control-status").textContent.includes("q 5/5")
        """
    )

    page.mouse('[data-action="forward"]', down=False)
    _wait_stops(mock, 1)


def test_control_revision_rejects_stale_state_and_capture_is_not_quiescence(
    dashboard,
):
    page, _mock = dashboard
    state = page.evaluate(
        """
        (() => {
          applyControlSnapshot({
            control_revision: 50,
            available: true,
            busy: true,
            owner: "manual",
            lease_id: "lease-50",
            current_command: {command_id:"head", phase:"moving"},
            planning_command: null,
            queue_depth: 4,
            queue_capacity: 5,
            capture: {phase:"failed", revision:7, error:"camera failed"}
          });
          applyControlSnapshot({
            control_revision: 49,
            busy: false,
            owner: null,
            lease_id: null,
            current_command: null,
            queue_depth: 0
          });
          return {
            revision: controlSnapshot.control_revision,
            lease: controlSnapshot.lease_id,
            queueDepth: controlSnapshot.queue_depth,
            capture: controlSnapshot.capture,
            captureOnlyBusy: controlCaptureOnlyBusy({
              busy:true, owner:"manual", lease_status:"stopped",
              current_command:null, planning_command:null, queue_depth:0,
              capture:{phase:"started", revision:8, error:null}
            }),
            quiescentWithCapture: manualControlIsQuiescent({
              busy:true, owner:"manual", lease_status:"stopped",
              current_command:null, planning_command:null, queue_depth:0,
              capture:{phase:"started", revision:8, error:null}
            }),
            agentNotQuiescent: manualControlIsQuiescent({
              busy:true, owner:"agent", lease_status:"idle",
              current_command:null, planning_command:null, queue_depth:0,
              capture:{phase:"idle", revision:8, error:null}
            })
          };
        })()
        """
    )
    assert state == {
        "revision": 50,
        "lease": "lease-50",
        "queueDepth": 4,
        "capture": {"phase": "failed", "revision": 7, "error": "camera failed"},
        "captureOnlyBusy": True,
        "quiescentWithCapture": True,
        "agentNotQuiescent": False,
    }


def test_sse_uses_timeline_and_per_camera_frame_revisions(dashboard):
    page, mock = dashboard
    with mock.lock:
        run_gets_before = mock.run_get_count
        frame_gets_before = mock.frame_get_count
    page.evaluate(
        """
        es.onmessage({data: JSON.stringify({
          state: "running",
          terminated: false,
          n_steps: 56,
          timeline_revision: 57,
          frame_idx: 1588,
          frame_revisions: {
            head: 1588, left_wrist: 1587, right_wrist: 1587
          }
        })})
        """
    )
    _wait_until(lambda: mock.run_get_count > run_gets_before)
    _wait_until(lambda: mock.frame_get_count > frame_gets_before)
    _wait_until(
        lambda: page.evaluate(
            """
            timelineRevision === 57 &&
            frameRevisions.head === 1588 &&
            frameIdx === 1588
            """
        )
    )


@pytest.mark.parametrize("terminal_kind", ["failed", "raw_success"])
def test_error_or_raw_success_stops_repeat_pump(dashboard, terminal_kind: str):
    page, mock = dashboard
    mock.reset(command_delay_s=1.2)
    page.reload()
    page.mouse('[data-action="forward"]', down=True)
    _wait_commands(mock, 3)
    first_id = mock.commands[0]["command_id"]
    page.evaluate(
        f"""
        applyControlSnapshot({{
          control_revision: controlSnapshot.control_revision + 100,
          available: {str(terminal_kind != "raw_success").lower()},
          busy: false,
          owner: null,
          lease_id: activeLeaseId,
          lease_status: "active",
          current_command: null,
          planning_command: null,
          queue_depth: controlQueueDepth(),
          queue_capacity: 5,
          success_latched: {str(terminal_kind == "raw_success").lower()},
          last_terminal: {{
            command_id: {json.dumps(first_id)},
            sequence: 1,
            phase: {json.dumps("completed" if terminal_kind == "raw_success" else "failed")},
            error: {json.dumps(None if terminal_kind == "raw_success" else "planner failed")},
            task_success: {str(terminal_kind == "raw_success").lower()}
          }}
        }})
        """
    )
    _wait_stops(mock, 1)
    count = len(mock.commands)
    page.sleep(450)
    assert len(mock.commands) == count
    assert mock.stops[-1]["reason"] == (
        "official_success" if terminal_kind == "raw_success" else "command_failed"
    )
    page.mouse('[data-action="forward"]', down=False)


@pytest.mark.parametrize(
    ("trigger", "expected_reason"),
    [
        (
            "document.querySelector('[data-action=\"forward\"]').dispatchEvent("
            "new PointerEvent('pointercancel', "
            "{bubbles:true,pointerId:activePointerId,pointerType:'mouse'}))",
            "released",
        ),
        (
            "document.querySelector('[data-action=\"forward\"]').dispatchEvent("
            "new PointerEvent('lostpointercapture', "
            "{bubbles:true,pointerId:activePointerId,pointerType:'mouse'}))",
            "released",
        ),
        ("window.dispatchEvent(new Event('blur'))", "window_blur"),
        (
            "Object.defineProperty(document, 'hidden', {value:true, configurable:true});"
            "document.dispatchEvent(new Event('visibilitychange'))",
            "page_hidden",
        ),
        (
            "document.dispatchEvent(new KeyboardEvent('keydown', "
            "{key:'Escape',bubbles:true,cancelable:true}))",
            "escape",
        ),
        ("document.querySelector('.controls-toggle').click()", "controls_collapsed"),
        (
            "document.querySelector('[data-target=\"left_arm\"]').click()",
            "target_changed",
        ),
        ("selectRun('mock-run')", "run_changed"),
    ],
)
def test_every_stop_event_ends_the_active_lease(
    dashboard, trigger: str, expected_reason: str
):
    page, mock = dashboard
    # Use a stable synthetic pointer id for this event matrix. Other tests
    # exercise real CDP mouse input; here deterministic identity matters
    # because pointercancel/lostpointercapture are themselves the subject.
    page.evaluate(
        """
        document.querySelector('[data-action="forward"]').dispatchEvent(
          new PointerEvent('pointerdown', {
            bubbles:true, cancelable:true, pointerId:301,
            pointerType:'mouse', button:0, isPrimary:true
          })
        )
        """
    )
    _wait_commands(mock, 1)
    page.sleep(60)
    page.evaluate(trigger)
    stops = _wait_until(
        lambda: (
            list(mock.stops)
            if any(item["reason"] == expected_reason for item in mock.stops)
            else None
        ),
        timeout=2,
    )
    assert any(item["reason"] == expected_reason for item in stops)
    page.evaluate(
        """
        document.querySelector('[data-action="forward"]').dispatchEvent(
          new PointerEvent('pointerup', {
            bubbles:true, cancelable:true, pointerId:301,
            pointerType:'mouse', button:0, isPrimary:true
          })
        )
        """
    )
    count = len(mock.commands)
    page.sleep(430)
    assert len(mock.commands) == count


def test_keyboard_ignores_auto_repeat_and_keyup_stops(dashboard):
    page, mock = dashboard
    page.evaluate(
        """
        (() => {
          const button = document.querySelector('[data-action="forward"]');
          button.dispatchEvent(new KeyboardEvent('keydown', {
            key: ' ', bubbles: true, cancelable: true, repeat: false
          }));
          button.dispatchEvent(new KeyboardEvent('keydown', {
            key: ' ', bubbles: true, cancelable: true, repeat: true
          }));
        })()
        """
    )
    page.sleep(100)
    page.evaluate(
        """
        document.querySelector('[data-action="forward"]').dispatchEvent(
          new KeyboardEvent('keyup', {key:' ', bubbles:true, cancelable:true})
        )
        """
    )
    page.sleep(360)
    assert len(mock.commands) == 1
    assert mock.stops[-1]["reason"] == "released"


@pytest.mark.parametrize("action", ["observe", "open", "close"])
def test_one_shot_actions_never_enter_repeat_loop(dashboard, action: str):
    page, mock = dashboard
    page.layout_fixture()
    if action in {"open", "close"}:
        page.evaluate("document.querySelector('[data-target=\"left_arm\"]').click()")
        page.sleep(60)
    selector = f'[data-action="{action}"]'
    page.mouse(selector, down=True)
    page.sleep(720)
    page.mouse(selector, down=False)
    page.sleep(120)
    assert len(mock.commands) == 1
    assert mock.commands[0]["action"] == action
    assert len(mock.stops) == 1
    assert mock.stops[0]["stop_mode"] == "clear_pending"


def test_target_collapse_transitions_are_last_click_wins(dashboard):
    page, mock = dashboard
    mock.reset(command_delay_s=1.5, stop_delay_s=0.16)
    page.reload()
    page.evaluate(
        """
        document.querySelector('[data-action="forward"]').dispatchEvent(
          new PointerEvent('pointerdown', {
            bubbles:true, cancelable:true, pointerId:901,
            pointerType:'mouse', button:0, isPrimary:true
          })
        )
        """
    )
    _wait_commands(mock, 1)

    target_results = page.evaluate(
        """
        Promise.all([
          selectControlTarget("left_arm"),
          selectControlTarget("right_arm")
        ])
        """
    )
    assert target_results == [False, True]
    assert page.evaluate(
        """
        selectedTarget === "right_arm" &&
        desiredTarget === "right_arm" &&
        document.querySelector('[data-target="right_arm"]')
          .getAttribute("aria-pressed") === "true"
        """
    )

    collapse_results = page.evaluate(
        """
        Promise.all([
          setControlsExpanded(false),
          setControlsExpanded(true)
        ])
        """
    )
    assert collapse_results == [False, True]
    assert page.evaluate(
        """
        controlsExpanded && desiredControlsExpanded &&
        !document.querySelector("#framewrap").classList.contains("controls-collapsed")
        """
    )


def test_pending_target_and_collapse_cannot_pollute_a_new_run(dashboard):
    page, _mock = dashboard
    result = page.evaluate(
        """
        (async () => {
          const original = stopAndWaitForManualControl;
          let release;
          const gate = new Promise(resolve => { release = resolve; });
          stopAndWaitForManualControl = () => gate;
          const target = selectControlTarget("left_arm");
          const collapse = setControlsExpanded(false);
          const run = selectRun("mock-run-next");
          release(true);
          const settled = await Promise.all([target, collapse, run]);
          stopAndWaitForManualControl = original;
          return {
            settled,
            curRun,
            selectedTarget,
            desiredTarget,
            controlsExpanded,
            desiredControlsExpanded
          };
        })()
        """
    )
    assert result == {
        "settled": [False, False, True],
        "curRun": "mock-run-next",
        "selectedTarget": "chassis",
        "desiredTarget": "chassis",
        "controlsExpanded": True,
        "desiredControlsExpanded": True,
    }


def test_camera_requests_serialize_latest_intent_and_failure_rolls_back(dashboard):
    page, mock = dashboard
    with mock.lock:
        mock.camera_delays["left_wrist"] = 0.22

    page.evaluate("setFrameKind('left_wrist')")
    _wait_until(lambda: len(mock.cameras) == 1)
    page.evaluate("setFrameKind('right_wrist')")
    _wait_until(
        lambda: (
            len(mock.cameras) == 2
            and all(item["state"] == "completed" for item in mock.cameras)
        ),
        timeout=3,
    )
    assert mock.max_camera_in_flight == 1
    assert [item["camera"] for item in mock.cameras] == [
        "left_wrist",
        "right_wrist",
    ]
    assert page.evaluate(
        """
        selectedCamera === "right_wrist" &&
        publishedCamera === "right_wrist" &&
        frameKind === "right_wrist" &&
        document.querySelector('[data-camera="right_wrist"]')
          .getAttribute("aria-pressed") === "true"
        """
    )

    with mock.lock:
        mock.camera_failures.add("head")
    page.evaluate("setFrameKind('head')")
    _wait_until(
        lambda: len(mock.cameras) == 3 and mock.cameras[-1]["state"] == "failed"
    )
    page.sleep(80)
    assert page.evaluate(
        """
        selectedCamera === "right_wrist" &&
        publishedCamera === "right_wrist" &&
        frameKind === "right_wrist" &&
        document.querySelector('[data-camera="right_wrist"]')
          .getAttribute("aria-pressed") === "true" &&
        document.querySelector('[data-camera="head"]')
          .getAttribute("aria-pressed") === "false"
        """
    )


def test_failed_video_preserves_visible_frame_and_releases_swap_queue(dashboard):
    page, mock = dashboard
    before = _wait_until(
        lambda: page.evaluate(
            """
            (() => {
              const visible = document.querySelector("img.visible");
              return visible ? {element: visible.id, src: visible.src} : null;
            })()
            """
        )
    )
    assert before["element"]
    assert page.evaluate(
        """
        playActionVideo(
          {step: 49, action: "pi0_nav_pick", has_action_video: true},
          {returnAfterEnd: true, returnKind: "head", nextFrameIdx: 1588}
        )
        """
    )
    _wait_until(
        lambda: page.evaluate(
            """
            swapInFlight === false &&
            swapQueue.length === 0 &&
            document.querySelector("#frameCap").textContent.includes(
              "Media unavailable"
            )
            """
        )
    )
    after_error = page.evaluate(
        """
        (() => {
          const visible = document.querySelector("img.visible");
          return {element: visible?.id, src: visible?.src};
        })()
        """
    )
    assert after_error == before

    page.evaluate("setFrameKind('right_wrist')")
    _wait_until(
        lambda: len(mock.cameras) == 1 and mock.cameras[0]["state"] == "completed"
    )
    _wait_until(
        lambda: page.evaluate(
            """
            frameKind === "right_wrist" &&
            selectedCamera === "right_wrist" &&
            swapInFlight === false
            """
        )
    )


def test_unavailable_target_has_disabled_semantics_and_visual_state(dashboard):
    page, mock = dashboard
    state = page.evaluate(
        """
        (() => {
          const button = document.querySelector('[data-target="left_arm"]');
          const enabledStyle = getComputedStyle(button);
          const enabled = {
            background: enabledStyle.backgroundImage,
            border: enabledStyle.borderColor,
            color: enabledStyle.color,
            cursor: enabledStyle.cursor
          };
          controlSnapshot = {...controlSnapshot, available: false};
          renderControlUI();
          const disabledStyle = getComputedStyle(button);
          button.click();
          return {
            enabled,
            disabled: {
              background: disabledStyle.backgroundImage,
              border: disabledStyle.borderColor,
              color: disabledStyle.color,
              cursor: disabledStyle.cursor
            },
            ariaDisabled: button.getAttribute("aria-disabled"),
            selectedTarget,
            splitterV: document.querySelector("#gutterV").title,
            splitterH: document.querySelector("#gutterH").title
          };
        })()
        """
    )
    assert state["ariaDisabled"] == "true"
    assert state["selectedTarget"] == "chassis"
    assert state["disabled"]["cursor"] == "not-allowed"
    assert state["disabled"]["background"] != state["enabled"]["background"]
    assert state["disabled"]["border"] != state["enabled"]["border"]
    assert state["disabled"]["color"] != state["enabled"]["color"]
    assert state["splitterV"] == "Drag to resize the column width."
    assert state["splitterH"] == "Drag to resize the frame height."
    assert not mock.commands
    assert not mock.cameras


@pytest.mark.parametrize("playback_kind", ["action", "episode"])
def test_playback_stops_active_lease_and_admits_no_repeat(
    dashboard, playback_kind: str
):
    page, mock = dashboard
    mock.reset(command_delay_s=1.5)
    page.reload()
    page.evaluate(
        """
        document.querySelector('[data-action="forward"]').dispatchEvent(
          new PointerEvent('pointerdown', {
            bubbles:true, cancelable:true, pointerId:902,
            pointerType:'mouse', button:0, isPrimary:true
          })
        );
        window.__videoPlayCalls = 0;
        for (const video of document.querySelectorAll("video")) {
          video.load = function() {
            queueMicrotask(() => this.dispatchEvent(new Event("canplay")));
          };
          video.play = function() {
            window.__videoPlayCalls++;
            return Promise.resolve();
          };
        }
        """
    )
    _wait_commands(mock, 1)

    if playback_kind == "action":
        result = page.evaluate(
            """
            playActionVideo(
              {step: 49, action: "pi0_nav_pick", has_action_video: true},
              {returnAfterEnd: true, returnKind: "head", nextFrameIdx: 1588}
            )
            """
        )
        reason = "action_playback"
    else:
        result = page.evaluate(
            """
            episodeVideoAvailable = true;
            playEpisodeVideo()
            """
        )
        reason = "episode_playback"
    assert result is True
    _wait_until(lambda: page.evaluate("window.__videoPlayCalls > 0"))
    assert any(item["reason"] == reason for item in mock.stops)
    command_count = len(mock.commands)
    page.sleep(720)
    assert len(mock.commands) == command_count == 1
    assert page.evaluate(
        """
        activeLeaseId === null &&
        activeLeaseToken === null &&
        controlCommandInFlight === false &&
        controlLoopPromise === null &&
        stopLeasePromise === null
        """
    )


def test_action_video_returns_to_previously_selected_camera(dashboard):
    page, _mock = dashboard
    page.evaluate("document.querySelector('[data-camera=\"left_wrist\"]').click()")
    page.sleep(120)
    assert page.evaluate(
        "selectedCamera === 'left_wrist' && frameKind === 'left_wrist'"
    )
    started = page.evaluate(
        """
        for (const video of document.querySelectorAll('video')) {
          video.load = function() {
            queueMicrotask(() => this.dispatchEvent(new Event("canplay")));
          };
          video.play = () => Promise.resolve();
        }
        playActionVideo(
          {step: 49, action: 'pi0_nav_pick', has_action_video: true},
          {returnAfterEnd: true, returnKind: lastRealtimeFrameKind, nextFrameIdx: 1588}
        )
        """
    )
    assert started is True
    _wait_until(lambda: page.evaluate("document.querySelector('video.visible') !== null"))
    page.evaluate(
        """
        (() => {
          const video = document.querySelector('video.visible') ||
            document.querySelector('video');
          video.dispatchEvent(new Event('ended'));
        })()
        """
    )
    page.sleep(380)
    restored = page.evaluate(
        """
        ({
          selectedCamera,
          frameKind,
          active: document.querySelector(
            '.behavior-frame-tabs [data-camera="left_wrist"]'
          ).classList.contains('active')
        })
        """
    )
    assert restored == {
        "selectedCamera": "left_wrist",
        "frameKind": "left_wrist",
        "active": True,
    }


def test_reference_layout_coordinates_and_write_acceptance_artifacts(dashboard):
    page, _mock = dashboard
    page.layout_fixture()
    page.sleep(120)
    geometry = page.evaluate(
        """
        (() => {
          const rect = selector => {
            const r = document.querySelector(selector).getBoundingClientRect();
            return {x:r.x, y:r.y, width:r.width, height:r.height};
          };
          return {
            frame: rect("#framewrap"),
            leftRail: rect(".control-left"),
            stage: rect(".frame-stage"),
            rightRail: rect(".control-right"),
            toggle: rect(".control-left .controls-toggle"),
            chassis: rect('[data-target="chassis"]'),
            leftArm: rect('[data-target="left_arm"]'),
                rightArm: rect('[data-target="right_arm"]'),
                tabs: rect(".behavior-frame-tabs"),
                tabHead: rect('[data-camera="head"]'),
                tabLeft: rect('[data-camera="left_wrist"]'),
                tabRight: rect('[data-camera="right_wrist"]'),
                dpad: rect(".dpad"),
                forward: rect('[data-action="forward"]'),
                backward: rect('[data-action="backward"]'),
                turnLeft: rect('[data-action="turn_left"]'),
                turnRight: rect('[data-action="turn_right"]'),
                observe: rect('[data-action="observe"]'),
                up: rect('[data-action="up"]'),
                down: rect('[data-action="down"]'),
                rotateLeft: rect('[data-action="rotate_left"]'),
                rotateRight: rect('[data-action="rotate_right"]'),
                open: rect('[data-action="open"]'),
                close: rect('[data-action="close"]'),
                frameCap: rect("#frameCap"),
                frameCapText: document.querySelector("#frameCap").textContent.trim(),
                timelineTitleText: document.querySelector(
                  ".col.right > .col .panel-title"
                ).textContent.replace(/\\s+/g, " ").trim(),
                timelineTitle: rect(".col.right > .col .panel-title")
          };
        })()
        """
    )
    expected = {
        "frame": {"x": 0, "y": 0, "width": 823, "height": 410},
        "leftRail": {"x": 0, "y": 0, "width": 211, "height": 410},
        "stage": {"x": 211, "y": 0, "width": 408, "height": 410},
        "rightRail": {"x": 619, "y": 0, "width": 204, "height": 410},
        "toggle": {"x": 22, "y": 8, "width": 162, "height": 27},
        "chassis": {"x": 63, "y": 69, "width": 76, "height": 26},
        "leftArm": {"x": 646, "y": 69, "width": 76, "height": 26},
        "rightArm": {"x": 731, "y": 69, "width": 76, "height": 26},
        "dpad": {"x": 51, "y": 146, "width": 96, "height": 96},
        "forward": {"x": 83, "y": 146, "width": 32, "height": 32},
        "backward": {"x": 83, "y": 210, "width": 32, "height": 32},
        "turnLeft": {"x": 51, "y": 178, "width": 32, "height": 32},
        "turnRight": {"x": 115, "y": 178, "width": 32, "height": 32},
        "observe": {"x": 79, "y": 287, "width": 39, "height": 39},
        "up": {"x": 668, "y": 131, "width": 39, "height": 39},
        "down": {"x": 748, "y": 131, "width": 39, "height": 39},
        "rotateLeft": {"x": 668, "y": 218, "width": 39, "height": 39},
        "rotateRight": {"x": 748, "y": 218, "width": 39, "height": 39},
        "open": {"x": 668, "y": 305, "width": 39, "height": 39},
        "close": {"x": 748, "y": 305, "width": 39, "height": 39},
        "frameCap": {"x": 10, "y": 374, "height": 27},
        "timelineTitle": {"x": 0, "y": 416, "width": 823, "height": 36},
    }
    page.screenshot_clip(LIVE_SCREENSHOT)
    with Image.open(LIVE_SCREENSHOT) as live, Image.open(REFERENCE) as reference:
        assert live.size == reference.size == (823, 526)
        overlay = Image.blend(reference.convert("RGB"), live.convert("RGB"), alpha=0.5)
        OVERLAY_SCREENSHOT.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(OVERLAY_SCREENSHOT)
        difference = ImageChops.difference(
            reference.convert("RGB"), live.convert("RGB")
        )
        changed = sum(1 for pixel in difference.getdata() if pixel != (0, 0, 0))
        assert changed > 0  # Product controls are live DOM, not a copied bitmap.

        # Compare deterministic UI chrome only. Simulator RGB, the dynamic
        # frame caption, and Timeline content are deliberately excluded.
        # The mask covers both control rails (including the camera tabs) while
        # retaining the reference-approved low-saturation selected state and
        # capability-disabled controls as intentional product differences.
        chrome_mask = Image.new("L", reference.size, 0)
        chrome_draw = ImageDraw.Draw(chrome_mask)
        chrome_draw.rectangle((0, 0, 210, 369), fill=255)
        chrome_draw.rectangle((619, 0, 822, 409), fill=255)
        # The immutable reference intentionally remains the before-state.
        # Geometry assertions below validate the explicitly moved left control
        # group; exclude the union of its old/new footprints from the pixel
        # similarity score so that the requested 22 px move is not counted as
        # a visual regression twice.
        chrome_draw.rectangle((8, 39, 194, 348), fill=0)
        chrome_mae = sum(ImageStat.Stat(difference, chrome_mask).mean) / 3
        chrome_pixels = [
            max(pixel)
            for pixel, included in zip(
                difference.getdata(), chrome_mask.getdata(), strict=True
            )
            if included
        ]
        chrome_within_24 = sum(value <= 24 for value in chrome_pixels) / len(
            chrome_pixels
        )
        assert chrome_mae <= 8.0, f"UI chrome MAE {chrome_mae:.3f} exceeded 8.0"
        assert chrome_within_24 >= 0.90, (
            "UI chrome pixels within 24 RGB levels "
            f"{chrome_within_24:.3%} fell below 90%"
        )

    assert LIVE_SCREENSHOT.exists()
    assert OVERLAY_SCREENSHOT.exists()

    errors = []
    for element, wanted in expected.items():
        for coordinate, value in wanted.items():
            actual = geometry[element][coordinate]
            if actual != pytest.approx(value, abs=1.0):
                errors.append(f"{element}.{coordinate}: {actual:.3f} != {value}")
    target_tops = {
        geometry["chassis"]["y"],
        geometry["leftArm"]["y"],
        geometry["rightArm"]["y"],
    }
    if len(target_tops) != 1:
        errors.append(
            "target controls are not vertically aligned: "
            f"{sorted(target_tops)!r}"
        )

    # The camera tabs are intentionally the original compact Dashboard tabs.
    for coordinate, value in (
        ("y", 4),
        ("x", 625),
        ("width", 196),
        ("height", 23),
    ):
        actual = geometry["tabs"][coordinate]
        if actual != pytest.approx(value, abs=1):
            errors.append(f"tabs.{coordinate}: {actual:.3f} != {value}")
    for element, wanted in {
        "tabHead": {"width": 48, "height": 23},
        "tabLeft": {"width": 63, "height": 23},
        "tabRight": {"width": 70, "height": 23},
    }.items():
        for coordinate, value in wanted.items():
            actual = geometry[element][coordinate]
            if actual != pytest.approx(value, abs=1):
                errors.append(f"{element}.{coordinate}: {actual:.3f} != {value}")
    if geometry["frameCapText"] != "head - frame #1587":
        errors.append(
            f"frameCapText: {geometry['frameCapText']!r} != 'head - frame #1587'"
        )
    expected_title = "ACTION TIMELINE 56 10 PRIMITIVES · CONTRACT V2"
    if geometry["timelineTitleText"].upper() != expected_title:
        errors.append(
            "timelineTitleText: "
            f"{geometry['timelineTitleText']!r} != {expected_title!r}"
        )
    assert not errors, "reference coordinate mismatches:\n" + "\n".join(errors)


def test_high_dpi_reference_layout_and_write_acceptance_artifacts(dashboard):
    if not HIGH_DPI_REFERENCE.exists():
        pytest.skip("optional local 1280x867 Dashboard reference is unavailable")
    assert (
        hashlib.sha256(HIGH_DPI_REFERENCE.read_bytes()).hexdigest()
        == HIGH_DPI_REFERENCE_SHA256
    )
    with Image.open(HIGH_DPI_REFERENCE) as reference:
        assert reference.size == (1280, 867)

    page, mock = dashboard
    with mock.lock:
        mock.use_high_dpi_frame = True
    try:
        page.high_dpi_layout_fixture()
        _wait_until(
            lambda: page.evaluate(
                """
                (() => {
                  const image = document.querySelector("img.visible");
                  return devicePixelRatio === 1.5
                    && image?.complete
                    && image.naturalWidth > 0
                    && document.querySelector("#framewrap")
                      .getBoundingClientRect().height === 438;
                })()
                """
            )
        )
        page.sleep(120)
        geometry = page.evaluate(
            """
            (() => {
              const rect = selector => {
                const r = document.querySelector(selector).getBoundingClientRect();
                return {x:r.x, y:r.y, width:r.width, height:r.height};
              };
              return {
                dpr: devicePixelRatio,
                viewport: {
                  width: document.documentElement.clientWidth,
                  height: document.documentElement.clientHeight
                },
                frame: rect("#framewrap"),
                leftRail: rect(".control-left"),
                stage: rect(".frame-stage"),
                rightRail: rect(".control-right"),
                toggle: rect(".control-left .controls-toggle"),
                tabs: rect(".behavior-frame-tabs"),
                chassis: rect('[data-target="chassis"]'),
                leftArm: rect('[data-target="left_arm"]'),
                rightArm: rect('[data-target="right_arm"]'),
                dpad: rect(".dpad"),
                observe: rect('[data-action="observe"]'),
                up: rect('[data-action="up"]'),
                rotateLeft: rect('[data-action="rotate_left"]'),
                open: rect('[data-action="open"]')
              };
            })()
            """
        )
        page.screenshot_clip(
            HIGH_DPI_LIVE_SCREENSHOT,
            width=1280 / 1.5,
            height=867 / 1.5,
        )
        with (
            Image.open(HIGH_DPI_LIVE_SCREENSHOT) as live,
            Image.open(HIGH_DPI_REFERENCE) as reference,
        ):
            assert live.size == reference.size == (1280, 867)
            overlay = Image.blend(
                reference.convert("RGB"),
                live.convert("RGB"),
                alpha=0.5,
            )
            HIGH_DPI_OVERLAY_SCREENSHOT.parent.mkdir(
                parents=True, exist_ok=True
            )
            overlay.save(HIGH_DPI_OVERLAY_SCREENSHOT)
            assert ImageChops.difference(
                reference.convert("RGB"), live.convert("RGB")
            ).getbbox() is not None

        assert HIGH_DPI_LIVE_SCREENSHOT.exists()
        assert HIGH_DPI_OVERLAY_SCREENSHOT.exists()
        assert geometry["dpr"] == 1.5
        assert geometry["frame"]["height"] == pytest.approx(438, abs=0.5)
        expected_physical_chrome = {
            ("leftRail", "width"): 325,
            ("stage", "x"): 325,
            ("rightRail", "x"): 962,
            ("toggle", "y"): 23,
            ("tabs", "y"): 16,
        }
        chrome_errors = [
            f"{element}.{coordinate}.physical: "
            f"{geometry[element][coordinate] * geometry['dpr']:.3f} != {wanted}"
            for (element, coordinate), wanted in expected_physical_chrome.items()
            if geometry[element][coordinate] * geometry["dpr"]
            != pytest.approx(wanted, abs=2.0)
        ]
        physical_y = {
            name: geometry[name]["y"] * geometry["dpr"]
            for name in (
                "chassis",
                "leftArm",
                "rightArm",
                "dpad",
                "observe",
                "up",
                "rotateLeft",
                "open",
            )
        }
        expected_physical_y = {
            "chassis": 149,
            "leftArm": 151,
            "rightArm": 151,
            "dpad": 252,
            "observe": 468,
            "up": 235,
            "rotateLeft": 365,
            "open": 494,
        }
        errors = [
            f"{name}.physical_y: {physical_y[name]:.3f} != {wanted}"
            for name, wanted in expected_physical_y.items()
            if physical_y[name] != pytest.approx(wanted, abs=2.0)
        ]
        errors.extend(chrome_errors)
        for name in ("chassis", "leftArm", "rightArm"):
            if geometry[name]["width"] != pytest.approx(76, abs=0.5):
                errors.append(
                    f"{name}.width: {geometry[name]['width']:.3f} != 76"
                )
            if geometry[name]["height"] != pytest.approx(26, abs=0.5):
                errors.append(
                    f"{name}.height: {geometry[name]['height']:.3f} != 26"
                )
        for name in ("observe", "up", "rotateLeft", "open"):
            if geometry[name]["width"] != pytest.approx(39, abs=0.5):
                errors.append(
                    f"{name}.width: {geometry[name]['width']:.3f} != 39"
                )
            if geometry[name]["height"] != pytest.approx(39, abs=0.5):
                errors.append(
                    f"{name}.height: {geometry[name]['height']:.3f} != 39"
                )
        assert not errors, "high-DPI coordinate mismatches:\n" + "\n".join(
            errors
        )
    finally:
        with mock.lock:
            mock.use_high_dpi_frame = False
        page.set_device_metrics(
            width=1406,
            height=580,
            device_scale_factor=1,
        )
        page.reload()


def test_high_dpi_calibration_is_scoped_to_the_right_panel_width(dashboard):
    """A narrow split Dashboard must not inherit the full-width visual fixture."""
    page, _mock = dashboard
    try:
        page.set_device_metrics(
            width=853,
            height=578,
            device_scale_factor=1.5,
        )
        page.reload()
        geometry = page.evaluate(
            """
            (() => {
              const rect = selector => {
                const r = document.querySelector(selector).getBoundingClientRect();
                return {x:r.x, y:r.y, width:r.width,height:r.height};
              };
              return {
                rightColumn: rect(".col.right"),
                frame: rect("#framewrap"),
                stage: rect(".frame-stage"),
                columns: getComputedStyle(
                  document.querySelector("#framewrap")
                ).gridTemplateColumns
              };
            })()
            """
        )
        assert geometry["rightColumn"]["width"] < 830
        assert geometry["frame"]["width"] == pytest.approx(
            geometry["rightColumn"]["width"], abs=0.5
        )
        assert geometry["stage"]["width"] >= 230
        assert geometry["columns"] != "216.667px 424.328px 212px"
    finally:
        page.set_device_metrics(
            width=1406,
            height=580,
            device_scale_factor=1,
        )
        page.reload()
