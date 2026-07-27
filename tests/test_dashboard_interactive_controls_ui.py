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
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "rpent/dashboard/index.html"
REFERENCE = REPO / "tests/fixtures/dashboard_interactive_controls_reference.png"
REFERENCE_SHA256 = "9d3071e672c81845c6e88247bcf0d0642a0a9b7c5b28c887823b99d5d3a1607c"
OUTPUT_DIR = Path("/home/ubuntu/lwb/RPent_outputs/dashboard_visual_acceptance")
LIVE_SCREENSHOT = OUTPUT_DIR / "interactive_controls_live_823x526.png"
OVERLAY_SCREENSHOT = OUTPUT_DIR / "interactive_controls_reference_overlay_50pct.png"


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

    @staticmethod
    def _control_base() -> dict[str, Any]:
        return {
            "available": True,
            "motion_available": True,
            "observe_available": True,
            "busy": False,
            "owner": None,
            "command_id": None,
            "lease_id": None,
            "phase": "idle",
            "error": None,
            "capabilities": {
                "base_available": True,
                "torso_available": True,
                "eef_available": {"left": True, "right": True},
                "wrist_rotation_available": {"left": True, "right": True},
                "gripper_available": {"left": True, "right": True},
            },
        }

    def reset(self, *, command_delay_s: float = 0.045) -> None:
        with self.lock:
            self.command_delay_s = command_delay_s
            self.commands: list[dict[str, Any]] = []
            self.heartbeats: list[dict[str, Any]] = []
            self.stops: list[dict[str, Any]] = []
            self.cameras: list[dict[str, Any]] = []
            self.max_in_flight = 0
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
        return {
            "id": "mock-run",
            "suite": "behavior_2025_challenge",
            "name": "picking_up_trash",
            "task": 1,
            "seed": 13,
            "state": "running",
            "terminated": False,
            "frame_idx": 1587,
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
            "control": self._control_base(),
            "usage": {"in": 0, "out": 0, "tool_calls": 10},
        }

    def accept(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        now = time.monotonic()
        key = (str(body["run"]), str(body["lease_id"]), int(body["sequence"]))
        with self.lock:
            prior = self._dedup.get(key)
            if prior is not None:
                return HTTPStatus.ACCEPTED, {
                    "command_id": prior["command_id"],
                    "phase": "accepted",
                    "deduplicated": True,
                }
            command = {
                **body,
                "command_id": f"visual-command-{len(self.commands) + 1}",
                "accepted_at": now,
                "complete_at": now + self.command_delay_s,
                "cancelled": False,
            }
            self.commands.append(command)
            self._dedup[key] = command
            active = sum(
                1
                for item in self.commands
                if not item["cancelled"] and item["complete_at"] > now
            )
            self.max_in_flight = max(self.max_in_flight, active)
            return HTTPStatus.ACCEPTED, {
                "command_id": command["command_id"],
                "phase": "accepted",
            }

    def control_state(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            control = self._control_base()
            if not self.commands:
                return {"control": control}
            command = self.commands[-1]
            if command["cancelled"]:
                phase = "cancelled"
            elif now >= command["complete_at"]:
                phase = "completed"
            else:
                phase = "moving"
            control.update(
                {
                    "busy": phase == "moving",
                    "owner": "manual" if phase == "moving" else None,
                    "command_id": command["command_id"],
                    "lease_id": command["lease_id"],
                    "target": command["target"],
                    "action": command["action"],
                    "phase": phase,
                }
            )
            return {"control": control}

    def stop(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.stops.append({**body, "at": time.monotonic()})
            lease_id = body.get("lease_id")
            for command in self.commands:
                if command["lease_id"] == lease_id:
                    command["cancelled"] = True
            return {"phase": "cancelled", "lease_id": lease_id}


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
            self._json(mock.run_payload())
        elif parsed.path == "/api/run/transcript":
            self._json({"events": []})
        elif parsed.path == "/api/run/control/state":
            self._json(mock.control_state())
        elif parsed.path == "/api/run/frame":
            self._send(HTTPStatus.OK, mock.head_png, "image/png")
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
            self._json({"phase": "moving"})
        elif parsed.path == "/api/run/control/stop":
            self._json(mock.stop(body))
        elif parsed.path == "/api/run/control/camera":
            with mock.lock:
                mock.cameras.append({**body, "at": time.monotonic()})
            self._json({"camera": body.get("camera")})
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
        self.call(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 1406,
                "height": 580,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
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

    def screenshot_clip(self, path: Path) -> None:
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
                    "width": 823,
                    "height": 526,
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
              button => ({ camera: button.dataset.camera, active: button.classList.contains("active") })
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
        {"camera": "head", "active": True},
        {"camera": "left_wrist", "active": False},
        {"camera": "right_wrist", "active": False},
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


def test_short_press_and_long_press_single_flight_timing(dashboard):
    page, mock = dashboard
    selector = '[data-action="forward"]'

    _dispatch_pointer(page, selector, "pointerdown", 201)
    page.sleep(95)
    _dispatch_pointer(page, selector, "pointerup", 201)
    page.sleep(420)
    assert len(mock.commands) == 1

    page.evaluate("stopActiveLease('between_press_cases')")
    page.sleep(120)
    mock.reset(command_delay_s=0.065)
    page.reload()
    _dispatch_pointer(page, selector, "pointerdown", 202)
    page.sleep(1050)
    _dispatch_pointer(page, selector, "pointerup", 202)
    page.sleep(120)
    commands = list(mock.commands)
    assert len(commands) >= 4
    accepted = [item["accepted_at"] for item in commands]
    intervals_ms = [
        (later - earlier) * 1000
        for earlier, later in zip(accepted, accepted[1:], strict=False)
    ]
    assert intervals_ms[0] >= 315
    assert all(interval >= 215 for interval in intervals_ms[1:])
    assert mock.max_in_flight == 1
    assert len(mock.heartbeats) >= 2
    heartbeat_intervals_ms = [
        (later["at"] - earlier["at"]) * 1000
        for earlier, later in zip(mock.heartbeats, mock.heartbeats[1:], strict=False)
    ]
    assert all(350 <= interval <= 500 for interval in heartbeat_intervals_ms)
    assert [item["sequence"] for item in commands] == list(range(1, len(commands) + 1))


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
    _dispatch_pointer(page, '[data-action="forward"]', "pointerdown", 301)
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
    page.mouse('[data-action="forward"]', down=False)
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
    assert not mock.stops


def test_action_video_returns_to_previously_selected_camera(dashboard):
    page, _mock = dashboard
    page.evaluate("document.querySelector('[data-camera=\"left_wrist\"]').click()")
    page.sleep(120)
    assert page.evaluate(
        "selectedCamera === 'left_wrist' && frameKind === 'left_wrist'"
    )
    page.evaluate(
        """
        playActionVideo(
          {step: 49, action: 'pi0_nav_pick', has_action_video: true},
          {returnAfterEnd: true, returnKind: lastRealtimeFrameKind, nextFrameIdx: 1588}
        );
        for (const video of document.querySelectorAll('video')) {
          video.dispatchEvent(new Event('canplay'));
        }
        """
    )
    page.sleep(50)
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
        "chassis": {"x": 63, "y": 47, "width": 76, "height": 26},
        "leftArm": {"x": 646, "y": 69, "width": 76, "height": 26},
        "rightArm": {"x": 731, "y": 69, "width": 76, "height": 26},
        "dpad": {"x": 51, "y": 124, "width": 96, "height": 96},
        "forward": {"x": 83, "y": 124, "width": 32, "height": 32},
        "backward": {"x": 83, "y": 188, "width": 32, "height": 32},
        "turnLeft": {"x": 51, "y": 156, "width": 32, "height": 32},
        "turnRight": {"x": 115, "y": 156, "width": 32, "height": 32},
        "observe": {"x": 79, "y": 265, "width": 39, "height": 39},
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

    assert LIVE_SCREENSHOT.exists()
    assert OVERLAY_SCREENSHOT.exists()

    errors = []
    for element, wanted in expected.items():
        for coordinate, value in wanted.items():
            actual = geometry[element][coordinate]
            if actual != pytest.approx(value, abs=1.0):
                errors.append(f"{element}.{coordinate}: {actual:.3f} != {value}")

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
