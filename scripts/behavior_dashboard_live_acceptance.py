#!/usr/bin/env python3
# Copyright (c) 2026 RPent contributors
"""Drive every BEHAVIOR Dashboard control against one live simulator run.

This probe is intentionally inert unless ``--execute-sim-controls`` is passed.
It uses the real Dashboard DOM and HTTP API, waits for command-terminal
receipts, and saves the browser, Timeline, camera-lineage, and visual evidence
needed to audit the run afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageChops, ImageStat

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("/home/ubuntu/lwb/Projects/test/image.png")
REFERENCE_SHA256 = "ef0d70afaf83b07a79e0cae37d23c90534a787c4ef8dcf8b0a0784e307369223"
ALL_ACTIONS = (
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
    "observe",
)
TARGETS = ("chassis", "left_arm", "right_arm")
CAMERAS = ("head", "left_wrist", "right_wrist")
CONTROL_MUTATION_ENDPOINTS = (
    "/api/run/control/command",
    "/api/run/control/heartbeat",
    "/api/run/control/stop",
)
BASE_TRANSLATION_STEP_M = 0.05
BASE_ROTATION_STEP_RAD = math.radians(5.0)
EEF_TRANSLATION_STEP_M = 0.03
TORSO_VERTICAL_STEP_M = 0.03
WRIST_ROTATION_STEP_RAD = math.radians(5.0)
DASHBOARD_BASE_EXECUTION_XY_STEP_M = 0.0075
DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD = math.radians(1.0)
BASE_TERMINAL_POSITION_TOLERANCE_M = 0.020
BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
HIGH_DPI_VISUAL_MAE_MAX = 10.0
HIGH_DPI_VISUAL_WITHIN_24_MIN = 0.88
# This script is the final release acceptance, not the pre-calibration probe.
# Planning-only wrist evidence is recorded by behavior_base_curobo_probe.py
# while the public wrist capability may still be disabled.  Only a separately
# admitted, per-hand visual calibration may make the release Dashboard satisfy
# this final 29-enabled / 4-disabled oracle.
RELEASE_EXPECTED_ENABLED = frozenset(
    {
        *(
            ("chassis", action)
            for action in (
                "observe",
                "forward",
                "backward",
                "turn_left",
                "turn_right",
                "up",
                "down",
            )
        ),
        *(
            (target, action)
            for target in ("left_arm", "right_arm")
            for action in (
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
                "observe",
            )
        ),
    }
)
RELEASE_EXPECTED_DISABLED = frozenset(
    (target, action)
    for target in TARGETS
    for action in ALL_ACTIONS
    if (target, action) not in RELEASE_EXPECTED_ENABLED
)
if len(RELEASE_EXPECTED_ENABLED) != 29 or len(RELEASE_EXPECTED_DISABLED) != 4:
    raise AssertionError(
        "release control oracle must be exactly 29 enabled / 4 disabled"
    )
MOTION_ORDER = {
    "chassis": (
        "observe",
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "up",
        "down",
        "rotate_left",
        "rotate_right",
        "close",
        "open",
    ),
    "left_arm": (
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "up",
        "down",
        "rotate_left",
        "rotate_right",
        "close",
        "open",
        "observe",
    ),
    "right_arm": (
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "up",
        "down",
        "rotate_left",
        "rotate_right",
        "close",
        "open",
        "observe",
    ),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--command-timeout-s", type=float, default=270.0)
    parser.add_argument("--sustain-seconds", type=float, default=600.0)
    parser.add_argument("--sustain-interval-s", type=float, default=5.0)
    parser.add_argument("--execute-sim-controls", action="store_true")
    parser.add_argument(
        "--planning-only-probes",
        action="store_true",
        help="plan and discard torso/wrist probes, then exit without browser actions",
    )
    return parser.parse_args()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")


def _progress(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "monotonic_s": time.monotonic(), **payload},
            sort_keys=True,
            default=str,
        ),
        flush=True,
    )


def _http_json(base_url: str, path: str) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + path, headers={"Cache-Control": "no-store"}
    )
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object")
    return value


def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Cache-Control": "no-store",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object")
    return value


def _http_bytes(base_url: str, path: str) -> bytes:
    request = Request(
        base_url.rstrip("/") + path, headers={"Cache-Control": "no-store"}
    )
    with urlopen(request, timeout=30) as response:
        return response.read()


def _load_chrome_page() -> type:
    test_path = REPO_ROOT / "tests" / "test_dashboard_interactive_controls_ui.py"
    spec = importlib.util.spec_from_file_location("_rpent_dashboard_ui", test_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Chrome fixture from {test_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._ChromePage


def _wait(predicate: Any, *, timeout_s: float, interval_s: float = 0.1) -> Any:
    deadline = time.monotonic() + timeout_s
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval_s)
    raise TimeoutError(f"condition not met after {timeout_s:.1f}s; last={last!r}")


def _run_payload(base_url: str, run_id: str) -> dict[str, Any]:
    return _http_json(base_url, f"/api/run?run={quote(run_id)}")


def _wait_for_run_ready(
    *,
    base_url: str,
    run_id: str,
    checks: Callable[[dict[str, Any]], dict[str, bool]],
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Wait until one live run satisfies every supplied readiness check."""

    def ready_payload() -> dict[str, Any] | None:
        payload = _run_payload(base_url, run_id)
        return payload if all(checks(payload).values()) else None

    return _wait(ready_payload, timeout_s=timeout_s)


def _manual_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    timeline = payload.get("timeline")
    if not isinstance(timeline, list):
        return []
    return [
        dict(item)
        for item in timeline
        if isinstance(item, dict) and item.get("source") == "dashboard_manual"
    ]


def _raw_success_latched(
    payload: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> bool:
    """Recognize only the Dashboard's raw-success latch or a returned true result."""

    control = payload.get("control")
    control = control if isinstance(control, dict) else {}
    result = result if isinstance(result, dict) else {}
    return bool(
        control.get("success_latched") is True
        or result.get("task_success") is True
    )


def _run_planning_only_probes(
    *,
    base_url: str,
    run_id: str,
    output_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    """Exercise the shared Dashboard/EnvClient planning path with zero action."""

    specs = (
        ("chassis", "up"),
        ("chassis", "down"),
        ("left_arm", "rotate_left"),
        ("left_arm", "rotate_right"),
        ("right_arm", "rotate_left"),
        ("right_arm", "rotate_right"),
    )
    baseline = _run_payload(base_url, run_id)
    baseline_timeline_revision = baseline.get("timeline_revision")
    baseline_step = baseline.get("simulator_step")
    baseline_indices = dict(baseline.get("frame_indices") or {})
    baseline_capture_group_id = baseline.get("capture_group_id")
    records: list[dict[str, Any]] = []
    for target, action in specs:
        before = _run_payload(base_url, run_id)
        if _raw_success_latched(before):
            break
        started = time.monotonic()
        result = _post_json(
            base_url,
            "/api/run/control/plan",
            {"run": run_id, "target": target, "action": action},
            timeout_s=timeout_s,
        )
        after = _run_payload(base_url, run_id)
        safety_certificate = result.get("safety_certificate")
        safety_certificate = (
            safety_certificate
            if isinstance(safety_certificate, dict)
            else {}
        )
        safety_checks = safety_certificate.get("checks")
        safety_checks = (
            safety_checks if isinstance(safety_checks, dict) else {}
        )
        prepared = result.get("prepared")
        prepared = prepared if isinstance(prepared, dict) else {}
        requested_step = prepared.get("requested_step")
        requested_step = (
            requested_step if isinstance(requested_step, dict) else {}
        )
        discard_receipt = result.get("discard_receipt")
        discard_receipt = (
            discard_receipt if isinstance(discard_receipt, dict) else {}
        )
        expected_motion_kind = "torso" if target == "chassis" else "eef"
        expected_hand = target.removesuffix("_arm")
        expected_direction = (
            "counterclockwise" if action == "rotate_left" else "clockwise"
        )
        expected_signed_rotation = (
            WRIST_ROTATION_STEP_RAD
            if action == "rotate_left"
            else -WRIST_ROTATION_STEP_RAD
        )
        calibration = prepared.get("calibration")
        calibration = calibration if isinstance(calibration, dict) else {}
        requested_step_matches = (
            requested_step.get("frame") == "world"
            and _finite_close(
                requested_step.get("torso_delta_z_m"),
                (
                    TORSO_VERTICAL_STEP_M
                    if action == "up"
                    else -TORSO_VERTICAL_STEP_M
                ),
                tolerance=1e-12,
            )
            if target == "chassis"
            else (
                requested_step.get("frame") == "wrist_camera"
                and requested_step.get("visual_direction") == expected_direction
                and _finite_close(
                    requested_step.get("angle_rad"),
                    WRIST_ROTATION_STEP_RAD,
                    tolerance=1e-12,
                )
                and _finite_close(
                    prepared.get("requested_rotation_rad"),
                    expected_signed_rotation,
                    tolerance=1e-12,
                )
            )
        )
        record = {
            "target": target,
            "action": action,
            "elapsed_s": round(time.monotonic() - started, 3),
            "result": result,
            "checks": {
                "endpoint_ok": result.get("ok") is True,
                "response_identity": (
                    result.get("target") == target
                    and result.get("action") == action
                ),
                "prepared_identity": (
                    prepared.get("target") == target
                    and prepared.get("action") == action
                    and prepared.get("motion_kind") == expected_motion_kind
                    and prepared.get("plan_id") == result.get("plan_id")
                ),
                "requested_step_exact": requested_step_matches,
                "planning_only": result.get("planning_only") is True,
                "release_admission_false": (
                    result.get("release_admission") is False
                ),
                "zero_action_verified": (
                    result.get("zero_action_verified") is True
                ),
                "env_step_delta_zero": result.get("env_step_delta") == 0,
                "safety_certificate_admitted": (
                    safety_certificate.get("admitted") is True
                ),
                "safety_motion_kind": (
                    safety_certificate.get("motion_kind")
                    == expected_motion_kind
                ),
                "selected_hand_exact": (
                    target == "chassis"
                    or (
                        safety_certificate.get("selected_hand") == expected_hand
                        and calibration.get("hand") == expected_hand
                    )
                ),
                "wrist_capture_lineage_fresh": (
                    target == "chassis"
                    or (
                        calibration.get("capture_group_id")
                        == baseline_capture_group_id
                        and calibration.get("step_index") == baseline_step
                    )
                ),
                "discard_receipt_terminal": (
                    result.get("discarded") is True
                    and discard_receipt.get("discarded") is True
                    and discard_receipt.get("status") == "discarded"
                    and discard_receipt.get("plan_id") == result.get("plan_id")
                ),
                "dual_attachment_collision": (
                    safety_certificate.get("attachment_hand_count") == 2
                    and safety_checks.get("dual_attachment_collision")
                    is True
                ),
                "collision_checks_complete": (
                    safety_checks.get("world_collision_check") is True
                    and safety_checks.get("self_collision_check") is True
                    and safety_checks.get("post_interpolation_check") is True
                    and safety_checks.get("collision_admitted") is True
                ),
                "all_safety_checks_passed": (
                    bool(safety_checks)
                    and all(value is True for value in safety_checks.values())
                ),
                "simulator_step_unchanged": (
                    after.get("simulator_step") == baseline_step
                ),
                "frame_indices_unchanged": (
                    dict(after.get("frame_indices") or {}) == baseline_indices
                ),
                "timeline_revision_unchanged": (
                    after.get("timeline_revision")
                    == baseline_timeline_revision
                ),
                "raw_success_not_latched": not _raw_success_latched(after),
            },
        }
        record["verdict"] = (
            "pass" if all(record["checks"].values()) else "fail"
        )
        records.append(record)
        _progress(
            "planning_only_probe_complete",
            target=target,
            action=action,
            verdict=record["verdict"],
        )
        if record["verdict"] != "pass":
            break
    final = _run_payload(base_url, run_id)
    summary = {
        "mode": "planning_only",
        "run_id": run_id,
        "expected_probe_count": len(specs),
        "completed_probe_count": len(records),
        "probes": records,
        "baseline": {
            "timeline_revision": baseline_timeline_revision,
            "simulator_step": baseline_step,
            "frame_indices": baseline_indices,
            "capture_group_id": baseline_capture_group_id,
        },
        "final": {
            "timeline_revision": final.get("timeline_revision"),
            "simulator_step": final.get("simulator_step"),
            "frame_indices": final.get("frame_indices"),
            "control_capabilities": dict(final.get("control") or {}).get(
                "capabilities"
            ),
        },
        "release_admission": False,
        "real_robot_deployment_allowed": False,
    }
    summary["verdict"] = (
        "pass"
        if len(records) == len(specs)
        and all(record["verdict"] == "pass" for record in records)
        and not _raw_success_latched(final)
        else "fail"
    )
    _json_dump(output_dir / "planning_only_probes.json", summary)
    return summary


def _refresh_planning_only_capture(
    *,
    base_url: str,
    run_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Publish one fresh atomic camera group without advancing simulation."""

    before = _run_payload(base_url, run_id)
    if _raw_success_latched(before):
        return {
            "verdict": "fail",
            "reason": "official_task_success_latched",
            "request_sent": False,
        }
    lease_id = str(uuid.uuid4())
    accepted = _post_json(
        base_url,
        "/api/run/control/command",
        {
            "run": run_id,
            "lease_id": lease_id,
            "sequence": 1,
            "target": "chassis",
            "action": "observe",
            "camera": "head",
        },
        timeout_s=timeout_s,
    )
    command_id = str(accepted.get("accepted_command_id") or "").strip()
    if not command_id:
        return {
            "verdict": "fail",
            "reason": "observe_command_not_accepted",
            "request_sent": True,
            "accepted": accepted,
        }
    after, row = _wait(
        lambda: _terminal_command_row(
            base_url,
            run_id,
            command_id,
        ),
        timeout_s=timeout_s,
    )
    result = row.get("result")
    result = result if isinstance(result, dict) else {}
    checks = {
        "accepted": accepted.get("accepted") is True,
        "terminal_completed": row.get("status") == "completed",
        "observe_action": (
            row.get("target") == "chassis"
            and row.get("action") == "observe"
        ),
        "primitive_success": result.get("primitive_success") is True,
        "simulator_step_unchanged": (
            after.get("simulator_step") == before.get("simulator_step")
        ),
        "fresh_capture_group": (
            isinstance(after.get("capture_group_id"), str)
            and bool(after["capture_group_id"])
            and after.get("capture_group_id") != before.get("capture_group_id")
        ),
        **_complete_frame_group_checks(after),
        "raw_success_not_latched": not _raw_success_latched(after, result),
    }
    return {
        "verdict": "pass" if all(checks.values()) else "fail",
        "request_sent": True,
        "lease_id": lease_id,
        "command_id": command_id,
        "checks": checks,
        "before": {
            "simulator_step": before.get("simulator_step"),
            "capture_group_id": before.get("capture_group_id"),
            "frame_indices": before.get("frame_indices"),
        },
        "after": {
            "simulator_step": after.get("simulator_step"),
            "capture_group_id": after.get("capture_group_id"),
            "frame_indices": after.get("frame_indices"),
        },
        "result": result,
    }


def _inject_fetch_audit(page: Any) -> None:
    page.evaluate(
        """
        (() => {
          window.__rpentControlFetches = [];
          window.__rpentConsoleErrors = [];
          const recordConsoleError = (kind, values) => {
            window.__rpentConsoleErrors.push({
              kind,
              values: values.map(value => {
                if (value instanceof Error) return value.stack || String(value);
                if (typeof value === "string") return value;
                try { return JSON.stringify(value); } catch { return String(value); }
              }),
              at_ms: performance.now()
            });
          };
          const originalConsoleError = console.error.bind(console);
          console.error = (...values) => {
            recordConsoleError("console.error", values);
            originalConsoleError(...values);
          };
          window.addEventListener("error", event => {
            recordConsoleError("window.error", [
              event.message || "window error",
              event.error || null
            ]);
          });
          window.addEventListener("unhandledrejection", event => {
            recordConsoleError("unhandledrejection", [event.reason]);
          });
          const originalFetch = window.fetch.bind(window);
          window.fetch = async (input, init = {}) => {
            const url = typeof input === "string" ? input : input.url;
            const started = performance.now();
            let body = null;
            try { body = init.body ? JSON.parse(init.body) : null; } catch { body = init.body || null; }
            const audited = String(url).includes("/api/run/control/");
            const audit = audited ? {
              url: String(url), method: init.method || "GET", body,
              started_ms: started, completed_ms: null
            } : null;
            if (audit) window.__rpentControlFetches.push(audit);
            try {
              const response = await originalFetch(input, init);
              if (audit) {
                let responseBody = null;
                try { responseBody = await response.clone().json(); } catch {}
                Object.assign(audit, {
                  status: response.status, response: responseBody,
                  completed_ms: performance.now()
                });
              }
              return response;
            } catch (error) {
              if (audit) {
                Object.assign(audit, {
                  error: String(error),
                  completed_ms: performance.now()
                });
              }
              throw error;
            }
          };
          return true;
        })()
        """
    )


def _fetch_audit(page: Any) -> list[dict[str, Any]]:
    value = page.evaluate("window.__rpentControlFetches || []")
    return value if isinstance(value, list) else []


def _select_target(page: Any, target: str) -> None:
    page.evaluate(f"document.querySelector('[data-target=\"{target}\"]').click()")
    _wait(
        lambda: page.evaluate(f"selectedTarget === {json.dumps(target)}"),
        timeout_s=3,
    )


def _select_camera(page: Any, camera: str) -> None:
    page.evaluate(f"document.querySelector('[data-camera=\"{camera}\"]').click()")
    _wait(
        lambda: page.evaluate(
            f"""
            (() => {{
              const button = document.querySelector('[data-camera="{camera}"]');
              return selectedCamera === {json.dumps(camera)}
                && frameKind === {json.dumps(camera)}
                && button.classList.contains("active");
            }})()
            """
        ),
        timeout_s=3,
    )


def _button_state(page: Any, action: str) -> dict[str, Any]:
    return page.evaluate(
        f"""
        (() => {{
          const button = document.querySelector('[data-action="{action}"]');
          const rect = button.getBoundingClientRect();
          const style = getComputedStyle(button);
          return {{
            aria_disabled: button.getAttribute("aria-disabled"),
            tooltip: button.dataset.tooltip,
            pressed: button.classList.contains("pressed"),
            rect: {{x:rect.x,y:rect.y,width:rect.width,height:rect.height}},
            background: style.background,
            border: style.border,
            color: style.color
          }};
        }})()
        """
    )


def _dispatch_pointer_event(
    page: Any,
    *,
    action: str,
    event_type: str,
    pointer_id: int,
) -> dict[str, Any]:
    """Drive the product's PointerEvent handlers without CDP input deadlock.

    Native CDP mouse input is covered by the browser unit suite.  On a live
    BEHAVIOR page, ``Input.dispatchMouseEvent`` waits for the long-running
    async pointer handler and can deadlock the acceptance driver before the
    command request is observable.  A DOM PointerEvent exercises the same
    product handler while keeping the CDP control channel responsive.
    """

    return page.evaluate(
        f"""
        (() => {{
          const button = document.querySelector('[data-action="{action}"]');
          if (!button) throw new Error("missing action button: {action}");
          button.dispatchEvent(new PointerEvent("{event_type}", {{
            bubbles: true,
            cancelable: true,
            pointerId: {pointer_id},
            pointerType: "mouse",
            button: 0,
            buttons: {"1" if event_type == "pointerdown" else "0"},
            isPrimary: true
          }}));
          return {{
            pressed: button.classList.contains("pressed"),
            lease_id: activeLeaseId,
            pressed_action: pressedAction
          }};
        }})()
        """
    )


def _dispatch_keyboard_activation(page: Any, *, action: str, key: str) -> None:
    code = "Space" if key == " " else key
    page.evaluate(
        f"""
        (() => {{
          const button = document.querySelector('[data-action="{action}"]');
          button.focus();
          for (const type of ["keydown", "keyup"]) {{
            button.dispatchEvent(new KeyboardEvent(type, {{
              bubbles: true,
              cancelable: true,
              key: {json.dumps(key)},
              code: {json.dumps(code)}
            }}));
          }}
        }})()
        """
    )


def _terminal_new_row(
    base_url: str,
    run_id: str,
    prior_ids: set[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload = _run_payload(base_url, run_id)
    candidates = [
        row
        for row in _manual_rows(payload)
        if str(row.get("command_id") or "") not in prior_ids
    ]
    if not candidates:
        return None
    row = candidates[-1]
    if row.get("status") not in {"completed", "failed", "cancelled"}:
        return None
    return payload, row


def _terminal_command_row(
    base_url: str,
    run_id: str,
    command_id: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    payload = _run_payload(base_url, run_id)
    row = next(
        (
            item
            for item in reversed(_manual_rows(payload))
            if str(item.get("command_id") or "") == command_id
        ),
        None,
    )
    if row is None or row.get("status") not in {
        "completed",
        "failed",
        "cancelled",
    }:
        return None
    return payload, row


def _save_frames(
    *,
    base_url: str,
    run_id: str,
    output_dir: Path,
    case_index: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "capture_group_id": payload.get("capture_group_id"),
        "simulator_step": payload.get("simulator_step"),
        "frame_indices": payload.get("frame_indices"),
        "frame_revisions": payload.get("frame_revisions"),
        "cameras": {},
    }
    for camera in CAMERAS:
        data = _http_bytes(
            base_url,
            f"/api/run/frame?run={quote(run_id)}&kind={camera}&t={time.time_ns()}",
        )
        path = output_dir / "frames" / f"{case_index:02d}_{camera}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        result["cameras"][camera] = {
            "path": str(path),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return result


def _finite_le(value: Any, limit: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric <= limit


def _finite_abs_le(value: Any, limit: float) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and abs(numeric) <= limit


def _finite_close(value: Any, expected: float, *, tolerance: float = 1e-7) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and abs(numeric - expected) <= tolerance


def _finite_vector(
    value: Any,
    expected: tuple[float, ...] | None = None,
    *,
    tolerance: float = 1e-7,
) -> bool:
    if not isinstance(value, list) or (
        expected is not None and len(value) != len(expected)
    ):
        return False
    try:
        numeric = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in numeric):
        return False
    return expected is None or all(
        abs(actual - wanted) <= tolerance
        for actual, wanted in zip(numeric, expected, strict=True)
    )


def _control_ready_checks(payload: dict[str, Any]) -> dict[str, bool]:
    control = payload.get("control")
    control = control if isinstance(control, dict) else {}
    capabilities = control.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    revisions = payload.get("frame_revisions")
    revisions = revisions if isinstance(revisions, dict) else {}
    indices = payload.get("frame_indices")
    indices = indices if isinstance(indices, dict) else {}
    eef_available = capabilities.get("eef_available")
    eef_available = eef_available if isinstance(eef_available, dict) else {}
    wrist_available = capabilities.get("wrist_rotation_available")
    wrist_available = wrist_available if isinstance(wrist_available, dict) else {}
    gripper_available = capabilities.get("gripper_available")
    gripper_available = gripper_available if isinstance(gripper_available, dict) else {}
    simulator_step = payload.get("simulator_step")
    head_revision = revisions.get("head")
    return {
        "run_state_running": payload.get("state") == "running",
        "run_not_terminated": payload.get("terminated") is False,
        "control_available": control.get("available") is True,
        "motion_available": control.get("motion_available") is True,
        "observe_available": control.get("observe_available") is True,
        "control_not_busy": control.get("busy") is False,
        "planner_available": capabilities.get("planner_available") is True,
        "position_control_ready": capabilities.get("position_control_ready") is True,
        "simulation_identity_verified": capabilities.get("simulation_identity")
        == "behavior_omnigibson_r1pro",
        "release_base_enabled": capabilities.get("base_available") is True,
        "release_torso_enabled": capabilities.get("torso_available") is True,
        "release_eef_enabled": all(
            eef_available.get(hand) is True for hand in ("left", "right")
        ),
        "release_wrist_enabled": all(
            wrist_available.get(hand) is True for hand in ("left", "right")
        ),
        "release_gripper_enabled": all(
            gripper_available.get(hand) is True for hand in ("left", "right")
        ),
        "release_oracle_cardinality_29_4": (
            len(RELEASE_EXPECTED_ENABLED) == 29 and len(RELEASE_EXPECTED_DISABLED) == 4
        ),
        "head_revision_visible": isinstance(head_revision, int) and head_revision > 0,
        "head_index_matches_simulator_step": isinstance(simulator_step, int)
        and indices.get("head") == simulator_step,
    }


def _planning_only_ready_checks(payload: dict[str, Any]) -> dict[str, bool]:
    """Require safe calibration inputs while wrist release remains disabled."""

    checks = _control_ready_checks(payload)
    checks.pop("release_wrist_enabled", None)
    control = payload.get("control")
    control = control if isinstance(control, dict) else {}
    capabilities = control.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    wrist_available = capabilities.get("wrist_rotation_available")
    wrist_available = wrist_available if isinstance(wrist_available, dict) else {}
    planner = capabilities.get("planner")
    planner = planner if isinstance(planner, dict) else {}
    geometry = planner.get("wrist_geometry")
    geometry = geometry if isinstance(geometry, dict) else {}
    checks.update(
        {
            "planning_wrist_release_fail_closed": all(
                wrist_available.get(hand) is False
                for hand in ("left", "right")
            ),
            "planning_wrist_geometry_verified": all(
                isinstance(geometry.get(hand), dict)
                and geometry[hand].get("verified") is True
                and geometry[hand].get("release_admission") is False
                and geometry[hand].get("real_visual_probe_required") is True
                for hand in ("left", "right")
            ),
            **_complete_frame_group_checks(payload),
        }
    )
    return checks


def _planning_only_pre_capture_ready_checks(
    payload: dict[str, Any],
) -> dict[str, bool]:
    """Admit one zero-step Observe before a three-camera group is published."""

    checks = _planning_only_ready_checks(payload)
    for name in _complete_frame_group_checks(payload):
        checks.pop(name, None)
    return checks


def _complete_frame_group_checks(payload: dict[str, Any]) -> dict[str, bool]:
    revisions = payload.get("frame_revisions")
    revisions = revisions if isinstance(revisions, dict) else {}
    indices = payload.get("frame_indices")
    indices = indices if isinstance(indices, dict) else {}
    simulator_step = payload.get("simulator_step")
    return {
        "capture_group_nonempty": payload.get("capture_group_id")
        not in {
            None,
            "",
        },
        "three_revisions_positive": all(
            isinstance(revisions.get(camera), int) and revisions[camera] > 0
            for camera in CAMERAS
        ),
        "three_indices_match_simulator_step": isinstance(simulator_step, int)
        and all(indices.get(camera) == simulator_step for camera in CAMERAS),
    }


def _manual_control_quiescent(payload: dict[str, Any]) -> bool:
    control = payload.get("control")
    control = control if isinstance(control, dict) else {}
    action_plane_idle = (
        control.get("lease_status", "idle") != "active"
        and control.get("current_command") is None
        and control.get("planning_command") is None
        and int(control.get("queue_depth") or 0) == 0
    )
    capture = _capture_snapshot(payload)
    capture_only_busy = (
        action_plane_idle
        and control.get("owner") == "manual"
        and capture.get("phase") in {"pending", "started"}
    )
    return action_plane_idle and (
        capture_only_busy
        or (
            control.get("busy") is not True
            and control.get("owner") is None
        )
    )


def _capture_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    control = payload.get("control")
    control = control if isinstance(control, dict) else {}
    capture = control.get("capture")
    return capture if isinstance(capture, dict) else {}


def _semantic_checks(
    *,
    target: str,
    action: str,
    row: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, bool]:
    """Validate the physical primitive and its safety evidence, not only HTTP success."""

    metrics = result.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    checks: dict[str, bool] = {
        "timeline_source_manual": row.get("source") == "dashboard_manual",
        "timeline_target_matches": row.get("target") == target,
        # Timeline ``action`` intentionally names the reused primitive
        # (for example navigate_to), while the manual UI action is retained
        # in the result so both the implementation and requested jog are
        # auditable.
        "manual_action_matches": row.get("manual_action") == action,
    }
    if action == "observe":
        checks.update(
            {
                "primitive_is_observe": row.get("primitive") == "observe",
                "observe_camera_recorded": metrics.get("camera") in CAMERAS,
            }
        )
        return checks

    if target == "chassis" and action in {"up", "down"}:
        expected_delta = TORSO_VERTICAL_STEP_M * (1.0 if action == "up" else -1.0)
        checks.update(
            {
                "primitive_is_jog_torso": metrics.get("manual_primitive")
                == "jog_torso",
                "torso_action_matches": metrics.get("manual_action") == action,
                "torso_link_matches": metrics.get("target_link") == "torso_link4",
                "torso_signed_fixed_step_matches": _finite_close(
                    metrics.get("requested_delta_z_m"),
                    expected_delta,
                ),
                "torso_fixed_server_step": metrics.get("fixed_server_step") is True,
            }
        )
        return checks

    if target == "chassis" and action in {
        "forward",
        "backward",
        "turn_left",
        "turn_right",
    }:
        isolation = metrics.get("navigation_isolation")
        isolation = isolation if isinstance(isolation, dict) else {}
        resampling = metrics.get("execution_resampling")
        resampling = resampling if isinstance(resampling, dict) else {}
        relative_motion = metrics.get("relative_motion")
        relative_motion = relative_motion if isinstance(relative_motion, dict) else {}
        row_args = row.get("args")
        row_args = row_args if isinstance(row_args, dict) else {}
        requested_step = row_args.get("requested_step")
        requested_step = requested_step if isinstance(requested_step, dict) else {}
        goal_construction = metrics.get("base_goal_construction")
        goal_construction = (
            goal_construction if isinstance(goal_construction, dict) else {}
        )
        terminal = metrics.get("navigation_terminal")
        terminal = terminal if isinstance(terminal, dict) else {}
        is_translation = action in {"forward", "backward"}
        expected_direction = action if is_translation else action.removeprefix("turn_")
        requested_signed_step = (
            BASE_TRANSLATION_STEP_M
            * (1.0 if action == "forward" else -1.0)
            if is_translation
            else math.degrees(BASE_ROTATION_STEP_RAD)
            * (1.0 if action == "turn_left" else -1.0)
        )
        checks.update(
            {
                "primitive_is_jog_base": (
                    row.get("primitive") == "navigate_to"
                    and row.get("action") == "navigate_to"
                    and row_args.get("detail") == "relative jog"
                ),
                "base_action_matches": row_args.get("action") == action,
                "base_fixed_server_step": (
                    requested_step.get("kind")
                    == ("translation" if is_translation else "rotation")
                ),
                "base_relative_kind_matches": relative_motion.get("kind")
                == ("translation" if is_translation else "rotation"),
                "base_direction_sign_matches": relative_motion.get("direction")
                == expected_direction,
                "base_translation_step_matches": (
                    not is_translation
                    or _finite_close(
                        relative_motion.get("distance_m"),
                        BASE_TRANSLATION_STEP_M,
                    )
                ),
                "base_rotation_step_matches": (
                    is_translation
                    or _finite_close(
                        relative_motion.get("angle_deg"),
                        math.degrees(BASE_ROTATION_STEP_RAD),
                    )
                ),
                "base_requested_step_frame_matches": (
                    goal_construction.get("method")
                    == "analytic_full_q_base_assignment"
                    and goal_construction.get("nonbase_locked_to_call_start")
                    is True
                ),
                "base_requested_step_matches": (
                    _finite_close(
                        requested_step.get("meters"),
                        requested_signed_step,
                    )
                    if is_translation
                    else _finite_close(
                        requested_step.get("degrees"),
                        requested_signed_step,
                    )
                ),
                "base_goal_finite": isinstance(metrics.get("base_goal"), list)
                and len(metrics["base_goal"]) == 3
                and _finite_vector(metrics["base_goal"]),
                "base_terminal_position_error_bounded": _finite_le(
                    metrics.get("final_position_error_m"),
                    BASE_TERMINAL_POSITION_TOLERANCE_M + 1e-7,
                ),
                "base_terminal_yaw_error_bounded": _finite_le(
                    metrics.get("final_yaw_error_rad"),
                    BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD + 1e-7,
                ),
                "base_terminal_contract_matches": (
                    _finite_close(
                        terminal.get("position_tolerance_m"),
                        BASE_TERMINAL_POSITION_TOLERANCE_M,
                    )
                    and _finite_close(
                        terminal.get("orientation_tolerance_rad"),
                        BASE_TERMINAL_ORIENTATION_TOLERANCE_RAD,
                    )
                ),
                "base_navigation_isolated": isolation.get("ok") is True,
                "minimum_jerk_execution": resampling.get("method")
                == "quintic_minimum_jerk",
                "base_xy_step_bounded": _finite_le(
                    resampling.get("measured_max_xy_step_m"),
                    DASHBOARD_BASE_EXECUTION_XY_STEP_M + 1e-7,
                ),
                "base_yaw_step_bounded": _finite_le(
                    resampling.get("measured_max_yaw_step_rad"),
                    DASHBOARD_BASE_EXECUTION_YAW_STEP_RAD + 1e-7,
                ),
            }
        )
        return checks

    hand = "left" if target == "left_arm" else "right"
    if action in {"open", "close"}:
        expected_opening = 1.0 if action == "open" else 0.0
        expected_latch = 1.0 if action == "open" else -1.0
        gripper_checks = {
            "primitive_is_set_gripper": metrics.get("manual_primitive")
            == "set_gripper",
            "gripper_hand_matches": metrics.get("hand") == hand,
            "gripper_opening_matches": metrics.get("opening") == expected_opening,
            "gripper_latch_matches": metrics.get("gripper_latch") == expected_latch,
            "single_gripper_network_call": metrics.get("network_primitive_calls")
            == 1,
            "gripper_no_retreat": metrics.get("retreat_executed") is False,
        }
        if action == "open":
            capability = result.get("capability")
            capability = capability if isinstance(capability, dict) else {}
            gripper_state = capability.get("gripper_state")
            gripper_state = (
                gripper_state if isinstance(gripper_state, dict) else {}
            )
            gripper_checks["open_capability_latch_restored"] = (
                gripper_state.get(hand) == "open"
            )
        checks.update(gripper_checks)
        return checks

    if action in {"rotate_left", "rotate_right"}:
        calibration = metrics.get("calibration")
        calibration = calibration if isinstance(calibration, dict) else {}
        counterclockwise = action == "rotate_left"
        try:
            ccw_sign = float(calibration.get("visual_ccw_angle_sign"))
        except (TypeError, ValueError):
            ccw_sign = math.nan
        expected_angle = (
            ccw_sign
            * WRIST_ROTATION_STEP_RAD
            * (1.0 if counterclockwise else -1.0)
        )
        checks.update(
            {
                "primitive_is_jog_wrist": metrics.get("manual_primitive")
                == "jog_wrist",
                "wrist_action_matches": metrics.get("manual_action") == action,
                "wrist_hand_matches": metrics.get("hand") == hand,
                "wrist_calibration_verified": calibration.get("verified") is True,
                "wrist_calibration_hand_matches": calibration.get("hand") == hand,
                "wrist_visual_direction_matches": metrics.get("visual_direction")
                == ("counterclockwise" if counterclockwise else "clockwise"),
                "wrist_signed_fixed_step_matches": math.isfinite(expected_angle)
                and _finite_close(
                    metrics.get("requested_rotation_rad"),
                    expected_angle,
                ),
                "wrist_fixed_server_step": metrics.get("fixed_server_step") is True,
                "wrist_position_drift_bounded": _finite_le(
                    metrics.get("final_position_drift_m"),
                    float(metrics.get("position_drift_limit_m", math.nan)) + 1e-9,
                ),
            }
        )
        return checks

    expected_delta_by_action = {
        "forward": (EEF_TRANSLATION_STEP_M, 0.0, 0.0),
        "backward": (-EEF_TRANSLATION_STEP_M, 0.0, 0.0),
        "turn_left": (0.0, EEF_TRANSLATION_STEP_M, 0.0),
        "turn_right": (0.0, -EEF_TRANSLATION_STEP_M, 0.0),
        "up": (0.0, 0.0, EEF_TRANSLATION_STEP_M),
        "down": (0.0, 0.0, -EEF_TRANSLATION_STEP_M),
    }
    expected_delta = expected_delta_by_action[action]
    requested_world = metrics.get("requested_delta_world")
    requested_world_ok = _finite_vector(requested_world) and _finite_close(
        math.sqrt(sum(float(value) ** 2 for value in requested_world)),
        EEF_TRANSLATION_STEP_M,
    )
    actual_target = metrics.get("actual_target")
    actual_target_ok = (
        isinstance(actual_target, list)
        and len(actual_target) == 3
        and _finite_vector(actual_target)
    )
    certificate = metrics.get("whole_body_certificate")
    certificate = certificate if isinstance(certificate, dict) else {}
    eef_path = metrics.get("selected_eef_path")
    eef_path = eef_path if isinstance(eef_path, dict) else {}
    collision = metrics.get("collision_admission")
    collision = collision if isinstance(collision, dict) else {}
    live_guard = metrics.get("whole_body_eef_path_guard")
    live_guard = live_guard if isinstance(live_guard, dict) else {}
    eef_checks = eef_path.get("checks")
    eef_checks = eef_checks if isinstance(eef_checks, dict) else {}
    fallback = result.get("fallback_offset", metrics.get("fallback_offset"))
    fallback_ok = (
        isinstance(fallback, list)
        and len(fallback) == 3
        and all(_finite_abs_le(value, 0.005 + 1e-9) for value in fallback)
    )
    fallback_orthogonal = (
        fallback_ok
        and requested_world_ok
        and abs(
            sum(
                float(offset) * float(delta)
                for offset, delta in zip(fallback, requested_world, strict=True)
            )
        )
        <= 1e-8
    )
    attempts = metrics.get("candidate_attempts")
    attempts = attempts if isinstance(attempts, list) else []
    final_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    checks.update(
        {
            "primitive_is_jog_eef": metrics.get("manual_primitive") == "jog_eef",
            "eef_requested_delta_matches": _finite_vector(
                metrics.get("requested_delta"),
                expected_delta,
            ),
            "eef_requested_frame_matches": metrics.get("requested_delta_frame")
            == "base_call_start",
            "eef_world_delta_fixed_norm": requested_world_ok,
            "eef_actual_target_finite": actual_target_ok,
            "eef_actual_target_matches_executed_candidate": actual_target_ok
            and final_attempt.get("target_xyz") == actual_target,
            "eef_fallback_matches_executed_candidate": fallback_ok
            and final_attempt.get("fallback_offset") == fallback,
            "eef_fallback_orthogonal_to_requested_delta": fallback_orthogonal,
            "eef_fixed_server_step": metrics.get("fixed_server_step") is True,
            "whole_body_motion_scope": metrics.get("motion_scope") == "whole_body",
            "whole_body_certificate_v3": certificate.get("schema_version") == 3,
            "certificate_hand_matches": certificate.get("selected_hand") == hand,
            "eef_path_admitted": eef_path.get("admitted") is True,
            "eef_corridor_checks_pass": bool(eef_checks)
            and all(value is True for value in eef_checks.values()),
            "eef_cartesian_step_bounded": _finite_le(
                eef_path.get("max_cartesian_step_m"),
                0.0022 + 1e-7,
            ),
            "eef_lateral_bounded": _finite_le(
                eef_path.get("max_segment_lateral_m"),
                0.005 + 1e-7,
            ),
            "selected_eef_goal_only": metrics.get("selected_eef_goal_count") == 1
            and metrics.get("inactive_eef_goal_count") == 0,
            "both_attachments_bound": metrics.get("attachment_hand_count") == 2,
            "collision_path_admitted": collision.get("admitted") is True
            and collision.get("self_collision_check") is True
            and collision.get("world_collision_check") is True
            and collision.get("post_interpolation_check") is True,
            "eef_terminal_error_bounded": _finite_le(
                metrics.get("final_position_error_m"),
                0.005 + 1e-7,
            ),
            "live_waypoint_position_bounded": _finite_le(
                live_guard.get("max_observed_live_waypoint_position_error_m"),
                0.005 + 1e-7,
            ),
            "live_waypoint_orientation_bounded": _finite_le(
                live_guard.get("max_observed_live_waypoint_orientation_error_rad"),
                math.radians(1.0) + 1e-7,
            ),
            "live_corridor_lateral_bounded": _finite_le(
                live_guard.get("max_observed_live_lateral_m"),
                0.005 + 1e-7,
            ),
            "fallback_offset_bounded": fallback_ok,
        }
    )
    return checks


def _gripper_safety_checks(
    *,
    before_rows: list[dict[str, Any]],
    target: str,
    action: str,
) -> dict[str, bool]:
    if target not in {"left_arm", "right_arm"} or action not in {
        "open",
        "close",
    }:
        return {}
    hand = "left" if target == "left_arm" else "right"
    latest_result = (
        before_rows[-1].get("result")
        if before_rows and isinstance(before_rows[-1].get("result"), dict)
        else {}
    )
    capability = latest_result.get("capability")
    capability = capability if isinstance(capability, dict) else {}
    attachments = capability.get("attachments")
    attachments = attachments if isinstance(attachments, dict) else {}
    by_hand = attachments.get("by_hand")
    by_hand = by_hand if isinstance(by_hand, dict) else {}
    hand_attachment = by_hand.get(hand)
    hand_attachment = hand_attachment if isinstance(hand_attachment, dict) else {}
    gripper_state = capability.get("gripper_state")
    gripper_state = gripper_state if isinstance(gripper_state, dict) else {}
    checks = {
        "attachment_feedback_available": attachments.get("available") is True,
        "attachment_identity_not_conflicted": attachments.get("conflict") is False,
        "selected_hand_unattached": hand_attachment.get("attached") is False,
    }
    if action == "close":
        checks["close_selected_gripper_initially_open"] = (
            gripper_state.get(hand) == "open"
        )
    else:
        checks["open_selected_gripper_initially_closed_or_open"] = (
            gripper_state.get(hand) in {"closed", "open"}
        )
    return checks


def _exercise_button(
    *,
    page: Any,
    base_url: str,
    run_id: str,
    output_dir: Path,
    target: str,
    action: str,
    case_index: int,
    timeout_s: float,
    expected_enabled: bool,
    hold_ms: int = 90,
) -> dict[str, Any]:
    _progress("button_begin", case_index=case_index, target=target, action=action)
    before = _run_payload(base_url, run_id)
    before_rows = _manual_rows(before)
    fetches_before = len(_fetch_audit(page))
    state = _button_state(page, action)
    record: dict[str, Any] = {
        "case_index": case_index,
        "target": target,
        "action": action,
        "oracle_expected_enabled": expected_enabled,
        "button": state,
        "before": {
            "timeline_count": len(before_rows),
            "timeline_revision": before.get("timeline_revision"),
            "simulator_step": before.get("simulator_step"),
            "frame_revisions": before.get("frame_revisions"),
            "capture_group_id": before.get("capture_group_id"),
            "capture": _capture_snapshot(before),
        },
    }
    actual_enabled = state["aria_disabled"] != "true"
    record["enabled"] = actual_enabled
    record["oracle_matches_dom"] = actual_enabled is expected_enabled
    if actual_enabled is not expected_enabled:
        record.update(
            {
                "verdict": "fail",
                "reason": (
                    "release oracle/DOM capability mismatch; no activation "
                    "events were sent"
                ),
                "command_requests": [],
            }
        )
        _progress(
            "button_terminal",
            case_index=case_index,
            target=target,
            action=action,
            verdict="fail",
            enabled=actual_enabled,
            oracle_expected_enabled=expected_enabled,
        )
        return record

    gripper_safety = _gripper_safety_checks(
        before_rows=before_rows,
        target=target,
        action=action,
    )
    if gripper_safety and not all(gripper_safety.values()):
        record.update(
            {
                "verdict": "fail",
                "reason": (
                    "gripper safety precondition failed; no activation events were sent"
                ),
                "gripper_safety_checks": gripper_safety,
                "command_requests": [],
            }
        )
        _progress(
            "button_terminal",
            case_index=case_index,
            target=target,
            action=action,
            verdict="fail",
            enabled=actual_enabled,
        )
        return record

    if not expected_enabled:
        _progress(
            "button_disabled_probe",
            case_index=case_index,
            target=target,
            action=action,
        )
        pointer_id = 1000 + case_index
        _dispatch_pointer_event(
            page,
            action=action,
            event_type="pointerdown",
            pointer_id=pointer_id,
        )
        _dispatch_pointer_event(
            page,
            action=action,
            event_type="pointerup",
            pointer_id=pointer_id,
        )
        _dispatch_keyboard_activation(page, action=action, key=" ")
        _dispatch_keyboard_activation(page, action=action, key="Enter")
        page.sleep(180)
        after = _run_payload(base_url, run_id)
        control_fetches = [
            item
            for item in _fetch_audit(page)[fetches_before:]
            if any(
                str(item.get("url", "")).endswith(endpoint)
                for endpoint in CONTROL_MUTATION_ENDPOINTS
            )
        ]
        endpoint_counts = {
            endpoint.rsplit("/", 1)[-1]: sum(
                str(item.get("url", "")).endswith(endpoint) for item in control_fetches
            )
            for endpoint in CONTROL_MUTATION_ENDPOINTS
        }
        after_state = {
            "timeline_count": len(_manual_rows(after)),
            "timeline_revision": after.get("timeline_revision"),
            "simulator_step": after.get("simulator_step"),
            "frame_revisions": after.get("frame_revisions"),
            "capture_group_id": after.get("capture_group_id"),
        }
        disabled_checks = {
            "zero_command_requests": endpoint_counts["command"] == 0,
            "zero_heartbeat_requests": endpoint_counts["heartbeat"] == 0,
            "zero_stop_requests": endpoint_counts["stop"] == 0,
            "timeline_unchanged": len(_manual_rows(after)) == len(before_rows),
            "timeline_revision_unchanged": (
                after.get("timeline_revision") == before.get("timeline_revision")
            ),
            "simulator_step_unchanged": (
                after.get("simulator_step") == before.get("simulator_step")
            ),
            "frame_revisions_unchanged": (
                after.get("frame_revisions") == before.get("frame_revisions")
            ),
            "capture_group_unchanged": (
                after.get("capture_group_id") == before.get("capture_group_id")
            ),
        }
        record.update(
            {
                "activation_events": [
                    "pointerdown",
                    "pointerup",
                    "Space.keydown",
                    "Space.keyup",
                    "Enter.keydown",
                    "Enter.keyup",
                ],
                "control_requests": control_fetches,
                "control_request_counts": endpoint_counts,
                "command_requests": [],
                "after": after_state,
                "checks": disabled_checks,
            }
        )
        record["verdict"] = "pass" if all(disabled_checks.values()) else "fail"
        _progress(
            "button_terminal",
            case_index=case_index,
            target=target,
            action=action,
            verdict=record["verdict"],
            enabled=False,
        )
        return record

    _progress(
        "button_pointerdown",
        case_index=case_index,
        target=target,
        action=action,
    )
    pointer_id = 1000 + case_index
    _dispatch_pointer_event(
        page,
        action=action,
        event_type="pointerdown",
        pointer_id=pointer_id,
    )
    page.sleep(hold_ms)
    _dispatch_pointer_event(
        page,
        action=action,
        event_type="pointerup",
        pointer_id=pointer_id,
    )
    accepted_requests = _wait(
        lambda: (
            values
            if (
                values := [
                    item
                    for item in _fetch_audit(page)[fetches_before:]
                    if str(item.get("url", "")).endswith("/api/run/control/command")
                    and item.get("status") == 202
                ]
            )
            else None
        ),
        timeout_s=10,
        interval_s=0.05,
    )
    if len(accepted_requests) != 1:
        raise RuntimeError(
            "expected exactly one accepted command request before terminal wait"
        )
    accepted_response = accepted_requests[0].get("response")
    accepted_response = accepted_response if isinstance(accepted_response, dict) else {}
    accepted_control = accepted_response.get("control")
    accepted_control = accepted_control if isinstance(accepted_control, dict) else {}
    accepted_command_id = accepted_response.get(
        "command_id",
        accepted_control.get("command_id"),
    )
    if not isinstance(accepted_command_id, str) or not accepted_command_id:
        raise RuntimeError("accepted command response omitted command_id")
    payload, row = _wait(
        lambda: _terminal_command_row(
            base_url,
            run_id,
            accepted_command_id,
        ),
        timeout_s=timeout_s,
        interval_s=0.15,
    )
    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    before_capture_revision = _capture_snapshot(before).get("revision")
    task_success = _raw_success_latched(payload, result)
    if task_success:
        # The terminal receipt is the raw-success cutover.  From this point the
        # acceptance driver performs only passive state reads: no playback wait,
        # frame HTTP read, observe, hold, or capture request is permitted.
        payload = _wait(
            lambda: (
                settled
                if (
                    _raw_success_latched(settled, result)
                    and _manual_control_quiescent(settled)
                    and _capture_snapshot(settled).get("phase")
                    in {"idle", "discarded"}
                )
                else None
            )
            if isinstance(
                (settled := _run_payload(base_url, run_id)),
                dict,
            )
            else None,
            timeout_s=10,
            interval_s=0.1,
        )
    else:
        _wait(
            lambda: page.evaluate(
                """
                activeLeaseId === null
                && !controlCommandInFlight
                && controlLoopPromise === null
                && stopLeasePromise === null
                """
            ),
            timeout_s=5,
        )
        payload = _wait(
            lambda: (
                (settled if _manual_control_quiescent(settled) else None)
                if isinstance(
                    (settled := _run_payload(base_url, run_id)),
                    dict,
                )
                else None
            ),
            timeout_s=10,
            interval_s=0.1,
        )
        timeline_step = int(row.get("step") or 0)
        _wait(
            lambda: page.evaluate(
                f"""
                autoActionLastStep >= {timeline_step}
                && autoActionPlayback === null
                && frameKind === selectedCamera
                """
            ),
            timeout_s=timeout_s,
            interval_s=0.1,
        )
        page.sleep(100)
        payload = _wait(
            lambda: (
                settled
                if (
                    isinstance((capture := _capture_snapshot(settled)), dict)
                    and capture.get("phase") in {"completed", "failed"}
                    and isinstance(capture.get("revision"), int)
                    and (
                        not isinstance(before_capture_revision, int)
                        or capture["revision"] > before_capture_revision
                    )
                )
                else None
            )
            if isinstance(
                (settled := _run_payload(base_url, run_id)),
                dict,
            )
            else None,
            timeout_s=10,
            interval_s=0.1,
        )
    action_terminal_payload = payload
    command_fetches = [
        item
        for item in _fetch_audit(page)[fetches_before:]
        if str(item.get("url", "")).endswith("/api/run/control/command")
    ]
    command_request = command_fetches[0] if len(command_fetches) == 1 else {}
    command_response = command_request.get("response")
    command_response = command_response if isinstance(command_response, dict) else {}
    response_control = command_response.get("control")
    response_control = response_control if isinstance(response_control, dict) else {}
    response_command_id = command_response.get(
        "command_id",
        response_control.get("command_id"),
    )
    request_body = command_request.get("body")
    request_body = request_body if isinstance(request_body, dict) else {}
    frame_evidence = (
        {
            "skipped": True,
            "reason": "official_task_success_latched",
            "post_success_frame_http_reads": 0,
        }
        if task_success
        else _save_frames(
            base_url=base_url,
            run_id=run_id,
            output_dir=output_dir,
            case_index=case_index,
            payload=payload,
        )
    )
    before_revisions = before.get("frame_revisions") or {}
    after_revisions = payload.get("frame_revisions") or {}
    revisions_advanced = all(
        isinstance(before_revisions.get(camera), int)
        and isinstance(after_revisions.get(camera), int)
        and after_revisions[camera] == before_revisions[camera] + 1
        for camera in CAMERAS
    )
    capture = _capture_snapshot(payload)
    capture_checks = {
        "action_terminal_quiescent_before_capture_wait": (
            _manual_control_quiescent(action_terminal_payload)
        ),
        "capture_plane_terminal": (
            capture.get("phase") in {"idle", "discarded"}
            if task_success
            else capture.get("phase") == "completed"
        ),
        "capture_revision_advanced": (
            task_success
            or (
                isinstance(capture.get("revision"), int)
                and (
                    not isinstance(before_capture_revision, int)
                    or capture["revision"] > before_capture_revision
                )
            )
        ),
    }
    record.update(
        {
            "enabled": True,
            "gripper_safety_checks": gripper_safety,
            "hold_ms": hold_ms,
            "command_requests": command_fetches,
            "command_id": row.get("command_id"),
            "timeline": row,
            "frame_evidence": frame_evidence,
            "after": {
                "timeline_count": len(_manual_rows(payload)),
                "timeline_revision": payload.get("timeline_revision"),
                "simulator_step": payload.get("simulator_step"),
                "frame_revisions": payload.get("frame_revisions"),
                "frame_indices": payload.get("frame_indices"),
                "capture_group_id": payload.get("capture_group_id"),
                "terminated": payload.get("terminated"),
                "success_latched": _raw_success_latched(payload, result),
            },
            "checks": {
                "exactly_one_command_request": len(command_fetches) == 1,
                "command_post_status_202": command_request.get("status") == 202,
                "command_request_target_action_match": (
                    request_body.get("run") == run_id
                    and request_body.get("target") == target
                    and request_body.get("action") == action
                ),
                "command_response_id_matches_timeline": (
                    response_command_id == row.get("command_id")
                    and bool(response_command_id)
                ),
                "oracle_matches_dom": record["oracle_matches_dom"],
                "terminal_completed": row.get("status") == "completed",
                "primitive_success": result.get("primitive_success") is True,
                "partial_motion_false": result.get("partial_motion") is not True,
                "backend_manual_control_quiescent": (
                    _manual_control_quiescent(action_terminal_payload)
                ),
                "three_revisions_advanced_once": (
                    revisions_advanced if not task_success
                    else after_revisions == before_revisions
                ),
                "raw_success_cutover_zero_frame_reads": (
                    not task_success
                    or frame_evidence.get("post_success_frame_http_reads") == 0
                ),
                **_complete_frame_group_checks(payload),
                **capture_checks,
                "observe_zero_step": (
                    action != "observe"
                    or payload.get("simulator_step") == before.get("simulator_step")
                ),
                **_semantic_checks(
                    target=target,
                    action=action,
                    row=row,
                    result=result,
                ),
            },
        }
    )
    record["verdict"] = "pass" if all(record["checks"].values()) else "fail"
    _progress(
        "button_terminal",
        case_index=case_index,
        target=target,
        action=action,
        verdict=record["verdict"],
        enabled=True,
        command_id=record.get("command_id"),
        status=row.get("status"),
    )
    return record


def _check_initial_ui(page: Any) -> dict[str, Any]:
    state = page.evaluate(
        """
        (() => {
          const targets = [...document.querySelectorAll(".target-button")].map(b => {
            const r = b.getBoundingClientRect();
            const s = getComputedStyle(b);
            return {
              target: b.dataset.target,
              selected: b.classList.contains("selected"),
              aria_disabled: b.getAttribute("aria-disabled"),
              width: r.width,
              height: r.height,
              padding: s.padding,
              font_size: s.fontSize,
              border: s.border,
              border_radius: s.borderRadius,
              box_shadow: s.boxShadow
            };
          });
          return {
            controls_expanded: controlsExpanded,
            selected_target: selectedTarget,
            selected_camera: selectedCamera,
            frame_kind: frameKind,
            visible_head_image: (() => {
              const image = document.querySelector("img.frame-media.visible");
              return !!image && image.complete && image.naturalWidth > 0;
            })(),
            targets,
            cameras: [...document.querySelectorAll(
              ".behavior-frame-tabs button"
            )].map(b => ({
              camera: b.dataset.camera,
              active: b.classList.contains("active")
            }))
          };
        })()
        """
    )
    targets = state.get("targets") if isinstance(state, dict) else None
    targets = targets if isinstance(targets, list) else []
    cameras = state.get("cameras") if isinstance(state, dict) else None
    cameras = cameras if isinstance(cameras, list) else []
    target_styles = [
        {
            key: item.get(key)
            for key in (
                "width",
                "height",
                "padding",
                "font_size",
                "border_radius",
            )
        }
        for item in targets
        if isinstance(item, dict)
    ]
    checks = {
        "controls_initially_expanded": state.get("controls_expanded") is True,
        "chassis_initially_selected": state.get("selected_target") == "chassis",
        "head_initially_selected": state.get("selected_camera") == "head",
        "head_initial_frame_kind": state.get("frame_kind") == "head",
        "head_image_visibly_decoded": state.get("visible_head_image") is True,
        "exact_three_targets": [
            item.get("target") for item in targets if isinstance(item, dict)
        ]
        == list(TARGETS),
        "target_styles_identical": len(target_styles) == 3
        and target_styles[1:] == target_styles[:-1],
        "only_chassis_selected": [
            item.get("target")
            for item in targets
            if isinstance(item, dict) and item.get("selected") is True
        ]
        == ["chassis"],
        "exact_three_camera_tabs": [
            item.get("camera") for item in cameras if isinstance(item, dict)
        ]
        == list(CAMERAS),
        "only_head_camera_active": [
            item.get("camera")
            for item in cameras
            if isinstance(item, dict) and item.get("active") is True
        ]
        == ["head"],
    }
    return {
        "state": state,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def _check_release_oracle_dom(page: Any) -> dict[str, Any]:
    fetches_before = len(_fetch_audit(page))
    cases = []
    for target in TARGETS:
        _select_target(page, target)
        for action in ALL_ACTIONS:
            state = _button_state(page, action)
            expected_enabled = (target, action) in RELEASE_EXPECTED_ENABLED
            actual_enabled = state.get("aria_disabled") != "true"
            cases.append(
                {
                    "target": target,
                    "action": action,
                    "expected_enabled": expected_enabled,
                    "actual_enabled": actual_enabled,
                    "matches": expected_enabled is actual_enabled,
                }
            )
    _select_target(page, "chassis")
    control_fetches = [
        item
        for item in _fetch_audit(page)[fetches_before:]
        if any(
            str(item.get("url", "")).endswith(endpoint)
            for endpoint in CONTROL_MUTATION_ENDPOINTS
        )
    ]
    enabled = sum(item["actual_enabled"] for item in cases)
    disabled = sum(not item["actual_enabled"] for item in cases)
    checks = {
        "exact_33_dom_cases": len(cases) == 33,
        "exact_29_enabled": enabled == 29,
        "exact_4_disabled": disabled == 4,
        "all_dom_states_match_release_oracle": all(item["matches"] for item in cases),
        "no_control_mutation_requests": not control_fetches,
        "restored_chassis_target": page.evaluate('selectedTarget === "chassis"')
        is True,
    }
    return {
        "cases": cases,
        "enabled": enabled,
        "disabled": disabled,
        "control_requests": control_fetches,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def _exercise_collapse(page: Any) -> dict[str, Any]:
    fetches_before = len(_fetch_audit(page))
    page.evaluate("document.querySelector('.controls-toggle').click()")
    _wait(
        lambda: page.evaluate("controlsExpanded === false"),
        timeout_s=3,
    )
    collapsed = page.evaluate(
        """
        (() => ({
          expanded: controlsExpanded,
          tabs: getComputedStyle(
            document.querySelector(".behavior-frame-tabs")
          ).display,
          rails: [...document.querySelectorAll(".control-rail")].map(
            rail => getComputedStyle(rail).display
          ),
          collapsed_toggle: getComputedStyle(
            document.querySelector(".collapsed-toggle")
          ).display
        }))()
        """
    )
    page.evaluate("document.querySelector('.collapsed-toggle').click()")
    _wait(
        lambda: page.evaluate("controlsExpanded === true"),
        timeout_s=3,
    )
    expanded = page.evaluate(
        """
        (() => ({
          expanded: controlsExpanded,
          tabs: getComputedStyle(
            document.querySelector(".behavior-frame-tabs")
          ).display,
          rails: [...document.querySelectorAll(".control-rail")].map(
            rail => getComputedStyle(rail).display
          )
        }))()
        """
    )
    control_fetches = [
        item
        for item in _fetch_audit(page)[fetches_before:]
        if any(
            str(item.get("url", "")).endswith(endpoint)
            for endpoint in CONTROL_MUTATION_ENDPOINTS
        )
    ]
    checks = {
        "collapsed_state_false": collapsed.get("expanded") is False,
        "collapsed_tabs_visible": collapsed.get("tabs") != "none",
        "collapsed_rails_hidden": bool(collapsed.get("rails"))
        and all(value == "none" for value in collapsed["rails"]),
        "collapsed_toggle_visible": collapsed.get("collapsed_toggle") != "none",
        "expanded_state_true": expanded.get("expanded") is True,
        "expanded_tabs_visible": expanded.get("tabs") != "none",
        "expanded_rails_visible": bool(expanded.get("rails"))
        and all(value != "none" for value in expanded["rails"]),
        "no_control_mutation_requests": not control_fetches,
    }
    return {
        "collapsed": collapsed,
        "expanded": expanded,
        "control_requests": control_fetches,
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def _exercise_camera(
    *,
    page: Any,
    base_url: str,
    run_id: str,
    camera: str,
) -> dict[str, Any]:
    before = _run_payload(base_url, run_id)
    fetches_before = len(_fetch_audit(page))
    _select_camera(page, camera)
    requests = _wait(
        lambda: (
            values
            if (
                values := [
                    item
                    for item in _fetch_audit(page)[fetches_before:]
                    if str(item.get("url", "")).endswith("/api/run/control/camera")
                    and (
                        isinstance(item.get("status"), int)
                        or item.get("error") is not None
                    )
                ]
            )
            else None
        ),
        timeout_s=5,
    )
    page.sleep(100)
    visible = _wait(
        lambda: (
            state
            if (
                (
                    state := page.evaluate(
                        """
                    (() => {
                      const image = document.querySelector(
                        "img.frame-media.visible"
                      );
                      if (!image) return null;
                      return {
                        complete: image.complete,
                        natural_width: image.naturalWidth,
                        src: image.currentSrc || image.src
                      };
                    })()
                    """
                    )
                )
                and state.get("complete") is True
                and int(state.get("natural_width", 0)) > 0
                and "/api/run/frame" in str(state.get("src"))
                and f"kind={camera}" in str(state.get("src"))
            )
            else None
        ),
        timeout_s=10,
    )
    after = _run_payload(base_url, run_id)
    frame_data = _http_bytes(
        base_url,
        f"/api/run/frame?run={quote(run_id)}&kind={camera}&t={time.time_ns()}",
    )
    dom = page.evaluate(
        f"""
        (() => {{
          const button = document.querySelector('[data-camera="{camera}"]');
          return {{
            selected_camera: selectedCamera,
            frame_kind: frameKind,
            active: button.classList.contains("active")
          }};
        }})()
        """
    )
    request = requests[-1]
    response = request.get("response")
    response = response if isinstance(response, dict) else {}
    response_control = response.get("control")
    response_control = response_control if isinstance(response_control, dict) else {}
    response_camera = response.get(
        "camera",
        response_control.get("selected_camera"),
    )
    after_control = after.get("control")
    after_control = after_control if isinstance(after_control, dict) else {}
    checks = {
        "exactly_one_camera_post": len(requests) == 1,
        "camera_post_method": str(request.get("method", "")).upper() == "POST",
        "camera_post_status_2xx": isinstance(request.get("status"), int)
        and 200 <= request["status"] < 300,
        "camera_post_body_matches": request.get("body")
        == {"run": run_id, "camera": camera},
        "camera_post_response_matches": response_camera == camera,
        "backend_selected_camera_matches": after_control.get("selected_camera")
        == camera,
        "selected_camera_matches": dom.get("selected_camera") == camera,
        "frame_kind_matches": dom.get("frame_kind") == camera,
        "active_tab_matches": dom.get("active") is True,
        "visible_camera_image_decoded": visible.get("complete") is True
        and int(visible.get("natural_width", 0)) > 0,
        "visible_camera_image_src_matches": f"kind={camera}" in str(visible.get("src")),
        "capture_group_unchanged": before.get("capture_group_id")
        == after.get("capture_group_id"),
        "simulator_step_unchanged": before.get("simulator_step")
        == after.get("simulator_step"),
        "timeline_revision_unchanged": before.get("timeline_revision")
        == after.get("timeline_revision"),
        "frame_revisions_unchanged": before.get("frame_revisions")
        == after.get("frame_revisions"),
        "frame_indices_unchanged": before.get("frame_indices")
        == after.get("frame_indices"),
        "frame_http_png_loaded": len(frame_data) > 8
        and frame_data.startswith(b"\x89PNG\r\n\x1a\n"),
    }
    return {
        "camera": camera,
        "before": {
            key: before.get(key)
            for key in (
                "capture_group_id",
                "simulator_step",
                "timeline_revision",
                "frame_revisions",
                "frame_indices",
            )
        },
        "after": {
            key: after.get(key)
            for key in (
                "capture_group_id",
                "simulator_step",
                "timeline_revision",
                "frame_revisions",
                "frame_indices",
            )
        },
        "dom": dom,
        "visible_image": visible,
        "requests": requests,
        "frame_http": {
            "bytes": len(frame_data),
            "sha256": hashlib.sha256(frame_data).hexdigest(),
        },
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def _sustain_live_run(
    *,
    page: Any,
    base_url: str,
    run_id: str,
    output_path: Path,
    duration_s: float,
    interval_s: float,
) -> dict[str, Any]:
    if duration_s <= 0.0:
        return {
            "duration_requested_s": duration_s,
            "samples": 0,
            "verdict": "not_run",
            "reason": "sustain duration was not positive",
        }
    interval_s = min(max(interval_s, 0.25), 60.0)
    started = time.monotonic()
    samples = 0
    observe_probes = 0
    baseline: dict[str, Any] | None = None
    passive_freshness_observed = False
    failed_checks: list[dict[str, Any]] = []
    while True:
        payload = _run_payload(base_url, run_id)
        control = payload.get("control")
        control = control if isinstance(control, dict) else {}
        if _raw_success_latched(payload):
            sample = {
                "sample_index": samples,
                "elapsed_s": round(time.monotonic() - started, 3),
                "official_success_stop": True,
                "control_revision": control.get("control_revision"),
                "timeline_revision": payload.get("timeline_revision"),
                "verdict": "pass",
            }
            _append_jsonl(output_path, sample)
            samples += 1
            break
        browser_health = page.evaluate(
            """
            (() => {
              const key = "__rpentAcceptanceSustainProbe";
              let probe = window[key];
              if (!probe || probe.run !== curRun) {
                probe = {
                  run: curRun,
                  source: null,
                  message_count: 0,
                  last_message_at_ms: null,
                  listener_attached: false,
                };
                window[key] = probe;
              }
              if (
                !!es
                && probe.source !== es
                && typeof es.addEventListener === "function"
              ) {
                probe.source = es;
                es.addEventListener("message", () => {
                  probe.message_count += 1;
                  probe.last_message_at_ms = Date.now();
                });
                probe.listener_attached = true;
              }
              return {
                document_ready: document.readyState === "complete",
                event_source_open_or_connecting:
                  !!es && es.readyState !== EventSource.CLOSED,
                current_run: curRun,
                sse_listener_attached: probe.listener_attached,
                sse_message_count: probe.message_count,
                sse_last_message_age_ms:
                  probe.last_message_at_ms == null
                    ? null
                    : Math.max(0, Date.now() - probe.last_message_at_ms),
              };
            })()
            """
        )
        frame_revisions = payload.get("frame_revisions")
        frame_revisions = (
            frame_revisions if isinstance(frame_revisions, dict) else {}
        )
        frame_indices = payload.get("frame_indices")
        frame_indices = frame_indices if isinstance(frame_indices, dict) else {}
        if baseline is None:
            baseline = {
                "control_revision": control.get("control_revision"),
                "capture_group_id": payload.get("capture_group_id"),
                "frame_revisions": dict(frame_revisions),
                "frame_indices": dict(frame_indices),
            }
        baseline_control_revision = baseline.get("control_revision")
        current_control_revision = control.get("control_revision")
        control_revision_advanced = bool(
            isinstance(baseline_control_revision, int)
            and isinstance(current_control_revision, int)
            and current_control_revision > baseline_control_revision
        )
        baseline_revisions = baseline.get("frame_revisions")
        baseline_revisions = (
            baseline_revisions if isinstance(baseline_revisions, dict) else {}
        )
        baseline_indices = baseline.get("frame_indices")
        baseline_indices = (
            baseline_indices if isinstance(baseline_indices, dict) else {}
        )
        frame_lineage_advanced = any(
            (
                isinstance(frame_revisions.get(camera), int)
                and isinstance(baseline_revisions.get(camera), int)
                and frame_revisions[camera] > baseline_revisions[camera]
            )
            or (
                isinstance(frame_indices.get(camera), int)
                and isinstance(baseline_indices.get(camera), int)
                and frame_indices[camera] > baseline_indices[camera]
            )
            for camera in CAMERAS
        )
        baseline_group = baseline.get("capture_group_id")
        current_group = payload.get("capture_group_id")
        capture_group_advanced = bool(
            baseline_group is not None
            and current_group is not None
            and current_group != baseline_group
        )
        sse_message_count = browser_health.get("sse_message_count")
        sse_message_advanced = bool(
            isinstance(sse_message_count, int)
            and not isinstance(sse_message_count, bool)
            and sse_message_count > 0
        )
        freshness = {
            "sse_message_advanced": sse_message_advanced,
            "control_revision_advanced": control_revision_advanced,
            "frame_lineage_advanced": (
                frame_lineage_advanced or capture_group_advanced
            ),
        }
        passive_freshness_observed = (
            passive_freshness_observed or any(freshness.values())
        )
        checks = {
            **_control_ready_checks(payload),
            **_complete_frame_group_checks(payload),
            "control_revision_readable": isinstance(
                control.get("control_revision"), int
            ),
            "browser_document_ready": (
                browser_health.get("document_ready") is True
            ),
            "browser_event_source_not_closed": (
                browser_health.get("event_source_open_or_connecting") is True
            ),
            "browser_current_run_matches": (
                browser_health.get("current_run") == run_id
            ),
        }
        camera_http: dict[str, Any] = {}
        for camera in CAMERAS:
            data = _http_bytes(
                base_url,
                (
                    f"/api/run/frame?run={quote(run_id)}&kind={camera}"
                    f"&t={time.time_ns()}"
                ),
            )
            camera_http[camera] = {
                "bytes": len(data),
                "png": len(data) > 8 and data.startswith(b"\x89PNG\r\n\x1a\n"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            checks[f"{camera}_http_png"] = camera_http[camera]["png"]
        sample = {
            "sample_index": samples,
            "elapsed_s": round(time.monotonic() - started, 3),
            "capture_group_id": payload.get("capture_group_id"),
            "simulator_step": payload.get("simulator_step"),
            "control_revision": control.get("control_revision"),
            "timeline_revision": payload.get("timeline_revision"),
            "frame_revisions": payload.get("frame_revisions"),
            "frame_indices": payload.get("frame_indices"),
            "passive_freshness": freshness,
            "browser_health": browser_health,
            "checks": checks,
            "camera_http": camera_http,
            "verdict": "pass" if all(checks.values()) else "fail",
        }
        _append_jsonl(output_path, sample)
        samples += 1
        if sample["verdict"] == "fail":
            failed_checks.append(sample)
            break
        elapsed = time.monotonic() - started
        if elapsed >= duration_s:
            break
        time.sleep(min(interval_s, duration_s - elapsed))
    elapsed_s = time.monotonic() - started
    return {
        "duration_requested_s": duration_s,
        "duration_observed_s": round(elapsed_s, 3),
        "interval_s": interval_s,
        "samples": samples,
        "observe_probes": observe_probes,
        "env_action_probes": 0,
        "passive_freshness_observed": passive_freshness_observed,
        "failed_samples": len(failed_checks),
        "verdict": (
            "pass"
            if (
                not failed_checks
                and (
                    passive_freshness_observed
                    or sample.get("official_success_stop") is True
                )
                and (
                    elapsed_s >= duration_s
                    or sample.get("official_success_stop") is True
                )
            )
            else "fail"
        ),
    }


def _probe_video_http(base_url: str, path: str) -> dict[str, Any]:
    request = Request(
        base_url.rstrip("/") + path,
        headers={"Cache-Control": "no-store", "Range": "bytes=0-31"},
    )
    with urlopen(request, timeout=30) as response:
        data = response.read(32)
        status = response.status
        content_type = response.headers.get_content_type()
    return {
        "status": status,
        "content_type": content_type,
        "first_bytes_hex": data.hex(),
        "mp4_signature": len(data) >= 8 and data[4:8] == b"ftyp",
    }


def _visible_video_state(page: Any) -> dict[str, Any] | None:
    value = page.evaluate(
        """
        (() => {
          const video = document.querySelector("video.frame-media.visible");
          if (!video) return null;
          return {
            src: video.currentSrc || video.src,
            ready_state: video.readyState,
            duration: video.duration,
            current_time: video.currentTime,
            playback_rate: video.playbackRate,
            ended: video.ended
          };
        })()
        """
    )
    return value if isinstance(value, dict) else None


def _finish_visible_video(page: Any) -> None:
    page.evaluate(
        """
        (() => {
          const video = document.querySelector("video.frame-media.visible");
          if (!video || !Number.isFinite(video.duration) || video.duration <= 0) {
            throw new Error("visible video has no finite duration");
          }
          window.__rpentAcceptanceVideoEnded = false;
          video.addEventListener(
            "ended",
            () => { window.__rpentAcceptanceVideoEnded = true; },
            { once: true }
          );
          video.muted = true;
          video.currentTime = Math.max(0, video.duration - 0.05);
          const promise = video.play();
          if (promise && typeof promise.catch === "function") {
            promise.catch(error => console.error("acceptance video play", error));
          }
        })()
        """
    )


def _exercise_media(
    *,
    page: Any,
    base_url: str,
    run_id: str,
) -> dict[str, Any]:
    payload = _run_payload(base_url, run_id)
    original_camera = page.evaluate("selectedCamera")
    timeline = payload.get("timeline")
    timeline = timeline if isinstance(timeline, list) else []
    clip = next(
        (
            item
            for item in reversed(timeline)
            if isinstance(item, dict) and item.get("has_action_video") is True
        ),
        None,
    )
    action_record: dict[str, Any]
    if clip is None:
        action_record = {
            "verdict": "not_run",
            "reason": "action video is not packaged in this run",
        }
    else:
        step = clip.get("step")
        http = _probe_video_http(
            base_url,
            f"/api/run/action-video?run={quote(run_id)}&step={quote(str(step))}",
        )
        started = page.evaluate(
            f"""
            playActionVideo(
              {json.dumps(clip)},
              {{
                returnAfterEnd: true,
                returnKind: {json.dumps(original_camera)}
              }}
            )
            """
        )
        ready = _wait(
            lambda: (
                state
                if (
                    (state := _visible_video_state(page))
                    and state.get("ready_state", 0) >= 2
                    and "/api/run/action-video" in str(state.get("src"))
                )
                else None
            ),
            timeout_s=15,
        )
        _finish_visible_video(page)
        _wait(
            lambda: page.evaluate(
                f"""
                window.__rpentAcceptanceVideoEnded === true
                && frameKind === {json.dumps(original_camera)}
                && selectedCamera === {json.dumps(original_camera)}
                """
            ),
            timeout_s=5,
        )
        action_checks = {
            "play_started": started is True,
            "http_status_2xx": 200 <= int(http["status"]) < 300,
            "http_content_type_video": str(http["content_type"]).startswith("video/"),
            "http_mp4_signature": http["mp4_signature"] is True,
            "real_video_ready": ready.get("ready_state", 0) >= 2,
            "playback_rate_half": _finite_close(
                ready.get("playback_rate"),
                0.5,
            ),
            "natural_ended_observed": page.evaluate(
                "window.__rpentAcceptanceVideoEnded === true"
            )
            is True,
            "restored_original_camera_after_end": page.evaluate(
                f"""
                frameKind === {json.dumps(original_camera)}
                && selectedCamera === {json.dumps(original_camera)}
                """
            )
            is True,
        }
        action_record = {
            "step": step,
            "http": http,
            "ready": ready,
            "checks": action_checks,
            "verdict": "pass" if all(action_checks.values()) else "fail",
        }

    episode_record = {
        "verdict": "not_run",
        "phase": "post_shutdown",
        "reason": (
            "episode video is sealed only after run shutdown; live acceptance "
            "does not stop the environment"
        ),
        "live_run_state": payload.get("state"),
        "live_has_video": payload.get("has_video"),
    }
    return {
        "original_camera": original_camera,
        "live_action_video": action_record,
        "post_shutdown_episode_video": episode_record,
        "verdict": action_record["verdict"],
    }


def _layout_and_screenshot(page: Any, output_dir: Path) -> dict[str, Any]:
    page.set_device_metrics(
        width=853,
        height=578,
        device_scale_factor=1.5,
    )
    geometry = page.evaluate(
        """
        (async () => {
          await document.fonts.ready;
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
          const rect = selector => {
            const r = document.querySelector(selector).getBoundingClientRect();
            return {x:r.x,y:r.y,width:r.width,height:r.height};
          };
          return {
            dpr:devicePixelRatio,
            frame:rect("#framewrap"), left:rect(".control-left"),
            stage:rect(".frame-stage"), right:rect(".control-right"),
            toggle:rect(".control-left .controls-toggle"),
            targets:[...document.querySelectorAll(".target-button")].map(b => {
              const r=b.getBoundingClientRect();
              return {target:b.dataset.target,x:r.x,y:r.y,width:r.width,height:r.height};
            }),
            tabs:rect(".behavior-frame-tabs"),
            dpad:rect(".dpad"), observe:rect('[data-action="observe"]'),
            up:rect('[data-action="up"]'),
            rotate_left:rect('[data-action="rotate_left"]'),
            open:rect('[data-action="open"]'),
            timeline_header:rect(".col.right > .col .panel-title")
          };
        })()
        """
    )
    page.evaluate(
        """
        (() => {
          const active = document.activeElement;
          if (active instanceof HTMLElement) active.blur();
          return document.activeElement === document.body;
        })()
        """
    )
    page.sleep(250)
    live_path = output_dir / "live_actual_1280x867.png"
    overlay_path = output_dir / "live_actual_reference_overlay_50pct_1280x867.png"
    diff_path = output_dir / "ui_only_masked_diff_1280x867.png"
    page.screenshot_clip(
        live_path,
        width=1280 / 1.5,
        height=867 / 1.5,
    )
    with Image.open(REFERENCE) as reference, Image.open(live_path) as live:
        reference = reference.convert("RGB")
        live = live.convert("RGB")
        reference_size = reference.size
        live_size = live.size
        if reference_size != (1280, 867) or live_size != reference_size:
            raise RuntimeError(
                "high-DPI Dashboard reference/live size mismatch: "
                f"reference={reference_size}, live={live_size}"
            )
        Image.blend(reference, live, 0.5).save(overlay_path)
        difference = ImageChops.difference(reference, live)
        mask = Image.new("L", reference.size, 255)
        # Dynamic simulator frame, caption, and live Timeline rows are not UI
        # chrome and therefore cannot be compared to the supplied example.
        mask.paste(0, (325, 0, 963, 657))
        mask.paste(0, (0, 598, 325, 657))
        mask.paste(0, (0, 719, 1280, 867))
        masked = Image.new("RGB", reference.size)
        masked.paste(difference, mask=mask)
        masked.save(diff_path)
        histogram = mask.histogram()
        included_pixels = sum(histogram[1:])
        stat = ImageStat.Stat(difference, mask=mask)
        mae = sum(stat.mean) / 3.0
        rmse = math.sqrt(sum(value * value for value in stat.rms) / 3.0)
        within_24_pixels = sum(
            max(pixel) <= 24
            for pixel, included in zip(
                difference.getdata(),
                mask.getdata(),
                strict=True,
            )
            if included
        )
        within_24_ratio = within_24_pixels / included_pixels
    expected_geometry = {
        "frame": {"x": 0, "y": 0, "width": 1280 / 1.5, "height": 438},
        "left": {"x": 0, "width": 325 / 1.5},
        "stage": {"x": 325 / 1.5, "width": 637 / 1.5},
        "right": {"x": 962 / 1.5, "width": 318 / 1.5},
        "toggle": {"y": 23 / 1.5},
        "tabs": {"y": 16 / 1.5},
        "dpad": {"y": 252 / 1.5, "width": 96, "height": 96},
        "observe": {"y": 468 / 1.5, "width": 39, "height": 39},
        "up": {"y": 235 / 1.5, "width": 39, "height": 39},
        "rotate_left": {"y": 365 / 1.5, "width": 39, "height": 39},
        "open": {"y": 494 / 1.5, "width": 39, "height": 39},
    }
    geometry_checks = {
        f"{element}_{coordinate}": _finite_close(
            geometry.get(element, {}).get(coordinate),
            expected,
            tolerance=2.0 / 1.5 if coordinate == "y" else 1.0,
        )
        for element, coordinates in expected_geometry.items()
        for coordinate, expected in coordinates.items()
    }
    expected_target_y = {
        "chassis": 149 / 1.5,
        "left_arm": 151 / 1.5,
        "right_arm": 151 / 1.5,
    }
    target_geometry_checks = {
        f"{target}_physical_y": _finite_close(
            next(
                (
                    item.get("y")
                    for item in geometry.get("targets", [])
                    if isinstance(item, dict)
                    and item.get("target") == target
                ),
                None,
            ),
            expected,
            tolerance=2.0 / 1.5,
        )
        for target, expected in expected_target_y.items()
    }
    checks = {
        "reference_hash_matches": hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
        == REFERENCE_SHA256,
        "reference_and_live_size_1280x867": (
            reference_size == live_size == (1280, 867)
        ),
        "device_pixel_ratio_1_5": _finite_close(
            geometry.get("dpr"),
            1.5,
            tolerance=1e-6,
        ),
        "ui_only_mae_within_threshold": mae <= HIGH_DPI_VISUAL_MAE_MAX,
        "ui_only_pixels_within_24_threshold": (
            within_24_ratio >= HIGH_DPI_VISUAL_WITHIN_24_MIN
        ),
        "target_buttons_vertically_aligned": len(
            {
                round(float(target.get("y")), 3)
                for target in geometry.get("targets", [])
                if isinstance(target, dict)
                and target.get("target") in {"chassis", "left_arm", "right_arm"}
                and isinstance(target.get("y"), (int, float))
            }
        )
        == 1,
        **geometry_checks,
        **target_geometry_checks,
    }
    return {
        "geometry": geometry,
        "expected_geometry": expected_geometry,
        "reference": str(REFERENCE),
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "live": str(live_path),
        "overlay": str(overlay_path),
        "ui_only_masked_diff": str(diff_path),
        "ui_only_included_pixels": included_pixels,
        "ui_only_rgb_mae": mae,
        "ui_only_rgb_rmse": rmse,
        "ui_only_within_24_ratio": within_24_ratio,
        "thresholds": {
            "ui_only_rgb_mae_max": HIGH_DPI_VISUAL_MAE_MAX,
            "ui_only_within_24_min": HIGH_DPI_VISUAL_WITHIN_24_MIN,
            "geometry_abs_tolerance_css_px": 1.0,
            "vertical_target_tolerance_physical_px": 2.0,
        },
        "checks": checks,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def _artifact_manifest(
    *,
    output_dir: Path,
    sections: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / "acceptance_manifest.json"
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path == manifest_path or path.name.endswith(".tmp"):
            continue
        data = path.read_bytes()
        artifacts.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "sections": sections,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    _json_dump(manifest_path, manifest)
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for artifact in loaded.get("artifacts", []):
        path = output_dir / artifact["path"]
        data = path.read_bytes()
        if len(data) != artifact["bytes"]:
            errors.append(f"{artifact['path']}: byte count changed")
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            errors.append(f"{artifact['path']}: sha256 changed")
    manifest["self_check"] = {
        "manifest_json_reloaded": isinstance(loaded, dict),
        "artifact_hashes_revalidated": not errors,
        "errors": errors,
    }
    _json_dump(manifest_path, manifest)
    return manifest


def main() -> int:
    args = _args()
    if not args.execute_sim_controls:
        raise SystemExit(
            "Refusing to send controls without --execute-sim-controls "
            "(BEHAVIOR/OmniGibson simulation only)."
        )
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise RuntimeError("Dashboard reference image identity changed")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"refusing to reuse non-empty acceptance output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _http_json(args.base_url, "/api/runs").get("runs") or []
    if args.run_id:
        run_id = args.run_id
    elif len(runs) == 1 and isinstance(runs[0], dict):
        run_id = str(runs[0]["id"])
    else:
        raise RuntimeError(f"expected one live run, got {runs!r}")

    if args.planning_only_probes:
        pre_capture_ready = _wait_for_run_ready(
            base_url=args.base_url,
            run_id=run_id,
            checks=_planning_only_pre_capture_ready_checks,
        )
        capture_preflight = _refresh_planning_only_capture(
            base_url=args.base_url,
            run_id=run_id,
            timeout_s=args.command_timeout_s,
        )
        _json_dump(
            output_dir / "planning_only_capture_preflight.json",
            capture_preflight,
        )
        _progress(
            "planning_only_capture_preflight_complete",
            verdict=capture_preflight["verdict"],
        )
        if capture_preflight["verdict"] != "pass":
            raise RuntimeError(
                "planning-only Dashboard capture preflight failed"
            )
        ready = _wait_for_run_ready(
            base_url=args.base_url,
            run_id=run_id,
            checks=_planning_only_ready_checks,
        )
        ready_record = {
            "payload": ready,
            "checks": _planning_only_ready_checks(ready),
            "pre_capture": {
                "payload": pre_capture_ready,
                "checks": _planning_only_pre_capture_ready_checks(
                    pre_capture_ready
                ),
            },
            "capture_preflight": capture_preflight,
            "verdict": "pass",
        }
        _json_dump(output_dir / "initial_run.json", ready_record)
        _progress("live_run_ready", run_id=run_id)
        summary = _run_planning_only_probes(
            base_url=args.base_url,
            run_id=run_id,
            output_dir=output_dir,
            timeout_s=args.command_timeout_s,
        )
        summary["capture_preflight"] = capture_preflight
        summary["reference_sha256"] = REFERENCE_SHA256
        summary["artifact_manifest"] = {"verdict": "pending"}
        _json_dump(output_dir / "planning_only_probes.json", summary)
        _json_dump(output_dir / "acceptance_summary.json", summary)
        sections = {
            "ready": ready_record,
            "planning_only": summary,
        }
        manifest = _artifact_manifest(output_dir=output_dir, sections=sections)
        manifest_ok = bool(
            manifest["self_check"]["manifest_json_reloaded"]
            and manifest["self_check"]["artifact_hashes_revalidated"]
            and not manifest["self_check"]["errors"]
        )
        summary["artifact_manifest"] = {
            "verdict": "pass" if manifest_ok else "fail",
            "artifact_count": manifest["artifact_count"],
        }
        if not manifest_ok:
            summary["verdict"] = "fail"
        _json_dump(output_dir / "planning_only_probes.json", summary)
        _json_dump(output_dir / "acceptance_summary.json", summary)
        manifest = _artifact_manifest(output_dir=output_dir, sections=sections)
        manifest_ok = bool(
            manifest["self_check"]["manifest_json_reloaded"]
            and manifest["self_check"]["artifact_hashes_revalidated"]
            and not manifest["self_check"]["errors"]
        )
        _progress(
            "planning_only_acceptance_complete",
            verdict=summary["verdict"],
            completed_probe_count=summary["completed_probe_count"],
        )
        if summary["verdict"] != "pass" or not manifest_ok:
            raise RuntimeError("planning-only Dashboard acceptance failed")
        return 0

    ready = _wait_for_run_ready(
        base_url=args.base_url,
        run_id=run_id,
        checks=_control_ready_checks,
    )
    ready_record = {
        "payload": ready,
        "checks": _control_ready_checks(ready),
        "verdict": "pass",
    }
    _json_dump(output_dir / "initial_run.json", ready_record)
    _progress("live_run_ready", run_id=run_id)

    ChromePage = _load_chrome_page()
    page = ChromePage(args.base_url.rstrip("/") + "/")
    matrix: list[dict[str, Any]] = []
    camera_cases: list[dict[str, Any]] = []
    run_snapshots = output_dir / "run_snapshots.jsonl"
    sustain_path = output_dir / "sustain_snapshots.jsonl"
    halt_reason: str | None = None
    official_success_stop = False
    initial_ui: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "browser did not initialize",
    }
    collapse: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "browser did not initialize",
    }
    release_oracle: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "browser did not initialize",
    }
    sustain: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "first chassis observe did not complete",
    }
    media: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "control matrix did not complete",
    }
    visual: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "browser did not reach visual capture",
    }
    console: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "browser did not reach console capture",
    }
    browser_error: dict[str, Any] = {
        "verdict": "pass",
        "error": None,
    }
    cleanup: dict[str, Any] = {
        "verdict": "not_run",
        "reason": "cleanup did not execute",
    }
    try:
        _inject_fetch_audit(page)
        _progress("browser_ready")
        _wait(
            lambda: page.evaluate(
                f"curRun === {json.dumps(run_id)} && behaviorControlsEnabled"
            ),
            timeout_s=10,
        )
        _wait(
            lambda: page.evaluate(
                """
                (() => {
                  const image = document.querySelector(
                    "img.frame-media.visible"
                  );
                  return frameKind === "head"
                    && !!image
                    && image.complete
                    && image.naturalWidth > 0;
                })()
                """
            ),
            timeout_s=10,
        )
        initial_ui = _check_initial_ui(page)
        _json_dump(output_dir / "initial_ui.json", initial_ui)
        _progress("initial_ui_saved", verdict=initial_ui["verdict"])

        release_oracle = _check_release_oracle_dom(page)
        _json_dump(output_dir / "release_oracle.json", release_oracle)
        _progress(
            "release_oracle_checked",
            verdict=release_oracle["verdict"],
            enabled=release_oracle.get("enabled"),
            disabled=release_oracle.get("disabled"),
        )

        # Collapse/expand is exercised before any command lease exists.
        collapse = _exercise_collapse(page)
        _json_dump(output_dir / "collapse_expand.json", collapse)
        _progress("collapse_expand_complete", verdict=collapse["verdict"])
        if (
            initial_ui["verdict"] != "pass"
            or release_oracle["verdict"] != "pass"
            or collapse["verdict"] != "pass"
        ):
            halt_reason = (
                "initial UI, release oracle, or collapse/expand gate did not pass"
            )

        case_index = 0
        for target in TARGETS:
            _select_target(page, target)
            _progress("target_selected", target=target)
            for action in MOTION_ORDER[target]:
                case_index += 1
                expected_enabled = (
                    target,
                    action,
                ) in RELEASE_EXPECTED_ENABLED
                if halt_reason is not None and expected_enabled:
                    matrix.append(
                        {
                            "case_index": case_index,
                            "target": target,
                            "action": action,
                            "oracle_expected_enabled": True,
                            "verdict": "not_run",
                            "reason": halt_reason,
                        }
                    )
                    continue
                record = _exercise_button(
                    page=page,
                    base_url=args.base_url,
                    run_id=run_id,
                    output_dir=output_dir,
                    target=target,
                    action=action,
                    case_index=case_index,
                    timeout_s=args.command_timeout_s,
                    expected_enabled=expected_enabled,
                    hold_ms=90,
                )
                matrix.append(record)
                _append_jsonl(run_snapshots, record)
                if case_index == 1:
                    if record.get("verdict") == "pass":
                        sustain = _sustain_live_run(
                            page=page,
                            base_url=args.base_url,
                            run_id=run_id,
                            output_path=sustain_path,
                            duration_s=args.sustain_seconds,
                            interval_s=args.sustain_interval_s,
                        )
                    else:
                        sustain = {
                            "verdict": "not_run",
                            "reason": (
                                "first chassis observe did not establish a "
                                "complete frame group"
                            ),
                        }
                    _json_dump(output_dir / "sustain_summary.json", sustain)
                    _progress("sustain_complete", verdict=sustain["verdict"])
                    if sustain["verdict"] != "pass":
                        halt_reason = "sustain health gate did not pass"
                timeline = record.get("timeline") or {}
                result = (
                    timeline.get("result")
                    if isinstance(timeline, dict)
                    and isinstance(timeline.get("result"), dict)
                    else {}
                )
                if record["verdict"] == "fail":
                    halt_reason = (
                        f"{target}/{action} failed or produced incomplete evidence"
                    )
                if (
                    record.get("after", {}).get("success_latched") is True
                    or result.get("task_success") is True
                ):
                    halt_reason = "official task success latched"
                    official_success_stop = True
                    break
            if official_success_stop:
                break

        if official_success_stop:
            seen = {(item.get("target"), item.get("action")) for item in matrix}
            for target in TARGETS:
                for action in MOTION_ORDER[target]:
                    if (target, action) in seen:
                        continue
                    case_index += 1
                    sealed = {
                        "case_index": case_index,
                        "target": target,
                        "action": action,
                        "oracle_expected_enabled": (
                            (target, action) in RELEASE_EXPECTED_ENABLED
                        ),
                        "verdict": "not_run",
                        "reason": "official task success latched",
                    }
                    matrix.append(sealed)
                    _append_jsonl(run_snapshots, sealed)

        if halt_reason is None:
            for camera in ("left_wrist", "right_wrist", "head"):
                try:
                    camera_record = _exercise_camera(
                        page=page,
                        base_url=args.base_url,
                        run_id=run_id,
                        camera=camera,
                    )
                except Exception as exc:
                    camera_record = {
                        "camera": camera,
                        "verdict": "fail",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                camera_cases.append(camera_record)
                _progress(
                    "camera_terminal",
                    camera=camera,
                    verdict=camera_record["verdict"],
                )
        _json_dump(output_dir / "camera_cases.json", camera_cases)

        if halt_reason is None and all(
            item.get("verdict") == "pass" for item in camera_cases
        ):
            try:
                media = _exercise_media(
                    page=page,
                    base_url=args.base_url,
                    run_id=run_id,
                )
            except Exception as exc:
                media = {
                    "verdict": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        if official_success_stop:
            media = {
                "verdict": "not_run",
                "reason": (
                    "official task success latched; no further control or "
                    "playback transitions were initiated"
                ),
            }
        _json_dump(output_dir / "media_acceptance.json", media)
        if not official_success_stop:
            _select_target(page, "chassis")
            if page.evaluate('selectedCamera !== "head" || frameKind !== "head"'):
                _select_camera(page, "head")
            _wait(
                lambda: page.evaluate(
                    """
                    (() => {
                      const image = document.querySelector(
                        "img.frame-media.visible"
                      );
                      return frameKind === "head"
                        && !!image
                        && image.complete
                        && image.naturalWidth > 0;
                    })()
                    """
                ),
                timeout_s=10,
            )
        visual = _layout_and_screenshot(page, output_dir)
        _json_dump(output_dir / "dom_geometry.json", visual)
        _wait(
            lambda: all(
                item.get("completed_ms") is not None for item in _fetch_audit(page)
            ),
            timeout_s=5,
        )
        _json_dump(output_dir / "browser_control_requests.json", _fetch_audit(page))
        console_errors = page.evaluate("window.__rpentConsoleErrors || []")
        console_errors = console_errors if isinstance(console_errors, list) else []
        console = {
            "errors": console_errors,
            "count": len(console_errors),
            "verdict": "pass" if not console_errors else "fail",
        }
        _json_dump(output_dir / "console_errors.json", console)
    except Exception as exc:
        browser_error = {
            "verdict": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
        halt_reason = halt_reason or "browser acceptance raised an exception"
        _json_dump(output_dir / "browser_error.json", browser_error)
    finally:
        try:
            page.evaluate("stopAndWaitForManualControl('acceptance_cleanup')")
            quiescent = _wait(
                lambda: (
                    (payload if _manual_control_quiescent(payload) else None)
                    if isinstance(
                        (payload := _run_payload(args.base_url, run_id)),
                        dict,
                    )
                    else None
                ),
                timeout_s=10,
            )
            cleanup = {
                "checks": {
                    "browser_stop_and_wait_completed": True,
                    "backend_manual_control_quiescent": (
                        _manual_control_quiescent(quiescent)
                    ),
                },
            }
            cleanup["verdict"] = "pass" if all(cleanup["checks"].values()) else "fail"
        except Exception as exc:
            cleanup = {
                "verdict": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _json_dump(output_dir / "cleanup.json", cleanup)
        try:
            _json_dump(
                output_dir / "browser_control_requests.json",
                _fetch_audit(page),
            )
            console_errors = page.evaluate("window.__rpentConsoleErrors || []")
            console_errors = console_errors if isinstance(console_errors, list) else []
            console = {
                "errors": console_errors,
                "count": len(console_errors),
                "verdict": "pass" if not console_errors else "fail",
            }
            _json_dump(output_dir / "console_errors.json", console)
        except Exception:
            if console.get("verdict") == "not_run":
                console = {
                    "verdict": "fail",
                    "reason": "console evidence unavailable during cleanup",
                }
        page.close()

    _json_dump(output_dir / "button_matrix.json", matrix)
    try:
        final_run = _run_payload(args.base_url, run_id)
    except Exception as exc:
        final_run = {
            "fetch_error": f"{type(exc).__name__}: {exc}",
        }
    _json_dump(output_dir / "final_run.json", final_run)
    enabled_cases = [
        item for item in matrix if item.get("oracle_expected_enabled") is True
    ]
    disabled_cases = [
        item for item in matrix if item.get("oracle_expected_enabled") is False
    ]
    matrix_summary = {
        "required_total": 33,
        "required_enabled": 29,
        "required_disabled": 4,
        "cases_total": len(matrix),
        "enabled_cases": len(enabled_cases),
        "disabled_cases": len(disabled_cases),
        "passed": sum(item.get("verdict") == "pass" for item in matrix),
        "failed": sum(item.get("verdict") == "fail" for item in matrix),
        "not_run": sum(item.get("verdict") == "not_run" for item in matrix),
    }
    matrix_summary["verdict"] = (
        "pass"
        if matrix_summary
        == {
            "required_total": 33,
            "required_enabled": 29,
            "required_disabled": 4,
            "cases_total": 33,
            "enabled_cases": 29,
            "disabled_cases": 4,
            "passed": 33,
            "failed": 0,
            "not_run": 0,
        }
        else "fail"
    )
    matrix_summary["required_result"] = (
        "33/33"
        if matrix_summary["verdict"] == "pass"
        else f"{matrix_summary['passed']}/33"
    )
    long_press = {
        "verdict": "not_run",
        "reason": (
            "live holds at or beyond the 320 ms repeat gate are intentionally "
            "not sent; repeat behavior is unit-test-only"
        ),
        "repeatable_long_press_coverage": "unit_only",
    }
    camera_summary = {
        "required": 3,
        "cases": len(camera_cases),
        "passed": sum(item.get("verdict") == "pass" for item in camera_cases),
        "cameras": [item.get("camera") for item in camera_cases],
    }
    camera_summary["verdict"] = (
        "pass"
        if camera_summary["cases"] == 3
        and camera_summary["passed"] == 3
        and camera_summary["cameras"] == ["left_wrist", "right_wrist", "head"]
        else "fail"
    )
    sections = {
        "ready": ready_record,
        "initial_ui": initial_ui,
        "release_oracle": release_oracle,
        "collapse_expand": collapse,
        "button_matrix": matrix_summary,
        "live_long_press": long_press,
        "camera_tabs": camera_summary,
        "media": media,
        "visual": visual,
        "console": console,
        "sustain": sustain,
        "browser_execution": browser_error,
        "cleanup": cleanup,
    }
    section_verdicts = {
        name: value.get("verdict")
        for name, value in sections.items()
        if isinstance(value, dict)
    }
    required_section_verdicts = {
        name: verdict
        for name, verdict in section_verdicts.items()
        if name not in {"media", "live_long_press"}
    }
    live_action_media = media.get("live_action_video")
    media_not_run_is_unpackaged = (
        media.get("verdict") == "not_run"
        and isinstance(live_action_media, dict)
        and live_action_media.get("verdict") == "not_run"
        and "not packaged" in str(live_action_media.get("reason", ""))
    )
    media_acceptable = media.get("verdict") == "pass" or media_not_run_is_unpackaged
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "reference_sha256": REFERENCE_SHA256,
        "button_matrix": matrix_summary,
        "section_verdicts": section_verdicts,
        "acceptance_policy": {
            "required_sections": sorted(required_section_verdicts),
            "media_not_run_allowed_only_when_unpackaged": True,
            "media_acceptable": media_acceptable,
        },
        "halt_reason": halt_reason,
        "official_task_success": _raw_success_latched(final_run),
        "artifact_manifest": {"verdict": "pending"},
        "verdict": (
            "pass"
            if required_section_verdicts
            and all(value == "pass" for value in required_section_verdicts.values())
            and media_acceptable
            else "fail"
        ),
    }
    _json_dump(output_dir / "acceptance_summary.json", summary)
    preliminary_manifest = _artifact_manifest(
        output_dir=output_dir,
        sections=sections,
    )
    preliminary_ok = bool(
        preliminary_manifest["self_check"]["manifest_json_reloaded"]
        and preliminary_manifest["self_check"]["artifact_hashes_revalidated"]
        and not preliminary_manifest["self_check"]["errors"]
    )
    summary["artifact_manifest"] = {
        "verdict": "pass" if preliminary_ok else "fail",
        "artifact_count": preliminary_manifest["artifact_count"],
    }
    if not preliminary_ok:
        summary["verdict"] = "fail"
    _json_dump(output_dir / "acceptance_summary.json", summary)
    manifest = _artifact_manifest(output_dir=output_dir, sections=sections)
    manifest_ok = bool(
        manifest["self_check"]["manifest_json_reloaded"]
        and manifest["self_check"]["artifact_hashes_revalidated"]
        and not manifest["self_check"]["errors"]
    )
    if not manifest_ok:
        summary["artifact_manifest"] = {
            "verdict": "fail",
            "artifact_count": manifest["artifact_count"],
        }
        summary["verdict"] = "fail"
        _json_dump(output_dir / "acceptance_summary.json", summary)
        manifest = _artifact_manifest(output_dir=output_dir, sections=sections)
        manifest_ok = bool(
            manifest["self_check"]["manifest_json_reloaded"]
            and manifest["self_check"]["artifact_hashes_revalidated"]
            and not manifest["self_check"]["errors"]
        )
    return 0 if summary["verdict"] == "pass" and manifest_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
