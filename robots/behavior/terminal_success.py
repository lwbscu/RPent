"""Validation for immutable raw BEHAVIOR success receipts.

Official task success is latched from ``info["done"]["success"]`` at the first
successful simulator step. Images, render state, later predicate values, and
video sealing are best-effort artifacts and never participate in this gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class TerminalReceiptValidation:
    """Result of validating one output-bound official-success receipt."""

    valid: bool
    terminal_image_path: Path | None = None
    reason: str | None = None


def _invalid(reason: str) -> TerminalReceiptValidation:
    return TerminalReceiptValidation(valid=False, reason=reason)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _receipt_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    direct = result.get("official_success_receipt")
    if isinstance(direct, dict):
        return direct
    monitor = result.get("pi0_nav_pick_monitor")
    if isinstance(monitor, dict) and isinstance(
        monitor.get("official_success_receipt"), dict
    ):
        return monitor["official_success_receipt"]
    info = result.get("info")
    if isinstance(info, dict):
        runtime = info.get("_rpent")
        monitor = (
            runtime.get("pi0_nav_pick_monitor") if isinstance(runtime, dict) else None
        )
        if isinstance(monitor, dict) and isinstance(
            monitor.get("official_success_receipt"), dict
        ):
            return monitor["official_success_receipt"]
    return None


def _exact_bool_at(record: dict[str, Any], path: tuple[str, ...]) -> bool | None:
    value: Any = record
    for field in path:
        if not isinstance(value, dict) or field not in value:
            return None
        value = value[field]
    return value if type(value) is bool else None


def _has_path(record: dict[str, Any], path: tuple[str, ...]) -> bool:
    value: Any = record
    for field in path:
        if not isinstance(value, dict) or field not in value:
            return False
        value = value[field]
    return True


def _record_declares_termination(record: dict[str, Any]) -> bool:
    for field in ("terminated", "truncated", "terminal"):
        if record.get(field) is True:
            return True
    if isinstance(record.get("runner_termination_reason"), str):
        if record["runner_termination_reason"]:
            return True
    done = record.get("info_done")
    if not isinstance(done, dict):
        return False
    for field in ("terminated", "truncated", "done"):
        if done.get(field) is True:
            return True
    conditions = done.get("termination_conditions")
    if not isinstance(conditions, dict):
        return False
    return any(value is True for value in conditions.values())


def summarize_action_trace_success(
    action_trace_bytes: bytes,
) -> dict[str, Any] | None:
    """Summarize immutable raw-success evidence from an action trace.

    Current traces use the exact boolean field ``info_done.success``.  The
    Parse and ordering anomalies are recorded as notes; once a raw true is
    observed they never revoke task success.
    """

    action_trace_sha256 = hashlib.sha256(action_trace_bytes).hexdigest()
    records: list[tuple[int, dict[str, Any], int | None]] = []
    malformed_lines = 0
    missing_step_records = 0
    duplicate_step_values = 0
    non_monotonic_step_records = 0
    invalid_current_success_values = 0
    seen_steps: set[int] = set()
    previous_step: int | None = None
    last_trace_step: int | None = None
    for line_number, line in enumerate(action_trace_bytes.splitlines(), start=1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_lines += 1
            continue
        if not isinstance(record, dict):
            malformed_lines += 1
            continue
        raw_step = record.get("step")
        step = (
            raw_step
            if isinstance(raw_step, int)
            and not isinstance(raw_step, bool)
            and raw_step >= 0
            else None
        )
        is_action_record = (
            record.get("event") == "step"
            or "step" in record
            or _has_path(record, ("info_done", "success"))
        )
        if step is None and is_action_record:
            missing_step_records += 1
        elif step is not None:
            if step in seen_steps:
                duplicate_step_values += 1
            if previous_step is not None and step < previous_step:
                non_monotonic_step_records += 1
            seen_steps.add(step)
            previous_step = step
            last_trace_step = step
        if _has_path(record, ("info_done", "success")):
            if _exact_bool_at(record, ("info_done", "success")) is None:
                invalid_current_success_values += 1
        records.append((line_number, record, step))

    selected_path = ("info_done", "success")
    field_path = "info_done.success"
    observations = [
        (line_number, step, value, record)
        for line_number, record, step in records
        if (value := _exact_bool_at(record, selected_path)) is not None
    ]
    if not any(value is True for _, _, value, _ in observations):
        return None

    first_true_index = next(
        index for index, (_, _, value, _) in enumerate(observations) if value is True
    )
    first_line, first_success_step, _, _ = observations[first_true_index]
    first_interval_end = first_success_step
    interval_closed = False
    success_count = 0
    last_success_step: int | None = None
    success_record_missing_step = False
    success_later_reverted = False
    for index, (_, step, value, _) in enumerate(observations):
        if value is True:
            success_count += 1
            last_success_step = step
            success_record_missing_step = success_record_missing_step or step is None
            if index >= first_true_index and not interval_closed:
                first_interval_end = step
        elif index > first_true_index:
            success_later_reverted = True
            interval_closed = True

    termination_before_success = any(
        line_number < first_line and _record_declares_termination(record)
        for line_number, record, _ in records
    )
    notes: list[str] = []
    if malformed_lines:
        notes.append(f"malformed_json_lines={malformed_lines}")
    if missing_step_records:
        notes.append(f"records_missing_step={missing_step_records}")
    if duplicate_step_values:
        notes.append(f"duplicate_step_values={duplicate_step_values}")
    if non_monotonic_step_records:
        notes.append(f"non_monotonic_step_records={non_monotonic_step_records}")
    if invalid_current_success_values:
        notes.append(
            f"invalid_info_done_success_values={invalid_current_success_values}"
        )
    if success_record_missing_step:
        notes.append("success_record_missing_step")
    if termination_before_success:
        notes.append("termination_observed_before_first_success")

    final_trace_success = None
    for _, record, step in reversed(records):
        if step is not None or record.get("event") == "step":
            final_trace_success = _exact_bool_at(record, selected_path)
            break

    return {
        "source": "behavior_action_trace",
        "field_path": field_path,
        "first_success_step": first_success_step,
        "success_interval": [first_success_step, first_interval_end],
        "success_count": success_count,
        "success_later_reverted": success_later_reverted,
        "last_success_step": last_success_step,
        "last_trace_step": last_trace_step,
        "final_trace_success": final_trace_success,
        "action_trace_sha256": action_trace_sha256,
        "receipt_sha256": None,
        "notes": notes,
    }


def _contained_receipt(
    output_dir: Path, value: Any
) -> tuple[Path | None, dict[str, Any] | None]:
    if not isinstance(value, str) or not value:
        return None, None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = output_dir / candidate
    if candidate.is_symlink():
        return None, None
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(output_dir.resolve())
        loaded = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None, None
    return resolved, loaded if isinstance(loaded, dict) else None


def validate_terminal_success_receipt(
    *,
    tool_name: str,
    step: Any,
    result: Any,
    output_dir: str | Path,
) -> TerminalReceiptValidation:
    """Validate raw official success without terminal-hold or image gates."""

    del tool_name
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        return _invalid("invalid trace step")
    if not isinstance(result, dict) or result.get("task_success") is not True:
        return _invalid("result lacks official task success")
    receipt = _receipt_from_result(result)
    if not isinstance(receipt, dict):
        receipt_path, receipt = _contained_receipt(
            Path(output_dir).resolve(),
            result.get("official_success_receipt_path"),
        )
        if receipt_path is None or receipt is None:
            return _invalid("missing official success receipt")
    if receipt.get("source") != 'info["done"]["success"]':
        return _invalid("official success source mismatch")
    raw_done = receipt.get("raw_done")
    if not isinstance(raw_done, dict) or raw_done.get("success") is not True:
        return _invalid("receipt does not contain raw done.success=true")
    for field in ("run_nonce", "attempt_nonce", "env_step", "receipt_sha256"):
        if field not in receipt:
            return _invalid(f"receipt omitted {field}")
    if (
        not isinstance(receipt["run_nonce"], str)
        or not receipt["run_nonce"]
        or not isinstance(receipt["attempt_nonce"], str)
        or not receipt["attempt_nonce"]
        or not isinstance(receipt["env_step"], int)
        or isinstance(receipt["env_step"], bool)
        or receipt["env_step"] < 0
    ):
        return _invalid("receipt lineage is invalid")
    claimed = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    actual = hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
    if claimed != actual:
        return _invalid("official success receipt digest mismatch")
    return TerminalReceiptValidation(valid=True)


def green_center_marker_visible(
    source: bytes | bytearray | Path | str | BinaryIO,
) -> bool:
    """Legacy diagnostic retained for readers; never an official-success gate."""

    del source
    return False


__all__ = [
    "TerminalReceiptValidation",
    "green_center_marker_visible",
    "summarize_action_trace_success",
    "validate_terminal_success_receipt",
]
