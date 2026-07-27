"""Runner-owned continuation across independent BEHAVIOR planner cycles.

The planner backend is intentionally allowed to end one SDK thread with an
ordinary natural-language response. That response is not a BEHAVIOR terminal
fact. This module keeps the already-started runtime and toolkit alive, starts a
fresh planner thread, and stops only when the runtime/toolkit exposes one of the
five trusted terminal conditions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from robots.behavior.redaction import redact_text, redact_value
from robots.behavior.task_specs import BehaviorTaskSpec, get_task_spec
from rpent.planner.base import PlannerResult

CONTINUATION_USER_MESSAGE = """\
RUNNER CONTINUATION CYCLE

This is the same BEHAVIOR episode and the same task. The environment, toolkit,
attachment state, run nonce, attempt nonce, and all cumulative execution
budgets are unchanged.

The previous planner thread ending with natural-language text did not end the
episode. Do not infer state or choose an action from that thread's prose.
Begin this cycle with a fresh public `observe` tool call, ground the next action
only in current public evidence and tool results, and continue acting.

Do not produce a standalone final answer while execution can continue. Only the
runner's five trusted terminal conditions can end the episode.
"""

_SUMMED_STAT_KEYS = {
    "elapsed_s",
    "output_chars",
    "tool_calls",
    "total_input_tokens",
    "total_cached_input_tokens",
    "total_output_tokens",
    "total_reasoning_output_tokens",
    "total_cache_creation_input_tokens",
    "total_cache_read_input_tokens",
    "total_cost_usd",
}
_DEPRECATED_PI0_LIMIT_REASONS = frozenset(
    {
        "call_chunk_limit",
        "global_vla_chunk_budget_exhausted",
        "max_total_vla_chunks",
        "max_vla_chunks_per_call",
    }
)
_TRUSTED_HARD_EXECUTION_BUDGET_REASONS = frozenset(
    {
        "max_episode_steps",
        "max_tool_calls",
        "max_wall_clock_s",
    }
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            redact_value(payload),
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_cycle(
    *,
    output_dir: Path,
    cycle_index: int,
    result: PlannerResult,
    turns_charged: int,
) -> dict[str, Any]:
    """Archive planner-owned rolling files before the next cycle overwrites them."""

    cycle_dir = output_dir / "planner_cycles" / f"cycle_{cycle_index:03d}"
    cycle_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Any] = {}
    artifact_fields = {
        "output_path": "output.txt",
        "raw_stream_path": "raw_stream.jsonl",
        "last_message_path": "last_message.txt",
    }
    for stat_key, archive_name in artifact_fields.items():
        raw_path = result.stats.get(stat_key)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        source = Path(raw_path)
        artifact: dict[str, Any] = {
            "source": str(source),
            "archived": False,
        }
        if source.is_file():
            target = cycle_dir / archive_name
            shutil.copy2(source, target)
            artifact.update(
                {
                    "archived": True,
                    "path": str(target),
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        artifacts[stat_key] = artifact

    record = {
        "schema_version": 1,
        "cycle_index": int(cycle_index),
        "turns_reported": result.stats.get("turns_used"),
        "turns_charged": int(turns_charged),
        "finish": result.finish_result,
        "error": redact_text(result.error) if result.error else None,
        "stats": result.stats,
        "artifacts": artifacts,
    }
    _atomic_write_json(cycle_dir / "cycle_result.json", record)
    return {
        "cycle_index": int(cycle_index),
        "path": str(cycle_dir),
        "turns_charged": int(turns_charged),
        "error": record["error"],
        "artifacts": artifacts,
    }


def _trusted_termination(
    *,
    task_spec: BehaviorTaskSpec,
    toolkit_state: dict[str, Any],
    operator_stop: bool,
    turn_budget_exhausted: bool,
) -> tuple[str | None, list[str]]:
    """Select a terminal reason from structured runner/runtime facts only."""

    budget_reasons = [
        str(reason)
        for reason in toolkit_state.get("exhausted_budgets", [])
        if (
            isinstance(reason, str)
            and reason
            and reason not in _DEPRECATED_PI0_LIMIT_REASONS
            and reason in _TRUSTED_HARD_EXECUTION_BUDGET_REASONS
        )
    ]
    if turn_budget_exhausted:
        budget_reasons.append("max_turns")
    budget_reasons = list(dict.fromkeys(budget_reasons))

    if toolkit_state.get("raw_official_success_verified") is True:
        return "official_task_success", budget_reasons
    if operator_stop:
        return "operator_stop", budget_reasons
    if toolkit_state.get("unrecoverable_infrastructure_termination") is True:
        return "unrecoverable_infrastructure_termination", budget_reasons
    terminal_policy = task_spec.terminal_failure_policy
    if (
        terminal_policy is not None
        and toolkit_state.get("visual_terminal_failure_verified") is True
        and toolkit_state.get("visual_terminal_failure_reason")
        == terminal_policy.runner_reason
    ):
        return terminal_policy.runner_reason, budget_reasons
    if budget_reasons:
        return "hard_execution_budget_exhausted", budget_reasons
    return None, budget_reasons


def _runner_state(toolkit: Any, runtime: Any = None) -> dict[str, Any]:
    owner = runtime if runtime is not None else toolkit
    snapshot = getattr(owner, "runner_continuation_state", None)
    if not callable(snapshot):
        raise TypeError("BEHAVIOR toolkit must expose runner_continuation_state()")
    state = snapshot()
    if not isinstance(state, dict):
        raise TypeError("runner_continuation_state() must return a dictionary")
    return state


def _aggregate_stats(
    *,
    cycle_stats: list[dict[str, Any]],
    turns_used: int,
    cycle_archives: list[dict[str, Any]],
    termination_reason: str,
    exhausted_budgets: list[str],
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "cycles": len(cycle_stats),
        "turns_used": int(turns_used),
        "termination_reason": termination_reason,
        "exhausted_budgets": list(exhausted_budgets),
        "planner_cycles": cycle_archives,
        "cycle_stats": cycle_stats,
    }
    for stats in cycle_stats:
        backend = stats.get("backend")
        if backend is not None:
            aggregate["backend"] = backend
        for key in _SUMMED_STAT_KEYS:
            value = stats.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                aggregate[key] = aggregate.get(key, 0) + value
    # Runner charging is authoritative when a planner reports zero turns for
    # an immediate natural-language final.
    aggregate["turns_used"] = int(turns_used)
    return aggregate


def run_behavior_planner_continuation(
    *,
    task_name: str,
    planner: Any,
    system_prompt: str,
    initial_user_message: str,
    toolkit: Any,
    max_turns: int,
    output_dir: str | Path,
    runtime: Any = None,
) -> PlannerResult:
    """Run fresh planner threads until one trusted BEHAVIOR terminal fact exists."""

    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    task_spec = get_task_spec(task_name)

    output_dir = Path(output_dir)
    messages: list[dict[str, Any]] = []
    cycle_stats: list[dict[str, Any]] = []
    cycle_archives: list[dict[str, Any]] = []
    cycle_errors: list[dict[str, Any]] = []
    turns_used = 0
    cycle_index = 0
    last_result = PlannerResult()
    termination_reason: str | None = None
    exhausted_budgets: list[str] = []

    while termination_reason is None:
        state_before = _runner_state(toolkit, runtime)
        termination_reason, exhausted_budgets = _trusted_termination(
            task_spec=task_spec,
            toolkit_state=state_before,
            operator_stop=False,
            turn_budget_exhausted=turns_used >= max_turns,
        )
        if termination_reason is not None:
            break

        cycle_index += 1
        remaining_turns = max_turns - turns_used
        cycle_message = (
            initial_user_message
            if cycle_index == 1
            else f"{initial_user_message.rstrip()}\n\n{CONTINUATION_USER_MESSAGE}"
        )
        operator_stop = False
        try:
            last_result = planner.solve(
                system_prompt=system_prompt,
                user_message=cycle_message,
                toolkit=toolkit,
                max_turns=remaining_turns,
            )
        except KeyboardInterrupt:
            operator_stop = True
            last_result = PlannerResult()
        except Exception as error:  # noqa: BLE001 - a cycle error is not terminal
            last_result = PlannerResult(
                error=redact_text(f"{type(error).__name__}: {error}")
            )

        reported_turns = last_result.stats.get("turns_used", 0)
        if not isinstance(reported_turns, int) or isinstance(reported_turns, bool):
            reported_turns = 0
        turns_charged = max(1, reported_turns)
        turns_charged = min(turns_charged, remaining_turns)
        turns_used += turns_charged

        for message in last_result.messages:
            if isinstance(message, dict):
                messages.append({"planner_cycle": cycle_index, **message})
            else:
                messages.append(
                    {
                        "planner_cycle": cycle_index,
                        "role": "planner",
                        "content": str(message),
                    }
                )
        cycle_stats.append(dict(last_result.stats))
        if last_result.error:
            cycle_errors.append(
                {
                    "cycle_index": cycle_index,
                    "error": redact_text(last_result.error),
                }
            )
        cycle_archives.append(
            _archive_cycle(
                output_dir=output_dir,
                cycle_index=cycle_index,
                result=last_result,
                turns_charged=turns_charged,
            )
        )

        state_after = _runner_state(toolkit, runtime)
        termination_reason, exhausted_budgets = _trusted_termination(
            task_spec=task_spec,
            toolkit_state=state_after,
            operator_stop=operator_stop,
            turn_budget_exhausted=turns_used >= max_turns,
        )
        _atomic_write_json(
            output_dir / "planner_cycles" / "manifest.json",
            {
                "schema_version": 1,
                "task_name": task_spec.task_name,
                "cycles": cycle_archives,
                "turns_used": turns_used,
                "max_turns": max_turns,
                "termination_reason": termination_reason,
                "exhausted_budgets": exhausted_budgets,
            },
        )

    if termination_reason == "official_task_success":
        task_success: bool | None = True
    elif termination_reason == "hard_execution_budget_exhausted" or (
        task_spec.terminal_failure_policy is not None
        and termination_reason == task_spec.terminal_failure_policy.runner_reason
    ):
        task_success = False
    else:
        # Operator stops and infrastructure loss do not establish a task-level
        # failure. Preserve the unknown outcome rather than manufacturing false.
        task_success = None
    finish_result: dict[str, Any] = {
        "_finish": True,
        "task_name": task_spec.task_name,
        "runner_termination_reason": termination_reason,
        "task_success": task_success,
        "official_success_source": 'info["done"]["success"]',
        "planner_cycles": cycle_index,
        "turns_used": turns_used,
        "exhausted_budgets": exhausted_budgets,
    }
    if isinstance(last_result.finish_result, dict):
        finish_result["last_planner_finish"] = last_result.finish_result

    stats = _aggregate_stats(
        cycle_stats=cycle_stats,
        turns_used=turns_used,
        cycle_archives=cycle_archives,
        termination_reason=str(termination_reason),
        exhausted_budgets=exhausted_budgets,
    )
    if cycle_errors:
        stats["cycle_errors"] = cycle_errors

    terminal_error = None
    if termination_reason == "unrecoverable_infrastructure_termination":
        terminal_error = (
            redact_text(last_result.error)
            if last_result.error
            else "BEHAVIOR runtime reported an unrecoverable infrastructure termination"
        )
    return PlannerResult(
        finish_result=finish_result,
        messages=messages,
        stats=stats,
        error=terminal_error,
    )


__all__ = [
    "CONTINUATION_USER_MESSAGE",
    "run_behavior_planner_continuation",
]
