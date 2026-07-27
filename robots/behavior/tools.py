"""Primitive handlers for the peer-capability BEHAVIOR toolkit."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from robots.behavior.run_manifest import PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION
from robots.behavior.schemas import (
    DEFAULT_ACTION_CHUNK,
    ENV_ACTION_SEGMENTS,
    FRAME_REVIEW_ASSESSMENTS,
    POLICY_STATE_SEGMENTS,
    extract_policy_state,
    segment_ranges,
    validate_action_chunk,
    validate_dashboard_manual_command,
    validate_relative_navigation_motion,
)
from robots.behavior.task_specs import (
    TURNING_ON_RADIO_TASK_SPEC,
    BehaviorTaskSpec,
    TerminalFailurePolicy,
    get_task_spec,
)
from rpent.utils.logging import get_logger, get_output_dir

logger = get_logger("behavior")

_PI0_NAV_PICK_MONITOR_FIELDS = (
    "executed_steps",
    "handoff_env_steps",
    "total_env_steps",
    "local_grasp_success",
    "action_source",
    "controller_state",
    "vla_actions_enabled",
    "official_success_receipt",
)
_PI0_NAV_PICK_OPTIONAL_MONITOR_FIELDS = (
    "capability",
    "visual_review",
    "stop_reason",
    "official_success_first_observed_this_chunk",
    "soft_success_observed",
    "terminal_classification",
)


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _as_bool(value: Any) -> bool:
    array = np.asarray(_jsonable(value))
    return bool(array.any()) if array.size else False


def _validated_attachment_count(capability: Any) -> int:
    """Validate the public, physical per-hand attachment capability."""

    if not isinstance(capability, dict):
        raise RuntimeError("runtime attachment capability must be a mapping")
    attachments = capability.get("attachments")
    if not isinstance(attachments, dict):
        raise RuntimeError("runtime attachment capability is missing attachments")
    if attachments.get("available") is not True:
        raise RuntimeError("runtime attachment capability is unavailable")
    if attachments.get("conflict") is not False:
        raise RuntimeError("runtime attachment capability reports a conflict")
    count = attachments.get("count")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise RuntimeError("runtime attachment count must be an integer")
    count = int(count)
    if count not in {0, 1, 2}:
        raise RuntimeError(f"invalid runtime attachment count: {count!r}")
    by_hand = attachments.get("by_hand")
    if not isinstance(by_hand, dict) or set(by_hand) != {"left", "right"}:
        raise RuntimeError(
            "runtime attachment capability must report left and right hands"
        )
    attached_count = 0
    for hand in ("left", "right"):
        state = by_hand.get(hand)
        if not isinstance(state, dict) or not isinstance(state.get("attached"), bool):
            raise RuntimeError(
                f"runtime attachment capability for {hand} must report attached"
            )
        attached_count += int(state["attached"])
    if attached_count != count:
        raise RuntimeError(
            "runtime attachment count disagrees with per-hand attachment state"
        )
    return count


def official_task_success(info: Any) -> bool:
    """Read only the official raw BEHAVIOR success bit."""
    if not isinstance(info, dict):
        return False
    done = info.get("done")
    if not isinstance(done, dict):
        return False
    value = done.get("success", False)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _public_info_summary(info: Any) -> Any:
    """Expose only official stop/accounting fields, never monitor internals."""
    if not isinstance(info, dict):
        return {}
    public: dict[str, Any] = {}
    if isinstance(info.get("done"), dict):
        public["done"] = _jsonable(info["done"])
    runtime = info.get("_rpent")
    if isinstance(runtime, dict):
        allowed_runtime_keys = {
            "total_env_steps",
            "global_env_steps",
            "run_nonce",
            "attempt_index",
            "attempt_nonce",
            "official_success_receipt",
        }
        public["_rpent"] = {
            key: _jsonable(runtime[key])
            for key in allowed_runtime_keys
            if key in runtime
        }
    return public


def _official_success_receipt_from_info(info: Any) -> dict[str, Any] | None:
    """Extract only the immutable raw-success receipt from runtime info."""

    if not isinstance(info, dict):
        return None
    runtime = info.get("_rpent")
    if not isinstance(runtime, dict):
        return None
    direct = runtime.get("official_success_receipt")
    if isinstance(direct, dict):
        return _jsonable(direct)
    monitor = runtime.get("pi0_nav_pick_monitor")
    if isinstance(monitor, dict) and isinstance(
        monitor.get("official_success_receipt"), dict
    ):
        return _jsonable(monitor["official_success_receipt"])
    return None


def _validated_official_success_receipt(value: Any) -> dict[str, Any] | None:
    """Validate the raw-success identity carried by the trusted env runtime."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("official success receipt must be a mapping")
    if value.get("source") != 'info["done"]["success"]':
        raise RuntimeError("official success receipt has an invalid source")
    raw_done = value.get("raw_done")
    if not isinstance(raw_done, dict) or raw_done.get("success") is not True:
        raise RuntimeError("official success receipt lacks raw done.success=true")
    return _jsonable(value)


_PRIVATE_RESULT_KEYS = {
    "activity_instance_id",
    "ground_truth",
    "gt",
    "hidden_state",
    "native_instance",
    "private_environment_metadata",
    "simulator_state",
    "stage",
    "suggested_next_action",
    "suggested_next_tool",
    "action_source",
    "call_dir",
    "contact_context_id",
    "handoff_state",
    "held_hand",
    "last_info",
    "last_reward",
    "per_hand",
    "raw_final_info_path",
    "result_path",
    "states_path",
    "terminal_success_evidence",
    "usable_post_pick_saved",
    "validator_trace_path",
    "video_path",
    "visual_review",
    "visual_review_required",
    "vla_disable_confirmation",
    "vla_enable_confirmation",
    "vla_endpoint",
    "vla_health_after_disable",
    "vla_pid",
    "vla_prepare_confirmation",
    "workflow_complete",
}
_TERMINAL_BUDGET_REASONS = {
    "global_env_step_budget_exhausted",
    "global_env_step_budget_insufficient_for_full_chunk",
    "global_tool_call_budget_exhausted",
    "global_wall_clock_budget_exhausted",
}
_PUBLIC_IMAGE_BYTE_FIELDS = frozenset(
    {
        "_image_bytes",
        "_depth_image_bytes",
        "_image_cam_bytes",
        "_image_wrist_bytes",
    }
)


def _sanitize_public_result(value: Any) -> Any:
    """Remove strategy hints and privileged diagnostics from public results."""

    if isinstance(value, dict):
        public: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if (
                lowered in _PRIVATE_RESULT_KEYS
                or lowered.startswith(("ground_truth_", "gt_", "private_"))
                or lowered.endswith("_ground_truth")
            ):
                continue
            if lowered in _PUBLIC_IMAGE_BYTE_FIELDS:
                if item is None:
                    public[str(key)] = None
                    continue
                if not isinstance(item, (bytes, bytearray, memoryview)):
                    raise TypeError(
                        f"{key} must contain bytes-like BEHAVIOR image data"
                    )
                public[str(key)] = bytes(item)
                continue
            public[str(key)] = _sanitize_public_result(item)
        return public
    if isinstance(value, list):
        return [_sanitize_public_result(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_public_result(item) for item in value]
    return _jsonable(value)


_PI0_PUBLIC_RESULT_KEYS = {
    "_finish",
    "runner_termination_reason",
    "name",
    "primitive_success",
    "task_success",
    "official_success_source",
    "official_success_receipt",
    "terminated",
    "truncated",
    "stop_reason",
    "requested_chunks",
    "exact_requested_chunks_completed",
    "chunks_used",
    "global_vla_chunks",
    "global_vla_invocations",
    "full_chunks_executed",
    "env_steps_used",
    "vla_env_steps_used",
    "handoff_env_steps_used",
    "total_env_steps",
    "max_episode_steps",
    "action_horizon",
    "required_action_shape",
    "elapsed_s",
    "global_elapsed_wall_clock_s",
    "attempt_index",
    "attempt_nonce",
    "run_nonce",
    "vla_call_index",
    "failed_preconditions",
    "error",
}


def _sanitize_pi0_public_result(value: dict[str, Any]) -> dict[str, Any]:
    """Return only the bounded skill envelope; keep audit detail on disk."""

    sanitized = _sanitize_public_result(value)
    if not isinstance(sanitized, dict):
        raise RuntimeError("pi0_nav_pick public result is not a mapping")
    return {key: sanitized[key] for key in _PI0_PUBLIC_RESULT_KEYS if key in sanitized}


def _runner_should_terminate(
    *,
    tool_name: str,
    task_success: bool,
    stop_reason: str,
    terminal_failure_receipt: Any = None,
    terminal_failure_policy: TerminalFailurePolicy | None = None,
) -> bool:
    """Return the runner-owned end-of-attempt decision."""

    visual_terminal_failure = bool(
        tool_name == "save_robot_state_checkpoint"
        and terminal_failure_policy is not None
        and stop_reason == terminal_failure_policy.condition
        and isinstance(terminal_failure_receipt, dict)
        and terminal_failure_receipt.get("source") == "llm_fresh_visual_observation"
        and terminal_failure_receipt.get("condition")
        == terminal_failure_policy.condition
        and terminal_failure_receipt.get("task_success") is False
    )
    return bool(
        task_success
        or visual_terminal_failure
        or stop_reason in _TERMINAL_BUDGET_REASONS
        or stop_reason.endswith("_budget_exhausted")
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)


class BehaviorPrimitives:
    """Only the handlers registered by :class:`BehaviorToolkit`."""

    def __init__(
        self,
        *,
        env: Any = None,
        model: Any = None,
        max_episode_steps: int | None = None,
        output_dir: str | Path | None = None,
        video_path: str | Path | None = None,
        action_horizon: int = DEFAULT_ACTION_CHUNK,
        initial_observation: dict[str, Any] | None = None,
        initial_info: Any = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        behavior_phase: str = "eval",
        task_name: str = TURNING_ON_RADIO_TASK_SPEC.task_name,
        public_seed: int = 0,
        initial_attempt_index: int = 1,
        job_id: str | None = None,
        max_tool_calls: int | None = 350,
        max_wall_clock_s: float = 86400.0,
        pure_vla_baseline: bool = False,
    ) -> None:
        self.env = env
        self.model = model
        self.max_episode_steps = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.action_horizon = int(action_horizon)
        configured_output = Path(output_dir) if output_dir else get_output_dir()
        # Planner-only construction does not need artifacts and can occur in
        # unit tests before logging initializes the process output directory.
        self.output_dir = configured_output or Path.cwd()
        self.video_path = (
            Path(video_path) if video_path else self.output_dir / "episode.mp4"
        )
        self._current_observation = initial_observation
        self._current_info = initial_info
        self._progress_callback = progress_callback
        self.behavior_phase = str(behavior_phase)
        if self.behavior_phase not in {"explore", "eval"}:
            raise ValueError("behavior_phase must be 'explore' or 'eval'")
        self.task_spec = get_task_spec(str(task_name))
        self.task_name = self.task_spec.task_name
        self.public_seed = int(public_seed)
        self.task_spec.instance_for_public_seed(self.public_seed)
        self.job_id = str(job_id) if job_id is not None else None
        initial_rpent = (
            initial_info.get("_rpent") if isinstance(initial_info, dict) else None
        )
        reported_attempt_index = (
            initial_rpent.get("attempt_index")
            if isinstance(initial_rpent, dict)
            else None
        )
        self.attempt_index = int(
            initial_attempt_index
            if self.job_id is not None
            else reported_attempt_index
            if reported_attempt_index is not None
            else initial_attempt_index
        )
        if self.attempt_index < 1:
            raise ValueError("initial_attempt_index must be at least 1")
        reported_attempt_nonce = (
            initial_rpent.get("attempt_nonce")
            if isinstance(initial_rpent, dict)
            else None
        )
        reported_run_nonce = (
            initial_rpent.get("run_nonce") if isinstance(initial_rpent, dict) else None
        )
        self.attempt_nonce = (
            str(reported_attempt_nonce)
            if isinstance(reported_attempt_nonce, str) and reported_attempt_nonce
            else secrets.token_hex(16)
        )
        self.run_nonce = (
            str(reported_run_nonce)
            if isinstance(reported_run_nonce, str) and reported_run_nonce
            else secrets.token_hex(16)
        )
        if not isinstance(pure_vla_baseline, bool):
            raise TypeError("pure_vla_baseline must be boolean")
        if max_tool_calls is None and not pure_vla_baseline:
            raise ValueError(
                "max_tool_calls=None is restricted to the pure VLA baseline"
            )
        self.pure_vla_baseline = pure_vla_baseline
        self.max_tool_calls = None if max_tool_calls is None else int(max_tool_calls)
        self.max_wall_clock_s = float(max_wall_clock_s)
        if self.max_tool_calls is not None and self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        if not np.isfinite(self.max_wall_clock_s) or self.max_wall_clock_s <= 0:
            raise ValueError("max_wall_clock_s must be finite and positive")
        self.started_monotonic = time.monotonic()
        self._global_vla_chunks = 0
        self._vla_call_index = 0
        # Attempt-local audit counter. A fresh Agent invocation owns each
        # outer-harness Explore attempt.
        self._vla_invocations = 0
        self.last_result: dict[str, Any] | None = None

    @property
    def elapsed_wall_clock_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_monotonic)

    def _resolved_task_spec(self) -> BehaviorTaskSpec:
        """Return the configured task spec, with legacy Radio compatibility."""

        task_spec = getattr(self, "task_spec", None)
        if task_spec is not None:
            return task_spec
        return get_task_spec(
            str(getattr(self, "task_name", TURNING_ON_RADIO_TASK_SPEC.task_name))
        )

    @property
    def total_env_steps(self) -> int:
        value = getattr(self.env, "total_env_steps", None)
        if value is not None:
            return int(value)
        info = self._current_info
        rpent = info.get("_rpent") if isinstance(info, dict) else None
        return int(rpent.get("total_env_steps", 0)) if isinstance(rpent, dict) else 0

    def _attempt_root(self) -> Path:
        if self.job_id is not None:
            return self.output_dir
        tag = self.task_spec.tag(self.public_seed)
        return self.output_dir / "attempts" / tag / f"attempt_{self.attempt_index:03d}"

    def failed_preconditions(self, name: str, input_dict: dict[str, Any]) -> list[str]:
        """Ask the env for read-only runtime facts before any physical handler."""

        guard = getattr(self.env, "guard_tool_call", None)
        if not callable(guard):
            return []
        result = guard(name=name, input_dict=dict(input_dict))
        if not isinstance(result, dict):
            return ["runtime_guard_invalid_response"]
        failed = result.get("failed_preconditions", [])
        if not isinstance(failed, list) or not all(
            isinstance(item, str) and item for item in failed
        ):
            return ["runtime_guard_invalid_response"]
        return list(failed)

    def observe(
        self,
        *,
        camera: str,
        frame_review: dict[str, Any] | None = None,
        depth_probe: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if camera not in {"head", "left_wrist", "right_wrist"}:
            raise ValueError("camera must be head, left_wrist, or right_wrist")
        if frame_review is not None and depth_probe is not None:
            raise ValueError("frame_review and depth_probe are mutually exclusive")
        if frame_review is not None:
            if not isinstance(frame_review, dict):
                raise ValueError("frame_review must be an object")
            if set(frame_review) != {"frame_id", "assessment"}:
                raise ValueError(
                    "frame_review requires exactly frame_id and assessment"
                )
            frame_id = frame_review["frame_id"]
            if not isinstance(frame_id, str) or not frame_id.strip():
                raise ValueError("frame_review.frame_id must be a non-empty string")
            assessment = frame_review["assessment"]
            if assessment not in FRAME_REVIEW_ASSESSMENTS:
                raise ValueError(
                    "frame_review.assessment must be "
                    "target_bearing_surface_confirmed, "
                    "opposite_surface_confirmed, or side_or_indeterminate"
                )
            frame_review = {
                "frame_id": frame_id.strip(),
                "assessment": assessment,
            }
        if depth_probe is not None:
            if not isinstance(depth_probe, dict):
                raise ValueError("depth_probe must be an object")
            required_keys = {
                "frame_id",
                "u",
                "v",
                "depth_window_px",
                "assessment",
            }
            if set(depth_probe) != required_keys:
                raise ValueError(
                    "depth_probe requires exactly frame_id, u, v, "
                    "depth_window_px, and assessment"
                )
            frame_id = depth_probe["frame_id"]
            if not isinstance(frame_id, str) or not frame_id.strip():
                raise ValueError("depth_probe.frame_id must be a non-empty string")
            normalized_probe: dict[str, Any] = {
                "frame_id": frame_id.strip(),
                "assessment": depth_probe["assessment"],
            }
            for field in ("u", "v"):
                value = depth_probe[field]
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise ValueError(
                        f"depth_probe.{field} must be an integer pixel coordinate"
                    )
                normalized_probe[field] = int(value)
            depth_window_px = depth_probe["depth_window_px"]
            if (
                isinstance(depth_window_px, bool)
                or not isinstance(depth_window_px, (int, np.integer))
                or int(depth_window_px) < 1
                or int(depth_window_px) > 31
            ):
                raise ValueError(
                    "depth_probe.depth_window_px must be an integer in [1,31]"
                )
            normalized_probe["depth_window_px"] = int(depth_window_px)
            if normalized_probe["assessment"] != "target_point_visually_confirmed":
                raise ValueError(
                    "depth_probe.assessment must be target_point_visually_confirmed"
                )
            depth_probe = normalized_probe
        observe_kwargs: dict[str, Any] = {"camera": camera}
        if frame_review is not None:
            observe_kwargs["frame_review"] = frame_review
        if depth_probe is not None:
            observe_kwargs["depth_probe"] = depth_probe
        return self._public_checkpoint_result(
            name="observe",
            result=self._env_method("observe")(**observe_kwargs),
        )

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        """Return internal manual-control capabilities without changing tools."""

        result = self._env_method("dashboard_control_capabilities")()
        if not isinstance(result, dict):
            raise RuntimeError(
                "dashboard_control_capabilities returned a non-mapping result"
            )
        return result

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
    ) -> dict[str, Any]:
        """Execute one fixed-size Dashboard command outside the LLM tool surface."""

        command = validate_dashboard_manual_command(
            target=target,
            action=action,
            camera=camera,
        )
        result = self._env_method("dashboard_manual_command")(**command)
        if not isinstance(result, dict):
            raise RuntimeError("dashboard_manual_command returned a non-mapping result")
        frames = result.get("_frames_bytes")
        capture_complete = bool(
            isinstance(frames, dict)
            and set(frames) == {"head", "left_wrist", "right_wrist"}
            and all(isinstance(value, bytes) for value in frames.values())
        )
        receipt = result.get("official_success_receipt")
        raw_done = receipt.get("raw_done") if isinstance(receipt, dict) else None
        success_without_capture = bool(
            result.get("task_success") is True
            and isinstance(receipt, dict)
            and receipt.get("source") == 'info["done"]["success"]'
            and isinstance(raw_done, dict)
            and raw_done.get("success") is True
            and isinstance(receipt.get("receipt_sha256"), str)
            and bool(receipt["receipt_sha256"])
            and isinstance(result.get("capture_error"), str)
            and bool(result["capture_error"])
        )
        if not capture_complete and not success_without_capture:
            raise RuntimeError(
                "dashboard_manual_command omitted the atomic three-camera capture"
            )
        return result

    def pixel_to_world(
        self,
        *,
        camera: str,
        frame_id: str,
        u: int,
        v: int,
        depth_window_px: int = 7,
    ) -> dict[str, Any]:
        if camera not in {"head", "left_wrist", "right_wrist"}:
            raise ValueError("camera must be head, left_wrist, or right_wrist")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")
        for field, value in (("u", u), ("v", v)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{field} must be an integer pixel coordinate")
        if (
            isinstance(depth_window_px, bool)
            or not isinstance(depth_window_px, (int, np.integer))
            or int(depth_window_px) < 1
            or int(depth_window_px) > 31
        ):
            raise ValueError("depth_window_px must be an integer in [1,31]")
        return self._public_checkpoint_result(
            name="pixel_to_world",
            result=self._env_method("pixel_to_world")(
                camera=camera,
                frame_id=frame_id,
                u=u,
                v=v,
                depth_window_px=depth_window_px,
            ),
        )

    def _env_method(self, name: str) -> Callable[..., dict[str, Any]]:
        if self.env is None:
            raise RuntimeError(f"{name} requires an active BEHAVIOR env")
        method = getattr(self.env, name, None)
        if not callable(method):
            raise RuntimeError(f"env does not implement {name}")
        return method

    def _public_checkpoint_result(
        self,
        *,
        name: str,
        result: Any,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise RuntimeError(f"{name} returned a non-mapping result")
        public = _sanitize_public_result(result)
        if not isinstance(public, dict):
            raise RuntimeError(f"{name} public result is not a mapping")
        for field in ("primitive_success", "task_success", "stop_reason"):
            if field not in public:
                raise RuntimeError(f"{name} result omitted {field}")
        if not isinstance(public["primitive_success"], bool):
            raise RuntimeError(f"{name} primitive_success must be boolean")
        if not isinstance(public["task_success"], bool):
            raise RuntimeError(f"{name} task_success must be boolean")
        if not isinstance(public["stop_reason"], str) or not public["stop_reason"]:
            raise RuntimeError(f"{name} stop_reason must be non-empty")
        info = public.get("info")
        if isinstance(info, dict):
            official = official_task_success(info)
            if public["task_success"] is not official:
                raise RuntimeError(
                    f"{name} task_success disagrees with info.done.success"
                )
        elif public.get("official_success_source") != 'info["done"]["success"]':
            raise RuntimeError(
                f"{name} result lacks verifiable official success provenance"
            )
        task_success = bool(public["task_success"])
        stop_reason = str(public["stop_reason"])
        task_spec = self._resolved_task_spec()
        runner_terminate = _runner_should_terminate(
            tool_name=name,
            task_success=task_success,
            stop_reason=stop_reason,
            terminal_failure_receipt=public.get("terminal_failure_receipt"),
            terminal_failure_policy=task_spec.terminal_failure_policy,
        )
        visual_terminal_failure = bool(
            runner_terminate
            and not task_success
            and name == "save_robot_state_checkpoint"
            and task_spec.terminal_failure_policy is not None
            and stop_reason == task_spec.terminal_failure_policy.condition
            and isinstance(public.get("terminal_failure_receipt"), dict)
        )
        public.update(
            {
                "_finish": runner_terminate,
                "runner_termination_reason": (
                    "official_task_success"
                    if task_success
                    else task_spec.terminal_failure_policy.runner_reason
                    if visual_terminal_failure
                    else "attempt_budget_exhausted"
                    if runner_terminate
                    else None
                ),
                "name": name,
                "official_success_source": 'info["done"]["success"]',
                "failed_preconditions": list(public.get("failed_preconditions") or []),
                "invalidated_receipts": list(public.get("invalidated_receipts") or []),
                "new_receipts": list(public.get("new_receipts") or []),
            }
        )
        self.last_result = public
        return public

    @staticmethod
    def _analytic_hand(hand: str) -> str:
        if not isinstance(hand, str) or hand not in {"left", "right"}:
            raise ValueError("hand must be 'left' or 'right'")
        return hand

    @classmethod
    def _validated_visual_hand_check(
        cls,
        *,
        hand: str,
        visual_hand_check: Any,
    ) -> dict[str, str]:
        hand = cls._analytic_hand(hand)
        if not isinstance(visual_hand_check, dict):
            raise ValueError("visual_hand_check must be an object")
        required = {"camera", "frame_id", "selected_hand", "assessment"}
        if set(visual_hand_check) != required:
            raise ValueError(
                "visual_hand_check requires exactly camera, frame_id, "
                "selected_hand, and assessment"
            )
        if visual_hand_check["camera"] != "head":
            raise ValueError("visual_hand_check.camera must be 'head'")
        frame_id = visual_hand_check["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("visual_hand_check.frame_id must be a non-empty string")
        selected_hand = visual_hand_check["selected_hand"]
        if not isinstance(selected_hand, str) or selected_hand not in {
            "left",
            "right",
        }:
            raise ValueError(
                "visual_hand_check.selected_hand must be 'left' or 'right'"
            )
        if visual_hand_check["assessment"] != "selected_hand_visually_confirmed":
            raise ValueError(
                "visual_hand_check.assessment must be "
                "'selected_hand_visually_confirmed'"
            )
        if selected_hand != hand:
            raise ValueError("hand must equal visual_hand_check.selected_hand")
        return {
            "camera": "head",
            "frame_id": frame_id.strip(),
            "selected_hand": selected_hand,
            "assessment": "selected_hand_visually_confirmed",
        }

    @classmethod
    def _validated_release_visual_check(
        cls,
        *,
        hand: str,
        release_visual_check: Any,
    ) -> dict[str, str] | None:
        hand = cls._analytic_hand(hand)
        if release_visual_check is None:
            return None
        if not isinstance(release_visual_check, dict):
            raise ValueError("release_visual_check must be an object")
        required = {"camera", "frame_id", "selected_hand", "assessment"}
        if set(release_visual_check) != required:
            raise ValueError(
                "release_visual_check requires exactly camera, frame_id, "
                "selected_hand, and assessment"
            )
        if release_visual_check["camera"] != "head":
            raise ValueError("release_visual_check.camera must be 'head'")
        frame_id = release_visual_check["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("release_visual_check.frame_id must be a non-empty string")
        selected_hand = release_visual_check["selected_hand"]
        if selected_hand not in {"left", "right"}:
            raise ValueError(
                "release_visual_check.selected_hand must be 'left' or 'right'"
            )
        if selected_hand != hand:
            raise ValueError("hand must equal release_visual_check.selected_hand")
        assessment = release_visual_check["assessment"]
        if assessment != "attached_object_fully_inside_receptacle_opening":
            raise ValueError(
                "release_visual_check.assessment must be "
                "'attached_object_fully_inside_receptacle_opening'"
            )
        return {
            "camera": "head",
            "frame_id": frame_id.strip(),
            "selected_hand": selected_hand,
            "assessment": "attached_object_fully_inside_receptacle_opening",
        }

    @staticmethod
    def _positive_number(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be finite and positive")
        number = float(value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    @staticmethod
    def _validated_navigation_visual_check(
        navigation_visual_check: Any,
    ) -> dict[str, str]:
        if not isinstance(navigation_visual_check, dict):
            raise ValueError("navigation_visual_check must be an object")
        required = {"camera", "frame_id", "assessment"}
        if set(navigation_visual_check) != required:
            raise ValueError(
                "navigation_visual_check requires exactly camera, frame_id, "
                "and assessment"
            )
        if navigation_visual_check["camera"] != "head":
            raise ValueError("navigation_visual_check.camera must be 'head'")
        frame_id = navigation_visual_check["frame_id"]
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError(
                "navigation_visual_check.frame_id must be a non-empty string"
            )
        if (
            navigation_visual_check["assessment"]
            != "navigation_target_visually_confirmed"
        ):
            raise ValueError(
                "navigation_visual_check.assessment must be "
                "'navigation_target_visually_confirmed'"
            )
        return {
            "camera": "head",
            "frame_id": frame_id.strip(),
            "assessment": "navigation_target_visually_confirmed",
        }

    @staticmethod
    def _validated_navigation_number(
        name: str,
        value: Any,
        *,
        minimum: float,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} must be a finite number")
        number = float(value)
        lower_ok = number >= minimum if minimum_inclusive else number > minimum
        upper_ok = maximum is None or number <= maximum
        if not np.isfinite(number) or not lower_ok or not upper_ok:
            bounds = (
                f"[{minimum},{maximum}]"
                if minimum_inclusive
                else f"({minimum},{maximum}]"
            )
            raise ValueError(f"{name} must be finite and within {bounds}")
        return number

    def navigate_to(
        self,
        *,
        projection_id: str | None = None,
        navigation_visual_check: dict[str, Any] | None = None,
        relative_motion: dict[str, Any] | None = None,
        standoff_m: float | None = None,
        max_travel_m: float | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """Execute one projection-bound or explicit base-relative motion."""

        projection_mode = relative_motion is None
        if projection_mode:
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("projection_id must be a non-empty string")
            if navigation_visual_check is None:
                raise ValueError(
                    "navigation_visual_check is required for projection navigation"
                )
            payload = {
                "projection_id": projection_id.strip(),
                "navigation_visual_check": (
                    self._validated_navigation_visual_check(navigation_visual_check)
                ),
                "standoff_m": self._validated_navigation_number(
                    "standoff_m",
                    0.85 if standoff_m is None else standoff_m,
                    minimum=0.45,
                    maximum=1.50,
                ),
                "max_travel_m": self._validated_navigation_number(
                    "max_travel_m",
                    1.0 if max_travel_m is None else max_travel_m,
                    minimum=0.0,
                    maximum=1.50,
                    minimum_inclusive=False,
                ),
            }
        else:
            if any(
                value is not None
                for value in (
                    projection_id,
                    navigation_visual_check,
                    standoff_m,
                    max_travel_m,
                )
            ):
                raise ValueError(
                    "relative_motion is mutually exclusive with projection "
                    "navigation arguments"
                )
            payload = {
                "relative_motion": validate_relative_navigation_motion(relative_motion)
            }
        payload["timeout_s"] = self._validated_navigation_number(
            "timeout_s",
            timeout_s,
            minimum=0.0,
            minimum_inclusive=False,
        )
        return self._public_checkpoint_result(
            name="navigate_to",
            result=self._env_method("navigate_to")(**payload),
        )

    def move_to(
        self,
        *,
        hand: str,
        target: dict[str, Any],
        visual_hand_check: dict[str, Any],
        position_tolerance_m: float = 0.02,
        max_travel_m: float = 0.25,
        timeout_s: float = 240.0,
    ) -> dict[str, Any]:
        """Execute one projection-bound or relative physical-hand motion."""

        hand = self._analytic_hand(hand)
        visual_hand_check = self._validated_visual_hand_check(
            hand=hand,
            visual_hand_check=visual_hand_check,
        )
        if not isinstance(target, dict):
            raise ValueError("target must be an object")
        target = dict(target)
        projection_id = target.get("projection_id")
        delta_xyz = target.get("delta_xyz")
        if (projection_id is None) == (delta_xyz is None):
            raise ValueError(
                "target requires exactly one of projection_id or delta_xyz"
            )
        if projection_id is not None:
            if not isinstance(projection_id, str) or not projection_id.strip():
                raise ValueError("target.projection_id must be a non-empty string")
            if "frame" in target:
                raise ValueError("projection targets must not include frame")
            target["projection_id"] = projection_id.strip()
            if "standoff_m" in target:
                standoff = float(target["standoff_m"])
                if not np.isfinite(standoff) or standoff < 0.0:
                    raise ValueError(
                        "target.standoff_m must be finite and non-negative"
                    )
                target["standoff_m"] = standoff
        else:
            values = np.asarray(delta_xyz, dtype=np.float64).reshape(-1)
            if values.size != 3 or not np.isfinite(values).all():
                raise ValueError("target.delta_xyz must contain three finite values")
            if target.get("frame") not in {"world", "eef"}:
                raise ValueError("relative target.frame must be 'world' or 'eef'")
            if "standoff_m" in target:
                raise ValueError("relative targets must not include standoff_m")
            target["delta_xyz"] = values.tolist()
        allowed = {"projection_id", "standoff_m", "delta_xyz", "frame"}
        unknown = set(target).difference(allowed)
        if unknown:
            raise ValueError(f"target contains unsupported fields: {sorted(unknown)}")
        tolerance = self._positive_number("position_tolerance_m", position_tolerance_m)
        travel = self._positive_number("max_travel_m", max_travel_m)
        return self._public_checkpoint_result(
            name="move_to",
            result=self._env_method("move_to")(
                hand=hand,
                target=target,
                visual_hand_check=visual_hand_check,
                position_tolerance_m=tolerance,
                max_travel_m=travel,
                timeout_s=self._positive_number("timeout_s", timeout_s),
            ),
        )

    def rotate_wrist(
        self,
        *,
        hand: str,
        relative_axis_angle: list[float],
        visual_hand_check: dict[str, Any],
        frame: str = "eef",
    ) -> dict[str, Any]:
        """Execute one orientation change for a visually selected hand."""

        hand = self._analytic_hand(hand)
        visual_hand_check = self._validated_visual_hand_check(
            hand=hand,
            visual_hand_check=visual_hand_check,
        )
        if frame not in {"world", "eef"}:
            raise ValueError("frame must be 'world' or 'eef'")
        kwargs: dict[str, Any] = {
            "hand": hand,
            "frame": frame,
            "visual_hand_check": visual_hand_check,
        }
        values = np.asarray(relative_axis_angle, dtype=np.float64).reshape(-1)
        if values.size != 4 or not np.isfinite(values).all():
            raise ValueError("relative_axis_angle must contain four finite values")
        kwargs["relative_axis_angle"] = values.tolist()
        return self._public_checkpoint_result(
            name="rotate_wrist",
            result=self._env_method("rotate_wrist")(**kwargs),
        )

    def close(
        self,
        *,
        hand: str,
        visual_hand_check: dict[str, Any],
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        hand = self._analytic_hand(hand)
        return self._public_checkpoint_result(
            name="close",
            result=self._env_method("close")(
                hand=hand,
                visual_hand_check=self._validated_visual_hand_check(
                    hand=hand,
                    visual_hand_check=visual_hand_check,
                ),
                timeout_s=self._positive_number("timeout_s", timeout_s),
            ),
        )

    def open(
        self,
        *,
        hand: str,
        visual_hand_check: dict[str, Any],
        release_visual_check: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        hand = self._analytic_hand(hand)
        env_kwargs: dict[str, Any] = {
            "hand": hand,
            "visual_hand_check": self._validated_visual_hand_check(
                hand=hand,
                visual_hand_check=visual_hand_check,
            ),
            "timeout_s": self._positive_number("timeout_s", timeout_s),
        }
        release_check = self._validated_release_visual_check(
            hand=hand,
            release_visual_check=release_visual_check,
        )
        if release_check is not None:
            env_kwargs["release_visual_check"] = release_check
        return self._public_checkpoint_result(
            name="open",
            result=self._env_method("open")(**env_kwargs),
        )

    def press(
        self,
        *,
        hand: str,
        visual_hand_check: dict[str, Any],
        projection_id: str,
        travel_m: float,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        hand = self._analytic_hand(hand)
        visual_hand_check = self._validated_visual_hand_check(
            hand=hand,
            visual_hand_check=visual_hand_check,
        )
        if not isinstance(projection_id, str) or not projection_id.strip():
            raise ValueError("projection_id must be a non-empty string")
        travel = self._positive_number("travel_m", travel_m)
        return self._public_checkpoint_result(
            name="press",
            result=self._env_method("press")(
                hand=hand,
                visual_hand_check=visual_hand_check,
                projection_id=projection_id.strip(),
                travel_m=travel,
                timeout_s=self._positive_number("timeout_s", timeout_s),
            ),
        )

    def save_robot_state_checkpoint(
        self,
        *,
        semantic_label: str | None = None,
        terminal_failure: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Capture synchronized RGB-D as a read-only visual anchor."""

        if semantic_label is not None:
            if not isinstance(semantic_label, str) or not semantic_label.strip():
                raise ValueError("semantic_label must be a non-empty string")
            if len(semantic_label) > 128:
                raise ValueError("semantic_label must contain at most 128 characters")
        if terminal_failure is not None:
            policy = self.task_spec.terminal_failure_policy
            if policy is None:
                raise ValueError(
                    f"{self.task_name} does not define a visual terminal-failure policy"
                )
            if not isinstance(terminal_failure, dict):
                raise ValueError("terminal_failure must be an object")
            required = {"condition", "cause", "camera", "frame_id"}
            if set(terminal_failure) != required:
                raise ValueError(
                    "terminal_failure requires exactly condition, cause, camera, "
                    "and frame_id"
                )
            if terminal_failure["condition"] != policy.condition:
                raise ValueError(
                    f"terminal_failure.condition must be {policy.condition}"
                )
            if terminal_failure["cause"] not in policy.causes:
                raise ValueError("terminal_failure.cause is invalid")
            if terminal_failure["camera"] not in policy.cameras:
                raise ValueError("terminal_failure.camera is invalid")
            if (
                not isinstance(terminal_failure["frame_id"], str)
                or not terminal_failure["frame_id"].strip()
            ):
                raise ValueError("terminal_failure.frame_id must be non-empty")
            terminal_failure = {
                **terminal_failure,
                "frame_id": terminal_failure["frame_id"].strip(),
            }
        kwargs: dict[str, Any] = {}
        if semantic_label is not None:
            kwargs["semantic_label"] = semantic_label.strip()
        if terminal_failure is not None:
            kwargs["terminal_failure"] = terminal_failure
        return self._public_checkpoint_result(
            name="save_robot_state_checkpoint",
            result=self._env_method("save_robot_state_checkpoint")(**kwargs),
        )

    @staticmethod
    def _pi0_nav_pick_state_record(
        *,
        chunk: int,
        env_steps: int,
        total_env_steps: int | None = None,
        obs: dict[str, Any],
        info: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        instruction: str,
        monitor: Any = None,
        model_info: Any = None,
    ) -> dict[str, Any]:
        compact = extract_policy_state(np.asarray(obs["states"], dtype=np.float32))
        raw_task_success = official_task_success(info)
        success_receipt = _validated_official_success_receipt(
            _official_success_receipt_from_info(info)
        )
        absolute_env_steps = (
            int(env_steps) if total_env_steps is None else int(total_env_steps)
        )
        if isinstance(monitor, dict) and "total_env_steps" in monitor:
            absolute_env_steps = int(monitor["total_env_steps"])
        return {
            "chunk": int(chunk),
            "env_steps": int(env_steps),
            "total_env_steps": absolute_env_steps,
            "instruction": instruction,
            "predicted_action_shape": (
                None if chunk == 0 else [DEFAULT_ACTION_CHUNK, 23]
            ),
            "local_grasp_success": bool(
                isinstance(monitor, dict) and monitor.get("local_grasp_success") is True
            ),
            "pi0_nav_pick_monitor": _jsonable(monitor),
            "raw_proprio": _jsonable(np.asarray(obs["states"], dtype=np.float32)),
            "policy_state": _jsonable(compact),
            "policy_state_segments": segment_ranges(POLICY_STATE_SEGMENTS),
            "env_action_segments": segment_ranges(ENV_ACTION_SEGMENTS),
            "reward": _jsonable(reward),
            "terminated": _as_bool(terminated),
            "truncated": _as_bool(truncated),
            "info": _jsonable(info),
            "model": _jsonable(model_info),
            "raw_task_success_at_record": raw_task_success,
            "task_success": bool(raw_task_success or success_receipt is not None),
            "official_success_receipt": success_receipt,
            "official_success_source": 'info["done"]["success"]',
        }

    def pi0_nav_pick(
        self,
        *,
        instruction: str,
        chunks: int,
        current_object_visual_check: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run exactly the LLM-requested number of complete Pi0 chunks."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if self.pure_vla_baseline and current_object_visual_check is not None:
            raise ValueError(
                "pure VLA baseline cannot consume public visual-review evidence"
            )
        if (
            isinstance(chunks, bool)
            or not isinstance(chunks, (int, np.integer))
            or int(chunks) < 1
        ):
            raise ValueError("chunks must be an integer greater than or equal to 1")
        requested_chunks = int(chunks)
        self._vla_invocations += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        call_root = self._attempt_root() / "vla_calls"
        while True:
            self._vla_call_index += 1
            call_dir = call_root / f"call_{self._vla_call_index:03d}"
            if not call_dir.exists():
                break
        call_dir.mkdir(parents=True, exist_ok=False)
        call_path = call_dir / "pi0_nav_pick_call.json"
        try:
            descriptor = os.open(
                call_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise RuntimeError("VLA call artifact collision") from exc
        claimed_at_unix_s = time.time()
        request_id = hashlib.sha256(
            (
                f"{self.output_dir.resolve()}\0{instruction}\0"
                f"{requested_chunks}\0{time.time_ns()}"
            ).encode("utf-8")
        ).hexdigest()
        call_record = {
            "schema_version": PI0_NAV_PICK_CALL_ARTIFACT_SCHEMA_VERSION,
            "name": "pi0_nav_pick",
            "request_id": request_id,
            "status": "pending",
            "instruction": instruction,
            "requested_chunks": requested_chunks,
            "current_object_visual_check": _jsonable(current_object_visual_check),
            "run_nonce": self.run_nonce,
            "attempt_index": self.attempt_index,
            "attempt_nonce": self.attempt_nonce,
            "global_vla_invocations": self._vla_invocations,
            "claimed_at_unix_s": claimed_at_unix_s,
            "result_path": str(call_dir / "pi0_nav_pick_result.json"),
        }
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(call_record, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        states_path = call_dir / "pi0_nav_pick_states.json"
        result_path = call_dir / "pi0_nav_pick_result.json"
        raw_final_info_path = call_dir / "pi0_nav_pick_raw_final_info.json"

        started = time.time()
        states: list[dict[str, Any]] = []
        chunks_used = 0
        full_chunks_executed = 0
        env_steps_used = 0
        vla_env_steps_used = 0
        handoff_env_steps_used = 0
        total_env_steps = 0
        task_success = False
        local_grasp_success = False
        terminated = False
        truncated = False
        stop_reason = "not_started"
        last_info: Any = self._current_info
        planner_handoff_ready = False
        lifecycle_finalized = False
        preflight_failed_preconditions: list[str] = []
        vla_rearm_active = False
        runtime_success_receipt: dict[str, Any] | None = None
        error: str | None = None
        obs = (
            self._current_observation
            if isinstance(self._current_observation, dict)
            else None
        )

        def latch_success_receipt(value: Any) -> None:
            nonlocal runtime_success_receipt, task_success
            receipt = _validated_official_success_receipt(value)
            if receipt is None:
                return
            if runtime_success_receipt is None:
                runtime_success_receipt = receipt
            elif runtime_success_receipt != receipt:
                raise RuntimeError(
                    "pi0_nav_pick official success receipt changed after first latch"
                )
            task_success = True

        def converge_vla_lifecycle() -> None:
            nonlocal vla_rearm_active
            nonlocal planner_handoff_ready
            nonlocal lifecycle_finalized
            nonlocal task_success

            if not vla_rearm_active:
                return
            disable_actions = getattr(self.model, "disable_actions", None)
            healthz = getattr(self.model, "healthz", None)
            finalize_paused_runtime = getattr(
                self.env,
                "finalize_paused_runtime",
                None,
            )
            if (
                not callable(disable_actions)
                or not callable(healthz)
                or not callable(finalize_paused_runtime)
            ):
                raise RuntimeError(
                    "Pi0 lifecycle convergence requires disable, health, and "
                    "env finalization APIs"
                )
            disabled = disable_actions()
            if (
                not isinstance(disabled, dict)
                or disabled.get("actions_enabled") is not False
            ):
                raise RuntimeError("Pi0 VLA disable was not confirmed")
            health = healthz()
            if (
                not isinstance(health, dict)
                or health.get("actions_enabled") is not False
            ):
                raise RuntimeError("Pi0 VLA health still permits actions")
            finalized = finalize_paused_runtime(
                {
                    "actions_enabled": False,
                    "endpoint": getattr(self.model, "endpoint", None),
                    "disable_confirmation": _jsonable(disabled),
                    "healthz": _jsonable(health),
                }
            )
            if (
                not isinstance(finalized, dict)
                or finalized.get("vla_actions_enabled") is not False
                or finalized.get("lifecycle_finalized") is not True
            ):
                raise RuntimeError("env did not confirm Pi0 lifecycle finalization")
            latch_success_receipt(finalized.get("official_success_receipt"))
            finalized_task_success = finalized.get("task_success")
            if not isinstance(finalized_task_success, bool):
                raise RuntimeError(
                    "env Pi0 lifecycle finalization omitted boolean task_success"
                )
            if finalized_task_success:
                task_success = True
            if task_success and runtime_success_receipt is None:
                raise RuntimeError(
                    "env finalized official success without its raw-success receipt"
                )
            finalized_capability = finalized.get("capability")
            if finalized_capability is not None:
                _validated_attachment_count(finalized_capability)
            planner_handoff_ready = finalized.get("controller_state") == "planner"
            lifecycle_finalized = True
            vla_rearm_active = False

        try:
            if self.env is None or self.model is None:
                raise RuntimeError("pi0_nav_pick requires env and model")
            if self.max_episode_steps is None or self.max_episode_steps <= 0:
                raise ValueError("pi0_nav_pick requires a positive episode horizon")
            if self.action_horizon != DEFAULT_ACTION_CHUNK:
                raise ValueError(
                    "pi0_nav_pick requires action_horizon=32 for complete chunks"
                )
            pi0_nav_pick_chunk_step = getattr(self.env, "pi0_nav_pick_chunk_step", None)
            if not callable(pi0_nav_pick_chunk_step):
                raise RuntimeError("pi0_nav_pick requires env.pi0_nav_pick_chunk_step")

            current_observation = getattr(self.env, "current_observation", None)
            if not callable(current_observation):
                raise RuntimeError("pi0_nav_pick requires env.current_observation")
            baseline_total = self.total_env_steps
            if baseline_total < 0:
                raise RuntimeError(
                    "pi0_nav_pick requires a non-negative current env-step total"
                )
            total_env_steps = int(baseline_total)
            stop_reason = "running"

            remaining_steps = self.max_episode_steps - total_env_steps
            required_env_steps = requested_chunks * DEFAULT_ACTION_CHUNK
            if remaining_steps < required_env_steps:
                stop_reason = "requested_chunks_exceed_remaining_episode_steps"
                preflight_failed_preconditions.append(
                    "insufficient_remaining_episode_steps"
                )
            elif self.elapsed_wall_clock_s >= self.max_wall_clock_s:
                stop_reason = "global_wall_clock_budget_exhausted"
                preflight_failed_preconditions.append(
                    "global_wall_clock_budget_exhausted"
                )

            if stop_reason == "running":
                prepare = getattr(self.env, "prepare_vla_invocation", None)
                if not callable(prepare):
                    raise RuntimeError(
                        "pi0_nav_pick requires env.prepare_vla_invocation"
                    )
                prepare_kwargs: dict[str, Any] = {
                    "invocation_id": request_id,
                    "call_index": self._vla_call_index,
                    "vla_status": None,
                }
                if self.pure_vla_baseline:
                    prepare_kwargs["baseline_internal_authorization"] = True
                if current_object_visual_check is not None:
                    prepare_kwargs["current_object_visual_check"] = (
                        current_object_visual_check
                    )
                preflight = prepare(
                    **prepare_kwargs,
                )
                if (
                    not isinstance(preflight, dict)
                    or preflight.get("primitive_success") is not True
                ):
                    if isinstance(preflight, dict):
                        failed = preflight.get("failed_preconditions", [])
                        if isinstance(failed, list) and all(
                            isinstance(item, str) and item for item in failed
                        ):
                            preflight_failed_preconditions = list(failed)
                        latch_success_receipt(preflight.get("official_success_receipt"))
                        if preflight.get("task_success") is True:
                            task_success = True
                        reported_total = preflight.get(
                            "total_env_steps", total_env_steps
                        )
                        if (
                            not isinstance(reported_total, bool)
                            and isinstance(reported_total, (int, np.integer))
                            and int(reported_total) >= 0
                        ):
                            total_env_steps = int(reported_total)
                    stop_reason = "vla_runtime_precondition_rejected"
                if stop_reason == "running":
                    reported_attachment_count = preflight.get(
                        "attachment_count_at_invocation_start"
                    )
                    attachments_present = preflight.get(
                        "attachments_present_at_invocation_start"
                    )
                    if (
                        isinstance(reported_attachment_count, bool)
                        or not isinstance(
                            reported_attachment_count,
                            (int, np.integer),
                        )
                        or int(reported_attachment_count) not in {0, 1, 2}
                        or not isinstance(attachments_present, bool)
                        or attachments_present != (int(reported_attachment_count) > 0)
                    ):
                        raise RuntimeError(
                            "pi0_nav_pick preflight returned inconsistent "
                            "per-hand attachment state"
                        )
                    # The successful env preflight may already have switched
                    # controller ownership to VLA, so every later failure must
                    # run the disable/finalize convergence path.
                    vla_rearm_active = True
                    enable_actions = getattr(self.model, "enable_actions", None)
                    if not callable(enable_actions):
                        raise RuntimeError("pi0_nav_pick requires model.enable_actions")
                    enabled = enable_actions()
                    if (
                        not isinstance(enabled, dict)
                        or enabled.get("actions_enabled") is not True
                    ):
                        raise RuntimeError("VLA action enable was not confirmed")
                    prepare_kwargs["vla_status"] = enabled
                    confirmed = prepare(**prepare_kwargs)
                    if not isinstance(confirmed, dict):
                        raise RuntimeError(
                            "env did not return a VLA action re-arm confirmation"
                        )
                    latch_success_receipt(confirmed.get("official_success_receipt"))
                    if confirmed.get("task_success") is True:
                        task_success = True
                    if confirmed.get("primitive_success") is not True:
                        stop_reason = "vla_runtime_precondition_rejected"
                        failed = confirmed.get("failed_preconditions", [])
                        if isinstance(failed, list) and all(
                            isinstance(item, str) and item for item in failed
                        ):
                            preflight_failed_preconditions = list(failed)
                    if task_success and runtime_success_receipt is None:
                        raise RuntimeError(
                            "VLA re-arm reported task success without raw receipt"
                        )
                    if (
                        stop_reason == "running"
                        and confirmed.get("primitive_success") is not True
                    ):
                        raise RuntimeError("env did not confirm VLA action re-arm")

            if stop_reason == "running":
                # Attached-state authorization is deliberately completed by both
                # prepare calls before this private sensor refresh.  The
                # refresh may replace the frame-cache head, so doing it earlier
                # would make an otherwise fresh public observe receipt stale
                # between validation and VLA re-arm confirmation.
                current = current_observation()
                if not isinstance(current, tuple) or len(current) != 2:
                    raise RuntimeError(
                        "pi0_nav_pick env.current_observation returned an invalid result"
                    )
                obs, info = current
                if not isinstance(obs, dict):
                    raise RuntimeError(
                        "pi0_nav_pick current observation must be a mapping"
                    )
                self._current_observation = obs
                self._current_info = info
                last_info = info
                rpent = info.get("_rpent") if isinstance(info, dict) else None
                refreshed_total = (
                    rpent.get("total_env_steps") if isinstance(rpent, dict) else None
                )
                if (
                    isinstance(refreshed_total, bool)
                    or not isinstance(refreshed_total, (int, np.integer))
                    or int(refreshed_total) < 0
                ):
                    raise RuntimeError(
                        "pi0_nav_pick current observation must report a non-negative "
                        "integer _rpent.total_env_steps"
                    )
                if int(refreshed_total) != total_env_steps:
                    raise RuntimeError(
                        "pi0_nav_pick private observation changed env-step accounting: "
                        f"before={total_env_steps} after={refreshed_total!r}"
                    )
                total_env_steps = int(refreshed_total)
                latch_success_receipt(_official_success_receipt_from_info(info))
                if official_task_success(info):
                    task_success = True
                    if runtime_success_receipt is None:
                        raise RuntimeError(
                            "current raw task success lacks immutable receipt"
                        )

            if stop_reason == "running" and obs is None:
                raise RuntimeError(
                    "pi0_nav_pick requires an initial or refreshed observation"
                )
            if obs is not None:
                states.append(
                    self._pi0_nav_pick_state_record(
                        chunk=0,
                        env_steps=0,
                        total_env_steps=total_env_steps,
                        obs=obs,
                        info=last_info,
                        reward=None,
                        terminated=False,
                        truncated=False,
                        instruction=instruction,
                    )
                )
            _write_json_atomic(states_path, states)

            while (
                stop_reason == "running"
                and not task_success
                and not terminated
                and not truncated
                and chunks_used < requested_chunks
            ):
                if self.max_episode_steps - total_env_steps < DEFAULT_ACTION_CHUNK:
                    raise RuntimeError(
                        "remaining episode steps changed after exact-chunk preflight"
                    )
                model_obs = dict(obs)
                # Preserve the configured language byte-for-byte. Adding a
                # navigation or hand hint changes the Pi0 policy distribution.
                model_obs["task_descriptions"] = instruction
                predicted, model_info = self.model.predict_action_batch(
                    model_obs, mode="eval"
                )
                actions = validate_action_chunk(
                    predicted, max_horizon=DEFAULT_ACTION_CHUNK
                )
                if actions.shape != (DEFAULT_ACTION_CHUNK, 23):
                    raise RuntimeError(
                        "pi0_nav_pick model must return exactly one complete "
                        f"[32,23] chunk, got {actions.shape}"
                    )

                previous_total = total_env_steps
                chunk_index = chunks_used + 1
                chunks_used = chunk_index
                self._global_vla_chunks += 1
                obs, reward, term, trunc, info = pi0_nav_pick_chunk_step(
                    actions,
                    chunk_index=chunk_index,
                )
                last_info = info
                terminated = _as_bool(term)
                truncated = _as_bool(trunc)

                rpent = info.get("_rpent") if isinstance(info, dict) else None
                monitor = (
                    rpent.get("pi0_nav_pick_monitor")
                    if isinstance(rpent, dict)
                    else None
                )
                if not isinstance(monitor, dict):
                    raise RuntimeError(
                        "pi0_nav_pick env result omitted pi0_nav_pick_monitor"
                    )
                missing = sorted(set(_PI0_NAV_PICK_MONITOR_FIELDS) - set(monitor))
                if missing:
                    raise RuntimeError(
                        f"pi0_nav_pick monitor missing required fields: {missing}"
                    )
                monitor = {
                    field: monitor[field]
                    for field in (
                        *_PI0_NAV_PICK_MONITOR_FIELDS,
                        *_PI0_NAV_PICK_OPTIONAL_MONITOR_FIELDS,
                    )
                    if field in monitor
                }
                # Visual artifacts are best-effort audit material. They are
                # never prerequisites for local grasp, task success, controller
                # transfer, or a later VLA invocation.
                visual_review = monitor.get("visual_review")
                if (
                    isinstance(visual_review, dict)
                    and self._progress_callback is not None
                ):
                    try:
                        self._progress_callback(
                            "pi0_nav_pick",
                            {
                                "chunk_index": int(chunk_index),
                                "env_step": int(monitor["total_env_steps"]),
                                "global_vla_invocations": self._vla_invocations,
                                "visual_review": visual_review,
                            },
                        )
                    except Exception:
                        logger.warning(
                            "dashboard progress update failed", exc_info=True
                        )
                monitor_stop_reason = monitor.get("stop_reason")
                if monitor_stop_reason is not None and (
                    not isinstance(monitor_stop_reason, str) or not monitor_stop_reason
                ):
                    raise RuntimeError(
                        "pi0_nav_pick monitor stop_reason must be null or non-empty"
                    )
                # Persist only the public monitor allowlist. Simulator-private
                # debugging fields must not cross into agent artifacts.
                public_info = dict(info)
                public_rpent = dict(rpent)
                public_rpent["pi0_nav_pick_monitor"] = monitor
                public_info["_rpent"] = public_rpent
                info = public_info
                last_info = info
                latch_success_receipt(monitor.get("official_success_receipt"))
                latch_success_receipt(_official_success_receipt_from_info(info))
                if official_task_success(info):
                    task_success = True
                    if runtime_success_receipt is None:
                        raise RuntimeError(
                            "raw task success lacks immutable env receipt"
                        )
                executed_steps = monitor["executed_steps"]
                if (
                    isinstance(executed_steps, bool)
                    or not isinstance(executed_steps, (int, np.integer))
                    or not 1 <= int(executed_steps) <= DEFAULT_ACTION_CHUNK
                ):
                    raise RuntimeError(
                        f"invalid pi0_nav_pick executed_steps: {executed_steps!r}"
                    )
                handoff_env_steps = monitor["handoff_env_steps"]
                if (
                    isinstance(handoff_env_steps, bool)
                    or not isinstance(handoff_env_steps, (int, np.integer))
                    or int(handoff_env_steps) < 0
                ):
                    raise RuntimeError(
                        f"invalid pi0_nav_pick handoff_env_steps: {handoff_env_steps!r}"
                    )
                reported_total = monitor["total_env_steps"]
                if (
                    isinstance(reported_total, bool)
                    or not isinstance(reported_total, (int, np.integer))
                    or int(reported_total) < previous_total + int(executed_steps)
                    or int(reported_total) > self.max_episode_steps
                ):
                    raise RuntimeError(
                        f"invalid pi0_nav_pick total_env_steps: {reported_total!r}"
                    )
                if not isinstance(monitor["local_grasp_success"], bool):
                    raise RuntimeError(
                        "pi0_nav_pick local_grasp_success must be boolean"
                    )
                local_grasp_success = monitor["local_grasp_success"] is True
                derived_handoff_steps = (
                    int(reported_total) - previous_total - int(executed_steps)
                )
                # A chunk may have physically completed before a later audit
                # check detects inconsistent runtime metadata.  Persist its
                # observed execution and absolute environment total first so
                # the result never rewrites a real 32-step action as zero work.
                env_steps_used += int(reported_total) - previous_total
                vla_env_steps_used += int(executed_steps)
                handoff_env_steps_used += int(handoff_env_steps)
                total_env_steps = int(reported_total)
                if int(executed_steps) == DEFAULT_ACTION_CHUNK:
                    full_chunks_executed += 1
                states.append(
                    self._pi0_nav_pick_state_record(
                        chunk=chunks_used,
                        env_steps=env_steps_used,
                        total_env_steps=total_env_steps,
                        obs=obs,
                        info=info,
                        reward=reward,
                        terminated=term,
                        truncated=trunc,
                        instruction=instruction,
                        monitor=monitor,
                        model_info=model_info,
                    )
                )
                self._current_observation = obs
                self._current_info = info
                if int(handoff_env_steps) != derived_handoff_steps:
                    raise RuntimeError(
                        "pi0_nav_pick handoff accounting mismatch: "
                        f"reported={handoff_env_steps!r} "
                        f"derived={derived_handoff_steps!r}"
                    )
                if isinstance(rpent, dict):
                    for field, expected in (
                        ("executed_steps", int(executed_steps)),
                        ("handoff_env_steps", int(handoff_env_steps)),
                        ("total_env_steps", int(reported_total)),
                    ):
                        value = rpent.get(field, expected)
                        if value != expected:
                            raise RuntimeError(
                                "pi0_nav_pick monitor/accounting mismatch: "
                                f"{field}={value!r} monitor={expected!r}"
                            )
                public_rpent["executed_steps"] = int(executed_steps)
                public_rpent["handoff_env_steps"] = int(handoff_env_steps)
                public_rpent["total_env_steps"] = int(reported_total)
                public_rpent["pi0_nav_pick_monitor"] = monitor
                public_info["_rpent"] = public_rpent
                info = public_info
                last_info = info
                capability = monitor.get("capability")
                runtime_attachment_count = _validated_attachment_count(capability)
                if local_grasp_success and runtime_attachment_count < 1:
                    raise RuntimeError(
                        "pi0_nav_pick local grasp lacks a runtime attachment"
                    )
                if not isinstance(monitor["controller_state"], str):
                    raise RuntimeError("pi0_nav_pick controller_state must be a string")
                if not isinstance(monitor["action_source"], str):
                    raise RuntimeError("pi0_nav_pick action_source must be a string")
                if not isinstance(monitor["vla_actions_enabled"], bool):
                    raise RuntimeError(
                        "pi0_nav_pick vla_actions_enabled must be boolean"
                    )
                if (
                    int(executed_steps) != DEFAULT_ACTION_CHUNK
                    and not task_success
                    and not terminated
                    and not truncated
                ):
                    raise RuntimeError(
                        "pi0_nav_pick env truncated a chunk without a terminal reason"
                    )
                if terminated:
                    stop_reason = "terminated"
                elif truncated:
                    stop_reason = "truncated"
                else:
                    stop_reason = "running"
                _write_json_atomic(states_path, states)

            if vla_rearm_active:
                converge_vla_lifecycle()
            exact_requested_chunks_completed = bool(
                chunks_used == requested_chunks
                and full_chunks_executed == requested_chunks
                and vla_env_steps_used == requested_chunks * DEFAULT_ACTION_CHUNK
            )
            if task_success:
                stop_reason = "official_task_success"
            elif exact_requested_chunks_completed:
                stop_reason = "requested_chunks_completed"
            elif terminated:
                stop_reason = "terminated"
            elif truncated:
                stop_reason = "truncated"
            elif stop_reason == "running":
                raise RuntimeError(
                    "Pi0 invocation returned without completing its exact chunks"
                )
        except Exception as exc:
            logger.exception("pi0_nav_pick failed")
            error = f"{type(exc).__name__}: {exc}"
            if vla_rearm_active:
                try:
                    converge_vla_lifecycle()
                except Exception as cleanup_exc:
                    logger.exception("pi0_nav_pick VLA error cleanup failed")
                    error = (
                        f"{error}; cleanup_failed="
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            stop_reason = "error"

        exact_requested_chunks_completed = bool(
            chunks_used == requested_chunks
            and full_chunks_executed == requested_chunks
            and vla_env_steps_used == requested_chunks * DEFAULT_ACTION_CHUNK
        )
        primitive_success = bool(
            error is None
            and lifecycle_finalized
            and (task_success or exact_requested_chunks_completed)
        )
        runner_terminate = bool(
            terminated
            or truncated
            or _runner_should_terminate(
                tool_name="pi0_nav_pick",
                task_success=bool(task_success),
                stop_reason=stop_reason,
                terminal_failure_policy=self.task_spec.terminal_failure_policy,
            )
        )
        result = {
            "_finish": runner_terminate,
            "runner_termination_reason": (
                "official_task_success"
                if task_success
                else "environment_terminated"
                if terminated
                else "environment_truncated"
                if truncated
                else "attempt_budget_exhausted"
                if runner_terminate
                else None
            ),
            "name": "pi0_nav_pick",
            "primitive_success": primitive_success,
            "task_success": bool(task_success),
            "official_success_source": 'info["done"]["success"]',
            "official_success_receipt": (
                runtime_success_receipt
                or _official_success_receipt_from_info(last_info)
            ),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
            "requested_chunks": requested_chunks,
            "exact_requested_chunks_completed": exact_requested_chunks_completed,
            "chunks_used": chunks_used,
            "global_vla_chunks": self._global_vla_chunks,
            "global_vla_invocations": self._vla_invocations,
            "full_chunks_executed": full_chunks_executed,
            "env_steps_used": env_steps_used,
            "vla_env_steps_used": vla_env_steps_used,
            "handoff_env_steps_used": handoff_env_steps_used,
            "total_env_steps": total_env_steps,
            "max_episode_steps": self.max_episode_steps,
            "action_horizon": self.action_horizon,
            "required_action_shape": [DEFAULT_ACTION_CHUNK, 23],
            "elapsed_s": round(time.time() - started, 2),
            "global_elapsed_wall_clock_s": round(self.elapsed_wall_clock_s, 3),
            "attempt_index": self.attempt_index,
            "attempt_nonce": self.attempt_nonce,
            "run_nonce": self.run_nonce,
            "vla_call_index": self._vla_call_index,
            "failed_preconditions": preflight_failed_preconditions,
            "error": error,
        }
        result = _sanitize_pi0_public_result(result)
        _write_json_atomic(states_path, states)
        _write_json_atomic(raw_final_info_path, _jsonable(last_info))
        _write_json_atomic(result_path, result)
        call_record.update(
            {
                "status": "completed",
                "completed_at_unix_s": time.time(),
                "outcome": "error" if error is not None else stop_reason,
                "task_success": bool(task_success),
                "requested_chunks": requested_chunks,
                "exact_requested_chunks_completed": (exact_requested_chunks_completed),
                "local_grasp_success": bool(local_grasp_success),
                "planner_handoff_ready": bool(planner_handoff_ready),
                "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            }
        )
        _write_json_atomic(call_path, call_record)
        self.last_result = result
        return result


__all__ = ["BehaviorPrimitives"]
