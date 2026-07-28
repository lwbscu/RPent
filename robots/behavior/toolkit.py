"""The peer-capability BEHAVIOR toolkit."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import re
import shutil
import threading
import time
import traceback
from pathlib import Path
from types import MappingProxyType
from typing import Any

from robots.behavior.schemas import (
    BEHAVIOR_TOOL_NAMES,
    behavior_tool_specs_for_task,
)
from robots.behavior.spec import BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S
from robots.behavior.task_specs import (
    BehaviorTaskSpec,
    get_task_spec,
)
from robots.behavior.tools import BehaviorPrimitives
from rpent.tools.toolkit import Toolkit, ToolResult

RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE = "raw_official_success_v1"
_OFFICIAL_SUCCESS_SOURCES = {
    'info["done"]["success"]',
    'info["done"]["success"] via task_success',
}
_PUBLICATION_SOURCE_ARTIFACTS = frozenset(
    {
        "official_success_receipt",
        "behavior_action_trace",
        "behavior_tool_trace",
        "final_result",
        "run_manifest",
        "session_manifest",
    }
)
_DEPRECATED_PI0_LIMIT_FIELDS = frozenset(
    {
        "max_chunks",
        "max_total_vla_chunks",
        "max_vla_chunks_per_call",
    }
)
_DEPRECATED_PI0_LIMIT_REASONS = frozenset(
    {
        "call_chunk_limit",
        "global_vla_chunk_budget_exhausted",
    }
)
_HARD_EXECUTION_BUDGET_PRECONDITIONS = frozenset(
    {
        "global_env_step_budget_exhausted",
        "global_tool_call_budget_exhausted",
        "global_wall_clock_budget_exhausted",
    }
)
_FORBIDDEN_PUBLICATION_KEYS = {
    "action",
    "pixel",
    "row",
    "col",
    "tool",
    "tools_used",
    "instruction",
    "instruction_patterns",
    "chunks",
    "requested_chunks",
    "max_chunks",
    "selected_max_chunks_range",
    "observed_chunks_used_range",
    "call_order",
    "camera",
    "camera_order",
    "sequence",
    "frame_id",
    "xyz",
    "pose",
    "qpos",
    "checkpoint_path",
    "activity_instance_id",
    "native_instance",
    "seed",
    "held_hand",
    "press_hand",
    "requested_role",
    "resolved_hand",
    "selected_hand",
    "semantic_role",
}


class BehaviorToolResult(ToolResult):
    """BEHAVIOR-only MCP packaging for public RGB-D observations."""

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        result = self.result
        if not isinstance(result, dict):
            return [
                {"type": "text", "text": str(result)[: self.MAX_TEXT_BYTES_IN_RESULT]}
            ]

        result_for_text = dict(result)
        image_payloads: list[tuple[str, str, bytes]] = []

        def extract(
            *,
            container: dict[str, Any],
            field_name: str,
            camera: str,
            kind: str,
        ) -> None:
            if field_name not in container:
                return
            value = container.pop(field_name)
            if value is None:
                return
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise TypeError(
                    f"{field_name} must contain bytes-like BEHAVIOR image data"
                )
            image_payloads.append((camera, kind, bytes(value)))

        primary_camera = str(result.get("camera") or "primary")
        extract(
            container=result_for_text,
            field_name="_image_bytes",
            camera=primary_camera,
            kind="rgb",
        )
        extract(
            container=result_for_text,
            field_name="_depth_image_bytes",
            camera=primary_camera,
            kind="depth_visualization",
        )
        extract(
            container=result_for_text,
            field_name="_image_cam_bytes",
            camera="camera",
            kind="rgb",
        )
        extract(
            container=result_for_text,
            field_name="_image_wrist_bytes",
            camera="wrist",
            kind="rgb",
        )

        checkpoint_images = result_for_text.get("images")
        if isinstance(checkpoint_images, dict):
            images_for_text: dict[str, Any] = {}
            for camera, view in checkpoint_images.items():
                camera_name = str(camera)
                if not isinstance(view, dict):
                    images_for_text[camera_name] = view
                    continue
                view_for_text = dict(view)
                extract(
                    container=view_for_text,
                    field_name="_image_bytes",
                    camera=camera_name,
                    kind="rgb",
                )
                extract(
                    container=view_for_text,
                    field_name="_depth_image_bytes",
                    camera=camera_name,
                    kind="depth_visualization",
                )
                images_for_text[camera_name] = view_for_text
            result_for_text["images"] = images_for_text

        if image_payloads:
            result_for_text["mcp_image_block_order"] = [
                {
                    "content_block_index": index,
                    "camera": camera,
                    "kind": kind,
                }
                for index, (camera, kind, _) in enumerate(image_payloads, start=1)
            ]

        text = json.dumps(result_for_text, indent=2, default=str)
        if len(text) > self.MAX_TEXT_BYTES_IN_RESULT:
            text = text[: self.MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for _, _, image_bytes in image_payloads:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                }
            )
        return blocks


class BehaviorToolkit(Toolkit):
    """Expose the current versioned peer primitives in the BEHAVIOR API."""

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        video_path: str | Path | None = None,
        dashboard: Any = None,
    ) -> None:
        # Official RPent always registers its shared file/IO tools. BEHAVIOR
        # intentionally exposes only its task-scoped public primitive surface,
        # so remove those registrations locally before adding BEHAVIOR tools.
        super().__init__(dashboard=dashboard)
        self._tools.clear()
        # BEHAVIOR is a deliberately closed control surface. Generic file and
        # lifecycle tools would bypass the synchronized public observations and
        # the runner-owned attempt lifecycle.
        primitives_kwargs = dict(primitives_kwargs)
        self._runtime_resource = primitives_kwargs.pop(
            "_dashboard_runtime_resource",
            None,
        )
        self._command_arbiter = primitives_kwargs.pop(
            "_dashboard_command_arbiter",
            None,
        )
        self._success_latch = primitives_kwargs.pop(
            "_dashboard_success_latch",
            None,
        )
        self._dashboard_motion_allowed = bool(
            primitives_kwargs.pop("_dashboard_motion_allowed", False)
        )
        self._dashboard_observe_allowed = bool(
            primitives_kwargs.pop("_dashboard_observe_allowed", False)
        )
        self._dashboard_control_unavailable_reason = primitives_kwargs.pop(
            "_dashboard_control_unavailable_reason",
            None,
        )
        self._dashboard_controller = None
        self._close_lock = threading.RLock()
        self._manual_intervention_latch = threading.Event()
        self._closed = False
        initial_info = primitives_kwargs.get("initial_info")
        self._shared_success_evidence: dict[str, Any] | None = None
        runtime_video_path = primitives_kwargs.get("video_path")
        if video_path is not None:
            requested_video_path = Path(video_path)
            if (
                runtime_video_path is not None
                and Path(runtime_video_path) != requested_video_path
            ):
                raise ValueError(
                    "BEHAVIOR runtime and toolkit video paths must identify "
                    "the same episode.mp4"
                )
            primitives_kwargs = {
                **primitives_kwargs,
                "video_path": requested_video_path,
            }
        self._primitives = BehaviorPrimitives(
            **primitives_kwargs,
            progress_callback=self._dashboard_progress,
        )
        bind_attempt = getattr(self._success_latch, "bind_attempt", None)
        if callable(bind_attempt):
            bind_attempt(
                run_nonce=self._primitives.run_nonce,
                attempt_nonce=self._primitives.attempt_nonce,
                attempt_index=int(self._primitives.attempt_index),
            )
        self._behavior_phase = str(primitives_kwargs.get("behavior_phase", "eval"))
        if self._behavior_phase not in {"explore", "eval"}:
            raise ValueError("behavior_phase must be 'explore' or 'eval'")
        self._max_tool_calls = int(primitives_kwargs.get("max_tool_calls", 350))
        if self._max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        self._tool_calls = 0
        self._started_monotonic = self._primitives.started_monotonic
        self._tool_trace: list[dict[str, Any]] = []
        self._current_attempt_trace: list[dict[str, Any]] = []
        self._official_task_success = False
        self._terminal_failure_latched = False
        self._execute_tool_lock = threading.Lock()

        self._task_spec = self._primitives.task_spec
        primitive_specs = behavior_tool_specs_for_task(self._task_spec)
        if tuple(primitive_specs) != BEHAVIOR_TOOL_NAMES:
            raise RuntimeError("public primitive schema order mismatch")

        handlers = {
            "pi0_nav_pick": self._primitives.pi0_nav_pick,
            "observe": self._primitives.observe,
            "pixel_to_world": self._primitives.pixel_to_world,
            "move_to": self._primitives.move_to,
            "rotate_wrist": self._primitives.rotate_wrist,
            "close": self._primitives.close,
            "open": self._primitives.open,
            "press": self._primitives.press,
            "save_robot_state_checkpoint": self._primitives.save_robot_state_checkpoint,
            "navigate_to": self._primitives.navigate_to,
        }
        for name, spec in primitive_specs.items():
            self.add_tool(name, spec, handlers[name])

        if tuple(spec["name"] for spec in self.get_tools_spec()) != BEHAVIOR_TOOL_NAMES:
            raise RuntimeError("registered toolkit does not match the frozen API")
        self._tools = MappingProxyType(dict(self._tools))
        if self._runtime_resource is not None:
            self._runtime_resource.toolkit = self
        initial_evidence = {"info": initial_info}
        if (
            self._receipt_binding_from_result(initial_evidence) is not None
            and self._success_latch is not None
            and self._success_latch.observe(initial_evidence)
        ):
            self._shared_success_evidence = self._success_evidence(initial_evidence)
            self._official_task_success = True

    def activate_dashboard_control(self) -> Any:
        """Create and bind one manual controller to this toolkit and State."""

        dashboard = self._dashboard
        bind_state = getattr(dashboard, "bind_controller", None)
        if not callable(bind_state):
            return None
        if self._dashboard_controller is not None:
            return self._dashboard_controller
        if self._command_arbiter is None or self._success_latch is None:
            raise RuntimeError(
                "Dashboard manual control requires the runtime-owned arbiter "
                "and raw-success latch"
            )

        from robots.behavior.dashboard_control import BehaviorDashboardController

        controller = BehaviorDashboardController(
            state=dashboard,
            arbiter=self._command_arbiter,
            success_latch=self._success_latch,
            motion_available=self._dashboard_motion_allowed,
            observe_available=self._dashboard_observe_allowed,
            unavailable_reason=self._dashboard_control_unavailable_reason,
        )
        resource = self._runtime_resource
        try:
            controller.bind_toolkit(self)
            bind_state(controller)
            self._dashboard_controller = controller
            if resource is not None:
                resource.attach_dashboard_control(
                    toolkit=self,
                    controller=controller,
                    dashboard_state=dashboard,
                )
            controller.activate()
        except BaseException:
            unbind = getattr(dashboard, "unbind_controller", None)
            if callable(unbind):
                try:
                    unbind(controller)
                except BaseException:
                    pass
            close = getattr(controller, "close", None)
            if callable(close):
                try:
                    close(timeout_s=0.0)
                except BaseException:
                    pass
            self._dashboard_controller = None
            if resource is not None and resource.dashboard_controller is controller:
                resource.dashboard_controller = None
            raise
        return controller

    def dashboard_control_capabilities(self) -> dict[str, Any]:
        """Return a sanitized fail-closed capability snapshot for the UI."""

        capability = self._primitives.dashboard_control_capabilities()
        if not isinstance(capability, dict):
            raise RuntimeError("dashboard control capability must be an object")
        public = dict(capability)
        simulation_verified = (
            public.get("simulation_identity") == "behavior_omnigibson_r1pro"
        )
        public["motion_available"] = bool(
            self._dashboard_motion_allowed
            and public.get("motion_available") is True
            and simulation_verified
            and public.get("planner_available") is True
            and public.get("position_control_ready") is True
        )
        public["observe_available"] = bool(
            self._dashboard_observe_allowed
            and public.get("observe_available") is True
            and simulation_verified
        )
        if not public["motion_available"] and not public.get(
            "motion_unavailable_reason"
        ):
            public["motion_unavailable_reason"] = (
                self._dashboard_control_unavailable_reason
                or "Manual motion capability is unavailable."
            )
        if not public["observe_available"] and not public.get(
            "observe_unavailable_reason"
        ):
            public["observe_unavailable_reason"] = (
                self._dashboard_control_unavailable_reason
                or "Manual observation capability is unavailable."
            )
        if (
            not public["motion_available"]
            and not public["observe_available"]
            and not public.get("unavailable_reason")
        ):
            public["unavailable_reason"] = public["motion_unavailable_reason"]
        return public

    def _require_dashboard_manual_reservation(self) -> None:
        """Require a live manual reservation for planning without borrowing it."""

        arbiter = getattr(self, "_command_arbiter", None)
        snapshot = getattr(arbiter, "snapshot", None)
        ownership = snapshot() if callable(snapshot) else {}
        if (
            not isinstance(ownership, dict)
            or ownership.get("owner") != "manual"
            or not isinstance(ownership.get("command_id"), str)
            or not ownership["command_id"]
        ):
            raise RuntimeError(
                "Dashboard planning requires an active manual reservation"
            )
        latch = getattr(self, "_success_latch", None)
        is_latched = getattr(latch, "is_latched", None)
        if callable(is_latched) and is_latched():
            raise RuntimeError(
                "raw task success is terminal; Dashboard RPC rejected"
            )

    def _require_dashboard_command_permit(self, command_id: str) -> str:
        """Return the exact command id after checking the shared arbiter."""

        arbiter = getattr(self, "_command_arbiter", None)
        require_permit = getattr(arbiter, "require_manual_permit", None)
        if not callable(require_permit):
            raise RuntimeError(
                "Dashboard primitive requires the shared command arbiter"
            )
        normalized = str(command_id or "").strip()
        require_permit(normalized)
        latch = getattr(self, "_success_latch", None)
        is_latched = getattr(latch, "is_latched", None)
        if callable(is_latched) and is_latched():
            raise RuntimeError(
                "raw task success is terminal; Dashboard RPC rejected"
            )
        return normalized

    def dashboard_prepare_manual_command(
        self,
        *,
        target: str,
        action: str,
        predecessor_plan_id: str | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        """Prepare one reserved manual motion without entering the public tools."""

        if self._closed:
            raise RuntimeError("BEHAVIOR toolkit is closed")
        self._require_dashboard_manual_reservation()
        self._manual_intervention_latch.set()
        handler = getattr(
            self._primitives,
            "dashboard_prepare_manual_command",
            None,
        )
        if not callable(handler):
            raise RuntimeError("Dashboard motion planning is unavailable")
        result = handler(
            target=target,
            action=action,
            predecessor_plan_id=predecessor_plan_id,
            background=background,
        )
        if not isinstance(result, dict):
            raise RuntimeError("Dashboard motion planning returned a non-object")
        return result

    def dashboard_execute_prepared_command(
        self,
        *,
        plan_id: str,
        command_id: str,
    ) -> dict[str, Any]:
        """Execute one prepared motion with an exact command-scoped permit."""

        if self._closed:
            raise RuntimeError("BEHAVIOR toolkit is closed")
        command_id = self._require_dashboard_command_permit(command_id)
        self._manual_intervention_latch.set()
        handler = getattr(
            self._primitives,
            "dashboard_execute_prepared_command",
            None,
        )
        if not callable(handler):
            raise RuntimeError("Dashboard prepared execution is unavailable")
        result = handler(plan_id=plan_id, command_id=command_id)
        if not isinstance(result, dict):
            raise RuntimeError("Dashboard prepared execution returned a non-object")
        receipt_binding = self._receipt_binding_from_result(result)
        if receipt_binding is not None:
            latch = self._success_latch
            if latch is not None and latch.observe(result):
                self._shared_success_evidence = self._success_evidence(result)
                self._official_task_success = True
        return result

    def dashboard_discard_prepared_command(
        self,
        *,
        plan_id: str,
    ) -> dict[str, Any]:
        """Discard one speculative prepared motion."""

        if self._closed:
            raise RuntimeError("BEHAVIOR toolkit is closed")
        handler = getattr(
            self._primitives,
            "dashboard_discard_prepared_command",
            None,
        )
        if not callable(handler):
            raise RuntimeError("Dashboard prepared-plan discard is unavailable")
        result = handler(plan_id=plan_id)
        if not isinstance(result, dict):
            raise RuntimeError("Dashboard plan discard returned a non-object")
        return result

    def dashboard_capture_views(
        self,
        *,
        command_id: str,
    ) -> dict[str, Any]:
        """Capture one full camera group under an exact manual permit."""

        if self._closed:
            raise RuntimeError("BEHAVIOR toolkit is closed")
        command_id = self._require_dashboard_command_permit(command_id)
        handler = getattr(self._primitives, "dashboard_capture_views", None)
        if not callable(handler):
            raise RuntimeError("Dashboard camera capture is unavailable")
        result = handler(command_id=command_id)
        if not isinstance(result, dict):
            raise RuntimeError("Dashboard camera capture returned a non-object")
        return result

    def dashboard_manual_command(
        self,
        *,
        target: str,
        action: str,
        camera: str,
        permit_command_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one controller-arbitrated manual primitive."""

        if self._closed:
            raise RuntimeError("BEHAVIOR toolkit is closed")
        arbiter = getattr(self, "_command_arbiter", None)
        require_permit = getattr(arbiter, "require_manual_permit", None)
        if not callable(require_permit):
            raise RuntimeError(
                "Dashboard manual primitive requires the shared command arbiter"
            )
        require_permit(str(permit_command_id or ""))
        if action != "observe":
            latch = getattr(self, "_manual_intervention_latch", None)
            if latch is None:
                latch = self.__dict__.setdefault(
                    "_manual_intervention_latch",
                    threading.Event(),
                )
            latch.set()
        handler = getattr(self._primitives, "dashboard_manual_command", None)
        if not callable(handler):
            raise RuntimeError("Dashboard manual primitive is unavailable")
        result = handler(target=target, action=action, camera=camera)
        if not isinstance(result, dict):
            raise RuntimeError("Dashboard manual primitive returned a non-object")
        receipt_binding = self._receipt_binding_from_result(result)
        if receipt_binding is not None:
            latch = self._success_latch
            if latch is not None and latch.observe(result):
                self._shared_success_evidence = self._success_evidence(result)
                self._official_task_success = True
        return result

    def _dashboard_progress(self, name: str, payload: dict[str, Any]) -> None:
        dashboard = self._dashboard
        callback = getattr(dashboard, "on_tool_progress", None)
        if callable(callback):
            public_payload = dict(payload)
            legacy_invocations = public_payload.pop("vla_invocations", None)
            if (
                "global_vla_invocations" not in public_payload
                and isinstance(legacy_invocations, int)
                and not isinstance(legacy_invocations, bool)
            ):
                public_payload["global_vla_invocations"] = legacy_invocations
            callback(name, public_payload)

    def _dashboard_start(self, name: str, input_dict: dict[str, Any]) -> None:
        dashboard = self._dashboard
        callback = getattr(dashboard, "on_tool_start", None)
        if callable(callback):
            callback(name, input_dict)

    def _budget_progress_payload(self) -> dict[str, Any]:
        """Return authoritative run-wide counters for every public result."""

        primitives = self._primitives
        return {
            "attempt_index": int(primitives.attempt_index),
            "attempt_nonce": primitives.attempt_nonce,
            "global_tool_calls": int(self._tool_calls),
            "max_tool_calls": int(self._max_tool_calls),
            "total_env_steps": int(primitives.total_env_steps),
            "max_episode_steps": int(primitives.max_episode_steps or 0),
            "global_vla_chunks": int(primitives._global_vla_chunks),
            "global_vla_invocations": int(primitives._vla_invocations),
            "elapsed_wall_clock_s": round(primitives.elapsed_wall_clock_s, 3),
            "max_wall_clock_s": float(primitives.max_wall_clock_s),
        }

    def _decorate_tool_result(
        self,
        name: str,
        input_dict: dict[str, Any],
        result: Any,
    ) -> Any:
        """Attach one budget snapshot before the unique dashboard callback."""

        del input_dict
        if not isinstance(result, dict):
            return result
        public_result = dict(result)
        # Normalize legacy primitive results to one run-wide public name.
        public_result.pop("vla_invocations", None)
        if name == "pi0_nav_pick":
            for field in _DEPRECATED_PI0_LIMIT_FIELDS:
                public_result.pop(field, None)
            if public_result.get("stop_reason") in _DEPRECATED_PI0_LIMIT_REASONS:
                public_result.pop("stop_reason", None)
                if (
                    public_result.get("runner_termination_reason")
                    == "attempt_budget_exhausted"
                ):
                    public_result.pop("runner_termination_reason", None)
                if (
                    public_result.get("_finish") is True
                    and public_result.get("task_success") is not True
                ):
                    public_result["_finish"] = False
        public_result.update(self._budget_progress_payload())
        return public_result

    @property
    def _recipe_tag(self) -> str:
        return self._publication_task_spec().tag(self._primitives.public_seed)

    def _publication_task_spec(self) -> BehaviorTaskSpec:
        """Resolve the task spec for runtime and legacy offline publishers."""

        task_spec = getattr(self, "_task_spec", None)
        if task_spec is None:
            task_spec = getattr(self._primitives, "task_spec", None)
        if task_spec is not None:
            return task_spec
        task_name = getattr(self._primitives, "task_name", None)
        if not isinstance(task_name, str) or not task_name:
            raise RuntimeError(
                "BEHAVIOR publication requires an explicit task_spec or task_name"
            )
        return get_task_spec(str(task_name))

    def runner_continuation_state(self) -> dict[str, Any]:
        """Return only structured facts trusted by the BEHAVIOR runner.

        Planner prose and ordinary primitive failures are deliberately absent.
        The runner combines this snapshot with its cumulative turn budget and
        an operator interrupt to decide whether another planner cycle is
        required.
        """

        primitives = self._primitives
        terminal_policy = self._task_spec.terminal_failure_policy
        exhausted_budgets: list[str] = []
        if self._tool_calls >= self._max_tool_calls:
            exhausted_budgets.append("max_tool_calls")
        if primitives.elapsed_wall_clock_s >= primitives.max_wall_clock_s:
            exhausted_budgets.append("max_wall_clock_s")
        if int(primitives.max_episode_steps or 0) > 0 and int(
            primitives.total_env_steps
        ) >= int(primitives.max_episode_steps):
            exhausted_budgets.append("max_episode_steps")
        runtime_unrecoverable = False
        visual_terminal_failure = False
        for record in reversed(self._tool_trace):
            result = record.get("result") if isinstance(record, dict) else None
            if not isinstance(result, dict):
                continue
            if (
                result.get("_finish") is True
                and result.get("runner_termination_reason")
                == "unrecoverable_infrastructure_termination"
            ):
                runtime_unrecoverable = True
            receipt = result.get("terminal_failure_receipt")
            if (
                terminal_policy is not None
                and result.get("_finish") is True
                and result.get("runner_termination_reason")
                == terminal_policy.runner_reason
                and result.get("task_success") is False
                and isinstance(receipt, dict)
                and receipt.get("source") == "llm_fresh_visual_observation"
                and receipt.get("condition") == terminal_policy.condition
                and receipt.get("task_success") is False
                and receipt.get("attempt_nonce") == primitives.attempt_nonce
                and receipt.get("attempt_index") == int(primitives.attempt_index)
            ):
                visual_terminal_failure = True
            if runtime_unrecoverable and visual_terminal_failure:
                break

        return {
            "raw_official_success_verified": self._has_verified_raw_success(),
            "visual_terminal_failure_verified": bool(
                self._terminal_failure_latched and visual_terminal_failure
            ),
            "visual_terminal_failure_reason": (
                terminal_policy.runner_reason
                if self._terminal_failure_latched
                and visual_terminal_failure
                and terminal_policy is not None
                else None
            ),
            "unrecoverable_infrastructure_termination": runtime_unrecoverable,
            "exhausted_budgets": exhausted_budgets,
            "attempt_index": int(primitives.attempt_index),
            "attempt_nonce": primitives.attempt_nonce,
            "run_nonce": primitives.run_nonce,
            "global_tool_calls": int(self._tool_calls),
            "total_env_steps": int(primitives.total_env_steps),
            "global_vla_chunks": int(primitives._global_vla_chunks),
            "global_vla_invocations": int(primitives._vla_invocations),
            "elapsed_wall_clock_s": round(primitives.elapsed_wall_clock_s, 3),
        }

    def _record_tool_result(
        self,
        name: str,
        input_dict: dict[str, Any],
        public_result: dict[str, Any],
    ) -> None:
        record = {
            "step": len(self._tool_trace) + 1,
            "tool": name,
            "input": self._artifact_value(input_dict),
            "result": self._artifact_value(public_result),
            "task_success": bool(public_result.get("task_success", False)),
            "attempt_index": int(self._primitives.attempt_index),
            "attempt_nonce": self._primitives.attempt_nonce,
            "global_tool_calls": int(self._tool_calls),
        }
        self._tool_trace.append(record)
        self._current_attempt_trace.append(record)
        self._write_json_atomic(
            self._primitives.output_dir / "behavior_tool_trace.jsonl",
            self._current_attempt_trace,
            json_lines=True,
        )
        self._write_json_atomic(
            self._primitives.output_dir / "traces" / f"{self._recipe_tag}.jsonl",
            self._tool_trace,
            json_lines=True,
        )

    def _has_verified_raw_success(self) -> bool:
        """Accept only a nonce-bound runtime raw-success receipt."""

        latch = getattr(self, "_success_latch", None)
        is_latched = getattr(latch, "is_latched", None)
        shared_evidence = getattr(self, "_shared_success_evidence", None)
        shared_success = bool(
            callable(is_latched)
            and is_latched()
            and isinstance(shared_evidence, dict)
            and self._receipt_binding_from_result(shared_evidence) is not None
        )
        return bool(
            shared_success
            or (
                self._official_task_success
                and any(
                    isinstance(record, dict)
                    and isinstance(record.get("result"), dict)
                    and record["result"].get("task_success") is True
                    and record["result"].get("official_success_source")
                    in _OFFICIAL_SUCCESS_SOURCES
                    and self._receipt_binding_from_result(record["result"]) is not None
                    for record in self._tool_trace
                )
            )
        )

    def _receipt_binding_from_result(
        self, result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Validate and reduce an immutable runtime success receipt."""

        receipt = result.get("official_success_receipt")
        if not isinstance(receipt, dict):
            for info_key in ("info", "last_info"):
                info = result.get(info_key)
                runtime = info.get("_rpent") if isinstance(info, dict) else None
                if isinstance(runtime, dict) and isinstance(
                    runtime.get("official_success_receipt"), dict
                ):
                    receipt = runtime["official_success_receipt"]
                    break
                monitor = (
                    runtime.get("pi0_nav_pick_monitor")
                    if isinstance(runtime, dict)
                    else None
                )
                if isinstance(monitor, dict) and isinstance(
                    monitor.get("official_success_receipt"), dict
                ):
                    receipt = monitor["official_success_receipt"]
                    break
        if not isinstance(receipt, dict):
            return None
        required = {
            "schema_version",
            "source",
            "run_nonce",
            "attempt_nonce",
            "attempt_index",
            "env_step",
            "raw_done",
            "receipt_sha256",
        }
        if set(receipt) != required:
            return None
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256", None)
        try:
            canonical = json.dumps(
                unsigned,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            return None
        run_nonce = receipt.get("run_nonce")
        attempt_nonce = receipt.get("attempt_nonce")
        attempt_index = receipt.get("attempt_index")
        env_step = receipt.get("env_step")
        raw_done = receipt.get("raw_done")
        if (
            type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1
            or receipt.get("source") != 'info["done"]["success"]'
            or not isinstance(raw_done, dict)
            or type(raw_done.get("success")) is not bool
            or raw_done["success"] is not True
            or not isinstance(run_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", run_nonce) is None
            or not isinstance(attempt_nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", attempt_nonce) is None
            or type(attempt_index) is not int
            or attempt_index < 1
            or type(env_step) is not int
            or env_step < 0
            or not isinstance(claimed, str)
            or re.fullmatch(r"[0-9a-f]{64}", claimed) is None
            or receipt.get("attempt_nonce")
            != getattr(self._primitives, "attempt_nonce", None)
            or receipt.get("attempt_index")
            != int(getattr(self._primitives, "attempt_index", -1))
            or (
                getattr(self._primitives, "run_nonce", None) is not None
                and receipt.get("run_nonce") != self._primitives.run_nonce
            )
            or not hmac.compare_digest(
                claimed,
                hashlib.sha256(canonical).hexdigest(),
            )
        ):
            return None
        return {
            "source": receipt["source"],
            "run_nonce": receipt["run_nonce"],
            "attempt_nonce": receipt["attempt_nonce"],
            "attempt_index": int(receipt["attempt_index"]),
            "env_step": int(receipt["env_step"]),
            "receipt_sha256": str(receipt["receipt_sha256"]),
        }

    @staticmethod
    def _success_evidence(result: dict[str, Any]) -> dict[str, Any]:
        """Copy only immutable receipt material from one verified result."""

        receipt = result.get("official_success_receipt")
        if not isinstance(receipt, dict):
            for info_key in ("info", "last_info"):
                info = result.get(info_key)
                runtime = info.get("_rpent") if isinstance(info, dict) else None
                if isinstance(runtime, dict) and isinstance(
                    runtime.get("official_success_receipt"), dict
                ):
                    receipt = runtime["official_success_receipt"]
                    break
                monitor = (
                    runtime.get("pi0_nav_pick_monitor")
                    if isinstance(runtime, dict)
                    else None
                )
                if isinstance(monitor, dict) and isinstance(
                    monitor.get("official_success_receipt"), dict
                ):
                    receipt = monitor["official_success_receipt"]
                    break
        if not isinstance(receipt, dict):
            raise RuntimeError("verified success result lost its immutable receipt")
        return {
            "task_success": True,
            "official_success_source": 'info["done"]["success"]',
            "official_success_receipt": dict(receipt),
        }

    @staticmethod
    def validate_symbolic_publication(records: list[dict[str, Any]]) -> None:
        """Reject run-specific geometry before publishing task-level guidance."""

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key).lower() in _FORBIDDEN_PUBLICATION_KEYS:
                        raise ValueError(f"publication contains forbidden field: {key}")
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(records)
        encoded = json.dumps(records, sort_keys=True, ensure_ascii=False)
        if re.search(r"/(?:home|tmp|mnt)/", encoded):
            raise ValueError("publication contains an absolute runtime path")
        lowered = encoded.lower()
        if re.search(
            r"\b(?:dynamic[- ]roles?|held[- ]roles?|press[- ]roles?|"
            r"held\s+and\s+press\s+roles?|press[- ]hand|"
            r"semantic[_ -]roles?|interaction[- ]roles?)\b",
            lowered,
        ):
            raise ValueError("publication contains deprecated hand-role language")
        for tool_name in BEHAVIOR_TOOL_NAMES:
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(tool_name.lower())}(?![a-z0-9_])",
                lowered,
            ):
                raise ValueError(
                    f"publication contains a fixed public tool name: {tool_name}"
                )
        if re.search(
            r"\b(?:first|second|third|next|then|finally|subsequently|afterwards)\b"
            r"|\bstep\s*(?:number\s*)?\d+\b",
            lowered,
        ):
            raise ValueError("publication contains an implicit capability sequence")
        if re.search(
            r"\b(?:chunks?\s*=\s*\d+|\d+\s+(?:complete\s+)?chunks?)\b",
            lowered,
        ):
            raise ValueError(
                "publication contains a fixed invocation-local chunk count"
            )
        for forbidden_phrase in (
            "first tool",
            "exactly once",
            "one permitted",
            "then call",
            "before finish",
        ):
            if forbidden_phrase in lowered:
                raise ValueError(
                    f"publication contains fixed ordering: {forbidden_phrase}"
                )

    @staticmethod
    def _public_outcome_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
        """Summarize outcome facts without replaying per-call hand assignments."""

        if not isinstance(record, dict):
            return None
        result = record.get("result")
        if not isinstance(result, dict):
            return None
        allowed = (
            "task_success",
            "primitive_success",
            "official_success_source",
            "runner_termination_reason",
            "stop_reason",
            "attempt_index",
            "total_env_steps",
        )
        return {key: result[key] for key in allowed if key in result}

    @staticmethod
    def _artifact_value(value: Any) -> Any:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"binary_omitted": True, "size_bytes": len(value)}
        if isinstance(value, dict):
            return {
                str(key): BehaviorToolkit._artifact_value(item)
                for key, item in value.items()
                if not str(key).startswith("_image")
            }
        if isinstance(value, (list, tuple)):
            return [BehaviorToolkit._artifact_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _write_json_atomic(path: Path, value: Any, *, json_lines: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        if json_lines:
            text = "".join(json.dumps(item, ensure_ascii=True) + "\n" for item in value)
        else:
            text = json.dumps(value, indent=2, ensure_ascii=True)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        # Admission, execution, terminal validation, and trace persistence are
        # one transaction. A concurrent waiter must re-evaluate raw success
        # only after the preceding call has fully persisted its terminal
        # result/receipt.
        execute_tool_lock = getattr(self, "_execute_tool_lock", None)
        if execute_tool_lock is None:
            # A few focused tests construct the facade with object.__new__.
            # Production instances always initialize this lock in __init__.
            execute_tool_lock = self.__dict__.setdefault(
                "_execute_tool_lock",
                threading.Lock(),
            )
        arbiter = getattr(self, "_command_arbiter", None)
        transaction = getattr(arbiter, "agent_transaction", None)
        transaction_context = (
            transaction() if callable(transaction) else contextlib.nullcontext()
        )
        with transaction_context:
            with execute_tool_lock:
                return self._execute_tool_locked(name, input_dict)

    def _execute_tool_locked(self, name: str, input_dict: dict[str, Any]):
        call_started = time.monotonic()
        rejected: list[str] = []
        precondition_error: str | None = None
        elapsed = self._primitives.elapsed_wall_clock_s
        verified_success = self._has_verified_raw_success()
        if name not in self._tools:
            rejected.append("unknown_tool")
        if self._tool_calls >= self._max_tool_calls:
            rejected.append("global_tool_call_budget_exhausted")
        if elapsed >= self._primitives.max_wall_clock_s:
            rejected.append("global_wall_clock_budget_exhausted")
        if verified_success:
            rejected.append("official_success_latched")
        if self._terminal_failure_latched:
            rejected.append("terminal_failure_latched")
        if not rejected and name in BEHAVIOR_TOOL_NAMES:
            try:
                rejected.extend(self._primitives.failed_preconditions(name, input_dict))
            except Exception as exc:
                rejected.append("precondition_check_error")
                precondition_error = type(exc).__name__
        rejected = [
            reason for reason in rejected if reason not in _DEPRECATED_PI0_LIMIT_REASONS
        ]
        if name in self._tools and self._tool_calls < self._max_tool_calls:
            self._tool_calls += 1
        if rejected:
            if name in self._tools:
                self._dashboard_start(name, input_dict)
            result = self._structured_rejection(
                name,
                input_dict,
                rejected,
                elapsed_s=time.monotonic() - call_started,
                precondition_error=precondition_error,
            )
            if name in self._tools:
                self._record_tool_result(name, input_dict, result.result)
            return result

        self._dashboard_start(name, input_dict)
        entry = self._tools.get(name)
        if entry is None:
            result = BehaviorToolResult(
                name=name,
                result={"error": f"unknown tool: {name}"},
            )
        else:
            handler = entry[1]
            try:
                payload = handler(**input_dict)
            except TypeError as exc:
                payload = {
                    "error": f"bad arguments for {name}: {exc}",
                    "got": input_dict,
                }
            except Exception as exc:
                payload = {"error": str(exc), "traceback": traceback.format_exc()}
            if isinstance(payload, dict):
                payload.setdefault(
                    "elapsed_s",
                    round(max(0.0, time.monotonic() - call_started), 6),
                )
            payload = self._decorate_tool_result(name, input_dict, payload)
            if isinstance(payload, dict):
                if payload.get("task_success") is True:
                    receipt_binding = self._receipt_binding_from_result(payload)
                    if receipt_binding is None:
                        # A positive primitive flag without the runtime-owned,
                        # nonce-bound raw-success receipt is not task success.
                        payload["task_success"] = False
                        payload["_finish"] = False
                        payload["official_success_receipt_valid"] = False
                        payload["stop_reason"] = (
                            "invalid_or_missing_official_success_receipt"
                        )
                    else:
                        payload["official_success_receipt_valid"] = True
                        latch = getattr(self, "_success_latch", None)
                        if latch is not None and latch.observe(payload):
                            self._shared_success_evidence = self._success_evidence(
                                payload
                            )
                        self._official_task_success = True
                self._last_tool_result = payload
                terminal_policy = self._task_spec.terminal_failure_policy
                if (
                    terminal_policy is not None
                    and name == "save_robot_state_checkpoint"
                    and payload.get("_finish") is True
                    and payload.get("task_success") is False
                    and payload.get("stop_reason") == terminal_policy.condition
                    and isinstance(payload.get("terminal_failure_receipt"), dict)
                    and payload["terminal_failure_receipt"].get("source")
                    == "llm_fresh_visual_observation"
                    and payload["terminal_failure_receipt"].get("condition")
                    == terminal_policy.condition
                    and payload["terminal_failure_receipt"].get("task_success") is False
                ):
                    self._terminal_failure_latched = True
            if self._dashboard is not None:
                self._dashboard.on_tool_result(name, payload)
            result = BehaviorToolResult(name=name, result=payload)
        public_result = result.result if isinstance(result.result, dict) else {}
        self._record_tool_result(name, input_dict, public_result)
        return result

    def _structured_rejection(
        self,
        name: str,
        input_dict: dict[str, Any],
        failed_preconditions: list[str],
        *,
        elapsed_s: float,
        precondition_error: str | None = None,
    ) -> ToolResult:
        """Return a side-effect-free, machine-readable guard rejection."""

        ordered_failures = [
            reason
            for reason in dict.fromkeys(failed_preconditions)
            if reason not in _DEPRECATED_PI0_LIMIT_REASONS
        ]
        verified_success = self._has_verified_raw_success()
        budget_exhausted = any(
            reason in _HARD_EXECUTION_BUDGET_PRECONDITIONS
            for reason in ordered_failures
        )
        payload = {
            "_finish": bool(
                verified_success or self._terminal_failure_latched or budget_exhausted
            ),
            "runner_termination_reason": (
                "official_task_success"
                if verified_success
                else (
                    self._task_spec.terminal_failure_policy.runner_reason
                    if self._task_spec.terminal_failure_policy is not None
                    else None
                )
                if self._terminal_failure_latched
                else "attempt_budget_exhausted"
                if budget_exhausted
                else None
            ),
            "name": name,
            "primitive_success": False,
            "task_success": verified_success,
            "stop_reason": "precondition_rejected",
            "official_success_source": 'info["done"]["success"]',
            "failed_preconditions": ordered_failures,
            "invalidated_receipts": [],
            "new_receipts": [],
            "elapsed_s": round(max(0.0, elapsed_s), 6),
            **self._budget_progress_payload(),
        }
        if precondition_error is not None:
            payload["precondition_error"] = precondition_error
        dashboard = self._dashboard
        callback = getattr(dashboard, "on_tool_result", None)
        if callable(callback):
            callback(name, payload)
        return BehaviorToolResult(name=name, result=payload)

    def close(self) -> None:
        close_lock = getattr(self, "_close_lock", None)
        if close_lock is None:
            close_lock = self.__dict__.setdefault("_close_lock", threading.RLock())
        with close_lock:
            if getattr(self, "_closed", False):
                return
            resource = getattr(self, "_runtime_resource", None)
            if resource is not None:
                resource.quiesce_control()
            else:
                controller = getattr(self, "_dashboard_controller", None)
                arbiter = getattr(self, "_command_arbiter", None)
                dashboard = getattr(self, "_dashboard", None)
                for owner in (controller, arbiter):
                    quiesce = getattr(owner, "quiesce", None)
                    if callable(quiesce):
                        quiesce()
                for label, owner in (
                    ("controller", controller),
                    ("arbiter", arbiter),
                ):
                    drain = getattr(owner, "drain", None)
                    if (
                        callable(drain)
                        and drain(BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S) is not True
                    ):
                        raise TimeoutError(
                            f"BEHAVIOR dashboard {label} did not drain before cleanup"
                        )
                unbind = getattr(dashboard, "unbind_controller", None)
                if callable(unbind) and controller is not None:
                    unbind(controller)
                close_controller = getattr(controller, "close", None)
                if callable(close_controller):
                    close_controller(timeout_s=BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S)
                close_arbiter = getattr(arbiter, "close", None)
                if callable(close_arbiter):
                    close_arbiter(timeout_s=BEHAVIOR_CONTROL_DRAIN_TIMEOUT_S)
            primitives = self._primitives
            model = getattr(primitives, "model", None)
            model_close = getattr(model, "close", None)
            if callable(model_close):
                model_close()
            env = getattr(primitives, "env", None)
            env_close = getattr(env, "close_transport", None)
            if resource is None and callable(env_close):
                env_close()
            self._closed = True

    def write_recipe(self, recipe_tag: str) -> str | None:
        if recipe_tag != self._recipe_tag:
            raise ValueError("recipe tag does not match the public seed")
        path = self._primitives.output_dir / f"recipe_{recipe_tag}.jsonl"
        final_record = self._tool_trace[-1] if self._tool_trace else None
        task_success = self._has_verified_raw_success()
        manual_latch = getattr(self, "_manual_intervention_latch", None)
        manual_intervention = bool(
            manual_latch is not None and manual_latch.is_set()
        )
        recipe_path = None
        publish_task_memory = bool(
            task_success
            and self._behavior_phase == "explore"
            and not manual_intervention
        )
        publication_reason = (
            "eligible_raw_success_explore"
            if publish_task_memory
            else "dashboard_manual_intervention"
            if manual_intervention
            else "official_task_success_not_verified"
            if not task_success
            else "phase_not_explore"
        )
        if publish_task_memory:
            symbolic_recipe = self._symbolic_recipe()
            self.validate_symbolic_publication(symbolic_recipe)
            self._write_json_atomic(path, symbolic_recipe, json_lines=True)
            recipe_path = str(path)
        result = {
            "success": task_success,
            "task_success": task_success,
            "publication_eligible": publish_task_memory,
            "manual_intervention": manual_intervention,
            "publication_reason": publication_reason,
            "publication_source": (
                RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE if publish_task_memory else None
            ),
            "official_success_source": 'info["done"]["success"] via task_success',
            "tool_calls": len(self._tool_trace),
            "global_vla_invocations": int(self._primitives._vla_invocations),
            "tool_trace_path": str(
                self._primitives.output_dir / "behavior_tool_trace.jsonl"
            ),
            "recipe_path": recipe_path,
            "last_tool": self._public_outcome_summary(final_record),
        }
        self._write_json_atomic(
            self._primitives.output_dir / "behavior_result.json",
            result,
        )
        audit = {
            "schema_version": 1,
            "tag": recipe_tag,
            "phase": self._behavior_phase,
            "task_success": task_success,
            "publication_eligible": publish_task_memory,
            "manual_intervention": manual_intervention,
            "publication_reason": publication_reason,
            "publication_source": (
                RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE if publish_task_memory else None
            ),
            "official_success_source": 'info["done"]["success"]',
            "attempts_used": int(self._primitives.attempt_index),
            "global_tool_calls": int(self._tool_calls),
            "global_vla_chunks": int(self._primitives._global_vla_chunks),
            "global_vla_invocations": int(self._primitives._vla_invocations),
            "total_env_steps": int(self._primitives.total_env_steps),
            "recipe_path": recipe_path,
        }
        self._write_json_atomic(
            self._primitives.output_dir / f"{recipe_tag}.json",
            audit,
        )
        # Canonical Task Memory and provenance are published by the outer
        # harness only after it can bind all six immutable source artifacts.
        self._seal_current_attempt(result=result)
        return recipe_path

    def _symbolic_recipe(self) -> list[dict[str, Any]]:
        task_spec = self._publication_task_spec()
        successful_attempt = max(
            (
                int(record.get("attempt_index", self._primitives.attempt_index))
                for record in self._tool_trace
                if record.get("task_success") is True
            ),
            default=int(self._primitives.attempt_index),
        )
        failure_categories: set[str] = set()
        for record in self._tool_trace:
            if (
                int(record.get("attempt_index", successful_attempt))
                >= successful_attempt
                or record.get("result", {}).get("primitive_success") is not False
            ):
                continue
            reason = str(record.get("result", {}).get("stop_reason", "")).lower()
            if any(token in reason for token in ("stale", "frame", "projection")):
                failure_categories.add("stale_perception_or_projection_evidence")
            elif any(token in reason for token in ("grasp", "held", "attachment")):
                failure_categories.add("object_control_not_established")
            elif "budget" in reason and "vla_chunk" not in reason:
                failure_categories.add("attempt_budget_exhausted")
            elif reason:
                failure_categories.add("action_attempt_did_not_advance_task")
        return [
            {
                "schema_version": 1,
                "kind": "task_level_symbolic_recipe",
                "task": task_spec.task_name,
                "source": RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE,
                "policy": (
                    "Choose capabilities independently from fresh public evidence and "
                    "runtime receipts. Do not replay a run-specific trace."
                ),
            },
            {
                "kind": "semantic_goal",
                "goal": task_spec.task_language,
            },
            {
                "kind": "evidence_contract",
                "requirements": (
                    "Fresh evidence selects the anatomical hand for each analytic action. "
                    "Semantic identity, object control, the current attachment state of "
                    "each hand, target projection, and contact geometry remain current, "
                    "public, and bound to the active episode."
                ),
            },
            {
                "kind": "safety_contract",
                "requirements": (
                    "Action capabilities attempt requested execution directly. The Agent "
                    "chooses subsequent actions from fresh public evidence and returned "
                    "feedback; runtime lifecycle, evidence-lineage, the independently "
                    "observed attachment identity of each hand, capability-specific state "
                    "requirements for the selected anatomical hand, finite budgets, "
                    "raw-success latching, and any terminal policy explicitly registered "
                    "for this task remain authoritative."
                ),
            },
            {
                "kind": "official_success_observation",
                "outcome": (
                    "Runtime-owned raw task success was observed during embodied "
                    "interaction."
                ),
            },
            {
                "kind": "failed_attempt_evidence",
                "categories": sorted(failure_categories),
                "lesson": "Re-ground evidence and revise the failed semantic hypothesis.",
            },
        ]

    def _publish_task_memory(
        self,
        *,
        recipe_tag: str,
        recipe_path: Path,
        official_success_receipt: dict[str, Any],
        source_artifacts_sha256: dict[str, str],
    ) -> None:
        """Publish frozen task memory from fully sealed Harness evidence."""

        task_spec = self._publication_task_spec()
        public_seed = int(getattr(self._primitives, "public_seed", 0))
        if recipe_tag != task_spec.tag(public_seed):
            raise ValueError("publication recipe tag does not match the task identity")
        if set(source_artifacts_sha256) != _PUBLICATION_SOURCE_ARTIFACTS:
            raise ValueError("publication requires exactly six source artifact hashes")
        if any(
            not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in source_artifacts_sha256.values()
        ):
            raise ValueError("publication source artifact hash is invalid")
        expected_receipt_fields = {
            "source",
            "run_nonce",
            "attempt_nonce",
            "attempt_index",
            "env_step",
            "receipt_sha256",
            "file_sha256",
        }
        if set(official_success_receipt) != expected_receipt_fields:
            raise ValueError("official success receipt binding schema mismatch")
        if (
            official_success_receipt.get("source") != 'info["done"]["success"]'
            or official_success_receipt.get("attempt_nonce")
            != getattr(self._primitives, "attempt_nonce", None)
            or official_success_receipt.get("attempt_index")
            != int(getattr(self._primitives, "attempt_index", -1))
            or not isinstance(official_success_receipt.get("run_nonce"), str)
            or not official_success_receipt["run_nonce"]
            or not isinstance(official_success_receipt.get("env_step"), int)
            or isinstance(official_success_receipt.get("env_step"), bool)
            or official_success_receipt["env_step"] < 0
            or any(
                re.fullmatch(r"[0-9a-f]{64}", str(official_success_receipt[key]))
                is None
                for key in ("receipt_sha256", "file_sha256")
            )
            or source_artifacts_sha256["official_success_receipt"]
            != official_success_receipt["file_sha256"]
        ):
            raise ValueError("official success receipt binding is invalid")
        memory_dir = self._primitives.output_dir / "memory"
        memory_path = memory_dir / f"{task_spec.task_name}.md"
        recipe_records = [
            json.loads(line)
            for line in recipe_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        failures = next(
            (
                record
                for record in recipe_records
                if record.get("kind") == "failed_attempt_evidence"
            ),
            {},
        )
        memory_text = """# {task_name} task memory

- Re-localize from fresh public RGB-D in every episode; no pose, pixel, frame, checkpoint, native instance, camera schedule, or physical-side assignment is transferable.
- Select the anatomical hand for each analytic action from fresh current head RGB-D evidence; never inherit a hand assignment from task prior or history.
- Evaluate the current attachment identity and gripper state of the left and right hands independently; both hands may carry attachments, and capability-specific requirements apply only to the freshly selected anatomical hand.
- Use current public evidence to judge each precision action. Fresh projection receipts and current per-hand attachment state are required where the capability calls for them.
- Action capabilities attempt requested execution directly; no planner collision, clearance, joint-margin, or dynamics safety certificate is provided. Use fresh visual evidence and returned feedback when choosing an action.
- Capability choice, model instruction, invocation count, and camera choice remain decisions for the current episode; recorded success does not prescribe them.
- A failed Explore attempt may be retried only by the outer harness in a fresh environment; Eval never retries.
- Only raw `info[\"done\"][\"success\"]` is task success.
- Attempt lifecycle, finite budgets, raw-success latching, and the selected task's registered terminal policy remain authoritative.
{task_specific_memory}

## Explore-validated experience

- Failure categories: {failure_categories}

These observations are advisory. Fresh evidence and runtime lifecycle, hand-state, budget, and terminal constraints remain authoritative.
""".format(
            task_name=task_spec.task_name,
            task_specific_memory=(
                "- A valid button-face hypothesis combines the red front face, "
                "black round or oval disk, white outer ring, and red center bump.\n"
                "- A fresh visually verified radio-tipped-flat terminal latch is "
                "authoritative."
                if task_spec.surface_review_policy is not None
                else "- Completion requires raw official success for the entire "
                "multi-object task; apparent placement of one item is not task success."
            ),
            failure_categories=json.dumps(
                failures.get("categories", []), ensure_ascii=False
            ),
        )
        if re.search(r"/(?:home|tmp|mnt)/", memory_text):
            raise ValueError("task memory contains an absolute runtime path")
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(memory_text, encoding="utf-8")
        provenance = {
            "schema_version": 2,
            "task": task_spec.task_name,
            "task_index": task_spec.task_index,
            "activity_definition_id": task_spec.activity_definition_id,
            "activity_instance_id": task_spec.instance_for_public_seed(public_seed),
            "public_seed": public_seed,
            "source": RAW_OFFICIAL_SUCCESS_PUBLICATION_SOURCE,
            "source_tag": recipe_tag,
            "success_source": 'info["done"]["success"]',
            "job_id": getattr(self._primitives, "job_id", None),
            "attempt_index": int(self._primitives.attempt_index),
            "attempt_nonce": getattr(self._primitives, "attempt_nonce", None),
            "task_success": True,
            "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            "memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
            "official_success_receipt": dict(official_success_receipt),
            "source_artifacts_sha256": dict(source_artifacts_sha256),
        }
        self._write_json_atomic(
            memory_dir / f"{task_spec.task_name}_provenance.json",
            provenance,
        )

    def _seal_current_attempt(self, *, result: dict[str, Any]) -> None:
        attempt_root = self._primitives._attempt_root()
        attempt_root.mkdir(parents=True, exist_ok=True)
        for name in (
            "run_manifest.json",
            "episode.mp4",
            "behavior_action_trace.jsonl",
            "behavior_tool_trace.jsonl",
        ):
            source = self._primitives.output_dir / name
            target = attempt_root / name
            if source.is_file() and source.resolve() != target.resolve():
                shutil.copy2(source, target)
        self._write_json_atomic(attempt_root / "behavior_result.json", result)
