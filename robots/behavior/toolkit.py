"""BEHAVIOR toolkit with isolated and hybrid public primitive surfaces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robots.behavior.schemas import (
    DECLARE_BUTTON_VISIBILITY_SPEC,
    EVALUATE_PREPRESS_GEOMETRY_SPEC,
    FULL_TASK_VLA_MODE,
    HYBRID_TOOL_NAMES,
    HYBRID_VLM_PI0_MODE,
    INSPECT_POST_PICK_STATE_SPEC,
    PI0_NAV_PICK_SPEC,
    PI0_NAV_PICK_VLA_MODE,
    PI0_NAVIGATE_TO_SPEC,
    PI0_PICK_SPEC,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOL_SPECS,
    PLANNER_TOOLS_MODE,
    POST_PICK_OBSERVE_SPEC,
    POST_PICK_PIXEL_TO_WORLD_SPEC,
    POST_PICK_RESTORE_ROBOT_STATE_CHECKPOINT_SPEC,
    POST_PICK_SAVE_ROBOT_STATE_CHECKPOINT_SPEC,
    POST_PICK_TOOL_NAMES,
    PREPRESS_MOVE_TO_SPEC,
    PREPRESS_ROTATE_WRIST_SPEC,
    RESTORE_ROBOT_STATE_CHECKPOINT_SPEC,
    RUN_FULL_TASK_SPEC,
    SAVE_ROBOT_STATE_CHECKPOINT_SPEC,
)
from robots.behavior.tools import BehaviorPrimitives
from rpent.tools.toolkit import Toolkit


class BehaviorToolkit(Toolkit):
    """Register exactly one closed BEHAVIOR control surface per runtime."""

    def __init__(
        self,
        *,
        control_mode: str = FULL_TASK_VLA_MODE,
        primitives_kwargs: dict[str, Any] | None = None,
        planner_client: Any = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        # BEHAVIOR is a deliberately closed control surface.  The generic
        # read/write/list/finish tools would let a VLM bypass the synchronized
        # RGB-D + proprio boundary and would also violate full_task_vla's
        # single-tool contract.
        self._tools.clear()
        self.control_mode = control_mode
        self._primitives: BehaviorPrimitives | None = None
        self._planner_client = planner_client
        self._hybrid_trace: list[dict[str, Any]] = []
        if control_mode == FULL_TASK_VLA_MODE:
            if primitives_kwargs is None:
                raise ValueError("full_task_vla mode requires primitives_kwargs")
            self._primitives = BehaviorPrimitives(**primitives_kwargs)
            self.add_tool(
                "run_full_task",
                RUN_FULL_TASK_SPEC,
                self._primitives.run_full_task,
            )
        elif control_mode == PLANNER_TOOLS_MODE:
            if planner_client is None:
                raise ValueError("planner_tools mode requires planner_client")
            self._primitives = BehaviorPrimitives(planner_backend=planner_client)
            for name in PLANNER_TOOL_NAMES:
                self.add_tool(
                    name,
                    PLANNER_TOOL_SPECS[name],
                    getattr(self._primitives, name),
                )
        elif control_mode == PI0_PICK_VLA_MODE:
            if primitives_kwargs is None:
                raise ValueError("pi0_pick_vla mode requires primitives_kwargs")
            self._primitives = BehaviorPrimitives(**primitives_kwargs)
            self.add_tool(
                "pi0_pick",
                PI0_PICK_SPEC,
                self._primitives.pi0_pick,
            )
        elif control_mode == PI0_NAV_PICK_VLA_MODE:
            if primitives_kwargs is None:
                raise ValueError(
                    "pi0_nav_pick_vla mode requires primitives_kwargs"
                )
            self._primitives = BehaviorPrimitives(**primitives_kwargs)
            self.add_tool(
                "pi0_nav_pick",
                PI0_NAV_PICK_SPEC,
                self._primitives.pi0_nav_pick,
            )
            post_pick_specs = {
                "inspect_post_pick_state": INSPECT_POST_PICK_STATE_SPEC,
                "observe": POST_PICK_OBSERVE_SPEC,
                "declare_button_visibility": DECLARE_BUTTON_VISIBILITY_SPEC,
                "pixel_to_world": POST_PICK_PIXEL_TO_WORLD_SPEC,
                "evaluate_prepress_geometry": EVALUATE_PREPRESS_GEOMETRY_SPEC,
                "move_to": PREPRESS_MOVE_TO_SPEC,
                "rotate_wrist": PREPRESS_ROTATE_WRIST_SPEC,
                "save_robot_state_checkpoint": (
                    POST_PICK_SAVE_ROBOT_STATE_CHECKPOINT_SPEC
                ),
                "restore_robot_state_checkpoint": (
                    POST_PICK_RESTORE_ROBOT_STATE_CHECKPOINT_SPEC
                ),
            }
            if tuple(post_pick_specs) != POST_PICK_TOOL_NAMES:
                raise RuntimeError("post-pick schema order mismatch")
            for name, spec in post_pick_specs.items():
                handler = {
                    "move_to": self._primitives.prepress_move_to,
                    "rotate_wrist": self._primitives.prepress_rotate_wrist,
                    "save_robot_state_checkpoint": (
                        self._primitives.save_post_pick_robot_state_checkpoint
                    ),
                }.get(name, getattr(self._primitives, name))
                self.add_tool(name, spec, handler)
        elif control_mode == HYBRID_VLM_PI0_MODE:
            if primitives_kwargs is None or planner_client is None:
                raise ValueError(
                    "hybrid_vlm_pi0 mode requires primitives_kwargs and planner_client"
                )
            hybrid_kwargs = dict(primitives_kwargs)
            hybrid_kwargs.update(
                {
                    "planner_backend": planner_client,
                    "hybrid_mode": True,
                }
            )
            self._primitives = BehaviorPrimitives(**hybrid_kwargs)
            for name in HYBRID_TOOL_NAMES:
                if name == "pi0_pick":
                    spec = PI0_PICK_SPEC
                elif name == "pi0_navigate_to":
                    spec = PI0_NAVIGATE_TO_SPEC
                elif name == "save_robot_state_checkpoint":
                    spec = SAVE_ROBOT_STATE_CHECKPOINT_SPEC
                elif name == "restore_robot_state_checkpoint":
                    spec = RESTORE_ROBOT_STATE_CHECKPOINT_SPEC
                else:
                    spec = PLANNER_TOOL_SPECS[name]
                self.add_tool(name, spec, getattr(self._primitives, name))
        else:
            raise ValueError(f"unknown BEHAVIOR control mode: {control_mode}")

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
            text = "".join(
                json.dumps(item, ensure_ascii=True) + "\n" for item in value
            )
        else:
            text = json.dumps(value, indent=2, ensure_ascii=True)
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def execute_tool(self, name: str, input_dict: dict[str, Any]):
        result = super().execute_tool(name, input_dict)
        if self.control_mode in {HYBRID_VLM_PI0_MODE, PI0_NAV_PICK_VLA_MODE}:
            record = {
                "step": len(self._hybrid_trace) + 1,
                "tool": name,
                "input": self._artifact_value(input_dict),
                "result": self._artifact_value(result.result),
                "task_success": bool(
                    isinstance(result.result, dict)
                    and result.result.get("task_success", False)
                ),
            }
            self._hybrid_trace.append(record)
            assert self._primitives is not None
            trace_name = (
                "hybrid_tool_trace.jsonl"
                if self.control_mode == HYBRID_VLM_PI0_MODE
                else "pi0_nav_pick_tool_trace.jsonl"
            )
            self._write_json_atomic(
                self._primitives.output_dir / trace_name,
                self._hybrid_trace,
                json_lines=True,
            )
        return result

    def close(self) -> None:
        primitives = self._primitives
        clients = (
            getattr(primitives, "model", None),
            getattr(primitives, "env", None),
            self._planner_client,
        )
        seen: set[int] = set()
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def write_recipe(self, recipe_tag: str) -> str | None:
        if self.control_mode in {HYBRID_VLM_PI0_MODE, PI0_NAV_PICK_VLA_MODE}:
            assert self._primitives is not None
            path = self._primitives.output_dir / f"recipe_{recipe_tag}.jsonl"
            self._write_json_atomic(path, self._hybrid_trace, json_lines=True)
            final_record = self._hybrid_trace[-1] if self._hybrid_trace else None
            task_success = bool(
                isinstance(final_record, dict)
                and final_record.get("task_success", False)
            )
            trace_name = (
                "hybrid_tool_trace.jsonl"
                if self.control_mode == HYBRID_VLM_PI0_MODE
                else "pi0_nav_pick_tool_trace.jsonl"
            )
            mode_result = {
                "control_mode": self.control_mode,
                "success": task_success,
                "task_success": task_success,
                "official_success_source": 'info["done"]["success"] via task_success',
                "tool_calls": len(self._hybrid_trace),
                "tool_trace_path": str(
                    self._primitives.output_dir / trace_name
                ),
                "recipe_path": str(path),
                "last_tool": final_record,
            }
            self._write_json_atomic(
                self._primitives.output_dir
                / (
                    "hybrid_result.json"
                    if self.control_mode == HYBRID_VLM_PI0_MODE
                    else "pi0_nav_pick_mode_result.json"
                ),
                mode_result,
            )
            return str(path)
        if self.control_mode == PLANNER_TOOLS_MODE or self._primitives is None:
            # Planner primitives are env-side closed-loop commands; replay
            # requires the env trace emitted by the executor, not a VLA recipe.
            return None
        path = self._primitives.output_dir / f"recipe_{recipe_tag}.jsonl"
        tool_name = {
            FULL_TASK_VLA_MODE: "run_full_task",
            PI0_PICK_VLA_MODE: "pi0_pick",
            PI0_NAV_PICK_VLA_MODE: "pi0_nav_pick",
        }[self.control_mode]
        result = self._primitives.last_result
        tool_input = (
            {}
            if tool_name == "run_full_task"
            else {
                key: result.get(key)
                for key in (
                    "instruction",
                    "hand",
                    "max_chunks",
                    "gripper_closed_threshold",
                    "required_closed_chunks",
                    "stop_on_closure_candidate",
                    "post_candidate_chunks",
                )
                if isinstance(result, dict) and key in result
            }
        )
        record = {
            "step": 1,
            "tool": tool_name,
            "input": tool_input,
            "result": result,
        }
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return str(path)
