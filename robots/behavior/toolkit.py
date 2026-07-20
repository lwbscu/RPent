"""BEHAVIOR toolkit with three mutually isolated public primitive surfaces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robots.behavior.schemas import (
    FULL_TASK_VLA_MODE,
    PI0_PICK_SPEC,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOL_SPECS,
    PLANNER_TOOLS_MODE,
    RUN_FULL_TASK_SPEC,
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
        else:
            raise ValueError(f"unknown BEHAVIOR control mode: {control_mode}")

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
        if self.control_mode == PLANNER_TOOLS_MODE or self._primitives is None:
            # Planner primitives are env-side closed-loop commands; replay
            # requires the env trace emitted by the executor, not a VLA recipe.
            return None
        path = self._primitives.output_dir / f"recipe_{recipe_tag}.jsonl"
        tool_name = (
            "run_full_task"
            if self.control_mode == FULL_TASK_VLA_MODE
            else "pi0_pick"
        )
        result = self._primitives.last_result
        tool_input = (
            {}
            if tool_name == "run_full_task"
            else {
                key: result.get(key)
                for key in (
                    "hand",
                    "instruction",
                    "max_chunks",
                    "gripper_closed_threshold",
                    "required_closed_chunks",
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
