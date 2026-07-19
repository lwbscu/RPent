"""BEHAVIOR toolkit with one environment-specific command."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robots.behavior.schemas import (
    FULL_TASK_VLA_MODE,
    PLANNER_TOOL_NAMES,
    PLANNER_TOOL_SPECS,
    PLANNER_TOOLS_MODE,
    RUN_FULL_TASK_SPEC,
)
from robots.behavior.tools import FullTaskRunner
from rpent.tools.toolkit import Toolkit


class BehaviorToolkit(Toolkit):
    """Common RPent tools plus one BEHAVIOR control surface."""

    def __init__(
        self,
        *,
        control_mode: str = FULL_TASK_VLA_MODE,
        runner_kwargs: dict[str, Any] | None = None,
        planner_client: Any = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self.control_mode = control_mode
        self._runner: FullTaskRunner | None = None
        self._planner_client = planner_client
        if control_mode == FULL_TASK_VLA_MODE:
            if runner_kwargs is None:
                raise ValueError("full_task_vla mode requires runner_kwargs")
            self._runner = FullTaskRunner(**runner_kwargs)
            self.add_tool(
                "run_full_task",
                RUN_FULL_TASK_SPEC,
                self._runner.run_full_task,
            )
        elif control_mode == PLANNER_TOOLS_MODE:
            if planner_client is None:
                raise ValueError("planner_tools mode requires planner_client")
            for name in PLANNER_TOOL_NAMES:
                self.add_tool(
                    name,
                    PLANNER_TOOL_SPECS[name],
                    getattr(planner_client, name),
                )
        else:
            raise ValueError(f"unknown BEHAVIOR control mode: {control_mode}")

    def close(self) -> None:
        if self._runner is not None:
            clients = (self._runner.model, self._runner.env)
        else:
            clients = (self._planner_client,)
        for client in clients:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def write_recipe(self, recipe_tag: str) -> str | None:
        if self._runner is None:
            # Planner primitives are env-side closed-loop commands; replay
            # requires the env trace emitted by the executor, not a VLA recipe.
            return None
        path = self._runner.output_dir / f"recipe_{recipe_tag}.jsonl"
        record = {
            "step": 1,
            "tool": "run_full_task",
            "input": {},
            "result": self._runner.last_result,
        }
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return str(path)
