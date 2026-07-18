"""BEHAVIOR toolkit with one environment-specific command."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robots.behavior.schemas import RUN_FULL_TASK_SPEC
from robots.behavior.tools import FullTaskRunner
from rpent.tools.toolkit import Toolkit


class BehaviorToolkit(Toolkit):
    """Common RPent tools plus ``run_full_task``."""

    def __init__(
        self,
        *,
        runner_kwargs: dict[str, Any],
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self._runner = FullTaskRunner(**runner_kwargs)
        self.add_tool(
            "run_full_task",
            RUN_FULL_TASK_SPEC,
            self._runner.run_full_task,
        )

    def close(self) -> None:
        for client in (self._runner.model, self._runner.env):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def write_recipe(self, recipe_tag: str) -> str:
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
