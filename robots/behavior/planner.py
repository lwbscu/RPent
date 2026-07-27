"""BEHAVIOR-only planner construction on top of the official RPent API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from robots.behavior.codex_planner import CodexPlanner
from rpent.planner.base import build_planner as build_official_planner


def build_behavior_planner(
    planner_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    base_url: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 8192,
    planner_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    dashboard: Any = None,
    no_images: bool = False,
):
    """Build a planner while keeping BEHAVIOR policy out of public RPent."""

    if planner_type == "codex":
        timeout_s = planner_timeout_s
        if timeout_s is None:
            timeout_s = int(
                os.environ.get(
                    "CODEX_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        output_root = Path(output_dir)
        return CodexPlanner(
            output_dir=output_root,
            repo_root=output_root,
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            timeout_s=timeout_s,
            extra_dirs=[],
            output_path=output_root / f"codex_{recipe_tag}.txt",
            dashboard=dashboard,
            tool_only=True,
        )
    return build_official_planner(
        planner_type,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name="behavior",
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        planner_timeout_s=planner_timeout_s,
        claude_code_max_budget_usd=claude_code_max_budget_usd,
        dashboard=dashboard,
        no_images=no_images,
    )


__all__ = ["build_behavior_planner"]
