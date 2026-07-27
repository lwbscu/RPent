"""BEHAVIOR environment plugin."""

from __future__ import annotations

from typing import Any

from robots.behavior.prompt_bundle import system_prompt, user_prompt
from robots.behavior.spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


def get_env_spec() -> EnvSpec:
    """Return BEHAVIOR identity, prompts, and shared-runner hooks."""
    from robots.behavior.runtime import (
        RESOURCE_POLICY,
        add_cli_args,
        finalize_run,
        init_runtime,
        parse_config,
        prepare_resources,
        run_planner,
    )

    return EnvSpec(
        name="behavior",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=add_cli_args,
        prepare_resources=prepare_resources,
        parse_config=parse_config,
        init_runtime=init_runtime,
        resource_policy=RESOURCE_POLICY,
        run_planner=run_planner,
        finalize_run=finalize_run,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the single sequential BEHAVIOR toolkit."""
    from robots.behavior.toolkit import BehaviorToolkit

    return BehaviorToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
        dashboard=dashboard,
    )


__all__ = ["get_env_spec", "get_toolkit"]
