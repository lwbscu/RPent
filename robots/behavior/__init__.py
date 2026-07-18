"""BEHAVIOR environment plugin."""
from __future__ import annotations

from typing import Any

from robots.behavior.prompt_bundle import system_prompt, user_prompt
from rpent.envs.env_spec import EnvSpec
from rpent.envs.prompt_bundle import PromptBundle


def get_env_spec() -> EnvSpec:
    """Return BEHAVIOR identity and prompts."""
    return EnvSpec(
        name="behavior",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
    )


def get_toolkit(
    *,
    runner_kwargs: dict[str, Any],
    dashboard: Any = None,
):
    """Return common RPent tools plus the single BEHAVIOR full-task tool."""
    from robots.behavior.toolkit import BehaviorToolkit

    return BehaviorToolkit(runner_kwargs=runner_kwargs, dashboard=dashboard)


def get_runtime_provider():
    """Return the BEHAVIOR CLI/runtime provider."""
    from robots.behavior.runtime_provider import BehaviorRuntimeProvider

    return BehaviorRuntimeProvider()


__all__ = ["get_env_spec", "get_runtime_provider", "get_toolkit"]
