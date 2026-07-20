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
    control_mode: str = "full_task_vla",
    primitives_kwargs: dict[str, Any] | None = None,
    planner_client: Any = None,
    dashboard: Any = None,
):
    """Return the selected closed BEHAVIOR control surface."""
    from robots.behavior.toolkit import BehaviorToolkit

    return BehaviorToolkit(
        control_mode=control_mode,
        primitives_kwargs=primitives_kwargs,
        planner_client=planner_client,
        dashboard=dashboard,
    )


def get_runtime_provider():
    """Return the BEHAVIOR CLI/runtime provider."""
    from robots.behavior.runtime_provider import BehaviorRuntimeProvider

    return BehaviorRuntimeProvider()


__all__ = ["get_env_spec", "get_runtime_provider", "get_toolkit"]
