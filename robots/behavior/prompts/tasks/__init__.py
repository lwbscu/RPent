"""Closed registry of task-specific BEHAVIOR prompt profiles."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from robots.behavior.prompts.tasks.picking_up_trash import (
    PICKING_UP_TRASH_PROMPT_NODES,
)
from robots.behavior.prompts.tasks.picking_up_trash import (
    PROMPT_PROFILE_ID as PICKING_UP_TRASH_PROMPT_PROFILE_ID,
)
from robots.behavior.prompts.tasks.turning_on_radio import (
    PROMPT_PROFILE_ID as TURNING_ON_RADIO_PROMPT_PROFILE_ID,
)
from robots.behavior.prompts.tasks.turning_on_radio import (
    TURNING_ON_RADIO_PROMPT_NODES,
)
from rpent.context.prompt_utils import PromptNode

TASK_PROMPT_PROFILES: Final[Mapping[str, PromptNode]] = MappingProxyType(
    {
        TURNING_ON_RADIO_PROMPT_PROFILE_ID: TURNING_ON_RADIO_PROMPT_NODES,
        PICKING_UP_TRASH_PROMPT_PROFILE_ID: PICKING_UP_TRASH_PROMPT_NODES,
    }
)


def get_task_prompt_profile(prompt_profile_id: str) -> PromptNode:
    """Return one registered task prompt profile, failing closed if unknown."""

    try:
        return TASK_PROMPT_PROFILES[prompt_profile_id]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"unsupported BEHAVIOR prompt profile: {prompt_profile_id!r}"
        ) from error


__all__ = [
    "TASK_PROMPT_PROFILES",
    "get_task_prompt_profile",
]
