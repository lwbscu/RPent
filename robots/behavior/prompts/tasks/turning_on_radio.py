"""Task-specific prompt nodes for ``turning_on_radio``."""

from __future__ import annotations

from typing import Final

from rpent.context.prompt_utils import PromptNode

PROMPT_PROFILE_ID: Final = "turning_on_radio"

TURNING_ON_RADIO_PROMPT_NODES: Final[PromptNode] = (
    """Prompt profile: `turning_on_radio`. This profile adds no extra
    task-specific manipulation ordering. Use the selected target prior,
    reviewed task knowledge, current public evidence, capability schemas, and
    runtime guards.""",
)

RADIO_PROMPT_NODES: Final[PromptNode] = TURNING_ON_RADIO_PROMPT_NODES

__all__ = [
    "PROMPT_PROFILE_ID",
    "RADIO_PROMPT_NODES",
    "TURNING_ON_RADIO_PROMPT_NODES",
]
