"""Prompts for isolated full-task, planner, and local-VLA BEHAVIOR modes."""
from __future__ import annotations

import json
from pathlib import Path

from robots.behavior.schemas import (
    HYBRID_VLM_PI0_MODE,
    PI0_NAV_PICK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOLS_MODE,
)
from rpent.context.prompt_utils import PromptNode

_PROMPT_DIR = Path(__file__).with_name("prompts")


def _load_prompt(filename: str) -> str:
    """Load one checked-in BEHAVIOR prompt fragment."""
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8").strip()


def _render_prompt(filename: str, **values: object) -> str:
    """Render the small set of explicit placeholders in a prompt file."""
    prompt = _load_prompt(filename)
    for name, value in values.items():
        placeholder = "{{ " + name + " }}"
        if placeholder not in prompt:
            raise ValueError(f"missing prompt placeholder {placeholder!r} in {filename}")
        prompt = prompt.replace(placeholder, str(value))
    if "{{" in prompt or "}}" in prompt:
        raise ValueError(f"unrendered prompt placeholder remains in {filename}")
    return prompt


def system_prompt() -> PromptNode:
    return """You control a BEHAVIOR R1Pro evaluation through RPent.

{{ behavior_system_instructions }}
"""


def user_prompt() -> PromptNode:
    return """Run BEHAVIOR task {{ task_name }} (task {{ task }}, definition
{{ activity_definition_id }}, public instance {{ activity_instance_id }}, seed
{{ seed }}) in {{ behavior_control_mode }} mode. The configured horizon is
{{ max_episode_steps }} environment steps.

{{ behavior_user_instructions }}
"""


FULL_TASK_SYSTEM_INSTRUCTIONS = _load_prompt("full_task_system.md")
FULL_TASK_USER_INSTRUCTIONS = _load_prompt("full_task_user.md")
TURNING_ON_RADIO_BUTTON_VISUAL_PRIOR = _load_prompt(
    "turning_on_radio_button_visual_prior.md"
)
PLANNER_SYSTEM_INSTRUCTIONS = "\n\n".join(
    [_load_prompt("planner_system.md")]
)
PLANNER_USER_INSTRUCTIONS = _load_prompt("planner_user.md")
PI0_PICK_SYSTEM_INSTRUCTIONS = _load_prompt("pi0_pick_system.md")
PI0_NAV_PICK_SYSTEM_INSTRUCTIONS = _load_prompt("pi0_nav_pick_system.md")
PI0_NAV_PICK_PREPRESS_RESUME_INSTRUCTIONS = _load_prompt(
    "pi0_nav_pick_prepress_resume_system.md"
)
HYBRID_SYSTEM_INSTRUCTIONS = _load_prompt("hybrid_system.md")


def _task_system_instructions(base: str, *, task_name: str | None) -> str:
    if task_name == "turning_on_radio":
        return "\n\n".join([base, TURNING_ON_RADIO_BUTTON_VISUAL_PRIOR])
    return base


def mode_instructions(
    control_mode: str,
    *,
    task_name: str | None = None,
    pi0_hand: str = "right",
    pi0_instruction: str = (
        "Grasp the radio securely with the selected hand and stop as soon as "
        "the local grasp is achieved."
    ),
    pi0_max_chunks: int = 24,
) -> dict[str, str]:
    if control_mode == PLANNER_TOOLS_MODE:
        return {
            "behavior_system_instructions": _task_system_instructions(
                PLANNER_SYSTEM_INSTRUCTIONS,
                task_name=task_name,
            ),
            "behavior_user_instructions": PLANNER_USER_INSTRUCTIONS,
        }
    if control_mode == PI0_PICK_VLA_MODE:
        return {
            "behavior_system_instructions": PI0_PICK_SYSTEM_INSTRUCTIONS,
            "behavior_user_instructions": _render_prompt(
                "pi0_pick_user.md",
                pi0_hand=json.dumps(str(pi0_hand)),
                pi0_instruction=json.dumps(str(pi0_instruction)),
                pi0_max_chunks=int(pi0_max_chunks),
            ),
        }
    if control_mode == PI0_NAV_PICK_VLA_MODE:
        nav_system = PI0_NAV_PICK_SYSTEM_INSTRUCTIONS
        if task_name == "turning_on_radio":
            nav_system = "\n\n".join(
                [
                    nav_system,
                    TURNING_ON_RADIO_BUTTON_VISUAL_PRIOR,
                    PI0_NAV_PICK_PREPRESS_RESUME_INSTRUCTIONS,
                ]
            )
        return {
            "behavior_system_instructions": nav_system,
            "behavior_user_instructions": _render_prompt(
                "pi0_nav_pick_user.md",
                pi0_instruction=json.dumps(str(pi0_instruction)),
            ),
        }
    if control_mode == HYBRID_VLM_PI0_MODE:
        return {
            "behavior_system_instructions": _task_system_instructions(
                HYBRID_SYSTEM_INSTRUCTIONS,
                task_name=task_name,
            ),
            "behavior_user_instructions": _render_prompt(
                "hybrid_user.md",
                pi0_hand=json.dumps(str(pi0_hand)),
                pi0_instruction=json.dumps(str(pi0_instruction)),
                pi0_max_chunks=int(pi0_max_chunks),
            ),
        }
    return {
        "behavior_system_instructions": FULL_TASK_SYSTEM_INSTRUCTIONS,
        "behavior_user_instructions": FULL_TASK_USER_INSTRUCTIONS,
    }


__all__ = [
    "mode_instructions",
    "system_prompt",
    "user_prompt",
]
