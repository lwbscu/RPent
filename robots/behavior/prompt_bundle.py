"""Prompts for BEHAVIOR full-task VLA and planner-tool modes."""
from __future__ import annotations

from rpent.context.prompt_utils import PromptNode


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


FULL_TASK_SYSTEM_INSTRUCTIONS = """
Call run_full_task exactly once. It synchronously runs the configured Pi0.5
policy until official task success, environment termination/truncation, the
episode horizon, or an error. Do not call lower-level robot tools: none are
exposed in this first integration stage. Treat only task_success from the tool
as evaluation success; reward, termination, truncation, and local progress are
not success signals.
"""


FULL_TASK_USER_INSTRUCTIONS = """
Call run_full_task once.
"""


PLANNER_SYSTEM_INSTRUCTIONS = """
Use only the planner tools exposed in this mode: observe, pixel_to_world,
navigate_to, move_to, pick, rotate_wrist, press, and release. Do not emit raw
23D joint actions. First observe an RGB frame, identify the target pixel, call
pixel_to_world with both u and v from the same frame_id, and then command the
appropriate hand explicitly. If a target is out of reach, call navigate_to
before arm motion. Primitive success is not BEHAVIOR task success; always read
task_success as a separate official field in tool results.
"""


PLANNER_USER_INSTRUCTIONS = """
Solve the task using planner tools. Re-observe after navigation or any failed
motion before converting pixels again. Stop only when the official task_success
field says the BEHAVIOR task succeeded or when the tools return a structured
non-recoverable failure.
"""


def mode_instructions(control_mode: str) -> dict[str, str]:
    if control_mode == "planner_tools":
        return {
            "behavior_system_instructions": PLANNER_SYSTEM_INSTRUCTIONS.strip(),
            "behavior_user_instructions": PLANNER_USER_INSTRUCTIONS.strip(),
        }
    return {
        "behavior_system_instructions": FULL_TASK_SYSTEM_INSTRUCTIONS.strip(),
        "behavior_user_instructions": FULL_TASK_USER_INSTRUCTIONS.strip(),
    }


__all__ = [
    "mode_instructions",
    "system_prompt",
    "user_prompt",
]
