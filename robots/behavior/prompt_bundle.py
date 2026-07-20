"""Prompts for isolated full-task, planner, and local-VLA BEHAVIOR modes."""
from __future__ import annotations

import json

from robots.behavior.schemas import PI0_PICK_VLA_MODE, PLANNER_TOOLS_MODE
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
Call BehaviorPrimitives.run_full_task through the run_full_task tool exactly
once. It synchronously runs the configured Pi0.5
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
pixel_to_world with both u (image column) and v (image row) from the same
frame_id, and then command the appropriate hand explicitly. The returned
surface_normal points out of the visible surface toward the camera; for a
guarded press, use its negative as press_direction so the motion goes into the
surface. For a grasp, approach_vector likewise points from pregrasp toward the
object. If a target is out of reach, call navigate_to before arm motion, then
observe again because the old frame_id is stale. Primitive success is not
BEHAVIOR task success; always read task_success as a separate official field in
tool results.
"""


PLANNER_USER_INSTRUCTIONS = """
Solve the task using planner tools. Re-observe after navigation or any failed
motion before converting pixels again. Stop only when the official task_success
field says the BEHAVIOR task succeeded or when the tools return a structured
non-recoverable failure.
"""


PI0_PICK_SYSTEM_INSTRUCTIONS = """
Call BehaviorPrimitives.pi0_pick through the pi0_pick tool exactly once. This
is a local Pi0.5/VLA grasp loop, not a full-task run and not the planner pick.
It executes validated 23D whole-body action chunks only until a local gripper
grasp validator accepts the grasp, an official environment stop, the local
chunk limit, or an error. A closure candidate is recorded but is neither a stop
condition nor proof that the object was picked: inspect the saved MP4 before
accepting the grasp. Never infer official task success from primitive_success
or local_grasp_success; only task_success mirrors raw info.done.success.
"""


def mode_instructions(
    control_mode: str,
    *,
    pi0_hand: str = "right",
    pi0_instruction: str = (
        "Grasp the radio securely with the selected hand and stop as soon as "
        "the local grasp is achieved."
    ),
    pi0_max_chunks: int = 24,
) -> dict[str, str]:
    if control_mode == PLANNER_TOOLS_MODE:
        return {
            "behavior_system_instructions": PLANNER_SYSTEM_INSTRUCTIONS.strip(),
            "behavior_user_instructions": PLANNER_USER_INSTRUCTIONS.strip(),
        }
    if control_mode == PI0_PICK_VLA_MODE:
        pi0_user_instructions = (
            "Call pi0_pick exactly once with "
            f"hand={json.dumps(str(pi0_hand))} and "
            f"instruction={json.dumps(str(pi0_instruction))} and "
            f"max_chunks={int(pi0_max_chunks)}. "
            "Do not call run_full_task or planner tools in this mode."
        )
        return {
            "behavior_system_instructions": PI0_PICK_SYSTEM_INSTRUCTIONS.strip(),
            "behavior_user_instructions": pi0_user_instructions,
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
