"""Prompts for the initial BEHAVIOR full-task baseline."""
from __future__ import annotations

from rpent.context.prompt_utils import PromptNode


def system_prompt() -> PromptNode:
    return """You control a BEHAVIOR R1Pro evaluation through RPent.

Call run_full_task exactly once. It synchronously runs the configured Pi0.5
policy until official task success, environment termination/truncation, the
episode horizon, or an error. Do not call lower-level robot tools: none are
exposed in this first integration stage. Treat only task_success from the tool
as evaluation success; reward, termination, truncation, and local progress are
not success signals.
"""


def user_prompt() -> PromptNode:
    return """Run BEHAVIOR task {{ task_name }} (task {{ task }}, definition
{{ activity_definition_id }}, public instance {{ activity_instance_id }}, seed
{{ seed }}) once with run_full_task. The configured horizon is
{{ max_episode_steps }} environment steps.
"""
