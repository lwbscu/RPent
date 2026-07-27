"""Assemble and validate the BEHAVIOR Explore and Eval prompt context."""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from typing import Final

from robots.behavior.prompts import system as system_parts
from robots.behavior.prompts import user as user_parts
from robots.behavior.prompts.tasks import get_task_prompt_profile
from robots.behavior.task_specs import BehaviorTaskSpec, get_task_spec
from rpent.context.prompt_utils import PromptNode, format_prompt

BEHAVIOR_PHASES: Final = ("explore", "eval")
_PLACEHOLDER = re.compile(r"\{\{ ([a-z][a-z0-9_]*) \}\}")
_R1PRO_ROBOT_PRIOR_PATH: Final = (
    Path(__file__).resolve().parent / "guides" / "R1Pro robot prior.md"
)
_MAX_R1PRO_ROBOT_PRIOR_BYTES: Final = 32_768
_R1PRO_ROBOT_PRIOR_MARKERS: Final = (
    "# R1Pro Robot Prior",
    "r1pro_hand_prior:",
    "scalar depth",
    "motion authorization",
)


def _load_r1pro_robot_prior() -> str:
    """Load the one task-independent, repo-local R1Pro geometry prior."""

    path = _R1PRO_ROBOT_PRIOR_PATH
    if path.is_symlink():
        raise ValueError("R1Pro robot prior must not be a symlink")
    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise ValueError("R1Pro robot prior is missing") from error
    except OSError as error:
        raise ValueError("R1Pro robot prior metadata is unreadable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("R1Pro robot prior must be a regular file")
    if metadata.st_size > _MAX_R1PRO_ROBOT_PRIOR_BYTES:
        raise ValueError("R1Pro robot prior exceeds the 32768-byte prompt input limit")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError("R1Pro robot prior is unreadable") from error
    if len(payload) != metadata.st_size:
        raise ValueError("R1Pro robot prior changed while being read")
    if len(payload) > _MAX_R1PRO_ROBOT_PRIOR_BYTES:
        raise ValueError("R1Pro robot prior exceeds the 32768-byte prompt input limit")
    try:
        rendered = payload.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise ValueError("R1Pro robot prior must be valid UTF-8") from error
    if not rendered:
        raise ValueError("R1Pro robot prior must be non-empty")
    missing = tuple(
        marker for marker in _R1PRO_ROBOT_PRIOR_MARKERS if marker not in rendered
    )
    if missing:
        raise ValueError(
            f"R1Pro robot prior is missing required markers: {list(missing)}"
        )
    return rendered


def _render_section(template: str, **values: object) -> str:
    """Render one phase-specific Python prompt section exactly once."""

    template_remainder = _PLACEHOLDER.sub("", template)
    if "{{" in template_remainder or "}}" in template_remainder:
        raise ValueError("malformed or unrendered Python prompt placeholder")
    required = set(_PLACEHOLDER.findall(template))
    missing = required.difference(values)
    if missing:
        raise ValueError(f"missing Python prompt values: {sorted(missing)}")
    return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), template)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _render_prior_attempt_summaries(
    value: str | None,
) -> str:
    """Validate the bounded, harness-sanitized Explore history."""

    if value is None:
        return "No earlier attempt summary is available."
    if not isinstance(value, str):
        raise ValueError("prior_attempt_summaries must be a string")
    rendered = value.strip()
    if not rendered:
        return "No earlier attempt summary is available."
    if len(rendered) > 16_000:
        raise ValueError("prior_attempt_summaries may contain at most 16000 characters")
    return rendered


def _seed_range_text(seeds: tuple[int, ...]) -> str:
    if len(seeds) == 1:
        return str(seeds[0])
    if seeds == tuple(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]} through {seeds[-1]}"
    return ", ".join(str(seed) for seed in seeds)


def _surface_review_guidance(spec: BehaviorTaskSpec) -> str:
    policy = spec.surface_review_policy
    if policy is None:
        return """No task-specific target/opposite-surface regression policy is
registered for this task. Do not manufacture a surface-regression cycle from
ordinary task-object views. Surface reviews do not disable `pi0_nav_pick` for
this task."""

    cycles = policy.opposite_cycles_before_pi0_disable
    return f"""Any task-surface classification uses the two-call `frame_review`
protocol. First call `observe` with only `camera` to capture one fresh public
frame. Inspect that returned RGB yourself. Then call `observe` again with the
same `camera` and:

```json
"frame_review": {{
  "frame_id": "<frame_id returned by the preceding observe>",
  "assessment": "{policy.target_assessment}"
}}
```

The second call submits the review of the cited earlier frame. It is review-only:
it does not capture a new frame, advance physics, or permit reviewing a frame
that was not returned by the immediately preceding public `observe`. Select
exactly one assessment allowed by the task policy:
`{policy.target_assessment}`, `{policy.opposite_assessment}`, or
`{policy.indeterminate_assessment}`. The selected task's reviewed target prior
and additional expert knowledge define their visual meaning. Never submit an
assessment in the call that captures the frame, infer it from tool metadata, or
reuse an assessment across a scene-changing action.

A complete VLA surface-recovery cycle requires this exact current-attempt
causal chain: a successful `rotate_wrist` call for literal `hand="left"` or
`hand="right"` whose return reports matching requested and resolved physical
hands, accepted fresh head-frame visual-hand evidence for that hand, and a
non-empty stable call-start attachment-fingerprint receipt for that selected
hand; a new two-call review of a fresh post-rotation frame accepted as
`{policy.target_assessment}` and therefore bound to that rotated state; the
immediately subsequent admitted `pi0_nav_pick`, which executes exactly the
LLM-requested N complete 32-action chunks, completes its controller handoff,
and returns without raw official success; and, prior to any scene-changing
action or another `pi0_nav_pick`, a new two-call review of a distinct fresh
frame accepted as `{policy.opposite_assessment}`. The other hand may
independently have its own attachment and does not make this selected-hand
chain ambiguous. A failed or missing selected-hand literal rotation, a missing
or unstable selected-hand call-start attachment fingerprint, a missing
post-rotation target-bearing review, a stale review, a rejected VLA call, a VLA
call that does not finish its requested complete chunks and handoff, or
`{policy.indeterminate_assessment}` does not complete a cycle.

After {cycles} complete VLA surface-recovery cycles in the same attempt, do not
call `pi0_nav_pick` again. This is neither success nor terminal failure and does
not end the episode. Continue from fresh evidence with the other public
primitives; their availability remains governed only by their schemas, runtime
facts, and the ordinary trusted terminal conditions."""


def _terminal_failure_guidance(spec: BehaviorTaskSpec) -> tuple[int, str, str]:
    policy = spec.terminal_failure_policy
    if policy is None:
        return (
            4,
            "",
            """No task-specific visual terminal-failure condition is registered
for this task. Do not submit a `terminal_failure` declaration. Object
displacement, loss, occlusion, or local action failure does not add another
trusted stop condition while execution remains available.""",
        )

    causes = ", ".join(f"`{cause}`" for cause in policy.causes)
    cameras = ", ".join(f"`{camera}`" for camera in policy.cameras)
    condition = policy.condition
    return (
        5,
        """5. a fresh, current visual observation establishes the task-specific
   terminal visual failure described by the selected target prior, the current
   attempt is marked failed, and official `task_success` remains false.""",
        f"""A tool return, gripper state, suspected loss, or object-control result
alone does not establish condition 5.

When and only when condition 5 is visually established, immediately call the
existing `save_robot_state_checkpoint` tool with `terminal_failure` containing
condition `{condition}`, one allowed cause ({causes}), one allowed public camera
role ({cameras}), and the fresh frame ID required by its current public schema.

Do not call any other tool afterward. Runtime validates that frame against the
current environment step, captures synchronized final RGB-D, freezes motion,
returns `_finish=true` with `task_success=false`, and transfers the sealed
attempt to failure analysis. If the visual frame does not establish the full
task-specific failure and one allowed cause, do not submit this declaration.""",
    )


def _checkpoint_terminal_guidance(spec: BehaviorTaskSpec) -> str:
    if spec.terminal_failure_policy is None:
        return """No task-specific terminal use of this compatibility tool is
registered. Do not submit `terminal_failure` for this task."""
    return """Its only terminal use is the structured `terminal_failure`
declaration described under condition 5 below; this does not add another
tool."""


def system_prompt() -> PromptNode:
    """Assemble the ordered BEHAVIOR system prompt tree."""

    return {
        "ROLE AND EVALUATION": system_parts.ROLE_AND_EVALUATION,
        "PUBLIC CELL AND CURRENT INVOCATION": "{{ behavior_public_cell }}",
        "ATTEMPT MODE": "{{ behavior_attempt_mode }}",
        "R1PRO ROBOT GEOMETRY PRIOR": "{{ behavior_robot_prior }}",
        "FROZEN EXPLORE MEMORY": "{{ behavior_frozen_memory }}",
        "REVIEWED RECIPE CATALOG PRIORS": system_parts.REVIEWED_RECIPE_CATALOG,
        "TASK TARGET PRIOR": system_parts.TARGET_PRIOR,
        "REVIEWED EXPLORE EXPERIENCE": system_parts.REVIEWED_EXPLORE_EXPERIENCE,
        "ADDITIONAL TASK EXPERT KNOWLEDGE": (system_parts.ADDITIONAL_EXPERT_KNOWLEDGE),
        "KNOWLEDGE PRECEDENCE": "{{ behavior_knowledge_precedence }}",
        "PERCEPTION ISOLATION AND EVIDENCE FRESHNESS": (
            system_parts.PERCEPTION_ISOLATION
        ),
        "UNORDERED PUBLIC CAPABILITY SURFACE": system_parts.PUBLIC_CAPABILITIES,
        "RUNTIME CAPABILITY CONTRACT": system_parts.RUNTIME_CONTRACT,
        "TASK-SPECIFIC EXECUTION GUIDANCE": "{{ behavior_task_prompt }}",
        "OFFICIAL SUCCESS AND RUNNER TERMINATION": system_parts.OFFICIAL_SUCCESS,
        "CONTINUATION REQUIREMENT": system_parts.CONTINUATION_REQUIREMENT,
        "ROBUST_RECIPE, TASK_MEMORY, REF_SOLVED_DIR, AND OUTPUTS": (
            "{{ behavior_outputs_policy }}"
        ),
    }


def user_prompt() -> PromptNode:
    """Assemble the ordered BEHAVIOR user prompt tree."""

    return {
        "CELL": user_parts.CELL,
        "BEGIN": user_parts.BEGIN,
    }


def build_prompt_context(
    *,
    phase: str,
    task_name: str,
    task_language: str,
    public_seed: int,
    recipe_tag: str,
    output_dir: str | Path,
    max_session_steps: int,
    global_tool_budget: int,
    wall_clock_seconds: int,
    target_prior: str,
    reviewed_explore_experience: str,
    additional_expert_knowledge: str,
    reviewed_repo_memory_manifest: str,
    reviewed_recipe_priors: str,
    reviewed_recipe_selection_manifest: str,
    robust_recipe: str | None = None,
    task_memory: str | None = None,
    memory_manifest: str | None = None,
    source_public_seed: int | None = None,
    source_recipe_tag: str | None = None,
    attempt_index: int = 1,
    job_id: str | None = None,
    prior_attempt_summaries: str | None = None,
) -> dict[str, str | int]:
    """Validate and bind one public BEHAVIOR protocol cell to prompt variables."""

    if phase not in BEHAVIOR_PHASES:
        raise ValueError(
            f"unsupported BEHAVIOR phase {phase!r}; expected one of {BEHAVIOR_PHASES}"
        )
    try:
        task_spec = get_task_spec(task_name)
    except ValueError as error:
        raise ValueError(f"unsupported BEHAVIOR task: {task_name!r}") from error
    if not isinstance(task_language, str) or not task_language.strip():
        raise ValueError("task_language must be a non-empty string")
    if task_language.strip() != task_spec.task_language:
        raise ValueError(
            "task_language must exactly match the selected BEHAVIOR TaskSpec"
        )
    if isinstance(public_seed, bool) or not isinstance(public_seed, int):
        raise ValueError("public_seed must be an integer")
    allowed_seeds = (
        task_spec.explore_public_seeds
        if phase == "explore"
        else task_spec.eval_public_seeds
    )
    if public_seed not in allowed_seeds:
        seed_label = "public seed" if len(allowed_seeds) == 1 else "public seeds"
        raise ValueError(
            f"{phase.title()} is restricted to {seed_label} "
            f"{_seed_range_text(allowed_seeds)} for {task_name}"
        )
    expected_tag = task_spec.tag(public_seed)
    if recipe_tag != expected_tag:
        raise ValueError(f"recipe_tag must be the public tag {expected_tag!r}")

    budgets = {
        "max_session_steps": _positive_int("max_session_steps", max_session_steps),
        "global_tool_budget": _positive_int("global_tool_budget", global_tool_budget),
        "wall_clock_seconds": _positive_int("wall_clock_seconds", wall_clock_seconds),
    }
    reviewed_values = {
        "target_prior": target_prior,
        "reviewed_explore_experience": reviewed_explore_experience,
        "additional_expert_knowledge": additional_expert_knowledge,
        "reviewed_repo_memory_manifest": reviewed_repo_memory_manifest,
        "reviewed_recipe_priors": reviewed_recipe_priors,
        "reviewed_recipe_selection_manifest": reviewed_recipe_selection_manifest,
    }
    for name, value in reviewed_values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    (
        trusted_stop_condition_count,
        task_terminal_stop_condition,
        task_terminal_failure_guidance,
    ) = _terminal_failure_guidance(task_spec)
    common_values: dict[str, object] = {
        "task_name": task_name,
        "task_language": json.dumps(task_language.strip(), ensure_ascii=False),
        "prompt_profile_id": task_spec.prompt_profile_id,
        "behavior_task_prompt": format_prompt(
            get_task_prompt_profile(task_spec.prompt_profile_id)
        ).strip(),
        "public_seed": public_seed,
        "recipe_tag": recipe_tag,
        "output_dir": str(output_dir),
        "task_surface_review_guidance": _surface_review_guidance(task_spec),
        "trusted_stop_condition_label": {
            4: "four",
            5: "five",
        }[trusted_stop_condition_count],
        "task_terminal_stop_condition": task_terminal_stop_condition,
        "task_terminal_failure_guidance": task_terminal_failure_guidance,
        "task_checkpoint_terminal_guidance": _checkpoint_terminal_guidance(task_spec),
        "behavior_robot_prior": _load_r1pro_robot_prior(),
        **{name: value.strip() for name, value in reviewed_values.items()},
        **budgets,
    }

    frozen_values = (robust_recipe, task_memory, memory_manifest)
    if phase == "explore":
        if any(value is not None for value in frozen_values):
            raise ValueError("Explore rejects frozen Eval memory inputs")
        if source_public_seed is not None or source_recipe_tag is not None:
            raise ValueError("Explore rejects frozen Eval source identity")
        attempt_index = _positive_int("attempt_index", attempt_index)
        if job_id is None:
            job_id = "standalone-explore-job"
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a non-empty string")
        render_values = {
            **common_values,
            "attempt_index": attempt_index,
            "job_id": job_id.strip(),
            "prior_attempt_summaries": _render_prior_attempt_summaries(
                prior_attempt_summaries
            ),
        }
        user = (
            f"Execute Explore attempt {attempt_index} in this one fresh episode. "
            "Choose among the public peer primitives from current public evidence "
            "and runtime guards. Follow the system CONTINUATION REQUIREMENT: "
            "while raw official success remains false and execution budget remains, "
            "continue toward another concrete tool call unless one of its "
            f"{common_values['trusted_stop_condition_label']} listed stop "
            "conditions is established. "
            "The outer harness "
            "alone decides whether to create another attempt."
        )
    else:
        if attempt_index != 1 or job_id is not None or prior_attempt_summaries:
            raise ValueError("Eval rejects Explore job/attempt context")
        if not all(isinstance(value, str) and value.strip() for value in frozen_values):
            raise ValueError(
                "Eval requires non-empty robust_recipe, task_memory, and "
                "memory_manifest"
            )
        if (
            isinstance(source_public_seed, bool)
            or not isinstance(source_public_seed, int)
            or source_public_seed not in task_spec.explore_public_seeds
        ):
            raise ValueError(
                "Eval frozen source public seed must belong to the selected "
                "task's Explore partition"
            )
        expected_source_tag = task_spec.tag(source_public_seed)
        if source_recipe_tag != expected_source_tag:
            raise ValueError(
                "Eval frozen source recipe tag must match its trusted "
                f"Explore source identity {expected_source_tag!r}"
            )
        render_values = {
            **common_values,
            "source_public_seed": source_public_seed,
            "source_recipe_tag": source_recipe_tag,
            "robust_recipe": robust_recipe,
            "task_memory": task_memory,
            "memory_manifest": memory_manifest,
        }
        user = (
            "Execute the single fresh evaluation attempt. Choose among the public "
            "peer primitives from current public evidence and runtime guards. "
            "Follow the system CONTINUATION REQUIREMENT: while raw official success "
            "remains false and execution budget remains, continue toward another "
            "concrete tool call unless one of its "
            f"{common_values['trusted_stop_condition_label']} listed stop "
            "conditions is established. "
            "The evaluation creates no retry and never "
            "updates frozen task memory."
        )

    phase_sections = {
        "explore": {
            "public_cell": system_parts.EXPLORE_PUBLIC_CELL,
            "attempt_mode": system_parts.EXPLORE_ATTEMPT_MODE,
            "frozen_memory": "",
            "knowledge_precedence": system_parts.EXPLORE_PRECEDENCE,
            "outputs_policy": system_parts.EXPLORE_OUTPUTS,
            "recipe_phase_scope": "for this Explore invocation",
            "budget_scope": "per-attempt",
            "budget_label": "Attempt",
        },
        "eval": {
            "public_cell": system_parts.EVAL_PUBLIC_CELL,
            "attempt_mode": system_parts.EVAL_ATTEMPT_MODE,
            "frozen_memory": system_parts.EVAL_FROZEN_MEMORY,
            "knowledge_precedence": system_parts.EVAL_PRECEDENCE,
            "outputs_policy": system_parts.EVAL_OUTPUTS,
            "recipe_phase_scope": "that are authorized for formal evaluation",
            "budget_scope": "episode",
            "budget_label": "Episode",
        },
    }[phase]
    context: dict[str, str | int] = {
        **render_values,
        "behavior_phase": phase,
        "recipe_phase_scope": phase_sections["recipe_phase_scope"],
        "budget_scope": phase_sections["budget_scope"],
        "budget_label": phase_sections["budget_label"],
        "behavior_public_cell": _render_section(
            phase_sections["public_cell"], **render_values
        ),
        "behavior_attempt_mode": _render_section(
            phase_sections["attempt_mode"], **render_values
        ),
        "behavior_frozen_memory": (
            _render_section(phase_sections["frozen_memory"], **render_values)
            if phase_sections["frozen_memory"]
            else ""
        ),
        "behavior_knowledge_precedence": phase_sections["knowledge_precedence"],
        "behavior_outputs_policy": _render_section(
            phase_sections["outputs_policy"], **render_values
        ),
        "behavior_user_instructions": user,
    }
    return context


__all__ = [
    "BEHAVIOR_PHASES",
    "build_prompt_context",
    "system_prompt",
    "user_prompt",
]
