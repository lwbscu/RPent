import inspect
from pathlib import Path

import pytest

import robots.behavior.prompt_bundle as prompt_bundle
from robots.behavior.prompt_bundle import (
    build_prompt_context,
    system_prompt,
    user_prompt,
)
from robots.behavior.prompts.tasks import (
    TASK_PROMPT_PROFILES,
    get_task_prompt_profile,
)
from robots.behavior.task_specs import PICKING_UP_TRASH_TASK_SPEC
from rpent.context.prompt_utils import format_prompt

PROMPT_DIR = Path(__file__).parents[1] / "robots" / "behavior" / "prompts"
TASK_PROMPT_DIR = PROMPT_DIR / "tasks"
GUIDE_DIR = Path(__file__).parents[1] / "robots" / "behavior" / "guides"
CANONICAL_PROMPT_MODULES = {
    "__init__.py",
    "system.py",
    "user.py",
}
CANONICAL_TASK_PROMPT_MODULES = {
    "__init__.py",
    "picking_up_trash.py",
    "turning_on_radio.py",
}
CANONICAL_GUIDES = {
    "R1Pro robot prior.md",
    "env_calibration.md",
    "pro_hybrid_guide.md",
    "strict_hybrid_guide.md",
}
R1PRO_ROBOT_PRIOR_PATH = GUIDE_DIR / "R1Pro robot prior.md"
PUBLIC_TOOLS = (
    "close",
    "move_to",
    "navigate_to",
    "observe",
    "open",
    "pi0_nav_pick",
    "pixel_to_world",
    "press",
    "rotate_wrist",
    "save_robot_state_checkpoint",
)
LEGACY_TOOLS = (
    "finish",
    "reset",
    "inspect_post_pick_state",
    "inspect_toggle_geometry",
    "post_pick_close_press_gripper",
    "post_pick_recenter_held_button",
    "post_pick_direct_finger_toggle",
    "post_success_hold_frames",
)
ORDERING_PHRASES = (
    "first physical",
    "exactly once",
    "one permitted",
    "single-call allowance",
    "pi0 gate",
    "post-pi0",
    "three-camera",
    "then re-observe",
    "then re-project",
    "one-use authorization",
    "then call `finish`",
)
COMMON_MARKERS = (
    "PERCEPTION-ISOLATED mode",
    "Perception isolation and evidence freshness",
    "Unordered public capability surface",
    "Exactly ten public primitives are available",
    "List order and grouping carry no execution meaning",
    "may switch freely between VLA and analytic capabilities",
    "`chunks=N`",
    "no fixed maximum",
    "upper requested work bound",
    "receipt-bound success partial is a normal terminal outcome",
    "`head`, `left_wrist`, and `right_wrist` are the three public cameras",
    "five analytic manipulation primitives",
    "latest synchronized public head capture",
    '`motion_scope="whole_body"`',
    "does not apply to `pi0_nav_pick`",
    "read-only visual anchor",
    "Task target prior",
    "Reviewed Explore experience",
    "Additional task expert knowledge",
    "runtime guards and current public evidence take precedence",
    "frozen Task Memory",
    "Runtime capability contract",
    "CONTINUATION REQUIREMENT",
    '`info["done"]["success"]`',
    "runner-owned termination",
    "produce a standalone final answer",
    "continue using the current episode",
)

TARGET_PRIOR = """TARGET_PRIOR_SENTINEL
Positive target evidence requires red_front_face=true,
black_round_or_oval_disk=true, white_outer_ring=true, and
red_center_bump=true together in one fresh public frame.
"""
REVIEWED_EXPLORE_EXPERIENCE = (
    "REVIEWED_EXPLORE_EXPERIENCE_SENTINEL: retain only anonymous semantic lessons."
)
ADDITIONAL_EXPERT_KNOWLEDGE = (
    "ADDITIONAL_EXPERT_KNOWLEDGE_SENTINEL: a control-face target lock requires "
    "the complete task-specific signature on one current major surface. "
    "Meaningful object motion invalidates the prior lock."
)
TASK_EXPERT_PHRASES = (
    "control-face target lock",
    "complete task-specific signature",
    "Meaningful object motion invalidates the prior lock",
)
TRASH_TASK_LANGUAGE = (
    "Put the three can of soda from the living room inside the tash can in the kitchen."
)
TRASH_TARGET_PRIOR = """TRASH_TARGET_PRIOR_SENTINEL
Every required soda can must be inside the task-relevant kitchen trash can.
Re-identify the objects and receptacle from fresh public evidence.
"""
TRASH_EXPLORE_EXPERIENCE = """TRASH_EXPLORE_EXPERIENCE_SENTINEL
No reviewed raw-official-success-derived Explore experience is registered.
"""


def _instructions(phase="eval", **overrides):
    public_seed = 0 if phase == "explore" else 1
    values = {
        "phase": phase,
        "task_name": "turning_on_radio",
        "task_language": (
            "Turn on the radio receiver that's on the table in the living room."
        ),
        "public_seed": public_seed,
        "recipe_tag": f"turning_on_radio_s{public_seed}",
        "output_dir": "/tmp/public-output",
        "max_session_steps": 24756,
        "global_tool_budget": 300,
        "wall_clock_seconds": 43200,
        "target_prior": TARGET_PRIOR,
        "reviewed_explore_experience": REVIEWED_EXPLORE_EXPERIENCE,
        "additional_expert_knowledge": ADDITIONAL_EXPERT_KNOWLEDGE,
        "reviewed_repo_memory_manifest": (
            '{"source": "reviewed-repository", "reviewed": true}'
        ),
        "reviewed_recipe_priors": (
            "Reviewed Recipe guidance stays semantic and evidence-conditioned."
        ),
        "reviewed_recipe_selection_manifest": (
            '{"catalog_sha256": "abc", "selected_ids": ["semantic-control-v1"]}'
        ),
    }
    if phase == "explore":
        values.update(
            {
                "attempt_index": 3,
                "job_id": "explore-job-abc",
                "prior_attempt_summaries": (
                    "Attempt 1: grasp lost with uncertain object control.\n"
                    "Attempt 2: the semantic hypothesis did not match the scene."
                ),
            }
        )
    else:
        values.update(
            {
                "source_public_seed": 0,
                "source_recipe_tag": "turning_on_radio_s0",
                "robust_recipe": "Use current evidence and state-conditioned knowledge.",
                "task_memory": "The radio control has reusable semantic cues.",
                "memory_manifest": '{"public_source_seed": 0, "verified": true}',
            }
        )
    values.update(overrides)
    context = build_prompt_context(**values)
    return {
        **context,
        "behavior_system_instructions": format_prompt(
            system_prompt(),
            variables=context,
        ),
        "behavior_user_instructions": format_prompt(
            user_prompt(),
            variables=context,
        ),
    }


def _trash_instructions(phase="explore", **overrides):
    public_seed = 0 if phase == "explore" else 10
    values = {
        "task_name": "picking_up_trash",
        "task_language": TRASH_TASK_LANGUAGE,
        "public_seed": public_seed,
        "recipe_tag": f"picking_up_trash_s{public_seed}",
        "target_prior": TRASH_TARGET_PRIOR,
        "reviewed_explore_experience": TRASH_EXPLORE_EXPERIENCE,
        "additional_expert_knowledge": (
            "No additional reviewed expert knowledge is registered for this task."
        ),
        "reviewed_recipe_priors": (
            "No reviewed Recipe entries are selected for this task."
        ),
        "reviewed_recipe_selection_manifest": (
            '{"catalog_sha256": "abc", "selected_entry_ids": []}'
        ),
    }
    if phase == "eval":
        values.update(
            {
                "source_public_seed": 0,
                "source_recipe_tag": "picking_up_trash_s0",
                "robust_recipe": (
                    "Use fresh evidence and task-local containment semantics."
                ),
                "task_memory": "All required soda cans belong inside the receptacle.",
                "memory_manifest": '{"public_source_seed": 0, "verified": true}',
            }
        )
    values.update(overrides)
    return _instructions(phase, **values)


def _prompt_source():
    return "\n".join(
        (PROMPT_DIR / filename).read_text(encoding="utf-8")
        for filename in sorted(CANONICAL_PROMPT_MODULES)
    )


def _guide_source():
    return "\n".join(
        (GUIDE_DIR / filename).read_text(encoding="utf-8")
        for filename in sorted(CANONICAL_GUIDES)
    )


def _normalized(value):
    return " ".join(value.split())


def test_active_prompt_and_guide_files_match_libero_plugin_layout():
    assert CANONICAL_PROMPT_MODULES == {
        path.name for path in PROMPT_DIR.iterdir() if path.is_file()
    }
    assert CANONICAL_TASK_PROMPT_MODULES == {
        path.name for path in TASK_PROMPT_DIR.iterdir() if path.is_file()
    }
    assert CANONICAL_GUIDES == {
        path.name for path in GUIDE_DIR.iterdir() if path.is_file()
    }


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_each_rendered_prompt_is_complete_and_exposes_exactly_ten_peer_tools(phase):
    source = _instructions(phase)["behavior_system_instructions"]
    for section in (
        "PUBLIC CELL AND CURRENT INVOCATION",
        "UNORDERED PUBLIC CAPABILITY SURFACE",
        "RUNTIME CAPABILITY CONTRACT",
        "TASK TARGET PRIOR",
        "REVIEWED EXPLORE EXPERIENCE",
        "ADDITIONAL TASK EXPERT KNOWLEDGE",
        "OFFICIAL SUCCESS AND RUNNER TERMINATION",
    ):
        assert section in source
    tool_section = source.split("UNORDERED PUBLIC CAPABILITY SURFACE", 1)[1].split(
        "RUNTIME CAPABILITY CONTRACT", 1
    )[0]
    listed = tuple(
        line.removeprefix("- `").removesuffix("`")
        for line in tool_section.splitlines()
        if line.startswith("- `")
    )
    assert listed == PUBLIC_TOOLS
    for legacy in LEGACY_TOOLS:
        assert f"`{legacy}`" not in source


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_user_prompt_refers_to_public_peer_surface_without_a_stale_count(phase):
    user = _normalized(_instructions(phase)["behavior_user_instructions"])

    assert (
        "Choose among the public peer primitives from current public evidence "
        "and runtime guards."
    ) in user
    assert "nine peer primitives" not in user
    assert "ten peer primitives" not in user


def test_common_protocol_semantics_have_parity_across_phases():
    sources = [
        _normalized(_instructions(phase)["behavior_system_instructions"]).lower()
        for phase in ("explore", "eval")
    ]
    for marker in COMMON_MARKERS:
        assert all(marker.lower() in source for source in sources)


def test_analytic_hand_selection_contract_is_identical_across_prompts():
    sections = []
    for phase in ("explore", "eval"):
        source = _instructions(phase)["behavior_system_instructions"]
        sections.append(
            source.split("The five analytic manipulation primitives", 1)[1].split(
                "The capability schemas are authoritative.", 1
            )[0]
        )
    assert sections[0] == sections[1]


@pytest.mark.parametrize(
    "factory",
    (_instructions, _trash_instructions),
    ids=("turning_on_radio", "picking_up_trash"),
)
@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_rendered_prompts_use_only_literal_physical_hands_and_wrist_cameras(
    factory, phase
):
    system = _normalized(factory(phase)["behavior_system_instructions"])

    for forbidden in (
        "held_wrist",
        "press_wrist",
        "dynamic role `held`",
        "dynamic role `press`",
        "semantic_role",
        "requested_role",
    ):
        assert forbidden not in system
    assert "`head`, `left_wrist`, and `right_wrist`" in system
    assert "exactly one literal `hand`, either `left` or `right`" in system
    assert "`selected_hand` must exactly equal the requested `hand`" in system


def test_canonical_prompt_modules_and_guides_are_task_agnostic():
    source = _prompt_source() + "\n" + _guide_source()
    for task_specific_text in (
        "turning_on_radio",
        "picking_up_trash",
        "task-relevant radio",
        "red_front_face",
        "black_round_or_oval_disk",
        "white_outer_ring",
        "red_center_bump",
        "radio_tipped_flat",
        *TASK_EXPERT_PHRASES,
    ):
        assert task_specific_text not in source
    rendered = _instructions("explore")["behavior_system_instructions"]
    assert rendered.count("TARGET_PRIOR_SENTINEL") == 1
    assert rendered.count("REVIEWED_EXPLORE_EXPERIENCE_SENTINEL") == 1
    assert rendered.count("ADDITIONAL_EXPERT_KNOWLEDGE_SENTINEL") == 1


def test_pi0_v5_success_stop_contract_is_shared_by_prompt_profiles_and_guide():
    sources = [
        _normalized(factory(phase)["behavior_system_instructions"])
        for factory in (_instructions, _trash_instructions)
        for phase in ("explore", "eval")
    ]
    sources.append(
        _normalized((GUIDE_DIR / "pro_hybrid_guide.md").read_text(encoding="utf-8"))
    )
    required_semantics = (
        ("upper requested work bound",),
        ("If raw official success and the allowed terminal exceptions remain absent",),
        ("exactly N complete",),
        (
            "stops physical task execution at the exact successful environment step",
            "physical task execution stops at the exact successful environment step",
        ),
        (
            "no later action, prediction, public-tool call, observation, "
            "capability read, or other task RPC",
            "no remaining chunk action, model prediction, public-tool call, "
            "observation, capability read, or other task RPC",
        ),
        ("receipt-bound success partial is a normal terminal outcome",),
        (
            "not counted as a complete chunk",
            "not counted in `full_chunks_executed`",
        ),
    )
    obsolete = (
        "raw-success-latched run both finish the requested N chunks",
        "raw official success do not shorten the requested batch",
        "continues through the remainder of its requested N complete chunks",
    )

    for source in sources:
        for alternatives in required_semantics:
            assert any(marker in source for marker in alternatives)
        for marker in obsolete:
            assert marker not in source


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_common_navigation_prompt_covers_projection_and_relative_pure_base_modes(
    phase,
):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    for marker in (
        "`navigate_to` has two mutually exclusive pure-base modes",
        "Projection mode consumes a fresh projection",
        "Relative mode needs no projection",
        "`forward` or `backward`",
        "`left` or `right`",
        "whole body moves or rotates together with the base",
        "after an admitted `navigate_to` returns, obtain a fresh "
        '`observe(camera="head")`',
        "not a fixed global tool sequence",
    ):
        assert marker in system
    assert "`navigate_to` is also outside the literal-hand contract" in _normalized(
        _guide_source()
    )
    capability = system.split("`navigate_to` has", 1)[1].split(
        "`move_to` accepts", 1
    )[0]
    for forbidden in (
        'hand="left"',
        'hand="right"',
        "role",
        "target_xyz",
        "delta_xyz",
        "chunks=",
    ):
        assert forbidden not in capability


def test_r1pro_robot_prior_guide_preserves_geometry_and_distance_semantics():
    prior = R1PRO_ROBOT_PRIOR_PATH.read_text(encoding="utf-8").strip()

    for marker in (
        "# R1Pro Robot Prior",
        "r1pro_hand_prior:",
        "units: meters",
        "left_right_same: true",
        "wrist_camera_origin_in_gripper_link:",
        "translation_xyz: [0.05051, 0.0028934, 0.0051317]",
        "translation_only: true",
        "complete_transform_required_for_3d_conversion: true",
        "distance_to_palm: 0.0509",
        "hand_points_in_palm:",
        "grip_point: [0.0, 0.0, -0.06]",
        "finger1_root_closed: [-0.000088709, 0.013453, -0.03689]",
        "finger2_root_closed: [0.000089046, -0.013453, -0.03689]",
        "fixed_distances_from_wrist_cam:",
        "grip_point: 0.0825",
        "finger1_root_closed: 0.0666",
        "finger2_root_closed: 0.0676",
        "gripper_kinematics:",
        "each_finger_joint_range: [0.0, 0.05]",
        "combined_added_finger_separation_range: [0.0, 0.10]",
        "physical_fingertip_aperture_calibrated: false",
    ):
        assert marker in prior

    normalized = _normalized(prior)
    assert (
        "first convert the LLM-selected target pixel and its valid depth into a "
        "3D point in the wrist-camera frame"
    ) in normalized
    assert (
        "Never estimate palm or grasp-center distance by subtracting a fixed "
        "offset from one scalar depth value"
    ) in normalized
    assert (
        "Transform that point from `wrist_cam` into the corresponding `palm` frame"
        in (normalized)
    )
    assert (
        "complete current `T_palm_from_camera` rotation and translation" in normalized
    )
    assert "The translation vector in this guide is not a complete transform" in (
        normalized
    )
    assert (
        "`optical_axis_depth_m` is the target surface's Z distance to the image plane"
        in normalized
    )
    assert "`camera_range_m` is the Euclidean range" in normalized
    assert "Do not interchange these two quantities" in normalized
    assert "For `left_wrist` or `right_wrist`" in normalized
    assert "matching physical side" in normalized
    assert "does not authorize selecting that hand" in normalized
    assert "`abs(r - d) <= target_to_reference <= r + d`" in normalized
    assert "For the palm reference, use `d = 0.0509 m`" in normalized
    assert "For the grip-point reference, use `d = 0.0825 m`" in normalized
    assert (
        "Do not apply these Euclidean bounds directly to `optical_axis_depth_m`"
        in normalized
    )


@pytest.mark.parametrize(
    "factory",
    (_instructions, _trash_instructions),
    ids=("turning_on_radio", "picking_up_trash"),
)
@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_shared_r1pro_prior_is_injected_exactly_once_outside_task_memory(
    factory, phase
):
    prompts = factory(phase)
    prior = R1PRO_ROBOT_PRIOR_PATH.read_text(encoding="utf-8").strip()
    system = prompts["behavior_system_instructions"]

    assert prompts["behavior_robot_prior"] == prior
    assert system.count(prior) == 1
    assert system.count("R1PRO ROBOT GEOMETRY PRIOR") == 1
    assert system.index("R1PRO ROBOT GEOMETRY PRIOR") < system.index(
        "TASK TARGET PRIOR"
    )
    assert prior not in prompts["behavior_task_prompt"]
    assert "r1pro_hand_prior:" not in prompts["target_prior"]
    assert "r1pro_hand_prior:" not in prompts["reviewed_explore_experience"]
    assert "r1pro_hand_prior:" not in prompts["additional_expert_knowledge"]
    if phase == "eval":
        assert "r1pro_hand_prior:" not in prompts["task_memory"]


def test_r1pro_prior_is_robot_geometry_not_task_or_action_authorization():
    prior = _normalized(R1PRO_ROBOT_PRIOR_PATH.read_text(encoding="utf-8"))
    lowered = prior.lower()

    assert "or select a hand for an episode" in lowered
    assert "must never be used by itself as" in lowered
    for limit in (
        "a contact or release gate",
        "official task-success evidence",
        "collision clearance or motion authorization",
        "justification for a fixed hand choice, fixed pixel, fixed pose, or fixed "
        "motion distance",
    ):
        assert limit in lowered
    for task_or_tool in (
        "turning_on_radio",
        "picking_up_trash",
        "pi0_nav_pick",
        "move_to",
        "observe(",
        "pixel_to_world",
        "open(",
        "close(",
        "press(",
    ):
        assert task_or_tool not in lowered


@pytest.mark.parametrize(
    ("case", "expected_error"),
    (
        ("missing", "is missing"),
        ("empty", "must be non-empty"),
        ("symlink", "must not be a symlink"),
        ("invalid_utf8", "must be valid UTF-8"),
        ("oversize", "exceeds the 32768-byte"),
        ("missing_marker", "missing required markers"),
    ),
)
def test_r1pro_robot_prior_loader_fails_closed(
    monkeypatch, tmp_path, case, expected_error
):
    path = tmp_path / "R1Pro robot prior.md"
    valid = R1PRO_ROBOT_PRIOR_PATH.read_bytes()

    if case == "empty":
        path.write_bytes(b"")
    elif case == "symlink":
        target = tmp_path / "real-prior.md"
        target.write_bytes(valid)
        path.symlink_to(target)
    elif case == "invalid_utf8":
        path.write_bytes(b"\xff\xfe")
    elif case == "oversize":
        path.write_bytes(valid + b"\n" + b"x" * 32_768)
    elif case == "missing_marker":
        path.write_text(
            "# R1Pro Robot Prior\nr1pro_hand_prior:\nscalar depth\n",
            encoding="utf-8",
        )
    elif case != "missing":
        raise AssertionError(f"unhandled test case: {case}")

    monkeypatch.setattr(prompt_bundle, "_R1PRO_ROBOT_PRIOR_PATH", path)
    with pytest.raises(ValueError, match=expected_error):
        _instructions("explore")


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_rendered_prompt_injects_each_task_knowledge_class_once(phase):
    prompt = _instructions(phase)["behavior_system_instructions"]
    assert prompt.count("TARGET_PRIOR_SENTINEL") == 1
    assert prompt.count("REVIEWED_EXPLORE_EXPERIENCE_SENTINEL") == 1
    assert prompt.count("ADDITIONAL_EXPERT_KNOWLEDGE_SENTINEL") == 1
    for phrase in TASK_EXPERT_PHRASES:
        assert prompt.count(phrase) == 1
    for field in (
        "red_front_face=true",
        "black_round_or_oval_disk=true",
        "white_outer_ring=true",
        "red_center_bump=true",
    ):
        assert prompt.count(field) == 1
    assert prompt.count("<target_prior>") == 1
    assert prompt.count("<explore_experience>") == 1
    assert prompt.count("<additional_expert_knowledge>") == 1


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_prompts_do_not_prescribe_tool_vla_or_camera_order(phase):
    lowered = _normalized(_instructions(phase)["behavior_system_instructions"]).lower()
    for phrase in ORDERING_PHRASES:
        assert phrase not in lowered
    assert "mandatory task role" in lowered
    assert "no primitive has protocol priority" in lowered
    assert "another vla invocation" in lowered
    assert "llm chooses any useful view" in lowered


def test_prompt_source_excludes_replayable_or_private_identity():
    source = _prompt_source()
    lowered = source.lower()
    for forbidden in (
        "activity_instance_id",
        "native instance",
        "initial_episode_mirror",
        "canonical restore",
        "rgb mae",
        "fixed hand",
    ):
        assert forbidden not in lowered
    for private_instance in (242, 109, 181, 187, 197, 203, 211, 212, 295, 298):
        assert str(private_instance) not in source
    assert '"selected_hand": "<left or right>"' in source
    assert '"selected_hand": "left"' not in source
    assert '"selected_hand": "right"' not in source


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_rendered_prompt_has_no_unrendered_placeholder(phase):
    prompt = _instructions(phase)["behavior_system_instructions"]
    assert "{{" not in prompt
    assert "turning_on_radio_s" in prompt
    assert "public seed" in prompt
    assert "one permitted" not in prompt.lower()


def test_explore_invocation_is_one_fresh_attempt_owned_by_outer_job():
    prompts = _instructions("explore")
    system = _normalized(prompts["behavior_system_instructions"])
    user = prompts["behavior_user_instructions"]
    assert prompts["public_seed"] == 0
    assert "outer harness owns multi-attempt exploration" in system
    assert "form one fresh attempt" in system
    assert "no configured aggregate attempt or resource limit" in system
    assert "CONTINUATION REQUIREMENT" in system
    assert "another concrete tool call" in user
    assert "explore-job-abc" in system
    assert "current attempt: `3`" in system
    assert "Attempt 1: grasp lost" in system
    assert "another attempt" in user
    for legacy in ("finish", "reset"):
        assert legacy not in user.lower()


def test_eval_is_one_fresh_attempt_without_retry_or_memory_update():
    prompts = _instructions("eval")
    system = _normalized(prompts["behavior_system_instructions"])
    user = prompts["behavior_user_instructions"]
    assert "one fresh evaluation attempt" in system
    assert "does not create another attempt" in system
    assert "Do not update frozen memory" in system
    assert "Use current evidence and state-conditioned knowledge." in system
    assert '"public_source_seed": 0' in system
    assert "never updates frozen task memory" in user
    for legacy in ("finish", "reset"):
        assert legacy not in user.lower()


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_continuation_requirement_has_exact_stop_conditions(phase):
    prompts = _instructions(phase)
    system = _normalized(prompts["behavior_system_instructions"])
    user = _normalized(prompts["behavior_user_instructions"])

    assert "honest natural-language final response" not in system
    assert "honest natural-language final response" not in user
    assert "unsuccessful or exhausted" not in system
    assert "five stop conditions below has been established" in system
    assert "do not produce a standalone final answer" in system
    assert "failed, rejected, unavailable, unreachable, or inconclusive" in system
    assert "Do not save a visual checkpoint merely as a reason to stop" in system
    assert 'raw `info["done"]["success"] == true`' in system
    assert "hard execution budget is exhausted" in system
    assert "operator explicitly requests stop" in system
    assert "unrecoverable infrastructure termination" in system
    assert "Do not infer condition 4 from an ordinary tool" in system
    assert "only when the runtime explicitly declares termination" in system
    assert "fresh, current visual observation" in system
    assert (
        "task-specific terminal visual failure described by the selected target prior"
        in system
    )
    assert "the current attempt is marked failed" in system
    assert "official `task_success` remains false" in system
    assert "object-control result alone does not establish condition 5" in system
    assert "condition `radio_tipped_flat`" in system
    assert "`knocked_over_by_robot_hand`" in system
    assert "`dropped_out_of_gripper`" in system
    assert "`head`, `left_wrist`, `right_wrist`" in system
    assert "fresh frame ID required by its current public schema" in system
    assert "Do not call any other tool afterward." in system
    assert "returns `_finish=true` with `task_success=false`" in system
    assert "full task-specific failure and one allowed cause" in system
    assert "Otherwise, continue using the current episode" in system
    assert "five listed stop conditions" in user


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_pi0_held_state_requires_fresh_visual_review(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert "one or more current hand attachments" in system
    assert "current_object_visual_check" in system
    assert "call `observe` once without `frame_review`" in system
    assert "not an unconditional reason to reject" in system


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_every_analytic_tool_requires_literal_fresh_head_bound_hand_selection(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert (
        "The five analytic manipulation primitives `move_to`, `rotate_wrist`, "
        "`close`, `open`, and `press`"
    ) in system
    assert "exactly one literal `hand`, either `left` or `right`" in system
    assert "Every call to any of these five primitives" in system
    assert "Either or both hands may have attachments" in system
    assert "must not make an otherwise valid literal-side call ambiguous" in system
    assert 'public `observe(camera="head")` frame' in system
    for field in (
        '"camera": "head"',
        '"frame_id": "<fresh head frame_id>"',
        '"selected_hand": "<left or right>"',
        '"assessment": "selected_hand_visually_confirmed"',
    ):
        assert field in system
    assert "must exactly equal the requested `hand`" in system
    assert "separate per-hand runtime fact" in system
    assert "does not resolve, rename, or override the requested hand" in system
    assert "never left/right image columns" in system
    assert "only by reviewing the cited head RGB" in system


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_visual_hand_evidence_lifetime_and_whole_body_contract_are_explicit(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert (
        "latest synchronized public head capture from the current run, attempt, "
        "and environment step"
    ) in system
    assert ("same evidence may be reused within that environment step") in system
    assert (
        "`pixel_to_world` and a rejected or read-only precondition check do not "
        "invalidate it"
    ) in system
    assert "must be rejected before any controller switch" in system
    assert "Any admitted environment action advances the step" in system
    assert "requires another fresh head observation" in system
    for result_field in (
        "`requested_hand`",
        "`resolved_hand`",
        '`motion_scope="whole_body"`',
    ):
        assert result_field in system
    assert "21 active DOFs" in system
    assert "base, trunk, and either arm may change only as members" in system
    assert "`open` and `close` remain selected-gripper-only" in system
    assert "recoverable fail-closed control failure" in system


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_pi0_is_excluded_from_analytic_hand_selection_contract(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert "hand-selection contract does not apply to `pi0_nav_pick`" in system
    assert "Do not pass `hand`, `role`, or `visual_hand_check` to it" in system
    assert "Pass its required `instruction` and `chunks` fields instead" in system
    assert "VLA still controls both arms through the unchanged 23D action schema" in (
        system
    )
    assert "completed Pi0 chunk executes all 32 `[32,23]` actions" in system


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_checkpoint_is_visual_only_and_never_a_gate(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert "read-only visual anchor" in system
    assert "does not save, restore, compare, or validate joint state" in system
    assert "never authorizes motion" in system
    assert "restore_robot_state_checkpoint" not in system


def test_libero_style_memory_names_and_deinstantiated_recipe_rules():
    explore = _normalized(_instructions("explore")["behavior_system_instructions"])
    evaluation = _normalized(_instructions("eval")["behavior_system_instructions"])
    for marker in ("ROBUST_RECIPE", "TASK_MEMORY", "REF_SOLVED_DIR", "SOURCE_TAG"):
        assert marker in explore
        assert marker in evaluation
    for forbidden in (
        "primitive names or order",
        "model instructions",
        "invocation counts",
        "per-invocation `chunks` values",
        "camera schedules",
        "fixed physical sides",
    ):
        assert forbidden in explore
        assert forbidden in evaluation


def test_eval_ref_solved_dir_is_verified_identity_not_fallback_or_search():
    evaluation = _normalized(_instructions("eval")["behavior_system_instructions"])
    assert (
        "only the logical identifier of the runtime-verified successful Explore "
        "Job root"
    ) in evaluation
    assert "already injected canonical top-level artifacts" in evaluation
    assert "cannot be searched" in evaluation
    assert "cannot discover, replace, or substitute" in evaluation
    assert "fallback" not in evaluation.lower()


def test_explore_history_is_bounded_and_eval_rejects_explore_context():
    with pytest.raises(ValueError, match="16000"):
        _instructions("explore", prior_attempt_summaries="x" * 16_001)
    with pytest.raises(ValueError, match="must be a string"):
        _instructions("explore", prior_attempt_summaries=["not", "the wire shape"])
    with pytest.raises(ValueError, match="rejects Explore"):
        _instructions("eval", job_id="job-not-allowed")


def test_explore_rejects_eval_memory_and_eval_requires_frozen_memory():
    with pytest.raises(ValueError, match="rejects frozen Eval memory"):
        _instructions("explore", robust_recipe="not allowed")
    with pytest.raises(ValueError, match="Eval requires non-empty"):
        _instructions("eval", task_memory=None)


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_task_scoped_knowledge_is_injected_with_explicit_precedence(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert "TASK TARGET PRIOR" in system
    assert "TARGET_PRIOR_SENTINEL" in system
    assert "REVIEWED EXPLORE EXPERIENCE" in system
    assert "REVIEWED_EXPLORE_EXPERIENCE_SENTINEL" in system
    assert "ADDITIONAL TASK EXPERT KNOWLEDGE" in system
    assert "ADDITIONAL_EXPERT_KNOWLEDGE_SENTINEL" in system
    assert '"source": "reviewed-repository"' in system
    assert (
        "runtime guards and current public evidence take precedence over frozen "
        "Task Memory"
    ) in system
    assert "frozen Task Memory" in system
    assert (
        "selected reviewed Recipe priors take precedence over reviewed Explore "
        "experience"
    ) in system
    assert "target prior defines task identity but remains subordinate" in system
    assert "No file-reading capability is exposed or needed" in system
    for unavailable_tool in ("read_text_file", "list_dir", "write_text_file"):
        assert unavailable_tool not in system


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_reviewed_recipe_priors_precede_reviewed_explore_experience(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert "REVIEWED RECIPE CATALOG PRIORS" in system
    assert "Reviewed Recipe guidance stays semantic" in system
    assert '"selected_ids": ["semantic-control-v1"]' in system
    assert "anonymous, reviewed task-level Recipe priors" in system
    assert "not a recorded execution" in system
    assert (
        "frozen Task Memory takes precedence over selected reviewed Recipe priors"
        in system
    )
    assert (
        "selected reviewed Recipe priors take precedence over reviewed Explore "
        "experience"
    ) in system


@pytest.mark.parametrize(
    "field",
    (
        "target_prior",
        "reviewed_explore_experience",
        "additional_expert_knowledge",
        "reviewed_repo_memory_manifest",
        "reviewed_recipe_priors",
        "reviewed_recipe_selection_manifest",
    ),
)
def test_reviewed_memory_and_recipe_inputs_are_required_and_nonempty(field):
    parameter = inspect.signature(build_prompt_context).parameters[field]
    assert parameter.default is inspect.Parameter.empty
    for invalid in (None, "", "   "):
        with pytest.raises(ValueError, match=field):
            _instructions("explore", **{field: invalid})
        with pytest.raises(ValueError, match=field):
            _instructions("eval", **{field: invalid})


@pytest.mark.parametrize(
    ("phase", "seed", "tag", "message"),
    (
        ("explore", 1, "turning_on_radio_s1", "public seed 0"),
        ("eval", 0, "turning_on_radio_s0", "public seeds 1 through 9"),
        ("eval", 10, "turning_on_radio_s10", "public seeds 1 through 9"),
        ("eval", 1, "wrong", "public tag"),
    ),
)
def test_public_split_and_tag_are_fail_closed(phase, seed, tag, message):
    with pytest.raises(ValueError, match=message):
        _instructions(phase, public_seed=seed, recipe_tag=tag)


def test_trash_eval_accepts_any_consistent_explore_source_identity() -> None:
    prompts = _trash_instructions(
        "eval",
        source_public_seed=1,
        source_recipe_tag="picking_up_trash_s1",
    )

    assert prompts["public_seed"] == 10
    assert prompts["recipe_tag"] == "picking_up_trash_s10"
    assert prompts["source_public_seed"] == 1
    assert prompts["source_recipe_tag"] == "picking_up_trash_s1"
    assert "picking_up_trash_s1" in prompts["behavior_system_instructions"]


@pytest.mark.parametrize(
    ("source_public_seed", "source_recipe_tag", "message"),
    (
        (10, "picking_up_trash_s10", "Explore partition"),
        (1, "picking_up_trash_s0", "recipe tag"),
        (1, "turning_on_radio_s1", "recipe tag"),
    ),
)
def test_trash_eval_rejects_inconsistent_or_non_explore_source_identity(
    source_public_seed,
    source_recipe_tag,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        _trash_instructions(
            "eval",
            source_public_seed=source_public_seed,
            source_recipe_tag=source_recipe_tag,
        )


def test_unknown_phase_task_invalid_budget_and_attempt_are_rejected():
    with pytest.raises(ValueError, match="unsupported BEHAVIOR phase"):
        _instructions("invalid")
    with pytest.raises(ValueError, match="unsupported BEHAVIOR task"):
        _instructions(task_name="another_task")
    with pytest.raises(ValueError, match="global_tool_budget"):
        _instructions(global_tool_budget=0)
    with pytest.raises(ValueError, match="attempt_index"):
        _instructions("explore", attempt_index=0)
    with pytest.raises(ValueError, match="job_id"):
        _instructions("explore", job_id="")
    with pytest.raises(ValueError, match="TaskSpec"):
        _instructions(task_language="A mismatched task instruction.")


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_trash_prompt_is_task_local_and_has_no_radio_policy_leakage(phase):
    prompts = _trash_instructions(phase)
    system = _normalized(prompts["behavior_system_instructions"])
    user = _normalized(prompts["behavior_user_instructions"])
    lowered = system.lower()

    assert TRASH_TASK_LANGUAGE in system
    assert "picking_up_trash_s" in system
    assert "TRASH_TARGET_PRIOR_SENTINEL" in system
    assert "TRASH_EXPLORE_EXPERIENCE_SENTINEL" in system
    assert "No reviewed Recipe entries are selected for this task." in system
    assert 'selected_entry_ids": []' in system
    assert "four stop conditions below has been established" in system
    assert "four listed stop conditions" in user
    assert "No task-specific visual terminal-failure condition is registered" in system
    assert "Do not submit a `terminal_failure` declaration" in system
    assert "No task-specific terminal use of this compatibility tool" in system
    assert "raw official success" in lowered
    assert "CONTINUATION REQUIREMENT" in system

    for radio_only in (
        "turning_on_radio",
        "radio",
        "button",
        "control-face",
        "radio_tipped_flat",
        "target_bearing_surface_confirmed",
        "opposite_surface_confirmed",
        "side_or_indeterminate",
        "condition 5",
    ):
        assert radio_only not in lowered


def test_task_prompt_registry_uses_exact_task_names_and_rejects_unknown_profiles():
    assert tuple(TASK_PROMPT_PROFILES) == (
        "turning_on_radio",
        "picking_up_trash",
    )
    assert (
        get_task_prompt_profile("turning_on_radio")
        is TASK_PROMPT_PROFILES["turning_on_radio"]
    )
    assert (
        get_task_prompt_profile("picking_up_trash")
        is TASK_PROMPT_PROFILES["picking_up_trash"]
    )

    with pytest.raises(ValueError, match="unsupported BEHAVIOR prompt profile"):
        get_task_prompt_profile("picking_up_trash_v1")


def test_prompt_builder_fails_closed_for_unknown_task_profile(monkeypatch):
    class InvalidPromptProfileSpec:
        prompt_profile_id = "picking_up_trash_v1"

        def __getattr__(self, name):
            return getattr(PICKING_UP_TRASH_TASK_SPEC, name)

    invalid_spec = InvalidPromptProfileSpec()
    monkeypatch.setattr(prompt_bundle, "get_task_spec", lambda _task_name: invalid_spec)

    with pytest.raises(ValueError, match="unsupported BEHAVIOR prompt profile"):
        _trash_instructions("explore")


@pytest.mark.parametrize(
    "factory",
    (_instructions, _trash_instructions),
    ids=("turning_on_radio", "picking_up_trash"),
)
def test_task_profile_is_identical_and_injected_once_across_explore_and_eval(factory):
    explore = factory("explore")
    evaluation = factory("eval")

    assert explore["prompt_profile_id"] == evaluation["prompt_profile_id"]
    assert explore["behavior_task_prompt"] == evaluation["behavior_task_prompt"]
    for prompts in (explore, evaluation):
        task_guidance = prompts["behavior_task_prompt"]
        system = prompts["behavior_system_instructions"]
        assert "TASK-SPECIFIC EXECUTION GUIDANCE" in system
        assert system.count(task_guidance) == 1


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_trash_profile_requires_lift_center_confirm_then_open(phase):
    guidance = _normalized(_trash_instructions(phase)["behavior_task_prompt"])

    ordered_markers = (
        "Lift before transfer",
        "do not move it directly or diagonally toward the trash can",
        '"delta_xyz": [0, 0, <positive_z>]',
        '"frame": "world"',
        'After the lift action, obtain a fresh `observe(camera="head")`',
        "complete hand and held can clear nearby furniture",
        "center of the receptacle opening",
        "`pixel_to_world` on that center",
        "safe standoff",
        "Approach from above",
        "Confirm before release",
        "obtain another fresh",
        '`open(hand="<left or right>")` for the same',
        "same hand still holds the same can",
        "clear downward path into the receptacle",
    )
    assert guidance.index("Keep the selected can hand closed") < guidance.index(
        "Lift before transfer"
    )
    release_flow = guidance[guidance.index("Lift before transfer") :]
    positions = [release_flow.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    for safety_marker in (
        "For the narrow act of releasing an attached can",
        "mandatory task-specific safety preconditions",
        "do not establish general tool priority",
        "never hard-code a side",
        "Pass that physical side as `hand`",
        "Either or both hands may have attachments",
        "`target_point_camera_xyz_m`",
        "`target_to_palm_m`",
        "`target_to_grip_point_m`",
        "`target_to_finger_roots_m`",
        "A head probe intentionally has no hand-relative distances",
        "do not verify object identity, clearance, contact, or a safe gripper command",
        "other arm, its gripper, and anything it holds remain stationary",
        "receptacle rim",
        "the other hand",
        "another small upward move",
        "do not sweep the hand or can",
        "do not open",
        "`release_visual_check`",
        '"assessment": "attached_object_fully_inside_receptacle_opening"',
        "exactly the same frame and physical hand",
        "hand-distance outputs remain guidance only",
        "applies only when `open` would release an attachment",
        "opening an empty hand does not require",
    ):
        assert safety_marker in guidance


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_trash_adaptive_two_can_search_then_scripted_navigation_order(phase):
    prompts = _trash_instructions(phase)
    system = _normalized(prompts["behavior_system_instructions"])
    phase_marker = "Explore job ID" if phase == "explore" else "evaluation cell"
    assert phase_marker in system
    guidance = _normalized(prompts["behavior_task_prompt"])
    lowered = guidance.lower()

    assert "chunks=20" not in system
    assert "chunks=20" not in guidance

    adaptive_markers = (
        "for every trash `pi0_nav_pick` invocation",
        "choose `chunks=n` explicitly and adaptively",
        "smallest positive number of complete 32-action chunks",
        "episode action steps",
        "complete-chunk budget",
        "never choose `n` merely from attachment count",
        "adapt `n` only between invocations",
        "do not automatically repeat or increase the previous `n`",
    )
    condition_markers = (
        "fresh runtime attachment facts",
        "`left` and `right` each currently holds an attachment",
        "required task soda can",
        "same fresh head frame has not yet identified",
    )
    followup_markers = (
        'immediately call `observe(camera="head")`',
        "consecutive vla calls are forbidden in this state",
        "body, rim, or opening to identify it",
        "stop blind pi0 search or navigation toward it",
        "need not be perfectly centered or completely visible",
        "project that current point with `pixel_to_world`",
        "pass the fresh receipt to `navigate_to`",
        "default `timeout_s=300`",
        "never force `timeout_s=20`",
        "after navigation returns, immediately obtain another fresh "
        '`observe(camera="head")`',
    )
    condition_positions = [lowered.index(marker) for marker in condition_markers]
    adaptive_positions = [lowered.index(marker) for marker in adaptive_markers]
    followup_positions = [lowered.index(marker) for marker in followup_markers]
    preserve = lowered.index("preserve both attachments")

    assert condition_positions == sorted(condition_positions)
    assert adaptive_positions == sorted(adaptive_positions)
    assert followup_positions == sorted(followup_positions)
    assert adaptive_positions[-1] < condition_positions[0]
    assert condition_positions[-1] < preserve < followup_positions[0]

    loaded_or_uncertain = lowered.index("loaded or may contain a previously placed can")
    pi0_forbidden = lowered.index(
        "do not use pi0 for any search, navigation, or local correction"
    )
    base_fixed = lowered.index("keep the base fixed as well")
    assert loaded_or_uncertain < pi0_forbidden < base_fixed
    assert "do not call `navigate_to` until" in lowered
    assert (
        "keep task motion frozen rather than risk spilling completed placements"
        in lowered
    )
    assert "use only fresh non-scene-changing observations" in lowered
    assert (
        "until a safe local analytic action or a trusted stop condition is available"
        in lowered
    )
    assert "do not declare completion or impossibility on your own" in lowered


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_trash_strict_opening_containment_is_only_a_pre_open_release_gate(phase):
    guidance = _normalized(_trash_instructions(phase)["behavior_task_prompt"])
    marker = (
        "strict opening-containment check belongs only to that fresh post-transfer "
        "release gate immediately before `open`"
    )
    assert marker in guidance
    assert (
        "not a prerequisite for identifying the receptacle, projecting a navigation "
        "target, calling `navigate_to`, lifting, or transferring a held can"
    ) in guidance
    assert guidance.index(marker) < guidance.index("Confirm before release")


@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_radio_prompt_has_no_trash_profile_leakage(phase):
    prompts = _instructions(phase)
    system = _normalized(prompts["behavior_system_instructions"])
    guidance = _normalized(prompts["behavior_task_prompt"])

    assert "Prompt profile: `turning_on_radio`" in guidance
    for trash_only in (
        "Prompt profile: `picking_up_trash`",
        "Lift before transfer",
        "Confirm before release",
        "required task soda can",
        "loaded or may contain a previously placed can",
        "keep task motion frozen rather than risk spilling completed placements",
    ):
        assert trash_only not in guidance
        assert trash_only not in system


@pytest.mark.parametrize(
    "factory",
    (_instructions, _trash_instructions),
    ids=("turning_on_radio", "picking_up_trash"),
)
@pytest.mark.parametrize("phase", ("explore", "eval"))
def test_task_profiles_preserve_common_pi0_surface_contract(factory, phase):
    prompts = factory(phase)
    system = _normalized(prompts["behavior_system_instructions"])
    guidance = _normalized(prompts["behavior_task_prompt"])

    for marker in (
        "must supply both an instruction grounded in the current subgoal",
        "`chunks=N`",
        "`chunks` is a required positive integer (`N >= 1`)",
        "has no fixed maximum",
        "is not a Pi0-specific usage quota",
        "hand-selection contract does not apply to `pi0_nav_pick`",
        "Do not pass `hand`, `role`, or `visual_hand_check` to it",
        "Pass its required `instruction` and `chunks` fields instead",
        "VLA still controls both arms through the unchanged 23D action schema",
    ):
        assert marker in system
    for obsolete_contract in (
        "LLM supplies only an instruction",
        "runs until a non-quota runtime boundary returns control",
    ):
        assert obsolete_contract not in system
    for obsolete_limit in (
        "max_chunks",
        "max_vla_chunks_per_call",
        "max_total_vla_chunks",
        "call_chunk_limit",
    ):
        assert obsolete_limit not in system
        assert obsolete_limit not in guidance
    if prompts["prompt_profile_id"] == "turning_on_radio":
        assert "chunks" not in guidance.lower()
        assert "`pi0_nav_pick`" not in guidance
    else:
        assert "`pi0_nav_pick`" in guidance
        assert "no fixed" in guidance
        assert "`chunks` value" in guidance
        assert "chunks=20" not in guidance
        for fixed_chunk_directive in (
            "`chunks=1`",
            "`chunks=2`",
            "`chunks=3`",
            "choose 1 chunk",
            "choose 2 chunks",
            "choose 3 chunks",
        ):
            assert fixed_chunk_directive not in guidance.lower()


@pytest.mark.parametrize("phase", ["explore", "eval"])
def test_radio_surface_cycle_uses_literal_selected_hand_attachment_receipts(phase):
    system = _normalized(_instructions(phase)["behavior_system_instructions"])
    assert 'literal `hand="left"` or `hand="right"`' in system
    assert "matching requested and resolved physical hands" in system
    assert "accepted fresh head-frame visual-hand evidence for that hand" in system
    assert "stable call-start attachment-fingerprint receipt" in system
    assert "The other hand may independently have its own attachment" in system
    assert "executes exactly the LLM-requested N complete 32-action chunks" in system
    assert "and returns without raw official success" in system
    assert (
        "missing or unstable selected-hand call-start attachment fingerprint" in system
    )


def test_rendering_treats_braces_inside_inputs_as_data():
    evaluation = _instructions(
        "eval", task_memory='Observed data: {"literal": "{{ not_a_placeholder }}"}'
    )["behavior_system_instructions"]
    explore = _instructions(
        "explore",
        prior_attempt_summaries="Literal {{ prior text }}",
        target_prior="Reviewed literal {{ target prior text }}",
        reviewed_explore_experience="Reviewed literal {{ experience text }}",
        additional_expert_knowledge="Reviewed literal {{ expert text }}",
    )["behavior_system_instructions"]
    assert "{{ not_a_placeholder }}" in evaluation
    assert "{{ prior text }}" in explore
    assert "{{ target prior text }}" in explore
    assert "{{ experience text }}" in explore
    assert "{{ expert text }}" in explore


def test_prompt_bundle_uses_python_prompt_nodes_and_one_explicit_robot_guide_loader():
    assert not hasattr(prompt_bundle, "_load_prompt")
    assert not hasattr(prompt_bundle, "_render_prompt")
    assert callable(prompt_bundle._load_r1pro_robot_prior)
    assert prompt_bundle._R1PRO_ROBOT_PRIOR_PATH == R1PRO_ROBOT_PRIOR_PATH
    assert isinstance(system_prompt(), dict)
    assert isinstance(user_prompt(), dict)


def test_user_prompt_exposes_only_public_identity():
    template = repr(user_prompt())
    assert "{{ recipe_tag }}" in template
    assert "{{ public_seed }}" in template
    assert "activity_instance_id" not in template
    assert "{{ seed }}" not in template
