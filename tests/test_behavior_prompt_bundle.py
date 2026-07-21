from pathlib import Path

import pytest

from robots.behavior.prompt_bundle import mode_instructions
from robots.behavior.schemas import (
    FULL_TASK_VLA_MODE,
    HYBRID_VLM_PI0_MODE,
    PI0_NAV_PICK_VLA_MODE,
    PI0_PICK_VLA_MODE,
    PLANNER_TOOLS_MODE,
)

PROMPT_DIR = Path(__file__).parents[1] / "robots" / "behavior" / "prompts"


def test_behavior_mode_prompts_are_checked_in_as_markdown():
    expected = {
        "full_task_system.md",
        "full_task_user.md",
        "hybrid_system.md",
        "hybrid_user.md",
        "pi0_nav_pick_system.md",
        "pi0_nav_pick_prepress_resume_system.md",
        "pi0_nav_pick_press_resume_system.md",
        "pi0_nav_pick_press_resume_user.md",
        "pi0_nav_pick_user.md",
        "pi0_pick_system.md",
        "pi0_pick_user.md",
        "planner_system.md",
        "planner_user.md",
        "turning_on_radio_button_visual_prior.md",
    }

    assert {path.name for path in PROMPT_DIR.glob("*.md")} == expected


@pytest.mark.parametrize(
    "mode",
    [PLANNER_TOOLS_MODE, HYBRID_VLM_PI0_MODE, PI0_NAV_PICK_VLA_MODE],
)
def test_turning_on_radio_prompt_enforces_two_stage_button_gate(mode):
    system = mode_instructions(
        mode,
        task_name="turning_on_radio",
    )["behavior_system_instructions"]
    normalized = " ".join(system.split())

    assert "Face classification" in system
    assert "black round or oval disk" in system
    assert "white outer ring" in system
    assert "raised red center" in normalized
    assert "`clear_slotted_back_face`" in system
    assert "`side_port`" in system
    assert "`ambiguous`" in system
    assert "red broad face contains a clearly elongated" in normalized
    assert "even if the white side port is also visible" in normalized
    assert "white side face contains a black oval port" in normalized
    assert "button_visible=false" in system
    assert "NOT_VISIBLE" in system
    assert "must not produce coordinates" in normalized
    assert "Press staging is projection-driven" in system
    assert "never" in system and "hard-code" in system
    assert "Do not press" in system or "do not press" in system
    assert "pause physics" in system or "Leave physics paused" in system
    assert "antiparallel" in system
    assert "opposition error" in system
    assert "at most 0.010 m" in system
    assert "at most 15 degrees" in normalized
    assert "0.03--0.06 m" in system
    assert "clear slotted back face may directly authorize" in normalized
    assert "side-port or ambiguous views" in normalized
    assert "Never ask the LLM to provide an exact held EEF xyz/quaternion" in system
    assert "multiple radio and held-EEF pose candidates" in normalized
    assert "sends eligible candidates to CuRobo" in normalized
    assert "final button localization must use a fresh dynamic" in normalized
    assert "press_wrist" in system
    assert "raised red center" in normalized
    assert "0.08 * min(image_width, image_height)" in normalized
    assert "0.10 * min(image_width, image_height)" in normalized
    assert "normalized current-frame constraints" in normalized
    assert "must not be artificially divided into small steps" in normalized
    assert "literal EEF target is not part of the public tool contract" in normalized


@pytest.mark.parametrize(
    "mode",
    [
        FULL_TASK_VLA_MODE,
        PI0_PICK_VLA_MODE,
        PLANNER_TOOLS_MODE,
        HYBRID_VLM_PI0_MODE,
    ],
)
def test_button_workflow_is_not_injected_into_other_tasks_or_closed_vla_modes(mode):
    system = mode_instructions(
        mode,
        task_name="other_task",
    )["behavior_system_instructions"]

    assert "Face classification" not in system


def test_local_pick_and_full_task_prompts_do_not_gain_post_pick_button_workflow():
    for mode in (FULL_TASK_VLA_MODE, PI0_PICK_VLA_MODE):
        system = mode_instructions(mode)["behavior_system_instructions"]
        assert "Face classification" not in system


def test_nav_pick_turning_on_radio_prompt_has_strict_prepress_phase():
    prompts = mode_instructions(
        PI0_NAV_PICK_VLA_MODE,
        task_name="turning_on_radio",
        pi0_instruction="Turn on the radio receiver.",
    )
    system = prompts["behavior_system_instructions"]
    user = prompts["behavior_user_instructions"]

    assert "Post-pick pre-press phase" in system
    assert 'inspect_post_pick_state(checkpoint_name="state_checkpoint_1")' in system
    assert '"role": "press"' in system
    assert '"kind": "press_staging"' in system
    assert "non-contact standoff" in system
    assert "knocked flat onto the table" in system
    assert "minimum table clearance" in system
    assert "XY, Z" in system
    assert "Do not press the button" in system
    assert "state_checkpoint_2" in system
    assert "deliberately center" in system
    assert "historical pixels" in system
    assert "true raised red center inside the required image-center region" in (
        " ".join(system.split())
    )
    assert "`pixel_to_world`" in system
    assert "rotate_wrist" in system
    assert "save_robot_state_checkpoint" in system
    assert "`project_button`" not in system
    assert "`save_prepress_checkpoint`" not in system
    assert "clear_slotted_back_face" in system
    assert "opposite broad face" in system
    assert "head_view=\"side\"" in system
    assert "perpendicular to the head optical axis" in system
    assert "runtime owns the current held-to-radio grasp transform" in " ".join(
        system.split()
    )
    assert "Do not calculate or submit an EEF pose" in system
    assert "Call `pi0_nav_pick` exactly once" in user
    assert "continue with `inspect_post_pick_state`" in user


def test_nav_pick_turning_on_radio_prompt_has_explicit_stage3_press_phase():
    system = (PROMPT_DIR / "pi0_nav_pick_press_resume_system.md").read_text(
        encoding="utf-8"
    )
    user = (PROMPT_DIR / "pi0_nav_pick_press_resume_user.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(system.split())

    assert "Post-prepress button-press phase" in system
    assert "state_checkpoint_2.json" in system
    assert "post_pick_close_press_gripper" in system
    assert "fully closed" in system
    assert "post_pick_recenter_held_button" in system
    assert "runtime-reported number of consecutive physics updates" in normalized
    assert "value was five in the successful run" in normalized
    assert "extra axial depth cannot repair a visible lateral miss" in normalized
    assert "post_success_hold_frames(frames=4)" in normalized
    assert 'info["done"]["success"]' in system
    assert "visibly green" in system
    assert "immediately pause environment stepping" in normalized
    assert "Do not retreat the press hand" in normalized
    assert "Do not call navigation or grasp again" in user
    assert "Call `BehaviorPrimitives.pi0_nav_pick`" not in system
    assert "Do not press the button in stage 2" not in system
    assert "save state_checkpoint_2" not in normalized


def test_nav_pick_stage3_press_prompt_is_not_injected_by_default():
    default_system = mode_instructions(
        PI0_NAV_PICK_VLA_MODE,
        task_name="turning_on_radio",
    )["behavior_system_instructions"]
    assert "Post-pick pre-press phase" in default_system
    assert "Post-prepress button-press phase" not in default_system


def test_markdown_user_prompt_templates_render_runtime_values():
    prompts = mode_instructions(
        HYBRID_VLM_PI0_MODE,
        pi0_hand="left",
        pi0_instruction="grasp only the radio handle",
        pi0_max_chunks=24,
    )
    user = prompts["behavior_user_instructions"]

    assert 'hand="left"' in user
    assert 'instruction="grasp only the radio handle"' in user
    assert "max_chunks=24" in user
    assert "{{" not in user
