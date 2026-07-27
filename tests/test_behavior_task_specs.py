from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from robots.behavior.task_specs import (
    PICKING_UP_TRASH_TASK_SPEC,
    TURNING_ON_RADIO_TASK_SPEC,
    get_task_spec,
    get_task_spec_by_index,
    resolve_task_spec,
)


def test_specs_resolve_by_name_and_index_and_reject_cross_task_pair() -> None:
    assert get_task_spec("turning_on_radio") is TURNING_ON_RADIO_TASK_SPEC
    assert get_task_spec_by_index(0) is TURNING_ON_RADIO_TASK_SPEC
    assert get_task_spec("picking_up_trash") is PICKING_UP_TRASH_TASK_SPEC
    assert get_task_spec_by_index(1) is PICKING_UP_TRASH_TASK_SPEC
    assert (
        resolve_task_spec(task_name="picking_up_trash", task_index=1)
        is PICKING_UP_TRASH_TASK_SPEC
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_task_spec(task_name="picking_up_trash", task_index=0)
    with pytest.raises(ValueError, match="unsupported BEHAVIOR task"):
        get_task_spec("unknown")
    with pytest.raises(ValueError, match="unsupported BEHAVIOR task index"):
        get_task_spec_by_index(99)


def test_turning_on_radio_preserves_existing_seed_and_task_policies() -> None:
    spec = TURNING_ON_RADIO_TASK_SPEC
    assert spec.task_index == 0
    assert spec.prompt_profile_id == "turning_on_radio"
    assert spec.activity_definition_id == 0
    assert spec.scene_model == "house_double_floor_lower"
    assert dict(spec.public_seed_to_instance) == {
        0: 242,
        1: 109,
        2: 181,
        3: 187,
        4: 197,
        5: 203,
        6: 211,
        7: 212,
        8: 295,
        9: 298,
    }
    assert spec.instance_for_public_seed(0, phase="explore") == 242
    assert spec.instance_for_public_seed(9, phase="eval") == 298
    assert spec.tag(0) == "turning_on_radio_s0"
    assert spec.terminal_failure_policy is not None
    assert spec.terminal_failure_policy.condition == "radio_tipped_flat"
    assert spec.terminal_failure_policy.cameras == (
        "head",
        "left_wrist",
        "right_wrist",
    )
    assert spec.surface_review_policy is not None
    assert spec.surface_review_policy.opposite_cycles_before_pi0_disable == 2
    assert spec.release_visual_policy is None


def test_picking_up_trash_uses_csv_order_and_authoritative_task_language() -> None:
    spec = PICKING_UP_TRASH_TASK_SPEC
    assert spec.task_index == 1
    assert spec.prompt_profile_id == "picking_up_trash"
    assert spec.activity_definition_id == 0
    assert spec.scene_model == "house_double_floor_lower"
    assert tuple(spec.public_seed_to_instance.values()) == (
        196,
        67,
        155,
        106,
        161,
        245,
        171,
        156,
        162,
        246,
        108,
        152,
        84,
        198,
        199,
        100,
        111,
        151,
        130,
        168,
    )
    assert spec.task_language == (
        "Put the three can of soda from the living room inside the tash can "
        "in the kitchen."
    )
    assert spec.explore_public_seeds == tuple(range(10))
    assert spec.eval_public_seeds == tuple(range(10, 20))
    assert spec.instance_for_public_seed(0, phase="explore") == 196
    assert spec.instance_for_public_seed(1, phase="explore") == 67
    assert spec.instance_for_public_seed(9, phase="explore") == 246
    assert spec.instance_for_public_seed(10, phase="eval") == 108
    assert spec.instance_for_public_seed(19, phase="eval") == 168
    assert spec.tag(0) == "picking_up_trash_s0"
    assert spec.state_dir_name == (
        "house_double_floor_lower_task_picking_up_trash_instances"
    )
    assert spec.terminal_failure_policy is None
    assert spec.surface_review_policy is None
    assert spec.release_visual_policy is not None
    assert spec.release_visual_policy.camera == "head"
    assert (
        spec.release_visual_policy.assessment
        == "attached_object_fully_inside_receptacle_opening"
    )


def test_seed_validation_is_task_and_phase_scoped() -> None:
    with pytest.raises(ValueError, match="does not allow s10 in explore"):
        PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(10, phase="explore")
    with pytest.raises(ValueError, match="does not allow s0 in eval"):
        PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(0, phase="eval")
    with pytest.raises(ValueError, match="does not allow s9 in eval"):
        PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(9, phase="eval")
    with pytest.raises(ValueError, match="no public seed s20"):
        PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(20)
    with pytest.raises(ValueError, match="unsupported BEHAVIOR phase"):
        PICKING_UP_TRASH_TASK_SPEC.instance_for_public_seed(0, phase="invalid")


def test_instance_classification_never_treats_242_globally() -> None:
    radio = TURNING_ON_RADIO_TASK_SPEC.classify_instance(242)
    trash = PICKING_UP_TRASH_TASK_SPEC.classify_instance(242)
    trash_public = PICKING_UP_TRASH_TASK_SPEC.classify_instance(196)

    assert (radio.kind, radio.public_seed) == ("explore", 0)
    assert (trash.kind, trash.public_seed) == ("candidate", None)
    assert (trash_public.kind, trash_public.public_seed) == ("explore", 0)


def test_registry_and_specs_are_immutable() -> None:
    with pytest.raises(TypeError):
        PICKING_UP_TRASH_TASK_SPEC.public_seed_to_instance[0] = 242
    with pytest.raises(FrozenInstanceError):
        PICKING_UP_TRASH_TASK_SPEC.task_name = "turning_on_radio"
